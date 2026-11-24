"""
Testing & Evaluation Patterns
Building reliable LLM applications
"""

import pytest
from unittest.mock import Mock, patch
from typing import Callable
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langsmith import traceable, Client
from dotenv import load_dotenv

load_dotenv()


# === Unit Testing with Mocks ===
class QAChain:
    """Simple Q&A chain for testing."""

    def __init__(self, llm=None):
        # NOTE (2026-11): gpt-4o-mini/gpt-4o shut down 2027-01-14 -> gpt-6-luna. reasoning_effort="none" lets it accept temperature + tools.
        self.llm = llm or ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0)
        self.prompt = ChatPromptTemplate.from_template(
            "Answer this question: {question}"
        )

    def ask(self, question: str) -> str:
        prompt_value = self.prompt.invoke({"question": question})
        response = self.llm.invoke(prompt_value)
        return response.content


def test_qa_chain_with_mock():
    """Test QA chain with mocked LLM."""

    # Create mock LLM
    mock_llm = Mock()
    mock_llm.invoke.return_value = AIMessage(content="Paris")

    # Test with mock
    chain = QAChain(llm=mock_llm)
    result = chain.ask("What is the capital of France?")

    assert result == "Paris"
    mock_llm.invoke.assert_called_once()


def test_qa_chain_handles_empty_response():
    """Test chain handles empty responses."""

    mock_llm = Mock()
    mock_llm.invoke.return_value = AIMessage(content="")

    chain = QAChain(llm=mock_llm)
    result = chain.ask("Empty question")

    assert result == ""


# === Integration Testing with Real LLM ===
class IntegrationTestSuite:
    """Integration tests with real LLM calls."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0)

    @traceable(name="integration_test")
    def test_basic_qa(self) -> dict:
        """Test basic question answering."""

        test_cases = [
            {
                "question": "What is 2 + 2?",
                "expected_contains": ["4", "four"],
            },
            {
                "question": "What color is the sky on a clear day?",
                "expected_contains": ["blue"],
            },
        ]

        results = []
        for case in test_cases:
            response = self.llm.invoke(case["question"])
            content = response.content.lower()

            passed = any(exp.lower() in content for exp in case["expected_contains"])

            # "The answer is 4" or "2 + 2 equals four" or "That would be 4."

            results.append(
                {
                    "question": case["question"],
                    "response": response.content,
                    "passed": passed,
                }
            )

        return {
            "total": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "results": results,
        }


def demo_integration_tests():
    """Run integration tests."""

    suite = IntegrationTestSuite()

    print("Integration Test Results:\n")

    results = suite.test_basic_qa()

    print(f"Passed: {results['passed']}/{results['total']}")

    for r in results["results"]:
        status = "✅" if r["passed"] else "❌"
        print(f"{status} {r['question']}")
        print(f"   Response: {r['response'][:50]}...")


# === Evaluation Framework ===
class LLMEvaluator:
    """Use LLM to evaluate LLM outputs."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0)

    @traceable(name="evaluate_response")
    def evaluate(self, question: str, response: str, reference: str = None) -> dict:
        """Evaluate a response on multiple dimensions."""

        eval_prompt = ChatPromptTemplate.from_template(
            """
Evaluate this response on a scale of 1-10 for each criterion.

Question: {question}
Response: {response}
{reference_section}

Rate each criterion (1-10):
1. Correctness: Is the information accurate?
2. Relevance: Does it answer the question?
3. Clarity: Is it easy to understand?
4. Completeness: Does it fully address the question?

Respond with ONLY a JSON object:
{{"correctness": X, "relevance": X, "clarity": X, "completeness": X, "overall": X}}
"""
        )

        reference_section = ""
        if reference:
            reference_section = f"Reference answer: {reference}"

        import json

        response_obj = self.llm.invoke(
            eval_prompt.format(
                question=question,
                response=response,
                reference_section=reference_section,
            )
        )

        try:
            scores = json.loads(response_obj.content)
            return scores
        except json.JSONDecodeError:
            return {"error": "Failed to parse evaluation"}


def demo_evaluation():
    """Demonstrate LLM evaluation."""

    evaluator = LLMEvaluator()

    # Test case
    question = "Explain what machine learning is in simple terms."
    response = "Machine learning is when computers learn from data instead of being explicitly programmed. It's like teaching a child by showing examples rather than giving them rules."
    reference = "Machine learning is a type of artificial intelligence where computers learn patterns from data to make predictions or decisions."

    print("LLM Evaluation Demo:\n")
    print(f"Question: {question}")
    print(f"Response: {response}")

    scores = evaluator.evaluate(question, response, reference)

    print("\nScores:")
    for metric, score in scores.items():
        print(f"  {metric}: {score}/10")


# === Regression Testing ===
class RegressionTestRunner:
    """Run regression tests against a test dataset."""

    def __init__(self, chain: Callable):
        self.chain = chain
        self.evaluator = LLMEvaluator()

    @traceable(name="regression_test")
    def run(self, test_cases: list[dict]) -> dict:
        """
        Run regression tests.

        test_cases: [{"input": ..., "expected": ...}, ...]
        """
        results = []
        total_score = 0

        for case in test_cases:
            # Get response from chain
            response = self.chain(case["input"])

            # Evaluate
            scores = self.evaluator.evaluate(
                question=case["input"],
                response=response,
                reference=case.get("expected"),
            )

            overall = scores.get("overall", 0)
            total_score += overall

            results.append(
                {
                    "input": case["input"],
                    "response": response,
                    "expected": case.get("expected"),
                    "scores": scores,
                    "passed": overall >= 7,  # Threshold
                }
            )

        return {
            "total": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "average_score": total_score / len(results) if results else 0,
            "results": results,
        }


def demo_regression_testing():
    """Demonstrate regression testing."""

    # Simple chain to test
    llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0)

    def qa_chain(question: str) -> str:
        return llm.invoke(question).content

    # Test cases
    test_cases = [
        {
            "input": "What is Python?",
            "expected": "Python is a programming language known for its simplicity.",
        },
        {"input": "What is 10 * 5?", "expected": "50"},
    ]

    runner = RegressionTestRunner(qa_chain)

    print("\nRegression Test Results:\n")

    results = runner.run(test_cases)

    print(f"Passed: {results['passed']}/{results['total']}")
    print(f"Average Score: {results['average_score']:.1f}/10")

    for r in results["results"]:
        status = "✅" if r["passed"] else "❌"
        print(f"\n{status} {r['input']}")
        print(f"   Response: {r['response'][:50]}...")
        print(f"   Overall Score: {r['scores'].get('overall', 'N/A')}/10")


"""
LangSmith Evaluation Datasets — Production Approach
Persistent, versioned test suites for LLM applications
"""

from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

client = Client()
llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0)


# ============================================================
# Step 1: Create an evaluation dataset
# ============================================================


def create_eval_dataset():
    """Create a dataset with test cases in LangSmith."""

    dataset_name = "qa-eval-dataset"

    # Delete if exists (for demo purposes — don't do this in production)
    existing = list(client.list_datasets(dataset_name=dataset_name))
    if existing:
        client.delete_dataset(dataset_id=existing[0].id)

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Q&A evaluation dataset for testing our chain",
    )

    # Add test examples — inputs and expected outputs
    examples = [
        {
            "inputs": {"question": "What is Python?"},
            "outputs": {
                "answer": "Python is a high-level programming language known for its readability and versatility."
            },
        },
        {"inputs": {"question": "What is 15 * 4?"}, "outputs": {"answer": "60"}},
        {
            "inputs": {"question": "What does HTML stand for?"},
            "outputs": {"answer": "HyperText Markup Language"},
        },
        {
            "inputs": {"question": "Name one benefit of exercise."},
            "outputs": {
                "answer": "Exercise improves cardiovascular health and reduces the risk of chronic diseases."
            },
        },
        {
            "inputs": {"question": "What is the capital of Japan?"},
            "outputs": {"answer": "Tokyo"},
        },
    ]

    for ex in examples:
        client.create_example(
            inputs=ex["inputs"], outputs=ex["outputs"], dataset_id=dataset.id
        )

    print(f"Created dataset '{dataset_name}' with {len(examples)} examples")
    return dataset_name


# ============================================================
# Step 2: Define the chain to evaluate
# ============================================================

prompt = ChatPromptTemplate.from_template("Answer this question concisely: {question}")
qa_chain = prompt | llm


@traceable(name="qa_target")
def qa_target(inputs: dict) -> dict:
    """
    Target function for LangSmith evaluation.
    Must accept a dict (inputs) and return a dict (outputs).
    """
    response = qa_chain.invoke({"question": inputs["question"]})
    return {"answer": response.content}


# ============================================================
# Step 3: Define evaluators
# ============================================================

# Evaluator: checks correctness against reference using LLM-as-judge
eval_llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0)


def correctness(run, example) -> dict:
    """LLM-as-judge evaluator for correctness against reference answer."""
    prediction = run.outputs.get("answer", "")
    reference = example.outputs.get("answer", "")
    question = example.inputs.get("question", "")

    grade_prompt = ChatPromptTemplate.from_template(
        "You are a grader. Given a question, a submission, and a reference answer, "
        "determine if the submission is correct, accurate, and factual compared to "
        "the reference answer.\n\n"
        "Question: {question}\n"
        "Submission: {submission}\n"
        "Reference: {reference}\n\n"
        "Respond with ONLY 'Y' if correct or 'N' if incorrect."
    )
    result = eval_llm.invoke(
        grade_prompt.format(
            question=question, submission=prediction, reference=reference
        )
    )
    score = 1.0 if result.content.strip().upper() == "Y" else 0.0
    return {"key": "correctness", "score": score}


def helpfulness(run, example) -> dict:
    """LLM-as-judge evaluator for helpfulness (no reference needed)."""
    prediction = run.outputs.get("answer", "")
    question = example.inputs.get("question", "")

    grade_prompt = ChatPromptTemplate.from_template(
        "You are a grader. Given a question and a response, "
        "determine if the response is helpful, clear, and easy to understand.\n\n"
        "Question: {question}\n"
        "Response: {response}\n\n"
        "Respond with ONLY 'Y' if helpful or 'N' if not helpful."
    )
    result = eval_llm.invoke(
        grade_prompt.format(question=question, response=prediction)
    )
    score = 1.0 if result.content.strip().upper() == "Y" else 0.0
    return {"key": "helpfulness", "score": score}


# Custom evaluator: simple keyword check
def contains_answer(run, example) -> dict:
    """
    Custom evaluator — checks if the response contains
    key terms from the expected answer.
    """
    prediction = run.outputs.get("answer", "").lower()
    reference = example.outputs.get("answer", "").lower()

    # Extract key words from reference (words > 3 chars)
    key_words = [word for word in reference.split() if len(word) > 3]

    # Check if at least 50% of key words appear in prediction
    if not key_words:
        return {"key": "contains_answer", "score": 1.0}

    matches = sum(1 for word in key_words if word in prediction)
    score = matches / len(key_words)

    return {"key": "contains_answer", "score": score}


# ============================================================
# Step 4: Run the evaluation
# ============================================================


def run_evaluation(dataset_name: str):
    """Run evaluation against the dataset."""

    print(f"\nRunning evaluation against '{dataset_name}'...\n")

    results = evaluate(
        qa_target,
        data=dataset_name,
        evaluators=[correctness, helpfulness, contains_answer],
        experiment_prefix="qa-chain-v1",  # Tags this run for comparison
        max_concurrency=2,
    )

    # Print summary
    print("\nEvaluation Results:")
    print("-" * 50)

    for result in results:
        question = result["run"].inputs.get("question", "N/A")
        answer = result["run"].outputs.get("answer", "N/A")

        print(f"\nQ: {question}")
        print(f"A: {answer[:80]}...")

        for eval_result in result["evaluation_results"]["results"]:
            print(f"  {eval_result.key}: {eval_result.score}")

    return results


# ============================================================
# Step 5: Compare experiments (after model or prompt change)
# ============================================================


def run_comparison(dataset_name: str):
    """
    Run a second experiment with a different config,
    then compare in LangSmith dashboard.
    """

    # New prompt — more detailed instructions
    detailed_prompt = ChatPromptTemplate.from_template(
        "Answer this question accurately and concisely. "
        "If it's a factual question, be precise. "
        "If it's a math question, show just the answer.\n\n"
        "Question: {question}"
    )
    v2_chain = detailed_prompt | llm

    @traceable(name="qa_target_v2")
    def qa_target_v2(inputs: dict) -> dict:
        response = v2_chain.invoke({"question": inputs["question"]})
        return {"answer": response.content}

    print("\nRunning v2 experiment for comparison...\n")

    results = evaluate(
        qa_target_v2,
        data=dataset_name,
        evaluators=[correctness, helpfulness, contains_answer],
        experiment_prefix="qa-chain-v2",  # Different prefix for comparison
        max_concurrency=2,
    )

    print("\nDone! Compare v1 vs v2 in LangSmith dashboard:")
    print("  → Go to your LangSmith project → Datasets → qa-eval-dataset")
    print("  → Click 'Compare Experiments' to see v1 vs v2 side by side")

    return results


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LangSmith Evaluation Datasets Demo")
    print("=" * 60)

    # Step 1: Create dataset
    dataset_name = create_eval_dataset()

    # Step 2: Run first evaluation (v1)
    print("\n" + "=" * 60)
    print("Experiment 1: Basic prompt (v1)")
    print("=" * 60)
    run_evaluation(dataset_name)

    # Step 3: Run second evaluation (v2) for comparison
    print("\n" + "=" * 60)
    print("Experiment 2: Detailed prompt (v2)")
    print("=" * 60)
    run_comparison(dataset_name)

    print("\n" + "=" * 60)
    print("All experiments logged to LangSmith!")
    print("=" * 60)


# if __name__ == "__main__":
# test_qa_chain_with_mock()
# print("All tests passed!")
# test_qa_chain_handles_empty_response()
# print("All tests passed!")
# demo_integration_tests()
# demo_evaluation()
# demo_regression_testing()
