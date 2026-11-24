from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
import operator
from dotenv import load_dotenv

load_dotenv()

"""
Cycles and Loops in LangGraph
Self-correcting agents and iterative refinement
"""

# NOTE (2026-11): gpt-4o-mini/gpt-4o shut down 2027-01-14 -> gpt-6-luna. reasoning_effort="none" lets it accept temperature + tools.
llm = init_chat_model("gpt-6-luna", reasoning_effort="none", temperature=0.0)


class CodeGenState(TypedDict):
    task: str
    code: str
    errors: Annotated[list[str], operator.add]
    iteration: int
    max_iterations: int
    success: bool


def demo_self_correcting_code():
    """Self-correcting code generator."""

    def generate_code(state: CodeGenState) -> dict:
        if state["iteration"] == 0:
            # First attempt
            prompt = f"Write Python code for: {state['task']}\nReturn only the code."
        else:
            # Correction attempt
            prompt = (
                f"Fix this Python code:\n{state['code']}\n\n"
                f"Errors:\n{state['errors'][-1]}\n\n"
                "Return only the corrected code."
            )

        response = llm.invoke(prompt)
        code = response.content.strip()

        # Clean up markdown code blocks if present
        if code.startswith("```"):
            code = code.split("```")[1]
            if code.startswith("python"):
                code = code[6:]

        return {"code": code, "iteration": state["iteration"] + 1}

    def validate_code(state: CodeGenState) -> dict:
        code = state["code"]

        # Step 1: Does it compile?
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            return {"errors": [f"SyntaxError: {e}"], "success": False}

        # Step 2: Does it RUN and produce correct results?
        test_cases = [
            ([3, 1, 4, 1, 5, 9], 5),  # normal case
            ([1, 1, 1], None),  # all same → no second largest
            ([7], None),  # single element
            ([3, -1, 3, 5, 5], 3),  # duplicates at top
        ]

        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            return {"errors": [f"Runtime error: {e}"], "success": False}

        if "solve" not in namespace:
            return {"errors": ["Function 'solve' not found in code"], "success": False}

        for inputs, expected in test_cases:
            try:
                result = namespace["solve"](inputs)
                if result != expected:
                    return {
                        "errors": [
                            f"solve({inputs}) returned {result}, expected {expected}"
                        ],
                        "success": False,
                    }
            except Exception as e:
                return {"errors": [f"solve({inputs}) raised {e}"], "success": False}

        return {"success": True}

    def should_continue(state: CodeGenState) -> Literal["generate", "end"]:
        if state["success"]:
            return "end"
        elif state["iteration"] >= state["max_iterations"]:
            return "end"
        else:
            return "generate"

    def finalize(state: CodeGenState) -> dict:
        return state

    graph = StateGraph(CodeGenState)

    graph.add_node("generate", generate_code)
    graph.add_node("validate", validate_code)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate", should_continue, {"generate": "generate", "end": "finalize"}
    )  # Loop back to "generate" if not successful and under max iterations, otherwise go to "finalize"
    graph.add_edge("finalize", END)

    app = graph.compile()

    # visualize the graph
    print("\n--- Mermaid Graph ---")
    # print(app.get_graph().draw_mermaid())

    # save as PNG
    png_bytes = app.get_graph().draw_mermaid_png()
    with open("graph_code.png", "wb") as f:
        f.write(png_bytes)
    print("\nGraph saved to graph_code.png")

    print("Self-Correcting Code Generator:\n")

    result = app.invoke(
        {
            "task": "a function that calculates factorial recursively",
            "code": "",
            "errors": [],
            "iteration": 0,
            "max_iterations": 3,
            "success": False,
        }
    )

    print(f"Task: {result['task']}")
    print(f"Iterations: {result['iteration']}")
    print(f"Success: {result['success']}")
    print(f"Final Code:\n{result['code']}")


class ResearchState(TypedDict):
    topic: str
    findings: Annotated[list[str], operator.add]
    questions: list[str]
    iteration: int
    max_depth: int
    summary: str


def demo_iterative_research():
    """Iterative research that goes deeper based on findings."""

    def research(state: ResearchState) -> dict:
        print(f"\n{'─' * 50}")
        print(f"📚 [RESEARCH] Depth {state['iteration'] + 1}/{state['max_depth']}")

        if state["iteration"] == 0:
            query = f"Give me 3 key facts about: {state['topic']}"
            print(f"   Starting fresh on: {state['topic']}")
        else:
            question = state["questions"][-1] if state["questions"] else "elaborate"
            query = f"Based on these findings:\n{state['findings'][-1]}\n\nGo deeper: {question}"
            print(f"   Following up on: {question}")

        response = llm.invoke(query)
        print(f"   ✅ Found {len(response.content.splitlines())} lines of findings")
        print(f"   Preview: {response.content[:120]}...")
        return {"findings": [response.content]}

    def generate_questions(state: ResearchState) -> dict:
        print(f"\n{'─' * 50}")
        print(f"🤔 [QUESTIONING] Analyzing latest findings...")

        response = llm.invoke(
            f"Based on this finding:\n{state['findings'][-1]}\n\n"
            "What's one deeper question to explore? Reply with just the question."
        )

        print(f"   Next question: {response.content.strip()}")

        return {"questions": [response.content], "iteration": state["iteration"] + 1}

    def synthesize(state: ResearchState) -> dict:
        print(f"\n{'─' * 50}")
        print(
            f"🧬 [SYNTHESIZE] Combining {len(state['findings'])} rounds of findings..."
        )

        all_findings = "\n\n".join(state["findings"])
        response = llm.invoke(
            f"Synthesize these findings into a coherent summary:\n\n{all_findings}"
        )

        print(f"   ✅ Summary generated ({len(response.content.split())} words)")
        return {"summary": response.content}

    def should_continue(state: ResearchState) -> Literal["research", "synthesize"]:
        if state["iteration"] >= state["max_depth"]:
            print(
                f"\n🏁 [ROUTER] Max depth reached ({state['iteration']}/{state['max_depth']}) → synthesizing"
            )
            return "synthesize"
        print(
            f"\n🔄 [ROUTER] Depth {state['iteration']}/{state['max_depth']} → going deeper"
        )
        return "research"

    graph = StateGraph(ResearchState)

    graph.add_node("research", research)
    graph.add_node("generate_questions", generate_questions)
    graph.add_node("synthesize", synthesize)

    graph.add_edge(START, "research")
    graph.add_edge("research", "generate_questions")
    graph.add_conditional_edges(
        "generate_questions",
        should_continue,
        {"research": "research", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)

    app = graph.compile()

    print("=" * 50)
    print("🔬 ITERATIVE RESEARCH WORKFLOW")
    print("=" * 50)

    result = app.invoke(
        {
            "topic": "quantum computing applications",
            "findings": [],
            "questions": [],
            "iteration": 0,
            "max_depth": 2,
            "summary": "",
        }
    )

    print(f"\n{'=' * 50}")
    print(f"📊 RESEARCH COMPLETE")
    print(f"   Topic: {result['topic']}")
    print(f"   Depth reached: {result['iteration']}")
    print(f"   Findings collected: {len(result['findings'])}")
    print(f"   Questions explored: {len(result['questions'])}")
    print(f"\n📝 Final Summary:\n{result['summary']}")


if __name__ == "__main__":
    # demo_self_correcting_code()
    demo_iterative_research()
