"""
Working with LLMs in LangChain V.1
Multiple providers, configuration, streaming, and cost optimization
"""

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


def demo_init_chat_model():
    chat_model = init_chat_model(
        model="gpt-4o-mini",
        # model_provider="openai",
        temperature=0.7,
        streaming=True,
        max_retries=3,
    )

    response = chat_model.invoke("What is the capital of France? Answer in one word.")
    print(f"Response: {response.content}")

    # easy to switch model providers
    if os.getenv("ANTHROPIC_API_KEY"):
        # NOTE (2026-11): Claude Sonnet 4.5 retires 2026-11-30, so this uses claude-sonnet-5-5.
        # The video passes temperature=0.7 here too; Sonnet 5.5 rejects non-default temperature
        # (400), so it is dropped. thinking="between_tools" keeps response.content a plain string.
        claude = init_chat_model(
            model="claude-sonnet-5-5",
            model_provider="anthropic",
            thinking={"type": "between_tools"},
            streaming=True,
            max_retries=3,
        )

        response = claude.invoke("What is the capital of France? Answer in one word.")
        print(f"Response from Anthropic: {response.content}")


def demo_model_comparison():
    prompt = "Explain recursion in one sentence."

    models = {
        "gpt-4o-mini": init_chat_model(
            model="gpt-4o-mini",
            temperature=0.7,
            streaming=False,
        ),
        "gpt-4o": init_chat_model(
            model="gpt-4o",
            temperature=0.7,
            streaming=False,
        ),
    }

    # add anthropic model if available
    if os.getenv("ANTHROPIC_API_KEY"):
        # NOTE (2026-11): same change as above - claude-sonnet-5-5, no temperature (400 on Sonnet 5.5).
        models["claude-sonnet-5-5"] = init_chat_model(
            model="claude-sonnet-5-5",
            model_provider="anthropic",
            thinking={"type": "between_tools"},
            streaming=False,
        )

    print(f"Prompt: {prompt}\n")

    for model_name, model in models.items():
        response = model.invoke(prompt)
        print(f"Response from {model_name}: {response.content}\n")


def demo_message():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # using message objects (more control over roles)
    messages = [
        SystemMessage(content="You are a pirate. Always answer like a pirate."),
        HumanMessage(content="What's the weather like today?"),
    ]
    # print("Using message objects:")
    # print(f"Messages: {messages[0]} | {messages[1]}")

    response = model.invoke(messages)
    print(f"Response from the Pirate: {response.content}")

    # Multi-turn conversation using message objects
    messages.append(response)  # add model's response to the conversation
    messages.append(HumanMessage(content="What about tomorrow?"))

    print("\nMulti-turn conversation:")
    response = model.invoke(messages)
    print(f"Follow-up response from the Pirate: {response.content}")


# Exercise: Multi-model setup
def exercise_multi_model():
    """
    EXERCISE: Create a function that:
    1. Takes a question and a list of model names
    2. Gets responses from all models
    3. Returns a dict of {model_name: response}

    Test with: question="What is AI?", models=["gpt-4o-mini", "gpt-4o"]
    """

    def get_responses(question: str, model_names: list[str]) -> dict[str, str]:
        responses = {}
        for model_name in model_names:
            model = init_chat_model(
                model=model_name,
                temperature=0.7,
                streaming=False,
            )
            response = model.invoke(question)
            responses[model_name] = response.content
        return responses

    # Test the function
    results = get_responses("What is AI?", ["gpt-4o-mini", "gpt-4o"])
    for model, answer in results.items():
        print(f"Response from {model}: {answer}\n")


if __name__ == "__main__":
    # demo_init_chat_model()
    # demo_model_comparison()
    # demo_message()
    exercise_multi_model()
