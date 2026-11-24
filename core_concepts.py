"""
LangChain Core Concepts - LCEL and Runnables
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()


def demo_basic_chain():
    """Demonstrates a basic chain using LCEL and Runnables."""

    # Component 1: Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in one sentence: {question}"
    )
    # NOTE (2026-11): gpt-4o-mini/gpt-4o shut down 2027-01-14 -> gpt-6-luna. reasoning_effort="none" lets it accept temperature + tools.
    model = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0.7)
    parser = StrOutputParser()

    # Compose with pipe operator
    chain = prompt | model | parser

    # Execute the chain with an input
    result = chain.invoke({"question": "What is LangChain?"})
    print(f"Response: {result}")

    return chain


def demo_batch_exectution():
    """Demonstrate batch execution for multiple inputs."""
    prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
    model = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Batch - run with multiple inputs
    inputs = [
        {"text": "Hello, how are you?"},
        {"text": "What is your name?"},
        {"text": "Where is the nearest restaurant?"},
    ]
    results = chain.batch(inputs)

    for text in zip(inputs, results):
        print(f"Input: {text[0]['text']} => Output: {text[1]}")


def demo_streaming():
    """Demonstrate streaming for real-time output."""
    prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")
    model = ChatOpenAI(
        model="gpt-6-luna",
        reasoning_effort="none",
        temperature=0.7,
    )
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Streaming - run with streaming enabled
    print("Streaming output: ")
    for chunk in chain.stream({"topic": "nature"}):
        print(chunk, end="", flush=True)
    print()  # for newline after streaming


def demo_schema_inspection():
    """Demonstrate input/output schema inspection."""
    prompt = ChatPromptTemplate.from_template("Summarize the following text: {text}")
    model = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Inspect input and output schemas
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input Schema: {input_schema}")
    print(f"Output Schema: {output_schema}")


# ------- Exercise the demos -------#
# Exercise: Build your first chain
def exercise_first_chain():
    """
    EXERCISE: Create a chain that:
    1. Takes a product name and target audience
    2. Generates a marketing tagline
    3. Returns just the tagline as a string

    Test with: product="AI Course", audience="developers"
    """

    # YOUR CODE HERE
    prompt = ChatPromptTemplate.from_template(
        "Create a marketing tagline for a product named '{product}' targeting '{audience}'."
    )
    model = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Test the chain
    result = chain.invoke({"product": "AI Course", "audience": "developers"})
    print(f"Marketing Tagline: {result}")


def new_way():
    # the univeral way to initialize a model
    model = init_chat_model("gpt-6-luna", reasoning_effort="none", temperature=0.7, max_tokens=1500)

    # Or provider-specific (still works)

    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic

    openai_model = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none",
                              temperature=0.7,
                              max_tokens=1500,
                              timeout=30,
                              max_retries=3)
    
    # NOTE (2026-11): Claude Sonnet 4.5 retires 2026-11-30, so this uses claude-sonnet-5-5.
    # thinking="between_tools" keeps the plain-text replies shown in the video (Sonnet 5.5 thinks by default).
    anthropic_model = ChatAnthropic(
        model="claude-sonnet-5-5", thinking={"type": "between_tools"}
    )


if __name__ == "__main__":
    # demo_basic_chain()
    # demo_batch_exectution()
    # demo_streaming()
    # demo_schema_inspection()
    exercise_first_chain()
