"""
LangSmith Setup and Observability
Production monitoring for LangChain/LangGraph
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from langsmith.run_trees import RunTree
from dotenv import load_dotenv

load_dotenv()

# Enable tracing
os.environ["LANGSMITH_TRACING"] = "true"


@traceable(name="basic_chaining")
def demo_basic_tracing():
    """Basic LangSmith tracing."""

    # NOTE (2026-11): gpt-4o-mini/gpt-4o shut down 2027-01-14 -> gpt-6-luna. reasoning_effort="none" lets it accept temperature + tools.
    llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0)

    prompt = ChatPromptTemplate.from_template("Explain {topic} in one sentence.")

    chain = prompt | llm | StrOutputParser()

    print("Basic Tracing Demo:\n")
    print("Running chain with LangSmith tracing enabled...")

    result = chain.invoke({"topic": "machine learning"})

    print(f"Result: {result}")
    print("\nCheck LangSmith dashboard for trace details.")


@traceable(name="named_runs_demo", tags=["production", "summarization"])
def demo_named_runs():
    """Name your runs for easier identification."""

    llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0)

    prompt = ChatPromptTemplate.from_template("Summarize: {text}")

    chain = prompt | llm | StrOutputParser()

    print("\nNamed Runs Demo:\n")

    result = chain.invoke(
        {"text": "LangSmith provides observability for LLM applications."}
    )

    print(f"Result: {result}")
    print("Run tagged with 'production', 'summarization'")


@traceable(name="trace_with_metadata_demo", tags=["metadata", "filtering"])
def demo_trace_with_metadata(user_id: str, request_type: str):
    """Add metadata to traces for filtering."""

    llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0)

    # Metadata is automatically captured
    result = llm.invoke(f"Hello from user {user_id}")

    return result.content


if __name__ == "__main__":
    demo_basic_tracing()
    demo_named_runs()
    demo_trace_with_metadata(user_id="user_123", request_type="greeting")
