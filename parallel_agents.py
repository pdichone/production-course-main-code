"""
Parallel Agent Execution in LangGraph
Running multiple agents simultaneously
"""

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import asyncio
from dotenv import load_dotenv

load_dotenv()

# NOTE (2026-11): gpt-4o-mini/gpt-4o shut down 2027-01-14 -> gpt-6-luna. reasoning_effort="none" lets it accept temperature + tools.
llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0.7)


class ParallelState(TypedDict):
    query: str
    research_result: str
    creative_result: str
    technical_result: str
    final_synthesis: str


def create_parallel_research():
    """Three research agents working in parallel."""

    def research_agent(state: ParallelState) -> dict:
        """Academic/factual research."""
        response = llm.invoke(
            [
                SystemMessage(
                    content="You are an academic researcher. Provide factual, well-sourced information."
                ),
                HumanMessage(content=f"Research this topic: {state['query']}"),
            ]
        )
        return {"research_result": response.content}

    def creative_agent(state: ParallelState) -> dict:
        """Creative perspectives."""
        response = llm.invoke(
            [
                SystemMessage(
                    content="You are a creative thinker. Provide novel perspectives and ideas."
                ),
                HumanMessage(content=f"Give creative insights on: {state['query']}"),
            ]
        )
        return {"creative_result": response.content}

    def technical_agent(state: ParallelState) -> dict:
        """Technical analysis."""
        response = llm.invoke(
            [
                SystemMessage(
                    content="You are a technical analyst. Provide practical, implementation-focused insights."
                ),
                HumanMessage(content=f"Analyze technically: {state['query']}"),
            ]
        )
        return {"technical_result": response.content}

    def synthesize(state: ParallelState) -> dict:
        """Combine all perspectives."""
        synthesis_prompt = f"""Synthesize these three perspectives into a comprehensive response:

        RESEARCH: {state['research_result']}

        CREATIVE: {state['creative_result']}

        TECHNICAL: {state['technical_result']}

        Create a unified, well-structured response."""

        response = llm.invoke(
            [
                SystemMessage(
                    content="You are an expert synthesizer. Combine multiple perspectives into coherent insights."
                ),
                HumanMessage(content=synthesis_prompt),
            ]
        )
        return {"final_synthesis": response.content}

    graph = StateGraph(ParallelState)

    graph.add_node("research", research_agent)
    graph.add_node("creative", creative_agent)
    graph.add_node("technical", technical_agent)
    graph.add_node("synthesize", synthesize)

    # Fan-out: START goes to all three agents
    graph.add_edge(START, "research")
    graph.add_edge(START, "creative")
    graph.add_edge(START, "technical")

    graph.add_edge("research", "synthesize")
    graph.add_edge("creative", "synthesize")
    graph.add_edge("technical", "synthesize")

    graph.add_edge("synthesize", END)

    return graph.compile()


def demo_parallel_execution():
    """Demo parallel agent execution."""

    agent = create_parallel_research()

    print("Parallel Agent Execution Demo:\n")

    result = agent.invoke(
        {
            "query": "The future of remote work",
            "research_result": "",
            "creative_result": "",
            "technical_result": "",
            "final_synthesis": "",
        }
    )

    print("Individual Perspectives:")
    print(f"\n[Research]\n{result['research_result'][:300]}...")
    print(f"\n[Creative]\n{result['creative_result'][:300]}...")
    print(f"\n[Technical]\n{result['technical_result'][:300]}...")

    print(f"\n{'='*50}")
    print(f"[SYNTHESIZED]\n{result['final_synthesis']}")


# Map-Reduce Pattern
class MapReduceState(TypedDict):
    documents: list[str]
    summaries: list[str]
    final_summary: str


def create_map_reduce_summarizer():
    """Summarize multiple documents in parallel."""

    def map_summarize(state: MapReduceState) -> dict:
        """Summarize each document (runs in parallel for each)."""
        summaries = []
        for doc in state["documents"]:
            response = llm.invoke(
                [
                    SystemMessage(content="Summarize this document in 2-3 sentences."),
                    HumanMessage(content=doc),
                ]
            )
            summaries.append(response.content)
        return {"summaries": summaries}

    def reduce_combine(state: MapReduceState) -> dict:
        """Combine all summaries."""
        all_summaries = "\n\n".join(
            [f"Summary {i+1}: {s}" for i, s in enumerate(state["summaries"])]
        )

        response = llm.invoke(
            [
                SystemMessage(
                    content="Combine these summaries into one coherent overview."
                ),
                HumanMessage(content=all_summaries),
            ]
        )
        return {"final_summary": response.content}

    graph = StateGraph(MapReduceState)
    graph.add_node("map", map_summarize)
    graph.add_node("reduce", reduce_combine)

    graph.add_edge(START, "map")
    graph.add_edge("map", "reduce")
    graph.add_edge("reduce", END)

    return graph.compile()


def demo_map_reduce():
    """Demo map-reduce pattern."""

    agent = create_map_reduce_summarizer()

    documents = [
        "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms and has a vast ecosystem of libraries.",
        "Machine learning is a subset of AI that enables systems to learn from data. Common approaches include supervised, unsupervised, and reinforcement learning.",
        "Cloud computing provides on-demand access to computing resources. Major providers include AWS, Azure, and Google Cloud Platform.",
    ]

    print("\nMap-Reduce Summarization Demo:\n")

    result = agent.invoke(
        {"documents": documents, "summaries": [], "final_summary": ""}
    )

    print("Individual summaries:")
    for i, summary in enumerate(result["summaries"]):
        print(f"  {i+1}. {summary}")

    print(f"\nCombined summary:\n{result['final_summary']}")


if __name__ == "__main__":
    # demo_parallel_execution()
    demo_map_reduce()
