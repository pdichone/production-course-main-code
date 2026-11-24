"""
Section 4 Project: Multi-Agent Research System
Combines all patterns from Section 4 into a working research pipeline.

Patterns used:
- Supervisor architecture (4.3)
- Parallel execution (4.5) via Send API
- Shared state / blackboard (4.6)
- Iterative refinement loop
"""

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing_extensions import TypedDict, Annotated
from typing import Literal
from pydantic import BaseModel, Field
import operator
import json
from dotenv import load_dotenv

load_dotenv()

# NOTE (2026-11): gpt-4o-mini/gpt-4o shut down 2027-01-14 -> gpt-6-luna. reasoning_effort="none" lets it accept temperature + tools.
llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0)
creative_llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0.7)


# ============================================================
# State Schema
# ============================================================


class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str
    search_queries: list[str]
    findings: Annotated[list[dict], operator.add]
    analysis: str
    report: str
    quality_score: float
    quality_feedback: str
    iteration: int


# State for individual search tasks (used with Send API)
class SearchTaskState(TypedDict):
    search_query: str
    findings: Annotated[list[dict], operator.add]


# ============================================================
# Node: Supervisor — Plans the research
# ============================================================


def supervisor(state: ResearchState) -> dict:
    """Plans research by generating targeted search queries."""

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a research supervisor. Given a topic, generate exactly 3 "
                    "specific search queries that will cover different angles of the topic. "
                    "Return ONLY a JSON array of strings. No markdown formatting."
                )
            ),
            HumanMessage(content=f"Research topic: {state['topic']}"),
        ]
    )

    try:
        queries = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback: split by newlines
        queries = [
            f"{state['topic']} overview",
            f"{state['topic']} latest developments",
            f"{state['topic']} practical applications",
        ]

    return {
        "search_queries": queries[:3],
        "messages": [
            AIMessage(
                content=f"[SUPERVISOR]: Planned {len(queries)} research queries: {queries}",
                name="supervisor",
            )
        ],
    }


# ============================================================
# Node: Search Agent — Executes a single search query
# (Launched in parallel via Send API)
# ============================================================


def search_agent(state: SearchTaskState) -> dict:
    """
    Executes one search query and returns findings.
    Each instance runs in parallel via the Send API.
    """
    query = state["search_query"]

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a web research agent. For the given search query, "
                    "provide 2-3 key findings. Each finding should have a 'title' "
                    "and 'detail' field. Return a JSON array. No markdown."
                )
            ),
            HumanMessage(content=f"Search query: {query}"),
        ]
    )

    try:
        results = json.loads(response.content)
    except json.JSONDecodeError:
        results = [{"title": query, "detail": response.content}]

    # Tag each finding with the query it came from
    for r in results:
        r["source_query"] = query

    return {"findings": results}


# ============================================================
# Edge: Fan-out searches using Send API
# ============================================================


def dispatch_searches(state: ResearchState) -> list[Send]:
    """Dynamically create parallel search tasks using Send API."""
    return [
        Send("search_agent", {"search_query": query, "findings": []})
        for query in state["search_queries"]
    ]


# ============================================================
# Node: Analyst — Synthesizes all findings
# ============================================================


def analyst(state: ResearchState) -> dict:
    """Reads all findings from the blackboard and synthesizes."""
    findings_text = json.dumps(state["findings"], indent=2)

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a research analyst. Synthesize the collected findings into "
                    "a clear analysis. Identify:\n"
                    "1. Key themes across all findings\n"
                    "2. Any contradictions or gaps\n"
                    "3. The most important insights\n\n"
                    "Write 2-3 paragraphs."
                )
            ),
            HumanMessage(
                content=(
                    f"Research topic: {state['topic']}\n\n"
                    f"Collected findings:\n{findings_text}"
                )
            ),
        ]
    )

    return {
        "analysis": response.content,
        "messages": [
            AIMessage(content=f"[ANALYST]: {response.content}", name="analyst")
        ],
    }


# ============================================================
# Node: Report Writer — Produces the final report
# ============================================================


def report_writer(state: ResearchState) -> dict:
    """Writes a structured research report from the analysis."""

    # Include quality feedback if this is a revision
    revision_note = ""
    if state["iteration"] > 0 and state.get("quality_feedback"):
        revision_note = (
            f"\n\nIMPORTANT — This is revision #{state['iteration']}. "
            f"Address this feedback: {state['quality_feedback']}"
        )

    response = creative_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a report writer. Produce a well-structured research report "
                    "with these sections:\n"
                    "1. Executive Summary (2-3 sentences)\n"
                    "2. Key Findings (bullet points)\n"
                    "3. Analysis (1-2 paragraphs)\n"
                    "4. Recommendations (3 actionable items)\n\n"
                    "Use markdown formatting. Be specific and actionable."
                    f"{revision_note}"
                )
            ),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n\n"
                    f"Analysis:\n{state['analysis']}\n\n"
                    f"Raw findings:\n{json.dumps(state['findings'][:6], indent=2)}"
                )
            ),
        ]
    )

    return {
        "report": response.content,
        "messages": [
            AIMessage(
                content=f"[REPORT WRITER]: Report {'revised' if state['iteration'] > 0 else 'drafted'}.",
                name="report_writer",
            )
        ],
    }


# ============================================================
# Node: Quality Checker — Reviews and scores the report
# ============================================================


class QualityReview(BaseModel):
    score: float = Field(description="Quality score from 0.0 to 1.0")
    feedback: str = Field(description="Specific feedback for improvement")
    approved: bool = Field(description="Whether the report meets quality standards")


def quality_checker(state: ResearchState) -> dict:
    """Reviews the report and either approves or sends back for revision."""

    review_llm = llm.with_structured_output(QualityReview)

    review = review_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a quality reviewer. Score the report on:\n"
                    "- Completeness: Does it cover the topic well?\n"
                    "- Clarity: Is it well-written and easy to understand?\n"
                    "- Actionability: Are recommendations specific?\n\n"
                    "Score from 0.0 to 1.0. Approve if score >= 0.7.\n"
                    "If this is iteration 2 or higher, be more lenient."
                )
            ),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Iteration: {state['iteration']}\n\n"
                    f"Report:\n{state['report']}"
                )
            ),
        ]
    )

    # Force approve after 2 iterations to prevent infinite loops
    approved = review.approved or state["iteration"] >= 2

    return {
        "quality_score": review.score,
        "quality_feedback": review.feedback,
        "iteration": state["iteration"] + 1,
        "messages": [
            AIMessage(
                content=(
                    f"[QUALITY CHECK]: Score {review.score:.1f} — "
                    f"{'APPROVED' if approved else 'REVISION NEEDED'}: {review.feedback}"
                ),
                name="quality_checker",
            )
        ],
    }


# ============================================================
# Build the full research system
# ============================================================


def create_research_system():
    """
    Builds the multi-agent research system graph.

    Flow:
    1. Supervisor plans search queries
    2. Search agents run in parallel (Send API)
    3. Analyst synthesizes findings
    4. Report writer produces report
    5. Quality checker reviews — loops back if needed
    """

    graph = StateGraph(ResearchState)

    # Add all nodes
    graph.add_node("supervisor", supervisor)
    graph.add_node("search_agent", search_agent)
    graph.add_node("analyst", analyst)
    graph.add_node("report_writer", report_writer)
    graph.add_node("quality_checker", quality_checker)

    # Edges
    graph.add_edge(START, "supervisor")

    # Supervisor → parallel search agents (dynamic fan-out)
    graph.add_conditional_edges("supervisor", dispatch_searches, ["search_agent"])

    # All search agents → analyst (fan-in)
    graph.add_edge("search_agent", "analyst")

    # Analyst → report writer
    graph.add_edge("analyst", "report_writer")

    # Report writer → quality checker
    graph.add_edge("report_writer", "quality_checker")

    # Quality gate: approve or revise
    graph.add_conditional_edges(
        "quality_checker", quality_gate, {"report_writer": "report_writer", "end": END}
    )

    return graph.compile()


# Demos
def demo_research_with_streaming():
    """Run the research system with step-by-step streaming output."""

    system = create_research_system()
    graph = system.get_graph()
    png_data = graph.draw_mermaid_png()

    with open("research_graph.png", "wb") as f:
        f.write(png_data)

    topic = "Best practices for building multi-agent AI systems"
    print(f"Streaming Research: {topic}\n")

    initial_state = {
        "messages": [],
        "topic": topic,
        "search_queries": [],
        "findings": [],
        "analysis": "",
        "report": "",
        "quality_score": 0.0,
        "quality_feedback": "",
        "iteration": 0,
    }

    # Stream updates to see each step as it happens
    for step in system.stream(initial_state, stream_mode="updates"):
        for node_name, update in step.items():
            print(f"[{node_name}] completed")

            # Show interesting state changes
            if "search_queries" in update and update["search_queries"]:
                print(f"  Planned queries: {update['search_queries']}")
            if "findings" in update and update["findings"]:
                print(f"  Found {len(update['findings'])} results")
            if "quality_score" in update:
                print(f"  Quality score: {update['quality_score']:.1f}")
            if "report" in update and update["report"]:
                print(f"  Report length: {len(update['report'])} chars")

        print()


# ============================================================
# Routing: Quality gate
# ============================================================


def quality_gate(state: ResearchState) -> Literal["report_writer", "end"]:
    """Route back to writer if quality is insufficient."""
    if state["quality_score"] >= 0.7 or state["iteration"] >= 2:
        return "end"
    return "report_writer"


def demo_individual_search():
    """Demo just the search agent for testing."""

    print("Individual Search Agent Test:\n")

    # Test the search agent directly
    result = search_agent(
        {"search_query": "LangGraph multi-agent patterns", "findings": []}
    )

    print(f"Findings from search:")
    for f in result["findings"]:
        print(f"  - {f.get('title', 'N/A')}: {f.get('detail', 'N/A')[:80]}...")


def demo_full_research():
    """Run the complete research pipeline."""

    system = create_research_system()

    print("Multi-Agent Research System Demo")
    print("=" * 60)

    topic = "The impact of AI agents on software development in 2026"
    print(f"Topic: {topic}\n")

    result = system.invoke(
        {
            "messages": [],
            "topic": topic,
            "search_queries": [],
            "findings": [],
            "analysis": "",
            "report": "",
            "quality_score": 0.0,
            "quality_feedback": "",
            "iteration": 0,
        }
    )

    # Print the conversation trace
    print("Agent Activity Log:")
    print("-" * 40)
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"  {msg.content[:120]}...")
    print()

    # Print final stats
    print(f"Total findings collected: {len(result['findings'])}")
    print(f"Quality score: {result['quality_score']:.1f}")
    print(f"Iterations: {result['iteration']}")
    print()

    # Print the final report
    print("=" * 60)
    print("FINAL RESEARCH REPORT")
    print("=" * 60)
    print(result["report"])


if __name__ == "__main__":
    # Run the individual search agent demo
    # demo_individual_search()
    # print("\n" + "=" * 50 + "\n")
    # Run the full research system demo with streaming output
    # demo_research_with_streaming()
    # print("\n" + "=" * 50 + "\n")
    # Run the full research system demo without streaming output
    demo_full_research()
