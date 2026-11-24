from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing_extensions import TypedDict, Annotated
from typing import Literal
from pydantic import BaseModel, Field
import operator
from dotenv import load_dotenv

load_dotenv()


class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]  # conversation history
    next_agent: str  # the agent that should act next
    task_complete: bool  # whether the task is complete or not
    final_response: str  # the final response to the user when the task is complete


# NOTE (2026-11): gpt-4o-mini/gpt-4o shut down 2027-01-14 -> gpt-6-luna. reasoning_effort="none" lets it accept temperature + tools.
llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0.0)


# define what the supervisor can decide
class RouteDecision(BaseModel):
    """The Supervisor's routing decision."""

    next: Literal["researcher", "writer", "critic", "FINISH"] = Field(
        description="Which agent to call next, or FINISH if done"
    )
    reasoning: str = Field(description="Why this agent was chosen")


# Create structured output for reliable routing
supervisor_llm = llm.with_structured_output(RouteDecision)


def supervisor(state: SupervisorState) -> dict:
    """The Supervisor decides what to do next."""

    system_prompt = """You are a supervisor managing a team of specialists:

    1. researcher - Gathers information and facts
    2. writer - Creates content and text
    3. critic - Reviews and improves work

    Based on the conversation, decide which agent should act next.

    Workflow:
    - Start with researcher to gather facts
    - Then writer to create content
    - Then critic to review
    - If critic suggests changes, send back to writer
    - When quality is good, respond with FINISH
    """

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    decision = supervisor_llm.invoke(messages)

    if decision.next == "FINISH":
        return {"next_agent": "FINISH", "task_complete": True}

    return {
        "next_agent": decision.next,
        "messages": [
            AIMessage(
                content=f"[Supervisor] Routing to {decision.next}: {decision.reasoning}"
            )
        ],
    }


def researcher(state: SupervisorState) -> dict:
    """Gathers information and facts."""

    system = """You are a research specialist. Your job:
    - Gather relevant facts and information
    - Be thorough but concise
    - Cite sources when possible
    - Focus on what's most useful for the task"""

    # Get the original task from the first human message
    task = next(
        (m.content for m in state["messages"] if isinstance(m, HumanMessage)), ""
    )

    response = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=f"Research this topic: {task}"),
        ]
    )

    return {"messages": [AIMessage(content=f"[Researcher] {response.content}")]}


def writer(state: SupervisorState) -> dict:
    """Creates content based on available information."""

    system = """You are a writing specialist. Your job:
    - Create clear, engaging content
    - Use the research provided
    - Match the requested format and tone
    - If there's critic feedback, incorporate it"""

    # Get recent context (research + any feedback)
    context = "\n".join([m.content for m in state["messages"][-5:]])

    response = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=f"Create content based on:\n{context}"),
        ]
    )

    return {"messages": [AIMessage(content=f"[Writer] {response.content}")]}


def critic(state: SupervisorState) -> dict:
    """Reviews work and provides feedback."""

    system = """You are a quality critic. Your job:
    - Review the content objectively
    - Provide specific, actionable feedback
    - If the work is good, say "APPROVED" and explain why
    - If it needs work, explain exactly what to improve"""

    # Get the most recent work
    context = "\n".join([m.content for m in state["messages"][-3:]])

    response = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=f"Review this work:\n{context}"),
        ]
    )

    return {"messages": [AIMessage(content=f"[Critic] {response.content}")]}


def finalize(state: SupervisorState) -> dict:
    """Extract the final response when task is complete."""

    # Find the last Writer output
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and "[Writer]" in msg.content:
            content = msg.content.replace("[Writer] ", "")
            return {"final_response": content}

    return {"final_response": "Task completed."}


def route_to_agent(state: SupervisorState) -> str:
    """Route based on Supervisor's decision."""
    if state.get("task_complete"):
        return "finalize"
    return state["next_agent"]


def build_multi_agent_system():
    """Build the complete multi-agent graph."""

    # Create graph
    graph = StateGraph(SupervisorState)

    # Add all nodes
    graph.add_node("supervisor", supervisor)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("critic", critic)
    graph.add_node("finalize", finalize)

    # Entry point: always start with Supervisor
    graph.add_edge(START, "supervisor")

    # Supervisor routes to specialists
    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "researcher": "researcher",
            "writer": "writer",
            "critic": "critic",
            "finalize": "finalize",
        },
    )

    # After each specialist, go back to Supervisor
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("critic", "supervisor")

    # Finalize ends the graph
    graph.add_edge("finalize", END)

    return graph.compile()


if __name__ == "__main__":
    # Build the system
    agent = build_multi_agent_system()

    # Initial state
    initial_state = {
        "messages": [
            HumanMessage(
                content="Write a short blog post about the benefits of AI in healthcare"
            )
        ],
        "next_agent": "",
        "task_complete": False,
        "final_response": "",
    }

    print("=" * 60)
    print("MULTI-AGENT SYSTEM")
    print("=" * 60)

    # Run the system
    result = agent.invoke(initial_state)

    # Show the conversation
    print("\nAgent Conversation:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            # Truncate for display
            content = (
                msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            )
            print(f"\n{content}")

    print("\n" + "=" * 60)
    print("FINAL OUTPUT:")
    print("=" * 60)
    print(result["final_response"])
