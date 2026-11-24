"""
Supervisor Architecture in LangGraph
One agent coordinates multiple specialist agents
"""

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from typing import Literal
from pydantic import BaseModel, Field
import operator
from dotenv import load_dotenv

load_dotenv()


class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    task_complete: bool
    final_response: str


# NOTE (2026-11): gpt-4o-mini/gpt-4o shut down 2027-01-14 -> gpt-6-luna. reasoning_effort="none" lets it accept temperature + tools.
llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0)


def create_supervisor_system():
    """Create a supervisor with specialist agents."""

    # Define the routing schema
    class RouteDecision(BaseModel):
        next: Literal["researcher", "writer", "critic", "FINISH"] = Field(
            description="The next agent to call, or FINISH if task is complete"
        )
        reasoning: str = Field(description="Why this agent was chosen")

    supervisor_llm = llm.with_structured_output(RouteDecision)

    # Supervisor node
    def supervisor(state: SupervisorState) -> dict:
        system_prompt = """You are a supervisor managing a team of specialists:

        1. researcher - Gathers information and facts
        2. writer - Creates content and text
        3. critic - Reviews and improves work

        Based on the conversation, decide which agent should act next.
        If the task is complete, respond with FINISH.

        Current conversation shows the progress so far."""

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

    # Define specialist agents (for demo purposes, they just echo the task)
    def researcher(state: SupervisorState) -> dict:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a research specialist. Gather facts and information relevant to the task. Be thorough but concise.",
                ),
                (
                    "human",
                    "Task context:\n{context}\n\nProvide your research findings.",
                ),
            ]
        )

        # Get task from first human message
        task = next(
            (m.content for m in state["messages"] if isinstance(m, HumanMessage)), ""
        )

        response = llm.invoke(prompt.format_messages(context=task))

        return {"messages": [AIMessage(content=f"[Researcher] {response.content}")]}

    def writer(state: SupervisorState) -> dict:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a writing specialist. Create clear, engaging content based on the available information.",
                ),
                ("human", "Previous work:\n{context}\n\nWrite the content."),
            ]
        )

        context = "\n".join([m.content for m in state["messages"][-5:]])
        response = llm.invoke(prompt.format_messages(context=context))

        return {"messages": [AIMessage(content=f"[Writer] {response.content}")]}

    def critic(state: SupervisorState) -> dict:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a quality critic. Review the work and provide constructive feedback. If the work is good, say so.",
                ),
                ("human", "Work to review:\n{context}\n\nProvide your critique."),
            ]
        )

        context = "\n".join([m.content for m in state["messages"][-3:]])
        response = llm.invoke(prompt.format_messages(context=context))

        return {"messages": [AIMessage(content=f"[Critic] {response.content}")]}

    def finalize(state: SupervisorState) -> dict:
        # Get the last substantial response
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and "[Writer]" in msg.content:
                content = msg.content.replace("[Writer] ", "")
                return {"final_response": content}

        return {"final_response": "Task completed."}

    # Route based on supervisor decision
    def route_to_agent(state: SupervisorState) -> str:
        if state.get("task_complete"):
            return "finalize"
        return state["next_agent"]

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("critic", critic)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "supervisor")

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

    # After each specialist, go back to supervisor
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("critic", "supervisor")
    graph.add_edge("finalize", END)

    return graph.compile()


def demo_supervisor():
    """Demo the supervisor system."""

    agent = create_supervisor_system()

    print("Supervisor Agent Demo:\n")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Write a short blog post about the benefits of AI in healthcare"
                )
            ],
            "next_agent": "",
            "task_complete": False,
            "final_response": "",
        }
    )

    print("Agent conversation:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")

    print(f"\n\nFinal Response:\n{result['final_response']}")


def demo_supervisor_trace():
    """Show supervisor decision-making."""

    agent = create_supervisor_system()

    print("\nSupervisor Decision Trace:\n")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Create a marketing tagline for a new coffee brand"
                )
            ],
            "next_agent": "",
            "task_complete": False,
            "final_response": "",
        }
    )

    print("Routing decisions:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and "[Supervisor]" in msg.content:
            print(f"  → {msg.content}")


if __name__ == "__main__":
    # demo_supervisor()
    demo_supervisor_trace()
