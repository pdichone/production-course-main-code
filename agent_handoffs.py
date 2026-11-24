"""
Agent Handoffs in LangGraph
Passing control and context between agents
"""

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from typing import Literal
from pydantic import BaseModel, Field
import operator
from dotenv import load_dotenv

load_dotenv()

# NOTE (2026-11): gpt-4o-mini/gpt-4o shut down 2027-01-14 -> gpt-6-luna. reasoning_effort="none" lets it accept temperature + tools.
llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0)


class HandoffState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: str
    handoff_reason: str
    context_summary: str


class HandoffDecision(BaseModel):
    handoff_to: Literal["sales", "support", "billing", "stay", "end"] = Field(
        description="Which agent to hand off to"
    )
    reason: str = Field(description="Reason for handoff")
    context: str = Field(description="Key context to pass to next agent")


def create_customer_service_system():

    def triage_agent(state: HandoffState) -> dict:
        """Initial triage to route customer."""
        system = """You are a customer service triage agent. Your job is to:
        1. Understand the customer's need
        2. Route to the appropriate specialist:
           - sales: Product questions, purchases, upgrades
           - support: Technical issues, bugs, how-to questions
           - billing: Payments, invoices, refunds
           - end: Simple questions you can answer directly

        Analyze the customer's message and decide where to route them."""

        handoff_llm = llm.with_structured_output(HandoffDecision)

        messages = [SystemMessage(content=system)] + state["messages"]
        decision = handoff_llm.invoke(messages)

        if decision.handoff_to == "end":
            # Answer directly
            response = llm.invoke(
                [
                    SystemMessage(
                        content="Provide a brief, helpful response to the customer."
                    ),
                    *state["messages"],
                ]
            )
            return {
                "messages": [AIMessage(content=f"[Triage] {response.content}")],
                "current_agent": "end",
            }

        return {
            "current_agent": decision.handoff_to,
            "handoff_reason": decision.reason,
            "context_summary": decision.context,
            "messages": [
                AIMessage(
                    content=f"[Triage] Transferring to {decision.handoff_to}: {decision.reason}"
                )
            ],
        }

    def sales_agent(state: HandoffState) -> dict:
        """Sales specialist."""
        system = f"""You are a sales specialist. Context from triage: {state.get('context_summary', 'None')}

            Help the customer with product questions and purchases.
            Be helpful and informative, not pushy."""

        response = llm.invoke([SystemMessage(content=system), *state["messages"]])

        return {
            "messages": [AIMessage(content=f"[Sales] {response.content}")],
            "current_agent": "sales_complete",
        }

    def support_agent(state: HandoffState) -> dict:
        """Technical support specialist."""
        system = f"""You are a technical support specialist. Context from triage: {state.get('context_summary', 'None')}

        Help the customer with technical issues.
        Be patient and provide step-by-step guidance."""

        response = llm.invoke([SystemMessage(content=system), *state["messages"]])

        return {
            "messages": [AIMessage(content=f"[Support] {response.content}")],
            "current_agent": "support_complete",
        }

    def billing_agent(state: HandoffState) -> dict:
        """Billing specialist."""
        system = f"""You are a billing specialist. Context from triage: {state.get('context_summary', 'None')}

        Help the customer with billing questions.
        Be clear about policies and next steps."""

        response = llm.invoke([SystemMessage(content=system), *state["messages"]])

        return {
            "messages": [AIMessage(content=f"[Billing] {response.content}")],
            "current_agent": "billing_complete",
        }

    def route_from_triage(state: HandoffState) -> str:
        agent = state["current_agent"]
        if agent in ["sales", "support", "billing"]:
            return agent
        return "end"

    graph = StateGraph(HandoffState)

    graph.add_node("triage", triage_agent)
    graph.add_node("sales", sales_agent)
    graph.add_node("support", support_agent)
    graph.add_node("billing", billing_agent)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        route_from_triage,
        {"sales": "sales", "support": "support", "billing": "billing", "end": END},
    )

    graph.add_edge("sales", END)
    graph.add_edge("support", END)
    graph.add_edge("billing", END)

    return graph.compile()


def demo_handoffs():
    """Demo customer service handoffs."""

    agent = create_customer_service_system()

    print("Customer Service Handoff Demo:\n")

    queries = [
        "My app keeps crashing when I try to upload photos",
        "I want to upgrade to the premium plan",
        "I was charged twice for my subscription",
        "What time do you close?",
    ]

    for query in queries:
        print(f"Customer: {query}")

        result = agent.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "current_agent": "",
                "handoff_reason": "",
                "context_summary": "",
            }
        )

        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                print(f"  {msg.content[:150]}...")

        print("-" * 50)


if __name__ == "__main__":
    demo_handoffs()
