"""LangGraph order support agent."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.state import AgentState
from app.orders import ORDER_TOOLS
from app.providers import get_chat_model
from app.rag import search_policies


def _default_tools() -> list[BaseTool]:
    return [*ORDER_TOOLS, search_policies]


def default_tool_error_handler(error: Exception) -> str:
    """Graceful error message returned to the agent when a tool encounters an outage or failure."""
    return "The orders service is temporarily unavailable. Please try again in a few moments."


def build_agent(
    *,
    llm: BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build and compile the order support graph."""
    model = llm or get_chat_model()
    agent_tools = list(tools) if tools is not None else _default_tools()
    model_with_tools = model.bind_tools(agent_tools)

    def assistant(state: AgentState) -> dict[str, list[BaseMessage]]:
        messages: list[BaseMessage] = [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"],
        ]
        return {"messages": [model_with_tools.invoke(messages)]}

    graph = StateGraph(AgentState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(agent_tools, handle_tool_errors=default_tool_error_handler))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges(
        "assistant",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "assistant")
    return graph.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def get_checkpointer() -> BaseCheckpointSaver:
    """Return the shared in-memory checkpointer."""
    return MemorySaver()


@lru_cache(maxsize=1)
def get_agent() -> CompiledStateGraph:
    """Return the shared agent configured from the environment with conversation memory."""
    return build_agent(checkpointer=get_checkpointer())
