from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents.nodes import call_model, should_continue, summarize_conversation
from src.agents.state import AgentState
from src.agents.tools import all_tools

tool_node = ToolNode(all_tools)

builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_node("sensitive_tools", tool_node)
builder.add_node("summarize", summarize_conversation)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "sensitive_tools": "sensitive_tools",
    "__end__": "summarize",
})
builder.add_edge("tools", "agent")
builder.add_edge("sensitive_tools", "agent")
builder.add_edge("summarize", END)

# For langgraph dev: compiled without checkpointer (server injects its own)
graph = builder.compile(
    interrupt_before=["sensitive_tools"],
)
