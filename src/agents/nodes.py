from typing import Literal

from langchain_core.messages import (
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_openrouter import ChatOpenRouter

from src.agents.prompts import SYSTEM_PROMPT
from src.agents.state import AgentState
from src.agents.tools import SENSITIVE_TOOLS, all_tools
from src.core.config import config

base_llm = ChatOpenRouter(model=config.LLM_MODEL)
llm_with_tools = base_llm.bind_tools(all_tools)

SUMMARIZE_THRESHOLD = 10


async def call_model(state: AgentState):
    summary = state.get("summary", "")
    sys_msg = SYSTEM_PROMPT
    if summary:
        sys_msg += f"\n\nConversation summary so far:\n{summary}"

    messages = [SystemMessage(content=sys_msg)] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


async def summarize_conversation(state: AgentState):
    if len(state["messages"]) <= SUMMARIZE_THRESHOLD:
        return {}

    summary = state.get("summary", "")
    if summary:
        prompt = (
            f"This is the summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        prompt = "Create a brief summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=prompt)]
    response = await base_llm.ainvoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-1]]
    return {"summary": response.content, "messages": delete_messages}


def should_continue(state: AgentState) -> Literal["tools", "sensitive_tools", "__end__"]:
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        tool_names = {tc["name"] for tc in last_message.tool_calls}
        if tool_names & SENSITIVE_TOOLS:
            return "sensitive_tools"
        return "tools"

    return "__end__"
