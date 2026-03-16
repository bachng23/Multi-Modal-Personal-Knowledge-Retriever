from langgraph.graph import MessagesState


class AgentState(MessagesState):
    summary: str = ""
