from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

def build_agent():
    # Define the schema for the agent's state.
    schema = {
        "question": str,
        "session_id": str,
        "tool": str,
        "query": str,
        "trace": str,
        "answer": str,
    }
    workflow = StateGraph(state_schema=schema)
    workflow.add_node("select_tool", RunnableLambda(select_tool_node))
    workflow.add_node("execute_tool", RunnableLambda(execute_tool_node))
    workflow.set_entry_point("select_tool")
    workflow.add_edge("select_tool", "execute_tool")
    workflow.add_edge("execute_tool", END)
    return workflow.compile()

