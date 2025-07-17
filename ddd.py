import requests
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

"""
LangGraph agent that:
- Prompts LLM with tool options, descriptions, and examples.
- Parses LLM output to decide tool and query.
- Validates tool call and returns agent's reasoning trace.
"""

# === LLM Connection Config (Snowflake Cortex) ===
API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
MODEL = "llama3.1-70b"
SYS_MSG = """
You are an AI agent that helps users interact with a Neo4j database using MCP tools.
Use only the following tools, choosing the best one based on the user's request:

TOOL DESCRIPTIONS:
- read_neo4j_cypher: For all read-only Cypher queries (MATCH, RETURN, count, etc).
- write_neo4j_cypher: For creating, updating, or deleting nodes/relationships.
- get_neo4j_schema: For requests about graph structure, schema, labels, or relationship types.

EXAMPLES:
User: How many nodes are in the graph?
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN count(n)

User: Create a Person node named Alice
Tool: write_neo4j_cypher
Query: CREATE (:Person {name: 'Alice'})

User: Show me the schema of the graph
Tool: get_neo4j_schema

Always output your reasoning, then the tool name and the Cypher query (if needed).
"""

def cortex_llm(prompt: str, session_id: str) -> str:
    """
    Send a prompt to Cortex LLM, get output (reasoning, tool name, query).
    """
    headers = {
        "Authorization": f'Snowflake Token="{API_KEY}"',
        "Content-Type": "application/json"
    }
    payload = {
        "query": {
            "aplctn_cd": "edagnai",
            "app_id": "edadip",
            "api_key": API_KEY,
            "method": "cortex",
            "model": MODEL,
            "sys_msg": SYS_MSG,
            "limit_convs": "0",
            "prompt": {
                "messages": [{"role": "user", "content": prompt}]
            },
            "session_id": session_id
        }
    }
    resp = requests.post(API_URL, headers=headers, json=payload, verify=False)
    return resp.text.partition("end_of_stream")[0].strip()

def parse_llm_output(llm_output):
    """
    Extract tool name, query, and reasoning from LLM output.
    Handles minor formatting errors robustly.
    """
    import re
    trace = llm_output.strip()
    tool = None
    query = None
    # Find tool
    tool_match = re.search(r"Tool: ([\w_]+)", llm_output, re.I)
    if tool_match:
        tool = tool_match.group(1)
    # Find query
    query_match = re.search(r"Query: (.+)", llm_output, re.I)
    if query_match:
        query = query_match.group(1).strip()
    return tool, query, trace

def select_tool_node(state):
    """
    Node that sends question to the LLM, parses tool/query, returns reasoning trace.
    """
    question = state["question"]
    session_id = state["session_id"]
    llm_output = cortex_llm(question, session_id)
    tool, query, trace = parse_llm_output(llm_output)
    return {
        "question": question,
        "session_id": session_id,
        "tool": tool,
        "query": query,
        "trace": trace
    }

def execute_tool_node(state):
    """
    Executes the chosen tool with the Cypher query (if present).
    Returns the answer and agent trace.
    """
    tool = state.get("tool")
    query = state.get("query")
    trace = state.get("trace")
    # Only allowed tools:
    valid_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    if tool not in valid_tools:
        return {"answer": f"⚠️ MCP tool not recognized: {tool}", "trace": trace}
    try:
        if tool == "get_neo4j_schema":
            result = requests.post("http://localhost:8000/mcp/get_neo4j_schema")
            answer = result.json() if result.ok else result.text
        else:
            # Send query to MCP tool
            data = {"query": query, "params": {}}
            result = requests.post(f"http://localhost:8000/mcp/{tool}", json=data)
            answer = result.json() if result.ok else result.text
        return {"answer": answer, "trace": trace}
    except Exception as e:
        return {"answer": f"⚠️ MCP execution failed: {str(e)}", "trace": trace}

def build_agent():
    """
    Build and compile the LangGraph workflow.
    """
    workflow = StateGraph()
    workflow.add_node("select_tool", RunnableLambda(select_tool_node))
    workflow.add_node("execute_tool", RunnableLambda(execute_tool_node))
    workflow.set_entry_point("select_tool")
    workflow.add_edge("select_tool", "execute_tool")
    workflow.add_edge("execute_tool", END)
    return workflow.compile()
