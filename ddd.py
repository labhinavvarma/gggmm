import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ... (rest of your imports and code) ...

def execute_tool_node(state: AgentState) -> dict:
    tool = state.tool
    query = state.query
    trace = state.trace
    answer = ""
    valid_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    try:
        if tool not in valid_tools:
            answer = f"⚠️ MCP tool not recognized: {tool}"
        elif tool == "get_neo4j_schema":
            result = requests.post("http://localhost:8000/mcp/get_neo4j_schema", headers=headers)
            answer = result.json() if result.ok else result.text
        else:
            data = {"query": query, "params": {}}
            result = requests.post(f"http://localhost:8000/mcp/{tool}", json=data, headers=headers)
            answer = result.json() if result.ok else result.text
    except Exception as e:
        answer = f"⚠️ MCP execution failed: {str(e)}"
    return {
        "question": state.question,
        "session_id": state.session_id,
        "tool": tool,
        "query": query,
        "trace": trace,
        "answer": answer
    }

