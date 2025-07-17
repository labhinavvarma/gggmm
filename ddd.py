import requests
import urllib3
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import re

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class AgentState(BaseModel):
    question: str
    session_id: str
    tool: str = ""
    query: str = ""
    trace: str = ""
    answer: str = ""

def clean_cypher_query(query: str) -> str:
    # Remove code block markers
    query = re.sub(r'```(?:cypher|sql)?\s*', '', query, flags=re.IGNORECASE)
    query = re.sub(r'```', '', query)
    
    # Remove extra whitespace and newlines
    query = re.sub(r'[\r\n]+', ' ', query)
    
    # Add proper spacing around keywords
    keywords = [
        "MATCH", "WITH", "RETURN", "ORDER BY", "UNWIND", "WHERE", "LIMIT",
        "SKIP", "CALL", "YIELD", "CREATE", "MERGE", "SET", "DELETE", "DETACH DELETE", "REMOVE"
    ]
    for kw in keywords:
        query = re.sub(rf'(?<!\s)({kw})', r' \1', query)
        query = re.sub(rf'({kw})([^\s\(])', r'\1 \2', query)
    
    # Clean up multiple spaces
    query = re.sub(r'\s+', ' ', query)
    return query.strip()

SYS_MSG = """
You are an expert AI assistant that helps users query and manage a Neo4j database by selecting and using one of three MCP tools. Choose the most appropriate tool and generate the correct Cypher query or action.

TOOL DESCRIPTIONS:
- read_neo4j_cypher:
    - Use for all read-only graph queries: exploring data, finding nodes/relationships, aggregation, reporting, analysis, counting.
    - Only run safe queries (MATCH, RETURN, WHERE, OPTIONAL MATCH, etc).
    - NEVER use this tool for CREATE, UPDATE, DELETE, SET, or any modification.
    - Returns a list of matching nodes, relationships, or computed values.

- write_neo4j_cypher:
    - Use ONLY for write queries: CREATE, MERGE, SET, DELETE, REMOVE, or modifying properties or structure.
    - Use to create nodes/edges, update, or delete data.
    - NEVER use this for data retrieval only.
    - Returns a confirmation that the action was executed.

- get_neo4j_schema:
    - Use when the user asks about structure, schema, labels, relationship types, available node kinds, or properties.
    - Returns a detailed schema graph, including node labels, relationship types, and property keys.

IMPORTANT GUIDELINES:
- ALWAYS output your reasoning and then the tool and Cypher query (if any).
- Use this EXACT format for your response:
  Tool: [tool_name]
  Query: [cypher_query_on_single_line]
- Do NOT use code blocks or markdown formatting for the query.
- Put the entire Cypher query on one line after "Query: ".
- If the user requests the number of nodes and the result is unexpectedly low, try the admin-level count as a fallback:
    CALL db.stats.retrieve('GRAPH COUNTS') YIELD data RETURN data['NodeCount'] AS node_count
- If the user asks for schema, always use get_neo4j_schema.
- For ambiguous requests, ask clarifying questions or choose the safest tool.

FEW-SHOT EXAMPLES:

User: How many nodes are in the graph?
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN count(n)

User: Give me the true node count (admin)
Tool: read_neo4j_cypher
Query: CALL db.stats.retrieve('GRAPH COUNTS') YIELD data RETURN data['NodeCount'] AS node_count

User: List all Person nodes
Tool: read_neo4j_cypher
Query: MATCH (n:Person) RETURN n

User: Show the schema of the database
Tool: get_neo4j_schema

User: Create a Person node named Alice
Tool: write_neo4j_cypher
Query: CREATE (:Person {name: 'Alice'})

User: List all nodes with most number of relationships
Tool: read_neo4j_cypher
Query: MATCH (n) WITH n, size((n)--()) as rel_count WITH collect({node: n, count: rel_count}) as node_rel_counts, max(rel_count) as max_rel_count UNWIND node_rel_counts as node_rel_count WHERE node_rel_count.count = max_rel_count RETURN node_rel_count.node

User: Update all Person nodes to set 'active' to true
Tool: write_neo4j_cypher
Query: MATCH (n:Person) SET n.active = true

User: Delete all nodes with label Temp
Tool: write_neo4j_cypher
Query: MATCH (n:Temp) DETACH DELETE n

User: What relationships exist between Employee and Department?
Tool: get_neo4j_schema

User: I want to change Bob's email
Tool: write_neo4j_cypher
Query: MATCH (n:Person {name: 'Bob'}) SET n.email = 'new@example.com'

User: What properties does a Project node have?
Tool: get_neo4j_schema

User: Remove the "retired" property from all Employee nodes
Tool: write_neo4j_cypher
Query: MATCH (e:Employee) REMOVE e.retired

ERROR CASES:
- If the query seems ambiguous or unsafe, clarify or refuse with an explanation.
- NEVER run write queries using read_neo4j_cypher.

ALLOWED TOOLS: Only use these exact tool names:
- read_neo4j_cypher
- write_neo4j_cypher
- get_neo4j_schema

Never invent, abbreviate, or use other names.
If unsure, ask a clarifying question.

ALWAYS explain your choice of tool before outputting the tool and Cypher.
REMEMBER: Put the query on a single line after "Query: " without code blocks.
"""

try:
    from config import CORTEX_CONFIG, SERVER_CONFIG, DEBUG_CONFIG
    API_URL = CORTEX_CONFIG["api_url"]
    API_KEY = CORTEX_CONFIG["api_key"]
    MODEL = CORTEX_CONFIG["model"]
    APP_CODE = CORTEX_CONFIG["app_code"]
    APP_ID = CORTEX_CONFIG["app_id"]
    MCP_PORT = SERVER_CONFIG["mcp_port"]
    PRINT_DEBUG = DEBUG_CONFIG["print_llm_output"]
    PRINT_QUERIES = DEBUG_CONFIG["print_queries"]
except ImportError:
    # Fallback to hardcoded values if config.py is not available
    API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
    API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
    MODEL = "llama3.1-70b"
    APP_CODE = "edagnai"
    APP_ID = "edadip"
    MCP_PORT = 8000
    PRINT_DEBUG = True
    PRINT_QUERIES = True

def cortex_llm(prompt: str, session_id: str) -> str:
    headers = {
        "Authorization": f'Snowflake Token="{API_KEY}"',
        "Content-Type": "application/json"
    }
    payload = {
        "query": {
            "aplctn_cd": APP_CODE,
            "app_id": APP_ID,
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
    allowed_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    trace = llm_output.strip()
    tool = None
    query = None
    
    # Find tool
    tool_match = re.search(r"Tool:\s*([\w_]+)", llm_output, re.I)
    if tool_match:
        tname = tool_match.group(1).strip()
        if tname in allowed_tools:
            tool = tname
    
    # Find query - handle both single line and multi-line queries
    query_match = re.search(r"Query:\s*(.*?)(?=\n\n|\n[A-Z]|\Z)", llm_output, re.I | re.DOTALL)
    if query_match:
        query_text = query_match.group(1).strip()
        
        # Handle code blocks
        if "```" in query_text:
            # Extract content between code blocks
            code_block_match = re.search(r'```(?:cypher|sql)?\s*\n?(.*?)\n?```', query_text, re.DOTALL | re.IGNORECASE)
            if code_block_match:
                query = code_block_match.group(1).strip()
            else:
                # If no proper code block, remove all ``` markers
                query = re.sub(r'```[^\n]*', '', query_text).strip()
        else:
            query = query_text
        
        # Clean up the query
        if query:
            query = clean_cypher_query(query)
    
    return tool, query, trace

def select_tool_node(state: AgentState) -> dict:
    llm_output = cortex_llm(state.question, state.session_id)
    tool, query, trace = parse_llm_output(llm_output)
    
    # Debug logging
    if PRINT_DEBUG:
        print(f"LLM Output: {llm_output}")
        print(f"Parsed Tool: {tool}")
        print(f"Parsed Query: {query}")
    
    return {
        "question": state.question,
        "session_id": state.session_id,
        "tool": tool or "",
        "query": query or "",
        "trace": trace or "",
        "answer": ""
    }

def execute_tool_node(state: AgentState) -> dict:
    tool = state.tool
    query = state.query
    trace = state.trace
    answer = ""
    valid_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    
    try:
        if not tool:
            answer = (
                "⚠️ The agent did not choose a valid tool (recognized: read_neo4j_cypher, write_neo4j_cypher, get_neo4j_schema). "
                "Please rephrase your question to be about graph data, updates, or schema."
            )
        elif tool not in valid_tools:
            answer = f"⚠️ MCP tool not recognized: {tool}. Only these are allowed: {', '.join(valid_tools)}"
        elif tool == "read_neo4j_cypher" and query and query.strip().lower() == "return db.name() as name":
            answer = "Your Neo4j does not support querying the database name via Cypher. Check your connection settings."
        elif tool == "get_neo4j_schema":
            result = requests.post(f"http://localhost:{MCP_PORT}/get_neo4j_schema", headers=headers)
            answer = result.json() if result.ok else result.text
        elif tool == "read_neo4j_cypher":
            if not query or not query.strip():
                answer = "⚠️ Sorry, I could not generate a valid Cypher query for your question. Please try to rephrase or clarify."
            else:
                query_clean = clean_cypher_query(query)
                if PRINT_QUERIES:
                    print(f"Executing query: {query_clean}")  # Debug logging
                
                node_count_query = (
                    query_clean.lower() == "match (n) return count(n)"
                    or query_clean.lower() == "match (n) return count(n) as node_count"
                )
                
                data = {"query": query_clean, "params": {}}
                result = requests.post(f"http://localhost:{MCP_PORT}/read_neo4j_cypher", json=data, headers=headers)
                answer = result.json() if result.ok else result.text
        elif tool == "write_neo4j_cypher":
            if not query or not query.strip():
                answer = "⚠️ Sorry, I could not generate a valid Cypher query for your action. Please try to rephrase or clarify."
            else:
                query_clean = clean_cypher_query(query)
                if PRINT_QUERIES:
                    print(f"Executing write query: {query_clean}")  # Debug logging
                
                data = {"query": query_clean, "params": {}}
                result = requests.post(f"http://localhost:{MCP_PORT}/write_neo4j_cypher", json=data, headers=headers)
                answer = result.json() if result.ok else result.text
        else:
            answer = f"Unknown tool: {tool}"
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

def build_agent():
    workflow = StateGraph(state_schema=AgentState)
    workflow.add_node("select_tool", RunnableLambda(select_tool_node))
    workflow.add_node("execute_tool", RunnableLambda(execute_tool_node))
    workflow.set_entry_point("select_tool")
    workflow.add_edge("select_tool", "execute_tool")
    workflow.add_edge("execute_tool", END)
    return workflow.compile()
