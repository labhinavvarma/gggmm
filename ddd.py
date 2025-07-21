import requests
import urllib3
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import re
import json

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class AgentState(BaseModel):
    question: str
    session_id: str
    tool: str = ""
    query: str = ""
    trace: str = ""
    answer: str = ""

def clean_cypher_query(query: str) -> str:
    query = re.sub(r'[\r\n]+', ' ', query)
    keywords = [
        "MATCH", "WITH", "RETURN", "ORDER BY", "UNWIND", "WHERE", "LIMIT",
        "SKIP", "CALL", "YIELD", "CREATE", "MERGE", "SET", "DELETE", "DETACH DELETE", "REMOVE"
    ]
    for kw in keywords:
        query = re.sub(rf'(?<!\s)({kw})', r' \1', query)
        query = re.sub(rf'({kw})([^\s\(])', r'\1 \2', query)
    query = re.sub(r'\s+', ' ', query)
    return query.strip()

def format_response_with_graph(result_data, tool_type):
    """Format the response based on tool type and result structure, including graph data"""
    try:
        if isinstance(result_data, str):
            try:
                result_data = json.loads(result_data)
            except:
                return str(result_data)
        
        if tool_type == "write_neo4j_cypher" and isinstance(result_data, dict):
            if "change_info" in result_data:
                change_info = result_data["change_info"]
                formatted_response = f"""
🔄 **Neo4j Write Operation Completed**

{change_info['summary']}

**Changes Made:**
{chr(10).join(f"• {change}" for change in change_info['changes'])}

**Query Executed:** `{change_info['query']}`
**Execution Time:** {change_info['execution_time_ms']}ms
**Timestamp:** {change_info['timestamp']}
                """.strip()
                
                # Include graph data if available
                if result_data.get("graph_data"):
                    formatted_response += f"\n\n**Graph Data Available:** {len(result_data['graph_data'].get('nodes', []))} nodes, {len(result_data['graph_data'].get('relationships', []))} relationships"
                    # Embed graph data for visualization
                    formatted_response += f"\n\nGRAPH_VIZ_DATA:{json.dumps(result_data['graph_data'])}"
                
                return formatted_response
        
        elif tool_type == "read_neo4j_cypher" and isinstance(result_data, dict):
            if "data" in result_data and "metadata" in result_data:
                data = result_data["data"]
                metadata = result_data["metadata"]
                graph_data = result_data.get("graph_data")
                
                # Format the data nicely
                if isinstance(data, list) and len(data) > 0:
                    if len(data) <= 5:  # Show all results for small datasets
                        formatted_data = json.dumps(data, indent=2)
                    else:  # Show first few and count for large datasets
                        formatted_data = f"{json.dumps(data[:3], indent=2)}\n... and {len(data) - 3} more records"
                else:
                    formatted_data = "No records found"
                
                formatted_response = f"""
📊 **Query Results**

**Records Found:** {metadata['record_count']}
**Execution Time:** {metadata['execution_time_ms']}ms
**Timestamp:** {metadata['timestamp']}

**Data:**
```json
{formatted_data}
```
                """.strip()
                
                # Add graph visualization info if available
                if graph_data and graph_data.get('nodes'):
                    node_count = len(graph_data['nodes'])
                    rel_count = len(graph_data.get('relationships', []))
                    formatted_response += f"\n\n🕸️ **Graph Visualization Available:** {node_count} nodes, {rel_count} relationships"
                    
                    # Show node labels summary
                    if node_count > 0:
                        label_counts = {}
                        for node in graph_data['nodes']:
                            for label in node.get('labels', ['Unknown']):
                                label_counts[label] = label_counts.get(label, 0) + 1
                        
                        label_summary = ", ".join([f"{label}: {count}" for label, count in label_counts.items()])
                        formatted_response += f"\n**Node Types:** {label_summary}"
                    
                    # Show relationship types summary
                    if rel_count > 0:
                        rel_types = {}
                        for rel in graph_data.get('relationships', []):
                            rel_type = rel.get('type', 'Unknown')
                            rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
                        
                        rel_summary = ", ".join([f"{rel_type}: {count}" for rel_type, count in rel_types.items()])
                        formatted_response += f"\n**Relationship Types:** {rel_summary}"
                    
                    # Embed graph data for visualization (this will be picked up by the UI)
                    formatted_response += f"\n\nGRAPH_VIZ_DATA:{json.dumps(graph_data)}"
                
                return formatted_response
        
        elif tool_type == "get_neo4j_schema" and isinstance(result_data, dict):
            if "schema" in result_data:
                schema = result_data["schema"]
                metadata = result_data.get("metadata", {})
                
                # Format schema information nicely
                schema_summary = []
                if isinstance(schema, dict):
                    for label, info in schema.items():
                        if isinstance(info, dict):
                            props = info.get('properties', {})
                            relationships = info.get('relationships', {})
                            schema_summary.append(f"**{label}**: {len(props)} properties, {len(relationships)} relationship types")
                
                formatted_response = f"""
🏗️ **Neo4j Schema Information**

**Execution Time:** {metadata.get('execution_time_ms', 'N/A')}ms
**Timestamp:** {metadata.get('timestamp', 'N/A')}

**Schema Summary:**
{chr(10).join(f"• {item}" for item in schema_summary)}

**Full Schema:**
```json
{json.dumps(schema, indent=2)[:1000]}{'...' if len(str(schema)) > 1000 else ''}
```
                """.strip()
                return formatted_response
        
        # Fallback for other formats
        return json.dumps(result_data, indent=2) if isinstance(result_data, (dict, list)) else str(result_data)
    
    except Exception as e:
        return f"Error formatting response: {str(e)}\nRaw data: {str(result_data)}"

SYS_MSG = """
You are an expert AI assistant that helps users query and manage a Neo4j database by selecting and using one of three MCP tools. Choose the most appropriate tool and generate the correct Cypher query or action.

TOOL DESCRIPTIONS:
- read_neo4j_cypher:
    - Use for all read-only graph queries: exploring data, finding nodes/relationships, aggregation, reporting, analysis, counting.
    - Only run safe queries (MATCH, RETURN, WHERE, OPTIONAL MATCH, etc).
    - NEVER use this tool for CREATE, UPDATE, DELETE, SET, or any modification.
    - Returns a list of matching nodes, relationships, or computed values.
    - Can return graph data for visualization when nodes/relationships are queried.

- write_neo4j_cypher:
    - Use ONLY for write queries: CREATE, MERGE, SET, DELETE, REMOVE, or modifying properties or structure.
    - Use to create nodes/edges, update, or delete data.
    - NEVER use this for data retrieval only.
    - Returns detailed change information with timestamps.
    - May return graph data if the query includes RETURN clauses.

- get_neo4j_schema:
    - Use when the user asks about structure, schema, labels, relationship types, available node kinds, or properties.
    - Returns a detailed schema graph, including node labels, relationship types, and property keys.

IMPORTANT GUIDELINES:
- ALWAYS output your reasoning and then the tool and Cypher query (if any).
- When users want to "see" or "show" or "display" nodes/relationships, prefer queries that return the actual graph objects for visualization.
- For visualization purposes, use queries like: MATCH (n:Person) RETURN n, MATCH (a)-[r]->(b) RETURN a, r, b
- If the user requests the number of nodes and the result is unexpectedly low, try the admin-level count as a fallback:
    CALL db.stats.retrieve('GRAPH COUNTS') YIELD data RETURN data['NodeCount'] AS node_count
- If the user asks for schema, always use get_neo4j_schema.
- For ambiguous requests, ask clarifying questions or choose the safest tool.
- ALWAYS include all required whitespace and line breaks between Cypher clauses.
- Write operations will show detailed change information including timestamps.
- Graph visualization will be available for queries that return nodes and relationships.

FEW-SHOT EXAMPLES:

User: How many nodes are in the graph?
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN count(n)

User: Show me all Person nodes
Tool: read_neo4j_cypher
Query: MATCH (n:Person) RETURN n

User: Display the relationships between people
Tool: read_neo4j_cypher
Query: MATCH (a:Person)-[r]->(b:Person) RETURN a, r, b

User: Show me the network of connections
Tool: read_neo4j_cypher
Query: MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 50

User: Visualize all employees and their departments
Tool: read_neo4j_cypher  
Query: MATCH (e:Employee)-[r:WORKS_IN]->(d:Department) RETURN e, r, d

User: Show the schema of the database
Tool: get_neo4j_schema

User: Create a Person node named Alice
Tool: write_neo4j_cypher
Query: CREATE (:Person {name: 'Alice'}) RETURN n

User: Connect Alice to Bob with a KNOWS relationship
Tool: write_neo4j_cypher
Query: MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[r:KNOWS]->(b) RETURN a, r, b

User: Update all Person nodes to set 'active' to true
Tool: write_neo4j_cypher
Query: MATCH (n:Person) SET n.active = true

User: Delete all nodes with label Temp
Tool: write_neo4j_cypher
Query: MATCH (n:Temp) DETACH DELETE n

User: What relationships exist between Employee and Department?
Tool: get_neo4j_schema

User: I want to see the graph structure
Tool: read_neo4j_cypher
Query: MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25

ERROR CASES:
- If the query seems ambiguous or unsafe, clarify or refuse with an explanation.
- NEVER run write queries using read_neo4j_cypher.

VISUALIZATION PREFERENCE:
- When users ask to "show", "display", "visualize", or "see" data, prefer queries that return graph objects (nodes and relationships) rather than just properties.
- Add RETURN clauses to CREATE/MERGE statements when users want to see the results.
- Use reasonable LIMIT clauses (10-50) for visualization queries to prevent overwhelming displays.

ALLOWED TOOLS: Only use these exact tool names:
- read_neo4j_cypher
- write_neo4j_cypher
- get_neo4j_schema

Never invent, abbreviate, or use other names.
If unsure, ask a clarifying question.

ALWAYS explain your choice of tool before outputting the tool and Cypher.
"""

API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
MODEL = "llama3.1-70b"

def cortex_llm(prompt: str, session_id: str) -> str:
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
    allowed_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    trace = llm_output.strip()
    tool = None
    query = None
    tool_match = re.search(r"Tool: ([\w_]+)", llm_output, re.I)
    if tool_match:
        tname = tool_match.group(1).strip()
        if tname in allowed_tools:
            tool = tname
        else:
            tool = None
    query_match = re.search(r"Query: (.+)", llm_output, re.I)
    if query_match:
        query = query_match.group(1).strip()
    return tool, query, trace

def select_tool_node(state: AgentState) -> dict:
    llm_output = cortex_llm(state.question, state.session_id)
    tool, query, trace = parse_llm_output(llm_output)
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
        elif tool == "read_neo4j_cypher" and query.strip().lower() == "return db.name() as name":
            answer = "Your Neo4j does not support querying the database name via Cypher. Check your connection settings."
        elif tool == "get_neo4j_schema":
            result = requests.post("http://localhost:8000/get_neo4j_schema", headers=headers)
            if result.ok:
                answer = format_response_with_graph(result.json(), tool)
            else:
                answer = f"❌ Schema query failed: {result.text}"
        elif tool == "read_neo4j_cypher":
            if not query or not query.strip():
                answer = "⚠️ Sorry, I could not generate a valid Cypher query for your question. Please try to rephrase or clarify."
            else:
                query_clean = clean_cypher_query(query)
                data = {"query": query_clean, "params": {}}
                result = requests.post("http://localhost:8000/read_neo4j_cypher", json=data, headers=headers)
                if result.ok:
                    answer = format_response_with_graph(result.json(), tool)
                else:
                    answer = f"❌ Read query failed: {result.text}"
        elif tool == "write_neo4j_cypher":
            if not query or not query.strip():
                answer = "⚠️ Sorry, I could not generate a valid Cypher query for your action. Please try to rephrase or clarify."
            else:
                query_clean = clean_cypher_query(query)
                data = {"query": query_clean, "params": {}}
                result = requests.post("http://localhost:8000/write_neo4j_cypher", json=data, headers=headers)
                if result.ok:
                    answer = format_response_with_graph(result.json(), tool)
                else:
                    answer = f"❌ Write query failed: {result.text}"
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
