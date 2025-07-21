import requests
import urllib3
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import re
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langgraph_agent")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class AgentState(BaseModel):
    question: str
    session_id: str
    tool: str = ""
    query: str = ""
    trace: str = ""
    answer: str = ""
    graph_data: dict = None

def clean_cypher_query(query: str) -> str:
    """Clean and format Cypher queries for execution"""
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

def extract_graph_data_from_response(response_text):
    """Extract graph data from response text if present"""
    try:
        # Look for embedded graph data
        graph_match = re.search(r'GRAPH_VIZ_DATA:(\{.*?\})', response_text)
        if graph_match:
            graph_data = json.loads(graph_match.group(1))
            # Remove the embedded data from the response text
            cleaned_response = response_text.replace(f"GRAPH_VIZ_DATA:{graph_match.group(1)}", "").strip()
            return cleaned_response, graph_data
    except Exception as e:
        logger.warning(f"Could not extract graph data: {e}")
    
    return response_text, None

def format_response_with_graph(result_data, tool_type):
    """Format the response based on tool type and result structure, including graph data"""
    try:
        if isinstance(result_data, str):
            try:
                result_data = json.loads(result_data)
            except:
                return str(result_data), None
        
        graph_data = None
        
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
                    graph_data = result_data["graph_data"]
                    node_count = len(graph_data.get('nodes', []))
                    rel_count = len(graph_data.get('relationships', []))
                    if node_count > 0 or rel_count > 0:
                        formatted_response += f"\n\n🕸️ **Graph Visualization Available:** {node_count} nodes, {rel_count} relationships"
                
                return formatted_response, graph_data
        
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
                    formatted_response += f"\n\n🕸️ **Interactive Graph Available:** {node_count} nodes, {rel_count} relationships"
                    
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
                
                return formatted_response, graph_data
        
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
{json.dumps(schema, indent=2)[:1500]}{'...' if len(str(schema)) > 1500 else ''}
```
                """.strip()
                return formatted_response, None
        
        # Fallback for other formats
        formatted_text = json.dumps(result_data, indent=2) if isinstance(result_data, (dict, list)) else str(result_data)
        return formatted_text, None
    
    except Exception as e:
        error_msg = f"Error formatting response: {str(e)}\nRaw data: {str(result_data)}"
        logger.error(error_msg)
        return error_msg, None

# Enhanced system message with visualization guidance
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
    - Include RETURN clauses to see created/modified nodes in visualization.

- get_neo4j_schema:
    - Use when the user asks about structure, schema, labels, relationship types, available node kinds, or properties.
    - Returns a detailed schema graph, including node labels, relationship types, and property keys.

VISUALIZATION OPTIMIZATION:
- When users want to "see", "show", "display", or "visualize" data, prefer queries that return actual graph objects (nodes and relationships).
- Use queries like: MATCH (n:Person) RETURN n, MATCH (a)-[r]->(b) RETURN a, r, b
- Add RETURN clauses to CREATE/MERGE operations when users want to see results.
- Use reasonable LIMIT clauses (10-50) for visualization queries.

IMPORTANT GUIDELINES:
- ALWAYS output your reasoning and then the tool and Cypher query (if any).
- For graph visualization, return complete nodes and relationships, not just properties.
- If the user requests counts and the result is unexpectedly low, try admin-level count:
    CALL db.stats.retrieve('GRAPH COUNTS') YIELD data RETURN data['NodeCount'] AS node_count
- If the user asks for schema, always use get_neo4j_schema.
- For ambiguous requests, ask clarifying questions or choose the safest tool.
- ALWAYS include proper whitespace between Cypher clauses.

FEW-SHOT EXAMPLES:

User: How many nodes are in the graph?
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN count(n) AS node_count

User: Show me all Person nodes
Tool: read_neo4j_cypher
Query: MATCH (n:Person) RETURN n LIMIT 20

User: Display relationships between people
Tool: read_neo4j_cypher
Query: MATCH (a:Person)-[r]->(b:Person) RETURN a, r, b LIMIT 25

User: Show me the network structure
Tool: read_neo4j_cypher
Query: MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 30

User: Visualize employees and departments
Tool: read_neo4j_cypher  
Query: MATCH (e:Employee)-[r:WORKS_IN]->(d:Department) RETURN e, r, d

User: Create a Person named Alice and show it
Tool: write_neo4j_cypher
Query: CREATE (n:Person {name: 'Alice', created_at: datetime()}) RETURN n

User: Connect Alice to Bob and visualize
Tool: write_neo4j_cypher
Query: MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[r:KNOWS {created_at: datetime()}]->(b) RETURN a, r, b

User: Show the database schema
Tool: get_neo4j_schema

User: What are the node types in my graph?
Tool: get_neo4j_schema

ERROR PREVENTION:
- NEVER run write queries using read_neo4j_cypher.
- NEVER use read_neo4j_cypher for CREATE, UPDATE, DELETE operations.
- Always validate query safety before execution.

ALLOWED TOOLS: read_neo4j_cypher, write_neo4j_cypher, get_neo4j_schema

ALWAYS explain your reasoning before selecting the tool and generating the query.
"""

# Cortex LLM configuration
API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
MODEL = "llama3.1-70b"

def cortex_llm(prompt: str, session_id: str) -> str:
    """Call the Cortex LLM API"""
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
    
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, verify=False, timeout=30)
        resp.raise_for_status()
        return resp.text.partition("end_of_stream")[0].strip()
    except Exception as e:
        logger.error(f"Cortex LLM API error: {e}")
        return f"Error calling Cortex LLM: {str(e)}"

def parse_llm_output(llm_output):
    """Parse LLM output to extract tool and query"""
    allowed_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    trace = llm_output.strip()
    tool = None
    query = None
    
    # Extract tool
    tool_match = re.search(r"Tool:\s*([\w_]+)", llm_output, re.I)
    if tool_match:
        tname = tool_match.group(1).strip()
        if tname in allowed_tools:
            tool = tname
    
    # Extract query
    query_match = re.search(r"Query:\s*(.+?)(?:\n|$)", llm_output, re.I | re.DOTALL)
    if query_match:
        query = query_match.group(1).strip()
    
    return tool, query, trace

def select_tool_node(state: AgentState) -> dict:
    """Node to select tool and generate query using LLM"""
    logger.info(f"Processing question: {state.question}")
    
    try:
        llm_output = cortex_llm(state.question, state.session_id)
        tool, query, trace = parse_llm_output(llm_output)
        
        logger.info(f"LLM selected tool: {tool}, query: {query[:100] if query else 'None'}")
        
        return {
            "question": state.question,
            "session_id": state.session_id,
            "tool": tool or "",
            "query": query or "",
            "trace": trace or "",
            "answer": "",
            "graph_data": None
        }
    except Exception as e:
        logger.error(f"Error in select_tool_node: {e}")
        return {
            "question": state.question,
            "session_id": state.session_id,
            "tool": "",
            "query": "",
            "trace": f"Error selecting tool: {str(e)}",
            "answer": f"❌ Error processing question: {str(e)}",
            "graph_data": None
        }

def execute_tool_node(state: AgentState) -> dict:
    """Node to execute the selected tool"""
    tool = state.tool
    query = state.query
    trace = state.trace
    answer = ""
    graph_data = None
    
    valid_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    
    logger.info(f"Executing tool: {tool}")
    
    try:
        if not tool:
            answer = (
                "⚠️ The agent did not choose a valid tool (recognized: read_neo4j_cypher, write_neo4j_cypher, get_neo4j_schema). "
                "Please rephrase your question to be about graph data, updates, or schema."
            )
        elif tool not in valid_tools:
            answer = f"⚠️ MCP tool not recognized: {tool}. Only these are allowed: {', '.join(valid_tools)}"
        elif tool == "get_neo4j_schema":
            result = requests.post("http://localhost:8000/get_neo4j_schema", headers=headers, timeout=30)
            if result.ok:
                answer, graph_data = format_response_with_graph(result.json(), tool)
            else:
                answer = f"❌ Schema query failed: {result.text}"
        elif tool == "read_neo4j_cypher":
            if not query or not query.strip():
                answer = "⚠️ Sorry, I could not generate a valid Cypher query for your question. Please try to rephrase or clarify."
            else:
                query_clean = clean_cypher_query(query)
                data = {"query": query_clean, "params": {}}
                result = requests.post("http://localhost:8000/read_neo4j_cypher", json=data, headers=headers, timeout=30)
                if result.ok:
                    answer, graph_data = format_response_with_graph(result.json(), tool)
                else:
                    answer = f"❌ Read query failed: {result.text}"
        elif tool == "write_neo4j_cypher":
            if not query or not query.strip():
                answer = "⚠️ Sorry, I could not generate a valid Cypher query for your action. Please try to rephrase or clarify."
            else:
                query_clean = clean_cypher_query(query)
                data = {"query": query_clean, "params": {}}
                result = requests.post("http://localhost:8000/write_neo4j_cypher", json=data, headers=headers, timeout=30)
                if result.ok:
                    answer, graph_data = format_response_with_graph(result.json(), tool)
                else:
                    answer = f"❌ Write query failed: {result.text}"
        else:
            answer = f"Unknown tool: {tool}"
    
    except requests.exceptions.Timeout:
        answer = "⚠️ Request timed out. The Neo4j server might be busy or unavailable."
    except requests.exceptions.ConnectionError:
        answer = "⚠️ Cannot connect to MCP server. Make sure it's running on port 8000."
    except Exception as e:
        logger.error(f"Error in execute_tool_node: {e}")
        answer = f"⚠️ Tool execution failed: {str(e)}"
    
    logger.info(f"Tool execution completed. Answer length: {len(answer)}")
    
    return {
        "question": state.question,
        "session_id": state.session_id,
        "tool": tool,
        "query": query,
        "trace": trace,
        "answer": answer,
        "graph_data": graph_data
    }

def build_agent():
    """Build and return the LangGraph agent"""
    workflow = StateGraph(state_schema=AgentState)
    
    # Add nodes
    workflow.add_node("select_tool", RunnableLambda(select_tool_node))
    workflow.add_node("execute_tool", RunnableLambda(execute_tool_node))
    
    # Set entry point
    workflow.set_entry_point("select_tool")
    
    # Add edges
    workflow.add_edge("select_tool", "execute_tool")
    workflow.add_edge("execute_tool", END)
    
    # Compile and return
    agent = workflow.compile()
    logger.info("LangGraph agent built successfully")
    return agent

# For testing purposes
if __name__ == "__main__":
    # Test the agent locally
    agent = build_agent()
    test_state = AgentState(
        question="Show me all Person nodes",
        session_id="test_session"
    )
    
    import asyncio
    
    async def test():
        result = await agent.ainvoke(test_state)
        print("Test Result:", result)
    
    # asyncio.run(test())
