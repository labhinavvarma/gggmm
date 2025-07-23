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
    node_limit: int = 1000

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

def optimize_query_for_visualization(query: str, node_limit: int = 1000) -> str:
    """Optimize queries for better Pyvis visualization performance"""
    query = query.strip()
    
    # Add reasonable limits to MATCH queries that don't have them
    if ("MATCH" in query.upper() and 
        "LIMIT" not in query.upper() and 
        "count(" not in query.lower() and
        "COUNT(" not in query):
        
        # For Pyvis visualization, use smaller limits for cleaner display
        if "RETURN" in query.upper():
            # Use smaller limits for better visualization
            limit = min(node_limit, 200) if node_limit > 200 else node_limit
            query += f" LIMIT {limit}"
    
    return query

def format_response_with_graph(result_data, tool_type, node_limit=5000):
    """Format the response for split-screen display"""
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
🔄 **Database Update Completed**

**⚡ Execution:** {change_info['execution_time_ms']}ms  
**🕐 Time:** {change_info['timestamp'][:19]}

**📝 Changes Made:**
{chr(10).join(f"{change}" for change in change_info['changes'])}

**🔧 Query:** `{change_info['query']}`
                """.strip()
                
                # Include graph data if available
                if result_data.get("graph_data"):
                    graph_data = result_data["graph_data"]
                    node_count = len(graph_data.get('nodes', []))
                    rel_count = len(graph_data.get('relationships', []))
                    if node_count > 0 or rel_count > 0:
                        formatted_response += f"\n\n🕸️ **Updated graph visualization** with {node_count} nodes and {rel_count} relationships"
                
                return formatted_response, graph_data
        
        elif tool_type == "read_neo4j_cypher" and isinstance(result_data, dict):
            if "data" in result_data and "metadata" in result_data:
                data = result_data["data"]
                metadata = result_data["metadata"]
                graph_data = result_data.get("graph_data")
                
                # Format response for split screen
                formatted_response = f"""
📊 **Query Results**

**🔢 Records:** {metadata['record_count']}  
**⚡ Time:** {metadata['execution_time_ms']}ms  
**🕐 Timestamp:** {metadata['timestamp'][:19]}
                """.strip()
                
                # Add data summary for non-graph queries
                if not graph_data or not graph_data.get('nodes'):
                    if isinstance(data, list) and len(data) > 0:
                        if len(data) <= 3:
                            formatted_response += f"\n\n**📋 Data:**\n```json\n{json.dumps(data, indent=2)}\n```"
                        else:
                            formatted_response += f"\n\n**📋 Sample Data:**\n```json\n{json.dumps(data[:2], indent=2)}\n... and {len(data) - 2} more records\n```"
                    else:
                        formatted_response += "\n\n**📋 Data:** No records found"
                
                # Add graph visualization info if available
                if graph_data and graph_data.get('nodes'):
                    node_count = len(graph_data['nodes'])
                    rel_count = len(graph_data.get('relationships', []))
                    
                    formatted_response += f"\n\n🕸️ **Graph visualization updated** with {node_count} nodes and {rel_count} relationships"
                    
                    # Show node types summary
                    if node_count > 0:
                        label_counts = {}
                        for node in graph_data['nodes']:
                            for label in node.get('labels', ['Unknown']):
                                label_counts[label] = label_counts.get(label, 0) + 1
                        
                        if len(label_counts) > 0:
                            label_summary = ", ".join([f"{label}({count})" for label, count in sorted(label_counts.items())])
                            formatted_response += f"\n**🏷️ Node Types:** {label_summary}"
                    
                    # Show if limited
                    if graph_data.get('limited'):
                        formatted_response += f"\n**⚠️ Display limited to {node_limit} nodes for performance**"
                
                return formatted_response, graph_data
        
        elif tool_type == "get_neo4j_schema" and isinstance(result_data, dict):
            if "schema" in result_data:
                schema = result_data["schema"]
                metadata = result_data.get("metadata", {})
                
                # Format schema information for split screen
                schema_summary = []
                if isinstance(schema, dict):
                    for label, info in schema.items():
                        if isinstance(info, dict):
                            props = info.get('properties', {})
                            relationships = info.get('relationships', {})
                            schema_summary.append(f"**{label}**: {len(props)} props, {len(relationships)} rels")
                
                formatted_response = f"""
🏗️ **Database Schema**

**⚡ Time:** {metadata.get('execution_time_ms', 'N/A')}ms

**📊 Overview:**
{chr(10).join(f"{item}" for item in schema_summary[:10])}
{f"... and {len(schema_summary) - 10} more types" if len(schema_summary) > 10 else ""}
                """.strip()
                
                return formatted_response, None
        
        # Fallback for other formats
        formatted_text = json.dumps(result_data, indent=2) if isinstance(result_data, (dict, list)) else str(result_data)
        return formatted_text, None
    
    except Exception as e:
        error_msg = f"❌ **Error formatting response:** {str(e)}"
        logger.error(error_msg)
        return error_msg, None

# Enhanced system message optimized for split-screen visualization
SYS_MSG = """
You are an expert AI assistant that helps users query and manage a Neo4j database by selecting and using one of three MCP tools. Choose the most appropriate tool and generate the correct Cypher query or action, following all instructions and best practices below.

TOOL DESCRIPTIONS:
- read_neo4j_cypher:
    - Use for all read-only graph queries: exploring data, finding nodes/relationships, aggregation, reporting, analysis, counting.
    - Only run safe queries (MATCH, RETURN, WHERE, OPTIONAL MATCH, etc).
    - Never use this tool for CREATE, UPDATE, DELETE, SET, or any modification.
    - Returns a list of matching nodes, relationships, or computed values.
    - Can return graph data for visualization when nodes/relationships are queried.

- write_neo4j_cypher:
    - Use only for write queries: CREATE, MERGE, SET, DELETE, REMOVE, or modifying properties or structure.
    - Use to create nodes/edges, update, or delete data.
    - Never use this for data retrieval only.
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
- Always output your reasoning and then the tool and Cypher query (if any).
- For graph visualization, return complete nodes and relationships, not just properties.
- If the user requests counts and the result is unexpectedly low, try admin-level count:
    CALL db.stats.retrieve('GRAPH COUNTS') YIELD data RETURN data['NodeCount'] AS node_count
- If the user asks for schema, always use get_neo4j_schema.
- For ambiguous requests, ask clarifying questions or choose the safest tool.
- Always include proper whitespace between Cypher clauses.

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
- Never run write queries using read_neo4j_cypher.
- Never use read_neo4j_cypher for CREATE, UPDATE, DELETE operations.
- Always validate query safety before execution.

ALLOWED TOOLS: read_neo4j_cypher, write_neo4j_cypher, get_neo4j_schema

ALWAYS explain your reasoning before selecting the tool and generating the query.

ADDITIONAL CYHER GENERATION GUIDELINES (for accuracy):

- Use only node labels, relationship types, and properties that are present in the schema provided.
- Do not invent or assume labels, properties, or relationship types.
- If the user request is ambiguous or refers to unknown elements, ask for clarification instead of guessing.
- Output only the Cypher query. Do not add comments, explanations, or any text outside the Cypher query itself.
- Use double quotes for all string values.
- Follow correct capitalization for properties, labels, and relationships as they appear in the schema.
- Always use explicit and precise patterns for matching, filtering, and returning results.
- Do not generate queries for graph visualizations. Only provide Cypher database queries unless a visualization is specifically requested.

Cypher Query Guidelines:
- Use MATCH to specify node and relationship patterns.
- Use WHERE for property-based filtering with operators such as =, >, <, <>, CONTAINS, STARTS WITH, and ENDS WITH.
- Use RETURN to specify explicit fields. Do not use RETURN * unless the schema is very small and well known.
- Use ORDER BY, LIMIT, and DISTINCT as required by the user's request.
- Use aggregation functions like COUNT, SUM, AVG, MIN, MAX for summary queries.
- Use OPTIONAL MATCH for optional relationships.
- For multi-pattern queries, separate patterns with commas in the MATCH clause.
- When the user requests multiple conditions, combine them logically in the WHERE clause.
- If the user’s intent is not fully clear, ask a clarifying question instead of generating a query.

Examples of correct Cypher queries:

Example 1: List all people in the database.
MATCH (p:Person) RETURN p

Example 2: Find all movies released after 2015.
MATCH (m:Movie) WHERE m.released > 2015 RETURN m

Example 3: Get all actors who acted in "Inception".
MATCH (a:Person)-[:ACTED_IN]->(m:Movie {title: "Inception"}) RETURN a

Example 4: Show the top 5 movies by box office revenue.
MATCH (m:Movie) RETURN m ORDER BY m.boxOffice DESC LIMIT 5

Example 5: Find unique cities where people live.
MATCH (p:Person)-[:LIVES_IN]->(c:City) RETURN DISTINCT c.name

Example 6: Find persons named John older than 40.
MATCH (p:Person) WHERE p.name = "John" AND p.age > 40 RETURN p

Example 7: Find people who acted and directed the same movie.
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(p) RETURN p, m

Example 8: List all persons and movies they may have directed (including those who have not directed any movie).
MATCH (p:Person) OPTIONAL MATCH (p)-[:DIRECTED]->(m:Movie) RETURN p, m

Examples of incorrect Cypher queries and why they are incorrect:

Incorrect Example 1: MATCH (a:Dog) RETURN a
Reason: "Dog" is not a node label in the schema.

Incorrect Example 2: MATCH (p:Person) WHERE p.fullname = "Alice" RETURN p
Reason: "fullname" is not a property in the schema.

Incorrect Example 3: MATCH (n) RETURN *
Reason: Using RETURN * is not recommended unless the schema is very small and known.

Incorrect Example 4: MATCH (a)-[:FRIEND]->(b) RETURN a, b
Reason: "FRIEND" is not a relationship type in the schema.

How to handle unclear or unsupported requests:
- If the user's request does not provide enough information, respond with: Unable to generate Cypher: Insufficient information.
- If the user requests a label, relationship, or property not present in the schema, respond with: Unable to generate Cypher: Entity or property not in schema.

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
    
    # Extract query - handle multi-line queries better
    query_match = re.search(r"Query:\s*(.+?)(?:\n\n|\n[A-Z]|$)", llm_output, re.I | re.DOTALL)
    if query_match:
        query = query_match.group(1).strip()
    
    return tool, query, trace

def select_tool_node(state: AgentState) -> dict:
    """Node to select tool and generate query using LLM"""
    logger.info(f"Processing question: {state.question}")
    
    try:
        llm_output = cortex_llm(state.question, state.session_id)
        tool, query, trace = parse_llm_output(llm_output)
        
        # Optimize query for visualization if needed
        if query and tool == "read_neo4j_cypher":
            query = optimize_query_for_visualization(query, state.node_limit)
        
        logger.info(f"LLM selected tool: {tool}, query: {query[:100] if query else 'None'}")
        
        return {
            "question": state.question,
            "session_id": state.session_id,
            "tool": tool or "",
            "query": query or "",
            "trace": trace or "",
            "answer": "",
            "graph_data": None,
            "node_limit": state.node_limit
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
            "graph_data": None,
            "node_limit": state.node_limit
        }

def execute_tool_node(state: AgentState) -> dict:
    """Node to execute the selected tool with enhanced graph support"""
    tool = state.tool
    query = state.query
    trace = state.trace
    node_limit = state.node_limit
    answer = ""
    graph_data = None
    
    valid_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    
    logger.info(f"Executing tool: {tool} with node limit: {node_limit}")
    
    try:
        if not tool:
            answer = "⚠️ I couldn't determine the right tool for your question. Try asking about viewing data, making changes, or exploring the database schema."
        elif tool not in valid_tools:
            answer = f"⚠️ Tool '{tool}' not recognized. Available tools: {', '.join(valid_tools)}"
        elif tool == "get_neo4j_schema":
            result = requests.post("http://localhost:8000/get_neo4j_schema", headers=headers, timeout=30)
            if result.ok:
                answer, graph_data = format_response_with_graph(result.json(), tool, node_limit)
            else:
                answer = f"❌ Schema query failed: {result.text}"
        elif tool == "read_neo4j_cypher":
            if not query or not query.strip():
                answer = "⚠️ I couldn't generate a valid query for your question. Try rephrasing or being more specific about what you want to see."
            else:
                query_clean = clean_cypher_query(query)
                data = {
                    "query": query_clean, 
                    "params": {},
                    "node_limit": node_limit
                }
                result = requests.post("http://localhost:8000/read_neo4j_cypher", json=data, headers=headers, timeout=45)
                if result.ok:
                    answer, graph_data = format_response_with_graph(result.json(), tool, node_limit)
                else:
                    answer = f"❌ Query failed: {result.text}"
        elif tool == "write_neo4j_cypher":
            if not query or not query.strip():
                answer = "⚠️ I couldn't generate a valid modification query. Please be more specific about what you want to create, update, or delete."
            else:
                query_clean = clean_cypher_query(query)
                data = {
                    "query": query_clean, 
                    "params": {},
                    "node_limit": node_limit
                }
                result = requests.post("http://localhost:8000/write_neo4j_cypher", json=data, headers=headers, timeout=45)
                if result.ok:
                    answer, graph_data = format_response_with_graph(result.json(), tool, node_limit)
                else:
                    answer = f"❌ Update failed: {result.text}"
        else:
            answer = f"❌ Unknown tool: {tool}"
    
    except requests.exceptions.Timeout:
        answer = "⚠️ Query timed out. Try a simpler query or reduce the data scope."
    except requests.exceptions.ConnectionError:
        answer = "⚠️ Cannot connect to the database server. Please check if all services are running."
    except Exception as e:
        logger.error(f"Error in execute_tool_node: {e}")
        answer = f"⚠️ Execution failed: {str(e)}"
    
    logger.info(f"Tool execution completed. Graph data: {'Yes' if graph_data else 'No'}")
    
    return {
        "question": state.question,
        "session_id": state.session_id,
        "tool": tool,
        "query": query,
        "trace": trace,
        "answer": answer,
        "graph_data": graph_data,
        "node_limit": node_limit
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
    logger.info("LangGraph agent built successfully for split-screen interface")
    return agent

# For testing purposes
if __name__ == "__main__":
    # Test the agent locally
    agent = build_agent()
    test_state = AgentState(
        question="Show me the network structure",
        session_id="test_session",
        node_limit=5000
    )
    
    import asyncio
    
    async def test():
        result = await agent.ainvoke(test_state)
        print("Test Result:", result)
    
    # asyncio.run(test())
