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
    node_limit: int = 5000

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

def optimize_query_for_visualization(query: str, node_limit: int = 5000) -> str:
    """Optimize queries for better visualization performance"""
    query = query.strip()
    
    # Add reasonable limits to MATCH queries that don't have them
    if ("MATCH" in query.upper() and 
        "LIMIT" not in query.upper() and 
        "count(" not in query.lower() and
        "COUNT(" not in query):
        
        # For visualization queries, add a reasonable limit
        if "RETURN" in query.upper():
            query += f" LIMIT {min(node_limit, 1000)}"
    
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
You are an expert AI assistant for a split-screen Neo4j graph explorer. The left side shows conversation, the right side shows interactive graph visualizations. Your goal is to provide great queries that create meaningful visualizations.

INTERFACE CONTEXT:
- Split-screen UI: Chat on left, graph visualization on right
- Node limit: 5000 for performance (you can use smaller limits for specific queries)
- Users see results immediately in both text and visual form
- Focus on queries that create meaningful, explorable graphs

TOOL DESCRIPTIONS:
- read_neo4j_cypher: For all read queries. Returns data + graph visualization when nodes/relationships are queried.
- write_neo4j_cypher: For create/update/delete operations. Shows changes + updated visualization.
- get_neo4j_schema: For schema information. Shows database structure overview.

VISUALIZATION OPTIMIZATION RULES:
1. When users want to "see", "show", "explore", or "visualize" data, prioritize queries returning nodes and relationships
2. Use LIMIT clauses to control visualization size (50-1000 nodes typically)
3. For exploration queries, prefer: MATCH (n)-[r]->(m) RETURN n, r, m LIMIT X
4. For specific entity queries: MATCH (n:Label) RETURN n LIMIT X
5. Always include RETURN clauses in CREATE/MERGE for immediate visualization

QUERY PATTERNS FOR GREAT VISUALIZATIONS:

Network Exploration:
- "Show network" → MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 100
- "Explore connections" → MATCH (n)-[r]-(m) WHERE n.property = 'value' RETURN n, r, m LIMIT 50

Entity Queries:
- "Show people" → MATCH (p:Person) RETURN p LIMIT 50  
- "Find companies" → MATCH (c:Company) RETURN c LIMIT 30

Relationship Queries:
- "Who works where" → MATCH (p:Person)-[r:WORKS_FOR]->(c:Company) RETURN p, r, c
- "Show hierarchy" → MATCH (a)-[r:MANAGES]->(b) RETURN a, r, b

Creation with Visualization:
- "Create person" → CREATE (p:Person {name: 'X'}) RETURN p
- "Connect people" → MATCH (a:Person {name: 'X'}), (b:Person {name: 'Y'}) CREATE (a)-[r:KNOWS]->(b) RETURN a, r, b

RESPONSE STYLE:
- Keep responses concise and visualization-focused
- Mention when graph updates will be visible
- Use engaging language about exploration
- Point out interesting patterns users can click on

EXAMPLES:

User: Show me the network structure
Tool: read_neo4j_cypher  
Query: MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 200

User: Create a person named Alice and connect her to existing people
Tool: write_neo4j_cypher
Query: CREATE (alice:Person {name: 'Alice', created: datetime()}) WITH alice MATCH (others:Person) WHERE others.name IN ['Bob', 'Charlie'] CREATE (alice)-[r:KNOWS]->(others) RETURN alice, r, others

User: Find all managers and their teams
Tool: read_neo4j_cypher
Query: MATCH (manager:Person)-[r:MANAGES]->(employee:Person) RETURN manager, r, employee LIMIT 100

User: What types of data do I have?
Tool: get_neo4j_schema

IMPORTANT:
- Always explain your reasoning briefly
- Focus on creating explorable, interactive visualizations
- Use appropriate LIMIT values for performance
- Ensure queries return graph objects when users want to see/explore data
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
