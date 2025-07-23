import requests
import urllib3
from pydantic import BaseModel
from typing import Optional
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
    graph_data: Optional[dict] = None
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

def enhance_query_for_relationships(query: str) -> str:
    """Enhance queries to always include relationships for graph visualization"""
    query = query.strip()
    
    # If it's a simple MATCH (n) RETURN n, enhance it to include relationships
    if re.match(r'MATCH\s*\([^)]+\)\s*RETURN\s+[^L]+(?:LIMIT\s+\d+)?$', query, re.I):
        # Convert "MATCH (n:Person) RETURN n" to include relationships
        base_match = re.search(r'MATCH\s*\(([^)]+)\)', query, re.I)
        limit_match = re.search(r'LIMIT\s+(\d+)', query, re.I)
        
        if base_match:
            node_pattern = base_match.group(1)
            limit_clause = f" LIMIT {limit_match.group(1)}" if limit_match else " LIMIT 50"
            
            # Enhanced query that returns nodes and their relationships
            enhanced_query = f"""
            MATCH (n{':' + node_pattern if not node_pattern.startswith(':') and ':' in node_pattern else ''})
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n, r, m{limit_clause}
            """.strip()
            
            logger.info(f"Enhanced query from '{query}' to '{enhanced_query}'")
            return enhanced_query
    
    # If query doesn't have relationships, try to add them
    if "RETURN" in query.upper() and "-[" not in query and "OPTIONAL MATCH" not in query.upper():
        # Add optional relationship matching
        parts = query.split("RETURN")
        if len(parts) == 2:
            match_part = parts[0].strip()
            return_part = parts[1].strip()
            
            # Extract main node variable (usually first letter after MATCH)
            node_var_match = re.search(r'MATCH\s*\(([a-zA-Z])', match_part, re.I)
            if node_var_match:
                node_var = node_var_match.group(1)
                enhanced_query = f"""
                {match_part}
                OPTIONAL MATCH ({node_var})-[r]-(m)
                RETURN {node_var}, r, m
                """ + (f" LIMIT {re.search(r'LIMIT s*(d+)', return_part, re.I).group(1)}" if "LIMIT" in return_part.upper() else " LIMIT 50")
                
                logger.info(f"Added relationships to query: '{enhanced_query.strip()}'")
                return enhanced_query.strip()
    
    return query

def format_response_with_graph(result_data, tool_type, node_limit=5000, question=""):
    """Format the response for display with enhanced explanations"""
    try:
        if isinstance(result_data, str):
            try:
                result_data = json.loads(result_data)
            except:
                return str(result_data), None
        
        graph_data = result_data.get("graph_data")
        
        if tool_type == "write_neo4j_cypher" and isinstance(result_data, dict):
            if "change_info" in result_data:
                change_info = result_data["change_info"]
                formatted_response = f"""
🔄 **Database Update Completed**

**⚡ Execution:** {change_info['execution_time_ms']}ms  
**🕐 Time:** {change_info['timestamp'][:19]}

**📝 Changes Made:**
{chr(10).join(f"• {change}" for change in change_info['changes'])}

**🔧 Query:** `{change_info['query']}`

---

## ✏️ What I Did
I executed a write operation to modify your Neo4j database. The changes have been applied and the graph visualization has been updated to reflect the current state.

**💡 What This Means:** Your database structure has been modified. Any new nodes or relationships are now part of your graph and will be visible in the visualization.

**📊 Next Steps:** The graph will automatically refresh to show the updated data with all relationships visible.
                """.strip()
                
                # Try to get refreshed graph data
                if not graph_data:
                    graph_data = get_refreshed_graph_data(node_limit)
                
                return formatted_response, graph_data
        
        elif tool_type == "read_neo4j_cypher" and isinstance(result_data, dict):
            if "data" in result_data and "metadata" in result_data:
                data = result_data["data"]
                metadata = result_data["metadata"]
                
                formatted_response = f"""
📊 **Query Results**

**🔢 Records:** {metadata['record_count']}  
**⚡ Time:** {metadata['execution_time_ms']}ms  
**🕐 Timestamp:** {metadata['timestamp'][:19]}

---

## 📈 What I Found
I successfully retrieved {metadata['record_count']} records from your Neo4j database.
                """.strip()
                
                # Add graph visualization info if available
                if graph_data and graph_data.get('nodes'):
                    node_count = len(graph_data['nodes'])
                    rel_count = len(graph_data.get('relationships', []))
                    
                    formatted_response += f"""

**🕸️ Graph Visualization:** Updated with {node_count} nodes and {rel_count} relationships

**🎨 Visual Elements:**
• Colored nodes representing different types
• Relationship lines showing connections between nodes
• Interactive features (drag, zoom, hover for details)
                    """
                    
                    # Show node types summary
                    if node_count > 0:
                        label_counts = {}
                        for node in graph_data['nodes']:
                            for label in node.get('labels', ['Unknown']):
                                label_counts[label] = label_counts.get(label, 0) + 1
                        
                        if len(label_counts) > 0:
                            label_summary = ", ".join([f"{label}({count})" for label, count in sorted(label_counts.items())])
                            formatted_response += f"\n**🏷️ Node Types:** {label_summary}"
                    
                    # Show relationship types
                    if rel_count > 0:
                        rel_types = list(set(rel.get('type', 'UNKNOWN') for rel in graph_data.get('relationships', [])))
                        if rel_types:
                            formatted_response += f"\n**🔗 Relationship Types:** {', '.join(sorted(rel_types))}"
                    
                    if graph_data.get('limited'):
                        formatted_response += f"\n**⚠️ Display limited to {node_limit} nodes for performance**"
                else:
                    # No graph data, show tabular results
                    if isinstance(data, list) and len(data) > 0:
                        if len(data) <= 3:
                            formatted_response += f"\n\n**📋 Data:**\n```json\n{json.dumps(data, indent=2)}\n```"
                        else:
                            formatted_response += f"\n\n**📋 Sample Data:**\n```json\n{json.dumps(data[:2], indent=2)}\n... and {len(data) - 2} more records\n```"
                    else:
                        formatted_response += "\n\n**📋 Result:** No records found matching your criteria"
                
                formatted_response += "\n\n**💡 What This Means:** " + interpret_query_results(question, metadata['record_count'], graph_data)
                
                return formatted_response, graph_data
        
        elif tool_type == "get_neo4j_schema" and isinstance(result_data, dict):
            if "schema" in result_data:
                schema = result_data["schema"]
                metadata = result_data.get("metadata", {})
                
                # Format schema information
                schema_summary = []
                if isinstance(schema, dict):
                    for label, info in schema.items():
                        if isinstance(info, dict):
                            props = info.get('properties', {})
                            relationships = info.get('relationships', {})
                            schema_summary.append(f"**{label}**: {len(props)} properties, {len(relationships)} relationship types")
                
                formatted_response = f"""
🏗️ **Database Schema Retrieved**

**⚡ Time:** {metadata.get('execution_time_ms', 'N/A')}ms

**📊 Schema Overview:**
{chr(10).join(f"• {item}" for item in schema_summary[:10])}
{f"... and {len(schema_summary) - 10} more types" if len(schema_summary) > 10 else ""}

---

## 🎯 What This Shows
I've analyzed your database structure and found the available node types, properties, and relationships. This schema information helps understand what data you can query and visualize.

**💡 Use This To:** Ask specific questions about your data, create new nodes/relationships, or explore existing connections.
                """.strip()
                
                return formatted_response, None
        
        # Fallback
        formatted_text = json.dumps(result_data, indent=2) if isinstance(result_data, (dict, list)) else str(result_data)
        return formatted_text, graph_data
    
    except Exception as e:
        error_msg = f"❌ **Error formatting response:** {str(e)}"
        logger.error(error_msg)
        return error_msg, None

def interpret_query_results(question: str, record_count: int, graph_data: dict) -> str:
    """Provide intelligent interpretation of query results"""
    question_lower = question.lower()
    
    if "count" in question_lower or "how many" in question_lower:
        return f"I found {record_count} items matching your criteria."
    elif "person" in question_lower or "people" in question_lower:
        if graph_data and graph_data.get('relationships'):
            return f"I found {record_count} people and their connections. The graph shows how they're related to each other and other entities."
        else:
            return f"I found {record_count} people in your database."
    elif record_count == 0:
        return "No data matched your search criteria. Try a broader query or check if the data exists."
    elif graph_data and len(graph_data.get('relationships', [])) > 0:
        return f"I retrieved {record_count} records with their relationships. The graph visualization shows the connections between different entities."
    else:
        return f"I successfully retrieved {record_count} records from your database."

def get_refreshed_graph_data(node_limit: int = 1000) -> dict:
    """Get fresh graph data to ensure visualization is current"""
    try:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        
        # Query to get current graph state with relationships
        refresh_query = f"""
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n, r, m
        LIMIT {min(node_limit, 100)}
        """
        
        data = {
            "query": refresh_query,
            "params": {},
            "node_limit": node_limit
        }
        
        result = requests.post("http://localhost:8000/read_neo4j_cypher", json=data, headers=headers, timeout=30)
        
        if result.ok:
            response_data = result.json()
            return response_data.get("graph_data")
        else:
            logger.warning(f"Could not refresh graph data: {result.text}")
            return None
            
    except Exception as e:
        logger.warning(f"Error refreshing graph data: {e}")
        return None

# Cortex LLM configuration
API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
MODEL = "llama3.1-70b"

# Very simple and reliable system message
SYS_MSG = """You are a Neo4j database assistant. You must respond with exactly this format:

Tool: [tool_name]
Query: [cypher_query]

TOOLS:
- read_neo4j_cypher (for viewing/reading data)
- write_neo4j_cypher (for creating/updating/deleting data)  
- get_neo4j_schema (for schema information)

EXAMPLES:

User: Show me all nodes
Tool: read_neo4j_cypher
Query: MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 50

User: Show me Person nodes
Tool: read_neo4j_cypher
Query: MATCH (n:Person) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 50

User: How many nodes?
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN count(n) AS node_count

User: Create a person named John
Tool: write_neo4j_cypher
Query: CREATE (n:Person {name: "John"}) RETURN n

User: Delete person John
Tool: write_neo4j_cypher
Query: MATCH (n:Person {name: "John"}) DETACH DELETE n

User: What is the schema?
Tool: get_neo4j_schema

ALWAYS include OPTIONAL MATCH for relationships in read queries to show connections in the graph."""

def cortex_llm(prompt: str, session_id: str) -> str:
    """Reliable Cortex LLM call with comprehensive debugging"""
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
        logger.info(f"🔄 Calling Cortex LLM for: {prompt[:50]}...")
        resp = requests.post(API_URL, headers=headers, json=payload, verify=False, timeout=30)
        resp.raise_for_status()
        
        raw_response = resp.text
        logger.info(f"📥 Raw response length: {len(raw_response)}")
        
        # Enhanced parsing
        if "end_of_stream" in raw_response:
            parsed_response = raw_response.partition("end_of_stream")[0].strip()
        else:
            parsed_response = raw_response.strip()
        
        logger.info(f"✂️ Parsed response: {parsed_response[:100]}...")
        return parsed_response
        
    except Exception as e:
        logger.error(f"❌ Cortex LLM API error: {e}")
        return f"Error calling Cortex LLM: {str(e)}"

def parse_llm_output_robust(llm_output):
    """Ultra-robust parsing with multiple fallbacks and relationship focus"""
    allowed_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    tool = None
    query = None
    
    logger.info(f"🔍 Parsing LLM output: {llm_output[:200]}...")
    
    # Try standard parsing first
    tool_match = re.search(r"Tool:\s*([\w_]+)", llm_output, re.I)
    if tool_match:
        tname = tool_match.group(1).strip()
        if tname in allowed_tools:
            tool = tname
            logger.info(f"✅ Found tool: {tool}")
    
    query_match = re.search(r"Query:\s*(.+?)(?:\n|$)", llm_output, re.I | re.DOTALL)
    if query_match:
        query = query_match.group(1).strip()
        logger.info(f"✅ Found query: {query[:50]}...")
    
    # If parsing failed, use intelligent keyword fallback
    if not tool:
        logger.warning("⚠️ LLM parsing failed, using keyword fallback...")
        tool, query = intelligent_keyword_fallback(llm_output)
    
    # Enhance query for relationships if it's a read operation
    if tool == "read_neo4j_cypher" and query:
        enhanced_query = enhance_query_for_relationships(query)
        if enhanced_query != query:
            logger.info(f"🔗 Enhanced query for relationships")
            query = enhanced_query
    
    logger.info(f"🎯 Final result - Tool: {tool}, Query: {query[:50] if query else 'None'}...")
    return tool, query, llm_output

def intelligent_keyword_fallback(text: str) -> tuple:
    """Intelligent keyword-based tool selection with relationship focus"""
    text_lower = text.lower()
    
    logger.info(f"🧠 Using intelligent fallback for: {text[:100]}...")
    
    # Schema-related keywords
    if any(word in text_lower for word in ["schema", "structure", "types", "labels", "properties", "what is in"]):
        return "get_neo4j_schema", ""
    
    # Write operation keywords
    elif any(word in text_lower for word in ["create", "add", "insert", "update", "set", "delete", "remove", "merge"]):
        if "person" in text_lower or "people" in text_lower:
            if "delete" in text_lower or "remove" in text_lower:
                return "write_neo4j_cypher", "MATCH (n:Person) DETACH DELETE n LIMIT 1"
            else:
                return "write_neo4j_cypher", "CREATE (n:Person {name: 'New Person'}) RETURN n"
        else:
            return "write_neo4j_cypher", "CREATE (n {name: 'New Node'}) RETURN n"
    
    # Read operation keywords - ALWAYS include relationships
    else:
        if "count" in text_lower or "how many" in text_lower:
            return "read_neo4j_cypher", "MATCH (n) RETURN count(n) AS node_count"
        elif "person" in text_lower or "people" in text_lower:
            return "read_neo4j_cypher", "MATCH (n:Person) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 50"
        elif "company" in text_lower:
            return "read_neo4j_cypher", "MATCH (n:Company) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 50"
        elif "relationship" in text_lower or "connection" in text_lower:
            return "read_neo4j_cypher", "MATCH (a)-[r]-(b) RETURN a, r, b LIMIT 50"
        else:
            # Default query that always includes relationships
            return "read_neo4j_cypher", "MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 50"

def select_tool_node(state: AgentState) -> dict:
    """Bulletproof tool selection with relationship focus"""
    logger.info(f"🤔 Processing question: {state.question}")
    
    try:
        # Try LLM first
        llm_output = cortex_llm(state.question, state.session_id)
        tool, query, trace = parse_llm_output_robust(llm_output)
        
        # If LLM completely failed, use pure keyword fallback
        if not tool:
            logger.warning("🔄 LLM completely failed, using pure keyword fallback...")
            tool, query = intelligent_keyword_fallback(state.question)
            trace = f"Pure keyword fallback used: {tool}"
        
        logger.info(f"✅ Final tool selection - Tool: {tool}, Query: {query[:100] if query else 'None'}")
        
        return {
            "question": state.question,
            "session_id": state.session_id,
            "tool": tool or "read_neo4j_cypher",  # Ultimate fallback
            "query": query or "MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 25",
            "trace": trace or "Fallback trace",
            "answer": "",
            "graph_data": None,
            "node_limit": state.node_limit
        }
    except Exception as e:
        logger.error(f"❌ Error in select_tool_node: {e}")
        
        # Emergency fallback
        tool, query = intelligent_keyword_fallback(state.question)
        
        return {
            "question": state.question,
            "session_id": state.session_id,
            "tool": tool,
            "query": query,
            "trace": f"Emergency fallback due to error: {str(e)}",
            "answer": "",
            "graph_data": None,
            "node_limit": state.node_limit
        }

def execute_tool_node(state: AgentState) -> dict:
    """Enhanced tool execution with relationship focus"""
    tool = state.tool
    query = state.query
    trace = state.trace
    node_limit = state.node_limit
    answer = ""
    graph_data = None
    
    valid_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    
    logger.info(f"⚡ Executing tool: '{tool}' with query: '{query[:100] if query else 'None'}...'")
    
    try:
        if not tool or tool not in valid_tools:
            # Ultimate fallback
            tool = "read_neo4j_cypher"
            query = "MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 25"
            logger.warning(f"⚠️ Using ultimate fallback: {tool}")
        
        if tool == "get_neo4j_schema":
            logger.info("📋 Getting database schema...")
            result = requests.post("http://localhost:8000/get_neo4j_schema", headers=headers, timeout=30)
            if result.ok:
                answer, graph_data = format_response_with_graph(result.json(), tool, node_limit, state.question)
                logger.info("✅ Schema retrieved successfully")
            else:
                logger.error(f"❌ Schema query failed: {result.text}")
                answer = f"❌ Schema query failed: {result.text}"
                
        elif tool == "read_neo4j_cypher":
            if not query:
                # Emergency query with relationships
                query = "MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 25"
                logger.warning("⚠️ No query provided, using emergency query with relationships")
            
            logger.info("📖 Executing read query with relationships...")
            query_clean = clean_cypher_query(query)
            data = {
                "query": query_clean, 
                "params": {},
                "node_limit": node_limit
            }
            result = requests.post("http://localhost:8000/read_neo4j_cypher", json=data, headers=headers, timeout=45)
            if result.ok:
                answer, graph_data = format_response_with_graph(result.json(), tool, node_limit, state.question)
                logger.info("✅ Read query executed successfully")
                
                # Log relationship info
                if graph_data:
                    rel_count = len(graph_data.get('relationships', []))
                    logger.info(f"🔗 Graph data includes {rel_count} relationships")
            else:
                logger.error(f"❌ Read query failed: {result.text}")
                answer = f"❌ Query failed: {result.text}"
                
        elif tool == "write_neo4j_cypher":
            if not query:
                logger.error("❌ No query provided for write operation")
                answer = "⚠️ I couldn't generate a valid modification query."
            else:
                logger.info("✏️ Executing write query...")
                query_clean = clean_cypher_query(query)
                data = {
                    "query": query_clean, 
                    "params": {},
                    "node_limit": node_limit
                }
                result = requests.post("http://localhost:8000/write_neo4j_cypher", json=data, headers=headers, timeout=45)
                if result.ok:
                    answer, graph_data = format_response_with_graph(result.json(), tool, node_limit, state.question)
                    logger.info("✅ Write query executed successfully")
                    
                    # Get refreshed graph data to show current state
                    if not graph_data:
                        logger.info("🔄 Getting refreshed graph data after write operation...")
                        graph_data = get_refreshed_graph_data(node_limit)
                        if graph_data:
                            rel_count = len(graph_data.get('relationships', []))
                            logger.info(f"🔗 Refreshed graph includes {rel_count} relationships")
                else:
                    logger.error(f"❌ Write query failed: {result.text}")
                    answer = f"❌ Update failed: {result.text}"
    
    except requests.exceptions.Timeout:
        logger.error("⏰ Request timed out")
        answer = "⚠️ Query timed out. Try a simpler query or reduce the data scope."
    except requests.exceptions.ConnectionError:
        logger.error("🔌 Connection error")
        answer = "⚠️ Cannot connect to the database server. Please check if all services are running."
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        answer = f"⚠️ Execution failed: {str(e)}"
    
    logger.info(f"🏁 Tool execution completed. Graph data: {'Yes' if graph_data else 'No'}")
    
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
    """Build the bulletproof LangGraph agent with relationship focus"""
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
    logger.info("🚀 Bulletproof LangGraph agent built with relationship focus")
    return agent

# For testing purposes
if __name__ == "__main__":
    agent = build_agent()
    test_state = AgentState(
        question="Show me all Person nodes with their relationships",
        session_id="test_session",
        node_limit=50
    )
    
    import asyncio
    
    async def test():
        result = await agent.ainvoke(test_state)
        print("Test Result:", result)
    
    # asyncio.run(test())
