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

def optimize_query_for_visualization(query: str, node_limit: int = 1000) -> str:
    """Optimize queries for better Pyvis visualization performance"""
    query = query.strip()
    
    if ("MATCH" in query.upper() and 
        "LIMIT" not in query.upper() and 
        "count(" not in query.lower() and
        "COUNT(" not in query):
        
        if "RETURN" in query.upper():
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
                
                formatted_response = f"""
📊 **Query Results**

**🔢 Records:** {metadata['record_count']}  
**⚡ Time:** {metadata['execution_time_ms']}ms  
**🕐 Timestamp:** {metadata['timestamp'][:19]}
                """.strip()
                
                if not graph_data or not graph_data.get('nodes'):
                    if isinstance(data, list) and len(data) > 0:
                        if len(data) <= 3:
                            formatted_response += f"\n\n**📋 Data:**\n```json\n{json.dumps(data, indent=2)}\n```"
                        else:
                            formatted_response += f"\n\n**📋 Sample Data:**\n```json\n{json.dumps(data[:2], indent=2)}\n... and {len(data) - 2} more records\n```"
                    else:
                        formatted_response += "\n\n**📋 Data:** No records found"
                
                if graph_data and graph_data.get('nodes'):
                    node_count = len(graph_data['nodes'])
                    rel_count = len(graph_data.get('relationships', []))
                    
                    formatted_response += f"\n\n🕸️ **Graph visualization updated** with {node_count} nodes and {rel_count} relationships"
                    
                    if node_count > 0:
                        label_counts = {}
                        for node in graph_data['nodes']:
                            for label in node.get('labels', ['Unknown']):
                                label_counts[label] = label_counts.get(label, 0) + 1
                        
                        if len(label_counts) > 0:
                            label_summary = ", ".join([f"{label}({count})" for label, count in sorted(label_counts.items())])
                            formatted_response += f"\n**🏷️ Node Types:** {label_summary}"
                    
                    if graph_data.get('limited'):
                        formatted_response += f"\n**⚠️ Display limited to {node_limit} nodes for performance**"
                
                return formatted_response, graph_data
        
        elif tool_type == "get_neo4j_schema" and isinstance(result_data, dict):
            if "schema" in result_data:
                schema = result_data["schema"]
                metadata = result_data.get("metadata", {})
                
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
        
        formatted_text = json.dumps(result_data, indent=2) if isinstance(result_data, (dict, list)) else str(result_data)
        return formatted_text, None
    
    except Exception as e:
        error_msg = f"❌ **Error formatting response:** {str(e)}"
        logger.error(error_msg)
        return error_msg, None

# Enhanced system message - simpler and more explicit
SYS_MSG = """You are a Neo4j database assistant. For each user question, you must select ONE tool and provide a Cypher query (if needed).

REQUIRED OUTPUT FORMAT:
Tool: [tool_name]
Query: [cypher_query_if_needed]

AVAILABLE TOOLS:
1. read_neo4j_cypher - for reading data (MATCH, RETURN, WHERE, COUNT, etc.)
2. write_neo4j_cypher - for modifying data (CREATE, MERGE, SET, DELETE, etc.) 
3. get_neo4j_schema - for schema information (no query needed)

EXAMPLES:

User: Show me all Person nodes
Tool: read_neo4j_cypher
Query: MATCH (n:Person) RETURN n LIMIT 25

User: How many nodes are there?
Tool: read_neo4j_cypher  
Query: MATCH (n) RETURN count(n) AS node_count

User: Create a person named John
Tool: write_neo4j_cypher
Query: CREATE (n:Person {name: "John"}) RETURN n

User: What is the database schema?
Tool: get_neo4j_schema

Always respond with Tool: and Query: on separate lines."""

# Cortex LLM configuration
API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
MODEL = "llama3.1-70b"

def cortex_llm(prompt: str, session_id: str) -> str:
    """Call the Cortex LLM API with enhanced debugging"""
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
        logger.info(f"🔄 Calling Cortex LLM with prompt: {prompt[:100]}...")
        resp = requests.post(API_URL, headers=headers, json=payload, verify=False, timeout=30)
        resp.raise_for_status()
        
        # Enhanced response parsing with debugging
        raw_response = resp.text
        logger.info(f"📥 Raw Cortex response length: {len(raw_response)}")
        logger.info(f"📥 Raw response preview: {raw_response[:200]}...")
        
        # Try different parsing approaches
        if "end_of_stream" in raw_response:
            parsed_response = raw_response.partition("end_of_stream")[0].strip()
            logger.info(f"✂️ Parsed response (end_of_stream): {parsed_response[:200]}...")
        else:
            parsed_response = raw_response.strip()
            logger.info(f"✂️ Parsed response (full): {parsed_response[:200]}...")
        
        return parsed_response
        
    except Exception as e:
        logger.error(f"❌ Cortex LLM API error: {e}")
        return f"Error calling Cortex LLM: {str(e)}"

def parse_llm_output(llm_output):
    """Enhanced parsing with better debugging and fallback logic"""
    allowed_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    trace = llm_output.strip()
    tool = None
    query = None
    
    logger.info(f"🔍 Parsing LLM output (length: {len(llm_output)})")
    logger.info(f"🔍 LLM output preview: {llm_output[:300]}...")
    
    # Multiple patterns to try for tool extraction
    tool_patterns = [
        r"Tool:\s*([\w_]+)",           # Standard format
        r"**Tool:**\s*([\w_]+)",       # Bold format
        r"Tool\s*=\s*([\w_]+)",        # Assignment format
        r"Selected tool:\s*([\w_]+)",  # Alternative format
        r"Using tool:\s*([\w_]+)",     # Another alternative
        r"I'll use:\s*([\w_]+)",       # Natural language
    ]
    
    for pattern in tool_patterns:
        tool_match = re.search(pattern, llm_output, re.I)
        if tool_match:
            tname = tool_match.group(1).strip()
            logger.info(f"🎯 Found tool candidate: '{tname}' using pattern: {pattern}")
            if tname in allowed_tools:
                tool = tname
                logger.info(f"✅ Valid tool found: {tool}")
                break
            else:
                logger.warning(f"⚠️ Invalid tool: '{tname}' not in {allowed_tools}")
    
    # Multiple patterns for query extraction
    query_patterns = [
        r"Query:\s*(.+?)(?:\n\n|\n[A-Z]|$)",      # Standard format
        r"**Query:**\s*(.+?)(?:\n\n|\n[A-Z]|$)",  # Bold format
        r"Query\s*=\s*(.+?)(?:\n\n|\n[A-Z]|$)",   # Assignment format
        r"Cypher:\s*(.+?)(?:\n\n|\n[A-Z]|$)",     # Alternative format
        r"```cypher\s*(.+?)\s*```",               # Code block format
        r"```\s*(.+?)\s*```",                     # Generic code block
    ]
    
    for pattern in query_patterns:
        query_match = re.search(pattern, llm_output, re.I | re.DOTALL)
        if query_match:
            query = query_match.group(1).strip()
            logger.info(f"🎯 Found query candidate using pattern: {pattern}")
            logger.info(f"🎯 Query preview: {query[:100]}...")
            if query and len(query) > 3:  # Basic validation
                logger.info(f"✅ Valid query found")
                break
    
    # Fallback logic for common patterns
    if not tool:
        logger.warning("⚠️ No tool found, attempting fallback logic...")
        
        # Check for common keywords to infer tool
        lower_output = llm_output.lower()
        if any(word in lower_output for word in ["show", "display", "find", "get", "match", "return", "count"]):
            tool = "read_neo4j_cypher"
            logger.info(f"🔄 Fallback: Inferred tool as {tool} based on read keywords")
        elif any(word in lower_output for word in ["create", "add", "insert", "update", "set", "delete", "merge"]):
            tool = "write_neo4j_cypher"
            logger.info(f"🔄 Fallback: Inferred tool as {tool} based on write keywords")
        elif any(word in lower_output for word in ["schema", "structure", "types", "labels", "properties"]):
            tool = "get_neo4j_schema"
            logger.info(f"🔄 Fallback: Inferred tool as {tool} based on schema keywords")
    
    # Fallback query generation if tool found but no query
    if tool and not query and tool != "get_neo4j_schema":
        logger.warning("⚠️ Tool found but no query, attempting to generate fallback query...")
        
        if tool == "read_neo4j_cypher":
            query = "MATCH (n) RETURN n LIMIT 25"
            logger.info(f"🔄 Fallback query generated: {query}")
        elif tool == "write_neo4j_cypher":
            query = "// No specific query could be generated"
            logger.info(f"🔄 Fallback query placeholder: {query}")
    
    logger.info(f"🎯 Final parsing results - Tool: {tool}, Query: {query[:50] if query else 'None'}...")
    
    return tool, query, trace

def select_tool_node(state: AgentState) -> dict:
    """Enhanced tool selection with better error handling"""
    logger.info(f"🤔 Processing question: {state.question}")
    
    try:
        llm_output = cortex_llm(state.question, state.session_id)
        tool, query, trace = parse_llm_output(llm_output)
        
        # Optimize query for visualization if needed
        if query and tool == "read_neo4j_cypher":
            query = optimize_query_for_visualization(query, state.node_limit)
        
        logger.info(f"✅ Tool selection complete - Tool: {tool}, Query: {query[:100] if query else 'None'}")
        
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
        logger.error(f"❌ Error in select_tool_node: {e}")
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
    """Enhanced tool execution with better debugging"""
    tool = state.tool
    query = state.query
    trace = state.trace
    node_limit = state.node_limit
    answer = ""
    graph_data = None
    
    valid_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    
    logger.info(f"⚡ Executing tool: '{tool}' with node limit: {node_limit}")
    logger.info(f"🔧 Query: {query[:200] if query else 'None'}...")
    
    try:
        if not tool:
            logger.error("❌ No tool selected")
            answer = "⚠️ I couldn't determine the right tool for your question. Please check the logs for debugging information. Try asking about viewing data, making changes, or exploring the database schema."
        elif tool not in valid_tools:
            logger.error(f"❌ Invalid tool: {tool}")
            answer = f"⚠️ Tool '{tool}' not recognized. Available tools: {', '.join(valid_tools)}"
        elif tool == "get_neo4j_schema":
            logger.info("📋 Calling get_neo4j_schema endpoint...")
            result = requests.post("http://localhost:8000/get_neo4j_schema", headers=headers, timeout=30)
            if result.ok:
                answer, graph_data = format_response_with_graph(result.json(), tool, node_limit)
                logger.info("✅ Schema retrieval successful")
            else:
                logger.error(f"❌ Schema query failed: {result.status_code} - {result.text}")
                answer = f"❌ Schema query failed: {result.text}"
        elif tool == "read_neo4j_cypher":
            if not query or not query.strip():
                logger.error("❌ No query provided for read operation")
                answer = "⚠️ I couldn't generate a valid query for your question. Try rephrasing or being more specific about what you want to see."
            else:
                logger.info("📖 Executing read query...")
                query_clean = clean_cypher_query(query)
                data = {
                    "query": query_clean, 
                    "params": {},
                    "node_limit": node_limit
                }
                result = requests.post("http://localhost:8000/read_neo4j_cypher", json=data, headers=headers, timeout=45)
                if result.ok:
                    answer, graph_data = format_response_with_graph(result.json(), tool, node_limit)
                    logger.info("✅ Read query successful")
                else:
                    logger.error(f"❌ Read query failed: {result.status_code} - {result.text}")
                    answer = f"❌ Query failed: {result.text}"
        elif tool == "write_neo4j_cypher":
            if not query or not query.strip():
                logger.error("❌ No query provided for write operation")
                answer = "⚠️ I couldn't generate a valid modification query. Please be more specific about what you want to create, update, or delete."
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
                    answer, graph_data = format_response_with_graph(result.json(), tool, node_limit)
                    logger.info("✅ Write query successful")
                else:
                    logger.error(f"❌ Write query failed: {result.status_code} - {result.text}")
                    answer = f"❌ Update failed: {result.text}"
        else:
            logger.error(f"❌ Unknown tool: {tool}")
            answer = f"❌ Unknown tool: {tool}"
    
    except requests.exceptions.Timeout:
        logger.error("⏰ Request timed out")
        answer = "⚠️ Query timed out. Try a simpler query or reduce the data scope."
    except requests.exceptions.ConnectionError:
        logger.error("🔌 Connection error")
        answer = "⚠️ Cannot connect to the database server. Please check if all services are running."
    except Exception as e:
        logger.error(f"💥 Unexpected error in execute_tool_node: {e}")
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
    logger.info("🚀 LangGraph agent built successfully for split-screen interface")
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
