"""
Enhanced LangGraph Agent with Schema Reading and Unlimited Query Capabilities
This version automatically reads and uses the Neo4j schema for better query generation
"""

import requests
import urllib3
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import re
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced_langgraph_agent")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class AgentState(BaseModel):
    question: str
    session_id: str
    tool: str = ""
    query: str = ""
    trace: str = ""
    answer: str = ""
    graph_data: Optional[dict] = None
    schema_info: Optional[dict] = None
    node_limit: int = 1000  # Default but can be overridden

# Global schema cache
SCHEMA_CACHE = {
    "labels": [],
    "relationship_types": [],
    "properties": {},
    "schema_graph": {},
    "last_updated": None
}

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

def fetch_neo4j_schema() -> Dict[str, Any]:
    """Fetch comprehensive Neo4j schema information"""
    try:
        logger.info("🔍 Fetching Neo4j schema...")
        
        # Get schema from MCP server
        response = requests.post("http://localhost:8000/get_neo4j_schema", 
                               headers={"Content-Type": "application/json"}, 
                               timeout=30)
        
        if response.status_code == 200:
            schema_data = response.json()
            
            # Extract comprehensive schema information
            schema = schema_data.get("schema", {})
            
            # Get additional schema details
            labels = []
            relationship_types = []
            properties = {}
            
            if isinstance(schema, dict):
                for label, info in schema.items():
                    if isinstance(info, dict):
                        labels.append(label)
                        if "properties" in info:
                            properties[label] = list(info["properties"].keys())
                        if "relationships" in info:
                            for rel_info in info["relationships"]:
                                if isinstance(rel_info, dict):
                                    rel_type = rel_info.get("type")
                                    if rel_type and rel_type not in relationship_types:
                                        relationship_types.append(rel_type)
            
            # Update global cache
            SCHEMA_CACHE.update({
                "labels": labels,
                "relationship_types": relationship_types,
                "properties": properties,
                "schema_graph": schema,
                "last_updated": datetime.now().isoformat()
            })
            
            logger.info(f"✅ Schema loaded: {len(labels)} labels, {len(relationship_types)} relationship types")
            return SCHEMA_CACHE
            
        else:
            logger.error(f"❌ Schema fetch failed: {response.status_code}")
            return {}
            
    except Exception as e:
        logger.error(f"❌ Error fetching schema: {e}")
        return {}

def get_enhanced_system_prompt() -> str:
    """Generate enhanced system prompt with schema information"""
    
    # Fetch latest schema if needed
    if not SCHEMA_CACHE.get("last_updated"):
        fetch_neo4j_schema()
    
    labels = SCHEMA_CACHE.get("labels", [])
    relationship_types = SCHEMA_CACHE.get("relationship_types", [])
    properties = SCHEMA_CACHE.get("properties", {})
    
    schema_context = ""
    if labels:
        schema_context += f"\nAVAILABLE NODE LABELS: {', '.join(labels)}\n"
    
    if relationship_types:
        schema_context += f"AVAILABLE RELATIONSHIP TYPES: {', '.join(relationship_types)}\n"
    
    if properties:
        schema_context += "\nNODE PROPERTIES BY TYPE:\n"
        for label, props in properties.items():
            if props:
                schema_context += f"- {label}: {', '.join(props[:10])}{'...' if len(props) > 10 else ''}\n"
    
    base_prompt = f"""You are an expert Neo4j database assistant with complete knowledge of the database schema. 

{schema_context}

TOOL SELECTION RULES:
1. read_neo4j_cypher - for ALL read operations (MATCH, RETURN, WHERE, COUNT, aggregations, reporting)
2. write_neo4j_cypher - for ALL write operations (CREATE, MERGE, SET, DELETE, UPDATE)
3. get_neo4j_schema - for schema exploration and database structure questions

IMPORTANT GUIDELINES:
- NEVER add arbitrary LIMIT clauses unless specifically requested
- Use the actual node labels and relationship types from the schema above
- Generate comprehensive queries that show the full network structure
- For exploration queries, return complete relationship paths
- Use property names that exist in the schema
- Always format your response exactly as: Tool: [tool_name] Query: [complete_query]

ENHANCED QUERY PATTERNS:

Network Exploration (NO LIMITS):
- "Show all connections": MATCH (n)-[r]->(m) RETURN n, r, m
- "Show network structure": MATCH p=()-[]-() RETURN p
- "Find central nodes": MATCH (n)-[r]-() RETURN n, count(r) as connections ORDER BY connections DESC

Data Discovery:
- "Show all Person nodes": MATCH (n:Person) RETURN n
- "Show relationships": MATCH ()-[r]-() RETURN DISTINCT type(r), count(r)
- "Show node types": MATCH (n) RETURN DISTINCT labels(n), count(n)

Complex Patterns:
- "Show communities": MATCH (n)-[r1]-(m)-[r2]-(o) RETURN n, r1, m, r2, o
- "Show paths": MATCH path = (start)-[*1..3]-(end) RETURN path
- "Show subgraphs": MATCH (center)-[r]-(connected) RETURN center, collect(r), collect(connected)

NEVER use LIMIT unless the user specifically asks for "top N" or "first X" results.
ALWAYS show complete network structures when exploring data.
Generate queries that reveal the full graph structure and relationships.

OUTPUT FORMAT:
Tool: [exact_tool_name]
Query: [complete_cypher_query_on_single_line]"""

    return base_prompt

# Cortex LLM configuration
API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
MODEL = "llama3.1-70b"

def cortex_llm(prompt: str, session_id: str) -> str:
    """Call the Cortex LLM API with enhanced schema context"""
    headers = {
        "Authorization": f'Snowflake Token="{API_KEY}"',
        "Content-Type": "application/json"
    }
    
    # Get enhanced system prompt with schema
    system_prompt = get_enhanced_system_prompt()
    
    payload = {
        "query": {
            "aplctn_cd": "edagnai",
            "app_id": "edadip",
            "api_key": API_KEY,
            "method": "cortex",
            "model": MODEL,
            "sys_msg": system_prompt,
            "limit_convs": "0",
            "prompt": {
                "messages": [{"role": "user", "content": prompt}]
            },
            "session_id": session_id
        }
    }
    
    try:
        logger.info(f"🔄 Calling Cortex LLM with schema-enhanced prompt...")
        resp = requests.post(API_URL, headers=headers, json=payload, verify=False, timeout=30)
        resp.raise_for_status()
        
        raw_response = resp.text
        logger.info(f"📥 Raw response length: {len(raw_response)}")
        
        if "end_of_stream" in raw_response:
            parsed_response = raw_response.partition("end_of_stream")[0].strip()
        else:
            parsed_response = raw_response.strip()
        
        return parsed_response
        
    except Exception as e:
        logger.error(f"❌ Cortex LLM API error: {e}")
        return f"Error calling Cortex LLM: {str(e)}"

def parse_llm_output_enhanced(llm_output, question):
    """Enhanced parsing with schema awareness"""
    allowed_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    trace = llm_output.strip()
    tool = None
    query = None
    
    logger.info(f"🔍 Parsing enhanced LLM output...")
    
    # Enhanced tool extraction patterns
    tool_patterns = [
        r"Tool:\s*([\w_]+)",
        r"**Tool:**\s*([\w_]+)",
        r"Tool\s*=\s*([\w_]+)",
        r"Selected tool:\s*([\w_]+)",
        r"I'll use:\s*([\w_]+)",
    ]
    
    for pattern in tool_patterns:
        tool_match = re.search(pattern, llm_output, re.I)
        if tool_match:
            tname = tool_match.group(1).strip()
            if tname in allowed_tools:
                tool = tname
                logger.info(f"✅ Tool found: {tool}")
                break
    
    # Enhanced query extraction
    query_patterns = [
        r"Query:\s*(.+?)(?:\n\n|\n[A-Z]|$)",
        r"**Query:**\s*(.+?)(?:\n\n|\n[A-Z]|$)",
        r"Cypher:\s*(.+?)(?:\n\n|\n[A-Z]|$)",
        r"```cypher\s*(.+?)\s*```",
        r"```\s*(.+?)\s*```",
    ]
    
    for pattern in query_patterns:
        query_match = re.search(pattern, llm_output, re.I | re.DOTALL)
        if query_match:
            query = query_match.group(1).strip()
            if query and len(query) > 3:
                logger.info(f"✅ Query found: {query[:100]}...")
                break
    
    # Schema-aware fallback logic
    if not tool:
        logger.warning("⚠️ No tool found, using schema-aware fallback...")
        
        q_lower = llm_output.lower()
        if any(word in q_lower for word in ["schema", "structure", "labels", "relationships", "types"]):
            tool = "get_neo4j_schema"
        elif any(word in q_lower for word in ["create", "add", "insert", "update", "set", "delete", "merge"]):
            tool = "write_neo4j_cypher"
        else:
            tool = "read_neo4j_cypher"
        
        logger.info(f"🔄 Fallback tool: {tool}")
    
    # Schema-aware query generation
    if tool and not query and tool != "get_neo4j_schema":
        if tool == "read_neo4j_cypher":
            # Generate exploration query based on schema
            labels = SCHEMA_CACHE.get("labels", [])
            if "Person" in labels:
                query = "MATCH (n:Person) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m"
            elif labels:
                query = f"MATCH (n:{labels[0]}) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m"
            else:
                query = "MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m"
        elif tool == "write_neo4j_cypher":
            query = "CREATE (n:ExampleNode {name: 'Generated', created: datetime()}) RETURN n"
        
        logger.info(f"🔄 Schema-aware query generated: {query}")
    
    # Remove any unwanted LIMIT clauses unless specifically requested
    if query and "LIMIT" in query.upper() and "limit" not in question.lower() and "first" not in question.lower() and "top" not in question.lower():
        query = re.sub(r'\s+LIMIT\s+\d+', '', query, flags=re.I)
        logger.info(f"🔧 Removed unnecessary LIMIT clause")
    
    return tool, query, trace

def format_response_with_graph_enhanced(result_data, tool_type, question=""):
    """Enhanced response formatting with better graph data handling"""
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
🔄 **Database Update Completed Successfully**

**⚡ Performance:** {change_info['execution_time_ms']}ms  
**🕐 Timestamp:** {change_info['timestamp'][:19]}

**📝 Changes Made:**
{chr(10).join(f"  {change}" for change in change_info['changes'])}

**🔧 Executed Query:** `{change_info['query']}`

✅ **Database state updated** - Network graph will refresh to show current state
                """.strip()
                
                if result_data.get("graph_data"):
                    graph_data = result_data["graph_data"]
                    node_count = len(graph_data.get('nodes', []))
                    rel_count = len(graph_data.get('relationships', []))
                    if node_count > 0 or rel_count > 0:
                        formatted_response += f"\n\n🕸️ **Updated visualization** showing {node_count} nodes and {rel_count} relationships"
                
                return formatted_response, graph_data
        
        elif tool_type == "read_neo4j_cypher" and isinstance(result_data, dict):
            if "data" in result_data and "metadata" in result_data:
                data = result_data["data"]
                metadata = result_data["metadata"]
                graph_data = result_data.get("graph_data")
                
                formatted_response = f"""
📊 **Neo4j Query Results**

**🔢 Records Found:** {metadata['record_count']}  
**⚡ Query Time:** {metadata['execution_time_ms']}ms  
**🕐 Executed:** {metadata['timestamp'][:19]}
                """.strip()
                
                # Enhanced data display
                if not graph_data or not graph_data.get('nodes'):
                    if isinstance(data, list) and len(data) > 0:
                        # Show meaningful sample of data
                        sample_size = min(len(data), 5)
                        formatted_response += f"\n\n**📋 Data Preview (showing {sample_size} of {len(data)} records):**\n```json\n{json.dumps(data[:sample_size], indent=2, default=str)}\n```"
                        
                        if len(data) > sample_size:
                            formatted_response += f"\n... and {len(data) - sample_size} more records"
                    else:
                        formatted_response += "\n\n**📋 Result:** No data found - try broader search criteria"
                
                if graph_data and graph_data.get('nodes'):
                    node_count = len(graph_data['nodes'])
                    rel_count = len(graph_data.get('relationships', []))
                    
                    formatted_response += f"\n\n🕸️ **Network Graph Generated**"
                    formatted_response += f"\n📊 **Nodes:** {node_count} | **Relationships:** {rel_count}"
                    
                    if node_count > 0:
                        # Analyze node types for better insight
                        node_types = {}
                        for node in graph_data['nodes']:
                            for label in node.get('labels', ['Unknown']):
                                node_types[label] = node_types.get(label, 0) + 1
                        
                        if len(node_types) > 0:
                            type_summary = ", ".join([f"{label}({count})" for label, count in sorted(node_types.items())])
                            formatted_response += f"\n🏷️ **Node Types:** {type_summary}"
                        
                        # Network density insight
                        density = rel_count / node_count if node_count > 0 else 0
                        if density > 2:
                            formatted_response += f"\n🕸️ **Network:** Highly connected ({density:.1f} connections/node)"
                        elif density > 1:
                            formatted_response += f"\n🔗 **Network:** Well connected ({density:.1f} connections/node)"
                        elif density > 0:
                            formatted_response += f"\n📊 **Network:** Moderately connected ({density:.1f} connections/node)"
                        else:
                            formatted_response += f"\n📍 **Network:** Isolated nodes (no relationships found)"
                
                return formatted_response, graph_data
        
        elif tool_type == "get_neo4j_schema" and isinstance(result_data, dict):
            if "schema" in result_data:
                schema = result_data["schema"]
                metadata = result_data.get("metadata", {})
                
                formatted_response = f"""
🏗️ **Neo4j Database Schema**

**⚡ Retrieved in:** {metadata.get('execution_time_ms', 'N/A')}ms
**🕐 Timestamp:** {metadata.get('timestamp', '')[:19]}

**📊 Database Structure Overview:**
                """.strip()
                
                if isinstance(schema, dict):
                    node_types = list(schema.keys())
                    formatted_response += f"\n\n**🏷️ Node Types Found:** {len(node_types)}"
                    
                    for i, (label, info) in enumerate(schema.items()):
                        if i < 10:  # Show first 10 in detail
                            if isinstance(info, dict):
                                props = info.get('properties', {})
                                relationships = info.get('relationships', [])
                                formatted_response += f"\n\n**{label}:**"
                                formatted_response += f"\n  • Properties: {len(props)} ({', '.join(list(props.keys())[:5])}{'...' if len(props) > 5 else ''})"
                                formatted_response += f"\n  • Relationships: {len(relationships)}"
                        
                    if len(node_types) > 10:
                        formatted_response += f"\n\n... and {len(node_types) - 10} more node types"
                
                # Update schema cache
                fetch_neo4j_schema()
                
                return formatted_response, None
        
        # Fallback formatting
        formatted_text = json.dumps(result_data, indent=2, default=str) if isinstance(result_data, (dict, list)) else str(result_data)
        return formatted_text, None
    
    except Exception as e:
        error_msg = f"❌ **Error formatting response:** {str(e)}"
        logger.error(error_msg)
        return error_msg, None

def select_tool_node_enhanced(state: AgentState) -> dict:
    """Enhanced tool selection with schema awareness"""
    logger.info(f"🤔 Processing question: {state.question}")
    
    try:
        # Ensure schema is loaded
        if not SCHEMA_CACHE.get("last_updated"):
            fetch_neo4j_schema()
        
        llm_output = cortex_llm(state.question, state.session_id)
        tool, query, trace = parse_llm_output_enhanced(llm_output, state.question)
        
        logger.info(f"✅ Enhanced tool selection - Tool: {tool}, Query: {query[:100] if query else 'None'}")
        
        return {
            "question": state.question,
            "session_id": state.session_id,
            "tool": tool or "",
            "query": query or "",
            "trace": trace or "",
            "answer": "",
            "graph_data": None,
            "schema_info": SCHEMA_CACHE,
            "node_limit": state.node_limit
        }
    except Exception as e:
        logger.error(f"❌ Error in enhanced select_tool_node: {e}")
        return {
            "question": state.question,
            "session_id": state.session_id,
            "tool": "",
            "query": "",
            "trace": f"Error selecting tool: {str(e)}",
            "answer": f"❌ Error processing question: {str(e)}",
            "graph_data": None,
            "schema_info": None,
            "node_limit": state.node_limit
        }

def execute_tool_node_enhanced(state: AgentState) -> dict:
    """Enhanced tool execution with better error handling and unlimited queries"""
    tool = state.tool
    query = state.query
    trace = state.trace
    question = state.question
    node_limit = state.node_limit
    answer = ""
    graph_data = None
    
    valid_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    
    logger.info(f"⚡ Executing enhanced tool: '{tool}'")
    logger.info(f"🔧 Query: {query[:200] if query else 'None'}...")
    
    try:
        if not tool:
            logger.error("❌ No tool selected")
            answer = "⚠️ I couldn't determine the right tool for your question. The schema suggests using read_neo4j_cypher for data exploration, write_neo4j_cypher for modifications, or get_neo4j_schema for structure information."
        
        elif tool not in valid_tools:
            logger.error(f"❌ Invalid tool: {tool}")
            answer = f"⚠️ Tool '{tool}' not recognized. Available tools: {', '.join(valid_tools)}"
        
        elif tool == "get_neo4j_schema":
            logger.info("📋 Executing enhanced schema retrieval...")
            result = requests.post("http://localhost:8000/get_neo4j_schema", headers=headers, timeout=30)
            if result.ok:
                # Also update local schema cache
                fetch_neo4j_schema()
                answer, graph_data = format_response_with_graph_enhanced(result.json(), tool, question)
                logger.info("✅ Enhanced schema retrieval successful")
            else:
                logger.error(f"❌ Schema query failed: {result.status_code} - {result.text}")
                answer = f"❌ Schema query failed: {result.text}"
        
        elif tool == "read_neo4j_cypher":
            if not query or not query.strip():
                logger.error("❌ No query provided for read operation")
                # Generate default exploration query using schema
                labels = SCHEMA_CACHE.get("labels", [])
                if labels:
                    query = f"MATCH (n:{labels[0]}) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m"
                    logger.info(f"🔧 Generated default exploration query: {query}")
                    answer = "⚠️ No specific query provided. Generated a schema-based exploration query."
                else:
                    query = "MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m"
                    answer = "⚠️ No specific query provided. Generated a general exploration query."
            
            if query:
                logger.info("📖 Executing enhanced read query...")
                query_clean = clean_cypher_query(query)
                data = {
                    "query": query_clean, 
                    "params": {},
                    "node_limit": node_limit * 10  # Allow more data for better visualization
                }
                result = requests.post("http://localhost:8000/read_neo4j_cypher", json=data, headers=headers, timeout=60)
                if result.ok:
                    answer, graph_data = format_response_with_graph_enhanced(result.json(), tool, question)
                    logger.info("✅ Enhanced read query successful")
                else:
                    logger.error(f"❌ Read query failed: {result.status_code} - {result.text}")
                    answer = f"❌ Query failed: {result.text}"
        
        elif tool == "write_neo4j_cypher":
            if not query or not query.strip():
                logger.error("❌ No query provided for write operation")
                answer = "⚠️ I couldn't generate a valid modification query. Please be more specific about what you want to create, update, or delete. You can reference the available node types from the schema."
            else:
                logger.info("✏️ Executing enhanced write query...")
                query_clean = clean_cypher_query(query)
                data = {
                    "query": query_clean, 
                    "params": {},
                    "node_limit": node_limit * 10
                }
                result = requests.post("http://localhost:8000/write_neo4j_cypher", json=data, headers=headers, timeout=60)
                if result.ok:
                    answer, graph_data = format_response_with_graph_enhanced(result.json(), tool, question)
                    logger.info("✅ Enhanced write query successful")
                else:
                    logger.error(f"❌ Write query failed: {result.status_code} - {result.text}")
                    answer = f"❌ Update failed: {result.text}"
        
        else:
            logger.error(f"❌ Unknown tool: {tool}")
            answer = f"❌ Unknown tool: {tool}"
    
    except requests.exceptions.Timeout:
        logger.error("⏰ Request timed out")
        answer = "⚠️ Query timed out. The query might be complex or the database might be busy. Try again or simplify the query."
    except requests.exceptions.ConnectionError:
        logger.error("🔌 Connection error")
        answer = "⚠️ Cannot connect to the database server. Please ensure the MCP server is running on port 8000."
    except Exception as e:
        logger.error(f"💥 Unexpected error in enhanced execute_tool_node: {e}")
        answer = f"⚠️ Execution failed: {str(e)}"
    
    logger.info(f"🏁 Enhanced tool execution completed. Graph data: {'Yes' if graph_data else 'No'}")
    
    return {
        "question": state.question,
        "session_id": state.session_id,
        "tool": tool,
        "query": query,
        "trace": trace,
        "answer": answer,
        "graph_data": graph_data,
        "schema_info": SCHEMA_CACHE,
        "node_limit": node_limit
    }

def build_enhanced_agent():
    """Build enhanced LangGraph agent with schema awareness"""
    
    # Initialize schema on startup
    logger.info("🚀 Building enhanced LangGraph agent with schema awareness...")
    fetch_neo4j_schema()
    
    workflow = StateGraph(state_schema=AgentState)
    
    # Add enhanced nodes
    workflow.add_node("select_tool", RunnableLambda(select_tool_node_enhanced))
    workflow.add_node("execute_tool", RunnableLambda(execute_tool_node_enhanced))
    
    # Set entry point
    workflow.set_entry_point("select_tool")
    
    # Add edges
    workflow.add_edge("select_tool", "execute_tool")
    workflow.add_edge("execute_tool", END)
    
    # Compile and return
    agent = workflow.compile()
    logger.info("🚀 Enhanced LangGraph agent built successfully with schema integration!")
    return agent

# Export function for use in app.py
def build_agent():
    """Main function to build the enhanced agent"""
    return build_enhanced_agent()

# For testing purposes
if __name__ == "__main__":
    # Test the enhanced agent
    agent = build_enhanced_agent()
    test_state = AgentState(
        question="Show me the complete network structure with all relationships",
        session_id="test_session_enhanced",
        node_limit=1000
    )
    
    import asyncio
    
    async def test():
        result = await agent.ainvoke(test_state)
        print("Enhanced Test Result:", result)
    
    # asyncio.run(test())
