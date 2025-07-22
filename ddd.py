"""
Enhanced LangGraph Agent with Real-time Neo4j NVL Support
This agent provides enhanced Neo4j operations with real-time visualization updates
"""

import requests
import urllib3
import json
import logging
import re
import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced_langgraph_agent")

# ============================================
# 🔧 CONFIGURATION - CHANGE THESE VALUES
# ============================================

# Cortex API Configuration
CORTEX_API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
CORTEX_API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"  # ⚠️ CHANGE THIS!
CORTEX_MODEL = "claude-4-sonnet"

# MCP Server Configuration
MCP_SERVER_PORT = 8000
MCP_BASE_URL = f"http://localhost:{MCP_SERVER_PORT}"

# ============================================

print("🔧 Enhanced LangGraph Agent Configuration:")
print(f"   Cortex API: {CORTEX_API_URL}")
print(f"   API Key Length: {len(CORTEX_API_KEY)} characters")
print(f"   Model: {CORTEX_MODEL}")
print(f"   MCP Server: {MCP_BASE_URL}")

class AgentState(BaseModel):
    question: str
    session_id: str
    intent: str = ""
    tool: str = ""
    query: str = ""
    raw_response: Dict[str, Any] = {}
    formatted_answer: str = ""
    trace: str = ""
    debug_info: str = ""
    error_count: int = 0
    last_error: str = ""
    # Enhanced fields for real-time NVL
    operation_type: str = ""  # create, read, update, delete, schema
    affected_nodes: int = 0
    affected_relationships: int = 0
    graph_changes: Dict[str, Any] = {}
    visualization_metadata: Dict[str, Any] = {}

# ============================================
# ENHANCED SYSTEM PROMPT WITH NVL AWARENESS
# ============================================

ENHANCED_SYSTEM_PROMPT = """
You are an expert AI assistant that helps users query and manage a Neo4j database with real-time visualization support using the Neo4j Visualization Library (NVL). Your responses will be used to update live graph visualizations, so be precise and comprehensive.

AVAILABLE TOOLS:
- read_neo4j_cypher: Execute read-only queries (MATCH, RETURN, WHERE, OPTIONAL MATCH, etc.)
  * Use for data exploration, analysis, counting, reporting
  * Results will be displayed in both text and real-time NVL graph visualization
  * NEVER use for CREATE, UPDATE, DELETE operations
  * Return meaningful data for graph visualization

- write_neo4j_cypher: Execute write operations (CREATE, MERGE, SET, DELETE, REMOVE, etc.)
  * Use for data modification, node/relationship creation/deletion
  * Changes will be reflected immediately in the live NVL visualization
  * Returns detailed operation counts for visualization updates
  * Triggers real-time graph updates via WebSocket

- get_neo4j_schema: Get database structure information
  * Use when users ask about schema, labels, relationship types, properties
  * Results help users understand the graph structure for visualization
  * Provides comprehensive schema information for NVL rendering

RESPONSE FORMAT REQUIREMENTS:
Always use this EXACT format:

Tool: [exact_tool_name]
Query: [complete_cypher_query_on_single_line]

IMPORTANT RULES:
1. Put the ENTIRE Cypher query on ONE line after "Query:"
2. Do NOT use code blocks, markdown, or multi-line formatting
3. Use exact tool names: read_neo4j_cypher, write_neo4j_cypher, get_neo4j_schema
4. For write operations, use DETACH DELETE for node deletions
5. Always provide complete, executable queries
6. Consider NVL visualization impact - queries should return meaningful graph data
7. For CREATE operations, return created nodes/relationships for immediate visualization
8. Use meaningful node properties for better NVL rendering (name, title, etc.)

ENHANCED EXAMPLES FOR REAL-TIME NVL:

User: How many nodes are in the graph?
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN count(n) as total_nodes

User: Show me all Person nodes with their connections
Tool: read_neo4j_cypher
Query: MATCH (p:Person)-[r]-(connected) RETURN p, r, connected LIMIT 20

User: Create a Person named Alice with age 30 who works at TechCorp
Tool: write_neo4j_cypher
Query: CREATE (alice:Person {name: 'Alice', age: 30, created: datetime()}), (company:Company {name: 'TechCorp', type: 'Technology'}) CREATE (alice)-[:WORKS_FOR {since: date(), position: 'Engineer'}]->(company) RETURN alice, company

User: Delete all TestNode nodes and their relationships
Tool: write_neo4j_cypher
Query: MATCH (n:TestNode) DETACH DELETE n

User: What's the database structure and schema?
Tool: get_neo4j_schema

User: Find the most connected nodes for visualization
Tool: read_neo4j_cypher
Query: MATCH (n) WITH n, size((n)--()) as connections WHERE connections > 0 RETURN n, connections ORDER BY connections DESC LIMIT 10

User: Create a social network example with 3 people
Tool: write_neo4j_cypher
Query: CREATE (alice:Person {name: 'Alice', age: 30, city: 'New York'}), (bob:Person {name: 'Bob', age: 25, city: 'San Francisco'}), (charlie:Person {name: 'Charlie', age: 35, city: 'Chicago'}) CREATE (alice)-[:FRIENDS_WITH {since: '2020-01-15', strength: 'close'}]->(bob), (bob)-[:FRIENDS_WITH {since: '2021-03-22', strength: 'casual'}]->(charlie), (charlie)-[:FRIENDS_WITH {since: '2019-11-08', strength: 'close'}]->(alice) RETURN alice, bob, charlie

User: Update all Person nodes to add a 'last_seen' property
Tool: write_neo4j_cypher
Query: MATCH (p:Person) SET p.last_seen = datetime() RETURN count(p) as updated_count

User: Show me all relationships between Company and Person nodes
Tool: read_neo4j_cypher
Query: MATCH (p:Person)-[r]-(c:Company) RETURN p, r, c LIMIT 25

REAL-TIME NVL VISUALIZATION CONSIDERATIONS:
- For read queries, return nodes and relationships when possible for graph display
- For write operations, return created/modified elements to show in visualization
- Consider query performance for visualization (use LIMIT for large result sets)
- Structure queries to provide meaningful graph data for the NVL interface
- Use descriptive property names that will render well in NVL (name, title, caption)
- For CREATE operations, include properties that help with node identification
- For relationship creation, include meaningful relationship types and properties

ERROR HANDLING:
- If query syntax is unclear, ask for clarification
- For ambiguous requests, suggest the most visualization-friendly interpretation
- Always validate Cypher syntax before responding
- Provide helpful error messages that guide users towards successful queries
"""

# ============================================
# ENHANCED LLM COMMUNICATION
# ============================================

def call_cortex_llm_enhanced(prompt: str, session_id: str) -> str:
    """Enhanced Cortex LLM call with better error handling and retry logic"""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            headers = {
                "Authorization": f'Snowflake Token="{CORTEX_API_KEY}"',
                "Content-Type": "application/json"
            }
            
            payload = {
                "query": {
                    "aplctn_cd": "edagnai",
                    "app_id": "edadip",
                    "api_key": CORTEX_API_KEY,
                    "method": "cortex",
                    "model": CORTEX_MODEL,
                    "sys_msg": ENHANCED_SYSTEM_PROMPT,
                    "limit_convs": "0",
                    "prompt": {
                        "messages": [{"role": "user", "content": prompt}]
                    },
                    "session_id": session_id
                }
            }
            
            logger.info(f"Enhanced LLM call (attempt {attempt + 1}): {prompt[:100]}...")
            response = requests.post(CORTEX_API_URL, headers=headers, json=payload, verify=False, timeout=45)
            
            if response.status_code == 200:
                result = response.text.partition("end_of_stream")[0].strip()
                logger.info(f"Enhanced LLM response received: {len(result)} characters")
                return result
            else:
                logger.error(f"Cortex API error: {response.status_code}")
                if attempt == max_retries - 1:
                    return f"Error: Cortex API returned {response.status_code}"
                    
        except Exception as e:
            logger.error(f"Enhanced LLM call failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return f"Error: Failed to call enhanced LLM after {max_retries} attempts - {str(e)}"
            
        # Wait before retry
        time.sleep(retry_delay * (attempt + 1))
    
    return "Error: Maximum retry attempts exceeded"

# ============================================
# ENHANCED INTENT AND OPERATION DETECTION
# ============================================

def determine_enhanced_intent_and_operation(question: str) -> Tuple[str, str]:
    """Enhanced intent and operation type determination for NVL visualization"""
    q_lower = question.lower()
    
    # Schema operations
    if any(word in q_lower for word in ['schema', 'structure', 'labels', 'relationships', 'types', 'properties', 'describe database']):
        return "schema_inquiry", "schema"
    
    # Write operations with more granular detection
    elif any(word in q_lower for word in ['create', 'add', 'insert', 'new', 'make', 'build']):
        if any(word in q_lower for word in ['person', 'people', 'user', 'employee']):
            return "create_person", "create"
        elif any(word in q_lower for word in ['company', 'organization', 'business']):
            return "create_company", "create"
        elif any(word in q_lower for word in ['relationship', 'connection', 'link', 'connect']):
            return "create_relationship", "create"
        else:
            return "create_generic", "create"
    
    elif any(word in q_lower for word in ['delete', 'remove', 'drop', 'clear', 'destroy']):
        if 'all' in q_lower:
            return "delete_all", "delete"
        elif any(word in q_lower for word in ['test', 'temp', 'temporary']):
            return "delete_test", "delete"
        else:
            return "delete_specific", "delete"
    
    elif any(word in q_lower for word in ['update', 'set', 'change', 'modify', 'edit', 'alter']):
        return "update_data", "update"
    
    elif any(word in q_lower for word in ['merge', 'upsert']):
        return "merge_data", "create"
    
    # Read operations with enhanced granularity
    elif any(word in q_lower for word in ['how many', 'count', 'number of', 'total']):
        return "count_query", "read"
    
    elif any(word in q_lower for word in ['show', 'list', 'find', 'get', 'select', 'display', 'retrieve']):
        if any(word in q_lower for word in ['connection', 'relationship', 'link', 'connected']):
            return "show_relationships", "read"
        elif any(word in q_lower for word in ['person', 'people', 'user']):
            return "show_people", "read"
        else:
            return "show_data", "read"
    
    elif any(word in q_lower for word in ['analyze', 'analysis', 'report', 'summary']):
        return "analyze_data", "read"
    
    elif any(word in q_lower for word in ['most connected', 'highly connected', 'popular', 'central']):
        return "find_central_nodes", "read"
    
    # Default to exploration
    else:
        return "explore_database", "read"

def parse_enhanced_llm_response(llm_output: str, question: str) -> Tuple[str, str, str, Dict[str, Any]]:
    """Enhanced LLM response parsing with comprehensive error handling and NVL support"""
    valid_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    
    debug_info = {
        "raw_output": llm_output[:400],
        "question": question,
        "parsing_steps": [],
        "enhanced_features": ["nv_support", "real_time_updates", "visualization_metadata"]
    }
    
    tool = None
    query = None
    trace = llm_output.strip()
    
    # Enhanced Strategy 1: Look for exact "Tool: xxx" pattern with variations
    tool_patterns = [
        r"Tool:\s*([a-zA-Z_]+)",
        r"tool:\s*([a-zA-Z_]+)",
        r"Tool\s*=\s*([a-zA-Z_]+)",
        r"Selected\s+tool:\s*([a-zA-Z_]+)",
        r"Using\s+tool:\s*([a-zA-Z_]+)"
    ]
    
    for pattern in tool_patterns:
        tool_match = re.search(pattern, llm_output, re.I)
        if tool_match:
            extracted_tool = tool_match.group(1).strip()
            if extracted_tool in valid_tools:
                tool = extracted_tool
                debug_info["parsing_steps"].append(f"Found tool via pattern '{pattern}': {tool}")
                break
    
    # Enhanced Strategy 2: Look for tool names in text with context
    if not tool:
        for valid_tool in valid_tools:
            if valid_tool.lower() in llm_output.lower():
                tool = valid_tool
                debug_info["parsing_steps"].append(f"Found tool in text: {tool}")
                break
    
    # Enhanced Strategy 3: Infer from question content with enhanced logic
    if not tool:
        intent, operation_type = determine_enhanced_intent_and_operation(question)
        
        if operation_type == "schema":
            tool = "get_neo4j_schema"
        elif operation_type in ["create", "delete", "update"]:
            tool = "write_neo4j_cypher"
        else:
            tool = "read_neo4j_cypher"
            
        debug_info["parsing_steps"].append(f"Enhanced inference: {tool} (intent: {intent}, operation: {operation_type})")
    
    # Enhanced Strategy 4: Extract query with multiple patterns
    query_patterns = [
        r"Query:\s*(.+?)(?=\n|$)",
        r"query:\s*(.+?)(?=\n|$)",
        r"Cypher:\s*(.+?)(?=\n|$)",
        r"cypher:\s*(.+?)(?=\n|$)",
        r"`([^`]+)`",
        r"```\s*cypher\s*([^```]+)\s*```",
        r"```([^```]+)```"
    ]
    
    for pattern in query_patterns:
        match = re.search(pattern, llm_output, re.I | re.MULTILINE | re.DOTALL)
        if match:
            extracted_query = match.group(1).strip()
            # Enhanced query cleaning
            extracted_query = re.sub(r'```[a-zA-Z]*', '', extracted_query)
            extracted_query = re.sub(r'```', '', extracted_query)
            extracted_query = re.sub(r'\s+', ' ', extracted_query).strip()
            
            # Validate query has Cypher keywords
            if len(extracted_query) > 5 and any(keyword in extracted_query.upper() for keyword in ['MATCH', 'CREATE', 'DELETE', 'MERGE', 'SET', 'RETURN', 'CALL', 'WITH']):
                query = extracted_query
                debug_info["parsing_steps"].append(f"Extracted valid query via pattern: {pattern}")
                break
    
    # Enhanced Strategy 5: Generate intelligent fallback queries
    if tool and not query and tool != "get_neo4j_schema":
        intent, operation_type = determine_enhanced_intent_and_operation(question)
        
        if tool == "read_neo4j_cypher":
            if "count" in intent:
                query = "MATCH (n) RETURN count(n) as total_nodes"
            elif "relationships" in intent:
                query = "MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 20"
            elif "people" in intent or "person" in intent:
                query = "MATCH (p:Person) RETURN p LIMIT 15"
            elif "central" in intent or "connected" in intent:
                query = "MATCH (n) WITH n, size((n)--()) as connections WHERE connections > 0 RETURN n, connections ORDER BY connections DESC LIMIT 10"
            else:
                query = "MATCH (n) RETURN n LIMIT 15"
                
        elif tool == "write_neo4j_cypher":
            if "person" in operation_type or "person" in question.lower():
                name_match = re.search(r'named?\s+([A-Za-z]+)', question, re.I)
                name = name_match.group(1) if name_match else 'TestUser'
                query = f"CREATE (p:Person {{name: '{name}', created: datetime()}}) RETURN p"
            elif "company" in operation_type or "company" in question.lower():
                company_match = re.search(r'(?:company|organization|business)(?:\s+called|\s+named)?\s+([A-Za-z]+)', question, re.I)
                company = company_match.group(1) if company_match else 'TestCorp'
                query = f"CREATE (c:Company {{name: '{company}', created: datetime()}}) RETURN c"
            elif "delete" in operation_type:
                if "test" in question.lower():
                    query = "MATCH (n:TestNode) DETACH DELETE n"
                elif "all" in question.lower():
                    query = "MATCH (n) DETACH DELETE n"
                else:
                    query = "MATCH (n) WHERE n.name = 'test' DETACH DELETE n"
            else:
                query = "CREATE (n:TestNode {name: 'test', created: datetime()}) RETURN n"
        
        debug_info["parsing_steps"].append(f"Generated enhanced fallback query: {query}")
    
    # Final validation and safe defaults
    if not tool:
        tool = "read_neo4j_cypher"
        query = "MATCH (n) RETURN count(n) as total"
        debug_info["parsing_steps"].append("Used safe default: read_neo4j_cypher")
    
    debug_info["final_result"] = {"tool": tool, "query": query, "trace_length": len(trace)}
    
    return tool, query, trace, debug_info

# ============================================
# ENHANCED MCP SERVER COMMUNICATION
# ============================================

async def call_mcp_server_enhanced(tool: str, query: str = None) -> Dict[str, Any]:
    """Enhanced MCP server communication with comprehensive error handling and NVL support"""
    try:
        headers = {"Content-Type": "application/json"}
        
        logger.info(f"Enhanced MCP call - Tool: {tool}, Query: {query}")
        
        if tool == "get_neo4j_schema":
            response = requests.post(f"{MCP_BASE_URL}/get_neo4j_schema", headers=headers, timeout=30)
            
        elif tool == "read_neo4j_cypher":
            if not query:
                return {"success": False, "error": "No query provided for read operation"}
            
            data = {"query": query, "params": {}}
            response = requests.post(f"{MCP_BASE_URL}/read_neo4j_cypher", json=data, headers=headers, timeout=30)
            
        elif tool == "write_neo4j_cypher":
            if not query:
                return {"success": False, "error": "No query provided for write operation"}
                
            data = {"query": query, "params": {}}
            response = requests.post(f"{MCP_BASE_URL}/write_neo4j_cypher", json=data, headers=headers, timeout=30)
            
        else:
            return {"success": False, "error": f"Unknown tool: {tool}"}
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Enhanced MCP response successful: {type(result)}")
            
            # Enhanced response processing for NVL
            if isinstance(result, list):
                # For read operations that return lists
                return {
                    "success": True,
                    "data": result,
                    "count": len(result),
                    "type": "list_result",
                    "visualization_ready": True
                }
            elif isinstance(result, dict):
                # For write operations and schema
                return {
                    "success": True,
                    "data": result,
                    "type": "dict_result",
                    "visualization_ready": True,
                    **result  # Merge any additional fields
                }
            else:
                return {
                    "success": True, 
                    "data": result, 
                    "type": "other",
                    "visualization_ready": False
                }
                
        else:
            logger.error(f"Enhanced MCP server error: {response.status_code} - {response.text}")
            return {
                "success": False, 
                "error": f"MCP server error: {response.status_code}",
                "details": response.text,
                "visualization_ready": False
            }
            
    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to enhanced MCP server at {MCP_BASE_URL}")
        return {
            "success": False, 
            "error": f"Cannot connect to enhanced MCP server. Is it running on port {MCP_SERVER_PORT}?",
            "connection_failed": True,
            "visualization_ready": False
        }
    except Exception as e:
        logger.error(f"Enhanced MCP server call failed: {e}")
        return {
            "success": False, 
            "error": f"Enhanced MCP server failed: {str(e)}",
            "visualization_ready": False
        }

# ============================================
# ENHANCED RESPONSE FORMATTING WITH NVL METADATA
# ============================================

def format_enhanced_response_with_nvl(tool: str, query: str, response: Dict[str, Any], debug_info: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
    """Enhanced response formatting with comprehensive NVL visualization metadata"""
    
    visualization_metadata = {
        "tool_used": tool,
        "query_executed": query,
        "operation_type": "unknown",
        "nodes_affected": 0,
        "relationships_affected": 0,
        "visualization_update_needed": False,
        "nvl_compatible": True,
        "real_time_update": False,
        "graph_changes": {}
    }
    
    if not response.get("success", False):
        error = response.get("error", "Unknown error")
        formatted_response = f"❌ **Enhanced Operation Failed**\n\n**Error:** {error}"
        
        if debug_info:
            formatted_response += f"\n\n**🔧 Enhanced Debug Information:**\n"
            for step in debug_info.get("parsing_steps", []):
                formatted_response += f"• {step}\n"
        
        visualization_metadata["nvl_compatible"] = False
        return formatted_response, visualization_metadata
    
    data = response.get("data", {})
    
    # Enhanced schema operation handling
    if tool == "get_neo4j_schema":
        visualization_metadata["operation_type"] = "schema"
        
        if isinstance(data, dict):
            labels = data.get("labels", [])
            rel_types = data.get("relationship_types", [])
            prop_keys = data.get("property_keys", [])
            label_samples = data.get("label_samples", {})
            
            formatted_response = f"""📊 **Enhanced Database Schema Retrieved**

**🏷️ Node Labels ({len(labels)}):**
{', '.join(labels[:25])}

**🔗 Relationship Types ({len(rel_types)}):**
{', '.join(rel_types[:25])}

**🔑 Property Keys ({len(prop_keys)}):**
{', '.join(prop_keys[:30])}"""

            if label_samples:
                formatted_response += f"\n\n**📋 Sample Data Available:** {len(label_samples)} label types with examples"

            formatted_response += "\n\n*Enhanced schema information updated for NVL visualization interface*"

        else:
            formatted_response = f"📊 **Enhanced Database Schema:**\n```json\n{json.dumps(data, indent=2)[:1000]}\n```"
        
        return formatted_response, visualization_metadata
    
    # Enhanced read operation handling
    elif tool == "read_neo4j_cypher":
        visualization_metadata["operation_type"] = "read"
        visualization_metadata["visualization_update_needed"] = True
        visualization_metadata["real_time_update"] = True
        
        if isinstance(data, list):
            count = len(data)
            visualization_metadata["nodes_affected"] = count
            
            if count == 0:
                formatted_response = "📊 **Enhanced Query Result:** No data found in the database"
            elif count == 1 and isinstance(data[0], dict) and len(data[0]) == 1:
                # Single value result (like count)
                key, value = list(data[0].items())[0]
                formatted_response = f"📊 **Enhanced Query Result:**\n\n**{key.replace('_', ' ').title()}:** {value:,}"
                if key in ["total_nodes", "total", "count"]:
                    formatted_response += f"\n\n🎯 **Visualization Status:** {value:,} items available for NVL rendering"
            else:
                formatted_response = f"📊 **Enhanced Query Results:** Found **{count:,}** records\n\n"
                
                # Enhanced sample display
                sample_size = min(5, count)
                for i, record in enumerate(data[:sample_size]):
                    formatted_response += f"**🔍 Record {i+1}:**\n"
                    if isinstance(record, dict):
                        for k, v in record.items():
                            # Format values for better display
                            if isinstance(v, dict) and 'properties' in str(v):
                                formatted_response += f"  • {k}: Node with properties\n"
                            elif isinstance(v, list):
                                formatted_response += f"  • {k}: [{len(v)} items]\n"
                            else:
                                formatted_response += f"  • {k}: {v}\n"
                    else:
                        formatted_response += f"  {record}\n"
                    formatted_response += "\n"
                
                if count > sample_size:
                    formatted_response += f"... and **{count - sample_size:,}** more records\n\n"
                
                formatted_response += "*🎨 Full results displayed in enhanced NVL visualization with real-time updates*"
        else:
            formatted_response = f"📊 **Enhanced Query Result:**\n```json\n{json.dumps(data, indent=2)[:800]}\n```"
    
    # Enhanced write operation handling
    elif tool == "write_neo4j_cypher":
        visualization_metadata["operation_type"] = "write"
        visualization_metadata["visualization_update_needed"] = True
        visualization_metadata["real_time_update"] = True
        
        if isinstance(data, dict):
            # Extract comprehensive operation counts
            nodes_created = data.get("nodes_created", 0)
            nodes_deleted = data.get("nodes_deleted", 0)
            rels_created = data.get("relationships_created", 0)
            rels_deleted = data.get("relationships_deleted", 0)
            props_set = data.get("properties_set", 0)
            labels_added = data.get("labels_added", 0)
            labels_removed = data.get("labels_removed", 0)
            
            visualization_metadata["nodes_affected"] = nodes_created + nodes_deleted
            visualization_metadata["relationships_affected"] = rels_created + rels_deleted
            visualization_metadata["graph_changes"] = {
                "nodes_created": nodes_created,
                "nodes_deleted": nodes_deleted,
                "relationships_created": rels_created,
                "relationships_deleted": rels_deleted,
                "properties_set": props_set
            }
            
            operations = []
            if nodes_created > 0:
                operations.append(f"🟢 **Created {nodes_created:,} node{'s' if nodes_created != 1 else ''}**")
            if nodes_deleted > 0:
                operations.append(f"🗑️ **Deleted {nodes_deleted:,} node{'s' if nodes_deleted != 1 else ''}**")
            if rels_created > 0:
                operations.append(f"🔗 **Created {rels_created:,} relationship{'s' if rels_created != 1 else ''}**")
            if rels_deleted > 0:
                operations.append(f"❌ **Deleted {rels_deleted:,} relationship{'s' if rels_deleted != 1 else ''}**")
            if props_set > 0:
                operations.append(f"📝 **Set {props_set:,} propert{'ies' if props_set != 1 else 'y'}**")
            if labels_added > 0:
                operations.append(f"🏷️ **Added {labels_added:,} label{'s' if labels_added != 1 else ''}**")
            if labels_removed > 0:
                operations.append(f"🏷️ **Removed {labels_removed:,} label{'s' if labels_removed != 1 else ''}**")
            
            if operations:
                formatted_response = "✅ **Enhanced Database Update Completed Successfully!**\n\n"
                formatted_response += "\n".join(operations)
                formatted_response += "\n\n🎨 **Real-time Visualization:** Changes immediately reflected in enhanced NVL graph with WebSocket updates"
                
                # Add performance and impact info
                total_changes = sum([nodes_created, nodes_deleted, rels_created, rels_deleted, props_set])
                if total_changes > 0:
                    formatted_response += f"\n\n**📊 Total Changes:** {total_changes:,}"
                    formatted_response += f"\n**🔄 Update Impact:** {visualization_metadata['nodes_affected']} nodes, {visualization_metadata['relationships_affected']} relationships affected"
            else:
                formatted_response = "✅ **Enhanced query executed successfully** (no structural changes made to the graph)"
        else:
            formatted_response = f"✅ **Enhanced Write Operation Result:**\n{json.dumps(data, indent=2)[:500]}"
    
    else:
        formatted_response = f"📊 **Enhanced Operation Result:**\n{json.dumps(data, indent=2)[:600]}"
    
    return formatted_response, visualization_metadata

# ============================================
# ENHANCED LANGGRAPH NODES
# ============================================

def analyze_and_select_tool_node_enhanced(state: AgentState) -> Dict[str, Any]:
    """Enhanced Node 1: Analyze question and select tool with comprehensive NVL support"""
    logger.info(f"Enhanced processing with NVL support: {state.question}")
    
    # Enhanced intent and operation type determination
    intent, operation_type = determine_enhanced_intent_and_operation(state.question)
    
    # Call enhanced LLM
    llm_output = call_cortex_llm_enhanced(state.question, state.session_id)
    
    # Enhanced parsing with NVL support
    tool, query, trace, debug_info = parse_enhanced_llm_response(llm_output, state.question)
    
    logger.info(f"Enhanced selection - Tool: {tool}, Query: {query}, Intent: {intent}")
    
    return {
        "question": state.question,
        "session_id": state.session_id,
        "intent": intent,
        "tool": tool,
        "query": query,
        "trace": trace,
        "debug_info": json.dumps(debug_info, indent=2),
        "raw_response": {},
        "formatted_answer": "",
        "error_count": state.error_count,
        "last_error": state.last_error,
        "operation_type": operation_type,
        "affected_nodes": 0,
        "affected_relationships": 0,
        "graph_changes": {},
        "visualization_metadata": {
            "intent": intent,
            "operation_type": operation_type,
            "nvl_ready": True
        }
    }

async def execute_tool_node_enhanced(state: AgentState) -> Dict[str, Any]:
    """Enhanced Node 2: Execute tool with comprehensive NVL visualization support"""
    logger.info(f"Enhanced execution with NVL support: {state.tool}")
    
    if not state.tool:
        formatted_answer = f"⚠️ **No valid tool selected for enhanced operation.**\n\n**🔧 Enhanced Debug Info:**\n{state.debug_info}"
        return {**state.dict(), "formatted_answer": formatted_answer}
    
    # Call enhanced MCP server
    raw_response = await call_mcp_server_enhanced(state.tool, state.query)
    
    # Parse enhanced debug info
    try:
        debug_info = json.loads(state.debug_info) if state.debug_info else {}
    except:
        debug_info = {}
    
    # Enhanced response formatting with comprehensive NVL metadata
    formatted_answer, viz_metadata = format_enhanced_response_with_nvl(
        state.tool, state.query, raw_response, debug_info
    )
    
    # Enhanced state update with comprehensive visualization metadata
    return {
        **state.dict(),
        "raw_response": raw_response,
        "formatted_answer": formatted_answer,
        "affected_nodes": viz_metadata.get("nodes_affected", 0),
        "affected_relationships": viz_metadata.get("relationships_affected", 0),
        "graph_changes": viz_metadata.get("graph_changes", {}),
        "visualization_metadata": viz_metadata
    }

# ============================================
# BUILD ENHANCED AGENT
# ============================================

def build_agent():
    """Build the enhanced LangGraph agent with comprehensive NVL support"""
    logger.info("Building enhanced LangGraph agent with comprehensive NVL support...")
    
    workflow = StateGraph(state_schema=AgentState)
    
    # Add enhanced nodes
    workflow.add_node("analyze_and_select_enhanced", RunnableLambda(analyze_and_select_tool_node_enhanced))
    workflow.add_node("execute_tool_enhanced", RunnableLambda(execute_tool_node_enhanced))
    
    # Set entry point and edges
    workflow.set_entry_point("analyze_and_select_enhanced")
    workflow.add_edge("analyze_and_select_enhanced", "execute_tool_enhanced")
    workflow.add_edge("execute_tool_enhanced", END)
    
    agent = workflow.compile()
    logger.info("✅ Enhanced LangGraph agent with comprehensive NVL support built successfully")
    
    return agent

# ============================================
# ENHANCED TESTING AND VALIDATION
# ============================================

def test_enhanced_agent():
    """Test the enhanced agent with comprehensive scenarios"""
    test_cases = [
        {
            "question": "How many nodes are in the graph?",
            "expected_tool": "read_neo4j_cypher",
            "expected_operation": "read",
            "expected_intent": "count_query"
        },
        {
            "question": "Create a Person named Alice with age 30 who works at TechCorp",
            "expected_tool": "write_neo4j_cypher",
            "expected_operation": "create",
            "expected_intent": "create_person"
        },
        {
            "question": "Show me the enhanced database schema with all details",
            "expected_tool": "get_neo4j_schema",
            "expected_operation": "schema",
            "expected_intent": "schema_inquiry"
        },
        {
            "question": "Delete all TestNode nodes and their relationships",
            "expected_tool": "write_neo4j_cypher",
            "expected_operation": "delete",
            "expected_intent": "delete_test"
        },
        {
            "question": "Find all Person nodes with their relationships for visualization",
            "expected_tool": "read_neo4j_cypher",
            "expected_operation": "read",
            "expected_intent": "show_relationships"
        },
        {
            "question": "Create a social network with 3 connected people",
            "expected_tool": "write_neo4j_cypher",
            "expected_operation": "create",
            "expected_intent": "create_generic"
        }
    ]
    
    print("🧪 Testing Enhanced LangGraph Agent with Comprehensive NVL Support")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['question']}")
        
        # Test enhanced parsing without LLM call
        tool, query, trace, debug_info = parse_enhanced_llm_response("", test_case['question'])
        intent, operation_type = determine_enhanced_intent_and_operation(test_case['question'])
        
        print(f"   Expected Tool: {test_case['expected_tool']}")
        print(f"   Actual Tool: {tool}")
        print(f"   Expected Operation: {test_case['expected_operation']}")
        print(f"   Actual Operation: {operation_type}")
        print(f"   Expected Intent: {test_case['expected_intent']}")
        print(f"   Actual Intent: {intent}")
        print(f"   Generated Query: {query}")
        print(f"   Tool Match: {'✅' if tool == test_case['expected_tool'] else '❌'}")
        print(f"   Operation Match: {'✅' if operation_type == test_case['expected_operation'] else '❌'}")

async def test_complete_enhanced_agent():
    """Test the complete enhanced agent with MCP integration"""
    agent = build_agent()
    
    test_question = "How many nodes are in the enhanced Neo4j graph?"
    print(f"\n🧪 Testing complete enhanced agent with: {test_question}")
    
    try:
        state = AgentState(
            question=test_question,
            session_id="enhanced_test_session"
        )
        
        result = await agent.ainvoke(state)
        
        print(f"Enhanced Tool: {result.get('tool', 'N/A')}")
        print(f"Enhanced Query: {result.get('query', 'N/A')}")
        print(f"Operation Type: {result.get('operation_type', 'N/A')}")
        print(f"Affected Nodes: {result.get('affected_nodes', 0)}")
        print(f"Enhanced Answer: {result.get('formatted_answer', 'N/A')[:300]}...")
        print(f"Visualization Metadata: {result.get('visualization_metadata', {})}")
        
    except Exception as e:
        print(f"❌ Enhanced agent error: {e}")

# ============================================
# ENHANCED UTILITY FUNCTIONS
# ============================================

def validate_enhanced_cypher_query(query: str) -> Tuple[bool, str]:
    """Enhanced Cypher query validation with NVL considerations"""
    if not query or not query.strip():
        return False, "Empty query"
    
    query_upper = query.upper().strip()
    
    # Check for basic Cypher keywords
    valid_starters = ['MATCH', 'CREATE', 'MERGE', 'DELETE', 'SET', 'REMOVE', 'RETURN', 'CALL', 'WITH', 'UNWIND']
    if not any(query_upper.startswith(starter) for starter in valid_starters):
        return False, "Query doesn't start with valid Cypher keyword"
    
    # Check for potentially dangerous operations
    dangerous_patterns = ['DROP', 'ALTER', 'TRUNCATE']
    if any(pattern in query_upper for pattern in dangerous_patterns):
        return False, f"Query contains potentially dangerous operations"
    
    # Enhanced validation for NVL compatibility
    if 'CREATE' in query_upper and 'RETURN' not in query_upper:
        return True, "Valid CREATE query (consider adding RETURN for better visualization)"
    
    return True, "Valid enhanced query"

def get_enhanced_query_complexity_estimate(query: str) -> str:
    """Enhanced query complexity estimation for NVL performance"""
    if not query:
        return "unknown"
    
    query_upper = query.upper()
    
    # Enhanced complexity heuristics
    if 'LIMIT' in query_upper:
        limit_match = re.search(r'LIMIT\s+(\d+)', query_upper)
        if limit_match:
            limit_value = int(limit_match.group(1))
            if limit_value <= 20:
                return "low"
            elif limit_value <= 100:
                return "medium"
            else:
                return "high"
        return "low"
    elif any(pattern in query_upper for pattern in ['COLLECT', 'UNWIND', 'REDUCE']):
        return "high"
    elif any(pattern in query_upper for pattern in ['COUNT', 'AVG', 'SUM', 'MAX', 'MIN']):
        return "medium"
    elif any(pattern in query_upper for pattern in ['ALL', 'EXISTS', 'SHORTEST']):
        return "high"
    else:
        return "medium"

if __name__ == "__main__":
    print("🚀 Enhanced LangGraph Agent with Comprehensive NVL Support")
    print("=" * 80)
    
    # Run enhanced tests
    test_enhanced_agent()
    
    # Test complete enhanced agent (requires MCP server)
    print("\n" + "=" * 80)
    asyncio.run(test_complete_enhanced_agent())
