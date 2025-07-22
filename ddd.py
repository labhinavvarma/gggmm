"""
Fixed LangGraph Agent with Enhanced Neo4j Operations and Visualization Support
This agent properly handles CREATE/DELETE counts and integrates with the visualization system
"""

import requests
import urllib3
import json
import logging
import re
import time
import asyncio
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fixed_langgraph_agent")

# ============================================
# 🔧 CONFIGURATION - CHANGE THESE VALUES
# ============================================

# Cortex API Configuration
CORTEX_API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
CORTEX_API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"  # ⚠️ CHANGE THIS!
CORTEX_MODEL = "claude-4-sonnet"

# FastAPI Server Configuration
FASTAPI_BASE_URL = "http://localhost:8000"

# ============================================

print("🔧 Fixed LangGraph Agent Configuration:")
print(f"   Cortex API: {CORTEX_API_URL}")
print(f"   API Key Length: {len(CORTEX_API_KEY)} characters")
print(f"   Model: {CORTEX_MODEL}")
print(f"   FastAPI Server: {FASTAPI_BASE_URL}")

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
    # New fields for enhanced visualization support
    operation_type: str = ""  # create, read, update, delete, schema
    affected_nodes: int = 0
    affected_relationships: int = 0
    graph_changes: Dict[str, Any] = {}

# ============================================
# ENHANCED SYSTEM PROMPT WITH VISUALIZATION AWARENESS
# ============================================

SYSTEM_PROMPT = """
You are an expert AI assistant that helps users query and manage a Neo4j database with real-time visualization support. Your responses will be used to update live graph visualizations, so be precise and comprehensive.

AVAILABLE TOOLS:
- read_neo4j_cypher: Execute read-only queries (MATCH, RETURN, WHERE, OPTIONAL MATCH, etc.)
  * Use for data exploration, analysis, counting, reporting
  * Results will be displayed in both text and graph visualization
  * NEVER use for CREATE, UPDATE, DELETE operations

- write_neo4j_cypher: Execute write operations (CREATE, MERGE, SET, DELETE, REMOVE, etc.)
  * Use for data modification, node/relationship creation/deletion
  * Changes will be reflected immediately in the live visualization
  * Returns detailed operation counts for visualization updates

- get_neo4j_schema: Get database structure information
  * Use when users ask about schema, labels, relationship types, properties
  * Results help users understand the graph structure for visualization

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
6. Consider visualization impact - queries should return meaningful graph data

ENHANCED EXAMPLES FOR VISUALIZATION:

User: How many nodes are in the graph?
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN count(n) as total_nodes

User: Show me all Person nodes with their connections
Tool: read_neo4j_cypher
Query: MATCH (p:Person)-[r]-(connected) RETURN p, r, connected LIMIT 20

User: Create a Person named Alice who works at TechCorp
Tool: write_neo4j_cypher
Query: CREATE (alice:Person {name: 'Alice', created: datetime()}), (company:Company {name: 'TechCorp'}) CREATE (alice)-[:WORKS_FOR {since: date()}]->(company) RETURN alice, company

User: Delete all nodes with no relationships
Tool: write_neo4j_cypher
Query: MATCH (n) WHERE NOT (n)--() DETACH DELETE n

User: What's the database structure?
Tool: get_neo4j_schema

User: Find the most connected nodes for visualization
Tool: read_neo4j_cypher
Query: MATCH (n) WITH n, size((n)--()) as connections WHERE connections > 0 RETURN n, connections ORDER BY connections DESC LIMIT 10

User: Create a social network example
Tool: write_neo4j_cypher
Query: CREATE (alice:Person {name: 'Alice', age: 30}), (bob:Person {name: 'Bob', age: 25}), (charlie:Person {name: 'Charlie', age: 35}) CREATE (alice)-[:FRIENDS_WITH {since: '2020'}]->(bob), (bob)-[:FRIENDS_WITH {since: '2021'}]->(charlie), (charlie)-[:FRIENDS_WITH {since: '2019'}]->(alice) RETURN alice, bob, charlie

VISUALIZATION CONSIDERATIONS:
- For read queries, return nodes and relationships when possible for graph display
- For write operations, return created/modified elements to show in visualization
- Consider query performance for visualization (use LIMIT for large result sets)
- Structure queries to provide meaningful graph data for the NVL interface

ERROR HANDLING:
- If query syntax is unclear, ask for clarification
- For ambiguous requests, suggest the most visualization-friendly interpretation
- Always validate Cypher syntax before responding
"""

# ============================================
# ENHANCED LLM COMMUNICATION
# ============================================

def call_cortex_llm(prompt: str, session_id: str) -> str:
    """Call Cortex LLM with enhanced error handling and retry logic"""
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
                    "sys_msg": SYSTEM_PROMPT,
                    "limit_convs": "0",
                    "prompt": {
                        "messages": [{"role": "user", "content": prompt}]
                    },
                    "session_id": session_id
                }
            }
            
            logger.info(f"Calling Cortex LLM (attempt {attempt + 1}): {prompt[:100]}...")
            response = requests.post(CORTEX_API_URL, headers=headers, json=payload, verify=False, timeout=30)
            
            if response.status_code == 200:
                result = response.text.partition("end_of_stream")[0].strip()
                logger.info(f"LLM response received: {len(result)} characters")
                return result
            else:
                logger.error(f"Cortex API error: {response.status_code}")
                if attempt == max_retries - 1:
                    return f"Error: Cortex API returned {response.status_code}"
                    
        except Exception as e:
            logger.error(f"Cortex LLM call failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return f"Error: Failed to call LLM after {max_retries} attempts - {str(e)}"
            
        # Wait before retry
        time.sleep(retry_delay * (attempt + 1))
    
    return "Error: Maximum retry attempts exceeded"

# ============================================
# ENHANCED TOOL SELECTION AND PARSING
# ============================================

def determine_intent_and_operation_type(question: str) -> tuple[str, str]:
    """Determine user intent and operation type for visualization"""
    q_lower = question.lower()
    
    # Schema operations
    if any(word in q_lower for word in ['schema', 'structure', 'labels', 'relationships', 'types', 'properties']):
        return "schema", "schema"
    
    # Write operations
    elif any(word in q_lower for word in ['create', 'add', 'insert', 'new', 'make']):
        return "create", "create"
    elif any(word in q_lower for word in ['delete', 'remove', 'drop', 'clear']):
        return "delete", "delete"
    elif any(word in q_lower for word in ['update', 'set', 'change', 'modify', 'edit']):
        return "update", "update"
    elif any(word in q_lower for word in ['merge', 'upsert']):
        return "upsert", "create"
    
    # Read operations
    elif any(word in q_lower for word in ['how many', 'count', 'number of']):
        return "count", "read"
    elif any(word in q_lower for word in ['show', 'list', 'find', 'get', 'select', 'display']):
        return "retrieve", "read"
    elif any(word in q_lower for word in ['analyze', 'analysis', 'report']):
        return "analyze", "read"
    
    # Default to read
    else:
        return "explore", "read"

def parse_llm_response_enhanced(llm_output: str, question: str) -> tuple[str, str, str, Dict[str, Any]]:
    """Enhanced LLM response parsing with better error handling and visualization support"""
    valid_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    
    debug_info = {
        "raw_output": llm_output[:300],
        "question": question,
        "parsing_steps": []
    }
    
    tool = None
    query = None
    trace = llm_output.strip()
    
    # Strategy 1: Look for exact "Tool: xxx" pattern
    tool_pattern = r"Tool:\s*([a-zA-Z_]+)"
    tool_match = re.search(tool_pattern, llm_output, re.I)
    if tool_match:
        extracted_tool = tool_match.group(1).strip()
        if extracted_tool in valid_tools:
            tool = extracted_tool
            debug_info["parsing_steps"].append(f"Found tool via pattern: {tool}")
    
    # Strategy 2: Look for tool names in text
    if not tool:
        for valid_tool in valid_tools:
            if valid_tool.lower() in llm_output.lower():
                tool = valid_tool
                debug_info["parsing_steps"].append(f"Found tool in text: {tool}")
                break
    
    # Strategy 3: Infer from question content
    if not tool:
        intent, operation_type = determine_intent_and_operation_type(question)
        
        if operation_type == "schema":
            tool = "get_neo4j_schema"
        elif operation_type in ["create", "delete", "update"]:
            tool = "write_neo4j_cypher"
        else:
            tool = "read_neo4j_cypher"
            
        debug_info["parsing_steps"].append(f"Inferred tool from question: {tool} (intent: {intent})")
    
    # Strategy 4: Extract query
    query_patterns = [
        r"Query:\s*(.+?)(?=\n|$)",
        r"query:\s*(.+?)(?=\n|$)",
        r"Cypher:\s*(.+?)(?=\n|$)",
        r"`([^`]+)`",
        r"```\s*cypher\s*([^```]+)\s*```",
        r"```([^```]+)```"
    ]
    
    for pattern in query_patterns:
        match = re.search(pattern, llm_output, re.I | re.MULTILINE | re.DOTALL)
        if match:
            extracted_query = match.group(1).strip()
            # Clean the query
            extracted_query = re.sub(r'```[a-zA-Z]*', '', extracted_query)
            extracted_query = re.sub(r'```', '', extracted_query)
            extracted_query = re.sub(r'\s+', ' ', extracted_query).strip()
            
            if len(extracted_query) > 5 and any(keyword in extracted_query.upper() for keyword in ['MATCH', 'CREATE', 'DELETE', 'MERGE', 'SET', 'RETURN']):
                query = extracted_query
                debug_info["parsing_steps"].append(f"Extracted query via pattern: {pattern}")
                break
    
    # Strategy 5: Generate fallback query if needed
    if tool and not query and tool != "get_neo4j_schema":
        intent, operation_type = determine_intent_and_operation_type(question)
        
        if tool == "read_neo4j_cypher":
            if "count" in intent:
                query = "MATCH (n) RETURN count(n) as total_nodes"
            else:
                query = "MATCH (n) RETURN n LIMIT 10"
        elif tool == "write_neo4j_cypher":
            if operation_type == "create":
                query = "CREATE (n:TestNode {name: 'test', created: datetime()}) RETURN n"
            elif operation_type == "delete":
                query = "MATCH (n:TestNode) DETACH DELETE n"
            else:
                query = "MATCH (n) SET n.updated = datetime() RETURN count(n) as updated"
        
        debug_info["parsing_steps"].append(f"Generated fallback query: {query}")
    
    # Final validation
    if not tool:
        tool = "read_neo4j_cypher"
        query = "MATCH (n) RETURN count(n) as total"
        debug_info["parsing_steps"].append("Used final fallback: read operation")
    
    debug_info["final_result"] = {"tool": tool, "query": query}
    
    return tool, query, trace, debug_info

# ============================================
# ENHANCED FASTAPI SERVER COMMUNICATION
# ============================================

async def call_fastapi_server_enhanced(tool: str, query: str = None) -> Dict[str, Any]:
    """Enhanced FastAPI server communication with better error handling and response parsing"""
    try:
        headers = {"Content-Type": "application/json"}
        
        logger.info(f"Calling FastAPI server - Tool: {tool}, Query: {query}")
        
        if tool == "get_neo4j_schema":
            response = requests.post(f"{FASTAPI_BASE_URL}/get_neo4j_schema", headers=headers, timeout=30)
            
        elif tool == "read_neo4j_cypher":
            if not query:
                return {"success": False, "error": "No query provided for read operation"}
            
            data = {"query": query, "params": {}}
            response = requests.post(f"{FASTAPI_BASE_URL}/read_neo4j_cypher", json=data, headers=headers, timeout=30)
            
        elif tool == "write_neo4j_cypher":
            if not query:
                return {"success": False, "error": "No query provided for write operation"}
                
            data = {"query": query, "params": {}}
            response = requests.post(f"{FASTAPI_BASE_URL}/write_neo4j_cypher", json=data, headers=headers, timeout=30)
            
        else:
            return {"success": False, "error": f"Unknown tool: {tool}"}
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"FastAPI server response successful: {type(result)}")
            
            # Ensure the response has the expected structure
            if isinstance(result, list):
                # For read operations that return lists
                return {
                    "success": True,
                    "data": result,
                    "count": len(result),
                    "type": "list_result"
                }
            elif isinstance(result, dict):
                # For write operations and schema
                return {
                    "success": True,
                    "data": result,
                    "type": "dict_result",
                    **result  # Merge any additional fields
                }
            else:
                return {"success": True, "data": result, "type": "other"}
                
        else:
            logger.error(f"FastAPI server error: {response.status_code} - {response.text}")
            return {
                "success": False, 
                "error": f"FastAPI server error: {response.status_code}",
                "details": response.text
            }
            
    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to FastAPI server at {FASTAPI_BASE_URL}")
        return {
            "success": False, 
            "error": f"Cannot connect to FastAPI server. Is it running on port 8000?",
            "connection_failed": True
        }
    except Exception as e:
        logger.error(f"FastAPI server call failed: {e}")
        return {"success": False, "error": f"FastAPI server failed: {str(e)}"}

# ============================================
# ENHANCED RESPONSE FORMATTING WITH VISUALIZATION SUPPORT
# ============================================

def format_response_with_visualization(tool: str, query: str, response: Dict[str, Any], debug_info: Dict[str, Any] = None) -> tuple[str, Dict[str, Any]]:
    """Enhanced response formatting with visualization metadata"""
    
    visualization_metadata = {
        "tool_used": tool,
        "query_executed": query,
        "operation_type": "unknown",
        "nodes_affected": 0,
        "relationships_affected": 0,
        "visualization_update_needed": False
    }
    
    if not response.get("success", False):
        error = response.get("error", "Unknown error")
        formatted_response = f"❌ **Operation Failed**\n\n**Error:** {error}"
        
        if debug_info:
            formatted_response += f"\n\n**Debug Information:**\n"
            for step in debug_info.get("parsing_steps", []):
                formatted_response += f"• {step}\n"
        
        return formatted_response, visualization_metadata
    
    data = response.get("data", {})
    
    # Handle schema operations
    if tool == "get_neo4j_schema":
        visualization_metadata["operation_type"] = "schema"
        
        if isinstance(data, dict):
            labels = data.get("labels", [])
            rel_types = data.get("relationship_types", [])
            prop_keys = data.get("property_keys", [])
            
            formatted_response = f"""📊 **Database Schema Retrieved**

**Node Labels ({len(labels)}):**
{', '.join(labels[:20])}

**Relationship Types ({len(rel_types)}):**
{', '.join(rel_types[:20])}

**Property Keys ({len(prop_keys)}):**
{', '.join(prop_keys[:20])}

*Schema information updated for visualization interface*"""

        else:
            formatted_response = f"📊 **Database Schema:**\n```json\n{json.dumps(data, indent=2)[:800]}\n```"
        
        return formatted_response, visualization_metadata
    
    # Handle read operations
    elif tool == "read_neo4j_cypher":
        visualization_metadata["operation_type"] = "read"
        visualization_metadata["visualization_update_needed"] = True
        
        if isinstance(data, list):
            count = len(data)
            visualization_metadata["nodes_affected"] = count
            
            if count == 0:
                formatted_response = "📊 **Query Result:** No data found"
            elif count == 1 and isinstance(data[0], dict) and len(data[0]) == 1:
                # Single value result (like count)
                key, value = list(data[0].items())[0]
                formatted_response = f"📊 **Query Result:**\n\n**{key.replace('_', ' ').title()}:** {value:,}"
            else:
                formatted_response = f"📊 **Query Results:** Found **{count:,}** records\n\n"
                
                # Show sample records
                sample_size = min(3, count)
                for i, record in enumerate(data[:sample_size]):
                    formatted_response += f"**Record {i+1}:**\n"
                    if isinstance(record, dict):
                        for k, v in record.items():
                            formatted_response += f"  • {k}: {v}\n"
                    else:
                        formatted_response += f"  {record}\n"
                    formatted_response += "\n"
                
                if count > sample_size:
                    formatted_response += f"... and **{count - sample_size:,}** more records\n\n"
                
                formatted_response += "*Full results available in graph visualization*"
        else:
            formatted_response = f"📊 **Query Result:**\n```json\n{json.dumps(data, indent=2)[:600]}\n```"
    
    # Handle write operations
    elif tool == "write_neo4j_cypher":
        visualization_metadata["operation_type"] = "write"
        visualization_metadata["visualization_update_needed"] = True
        
        if isinstance(data, dict):
            # Extract operation counts
            nodes_created = data.get("nodes_created", 0)
            nodes_deleted = data.get("nodes_deleted", 0)
            rels_created = data.get("relationships_created", 0)
            rels_deleted = data.get("relationships_deleted", 0)
            props_set = data.get("properties_set", 0)
            labels_added = data.get("labels_added", 0)
            labels_removed = data.get("labels_removed", 0)
            
            visualization_metadata["nodes_affected"] = nodes_created + nodes_deleted
            visualization_metadata["relationships_affected"] = rels_created + rels_deleted
            
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
                formatted_response = "✅ **Database Update Completed Successfully!**\n\n"
                formatted_response += "\n".join(operations)
                formatted_response += "\n\n*Changes immediately reflected in live graph visualization*"
                
                # Add performance info if available
                total_changes = sum([nodes_created, nodes_deleted, rels_created, rels_deleted, props_set])
                if total_changes > 0:
                    formatted_response += f"\n\n**Total Changes:** {total_changes:,}"
            else:
                formatted_response = "✅ **Query executed successfully** (no structural changes made)"
        else:
            formatted_response = f"✅ **Write Operation Result:**\n{json.dumps(data, indent=2)[:400]}"
    
    else:
        formatted_response = f"📊 **Operation Result:**\n{json.dumps(data, indent=2)[:500]}"
    
    return formatted_response, visualization_metadata

# ============================================
# ENHANCED LANGGRAPH NODES
# ============================================

def analyze_and_select_tool_node_enhanced(state: AgentState) -> Dict[str, Any]:
    """Enhanced Node 1: Analyze question and select tool with visualization awareness"""
    logger.info(f"Processing question with visualization support: {state.question}")
    
    # Determine intent and operation type
    intent, operation_type = determine_intent_and_operation_type(state.question)
    
    # Call LLM
    llm_output = call_cortex_llm(state.question, state.session_id)
    
    # Parse with enhanced parsing
    tool, query, trace, debug_info = parse_llm_response_enhanced(llm_output, state.question)
    
    logger.info(f"Selected tool: {tool}, query: {query}, intent: {intent}")
    
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
        "graph_changes": {}
    }

async def execute_tool_node_enhanced(state: AgentState) -> Dict[str, Any]:
    """Enhanced Node 2: Execute tool with visualization metadata"""
    logger.info(f"Executing tool with visualization support: {state.tool}")
    
    if not state.tool:
        formatted_answer = f"⚠️ **No valid tool selected.**\n\n**Debug Info:**\n{state.debug_info}"
        return {**state.dict(), "formatted_answer": formatted_answer}
    
    # Call FastAPI server
    raw_response = await call_fastapi_server_enhanced(state.tool, state.query)
    
    # Parse debug info
    try:
        debug_info = json.loads(state.debug_info) if state.debug_info else {}
    except:
        debug_info = {}
    
    # Format response with visualization metadata
    formatted_answer, viz_metadata = format_response_with_visualization(
        state.tool, state.query, raw_response, debug_info
    )
    
    # Update state with visualization metadata
    return {
        **state.dict(),
        "raw_response": raw_response,
        "formatted_answer": formatted_answer,
        "affected_nodes": viz_metadata.get("nodes_affected", 0),
        "affected_relationships": viz_metadata.get("relationships_affected", 0),
        "graph_changes": viz_metadata
    }

# ============================================
# BUILD ENHANCED AGENT
# ============================================

def build_agent():
    """Build the enhanced LangGraph agent with visualization support"""
    logger.info("Building enhanced LangGraph agent with visualization support...")
    
    workflow = StateGraph(state_schema=AgentState)
    
    # Add nodes
    workflow.add_node("analyze_and_select", RunnableLambda(analyze_and_select_tool_node_enhanced))
    workflow.add_node("execute_tool", RunnableLambda(execute_tool_node_enhanced))
    
    # Set entry point and edges
    workflow.set_entry_point("analyze_and_select")
    workflow.add_edge("analyze_and_select", "execute_tool")
    workflow.add_edge("execute_tool", END)
    
    agent = workflow.compile()
    logger.info("✅ Enhanced LangGraph agent with visualization support built successfully")
    
    return agent

# ============================================
# TESTING AND VALIDATION
# ============================================

def test_enhanced_agent():
    """Test the enhanced agent with various scenarios"""
    test_cases = [
        {
            "question": "How many nodes are in the graph?",
            "expected_tool": "read_neo4j_cypher",
            "expected_operation": "read"
        },
        {
            "question": "Create a Person named Alice with age 30",
            "expected_tool": "write_neo4j_cypher",
            "expected_operation": "create"
        },
        {
            "question": "Show me the database schema",
            "expected_tool": "get_neo4j_schema",
            "expected_operation": "schema"
        },
        {
            "question": "Delete all TestNode nodes",
            "expected_tool": "write_neo4j_cypher",
            "expected_operation": "delete"
        },
        {
            "question": "Find all Person nodes with their relationships",
            "expected_tool": "read_neo4j_cypher",
            "expected_operation": "read"
        }
    ]
    
    print("🧪 Testing Enhanced LangGraph Agent with Visualization Support")
    print("=" * 70)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['question']}")
        
        # Test parsing without LLM call
        tool, query, trace, debug_info = parse_llm_response_enhanced("", test_case['question'])
        
        print(f"   Expected Tool: {test_case['expected_tool']}")
        print(f"   Actual Tool: {tool}")
        print(f"   Expected Operation: {test_case['expected_operation']}")
        print(f"   Generated Query: {query}")
        print(f"   Match: {'✅' if tool == test_case['expected_tool'] else '❌'}")

async def test_complete_agent():
    """Test the complete agent with FastAPI integration"""
    agent = build_agent()
    
    test_question = "How many nodes are in the graph?"
    print(f"\n🧪 Testing complete enhanced agent with: {test_question}")
    
    try:
        state = AgentState(
            question=test_question,
            session_id="test_session_enhanced"
        )
        
        result = await agent.ainvoke(state)
        
        print(f"Tool: {result.get('tool', 'N/A')}")
        print(f"Query: {result.get('query', 'N/A')}")
        print(f"Operation Type: {result.get('operation_type', 'N/A')}")
        print(f"Affected Nodes: {result.get('affected_nodes', 0)}")
        print(f"Answer: {result.get('formatted_answer', 'N/A')[:200]}...")
        print(f"Visualization Metadata: {result.get('graph_changes', {})}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# ============================================
# UTILITY FUNCTIONS
# ============================================

def validate_cypher_query(query: str) -> tuple[bool, str]:
    """Basic Cypher query validation"""
    if not query or not query.strip():
        return False, "Empty query"
    
    query_upper = query.upper().strip()
    
    # Check for basic Cypher keywords
    valid_starters = ['MATCH', 'CREATE', 'MERGE', 'DELETE', 'SET', 'REMOVE', 'RETURN', 'CALL', 'WITH']
    if not any(query_upper.startswith(starter) for starter in valid_starters):
        return False, "Query doesn't start with valid Cypher keyword"
    
    # Check for potential injection patterns
    dangerous_patterns = ['DROP', 'ALTER', 'TRUNCATE']
    if any(pattern in query_upper for pattern in dangerous_patterns):
        return False, f"Query contains potentially dangerous operations"
    
    return True, "Valid"

def get_query_complexity_estimate(query: str) -> str:
    """Estimate query complexity for performance considerations"""
    if not query:
        return "unknown"
    
    query_upper = query.upper()
    
    # Simple heuristics
    if 'LIMIT' in query_upper:
        return "low"
    elif any(pattern in query_upper for pattern in ['JOIN', 'COLLECT', 'UNWIND']):
        return "medium"
    elif any(pattern in query_upper for pattern in ['ALL', 'EXISTS', 'SHORTEST']):
        return "high"
    else:
        return "medium"

if __name__ == "__main__":
    print("🚀 Enhanced LangGraph Agent with Visualization Support")
    print("=" * 60)
    
    # Run tests
    test_enhanced_agent()
    
    # Test complete agent (requires FastAPI server)
    print("\n" + "=" * 60)
    asyncio.run(test_complete_agent())
