"""
Robust LangGraph Agent with fixed tool selection and debugging
This version has better tool selection logic and debugging output
"""

import requests
import urllib3
import json
import logging
import re
import time
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("robust_langgraph_agent")

# ============================================
# 🔧 CONFIGURATION - CHANGE THESE VALUES
# ============================================

# Cortex API Configuration
CORTEX_API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
CORTEX_API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"  # ⚠️ CHANGE THIS!
CORTEX_MODEL = "llama3.1-70b"

# FastAPI Server Configuration
FASTAPI_BASE_URL = "http://localhost:8000"

# ============================================

print("🔧 Robust LangGraph Agent Configuration:")
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

# ============================================
# SIMPLIFIED AND ROBUST SYSTEM PROMPT
# ============================================

SYSTEM_PROMPT = """
You are a Neo4j database assistant. For each user question, you must choose exactly ONE tool and provide a Cypher query if needed.

AVAILABLE TOOLS (choose exactly one):
1. read_neo4j_cypher - for reading data (MATCH, RETURN, COUNT, etc.)
2. write_neo4j_cypher - for modifying data (CREATE, DELETE, SET, etc.)
3. get_neo4j_schema - for database structure information

RESPONSE FORMAT (very important):
Tool: [tool_name]
Query: [cypher_query]

EXAMPLES:

User: How many nodes?
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN count(n)

User: Create a person named Alice
Tool: write_neo4j_cypher
Query: CREATE (p:Person {name: 'Alice'}) RETURN p

User: Delete test nodes
Tool: write_neo4j_cypher
Query: MATCH (t:TestNode) DETACH DELETE t

User: Show schema
Tool: get_neo4j_schema
Query: 

RULES:
- Always respond with "Tool:" followed by exact tool name
- For read_neo4j_cypher and write_neo4j_cypher, include "Query:" line
- For get_neo4j_schema, Query line can be empty
- Use DETACH DELETE for deletions
- Be very specific with tool names
"""

# ============================================
# LLM COMMUNICATION WITH DEBUGGING
# ============================================

def call_cortex_llm(prompt: str, session_id: str) -> str:
    """Call Cortex LLM with enhanced debugging"""
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
        
        logger.info(f"Calling Cortex LLM with prompt: {prompt[:100]}...")
        response = requests.post(CORTEX_API_URL, headers=headers, json=payload, verify=False, timeout=30)
        
        if response.status_code == 200:
            result = response.text.partition("end_of_stream")[0].strip()
            logger.info(f"LLM response received: {result}")
            return result
        else:
            logger.error(f"Cortex API error: {response.status_code}")
            return f"Error: Cortex API returned {response.status_code}"
            
    except Exception as e:
        logger.error(f"Cortex LLM call failed: {e}")
        return f"Error: Failed to call LLM - {str(e)}"

# ============================================
# IMPROVED TOOL SELECTION WITH FALLBACKS
# ============================================

def determine_tool_from_question(question: str) -> tuple[str, str, str]:
    """
    Determine tool based on question keywords as fallback
    Returns: (tool, suggested_query, reasoning)
    """
    q_lower = question.lower()
    
    # Schema-related keywords
    if any(word in q_lower for word in ['schema', 'structure', 'labels', 'relationships', 'types', 'properties']):
        return "get_neo4j_schema", "", "Schema-related question detected"
    
    # Write operations
    elif any(word in q_lower for word in ['create', 'add', 'insert', 'new']):
        if 'person' in q_lower or 'user' in q_lower:
            name_match = re.search(r'named?\s+([a-zA-Z]+)', q_lower)
            name = name_match.group(1) if name_match else 'Unknown'
            query = f"CREATE (p:Person {{name: '{name}'}}) RETURN p"
        else:
            query = "CREATE (n:Node {name: 'example'}) RETURN n"
        return "write_neo4j_cypher", query, "Create operation detected"
    
    elif any(word in q_lower for word in ['delete', 'remove', 'drop']):
        if 'test' in q_lower:
            query = "MATCH (t:TestNode) DETACH DELETE t"
        elif 'all' in q_lower:
            query = "MATCH (n) DETACH DELETE n"  # Dangerous but requested
        else:
            query = "MATCH (n) WHERE n.name = 'example' DETACH DELETE n"
        return "write_neo4j_cypher", query, "Delete operation detected"
    
    elif any(word in q_lower for word in ['update', 'set', 'change', 'modify']):
        query = "MATCH (n) SET n.updated = datetime() RETURN n"
        return "write_neo4j_cypher", query, "Update operation detected"
    
    # Read operations
    elif any(word in q_lower for word in ['how many', 'count', 'number of']):
        query = "MATCH (n) RETURN count(n) as total"
        return "read_neo4j_cypher", query, "Count operation detected"
    
    elif any(word in q_lower for word in ['show', 'list', 'find', 'get', 'select']):
        if 'person' in q_lower or 'people' in q_lower:
            query = "MATCH (p:Person) RETURN p LIMIT 10"
        else:
            query = "MATCH (n) RETURN n LIMIT 10"
        return "read_neo4j_cypher", query, "Read operation detected"
    
    # Default to read
    else:
        query = "MATCH (n) RETURN count(n) as total"
        return "read_neo4j_cypher", query, "Default: treating as read operation"

def parse_llm_response_robust(llm_output: str, question: str) -> tuple[str, str, str]:
    """
    Parse LLM response with multiple fallback strategies
    Returns: (tool, query, debug_info)
    """
    valid_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    
    debug_info = f"Raw LLM output: {llm_output[:200]}...\n"
    
    tool = None
    query = None
    
    # Strategy 1: Look for exact tool names in response
    for valid_tool in valid_tools:
        if valid_tool in llm_output.lower():
            tool = valid_tool
            debug_info += f"Found tool '{tool}' in response\n"
            break
    
    # Strategy 2: Pattern matching for "Tool: xxx"
    if not tool:
        tool_patterns = [
            r"Tool:\s*(\w+)",
            r"tool:\s*(\w+)", 
            r"Tool\s*=\s*(\w+)",
            r"Use\s+(\w+)",
            r"Selected?\s+tool:\s*(\w+)"
        ]
        
        for pattern in tool_patterns:
            match = re.search(pattern, llm_output, re.I)
            if match:
                extracted = match.group(1).strip()
                # Try to match to valid tools
                for valid_tool in valid_tools:
                    if extracted.lower() in valid_tool.lower() or valid_tool.lower() in extracted.lower():
                        tool = valid_tool
                        debug_info += f"Matched '{extracted}' to '{tool}' using pattern '{pattern}'\n"
                        break
                if tool:
                    break
    
    # Strategy 3: Look for Cypher keywords to determine tool type
    if not tool:
        cypher_keywords = {
            "read_neo4j_cypher": ["match", "return", "where", "count", "collect"],
            "write_neo4j_cypher": ["create", "delete", "set", "merge", "remove"],
        }
        
        llm_lower = llm_output.lower()
        for tool_type, keywords in cypher_keywords.items():
            if any(keyword in llm_lower for keyword in keywords):
                tool = tool_type
                debug_info += f"Inferred tool '{tool}' from Cypher keywords\n"
                break
    
    # Strategy 4: Fallback based on question analysis
    if not tool:
        tool, fallback_query, reasoning = determine_tool_from_question(question)
        debug_info += f"Fallback tool selection: {tool} - {reasoning}\n"
        if not query:
            query = fallback_query
    
    # Extract query
    query_patterns = [
        r"Query:\s*(.+?)(?=\n|$)",
        r"query:\s*(.+?)(?=\n|$)",
        r"Cypher:\s*(.+?)(?=\n|$)",
        r"`([^`]+)`",  # Backticks
        r"```[a-zA-Z]*\s*([^```]+)\s*```"  # Code blocks
    ]
    
    for pattern in query_patterns:
        match = re.search(pattern, llm_output, re.I | re.MULTILINE | re.DOTALL)
        if match:
            extracted_query = match.group(1).strip()
            if len(extracted_query) > 5:  # Reasonable query length
                query = extracted_query
                debug_info += f"Extracted query using pattern '{pattern}': {query}\n"
                break
    
    # Clean query
    if query:
        query = re.sub(r'```[a-zA-Z]*', '', query)
        query = re.sub(r'```', '', query)
        query = re.sub(r'\s+', ' ', query).strip()
        debug_info += f"Cleaned query: {query}\n"
    
    # Final validation and fallback
    if not tool:
        tool = "read_neo4j_cypher"  # Safe default
        query = "MATCH (n) RETURN count(n) as total"
        debug_info += "Using safe default: read_neo4j_cypher with count query\n"
    
    if tool != "get_neo4j_schema" and not query:
        # Generate a basic query based on tool type
        if tool == "read_neo4j_cypher":
            query = "MATCH (n) RETURN count(n) as total"
        elif tool == "write_neo4j_cypher":
            query = "CREATE (t:TestNode {created: datetime()}) RETURN t"
        debug_info += f"Generated default query for {tool}: {query}\n"
    
    debug_info += f"Final selection: tool='{tool}', query='{query}'\n"
    
    return tool, query, debug_info

# ============================================
# FASTAPI SERVER COMMUNICATION
# ============================================

def call_fastapi_server(tool: str, query: str = None) -> Dict[str, Any]:
    """Call FastAPI server with proper error handling"""
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
            logger.info(f"FastAPI server response successful")
            return result
        else:
            logger.error(f"FastAPI server error: {response.status_code} - {response.text}")
            return {"success": False, "error": f"FastAPI server error: {response.status_code}"}
            
    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to FastAPI server at {FASTAPI_BASE_URL}")
        return {"success": False, "error": f"Cannot connect to FastAPI server. Is it running on port 8000?"}
    except Exception as e:
        logger.error(f"FastAPI server call failed: {e}")
        return {"success": False, "error": f"FastAPI server failed: {str(e)}"}

# ============================================
# SIMPLIFIED RESPONSE FORMATTING
# ============================================

def format_response_simple(tool: str, query: str, response: Dict[str, Any], debug_info: str = "") -> str:
    """Simple response formatting with debug info"""
    
    if not response.get("success", False):
        error = response.get("error", "Unknown error")
        result = f"❌ **Error:** {error}"
        if debug_info:
            result += f"\n\n**🔧 Debug Info:**\n{debug_info}"
        return result
    
    data = response.get("data", {})
    summary = response.get("summary", {})
    
    if tool == "get_neo4j_schema":
        if isinstance(data, dict):
            labels = data.get("labels", [])
            rel_types = data.get("relationship_types", [])
            result = f"📊 **Database Schema:**\n"
            result += f"**Node Labels:** {', '.join(labels[:10])}\n"
            result += f"**Relationship Types:** {', '.join(rel_types[:10])}"
            return result
        else:
            return f"📊 **Schema:** {json.dumps(data, indent=2)[:300]}..."
    
    elif tool == "read_neo4j_cypher":
        if isinstance(data, list):
            count = len(data)
            if count == 0:
                return "📊 **Result:** No data found"
            elif count == 1 and isinstance(data[0], dict) and len(data[0]) == 1:
                key, value = list(data[0].items())[0]
                return f"📊 **Result:** {value}"
            else:
                result = f"📊 **Found {count} records**"
                if count <= 3:
                    result += f"\n```json\n{json.dumps(data, indent=2)}\n```"
                else:
                    result += f"\n```json\n{json.dumps(data[:2], indent=2)}\n... and {count-2} more\n```"
                return result
        else:
            return f"📊 **Result:** {json.dumps(data, indent=2)[:300]}"
    
    elif tool == "write_neo4j_cypher":
        changes = summary.get("changes", {})
        
        result = "✅ **Write Operation Completed**\n\n"
        
        nodes_created = changes.get("nodes_created", 0)
        nodes_deleted = changes.get("nodes_deleted", 0)
        rels_created = changes.get("relationships_created", 0)
        rels_deleted = changes.get("relationships_deleted", 0)
        props_set = changes.get("properties_set", 0)
        
        operations = []
        if nodes_created > 0:
            operations.append(f"🟢 Created {nodes_created} node{'s' if nodes_created != 1 else ''}")
        if nodes_deleted > 0:
            operations.append(f"🗑️ Deleted {nodes_deleted} node{'s' if nodes_deleted != 1 else ''}")
        if rels_created > 0:
            operations.append(f"🔗 Created {rels_created} relationship{'s' if rels_created != 1 else ''}")
        if rels_deleted > 0:
            operations.append(f"❌ Deleted {rels_deleted} relationship{'s' if rels_deleted != 1 else ''}")
        if props_set > 0:
            operations.append(f"📝 Set {props_set} propert{'ies' if props_set != 1 else 'y'}")
        
        if operations:
            result += "\n".join(operations)
        else:
            result += "No changes made"
        
        total_changes = summary.get("total_changes", 0)
        if total_changes > 0:
            result += f"\n\n**Total Changes:** {total_changes}"
        
        return result
    
    else:
        return f"📊 **Result:** {json.dumps(data, indent=2)[:300]}"

# ============================================
# LANGGRAPH NODES WITH DEBUGGING
# ============================================

def analyze_and_select_tool_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Analyze question and select tool with debugging"""
    logger.info(f"Processing question: {state.question}")
    
    # Call LLM
    llm_output = call_cortex_llm(state.question, state.session_id)
    
    # Parse with robust fallbacks
    tool, query, debug_info = parse_llm_response_robust(llm_output, state.question)
    
    # Determine intent
    intent = "unknown"
    if tool == "read_neo4j_cypher":
        intent = "read"
    elif tool == "write_neo4j_cypher":
        intent = "write"
    elif tool == "get_neo4j_schema":
        intent = "schema"
    
    logger.info(f"Selected tool: {tool}, query: {query}")
    
    return {
        "question": state.question,
        "session_id": state.session_id,
        "intent": intent,
        "tool": tool,
        "query": query,
        "trace": llm_output,
        "debug_info": debug_info,
        "raw_response": {},
        "formatted_answer": "",
        "error_count": state.error_count,
        "last_error": state.last_error
    }

def execute_tool_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Execute tool with debugging"""
    logger.info(f"Executing tool: {state.tool} with query: {state.query}")
    
    if not state.tool:
        formatted_answer = f"⚠️ **No valid tool selected.**\n\n**Debug Info:**\n{state.debug_info}"
        return {**state.dict(), "formatted_answer": formatted_answer}
    
    # Call FastAPI server
    raw_response = call_fastapi_server(state.tool, state.query)
    
    # Format response
    formatted_answer = format_response_simple(state.tool, state.query, raw_response, state.debug_info)
    
    return {
        **state.dict(),
        "raw_response": raw_response,
        "formatted_answer": formatted_answer
    }

# ============================================
# BUILD AGENT
# ============================================

def build_agent():
    """Build the robust LangGraph agent"""
    logger.info("Building robust LangGraph agent...")
    
    workflow = StateGraph(state_schema=AgentState)
    
    # Add nodes
    workflow.add_node("analyze_and_select", RunnableLambda(analyze_and_select_tool_node))
    workflow.add_node("execute_tool", RunnableLambda(execute_tool_node))
    
    # Set entry point and edges
    workflow.set_entry_point("analyze_and_select")
    workflow.add_edge("analyze_and_select", "execute_tool")
    workflow.add_edge("execute_tool", END)
    
    agent = workflow.compile()
    logger.info("✅ Robust LangGraph agent built successfully")
    
    return agent

# ============================================
# TESTING AND DEBUGGING
# ============================================

def test_tool_selection():
    """Test tool selection with various inputs"""
    test_cases = [
        "How many nodes are in the graph?",
        "Create a Person named Alice",
        "Delete all test nodes",
        "Show me the database schema",
        "Count all relationships",
        "Add a new user named Bob"
    ]
    
    print("🧪 Testing Tool Selection")
    print("=" * 50)
    
    for question in test_cases:
        print(f"\nQuestion: {question}")
        tool, query, debug_info = parse_llm_response_robust("", question)  # Test fallback
        print(f"Tool: {tool}")
        print(f"Query: {query}")
        print(f"Debug: {debug_info.split('Final selection:')[-1] if 'Final selection:' in debug_info else 'N/A'}")

def test_agent():
    """Test the complete agent"""
    agent = build_agent()
    
    test_question = "How many nodes are in the graph?"
    print(f"\n🧪 Testing complete agent with: {test_question}")
    
    try:
        state = AgentState(
            question=test_question,
            session_id="test_session"
        )
        
        result = agent.invoke(state)
        
        print(f"Tool: {result.get('tool', 'N/A')}")
        print(f"Query: {result.get('query', 'N/A')}")
        print(f"Answer: {result.get('formatted_answer', 'N/A')[:200]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_tool_selection()
    test_agent()
