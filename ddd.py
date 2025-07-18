"""
Fixed LangGraph Agent that properly shows CREATE/DELETE counts in UI
This agent correctly parses FastAPI responses and formats them for display
"""

import requests
import urllib3
import json
import logging
import re
import time
from typing import Dict, Any, Optional
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
CORTEX_MODEL = "llama3.1-70b"

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
    error_count: int = 0
    last_error: str = ""

# ============================================
# IMPROVED SYSTEM PROMPT
# ============================================

SYSTEM_PROMPT = """
You are an expert AI assistant that helps users query and manage a Neo4j database using FastAPI tools.

AVAILABLE TOOLS:
- read_neo4j_cypher: Execute read-only Cypher queries (MATCH, RETURN, WHERE, etc.)
- write_neo4j_cypher: Execute write Cypher queries (CREATE, MERGE, SET, DELETE, etc.)
- get_neo4j_schema: Get database schema information

GUIDELINES:
- Always explain your reasoning before selecting a tool
- Choose the appropriate tool based on the user's intent
- For schema questions, use get_neo4j_schema
- For data queries, use read_neo4j_cypher
- For data modifications, use write_neo4j_cypher
- ALWAYS use DETACH DELETE when deleting nodes (never plain DELETE)
- For creating multiple items, use UNWIND or multiple CREATE statements

RESPONSE FORMAT:
Always respond with:
Tool: [exact_tool_name]
Query: [cypher_query_if_needed]

EXAMPLES:

User: How many nodes are in the graph?
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN count(n) as total_nodes

User: Create a Person named Alice with age 30
Tool: write_neo4j_cypher
Query: CREATE (p:Person {name: 'Alice', age: 30, created: datetime()}) RETURN p

User: Delete all test nodes
Tool: write_neo4j_cypher
Query: MATCH (t:TestNode) DETACH DELETE t

User: Show the database schema
Tool: get_neo4j_schema

User: Create 3 people: Alice, Bob, Carol
Tool: write_neo4j_cypher
Query: UNWIND [{name: 'Alice'}, {name: 'Bob'}, {name: 'Carol'}] as person CREATE (p:Person {name: person.name, created: datetime()}) RETURN p

IMPORTANT:
- Use DETACH DELETE for all deletions
- Always include RETURN statements when possible to show what was affected
- Be specific about what you're creating or deleting
"""

# ============================================
# LLM COMMUNICATION
# ============================================

def call_cortex_llm(prompt: str, session_id: str) -> str:
    """Call Cortex LLM with error handling"""
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
        
        logger.info("Calling Cortex LLM...")
        response = requests.post(CORTEX_API_URL, headers=headers, json=payload, verify=False, timeout=30)
        
        if response.status_code == 200:
            result = response.text.partition("end_of_stream")[0].strip()
            logger.info(f"LLM response received: {len(result)} characters")
            return result
        else:
            logger.error(f"Cortex API error: {response.status_code}")
            return f"Error: Cortex API returned {response.status_code}"
            
    except Exception as e:
        logger.error(f"Cortex LLM call failed: {e}")
        return f"Error: Failed to call LLM - {str(e)}"

# ============================================
# FASTAPI SERVER COMMUNICATION
# ============================================

def call_fastapi_server(tool: str, query: str = None) -> Dict[str, Any]:
    """Call FastAPI server with proper error handling and response parsing"""
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
# RESPONSE PARSING AND FORMATTING
# ============================================

def parse_llm_response(llm_output: str) -> tuple[str, str, str]:
    """Parse LLM response to extract tool and query"""
    valid_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    
    tool = None
    query = None
    trace = llm_output.strip()
    
    # Extract tool
    tool_match = re.search(r"Tool:\s*(\w+)", llm_output, re.I)
    if tool_match:
        extracted_tool = tool_match.group(1).strip()
        if extracted_tool in valid_tools:
            tool = extracted_tool
    
    # Extract query
    query_match = re.search(r"Query:\s*(.+?)(?=\n|$)", llm_output, re.I | re.MULTILINE)
    if query_match:
        query = query_match.group(1).strip()
        # Clean query
        query = re.sub(r'```[a-zA-Z]*', '', query)
        query = re.sub(r'```', '', query)
        query = re.sub(r'\s+', ' ', query).strip()
    
    return tool, query, trace

def format_fastapi_response(tool: str, query: str, response: Dict[str, Any]) -> str:
    """Format FastAPI response into user-friendly message with proper CREATE/DELETE counts"""
    
    if not response.get("success", False):
        error = response.get("error", "Unknown error")
        return f"❌ **Error:** {error}"
    
    data = response.get("data", {})
    summary = response.get("summary", {})
    
    # Handle different tool types
    if tool == "get_neo4j_schema":
        if isinstance(data, dict):
            labels = data.get("labels", [])
            rel_types = data.get("relationship_types", [])
            prop_keys = data.get("property_keys", [])
            
            answer = f"📊 **Database Schema:**\n\n"
            answer += f"**Node Labels ({len(labels)}):** {', '.join(labels[:10])}"
            if len(labels) > 10:
                answer += f" ... and {len(labels) - 10} more"
            answer += f"\n\n**Relationship Types ({len(rel_types)}):** {', '.join(rel_types[:10])}"
            if len(rel_types) > 10:
                answer += f" ... and {len(rel_types) - 10} more"
            answer += f"\n\n**Property Keys ({len(prop_keys)}):** {', '.join(prop_keys[:10])}"
            if len(prop_keys) > 10:
                answer += f" ... and {len(prop_keys) - 10} more"
            
            return answer
        else:
            return f"📊 **Schema:** {json.dumps(data, indent=2)[:500]}..."
    
    elif tool == "read_neo4j_cypher":
        if isinstance(data, list):
            count = len(data)
            if count == 0:
                return "📊 **Result:** No data found"
            elif count == 1 and isinstance(data[0], dict) and len(data[0]) == 1:
                # Single value result (like count queries)
                key, value = list(data[0].items())[0]
                return f"📊 **Result:** {value}"
            else:
                answer = f"📊 **Result:** Found {count} records"
                
                # Show first few records
                if count <= 3:
                    answer += f"\n\n```json\n{json.dumps(data, indent=2)}\n```"
                else:
                    answer += f"\n\n```json\n{json.dumps(data[:3], indent=2)}\n... and {count - 3} more records\n```"
                
                return answer
        else:
            return f"📊 **Result:** {json.dumps(data, indent=2)[:500]}"
    
    elif tool == "write_neo4j_cypher":
        # This is the key fix - properly format write operation results
        changes = summary.get("changes", {})
        
        # Build the response with clear CREATE/DELETE indicators
        answer = "✅ **Write Operation Completed:**\n\n"
        
        # Show what was created
        nodes_created = changes.get("nodes_created", 0)
        rels_created = changes.get("relationships_created", 0)
        props_set = changes.get("properties_set", 0)
        labels_added = changes.get("labels_added", 0)
        
        # Show what was deleted
        nodes_deleted = changes.get("nodes_deleted", 0)
        rels_deleted = changes.get("relationships_deleted", 0)
        labels_removed = changes.get("labels_removed", 0)
        
        # Create summary
        operations = []
        
        if nodes_created > 0:
            operations.append(f"🟢 **Created {nodes_created} node{'s' if nodes_created != 1 else ''}**")
        
        if rels_created > 0:
            operations.append(f"🔗 **Created {rels_created} relationship{'s' if rels_created != 1 else ''}**")
        
        if props_set > 0:
            operations.append(f"📝 **Set {props_set} propert{'ies' if props_set != 1 else 'y'}**")
        
        if labels_added > 0:
            operations.append(f"🏷️ **Added {labels_added} label{'s' if labels_added != 1 else ''}**")
        
        if nodes_deleted > 0:
            operations.append(f"🗑️ **Deleted {nodes_deleted} node{'s' if nodes_deleted != 1 else ''}**")
        
        if rels_deleted > 0:
            operations.append(f"❌ **Deleted {rels_deleted} relationship{'s' if rels_deleted != 1 else ''}**")
        
        if labels_removed > 0:
            operations.append(f"🏷️ **Removed {labels_removed} label{'s' if labels_removed != 1 else ''}**")
        
        if operations:
            answer += "\n".join(operations)
        else:
            answer += "No changes made to the database"
        
        # Show total changes
        total_changes = summary.get("total_changes", 0)
        if total_changes > 0:
            answer += f"\n\n**Total Changes:** {total_changes}"
        
        # Show returned data if any
        if data and len(data) > 0:
            answer += f"\n\n**Returned Data:** {len(data)} record{'s' if len(data) != 1 else ''}"
            if len(data) <= 3:
                answer += f"\n```json\n{json.dumps(data, indent=2)}\n```"
        
        # Show query execution time if available
        query_time = summary.get("query_time_ms", 0)
        if query_time > 0:
            answer += f"\n\n⏱️ **Execution Time:** {query_time}ms"
        
        return answer
    
    else:
        return f"📊 **Result:** {json.dumps(data, indent=2)[:500]}"

# ============================================
# LANGGRAPH NODES
# ============================================

def analyze_and_select_tool_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Analyze question and select appropriate tool"""
    logger.info(f"Processing question: {state.question}")
    
    # Call LLM to analyze question and select tool
    llm_output = call_cortex_llm(state.question, state.session_id)
    tool, query, trace = parse_llm_response(llm_output)
    
    # Determine intent based on tool and query
    intent = "unknown"
    if tool == "read_neo4j_cypher":
        if query and "count" in query.lower():
            intent = "count"
        elif query and any(word in query.lower() for word in ["match", "return", "where"]):
            intent = "query"
        else:
            intent = "read"
    elif tool == "write_neo4j_cypher":
        if query and "create" in query.lower():
            intent = "create"
        elif query and "delete" in query.lower():
            intent = "delete"
        elif query and any(word in query.lower() for word in ["set", "update", "merge"]):
            intent = "update"
        else:
            intent = "write"
    elif tool == "get_neo4j_schema":
        intent = "schema"
    
    return {
        "question": state.question,
        "session_id": state.session_id,
        "intent": intent,
        "tool": tool or "",
        "query": query or "",
        "trace": trace,
        "raw_response": {},
        "formatted_answer": "",
        "error_count": state.error_count,
        "last_error": state.last_error
    }

def execute_tool_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Execute the selected tool and format response"""
    logger.info(f"Executing tool: {state.tool} with query: {state.query}")
    
    if not state.tool:
        formatted_answer = "⚠️ **No valid tool selected.** Please rephrase your question to be more specific about what you want to do with the Neo4j database."
        return {**state.dict(), "formatted_answer": formatted_answer}
    
    # Call FastAPI server
    raw_response = call_fastapi_server(state.tool, state.query)
    
    if not raw_response.get("success", False):
        error = raw_response.get("error", "Unknown error")
        formatted_answer = f"⚠️ **Error:** {error}"
        return {
            **state.dict(),
            "raw_response": raw_response,
            "formatted_answer": formatted_answer,
            "error_count": state.error_count + 1,
            "last_error": error
        }
    
    # Format the response for user display
    formatted_answer = format_fastapi_response(state.tool, state.query, raw_response)
    
    return {
        **state.dict(),
        "raw_response": raw_response,
        "formatted_answer": formatted_answer
    }

# ============================================
# BUILD AGENT
# ============================================

def build_agent():
    """Build the fixed LangGraph agent"""
    logger.info("Building fixed LangGraph agent...")
    
    workflow = StateGraph(state_schema=AgentState)
    
    # Add nodes
    workflow.add_node("analyze_and_select", RunnableLambda(analyze_and_select_tool_node))
    workflow.add_node("execute_tool", RunnableLambda(execute_tool_node))
    
    # Set entry point and edges
    workflow.set_entry_point("analyze_and_select")
    workflow.add_edge("analyze_and_select", "execute_tool")
    workflow.add_edge("execute_tool", END)
    
    agent = workflow.compile()
    logger.info("✅ Fixed LangGraph agent built successfully")
    
    return agent

# ============================================
# TESTING FUNCTIONS
# ============================================

def test_agent_with_examples():
    """Test the agent with various examples to verify CREATE/DELETE display"""
    agent = build_agent()
    
    test_cases = [
        "How many nodes are in the graph?",
        "Create a Person named Alice with age 30",
        "Create 3 test nodes for deletion",
        "Delete all TestNode nodes",
        "Show me the database schema",
        "Update all Person nodes to set active = true"
    ]
    
    print("🧪 Testing Fixed LangGraph Agent")
    print("=" * 50)
    
    for i, question in enumerate(test_cases, 1):
        print(f"\n🔸 Test {i}: {question}")
        print("-" * 30)
        
        try:
            state = AgentState(
                question=question,
                session_id=f"test_session_{i}"
            )
            
            result = agent.invoke(state)
            
            print(f"Intent: {result.get('intent', 'N/A')}")
            print(f"Tool: {result.get('tool', 'N/A')}")
            print(f"Query: {result.get('query', 'N/A')}")
            print(f"Answer: {result.get('formatted_answer', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Testing completed")

if __name__ == "__main__":
    test_agent_with_examples()
