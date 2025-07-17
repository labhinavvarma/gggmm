# ============================================
# STEP-BY-STEP STARTUP DEBUG GUIDE
# ============================================

"""
Follow this exact order to identify and fix connection issues:
"""

# ============================================
# STEP 1: Clean LangGraph Agent (langgraph_agent.py)
# ============================================

"""
Save this as langgraph_agent.py
This has NO Streamlit dependencies and clean connections
"""

import requests
import urllib3
import json
import logging
import re
import time
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import Dict, Any

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langgraph_agent")

# ============================================
# 🔧 CONFIGURATION - CHANGE THESE VALUES
# ============================================

# Cortex API Configuration
API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"  # ⚠️ CHANGE THIS!
MODEL = "llama3.1-70b"

# MCP Server Configuration
MCP_BASE_URL = "http://localhost:8000"

# ============================================

print("🔧 LangGraph Agent Configuration:")
print(f"   Cortex API: {API_URL}")
print(f"   API Key Length: {len(API_KEY)} characters")
print(f"   Model: {MODEL}")
print(f"   MCP Server: {MCP_BASE_URL}")

class AgentState(BaseModel):
    question: str
    session_id: str
    tool: str = ""
    query: str = ""
    trace: str = ""
    answer: str = ""
    error_count: int = 0
    last_error: str = ""

def call_cortex_llm(prompt: str, session_id: str) -> str:
    """Call Cortex LLM with error handling"""
    try:
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
                "sys_msg": SYSTEM_PROMPT,
                "limit_convs": "0",
                "prompt": {
                    "messages": [{"role": "user", "content": prompt}]
                },
                "session_id": session_id
            }
        }
        
        logger.info("Calling Cortex LLM...")
        response = requests.post(API_URL, headers=headers, json=payload, verify=False, timeout=30)
        
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

def call_mcp_server(tool: str, query: str = None) -> Dict[str, Any]:
    """Call MCP server with retries"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            headers = {"Content-Type": "application/json"}
            
            logger.info(f"Calling MCP server - Tool: {tool}, Attempt: {attempt + 1}")
            
            if tool == "get_neo4j_schema":
                response = requests.post(f"{MCP_BASE_URL}/get_neo4j_schema", headers=headers, timeout=15)
                
            elif tool == "read_neo4j_cypher":
                if not query:
                    return {"error": "No query provided for read operation"}
                
                data = {"query": query, "params": {}}
                response = requests.post(f"{MCP_BASE_URL}/read_neo4j_cypher", json=data, headers=headers, timeout=15)
                
            elif tool == "write_neo4j_cypher":
                if not query:
                    return {"error": "No query provided for write operation"}
                    
                data = {"query": query, "params": {}}
                response = requests.post(f"{MCP_BASE_URL}/write_neo4j_cypher", json=data, headers=headers, timeout=15)
                
            else:
                return {"error": f"Unknown tool: {tool}"}
            
            if response.status_code == 200:
                result = response.json()
                logger.info("MCP server response successful")
                return {"success": True, "data": result}
            else:
                logger.error(f"MCP server error: {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return {"error": f"MCP server error: {response.status_code}"}
                
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to MCP server at {MCP_BASE_URL}")
            return {"error": f"Cannot connect to MCP server. Is it running on port 8000?"}
        except Exception as e:
            logger.error(f"MCP server call failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return {"error": f"MCP server failed after {max_retries} attempts: {str(e)}"}
    
    return {"error": "Max retries exceeded"}

SYSTEM_PROMPT = """
You are an expert AI assistant that helps users query and manage a Neo4j database.

TOOL DESCRIPTIONS:
- read_neo4j_cypher: Use for all read-only queries (MATCH, RETURN, WHERE, etc)
- write_neo4j_cypher: Use for write queries (CREATE, MERGE, SET, DELETE, etc)
- get_neo4j_schema: Use when asking about database structure

GUIDELINES:
- ALWAYS output your reasoning and then the tool and query
- Choose the appropriate tool based on the user's intent

EXAMPLES:

User: How many nodes are in the graph?
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN count(n)

User: Create a Person named Alice
Tool: write_neo4j_cypher
Query: CREATE (:Person {name: 'Alice'})

User: Show the schema
Tool: get_neo4j_schema

ALWAYS provide the exact tool name and query.
"""

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

def select_tool_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Select tool and generate query"""
    logger.info(f"Processing question: {state.question}")
    
    llm_output = call_cortex_llm(state.question, state.session_id)
    tool, query, trace = parse_llm_response(llm_output)
    
    return {
        "question": state.question,
        "session_id": state.session_id,
        "tool": tool or "",
        "query": query or "",
        "trace": trace,
        "answer": "",
        "error_count": state.error_count,
        "last_error": state.last_error
    }

def execute_tool_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Execute the selected tool"""
    logger.info(f"Executing tool: {state.tool}")
    
    if not state.tool:
        answer = "⚠️ No valid tool selected. Please rephrase your question."
        return {**state.dict(), "answer": answer}
    
    # Call MCP server
    result = call_mcp_server(state.tool, state.query)
    
    if "error" in result:
        answer = f"⚠️ Error: {result['error']}"
        return {
            **state.dict(),
            "answer": answer,
            "error_count": state.error_count + 1,
            "last_error": result['error']
        }
    
    # Format successful result
    data = result.get("data", {})
    
    if state.tool == "get_neo4j_schema":
        if isinstance(data, dict):
            labels = data.get("labels", [])
            rel_types = data.get("relationship_types", [])
            answer = f"📊 **Database Schema:**\n\n**Node Labels:** {', '.join(labels[:10])}\n**Relationship Types:** {', '.join(rel_types[:10])}"
        else:
            answer = f"📊 **Schema:** {json.dumps(data, indent=2)[:500]}..."
    
    elif state.tool == "read_neo4j_cypher":
        if isinstance(data, list):
            count = len(data)
            if count == 0:
                answer = "📊 **Result:** No data found"
            elif count == 1 and isinstance(data[0], dict) and len(data[0]) == 1:
                # Single value result (like count)
                key, value = list(data[0].items())[0]
                answer = f"📊 **Result:** {value}"
            else:
                answer = f"📊 **Result:** Found {count} records\n\n{json.dumps(data[:3], indent=2)}"
                if count > 3:
                    answer += f"\n... and {count - 3} more records"
        else:
            answer = f"📊 **Result:** {json.dumps(data, indent=2)[:500]}"
    
    elif state.tool == "write_neo4j_cypher":
        if isinstance(data, dict):
            created = data.get("nodes_created", 0)
            deleted = data.get("nodes_deleted", 0)
            answer = f"✅ **Write Operation Completed:**\n- Nodes created: {created}\n- Nodes deleted: {deleted}"
        else:
            answer = f"✅ **Write operation completed:** {data}"
    
    else:
        answer = f"📊 **Result:** {json.dumps(data, indent=2)[:500]}"
    
    return {**state.dict(), "answer": answer}

def build_agent():
    """Build the LangGraph agent"""
    logger.info("Building LangGraph agent...")
    
    workflow = StateGraph(state_schema=AgentState)
    
    # Add nodes
    workflow.add_node("select_tool", RunnableLambda(select_tool_node))
    workflow.add_node("execute_tool", RunnableLambda(execute_tool_node))
    
    # Set entry point and edges
    workflow.set_entry_point("select_tool")
    workflow.add_edge("select_tool", "execute_tool")
    workflow.add_edge("execute_tool", END)
    
    agent = workflow.compile()
    logger.info("✅ LangGraph agent built successfully")
    
    return agent

# Test function
def test_agent():
    """Test the agent locally"""
    print("🧪 Testing LangGraph Agent...")
    
    try:
        agent = build_agent()
        print("✅ Agent built successfully")
        
        # Test MCP connection
        result = call_mcp_server("get_neo4j_schema")
        if "error" in result:
            print(f"❌ MCP connection failed: {result['error']}")
        else:
            print("✅ MCP server connection successful")
            
        return agent
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return None

if __name__ == "__main__":
    test_agent()

# ============================================
# STEP 2: Clean FastAPI App (app.py)
# ============================================

"""
Save this as app.py
This imports the LangGraph agent cleanly
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import logging
import requests
import time
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastapi_app")

# Configuration
APP_PORT = 8081
MCP_PORT = 8000

print("🔧 FastAPI App Configuration:")
print(f"   App Port: {APP_PORT}")
print(f"   MCP Port: {MCP_PORT}")

# Initialize FastAPI app
app = FastAPI(
    title="Neo4j LangGraph Agent API",
    description="FastAPI server for Neo4j LangGraph agent",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent variable
agent = None

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    trace: str
    tool: str
    query: str
    answer: str
    session_id: str
    success: bool = True
    error: Optional[str] = None

def check_mcp_server():
    """Check if MCP server is running"""
    try:
        response = requests.get(f"http://localhost:{MCP_PORT}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the agent when the app starts"""
    global agent
    
    print("🚀 Starting Neo4j LangGraph FastAPI App...")
    print("=" * 50)
    
    # Check MCP server first
    print("🔍 Checking MCP server connection...")
    if check_mcp_server():
        print("✅ MCP server is running and accessible")
    else:
        print("❌ Cannot connect to MCP server!")
        print(f"❌ Please make sure mcpserver.py is running on port {MCP_PORT}")
        print("❌ FastAPI app will start but agent functionality will be limited")
    
    # Import and build agent
    try:
        print("📦 Importing LangGraph agent...")
        
        # Import the agent module
        from langgraph_agent import build_agent, AgentState
        
        print("🔨 Building LangGraph agent...")
        agent = build_agent()
        print("✅ Agent built successfully")
        
        print(f"🌐 FastAPI app ready on port {APP_PORT}")
        print("=" * 50)
        
    except ImportError as e:
        print(f"❌ Failed to import langgraph_agent: {e}")
        print("❌ Make sure langgraph_agent.py is in the same directory")
        agent = None
    except Exception as e:
        print(f"❌ Failed to build agent: {e}")
        agent = None

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    if agent is None:
        raise HTTPException(
            status_code=503, 
            detail="Agent not initialized. Check server logs for import errors."
        )
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info(f"Processing question: {request.question}")
        
        # Import AgentState here to avoid circular imports
        from langgraph_agent import AgentState
        
        # Create agent state
        state = AgentState(
            question=request.question,
            session_id=session_id
        )
        
        # Run the agent
        start_time = time.time()
        result = await agent.ainvoke(state)
        processing_time = time.time() - start_time
        
        logger.info(f"Agent completed in {processing_time:.2f}s - Tool: {result.get('tool')}")
        
        return ChatResponse(
            trace=result.get("trace", ""),
            tool=result.get("tool", ""),
            query=result.get("query", ""),
            answer=result.get("answer", "No answer generated"),
            session_id=session_id,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return ChatResponse(
            trace=f"Error: {str(e)}",
            tool="",
            query="",
            answer=f"⚠️ Error processing request: {str(e)}",
            session_id=request.session_id or str(uuid.uuid4()),
            success=False,
            error=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "app": "healthy",
        "agent": "ready" if agent else "not_initialized",
        "mcp_server": "healthy" if check_mcp_server() else "disconnected"
    }
    
    return status

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Neo4j LangGraph Agent API",
        "version": "1.0.0",
        "status": "ready" if agent else "initializing",
        "ports": {
            "app": APP_PORT,
            "mcp": MCP_PORT
        }
    }

def main():
    """Main function to run the FastAPI server"""
    print("=" * 60)
    print("🚀 NEO4J LANGGRAPH FASTAPI APP")
    print("=" * 60)
    print("📋 Prerequisites:")
    print("   1. MCP server running on port 8000")
    print("   2. langgraph_agent.py in same directory")
    print("   3. Neo4j database accessible")
    print("=" * 60)
    
    # Check prerequisites
    print("🔍 Checking prerequisites...")
    
    # Check if langgraph_agent.py exists
    import os
    if os.path.exists("langgraph_agent.py"):
        print("✅ langgraph_agent.py found")
    else:
        print("❌ langgraph_agent.py not found in current directory")
    
    # Check MCP server
    if check_mcp_server():
        print("✅ MCP server is accessible")
    else:
        print("❌ MCP server not accessible - start mcpserver.py first")
    
    print("=" * 60)
    print("🚀 Starting FastAPI server...")
    
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=APP_PORT,
        log_level="info",
        reload=False
    )

if __name__ == "__main__":
    main()

# ============================================
# STEP 3: Clean Streamlit UI (ui.py)
# ============================================

"""
Save this as ui.py
Run this LAST after the other components are running
"""

import streamlit as st
import requests
import uuid
import time
from datetime import datetime

# Configuration
APP_PORT = 8081  # FastAPI app port
MCP_PORT = 8000  # MCP server port

# Page configuration
st.set_page_config(
    page_title="Neo4j LangGraph Agent",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

def check_component_health(port: int, component_name: str) -> dict:
    """Check health of a specific component"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "data": response.json()}
        else:
            return {"status": "error", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "disconnected", "error": f"{component_name} not running on port {port}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def send_chat_message(question: str):
    """Send chat message to FastAPI app"""
    try:
        payload = {
            "question": question,
            "session_id": st.session_state.session_id
        }
        
        start_time = time.time()
        
        response = requests.post(
            f"http://localhost:{APP_PORT}/chat",
            json=payload,
            timeout=30
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            result["response_time"] = response_time
            return result
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "answer": f"❌ Server error: {response.status_code}",
                "response_time": response_time
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to FastAPI app",
            "answer": "❌ FastAPI app not running. Start app.py on port 8081.",
            "response_time": 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "answer": f"❌ Request failed: {str(e)}",
            "response_time": 0
        }

# Main UI
st.title("🧠 Neo4j LangGraph Agent")
st.markdown("**AI-powered database assistant**")

# Sidebar with system status
with st.sidebar:
    st.markdown("## 🔧 System Status")
    
    # Check MCP Server
    mcp_health = check_component_health(MCP_PORT, "MCP Server")
    if mcp_health["status"] == "healthy":
        st.success("🟢 MCP Server: Online")
    else:
        st.error(f"🔴 MCP Server: {mcp_health['error']}")
    
    # Check FastAPI App
    app_health = check_component_health(APP_PORT, "FastAPI App")
    if app_health["status"] == "healthy":
        st.success("🟢 FastAPI App: Online")
        # Show agent status if available
        if "data" in app_health and "agent" in app_health["data"]:
            agent_status = app_health["data"]["agent"]
            if agent_status == "ready":
                st.success("🤖 Agent: Ready")
            else:
                st.warning(f"🤖 Agent: {agent_status}")
    else:
        st.error(f"🔴 FastAPI App: {app_health['error']}")
    
    # Setup instructions
    st.markdown("## 📋 Setup Order")
    st.markdown("""
    1. Start `mcpserver.py` (port 8000)
    2. Start `app.py` (port 8081)  
    3. Start `ui.py` (this interface)
    """)
    
    # Example queries
    st.markdown("## 💡 Examples")
    examples = [
        "How many nodes are in the graph?",
        "Show me the database schema",
        "Create a Person named Alice",
        "List all node labels"
    ]
    
    for example in examples:
        if st.button(example, key=f"ex_{hash(example)}"):
            st.session_state.example_query = example

# Main chat interface
st.markdown("## 💬 Chat Interface")

# Chat input
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "Ask about your Neo4j database:",
        placeholder="e.g., How many nodes are in the graph?",
        height=100
    )
    
    submitted = st.form_submit_button("🚀 Send Query", use_container_width=True)

# Handle example selection
if hasattr(st.session_state, 'example_query'):
    user_input = st.session_state.example_query
    submitted = True
    st.info(f"Running example: {user_input}")
    delattr(st.session_state, 'example_query')

# Process query
if submitted and user_input:
    # Check if FastAPI is available first
    app_health = check_component_health(APP_PORT, "FastAPI App")
    if app_health["status"] != "healthy":
        st.error("❌ Cannot send query: FastAPI app is not running!")
        st.error("Please start app.py first on port 8081")
    else:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send to agent
        with st.spinner("🧠 Processing..."):
            result = send_chat_message(user_input)
        
        # Add agent response
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result,
            "timestamp": datetime.now().isoformat()
        })
        
        # Show immediate result
        st.markdown("---")
        
        if result.get("success", True):
            response_time = result.get("response_time", 0)
            st.success(f"✅ Query processed in {response_time:.2f}s")
            
            # Display components
            if result.get("tool"):
                st.info(f"🔧 Tool: {result['tool']}")
            
            if result.get("query"):
                st.code(result["query"], language="cypher")
            
            if result.get("answer"):
                st.markdown(f"**Result:** {result['answer']}")
        else:
            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

# Chat history
if st.session_state.messages:
    st.markdown("---")
    st.markdown("## 📝 Recent Messages")
    
    for message in reversed(st.session_state.messages[-5:]):  # Show last 5
        if message["role"] == "user":
            st.markdown(f"**🧑 You:** {message['content']}")
        else:
            result = message["content"]
            with st.expander(f"🤖 Agent: {result.get('tool', 'Response')}", expanded=False):
                if result.get("answer"):
                    st.markdown(result["answer"])
                if result.get("query"):
                    st.code(result["query"], language="cypher")
else:
    st.info("🌟 Welcome! Ask any question about your Neo4j database.")

# ============================================
# STEP 4: Debug Script (debug_system.py)
# ============================================

"""
Save this as debug_system.py
Run this to check all components step by step
"""

import requests
import time
import subprocess
import sys
import os

def check_port(port: int, service_name: str):
    """Check if a service is running on a specific port"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ {service_name} (port {port}): Online")
            return True
        else:
            print(f"❌ {service_name} (port {port}): HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ {service_name} (port {port}): Not running")
        return False
    except Exception as e:
        print(f"❌ {service_name} (port {port}): Error - {e}")
        return False

def check_files():
    """Check if all required files exist"""
    required_files = ["mcpserver.py", "langgraph_agent.py", "app.py", "ui.py"]
    
    print("🔍 Checking required files...")
    all_exist = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}: Found")
        else:
            print(f"❌ {file}: Missing")
            all_exist = False
    
    return all_exist

def test_mcp_connection():
    """Test MCP server connection"""
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            neo4j_status = health_data.get("neo4j", {}).get("status", "unknown")
            print(f"✅ MCP Server: Online (Neo4j: {neo4j_status})")
            return True
        else:
            print(f"❌ MCP Server: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ MCP Server: {e}")
        return False

def test_fas
