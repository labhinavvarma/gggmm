# ============================================
# 1. HARDCODED MCP SERVER (mcpserver.py)
# ============================================

"""
MCP Server with Hardcoded Configuration
Run this FIRST on port 8000
"""

import asyncio
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import AsyncGraphDatabase, AsyncDriver
import traceback
import uvicorn

# ============================================
# 🔧 HARDCODED CONFIGURATION - CHANGE THESE VALUES
# ============================================

# Neo4j Database Configuration
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_neo4j_password"  # ⚠️ CHANGE THIS!
NEO4J_DATABASE = "neo4j"

# Server Configuration
MCP_SERVER_PORT = 8000
MCP_SERVER_HOST = "0.0.0.0"

# Logging Configuration
LOG_LEVEL = "INFO"
ENABLE_DEBUG = True

# ============================================
# END OF CONFIGURATION
# ============================================

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_server")

# Print configuration on startup
logger.info("🔧 MCP Server Configuration:")
logger.info(f"   Neo4j URI: {NEO4J_URI}")
logger.info(f"   Neo4j User: {NEO4J_USER}")
logger.info(f"   Neo4j Database: {NEO4J_DATABASE}")
logger.info(f"   Server Port: {MCP_SERVER_PORT}")

# Initialize Neo4j driver
driver = AsyncGraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
    connection_timeout=10,
    max_connection_lifetime=3600,
    max_connection_pool_size=50
)

app = FastAPI(
    title="MCP Neo4j Server",
    description="MCP Server for Neo4j operations",
    version="1.0.0"
)

class CypherRequest(BaseModel):
    query: str
    params: dict = {}

@app.post("/read_neo4j_cypher")
async def read_neo4j_cypher(request: CypherRequest):
    """Execute read-only Cypher queries"""
    try:
        logger.info(f"Executing READ query: {request.query}")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run(request.query, request.params)
            records = await result.data()
            
        logger.info(f"Query returned {len(records)} records")
        return records
        
    except Exception as e:
        logger.error(f"Read query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/write_neo4j_cypher")
async def write_neo4j_cypher(request: CypherRequest):
    """Execute write Cypher queries"""
    try:
        logger.info(f"Executing WRITE query: {request.query}")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run(request.query, request.params)
            summary = await result.consume()
            
        # Get counters
        counters = summary.counters
        
        response = {
            "success": True,
            "nodes_created": counters.nodes_created,
            "nodes_deleted": counters.nodes_deleted,
            "relationships_created": counters.relationships_created,
            "relationships_deleted": counters.relationships_deleted,
            "properties_set": counters.properties_set,
            "labels_added": counters.labels_added,
            "labels_removed": counters.labels_removed
        }
        
        logger.info(f"Write query completed: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Write query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/get_neo4j_schema")
async def get_neo4j_schema():
    """Get database schema"""
    try:
        logger.info("Fetching database schema")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            # Try APOC first
            try:
                apoc_result = await session.run("CALL apoc.meta.schema() YIELD value RETURN value")
                apoc_record = await apoc_result.single()
                if apoc_record:
                    schema = apoc_record["value"]
                    logger.info("Schema fetched using APOC")
                    return schema
            except Exception:
                logger.info("APOC not available, using fallback queries")
            
            # Fallback to basic queries
            labels_result = await session.run("CALL db.labels() YIELD label RETURN collect(label) as labels")
            labels_record = await labels_result.single()
            labels = labels_record["labels"] if labels_record else []
            
            rels_result = await session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types")
            rels_record = await rels_result.single()
            rel_types = rels_record["types"] if rels_record else []
            
            props_result = await session.run("CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) as keys")
            props_record = await props_result.single()
            prop_keys = props_record["keys"] if props_record else []
        
        schema = {
            "labels": labels,
            "relationship_types": rel_types,
            "property_keys": prop_keys,
            "source": "fallback_queries"
        }
        
        logger.info(f"Schema fetched: {len(labels)} labels, {len(rel_types)} rel types, {len(prop_keys)} properties")
        return schema
        
    except Exception as e:
        logger.error(f"Schema fetch failed: {e}")
        return {
            "labels": [],
            "relationship_types": [],
            "property_keys": [],
            "error": f"Schema fetch failed: {str(e)}",
            "source": "error"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run("RETURN 1 as test")
            record = await result.single()
            
        neo4j_status = "connected" if record and record["test"] == 1 else "test_failed"
        
        return {
            "status": "healthy",
            "neo4j": {
                "status": neo4j_status,
                "uri": NEO4J_URI,
                "database": NEO4J_DATABASE,
                "user": NEO4J_USER
            },
            "server": {
                "port": MCP_SERVER_PORT,
                "host": MCP_SERVER_HOST
            }
        }
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy", 
            "neo4j": {
                "status": "disconnected",
                "error": str(e)
            }
        }

@app.on_event("startup")
async def startup_event():
    """Test connection on startup"""
    logger.info("🚀 Starting MCP Neo4j Server...")
    
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run("RETURN 1 as test")
            record = await result.single()
            
        if record and record["test"] == 1:
            logger.info("✅ Neo4j connection successful!")
        else:
            logger.error("❌ Neo4j connection test failed")
            
    except Exception as e:
        logger.error(f"❌ Neo4j connection failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown"""
    logger.info("🛑 Shutting down MCP Neo4j Server...")
    await driver.close()

if __name__ == "__main__":
    logger.info("🌐 Starting MCP server on port 8000...")
    uvicorn.run(
        "mcpserver:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )

# ============================================
# 2. LANGGRAPH AGENT (langgraph_agent.py)
# ============================================

"""
Complete LangGraph Agent with improved connectivity
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
from typing import Dict, Any, Optional

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langgraph_agent")

# Configuration
API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"  # Change this!
MODEL = "llama3.1-70b"
MCP_BASE_URL = "http://localhost:8000"

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
    """Call Cortex LLM with improved error handling"""
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
        
        logger.info(f"Calling Cortex LLM...")
        response = requests.post(API_URL, headers=headers, json=payload, verify=False, timeout=60)
        
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
    """Call MCP server with improved error handling and retries"""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            headers = {"Content-Type": "application/json"}
            
            logger.info(f"Calling MCP server - Tool: {tool}, Query: {query}, Attempt: {attempt + 1}")
            
            if tool == "get_neo4j_schema":
                response = requests.post(f"{MCP_BASE_URL}/get_neo4j_schema", headers=headers, timeout=30)
                
            elif tool == "read_neo4j_cypher":
                if not query:
                    return {"error": "No query provided for read operation"}
                
                data = {"query": query, "params": {}}
                response = requests.post(f"{MCP_BASE_URL}/read_neo4j_cypher", json=data, headers=headers, timeout=30)
                
            elif tool == "write_neo4j_cypher":
                if not query:
                    return {"error": "No query provided for write operation"}
                    
                data = {"query": query, "params": {}}
                response = requests.post(f"{MCP_BASE_URL}/write_neo4j_cypher", json=data, headers=headers, timeout=30)
                
            else:
                return {"error": f"Unknown tool: {tool}"}
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"MCP server response successful")
                return {"success": True, "data": result}
            else:
                logger.error(f"MCP server error: {response.status_code} - {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return {"error": f"MCP server error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Failed to call MCP server (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return {"error": f"Failed to call MCP server after {max_retries} attempts: {str(e)}"}
    
    return {"error": "Max retries exceeded"}

SYSTEM_PROMPT = """
You are an expert AI assistant that helps users query and manage a Neo4j database by selecting and using one of three MCP tools.

TOOL DESCRIPTIONS:
- read_neo4j_cypher: Use for all read-only graph queries (MATCH, RETURN, WHERE, etc)
- write_neo4j_cypher: Use for write queries (CREATE, MERGE, SET, DELETE, etc)
- get_neo4j_schema: Use when asking about database structure, labels, relationships

IMPORTANT GUIDELINES:
- ALWAYS output your reasoning and then the tool and Cypher query
- Choose the tool based on the user's intent
- For schema questions, use get_neo4j_schema
- For data queries, use read_neo4j_cypher
- For data modifications, use write_neo4j_cypher

EXAMPLES:

User: How many nodes are in the graph?
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN count(n)

User: Create a Person named Alice
Tool: write_neo4j_cypher
Query: CREATE (:Person {name: 'Alice'})

User: Show the schema
Tool: get_neo4j_schema

ALWAYS explain your choice and provide the exact tool name and query.
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
    
    return tool, query, trace

def clean_cypher_query(query: str) -> str:
    """Clean Cypher query"""
    if not query:
        return ""
    
    # Remove code blocks
    query = re.sub(r'```[a-zA-Z]*', '', query)
    query = re.sub(r'```', '', query)
    
    # Clean whitespace
    query = re.sub(r'\s+', ' ', query)
    return query.strip()

def select_tool_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Select tool and generate query"""
    logger.info(f"Processing question: {state.question}")
    
    llm_output = call_cortex_llm(state.question, state.session_id)
    tool, query, trace = parse_llm_response(llm_output)
    
    if query:
        query = clean_cypher_query(query)
    
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

# ============================================
# 3. FASTAPI APP (app.py)
# ============================================

"""
FastAPI application with LangGraph agent
Run this SECOND on port 8081
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uuid
import logging
from typing import Optional
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our agent
from langgraph_agent import build_agent, AgentState

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastapi_app")

# Configuration
APP_PORT = 8081

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

@app.on_event("startup")
async def startup_event():
    """Initialize the agent when the app starts"""
    global agent
    try:
        logger.info("🚀 Starting Neo4j LangGraph FastAPI App...")
        
        # Test MCP server connection first
        try:
            import requests
            mcp_health = requests.get("http://localhost:8000/health", timeout=5)
            if mcp_health.status_code == 200:
                logger.info("✅ MCP server is running")
            else:
                logger.warning("⚠️ MCP server health check failed")
        except Exception as e:
            logger.error(f"❌ Cannot connect to MCP server: {e}")
            logger.error("❌ Make sure to start mcpserver.py first on port 8000!")
        
        # Build agent
        logger.info("Building LangGraph agent...")
        agent = build_agent()
        logger.info("✅ Agent built successfully")
        logger.info(f"🌐 App ready on port {APP_PORT}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize app: {e}")
        raise

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        if agent is None:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info(f"Processing question: {request.question}")
        
        # Create agent state
        state = AgentState(
            question=request.question,
            session_id=session_id
        )
        
        # Run the agent
        try:
            result = await agent.ainvoke(state)
            
            logger.info(f"Agent completed - Tool: {result.get('tool')}")
            
            return ChatResponse(
                trace=result.get("trace", ""),
                tool=result.get("tool", ""),
                query=result.get("query", ""),
                answer=result.get("answer", "No answer generated"),
                session_id=session_id,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return ChatResponse(
                trace=f"Agent error: {str(e)}",
                tool="",
                query="",
                answer=f"⚠️ Agent execution failed: {str(e)}",
                session_id=session_id,
                success=False,
                error=str(e)
            )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        status = {"app": "healthy", "agent": "not_initialized" if agent is None else "ready"}
        
        # Check MCP server
        try:
            import requests
            mcp_response = requests.get("http://localhost:8000/health", timeout=5)
            status["mcp_server"] = "healthy" if mcp_response.status_code == 200 else "unhealthy"
        except:
            status["mcp_server"] = "disconnected"
        
        return status
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Neo4j LangGraph Agent API",
        "version": "1.0.0",
        "status": "ready" if agent else "initializing",
        "endpoints": {
            "chat": "/chat",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting FastAPI app...")
    logger.info("⚠️ Make sure MCP server is running on port 8000!")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=APP_PORT,
        reload=True,
        log_level="info"
    )

# ============================================
# 4. STREAMLIT UI (ui.py)
# ============================================

"""
Streamlit UI for Neo4j LangGraph Agent
Run this LAST after starting the other components
"""

import streamlit as st
import requests
import uuid
import json
import time
from datetime import datetime

# Configuration
APP_PORT = 8081
MCP_PORT = 8000

# Page configuration
st.set_page_config(
    page_title="Neo4j LangGraph Agent",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    
    .status-healthy {
        background: linear-gradient(90deg, #00d4aa 0%, #00b894 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin: 0.2rem;
        display: inline-block;
        font-weight: bold;
    }
    
    .status-error {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin: 0.2rem;
        display: inline-block;
        font-weight: bold;
    }
    
    .tool-badge {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .query-display {
        background: #1e1e1e;
        color: #f8f8f2;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        border-left: 4px solid #50fa7b;
    }
    
    .result-display {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00d4aa;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

def check_system_health():
    """Check health of all components"""
    health_status = {}
    
    # Check FastAPI app
    try:
        app_response = requests.get(f"http://localhost:{APP_PORT}/health", timeout=5)
        if app_response.status_code == 200:
            health_data = app_response.json()
            health_status["app"] = {"status": "healthy", "data": health_data}
        else:
            health_status["app"] = {"status": "error", "error": f"HTTP {app_response.status_code}"}
    except Exception as e:
        health_status["app"] = {"status": "error", "error": str(e)}
    
    # Check MCP server
    try:
        mcp_response = requests.get(f"http://localhost:{MCP_PORT}/health", timeout=5)
        if mcp_response.status_code == 200:
            health_status["mcp"] = {"status": "healthy", "data": mcp_response.json()}
        else:
            health_status["mcp"] = {"status": "error", "error": f"HTTP {mcp_response.status_code}"}
    except Exception as e:
        health_status["mcp"] = {"status": "error", "error": str(e)}
    
    return health_status

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
            timeout=60
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
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "answer": f"❌ Request failed: {str(e)}",
            "response_time": 0
        }

# Main UI
st.title("🧠 Neo4j LangGraph Agent")
st.markdown("**AI-powered database assistant with LangGraph workflow**")

# Sidebar with system status
with st.sidebar:
    st.markdown("## 🔧 System Status")
    
    # Check system health
    health_status = check_system_health()
    
    # Display component status
    for component, status in health_status.items():
        if status["status"] == "healthy":
            st.markdown(f"""
            <div class="status-healthy">
                ✅ {component.upper()}: Online
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-error">
                ❌ {component.upper()}: {status.get('error', 'Error')}
            </div>
            """, unsafe_allow_html=True)
    
    # Connection instructions
    st.markdown("## 📋 Setup Instructions")
    st.markdown("""
    **Required Order:**
    1. Start `mcpserver.py` (port 8000)
    2. Start `app.py` (port 8081)  
    3. Start `ui.py` (this interface)
    
    **Configuration:**
    - Update Neo4j password in mcpserver.py
    - Update Cortex API key in langgraph_agent.py
    """)
    
    # Example queries
    st.markdown("## 💡 Example Queries")
    
    examples = [
        "How many nodes are in the graph?",
        "Show me the database schema",
        "Create a Person named Alice",
        "List all node labels",
        "Find nodes with most connections"
    ]
    
    for example in examples:
        if st.button(example, key=f"example_{hash(example)}"):
            st.session_state.selected_query = example

# Main chat interface
st.markdown("## 💬 Chat Interface")

# Chat input
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "Ask me anything about your Neo4j database:",
        placeholder="e.g., How many nodes are in the graph?",
        height=100
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        submitted = st.form_submit_button("🚀 Send Query", use_container_width=True)
    
    with col2:
        if st.form_submit_button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

# Handle example query selection
if hasattr(st.session_state, 'selected_query'):
    user_input = st.session_state.selected_query
    submitted = True
    st.info(f"Running example: {user_input}")
    delattr(st.session_state, 'selected_query')

# Process query
if submitted and user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })
    
    # Send to agent
    with st.spinner("🧠 Processing with LangGraph agent..."):
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
    else:
        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    # Display result components
    if result.get("tool"):
        st.markdown(f"""
        <div class="tool-badge">
            🔧 Tool: {result["tool"]}
        </div>
        """, unsafe_allow_html=True)
    
    if result.get("query"):
        st.markdown("**Generated Query:**")
        st.markdown(f"""
        <div class="query-display">
            {result["query"]}
        </div>
        """, unsafe_allow_html=True)
    
    if result.get("answer"):
        st.markdown("**Result:**")
        st.markdown(f"""
        <div class="result-display">
            {result["answer"]}
        </div>
        """, unsafe_allow_html=True)

# Chat history
st.markdown("---")
st.markdown("## 📝 Chat History")

if st.session_state.messages:
    for message in reversed(st.session_state.messages[-10:]):  # Show last 10
        if message["role"] == "user":
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;">
                <strong>🧑 You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            result = message["content"]
            
            with st.expander(f"🤖 Agent Response - {result.get('tool', 'Unknown Tool')}", expanded=False):
                if result.get("tool"):
                    st.markdown(f"**Tool:** {result['tool']}")
                
                if result.get("query"):
                    st.markdown("**Query:**")
                    st.code(result["query"], language="cypher")
                
                if result.get("answer"):
                    st.markdown("**Answer:**")
                    st.markdown(result["answer"])
                
                if result.get("trace"):
                    st.markdown("**Trace:**")
                    st.text(result["trace"])
else:
    st.info("🌟 Welcome! Ask any question about your Neo4j database.")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>🧠 Neo4j LangGraph Agent</strong></p>
    <p>Session: <code>{st.session_state.session_id[:8]}...</code></p>
</div>
""", unsafe_allow_html=True)

# ============================================
# 5. STARTUP SCRIPT (start_system.py)
# ============================================

"""
Startup script to run all components in order
"""

import subprocess
import time
import sys
import os

def start_component(script_name, port, component_name):
    """Start a component and return the process"""
    print(f"🚀 Starting {component_name} on port {port}...")
    
    try:
        process = subprocess.Popen([
            sys.executable, script_name
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print(f"✅ {component_name} started with PID {process.pid}")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start {component_name}: {e}")
        return None

def main():
    print("🧠 Starting Complete Neo4j LangGraph System")
    print("=" * 50)
    
    processes = []
    
    # Start MCP Server
    mcp_process = start_component("mcpserver.py", 8000, "MCP Server")
    if mcp_process:
        processes.append(mcp_process)
        time.sleep(3)  # Wait for MCP server to start
    
    # Start FastAPI App
    app_process = start_component("app.py", 8081, "FastAPI App")
    if app_process:
        processes.append(app_process)
        time.sleep(3)  # Wait for app to start
    
    # Start Streamlit UI
    print("🚀 Starting Streamlit UI...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "ui.py"])
    except KeyboardInterrupt:
        print("\n🛑 Shutting down all components...")
        for process in processes:
            process.terminate()
        print("✅ All components stopped")

if __name__ == "__main__":
    main()
