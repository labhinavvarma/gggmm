"""
Complete FastAPI Application with Fixed LangGraph Agent
This combines the FastAPI server with the fixed LangGraph agent
Run this on port 8081 (after starting the tool server on port 8000)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import logging
import requests
import time
from typing import Optional
import sys
import os

# Import our fixed agent
from fixed_langgraph_agent import build_agent, AgentState

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("complete_fastapi_app")

# Configuration
APP_PORT = 8081
TOOL_SERVER_PORT = 8000

print("🔧 Complete FastAPI App Configuration:")
print(f"   App Port: {APP_PORT}")
print(f"   Tool Server Port: {TOOL_SERVER_PORT}")

# Initialize FastAPI app
app = FastAPI(
    title="Complete Neo4j LangGraph Agent API",
    description="FastAPI server with fixed LangGraph agent that shows CREATE/DELETE counts",
    version="2.0.0"
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

# ============================================
# PYDANTIC MODELS
# ============================================

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    # Standard fields
    trace: str
    tool: str
    query: str
    answer: str
    session_id: str
    success: bool = True
    error: Optional[str] = None
    
    # Enhanced fields for better UI display
    intent: str = ""
    raw_response: dict = {}
    processing_time: float = 0.0

class HealthResponse(BaseModel):
    status: str
    components: dict
    agent_ready: bool

# ============================================
# UTILITY FUNCTIONS
# ============================================

def check_tool_server():
    """Check if the tool server is running"""
    try:
        response = requests.get(f"http://localhost:{TOOL_SERVER_PORT}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

# ============================================
# STARTUP/SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize the agent when the app starts"""
    global agent
    
    print("🚀 Starting Complete Neo4j LangGraph FastAPI App...")
    print("=" * 50)
    
    # Check tool server first
    print("🔍 Checking tool server connection...")
    tool_server_ok, tool_health = check_tool_server()
    
    if tool_server_ok:
        print("✅ Tool server is running and accessible")
        if tool_health:
            neo4j_status = tool_health.get("neo4j", {}).get("status", "unknown")
            print(f"   Neo4j Status: {neo4j_status}")
    else:
        print("❌ Cannot connect to tool server!")
        print(f"❌ Please make sure the tool server is running on port {TOOL_SERVER_PORT}")
        print("❌ App will start but agent functionality will be limited")
    
    # Build LangGraph agent
    try:
        print("🔨 Building fixed LangGraph agent...")
        agent = build_agent()
        print("✅ Fixed LangGraph agent built successfully")
        
        # Test the agent with a simple query
        print("🧪 Testing agent with sample query...")
        try:
            test_state = AgentState(
                question="How many nodes are in the graph?",
                session_id="startup_test"
            )
            test_result = await agent.ainvoke(test_state)
            print("✅ Agent test successful")
        except Exception as e:
            print(f"⚠️ Agent test failed: {e}")
        
        print(f"🌐 Complete FastAPI app ready on port {APP_PORT}")
        print("=" * 50)
        print("📋 Available endpoints:")
        print("   • GET  /health - Health check")
        print("   • POST /chat - Chat with fixed agent (shows CREATE/DELETE counts)")
        print("   • GET  /agent-info - Agent information")
        print("   • GET  /test-examples - Test agent with examples")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Failed to build agent: {e}")
        agent = None

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    print("🛑 Shutting down Complete FastAPI App...")
    print("✅ Shutdown complete")

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Complete Neo4j LangGraph Agent API",
        "version": "2.0.0",
        "description": "FastAPI server with fixed LangGraph agent that properly shows CREATE/DELETE counts",
        "features": [
            "Fixed CREATE/DELETE count display",
            "Proper response formatting",
            "Enhanced error handling",
            "Detailed operation summaries",
            "Real-time processing feedback"
        ],
        "endpoints": {
            "chat": "/chat - Chat with the fixed agent",
            "health": "/health - System health check",
            "agent_info": "/agent-info - Agent information",
            "test_examples": "/test-examples - Test with examples"
        },
        "status": "ready" if agent else "initializing"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    tool_server_ok, tool_health = check_tool_server()
    
    components = {
        "agent": {
            "status": "ready" if agent else "not_initialized",
            "type": "Fixed LangGraph Agent"
        },
        "tool_server": {
            "status": "healthy" if tool_server_ok else "disconnected",
            "port": TOOL_SERVER_PORT,
            "health_data": tool_health
        }
    }
    
    overall_status = "healthy" if agent and tool_server_ok else "degraded"
    
    return HealthResponse(
        status=overall_status,
        components=components,
        agent_ready=agent is not None
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint using fixed LangGraph agent"""
    if agent is None:
        raise HTTPException(
            status_code=503, 
            detail="Agent not initialized. Check server logs for errors."
        )
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info(f"Processing question: {request.question}")
        
        # Create agent state
        state = AgentState(
            question=request.question,
            session_id=session_id
        )
        
        # Run the agent and measure time
        start_time = time.time()
        result = await agent.ainvoke(state)
        processing_time = time.time() - start_time
        
        logger.info(f"Agent completed in {processing_time:.2f}s - Tool: {result.get('tool')}")
        
        return ChatResponse(
            # Standard fields for backward compatibility
            trace=result.get("trace", ""),
            tool=result.get("tool", ""),
            query=result.get("query", ""),
            answer=result.get("formatted_answer", "No answer generated"),
            session_id=session_id,
            success=True,
            
            # Enhanced fields
            intent=result.get("intent", ""),
            raw_response=result.get("raw_response", {}),
            processing_time=processing_time
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
            error=str(e),
            intent="error",
            processing_time=0.0
        )

@app.get("/agent-info")
async def get_agent_info():
    """Get information about the agent"""
    if agent is None:
        return {"status": "not_initialized", "agent": None}
    
    return {
        "status": "ready",
        "agent": {
            "type": "Fixed LangGraph Agent",
            "version": "2.0.0",
            "features": [
                "Proper CREATE/DELETE count display",
                "Enhanced response formatting", 
                "Better error handling",
                "Intent classification",
                "Detailed operation summaries"
            ],
            "nodes": [
                "analyze_and_select - Analyzes question and selects tool",
                "execute_tool - Executes tool and formats response"
            ],
            "supported_tools": [
                "read_neo4j_cypher - Execute read queries",
                "write_neo4j_cypher - Execute write queries (with proper counts)",
                "get_neo4j_schema - Get database schema"
            ],
            "improvements": [
                "Fixed node deletion counting",
                "Added relationship deletion tracking", 
                "Enhanced CREATE operation feedback",
                "Better query parsing and validation",
                "Improved error messages"
            ]
        }
    }

@app.get("/test-examples")
async def test_examples():
    """Test the agent with example queries"""
    if agent is None:
        return {"error": "Agent not initialized"}
    
    examples = [
        {
            "question": "How many nodes are in the graph?",
            "expected_tool": "read_neo4j_cypher",
            "expected_intent": "count"
        },
        {
            "question": "Create a Person named TestUser with age 25",
            "expected_tool": "write_neo4j_cypher", 
            "expected_intent": "create"
        },
        {
            "question": "Delete all TestNode nodes",
            "expected_tool": "write_neo4j_cypher",
            "expected_intent": "delete"
        },
        {
            "question": "Show me the database schema",
            "expected_tool": "get_neo4j_schema",
            "expected_intent": "schema"
        }
    ]
    
    results = []
    
    for i, example in enumerate(examples):
        try:
            state = AgentState(
                question=example["question"],
                session_id=f"test_{i}"
            )
            
            start_time = time.time()
            result = await agent.ainvoke(state)
            processing_time = time.time() - start_time
            
            results.append({
                "question": example["question"],
                "expected": {
                    "tool": example["expected_tool"],
                    "intent": example["expected_intent"]
                },
                "actual": {
                    "tool": result.get("tool", ""),
                    "intent": result.get("intent", ""),
                    "query": result.get("query", ""),
                    "answer": result.get("formatted_answer", "")[:200] + "..." if len(result.get("formatted_answer", "")) > 200 else result.get("formatted_answer", "")
                },
                "success": bool(result.get("tool")) and not result.get("last_error"),
                "processing_time": processing_time
            })
            
        except Exception as e:
            results.append({
                "question": example["question"],
                "expected": example,
                "actual": {"error": str(e)},
                "success": False,
                "processing_time": 0.0
            })
    
    success_count = sum(1 for r in results if r["success"])
    
    return {
        "total_tests": len(examples),
        "successful": success_count,
        "success_rate": f"{(success_count / len(examples) * 100):.1f}%",
        "results": results
    }

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """Main function to run the complete FastAPI application"""
    print("=" * 60)
    print("🚀 COMPLETE NEO4J LANGGRAPH FASTAPI APP")
    print("=" * 60)
    print("🏗️  Architecture: FastAPI + Fixed LangGraph + Tool Server")
    print("🔧 Configuration:")
    print(f"   📍 App Port: {APP_PORT}")
    print(f"   🛠️ Tool Server Port: {TOOL_SERVER_PORT}")
    print("=" * 60)
    print("✨ Key Fixes:")
    print("   • CREATE/DELETE counts now display properly")
    print("   • Enhanced response formatting")
    print("   • Better error handling and validation")
    print("   • Detailed operation summaries")
    print("   • Intent classification for queries")
    print("=" * 60)
    print("📋 Prerequisites:")
    print("   1. Tool server running on port 8000")
    print("   2. Neo4j database accessible")
    print("   3. Cortex API key configured")
    print("=" * 60)
    
    # Check prerequisites
    print("🔍 Checking prerequisites...")
    
    # Check if fixed_langgraph_agent.py exists
    if os.path.exists("fixed_langgraph_agent.py"):
        print("✅ fixed_langgraph_agent.py found")
    else:
        print("❌ fixed_langgraph_agent.py not found in current directory")
        print("❌ Please create this file from the artifact above")
        return
    
    # Check tool server
    tool_server_ok, _ = check_tool_server()
    if tool_server_ok:
        print("✅ Tool server is accessible")
    else:
        print("❌ Tool server not accessible - make sure it's running on port 8000")
    
    print("=" * 60)
    print("🚀 Starting complete FastAPI application...")
    
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
