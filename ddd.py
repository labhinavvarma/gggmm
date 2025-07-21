"""
Fixed FastAPI App - Resolves Pydantic validation error
This fixes the "graph_data Input should be a valid dictionary" error
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import logging
import requests
import time
from typing import Optional, Dict, Any
import sys
import os

# Import our robust agent
try:
    from robust_langgraph_agent import build_agent, AgentState
    AGENT_IMPORTED = True
except ImportError:
    try:
        from enhanced_langgraph_agent import build_agent, AgentState
        AGENT_IMPORTED = True
    except ImportError:
        try:
            from fixed_langgraph_agent import build_agent, AgentState
            AGENT_IMPORTED = True
        except ImportError:
            AGENT_IMPORTED = False
            print("❌ No agent module found. Please ensure one of the agent files exists.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fixed_fastapi_app")

# Configuration
APP_PORT = 8081
TOOL_SERVER_PORT = 8000

print("🔧 Fixed FastAPI App Configuration:")
print(f"   App Port: {APP_PORT}")
print(f"   Tool Server Port: {TOOL_SERVER_PORT}")
print(f"   Agent Imported: {AGENT_IMPORTED}")

# Initialize FastAPI app
app = FastAPI(
    title="Fixed Neo4j LangGraph Agent API",
    description="FastAPI server with fixed Pydantic validation",
    version="2.0.1"
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
# FIXED PYDANTIC MODELS
# ============================================

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    # Core required fields
    trace: str
    tool: str
    query: str
    answer: str
    session_id: str
    success: bool = True
    error: Optional[str] = None
    
    # Optional enhanced fields with proper defaults
    intent: str = ""
    raw_response: Dict[str, Any] = {}
    processing_time: float = 0.0
    debug_info: str = ""
    
    # Remove problematic graph_data field or ensure it has proper default
    # graph_data: Dict[str, Any] = {}  # Commented out - was causing the error

class HealthResponse(BaseModel):
    status: str
    components: Dict[str, Any] = {}
    agent_ready: bool = False

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

def safe_get(dictionary: Dict, key: str, default: Any = ""):
    """Safely get value from dictionary with default"""
    try:
        return dictionary.get(key, default) if isinstance(dictionary, dict) else default
    except:
        return default

# ============================================
# STARTUP/SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize the agent when the app starts"""
    global agent
    
    print("🚀 Starting Fixed Neo4j LangGraph FastAPI App...")
    print("=" * 50)
    
    if not AGENT_IMPORTED:
        print("❌ No agent module imported - limited functionality")
        return
    
    # Check tool server first
    print("🔍 Checking tool server connection...")
    tool_server_ok, tool_health = check_tool_server()
    
    if tool_server_ok:
        print("✅ Tool server is running and accessible")
        if tool_health:
            neo4j_status = safe_get(safe_get(tool_health, "neo4j", {}), "status", "unknown")
            print(f"   Neo4j Status: {neo4j_status}")
    else:
        print("❌ Cannot connect to tool server!")
        print(f"❌ Please make sure the tool server is running on port {TOOL_SERVER_PORT}")
        print("❌ App will start but agent functionality will be limited")
    
    # Build LangGraph agent
    try:
        print("🔨 Building LangGraph agent...")
        agent = build_agent()
        print("✅ LangGraph agent built successfully")
        
        print(f"🌐 Fixed FastAPI app ready on port {APP_PORT}")
        print("=" * 50)
        print("🛠️ Fixes applied:")
        print("   • Fixed Pydantic validation error")
        print("   • Removed problematic graph_data field")
        print("   • Added safe dictionary access")
        print("   • Enhanced error handling")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Failed to build agent: {e}")
        agent = None

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    print("🛑 Shutting down Fixed FastAPI App...")
    print("✅ Shutdown complete")

# ============================================
# FIXED API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Fixed Neo4j LangGraph Agent API",
        "version": "2.0.1",
        "description": "FastAPI server with fixed Pydantic validation errors",
        "fixes": [
            "Resolved graph_data validation error",
            "Fixed ChatResponse model",
            "Added safe dictionary access",
            "Enhanced error handling"
        ],
        "endpoints": {
            "chat": "/chat - Chat with the agent",
            "health": "/health - System health check"
        },
        "status": "ready" if agent else "limited" if not AGENT_IMPORTED else "initializing"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        tool_server_ok, tool_health = check_tool_server()
        
        components = {
            "agent": {
                "status": "ready" if agent else "not_initialized",
                "imported": AGENT_IMPORTED
            },
            "tool_server": {
                "status": "healthy" if tool_server_ok else "disconnected",
                "port": TOOL_SERVER_PORT,
                "health_data": tool_health if tool_health else {}
            }
        }
        
        overall_status = "healthy" if agent and tool_server_ok else "degraded"
        
        return HealthResponse(
            status=overall_status,
            components=components,
            agent_ready=agent is not None and AGENT_IMPORTED
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="error",
            components={"error": str(e)},
            agent_ready=False
        )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Fixed chat endpoint with proper error handling"""
    
    # Check if agent is available
    if not AGENT_IMPORTED:
        return ChatResponse(
            trace="Agent module not imported",
            tool="",
            query="",
            answer="❌ Agent not available. Please check that the agent module is properly installed.",
            session_id=request.session_id or str(uuid.uuid4()),
            success=False,
            error="Agent module not imported"
        )
    
    if agent is None:
        return ChatResponse(
            trace="Agent not initialized",
            tool="",
            query="",
            answer="❌ Agent not initialized. Please check server logs for initialization errors.",
            session_id=request.session_id or str(uuid.uuid4()),
            success=False,
            error="Agent not initialized"
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
        
        try:
            result = await agent.ainvoke(state)
            processing_time = time.time() - start_time
        except Exception as agent_error:
            logger.error(f"Agent execution failed: {agent_error}")
            processing_time = time.time() - start_time
            
            return ChatResponse(
                trace=f"Agent execution error: {str(agent_error)}",
                tool="",
                query="",
                answer=f"❌ Agent execution failed: {str(agent_error)}",
                session_id=session_id,
                success=False,
                error=str(agent_error),
                processing_time=processing_time
            )
        
        logger.info(f"Agent completed in {processing_time:.2f}s - Tool: {safe_get(result, 'tool')}")
        
        # Safely extract all fields with defaults
        return ChatResponse(
            # Core fields with safe defaults
            trace=safe_get(result, "trace", ""),
            tool=safe_get(result, "tool", ""),
            query=safe_get(result, "query", ""),
            answer=safe_get(result, "formatted_answer") or safe_get(result, "answer", "No answer generated"),
            session_id=session_id,
            success=True,
            
            # Optional fields with safe defaults
            intent=safe_get(result, "intent", ""),
            raw_response=safe_get(result, "raw_response", {}),
            processing_time=processing_time,
            debug_info=safe_get(result, "debug_info", "")
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        
        # Return a safe error response
        return ChatResponse(
            trace=f"Endpoint error: {str(e)}",
            tool="",
            query="",
            answer=f"⚠️ Error processing request: {str(e)}",
            session_id=request.session_id or str(uuid.uuid4()),
            success=False,
            error=str(e),
            processing_time=0.0
        )

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check current state"""
    return {
        "agent_imported": AGENT_IMPORTED,
        "agent_ready": agent is not None,
        "app_port": APP_PORT,
        "tool_server_port": TOOL_SERVER_PORT,
        "available_modules": [
            name for name in [
                "robust_langgraph_agent",
                "enhanced_langgraph_agent", 
                "fixed_langgraph_agent"
            ] if name in sys.modules
        ]
    }

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """Main function to run the fixed FastAPI application"""
    print("=" * 60)
    print("🔧 FIXED NEO4J LANGGRAPH FASTAPI APP")
    print("=" * 60)
    print("🏗️  Architecture: FastAPI + LangGraph + Tool Server")
    print("🔧 Configuration:")
    print(f"   📍 App Port: {APP_PORT}")
    print(f"   🛠️ Tool Server Port: {TOOL_SERVER_PORT}")
    print(f"   🤖 Agent Available: {AGENT_IMPORTED}")
    print("=" * 60)
    print("✅ Fixes Applied:")
    print("   • Resolved Pydantic validation error")
    print("   • Fixed ChatResponse model structure")
    print("   • Added safe dictionary access")
    print("   • Enhanced error handling and logging")
    print("   • Removed problematic graph_data field")
    print("=" * 60)
    
    if not AGENT_IMPORTED:
        print("⚠️  WARNING: No agent module found!")
        print("⚠️  Available agent files:")
        print("   • robust_langgraph_agent.py (recommended)")
        print("   • enhanced_langgraph_agent.py")
        print("   • fixed_langgraph_agent.py")
        print("⚠️  App will start but with limited functionality")
        print("=" * 60)
    
    print("🚀 Starting fixed FastAPI application...")
    
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
