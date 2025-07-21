from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langgraph_agent import build_agent, AgentState
import uuid
import logging
import uvicorn
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("neo4j_agent_app")

# Create FastAPI app
app = FastAPI(
    title="Neo4j LangGraph MCP Agent API",
    description="AI-powered Neo4j graph database agent with visualization support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for cross-origin requests (useful for Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent at startup
agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize the LangGraph agent when the server starts"""
    global agent
    try:
        logger.info("🚀 Starting Neo4j LangGraph Agent server...")
        agent = build_agent()
        logger.info("✅ LangGraph agent initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {e}")
        raise e

# Request/Response models
class ChatRequest(BaseModel):
    question: str
    session_id: str = None

class ChatResponse(BaseModel):
    trace: str
    tool: str
    query: str
    answer: str
    graph_data: dict = None
    session_id: str
    timestamp: str
    success: bool = True
    error: str = None

# Health check endpoint
@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Neo4j LangGraph MCP Agent",
        "timestamp": datetime.now().isoformat(),
        "agent_ready": agent is not None
    }

@app.get("/health")
async def detailed_health():
    """Detailed health check with service dependencies"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    # Check agent status
    health_status["services"]["langgraph_agent"] = {
        "status": "up" if agent is not None else "down",
        "ready": agent is not None
    }
    
    # Check MCP server connectivity
    try:
        import requests
        mcp_response = requests.get("http://localhost:8000/", timeout=5)
        health_status["services"]["mcp_server"] = {
            "status": "up" if mcp_response.status_code == 200 else "down",
            "url": "http://localhost:8000"
        }
    except Exception as e:
        health_status["services"]["mcp_server"] = {
            "status": "down",
            "error": str(e),
            "url": "http://localhost:8000"
        }
    
    # Check Neo4j connectivity (via MCP server)
    try:
        import requests
        neo4j_response = requests.post(
            "http://localhost:8000/get_neo4j_schema", 
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        health_status["services"]["neo4j"] = {
            "status": "up" if neo4j_response.status_code == 200 else "down"
        }
    except Exception as e:
        health_status["services"]["neo4j"] = {
            "status": "down",
            "error": str(e)
        }
    
    # Determine overall status
    all_up = all(
        service.get("status") == "up" 
        for service in health_status["services"].values()
    )
    health_status["status"] = "healthy" if all_up else "degraded"
    
    return health_status

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for the Neo4j LangGraph Agent
    
    Accepts a question and optional session_id, returns the agent's response
    with tool information, Cypher query, and optional graph visualization data.
    """
    if agent is None:
        logger.error("Agent not initialized")
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    # Generate session_id if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    logger.info(f"🤔 Processing chat request - Session: {session_id[:8]}... Question: {request.question[:100]}")
    
    try:
        # Create agent state
        state = AgentState(
            question=request.question,
            session_id=session_id
        )
        
        # Run the agent
        logger.info(f"🔄 Running LangGraph agent...")
        result = await agent.ainvoke(state)
        
        logger.info(f"✅ Agent completed - Tool: {result.get('tool')}, Answer length: {len(result.get('answer', ''))}")
        
        # Prepare response
        response = ChatResponse(
            trace=result.get("trace", ""),
            tool=result.get("tool", ""),
            query=result.get("query", ""),
            answer=result.get("answer", ""),
            graph_data=result.get("graph_data"),
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            success=True
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Chat request failed: {str(e)}")
        
        # Return error response
        error_response = ChatResponse(
            trace=f"Error occurred: {str(e)}",
            tool="",
            query="",
            answer=f"❌ Sorry, I encountered an error: {str(e)}",
            graph_data=None,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            success=False,
            error=str(e)
        )
        
        return error_response

@app.post("/agent/invoke")
async def invoke_agent(request: dict):
    """
    Direct agent invocation endpoint (alternative to /chat)
    Accepts raw agent state and returns raw agent response
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Create AgentState from request
        state = AgentState(**request)
        result = await agent.ainvoke(state)
        return result
    except Exception as e:
        logger.error(f"Agent invocation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/status")
async def agent_status():
    """Get the current status of the LangGraph agent"""
    return {
        "agent_initialized": agent is not None,
        "agent_type": "LangGraph Neo4j MCP Agent",
        "timestamp": datetime.now().isoformat()
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# Development server configuration
if __name__ == "__main__":
    logger.info("🚀 Starting Neo4j LangGraph MCP Agent server in development mode...")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8081,
        reload=True,
        log_level="info",
        reload_includes=["*.py"],
        reload_excludes=["test_*", "__pycache__"]
    )

# Production server command:
# uvicorn app:app --host 0.0.0.0 --port 8081 --workers 1
