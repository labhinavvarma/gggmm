from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from langgraph_agent import build_agent, AgentState  # Import from unlimited langgraph_agent
import uuid
import logging
import uvicorn
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("unlimited_neo4j_app")

# Create FastAPI app optimized for unlimited display
app = FastAPI(
    title="Unlimited Neo4j Graph Explorer API",
    description="AI-powered Neo4j graph database agent with UNLIMITED visualization - displays everything according to commands",
    version="3.0.0 - UNLIMITED",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for unlimited UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent at startup
agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize the unlimited LangGraph agent when the server starts"""
    global agent
    try:
        logger.info("🚀 Starting UNLIMITED Neo4j Graph Explorer Agent server...")
        logger.info("🕸️ UNLIMITED MODE: Displays ALL data according to commands - NO artificial limits")
        agent = build_agent()
        logger.info("✅ UNLIMITED LangGraph agent initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize unlimited agent: {e}")
        raise e

# Enhanced request/response models for unlimited display
class ChatRequest(BaseModel):
    question: str
    session_id: str = None
    node_limit: int = None  # None means unlimited

class ChatResponse(BaseModel):
    trace: str
    tool: str
    query: str
    answer: str
    graph_data: Optional[dict] = None
    session_id: str
    timestamp: str
    node_limit: Optional[int] = None  # None means unlimited
    success: bool = True
    error: Optional[str] = None
    execution_time_ms: float = 0
    unlimited_mode: bool = True

# Health check endpoint
@app.get("/")
async def health_check():
    """Health check endpoint for unlimited display"""
    return {
        "status": "healthy",
        "service": "UNLIMITED Neo4j Graph Explorer API",
        "version": "3.0.0 - UNLIMITED",
        "features": ["unlimited_display", "no_artificial_limits", "command_based_visualization"],
        "node_limits": "NONE - displays everything according to commands",
        "timestamp": datetime.now().isoformat(),
        "agent_ready": agent is not None,
        "unlimited_mode": True
    }

@app.get("/health")
async def detailed_health():
    """Comprehensive health check for unlimited system"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {},
        "configuration": {
            "node_limits": "NONE",
            "artificial_limits": "DISABLED",
            "interface_type": "unlimited_display",
            "performance_mode": "unlimited_processing"
        }
    }
    
    # Check agent status
    health_status["services"]["unlimited_langgraph_agent"] = {
        "status": "up" if agent is not None else "down",
        "ready": agent is not None,
        "features": ["unlimited_visualization", "no_node_limiting", "command_based_display"]
    }
    
    # Check MCP server connectivity
    try:
        import requests
        mcp_response = requests.get("http://localhost:8000/", timeout=5)
        health_status["services"]["unlimited_mcp_server"] = {
            "status": "up" if mcp_response.status_code == 200 else "down",
            "url": "http://localhost:8000",
            "features": ["unlimited_graph_extraction", "no_node_limiting", "complete_data_processing"]
        }
    except Exception as e:
        health_status["services"]["unlimited_mcp_server"] = {
            "status": "down",
            "error": str(e),
            "url": "http://localhost:8000"
        }
    
    # Check Neo4j connectivity via MCP server
    try:
        import requests
        neo4j_response = requests.post(
            "http://localhost:8000/get_neo4j_schema", 
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        health_status["services"]["neo4j"] = {
            "status": "up" if neo4j_response.status_code == 200 else "down",
            "features": ["graph_database", "unlimited_cypher_queries", "complete_data_access"]
        }
    except Exception as e:
        health_status["services"]["neo4j"] = {
            "status": "down",
            "error": str(e)
        }
    
    # Check unlimited capabilities
    try:
        import requests
        unlimited_response = requests.get("http://localhost:8000/unlimited_display_info", timeout=10)
        if unlimited_response.status_code == 200:
            unlimited_data = unlimited_response.json()
            health_status["unlimited_capabilities"] = unlimited_data
    except Exception:
        health_status["unlimited_capabilities"] = {"error": "Could not retrieve unlimited display info"}
    
    # Determine overall status
    all_up = all(
        service.get("status") == "up" 
        for service in health_status["services"].values()
    )
    health_status["status"] = "healthy" if all_up else "degraded"
    
    return health_status

@app.post("/chat", response_model=ChatResponse)
async def unlimited_chat(request: ChatRequest):
    """
    UNLIMITED chat endpoint - processes questions and displays ALL data according to commands
    
    NO artificial node limits - shows everything the query specifies
    """
    if agent is None:
        logger.error("Unlimited agent not initialized")
        raise HTTPException(status_code=500, detail="Unlimited agent not initialized")
    
    # Generate session_id if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    logger.info(f"🤔 Processing UNLIMITED chat request - Session: {session_id[:8]}...")
    logger.info(f"📊 Question: {request.question}")
    logger.info(f"🚀 UNLIMITED MODE: No artificial limits will be applied")
    
    start_time = datetime.now()
    
    try:
        # Create unlimited agent state
        state = AgentState(
            question=request.question,
            session_id=session_id,
            node_limit=None  # CRITICAL: None for unlimited
        )
        
        # Run the unlimited agent
        logger.info(f"🔄 Running UNLIMITED LangGraph agent...")
        result = await agent.ainvoke(state)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        logger.info(f"✅ Unlimited agent completed - Tool: {result.get('tool')}")
        logger.info(f"📈 Execution time: {execution_time:.2f}ms")
        
        # Check if we have graph data
        has_graph_data = result.get('graph_data') and result.get('graph_data', {}).get('nodes')
        if has_graph_data:
            node_count = len(result['graph_data']['nodes'])
            rel_count = len(result['graph_data'].get('relationships', []))
            logger.info(f"🕸️ UNLIMITED graph data: {node_count} nodes, {rel_count} relationships")
        
        # Prepare unlimited response
        response = ChatResponse(
            trace=result.get("trace", ""),
            tool=result.get("tool", ""),
            query=result.get("query", ""),
            answer=result.get("answer", ""),
            graph_data=result.get("graph_data") if result.get("graph_data") else None,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            node_limit=None,  # Always None for unlimited
            execution_time_ms=execution_time,
            success=True,
            unlimited_mode=True
        )
        
        return response
        
    except Exception as e:
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        logger.error(f"❌ Unlimited chat request failed: {str(e)}")
        
        # Return unlimited error response
        error_response = ChatResponse(
            trace=f"Error occurred in unlimited mode: {str(e)}",
            tool="",
            query="",
            answer=f"❌ I encountered an error processing your unlimited request: {str(e)}",
            graph_data=None,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            node_limit=None,
            execution_time_ms=execution_time,
            success=False,
            error=str(e),
            unlimited_mode=True
        )
        
        return error_response

@app.post("/agent/invoke")
async def invoke_unlimited_agent(request: dict):
    """
    Direct unlimited agent invocation endpoint
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="Unlimited agent not initialized")
    
    try:
        # Ensure unlimited mode
        request['node_limit'] = None  # Force unlimited
            
        # Create AgentState from request
        state = AgentState(**request)
        result = await agent.ainvoke(state)
        return result
    except Exception as e:
        logger.error(f"Unlimited agent invocation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/status")
async def unlimited_agent_status():
    """Get the current status of the unlimited LangGraph agent"""
    return {
        "agent_initialized": agent is not None,
        "agent_type": "UNLIMITED LangGraph Neo4j Graph Explorer Agent",
        "interface_type": "unlimited_display",
        "node_limits": "NONE",
        "artificial_limits": "DISABLED",
        "features": [
            "unlimited_visualization",
            "no_node_limiting",
            "command_based_display", 
            "complete_data_processing",
            "unrestricted_queries"
        ],
        "performance_notes": [
            "Large datasets will be processed completely",
            "Execution time may be longer for complex queries",
            "Browser performance depends on data size"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/graph/unlimited_sample")
async def get_unlimited_sample_graph():
    """Get a complete sample of the graph without any limits"""
    try:
        import requests
        
        response = requests.get(
            "http://localhost:8000/sample_graph",
            timeout=60  # Longer timeout for unlimited data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except Exception as e:
        logger.error(f"Error getting unlimited sample graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/stats")
async def get_unlimited_graph_statistics():
    """Get comprehensive graph statistics without any restrictions"""
    try:
        import requests
        response = requests.get("http://localhost:8000/graph_stats", timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except Exception as e:
        logger.error(f"Error getting unlimited graph statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/unlimited/info")
async def get_unlimited_info():
    """Get detailed information about unlimited display capabilities"""
    try:
        import requests
        response = requests.get("http://localhost:8000/unlimited_display_info", timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except Exception as e:
        logger.error(f"Error getting unlimited info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced error handlers for unlimited mode
@app.exception_handler(HTTPException)
async def unlimited_http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions in unlimited mode"""
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "service": "UNLIMITED Neo4j Graph Explorer API",
            "request_path": str(request.url.path),
            "unlimited_mode": True
        }
    )

@app.exception_handler(Exception)
async def unlimited_general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions in unlimited mode"""
    logger.error(f"Unexpected error in unlimited mode: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error in unlimited mode",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
            "service": "UNLIMITED Neo4j Graph Explorer API",
            "request_path": str(request.url.path),
            "unlimited_mode": True
        }
    )

# Development server configuration
if __name__ == "__main__":
    logger.info("🚀 Starting UNLIMITED Neo4j Graph Explorer API...")
    logger.info("🕸️ UNLIMITED MODE: Displays ALL data according to commands")
    logger.info("⚠️ Performance: Large datasets will be processed completely - may take time")
    
    uvicorn.run(
        "unlimited_app:app",
        host="0.0.0.0",
        port=8020,
        reload=True,
        log_level="info",
        reload_includes=["*.py"],
        reload_excludes=["test_*", "__pycache__"]
    )

# Production server command:
# uvicorn unlimited_app:app --host 0.0.0.0 --port 8020 --workers 1
