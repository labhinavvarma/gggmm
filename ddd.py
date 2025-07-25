from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from langgraph_agent import build_agent, AgentState  # Fixed import
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

# Create FastAPI app with enhanced metadata
app = FastAPI(
    title="Neo4j Graph Explorer API",
    description="AI-powered Neo4j graph database agent with split-screen visualization support",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enhanced CORS middleware for split-screen UI
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
        logger.info("🚀 Starting Neo4j Graph Explorer Agent server...")
        logger.info("🎨 Optimized for split-screen interface with 5000 node support")
        agent = build_agent()
        logger.info("✅ LangGraph agent initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {e}")
        raise e

# Enhanced request/response models
class ChatRequest(BaseModel):
    question: str
    session_id: str = None
    node_limit: int = 5000  # Default node limit for visualization

class ChatResponse(BaseModel):
    trace: str
    tool: str
    query: str
    answer: str
    graph_data: Optional[dict] = None
    session_id: str
    timestamp: str
    node_limit: int
    success: bool = True
    error: Optional[str] = None
    execution_time_ms: float = 0

# Health check endpoint
@app.get("/")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy",
        "service": "Neo4j Graph Explorer API",
        "version": "2.0.0",
        "features": ["split_screen_ui", "5000_node_support", "interactive_visualization"],
        "timestamp": datetime.now().isoformat(),
        "agent_ready": agent is not None
    }

@app.get("/health")
async def detailed_health():
    """Comprehensive health check with all service dependencies"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {},
        "configuration": {
            "max_node_limit": 5000,
            "default_node_limit": 5000,
            "interface_type": "split_screen"
        }
    }
    
    # Check agent status
    health_status["services"]["langgraph_agent"] = {
        "status": "up" if agent is not None else "down",
        "ready": agent is not None,
        "features": ["visualization_optimization", "node_limiting", "split_screen_support"]
    }
    
    # Check MCP server connectivity
    try:
        import requests
        mcp_response = requests.get("http://localhost:8000/", timeout=5)
        health_status["services"]["mcp_server"] = {
            "status": "up" if mcp_response.status_code == 200 else "down",
            "url": "http://localhost:8000",
            "features": ["graph_extraction", "node_limiting", "optimization"]
        }
    except Exception as e:
        health_status["services"]["mcp_server"] = {
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
            "features": ["graph_database", "cypher_queries", "apoc_procedures"]
        }
    except Exception as e:
        health_status["services"]["neo4j"] = {
            "status": "down",
            "error": str(e)
        }
    
    # Check graph statistics
    try:
        import requests
        stats_response = requests.get("http://localhost:8000/graph_stats", timeout=10)
        if stats_response.status_code == 200:
            stats_data = stats_response.json()
            health_status["graph_statistics"] = stats_data.get("stats", {})
    except Exception:
        health_status["graph_statistics"] = {"error": "Could not retrieve graph statistics"}
    
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
    Enhanced chat endpoint optimized for split-screen interface
    
    Processes questions and returns responses with graph visualization data
    optimized for the split-screen UI layout.
    """
    if agent is None:
        logger.error("Agent not initialized")
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    # Generate session_id if not provided
    session_id = request.session_id or str(uuid.uuid4())
    node_limit = min(request.node_limit, 10000)  # Cap at 10k for performance
    
    logger.info(f"🤔 Processing chat request - Session: {session_id[:8]}...")
    logger.info(f"📊 Question: {request.question[:100]} (Node limit: {node_limit})")
    
    start_time = datetime.now()
    
    try:
        # Create agent state
        state = AgentState(
            question=request.question,
            session_id=session_id,
            node_limit=node_limit
        )
        
        # Run the agent
        logger.info(f"🔄 Running LangGraph agent with visualization optimization...")
        result = await agent.ainvoke(state)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        logger.info(f"✅ Agent completed - Tool: {result.get('tool')}")
        logger.info(f"📈 Execution time: {execution_time:.2f}ms")
        
        # Check if we have graph data
        has_graph_data = result.get('graph_data') and result.get('graph_data', {}).get('nodes')
        if has_graph_data:
            node_count = len(result['graph_data']['nodes'])
            rel_count = len(result['graph_data'].get('relationships', []))
            logger.info(f"🕸️ Graph data: {node_count} nodes, {rel_count} relationships")
        
        # Prepare enhanced response
        response = ChatResponse(
            trace=result.get("trace", ""),
            tool=result.get("tool", ""),
            query=result.get("query", ""),
            answer=result.get("answer", ""),
            graph_data=result.get("graph_data") if result.get("graph_data") else None,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            node_limit=node_limit,
            execution_time_ms=execution_time,
            success=True
        )
        
        return response
        
    except Exception as e:
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        logger.error(f"❌ Chat request failed: {str(e)}")
        
        # Return enhanced error response
        error_response = ChatResponse(
            trace=f"Error occurred: {str(e)}",
            tool="",
            query="",
            answer=f"❌ I encountered an error processing your request: {str(e)}",
            graph_data=None,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            node_limit=node_limit,
            execution_time_ms=execution_time,
            success=False,
            error=str(e)
        )
        
        return error_response

@app.post("/agent/invoke")
async def invoke_agent(request: dict):
    """
    Direct agent invocation endpoint with node limit support
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Ensure node_limit is set
        if 'node_limit' not in request:
            request['node_limit'] = 5000
            
        # Create AgentState from request
        state = AgentState(**request)
        result = await agent.ainvoke(state)
        return result
    except Exception as e:
        logger.error(f"Agent invocation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/status")
async def agent_status():
    """Get the current status of the LangGraph agent with configuration info"""
    return {
        "agent_initialized": agent is not None,
        "agent_type": "LangGraph Neo4j Graph Explorer Agent",
        "interface_type": "split_screen",
        "max_node_limit": 10000,
        "default_node_limit": 5000,
        "features": [
            "interactive_visualization",
            "node_limiting",
            "query_optimization", 
            "split_screen_layout",
            "real_time_updates"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/graph/sample/{node_limit}")
async def get_sample_graph(node_limit: int = 5000):
    """Get a sample graph with specified node limit"""
    try:
        import requests
        capped_limit = min(node_limit, 5000)
        
        response = requests.get(
            f"http://localhost:8000/sample_graph?node_limit={capped_limit}",
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except Exception as e:
        logger.error(f"Error getting sample graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/stats")
async def get_graph_statistics():
    """Get comprehensive graph statistics for the split-screen interface"""
    try:
        import requests
        response = requests.get("http://localhost:8000/graph_stats", timeout=15)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except Exception as e:
        logger.error(f"Error getting graph statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/optimize")
async def optimize_query(query: str, node_limit: int = 5000):
    """Optimize a Cypher query for better visualization performance"""
    try:
        import requests
        response = requests.get(
            f"http://localhost:8000/optimize_query/{node_limit}?query={query}",
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except Exception as e:
        logger.error(f"Error optimizing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with enhanced error info"""
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "service": "Neo4j Graph Explorer API",
            "request_path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with detailed logging"""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
            "service": "Neo4j Graph Explorer API",
            "request_path": str(request.url.path)
        }
    )

# Development server configuration
if __name__ == "__main__":
    logger.info("🚀 Starting Neo4j Graph Explorer API in development mode...")
    logger.info("🎨 Optimized for split-screen interface with enhanced visualization")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8020,
        reload=True,
        log_level="info",
        reload_includes=["*.py"],
        reload_excludes=["test_*", "__pycache__"]
    )

# Production server command:
# uvicorn app:app --host 0.0.0.0 --port 8020 --workers 1
