from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import logging
import uvicorn
from datetime import datetime

# Import the enhanced agent
try:
    from enhanced_agent_with_explanations import build_agent, AgentState
except ImportError:
    # Fallback to original if enhanced not available
    from langgraph_agent import build_agent, AgentState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("neo4j_agent_app")

# Create FastAPI app
app = FastAPI(
    title="Neo4j Graph Explorer API - Enhanced",
    description="AI-powered Neo4j graph database agent with explanations and auto-refresh",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
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
    """Initialize the enhanced LangGraph agent when the server starts"""
    global agent
    try:
        logger.info("🚀 Starting Enhanced Neo4j Graph Explorer Agent server...")
        logger.info("✨ Features: Explanations, Auto-refresh, Better UX")
        agent = build_agent()
        logger.info("✅ Enhanced LangGraph agent initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {e}")
        raise e

# Enhanced request/response models
class ChatRequest(BaseModel):
    question: str
    session_id: str = None
    node_limit: int = 1000  # Reduced default for better performance

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
    explanation: Optional[str] = None  # New field for explanations

# Health check endpoint
@app.get("/")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy",
        "service": "Neo4j Graph Explorer API - Enhanced",
        "version": "2.1.0",
        "features": [
            "explanations", 
            "auto_refresh", 
            "enhanced_ui", 
            "improved_error_handling",
            "conversation_history",
            "quick_actions"
        ],
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
            "max_node_limit": 1000,
            "default_node_limit": 100,
            "interface_type": "enhanced_split_screen",
            "features_enabled": [
                "explanations",
                "auto_refresh", 
                "conversation_history",
                "quick_actions",
                "graph_statistics"
            ]
        }
    }
    
    # Check agent status
    health_status["services"]["langgraph_agent"] = {
        "status": "up" if agent is not None else "down",
        "ready": agent is not None,
        "type": "enhanced_agent",
        "features": [
            "explanations", 
            "auto_refresh",
            "keyword_fallback",
            "enhanced_parsing"
        ]
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
    
    # Check Neo4j connectivity
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
    
    # Get graph statistics if available
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
    Enhanced chat endpoint with explanations and auto-refresh
    
    Processes questions and returns responses with detailed explanations
    and automatically refreshed graph visualization data.
    """
    if agent is None:
        logger.error("Agent not initialized")
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    # Generate session_id if not provided
    session_id = request.session_id or str(uuid.uuid4())
    node_limit = min(request.node_limit, 1000)  # Cap at 1k for performance
    
    logger.info(f"🤔 Processing enhanced chat request - Session: {session_id[:8]}...")
    logger.info(f"💬 Question: {request.question[:100]}... (Node limit: {node_limit})")
    
    start_time = datetime.now()
    
    try:
        # Create enhanced agent state
        state = AgentState(
            question=request.question,
            session_id=session_id,
            node_limit=node_limit
        )
        
        # Run the enhanced agent
        logger.info(f"🔄 Running enhanced LangGraph agent...")
        result = await agent.ainvoke(state)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        logger.info(f"✅ Enhanced agent completed - Tool: {result.get('tool')}")
        logger.info(f"📈 Execution time: {execution_time:.2f}ms")
        
        # Check if we have graph data
        has_graph_data = result.get('graph_data') and result.get('graph_data', {}).get('nodes')
        if has_graph_data:
            node_count = len(result['graph_data']['nodes'])
            rel_count = len(result['graph_data'].get('relationships', []))
            logger.info(f"🕸️ Graph data: {node_count} nodes, {rel_count} relationships")
        
        # Extract explanation from answer if it contains formatting
        answer = result.get("answer", "")
        explanation = None
        
        # Try to extract explanation section
        if "## " in answer:
            parts = answer.split("---")
            if len(parts) >= 2:
                explanation = parts[1].strip() if len(parts) > 1 else None
        
        # Prepare enhanced response
        response = ChatResponse(
            trace=result.get("trace", ""),
            tool=result.get("tool", ""),
            query=result.get("query", ""),
            answer=answer,
            graph_data=result.get("graph_data") if has_graph_data else None,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            node_limit=node_limit,
            execution_time_ms=execution_time,
            success=True,
            explanation=explanation
        )
        
        return response
        
    except Exception as e:
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        logger.error(f"❌ Enhanced chat request failed: {str(e)}")
        
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
    Direct agent invocation endpoint with enhanced features
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Ensure node_limit is set
        if 'node_limit' not in request:
            request['node_limit'] = 100
            
        # Create AgentState from request
        state = AgentState(**request)
        result = await agent.ainvoke(state)
        return result
    except Exception as e:
        logger.error(f"Agent invocation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/status")
async def agent_status():
    """Get the current status of the enhanced LangGraph agent"""
    return {
        "agent_initialized": agent is not None,
        "agent_type": "Enhanced LangGraph Neo4j Graph Explorer Agent",
        "interface_type": "enhanced_split_screen",
        "max_node_limit": 1000,
        "default_node_limit": 100,
        "features": [
            "detailed_explanations",
            "auto_refresh_graphs",
            "conversation_history",
            "quick_actions",
            "enhanced_error_handling",
            "keyword_fallback",
            "improved_parsing",
            "graph_statistics"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/graph/sample/{node_limit}")
async def get_sample_graph(node_limit: int = 100):
    """Get a sample graph with specified node limit"""
    try:
        import requests
        capped_limit = min(node_limit, 500)  # Reduced max for better performance
        
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
    """Get comprehensive graph statistics for the enhanced interface"""
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

@app.get("/conversation/export/{session_id}")
async def export_conversation(session_id: str):
    """Export conversation history for a session"""
    # This would typically connect to a database to retrieve conversation history
    # For now, return a placeholder
    return {
        "session_id": session_id,
        "conversations": [],
        "exported_at": datetime.now().isoformat(),
        "note": "Conversation export feature - implement with persistent storage"
    }

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
            "service": "Enhanced Neo4j Graph Explorer API",
            "request_path": str(request.url.path),
            "suggestion": "Check the logs for more details or try a simpler query"
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
            "service": "Enhanced Neo4j Graph Explorer API",
            "request_path": str(request.url.path),
            "suggestion": "Please check if all services are running and try again"
        }
    )

# Development server configuration
if __name__ == "__main__":
    logger.info("🚀 Starting Enhanced Neo4j Graph Explorer API in development mode...")
    logger.info("✨ Features: Explanations, Auto-refresh, Enhanced UX, Conversation History")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8081,
        reload=True,
        log_level="info",
        reload_includes=["*.py"],
        reload_excludes=["test_*", "__pycache__"]
    )
