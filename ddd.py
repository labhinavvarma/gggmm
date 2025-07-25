from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from langgraph_agent import build_schema_aware_agent, SchemaAwareAgentState, schema_manager
import uuid
import logging
import uvicorn
from datetime import datetime
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("schema_aware_neo4j_app")

# Create FastAPI app with enhanced metadata
app = FastAPI(
    title="Schema-Aware Neo4j Graph Explorer API",
    description="AI-powered Neo4j graph database agent with complete schema awareness and advanced visualization",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize schema-aware agent at startup
agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize the Schema-Aware LangGraph agent when the server starts"""
    global agent
    try:
        logger.info("🚀 Starting Schema-Aware Neo4j Graph Explorer Agent...")
        logger.info("📊 Loading complete database schema for intelligent query generation...")
        
        # Build schema-aware agent (this will auto-load schema)
        agent = build_schema_aware_agent()
        
        # Get schema info for logging
        schema_info = schema_manager.get_cached_schema()
        if schema_info:
            enhanced_info = schema_info.get("enhanced_info", {})
            node_types = len(enhanced_info.get("node_labels", [[]])[0]) if enhanced_info.get("node_labels") else 0
            rel_types = len(enhanced_info.get("relationship_types", [[]])[0]) if enhanced_info.get("relationship_types") else 0
            logger.info(f"🧠 Schema loaded: {node_types} node types, {rel_types} relationship types")
        
        logger.info("✅ Schema-Aware LangGraph agent initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize schema-aware agent: {e}")
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
    schema_aware: bool = True  # New field to indicate schema awareness

# Health check endpoint
@app.get("/")
async def health_check():
    """Enhanced health check endpoint with schema awareness"""
    schema_status = "unknown"
    schema_info = {}
    
    try:
        cached_schema = schema_manager.get_cached_schema()
        if cached_schema:
            schema_status = "loaded"
            enhanced_info = cached_schema.get("enhanced_info", {})
            schema_info = {
                "node_types": len(enhanced_info.get("node_labels", [[]])[0]) if enhanced_info.get("node_labels") else 0,
                "relationship_types": len(enhanced_info.get("relationship_types", [[]])[0]) if enhanced_info.get("relationship_types") else 0,
                "last_updated": cached_schema.get("last_updated", "unknown")
            }
        else:
            schema_status = "not_loaded"
    except Exception:
        schema_status = "error"
    
    return {
        "status": "healthy",
        "service": "Schema-Aware Neo4j Graph Explorer API",
        "version": "3.0.0",
        "features": [
            "schema_awareness", 
            "intelligent_query_generation", 
            "multi_tier_connections",
            "query_validation",
            "5000_node_support", 
            "interactive_visualization"
        ],
        "timestamp": datetime.now().isoformat(),
        "agent_ready": agent is not None,
        "schema_status": schema_status,
        "schema_info": schema_info
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
            "interface_type": "schema_aware_split_screen",
            "schema_aware": True
        }
    }
    
    # Check schema-aware agent status
    health_status["services"]["schema_aware_agent"] = {
        "status": "up" if agent is not None else "down",
        "ready": agent is not None,
        "features": [
            "schema_awareness",
            "intelligent_query_generation", 
            "query_validation",
            "multi_tier_support",
            "visualization_optimization"
        ]
    }
    
    # Check schema manager status
    try:
        cached_schema = schema_manager.get_cached_schema()
        if cached_schema:
            enhanced_info = cached_schema.get("enhanced_info", {})
            health_status["services"]["schema_manager"] = {
                "status": "up",
                "node_types": len(enhanced_info.get("node_labels", [[]])[0]) if enhanced_info.get("node_labels") else 0,
                "relationship_types": len(enhanced_info.get("relationship_types", [[]])[0]) if enhanced_info.get("relationship_types") else 0,
                "last_updated": cached_schema.get("last_updated"),
                "features": ["schema_caching", "auto_refresh", "validation"]
            }
        else:
            health_status["services"]["schema_manager"] = {
                "status": "down",
                "error": "Schema not loaded"
            }
    except Exception as e:
        health_status["services"]["schema_manager"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check MCP server connectivity
    try:
        mcp_response = requests.get("http://localhost:8000/", timeout=5)
        health_status["services"]["mcp_server"] = {
            "status": "up" if mcp_response.status_code == 200 else "down",
            "url": "http://localhost:8000",
            "features": ["graph_extraction", "node_limiting", "optimization", "schema_queries"]
        }
    except Exception as e:
        health_status["services"]["mcp_server"] = {
            "status": "down",
            "error": str(e),
            "url": "http://localhost:8000"
        }
    
    # Check Neo4j connectivity via MCP server
    try:
        neo4j_response = requests.post(
            "http://localhost:8000/get_neo4j_schema", 
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        health_status["services"]["neo4j"] = {
            "status": "up" if neo4j_response.status_code == 200 else "down",
            "features": ["graph_database", "cypher_queries", "apoc_procedures", "schema_introspection"]
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
    Enhanced schema-aware chat endpoint
    
    Processes questions using complete database schema knowledge for
    intelligent query generation and improved accuracy.
    """
    if agent is None:
        logger.error("Schema-aware agent not initialized")
        raise HTTPException(status_code=500, detail="Schema-aware agent not initialized")
    
    # Generate session_id if not provided
    session_id = request.session_id or str(uuid.uuid4())
    node_limit = min(request.node_limit, 10000)  # Cap at 10k for performance
    
    logger.info(f"🧠 Processing schema-aware chat request - Session: {session_id[:8]}...")
    logger.info(f"📊 Question: {request.question[:100]} (Node limit: {node_limit})")
    
    start_time = datetime.now()
    
    try:
        # Create schema-aware agent state
        state = SchemaAwareAgentState(
            question=request.question,
            session_id=session_id,
            node_limit=node_limit
        )
        
        # Run the schema-aware agent
        logger.info(f"🔄 Running Schema-Aware LangGraph agent...")
        result = await agent.ainvoke(state)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        logger.info(f"✅ Schema-aware agent completed - Tool: {result.get('tool')}")
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
            success=True,
            schema_aware=True
        )
        
        return response
        
    except Exception as e:
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        logger.error(f"❌ Schema-aware chat request failed: {str(e)}")
        
        # Return enhanced error response
        error_response = ChatResponse(
            trace=f"Schema-aware processing error: {str(e)}",
            tool="",
            query="",
            answer=f"❌ I encountered an error processing your request with schema awareness: {str(e)}",
            graph_data=None,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            node_limit=node_limit,
            execution_time_ms=execution_time,
            success=False,
            error=str(e),
            schema_aware=True
        )
        
        return error_response

# ============================================================================
# SCHEMA MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/schema/summary")
async def get_schema_summary():
    """Get the current database schema summary"""
    try:
        summary = schema_manager.schema_summary
        cached_schema = schema_manager.get_cached_schema()
        
        if not summary or not cached_schema:
            logger.info("Schema not cached, fetching fresh schema...")
            schema_manager.fetch_database_schema()
            summary = schema_manager.schema_summary
            cached_schema = schema_manager.get_cached_schema()
        
        schema_info = {}
        if cached_schema:
            enhanced_info = cached_schema.get("enhanced_info", {})
            schema_info = {
                "node_types": len(enhanced_info.get("node_labels", [[]])[0]) if enhanced_info.get("node_labels") else 0,
                "relationship_types": len(enhanced_info.get("relationship_types", [[]])[0]) if enhanced_info.get("relationship_types") else 0,
                "property_keys": len(enhanced_info.get("property_keys", [[]])[0]) if enhanced_info.get("property_keys") else 0
            }
        
        return {
            "summary": summary or "Schema summary not available",
            "schema_info": schema_info,
            "last_updated": schema_manager.last_schema_update.isoformat() if schema_manager.last_schema_update else None,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting schema summary: {e}")
        return {"error": str(e), "status": "error"}

@app.post("/schema/refresh")
async def refresh_schema():
    """Manually refresh the database schema cache"""
    try:
        logger.info("🔄 Manually refreshing schema cache...")
        schema_info = schema_manager.fetch_database_schema()
        
        if schema_info:
            enhanced_info = schema_info.get("enhanced_info", {})
            node_types = len(enhanced_info.get("node_labels", [[]])[0]) if enhanced_info.get("node_labels") else 0
            rel_types = len(enhanced_info.get("relationship_types", [[]])[0]) if enhanced_info.get("relationship_types") else 0
            prop_keys = len(enhanced_info.get("property_keys", [[]])[0]) if enhanced_info.get("property_keys") else 0
            
            logger.info(f"✅ Schema refreshed: {node_types} node types, {rel_types} relationship types")
            
            return {
                "message": "Schema refreshed successfully",
                "node_types": node_types,
                "relationship_types": rel_types,
                "property_keys": prop_keys,
                "last_updated": schema_manager.last_schema_update.isoformat() if schema_manager.last_schema_update else None,
                "status": "success"
            }
        else:
            return {"error": "Failed to refresh schema", "status": "error"}
            
    except Exception as e:
        logger.error(f"Error refreshing schema: {e}")
        return {"error": str(e), "status": "error"}

@app.get("/schema/validation")
async def validate_query_endpoint(query: str):
    """Validate a Cypher query against the current schema"""
    try:
        schema_info = schema_manager.get_cached_schema()
        if not schema_info:
            schema_info = schema_manager.fetch_database_schema()
        
        # Import validation function
        from langgraph_agent import validate_query_against_schema
        is_valid, message = validate_query_against_schema(query, schema_info)
        
        return {
            "query": query,
            "is_valid": is_valid,
            "validation_message": message,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error validating query: {e}")
        return {"error": str(e), "status": "error"}

@app.get("/schema/details")
async def get_detailed_schema():
    """Get detailed schema information including counts and examples"""
    try:
        schema_info = schema_manager.get_cached_schema()
        if not schema_info:
            schema_info = schema_manager.fetch_database_schema()
        
        if not schema_info:
            return {"error": "Schema not available", "status": "error"}
        
        enhanced_info = schema_info.get("enhanced_info", {})
        
        return {
            "node_labels": enhanced_info.get("node_labels", []),
            "relationship_types": enhanced_info.get("relationship_types", []),
            "property_keys": enhanced_info.get("property_keys", []),
            "node_counts": enhanced_info.get("node_counts", []),
            "relationship_counts": enhanced_info.get("relationship_counts", []),
            "raw_schema": schema_info.get("raw_schema", {}),
            "last_updated": schema_info.get("last_updated"),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting detailed schema: {e}")
        return {"error": str(e), "status": "error"}

# ============================================================================
# ENHANCED AGENT ENDPOINTS
# ============================================================================

@app.post("/agent/invoke")
async def invoke_agent(request: dict):
    """
    Direct schema-aware agent invocation endpoint
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="Schema-aware agent not initialized")
    
    try:
        # Ensure required fields
        if 'node_limit' not in request:
            request['node_limit'] = 5000
        if 'question' not in request:
            raise HTTPException(status_code=400, detail="Question field required")
        if 'session_id' not in request:
            request['session_id'] = str(uuid.uuid4())
            
        # Create SchemaAwareAgentState from request
        state = SchemaAwareAgentState(**request)
        result = await agent.ainvoke(state)
        return result
    except Exception as e:
        logger.error(f"Schema-aware agent invocation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/status")
async def agent_status():
    """Get the current status of the Schema-Aware LangGraph agent"""
    schema_status = {}
    try:
        cached_schema = schema_manager.get_cached_schema()
        if cached_schema:
            enhanced_info = cached_schema.get("enhanced_info", {})
            schema_status = {
                "loaded": True,
                "node_types": len(enhanced_info.get("node_labels", [[]])[0]) if enhanced_info.get("node_labels") else 0,
                "relationship_types": len(enhanced_info.get("relationship_types", [[]])[0]) if enhanced_info.get("relationship_types") else 0,
                "last_updated": cached_schema.get("last_updated")
            }
        else:
            schema_status = {"loaded": False}
    except Exception:
        schema_status = {"loaded": False, "error": "Schema check failed"}
    
    return {
        "agent_initialized": agent is not None,
        "agent_type": "Schema-Aware LangGraph Neo4j Graph Explorer Agent",
        "interface_type": "schema_aware_split_screen",
        "max_node_limit": 10000,
        "default_node_limit": 5000,
        "schema_status": schema_status,
        "features": [
            "schema_awareness",
            "intelligent_query_generation",
            "query_validation",
            "multi_tier_connections",
            "interactive_visualization",
            "node_limiting",
            "query_optimization", 
            "split_screen_layout",
            "real_time_updates"
        ],
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# LEGACY ENDPOINTS (for backward compatibility)
# ============================================================================

@app.get("/graph/sample/{node_limit}")
async def get_sample_graph(node_limit: int = 5000):
    """Get a sample graph with specified node limit"""
    try:
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
    """Get comprehensive graph statistics"""
    try:
        response = requests.get("http://localhost:8000/graph_stats", timeout=15)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except Exception as e:
        logger.error(f"Error getting graph statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ERROR HANDLERS
# ============================================================================

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
            "service": "Schema-Aware Neo4j Graph Explorer API",
            "request_path": str(request.url.path),
            "schema_aware": True
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
            "service": "Schema-Aware Neo4j Graph Explorer API",
            "request_path": str(request.url.path),
            "schema_aware": True
        }
    )

# Development server configuration
if __name__ == "__main__":
    logger.info("🚀 Starting Schema-Aware Neo4j Graph Explorer API...")
    logger.info("🧠 Enhanced with complete database schema intelligence")
    
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
