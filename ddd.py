from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langgraph_agent import build_agent, AgentState
import uuid
import logging
import asyncio
from typing import Optional

# Try to import config, fallback to defaults
try:
    from config import SERVER_CONFIG, DEBUG_CONFIG, TIMEOUT_CONFIG
    APP_PORT = SERVER_CONFIG["app_port"]
    ENABLE_DEBUG = DEBUG_CONFIG["enable_debug_logging"]
    CORTEX_TIMEOUT = TIMEOUT_CONFIG["cortex_timeout"]
except ImportError:
    APP_PORT = 8081
    ENABLE_DEBUG = True
    CORTEX_TIMEOUT = 30

# Set up logging
if ENABLE_DEBUG:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("neo4j_langgraph_app")

# Initialize FastAPI app
app = FastAPI(
    title="Neo4j LangGraph MCP+LLM Agent",
    description="AI Agent for Neo4j database queries using LangGraph and Snowflake Cortex",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent at startup
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
        logger.info("🚀 Starting Neo4j LangGraph MCP Agent...")
        logger.info("Building LangGraph agent...")
        agent = build_agent()
        logger.info("✅ Agent built successfully")
        logger.info(f"🌐 App will be available on port {APP_PORT}")
    except Exception as e:
        logger.error(f"❌ Failed to build agent: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up when the app shuts down"""
    logger.info("🛑 Shutting down Neo4j LangGraph MCP Agent...")
    logger.info("👋 Goodbye!")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint that processes user questions through the LangGraph agent
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info(f"📝 Processing question: {request.question}")
        logger.info(f"🆔 Session ID: {session_id}")
        
        # Create agent state
        state = AgentState(
            question=request.question,
            session_id=session_id
        )
        
        # Run the agent with timeout
        try:
            result = await asyncio.wait_for(
                agent.ainvoke(state),
                timeout=CORTEX_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"⏰ Agent execution timed out after {CORTEX_TIMEOUT}s")
            return ChatResponse(
                trace="Agent execution timed out",
                tool="",
                query="",
                answer="⚠️ The request timed out. Please try again or rephrase your question.",
                session_id=session_id,
                success=False,
                error="Timeout"
            )
        
        logger.info(f"✅ Agent completed successfully")
        logger.info(f"🔧 Tool used: {result.get('tool', 'none')}")
        logger.info(f"📊 Query: {result.get('query', 'none')}")
        
        return ChatResponse(
            trace=result.get("trace", ""),
            tool=result.get("tool", ""),
            query=result.get("query", ""),
            answer=result.get("answer", "No answer generated"),
            session_id=session_id,
            success=True
        )
        
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {e}")
        return ChatResponse(
            trace=f"Error occurred: {str(e)}",
            tool="",
            query="",
            answer=f"⚠️ An error occurred while processing your request: {str(e)}",
            session_id=request.session_id or str(uuid.uuid4()),
            success=False,
            error=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if agent is initialized
        if agent is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "Agent not initialized",
                    "services": {
                        "agent": "not_initialized",
                        "mcp_server": "unknown"
                    }
                }
            )
        
        # Test MCP server connection
        import requests
        try:
            from config import SERVER_CONFIG
            mcp_port = SERVER_CONFIG["mcp_port"]
        except ImportError:
            mcp_port = 8000
            
        try:
            mcp_response = requests.get(f"http://localhost:{mcp_port}/health", timeout=5)
            mcp_status = "healthy" if mcp_response.status_code == 200 else "unhealthy"
        except Exception as e:
            mcp_status = f"error: {str(e)}"
        
        return {
            "status": "healthy",
            "message": "All systems operational",
            "services": {
                "agent": "initialized",
                "mcp_server": mcp_status
            },
            "port": APP_PORT
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "message": f"Health check failed: {str(e)}",
                "services": {
                    "agent": "error",
                    "mcp_server": "unknown"
                }
            }
        )

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "Neo4j LangGraph MCP+LLM Agent",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs"
        },
        "description": "AI Agent for Neo4j database queries using LangGraph and Snowflake Cortex"
    }

@app.get("/status")
async def get_status():
    """Get detailed status information"""
    try:
        # Get configuration info
        try:
            from config import NEO4J_CONFIG, CORTEX_CONFIG, SERVER_CONFIG
            config_status = "loaded"
            neo4j_uri = NEO4J_CONFIG["uri"]
            cortex_model = CORTEX_CONFIG["model"]
        except ImportError:
            config_status = "using_defaults"
            neo4j_uri = "neo4j://localhost:7687"
            cortex_model = "llama3.1-70b"
        
        return {
            "agent_status": "initialized" if agent else "not_initialized",
            "config_status": config_status,
            "neo4j_uri": neo4j_uri,
            "cortex_model": cortex_model,
            "debug_enabled": ENABLE_DEBUG,
            "timeout": CORTEX_TIMEOUT,
            "port": APP_PORT
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Status check failed: {str(e)}"}
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if ENABLE_DEBUG else "An error occurred"
        }
    )

# Include router if it exists
try:
    from router import router
    app.include_router(router)
    logger.info("✅ Additional routes loaded from router.py")
except ImportError:
    logger.info("ℹ️  No additional router found")

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Neo4j LangGraph MCP Agent directly...")
    logger.info(f"🌐 Server will run on http://localhost:{APP_PORT}")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=APP_PORT,
        reload=True,
        log_level="debug" if ENABLE_DEBUG else "info"
    )
