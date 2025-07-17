from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from complete_langgraph_agent import build_agent, AgentState
import uuid
import logging
from typing import Optional

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("complete_app")

# Initialize FastAPI app
app = FastAPI(
    title="Complete Neo4j LangGraph Agent",
    description="Complete AI Agent with integrated prompts for Neo4j database queries",
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

# Initialize agent
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
    
    # Additional fields for debugging
    intent: str = ""
    execution_steps: list = []

@app.on_event("startup")
async def startup_event():
    """Initialize the complete agent when the app starts"""
    global agent
    try:
        logger.info("🚀 Starting Complete Neo4j LangGraph Agent...")
        logger.info("📋 Building agent with integrated prompts...")
        
        agent = build_agent()
        
        logger.info("✅ Complete agent initialized successfully")
        logger.info("🧠 Agent features:")
        logger.info("   - Intent analysis")
        logger.info("   - Tool selection") 
        logger.info("   - Query generation")
        logger.info("   - Error handling with retries")
        logger.info("   - Result formatting")
        logger.info("   - All prompts integrated")
        
    except Exception as e:
        logger.error(f"❌ Failed to build complete agent: {e}")
        raise

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint using complete LangGraph agent"""
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
        
        # Run the complete agent
        try:
            logger.info("🔄 Running complete LangGraph agent...")
            result = await agent.ainvoke(state)
            logger.info("✅ Complete agent execution finished")
            
            # Extract execution steps for debugging
            execution_steps = []
            if result.get("intent"):
                execution_steps.append(f"Intent: {result.get('intent')}")
            if result.get("tool"):
                execution_steps.append(f"Tool: {result.get('tool')}")
            if result.get("query"):
                execution_steps.append(f"Query: {result.get('query')}")
            if result.get("execution_result"):
                execution_steps.append("Execution: Completed")
            if result.get("formatted_result"):
                execution_steps.append("Formatting: Completed")
            
            logger.info(f"📊 Execution steps: {' → '.join(execution_steps)}")
            
            return ChatResponse(
                trace=result.get("trace", ""),
                tool=result.get("tool", ""),
                query=result.get("query", ""),
                answer=result.get("answer", "No answer generated"),
                session_id=session_id,
                intent=result.get("intent", ""),
                execution_steps=execution_steps,
                success=True
            )
            
        except Exception as e:
            logger.error(f"❌ Complete agent execution failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return ChatResponse(
                trace="Complete agent execution failed",
                tool="",
                query="",
                answer=f"❌ The agent encountered an error: {str(e)}",
                session_id=session_id,
                execution_steps=["Error occurred during execution"],
                success=False,
                error=str(e)
            )
        
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}")
        return ChatResponse(
            trace="",
            tool="",
            query="",
            answer=f"❌ System error: {str(e)}",
            session_id=request.session_id or str(uuid.uuid4()),
            execution_steps=["System error"],
            success=False,
            error=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check agent status
        agent_status = "initialized" if agent else "not_initialized"
        
        # Test MCP server
        import requests
        try:
            mcp_response = requests.get("http://localhost:8000/health", timeout=5)
            mcp_status = "healthy" if mcp_response.status_code == 200 else "unhealthy"
            mcp_details = mcp_response.json() if mcp_response.status_code == 200 else {}
        except Exception as e:
            mcp_status = "error"
            mcp_details = {"error": str(e)}
        
        # Test Cortex API
        from complete_langgraph_agent import call_cortex_llm
        try:
            test_response = call_cortex_llm("Test connection", "health_check")
            cortex_status = "healthy" if "Error:" not in test_response else "error"
        except Exception as e:
            cortex_status = "error"
        
        return {
            "status": "healthy" if agent_status == "initialized" else "unhealthy",
            "services": {
                "complete_agent": agent_status,
                "mcp_server": mcp_status,
                "cortex_llm": cortex_status
            },
            "mcp_details": mcp_details,
            "agent_features": [
                "Intent Analysis",
                "Tool Selection", 
                "Query Generation",
                "Error Handling",
                "Result Formatting",
                "Integrated Prompts"
            ]
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy", 
            "error": str(e)
        }

@app.get("/")
async def root():
    """Root endpoint with complete agent info"""
    return {
        "service": "Complete Neo4j LangGraph Agent",
        "version": "2.0.0",
        "description": "Complete AI agent with integrated prompts for Neo4j database queries",
        "features": [
            "Intent Analysis with specialized prompts",
            "Intelligent tool selection",
            "Dynamic Cypher query generation", 
            "Error handling and retries",
            "Result formatting and presentation",
            "Multi-step reasoning workflow"
        ],
        "endpoints": {
            "chat": "/chat - Main chat interface",
            "health": "/health - System health check",
            "docs": "/docs - API documentation"
        },
        "workflow": [
            "1. Analyze user intent",
            "2. Select appropriate tool",
            "3. Generate Cypher query (if needed)",
            "4. Execute on MCP server", 
            "5. Format results for user",
            "6. Handle errors with retries"
        ]
    }

@app.get("/agent-info")
async def agent_info():
    """Get detailed agent information"""
    return {
        "agent_type": "Complete LangGraph Agent",
        "nodes": [
            "analyze_intent - Determines user's intent from question",
            "select_tool - Chooses appropriate Neo4j tool",
            "generate_query - Creates Cypher query",
            "execute_tool - Runs query on MCP server",
            "format_result - Formats response for user",
            "handle_error - Manages errors and retries"
        ],
        "prompts": [
            "Intent Analysis Prompt - Classifies user intent",
            "Tool Selection Prompt - Maps intent to tools",
            "Cypher Generation Prompt - Creates queries",
            "Result Formatting Prompt - Formats output",
            "Error Analysis Prompt - Handles failures"
        ],
        "routing": "Conditional routing based on state and results",
        "error_handling": "Automatic retries with intelligent error analysis",
        "session_management": "Persistent sessions across requests"
    }

@app.get("/test-workflow")
async def test_workflow():
    """Test the complete workflow with sample questions"""
    if not agent:
        return {"error": "Agent not initialized"}
    
    test_questions = [
        "How many nodes are in the graph?",
        "Show me the database schema", 
        "Create a test Person node",
        "List all node types"
    ]
    
    results = []
    
    for question in test_questions:
        try:
            state = AgentState(
                question=question,
                session_id="test_workflow"
            )
            
            result = await agent.ainvoke(state)
            
            results.append({
                "question": question,
                "intent": result.get("intent", ""),
                "tool": result.get("tool", ""),
                "query": result.get("query", ""),
                "success": bool(result.get("answer", "")),
                "answer_preview": result.get("answer", "")[:100] + "..." if result.get("answer", "") else ""
            })
            
        except Exception as e:
            results.append({
                "question": question,
                "error": str(e),
                "success": False
            })
    
    return {
        "test_summary": f"Tested {len(test_questions)} questions",
        "success_count": sum(1 for r in results if r.get("success", False)),
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Complete Neo4j LangGraph Agent directly...")
    logger.info("🧠 This version includes all integrated prompts and workflows")
    
    uvicorn.run(
        "complete_app:app",
        host="0.0.0.0",
        port=8081,
        reload=True,
        log_level="info"
    )
