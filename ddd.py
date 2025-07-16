from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, List, Optional
import uuid
import json
from datetime import datetime, date
import logging
from contextlib import asynccontextmanager

# Import your health agent
try:
    from health_agent_core import HealthAnalysisAgent, Config
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    import_error = str(e)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
health_agent = None
sessions: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the health agent on startup"""
    global health_agent
    
    if AGENT_AVAILABLE:
        try:
            config = Config()
            health_agent = HealthAnalysisAgent(config)
            logger.info("✅ Health Analysis Agent initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Health Agent: {str(e)}")
            health_agent = None
    else:
        logger.error(f"❌ Health Agent not available: {import_error}")
    
    yield
    logger.info("🔄 Shutting down Health Agent FastAPI server")

# Create FastAPI app
app = FastAPI(
    title="Health Analysis Agent API",
    description="FastAPI wrapper for comprehensive health analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PatientData(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    ssn: str = Field(..., min_length=9, max_length=11)
    date_of_birth: str = Field(..., description="Date in YYYY-MM-DD format")
    gender: str = Field(..., pattern=r'^[MF]$')
    zip_code: str = Field(..., min_length=5, max_length=10)
    
    @field_validator('date_of_birth')
    @classmethod
    def validate_date_of_birth(cls, v):
        try:
            birth_date = datetime.strptime(v, '%Y-%m-%d').date()
            today = date.today()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            
            if age > 150 or age < 0:
                raise ValueError('Invalid age')
            return v
        except ValueError:
            raise ValueError('Invalid date format. Use YYYY-MM-DD')
    
    @field_validator('ssn')
    @classmethod
    def validate_ssn(cls, v):
        ssn_digits = ''.join(filter(str.isdigit, v))
        if len(ssn_digits) != 9:
            raise ValueError('SSN must contain exactly 9 digits')
        return ssn_digits
    
    @field_validator('zip_code')
    @classmethod
    def validate_zip_code(cls, v):
        zip_digits = ''.join(filter(str.isdigit, v))
        if len(zip_digits) < 5:
            raise ValueError('Zip code must contain at least 5 digits')
        return zip_digits

class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1, max_length=1000)
    chat_history: Optional[List[Dict[str, str]]] = Field(default_factory=list)

class SimpleChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1, max_length=1000)

# Helper functions
def check_agent_available():
    """Check if health agent is available"""
    if not AGENT_AVAILABLE:
        raise HTTPException(status_code=503, detail=f"Health Agent not available: {import_error}")
    if not health_agent:
        raise HTTPException(status_code=503, detail="Health Agent not initialized")

def create_session(patient_data: dict, status: str = "running") -> str:
    """Create a new session"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "status": status,
        "created_at": datetime.now().isoformat(),
        "patient_data": patient_data,
        "progress": 0,
        "current_step": "Initializing...",
        "results": None,
        "error": None
    }
    return session_id

def update_session(session_id: str, **kwargs):
    """Update session data"""
    if session_id in sessions:
        sessions[session_id].update(kwargs)

# Main endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Health Analysis Agent API",
        "version": "1.0.0",
        "status": "running",
        "agent_available": AGENT_AVAILABLE and health_agent is not None,
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze-sync",
            "chat": "/chat",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agent_available": AGENT_AVAILABLE and health_agent is not None,
        "total_sessions": len(sessions)
    }

@app.get("/info")
async def get_server_info():
    """Get detailed server information"""
    return {
        "title": "Health Analysis Agent API",
        "version": "1.0.0", 
        "description": "FastAPI wrapper for comprehensive health analysis",
        "agent_available": AGENT_AVAILABLE,
        "agent_initialized": health_agent is not None,
        "total_sessions": len(sessions),
        "endpoints": {
            "health": "GET /health - Health check",
            "analyze": "POST /analyze-sync - Run health analysis",
            "chat": "POST /chat - Chat with analysis results",
            "chat_simple": "POST /chat-simple - Simple chat interface",
            "status": "GET /status/{session_id} - Check analysis status",
            "results": "GET /results/{session_id} - Get analysis results",
            "sessions": "GET /sessions - List all sessions",
            "debug": "GET /debug/{session_id} - Debug session info"
        }
    }

@app.post("/analyze")
async def analyze_patient_data(patient_data: PatientData, background_tasks: BackgroundTasks):
    """Start asynchronous health analysis"""
    check_agent_available()
    
    session_id = create_session(patient_data.dict())
    background_tasks.add_task(run_analysis_task, session_id, patient_data.dict())
    
    return {
        "success": True,
        "session_id": session_id,
        "message": "Analysis started. Use session ID to check status and get results.",
        "status_endpoint": f"/status/{session_id}",
        "results_endpoint": f"/results/{session_id}"
    }

@app.post("/analyze-sync")
async def analyze_patient_data_sync(patient_data: PatientData):
    """Run synchronous health analysis (recommended)"""
    check_agent_available()
    
    session_id = create_session(patient_data.dict(), "running")
    
    try:
        logger.info(f"🔬 Starting synchronous analysis for session {session_id}")
        
        # Update session status
        update_session(session_id, current_step="Running comprehensive analysis...")
        
        # Run analysis
        results = health_agent.run_analysis(patient_data.dict())
        
        # Update session with results
        update_session(session_id, 
            status="completed", 
            results=results, 
            progress=100,
            current_step="Analysis completed",
            completed_at=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Analysis completed for session {session_id}")
        
        return {
            "success": results.get("success", False),
            "session_id": session_id,
            "message": "Analysis completed successfully" if results.get("success") else "Analysis completed with errors",
            "analysis_results": results,
            "chat_ready": results.get("chatbot_ready", False),
            "errors": results.get("errors", [])
        }
        
    except Exception as e:
        logger.error(f"❌ Analysis failed for session {session_id}: {str(e)}")
        
        update_session(session_id, 
            status="error", 
            error=str(e),
            current_step=f"Error: {str(e)}",
            completed_at=datetime.now().isoformat()
        )
        
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/chat")
async def chat_with_analysis(chat_request: ChatRequest):
    """Chat with medical analysis data - works with empty chat history"""
    check_agent_available()
    
    session_id = chat_request.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    if session_data["status"] != "completed" or not session_data.get("results"):
        raise HTTPException(status_code=400, detail="Analysis not completed or failed")
    
    results = session_data["results"]
    if not results.get("chatbot_ready", False):
        raise HTTPException(status_code=400, detail="Chatbot not ready. Analysis may have failed.")
    
    try:
        chatbot_context = results.get("chatbot_context", {})
        
        # Ensure chat_history is always a list
        chat_history = chat_request.chat_history or []
        
        # Debug logging
        logger.info(f"💬 Chat question: {chat_request.question}")
        logger.info(f"📝 Chat history length: {len(chat_history)}")
        logger.info(f"📊 Context available: {bool(chatbot_context)}")
        
        # Validate chatbot context
        if not chatbot_context:
            logger.error("❌ No chatbot context available")
            return {
                "success": False,
                "session_id": session_id,
                "response": "Chat context not available. Please run analysis first.",
                "updated_chat_history": chat_history
            }
        
        # Call the health agent chat method
        response = health_agent.chat_with_data(
            chat_request.question,
            chatbot_context,
            chat_history
        )
        
        # Validate response
        if not response or response.strip() == "":
            logger.error("❌ Empty response from health agent")
            response = "I'm sorry, I couldn't generate a response. Please try rephrasing your question."
        elif '"detail"' in response or response.startswith("Error"):
            logger.error(f"❌ Error in response: {response}")
            response = "I encountered an error processing your question. Please try a different question about the medical analysis."
        
        # Create updated chat history
        updated_history = chat_history + [
            {"role": "user", "content": chat_request.question},
            {"role": "assistant", "content": response}
        ]
        
        logger.info(f"✅ Chat successful, response length: {len(response)}")
        
        return {
            "success": True,
            "session_id": session_id,
            "response": response,
            "updated_chat_history": updated_history
        }
        
    except Exception as e:
        logger.error(f"❌ Chat failed with exception: {str(e)}")
        
        error_response = f"I encountered an error: {str(e)}. Please try a simpler question about the medical analysis."
        
        return {
            "success": False,
            "session_id": session_id,
            "response": error_response,
            "updated_chat_history": (chat_request.chat_history or []) + [
                {"role": "user", "content": chat_request.question},
                {"role": "assistant", "content": error_response}
            ]
        }

@app.post("/chat-simple")
async def simple_chat(request: SimpleChatRequest):
    """Simple chat endpoint without chat history requirement"""
    try:
        # Create a ChatRequest object with empty chat history
        chat_request = ChatRequest(
            session_id=request.session_id,
            question=request.question,
            chat_history=[]
        )
        
        # Use the main chat function
        return await chat_with_analysis(chat_request)
        
    except Exception as e:
        logger.error(f"❌ Simple chat error: {str(e)}")
        return {
            "success": False,
            "session_id": request.session_id,
            "response": f"Simple chat error: {str(e)}",
            "updated_chat_history": []
        }

@app.get("/status/{session_id}")
async def get_analysis_status(session_id: str):
    """Get analysis status"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    return {
        "session_id": session_id,
        "status": session_data["status"],
        "progress": session_data.get("progress", 0),
        "current_step": session_data.get("current_step", "Unknown"),
        "results_available": session_data.get("results") is not None,
        "chat_ready": session_data.get("results", {}).get("chatbot_ready", False) if session_data.get("results") else False,
        "created_at": session_data.get("created_at"),
        "completed_at": session_data.get("completed_at")
    }

@app.get("/results/{session_id}")
async def get_analysis_results(session_id: str):
    """Get analysis results"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    if not session_data.get("results"):
        raise HTTPException(status_code=404, detail="Results not available for this session")
    
    return {
        "session_id": session_id,
        "results": session_data["results"],
        "completed_at": session_data.get("completed_at"),
        "analysis_successful": session_data["results"].get("success", False),
        "chat_ready": session_data["results"].get("chatbot_ready", False)
    }

@app.get("/sessions")
async def list_sessions():
    """List all sessions"""
    return {
        "total_sessions": len(sessions),
        "sessions": [
            {
                "session_id": sid,
                "status": data["status"],
                "created_at": data["created_at"],
                "progress": data.get("progress", 0),
                "chat_ready": data.get("results", {}).get("chatbot_ready", False) if data.get("results") else False
            }
            for sid, data in sessions.items()
        ]
    }

@app.get("/debug/{session_id}")
async def debug_session(session_id: str):
    """Debug session information"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    results = session_data.get("results", {})
    
    debug_info = {
        "session_id": session_id,
        "session_status": session_data.get("status"),
        "analysis_success": results.get("success", False),
        "chatbot_ready": results.get("chatbot_ready", False),
        "has_chatbot_context": bool(results.get("chatbot_context")),
        "chatbot_context_keys": list(results.get("chatbot_context", {}).keys()) if results.get("chatbot_context") else [],
        "errors": results.get("errors", []),
        "step_status": results.get("step_status", {}),
        "processing_complete": results.get("processing_complete", False),
        "heart_attack_prediction": bool(results.get("heart_attack_prediction")),
        "entity_extraction": bool(results.get("entity_extraction")),
        "created_at": session_data.get("created_at"),
        "completed_at": session_data.get("completed_at")
    }
    
    return debug_info

@app.get("/test-chat/{session_id}")
async def test_chat_endpoint(session_id: str):
    """Test if chat is working for a session"""
    try:
        test_request = ChatRequest(
            session_id=session_id,
            question="Hello, are you working?",
            chat_history=[]
        )
        
        result = await chat_with_analysis(test_request)
        return {
            "test_successful": result["success"],
            "session_id": session_id,
            "test_response": result["response"],
            "response_length": len(result["response"]) if result.get("response") else 0
        }
        
    except Exception as e:
        return {
            "test_successful": False,
            "session_id": session_id,
            "error": str(e)
        }

# Background task for async analysis
async def run_analysis_task(session_id: str, patient_data: Dict[str, Any]):
    """Background task for analysis"""
    try:
        logger.info(f"🔬 Starting background analysis for session {session_id}")
        
        update_session(session_id, progress=10, current_step="Processing patient data...")
        
        # Run the analysis
        results = health_agent.run_analysis(patient_data)
        
        update_session(session_id,
            status="completed",
            results=results,
            progress=100,
            current_step="Analysis completed",
            completed_at=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Background analysis completed for session {session_id}")
        
    except Exception as e:
        logger.error(f"❌ Background analysis failed for session {session_id}: {str(e)}")
        
        update_session(session_id, 
            status="error", 
            error=str(e),
            current_step=f"Error: {str(e)}",
            completed_at=datetime.now().isoformat()
        )

# Test endpoints
@app.get("/test/agent")
async def test_agent_connection():
    """Test the health agent connection"""
    if not AGENT_AVAILABLE:
        return {"success": False, "error": import_error}
    
    if not health_agent:
        return {"success": False, "error": "Agent not initialized"}
    
    try:
        # Test LLM connection
        llm_test = health_agent.test_llm_connection()
        
        # Test backend connection  
        backend_test = health_agent.test_backend_connection()
        
        # Test FastAPI connection
        fastapi_test = health_agent.test_fastapi_connection()
        
        return {
            "success": True,
            "agent_available": True,
            "llm_connection": llm_test,
            "backend_connection": backend_test,
            "fastapi_connection": fastapi_test,
            "config": {
                "heart_attack_api_url": health_agent.config.heart_attack_api_url,
                "backend_url": health_agent.config.fastapi_url
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/test/sample-data")
async def get_sample_data():
    """Get sample patient data for testing"""
    return {
        "sample_patient_data": {
            "first_name": "John",
            "last_name": "Doe", 
            "ssn": "123456789",
            "date_of_birth": "1980-01-01",
            "gender": "M",
            "zip_code": "12345"
        },
        "usage": "POST this data to /analyze-sync to test the analysis"
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {"message": f"Session {session_id} deleted successfully"}

@app.delete("/sessions")
async def clear_all_sessions():
    """Clear all sessions"""
    count = len(sessions)
    sessions.clear()
    return {"message": f"Cleared {count} sessions"}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Health Analysis Agent FastAPI server starting up...")
    logger.info(f"🏥 Agent available: {AGENT_AVAILABLE}")
    if AGENT_AVAILABLE and health_agent:
        logger.info("✅ Health agent initialized and ready")
    else:
        logger.warning("⚠️ Health agent not available or failed to initialize")

if __name__ == "__main__":
    import uvicorn
    
    # Fixed port 8008 as requested
    PORT = 8008
    HOST = "127.0.0.1"
    
    print(f"🚀 Starting Health Agent FastAPI server...")
    print(f"🌐 Server URL: http://{HOST}:{PORT}")
    print(f"📋 API Documentation: http://{HOST}:{PORT}/docs")
    print(f"🔍 Health Check: http://{HOST}:{PORT}/health")
    print(f"🧪 Test Agent: http://{HOST}:{PORT}/test/agent")
    print("-" * 50)
    
    try:
        uvicorn.run("__main__:app", host=HOST, port=PORT, reload=True)
    except OSError as e:
        if "10013" in str(e) or "Address already in use" in str(e):
            print(f"❌ Port {PORT} is already in use!")
            print("💡 Solutions:")
            print(f"   1. Kill the process using port {PORT}")
            print(f"   2. Run: netstat -ano | findstr :{PORT}")
            print(f"   3. Then: taskkill /PID <PID> /F")
            print(f"   4. Or restart your computer")
        else:
            print(f"❌ Server failed to start: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
