from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, List, Optional
import asyncio
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

# Global variables for agent and session management
health_agent = None
active_sessions: Dict[str, Dict[str, Any]] = {}
session_results: Dict[str, Dict[str, Any]] = {}

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
    
    # Cleanup
    logger.info("🔄 Shutting down Health Agent FastAPI server")

# Create FastAPI app
app = FastAPI(
    title="Health Analysis Agent API",
    description="FastAPI wrapper for the comprehensive health analysis agent",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class PatientData(BaseModel):
    """Patient data model for health analysis"""
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    ssn: str = Field(..., min_length=9, max_length=11)
    date_of_birth: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}

class ChatRequest(BaseModel):
    """Chat request model"""
    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1, max_length=1000)
    chat_history: Optional[List[Dict[str, str]]] = []

class AnalysisResponse(BaseModel):
    """Analysis response model"""
    success: bool
    session_id: str
    message: str
    analysis_results: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None

class ChatResponse(BaseModel):
    """Chat response model"""
    success: bool
    session_id: str
    response: str
    updated_chat_history: List[Dict[str, str]]

class StatusResponse(BaseModel):
    """Status response model"""
    session_id: str
    status: str  # 'running', 'completed', 'error', 'not_found'
    progress: Optional[int] = None
    current_step: Optional[str] = None
    results_available: bool = False

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agent_available": AGENT_AVAILABLE,
        "agent_status": "initialized" if health_agent else "not_initialized"
    }

# Get server info
@app.get("/info")
async def get_server_info():
    """Get server information"""
    return {
        "title": "Health Analysis Agent API",
        "version": "1.0.0",
        "description": "FastAPI wrapper for comprehensive health analysis",
        "agent_available": AGENT_AVAILABLE,
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "chat": "/chat",
            "status": "/status/{session_id}",
            "results": "/results/{session_id}"
        }
    }

# Main analysis endpoint
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_patient_data(patient_data: PatientData, background_tasks: BackgroundTasks):
    """
    Run comprehensive health analysis on patient data
    """
    if not AGENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Health Analysis Agent not available: {import_error}"
        )
    
    if not health_agent:
        raise HTTPException(
            status_code=503,
            detail="Health Analysis Agent not initialized"
        )
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    # Store session info
    active_sessions[session_id] = {
        "status": "running",
        "created_at": datetime.now().isoformat(),
        "patient_data": patient_data.dict(),
        "progress": 0,
        "current_step": "Initializing analysis..."
    }
    
    # Start analysis in background
    background_tasks.add_task(run_analysis_task, session_id, patient_data.dict())
    
    return AnalysisResponse(
        success=True,
        session_id=session_id,
        message="Health analysis started successfully. Use the session ID to check status and retrieve results.",
        analysis_results=None,
        errors=None
    )

# Synchronous analysis endpoint (for immediate results)
@app.post("/analyze-sync", response_model=AnalysisResponse)
async def analyze_patient_data_sync(patient_data: PatientData):
    """
    Run comprehensive health analysis synchronously (blocks until complete)
    """
    if not AGENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Health Analysis Agent not available: {import_error}"
        )
    
    if not health_agent:
        raise HTTPException(
            status_code=503,
            detail="Health Analysis Agent not initialized"
        )
    
    session_id = str(uuid.uuid4())
    
    try:
        # Run analysis synchronously
        logger.info(f"🔬 Starting synchronous analysis for session {session_id}")
        
        # Convert patient data to dict and run analysis
        patient_dict = patient_data.dict()
        results = health_agent.run_analysis(patient_dict)
        
        # Store results
        session_results[session_id] = {
            "results": results,
            "completed_at": datetime.now().isoformat(),
            "session_id": session_id
        }
        
        active_sessions[session_id] = {
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "patient_data": patient_dict,
            "progress": 100,
            "current_step": "Analysis completed"
        }
        
        return AnalysisResponse(
            success=results.get("success", False),
            session_id=session_id,
            message="Health analysis completed successfully" if results.get("success") else "Analysis completed with errors",
            analysis_results=results,
            errors=results.get("errors", [])
        )
        
    except Exception as e:
        logger.error(f"❌ Analysis failed for session {session_id}: {str(e)}")
        
        active_sessions[session_id] = {
            "status": "error",
            "created_at": datetime.now().isoformat(),
            "error": str(e),
            "patient_data": patient_data.dict(),
            "progress": 0,
            "current_step": f"Error: {str(e)}"
        }
        
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_with_analysis(chat_request: ChatRequest):
    """
    Chat with the medical analysis data
    """
    if not AGENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Health Analysis Agent not available: {import_error}"
        )
    
    if not health_agent:
        raise HTTPException(
            status_code=503,
            detail="Health Analysis Agent not initialized"
        )
    
    session_id = chat_request.session_id
    
    # Check if session exists and has results
    if session_id not in session_results:
        raise HTTPException(
            status_code=404,
            detail="Session not found or analysis not completed"
        )
    
    session_data = session_results[session_id]
    results = session_data["results"]
    
    # Check if chatbot is ready
    if not results.get("chatbot_ready", False):
        raise HTTPException(
            status_code=400,
            detail="Chatbot not ready for this session. Analysis may have failed."
        )
    
    try:
        # Get chatbot context
        chatbot_context = results.get("chatbot_context", {})
        
        # Get or initialize chat history
        chat_history = chat_request.chat_history or []
        
        # Call the chatbot
        response = health_agent.chat_with_data(
            chat_request.question,
            chatbot_context,
            chat_history
        )
        
        # Update chat history
        updated_history = chat_history + [
            {"role": "user", "content": chat_request.question},
            {"role": "assistant", "content": response}
        ]
        
        return ChatResponse(
            success=True,
            session_id=session_id,
            response=response,
            updated_chat_history=updated_history
        )
        
    except Exception as e:
        logger.error(f"❌ Chat failed for session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )

# Status endpoint
@app.get("/status/{session_id}", response_model=StatusResponse)
async def get_analysis_status(session_id: str):
    """
    Get the status of a running analysis
    """
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    
    session_info = active_sessions[session_id]
    
    return StatusResponse(
        session_id=session_id,
        status=session_info["status"],
        progress=session_info.get("progress", 0),
        current_step=session_info.get("current_step", "Unknown"),
        results_available=session_id in session_results
    )

# Results endpoint
@app.get("/results/{session_id}")
async def get_analysis_results(session_id: str):
    """
    Get the results of a completed analysis
    """
    if session_id not in session_results:
        raise HTTPException(
            status_code=404,
            detail="Results not found for this session"
        )
    
    return session_results[session_id]

# List sessions endpoint
@app.get("/sessions")
async def list_sessions():
    """
    List all active and completed sessions
    """
    return {
        "active_sessions": len(active_sessions),
        "completed_sessions": len(session_results),
        "sessions": [
            {
                "session_id": sid,
                "status": info["status"],
                "created_at": info["created_at"],
                "progress": info.get("progress", 0)
            }
            for sid, info in active_sessions.items()
        ]
    }

# Background task for running analysis
async def run_analysis_task(session_id: str, patient_data: Dict[str, Any]):
    """
    Background task to run health analysis
    """
    try:
        logger.info(f"🔬 Starting background analysis for session {session_id}")
        
        # Update progress
        active_sessions[session_id]["progress"] = 10
        active_sessions[session_id]["current_step"] = "Initializing workflow..."
        
        # Run the analysis (this is blocking, but runs in background)
        results = health_agent.run_analysis(patient_data)
        
        # Update progress
        active_sessions[session_id]["progress"] = 90
        active_sessions[session_id]["current_step"] = "Finalizing results..."
        
        # Store results
        session_results[session_id] = {
            "results": results,
            "completed_at": datetime.now().isoformat(),
            "session_id": session_id
        }
        
        # Update session status
        active_sessions[session_id]["status"] = "completed"
        active_sessions[session_id]["progress"] = 100
        active_sessions[session_id]["current_step"] = "Analysis completed successfully"
        active_sessions[session_id]["completed_at"] = datetime.now().isoformat()
        
        logger.info(f"✅ Analysis completed for session {session_id}")
        
    except Exception as e:
        logger.error(f"❌ Background analysis failed for session {session_id}: {str(e)}")
        
        # Update session with error
        active_sessions[session_id]["status"] = "error"
        active_sessions[session_id]["error"] = str(e)
        active_sessions[session_id]["current_step"] = f"Error: {str(e)}"
        active_sessions[session_id]["completed_at"] = datetime.now().isoformat()

# Test endpoints for development
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
        
        return {
            "success": True,
            "agent_available": True,
            "llm_connection": llm_test,
            "backend_connection": backend_test
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Sample patient data for testing
@app.get("/test/sample-data")
async def get_sample_patient_data():
    """Get sample patient data for testing"""
    return {
        "sample_patient_data": {
            "first_name": "John",
            "last_name": "Doe",
            "ssn": "123456789",
            "date_of_birth": "1980-01-01",
            "gender": "M",
            "zip_code": "12345"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI server
    uvicorn.run(
        "health_agent_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ))
    gender: str = Field(..., pattern=r'^[MF]

class ChatRequest(BaseModel):
    """Chat request model"""
    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1, max_length=1000)
    chat_history: Optional[List[Dict[str, str]]] = []

class AnalysisResponse(BaseModel):
    """Analysis response model"""
    success: bool
    session_id: str
    message: str
    analysis_results: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None

class ChatResponse(BaseModel):
    """Chat response model"""
    success: bool
    session_id: str
    response: str
    updated_chat_history: List[Dict[str, str]]

class StatusResponse(BaseModel):
    """Status response model"""
    session_id: str
    status: str  # 'running', 'completed', 'error', 'not_found'
    progress: Optional[int] = None
    current_step: Optional[str] = None
    results_available: bool = False

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agent_available": AGENT_AVAILABLE,
        "agent_status": "initialized" if health_agent else "not_initialized"
    }

# Get server info
@app.get("/info")
async def get_server_info():
    """Get server information"""
    return {
        "title": "Health Analysis Agent API",
        "version": "1.0.0",
        "description": "FastAPI wrapper for comprehensive health analysis",
        "agent_available": AGENT_AVAILABLE,
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "chat": "/chat",
            "status": "/status/{session_id}",
            "results": "/results/{session_id}"
        }
    }

# Main analysis endpoint
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_patient_data(patient_data: PatientData, background_tasks: BackgroundTasks):
    """
    Run comprehensive health analysis on patient data
    """
    if not AGENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Health Analysis Agent not available: {import_error}"
        )
    
    if not health_agent:
        raise HTTPException(
            status_code=503,
            detail="Health Analysis Agent not initialized"
        )
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    # Store session info
    active_sessions[session_id] = {
        "status": "running",
        "created_at": datetime.now().isoformat(),
        "patient_data": patient_data.dict(),
        "progress": 0,
        "current_step": "Initializing analysis..."
    }
    
    # Start analysis in background
    background_tasks.add_task(run_analysis_task, session_id, patient_data.dict())
    
    return AnalysisResponse(
        success=True,
        session_id=session_id,
        message="Health analysis started successfully. Use the session ID to check status and retrieve results.",
        analysis_results=None,
        errors=None
    )

# Synchronous analysis endpoint (for immediate results)
@app.post("/analyze-sync", response_model=AnalysisResponse)
async def analyze_patient_data_sync(patient_data: PatientData):
    """
    Run comprehensive health analysis synchronously (blocks until complete)
    """
    if not AGENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Health Analysis Agent not available: {import_error}"
        )
    
    if not health_agent:
        raise HTTPException(
            status_code=503,
            detail="Health Analysis Agent not initialized"
        )
    
    session_id = str(uuid.uuid4())
    
    try:
        # Run analysis synchronously
        logger.info(f"🔬 Starting synchronous analysis for session {session_id}")
        
        # Convert patient data to dict and run analysis
        patient_dict = patient_data.dict()
        results = health_agent.run_analysis(patient_dict)
        
        # Store results
        session_results[session_id] = {
            "results": results,
            "completed_at": datetime.now().isoformat(),
            "session_id": session_id
        }
        
        active_sessions[session_id] = {
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "patient_data": patient_dict,
            "progress": 100,
            "current_step": "Analysis completed"
        }
        
        return AnalysisResponse(
            success=results.get("success", False),
            session_id=session_id,
            message="Health analysis completed successfully" if results.get("success") else "Analysis completed with errors",
            analysis_results=results,
            errors=results.get("errors", [])
        )
        
    except Exception as e:
        logger.error(f"❌ Analysis failed for session {session_id}: {str(e)}")
        
        active_sessions[session_id] = {
            "status": "error",
            "created_at": datetime.now().isoformat(),
            "error": str(e),
            "patient_data": patient_data.dict(),
            "progress": 0,
            "current_step": f"Error: {str(e)}"
        }
        
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_with_analysis(chat_request: ChatRequest):
    """
    Chat with the medical analysis data
    """
    if not AGENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Health Analysis Agent not available: {import_error}"
        )
    
    if not health_agent:
        raise HTTPException(
            status_code=503,
            detail="Health Analysis Agent not initialized"
        )
    
    session_id = chat_request.session_id
    
    # Check if session exists and has results
    if session_id not in session_results:
        raise HTTPException(
            status_code=404,
            detail="Session not found or analysis not completed"
        )
    
    session_data = session_results[session_id]
    results = session_data["results"]
    
    # Check if chatbot is ready
    if not results.get("chatbot_ready", False):
        raise HTTPException(
            status_code=400,
            detail="Chatbot not ready for this session. Analysis may have failed."
        )
    
    try:
        # Get chatbot context
        chatbot_context = results.get("chatbot_context", {})
        
        # Get or initialize chat history
        chat_history = chat_request.chat_history or []
        
        # Call the chatbot
        response = health_agent.chat_with_data(
            chat_request.question,
            chatbot_context,
            chat_history
        )
        
        # Update chat history
        updated_history = chat_history + [
            {"role": "user", "content": chat_request.question},
            {"role": "assistant", "content": response}
        ]
        
        return ChatResponse(
            success=True,
            session_id=session_id,
            response=response,
            updated_chat_history=updated_history
        )
        
    except Exception as e:
        logger.error(f"❌ Chat failed for session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )

# Status endpoint
@app.get("/status/{session_id}", response_model=StatusResponse)
async def get_analysis_status(session_id: str):
    """
    Get the status of a running analysis
    """
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    
    session_info = active_sessions[session_id]
    
    return StatusResponse(
        session_id=session_id,
        status=session_info["status"],
        progress=session_info.get("progress", 0),
        current_step=session_info.get("current_step", "Unknown"),
        results_available=session_id in session_results
    )

# Results endpoint
@app.get("/results/{session_id}")
async def get_analysis_results(session_id: str):
    """
    Get the results of a completed analysis
    """
    if session_id not in session_results:
        raise HTTPException(
            status_code=404,
            detail="Results not found for this session"
        )
    
    return session_results[session_id]

# List sessions endpoint
@app.get("/sessions")
async def list_sessions():
    """
    List all active and completed sessions
    """
    return {
        "active_sessions": len(active_sessions),
        "completed_sessions": len(session_results),
        "sessions": [
            {
                "session_id": sid,
                "status": info["status"],
                "created_at": info["created_at"],
                "progress": info.get("progress", 0)
            }
            for sid, info in active_sessions.items()
        ]
    }

# Background task for running analysis
async def run_analysis_task(session_id: str, patient_data: Dict[str, Any]):
    """
    Background task to run health analysis
    """
    try:
        logger.info(f"🔬 Starting background analysis for session {session_id}")
        
        # Update progress
        active_sessions[session_id]["progress"] = 10
        active_sessions[session_id]["current_step"] = "Initializing workflow..."
        
        # Run the analysis (this is blocking, but runs in background)
        results = health_agent.run_analysis(patient_data)
        
        # Update progress
        active_sessions[session_id]["progress"] = 90
        active_sessions[session_id]["current_step"] = "Finalizing results..."
        
        # Store results
        session_results[session_id] = {
            "results": results,
            "completed_at": datetime.now().isoformat(),
            "session_id": session_id
        }
        
        # Update session status
        active_sessions[session_id]["status"] = "completed"
        active_sessions[session_id]["progress"] = 100
        active_sessions[session_id]["current_step"] = "Analysis completed successfully"
        active_sessions[session_id]["completed_at"] = datetime.now().isoformat()
        
        logger.info(f"✅ Analysis completed for session {session_id}")
        
    except Exception as e:
        logger.error(f"❌ Background analysis failed for session {session_id}: {str(e)}")
        
        # Update session with error
        active_sessions[session_id]["status"] = "error"
        active_sessions[session_id]["error"] = str(e)
        active_sessions[session_id]["current_step"] = f"Error: {str(e)}"
        active_sessions[session_id]["completed_at"] = datetime.now().isoformat()

# Test endpoints for development
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
        
        return {
            "success": True,
            "agent_available": True,
            "llm_connection": llm_test,
            "backend_connection": backend_test
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Sample patient data for testing
@app.get("/test/sample-data")
async def get_sample_patient_data():
    """Get sample patient data for testing"""
    return {
        "sample_patient_data": {
            "first_name": "John",
            "last_name": "Doe",
            "ssn": "123456789",
            "date_of_birth": "1980-01-01",
            "gender": "M",
            "zip_code": "12345"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI server
    uvicorn.run(
        "health_agent_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ))
    zip_code: str = Field(..., min_length=5, max_length=10)
    
    @field_validator('date_of_birth')
    @classmethod
    def validate_date_of_birth(cls, v):
        try:
            birth_date = datetime.strptime(v, '%Y-%m-%d').date()
            today = date.today()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            
            if age > 150:
                raise ValueError('Age cannot be greater than 150 years')
            elif age < 0:
                raise ValueError('Date of birth cannot be in the future')
            
            return v
        except ValueError as e:
            raise ValueError(f'Invalid date format or value: {str(e)}')
    
    @field_validator('ssn')
    @classmethod
    def validate_ssn(cls, v):
        # Remove any non-digit characters
        ssn_digits = ''.join(filter(str.isdigit, v))
        if len(ssn_digits) != 9:
            raise ValueError('SSN must contain exactly 9 digits')
        return ssn_digits
    
    @field_validator('zip_code')
    @classmethod
    def validate_zip_code(cls, v):
        # Remove any non-digit characters
        zip_digits = ''.join(filter(str.isdigit, v))
        if len(zip_digits) < 5:
            raise ValueError('Zip code must contain at least 5 digits')
        return zip_digits

class ChatRequest(BaseModel):
    """Chat request model"""
    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1, max_length=1000)
    chat_history: Optional[List[Dict[str, str]]] = []

class AnalysisResponse(BaseModel):
    """Analysis response model"""
    success: bool
    session_id: str
    message: str
    analysis_results: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None

class ChatResponse(BaseModel):
    """Chat response model"""
    success: bool
    session_id: str
    response: str
    updated_chat_history: List[Dict[str, str]]

class StatusResponse(BaseModel):
    """Status response model"""
    session_id: str
    status: str  # 'running', 'completed', 'error', 'not_found'
    progress: Optional[int] = None
    current_step: Optional[str] = None
    results_available: bool = False

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agent_available": AGENT_AVAILABLE,
        "agent_status": "initialized" if health_agent else "not_initialized"
    }

# Get server info
@app.get("/info")
async def get_server_info():
    """Get server information"""
    return {
        "title": "Health Analysis Agent API",
        "version": "1.0.0",
        "description": "FastAPI wrapper for comprehensive health analysis",
        "agent_available": AGENT_AVAILABLE,
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "chat": "/chat",
            "status": "/status/{session_id}",
            "results": "/results/{session_id}"
        }
    }

# Main analysis endpoint
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_patient_data(patient_data: PatientData, background_tasks: BackgroundTasks):
    """
    Run comprehensive health analysis on patient data
    """
    if not AGENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Health Analysis Agent not available: {import_error}"
        )
    
    if not health_agent:
        raise HTTPException(
            status_code=503,
            detail="Health Analysis Agent not initialized"
        )
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    # Store session info
    active_sessions[session_id] = {
        "status": "running",
        "created_at": datetime.now().isoformat(),
        "patient_data": patient_data.dict(),
        "progress": 0,
        "current_step": "Initializing analysis..."
    }
    
    # Start analysis in background
    background_tasks.add_task(run_analysis_task, session_id, patient_data.dict())
    
    return AnalysisResponse(
        success=True,
        session_id=session_id,
        message="Health analysis started successfully. Use the session ID to check status and retrieve results.",
        analysis_results=None,
        errors=None
    )

# Synchronous analysis endpoint (for immediate results)
@app.post("/analyze-sync", response_model=AnalysisResponse)
async def analyze_patient_data_sync(patient_data: PatientData):
    """
    Run comprehensive health analysis synchronously (blocks until complete)
    """
    if not AGENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Health Analysis Agent not available: {import_error}"
        )
    
    if not health_agent:
        raise HTTPException(
            status_code=503,
            detail="Health Analysis Agent not initialized"
        )
    
    session_id = str(uuid.uuid4())
    
    try:
        # Run analysis synchronously
        logger.info(f"🔬 Starting synchronous analysis for session {session_id}")
        
        # Convert patient data to dict and run analysis
        patient_dict = patient_data.dict()
        results = health_agent.run_analysis(patient_dict)
        
        # Store results
        session_results[session_id] = {
            "results": results,
            "completed_at": datetime.now().isoformat(),
            "session_id": session_id
        }
        
        active_sessions[session_id] = {
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "patient_data": patient_dict,
            "progress": 100,
            "current_step": "Analysis completed"
        }
        
        return AnalysisResponse(
            success=results.get("success", False),
            session_id=session_id,
            message="Health analysis completed successfully" if results.get("success") else "Analysis completed with errors",
            analysis_results=results,
            errors=results.get("errors", [])
        )
        
    except Exception as e:
        logger.error(f"❌ Analysis failed for session {session_id}: {str(e)}")
        
        active_sessions[session_id] = {
            "status": "error",
            "created_at": datetime.now().isoformat(),
            "error": str(e),
            "patient_data": patient_data.dict(),
            "progress": 0,
            "current_step": f"Error: {str(e)}"
        }
        
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_with_analysis(chat_request: ChatRequest):
    """
    Chat with the medical analysis data
    """
    if not AGENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Health Analysis Agent not available: {import_error}"
        )
    
    if not health_agent:
        raise HTTPException(
            status_code=503,
            detail="Health Analysis Agent not initialized"
        )
    
    session_id = chat_request.session_id
    
    # Check if session exists and has results
    if session_id not in session_results:
        raise HTTPException(
            status_code=404,
            detail="Session not found or analysis not completed"
        )
    
    session_data = session_results[session_id]
    results = session_data["results"]
    
    # Check if chatbot is ready
    if not results.get("chatbot_ready", False):
        raise HTTPException(
            status_code=400,
            detail="Chatbot not ready for this session. Analysis may have failed."
        )
    
    try:
        # Get chatbot context
        chatbot_context = results.get("chatbot_context", {})
        
        # Get or initialize chat history
        chat_history = chat_request.chat_history or []
        
        # Call the chatbot
        response = health_agent.chat_with_data(
            chat_request.question,
            chatbot_context,
            chat_history
        )
        
        # Update chat history
        updated_history = chat_history + [
            {"role": "user", "content": chat_request.question},
            {"role": "assistant", "content": response}
        ]
        
        return ChatResponse(
            success=True,
            session_id=session_id,
            response=response,
            updated_chat_history=updated_history
        )
        
    except Exception as e:
        logger.error(f"❌ Chat failed for session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )

# Status endpoint
@app.get("/status/{session_id}", response_model=StatusResponse)
async def get_analysis_status(session_id: str):
    """
    Get the status of a running analysis
    """
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    
    session_info = active_sessions[session_id]
    
    return StatusResponse(
        session_id=session_id,
        status=session_info["status"],
        progress=session_info.get("progress", 0),
        current_step=session_info.get("current_step", "Unknown"),
        results_available=session_id in session_results
    )

# Results endpoint
@app.get("/results/{session_id}")
async def get_analysis_results(session_id: str):
    """
    Get the results of a completed analysis
    """
    if session_id not in session_results:
        raise HTTPException(
            status_code=404,
            detail="Results not found for this session"
        )
    
    return session_results[session_id]

# List sessions endpoint
@app.get("/sessions")
async def list_sessions():
    """
    List all active and completed sessions
    """
    return {
        "active_sessions": len(active_sessions),
        "completed_sessions": len(session_results),
        "sessions": [
            {
                "session_id": sid,
                "status": info["status"],
                "created_at": info["created_at"],
                "progress": info.get("progress", 0)
            }
            for sid, info in active_sessions.items()
        ]
    }

# Background task for running analysis
async def run_analysis_task(session_id: str, patient_data: Dict[str, Any]):
    """
    Background task to run health analysis
    """
    try:
        logger.info(f"🔬 Starting background analysis for session {session_id}")
        
        # Update progress
        active_sessions[session_id]["progress"] = 10
        active_sessions[session_id]["current_step"] = "Initializing workflow..."
        
        # Run the analysis (this is blocking, but runs in background)
        results = health_agent.run_analysis(patient_data)
        
        # Update progress
        active_sessions[session_id]["progress"] = 90
        active_sessions[session_id]["current_step"] = "Finalizing results..."
        
        # Store results
        session_results[session_id] = {
            "results": results,
            "completed_at": datetime.now().isoformat(),
            "session_id": session_id
        }
        
        # Update session status
        active_sessions[session_id]["status"] = "completed"
        active_sessions[session_id]["progress"] = 100
        active_sessions[session_id]["current_step"] = "Analysis completed successfully"
        active_sessions[session_id]["completed_at"] = datetime.now().isoformat()
        
        logger.info(f"✅ Analysis completed for session {session_id}")
        
    except Exception as e:
        logger.error(f"❌ Background analysis failed for session {session_id}: {str(e)}")
        
        # Update session with error
        active_sessions[session_id]["status"] = "error"
        active_sessions[session_id]["error"] = str(e)
        active_sessions[session_id]["current_step"] = f"Error: {str(e)}"
        active_sessions[session_id]["completed_at"] = datetime.now().isoformat()

# Test endpoints for development
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
        
        return {
            "success": True,
            "agent_available": True,
            "llm_connection": llm_test,
            "backend_connection": backend_test
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Sample patient data for testing
@app.get("/test/sample-data")
async def get_sample_patient_data():
    """Get sample patient data for testing"""
    return {
        "sample_patient_data": {
            "first_name": "John",
            "last_name": "Doe",
            "ssn": "123456789",
            "date_of_birth": "1980-01-01",
            "gender": "M",
            "zip_code": "12345"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI server
    uvicorn.run(
        "health_agent_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
