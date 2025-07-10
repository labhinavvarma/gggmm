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
            logger.info("✅ Health Analysis Agent initialized")
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
    allow_origins=["*"],
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
    chat_history: Optional[List[Dict[str, str]]] = []

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
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agent_available": AGENT_AVAILABLE and health_agent is not None
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
        "message": "Analysis started. Use session ID to check status."
    }

@app.post("/analyze-sync")
async def analyze_patient_data_sync(patient_data: PatientData):
    """Run synchronous health analysis"""
    check_agent_available()
    
    session_id = create_session(patient_data.dict(), "running")
    
    try:
        logger.info(f"🔬 Starting analysis for session {session_id}")
        results = health_agent.run_analysis(patient_data.dict())
        
        update_session(session_id, 
            status="completed", 
            results=results, 
            progress=100,
            current_step="Completed",
            completed_at=datetime.now().isoformat()
        )
        
        return {
            "success": results.get("success", False),
            "session_id": session_id,
            "message": "Analysis completed",
            "analysis_results": results
        }
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        update_session(session_id, status="error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/chat")
async def chat_with_analysis(chat_request: ChatRequest):
    """Chat with medical analysis data"""
    check_agent_available()
    
    session_id = chat_request.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    if session_data["status"] != "completed" or not session_data.get("results"):
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    results = session_data["results"]
    if not results.get("chatbot_ready", False):
        raise HTTPException(status_code=400, detail="Chatbot not ready")
    
    try:
        chatbot_context = results.get("chatbot_context", {})
        response = health_agent.chat_with_data(
            chat_request.question,
            chatbot_context,
            chat_request.chat_history
        )
        
        updated_history = chat_request.chat_history + [
            {"role": "user", "content": chat_request.question},
            {"role": "assistant", "content": response}
        ]
        
        return {
            "success": True,
            "session_id": session_id,
            "response": response,
            "updated_chat_history": updated_history
        }
        
    except Exception as e:
        logger.error(f"❌ Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

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
        "results_available": session_data.get("results") is not None
    }

@app.get("/results/{session_id}")
async def get_analysis_results(session_id: str):
    """Get analysis results"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    if not session_data.get("results"):
        raise HTTPException(status_code=404, detail="Results not available")
    
    return {
        "session_id": session_id,
        "results": session_data["results"],
        "completed_at": session_data.get("completed_at")
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
                "progress": data.get("progress", 0)
            }
            for sid, data in sessions.items()
        ]
    }

# Background task
async def run_analysis_task(session_id: str, patient_data: Dict[str, Any]):
    """Background task for analysis"""
    try:
        logger.info(f"🔬 Starting background analysis for {session_id}")
        
        update_session(session_id, progress=10, current_step="Processing...")
        results = health_agent.run_analysis(patient_data)
        
        update_session(session_id,
            status="completed",
            results=results,
            progress=100,
            current_step="Completed",
            completed_at=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Analysis completed for {session_id}")
        
    except Exception as e:
        logger.error(f"❌ Background analysis failed: {str(e)}")
        update_session(session_id, 
            status="error", 
            error=str(e),
            current_step=f"Error: {str(e)}"
        )

# Test endpoints
@app.get("/test/sample-data")
async def get_sample_data():
    """Get sample patient data"""
    return {
        "first_name": "John",
        "last_name": "Doe", 
        "ssn": "123456789",
        "date_of_birth": "1980-01-01",
        "gender": "M",
        "zip_code": "12345"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("health_agent_fastapi:app", host="0.0.0.0", port=8000, reload=True)
