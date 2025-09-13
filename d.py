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
    description="FastAPI wrapper for comprehensive health analysis with JSON graph generation",
    version="2.0.0",
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

class ChatResponse(BaseModel):
    success: bool
    session_id: str
    response: str
    updated_chat_history: List[Dict[str, str]]
    # New fields for JSON graph generation
    graph_present: Optional[int] = 0
    graph_boundaries: Optional[Dict[str, Any]] = None
    json_graph_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
 
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

def safe_get_string_response(response_data: Any) -> str:
    """Safely extract string response from either string or dict response"""
    try:
        if isinstance(response_data, str):
            return response_data
        elif isinstance(response_data, dict):
            return response_data.get("response", str(response_data))
        else:
            return str(response_data)
    except Exception as e:
        logger.error(f"Error extracting response: {e}")
        return f"Error processing response: {str(e)}"
 
# Main endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agent_available": AGENT_AVAILABLE and health_agent is not None,
        "features": {
            "json_graph_generation": True,
            "boundary_markers": True,
            "matplotlib_conversion": True
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
 
@app.post("/chat", response_model=ChatResponse)
async def chat_with_analysis(chat_request: ChatRequest):
    """Chat with medical analysis data - Updated to handle JSON response format"""
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
       
        # Debug logging
        logger.info(f"💬 Chat request: {chat_request.question}")
        logger.info(f"📊 Context keys: {list(chatbot_context.keys())}")
       
        # Call the chat method - now expecting dict response
        chat_response = health_agent.chat_with_data(
            chat_request.question,
            chatbot_context,
            chat_request.chat_history
        )
       
        logger.info(f"🔍 Chat response type: {type(chat_response)}")
        
        # Handle both old string responses and new dict responses
        if isinstance(chat_response, dict):
            # New format with JSON graph support
            success = chat_response.get("success", True)
            response_text = chat_response.get("response", "")
            updated_history = chat_response.get("updated_chat_history", chat_request.chat_history + [
                {"role": "user", "content": chat_request.question},
                {"role": "assistant", "content": response_text}
            ])
            
            # Extract additional fields for JSON graph generation
            graph_present = chat_response.get("graph_present", 0)
            graph_boundaries = chat_response.get("graph_boundaries", {
                "start_position": None, 
                "end_position": None, 
                "has_markers": False
            })
            json_graph_data = chat_response.get("json_graph_data", None)
            error = chat_response.get("error", None)
            
            # Validate response text
            if not response_text or (isinstance(response_text, str) and response_text.strip() == ""):
                logger.warning(f"⚠️ Empty response text detected")
                response_text = "I'm having trouble processing your request. Please try a different question."
                success = False
            
            return ChatResponse(
                success=success,
                session_id=session_id,
                response=response_text,
                updated_chat_history=updated_history,
                graph_present=graph_present,
                graph_boundaries=graph_boundaries,
                json_graph_data=json_graph_data,
                error=error
            )
            
        elif isinstance(chat_response, str):
            # Legacy string response format
            if not chat_response or chat_response.strip() == "" or '"detail"' in chat_response:
                logger.error(f"❌ Empty or error response: {chat_response}")
                response_text = "I'm having trouble processing your request. Please try a different question or check if the analysis completed successfully."
                success = False
            else:
                response_text = chat_response
                success = True
            
            updated_history = chat_request.chat_history + [
                {"role": "user", "content": chat_request.question},
                {"role": "assistant", "content": response_text}
            ]
            
            return ChatResponse(
                success=success,
                session_id=session_id,
                response=response_text,
                updated_chat_history=updated_history,
                graph_present=0,
                graph_boundaries={"start_position": None, "end_position": None, "has_markers": False},
                json_graph_data=None,
                error=None if success else "Legacy response format"
            )
        else:
            # Unexpected response type
            logger.error(f"❌ Unexpected response type: {type(chat_response)}")
            response_text = f"Unexpected response format: {type(chat_response)}"
            
            return ChatResponse(
                success=False,
                session_id=session_id,
                response=response_text,
                updated_chat_history=chat_request.chat_history + [
                    {"role": "user", "content": chat_request.question},
                    {"role": "assistant", "content": response_text}
                ],
                graph_present=0,
                graph_boundaries={"start_position": None, "end_position": None, "has_markers": False},
                json_graph_data=None,
                error=f"Unexpected response type: {type(chat_response)}"
            )
       
    except Exception as e:
        logger.error(f"❌ Chat failed: {str(e)}")
        error_response = f"Chat error: {str(e)}"
        
        return ChatResponse(
            success=False,
            session_id=session_id,
            response=error_response,
            updated_chat_history=chat_request.chat_history + [
                {"role": "user", "content": chat_request.question},
                {"role": "assistant", "content": error_response}
            ],
            graph_present=0,
            graph_boundaries={"start_position": None, "end_position": None, "has_markers": False},
            json_graph_data=None,
            error=str(e)
        )

@app.post("/chat/graph-test")
async def test_graph_generation(chat_request: ChatRequest):
    """Test endpoint specifically for graph generation"""
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
        
        # Force a graph request for testing
        test_queries = [
            "show me a diagnosis frequency chart",
            "create a medication distribution chart",
            "generate a risk assessment visualization",
            "show me a timeline of healthcare activities"
        ]
        
        test_query = chat_request.question if any(keyword in chat_request.question.lower() for keyword in ['chart', 'graph', 'show', 'create', 'generate']) else test_queries[0]
        
        logger.info(f"🧪 Testing graph generation with query: {test_query}")
        
        chat_response = health_agent.chat_with_data(
            test_query,
            chatbot_context,
            chat_request.chat_history
        )
        
        return {
            "test_query": test_query,
            "response_type": type(chat_response).__name__,
            "response_data": chat_response,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ Graph test failed: {str(e)}")
        return {
            "test_query": chat_request.question,
            "response_type": "error",
            "response_data": str(e),
            "success": False
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
        "json_graph_generation_ready": session_data.get("results", {}).get("json_graph_generation_ready", False) if session_data.get("results") else False
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
        "completed_at": session_data.get("completed_at"),
        "features": {
            "chatbot_ready": session_data["results"].get("chatbot_ready", False),
            "json_graph_generation_ready": session_data["results"].get("json_graph_generation_ready", False),
            "enhancement_version": session_data["results"].get("enhancement_version", "unknown")
        }
    }

@app.get("/results/{session_id}/summary")
async def get_analysis_summary(session_id: str):
    """Get condensed analysis summary"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
   
    session_data = sessions[session_id]
    if not session_data.get("results"):
        raise HTTPException(status_code=404, detail="Results not available")
    
    results = session_data["results"]
    
    return {
        "session_id": session_id,
        "success": results.get("success", False),
        "summary": {
            "health_trajectory": results.get("health_trajectory", ""),
            "final_summary": results.get("final_summary", ""),
            "heart_attack_prediction": results.get("heart_attack_prediction", {}),
            "entity_extraction": results.get("entity_extraction", {}),
            "chatbot_ready": results.get("chatbot_ready", False),
            "json_graph_generation_ready": results.get("json_graph_generation_ready", False)
        },
        "processing_info": {
            "steps_completed": results.get("processing_steps_completed", 0),
            "enhancement_version": results.get("enhancement_version", "unknown"),
            "comprehensive_analysis": results.get("comprehensive_analysis", False)
        }
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
                "chatbot_ready": data.get("results", {}).get("chatbot_ready", False) if data.get("results") else False,
                "json_graph_ready": data.get("results", {}).get("json_graph_generation_ready", False) if data.get("results") else False
            }
            for sid, data in sessions.items()
        ]
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {
        "success": True,
        "message": f"Session {session_id} deleted",
        "remaining_sessions": len(sessions)
    }

@app.delete("/sessions")
async def clear_all_sessions():
    """Clear all sessions"""
    session_count = len(sessions)
    sessions.clear()
    return {
        "success": True,
        "message": f"Cleared {session_count} sessions",
        "remaining_sessions": 0
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
        logger.info(f"📊 JSON graph generation ready: {results.get('json_graph_generation_ready', False)}")
       
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

@app.get("/test/graph-queries")
async def get_sample_graph_queries():
    """Get sample queries for testing graph generation"""
    return {
        "diagnosis_charts": [
            "show me a diagnosis frequency chart",
            "create a diagnosis distribution visualization",
            "generate a chart of ICD-10 codes"
        ],
        "medication_charts": [
            "show me medication distribution",
            "create a pharmacy chart",
            "generate medication frequency visualization"
        ],
        "risk_charts": [
            "show me risk assessment dashboard",
            "create a risk visualization",
            "generate health risk chart"
        ],
        "timeline_charts": [
            "show me healthcare timeline",
            "create a timeline visualization",
            "generate activity timeline chart"
        ],
        "general_requests": [
            "create a chart",
            "show me a graph",
            "generate visualization"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8018, reload=True)
