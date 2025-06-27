import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.routing import Mount
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import pandas as pd
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import MCP modules
try:
    from mcp.server.sse import SseServerTransport
    from mcpserver import mcp
    from router import router
    MCP_AVAILABLE = True
    print("✅ MCP modules loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: Could not import MCP modules: {e}")
    MCP_AVAILABLE = False

# Create FastAPI app with proper configuration
app = FastAPI(
    title="Combined Healthcare & Heart Disease API",
    version="1.0",
    description="Combined API with heart disease prediction, Milliman healthcare APIs, and MCP integration"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ===== HEART DISEASE MODEL SETUP =====

# Load heart disease model globally
MODEL_PACKAGE_PATH = "heart_disease_model_package.pkl"
model_data = None

print("📁 Checking heart disease model files...")
try:
    if os.path.exists(MODEL_PACKAGE_PATH):
        print("🔄 Loading heart disease model package...")
        model_package = joblib.load(MODEL_PACKAGE_PATH)
        model_data = {
            'model': model_package['model'],
            'scaler': model_package['scaler'],
            'feature_names': model_package['feature_names']
        }
        print("✅ Heart disease model loaded successfully")
    else:
        print(f"⚠️ Heart disease model package not found: {MODEL_PACKAGE_PATH}")
        model_data = None
except Exception as e:
    print(f"⚠️ Error loading heart disease model: {str(e)}")
    model_data = None

def categorize_risk(prob: float) -> str:
    """Categorize heart disease risk based on probability"""
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    else:
        return "High"

# ===== PYDANTIC MODELS =====

# Heart Disease Models
class HeartDiseaseRequest(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age in years")
    gender: int = Field(..., ge=0, le=1, description="Gender (0=Female, 1=Male)")
    diabetes: int = Field(..., ge=0, le=1, description="Diabetes status (0=No, 1=Yes)")
    high_bp: int = Field(..., ge=0, le=1, description="High blood pressure (0=No, 1=Yes)")
    smoking: int = Field(..., ge=0, le=1, description="Smoking status (0=No, 1=Yes)")

class HeartDiseaseResponse(BaseModel):
    prediction: str
    probability: float
    risk_level: str
    confidence: float
    input_data: Dict[str, Any]

# Healthcare Models
class HealthcarePatientRequest(BaseModel):
    first_name: str = Field(..., description="Patient's first name")
    last_name: str = Field(..., description="Patient's last name")
    ssn: str = Field(..., description="Social Security Number")
    date_of_birth: str = Field(..., description="Date of birth (YYYY-MM-DD)")
    gender: str = Field(..., description="Gender (M/F)")
    zip_code: str = Field(..., description="ZIP code")

class HealthResponse(BaseModel):
    status: str
    heart_disease_model_loaded: bool
    mcp_available: bool
    model_type: Optional[str] = None

# ===== ESSENTIAL ENDPOINTS =====

@app.get("/")
async def root():
    """Root endpoint - confirms server is running"""
    return {
        "message": "Combined Healthcare & Heart Disease API",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "heart_disease": "Heart disease risk prediction",
            "healthcare": "Milliman healthcare APIs"
        },
        "endpoints": {
            "health": "/health",
            "heart_disease": "/predict",
            "medical": "/medical/submit",
            "pharmacy": "/pharmacy/submit", 
            "mcid": "/mcid/search",
            "debug": "/debug/routes"
        },
        "heart_disease_model": "available" if model_data else "unavailable",
        "mcp_available": MCP_AVAILABLE,
        "version": "1.0"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - REQUIRED for diagnostics"""
    return HealthResponse(
        status="healthy",
        heart_disease_model_loaded=model_data is not None,
        mcp_available=MCP_AVAILABLE,
        model_type=type(model_data['model']).__name__ if model_data else None
    )

@app.get("/debug/routes")
async def debug_routes():
    """Debug endpoint - lists all available routes"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods) if hasattr(route, 'methods') else [],
                "name": getattr(route, 'name', 'N/A')
            })
   
    return {
        "routes": routes,
        "total_routes": len(routes),
        "heart_disease_model": "available" if model_data else "unavailable",
        "mcp_available": MCP_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

# ===== HEART DISEASE ENDPOINTS =====

@app.post("/predict", response_model=HeartDiseaseResponse)
async def predict_heart_disease_risk(request: HeartDiseaseRequest):
    """
    Predict heart disease risk using trained AdaBoost model.
    
    This endpoint takes patient information and returns a heart disease risk prediction
    with probability, risk level, and confidence score.
    """
    if not model_data:
        raise HTTPException(
            status_code=503, 
            detail="Heart disease model not available. Please ensure model file is present."
        )
    
    try:
        # Extract request data
        age = request.age
        gender = request.gender
        diabetes = request.diabetes
        high_bp = request.high_bp
        smoking = request.smoking
        
        logger.info(f"Heart disease prediction request: Age={age}, Gender={gender}, Diabetes={diabetes}, High_BP={high_bp}, Smoking={smoking}")
        
        # Create input DataFrame
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Diabetes': [diabetes],
            'High_BP': [high_bp],
            'Smoking': [smoking]
        })
        
        # Scale only the Age column
        input_scaled = input_data.copy()
        input_scaled['Age'] = model_data['scaler'].transform(input_data[['Age']])
        
        # Make prediction
        probability = model_data['model'].predict_proba(input_scaled)[0][1]
        risk_level = categorize_risk(probability)
        confidence = probability if probability > 0.5 else 1 - probability
        
        prediction_text = f"Heart Disease Risk: {probability:.1%} ({risk_level} Risk)"
        
        logger.info(f"✅ Heart disease prediction completed: {prediction_text} | Confidence: {confidence:.1%}")
        
        return HeartDiseaseResponse(
            prediction=prediction_text,
            probability=round(probability, 4),
            risk_level=risk_level,
            confidence=round(confidence, 4),
            input_data={
                "age": age,
                "gender": "Male" if gender == 1 else "Female",
                "diabetes": "Yes" if diabetes == 1 else "No",
                "high_bp": "Yes" if high_bp == 1 else "No",
                "smoking": "Yes" if smoking == 1 else "No"
            }
        )
        
    except Exception as e:
        logger.error(f"Heart disease prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Heart disease prediction failed: {str(e)}"
        )

@app.post("/predict-simple")
async def predict_simple(
    age: int,
    gender: int,
    diabetes: int,
    high_bp: int,
    smoking: int
):
    """
    Simple heart disease prediction endpoint with query parameters.
    """
    try:
        request = HeartDiseaseRequest(
            age=age,
            gender=gender,
            diabetes=diabetes,
            high_bp=high_bp,
            smoking=smoking
        )
        return await predict_heart_disease_risk(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/heart-disease/model-info")
async def get_heart_disease_model_info():
    """Get information about the loaded heart disease model"""
    if not model_data:
        raise HTTPException(status_code=503, detail="Heart disease model not loaded")
    
    return {
        "model_type": type(model_data['model']).__name__,
        "feature_names": model_data['feature_names'],
        "scaler_type": type(model_data['scaler']).__name__,
        "risk_categories": {
            "low": "< 30% probability",
            "medium": "30-70% probability", 
            "high": "> 70% probability"
        },
        "model_loaded": True
    }

# ===== MCP INTEGRATION =====

if MCP_AVAILABLE:
    print("🔧 Setting up MCP integration...")
   
    try:
        # SSE setup for MCP
        sse = SseServerTransport("/messages")
        app.router.routes.append(Mount("/messages", app=sse.handle_post_message))
       
        @app.get("/messages", tags=["MCP"], include_in_schema=True)
        def messages_docs(session_id: str):
            """MCP messages endpoint documentation"""
            return {"message": "MCP SSE endpoint", "session_id": session_id}
       
        @app.get("/sse", tags=["MCP"])
        async def handle_sse(request: Request):
            """Handle Server-Sent Events for MCP"""
            async with sse.connect_sse(request.scope, request.receive, request._send) as (r, w):
                await mcp._mcp_server.run(r, w, mcp._mcp_server.create_initialization_options())
       
        print("✅ SSE setup completed")
    except Exception as e:
        print(f"⚠️ Warning: Could not set up SSE for MCP: {e}")
   
    # Include the main router with healthcare endpoints
    try:
        app.include_router(router)
        print("✅ Healthcare router included successfully")
        print("📡 Available healthcare endpoints: /medical/submit, /pharmacy/submit, /mcid/search, /token, /all")
    except Exception as e:
        print(f"❌ Error including healthcare router: {e}")
        MCP_AVAILABLE = False

# ===== HEALTHCARE API ENDPOINTS (if MCP not available) =====

if not MCP_AVAILABLE:
    print("⚠️ MCP not available - creating fallback healthcare endpoints...")
   
    @app.post("/medical/submit")
    async def fallback_medical():
        return {
            "status_code": 503,
            "error": "MCP server not available - check mcpserver.py and router.py",
            "service": "medical",
            "fallback": True
        }
   
    @app.post("/pharmacy/submit")
    async def fallback_pharmacy():
        return {
            "status_code": 503,
            "error": "MCP server not available - check mcpserver.py and router.py",
            "service": "pharmacy",
            "fallback": True
        }
   
    @app.post("/mcid/search")
    async def fallback_mcid():
        return {
            "status_code": 503,
            "error": "MCP server not available - check mcpserver.py and router.py",
            "service": "mcid",
            "fallback": True
        }
   
    @app.post("/token")
    async def fallback_token():
        return {
            "status_code": 503,
            "error": "MCP server not available - check mcpserver.py and router.py",
            "service": "token",
            "fallback": True
        }
   
    @app.post("/all")
    async def fallback_all():
        return {
            "status_code": 503,
            "error": "MCP server not available - check mcpserver.py and router.py",
            "service": "all",
            "fallback": True
        }
   
    print("⚠️ Fallback healthcare endpoints created")

# ===== COMBINED ENDPOINTS =====

@app.post("/comprehensive-health-assessment")
async def comprehensive_health_assessment(
    patient: HealthcarePatientRequest,
    heart_disease_data: HeartDiseaseRequest
):
    """
    Comprehensive health assessment combining heart disease prediction and healthcare data.
    """
    results = {
        "patient_info": patient.dict(),
        "timestamp": datetime.now().isoformat()
    }
    
    # Heart disease prediction
    if model_data:
        try:
            heart_disease_result = await predict_heart_disease_risk(heart_disease_data)
            results["heart_disease_prediction"] = heart_disease_result.dict()
        except Exception as e:
            results["heart_disease_prediction"] = {"error": str(e)}
    else:
        results["heart_disease_prediction"] = {"error": "Heart disease model not available"}
    
    # Healthcare data (if MCP available)
    if MCP_AVAILABLE:
        results["healthcare_data"] = {
            "note": "Healthcare APIs available via MCP tools",
            "endpoints": ["/medical/submit", "/pharmacy/submit", "/mcid/search"]
        }
    else:
        results["healthcare_data"] = {"error": "Healthcare APIs not available - MCP not loaded"}
    
    return results

# ===== STARTUP CONFIGURATION =====

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*60)
    print("🚀 Combined Healthcare & Heart Disease API Starting Up")
    print("="*60)
    print(f"🏥 Heart Disease Model: {'Available' if model_data else 'Not Available'}")
    print(f"📡 MCP Available: {MCP_AVAILABLE}")
    print(f"🏥 Health Check: http://localhost:8080/health")
    print(f"🐛 Debug Routes: http://localhost:8080/debug/routes")
    print(f"📍 Root Endpoint: http://localhost:8080/")
   
    print("\n✅ Heart Disease Endpoints:")
    print("   • POST /predict")
    print("   • POST /predict-simple")
    print("   • GET /heart-disease/model-info")
    
    if MCP_AVAILABLE:
        print("\n✅ Healthcare endpoints available:")
        print("   • POST /medical/submit")
        print("   • POST /pharmacy/submit")
        print("   • POST /mcid/search")
        print("   • POST /token")
        print("   • POST /all")
    else:
        print("\n⚠️ Healthcare endpoints: fallback mode only")
    
    print("\n🔗 Combined Endpoints:")
    print("   • POST /comprehensive-health-assessment")
    
    print("="*60)

# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    print("🔧 Starting Combined Healthcare & Heart Disease API server...")
    print("📍 Server will be available at: http://localhost:8080")
    print("🏥 Test health endpoint: http://localhost:8080/health")
    print("📚 API Documentation: http://localhost:8080/docs")
   
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8080,
            log_level="info",
            reload=False  # Set to True for development
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        print("💡 Try a different port: uvicorn app:app --port 8081")
