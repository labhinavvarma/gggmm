from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import pickle
import os
import logging
import sys
import uvicorn
from typing import Optional

# Set up logging with more detail
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("🚀 Starting Heart Disease FastAPI Server Setup...")

# Initialize FastAPI
app = FastAPI(
    title="Heart Disease Prediction API",
    description="AdaBoost Heart Disease Risk Prediction API with 5 features",
    version="1.0.0"
)

# Request model for heart disease prediction
class HeartDiseaseRequest(BaseModel):
    age: int
    gender: int  # 0=Female, 1=Male
    diabetes: int  # 0=No, 1=Yes
    high_bp: int  # 0=No, 1=Yes
    smoking: int  # 0=No, 1=Yes

# Response model for heart disease prediction
class HeartDiseaseResponse(BaseModel):
    risk_probability: float
    risk_percentage: str
    risk_level: str
    confidence: float
    binary_prediction: int
    message: str
    model_info: dict

# Load model globally (only once at startup)
MODEL_PACKAGE_PATH = "heart_disease_model_package.pkl"
model_data = None

print("📁 Checking model files...")
print(f"   Model package path: {MODEL_PACKAGE_PATH} - {'✅ Exists' if os.path.exists(MODEL_PACKAGE_PATH) else '❌ Missing'}")

try:
    # Load the combined model package
    if os.path.exists(MODEL_PACKAGE_PATH):
        print("🔄 Loading model package...")
        model_package = joblib.load(MODEL_PACKAGE_PATH)
        model_data = {
            'model': model_package['model'],
            'scaler': model_package['scaler'],
            'feature_names': model_package['feature_names']
        }
        print("✅ Heart disease model loaded successfully")
        print(f"   Model type: {type(model_data['model'])}")
        print(f"   Scaler type: {type(model_data['scaler'])}")
        print(f"   Features: {len(model_data['feature_names'])} features")
        print(f"   Feature names: {model_data['feature_names']}")
    else:
        print(f"❌ Model package not found: {MODEL_PACKAGE_PATH}")
        raise RuntimeError("Model package not found. Train and save your model first.")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    logger.error(f"Failed to load model: {str(e)}")
    sys.exit(1)

def categorize_risk(prob: float) -> str:
    """Categorize risk based on probability"""
    if prob < 0.3:
        return "LOW"
    elif prob < 0.7:
        return "MODERATE"
    else:
        return "HIGH"

def get_risk_message(risk_level: str, probability: float) -> str:
    """Get risk message based on level"""
    messages = {
        "LOW": f"Low risk of heart disease ({probability:.1%}). Continue healthy lifestyle practices.",
        "MODERATE": f"Moderate risk of heart disease ({probability:.1%}). Regular monitoring recommended.",
        "HIGH": f"High risk of heart disease ({probability:.1%}). Immediate medical consultation recommended."
    }
    return messages.get(risk_level, f"Risk assessment: {probability:.1%}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "model": "AdaBoost Classifier",
        "features": model_data['feature_names'] if model_data else [],
        "endpoints": {
            "prediction": "/predict",
            "health": "/health",
            "model_info": "/model-info"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_data is not None,
        "model_type": type(model_data['model']).__name__ if model_data else None
    }

# Model information endpoint
@app.get("/model-info")
async def model_info():
    """Get model information"""
    if not model_data:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": type(model_data['model']).__name__,
        "scaler_type": type(model_data['scaler']).__name__,
        "features": model_data['feature_names'],
        "feature_count": len(model_data['feature_names']),
        "model_parameters": {
            "n_estimators": getattr(model_data['model'], 'n_estimators', 'N/A'),
            "learning_rate": getattr(model_data['model'], 'learning_rate', 'N/A'),
            "algorithm": getattr(model_data['model'], 'algorithm', 'N/A')
        }
    }

# Main prediction endpoint - THIS IS THE NEW ENDPOINT
@app.post("/predict", response_model=HeartDiseaseResponse)
async def predict_heart_disease(request: HeartDiseaseRequest):
    """
    Predict heart disease risk using AdaBoost model with 5 features
    
    Features:
    - age: Age in years (integer)
    - gender: Gender (0=Female, 1=Male)
    - diabetes: Diabetes status (0=No, 1=Yes)
    - high_bp: High blood pressure (0=No, 1=Yes)
    - smoking: Smoking status (0=No, 1=Yes)
    
    Returns:
    - risk_probability: Probability of heart disease (0.0 to 1.0)
    - risk_level: Risk category (LOW/MODERATE/HIGH)
    - confidence: Model confidence
    - binary_prediction: Binary prediction (0=No Risk, 1=Risk)
    - message: Human-readable risk assessment
    """
    try:
        logger.info(f"🎯 Received prediction request: age={request.age}, gender={request.gender}, diabetes={request.diabetes}, high_bp={request.high_bp}, smoking={request.smoking}")
        
        # Validate model availability
        if not model_data:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Input validation
        if not isinstance(request.age, int) or request.age < 0 or request.age > 120:
            raise HTTPException(status_code=400, detail="Age must be an integer between 0 and 120")
        
        binary_features = [request.gender, request.diabetes, request.high_bp, request.smoking]
        feature_names = ['Gender', 'Diabetes', 'High_BP', 'Smoking']
        
        for i, value in enumerate(binary_features):
            if not isinstance(value, int) or value not in [0, 1]:
                raise HTTPException(status_code=400, detail=f"{feature_names[i]} must be 0 or 1")

        # Create input DataFrame with the 5 features
        input_data = pd.DataFrame({
            'Age': [request.age],
            'Gender': [request.gender],
            'Diabetes': [request.diabetes],
            'High_BP': [request.high_bp],
            'Smoking': [request.smoking]
        })
        
        logger.info(f"📊 Input data created: {input_data.iloc[0].to_dict()}")
        
        # Scale only the Age column (following the original pattern)
        input_scaled = input_data.copy()
        input_scaled['Age'] = model_data['scaler'].transform(input_data[['Age']])
        
        logger.info(f"📊 Scaled Age: {input_scaled['Age'].iloc[0]}")
        
        # Make prediction
        probability = model_data['model'].predict_proba(input_scaled)[0][1]
        binary_prediction = model_data['model'].predict(input_scaled)[0]
        risk_level = categorize_risk(probability)
        
        # Calculate confidence
        confidence = probability if probability > 0.5 else 1 - probability
        
        # Create response
        response = HeartDiseaseResponse(
            risk_probability=float(probability),
            risk_percentage=f"{probability:.1%}",
            risk_level=risk_level,
            confidence=float(confidence),
            binary_prediction=int(binary_prediction),
            message=get_risk_message(risk_level, probability),
            model_info={
                "model_type": type(model_data['model']).__name__,
                "features_used": model_data['feature_names'],
                "input_features": {
                    "Age": request.age,
                    "Gender": "Male" if request.gender == 1 else "Female",
                    "Diabetes": "Yes" if request.diabetes == 1 else "No",
                    "High_BP": "Yes" if request.high_bp == 1 else "No",
                    "Smoking": "Yes" if request.smoking == 1 else "No"
                }
            }
        )
        
        logger.info(f"✅ Prediction completed: {risk_level} risk ({probability:.1%})")
        return response
        
    except HTTPException as he:
        logger.error(f"❌ HTTP error: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Legacy endpoint for compatibility with original MCP calls
@app.post("/tools/call")
async def mcp_legacy_endpoint(request: dict):
    """
    Legacy MCP endpoint for backwards compatibility
    """
    try:
        # Extract arguments from MCP format
        tool_name = request.get("name", "")
        arguments = request.get("arguments", {})
        
        if tool_name != "predict-heart-disease-risk":
            raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")
        
        # Convert to our standard format
        heart_request = HeartDiseaseRequest(
            age=arguments.get("age", 50),
            gender=arguments.get("gender", 0),
            diabetes=arguments.get("diabetes", 0),
            high_bp=arguments.get("high_bp", 0),
            smoking=arguments.get("smoking", 0)
        )
        
        # Get prediction
        prediction = await predict_heart_disease(heart_request)
        
        # Format as MCP response
        mcp_response = {
            "content": [{
                "text": f"Heart Disease Risk: {prediction.risk_percentage} ({prediction.risk_level} Risk) | Confidence: {prediction.confidence:.1%}"
            }]
        }
        
        return mcp_response
        
    except Exception as e:
        logger.error(f"❌ MCP legacy endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MCP call failed: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(requests: list[HeartDiseaseRequest]):
    """
    Batch prediction endpoint for multiple patients
    """
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 predictions per batch")
    
    results = []
    for i, request in enumerate(requests):
        try:
            prediction = await predict_heart_disease(request)
            results.append({
                "index": i,
                "success": True,
                "prediction": prediction
            })
        except Exception as e:
            results.append({
                "index": i,
                "success": False,
                "error": str(e)
            })
    
    return {"batch_results": results}

print("✅ FastAPI endpoints setup completed")

if __name__ == "__main__":
    try:
        print("🚀 Starting Heart Disease FastAPI server...")
        print("📡 Server will be available at:")
        print("   - Main API: http://localhost:8000")
        print("   - Prediction endpoint: http://localhost:8000/predict")
        print("   - Health check: http://localhost:8000/health")
        print("   - Model info: http://localhost:8000/model-info")
        print("   - API docs: http://localhost:8000/docs")
        print("   - Legacy MCP: http://localhost:8000/tools/call")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 60)
        
        # Run the server
        uvicorn.run(
            "MLHDmcpserver:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server startup failed: {str(e)}")
        logger.error(f"Server startup failed: {str(e)}")
        sys.exit(1)
