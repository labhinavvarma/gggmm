from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import logging
import sys
import uvicorn
from typing import Dict, List, Any, Optional
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("🚀 Starting Heart Disease MCP Server with FastAPI...")

# Initialize FastAPI with MCP-specific configuration
app = FastAPI(
    title="Heart Disease MCP Server",
    description="MCP-compliant Heart Disease Prediction Server using FastAPI",
    version="1.0.0"
)

# MCP Protocol Models
class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: str
    params: Optional[Dict[str, Any]] = None

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class MCPError(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None

# Tool Models
class ToolInfo(BaseModel):
    name: str
    description: str
    inputSchema: Dict[str, Any]

class ToolCallRequest(BaseModel):
    name: str
    arguments: Dict[str, Any]

class ToolCallResponse(BaseModel):
    content: List[Dict[str, Any]]
    isError: Optional[bool] = False

# Load model globally
MODEL_PACKAGE_PATH = "heart_disease_model_package.pkl"
model_data = None

print("📁 Checking model files...")
print(f"   Model package path: {MODEL_PACKAGE_PATH} - {'✅ Exists' if os.path.exists(MODEL_PACKAGE_PATH) else '❌ Missing'}")

try:
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
        print(f"   Features: {model_data['feature_names']}")
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
        return "Low"
    elif prob < 0.7:
        return "Medium"
    else:
        return "High"

# MCP Protocol Endpoints

@app.post("/mcp")
async def mcp_endpoint(request: MCPRequest):
    """Main MCP endpoint that handles all MCP protocol requests"""
    try:
        if request.method == "initialize":
            return MCPResponse(
                id=request.id,
                result={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "heart-disease-mcp-server",
                        "version": "1.0.0"
                    }
                }
            )
        
        elif request.method == "tools/list":
            tools = [
                ToolInfo(
                    name="predict-heart-disease-risk",
                    description="Predict heart disease risk using trained AdaBoost model with 5 features",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "age": {
                                "type": "integer",
                                "description": "Age in years"
                            },
                            "gender": {
                                "type": "integer",
                                "description": "Gender (0=Female, 1=Male)"
                            },
                            "diabetes": {
                                "type": "integer",
                                "description": "Diabetes status (0=No, 1=Yes)"
                            },
                            "high_bp": {
                                "type": "integer",
                                "description": "High blood pressure (0=No, 1=Yes)"
                            },
                            "smoking": {
                                "type": "integer",
                                "description": "Smoking status (0=No, 1=Yes)"
                            }
                        },
                        "required": ["age", "gender", "diabetes", "high_bp", "smoking"]
                    }
                )
            ]
            
            return MCPResponse(
                id=request.id,
                result={"tools": [tool.dict() for tool in tools]}
            )
        
        elif request.method == "tools/call":
            return await handle_tool_call(request)
        
        else:
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=-32601,
                    message=f"Method not found: {request.method}"
                ).dict()
            )
    
    except Exception as e:
        logger.error(f"MCP endpoint error: {str(e)}")
        return MCPResponse(
            id=request.id,
            error=MCPError(
                code=-32603,
                message="Internal error",
                data=str(e)
            ).dict()
        )

async def handle_tool_call(request: MCPRequest) -> MCPResponse:
    """Handle tool call requests"""
    try:
        params = request.params or {}
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "predict-heart-disease-risk":
            # Extract and validate arguments
            age = arguments.get("age")
            gender = arguments.get("gender")
            diabetes = arguments.get("diabetes")
            high_bp = arguments.get("high_bp")
            smoking = arguments.get("smoking")
            
            # Input validation
            if not isinstance(age, int) or age < 0 or age > 120:
                raise ValueError("Age must be an integer between 0 and 120")
            
            binary_features = [gender, diabetes, high_bp, smoking]
            feature_names = ['Gender', 'Diabetes', 'High_BP', 'Smoking']
            
            for i, value in enumerate(binary_features):
                if not isinstance(value, int) or value not in [0, 1]:
                    raise ValueError(f"{feature_names[i]} must be 0 or 1")
            
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
            
            result = f"Heart Disease Risk: {probability:.1%} ({risk_level} Risk) | Confidence: {confidence:.1%}"
            
            logger.info(f"✅ Prediction completed: {result}")
            
            return MCPResponse(
                id=request.id,
                result=ToolCallResponse(
                    content=[{"type": "text", "text": result}]
                ).dict()
            )
        
        else:
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=-32602,
                    message=f"Unknown tool: {tool_name}"
                ).dict()
            )
    
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return MCPResponse(
            id=request.id,
            error=MCPError(
                code=-32602,
                message="Invalid parameters",
                data=str(ve)
            ).dict()
        )
    except Exception as e:
        logger.error(f"Tool call error: {str(e)}")
        return MCPResponse(
            id=request.id,
            error=MCPError(
                code=-32603,
                message="Internal error",
                data=str(e)
            ).dict()
        )

# Additional FastAPI endpoints for direct API access (non-MCP)

@app.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "message": "Heart Disease MCP Server with FastAPI",
        "version": "1.0.0",
        "protocol": "MCP (Model Context Protocol)",
        "endpoints": {
            "mcp": "/mcp",
            "health": "/health",
            "tools": "/tools"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_data is not None,
        "model_type": type(model_data['model']).__name__ if model_data else None,
        "protocol": "MCP 2024-11-05"
    }

@app.get("/tools")
async def list_tools():
    """List available tools (non-MCP format)"""
    return {
        "tools": [
            {
                "name": "predict-heart-disease-risk",
                "description": "Predict heart disease risk using trained AdaBoost model",
                "parameters": {
                    "age": "Age in years (integer)",
                    "gender": "Gender (0=Female, 1=Male)",
                    "diabetes": "Diabetes status (0=No, 1=Yes)",
                    "high_bp": "High blood pressure (0=No, 1=Yes)",
                    "smoking": "Smoking status (0=No, 1=Yes)"
                }
            }
        ]
    }

# Legacy endpoint for compatibility
@app.post("/predict")
async def direct_predict(
    age: int,
    gender: int,
    diabetes: int,
    high_bp: int,
    smoking: int
):
    """Direct prediction endpoint (non-MCP)"""
    try:
        # Reuse the MCP tool logic
        mcp_request = MCPRequest(
            method="tools/call",
            params={
                "name": "predict-heart-disease-risk",
                "arguments": {
                    "age": age,
                    "gender": gender,
                    "diabetes": diabetes,
                    "high_bp": high_bp,
                    "smoking": smoking
                }
            }
        )
        
        response = await handle_tool_call(mcp_request)
        
        if response.error:
            raise HTTPException(status_code=400, detail=response.error)
        
        return response.result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

print("✅ MCP FastAPI server setup completed")

if __name__ == "__main__":
    try:
        print("🚀 Starting Heart Disease MCP Server with FastAPI...")
        print("📡 Server endpoints:")
        print("   - MCP Protocol: http://localhost:8000/mcp")
        print("   - Health Check: http://localhost:8000/health")
        print("   - Tools List: http://localhost:8000/tools")
        print("   - Direct Predict: http://localhost:8000/predict")
        print("   - API Docs: http://localhost:8000/docs")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 60)
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server startup failed: {str(e)}")
        logger.error(f"Server startup failed: {str(e)}")
        sys.exit(1)
