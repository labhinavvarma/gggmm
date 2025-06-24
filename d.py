from mcp.server.fastmcp import FastMCP, Context
from fastapi import HTTPException
import pandas as pd
import joblib
import pickle
import os
import logging
import sys

# Set up logging with more detail
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("🚀 Starting Heart Disease MCP Server Setup...")

# Initialize FastMCP
try:
    mcp = FastMCP("Heart Disease ML App")
    print("✅ FastMCP initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize FastMCP: {e}")
    sys.exit(1)

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
        return "Low"
    elif prob < 0.7:
        return "Medium"
    else:
        return "High"

print("🔧 Setting up MCP tool...")

@mcp.tool(
    name="predict-heart-disease-risk",
    description="""
    Predict heart disease risk using trained AdaBoost model with reduced features.

    Args:
        age (int): Age in years
        gender (int): Gender (0=Female, 1=Male)
        diabetes (int): Diabetes (0=No, 1=Yes)
        high_bp (int): High blood pressure (0=No, 1=Yes)
        smoking (int): Smoking status (0=No, 1=Yes)

    Returns:
        str: Heart disease risk prediction with probability and risk level
    """
)
async def predict_heart_disease_risk(
    ctx: Context, 
    age: int,
    gender: int,
    diabetes: int,
    high_bp: int,
    smoking: int
) -> str:
    try:
        print(f"🎯 Received prediction request for patient (age: {age}, gender: {gender}, diabetes: {diabetes}, bp: {high_bp}, smoking: {smoking})")
        
        # Input validation
        binary_features = [gender, diabetes, high_bp, smoking]
        
        for i, value in enumerate(binary_features):
            if not isinstance(value, int) or value not in [0, 1]:
                feature_names = ['Gender', 'Diabetes', 'High_BP', 'Smoking']
                raise ValueError(f"{feature_names[i]} must be 0 or 1")
        
        if not isinstance(age, int) or age < 0 or age > 120:
            raise ValueError("Age must be a positive integer between 0 and 120")

        # Create input DataFrame with only selected features
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Diabetes': [diabetes],
            'High_BP': [high_bp],
            'Smoking': [smoking]
        })
        
        # Scale only the Age column (following MLmcp.py pattern)
        input_scaled = input_data.copy()
        input_scaled['Age'] = model_data['scaler'].transform(input_data[['Age']])
        
        # Make prediction (following MLmcp.py pattern)
        probability = model_data['model'].predict_proba(input_scaled)[0][1]
        risk_level = categorize_risk(probability)
        
        # Calculate confidence
        confidence = probability if probability > 0.5 else 1 - probability
        
        result = f"Heart Disease Risk: {probability:.1%} ({risk_level} Risk) | Confidence: {confidence:.1%}"
        print(f"✅ Prediction completed: {result}")
        
        # Format and return result (following MLmcp.py pattern)
        return result
        
    except ValueError as ve:
        print(f"❌ Validation error: {str(ve)}")
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

print("✅ MCP tool setup completed")

if __name__ == "__main__":
    try:
        print("🚀 Starting Heart Disease MCP server...")
        print("📡 Server will be available at http://localhost:8000")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        mcp.run(transport="sse")
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server startup failed: {str(e)}")
        logger.error(f"Server startup failed: {str(e)}")
        sys.exit(1) 
