import requests
import httpx
import asyncio
import pandas as pd
import joblib
import os
import logging
import sys
from typing import Dict, Any, List, TypedDict, Literal, Optional
from pydantic import BaseModel
from fastmcp import FastMCP

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Local Message type for MCP prompts
class Message(TypedDict):
    role: Literal["user", "system", "assistant"]
    content: str

# Initialize MCP instance
mcp = FastMCP("Combined Healthcare & Heart Disease API")

# ===== HEART DISEASE MODEL SETUP =====

# Load heart disease model globally
MODEL_PACKAGE_PATH = "heart_disease_model_package.pkl"
model_data = None

print("📁 Checking heart disease model files...")
print(f"   Model package path: {MODEL_PACKAGE_PATH} - {'✅ Exists' if os.path.exists(MODEL_PACKAGE_PATH) else '❌ Missing'}")

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
        print(f"   Model type: {type(model_data['model'])}")
        print(f"   Features: {model_data['feature_names']}")
    else:
        print(f"⚠️ Heart disease model package not found: {MODEL_PACKAGE_PATH}")
        print("   Heart disease prediction tools will be disabled")
        model_data = None
except Exception as e:
    print(f"⚠️ Error loading heart disease model: {str(e)}")
    logger.warning(f"Heart disease model not available: {str(e)}")
    model_data = None

def categorize_risk(prob: float) -> str:
    """Categorize heart disease risk based on probability"""
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    else:
        return "High"

# ===== MILLIMAN API SETUP =====

# Helper model for incoming user data
class UserInput(BaseModel):
    first_name: str
    last_name: str
    ssn: str
    date_of_birth: str  # Format: YYYY-MM-DD
    gender: str
    zip_code: str

# Token endpoint config
TOKEN_URL = "https://securefed.antheminc.com/as/token.oauth2"
TOKEN_PAYLOAD = {
    "grant_type": "client_credentials",
    "client_id": "MILLIMAN",
    "client_secret": "qCZpW9ixf7KTQh5Ws5YmUUqcO6JRfz0GsITmFS87RHLOls8fh0pv8TcyVEVmWRQa"
}
TOKEN_HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}

def get_access_token_sync() -> str | None:
    try:
        r = requests.post(TOKEN_URL, data=TOKEN_PAYLOAD, headers=TOKEN_HEADERS)
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception:
        return None

async def async_get_token() -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(TOKEN_URL, data=TOKEN_PAYLOAD, headers=TOKEN_HEADERS)
            return {"status_code": r.status_code, "body": r.json()}
        except Exception as e:
            return {"status_code": 500, "error": str(e)}

async def async_submit_request(user: UserInput, url: str) -> Dict[str, Any]:
    token = await asyncio.to_thread(get_access_token_sync)
    if not token:
        return {"status_code": 500, "error": "Access token not found"}

    payload = {
        "requestId": "77554079",
        "firstName": user.first_name,
        "lastName": user.last_name,
        "ssn": user.ssn,
        "dateOfBirth": user.date_of_birth,
        "gender": user.gender,
        "zipCodes": [user.zip_code],
        "callerId": "Anthem-InternalTesting"
    }

    headers = {"Authorization": token, "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload)

    if r.status_code != 200:
        return {"status_code": r.status_code, "error": r.text, "url": url}

    return {"status_code": r.status_code, "body": r.json()}

async def async_submit_medical_request(user: UserInput):
    return await async_submit_request(user, "https://hix-clm-internaltesting-prod.anthem.com/medical")

async def async_submit_pharmacy_request(user: UserInput):
    return await async_submit_request(user, "https://hix-clm-internaltesting-prod.anthem.com/pharmacy")

async def async_mcid_search(user: UserInput) -> Dict[str, Any]:
    token = await asyncio.to_thread(get_access_token_sync)
    if not token:
        return {"status_code": 500, "error": "Access token not found"}

    url = "https://mcid-app-prod.anthem.com:443/MCIDExternalService/V2/extSearchService/json"
    headers = {"Content-Type": "application/json", "Apiuser": "MillimanUser", "Authorization": token}

    mcid_payload = {
        "requestID": "1",
        "processStatus": {"completed": "false", "isMemput": "false"},
        "consumer": [{
            "fname": user.first_name,
            "lname": user.last_name,
            "sex": user.gender,
            "dob": user.date_of_birth.replace("-", ""),
            "addressList": [{"type": "P", "zip": user.zip_code}],
            "id": {"ssn": user.ssn}
        }],
        "searchSetting": {"minScore": "100", "maxResult": "1"}
    }

    async with httpx.AsyncClient(verify=False) as client:
        try:
            r = await client.post(url, headers=headers, json=mcid_payload, timeout=30)
            if r.status_code == 401:
                return {"status_code": 401, "error": "Unauthorized", "response_text": r.text}
            return {"status_code": r.status_code, "body": r.json()}
        except Exception as e:
            return {"status_code": 500, "error": str(e)}

# ===== HEART DISEASE MCP TOOLS =====

@mcp.tool()
def predict_heart_disease_risk(
    age: int,
    gender: int,
    diabetes: int,
    high_bp: int,
    smoking: int
) -> str:
    """
    Predict heart disease risk using trained AdaBoost model with 5 features.
    
    Args:
        age: Age in years (integer, 0-120)
        gender: Gender (0=Female, 1=Male)
        diabetes: Diabetes status (0=No, 1=Yes)
        high_bp: High blood pressure (0=No, 1=Yes)
        smoking: Smoking status (0=No, 1=Yes)
    
    Returns:
        Heart disease risk prediction with probability and risk level
    """
    if not model_data:
        return "Error: Heart disease model not available. Please ensure model file is present."
    
    try:
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
        
        logger.info(f"✅ Heart disease prediction completed: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Heart disease prediction error: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.tool()
def get_heart_disease_model_info() -> str:
    """
    Get information about the loaded heart disease prediction model.
    
    Returns:
        Model information including type, features, and status
    """
    if not model_data:
        return "Heart disease model not available. Please ensure model file is present."
    
    return f"""Heart Disease Model Information:
- Model Type: {type(model_data['model']).__name__}
- Features: {', '.join(model_data['feature_names'])}
- Scaler Type: {type(model_data['scaler']).__name__}
- Status: Loaded and Ready
- Risk Categories: Low (<30%), Medium (30-70%), High (>70%)"""

# ===== MILLIMAN HEALTHCARE MCP TOOLS =====

@mcp.tool()
async def get_token_tool() -> Dict[str, Any]:
    """
    Get authentication token for Milliman healthcare APIs.
    
    Returns:
        Token response with status code and token data
    """
    return await async_get_token()

@mcp.tool()
async def medical_submit(
    first_name: str, last_name: str, ssn: str,
    date_of_birth: str, gender: str, zip_code: str
) -> Dict[str, Any]:
    """
    Submit medical claim request to Milliman healthcare API.
    
    Args:
        first_name: Patient's first name
        last_name: Patient's last name
        ssn: Social Security Number
        date_of_birth: Date of birth (YYYY-MM-DD format)
        gender: Gender (M/F)
        zip_code: ZIP code
    
    Returns:
        Medical claim submission response
    """
    return await async_submit_medical_request(UserInput(**locals()))

@mcp.tool()
async def pharmacy_submit(
    first_name: str, last_name: str, ssn: str,
    date_of_birth: str, gender: str, zip_code: str
) -> Dict[str, Any]:
    """
    Submit pharmacy claim request to Milliman healthcare API.
    
    Args:
        first_name: Patient's first name
        last_name: Patient's last name
        ssn: Social Security Number
        date_of_birth: Date of birth (YYYY-MM-DD format)
        gender: Gender (M/F)
        zip_code: ZIP code
    
    Returns:
        Pharmacy claim submission response
    """
    return await async_submit_pharmacy_request(UserInput(**locals()))

@mcp.tool()
async def mcid_search(
    first_name: str, last_name: str, ssn: str,
    date_of_birth: str, gender: str, zip_code: str
) -> Dict[str, Any]:
    """
    Search for member using MCID (Member Consumer ID) service.
    
    Args:
        first_name: Patient's first name
        last_name: Patient's last name
        ssn: Social Security Number
        date_of_birth: Date of birth (YYYY-MM-DD format)
        gender: Gender (M/F)
        zip_code: ZIP code
    
    Returns:
        MCID search results
    """
    return await async_mcid_search(UserInput(**locals()))

@mcp.tool()
async def get_all_healthcare_data(
    first_name: str, last_name: str, ssn: str,
    date_of_birth: str, gender: str, zip_code: str
) -> Dict[str, Any]:
    """
    Get comprehensive healthcare data from all Milliman APIs (medical, pharmacy, MCID).
    
    Args:
        first_name: Patient's first name
        last_name: Patient's last name
        ssn: Social Security Number
        date_of_birth: Date of birth (YYYY-MM-DD format)
        gender: Gender (M/F)
        zip_code: ZIP code
    
    Returns:
        Combined results from all healthcare APIs
    """
    tok, med, pharm, mcid = await asyncio.gather(
        async_get_token(),
        async_submit_medical_request(UserInput(**locals())),
        async_submit_pharmacy_request(UserInput(**locals())),
        async_mcid_search(UserInput(**locals()))
    )

    return {
        "get_token": tok,
        "medical_submit": med,
        "pharmacy_submit": pharm,
        "mcid_search": mcid
    }

# ===== MCP PROMPTS =====

@mcp.prompt(name="healthcare-summary", description="Summarize healthcare data intent")
async def healthcare_summary_prompt(query: str) -> List[Message]:
    """Generate a summary prompt for healthcare data queries."""
    return [{"role": "user", "content": f"Healthcare data summary request: {query}"}]

@mcp.prompt(name="heart-disease-analysis", description="Analyze heart disease risk factors")
async def heart_disease_analysis_prompt(
    age: int, gender: str, medical_history: str
) -> List[Message]:
    """Generate a prompt for heart disease risk analysis."""
    return [{
        "role": "user", 
        "content": f"Analyze heart disease risk for: Age {age}, Gender {gender}, Medical History: {medical_history}"
    }]

@mcp.prompt(name="combined-health-assessment", description="Combined health assessment prompt")
async def combined_health_assessment_prompt(
    patient_info: str, risk_factors: str
) -> List[Message]:
    """Generate a comprehensive health assessment prompt."""
    return [{
        "role": "user",
        "content": f"Comprehensive health assessment for patient: {patient_info}. Risk factors: {risk_factors}. Please analyze both heart disease risk and healthcare claim patterns."
    }]

print("✅ Combined MCP server setup completed")
print(f"🏥 Heart disease model: {'Available' if model_data else 'Not Available'}")
print("📡 Available MCP tools:")
print("   Heart Disease:")
print("   - predict_heart_disease_risk")
print("   - get_heart_disease_model_info")
print("   Healthcare APIs:")
print("   - get_token_tool")
print("   - medical_submit")
print("   - pharmacy_submit") 
print("   - mcid_search")
print("   - get_all_healthcare_data")

if __name__ == "__main__":
    try:
        print("🚀 Starting Combined Healthcare & Heart Disease MCP Server...")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 60)
        
        # Run the MCP server
        mcp.run()
        
    except KeyboardInterrupt:
        print("\n👋 Combined MCP Server stopped by user")
    except Exception as e:
        print(f"❌ Combined MCP Server startup failed: {str(e)}")
        logger.error(f"Combined MCP Server startup failed: {str(e)}")
        sys.exit(1)
