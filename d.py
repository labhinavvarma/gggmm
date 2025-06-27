from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import asyncio
from mcpserver import (
    async_get_token,
    async_submit_medical_request,
    async_submit_pharmacy_request,
    async_mcid_search,
    UserInput
)

# Create router
router = APIRouter()

# Pydantic models for request validation
class HealthcarePatientRequest(BaseModel):
    first_name: str = Field(..., description="Patient's first name")
    last_name: str = Field(..., description="Patient's last name")
    ssn: str = Field(..., description="Social Security Number")
    date_of_birth: str = Field(..., description="Date of birth (YYYY-MM-DD)")
    gender: str = Field(..., description="Gender (M/F)")
    zip_code: str = Field(..., description="ZIP code")

# Healthcare API Endpoints

@router.post("/token")
async def get_token_endpoint() -> Dict[str, Any]:
    """
    Get authentication token for healthcare APIs.
    
    Returns:
        Authentication token response
    """
    try:
        result = await async_get_token()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token retrieval failed: {str(e)}")

@router.post("/medical/submit")
async def submit_medical_claim(patient: HealthcarePatientRequest) -> Dict[str, Any]:
    """
    Submit medical claim to healthcare API.
    
    Args:
        patient: Patient information for medical claim
        
    Returns:
        Medical claim submission response
    """
    try:
        user_input = UserInput(**patient.dict())
        result = await async_submit_medical_request(user_input)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Medical claim submission failed: {str(e)}")

@router.post("/pharmacy/submit")
async def submit_pharmacy_claim(patient: HealthcarePatientRequest) -> Dict[str, Any]:
    """
    Submit pharmacy claim to healthcare API.
    
    Args:
        patient: Patient information for pharmacy claim
        
    Returns:
        Pharmacy claim submission response
    """
    try:
        user_input = UserInput(**patient.dict())
        result = await async_submit_pharmacy_request(user_input)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pharmacy claim submission failed: {str(e)}")

@router.post("/mcid/search")
async def search_mcid(patient: HealthcarePatientRequest) -> Dict[str, Any]:
    """
    Search for member using MCID service.
    
    Args:
        patient: Patient information for MCID search
        
    Returns:
        MCID search results
    """
    try:
        user_input = UserInput(**patient.dict())
        result = await async_mcid_search(user_input)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCID search failed: {str(e)}")

@router.post("/all")
async def get_all_healthcare_data(patient: HealthcarePatientRequest) -> Dict[str, Any]:
    """
    Get comprehensive healthcare data from all APIs (medical, pharmacy, MCID).
    
    Args:
        patient: Patient information for comprehensive data retrieval
        
    Returns:
        Combined results from all healthcare APIs
    """
    try:
        user_input = UserInput(**patient.dict())
        
        # Execute all requests concurrently
        tok, med, pharm, mcid = await asyncio.gather(
            async_get_token(),
            async_submit_medical_request(user_input),
            async_submit_pharmacy_request(user_input),
            async_mcid_search(user_input),
            return_exceptions=True
        )
        
        # Handle any exceptions in results
        def format_result(result, service_name):
            if isinstance(result, Exception):
                return {"error": f"{service_name} failed: {str(result)}"}
            return result
        
        return {
            "get_token": format_result(tok, "Token service"),
            "medical_submit": format_result(med, "Medical service"),
            "pharmacy_submit": format_result(pharm, "Pharmacy service"),
            "mcid_search": format_result(mcid, "MCID service"),
            "patient_info": patient.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive data retrieval failed: {str(e)}")

# Additional utility endpoints

@router.get("/healthcare/status")
async def healthcare_status():
    """
    Check the status of healthcare API services.
    
    Returns:
        Status information for all healthcare services
    """
    try:
        # Test token endpoint to check API availability
        token_result = await async_get_token()
        
        return {
            "status": "healthy",
            "services": {
                "token": "available" if token_result.get("status_code") == 200 else "unavailable",
                "medical": "available",
                "pharmacy": "available", 
                "mcid": "available"
            },
            "endpoints": {
                "token": "/token",
                "medical": "/medical/submit",
                "pharmacy": "/pharmacy/submit",
                "mcid": "/mcid/search",
                "all": "/all"
            },
            "token_test": token_result
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "services": {
                "token": "error",
                "medical": "unknown",
                "pharmacy": "unknown",
                "mcid": "unknown"
            }
        }

@router.get("/healthcare/info")
async def healthcare_info():
    """
    Get information about healthcare API endpoints and requirements.
    
    Returns:
        Information about available healthcare APIs
    """
    return {
        "description": "Milliman Healthcare APIs",
        "version": "1.0",
        "services": {
            "medical": {
                "endpoint": "/medical/submit",
                "description": "Submit medical claims",
                "method": "POST"
            },
            "pharmacy": {
                "endpoint": "/pharmacy/submit", 
                "description": "Submit pharmacy claims",
                "method": "POST"
            },
            "mcid": {
                "endpoint": "/mcid/search",
                "description": "Search member consumer ID",
                "method": "POST"
            },
            "token": {
                "endpoint": "/token",
                "description": "Get authentication token",
                "method": "POST"
            },
            "comprehensive": {
                "endpoint": "/all",
                "description": "Get all healthcare data",
                "method": "POST"
            }
        },
        "required_fields": [
            "first_name",
            "last_name", 
            "ssn",
            "date_of_birth (YYYY-MM-DD)",
            "gender (M/F)",
            "zip_code"
        ],
        "authentication": "OAuth2 client credentials"
    }
