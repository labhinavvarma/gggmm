#!/usr/bin/env python3
"""
Working FastMCP Server for Milliman APIs
========================================

A properly working FastMCP server that clients can connect to.
Uses @mcp.tool() and @mcp.prompt() decorators for proper MCP integration.

Usage:
    python working_fastmcp_server.py
"""

import asyncio
import json
import logging
import sys
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import requests
import httpx

# FastMCP imports
try:
    from fastmcp import FastMCP
except ImportError as e:
    print(f"❌ Missing FastMCP: {e}")
    print("📦 Install with: pip install fastmcp")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fastmcp_server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("fastmcp-server")

# Initialize FastMCP server
mcp = FastMCP("MillimanServer")

# API Configuration
TOKEN_URL = "https://securefed.antheminc.com/as/token.oauth2"
TOKEN_PAYLOAD = {
    'grant_type': 'client_credentials',
    'client_id': 'MILLIMAN',
    'client_secret': 'mWhfhufhjhifhi;fvhifhifuye7twr6w5eaesrfghjko9876543ewsaxcvbnmkloi98765resxcvbjkoiuytresxcvbnmkiuytgQa'
}
TOKEN_HEADERS = {'Content-Type': 'application/x-www-form-urlencoded'}

# Helper Functions
def get_access_token_sync():
    """Get access token synchronously"""
    try:
        logger.info("🔑 Requesting access token...")
        response = requests.post(TOKEN_URL, data=TOKEN_PAYLOAD, headers=TOKEN_HEADERS, timeout=30)
        response.raise_for_status()
        token = response.json().get("access_token")
        if token:
            logger.info("✅ Access token obtained successfully")
        return token
    except Exception as e:
        logger.error(f"❌ Failed to get access token: {e}")
        return None

async def async_get_token():
    """Get access token asynchronously"""
    async with httpx.AsyncClient() as client:
        try:
            logger.info("🔑 Requesting access token (async)...")
            response = await client.post(TOKEN_URL, data=TOKEN_PAYLOAD, headers=TOKEN_HEADERS, timeout=30.0)
            
            result = {
                'status_code': response.status_code,
                'timestamp': datetime.now().isoformat(),
                'success': response.status_code == 200,
                'operation': 'get_token'
            }
            
            if response.status_code == 200:
                token_data = response.json()
                result['body'] = token_data
                result['access_token'] = token_data.get('access_token', '')
                result['token_type'] = token_data.get('token_type', 'bearer')
                result['expires_in'] = token_data.get('expires_in', 3600)
                logger.info("✅ Access token retrieved successfully")
            else:
                result['error'] = response.text
                logger.error(f"❌ Token request failed: {response.status_code}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Exception getting token: {e}")
            return {
                'status_code': 500,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'operation': 'get_token'
            }

async def async_submit_medical_request(first_name: str, last_name: str, ssn: str, 
                                     date_of_birth: str, gender: str, zip_code: str):
    """Submit medical request to Milliman API"""
    access_token = await asyncio.to_thread(get_access_token_sync)
    
    if not access_token:
        return {
            'status_code': 500,
            'error': 'Failed to obtain access token for medical request',
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'operation': 'medical_submit'
        }
    
    # Medical API URL - update this with the actual endpoint
    medical_url = 'https://api.milliman.healthcare/medical/submit'
    
    payload = {
        "requestId": str(uuid.uuid4()),
        "firstName": first_name,
        "lastName": last_name,
        "ssn": ssn,
        "dateOfBirth": date_of_birth,
        "gender": gender,
        "zipCodes": [zip_code],
        "callerId": "Milliman-MCP-Server",
        "timestamp": datetime.now().isoformat()
    }
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    try:
        logger.info(f"🏥 Submitting medical request for {first_name} {last_name}")
        
        async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
            response = await client.post(medical_url, headers=headers, json=payload)
            
            result = {
                'status_code': response.status_code,
                'timestamp': datetime.now().isoformat(),
                'success': response.status_code == 200,
                'request_id': payload['requestId'],
                'operation': 'medical_submit',
                'patient': f"{first_name} {last_name}"
            }
            
            if response.status_code == 200:
                try:
                    result['body'] = response.json() if response.content else {}
                    logger.info(f"✅ Medical request successful for {first_name} {last_name}")
                except:
                    result['body'] = {'raw_response': response.text}
            else:
                result['error'] = response.text
                result['url'] = medical_url
                logger.error(f"❌ Medical request failed: {response.status_code} - {response.text}")
            
            return result
            
    except Exception as e:
        logger.error(f"❌ Exception in medical request: {e}")
        return {
            'status_code': 500,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'operation': 'medical_submit',
            'patient': f"{first_name} {last_name}"
        }

async def async_mcid_search(first_name: str, last_name: str, ssn: str,
                          date_of_birth: str, gender: str, zip_code: str):
    """Search MCID database"""
    access_token = await asyncio.to_thread(get_access_token_sync)
    
    if not access_token:
        return {
            'status_code': 500,
            'error': 'Failed to obtain access token for MCID search',
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'operation': 'mcid_search'
        }
    
    # MCID API URL
    mcid_url = 'https://hix-clm-internaltesting-prod.anthem.com/medical'
    
    headers = {
        'Content-Type': 'application/json',
        'Apiuser': 'MillimanUser',
        'Authorization': f'Bearer {access_token}'
    }
    
    mcid_payload = {
        "requestID": str(uuid.uuid4()),
        "processStatus": {
            "completed": "false",
            "isMemput": "false",
            "errorCode": None,
            "errorText": None
        },
        "consumer": [
            {
                "fname": first_name,
                "lname": last_name,
                "mname": None,
                "sex": gender,
                "dateOfBirth": date_of_birth.replace("-", ""),
                "addressList": [
                    {
                        "type": "P",
                        "zip": zip_code
                    }
                ],
                "id": {
                    "ssn": ssn
                }
            }
        ],
        "searchSetting": {
            "minScore": "100",
            "maxResult": "1"
        }
    }
    
    try:
        logger.info(f"🔍 Searching MCID for {first_name} {last_name}")
        
        async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
            response = await client.post(mcid_url, headers=headers, json=mcid_payload)
            
            result = {
                'status_code': response.status_code,
                'timestamp': datetime.now().isoformat(),
                'success': response.status_code == 200,
                'request_id': mcid_payload['requestID'],
                'operation': 'mcid_search',
                'patient': f"{first_name} {last_name}"
            }
            
            if response.status_code == 200:
                try:
                    result['body'] = response.json() if response.content else {}
                    logger.info(f"✅ MCID search successful for {first_name} {last_name}")
                except:
                    result['body'] = {'raw_response': response.text}
            elif response.status_code == 401:
                result['error'] = 'Unauthorized - Check API credentials and token'
                result['response_text'] = response.text
                logger.error(f"❌ MCID search unauthorized: {response.text}")
            else:
                result['error'] = response.text
                logger.error(f"❌ MCID search failed: {response.status_code}")
            
            return result
            
    except Exception as e:
        logger.error(f"❌ Exception in MCID search: {e}")
        return {
            'status_code': 500,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'operation': 'mcid_search',
            'patient': f"{first_name} {last_name}"
        }

# MCP Tool Definitions using @mcp.tool() decorator

@mcp.tool()
async def get_token() -> Dict[str, Any]:
    """
    Get access token for Milliman API authentication.
    
    This tool retrieves an OAuth2 access token that can be used to authenticate
    with Milliman healthcare APIs. The token is required for all other API operations.
    
    Returns:
        Dict containing:
        - status_code: HTTP status code
        - success: Whether the operation was successful
        - access_token: The actual token (if successful)
        - token_type: Type of token (usually 'bearer')
        - expires_in: Token expiration time in seconds
        - timestamp: When the token was obtained
    """
    logger.info("🔧 Tool called: get_token")
    return await async_get_token()

@mcp.tool()
async def medical_submit(
    first_name: str,
    last_name: str,
    ssn: str,
    date_of_birth: str,
    gender: str,
    zip_code: str
) -> Dict[str, Any]:
    """
    Submit a medical record request to Milliman healthcare APIs.
    
    This tool submits a request to retrieve medical records and health information
    for a specific patient. All patient information must be provided.
    
    Args:
        first_name: Patient's first name
        last_name: Patient's last name
        ssn: Social Security Number (9 digits)
        date_of_birth: Date of birth in YYYY-MM-DD format
        gender: Patient's gender ('M' for Male, 'F' for Female)
        zip_code: Patient's zip code (5+ digits)
    
    Returns:
        Dict containing:
        - status_code: HTTP response status
        - success: Whether the request was successful
        - body: Medical data response (if successful)
        - request_id: Unique identifier for this request
        - patient: Patient name for reference
        - timestamp: When the request was made
    """
    logger.info(f"🔧 Tool called: medical_submit for {first_name} {last_name}")
    return await async_submit_medical_request(first_name, last_name, ssn, date_of_birth, gender, zip_code)

@mcp.tool()
async def mcid_search(
    first_name: str,
    last_name: str,
    ssn: str,
    date_of_birth: str,
    gender: str,
    zip_code: str
) -> Dict[str, Any]:
    """
    Search the MCID (Member Coverage ID) database for patient information.
    
    This tool searches the MCID database to find patient coverage information
    and member identifiers. This is typically used for patient identification
    and insurance verification.
    
    Args:
        first_name: Patient's first name
        last_name: Patient's last name
        ssn: Social Security Number (9 digits)
        date_of_birth: Date of birth in YYYY-MM-DD format
        gender: Patient's gender ('M' for Male, 'F' for Female)
        zip_code: Patient's zip code (5+ digits)
    
    Returns:
        Dict containing:
        - status_code: HTTP response status
        - success: Whether the search was successful
        - body: MCID search results (if successful)
        - request_id: Unique identifier for this search
        - patient: Patient name for reference
        - timestamp: When the search was performed
    """
    logger.info(f"🔧 Tool called: mcid_search for {first_name} {last_name}")
    return await async_mcid_search(first_name, last_name, ssn, date_of_birth, gender, zip_code)

@mcp.tool()
async def get_all_data(
    first_name: str,
    last_name: str,
    ssn: str,
    date_of_birth: str,
    gender: str,
    zip_code: str
) -> Dict[str, Any]:
    """
    Get comprehensive patient data by calling all available APIs concurrently.
    
    This tool performs a complete data retrieval by simultaneously calling:
    - Token authentication
    - Medical record submission
    - MCID database search
    
    This provides a comprehensive view of all available patient information
    from multiple Milliman healthcare data sources.
    
    Args:
        first_name: Patient's first name
        last_name: Patient's last name
        ssn: Social Security Number (9 digits)
        date_of_birth: Date of birth in YYYY-MM-DD format
        gender: Patient's gender ('M' for Male, 'F' for Female)
        zip_code: Patient's zip code (5+ digits)
    
    Returns:
        Dict containing:
        - get_token: Authentication token results
        - medical_submit: Medical records data
        - mcid_search: MCID search results
        - summary: Overall operation summary
        - timestamp: When the operation was performed
        - patient: Patient name for reference
    """
    logger.info(f"🔧 Tool called: get_all_data for {first_name} {last_name}")
    
    try:
        # Run all operations concurrently
        token_task = async_get_token()
        medical_task = async_submit_medical_request(first_name, last_name, ssn, date_of_birth, gender, zip_code)
        mcid_task = async_mcid_search(first_name, last_name, ssn, date_of_birth, gender, zip_code)
        
        logger.info("🔄 Running concurrent API calls...")
        token_result, medical_result, mcid_result = await asyncio.gather(
            token_task, medical_task, mcid_task, return_exceptions=True
        )
        
        # Handle any exceptions
        if isinstance(token_result, Exception):
            token_result = {"error": str(token_result), "success": False}
        if isinstance(medical_result, Exception):
            medical_result = {"error": str(medical_result), "success": False}
        if isinstance(mcid_result, Exception):
            mcid_result = {"error": str(mcid_result), "success": False}
        
        # Create summary
        successful_operations = []
        failed_operations = []
        
        if token_result.get("success", False):
            successful_operations.append("token")
        else:
            failed_operations.append("token")
            
        if medical_result.get("success", False):
            successful_operations.append("medical")
        else:
            failed_operations.append("medical")
            
        if mcid_result.get("success", False):
            successful_operations.append("mcid")
        else:
            failed_operations.append("mcid")
        
        result = {
            "get_token": token_result,
            "medical_submit": medical_result,
            "mcid_search": mcid_result,
            "summary": {
                "total_operations": 3,
                "successful_operations": len(successful_operations),
                "failed_operations": len(failed_operations),
                "success_list": successful_operations,
                "failure_list": failed_operations,
                "overall_success": len(failed_operations) == 0
            },
            "timestamp": datetime.now().isoformat(),
            "patient": f"{first_name} {last_name}",
            "operation": "get_all_data"
        }
        
        logger.info(f"✅ Comprehensive data retrieval completed for {first_name} {last_name}")
        logger.info(f"📊 Success rate: {len(successful_operations)}/3 operations successful")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in get_all_data: {e}")
        return {
            "error": f"Failed to get comprehensive data: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "patient": f"{first_name} {last_name}",
            "operation": "get_all_data",
            "success": False
        }

# MCP Prompt Definitions using @mcp.prompt() decorator

@mcp.prompt()
async def healthcare_assistant_prompt(query: str = "") -> str:
    """
    Healthcare AI assistant prompt for Milliman API interactions.
    
    This prompt configures the AI assistant to help users interact with
    Milliman healthcare APIs through natural language commands.
    
    Args:
        query: The user's healthcare query or request
    
    Returns:
        Formatted prompt for the AI assistant
    """
    prompt = f"""You are a specialized healthcare AI assistant that helps users interact with Milliman medical APIs through natural language. You have access to the following tools:

🔑 **get_token**: Retrieve authentication tokens for API access
🏥 **medical_submit**: Submit medical record requests for patients
🔍 **mcid_search**: Search MCID database for patient coverage information  
📊 **get_all_data**: Get comprehensive patient data from all sources

**Your capabilities:**
- Process patient information and make appropriate API calls
- Extract patient details from natural language input
- Provide detailed explanations of API responses
- Help with healthcare data analysis and interpretation
- Ensure HIPAA-compliant handling of patient information

**Patient data format required:**
- First Name and Last Name
- SSN (9 digits)
- Date of Birth (YYYY-MM-DD format)
- Gender (M/F)
- Zip Code (5+ digits)

**Guidelines:**
- Always validate patient information before making API calls
- Provide clear explanations of API responses
- Maintain patient privacy and confidentiality
- Use appropriate medical terminology
- Help users understand healthcare data results

**Current user query:** {query}

Please assist the user with their healthcare data request using the available tools."""

    return prompt

@mcp.prompt()
async def patient_data_extraction_prompt(text: str) -> str:
    """
    Prompt for extracting patient information from natural language text.
    
    Args:
        text: Natural language text containing patient information
    
    Returns:
        Prompt for extracting structured patient data
    """
    prompt = f"""Extract patient information from the following text and structure it for API calls:

Text to analyze: "{text}"

Please extract the following information if available:
- First Name
- Last Name  
- SSN (Social Security Number)
- Date of Birth (convert to YYYY-MM-DD format)
- Gender (M for Male, F for Female)
- Zip Code

Format the extracted information as JSON and indicate which fields are missing or need clarification.

If the text contains a request for a specific operation (get token, medical records, MCID search, or comprehensive data), please identify that as well.

Example output format:
{{
  "patient_data": {{
    "first_name": "John",
    "last_name": "Smith", 
    "ssn": "123456789",
    "date_of_birth": "1980-01-15",
    "gender": "M",
    "zip_code": "12345"
  }},
  "requested_operation": "get_all_data",
  "missing_fields": [],
  "confidence": "high"
}}
"""
    return prompt

@mcp.prompt()
async def api_response_interpreter_prompt(api_response: str, operation: str) -> str:
    """
    Prompt for interpreting and explaining API responses to users.
    
    Args:
        api_response: The API response to interpret
        operation: The operation that was performed
    
    Returns:
        Prompt for interpreting API responses
    """
    prompt = f"""Please interpret and explain the following {operation} API response in user-friendly terms:

API Response:
{api_response}

Please provide:
1. **Status Summary**: Whether the operation was successful or failed
2. **Key Findings**: Important information from the response
3. **Data Interpretation**: What the medical/healthcare data means
4. **Next Steps**: Recommended actions or follow-up operations
5. **Technical Details**: Any error codes or technical information that might be relevant

Format your response to be helpful for healthcare professionals who need to understand the patient data and any issues that occurred.

If there are errors, please explain what they mean and suggest solutions.
If the operation was successful, highlight the key patient information that was retrieved.
"""
    return prompt

# Server startup and configuration
async def main():
    """Start the FastMCP server"""
    logger.info("🚀 Starting Milliman FastMCP Server...")
    
    try:
        # Log available tools and prompts
        logger.info("🔧 Available MCP Tools:")
        logger.info("  - get_token: Get API authentication token")
        logger.info("  - medical_submit: Submit medical record requests")
        logger.info("  - mcid_search: Search MCID database")
        logger.info("  - get_all_data: Get comprehensive patient data")
        
        logger.info("📝 Available MCP Prompts:")
        logger.info("  - healthcare_assistant_prompt: Main AI assistant configuration")
        logger.info("  - patient_data_extraction_prompt: Extract patient info from text")
        logger.info("  - api_response_interpreter_prompt: Interpret API responses")
        
        # Test token endpoint connectivity
        logger.info("🔍 Testing API connectivity...")
        test_token = get_access_token_sync()
        if test_token:
            logger.info("✅ API connectivity test successful")
        else:
            logger.warning("⚠️ API connectivity test failed - check network and credentials")
        
        # Start the FastMCP server
        logger.info("🌐 Starting FastMCP server on SSE transport...")
        await mcp.run(transport="sse", host="0.0.0.0", port=8000)
        
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Run the server
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 FastMCP Server stopped")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
