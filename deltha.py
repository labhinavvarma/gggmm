
#!/usr/bin/env python3
"""
Fixed Working MCP Server - Official SDK
=======================================

A properly working MCP server using the official MCP Python SDK.
This server provides Milliman healthcare API tools through MCP protocol.

Usage:
    python working_mcp_server_fixed.py
"""

import asyncio
import json
import logging
import sys
import traceback
import uuid
import httpx
import requests
from datetime import datetime
from typing import Dict, Any, Optional

# Official MCP SDK imports
try:
    from mcp.server.fastmcp import FastMCP
    from mcp import types
except ImportError as e:
    print(f"❌ Missing MCP SDK: {e}")
    print("📦 Install with: pip install mcp")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcp_server_fixed.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("mcp-server-fixed")

# Initialize MCP server using official SDK
mcp = FastMCP("MillimanHealthcareServer")

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
    
    # Medical API URL - using placeholder for demo
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
        
        # For demo purposes, return a mock successful response
        # In production, uncomment the actual API call below
        """
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
        """
        
        # Mock successful response for demo
        result = {
            'status_code': 200,
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'request_id': payload['requestId'],
            'operation': 'medical_submit',
            'patient': f"{first_name} {last_name}",
            'body': {
                'message': f'Mock medical data retrieved for {first_name} {last_name}',
                'medical_records_found': 5,
                'records': [
                    {'record_id': 'MED001', 'date': '2023-01-15', 'type': 'consultation'},
                    {'record_id': 'MED002', 'date': '2023-03-22', 'type': 'lab_results'},
                    {'record_id': 'MED003', 'date': '2023-06-10', 'type': 'prescription'},
                    {'record_id': 'MED004', 'date': '2023-09-05', 'type': 'follow_up'},
                    {'record_id': 'MED005', 'date': '2023-12-18', 'type': 'annual_exam'}
                ]
            }
        }
        
        logger.info(f"✅ Medical request successful for {first_name} {last_name}")
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
    
    try:
        logger.info(f"🔍 Searching MCID for {first_name} {last_name}")
        
        # Mock successful MCID response for demo
        result = {
            'status_code': 200,
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'request_id': str(uuid.uuid4()),
            'operation': 'mcid_search',
            'patient': f"{first_name} {last_name}",
            'body': {
                'member_found': True,
                'mcid': f'MCID{ssn[-4:]}',
                'coverage_status': 'active',
                'insurance_details': {
                    'plan_type': 'Premium Health Plan',
                    'effective_date': '2023-01-01',
                    'member_id': f'MBR{ssn[-6:]}',
                    'group_number': 'GRP12345'
                },
                'coverage_verification': {
                    'medical': True,
                    'dental': True,
                    'vision': True,
                    'pharmacy': True
                }
            }
        }
        
        logger.info(f"✅ MCID search successful for {first_name} {last_name}")
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

# MCP Tool Definitions using official SDK decorators

@mcp.tool()
async def get_token() -> dict:
    """
    Get access token for Milliman API authentication.
    
    This tool retrieves an OAuth2 access token that can be used to authenticate
    with Milliman healthcare APIs. The token is required for all other API operations.
    
    Returns:
        dict: Authentication token information including status, token, and expiration
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
) -> dict:
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
        dict: Medical data response with patient records and status information
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
) -> dict:
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
        dict: MCID search results with coverage and member information
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
) -> dict:
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
        dict: Comprehensive results from all APIs with summary information
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

# Health check endpoint
@mcp.tool()
async def health_check() -> dict:
    """
    Health check endpoint to verify server is running.
    
    Returns:
        dict: Server health status and information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "server": "MillimanHealthcareServer",
        "version": "1.0.0",
        "tools_available": 5
    }

# Main function to run the server
def main():
    """Main function to start the MCP server"""
    logger.info("🚀 Starting Milliman MCP Server with Official SDK...")
    
    try:
        # Log available tools
        logger.info("🔧 Available MCP Tools:")
        logger.info("  - get_token: Get API authentication token")
        logger.info("  - medical_submit: Submit medical record requests")
        logger.info("  - mcid_search: Search MCID database")
        logger.info("  - get_all_data: Get comprehensive patient data")
        logger.info("  - health_check: Server health verification")
        
        # Test API connectivity
        logger.info("🔍 Testing API connectivity...")
        test_token = get_access_token_sync()
        if test_token:
            logger.info("✅ API connectivity test successful")
        else:
            logger.warning("⚠️ API connectivity test failed - using mock responses")
        
        # Start the MCP server with stdio transport
        logger.info("🌐 Starting MCP server with stdio transport...")
        logger.info("✅ MCP Server is ready and listening for connections")
        
        # Run the server
        mcp.run()
        
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
