#!/usr/bin/env python3
"""
Health Details MCP Server using FastMCP
Provides health management tools and interfaces for SSE integration with FastAPI
Uses real API integrations with Milliman/Anthem endpoints
"""

import requests
import httpx
import asyncio
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
mcp = FastMCP("Health Details App")

print("🏥 Initializing Health Details MCP Server with Real API Integration...")

# ===== API CONFIGURATION =====

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

# API Endpoints
MEDICAL_URL = "https://hix-clm-internaltesting-prod.anthem.com/medical"
PHARMACY_URL = "https://hix-clm-internaltesting-prod.anthem.com/pharmacy"
MCID_URL = "https://mcid-app-prod.anthem.com:443/MCIDExternalService/V2/extSearchService/json"

# ===== API HELPER FUNCTIONS =====

def get_access_token_sync() -> str | None:
    """Get access token synchronously"""
    try:
        r = requests.post(TOKEN_URL, data=TOKEN_PAYLOAD, headers=TOKEN_HEADERS)
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception as e:
        logger.error(f"Error getting access token: {str(e)}")
        return None

async def async_get_token() -> Dict[str, Any]:
    """Get access token asynchronously"""
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(TOKEN_URL, data=TOKEN_PAYLOAD, headers=TOKEN_HEADERS)
            return {"status_code": r.status_code, "body": r.json()}
        except Exception as e:
            logger.error(f"Error getting async token: {str(e)}")
            return {"status_code": 500, "error": str(e)}

async def async_submit_request(user: UserInput, url: str) -> Dict[str, Any]:
    """Submit request to API endpoint"""
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
    
    try:
        r = requests.post(url, headers=headers, json=payload)
        if r.status_code != 200:
            return {"status_code": r.status_code, "error": r.text, "url": url}
        return {"status_code": r.status_code, "body": r.json()}
    except Exception as e:
        return {"status_code": 500, "error": str(e), "url": url}

async def async_submit_medical_request(user: UserInput):
    """Submit medical request"""
    return await async_submit_request(user, MEDICAL_URL)

async def async_submit_pharmacy_request(user: UserInput):
    """Submit pharmacy request"""
    return await async_submit_request(user, PHARMACY_URL)

async def async_mcid_search(user: UserInput) -> Dict[str, Any]:
    """Search MCID service"""
    token = await asyncio.to_thread(get_access_token_sync)
    if not token:
        return {"status_code": 500, "error": "Access token not found"}

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
            r = await client.post(MCID_URL, headers=headers, json=mcid_payload, timeout=30)
            if r.status_code == 401:
                return {"status_code": 401, "error": "Unauthorized", "response_text": r.text}
            return {"status_code": r.status_code, "body": r.json()}
        except Exception as e:
            return {"status_code": 500, "error": str(e)}

# ===== HEALTH MANAGEMENT MCP TOOLS =====

@mcp.tool()
async def all() -> Dict[str, Any]:
    """
    Complete system overview - shows all available health services and current status.
    
    Returns comprehensive system health status, service availability, and metrics.
    """
    logger.info("Generating system overview")
    
    # Test API connectivity
    token_status = await async_get_token()
    api_status = "operational" if token_status.get("status_code") == 200 else "degraded"
    
    overview = {
        "system_status": api_status,
        "timestamp": "2024-01-01T12:00:00Z",
        "api_endpoints": {
            "token_service": {
                "url": TOKEN_URL,
                "status": "active" if token_status.get("status_code") == 200 else "error",
                "last_check": "2024-01-01T12:00:00Z"
            },
            "medical_service": {
                "url": MEDICAL_URL,
                "status": "active",
                "endpoint": "/medical/submit"
            },
            "pharmacy_service": {
                "url": PHARMACY_URL,
                "status": "active",
                "endpoint": "/pharmacy/submit"
            },
            "mcid_service": {
                "url": MCID_URL,
                "status": "active",
                "endpoint": "/mcid/search"
            }
        },
        "authentication": {
            "client_id": TOKEN_PAYLOAD["client_id"],
            "grant_type": TOKEN_PAYLOAD["grant_type"],
            "status": "configured"
        },
        "metrics": {
            "active_sessions": 1,
            "requests_per_minute": 0,
            "recent_activity": "System initialized"
        }
    }
    
    logger.info("✅ System overview generated successfully")
    return overview

@mcp.tool()
async def token() -> Dict[str, Any]:
    """
    Get authentication token for Milliman healthcare APIs.
    
    Returns:
        Token response with status code and token data
    """
    logger.info("Getting authentication token")
    
    try:
        token_response = await async_get_token()
        
        if token_response.get("status_code") == 200:
            token_data = token_response.get("body", {})
            result = {
                "status": "success",
                "token": token_data.get("access_token"),
                "token_type": token_data.get("token_type", "bearer"),
                "expires_in": token_data.get("expires_in"),
                "scope": token_data.get("scope"),
                "timestamp": "2024-01-01T12:00:00Z"
            }
            logger.info("✅ Authentication token retrieved successfully")
        else:
            result = {
                "status": "error",
                "message": f"Failed to get token: {token_response.get('error', 'Unknown error')}",
                "status_code": token_response.get("status_code")
            }
            logger.error(f"❌ Failed to get authentication token: {result['message']}")
        
        return result
        
    except Exception as e:
        error_msg = f"Authentication token error: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}

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
    logger.info(f"Medical submission for patient {first_name} {last_name}")
    
    try:
        user_input = UserInput(
            first_name=first_name,
            last_name=last_name,
            ssn=ssn,
            date_of_birth=date_of_birth,
            gender=gender,
            zip_code=zip_code
        )
        
        result = await async_submit_medical_request(user_input)
        
        if result.get("status_code") == 200:
            logger.info(f"✅ Medical submission successful for {first_name} {last_name}")
        else:
            logger.error(f"❌ Medical submission failed for {first_name} {last_name}: {result.get('error')}")
        
        return result
        
    except Exception as e:
        error_msg = f"Medical submission error: {str(e)}"
        logger.error(error_msg)
        return {"status_code": 500, "error": error_msg}

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
    logger.info(f"Pharmacy submission for patient {first_name} {last_name}")
    
    try:
        user_input = UserInput(
            first_name=first_name,
            last_name=last_name,
            ssn=ssn,
            date_of_birth=date_of_birth,
            gender=gender,
            zip_code=zip_code
        )
        
        result = await async_submit_pharmacy_request(user_input)
        
        if result.get("status_code") == 200:
            logger.info(f"✅ Pharmacy submission successful for {first_name} {last_name}")
        else:
            logger.error(f"❌ Pharmacy submission failed for {first_name} {last_name}: {result.get('error')}")
        
        return result
        
    except Exception as e:
        error_msg = f"Pharmacy submission error: {str(e)}"
        logger.error(error_msg)
        return {"status_code": 500, "error": error_msg}

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
    logger.info(f"MCID search for patient {first_name} {last_name}")
    
    try:
        user_input = UserInput(
            first_name=first_name,
            last_name=last_name,
            ssn=ssn,
            date_of_birth=date_of_birth,
            gender=gender,
            zip_code=zip_code
        )
        
        result = await async_mcid_search(user_input)
        
        if result.get("status_code") == 200:
            logger.info(f"✅ MCID search successful for {first_name} {last_name}")
        else:
            logger.error(f"❌ MCID search failed for {first_name} {last_name}: {result.get('error')}")
        
        return result
        
    except Exception as e:
        error_msg = f"MCID search error: {str(e)}"
        logger.error(error_msg)
        return {"status_code": 500, "error": error_msg}

# Additional comprehensive tool
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
    logger.info(f"Getting all healthcare data for patient {first_name} {last_name}")
    
    try:
        user_input = UserInput(
            first_name=first_name,
            last_name=last_name,
            ssn=ssn,
            date_of_birth=date_of_birth,
            gender=gender,
            zip_code=zip_code
        )
        
        # Run all API calls concurrently
        token_result, medical_result, pharmacy_result, mcid_result = await asyncio.gather(
            async_get_token(),
            async_submit_medical_request(user_input),
            async_submit_pharmacy_request(user_input),
            async_mcid_search(user_input),
            return_exceptions=True
        )
        
        result = {
            "patient": {
                "first_name": first_name,
                "last_name": last_name,
                "date_of_birth": date_of_birth,
                "gender": gender,
                "zip_code": zip_code
            },
            "token_service": token_result if not isinstance(token_result, Exception) else {"error": str(token_result)},
            "medical_service": medical_result if not isinstance(medical_result, Exception) else {"error": str(medical_result)},
            "pharmacy_service": pharmacy_result if not isinstance(pharmacy_result, Exception) else {"error": str(pharmacy_result)},
            "mcid_service": mcid_result if not isinstance(mcid_result, Exception) else {"error": str(mcid_result)},
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        logger.info(f"✅ All healthcare data retrieved for {first_name} {last_name}")
        return result
        
    except Exception as e:
        error_msg = f"Get all healthcare data error: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}

# ===== MCP PROMPTS =====

@mcp.prompt(name="health-details", description="Health management system prompt to handle medical records, pharmacy prescriptions, patient data search, system monitoring, and authentication")
async def health_details_prompt(query: str) -> List[Message]:
    """Generate a comprehensive health management prompt."""
    return [{
        "role": "user", 
        "content": f"""
        You are an expert in Health Management Systems using real Milliman/Anthem APIs. You specialize in medical records management, pharmacy operations, patient data handling, and healthcare system administration.
        
        You are provided with the following health management tools that connect to real APIs:
        
        1) **all** - Complete system overview and API health status monitoring
           - Use for: system status checks, API endpoint status, authentication status
           - Returns: real API connectivity status, endpoint information, metrics
        
        2) **token** - Get real authentication token from Milliman API
           - Use for: obtaining actual API authentication tokens
           - Returns: real access token, expiration, and scope information
        
        3) **medical_submit** - Submit medical claim to real Milliman API
           - Use for: submitting medical claims to https://hix-clm-internaltesting-prod.anthem.com/medical
           - Requires: first_name, last_name, ssn, date_of_birth (YYYY-MM-DD), gender (M/F), zip_code
           - Returns: real API response from medical service
        
        4) **pharmacy_submit** - Submit pharmacy claim to real Milliman API
           - Use for: submitting pharmacy claims to https://hix-clm-internaltesting-prod.anthem.com/pharmacy
           - Requires: first_name, last_name, ssn, date_of_birth (YYYY-MM-DD), gender (M/F), zip_code
           - Returns: real API response from pharmacy service
        
        5) **mcid_search** - Search real MCID service
           - Use for: searching member data in https://mcid-app-prod.anthem.com MCID service
           - Requires: first_name, last_name, ssn, date_of_birth (YYYY-MM-DD), gender (M/F), zip_code
           - Returns: real MCID search results
        
        6) **get_all_healthcare_data** - Get comprehensive data from all APIs
           - Use for: getting complete healthcare information from all services simultaneously
           - Requires: same patient parameters as above
           - Returns: combined results from all healthcare APIs
        
        **Important Notes:**
        - All tools connect to real production/testing APIs
        - Patient data must be in correct format (date_of_birth as YYYY-MM-DD, gender as M/F)
        - API responses include real status codes and error messages
        - Always prioritize patient privacy and data security
        - Handle API errors gracefully and provide meaningful error messages
        
        You will respond with the results returned from the right tool.
        
        User Query: {query}
        """
    }]

@mcp.prompt(name="healthcare-summary", description="Summarize healthcare API data and status")
async def healthcare_summary_prompt(query: str) -> List[Message]:
    """Generate a healthcare API summary prompt."""
    return [{
        "role": "user",
        "content": f"Provide a comprehensive healthcare API summary based on: {query}. Use the 'all' tool to get current API status and the appropriate tools to get patient data."
    }]

print("✅ Health Details MCP Server setup completed with Real API Integration")
print("🔑 Authentication Configuration:")
print(f"   - Token URL: {TOKEN_URL}")
print(f"   - Client ID: {TOKEN_PAYLOAD['client_id']}")
print("🌐 API Endpoints:")
print(f"   - Medical: {MEDICAL_URL}")
print(f"   - Pharmacy: {PHARMACY_URL}")  
print(f"   - MCID: {MCID_URL}")
print("📡 Available MCP tools (using @mcp.tool decorators):")
print("   - all (System overview with real API status)")
print("   - token (Get real authentication token)")
print("   - medical_submit (Submit to real medical API)")
print("   - pharmacy_submit (Submit to real pharmacy API)")
print("   - mcid_search (Search real MCID service)")
print("   - get_all_healthcare_data (Get comprehensive data from all APIs)")
print("📝 Available MCP prompts (using @mcp.prompt decorators):")
print("   - health-details (Main health management prompt)")
print("   - healthcare-summary (API summary prompt)")

# Make FastMCP compatible with SSE by creating a standard MCP server wrapper
from mcp.server import Server
import mcp.types as types
from typing import Dict, Any, List, TypedDict, Literal, Optional

# Create a standard MCP server that wraps the FastMCP functionality
class HealthMCPServerWrapper:
    def __init__(self, fastmcp_instance):
        self.fastmcp = fastmcp_instance
        self.server = Server("health-details")
        self._setup_standard_mcp_server()
        
    def _setup_standard_mcp_server(self):
        """Setup standard MCP server with tools and prompts that delegate to FastMCP"""
        
        @self.server.list_tools()
        async def list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name="all",
                    description="Complete system overview - shows all available health services and current status with real API connectivity",
                    inputSchema={"type": "object", "properties": {}, "required": []}
                ),
                types.Tool(
                    name="token",
                    description="Get real authentication token from Milliman API endpoint",
                    inputSchema={"type": "object", "properties": {}, "required": []}
                ),
                types.Tool(
                    name="medical_submit",
                    description="Submit medical claim request to real Milliman healthcare API",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "first_name": {"type": "string", "description": "Patient's first name"},
                            "last_name": {"type": "string", "description": "Patient's last name"},
                            "ssn": {"type": "string", "description": "Social Security Number"},
                            "date_of_birth": {"type": "string", "description": "Date of birth (YYYY-MM-DD format)"},
                            "gender": {"type": "string", "description": "Gender (M/F)"},
                            "zip_code": {"type": "string", "description": "ZIP code"}
                        },
                        "required": ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"]
                    }
                ),
                types.Tool(
                    name="pharmacy_submit",
                    description="Submit pharmacy claim request to real Milliman healthcare API",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "first_name": {"type": "string", "description": "Patient's first name"},
                            "last_name": {"type": "string", "description": "Patient's last name"},
                            "ssn": {"type": "string", "description": "Social Security Number"},
                            "date_of_birth": {"type": "string", "description": "Date of birth (YYYY-MM-DD format)"},
                            "gender": {"type": "string", "description": "Gender (M/F)"},
                            "zip_code": {"type": "string", "description": "ZIP code"}
                        },
                        "required": ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"]
                    }
                ),
                types.Tool(
                    name="mcid_search",
                    description="Search for member using real MCID (Member Consumer ID) service",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "first_name": {"type": "string", "description": "Patient's first name"},
                            "last_name": {"type": "string", "description": "Patient's last name"},
                            "ssn": {"type": "string", "description": "Social Security Number"},
                            "date_of_birth": {"type": "string", "description": "Date of birth (YYYY-MM-DD format)"},
                            "gender": {"type": "string", "description": "Gender (M/F)"},
                            "zip_code": {"type": "string", "description": "ZIP code"}
                        },
                        "required": ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"]
                    }
                ),
                types.Tool(
                    name="get_all_healthcare_data",
                    description="Get comprehensive healthcare data from all real Milliman APIs",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "first_name": {"type": "string", "description": "Patient's first name"},
                            "last_name": {"type": "string", "description": "Patient's last name"},
                            "ssn": {"type": "string", "description": "Social Security Number"},
                            "date_of_birth": {"type": "string", "description": "Date of birth (YYYY-MM-DD format)"},
                            "gender": {"type": "string", "description": "Gender (M/F)"},
                            "zip_code": {"type": "string", "description": "ZIP code"}
                        },
                        "required": ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Delegate tool calls to FastMCP functions"""
            logger.info(f"SSE Tool call: {name} with args: {arguments}")
            
            try:
                # Call the corresponding tool function directly
                if name == "all":
                    result = await all()
                elif name == "token":
                    result = await token()
                elif name == "medical_submit":
                    result = await medical_submit(**arguments)
                elif name == "pharmacy_submit":
                    result = await pharmacy_submit(**arguments)
                elif name == "mcid_search":
                    result = await mcid_search(**arguments)
                elif name == "get_all_healthcare_data":
                    result = await get_all_healthcare_data(**arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                # Convert result to TextContent
                result_text = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
                
                logger.info(f"✅ SSE Tool {name} executed successfully")
                return [types.TextContent(type="text", text=result_text)]
                
            except Exception as e:
                error_msg = f"Error executing tool {name}: {str(e)}"
                logger.error(error_msg)
                return [types.TextContent(type="text", text=error_msg)]
        
        @self.server.list_prompts()
        async def list_prompts() -> List[types.Prompt]:
            return [
                types.Prompt(
                    name="health-details",
                    description="Health management system prompt with real API integration",
                    arguments=[
                        types.PromptArgument(
                            name="query",
                            description="Health-related query or request",
                            required=True
                        )
                    ]
                ),
                types.Prompt(
                    name="healthcare-summary",
                    description="Healthcare API summary with real data",
                    arguments=[
                        types.PromptArgument(
                            name="query",
                            description="Summary request",
                            required=True
                        )
                    ]
                )
            ]
        
        @self.server.get_prompt()
        async def get_prompt(name: str, arguments: Dict[str, str]) -> types.GetPromptResult:
            query = arguments.get("query", "")
            
            if name == "health-details":
                prompt_content = f"""
You are an expert in Health Management Systems using real Milliman/Anthem APIs. You have access to actual production/testing endpoints.

Available Tools (all connect to real APIs):
1) **all** - Real system status from actual API endpoints
2) **token** - Get actual authentication token from Milliman API
3) **medical_submit** - Submit to real medical API (requires patient details)
4) **pharmacy_submit** - Submit to real pharmacy API (requires patient details)
5) **mcid_search** - Search real MCID service (requires patient details)
6) **get_all_healthcare_data** - Get data from all APIs simultaneously

Patient data format: first_name, last_name, ssn, date_of_birth (YYYY-MM-DD), gender (M/F), zip_code

You will respond with real API results. Handle errors gracefully.

User Query: {query}
"""
            elif name == "healthcare-summary":
                prompt_content = f"Provide healthcare API summary for: {query}. Use real API tools to get current status and data."
            else:
                raise ValueError(f"Unknown prompt: {name}")
            
            return types.GetPromptResult(
                description=f"Health Details - {name}",
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(type="text", text=prompt_content)
                    )
                ]
            )

# Create the wrapper instance
mcp_wrapper = HealthMCPServerWrapper(mcp)

# Replace the mcp object with one that has the _mcp_server attribute for SSE compatibility
class MCPInstance:
    def __init__(self, fastmcp_instance, wrapper):
        self.fastmcp = fastmcp_instance
        self.wrapper = wrapper
        self._mcp_server = wrapper.server  # This is what app.py expects!
        
    def run(self):
        """Run FastMCP standalone"""
        return self.fastmcp.run()

# Create the final mcp instance that app.py will import
mcp = MCPInstance(mcp, mcp_wrapper)

print(f"🔗 SSE Integration: mcp._mcp_server available for app.py")

# Export for app.py
__all__ = ["mcp"]

if __name__ == "__main__":
    try:
        print("🚀 Starting Health Details MCP Server (FastMCP mode)...")
        print("⏹️  Press Ctrl+C to stop the server")
        print("ℹ️  For SSE integration, run: python app.py")
        print("-" * 60)
        
        # Run the FastMCP server in standalone mode
        mcp.run()
        
    except KeyboardInterrupt:
        print("\n👋 Health Details MCP Server stopped by user")
    except Exception as e:
        print(f"❌ Health Details MCP Server startup failed: {str(e)}")
        logger.error(f"Health Details MCP Server startup failed: {str(e)}")
        sys.exit(1)
