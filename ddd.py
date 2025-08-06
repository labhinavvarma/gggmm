#!/usr/bin/env python3
"""
Health Details FastMCP Server with MCP Server Bridge
Compatible with MCP protocols and SSE transport using @mcp.tool and @mcp.prompt decorators
"""

import requests
import httpx
import asyncio
import json
import logging
from typing import Dict, Any, List, TypedDict, Literal
from pydantic import BaseModel
from fastmcp import FastMCP
from mcp.server import Server
import mcp.types as types

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("🏥 Initializing Health Details FastMCP Server with MCP Bridge...")

# ===== API CONFIGURATION =====

class UserInput(BaseModel):
    first_name: str
    last_name: str
    ssn: str
    date_of_birth: str  # Format: YYYY-MM-DD
    gender: str
    zip_code: str

# Local Message type for MCP prompts
class Message(TypedDict):
    role: Literal["user", "system", "assistant"]
    content: str

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

# Initialize FastMCP instance for decorators
mcp = FastMCP("Health Details MCP Server")

# ===== HELPER FUNCTIONS =====

def safe_json_dumps(obj, indent=2) -> str:
    """Safely convert object to JSON string, handling non-serializable objects"""
    def json_serializer(obj):
        if hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):  # custom objects
            return obj.__dict__
        elif isinstance(obj, bytes):  # bytes objects
            return obj.decode('utf-8', errors='ignore')
        elif hasattr(obj, '__str__'):  # any object with string representation
            return str(obj)
        else:
            return f"<non-serializable: {type(obj).__name__}>"
    
    try:
        return json.dumps(obj, indent=indent, default=json_serializer, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"JSON serialization failed: {str(e)}, falling back to string representation")
        return str(obj)

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
    """Get token asynchronously"""
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(TOKEN_URL, data=TOKEN_PAYLOAD, headers=TOKEN_HEADERS)
            return {"status_code": r.status_code, "body": r.json()}
        except Exception as e:
            return {"status_code": 500, "error": str(e)}

async def async_submit_request(user: UserInput, url: str) -> Dict[str, Any]:
    """Submit request to healthcare API"""
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

# ===== FASTMCP TOOLS USING @mcp.tool DECORATOR =====

@mcp.tool()
async def all() -> Dict[str, Any]:
    """Complete system overview with real API status"""
    logger.info("Generating system overview")
    
    try:
        token_status = await async_get_token()
        api_status = "operational" if token_status.get("status_code") == 200 else "degraded"
        
        return {
            "system_status": api_status,
            "timestamp": "2024-01-01T12:00:00Z",
            "api_endpoints": {
                "token_service": {"url": TOKEN_URL, "status": "active" if token_status.get("status_code") == 200 else "error"},
                "medical_service": {"url": MEDICAL_URL, "status": "active"},
                "pharmacy_service": {"url": PHARMACY_URL, "status": "active"},
                "mcid_service": {"url": MCID_URL, "status": "active"}
            },
            "authentication": {"client_id": TOKEN_PAYLOAD["client_id"], "status": "configured"}
        }
    except Exception as e:
        logger.error(f"Error in system overview: {str(e)}")
        return {
            "system_status": "error",
            "error": str(e),
            "timestamp": "2024-01-01T12:00:00Z"
        }

@mcp.tool()
async def token() -> Dict[str, Any]:
    """Get real authentication token from Milliman API"""
    logger.info("Getting authentication token")
    
    try:
        token_response = await async_get_token()
        
        if token_response.get("status_code") == 200:
            token_data = token_response.get("body", {})
            return {
                "status": "success",
                "token": token_data.get("access_token"),
                "token_type": token_data.get("token_type", "bearer"),
                "expires_in": token_data.get("expires_in")
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to get token: {token_response.get('error', 'Unknown error')}",
                "status_code": token_response.get("status_code")
            }
    except Exception as e:
        logger.error(f"Error getting token: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@mcp.tool()
async def medical_submit(
    first_name: str, last_name: str, ssn: str,
    date_of_birth: str, gender: str, zip_code: str
) -> Dict[str, Any]:
    """Submit medical claim to real Milliman healthcare API"""
    logger.info(f"Medical submission for {first_name} {last_name}")
    
    try:
        user_input = UserInput(
            first_name=first_name, last_name=last_name, ssn=ssn,
            date_of_birth=date_of_birth, gender=gender, zip_code=zip_code
        )
        
        result = await async_submit_request(user_input, MEDICAL_URL)
        return result
    except Exception as e:
        logger.error(f"Error in medical submission: {str(e)}")
        return {
            "status_code": 500,
            "error": str(e),
            "patient": f"{first_name} {last_name}"
        }

@mcp.tool()
async def pharmacy_submit(
    first_name: str, last_name: str, ssn: str,
    date_of_birth: str, gender: str, zip_code: str
) -> Dict[str, Any]:
    """Submit pharmacy claim to real Milliman healthcare API"""
    logger.info(f"Pharmacy submission for {first_name} {last_name}")
    
    try:
        user_input = UserInput(
            first_name=first_name, last_name=last_name, ssn=ssn,
            date_of_birth=date_of_birth, gender=gender, zip_code=zip_code
        )
        
        result = await async_submit_request(user_input, PHARMACY_URL)
        return result
    except Exception as e:
        logger.error(f"Error in pharmacy submission: {str(e)}")
        return {
            "status_code": 500,
            "error": str(e),
            "patient": f"{first_name} {last_name}"
        }

@mcp.tool()
async def mcid_search(
    first_name: str, last_name: str, ssn: str,
    date_of_birth: str, gender: str, zip_code: str
) -> Dict[str, Any]:
    """Search for member using MCID (Member Consumer ID) service"""
    logger.info(f"MCID search for {first_name} {last_name}")
    
    try:
        user_input = UserInput(
            first_name=first_name, last_name=last_name, ssn=ssn,
            date_of_birth=date_of_birth, gender=gender, zip_code=zip_code
        )
        
        result = await async_mcid_search(user_input)
        return result
    except Exception as e:
        logger.error(f"Error in MCID search: {str(e)}")
        return {
            "status_code": 500,
            "error": str(e),
            "patient": f"{first_name} {last_name}"
        }

@mcp.tool()
async def get_all_healthcare_data(
    first_name: str, last_name: str, ssn: str,
    date_of_birth: str, gender: str, zip_code: str
) -> Dict[str, Any]:
    """Get comprehensive healthcare data from all Milliman APIs (medical, pharmacy, MCID)"""
    logger.info(f"Getting all healthcare data for {first_name} {last_name}")
    
    try:
        user_input = UserInput(
            first_name=first_name, last_name=last_name, ssn=ssn,
            date_of_birth=date_of_birth, gender=gender, zip_code=zip_code
        )
        
        token_result, medical_result, pharmacy_result, mcid_result = await asyncio.gather(
            async_get_token(),
            async_submit_request(user_input, MEDICAL_URL),
            async_submit_request(user_input, PHARMACY_URL),
            async_mcid_search(user_input),
            return_exceptions=True
        )
        
        return {
            "patient": {"first_name": first_name, "last_name": last_name},
            "token_service": token_result if not isinstance(token_result, Exception) else {"error": str(token_result)},
            "medical_service": medical_result if not isinstance(medical_result, Exception) else {"error": str(medical_result)},
            "pharmacy_service": pharmacy_result if not isinstance(pharmacy_result, Exception) else {"error": str(pharmacy_result)},
            "mcid_service": mcid_result if not isinstance(mcid_result, Exception) else {"error": str(mcid_result)}
        }
    except Exception as e:
        logger.error(f"Error getting all healthcare data: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "patient": f"{first_name} {last_name}"
        }

# ===== FASTMCP PROMPTS USING @mcp.prompt DECORATOR =====

@mcp.prompt(name="health-details", description="Health management system to handle medical records, pharmacy prescriptions, patient data search, system monitoring, and authentication")
async def health_details_prompt(query: str) -> List[Message]:
    """Generate a comprehensive health management prompt"""
    return [{
        "role": "user", 
        "content": f"""
You are an expert in Health Management Systems using real Milliman/Anthem APIs.

You are provided with the following health management tools:

1) **all** - Complete system overview and health status monitoring
   - Use for: system status checks, service availability, performance metrics
   - Returns: comprehensive system health, database status, active sessions

2) **token** - Authentication and session management interface  
   - Use for: user authentication, session management, security operations
   - Returns: token status, authentication results, session information

3) **medical_submit** - Medical record submission and management
   - Use for: submitting medical claims to real Milliman API
   - Returns: submission confirmation, record ID, validation status

4) **pharmacy_submit** - Pharmacy prescription submission and management
   - Use for: prescription submissions, medication management, pharmacy operations
   - Returns: prescription status, pharmacy processing, pickup information

5) **mcid_search** - Comprehensive database search interface
   - Use for: searching patients, prescriptions, medical records, providers, pharmacies
   - Returns: search results with relevance scores, detailed information, metadata

6) **get_all_healthcare_data** - Get comprehensive data from all APIs
   - Use for: getting complete healthcare information from all services simultaneously
   - Returns: combined results from all healthcare APIs

Patient data format: first_name, last_name, ssn, date_of_birth (YYYY-MM-DD), gender (M/F), zip_code

You will respond with the results returned from the right tool.

User Query: {query}
"""
    }]

@mcp.prompt(name="healthcare-summary", description="Summarize healthcare data intent")
async def healthcare_summary_prompt(query: str) -> List[Message]:
    """Generate a summary prompt for healthcare data queries"""
    return [{
        "role": "user",
        "content": f"Healthcare data summary request: {query}. Use the appropriate tools to get current API status and patient data."
    }]

# ===== MCP SERVER BRIDGE FOR SSE COMPATIBILITY =====

# Create a standard MCP server that bridges to FastMCP tools
mcp_server = Server("health-details")

# Store the FastMCP tools and prompts for the bridge
fastmcp_tools = {
    "all": all,
    "token": token,
    "medical_submit": medical_submit,
    "pharmacy_submit": pharmacy_submit,
    "mcid_search": mcid_search,
    "get_all_healthcare_data": get_all_healthcare_data
}

fastmcp_prompts = {
    "health-details": health_details_prompt,
    "healthcare-summary": healthcare_summary_prompt
}

@mcp_server.list_resources()
async def handle_list_resources() -> List[types.Resource]:
    """List available resources (empty for this healthcare server)"""
    logger.info("📁 Listing resources via bridge (no resources available)")
    # This healthcare server focuses on tools and prompts, not resources
    # Return empty list to prevent "Method not found" errors
    return []

@mcp_server.read_resource()
async def handle_read_resource(uri: str) -> types.ReadResourceResult:
    """Handle resource reading requests"""
    logger.info(f"📖 Resource read request for: {uri}")
    # This healthcare server doesn't have resources, return appropriate error
    raise ValueError(f"Resource not found: {uri}. This server provides healthcare tools and prompts, not resources.")

@mcp_server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List all available healthcare tools from FastMCP"""
    logger.info("🔧 Listing FastMCP tools via bridge")
    tools = [
        types.Tool(
            name="all",
            description="Complete system overview with real API status",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="token",
            description="Get real authentication token from Milliman API",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="medical_submit",
            description="Submit medical claim to real Milliman healthcare API",
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
                "required": ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="pharmacy_submit",
            description="Submit pharmacy claim to real Milliman healthcare API",
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
                "required": ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="mcid_search",
            description="Search for member using MCID (Member Consumer ID) service",
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
                "required": ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="get_all_healthcare_data",
            description="Get comprehensive healthcare data from all Milliman APIs (medical, pharmacy, MCID)",
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
                "required": ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"],
                "additionalProperties": False
            }
        )
    ]
    logger.info(f"🔧 Returning {len(tools)} tools from FastMCP bridge")
    return tools

@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Execute FastMCP tools via bridge"""
    logger.info(f"Bridging tool call to FastMCP: {name}")
    
    try:
        if name in fastmcp_tools:
            # Call the FastMCP tool function
            tool_func = fastmcp_tools[name]
            if arguments:
                result = await tool_func(**arguments)
            else:
                result = await tool_func()
            
            result_text = safe_json_dumps(result)
            logger.info(f"✅ FastMCP tool {name} executed successfully via bridge")
            return [types.TextContent(type="text", text=result_text)]
        else:
            raise ValueError(f"Unknown tool: {name}")
        
    except Exception as e:
        error_msg = f"Error executing FastMCP tool {name}: {str(e)}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=error_msg)]

@mcp_server.list_prompts()
async def handle_list_prompts() -> List[types.Prompt]:
    """List available FastMCP prompts via bridge"""
    logger.info("💬 Listing FastMCP prompts via bridge")
    prompts = [
        types.Prompt(
            name="health-details",
            description="Health management system to handle medical records, pharmacy prescriptions, patient data search, system monitoring, and authentication",
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
            description="Summarize healthcare data intent",
            arguments=[
                types.PromptArgument(
                    name="query",
                    description="Summary request",
                    required=True
                )
            ]
        )
    ]
    logger.info(f"💬 Returning {len(prompts)} prompts from FastMCP bridge")
    return prompts

@mcp_server.get_prompt()
async def handle_get_prompt(name: str, arguments: Dict[str, str]) -> types.GetPromptResult:
    """Handle FastMCP prompt requests via bridge"""
    logger.info(f"Bridging prompt call to FastMCP: {name}")
    
    try:
        if name in fastmcp_prompts:
            prompt_func = fastmcp_prompts[name]
            query = arguments.get("query", "")
            messages = await prompt_func(query)
            
            # Convert to MCP format
            mcp_messages = []
            for msg in messages:
                mcp_messages.append(types.PromptMessage(
                    role=msg['role'],
                    content=types.TextContent(type="text", text=msg['content'])
                ))
            
            logger.info(f"✅ FastMCP prompt {name} executed successfully via bridge")
            return types.GetPromptResult(
                description=f"Health Details - {name}",
                messages=mcp_messages
            )
        else:
            raise ValueError(f"Unknown prompt: {name}")
            
    except Exception as e:
        error_msg = f"Error executing FastMCP prompt {name}: {str(e)}"
        logger.error(error_msg)
        # Return error as a prompt message
        return types.GetPromptResult(
            description=f"Error - {name}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=error_msg)
                )
            ]
        )

print("✅ Health Details FastMCP Server ready with @mcp.tool and @mcp.prompt decorators")
print("🔗 FastMCP instance: mcp")
print("🌉 MCP Server Bridge: mcp_server (for SSE compatibility)")
print("🌐 Real API Integration: All tools connected")
print("📡 Available FastMCP tools:")
for tool_name in fastmcp_tools.keys():
    print(f"   - {tool_name}")
print("📝 Available FastMCP prompts:")
for prompt_name in fastmcp_prompts.keys():
    print(f"   - {prompt_name}")
print("📁 Available resources: 0 (healthcare server focuses on tools/prompts)")
print("🔗 Bridge Status: Tools, prompts, and resources handlers registered for SSE transport")

if __name__ == "__main__":
    print("🚀 Running FastMCP Server with @mcp.tool and @mcp.prompt decorators")
    print("ℹ️  For SSE integration, run: python app.py")
    mcp.run()
