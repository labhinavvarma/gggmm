from fastapi import APIRouter, HTTPException
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/cortex",
    tags=["Healthcare FastMCP API"]
)

# Alias for app.py compatibility
route = router

@route.get("/health/status")
async def get_health_status():
    """Get basic health system status for FastMCP server"""
    return {
        "status": "operational",
        "service": "Health Details FastMCP Server",
        "server_type": "FastMCP",
        "api_integration": "Milliman/Anthem APIs",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mcp_server": "Health Details MCP Server",
        "version": "1.0.0",
        "transport": "SSE",
        "decorators": ["@mcp.tool", "@mcp.prompt"]
    }

@route.get("/health/tools")
async def get_available_tools():
    """Get list of available FastMCP tools with detailed information"""
    return {
        "framework": "FastMCP",
        "tools": [
            {
                "name": "all",
                "description": "Complete system overview with real API status",
                "type": "system",
                "decorator": "@mcp.tool()",
                "parameters": "None",
                "returns": "System status and API health"
            },
            {
                "name": "token", 
                "description": "Get real authentication token from Milliman API",
                "type": "authentication",
                "decorator": "@mcp.tool()",
                "parameters": "None",
                "returns": "Authentication token and metadata"
            },
            {
                "name": "medical_submit",
                "description": "Submit medical claim to real Milliman healthcare API", 
                "type": "medical",
                "decorator": "@mcp.tool()",
                "parameters": "Patient data (first_name, last_name, ssn, date_of_birth, gender, zip_code)",
                "returns": "Medical claim submission result"
            },
            {
                "name": "pharmacy_submit",
                "description": "Submit pharmacy claim to real Milliman healthcare API",
                "type": "pharmacy",
                "decorator": "@mcp.tool()",
                "parameters": "Patient data (first_name, last_name, ssn, date_of_birth, gender, zip_code)",
                "returns": "Pharmacy claim submission result"
            },
            {
                "name": "mcid_search",
                "description": "Search for member using MCID (Member Consumer ID) service",
                "type": "search",
                "decorator": "@mcp.tool()",
                "parameters": "Patient data (first_name, last_name, ssn, date_of_birth, gender, zip_code)",
                "returns": "Member search results from MCID service"
            },
            {
                "name": "get_all_healthcare_data",
                "description": "Get comprehensive healthcare data from all Milliman APIs",
                "type": "comprehensive",
                "decorator": "@mcp.tool()",
                "parameters": "Patient data (first_name, last_name, ssn, date_of_birth, gender, zip_code)",
                "returns": "Combined results from all healthcare APIs"
            }
        ],
        "prompts": [
            {
                "name": "health-details",
                "description": "Health management system with real API integration",
                "decorator": "@mcp.prompt(name='health-details', description='...')",
                "arguments": ["query (required)"],
                "returns": "Comprehensive health management prompt"
            },
            {
                "name": "healthcare-summary", 
                "description": "Healthcare API summary with real data",
                "decorator": "@mcp.prompt(name='healthcare-summary', description='...')",
                "arguments": ["query (required)"],
                "returns": "Healthcare data summary prompt"
            }
        ],
        "integration": "FastMCP + SSE transport compatible",
        "total_tools": 6,
        "total_prompts": 2,
        "mcp_protocol_version": "2024-11-05"
    }

@route.get("/test")
async def test_endpoint():
    """Test endpoint to verify FastMCP router functionality"""
    return {
        "message": "Health Details FastMCP Server is running",
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "router": "active",
        "mcp_integration": "FastMCP enabled",
        "transport": "SSE",
        "decorators": {
            "tools": "@mcp.tool()",
            "prompts": "@mcp.prompt(name='...', description='...')"
        }
    }

@route.get("/info")
async def get_server_info():
    """Get detailed FastMCP server information"""
    return {
        "server_name": "Healthcare FastMCP Server",
        "mcp_server_id": "Health Details MCP Server", 
        "framework": "FastMCP",
        "description": "Model Context Protocol server for healthcare data management using FastMCP framework",
        "api_integrations": [
            "Milliman Token Service",
            "Anthem Medical API",
            "Anthem Pharmacy API", 
            "MCID Member Search Service"
        ],
        "transport": "Server-Sent Events (SSE)",
        "capabilities": {
            "tools": True,
            "prompts": True,
            "resources": False,
            "sampling": False,
            "automatic_registration": True,
            "type_validation": True,
            "async_support": True
        },
        "decorators": {
            "tool_decorator": "@mcp.tool()",
            "prompt_decorator": "@mcp.prompt(name='...', description='...')",
            "features": [
                "Automatic function registration",
                "Type hints validation", 
                "Async function support",
                "Parameter validation",
                "Return type handling"
            ]
        },
        "endpoints": {
            "sse_connection": "/sse",
            "message_handling": "/messages",
            "health_check": "/health",
            "router_base": "/api/cortex"
        },
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@route.get("/patient/schema")
async def get_patient_schema():
    """Get the patient data schema required for healthcare tools"""
    return {
        "patient_data_schema": {
            "type": "object",
            "properties": {
                "first_name": {
                    "type": "string",
                    "description": "Patient's first name",
                    "example": "John",
                    "required": True
                },
                "last_name": {
                    "type": "string", 
                    "description": "Patient's last name",
                    "example": "Doe",
                    "required": True
                },
                "ssn": {
                    "type": "string",
                    "description": "Social Security Number",
                    "pattern": "^\\d{3}-\\d{2}-\\d{4}$",
                    "example": "123-45-6789",
                    "required": True
                },
                "date_of_birth": {
                    "type": "string",
                    "description": "Date of birth in YYYY-MM-DD format",
                    "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                    "example": "1990-01-01",
                    "required": True
                },
                "gender": {
                    "type": "string",
                    "description": "Gender",
                    "enum": ["M", "F"],
                    "example": "M",
                    "required": True
                },
                "zip_code": {
                    "type": "string",
                    "description": "ZIP code",
                    "pattern": "^\\d{5}(-\\d{4})?$",
                    "example": "12345",
                    "required": True
                }
            },
            "required": ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"]
        },
        "usage": "This schema is required for all healthcare tools that process patient data",
        "tools_using_schema": [
            "medical_submit",
            "pharmacy_submit", 
            "mcid_search",
            "get_all_healthcare_data"
        ],
        "validation": "FastMCP automatically validates parameters against function signatures"
    }

@route.get("/api/endpoints")
async def list_api_endpoints():
    """List all available API endpoints"""
    return {
        "external_healthcare_apis": {
            "token_service": {
                "url": "https://securefed.antheminc.com/as/token.oauth2",
                "description": "Milliman authentication service",
                "method": "POST"
            },
            "medical_api": {
                "url": "https://hix-clm-internaltesting-prod.anthem.com/medical",
                "description": "Anthem medical claims API",
                "method": "POST"
            },
            "pharmacy_api": {
                "url": "https://hix-clm-internaltesting-prod.anthem.com/pharmacy",
                "description": "Anthem pharmacy claims API", 
                "method": "POST"
            },
            "mcid_service": {
                "url": "https://mcid-app-prod.anthem.com:443/MCIDExternalService/V2/extSearchService/json",
                "description": "MCID member search service",
                "method": "POST"
            }
        },
        "server_endpoints": {
            "health_status": "/api/cortex/health/status",
            "available_tools": "/api/cortex/health/tools",
            "test": "/api/cortex/test",
            "server_info": "/api/cortex/info",
            "patient_schema": "/api/cortex/patient/schema",
            "api_endpoints": "/api/cortex/api/endpoints",
            "fastmcp_demo": "/api/cortex/fastmcp/demo"
        },
        "mcp_endpoints": {
            "sse_connection": "/sse",
            "message_posting": "/messages",
            "capabilities": "/mcp/capabilities",
            "tools": "/mcp/tools",
            "prompts": "/mcp/prompts"
        },
        "fastmcp_endpoints": {
            "framework_info": "/fastmcp/info"
        }
    }

@route.get("/fastmcp/demo")
async def fastmcp_demo():
    """Demonstrate FastMCP features and capabilities"""
    return {
        "fastmcp_demonstration": {
            "framework": "FastMCP",
            "tagline": "The fastest way to build MCP servers",
            "key_features": [
                "Decorator-based tool registration (@mcp.tool)",
                "Decorator-based prompt registration (@mcp.prompt)",
                "Automatic type validation from function signatures",
                "Built-in async support",
                "SSE transport compatibility",
                "Zero-configuration setup"
            ],
            "example_tool_definition": {
                "code": """
@mcp.tool()
async def medical_submit(
    first_name: str, last_name: str, ssn: str,
    date_of_birth: str, gender: str, zip_code: str
) -> Dict[str, Any]:
    '''Submit medical claim to real Milliman healthcare API'''
    # Tool implementation here
    return result
                """,
                "description": "Tools are automatically registered and validated"
            },
            "example_prompt_definition": {
                "code": """
@mcp.prompt(name="health-details", description="Health management system")
async def health_details_prompt(query: str) -> List[Message]:
    '''Generate a comprehensive health management prompt'''
    return [{"role": "user", "content": f"Query: {query}"}]
                """,
                "description": "Prompts are automatically registered with metadata"
            },
            "benefits": [
                "Less boilerplate code",
                "Type safety built-in",
                "Easy to maintain",
                "Fast development cycle",
                "MCP protocol compliance",
                "Production ready"
            ]
        },
        "implementation_stats": {
            "total_tools": 6,
            "total_prompts": 2,
            "lines_of_code_saved": "~200+ compared to manual MCP setup",
            "setup_time": "< 5 minutes",
            "maintenance_overhead": "minimal"
        }
    }

# Error handler example
@route.get("/error-test")
async def error_test():
    """Test endpoint that raises an error for testing error handling"""
    raise HTTPException(
        status_code=418, 
        detail="I'm a teapot - this is a FastMCP test error for demonstration purposes"
    )
