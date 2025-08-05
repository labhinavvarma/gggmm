import uvicorn
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.sse import SseServerTransport

# Import the FastMCP instance from mcpserver.py
from mcpserver import mcp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("🚀 Starting Healthcare FastMCP + FastAPI Application...")

app = FastAPI(
    title="Healthcare FastMCP Server",
    description="Model Context Protocol server for healthcare data management with Milliman/Anthem APIs using FastMCP and SSE transport",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create SSE transport for MCP communication
sse_transport = SseServerTransport("/messages")

# Mount the message handler for POST requests to /messages
app.mount("/messages", sse_transport.handle_post_message)

@app.get("/sse")
async def handle_sse_connection(request: Request):
    """
    SSE endpoint for MCP client connections
    This endpoint handles the Server-Sent Events connection for FastMCP communication
    """
    logger.info("🔗 New FastMCP SSE connection established")
    
    try:
        # Establish SSE connection with the FastMCP server
        async with sse_transport.connect_sse(
            request.scope, 
            request.receive, 
            request._send
        ) as (read_stream, write_stream):
            
            logger.info("✅ SSE streams established, starting FastMCP server")
            
            # Access the underlying MCP server from FastMCP instance
            # FastMCP exposes the server via the .server property
            underlying_server = mcp.server
            
            # Run the MCP server with the established streams
            await underlying_server.run(
                read_stream,
                write_stream,
                underlying_server.create_initialization_options(),
            )
            
    except Exception as e:
        logger.error(f"❌ SSE connection error: {str(e)}")
        raise

@app.get("/messages")
def messages_docs(session_id: str):
    """
    Messages endpoint documentation
    
    This endpoint is used for posting messages to SSE clients.
    Note: This route is for documentation purposes only.
    The actual implementation is handled by the mounted SSE transport.
    """
    return {
        "message": "This endpoint handles FastMCP message posting",
        "session_id": session_id,
        "method": "POST",
        "content_type": "application/json",
        "transport": "SSE",
        "server_type": "FastMCP"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Healthcare FastMCP Server",
        "version": "1.0.0",
        "mcp_server": "Health Details MCP Server",
        "server_type": "FastMCP",
        "transport": "SSE",
        "endpoints": {
            "sse": "/sse",
            "messages": "/messages",
            "health": "/health",
            "capabilities": "/mcp/capabilities",
            "tools": "/mcp/tools",
            "prompts": "/mcp/prompts"
        }
    }

@app.get("/mcp/capabilities")
async def mcp_capabilities():
    """Get FastMCP server capabilities and available tools"""
    try:
        # Access FastMCP tools and prompts through the server
        server = mcp.server
        
        # Get tools by calling the list_tools handler
        tools_response = await server._list_tools_handler()
        tools = tools_response if hasattr(tools_response, '__iter__') else []
        
        # Get prompts by calling the list_prompts handler  
        prompts_response = await server._list_prompts_handler()
        prompts = prompts_response if hasattr(prompts_response, '__iter__') else []
        
        return {
            "server_name": "Health Details MCP Server",
            "server_type": "FastMCP",
            "version": "1.0.0",
            "transport": "SSE",
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                } for tool in tools
            ],
            "prompts": [
                {
                    "name": prompt.name, 
                    "description": prompt.description,
                    "arguments": [
                        {
                            "name": arg.name,
                            "description": arg.description,
                            "required": arg.required
                        } for arg in prompt.arguments
                    ]
                } for prompt in prompts
            ],
            "endpoints": {
                "sse_connection": "http://localhost:8000/sse",
                "message_posting": "http://localhost:8000/messages"
            }
        }
    except Exception as e:
        logger.error(f"Error getting FastMCP capabilities: {str(e)}")
        # Fallback response with known tools
        return {
            "server_name": "Health Details MCP Server",
            "server_type": "FastMCP",
            "version": "1.0.0",
            "transport": "SSE",
            "tools": [
                {"name": "all", "description": "Complete system overview with real API status"},
                {"name": "token", "description": "Get real authentication token from Milliman API"},
                {"name": "medical_submit", "description": "Submit medical claim to real Milliman healthcare API"},
                {"name": "pharmacy_submit", "description": "Submit pharmacy claim to real Milliman healthcare API"},
                {"name": "mcid_search", "description": "Search for member using MCID service"},
                {"name": "get_all_healthcare_data", "description": "Get comprehensive healthcare data from all APIs"}
            ],
            "prompts": [
                {"name": "health-details", "description": "Health management system with real API integration"},
                {"name": "healthcare-summary", "description": "Healthcare API summary with real data"}
            ],
            "error": f"Could not retrieve full capabilities: {str(e)}",
            "endpoints": {
                "sse_connection": "http://localhost:8000/sse",
                "message_posting": "http://localhost:8000/messages"
            }
        }

@app.get("/mcp/tools")
async def list_mcp_tools():
    """List all available FastMCP tools"""
    try:
        server = mcp.server
        tools = await server._list_tools_handler()
        
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description
                } for tool in tools
            ],
            "count": len(tools),
            "server_type": "FastMCP"
        }
    except Exception as e:
        logger.error(f"Error listing FastMCP tools: {str(e)}")
        return {
            "error": "Could not list FastMCP tools",
            "message": str(e),
            "server_type": "FastMCP"
        }

@app.get("/mcp/prompts")
async def list_mcp_prompts():
    """List all available FastMCP prompts"""
    try:
        server = mcp.server
        prompts = await server._list_prompts_handler()
        
        return {
            "prompts": [
                {
                    "name": prompt.name,
                    "description": prompt.description,
                    "arguments": [arg.name for arg in prompt.arguments]
                } for prompt in prompts
            ],
            "count": len(prompts),
            "server_type": "FastMCP"
        }
    except Exception as e:
        logger.error(f"Error listing FastMCP prompts: {str(e)}")
        return {
            "error": "Could not list FastMCP prompts", 
            "message": str(e),
            "server_type": "FastMCP"
        }

@app.get("/fastmcp/info")
async def fastmcp_info():
    """Get FastMCP specific information"""
    return {
        "framework": "FastMCP",
        "description": "Fast and easy way to build MCP servers",
        "decorators_used": ["@mcp.tool", "@mcp.prompt"],
        "features": {
            "automatic_tool_registration": True,
            "automatic_prompt_registration": True,
            "type_validation": True,
            "async_support": True,
            "sse_transport": True
        },
        "healthcare_tools": [
            "all", "token", "medical_submit", 
            "pharmacy_submit", "mcid_search", "get_all_healthcare_data"
        ],
        "healthcare_prompts": [
            "health-details", "healthcare-summary"
        ],
        "api_integrations": [
            "Milliman Token Service",
            "Anthem Medical API", 
            "Anthem Pharmacy API",
            "MCID Member Search Service"
        ]
    }

# Include the router from router.py
try:
    from router import route
    app.include_router(route)
    logger.info("✅ Router included successfully")
except ImportError as e:
    logger.warning(f"⚠️  Could not import router: {e}")

if __name__ == "__main__":
    print("=" * 70)
    print("🏥 HEALTHCARE FASTMCP + FASTAPI SERVER STARTING")
    print("=" * 70)
    print("🔧 Framework: FastMCP with @mcp.tool and @mcp.prompt decorators")
    print("🚀 Transport: Server-Sent Events (SSE)")
    print("🔗 SSE Endpoint: http://localhost:8000/sse")
    print("📬 Messages Endpoint: http://localhost:8000/messages")
    print("💚 Health Check: http://localhost:8000/health")
    print("🛠️  Tools List: http://localhost:8000/mcp/tools")
    print("💬 Prompts List: http://localhost:8000/mcp/prompts")
    print("📋 Capabilities: http://localhost:8000/mcp/capabilities")
    print("⚡ FastMCP Info: http://localhost:8000/fastmcp/info")
    print("=" * 70)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
