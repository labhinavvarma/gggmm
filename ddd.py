from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
import asyncio
import httpx
from datetime import datetime
import time

# Import your MCP server tools and Brave key configuration
try:
    from mcpserver import (
        calculate, test_tool, diagnostic, 
        get_weather, dfw_text2sql, dfw_search,
        brave_web_search, brave_local_search,
        set_brave_api_key  # Function to set the API key
    )
    MCP_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import MCP tools: {e}")
    MCP_TOOLS_AVAILABLE = False

route = APIRouter()

class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]

class ToolCallResponse(BaseModel):
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None

class BraveKeyRequest(BaseModel):
    api_key: str

# Mock context class for tools that need it
class MockContext:
    async def info(self, message: str):
        print(f"ℹ️ INFO: {message}")
    
    async def warning(self, message: str):
        print(f"⚠️ WARNING: {message}")
    
    async def error(self, message: str):
        print(f"❌ ERROR: {message}")

@route.post("/configure_brave_key")
async def configure_brave_key(request: BraveKeyRequest):
    """Configure Brave API key from the client"""
    try:
        if not MCP_TOOLS_AVAILABLE:
            return {"success": False, "error": "MCP tools not available"}
        
        # Set the API key in the MCP server
        set_brave_api_key(request.api_key)
        
        return {
            "success": True,
            "message": "Brave API key configured successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to configure API key: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@route.post("/tool_call", response_model=ToolCallResponse)
async def handle_tool_call(request: ToolCallRequest):
    """Handle MCP tool calls via HTTP API (Updated with Brave Search)"""
    try:
        tool_name = request.tool_name
        arguments = request.arguments
        
        print(f"🔧 Received tool call: {tool_name} with args: {arguments}")
        
        if not MCP_TOOLS_AVAILABLE:
            return ToolCallResponse(
                success=False,
                error="MCP tools not available - import failed"
            )
        
        # Create mock context for tools that need it
        ctx = MockContext()
        
        # Route to appropriate tool (now includes Brave Search)
        result = None
        
        if tool_name == "calculator":
            expression = arguments.get("expression", "")
            result = calculate(expression)
            
        elif tool_name == "test_tool":
            message = arguments.get("message", "test")
            result = await test_tool(message)
            
        elif tool_name == "diagnostic":
            test_type = arguments.get("test_type", "basic")
            result = await diagnostic(test_type)
            
        elif tool_name == "get_weather":
            place = arguments.get("place", "")
            result = await get_weather(place, ctx)
            
        elif tool_name == "DFWAnalyst":
            prompt = arguments.get("prompt", "")
            result = await dfw_text2sql(prompt, ctx)
            
        elif tool_name == "DFWSearch":
            query = arguments.get("query", "")
            result = await dfw_search(ctx, query)
            
        elif tool_name == "brave_web_search":
            query = arguments.get("query", "")
            count = arguments.get("count", 10)
            offset = arguments.get("offset", 0)
            result = await brave_web_search(query, ctx, count, offset)
            
        elif tool_name == "brave_local_search":
            query = arguments.get("query", "")
            count = arguments.get("count", 5)
            result = await brave_local_search(query, ctx, count)
            
        else:
            return ToolCallResponse(
                success=False,
                error=f"Unknown tool: {tool_name}. Available tools: calculator, test_tool, diagnostic, get_weather, DFWAnalyst, DFWSearch, brave_web_search, brave_local_search"
            )
        
        print(f"✅ Tool {tool_name} executed successfully")
        
        return ToolCallResponse(
            success=True,
            result=result
        )
        
    except Exception as e:
        print(f"❌ Tool call error: {e}")
        return ToolCallResponse(
            success=False,
            error=str(e)
        )

@route.get("/tools")
async def list_available_tools():
    """List all available MCP tools (Updated with Brave Search)"""
    if not MCP_TOOLS_AVAILABLE:
        return {"error": "MCP tools not available"}
    
    tools = {
        "calculator": {
            "description": "Evaluates basic arithmetic expressions",
            "args": {"expression": "string"}
        },
        "test_tool": {
            "description": "Simple test tool to verify tool calling works",
            "args": {"message": "string"}
        },
        "diagnostic": {
            "description": "Diagnostic tool to test MCP functionality",
            "args": {"test_type": "string"}
        },
        "get_weather": {
            "description": "Get current weather information for a location",
            "args": {"place": "string"}
        },
        "DFWAnalyst": {
            "description": "Converts text to valid SQL for HEDIS value sets and code sets",
            "args": {"prompt": "string"}
        },
        "DFWSearch": {
            "description": "Searches HEDIS measure specification documents",
            "args": {"query": "string"}
        },
        "brave_web_search": {
            "description": "Search the web using Brave Search API for fresh, unbiased results",
            "args": {"query": "string", "count": "integer (optional)", "offset": "integer (optional)"}
        },
        "brave_local_search": {
            "description": "Search for local businesses and places using Brave Search API",
            "args": {"query": "string", "count": "integer (optional)"}
        }
    }
    
    return {
        "tools": tools,
        "count": len(tools),
        "available": MCP_TOOLS_AVAILABLE,
        "brave_search": "integrated_server_side"
    }

@route.get("/health")
async def health_check():
    """Health check endpoint (Updated with Brave Search)"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mcp_tools_available": MCP_TOOLS_AVAILABLE,
        "total_tools": 8,  # Updated count
        "search_integration": "brave_search_server_side",
        "features": ["HEDIS", "Weather", "Calculator", "Brave Search", "Diagnostics"]
    }

@route.post("/test_connection")
async def test_mcp_connection():
    """Test the MCP connection by calling a simple tool (Updated)"""
    try:
        test_request = ToolCallRequest(
            tool_name="test_tool",
            arguments={"message": "connection test"}
        )
        
        result = await handle_tool_call(test_request)
        
        return {
            "connection_status": "success" if result.success else "failed",
            "test_result": result.result if result.success else result.error,
            "timestamp": datetime.now().isoformat(),
            "available_tools": [
                "calculator", "test_tool", "diagnostic", "get_weather", 
                "DFWAnalyst", "DFWSearch", "brave_web_search", "brave_local_search"
            ]
        }
        
    except Exception as e:
        return {
            "connection_status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@route.post("/test_prompt")
async def test_prompt_with_tool():
    """Test a prompt that should trigger a tool call (Updated with Brave Search)"""
    try:
        # Test different prompt patterns (updated with Brave Search)
        test_prompts = [
            "Use the calculator tool to calculate: 25 * 4 + 10",
            "Use the test_tool with message: prompt test",
            "Use the diagnostic tool with test_type: basic",
            "Use the get_weather tool for: New York",
            "Use the brave_web_search tool to search for: latest AI news",
            "Use the brave_local_search tool to find: pizza restaurants NYC"
        ]
        
        results = []
        
        for prompt in test_prompts:
            # Simple tool detection logic
            tool_call = None
            
            if "calculator" in prompt.lower():
                parts = prompt.split(":")
                if len(parts) > 1:
                    expression = parts[1].strip()
                    tool_call = ToolCallRequest(
                        tool_name="calculator",
                        arguments={"expression": expression}
                    )
            elif "test_tool" in prompt.lower():
                parts = prompt.split(":")
                if len(parts) > 1:
                    message = parts[1].strip()
                    tool_call = ToolCallRequest(
                        tool_name="test_tool", 
                        arguments={"message": message}
                    )
            elif "diagnostic" in prompt.lower():
                parts = prompt.split(":")
                if len(parts) > 1:
                    test_type = parts[1].strip()
                    tool_call = ToolCallRequest(
                        tool_name="diagnostic",
                        arguments={"test_type": test_type}
                    )
            elif "weather" in prompt.lower():
                parts = prompt.split(":")
                if len(parts) > 1:
                    place = parts[1].strip()
                    tool_call = ToolCallRequest(
                        tool_name="get_weather",
                        arguments={"place": place}
                    )
            elif "brave_web_search" in prompt.lower():
                parts = prompt.split(":")
                if len(parts) > 1:
                    query = parts[1].strip()
                    tool_call = ToolCallRequest(
                        tool_name="brave_web_search",
                        arguments={"query": query, "count": 3}
                    )
            elif "brave_local_search" in prompt.lower():
                parts = prompt.split(":")
                if len(parts) > 1:
                    query = parts[1].strip()
                    tool_call = ToolCallRequest(
                        tool_name="brave_local_search",
                        arguments={"query": query, "count": 3}
                    )
            
            if tool_call:
                result = await handle_tool_call(tool_call)
                results.append({
                    "prompt": prompt,
                    "tool_detected": tool_call.tool_name,
                    "success": result.success,
                    "result": result.result if result.success else result.error
                })
            else:
                results.append({
                    "prompt": prompt,
                    "tool_detected": None,
                    "success": False,
                    "result": "No tool detected in prompt"
                })
        
        return {
            "test_results": results,
            "timestamp": datetime.now().isoformat(),
            "brave_search": "integrated_in_server"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Weather cache endpoint for monitoring
@route.get("/weather_cache")
async def get_weather_cache_status():
    """Get weather cache status"""
    try:
        # Import weather cache from mcpserver
        from mcpserver import weather_cache, is_weather_cache_valid
        
        cache_status = {}
        for location, cache_entry in weather_cache.items():
            cache_status[location] = {
                "cached_at": datetime.fromtimestamp(cache_entry['timestamp']).isoformat(),
                "is_valid": is_weather_cache_valid(cache_entry),
                "age_seconds": time.time() - cache_entry['timestamp']
            }
        
        return {
            "cache_entries": len(cache_status),
            "cache_status": cache_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except ImportError:
        return {
            "error": "Weather cache not available",
            "timestamp": datetime.now().isoformat()
        }

# Brave Search cache endpoint
@route.get("/brave_cache")
async def get_brave_cache_status():
    """Get Brave search cache status"""
    try:
        # Import Brave cache from mcpserver
        from mcpserver import brave_search_cache, is_brave_cache_valid
        
        cache_status = {}
        for search_key, cache_entry in brave_search_cache.items():
            cache_status[search_key] = {
                "cached_at": datetime.fromtimestamp(cache_entry['timestamp']).isoformat(),
                "is_valid": is_brave_cache_valid(cache_entry),
                "age_seconds": time.time() - cache_entry['timestamp']
            }
        
        return {
            "cache_entries": len(cache_status),
            "cache_status": cache_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except ImportError:
        return {
            "error": "Brave search cache not available",
            "timestamp": datetime.now().isoformat()
        }

# System information endpoint
@route.get("/system_info")
async def get_system_info():
    """Get comprehensive system information"""
    try:
        info = {
            "server": {
                "name": "DataFlyWheel MCP Server with Brave Search",
                "version": "2.0.0",
                "timestamp": datetime.now().isoformat()
            },
            "tools": {
                "hedis": ["DFWAnalyst", "DFWSearch"],
                "search": ["brave_web_search", "brave_local_search"],
                "utility": ["calculator", "get_weather", "test_tool", "diagnostic"],
                "total_count": 8
            },
            "features": {
                "weather_caching": True,
                "hedis_integration": True,
                "brave_search": "server_integrated",
                "api_key_configuration": "client_side",
                "diagnostics": True
            },
            "endpoints": {
                "/tools": "List available tools",
                "/tool_call": "Execute tool calls",
                "/configure_brave_key": "Configure Brave API key",
                "/health": "Health check",
                "/weather_cache": "Weather cache status",
                "/brave_cache": "Brave search cache status",
                "/system_info": "System information"
            }
        }
        
        return info
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    # For testing the router directly
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="MCP Tool Router with Brave Search")
    app.include_router(route)
    
    uvicorn.run(app, host="0.0.0.0", port=8082)
