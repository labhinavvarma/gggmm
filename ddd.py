from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
import asyncio
import httpx
from datetime import datetime
import time

# Import your MCP server tools (updated without Wikipedia and DuckDuckGo)
try:
    from mcpserver import (
        calculate, test_tool, diagnostic, 
        get_weather, dfw_text2sql, dfw_search
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

# Mock context class for tools that need it
class MockContext:
    async def info(self, message: str):
        print(f"ℹ️ INFO: {message}")
    
    async def warning(self, message: str):
        print(f"⚠️ WARNING: {message}")
    
    async def error(self, message: str):
        print(f"❌ ERROR: {message}")

@route.post("/tool_call", response_model=ToolCallResponse)
async def handle_tool_call(request: ToolCallRequest):
    """Handle MCP tool calls via HTTP API (Updated without Wikipedia/DuckDuckGo)"""
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
        
        # Route to appropriate tool (removed Wikipedia and DuckDuckGo)
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
            
        # Note: Brave Search tools are now handled client-side
        elif tool_name in ["brave_web_search", "brave_local_search"]:
            return ToolCallResponse(
                success=False,
                error=f"Tool {tool_name} is now handled client-side in Streamlit. Please use the Web Search or Local Search modes in the client."
            )
            
        else:
            return ToolCallResponse(
                success=False,
                error=f"Unknown tool: {tool_name}. Available tools: calculator, test_tool, diagnostic, get_weather, DFWAnalyst, DFWSearch"
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
    """List all available MCP tools (Updated without Wikipedia/DuckDuckGo)"""
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
        }
    }
    
    # Add note about client-side tools
    client_side_tools = {
        "brave_web_search": {
            "description": "Web search using Brave Search API (Client-side)",
            "location": "streamlit_client",
            "args": {"query": "string", "count": "integer (optional)"}
        },
        "brave_local_search": {
            "description": "Local business search using Brave Search API (Client-side)",
            "location": "streamlit_client",
            "args": {"query": "string", "count": "integer (optional)"}
        }
    }
    
    return {
        "server_tools": tools,
        "client_tools": client_side_tools,
        "server_count": len(tools),
        "client_count": len(client_side_tools),
        "total_count": len(tools) + len(client_side_tools),
        "available": MCP_TOOLS_AVAILABLE,
        "note": "Brave Search tools are integrated in the Streamlit client for optimal performance"
    }

@route.get("/health")
async def health_check():
    """Health check endpoint (Updated)"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mcp_tools_available": MCP_TOOLS_AVAILABLE,
        "server_tools": 6,  # Updated count
        "client_tools": 2,  # Brave Search tools
        "search_integration": "brave_search_client_side"
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
            "available_tools": ["calculator", "test_tool", "diagnostic", "get_weather", "DFWAnalyst", "DFWSearch"],
            "client_tools": ["brave_web_search", "brave_local_search"]
        }
        
    except Exception as e:
        return {
            "connection_status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Additional endpoint for direct prompt testing (Updated)
@route.post("/test_prompt")
async def test_prompt_with_tool():
    """Test a prompt that should trigger a tool call (Updated)"""
    try:
        # Test different prompt patterns (updated without search tools)
        test_prompts = [
            "Use the calculator tool to calculate: 25 * 4 + 10",
            "Use the test_tool with message: prompt test",
            "Use the diagnostic tool with test_type: basic",
            "Use the get_weather tool for: New York"
        ]
        
        results = []
        
        for prompt in test_prompts:
            # Simple tool detection logic
            tool_call = None
            
            if "calculator" in prompt.lower():
                # Extract expression
                parts = prompt.split(":")
                if len(parts) > 1:
                    expression = parts[1].strip()
                    tool_call = ToolCallRequest(
                        tool_name="calculator",
                        arguments={"expression": expression}
                    )
            elif "test_tool" in prompt.lower():
                # Extract message
                parts = prompt.split(":")
                if len(parts) > 1:
                    message = parts[1].strip()
                    tool_call = ToolCallRequest(
                        tool_name="test_tool", 
                        arguments={"message": message}
                    )
            elif "diagnostic" in prompt.lower():
                # Extract test_type
                parts = prompt.split(":")
                if len(parts) > 1:
                    test_type = parts[1].strip()
                    tool_call = ToolCallRequest(
                        tool_name="diagnostic",
                        arguments={"test_type": test_type}
                    )
            elif "weather" in prompt.lower():
                # Extract location
                parts = prompt.split(":")
                if len(parts) > 1:
                    place = parts[1].strip()
                    tool_call = ToolCallRequest(
                        tool_name="get_weather",
                        arguments={"place": place}
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
            "note": "Search tools are now handled client-side"
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

# New endpoint for system info
@route.get("/system_info")
async def get_system_info():
    """Get comprehensive system information"""
    try:
        info = {
            "server": {
                "name": "DataFlyWheel MCP Server",
                "version": "2.0.0",
                "timestamp": datetime.now().isoformat()
            },
            "tools": {
                "server_side": ["calculator", "test_tool", "diagnostic", "get_weather", "DFWAnalyst", "DFWSearch"],
                "client_side": ["brave_web_search", "brave_local_search"],
                "total_count": 8
            },
            "features": {
                "weather_caching": True,
                "hedis_integration": True,
                "brave_search": "client_integrated",
                "diagnostics": True
            },
            "endpoints": {
                "/tools": "List available tools",
                "/tool_call": "Execute tool calls",
                "/health": "Health check",
                "/weather_cache": "Weather cache status",
                "/test_connection": "Test MCP connection",
                "/test_prompt": "Test prompt processing",
                "/system_info": "System information"
            }
        }
        
        return info
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Test endpoint for Brave Search integration info
@route.get("/search_info")
async def get_search_info():
    """Get information about search capabilities"""
    return {
        "search_providers": {
            "brave_search": {
                "status": "integrated_client_side",
                "capabilities": ["web_search", "local_search"],
                "api_version": "v1",
                "rate_limits": "60 requests/minute",
                "features": ["fresh_results", "unbiased_search", "no_tracking"]
            }
        },
        "migration_notes": {
            "wikipedia_search": "Replaced with brave_web_search for better coverage",
            "duckduckgo_search": "Replaced with brave_web_search for API reliability",
            "benefits": [
                "More reliable API",
                "Better rate limits",
                "Fresher results",
                "Independent search engine"
            ]
        },
        "usage": {
            "web_search": "Use 'Web Search' mode in Streamlit client",
            "local_search": "Use 'Local Search' mode in Streamlit client",
            "integration": "Seamlessly integrated in client interface"
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # For testing the router directly
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="MCP Tool Router Test - Updated")
    app.include_router(route)
    
    uvicorn.run(app, host="0.0.0.0", port=8082)
