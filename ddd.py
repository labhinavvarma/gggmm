from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
import asyncio
import httpx
from datetime import datetime
import time

# Import your MCP server tools (you'll need to adjust this import path)
try:
    from mcpserver import (
        calculate, test_tool, diagnostic, 
        wikipedia_search, duckduckgo_search, get_weather,
        dfw_text2sql, dfw_search
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
    """Handle MCP tool calls via HTTP API"""
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
        
        # Route to appropriate tool
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
            
        elif tool_name == "wikipedia_search":
            query = arguments.get("query", "")
            max_results = arguments.get("max_results", 3)
            result = await wikipedia_search(query, ctx, max_results)
            
        elif tool_name == "duckduckgo_search":
            query = arguments.get("query", "")
            max_results = arguments.get("max_results", 3)
            result = await duckduckgo_search(query, ctx, max_results)
            
        elif tool_name == "get_weather":
            place = arguments.get("place", "")
            result = await get_weather(place, ctx)
            
        elif tool_name == "DFWAnalyst":
            prompt = arguments.get("prompt", "")
            result = await dfw_text2sql(prompt, ctx)
            
        elif tool_name == "DFWSearch":
            query = arguments.get("query", "")
            result = await dfw_search(ctx, query)
            
        else:
            return ToolCallResponse(
                success=False,
                error=f"Unknown tool: {tool_name}"
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
    """List all available MCP tools"""
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
        "wikipedia_search": {
            "description": "Search Wikipedia for current information",
            "args": {"query": "string", "max_results": "integer (optional)"}
        },
        "duckduckgo_search": {
            "description": "Search the web using DuckDuckGo for latest information",
            "args": {"query": "string", "max_results": "integer (optional)"}
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
    
    return {
        "tools": tools,
        "count": len(tools),
        "available": MCP_TOOLS_AVAILABLE
    }

@route.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mcp_tools_available": MCP_TOOLS_AVAILABLE
    }

@route.post("/test_connection")
async def test_mcp_connection():
    """Test the MCP connection by calling a simple tool"""
    try:
        test_request = ToolCallRequest(
            tool_name="test_tool",
            arguments={"message": "connection test"}
        )
        
        result = await handle_tool_call(test_request)
        
        return {
            "connection_status": "success" if result.success else "failed",
            "test_result": result.result if result.success else result.error,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "connection_status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Additional endpoint for direct prompt testing
@route.post("/test_prompt")
async def test_prompt_with_tool():
    """Test a prompt that should trigger a tool call"""
    try:
        # Test different prompt patterns
        test_prompts = [
            "Use the calculator tool to calculate: 25 * 4 + 10",
            "Use the test_tool with message: prompt test",
            "Use the diagnostic tool with test_type: basic",
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
            "timestamp": datetime.now().isoformat()
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

if __name__ == "__main__":
    # For testing the router directly
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="MCP Tool Router Test")
    app.include_router(route)
    
    uvicorn.run(app, host="0.0.0.0", port=8022)
