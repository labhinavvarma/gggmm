import uvicorn
from fastapi import (
    FastAPI,
    Request,
    HTTPException
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mcp.server.sse import SseServerTransport
from starlette.routing import Mount
import json
from datetime import datetime

# Import your MCP server and router
from mcpserver import mcp
from router import route

app = FastAPI(
    title="DataFlyWheel MCP Server",
    description="Enhanced MCP server with tool integration and caching",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SSE transport
sse = SseServerTransport("/messages/")

# Mount SSE message handler
app.router.routes.append(Mount("/messages/", app=sse.handle_post_message))

# Include the router with tool endpoints
app.include_router(route, prefix="/api/v1", tags=["MCP Tools"])

@app.get("/sse", tags=["MCP"])
async def handle_sse(request: Request):
    """
    SSE endpoint that connects to the MCP server
    
    This endpoint establishes a Server-Sent Events connection with the client
    and forwards communication to the Model Context Protocol server.
    """
    try:
        # Use sse.connect_sse to establish an SSE connection with the MCP server
        async with sse.connect_sse(request.scope, request.receive, request._send) as (
            read_stream,
            write_stream,
        ):
            # Run the MCP server with the established streams
            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )
    except Exception as e:
        print(f"SSE connection error: {e}")
        raise HTTPException(status_code=500, detail=f"SSE connection failed: {str(e)}")

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with server information"""
    return {
        "name": "DataFlyWheel MCP Server",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "sse": "/sse",
            "tools": "/api/v1/tools", 
            "tool_call": "/api/v1/tool_call",
            "health": "/api/v1/health",
            "weather_cache": "/api/v1/weather_cache"
        },
        "features": [
            "Enhanced Wikipedia search with fresh data",
            "DuckDuckGo web search with content analysis", 
            "Weather data with caching and validation",
            "HEDIS tools integration",
            "Calculator and diagnostic tools"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["Info"])
async def health_check():
    """Enhanced health check endpoint"""
    try:
        # Test MCP server availability
        mcp_status = "available" if mcp else "unavailable"
        
        # Check tool availability
        tool_status = {}
        try:
            from mcpserver import (
                calculate, test_tool, diagnostic,
                wikipedia_search, duckduckgo_search, get_weather,
                dfw_text2sql, dfw_search
            )
            tool_status = {
                "calculator": "available",
                "test_tool": "available", 
                "diagnostic": "available",
                "wikipedia_search": "available",
                "duckduckgo_search": "available",
                "get_weather": "available",
                "DFWAnalyst": "available",
                "DFWSearch": "available"
            }
        except ImportError as e:
            tool_status = {"error": f"Tools import failed: {e}"}
        
        # Check weather cache
        cache_status = "unknown"
        try:
            from mcpserver import weather_cache
            cache_status = f"{len(weather_cache)} entries cached"
        except:
            cache_status = "cache unavailable"
        
        return {
            "status": "healthy",
            "mcp_server": mcp_status,
            "tools": tool_status,
            "weather_cache": cache_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.post("/test_integration", tags=["Testing"])
async def test_full_integration():
    """Test the full MCP integration pipeline"""
    try:
        test_results = []
        
        # Test 1: Simple calculator
        from router import handle_tool_call, ToolCallRequest
        
        calc_request = ToolCallRequest(
            tool_name="calculator",
            arguments={"expression": "10 + 5 * 2"}
        )
        calc_result = await handle_tool_call(calc_request)
        test_results.append({
            "test": "calculator", 
            "success": calc_result.success,
            "result": calc_result.result if calc_result.success else calc_result.error
        })
        
        # Test 2: Test tool
        test_request = ToolCallRequest(
            tool_name="test_tool",
            arguments={"message": "integration test"}
        )
        test_result = await handle_tool_call(test_request)
        test_results.append({
            "test": "test_tool",
            "success": test_result.success, 
            "result": test_result.result if test_result.success else test_result.error
        })
        
        # Test 3: Weather (with fallback location)
        weather_request = ToolCallRequest(
            tool_name="get_weather",
            arguments={"place": "New York"}
        )
        weather_result = await handle_tool_call(weather_request)
        test_results.append({
            "test": "weather",
            "success": weather_result.success,
            "result": weather_result.result[:200] + "..." if weather_result.success and len(str(weather_result.result)) > 200 else weather_result.result if weather_result.success else weather_result.error
        })
        
        success_count = sum(1 for r in test_results if r["success"])
        
        return {
            "integration_test": "completed",
            "tests_run": len(test_results),
            "tests_passed": success_count,
            "success_rate": f"{success_count}/{len(test_results)}",
            "results": test_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "integration_test": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    print(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url),
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    print("🚀 DataFlyWheel MCP Server starting up...")
    print("✅ Server initialized with enhanced tools and caching")
    
    # Test basic tool availability
    try:
        from mcpserver import calculate
        test_calc = calculate("2+2")
        print(f"✅ Calculator tool test: {test_calc}")
    except Exception as e:
        print(f"⚠️ Calculator tool test failed: {e}")
    
    print("🌐 Server ready for connections")

@app.on_event("shutdown") 
async def shutdown_event():
    """Shutdown event handler"""
    print("🛑 DataFlyWheel MCP Server shutting down...")
    
    # Clear weather cache
    try:
        from mcpserver import weather_cache
        cache_size = len(weather_cache)
        weather_cache.clear()
        print(f"🧹 Cleared weather cache ({cache_size} entries)")
    except:
        pass
    
    print("✅ Server shutdown complete")

if __name__ == "__main__":
    print("Starting DataFlyWheel MCP Server...")
    print("Enhanced features:")
    print("- Fixed Wikipedia search with current data")
    print("- Enhanced DuckDuckGo search with fresh content")
    print("- Weather tools with caching and validation") 
    print("- All prompts now properly invoke tools")
    print("- Tool call debugging and fallback support")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8081,
        log_level="info",
        access_log=True
    )
