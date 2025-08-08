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
from contextlib import asynccontextmanager
import json
from datetime import datetime
import asyncio

# Import your MCP server and router
from mcpserver import mcp
from router import route

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    
    # === STARTUP ===
    print("🚀 DataFlyWheel MCP Server with Brave Search starting up...")
    print("✅ Server initialized with Brave Search integration and caching")
    
    # Test basic tool availability
    try:
        from mcpserver import calculate
        test_calc = calculate("2+2")
        print(f"✅ Calculator tool test: {test_calc}")
    except Exception as e:
        print(f"⚠️ Calculator tool test failed: {e}")
    
    # Test Brave Search availability
    try:
        from mcpserver import BRAVE_API_KEY
        if BRAVE_API_KEY:
            print(f"✅ Brave API key configured: {BRAVE_API_KEY[:10]}...")
        else:
            print("⚠️ Brave API key not configured yet (will be set from client)")
    except Exception as e:
        print(f"⚠️ Brave Search setup: {e}")
    
    print("🌐 Server ready for connections")
    print("🔍 Brave Search: Ready for web and local search")
    
    # Yield control to the application
    yield
    
    # === SHUTDOWN ===
    print("🛑 DataFlyWheel MCP Server shutting down...")
    
    # Clear weather cache
    try:
        from mcpserver import weather_cache
        cache_size = len(weather_cache)
        weather_cache.clear()
        print(f"🧹 Cleared weather cache ({cache_size} entries)")
    except Exception as e:
        print(f"⚠️ Failed to clear weather cache: {e}")
    
    # Clear Brave search cache
    try:
        from mcpserver import brave_search_cache
        cache_size = len(brave_search_cache)
        brave_search_cache.clear()
        print(f"🧹 Cleared Brave search cache ({cache_size} entries)")
    except Exception as e:
        print(f"⚠️ Failed to clear Brave search cache: {e}")
    
    print("✅ Server shutdown complete")

# Create FastAPI app with lifespan handler
app = FastAPI(
    title="DataFlyWheel MCP Server with Brave Search",
    description="Enhanced MCP server with Brave Search integration and caching",
    version="2.1.0",
    lifespan=lifespan
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
        "name": "DataFlyWheel MCP Server with Brave Search",
        "version": "2.1.0",
        "status": "running",
        "endpoints": {
            "sse": "/sse",
            "tools": "/api/v1/tools", 
            "tool_call": "/api/v1/tool_call",
            "configure_brave_key": "/api/v1/configure_brave_key",
            "health": "/api/v1/health",
            "weather_cache": "/api/v1/weather_cache",
            "brave_cache": "/api/v1/brave_cache"
        },
        "features": [
            "Brave Web Search with fresh, unbiased results",
            "Brave Local Search for businesses and places", 
            "Weather data with caching and validation",
            "HEDIS tools integration",
            "Calculator and diagnostic tools",
            "No tracking or data collection"
        ],
        "search_engine": "Brave Search API",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["Info"])
async def health_check():
    """Enhanced health check endpoint"""
    try:
        # Test MCP server availability
        mcp_status = "available" if mcp else "unavailable"
        
        # Check tool availability - Updated for Brave Search
        tool_status = {}
        try:
            from mcpserver import (
                calculate, test_tool, diagnostic,
                brave_web_search, brave_local_search, get_weather,
                dfw_text2sql, dfw_search, BRAVE_API_KEY
            )
            tool_status = {
                "calculator": "available",
                "test_tool": "available", 
                "diagnostic": "available",
                "brave_web_search": "available",
                "brave_local_search": "available",
                "get_weather": "available",
                "DFWAnalyst": "available",
                "DFWSearch": "available"
            }
            
            # Add Brave API key status
            tool_status["brave_api_key"] = "configured" if BRAVE_API_KEY else "not_configured"
            
        except ImportError as e:
            tool_status = {"error": f"Tools import failed: {e}"}
        
        # Check weather cache
        weather_cache_status = "unknown"
        try:
            from mcpserver import weather_cache
            weather_cache_status = f"{len(weather_cache)} entries cached"
        except Exception as e:
            weather_cache_status = f"cache unavailable: {e}"
        
        # Check Brave search cache
        brave_cache_status = "unknown"
        try:
            from mcpserver import brave_search_cache
            brave_cache_status = f"{len(brave_search_cache)} searches cached"
        except Exception as e:
            brave_cache_status = f"cache unavailable: {e}"
        
        return {
            "status": "healthy",
            "mcp_server": mcp_status,
            "tools": tool_status,
            "weather_cache": weather_cache_status,
            "brave_cache": brave_cache_status,
            "search_engine": "Brave Search API",
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
    """Test the full MCP integration pipeline with Brave Search"""
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
        
        # Test 4: Brave Web Search (if API key configured)
        try:
            from mcpserver import BRAVE_API_KEY
            if BRAVE_API_KEY:
                brave_request = ToolCallRequest(
                    tool_name="brave_web_search",
                    arguments={"query": "test search", "count": 1}
                )
                brave_result = await handle_tool_call(brave_request)
                test_results.append({
                    "test": "brave_web_search",
                    "success": brave_result.success,
                    "result": brave_result.result[:200] + "..." if brave_result.success and len(str(brave_result.result)) > 200 else brave_result.result if brave_result.success else brave_result.error
                })
            else:
                test_results.append({
                    "test": "brave_web_search",
                    "success": False,
                    "result": "API key not configured"
                })
        except Exception as e:
            test_results.append({
                "test": "brave_web_search",
                "success": False,
                "result": f"Error: {str(e)}"
            })
        
        success_count = sum(1 for r in test_results if r["success"])
        
        return {
            "integration_test": "completed",
            "tests_run": len(test_results),
            "tests_passed": success_count,
            "success_rate": f"{success_count}/{len(test_results)}",
            "results": test_results,
            "search_engine": "Brave Search API",
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

# Port detection and management
def find_available_port(start_port=8081, max_attempts=10):
    """Find an available port starting from start_port"""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(('0.0.0.0', port))
                return port
            except OSError:
                continue
    
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts}")

if __name__ == "__main__":
    print("Starting DataFlyWheel MCP Server with Brave Search...")
    print("Enhanced features:")
    print("- Brave Web Search with fresh, unbiased results")
    print("- Brave Local Search for businesses and places")
    print("- Weather tools with caching and validation") 
    print("- All prompts now properly invoke tools")
    print("- Tool call debugging and fallback support")
    print("- No tracking or data collection")
    print("- API key configuration from client")
    
    # Try to find an available port
    try:
        port = find_available_port(8081)
        if port != 8081:
            print(f"⚠️ Port 8081 is busy, using port {port} instead")
        else:
            print(f"✅ Using default port {port}")
    except RuntimeError as e:
        print(f"❌ {e}")
        print("Please manually specify a different port or stop the process using port 8081")
        exit(1)
    
    print(f"🚀 Server will start at: http://0.0.0.0:{port}")
    print(f"📡 SSE endpoint: http://0.0.0.0:{port}/sse")
    print(f"🔧 API endpoints: http://0.0.0.0:{port}/api/v1/")
    print(f"❤️ Health check: http://0.0.0.0:{port}/health")
    print(f"🔑 Configure Brave key: http://0.0.0.0:{port}/api/v1/configure_brave_key")
    print(f"🔍 Brave Search: Ready (configure API key via Streamlit client)")
    
    # Check if something is already running on the port
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex(('localhost', port))
        if result == 0:
            print(f"⚠️ Warning: Something is already running on port {port}")
            print("🔄 Attempting to start anyway...")
    
    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        
        if "address already in use" in str(e).lower():
            print(f"💡 Port {port} is in use. Try:")
            print(f"   1. Kill the process: sudo lsof -ti:{port} | xargs sudo kill -9")
            print(f"   2. Or use a different port: python3 app.py --port 8082")
            print(f"   3. Or check what's running: sudo netstat -tulpn | grep {port}")
        
        exit(1)
