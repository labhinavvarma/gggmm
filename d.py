# app.py - Complete fixed version with health endpoints
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.routing import Mount
from datetime import datetime

# Import your existing MCP modules
try:
    from mcp.server.sse import SseServerTransport
    from mcpserver import mcp
    from router import router
    MCP_AVAILABLE = True
    print("✅ MCP modules loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: Could not import MCP modules: {e}")
    MCP_AVAILABLE = False

# Create FastAPI app with proper configuration
app = FastAPI(
    title="Milliman Combined API",
    version="1.0",
    description="Healthcare API with MCP integration and health endpoints"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ===== ESSENTIAL ENDPOINTS FOR CHATBOT =====

@app.get("/")
async def root():
    """Root endpoint - confirms server is running"""
    return {
        "message": "Milliman Healthcare API",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "health_check": "/health",
        "debug": "/debug/routes",
        "mcp_available": MCP_AVAILABLE,
        "version": "1.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint - REQUIRED for chatbot diagnostics"""
    return {
        "status": "healthy",
        "service": "Milliman Combined API",
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "server_running": True,
        "mcp_available": MCP_AVAILABLE,
        "endpoints_available": {
            "medical": "/medical/submit (POST)",
            "pharmacy": "/pharmacy/submit (POST)",
            "mcid": "/mcid/search (POST)",
            "token": "/token (POST)",
            "all": "/all (POST)"
        },
        "router_status": "mounted" if MCP_AVAILABLE else "unavailable"
    }

@app.get("/debug/routes")
async def debug_routes():
    """Debug endpoint - lists all available routes"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods) if hasattr(route, 'methods') else [],
                "name": getattr(route, 'name', 'N/A')
            })
    
    return {
        "routes": routes,
        "total_routes": len(routes),
        "mcp_available": MCP_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

# ===== MCP INTEGRATION =====

if MCP_AVAILABLE:
    print("🔧 Setting up MCP integration...")
    
    try:
        # SSE setup for MCP
        sse = SseServerTransport("/messages")
        app.router.routes.append(Mount("/messages", app=sse.handle_post_message))
        
        @app.get("/messages", tags=["MCP"], include_in_schema=True)
        def messages_docs(session_id: str):
            """MCP messages endpoint documentation"""
            return {"message": "MCP SSE endpoint", "session_id": session_id}
        
        @app.get("/sse", tags=["MCP"])
        async def handle_sse(request: Request):
            """Handle Server-Sent Events for MCP"""
            async with sse.connect_sse(request.scope, request.receive, request._send) as (r, w):
                await mcp._mcp_server.run(r, w, mcp._mcp_server.create_initialization_options())
        
        print("✅ SSE setup completed")
    except Exception as e:
        print(f"⚠️ Warning: Could not set up SSE for MCP: {e}")
    
    # Include the main router with healthcare endpoints
    try:
        app.include_router(router)
        print("✅ Healthcare router included successfully")
        print("📡 Available endpoints: /medical/submit, /pharmacy/submit, /mcid/search, /token, /all")
    except Exception as e:
        print(f"❌ Error including healthcare router: {e}")
        MCP_AVAILABLE = False

# ===== FALLBACK ENDPOINTS (if MCP not available) =====

if not MCP_AVAILABLE:
    print("⚠️ MCP not available - creating fallback endpoints...")
    
    @app.post("/medical/submit")
    async def fallback_medical():
        return {
            "status_code": 503,
            "error": "MCP server not available - check mcpserver.py and router.py",
            "service": "medical",
            "fallback": True
        }
    
    @app.post("/pharmacy/submit")
    async def fallback_pharmacy():
        return {
            "status_code": 503,
            "error": "MCP server not available - check mcpserver.py and router.py",
            "service": "pharmacy",
            "fallback": True
        }
    
    @app.post("/mcid/search")
    async def fallback_mcid():
        return {
            "status_code": 503,
            "error": "MCP server not available - check mcpserver.py and router.py",
            "service": "mcid",
            "fallback": True
        }
    
    @app.post("/token")
    async def fallback_token():
        return {
            "status_code": 503,
            "error": "MCP server not available - check mcpserver.py and router.py",
            "service": "token",
            "fallback": True
        }
    
    @app.post("/all")
    async def fallback_all():
        return {
            "status_code": 503,
            "error": "MCP server not available - check mcpserver.py and router.py",
            "service": "all",
            "fallback": True
        }
    
    print("⚠️ Fallback endpoints created - server will start but healthcare APIs won't work")

# ===== STARTUP CONFIGURATION =====

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*50)
    print("🚀 Milliman Healthcare API Starting Up")
    print("="*50)
    print(f"📡 MCP Available: {MCP_AVAILABLE}")
    print(f"🏥 Health Check: http://localhost:8000/health")
    print(f"🐛 Debug Routes: http://localhost:8000/debug/routes")
    print(f"📍 Root Endpoint: http://localhost:8000/")
    
    if MCP_AVAILABLE:
        print("✅ Healthcare endpoints available:")
        print("   • POST /medical/submit")
        print("   • POST /pharmacy/submit")
        print("   • POST /mcid/search")
        print("   • POST /token")
        print("   • POST /all")
    else:
        print("⚠️ MCP not available - only fallback endpoints")
    
    print("="*50)

# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    print("🔧 Starting Milliman Healthcare API server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("🏥 Test health endpoint: http://localhost:8000/health")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False  # Set to True for development
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        print("💡 Try a different port: uvicorn app:app --port 8001")
