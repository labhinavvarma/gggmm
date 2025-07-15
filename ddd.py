#!/usr/bin/env python3
"""
FastAPI Wrapper for ConnectIQ Neo4j MCP Server
Imports mcpserver.py and provides web interface with SSE transport
"""

import asyncio
import logging
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from mcp.server.sse import SseServerTransport
from starlette.routing import Mount

# Import our MCP server
try:
    from mcpserver import (
        mcp, 
        get_mcp_server, 
        get_neo4j_connection, 
        is_connection_healthy, 
        initialize_connection
    )
    MCPSERVER_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import mcpserver: {e}")
    print("🔧 Make sure mcpserver.py is in the same directory")
    MCPSERVER_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fastapi_wrapper")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🌐 Starting FastAPI wrapper for ConnectIQ MCP Server...")
    logger.info("=" * 60)
    
    if MCPSERVER_AVAILABLE:
        # Initialize MCP server connection
        try:
            connection_success = await initialize_connection(show_stats=False)
            if connection_success:
                logger.info("✅ MCP Server connection established")
                logger.info("🗄️  Database: connectiq")
                logger.info("📊 MCP tools ready")
                
                # Log database stats
                neo4j_conn = get_neo4j_connection()
                if neo4j_conn and neo4j_conn.database_stats:
                    stats = neo4j_conn.database_stats
                    logger.info(f"📈 Database: {stats.get('total_nodes', 0):,} nodes, {stats.get('total_relationships', 0):,} relationships")
                
            else:
                logger.warning("⚠️  MCP Server connection failed - FastAPI will run in limited mode")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MCP Server: {e}")
    else:
        logger.error("❌ MCP Server not available - FastAPI running in standalone mode")
    
    logger.info("🌐 FastAPI wrapper ready")
    logger.info("📡 SSE transport available at /sse")
    logger.info("💬 Messages endpoint at /messages/")
    logger.info("📚 API docs at /docs")
    logger.info("=" * 60)
    
    yield
    
    # Cleanup
    logger.info("📴 Shutting down FastAPI wrapper...")
    if MCPSERVER_AVAILABLE:
        neo4j_conn = get_neo4j_connection()
        if neo4j_conn:
            await neo4j_conn.close()
    logger.info("✅ FastAPI wrapper cleanup complete")

# Create FastAPI application
app = FastAPI(
    title="ConnectIQ Neo4j MCP Wrapper",
    description="FastAPI wrapper for ConnectIQ Neo4j MCP Server with SSE transport",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create SSE transport for MCP communication (only if MCP server available)
if MCPSERVER_AVAILABLE:
    sse = SseServerTransport("/messages/")
    app.router.routes.append(Mount("/messages/", app=sse.handle_post_message))

# ===================
# WEB INTERFACE
# ===================

@app.get("/", response_class=HTMLResponse, tags=["Interface"])
async def root():
    """Root endpoint with web interface"""
    if not MCPSERVER_AVAILABLE:
        return """
        <html>
            <head><title>ConnectIQ MCP Wrapper - Error</title></head>
            <body style="font-family: Arial, sans-serif; margin: 40px;">
                <h1>❌ MCP Server Not Available</h1>
                <p>The MCP server could not be imported. Make sure mcpserver.py is available.</p>
                <p><strong>To fix this:</strong></p>
                <ol>
                    <li>Ensure mcpserver.py is in the same directory</li>
                    <li>Run: <code>python mcpserver.py</code> first to test the MCP server</li>
                    <li>Then run: <code>python app.py</code></li>
                </ol>
            </body>
        </html>
        """
    
    neo4j_conn = get_neo4j_connection()
    connection_status = neo4j_conn.connection_status if neo4j_conn else "unknown"
    stats = neo4j_conn.database_stats if neo4j_conn and neo4j_conn.database_stats else {}
    
    return f"""
    <html>
        <head>
            <title>ConnectIQ Neo4j MCP Wrapper</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .status {{ padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .status.connected {{ background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }}
                .status.disconnected {{ background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; }}
                .endpoints {{ background: #e9ecef; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .endpoint {{ margin: 5px 0; }}
                .endpoint a {{ color: #007bff; text-decoration: none; }}
                .endpoint a:hover {{ text-decoration: underline; }}
                .tools {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .tool {{ background: white; margin: 5px 0; padding: 10px; border-radius: 5px; border-left: 3px solid #28a745; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏥 ConnectIQ Neo4j MCP Wrapper</h1>
                    <p>FastAPI wrapper for Model Context Protocol server</p>
                </div>
                
                <div class="status {'connected' if connection_status == 'connected' else 'disconnected'}">
                    <strong>Connection Status:</strong> {connection_status.upper()}
                    <br><strong>Database:</strong> connectiq @ neo4j://10.189.116.237:7687
                </div>
                
                <div class="stats">
                    <div class="stat-card">
                        <h3>📊 Database Statistics</h3>
                        <p><strong>Nodes:</strong> {stats.get('total_nodes', 'N/A'):,}</p>
                        <p><strong>Relationships:</strong> {stats.get('total_relationships', 'N/A'):,}</p>
                        <p><strong>Last Updated:</strong> {stats.get('last_updated', 'N/A')}</p>
                    </div>
                    
                    <div class="stat-card">
                        <h3>🔧 MCP Server</h3>
                        <p><strong>Status:</strong> {'Ready' if is_connection_healthy() else 'Not Ready'}</p>
                        <p><strong>Tools:</strong> 6 available</p>
                        <p><strong>Transport:</strong> SSE</p>
                    </div>
                    
                    <div class="stat-card">
                        <h3>🌐 FastAPI Wrapper</h3>
                        <p><strong>Port:</strong> 8001</p>
                        <p><strong>CORS:</strong> Enabled</p>
                        <p><strong>Auto-reload:</strong> {'Yes' if '--reload' in str(sys.argv) else 'No'}</p>
                    </div>
                </div>
                
                <div class="endpoints">
                    <h3>📡 Available Endpoints</h3>
                    <div class="endpoint">📚 <a href="/docs">API Documentation</a> - Interactive API docs</div>
                    <div class="endpoint">💚 <a href="/health">Health Check</a> - System health status</div>
                    <div class="endpoint">📊 <a href="/status">Detailed Status</a> - Comprehensive status info</div>
                    <div class="endpoint">🧪 <a href="/database/quick-test">Database Test</a> - Quick connectivity test</div>
                    <div class="endpoint">🔍 <a href="/database/schema-summary">Schema Summary</a> - Database structure</div>
                    <div class="endpoint">📡 <a href="/sse">SSE Endpoint</a> - MCP communication (EventSource)</div>
                    <div class="endpoint">💬 <a href="/messages/">Messages</a> - MCP message handler</div>
                </div>
                
                <div class="tools">
                    <h3>🛠️ Available MCP Tools</h3>
                    <div class="tool"><strong>check_connection_health</strong> - Monitor database connection and performance</div>
                    <div class="tool"><strong>execute_cypher</strong> - Execute Cypher queries on ConnectIQ database</div>
                    <div class="tool"><strong>get_database_schema</strong> - Get comprehensive database schema</div>
                    <div class="tool"><strong>get_database_statistics</strong> - Get detailed database statistics</div>
                    <div class="tool"><strong>explore_healthcare_data</strong> - Explore healthcare data patterns</div>
                    <div class="tool"><strong>test_database_queries</strong> - Test database functionality</div>
                </div>
                
                <div style="text-align: center; margin-top: 30px; color: #6c757d;">
                    <p>🏥 ConnectIQ Healthcare Database Integration</p>
                    <p>Powered by Neo4j + FastMCP + FastAPI</p>
                </div>
            </div>
        </body>
    </html>
    """

# ===================
# API ENDPOINTS
# ===================

@app.get("/health", tags=["Health"])
async def health_check():
    """Comprehensive health check endpoint"""
    if not MCPSERVER_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": "MCP server not available",
                "mcpserver_imported": False,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    neo4j_conn = get_neo4j_connection()
    
    if not neo4j_conn:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": "Neo4j connection not initialized",
                "mcpserver_imported": True,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    try:
        # Get detailed health status
        health_status = await neo4j_conn.health_check_async()
        
        if health_status.get("healthy", False):
            return {
                "status": "healthy",
                "database": health_status,
                "mcp_server": {
                    "status": "ready",
                    "tools_available": 6,
                    "imported": True
                },
                "fastapi_wrapper": {
                    "status": "running",
                    "sse_transport": True,
                    "port": 8001
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "database": health_status,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/status", tags=["Health"])
async def detailed_status():
    """Detailed system status"""
    status_info = {
        "fastapi_wrapper": {
            "name": "ConnectIQ Neo4j MCP Wrapper",
            "version": "1.0.0",
            "port": 8001,
            "sse_transport": MCPSERVER_AVAILABLE,
            "startup_time": datetime.now().isoformat()
        },
        "mcp_server": {
            "imported": MCPSERVER_AVAILABLE,
            "ready": False,
            "tools_available": [],
            "connection_status": "unknown"
        },
        "database": {
            "type": "Neo4j",
            "name": "connectiq",
            "uri": "neo4j://10.189.116.237:7687",
            "status": "unknown",
            "statistics": {}
        }
    }
    
    if MCPSERVER_AVAILABLE:
        neo4j_conn = get_neo4j_connection()
        
        status_info["mcp_server"].update({
            "ready": bool(get_mcp_server()),
            "tools_available": [
                "check_connection_health",
                "execute_cypher",
                "get_database_schema",
                "get_database_statistics", 
                "explore_healthcare_data",
                "test_database_queries"
            ],
            "connection_status": neo4j_conn.connection_status if neo4j_conn else "unknown"
        })
        
        if neo4j_conn:
            status_info["database"].update({
                "status": neo4j_conn.connection_status,
                "last_health_check": neo4j_conn.last_health_check.isoformat() if neo4j_conn.last_health_check else None,
                "connection_error": neo4j_conn.connection_error,
                "statistics": neo4j_conn.database_stats
            })
    
    return status_info

@app.get("/database/quick-test", tags=["Database"])
async def quick_database_test():
    """Quick database connectivity test"""
    if not MCPSERVER_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="MCP server not available"
        )
    
    if not is_connection_healthy():
        raise HTTPException(
            status_code=503, 
            detail="Database connection not available"
        )
    
    neo4j_conn = get_neo4j_connection()
    
    try:
        # Simple test query
        result = neo4j_conn.execute_query_sync("RETURN 'ConnectIQ FastAPI Test' as message, datetime() as timestamp")
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "test": "passed",
            "wrapper": "FastAPI",
            "result": result.get("records", []),
            "database": "connectiq",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/schema-summary", tags=["Database"])
async def get_schema_summary():
    """Get a quick summary of the database schema"""
    if not MCPSERVER_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="MCP server not available"
        )
    
    if not is_connection_healthy():
        raise HTTPException(
            status_code=503, 
            detail="Database connection not available"
        )
    
    neo4j_conn = get_neo4j_connection()
    
    try:
        # Get basic schema information
        queries = {
            "node_types": "MATCH (n) RETURN DISTINCT labels(n) as labels, count(n) as count ORDER BY count DESC LIMIT 10",
            "total_nodes": "MATCH (n) RETURN count(n) as total",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) as total",
            "relationship_types": "MATCH ()-[r]->() RETURN DISTINCT type(r) as rel_type, count(r) as count ORDER BY count DESC LIMIT 5"
        }
        
        schema_summary = {}
        
        for query_name, query in queries.items():
            result = neo4j_conn.execute_query_sync(query)
            if "error" not in result:
                schema_summary[query_name] = result.get("records", [])
            else:
                schema_summary[query_name] = f"Error: {result['error']}"
        
        return {
            "database": "connectiq",
            "wrapper": "FastAPI",
            "schema_summary": schema_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sse", tags=["MCP"])
async def handle_sse(request: Request):
    """SSE endpoint for MCP communication"""
    if not MCPSERVER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="MCP server not available - cannot establish SSE connection"
        )
    
    logger.info("🔗 SSE connection established for MCP communication")
    
    try:
        # Ensure database is ready
        if not is_connection_healthy():
            logger.warning("⚠️  SSE connection attempted but database not ready")
        
        # Use sse.connect_sse to establish connection with MCP server
        async with sse.connect_sse(request.scope, request.receive, request._send) as (
            read_stream,
            write_stream,
        ):
            logger.info("📡 MCP server handling SSE streams via FastAPI wrapper")
            
            # Run the MCP server with the established streams
            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )
            
    except Exception as e:
        logger.error(f"❌ SSE connection error in FastAPI wrapper: {e}")
        raise HTTPException(status_code=500, detail=f"SSE connection failed: {str(e)}")

# Development endpoints
@app.post("/tools/test", tags=["Development"])
async def test_mcp_tool(tool_name: str, arguments: Dict[str, Any] = None):
    """Test MCP tools directly (for development purposes)"""
    if not MCPSERVER_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="MCP server not available"
        )
    
    if not is_connection_healthy():
        raise HTTPException(
            status_code=503, 
            detail="Database connection not available"
        )
    
    available_tools = {
        "health": "check_connection_health",
        "schema": "get_database_schema",
        "stats": "get_database_statistics",
        "explore": "explore_healthcare_data",
        "test": "test_database_queries"
    }
    
    if tool_name not in available_tools:
        raise HTTPException(
            status_code=400, 
            detail=f"Tool not found. Available: {list(available_tools.keys())}"
        )
    
    try:
        return {
            "tool": tool_name,
            "wrapper": "FastAPI",
            "message": "Use SSE endpoint /sse for proper MCP communication",
            "available_tools": list(available_tools.keys()),
            "sse_endpoint": "/sse",
            "mcp_server_ready": is_connection_healthy(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===================
# SERVER STARTUP
# ===================

def run_development_server():
    """Run development server with hot reload"""
    logger.info("🔧 Starting FastAPI wrapper in DEVELOPMENT mode...")
    
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        log_level="info",
        access_log=True
    )

def run_production_server():
    """Run production server"""
    logger.info("🚀 Starting FastAPI wrapper in PRODUCTION mode...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        workers=1,  # Single worker for Neo4j connection stability
        log_level="info",
        access_log=True
    )

def run_local_server():
    """Run local server for testing"""
    logger.info("🏠 Starting FastAPI wrapper in LOCAL mode...")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    import sys
    
    # Print startup banner
    print("🌐" + "="*58 + "🌐")
    print("  ConnectIQ Neo4j MCP FastAPI Wrapper")
    print("  Web Interface for MCP Server")
    print("🌐" + "="*58 + "🌐")
    print()
    
    if not MCPSERVER_AVAILABLE:
        print("❌ MCP Server not available!")
        print("🔧 Make sure mcpserver.py is in the same directory")
        print("📋 To fix this:")
        print("   1. Run: python mcpserver.py (to test MCP server)")
        print("   2. Then run: python app.py")
        print()
        sys.exit(1)
    
    print("📍 Database: connectiq @ neo4j://10.189.116.237:7687")
    print("🔧 MCP Server: Available")
    print("📡 Transport: SSE (Server-Sent Events)")
    print("🌐 Port: 8001")
    print()
    
    # Check command line arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else "local"
    
    if mode == "dev":
        print("🔧 Running FastAPI wrapper in DEVELOPMENT mode")
        print("   Features: Hot reload, debug logging")
        print("   Access: http://127.0.0.1:8001")
        print("   Docs: http://127.0.0.1:8001/docs")
        print("   SSE: http://127.0.0.1:8001/sse")
        print()
        run_development_server()
        
    elif mode == "prod":
        print("🚀 Running FastAPI wrapper in PRODUCTION mode")
        print("   Access: http://0.0.0.0:8001")
        print("   Docs: http://0.0.0.0:8001/docs")
        print("   SSE: http://0.0.0.0:8001/sse")
        print()
        run_production_server()
        
    else:
        print("🏠 Running FastAPI wrapper in LOCAL mode")
        print("   Access: http://127.0.0.1:8001")
        print("   Web UI: http://127.0.0.1:8001")
        print("   Docs: http://127.0.0.1:8001/docs") 
        print("   SSE: http://127.0.0.1:8001/sse")
        print("   Health: http://127.0.0.1:8001/health")
        print()
        print("💡 Usage:")
        print("   python app.py          # Local mode")
        print("   python app.py dev      # Development mode")
        print("   python app.py prod     # Production mode")
        print()
        print("🔗 Make sure to run 'python mcpserver.py' first to see connection details")
        print()
        run_local_server()
