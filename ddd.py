import asyncio
import logging
import argparse
import os
from typing import Literal
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.sse import SseServerTransport
from starlette.routing import Mount

# Import our MCP server
from neo4j_mcp_server import mcp, cleanup

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_app")

def create_fastapi_app() -> FastAPI:
    """Create FastAPI application with MCP integration"""
    
    app = FastAPI(
        title="Neo4j ConnectIQ MCP Server",
        description="Model Context Protocol server for Neo4j ConnectIQ database",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Create SSE transport for real-time communication
    sse = SseServerTransport("/messages/")
    
    # Mount SSE message handler
    app.router.routes.append(Mount("/messages/", app=sse.handle_post_message))
    
    @app.get("/", tags=["Info"])
    async def root():
        """Root endpoint with server information"""
        return {
            "message": "Neo4j ConnectIQ MCP Server",
            "version": "1.0.0",
            "database": "connectiq",
            "endpoints": {
                "sse": "/sse",
                "messages": "/messages/",
                "docs": "/docs",
                "health": "/health"
            }
        }
    
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint"""
        try:
            # Test database connection by getting a simple count
            from neo4j_mcp_server import get_driver, DB_CONFIG
            driver = await get_driver()
            async with driver.session(database=DB_CONFIG["database"]) as session:
                result = await session.run("RETURN 1 as test")
                await result.consume()
            
            return {
                "status": "healthy",
                "database": "connected",
                "timestamp": asyncio.get_event_loop().time()
            }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }
    
    @app.get("/sse", tags=["MCP"])
    async def handle_sse(request: Request):
        """
        SSE endpoint that connects to the MCP server.
        This endpoint establishes a Server-Sent Events connection with the client
        and forwards communication to the Model Context Protocol server.
        """
        logger.info("SSE connection established")
        
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
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown"""
        logger.info("Shutting down MCP server...")
        await cleanup()
    
    return app

async def run_stdio():
    """Run MCP server with stdio transport"""
    logger.info("Starting Neo4j ConnectIQ MCP Server with stdio transport...")
    await mcp.run_stdio_async()

async def run_sse(host: str = "127.0.0.1", port: int = 8000, path: str = "/mcp/"):
    """Run MCP server with SSE transport"""
    logger.info(f"Starting Neo4j ConnectIQ MCP Server with SSE transport on {host}:{port}...")
    await mcp.run_sse_async(host=host, port=port, path=path)

def run_http(host: str = "127.0.0.1", port: int = 8000):
    """Run MCP server with HTTP transport using FastAPI"""
    logger.info(f"Starting Neo4j ConnectIQ MCP Server with HTTP transport on {host}:{port}...")
    
    app = create_fastapi_app()
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Neo4j ConnectIQ MCP Server")
    
    parser.add_argument(
        "--transport", 
        type=str, 
        choices=["stdio", "sse", "http"], 
        default="http",
        help="Transport type (default: http)"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--path", 
        type=str, 
        default="/mcp/",
        help="SSE path (default: /mcp/)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Neo4j ConnectIQ MCP Server")
    logger.info("=" * 60)
    logger.info(f"Database: connectiq")
    logger.info(f"Host: {DB_CONFIG['uri']}")
    logger.info(f"Transport: {args.transport}")
    logger.info(f"Server: {args.host}:{args.port}")
    logger.info("=" * 60)
    
    try:
        if args.transport == "stdio":
            await run_stdio()
        elif args.transport == "sse":
            await run_sse(args.host, args.port, args.path)
        elif args.transport == "http":
            run_http(args.host, args.port)
        else:
            raise ValueError(f"Invalid transport: {args.transport}")
            
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        await cleanup()

# Alternative direct run functions for different scenarios
def run_dev_server():
    """Run development server with hot reload"""
    logger.info("Starting development server with hot reload...")
    
    app = create_fastapi_app()
    
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="debug"
    )

def run_production_server():
    """Run production server"""
    logger.info("Starting production server...")
    
    app = create_fastapi_app()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Neo4j connections work better with single worker
        log_level="info",
        access_log=True
    )

# Create app instance for direct uvicorn usage
app = create_fastapi_app()

if __name__ == "__main__":
    # Import database config for logging
    from neo4j_mcp_server import DB_CONFIG
    
    # Check if we should run in development mode
    if os.getenv("DEV_MODE") == "true":
        run_dev_server()
    elif os.getenv("PRODUCTION") == "true":
        run_production_server()
    else:
        # Run with command line arguments
        asyncio.run(main())
