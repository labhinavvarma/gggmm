# app.py - Fixed version
import uvicorn

from fastapi import (
    FastAPI,
    Request
)

from fastapi.middleware.cors import CORSMiddleware
from mcp.server.sse import SseServerTransport
from starlette.routing import Mount

from mcpserver import _mcp_server  # Import the fixed server instance
 
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
sse = SseServerTransport("/messages/")
 
app.router.routes.append(Mount("/messages/", app=sse.handle_post_message))
 
@app.get("/sse", tags=["MCP"])
async def handle_sse(request: Request):
    """
    SSE endpoint that connects to the MCP server
 
    This endpoint establishes a Server-Sent Events connection with the client
    and forwards communication to the Model Context Protocol server.
    """
    # Use sse.connect_sse to establish an SSE connection with the MCP server
    async with sse.connect_sse(request.scope, request.receive, request._send) as (
        read_stream,
        write_stream,
    ):
        # Run the MCP server with the established streams
        await _mcp_server.run(
            read_stream,
            write_stream,
            _mcp_server.create_initialization_options(),
        )
 
 
from router import route
 
app.include_router(route)
 
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
