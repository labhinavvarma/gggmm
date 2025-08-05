import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.sse import SseServerTransport
from starlette.routing import Mount
from mcpserver import mcp
from router import router  # make sure it's called 'router', not 'route'

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the SSE server at a **different path** to avoid route conflicts
sse = SseServerTransport("/messages/")
app.router.routes.append(Mount("/messages/", app=sse.handle_post_message))

# Your REST API endpoints (GET/POST for /api/messages)
app.include_router(router, prefix="/api")

@app.get("/sse", tags=["MCP"])
async def handle_sse(request: Request):
    """
    SSE endpoint that connects to the MCP server
    """
    async with sse.connect_sse(request.scope, request.receive, request._send) as (
        read_stream,
        write_stream,
    ):
        await mcp._mcp_server.run(
            read_stream,
            write_stream,
            mcp._mcp_server.create_initialization_options(),
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)

