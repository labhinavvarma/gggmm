
# app.py
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.routing import Mount
from mcp.server.sse import SseServerTransport  # ensure fastmcp >=0.1.38

from mcpserver import mcp
from router import route

app = FastAPI(title="Milliman MCP Server", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

sse = SseServerTransport("/messages")
app.router.routes.append(Mount("/messages", app=sse.handle_post_message))

@app.get("/messages", tags=["MCP"], include_in_schema=True)
def messages_docs(session_id: str):
    """Only for docs; actual posting handled by transport."""
    pass

@app.get("/sse", tags=["MCP"])
async def handle_sse(request: Request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as (r, w):
        await mcp._mcp_server.run(r, w, mcp._mcp_server.create_initialization_options())

app.include_router(route)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
