from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcpserver import app as mcp_app, mcp
from mcp.server.sse import SseServerTransport
from starlette.routing import Mount

app = FastAPI(title="Universal MCP App")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Mount MCP FastAPI app
app.mount("/", mcp_app)

# Mount SSE transport for LLM agents if needed
app.router.routes.append(Mount("/messages", app=SseServerTransport("/messages")))
