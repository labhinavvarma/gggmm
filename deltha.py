# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import SseServerTransport
from mcpserver import mcp
from router import router

app = FastAPI(
    title="Milliman MCP Server",
    description="API and SSE for Milliman Claims",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

mcp.include_fastapi(app, mount_path="/sse", transport=SseServerTransport())

app.include_router(router)
