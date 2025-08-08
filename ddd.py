from fastapi import APIRouter, HTTPException
from typing import Dict, Any

# Create router instance
route = APIRouter()

@route.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with server information
    """
    return {
        "server": "MCP Brave Search Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "sse": "/sse",
            "messages": "/messages/",
            "docs": "/docs",
            "health": "/health"
        },
        "available_tools": [
            "configure_brave_key",
            "brave_web_search", 
            "brave_local_search",
            "calculator",
            "get_weather",
            "get_cache_stats",
            "clear_cache"
        ],
        "available_prompts": [
            "brave_search_expert",
            "local_business_finder", 
            "weather_assistant",
            "calculation_helper"
        ]
    }

@route.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "server": "MCP Brave Search Server",
        "mcp_server_ready": True
    }

@route.get("/info", tags=["Info"])
async def server_info():
    """
    Server information endpoint
    """
    return {
        "name": "Brave Search MCP Server",
        "description": "FastAPI app with MCP server for Brave Search integration",
        "version": "1.0.0",
        "mcp": {
            "transport": "sse",
            "endpoint": "/sse"
        },
        "features": [
            "Web Search via Brave Search API",
            "Local Business Search", 
            "Mathematical Calculator",
            "Weather Information",
            "Caching System",
            "Multiple Client Support"
        ]
    }
