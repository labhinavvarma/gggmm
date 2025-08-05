from fastapi import (
    APIRouter,
)

router = APIRouter(
    prefix="/api/cortex"
)

# Alias for app.py compatibility
route = router

@route.get("/health/status")
async def get_health_status():
    """Get basic health system status"""
    return {
        "status": "operational",
        "service": "Health Details MCP Server",
        "api_integration": "Milliman/Anthem APIs",
        "timestamp": "2024-01-01T12:00:00Z"
    }

@route.get("/health/tools")
async def get_available_tools():
    """Get list of available MCP tools"""
    return {
        "tools": [
            "all", "token", "medical_submit", 
            "pharmacy_submit", "mcid_search", "get_all_healthcare_data"
        ],
        "prompts": ["health-details", "healthcare-summary"],
        "integration": "FastMCP + SSE compatible"
    }

@route.get("/test")
async def test_endpoint():
    """Test endpoint"""
    return {"message": "Health Details MCP Server is running", "status": "ok"}from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Create router instance
route = APIRouter(prefix="/api/v1", tags=["Health API"])

@route.get("/health/status")
async def get_health_status():
    """Get basic health system status"""
    try:
        return {
            "status": "operational",
            "service": "Health Details MCP Server",
            "api_integration": "Milliman/Anthem APIs",
            "timestamp": "2024-01-01T12:00:00Z",
            "endpoints": {
                "sse": "/sse",
                "messages": "/messages"
            }
        }
    except Exception as e:
        logger.error(f"Error getting health status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@route.get("/health/tools")
async def get_available_tools():
    """Get list of available MCP tools"""
    return {
        "tools": [
            {
                "name": "all",
                "description": "Complete system overview with real API status", 
                "type": "system"
            },
            {
                "name": "token",
                "description": "Get real authentication token from Milliman API",
                "type": "authentication"
            },
            {
                "name": "medical_submit",
                "description": "Submit to real medical API",
                "type": "healthcare_api",
                "endpoint": "https://hix-clm-internaltesting-prod.anthem.com/medical"
            },
            {
                "name": "pharmacy_submit", 
                "description": "Submit to real pharmacy API",
                "type": "healthcare_api",
                "endpoint": "https://hix-clm-internaltesting-prod.anthem.com/pharmacy"
            },
            {
                "name": "mcid_search",
                "description": "Search real MCID service", 
                "type": "healthcare_api",
                "endpoint": "https://mcid-app-prod.anthem.com:443/MCIDExternalService/V2/extSearchService/json"
            },
            {
                "name": "get_all_healthcare_data",
                "description": "Get comprehensive data from all APIs",
                "type": "comprehensive"
            }
        ],
        "prompts": [
            "health-details",
            "healthcare-summary"
        ],
        "integration": "FastMCP + SSE compatible"
    }

@route.get("/health/endpoints")
async def get_api_endpoints():
    """Get configured API endpoints"""
    return {
        "authentication": {
            "url": "https://securefed.antheminc.com/as/token.oauth2",
            "client_id": "MILLIMAN",
            "grant_type": "client_credentials"
        },
        "medical_api": "https://hix-clm-internaltesting-prod.anthem.com/medical",
        "pharmacy_api": "https://hix-clm-internaltesting-prod.anthem.com/pharmacy", 
        "mcid_api": "https://mcid-app-prod.anthem.com:443/MCIDExternalService/V2/extSearchService/json",
        "status": "configured"
    }

@route.get("/health/test")
async def test_mcp_integration():
    """Test MCP server integration"""
    return {
        "mcp_server": "Health Details App",
        "fastmcp_tools": 6,
        "sse_compatible": True,
        "real_api_integration": True,
        "test_status": "ready"
    }
