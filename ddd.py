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
    return {"message": "Health Details MCP Server is running", "status": "ok"}
