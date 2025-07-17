from fastapi import APIRouter

"""
Health and debug endpoints for FastAPI app.
"""

router = APIRouter()

@router.get("/")
def health():
    "Basic health check for the service."
    return {"status": "OK"}
