from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx

router = APIRouter()
MCP_URL = "http://localhost:8000/mcp/"

class CypherRequest(BaseModel):
    query: str
    params: dict = {}

@router.post("/cypher/read")
async def read_cypher(request: CypherRequest):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{MCP_URL}read_neo4j_cypher", json=request.dict())
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=str(e))

@router.post("/cypher/write")
async def write_cypher(request: CypherRequest):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{MCP_URL}write_neo4j_cypher", json=request.dict())
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=str(e))

@router.get("/schema")
async def get_schema():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{MCP_URL}get_neo4j_schema")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=str(e))
