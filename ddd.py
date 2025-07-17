from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from neo4j import AsyncGraphDatabase, AsyncTransaction
from pydantic import BaseModel
from typing import Optional, Any
import re
import json
import uvicorn

# -------- Neo4j connection config ---------
NEO4J_URI = "neo4j://10.189.116.237:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Vkg5d$F!pLq2@9vRwE="
DATABASE = "connectiq"
driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# -------- FastAPI app --------
app = FastAPI(title="Neo4j FastAPI MCP Replacement")

# -------- Helper Functions --------
def _is_write(query: str) -> bool:
    return bool(re.search(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP)\b", query, re.IGNORECASE))

async def _read(tx: AsyncTransaction, query: str, params: dict[str, Any]) -> list:
    res = await tx.run(query, params)
    eager_results = await res.to_eager_result()
    return [r.data() for r in eager_results.records]

async def _write(tx: AsyncTransaction, query: str, params: dict[str, Any]):
    return await tx.run(query, params)

# -------- Request Schemas --------
class CypherRequest(BaseModel):
    query: str
    params: Optional[dict] = {}

# -------- Endpoints --------

@app.post("/read_neo4j_cypher", response_class=JSONResponse)
async def read_neo4j_cypher(request: CypherRequest):
    if _is_write(request.query):
        raise HTTPException(status_code=400, detail="Only read queries allowed")
    async with driver.session(database=DATABASE) as session:
        results = await session.execute_read(_read, request.query, request.params or {})
        return JSONResponse(content=results)

@app.post("/write_neo4j_cypher", response_class=JSONResponse)
async def write_neo4j_cypher(request: CypherRequest):
    if not _is_write(request.query):
        raise HTTPException(status_code=400, detail="Only write queries allowed")
    async with driver.session(database=DATABASE) as session:
        await session.execute_write(_write, request.query, request.params or {})
        return JSONResponse(content={"status": "Write query executed"})

@app.post("/get_neo4j_schema", response_class=JSONResponse)
async def get_neo4j_schema():
    query = "CALL apoc.meta.schema();"
    async with driver.session(database=DATABASE) as session:
        results = await session.execute_read(_read, query, {})
        val = results[0].get("value") if results else {}
        return JSONResponse(content=val or {})

if __name__ == "__main__":
    uvicorn.run("mcpserver:app", host="0.0.0.0", port=8000, reload=True)
