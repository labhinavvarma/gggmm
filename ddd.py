import json
import logging
import re
from typing import Any, Optional
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult, TextContent
from fastmcp.server import FastMCP
from neo4j import AsyncGraphDatabase, AsyncTransaction
from pydantic import Field

logger = logging.getLogger("mcp_neo4j")

def _is_write(query: str) -> bool:
    return bool(re.search(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP)\b", query, re.IGNORECASE))

async def _read(tx: AsyncTransaction, query: str, params: dict[str, Any]) -> str:
    res = await tx.run(query, params)
    eager_results = await res.to_eager_result()
    return json.dumps([r.data() for r in eager_results.records], default=str)

async def _write(tx: AsyncTransaction, query: str, params: dict[str, Any]):
    return await tx.run(query, params)

# Neo4j connection
neo4j_driver = AsyncGraphDatabase.driver(
    "neo4j://10.189.116.237:7687", auth=("neo4j", "Vkg5d$F!pLq2@9vRwE=")
)
mcp = FastMCP("mcp-neo4j", stateless_http=True)

@mcp.tool(description="Returns schema metadata for the Neo4j database (labels, properties, relationship types).")
async def get_neo4j_schema() -> list[ToolResult]:
    """
    Return graph structure, labels, properties, and relationships.
    """
    query = "CALL apoc.meta.schema();"
    async with neo4j_driver.session(database="connectiq") as session:
        result = await session.execute_read(_read, query, {})
        val = json.loads(result)[0].get("value")
        return [ToolResult(content=[TextContent(type="text", text=json.dumps(val))])]

@mcp.tool(description="Run a Cypher read-only query (MATCH, RETURN, etc). Use for analysis, counts, selects.")
async def read_neo4j_cypher(
    query: str = Field(..., description="The Cypher read query."),
    params: Optional[dict[str, Any]] = Field(None, description="Query parameters."),
) -> list[ToolResult]:
    """
    Executes a read query on the Neo4j database.
    """
    if _is_write(query):
        raise ToolError("Only read queries allowed.")
    async with neo4j_driver.session(database="connectiq") as session:
        result = await session.execute_read(_read, query, params or {})
        return [ToolResult(content=[TextContent(type="text", text=result)])]

@mcp.tool(description="Run a Cypher write query (CREATE, MERGE, DELETE, SET, etc). Use for creating, updating, or deleting data.")
async def write_neo4j_cypher(
    query: str = Field(..., description="The Cypher write query."),
    params: Optional[dict[str, Any]] = Field(None, description="Query parameters."),
) -> list[ToolResult]:
    """
    Executes a write query on the Neo4j database.
    """
    if not _is_write(query):
        raise ToolError("Only write queries allowed.")
    async with neo4j_driver.session(database="connectiq") as session:
        result = await session.execute_write(_write, query, params or {})
        return [ToolResult(content=[TextContent(type="text", text="Write query executed.")])]

if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_http_async(host="0.0.0.0", port=8000, path="/mcp/"))
