# mcpserver.py

import json
import re
import logging
from typing import Any, Optional
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult, TextContent
from fastmcp.server import FastMCP
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncTransaction
from pydantic import Field

logger = logging.getLogger("neo4j_mcp")
logging.basicConfig(level=logging.INFO)

def _is_write_query(query: str) -> bool:
    return bool(re.search(r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP)\b", query, re.IGNORECASE))

async def _read(tx: AsyncTransaction, query: str, params: dict[str, Any]) -> str:
    result = await tx.run(query, params or {})
    eager = await result.to_eager_result()
    return json.dumps([r.data() for r in eager.records], default=str)

async def _write(tx: AsyncTransaction, query: str, params: dict[str, Any]) -> str:
    result = await tx.run(query, params or {})
    summary = await result.consume()
    return json.dumps(summary.counters._raw_data, default=str)

def create_mcp_server(driver: AsyncDriver, database: str) -> FastMCP:
    mcp = FastMCP("neo4j-cypher", stateless_http=True)

    @mcp.tool(name="read_neo4j_cypher")
    async def read_neo4j_cypher(
        query: str = Field(...), params: Optional[dict[str, Any]] = None
    ) -> list[ToolResult]:
        if _is_write_query(query):
            raise ToolError("Write-type queries not allowed in read tool.")
        try:
            async with driver.session(database=database) as session:
                result = await session.execute_read(_read, query, params)
                return ToolResult(content=[TextContent(type="text", text=result)])
        except Exception as e:
            logger.error(f"Read error: {e}")
            raise ToolError(str(e))

    @mcp.tool(name="write_neo4j_cypher")
    async def write_neo4j_cypher(
        query: str = Field(...), params: Optional[dict[str, Any]] = None
    ) -> list[ToolResult]:
        if not _is_write_query(query):
            raise ToolError("Only write-type queries allowed in write tool.")
        try:
            async with driver.session(database=database) as session:
                result = await session.execute_write(_write, query, params)
                return ToolResult(content=[TextContent(type="text", text=result)])
        except Exception as e:
            logger.error(f"Write error: {e}")
            raise ToolError(str(e))

    return mcp

async def run_mcp_server():
    driver = AsyncGraphDatabase.driver(
        "neo4j://10.189.116.237:7687",
        auth=("neo4j", "Vkg5d$F!pLq2@9vRwE="),
    )
    mcp = create_mcp_server(driver, "connectiq")
    await mcp.run_sse_async(host="0.0.0.0", port=8000, path="/messages/")
