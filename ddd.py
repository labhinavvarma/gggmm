# mcpserver.py - Fixed version with TaskGroup error handling

import json
import re
import logging
import asyncio
import sys
from typing import Any, Optional
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult, TextContent
from fastmcp.server import FastMCP
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncTransaction
from pydantic import Field

logger = logging.getLogger("neo4j_mcp")
logging.basicConfig(level=logging.INFO)

def _is_write_query(query: str) -> bool:
    """Check if query is a write operation."""
    return bool(re.search(r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP)\b", query, re.IGNORECASE))

async def _read(tx: AsyncTransaction, query: str, params: dict[str, Any]) -> str:
    """Execute read query with error handling."""
    try:
        result = await tx.run(query, params or {})
        eager = await result.to_eager_result()
        return json.dumps([r.data() for r in eager.records], default=str)
    except Exception as e:
        logger.error(f"Read query failed: {e}")
        raise

async def _write(tx: AsyncTransaction, query: str, params: dict[str, Any]) -> str:
    """Execute write query with error handling."""
    try:
        result = await tx.run(query, params or {})
        summary = await result.consume()
        return json.dumps(summary.counters._raw_data, default=str)
    except Exception as e:
        logger.error(f"Write query failed: {e}")
        raise

def create_mcp_server(driver: AsyncDriver, database: str) -> FastMCP:
    """Create MCP server with enhanced error handling."""
    mcp = FastMCP("neo4j-cypher", stateless_http=True)

    @mcp.tool(name="read_neo4j_cypher")
    async def read_neo4j_cypher(
        query: str = Field(..., description="Cypher query to execute"),
        params: Optional[dict[str, Any]] = Field(None, description="Query parameters")
    ) -> list[ToolResult]:
        """Execute read-only Cypher queries."""
        if _is_write_query(query):
            error_msg = "Write-type queries not allowed in read tool."
            logger.warning(f"Rejected write query: {query}")
            raise ToolError(error_msg)
        
        try:
            async with driver.session(database=database) as session:
                result = await session.execute_read(_read, query, params)
                logger.info(f"Read query executed successfully: {query[:100]}...")
                return [ToolResult(content=[TextContent(type="text", text=result)])]
        except Exception as e:
            error_msg = f"Read query error: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="write_neo4j_cypher")
    async def write_neo4j_cypher(
        query: str = Field(..., description="Cypher write query to execute"),
        params: Optional[dict[str, Any]] = Field(None, description="Query parameters")
    ) -> list[ToolResult]:
        """Execute write Cypher queries."""
        if not _is_write_query(query):
            error_msg = "Only write-type queries allowed in write tool."
            logger.warning(f"Rejected read query in write tool: {query}")
            raise ToolError(error_msg)
        
        try:
            async with driver.session(database=database) as session:
                result = await session.execute_write(_write, query, params)
                logger.info(f"Write query executed successfully: {query[:100]}...")
                return [ToolResult(content=[TextContent(type="text", text=result)])]
        except Exception as e:
            error_msg = f"Write query error: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="health_check")
    async def health_check() -> list[ToolResult]:
        """Check Neo4j connection health."""
        try:
            async with driver.session(database=database) as session:
                result = await session.run("RETURN 1 as health")
                await result.consume()
                return [ToolResult(content=[TextContent(type="text", text='{"status": "healthy"}')])]
        except Exception as e:
            error_msg = f"Health check failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    return mcp

async def safe_run_server(driver: AsyncDriver, database: str):
    """Run MCP server with comprehensive error handling."""
    try:
        mcp = create_mcp_server(driver, database)
        logger.info("Starting MCP server on http://0.0.0.0:8000/messages/")
        await mcp.run_sse_async(host="0.0.0.0", port=8000, path="/messages/")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

async def run_mcp_server():
    """Main server function with TaskGroup error handling."""
    driver = None
    try:
        # Create Neo4j driver
        driver = AsyncGraphDatabase.driver(
            "neo4j://10.189.116.237:7687",
            auth=("neo4j", "Vkg5d$F!pLq2@9vRwE="),
        )
        
        # Test connection
        async with driver.session(database="connectiq") as session:
            await session.run("RETURN 1")
        
        logger.info("Neo4j connection established successfully")
        
        # Run server with proper error handling
        await safe_run_server(driver, "connectiq")
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise
    finally:
        if driver:
            await driver.close()
            logger.info("Neo4j driver closed")

def main():
    """Main entry point with Python version compatibility."""
    try:
        if sys.version_info >= (3, 11):
            # Python 3.11+ with TaskGroup support
            asyncio.run(run_mcp_server())
        else:
            # Python 3.10 and below
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_mcp_server())
            finally:
                loop.close()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
