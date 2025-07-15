# mcpserver.py - STDIO transport version (more stable than SSE)

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
    """Create MCP server with STDIO transport."""
    mcp = FastMCP("neo4j-cypher")

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
                logger.info(f"Read query executed successfully")
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
                logger.info(f"Write query executed successfully")
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
                health_info = {
                    "status": "healthy",
                    "database": database,
                    "connection": "active"
                }
                return [ToolResult(content=[TextContent(type="text", text=json.dumps(health_info))])]
        except Exception as e:
            error_msg = f"Health check failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="count_nodes")
    async def count_nodes() -> list[ToolResult]:
        """Count total nodes in the database."""
        try:
            async with driver.session(database=database) as session:
                result = await session.run("MATCH (n) RETURN count(n) as total_nodes")
                record = await result.single()
                count = record["total_nodes"] if record else 0
                
                result_data = {
                    "total_nodes": count,
                    "database": database,
                    "query": "MATCH (n) RETURN count(n)"
                }
                
                return [ToolResult(content=[TextContent(type="text", text=json.dumps(result_data))])]
        except Exception as e:
            error_msg = f"Count nodes failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="list_labels")
    async def list_labels() -> list[ToolResult]:
        """List all node labels in the database."""
        try:
            async with driver.session(database=database) as session:
                result = await session.run("CALL db.labels()")
                labels = [record["label"] for record in result]
                
                result_data = {
                    "labels": labels,
                    "count": len(labels),
                    "database": database
                }
                
                return [ToolResult(content=[TextContent(type="text", text=json.dumps(result_data))])]
        except Exception as e:
            error_msg = f"List labels failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    return mcp

async def run_mcp_server():
    """Main server function with STDIO transport."""
    driver = None
    try:
        # Create Neo4j driver
        driver = AsyncGraphDatabase.driver(
            "neo4j://10.189.116.237:7687",
            auth=("neo4j", "Vkg5d$F!pLq2@9vRwE="),
        )
        
        # Test connection
        async with driver.session(database="connectiq") as session:
            result = await session.run("RETURN 1 as test")
            await result.consume()
        
        logger.info("Neo4j connection established successfully")
        
        # Create MCP server
        mcp = create_mcp_server(driver, "connectiq")
        
        # Use STDIO transport (more stable than SSE)
        logger.info("Starting MCP server with STDIO transport")
        await mcp.run_async(transport="stdio")
        
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
    """Main entry point with proper async handling."""
    try:
        asyncio.run(run_mcp_server())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
