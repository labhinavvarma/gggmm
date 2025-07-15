# mcpserver_http.py - Modern MCP server using HTTP transport (not deprecated SSE)

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
    """Create MCP server with modern HTTP transport."""
    # Create server without stateless_http parameter (deprecated)
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
                return [ToolResult(content=[TextContent(type="text", text='{"status": "healthy", "database": "' + database + '"}')])]
        except Exception as e:
            error_msg = f"Health check failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="get_database_info")
    async def get_database_info() -> list[ToolResult]:
        """Get database information."""
        try:
            async with driver.session(database=database) as session:
                # Get node count
                result = await session.run("MATCH (n) RETURN count(n) as node_count")
                record = await result.single()
                node_count = record["node_count"] if record else 0
                
                # Get relationship count
                result = await session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                record = await result.single()
                rel_count = record["rel_count"] if record else 0
                
                info = {
                    "database": database,
                    "node_count": node_count,
                    "relationship_count": rel_count,
                    "status": "connected"
                }
                
                return [ToolResult(content=[TextContent(type="text", text=json.dumps(info))])]
        except Exception as e:
            error_msg = f"Database info query failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    return mcp

async def run_mcp_server():
    """Main server function using modern HTTP transport."""
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
        
        # Use modern HTTP transport instead of deprecated SSE
        logger.info("Starting MCP server with HTTP transport on http://0.0.0.0:8000/")
        await mcp.run_async(
            transport="http",
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        
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
