# mcpserver.py - FIXED VERSION - Return raw content, not ToolResult objects

import json
import re
import logging
import asyncio
import sys
from typing import Any, Optional
from fastmcp.exceptions import ToolError
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
    """Create MCP server with FIXED tool return values."""
    mcp = FastMCP("neo4j-cypher")

    @mcp.tool(name="read_neo4j_cypher")
    async def read_neo4j_cypher(
        query: str = Field(..., description="Cypher query to execute"),
        params: Optional[dict[str, Any]] = Field(None, description="Query parameters")
    ) -> str:  # Return STRING, not list[ToolResult]
        """Execute read-only Cypher queries."""
        if _is_write_query(query):
            error_msg = "Write-type queries not allowed in read tool."
            logger.warning(f"Rejected write query: {query}")
            raise ToolError(error_msg)
        
        try:
            async with driver.session(database=database) as session:
                result_text = await session.execute_read(_read, query, params)
                logger.info(f"Read query executed successfully")
                return result_text  # Return raw JSON string
        except Exception as e:
            error_msg = f"Read query error: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="write_neo4j_cypher")
    async def write_neo4j_cypher(
        query: str = Field(..., description="Cypher write query to execute"),
        params: Optional[dict[str, Any]] = Field(None, description="Query parameters")
    ) -> str:  # Return STRING, not list[ToolResult]
        """Execute write Cypher queries."""
        if not _is_write_query(query):
            error_msg = "Only write-type queries allowed in write tool."
            logger.warning(f"Rejected read query in write tool: {query}")
            raise ToolError(error_msg)
        
        try:
            async with driver.session(database=database) as session:
                result_text = await session.execute_write(_write, query, params)
                logger.info(f"Write query executed successfully")
                return result_text  # Return raw JSON string
        except Exception as e:
            error_msg = f"Write query error: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="health_check")
    async def health_check() -> str:  # Return STRING, not list[ToolResult]
        """Check Neo4j connection health."""
        try:
            async with driver.session(database=database) as session:
                result = await session.run("RETURN 1 as health")
                await result.consume()
                
                health_info = {
                    "status": "healthy",
                    "database": database,
                    "message": "Connection successful"
                }
                
                return json.dumps(health_info)  # Return raw JSON string
        except Exception as e:
            error_msg = f"Health check failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="count_nodes")
    async def count_nodes() -> str:  # Return STRING, not list[ToolResult]
        """Count total nodes in the database."""
        try:
            async with driver.session(database=database) as session:
                result = await session.run("MATCH (n) RETURN count(n) as total_nodes")
                record = await result.single()
                count = record["total_nodes"] if record else 0
                
                result_data = {
                    "total_nodes": count,
                    "database": database,
                    "query": "MATCH (n) RETURN count(n)",
                    "status": "success"
                }
                
                return json.dumps(result_data)  # Return raw JSON string
        except Exception as e:
            error_msg = f"Count nodes failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="list_labels")
    async def list_labels() -> str:  # Return STRING, not list[ToolResult]
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
                
                return json.dumps(result_data)  # Return raw JSON string
        except Exception as e:
            error_msg = f"List labels failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="simple_test")
    async def simple_test() -> str:  # Return STRING, not list[ToolResult]
        """Simple test that returns basic data."""
        test_data = {
            "message": "Hello from MCP server!",
            "timestamp": str(asyncio.get_event_loop().time()),
            "test": True,
            "server": "neo4j-cypher"
        }
        
        return json.dumps(test_data)  # Return raw JSON string

    return mcp

async def run_mcp_server():
    """Main server function."""
    driver = None
    try:
        logger.info("🚀 Starting MCP server initialization...")
        
        # Create Neo4j driver
        driver = AsyncGraphDatabase.driver(
            "neo4j://10.189.116.237:7687",
            auth=("neo4j", "Vkg5d$F!pLq2@9vRwE="),
        )
        
        # Test connection
        async with driver.session(database="connectiq") as session:
            result = await session.run("RETURN 1 as test")
            await result.consume()
        
        logger.info("✅ Neo4j connection established successfully")
        
        # Create MCP server
        mcp = create_mcp_server(driver, "connectiq")
        logger.info("✅ MCP server created with fixed tools")
        
        # Use STDIO transport
        logger.info("🚀 Starting MCP server with STDIO transport")
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
    """Main entry point."""
    try:
        asyncio.run(run_mcp_server())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
