# mcpserver_debug.py - Debug version with detailed logging

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

# Enhanced logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("neo4j_mcp")

def _is_write_query(query: str) -> bool:
    """Check if query is a write operation."""
    return bool(re.search(r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP)\b", query, re.IGNORECASE))

async def _read(tx: AsyncTransaction, query: str, params: dict[str, Any]) -> str:
    """Execute read query with error handling."""
    try:
        logger.info(f"Executing read query: {query}")
        result = await tx.run(query, params or {})
        eager = await result.to_eager_result()
        data = [r.data() for r in eager.records]
        json_result = json.dumps(data, default=str)
        logger.info(f"Read query result: {json_result[:200]}...")
        return json_result
    except Exception as e:
        logger.error(f"Read query failed: {e}")
        raise

async def _write(tx: AsyncTransaction, query: str, params: dict[str, Any]) -> str:
    """Execute write query with error handling."""
    try:
        logger.info(f"Executing write query: {query}")
        result = await tx.run(query, params or {})
        summary = await result.consume()
        json_result = json.dumps(summary.counters._raw_data, default=str)
        logger.info(f"Write query result: {json_result}")
        return json_result
    except Exception as e:
        logger.error(f"Write query failed: {e}")
        raise

def create_mcp_server(driver: AsyncDriver, database: str) -> FastMCP:
    """Create MCP server with enhanced debugging."""
    mcp = FastMCP("neo4j-cypher")

    @mcp.tool(name="read_neo4j_cypher")
    async def read_neo4j_cypher(
        query: str = Field(..., description="Cypher query to execute"),
        params: Optional[dict[str, Any]] = Field(None, description="Query parameters")
    ) -> list[ToolResult]:
        """Execute read-only Cypher queries."""
        logger.info(f"🔧 read_neo4j_cypher called with query: {query}")
        
        if _is_write_query(query):
            error_msg = "Write-type queries not allowed in read tool."
            logger.warning(f"Rejected write query: {query}")
            raise ToolError(error_msg)
        
        try:
            async with driver.session(database=database) as session:
                result_text = await session.execute_read(_read, query, params)
                logger.info(f"🔧 Query executed, result length: {len(result_text)}")
                logger.info(f"🔧 Result content: {result_text[:500]}...")
                
                # Create TextContent
                text_content = TextContent(type="text", text=result_text)
                logger.info(f"🔧 TextContent created: {type(text_content)}")
                
                # Create ToolResult
                tool_result = ToolResult(content=[text_content])
                logger.info(f"🔧 ToolResult created: {type(tool_result)}")
                logger.info(f"🔧 ToolResult.content: {tool_result.content}")
                logger.info(f"🔧 ToolResult.content[0]: {tool_result.content[0]}")
                logger.info(f"🔧 ToolResult.content[0].text: {tool_result.content[0].text[:200]}...")
                
                result_list = [tool_result]
                logger.info(f"🔧 Returning list with {len(result_list)} items")
                return result_list
                
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
        logger.info(f"🔧 write_neo4j_cypher called with query: {query}")
        
        if not _is_write_query(query):
            error_msg = "Only write-type queries allowed in write tool."
            logger.warning(f"Rejected read query in write tool: {query}")
            raise ToolError(error_msg)
        
        try:
            async with driver.session(database=database) as session:
                result_text = await session.execute_write(_write, query, params)
                logger.info(f"🔧 Write query executed, result: {result_text}")
                
                text_content = TextContent(type="text", text=result_text)
                tool_result = ToolResult(content=[text_content])
                return [tool_result]
                
        except Exception as e:
            error_msg = f"Write query error: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="health_check")
    async def health_check() -> list[ToolResult]:
        """Check Neo4j connection health."""
        logger.info(f"🔧 health_check called")
        try:
            async with driver.session(database=database) as session:
                result = await session.run("RETURN 1 as health")
                await result.consume()
                
                health_info = {
                    "status": "healthy",
                    "database": database,
                    "message": "Connection successful"
                }
                
                health_json = json.dumps(health_info)
                logger.info(f"🔧 Health check result: {health_json}")
                
                text_content = TextContent(type="text", text=health_json)
                tool_result = ToolResult(content=[text_content])
                logger.info(f"🔧 Health check returning: {type(tool_result)}")
                return [tool_result]
                
        except Exception as e:
            error_msg = f"Health check failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="count_nodes")
    async def count_nodes() -> list[ToolResult]:
        """Count total nodes in the database."""
        logger.info(f"🔧 count_nodes called")
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
                
                result_json = json.dumps(result_data)
                logger.info(f"🔧 Count nodes result: {result_json}")
                
                text_content = TextContent(type="text", text=result_json)
                tool_result = ToolResult(content=[text_content])
                logger.info(f"🔧 Count nodes returning ToolResult with content: {tool_result.content[0].text}")
                return [tool_result]
                
        except Exception as e:
            error_msg = f"Count nodes failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="simple_test")
    async def simple_test() -> list[ToolResult]:
        """Simple test that returns basic data."""
        logger.info(f"🔧 simple_test called")
        
        test_data = {
            "message": "Hello from MCP server!",
            "timestamp": str(asyncio.get_event_loop().time()),
            "test": True
        }
        
        result_json = json.dumps(test_data)
        logger.info(f"🔧 Simple test result: {result_json}")
        
        text_content = TextContent(type="text", text=result_json)
        tool_result = ToolResult(content=[text_content])
        logger.info(f"🔧 Simple test returning: {type(tool_result)} with {len(tool_result.content)} content items")
        logger.info(f"🔧 Content item 0 text: {tool_result.content[0].text}")
        
        return [tool_result]

    return mcp

async def run_mcp_server():
    """Main server function with detailed logging."""
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
        logger.info("✅ MCP server created with tools")
        
        # Use STDIO transport
        logger.info("🚀 Starting MCP server with STDIO transport")
        await mcp.run_async(transport="stdio")
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        import traceback
        traceback.print_exc()
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
