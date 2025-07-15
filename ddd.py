# mcpserver.py - COMPLETELY FIXED VERSION - Handles Neo4j async results properly

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
    """Execute read query with proper async handling."""
    try:
        result = await tx.run(query, params or {})
        
        # Collect all records using async iteration
        records = []
        async for record in result:
            records.append(record.data())
        
        return json.dumps(records, default=str)
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
    """Create MCP server with COMPLETELY FIXED tool return values."""
    mcp = FastMCP("neo4j-cypher")

    @mcp.tool(name="simple_test")
    async def simple_test() -> str:
        """Simple test tool."""
        test_data = {
            "status": "success",
            "message": "MCP server is working correctly!",
            "tool": "simple_test"
        }
        return json.dumps(test_data)

    @mcp.tool(name="read_neo4j_cypher")
    async def read_neo4j_cypher(
        query: str = Field(..., description="Cypher query to execute"),
        params: Optional[dict[str, Any]] = Field(None, description="Query parameters")
    ) -> str:
        """Execute read-only Cypher queries."""
        if _is_write_query(query):
            error_msg = "Write-type queries not allowed in read tool."
            logger.warning(f"Rejected write query: {query}")
            raise ToolError(error_msg)
        
        try:
            async with driver.session(database=database) as session:
                result_text = await session.execute_read(_read, query, params)
                logger.info(f"Read query executed successfully")
                return result_text
        except Exception as e:
            error_msg = f"Read query error: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="write_neo4j_cypher")
    async def write_neo4j_cypher(
        query: str = Field(..., description="Cypher write query to execute"),
        params: Optional[dict[str, Any]] = Field(None, description="Query parameters")
    ) -> str:
        """Execute write Cypher queries."""
        if not _is_write_query(query):
            error_msg = "Only write-type queries allowed in write tool."
            logger.warning(f"Rejected read query in write tool: {query}")
            raise ToolError(error_msg)
        
        try:
            async with driver.session(database=database) as session:
                result_text = await session.execute_write(_write, query, params)
                logger.info(f"Write query executed successfully")
                return result_text
        except Exception as e:
            error_msg = f"Write query error: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="health_check")
    async def health_check() -> str:
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
                
                return json.dumps(health_info)
        except Exception as e:
            error_msg = f"Health check failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="count_nodes")
    async def count_nodes() -> str:
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
                
                return json.dumps(result_data)
        except Exception as e:
            error_msg = f"Count nodes failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="list_labels")
    async def list_labels() -> str:
        """List all node labels in the database - COMPLETELY FIXED."""
        try:
            async with driver.session(database=database) as session:
                # Use the _read function which properly handles async iteration
                result_text = await session.execute_read(_read, "CALL db.labels()", {})
                
                # Parse the JSON result to extract labels
                raw_data = json.loads(result_text)
                labels = [record.get("label", "") for record in raw_data if record.get("label")]
                
                result_data = {
                    "labels": labels,
                    "count": len(labels),
                    "database": database,
                    "status": "success"
                }
                
                return json.dumps(result_data)
        except Exception as e:
            error_msg = f"List labels failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="list_relationships")
    async def list_relationships() -> str:
        """List all relationship types in the database."""
        try:
            async with driver.session(database=database) as session:
                # Use the _read function which properly handles async iteration
                result_text = await session.execute_read(_read, "CALL db.relationshipTypes()", {})
                
                # Parse the JSON result to extract relationship types
                raw_data = json.loads(result_text)
                rel_types = [record.get("relationshipType", "") for record in raw_data if record.get("relationshipType")]
                
                result_data = {
                    "relationship_types": rel_types,
                    "count": len(rel_types),
                    "database": database,
                    "status": "success"
                }
                
                return json.dumps(result_data)
        except Exception as e:
            error_msg = f"List relationships failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

    @mcp.tool(name="database_summary")
    async def database_summary() -> str:
        """Get a comprehensive database summary."""
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
                
                # Get labels using the fixed method
                labels_result = await session.execute_read(_read, "CALL db.labels()", {})
                labels_data = json.loads(labels_result)
                labels = [record.get("label", "") for record in labels_data if record.get("label")]
                
                # Get relationship types using the fixed method
                rel_types_result = await session.execute_read(_read, "CALL db.relationshipTypes()", {})
                rel_types_data = json.loads(rel_types_result)
                rel_types = [record.get("relationshipType", "") for record in rel_types_data if record.get("relationshipType")]
                
                summary_data = {
                    "database": database,
                    "node_count": node_count,
                    "relationship_count": rel_count,
                    "label_count": len(labels),
                    "labels": labels,
                    "relationship_type_count": len(rel_types),
                    "relationship_types": rel_types,
                    "status": "summary_complete"
                }
                
                return json.dumps(summary_data)
        except Exception as e:
            error_msg = f"Database summary failed: {str(e)}"
            logger.error(error_msg)
            raise ToolError(error_msg)

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
        logger.info("✅ MCP server created with completely fixed tools")
        
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
