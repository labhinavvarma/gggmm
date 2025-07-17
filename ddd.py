# mcpserver.py

import json
import re
import logging
import asyncio
import signal
import sys
from typing import Any, Optional, Dict, List
from datetime import datetime
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult, TextContent
from fastmcp.server import FastMCP
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncTransaction
from neo4j.exceptions import Neo4jError, ServiceUnavailable, AuthError
from pydantic import Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("neo4j_mcp")

# Neo4j Configuration
NEO4J_URI = "neo4j://10.189.116.237:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Vkg5d$F!pLq2@9vRwE="
NEO4J_DATABASE = "connectiq"

# Server Configuration
MCP_HOST = "0.0.0.0"
MCP_PORT = 8000
MCP_PATH = "/messages/"

class Neo4jMCPServer:
    """Enhanced Neo4j MCP Server with better error handling and monitoring"""
    
    def __init__(self):
        self.driver: Optional[AsyncDriver] = None
        self.mcp_server: Optional[FastMCP] = None
        self.is_running = False
        self.connection_verified = False
        
    async def initialize_driver(self) -> bool:
        """Initialize Neo4j driver with connection validation"""
        try:
            logger.info("🔌 Connecting to Neo4j database...")
            
            self.driver = AsyncGraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            
            # Verify connection
            await self.verify_connection()
            logger.info("✅ Neo4j connection established successfully")
            return True
            
        except AuthError as e:
            logger.error(f"❌ Neo4j authentication failed: {e}")
            return False
        except ServiceUnavailable as e:
            logger.error(f"❌ Neo4j service unavailable: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Neo4j driver: {e}")
            return False
    
    async def verify_connection(self):
        """Verify Neo4j connection and database access"""
        if not self.driver:
            raise Exception("Driver not initialized")
        
        try:
            async with self.driver.session(database=NEO4J_DATABASE) as session:
                result = await session.run("RETURN 1 as test")
                record = await result.single()
                if record and record["test"] == 1:
                    self.connection_verified = True
                    logger.info(f"✅ Database '{NEO4J_DATABASE}' connection verified")
                else:
                    raise Exception("Connection test failed")
        except Exception as e:
            logger.error(f"❌ Connection verification failed: {e}")
            raise
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Get database information for diagnostics"""
        if not self.driver or not self.connection_verified:
            return {"error": "Database not connected"}
        
        try:
            async with self.driver.session(database=NEO4J_DATABASE) as session:
                # Get node counts
                node_result = await session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = (await node_result.single())["node_count"]
                
                # Get relationship counts
                rel_result = await session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count = (await rel_result.single())["rel_count"]
                
                # Get node labels
                labels_result = await session.run("CALL db.labels()")
                labels = [record["label"] for record in await labels_result.data()]
                
                # Get relationship types
                types_result = await session.run("CALL db.relationshipTypes()")
                rel_types = [record["relationshipType"] for record in await types_result.data()]
                
                return {
                    "database": NEO4J_DATABASE,
                    "node_count": node_count,
                    "relationship_count": rel_count,
                    "node_labels": labels,
                    "relationship_types": rel_types,
                    "connection_time": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {"error": str(e)}
    
    def _is_write_query(self, query: str) -> bool:
        """Determine if query is a write operation"""
        write_patterns = [
            r"\bCREATE\b", r"\bMERGE\b", r"\bDELETE\b", 
            r"\bSET\b", r"\bREMOVE\b", r"\bDROP\b",
            r"\bDETACH\s+DELETE\b"
        ]
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in write_patterns)
    
    def _sanitize_query(self, query: str) -> str:
        """Basic query sanitization"""
        # Remove potentially dangerous operations
        dangerous_patterns = [
            r"\bCALL\s+dbms\.", r"\bCALL\s+db\.",
            r"\bLOAD\s+CSV\b", r"\bUSING\s+PERIODIC\s+COMMIT\b"
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                raise ToolError(f"Query contains restricted operation: {pattern}")
        
        return query.strip()
    
    async def _execute_read(self, tx: AsyncTransaction, query: str, params: Dict[str, Any]) -> str:
        """Execute read query and return formatted results"""
        try:
            result = await tx.run(query, params or {})
            records = await result.data()
            
            # Format results for better readability
            if not records:
                return json.dumps({"message": "No results found", "count": 0})
            
            return json.dumps({
                "data": records,
                "count": len(records),
                "query_type": "READ"
            }, default=str, indent=2)
            
        except Exception as e:
            logger.error(f"Read query error: {e}")
            raise ToolError(f"Read query failed: {str(e)}")
    
    async def _execute_write(self, tx: AsyncTransaction, query: str, params: Dict[str, Any]) -> str:
        """Execute write query and return summary"""
        try:
            result = await tx.run(query, params or {})
            summary = await result.consume()
            
            counters = summary.counters
            return json.dumps({
                "summary": {
                    "nodes_created": counters.nodes_created,
                    "nodes_deleted": counters.nodes_deleted,
                    "relationships_created": counters.relationships_created,
                    "relationships_deleted": counters.relationships_deleted,
                    "properties_set": counters.properties_set,
                    "labels_added": counters.labels_added,
                    "labels_removed": counters.labels_removed
                },
                "query_type": "WRITE",
                "execution_time": summary.result_available_after + summary.result_consumed_after
            }, default=str, indent=2)
            
        except Exception as e:
            logger.error(f"Write query error: {e}")
            raise ToolError(f"Write query failed: {str(e)}")
    
    def create_mcp_server(self) -> FastMCP:
        """Create and configure MCP server with tools"""
        mcp = FastMCP("neo4j-cypher", stateless_http=True)
        
        @mcp.tool(name="read_neo4j_cypher")
        async def read_neo4j_cypher(
            query: str = Field(..., description="Cypher query to execute (READ operations only)"),
            params: Optional[Dict[str, Any]] = Field(None, description="Query parameters")
        ) -> List[ToolResult]:
            """Execute read-only Cypher queries against Neo4j database"""
            
            if not self.connection_verified:
                raise ToolError("Database connection not available")
            
            # Sanitize and validate query
            clean_query = self._sanitize_query(query)
            
            if self._is_write_query(clean_query):
                raise ToolError("Write operations not allowed in read tool. Use write_neo4j_cypher instead.")
            
            try:
                logger.info(f"Executing READ query: {clean_query[:100]}...")
                
                async with self.driver.session(database=NEO4J_DATABASE) as session:
                    result = await session.execute_read(self._execute_read, clean_query, params)
                    return [ToolResult(content=[TextContent(type="text", text=result)])]
                    
            except Neo4jError as e:
                error_msg = f"Neo4j error: {str(e)}"
                logger.error(error_msg)
                raise ToolError(error_msg)
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(error_msg)
                raise ToolError(error_msg)
        
        @mcp.tool(name="write_neo4j_cypher")
        async def write_neo4j_cypher(
            query: str = Field(..., description="Cypher query to execute (WRITE operations only)"),
            params: Optional[Dict[str, Any]] = Field(None, description="Query parameters")
        ) -> List[ToolResult]:
            """Execute write Cypher queries against Neo4j database"""
            
            if not self.connection_verified:
                raise ToolError("Database connection not available")
            
            # Sanitize and validate query
            clean_query = self._sanitize_query(query)
            
            if not self._is_write_query(clean_query):
                raise ToolError("Only write operations allowed in write tool. Use read_neo4j_cypher for queries.")
            
            try:
                logger.info(f"Executing WRITE query: {clean_query[:100]}...")
                
                async with self.driver.session(database=NEO4J_DATABASE) as session:
                    result = await session.execute_write(self._execute_write, clean_query, params)
                    return [ToolResult(content=[TextContent(type="text", text=result)])]
                    
            except Neo4jError as e:
                error_msg = f"Neo4j error: {str(e)}"
                logger.error(error_msg)
                raise ToolError(error_msg)
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(error_msg)
                raise ToolError(error_msg)
        
        @mcp.tool(name="get_database_schema")
        async def get_database_schema() -> List[ToolResult]:
            """Get Neo4j database schema information"""
            
            if not self.connection_verified:
                raise ToolError("Database connection not available")
            
            try:
                schema_info = await self.get_database_info()
                return [ToolResult(content=[TextContent(type="text", text=json.dumps(schema_info, indent=2))])]
            except Exception as e:
                error_msg = f"Failed to get schema: {str(e)}"
                logger.error(error_msg)
                raise ToolError(error_msg)
        
        @mcp.tool(name="health_check")
        async def health_check() -> List[ToolResult]:
            """Check server and database health"""
            
            health_status = {
                "server_status": "running" if self.is_running else "stopped",
                "database_connected": self.connection_verified,
                "timestamp": datetime.now().isoformat()
            }
            
            if self.connection_verified:
                try:
                    db_info = await self.get_database_info()
                    health_status.update(db_info)
                except Exception as e:
                    health_status["database_error"] = str(e)
            
            return [ToolResult(content=[TextContent(type="text", text=json.dumps(health_status, indent=2))])]
        
        return mcp
    
    async def start_server(self):
        """Start the MCP server"""
        try:
            # Initialize Neo4j connection
            if not await self.initialize_driver():
                logger.error("❌ Failed to initialize database connection")
                return False
            
            # Create MCP server
            self.mcp_server = self.create_mcp_server()
            logger.info("🚀 Starting MCP server...")
            
            # Set running flag
            self.is_running = True
            
            # Start server
            await self.mcp_server.run_sse_async(
                host=MCP_HOST,
                port=MCP_PORT,
                path=MCP_PATH
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to start MCP server: {e}")
            self.is_running = False
            return False
    
    async def stop_server(self):
        """Stop the MCP server and close connections"""
        logger.info("🛑 Stopping MCP server...")
        
        self.is_running = False
        
        if self.driver:
            await self.driver.close()
            logger.info("✅ Neo4j driver closed")
        
        logger.info("✅ MCP server stopped")
    
    def setup_signal_handlers(self, enable=True):
        """Setup signal handlers for graceful shutdown (only in main thread)"""
        if not enable:
            return
            
        try:
            # Check if we're in the main thread
            import threading
            if threading.current_thread() is not threading.main_thread():
                logger.info("⚠️  Skipping signal handlers (not in main thread)")
                return
                
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, shutting down...")
                asyncio.create_task(self.stop_server())
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            logger.info("✅ Signal handlers registered")
        except Exception as e:
            logger.warning(f"⚠️  Could not setup signal handlers: {e}")
            # This is okay, signal handlers are optional

# Global server instance
_server_instance = None

async def run_mcp_server(enable_signals=True):
    """Main function to run the MCP server"""
    global _server_instance
    
    logger.info("🌟 Starting Neo4j MCP Server...")
    
    _server_instance = Neo4jMCPServer()
    _server_instance.setup_signal_handlers(enable=enable_signals)
    
    try:
        await _server_instance.start_server()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise  # Re-raise for caller to handle
    finally:
        if _server_instance:
            await _server_instance.stop_server()

async def get_server_status():
    """Get current server status"""
    global _server_instance
    
    if not _server_instance:
        return {"status": "not_initialized"}
    
    return {
        "status": "running" if _server_instance.is_running else "stopped",
        "database_connected": _server_instance.connection_verified,
        "database": NEO4J_DATABASE
    }

# CLI entry point
if __name__ == "__main__":
    print("🌟 Starting Neo4j MCP Server...")
    print(f"📡 Server will run on: http://{MCP_HOST}:{MCP_PORT}{MCP_PATH}")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        # Check if we're in an existing event loop (like Jupyter)
        try:
            loop = asyncio.get_running_loop()
            print("⚠️  Running in existing event loop")
            # If we're in an existing loop, create a new one
            import threading
            
            def run_server():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                # Enable signals only when running directly (in main thread)
                is_main_thread = threading.current_thread() is threading.main_thread()
                new_loop.run_until_complete(run_mcp_server(enable_signals=is_main_thread))
            
            thread = threading.Thread(target=run_server)
            thread.daemon = True
            thread.start()
            thread.join()
            
        except RuntimeError:
            # No existing loop, run normally (enable signals)
            asyncio.run(run_mcp_server(enable_signals=True))
            
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
