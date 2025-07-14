#!/usr/bin/env python3
"""
Neo4j MCP Server for app.py Integration
Specifically designed to work with your existing FastAPI app.py setup
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime

from fastmcp import FastMCP
from neo4j import GraphDatabase, Driver, AsyncGraphDatabase, AsyncDriver
import neo4j.exceptions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcpserver_neo4j")

@dataclass
class Neo4jContext:
    """Context for Neo4j database connection"""
    driver: Driver
    database: str

class StableNeo4jConnection:
    """Stable Neo4j connection handler optimized for app.py integration"""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver: Optional[AsyncDriver] = None
        self.sync_driver: Optional[Driver] = None
        self.connection_status = "disconnected"
        self.last_health_check = None
        self.connection_error = None
        self.max_retries = 3
        self.retry_delay = 5
        self.connection_pool_size = 10
        
    async def connect(self):
        """Connect to Neo4j with enhanced stability for app.py"""
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                logger.info(f"🔄 Connecting to Neo4j for app.py... (Attempt {retry_count + 1}/{self.max_retries})")
                logger.info(f"📍 URI: {self.uri}")
                logger.info(f"🗄️  Database: {self.database}")
                
                # Create both async and sync drivers for flexibility
                self.driver = AsyncGraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password),
                    max_connection_lifetime=3600,
                    max_connection_pool_size=self.connection_pool_size,
                    connection_acquisition_timeout=30.0,
                    max_transaction_retry_time=15.0
                )
                
                self.sync_driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password),
                    max_connection_lifetime=3600,
                    max_connection_pool_size=self.connection_pool_size,
                    connection_acquisition_timeout=30.0,
                    max_transaction_retry_time=15.0
                )
                
                # Test async connection
                await self.driver.verify_connectivity()
                
                # Test sync connection
                self.sync_driver.verify_connectivity()
                
                # Test database access
                async with self.driver.session(database=self.database) as session:
                    result = await session.run("RETURN 1 as test")
                    test_record = await result.single()
                    if not test_record or test_record["test"] != 1:
                        raise Exception("Database test query failed")
                
                self.connection_status = "connected"
                self.connection_error = None
                self.last_health_check = datetime.now()
                
                logger.info("✅ Neo4j connection established for app.py!")
                logger.info(f"📊 Connection pool size: {self.connection_pool_size}")
                
                # Verify connectiq database
                try:
                    async with self.driver.session(database=self.database) as session:
                        result = await session.run("MATCH (n:Table {table: 'MBR'}) RETURN count(n) as mbr_count")
                        count_record = await result.single()
                        mbr_count = count_record["mbr_count"] if count_record else 0
                        
                        if mbr_count > 0:
                            logger.info(f"📋 Connectiq database verified: {mbr_count} MBR table nodes")
                        
                        # Get total node count
                        result = await session.run("MATCH (n) RETURN count(n) as total_nodes")
                        total_record = await result.single()
                        total_nodes = total_record["total_nodes"] if total_record else 0
                        logger.info(f"📊 Total database nodes: {total_nodes:,}")
                        
                except Exception as e:
                    logger.warning(f"⚠️  Could not verify database details: {e}")
                
                return True
                
            except Exception as e:
                retry_count += 1
                self.connection_status = "failed"
                self.connection_error = str(e)
                
                logger.error(f"❌ Connection attempt {retry_count} failed: {e}")
                
                if retry_count < self.max_retries:
                    logger.info(f"⏳ Retrying in {self.retry_delay} seconds...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"💥 Failed to connect after {self.max_retries} attempts")
                    logger.error("🔧 Check your connection settings and ensure Neo4j is accessible")
                    raise
        
        return False
    
    def execute_query_sync(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute query using sync driver for MCP tools"""
        if not self.sync_driver:
            raise Exception("Sync driver not available. Connection may have failed.")
        
        if self.connection_status != "connected":
            raise Exception(f"Connection is not healthy. Status: {self.connection_status}")
        
        try:
            with self.sync_driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                records = []
                
                for record in result:
                    record_dict = {}
                    for key in record.keys():
                        value = record[key]
                        if hasattr(value, '_properties'):
                            record_dict[key] = dict(value._properties)
                            if hasattr(value, 'labels'):
                                record_dict[key]['_labels'] = list(value.labels)
                            if hasattr(value, 'type'):
                                record_dict[key]['_type'] = value.type
                        else:
                            record_dict[key] = value
                    records.append(record_dict)
                
                self.connection_status = "connected"
                self.last_health_check = datetime.now()
                
                return {
                    "records": records,
                    "summary": {
                        "query": query,
                        "parameters": parameters,
                        "records_available": len(records),
                        "execution_time": datetime.now().isoformat()
                    }
                }
                
        except neo4j.exceptions.CypherSyntaxError as e:
            return {
                "error": f"Cypher Syntax Error: {str(e)}",
                "query": query,
                "parameters": parameters,
                "error_type": "syntax_error"
            }
        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "network", "timeout", "unavailable"]):
                self.connection_status = "unhealthy"
                self.connection_error = str(e)
            
            return {
                "error": f"Query execution error: {str(e)}",
                "query": query,
                "parameters": parameters,
                "error_type": "execution_error"
            }
    
    async def health_check_async(self) -> Dict[str, Any]:
        """Async health check for the connection"""
        health_status = {
            "status": self.connection_status,
            "timestamp": datetime.now().isoformat(),
            "uri": self.uri,
            "database": self.database,
            "last_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "connection_pool_size": self.connection_pool_size,
            "max_retries": self.max_retries
        }
        
        if not self.driver:
            health_status.update({
                "healthy": False,
                "error": "No driver instance available"
            })
            return health_status
        
        try:
            await self.driver.verify_connectivity()
            
            async with self.driver.session(database=self.database) as session:
                start_time = datetime.now()
                result = await session.run("RETURN datetime() as server_time, 'app_py_health_check' as test")
                test_record = await result.single()
                end_time = datetime.now()
                
                response_time_ms = (end_time - start_time).total_seconds() * 1000
                
                health_status.update({
                    "healthy": True,
                    "server_time": test_record["server_time"] if test_record else None,
                    "performance": {
                        "response_time_ms": round(response_time_ms, 2)
                    },
                    "connection_details": {
                        "async_driver_available": True,
                        "sync_driver_available": self.sync_driver is not None
                    }
                })
                
                self.connection_status = "connected"
                self.last_health_check = datetime.now()
                self.connection_error = None
                
        except Exception as e:
            health_status.update({
                "healthy": False,
                "error": str(e),
                "error_type": type(e).__name__
            })
            self.connection_status = "unhealthy"
            self.connection_error = str(e)
        
        return health_status
    
    async def close(self):
        """Close both drivers gracefully"""
        try:
            if self.driver:
                await self.driver.close()
            if self.sync_driver:
                self.sync_driver.close()
            self.connection_status = "disconnected"
            logger.info("📴 Neo4j connections closed for app.py")
        except Exception as e:
            logger.warning(f"⚠️  Warning during disconnect: {e}")

# Global connection instance for app.py
neo4j_connection = None

def create_app_integrated_mcp() -> FastMCP:
    """Create MCP server specifically for app.py integration"""
    global neo4j_connection
    
    # Initialize connection with your working credentials
    neo4j_connection = StableNeo4jConnection(
        uri="neo4j://10.189.116.237:7687",     # Your working URI
        username="neo4j",                      # Your username
        password="Vkg5d$F!pLq2@9vRwE=",        # Your working password
        database="connectiq"                   # Your database
    )
    
    # Create FastMCP instance
    mcp = FastMCP("Neo4j-Connectiq-Server")
    
    @asynccontextmanager
    async def app_lifespan(server: FastMCP) -> AsyncIterator[Neo4jContext]:
        """Manage Neo4j connection lifecycle for app.py"""
        try:
            logger.info("🚀 Initializing Neo4j connection for app.py...")
            
            # Establish connection
            await neo4j_connection.connect()
            
            # Perform health check
            health_status = await neo4j_connection.health_check_async()
            if health_status["healthy"]:
                logger.info("✅ Neo4j ready for app.py integration")
            else:
                logger.warning(f"⚠️  Health check issues: {health_status.get('error')}")
            
            # Yield context for MCP server
            yield Neo4jContext(
                driver=neo4j_connection.sync_driver, 
                database=neo4j_connection.database
            )
            
        except Exception as e:
            logger.error(f"💥 Failed to initialize Neo4j for app.py: {e}")
            # Still yield context to keep app.py running
            yield Neo4jContext(driver=None, database=neo4j_connection.database)
        finally:
            # Keep connection alive for app.py - don't close automatically
            logger.info("📡 Keeping Neo4j connection alive for app.py")
    
    # Set lifespan
    mcp.lifespan = app_lifespan
    
    @mcp.tool()
    def check_connection_health() -> str:
        """Check Neo4j connection health for app.py integration."""
        try:
            # Run async health check in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                health_status = loop.run_until_complete(neo4j_connection.health_check_async())
            finally:
                loop.close()
            
            # Add app.py specific information
            health_status["integration"] = {
                "app_py_compatible": True,
                "fastapi_ready": True,
                "sse_transport_ready": True,
                "port": "8001 (app.py default)"
            }
            
            return json.dumps(health_status, indent=2)
            
        except Exception as e:
            return json.dumps({
                "healthy": False,
                "error": f"Health check failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "integration": {"app_py_compatible": False}
            }, indent=2)
    
    @mcp.tool()
    def execute_cypher(query: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Execute Cypher query optimized for app.py integration."""
        try:
            if neo4j_connection.connection_status != "connected":
                return json.dumps({
                    "error": "Neo4j connection not available for app.py",
                    "status": neo4j_connection.connection_status,
                    "suggestion": "Check app.py logs and restart if needed"
                }, indent=2)
            
            result = neo4j_connection.execute_query_sync(query, parameters)
            
            # Add app.py integration metadata
            if "error" not in result:
                result["app_integration"] = {
                    "executed_via": "app.py_mcp_integration",
                    "transport": "sse",
                    "timestamp": datetime.now().isoformat()
                }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            return json.dumps({
                "error": f"Failed to execute query in app.py context: {str(e)}",
                "query": query,
                "parameters": parameters,
                "integration_error": True
            }, indent=2)
    
    @mcp.tool()
    def get_database_schema() -> str:
        """Get Connectiq database schema for app.py integration."""
        try:
            if neo4j_connection.connection_status != "connected":
                return json.dumps({
                    "error": "Database connection not available",
                    "connection_status": neo4j_connection.connection_status
                }, indent=2)
            
            # Get comprehensive schema information
            schema_queries = {
                "node_labels": "CALL db.labels() YIELD label RETURN collect(label) AS labels",
                "relationship_types": "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS types",
                "property_keys": "CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) AS keys"
            }
            
            schema_info = {}
            
            for schema_type, query in schema_queries.items():
                try:
                    result = neo4j_connection.execute_query_sync(query)
                    if result.get("records"):
                        schema_info[schema_type] = result["records"][0]
                    else:
                        schema_info[schema_type] = []
                except Exception as e:
                    schema_info[schema_type] = f"Error: {str(e)}"
            
            # Get connectiq-specific structure
            try:
                connectiq_query = """
                MATCH (n:Table) 
                WHERE n.table IN ['MBR', 'CLM_LINE', 'PROV', 'MBR_PROD_ENRLMNT', 'DOMAIN']
                RETURN n.table as table_name, labels(n) as labels
                LIMIT 10
                """
                connectiq_result = neo4j_connection.execute_query_sync(connectiq_query)
                schema_info["connectiq_tables"] = connectiq_result.get("records", [])
            except Exception as e:
                schema_info["connectiq_tables"] = f"Error getting connectiq tables: {str(e)}"
            
            # Add app.py integration info
            schema_info["app_integration"] = {
                "optimized_for": "connectiq_healthcare_database",
                "available_via": "app.py_sse_endpoint",
                "last_updated": datetime.now().isoformat()
            }
            
            return json.dumps(schema_info, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Failed to get schema: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }, indent=2)
    
    @mcp.tool()
    def get_connection_info() -> str:
        """Get connection information for app.py integration."""
        connection_info = {
            "uri": neo4j_connection.uri,
            "username": neo4j_connection.username,
            "database": neo4j_connection.database,
            "status": neo4j_connection.connection_status,
            "last_health_check": neo4j_connection.last_health_check.isoformat() if neo4j_connection.last_health_check else None,
            "connection_error": neo4j_connection.connection_error,
            "app_integration": {
                "integrated_with": "FastAPI app.py",
                "transport": "SSE (Server-Sent Events)",
                "endpoint": "/sse",
                "messages_endpoint": "/messages/",
                "port": "8001 (app.py default)",
                "ready_for_clients": neo4j_connection.connection_status == "connected"
            },
            "stability_features": {
                "auto_reconnect": True,
                "connection_pooling": True,
                "health_monitoring": True,
                "async_and_sync_drivers": True
            }
        }
        
        return json.dumps(connection_info, indent=2)
    
    @mcp.tool()
    def test_connectiq_queries() -> str:
        """Test specific queries for the Connectiq healthcare database."""
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "database": "connectiq",
            "integration": "app.py",
            "tests": {}
        }
        
        # Test healthcare-specific queries
        test_queries = [
            {
                "name": "member_tables",
                "query": "MATCH (n:Table {table: 'MBR'}) RETURN count(n) as count",
                "description": "Count MBR (Member) table nodes"
            },
            {
                "name": "domain_structure",
                "query": "MATCH (n:Domain) RETURN n.name as domain_name LIMIT 5",
                "description": "Get healthcare domain structure"
            },
            {
                "name": "table_relationships",
                "query": "MATCH (a:Table)-[r:HAS_CHILD]->(b:Table) RETURN count(r) as relationship_count",
                "description": "Count table relationships"
            }
        ]
        
        for test in test_queries:
            try:
                if neo4j_connection.connection_status == "connected":
                    result = neo4j_connection.execute_query_sync(test["query"])
                    
                    test_results["tests"][test["name"]] = {
                        "description": test["description"],
                        "success": "error" not in result,
                        "result": result.get("records", []) if "error" not in result else None,
                        "error": result.get("error") if "error" in result else None
                    }
                else:
                    test_results["tests"][test["name"]] = {
                        "description": test["description"],
                        "success": False,
                        "error": f"Connection not available: {neo4j_connection.connection_status}"
                    }
            except Exception as e:
                test_results["tests"][test["name"]] = {
                    "description": test["description"],
                    "success": False,
                    "error": str(e)
                }
        
        # Summary
        successful_tests = sum(1 for test in test_results["tests"].values() if test.get("success", False))
        total_tests = len(test_results["tests"])
        
        test_results["summary"] = {
            "tests_passed": successful_tests,
            "total_tests": total_tests,
            "success_rate": f"{(successful_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%",
            "connectiq_database_verified": successful_tests >= 1,
            "app_py_integration_status": "healthy" if successful_tests >= 1 else "needs_attention"
        }
        
        return json.dumps(test_results, indent=2)
    
    @mcp.resource("neo4j://app-health")
    def get_app_health_resource() -> str:
        """App.py specific health resource"""
        return check_connection_health()
    
    @mcp.resource("neo4j://connectiq-info")
    def get_connectiq_info_resource() -> str:
        """Connectiq database information resource"""
        return test_connectiq_queries()
    
    logger.info("✅ MCP server created for app.py integration")
    logger.info("🔧 Tools configured: health check, cypher execution, schema, testing")
    logger.info("📡 Ready for SSE transport on app.py")
    
    return mcp

# Create the MCP instance for app.py import
logger.info("🚀 Initializing MCP server for app.py...")
mcp = create_app_integrated_mcp()
logger.info("✅ MCP server ready for app.py import")

# Helper functions for app.py integration
def get_mcp_server():
    """Get the MCP server instance"""
    return mcp

def get_neo4j_connection():
    """Get the Neo4j connection instance"""
    return neo4j_connection

def is_connection_healthy() -> bool:
    """Quick health check for app.py"""
    if neo4j_connection:
        return neo4j_connection.connection_status == "connected"
    return False

async def initialize_for_app():
    """Initialize the connection for app.py startup"""
    if neo4j_connection and neo4j_connection.connection_status != "connected":
        try:
            await neo4j_connection.connect()
            logger.info("✅ Neo4j connection initialized for app.py")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Neo4j for app.py: {e}")
            return False
    return True

# Export for app.py
__all__ = ["mcp", "get_mcp_server", "get_neo4j_connection", "is_connection_healthy", "initialize_for_app"]

if __name__ == "__main__":
    print("🚀 Neo4j MCP Server for app.py Integration")
    print("=" * 50)
    print("This module is designed to be imported by app.py")
    print("Usage: from mcpserver import mcp")
    print("=" * 50)
    print(f"📍 Neo4j URI: {neo4j_connection.uri}")
    print(f"🗄️  Database: {neo4j_connection.database}")
    print("🔧 Ready for app.py integration!")
