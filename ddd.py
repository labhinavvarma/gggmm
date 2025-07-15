
#!/usr/bin/env python3
"""
Standalone Neo4j MCP Server for ConnectIQ Database
Can be run independently to show connection stats or imported by app.py
"""

import asyncio
import json
import logging
import sys
from typing import Dict, Any, Optional
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
    """Stable Neo4j connection handler with display capabilities"""
    
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
        self.database_stats = {}
        
    async def connect(self, show_stats: bool = False):
        """Connect to Neo4j and optionally display database statistics"""
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                if show_stats:
                    print(f"🔄 Connecting to Neo4j... (Attempt {retry_count + 1}/{self.max_retries})")
                    print(f"📍 URI: {self.uri}")
                    print(f"🗄️  Database: {self.database}")
                else:
                    logger.info(f"🔄 Connecting to Neo4j... (Attempt {retry_count + 1}/{self.max_retries})")
                    logger.info(f"📍 URI: {self.uri}")
                    logger.info(f"🗄️  Database: {self.database}")
                
                # Create both async and sync drivers
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
                
                # Test connections
                await self.driver.verify_connectivity()
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
                
                if show_stats:
                    print("✅ Neo4j connection established!")
                    print(f"📊 Connection pool size: {self.connection_pool_size}")
                else:
                    logger.info("✅ Neo4j connection established!")
                    logger.info(f"📊 Connection pool size: {self.connection_pool_size}")
                
                # Get and display database statistics
                await self._get_database_statistics(show_stats)
                
                return True
                
            except Exception as e:
                retry_count += 1
                self.connection_status = "failed"
                self.connection_error = str(e)
                
                if show_stats:
                    print(f"❌ Connection attempt {retry_count} failed: {e}")
                else:
                    logger.error(f"❌ Connection attempt {retry_count} failed: {e}")
                
                if retry_count < self.max_retries:
                    if show_stats:
                        print(f"⏳ Retrying in {self.retry_delay} seconds...")
                    else:
                        logger.info(f"⏳ Retrying in {self.retry_delay} seconds...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    if show_stats:
                        print(f"💥 Failed to connect after {self.max_retries} attempts")
                        print("🔧 Check your connection settings and ensure Neo4j is accessible")
                    else:
                        logger.error(f"💥 Failed to connect after {self.max_retries} attempts")
                        logger.error("🔧 Check your connection settings and ensure Neo4j is accessible")
                    raise
        
        return False
    
    async def _get_database_statistics(self, show_stats: bool = False):
        """Get and optionally display database statistics"""
        try:
            async with self.driver.session(database=self.database) as session:
                # Get total node count
                result = await session.run("MATCH (n) RETURN count(n) as total_nodes")
                total_record = await result.single()
                total_nodes = total_record["total_nodes"] if total_record else 0
                
                # Get total relationship count
                result = await session.run("MATCH ()-[r]->() RETURN count(r) as total_relationships")
                rel_record = await result.single()
                total_relationships = rel_record["total_relationships"] if rel_record else 0
                
                # Get node type distribution
                result = await session.run("""
                    MATCH (n) 
                    RETURN DISTINCT labels(n) as node_type, count(n) as count 
                    ORDER BY count DESC 
                    LIMIT 10
                """)
                node_types = []
                async for record in result:
                    node_types.append({
                        "labels": record["node_type"],
                        "count": record["count"]
                    })
                
                # Get relationship type distribution
                result = await session.run("""
                    MATCH ()-[r]->() 
                    RETURN type(r) as rel_type, count(r) as count 
                    ORDER BY count DESC 
                    LIMIT 5
                """)
                rel_types = []
                async for record in result:
                    rel_types.append({
                        "type": record["rel_type"],
                        "count": record["count"]
                    })
                
                # Check for ConnectIQ-specific structures
                result = await session.run("""
                    MATCH (n) 
                    WHERE any(label IN labels(n) WHERE label IN ['Table', 'Domain', 'Column', 'MBR', 'CLM_LINE', 'PROV'])
                    RETURN labels(n) as healthcare_type, count(n) as count
                    ORDER BY count DESC
                    LIMIT 5
                """)
                healthcare_structures = []
                async for record in result:
                    healthcare_structures.append({
                        "type": record["healthcare_type"],
                        "count": record["count"]
                    })
                
                # Store statistics
                self.database_stats = {
                    "total_nodes": total_nodes,
                    "total_relationships": total_relationships,
                    "node_types": node_types,
                    "relationship_types": rel_types,
                    "healthcare_structures": healthcare_structures,
                    "last_updated": datetime.now().isoformat()
                }
                
                # Display statistics if requested
                if show_stats:
                    self._display_database_statistics()
                else:
                    logger.info(f"📊 Database statistics loaded: {total_nodes:,} nodes, {total_relationships:,} relationships")
                        
        except Exception as e:
            error_msg = f"⚠️  Could not retrieve database statistics: {e}"
            if show_stats:
                print(error_msg)
            else:
                logger.warning(error_msg)
    
    def _display_database_statistics(self):
        """Display database statistics in a formatted way"""
        stats = self.database_stats
        
        print("\n" + "="*60)
        print("📊 CONNECTIQ DATABASE STATISTICS")
        print("="*60)
        
        print(f"📈 Total Nodes: {stats['total_nodes']:,}")
        print(f"🔗 Total Relationships: {stats['total_relationships']:,}")
        
        print(f"\n🏷️  Top Node Types:")
        for i, node_type in enumerate(stats['node_types'][:5], 1):
            labels = node_type['labels'] if node_type['labels'] else ['(No Label)']
            print(f"   {i}. {labels}: {node_type['count']:,} nodes")
        
        if stats['relationship_types']:
            print(f"\n🔗 Relationship Types:")
            for i, rel_type in enumerate(stats['relationship_types'], 1):
                print(f"   {i}. {rel_type['type']}: {rel_type['count']:,} relationships")
        
        if stats['healthcare_structures']:
            print(f"\n🏥 Healthcare Structures Found:")
            for i, structure in enumerate(stats['healthcare_structures'], 1):
                print(f"   {i}. {structure['type']}: {structure['count']:,} nodes")
        
        # Calculate data density
        if stats['total_nodes'] > 0:
            density = stats['total_relationships'] / stats['total_nodes']
            print(f"\n📊 Data Density: {density:.2f} relationships per node")
        
        print(f"\n⏰ Statistics Updated: {stats['last_updated']}")
        print("="*60)
    
    def execute_query_sync(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute query using sync driver"""
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
                        # Handle Neo4j node/relationship objects
                        if hasattr(value, '_properties'):
                            record_dict[key] = dict(value._properties)
                            if hasattr(value, 'labels'):
                                record_dict[key]['_labels'] = list(value.labels)
                            if hasattr(value, 'type'):
                                record_dict[key]['_type'] = value.type
                        else:
                            record_dict[key] = value
                    records.append(record_dict)
                
                # Get summary information
                summary = result.consume()
                
                self.connection_status = "connected"
                self.last_health_check = datetime.now()
                
                return {
                    "records": records,
                    "summary": {
                        "query": query,
                        "parameters": parameters,
                        "records_available": len(records),
                        "execution_time": datetime.now().isoformat(),
                        "counters": dict(summary.counters) if hasattr(summary, 'counters') else {}
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
            "max_retries": self.max_retries,
            "database_stats": self.database_stats
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
                result = await session.run("RETURN datetime() as server_time, 'health_check' as test")
                test_record = await result.single()
                end_time = datetime.now()
                
                response_time_ms = (end_time - start_time).total_seconds() * 1000
                
                health_status.update({
                    "healthy": True,
                    "server_time": str(test_record["server_time"]) if test_record else None,
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
            logger.info("📴 Neo4j connections closed")
        except Exception as e:
            logger.warning(f"⚠️  Warning during disconnect: {e}")

# Global connection instance
neo4j_connection = None

def create_mcp_server() -> FastMCP:
    """Create MCP server for ConnectIQ database"""
    global neo4j_connection
    
    # Initialize connection with your credentials
    neo4j_connection = StableNeo4jConnection(
        uri="neo4j://10.189.116.237:7687",
        username="neo4j",
        password="Vkg5d$F!pLq2@9vRwE=",
        database="connectiq"
    )
    
    # Create FastMCP instance
    mcp = FastMCP("Neo4j-ConnectIQ-Server")
    
    @asynccontextmanager
    async def app_lifespan(server: FastMCP) -> AsyncIterator[Neo4jContext]:
        """Manage Neo4j connection lifecycle"""
        try:
            logger.info("🚀 Initializing Neo4j MCP Server...")
            
            # Establish connection (don't show stats when used as import)
            await neo4j_connection.connect(show_stats=False)
            
            # Perform health check
            health_status = await neo4j_connection.health_check_async()
            if health_status["healthy"]:
                logger.info("✅ Neo4j ready for MCP integration")
            else:
                logger.warning(f"⚠️  Health check issues: {health_status.get('error')}")
            
            yield Neo4jContext(
                driver=neo4j_connection.sync_driver, 
                database=neo4j_connection.database
            )
            
        except Exception as e:
            logger.error(f"💥 Failed to initialize Neo4j: {e}")
            yield Neo4jContext(driver=None, database=neo4j_connection.database)
        finally:
            logger.info("📡 Keeping Neo4j connection alive")
    
    mcp.lifespan = app_lifespan
    
    # ===================
    # MCP TOOLS
    # ===================
    
    @mcp.tool()
    def check_connection_health() -> str:
        """Check Neo4j connection health and get database statistics."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                health_status = loop.run_until_complete(neo4j_connection.health_check_async())
            finally:
                loop.close()
            
            return json.dumps(health_status, indent=2, default=str)
            
        except Exception as e:
            return json.dumps({
                "healthy": False,
                "error": f"Health check failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }, indent=2)
    
    @mcp.tool()
    def execute_cypher(query: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Execute Cypher query on ConnectIQ database."""
        try:
            if neo4j_connection.connection_status != "connected":
                return json.dumps({
                    "error": "Neo4j connection not available",
                    "status": neo4j_connection.connection_status,
                    "suggestion": "Check connection and restart server if needed"
                }, indent=2)
            
            result = neo4j_connection.execute_query_sync(query, parameters)
            
            # Add execution metadata
            if "error" not in result:
                result["execution_metadata"] = {
                    "database": "connectiq",
                    "timestamp": datetime.now().isoformat()
                }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            return json.dumps({
                "error": f"Failed to execute query: {str(e)}",
                "query": query,
                "parameters": parameters
            }, indent=2)
    
    @mcp.tool()
    def get_database_schema() -> str:
        """Get comprehensive ConnectIQ database schema."""
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
            
            # Add database statistics
            schema_info["database_statistics"] = neo4j_connection.database_stats
            
            # Get ConnectIQ-specific structure
            try:
                connectiq_query = """
                MATCH (n) 
                WHERE any(label IN labels(n) WHERE label IN ['Table', 'Domain', 'Column', 'MBR', 'CLM_LINE', 'PROV'])
                RETURN labels(n) as node_type, count(n) as count
                ORDER BY count DESC
                LIMIT 20
                """
                connectiq_result = neo4j_connection.execute_query_sync(connectiq_query)
                schema_info["connectiq_structures"] = connectiq_result.get("records", [])
            except Exception as e:
                schema_info["connectiq_structures"] = f"Error: {str(e)}"
            
            schema_info["metadata"] = {
                "database": "connectiq",
                "generated_at": datetime.now().isoformat()
            }
            
            return json.dumps(schema_info, indent=2, default=str)
            
        except Exception as e:
            return json.dumps({
                "error": f"Failed to get schema: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }, indent=2)
    
    @mcp.tool()
    def get_database_statistics() -> str:
        """Get detailed ConnectIQ database statistics and metrics."""
        try:
            if neo4j_connection.connection_status != "connected":
                return json.dumps({
                    "error": "Database connection not available",
                    "connection_status": neo4j_connection.connection_status
                }, indent=2)
            
            # Get current statistics
            stats = neo4j_connection.database_stats.copy()
            
            # Add additional metrics
            additional_queries = [
                {
                    "name": "property_distribution",
                    "query": "MATCH (n) WHERE size(keys(n)) > 0 WITH keys(n) as props UNWIND props as prop RETURN prop, count(prop) as frequency ORDER BY frequency DESC LIMIT 10",
                    "description": "Most common properties"
                },
                {
                    "name": "connectivity_stats", 
                    "query": "MATCH (n) WITH n, size((n)--()) as degree WHERE degree > 0 RETURN min(degree) as min_connections, max(degree) as max_connections, avg(degree) as avg_connections, count(n) as connected_nodes",
                    "description": "Node connectivity statistics"
                }
            ]
            
            for query_info in additional_queries:
                try:
                    result = neo4j_connection.execute_query_sync(query_info["query"])
                    stats[query_info["name"]] = {
                        "description": query_info["description"],
                        "data": result.get("records", [])
                    }
                except Exception as e:
                    stats[query_info["name"]] = {
                        "description": query_info["description"],
                        "error": str(e)
                    }
            
            # Add database health metrics
            stats["connection_info"] = {
                "status": neo4j_connection.connection_status,
                "uri": neo4j_connection.uri,
                "database": neo4j_connection.database,
                "last_health_check": neo4j_connection.last_health_check.isoformat() if neo4j_connection.last_health_check else None
            }
            
            return json.dumps(stats, indent=2, default=str)
            
        except Exception as e:
            return json.dumps({
                "error": f"Failed to get database statistics: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }, indent=2)
    
    @mcp.tool()
    def explore_healthcare_data() -> str:
        """Explore ConnectIQ healthcare data patterns and structures."""
        try:
            if neo4j_connection.connection_status != "connected":
                return json.dumps({
                    "error": "Database connection not available"
                }, indent=2)
            
            exploration_results = {
                "database": "connectiq",
                "exploration_timestamp": datetime.now().isoformat(),
                "healthcare_analysis": {}
            }
            
            # Healthcare-specific exploration queries
            healthcare_queries = [
                {
                    "name": "member_analysis",
                    "query": "MATCH (n) WHERE any(label IN labels(n) WHERE label CONTAINS 'MBR' OR label CONTAINS 'Member') RETURN labels(n) as member_types, count(n) as count ORDER BY count DESC",
                    "description": "Member data analysis"
                },
                {
                    "name": "claims_analysis",
                    "query": "MATCH (n) WHERE any(label IN labels(n) WHERE label CONTAINS 'CLM' OR label CONTAINS 'Claim') RETURN labels(n) as claim_types, count(n) as count ORDER BY count DESC",
                    "description": "Claims data analysis"
                },
                {
                    "name": "provider_analysis",
                    "query": "MATCH (n) WHERE any(label IN labels(n) WHERE label CONTAINS 'PROV' OR label CONTAINS 'Provider') RETURN labels(n) as provider_types, count(n) as count ORDER BY count DESC",
                    "description": "Provider data analysis"
                },
                {
                    "name": "table_structures",
                    "query": "MATCH (n:Table) RETURN n.table as table_name, count(n) as instances ORDER BY instances DESC LIMIT 10",
                    "description": "Healthcare table structures"
                }
            ]
            
            for exploration in healthcare_queries:
                try:
                    result = neo4j_connection.execute_query_sync(exploration["query"])
                    exploration_results["healthcare_analysis"][exploration["name"]] = {
                        "description": exploration["description"],
                        "data": result.get("records", []),
                        "success": "error" not in result
                    }
                except Exception as e:
                    exploration_results["healthcare_analysis"][exploration["name"]] = {
                        "description": exploration["description"],
                        "error": str(e),
                        "success": False
                    }
            
            # Add summary insights
            successful_analyses = sum(1 for analysis in exploration_results["healthcare_analysis"].values() if analysis.get("success", False))
            exploration_results["insights"] = {
                "successful_analyses": successful_analyses,
                "healthcare_data_richness": "high" if successful_analyses >= 3 else "moderate" if successful_analyses >= 2 else "low",
                "ready_for_healthcare_analytics": successful_analyses >= 2
            }
            
            return json.dumps(exploration_results, indent=2, default=str)
            
        except Exception as e:
            return json.dumps({
                "error": f"Failed to explore healthcare data: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }, indent=2)
    
    @mcp.tool()
    def test_database_queries() -> str:
        """Test various database queries to verify ConnectIQ functionality."""
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "database": "connectiq",
            "tests": {}
        }
        
        # Test queries
        test_queries = [
            {
                "name": "basic_connectivity",
                "query": "RETURN 'ConnectIQ Test' as message, datetime() as timestamp",
                "description": "Basic connectivity test"
            },
            {
                "name": "node_count",
                "query": "MATCH (n) RETURN count(n) as total_nodes",
                "description": "Total node count"
            },
            {
                "name": "relationship_count",
                "query": "MATCH ()-[r]->() RETURN count(r) as total_relationships",
                "description": "Total relationship count"
            },
            {
                "name": "sample_data",
                "query": "MATCH (n) RETURN labels(n) as node_type, keys(n) as properties LIMIT 3",
                "description": "Sample data inspection"
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
            "database_verified": successful_tests >= 2,
            "status": "healthy" if successful_tests >= 2 else "needs_attention"
        }
        
        return json.dumps(test_results, indent=2, default=str)
    
    logger.info("✅ MCP server created with ConnectIQ tools")
    logger.info("🔧 Available tools: health, cypher, schema, statistics, exploration, testing")
    
    return mcp

# Create the MCP instance
logger.info("🚀 Initializing ConnectIQ MCP server...")
mcp = create_mcp_server()
logger.info("✅ MCP server ready")

# Helper functions for external use
def get_mcp_server():
    """Get the MCP server instance"""
    return mcp

def get_neo4j_connection():
    """Get the Neo4j connection instance"""
    return neo4j_connection

def is_connection_healthy() -> bool:
    """Quick health check"""
    if neo4j_connection:
        return neo4j_connection.connection_status == "connected"
    return False

async def initialize_connection(show_stats: bool = False):
    """Initialize the connection with optional stats display"""
    if neo4j_connection and neo4j_connection.connection_status != "connected":
        try:
            await neo4j_connection.connect(show_stats=show_stats)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Neo4j: {e}")
            return False
    return True

# Export for app.py
__all__ = ["mcp", "get_mcp_server", "get_neo4j_connection", "is_connection_healthy", "initialize_connection"]

# Main execution for standalone mode
async def main():
    """Main function for standalone execution"""
    print("🏥" + "="*58 + "🏥")
    print("  ConnectIQ Neo4j MCP Server")
    print("  Standalone Mode - Connection Display")
    print("🏥" + "="*58 + "🏥")
    print()
    
    try:
        # Initialize connection with stats display
        await initialize_connection(show_stats=True)
        
        if is_connection_healthy():
            print("\n🚀 MCP Server is ready!")
            print("🔧 Available MCP Tools:")
            print("   - check_connection_health")
            print("   - execute_cypher")
            print("   - get_database_schema") 
            print("   - get_database_statistics")
            print("   - explore_healthcare_data")
            print("   - test_database_queries")
            print()
            print("📡 To use with FastAPI wrapper:")
            print("   python app.py")
            print()
            print("💡 This server is now ready to be imported by app.py")
            
            # Keep the server running for demonstration
            print("\n⏳ Press Ctrl+C to stop the server...")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n👋 Shutting down MCP server...")
                await neo4j_connection.close()
                print("✅ Server stopped")
        else:
            print("\n❌ Failed to establish database connection")
            print("🔧 Check your Neo4j server and connection settings")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Error starting MCP server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
