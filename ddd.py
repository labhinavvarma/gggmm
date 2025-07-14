#!/usr/bin/env python3
"""
Neo4j MCP Server using FastMCP
A Model Context Protocol server for interacting with Neo4j databases.
Can be run standalone or integrated with app.py
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime

from fastmcp import FastMCP
from neo4j import GraphDatabase, Driver
import neo4j.exceptions


@dataclass
class Neo4jContext:
    """Context for Neo4j database connection"""
    driver: Driver
    database: str


class Neo4jDatabase:
    """Neo4j database connection handler"""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None
        self.connection_status = "disconnected"
        self.last_health_check = None
        self.connection_error = None
    
    async def connect(self):
        """Connect to Neo4j database"""
        try:
            print(f"🔄 Attempting to connect to Neo4j at {self.uri}...")
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # Test connection
            await self.driver.verify_connectivity()
            self.connection_status = "connected"
            self.connection_error = None
            self.last_health_check = datetime.now()
            print(f"✅ Successfully connected to Neo4j at {self.uri}")
            print(f"📊 Database: {self.database}")
            return True
        except Exception as e:
            self.connection_status = "failed"
            self.connection_error = str(e)
            print(f"❌ Failed to connect to Neo4j: {e}")
            print(f"🔧 Check your connection settings:")
            print(f"   - URI: {self.uri}")
            print(f"   - Username: {self.username}")
            print(f"   - Database: {self.database}")
            raise
    
    async def disconnect(self):
        """Disconnect from Neo4j database"""
        if self.driver:
            await self.driver.close()
            self.connection_status = "disconnected"
            print("📴 Disconnected from Neo4j")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the Neo4j connection"""
        health_status = {
            "status": self.connection_status,
            "timestamp": datetime.now().isoformat(),
            "uri": self.uri,
            "database": self.database,
            "last_check": self.last_health_check.isoformat() if self.last_health_check else None
        }
        
        if not self.driver:
            health_status.update({
                "healthy": False,
                "error": "No driver instance available"
            })
            return health_status
        
        try:
            # Test basic connectivity
            await self.driver.verify_connectivity()
            
            # Test database access with a simple query
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                test_result = result.single()
                
            health_status.update({
                "healthy": True,
                "test_query_result": test_result["test"] if test_result else None,
                "connection_pool_info": {
                    "active_connections": "Available" if self.driver else "N/A"
                }
            })
            
            self.connection_status = "healthy"
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
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a Cypher query and return results"""
        if not self.driver:
            raise Exception("Not connected to Neo4j database")
        
        if self.connection_status not in ["connected", "healthy"]:
            raise Exception(f"Connection is not healthy. Status: {self.connection_status}")
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                records = []
                
                for record in result:
                    # Convert Neo4j record to dictionary
                    record_dict = {}
                    for key in record.keys():
                        value = record[key]
                        # Handle Neo4j types
                        if hasattr(value, '_properties'):  # Node or Relationship
                            record_dict[key] = dict(value._properties)
                            if hasattr(value, 'labels'):  # Node
                                record_dict[key]['_labels'] = list(value.labels)
                            if hasattr(value, 'type'):  # Relationship
                                record_dict[key]['_type'] = value.type
                        else:
                            record_dict[key] = value
                    records.append(record_dict)
                
                return {
                    "records": records,
                    "summary": {
                        "query": query,
                        "parameters": parameters,
                        "records_available": len(records)
                    }
                }
        except neo4j.exceptions.CypherSyntaxError as e:
            return {
                "error": f"Cypher Syntax Error: {str(e)}",
                "query": query,
                "parameters": parameters
            }
        except Exception as e:
            # Update connection status if there's a connection issue
            if "connection" in str(e).lower() or "network" in str(e).lower():
                self.connection_status = "unhealthy"
                self.connection_error = str(e)
            
            return {
                "error": f"Query execution error: {str(e)}",
                "query": query,
                "parameters": parameters
            }


# Global database instance - can be shared with app.py
neo4j_db = None

# Global MCP server instance for app.py integration
mcp = None


def init_neo4j_connection(uri: str = "bolt://localhost:7687", 
                         username: str = "neo4j", 
                         password: str = "password", 
                         database: str = "neo4j"):
    """Initialize Neo4j connection - can be called from app.py"""
    global neo4j_db
    neo4j_db = Neo4jDatabase(uri, username, password, database)
    return neo4j_db


def get_neo4j_connection():
    """Get the current Neo4j connection - can be used by app.py"""
    global neo4j_db
    return neo4j_db


def create_mcp_server(neo4j_instance: Optional[Neo4jDatabase] = None):
    """Create and configure MCP server - can be called from app.py"""
    global neo4j_db
    
    if neo4j_instance:
        neo4j_db = neo4j_instance
    elif neo4j_db is None:
        # Initialize with default configuration
        neo4j_db = init_neo4j_connection()
    
    # Initialize FastMCP server
    mcp = FastMCP("Neo4j-Server")
    
    @asynccontextmanager
    async def app_lifespan(server: FastMCP) -> AsyncIterator[Neo4jContext]:
        """Manage Neo4j connection lifecycle"""
        try:
            # Check if connection already exists (shared with app.py)
            if neo4j_db.connection_status == "connected":
                print("🔄 Using existing Neo4j connection from app.py")
                yield Neo4jContext(driver=neo4j_db.driver, database=neo4j_db.database)
            else:
                # Perform initial connection with retries
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        await neo4j_db.connect()
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"⏳ Retrying connection in 5 seconds... (Attempt {retry_count + 1}/{max_retries})")
                            await asyncio.sleep(5)
                        else:
                            print(f"💥 Failed to connect after {max_retries} attempts")
                            raise
                
                # Perform initial health check
                print("🔍 Performing initial health check...")
                health_status = await neo4j_db.health_check()
                if health_status["healthy"]:
                    print("✅ Initial health check passed")
                else:
                    print(f"⚠️  Initial health check failed: {health_status['error']}")
                
                yield Neo4jContext(driver=neo4j_db.driver, database=neo4j_db.database)
        finally:
            # Don't disconnect if connection is shared with app.py
            if not hasattr(neo4j_db, '_shared_connection'):
                await neo4j_db.disconnect()
    
    # Set up lifespan management
    mcp.lifespan = app_lifespan
    
    # Register all MCP tools
    register_mcp_tools(mcp)
    
    return mcp


def register_mcp_tools(mcp: FastMCP):
    """Register all MCP tools with the server"""
    
    @mcp.tool()
    def check_connection_health() -> str:
        """
        Check the health of the Neo4j database connection.
        
        Returns:
            JSON string containing connection health status and diagnostics
        """
        try:
            # Run health check synchronously by creating a new event loop if needed
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                # If we're in an async context, we need to handle this differently
                health_status = {
                    "status": neo4j_db.connection_status,
                    "timestamp": datetime.now().isoformat(),
                    "uri": neo4j_db.uri,
                    "database": neo4j_db.database,
                    "last_check": neo4j_db.last_health_check.isoformat() if neo4j_db.last_health_check else None,
                    "healthy": neo4j_db.connection_status in ["connected", "healthy"],
                    "error": neo4j_db.connection_error
                }
            else:
                health_status = loop.run_until_complete(neo4j_db.health_check())
            
            return json.dumps(health_status, indent=2)
        except Exception as e:
            return json.dumps({
                "healthy": False,
                "error": f"Health check failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }, indent=2)

    @mcp.tool()
    def get_connection_info() -> str:
        """
        Get detailed information about the current Neo4j connection.
        
        Returns:
            JSON string containing connection details and status
        """
        connection_info = {
            "uri": neo4j_db.uri,
            "username": neo4j_db.username,
            "database": neo4j_db.database,
            "status": neo4j_db.connection_status,
            "last_health_check": neo4j_db.last_health_check.isoformat() if neo4j_db.last_health_check else None,
            "connection_error": neo4j_db.connection_error,
            "driver_available": neo4j_db.driver is not None
        }
        
        return json.dumps(connection_info, indent=2)

    @mcp.tool()
    def execute_cypher(query: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute a Cypher query against the Neo4j database.
        
        Args:
            query: The Cypher query to execute
            parameters: Optional dictionary of parameters for the query
        
        Returns:
            JSON string containing query results or error information
        """
        try:
            result = neo4j_db.execute_query(query, parameters)
            return json.dumps(result, indent=2, default=str)
        except Exception as e:
            return json.dumps({
                "error": f"Failed to execute query: {str(e)}",
                "query": query,
                "parameters": parameters,
                "connection_status": neo4j_db.connection_status
            }, indent=2)

    @mcp.tool()
    def get_database_schema() -> str:
        """
        Get the database schema including node labels, relationship types, and property keys.
        
        Returns:
            JSON string containing the database schema information
        """
        # Check connection health first
        if neo4j_db.connection_status not in ["connected", "healthy"]:
            return json.dumps({
                "error": f"Database connection is not healthy. Status: {neo4j_db.connection_status}",
                "connection_error": neo4j_db.connection_error
            }, indent=2)
        
        schema_queries = {
            "node_labels": "CALL db.labels() YIELD label RETURN collect(label) AS labels",
            "relationship_types": "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS types",
            "property_keys": "CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) AS keys"
        }
        
        schema_info = {}
        
        for schema_type, query in schema_queries.items():
            try:
                result = neo4j_db.execute_query(query)
                if result.get("records"):
                    schema_info[schema_type] = result["records"][0]
                else:
                    schema_info[schema_type] = []
            except Exception as e:
                schema_info[schema_type] = f"Error: {str(e)}"
        
        # Get sample of database structure
        try:
            sample_query = """
            MATCH (n)
            WITH labels(n) AS nodeLabels, keys(n) AS nodeKeys
            RETURN nodeLabels, collect(DISTINCT nodeKeys) AS properties
            LIMIT 10
            """
            sample_result = neo4j_db.execute_query(sample_query)
            schema_info["sample_structure"] = sample_result.get("records", [])
        except Exception as e:
            schema_info["sample_structure"] = f"Error getting sample: {str(e)}"
        
        return json.dumps(schema_info, indent=2)

    @mcp.tool()
    def get_database_info() -> str:
        """
        Get general information about the Neo4j database.
        
        Returns:
            JSON string containing database information
        """
        # Check connection health first
        if neo4j_db.connection_status not in ["connected", "healthy"]:
            return json.dumps({
                "error": f"Database connection is not healthy. Status: {neo4j_db.connection_status}",
                "connection_error": neo4j_db.connection_error
            }, indent=2)
        
        info_queries = {
            "node_count": "MATCH (n) RETURN count(n) AS nodeCount",
            "relationship_count": "MATCH ()-[r]->() RETURN count(r) AS relationshipCount",
            "database_info": "CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition"
        }
        
        db_info = {}
        
        for info_type, query in info_queries.items():
            try:
                result = neo4j_db.execute_query(query)
                if result.get("records"):
                    db_info[info_type] = result["records"]
                else:
                    db_info[info_type] = "No data"
            except Exception as e:
                db_info[info_type] = f"Error: {str(e)}"
        
        return json.dumps(db_info, indent=2)

    @mcp.tool()
    def create_node(labels: List[str], properties: Dict[str, Any]) -> str:
        """
        Create a new node in the Neo4j database.
        
        Args:
            labels: List of labels for the node
            properties: Dictionary of properties for the node
        
        Returns:
            JSON string containing the result of the node creation
        """
        if not labels:
            return json.dumps({"error": "At least one label is required"})
        
        # Check connection health first
        if neo4j_db.connection_status not in ["connected", "healthy"]:
            return json.dumps({
                "error": f"Database connection is not healthy. Status: {neo4j_db.connection_status}",
                "connection_error": neo4j_db.connection_error
            }, indent=2)
        
        # Build the Cypher query
        labels_str = ":".join(labels)
        query = f"CREATE (n:{labels_str} $properties) RETURN n"
        
        try:
            result = neo4j_db.execute_query(query, {"properties": properties})
            return json.dumps({
                "success": True,
                "message": f"Created node with labels {labels}",
                "result": result
            }, indent=2, default=str)
        except Exception as e:
            return json.dumps({
                "error": f"Failed to create node: {str(e)}",
                "labels": labels,
                "properties": properties
            }, indent=2)

    @mcp.tool()
    def find_nodes(label: str, properties: Optional[Dict[str, Any]] = None, limit: int = 10) -> str:
        """
        Find nodes in the Neo4j database by label and optional properties.
        
        Args:
            label: The node label to search for
            properties: Optional dictionary of properties to filter by
            limit: Maximum number of nodes to return (default: 10)
        
        Returns:
            JSON string containing the found nodes
        """
        # Check connection health first
        if neo4j_db.connection_status not in ["connected", "healthy"]:
            return json.dumps({
                "error": f"Database connection is not healthy. Status: {neo4j_db.connection_status}",
                "connection_error": neo4j_db.connection_error
            }, indent=2)
        
        # Build the Cypher query
        query = f"MATCH (n:{label}"
        
        if properties:
            # Add property filters
            prop_conditions = []
            for key, value in properties.items():
                if isinstance(value, str):
                    prop_conditions.append(f"n.{key} = '{value}'")
                else:
                    prop_conditions.append(f"n.{key} = {value}")
            
            if prop_conditions:
                query += " WHERE " + " AND ".join(prop_conditions)
        
        query += f") RETURN n LIMIT {limit}"
        
        try:
            result = neo4j_db.execute_query(query)
            return json.dumps(result, indent=2, default=str)
        except Exception as e:
            return json.dumps({
                "error": f"Failed to find nodes: {str(e)}",
                "label": label,
                "properties": properties
            }, indent=2)

    @mcp.resource("neo4j://schema")
    def get_schema_resource() -> str:
        """Provide the database schema as a resource"""
        return get_database_schema()

    @mcp.resource("neo4j://info")
    def get_info_resource() -> str:
        """Provide database information as a resource"""
        return get_database_info()

    @mcp.resource("neo4j://health")
    def get_health_resource() -> str:
        """Provide connection health status as a resource"""
        return check_connection_health()

    @mcp.prompt()
    def cypher_query_prompt(description: str) -> str:
        """
        Generate a Cypher query based on a natural language description.
        
        Args:
            description: Natural language description of what you want to query
        """
        schema = get_database_schema()
        
        return f"""You are a Neo4j Cypher query expert. Based on the following database schema and user request, generate an appropriate Cypher query.

Database Schema:
{schema}

User Request: {description}

Please provide:
1. A Cypher query that addresses the user's request
2. A brief explanation of what the query does
3. Any assumptions made about the data structure

Cypher Query:"""


# Initialize default MCP server instance for app.py integration
# This will be imported by app.py as: from mcpserver import mcp
print("🔧 Initializing Neo4j MCP Server for app.py integration...")

# Initialize with default/hardcoded configuration
init_neo4j_connection(
    uri="bolt://localhost:7687",           # Your Neo4j URI
    username="neo4j",                      # Your Neo4j username  
    password="password",                   # Your Neo4j password - CHANGE THIS!
    database="neo4j"                       # Your Neo4j database name
)

# Create the default MCP server instance that app.py will import
mcp = create_mcp_server()

print("✅ Neo4j MCP Server initialized and ready for app.py")
print("=" * 50)
print(f"📍 Neo4j URI: {neo4j_db.uri}")
print(f"👤 Username: {neo4j_db.username}")
print(f"🗄️  Database: {neo4j_db.database}")
print("=" * 50)
print("⚠️  Remember to update the hardcoded credentials!")
print()


def run_standalone():
    """Run the MCP server as a standalone application"""
    # Print startup information
    print("🚀 Starting Neo4j FastMCP Server (Standalone Mode)")
    print("🔄 Using existing MCP server instance...")
    print()
    
    # Run the server
    mcp.run()


# For app.py integration, also provide these helper functions
def get_mcp_server():
    """Get the MCP server instance for app.py"""
    return mcp

def get_neo4j_db():
    """Get the Neo4j database instance for app.py"""
    return neo4j_db

def update_neo4j_config(uri: str = None, username: str = None, password: str = None, database: str = None):
    """Update Neo4j configuration after import - useful for app.py"""
    global neo4j_db
    if uri:
        neo4j_db.uri = uri
    if username:
        neo4j_db.username = username
    if password:
        neo4j_db.password = password
    if database:
        neo4j_db.database = database
    print(f"✅ Updated Neo4j config: {neo4j_db.uri} | {neo4j_db.username} | {neo4j_db.database}")


# Initialize default MCP server instance for app.py integration
# This will be imported by app.py as: from mcpserver import mcp
print("🔧 Initializing Neo4j MCP Server for app.py integration...")

# Initialize with default/hardcoded configuration
init_neo4j_connection(
    uri="bolt://localhost:7687",           # Your Neo4j URI
    username="neo4j",                      # Your Neo4j username  
    password="password",                   # Your Neo4j password - CHANGE THIS!
    database="neo4j"                       # Your Neo4j database name
)

# Create the default MCP server instance that app.py will import
mcp = create_mcp_server()

print("✅ Neo4j MCP Server initialized and ready for app.py")
print("=" * 50)
print(f"📍 Neo4j URI: {neo4j_db.uri}")
print(f"👤 Username: {neo4j_db.username}")
print(f"🗄️  Database: {neo4j_db.database}")
print("=" * 50)
print("⚠️  Remember to update the hardcoded credentials!")
print()


def run_standalone():
    """Run the MCP server as a standalone application"""
    # Print startup information
    print("🚀 Starting Neo4j FastMCP Server (Standalone Mode)")
    print("🔄 Using existing MCP server instance...")
    print()
    
    # Run the server
    mcp.run()


# For app.py integration, also provide these helper functions
def get_mcp_server():
    """Get the MCP server instance for app.py"""
    return mcp

def get_neo4j_db():
    """Get the Neo4j database instance for app.py"""
    return neo4j_db

def update_neo4j_config(uri: str = None, username: str = None, password: str = None, database: str = None):
    """Update Neo4j configuration after import - useful for app.py"""
    global neo4j_db
    if uri:
        neo4j_db.uri = uri
    if username:
        neo4j_db.username = username
    if password:
        neo4j_db.password = password
    if database:
        neo4j_db.database = database
    print(f"✅ Updated Neo4j config: {neo4j_db.uri} | {neo4j_db.username} | {neo4j_db.database}")


if __name__ == "__main__":
    run_standalone()
