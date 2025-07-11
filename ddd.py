#!/usr/bin/env python3
"""
Neo4j MCP Server using FastMCP
A Model Context Protocol server for interacting with Neo4j databases.
"""

import os
import json
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass

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
    
    async def connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # Test connection
            await self.driver.verify_connectivity()
            print(f"✅ Connected to Neo4j at {self.uri}")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Neo4j database"""
        if self.driver:
            await self.driver.close()
            print("📴 Disconnected from Neo4j")
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a Cypher query and return results"""
        if not self.driver:
            raise Exception("Not connected to Neo4j database")
        
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
            return {
                "error": f"Query execution error: {str(e)}",
                "query": query,
                "parameters": parameters
            }


# Initialize FastMCP server
mcp = FastMCP("Neo4j-Server")

# Database instance
neo4j_db = Neo4jDatabase(
    uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    username=os.getenv("NEO4J_USERNAME", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", "password"),
    database=os.getenv("NEO4J_DATABASE", "neo4j")
)


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[Neo4jContext]:
    """Manage Neo4j connection lifecycle"""
    try:
        await neo4j_db.connect()
        yield Neo4jContext(driver=neo4j_db.driver, database=neo4j_db.database)
    finally:
        await neo4j_db.disconnect()


# Set up lifespan management
mcp = FastMCP("Neo4j-Server", lifespan=app_lifespan)


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
            "parameters": parameters
        }, indent=2)


@mcp.tool()
def get_database_schema() -> str:
    """
    Get the database schema including node labels, relationship types, and property keys.
    
    Returns:
        JSON string containing the database schema information
    """
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


if __name__ == "__main__":
    # Print startup information
    print("🚀 Starting Neo4j FastMCP Server")
    print(f"📍 Neo4j URI: {neo4j_db.uri}")
    print(f"👤 Username: {neo4j_db.username}")
    print(f"🗄️  Database: {neo4j_db.database}")
    print()
    
    # Run the server
    mcp.run()
