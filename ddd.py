#!/usr/bin/env python3
"""
MCP Server Module for Neo4j

This module creates a FastMCP server instance with Neo4j tools that can be imported
and used with FastAPI SSE transport.
"""

import json
import logging
import os
import re
import time
from typing import Any, Optional

import mcp.types as types
from mcp.server.fastmcp import FastMCP
from neo4j import (
    AsyncDriver,
    AsyncGraphDatabase,
    AsyncResult,
    AsyncTransaction,
    GraphDatabase,
)
from neo4j.exceptions import DatabaseError
from pydantic import Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_neo4j_cypher")


def get_config() -> tuple[str, str, str, str]:
    """
    Get hard-coded Neo4j configuration.
    
    Returns:
        Tuple of (db_url, username, password, database)
    """
    # For local Neo4j:
    db_url = "bolt://localhost:7687"
    username = "neo4j"
    password = "password"
    database = "neo4j"
    
    # For Neo4j Aura (uncomment and update with your credentials):
    # db_url = "neo4j+s://your-instance.databases.neo4j.io"
    # username = "neo4j"
    # password = "your-aura-password"
    # database = "neo4j"
    
    return db_url, username, password, database


def healthcheck(db_url: str, username: str, password: str, database: str) -> bool:
    """
    Confirm that Neo4j is running.
    Returns True if successful, False if failed.
    """
    try:
        sync_driver = GraphDatabase.driver(db_url, auth=(username, password))
        with sync_driver.session(database=database) as session:
            session.run("RETURN 1")
        sync_driver.close()
        logger.info("✓ Neo4j connection verified")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Neo4j connection failed: {e}")
        return False


async def _read(tx: AsyncTransaction, query: str, params: dict[str, Any]) -> str:
    """Execute a read transaction and return JSON results."""
    raw_results = await tx.run(query, params)
    eager_results = await raw_results.to_eager_result()
    return json.dumps([r.data() for r in eager_results.records], default=str, indent=2)


async def _write(tx: AsyncTransaction, query: str, params: dict[str, Any]) -> AsyncResult:
    """Execute a write transaction."""
    return await tx.run(query, params)


def _is_write_query(query: str) -> bool:
    """Check if the query is a write query by looking for write operations."""
    write_keywords = r"\b(MERGE|CREATE|SET|DELETE|REMOVE|ADD|DROP|DETACH)\b"
    return re.search(write_keywords, query, re.IGNORECASE) is not None


def _sanitize_query(query: str) -> str:
    """Basic query sanitization - remove comments and extra whitespace."""
    # Remove single-line comments
    query = re.sub(r'//.*$', '', query, flags=re.MULTILINE)
    # Remove multi-line comments
    query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
    # Normalize whitespace
    query = ' '.join(query.split())
    return query.strip()


# Get configuration
db_url, username, password, database = get_config()

# Create async driver
neo4j_driver = AsyncGraphDatabase.driver(db_url, auth=(username, password))

# Perform health check (non-blocking)
healthcheck(db_url, username, password, database)

# Create FastMCP instance
mcp: FastMCP = FastMCP(
    "neo4j-mcp-server",
    dependencies=["neo4j", "pydantic"]
)


@mcp.prompt()
async def neo4j_tool_usage_guide() -> str:
    """
    Neo4j Tool Usage Guide and Decision Framework
    
    This prompt provides comprehensive guidance on when and how to use each Neo4j tool
    to help AI assistants make optimal decisions about tool selection.
    """
    return """
# Neo4j Tool Usage Guide

## Available Tools Overview

### 1. get_neo4j_schema()
**Purpose**: Retrieve the complete database schema
**When to use**:
- First interaction with an unknown Neo4j database
- User asks about database structure, node types, or relationships
- Before writing complex queries to understand available properties
- When user asks "what data do you have?" or "show me the schema"
- To validate if certain node labels or relationships exist

**Examples of when to use**:
- "What kind of data is in this database?"
- "Show me all the node types"
- "What properties does the User node have?"
- "What relationships exist in this database?"

### 2. read_neo4j_cypher(query, params)
**Purpose**: Execute read-only Cypher queries
**When to use**:
- Retrieving data without modifying the database
- Searching for specific nodes or relationships
- Aggregating data (COUNT, SUM, AVG, etc.)
- Complex filtering and pattern matching
- Reporting and analytics queries

**Query types allowed**:
- MATCH, RETURN, WITH, UNWIND
- CALL (for read-only procedures)
- WHERE, ORDER BY, LIMIT, SKIP
- Aggregation functions
- OPTIONAL MATCH

**Examples of when to use**:
- "Find all users with age > 25"
- "Show me the most connected person"
- "Count how many products were sold last month"
- "Find shortest path between two nodes"
- "Get all friends of a specific user"

### 3. write_neo4j_cypher(query, params)
**Purpose**: Execute write operations that modify the database
**When to use**:
- Creating new nodes or relationships
- Updating existing properties
- Deleting nodes or relationships
- Merging data (MERGE operations)
- Setting or removing labels
- Creating constraints or indexes

**Query types allowed**:
- CREATE, MERGE, SET, DELETE, REMOVE
- ADD, DROP (for labels)
- DETACH DELETE
- Creating constraints and indexes

**Examples of when to use**:
- "Create a new user named John"
- "Add a friendship relationship between Alice and Bob"
- "Update the price of product ID 123"
- "Delete all nodes with label 'Temporary'"
- "Create an index on User.email"

## Decision Framework

### Step 1: Understand the Intent
1. **Exploration/Discovery** → Use `get_neo4j_schema()`
2. **Data Retrieval/Query** → Use `read_neo4j_cypher()`
3. **Data Modification** → Use `write_neo4j_cypher()`

### Step 2: Schema First Approach
If you don't know the database structure:
1. Start with `get_neo4j_schema()` to understand available nodes and relationships
2. Then proceed with appropriate read or write operations

### Step 3: Safety Considerations
- Always prefer `read_neo4j_cypher()` when possible
- Use `write_neo4j_cypher()` only when explicitly asked to modify data
- Confirm write operations with user before executing
- Use MERGE instead of CREATE to avoid duplicates

### Step 4: Query Validation
Before executing:
- Ensure query syntax is correct
- Use parameterized queries to prevent injection
- Validate that referenced labels and properties exist in schema

## Best Practices

### For Schema Exploration:
1. Always call `get_neo4j_schema()` first for unknown databases
2. Use schema information to construct better queries
3. Validate node labels and properties against schema

### For Read Operations:
1. Use LIMIT to prevent large result sets
2. Use parameters for dynamic values
3. Consider performance with proper WHERE clauses
4. Use EXPLAIN or PROFILE for optimization

### For Write Operations:
1. Use MERGE when you want to avoid duplicates
2. Always specify sufficient properties to identify nodes
3. Use transactions for multi-step operations
4. Validate data before writing
5. Provide meaningful feedback about what was changed

## Common Patterns

### Initial Database Exploration:
```
1. get_neo4j_schema() - Understand structure
2. read_neo4j_cypher("MATCH (n) RETURN labels(n), count(*) GROUP BY labels(n)") - Count nodes by type
3. read_neo4j_cypher("MATCH ()-[r]->() RETURN type(r), count(*) GROUP BY type(r)") - Count relationships by type
```

### Creating Related Data:
```
1. Use MERGE to find or create nodes
2. Use CREATE for relationships if you're sure they don't exist
3. Use SET to update properties
```

### Data Analysis Workflow:
```
1. get_neo4j_schema() - Understand structure
2. read_neo4j_cypher() - Explore data patterns
3. read_neo4j_cypher() - Execute analysis queries
4. write_neo4j_cypher() - Store results if needed
```

## Error Handling
- Check for ProcedureNotFound errors (APOC plugin required for schema)
- Validate query syntax before execution
- Handle connection timeouts gracefully
- Provide meaningful error messages to users

Remember: When in doubt, start with schema exploration, then read, and only write when explicitly requested and confirmed.
"""


@mcp.tool()
async def get_neo4j_schema() -> list[types.TextContent]:
    """
    Get the complete schema of the Neo4j database including nodes, relationships, and properties.
    This provides a comprehensive view of the database structure.
    
    Returns:
        JSON representation of the database schema with node labels, properties, and relationships.
    """
    get_schema_query = """
    CALL apoc.meta.data() 
    YIELD label, property, type, other, unique, index, elementType
    WHERE elementType = 'node' AND NOT label STARTS WITH '_'
    WITH label, 
        COLLECT(CASE WHEN type <> 'RELATIONSHIP' 
            THEN [property, type + CASE WHEN unique THEN " unique" ELSE "" END + 
                  CASE WHEN index THEN " indexed" ELSE "" END] END) AS attributes,
        COLLECT(CASE WHEN type = 'RELATIONSHIP' 
            THEN [property, HEAD(other)] END) AS relationships
    RETURN label, 
           apoc.map.fromPairs(attributes) AS attributes, 
           apoc.map.fromPairs(relationships) AS relationships
    ORDER BY label
    """

    try:
        async with neo4j_driver.session(database=database) as session:
            results_json_str = await session.execute_read(_read, get_schema_query, {})
            logger.info("Successfully retrieved Neo4j schema")
            return [types.TextContent(type="text", text=results_json_str)]

    except Exception as e:
        error_msg = f"Error retrieving schema: {e}"
        if "ProcedureNotFound" in str(e):
            error_msg += "\n\nSuggestion: Install and enable the APOC plugin in your Neo4j database."
        logger.error(error_msg)
        return [types.TextContent(type="text", text=error_msg)]


@mcp.tool()
async def read_neo4j_cypher(
    query: str = Field(..., description="The Cypher query to execute (read-only operations like MATCH, RETURN, etc.)"),
    params: Optional[dict[str, Any]] = Field(
        default=None, 
        description="Optional parameters to pass to the Cypher query as a dictionary"
    ),
) -> list[types.TextContent]:
    """
    Execute a read-only Cypher query on the Neo4j database.
    
    This tool only allows MATCH, RETURN, WITH, UNWIND, CALL (for read procedures), 
    and other read-only operations. Write operations are blocked for safety.
    
    Args:
        query: The Cypher query string to execute
        params: Optional dictionary of parameters to substitute in the query
        
    Returns:
        JSON representation of the query results
    """
    if not query:
        return [types.TextContent(type="text", text="Error: Query cannot be empty")]
    
    # Sanitize and validate query
    clean_query = _sanitize_query(query)
    
    if _is_write_query(clean_query):
        error_msg = "Error: Only read operations (MATCH, RETURN, etc.) are allowed. Use write_neo4j_cypher for write operations."
        return [types.TextContent(type="text", text=error_msg)]

    try:
        async with neo4j_driver.session(database=database) as session:
            query_params = params or {}
            results_json_str = await session.execute_read(_read, clean_query, query_params)
            
            logger.info(f"Read query executed successfully. Result length: {len(results_json_str)} characters")
            return [types.TextContent(type="text", text=results_json_str)]

    except Exception as e:
        error_msg = f"Database error executing read query: {e}\n\nQuery: {clean_query}\nParams: {params}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=error_msg)]


@mcp.tool()
async def write_neo4j_cypher(
    query: str = Field(..., description="The Cypher query to execute (write operations like CREATE, MERGE, SET, DELETE, etc.)"),
    params: Optional[dict[str, Any]] = Field(
        default=None, 
        description="Optional parameters to pass to the Cypher query as a dictionary"
    ),
) -> list[types.TextContent]:
    """
    Execute a write Cypher query on the Neo4j database.
    
    This tool allows CREATE, MERGE, SET, DELETE, REMOVE, and other write operations.
    Use with caution as this can modify your database.
    
    Args:
        query: The Cypher query string to execute
        params: Optional dictionary of parameters to substitute in the query
        
    Returns:
        Summary of the write operation including counts of nodes/relationships created/updated/deleted
    """
    if not query:
        return [types.TextContent(type="text", text="Error: Query cannot be empty")]
    
    # Sanitize and validate query
    clean_query = _sanitize_query(query)
    
    if not _is_write_query(clean_query):
        error_msg = "Error: Only write operations (CREATE, MERGE, SET, DELETE, etc.) are allowed. Use read_neo4j_cypher for read operations."
        return [types.TextContent(type="text", text=error_msg)]

    try:
        async with neo4j_driver.session(database=database) as session:
            query_params = params or {}
            raw_results = await session.execute_write(_write, clean_query, query_params)
            
            # Extract summary counters
            counters = raw_results._summary.counters
            counters_dict = {
                "nodes_created": counters.nodes_created,
                "nodes_deleted": counters.nodes_deleted,
                "relationships_created": counters.relationships_created,
                "relationships_deleted": counters.relationships_deleted,
                "properties_set": counters.properties_set,
                "labels_added": counters.labels_added,
                "labels_removed": counters.labels_removed,
                "indexes_added": counters.indexes_added,
                "indexes_removed": counters.indexes_removed,
                "constraints_added": counters.constraints_added,
                "constraints_removed": counters.constraints_removed,
            }
            
            counters_json_str = json.dumps(counters_dict, indent=2)
            logger.info(f"Write query executed successfully: {counters_json_str}")
            
            return [types.TextContent(type="text", text=counters_json_str)]

    except Exception as e:
        error_msg = f"Database error executing write query: {e}\n\nQuery: {clean_query}\nParams: {params}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=error_msg)]


# Log successful initialization
logger.info("🚀 Neo4j MCP Server module loaded successfully")

# Make mcp available for import
__all__ = ['mcp']
