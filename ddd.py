import json
import logging
import re
from typing import Any, Optional, List
import asyncio

from fastmcp.server import FastMCP
from fastmcp.tools.tool import ToolResult, TextContent
from fastmcp.prompts.prompt import PromptResult, PromptMessage
from neo4j import (
    AsyncDriver,
    AsyncGraphDatabase,
    AsyncResult,
    AsyncTransaction,
)
from pydantic import Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neo4j_mcp_server")

# Database configuration
DB_CONFIG = {
    "uri": "neo4j://10.189.116.237:7687",
    "database": "connectiq",
    "user": "neo4j",
    "password": "Vkg5d$F!pLq2@9vRwE="
}

# Global driver instance
neo4j_driver: Optional[AsyncDriver] = None

async def get_driver() -> AsyncDriver:
    """Get or create the Neo4j driver"""
    global neo4j_driver
    if neo4j_driver is None:
        neo4j_driver = AsyncGraphDatabase.driver(
            DB_CONFIG["uri"],
            auth=(DB_CONFIG["user"], DB_CONFIG["password"])
        )
    return neo4j_driver

async def close_driver():
    """Close the Neo4j driver"""
    global neo4j_driver
    if neo4j_driver:
        await neo4j_driver.close()
        neo4j_driver = None

async def _read(tx: AsyncTransaction, query: str, params: dict[str, Any]) -> str:
    """Execute a read transaction and return JSON results"""
    raw_results = await tx.run(query, params)
    eager_results = await raw_results.to_eager_result()
    return json.dumps([r.data() for r in eager_results.records], default=str)

async def _write(tx: AsyncTransaction, query: str, params: dict[str, Any]) -> AsyncResult:
    """Execute a write transaction"""
    return await tx.run(query, params)

def _is_write_query(query: str) -> bool:
    """Check if the query is a write query"""
    return re.search(r"\b(MERGE|CREATE|SET|DELETE|REMOVE|ADD)\b", query, re.IGNORECASE) is not None

def _clean_schema(schema: dict) -> dict:
    """Clean and format the schema for better readability"""
    cleaned = {}
    
    for key, entry in schema.items():
        new_entry = {"type": entry["type"]}
        
        if "count" in entry:
            new_entry["count"] = entry["count"]
        
        labels = entry.get("labels", [])
        if labels:
            new_entry["labels"] = labels
        
        # Clean properties
        props = entry.get("properties", {})
        clean_props = {}
        for pname, pinfo in props.items():
            cp = {}
            if "indexed" in pinfo:
                cp["indexed"] = pinfo["indexed"]
            if "type" in pinfo:
                cp["type"] = pinfo["type"]
            if cp:
                clean_props[pname] = cp
        if clean_props:
            new_entry["properties"] = clean_props
        
        # Clean relationships
        if entry.get("relationships"):
            rels_out = {}
            for rel_name, rel in entry["relationships"].items():
                cr = {}
                if "direction" in rel:
                    cr["direction"] = rel["direction"]
                
                rlabels = rel.get("labels", [])
                if rlabels:
                    cr["labels"] = rlabels
                
                rprops = rel.get("properties", {})
                clean_rprops = {}
                for rpname, rpinfo in rprops.items():
                    crp = {}
                    if "indexed" in rpinfo:
                        crp["indexed"] = rpinfo["indexed"]
                    if "type" in rpinfo:
                        crp["type"] = rpinfo["type"]
                    if crp:
                        clean_rprops[rpname] = crp
                if clean_rprops:
                    cr["properties"] = clean_rprops
                
                if cr:
                    rels_out[rel_name] = cr
            
            if rels_out:
                new_entry["relationships"] = rels_out
        
        cleaned[key] = new_entry
    
    return cleaned

# Create the FastMCP server
mcp = FastMCP("neo4j-connectiq-server")

@mcp.tool()
async def get_neo4j_schema() -> ToolResult:
    """
    Get the complete schema of the Neo4j ConnectIQ database.
    This includes all node types, their properties, and relationships.
    Requires APOC plugin to be installed.
    """
    driver = await get_driver()
    
    get_schema_query = "CALL apoc.meta.schema();"
    
    try:
        async with driver.session(database=DB_CONFIG["database"]) as session:
            results_json_str = await session.execute_read(_read, get_schema_query, {})
            
            logger.info("Successfully retrieved database schema")
            
            schema_data = json.loads(results_json_str)
            if schema_data and len(schema_data) > 0:
                schema = schema_data[0].get('value', {})
                cleaned_schema = _clean_schema(schema)
                return ToolResult(
                    content=[TextContent(
                        type="text", 
                        text=json.dumps(cleaned_schema, indent=2)
                    )]
                )
            else:
                return ToolResult(
                    content=[TextContent(
                        type="text", 
                        text=json.dumps({"message": "No schema data found"})
                    )]
                )
                
    except Exception as e:
        logger.error(f"Error retrieving schema: {e}")
        return ToolResult(
            content=[TextContent(
                type="text", 
                text=json.dumps({"error": str(e)})
            )]
        )

@mcp.tool()
async def execute_cypher_read(
    query: str = Field(..., description="The Cypher query to execute (read-only)"),
    params: Optional[dict[str, Any]] = Field(None, description="Parameters for the Cypher query")
) -> ToolResult:
    """
    Execute a read-only Cypher query on the ConnectIQ Neo4j database.
    Only MATCH, RETURN, WITH, UNWIND, and other read operations are allowed.
    """
    if params is None:
        params = {}
    
    if _is_write_query(query):
        return ToolResult(
            content=[TextContent(
                type="text", 
                text=json.dumps({"error": "Write operations not allowed in read query"})
            )]
        )
    
    driver = await get_driver()
    
    try:
        async with driver.session(database=DB_CONFIG["database"]) as session:
            results_json_str = await session.execute_read(_read, query, params)
            
            logger.info(f"Successfully executed read query: {query[:100]}...")
            
            return ToolResult(
                content=[TextContent(type="text", text=results_json_str)]
            )
            
    except Exception as e:
        logger.error(f"Error executing read query: {e}")
        return ToolResult(
            content=[TextContent(
                type="text", 
                text=json.dumps({"error": str(e), "query": query, "params": params})
            )]
        )

@mcp.tool()
async def execute_cypher_write(
    query: str = Field(..., description="The Cypher query to execute (write operations)"),
    params: Optional[dict[str, Any]] = Field(None, description="Parameters for the Cypher query")
) -> ToolResult:
    """
    Execute a write Cypher query on the ConnectIQ Neo4j database.
    Includes CREATE, MERGE, SET, DELETE, REMOVE operations.
    Use with caution as this modifies the database.
    """
    if params is None:
        params = {}
    
    if not _is_write_query(query):
        return ToolResult(
            content=[TextContent(
                type="text", 
                text=json.dumps({"error": "Only write operations allowed in write query"})
            )]
        )
    
    driver = await get_driver()
    
    try:
        async with driver.session(database=DB_CONFIG["database"]) as session:
            raw_results = await session.execute_write(_write, query, params)
            counters = raw_results._summary.counters.__dict__
            
            logger.info(f"Successfully executed write query: {query[:100]}...")
            logger.info(f"Write operation results: {counters}")
            
            return ToolResult(
                content=[TextContent(
                    type="text", 
                    text=json.dumps(counters, default=str)
                )]
            )
            
    except Exception as e:
        logger.error(f"Error executing write query: {e}")
        return ToolResult(
            content=[TextContent(
                type="text", 
                text=json.dumps({"error": str(e), "query": query, "params": params})
            )]
        )

@mcp.tool()
async def get_database_info() -> ToolResult:
    """
    Get basic information about the ConnectIQ Neo4j database including
    node counts, relationship counts, and database statistics.
    """
    driver = await get_driver()
    
    info_queries = [
        ("total_nodes", "MATCH (n) RETURN count(n) as count"),
        ("total_relationships", "MATCH ()-[r]->() RETURN count(r) as count"),
        ("node_labels", "CALL db.labels() YIELD label RETURN collect(label) as labels"),
        ("relationship_types", "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"),
        ("database_info", "CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition")
    ]
    
    results = {}
    
    try:
        async with driver.session(database=DB_CONFIG["database"]) as session:
            for info_type, query in info_queries:
                try:
                    result_json = await session.execute_read(_read, query, {})
                    result_data = json.loads(result_json)
                    results[info_type] = result_data
                except Exception as e:
                    results[info_type] = {"error": str(e)}
        
        return ToolResult(
            content=[TextContent(
                type="text", 
                text=json.dumps(results, indent=2)
            )]
        )
        
    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        return ToolResult(
            content=[TextContent(
                type="text", 
                text=json.dumps({"error": str(e)})
            )]
        )

# Prompts for common Neo4j operations
@mcp.prompt()
async def cypher_query_helper(
    task: str = Field(..., description="What you want to do with the database"),
    context: Optional[str] = Field(None, description="Additional context about the data you're working with")
) -> PromptResult:
    """
    Generate Cypher query suggestions based on your task description.
    Helps create queries for the ConnectIQ Neo4j database.
    """
    
    # Get schema to provide context
    schema_result = await get_neo4j_schema()
    schema_text = schema_result.content[0].text if schema_result.content else "{}"
    
    prompt_text = f"""You are a Neo4j Cypher expert working with the ConnectIQ database.

Database Schema:
{schema_text}

User Task: {task}
{f"Additional Context: {context}" if context else ""}

Please provide:
1. A Cypher query to accomplish this task
2. Explanation of what the query does
3. Any important considerations or alternatives
4. Expected output format

Guidelines:
- Use exact node labels and property names from the schema
- Include LIMIT clauses for large result sets
- Prefer MATCH over CREATE when exploring data
- Use parameterized queries when appropriate
- Consider performance implications

Example format:
```cypher
MATCH (n:NodeType)
WHERE n.property = 'value'
RETURN n.property
LIMIT 100
```

Explanation: This query finds nodes of type 'NodeType' where property equals 'value' and returns the property, limited to 100 results.
"""
    
    return PromptResult(
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=prompt_text)
            )
        ]
    )

@mcp.prompt()
async def data_exploration_guide() -> PromptResult:
    """
    Provides a guide for exploring the ConnectIQ Neo4j database structure and data.
    """
    
    prompt_text = """You are exploring the ConnectIQ Neo4j database. Here's a systematic approach:

## Step 1: Understand the Schema
```cypher
CALL apoc.meta.schema()
```

## Step 2: Check Node Counts
```cypher
MATCH (n) 
RETURN labels(n) as NodeType, count(n) as Count 
ORDER BY Count DESC
```

## Step 3: Explore Relationships
```cypher
MATCH ()-[r]->() 
RETURN type(r) as RelationshipType, count(r) as Count 
ORDER BY Count DESC
```

## Step 4: Sample Data Inspection
```cypher
MATCH (n) 
RETURN labels(n), keys(n), n 
LIMIT 5
```

## Step 5: Find Highly Connected Nodes
```cypher
MATCH (n)
WITH n, size((n)--()) as degree
WHERE degree > 10
RETURN labels(n), n, degree
ORDER BY degree DESC
LIMIT 10
```

## Step 6: Analyze Data Quality
```cypher
// Check for nodes without properties
MATCH (n) 
WHERE size(keys(n)) = 0 
RETURN labels(n), count(n)
```

## Common Patterns to Look For:
- Central hub nodes (high connectivity)
- Isolated components
- Missing or null properties
- Data distribution patterns
- Temporal patterns (if timestamps exist)

Start with these queries to understand your data structure and content.
"""
    
    return PromptResult(
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=prompt_text)
            )
        ]
    )

@mcp.prompt()
async def performance_optimization_tips() -> PromptResult:
    """
    Provides tips for optimizing Cypher queries and database performance.
    """
    
    prompt_text = """Neo4j Performance Optimization Guide for ConnectIQ Database:

## Index Management
```cypher
// Check existing indexes
SHOW INDEXES

// Create indexes for frequently queried properties
CREATE INDEX FOR (n:NodeType) ON (n.propertyName)

// Create composite indexes for multi-property queries
CREATE INDEX FOR (n:NodeType) ON (n.prop1, n.prop2)
```

## Query Optimization Techniques

### 1. Use PROFILE and EXPLAIN
```cypher
PROFILE MATCH (n:Person {name: 'John'}) RETURN n
EXPLAIN MATCH (n:Person {name: 'John'}) RETURN n
```

### 2. Efficient Filtering
```cypher
// Good: Filter early
MATCH (p:Person {active: true})
WHERE p.age > 25
RETURN p

// Avoid: Late filtering
MATCH (p:Person)
WHERE p.active = true AND p.age > 25
RETURN p
```

### 3. Limit Result Sets
```cypher
// Always use LIMIT for exploration
MATCH (n) RETURN n LIMIT 100

// Use pagination for large datasets
MATCH (n:Person)
RETURN n
ORDER BY n.created
SKIP 1000 LIMIT 100
```

### 4. Efficient Relationship Traversal
```cypher
// Good: Specify relationship direction
MATCH (a)-[:FOLLOWS]->(b)

// Avoid: Bidirectional when not needed
MATCH (a)-[:FOLLOWS]-(b)
```

### 5. Use Parameters
```cypher
// Good: Parameterized query
MATCH (p:Person {id: $person_id}) RETURN p

// Avoid: String concatenation
// MATCH (p:Person {id: '" + person_id + "'}) RETURN p
```

## Memory Management
- Use `PERIODIC COMMIT` for large writes
- Break large operations into smaller batches
- Monitor heap usage and tune accordingly

## Common Anti-patterns to Avoid
- Cartesian products (missing WHERE clauses)
- Unnecessary OPTIONAL MATCH
- Missing indexes on frequently queried properties
- Not using relationship types for filtering
"""
    
    return PromptResult(
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=prompt_text)
            )
        ]
    )

# Cleanup function
async def cleanup():
    """Cleanup resources on shutdown"""
    await close_driver()

# Register cleanup
import atexit
atexit.register(lambda: asyncio.run(cleanup()))
