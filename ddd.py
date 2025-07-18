"""
Pure FastAPI Server with Neo4j Tools - No FastMCP dependency
Run this on port 8000
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from neo4j import AsyncGraphDatabase, AsyncDriver
import uvicorn

# ============================================
# 🔧 CONFIGURATION - CHANGE THESE VALUES
# ============================================

# Neo4j Configuration
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_neo4j_password"  # ⚠️ CHANGE THIS!
NEO4J_DATABASE = "neo4j"

# Server Configuration
SERVER_PORT = 8000
SERVER_HOST = "0.0.0.0"

# ============================================

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastapi_server")

print(f"🔧 FastAPI Server Configuration:")
print(f"   Neo4j URI: {NEO4J_URI}")
print(f"   Neo4j User: {NEO4J_USER}")
print(f"   Neo4j Database: {NEO4J_DATABASE}")
print(f"   Server Port: {SERVER_PORT}")

# Initialize Neo4j driver
try:
    driver = AsyncGraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        connection_timeout=10,
        max_connection_lifetime=3600,
        max_connection_pool_size=50
    )
    print("✅ Neo4j driver initialized")
except Exception as e:
    print(f"❌ Failed to initialize Neo4j driver: {e}")
    driver = None

# Initialize FastAPI
app = FastAPI(
    title="Neo4j FastAPI Server",
    description="Simple FastAPI server with Neo4j tool endpoints",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# PYDANTIC MODELS
# ============================================

class CypherRequest(BaseModel):
    query: str
    params: dict = {}

class CypherResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    neo4j: Dict[str, Any]
    server: Dict[str, Any]
    tools: List[str]

class ToolResponse(BaseModel):
    tool_name: str
    description: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None

# ============================================
# TOOL FUNCTIONS
# ============================================

async def read_neo4j_cypher_tool(query: str, params: dict = {}) -> List[Dict[str, Any]]:
    """
    Execute read-only Cypher queries against Neo4j database.
    
    Args:
        query: The Cypher query to execute (MATCH, RETURN, WHERE, etc.)
        params: Optional parameters for the query
        
    Returns:
        List of records returned by the query
    """
    if driver is None:
        raise Exception("Neo4j driver not initialized")
    
    try:
        logger.info(f"Executing READ query: {query}")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run(query, params)
            records = await result.data()
            
        logger.info(f"Query returned {len(records)} records")
        return records
        
    except Exception as e:
        logger.error(f"Read query failed: {e}")
        raise Exception(f"Query failed: {str(e)}")

async def write_neo4j_cypher_tool(query: str, params: dict = {}) -> Dict[str, Any]:
    """
    Execute write Cypher queries against Neo4j database.
    
    Args:
        query: The Cypher query to execute (CREATE, MERGE, SET, DELETE, etc.)
        params: Optional parameters for the query
        
    Returns:
        Summary of changes made to the database
    """
    if driver is None:
        raise Exception("Neo4j driver not initialized")
    
    try:
        logger.info(f"Executing WRITE query: {query}")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run(query, params)
            summary = await result.consume()
            
        # Get counters
        counters = summary.counters
        
        response = {
            "success": True,
            "nodes_created": counters.nodes_created,
            "nodes_deleted": counters.nodes_deleted,
            "relationships_created": counters.relationships_created,
            "relationships_deleted": counters.relationships_deleted,
            "properties_set": counters.properties_set,
            "labels_added": counters.labels_added,
            "labels_removed": counters.labels_removed
        }
        
        logger.info(f"Write query completed: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Write query failed: {e}")
        raise Exception(f"Query failed: {str(e)}")

async def get_neo4j_schema_tool() -> Dict[str, Any]:
    """
    Get the schema of the Neo4j database including labels, relationship types, and properties.
    
    Returns:
        Dictionary containing database schema information
    """
    if driver is None:
        raise Exception("Neo4j driver not initialized")
    
    try:
        logger.info("Fetching database schema")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            # Try APOC first
            try:
                apoc_result = await session.run("CALL apoc.meta.schema() YIELD value RETURN value")
                apoc_record = await apoc_result.single()
                if apoc_record:
                    schema = apoc_record["value"]
                    logger.info("Schema fetched using APOC")
                    return schema
            except Exception:
                logger.info("APOC not available, using fallback queries")
            
            # Fallback to basic queries
            labels_result = await session.run("CALL db.labels() YIELD label RETURN collect(label) as labels")
            labels_record = await labels_result.single()
            labels = labels_record["labels"] if labels_record else []
            
            rels_result = await session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types")
            rels_record = await rels_result.single()
            rel_types = rels_record["types"] if rels_record else []
            
            props_result = await session.run("CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) as keys")
            props_record = await props_result.single()
            prop_keys = props_record["keys"] if props_record else []
        
        schema = {
            "labels": labels,
            "relationship_types": rel_types,
            "property_keys": prop_keys,
            "source": "fallback_queries"
        }
        
        logger.info(f"Schema fetched: {len(labels)} labels, {len(rel_types)} rel types, {len(prop_keys)} properties")
        return schema
        
    except Exception as e:
        logger.error(f"Schema fetch failed: {e}")
        return {
            "labels": [],
            "relationship_types": [],
            "property_keys": [],
            "error": f"Schema fetch failed: {str(e)}",
            "source": "error"
        }

# ============================================
# STARTUP/SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 Starting FastAPI Neo4j Server...")
    print("=" * 50)
    
    if NEO4J_PASSWORD == "your_neo4j_password":
        print("⚠️  WARNING: Using default password!")
        print("⚠️  Please change NEO4J_PASSWORD in the configuration section")
    
    if driver is None:
        print("❌ Neo4j driver initialization failed")
        return
    
    # Test Neo4j connection
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run("RETURN 1 as test")
            record = await result.single()
            
        if record and record["test"] == 1:
            print("✅ Neo4j connection successful!")
            
            # Count nodes
            try:
                async with driver.session(database=NEO4J_DATABASE) as session:
                    count_result = await session.run("MATCH (n) RETURN count(n) as node_count")
                    count_record = await count_result.single()
                    node_count = count_record["node_count"] if count_record else 0
                    
                print(f"📊 Found {node_count} nodes in the database")
            except Exception as e:
                print(f"⚠️  Could not count nodes: {e}")
            
        else:
            print("❌ Neo4j connection test failed")
            
    except Exception as e:
        print("❌ Neo4j connection failed!")
        print(f"   Error: {e}")
    
    print("=" * 50)
    print(f"🌐 FastAPI server ready on http://localhost:{SERVER_PORT}")
    print("📋 Available endpoints:")
    print("   • GET  /health - Health check")
    print("   • POST /read_neo4j_cypher - Execute read queries")
    print("   • POST /write_neo4j_cypher - Execute write queries")
    print("   • POST /get_neo4j_schema - Get database schema")
    print("   • GET  /tools - List available tools")
    print("   • GET  /test/count-nodes - Quick node count")
    print("   • GET  /test/node-labels - Quick labels test")
    print("   • GET  /docs - API documentation")
    print("=" * 50)

@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown"""
    print("🛑 Shutting down FastAPI Neo4j Server...")
    if driver:
        await driver.close()
        print("✅ Neo4j driver closed")

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", summary="Root endpoint")
async def root():
    """Root endpoint with server information"""
    return {
        "service": "FastAPI Neo4j Server",
        "version": "1.0.0",
        "description": "Simple FastAPI server with Neo4j tool endpoints",
        "architecture": "FastAPI + Neo4j",
        "endpoints": {
            "health": "/health - Health check",
            "read_cypher": "/read_neo4j_cypher - Execute read queries",
            "write_cypher": "/write_neo4j_cypher - Execute write queries",
            "schema": "/get_neo4j_schema - Get database schema",
            "tools": "/tools - List available tools",
            "docs": "/docs - API documentation"
        },
        "tools": {
            "read_neo4j_cypher": "Execute read-only Cypher queries",
            "write_neo4j_cypher": "Execute write Cypher queries", 
            "get_neo4j_schema": "Get database schema"
        },
        "neo4j": {
            "uri": NEO4J_URI,
            "database": NEO4J_DATABASE,
            "user": NEO4J_USER
        }
    }

@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    """Health check endpoint"""
    if driver is None:
        return HealthResponse(
            status="unhealthy",
            neo4j={"status": "driver_not_initialized"},
            server={"port": SERVER_PORT, "type": "FastAPI"},
            tools=[]
        )
    
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run("RETURN 1 as test")
            record = await result.single()
            
        neo4j_status = "connected" if record and record["test"] == 1 else "test_failed"
        
        return HealthResponse(
            status="healthy",
            neo4j={
                "status": neo4j_status,
                "uri": NEO4J_URI,
                "database": NEO4J_DATABASE,
                "user": NEO4J_USER
            },
            server={
                "port": SERVER_PORT,
                "host": SERVER_HOST,
                "type": "FastAPI"
            },
            tools=["read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"]
        )
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            neo4j={
                "status": "disconnected",
                "error": str(e)
            },
            server={"port": SERVER_PORT, "type": "FastAPI"},
            tools=[]
        )

@app.get("/tools", summary="List available tools")
async def list_tools():
    """List all available tools"""
    return {
        "tools": [
            {
                "name": "read_neo4j_cypher",
                "description": "Execute read-only Cypher queries against Neo4j database",
                "endpoint": "/read_neo4j_cypher",
                "method": "POST",
                "parameters": {
                    "query": "string - The Cypher query to execute",
                    "params": "object - Optional parameters for the query"
                }
            },
            {
                "name": "write_neo4j_cypher",
                "description": "Execute write Cypher queries against Neo4j database",
                "endpoint": "/write_neo4j_cypher",
                "method": "POST",
                "parameters": {
                    "query": "string - The Cypher query to execute",
                    "params": "object - Optional parameters for the query"
                }
            },
            {
                "name": "get_neo4j_schema",
                "description": "Get the schema of the Neo4j database",
                "endpoint": "/get_neo4j_schema",
                "method": "POST",
                "parameters": {}
            }
        ]
    }

@app.post("/read_neo4j_cypher", response_model=CypherResponse, summary="Execute read query")
async def read_cypher_endpoint(request: CypherRequest):
    """Execute read-only Cypher queries"""
    try:
        result = await read_neo4j_cypher_tool(request.query, request.params)
        return CypherResponse(success=True, data=result)
    except Exception as e:
        logger.error(f"Read cypher endpoint error: {e}")
        return CypherResponse(success=False, error=str(e))

@app.post("/write_neo4j_cypher", response_model=CypherResponse, summary="Execute write query")
async def write_cypher_endpoint(request: CypherRequest):
    """Execute write Cypher queries"""
    try:
        result = await write_neo4j_cypher_tool(request.query, request.params)
        return CypherResponse(success=True, data=result)
    except Exception as e:
        logger.error(f"Write cypher endpoint error: {e}")
        return CypherResponse(success=False, error=str(e))

@app.post("/get_neo4j_schema", response_model=CypherResponse, summary="Get database schema")
async def get_schema_endpoint():
    """Get database schema information"""
    try:
        result = await get_neo4j_schema_tool()
        return CypherResponse(success=True, data=result)
    except Exception as e:
        logger.error(f"Get schema endpoint error: {e}")
        return CypherResponse(success=False, error=str(e))

# ============================================
# TOOL ENDPOINTS WITH DETAILED RESPONSES
# ============================================

@app.post("/tool/read_cypher", response_model=ToolResponse, summary="Tool: Read Cypher")
async def tool_read_cypher(request: CypherRequest):
    """Tool-style endpoint for read operations with detailed response"""
    try:
        result = await read_neo4j_cypher_tool(request.query, request.params)
        return ToolResponse(
            tool_name="read_neo4j_cypher",
            description="Executed read-only Cypher query",
            success=True,
            result=result
        )
    except Exception as e:
        return ToolResponse(
            tool_name="read_neo4j_cypher",
            description="Failed to execute read-only Cypher query",
            success=False,
            error=str(e)
        )

@app.post("/tool/write_cypher", response_model=ToolResponse, summary="Tool: Write Cypher")
async def tool_write_cypher(request: CypherRequest):
    """Tool-style endpoint for write operations with detailed response"""
    try:
        result = await write_neo4j_cypher_tool(request.query, request.params)
        return ToolResponse(
            tool_name="write_neo4j_cypher",
            description="Executed write Cypher query",
            success=True,
            result=result
        )
    except Exception as e:
        return ToolResponse(
            tool_name="write_neo4j_cypher",
            description="Failed to execute write Cypher query",
            success=False,
            error=str(e)
        )

@app.post("/tool/get_schema", response_model=ToolResponse, summary="Tool: Get Schema")
async def tool_get_schema():
    """Tool-style endpoint for schema operations with detailed response"""
    try:
        result = await get_neo4j_schema_tool()
        return ToolResponse(
            tool_name="get_neo4j_schema",
            description="Retrieved Neo4j database schema",
            success=True,
            result=result
        )
    except Exception as e:
        return ToolResponse(
            tool_name="get_neo4j_schema",
            description="Failed to retrieve Neo4j database schema",
            success=False,
            error=str(e)
        )

# ============================================
# QUICK TEST ENDPOINTS
# ============================================

@app.get("/test/count-nodes", summary="Quick test: Count nodes")
async def test_count_nodes():
    """Quick test endpoint to count nodes"""
    try:
        result = await read_neo4j_cypher_tool("MATCH (n) RETURN count(n) as node_count", {})
        count = result[0]["node_count"] if result and len(result) > 0 else 0
        return {"success": True, "node_count": count, "query": "MATCH (n) RETURN count(n)"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/test/node-labels", summary="Quick test: Get node labels")
async def test_node_labels():
    """Quick test endpoint to get node labels"""
    try:
        result = await get_neo4j_schema_tool()
        labels = result.get("labels", [])
        return {"success": True, "labels": labels, "count": len(labels)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/test/create-test-node", summary="Quick test: Create test node")
async def test_create_node():
    """Quick test endpoint to create a test node"""
    try:
        query = "CREATE (t:TestNode {name: 'FastAPI Test', created: datetime()}) RETURN t"
        result = await write_neo4j_cypher_tool(query, {})
        return {"success": True, "result": result, "query": query}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/test/delete-test-nodes", summary="Quick test: Delete test nodes")
async def test_delete_nodes():
    """Quick test endpoint to delete test nodes"""
    try:
        query = "MATCH (t:TestNode) DELETE t"
        result = await write_neo4j_cypher_tool(query, {})
        return {"success": True, "result": result, "query": query}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """Main function to run the FastAPI server"""
    print("=" * 60)
    print("🌐 FASTAPI NEO4J SERVER")
    print("=" * 60)
    print("🏗️  Architecture: Pure FastAPI + Neo4j")
    print("🔧 Configuration:")
    print(f"   📍 Neo4j URI: {NEO4J_URI}")
    print(f"   👤 Neo4j User: {NEO4J_USER}")
    print(f"   🗄️  Neo4j Database: {NEO4J_DATABASE}")
    print(f"   🌐 Server: {SERVER_HOST}:{SERVER_PORT}")
    print("=" * 60)
    
    # Final checks
    if NEO4J_PASSWORD == "your_neo4j_password":
        print("⚠️  WARNING: You're using the default Neo4j password!")
        print("⚠️  Please change NEO4J_PASSWORD at the top of this file")
    
    if driver is None:
        print("❌ Cannot start server - Neo4j driver failed to initialize")
        return
    
    print("🚀 Starting FastAPI server...")
    print("📋 This server provides:")
    print("   • RESTful API endpoints for Neo4j operations")
    print("   • Tool-style endpoints with detailed responses")
    print("   • Quick test endpoints for debugging")
    print("   • Built-in documentation at /docs")
    print("   • Health checks and monitoring")
    
    try:
        uvicorn.run(
            app,
            host=SERVER_HOST,
            port=SERVER_PORT,
            log_level="info",
            reload=False
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

if __name__ == "__main__":
    main()
