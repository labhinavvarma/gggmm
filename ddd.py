"""
Fixed FastAPI Server with proper node deletion and counter handling
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
    title="Neo4j FastAPI Server (Fixed Deletions)",
    description="FastAPI server with proper Neo4j operations and deletion handling",
    version="1.0.1"
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
    summary: Optional[Dict[str, Any]] = None  # Added summary field

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
    summary: Optional[Dict[str, Any]] = None  # Added summary field

# ============================================
# FIXED TOOL FUNCTIONS
# ============================================

async def read_neo4j_cypher_tool(query: str, params: dict = {}) -> Dict[str, Any]:
    """
    Execute read-only Cypher queries against Neo4j database.
    
    Args:
        query: The Cypher query to execute (MATCH, RETURN, WHERE, etc.)
        params: Optional parameters for the query
        
    Returns:
        Dictionary with data and summary information
    """
    if driver is None:
        raise Exception("Neo4j driver not initialized")
    
    try:
        logger.info(f"Executing READ query: {query}")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run(query, params)
            records = await result.data()
            summary = await result.consume()
            
        response = {
            "data": records,
            "summary": {
                "query_type": "READ",
                "records_returned": len(records),
                "query_time_ms": summary.result_available_after + summary.result_consumed_after
            }
        }
        
        logger.info(f"READ query returned {len(records)} records")
        return response
        
    except Exception as e:
        logger.error(f"Read query failed: {e}")
        raise Exception(f"Query failed: {str(e)}")

async def write_neo4j_cypher_tool(query: str, params: dict = {}) -> Dict[str, Any]:
    """
    Execute write Cypher queries against Neo4j database with proper deletion tracking.
    
    Args:
        query: The Cypher query to execute (CREATE, MERGE, SET, DELETE, etc.)
        params: Optional parameters for the query
        
    Returns:
        Dictionary with data and detailed summary of changes
    """
    if driver is None:
        raise Exception("Neo4j driver not initialized")
    
    try:
        logger.info(f"Executing WRITE query: {query}")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run(query, params)
            
            # Get any returned data first
            records = []
            try:
                records = await result.data()
            except:
                # Some write operations don't return data
                pass
            
            # Get the summary with counters
            summary = await result.consume()
            counters = summary.counters
            
        # Build detailed response with all counters
        response = {
            "data": records,
            "summary": {
                "query_type": "WRITE",
                "success": True,
                "changes": {
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
                    "constraints_removed": counters.constraints_removed
                },
                "total_changes": (
                    counters.nodes_created + counters.nodes_deleted +
                    counters.relationships_created + counters.relationships_deleted +
                    counters.properties_set + counters.labels_added + counters.labels_removed
                ),
                "query_time_ms": summary.result_available_after + summary.result_consumed_after,
                "records_returned": len(records)
            }
        }
        
        changes = response["summary"]["changes"]
        logger.info(f"WRITE query completed - Nodes: +{changes['nodes_created']}/-{changes['nodes_deleted']}, "
                   f"Rels: +{changes['relationships_created']}/-{changes['relationships_deleted']}, "
                   f"Props: {changes['properties_set']}")
        
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
                    return {
                        "data": schema,
                        "summary": {
                            "query_type": "SCHEMA",
                            "source": "apoc.meta.schema"
                        }
                    }
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
            "property_keys": prop_keys
        }
        
        response = {
            "data": schema,
            "summary": {
                "query_type": "SCHEMA",
                "source": "fallback_queries",
                "labels_count": len(labels),
                "relationship_types_count": len(rel_types),
                "property_keys_count": len(prop_keys)
            }
        }
        
        logger.info(f"Schema fetched: {len(labels)} labels, {len(rel_types)} rel types, {len(prop_keys)} properties")
        return response
        
    except Exception as e:
        logger.error(f"Schema fetch failed: {e}")
        return {
            "data": {
                "labels": [],
                "relationship_types": [],
                "property_keys": [],
                "error": f"Schema fetch failed: {str(e)}"
            },
            "summary": {
                "query_type": "SCHEMA",
                "source": "error"
            }
        }

# ============================================
# STARTUP/SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 Starting Fixed FastAPI Neo4j Server...")
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
    print(f"🌐 Fixed FastAPI server ready on http://localhost:{SERVER_PORT}")
    print("📋 Available endpoints:")
    print("   • GET  /health - Health check")
    print("   • POST /read_neo4j_cypher - Execute read queries")
    print("   • POST /write_neo4j_cypher - Execute write queries (fixed deletions)")
    print("   • POST /get_neo4j_schema - Get database schema")
    print("   • GET  /test/* - Test endpoints for deletion verification")
    print("=" * 50)

@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown"""
    print("🛑 Shutting down Fixed FastAPI Neo4j Server...")
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
        "service": "Fixed FastAPI Neo4j Server",
        "version": "1.0.1",
        "description": "FastAPI server with proper Neo4j operations and deletion handling",
        "architecture": "FastAPI + Neo4j",
        "fixes": [
            "Proper node deletion tracking",
            "Detailed operation summaries",
            "Fixed counter reporting",
            "Better error handling for write operations"
        ],
        "endpoints": {
            "health": "/health - Health check",
            "read_cypher": "/read_neo4j_cypher - Execute read queries",
            "write_cypher": "/write_neo4j_cypher - Execute write queries (fixed)",
            "schema": "/get_neo4j_schema - Get database schema",
            "test_deletion": "/test/test-deletion - Test deletion operations",
            "docs": "/docs - API documentation"
        }
    }

@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    """Health check endpoint"""
    if driver is None:
        return HealthResponse(
            status="unhealthy",
            neo4j={"status": "driver_not_initialized"},
            server={"port": SERVER_PORT, "type": "FastAPI-Fixed"},
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
                "type": "FastAPI-Fixed",
                "version": "1.0.1"
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
            server={"port": SERVER_PORT, "type": "FastAPI-Fixed"},
            tools=[]
        )

@app.post("/read_neo4j_cypher", response_model=CypherResponse, summary="Execute read query")
async def read_cypher_endpoint(request: CypherRequest):
    """Execute read-only Cypher queries"""
    try:
        result = await read_neo4j_cypher_tool(request.query, request.params)
        return CypherResponse(
            success=True, 
            data=result["data"],
            summary=result["summary"]
        )
    except Exception as e:
        logger.error(f"Read cypher endpoint error: {e}")
        return CypherResponse(success=False, error=str(e))

@app.post("/write_neo4j_cypher", response_model=CypherResponse, summary="Execute write query (fixed)")
async def write_cypher_endpoint(request: CypherRequest):
    """Execute write Cypher queries with proper deletion tracking"""
    try:
        result = await write_neo4j_cypher_tool(request.query, request.params)
        return CypherResponse(
            success=True, 
            data=result["data"],
            summary=result["summary"]
        )
    except Exception as e:
        logger.error(f"Write cypher endpoint error: {e}")
        return CypherResponse(success=False, error=str(e))

@app.post("/get_neo4j_schema", response_model=CypherResponse, summary="Get database schema")
async def get_schema_endpoint():
    """Get database schema information"""
    try:
        result = await get_neo4j_schema_tool()
        return CypherResponse(
            success=True, 
            data=result["data"],
            summary=result["summary"]
        )
    except Exception as e:
        logger.error(f"Get schema endpoint error: {e}")
        return CypherResponse(success=False, error=str(e))

# ============================================
# DELETION TEST ENDPOINTS
# ============================================

@app.get("/test/test-deletion", summary="Test deletion operations")
async def test_deletion():
    """Test creation and deletion to verify counters work"""
    try:
        # Step 1: Create test nodes
        create_result = await write_neo4j_cypher_tool(
            "CREATE (t1:DeletionTest {name: 'Test1', created: datetime()}), (t2:DeletionTest {name: 'Test2', created: datetime()}) RETURN t1, t2",
            {}
        )
        
        # Step 2: Count test nodes
        count_result = await read_neo4j_cypher_tool(
            "MATCH (t:DeletionTest) RETURN count(t) as count",
            {}
        )
        
        # Step 3: Delete test nodes
        delete_result = await write_neo4j_cypher_tool(
            "MATCH (t:DeletionTest) DETACH DELETE t",
            {}
        )
        
        # Step 4: Verify deletion
        verify_result = await read_neo4j_cypher_tool(
            "MATCH (t:DeletionTest) RETURN count(t) as count",
            {}
        )
        
        return {
            "success": True,
            "test_steps": {
                "1_create": {
                    "query": "CREATE test nodes",
                    "nodes_created": create_result["summary"]["changes"]["nodes_created"],
                    "summary": create_result["summary"]
                },
                "2_count_before": {
                    "query": "Count before deletion",
                    "count": count_result["data"][0]["count"],
                    "summary": count_result["summary"]
                },
                "3_delete": {
                    "query": "DELETE test nodes",
                    "nodes_deleted": delete_result["summary"]["changes"]["nodes_deleted"],
                    "summary": delete_result["summary"]
                },
                "4_count_after": {
                    "query": "Count after deletion",
                    "count": verify_result["data"][0]["count"],
                    "summary": verify_result["summary"]
                }
            },
            "conclusion": {
                "deletion_working": delete_result["summary"]["changes"]["nodes_deleted"] > 0,
                "nodes_created": create_result["summary"]["changes"]["nodes_created"],
                "nodes_deleted": delete_result["summary"]["changes"]["nodes_deleted"],
                "verification_passed": verify_result["data"][0]["count"] == 0
            }
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/test/create-test-nodes", summary="Create test nodes for deletion")
async def create_test_nodes():
    """Create some test nodes that can be deleted"""
    try:
        result = await write_neo4j_cypher_tool(
            "UNWIND range(1, 5) as i CREATE (t:TestNode {id: i, name: 'TestNode' + i, created: datetime()}) RETURN count(t) as created",
            {}
        )
        
        return {
            "success": True,
            "message": "Created test nodes for deletion testing",
            "nodes_created": result["summary"]["changes"]["nodes_created"],
            "summary": result["summary"]
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/test/delete-test-nodes", summary="Delete test nodes")
async def delete_test_nodes():
    """Delete test nodes and show deletion counts"""
    try:
        # First count existing test nodes
        count_result = await read_neo4j_cypher_tool(
            "MATCH (t:TestNode) RETURN count(t) as before_count",
            {}
        )
        
        # Delete the nodes
        delete_result = await write_neo4j_cypher_tool(
            "MATCH (t:TestNode) DETACH DELETE t",
            {}
        )
        
        # Count remaining nodes
        verify_result = await read_neo4j_cypher_tool(
            "MATCH (t:TestNode) RETURN count(t) as after_count",
            {}
        )
        
        before_count = count_result["data"][0]["before_count"] if count_result["data"] else 0
        after_count = verify_result["data"][0]["after_count"] if verify_result["data"] else 0
        
        return {
            "success": True,
            "message": "Deleted test nodes",
            "before_deletion": before_count,
            "nodes_deleted": delete_result["summary"]["changes"]["nodes_deleted"],
            "after_deletion": after_count,
            "deletion_successful": delete_result["summary"]["changes"]["nodes_deleted"] > 0,
            "summary": delete_result["summary"]
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/test/count-all-nodes", summary="Count all nodes by label")
async def count_all_nodes():
    """Count all nodes grouped by label"""
    try:
        result = await read_neo4j_cypher_tool(
            "MATCH (n) RETURN labels(n)[0] as label, count(n) as count ORDER BY count DESC",
            {}
        )
        
        total_result = await read_neo4j_cypher_tool(
            "MATCH (n) RETURN count(n) as total",
            {}
        )
        
        total_count = total_result["data"][0]["total"] if total_result["data"] else 0
        
        return {
            "success": True,
            "total_nodes": total_count,
            "nodes_by_label": result["data"],
            "summary": result["summary"]
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """Main function to run the FastAPI server"""
    print("=" * 60)
    print("🔧 FIXED FASTAPI NEO4J SERVER")
    print("=" * 60)
    print("🏗️  Architecture: FastAPI + Neo4j (Fixed Deletions)")
    print("🔧 Configuration:")
    print(f"   📍 Neo4j URI: {NEO4J_URI}")
    print(f"   👤 Neo4j User: {NEO4J_USER}")
    print(f"   🗄️  Neo4j Database: {NEO4J_DATABASE}")
    print(f"   🌐 Server: {SERVER_HOST}:{SERVER_PORT}")
    print("🛠️  Fixes:")
    print("   • Proper node deletion tracking")
    print("   • Detailed operation summaries")
    print("   • Fixed counter reporting")
    print("   • Test endpoints for verification")
    print("=" * 60)
    
    # Final checks
    if NEO4J_PASSWORD == "your_neo4j_password":
        print("⚠️  WARNING: You're using the default Neo4j password!")
        print("⚠️  Please change NEO4J_PASSWORD at the top of this file")
    
    if driver is None:
        print("❌ Cannot start server - Neo4j driver failed to initialize")
        return
    
    print("🚀 Starting fixed FastAPI server...")
    
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
