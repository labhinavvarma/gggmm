"""
Clean MCP Server for Neo4j Operations
NO Streamlit dependencies - Pure FastAPI only
Run this FIRST on port 8000
"""

import asyncio
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import AsyncGraphDatabase, AsyncDriver
import traceback
import uvicorn

# ============================================
# 🔧 CONFIGURATION - CHANGE THESE VALUES
# ============================================

# Neo4j Database Configuration
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_neo4j_password"  # ⚠️ CHANGE THIS!
NEO4J_DATABASE = "neo4j"

# Server Configuration
MCP_SERVER_PORT = 8000
MCP_SERVER_HOST = "0.0.0.0"

# Logging Configuration
LOG_LEVEL = "INFO"
ENABLE_DEBUG = True

# ============================================

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_server")

print("🔧 MCP Server Configuration:")
print(f"   Neo4j URI: {NEO4J_URI}")
print(f"   Neo4j User: {NEO4J_USER}")
print(f"   Neo4j Database: {NEO4J_DATABASE}")
print(f"   Server Port: {MCP_SERVER_PORT}")
print(f"   Password Length: {len(NEO4J_PASSWORD)} characters")

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

# FastAPI app
app = FastAPI(
    title="MCP Neo4j Server",
    description="MCP Server for Neo4j operations - NO Streamlit dependencies",
    version="1.0.0"
)

class CypherRequest(BaseModel):
    query: str
    params: dict = {}

@app.post("/read_neo4j_cypher")
async def read_neo4j_cypher(request: CypherRequest):
    """Execute read-only Cypher queries"""
    if driver is None:
        raise HTTPException(status_code=500, detail="Neo4j driver not initialized")
    
    try:
        logger.info(f"Executing READ query: {request.query}")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run(request.query, request.params)
            records = await result.data()
            
        logger.info(f"Query returned {len(records)} records")
        return records
        
    except Exception as e:
        logger.error(f"Read query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/write_neo4j_cypher")
async def write_neo4j_cypher(request: CypherRequest):
    """Execute write Cypher queries"""
    if driver is None:
        raise HTTPException(status_code=500, detail="Neo4j driver not initialized")
    
    try:
        logger.info(f"Executing WRITE query: {request.query}")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run(request.query, request.params)
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
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/get_neo4j_schema")
async def get_neo4j_schema():
    """Get database schema"""
    if driver is None:
        raise HTTPException(status_code=500, detail="Neo4j driver not initialized")
    
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if driver is None:
        return {
            "status": "unhealthy",
            "neo4j": {"status": "driver_not_initialized"},
            "server": {"port": MCP_SERVER_PORT}
        }
    
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run("RETURN 1 as test")
            record = await result.single()
            
        neo4j_status = "connected" if record and record["test"] == 1 else "test_failed"
        
        return {
            "status": "healthy",
            "neo4j": {
                "status": neo4j_status,
                "uri": NEO4J_URI,
                "database": NEO4J_DATABASE,
                "user": NEO4J_USER
            },
            "server": {
                "port": MCP_SERVER_PORT,
                "host": MCP_SERVER_HOST
            }
        }
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy", 
            "neo4j": {
                "status": "disconnected",
                "error": str(e)
            },
            "server": {"port": MCP_SERVER_PORT}
        }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "MCP Neo4j Server",
        "version": "1.0.0",
        "description": "Pure FastAPI MCP Server - No Streamlit dependencies",
        "endpoints": {
            "health": "/health",
            "read_cypher": "/read_neo4j_cypher",
            "write_cypher": "/write_neo4j_cypher",
            "schema": "/get_neo4j_schema"
        },
        "neo4j": {
            "uri": NEO4J_URI,
            "database": NEO4J_DATABASE,
            "user": NEO4J_USER
        }
    }

@app.on_event("startup")
async def startup_event():
    """Test connection on startup"""
    print("🚀 Starting MCP Neo4j Server...")
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
            
            # Test a simple query to count nodes
            try:
                async with driver.session(database=NEO4J_DATABASE) as session:
                    count_result = await session.run("MATCH (n) RETURN count(n) as node_count")
                    count_record = await count_result.single()
                    node_count = count_record["node_count"] if count_record else 0
                    
                print(f"📊 Found {node_count} nodes in the database")
            except Exception as e:
                print(f"⚠️  Could not count nodes: {e}")
            
        else:
            print("❌ Neo4j connection test failed - query returned unexpected result")
            
    except Exception as e:
        print("❌ Neo4j connection failed!")
        print(f"   Error: {e}")
        print("🔧 Please check your configuration:")
        print(f"   📍 URI: {NEO4J_URI}")
        print(f"   👤 User: {NEO4J_USER}")
        print(f"   🔐 Password: Check if correct")
        print(f"   🗄️  Database: {NEO4J_DATABASE}")

@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown"""
    print("🛑 Shutting down MCP Neo4j Server...")
    if driver:
        await driver.close()
        print("✅ Neo4j driver closed")

def main():
    """Main function to run the server"""
    print("=" * 60)
    print("🧠 MCP NEO4J SERVER - PURE FASTAPI")
    print("=" * 60)
    print("🔧 Configuration:")
    print(f"   📍 Neo4j URI: {NEO4J_URI}")
    print(f"   👤 Neo4j User: {NEO4J_USER}")
    print(f"   🗄️  Neo4j Database: {NEO4J_DATABASE}")
    print(f"   🌐 Server: {MCP_SERVER_HOST}:{MCP_SERVER_PORT}")
    print("=" * 60)
    
    # Final checks
    if NEO4J_PASSWORD == "your_neo4j_password":
        print("⚠️  WARNING: You're using the default password!")
        print("⚠️  Please change NEO4J_PASSWORD at the top of this file")
    
    if driver is None:
        print("❌ Cannot start server - Neo4j driver failed to initialize")
        return
    
    print("🚀 Starting server...")
    
    try:
        uvicorn.run(
            app,
            host=MCP_SERVER_HOST,
            port=MCP_SERVER_PORT,
            log_level=LOG_LEVEL.lower(),
            reload=False  # Disable reload to avoid import issues
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

if __name__ == "__main__":
    main()
