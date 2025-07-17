"""
MCP Server with Hardcoded Configuration
Simple and direct - just change the values below and run!
"""

import asyncio
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import AsyncGraphDatabase, AsyncDriver
import traceback

# ============================================
# 🔧 HARDCODED CONFIGURATION - CHANGE THESE VALUES
# ============================================

# Neo4j Database Configuration
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_neo4j_password"  # ⚠️ CHANGE THIS!
NEO4J_DATABASE = "neo4j"

# Connection Settings
NEO4J_CONNECTION_TIMEOUT = 10
NEO4J_MAX_POOL_SIZE = 50
NEO4J_MAX_LIFETIME = 3600

# Server Configuration
MCP_SERVER_PORT = 8000
MCP_SERVER_HOST = "0.0.0.0"

# Logging Configuration
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
ENABLE_DEBUG = True

# Security Settings (for production)
NEO4J_ENCRYPTED = False
NEO4J_TRUST = "TRUST_ALL_CERTIFICATES"

# ============================================
# END OF CONFIGURATION
# ============================================

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hardcoded_mcp_server")

# Print configuration on startup
logger.info("🔧 MCP Server Configuration:")
logger.info(f"   Neo4j URI: {NEO4J_URI}")
logger.info(f"   Neo4j User: {NEO4J_USER}")
logger.info(f"   Neo4j Database: {NEO4J_DATABASE}")
logger.info(f"   Neo4j Password: {'*' * len(NEO4J_PASSWORD)}")
logger.info(f"   Server Port: {MCP_SERVER_PORT}")
logger.info(f"   Debug Mode: {ENABLE_DEBUG}")

# Initialize Neo4j driver with hardcoded configuration
driver = AsyncGraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
    connection_timeout=NEO4J_CONNECTION_TIMEOUT,
    max_connection_lifetime=NEO4J_MAX_LIFETIME,
    max_connection_pool_size=NEO4J_MAX_POOL_SIZE,
    encrypted=NEO4J_ENCRYPTED,
    trust=NEO4J_TRUST
)

app = FastAPI(
    title="Hardcoded MCP Neo4j Server",
    description="Simple MCP Server with hardcoded configuration",
    version="1.0.0"
)

class CypherRequest(BaseModel):
    query: str
    params: dict = {}

@app.post("/read_neo4j_cypher")
async def read_neo4j_cypher(request: CypherRequest):
    """Execute read-only Cypher queries"""
    try:
        logger.info(f"Executing READ query: {request.query}")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run(request.query, request.params)
            records = await result.data()
            
        logger.info(f"Query returned {len(records)} records")
        return records
        
    except Exception as e:
        logger.error(f"Read query failed: {e}")
        if ENABLE_DEBUG:
            logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/write_neo4j_cypher")
async def write_neo4j_cypher(request: CypherRequest):
    """Execute write Cypher queries"""
    try:
        logger.info(f"Executing WRITE query: {request.query}")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run(request.query, request.params)
            summary = result.consume()
            
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
        if ENABLE_DEBUG:
            logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/get_neo4j_schema")
async def get_neo4j_schema():
    """Get database schema"""
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
            except Exception as apoc_error:
                logger.info(f"APOC not available: {apoc_error}, using fallback queries")
            
            # Fallback to basic queries
            # Get node labels
            labels_result = await session.run("CALL db.labels() YIELD label RETURN collect(label) as labels")
            labels_record = await labels_result.single()
            labels = labels_record["labels"] if labels_record else []
            
            # Get relationship types
            rels_result = await session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types")
            rels_record = await rels_result.single()
            rel_types = rels_record["types"] if rels_record else []
            
            # Get property keys
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
        if ENABLE_DEBUG:
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return error info instead of failing
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
                "host": MCP_SERVER_HOST,
                "debug": ENABLE_DEBUG,
                "log_level": LOG_LEVEL
            },
            "configuration": "hardcoded"
        }
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy", 
            "neo4j": {
                "status": "disconnected",
                "error": str(e),
                "uri": NEO4J_URI,
                "database": NEO4J_DATABASE
            },
            "configuration": "hardcoded"
        }

@app.get("/config")
async def get_current_config():
    """Get current hardcoded configuration (without password)"""
    return {
        "configuration_type": "hardcoded",
        "neo4j": {
            "uri": NEO4J_URI,
            "user": NEO4J_USER,
            "password": "***HIDDEN***",
            "database": NEO4J_DATABASE,
            "connection_timeout": NEO4J_CONNECTION_TIMEOUT,
            "max_pool_size": NEO4J_MAX_POOL_SIZE,
            "max_lifetime": NEO4J_MAX_LIFETIME,
            "encrypted": NEO4J_ENCRYPTED
        },
        "server": {
            "port": MCP_SERVER_PORT,
            "host": MCP_SERVER_HOST,
            "log_level": LOG_LEVEL,
            "debug_enabled": ENABLE_DEBUG
        },
        "instructions": {
            "how_to_change": "Edit the values at the top of hardcoded_mcpserver.py",
            "required_changes": [
                "NEO4J_PASSWORD - Set your Neo4j password",
                "NEO4J_URI - Update if not using localhost",
                "NEO4J_USER - Update if not using 'neo4j'"
            ]
        }
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Hardcoded MCP Neo4j Server",
        "version": "1.0.0",
        "description": "Simple MCP Server with hardcoded configuration",
        "configuration": {
            "type": "hardcoded",
            "neo4j_uri": NEO4J_URI,
            "neo4j_database": NEO4J_DATABASE,
            "server_port": MCP_SERVER_PORT,
            "debug_enabled": ENABLE_DEBUG
        },
        "endpoints": {
            "read_cypher": "/read_neo4j_cypher - Execute read queries",
            "write_cypher": "/write_neo4j_cypher - Execute write queries", 
            "schema": "/get_neo4j_schema - Get database schema",
            "health": "/health - Check system health",
            "config": "/config - View current configuration"
        },
        "instructions": {
            "setup": [
                "1. Edit NEO4J_PASSWORD at the top of this file",
                "2. Update NEO4J_URI if not using localhost",
                "3. Restart the server"
            ]
        }
    }

@app.on_event("startup")
async def startup_event():
    """Test connection on startup"""
    logger.info("🚀 Starting Hardcoded MCP Neo4j Server...")
    logger.info("=" * 50)
    logger.info("📊 HARDCODED CONFIGURATION:")
    logger.info(f"   📍 Neo4j URI: {NEO4J_URI}")
    logger.info(f"   👤 Neo4j User: {NEO4J_USER}")
    logger.info(f"   🗄️  Neo4j Database: {NEO4J_DATABASE}")
    logger.info(f"   🔐 Password Length: {len(NEO4J_PASSWORD)} characters")
    logger.info(f"   🌐 Server: {MCP_SERVER_HOST}:{MCP_SERVER_PORT}")
    logger.info(f"   🔧 Debug Mode: {ENABLE_DEBUG}")
    logger.info("=" * 50)
    
    # Test Neo4j connection
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run("RETURN 1 as test")
            record = await result.single()
            
        if record and record["test"] == 1:
            logger.info("✅ Neo4j connection successful!")
            
            # Test a simple query
            async with driver.session(database=NEO4J_DATABASE) as session:
                count_result = await session.run("MATCH (n) RETURN count(n) as node_count")
                count_record = await count_result.single()
                node_count = count_record["node_count"] if count_record else 0
                
            logger.info(f"📊 Found {node_count} nodes in the database")
            
        else:
            logger.error("❌ Neo4j connection test failed - query returned unexpected result")
            
    except Exception as e:
        logger.error("❌ Neo4j connection failed!")
        logger.error(f"   Error: {e}")
        logger.error("🔧 Please check your hardcoded configuration:")
        logger.error(f"   📍 URI: {NEO4J_URI}")
        logger.error(f"   👤 User: {NEO4J_USER}")
        logger.error(f"   🔐 Password: Check if correct")
        logger.error(f"   🗄️  Database: {NEO4J_DATABASE}")
        logger.error("")
        logger.error("💡 Common fixes:")
        logger.error("   • Make sure Neo4j is running")
        logger.error("   • Check NEO4J_PASSWORD is correct")
        logger.error("   • Verify NEO4J_URI is accessible")
        logger.error("   • Ensure user has proper permissions")

@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown"""
    logger.info("🛑 Shutting down Hardcoded MCP Neo4j Server...")
    await driver.close()
    logger.info("✅ Neo4j driver closed")

if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("🧠 HARDCODED MCP NEO4J SERVER")
    logger.info("=" * 60)
    logger.info("🔧 Configuration is hardcoded in this file")
    logger.info("📝 To change settings, edit the variables at the top")
    logger.info(f"🌐 Starting server on {MCP_SERVER_HOST}:{MCP_SERVER_PORT}")
    logger.info("=" * 60)
    
    # Check if password is still default
    if NEO4J_PASSWORD == "your_neo4j_password":
        logger.warning("⚠️  WARNING: You're using the default password!")
        logger.warning("⚠️  Please change NEO4J_PASSWORD at the top of this file")
        logger.warning("⚠️  Current password: your_neo4j_password")
    
    uvicorn.run(
        "hardcoded_mcpserver:app", 
        host=MCP_SERVER_HOST, 
        port=MCP_SERVER_PORT, 
        reload=True,
        log_level=LOG_LEVEL.lower()
    )
