import asyncio
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import AsyncGraphDatabase
import traceback

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_mcp_server")

# Neo4j Configuration
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_neo4j_password"  # Change this!
NEO4J_DATABASE = "neo4j"

# Initialize Neo4j driver
driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

app = FastAPI(title="Simple MCP Neo4j Server")

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
            "properties_set": counters.properties_set
        }
        
        logger.info(f"Write query completed: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Write query failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/get_neo4j_schema")
async def get_neo4j_schema():
    """Get database schema"""
    try:
        logger.info("Fetching database schema")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
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
            "property_keys": prop_keys
        }
        
        logger.info(f"Schema fetched: {len(labels)} labels, {len(rel_types)} rel types, {len(prop_keys)} properties")
        return schema
        
    except Exception as e:
        logger.error(f"Schema fetch failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return a basic response even if schema fetch fails
        return {
            "labels": [],
            "relationship_types": [],
            "property_keys": [],
            "error": f"Schema fetch failed: {str(e)}"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run("RETURN 1 as test")
            record = await result.single()
            
        if record and record["test"] == 1:
            return {"status": "healthy", "neo4j": "connected"}
        else:
            return {"status": "unhealthy", "neo4j": "test_failed"}
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "neo4j": "disconnected", "error": str(e)}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Simple MCP Neo4j Server",
        "status": "running",
        "endpoints": ["/read_neo4j_cypher", "/write_neo4j_cypher", "/get_neo4j_schema", "/health"]
    }

@app.on_event("startup")
async def startup_event():
    """Test connection on startup"""
    logger.info("🚀 Starting Simple MCP Neo4j Server...")
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run("RETURN 1 as test")
            record = await result.single()
            
        if record and record["test"] == 1:
            logger.info("✅ Neo4j connection successful")
        else:
            logger.error("❌ Neo4j connection test failed")
            
    except Exception as e:
        logger.error(f"❌ Neo4j connection failed: {e}")
        logger.error("Please check your Neo4j configuration!")

@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown"""
    logger.info("🛑 Shutting down Simple MCP Neo4j Server...")
    await driver.close()
    logger.info("✅ Neo4j driver closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("simple_mcpserver:app", host="0.0.0.0", port=8000, reload=True)
