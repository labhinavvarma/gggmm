import asyncio
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncTransaction
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_neo4j_cypher")

try:
    from config import NEO4J_CONFIG, DEBUG_CONFIG
    NEO4J_URI = NEO4J_CONFIG["uri"]
    NEO4J_USER = NEO4J_CONFIG["user"]
    NEO4J_PASSWORD = NEO4J_CONFIG["password"]
    NEO4J_DATABASE = NEO4J_CONFIG["database"]
    ENABLE_DEBUG = DEBUG_CONFIG["enable_debug_logging"]
except ImportError:
    # Fallback to hardcoded values if config.py is not available
    NEO4J_URI = "neo4j://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "your_neo4j_password"
    NEO4J_DATABASE = "neo4j"
    ENABLE_DEBUG = True

# Set logging level based on config
if ENABLE_DEBUG:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

driver: AsyncDriver = AsyncGraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
)

app = FastAPI(title="MCP Neo4j Cypher API")

class CypherRequest(BaseModel):
    query: str
    params: dict = {}

@app.post("/read_neo4j_cypher")
async def read_neo4j_cypher(request: CypherRequest):
    try:
        logger.info(f"Executing read query: {request.query}")
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_read(_read, request.query, request.params or {})
        
        parsed_result = json.loads(result)
        logger.info(f"Read query returned {len(parsed_result)} records")
        return parsed_result
    except Exception as e:
        logger.error(f"Error in read_neo4j_cypher: {e}")
        logger.error(f"Query was: {request.query}")
        raise HTTPException(status_code=500, detail=f"Neo4j read error: {str(e)}")

@app.post("/write_neo4j_cypher")
async def write_neo4j_cypher(request: CypherRequest):
    try:
        logger.info(f"Executing write query: {request.query}")
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_write(_write, request.query, request.params or {})
        
        counters = result._summary.counters
        logger.info(f"Write query completed. Counters: {counters}")
        return {
            "result": str(counters),
            "success": True,
            "counters": {
                "nodes_created": counters.nodes_created,
                "nodes_deleted": counters.nodes_deleted,
                "relationships_created": counters.relationships_created,
                "relationships_deleted": counters.relationships_deleted,
                "properties_set": counters.properties_set,
                "labels_added": counters.labels_added,
                "labels_removed": counters.labels_removed
            }
        }
    except Exception as e:
        logger.error(f"Error in write_neo4j_cypher: {e}")
        logger.error(f"Query was: {request.query}")
        raise HTTPException(status_code=500, detail=f"Neo4j write error: {str(e)}")

@app.post("/get_neo4j_schema")
async def get_neo4j_schema():
    get_schema_query = "CALL apoc.meta.schema();"
    try:
        logger.info("Fetching Neo4j schema")
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_read(_read, get_schema_query, {})
        
        parsed_result = json.loads(result)
        if parsed_result and len(parsed_result) > 0:
            schema = parsed_result[0].get('value', {})
            logger.info(f"Schema fetched successfully with {len(schema)} elements")
            return schema
        else:
            logger.warning("Schema query returned empty result")
            return {"error": "Schema query returned empty result"}
    except Exception as e:
        logger.error(f"Error in get_neo4j_schema: {e}")
        # Fallback to basic schema queries if APOC is not available
        try:
            logger.info("Trying fallback schema queries")
            async with driver.session(database=NEO4J_DATABASE) as session:
                # Get labels
                labels_result = await session.execute_read(_read, "CALL db.labels() YIELD label RETURN collect(label) as labels", {})
                labels = json.loads(labels_result)[0].get('labels', [])
                
                # Get relationship types
                rels_result = await session.execute_read(_read, "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types", {})
                rel_types = json.loads(rels_result)[0].get('types', [])
                
                # Get property keys
                props_result = await session.execute_read(_read, "CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) as keys", {})
                prop_keys = json.loads(props_result)[0].get('keys', [])
                
                return {
                    "labels": labels,
                    "relationship_types": rel_types,
                    "property_keys": prop_keys,
                    "note": "Basic schema info (APOC not available)"
                }
        except Exception as fallback_e:
            logger.error(f"Fallback schema queries also failed: {fallback_e}")
            raise HTTPException(status_code=500, detail=f"Schema fetch error: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_read(_read, "RETURN 1 as test", {})
        return {"status": "healthy", "neo4j": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "neo4j": "disconnected", "error": str(e)}

async def _read(tx: AsyncTransaction, query: str, params: dict):
    try:
        res = await tx.run(query, params)
        records = await res.to_eager_result()
        return json.dumps([r.data() for r in records.records], default=str)
    except Exception as e:
        logger.error(f"Transaction read error: {e}")
        raise

async def _write(tx: AsyncTransaction, query: str, params: dict):
    try:
        return await tx.run(query, params)
    except Exception as e:
        logger.error(f"Transaction write error: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    logger.info("MCP Neo4j server starting up...")
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_read(_read, "RETURN 1 as test", {})
        logger.info("Neo4j connection established successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("MCP Neo4j server shutting down...")
    await driver.close()
    logger.info("Neo4j driver closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcpserver:app", host="0.0.0.0", port=8000, reload=True)
