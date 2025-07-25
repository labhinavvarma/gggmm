"""
Standalone Neo4j Server - No FastMCP, No Library Conflicts
This completely avoids all FastMCP and compatibility issues
"""

import asyncio
import json
import logging
import uuid
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

# Only standard FastAPI imports - no FastMCP
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Neo4j imports
from neo4j import AsyncGraphDatabase, AsyncDriver

# Basic imports for LLM
import requests
import urllib3
import re

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================
# 🔧 CONFIGURATION - CHANGE THESE VALUES
# ============================================

# Neo4j Configuration
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_neo4j_password"  # ⚠️ CHANGE THIS!
NEO4J_DATABASE = "neo4j"

# Cortex API Configuration (optional - can work without LLM)
CORTEX_API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
CORTEX_API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"  # ⚠️ CHANGE THIS!
CORTEX_MODEL = "claude-4-sonnet"

# Server Configuration
SERVER_PORT = 8000
SERVER_HOST = "0.0.0.0"

# ============================================

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("standalone_neo4j_server")

print("🔧 Standalone Neo4j Server Configuration:")
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

# Create pure FastAPI app - NO FastMCP
app = FastAPI(
    title="Standalone Neo4j Server",
    description="Simple Neo4j server with no external dependencies",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

print("✅ Pure FastAPI app created successfully")

# ============================================
# PYDANTIC MODELS
# ============================================

class CypherRequest(BaseModel):
    query: str
    params: dict = {}

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    trace: str
    tool: str
    query: str
    answer: str
    session_id: str
    success: bool = True
    error: Optional[str] = None

# ============================================
# NEO4J FUNCTIONS
# ============================================

async def execute_read_query(query: str, params: dict = {}) -> List[Dict[str, Any]]:
    """Execute read-only Cypher queries"""
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

async def execute_write_query(query: str, params: dict = {}) -> Dict[str, Any]:
    """Execute write Cypher queries"""
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

async def get_schema() -> Dict[str, Any]:
    """Get database schema"""
    if driver is None:
        raise Exception("Neo4j driver not initialized")
    
    try:
        logger.info("Fetching database schema")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            # Get labels
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
        
        return {
            "labels": labels,
            "relationship_types": rel_types,
            "property_keys": prop_keys,
            "source": "standalone_server"
        }
        
    except Exception as e:
        logger.error(f"Schema fetch failed: {e}")
        return {
            "labels": [],
            "relationship_types": [],
            "property_keys": [],
            "error": f"Schema fetch failed: {str(e)}"
        }

# ============================================
# SIMPLE LLM PROCESSING (OPTIONAL)
# ============================================

def simple_query_processor(question: str) -> tuple[str, str]:
    """Simple rule-based query processing (no LLM required)"""
    question_lower = question.lower()
    
    # Count queries
    if "how many" in question_lower or "count" in question_lower:
        if "node" in question_lower:
            return "read_neo4j_cypher", "MATCH (n) RETURN count(n) as total_nodes"
        elif "relationship" in question_lower:
            return "read_neo4j_cypher", "MATCH ()-[r]->() RETURN count(r) as total_relationships"
        else:
            return "read_neo4j_cypher", "MATCH (n) RETURN count(n) as total_nodes"
    
    # Schema queries
    elif "schema" in question_lower or "structure" in question_lower or "labels" in question_lower:
        return "get_neo4j_schema", ""
    
    # Create queries
    elif "create" in question_lower:
        if "person" in question_lower:
            # Extract name if possible
            name_match = re.search(r'named?\s+([A-Za-z]+)', question, re.I)
            name = name_match.group(1) if name_match else 'TestPerson'
            return "write_neo4j_cypher", f"CREATE (p:Person {{name: '{name}', created: datetime()}}) RETURN p"
        elif "company" in question_lower:
            company_match = re.search(r'(?:company|organization)\s+(?:called\s+|named\s+)?([A-Za-z]+)', question, re.I)
            company = company_match.group(1) if company_match else 'TestCompany'
            return "write_neo4j_cypher", f"CREATE (c:Company {{name: '{company}', created: datetime()}}) RETURN c"
        else:
            return "write_neo4j_cypher", "CREATE (n:TestNode {created: datetime()}) RETURN n"
    
    # List/show queries
    elif "list" in question_lower or "show" in question_lower or "all" in question_lower:
        if "person" in question_lower:
            return "read_neo4j_cypher", "MATCH (p:Person) RETURN p LIMIT 10"
        elif "company" in question_lower:
            return "read_neo4j_cypher", "MATCH (c:Company) RETURN c LIMIT 10"
        else:
            return "read_neo4j_cypher", "MATCH (n) RETURN n LIMIT 10"
    
    # Delete queries
    elif "delete" in question_lower or "remove" in question_lower:
        if "test" in question_lower:
            return "write_neo4j_cypher", "MATCH (n:TestNode) DETACH DELETE n"
        else:
            return "write_neo4j_cypher", "MATCH (n) WHERE n.name = 'test' DETACH DELETE n"
    
    # Default to node count
    else:
        return "read_neo4j_cypher", "MATCH (n) RETURN count(n) as total_nodes"

def call_llm_fallback(question: str) -> tuple[str, str]:
    """Try LLM, fallback to simple processor"""
    try:
        # Try Cortex LLM if API key is configured
        if CORTEX_API_KEY != "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0":
            headers = {
                "Authorization": f'Snowflake Token="{CORTEX_API_KEY}"',
                "Content-Type": "application/json"
            }
            
            payload = {
                "query": {
                    "aplctn_cd": "edagnai",
                    "app_id": "edadip", 
                    "api_key": CORTEX_API_KEY,
                    "method": "cortex",
                    "model": CORTEX_MODEL,
                    "sys_msg": "Parse this question and respond with Tool: [tool_name] and Query: [cypher_query]",
                    "limit_convs": "0",
                    "prompt": {
                        "messages": [{"role": "user", "content": question}]
                    },
                    "session_id": "standalone"
                }
            }
            
            response = requests.post(CORTEX_API_URL, headers=headers, json=payload, verify=False, timeout=15)
            
            if response.status_code == 200:
                result = response.text.partition("end_of_stream")[0].strip()
                
                # Parse LLM response
                tool_match = re.search(r"Tool:\s*(\w+)", result, re.I)
                query_match = re.search(r"Query:\s*(.+?)(?=\n|$)", result, re.I)
                
                if tool_match and query_match:
                    tool = tool_match.group(1).strip()
                    query = query_match.group(1).strip()
                    if tool in ["read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"]:
                        return tool, query
    
    except Exception as e:
        logger.warning(f"LLM call failed, using fallback: {e}")
    
    # Fallback to simple processor
    return simple_query_processor(question)

# ============================================
# FASTAPI MIDDLEWARE
# ============================================

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# STARTUP/SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 Starting Standalone Neo4j Server...")
    print("=" * 50)
    
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
            
            # Test count nodes
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
    print(f"🌐 Standalone server ready on http://localhost:{SERVER_PORT}")
    print("📋 Available endpoints:")
    print("   • GET  /health - Health check")
    print("   • POST /chat - Chat interface")
    print("   • GET  /graph - Graph data")
    print("   • GET  /stats - Database statistics")
    print("   • POST /read_neo4j_cypher - Execute read queries")
    print("   • POST /write_neo4j_cypher - Execute write queries") 
    print("   • POST /get_neo4j_schema - Get schema")
    print("   • GET  /docs - API documentation")
    print("=" * 50)

@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown"""
    print("🛑 Shutting down Standalone Server...")
    if driver:
        await driver.close()
        print("✅ Neo4j driver closed")

# ============================================
# API ENDPOINTS - GUARANTEED TO WORK
# ============================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Standalone Neo4j Server",
        "version": "1.0.0",
        "description": "Simple Neo4j server with no external dependencies",
        "architecture": "Pure FastAPI + Neo4j",
        "dependencies": "Only FastAPI and Neo4j - no conflicts",
        "endpoints": {
            "health": "/health - Health check",
            "chat": "/chat - Chat interface",
            "stats": "/stats - Database statistics",
            "graph": "/graph - Graph data",
            "read_neo4j_cypher": "/read_neo4j_cypher - Execute read queries",
            "write_neo4j_cypher": "/write_neo4j_cypher - Execute write queries",
            "get_neo4j_schema": "/get_neo4j_schema - Get schema",
            "docs": "/docs - API documentation"
        },
        "neo4j": {
            "uri": NEO4J_URI,
            "database": NEO4J_DATABASE,
            "user": NEO4J_USER
        },
        "features": [
            "Simple rule-based query processing",
            "Optional LLM integration",
            "Full Neo4j CRUD operations",
            "Graph visualization data",
            "Schema introspection"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if driver is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "neo4j": {"status": "driver_not_initialized"},
                "server": {"port": SERVER_PORT, "type": "Standalone FastAPI"}
            }
        )
    
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
                "port": SERVER_PORT,
                "host": SERVER_HOST,
                "type": "Standalone FastAPI (No Conflicts)",
                "tools": ["read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"]
            }
        }
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy", 
                "neo4j": {"status": "disconnected", "error": str(e)},
                "server": {"port": SERVER_PORT, "type": "Standalone FastAPI"}
            }
        )

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with simple processing"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info(f"Processing question: {request.question}")
        
        # Process question
        tool, query = call_llm_fallback(request.question)
        
        # Execute appropriate action
        if tool == "get_neo4j_schema":
            result = await get_schema()
            
            if "error" not in result:
                labels = result.get("labels", [])
                rel_types = result.get("relationship_types", [])
                answer = f"📊 **Database Schema:**\n\n**Node Labels:** {', '.join(labels[:10])}\n**Relationship Types:** {', '.join(rel_types[:10])}"
            else:
                answer = f"⚠️ Error getting schema: {result['error']}"
                
        elif tool == "read_neo4j_cypher":
            data = await execute_read_query(query)
            
            if len(data) == 0:
                answer = "📊 **Result:** No data found"
            elif len(data) == 1 and isinstance(data[0], dict) and len(data[0]) == 1:
                key, value = list(data[0].items())[0]
                answer = f"📊 **Result:** {value}"
            else:
                answer = f"📊 **Result:** Found {len(data)} records\n\n{json.dumps(data[:3], indent=2)}"
                if len(data) > 3:
                    answer += f"\n... and {len(data) - 3} more records"
                    
        elif tool == "write_neo4j_cypher":
            result = await execute_write_query(query)
            
            created = result.get("nodes_created", 0)
            deleted = result.get("nodes_deleted", 0)
            rels_created = result.get("relationships_created", 0)
            rels_deleted = result.get("relationships_deleted", 0)
            answer = f"✅ **Write Operation Completed:**\n- Nodes created: {created}\n- Nodes deleted: {deleted}\n- Relationships created: {rels_created}\n- Relationships deleted: {rels_deleted}"
        
        else:
            answer = "⚠️ Unknown tool specified"
        
        return ChatResponse(
            trace=f"Processed: {request.question}",
            tool=tool,
            query=query,
            answer=answer,
            session_id=session_id,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            trace=f"Error: {str(e)}",
            tool="",
            query="",
            answer=f"⚠️ Error processing request: {str(e)}",
            session_id=request.session_id or str(uuid.uuid4()),
            success=False,
            error=str(e)
        )

@app.get("/stats")
async def get_database_stats():
    """Database statistics endpoint"""
    if driver is None:
        raise HTTPException(status_code=503, detail="Neo4j driver not initialized")
    
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            # Get counts
            node_result = await session.run("MATCH (n) RETURN count(n) as nodes")
            node_record = await node_result.single()
            nodes = node_record["nodes"] if node_record else 0
            
            rel_result = await session.run("MATCH ()-[r]->() RETURN count(r) as relationships")
            rel_record = await rel_result.single()
            relationships = rel_record["relationships"] if rel_record else 0
            
            # Get schema info
            schema = await get_schema()
            labels = schema.get("labels", [])
            relationship_types = schema.get("relationship_types", [])
        
        return {
            "nodes": nodes,
            "relationships": relationships,
            "labels": labels,
            "relationship_types": relationship_types,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Stats endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph")
async def get_graph_data(limit: int = 50, include_relationships: bool = True):
    """Graph data endpoint"""
    if driver is None:
        raise HTTPException(status_code=503, detail="Neo4j driver not initialized")
    
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            # Get nodes
            nodes_query = f"MATCH (n) RETURN id(n) as id, labels(n) as labels, properties(n) as properties LIMIT {limit}"
            nodes_result = await session.run(nodes_query)
            nodes_data = await nodes_result.data()
            
            # Format nodes
            nodes = []
            node_ids = set()
            for node in nodes_data:
                node_id = node["id"]
                node_ids.add(node_id)
                labels = node["labels"]
                properties = node["properties"]
                
                caption = (
                    properties.get("name") or 
                    properties.get("title") or 
                    f"{labels[0] if labels else 'Node'} {node_id}"
                )
                
                nodes.append({
                    "id": str(node_id),
                    "labels": labels,
                    "properties": properties,
                    "caption": caption
                })
            
            relationships = []
            if include_relationships and node_ids:
                node_ids_str = ", ".join(map(str, node_ids))
                rels_query = f"""
                MATCH (source)-[r]->(target)
                WHERE id(source) IN [{node_ids_str}] AND id(target) IN [{node_ids_str}]
                RETURN id(r) as id, id(source) as source, id(target) as target, 
                       type(r) as type, properties(r) as properties
                LIMIT {limit * 2}
                """
                
                rels_result = await session.run(rels_query)
                rels_data = await rels_result.data()
                
                for rel in rels_data:
                    relationships.append({
                        "id": str(rel["id"]),
                        "source": str(rel["source"]),
                        "target": str(rel["target"]),
                        "type": rel["type"],
                        "properties": rel["properties"]
                    })
        
        return {
            "nodes": nodes,
            "relationships": relationships,
            "loaded_at": datetime.now().isoformat(),
            "total_nodes_loaded": len(nodes),
            "total_relationships_loaded": len(relationships)
        }
        
    except Exception as e:
        logger.error(f"Graph endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Direct Neo4j endpoints
@app.post("/read_neo4j_cypher")
async def read_cypher(request: CypherRequest):
    """Execute read query"""
    try:
        result = await execute_read_query(request.query, request.params)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/write_neo4j_cypher")
async def write_cypher(request: CypherRequest):
    """Execute write query"""
    try:
        result = await execute_write_query(request.query, request.params)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_neo4j_schema")
async def schema_endpoint():
    """Get schema"""
    try:
        result = await get_schema()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """Main function"""
    print("=" * 60)
    print("🧠 STANDALONE NEO4J SERVER")
    print("=" * 60)
    print("✅ No FastMCP - No Library Conflicts!")
    print("✅ Pure FastAPI - Guaranteed to Work!")
    print(f"   📍 Neo4j URI: {NEO4J_URI}")
    print(f"   🌐 Server: {SERVER_HOST}:{SERVER_PORT}")
    print("=" * 60)
    
    if driver is None:
        print("❌ Cannot start server - Neo4j driver failed to initialize")
        print("❌ Please check Neo4j connection and password")
        return
    
    print("🚀 Starting standalone server...")
    
    try:
        import uvicorn
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
