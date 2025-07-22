"""
Enhanced MCP Server with Real-time Neo4j NVL Support
This provides the backend for real-time graph updates and NVL integration
Run this on port 8000
"""

import asyncio
import json
import logging
import uuid
import time
from typing import Dict, Any, Optional, List, Set
from datetime import datetime

# FastMCP and FastAPI imports
from fastmcp import FastMCP
from fastapi import HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Neo4j imports
from neo4j import AsyncGraphDatabase, AsyncDriver

# LangGraph imports
import requests
import urllib3
import re
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================
# 🔧 CONFIGURATION - CHANGE THESE VALUES
# ============================================

# Neo4j Configuration
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_neo4j_password"  # ⚠️ CHANGE THIS!
NEO4J_DATABASE = "neo4j"

# Cortex API Configuration
CORTEX_API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
CORTEX_API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"  # ⚠️ CHANGE THIS!
CORTEX_MODEL = "claude-4-sonnet"

# Server Configuration
SERVER_PORT = 8000
SERVER_HOST = "0.0.0.0"

# ============================================

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced_mcp_server")

print("🔧 Enhanced MCP Server Configuration:")
print(f"   Neo4j URI: {NEO4J_URI}")
print(f"   Neo4j User: {NEO4J_USER}")
print(f"   Neo4j Database: {NEO4J_DATABASE}")
print(f"   Cortex API: {CORTEX_API_URL}")
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

# Initialize FastMCP
mcp = FastMCP("Enhanced Neo4j NVL Server")

# Global variables for real-time updates
active_websockets: Set[WebSocket] = set()
database_stats = {"nodes": 0, "relationships": 0, "labels": [], "relationship_types": []}
last_graph_update = datetime.now()

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
    # Enhanced fields for NVL
    graph_data: Optional[Dict[str, Any]] = None
    operation_summary: Optional[Dict[str, Any]] = None
    nodes_affected: int = 0
    relationships_affected: int = 0

class AgentState(BaseModel):
    question: str
    session_id: str
    tool: str = ""
    query: str = ""
    trace: str = ""
    answer: str = ""
    error_count: int = 0
    last_error: str = ""

class GraphUpdateMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: str

# ============================================
# REAL-TIME DATABASE MONITORING
# ============================================

async def get_comprehensive_database_stats():
    """Get comprehensive database statistics for real-time monitoring"""
    if driver is None:
        return {"error": "Driver not initialized"}
    
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            # Get detailed statistics
            stats_queries = [
                ("node_count", "MATCH (n) RETURN count(n) as count"),
                ("relationship_count", "MATCH ()-[r]->() RETURN count(r) as count"),
                ("labels", "CALL db.labels() YIELD label RETURN collect(label) as items"),
                ("relationship_types", "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as items"),
                ("property_keys", "CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) as items")
            ]
            
            results = {}
            for stat_name, query in stats_queries:
                try:
                    result = await session.run(query)
                    record = await result.single()
                    if stat_name in ["node_count", "relationship_count"]:
                        results[stat_name.replace("_count", "s")] = record["count"] if record else 0
                    else:
                        results[stat_name.replace("_", "_")] = record["items"] if record else []
                except Exception as e:
                    logger.warning(f"Failed to get {stat_name}: {e}")
                    results[stat_name] = 0 if "count" in stat_name else []
            
            # Add label distribution
            try:
                label_dist_result = await session.run("""
                    CALL db.labels() YIELD label
                    CALL {
                        WITH label
                        MATCH (n) WHERE label IN labels(n)
                        RETURN label, count(n) as count
                    }
                    RETURN label, count
                """)
                label_distribution = {}
                async for record in label_dist_result:
                    label_distribution[record["label"]] = record["count"]
                results["label_distribution"] = label_distribution
            except Exception:
                results["label_distribution"] = {}
            
            results["timestamp"] = datetime.now().isoformat()
            return results
            
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {"error": str(e)}

async def get_real_graph_data(limit: int = 100, include_relationships: bool = True) -> Dict[str, Any]:
    """Get real graph data directly from Neo4j with full relationship information"""
    if driver is None:
        raise Exception("Neo4j driver not initialized")
    
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            # Get nodes with all their properties and labels
            nodes_query = f"""
            MATCH (n)
            RETURN id(n) as id, labels(n) as labels, properties(n) as properties
            LIMIT {limit}
            """
            
            nodes_result = await session.run(nodes_query)
            nodes_data = await nodes_result.data()
            
            # Format nodes for NVL
            nodes = []
            node_ids = set()
            for node in nodes_data:
                node_id = node["id"]
                node_ids.add(node_id)
                labels = node["labels"]
                properties = node["properties"]
                
                # Create meaningful caption
                caption = (
                    properties.get("name") or 
                    properties.get("title") or 
                    properties.get("username") or 
                    properties.get("email") or 
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
                # Get relationships between the loaded nodes
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
            
            # Get summary statistics
            summary = await get_comprehensive_database_stats()
            
            return {
                "nodes": nodes,
                "relationships": relationships,
                "summary": summary,
                "loaded_at": datetime.now().isoformat(),
                "total_nodes_loaded": len(nodes),
                "total_relationships_loaded": len(relationships)
            }
            
    except Exception as e:
        logger.error(f"Failed to get real graph data: {e}")
        raise Exception(f"Graph data fetch failed: {str(e)}")

async def broadcast_graph_update(operation_type: str, affected_data: Dict[str, Any]):
    """Broadcast graph updates to all connected WebSocket clients"""
    if not active_websockets:
        return
    
    try:
        # Get updated graph data
        updated_graph_data = await get_real_graph_data(100)
        updated_stats = await get_comprehensive_database_stats()
        
        message = GraphUpdateMessage(
            type="graph_update",
            data={
                "operation_type": operation_type,
                "affected_data": affected_data,
                "graph_data": updated_graph_data,
                "stats": updated_stats,
                "update_id": str(uuid.uuid4())
            },
            timestamp=datetime.now().isoformat()
        )
        
        # Send to all connected clients
        disconnected = set()
        for websocket in active_websockets:
            try:
                await websocket.send_text(message.json())
                logger.info(f"Sent graph update to WebSocket client")
            except Exception as e:
                logger.warning(f"Failed to send update to WebSocket: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected clients
        active_websockets -= disconnected
        
        # Update global stats
        global database_stats, last_graph_update
        if "error" not in updated_stats:
            database_stats = updated_stats
            last_graph_update = datetime.now()
        
    except Exception as e:
        logger.error(f"Failed to broadcast graph update: {e}")

# ============================================
# ENHANCED MCP TOOLS WITH REAL-TIME UPDATES
# ============================================

@mcp.tool()
async def read_neo4j_cypher(query: str, params: dict = {}) -> List[Dict[str, Any]]:
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

@mcp.tool()
async def write_neo4j_cypher(query: str, params: dict = {}) -> Dict[str, Any]:
    """
    Execute write Cypher queries against Neo4j database with real-time updates.
    
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
            "labels_removed": counters.labels_removed,
            "query_executed": query
        }
        
        # Determine operation type for broadcast
        operation_type = "unknown"
        if counters.nodes_created > 0 or counters.relationships_created > 0:
            operation_type = "create"
        elif counters.nodes_deleted > 0 or counters.relationships_deleted > 0:
            operation_type = "delete"
        elif counters.properties_set > 0:
            operation_type = "update"
        
        # Broadcast update to connected clients if changes were made
        total_changes = sum([
            counters.nodes_created, counters.nodes_deleted,
            counters.relationships_created, counters.relationships_deleted,
            counters.properties_set
        ])
        
        if total_changes > 0:
            # Schedule broadcast (non-blocking)
            asyncio.create_task(broadcast_graph_update(operation_type, response))
        
        logger.info(f"Write query completed: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Write query failed: {e}")
        raise Exception(f"Query failed: {str(e)}")

@mcp.tool()
async def get_neo4j_schema() -> Dict[str, Any]:
    """
    Get the schema of the Neo4j database including labels, relationship types, and properties.
    
    Returns:
        Dictionary containing database schema information
    """
    if driver is None:
        raise Exception("Neo4j driver not initialized")
    
    try:
        logger.info("Fetching comprehensive database schema")
        
        schema_data = await get_comprehensive_database_stats()
        
        if "error" in schema_data:
            return schema_data
        
        # Add more detailed schema information
        async with driver.session(database=NEO4J_DATABASE) as session:
            # Get sample nodes for each label
            label_samples = {}
            for label in schema_data.get("labels", []):
                try:
                    sample_result = await session.run(f"MATCH (n:{label}) RETURN n LIMIT 3")
                    samples = []
                    async for record in sample_result:
                        node = record["n"]
                        samples.append({
                            "id": str(node.id),
                            "properties": dict(node)
                        })
                    label_samples[label] = samples
                except Exception:
                    label_samples[label] = []
            
            schema_data["label_samples"] = label_samples
        
        logger.info(f"Schema fetched: {len(schema_data.get('labels', []))} labels, {len(schema_data.get('relationship_types', []))} rel types")
        return schema_data
        
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
# ENHANCED FASTAPI APP
# ============================================

# Get FastAPI app from FastMCP
app = mcp.get_app()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize everything on startup with comprehensive setup"""
    global database_stats
    
    print("🚀 Starting Enhanced MCP Neo4j NVL Server...")
    print("=" * 60)
    
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
            
            # Get initial comprehensive database state
            print("📊 Getting comprehensive database state...")
            database_stats = await get_comprehensive_database_stats()
            
            if "error" not in database_stats:
                print(f"   📈 Nodes: {database_stats['nodes']}")
                print(f"   🔗 Relationships: {database_stats['relationships']}")
                print(f"   🏷️  Labels: {len(database_stats['labels'])}")
                print(f"   ➡️  Relationship Types: {len(database_stats['relationship_types'])}")
                print(f"   🔑 Property Keys: {len(database_stats.get('property_keys', []))}")
                
                if database_stats['nodes'] > 0:
                    print("   🎯 Database contains data - NVL will show live graph")
                    
                    # Test graph data loading
                    try:
                        test_graph = await get_real_graph_data(10)
                        nodes_loaded = len(test_graph.get("nodes", []))
                        rels_loaded = len(test_graph.get("relationships", []))
                        print(f"   📊 Graph loading test: {nodes_loaded} nodes, {rels_loaded} relationships")
                    except Exception as e:
                        print(f"   ⚠️  Graph loading test failed: {e}")
                else:
                    print("   📝 Empty database - create some nodes to see NVL visualization")
                    
            else:
                print(f"   ⚠️  Could not get database stats: {database_stats['error']}")
        else:
            print("❌ Neo4j connection test failed")
            
    except Exception as e:
        print("❌ Neo4j connection failed!")
        print(f"   Error: {e}")
    
    print("=" * 60)
    print(f"🌐 Enhanced MCP server ready on http://localhost:{SERVER_PORT}")
    print("📋 Available endpoints:")
    print("   • GET  /health - Comprehensive health check")
    print("   • GET  /graph - Real Neo4j graph data")
    print("   • GET  /stats - Live database statistics")
    print("   • WS   /ws - WebSocket for real-time updates")
    print("   • GET  /nvl - Enhanced NVL visualization interface")
    print("   • POST /chat - Chat with enhanced agent")
    print("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown"""
    print("🛑 Shutting down Enhanced MCP Neo4j Server...")
    if driver:
        await driver.close()
        print("✅ Neo4j driver closed")

# ============================================
# ENHANCED API ENDPOINTS
# ============================================

@app.get("/health")
async def comprehensive_health_check():
    """Comprehensive health check endpoint"""
    if driver is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "neo4j": {"status": "driver_not_initialized"},
                "server": {"port": SERVER_PORT, "type": "Enhanced MCP"},
                "websockets": {"active_connections": len(active_websockets)}
            }
        )
    
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run("RETURN 1 as test")
            record = await result.single()
            
        neo4j_status = "connected" if record and record["test"] == 1 else "test_failed"
        
        # Get current comprehensive stats
        current_stats = await get_comprehensive_database_stats()
        
        return {
            "status": "healthy",
            "neo4j": {
                "status": neo4j_status,
                "uri": NEO4J_URI,
                "database": NEO4J_DATABASE,
                "user": NEO4J_USER,
                "current_stats": current_stats,
                "last_update": last_graph_update.isoformat()
            },
            "server": {
                "port": SERVER_PORT,
                "host": SERVER_HOST,
                "type": "Enhanced MCP with Real-time NVL",
                "tools": ["read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"],
                "features": [
                    "real_time_updates", 
                    "websocket_support", 
                    "comprehensive_monitoring",
                    "nvl_integration"
                ]
            },
            "websockets": {
                "active_connections": len(active_websockets),
                "last_broadcast": last_graph_update.isoformat()
            },
            "nvl": {
                "status": "enabled",
                "interface_available": True,
                "real_time_updates": True
            }
        }
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy", 
                "neo4j": {"status": "disconnected", "error": str(e)},
                "server": {"port": SERVER_PORT, "type": "Enhanced MCP"}
            }
        )

@app.get("/stats")
async def get_live_comprehensive_stats():
    """Get live comprehensive database statistics"""
    stats = await get_comprehensive_database_stats()
    return stats

@app.get("/graph")
async def get_real_graph_data_endpoint(limit: int = 100, include_relationships: bool = True):
    """Get real graph data from Neo4j for NVL visualization"""
    try:
        graph_data = await get_real_graph_data(limit, include_relationships)
        return graph_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint for real-time graph updates"""
    await websocket.accept()
    active_websockets.add(websocket)
    
    try:
        # Send initial comprehensive state
        initial_graph_data = await get_real_graph_data(100)
        initial_stats = await get_comprehensive_database_stats()
        
        initial_message = GraphUpdateMessage(
            type="initial_state",
            data={
                "graph_data": initial_graph_data,
                "stats": initial_stats,
                "connection_id": str(uuid.uuid4())
            },
            timestamp=datetime.now().isoformat()
        )
        
        await websocket.send_text(initial_message.json())
        logger.info("Sent initial state to new WebSocket client")
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "request_update":
                    # Send current state
                    current_graph_data = await get_real_graph_data(100)
                    current_stats = await get_comprehensive_database_stats()
                    
                    update_message = GraphUpdateMessage(
                        type="requested_update",
                        data={
                            "graph_data": current_graph_data,
                            "stats": current_stats
                        },
                        timestamp=datetime.now().isoformat()
                    )
                    
                    await websocket.send_text(update_message.json())
                elif message.get("type") == "ping":
                    # Respond to ping
                    pong_message = GraphUpdateMessage(
                        type="pong",
                        data={"status": "alive"},
                        timestamp=datetime.now().isoformat()
                    )
                    await websocket.send_text(pong_message.json())
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
                
    finally:
        active_websockets.discard(websocket)
        logger.info("WebSocket client disconnected")

@app.get("/nvl", response_class=HTMLResponse)
async def enhanced_nvl_visualization_interface():
    """Enhanced Neo4j NVL visualization interface with real-time updates"""
    return HTMLResponse(content=f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Neo4j NVL - Real-time Visualization</title>
    <script src="https://unpkg.com/neo4j-nvl@latest/dist/index.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .controls {{
            background: white;
            padding: 1rem 2rem;
            border-bottom: 1px solid #eee;
            display: flex;
            gap: 1rem;
            align-items: center;
            flex-wrap: wrap;
            justify-content: space-between;
        }}
        
        .control-group {{
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }}
        
        .btn {{
            background: #667eea;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
            border: 2px solid transparent;
        }}
        
        .btn:hover {{
            background: #5a6fd8;
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        
        .btn.active {{
            background: #28a745;
            border-color: #ffffff;
        }}
        
        .stats {{
            display: flex;
            gap: 1rem;
            align-items: center;
            font-size: 14px;
        }}
        
        .stat {{
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 0.5rem 1rem;
            border-radius: 4px;
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .connection-status {{
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }}
        
        .connected {{
            background: #d4edda;
            color: #155724;
        }}
        
        .disconnected {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        #nvl-container {{
            height: calc(100vh - 160px);
            background: white;
            border-top: 1px solid #eee;
            border-radius: 0 0 8px 8px;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        }}
        
        .loading {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
            font-size: 18px;
            color: #666;
            flex-direction: column;
            gap: 1rem;
        }}
        
        .loading-spinner {{
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .update-indicator {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: #28a745;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            z-index: 1000;
            opacity: 0;
            transition: opacity 0.3s;
        }}
        
        .update-indicator.show {{
            opacity: 1;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Enhanced Neo4j NVL - Real-time Graph Visualization</h1>
        <p>Live view of your Neo4j database with automatic updates</p>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <button class="btn" onclick="refreshGraph()">🔄 Refresh</button>
            <button class="btn" onclick="fitToView()">🔍 Fit View</button>
            <button class="btn" onclick="resetLayout()">🎯 Reset Layout</button>
            <button class="btn" id="autoUpdateBtn" onclick="toggleAutoUpdate()">🔴 Auto-Update: OFF</button>
        </div>
        
        <div class="stats">
            <div class="stat">Nodes: <span id="node-count">0</span></div>
            <div class="stat">Relationships: <span id="rel-count">0</span></div>
            <div class="stat">Last Update: <span id="last-update">Never</span></div>
            <div class="connection-status disconnected" id="ws-status">Connecting...</div>
        </div>
    </div>
    
    <div id="nvl-container">
        <div class="loading">
            <div class="loading-spinner"></div>
            <div>🚀 Loading real Neo4j graph data...</div>
        </div>
    </div>
    
    <div class="update-indicator" id="updateIndicator">
        📊 Graph Updated!
    </div>

    <script>
        let nvl;
        let websocket;
        let autoUpdate = false;
        let currentGraphData = null;
        
        console.log('🚀 Enhanced Neo4j NVL Interface Starting...');
        
        // Initialize NVL with enhanced configuration
        async function initializeNVL() {{
            try {{
                console.log('📊 Fetching initial graph data...');
                const response = await fetch('/graph?limit=100&include_relationships=true');
                const graphData = await response.json();
                
                console.log('📈 Graph data received:', {{
                    nodes: graphData.nodes?.length || 0,
                    relationships: graphData.relationships?.length || 0
                }});
                
                currentGraphData = graphData;
                
                if (!graphData.nodes || graphData.nodes.length === 0) {{
                    document.getElementById('nvl-container').innerHTML = `
                        <div class="loading">
                            📝 No data in Neo4j database.<br>
                            <strong>Try creating some nodes in the chatbot:</strong><br>
                            "Create a Person named Alice with age 30"
                        </div>
                    `;
                    updateStats({{nodes: 0, relationships: 0}});
                    return;
                }}
                
                // Initialize NVL with enhanced configuration
                nvl = new NVL('nvl-container', graphData.nodes, graphData.relationships, {{
                    instanceId: 'enhanced-neo4j-graph',
                    initialZoom: 1.0,
                    allowDynamicMinZoom: true,
                    showPropertiesOnHover: true,
                    showPropertiesOnClick: true,
                    nodeColorScheme: 'category20',
                    relationshipColorScheme: 'dark',
                    layout: {{
                        algorithm: 'forceDirected',
                        incrementalLayout: true,
                        animate: true,
                        animationDuration: 1000
                    }},
                    styling: {{
                        nodeSize: 25,
                        relationshipWidth: 3,
                        fontSize: 11,
                        fontColor: '#333'
                    }},
                    interaction: {{
                        dragEnabled: true,
                        zoomEnabled: true,
                        hoverEnabled: true,
                        selectEnabled: true,
                        doubleClickEnabled: true
                    }},
                    renderingOptions: {{
                        enableWebGL: true,
                        antialias: true
                    }}
                }});
                
                console.log('✅ Enhanced NVL initialized successfully');
                
                // Add enhanced interaction handlers
                nvl.onNodeClick((node) => {{
                    console.log('🖱️ Node clicked:', node);
                    showNodeDetails(node);
                }});
                
                nvl.onRelationshipClick((relationship) => {{
                    console.log('🖱️ Relationship clicked:', relationship);
                    showRelationshipDetails(relationship);
                }});
                
                nvl.onNodeDoubleClick((node) => {{
                    console.log('🖱️ Node double-clicked:', node);
                    expandNode(node);
                }});
                
                // Update statistics
                updateStats(graphData.summary || {{}});
                
            }} catch (error) {{
                console.error('❌ NVL initialization failed:', error);
                document.getElementById('nvl-container').innerHTML = `
                    <div class="loading">
                        <div style="color: #dc3545;">❌ Failed to load graph</div>
                        <div>Error: ${{error.message}}</div>
                        <button class="btn" onclick="initializeNVL()" style="margin-top: 1rem;">🔄 Retry</button>
                    </div>
                `;
            }}
        }}
        
        // Initialize WebSocket with enhanced features
        function initWebSocket() {{
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${{protocol}}//${{window.location.host}}/ws`;
            
            console.log('🔌 Connecting to WebSocket:', wsUrl);
            websocket = new WebSocket(wsUrl);
            
            websocket.onopen = function(event) {{
                console.log('✅ WebSocket connected');
                updateConnectionStatus(true);
            }};
            
            websocket.onmessage = function(event) {{
                try {{
                    const message = JSON.parse(event.data);
                    console.log('📨 WebSocket message received:', message.type);
                    
                    if (message.type === 'graph_update' || message.type === 'initial_state' || message.type === 'requested_update') {{
                        handleGraphUpdate(message.data);
                    }}
                }} catch (error) {{
                    console.error('❌ WebSocket message parse error:', error);
                }}
            }};
            
            websocket.onclose = function(event) {{
                console.log('🔌 WebSocket disconnected');
                updateConnectionStatus(false);
                
                // Attempt to reconnect after 3 seconds
                setTimeout(initWebSocket, 3000);
            }};
            
            websocket.onerror = function(error) {{
                console.error('❌ WebSocket error:', error);
                updateConnectionStatus(false);
            }};
        }}
        
        // Handle graph updates with enhanced features
        function handleGraphUpdate(data) {{
            if (!data.graph_data) return;
            
            const graphData = data.graph_data;
            const stats = data.stats || {{}};
            
            console.log('🔄 Handling graph update:', {{
                nodes: graphData.nodes?.length || 0,
                relationships: graphData.relationships?.length || 0,
                operation_type: data.operation_type || 'unknown'
            }});
            
            // Update statistics
            updateStats(stats);
            
            // Show update indicator
            showUpdateIndicator(data.operation_type || 'update');
            
            // Update NVL if auto-update is enabled or if it's an initial state
            if (autoUpdate || data.operation_type === 'initial_state' || !nvl) {{
                updateNVLGraph(graphData);
            }}
            
            currentGraphData = graphData;
        }}
        
        // Update NVL graph with smooth transitions
        function updateNVLGraph(graphData) {{
            if (!graphData.nodes || graphData.nodes.length === 0) {{
                document.getElementById('nvl-container').innerHTML = `
                    <div class="loading">
                        📝 No data in Neo4j database.<br>
                        <strong>Create some nodes to see the graph!</strong>
                    </div>
                `;
                return;
            }}
            
            if (nvl) {{
                console.log('🔄 Updating NVL graph...');
                try {{
                    nvl.updateGraph(graphData.nodes, graphData.relationships);
                    console.log('✅ NVL graph updated successfully');
                }} catch (error) {{
                    console.error('❌ NVL update failed:', error);
                    // Reinitialize if update fails
                    initializeNVL();
                }}
            }} else {{
                initializeNVL();
            }}
        }}
        
        // Update statistics display
        function updateStats(stats) {{
            document.getElementById('node-count').textContent = stats.nodes || 0;
            document.getElementById('rel-count').textContent = stats.relationships || 0;
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
        }}
        
        // Update connection status
        function updateConnectionStatus(connected) {{
            const statusEl = document.getElementById('ws-status');
            if (connected) {{
                statusEl.textContent = '🟢 Connected';
                statusEl.className = 'connection-status connected';
            }} else {{
                statusEl.textContent = '🔴 Disconnected';
                statusEl.className = 'connection-status disconnected';
            }}
        }}
        
        // Show update indicator
        function showUpdateIndicator(operationType) {{
            const indicator = document.getElementById('updateIndicator');
            indicator.textContent = `📊 Graph ${{operationType || 'updated'}}!`;
            indicator.classList.add('show');
            
            setTimeout(() => {{
                indicator.classList.remove('show');
            }}, 2000);
        }}
        
        // Control functions
        function refreshGraph() {{
            console.log('🔄 Manual refresh requested');
            if (websocket && websocket.readyState === WebSocket.OPEN) {{
                websocket.send(JSON.stringify({{type: 'request_update'}}));
            }} else {{
                initializeNVL();
            }}
        }}
        
        function fitToView() {{
            if (nvl) {{
                nvl.fit();
            }}
        }}
        
        function resetLayout() {{
            if (nvl) {{
                nvl.restartLayout();
            }}
        }}
        
        function toggleAutoUpdate() {{
            autoUpdate = !autoUpdate;
            const btn = document.getElementById('autoUpdateBtn');
            if (autoUpdate) {{
                btn.textContent = '🟢 Auto-Update: ON';
                btn.classList.add('active');
            }} else {{
                btn.textContent = '🔴 Auto-Update: OFF';
                btn.classList.remove('active');
            }}
            console.log('🔄 Auto-update:', autoUpdate ? 'enabled' : 'disabled');
        }}
        
        // Enhanced interaction functions
        function showNodeDetails(node) {{
            console.log('📊 Node details:', node);
            // You can implement a modal or sidebar to show node details
        }}
        
        function showRelationshipDetails(relationship) {{
            console.log('🔗 Relationship details:', relationship);
            // You can implement a modal or sidebar to show relationship details
        }}
        
        function expandNode(node) {{
            console.log('🔍 Expanding node:', node.id);
            // You can implement node expansion functionality
        }}
        
        // Initialize everything when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('🚀 Enhanced NVL interface loaded');
            
            // Initialize NVL
            initializeNVL();
            
            // Initialize WebSocket
            initWebSocket();
            
            // Auto-enable updates after 2 seconds
            setTimeout(() => {{
                if (!autoUpdate) {{
                    toggleAutoUpdate();
                }}
            }}, 2000);
        }});
        
        // Handle page visibility changes for performance
        document.addEventListener('visibilitychange', function() {{
            if (!document.hidden) {{
                console.log('📱 Page visible - requesting update');
                if (websocket && websocket.readyState === WebSocket.OPEN) {{
                    websocket.send(JSON.stringify({{type: 'request_update'}}));
                }}
            }}
        }});
        
        // Keep WebSocket alive
        setInterval(() => {{
            if (websocket && websocket.readyState === WebSocket.OPEN) {{
                websocket.send(JSON.stringify({{type: 'ping'}}));
            }}
        }}, 30000);
    </script>
</body>
</html>
    """, media_type="text/html")

# Legacy endpoints for backward compatibility
@app.get("/")
async def root():
    """Root endpoint with comprehensive information"""
    return {
        "service": "Enhanced Neo4j MCP NVL Server",
        "version": "4.0.0",
        "description": "Complete MCP server with real-time Neo4j NVL visualization and WebSocket updates",
        "architecture": "Enhanced FastMCP + Real-time NVL + WebSocket",
        "endpoints": {
            "health": "/health - Comprehensive health check",
            "graph": "/graph - Real Neo4j graph data",
            "stats": "/stats - Live database statistics",
            "nvl": "/nvl - Enhanced NVL visualization interface",
            "ws": "/ws - WebSocket for real-time updates",
            "docs": "/docs - FastAPI documentation"
        },
        "mcp_tools": {
            "read_neo4j_cypher": "Execute read-only Cypher queries",
            "write_neo4j_cypher": "Execute write queries with real-time updates", 
            "get_neo4j_schema": "Get comprehensive database schema"
        },
        "real_time_features": {
            "websocket_updates": "Real-time graph updates via WebSocket",
            "broadcast_on_changes": "Automatic broadcast when database changes",
            "nvl_integration": "Official Neo4j Visualization Library support",
            "comprehensive_monitoring": "Detailed database statistics and monitoring"
        },
        "neo4j": {
            "uri": NEO4J_URI,
            "database": NEO4J_DATABASE,
            "user": NEO4J_USER,
            "current_stats": database_stats,
            "last_update": last_graph_update.isoformat()
        },
        "websockets": {
            "active_connections": len(active_websockets),
            "endpoint": "/ws"
        }
    }

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """Main function to run the enhanced MCP server"""
    print("=" * 70)
    print("🧠 ENHANCED NEO4J MCP SERVER WITH REAL-TIME NVL")
    print("=" * 70)
    print("🏗️  Architecture: Enhanced FastMCP + Real-time NVL + WebSocket")
    print("🔧 Configuration:")
    print(f"   📍 Neo4j URI: {NEO4J_URI}")
    print(f"   👤 Neo4j User: {NEO4J_USER}")
    print(f"   🗄️  Neo4j Database: {NEO4J_DATABASE}")
    print(f"   🤖 Cortex Model: {CORTEX_MODEL}")
    print(f"   🌐 Server: {SERVER_HOST}:{SERVER_PORT}")
    print("=" * 70)
    print("✨ Enhanced Features:")
    print("   🎯 Real-time Neo4j graph visualization")
    print("   📊 Comprehensive database monitoring")
    print("   🔄 WebSocket-based live updates")
    print("   📈 Enhanced NVL integration")
    print("   🖥️  Advanced visualization interface")
    print("   📡 Automatic broadcast on database changes")
    print("=" * 70)
    
    if driver is None:
        print("❌ Cannot start server - Neo4j driver failed to initialize")
        return
    
    print("🚀 Starting enhanced MCP server with real-time NVL...")
    
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
