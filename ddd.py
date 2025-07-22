"""
Working FastAPI Server with MCP-style Tools and Neo4j NVL Integration
This version uses pure FastAPI without FastMCP dependency issues
Run this on port 8000
"""

import asyncio
import json
import logging
import uuid
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

# FastAPI imports (no FastMCP dependency)
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
logger = logging.getLogger("working_fastapi_server")

print("🔧 Working FastAPI Server Configuration:")
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

# Initialize FastAPI directly (no FastMCP)
app = FastAPI(
    title="Neo4j Enhanced Agent with NVL",
    description="FastAPI server with LangGraph agent and Neo4j NVL visualization",
    version="3.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for live updates
active_websockets = set()
database_stats = {"nodes": 0, "relationships": 0, "labels": [], "relationship_types": []}

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
    graph_data: Optional[Dict[str, Any]] = None
    operation_summary: Optional[Dict[str, Any]] = None

class AgentState(BaseModel):
    question: str
    session_id: str
    tool: str = ""
    query: str = ""
    trace: str = ""
    answer: str = ""
    error_count: int = 0
    last_error: str = ""

class GraphData(BaseModel):
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    summary: Dict[str, Any]

# ============================================
# MCP-STYLE TOOL FUNCTIONS (Pure Python Functions)
# ============================================

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

async def write_neo4j_cypher(query: str, params: dict = {}) -> Dict[str, Any]:
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
            "labels_removed": counters.labels_removed,
            "total_changes": (counters.nodes_created + counters.nodes_deleted + 
                            counters.relationships_created + counters.relationships_deleted + 
                            counters.properties_set)
        }
        
        # Broadcast update to connected clients
        asyncio.create_task(broadcast_database_update())
        
        logger.info(f"Write query completed: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Write query failed: {e}")
        raise Exception(f"Query failed: {str(e)}")

async def get_neo4j_schema() -> Dict[str, Any]:
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
        
        schema = {
            "labels": labels,
            "relationship_types": rel_types,
            "property_keys": prop_keys,
            "source": "database_queries",
            "timestamp": datetime.now().isoformat()
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
# DATABASE MONITORING FUNCTIONS
# ============================================

async def get_database_stats():
    """Get comprehensive database statistics"""
    if driver is None:
        return {"error": "Driver not initialized"}
    
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            # Get node count
            node_result = await session.run("MATCH (n) RETURN count(n) as node_count")
            node_record = await node_result.single()
            node_count = node_record["node_count"] if node_record else 0
            
            # Get relationship count
            rel_result = await session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
            rel_record = await rel_result.single()
            rel_count = rel_record["rel_count"] if rel_record else 0
            
            # Get labels
            labels_result = await session.run("CALL db.labels() YIELD label RETURN collect(label) as labels")
            labels_record = await labels_result.single()
            labels = labels_record["labels"] if labels_record else []
            
            # Get relationship types
            types_result = await session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types")
            types_record = await types_result.single()
            rel_types = types_record["types"] if types_record else []
            
            return {
                "nodes": node_count,
                "relationships": rel_count,
                "labels": labels,
                "relationship_types": rel_types,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {"error": str(e)}

async def get_graph_data(limit: int = 100) -> GraphData:
    """Get graph data for visualization"""
    if driver is None:
        raise Exception("Neo4j driver not initialized")
    
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            # Get nodes with their properties
            nodes_result = await session.run(f"""
                MATCH (n)
                RETURN id(n) as id, labels(n) as labels, properties(n) as properties
                LIMIT {limit}
            """)
            nodes_data = await nodes_result.data()
            
            # Get relationships
            rels_result = await session.run(f"""
                MATCH (n)-[r]->(m)
                RETURN id(n) as source, id(m) as target, id(r) as id, 
                       type(r) as type, properties(r) as properties
                LIMIT {limit}
            """)
            rels_data = await rels_result.data()
            
            # Format nodes for NVL
            nodes = []
            for node in nodes_data:
                node_id = str(node["id"])
                labels = node["labels"]
                properties = node["properties"]
                
                # Determine display name
                display_name = properties.get("name") or properties.get("title") or f"Node {node_id}"
                
                nodes.append({
                    "id": node_id,
                    "labels": labels,
                    "properties": properties,
                    "caption": display_name
                })
            
            # Format relationships for NVL
            relationships = []
            for rel in rels_data:
                relationships.append({
                    "id": str(rel["id"]),
                    "source": str(rel["source"]),
                    "target": str(rel["target"]),
                    "type": rel["type"],
                    "properties": rel["properties"]
                })
            
            # Get summary
            summary = await get_database_stats()
            
            return GraphData(
                nodes=nodes,
                relationships=relationships,
                summary=summary
            )
            
    except Exception as e:
        logger.error(f"Failed to get graph data: {e}")
        raise Exception(f"Graph data fetch failed: {str(e)}")

async def broadcast_database_update():
    """Broadcast database updates to all connected WebSockets"""
    if not active_websockets:
        return
    
    try:
        stats = await get_database_stats()
        message = {
            "type": "database_update",
            "data": stats
        }
        
        # Send to all connected clients
        disconnected = set()
        for websocket in active_websockets:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.add(websocket)
        
        # Remove disconnected clients
        active_websockets -= disconnected
        
    except Exception as e:
        logger.error(f"Failed to broadcast update: {e}")

# ============================================
# LANGGRAPH AGENT LOGIC
# ============================================

SYSTEM_PROMPT = """
You are an expert AI assistant that helps users query and manage a Neo4j database using specialized tools.

AVAILABLE TOOLS:
- read_neo4j_cypher: Execute read-only queries (MATCH, RETURN, WHERE, etc.)
- write_neo4j_cypher: Execute write queries (CREATE, MERGE, SET, DELETE, etc.)
- get_neo4j_schema: Get database schema information

GUIDELINES:
- Always explain your reasoning before selecting a tool
- Choose the appropriate tool based on the user's intent
- For schema questions, use get_neo4j_schema
- For data queries, use read_neo4j_cypher
- For data modifications, use write_neo4j_cypher

RESPONSE FORMAT:
Always use this EXACT format:

Tool: [tool_name]
Query: [cypher_query_on_single_line]

EXAMPLES:

User: How many nodes are in the graph?
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN count(n) as total_nodes

User: Create a Person named Alice with age 30
Tool: write_neo4j_cypher
Query: CREATE (p:Person {name: 'Alice', age: 30}) RETURN p

User: Show me the database schema
Tool: get_neo4j_schema
"""

def call_cortex_llm(prompt: str, session_id: str) -> str:
    """Call Cortex LLM with error handling"""
    try:
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
                "sys_msg": SYSTEM_PROMPT,
                "limit_convs": "0",
                "prompt": {
                    "messages": [{"role": "user", "content": prompt}]
                },
                "session_id": session_id
            }
        }
        
        logger.info("Calling Cortex LLM...")
        response = requests.post(CORTEX_API_URL, headers=headers, json=payload, verify=False, timeout=30)
        
        if response.status_code == 200:
            result = response.text.partition("end_of_stream")[0].strip()
            logger.info(f"LLM response received: {len(result)} characters")
            return result
        else:
            logger.error(f"Cortex API error: {response.status_code}")
            return f"Error: Cortex API returned {response.status_code}"
            
    except Exception as e:
        logger.error(f"Cortex LLM call failed: {e}")
        return f"Error: Failed to call LLM - {str(e)}"

def parse_llm_response(llm_output: str) -> tuple[str, str, str]:
    """Parse LLM response to extract tool and query"""
    valid_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    
    tool = None
    query = None
    trace = llm_output.strip()
    
    # Extract tool
    tool_match = re.search(r"Tool:\s*(\w+)", llm_output, re.I)
    if tool_match:
        extracted_tool = tool_match.group(1).strip()
        if extracted_tool in valid_tools:
            tool = extracted_tool
    
    # Extract query
    query_match = re.search(r"Query:\s*(.+?)(?=\n|$)", llm_output, re.I | re.MULTILINE)
    if query_match:
        query = query_match.group(1).strip()
        # Clean query
        query = re.sub(r'```[a-zA-Z]*', '', query)
        query = re.sub(r'```', '', query)
        query = re.sub(r'\s+', ' ', query).strip()
    
    return tool, query, trace

async def execute_tool_function(tool: str, query: str = None) -> Dict[str, Any]:
    """Execute tool function directly"""
    try:
        if tool == "get_neo4j_schema":
            result = await get_neo4j_schema()
            return {"success": True, "data": result, "type": "schema"}
        
        elif tool == "read_neo4j_cypher":
            if not query:
                return {"error": "No query provided for read operation"}
            result = await read_neo4j_cypher(query, {})
            return {"success": True, "data": result, "type": "read", "query": query}
        
        elif tool == "write_neo4j_cypher":
            if not query:
                return {"error": "No query provided for write operation"}
            result = await write_neo4j_cypher(query, {})
            return {"success": True, "data": result, "type": "write", "query": query}
        
        else:
            return {"error": f"Unknown tool: {tool}"}
            
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return {"error": str(e)}

def select_tool_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Select tool and generate query"""
    logger.info(f"Processing question: {state.question}")
    
    llm_output = call_cortex_llm(state.question, state.session_id)
    tool, query, trace = parse_llm_response(llm_output)
    
    return {
        "question": state.question,
        "session_id": state.session_id,
        "tool": tool or "",
        "query": query or "",
        "trace": trace,
        "answer": "",
        "error_count": state.error_count,
        "last_error": state.last_error
    }

async def execute_tool_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Execute the selected tool"""
    logger.info(f"Executing tool: {state.tool}")
    
    if not state.tool:
        answer = "⚠️ No valid tool selected. Please rephrase your question."
        return {**state.dict(), "answer": answer}
    
    # Execute tool function
    result = await execute_tool_function(state.tool, state.query)
    
    if "error" in result:
        answer = f"❌ **Error:** {result['error']}"
        return {
            **state.dict(),
            "answer": answer,
            "error_count": state.error_count + 1,
            "last_error": result['error']
        }
    
    # Format successful result
    data = result.get("data", {})
    result_type = result.get("type", "unknown")
    
    if result_type == "schema":
        if isinstance(data, dict):
            labels = data.get("labels", [])
            rel_types = data.get("relationship_types", [])
            prop_keys = data.get("property_keys", [])
            answer = f"""📊 **Database Schema:**

**Node Labels ({len(labels)}):** {', '.join(labels[:15])}
**Relationship Types ({len(rel_types)}):** {', '.join(rel_types[:15])}
**Property Keys ({len(prop_keys)}):** {', '.join(prop_keys[:15])}

*Schema data updated in real-time visualization*"""
        else:
            answer = f"📊 **Schema:** {json.dumps(data, indent=2)[:500]}..."
    
    elif result_type == "read":
        if isinstance(data, list):
            count = len(data)
            if count == 0:
                answer = "📊 **Query Result:** No data found"
            elif count == 1 and isinstance(data[0], dict) and len(data[0]) == 1:
                # Single value result (like count)
                key, value = list(data[0].items())[0]
                answer = f"📊 **Query Result:** {key} = **{value:,}**"
            else:
                answer = f"📊 **Query Result:** Found **{count:,}** records\n\n"
                # Show sample data
                for i, record in enumerate(data[:3]):
                    answer += f"**Record {i+1}:** {json.dumps(record, indent=2)}\n\n"
                if count > 3:
                    answer += f"... and **{count - 3:,}** more records"
                answer += "\n\n*Full results shown in graph visualization*"
        else:
            answer = f"📊 **Query Result:** {json.dumps(data, indent=2)[:500]}"
    
    elif result_type == "write":
        if isinstance(data, dict):
            nodes_created = data.get("nodes_created", 0)
            nodes_deleted = data.get("nodes_deleted", 0)
            rels_created = data.get("relationships_created", 0)
            rels_deleted = data.get("relationships_deleted", 0)
            props_set = data.get("properties_set", 0)
            total_changes = data.get("total_changes", 0)
            
            operations = []
            if nodes_created > 0:
                operations.append(f"🟢 **Created {nodes_created:,} node{'s' if nodes_created != 1 else ''}**")
            if nodes_deleted > 0:
                operations.append(f"🗑️ **Deleted {nodes_deleted:,} node{'s' if nodes_deleted != 1 else ''}**")
            if rels_created > 0:
                operations.append(f"🔗 **Created {rels_created:,} relationship{'s' if rels_created != 1 else ''}**")
            if rels_deleted > 0:
                operations.append(f"❌ **Deleted {rels_deleted:,} relationship{'s' if rels_deleted != 1 else ''}**")
            if props_set > 0:
                operations.append(f"📝 **Set {props_set:,} propert{'ies' if props_set != 1 else 'y'}**")
            
            if operations:
                answer = "✅ **Database Update Completed Successfully!**\n\n" + "\n".join(operations)
                if total_changes > 0:
                    answer += f"\n\n**Total Changes:** {total_changes:,}"
                answer += "\n\n*Changes immediately reflected in live graph visualization*"
            else:
                answer = "✅ **Query executed successfully** (no structural changes made)"
        else:
            answer = f"✅ **Write Operation Result:** {data}"
    
    else:
        answer = f"📊 **Result:** {json.dumps(data, indent=2)[:500]}"
    
    return {**state.dict(), "answer": answer}

def build_agent():
    """Build the LangGraph agent"""
    logger.info("Building LangGraph agent...")
    
    workflow = StateGraph(state_schema=AgentState)
    
    # Add nodes
    workflow.add_node("select_tool", RunnableLambda(select_tool_node))
    workflow.add_node("execute_tool", RunnableLambda(execute_tool_node))
    
    # Set entry point and edges
    workflow.set_entry_point("select_tool")
    workflow.add_edge("select_tool", "execute_tool")
    workflow.add_edge("execute_tool", END)
    
    agent = workflow.compile()
    logger.info("✅ LangGraph agent built successfully")
    
    return agent

# Initialize agent
agent = None

# ============================================
# STARTUP AND SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize everything on startup"""
    global agent, database_stats
    
    print("🚀 Starting Working FastAPI Neo4j Server with NVL...")
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
            
            # Get initial database statistics
            print("📊 Getting initial database state...")
            database_stats = await get_database_stats()
            
            if "error" not in database_stats:
                print(f"   📈 Nodes: {database_stats['nodes']}")
                print(f"   🔗 Relationships: {database_stats['relationships']}")
                print(f"   🏷️  Labels: {len(database_stats['labels'])}")
                print(f"   ➡️  Relationship Types: {len(database_stats['relationship_types'])}")
            else:
                print(f"   ⚠️  Could not get database stats: {database_stats['error']}")
            
        else:
            print("❌ Neo4j connection test failed")
            
    except Exception as e:
        print("❌ Neo4j connection failed!")
        print(f"   Error: {e}")
    
    # Build LangGraph agent
    try:
        print("🔨 Building LangGraph agent...")
        agent = build_agent()
        print("✅ LangGraph agent built successfully")
    except Exception as e:
        print(f"❌ Failed to build agent: {e}")
        agent = None
    
    print("=" * 60)
    print(f"🌐 Working FastAPI server ready on http://localhost:{SERVER_PORT}")
    print("📋 Available endpoints:")
    print("   • GET  /health - Health check")
    print("   • POST /chat - Chat with agent")
    print("   • GET  /graph - Get graph data")
    print("   • GET  /stats - Live database stats")
    print("   • WS   /ws - WebSocket for live updates")
    print("   • GET  /viz - Neo4j visualization")
    print("   • GET  /docs - API documentation")
    print("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown"""
    print("🛑 Shutting down Working FastAPI Server...")
    if driver:
        await driver.close()
        print("✅ Neo4j driver closed")

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Working Neo4j FastAPI Server with NVL",
        "version": "3.1.0",
        "description": "Pure FastAPI server (no FastMCP) with LangGraph agent and Neo4j NVL visualization",
        "architecture": "FastAPI + LangGraph + Neo4j + NVL",
        "fix_applied": "Removed FastMCP dependency, using pure FastAPI",
        "endpoints": {
            "health": "/health - System health check",
            "chat": "/chat - Chat with LangGraph agent",
            "graph": "/graph - Get graph data for visualization",
            "stats": "/stats - Live database statistics",
            "viz": "/viz - Neo4j NVL visualization interface",
            "ws": "/ws - WebSocket for live updates",
            "docs": "/docs - API documentation"
        },
        "tools": {
            "read_neo4j_cypher": "Execute read-only Cypher queries",
            "write_neo4j_cypher": "Execute write Cypher queries with live updates", 
            "get_neo4j_schema": "Get database schema"
        },
        "features": ["real_time_visualization", "live_updates", "websocket_support", "langgraph_agent"],
        "neo4j": {
            "uri": NEO4J_URI,
            "database": NEO4J_DATABASE,
            "user": NEO4J_USER,
            "current_stats": database_stats
        }
    }

@app.get("/health")
async def health_check():
    """System health check"""
    if driver is None:
        return {
            "status": "unhealthy",
            "neo4j": {"status": "driver_not_initialized"},
            "agent": {"status": "not_initialized" if agent is None else "ready"},
            "server": {"port": SERVER_PORT, "type": "Working FastAPI"}
        }
    
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run("RETURN 1 as test")
            record = await result.single()
            
        neo4j_status = "connected" if record and record["test"] == 1 else "test_failed"
        current_stats = await get_database_stats()
        
        return {
            "status": "healthy",
            "neo4j": {
                "status": neo4j_status,
                "uri": NEO4J_URI,
                "database": NEO4J_DATABASE,
                "current_stats": current_stats
            },
            "agent": {
                "status": "ready" if agent else "not_initialized",
                "model": CORTEX_MODEL
            },
            "visualization": {
                "status": "enabled",
                "active_connections": len(active_websockets)
            },
            "server": {
                "port": SERVER_PORT,
                "type": "Working FastAPI with NVL",
                "tools": ["read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"]
            }
        }
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy", 
            "neo4j": {"status": "disconnected", "error": str(e)},
            "agent": {"status": "not_initialized" if agent is None else "ready"},
            "server": {"port": SERVER_PORT}
        }

@app.get("/stats")
async def get_live_stats():
    """Get live database statistics"""
    return await get_database_stats()

@app.get("/graph")
async def get_graph_data_endpoint(limit: int = 100):
    """Get graph data for visualization"""
    try:
        graph_data = await get_graph_data(limit)
        return graph_data.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint using LangGraph agent"""
    if agent is None:
        raise HTTPException(
            status_code=503, 
            detail="Agent not initialized. Check server logs for errors."
        )
    
    try:
        session_id = request.session_id or str(uuid.uuid4())
        logger.info(f"Processing question: {request.question}")
        
        # Create agent state
        state = AgentState(
            question=request.question,
            session_id=session_id
        )
        
        # Run the agent
        start_time = time.time()
        result = await agent.ainvoke(state)
        processing_time = time.time() - start_time
        
        # Get graph data if available
        graph_data = None
        operation_summary = None
        
        if result.get("tool") and result.get("query"):
            try:
                graph_data_obj = await get_graph_data(50)
                graph_data = graph_data_obj.dict()
                operation_summary = await get_database_stats()
            except Exception as e:
                logger.warning(f"Could not get graph data: {e}")
        
        logger.info(f"Agent completed in {processing_time:.2f}s - Tool: {result.get('tool')}")
        
        return ChatResponse(
            trace=result.get("trace", ""),
            tool=result.get("tool", ""),
            query=result.get("query", ""),
            answer=result.get("answer", "No answer generated"),
            session_id=session_id,
            success=True,
            graph_data=graph_data,
            operation_summary=operation_summary
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return ChatResponse(
            trace=f"Error: {str(e)}",
            tool="",
            query="",
            answer=f"⚠️ Error processing request: {str(e)}",
            session_id=request.session_id or str(uuid.uuid4()),
            success=False,
            error=str(e)
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live database updates"""
    await websocket.accept()
    active_websockets.add(websocket)
    
    try:
        # Send initial database state
        initial_stats = await get_database_stats()
        await websocket.send_json({
            "type": "initial_state",
            "data": initial_stats
        })
        
        # Keep connection alive and handle messages
        while True:
            try:
                data = await websocket.receive_json()
                
                if data.get("type") == "request_update":
                    current_stats = await get_database_stats()
                    await websocket.send_json({
                        "type": "database_update", 
                        "data": current_stats
                    })
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
                
    finally:
        active_websockets.discard(websocket)

@app.get("/viz", response_class=HTMLResponse)
async def visualization_interface():
    """Neo4j visualization interface using NVL"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neo4j Live Visualization - Working Version</title>
    <script src="https://unpkg.com/neo4j-nvl@latest/dist/index.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0; padding: 0; background: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .controls {
            background: white; padding: 1rem 2rem;
            border-bottom: 1px solid #eee;
            display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;
        }
        .btn {
            background: #667eea; color: white; border: none;
            padding: 0.5rem 1rem; border-radius: 4px;
            cursor: pointer; font-size: 14px; transition: background 0.2s;
        }
        .btn:hover { background: #5a6fd8; }
        .stats {
            display: flex; gap: 2rem; align-items: center; font-size: 14px;
        }
        .stat {
            background: #f8f9fa; padding: 0.5rem 1rem; border-radius: 4px;
            border-left: 3px solid #667eea;
        }
        #nvl-container {
            height: calc(100vh - 200px); background: white;
            border-top: 1px solid #eee;
        }
        .status { display: inline-block; padding: 0.25rem 0.5rem;
            border-radius: 3px; font-size: 12px; font-weight: bold; }
        .status.connected { background: #d4edda; color: #155724; }
        .status.disconnected { background: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Neo4j Live Visualization - Working Version</h1>
        <p>Real-time graph visualization with Neo4j NVL (No FastMCP dependency)</p>
    </div>
    
    <div class="controls">
        <button class="btn" onclick="refreshGraph()">🔄 Refresh Graph</button>
        <button class="btn" onclick="loadAllNodes()">📊 Load All Nodes</button>
        <button class="btn" onclick="clearGraph()">🧹 Clear</button>
        
        <div class="stats">
            <div class="stat"><strong>Nodes:</strong> <span id="node-count">0</span></div>
            <div class="stat"><strong>Relationships:</strong> <span id="rel-count">0</span></div>
            <div class="stat"><strong>Status:</strong> <span id="connection-status" class="status disconnected">Connecting...</span></div>
        </div>
    </div>
    
    <div id="nvl-container"></div>

    <script>
        let nvl; let websocket;
        
        async function initNVL() {
            try {
                const response = await fetch('/graph?limit=50');
                const graphData = await response.json();
                
                nvl = new NVL('nvl-container', graphData.nodes, graphData.relationships, {
                    instanceId: 'working-neo4j-viz',
                    initialZoom: 1.5,
                    allowDynamicMinZoom: true,
                    showPropertiesOnHover: true,
                    nodeColorScheme: 'category10',
                    relationshipColorScheme: 'dark',
                    layout: { algorithm: 'forceDirected', incrementalLayout: true }
                });
                
                updateStats(graphData.summary);
                console.log('✅ NVL initialized successfully');
                
            } catch (error) {
                console.error('❌ Failed to initialize NVL:', error);
                document.getElementById('connection-status').textContent = 'Error';
                document.getElementById('connection-status').className = 'status disconnected';
            }
        }
        
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            websocket = new WebSocket(wsUrl);
            
            websocket.onopen = function(event) {
                console.log('✅ WebSocket connected');
                document.getElementById('connection-status').textContent = 'Connected';
                document.getElementById('connection-status').className = 'status connected';
            };
            
            websocket.onmessage = function(event) {
                const message = JSON.parse(event.data);
                if (message.type === 'database_update' || message.type === 'initial_state') {
                    updateStats(message.data);
                    refreshGraph();
                }
            };
            
            websocket.onclose = function(event) {
                console.log('❌ WebSocket disconnected');
                document.getElementById('connection-status').textContent = 'Disconnected';
                document.getElementById('connection-status').className = 'status disconnected';
                setTimeout(initWebSocket, 5000);
            };
        }
        
        function updateStats(stats) {
            if (stats) {
                document.getElementById('node-count').textContent = stats.nodes || 0;
                document.getElementById('rel-count').textContent = stats.relationships || 0;
            }
        }
        
        async function refreshGraph() {
            try {
                const response = await fetch('/graph?limit=100');
                const graphData = await response.json();
                if (nvl) {
                    nvl.updateGraph(graphData.nodes, graphData.relationships);
                    updateStats(graphData.summary);
                }
            } catch (error) {
                console.error('Failed to refresh graph:', error);
            }
        }
        
        async function loadAllNodes() {
            try {
                const response = await fetch('/graph?limit=500');
                const graphData = await response.json();
                if (nvl) {
                    nvl.updateGraph(graphData.nodes, graphData.relationships);
                    updateStats(graphData.summary);
                }
            } catch (error) {
                console.error('Failed to load all nodes:', error);
            }
        }
        
        function clearGraph() {
            if (nvl) {
                nvl.clearGraph();
                updateStats({nodes: 0, relationships: 0});
            }
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            initNVL();
            initWebSocket();
        });
    </script>
</body>
</html>
    """, media_type="text/html")

# Legacy endpoints for compatibility
@app.post("/read_neo4j_cypher")
async def legacy_read_cypher(request: CypherRequest):
    """Legacy endpoint for read operations"""
    try:
        result = await read_neo4j_cypher(request.query, request.params)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/write_neo4j_cypher")
async def legacy_write_cypher(request: CypherRequest):
    """Legacy endpoint for write operations"""
    try:
        result = await write_neo4j_cypher(request.query, request.params)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_neo4j_schema")
async def legacy_get_schema():
    """Legacy endpoint for schema operations"""
    try:
        result = await get_neo4j_schema()
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """Main function to run the working FastAPI server"""
    print("=" * 70)
    print("🧠 WORKING NEO4J FASTAPI SERVER WITH NVL")
    print("=" * 70)
    print("🔧 Fix Applied: Removed FastMCP dependency, using pure FastAPI")
    print("🏗️  Architecture: FastAPI + LangGraph + Neo4j + NVL")
    print("🔧 Configuration:")
    print(f"   📍 Neo4j URI: {NEO4J_URI}")
    print(f"   👤 Neo4j User: {NEO4J_USER}")
    print(f"   🗄️  Neo4j Database: {NEO4J_DATABASE}")
    print(f"   🤖 Cortex Model: {CORTEX_MODEL}")
    print(f"   🌐 Server: {SERVER_HOST}:{SERVER_PORT}")
    print("=" * 70)
    print("✨ Features:")
    print("   🎯 Neo4j visualization with NVL")
    print("   📊 Live database statistics")
    print("   🔄 WebSocket live updates")
    print("   📈 LangGraph AI agent")
    print("   🖥️  Visualization interface at /viz")
    print("   📚 API documentation at /docs")
    print("=" * 70)
    
    if NEO4J_PASSWORD == "your_neo4j_password":
        print("⚠️  WARNING: Change NEO4J_PASSWORD before using!")
    
    if driver is None:
        print("❌ Cannot start - Neo4j driver failed to initialize")
        return
    
    print("🚀 Starting working FastAPI server...")
    
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
