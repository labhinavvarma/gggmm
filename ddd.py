"""
Fixed MCP Server with All Required Endpoints for Streamlit UI
This ensures all endpoints match what the UI expects
"""

import asyncio
import json
import logging
import uuid
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

# FastMCP and FastAPI imports
from fastmcp import FastMCP
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
logger = logging.getLogger("fixed_mcp_server")

print("🔧 Fixed MCP Server Configuration:")
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

# Initialize FastMCP
mcp = FastMCP("Fixed Neo4j MCP Server")

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

class AgentState(BaseModel):
    question: str
    session_id: str
    tool: str = ""
    query: str = ""
    trace: str = ""
    answer: str = ""
    error_count: int = 0
    last_error: str = ""

# ============================================
# MCP TOOLS WITH @mcp.tool DECORATORS
# ============================================

@mcp.tool()
async def read_neo4j_cypher(query: str, params: dict = {}) -> List[Dict[str, Any]]:
    """
    Execute read-only Cypher queries against Neo4j database.
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
    Execute write Cypher queries against Neo4j database.
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

@mcp.tool()
async def get_neo4j_schema() -> Dict[str, Any]:
    """
    Get the schema of the Neo4j database.
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
            "source": "fixed_mcp_server"
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
# LANGGRAPH AGENT LOGIC
# ============================================

SYSTEM_PROMPT = """
You are an expert AI assistant that helps users query and manage a Neo4j database using MCP tools.

AVAILABLE TOOLS:
- read_neo4j_cypher: Execute read-only Cypher queries (MATCH, RETURN, WHERE, etc.)
- write_neo4j_cypher: Execute write Cypher queries (CREATE, MERGE, SET, DELETE, etc.)
- get_neo4j_schema: Get database schema information

GUIDELINES:
- Always explain your reasoning before selecting a tool
- Choose the appropriate tool based on the user's intent
- For schema questions, use get_neo4j_schema
- For data queries, use read_neo4j_cypher
- For data modifications, use write_neo4j_cypher

RESPONSE FORMAT:
Tool: [tool_name]
Query: [cypher_query_on_single_line]

EXAMPLES:

User: How many nodes are in the graph?
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN count(n)

User: Create a Person named Alice
Tool: write_neo4j_cypher
Query: CREATE (:Person {name: 'Alice'})

User: Show the schema
Tool: get_neo4j_schema

Always provide the exact tool name and query (if applicable).
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

async def execute_mcp_tool(tool: str, query: str = None) -> Dict[str, Any]:
    """Execute MCP tool directly"""
    try:
        if tool == "get_neo4j_schema":
            result = await get_neo4j_schema()
            return {"success": True, "data": result}
        
        elif tool == "read_neo4j_cypher":
            if not query:
                return {"error": "No query provided for read operation"}
            result = await read_neo4j_cypher(query, {})
            return {"success": True, "data": result}
        
        elif tool == "write_neo4j_cypher":
            if not query:
                return {"error": "No query provided for write operation"}
            result = await write_neo4j_cypher(query, {})
            return {"success": True, "data": result}
        
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
    
    # Execute MCP tool directly
    result = await execute_mcp_tool(state.tool, state.query)
    
    if "error" in result:
        answer = f"⚠️ Error: {result['error']}"
        return {
            **state.dict(),
            "answer": answer,
            "error_count": state.error_count + 1,
            "last_error": result['error']
        }
    
    # Format successful result
    data = result.get("data", {})
    
    if state.tool == "get_neo4j_schema":
        if isinstance(data, dict):
            labels = data.get("labels", [])
            rel_types = data.get("relationship_types", [])
            answer = f"📊 **Database Schema:**\n\n**Node Labels:** {', '.join(labels[:10])}\n**Relationship Types:** {', '.join(rel_types[:10])}"
        else:
            answer = f"📊 **Schema:** {json.dumps(data, indent=2)[:500]}..."
    
    elif state.tool == "read_neo4j_cypher":
        if isinstance(data, list):
            count = len(data)
            if count == 0:
                answer = "📊 **Result:** No data found"
            elif count == 1 and isinstance(data[0], dict) and len(data[0]) == 1:
                # Single value result (like count)
                key, value = list(data[0].items())[0]
                answer = f"📊 **Result:** {value}"
            else:
                answer = f"📊 **Result:** Found {count} records\n\n{json.dumps(data[:3], indent=2)}"
                if count > 3:
                    answer += f"\n... and {count - 3} more records"
        else:
            answer = f"📊 **Result:** {json.dumps(data, indent=2)[:500]}"
    
    elif state.tool == "write_neo4j_cypher":
        if isinstance(data, dict):
            created = data.get("nodes_created", 0)
            deleted = data.get("nodes_deleted", 0)
            rels_created = data.get("relationships_created", 0)
            rels_deleted = data.get("relationships_deleted", 0)
            answer = f"✅ **Write Operation Completed:**\n- Nodes created: {created}\n- Nodes deleted: {deleted}\n- Relationships created: {rels_created}\n- Relationships deleted: {rels_deleted}"
        else:
            answer = f"✅ **Write operation completed:** {data}"
    
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
# FASTAPI ENDPOINTS (FIXED)
# ============================================

# Add FastAPI endpoints to FastMCP
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
    """Initialize everything on startup"""
    global agent
    
    print("🚀 Starting Fixed MCP Neo4j Server...")
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
    
    # Build LangGraph agent
    try:
        print("🔨 Building LangGraph agent...")
        agent = build_agent()
        print("✅ LangGraph agent built successfully")
    except Exception as e:
        print(f"❌ Failed to build agent: {e}")
        agent = None
    
    print("=" * 50)
    print(f"🌐 Fixed MCP server ready on http://localhost:{SERVER_PORT}")
    print("📋 Available endpoints:")
    print("   • GET  /health - Health check")
    print("   • POST /chat - Chat with agent")
    print("   • GET  /graph - Graph data")
    print("   • GET  /stats - Database statistics")
    print("   • MCP tools available via FastMCP")
    print("=" * 50)

@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown"""
    print("🛑 Shutting down Fixed MCP Server...")
    if driver:
        await driver.close()
        print("✅ Neo4j driver closed")

# ============================================
# FIXED ENDPOINTS TO MATCH UI EXPECTATIONS
# ============================================

@app.get("/health")
async def health_check():
    """Health check endpoint - FIXED"""
    if driver is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "neo4j": {"status": "driver_not_initialized"},
                "agent": {"status": "not_initialized" if agent is None else "ready"},
                "server": {"port": SERVER_PORT, "type": "Fixed MCP"}
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
            "agent": {
                "status": "ready" if agent else "not_initialized"
            },
            "server": {
                "port": SERVER_PORT,
                "host": SERVER_HOST,
                "type": "Fixed MCP",
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
                "agent": {"status": "not_initialized" if agent is None else "ready"},
                "server": {"port": SERVER_PORT, "type": "Fixed MCP"}
            }
        )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint - FIXED"""
    if agent is None:
        raise HTTPException(
            status_code=503, 
            detail="Agent not initialized. Check server logs for errors."
        )
    
    try:
        # Generate session ID if not provided
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
        
        logger.info(f"Agent completed in {processing_time:.2f}s - Tool: {result.get('tool')}")
        
        return ChatResponse(
            trace=result.get("trace", ""),
            tool=result.get("tool", ""),
            query=result.get("query", ""),
            answer=result.get("answer", "No answer generated"),
            session_id=session_id,
            success=True
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

@app.get("/stats")
async def get_database_stats():
    """Database statistics endpoint - FIXED (was missing)"""
    if driver is None:
        raise HTTPException(status_code=503, detail="Neo4j driver not initialized")
    
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            # Get node count
            node_result = await session.run("MATCH (n) RETURN count(n) as nodes")
            node_record = await node_result.single()
            nodes = node_record["nodes"] if node_record else 0
            
            # Get relationship count
            rel_result = await session.run("MATCH ()-[r]->() RETURN count(r) as relationships")
            rel_record = await rel_result.single()
            relationships = rel_record["relationships"] if rel_record else 0
            
            # Get labels
            labels_result = await session.run("CALL db.labels() YIELD label RETURN collect(label) as labels")
            labels_record = await labels_result.single()
            labels = labels_record["labels"] if labels_record else []
            
            # Get relationship types
            types_result = await session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types")
            types_record = await types_result.single()
            relationship_types = types_record["types"] if types_record else []
        
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
    """Graph data endpoint - FIXED (was missing)"""
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
                
                # Create meaningful caption
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
                # Get relationships between loaded nodes
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

@app.get("/")
async def root():
    """Root endpoint - FIXED"""
    return {
        "service": "Fixed Neo4j MCP Server",
        "version": "2.0.0",
        "description": "Fixed MCP server with all required endpoints for Streamlit UI",
        "architecture": "FastMCP + LangGraph + Neo4j",
        "endpoints": {
            "health": "/health - Health check",
            "chat": "/chat - Chat with LangGraph agent",
            "stats": "/stats - Database statistics",
            "graph": "/graph - Graph data for visualization",
            "docs": "/docs - FastAPI documentation"
        },
        "mcp_tools": {
            "read_neo4j_cypher": "Execute read-only Cypher queries",
            "write_neo4j_cypher": "Execute write Cypher queries", 
            "get_neo4j_schema": "Get database schema"
        },
        "neo4j": {
            "uri": NEO4J_URI,
            "database": NEO4J_DATABASE,
            "user": NEO4J_USER
        },
        "agent": {
            "status": "ready" if agent else "not_initialized",
            "model": CORTEX_MODEL
        }
    }

# Legacy endpoints for backward compatibility
@app.post("/read_neo4j_cypher")
async def legacy_read_cypher(request: CypherRequest):
    """Legacy endpoint for read operations"""
    try:
        result = await read_neo4j_cypher(request.query, request.params)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/write_neo4j_cypher")
async def legacy_write_cypher(request: CypherRequest):
    """Legacy endpoint for write operations"""
    try:
        result = await write_neo4j_cypher(request.query, request.params)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_neo4j_schema")
async def legacy_get_schema():
    """Legacy endpoint for schema operations"""
    try:
        result = await get_neo4j_schema()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """Main function to run the fixed MCP server"""
    print("=" * 60)
    print("🧠 FIXED NEO4J MCP SERVER")
    print("=" * 60)
    print("🏗️  Architecture: Fixed FastMCP + LangGraph + Neo4j")
    print("🔧 Configuration:")
    print(f"   📍 Neo4j URI: {NEO4J_URI}")
    print(f"   👤 Neo4j User: {NEO4J_USER}")
    print(f"   🗄️  Neo4j Database: {NEO4J_DATABASE}")
    print(f"   🤖 Cortex Model: {CORTEX_MODEL}")
    print(f"   🌐 Server: {SERVER_HOST}:{SERVER_PORT}")
    print("=" * 60)
    print("✅ Fixed endpoints:")
    print("   • /health - Health check")
    print("   • /chat - Chat with agent")
    print("   • /stats - Database statistics")
    print("   • /graph - Graph data")
    print("   • All MCP tools available")
    print("=" * 60)
    
    if driver is None:
        print("❌ Cannot start server - Neo4j driver failed to initialize")
        return
    
    print("🚀 Starting fixed MCP server...")
    
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
