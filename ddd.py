import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncTransaction

# Correct MCP imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_neo4j_server")

# Neo4j Configuration
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_neo4j_password"
NEO4J_DATABASE = "neo4j"

driver: AsyncDriver = AsyncGraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# Initialize MCP Server properly
server = Server("neo4j-mcp-server")

# FastAPI app
app = FastAPI(title="MCP Neo4j Server with FastAPI", version="1.0.0")

# Pydantic models for FastAPI
class CypherRequest(BaseModel):
    query: str
    params: dict = {}
    node_limit: int = 5000

# Utility functions
def extract_graph_data_optimized(records, node_limit=2000):
    """Extract nodes and relationships from Neo4j records optimized for visualization"""
    nodes = {}
    relationships = []
    
    # Process records and extract graph objects
    for record in records:
        for key, value in record.items():
            # Handle nodes
            if hasattr(value, 'labels'):  # It's a node
                node_id = str(value.element_id)
                if len(nodes) < node_limit:
                    properties = dict(value)
                    if 'name' not in properties and 'title' not in properties:
                        properties['name'] = f"Node {len(nodes) + 1}"
                    
                    nodes[node_id] = {
                        'id': node_id,
                        'labels': list(value.labels),
                        'properties': properties
                    }
            
            # Handle relationships
            elif hasattr(value, 'type'):  # It's a relationship
                start_node_id = str(value.start_node.element_id)
                end_node_id = str(value.end_node.element_id)
                
                rel = {
                    'id': str(value.element_id),
                    'type': value.type,
                    'startNode': start_node_id,
                    'endNode': end_node_id,
                    'properties': dict(value)
                }
                relationships.append(rel)
                
                # Add connected nodes if not already present
                if start_node_id not in nodes and len(nodes) < node_limit:
                    start_props = dict(value.start_node)
                    if 'name' not in start_props and 'title' not in start_props:
                        start_props['name'] = f"Node {len(nodes) + 1}"
                    
                    nodes[start_node_id] = {
                        'id': start_node_id,
                        'labels': list(value.start_node.labels),
                        'properties': start_props
                    }
                
                if end_node_id not in nodes and len(nodes) < node_limit:
                    end_props = dict(value.end_node)
                    if 'name' not in end_props and 'title' not in end_props:
                        end_props['name'] = f"Node {len(nodes) + 1}"
                    
                    nodes[end_node_id] = {
                        'id': end_node_id,
                        'labels': list(value.end_node.labels),
                        'properties': end_props
                    }
            
            # Handle lists
            elif isinstance(value, list):
                for item in value:
                    if hasattr(item, 'labels') and len(nodes) < node_limit:
                        node_id = str(item.element_id)
                        item_props = dict(item)
                        if 'name' not in item_props and 'title' not in item_props:
                            item_props['name'] = f"Node {len(nodes) + 1}"
                        
                        nodes[node_id] = {
                            'id': node_id,
                            'labels': list(item.labels),
                            'properties': item_props
                        }
                    elif hasattr(item, 'type'):
                        rel = {
                            'id': str(item.element_id),
                            'type': item.type,
                            'startNode': str(item.start_node.element_id),
                            'endNode': str(item.end_node.element_id),
                            'properties': dict(item)
                        }
                        relationships.append(rel)
    
    # Filter relationships to only include those between visible nodes
    visible_node_ids = set(nodes.keys())
    filtered_relationships = [
        rel for rel in relationships 
        if rel['startNode'] in visible_node_ids and rel['endNode'] in visible_node_ids
    ]
    
    return {
        'nodes': list(nodes.values()),
        'relationships': filtered_relationships,
        'total_nodes': len(nodes),
        'total_relationships': len(filtered_relationships),
        'limited': len(nodes) >= node_limit
    }

def format_change_summary(counters, query: str, execution_time: float):
    """Format a detailed summary of what changed in Neo4j"""
    timestamp = datetime.now().isoformat()
    
    changes = []
    if counters.nodes_created > 0:
        changes.append(f"✅ {counters.nodes_created} node(s) created")
    if counters.nodes_deleted > 0:
        changes.append(f"🗑️ {counters.nodes_deleted} node(s) deleted")
    if counters.relationships_created > 0:
        changes.append(f"🔗 {counters.relationships_created} relationship(s) created")
    if counters.relationships_deleted > 0:
        changes.append(f"💥 {counters.relationships_deleted} relationship(s) deleted")
    if counters.properties_set > 0:
        changes.append(f"📝 {counters.properties_set} property(ies) set")
    if counters.labels_added > 0:
        changes.append(f"🏷️ {counters.labels_added} label(s) added")
    if counters.labels_removed > 0:
        changes.append(f"🏷️ {counters.labels_removed} label(s) removed")
    
    if not changes:
        changes.append("ℹ️ No changes detected")
    
    return {
        "timestamp": timestamp,
        "execution_time_ms": round(execution_time * 1000, 2),
        "query": query,
        "changes": changes,
        "summary": f"🕐 {timestamp} | ⚡ {round(execution_time * 1000, 2)}ms | {' | '.join(changes)}",
        "raw_counters": {
            "nodes_created": counters.nodes_created,
            "nodes_deleted": counters.nodes_deleted,
            "relationships_created": counters.relationships_created,
            "relationships_deleted": counters.relationships_deleted,
            "properties_set": counters.properties_set,
            "labels_added": counters.labels_added,
            "labels_removed": counters.labels_removed,
        }
    }

# Core database functions
async def _read(tx: AsyncTransaction, query: str, params: dict):
    res = await tx.run(query, params)
    records = await res.to_eager_result()
    return json.dumps([r.data() for r in records.records], default=str)

async def _read_for_viz(tx: AsyncTransaction, query: str, params: dict):
    """Read query specifically for graph visualization"""
    res = await tx.run(query, params)
    records = await res.to_eager_result()
    return records.records

async def _write(tx: AsyncTransaction, query: str, params: dict):
    return await tx.run(query, params)

# =============================================================================
# MCP TOOLS SECTION WITH PROPER @server.tool() DECORATORS
# =============================================================================

@server.tool()
async def read_neo4j_cypher(
    query: str,
    params: Optional[str] = None,
    node_limit: int = 5000
) -> str:
    """
    Execute a read-only Cypher query against the Neo4j database.
    
    Args:
        query: The Cypher query to execute (must be read-only)
        params: Optional JSON string of parameters for the query
        node_limit: Maximum number of nodes to return for visualization
    
    Returns:
        JSON string containing query results, metadata, and optional graph data
    """
    try:
        start_time = datetime.now()
        
        # Parse params if provided
        parsed_params = json.loads(params) if params else {}
        
        # Validate that this is a read-only query
        write_keywords = ['CREATE', 'MERGE', 'SET', 'DELETE', 'REMOVE', 'DROP']
        query_upper = query.upper().strip()
        if any(keyword in query_upper for keyword in write_keywords):
            raise ValueError("Write operations not allowed in read_neo4j_cypher. Use write_neo4j_cypher instead.")
        
        logger.info(f"🔍 MCP: Executing read query: {query[:100]}...")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_read(_read, query, parsed_params)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        result_data = json.loads(result)
        
        # Extract graph data for visualization
        graph_data = None
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                viz_result = await session.execute_read(_read_for_viz, query, parsed_params)
            graph_data = extract_graph_data_optimized(viz_result, node_limit)
        except Exception as e:
            logger.warning(f"Could not extract graph data: {e}")
        
        response = {
            "success": True,
            "data": result_data,
            "metadata": {
                "timestamp": start_time.isoformat(),
                "execution_time_ms": round(execution_time * 1000, 2),
                "query": query,
                "record_count": len(result_data),
                "node_limit": node_limit,
                "has_graph_data": bool(graph_data and (graph_data.get('nodes') or graph_data.get('relationships')))
            },
            "graph_data": graph_data if graph_data and (graph_data.get('nodes') or graph_data.get('relationships')) else None
        }
        
        logger.info(f"✅ MCP: Read query completed: {len(result_data)} records, {execution_time*1000:.1f}ms")
        return json.dumps(response)
        
    except Exception as e:
        logger.error(f"❌ MCP: Error in read_neo4j_cypher: {e}")
        error_response = {
            "success": False,
            "error": str(e),
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
        return json.dumps(error_response)

@server.tool()
async def write_neo4j_cypher(
    query: str,
    params: Optional[str] = None,
    node_limit: int = 5000
) -> str:
    """
    Execute a write Cypher query against the Neo4j database.
    
    Args:
        query: The Cypher query to execute (CREATE, MERGE, SET, DELETE, etc.)
        params: Optional JSON string of parameters for the query
        node_limit: Maximum number of nodes to return for visualization
    
    Returns:
        JSON string containing operation results, change summary, and optional graph data
    """
    try:
        start_time = datetime.now()
        
        # Parse params if provided
        parsed_params = json.loads(params) if params else {}
        
        logger.info(f"✏️  MCP: Executing write query: {query[:100]}...")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_write(_write, query, parsed_params)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Format detailed change information
        change_info = format_change_summary(result._summary.counters, query, execution_time)
        
        logger.info(f"💾 MCP: Neo4j Write Operation: {change_info['summary']}")
        
        response = {
            "success": True,
            "result": "SUCCESS",
            "change_info": change_info
        }
        
        # If the write operation returns data (RETURN clause), get visualization data
        graph_data = None
        try:
            if "RETURN" in query.upper():
                async with driver.session(database=NEO4J_DATABASE) as session:
                    viz_result = await session.execute_read(_read_for_viz, query, parsed_params)
                graph_data = extract_graph_data_optimized(viz_result, node_limit)
                response["graph_data"] = graph_data
                logger.info(f"📊 MCP: Write operation returned graph data: {len(graph_data.get('nodes', []))} nodes")
        except Exception as e:
            logger.warning(f"Could not extract graph data from write operation: {e}")
            response["graph_data"] = None
            
        return json.dumps(response)
        
    except Exception as e:
        logger.error(f"❌ MCP: Error in write_neo4j_cypher: {e}")
        error_response = {
            "success": False,
            "error": str(e),
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
        return json.dumps(error_response)

@server.tool()
async def get_neo4j_schema() -> str:
    """
    Get the Neo4j database schema including node labels, relationships, and properties.
    
    Returns:
        JSON string containing the database schema information
    """
    try:
        start_time = datetime.now()
        get_schema_query = "CALL apoc.meta.schema();"
        
        logger.info("🏗️  MCP: Retrieving database schema...")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_read(_read, get_schema_query, {})
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        schema_data = json.loads(result)
        schema = schema_data[0].get('value') if schema_data else {}
        
        logger.info(f"✅ MCP: Schema retrieved: {len(schema)} node types, {execution_time*1000:.1f}ms")
        
        response = {
            "success": True,
            "schema": schema,
            "metadata": {
                "timestamp": start_time.isoformat(),
                "execution_time_ms": round(execution_time * 1000, 2),
                "query": get_schema_query,
                "node_types_count": len(schema)
            },
            "graph_data": None
        }
        
        return json.dumps(response)
        
    except Exception as e:
        logger.error(f"❌ MCP: Error in get_neo4j_schema: {e}")
        error_response = {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        return json.dumps(error_response)

@server.tool()
async def get_graph_stats() -> str:
    """
    Get comprehensive statistics about the Neo4j graph database.
    
    Returns:
        JSON string containing various graph statistics
    """
    stats_queries = [
        ("total_nodes", "MATCH (n) RETURN count(n) as count"),
        ("total_relationships", "MATCH ()-[r]->() RETURN count(r) as count"),
        ("node_labels", "CALL db.labels() YIELD label RETURN collect(label) as labels"),
        ("relationship_types", "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"),
        ("node_label_counts", "MATCH (n) RETURN labels(n)[0] as label, count(*) as count ORDER BY count DESC"),
        ("relationship_type_counts", "MATCH ()-[r]->() RETURN type(r) as type, count(*) as count ORDER BY count DESC")
    ]
    
    stats = {}
    start_time = datetime.now()
    
    logger.info("📊 MCP: Collecting graph statistics...")
    
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            for stat_name, query in stats_queries:
                try:
                    result = await session.execute_read(_read, query, {})
                    data = json.loads(result)
                    stats[stat_name] = data
                    logger.debug(f"✅ Collected {stat_name}")
                except Exception as e:
                    logger.warning(f"⚠️  Could not get {stat_name}: {e}")
                    stats[stat_name] = []
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Calculate summary metrics
        total_nodes = stats.get('total_nodes', [{}])[0].get('count', 0)
        total_rels = stats.get('total_relationships', [{}])[0].get('count', 0)
        
        logger.info(f"✅ MCP: Statistics collected: {total_nodes} nodes, {total_rels} relationships, {execution_time*1000:.1f}ms")
        
        response = {
            "success": True,
            "stats": stats,
            "summary": {
                "total_nodes": total_nodes,
                "total_relationships": total_rels,
                "density": total_rels / max(total_nodes, 1) if total_nodes > 0 else 0
            },
            "metadata": {
                "timestamp": start_time.isoformat(),
                "execution_time_ms": round(execution_time * 1000, 2)
            }
        }
        
        return json.dumps(response)
        
    except Exception as e:
        logger.error(f"❌ MCP: Error getting graph stats: {e}")
        error_response = {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        return json.dumps(error_response)

@server.tool()
async def get_sample_graph(node_limit: int = 200) -> str:
    """
    Get a sample of the graph for visualization.
    
    Args:
        node_limit: Maximum number of nodes to return
    
    Returns:
        JSON string containing sample graph data
    """
    # Cap the node limit for performance
    safe_limit = min(node_limit, 1000)
    query = f"""
    MATCH (n)-[r]->(m)
    RETURN n, r, m
    LIMIT {safe_limit}
    """
    
    try:
        start_time = datetime.now()
        
        logger.info(f"🎲 MCP: Getting sample graph with limit {safe_limit}...")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_read(_read_for_viz, query, {})
        
        graph_data = extract_graph_data_optimized(result, safe_limit)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        node_count = len(graph_data.get('nodes', []))
        rel_count = len(graph_data.get('relationships', []))
        
        logger.info(f"✅ MCP: Sample graph retrieved: {node_count} nodes, {rel_count} relationships, {execution_time*1000:.1f}ms")
        
        response = {
            "success": True,
            "graph_data": graph_data,
            "metadata": {
                "query": query,
                "timestamp": start_time.isoformat(),
                "execution_time_ms": round(execution_time * 1000, 2),
                "requested_limit": node_limit,
                "actual_limit": safe_limit,
                "nodes_returned": node_count,
                "relationships_returned": rel_count
            }
        }
        
        return json.dumps(response)
        
    except Exception as e:
        logger.error(f"❌ MCP: Error getting sample graph: {e}")
        error_response = {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        return json.dumps(error_response)

# =============================================================================
# FASTAPI ENDPOINTS SECTION (Wrapper functions for MCP tools)
# =============================================================================

@app.get("/")
async def health_check():
    """Enhanced health check endpoint"""
    try:
        # Test database connection
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_read(_read, "MATCH (n) RETURN count(n) as count LIMIT 1", {})
            node_count = json.loads(result)[0].get('count', 0)
        
        return {
            "status": "healthy", 
            "service": "Neo4j MCP Server with FastAPI", 
            "version": "2.0.0",
            "neo4j_connection": "✅ Connected",
            "database": NEO4J_DATABASE,
            "total_nodes": node_count,
            "mcp_tools": [
                "read_neo4j_cypher", 
                "write_neo4j_cypher", 
                "get_neo4j_schema", 
                "get_graph_stats", 
                "get_sample_graph"
            ],
            "max_nodes": 5000,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Neo4j MCP Server with FastAPI",
            "neo4j_connection": f"❌ Failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.post("/read_neo4j_cypher")
async def api_read_neo4j_cypher(request: CypherRequest):
    """FastAPI endpoint that calls the MCP tool"""
    try:
        # Convert params to JSON string for MCP tool
        params_json = json.dumps(request.params) if request.params else None
        
        # Call the MCP tool
        result_json = await read_neo4j_cypher(
            query=request.query,
            params=params_json,
            node_limit=request.node_limit
        )
        
        # Parse the result
        result = json.loads(result_json)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"FastAPI read endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/write_neo4j_cypher")
async def api_write_neo4j_cypher(request: CypherRequest):
    """FastAPI endpoint that calls the MCP tool"""
    try:
        # Convert params to JSON string for MCP tool
        params_json = json.dumps(request.params) if request.params else None
        
        # Call the MCP tool
        result_json = await write_neo4j_cypher(
            query=request.query,
            params=params_json,
            node_limit=request.node_limit
        )
        
        # Parse the result
        result = json.loads(result_json)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"FastAPI write endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_neo4j_schema")
async def api_get_neo4j_schema():
    """FastAPI endpoint that calls the MCP tool"""
    try:
        # Call the MCP tool
        result_json = await get_neo4j_schema()
        
        # Parse the result
        result = json.loads(result_json)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"FastAPI schema endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph_stats")
async def api_get_graph_stats():
    """FastAPI endpoint that calls the MCP tool"""
    try:
        # Call the MCP tool
        result_json = await get_graph_stats()
        
        # Parse the result
        result = json.loads(result_json)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"FastAPI stats endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sample_graph")
async def api_get_sample_graph(node_limit: int = 200):
    """FastAPI endpoint that calls the MCP tool"""
    try:
        # Call the MCP tool
        result_json = await get_sample_graph(node_limit=node_limit)
        
        # Parse the result
        result = json.loads(result_json)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"FastAPI sample graph endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimize_query/{node_limit}")
async def optimize_query_for_limit(node_limit: int, query: str):
    """Optimize a query to respect node limits"""
    try:
        optimized_query = query
        
        if "MATCH" in query.upper() and "LIMIT" not in query.upper():
            if "RETURN" in query.upper():
                parts = query.rsplit("RETURN", 1)
                if len(parts) == 2:
                    optimized_query = f"{parts[0]}RETURN {parts[1]} LIMIT {node_limit}"
            else:
                optimized_query = f"{query} LIMIT {node_limit}"
        
        return {
            "success": True,
            "original_query": query,
            "optimized_query": optimized_query,
            "node_limit": node_limit,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Query optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# MCP SERVER SETUP
# =============================================================================

async def run_mcp_server():
    """Run the MCP server"""
    logger.info("🚀 Starting Neo4j MCP Server with @server.tool() decorators...")
    logger.info("🔧 MCP Tools: read_neo4j_cypher, write_neo4j_cypher, get_neo4j_schema, get_graph_stats, get_sample_graph")
    
    # Run the server using stdio transport
    async with server:
        await stdio_server(server)

def run_fastapi_server():
    """Run the FastAPI server"""
    import uvicorn
    logger.info("🚀 Starting Neo4j FastAPI Server with MCP Tools...")
    logger.info("🔧 FastAPI endpoints wrapping MCP tools with @server.tool() decorators")
    logger.info(f"🔗 Neo4j Connection: {NEO4J_URI}")
    uvicorn.run("mcpserver:app", host="0.0.0.0", port=8000, reload=True)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "mcp":
        # Run as MCP server
        asyncio.run(run_mcp_server())
    else:
        # Run as FastAPI server (default)
        run_fastapi_server()
