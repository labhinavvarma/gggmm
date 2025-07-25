import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Neo4j imports
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncTransaction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_neo4j_server")

# Neo4j Configuration
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_neo4j_password"
NEO4J_DATABASE = "neo4j"

# Initialize Neo4j driver
driver: AsyncDriver = AsyncGraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# Create FastAPI instance
app = FastAPI(
    title="Neo4j Graph Explorer MCP Server",
    description="MCP-style server with FastAPI endpoints for Neo4j graph database",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class CypherRequest(BaseModel):
    query: str
    params: dict = {}
    node_limit: int = 5000

class CypherResponse(BaseModel):
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    graph_data: Optional[Dict[str, Any]] = None

class MCPToolRequest(BaseModel):
    tool: str
    arguments: Dict[str, Any] = {}

class MCPToolResponse(BaseModel):
    tool: str
    result: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None
    timestamp: str

# Enhanced graph data extraction for unlimited display
def extract_graph_data_unlimited(records, node_limit=None):
    """
    Extract nodes and relationships with unlimited display capability
    Optimized for Neo4j Browser-like experience
    """
    nodes = {}
    relationships = []
    
    # If no limit specified, process all data
    effective_limit = node_limit if node_limit is not None else float('inf')
    
    logger.info(f"🕸️ Processing graph data with limit: {'unlimited' if node_limit is None else node_limit}")
    
    # Process records and extract graph objects
    for record in records:
        for key, value in record.items():
            # Handle nodes
            if hasattr(value, 'labels'):  # It's a node
                node_id = str(value.element_id)
                if len(nodes) < effective_limit:
                    # Enhanced property extraction
                    properties = dict(value)
                    
                    # Ensure we have a meaningful display name
                    if not any(prop in properties for prop in ['name', 'title', 'displayName']):
                        # Create a meaningful name based on properties
                        labels = list(value.labels)
                        if labels:
                            # Use first significant property or create from label
                            significant_props = [k for k, v in properties.items() 
                                               if isinstance(v, str) and len(str(v)) < 50]
                            if significant_props:
                                properties['name'] = f"{labels[0]}_{properties[significant_props[0]]}"
                            else:
                                properties['name'] = f"{labels[0]}_{len(nodes) + 1}"
                        else:
                            properties['name'] = f"Node_{len(nodes) + 1}"
                    
                    nodes[node_id] = {
                        'id': node_id,
                        'labels': list(value.labels),
                        'properties': properties,
                        'degree': 0  # Will be calculated later
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
                
                # Add connected nodes if not already present and within limit
                for node_ref, node_id_key in [(value.start_node, start_node_id), (value.end_node, end_node_id)]:
                    if node_id_key not in nodes and len(nodes) < effective_limit:
                        node_props = dict(node_ref)
                        if not any(prop in node_props for prop in ['name', 'title', 'displayName']):
                            labels = list(node_ref.labels)
                            if labels:
                                node_props['name'] = f"{labels[0]}_{len(nodes) + 1}"
                            else:
                                node_props['name'] = f"Node_{len(nodes) + 1}"
                        
                        nodes[node_id_key] = {
                            'id': node_id_key,
                            'labels': list(node_ref.labels),
                            'properties': node_props,
                            'degree': 0
                        }
            
            # Handle lists (might contain nodes/relationships)
            elif isinstance(value, list):
                for item in value:
                    if hasattr(item, 'labels') and len(nodes) < effective_limit:  # Node in list
                        node_id = str(item.element_id)
                        if node_id not in nodes:
                            item_props = dict(item)
                            if not any(prop in item_props for prop in ['name', 'title', 'displayName']):
                                labels = list(item.labels)
                                if labels:
                                    item_props['name'] = f"{labels[0]}_{len(nodes) + 1}"
                                else:
                                    item_props['name'] = f"Node_{len(nodes) + 1}"
                            
                            nodes[node_id] = {
                                'id': node_id,
                                'labels': list(item.labels),
                                'properties': item_props,
                                'degree': 0
                            }
                    elif hasattr(item, 'type'):  # Relationship in list
                        rel = {
                            'id': str(item.element_id),
                            'type': item.type,
                            'startNode': str(item.start_node.element_id),
                            'endNode': str(item.end_node.element_id),
                            'properties': dict(item)
                        }
                        relationships.append(rel)
    
    # Calculate node degrees for Neo4j Browser-like sizing
    node_degrees = {}
    visible_node_ids = set(nodes.keys())
    
    for rel in relationships:
        start_id = rel['startNode']
        end_id = rel['endNode']
        if start_id in visible_node_ids and end_id in visible_node_ids:
            node_degrees[start_id] = node_degrees.get(start_id, 0) + 1
            node_degrees[end_id] = node_degrees.get(end_id, 0) + 1
    
    # Update node degrees
    for node_id, node_data in nodes.items():
        node_data['degree'] = node_degrees.get(node_id, 0)
    
    # Filter relationships to only include those between visible nodes
    filtered_relationships = [
        rel for rel in relationships 
        if rel['startNode'] in visible_node_ids and rel['endNode'] in visible_node_ids
    ]
    
    # Enhanced statistics
    node_type_stats = {}
    relationship_type_stats = {}
    
    for node in nodes.values():
        labels = node.get('labels', ['Unknown'])
        primary_label = labels[0] if labels else 'Unknown'
        node_type_stats[primary_label] = node_type_stats.get(primary_label, 0) + 1
    
    for rel in filtered_relationships:
        rel_type = rel.get('type', 'Unknown')
        relationship_type_stats[rel_type] = relationship_type_stats.get(rel_type, 0) + 1
    
    logger.info(f"✅ Graph extraction complete: {len(nodes)} nodes, {len(filtered_relationships)} relationships")
    
    return {
        'nodes': list(nodes.values()),
        'relationships': filtered_relationships,
        'total_nodes': len(nodes),
        'total_relationships': len(filtered_relationships),
        'limited': len(nodes) >= effective_limit if effective_limit != float('inf') else False,
        'node_type_stats': node_type_stats,
        'relationship_type_stats': relationship_type_stats,
        'max_degree': max(node_degrees.values()) if node_degrees else 0,
        'avg_degree': sum(node_degrees.values()) / len(node_degrees) if node_degrees else 0
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
        "summary": f"🕐 {timestamp} | ⚡ {round(execution_time * 1000, 2)}ms | {' | '.join(changes)}"
    }

# Database helper functions
async def _read(tx: AsyncTransaction, query: str, params: dict):
    res = await tx.run(query, params)
    records = await res.to_eager_result()
    return json.dumps([r.data() for r in records.records], default=str)

async def _read_for_viz(tx: AsyncTransaction, query: str, params: dict):
    """Read query specifically for graph visualization"""
    res = await tx.run(query, params)
    records = await res.to_eager_result()
    return records.records  # Return raw records for graph extraction

async def _write(tx: AsyncTransaction, query: str, params: dict):
    return await tx.run(query, params)

# =============================================================================
# MCP-STYLE TOOL FUNCTIONS
# =============================================================================

async def tool_read_neo4j_cypher(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """MCP tool: Execute a read-only Cypher query against Neo4j database"""
    try:
        query = arguments.get("query", "")
        params = arguments.get("params", {})
        node_limit = arguments.get("node_limit", 5000)
        
        if not query:
            raise ValueError("Query parameter is required")
        
        start_time = datetime.now()
        
        # Support unlimited display
        if node_limit == 0 or node_limit == -1:
            node_limit = None
        
        logger.info(f"🔍 Executing read query with limit: {'unlimited' if node_limit is None else node_limit}")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_read(_read, query, params)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        result_data = json.loads(result)
        
        # Extract graph data for visualization
        graph_data = None
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                viz_result = await session.execute_read(_read_for_viz, query, params)
            graph_data = extract_graph_data_unlimited(viz_result, node_limit)
        except Exception as e:
            logger.warning(f"Could not extract graph data for visualization: {e}")
        
        # Check if we have graph data
        has_graph_data = graph_data and (graph_data.get('nodes') or graph_data.get('relationships'))
        
        return {
            "data": result_data,
            "metadata": {
                "timestamp": start_time.isoformat(),
                "execution_time_ms": round(execution_time * 1000, 2),
                "query": query,
                "record_count": len(result_data),
                "has_graph_data": has_graph_data,
                "node_limit": "unlimited" if node_limit is None else node_limit,
                "unlimited_mode": node_limit is None
            },
            "graph_data": graph_data if has_graph_data else None
        }
        
    except Exception as e:
        logger.error(f"Error in read_neo4j_cypher: {e}")
        raise Exception(f"Neo4j read query failed: {str(e)}")

async def tool_write_neo4j_cypher(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """MCP tool: Execute a write Cypher query against Neo4j database"""
    try:
        query = arguments.get("query", "")
        params = arguments.get("params", {})
        node_limit = arguments.get("node_limit", 5000)
        
        if not query:
            raise ValueError("Query parameter is required")
        
        start_time = datetime.now()
        
        logger.info(f"✏️ Executing write query: {query[:100]}...")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_write(_write, query, params)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Format detailed change information
        change_info = format_change_summary(result._summary.counters, query, execution_time)
        
        # Log the change
        logger.info(f"Neo4j Write Operation: {change_info['summary']}")
        
        response = {
            "result": "SUCCESS",
            "change_info": change_info
        }
        
        # If the write operation returns data (like MERGE or CREATE with RETURN), 
        # try to get visualization data
        graph_data = None
        try:
            if "RETURN" in query.upper():
                async with driver.session(database=NEO4J_DATABASE) as session:
                    viz_result = await session.execute_read(_read_for_viz, query, params)
                graph_data = extract_graph_data_unlimited(viz_result, node_limit)
                response["graph_data"] = graph_data
        except Exception:
            response["graph_data"] = None
        
        return response
        
    except Exception as e:
        logger.error(f"Error in write_neo4j_cypher: {e}")
        raise Exception(f"Neo4j write query failed: {str(e)}")

async def tool_get_neo4j_schema(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """MCP tool: Get the Neo4j database schema"""
    get_schema_query = "CALL apoc.meta.schema();"
    try:
        start_time = datetime.now()
        
        logger.info("📋 Retrieving Neo4j database schema...")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_read(_read, get_schema_query, {})
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        schema = json.loads(result)[0].get('value')
        
        return {
            "schema": schema,
            "metadata": {
                "timestamp": start_time.isoformat(),
                "execution_time_ms": round(execution_time * 1000, 2)
            },
            "graph_data": None
        }
    except Exception as e:
        logger.error(f"Error in get_neo4j_schema: {e}")
        raise Exception(f"Neo4j schema query failed: {str(e)}")

async def tool_get_graph_stats(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """MCP tool: Get comprehensive graph statistics"""
    stats_queries = [
        ("total_nodes", "MATCH (n) RETURN count(n) as count"),
        ("total_relationships", "MATCH ()-[r]->() RETURN count(r) as count"),
        ("node_labels", "CALL db.labels() YIELD label RETURN collect(label) as labels"),
        ("relationship_types", "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"),
        ("node_label_counts", "MATCH (n) RETURN labels(n)[0] as label, count(*) as count ORDER BY count DESC"),
        ("relationship_type_counts", "MATCH ()-[r]->() RETURN type(r) as type, count(*) as count ORDER BY count DESC"),
        ("connected_components", "MATCH (n) WHERE (n)--() RETURN count(DISTINCT n) as connected_nodes"),
        ("isolated_nodes", "MATCH (n) WHERE NOT (n)--() RETURN count(n) as isolated"),
    ]
    
    stats = {}
    
    try:
        logger.info("📊 Collecting comprehensive graph statistics...")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            for stat_name, query in stats_queries:
                try:
                    result = await session.execute_read(_read, query, {})
                    data = json.loads(result)
                    stats[stat_name] = data
                except Exception as e:
                    logger.warning(f"Could not get {stat_name}: {e}")
                    stats[stat_name] = []
        
        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat(),
            "unlimited_support": True
        }
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}")
        raise Exception(f"Graph statistics query failed: {str(e)}")

async def tool_get_sample_graph(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """MCP tool: Get a sample of the graph for visualization"""
    node_limit = arguments.get("node_limit", 200)
    
    # Support unlimited sampling
    if node_limit == 0 or node_limit == -1:
        node_limit = None
    
    if node_limit is None:
        query = """
        MATCH (n) 
        OPTIONAL MATCH (n)-[r]-(m) 
        RETURN n, r, m
        """
    else:
        query = f"""
        MATCH (n)-[r]->(m)
        RETURN n, r, m
        LIMIT {min(node_limit, 10000)}
        """
    
    try:
        logger.info(f"🎲 Getting sample graph with limit: {'unlimited' if node_limit is None else node_limit}")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_read(_read_for_viz, query, {})
        
        graph_data = extract_graph_data_unlimited(result, node_limit)
        
        return {
            "graph_data": graph_data,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "node_limit": "unlimited" if node_limit is None else node_limit
        }
    except Exception as e:
        logger.error(f"Error getting sample graph: {e}")
        raise Exception(f"Sample graph query failed: {str(e)}")

# MCP tool registry
MCP_TOOLS = {
    "read_neo4j_cypher": {
        "function": tool_read_neo4j_cypher,
        "description": "Execute read-only Cypher queries",
        "parameters": ["query", "params", "node_limit"]
    },
    "write_neo4j_cypher": {
        "function": tool_write_neo4j_cypher,
        "description": "Execute write Cypher queries",
        "parameters": ["query", "params", "node_limit"]
    },
    "get_neo4j_schema": {
        "function": tool_get_neo4j_schema,
        "description": "Get database schema information",
        "parameters": []
    },
    "get_graph_stats": {
        "function": tool_get_graph_stats,
        "description": "Get comprehensive graph statistics",
        "parameters": []
    },
    "get_sample_graph": {
        "function": tool_get_sample_graph,
        "description": "Get sample graph for visualization",
        "parameters": ["node_limit"]
    }
}

# =============================================================================
# FASTAPI ENDPOINTS
# =============================================================================

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Neo4j Graph Explorer MCP Server",
        "version": "3.0.0",
        "features": ["mcp_tools", "unlimited_display", "enhanced_performance"],
        "timestamp": datetime.now().isoformat(),
        "tools_available": len(MCP_TOOLS)
    }

@app.get("/tools")
async def list_tools():
    """List all available MCP tools"""
    return {
        "tools": [
            {
                "name": name,
                "description": info["description"],
                "parameters": info["parameters"]
            }
            for name, info in MCP_TOOLS.items()
        ],
        "total_tools": len(MCP_TOOLS),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/tools/call")
async def call_tool(request: MCPToolRequest):
    """Call an MCP tool by name"""
    try:
        if request.tool not in MCP_TOOLS:
            raise HTTPException(status_code=404, detail=f"Tool '{request.tool}' not found")
        
        tool_info = MCP_TOOLS[request.tool]
        tool_function = tool_info["function"]
        
        logger.info(f"🛠️ Calling tool: {request.tool}")
        
        result = await tool_function(request.arguments)
        
        return MCPToolResponse(
            tool=request.tool,
            result=result,
            success=True,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Tool '{request.tool}' failed: {e}")
        return MCPToolResponse(
            tool=request.tool,
            result={},
            success=False,
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

# Direct HTTP endpoints for backward compatibility
@app.post("/read_neo4j_cypher", response_model=CypherResponse)
async def http_read_neo4j_cypher(request: CypherRequest):
    """HTTP endpoint for read queries"""
    try:
        result = await tool_read_neo4j_cypher({
            "query": request.query,
            "params": request.params,
            "node_limit": request.node_limit
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/write_neo4j_cypher")
async def http_write_neo4j_cypher(request: CypherRequest):
    """HTTP endpoint for write queries"""
    try:
        result = await tool_write_neo4j_cypher({
            "query": request.query,
            "params": request.params,
            "node_limit": request.node_limit
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_neo4j_schema")
async def http_get_neo4j_schema():
    """HTTP endpoint for schema queries"""
    try:
        result = await tool_get_neo4j_schema({})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph_stats")
async def http_get_graph_stats():
    """HTTP endpoint for graph statistics"""
    try:
        result = await tool_get_graph_stats({})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sample_graph")
async def http_get_sample_graph(node_limit: int = 200):
    """HTTP endpoint for sample graph"""
    try:
        result = await tool_get_sample_graph({"node_limit": node_limit})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Neo4j Graph Explorer MCP Server...")
    logger.info("🛠️ MCP Tools: read_neo4j_cypher, write_neo4j_cypher, get_neo4j_schema, get_graph_stats, get_sample_graph")
    logger.info("🌐 HTTP API: Available at /docs for OpenAPI documentation")
    logger.info("🔧 MCP Tools: Available at /tools for tool listing")
    
    uvicorn.run(
        "mcpserver:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
