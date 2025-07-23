import asyncio
import json
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncTransaction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_neo4j_cypher")

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_neo4j_password"
NEO4J_DATABASE = "neo4j"

driver: AsyncDriver = AsyncGraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
)

app = FastAPI(title="MCP Neo4j Cypher API - Enhanced for Large Graphs")

class CypherRequest(BaseModel):
    query: str
    params: dict = {}
    node_limit: int = 5000  # Default limit for large graphs

def extract_graph_data_optimized(records, node_limit=5000):
    """Extract nodes and relationships from Neo4j records with optimization for large graphs"""
    nodes = {}
    relationships = []
    
    # Process records and extract graph objects
    for record in records:
        for key, value in record.items():
            # Handle nodes
            if hasattr(value, 'labels'):  # It's a node
                node_id = value.element_id
                if len(nodes) < node_limit:  # Respect node limit
                    nodes[node_id] = {
                        'id': node_id,
                        'labels': list(value.labels),
                        'properties': dict(value)
                    }
            
            # Handle relationships
            elif hasattr(value, 'type'):  # It's a relationship
                rel = {
                    'id': value.element_id,
                    'type': value.type,
                    'startNode': value.start_node.element_id,
                    'endNode': value.end_node.element_id,
                    'properties': dict(value)
                }
                relationships.append(rel)
                
                # Add connected nodes if not already present and within limit
                start_node_id = value.start_node.element_id
                if start_node_id not in nodes and len(nodes) < node_limit:
                    nodes[start_node_id] = {
                        'id': start_node_id,
                        'labels': list(value.start_node.labels),
                        'properties': dict(value.start_node)
                    }
                
                end_node_id = value.end_node.element_id
                if end_node_id not in nodes and len(nodes) < node_limit:
                    nodes[end_node_id] = {
                        'id': end_node_id,
                        'labels': list(value.end_node.labels),
                        'properties': dict(value.end_node)
                    }
            
            # Handle lists (might contain nodes/relationships)
            elif isinstance(value, list):
                for item in value:
                    if hasattr(item, 'labels') and len(nodes) < node_limit:  # Node in list
                        node_id = item.element_id
                        nodes[node_id] = {
                            'id': node_id,
                            'labels': list(item.labels),
                            'properties': dict(item)
                        }
                    elif hasattr(item, 'type'):  # Relationship in list
                        rel = {
                            'id': item.element_id,
                            'type': item.type,
                            'startNode': item.start_node.element_id,
                            'endNode': item.end_node.element_id,
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
    if counters.indexes_added > 0:
        changes.append(f"📊 {counters.indexes_added} index(es) added")
    if counters.indexes_removed > 0:
        changes.append(f"📊 {counters.indexes_removed} index(es) removed")
    if counters.constraints_added > 0:
        changes.append(f"🔒 {counters.constraints_added} constraint(s) added")
    if counters.constraints_removed > 0:
        changes.append(f"🔒 {counters.constraints_removed} constraint(s) removed")
    
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
            "indexes_added": counters.indexes_added,
            "indexes_removed": counters.indexes_removed,
            "constraints_added": counters.constraints_added,
            "constraints_removed": counters.constraints_removed
        }
    }

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Neo4j MCP Server", "max_nodes": 5000}

@app.post("/read_neo4j_cypher")
async def read_neo4j_cypher(request: CypherRequest):
    try:
        start_time = datetime.now()
        node_limit = request.node_limit or 5000
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_read(_read_with_graph, request.query, request.params or {})
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        result_data = json.loads(result)
        
        # Extract graph data for visualization with node limit
        graph_data = None
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                viz_result = await session.execute_read(_read_for_viz, request.query, request.params or {})
            graph_data = extract_graph_data_optimized(viz_result, node_limit)
        except Exception as e:
            logger.warning(f"Could not extract graph data for visualization: {e}")
        
        # Check if we have graph data (nodes/relationships)
        has_graph_data = graph_data and (graph_data.get('nodes') or graph_data.get('relationships'))
        
        response = {
            "data": result_data,
            "metadata": {
                "timestamp": start_time.isoformat(),
                "execution_time_ms": round(execution_time * 1000, 2),
                "query": request.query,
                "record_count": len(result_data),
                "has_graph_data": has_graph_data,
                "node_limit": node_limit
            },
            "graph_data": graph_data if has_graph_data else None
        }
            
        return response
        
    except Exception as e:
        logger.error(f"Error in read_neo4j_cypher: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/write_neo4j_cypher")
async def write_neo4j_cypher(request: CypherRequest):
    try:
        start_time = datetime.now()
        node_limit = request.node_limit or 5000
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_write(_write, request.query, request.params or {})
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Format detailed change information
        change_info = format_change_summary(result._summary.counters, request.query, execution_time)
        
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
            if "RETURN" in request.query.upper():
                async with driver.session(database=NEO4J_DATABASE) as session:
                    viz_result = await session.execute_read(_read_for_viz, request.query, request.params or {})
                graph_data = extract_graph_data_optimized(viz_result, node_limit)
                response["graph_data"] = graph_data
        except Exception:
            response["graph_data"] = None
            
        return response
        
    except Exception as e:
        logger.error(f"Error in write_neo4j_cypher: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_neo4j_schema")
async def get_neo4j_schema():
    get_schema_query = "CALL apoc.meta.schema();"
    try:
        start_time = datetime.now()
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sample_graph")
async def get_sample_graph(node_limit: int = 5000):
    """Get a sample of the graph for visualization with configurable limit"""
    query = f"""
    MATCH (n)-[r]->(m)
    RETURN n, r, m
    LIMIT {min(node_limit, 5000)}
    """
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_read(_read_for_viz, query, {})
        
        graph_data = extract_graph_data_optimized(result, node_limit)
        
        return {
            "graph_data": graph_data,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "node_limit": node_limit
        }
    except Exception as e:
        logger.error(f"Error getting sample graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph_stats")
async def get_graph_stats():
    """Get comprehensive graph statistics"""
    stats_queries = [
        ("total_nodes", "MATCH (n) RETURN count(n) as count"),
        ("total_relationships", "MATCH ()-[r]->() RETURN count(r) as count"),
        ("node_labels", "CALL db.labels() YIELD label RETURN collect(label) as labels"),
        ("relationship_types", "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"),
        ("node_label_counts", "MATCH (n) RETURN labels(n)[0] as label, count(*) as count ORDER BY count DESC"),
        ("relationship_type_counts", "MATCH ()-[r]->() RETURN type(r) as type, count(*) as count ORDER BY count DESC")
    ]
    
    stats = {}
    
    try:
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
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimize_query/{node_limit}")
async def optimize_query_for_limit(node_limit: int, query: str):
    """Optimize a query to respect node limits"""
    optimized_query = query
    
    # Add LIMIT if not present and query is a MATCH
    if "MATCH" in query.upper() and "LIMIT" not in query.upper():
        if "RETURN" in query.upper():
            # Insert LIMIT before the last part
            parts = query.rsplit("RETURN", 1)
            if len(parts) == 2:
                optimized_query = f"{parts[0]}RETURN {parts[1]} LIMIT {node_limit}"
        else:
            optimized_query = f"{query} LIMIT {node_limit}"
    
    return {
        "original_query": query,
        "optimized_query": optimized_query,
        "node_limit": node_limit
    }

async def _read(tx: AsyncTransaction, query: str, params: dict):
    res = await tx.run(query, params)
    records = await res.to_eager_result()
    return json.dumps([r.data() for r in records.records], default=str)

async def _read_with_graph(tx: AsyncTransaction, query: str, params: dict):
    """Read query that preserves graph structure information"""
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcpserver:app", host="0.0.0.0", port=8000, reload=True)
