import asyncio
import json
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncTransaction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unlimited_mcp_neo4j")

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_neo4j_password"
NEO4J_DATABASE = "neo4j"

driver: AsyncDriver = AsyncGraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
)

app = FastAPI(title="Unlimited Neo4j MCP Server - No Artificial Limits")

class CypherRequest(BaseModel):
    query: str
    params: dict = {}
    node_limit: int = None  # None means unlimited

def extract_unlimited_graph_data(records):
    """
    Extract ALL nodes and relationships without any artificial limits
    This function processes the complete result set as specified by the query
    """
    nodes = {}
    relationships = []
    
    logger.info(f"🚀 Processing UNLIMITED graph data - extracting ALL results")
    
    # Process ALL records and extract ALL graph objects
    for record in records:
        for key, value in record.items():
            # Handle nodes
            if hasattr(value, 'labels'):  # It's a node
                node_id = str(value.element_id)
                
                # Extract all properties
                properties = dict(value)
                
                # Ensure we have a meaningful display name
                if not any(prop in properties for prop in ['name', 'title', 'displayName']):
                    labels = list(value.labels)
                    if labels:
                        # Create a meaningful name from available properties
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
                
                # Add connected nodes if not already present
                for node_ref, node_id_key in [(value.start_node, start_node_id), (value.end_node, end_node_id)]:
                    if node_id_key not in nodes:
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
                    if hasattr(item, 'labels'):  # Node in list
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
    
    # Calculate node degrees
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
    
    # Include ALL relationships between visible nodes
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
    
    logger.info(f"✅ UNLIMITED graph extraction complete: {len(nodes)} nodes, {len(filtered_relationships)} relationships")
    
    return {
        'nodes': list(nodes.values()),
        'relationships': filtered_relationships,
        'total_nodes': len(nodes),
        'total_relationships': len(filtered_relationships),
        'limited': False,  # Never limited in unlimited mode
        'node_type_stats': node_type_stats,
        'relationship_type_stats': relationship_type_stats,
        'max_degree': max(node_degrees.values()) if node_degrees else 0,
        'avg_degree': sum(node_degrees.values()) / len(node_degrees) if node_degrees else 0,
        'unlimited_mode': True
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
    return {
        "status": "healthy", 
        "service": "Unlimited Neo4j MCP Server", 
        "features": ["unlimited_display", "no_artificial_limits", "complete_graph_extraction"],
        "node_limits": "NONE - displays everything according to query"
    }

@app.post("/read_neo4j_cypher")
async def read_neo4j_cypher_unlimited(request: CypherRequest):
    """UNLIMITED read endpoint - processes ALL data according to the query"""
    try:
        start_time = datetime.now()
        
        logger.info(f"📊 Executing UNLIMITED query: {request.query[:100]}...")
        
        # Execute query without any artificial limits
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_read(_read_with_graph, request.query, request.params or {})
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        result_data = json.loads(result)
        
        # Extract ALL graph data without limits
        graph_data = None
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                viz_result = await session.execute_read(_read_for_viz, request.query, request.params or {})
            
            # Use unlimited extraction function
            graph_data = extract_unlimited_graph_data(viz_result)
            logger.info(f"🕸️ UNLIMITED graph extracted: {graph_data.get('total_nodes', 0)} nodes, {graph_data.get('total_relationships', 0)} relationships")
            
        except Exception as e:
            logger.warning(f"Could not extract graph data for visualization: {e}")
        
        # Check if we have graph data
        has_graph_data = graph_data and (graph_data.get('nodes') or graph_data.get('relationships'))
        
        response = {
            "data": result_data,
            "metadata": {
                "timestamp": start_time.isoformat(),
                "execution_time_ms": round(execution_time * 1000, 2),
                "query": request.query,
                "record_count": len(result_data),
                "has_graph_data": has_graph_data,
                "node_limit": "UNLIMITED",
                "unlimited_mode": True
            },
            "graph_data": graph_data if has_graph_data else None
        }
            
        return response
        
    except Exception as e:
        logger.error(f"Error in unlimited read_neo4j_cypher: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/write_neo4j_cypher")
async def write_neo4j_cypher_unlimited(request: CypherRequest):
    """UNLIMITED write endpoint - no restrictions on returned data"""
    try:
        start_time = datetime.now()
        
        logger.info(f"✏️ Executing UNLIMITED write query: {request.query[:100]}...")
        
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
        # extract ALL visualization data without limits
        graph_data = None
        try:
            if "RETURN" in request.query.upper():
                async with driver.session(database=NEO4J_DATABASE) as session:
                    viz_result = await session.execute_read(_read_for_viz, request.query, request.params or {})
                graph_data = extract_unlimited_graph_data(viz_result)
                response["graph_data"] = graph_data
        except Exception:
            response["graph_data"] = None
            
        return response
        
    except Exception as e:
        logger.error(f"Error in unlimited write_neo4j_cypher: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_neo4j_schema")
async def get_neo4j_schema():
    """Get complete Neo4j schema without restrictions"""
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
                "execution_time_ms": round(execution_time * 1000, 2),
                "unlimited_mode": True
            },
            "graph_data": None
        }
    except Exception as e:
        logger.error(f"Error in get_neo4j_schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sample_graph")
async def get_unlimited_sample_graph():
    """Get a complete sample of the graph without any artificial limits"""
    query = """
    MATCH (n) 
    OPTIONAL MATCH (n)-[r]-(m) 
    RETURN n, r, m
    """
    
    try:
        logger.info("🚀 Getting UNLIMITED sample graph...")
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_read(_read_for_viz, query, {})
        
        graph_data = extract_unlimited_graph_data(result)
        
        return {
            "graph_data": graph_data,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "node_limit": "UNLIMITED",
            "unlimited_mode": True
        }
    except Exception as e:
        logger.error(f"Error getting unlimited sample graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph_stats")
async def get_unlimited_graph_stats():
    """Get comprehensive graph statistics without any restrictions"""
    stats_queries = [
        ("total_nodes", "MATCH (n) RETURN count(n) as count"),
        ("total_relationships", "MATCH ()-[r]->() RETURN count(r) as count"),
        ("node_labels", "CALL db.labels() YIELD label RETURN collect(label) as labels"),
        ("relationship_types", "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"),
        ("node_label_counts", "MATCH (n) RETURN labels(n)[0] as label, count(*) as count ORDER BY count DESC"),
        ("relationship_type_counts", "MATCH ()-[r]->() RETURN type(r) as type, count(*) as count ORDER BY count DESC"),
        ("connected_components", "MATCH (n) WHERE (n)--() RETURN count(DISTINCT n) as connected_nodes"),
        ("isolated_nodes", "MATCH (n) WHERE NOT (n)--() RETURN count(n) as isolated"),
        ("avg_degree", "MATCH (n) OPTIONAL MATCH (n)-[r]-() RETURN avg(count(r)) as avg_degree"),
        ("max_degree", "MATCH (n) OPTIONAL MATCH (n)-[r]-() RETURN max(count(r)) as max_degree"),
        ("database_size", "MATCH (n) OPTIONAL MATCH (n)-[r]-() RETURN count(n) as nodes, count(r) as relationships")
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
            "timestamp": datetime.now().isoformat(),
            "unlimited_support": True,
            "artificial_limits": "NONE"
        }
    except Exception as e:
        logger.error(f"Error getting unlimited graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/unlimited_display_info")
async def get_unlimited_display_info():
    """Get information about unlimited display capabilities"""
    return {
        "unlimited_display": True,
        "artificial_limits": "NONE",
        "features": [
            "complete_graph_traversal",
            "no_node_limits",
            "no_relationship_limits", 
            "full_data_extraction",
            "unlimited_visualization",
            "command_based_display"
        ],
        "description": "This server processes ALL data according to your queries without any artificial limits",
        "performance_notes": [
            "Large datasets may take longer to process",
            "Browser may become slow with very large graphs (>10k nodes)",
            "Consider using specific filters in your queries for better performance"
        ],
        "example_unlimited_queries": {
            "all_nodes": "MATCH (n) RETURN n",
            "complete_graph": "MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m",
            "all_relationships": "MATCH (a)-[r]->(b) RETURN a, r, b",
            "specific_entity_complete": "MATCH (n:Person) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m"
        }
    }

async def _read(tx: AsyncTransaction, query: str, params: dict):
    """Execute read query and return ALL results"""
    res = await tx.run(query, params)
    records = await res.to_eager_result()
    return json.dumps([r.data() for r in records.records], default=str)

async def _read_with_graph(tx: AsyncTransaction, query: str, params: dict):
    """Execute read query that preserves ALL graph structure information"""
    res = await tx.run(query, params)
    records = await res.to_eager_result()
    return json.dumps([r.data() for r in records.records], default=str)

async def _read_for_viz(tx: AsyncTransaction, query: str, params: dict):
    """Execute read query specifically for UNLIMITED graph visualization"""
    res = await tx.run(query, params)
    records = await res.to_eager_result()
    return records.records  # Return ALL raw records for graph extraction

async def _write(tx: AsyncTransaction, query: str, params: dict):
    """Execute write query without restrictions"""
    return await tx.run(query, params)

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting UNLIMITED Neo4j MCP Server...")
    logger.info("🕸️ Features: UNLIMITED display, NO artificial limits, complete data extraction")
    logger.info("⚠️ Performance: Large datasets will be processed completely - may take time")
    uvicorn.run("unlimited_mcpserver:app", host="0.0.0.0", port=8000, reload=True)
