import asyncio
import json
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncTransaction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_neo4j_cypher_enhanced")

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_neo4j_password"
NEO4J_DATABASE = "neo4j"

driver: AsyncDriver = AsyncGraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
)

app = FastAPI(title="Enhanced Neo4j MCP Server - Unlimited Graph Display Support")

class CypherRequest(BaseModel):
    query: str
    params: dict = {}
    node_limit: int = 5000  # Default limit, can be set to None for unlimited

# Enhanced database configuration for unlimited display
UNLIMITED_DISPLAY_CONFIG = {
    "max_query_timeout": 300,  # 5 minutes for complex unlimited queries
    "memory_optimization": True,
    "streaming_results": True,
    "batch_processing": True,
    "neo4j_page_cache": "2G",  # Increase page cache for large graphs
    "neo4j_heap_size": "4G"    # Increase heap for complex queries
}

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
    """Enhanced health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Enhanced Neo4j MCP Server", 
        "features": ["unlimited_display", "enhanced_performance", "neo4j_browser_experience"],
        "max_nodes": "unlimited"
    }

@app.post("/read_neo4j_cypher")
async def read_neo4j_cypher_unlimited(request: CypherRequest):
    """Enhanced read endpoint with unlimited display support"""
    try:
        start_time = datetime.now()
        
        # Support unlimited display by accepting None as node_limit
        node_limit = request.node_limit
        if node_limit == 0 or node_limit == -1:  # Special values for unlimited
            node_limit = None
        
        logger.info(f"📊 Executing query with limit: {'unlimited' if node_limit is None else node_limit}")
        
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_read(_read_with_graph, request.query, request.params or {})
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        result_data = json.loads(result)
        
        # Extract graph data with unlimited capability
        graph_data = None
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                viz_result = await session.execute_read(_read_for_viz, request.query, request.params or {})
            
            # Use unlimited extraction function
            graph_data = extract_graph_data_unlimited(viz_result, node_limit)
            logger.info(f"🕸️ Graph extracted: {graph_data.get('total_nodes', 0)} nodes, {graph_data.get('total_relationships', 0)} relationships")
            
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
                "node_limit": "unlimited" if node_limit is None else node_limit,
                "unlimited_mode": node_limit is None
            },
            "graph_data": graph_data if has_graph_data else None
        }
            
        return response
        
    except Exception as e:
        logger.error(f"Error in read_neo4j_cypher_unlimited: {e}")
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
        # try to get visualization data with unlimited support
        graph_data = None
        try:
            if "RETURN" in request.query.upper():
                async with driver.session(database=NEO4J_DATABASE) as session:
                    viz_result = await session.execute_read(_read_for_viz, request.query, request.params or {})
                graph_data = extract_graph_data_unlimited(viz_result, node_limit)
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
async def get_sample_graph(node_limit: int = 200):
    """Get a sample of the graph for visualization with unlimited support"""
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph_stats")
async def get_graph_stats():
    """Get comprehensive graph statistics with enhanced metrics"""
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
        ("max_degree", "MATCH (n) OPTIONAL MATCH (n)-[r]-() RETURN max(count(r)) as max_degree")
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
            "unlimited_support": True
        }
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimize_query/{node_limit}")
async def optimize_query_for_limit(node_limit: int, query: str):
    """Optimize a query to respect node limits or enable unlimited mode"""
    optimized_query = query
    
    # Handle unlimited mode
    if node_limit == 0 or node_limit == -1:
        # Remove LIMIT clauses for unlimited display
        optimized_query = query.replace("LIMIT", "-- LIMIT REMOVED FOR UNLIMITED DISPLAY --")
        return {
            "original_query": query,
            "optimized_query": optimized_query,
            "node_limit": "unlimited",
            "mode": "unlimited"
        }
    
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
        "node_limit": node_limit,
        "mode": "limited"
    }

def get_unlimited_display_queries():
    """Pre-optimized queries for unlimited display scenarios"""
    
    return {
        "complete_graph": """
            MATCH (n) 
            OPTIONAL MATCH (n)-[r]-(m) 
            RETURN n, r, m
        """,
        
        "all_connections": """
            MATCH (a)-[r]->(b) 
            RETURN a, r, b 
            UNION 
            MATCH (isolated) 
            WHERE NOT (isolated)--() 
            RETURN isolated, null as r, null as b
        """,
        
        "complete_network_paths": """
            MATCH path = (a)-[*1..3]-(b) 
            WHERE a <> b 
            RETURN path
        """,
        
        "schema_visualization": """
            MATCH (n)-[r]-(m) 
            RETURN DISTINCT labels(n) as StartLabels, type(r) as RelType, labels(m) as EndLabels
            UNION
            MATCH (isolated) 
            WHERE NOT (isolated)--() 
            RETURN DISTINCT labels(isolated) as StartLabels, null as RelType, null as EndLabels
        """,
        
        "connected_components": """
            MATCH (n) 
            OPTIONAL MATCH path = (n)-[*1..10]-(connected) 
            RETURN n, collect(DISTINCT connected) as component
        """
    }

@app.get("/unlimited_queries")
async def get_unlimited_queries():
    """Get pre-optimized unlimited display queries"""
    return {
        "queries": get_unlimited_display_queries(),
        "description": "Pre-optimized queries for unlimited graph display",
        "features": ["complete_graph_traversal", "no_artificial_limits", "performance_optimized"]
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
    """Read query specifically for graph visualization with unlimited support"""
    res = await tx.run(query, params)
    records = await res.to_eager_result()
    return records.records  # Return raw records for graph extraction

async def _write(tx: AsyncTransaction, query: str, params: dict):
    return await tx.run(query, params)

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Enhanced Neo4j MCP Server with Unlimited Display Support...")
    logger.info("🕸️ Features: Unlimited graph display, enhanced performance, Neo4j Browser experience")
    uvicorn.run("mcpserver_enhanced:app", host="0.0.0.0", port=8000, reload=True)
