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

app = FastAPI(title="MCP Neo4j Cypher API")

class CypherRequest(BaseModel):
    query: str
    params: dict = {}

def extract_graph_data(records):
    """Extract nodes and relationships from Neo4j records for visualization"""
    nodes = {}
    relationships = []
    
    for record in records:
        for key, value in record.items():
            if hasattr(value, 'labels'):  # It's a node
                node_id = value.element_id
                nodes[node_id] = {
                    'id': node_id,
                    'labels': list(value.labels),
                    'properties': dict(value)
                }
            elif hasattr(value, 'type'):  # It's a relationship
                rel = {
                    'id': value.element_id,
                    'type': value.type,
                    'startNode': value.start_node.element_id,
                    'endNode': value.end_node.element_id,
                    'properties': dict(value)
                }
                relationships.append(rel)
                
                # Also add the connected nodes if not already present
                start_node_id = value.start_node.element_id
                if start_node_id not in nodes:
                    nodes[start_node_id] = {
                        'id': start_node_id,
                        'labels': list(value.start_node.labels),
                        'properties': dict(value.start_node)
                    }
                
                end_node_id = value.end_node.element_id
                if end_node_id not in nodes:
                    nodes[end_node_id] = {
                        'id': end_node_id,
                        'labels': list(value.end_node.labels),
                        'properties': dict(value.end_node)
                    }
    
    return {
        'nodes': list(nodes.values()),
        'relationships': relationships
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

@app.post("/read_neo4j_cypher")
async def read_neo4j_cypher(request: CypherRequest):
    try:
        start_time = datetime.now()
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_read(_read_with_graph, request.query, request.params or {})
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        result_data = json.loads(result)
        
        # Check if we have graph data (nodes/relationships)
        has_graph_data = any(
            hasattr(record.get(key), 'labels') or hasattr(record.get(key), 'type')
            for record in result_data
            for key in record.keys()
            if record.get(key) is not None
        )
        
        response = {
            "data": result_data,
            "metadata": {
                "timestamp": start_time.isoformat(),
                "execution_time_ms": round(execution_time * 1000, 2),
                "query": request.query,
                "record_count": len(result_data),
                "has_graph_data": has_graph_data
            }
        }
        
        # Add graph visualization data if applicable
        if has_graph_data and result_data:
            try:
                # Re-run query to get graph objects for visualization
                async with driver.session(database=NEO4J_DATABASE) as session:
                    viz_result = await session.execute_read(_read_for_viz, request.query, request.params or {})
                response["graph_data"] = viz_result
            except Exception as e:
                logger.warning(f"Could not extract graph data for visualization: {e}")
                response["graph_data"] = None
        else:
            response["graph_data"] = None
            
        return response
        
    except Exception as e:
        logger.error(f"Error in read_neo4j_cypher: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/write_neo4j_cypher")
async def write_neo4j_cypher(request: CypherRequest):
    try:
        start_time = datetime.now()
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
        try:
            # Check if query has RETURN clause
            if "RETURN" in request.query.upper():
                async with driver.session(database=NEO4J_DATABASE) as session:
                    viz_result = await session.execute_read(_read_for_viz, request.query, request.params or {})
                response["graph_data"] = viz_result
            else:
                response["graph_data"] = None
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
async def get_sample_graph():
    """Get a sample of the graph for visualization"""
    query = """
    MATCH (n)-[r]->(m)
    RETURN n, r, m
    LIMIT 50
    """
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.execute_read(_read_for_viz, query, {})
        return {
            "graph_data": result,
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting sample graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    
    # Extract graph data for visualization
    return extract_graph_data(records.records)

async def _write(tx: AsyncTransaction, query: str, params: dict):
    return await tx.run(query, params)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcpserver:app", host="0.0.0.0", port=8000, reload=True)
