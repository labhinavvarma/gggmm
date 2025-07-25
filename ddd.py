# =============================================================================
# BACKEND ENHANCEMENTS FOR UNLIMITED GRAPH DISPLAY
# =============================================================================

# 1. UPDATE mcpserver.py - Enhanced graph extraction for unlimited display
# =============================================================================

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

# 2. UPDATE mcpserver.py - Enhanced read endpoint
# =============================================================================

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

# 3. UPDATE langgraph_agent.py - Enhanced system message for unlimited display
# =============================================================================

def create_unlimited_display_system_message() -> str:
    """Enhanced system message for unlimited graph display"""
    
    schema_info = schema_manager.get_schema_for_query_generation()
    schema_summary = schema_manager.schema_summary or "Schema summary not available"
    
    system_message = f"""You are a Neo4j database expert with UNLIMITED GRAPH DISPLAY capability and complete schema knowledge.

🎯 **ACTUAL DATABASE SCHEMA:**
{schema_info}

**UNLIMITED DISPLAY MODE:**
✅ NO artificial node limits - show complete graphs
✅ Use unlimited queries for comprehensive visualization  
✅ Generate queries that reveal the entire graph structure
✅ Focus on complete network topology and connectivity

**ENHANCED QUERY EXAMPLES FOR UNLIMITED DISPLAY:**

User: "Show me the entire graph structure"
Tool: read_neo4j_cypher
Query: MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m

User: "Display complete database schema"
Tool: get_neo4j_schema

User: "Show all nodes and their connections"
Tool: read_neo4j_cypher
Query: MATCH (n)-[r]-(m) RETURN n, r, m UNION MATCH (isolated) WHERE NOT (isolated)--() RETURN isolated, null, null

User: "Find the complete network structure"
Tool: read_neo4j_cypher
Query: MATCH path = (a)-[*1..3]-(b) RETURN path

User: "Show me all connected components"
Tool: read_neo4j_cypher
Query: MATCH (n) OPTIONAL MATCH (n)-[r*1..10]-(connected) RETURN n, collect(DISTINCT connected) as component

User: "Display the entire relationship network"
Tool: read_neo4j_cypher
Query: MATCH (a)-[r]->(b) RETURN a, r, b

**UNLIMITED DISPLAY RULES:**
- Remove all LIMIT clauses unless specifically requested
- Use comprehensive MATCH patterns that capture entire graph
- Include isolated nodes in results
- Generate queries that show complete connectivity
- Focus on revealing the full graph topology

**CURRENT SCHEMA:**
{schema_summary}

Generate queries that show the COMPLETE graph structure without artificial limitations."""
    
    return system_message

# 4. UPDATE app.py - Enhanced chat endpoint for unlimited support
# =============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat_unlimited(request: ChatRequest):
    """Enhanced chat endpoint with unlimited display support"""
    if agent is None:
        logger.error("Agent not initialized")
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    session_id = request.session_id or str(uuid.uuid4())
    
    # Support unlimited display
    node_limit = request.node_limit
    if node_limit >= 50000:  # Treat very high limits as unlimited
        node_limit = None
        unlimited_mode = True
    else:
        unlimited_mode = False
    
    logger.info(f"🧠 Processing {'unlimited' if unlimited_mode else 'limited'} chat request...")
    logger.info(f"📊 Question: {request.question[:100]} (Limit: {'unlimited' if unlimited_mode else node_limit})")
    
    start_time = datetime.now()
    
    try:
        # Create appropriate state for unlimited display
        if SCHEMA_AWARE:
            state = SchemaAwareAgentState(
                question=request.question,
                session_id=session_id,
                node_limit=node_limit if not unlimited_mode else 100000  # Use high number for unlimited
            )
        else:
            state = AgentState(
                question=request.question,
                session_id=session_id,
                node_limit=node_limit if not unlimited_mode else 100000
            )
        
        # Run the agent with unlimited capability
        logger.info(f"🔄 Running agent with {'unlimited' if unlimited_mode else 'limited'} display mode...")
        result = await agent.ainvoke(state)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        logger.info(f"✅ Agent completed - Tool: {result.get('tool')}")
        logger.info(f"📈 Execution time: {execution_time:.2f}ms")
        
        # Enhanced graph data handling for unlimited display
        has_graph_data = result.get('graph_data') and result.get('graph_data', {}).get('nodes')
        if has_graph_data:
            node_count = len(result['graph_data']['nodes'])
            rel_count = len(result['graph_data'].get('relationships', []))
            logger.info(f"🕸️ {'Unlimited' if unlimited_mode else 'Limited'} graph data: {node_count} nodes, {rel_count} relationships")
        
        # Enhanced response
        response = ChatResponse(
            trace=result.get("trace", ""),
            tool=result.get("tool", ""),
            query=result.get("query", ""),
            answer=result.get("answer", ""),
            graph_data=result.get("graph_data") if result.get("graph_data") else None,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            node_limit=request.node_limit,  # Return original request limit
            execution_time_ms=execution_time,
            success=True,
            schema_aware=SCHEMA_AWARE
        )
        
        return response
        
    except Exception as e:
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        logger.error(f"❌ Chat request failed: {str(e)}")
        
        error_response = ChatResponse(
            trace=f"Error in {'unlimited' if unlimited_mode else 'limited'} mode: {str(e)}",
            tool="",
            query="",
            answer=f"❌ Error in {'unlimited' if unlimited_mode else 'limited'} display mode: {str(e)}",
            graph_data=None,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            node_limit=request.node_limit,
            execution_time_ms=execution_time,
            success=False,
            error=str(e),
            schema_aware=SCHEMA_AWARE
        )
        
        return error_response

# 5. ENHANCED QUERY OPTIMIZATION FOR UNLIMITED DISPLAY
# =============================================================================

def optimize_query_for_unlimited_visualization(query: str, unlimited_mode: bool = False) -> str:
    """Enhanced query optimization for unlimited display"""
    query = query.strip()
    
    if unlimited_mode:
        # Remove artificial limits for unlimited display
        query = re.sub(r'\s+LIMIT\s+\d+', '', query, flags=re.IGNORECASE)
        
        # Enhance queries for better unlimited visualization
        if "MATCH (n)" in query.upper() and "RETURN n" in query.upper():
            # Add relationship context for better visualization
            if "OPTIONAL MATCH" not in query.upper():
                query = query.replace("RETURN n", "OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m")
        
        logger.info(f"🚀 Optimized query for unlimited display: {query[:100]}...")
        
    else:
        # Apply reasonable limits for standard display
        if ("MATCH" in query.upper() and 
            "LIMIT" not in query.upper() and 
            "count(" not in query.lower() and
            "COUNT(" not in query):
            
            if "RETURN" in query.upper():
                query += " LIMIT 50"
    
    return query

# 6. PERFORMANCE OPTIMIZATIONS FOR UNLIMITED DISPLAY
# =============================================================================

# Enhanced database configuration for unlimited display
UNLIMITED_DISPLAY_CONFIG = {
    "max_query_timeout": 300,  # 5 minutes for complex unlimited queries
    "memory_optimization": True,
    "streaming_results": True,
    "batch_processing": True,
    "neo4j_page_cache": "2G",  # Increase page cache for large graphs
    "neo4j_heap_size": "4G"    # Increase heap for complex queries
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

# 7. USAGE INSTRUCTIONS
# =============================================================================

UNLIMITED_DISPLAY_INTEGRATION = """
🚀 UNLIMITED DISPLAY INTEGRATION STEPS:

1. UPDATE mcpserver.py:
   - Replace extract_graph_data_optimized() with extract_graph_data_unlimited()
   - Update read_neo4j_cypher endpoint to read_neo4j_cypher_unlimited()

2. UPDATE langgraph_agent.py:
   - Add create_unlimited_display_system_message() function
   - Update optimize_query_for_visualization() with unlimited support
   - Modify cortex_llm() to use unlimited system message when needed

3. UPDATE app.py:
   - Replace chat endpoint with chat_unlimited()
   - Add unlimited mode detection (node_limit >= 50000 = unlimited)
   - Enhanced error handling for unlimited queries

4. REPLACE ui.py:
   - Use the Neo4j Browser-like UI provided above
   - Includes unlimited display controls and Neo4j styling

5. RESTART SERVICES:
   - Restart mcpserver.py (port 8000)
   - Restart app.py (port 8020)  
   - Restart ui.py with Streamlit

EXPECTED RESULTS:
✅ Complete graph visualization without artificial limits
✅ Neo4j Browser-like appearance and functionality
✅ Schema-aware intelligent query generation
✅ Enhanced stability and performance for large graphs
✅ Real-time statistics and graph analytics
"""

print(UNLIMITED_DISPLAY_INTEGRATION)
