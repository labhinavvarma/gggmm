import streamlit as st
import requests
import uuid
import json
from datetime import datetime
import streamlit.components.v1 as components

# Page configuration with wide layout and collapsed sidebar
st.set_page_config(
    page_title="Neo4j Graph Explorer", 
    page_icon="🕸️", 
    layout="wide",
    initial_sidebar_state="collapsed"  # Sidebar collapsed by default
)

# Custom CSS for better styling and split layout
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .chat-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        height: calc(100vh - 200px);
        overflow-y: auto;
    }
    
    .viz-container {
        background: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        height: calc(100vh - 200px);
        border: 2px solid #e0e0e0;
    }
    
    .chat-message {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .bot-message {
        background: #f0f2f6;
        border-left: 4px solid #764ba2;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "change_history" not in st.session_state:
    st.session_state.change_history = []
if "current_graph_data" not in st.session_state:
    st.session_state.current_graph_data = None
if "node_limit" not in st.session_state:
    st.session_state.node_limit = 5000

def create_enhanced_graph_visualization(graph_data, height=600, node_limit=5000):
    """Create an enhanced interactive Neo4j graph visualization with 5000 node support"""
    if not graph_data or not graph_data.get('nodes'):
        return None
    
    # Limit nodes if too many
    nodes = graph_data.get('nodes', [])
    relationships = graph_data.get('relationships', [])
    
    if len(nodes) > node_limit:
        nodes = nodes[:node_limit]
        st.warning(f"⚠️ Limiting display to {node_limit} nodes (total: {len(graph_data.get('nodes', []))})")
    
    # Enhanced color scheme for different node labels
    label_colors = {
        'Person': '#FF6B6B',
        'User': '#FF6B6B',
        'Employee': '#FF8E53',
        'Customer': '#FF6B9D',
        'Company': '#4ECDC4', 
        'Organization': '#4ECDC4',
        'Project': '#45B7D1',
        'Task': '#45B7D1',
        'Department': '#96CEB4',
        'Team': '#96CEB4',
        'Location': '#FECA57',
        'City': '#FECA57',
        'Country': '#FECA57',
        'Product': '#FF9FF3',
        'Service': '#FF9FF3',
        'Order': '#54A0FF',
        'Transaction': '#54A0FF',
        'Category': '#5F27CD',
        'Tag': '#5F27CD',
        'Event': '#00D2D3',
        'Meeting': '#00D2D3',
        'Document': '#FD79A8',
        'File': '#FD79A8',
        'Message': '#A29BFE',
        'Post': '#A29BFE'
    }
    
    # Build nodes for visualization with enhanced properties
    vis_nodes = []
    node_stats = {}
    
    for i, node in enumerate(nodes):
        primary_label = node['labels'][0] if node['labels'] else 'Node'
        color = label_colors.get(primary_label, '#95A5A6')
        
        # Track node statistics
        node_stats[primary_label] = node_stats.get(primary_label, 0) + 1
        
        # Create enhanced display label
        display_props = node.get('properties', {})
        display_label = (
            display_props.get('name') or 
            display_props.get('title') or 
            display_props.get('label') or 
            display_props.get('id') or 
            f"{primary_label}_{i}"
        )
        
        # Calculate node size based on relationships
        rel_count = len([r for r in relationships if r['startNode'] == node['id'] or r['endNode'] == node['id']])
        node_size = min(50, max(15, 15 + rel_count * 2))  # Size based on connections
        
        vis_nodes.append({
            'id': node['id'],
            'label': str(display_label)[:25],  # Limit label length
            'color': {'background': color, 'border': '#2c3e50'},
            'size': node_size,
            'font': {'color': 'white', 'size': 12, 'face': 'Arial'},
            'neo4j_labels': node['labels'],
            'neo4j_properties': display_props,
            'title': f"{primary_label}: {display_label}\\nConnections: {rel_count}"  # Tooltip
        })
    
    # Build edges for visualization with enhanced styling
    vis_edges = []
    edge_stats = {}
    
    # Filter relationships to only include those between visible nodes
    visible_node_ids = {node['id'] for node in nodes}
    filtered_relationships = [
        rel for rel in relationships 
        if rel['startNode'] in visible_node_ids and rel['endNode'] in visible_node_ids
    ]
    
    for rel in filtered_relationships:
        rel_type = rel['type']
        edge_stats[rel_type] = edge_stats.get(rel_type, 0) + 1
        
        # Enhanced edge styling
        edge_colors = {
            'KNOWS': '#3498db',
            'WORKS_FOR': '#e74c3c', 
            'MANAGES': '#f39c12',
            'BELONGS_TO': '#9b59b6',
            'LOCATED_IN': '#2ecc71',
            'PART_OF': '#34495e',
            'RELATED_TO': '#95a5a6',
            'CONTAINS': '#e67e22',
            'USES': '#1abc9c'
        }
        
        edge_color = edge_colors.get(rel_type, '#7f8c8d')
        
        vis_edges.append({
            'id': rel['id'],
            'from': rel['startNode'],
            'to': rel['endNode'],
            'label': rel_type,
            'arrows': {'to': {'enabled': True, 'scaleFactor': 1}},
            'color': {'color': edge_color, 'highlight': '#ff0000'},
            'width': 2,
            'font': {'color': '#2c3e50', 'size': 10, 'face': 'Arial'},
            'neo4j_type': rel_type,
            'neo4j_properties': rel.get('properties', {}),
            'title': f"Relationship: {rel_type}"
        })
    
    # Create comprehensive statistics
    total_nodes = len(vis_nodes)
    total_edges = len(vis_edges)
    
    # Create the enhanced HTML with vis.js
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            #neo4j-graph {{
                width: 100%;
                height: {height}px;
                border: 2px solid #ddd;
                border-radius: 12px;
                background: radial-gradient(circle, #f8f9fa 0%, #e9ecef 100%);
                position: relative;
            }}
            
            .graph-controls {{
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(255,255,255,0.95);
                padding: 12px;
                border-radius: 8px;
                font-size: 12px;
                z-index: 1000;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                max-width: 250px;
            }}
            
            .graph-stats {{
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(255,255,255,0.95);
                padding: 12px;
                border-radius: 8px;
                font-size: 12px;
                z-index: 1000;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            
            .control-button {{
                background: #667eea;
                color: white;
                border: none;
                padding: 8px 12px;
                margin: 2px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 11px;
            }}
            
            .control-button:hover {{
                background: #764ba2;
            }}
            
            .legend {{
                position: absolute;
                bottom: 10px;
                left: 10px;
                background: rgba(255,255,255,0.95);
                padding: 12px;
                border-radius: 8px;
                font-size: 11px;
                z-index: 1000;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                max-height: 200px;
                overflow-y: auto;
                max-width: 250px;
            }}
            
            .legend-item {{
                display: flex;
                align-items: center;
                margin: 3px 0;
            }}
            
            .legend-color {{
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
                border: 1px solid #333;
            }}
        </style>
    </head>
    <body>
        <!-- Graph Controls -->
        <div class="graph-controls">
            <div><strong>🎛️ Graph Controls</strong></div>
            <button class="control-button" onclick="network.fit()">🔍 Fit View</button>
            <button class="control-button" onclick="togglePhysics()">⚡ Toggle Physics</button>
            <button class="control-button" onclick="resetLayout()">🔄 Reset Layout</button>
            <button class="control-button" onclick="exportGraph()">📁 Export</button>
        </div>
        
        <!-- Graph Statistics -->
        <div class="graph-stats">
            <div><strong>📊 Graph Statistics</strong></div>
            <div>🔵 Nodes: {total_nodes}</div>
            <div>🔗 Edges: {total_edges}</div>
            <div>📏 Density: {(total_edges / max(1, total_nodes * (total_nodes - 1) / 2) * 100):.1f}%</div>
        </div>
        
        <!-- Node Type Legend -->
        <div class="legend">
            <div><strong>🏷️ Node Types</strong></div>
            {''.join([f'<div class="legend-item"><div class="legend-color" style="background-color: {label_colors.get(label, "#95A5A6")}"></div>{label} ({count})</div>' for label, count in node_stats.items()])}
        </div>
        
        <div id="neo4j-graph"></div>
        
        <script type="text/javascript">
            // Graph data
            var nodes = new vis.DataSet({json.dumps(vis_nodes)});
            var edges = new vis.DataSet({json.dumps(vis_edges)});
            
            // Create network
            var container = document.getElementById('neo4j-graph');
            var data = {{
                nodes: nodes,
                edges: edges
            }};
            
            var options = {{
                nodes: {{
                    shape: 'dot',
                    font: {{
                        size: 12,
                        color: 'white',
                        face: 'Arial'
                    }},
                    borderWidth: 2,
                    shadow: {{
                        enabled: true,
                        color: 'rgba(0,0,0,0.3)',
                        size: 3,
                        x: 2,
                        y: 2
                    }},
                    chosen: {{
                        node: function(values, id, selected, hovering) {{
                            values.shadow = true;
                            values.shadowColor = 'rgba(0,0,0,0.5)';
                            values.shadowSize = 5;
                        }}
                    }}
                }},
                edges: {{
                    width: 2,
                    shadow: {{
                        enabled: true,
                        color: 'rgba(0,0,0,0.2)',
                        size: 2
                    }},
                    smooth: {{
                        type: 'continuous',
                        forceDirection: 'none',
                        roundness: 0.3
                    }},
                    font: {{
                        size: 10,
                        face: 'Arial',
                        align: 'middle'
                    }},
                    chosen: {{
                        edge: function(values, id, selected, hovering) {{
                            values.width = 4;
                            values.color = '#ff0000';
                        }}
                    }}
                }},
                physics: {{
                    enabled: true,
                    stabilization: {{
                        enabled: true,
                        iterations: 100,
                        updateInterval: 25
                    }},
                    barnesHut: {{
                        gravitationalConstant: -5000,
                        centralGravity: 0.3,
                        springLength: 200,
                        springConstant: 0.05,
                        damping: 0.09,
                        avoidOverlap: 0.2
                    }},
                    maxVelocity: 50,
                    minVelocity: 0.1,
                    solver: 'barnesHut',
                    timestep: 0.35,
                    adaptiveTimestep: true
                }},
                interaction: {{
                    hover: true,
                    tooltipDelay: 200,
                    hideEdgesOnDrag: false,
                    hideEdgesOnZoom: false
                }},
                layout: {{
                    improvedLayout: true,
                    clusterThreshold: 150
                }}
            }};
            
            var network = new vis.Network(container, data, options);
            var physicsEnabled = true;
            
            // Control functions
            function togglePhysics() {{
                physicsEnabled = !physicsEnabled;
                network.setOptions({{physics: {{enabled: physicsEnabled}}}});
            }}
            
            function resetLayout() {{
                network.setOptions({{physics: {{enabled: true}}}});
                network.stabilize();
            }}
            
            function exportGraph() {{
                var nodePositions = network.getPositions();
                var exportData = {{
                    nodes: nodes.get(),
                    edges: edges.get(),
                    positions: nodePositions
                }};
                console.log('Graph data:', exportData);
                alert('Graph data exported to console. Check browser developer tools.');
            }}
            
            // Enhanced click events
            network.on("click", function (params) {{
                if (params.nodes.length > 0) {{
                    var nodeId = params.nodes[0];
                    var node = nodes.get(nodeId);
                    var details = '🔵 Node Details\\n\\n' +
                                'Label: ' + node.label + '\\n' +
                                'Type: ' + node.neo4j_labels.join(', ') + '\\n' +
                                'Properties: ' + JSON.stringify(node.neo4j_properties, null, 2);
                    alert(details);
                }}
                if (params.edges.length > 0) {{
                    var edgeId = params.edges[0];
                    var edge = edges.get(edgeId);
                    var details = '🔗 Relationship Details\\n\\n' +
                                'Type: ' + edge.neo4j_type + '\\n' +
                                'Properties: ' + JSON.stringify(edge.neo4j_properties, null, 2);
                    alert(details);
                }}
            }});
            
            // Hover effects
            network.on("hoverNode", function (params) {{
                container.style.cursor = 'pointer';
            }});
            
            network.on("blurNode", function (params) {{
                container.style.cursor = 'default';
            }});
            
            // Stabilization progress
            network.on("stabilizationProgress", function(params) {{
                var progress = params.iterations / params.total;
                console.log('Stabilization progress: ' + Math.round(progress * 100) + '%');
            }});
            
            network.on("stabilizationIterationsDone", function() {{
                console.log('Graph stabilization complete');
            }});
        </script>
    </body>
    </html>
    """
    
    return html_content

# Collapsed sidebar with essential controls
with st.sidebar:
    st.header("🎛️ Graph Controls")
    
    # Node limit control
    st.session_state.node_limit = st.slider(
        "Max Nodes to Display", 
        min_value=100, 
        max_value=10000, 
        value=st.session_state.node_limit,
        step=500
    )
    
    if st.button("🎯 Sample Graph", use_container_width=True):
        try:
            sample_result = requests.get("http://localhost:8000/sample_graph", timeout=10)
            if sample_result.status_code == 200:
                sample_data = sample_result.json()
                if sample_data.get('graph_data'):
                    st.session_state.current_graph_data = sample_data['graph_data']
                    st.success("Sample loaded!")
                    st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    if st.button("🗑️ Clear Graph", use_container_width=True):
        st.session_state.current_graph_data = None
        st.rerun()
    
    st.markdown("---")
    st.header("📊 Quick Stats")
    
    # Connection status
    try:
        health_check = requests.get("http://localhost:8081/health", timeout=2)
        if health_check.status_code == 200:
            st.success("🟢 System Online")
        else:
            st.error("🔴 System Issues")
    except:
        st.error("🔴 System Offline")
    
    # Display current graph stats
    if st.session_state.current_graph_data:
        nodes = st.session_state.current_graph_data.get('nodes', [])
        rels = st.session_state.current_graph_data.get('relationships', [])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nodes", len(nodes))
        with col2:
            st.metric("Edges", len(rels))

# Main header
st.markdown("""
<div class="main-header">
    <h1>🕸️ Neo4j Graph Explorer</h1>
    <p>AI-Powered Graph Database Interface with Interactive Visualization</p>
</div>
""", unsafe_allow_html=True)

# Split screen layout: Chat on left, Visualization on right
left_col, right_col = st.columns([1, 1])

# LEFT COLUMN: Chat Interface
with left_col:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Quick action buttons
    st.markdown("### 🚀 Quick Actions")
    
    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("📊 Schema", use_container_width=True):
            st.session_state.quick_query = "Show me the database schema"
        if st.button("👥 People", use_container_width=True):
            st.session_state.quick_query = "MATCH (p:Person) RETURN p LIMIT 50"
    
    with action_col2:
        if st.button("🔢 Count", use_container_width=True):
            st.session_state.quick_query = "How many nodes are in the graph?"
        if st.button("🔗 Network", use_container_width=True):
            st.session_state.quick_query = f"MATCH (a)-[r]->(b) RETURN a, r, b LIMIT {st.session_state.node_limit}"
    
    # Chat input form
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_area(
            "💬 Ask about your graph data:",
            height=100,
            placeholder="e.g., 'Show me all connected nodes' or 'Create a new person named Alice'"
        )
        
        submitted = st.form_submit_button("🚀 Send Query", use_container_width=True)
    
    # Handle quick query button clicks
    if 'quick_query' in st.session_state:
        user_query = st.session_state.quick_query
        submitted = True
        del st.session_state.quick_query
    
    # Display conversation history in chat format
    st.markdown("### 💬 Conversation")
    
    # Process new query
    if submitted and user_query:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_query,
            "timestamp": datetime.now().isoformat()
        })
        
        # Show loading spinner
        with st.spinner("🤔 Processing your query..."):
            try:
                session_id = str(uuid.uuid4())
                payload = {"question": user_query, "session_id": session_id}
                
                response = requests.post("http://localhost:8081/chat", json=payload, timeout=45)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Add bot response to history
                    st.session_state.messages.append({
                        "role": "bot",
                        "content": result['answer'],
                        "tool": result['tool'],
                        "query": result['query'],
                        "graph_data": result.get('graph_data'),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Update current graph data if available
                    if result.get('graph_data'):
                        st.session_state.current_graph_data = result['graph_data']
                    
                else:
                    st.session_state.messages.append({
                        "role": "bot",
                        "content": f"❌ Error {response.status_code}: {response.text}",
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except Exception as e:
                st.session_state.messages.append({
                    "role": "bot",
                    "content": f"❌ Error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
        
        st.rerun()
    
    # Display messages in reverse order (newest first)
    for msg in reversed(st.session_state.messages[-10:]):  # Show last 10 messages
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-message">
                <strong>🧑 You:</strong><br>
                {msg["content"]}
                <small style="color: #666;">⏰ {msg["timestamp"][:19]}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            tool_info = ""
            if msg.get("tool"):
                tool_info = f"<br><small>🔧 Tool: {msg['tool']}</small>"
            if msg.get("query"):
                tool_info += f"<br><small>📝 Query: <code>{msg['query'][:100]}...</code></small>"
            
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Assistant:</strong><br>
                {msg["content"]}{tool_info}
                <small style="color: #666;">⏰ {msg["timestamp"][:19]}</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT COLUMN: Visualization Panel
with right_col:
    st.markdown('<div class="viz-container">', unsafe_allow_html=True)
    
    st.markdown("### 🕸️ Graph Visualization")
    
    # Display current graph or placeholder
    if st.session_state.current_graph_data:
        graph_data = st.session_state.current_graph_data
        nodes = graph_data.get('nodes', [])
        relationships = graph_data.get('relationships', [])
        
        if nodes:
            # Show graph stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔵 Nodes", len(nodes))
            with col2:
                st.metric("🔗 Relationships", len(relationships))
            with col3:
                density = len(relationships) / max(1, len(nodes) * (len(nodes) - 1) / 2) * 100
                st.metric("📏 Density", f"{density:.1f}%")
            
            # Create and display graph
            graph_html = create_enhanced_graph_visualization(
                graph_data, 
                height=500, 
                node_limit=st.session_state.node_limit
            )
            
            if graph_html:
                components.html(graph_html, height=520)
                
                # Additional graph information
                with st.expander("📋 Graph Details"):
                    # Node type breakdown
                    node_types = {}
                    for node in nodes:
                        for label in node.get('labels', ['Unknown']):
                            node_types[label] = node_types.get(label, 0) + 1
                    
                    st.write("**Node Types:**")
                    for label, count in sorted(node_types.items()):
                        st.write(f"• {label}: {count}")
                    
                    # Relationship type breakdown
                    if relationships:
                        rel_types = {}
                        for rel in relationships:
                            rel_type = rel.get('type', 'Unknown')
                            rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
                        
                        st.write("**Relationship Types:**")
                        for rel_type, count in sorted(rel_types.items()):
                            st.write(f"• {rel_type}: {count}")
            else:
                st.error("Could not create graph visualization")
        else:
            st.info("No nodes to display")
    else:
        # Placeholder when no graph data
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #666;">
            <h3>🎯 Ready for Graph Exploration</h3>
            <p>Ask a question or click a quick action to see your graph visualization here!</p>
            <p><strong>Try these examples:</strong></p>
            <ul style="text-align: left; display: inline-block;">
                <li>"Show me all Person nodes"</li>
                <li>"Display the network structure"</li>
                <li>"MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 100"</li>
                <li>"Create a person named John"</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer status bar
st.markdown("---")
status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    try:
        health = requests.get("http://localhost:8081/health", timeout=2)
        if health.status_code == 200:
            st.markdown("🟢 **Agent Online**")
        else:
            st.markdown("🔴 **Agent Issues**")
    except:
        st.markdown("🔴 **Agent Offline**")

with status_col2:
    try:
        mcp = requests.get("http://localhost:8000/", timeout=2)
        if mcp.status_code == 200:
            st.markdown("🟢 **Neo4j Connected**")
        else:
            st.markdown("🔴 **Neo4j Issues**")
    except:
        st.markdown("🔴 **Neo4j Offline**")

with status_col3:
    st.markdown(f"💬 **Messages: {len(st.session_state.messages)}**")

with status_col4:
    st.markdown(f"📊 **Node Limit: {st.session_state.node_limit}**")
