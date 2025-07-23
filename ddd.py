import streamlit as st
import requests
import uuid
import json
from datetime import datetime
import tempfile
import os
from pyvis.network import Network

# Page configuration with wide layout and collapsed sidebar
st.set_page_config(
    page_title="Neo4j Graph Explorer", 
    page_icon="🕸️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean split-screen layout
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
        height: 85vh;
        overflow-y: auto;
        border: 1px solid #dee2e6;
    }
    
    .graph-container {
        background: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        height: 85vh;
        border: 1px solid #dee2e6;
    }
    
    .chat-message {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .bot-message {
        background: #f0f2f6;
        border-left: 4px solid #764ba2;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        margin: 0.25rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    
    /* Improve split layout */
    .element-container {
        margin: 0 !important;
    }
    
    /* Graph container improvements */
    .graph-stats {
        background: #e3f2fd;
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #bbdefb;
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
    st.session_state.node_limit = 1000

def create_pyvis_graph(graph_data, height="700px"):
    """Create a clean Pyvis graph visualization"""
    if not graph_data or not graph_data.get('nodes'):
        return None
    
    # Create network with clean settings
    net = Network(
        height=height, 
        width="100%", 
        notebook=False, 
        bgcolor="#f8f9fa", 
        font_color="#2c3e50"
    )
    
    # Color scheme for different node types
    node_colors = {
        'Person': '#e74c3c',
        'User': '#e74c3c',
        'Employee': '#e67e22',
        'Customer': '#e91e63',
        'Company': '#2ecc71',
        'Organization': '#27ae60',
        'Project': '#3498db',
        'Task': '#5dade2',
        'Department': '#1abc9c',
        'Team': '#16a085',
        'Location': '#f39c12',
        'City': '#f1c40f',
        'Product': '#9b59b6',
        'Service': '#8e44ad',
        'Order': '#34495e',
        'Category': '#7f8c8d'
    }
    
    # Add nodes
    nodes = graph_data.get('nodes', [])
    for node in nodes:
        node_id = node['id']
        labels = node.get('labels', ['Unknown'])
        properties = node.get('properties', {})
        
        # Determine primary label and color
        primary_label = labels[0] if labels else 'Node'
        color = node_colors.get(primary_label, '#95a5a6')
        
        # Create display label
        display_name = (
            properties.get('name') or 
            properties.get('title') or 
            properties.get('label') or 
            f"{primary_label}"
        )
        
        # Create tooltip with properties
        tooltip_parts = [f"Type: {primary_label}"]
        for key, value in properties.items():
            if key not in ['id'] and value is not None:
                tooltip_parts.append(f"{key}: {str(value)[:50]}")
        tooltip = "\\n".join(tooltip_parts)
        
        # Add node to network
        net.add_node(
            node_id,
            label=str(display_name)[:20],  # Limit label length
            title=tooltip,
            color=color,
            size=20,
            font={'size': 12, 'color': '#2c3e50'},
            shape="dot"
        )
    
    # Add relationships
    relationships = graph_data.get('relationships', [])
    for rel in relationships:
        start_node = rel.get('startNode')
        end_node = rel.get('endNode')
        rel_type = rel.get('type', 'RELATED')
        
        if start_node and end_node:
            # Create tooltip for relationship
            rel_props = rel.get('properties', {})
            rel_tooltip = f"Type: {rel_type}"
            if rel_props:
                rel_tooltip += "\\n" + "\\n".join([f"{k}: {v}" for k, v in rel_props.items()])
            
            net.add_edge(
                start_node, 
                end_node, 
                label=rel_type,
                title=rel_tooltip,
                color={'color': '#7f8c8d'},
                width=2
            )
    
    # Configure physics for better layout
    net.repulsion(
        node_distance=150,
        central_gravity=0.15,
        spring_length=250,
        spring_strength=0.05,
        damping=0.2
    )
    
    # Set physics options for cleaner layout
    net.set_options('''
    var options = {
      "edges": {
        "color": {"inherit": false},
        "smooth": {"type": "continuous"}
      },
      "nodes": {
        "shape": "dot",
        "size": 20,
        "font": {"size": 12},
        "borderWidth": 2,
        "shadow": true
      },
      "physics": {
        "repulsion": {
          "centralGravity": 0.15,
          "springLength": 250,
          "springConstant": 0.05,
          "nodeDistance": 150,
          "damping": 0.2
        },
        "minVelocity": 0.75,
        "solver": "repulsion"
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200
      }
    }
    ''')
    
    return net

# Collapsed sidebar with essential controls
with st.sidebar:
    st.header("🎛️ Graph Controls")
    
    # Node limit control
    st.session_state.node_limit = st.slider(
        "Max Nodes to Display", 
        min_value=50, 
        max_value=2000, 
        value=st.session_state.node_limit,
        step=50
    )
    
    if st.button("🎯 Sample Graph", use_container_width=True):
        try:
            sample_result = requests.get(f"http://localhost:8000/sample_graph?node_limit={st.session_state.node_limit}", timeout=10)
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
        
        st.metric("Nodes", len(nodes))
        st.metric("Edges", len(rels))

# Main header
st.markdown("""
<div class="main-header">
    <h1>🕸️ Neo4j Graph Explorer</h1>
    <p>AI-Powered Graph Database Interface with Clean Visualization</p>
</div>
""", unsafe_allow_html=True)

# Split screen layout: Chat on left, Visualization on right (50-50 split)
left_col, right_col = st.columns([1, 1], gap="medium")

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
            st.session_state.quick_query = f"MATCH (p:Person) RETURN p LIMIT {min(st.session_state.node_limit, 100)}"
    
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
    
    # Display conversation history
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
                payload = {
                    "question": user_query, 
                    "session_id": session_id,
                    "node_limit": st.session_state.node_limit
                }
                
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
    for msg in reversed(st.session_state.messages[-8:]):  # Show last 8 messages
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
                tool_info += f"<br><small>📝 Query: <code>{msg['query'][:80]}...</code></small>"
            
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Assistant:</strong><br>
                {msg["content"]}{tool_info}
                <small style="color: #666;">⏰ {msg["timestamp"][:19]}</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT COLUMN: Graph Visualization
with right_col:
    st.markdown('<div class="graph-container">', unsafe_allow_html=True)
    
    st.markdown("### 🕸️ Graph Visualization")
    
    # Display current graph or placeholder
    if st.session_state.current_graph_data:
        graph_data = st.session_state.current_graph_data
        nodes = graph_data.get('nodes', [])
        relationships = graph_data.get('relationships', [])
        
        if nodes:
            # Show graph stats
            st.markdown(f"""
            <div class="graph-stats">
                <strong>📊 Graph Statistics:</strong> 
                🔵 {len(nodes)} nodes | 🔗 {len(relationships)} relationships
                {' | ⚠️ Limited view' if len(nodes) >= st.session_state.node_limit else ''}
            </div>
            """, unsafe_allow_html=True)
            
            # Create and display Pyvis graph
            try:
                net = create_pyvis_graph(graph_data, height="650px")
                
                if net:
                    # Save graph to temporary file
                    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as tmp_file:
                        net.save_graph(tmp_file.name)
                        tmp_path = tmp_file.name
                    
                    # Display the graph
                    try:
                        with open(tmp_path, 'r') as f:
                            graph_html = f.read()
                        st.components.v1.html(graph_html, height=670, scrolling=False)
                    finally:
                        # Clean up temporary file
                        try:
                            os.remove(tmp_path)
                        except:
                            pass
                    
                    # Additional graph information
                    with st.expander("📋 Graph Details"):
                        # Node type breakdown
                        node_types = {}
                        for node in nodes:
                            for label in node.get('labels', ['Unknown']):
                                node_types[label] = node_types.get(label, 0) + 1
                        
                        if node_types:
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
                    
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                st.info("Try reducing the node limit or simplifying your query")
        else:
            st.info("No nodes to display")
    else:
        # Placeholder when no graph data
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: #666; border: 2px dashed #ddd; border-radius: 10px; background: #f9f9f9;">
            <h3>🎯 Ready for Graph Exploration</h3>
            <p>Ask a question or click a quick action to see your graph visualization here!</p>
            <br>
            <p><strong>Try these examples:</strong></p>
            <div style="text-align: left; display: inline-block;">
                • "Show me all Person nodes"<br>
                • "Display the network structure"<br>
                • "MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 50"<br>
                • "Create a person named John"
            </div>
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
