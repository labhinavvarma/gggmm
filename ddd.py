"""
Integrated Neo4j Chatbot with Real-time Graph Visualization
This version provides a native Streamlit chat interface with live graph updates
"""

import streamlit as st
import requests
import uuid
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import networkx as nx
from streamlit.runtime.caching import cache_data

# Configuration
ENHANCED_SERVER_PORT = 8000

# Page configuration
st.set_page_config(
    page_title="Neo4j Live Chatbot with Graph Visualization",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    
    .graph-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 2px solid #e0e0e0;
    }
    
    .stats-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    
    .update-indicator {
        background: #28a745;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None
if "database_stats" not in st.session_state:
    st.session_state.database_stats = {}
if "initial_load_done" not in st.session_state:
    st.session_state.initial_load_done = False
if "last_update_time" not in st.session_state:
    st.session_state.last_update_time = None

# ============================================
# HELPER FUNCTIONS
# ============================================

def check_server_health():
    """Check health of the server"""
    try:
        response = requests.get(f"http://localhost:{ENHANCED_SERVER_PORT}/health", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "data": response.json()}
        else:
            return {"status": "error", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "disconnected", "error": f"Server not running on port {ENHANCED_SERVER_PORT}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def get_database_stats():
    """Get current database statistics"""
    try:
        response = requests.get(f"http://localhost:{ENHANCED_SERVER_PORT}/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def get_complete_graph_data():
    """Get complete graph data - loads ALL data from Neo4j"""
    try:
        # First get database stats to determine how much data we have
        stats = get_database_stats()
        if "error" in stats:
            return {"error": stats["error"]}
        
        total_nodes = stats.get("nodes", 0)
        
        # Load all data (or limit to reasonable amount for visualization)
        limit = min(total_nodes, 1000)  # Limit to 1000 nodes for performance
        
        response = requests.get(f"http://localhost:{ENHANCED_SERVER_PORT}/graph?limit={limit}", timeout=30)
        if response.status_code == 200:
            graph_data = response.json()
            # Add metadata about the load
            graph_data["loaded_at"] = datetime.now().isoformat()
            graph_data["total_available"] = total_nodes
            graph_data["loaded_count"] = len(graph_data.get("nodes", []))
            return graph_data
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def send_chat_message(question: str):
    """Send chat message to server and get response with graph data"""
    try:
        payload = {
            "question": question,
            "session_id": st.session_state.session_id
        }
        
        start_time = time.time()
        
        response = requests.post(
            f"http://localhost:{ENHANCED_SERVER_PORT}/chat",
            json=payload,
            timeout=60  # Longer timeout for complex operations
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            result["response_time"] = response_time
            return result
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "answer": f"❌ Server error: {response.status_code}",
                "response_time": response_time
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "answer": f"❌ Request failed: {str(e)}",
            "response_time": 0
        }

def create_interactive_graph(graph_data, title="Neo4j Graph Visualization"):
    """Create an interactive network graph using Plotly with enhanced styling"""
    if not graph_data or "error" in graph_data:
        return None
    
    nodes = graph_data.get("nodes", [])
    relationships = graph_data.get("relationships", [])
    
    if not nodes:
        return None
    
    try:
        # Create network graph with NetworkX
        G = nx.Graph()
        
        # Prepare node data
        node_info = {}
        node_colors = {}
        node_sizes = {}
        
        for node in nodes:
            node_id = str(node["id"])
            labels = node.get("labels", [])
            properties = node.get("properties", {})
            
            G.add_node(node_id)
            
            # Create display info
            display_name = properties.get("name") or properties.get("title") or f"Node {node_id}"
            
            # Create hover info
            hover_text = f"<b>{display_name}</b><br>"
            if labels:
                hover_text += f"Labels: {', '.join(labels)}<br>"
            
            # Add key properties to hover
            for key, value in list(properties.items())[:5]:  # Show first 5 properties
                if key not in ['name', 'title']:
                    hover_text += f"{key}: {value}<br>"
            
            node_info[node_id] = {
                "display_name": display_name,
                "hover_text": hover_text,
                "labels": labels,
                "properties": properties
            }
            
            # Color by primary label
            primary_label = labels[0] if labels else "Unknown"
            node_colors[node_id] = primary_label
            
            # Size by number of properties (or default)
            node_sizes[node_id] = max(20, min(50, len(properties) * 5))
        
        # Add edges
        edge_info = []
        for rel in relationships:
            source = str(rel["source"])
            target = str(rel["target"])
            rel_type = rel.get("type", "CONNECTED")
            
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target)
                edge_info.append({
                    "source": source,
                    "target": target,
                    "type": rel_type,
                    "properties": rel.get("properties", {})
                })
        
        # Get layout
        if len(G.nodes) > 100:
            # Use faster layout for large graphs
            pos = nx.spring_layout(G, k=1, iterations=20)
        else:
            # Use better layout for smaller graphs
            pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edge traces
        edge_traces = []
        
        # Group edges by type for different colors
        edge_types = {}
        for edge_data in edge_info:
            edge_type = edge_data["type"]
            if edge_type not in edge_types:
                edge_types[edge_type] = []
            edge_types[edge_type].append(edge_data)
        
        # Create trace for each edge type
        colors = px.colors.qualitative.Set3
        for i, (edge_type, edges) in enumerate(edge_types.items()):
            edge_x = []
            edge_y = []
            edge_hover = []
            
            for edge_data in edges:
                source = edge_data["source"]
                target = edge_data["target"]
                
                if source in pos and target in pos:
                    x0, y0 = pos[source]
                    x1, y1 = pos[target]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    
                    # Hover info for edge
                    source_name = node_info[source]["display_name"]
                    target_name = node_info[target]["display_name"]
                    edge_hover.append(f"{source_name} → {target_name}<br>Type: {edge_type}")
            
            if edge_x:  # Only create trace if we have edges
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=2, color=colors[i % len(colors)]),
                    hoverinfo='none',
                    mode='lines',
                    name=f"{edge_type} ({len(edges)})",
                    showlegend=True
                )
                edge_traces.append(edge_trace)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_hover = []
        node_color_values = []
        node_size_values = []
        
        # Get unique labels for color mapping
        unique_labels = list(set(node_colors.values()))
        color_map = {label: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)] 
                    for i, label in enumerate(unique_labels)}
        
        for node_id in G.nodes():
            if node_id in pos:
                x, y = pos[node_id]
                node_x.append(x)
                node_y.append(y)
                
                info = node_info[node_id]
                node_text.append(info["display_name"])
                node_hover.append(info["hover_text"])
                
                primary_label = node_colors[node_id]
                node_color_values.append(color_map.get(primary_label, "#888888"))
                node_size_values.append(node_sizes[node_id])
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=node_hover,
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            marker=dict(
                size=node_size_values,
                color=node_color_values,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            name="Nodes",
            showlegend=False
        )
        
        # Create the figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            title={
                'text': f'{title}<br><sub>Nodes: {len(nodes)} | Relationships: {len(relationships)} | Types: {len(edge_types)}</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=80),
            annotations=[
                dict(
                    text=f"Last updated: {datetime.now().strftime('%H:%M:%S')}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(color="#888", size=10)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating graph: {e}")
        return None

def refresh_graph_data():
    """Refresh the graph data and update session state"""
    with st.spinner("🔄 Refreshing graph data..."):
        new_graph_data = get_complete_graph_data()
        if "error" not in new_graph_data:
            st.session_state.graph_data = new_graph_data
            st.session_state.last_update_time = datetime.now()
            st.session_state.database_stats = get_database_stats()
            return True
        else:
            st.error(f"Failed to refresh graph: {new_graph_data['error']}")
            return False

# ============================================
# MAIN APPLICATION
# ============================================

# Header
st.markdown("""
<div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 1rem;">
    <h1>🧠 Neo4j Live Chatbot with Real-time Graph Visualization</h1>
    <p>Chat with your Neo4j database and see changes instantly in the graph visualization</p>
</div>
""", unsafe_allow_html=True)

# Check server status
server_health = check_server_health()

if server_health["status"] != "healthy":
    st.error(f"❌ Server not available: {server_health['error']}")
    st.info("Please start the enhanced server: `python enhanced_fastmcp_server.py`")
    st.stop()

# Create main layout
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("## 💬 Neo4j Chat Interface")
    
    # Initial load of graph data
    if not st.session_state.initial_load_done:
        with st.spinner("🚀 Loading complete Neo4j graph data..."):
            initial_graph_data = get_complete_graph_data()
            if "error" not in initial_graph_data:
                st.session_state.graph_data = initial_graph_data
                st.session_state.database_stats = get_database_stats()
                st.session_state.initial_load_done = True
                st.session_state.last_update_time = datetime.now()
                
                # Show initial load success
                loaded_count = initial_graph_data.get("loaded_count", 0)
                total_available = initial_graph_data.get("total_available", 0)
                st.success(f"✅ Loaded {loaded_count} nodes from Neo4j database (Total: {total_available})")
            else:
                st.error(f"❌ Failed to load initial graph data: {initial_graph_data['error']}")
    
    # Display current database stats
    if st.session_state.database_stats and "error" not in st.session_state.database_stats:
        stats = st.session_state.database_stats
        
        st.markdown("""
        <div class="stats-container">
            <h4>📊 Database Statistics</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Nodes", stats.get("nodes", 0))
        with col_b:
            st.metric("Relationships", stats.get("relationships", 0))
        with col_c:
            st.metric("Labels", len(stats.get("labels", [])))
        with col_d:
            st.metric("Types", len(stats.get("relationship_types", [])))
    
    # Manual refresh button
    if st.button("🔄 Refresh Graph Data", use_container_width=True):
        if refresh_graph_data():
            st.success("✅ Graph data refreshed!")
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                # Display assistant response
                result = message["content"]
                
                if result.get("success", True):
                    # Show tool and query info
                    if result.get("tool"):
                        st.info(f"🔧 **Tool:** {result['tool']}")
                    
                    if result.get("query"):
                        st.code(result["query"], language="cypher")
                    
                    # Show the answer
                    if result.get("answer"):
                        st.markdown(result["answer"])
                    
                    # Show performance info
                    if result.get("response_time"):
                        st.caption(f"⏱️ Processed in {result['response_time']:.2f}s")
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

    # Chat input
    if prompt := st.chat_input("Ask about your Neo4j database (e.g., 'How many nodes are there?', 'Create a Person named Alice')"):
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Processing your request..."):
                result = send_chat_message(prompt)
            
            # Add assistant message to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": result,
                "timestamp": datetime.now().isoformat()
            })
            
            # Display response
            if result.get("success", True):
                # Show tool and query info
                if result.get("tool"):
                    st.info(f"🔧 **Tool:** {result['tool']}")
                
                if result.get("query"):
                    st.code(result["query"], language="cypher")
                
                # Show the answer
                if result.get("answer"):
                    st.markdown(result["answer"])
                
                # Show performance info
                if result.get("response_time"):
                    st.caption(f"⏱️ Processed in {result['response_time']:.2f}s")
                
                # If this was a write operation, refresh the graph
                if result.get("tool") == "write_neo4j_cypher":
                    with st.spinner("🔄 Updating graph visualization..."):
                        if refresh_graph_data():
                            st.success("✅ Graph visualization updated!")
                            st.rerun()
            else:
                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

with col2:
    st.markdown("## 🎨 Live Graph Visualization")
    
    # Display last update time
    if st.session_state.last_update_time:
        st.caption(f"Last updated: {st.session_state.last_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display graph
    if st.session_state.graph_data and "error" not in st.session_state.graph_data:
        graph_data = st.session_state.graph_data
        
        # Create and display the interactive graph
        fig = create_interactive_graph(
            graph_data, 
            title="Live Neo4j Graph Visualization"
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="main_graph")
            
            # Show graph information
            nodes = graph_data.get("nodes", [])
            relationships = graph_data.get("relationships", [])
            loaded_count = graph_data.get("loaded_count", len(nodes))
            total_available = graph_data.get("total_available", loaded_count)
            
            st.info(f"📊 Showing {loaded_count} of {total_available} total nodes")
            
            if loaded_count < total_available:
                st.warning(f"⚠️ Displaying limited data for performance. Total nodes in database: {total_available}")
        else:
            st.warning("⚠️ Could not generate graph visualization")
    
    else:
        if st.session_state.graph_data and "error" in st.session_state.graph_data:
            st.error(f"❌ Graph data error: {st.session_state.graph_data['error']}")
        else:
            st.info("📊 Loading graph data...")

# Sidebar with quick actions and examples
with st.sidebar:
    st.markdown("## 🚀 Quick Actions")
    
    # Example queries
    st.markdown("### 💡 Try These Queries:")
    
    example_queries = [
        "How many nodes are in the graph?",
        "Show me the database schema",
        "Create a Person named Alice with age 30",
        "Create a Company called TechCorp",
        "Connect Alice to TechCorp as an employee",
        "List all Person nodes",
        "Find nodes with the most connections",
        "Delete all nodes labeled TestNode",
        "Create 5 connected Person nodes",
        "Show me all relationship types"
    ]
    
    for query in example_queries:
        if st.button(query, key=f"example_{hash(query)}", use_container_width=True):
            # Add to chat
            st.session_state.messages.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            })
            
            # Process the query
            with st.spinner("🧠 Processing example query..."):
                result = send_chat_message(query)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": result,
                "timestamp": datetime.now().isoformat()
            })
            
            # Refresh graph if it was a write operation
            if result.get("tool") == "write_neo4j_cypher":
                refresh_graph_data()
            
            st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Manual graph refresh
    if st.button("🔄 Force Graph Refresh", use_container_width=True):
        if refresh_graph_data():
            st.success("✅ Graph refreshed!")
            st.rerun()
    
    # Session info
    st.markdown("---")
    st.markdown("### 📋 Session Info")
    st.text(f"ID: {st.session_state.session_id[:8]}...")
    st.text(f"Messages: {len(st.session_state.messages)}")
    if st.session_state.database_stats:
        stats = st.session_state.database_stats
        if "error" not in stats:
            st.text(f"DB Nodes: {stats.get('nodes', 0)}")
            st.text(f"DB Relationships: {stats.get('relationships', 0)}")

# Auto-refresh mechanism (optional)
if st.checkbox("🔄 Auto-refresh every 30 seconds", value=False):
    # This would auto-refresh the graph data
    time.sleep(1)  # Small delay
    if int(time.time()) % 30 == 0:  # Every 30 seconds
        if refresh_graph_data():
            st.rerun()
