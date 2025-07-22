
"""
Fixed Neo4j Chatbot UI with Auto Graph Updates - NO NESTED EXPANDERS
This version automatically shows the Neo4j graph and updates it after each chat interaction
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

# Configuration
ENHANCED_SERVER_PORT = 8000

# Page configuration
st.set_page_config(
    page_title="Neo4j Live Chatbot with Auto Graph Updates",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .status-healthy {
        background: linear-gradient(90deg, #00d4aa 0%, #00b894 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin: 0.2rem;
        display: inline-block;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .status-error {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin: 0.2rem;
        display: inline-block;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .tool-badge {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .query-display {
        background: #1e1e1e;
        color: #f8f8f2;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        border-left: 4px solid #50fa7b;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .result-display {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00d4aa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .chat-message {
        background: white;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #00d4aa;
        border-radius: 50%;
        animation: pulse 2s infinite;
        margin-right: 0.5rem;
    }
    
    .update-indicator {
        background: #28a745;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        animation: pulse 2s infinite;
        margin: 0.5rem 0;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.2rem;
    }
    
    .graph-container {
        background: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 2px solid #e0e0e0;
        margin: 1rem 0;
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
if "initial_graph_loaded" not in st.session_state:
    st.session_state.initial_graph_loaded = False
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

def get_graph_data(limit: int = 100):
    """Get graph data for visualization"""
    try:
        response = requests.get(f"http://localhost:{ENHANCED_SERVER_PORT}/graph?limit={limit}", timeout=15)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def send_chat_message(question: str):
    """Send chat message to server"""
    try:
        payload = {
            "question": question,
            "session_id": st.session_state.session_id
        }
        
        start_time = time.time()
        
        response = requests.post(
            f"http://localhost:{ENHANCED_SERVER_PORT}/chat",
            json=payload,
            timeout=30
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
            
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to server",
            "answer": "❌ Server not running. Start the enhanced server on port 8000.",
            "response_time": 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "answer": f"❌ Request failed: {str(e)}",
            "response_time": 0
        }

def create_network_graph(graph_data, title="Neo4j Live Graph"):
    """Create an enhanced network graph using Plotly"""
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
            for key, value in list(properties.items())[:5]:
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
            
            # Size by number of properties
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
            pos = nx.spring_layout(G, k=1, iterations=20)
        else:
            pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edge traces
        edge_traces = []
        edge_types = {}
        for edge_data in edge_info:
            edge_type = edge_data["type"]
            if edge_type not in edge_types:
                edge_types[edge_type] = []
            edge_types[edge_type].append(edge_data)
        
        colors = px.colors.qualitative.Set3
        for i, (edge_type, edges) in enumerate(edge_types.items()):
            edge_x = []
            edge_y = []
            
            for edge_data in edges:
                source = edge_data["source"]
                target = edge_data["target"]
                
                if source in pos and target in pos:
                    x0, y0 = pos[source]
                    x1, y1 = pos[target]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
            
            if edge_x:
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
                'text': f'{title}<br><sub>Nodes: {len(nodes)} | Relationships: {len(relationships)} | Last Updated: {datetime.now().strftime("%H:%M:%S")}</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=80),
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

def auto_refresh_graph():
    """Auto refresh graph data and update session state"""
    try:
        new_graph_data = get_graph_data(100)
        if "error" not in new_graph_data:
            st.session_state.graph_data = new_graph_data
            st.session_state.last_update_time = datetime.now()
            
            # Update database stats
            new_stats = get_database_stats()
            if "error" not in new_stats:
                st.session_state.database_stats = new_stats
            
            return True
        return False
    except Exception as e:
        st.error(f"Failed to refresh graph: {e}")
        return False

# ============================================
# MAIN UI LAYOUT
# ============================================

# Header
st.markdown("""
<div class="header-container">
    <h1>🧠 Neo4j Live Chatbot with Auto Graph Updates</h1>
    <p>Chat with your Neo4j database and see changes instantly in the graph visualization</p>
    <p><strong>✅ FIXED VERSION - No Nested Expanders</strong></p>
</div>
""", unsafe_allow_html=True)

# Check server status first
server_health = check_server_health()

if server_health["status"] != "healthy":
    st.error(f"❌ Server not available: {server_health['error']}")
    st.info("Please start the enhanced server: `python enhanced_fastmcp_server.py`")
    st.stop()

# ============================================
# AUTO-LOAD GRAPH ON STARTUP
# ============================================

if not st.session_state.initial_graph_loaded:
    with st.spinner("🚀 Auto-loading Neo4j graph data..."):
        initial_graph_data = get_graph_data(100)
        if "error" not in initial_graph_data:
            st.session_state.graph_data = initial_graph_data
            st.session_state.initial_graph_loaded = True
            st.session_state.last_update_time = datetime.now()
            
            # Load database stats
            initial_stats = get_database_stats()
            if "error" not in initial_stats:
                st.session_state.database_stats = initial_stats
            
            st.success("✅ Graph data auto-loaded successfully!")
        else:
            st.error(f"❌ Failed to auto-load graph: {initial_graph_data['error']}")

# ============================================
# CREATE MAIN LAYOUT - TWO COLUMNS
# ============================================

col1, col2 = st.columns([1, 1.2])

# ============================================
# LEFT COLUMN - CHAT INTERFACE
# ============================================

with col1:
    st.markdown("## 💬 Neo4j Chat Interface")
    
    # Display current database stats
    if st.session_state.database_stats and "error" not in st.session_state.database_stats:
        stats = st.session_state.database_stats
        
        st.markdown("### 📊 Current Database Status")
        
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
    if st.button("🔄 Refresh Graph & Stats", use_container_width=True):
        if auto_refresh_graph():
            st.success("✅ Graph and stats refreshed!")
            st.rerun()
    
    # ============================================
    # CHAT MESSAGES DISPLAY - NO EXPANDERS
    # ============================================
    
    st.markdown("### 📝 Chat History")
    
    # Display chat messages (last 5 for performance)
    recent_messages = st.session_state.messages[-10:] if len(st.session_state.messages) > 10 else st.session_state.messages
    
    for message in recent_messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>🧑 You:</strong> {message["content"]}
                <br><small>⏰ {datetime.fromisoformat(message["timestamp"]).strftime("%H:%M:%S")}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            result = message["content"]
            
            st.markdown(f"""
            <div class="assistant-message">
                <strong>🤖 Neo4j Agent:</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Display response components WITHOUT expanders
            if result.get("tool"):
                st.markdown(f"""
                <div class="tool-badge">
                    🔧 Tool: {result["tool"]}
                </div>
                """, unsafe_allow_html=True)
            
            if result.get("query"):
                st.markdown("**Generated Query:**")
                st.markdown(f"""
                <div class="query-display">
                    {result["query"]}
                </div>
                """, unsafe_allow_html=True)
            
            if result.get("answer"):
                st.markdown("**Result:**")
                st.markdown(f"""
                <div class="result-display">
                    {result["answer"]}
                </div>
                """, unsafe_allow_html=True)
            
            if result.get("response_time"):
                st.caption(f"⏱️ Processed in {result['response_time']:.2f}s")
    
    # ============================================
    # CHAT INPUT
    # ============================================
    
    st.markdown("### ✍️ Ask Your Question")
    
    # Chat input using st.chat_input
    if prompt := st.chat_input("Ask about your Neo4j database (e.g., 'How many nodes are there?', 'Create a Person named Alice')"):
        # Add user message
        user_message = {
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(user_message)
        
        # Process the request
        with st.spinner("🧠 Processing your request..."):
            result = send_chat_message(prompt)
        
        # Add assistant message
        assistant_message = {
            "role": "assistant",
            "content": result,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(assistant_message)
        
        # Auto-refresh graph if it was a write operation
        if result.get("tool") == "write_neo4j_cypher" and result.get("success", True):
            with st.spinner("🔄 Updating graph visualization..."):
                if auto_refresh_graph():
                    st.markdown("""
                    <div class="update-indicator">
                        ✅ Graph updated with your changes!
                    </div>
                    """, unsafe_allow_html=True)
        
        # Rerun to show new messages
        st.rerun()

# ============================================
# RIGHT COLUMN - LIVE GRAPH VISUALIZATION
# ============================================

with col2:
    st.markdown("## 🎨 Live Neo4j Graph Visualization")
    
    # Display last update time
    if st.session_state.last_update_time:
        st.caption(f"🕒 Last updated: {st.session_state.last_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display the graph
    if st.session_state.graph_data and "error" not in st.session_state.graph_data:
        graph_data = st.session_state.graph_data
        
        # Create and display the interactive graph
        fig = create_network_graph(graph_data, title="🌐 Live Neo4j Graph")
        
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="live_graph")
            
            # Show graph information
            nodes = graph_data.get("nodes", [])
            relationships = graph_data.get("relationships", [])
            
            st.markdown("### 📈 Graph Statistics")
            
            col_x, col_y, col_z = st.columns(3)
            with col_x:
                st.metric("Visible Nodes", len(nodes))
            with col_y:
                st.metric("Visible Relationships", len(relationships))
            with col_z:
                if graph_data.get("summary"):
                    total_nodes = graph_data["summary"].get("nodes", 0)
                    st.metric("Total in Database", total_nodes)
            
            # Show sample node information
            if nodes:
                st.markdown("### 👁️ Sample Node Data")
                for i, node in enumerate(nodes[:3]):
                    labels = ", ".join(node.get("labels", []))
                    caption = node.get("caption", f"Node {node['id']}")
                    st.markdown(f"**{i+1}.** {caption} `({labels})`")
        else:
            st.warning("⚠️ Could not generate graph visualization")
    
    else:
        if st.session_state.graph_data and "error" in st.session_state.graph_data:
            st.error(f"❌ Graph data error: {st.session_state.graph_data['error']}")
        else:
            st.info("📊 Loading graph data...")

# ============================================
# SIDEBAR - QUICK ACTIONS
# ============================================

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
            user_message = {
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(user_message)
            
            # Process the query
            with st.spinner("🧠 Processing example query..."):
                result = send_chat_message(query)
            
            assistant_message = {
                "role": "assistant",
                "content": result,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(assistant_message)
            
            # Refresh graph if it was a write operation
            if result.get("tool") == "write_neo4j_cypher" and result.get("success", True):
                auto_refresh_graph()
            
            st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Manual graph refresh
    if st.button("🔄 Force Graph Refresh", use_container_width=True):
        if auto_refresh_graph():
            st.success("✅ Graph refreshed!")
            st.rerun()
    
    # NVL Visualization Link
    if st.button("🌐 Open NVL Visualization", use_container_width=True):
        st.markdown(f"[🌐 Open Neo4j NVL Interface](http://localhost:{ENHANCED_SERVER_PORT}/viz)")
    
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

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 0.5rem; margin-top: 2rem;">
    <h3>🧠 Neo4j Live Chatbot v3.2 - FIXED & ENHANCED</h3>
    <p><strong>✅ FIXES:</strong> No nested expanders, Auto graph loading, Real-time updates</p>
    <p><strong>🚀 FEATURES:</strong> Live visualization, Auto-refresh after changes, Clean UI</p>
    <p><strong>📡 Session:</strong> <code>{st.session_state.session_id[:8]}...</code></p>
    <p><strong>🌐 NVL Interface:</strong> <a href="http://localhost:{ENHANCED_SERVER_PORT}/viz" target="_blank">http://localhost:{ENHANCED_SERVER_PORT}/viz</a></p>
</div>
""", unsafe_allow_html=True)
