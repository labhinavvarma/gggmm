"""
Enhanced Streamlit UI with Neo4j NVL Visualization Integration
This provides a comprehensive interface for the Neo4j FastMCP Agent with live graph visualization
Run this AFTER starting the enhanced FastMCP server
"""

import streamlit as st
import requests
import uuid
import time
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from streamlit.components.v1 import html
import pandas as pd

# Configuration - Updated for Enhanced FastMCP
ENHANCED_FASTMCP_PORT = 8000

# Page configuration
st.set_page_config(
    page_title="Neo4j Enhanced Agent with NVL",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    
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
    }
    
    .result-display {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00d4aa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .nvl-badge {
        background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .viz-container {
        background: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .chat-message {
        background: white;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
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
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
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
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False

# ============================================
# HELPER FUNCTIONS
# ============================================

def check_enhanced_fastmcp_health():
    """Check health of Enhanced FastMCP server"""
    try:
        response = requests.get(f"http://localhost:{ENHANCED_FASTMCP_PORT}/health", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "data": response.json()}
        else:
            return {"status": "error", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "disconnected", "error": f"Enhanced FastMCP server not running on port {ENHANCED_FASTMCP_PORT}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def get_database_stats():
    """Get current database statistics"""
    try:
        response = requests.get(f"http://localhost:{ENHANCED_FASTMCP_PORT}/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def get_graph_data(limit: int = 50):
    """Get graph data for visualization"""
    try:
        response = requests.get(f"http://localhost:{ENHANCED_FASTMCP_PORT}/graph?limit={limit}", timeout=15)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def send_chat_message(question: str):
    """Send chat message to Enhanced FastMCP server"""
    try:
        payload = {
            "question": question,
            "session_id": st.session_state.session_id
        }
        
        start_time = time.time()
        
        response = requests.post(
            f"http://localhost:{ENHANCED_FASTMCP_PORT}/chat",
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
            "error": "Cannot connect to Enhanced FastMCP server",
            "answer": "❌ Enhanced FastMCP server not running. Start enhanced_fastmcp_server.py on port 8000.",
            "response_time": 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "answer": f"❌ Request failed: {str(e)}",
            "response_time": 0
        }

def create_network_graph(graph_data):
    """Create a network graph using Plotly"""
    if not graph_data or "error" in graph_data:
        return None
    
    nodes = graph_data.get("nodes", [])
    relationships = graph_data.get("relationships", [])
    
    if not nodes:
        return None
    
    # Create network graph
    import networkx as nx
    
    G = nx.Graph()
    
    # Add nodes
    node_labels = {}
    for node in nodes:
        node_id = node["id"]
        G.add_node(node_id)
        node_labels[node_id] = node.get("caption", f"Node {node_id}")
    
    # Add edges
    for rel in relationships:
        source = rel["source"]
        target = rel["target"]
        if source in G.nodes and target in G.nodes:
            G.add_edge(source, target)
    
    # Get layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create traces
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.extend([x0, x1, None])
        edge_trace.extend([y0, y1, None])
    
    # Create edge trace
    edge_trace_plot = go.Scatter(
        x=edge_trace[::3],
        y=edge_trace[1::3],
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = [node_labels.get(node, node) for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            reversescale=True,
            color=[],
            size=20,
            colorbar=dict(
                thickness=15,
                len=0.5,
                x=1.05
            ),
            line=dict(width=2)
        )
    )
    
    # Color nodes by degree
    node_adjacencies = []
    for node in G.nodes():
        node_adjacencies.append(len(list(G.neighbors(node))))
    
    node_trace.marker.color = node_adjacencies
    
    # Create figure
    fig = go.Figure(data=[edge_trace_plot, node_trace],
                   layout=go.Layout(
                       title=f'Neo4j Graph Visualization ({len(nodes)} nodes, {len(relationships)} relationships)',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Interactive Neo4j Graph",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor="left", yanchor="bottom",
                           font=dict(color="#888", size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    return fig

# ============================================
# MAIN UI LAYOUT
# ============================================

# Header
st.markdown("""
<div class="header-container">
    <h1>🧠 Neo4j Enhanced Agent with NVL Visualization</h1>
    <p>Real-time graph database interaction with live visualization using Neo4j NVL</p>
    <div class="nvl-badge">
        🚀 Enhanced FastMCP + LangGraph + Neo4j NVL + WebSocket
    </div>
</div>
""", unsafe_allow_html=True)

# Create main columns
main_col, viz_col = st.columns([1, 1])

# ============================================
# SIDEBAR - SYSTEM STATUS AND CONTROLS
# ============================================

with st.sidebar:
    st.markdown("## 🔧 System Status")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("🔄 Auto-refresh stats", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
    
    if st.button("🔄 Refresh Now"):
        st.rerun()
    
    # Check Enhanced FastMCP Server
    fastmcp_health = check_enhanced_fastmcp_health()
    if fastmcp_health["status"] == "healthy":
        st.markdown("""
        <div class="status-healthy">
            <span class="live-indicator"></span>Enhanced FastMCP: Online
        </div>
        """, unsafe_allow_html=True)
        
        # Show detailed status
        health_data = fastmcp_health.get("data", {})
        neo4j_status = health_data.get("neo4j", {}).get("status", "unknown")
        agent_status = health_data.get("agent", {}).get("status", "unknown")
        viz_status = health_data.get("visualization", {}).get("status", "unknown")
        
        st.markdown("### 📊 Components")
        components_df = pd.DataFrame([
            {"Component": "Neo4j", "Status": neo4j_status},
            {"Component": "Agent", "Status": agent_status},
            {"Component": "Visualization", "Status": viz_status},
            {"Component": "WebSocket", "Status": f"{health_data.get('visualization', {}).get('active_connections', 0)} connections"}
        ])
        st.dataframe(components_df, hide_index=True)
        
        # Get and display database stats
        database_stats = get_database_stats()
        if "error" not in database_stats:
            st.session_state.database_stats = database_stats
            
            st.markdown("### 📈 Database Stats")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Nodes", database_stats.get("nodes", 0))
                st.metric("Labels", len(database_stats.get("labels", [])))
            with col2:
                st.metric("Relationships", database_stats.get("relationships", 0))
                st.metric("Types", len(database_stats.get("relationship_types", [])))
            
            # Show labels and relationship types
            if database_stats.get("labels"):
                st.markdown("**Node Labels:**")
                st.text(", ".join(database_stats["labels"][:10]))
            
            if database_stats.get("relationship_types"):
                st.markdown("**Relationship Types:**")
                st.text(", ".join(database_stats["relationship_types"][:10]))
    
    else:
        st.markdown(f"""
        <div class="status-error">
            ❌ Enhanced FastMCP: {fastmcp_health['error']}
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Actions
    st.markdown("## ⚡ Quick Actions")
    
    if st.button("🎯 Open NVL Visualization", use_container_width=True):
        st.markdown(f"""
        <script>
        window.open('http://localhost:{ENHANCED_FASTMCP_PORT}/viz', '_blank');
        </script>
        """, unsafe_allow_html=True)
        st.info(f"Opening NVL visualization at http://localhost:{ENHANCED_FASTMCP_PORT}/viz")
    
    if st.button("📊 Refresh Graph Data", use_container_width=True):
        graph_data = get_graph_data(100)
        if "error" not in graph_data:
            st.session_state.graph_data = graph_data
            st.success("Graph data refreshed!")
        else:
            st.error(f"Failed to refresh: {graph_data['error']}")
    
    # Visualization Settings
    st.markdown("## 🎨 Visualization Settings")
    
    viz_limit = st.slider("Graph Node Limit", min_value=10, max_value=500, value=50, step=10)
    show_properties = st.checkbox("Show Node Properties", value=True)
    
    # Example queries
    st.markdown("## 💡 Example Queries")
    examples = [
        "How many nodes are in the graph?",
        "Show me the database schema",
        "Create a Person named Alice with age 30",
        "Create a Company named TechCorp",
        "Connect Alice to TechCorp as an employee",
        "List all node labels",
        "Find nodes with most connections",
        "Delete all TestNode nodes",
        "Show me all Person nodes"
    ]
    
    for example in examples:
        if st.button(example, key=f"ex_{hash(example)}", use_container_width=True):
            st.session_state.example_query = example

# ============================================
# MAIN CHAT INTERFACE
# ============================================

with main_col:
    st.markdown("## 💬 Chat Interface")
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Ask about your Neo4j database:",
            placeholder="e.g., How many nodes are in the graph?",
            height=100,
            key="chat_input"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            submitted = st.form_submit_button("🚀 Send Query", use_container_width=True)
        
        with col2:
            if st.form_submit_button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        with col3:
            if st.form_submit_button("📊 View Graph", use_container_width=True):
                graph_data = get_graph_data(viz_limit)
                if "error" not in graph_data:
                    st.session_state.graph_data = graph_data
    
    # Handle example selection
    if hasattr(st.session_state, 'example_query'):
        user_input = st.session_state.example_query
        submitted = True
        st.info(f"🎯 Running example: {user_input}")
        delattr(st.session_state, 'example_query')
    
    # Process query
    if submitted and user_input:
        # Check if Enhanced FastMCP is available first
        fastmcp_health = check_enhanced_fastmcp_health()
        if fastmcp_health["status"] != "healthy":
            st.error("❌ Cannot send query: Enhanced FastMCP server is not running!")
            st.error("Please start enhanced_fastmcp_server.py first on port 8000")
        else:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })
            
            # Send to Enhanced FastMCP agent
            with st.spinner("🧠 Processing with Enhanced FastMCP..."):
                result = send_chat_message(user_input)
            
            # Add agent response
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update graph data if available
            if result.get("graph_data"):
                st.session_state.graph_data = result["graph_data"]
            
            # Show immediate result
            st.markdown("---")
            
            if result.get("success", True):
                response_time = result.get("response_time", 0)
                st.success(f"✅ Processed by Enhanced FastMCP in {response_time:.2f}s")
                
                # Display components
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
                
                # Show operation summary if available
                if result.get("operation_summary"):
                    summary = result["operation_summary"]
                    if "error" not in summary:
                        st.info(f"📊 Database now has {summary.get('nodes', 0)} nodes and {summary.get('relationships', 0)} relationships")
                        
            else:
                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    # Recent Messages Display
    if st.session_state.messages:
        st.markdown("---")
        st.markdown("## 📝 Recent Messages")
        
        # Show last 5 messages
        for message in reversed(st.session_state.messages[-5:]):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message">
                    <strong>🧑 You:</strong> {message["content"]}
                    <br><small>⏰ {message.get("timestamp", "")}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                result = message["content"]
                
                with st.expander(f"🤖 Enhanced Agent: {result.get('tool', 'Response')}", expanded=False):
                    if result.get("tool"):
                        st.markdown(f"**Tool:** {result['tool']}")
                    
                    if result.get("query"):
                        st.markdown("**Query:**")
                        st.code(result["query"], language="cypher")
                    
                    if result.get("answer"):
                        st.markdown("**Answer:**")
                        st.markdown(result["answer"])
                    
                    if result.get("response_time"):
                        st.markdown(f"**Processing Time:** {result['response_time']:.2f}s")
                    
                    if result.get("trace"):
                        with st.expander("🔍 Trace Details"):
                            st.text(result["trace"])

# ============================================
# VISUALIZATION COLUMN
# ============================================

with viz_col:
    st.markdown("## 🎨 Graph Visualization")
    
    # Visualization tabs
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Interactive Graph", "📈 Statistics", "🌐 NVL Interface"])
    
    with viz_tab1:
        # Load graph data if not already loaded
        if st.session_state.graph_data is None and st.button("📊 Load Graph Data"):
            with st.spinner("Loading graph data..."):
                graph_data = get_graph_data(viz_limit)
                if "error" not in graph_data:
                    st.session_state.graph_data = graph_data
                else:
                    st.error(f"Failed to load graph data: {graph_data['error']}")
        
        # Display graph visualization
        if st.session_state.graph_data:
            graph_data = st.session_state.graph_data
            
            if "error" not in graph_data:
                nodes = graph_data.get("nodes", [])
                relationships = graph_data.get("relationships", [])
                
                if nodes:
                    # Create and display network graph
                    fig = create_network_graph(graph_data)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show node and relationship details
                    st.markdown("### 📋 Graph Details")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Nodes in View", len(nodes))
                        if nodes and show_properties:
                            with st.expander("👁️ Sample Nodes"):
                                for node in nodes[:5]:
                                    st.json({
                                        "id": node["id"],
                                        "labels": node["labels"],
                                        "caption": node["caption"]
                                    })
                    
                    with col2:
                        st.metric("Relationships in View", len(relationships))
                        if relationships:
                            with st.expander("🔗 Sample Relationships"):
                                for rel in relationships[:5]:
                                    st.json({
                                        "type": rel["type"],
                                        "source": rel["source"],
                                        "target": rel["target"]
                                    })
                else:
                    st.info("📝 No nodes found in the database. Create some data to see the visualization!")
            else:
                st.error(f"Failed to load graph: {graph_data['error']}")
        else:
            st.info("📊 Click 'Load Graph Data' to see the visualization")
    
    with viz_tab2:
        # Statistics and metrics
        if st.session_state.database_stats:
            stats = st.session_state.database_stats
            
            if "error" not in stats:
                # Create metrics display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <h3>📊 Nodes</h3>
                        <h2>{}</h2>
                    </div>
                    """.format(stats.get("nodes", 0)), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="metric-card">
                        <h3>🔗 Relationships</h3>
                        <h2>{}</h2>
                    </div>
                    """.format(stats.get("relationships", 0)), unsafe_allow_html=True)
                
                # Labels chart
                if stats.get("labels"):
                    st.markdown("### 🏷️ Node Labels")
                    labels_df = pd.DataFrame({
                        "Label": stats["labels"],
                        "Count": [1] * len(stats["labels"])  # Placeholder - would need actual counts
                    })
                    fig_labels = px.bar(labels_df, x="Label", y="Count", title="Node Labels Distribution")
                    st.plotly_chart(fig_labels, use_container_width=True)
                
                # Relationship types chart
                if stats.get("relationship_types"):
                    st.markdown("### ➡️ Relationship Types")
                    rel_types_df = pd.DataFrame({
                        "Type": stats["relationship_types"],
                        "Count": [1] * len(stats["relationship_types"])  # Placeholder
                    })
                    fig_rels = px.pie(rel_types_df, values="Count", names="Type", title="Relationship Types")
                    st.plotly_chart(fig_rels, use_container_width=True)
                
                # Last updated
                if stats.get("timestamp"):
                    st.markdown(f"**Last Updated:** {stats['timestamp']}")
        else:
            st.info("📈 Database statistics will appear here")
    
    with viz_tab3:
        # NVL Interface integration
        st.markdown("### 🌐 Neo4j NVL Interface")
        
        st.markdown(f"""
        The Neo4j Visualization Library (NVL) provides an advanced, interactive graph visualization.
        
        **Features:**
        - Real-time graph rendering
        - Interactive node exploration  
        - Live database updates via WebSocket
        - Advanced layout algorithms
        - Property inspection on hover
        """)
        
        if st.button("🚀 Open Full NVL Interface", key="open_nvl"):
            # Create a link to the NVL interface
            st.markdown(f"""
            <a href="http://localhost:{ENHANCED_FASTMCP_PORT}/viz" target="_blank">
                <button style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem 2rem; border: none; border-radius: 0.5rem; cursor: pointer; font-size: 16px; text-decoration: none;">
                    🌐 Open NVL Visualization
                </button>
            </a>
            """, unsafe_allow_html=True)
        
        # Embed a simplified NVL view (if server is running)
        fastmcp_health = check_enhanced_fastmcp_health()
        if fastmcp_health["status"] == "healthy":
            if st.checkbox("📺 Show Embedded View", key="embed_nvl"):
                # Embed the NVL interface
                html(f"""
                <iframe 
                    src="http://localhost:{ENHANCED_FASTMCP_PORT}/viz" 
                    width="100%" 
                    height="600" 
                    frameborder="0"
                    style="border-radius: 0.5rem;">
                </iframe>
                """, height=650)
        else:
            st.warning("⚠️ Enhanced FastMCP server must be running to use NVL interface")

# ============================================
# AUTO-REFRESH LOGIC
# ============================================

# Auto-refresh mechanism
if st.session_state.auto_refresh:
    # Use st.empty() to create a placeholder for auto-refresh
    placeholder = st.empty()
    
    # Auto-refresh every 30 seconds
    time.sleep(1)
    if time.time() % 30 < 1:  # Rough approximation
        st.rerun()

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 0.5rem; margin-top: 2rem;">
    <h3>🧠 Neo4j Enhanced Agent with NVL v3.0</h3>
    <p><strong>Enhanced Architecture:</strong> FastMCP + @mcp.tool + LangGraph + Neo4j NVL + WebSocket</p>
    <p><strong>Features:</strong> Real-time visualization, Live updates, Interactive graphs, Database monitoring</p>
    <p><strong>Session:</strong> <code>{st.session_state.session_id[:8]}...</code></p>
    <div style="margin-top: 1rem;">
        <span class="live-indicator"></span>
        <span>Live connection to Enhanced FastMCP Server on port {ENHANCED_FASTMCP_PORT}</span>
    </div>
</div>
""", unsafe_allow_html=True)
