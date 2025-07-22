
"""
Fixed UI - Simple NVL Integration (No IframeMixin Error)
This version fixes the IframeMixin_html() error with a simplified approach
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
import networkx as nx

# Configuration
ENHANCED_APP_PORT = 8081

# Page configuration
st.set_page_config(
    page_title="Fixed Neo4j Chatbot with Simple NVL",
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
    
    .simple-graph-container {
        border: 2px solid #e0e0e0;
        border-radius: 0.5rem;
        background: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        padding: 1rem;
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
# HELPER FUNCTIONS (FIXED)
# ============================================

def check_enhanced_server_health():
    """Check health of the enhanced server"""
    try:
        response = requests.get(f"http://localhost:{ENHANCED_APP_PORT}/health", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "data": response.json()}
        else:
            return {"status": "error", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "disconnected", "error": f"Enhanced server not running on port {ENHANCED_APP_PORT}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def get_enhanced_database_stats():
    """Get enhanced database statistics"""
    try:
        response = requests.get(f"http://localhost:{ENHANCED_APP_PORT}/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def get_enhanced_graph_data(limit: int = 100):
    """Get enhanced graph data"""
    try:
        response = requests.get(f"http://localhost:{ENHANCED_APP_PORT}/graph?limit={limit}", timeout=15)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def send_enhanced_chat_message(question: str):
    """Send chat message to enhanced server"""
    try:
        payload = {
            "question": question,
            "session_id": st.session_state.session_id
        }
        
        start_time = time.time()
        
        response = requests.post(
            f"http://localhost:{ENHANCED_APP_PORT}/chat",
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
                "answer": f"❌ Enhanced server error: {response.status_code}",
                "response_time": response_time
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to enhanced server",
            "answer": "❌ Enhanced server not running. Start the enhanced server on port 8081.",
            "response_time": 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "answer": f"❌ Enhanced request failed: {str(e)}",
            "response_time": 0
        }

def create_simple_plotly_graph(graph_data, title="Neo4j Graph"):
    """Create simple Plotly graph (FIXED - No iframe issues)"""
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
        for rel in relationships:
            source = str(rel["source"])
            target = str(rel["target"])
            
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target)
        
        # Get layout
        if len(G.nodes) > 100:
            pos = nx.spring_layout(G, k=1, iterations=20)
        else:
            pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Relationships'
        )
        
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
            name="Nodes"
        )
        
        # Create the figure
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title={
                'text': f'{title}<br><sub>Nodes: {len(nodes)} | Relationships: {len(relationships)} | Updated: {datetime.now().strftime("%H:%M:%S")}</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=80),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif")
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating simple graph: {e}")
        return None

def auto_refresh_enhanced_graph():
    """Auto refresh enhanced graph data and update session state"""
    try:
        new_graph_data = get_enhanced_graph_data(100)
        if "error" not in new_graph_data:
            st.session_state.graph_data = new_graph_data
            st.session_state.last_update_time = datetime.now()
            
            # Update database stats
            new_stats = get_enhanced_database_stats()
            if "error" not in new_stats:
                st.session_state.database_stats = new_stats
            
            return True
        return False
    except Exception as e:
        st.error(f"Failed to refresh enhanced graph: {e}")
        return False

# ============================================
# MAIN UI LAYOUT (FIXED)
# ============================================

# Header
st.markdown("""
<div class="header-container">
    <h1>🧠 Fixed Neo4j Chatbot with Simple Graph</h1>
    <p>Real-time graph database interaction - FIXED IframeMixin Error</p>
    <p><strong>✅ WORKING SOLUTION: Simple Plotly + Real-time Updates</strong></p>
</div>
""", unsafe_allow_html=True)

# Check enhanced server status first
enhanced_server_health = check_enhanced_server_health()

if enhanced_server_health["status"] != "healthy":
    st.error(f"❌ Enhanced server not available: {enhanced_server_health['error']}")
    st.info("Please start the enhanced server: `python app.py`")
    st.stop()

# ============================================
# AUTO-LOAD ENHANCED GRAPH DATA ON STARTUP
# ============================================

if not st.session_state.initial_graph_loaded:
    with st.spinner("🚀 Auto-loading REAL Neo4j graph data..."):
        initial_graph_data = get_enhanced_graph_data(100)
        if "error" not in initial_graph_data:
            st.session_state.graph_data = initial_graph_data
            st.session_state.initial_graph_loaded = True
            st.session_state.last_update_time = datetime.now()
            
            # Load enhanced database stats
            initial_stats = get_enhanced_database_stats()
            if "error" not in initial_stats:
                st.session_state.database_stats = initial_stats
            
            nodes_count = len(initial_graph_data.get("nodes", []))
            rels_count = len(initial_graph_data.get("relationships", []))
            st.success(f"✅ Loaded REAL enhanced data: {nodes_count} nodes, {rels_count} relationships")
        else:
            st.error(f"❌ Failed to load enhanced graph: {initial_graph_data['error']}")

# ============================================
# CREATE MAIN LAYOUT - TWO COLUMNS
# ============================================

col1, col2 = st.columns([1, 1.5])

# ============================================
# LEFT COLUMN - ENHANCED CHAT INTERFACE
# ============================================

with col1:
    st.markdown("## 💬 Enhanced Neo4j Chat")
    
    # Display enhanced database stats
    if st.session_state.database_stats and "error" not in st.session_state.database_stats:
        stats = st.session_state.database_stats
        
        st.markdown("### 📊 Enhanced Database Status")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total Nodes", stats.get("nodes", 0))
            st.metric("Total Relationships", stats.get("relationships", 0))
        with col_b:
            st.metric("Node Labels", len(stats.get("labels", [])))
            st.metric("Relationship Types", len(stats.get("relationship_types", [])))
    
    # Enhanced manual refresh button
    if st.button("🔄 Refresh Enhanced Graph", use_container_width=True):
        if auto_refresh_enhanced_graph():
            st.success("✅ Enhanced graph data refreshed!")
            st.rerun()
    
    # ============================================
    # ENHANCED CHAT MESSAGES DISPLAY
    # ============================================
    
    st.markdown("### 📝 Enhanced Chat History")
    
    # Display enhanced chat messages (last 8 for performance)
    recent_messages = st.session_state.messages[-8:] if len(st.session_state.messages) > 8 else st.session_state.messages
    
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
                <strong>🤖 Enhanced Neo4j Agent:</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Display enhanced response components
            if result.get("tool"):
                st.markdown(f"""
                <div class="tool-badge">
                    🔧 Enhanced Tool: {result["tool"]}
                </div>
                """, unsafe_allow_html=True)
            
            if result.get("query"):
                st.markdown("**Generated Enhanced Query:**")
                st.markdown(f"""
                <div class="query-display">
                    {result["query"]}
                </div>
                """, unsafe_allow_html=True)
            
            if result.get("answer"):
                st.markdown("**Enhanced Result:**")
                st.markdown(f"""
                <div class="result-display">
                    {result["answer"]}
                </div>
                """, unsafe_allow_html=True)
            
            if result.get("response_time"):
                st.caption(f"⏱️ Enhanced processing: {result['response_time']:.2f}s")
    
    # ============================================
    # ENHANCED CHAT INPUT
    # ============================================
    
    st.markdown("### ✍️ Ask Enhanced Question")
    
    # Enhanced chat input
    if prompt := st.chat_input("Ask about your Neo4j database (enhanced with real-time updates)"):
        # Add user message
        user_message = {
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(user_message)
        
        # Process the enhanced request
        with st.spinner("🧠 Processing with enhanced agent..."):
            result = send_enhanced_chat_message(prompt)
        
        # Add enhanced assistant message
        assistant_message = {
            "role": "assistant",
            "content": result,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(assistant_message)
        
        # Auto-refresh enhanced graph if it was a write operation
        if result.get("tool") == "write_neo4j_cypher" and result.get("success", True):
            with st.spinner("🔄 Updating enhanced graph visualization..."):
                if auto_refresh_enhanced_graph():
                    st.markdown("""
                    <div class="update-indicator">
                        ✅ Enhanced graph updated with your changes!
                    </div>
                    """, unsafe_allow_html=True)
        
        # Rerun to show new messages and updated enhanced graph
        st.rerun()

# ============================================
# RIGHT COLUMN - SIMPLE PLOTLY GRAPH (FIXED)
# ============================================

with col2:
    st.markdown("## 🎨 Enhanced Neo4j Graph (Fixed)")
    
    # Display enhanced last update time
    if st.session_state.last_update_time:
        st.caption(f"🕒 Enhanced data updated: {st.session_state.last_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display the enhanced graph using simple Plotly (NO IFRAME ISSUES)
    if st.session_state.graph_data and "error" not in st.session_state.graph_data:
        graph_data = st.session_state.graph_data
        
        # Show enhanced graph summary
        nodes = graph_data.get("nodes", [])
        relationships = graph_data.get("relationships", [])
        
        st.markdown("### 📈 Enhanced Graph Statistics")
        
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            st.metric("Enhanced Nodes", len(nodes))
        with col_y:
            st.metric("Enhanced Relationships", len(relationships))
        with col_z:
            if graph_data.get("summary"):
                total_nodes = graph_data["summary"].get("nodes", 0)
                st.metric("Total in Enhanced DB", total_nodes)
        
        # Create and display the simple Plotly graph (FIXED)
        if nodes:
            fig = create_simple_plotly_graph(graph_data, title="🌐 Enhanced Neo4j Graph")
            
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="enhanced_simple_graph")
                
                # Show enhanced sample data
                st.markdown("### 👁️ Enhanced Sample Node Data")
                for i, node in enumerate(nodes[:3]):
                    labels = ", ".join(node.get("labels", []))
                    properties = node.get("properties", {})
                    caption = node.get("caption", f"Node {node['id']}")
                    
                    st.markdown(f"**{i+1}.** {caption}")
                    st.caption(f"Enhanced Labels: `{labels}` | Properties: {len(properties)} items")
            else:
                st.warning("⚠️ Could not generate enhanced graph visualization")
        else:
            st.info("📝 No nodes found in enhanced Neo4j database. Create some data to see the enhanced graph!")
            
            # Show message for empty database
            st.markdown("""
            <div class="simple-graph-container">
                <div style="text-align: center; padding: 2rem; color: #666;">
                    <h3>📊 Empty Database</h3>
                    <p>Create some nodes to see the enhanced graph visualization!</p>
                    <p><strong>Try:</strong> "Create a Person named Alice with age 30"</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        if st.session_state.graph_data and "error" in st.session_state.graph_data:
            st.error(f"❌ Enhanced graph data error: {st.session_state.graph_data['error']}")
        else:
            st.info("📊 Loading enhanced Neo4j graph data...")

# ============================================
# SIDEBAR - ENHANCED QUICK ACTIONS
# ============================================

with st.sidebar:
    st.markdown("## 🚀 Enhanced Quick Actions")
    
    # Enhanced Direct Links
    if st.button("🌐 Enhanced Server UI", use_container_width=True):
        st.markdown(f"[🌐 Enhanced Interface](http://localhost:{ENHANCED_APP_PORT}/ui)")
    
    if st.button("🎯 Direct NVL Interface", use_container_width=True):
        st.markdown(f"[🎯 Direct NVL](http://localhost:8000/nvl)")
    
    # Enhanced example queries
    st.markdown("### 💡 Try These Enhanced Queries:")
    
    enhanced_example_queries = [
        "How many nodes are in the graph?",
        "Show me the enhanced database schema",
        "Create a Person named Alice with age 30",
        "Create a Company called TechCorp",
        "Connect Alice to TechCorp with relationship WORKS_FOR",
        "List all Person nodes",
        "Find the most connected nodes",
        "Create 3 connected Person nodes",
        "Delete all TestNode nodes",
        "Show me all node labels and relationship types"
    ]
    
    for query in enhanced_example_queries:
        if st.button(query, key=f"enhanced_example_{hash(query)}", use_container_width=True):
            # Add to enhanced chat
            user_message = {
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(user_message)
            
            # Process the enhanced query
            with st.spinner("🧠 Processing enhanced query..."):
                result = send_enhanced_chat_message(query)
            
            assistant_message = {
                "role": "assistant",
                "content": result,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(assistant_message)
            
            # Refresh enhanced graph if it was a write operation
            if result.get("tool") == "write_neo4j_cypher" and result.get("success", True):
                auto_refresh_enhanced_graph()
            
            st.rerun()
    
    # Enhanced control buttons
    if st.button("🗑️ Clear Enhanced Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔄 Force Enhanced Refresh", use_container_width=True):
        if auto_refresh_enhanced_graph():
            st.success("✅ Enhanced graph refreshed!")
            st.rerun()
    
    # Enhanced session info
    st.markdown("---")
    st.markdown("### 📋 Enhanced Session Info")
    st.text(f"Session ID: {st.session_state.session_id[:8]}...")
    st.text(f"Enhanced Messages: {len(st.session_state.messages)}")
    if st.session_state.database_stats:
        stats = st.session_state.database_stats
        if "error" not in stats:
            st.text(f"Enhanced DB Nodes: {stats.get('nodes', 0)}")
            st.text(f"Enhanced DB Rels: {stats.get('relationships', 0)}")

# ============================================
# ENHANCED FOOTER (FIXED)
# ============================================

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 0.5rem; margin-top: 2rem;">
    <h3>🧠 Fixed Neo4j Enhanced Chatbot v6.0</h3>
    <p><strong>✅ FIXED:</strong> No more IframeMixin_html() errors</p>
    <p><strong>🎯 SOLUTION:</strong> Simple Plotly + Real-time Updates + Enhanced Features</p>
    <p><strong>🔄 FEATURES:</strong> Live graph updates, Real-time chat, Enhanced database interaction</p>
    <p><strong>📡 Session:</strong> <code>{st.session_state.session_id[:8]}...</code></p>
    <p><strong>🌐 Enhanced Interfaces:</strong></p>
    <p>• <a href="http://localhost:{ENHANCED_APP_PORT}/ui" target="_blank">Enhanced Interface</a></p>
    <p>• <a href="http://localhost:8000/nvl" target="_blank">Direct NVL Interface</a></p>
</div>
""", unsafe_allow_html=True)
