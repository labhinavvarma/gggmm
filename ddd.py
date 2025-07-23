import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import requests
import json
import os
import tempfile
from datetime import datetime
import uuid
import traceback

# Page configuration
st.set_page_config(
    page_title="Neo4j Graph Explorer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        max-width: 100%;
    }
    
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    
    .stButton button:hover {
        background-color: #0d5aa7;
    }
    
    .metric-container {
        background-color: #f1f3f4;
        padding: 0.75rem;
        border-radius: 0.25rem;
        text-align: center;
        margin: 0.25rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    defaults = {
        "conversation_history": [],
        "graph_data": None,
        "last_response": None,
        "session_id": str(uuid.uuid4()),
        "connection_status": "unknown"
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# Header
st.markdown('<h1 class="main-header">🕸️ Neo4j Graph Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6c757d;"><strong>Guaranteed Relationship Visibility</strong> • Interactive Graph Network</p>', unsafe_allow_html=True)

def test_api_connection():
    """Test if the API is working"""
    try:
        response = requests.get("http://localhost:8081/health", timeout=5)
        if response.status_code == 200:
            st.session_state.connection_status = "connected"
            return True
        else:
            st.session_state.connection_status = "error"
            return False
    except:
        st.session_state.connection_status = "disconnected"
        return False

def call_agent_api(question: str, node_limit: int = 50) -> dict:
    """Bulletproof API call"""
    try:
        api_url = "http://localhost:8081/chat"
        payload = {
            "question": question,
            "session_id": st.session_state.session_id,
            "node_limit": node_limit
        }
        
        with st.spinner("🤖 Processing your request..."):
            response = requests.post(api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            st.session_state.connection_status = "connected"
            return result
            
    except requests.exceptions.ConnectionError:
        st.session_state.connection_status = "disconnected"
        st.error("❌ Cannot connect to agent API. Please ensure the FastAPI server is running on port 8081.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. The query might be too complex.")
        return None
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
        return None

def get_node_color(labels):
    """Get color for node based on labels"""
    if not labels:
        return "#95afc0"
    
    label = labels[0] if isinstance(labels, list) else str(labels)
    
    colors = {
        "Person": "#FF6B6B",
        "Movie": "#4ECDC4", 
        "Company": "#45B7D1",
        "Product": "#96CEB4",
        "Location": "#FECA57",
        "Event": "#FF9FF3",
        "User": "#A55EEA",
        "Order": "#26DE81",
        "Category": "#FD79A8",
        "Department": "#6C5CE7",
        "Project": "#FDCB6E",
        "Actor": "#00CEC9",
        "Director": "#E84393"
    }
    
    return colors.get(label, "#95afc0")

def get_relationship_color(rel_type):
    """Get color for relationship"""
    colors = {
        "KNOWS": "#e74c3c",
        "WORKS_FOR": "#3498db",
        "MANAGES": "#9b59b6",
        "LOCATED_IN": "#f39c12",
        "BELONGS_TO": "#27ae60",
        "CREATED": "#e91e63",
        "OWNS": "#673ab7",
        "USES": "#009688",
        "ACTED_IN": "#2196f3",
        "DIRECTED": "#ff9800",
        "PRODUCED": "#4caf50"
    }
    return colors.get(rel_type, "#555555")

def render_network_graph(graph_data: dict) -> bool:
    """Bulletproof graph rendering with guaranteed relationship visibility"""
    
    if not graph_data:
        st.info("🔍 No graph data available.")
        return False
    
    try:
        nodes = graph_data.get("nodes", [])
        relationships = graph_data.get("relationships", [])
        
        if not nodes:
            st.info("📊 No nodes found in the data.")
            return False
        
        # Display processing info
        st.markdown(f'<div class="success-box">📊 <strong>Processing:</strong> {len(nodes)} nodes, {len(relationships)} relationships</div>', unsafe_allow_html=True)
        
        # Show relationship types if found
        if relationships:
            rel_types = list(set(rel.get('type', 'UNKNOWN') for rel in relationships))
            st.markdown(f'<div class="success-box">🔗 <strong>Relationship Types:</strong> {", ".join(sorted(rel_types))}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">⚠️ <strong>No relationships found in data</strong> - Graph will show nodes only</div>', unsafe_allow_html=True)
        
        # Create Pyvis network
        net = Network(
            height="600px",
            width="100%", 
            bgcolor="#ffffff",
            font_color="#333333",
            directed=True
        )
        
        # Add nodes safely
        added_nodes = set()
        for i, node in enumerate(nodes):
            try:
                node_id = str(node.get("id", f"node_{i}"))
                if node_id in added_nodes:
                    node_id = f"{node_id}_{i}"
                
                props = node.get("properties", {})
                labels = node.get("labels", ["Unknown"])
                
                # Create display name
                name = props.get("name", props.get("title", f"Node_{i}"))
                display_name = str(name)[:20]
                
                # Create tooltip
                tooltip = f"ID: {node_id}\\nType: {labels[0] if labels else 'Unknown'}"
                if props:
                    tooltip += f"\\nProperties: {len(props)}"
                
                # Add node
                color = get_node_color(labels)
                net.add_node(
                    node_id,
                    label=display_name,
                    title=tooltip,
                    color=color,
                    size=25
                )
                added_nodes.add(node_id)
                
            except Exception as e:
                st.warning(f"⚠️ Skipped node {i}: {str(e)}")
                continue
        
        # Add relationships safely
        added_edges = 0
        for i, rel in enumerate(relationships):
            try:
                start_id = str(rel.get("startNode", rel.get("start", "")))
                end_id = str(rel.get("endNode", rel.get("end", "")))
                rel_type = str(rel.get("type", "CONNECTED"))
                
                # Only add if both nodes exist
                if start_id in added_nodes and end_id in added_nodes:
                    color = get_relationship_color(rel_type)
                    
                    net.add_edge(
                        start_id,
                        end_id,
                        label=rel_type,
                        color=color,
                        width=3,
                        arrows={'to': {'enabled': True, 'scaleFactor': 1.2}},
                        title=f"Type: {rel_type}"
                    )
                    added_edges += 1
                    
            except Exception as e:
                st.warning(f"⚠️ Skipped relationship {i}: {str(e)}")
                continue
        
        st.write(f"✅ **Graph created:** {len(added_nodes)} nodes, {added_edges} relationships")
        
        # Configure physics
        net.set_options("""
        var options = {
            "physics": {
                "enabled": true,
                "stabilization": {"iterations": 100},
                "barnesHut": {
                    "gravitationalConstant": -8000,
                    "centralGravity": 0.3,
                    "springLength": 150,
                    "springConstant": 0.04,
                    "damping": 0.09
                }
            },
            "interaction": {
                "dragNodes": true,
                "dragView": true,
                "zoomView": true
            },
            "edges": {
                "smooth": {"type": "continuous"}
            }
        }
        """)
        
        # Generate HTML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            net.save_graph(f.name)
            html_file = f.name
        
        # Read and display
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Simple wrapper
        wrapped_html = f"""
        <div style="
            border: 2px solid #1f77b4; 
            border-radius: 8px; 
            overflow: hidden; 
            background: white;
        ">
            <div style="
                background: #1f77b4;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
            ">
                🕸️ Interactive Graph | {len(added_nodes)} Nodes | {added_edges} Relationships
            </div>
            {html_content}
        </div>
        """
        
        # Display
        components.html(wrapped_html, height=650, scrolling=False)
        
        # Cleanup
        try:
            os.unlink(html_file)
        except:
            pass
        
        return True
        
    except Exception as e:
        st.error(f"❌ Graph rendering failed: {str(e)}")
        
        with st.expander("🔍 Debug Information", expanded=True):
            st.write("**Error Details:**")
            st.code(traceback.format_exc())
            st.write("**Raw Graph Data:**")
            st.json(graph_data)
        
        return False

# Main layout
col1, col2 = st.columns([1, 2], gap="medium")

with col1:
    st.markdown("### 💬 Chat Interface")
    
    # Connection status
    if st.button("🔌 Test Connection"):
        if test_api_connection():
            st.success("✅ API connection working!")
        else:
            st.error("❌ API connection failed. Please start the FastAPI server.")
    
    # Display connection status
    status_colors = {
        "connected": "🟢",
        "disconnected": "🔴", 
        "error": "🟡",
        "unknown": "⚪"
    }
    st.write(f"**Status:** {status_colors.get(st.session_state.connection_status, '⚪')} {st.session_state.connection_status}")
    
    st.divider()
    
    # Quick actions guaranteed to show relationships
    st.markdown("#### 🚀 Quick Actions (With Relationships)")
    quick_actions = [
        ("Show Network", "Show me all nodes with their relationships"),
        ("Person Network", "Show me all Person nodes and their connections"),
        ("Full Graph", "Display the complete graph with all relationships"),
        ("Database Overview", "Give me an overview of the database structure"),
        ("Sample Network", "Show me a sample of connected data")
    ]
    
    for action_name, action_query in quick_actions:
        if st.button(action_name, key=f"quick_{action_name}"):
            result = call_agent_api(action_query, node_limit=30)
            if result:
                st.session_state.conversation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "question": action_query,
                    "answer": result.get("answer", ""),
                    "graph_data": result.get("graph_data")
                })
                
                if result.get("graph_data"):
                    st.session_state.graph_data = result["graph_data"]
                    st.success("✅ Graph updated!")
                
                st.session_state.last_response = result
                st.rerun()
    
    st.divider()
    
    # Question input
    st.markdown("#### ✍️ Ask Your Question")
    st.info("💡 The system automatically includes relationships in all queries!")
    
    with st.form("question_form", clear_on_submit=True):
        user_question = st.text_area(
            "Your question:",
            placeholder="e.g., Show me Person nodes, Find all companies, What's in the database?",
            height=80
        )
        
        node_limit = st.selectbox(
            "Max nodes:",
            [10, 25, 50, 75],
            index=1,
            help="Fewer nodes = clearer relationships"
        )
        
        submit_button = st.form_submit_button("🚀 Ask")
    
    if submit_button and user_question.strip():
        result = call_agent_api(user_question.strip(), node_limit)
        
        if result:
            st.session_state.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
