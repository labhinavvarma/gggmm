"""
Simplified Neo4j Graph Visualization - Fixed Version
Removes problematic settings and focuses on clean graph display
"""

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

# Simplified CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        max-width: 100%;
    }
    
    .graph-container {
        border: 2px solid #00BCD4;
        border-radius: 10px;
        overflow: hidden;
        background: white;
        margin: 1rem 0;
    }
    
    .graph-header {
        background: #00857C;
        color: white;
        padding: 1rem;
        font-weight: bold;
    }
    
    .metric-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .success-msg {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-msg {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.title("🕸️ Neo4j Graph Explorer")
st.write("Clean and simple graph visualization")

def get_simple_colors():
    """Simple color palette for nodes"""
    return {
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
        "Technology": "#00CEC9",
        "default": "#95A5A6"
    }

def extract_node_name(node):
    """Extract display name from node properties"""
    try:
        props = node.get("properties", {})
        labels = node.get("labels", ["Node"])
        
        # Try common name fields
        for field in ["name", "title", "fullName", "username", "label"]:
            if field in props and props[field]:
                return str(props[field])[:30]
        
        # Fallback to label + ID
        label = labels[0] if labels else "Node"
        node_id = str(node.get("id", ""))[-6:]
        return f"{label}_{node_id}"
        
    except:
        return f"Node_{hash(str(node)) % 1000}"

def create_simple_graph(graph_data):
    """Create a simple, working graph visualization"""
    
    if not graph_data:
        st.info("No graph data available")
        return False
    
    try:
        nodes = graph_data.get("nodes", [])
        relationships = graph_data.get("relationships", [])
        
        if not nodes:
            st.info("No nodes found in data")
            return False
        
        st.write(f"**Creating graph:** {len(nodes)} nodes, {len(relationships)} relationships")
        
        # Create simple network with minimal settings
        net = Network(
            height="600px",
            width="100%",
            bgcolor="#ffffff",
            font_color="#333333"
        )
        
        # Disable physics to prevent loading issues
        net.set_options("""
        var options = {
          "physics": {
            "enabled": false
          },
          "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
          }
        }
        """)
        
        # Add nodes with simple styling
        colors = get_simple_colors()
        node_mapping = {}
        
        for i, node in enumerate(nodes):
            try:
                node_id = f"n{i}"
                display_name = extract_node_name(node)
                
                # Get node color
                labels = node.get("labels", ["Unknown"])
                main_label = labels[0] if labels else "Unknown"
                color = colors.get(main_label, colors["default"])
                
                # Simple tooltip
                props = node.get("properties", {})
                tooltip = f"<b>{display_name}</b><br>Type: {main_label}"
                
                # Add key properties to tooltip
                for key in ["id", "age", "role", "email"]:
                    if key in props and props[key]:
                        tooltip += f"<br>{key}: {props[key]}"
                
                net.add_node(
                    node_id,
                    label=display_name,
                    color=color,
                    size=20,
                    title=tooltip
                )
                
                # Store mapping for relationships
                original_id = str(node.get("id", f"node_{i}"))
                node_mapping[original_id] = node_id
                
            except Exception as e:
                st.warning(f"Skipped node {i}: {str(e)}")
                continue
        
        # Add relationships
        added_edges = 0
        for i, rel in enumerate(relationships):
            try:
                start_id = str(rel.get("startNode", rel.get("start", "")))
                end_id = str(rel.get("endNode", rel.get("end", "")))
                rel_type = str(rel.get("type", "CONNECTED"))
                
                start_node = node_mapping.get(start_id)
                end_node = node_mapping.get(end_id)
                
                if start_node and end_node:
                    net.add_edge(
                        start_node,
                        end_node,
                        label=rel_type,
                        color="#666666",
                        width=2
                    )
                    added_edges += 1
                    
            except Exception as e:
                st.warning(f"Skipped relationship {i}: {str(e)}")
                continue
        
        st.success(f"✅ Graph created: {len(node_mapping)} nodes, {added_edges} relationships")
        
        # Generate HTML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            net.save_graph(f.name)
            html_file = f.name
        
        # Read and display HTML
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Simple wrapper
        wrapped_html = f"""
        <div class="graph-container">
            <div class="graph-header">
                Interactive Graph: {len(node_mapping)} Nodes, {added_edges} Relationships
            </div>
            {html_content}
        </div>
        """
        
        components.html(wrapped_html, height=650, scrolling=False)
        
        # Cleanup
        try:
            os.unlink(html_file)
        except:
            pass
        
        return True
        
    except Exception as e:
        st.error(f"❌ Graph creation failed: {str(e)}")
        st.code(traceback.format_exc())
        return False

def call_agent_api(question):
    """Call the agent API"""
    try:
        response = requests.post(
            "http://localhost:8081/chat",
            json={
                "question": question,
                "session_id": st.session_state.session_id,
                "node_limit": 1000
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# Main interface
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("💬 Query Interface")
    
    # Simple suggestions
    if st.button("Show all nodes", use_container_width=True):
        result = call_agent_api("Show me all nodes and relationships")
        if result and result.get("graph_data"):
            st.session_state.graph_data = result["graph_data"]
            st.rerun()
    
    if st.button("Database schema", use_container_width=True):
        result = call_agent_api("Show me the database schema")
        if result and result.get("graph_data"):
            st.session_state.graph_data = result["graph_data"]
            st.rerun()
    
    # Custom query input
    with st.form("query_form"):
        question = st.text_area("Your question:", height=100)
        submit = st.form_submit_button("Execute Query", use_container_width=True)
    
    if submit and question:
        result = call_agent_api(question)
        if result:
            if result.get("graph_data"):
                st.session_state.graph_data = result["graph_data"]
            st.success("Query executed!")
            st.rerun()
    
    # Test data button
    if st.button("Load Test Data", use_container_width=True):
        test_data = {
            "nodes": [
                {"id": "1", "labels": ["Person"], "properties": {"name": "Alice", "age": 30}},
                {"id": "2", "labels": ["Person"], "properties": {"name": "Bob", "age": 25}},
                {"id": "3", "labels": ["Company"], "properties": {"name": "TechCorp"}},
                {"id": "4", "labels": ["Project"], "properties": {"name": "AI Project"}}
            ],
            "relationships": [
                {"startNode": "1", "endNode": "2", "type": "KNOWS"},
                {"startNode": "1", "endNode": "3", "type": "WORKS_FOR"},
                {"startNode": "2", "endNode": "3", "type": "WORKS_FOR"},
                {"startNode": "1", "endNode": "4", "type": "ASSIGNED_TO"}
            ]
        }
        st.session_state.graph_data = test_data
        st.success("Test data loaded!")
        st.rerun()

with col2:
    st.subheader("🎨 Graph Visualization")
    
    if st.session_state.graph_data:
        nodes = st.session_state.graph_data.get("nodes", [])
        relationships = st.session_state.graph_data.get("relationships", [])
        
        # Show metrics
        col2_1, col2_2, col2_3 = st.columns(3)
        with col2_1:
            st.metric("Nodes", len(nodes))
        with col2_2:
            st.metric("Relationships", len(relationships))
        with col2_3:
            connectivity = len(relationships) / max(len(nodes), 1) if nodes else 0
            st.metric("Connectivity", f"{connectivity:.1f}")
        
        # Create and display graph
        create_simple_graph(st.session_state.graph_data)
        
    else:
        st.info("👈 Load data or execute a query to see the graph visualization")
        
        # Show instructions
        st.markdown("""
        **Getting Started:**
        1. Click "Load Test Data" to see a sample graph
        2. Use the suggestion buttons for common queries
        3. Enter your own questions in the text area
        4. The graph will appear here once data is loaded
        """)

st.markdown("---")
st.caption(f"Session: {st.session_state.session_id[:8]}... | Simple Graph Explorer")
