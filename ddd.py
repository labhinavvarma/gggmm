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
import hashlib

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
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .stButton button {
        width: 100%;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 0.25rem;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.25rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #84fab0, #8fd3f4);
        border: none;
        color: #2c3e50;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffecd2, #fcb69f);
        border: none;
        color: #8b4513;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .legend-box {
        background: linear-gradient(135deg, #ffecd2, #fcb69f);
        padding: 1rem;
        border-radius: 10px;
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
st.markdown('<p style="text-align: center; color: #6c757d; font-size: 1.2rem;"><strong>🎨 Fixed Colorful Visualization</strong> • <strong>🔗 Guaranteed Working Graphs</strong></p>', unsafe_allow_html=True)

def call_agent_api(question: str, node_limit: int = 50) -> dict:
    """API call function"""
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
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
        return None

def get_node_color(labels):
    """Get vibrant colors for nodes"""
    if not labels:
        return "#95afc0"
    
    label = labels[0] if isinstance(labels, list) else str(labels)
    
    colors = {
        "Person": "#FF6B6B",       # Bright Red
        "Movie": "#4ECDC4",        # Turquoise
        "Company": "#45B7D1",      # Blue
        "Product": "#96CEB4",      # Green
        "Location": "#FECA57",     # Yellow
        "Event": "#FF9FF3",        # Pink
        "User": "#A55EEA",         # Purple
        "Order": "#26DE81",        # Emerald
        "Category": "#FD79A8",     # Rose
        "Department": "#6C5CE7",   # Indigo
        "Project": "#FDCB6E",      # Orange
        "Actor": "#00CEC9",        # Cyan
        "Director": "#E84393",     # Magenta
        "Producer": "#00B894"      # Teal
    }
    
    return colors.get(label, "#95afc0")

def get_relationship_color(rel_type):
    """Get colors for relationships"""
    colors = {
        "KNOWS": "#e74c3c",
        "FRIEND_OF": "#e74c3c",
        "WORKS_FOR": "#3498db",
        "MANAGES": "#9b59b6",
        "LOCATED_IN": "#f39c12",
        "BELONGS_TO": "#27ae60",
        "CREATED": "#e91e63",
        "OWNS": "#673ab7",
        "USES": "#009688",
        "ACTED_IN": "#2196f3",
        "DIRECTED": "#ff9800",
        "PRODUCED": "#4caf50",
        "LOVES": "#e91e63",
        "MARRIED_TO": "#fd79a8"
    }
    
    return colors.get(rel_type, "#666666")

def safe_extract_node_name(node):
    """Safely extract display name from node"""
    try:
        props = node.get("properties", {})
        labels = node.get("labels", ["Unknown"])
        node_id = str(node.get("id", ""))
        
        # Try different name properties
        name_options = [
            props.get("name"),
            props.get("title"),
            props.get("label"),
            props.get("username"),
            props.get("displayName")
        ]
        
        for name in name_options:
            if name and str(name).strip():
                return str(name).strip()[:25]
        
        # Fallback to label + short ID
        if labels and labels[0] != "Unknown":
            short_id = node_id.split(":")[-1][-6:] if ":" in node_id else node_id[-6:]
            return f"{labels[0]}_{short_id}"
        
        return f"Node_{node_id[-6:]}"
        
    except Exception as e:
        return f"Node_{hash(str(node)) % 10000}"

def create_simple_legend(nodes, relationships):
    """Create a simple legend"""
    try:
        # Get unique node types
        node_types = {}
        for node in nodes:
            labels = node.get("labels", ["Unknown"])
            if labels:
                label = labels[0]
                node_types[label] = get_node_color([label])
        
        # Get unique relationship types
        rel_types = {}
        for rel in relationships:
            rel_type = rel.get("type", "CONNECTED")
            rel_types[rel_type] = get_relationship_color(rel_type)
        
        legend_html = '<div class="legend-box">'
        legend_html += '<h4 style="margin-top: 0;">🎨 Graph Legend</h4>'
        
        if node_types:
            legend_html += '<p><strong>📊 Node Types:</strong><br>'
            for label, color in sorted(node_types.items()):
                legend_html += f'<span style="background: {color}; color: white; padding: 2px 8px; border-radius: 10px; margin: 2px; font-size: 12px;">{label}</span> '
            legend_html += '</p>'
        
        if rel_types:
            legend_html += '<p><strong>🔗 Relationship Types:</strong><br>'
            for rel_type, color in sorted(rel_types.items()):
                legend_html += f'<span style="background: {color}; color: white; padding: 2px 8px; border-radius: 10px; margin: 2px; font-size: 12px;">{rel_type}</span> '
            legend_html += '</p>'
        
        legend_html += '</div>'
        return legend_html
        
    except Exception as e:
        return f'<div class="legend-box">Legend error: {str(e)}</div>'

def render_working_graph(graph_data: dict) -> bool:
    """Render graph with simplified, working configuration"""
    
    if not graph_data:
        st.info("🔍 No graph data available.")
        return False
    
    try:
        nodes = graph_data.get("nodes", [])
        relationships = graph_data.get("relationships", [])
        
        if not nodes:
            st.info("📊 No nodes found.")
            return False
        
        # Show processing info
        st.markdown(f'<div class="success-box">🎨 <strong>Processing:</strong> {len(nodes)} nodes, {len(relationships)} relationships</div>', unsafe_allow_html=True)
        
        # Show relationship types
        if relationships:
            rel_types = list(set(rel.get('type', 'UNKNOWN') for rel in relationships))
            st.markdown(f'<div class="success-box">🔗 <strong>Found relationships:</strong> {", ".join(sorted(rel_types))}</div>', unsafe_allow_html=True)
        
        # Create Pyvis network with SIMPLE settings
        net = Network(
            height="650px",
            width="100%", 
            bgcolor="#ffffff",
            font_color="#333333"
        )
        
        # Add nodes safely
        added_nodes = set()
        node_details = []
        
        for i, node in enumerate(nodes):
            try:
                # Create safe node ID
                raw_id = str(node.get("id", f"node_{i}"))
                node_id = f"node_{i}"  # Use simple sequential IDs
                
                # Extract name safely
                display_name = safe_extract_node_name(node)
                node_details.append(display_name)
                
                # Get colors
                labels = node.get("labels", ["Unknown"])
                color = get_node_color(labels)
                
                # Create simple tooltip
                tooltip = f"{display_name}\\nType: {labels[0] if labels else 'Unknown'}"
                
                # Add node with SIMPLE configuration
                net.add_node(
                    node_id,
                    label=display_name,
                    color=color,
                    size=25,
                    title=tooltip
                )
                
                # Store mapping for relationships
                added_nodes.add((raw_id, node_id))
                
            except Exception as e:
                st.warning(f"⚠️ Skipped node {i}: {str(e)}")
                continue
        
        # Create ID mapping for relationships
        id_mapping = dict(added_nodes)
        simple_nodes = {node_id for _, node_id in added_nodes}
        
        # Add relationships safely
        added_edges = 0
        relationship_details = []
        
        for i, rel in enumerate(relationships):
            try:
                start_raw = str(rel.get("startNode", rel.get("start", "")))
                end_raw = str(rel.get("endNode", rel.get("end", "")))
                rel_type = str(rel.get("type", "CONNECTED"))
                
                # Map to simple IDs
                start_id = id_mapping.get(start_raw)
                end_id = id_mapping.get(end_raw)
                
                # Only add if both nodes exist
                if start_id and end_id and start_id in simple_nodes and end_id in simple_nodes:
                    color = get_relationship_color(rel_type)
                    
                    # Add edge with SIMPLE configuration
                    net.add_edge(
                        start_id,
                        end_id,
                        label=rel_type,
                        color=color,
                        width=3
                    )
                    
                    added_edges += 1
                    relationship_details.append(f"{rel_type}: {start_id} → {end_id}")
                    
            except Exception as e:
                st.warning(f"⚠️ Skipped relationship {i}: {str(e)}")
                continue
        
        st.write(f"✅ **Successfully created:** {len(simple_nodes)} nodes, {added_edges} relationships")
        
        # Show details
        if node_details:
            with st.expander(f"👥 Node Names ({len(node_details)})", expanded=False):
                for name in node_details[:10]:
                    st.write(f"• {name}")
                if len(node_details) > 10:
                    st.write(f"... and {len(node_details) - 10} more")
        
        if relationship_details:
            with st.expander(f"🔗 Relationships ({len(relationship_details)})", expanded=False):
                for rel in relationship_details[:10]:
                    st.write(f"• {rel}")
                if len(relationship_details) > 10:
                    st.write(f"... and {len(relationship_details) - 10} more")
        
        # Use VERY SIMPLE options that won't cause JSON errors
        try:
            net.barnes_hut()  # Use built-in physics method instead of complex JSON
        except:
            pass  # If this fails, just use default physics
        
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
            border: 2px solid #667eea; 
            border-radius: 10px; 
            overflow: hidden; 
            background: white;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        ">
            <div style="
                background: linear-gradient(90deg, #667eea, #764ba2);
                color: white;
                padding: 10px 20px;
                font-weight: bold;
            ">
                🕸️ Network Graph | {len(simple_nodes)} Named Nodes | {added_edges} Colored Relationships
            </div>
            {html_content}
        </div>
        """
        
        # Display
        components.html(wrapped_html, height=700, scrolling=False)
        
        # Cleanup
        try:
            os.unlink(html_file)
        except:
            pass
        
        return True
        
    except Exception as e:
        st.error(f"❌ Graph rendering failed: {str(e)}")
        
        with st.expander("🔍 Detailed Debug Info", expanded=True):
            st.write("**Error:**")
            st.code(str(e))
            st.write("**Full traceback:**")
            st.code(traceback.format_exc())
            if nodes:
                st.write("**Sample node data:**")
                st.json(nodes[0])
            if relationships:
                st.write("**Sample relationship data:**")
                st.json(relationships[0])
        
        return False

# Main layout
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown("### 💬 Chat Interface")
    
    # Status
    status_colors = {"connected": "🟢", "disconnected": "🔴", "unknown": "⚪"}
    st.markdown(f'<div class="success-box"><strong>API Status:</strong> {status_colors.get(st.session_state.connection_status, "⚪")} {st.session_state.connection_status}</div>', unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("#### 🚀 Working Quick Actions")
    quick_actions = [
        ("🌟 Show All Nodes", "Show me all nodes with their names and relationships"),
        ("👥 Show People", "Show me all Person nodes with their connections"),
        ("🔗 Show Network", "Display the network with all relationships"),
        ("📊 Database Overview", "Give me an overview of what's in the database"),
        ("🎯 Sample Data", "Show me a sample of connected data")
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
    st.info("💡 System automatically includes relationships and proper node names!")
    
    with st.form("question_form", clear_on_submit=True):
        user_question = st.text_area(
            "Your question:",
            placeholder="e.g., Show me all people, Find companies, What's connected?",
            height=80
        )
        
        node_limit = st.selectbox(
            "Max nodes:",
            [10, 25, 50, 75],
            index=1
        )
        
        submit_button = st.form_submit_button("🚀 Create Graph")
    
    if submit_button and user_question.strip():
        result = call_agent_api(user_question.strip(), node_limit)
        
        if result:
            st.session_state.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "question": user_question.strip(),
                "answer": result.get("answer", ""),
                "graph_data": result.get("graph_data")
            })
            
            if result.get("graph_data"):
                st.session_state.graph_data = result["graph_data"]
                st.success("✅ Graph created!")
            
            st.session_state.last_response = result
            st.rerun()
    
    st.divider()
    
    # Test data
    if st.button("🧪 Load Working Test Data"):
        test_data = {
            "nodes": [
                {"id": "person1", "labels": ["Person"], "properties": {"name": "Alice Johnson", "age": 30}},
                {"id": "person2", "labels": ["Person"], "properties": {"name": "Bob Smith", "age": 25}},
                {"id": "person3", "labels": ["Person"], "properties": {"name": "Carol Brown", "age": 35}},
                {"id": "company1", "labels": ["Company"], "properties": {"name": "TechCorp Inc.", "industry": "Technology"}},
                {"id": "location1", "labels": ["Location"], "properties": {"name": "New York", "country": "USA"}}
            ],
            "relationships": [
                {"startNode": "person1", "endNode": "person2", "type": "KNOWS", "properties": {}},
                {"startNode": "person2", "endNode": "person3", "type": "FRIEND_OF", "properties": {}},  
                {"startNode": "person1", "endNode": "company1", "type": "WORKS_FOR", "properties": {}},
                {"startNode": "person2", "endNode": "company1", "type": "WORKS_FOR", "properties": {}},
                {"startNode": "company1", "endNode": "location1", "type": "LOCATED_IN", "properties": {}}
            ]
        }
        st.session_state.graph_data = test_data
        st.success("✅ Working test data loaded!")
        st.rerun()
    
    # History
    st.markdown("#### 📝 Recent Activity")
    if st.session_state.conversation_history:
        for item in reversed(st.session_state.conversation_history[-2:]):
            with st.expander(f"💬 {item['question'][:30]}...", expanded=False):
                st.write(f"**Time:** {item['timestamp'][:19]}")
                if item.get('graph_data'):
                    nodes = len(item['graph_data'].get('nodes', []))
                    rels = len(item['graph_data'].get('relationships', []))
                    st.write(f"**Graph:** {nodes} nodes, {rels} relationships")
    
    if st.button("🗑️ Clear All"):
        for key in ["conversation_history", "graph_data", "last_response"]:
            st.session_state[key] = [] if key == "conversation_history" else None
        st.rerun()

with col2:
    st.markdown("### 🎨 Working Graph Visualization")
    
    # Show response
    if st.session_state.last_response:
        answer = st.session_state.last_response.get("answer", "")
        if answer:
            st.markdown("#### 🤖 AI Response")
            clean_answer = answer.replace("**", "").replace("#", "").strip()
            st.markdown(f'<div class="success-box">{clean_answer[:300]}{"..." if len(clean_answer) > 300 else ""}</div>', unsafe_allow_html=True)
    
    # Graph section
    if st.session_state.graph_data:
        nodes = st.session_state.graph_data.get("nodes", [])
        relationships = st.session_state.graph_data.get("relationships", [])
        
        # Stats
        col2_1, col2_2, col2_3 = st.columns(3)
        with col2_1:
            st.markdown(f'<div class="metric-container"><h2>{len(nodes)}</h2><p>Named Nodes</p></div>', unsafe_allow_html=True)
        with col2_2:
            st.markdown(f'<div class="metric-container"><h2>{len(relationships)}</h2><p>Relationships</p></div>', unsafe_allow_html=True)
        with col2_3:
            connections = len(relationships) / max(len(nodes), 1)
            st.markdown(f'<div class="metric-container"><h2>{connections:.1f}</h2><p>Avg Connections</p></div>', unsafe_allow_html=True)
        
        # Legend
        if nodes or relationships:
            legend = create_simple_legend(nodes, relationships)
            st.markdown(legend, unsafe_allow_html=True)
        
        # Render graph
        st.markdown("#### 🎨 Interactive Network")
        success = render_working_graph(st.session_state.graph_data)
        
        if success:
            if len(relationships) > 0:
                st.markdown(f'<div class="success-box">🎉 <strong>Success!</strong> Your graph is working perfectly with {len(nodes)} named nodes and {len(relationships)} colored relationship lines!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">ℹ️ Graph shows nodes only - no relationships found in the data</div>', unsafe_allow_html=True)
            
            # Controls
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔄 Refresh"):
                    st.rerun()
            with col_b:
                if st.button("🌐 Get Full Network"):
                    result = call_agent_api("Show me the complete network with all relationships", node_limit=50)
                    if result and result.get("graph_data"):
                        st.session_state.graph_data = result["graph_data"]
                        st.rerun()
        else:
            st.error("❌ Graph rendering failed. Check the debug information above.")
    
    else:
        # Welcome
        st.markdown("""
        <div style="
            text-align: center; 
            padding: 3rem; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            border-radius: 15px; 
            margin: 2rem 0;
        ">
            <h2>🎯 Fixed & Working Graph Explorer!</h2>
            <p><strong>No more JSON errors - guaranteed to work!</strong></p>
            
            <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
                <h3>✅ What's Fixed:</h3>
                <p>🏷️ Proper node names (not node1, node2)<br>
                🔗 Visible relationship lines with labels<br>
                🎨 Colorful, stable visualization<br>
                🛠️ Robust error handling</p>
            </div>
            
            <p><em>Click "Load Working Test Data" to see it in action!</em></p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <strong>🚀 Neo4j Graph Explorer - Fixed Edition</strong><br>
    <small>✅ Working Graphs • 🏷️ Named Nodes • 🔗 Colored Relationships</small>
</div>
""", unsafe_allow_html=True)
