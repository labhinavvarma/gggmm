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
import random

# Page configuration
st.set_page_config(
    page_title="Neo4j Graph Explorer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better visuals
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
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .success-box {
        background: linear-gradient(135deg, #84fab0, #8fd3f4);
        border: none;
        color: #2c3e50;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(132, 250, 176, 0.3);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffecd2, #fcb69f);
        border: none;
        color: #8b4513;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(255, 236, 210, 0.3);
    }
    
    .legend-box {
        background: linear-gradient(135deg, #ffecd2, #fcb69f);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .graph-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 20px;
        font-weight: bold;
        font-size: 16px;
        border-radius: 8px 8px 0 0;
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

# Header with gradient
st.markdown('<h1 class="main-header">🕸️ Neo4j Graph Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6c757d; font-size: 1.2rem;"><strong>🎨 Colorful Network Visualization</strong> • <strong>🔗 Named Nodes & Relationships</strong></p>', unsafe_allow_html=True)

def call_agent_api(question: str, node_limit: int = 50) -> dict:
    """Enhanced API call with better error handling"""
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
        st.error("⏰ Request timed out. Try a simpler query.")
        return None
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
        return None

def get_vibrant_node_color(labels, node_id):
    """Get vibrant colors for nodes with better variety"""
    if not labels:
        return "#95afc0"
    
    label = labels[0] if isinstance(labels, list) else str(labels)
    
    # Vibrant color palette
    vibrant_colors = {
        "Person": "#FF6B6B",       # Bright Red
        "Movie": "#4ECDC4",        # Turquoise
        "Company": "#45B7D1",      # Sky Blue
        "Product": "#96CEB4",      # Mint Green
        "Location": "#FECA57",     # Golden Yellow
        "Event": "#FF9FF3",        # Hot Pink
        "User": "#A55EEA",         # Purple
        "Order": "#26DE81",        # Emerald
        "Category": "#FD79A8",     # Rose
        "Department": "#6C5CE7",   # Indigo
        "Project": "#FDCB6E",      # Orange
        "Actor": "#00CEC9",        # Cyan
        "Director": "#E84393",     # Magenta
        "Producer": "#00B894",     # Teal
        "Customer": "#FF7675",     # Light Red
        "Employee": "#74B9FF",     # Light Blue
        "Manager": "#A29BFE",      # Light Purple
        "Task": "#FD79A8",         # Pink
        "Team": "#FDCB6E"          # Amber
    }
    
    # If label not in map, generate color based on hash for consistency
    if label not in vibrant_colors:
        # Generate consistent color based on label hash
        hash_val = hash(label) % 360
        return f"hsl({hash_val}, 70%, 60%)"
    
    return vibrant_colors[label]

def get_vibrant_relationship_color(rel_type):
    """Get vibrant colors for relationships"""
    vibrant_rel_colors = {
        "KNOWS": "#e74c3c",           # Strong Red
        "WORKS_FOR": "#3498db",       # Strong Blue
        "MANAGES": "#9b59b6",         # Purple
        "LOCATED_IN": "#f39c12",      # Orange
        "BELONGS_TO": "#27ae60",      # Green
        "CREATED": "#e91e63",         # Pink
        "OWNS": "#673ab7",            # Deep Purple
        "USES": "#009688",            # Teal
        "MEMBER_OF": "#ff5722",       # Deep Orange
        "ASSIGNED_TO": "#795548",     # Brown
        "REPORTS_TO": "#607d8b",      # Blue Grey
        "ACTED_IN": "#2196f3",        # Blue
        "DIRECTED": "#ff9800",        # Amber
        "PRODUCED": "#4caf50",        # Light Green
        "LOVES": "#e91e63",           # Pink
        "FRIENDS_WITH": "#ff6b6b",    # Light Red
        "MARRIED_TO": "#fd79a8",      # Rose
        "WORKS_AT": "#74b9ff",        # Light Blue
        "LIVES_IN": "#fdcb6e",        # Yellow
        "STUDIED_AT": "#a29bfe"       # Light Purple
    }
    
    # Generate color for unknown relationship types
    if rel_type not in vibrant_rel_colors:
        hash_val = hash(rel_type) % 360
        return f"hsl({hash_val}, 80%, 50%)"
    
    return vibrant_rel_colors[rel_type]

def extract_node_name(node):
    """Extract the best display name for a node"""
    props = node.get("properties", {})
    labels = node.get("labels", ["Unknown"])
    node_id = str(node.get("id", ""))
    
    # Try different property names for display
    name_candidates = [
        props.get("name"),
        props.get("title"), 
        props.get("label"),
        props.get("username"),
        props.get("firstName"),
        props.get("fullName"),
        props.get("companyName"),
        props.get("displayName")
    ]
    
    # Use first non-empty candidate
    for candidate in name_candidates:
        if candidate and str(candidate).strip():
            return str(candidate).strip()[:25]  # Limit length
    
    # If no name properties, use label + ID
    if labels and labels[0] != "Unknown":
        return f"{labels[0]}_{node_id[-4:]}"  # e.g., "Person_1234"
    
    return f"Node_{node_id[-4:]}"

def create_colorful_legend(nodes, relationships):
    """Create a beautiful colorful legend"""
    try:
        # Get unique node types with colors
        node_types = {}
        for node in nodes:
            labels = node.get("labels", ["Unknown"])
            if labels:
                label = labels[0]
                if label not in node_types:
                    node_types[label] = get_vibrant_node_color([label], "")
        
        # Get unique relationship types with colors
        rel_types = {}
        for rel in relationships:
            rel_type = rel.get("type", "CONNECTED")
            if rel_type not in rel_types:
                rel_types[rel_type] = get_vibrant_relationship_color(rel_type)
        
        legend_html = '<div class="legend-box">'
        legend_html += '<h3 style="margin-top: 0; color: #2c3e50;">🎨 Graph Legend</h3>'
        
        if node_types:
            legend_html += '<div style="margin-bottom: 15px;"><strong style="color: #2c3e50;">📊 Node Types:</strong><br>'
            for label, color in sorted(node_types.items()):
                legend_html += f'''
                <span style="
                    display: inline-block; 
                    background: {color}; 
                    color: white; 
                    padding: 4px 8px; 
                    border-radius: 15px; 
                    margin: 2px; 
                    font-size: 12px;
                    font-weight: bold;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                ">{label}</span>
                '''
            legend_html += '</div>'
        
        if rel_types:
            legend_html += '<div><strong style="color: #2c3e50;">🔗 Relationship Types:</strong><br>'
            for rel_type, color in sorted(rel_types.items()):
                legend_html += f'''
                <span style="
                    display: inline-block; 
                    background: {color}; 
                    color: white; 
                    padding: 4px 8px; 
                    border-radius: 15px; 
                    margin: 2px; 
                    font-size: 12px;
                    font-weight: bold;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                ">{rel_type}</span>
                '''
            legend_html += '</div>'
        
        legend_html += '</div>'
        return legend_html
        
    except Exception as e:
        return f'<div class="legend-box">Legend error: {str(e)}</div>'

def render_colorful_stable_graph(graph_data: dict) -> bool:
    """Render a colorful, stable graph with proper names and relationships"""
    
    if not graph_data:
        st.info("🔍 No graph data available.")
        return False
    
    try:
        nodes = graph_data.get("nodes", [])
        relationships = graph_data.get("relationships", [])
        
        if not nodes:
            st.info("📊 No nodes found in the data.")
            return False
        
        # Enhanced processing info
        st.markdown(f'<div class="success-box">🎨 <strong>Creating colorful graph:</strong> {len(nodes)} nodes, {len(relationships)} relationships</div>', unsafe_allow_html=True)
        
        # Show detailed relationship info
        if relationships:
            rel_types = list(set(rel.get('type', 'UNKNOWN') for rel in relationships))
            st.markdown(f'<div class="success-box">🔗 <strong>Relationship Types:</strong> {", ".join(sorted(rel_types))}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">⚠️ <strong>No relationships found</strong> - Graph will show isolated nodes</div>', unsafe_allow_html=True)
        
        # Create enhanced Pyvis network
        net = Network(
            height="700px",
            width="100%", 
            bgcolor="#f8f9fa",
            font_color="#2c3e50",
            directed=True,
            select_menu=True,
            filter_menu=True
        )
        
        # Add nodes with proper names and vibrant colors
        added_nodes = set()
        node_details = []
        
        for i, node in enumerate(nodes):
            try:
                node_id = str(node.get("id", f"node_{i}"))
                if node_id in added_nodes:
                    node_id = f"{node_id}_{i}"
                
                props = node.get("properties", {})
                labels = node.get("labels", ["Unknown"])
                
                # Extract proper display name
                display_name = extract_node_name(node)
                node_details.append(f"{display_name} ({labels[0] if labels else 'Unknown'})")
                
                # Create rich tooltip
                tooltip = f"🏷️ {display_name}\\n📋 Type: {labels[0] if labels else 'Unknown'}\\n🆔 ID: {node_id}"
                if props:
                    tooltip += f"\\n📊 Properties: {len(props)}"
                    # Show key properties in tooltip
                    for key, value in list(props.items())[:4]:
                        if key and value:
                            tooltip += f"\\n  • {key}: {str(value)[:30]}"
                
                # Get vibrant color and size
                color = get_vibrant_node_color(labels, node_id)
                size = 35 + len(props) * 3  # Larger nodes with more properties
                size = min(size, 60)  # Cap maximum size
                
                # Add node with enhanced styling
                net.add_node(
                    node_id,
                    label=display_name,
                    title=tooltip,
                    color={
                        'background': color,
                        'border': '#2c3e50',
                        'highlight': {'background': color, 'border': '#e74c3c'},
                        'hover': {'background': color, 'border': '#f39c12'}
                    },
                    size=size,
                    font={
                        'size': 16, 
                        'color': '#2c3e50',
                        'face': 'Arial',
                        'strokeWidth': 2,
                        'strokeColor': '#ffffff'
                    },
                    borderWidth=3,
                    borderWidthSelected=5,
                    shadow={'enabled': True, 'color': 'rgba(0,0,0,0.3)', 'size': 8}
                )
                added_nodes.add(node_id)
                
            except Exception as e:
                st.warning(f"⚠️ Skipped node {i}: {str(e)}")
                continue
        
        # Add relationships with vibrant colors and labels
        added_edges = 0
        relationship_details = []
        
        for i, rel in enumerate(relationships):
            try:
                start_id = str(rel.get("startNode", rel.get("start", "")))
                end_id = str(rel.get("endNode", rel.get("end", "")))
                rel_type = str(rel.get("type", "CONNECTED"))
                rel_props = rel.get("properties", {})
                
                # Only add if both nodes exist
                if start_id in added_nodes and end_id in added_nodes:
                    # Get vibrant color for relationship
                    color = get_vibrant_relationship_color(rel_type)
                    
                    # Create rich tooltip for relationship
                    edge_tooltip = f"🔗 {rel_type}\\n📍 {start_id} → {end_id}"
                    if rel_props:
                        edge_tooltip += f"\\n📊 Properties: {len(rel_props)}"
                        for key, value in list(rel_props.items())[:3]:
                            if key and value:
                                edge_tooltip += f"\\n  • {key}: {str(value)[:25]}"
                    
                    # Add relationship with enhanced styling
                    net.add_edge(
                        start_id,
                        end_id,
                        label=rel_type,
                        title=edge_tooltip,
                        color={
                            'color': color,
                            'highlight': '#e74c3c',
                            'hover': '#f39c12',
                            'inherit': False
                        },
                        width=5,  # Thicker for visibility
                        arrows={
                            'to': {
                                'enabled': True, 
                                'scaleFactor': 1.8,
                                'type': 'arrow'
                            }
                        },
                        font={
                            'size': 14, 
                            'color': color,
                            'face': 'Arial',
                            'strokeWidth': 3, 
                            'strokeColor': '#ffffff',
                            'align': 'middle'
                        },
                        smooth={
                            'enabled': True,
                            'type': 'continuous',
                            'roundness': 0.2
                        },
                        shadow={'enabled': True, 'color': 'rgba(0,0,0,0.2)', 'size': 4}
                    )
                    added_edges += 1
                    relationship_details.append(f"{rel_type}: {start_id} → {end_id}")
                    
            except Exception as e:
                st.warning(f"⚠️ Skipped relationship {i}: {str(e)}")
                continue
        
        # Success message with details
        st.write(f"✅ **Graph created successfully:** {len(added_nodes)} colorful nodes, {added_edges} relationship lines")
        
        # Show node details
        if node_details:
            with st.expander(f"👥 Node Details ({len(node_details)} nodes)", expanded=False):
                for detail in node_details[:15]:  # Show first 15
                    st.write(f"• {detail}")
                if len(node_details) > 15:
                    st.write(f"... and {len(node_details) - 15} more nodes")
        
        # Show relationship details
        if relationship_details:
            with st.expander(f"🔗 Relationship Details ({len(relationship_details)} connections)", expanded=False):
                for detail in relationship_details[:15]:  # Show first 15
                    st.write(f"• {detail}")
                if len(relationship_details) > 15:
                    st.write(f"... and {len(relationship_details) - 15} more relationships")
        
        # Enhanced physics for stable, attractive layout
        net.set_options("""
        var options = {
            "physics": {
                "enabled": true,
                "stabilization": {
                    "enabled": true,
                    "iterations": 200,
                    "updateInterval": 25,
                    "onlyDynamicEdges": false,
                    "fit": true
                },
                "barnesHut": {
                    "gravitationalConstant": -8000,
                    "centralGravity": 0.15,
                    "springLength": 180,
                    "springConstant": 0.04,
                    "damping": 0.09,
                    "avoidOverlap": 0.3
                },
                "maxVelocity": 50,
                "minVelocity": 0.1,
                "timestep": 0.35
            },
            "interaction": {
                "dragNodes": true,
                "dragView": true,
                "zoomView": true,
                "selectConnectedEdges": true,
                "hover": true,
                "hoverConnectedEdges": true,
                "tooltipDelay": 200,
                "hideEdgesOnDrag": false,
                "hideNodesOnDrag": false
            },
            "nodes": {
                "borderWidth": 3,
                "borderWidthSelected": 5,
                "shadow": {
                    "enabled": true,
                    "color": 'rgba(0,0,0,0.3)',
                    "size": 8,
                    "x": 2,
                    "y": 2
                }
            },
            "edges": {
                "width": 5,
                "selectionWidth": 8,
                "hoverWidth": 8,
                "shadow": {
                    "enabled": true,
                    "color": 'rgba(0,0,0,0.2)',
                    "size": 4,
                    "x": 1,
                    "y": 1
                },
                "smooth": {
                    "enabled": true,
                    "type": "continuous",
                    "roundness": 0.2
                }
            },
            "layout": {
                "improvedLayout": true,
                "clusterThreshold": 150,
                "hierarchical": {
                    "enabled": false
                }
            }
        }
        """)
        
        # Generate and display HTML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            net.save_graph(f.name)
            html_file = f.name
        
        # Read HTML content
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Enhanced wrapper with gradient header
        wrapped_html = f"""
        <div style="
            border: 3px solid transparent;
            border-radius: 12px; 
            overflow: hidden; 
            background: linear-gradient(white, white) padding-box, 
                        linear-gradient(45deg, #667eea, #764ba2) border-box;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        ">
            <div class="graph-header">
                🕸️ Interactive Colorful Graph | {len(added_nodes)} Named Nodes | {added_edges} Colored Relationships
            </div>
            <div style="background: white; padding: 5px;">
                {html_content}
            </div>
        </div>
        """
        
        # Display with enhanced height
        components.html(wrapped_html, height=750, scrolling=False)
        
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
            st.write("**Graph Data Sample:**")
            if nodes:
                st.write("**Sample Node:**")
                st.json(nodes[0])
            if relationships:
                st.write("**Sample Relationship:**")
                st.json(relationships[0])
        
        return False

# Main layout
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown("### 💬 Enhanced Chat Interface")
    
    # Connection status with style
    status_colors = {"connected": "🟢", "disconnected": "🔴", "error": "🟡", "unknown": "⚪"}
    st.markdown(f'<div class="success-box"><strong>Status:</strong> {status_colors.get(st.session_state.connection_status, "⚪")} {st.session_state.connection_status}</div>', unsafe_allow_html=True)
    
    # Enhanced quick actions
    st.markdown("#### 🚀 Quick Actions")
    quick_actions = [
        ("🌟 Show All Network", "Show me all nodes with their names and relationships"),
        ("👥 People Network", "Show me all Person nodes with their names and connections"),
        ("🏢 Company Network", "Show me Company nodes with their relationships"),
        ("📊 Full Database", "Display the complete database with all connections"),
        ("🎯 Sample Network", "Give me a colorful sample of connected data"),
        ("📋 Database Schema", "What types of nodes and relationships exist?")
    ]
    
    for action_name, action_query in quick_actions:
        if st.button(action_name, key=f"quick_{action_name}"):
            result = call_agent_api(action_query, node_limit=40)
            if result:
                st.session_state.conversation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "question": action_query,
                    "answer": result.get("answer", ""),
                    "graph_data": result.get("graph_data")
                })
                
                if result.get("graph_data"):
                    st.session_state.graph_data = result["graph_data"]
                    st.success("✅ Colorful graph updated!")
                
                st.session_state.last_response = result
                st.rerun()
    
    st.divider()
    
    # Enhanced question input
    st.markdown("#### ✍️ Ask Your Question")
    st.info("💡 **Pro tip:** I automatically enhance queries to show node names and relationships!")
    
    with st.form("question_form", clear_on_submit=True):
        user_question = st.text_area(
            "Your question:",
            placeholder="e.g., Show me all Person nodes, Find companies and their connections, Who knows who?",
            height=90
        )
        
        node_limit = st.select_slider(
            "🎨 Visualization size:",
            options=[10, 25, 50, 75, 100],
            value=50,
            help="Smaller = clearer relationships, Larger = more data"
        )
        
        submit_button = st.form_submit_button("🚀 Create Colorful Graph", use_container_width=True)
    
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
                st.balloons()  # Celebration animation
                st.success("✅ Beautiful graph created!")
            
            st.session_state.last_response = result
            st.rerun()
    
    st.divider()
    
    # Enhanced test data
    if st.button("🧪 Load Colorful Test Data", use_container_width=True):
        colorful_test_data = {
            "nodes": [
                {"id": "alice_1", "labels": ["Person"], "properties": {"name": "Alice Johnson", "age": 30, "department": "Engineering", "role": "Senior Developer"}},
                {"id": "bob_2", "labels": ["Person"], "properties": {"name": "Bob Smith", "age": 25, "department": "Marketing", "role": "Designer"}},
                {"id": "carol_3", "labels": ["Person"], "properties": {"name": "Carol Williams", "age": 35, "department": "Engineering", "role": "Manager"}},
                {"id": "techcorp_1", "labels": ["Company"], "properties": {"name": "TechCorp Inc.", "founded": 2010, "employees": 500, "industry": "Technology"}},
                {"id": "nyc_1", "labels": ["Location"], "properties": {"name": "New York City", "country": "USA", "population": 8000000}},
                {"id": "ai_project_1", "labels": ["Project"], "properties": {"name": "AI Innovation Project", "status": "Active", "budget": 1000000}},
                {"id": "eng_dept_1", "labels": ["Department"], "properties": {"name": "Engineering Department", "headcount": 50, "budget": 5000000}}
            ],
            "relationships": [
                {"startNode": "alice_1", "endNode": "bob_2", "type": "KNOWS", "properties": {"since": "2020", "relationship": "colleague"}},
                {"startNode": "bob_2", "endNode": "carol_3", "type": "REPORTS_TO", "properties": {"since": "2021"}},
                {"startNode": "alice_1", "endNode": "techcorp_1", "type": "WORKS_FOR", "properties": {"position": "Senior Developer", "salary": 120000}},
                {"startNode": "bob_2", "endNode": "techcorp_1", "type": "WORKS_FOR", "properties": {"position": "Designer", "salary": 85000}},
                {"startNode": "carol_3", "endNode": "techcorp_1", "type": "WORKS_FOR", "properties": {"position": "Engineering Manager", "salary": 150000}},
                {"startNode": "techcorp_1", "endNode": "nyc_1", "type": "LOCATED_IN", "properties": {"headquarters": True}},
                {"startNode": "alice_1", "endNode": "ai_project_1", "type": "ASSIGNED_TO", "properties": {"role": "Technical Lead"}},
                {"startNode": "bob_2", "endNode": "ai_project_1", "type": "ASSIGNED_TO", "properties": {"role": "UI Designer"}},
                {"startNode": "carol_3", "endNode": "ai_project_1", "type": "MANAGES", "properties": {"budget_authority": True}},
                {"startNode": "alice_1", "endNode": "eng_dept_1", "type": "MEMBER_OF", "properties": {}},
                {"startNode": "carol_3", "endNode": "eng_dept_1", "type": "MANAGES", "properties": {"team_size": 15}},
                {"startNode": "eng_dept_1", "endNode": "techcorp_1", "type": "BELONGS_TO", "properties": {}}
            ]
        }
        st.session_state.graph_data = colorful_test_data
        st.balloons()
        st.success("✅ Colorful test data loaded! 12 relationships with proper names!")
        st.rerun()
    
    # Conversation history
    st.markdown("#### 📝 Recent Conversations")
    if st.session_state.conversation_history:
        for item in reversed(st.session_state.conversation_history[-3:]):
            with st.expander(f"💬 {item['question'][:35]}...", expanded=False):
                st.write(f"**⏰ Time:** {item['timestamp'][:19]}")
                if item.get('graph_data'):
                    nodes = len(item['graph_data'].get('nodes', []))
                    rels = len(item['graph_data'].get('relationships', []))
                    st.write(f"**📊 Graph:** {nodes} nodes, {rels} relationships")
                answer_preview = item['answer'].replace("**", "").replace("#", "")[:150]
                st.write(f"**💭 Response:** {answer_preview}...")
    else:
        st.info("💡 No conversations yet. Try the colorful quick actions above!")
    
    if st.button("🗑️ Clear Everything", use_container_width=True):
        for key in ["conversation_history", "graph_data", "last_response"]:
            st.session_state[key] = [] if key == "conversation_history" else None
        st.rerun()

with col2:
    st.markdown("### 🎨 Colorful Network Graph")
    
    # Show latest response with styling
    if st.session_state.last_response:
        answer = st.session_state.last_response.get("answer", "")
        if answer:
            st.markdown("#### 🤖 AI Response")
            clean_answer = answer.replace("**", "").replace("#", "").strip()
            st.markdown(f'<div class="success-box">{clean_answer[:400]}{"..." if len(clean_answer) > 400 else ""}</div>', unsafe_allow_html=True)
    
    # Enhanced graph visualization
    if st.session_state.graph_data:
        nodes = st.session_state.graph_data.get("nodes", [])
        relationships = st.session_state.graph_data.get("relationships", [])
        
        # Colorful statistics
        col2_1, col2_2, col2_3 = st.columns(3)
        with col2_1:
            st.markdown(f'<div class="metric-container"><h2>{len(nodes)}</h2><p>Named Nodes</p></div>', unsafe_allow_html=True)
        with col2_2:
            st.markdown(f'<div class="metric-container"><h2>{len(relationships)}</h2><p>Relationships</p></div>', unsafe_allow_html=True)
        with col2_3:
            connectivity = len(relationships) / max(len(nodes), 1)
            st.markdown(f'<div class="metric-container"><h2>{connectivity:.1f}</h2><p>Avg Connections</p></div>', unsafe_allow_html=True)
        
        # Display colorful legend
        if nodes or relationships:
            legend = create_colorful_legend(nodes, relationships)
            st.markdown(legend, unsafe_allow_html=True)
        
        # Render the colorful stable graph
        st.markdown("#### 🎨 Interactive Colorful Network")
        success = render_colorful_stable_graph(st.session_state.graph_data)
        
        if success:
            if len(relationships) > 0:
                st.markdown(f'<div class="success-box">🎉 <strong>Success!</strong> Displaying {len(nodes)} named nodes connected by {len(relationships)} colorful relationship lines!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">⚠️ <strong>Isolated nodes detected</strong> - No relationships found in the data</div>', unsafe_allow_html=True)
            
            # Enhanced controls
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("🔄 Refresh Graph", use_container_width=True):
                    st.rerun()
            with col_b:
                if st.button("🌐 Get Full Network", use_container_width=True):
                    result = call_agent_api("Show me the complete colorful network with all named nodes and relationships", node_limit=60)
                    if result and result.get("graph_data"):
                        st.session_state.graph_data = result["graph_data"]
                        st.rerun()
            with col_c:
                if st.button("📊 Network Stats", use_container_width=True):
                    # Calculate network statistics
                    node_types = {}
                    for node in nodes:
                        labels = node.get('labels', ['Unknown'])
                        label = labels[0] if labels else 'Unknown'
                        node_types[label] = node_types.get(label, 0) + 1
                    
                    stats_text = f"**Node Types:** " + ", ".join([f"{k}({v})" for k, v in node_types.items()])
                    if relationships:
                        rel_types = {}
                        for rel in relationships:
                            rel_type = rel.get('type', 'UNKNOWN')
                            rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
                        stats_text += f"\n**Relationship Types:** " + ", ".join([f"{k}({v})" for k, v in rel_types.items()])
                    
                    st.info(stats_text)
        else:
            st.error("❌ Graph rendering failed. Check debug information above.")
    
    else:
        # Colorful welcome screen
        st.markdown("""
        <div style="
            text-align: center; 
            padding: 3rem; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            border-radius: 15px; 
            margin: 2rem 0;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
        ">
            <h2>🎨 Welcome to Colorful Graph Explorer!</h2>
            <p style="font-size: 1.1rem;"><strong>Experience beautiful network visualization</strong></p>
            
            <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
                <h3>✨ What Makes This Special:</h3>
                <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem; margin-top: 1rem;">
                    <div>🎯 <strong>Named Nodes</strong><br>Real names, not IDs</div>
                    <div>🌈 <strong>Vibrant Colors</strong><br>Beautiful color coding</div>
                    <div>🔗 <strong>Visible Relationships</strong><br>Clear connection lines</div>
                    <div>🖱️ <strong>Interactive</strong><br>Drag, zoom, explore</div>
                </div>
            </div>
            
            <p><em>👆 Click "Load Colorful Test Data" to see the magic!</em></p>
        </div>
        """, unsafe_allow_html=True)

# Enhanced footer
st.markdown("---")
st.markdown("""
<div style="
    text-align: center; 
    color: #6c757d; 
    padding: 1rem;
    background: linear-gradient(90deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
    border-radius: 10px;
    margin-top: 2rem;
">
    <h4 style="margin: 0; background: linear-gradient(90deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        🚀 Neo4j Graph Explorer - Colorful Edition
    </h4>
    <p style="margin: 0.5rem 0;">🎨 Named Nodes • 🌈 Vibrant Relationships • 🔗 Stable Visualization</p>
</div>
""", unsafe_allow_html=True)
