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

# Enhanced CSS for better relationship visibility
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1.5rem;
    }
    
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    
    .response-container {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .graph-container {
        background-color: #ffffff;
        padding: 0.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0;
        width: 100%;
        min-height: 650px;
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
    
    .legend-container {
        background-color: #f8f9fa;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .relationship-debug {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #ffeaa7;
        margin: 0.5rem 0;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "graph_data" not in st.session_state:
        st.session_state.graph_data = None
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "error_count" not in st.session_state:
        st.session_state.error_count = 0

init_session_state()

# Header
st.markdown('<h1 class="main-header">🕸️ Neo4j Graph Explorer</h1>', unsafe_allow_html=True)

def call_agent_api(question: str, node_limit: int = 50) -> dict:
    """Enhanced API call that ensures relationships are included"""
    try:
        api_url = "http://localhost:8081/chat"
        
        # Modify questions to ensure relationships are included
        enhanced_question = question
        if any(word in question.lower() for word in ["show", "display", "get", "find"]):
            if "relationship" not in question.lower() and "connection" not in question.lower():
                enhanced_question = f"{question} with their relationships and connections"
        
        payload = {
            "question": enhanced_question,
            "session_id": st.session_state.session_id,
            "node_limit": node_limit
        }
        
        with st.spinner("🤖 Processing your request..."):
            response = requests.post(api_url, json=payload, timeout=45)
            response.raise_for_status()
            result = response.json()
            
            st.session_state.error_count = 0
            return result
            
    except requests.exceptions.ConnectionError:
        st.session_state.error_count += 1
        st.error("❌ Cannot connect to agent API (port 8081). Please start the FastAPI server.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. Try a simpler query.")
        return None
    except Exception as e:
        st.session_state.error_count += 1
        st.error(f"❌ API Error: {str(e)}")
        return None

def get_node_color(labels):
    """Enhanced color assignment"""
    if not labels or len(labels) == 0:
        return "#95afc0"
    
    label = labels[0] if isinstance(labels, list) else str(labels)
    
    colors = {
        "Person": "#FF6B6B",      # Bright Red
        "Movie": "#4ECDC4",       # Teal
        "Company": "#45B7D1",     # Blue
        "Product": "#96CEB4",     # Green
        "Location": "#FECA57",    # Yellow
        "Event": "#FF9FF3",       # Pink
        "User": "#A55EEA",        # Purple
        "Order": "#26DE81",       # Mint
        "Category": "#FD79A8",    # Rose
        "Department": "#6C5CE7",  # Indigo
        "Project": "#FDCB6E",     # Orange
        "Task": "#E17055",        # Coral
        "Actor": "#00CEC9",       # Cyan
        "Director": "#E84393",    # Magenta
        "Producer": "#00B894"     # Emerald
    }
    
    return colors.get(label, "#95afc0")

def get_relationship_color(rel_type):
    """Enhanced relationship colors - more visible"""
    colors = {
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
        "PRODUCED": "#4caf50"         # Light Green
    }
    return colors.get(rel_type, "#333333")  # Darker default for visibility

def render_graph_with_relationships(graph_data: dict) -> bool:
    """Enhanced graph rendering with focus on relationship visibility"""
    
    if not graph_data:
        st.info("🔍 No graph data available. Run a query to visualize your database!")
        return False
    
    try:
        nodes = graph_data.get("nodes", [])
        relationships = graph_data.get("relationships", [])
        
        if not nodes:
            st.info("📊 No nodes found. Try a query that returns graph data.")
            return False
        
        # Enhanced debug information
        st.write(f"📊 **Data Analysis:** {len(nodes)} nodes, {len(relationships)} relationships")
        
        # Debug relationships structure
        if relationships:
            rel_types = list(set(rel.get('type', 'UNKNOWN') for rel in relationships))
            st.markdown(f'<div class="relationship-debug">🔗 <strong>Relationship Types Found:</strong> {", ".join(rel_types)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="relationship-debug">⚠️ <strong>No relationships found in data</strong> - This might be why lines aren\'t showing</div>', unsafe_allow_html=True)
        
        # Create network with enhanced settings for relationship visibility
        net = Network(
            height="650px",
            width="100%", 
            bgcolor="#f8f9fa",
            font_color="#2c3e50",
            directed=True  # Enable directed edges for better visibility
        )
        
        # Process nodes with enhanced styling
        added_nodes = set()
        node_positions = {}  # Track positions for better layout
        
        for i, node in enumerate(nodes):
            try:
                node_id = str(node.get("id", f"node_{i}"))
                if node_id in added_nodes:
                    node_id = f"{node_id}_{i}"
                
                props = node.get("properties", {})
                labels = node.get("labels", ["Unknown"])
                
                # Enhanced display name
                display_name = str(props.get("name", props.get("title", props.get("label", f"Node_{i}"))))[:25]
                
                # Enhanced tooltip with more info
                tooltip = f"🆔 ID: {node_id}\\n🏷️ Type: {labels[0] if labels else 'Unknown'}"
                if props:
                    tooltip += f"\\n📊 Properties: {len(props)}"
                    # Show key properties
                    for key, value in list(props.items())[:3]:
                        tooltip += f"\\n• {key}: {str(value)[:30]}"
                
                # Enhanced node styling
                color = get_node_color(labels)
                size = 30 + len(props) * 2  # Larger nodes with more properties
                
                net.add_node(
                    node_id,
                    label=display_name,
                    title=tooltip,
                    color={
                        'background': color,
                        'border': '#2c3e50',
                        'highlight': {'background': color, 'border': '#e74c3c'}
                    },
                    size=size,
                    font={'size': 14, 'color': '#2c3e50'},
                    borderWidth=2
                )
                added_nodes.add(node_id)
                
            except Exception as e:
                st.warning(f"⚠️ Skipped node {i}: {str(e)}")
                continue
        
        # Enhanced relationship processing with better visibility
        added_edges = 0
        relationship_debug = []
        
        for i, rel in enumerate(relationships):
            try:
                # Handle different relationship data formats
                start_id = str(rel.get("startNode", rel.get("start", rel.get("source", ""))))
                end_id = str(rel.get("endNode", rel.get("end", rel.get("target", ""))))
                rel_type = str(rel.get("type", "CONNECTED"))
                rel_props = rel.get("properties", {})
                
                relationship_debug.append(f"{start_id} -[{rel_type}]-> {end_id}")
                
                # Only add if both nodes exist
                if start_id in added_nodes and end_id in added_nodes:
                    color = get_relationship_color(rel_type)
                    
                    # Enhanced edge tooltip
                    edge_tooltip = f"🔗 {rel_type}\\n📍 {start_id} → {end_id}"
                    if rel_props:
                        edge_tooltip += f"\\n📊 Properties: {len(rel_props)}"
                        for key, value in list(rel_props.items())[:2]:
                            edge_tooltip += f"\\n• {key}: {str(value)[:20]}"
                    
                    # Add edge with enhanced visibility
                    net.add_edge(
                        start_id,
                        end_id,
                        label=rel_type,
                        title=edge_tooltip,
                        color={
                            'color': color,
                            'highlight': '#e74c3c',
                            'hover': '#f39c12'
                        },
                        width=4,  # Thicker lines for visibility
                        arrows={
                            'to': {
                                'enabled': True, 
                                'scaleFactor': 1.5,
                                'type': 'arrow'
                            }
                        },
                        font={'size': 12, 'color': color, 'strokeWidth': 3, 'strokeColor': '#ffffff'},
                        smooth={'type': 'continuous', 'roundness': 0.1}
                    )
                    added_edges += 1
                else:
                    if start_id not in added_nodes:
                        st.warning(f"⚠️ Relationship {i}: Start node '{start_id}' not found")
                    if end_id not in added_nodes:
                        st.warning(f"⚠️ Relationship {i}: End node '{end_id}' not found")
                    
            except Exception as e:
                st.warning(f"⚠️ Skipped relationship {i}: {str(e)}")
                continue
        
        # Show relationship processing results
        st.write(f"✅ **Successfully added:** {len(added_nodes)} nodes, {added_edges} relationships")
        
        if relationship_debug:
            with st.expander("🔍 Relationship Processing Details", expanded=False):
                st.write("**Relationships found in data:**")
                for rel_info in relationship_debug[:10]:  # Show first 10
                    st.code(rel_info)
                if len(relationship_debug) > 10:
                    st.write(f"... and {len(relationship_debug) - 10} more")
        
        # Enhanced physics settings for better relationship visibility
        net.set_options("""
        var options = {
            "physics": {
                "enabled": true,
                "stabilization": {
                    "enabled": true,
                    "iterations": 150
                },
                "barnesHut": {
                    "gravitationalConstant": -8000,
                    "centralGravity": 0.1,
                    "springLength": 200,
                    "springConstant": 0.04,
                    "damping": 0.09,
                    "avoidOverlap": 0.5
                }
            },
            "interaction": {
                "dragNodes": true,
                "dragView": true,
                "zoomView": true,
                "selectConnectedEdges": true,
                "hover": true
            },
            "edges": {
                "width": 4,
                "selectionWidth": 6,
                "hoverWidth": 6,
                "smooth": {
                    "enabled": true,
                    "type": "continuous",
                    "roundness": 0.1
                }
            },
            "nodes": {
                "borderWidth": 2,
                "borderWidthSelected": 4
            }
        }
        """)
        
        # Generate and display HTML
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                net.save_graph(f.name)
                html_file = f.name
            
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Enhanced wrapper with better styling
            wrapped_html = f"""
            <div style="
                border: 2px solid #ddd; 
                border-radius: 8px; 
                overflow: hidden; 
                background: white;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            ">
                <div style="
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 8px 16px;
                    font-weight: bold;
                    font-size: 14px;
                ">
                    🕸️ Neo4j Graph | {len(added_nodes)} Nodes | {added_edges} Relationships
                </div>
                {html_content}
            </div>
            """
            
            # Display with larger height for better visibility
            components.html(wrapped_html, height=680, scrolling=False)
            
            # Cleanup
            try:
                os.unlink(html_file)
            except:
                pass
            
            return True
            
        except Exception as e:
            st.error(f"❌ HTML rendering error: {str(e)}")
            return False
        
    except Exception as e:
        st.error(f"❌ Graph rendering failed: {str(e)}")
        
        with st.expander("🔍 Debug Information", expanded=True):
            st.write("**Error Details:**")
            st.code(traceback.format_exc())
            st.write("**Graph Data Structure:**")
            st.json(graph_data)
        
        return False

def create_enhanced_legend(nodes, relationships):
    """Create enhanced legend with relationship info"""
    try:
        node_types = list(set(labels[0] for node in nodes for labels in [node.get("labels", ["Unknown"])] if labels))
        rel_types = list(set(rel.get("type", "CONNECTED") for rel in relationships))
        
        legend_html = '<div class="legend-container">'
        legend_html += '<h4>🎨 Graph Legend</h4>'
        
        if node_types:
            legend_html += '<p><strong>📊 Node Types:</strong></p>'
            for node_type in sorted(node_types):
                color = get_node_color([node_type])
                legend_html += f'<span style="color: {color}; font-size: 16px;">●</span> <strong>{node_type}</strong> &nbsp;&nbsp;'
        
        if rel_types:
            legend_html += '<p><strong>🔗 Relationship Types:</strong></p>'
            for rel_type in sorted(rel_types):
                color = get_relationship_color(rel_type)
                legend_html += f'<span style="color: {color}; font-size: 14px; font-weight: bold;">━━━</span> <strong>{rel_type}</strong> &nbsp;&nbsp;'
        
        legend_html += '</div>'
        return legend_html
        
    except Exception as e:
        return f'<div class="legend-container">Legend error: {str(e)}</div>'

# Main layout
col1, col2 = st.columns([1, 2], gap="medium")

with col1:
    st.markdown("### 💬 Chat Interface")
    
    # Connection status
    if st.session_state.error_count > 3:
        st.error("⚠️ Multiple connection failures. Please check if all services are running.")
        if st.button("🔄 Reset Connection"):
            st.session_state.error_count = 0
            st.rerun()
    
    # Enhanced quick actions focused on relationships
    st.markdown("#### 🚀 Quick Actions")
    quick_actions = [
        ("Show all with connections", "Show me all nodes with their relationships and connections"),
        ("Person network", "Show me all Person nodes and how they are connected"),
        ("Company relationships", "Display Company nodes with their relationships"),
        ("Full network", "Show me the complete network structure with all connections"),
        ("Relationship types", "What types of relationships exist in the database?"),
        ("Connected components", "Show me groups of connected nodes")
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
                    st.success("✅ Graph updated with relationships!")
                
                st.session_state.last_response = result
                st.rerun()
    
    st.divider()
    
    # Enhanced question input with relationship hints
    st.markdown("#### ✍️ Ask a Question")
    st.info("💡 **Tip:** Add 'with relationships' or 'and connections' to ensure relationship lines are shown!")
    
    with st.form("question_form", clear_on_submit=True):
        user_question = st.text_area(
            "Your question:",
            placeholder="e.g., Show me Person nodes with their relationships, Find Alice and her connections",
            height=80
        )
        
        # Checkbox to force relationship inclusion
        include_relationships = st.checkbox(
            "🔗 Force include relationships", 
            value=True, 
            help="Automatically add relationship data to your query"
        )
        
        node_limit = st.selectbox(
            "Node limit:",
            [10, 25, 50, 75, 100],
            index=2,
            help="Smaller limits show relationships more clearly"
        )
        
        submit_button = st.form_submit_button("🚀 Submit")
    
    if submit_button and user_question.strip():
        # Enhance question if checkbox is selected
        final_question = user_question.strip()
        if include_relationships and "relationship" not in final_question.lower():
            final_question += " with their relationships and connections"
        
        result = call_agent_api(final_question, node_limit)
        
        if result:
            st.session_state.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "question": final_question,
                "answer": result.get("answer", ""),
                "graph_data": result.get("graph_data")
            })
            
            if result.get("graph_data"):
                st.session_state.graph_data = result["graph_data"]
                st.success("✅ Graph updated with relationships!")
            
            st.session_state.last_response = result
            st.rerun()
    
    st.divider()
    
    # Enhanced test data with more relationships
    if st.button("🧪 Load Test Graph with Relationships"):
        try:
            sample_data = {
                "nodes": [
                    {"id": "1", "labels": ["Person"], "properties": {"name": "Alice", "age": 30}},
                    {"id": "2", "labels": ["Person"], "properties": {"name": "Bob", "age": 25}},
                    {"id": "3", "labels": ["Person"], "properties": {"name": "Charlie", "age": 35}},
                    {"id": "4", "labels": ["Company"], "properties": {"name": "TechCorp"}},
                    {"id": "5", "labels": ["Location"], "properties": {"name": "NYC"}},
                    {"id": "6", "labels": ["Project"], "properties": {"name": "AI Project"}}
                ],
                "relationships": [
                    {"startNode": "1", "endNode": "2", "type": "KNOWS", "properties": {"since": "2020"}},
                    {"startNode": "2", "endNode": "3", "type": "KNOWS", "properties": {"since": "2021"}},
                    {"startNode": "1", "endNode": "4", "type": "WORKS_FOR", "properties": {"role": "Engineer"}},
                    {"startNode": "2", "endNode": "4", "type": "WORKS_FOR", "properties": {"role": "Developer"}},
                    {"startNode": "3", "endNode": "4", "type": "MANAGES", "properties": {"department": "Tech"}},
                    {"startNode": "4", "endNode": "5", "type": "LOCATED_IN", "properties": {}},
                    {"startNode": "1", "endNode": "6", "type": "ASSIGNED_TO", "properties": {"role": "Lead"}},
                    {"startNode": "2", "endNode": "6", "type": "ASSIGNED_TO", "properties": {"role": "Dev"}}
                ]
            }
            st.session_state.graph_data = sample_data
            st.success("✅ Test graph with 8 relationships loaded!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load test data: {e}")
    
    # Recent conversations
    st.markdown("#### 📝 Recent Conversations")
    if st.session_state.conversation_history:
        for item in reversed(st.session_state.conversation_history[-2:]):
            with st.expander(f"💬 {item['question'][:40]}...", expanded=False):
                st.write(f"**Time:** {item['timestamp'][:19]}")
                if item.get('graph_data'):
                    rel_count = len(item['graph_data'].get('relationships', []))
                    st.write(f"**Relationships:** {rel_count}")
                st.write(f"**Answer:** {item['answer'][:100]}...")
    else:
        st.info("No conversations yet.")
    
    if st.button("🗑️ Clear All"):
        st.session_state.conversation_history = []
        st.session_state.graph_data = None
        st.session_state.last_response = None
        st.session_state.error_count = 0
        st.rerun()

with col2:
    st.markdown("### 🕸️ Graph with Relationships")
    
    # Response display
    if st.session_state.last_response:
        answer = st.session_state.last_response.get("answer", "")
        if answer:
            with st.container():
                st.markdown("#### 🤖 Latest Response")
                clean_answer = answer.replace("**", "").replace("#", "").strip()
                st.info(clean_answer[:200] + "..." if len(clean_answer) > 200 else clean_answer)
    
    # Graph section with enhanced relationship focus
    if st.session_state.graph_data:
        nodes = st.session_state.graph_data.get("nodes", [])
        relationships = st.session_state.graph_data.get("relationships", [])
        
        # Enhanced statistics
        col2_1, col2_2, col2_3 = st.columns(3)
        with col2_1:
            st.markdown(f'<div class="metric-container"><strong>{len(nodes)}</strong><br>Nodes</div>', unsafe_allow_html=True)
        with col2_2:
            st.markdown(f'<div class="metric-container"><strong>{len(relationships)}</strong><br>Relationships</div>', unsafe_allow_html=True)
        with col2_3:
            connectivity = len(relationships) / max(len(nodes), 1)
            st.markdown(f'<div class="metric-container"><strong>{connectivity:.1f}</strong><br>Connectivity</div>', unsafe_allow_html=True)
        
        # Enhanced legend
        if nodes or relationships:
            legend = create_enhanced_legend(nodes, relationships)
            st.markdown(legend, unsafe_allow_html=True)
        
        # Render graph with relationship focus
        st.markdown("#### 🎨 Interactive Network")
        with st.container():
            success = render_graph_with_relationships(st.session_state.graph_data)
            
            if success:
                if len(relationships) > 0:
                    st.success(f"✅ Graph with {len(relationships)} relationship lines displayed!")
                else:
                    st.warning("⚠️ No relationships found in data - try queries that include 'with relationships'")
                
                # Enhanced controls
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("🔄 Refresh"):
                        st.rerun()
                with col_b:
                    if st.button("🔗 Show Connections"):
                        result = call_agent_api("Show me all connections and relationships in the database", node_limit=50)
                        if result and result.get("graph_data"):
                            st.session_state.graph_data = result["graph_data"]
                            st.rerun()
                with col_c:
                    if st.button("📊 Network Stats"):
                        st.info(f"Nodes: {len(nodes)}, Relationships: {len(relationships)}, Avg Connections: {len(relationships)*2/max(len(nodes),1):.1f}")
            else:
                st.error("❌ Graph rendering failed. Check debug info above.")
    
    else:
        # Welcome screen with relationship focus
        st.markdown("""
        <div style="
            text-align: center; 
            padding: 2rem; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            border-radius: 1rem; 
            margin: 1rem 0;
        ">
            <h3>🕸️ Neo4j Graph with Relationships</h3>
            <p>Visualize nodes <strong>AND</strong> their connecting relationships</p>
            <p><strong>✨ Features:</strong></p>
            <p>🎨 Colored nodes • 🔗 <strong>Visible relationship lines</strong> • 🏷️ Relationship labels • 🖱️ Interactive</p>
            <p><em>Click "Load Test Graph with Relationships" to see relationships!</em></p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 0.5rem;">
    <small>🚀 <strong>Neo4j Graph Explorer</strong> | Enhanced for Relationship Visibility | 
    <a href="http://localhost:8081/docs" target="_blank">API</a></small>
</div>
""", unsafe_allow_html=True)
