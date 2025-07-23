import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import requests
import json
import os
import tempfile
from datetime import datetime
import uuid
import random

# Page configuration
st.set_page_config(
    page_title="Neo4j Graph Explorer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .response-container {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .graph-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #dee2e6;
        margin: 1rem 0;
        min-height: 600px;
    }
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }
    .stButton button:hover {
        background-color: #0d5aa7;
    }
    .metric-container {
        background-color: #f1f3f4;
        padding: 0.5rem;
        border-radius: 0.25rem;
        text-align: center;
    }
    .legend-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Header
st.markdown('<h1 class="main-header">🕸️ Neo4j Graph Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6c757d;">Ask questions, run queries, and visualize your Neo4j database in real-time</p>', unsafe_allow_html=True)

def call_agent_api(question: str, node_limit: int = 100) -> dict:
    """Call the FastAPI agent endpoint"""
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
            return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the agent API. Please ensure the FastAPI server is running on port 8081.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. Please try a simpler query.")
        return None
    except Exception as e:
        st.error(f"❌ Error calling agent API: {str(e)}")
        return None

def get_node_color_and_size(labels, properties=None):
    """Get color and size based on node labels and properties"""
    if not labels:
        return "#95afc0", 25
    
    label = labels[0] if isinstance(labels, list) else labels
    
    # Color mapping with vibrant colors
    color_map = {
        "Person": "#FF6B6B",      # Red
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
        "Unknown": "#95AFC0"      # Gray
    }
    
    # Size based on properties (nodes with more properties are larger)
    base_size = 25
    if properties:
        property_count = len(properties)
        size = base_size + (property_count * 3)  # Larger nodes for more properties
        size = min(size, 50)  # Cap at 50
    else:
        size = base_size
    
    return color_map.get(label, "#95AFC0"), size

def get_relationship_color_and_width(rel_type):
    """Get color and width for relationships based on type"""
    
    # Relationship color mapping
    rel_colors = {
        "KNOWS": "#FF6B6B",           # Red
        "WORKS_FOR": "#4ECDC4",       # Teal  
        "MANAGES": "#45B7D1",         # Blue
        "LOCATED_IN": "#FECA57",      # Yellow
        "BELONGS_TO": "#96CEB4",      # Green
        "CREATED": "#FF9FF3",         # Pink
        "OWNS": "#A55EEA",            # Purple
        "USES": "#26DE81",            # Mint
        "DEPENDS_ON": "#FD79A8",      # Rose
        "MEMBER_OF": "#6C5CE7",       # Indigo
        "ASSIGNED_TO": "#FDCB6E",     # Orange
        "REPORTS_TO": "#E17055",      # Coral
        "ACTED_IN": "#00CEC9",        # Cyan
        "DIRECTED": "#E84393",        # Magenta
        "PRODUCED": "#00B894",        # Emerald
    }
    
    # Width based on relationship importance
    width_map = {
        "KNOWS": 2,
        "WORKS_FOR": 3,
        "MANAGES": 4,
        "OWNS": 4,
        "CREATED": 3,
        "MEMBER_OF": 2,
        "BELONGS_TO": 2,
    }
    
    color = rel_colors.get(rel_type, "#666666")
    width = width_map.get(rel_type, 2)
    
    return color, width

def create_graph_legend(nodes, relationships):
    """Create a legend for the graph showing colors and types"""
    
    # Get unique node labels
    node_labels = set()
    for node in nodes:
        labels = node.get("labels", ["Unknown"])
        node_labels.update(labels)
    
    # Get unique relationship types
    rel_types = set()
    for rel in relationships:
        rel_types.add(rel.get("type", "CONNECTED"))
    
    legend_html = """
    <div class="legend-container">
        <h4>🎨 Graph Legend</h4>
        <div style="display: flex; flex-wrap: wrap; gap: 1rem;">
    """
    
    # Node legend
    if node_labels:
        legend_html += "<div><strong>📊 Node Types:</strong><br>"
        for label in sorted(node_labels):
            color, _ = get_node_color_and_size([label])
            legend_html += f'<span style="color: {color}; font-size: 18px;">●</span> {label} &nbsp;&nbsp;'
        legend_html += "</div>"
    
    # Relationship legend
    if rel_types:
        legend_html += "<div><strong>🔗 Relationship Types:</strong><br>"
        for rel_type in sorted(rel_types):
            color, _ = get_relationship_color_and_width(rel_type)
            legend_html += f'<span style="color: {color}; font-size: 14px;">━━</span> {rel_type} &nbsp;&nbsp;'
        legend_html += "</div>"
    
    legend_html += "</div></div>"
    
    return legend_html

def render_graph_with_relationships(graph_data: dict) -> bool:
    """Enhanced graph rendering with proper relationships and colors"""
    
    if not graph_data:
        st.info("🔍 No graph data available. Ask a question to visualize the database!")
        return False
    
    nodes = graph_data.get("nodes", [])
    relationships = graph_data.get("relationships", [])
    
    if not nodes:
        st.info("📊 No nodes to display. Try a query that returns graph data.")
        return False
    
    try:
        # Debug information
        st.write(f"📊 **Graph Info:** {len(nodes)} nodes, {len(relationships)} relationships")
        
        # Show legend
        legend_html = create_graph_legend(nodes, relationships)
        st.markdown(legend_html, unsafe_allow_html=True)
        
        # Create Pyvis network with enhanced settings
        net = Network(
            height="650px", 
            width="100%", 
            bgcolor="#f8f9fa",  # Light background
            font_color="#2c3e50",
            directed=True,
            select_menu=False,
            filter_menu=False
        )
        
        # Add nodes with enhanced styling
        node_ids = set()  # Track added nodes
        for node in nodes:
            node_id = str(node.get("id", f"node_{random.randint(1000, 9999)}"))
            properties = node.get("properties", {})
            labels = node.get("labels", ["Unknown"])
            
            # Create display label (limit length)
            display_name = (
                properties.get("name") or 
                properties.get("title") or 
                properties.get("label") or 
                f"{labels[0] if labels else 'Node'}_{node_id[:8]}"
            )
            display_name = str(display_name)[:25]  # Limit length
            
            # Create detailed hover info
            hover_info = f"🏷️ Labels: {', '.join(labels)}\n🆔 ID: {node_id}"
            if properties:
                hover_info += "\n📋 Properties:"
                for key, value in list(properties.items())[:5]:  # Show first 5 properties
                    hover_info += f"\n  • {key}: {str(value)[:40]}"
                if len(properties) > 5:
                    hover_info += f"\n  ... and {len(properties) - 5} more"
            
            # Get color and size
            color, size = get_node_color_and_size(labels, properties)
            
            # Add node with enhanced styling
            net.add_node(
                node_id,
                label=display_name,
                title=hover_info,
                color={
                    'background': color,
                    'border': '#2c3e50',
                    'highlight': {'background': color, 'border': '#e74c3c'}
                },
                size=size,
                font={'size': 16, 'color': '#2c3e50', 'face': 'Arial'},
                borderWidth=2,
                borderWidthSelected=4,
                shape='dot'
            )
            node_ids.add(node_id)
        
        # Add relationships with enhanced styling
        relationship_count = 0
        for rel in relationships:
            # Handle different relationship data formats
            start_id = str(rel.get("startNode", rel.get("start", rel.get("source", ""))))
            end_id = str(rel.get("endNode", rel.get("end", rel.get("target", ""))))
            rel_type = rel.get("type", "CONNECTED")
            rel_props = rel.get("properties", {})
            
            # Only add edge if both nodes exist
            if start_id in node_ids and end_id in node_ids:
                # Create relationship hover info
                edge_info = f"🔗 Type: {rel_type}\n🆔 From: {start_id} → {end_id}"
                if rel_props:
                    edge_info += "\n📋 Properties:"
                    for key, value in list(rel_props.items())[:3]:
                        edge_info += f"\n  • {key}: {str(value)[:30]}"
                
                # Get relationship color and width
                rel_color, rel_width = get_relationship_color_and_width(rel_type)
                
                # Add edge with enhanced styling
                net.add_edge(
                    start_id,
                    end_id,
                    label=rel_type,
                    title=edge_info,
                    color={
                        'color': rel_color,
                        'highlight': '#e74c3c',
                        'hover': '#f39c12'
                    },
                    width=rel_width,
                    arrows={
                        'to': {
                            'enabled': True, 
                            'scaleFactor': 1.2,
                            'type': 'arrow'
                        }
                    },
                    smooth={
                        'enabled': True,
                        'type': 'continuous',
                        'roundness': 0.2
                    },
                    font={'size': 12, 'color': '#2c3e50', 'strokeWidth': 2, 'strokeColor': '#ffffff'}
                )
                relationship_count += 1
        
        st.write(f"✅ Successfully added {len(node_ids)} nodes and {relationship_count} relationships")
        
        # Configure enhanced physics and layout
        net.set_options("""
        var options = {
            "physics": {
                "enabled": true,
                "stabilization": {
                    "enabled": true,
                    "iterations": 200,
                    "updateInterval": 25
                },
                "barnesHut": {
                    "gravitationalConstant": -8000,
                    "centralGravity": 0.15,
                    "springLength": 120,
                    "springConstant": 0.05,
                    "damping": 0.09,
                    "avoidOverlap": 0.2
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
                "tooltipDelay": 200
            },
            "nodes": {
                "borderWidth": 2,
                "borderWidthSelected": 4,
                "shadow": {
                    "enabled": true,
                    "color": 'rgba(0,0,0,0.2)',
                    "size": 8,
                    "x": 2,
                    "y": 2
                }
            },
            "edges": {
                "shadow": {
                    "enabled": true,
                    "color": 'rgba(0,0,0,0.1)',
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
                "clusterThreshold": 150
            }
        }
        """)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            net.save_graph(f.name)
            html_file = f.name
        
        # Read and enhance the HTML
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Enhanced HTML wrapper with better styling
        enhanced_html = f"""
        <div style="
            border: 2px solid #ddd; 
            border-radius: 12px; 
            overflow: hidden; 
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        ">
            <div style="
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 16px;
            ">
                🕸️ Interactive Neo4j Graph | {len(node_ids)} Nodes | {relationship_count} Relationships
            </div>
            <div style="background: white; padding: 5px;">
                {html_content}
            </div>
        </div>
        """
        
        # Display in Streamlit
        components.html(enhanced_html, height=720, scrolling=False)
        
        # Clean up temp file
        try:
            os.unlink(html_file)
        except:
            pass
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error rendering graph: {str(e)}")
        
        # Show detailed error info
        with st.expander("🔍 Error Details", expanded=True):
            st.write(f"**Error:** {str(e)}")
            st.write(f"**Nodes available:** {len(nodes)}")
            st.write(f"**Relationships available:** {len(relationships)}")
            
            if nodes:
                st.write("**Sample Node:**")
                st.json(nodes[0])
            
            if relationships:
                st.write("**Sample Relationship:**")
                st.json(relationships[0])
        
        return False

def display_conversation_item(item: dict):
    """Display a conversation item with proper formatting"""
    timestamp = item.get("timestamp", datetime.now().isoformat())
    question = item.get("question", "")
    answer = item.get("answer", "")
    
    with st.container():
        # User question
        st.markdown(f"""
        <div style="background: #e3f2fd; padding: 0.75rem; border-radius: 0.5rem; margin-bottom: 0.5rem;">
            <strong>🧑‍💻 You ({timestamp[:19]}):</strong><br>
            {question}
        </div>
        """, unsafe_allow_html=True)
        
        # Agent response
        formatted_answer = answer.replace('\n', '<br>').replace('**', '<strong>')
        if '**' in formatted_answer:
            formatted_answer = formatted_answer.replace('**', '</strong>')
        st.markdown(f"""
        <div class="response-container">
            <strong>🤖 Agent Response:</strong><br>
            {formatted_answer}
        </div>
        """, unsafe_allow_html=True)

# Main layout with columns
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 💬 Chat Interface")
    
    # Quick action buttons
    st.markdown("#### 🚀 Quick Actions")
    quick_actions = [
        ("Show all nodes", "Show me all nodes with their relationships"),
        ("Database schema", "What is the database schema?"),
        ("Count nodes", "How many nodes are in the database?"),
        ("Person nodes", "Show me all Person nodes and their connections"),
        ("All relationships", "Display all relationships in the database"),
        ("Sample network", "Give me a sample of the network structure")
    ]
    
    for action_name, action_query in quick_actions:
        if st.button(action_name, key=f"quick_{action_name}"):
            result = call_agent_api(action_query, node_limit=75)
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
    
    # Custom question input
    st.markdown("#### ✍️ Ask a Question")
    with st.form("question_form", clear_on_submit=True):
        user_question = st.text_area(
            "Your question:",
            placeholder="e.g., Show me Person nodes and their relationships, Create connections between Alice and Bob...",
            height=100
        )
        
        node_limit = st.slider(
            "Max nodes to display:",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Limit nodes for better performance and visualization"
        )
        
        submit_button = st.form_submit_button("🚀 Submit")
    
    if submit_button and user_question.strip():
        result = call_agent_api(user_question.strip(), node_limit)
        
        if result:
            st.session_state.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "question": user_question.strip(),
                "answer": result.get("answer", ""),
                "graph_data": result.get("graph_data"),
                "tool": result.get("tool", ""),
                "query": result.get("query", "")
            })
            
            if result.get("graph_data"):
                st.session_state.graph_data = result["graph_data"]
                st.success("✅ Graph updated with latest data and relationships!")
            
            st.session_state.last_response = result
            st.rerun()
    
    st.divider()
    
    # Test graph with enhanced sample data
    if st.button("🧪 Test Graph with Sample Data"):
        sample_data = {
            "nodes": [
                {"id": "1", "labels": ["Person"], "properties": {"name": "Alice", "age": 30, "department": "Engineering"}},
                {"id": "2", "labels": ["Person"], "properties": {"name": "Bob", "age": 25, "role": "Developer"}},
                {"id": "3", "labels": ["Company"], "properties": {"name": "TechCorp", "founded": 2010, "size": "Large"}},
                {"id": "4", "labels": ["Location"], "properties": {"name": "New York", "country": "USA"}},
                {"id": "5", "labels": ["Project"], "properties": {"name": "GraphDB Project", "status": "Active"}},
                {"id": "6", "labels": ["Department"], "properties": {"name": "Engineering", "budget": 500000}}
            ],
            "relationships": [
                {"startNode": "1", "endNode": "2", "type": "KNOWS", "properties": {"since": "2020"}},
                {"startNode": "1", "endNode": "3", "type": "WORKS_FOR", "properties": {"position": "Senior Engineer"}},
                {"startNode": "2", "endNode": "3", "type": "WORKS_FOR", "properties": {"position": "Developer"}},
                {"startNode": "3", "endNode": "4", "type": "LOCATED_IN", "properties": {"headquarters": True}},
                {"startNode": "1", "endNode": "5", "type": "ASSIGNED_TO", "properties": {"role": "Lead"}},
                {"startNode": "2", "endNode": "5", "type": "ASSIGNED_TO", "properties": {"role": "Developer"}},
                {"startNode": "1", "endNode": "6", "type": "MEMBER_OF", "properties": {}},
                {"startNode": "6", "endNode": "3", "type": "BELONGS_TO", "properties": {}}
            ]
        }
        st.session_state.graph_data = sample_data
        st.success("✅ Enhanced sample graph with relationships loaded!")
        st.rerun()
    
    # Conversation history
    st.markdown("#### 📝 Recent Conversations")
    if st.session_state.conversation_history:
        for item in reversed(st.session_state.conversation_history[-3:]):
            with st.expander(f"🗨️ {item['question'][:50]}...", expanded=False):
                st.write(f"**Time:** {item['timestamp'][:19]}")
                if item.get('tool'):
                    st.write(f"**Tool:** {item['tool']}")
                if item.get('query'):
                    st.code(item['query'], language='cypher')
                st.markdown(item['answer'])
    else:
        st.info("No conversations yet. Ask a question to get started!")
    
    if st.button("🗑️ Clear History"):
        st.session_state.conversation_history = []
        st.session_state.graph_data = None
        st.session_state.last_response = None
        st.rerun()

with col2:
    st.markdown("### 🕸️ Enhanced Graph Visualization")
    
    # Display current response
    if st.session_state.last_response:
        answer = st.session_state.last_response.get("answer", "")
        if answer:
            st.markdown('<div class="response-container">', unsafe_allow_html=True)
            st.markdown("#### 🤖 Latest Response")
            formatted_answer = answer.replace('\n', '  \n')
            st.markdown(formatted_answer)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Debug section
    with st.expander("🔍 Graph Data Info", expanded=False):
        if st.session_state.graph_data:
            nodes = st.session_state.graph_data.get('nodes', [])
            relationships = st.session_state.graph_data.get('relationships', [])
            
            st.write("✅ Graph data available")
            st.write(f"**Nodes:** {len(nodes)}")
            st.write(f"**Relationships:** {len(relationships)}")
            
            if nodes:
                st.write("**Sample Node Structure:**")
                st.json(nodes[0])
            
            if relationships:
                st.write("**Sample Relationship Structure:**")
                st.json(relationships[0])
                
                # Show relationship types
                rel_types = list(set(rel.get('type', 'UNKNOWN') for rel in relationships))
                st.write(f"**Relationship Types:** {', '.join(rel_types)}")
        else:
            st.write("❌ No graph data available")
    
    # Render enhanced graph
    if st.session_state.graph_data:
        st.markdown("#### 🎨 Interactive Network Graph")
        
        # Graph statistics
        nodes_count = len(st.session_state.graph_data.get("nodes", []))
        rels_count = len(st.session_state.graph_data.get("relationships", []))
        
        col2_1, col2_2, col2_3 = st.columns(3)
        with col2_1:
            st.markdown(f'<div class="metric-container"><strong>{nodes_count}</strong><br>Nodes</div>', unsafe_allow_html=True)
        with col2_2:
            st.markdown(f'<div class="metric-container"><strong>{rels_count}</strong><br>Relationships</div>', unsafe_allow_html=True)
        with col2_3:
            density = rels_count/max(nodes_count,1) if nodes_count > 0 else 0
            st.markdown(f'<div class="metric-container"><strong>{density:.1f}</strong><br>Density</div>', unsafe_allow_html=True)
        
        # Render the enhanced graph
        success = render_graph_with_relationships(st.session_state.graph_data)
        
        if success:
            st.success(f"✅ Displaying {nodes_count} nodes and {rels_count} relationships with colors!")
            
            # Graph controls
            col2_4, col2_5, col2_6 = st.columns(3)
            with col2_4:
                if st.button("🔄 Refresh Graph"):
                    refresh_result = call_agent_api("Show me the current network with all relationships", node_limit=75)
                    if refresh_result and refresh_result.get("graph_data"):
                        st.session_state.graph_data = refresh_result["graph_data"]
                        st.rerun()
            
            with col2_5:
                if st.button("🎯 Center Graph"):
                    st.info("💡 Use mouse wheel to zoom and drag to pan the graph!")
            
            with col2_6:
                if st.button("💾 Download"):
                    graph_json = json.dumps(st.session_state.graph_data, indent=2)
                    st.download_button(
                        label="📥 JSON",
                        data=graph_json,
                        file_name=f"neo4j_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        else:
            st.error("❌ Failed to render graph - check debug info above")
    else:
        # Welcome message
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 1rem; margin: 2rem 0;">
            <h3>🎯 Welcome to Enhanced Neo4j Graph Explorer!</h3>
            <p>Visualize your Neo4j database with <strong>colored nodes</strong> and <strong>relationship types</strong></p>
            <p><strong>✨ Features:</strong></p>
            <ul style="text-align: left; display: inline-block; color: #f8f9fa;">
                <li>🎨 Color-coded nodes by type</li>
                <li>🔗 Visible relationship labels</li>
                <li>📊 Interactive graph legend</li>
                <li>🖱️ Drag, zoom, and hover interactions</li>
                <li>📈 Real-time graph updates</li>
            </ul>
            <p><em>💡 Click "Test Graph with Sample Data" to see the enhanced visualization!</em></p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p>🚀 <strong>Enhanced Neo4j Graph Explorer</strong> | 
    🎨 Colored Nodes & Relationships | 
    🔗 Interactive Network Visualization |
    <a href="http://localhost:8081/docs" target="_blank">API Docs</a>
    </p>
</div>
""", unsafe_allow_html=True)
