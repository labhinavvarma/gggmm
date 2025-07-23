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

# Fixed CSS to remove white gaps and improve stability
st.markdown("""
<style>
    /* Remove default margins and padding */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1.5rem;
    }
    
    /* Chat container */
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    
    /* Response container */
    .response-container {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    /* Graph container - fixed to remove gaps */
    .graph-container {
        background-color: #ffffff;
        padding: 0.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0;
        width: 100%;
        min-height: 600px;
    }
    
    /* Button styling */
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
        transform: translateY(-1px);
    }
    
    /* Metrics */
    .metric-container {
        background-color: #f1f3f4;
        padding: 0.75rem;
        border-radius: 0.25rem;
        text-align: center;
        margin: 0.25rem 0;
        border: 1px solid #e0e0e0;
    }
    
    /* Legend */
    .legend-container {
        background-color: #f8f9fa;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    /* Fix column gaps */
    .css-1r6slb0 {
        gap: 1rem;
    }
    
    /* Remove extra whitespace */
    .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Graph frame styling */
    .graph-frame {
        border: 2px solid #ddd;
        border-radius: 8px;
        overflow: hidden;
        background: white;
        margin: 0;
        padding: 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with better error handling
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
    """Stable API call with better error handling"""
    try:
        api_url = "http://localhost:8081/chat"
        payload = {
            "question": question,
            "session_id": st.session_state.session_id,
            "node_limit": node_limit
        }
        
        with st.spinner("🤖 Processing your request..."):
            response = requests.post(api_url, json=payload, timeout=45)
            response.raise_for_status()
            result = response.json()
            
            # Reset error count on success
            st.session_state.error_count = 0
            return result
            
    except requests.exceptions.ConnectionError:
        st.session_state.error_count += 1
        st.error("❌ Cannot connect to agent API (port 8081). Please start the FastAPI server.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. Try a simpler query or increase timeout.")
        return None
    except Exception as e:
        st.session_state.error_count += 1
        st.error(f"❌ API Error: {str(e)}")
        return None

def get_safe_node_color(labels):
    """Safe color assignment with fallback"""
    try:
        if not labels or len(labels) == 0:
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
            "Project": "#FDCB6E"
        }
        
        return colors.get(label, "#95afc0")
    except:
        return "#95afc0"

def get_safe_edge_color(rel_type):
    """Safe edge color with fallback"""
    try:
        colors = {
            "KNOWS": "#FF6B6B",
            "WORKS_FOR": "#4ECDC4",
            "MANAGES": "#45B7D1", 
            "LOCATED_IN": "#FECA57",
            "BELONGS_TO": "#96CEB4",
            "CREATED": "#FF9FF3",
            "OWNS": "#A55EEA",
            "USES": "#26DE81"
        }
        return colors.get(rel_type, "#666666")
    except:
        return "#666666"

def render_stable_graph(graph_data: dict) -> bool:
    """Stable graph rendering with comprehensive error handling"""
    
    if not graph_data:
        st.info("🔍 No graph data available. Run a query to visualize your database!")
        return False
    
    try:
        nodes = graph_data.get("nodes", [])
        relationships = graph_data.get("relationships", [])
        
        if not nodes:
            st.info("📊 No nodes found. Try a query that returns graph data like 'Show me all nodes'")
            return False
        
        # Debug info
        st.write(f"📊 **Processing:** {len(nodes)} nodes, {len(relationships)} relationships")
        
        # Create network with stable settings
        net = Network(
            height="600px",
            width="100%", 
            bgcolor="#ffffff",
            font_color="#333333",
            directed=False  # Simplified for stability
        )
        
        # Process nodes safely
        added_nodes = set()
        for i, node in enumerate(nodes):
            try:
                # Safe node ID extraction
                node_id = str(node.get("id", f"node_{i}"))
                if node_id in added_nodes:
                    node_id = f"{node_id}_{i}"
                
                # Safe property extraction
                props = node.get("properties", {})
                labels = node.get("labels", ["Unknown"])
                
                # Create display name safely
                display_name = str(props.get("name", props.get("title", f"Node_{i}")))[:20]
                
                # Safe color assignment
                color = get_safe_node_color(labels)
                
                # Create simple tooltip
                tooltip = f"ID: {node_id}\\nType: {labels[0] if labels else 'Unknown'}"
                if props:
                    prop_count = len(props)
                    tooltip += f"\\nProperties: {prop_count}"
                
                # Add node with basic styling
                net.add_node(
                    node_id,
                    label=display_name,
                    title=tooltip,
                    color=color,
                    size=20
                )
                added_nodes.add(node_id)
                
            except Exception as e:
                st.warning(f"⚠️ Skipped node {i}: {str(e)}")
                continue
        
        # Process relationships safely
        added_edges = 0
        for i, rel in enumerate(relationships):
            try:
                # Safe relationship extraction
                start_id = str(rel.get("startNode", rel.get("start", "")))
                end_id = str(rel.get("endNode", rel.get("end", "")))
                rel_type = str(rel.get("type", "CONNECTED"))
                
                # Only add if both nodes exist
                if start_id in added_nodes and end_id in added_nodes:
                    color = get_safe_edge_color(rel_type)
                    
                    net.add_edge(
                        start_id,
                        end_id,
                        label=rel_type,
                        color=color,
                        width=2,
                        title=f"Type: {rel_type}"
                    )
                    added_edges += 1
                    
            except Exception as e:
                st.warning(f"⚠️ Skipped relationship {i}: {str(e)}")
                continue
        
        st.write(f"✅ **Added:** {len(added_nodes)} nodes, {added_edges} relationships")
        
        # Simple physics settings for stability
        net.set_options("""
        var options = {
            "physics": {
                "enabled": true,
                "stabilization": {"iterations": 100}
            },
            "interaction": {
                "dragNodes": true,
                "dragView": true,
                "zoomView": true
            }
        }
        """)
        
        # Generate and display HTML
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                net.save_graph(f.name)
                html_file = f.name
            
            # Read HTML content
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Simple wrapper to prevent CSS conflicts
            wrapped_html = f"""
            <div class="graph-frame">
                {html_content}
            </div>
            """
            
            # Display in Streamlit
            components.html(wrapped_html, height=620, scrolling=False)
            
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
        
        # Show debug info on error
        with st.expander("🔍 Debug Information", expanded=True):
            st.write("**Error Details:**")
            st.code(traceback.format_exc())
            st.write("**Graph Data Structure:**")
            st.json(graph_data)
        
        return False

def create_simple_legend(nodes, relationships):
    """Create a simple legend"""
    try:
        # Get unique node types
        node_types = set()
        for node in nodes:
            labels = node.get("labels", ["Unknown"])
            if labels:
                node_types.add(labels[0])
        
        # Get unique relationship types  
        rel_types = set()
        for rel in relationships:
            rel_types.add(rel.get("type", "CONNECTED"))
        
        legend_html = '<div class="legend-container">'
        legend_html += '<strong>🎨 Graph Legend:</strong><br>'
        
        if node_types:
            legend_html += '<strong>Nodes:</strong> '
            for node_type in sorted(node_types):
                color = get_safe_node_color([node_type])
                legend_html += f'<span style="color: {color};">● {node_type}</span> '
        
        if rel_types:
            legend_html += '<br><strong>Relationships:</strong> '
            for rel_type in sorted(rel_types):
                color = get_safe_edge_color(rel_type)
                legend_html += f'<span style="color: {color};">━ {rel_type}</span> '
        
        legend_html += '</div>'
        return legend_html
        
    except:
        return '<div class="legend-container">Legend unavailable</div>'

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
    
    # Quick actions with simpler queries
    st.markdown("#### 🚀 Quick Actions")
    quick_actions = [
        ("Show nodes", "Show me all nodes"),
        ("Show relationships", "Display relationships between nodes"),
        ("Count nodes", "How many nodes are there?"),
        ("Database info", "What is in the database?"),
        ("Sample data", "Give me a sample of the graph")
    ]
    
    for action_name, action_query in quick_actions:
        if st.button(action_name, key=f"quick_{action_name}"):
            result = call_agent_api(action_query, node_limit=30)  # Smaller limit for stability
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
    
    # Simple question input
    st.markdown("#### ✍️ Ask a Question")
    with st.form("question_form", clear_on_submit=True):
        user_question = st.text_area(
            "Your question:",
            placeholder="e.g., Show me all nodes, What relationships exist?",
            height=80
        )
        
        node_limit = st.selectbox(
            "Node limit:",
            [10, 25, 50, 75, 100],
            index=2,  # Default to 50
            help="Smaller limits are more stable"
        )
        
        submit_button = st.form_submit_button("🚀 Submit")
    
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
                st.success("✅ Graph updated!")
            
            st.session_state.last_response = result
            st.rerun()
    
    st.divider()
    
    # Test with reliable sample data
    if st.button("🧪 Load Test Graph"):
        try:
            sample_data = {
                "nodes": [
                    {"id": "1", "labels": ["Person"], "properties": {"name": "Alice"}},
                    {"id": "2", "labels": ["Person"], "properties": {"name": "Bob"}},
                    {"id": "3", "labels": ["Company"], "properties": {"name": "TechCorp"}},
                    {"id": "4", "labels": ["Location"], "properties": {"name": "NYC"}}
                ],
                "relationships": [
                    {"startNode": "1", "endNode": "2", "type": "KNOWS"},
                    {"startNode": "1", "endNode": "3", "type": "WORKS_FOR"},
                    {"startNode": "3", "endNode": "4", "type": "LOCATED_IN"}
                ]
            }
            st.session_state.graph_data = sample_data
            st.success("✅ Test graph loaded!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load test data: {e}")
    
    # Recent conversations
    st.markdown("#### 📝 History")
    if st.session_state.conversation_history:
        for item in reversed(st.session_state.conversation_history[-2:]):  # Show last 2
            with st.expander(f"💬 {item['question'][:30]}...", expanded=False):
                st.write(f"**Time:** {item['timestamp'][:19]}")
                st.write(f"**Answer:** {item['answer'][:100]}...")
    else:
        st.info("No history yet.")
    
    if st.button("🗑️ Clear All"):
        st.session_state.conversation_history = []
        st.session_state.graph_data = None
        st.session_state.last_response = None
        st.session_state.error_count = 0
        st.rerun()

with col2:
    st.markdown("### 🕸️ Graph Visualization")
    
    # Response display
    if st.session_state.last_response:
        answer = st.session_state.last_response.get("answer", "")
        if answer:
            with st.container():
                st.markdown("#### 🤖 Latest Response")
                # Clean answer display
                clean_answer = answer.replace("**", "").replace("#", "").strip()
                st.info(clean_answer[:300] + "..." if len(clean_answer) > 300 else clean_answer)
    
    # Graph section
    if st.session_state.graph_data:
        nodes = st.session_state.graph_data.get("nodes", [])
        relationships = st.session_state.graph_data.get("relationships", [])
        
        # Statistics
        col2_1, col2_2, col2_3 = st.columns(3)
        with col2_1:
            st.markdown(f'<div class="metric-container"><strong>{len(nodes)}</strong><br>Nodes</div>', unsafe_allow_html=True)
        with col2_2:
            st.markdown(f'<div class="metric-container"><strong>{len(relationships)}</strong><br>Edges</div>', unsafe_allow_html=True)
        with col2_3:
            density = len(relationships) / max(len(nodes), 1)
            st.markdown(f'<div class="metric-container"><strong>{density:.1f}</strong><br>Density</div>', unsafe_allow_html=True)
        
        # Legend
        legend = create_simple_legend(nodes, relationships)
        st.markdown(legend, unsafe_allow_html=True)
        
        # Render graph
        st.markdown("#### 🎨 Interactive Graph")
        with st.container():
            success = render_stable_graph(st.session_state.graph_data)
            
            if success:
                st.success(f"✅ Graph rendered successfully!")
                
                # Simple controls
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🔄 Refresh"):
                        st.rerun()
                with col_b:
                    if st.button("📊 Get Stats"):
                        st.info(f"Nodes: {len(nodes)}, Relationships: {len(relationships)}")
            else:
                st.error("❌ Graph rendering failed. Try with simpler data or check the debug info above.")
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="
            text-align: center; 
            padding: 2rem; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            border-radius: 1rem; 
            margin: 1rem 0;
        ">
            <h3>🎯 Neo4j Graph Explorer</h3>
            <p>Visualize your Neo4j database with interactive graphs</p>
            <p><strong>✨ Features:</strong></p>
            <p>🎨 Colored nodes • 🔗 Relationship visualization • 🖱️ Interactive controls</p>
            <p><em>Click "Load Test Graph" to see a demo!</em></p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 0.5rem;">
    <small>🚀 <strong>Neo4j Graph Explorer</strong> | Stable Version | 
    <a href="http://localhost:8081/docs" target="_blank">API</a></small>
</div>
""", unsafe_allow_html=True)
