import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import requests
import json
import os
import tempfile
from datetime import datetime
import uuid

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

def get_node_color(labels):
    """Get color based on node labels"""
    if not labels:
        return "#95afc0"
    
    label = labels[0] if isinstance(labels, list) else labels
    
    color_map = {
        "Person": "#ff6b6b",
        "Movie": "#4ecdc4", 
        "Company": "#45b7d1",
        "Product": "#96ceb4",
        "Location": "#feca57",
        "Event": "#ff9ff3",
        "User": "#a55eea",
        "Order": "#26de81",
        "Category": "#fd79a8",
        "Unknown": "#95afc0"
    }
    return color_map.get(label, "#95afc0")

def render_graph(graph_data: dict) -> bool:
    """Render the graph using Pyvis with better error handling"""
    
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
        st.write(f"📊 **Graph Debug Info:** {len(nodes)} nodes, {len(relationships)} relationships")
        
        # Create Pyvis network
        net = Network(
            height="600px", 
            width="100%", 
            bgcolor="#ffffff", 
            font_color="#333333",
            directed=True
        )
        
        # Add nodes
        for node in nodes:
            node_id = str(node.get("id", "unknown"))
            properties = node.get("properties", {})
            labels = node.get("labels", ["Unknown"])
            
            # Create display label
            display_name = (
                properties.get("name") or 
                properties.get("title") or 
                properties.get("label") or 
                f"Node_{node_id}"
            )
            
            # Create hover info
            hover_info = f"ID: {node_id}\\nLabels: {', '.join(labels)}"
            for key, value in list(properties.items())[:5]:  # Show first 5 properties
                hover_info += f"\\n{key}: {str(value)[:50]}"
            
            # Add node
            net.add_node(
                node_id,
                label=str(display_name)[:30],  # Limit label length
                title=hover_info,
                color=get_node_color(labels),
                size=25,
                font={'size': 14}
            )
        
        # Add edges/relationships
        for rel in relationships:
            start_id = str(rel.get("startNode", rel.get("start", "")))
            end_id = str(rel.get("endNode", rel.get("end", "")))
            rel_type = rel.get("type", "CONNECTED")
            rel_props = rel.get("properties", {})
            
            # Only add edge if both nodes exist
            if start_id and end_id:
                # Create edge hover info
                edge_info = f"Type: {rel_type}"
                for key, value in list(rel_props.items())[:3]:
                    edge_info += f"\\n{key}: {str(value)[:30]}"
                
                net.add_edge(
                    start_id,
                    end_id,
                    label=rel_type,
                    title=edge_info,
                    color={'color': '#666666', 'highlight': '#ff0000'},
                    arrows={'to': {'enabled': True, 'scaleFactor': 1}},
                    width=2
                )
        
        # Configure physics for better layout
        net.set_options("""
        var options = {
            "physics": {
                "enabled": true,
                "stabilization": {"iterations": 100},
                "barnesHut": {
                    "gravitationalConstant": -8000,
                    "centralGravity": 0.3,
                    "springLength": 100,
                    "springConstant": 0.04,
                    "damping": 0.09,
                    "avoidOverlap": 0.1
                }
            },
            "interaction": {
                "dragNodes": true,
                "dragView": true,
                "zoomView": true
            },
            "nodes": {
                "borderWidth": 2,
                "borderWidthSelected": 4
            },
            "edges": {
                "width": 2,
                "smooth": {
                    "type": "continuous"
                }
            }
        }
        """)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            net.save_graph(f.name)
            html_file = f.name
        
        # Read and display the HTML
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Enhanced HTML wrapper
        enhanced_html = f"""
        <div style="border: 1px solid #ddd; border-radius: 8px; overflow: hidden; background: white;">
            {html_content}
        </div>
        """
        
        # Display in Streamlit
        components.html(enhanced_html, height=650, scrolling=False)
        
        # Clean up temp file
        try:
            os.unlink(html_file)
        except:
            pass
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error rendering graph: {str(e)}")
        
        # Fallback: Show raw data
        with st.expander("🔍 Raw Graph Data (Debug)", expanded=False):
            st.json(graph_data)
        
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
        formatted_answer = answer.replace('\n', '<br>').replace('**', '<strong>').replace('**', '</strong>')
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
        ("Show all nodes", "Show me all nodes in the database"),
        ("Database schema", "What is the database schema?"),
        ("Count nodes", "How many nodes are in the database?"),
        ("Person nodes", "Show me all Person nodes"),
        ("Relationships", "Display all relationships"),
        ("Sample data", "Give me a sample of the graph")
    ]
    
    for action_name, action_query in quick_actions:
        if st.button(action_name, key=f"quick_{action_name}"):
            # Execute quick action
            result = call_agent_api(action_query, node_limit=50)  # Smaller limit for quick actions
            if result:
                # Add to conversation history
                st.session_state.conversation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "question": action_query,
                    "answer": result.get("answer", ""),
                    "graph_data": result.get("graph_data")
                })
                
                # Update current graph data
                if result.get("graph_data"):
                    st.session_state.graph_data = result["graph_data"]
                    st.success("✅ Graph updated!")
                
                st.session_state.last_response = result
                st.rerun()
    
    st.divider()
    
    # Custom question input
    st.markdown("#### ✍️ Ask a Question")
    with st.form("question_form", clear_on_submit=True):
        user_question = st.text_area(
            "Your question:",
            placeholder="e.g., Show me all Person nodes, Delete a person named John, Create a new company...",
            height=100
        )
        
        node_limit = st.slider(
            "Max nodes to display:",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            help="Limit the number of nodes in the visualization for better performance"
        )
        
        submit_button = st.form_submit_button("🚀 Submit")
    
    if submit_button and user_question.strip():
        # Call the agent
        result = call_agent_api(user_question.strip(), node_limit)
        
        if result:
            # Add to conversation history
            st.session_state.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "question": user_question.strip(),
                "answer": result.get("answer", ""),
                "graph_data": result.get("graph_data"),
                "tool": result.get("tool", ""),
                "query": result.get("query", "")
            })
            
            # Update current graph data (important for auto-refresh)
            if result.get("graph_data"):
                st.session_state.graph_data = result["graph_data"]
                st.success("✅ Graph updated with latest data!")
            
            st.session_state.last_response = result
            st.rerun()
    
    st.divider()
    
    # Test graph with sample data button
    if st.button("🧪 Test Graph with Sample Data"):
        # Create sample graph data for testing
        sample_data = {
            "nodes": [
                {"id": "1", "labels": ["Person"], "properties": {"name": "Alice", "age": 30}},
                {"id": "2", "labels": ["Person"], "properties": {"name": "Bob", "age": 25}},
                {"id": "3", "labels": ["Company"], "properties": {"name": "TechCorp"}},
                {"id": "4", "labels": ["Location"], "properties": {"name": "New York"}}
            ],
            "relationships": [
                {"startNode": "1", "endNode": "2", "type": "KNOWS", "properties": {}},
                {"startNode": "1", "endNode": "3", "type": "WORKS_FOR", "properties": {}},
                {"startNode": "3", "endNode": "4", "type": "LOCATED_IN", "properties": {}}
            ]
        }
        st.session_state.graph_data = sample_data
        st.success("✅ Sample graph data loaded!")
        st.rerun()
    
    # Conversation history
    st.markdown("#### 📝 Recent Conversations")
    if st.session_state.conversation_history:
        # Show last 3 conversations
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
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.conversation_history = []
        st.session_state.graph_data = None
        st.session_state.last_response = None
        st.rerun()

with col2:
    st.markdown("### 🕸️ Graph Visualization")
    
    # Display current response if available
    if st.session_state.last_response:
        answer = st.session_state.last_response.get("answer", "")
        if answer:
            st.markdown('<div class="response-container">', unsafe_allow_html=True)
            st.markdown("#### 🤖 Latest Response")
            # Format the answer better
            formatted_answer = answer.replace('\n', '  \n')  # Markdown line breaks
            st.markdown(formatted_answer)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Debug section
    with st.expander("🔍 Debug Info", expanded=False):
        if st.session_state.graph_data:
            st.write("✅ Graph data available")
            st.write(f"Nodes: {len(st.session_state.graph_data.get('nodes', []))}")
            st.write(f"Relationships: {len(st.session_state.graph_data.get('relationships', []))}")
            
            # Show first few nodes for debugging
            nodes = st.session_state.graph_data.get('nodes', [])
            if nodes:
                st.write("Sample node:")
                st.json(nodes[0])
        else:
            st.write("❌ No graph data available")
    
    # Render graph
    if st.session_state.graph_data:
        st.markdown("#### 🎨 Interactive Graph")
        
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
        
        # Render the graph
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        success = render_graph(st.session_state.graph_data)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if success:
            st.success(f"✅ Displaying {nodes_count} nodes and {rels_count} relationships")
            
            # Graph controls
            col2_4, col2_5 = st.columns(2)
            with col2_4:
                if st.button("🔄 Refresh Graph"):
                    # Force refresh by calling a simple query
                    refresh_result = call_agent_api("Show me the current graph structure", node_limit=50)
                    if refresh_result and refresh_result.get("graph_data"):
                        st.session_state.graph_data = refresh_result["graph_data"]
                        st.rerun()
            
            with col2_5:
                if st.button("💾 Download Graph Data"):
                    # Prepare download
                    graph_json = json.dumps(st.session_state.graph_data, indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=graph_json,
                        file_name=f"neo4j_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        else:
            st.error("❌ Failed to render graph - check debug info above")
    else:
        # Welcome message when no graph data
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 1rem; margin: 2rem 0;">
            <h3>🎯 Welcome to Neo4j Graph Explorer!</h3>
            <p>Ask a question or use the quick actions to start exploring your database.</p>
            <p><strong>Try these examples:</strong></p>
            <ul style="text-align: left; display: inline-block;">
                <li>"Show me all Person nodes"</li>
                <li>"How many nodes are in the database?"</li>
                <li>"What is the database schema?"</li>
                <li>"Create a person named Alice"</li>
                <li>"Delete a person named John"</li>
            </ul>
            <p><em>💡 Or click "Test Graph with Sample Data" to see a demo visualization</em></p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p>🚀 <strong>Neo4j Graph Explorer</strong> | Built with Streamlit & LangGraph | 
    <a href="http://localhost:8081/docs" target="_blank">API Docs</a> | 
    <a href="http://localhost:8000" target="_blank">MCP Server</a>
    </p>
</div>
""", unsafe_allow_html=True)
