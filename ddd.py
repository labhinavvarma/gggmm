import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import networkx as nx
import requests
import json
import os
from datetime import datetime

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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None
if "last_response" not in st.session_state:
    st.session_state.last_response = None

# Header
st.markdown('<h1 class="main-header">🕸️ Neo4j Graph Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6c757d;">Ask questions, run queries, and visualize your Neo4j database in real-time</p>', unsafe_allow_html=True)

def call_agent_api(question: str, node_limit: int = 1000) -> dict:
    """Call the FastAPI agent endpoint"""
    try:
        api_url = "http://localhost:8081/chat"
        payload = {
            "question": question,
            "session_id": st.session_state.get("session_id", "streamlit_session"),
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

def render_graph(graph_data: dict, container_key: str = "main"):
    """Render the graph using Pyvis"""
    if not graph_data or not graph_data.get("nodes"):
        return False
    
    try:
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes with enhanced styling
        for node in graph_data["nodes"]:
            node_id = str(node["id"])
            properties = node.get("properties", {})
            labels = node.get("labels", ["Unknown"])
            
            # Create display name
            display_name = properties.get("name", properties.get("title", f"Node {node_id}"))
            
            # Add node with styling based on labels
            color = get_node_color(labels[0] if labels else "Unknown")
            G.add_node(
                node_id, 
                label=display_name,
                title=f"Labels: {', '.join(labels)}\nProperties: {json.dumps(properties, indent=2)}",
                color=color,
                size=20
            )
        
        # Add edges
        for rel in graph_data.get("relationships", []):
            start_id = str(rel["startNode"])
            end_id = str(rel["endNode"])
            rel_type = rel.get("type", "CONNECTED")
            
            if start_id in G.nodes and end_id in G.nodes:
                G.add_edge(
                    start_id, 
                    end_id, 
                    label=rel_type,
                    title=f"Type: {rel_type}\nProperties: {json.dumps(rel.get('properties', {}), indent=2)}",
                    color="#999999"
                )
        
        # Create Pyvis network
        net = Network(
            height="600px", 
            width="100%", 
            bgcolor="#ffffff", 
            font_color="#333333",
            select_menu=True,
            filter_menu=True
        )
        
        # Configure physics
        net.set_options("""
        var options = {
            "physics": {
                "enabled": true,
                "stabilization": {"iterations": 100},
                "barnesHut": {
                    "gravitationalConstant": -8000,
                    "centralGravity": 0.3,
                    "springLength": 95,
                    "springConstant": 0.04,
                    "damping": 0.09
                }
            }
        }
        """)
        
        net.from_nx(G)
        
        # Save and display
        html_path = f"graph_{container_key}.html"
        net.save_graph(html_path)
        
        with open(html_path, "r", encoding="utf-8") as f:
            html_code = f.read()
        
        # Enhanced HTML with better styling
        enhanced_html = f"""
        <div style="background: white; border-radius: 8px; padding: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            {html_code}
        </div>
        """
        
        components.html(enhanced_html, height=650)
        
        # Clean up
        if os.path.exists(html_path):
            os.remove(html_path)
        
        return True
        
    except Exception as e:
        st.error(f"Error rendering graph: {str(e)}")
        return False

def get_node_color(label: str) -> str:
    """Get color based on node label"""
    color_map = {
        "Person": "#ff6b6b",
        "Movie": "#4ecdc4", 
        "Company": "#45b7d1",
        "Product": "#96ceb4",
        "Location": "#feca57",
        "Event": "#ff9ff3",
        "Unknown": "#95afc0"
    }
    return color_map.get(label, "#95afc0")

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
        st.markdown(f"""
        <div class="response-container">
            <strong>🤖 Agent Response:</strong><br>
            {answer.replace('**', '<strong>').replace('**', '</strong>').replace('\n', '<br>')}
        </div>
        """, unsafe_allow_html=True)

# Main layout with columns
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 💬 Chat Interface")
    
    # Quick action buttons
    st.markdown("#### 🚀 Quick Actions")
    quick_actions = [
        ("Show all nodes", "MATCH (n) RETURN n LIMIT 25"),
        ("Show database schema", "What is the database schema?"),
        ("Count all nodes", "How many nodes are in the database?"),
        ("Show Person nodes", "Show me all Person nodes"),
        ("Show relationships", "Display all relationships")
    ]
    
    for action_name, action_query in quick_actions:
        if st.button(action_name, key=f"quick_{action_name}"):
            # Execute quick action
            result = call_agent_api(action_query)
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
            max_value=1000,
            value=100,
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
            st.markdown(answer)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Render graph
    if st.session_state.graph_data:
        st.markdown("#### 🎨 Interactive Graph")
        
        # Graph statistics
        nodes_count = len(st.session_state.graph_data.get("nodes", []))
        rels_count = len(st.session_state.graph_data.get("relationships", []))
        
        col2_1, col2_2, col2_3 = st.columns(3)
        with col2_1:
            st.metric("Nodes", nodes_count)
        with col2_2:
            st.metric("Relationships", rels_count)
        with col2_3:
            st.metric("Density", f"{rels_count/max(nodes_count,1):.1f}")
        
        # Render the graph
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        success = render_graph(st.session_state.graph_data, "main_display")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if success:
            st.success(f"✅ Displaying {nodes_count} nodes and {rels_count} relationships")
            
            # Graph controls
            col2_4, col2_5 = st.columns(2)
            with col2_4:
                if st.button("🔄 Refresh Graph"):
                    # Force refresh by calling a simple query
                    refresh_result = call_agent_api("Show me the current graph structure")
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
            st.error("❌ Failed to render graph")
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
