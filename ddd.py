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
import pickle
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Neo4j Graph Explorer - Enhanced", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better relationship visibility
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        max-width: 100%;
    }
    
    .neo4j-header {
        background: linear-gradient(135deg, #008cc1 0%, #0056d6 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 140, 193, 0.3);
    }
    
    .chat-history-panel {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border: 2px solid #008cc1;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .chat-entry {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-left: 4px solid #008cc1;
    }
    
    .chat-entry:hover {
        background: #f8f9fa;
        cursor: pointer;
        border-left: 4px solid #0056d6;
    }
    
    .chat-question {
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .chat-details {
        font-size: 0.9rem;
        color: #6c757d;
    }
    
    .chat-query {
        background: #2d3748;
        color: #68d391;
        padding: 0.5rem;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        margin: 0.3rem 0;
    }
    
    .debug-panel {
        background: #f8f9fa;
        border: 1px solid #008cc1;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .success-panel {
        background: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .stats-panel {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Chat history persistence
CHAT_HISTORY_FILE = Path("chat_history.pkl")

def load_chat_history():
    """Load chat history from file"""
    try:
        if CHAT_HISTORY_FILE.exists():
            with open(CHAT_HISTORY_FILE, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.sidebar.error(f"Error loading chat history: {e}")
    return []

def save_chat_history(history):
    """Save chat history to file"""
    try:
        with open(CHAT_HISTORY_FILE, 'wb') as f:
            pickle.dump(history, f)
    except Exception as e:
        st.sidebar.error(f"Error saving chat history: {e}")

def add_to_chat_history(question, response_data, graph_data=None):
    """Add a new entry to chat history"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "tool": response_data.get("tool", ""),
        "query": response_data.get("query", ""),
        "answer": response_data.get("answer", ""),
        "execution_time": response_data.get("execution_time_ms", 0),
        "success": response_data.get("success", False),
        "has_graph_data": graph_data is not None,
        "node_count": len(graph_data.get("nodes", [])) if graph_data else 0,
        "relationship_count": len(graph_data.get("relationships", [])) if graph_data else 0,
        "session_id": st.session_state.session_id
    }
    
    st.session_state.chat_history.insert(0, entry)  # Add to beginning
    
    # Keep only last 100 entries
    st.session_state.chat_history = st.session_state.chat_history[:100]
    
    # Save to file
    save_chat_history(st.session_state.chat_history)

# Initialize session state
def init_session_state():
    defaults = {
        "graph_data": None,
        "last_response": None,
        "session_id": str(uuid.uuid4()),
        "debug_mode": False,
        "chat_history": load_chat_history(),
        "selected_history_entry": None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# Header
st.markdown('''
<div class="neo4j-header">
    <h1>🗄️ Neo4j Graph Explorer - Enhanced</h1>
    <p><strong>Fixed Relationships</strong> • <strong>Complete Chat History</strong> • <strong>Enhanced Visualization</strong></p>
</div>
''', unsafe_allow_html=True)

def safe_extract_node_name(node):
    """Safely extract display name from node"""
    try:
        props = node.get("properties", {})
        labels = node.get("labels", ["Unknown"])
        node_id = str(node.get("id", ""))
        
        # Try different name properties
        for name_prop in ["name", "title", "displayName", "username", "fullName", "firstName"]:
            if name_prop in props and props[name_prop]:
                return str(props[name_prop]).strip()[:25]
        
        # Fallback to label + ID
        if labels and labels[0] != "Unknown":
            short_id = node_id.split(":")[-1][-4:] if ":" in node_id else node_id[-4:]
            return f"{labels[0]}_{short_id}"
        
        return f"Node_{node_id[-6:] if len(node_id) > 6 else node_id}"
        
    except Exception:
        return f"Node_{hash(str(node)) % 10000}"

def create_enhanced_graph_with_fixed_relationships(graph_data: dict) -> bool:
    """Create enhanced graph with properly visible relationships"""
    
    if not graph_data:
        st.info("🔍 No graph data provided")
        return False
    
    try:
        nodes = graph_data.get("nodes", [])
        relationships = graph_data.get("relationships", [])
        
        if st.session_state.debug_mode:
            st.write(f"**Debug:** Processing {len(nodes)} nodes, {len(relationships)} relationships")
        
        if not nodes:
            st.warning("📊 No nodes found in graph data")
            return False
        
        # Create network with enhanced relationship visibility settings
        net = Network(
            height="750px",
            width="100%", 
            bgcolor="#FFFFFF",
            font_color="#2C3E50",
            directed=True,
            select_menu=False,
            filter_menu=False
        )
        
        # Enhanced color scheme
        node_colors = {
            "EDA": "#E74C3C",        # Red
            "Person": "#3498DB",     # Blue  
            "User": "#9B59B6",       # Purple
            "Company": "#27AE60",    # Green
            "Department": "#F39C12", # Orange
            "Group": "#E67E22",      # Dark Orange
            "Team": "#8E44AD",       # Dark Purple
            "Project": "#16A085",    # Teal
            "Default": "#95A5A6"     # Gray
        }
        
        # Enhanced relationship colors
        relationship_colors = {
            "WORKS_IN": "#3498DB",    # Blue
            "MANAGES": "#E74C3C",     # Red
            "REPORTS_TO": "#9B59B6",  # Purple
            "MEMBER_OF": "#27AE60",   # Green
            "COLLABORATES": "#F39C12", # Orange
            "KNOWS": "#E67E22",       # Dark Orange
            "LEADS": "#8E44AD",       # Dark Purple
            "BELONGS_TO": "#16A085",  # Teal
            "Default": "#34495E"      # Dark Gray
        }
        
        # Process nodes with enhanced properties
        node_mapping = {}
        added_nodes = 0
        
        for i, node in enumerate(nodes):
            try:
                # Use consistent node ID
                original_id = str(node.get("id", f"node_{i}"))
                display_id = f"n_{i}"  # Simple display ID
                
                # Store mapping for relationships
                node_mapping[original_id] = display_id
                
                # Extract display information
                display_name = safe_extract_node_name(node)
                labels = node.get("labels", ["Unknown"])
                primary_label = labels[0] if labels else "Unknown"
                
                # Get enhanced color
                color = node_colors.get(primary_label, node_colors["Default"])
                
                # Create detailed tooltip
                props = node.get("properties", {})
                tooltip_parts = [
                    f"🏷️ Type: {primary_label}",
                    f"📛 Name: {display_name}",
                    f"🆔 ID: {original_id}"
                ]
                
                # Add key properties
                for key, value in list(props.items())[:4]:
                    if key not in ['name', 'title', 'displayName'] and value:
                        tooltip_parts.append(f"📝 {key}: {str(value)[:30]}")
                
                tooltip = "\\n".join(tooltip_parts)
                
                # Add node with enhanced styling
                net.add_node(
                    display_id,
                    label=display_name,
                    color={
                        'background': color,
                        'border': '#2C3E50',
                        'highlight': {'background': color, 'border': '#E74C3C'},
                        'hover': {'background': color, 'border': '#F39C12'}
                    },
                    size=30,
                    title=tooltip,
                    font={
                        'size': 16,
                        'color': '#FFFFFF',
                        'face': 'Arial',
                        'strokeWidth': 3,
                        'strokeColor': '#2C3E50'
                    },
                    borderWidth=3,
                    shadow={'enabled': True, 'color': 'rgba(0,0,0,0.3)', 'size': 10}
                )
                
                added_nodes += 1
                
                if st.session_state.debug_mode:
                    st.write(f"✅ Added node: {original_id} → {display_id} ({display_name})")
                
            except Exception as e:
                st.warning(f"⚠️ Error processing node {i}: {str(e)}")
                continue
        
        st.write(f"✅ **Added {added_nodes} nodes successfully**")
        
        # Process relationships with enhanced visibility
        added_edges = 0
        relationship_debug = []
        
        for i, rel in enumerate(relationships):
            try:
                start_raw = str(rel.get("startNode", ""))
                end_raw = str(rel.get("endNode", ""))
                rel_type = str(rel.get("type", "CONNECTED"))
                
                # Map to display IDs
                start_id = node_mapping.get(start_raw)
                end_id = node_mapping.get(end_raw)
                
                if start_id and end_id:
                    # Get relationship color
                    rel_color = relationship_colors.get(rel_type, relationship_colors["Default"])
                    
                    # Create relationship tooltip
                    rel_props = rel.get("properties", {})
                    rel_tooltip_parts = [f"🔗 Type: {rel_type}"]
                    for key, value in list(rel_props.items())[:3]:
                        rel_tooltip_parts.append(f"📝 {key}: {str(value)[:25]}")
                    rel_tooltip = "\\n".join(rel_tooltip_parts)
                    
                    # Add edge with enhanced visibility
                    net.add_edge(
                        start_id,
                        end_id,
                        label=rel_type,
                        color={
                            'color': rel_color,
                            'highlight': '#E74C3C',
                            'hover': '#F39C12',
                            'opacity': 0.8
                        },
                        width=4,  # Thicker lines for better visibility
                        title=rel_tooltip,
                        font={
                            'size': 14,
                            'color': '#2C3E50',
                            'face': 'Arial',
                            'strokeWidth': 2,
                            'strokeColor': '#FFFFFF',
                            'align': 'middle'
                        },
                        arrows={
                            'to': {
                                'enabled': True,
                                'scaleFactor': 1.2,
                                'type': 'arrow'
                            }
                        },
                        smooth={
                            'enabled': True,
                            'type': 'dynamic',
                            'roundness': 0.3
                        },
                        shadow={'enabled': True, 'color': 'rgba(0,0,0,0.2)', 'size': 6}
                    )
                    
                    added_edges += 1
                    
                    if st.session_state.debug_mode:
                        relationship_debug.append(f"✅ {start_raw} → {end_raw} ({rel_type})")
                        
                else:
                    if st.session_state.debug_mode:
                        relationship_debug.append(f"❌ Missing nodes: {start_raw} → {end_raw}")
                    
            except Exception as e:
                st.warning(f"⚠️ Error processing relationship {i}: {str(e)}")
                relationship_debug.append(f"💥 Error: {str(e)}")
                continue
        
        st.write(f"✅ **Added {added_edges} relationships successfully**")
        
        if st.session_state.debug_mode and relationship_debug:
            with st.expander("🔍 Relationship Debug Details"):
                for debug_msg in relationship_debug[:10]:  # Show first 10
                    st.text(debug_msg)
        
        if added_edges == 0 and len(relationships) > 0:
            st.error("❌ **No relationships were rendered!** Check node ID mapping.")
            if st.session_state.debug_mode:
                st.write("**Node mapping:**", node_mapping)
                st.write("**Sample relationship:**", relationships[0] if relationships else "None")
        
        # Enhanced physics configuration for better relationship visibility
        net.set_options("""
        var options = {
          "configure": {
            "enabled": false
          },
          "edges": {
            "color": {
              "inherit": false
            },
            "smooth": {
              "enabled": true,
              "type": "dynamic",
              "roundness": 0.3
            },
            "width": 4,
            "selectionWidth": 6
          },
          "nodes": {
            "borderWidth": 3,
            "borderWidthSelected": 5
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "enabled": true,
              "iterations": 150,
              "updateInterval": 25
            },
            "barnesHut": {
              "theta": 0.4,
              "gravitationalConstant": -12000,
              "centralGravity": 0.4,
              "springLength": 150,
              "springConstant": 0.05,
              "damping": 0.1,
              "avoidOverlap": 0.2
            },
            "maxVelocity": 40,
            "minVelocity": 0.75,
            "timestep": 0.5
          },
          "interaction": {
            "hover": true,
            "hoverConnectedEdges": true,
            "selectConnectedEdges": true,
            "tooltipDelay": 200,
            "zoomView": true,
            "dragView": true
          },
          "layout": {
            "improvedLayout": true,
            "clusterThreshold": 150
          }
        }
        """)
        
        # Save and display
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            net.save_graph(f.name)
            html_file = f.name
        
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Enhanced wrapper with statistics
        wrapped_html = f"""
        <div style="border: 3px solid #008cc1; border-radius: 15px; overflow: hidden; background: #ffffff; box-shadow: 0 8px 32px rgba(0, 140, 193, 0.2);">
            <div style="background: linear-gradient(90deg, #008cc1, #0056d6); color: white; padding: 12px 20px; font-weight: bold; display: flex; justify-content: space-between; align-items: center;">
                <span>🕸️ Enhanced Neo4j Graph Visualization</span>
                <span>{added_nodes} Nodes • {added_edges} Relationships • Fixed Visibility</span>
            </div>
            <div style="position: relative;">
                {html_content}
            </div>
        </div>
        """
        
        # Display with enhanced height
        components.html(wrapped_html, height=800, scrolling=False)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'''
            <div class="stats-panel">
                <h3>📊 Nodes</h3>
                <h2>{added_nodes}</h2>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="stats-panel">
                <h3>🔗 Relationships</h3>
                <h2>{added_edges}</h2>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            connectivity = f"{(added_edges / max(added_nodes, 1) * 100):.1f}%"
            st.markdown(f'''
            <div class="stats-panel">
                <h3>🌐 Connectivity</h3>
                <h2>{connectivity}</h2>
            </div>
            ''', unsafe_allow_html=True)
        
        # Cleanup
        try:
            os.unlink(html_file)
        except:
            pass
        
        return True
        
    except Exception as e:
        st.error(f"❌ Graph creation failed: {str(e)}")
        if st.session_state.debug_mode:
            st.code(traceback.format_exc())
        return False

def call_agent_api(question: str, node_limit: int = 100) -> dict:
    """Enhanced API call with chat history integration"""
    try:
        api_url = "http://localhost:8020/chat"
        
        payload = {
            "question": question,
            "session_id": st.session_state.session_id,
            "node_limit": node_limit
        }
        
        with st.spinner("🤖 Processing question..."):
            response = requests.post(api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            # Add to chat history
            graph_data = result.get("graph_data")
            add_to_chat_history(question, result, graph_data)
            
            return result
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to agent API. Make sure the server is running on port 8020.")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. Try a simpler question.")
        return None
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
        return None

def display_chat_history():
    """Display comprehensive chat history"""
    if not st.session_state.chat_history:
        st.info("📭 No chat history yet. Start asking questions!")
        return
    
    st.markdown("### 📚 Complete Chat History")
    
    # Chat history controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search chat history:", placeholder="Search questions, queries, or answers...")
    
    with col2:
        show_successful_only = st.checkbox("✅ Successful only", value=False)
    
    with col3:
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            save_chat_history([])
            st.rerun()
    
    # Filter chat history
    filtered_history = st.session_state.chat_history
    
    if search_term:
        filtered_history = [
            entry for entry in filtered_history
            if search_term.lower() in entry.get("question", "").lower() or
               search_term.lower() in entry.get("query", "").lower() or
               search_term.lower() in entry.get("answer", "").lower()
        ]
    
    if show_successful_only:
        filtered_history = [entry for entry in filtered_history if entry.get("success", False)]
    
    st.write(f"**Showing {len(filtered_history)} of {len(st.session_state.chat_history)} entries**")
    
    # Display chat entries
    for i, entry in enumerate(filtered_history[:20]):  # Show last 20 entries
        with st.container():
            # Create clickable chat entry
            entry_key = f"chat_entry_{i}"
            
            timestamp = datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            success_icon = "✅" if entry.get("success", False) else "❌"
            graph_icon = "🕸️" if entry.get("has_graph_data", False) else "📊"
            
            # Chat entry header
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{success_icon} {entry['question']}**")
            
            with col2:
                st.text(f"{graph_icon} {entry.get('node_count', 0)}N/{entry.get('relationship_count', 0)}R")
            
            with col3:
                if st.button("🔄 Replay", key=f"replay_{i}"):
                    st.session_state.selected_history_entry = entry
                    st.rerun()
            
            # Chat entry details in expander
            with st.expander(f"📋 Details - {timestamp}"):
                st.markdown(f"**🕐 Time:** {timestamp}")
                st.markdown(f"**⚡ Execution:** {entry.get('execution_time', 0):.1f}ms")
                st.markdown(f"**🔧 Tool:** {entry.get('tool', 'unknown')}")
                
                if entry.get('query'):
                    st.markdown("**🔍 Generated Query:**")
                    st.code(entry['query'], language='cypher')
                
                if entry.get('answer'):
                    st.markdown("**💬 Answer:**")
                    st.markdown(entry['answer'])
                
                if entry.get('has_graph_data'):
                    st.markdown(f"**🕸️ Graph Data:** {entry.get('node_count', 0)} nodes, {entry.get('relationship_count', 0)} relationships")
            
            st.divider()

# Main layout
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown("### 🎛️ Enhanced Controls")
    
    # Debug mode toggle
    st.session_state.debug_mode = st.checkbox("🐛 Debug Mode", value=st.session_state.debug_mode)
    
    # Quick test buttons
    with st.expander("🧪 Quick Tests", expanded=False):
        if st.button("🔍 Test Database", use_container_width=True):
            try:
                response = requests.post("http://localhost:8000/read_neo4j_cypher",
                                       json={"query": "MATCH (n) RETURN count(n) as total"})
                if response.ok:
                    total = response.json()['data'][0]['total']
                    st.success(f"✅ Database: {total} nodes")
                else:
                    st.error("❌ Database connection failed")
            except:
                st.error("❌ Cannot connect to database")
        
        if st.button("🕸️ Test Relationships", use_container_width=True):
            try:
                response = requests.post("http://localhost:8000/read_neo4j_cypher",
                                       json={"query": "MATCH ()-[r]->() RETURN count(r) as total"})
                if response.ok:
                    total = response.json()['data'][0]['total']
                    st.success(f"✅ Relationships: {total}")
                else:
                    st.error("❌ Relationship query failed")
            except:
                st.error("❌ Cannot test relationships")
    
    # Quick sample questions
    st.markdown("#### 💡 Sample Questions")
    sample_questions = [
        "Show me EDA group with relationships",
        "Display all nodes and connections", 
        "Find the network structure",
        "Show me all departments",
        "Display Person nodes with relationships"
    ]
    
    for question in sample_questions:
        if st.button(f"💬 {question}", use_container_width=True):
            result = call_agent_api(question)
            if result:
                st.session_state.last_response = result
                if result.get("graph_data"):
                    st.session_state.graph_data = result["graph_data"]
                st.rerun()
    
    st.divider()
    
    # Custom question input
    st.markdown("#### ✍️ Ask Your Question")
    
    with st.form("question_form"):
        user_question = st.text_area(
            "Enter your question:",
            placeholder="e.g., Show me all teams and their members",
            height=80
        )
        
        node_limit = st.slider("Node Limit:", 10, 500, 100)
        submit_question = st.form_submit_button("🚀 Ask", use_container_width=True)
    
    if submit_question and user_question.strip():
        result = call_agent_api(user_question.strip(), node_limit)
        if result:
            st.session_state.last_response = result
            if result.get("graph_data"):
                st.session_state.graph_data = result["graph_data"]
            st.rerun()
    
    st.divider()
    
    # Chat History Section
    display_chat_history()

with col2:
    st.markdown("### 🕸️ Enhanced Graph Visualization")
    
    # Handle replay from chat history
    if st.session_state.selected_history_entry:
        entry = st.session_state.selected_history_entry
        st.info(f"🔄 Replaying: {entry['question']}")
        
        # Simulate the response
        st.session_state.last_response = {
            "tool": entry.get("tool", ""),
            "query": entry.get("query", ""),
            "answer": entry.get("answer", ""),
            "success": entry.get("success", False)
        }
        
        # Re-execute the query to get graph data
        if entry.get("query"):
            try:
                response = requests.post("http://localhost:8000/read_neo4j_cypher",
                                       json={"query": entry["query"], "params": {}, "node_limit": 100})
                if response.ok:
                    data = response.json()
                    if data.get("graph_data"):
                        st.session_state.graph_data = data["graph_data"]
            except:
                pass
        
        st.session_state.selected_history_entry = None
    
    # Show last response
    if st.session_state.last_response:
        answer = st.session_state.last_response.get("answer", "")
        query = st.session_state.last_response.get("query", "")
        
        if answer:
            st.markdown(f'''
            <div class="success-panel">
                <h4>🤖 Response</h4>
                <p>{answer}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        if query:
            st.markdown("**🔍 Generated Query:**")
            st.code(query, language='cypher')
    
    # Render the enhanced graph
    if st.session_state.graph_data:
        success = create_enhanced_graph_with_fixed_relationships(st.session_state.graph_data)
        
        if success:
            st.success("🎉 **Graph rendered successfully with visible relationships!**")
        else:
            st.error("❌ **Graph rendering failed.** Check debug information above.")
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 4rem; background: linear-gradient(135deg, #008cc1 0%, #0056d6 100%); color: white; border-radius: 15px; margin: 2rem 0;">
            <h2>🚀 Enhanced Neo4j Explorer</h2>
            <p><strong>✨ Fixed Relationships • 📚 Complete Chat History • 🎨 Better Visualization</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **🎯 New Features:**
        - 🔗 **Fixed relationship visibility** - Thicker lines, better colors, enhanced physics
        - 📚 **Complete chat history** - All your questions and answers stored
        - 🔄 **Replay functionality** - Click replay on any previous question
        - 🔍 **Search chat history** - Find previous conversations easily
        - 📊 **Enhanced statistics** - Node counts, relationship counts, connectivity metrics
        - 🐛 **Better debugging** - More detailed information when things go wrong
        
        **🚀 Try asking:**
        - "Show me EDA group with relationships"
        - "Display all nodes and connections"
        - "Find the network structure"
        """)

# Enhanced footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <strong>🚀 Enhanced Neo4j Graph Explorer</strong><br>
    Fixed Relationships • Complete Chat History • Enhanced Visualization<br>
    <small>Chat entries: {len(st.session_state.chat_history)} | Session: {st.session_state.session_id[:8]}...</small>
</div>
""", unsafe_allow_html=True)
