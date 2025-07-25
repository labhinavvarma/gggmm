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

# Page configuration for schema-aware unlimited display
st.set_page_config(
    page_title="Neo4j Graph Explorer - SCHEMA-AWARE", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for schema-aware display
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        max-width: 100%;
    }
    
    .schema-aware-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .schema-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .unlimited-badge {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .schema-panel {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .unlimited-panel {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .chat-history-panel {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border: 2px solid #667eea;
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
        border-left: 4px solid #667eea;
    }
    
    .chat-entry:hover {
        background: #f8f9fa;
        cursor: pointer;
        border-left: 4px solid #764ba2;
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
    
    .schema-item {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 5px;
        font-family: monospace;
    }
    
    .warning-panel {
        background: linear-gradient(135deg, #ffa502, #ff6348);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Chat history persistence
CHAT_HISTORY_FILE = Path("schema_aware_chat_history.pkl")

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
    """Add a new entry to schema-aware chat history"""
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
        "session_id": st.session_state.session_id,
        "unlimited_mode": True,
        "schema_aware": True,
        "schema_context": response_data.get("schema_context", {})
    }
    
    st.session_state.chat_history.insert(0, entry)
    st.session_state.chat_history = st.session_state.chat_history[:100]
    save_chat_history(st.session_state.chat_history)

# Initialize session state for schema-aware unlimited display
def init_session_state():
    defaults = {
        "graph_data": None,
        "last_response": None,
        "session_id": str(uuid.uuid4()),
        "debug_mode": False,
        "chat_history": load_chat_history(),
        "selected_history_entry": None,
        "unlimited_mode": True,
        "schema_aware": True,
        "schema_status": None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# Load schema status
def load_schema_status():
    """Load the current schema status from the API"""
    try:
        response = requests.get("http://localhost:8020/schema/status", timeout=10)
        if response.ok:
            st.session_state.schema_status = response.json()
        return st.session_state.schema_status
    except:
        return None

# Header for schema-aware unlimited display
st.markdown('''
<div class="schema-aware-header">
    <h1>🧠 Neo4j Graph Explorer - SCHEMA-AWARE MODE</h1>
    <p><strong>INTELLIGENT RESPONSES</strong> • <strong>NO NODE LIMITS</strong> • <strong>COMPLETE DATA DISPLAY</strong></p>
    <div>
        <span class="schema-badge">SCHEMA-AWARE</span>
        <span class="unlimited-badge">UNLIMITED</span>
        <span class="schema-badge">INTELLIGENT</span>
        <span class="unlimited-badge">NO LIMITS</span>
    </div>
</div>
''', unsafe_allow_html=True)

def display_schema_status():
    """Display the current schema status"""
    schema_status = load_schema_status()
    
    if schema_status:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f'''
            <div class="stats-panel">
                <h3>📊 Node Labels</h3>
                <h2>{schema_status['statistics']['total_labels']}</h2>
                <small>Types Available</small>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="stats-panel">
                <h3>🔗 Relationships</h3>
                <h2>{schema_status['statistics']['total_relationships']}</h2>
                <small>Types Available</small>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="stats-panel">
                <h3>📝 Properties</h3>
                <h2>{schema_status['statistics']['total_properties']}</h2>
                <small>Total Available</small>
            </div>
            ''', unsafe_allow_html=True)
        
        # Show last updated
        if schema_status.get('last_updated'):
            last_updated = datetime.fromisoformat(schema_status['last_updated'].replace('Z', '+00:00'))
            st.markdown(f"**🕐 Schema Last Updated:** {last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return schema_status
    else:
        st.error("❌ Could not load schema status")
        return None

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

def create_schema_aware_graph_visualization(graph_data: dict) -> bool:
    """Create SCHEMA-AWARE UNLIMITED graph visualization"""
    
    if not graph_data:
        st.info("🔍 No graph data provided")
        return False
    
    try:
        nodes = graph_data.get("nodes", [])
        relationships = graph_data.get("relationships", [])
        
        if st.session_state.debug_mode:
            st.write(f"**SCHEMA-AWARE Debug:** Processing {len(nodes)} nodes, {len(relationships)} relationships")
        
        if not nodes:
            st.warning("📊 No nodes found in graph data")
            return False
        
        # Display schema-aware unlimited mode info
        if len(nodes) > 1000:
            st.markdown(f'''
            <div class="warning-panel">
                <h3>⚠️ SCHEMA-AWARE UNLIMITED MODE</h3>
                <p>Processing {len(nodes)} nodes and {len(relationships)} relationships</p>
                <p>Using database schema for intelligent visualization</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Create network for schema-aware unlimited display
        net = Network(
            height="800px",
            width="100%", 
            bgcolor="#FFFFFF",
            font_color="#2C3E50",
            directed=True,
            select_menu=True,
            filter_menu=True
        )
        
        # Schema-aware enhanced color scheme
        node_colors = {
            "EDA": "#E74C3C",
            "Person": "#3498DB",
            "User": "#9B59B6",
            "Company": "#27AE60",
            "Department": "#F39C12",
            "Group": "#E67E22",
            "Team": "#8E44AD",
            "Project": "#16A085",
            "Movie": "#2980B9",
            "Product": "#D35400",
            "Actor": "#1ABC9C",
            "Director": "#E67E22",
            "Default": "#95A5A6"
        }
        
        relationship_colors = {
            "WORKS_IN": "#3498DB",
            "MANAGES": "#E74C3C",
            "REPORTS_TO": "#9B59B6",
            "MEMBER_OF": "#27AE60",
            "COLLABORATES": "#F39C12",
            "KNOWS": "#E67E22",
            "LEADS": "#8E44AD",
            "BELONGS_TO": "#16A085",
            "ACTED_IN": "#2980B9",
            "DIRECTED": "#D35400",
            "PRODUCED": "#1ABC9C",
            "Default": "#34495E"
        }
        
        # Process ALL nodes with schema awareness
        node_mapping = {}
        added_nodes = 0
        
        for i, node in enumerate(nodes):
            try:
                original_id = str(node.get("id", f"node_{i}"))
                display_id = f"n_{i}"
                
                node_mapping[original_id] = display_id
                
                display_name = safe_extract_node_name(node)
                labels = node.get("labels", ["Unknown"])
                primary_label = labels[0] if labels else "Unknown"
                
                color = node_colors.get(primary_label, node_colors["Default"])
                
                props = node.get("properties", {})
                tooltip_parts = [
                    f"🧠 Schema-Aware Visualization",
                    f"🏷️ Type: {primary_label}",
                    f"📛 Name: {display_name}",
                    f"🆔 ID: {original_id}",
                    f"🔗 Labels: {', '.join(labels)}"
                ]
                
                # Add key properties to tooltip
                for key, value in list(props.items())[:5]:
                    if key not in ['name', 'title', 'displayName'] and value:
                        tooltip_parts.append(f"📝 {key}: {str(value)[:50]}")
                
                tooltip = "\\n".join(tooltip_parts)
                
                # Add node with schema-aware styling
                net.add_node(
                    display_id,
                    label=display_name,
                    color={
                        'background': color,
                        'border': '#2C3E50',
                        'highlight': {'background': color, 'border': '#667eea'},
                        'hover': {'background': color, 'border': '#764ba2'}
                    },
                    size=25,
                    title=tooltip,
                    font={
                        'size': 14,
                        'color': '#FFFFFF',
                        'face': 'Arial',
                        'strokeWidth': 2,
                        'strokeColor': '#2C3E50'
                    },
                    borderWidth=2,
                    shadow={'enabled': True, 'color': 'rgba(102, 126, 234, 0.2)', 'size': 5}
                )
                
                added_nodes += 1
                
                if st.session_state.debug_mode and i < 10:
                    st.write(f"✅ Schema-aware node: {original_id} → {display_id} ({display_name})")
                
            except Exception as e:
                st.warning(f"⚠️ Error processing node {i}: {str(e)}")
                continue
        
        st.write(f"✅ **SCHEMA-AWARE: Added {added_nodes} nodes successfully**")
        
        # Process ALL relationships with schema awareness
        added_edges = 0
        
        for i, rel in enumerate(relationships):
            try:
                start_raw = str(rel.get("startNode", ""))
                end_raw = str(rel.get("endNode", ""))
                rel_type = str(rel.get("type", "CONNECTED"))
                
                start_id = node_mapping.get(start_raw)
                end_id = node_mapping.get(end_raw)
                
                if start_id and end_id:
                    rel_color = relationship_colors.get(rel_type, relationship_colors["Default"])
                    
                    rel_props = rel.get("properties", {})
                    rel_tooltip_parts = [
                        f"🧠 Schema-Aware Relationship",
                        f"🔗 Type: {rel_type}"
                    ]
                    for key, value in list(rel_props.items())[:3]:
                        rel_tooltip_parts.append(f"📝 {key}: {str(value)[:30]}")
                    rel_tooltip = "\\n".join(rel_tooltip_parts)
                    
                    # Add edge with schema-aware styling
                    net.add_edge(
                        start_id,
                        end_id,
                        label=rel_type,
                        color={
                            'color': rel_color,
                            'highlight': '#667eea',
                            'hover': '#764ba2',
                            'opacity': 0.7
                        },
                        width=3,
                        title=rel_tooltip,
                        font={
                            'size': 12,
                            'color': '#2C3E50',
                            'face': 'Arial',
                            'strokeWidth': 1,
                            'strokeColor': '#FFFFFF',
                            'align': 'middle'
                        },
                        arrows={
                            'to': {
                                'enabled': True,
                                'scaleFactor': 1.0,
                                'type': 'arrow'
                            }
                        },
                        smooth={
                            'enabled': True,
                            'type': 'dynamic',
                            'roundness': 0.2
                        },
                        shadow={'enabled': True, 'color': 'rgba(102, 126, 234, 0.1)', 'size': 3}
                    )
                    
                    added_edges += 1
                    
            except Exception as e:
                continue
        
        st.write(f"✅ **SCHEMA-AWARE: Added {added_edges} relationships successfully**")
        
        # Enhanced physics for schema-aware display
        net.set_options("""
        var options = {
          "configure": {
            "enabled": true
          },
          "edges": {
            "color": {
              "inherit": false
            },
            "smooth": {
              "enabled": true,
              "type": "dynamic",
              "roundness": 0.2
            },
            "width": 3,
            "selectionWidth": 5
          },
          "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "size": 25
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "enabled": true,
              "iterations": 200,
              "updateInterval": 50
            },
            "barnesHut": {
              "theta": 0.5,
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 120,
              "springConstant": 0.04,
              "damping": 0.15,
              "avoidOverlap": 0.1
            },
            "maxVelocity": 30,
            "minVelocity": 1,
            "timestep": 0.5
          },
          "interaction": {
            "hover": true,
            "hoverConnectedEdges": true,
            "selectConnectedEdges": true,
            "tooltipDelay": 300,
            "zoomView": true,
            "dragView": true
          },
          "layout": {
            "improvedLayout": true,
            "clusterThreshold": 500
          }
        }
        """)
        
        # Save and display with schema-aware wrapper
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            net.save_graph(f.name)
            html_file = f.name
        
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Schema-aware unlimited display wrapper
        wrapped_html = f"""
        <div style="border: 3px solid #667eea; border-radius: 15px; overflow: hidden; background: #ffffff; box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);">
            <div style="background: linear-gradient(90deg, #667eea, #764ba2); color: white; padding: 12px 20px; font-weight: bold; display: flex; justify-content: space-between; align-items: center;">
                <span>🧠 SCHEMA-AWARE Neo4j Graph Visualization</span>
                <span>{added_nodes} Nodes • {added_edges} Relationships • INTELLIGENT & UNLIMITED</span>
            </div>
            <div style="position: relative;">
                {html_content}
            </div>
        </div>
        """
        
        # Display with enhanced height
        components.html(wrapped_html, height=850, scrolling=False)
        
        # Show schema-aware unlimited statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'''
            <div class="stats-panel">
                <h3>📊 Nodes</h3>
                <h2>{added_nodes}</h2>
                <small>SCHEMA-AWARE</small>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="stats-panel">
                <h3>🔗 Relationships</h3>
                <h2>{added_edges}</h2>
                <small>VALIDATED</small>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            connectivity = f"{(added_edges / max(added_nodes, 1) * 100):.1f}%"
            st.markdown(f'''
            <div class="stats-panel">
                <h3>🌐 Connectivity</h3>
                <h2>{connectivity}</h2>
                <small>ANALYZED</small>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'''
            <div class="stats-panel">
                <h3>🧠 Mode</h3>
                <h2>∞+🧠</h2>
                <small>SMART & UNLIMITED</small>
            </div>
            ''', unsafe_allow_html=True)
        
        # Cleanup
        try:
            os.unlink(html_file)
        except:
            pass
        
        return True
        
    except Exception as e:
        st.error(f"❌ Schema-aware graph creation failed: {str(e)}")
        if st.session_state.debug_mode:
            st.code(traceback.format_exc())
        return False

def call_schema_aware_agent_api(question: str) -> dict:
    """Call the schema-aware unlimited API"""
    try:
        api_url = "http://localhost:8020/chat"
        
        payload = {
            "question": question,
            "session_id": st.session_state.session_id,
            "node_limit": None,  # Always unlimited
            "use_schema": True   # Enable schema awareness
        }
        
        with st.spinner("🧠 Processing schema-aware unlimited question..."):
            response = requests.post(api_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # Add to schema-aware chat history
            graph_data = result.get("graph_data")
            add_to_chat_history(question, result, graph_data)
            
            return result
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to schema-aware agent API. Make sure the server is running on port 8020.")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. Large schema-aware datasets may take time to process.")
        return None
    except Exception as e:
        st.error(f"❌ Schema-aware API Error: {str(e)}")
        return None

def display_schema_aware_chat_history():
    """Display schema-aware chat history"""
    if not st.session_state.chat_history:
        st.info("📭 No schema-aware chat history yet. Start asking questions!")
        return
    
    st.markdown("### 📚 Schema-Aware Chat History")
    
    # Chat history controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search schema-aware history:", placeholder="Search questions, queries, or answers...")
    
    with col2:
        show_successful_only = st.checkbox("✅ Successful only", value=False)
    
    with col3:
        if st.button("🗑️ Clear Schema-Aware History"):
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
    
    st.write(f"**Showing {len(filtered_history)} of {len(st.session_state.chat_history)} schema-aware entries**")
    
    # Display chat entries
    for i, entry in enumerate(filtered_history[:20]):
        with st.container():
            timestamp = datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            success_icon = "✅" if entry.get("success", False) else "❌"
            graph_icon = "🧠" if entry.get("has_graph_data", False) else "📊"
            
            # Chat entry header
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{success_icon} {entry['question']}**")
                st.markdown('<span class="schema-badge">SCHEMA-AWARE</span><span class="unlimited-badge">UNLIMITED</span>', unsafe_allow_html=True)
            
            with col2:
                st.text(f"{graph_icon} {entry.get('node_count', 0)}N/{entry.get('relationship_count', 0)}R")
            
            with col3:
                if st.button("🔄 Replay", key=f"schema_replay_{i}"):
                    st.session_state.selected_history_entry = entry
                    st.rerun()
            
            # Chat entry details
            with st.expander(f"📋 Schema-Aware Details - {timestamp}"):
                st.markdown(f"**🕐 Time:** {timestamp}")
                st.markdown(f"**⚡ Execution:** {entry.get('execution_time', 0):.1f}ms")
                st.markdown(f"**🔧 Tool:** {entry.get('tool', 'unknown')}")
                st.markdown('<span class="schema-badge">SCHEMA-VALIDATED</span><span class="unlimited-badge">NO LIMITS</span>', unsafe_allow_html=True)
                
                if entry.get('query'):
                    st.markdown("**🔍 Generated Query (Schema-Aware):**")
                    st.code(entry['query'], language='cypher')
                
                if entry.get('answer'):
                    st.markdown("**💬 Answer:**")
                    st.markdown(entry['answer'])
                
                if entry.get('has_graph_data'):
                    st.markdown(f"**🧠 Schema-Aware Graph Data:** {entry.get('node_count', 0)} nodes, {entry.get('relationship_count', 0)} relationships")
                
                # Show schema context if available
                schema_context = entry.get('schema_context', {})
                if schema_context:
                    st.markdown(f"**📊 Schema Context:** {schema_context.get('labels_count', 0)} labels, {schema_context.get('relationships_count', 0)} relationships")
            
            st.divider()

# Main layout for schema-aware unlimited display
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown("### 🧠 Schema-Aware Controls")
    
    # Schema status display
    st.markdown("#### 📊 Database Schema Status")
    schema_status = display_schema_status()
    
    if schema_status:
        # Schema refresh button
        if st.button("🔄 Refresh Schema", use_container_width=True):
            try:
                response = requests.post("http://localhost:8020/schema/refresh", timeout=30)
                if response.ok:
                    st.success("✅ Schema refreshed successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to refresh schema")
            except Exception as e:
                st.error(f"❌ Error refreshing schema: {e}")
        
        # Show schema details in expander
        with st.expander("🔍 View Schema Details", expanded=False):
            st.markdown("**📊 Available Node Labels:**")
            for label in schema_status.get('labels', [])[:10]:
                st.markdown(f'<div class="schema-item">{label}</div>', unsafe_allow_html=True)
            if len(schema_status.get('labels', [])) > 10:
                st.markdown(f"... and {len(schema_status['labels']) - 10} more")
            
            st.markdown("**🔗 Available Relationships:**")
            for rel in schema_status.get('relationship_types', [])[:10]:
                st.markdown(f'<div class="schema-item">{rel}</div>', unsafe_allow_html=True)
            if len(schema_status.get('relationship_types', [])) > 10:
                st.markdown(f"... and {len(schema_status['relationship_types']) - 10} more")
    
    # Schema-aware mode indicator
    st.markdown('''
    <div class="schema-panel">
        <h3>🧠 SCHEMA-AWARE MODE ACTIVE</h3>
        <p>• Complete database schema loaded</p>
        <p>• Intelligent query generation</p>
        <p>• Schema validation enabled</p>
        <p>• Structured responses</p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="unlimited-panel">
        <h3>🚀 UNLIMITED MODE ACTIVE</h3>
        <p>• No artificial node limits</p>
        <p>• Complete data display</p>
        <p>• Command-based visualization</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Debug mode toggle
    st.session_state.debug_mode = st.checkbox("🐛 Debug Mode", value=st.session_state.debug_mode)
    
    # Schema-aware sample questions
    st.markdown("#### 💡 Schema-Aware Sample Questions")
    
    # Get schema-aware examples
    try:
        response = requests.get("http://localhost:8020/schema/examples", timeout=10)
        if response.ok:
            examples = response.json().get('schema_aware_examples', [])
            for example in examples[:6]:  # Show first 6 examples
                if st.button(f"🧠 {example['query']}", use_container_width=True):
                    result = call_schema_aware_agent_api(example['query'])
                    if result:
                        st.session_state.last_response = result
                        if result.get("graph_data"):
                            st.session_state.graph_data = result["graph_data"]
                        st.rerun()
    except:
        # Fallback examples
        schema_aware_questions = [
            "Show me all the data types in my database",
            "Display nodes with their relationships", 
            "What are the main entities in my graph?",
            "Show me the complete network structure",
            "Find all the relationships available",
            "Explore the database structure"
        ]
        
        for question in schema_aware_questions:
            if st.button(f"🧠 {question}", use_container_width=True):
                result = call_schema_aware_agent_api(question)
                if result:
                    st.session_state.last_response = result
                    if result.get("graph_data"):
                        st.session_state.graph_data = result["graph_data"]
                    st.rerun()
    
    st.divider()
    
    # Custom schema-aware question input
    st.markdown("#### ✍️ Ask Schema-Aware Question")
    
    with st.form("schema_aware_question_form"):
        user_question = st.text_area(
            "Enter your schema-aware question:",
            placeholder="e.g., Show me all Person nodes with their Company relationships",
            height=80
        )
        
        st.markdown('''
        <div class="schema-panel">
            <h4>🧠 SCHEMA-AWARE UNLIMITED</h4>
            <p>Will use database schema for intelligent responses</p>
            <p>Shows ALL results - no artificial limits</p>
        </div>
        ''', unsafe_allow_html=True)
        
        submit_question = st.form_submit_button("🧠 Ask Schema-Aware", use_container_width=True)
    
    if submit_question and user_question.strip():
        result = call_schema_aware_agent_api(user_question.strip())
        if result:
            st.session_state.last_response = result
            if result.get("graph_data"):
                st.session_state.graph_data = result["graph_data"]
            st.rerun()
    
    st.divider()
    
    # Schema-aware Chat History Section
    display_schema_aware_chat_history()

with col2:
    st.markdown("### 🧠 Schema-Aware Graph Visualization")
    
    # Handle replay from schema-aware chat history
    if st.session_state.selected_history_entry:
        entry = st.session_state.selected_history_entry
        st.info(f"🔄 Replaying schema-aware: {entry['question']}")
        
        # Simulate the schema-aware response
        st.session_state.last_response = {
            "tool": entry.get("tool", ""),
            "query": entry.get("query", ""),
            "answer": entry.get("answer", ""),
            "success": entry.get("success", False),
            "unlimited_mode": True,
            "schema_aware": True,
            "schema_context": entry.get("schema_context", {})
        }
        
        # Re-execute the schema-aware query
        if entry.get("query"):
            try:
                response = requests.post("http://localhost:8000/read_neo4j_cypher",
                                       json={"query": entry["query"], "params": {}, "node_limit": None})
                if response.ok:
                    data = response.json()
                    if data.get("graph_data"):
                        st.session_state.graph_data = data["graph_data"]
            except:
                pass
        
        st.session_state.selected_history_entry = None
    
    # Show last schema-aware response
    if st.session_state.last_response:
        answer = st.session_state.last_response.get("answer", "")
        query = st.session_state.last_response.get("query", "")
        schema_context = st.session_state.last_response.get("schema_context", {})
        
        if answer:
            st.markdown(f'''
            <div class="success-panel">
                <h4>🧠 Schema-Aware Response</h4>
                <p>{answer}</p>
                <span class="schema-badge">SCHEMA-VALIDATED</span>
                <span class="unlimited-badge">NO LIMITS</span>
            </div>
            ''', unsafe_allow_html=True)
        
        if query:
            st.markdown("**🔍 Generated Schema-Aware Query:**")
            st.code(query, language='cypher')
        
        if schema_context:
            st.markdown(f"**📊 Schema Context:** {schema_context.get('labels_count', 0)} labels, {schema_context.get('relationships_count', 0)} relationships used")
    
    # Render the schema-aware graph
    if st.session_state.graph_data:
        success = create_schema_aware_graph_visualization(st.session_state.graph_data)
        
        if success:
            st.success("🎉 **Schema-aware unlimited graph rendered successfully - ALL data with intelligent structure!**")
        else:
            st.error("❌ **Schema-aware graph rendering failed.** Check debug information above.")
    
    else:
        # Schema-aware welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 4rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin: 2rem 0;">
            <h2>🧠 SCHEMA-AWARE Neo4j Explorer</h2>
            <p><strong>INTELLIGENT RESPONSES • NO NODE LIMITS • COMPLETE DATA DISPLAY</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **🧠 SCHEMA-AWARE Features:**
        - 🏗️ **Complete schema loading** - Knows your exact database structure
        - 🎯 **Intelligent query generation** - Uses actual labels and relationships
        - ✅ **Schema validation** - Prevents errors with invalid queries
        - 📊 **Structured responses** - Provides detailed explanations based on your data model
        - 🚫 **No artificial limits** - Shows ALL data according to your commands
        - 🔍 **Smart suggestions** - Recommends valid queries based on your schema
        
        **🧠 Try schema-aware commands:**
        - "Show me all the data types in my database"
        - "Display nodes with their relationships" 
        - "What are the main entities in my graph?"
        - "Find all available relationship types"
        
        **🎯 Benefits:**
        - Get accurate responses based on your actual data structure
        - No more guessing about available labels or relationships
        - Intelligent suggestions and error prevention
        - Complete visualization without artificial limits
        """)

# Enhanced schema-aware footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <strong>🧠 SCHEMA-AWARE Neo4j Graph Explorer</strong><br>
    Intelligent Responses • No Node Limits • Complete Data Display<br>
    <span class="schema-badge">SCHEMA-AWARE</span>
    <span class="unlimited-badge">UNLIMITED</span>
    <small>Chat entries: {len(st.session_state.chat_history)} | Session: {st.session_state.session_id[:8]}...</small>
</div>
""", unsafe_allow_html=True)
