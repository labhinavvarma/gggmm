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
import colorsys

# Page configuration
st.set_page_config(
    page_title="Neo4j Graph Explorer - Browser Experience", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with Neo4j Browser styling
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
    
    .schema-panel {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border: 1px solid #008cc1;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .node-type-badge {
        display: inline-block;
        padding: 4px 12px;
        margin: 2px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .relationship-badge {
        display: inline-block;
        padding: 3px 10px;
        margin: 2px;
        border-radius: 15px;
        font-size: 11px;
        font-weight: 600;
        color: white;
        background: linear-gradient(45deg, #667eea, #764ba2);
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }
    
    .graph-controls {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .schema-stats {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .query-panel {
        background: #f8f9fa;
        border: 2px solid #008cc1;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .cypher-display {
        background: #2d3748;
        color: #68d391;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        border-left: 4px solid #68d391;
        margin: 0.5rem 0;
    }
    
    .stButton button {
        background: linear-gradient(45deg, #008cc1, #0056d6);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 140, 193, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 140, 193, 0.4);
        background: linear-gradient(45deg, #0056d6, #008cc1);
    }
    
    .graph-wrapper {
        border: 3px solid #008cc1;
        border-radius: 15px;
        overflow: hidden;
        background: #ffffff;
        box-shadow: 0 8px 32px rgba(0, 140, 193, 0.2);
        margin: 1rem 0;
    }
    
    .graph-header {
        background: linear-gradient(90deg, #008cc1, #0056d6);
        color: white;
        padding: 12px 20px;
        font-weight: bold;
        font-size: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with Neo4j Browser-like features
def init_session_state():
    defaults = {
        "conversation_history": [],
        "graph_data": None,
        "last_response": None,
        "session_id": str(uuid.uuid4()),
        "connection_status": "unknown",
        "schema_info": None,
        "node_types": [],
        "relationship_types": [],
        "graph_layout": "physics",
        "show_labels": True,
        "show_relationships": True,
        "node_size_mode": "degree",
        "color_scheme": "smart",
        "unlimited_display": True,
        "schema_loaded": False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# Neo4j Browser-like header
st.markdown('''
<div class="neo4j-header">
    <h1>🗄️ Neo4j Graph Explorer</h1>
    <p><strong>Complete Schema Visualization</strong> • <strong>Unlimited Graph Display</strong> • <strong>Neo4j Browser Experience</strong></p>
</div>
''', unsafe_allow_html=True)

def get_schema_information():
    """Fetch complete schema information from the API"""
    try:
        response = requests.get("http://localhost:8020/schema/summary", timeout=10)
        if response.ok:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Schema fetch error: {str(e)}")
        return None

def generate_smart_colors(items, base_hue=0.6):
    """Generate visually distinct colors for graph elements"""
    colors = []
    for i, item in enumerate(items):
        # Use golden ratio for optimal color distribution
        hue = (base_hue + i * 0.618033988749895) % 1
        saturation = 0.7 + (i % 3) * 0.1  # Vary saturation
        lightness = 0.5 + (i % 2) * 0.15   # Vary lightness
        
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        colors.append(hex_color)
    
    return colors

def get_neo4j_style_node_color(labels, node_types=None):
    """Get Neo4j Browser-style colors for nodes"""
    if not labels:
        return "#BDC3C7"
    
    label = labels[0] if isinstance(labels, list) else str(labels)
    
    # Neo4j Browser inspired color palette
    neo4j_colors = {
        "Person": "#DA7194",       # Pink
        "User": "#4C8EDA",         # Blue  
        "Employee": "#DA7194",     # Pink
        "Customer": "#F79767",     # Orange
        "Movie": "#8DCC93",        # Green
        "Film": "#8DCC93",         # Green
        "Actor": "#F25A29",        # Red-Orange
        "Director": "#A29BFE",     # Purple
        "Producer": "#6C5CE7",     # Deep Purple
        "Company": "#00B894",      # Teal
        "Organization": "#00B894", # Teal
        "Department": "#FDCB6E",   # Yellow
        "Product": "#E84393",      # Magenta
        "Service": "#00CEC9",      # Cyan
        "Location": "#A29BFE",     # Purple
        "City": "#6C5CE7",         # Deep Purple
        "Country": "#5F27CD",      # Dark Purple
        "Event": "#FF7675",        # Light Red
        "Project": "#74B9FF",      # Light Blue
        "Database": "#00B894",     # Teal
        "Server": "#636E72",       # Gray
        "Network": "#2D3436",      # Dark Gray
        "Group": "#00CEC9",        # Cyan
        "Team": "#74B9FF",         # Light Blue
        "Document": "#DDD",        # Light Gray
        "File": "#B2BEC3",         # Gray
        "Category": "#FDCB6E",     # Yellow
        "Tag": "#A29BFE"           # Purple
    }
    
    return neo4j_colors.get(label, "#95A5A6")

def get_neo4j_style_relationship_color(rel_type):
    """Get Neo4j Browser-style colors for relationships"""
    colors = {
        "KNOWS": "#DA7194",        # Pink
        "FRIEND_OF": "#DA7194",    # Pink
        "WORKS_FOR": "#4C8EDA",    # Blue
        "WORKS_IN": "#4C8EDA",     # Blue
        "MANAGES": "#6C5CE7",      # Purple
        "REPORTS_TO": "#A29BFE",   # Light Purple
        "LOCATED_IN": "#F79767",   # Orange
        "LIVES_IN": "#F79767",     # Orange
        "BELONGS_TO": "#8DCC93",   # Green
        "OWNS": "#00B894",         # Teal
        "CREATED": "#FDCB6E",      # Yellow
        "USES": "#E84393",         # Magenta
        "ACTED_IN": "#8DCC93",     # Green
        "DIRECTED": "#6C5CE7",     # Purple
        "PRODUCED": "#00CEC9",     # Cyan
        "LOVES": "#E84393",        # Magenta
        "MARRIED_TO": "#DA7194",   # Pink
        "CONNECTED": "#95A5A6",    # Gray
        "RELATED": "#95A5A6"       # Gray
    }
    
    return colors.get(rel_type, "#95A5A6")

def create_neo4j_browser_like_graph(graph_data: dict, unlimited_display: bool = True) -> bool:
    """Create a Neo4j Browser-like graph visualization with unlimited display capability"""
    
    if not graph_data:
        st.info("🔍 No graph data available for visualization.")
        return False
    
    try:
        nodes = graph_data.get("nodes", [])
        relationships = graph_data.get("relationships", [])
        
        if not nodes:
            st.info("📊 No nodes found in the current dataset.")
            return False
        
        # Show comprehensive statistics
        total_nodes = len(nodes)
        total_relationships = len(relationships)
        
        st.markdown(f'''
        <div class="schema-stats">
            <h3>🕸️ Complete Graph Visualization</h3>
            <p><strong>{total_nodes:,} Nodes</strong> • <strong>{total_relationships:,} Relationships</strong> • <strong>Unlimited Display</strong></p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Create Neo4j-style network with enhanced settings
        net = Network(
            height="800px",  # Taller for better view
            width="100%", 
            bgcolor="#FFFFFF",
            font_color="#2C3E50",
            directed=True,
            select_menu=False,  # Disable selection menu for cleaner look
            filter_menu=False   # Disable filter menu
        )
        
        # Process all nodes without artificial limits
        added_nodes = set()
        node_stats = {}
        
        st.info(f"🎨 Processing {total_nodes:,} nodes for Neo4j Browser-like visualization...")
        
        # Calculate node degrees for sizing
        node_degrees = {}
        for rel in relationships:
            start_id = str(rel.get("startNode", ""))
            end_id = str(rel.get("endNode", ""))
            node_degrees[start_id] = node_degrees.get(start_id, 0) + 1
            node_degrees[end_id] = node_degrees.get(end_id, 0) + 1
        
        for i, node in enumerate(nodes):
            try:
                node_id = f"node_{i}"
                raw_id = str(node.get("id", f"node_{i}"))
                
                # Extract display information
                display_name = safe_extract_node_name(node)
                labels = node.get("labels", ["Unknown"])
                primary_label = labels[0] if labels else "Unknown"
                
                # Track node types for statistics
                node_stats[primary_label] = node_stats.get(primary_label, 0) + 1
                
                # Neo4j Browser-style sizing based on connectivity
                base_size = 25
                degree = node_degrees.get(raw_id, 0)
                if degree > 10:
                    size = base_size + 20  # Hub nodes
                elif degree > 5:
                    size = base_size + 10  # Well-connected nodes
                elif degree > 2:
                    size = base_size + 5   # Connected nodes
                else:
                    size = base_size       # Regular nodes
                
                # Neo4j Browser-style colors
                color = get_neo4j_style_node_color(labels)
                
                # Enhanced tooltip with comprehensive information
                props = node.get("properties", {})
                tooltip_parts = [
                    f"🏷️ Type: {primary_label}",
                    f"📛 Name: {display_name}",
                    f"🔗 Connections: {degree}"
                ]
                
                # Add key properties
                prop_count = 0
                for key, value in props.items():
                    if key not in ['name', 'title', 'displayName'] and prop_count < 5:
                        tooltip_parts.append(f"📝 {key}: {str(value)[:50]}...")
                        prop_count += 1
                
                if len(props) > prop_count + 3:
                    tooltip_parts.append(f"📋 +{len(props) - prop_count - 3} more properties")
                
                tooltip = "\\n".join(tooltip_parts)
                
                # Add node with Neo4j Browser styling
                net.add_node(
                    node_id,
                    label=display_name,
                    color={
                        'background': color,
                        'border': '#2C3E50',
                        'highlight': {
                            'background': color,
                            'border': '#E74C3C'
                        },
                        'hover': {
                            'background': color,
                            'border': '#F39C12'
                        }
                    },
                    size=size,
                    title=tooltip,
                    font={
                        'size': max(14, min(20, 12 + degree // 2)),  # Dynamic font size
                        'color': '#FFFFFF',
                        'face': 'Arial',
                        'strokeWidth': 2,
                        'strokeColor': '#2C3E50'
                    },
                    borderWidth=2,
                    borderWidthSelected=4,
                    shadow={
                        'enabled': True,
                        'color': 'rgba(0,0,0,0.3)',
                        'size': 8,
                        'x': 2,
                        'y': 2
                    },
                    margin={
                        'top': 8,
                        'bottom': 8,
                        'left': 8,
                        'right': 8
                    }
                )
                
                added_nodes.add((raw_id, node_id))
                
            except Exception as e:
                st.warning(f"⚠️ Skipped node {i}: {str(e)}")
                continue
        
        # Process all relationships without limits
        id_mapping = dict(added_nodes)
        simple_nodes = {node_id for _, node_id in added_nodes}
        
        added_edges = 0
        relationship_stats = {}
        
        for i, rel in enumerate(relationships):
            try:
                start_raw = str(rel.get("startNode", ""))
                end_raw = str(rel.get("endNode", ""))
                rel_type = str(rel.get("type", "CONNECTED"))
                
                # Track relationship types
                relationship_stats[rel_type] = relationship_stats.get(rel_type, 0) + 1
                
                start_id = id_mapping.get(start_raw)
                end_id = id_mapping.get(end_raw)
                
                if start_id and end_id and start_id in simple_nodes and end_id in simple_nodes:
                    color = get_neo4j_style_relationship_color(rel_type)
                    
                    # Enhanced relationship properties
                    rel_props = rel.get("properties", {})
                    rel_tooltip_parts = [f"Type: {rel_type}"]
                    for key, value in list(rel_props.items())[:3]:
                        rel_tooltip_parts.append(f"{key}: {str(value)[:30]}")
                    rel_tooltip = "\\n".join(rel_tooltip_parts)
                    
                    # Add relationship with Neo4j Browser styling
                    net.add_edge(
                        start_id,
                        end_id,
                        label=rel_type,
                        color={
                            'color': color,
                            'highlight': '#E74C3C',
                            'hover': '#F39C12'
                        },
                        width=3,
                        title=rel_tooltip,
                        font={
                            'size': 12,
                            'color': '#2C3E50',
                            'face': 'Arial',
                            'strokeWidth': 2,
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
                        shadow={
                            'enabled': True,
                            'color': 'rgba(0,0,0,0.1)',
                            'size': 4,
                            'x': 1,
                            'y': 1
                        }
                    )
                    
                    added_edges += 1
                    
            except Exception as e:
                st.warning(f"⚠️ Skipped relationship {i}: {str(e)}")
                continue
        
        # Neo4j Browser-like physics configuration for stability
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
              "roundness": 0.2
            }
          },
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
              "theta": 0.5,
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 120,
              "springConstant": 0.04,
              "damping": 0.09,
              "avoidOverlap": 0.1
            },
            "maxVelocity": 50,
            "minVelocity": 0.75,
            "timestep": 0.5
          },
          "interaction": {
            "hover": true,
            "hoverConnectedEdges": true,
            "selectConnectedEdges": false,
            "tooltipDelay": 200,
            "zoomView": true,
            "dragView": true
          },
          "layout": {
            "improvedLayout": true,
            "clusterThreshold": 150,
            "hierarchical": false
          }
        }
        """)
        
        st.success(f"✅ **Neo4j Browser-Like Graph Created:** {len(simple_nodes):,} nodes, {added_edges:,} relationships")
        
        # Display statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Node Type Distribution")
            for node_type, count in sorted(node_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                color = get_neo4j_style_node_color([node_type])
                st.markdown(f'''
                <div class="node-type-badge" style="background-color: {color};">
                    {node_type}: {count:,}
                </div>
                ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🔗 Relationship Type Distribution")
            for rel_type, count in sorted(relationship_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                st.markdown(f'''
                <div class="relationship-badge">
                    {rel_type}: {count:,}
                </div>
                ''', unsafe_allow_html=True)
        
        # Generate and display the graph
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            net.save_graph(f.name)
            html_file = f.name
        
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Neo4j Browser-like wrapper with enhanced styling
        wrapped_html = f"""
        <div class="graph-wrapper">
            <div class="graph-header">
                <span>🕸️ Complete Neo4j Graph Visualization</span>
                <span>{len(simple_nodes):,} Nodes • {added_edges:,} Relationships • Unlimited Display</span>
            </div>
            <div style="position: relative;">
                {html_content}
            </div>
        </div>
        """
        
        # Display with increased height for better viewing
        components.html(wrapped_html, height=850, scrolling=False)
        
        # Cleanup
        try:
            os.unlink(html_file)
        except:
            pass
        
        return True
        
    except Exception as e:
        st.error(f"❌ Neo4j Browser-like graph rendering failed: {str(e)}")
        
        with st.expander("🔍 Debug Information"):
            st.code(str(e))
            st.code(traceback.format_exc())
        
        return False

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
            props.get("displayName"),
            props.get("username"),
            props.get("fullName"),
            props.get("firstName")
        ]
        
        for name in name_options:
            if name and str(name).strip():
                return str(name).strip()[:30]
        
        # Fallback to label + ID
        if labels and labels[0] != "Unknown":
            short_id = node_id.split(":")[-1][-4:] if ":" in node_id else node_id[-4:]
            return f"{labels[0]}_{short_id}"
        
        return f"Node_{node_id[-6:] if len(node_id) > 6 else node_id}"
        
    except Exception as e:
        return f"Node_{hash(str(node)) % 10000}"

def call_agent_api(question: str, node_limit: int = None) -> dict:
    """Enhanced API call with unlimited support"""
    try:
        api_url = "http://localhost:8020/chat"
        
        # Use very high limit for unlimited display
        effective_limit = node_limit if node_limit else 50000
        
        payload = {
            "question": question,
            "session_id": st.session_state.session_id,
            "node_limit": effective_limit
        }
        
        with st.spinner("🤖 Processing with unlimited graph capability..."):
            response = requests.post(api_url, json=payload, timeout=120)  # Longer timeout
            response.raise_for_status()
            result = response.json()
            
            st.session_state.connection_status = "connected"
            return result
            
    except requests.exceptions.ConnectionError:
        st.session_state.connection_status = "disconnected"
        st.error("❌ Cannot connect to agent API. Please ensure the server is running on port 8020.")
        return None
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
        return None

# Main layout
col1, col2 = st.columns([1, 3], gap="large")

with col1:
    st.markdown("### 🎛️ Neo4j Browser Controls")
    
    # Connection status
    status_colors = {"connected": "🟢", "disconnected": "🔴", "unknown": "⚪"}
    st.markdown(f'''
    <div class="schema-panel">
        <strong>Connection Status:</strong> {status_colors.get(st.session_state.connection_status, "⚪")} {st.session_state.connection_status}
    </div>
    ''', unsafe_allow_html=True)
    
    # Schema information panel
    st.markdown("#### 📊 Database Schema")
    schema_info = get_schema_information()
    
    if schema_info and schema_info.get("status") == "success":
        schema_data = schema_info.get("schema_info", {})
        
        st.markdown(f'''
        <div class="schema-panel">
            <h4>🧠 Schema Intelligence</h4>
            <p><strong>Node Types:</strong> {schema_data.get("node_types", 0)}</p>
            <p><strong>Relationship Types:</strong> {schema_data.get("relationship_types", 0)}</p>
            <p><strong>Properties:</strong> {schema_data.get("property_keys", 0)}</p>
            <p><strong>Status:</strong> ✅ Schema-Aware</p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.session_state.schema_loaded = True
        
        if st.button("🔄 Refresh Schema", use_container_width=True):
            try:
                refresh_response = requests.post("http://localhost:8020/schema/refresh", timeout=30)
                if refresh_response.ok:
                    st.success("✅ Schema refreshed successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to refresh schema")
            except Exception as e:
                st.error(f"❌ Schema refresh error: {str(e)}")
    else:
        st.warning("⚠️ Schema information not available")
    
    st.divider()
    
    # Neo4j Browser-like query suggestions
    st.markdown("#### 💡 Neo4j Browser-Style Queries")
    st.info("✨ **Unlimited Display** - Show complete graphs without artificial limits")
    
    neo4j_queries = [
        ("Complete Graph", "Show me the entire graph structure"),
        ("Schema Overview", "Display the complete database schema"),
        ("All Node Types", "Show me all different types of nodes"),
        ("All Relationships", "Display all relationship types"),
        ("Connected Components", "Find all connected components"),
        ("Hub Nodes", "Show me the most connected nodes"),
        ("Network Paths", "Display network paths and connections"),
        ("Graph Statistics", "Show me comprehensive graph statistics")
    ]
    
    for i, (name, query) in enumerate(neo4j_queries):
        if st.button(f"🔍 {name}", key=f"neo4j_query_{i}", use_container_width=True):
            result = call_agent_api(query)
            if result:
                if result.get("graph_data"):
                    st.session_state.graph_data = result["graph_data"]
                st.session_state.last_response = result
                st.rerun()
    
    st.divider()
    
    # Custom query input
    st.markdown("#### ✍️ Custom Cypher Query")
    
    with st.form("neo4j_query_form"):
        user_question = st.text_area(
            "Ask anything about your graph:",
            placeholder="e.g., Show me all nodes connected to Person nodes through any relationship",
            height=100
        )
        
        unlimited_mode = st.checkbox(
            "🚀 Unlimited Display Mode", 
            value=True,
            help="Remove all limits and show the complete graph structure"
        )
        
        submit_button = st.form_submit_button("🚀 Execute Query", use_container_width=True)
    
    if submit_button and user_question.strip():
        node_limit = None if unlimited_mode else 1000
        result = call_agent_api(user_question.strip(), node_limit)
        
        if result:
            if result.get("graph_data"):
                st.session_state.graph_data = result["graph_data"]
            st.session_state.last_response = result
            st.success("✅ Query executed with unlimited display capability!")
            st.rerun()
    
    st.divider()
    
    # Graph controls
    st.markdown("#### 🎨 Visualization Controls")
    
    if st.button("🕸️ Load Complete Schema Graph", use_container_width=True):
        result = call_agent_api("Show me the complete database structure with all nodes and relationships")
        if result:
            if result.get("graph_data"):
                st.session_state.graph_data = result["graph_data"]
            st.session_state.last_response = result
            st.rerun()
    
    if st.button("📊 Load Sample Network", use_container_width=True):
        result = call_agent_api("Show me a comprehensive sample of the network structure")
        if result:
            if result.get("graph_data"):
                st.session_state.graph_data = result["graph_data"]
            st.session_state.last_response = result
            st.rerun()
    
    if st.button("🗑️ Clear Graph", use_container_width=True):
        st.session_state.graph_data = None
        st.session_state.last_response = None
        st.success("🧹 Graph cleared!")
        st.rerun()

with col2:
    st.markdown("### 🕸️ Neo4j Browser-Like Visualization")
    
    # Show last response if available
    if st.session_state.last_response:
        answer = st.session_state.last_response.get("answer", "")
        if answer:
            st.markdown(f'''
            <div class="query-panel">
                <h4>🤖 Query Response</h4>
                <p>{answer}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Show executed query
        query = st.session_state.last_response.get("query", "")
        if query:
            st.markdown(f'''
            <div class="cypher-display">
                <strong>Executed Cypher Query:</strong><br>
                {query}
            </div>
            ''', unsafe_allow_html=True)
    
    # Render the graph
    if st.session_state.graph_data:
        success = create_neo4j_browser_like_graph(
            st.session_state.graph_data, 
            unlimited_display=True
        )
        
        if success:
            nodes_count = len(st.session_state.graph_data.get("nodes", []))
            rels_count = len(st.session_state.graph_data.get("relationships", []))
            
            st.markdown(f'''
            <div class="schema-stats">
                🎉 <strong>Success!</strong> Neo4j Browser-like visualization created with {nodes_count:,} nodes and {rels_count:,} relationships!
                <br><strong>✨ Unlimited Display Mode Active</strong> - Complete graph structure shown
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.error("❌ Graph rendering failed. Check the debug information above.")
    
    else:
        # Welcome screen with Neo4j styling
        st.markdown("""
        <div style="text-align: center; padding: 4rem; background: linear-gradient(135deg, #008cc1 0%, #0056d6 100%); color: white; border-radius: 15px; margin: 2rem 0; box-shadow: 0 8px 32px rgba(0, 140, 193, 0.3);">
            <h2>🗄️ Neo4j Browser Experience</h2>
            <p><strong>Complete Schema Visualization • Unlimited Graph Display • Enhanced Stability</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f8f9fa, #e9ecef); color: #2c3e50; padding: 2rem; border-radius: 15px; margin: 1rem 0; border: 2px solid #008cc1;">
            <h3 style="text-align: center; margin-top: 0; color: #008cc1;">🚀 Enhanced Features:</h3>
            <div style="text-align: left;">
                <p>🕸️ <strong>Unlimited Display</strong> - Show complete graphs without artificial limits</p>
                <p>🎨 <strong>Neo4j Browser Styling</strong> - Authentic Neo4j look and feel</p>
                <p>🧠 <strong>Schema Intelligence</strong> - Uses complete database schema for optimization</p>
                <p>⚡ <strong>Enhanced Stability</strong> - Improved physics and layout algorithms</p>
                <p>🔗 <strong>Smart Connectivity</strong> - Better relationship visualization and clustering</p>
                <p>📊 <strong>Real-time Statistics</strong> - Live node and relationship type distribution</p>
                <p>🎯 <strong>Intelligent Sizing</strong> - Node sizes based on connectivity (hub detection)</p>
                <p>🌈 <strong>Smart Colors</strong> - Neo4j-inspired color palette for optimal distinction</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Enhanced footer
st.markdown("---")
st.markdown("""
<div style="
    text-align: center; 
    color: #6c757d; 
    padding: 1.5rem;
    background: linear-gradient(90deg, rgba(0, 140, 193, 0.1), rgba(0, 86, 214, 0.1));
    border-radius: 15px;
    margin-top: 2rem;
    border: 1px solid #008cc1;
">
    <h4 style="margin: 0; background: linear-gradient(90deg, #008cc1, #0056d6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        🗄️ Neo4j Graph Explorer - Browser Experience
    </h4>
    <p style="margin: 0.5rem 0;">🕸️ Unlimited Display • 🧠 Schema Intelligence • 🎨 Neo4j Styling • ⚡ Enhanced Performance</p>
</div>
""", unsafe_allow_html=True)
