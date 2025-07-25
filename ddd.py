"""
Compatible Streamlit UI for Standalone Neo4j Server
Perfectly matched to work with the standalone_server.py
"""

import streamlit as st
import requests
import uuid
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
import streamlit.components.v1 as components

# ============================================
# CONFIGURATION - MATCHED TO STANDALONE SERVER
# ============================================

# Backend server configuration (matches standalone_server.py)
BACKEND_PORT = 8000
BASE_URL = f"http://localhost:{BACKEND_PORT}"

print(f"🔧 Compatible Streamlit UI Configuration:")
print(f"   Backend: Standalone Neo4j Server")
print(f"   Port: {BACKEND_PORT}")
print(f"   Base URL: {BASE_URL}")

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Neo4j Chatbot - Compatible UI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for compatibility
st.markdown("""
<style>
    .main { 
        padding-top: 1rem; 
    }
    
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .status-healthy {
        background: linear-gradient(90deg, #00d4aa 0%, #00b894 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin: 0.2rem;
        display: inline-block;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .status-error {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin: 0.2rem;
        display: inline-block;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .tool-badge {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .query-display {
        background: #1e1e1e;
        color: #f8f8f2;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        border-left: 4px solid #50fa7b;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        overflow-x: auto;
    }
    
    .result-display {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00d4aa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .connection-indicator {
        padding: 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .connected {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .disconnected {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .standalone-badge {
        background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None
if "database_stats" not in st.session_state:
    st.session_state.database_stats = {}
if "connection_status" not in st.session_state:
    st.session_state.connection_status = "checking"
if "last_update_time" not in st.session_state:
    st.session_state.last_update_time = None

# ============================================
# HELPER FUNCTIONS - COMPATIBLE WITH STANDALONE SERVER
# ============================================

@st.cache_data(ttl=10)
def check_standalone_server():
    """Check health of the standalone server"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "data": response.json()}
        else:
            return {"status": "error", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "disconnected", "error": f"Standalone server not running on port {BACKEND_PORT}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def get_database_stats():
    """Get current database statistics from standalone server"""
    try:
        response = requests.get(f"{BASE_URL}/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def get_graph_data(limit: int = 50):
    """Get graph data for visualization from standalone server"""
    try:
        response = requests.get(f"{BASE_URL}/graph?limit={limit}&include_relationships=true", timeout=15)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def send_chat_message(question: str):
    """Send chat message to standalone server"""
    try:
        payload = {
            "question": question,
            "session_id": st.session_state.session_id
        }
        
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/chat",
            json=payload,
            timeout=30
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            result["response_time"] = response_time
            return result
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "answer": f"❌ Server error: {response.status_code}",
                "response_time": response_time
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to standalone server",
            "answer": f"❌ Standalone server not running on port {BACKEND_PORT}. Please start: python standalone_server.py",
            "response_time": 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "answer": f"❌ Request failed: {str(e)}",
            "response_time": 0
        }

def create_compatible_graph_visualization(graph_data, height=600):
    """Create D3.js graph visualization compatible with standalone server data"""
    
    nodes = graph_data.get("nodes", [])
    relationships = graph_data.get("relationships", [])
    
    # Create unique key for forcing updates
    update_key = int(time.time())
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Compatible Neo4j Graph</title>
        <script src="https://unpkg.com/d3@7"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #f8f9fa;
            }}
            
            .header {{
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color: white;
                padding: 1rem;
                text-align: center;
                font-weight: bold;
            }}
            
            .controls {{
                background: white;
                padding: 1rem;
                border-bottom: 1px solid #eee;
                display: flex;
                gap: 1rem;
                align-items: center;
                flex-wrap: wrap;
                justify-content: center;
            }}
            
            .btn {{
                background: #28a745;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                transition: background 0.2s;
            }}
            
            .btn:hover {{
                background: #218838;
            }}
            
            .stats {{
                display: flex;
                gap: 1rem;
                align-items: center;
                font-size: 14px;
            }}
            
            .stat {{
                background: #f8f9fa;
                padding: 0.5rem;
                border-radius: 4px;
                border-left: 3px solid #28a745;
                font-weight: bold;
            }}
            
            #graph-container {{
                height: {height - 120}px;
                background: white;
                border-top: 1px solid #eee;
                position: relative;
            }}
            
            #graph-svg {{
                width: 100%;
                height: 100%;
            }}
            
            .node {{
                cursor: pointer;
                stroke: #fff;
                stroke-width: 2px;
            }}
            
            .link {{
                stroke: #999;
                stroke-opacity: 0.8;
                stroke-width: 2px;
                fill: none;
            }}
            
            .node-label {{
                font-size: 12px;
                font-family: Arial, sans-serif;
                fill: #333;
                text-anchor: middle;
                pointer-events: none;
                user-select: none;
            }}
            
            .link-label {{
                font-size: 10px;
                font-family: Arial, sans-serif;
                fill: #666;
                text-anchor: middle;
                pointer-events: none;
                user-select: none;
            }}
            
            .tooltip {{
                position: absolute;
                padding: 8px;
                background: rgba(0, 0, 0, 0.8);
                color: white;
                border-radius: 4px;
                font-size: 12px;
                pointer-events: none;
                z-index: 1000;
                opacity: 0;
                transition: opacity 0.2s;
            }}
            
            .loading {{
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100%;
                font-size: 18px;
                color: #666;
                flex-direction: column;
                gap: 1rem;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            🧠 Compatible Neo4j Graph - Standalone Server (Update #{update_key})
        </div>
        
        <div class="controls">
            <div class="stats">
                <div class="stat">Nodes: {len(nodes)}</div>
                <div class="stat">Relationships: {len(relationships)}</div>
                <div class="stat">Updated: {datetime.now().strftime('%H:%M:%S')}</div>
            </div>
            <button class="btn" onclick="restartSimulation()">🔄 Restart</button>
            <button class="btn" onclick="centerGraph()">🎯 Center</button>
            <button class="btn" onclick="zoomToFit()">🔍 Fit</button>
        </div>
        
        <div id="graph-container">
            {('<div class="loading">📝 No data in Neo4j database.<br>Try: "Create a Person named Alice"</div>' if len(nodes) == 0 else '<svg id="graph-svg"></svg>')}
        </div>
        
        <div class="tooltip" id="tooltip"></div>

        <script>
            // Graph data from standalone server
            const graphData = {{
                nodes: {json.dumps(nodes)},
                relationships: {json.dumps(relationships)}
            }};
            
            console.log('📊 Compatible Graph Data:', {{
                nodes: graphData.nodes.length,
                relationships: graphData.relationships.length
            }});
            
            let svg, g, simulation, link, node, nodeLabels, linkLabels;
            
            // Color scale for node labels
            const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
            
            function initializeGraph() {{
                if (graphData.nodes.length === 0) return;
                
                const container = d3.select('#graph-container');
                const containerRect = container.node().getBoundingClientRect();
                const width = containerRect.width;
                const height = containerRect.height;
                
                // Clear any existing SVG
                d3.select('#graph-svg').selectAll('*').remove();
                
                svg = d3.select('#graph-svg')
                    .attr('width', width)
                    .attr('height', height);
                
                g = svg.append('g');
                
                // Add zoom behavior
                const zoom = d3.zoom()
                    .scaleExtent([0.1, 4])
                    .on('zoom', function(event) {{
                        g.attr('transform', event.transform);
                    }});
                
                svg.call(zoom);
                
                // Process data for D3 (compatible with standalone server format)
                const nodes = graphData.nodes.map(d => ({{
                    ...d,
                    id: d.id.toString(),
                    label: d.caption || d.properties?.name || `Node ${{d.id}}`,
                    color: colorScale(d.labels?.[0] || 'default'),
                    size: 20 + (Object.keys(d.properties || {{}}).length * 2)
                }}));
                
                const links = graphData.relationships.map(d => ({{
                    ...d,
                    source: d.source.toString(),
                    target: d.target.toString(),
                    label: d.type || 'RELATES_TO'
                }}));
                
                console.log('🔗 Processed compatible data:', {{ nodes: nodes.length, links: links.length }});
                
                // Create force simulation
                simulation = d3.forceSimulation(nodes)
                    .force('link', d3.forceLink(links).id(d => d.id).distance(100).strength(0.8))
                    .force('charge', d3.forceManyBody().strength(-300))
                    .force('center', d3.forceCenter(width / 2, height / 2))
                    .force('collision', d3.forceCollide().radius(d => d.size + 5));
                
                // Create links
                link = g.append('g')
                    .selectAll('.link')
                    .data(links)
                    .enter().append('line')
                    .attr('class', 'link')
                    .style('stroke-width', 2)
                    .style('stroke', '#999')
                    .style('opacity', 0.8);
                
                // Create link labels
                linkLabels = g.append('g')
                    .selectAll('.link-label')
                    .data(links)
                    .enter().append('text')
                    .attr('class', 'link-label')
                    .text(d => d.label)
                    .style('font-size', '10px')
                    .style('fill', '#666')
                    .style('text-anchor', 'middle');
                
                // Create nodes
                node = g.append('g')
                    .selectAll('.node')
                    .data(nodes)
                    .enter().append('circle')
                    .attr('class', 'node')
                    .attr('r', d => d.size)
                    .style('fill', d => d.color)
                    .style('stroke', '#fff')
                    .style('stroke-width', 2)
                    .on('mouseover', handleMouseOver)
                    .on('mouseout', handleMouseOut)
                    .on('click', handleClick)
                    .call(d3.drag()
                        .on('start', dragstarted)
                        .on('drag', dragged)
                        .on('end', dragended));
                
                // Create node labels
                nodeLabels = g.append('g')
                    .selectAll('.node-label')
                    .data(nodes)
                    .enter().append('text')
                    .attr('class', 'node-label')
                    .text(d => d.label)
                    .style('font-size', '12px')
                    .style('fill', '#333')
                    .style('text-anchor', 'middle')
                    .style('pointer-events', 'none');
                
                // Update positions on simulation tick
                simulation.on('tick', () => {{
                    link
                        .attr('x1', d => d.source.x)
                        .attr('y1', d => d.source.y)
                        .attr('x2', d => d.target.x)
                        .attr('y2', d => d.target.y);
                    
                    linkLabels
                        .attr('x', d => (d.source.x + d.target.x) / 2)
                        .attr('y', d => (d.source.y + d.target.y) / 2);
                    
                    node
                        .attr('cx', d => d.x)
                        .attr('cy', d => d.y);
                    
                    nodeLabels
                        .attr('x', d => d.x)
                        .attr('y', d => d.y + 5);
                }});
                
                console.log('✅ Compatible graph initialized');
            }}
            
            // Event handlers
            function handleMouseOver(event, d) {{
                const tooltip = d3.select('#tooltip');
                
                let content = `<strong>${{d.label}}</strong><br/>`;
                content += `<strong>Labels:</strong> ${{d.labels?.join(', ') || 'None'}}<br/>`;
                
                if (d.properties && Object.keys(d.properties).length > 0) {{
                    content += '<strong>Properties:</strong><br/>';
                    Object.entries(d.properties).forEach(([key, value]) => {{
                        content += `  ${{key}}: ${{value}}<br/>`;
                    }});
                }}
                
                tooltip
                    .html(content)
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 10) + 'px')
                    .style('opacity', 1);
                
                d3.select(event.target)
                    .style('stroke', '#ff6b6b')
                    .style('stroke-width', 3);
            }}
            
            function handleMouseOut(event, d) {{
                d3.select('#tooltip').style('opacity', 0);
                
                d3.select(event.target)
                    .style('stroke', '#fff')
                    .style('stroke-width', 2);
            }}
            
            function handleClick(event, d) {{
                console.log('🖱️ Node clicked:', d);
            }}
            
            // Drag behavior
            function dragstarted(event, d) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }}
            
            function dragged(event, d) {{
                d.fx = event.x;
                d.fy = event.y;
            }}
            
            function dragended(event, d) {{
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }}
            
            // Control functions
            function restartSimulation() {{
                if (simulation) {{
                    simulation.alpha(1).restart();
                }}
            }}
            
            function centerGraph() {{
                if (svg) {{
                    const containerRect = d3.select('#graph-container').node().getBoundingClientRect();
                    const transform = d3.zoomIdentity.translate(containerRect.width / 2, containerRect.height / 2).scale(1);
                    svg.transition().duration(750).call(
                        svg.__zoom__.transform,
                        transform
                    );
                }}
            }}
            
            function zoomToFit() {{
                if (g && svg) {{
                    const bounds = g.node().getBBox();
                    const containerRect = d3.select('#graph-container').node().getBoundingClientRect();
                    const width = containerRect.width;
                    const height = containerRect.height;
                    
                    const scale = Math.min(width / bounds.width, height / bounds.height) * 0.8;
                    const translate = [width / 2 - scale * (bounds.x + bounds.width / 2), height / 2 - scale * (bounds.y + bounds.height / 2)];
                    
                    svg.transition().duration(750).call(
                        svg.__zoom__.transform,
                        d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
                    );
                }}
            }}
            
            // Initialize when page loads
            document.addEventListener('DOMContentLoaded', function() {{
                console.log('🚀 Initializing compatible graph visualization...');
                if (graphData.nodes.length > 0) {{
                    setTimeout(initializeGraph, 100);
                }}
            }});
            
            // Handle resize
            window.addEventListener('resize', function() {{
                if (simulation) {{
                    setTimeout(initializeGraph, 250);
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    return html_content

def update_connection_status():
    """Update connection status for standalone server"""
    health = check_standalone_server()
    st.session_state.connection_status = health["status"]
    return health

# ============================================
# ENHANCED EXAMPLE PROMPTS FOR STANDALONE SERVER
# ============================================

COMPATIBLE_EXAMPLE_PROMPTS = {
    "🔍 Basic Queries": [
        "How many nodes are in the graph?",
        "How many relationships are in the graph?",
        "Show me the database schema",
        "List all node labels",
        "List all relationship types"
    ],
    
    "👥 Create People & Companies": [
        "Create a Person named Alice with age 30",
        "Create a Person named Bob with age 25",
        "Create a Company named TechCorp",
        "Create a Company named DataInc with employees 500",
        "Create a Person named Charlie who works at Microsoft"
    ],
    
    "🔗 Create Relationships": [
        "Connect Alice to TechCorp with relationship WORKS_FOR",
        "Create a FRIENDS_WITH relationship between Alice and Bob",
        "Connect Bob to DataInc as an EMPLOYEE",
        "Create a MANAGES relationship from Charlie to Alice",
        "Connect TechCorp and DataInc with PARTNERS_WITH relationship"
    ],
    
    "📊 Data Analysis": [
        "Show me all Person nodes",
        "Find all Company nodes",
        "List all relationships",
        "Show me nodes with properties",
        "Find all connected nodes"
    ],
    
    "🏗️ Create Networks": [
        "Create a social network with 3 connected friends",
        "Create a company hierarchy with CEO and employees",
        "Build a project team with 4 people",
        "Create a family tree with parents and children",
        "Build a university structure"
    ],
    
    "🧹 Cleanup Operations": [
        "Delete all TestNode nodes",
        "Remove all test data",
        "Delete nodes with no relationships",
        "Clean up old test nodes",
        "Remove all temporary data"
    ],
    
    "🔄 Update Operations": [
        "Update all Person nodes to add status active",
        "Set the last_updated property for all nodes",
        "Update Alice's age to 31",
        "Add department property to all employees",
        "Update all nodes with current timestamp"
    ],
    
    "🎯 Advanced Queries": [
        "Find nodes with the most connections",
        "Show me all people and their companies",
        "Find all nodes connected to Alice",
        "List all paths between nodes",
        "Show me the most connected person"
    ]
}

# ============================================
# MAIN UI LAYOUT
# ============================================

# Header
st.markdown("""
<div class="header-container">
    <h1>🧠 Neo4j Chatbot - Compatible UI</h1>
    <p>Perfectly matched to work with the standalone Neo4j server</p>
    <div class="standalone-badge">
        ✅ COMPATIBLE with standalone_server.py
    </div>
</div>
""", unsafe_allow_html=True)

# Check connection status
health_status = update_connection_status()

# Display connection status
if health_status["status"] == "healthy":
    st.markdown(f"""
    <div class="connection-indicator connected">
        ✅ Connected to Standalone Neo4j Server on port {BACKEND_PORT}
    </div>
    """, unsafe_allow_html=True)
    
    # Show server details
    if "data" in health_status:
        health_data = health_status["data"]
        if "neo4j" in health_data:
            neo4j_info = health_data["neo4j"]
            st.success(f"Neo4j Status: {neo4j_info.get('status', 'unknown')}")
        if "server" in health_data:
            server_info = health_data["server"]
            st.info(f"Server Type: {server_info.get('type', 'unknown')}")
else:
    st.markdown(f"""
    <div class="connection-indicator disconnected">
        ❌ Cannot connect to standalone server on port {BACKEND_PORT}
        <br>Error: {health_status.get('error', 'Unknown error')}
    </div>
    """, unsafe_allow_html=True)
    
    st.error("**Please check:**")
    st.error(f"1. Standalone server is running: `python standalone_server.py`")
    st.error("2. Neo4j database is accessible")
    st.error("3. Server started without errors")
    
    if st.button("🔄 Retry Connection"):
        st.cache_data.clear()
        st.rerun()

# Only show main interface if connected
if health_status["status"] == "healthy":
    
    # Create main layout
    col1, col2 = st.columns([1, 1.5])
    
    # ============================================
    # LEFT COLUMN - CHAT INTERFACE
    # ============================================
    
    with col1:
        st.markdown("## 💬 Compatible Chat Interface")
        
        # Display current database stats
        stats = get_database_stats()
        if "error" not in stats:
            st.markdown("### 📊 Live Database Statistics")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Nodes", stats.get("nodes", 0))
                st.metric("Relationships", stats.get("relationships", 0))
            with col_b:
                st.metric("Labels", len(stats.get("labels", [])))
                st.metric("Rel Types", len(stats.get("relationship_types", [])))
        
        # Manual refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            # Refresh graph data
            try:
                new_graph_data = get_graph_data(50)
                if "error" not in new_graph_data:
                    st.session_state.graph_data = new_graph_data
                    st.session_state.last_update_time = datetime.now()
                    st.success("✅ Data refreshed from standalone server!")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to refresh: {e}")
        
        # Chat messages display
        st.markdown("### 📝 Recent Messages")
        
        # Display recent messages
        for message in st.session_state.messages[-6:]:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>🧑 You:</strong> {message["content"]}
                    <br><small>⏰ {datetime.fromisoformat(message["timestamp"]).strftime("%H:%M:%S")}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                result = message["content"]
                
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 Standalone Agent:</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Display response components
                if result.get("tool"):
                    st.markdown(f"""
                    <div class="tool-badge">
                        🔧 {result["tool"]}
                    </div>
                    """, unsafe_allow_html=True)
                
                if result.get("query"):
                    st.markdown("**Query:**")
                    st.markdown(f"""
                    <div class="query-display">
                        {result["query"]}
                    </div>
                    """, unsafe_allow_html=True)
                
                if result.get("answer"):
                    st.markdown("**Result:**")
                    st.markdown(f"""
                    <div class="result-display">
                        {result["answer"]}
                    </div>
                    """, unsafe_allow_html=True)
                
                if result.get("response_time"):
                    st.caption(f"⏱️ {result['response_time']:.2f}s")
        
        # Chat input
        st.markdown("### ✍️ Ask Your Question")
        
        # Chat input
        if prompt := st.chat_input("Ask about your Neo4j database (e.g., 'How many nodes?', 'Create a Person named Alice')"):
            # Add user message
            user_message = {
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(user_message)
            
            # Process the request
            with st.spinner("🧠 Processing with standalone server..."):
                result = send_chat_message(prompt)
            
            # Add assistant message
            assistant_message = {
                "role": "assistant",
                "content": result,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(assistant_message)
            
            # Refresh graph if it was a write operation
            if result.get("tool") == "write_neo4j_cypher" and result.get("success", True):
                try:
                    new_graph_data = get_graph_data(50)
                    if "error" not in new_graph_data:
                        st.session_state.graph_data = new_graph_data
                        st.session_state.last_update_time = datetime.now()
                        st.success("🔄 Graph updated automatically!")
                except Exception as e:
                    st.warning(f"Could not refresh graph: {e}")
            
            st.rerun()
    
    # ============================================
    # RIGHT COLUMN - COMPATIBLE GRAPH VISUALIZATION
    # ============================================
    
    with col2:
        st.markdown("## 🎨 Compatible Neo4j Graph")
        
        # Load graph data if not already loaded
        if st.session_state.graph_data is None:
            with st.spinner("📊 Loading graph data from standalone server..."):
                try:
                    graph_data = get_graph_data(50)
                    if "error" not in graph_data:
                        st.session_state.graph_data = graph_data
                        st.session_state.last_update_time = datetime.now()
                except Exception as e:
                    st.error(f"Failed to load graph: {e}")
        
        # Display graph
        if st.session_state.graph_data and "error" not in st.session_state.graph_data:
            graph_data = st.session_state.graph_data
            
            # Show graph info
            nodes = graph_data.get("nodes", [])
            relationships = graph_data.get("relationships", [])
            
            st.markdown("### 📈 Graph Overview")
            col_x, col_y = st.columns(2)
            with col_x:
                st.metric("Loaded Nodes", len(nodes))
            with col_y:
                st.metric("Loaded Relationships", len(relationships))
            
            if st.session_state.last_update_time:
                st.caption(f"Last updated: {st.session_state.last_update_time.strftime('%H:%M:%S')}")
            
            # Create and display compatible graph
            if nodes:
                compatible_graph_html = create_compatible_graph_visualization(graph_data, height=600)
                
                components.html(
                    compatible_graph_html,
                    height=600,
                    key=f"compatible_graph_{int(time.time())}"
                )
            else:
                st.info("📝 No nodes in database. Use the example prompts to create some data!")
        
        elif st.session_state.graph_data and "error" in st.session_state.graph_data:
            st.error(f"Graph error: {st.session_state.graph_data['error']}")
        else:
            st.info("📊 Click 'Refresh Data' to load the graph visualization")

# ============================================
# SIDEBAR - COMPATIBLE PROMPTS
# ============================================

with st.sidebar:
    st.markdown("## 🚀 Compatible Example Prompts")
    
    # Server status
    if health_status["status"] == "healthy":
        st.markdown("### ✅ Server Status")
        st.success(f"Connected to standalone server on port {BACKEND_PORT}")
    else:
        st.markdown("### ❌ Server Status")
        st.error(f"Not connected to port {BACKEND_PORT}")
    
    # Compatible example prompts organized by category
    for category, prompts in COMPATIBLE_EXAMPLE_PROMPTS.items():
        st.markdown(f"### {category}")
        
        for prompt in prompts:
            if st.button(prompt, key=f"prompt_{hash(prompt)}", use_container_width=True):
                # Add to chat
                user_message = {
                    "role": "user",
                    "content": prompt,
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(user_message)
                
                # Process the query
                with st.spinner("🧠 Processing..."):
                    result = send_chat_message(prompt)
                
                assistant_message = {
                    "role": "assistant",
                    "content": result,
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(assistant_message)
                
                # Refresh graph if needed
                if result.get("tool") == "write_neo4j_cypher" and result.get("success", True):
                    try:
                        new_graph_data = get_graph_data(50)
                        if "error" not in new_graph_data:
                            st.session_state.graph_data = new_graph_data
                            st.session_state.last_update_time = datetime.now()
                    except Exception:
                        pass
                
                st.rerun()
    
    # Clear chat
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Session info
    st.markdown("### 📋 Session Info")
    st.text(f"ID: {st.session_state.session_id[:8]}...")
    st.text(f"Messages: {len(st.session_state.messages)}")
    if st.session_state.graph_data:
        nodes_count = len(st.session_state.graph_data.get("nodes", []))
        rels_count = len(st.session_state.graph_data.get("relationships", []))
        st.text(f"Graph: {nodes_count} nodes, {rels_count} rels")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 0.5rem; margin-top: 2rem;">
    <h3>🧠 Compatible Neo4j Chatbot v1.0</h3>
    <p><strong>✅ PERFECT COMPATIBILITY:</strong> Designed specifically for standalone_server.py</p>
    <p><strong>🎨 STABLE GRAPH:</strong> D3.js visualization with visible relationship lines</p>
    <p><strong>🔄 REAL-TIME UPDATES:</strong> Graph refreshes automatically after changes</p>
    <p><strong>💡 SMART PROMPTS:</strong> {sum(len(prompts) for prompts in COMPATIBLE_EXAMPLE_PROMPTS.values())} compatible examples</p>
    <p><strong>🌐 Backend:</strong> Standalone Neo4j Server on port {BACKEND_PORT}</p>
    <p><strong>📡 Connection:</strong> {health_status["status"].upper()}</p>
</div>
""", unsafe_allow_html=True)
