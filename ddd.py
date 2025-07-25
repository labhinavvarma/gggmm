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
    page_title="Neo4j Graph Explorer - Fixed UI", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
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
    
    .debug-panel {
        background: #f8f9fa;
        border: 1px solid #008cc1;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .error-panel {
        background: #ffebee;
        border: 1px solid #f44336;
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    defaults = {
        "conversation_history": [],
        "graph_data": None,
        "last_response": None,
        "session_id": str(uuid.uuid4()),
        "debug_mode": False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# Header
st.markdown('''
<div class="neo4j-header">
    <h1>🗄️ Neo4j Graph Explorer - Fixed UI</h1>
    <p><strong>Enhanced Debugging</strong> • <strong>Better Error Handling</strong> • <strong>Graph Visualization Fix</strong></p>
</div>
''', unsafe_allow_html=True)

def safe_extract_node_name(node):
    """Safely extract display name from node"""
    try:
        props = node.get("properties", {})
        labels = node.get("labels", ["Unknown"])
        node_id = str(node.get("id", ""))
        
        # Try different name properties
        for name_prop in ["name", "title", "displayName", "username", "fullName"]:
            if name_prop in props and props[name_prop]:
                return str(props[name_prop]).strip()[:30]
        
        # Fallback to label + ID
        if labels and labels[0] != "Unknown":
            short_id = node_id.split(":")[-1][-4:] if ":" in node_id else node_id[-4:]
            return f"{labels[0]}_{short_id}"
        
        return f"Node_{node_id[-6:] if len(node_id) > 6 else node_id}"
        
    except Exception as e:
        return f"Node_{hash(str(node)) % 10000}"

def create_enhanced_graph(graph_data: dict) -> bool:
    """Create enhanced graph visualization with better error handling"""
    
    if not graph_data:
        st.info("🔍 No graph data provided")
        return False
    
    try:
        nodes = graph_data.get("nodes", [])
        relationships = graph_data.get("relationships", [])
        
        st.write(f"**Debug:** Received {len(nodes)} nodes, {len(relationships)} relationships")
        
        if not nodes:
            st.warning("📊 No nodes found in graph data")
            if st.session_state.debug_mode:
                st.json(graph_data)
            return False
        
        # Create network with enhanced settings
        net = Network(
            height="700px",
            width="100%", 
            bgcolor="#FFFFFF",
            font_color="#2C3E50",
            directed=True
        )
        
        # Process nodes with better error handling
        added_nodes = set()
        node_colors = {
            "EDA": "#DA7194",      # Pink
            "Person": "#4C8EDA",   # Blue  
            "User": "#4C8EDA",     # Blue
            "Company": "#00B894",  # Teal
            "Department": "#FDCB6E", # Yellow
            "Group": "#A29BFE",    # Purple
            "Default": "#95A5A6"   # Gray
        }
        
        for i, node in enumerate(nodes):
            try:
                node_id = f"node_{i}"
                
                # Extract display information safely
                display_name = safe_extract_node_name(node)
                labels = node.get("labels", ["Unknown"])
                primary_label = labels[0] if labels else "Unknown"
                
                # Get color
                color = node_colors.get(primary_label, node_colors["Default"])
                
                # Create tooltip
                props = node.get("properties", {})
                tooltip_parts = [f"Type: {primary_label}", f"Name: {display_name}"]
                
                # Add properties to tooltip
                for key, value in list(props.items())[:3]:
                    if key not in ['name', 'title', 'displayName']:
                        tooltip_parts.append(f"{key}: {str(value)[:30]}")
                
                tooltip = "\\n".join(tooltip_parts)
                
                # Add node
                net.add_node(
                    node_id,
                    label=display_name,
                    color=color,
                    size=25,
                    title=tooltip,
                    font={'size': 14, 'color': '#2C3E50'}
                )
                
                added_nodes.add((str(node.get("id", i)), node_id))
                
            except Exception as e:
                st.warning(f"⚠️ Error processing node {i}: {str(e)}")
                continue
        
        st.write(f"**Debug:** Successfully added {len(added_nodes)} nodes")
        
        # Process relationships
        id_mapping = dict(added_nodes)
        added_edges = 0
        
        for i, rel in enumerate(relationships):
            try:
                start_raw = str(rel.get("startNode", ""))
                end_raw = str(rel.get("endNode", ""))
                rel_type = str(rel.get("type", "CONNECTED"))
                
                start_id = id_mapping.get(start_raw)
                end_id = id_mapping.get(end_raw)
                
                if start_id and end_id:
                    net.add_edge(
                        start_id,
                        end_id,
                        label=rel_type,
                        color="#95A5A6",
                        width=2,
                        title=f"Type: {rel_type}"
                    )
                    added_edges += 1
                else:
                    if st.session_state.debug_mode:
                        st.write(f"Debug: Skipped relationship - start:{start_raw}→{start_id}, end:{end_raw}→{end_id}")
                    
            except Exception as e:
                st.warning(f"⚠️ Error processing relationship {i}: {str(e)}")
                continue
        
        st.write(f"**Debug:** Successfully added {added_edges} relationships")
        
        if added_edges == 0 and len(nodes) > 1:
            st.warning("⚠️ No relationships were added - check relationship data format")
        
        # Configure physics
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100}
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 200
          }
        }
        """)
        
        # Save and display
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            net.save_graph(f.name)
            html_file = f.name
        
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Display with better error handling
        try:
            components.html(html_content, height=750, scrolling=False)
            st.success(f"✅ Graph displayed: {len(added_nodes)} nodes, {added_edges} relationships")
        except Exception as e:
            st.error(f"❌ Failed to display graph: {str(e)}")
            return False
        finally:
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
    """Enhanced API call with better error handling"""
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

def test_direct_query(query: str) -> dict:
    """Test direct MCP server query"""
    try:
        api_url = "http://localhost:8000/read_neo4j_cypher"
        
        payload = {
            "query": query,
            "params": {},
            "node_limit": 50
        }
        
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        st.error(f"❌ Direct query failed: {str(e)}")
        return None

# Main layout
col1, col2 = st.columns([1, 3], gap="large")

with col1:
    st.markdown("### 🔧 Enhanced Controls")
    
    # Debug mode toggle
    st.session_state.debug_mode = st.checkbox("🐛 Debug Mode", value=st.session_state.debug_mode)
    
    # Quick tests
    st.markdown("#### 🧪 Quick Tests")
    
    if st.button("🔍 Test Database Connection", use_container_width=True):
        result = test_direct_query("MATCH (n) RETURN count(n) as total")
        if result:
            total = result['data'][0]['total'] if result['data'] else 0
            if total > 0:
                st.success(f"✅ Database connected: {total} nodes")
            else:
                st.warning("⚠️ Database is empty!")
                if st.button("🚀 Create Sample Data"):
                    # Create sample data
                    sample_queries = [
                        "CREATE (eda:EDA {name: 'EDA Team', type: 'department'})",
                        "CREATE (john:Person {name: 'John Doe', role: 'analyst'})",
                        "CREATE (jane:Person {name: 'Jane Smith', role: 'manager'})",
                        "MATCH (eda:EDA), (john:Person) CREATE (john)-[:WORKS_IN]->(eda)",
                        "MATCH (eda:EDA), (jane:Person) CREATE (jane)-[:MANAGES]->(eda)"
                    ]
                    
                    for query in sample_queries:
                        requests.post("http://localhost:8000/write_neo4j_cypher", 
                                    json={"query": query, "params": {}})
                    
                    st.success("✅ Sample data created!")
                    st.rerun()
    
    if st.button("🕸️ Test Graph Extraction", use_container_width=True):
        result = test_direct_query("MATCH (n) RETURN n LIMIT 5")
        if result:
            has_graph = result.get('graph_data') is not None
            if has_graph:
                nodes = len(result['graph_data'].get('nodes', []))
                rels = len(result['graph_data'].get('relationships', []))
                st.success(f"✅ Extraction works: {nodes} nodes, {rels} rels")
            else:
                st.error("❌ No graph data extracted")
    
    st.divider()
    
    # Direct query test
    st.markdown("#### 🔍 Direct Query Test")
    direct_query = st.text_input("Cypher Query:", value="MATCH (n) RETURN n LIMIT 5")
    
    if st.button("▶️ Execute Direct", use_container_width=True):
        result = test_direct_query(direct_query)
        if result:
            st.json(result)
            if result.get('graph_data'):
                st.session_state.graph_data = result['graph_data']
                st.rerun()
    
    st.divider()
    
    # Agent questions
    st.markdown("#### 🤖 Agent Questions")
    
    test_questions = [
        "How many nodes are in the database?",
        "Show me some data",
        "Show me EDA group with relationships",
        "Display all nodes",
        "Find connections"
    ]
    
    for question in test_questions:
        if st.button(f"❓ {question}", use_container_width=True):
            result = call_agent_api(question)
            if result:
                st.session_state.last_response = result
                if result.get("graph_data"):
                    st.session_state.graph_data = result["graph_data"]
                st.rerun()
    
    st.divider()
    
    # Custom question
    st.markdown("#### ✍️ Custom Question")
    user_question = st.text_area("Ask anything:", height=100)
    
    if st.button("🚀 Ask Agent", use_container_width=True):
        if user_question.strip():
            result = call_agent_api(user_question.strip())
            if result:
                st.session_state.last_response = result
                if result.get("graph_data"):
                    st.session_state.graph_data = result["graph_data"]
                st.rerun()

with col2:
    st.markdown("### 🕸️ Graph Visualization")
    
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
        
        if query and st.session_state.debug_mode:
            st.markdown(f'''
            <div class="debug-panel">
                <h4>🔍 Generated Query</h4>
                <code>{query}</code>
            </div>
            ''', unsafe_allow_html=True)
    
    # Render graph
    if st.session_state.graph_data:
        if st.session_state.debug_mode:
            st.markdown("#### 🐛 Debug: Raw Graph Data")
            st.json(st.session_state.graph_data)
        
        success = create_enhanced_graph(st.session_state.graph_data)
        
        if not success:
            st.markdown('''
            <div class="error-panel">
                <h4>❌ Graph Rendering Failed</h4>
                <p>Check the debug information above for details.</p>
            </div>
            ''', unsafe_allow_html=True)
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 4rem; background: linear-gradient(135deg, #008cc1 0%, #0056d6 100%); color: white; border-radius: 15px; margin: 2rem 0;">
            <h2>🔧 Fixed Graph Explorer</h2>
            <p><strong>Enhanced Debugging • Better Error Handling • Improved Visualization</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **🚀 Getting Started:**
        1. Click "🔍 Test Database Connection" to check if you have data
        2. If empty, click "🚀 Create Sample Data" 
        3. Try "🕸️ Test Graph Extraction" to verify extraction works
        4. Ask questions like "Show me EDA group with relationships"
        
        **🐛 Debugging:**
        - Enable "Debug Mode" to see detailed information
        - Use "Direct Query Test" to test raw Cypher queries
        - Check the debug panels for error details
        """)

# Enhanced footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <strong>🔧 Enhanced Neo4j Graph Explorer</strong><br>
    Better Error Handling • Enhanced Debugging • Improved Visualization
</div>
""", unsafe_allow_html=True)
