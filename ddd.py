import streamlit as st
import requests
import uuid
import json
from datetime import datetime
import streamlit.components.v1 as components

st.set_page_config(page_title="Neo4j LLM Agent", page_icon="🧠", layout="wide")

st.title("🧠 Neo4j LangGraph MCP Agent with Graph Visualization")
st.markdown("**Enhanced with Change Tracking & Interactive Graph Display**")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "change_history" not in st.session_state:
    st.session_state.change_history = []

def create_graph_visualization(graph_data, height=400):
    """Create an interactive Neo4j graph visualization"""
    if not graph_data or not graph_data.get('nodes'):
        return None
    
    # Prepare data for visualization
    nodes = graph_data.get('nodes', [])
    relationships = graph_data.get('relationships', [])
    
    # Create unique colors for different node labels
    label_colors = {
        'Person': '#FF6B6B',
        'Company': '#4ECDC4', 
        'Project': '#45B7D1',
        'Department': '#96CEB4',
        'Location': '#FECA57',
        'Product': '#FF9FF3',
        'Order': '#54A0FF',
        'Customer': '#5F27CD'
    }
    
    # Build nodes for visualization
    vis_nodes = []
    for node in nodes:
        primary_label = node['labels'][0] if node['labels'] else 'Node'
        color = label_colors.get(primary_label, '#95A5A6')
        
        # Create display label
        display_props = node.get('properties', {})
        display_label = display_props.get('name') or display_props.get('title') or display_props.get('id') or primary_label
        
        vis_nodes.append({
            'id': node['id'],
            'label': str(display_label)[:20],  # Limit label length
            'color': color,
            'size': 20,
            'font': {'color': 'white', 'size': 12},
            'neo4j_labels': node['labels'],
            'neo4j_properties': display_props
        })
    
    # Build edges for visualization  
    vis_edges = []
    for rel in relationships:
        vis_edges.append({
            'id': rel['id'],
            'from': rel['startNode'],
            'to': rel['endNode'],
            'label': rel['type'],
            'arrows': 'to',
            'color': {'color': '#7f8c8d'},
            'font': {'color': '#2c3e50', 'size': 10},
            'neo4j_type': rel['type'],
            'neo4j_properties': rel.get('properties', {})
        })
    
    # Create the HTML with vis.js
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            #neo4j-graph {{
                width: 100%;
                height: {height}px;
                border: 1px solid #ddd;
                border-radius: 8px;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            }}
            .graph-info {{
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(255,255,255,0.9);
                padding: 8px;
                border-radius: 4px;
                font-size: 12px;
                z-index: 1000;
            }}
        </style>
    </head>
    <body>
        <div class="graph-info">
            📊 Nodes: {len(vis_nodes)} | 🔗 Relationships: {len(vis_edges)}
        </div>
        <div id="neo4j-graph"></div>
        
        <script type="text/javascript">
            // Graph data
            var nodes = new vis.DataSet({json.dumps(vis_nodes)});
            var edges = new vis.DataSet({json.dumps(vis_edges)});
            
            // Create network
            var container = document.getElementById('neo4j-graph');
            var data = {{
                nodes: nodes,
                edges: edges
            }};
            
            var options = {{
                nodes: {{
                    shape: 'dot',
                    size: 20,
                    font: {{
                        size: 12,
                        color: 'white'
                    }},
                    borderWidth: 2,
                    shadow: true
                }},
                edges: {{
                    width: 2,
                    shadow: true,
                    smooth: {{
                        type: 'continuous'
                    }}
                }},
                physics: {{
                    stabilization: false,
                    barnesHut: {{
                        gravitationalConstant: -2000,
                        springConstant: 0.001,
                        springLength: 200
                    }}
                }},
                interaction: {{
                    hover: true,
                    tooltipDelay: 200
                }}
            }};
            
            var network = new vis.Network(container, data, options);
            
            // Add click event for node details
            network.on("click", function (params) {{
                if (params.nodes.length > 0) {{
                    var nodeId = params.nodes[0];
                    var node = nodes.get(nodeId);
                    var details = 'Node: ' + node.label + '\\n' +
                                'Labels: ' + node.neo4j_labels.join(', ') + '\\n' +
                                'Properties: ' + JSON.stringify(node.neo4j_properties, null, 2);
                    alert(details);
                }}
                if (params.edges.length > 0) {{
                    var edgeId = params.edges[0];
                    var edge = edges.get(edgeId);
                    var details = 'Relationship: ' + edge.neo4j_type + '\\n' +
                                'Properties: ' + JSON.stringify(edge.neo4j_properties, null, 2);
                    alert(details);
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    return html_content

# Sidebar for recent changes and graph controls
with st.sidebar:
    st.header("🔄 Recent Changes")
    if st.session_state.change_history:
        for change in reversed(st.session_state.change_history[-5:]):
            with st.expander(f"⏰ {change['timestamp'][:19]}"):
                st.write(f"**Tool:** {change['tool']}")
                st.write(f"**Query:** `{change['query'][:50]}...`")
                st.write(f"**Changes:** {change['summary']}")
    else:
        st.write("No changes yet")
    
    if st.button("Clear History"):
        st.session_state.change_history = []
        st.rerun()
    
    st.markdown("---")
    st.header("📊 Graph Visualization")
    
    # Sample graph button
    if st.button("🎯 Show Sample Graph", use_container_width=True):
        try:
            sample_result = requests.get("http://localhost:8000/sample_graph", timeout=10)
            if sample_result.status_code == 200:
                sample_data = sample_result.json()
                if sample_data.get('graph_data'):
                    st.session_state['show_sample_graph'] = sample_data['graph_data']
                    st.success("Sample graph loaded!")
                    st.rerun()
                else:
                    st.warning("No graph data found for sample")
            else:
                st.error("Failed to load sample graph")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Main interface
col1, col2 = st.columns([3, 1])

with col1:
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_input(
            "Ask a question:", 
            key="chat_input", 
            placeholder="e.g. Show me all Person nodes or Create a relationship between Alice and Bob"
        )
        submitted = st.form_submit_button("Send", use_container_width=True)

with col2:
    st.markdown("**Quick Actions:**")
    if st.button("📊 Show Schema", use_container_width=True):
        user_query = "Show me the database schema"
        submitted = True
    if st.button("🔢 Count Nodes", use_container_width=True):
        user_query = "How many nodes are in the graph?"
        submitted = True
    if st.button("🔗 Show Relationships", use_container_width=True):
        user_query = "MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 20"
        submitted = True
    if st.button("👥 Show All People", use_container_width=True):
        user_query = "MATCH (p:Person) RETURN p LIMIT 10"
        submitted = True

def display_formatted_answer_with_graph(answer, tool, query, graph_data=None):
    """Display the answer with appropriate formatting and graph visualization"""
    
    # Check if this was a write operation with changes
    if tool == "write_neo4j_cypher" and "Neo4j Write Operation Completed" in answer:
        st.success("✅ Write Operation Successful!")
        st.markdown(answer)
        
        # Try to extract change information for history
        try:
            lines = answer.split('\n')
            summary_line = next((line for line in lines if '🕐' in line), "")
            if summary_line:
                change_record = {
                    'timestamp': datetime.now().isoformat(),
                    'tool': tool,
                    'query': query,
                    'summary': summary_line.replace('🕐', '').strip()
                }
                st.session_state.change_history.append(change_record)
        except Exception:
            pass
    
    elif tool == "read_neo4j_cypher" and "Query Results" in answer:
        st.info("📊 Query Results")
        st.markdown(answer)
    
    elif tool == "get_neo4j_schema" and "Schema Information" in answer:
        st.info("🏗️ Schema Information")
        st.markdown(answer)
    
    else:
        # Default display
        if "❌" in answer or "⚠️" in answer:
            st.error(answer)
        elif "✅" in answer:
            st.success(answer)
        else:
            st.markdown(answer)
    
    # Display graph visualization if available
    if graph_data and graph_data.get('nodes'):
        st.markdown("---")
        st.subheader("🕸️ Interactive Graph Visualization")
        
        # Create tabs for different views
        viz_tab, data_tab = st.tabs(["🎨 Graph View", "📋 Raw Data"])
        
        with viz_tab:
            graph_html = create_graph_visualization(graph_data, height=500)
            if graph_html:
                components.html(graph_html, height=520)
            else:
                st.warning("Could not create graph visualization")
        
        with data_tab:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔵 Nodes:**")
                for node in graph_data.get('nodes', [])[:10]:  # Show first 10
                    labels = ', '.join(node.get('labels', []))
                    props = node.get('properties', {})
                    display_name = props.get('name') or props.get('title') or 'Unnamed'
                    st.write(f"• **{display_name}** ({labels})")
                    if len(props) > 0:
                        st.json(dict(list(props.items())[:3]), expanded=False)  # Show first 3 properties
            
            with col2:
                st.markdown("**🔗 Relationships:**")
                for rel in graph_data.get('relationships', [])[:10]:  # Show first 10
                    rel_type = rel.get('type', 'Unknown')
                    props = rel.get('properties', {})
                    st.write(f"• **{rel_type}**")
                    if len(props) > 0:
                        st.json(dict(list(props.items())[:3]), expanded=False)

# Handle sample graph display
if 'show_sample_graph' in st.session_state:
    st.markdown("---")
    st.subheader("🎯 Sample Graph")
    display_formatted_answer_with_graph("Sample graph data", "read_neo4j_cypher", "Sample query", st.session_state['show_sample_graph'])
    if st.button("Clear Sample Graph"):
        del st.session_state['show_sample_graph']
        st.rerun()

if submitted and user_query:
    session_id = str(uuid.uuid4())
    payload = {
        "question": user_query,
        "session_id": session_id
    }

    # Show loading spinner
    with st.spinner(f"🤔 Processing: {user_query}"):
        try:
            response = requests.post("http://localhost:8081/chat", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                st.markdown("---")
                
                # Show user query
                with st.chat_message("user"):
                    st.write(user_query)
                
                # Show agent response
                with st.chat_message("assistant"):
                    # Show trace (LLM reasoning) in an expander
                    with st.expander("🧠 Agent Reasoning"):
                        st.code(result['trace'], language="text")
                    
                    # Show tool and query used
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"🔧 **Tool:** {result['tool']}")
                    with col2:
                        if result['query']:
                            st.code(result['query'], language="cypher")
                    
                    # Parse graph data from response if available
                    graph_data = None
                    if 'graph_data' in result.get('answer', ''):
                        try:
                            # Try to extract graph data from the response
                            import re
                            graph_match = re.search(r'"graph_data":\s*(\{.*?\})', result['answer'])
                            if graph_match:
                                graph_data = json.loads(graph_match.group(1))
                        except:
                            pass
                    
                    # Display the formatted answer with graph visualization
                    display_formatted_answer_with_graph(result['answer'], result['tool'], result['query'], graph_data)
                
                # Store in session state for history
                st.session_state.messages.append({
                    "user": user_query,
                    "bot": result['answer'],
                    "tool": result['tool'],
                    "query": result['query'],
                    "trace": result['trace'],
                    "graph_data": graph_data,
                    "timestamp": datetime.now().isoformat()
                })
                
            else:
                st.error(f"❌ Server Error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The server might be busy.")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to the server. Make sure the FastAPI server is running on port 8081.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

# Display conversation history
if st.session_state.messages:
    st.markdown("---")
    st.header("💬 Conversation History")
    
    for i, msg in enumerate(reversed(st.session_state.messages)):
        with st.expander(f"🕐 {msg.get('timestamp', '')[:19]} - {msg['user'][:50]}..."):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown("**Query:**")
                st.write(msg['user'])
                st.markdown("**Tool Used:**")
                st.code(msg['tool'])
                if msg['query']:
                    st.markdown("**Cypher Query:**")
                    st.code(msg['query'], language="cypher")
            
            with col2:
                st.markdown("**Response:**")
                st.markdown(msg['bot'])
                
                # Show graph data if available
                if msg.get('graph_data'):
                    st.markdown("**Graph Data:**")
                    graph_html = create_graph_visualization(msg['graph_data'], height=300)
                    if graph_html:
                        components.html(graph_html, height=320)

# Footer with connection status
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    try:
        health_check = requests.get("http://localhost:8081/", timeout=2)
        if health_check.status_code == 200:
            st.success("🟢 FastAPI Server Connected")
        else:
            st.error("🔴 FastAPI Server Issues")
    except:
        st.error("🔴 FastAPI Server Disconnected")

with col2:
    try:
        mcp_check = requests.post("http://localhost:8000/get_neo4j_schema", timeout=2)
        if mcp_check.status_code == 200:
            st.success("🟢 Neo4j Connected")
        else:
            st.error("🔴 Neo4j Issues")
    except:
        st.error("🔴 Neo4j Disconnected")

with col3:
    st.info(f"💬 Messages: {len(st.session_state.messages)} | 🔄 Changes: {len(st.session_state.change_history)}")

# Add some CSS for better styling
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 16px;
}
.stExpander {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)
