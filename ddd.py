import streamlit as st
from pyvis.network import Network
from neo4j import GraphDatabase
import tempfile
import os
import random

st.set_page_config(layout="wide")
st.title("🔎 Neo4j Graph Visualization (Streamlit + Pyvis)")

# --- Sidebar for login details and settings ---
with st.sidebar:
    st.header("Neo4j Connection")
    NEO4J_URI = st.text_input("Bolt URI", value="neo4j://localhost:7687")
    NEO4J_USER = st.text_input("Username", value="neo4j")
    NEO4J_PASSWORD = st.text_input("Password", type="password")
    NEO4J_DATABASE = st.text_input("Database", value="neo4j")
    MAX_NODES = st.slider("Max nodes to visualize", 100, 5000, 500, step=100)
    st.markdown("---")
    st.write("Adjust the Cypher query below to explore different parts of your graph:")

cypher_default = f"MATCH (n)-[r]->(m) RETURN n, r, m LIMIT {MAX_NODES}"
cypher_query = st.text_area("Cypher Query", cypher_default, height=80, key=f"cypher_{MAX_NODES}")

run_viz = st.button("Show Graph")

def color_from_string(s):
    # Generates a darker color hex from a string (label/type) for white background
    random.seed(hash(s) % (2**32))
    # Generate darker colors for better visibility on white background
    r = random.randint(0, 150)
    g = random.randint(0, 150) 
    b = random.randint(0, 150)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def get_neo4j_graph(uri, user, password, database, cypher):
    driver = None
    nodes = {}
    edges = []
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=database) as session:
            result = session.run(cypher)
            for record in result:
                # Process each field in the record
                for field_name in record.keys():
                    value = record[field_name]
                    
                    # Check if it's a Neo4j Node
                    if hasattr(value, 'id') and hasattr(value, 'labels'):
                        node_id = value.id
                        node_properties = dict(value)
                        node_properties['labels'] = list(value.labels)
                        nodes[node_id] = node_properties
                    
                    # Check if it's a Neo4j Relationship
                    elif hasattr(value, 'type') and hasattr(value, 'start_node') and hasattr(value, 'end_node'):
                        start_node_id = value.start_node.id
                        end_node_id = value.end_node.id
                        rel_type = value.type
                        rel_properties = dict(value)
                        
                        # Ensure start and end nodes are in our nodes dict
                        if start_node_id not in nodes:
                            start_node_props = dict(value.start_node)
                            start_node_props['labels'] = list(value.start_node.labels)
                            nodes[start_node_id] = start_node_props
                            
                        if end_node_id not in nodes:
                            end_node_props = dict(value.end_node)
                            end_node_props['labels'] = list(value.end_node.labels)
                            nodes[end_node_id] = end_node_props
                        
                        edges.append((start_node_id, end_node_id, rel_type, rel_properties))
                        
    except Exception as e:
        st.error(f"Error querying Neo4j: {e}")
        st.error("Make sure your Neo4j database is running and connection details are correct.")
    finally:
        if driver:
            driver.close()
    
    return nodes, edges

def draw_pyvis_graph(nodes, edges):
    # White background with dark text
    net = Network(height="800px", width="100%", notebook=False, bgcolor="#FFFFFF", font_color="#000000")

    label_color_map = {}
    reltype_color_map = {}

    # Assign a color per label and relationship type
    for node in nodes.values():
        labels = node.get('labels', [])
        for label in labels:
            if label not in label_color_map:
                label_color_map[label] = color_from_string(label)
                
    for _, _, rel_type, _ in edges:
        if rel_type not in reltype_color_map:
            reltype_color_map[rel_type] = color_from_string(rel_type)

    # Display color legend
    st.sidebar.markdown("### Node Colors by Label")
    for label, color in label_color_map.items():
        st.sidebar.markdown(f'<span style="color: {color}">●</span> {label}', unsafe_allow_html=True)
    
    if reltype_color_map:
        st.sidebar.markdown("### Relationship Colors")
        for rel_type, color in reltype_color_map.items():
            st.sidebar.markdown(f'<span style="color: {color}">●</span> {rel_type}', unsafe_allow_html=True)

    # Add nodes with proper styling for white background
    for node_id, node_data in nodes.items():
        labels = node_data.get('labels', [])
        
        # Create display label - use name, title, or first available property
        display_label = node_data.get('name') or node_data.get('title') or \
                       node_data.get('id') or f"Node {node_id}"
        
        # Create tooltip with all properties
        tooltip_parts = []
        if labels:
            tooltip_parts.append(f"<b>Labels:</b> {', '.join(labels)}")
        
        for key, value in node_data.items():
            if key != 'labels' and value is not None:
                tooltip_parts.append(f"<b>{key}:</b> {value}")
        
        tooltip = "<br>".join(tooltip_parts)
        
        # Choose color based on primary label
        primary_label = labels[0] if labels else 'Unknown'
        node_color = label_color_map.get(primary_label, "#666666")
        
        net.add_node(
            node_id, 
            label=str(display_label),
            title=tooltip,
            color=node_color,
            size=20,
            font={'color': '#000000', 'size': 14}  # Dark font for white background
        )

    # Add edges with proper styling
    for start_id, end_id, rel_type, rel_props in edges:
        # Create edge tooltip
        tooltip_parts = [f"<b>Type:</b> {rel_type}"]
        for key, value in rel_props.items():
            if value is not None:
                tooltip_parts.append(f"<b>{key}:</b> {value}")
        
        edge_tooltip = "<br>".join(tooltip_parts)
        edge_color = reltype_color_map.get(rel_type, "#666666")
        
        net.add_edge(
            start_id, 
            end_id, 
            label=rel_type,
            title=edge_tooltip,
            color=edge_color,
            width=2,
            font={'color': '#000000', 'size': 12}  # Dark font for labels
        )

    # Configure physics for better layout
    net.repulsion(
        node_distance=150,
        central_gravity=0.1,
        spring_length=200,
        spring_strength=0.05,
        damping=0.1
    )
    
    # Set options for white background
    net.set_options('''
    var options = {
      "edges": {
        "color": {"inherit": false},
        "smooth": {"type": "continuous"},
        "arrows": {"to": {"enabled": true, "scaleFactor": 1}},
        "font": {"color": "#000000", "size": 12}
      },
      "nodes": {
        "shape": "dot",
        "size": 20,
        "font": {"color": "#000000", "size": 14},
        "borderWidth": 2,
        "borderWidthSelected": 3
      },
      "physics": {
        "repulsion": {
          "centralGravity": 0.1,
          "springLength": 200,
          "springConstant": 0.05,
          "nodeDistance": 150,
          "damping": 0.1
        },
        "minVelocity": 0.75,
        "solver": "repulsion"
      },
      "interaction": {
        "hover": true,
        "selectConnectedEdges": true,
        "tooltipDelay": 200
      }
    }
    ''')
    
    return net

if run_viz:
    # Validation
    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE]):
        st.warning("⚠️ Please fill all connection fields in the sidebar.")
    else:
        with st.spinner("Querying Neo4j and building the graph..."):
            nodes, edges = get_neo4j_graph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, cypher_query)
            
            if nodes:
                st.success(f"✅ Found {len(nodes)} nodes and {len(edges)} relationships.")
                
                # Display some statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nodes", len(nodes))
                with col2:
                    st.metric("Relationships", len(edges))
                with col3:
                    node_types = set()
                    for node in nodes.values():
                        node_types.update(node.get('labels', []))
                    st.metric("Node Types", len(node_types))
                
                # Generate and display the graph
                net = draw_pyvis_graph(nodes, edges)
                
                # Save and display the graph
                with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='utf-8') as tmp_file:
                    net.save_graph(tmp_file.name)
                    tmp_path = tmp_file.name
                
                try:
                    with open(tmp_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=850, scrolling=False)
                finally:
                    os.remove(tmp_path)
                    
                # Show sample data for debugging
                with st.expander("🔍 Sample Data (for debugging)"):
                    if nodes:
                        st.write("**Sample Node:**")
                        sample_node = next(iter(nodes.values()))
                        st.json(sample_node)
                    
                    if edges:
                        st.write("**Sample Relationship:**")
                        sample_edge = edges[0]
                        st.write(f"Start: {sample_edge[0]}, End: {sample_edge[1]}, Type: {sample_edge[2]}")
                        if sample_edge[3]:
                            st.json(sample_edge[3])
            else:
                st.warning("⚠️ No data found. Please check:")
                st.write("1. Your Cypher query returns nodes and relationships")
                st.write("2. Your Neo4j connection details are correct")
                st.write("3. Your database contains data matching the query")
                
                st.info("💡 Try a simpler query like: `MATCH (n) RETURN n LIMIT 10`")
