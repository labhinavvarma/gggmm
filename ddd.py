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
    # Generates a color hex from a string (label/type) for consistency
    random.seed(hash(s) % (2**32))
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def get_neo4j_graph(uri, user, password, database, cypher):
    driver = None
    nodes = {}
    edges = []
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=database) as session:
            result = session.run(cypher)
            for record in result:
                # For every field in the record, check if it's a node or a relationship
                for field in record.keys():
                    val = record.get(field)
                    # Node: has .id and .items (Neo4j Node)
                    if hasattr(val, "id") and hasattr(val, "items"):
                        node_dict = dict(val.items())
                        node_dict['labels'] = list(val.labels) if hasattr(val, 'labels') else []
                        nodes[val.id] = node_dict
                    # Relationship: has .type, .start_node_id, .end_node_id, .items
                    elif hasattr(val, "type") and hasattr(val, "start_node_id") and hasattr(val, "end_node_id"):
                        rel_props = dict(val.items())
                        edges.append((val.start_node_id, val.end_node_id, val.type, rel_props))
    except Exception as e:
        st.error(f"Error querying Neo4j: {e}")
    finally:
        if driver:
            driver.close()
    return nodes, edges

def draw_pyvis_graph(nodes, edges):
    net = Network(height="800px", width="100%", notebook=False, bgcolor="#222222", font_color="white")

    label_color_map = {}
    reltype_color_map = {}

    # Assign a color per label and reltype
    for node in nodes.values():
        labels = node.get('labels', []) if isinstance(node.get('labels', []), list) else []
        for label in labels:
            if label not in label_color_map:
                label_color_map[label] = color_from_string(label)
    for _, _, rel_type, *_ in edges:
        if rel_type not in reltype_color_map:
            reltype_color_map[rel_type] = color_from_string(rel_type)

    # Add nodes with color and tooltips
    for node_id, node in nodes.items():
        labels = node.get('labels', [])
        label = labels[0] if labels else node.get('name', f"Node {node_id}")
        props = "<br>".join([f"<b>{k}</b>: {v}" for k, v in node.items() if k != 'labels'])
        # If multiple labels, show all as tooltip
        label_title = ", ".join(labels) if labels else label
        color = label_color_map.get(labels[0], "#CCCCCC") if labels else "#CCCCCC"
        net.add_node(node_id, label=label, title=label_title + "<br>" + props, shape="dot", color=color)

    # Add edges with color and tooltips for properties
    for src, tgt, rel_type, rel_props in edges:
        title = "<br>".join([f"<b>{k}</b>: {v}" for k, v in rel_props.items()]) if rel_props else rel_type
        color = reltype_color_map.get(rel_type, "#888888")
        net.add_edge(src, tgt, label=rel_type, title=title, color=color)

    net.repulsion(node_distance=120, spring_length=200)
    net.set_options('''
    var options = {
      "edges": {"color": {"inherit": false}, "smooth": false},
      "nodes": {"shape": "dot", "size": 14},
      "physics": {"repulsion": {
        "centralGravity": 0.11,
        "springLength": 200,
        "springConstant": 0.04,
        "nodeDistance": 140,
        "damping": 0.15
        },
        "minVelocity": 0.75
      }
    }
    ''')
    return net

if run_viz:
    # --- Basic validation ---
    if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD and NEO4J_DATABASE):
        st.warning("Please fill all connection fields in the sidebar.")
    else:
        with st.spinner("Querying Neo4j and building the graph..."):
            nodes, edges = get_neo4j_graph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, cypher_query)
            if nodes:
                st.success(f"Visualizing {len(nodes)} nodes and {len(edges)} relationships.")
                net = draw_pyvis_graph(nodes, edges)
                with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as tmp_file:
                    net.save_graph(tmp_file.name)
                    tmp_path = tmp_file.name
                with open(tmp_path, 'r', encoding="utf-8") as f:
                    html = f.read()
                st.components.v1.html(html, height=830, scrolling=True)
                os.remove(tmp_path)
            else:
                st.warning("No data to display. Please check your Cypher query or connection settings.")

