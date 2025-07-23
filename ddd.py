import streamlit as st
from pyvis.network import Network
from neo4j import GraphDatabase
import tempfile
import os

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
    st.markdown("-----")
    st.write("Adjust the Cypher query below to explore different parts of your graph:")

cypher_default = f"MATCH (n)-[r]->(m) RETURN n, r, m LIMIT {MAX_NODES}"
cypher_query = st.text_area("Cypher Query", cypher_default, height=80)

run_viz = st.button("Show Graph")

def get_neo4j_graph(uri, user, password, database, cypher):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    nodes = {}
    edges = []
    try:
        with driver.session(database=database) as session:
            result = session.run(cypher)
            for record in result:
                n = record.get('n')
                m = record.get('m')
                r = record.get('r')
                if n and n.id not in nodes:
                    nodes[n.id] = n
                if m and m.id not in nodes:
                    nodes[m.id] = m
                if n and m and r:
                    edges.append((n.id, m.id, r.type))
    except Exception as e:
        st.error(f"Error querying Neo4j: {e}")
    finally:
        driver.close()
    return nodes, edges

def draw_pyvis_graph(nodes, edges):
    net = Network(height="800px", width="100%", notebook=False, bgcolor="#222222", font_color="white")
    for node_id, node in nodes.items():
        # Use 'name' property if present, otherwise id
        label = node.get('name', f"Node {node_id}")
        props = "\n".join([f"{k}: {v}" for k, v in dict(node).items() if k != 'name'])
        net.add_node(node_id, label=label, title=props, shape="dot")
    for src, tgt, rel_type in edges:
        net.add_edge(src, tgt, label=rel_type)
    net.repulsion(node_distance=120, spring_length=200)
    net.set_options('''
    var options = {
      "edges": {"color": {"inherit": true},"smooth": false},
      "nodes": {"shape": "dot","size": 14},
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
    with st.spinner("Querying Neo4j and building the graph..."):
        nodes, edges = get_neo4j_graph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, cypher_query)
        if nodes:
            st.success(f"Visualizing {len(nodes)} nodes and {len(edges)} relationships.")
            net = draw_pyvis_graph(nodes, edges)
            # Save and display the graph
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as tmp_file:
                net.save_graph(tmp_file.name)
                tmp_path = tmp_file.name
            st.components.v1.html(open(tmp_path, 'r').read(), height=830, scrolling=True)
            os.remove(tmp_path)
        else:
            st.warning("No data to display. Please check your Cypher query or connection settings.")
