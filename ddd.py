import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import requests
import json
import tempfile
from datetime import datetime
import uuid
import time
import traceback

# Page configuration
st.set_page_config(
    page_title="Neo4j Graph Explorer Pro",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styles (skip for brevity; you can paste your CSS block here) ---

# --- Session State Initialization ---
def init_stable_session_state():
    defaults = {
        "messages": [],
        "graph_data": None,
        "last_response": None,
        "session_id": str(uuid.uuid4()),
        "connection_status": "unknown",
        "chat_history": [],
        "graph_settings": {
            "node_size": 25,
            "edge_width": 2,
            "physics_enabled": True,
            "show_labels": True
        }
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
init_stable_session_state()

# --- Helper Functions ---

def extract_display_name(node):
    props = node.get("properties", {})
    for field in ["name", "title", "fullName", "displayName", "username", "label"]:
        if field in props and props[field]:
            return str(props[field])[:40]
    labels = node.get("labels", ["Node"])
    return f"{labels[0]}_{str(node.get('id',''))[:6]}"

def get_neo4j_colors():
    return {
        "Person": "#FF6B6B",
        "Company": "#45B7D1",
        "Location": "#FECA57",
        "Project": "#FDCB6E",
        "default": "#95A5A6"
    }

def get_relationship_colors():
    return {
        "KNOWS": "#E74C3C",
        "WORKS_FOR": "#3498DB",
        "MANAGES": "#9B59B6",
        "default": "#666666"
    }

# --- MAIN GRAPH RENDER FUNCTION (FIXED) ---

def render_neo4j_graph(graph_data: dict) -> bool:
    if not graph_data:
        st.info("🔍 No graph data available for visualization.")
        return False
    try:
        nodes = graph_data.get("nodes", [])
        relationships = graph_data.get("relationships", [])
        if not nodes:
            st.info("📊 No nodes found in the current result set.")
            return False

        net = Network(
            height="700px",
            width="100%",
            bgcolor="#F8F9FA",
            font_color="#2C3E50",
        )
        # --- NODE ID MAP: Neo4j IDs to Pyvis IDs ---
        node_id_map = {}
        node_colors = get_neo4j_colors()

        for i, node in enumerate(nodes):
            node_uid = str(node.get("id"))
            display_name = extract_display_name(node)
            labels = node.get("labels", ["Unknown"])
            color = node_colors.get(labels[0], node_colors["default"])
            props = node.get("properties", {})
            tooltip = "<br>".join([
                f"<b>{display_name}</b>",
                f"Type: {labels[0]}",
                *[f"{k}: {v}" for k, v in list(props.items())[:3]]
            ])
            net.add_node(
                node_uid,
                label=display_name,
                color=color,
                size=st.session_state.graph_settings["node_size"],
                title=tooltip,
                shape='dot'
            )
            node_id_map[node_uid] = node_uid  # Neo4j id string to itself

        rel_colors = get_relationship_colors()
        for rel in relationships:
            start_id = str(rel.get("startNode", rel.get("start", "")))
            end_id = str(rel.get("endNode", rel.get("end", "")))
            rel_type = str(rel.get("type", "CONNECTED"))
            color = rel_colors.get(rel_type, rel_colors["default"])
            rel_props = rel.get("properties", {})
            rel_tooltip = f"<b>{rel_type}</b>" + "".join([f"<br>{k}: {v}" for k, v in list(rel_props.items())[:2]])
            # Only add edges if both nodes exist
            if start_id in node_id_map and end_id in node_id_map:
                net.add_edge(
                    start_id,
                    end_id,
                    label=rel_type,
                    color=color,
                    width=st.session_state.graph_settings["edge_width"],
                    title=rel_tooltip,
                    arrows='to'
                )

        # Enable/Disable physics
        if st.session_state.graph_settings["physics_enabled"]:
            net.toggle_physics(True)
        else:
            net.toggle_physics(False)

        # Save and display graph
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            components.html(open(tmp_file.name, "r", encoding="utf-8").read(), height=700, scrolling=True)
        return True

    except Exception as e:
        st.error(f"❌ Graph rendering error: {str(e)}")
        st.code(traceback.format_exc())
        return False

# --- EXAMPLE TEST DATA LOADER ---
if st.button("🧪 Load Test Graph", use_container_width=True):
    st.session_state.graph_data = {
        "nodes": [
            {"id": "p1", "labels": ["Person"], "properties": {"name": "Alice"}},
            {"id": "p2", "labels": ["Person"], "properties": {"name": "Bob"}},
            {"id": "c1", "labels": ["Company"], "properties": {"name": "TechCorp"}},
        ],
        "relationships": [
            {"startNode": "p1", "endNode": "p2", "type": "KNOWS", "properties": {"since": "2020"}},
            {"startNode": "p1", "endNode": "c1", "type": "WORKS_FOR", "properties": {"since": "2019"}},
        ]
    }
    st.success("Test data loaded! Scroll down to see the graph.")

# --- MAIN APP UI ---
st.title("🕸️ Neo4j Graph Explorer Pro")
if st.session_state.graph_data:
    st.markdown("### 🖼️ Interactive Graph View")
    render_neo4j_graph(st.session_state.graph_data)
else:
    st.info("Click '🧪 Load Test Graph' to demo, or run a query to see your Neo4j data!")

