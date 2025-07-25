def render_neo4j_graph(graph_data: dict) -> bool:
    """Render graph with Neo4j-like appearance and unlimited nodes (FIXED for ID mapping)."""
    import streamlit as st
    from pyvis.network import Network
    import tempfile
    import traceback

    if not graph_data:
        st.info("🔍 No graph data available for visualization.")
        return False

    try:
        nodes = graph_data.get("nodes", [])
        relationships = graph_data.get("relationships", [])
        if not nodes:
            st.info("📊 No nodes found in the current result set.")
            return False

        # Create Pyvis network
        net = Network(
            height="700px",
            width="100%",
            bgcolor="#F8F9FA",
            font_color="#2C3E50",
        )

        # --- Always use Neo4j node IDs! ---
        added_nodes = set()
        node_colors = get_neo4j_colors()

        for node in nodes:
            node_id = str(node.get("id"))  # Use Neo4j's real node ID!
            display_name = extract_display_name(node)
            labels = node.get("labels", ["Unknown"])
            main_label = labels[0] if labels else "Unknown"
            color = node_colors.get(main_label, node_colors["default"])
            props = node.get("properties", {})
            tooltip_parts = [f"<b>{display_name}</b>", f"Type: {main_label}"]
            important_props = ["id", "name", "title", "description", "email", "age", "role"]
            for prop in important_props:
                if prop in props and props[prop]:
                    tooltip_parts.append(f"{prop}: {props[prop]}")
            tooltip = "<br>".join(tooltip_parts)
            net.add_node(
                node_id,
                label=display_name,
                color=color,
                size=st.session_state.graph_settings["node_size"],
                title=tooltip,
                shape='dot'
            )
            added_nodes.add(node_id)

        rel_colors = get_relationship_colors()
        added_edges = 0

        for rel in relationships:
            start_id = str(rel.get("startNode", rel.get("start", "")))
            end_id = str(rel.get("endNode", rel.get("end", "")))
            rel_type = str(rel.get("type", "CONNECTED"))
            color = rel_colors.get(rel_type, rel_colors["default"])
            rel_props = rel.get("properties", {})
            rel_tooltip = f"<b>{rel_type}</b>"
            if rel_props:
                for key, value in list(rel_props.items())[:3]:
                    rel_tooltip += f"<br>{key}: {value}"
            # Only add edges if both nodes exist
            if start_id in added_nodes and end_id in added_nodes:
                net.add_edge(
                    start_id,
                    end_id,
                    label=rel_type,
                    color=color,
                    width=st.session_state.graph_settings["edge_width"],
                    title=rel_tooltip,
                    arrows='to'
                )
                added_edges += 1

        # Enable/Disable physics
        if st.session_state.graph_settings["physics_enabled"]:
            net.toggle_physics(True)
        else:
            net.toggle_physics(False)

        # Save and display the graph
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            html_content = open(tmp_file.name, "r", encoding="utf-8").read()
            st.components.v1.html(html_content, height=700, scrolling=True)
        return True

    except Exception as e:
        st.error(f"❌ Graph rendering error: {str(e)}")
        st.code(traceback.format_exc())
        return False
