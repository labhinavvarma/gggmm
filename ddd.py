def get_node_color(labels):
    """Get beautiful, vibrant colors for nodes"""
    if not labels:
        return "#95afc0"
    
    label = labels[0] if isinstance(labels, list) else str(labels)
    
    # Beautiful high-contrast color palette
    colors = {
        "Person": "#FF4757",        # Bright Red
        "User": "#5352ED",          # Bright Purple  
        "Employee": "#3742FA",      # Royal Blue
        "Customer": "#FF6B35",      # Bright Orange
        "Movie": "#00D2D3",         # Cyan
        "Film": "#00A8A8",          # Dark Cyan
        "Actor": "#FFD700",         # Gold
        "Director": "#9C27B0",      # Purple
        "Producer": "#673AB7",      # Deep Purple
        "Company": "#00BF63",       # Emerald
        "Organization": "#00A651",  # Green
        "Department": "#FFC107",    # Amber
        "Product": "#E91E63",       # Pink
        "Service": "#FF5722",       # Deep Orange
        "Location": "#8E24AA",      # Violet
        "City": "#7B1FA2",          # Dark Violet
        "Country": "#6A1B9A",       # Purple
        "Event": "#FF9800",         # Orange
        "Project": "#4CAF50",       # Light Green
        "Database": "#00BCD4",      # Cyan
        "Server": "#009688",        # Teal
        "Network": "#795548",       # Brown
        "Group": "#607D8B",         # Blue Gray
        "Team": "#455A64",          # Dark Blue Gray
        "Document": "#9E9E9E",      # Gray
        "File": "#757575",          # Dark Gray
        "Unknown": "#B0BEC5"        # Light Blue Gray
    }
    
    return colors.get(label, "#6C5CE7")

def get_relationship_color(rel_type):
    """Get beautiful colors for relationships"""
    colors = {
        "KNOWS": "#FF4757",         # Red
        "FRIEND_OF": "#FF3838",     # Bright Red
        "WORKS_FOR": "#3742FA",     # Blue
        "WORKS_IN": "#2F3542",      # Dark Blue
        "MANAGES": "#5352ED",       # Purple
        "REPORTS_TO": "#8B4AE8",    # Light Purple
        "LOCATED_IN": "#FFA502",    # Orange
        "LIVES_IN": "#FF6348",      # Red Orange
        "BELONGS_TO": "#2ED573",    # Green
        "OWNS": "#1DD1A1",          # Teal
        "CREATED": "#FFD32A",       # Yellow
        "USES": "#FF9FF3",          # Pink
        "ACTED_IN": "#54A0FF",      # Light Blue
        "DIRECTED": "#5F27CD",      # Purple
        "PRODUCED": "#00D2D3",      # Cyan
        "LOVES": "#FF3838",         # Bright Red
        "MARRIED_TO": "#FF6B9D",    # Pink
        "CONNECTED": "#778CA3",     # Gray Blue
        "RELATED": "#4B6584"        # Blue Gray
    }
    
    return colors.get(rel_type, "#778CA3")

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
                return str(name).strip()[:25]
        
        # Fallback to label + ID
        if labels and labels[0] != "Unknown":
            short_id = node_id.split(":")[-1][-4:] if ":" in node_id else node_id[-4:]
            return f"{labels[0]}_{short_id}"
        
        return f"Node_{node_id[-6:] if len(node_id) > 6 else node_id}"
        
    except Exception as e:
        return f"Node_{hash(str(node)) % 10000}"

def render_working_graph(graph_data: dict) -> bool:
    """Render beautiful, clean graph with no settings - only pure visualization"""
    
    if not graph_data:
        st.info("🔍 No graph data available.")
        return False
    
    try:
        nodes = graph_data.get("nodes", [])
        relationships = graph_data.get("relationships", [])
        
        if not nodes:
            st.info("📊 No nodes found.")
            return False
        
        # Show what we're creating
        st.markdown(f'<div class="success-box">🎨 <strong>Creating Beautiful Graph:</strong> {len(nodes)} nodes, {len(relationships)} relationships</div>', unsafe_allow_html=True)
        
        # Create clean Pyvis network - NO MENUS OR SETTINGS
        net = Network(
            height="700px",
            width="100%", 
            bgcolor="#FFFFFF",              # Pure white background
            font_color="#2C3E50",          # Dark text
            directed=True                   # Show direction arrows
        )
        
        # Disable ALL menus and settings
        net.show_buttons(filter_=['physics'])  # Only show physics button, then we'll hide it
        
        # Process nodes with beautiful styling
        added_nodes = set()
        
        for i, node in enumerate(nodes):
            try:
                # Simple node ID
                node_id = f"node_{i}"
                raw_id = str(node.get("id", f"node_{i}"))
                
                # Get display name
                display_name = safe_extract_node_name(node)
                
                # Get styling
                labels = node.get("labels", ["Unknown"])
                color = get_node_color(labels)
                
                # Create tooltip
                props = node.get("properties", {})
                tooltip_parts = [f"Type: {labels[0]}", f"Name: {display_name}"]
                for key, value in list(props.items())[:3]:
                    if key not in ['name', 'title', 'displayName']:
                        tooltip_parts.append(f"{key}: {str(value)[:30]}")
                tooltip = "\\n".join(tooltip_parts)
                
                # Add node with BEAUTIFUL styling
                net.add_node(
                    node_id,
                    label=display_name,
                    color={
                        'background': color,
                        'border': '#2C3E50',        # Dark border
                        'highlight': {
                            'background': color,
                            'border': '#E74C3C'      # Red highlight
                        }
                    },
                    size=45,                         # Large, consistent size
                    title=tooltip,
                    font={
                        'size': 18,                  # BIG labels
                        'color': '#FFFFFF',          # White text
                        'face': 'Arial Black',       # Bold font
                        'strokeWidth': 2,            # Text outline
                        'strokeColor': '#2C3E50'     # Dark outline
                    },
                    borderWidth=3,                   # Thick border
                    shadow={
                        'enabled': True,
                        'color': 'rgba(0,0,0,0.3)',
                        'size': 10,
                        'x': 3,
                        'y': 3
                    },
                    margin=10                        # Space around label
                )
                
                added_nodes.add((raw_id, node_id))
                
            except Exception as e:
                continue
        
        # Process relationships with beautiful styling
        id_mapping = dict(added_nodes)
        simple_nodes = {node_id for _, node_id in added_nodes}
        
        added_edges = 0
        for i, rel in enumerate(relationships):
            try:
                start_raw = str(rel.get("startNode", ""))
                end_raw = str(rel.get("endNode", ""))
                rel_type = str(rel.get("type", "CONNECTED"))
                
                start_id = id_mapping.get(start_raw)
                end_id = id_mapping.get(end_raw)
                
                if start_id and end_id and start_id in simple_nodes and end_id in simple_nodes:
                    color = get_relationship_color(rel_type)
                    
                    # Add beautiful relationship
                    net.add_edge(
                        start_id,
                        end_id,
                        label=rel_type,
                        color={
                            'color': color,
                            'highlight': '#E74C3C'   # Red highlight
                        },
                        width=4,                     # Thick lines
                        font={
                            'size': 14,              # Big relationship labels
                            'color': '#2C3E50',      # Dark text
                            'face': 'Arial Bold',
                            'strokeWidth': 2,
                            'strokeColor': '#FFFFFF',
                            'align': 'middle'
                        },
                        arrows={
                            'to': {
                                'enabled': True,
                                'scaleFactor': 1.2   # Bigger arrows
                            }
                        },
                        smooth={
                            'enabled': True,
                            'type': 'continuous',
                            'roundness': 0.3         # Smooth curves
                        }
                    )
                    
                    added_edges += 1
                    
            except Exception as e:
                continue
        
        # Set beautiful physics - no complex JSON, just clean settings
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
              "type": "continuous"
            }
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "enabled": true,
              "iterations": 100
            },
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 100,
              "springConstant": 0.04,
              "damping": 0.09,
              "avoidOverlap": 0.1
            }
          },
          "interaction": {
            "hover": true,
            "selectConnectedEdges": false
          }
        }
        """)
        
        st.write(f"✨ **Beautiful Graph Ready:** {len(simple_nodes)} highlighted nodes, {added_edges} styled relationships")
        
        # Generate and clean the HTML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            net.save_graph(f.name)
            html_file = f.name
        
        # Read and modify HTML to remove settings
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Remove the settings/configuration panel by hiding it with CSS
        clean_html = html_content.replace(
            '<body>',
            '''<body>
            <style>
                /* Hide all configuration panels and buttons */
                .vis-configuration-wrapper,
                .vis-configuration,
                .vis-config-button,
                .vis-config-item,
                .vis-config-range,
                #config,
                .config,
                [id*="config"],
                [class*="config"] {
                    display: none !important;
                    visibility: hidden !important;
                }
                
                /* Make sure the network takes full space */
                #mynetworkid {
                    width: 100% !important;
                    height: 700px !important;
                    border: none !important;
                }
                
                /* Beautiful gradient background */
                body {
                    margin: 0;
                    padding: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }
                
                /* Style the network container */
                .vis-network {
                    background: #FFFFFF !important;
                    border-radius: 15px !important;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3) !important;
                    margin: 10px !important;
                }
            </style>'''
        )
        
        # Beautiful wrapper
        wrapped_html = f"""
        <div style="
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 20px;
            padding: 5px;
            box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        ">
            <div style="
                background: linear-gradient(135deg, #2C3E50, #34495E);
                color: white;
                padding: 15px 25px;
                border-radius: 15px 15px 0 0;
                font-weight: bold;
                text-align: center;
                font-size: 18px;
            ">
                🎨 Beautiful Network Graph • {len(simple_nodes)} Nodes • {added_edges} Relationships
            </div>
            <div style="background: white; border-radius: 0 0 15px 15px; overflow: hidden;">
                {clean_html}
            </div>
        </div>
        """
        
        # Display the clean, beautiful graph
        components.html(wrapped_html, height=800, scrolling=False)
        
        # Show legend in a beautiful way
        st.markdown("### 🎨 Graph Legend")
        
        # Node types legend
        node_types = {}
        for node in nodes:
            labels = node.get("labels", ["Unknown"])
            if labels:
                label = labels[0]
                node_types[label] = get_node_color([label])
        
        if node_types:
            legend_html = '<div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0;">'
            for label, color in sorted(node_types.items()):
                legend_html += f'''
                <div style="
                    background: {color}; 
                    color: white; 
                    padding: 8px 15px; 
                    border-radius: 25px; 
                    font-weight: bold;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
                    border: 2px solid #2C3E50;
                ">
                    {label}
                </div>
                '''
            legend_html += '</div>'
            st.markdown(legend_html, unsafe_allow_html=True)
        
        # Relationship types legend
        rel_types = {}
        for rel in relationships:
            rel_type = rel.get("type", "CONNECTED")
            rel_types[rel_type] = get_relationship_color(rel_type)
        
        if rel_types:
            st.markdown("**🔗 Relationship Types:**")
            rel_legend_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0;">'
            for rel_type, color in sorted(rel_types.items()):
                rel_legend_html += f'''
                <div style="
                    background: {color}; 
                    color: white; 
                    padding: 6px 12px; 
                    border-radius: 20px; 
                    font-size: 12px;
                    font-weight: bold;
                    box-shadow: 0 3px 8px rgba(0,0,0,0.2);
                ">
                    {rel_type}
                </div>
                '''
            rel_legend_html += '</div>'
            st.markdown(rel_legend_html, unsafe_allow_html=True)
        
        # Cleanup
        try:
            os.unlink(html_file)
        except:
            pass
            
        return True
        
    except Exception as e:
        st.error(f"❌ Graph rendering failed: {str(e)}")
        st.code(traceback.format_exc())
        return False
