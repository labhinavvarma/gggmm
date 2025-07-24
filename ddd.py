def get_node_color(labels):
    """Get vibrant, highly differentiated colors for nodes based on Neo4j labels"""
    if not labels:
        return "#95afc0"
    
    label = labels[0] if isinstance(labels, list) else str(labels)
    
    # Enhanced color palette with high contrast and vibrant colors
    colors = {
        "Person": "#E74C3C",        # Bright Red
        "User": "#8E44AD",          # Purple
        "Employee": "#3498DB",       # Bright Blue
        "Customer": "#E67E22",       # Orange
        "Movie": "#1ABC9C",         # Turquoise
        "Film": "#16A085",          # Dark Turquoise
        "Actor": "#F39C12",         # Yellow-Orange
        "Director": "#9B59B6",      # Violet
        "Producer": "#34495E",      # Dark Blue-Gray
        "Company": "#2ECC71",       # Emerald Green
        "Organization": "#27AE60",  # Green
        "Department": "#F1C40F",    # Bright Yellow
        "Product": "#E91E63",       # Pink
        "Service": "#FF5722",       # Deep Orange
        "Location": "#9C27B0",      # Purple
        "City": "#673AB7",          # Deep Purple
        "Country": "#3F51B5",       # Indigo
        "Event": "#FF9800",         # Amber
        "Project": "#4CAF50",       # Light Green
        "Task": "#8BC34A",          # Light Green
        "Order": "#CDDC39",         # Lime
        "Purchase": "#FFEB3B",      # Yellow
        "Category": "#FFC107",      # Amber
        "Tag": "#FF5722",           # Deep Orange
        "Brand": "#795548",         # Brown
        "Model": "#607D8B",         # Blue Gray
        "Vehicle": "#212121",       # Almost Black
        "Building": "#424242",      # Dark Gray
        "Room": "#757575",          # Gray
        "Device": "#9E9E9E",        # Light Gray
        "Software": "#00BCD4",      # Cyan
        "Application": "#009688",   # Teal
        "Database": "#4DB6AC",      # Light Teal
        "Server": "#26A69A",        # Medium Teal
        "Network": "#00897B",       # Dark Teal
        "Account": "#006064",       # Very Dark Teal
        "Profile": "#004D40",       # Darkest Teal
        "Group": "#1B5E20",         # Dark Green
        "Team": "#2E7D32",          # Medium Green
        "Role": "#388E3C",          # Light Medium Green
        "Permission": "#43A047",    # Lighter Green
        "Access": "#4CAF50",        # Light Green
        "Resource": "#66BB6A",      # Very Light Green
        "Document": "#81C784",      # Pale Green
        "File": "#A5D6A7",          # Very Pale Green
        "Folder": "#C8E6C9",        # Extremely Pale Green
        # Default fallback for unknown types
        "Unknown": "#BDC3C7"        # Light Gray
    }
    
    return colors.get(label, "#95afc0")

def get_node_size(labels, properties=None):
    """Get node size based on node type and importance"""
    if not labels:
        return 40
    
    label = labels[0] if isinstance(labels, list) else str(labels)
    
    # Size mapping - more important/central nodes are larger
    sizes = {
        "Person": 50,       # People are often central
        "User": 50,
        "Employee": 45,
        "Customer": 45,
        "Company": 60,      # Companies are often central hubs
        "Organization": 60,
        "Department": 50,
        "Movie": 45,
        "Director": 50,
        "Producer": 45,
        "Actor": 40,
        "Project": 55,      # Projects often connect many entities
        "Event": 45,
        "Location": 50,     # Locations often connect many things
        "City": 50,
        "Country": 55,
        "Product": 40,
        "Service": 40,
        "Order": 35,
        "Category": 45,
        "Database": 50,
        "Server": 50,
        "Network": 45,
        "Group": 45,
        "Team": 45,
        "Document": 35,
        "File": 30,
        # Default
        "Unknown": 40
    }
    
    base_size = sizes.get(label, 40)
    
    # Increase size if node has many properties (indicates importance)
    if properties and len(properties) > 5:
        base_size += 5
    elif properties and len(properties) > 3:
        base_size += 3
    
    return base_size

def get_enhanced_font_config(labels):
    """Get enhanced font configuration for better visibility"""
    if not labels:
        return {'size': 16, 'color': '#000000', 'face': 'Arial', 'strokeWidth': 2, 'strokeColor': '#ffffff'}
    
    label = labels[0] if isinstance(labels, list) else str(labels)
    
    # Base font configuration with larger sizes and better contrast
    base_config = {
        'face': 'Arial Black',  # Bolder font
        'strokeWidth': 3,       # White outline for better readability
        'strokeColor': '#ffffff',
        'color': '#000000'      # Black text for maximum contrast
    }
    
    # Size based on node importance
    font_sizes = {
        "Person": 18,
        "User": 18,
        "Employee": 17,
        "Customer": 17,
        "Company": 20,      # Largest for most important
        "Organization": 20,
        "Department": 18,
        "Movie": 17,
        "Director": 18,
        "Producer": 17,
        "Actor": 16,
        "Project": 19,
        "Event": 17,
        "Location": 18,
        "City": 18,
        "Country": 19,
        "Product": 16,
        "Service": 16,
        "Database": 18,
        "Server": 18,
        "Network": 17,
        "Group": 17,
        "Team": 17,
        # Default
        "Unknown": 16
    }
    
    base_config['size'] = font_sizes.get(label, 16)
    return base_config

def render_working_graph(graph_data: dict) -> bool:
    """Render graph with enhanced visualization - bigger labels and more highlighted nodes"""
    
    if not graph_data:
        st.info("🔍 No graph data available.")
        return False
    
    try:
        nodes = graph_data.get("nodes", [])
        relationships = graph_data.get("relationships", [])
        
        if not nodes:
            st.info("📊 No nodes found.")
            return False
        
        # Show processing info
        st.markdown(f'<div class="success-box">🎨 <strong>Enhanced Visualization:</strong> {len(nodes)} highlighted nodes, {len(relationships)} colored relationships</div>', unsafe_allow_html=True)
        
        # Debug: Show sample node data
        if nodes:
            debug_key = f"debug_nodes_{hash(str(nodes[0])) % 1000}"
            show_debug = st.checkbox("🔍 Show Sample Node Data (Debug)", key=debug_key)
            if show_debug:
                st.write("**First node data:**")
                st.json(nodes[0])
                if len(nodes) > 1:
                    st.write("**Second node data:**")
                    st.json(nodes[1])
        
        # Show relationship types
        if relationships:
            rel_types = list(set(rel.get('type', 'UNKNOWN') for rel in relationships))
            st.markdown(f'<div class="success-box">🔗 <strong>Found relationships:</strong> {", ".join(sorted(rel_types))}</div>', unsafe_allow_html=True)
        
        # Create Pyvis network with enhanced settings
        net = Network(
            height="700px",        # Slightly taller
            width="100%", 
            bgcolor="#f8f9fa",     # Light background for better contrast
            font_color="#333333",
            # Enhanced physics for better node spacing
            select_menu=True,
            filter_menu=True,
        )
        
        # Add nodes with enhanced styling
        added_nodes = set()
        node_details = []
        
        for i, node in enumerate(nodes):
            try:
                # Create safe node ID
                raw_id = str(node.get("id", f"node_{i}"))
                node_id = f"node_{i}"  # Use simple sequential IDs for internal reference
                
                # Extract name safely
                display_name = safe_extract_node_name(node)
                node_details.append(display_name)
                
                # Debug: Show what names we're extracting
                if i < 3:  # Show first 3 for debugging
                    st.write(f"🔍 Debug: Node {i} → Name: '{display_name}' from {node.get('properties', {}).get('name', 'NO NAME FOUND')}")
                
                # Get enhanced styling
                labels = node.get("labels", ["Unknown"])
                color = get_node_color(labels)
                node_size = get_node_size(labels, node.get("properties"))
                font_config = get_enhanced_font_config(labels)
                
                # Create enhanced tooltip with actual data
                props = node.get("properties", {})
                tooltip_parts = [f"🏷️ {labels[0]}", f"📛 {display_name}"]
                
                # Add key properties to tooltip
                for key, value in list(props.items())[:4]:  # Show first 4 properties
                    if key not in ['name', 'title', 'displayName']:  # Don't repeat name variants
                        tooltip_parts.append(f"📝 {key}: {value}")
                
                tooltip = "\\n".join(tooltip_parts)
                
                # Add node with ENHANCED styling
                net.add_node(
                    node_id,
                    label=display_name,         # Display name as label
                    color={
                        'background': color,
                        'border': '#2c3e50',    # Dark border for definition
                        'highlight': {
                            'background': color,
                            'border': '#e74c3c'  # Red highlight border
                        },
                        'hover': {
                            'background': color,
                            'border': '#f39c12'  # Orange hover border
                        }
                    },
                    size=node_size,             # Dynamic size based on importance
                    title=tooltip,
                    font=font_config,           # Enhanced font with outline
                    borderWidth=3,              # Thicker border
                    borderWidthSelected=5,      # Even thicker when selected
                    shadow={                    # Add shadow for depth
                        'enabled': True,
                        'color': 'rgba(0,0,0,0.3)',
                        'size': 8,
                        'x': 2,
                        'y': 2
                    },
                    margin={                    # Add margin for label breathing room
                        'top': 5,
                        'bottom': 5,
                        'left': 5,
                        'right': 5
                    }
                )
                
                # Store mapping for relationships
                added_nodes.add((raw_id, node_id))
                
            except Exception as e:
                st.warning(f"⚠️ Skipped node {i}: {str(e)}")
                continue
        
        # Create ID mapping for relationships
        id_mapping = dict(added_nodes)
        simple_nodes = {node_id for _, node_id in added_nodes}
        
        # Add relationships with enhanced styling
        added_edges = 0
        relationship_details = []
        
        for i, rel in enumerate(relationships):
            try:
                start_raw = str(rel.get("startNode", rel.get("start", "")))
                end_raw = str(rel.get("endNode", rel.get("end", "")))
                rel_type = str(rel.get("type", "CONNECTED"))
                
                # Map to simple IDs
                start_id = id_mapping.get(start_raw)
                end_id = id_mapping.get(end_raw)
                
                # Only add if both nodes exist
                if start_id and end_id and start_id in simple_nodes and end_id in simple_nodes:
                    color = get_relationship_color(rel_type)
                    
                    # Add edge with ENHANCED styling
                    net.add_edge(
                        start_id,
                        end_id,
                        label=rel_type,
                        color={
                            'color': color,
                            'highlight': '#e74c3c',  # Red highlight
                            'hover': '#f39c12'       # Orange hover
                        },
                        width=4,                     # Thicker lines
                        font={
                            'size': 12,
                            'color': '#2c3e50',
                            'face': 'Arial',
                            'strokeWidth': 2,
                            'strokeColor': '#ffffff',
                            'align': 'middle'
                        },
                        shadow={                     # Add shadow to relationships too
                            'enabled': True,
                            'color': 'rgba(0,0,0,0.2)',
                            'size': 4,
                            'x': 1,
                            'y': 1
                        },
                        smooth={                     # Smooth curves for better aesthetics
                            'enabled': True,
                            'type': 'dynamic',
                            'roundness': 0.2
                        }
                    )
                    
                    added_edges += 1
                    relationship_details.append(f"{rel_type}: {start_id} → {end_id}")
                    
            except Exception as e:
                st.warning(f"⚠️ Skipped relationship {i}: {str(e)}")
                continue
        
        st.write(f"✅ **Enhanced Graph Created:** {len(simple_nodes)} highlighted nodes, {added_edges} styled relationships")
        
        # Show details with checkboxes
        node_details_key = f"node_details_{hash(str(node_details)) % 1000}"
        show_node_details = st.checkbox(f"👥 Show Enhanced Node Names ({len(node_details)})", key=node_details_key)
        
        if show_node_details:
            # Group by node type for better organization
            node_by_type = {}
            for i, node in enumerate(nodes):
                labels = node.get("labels", ["Unknown"])
                node_type = labels[0] if labels else "Unknown"
                if node_type not in node_by_type:
                    node_by_type[node_type] = []
                node_by_type[node_type].append(node_details[i])
            
            for node_type, names in sorted(node_by_type.items()):
                color = get_node_color([node_type])
                st.markdown(f'<div style="background: {color}; color: white; padding: 5px; border-radius: 5px; margin: 2px 0;"><strong>{node_type} ({len(names)}):</strong> {", ".join(names[:5])}{f" and {len(names)-5} more" if len(names) > 5 else ""}</div>', unsafe_allow_html=True)
        
        rel_details_key = f"rel_details_{hash(str(relationship_details)) % 1000}"
        show_rel_details = st.checkbox(f"🔗 Show Enhanced Relationships ({len(relationship_details)})", key=rel_details_key)
        
        if show_rel_details:
            for rel in relationship_details[:10]:
                st.write(f"• {rel}")
            if len(relationship_details) > 10:
                st.write(f"... and {len(relationship_details) - 10} more")
        
        # Enhanced physics configuration
        try:
            net.set_options("""
            var options = {
              "physics": {
                "enabled": true,
                "stabilization": {"iterations": 100},
                "barnesHut": {
                  "gravitationalConstant": -8000,
                  "centralGravity": 0.3,
                  "springLength": 95,
                  "springConstant": 0.04,
                  "damping": 0.09,
                  "avoidOverlap": 0.1
                }
              },
              "interaction": {
                "hover": true,
                "hoverConnectedEdges": true,
                "selectConnectedEdges": false,
                "tooltipDelay": 200
              },
              "layout": {
                "improvedLayout": true,
                "randomSeed": 2
              }
            }
            """)
        except:
            # Fallback to simple physics
            net.barnes_hut()
        
        # Generate HTML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            net.save_graph(f.name)
            html_file = f.name
        
        # Read and display with enhanced wrapper
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Enhanced wrapper with gradient border
        wrapped_html = f"""
        <div style="
            border: 4px solid;
            border-image: linear-gradient(45deg, #667eea, #764ba2, #667eea) 1;
            border-radius: 15px; 
            overflow: hidden; 
            background: white;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        ">
            <div style="
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 15px 25px;
                font-weight: bold;
                font-size: 16px;
            ">
                🎨 Enhanced Network Graph | {len(simple_nodes)} Highlighted Nodes | {added_edges} Styled Relationships | Interactive Visualization
            </div>
            {html_content}
        </div>
        """
        
        # Display with increased height for better view
        components.html(wrapped_html, height=750, scrolling=False)
        
        # Cleanup
        try:
            os.unlink(html_file)
        except:
            pass
        
        return True
        
    except Exception as e:
        st.error(f"❌ Enhanced graph rendering failed: {str(e)}")
        
        debug_details_key = f"debug_details_{hash(str(e)) % 1000}"
        show_debug = st.checkbox("🔍 Show Detailed Debug Info", key=debug_details_key)
        
        if show_debug:
            st.write("**Error:**")
            st.code(str(e))
            st.write("**Full traceback:**")
            st.code(traceback.format_exc())
            if nodes:
                st.write("**Sample node data:**")
                st.json(nodes[0])
            if relationships:
                st.write("**Sample relationship data:**")
                st.json(relationships[0])
        
        return False
