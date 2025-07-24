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
import hashlib

# Page configuration
st.set_page_config(
    page_title="Neo4j Graph Explorer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better history styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        max-width: 100%;
    .suggestion-button {
        background: linear-gradient(45deg, #a8edea, #fed6e3) !important;
        color: #2c3e50 !important;
        border: 1px solid #667eea !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 0.75rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .suggestion-button:hover {
        background: linear-gradient(45deg, #fed6e3, #a8edea) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
    }
    
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .stButton button {
        width: 100%;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 0.25rem;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .suggestion-button {
        background: linear-gradient(45deg, #a8edea, #fed6e3) !important;
        color: #2c3e50 !important;
        border: 1px solid #667eea !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 0.75rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .suggestion-button:hover {
        background: linear-gradient(45deg, #fed6e3, #a8edea) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.25rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #84fab0, #8fd3f4);
        border: none;
        color: #2c3e50;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffecd2, #fcb69f);
        border: none;
        color: #8b4513;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .legend-box {
        background: linear-gradient(135deg, #ffecd2, #fcb69f);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .detailed-response-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    
    .response-section {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid rgba(255, 255, 255, 0.3);
    }
    
    .insight-box {
        background: linear-gradient(135deg, #a8edea, #fed6e3);
        color: #2c3e50;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .history-entry {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(240, 147, 251, 0.3);
    }
    
    .history-header {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    
    .history-section {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.25rem 0;
        border-left: 3px solid rgba(255, 255, 255, 0.4);
    }
    
    .cypher-query {
        background: rgba(0, 0, 0, 0.3);
        color: #a8ff60;
        padding: 0.75rem;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        margin: 0.5rem 0;
        border-left: 3px solid #a8ff60;
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
        "connection_status": "unknown",
        "detailed_analysis": None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# Header
st.markdown('<h1 class="main-header">🕸️ Neo4j Graph Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6c757d; font-size: 1.2rem;"><strong>🔍 Complete Query Tracking</strong> • <strong>📊 Detailed Analysis</strong></p>', unsafe_allow_html=True)

def call_agent_api(question: str, node_limit: int = 50) -> dict:
    """Enhanced API call function with better response tracking"""
    try:
        api_url = "http://localhost:8081/chat"
        payload = {
            "question": question,
            "session_id": st.session_state.session_id,
            "node_limit": node_limit
        }
        
        with st.spinner("🤖 Processing your request and generating detailed analysis..."):
            response = requests.post(api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            st.session_state.connection_status = "connected"
            return result
            
    except requests.exceptions.ConnectionError:
        st.session_state.connection_status = "disconnected"
        st.error("❌ Cannot connect to agent API. Please ensure the FastAPI server is running on port 8081.")
        return None
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
        return None

def create_detailed_history_entry(question: str, result: dict, detailed_analysis: dict = None) -> dict:
    """Create a comprehensive history entry with all details"""
    timestamp = datetime.now().isoformat()
    
    # Extract basic info
    answer = result.get("answer", "")
    graph_data = result.get("graph_data", {})
    tool_used = result.get("tool", "unknown")
    cypher_query = result.get("query", "")
    trace = result.get("trace", "")
    execution_time = result.get("execution_time_ms", 0)
    
    # Analyze graph data for history
    nodes = graph_data.get("nodes", []) if graph_data else []
    relationships = graph_data.get("relationships", []) if graph_data else []
    
    # Get node types and relationship types
    node_types = {}
    if nodes:
        for node in nodes:
            labels = node.get("labels", ["Unknown"])
            label = labels[0] if labels else "Unknown"
            node_types[label] = node_types.get(label, 0) + 1
    
    relationship_types = {}
    if relationships:
        for rel in relationships:
            rel_type = rel.get("type", "Unknown")
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
    
    # Calculate key metrics
    connectivity = len(relationships) / max(len(nodes), 1) if nodes else 0
    total_properties = sum(len(node.get("properties", {})) for node in nodes) if nodes else 0
    avg_properties = total_properties / len(nodes) if nodes else 0
    
    # Create comprehensive history entry
    history_entry = {
        "timestamp": timestamp,
        "question": question,
        "answer": answer,
        "graph_data": graph_data,
        
        # Enhanced fields for detailed history
        "tool_used": tool_used,
        "cypher_query": cypher_query,
        "trace": trace,
        "execution_time_ms": execution_time,
        
        # Graph analysis summary
        "nodes_count": len(nodes),
        "relationships_count": len(relationships),
        "node_types": node_types,
        "relationship_types": relationship_types,
        "connectivity": connectivity,
        "avg_properties": avg_properties,
        
        # Detailed analysis if available
        "detailed_analysis": detailed_analysis,
        
        # Summary insights
        "key_insights": generate_history_insights(nodes, relationships, question),
        "data_summary": create_data_summary(nodes, relationships),
        
        # Success metrics
        "success": bool(result.get("success", True)),
        "error": result.get("error"),
    }
    
    return history_entry

def generate_history_insights(nodes, relationships, question):
    """Generate key insights for history display"""
    insights = []
    
    if not nodes:
        insights.append("🔍 No data found - query returned empty results")
        return insights
    
    # Network insights
    if relationships:
        density = len(relationships) / len(nodes)
        if density > 2:
            insights.append(f"🕸️ Highly connected network ({density:.1f} connections/node)")
        elif density > 1:
            insights.append(f"🔗 Well connected ({density:.1f} connections/node)")
        else:
            insights.append(f"📊 Sparse network ({density:.1f} connections/node)")
    else:
        insights.append("📍 Isolated nodes (no relationships found)")
    
    # Data type insights
    node_types = list(set(node.get("labels", [None])[0] for node in nodes if node.get("labels")))
    if len(node_types) == 1:
        insights.append(f"🎯 Focused on {node_types[0]} entities")
    elif len(node_types) > 1:
        insights.append(f"🎨 Diverse data: {len(node_types)} entity types")
    
    # Query type insights
    if "person" in question.lower() or "people" in question.lower():
        insights.append("👥 Social network analysis")
    elif "company" in question.lower():
        insights.append("🏢 Business network analysis")
    elif "all" in question.lower():
        insights.append("📊 Comprehensive database query")
    
    return insights[:3]  # Limit to 3 key insights

def create_data_summary(nodes, relationships):
    """Create a concise data summary for history"""
    if not nodes:
        return "No data returned"
    
    # Node summary
    node_types = {}
    for node in nodes:
        labels = node.get("labels", ["Unknown"])
        label = labels[0] if labels else "Unknown"
        node_types[label] = node_types.get(label, 0) + 1
    
    node_summary = ", ".join([f"{k}({v})" for k, v in node_types.items()])
    
    # Relationship summary
    if relationships:
        rel_types = {}
        for rel in relationships:
            rel_type = rel.get("type", "Unknown")
            rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
        
        rel_summary = ", ".join([f"{k}({v})" for k, v in rel_types.items()])
        return f"Nodes: {node_summary} | Relationships: {rel_summary}"
    else:
        return f"Nodes: {node_summary} | No relationships"

def display_detailed_history_entry(entry: dict, index: int):
    """Display a comprehensive history entry without nested expanders"""
    timestamp = entry.get("timestamp", "")
    question = entry.get("question", "")
    
    # Create expandable history entry
    with st.expander(
        f"📊 Query #{len(st.session_state.conversation_history) - index}: {question[:50]}..." + 
        f" | {entry.get('nodes_count', 0)} nodes, {entry.get('relationships_count', 0)} rels",
        expanded=False
    ):
        # History entry container
        st.markdown(f'''
        <div class="history-entry">
            <div class="history-header">
                🕐 <strong>Executed:</strong> {timestamp[:19]} | 
                ⚡ <strong>Time:</strong> {entry.get("execution_time_ms", 0):.1f}ms |
                🔧 <strong>Tool:</strong> {entry.get("tool_used", "unknown")}
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Question section
        st.markdown("**❓ Original Question:**")
        st.info(question)
        
        # Cypher query section (if available)
        if entry.get("cypher_query"):
            st.markdown("**🔧 Cypher Query Executed:**")
            st.markdown(f'''
            <div class="cypher-query">
                {entry.get("cypher_query", "No query available")}
            </div>
            ''', unsafe_allow_html=True)
        
        # Results overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes Found", entry.get("nodes_count", 0))
        with col2:
            st.metric("Relationships", entry.get("relationships_count", 0))
        with col3:
            st.metric("Connectivity", f"{entry.get('connectivity', 0):.1f}")
        with col4:
            st.metric("Avg Properties", f"{entry.get('avg_properties', 0):.1f}")
        
        # Data composition
        if entry.get("node_types") or entry.get("relationship_types"):
            st.markdown("**📊 Data Composition:**")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if entry.get("node_types"):
                    st.write("**Node Types:**")
                    for node_type, count in entry["node_types"].items():
                        st.write(f"• {node_type}: {count}")
            
            with col_b:
                if entry.get("relationship_types"):
                    st.write("**Relationship Types:**")
                    for rel_type, count in entry["relationship_types"].items():
                        st.write(f"• {rel_type}: {count}")
        
        # Key insights
        if entry.get("key_insights"):
            st.markdown("**🔍 Key Insights from this Query:**")
            for insight in entry["key_insights"]:
                st.markdown(f"• {insight}")
        
        # Data summary
        if entry.get("data_summary"):
            st.markdown("**📋 Data Summary:**")
            st.code(entry["data_summary"])
        
        # Full response section - using checkbox instead of nested expander
        if entry.get("answer"):
            st.markdown("**🤖 AI Response:**")
            answer_preview = entry["answer"].replace("**", "").replace("#", "")
            
            # Create unique key for this checkbox
            show_full_key = f"show_full_{index}_{hash(question) % 1000}"
            
            if len(answer_preview) > 200:
                st.write(f"{answer_preview[:200]}...")
                
                # Use checkbox to toggle full response instead of nested expander
                show_full = st.checkbox("📖 Show Full Response", key=show_full_key)
                if show_full:
                    st.markdown("---")
                    st.markdown("**Complete Response:**")
                    st.markdown(entry["answer"])
                    st.markdown("---")
            else:
                st.write(answer_preview)
        
        # Technical details section - using checkbox instead of nested expander
        tech_details_key = f"tech_details_{index}_{hash(question) % 1000}"
        show_tech_details = st.checkbox("🔧 Show Technical Details", key=tech_details_key)
        
        if show_tech_details:
            st.markdown("---")
            st.markdown("**Technical Details:**")
            st.write(f"**Tool Used:** {entry.get('tool_used', 'unknown')}")
            st.write(f"**Session ID:** {entry.get('trace', 'N/A')[:50]}...")
            st.write(f"**Success:** {'✅ Yes' if entry.get('success', True) else '❌ No'}")
            if entry.get("error"):
                st.error(f"Error: {entry['error']}")
            
            # Show raw graph data structure
            if entry.get("graph_data") and entry["graph_data"].get("nodes"):
                st.write("**Sample Node Structure:**")
                st.json(entry["graph_data"]["nodes"][0])
            
            if entry.get("graph_data") and entry["graph_data"].get("relationships"):
                st.write("**Sample Relationship Structure:**")
                st.json(entry["graph_data"]["relationships"][0])
            st.markdown("---")
        
        # Action buttons for this history entry
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            if st.button(f"🔄 Repeat Query #{len(st.session_state.conversation_history) - index}", key=f"repeat_{index}"):
                # Re-execute the same question
                result = call_agent_api(question, 30)
                if result:
                    if result.get("graph_data"):
                        st.session_state.graph_data = result["graph_data"]
                        detailed_analysis = generate_detailed_analysis(
                            result["graph_data"], 
                            question, 
                            result.get("answer", "")
                        )
                        st.session_state.detailed_analysis = detailed_analysis
                    
                    st.session_state.last_response = result
                    st.rerun()
        
        with col_y:
            if entry.get("graph_data") and st.button(f"📊 Load Graph #{len(st.session_state.conversation_history) - index}", key=f"load_{index}"):
                # Load this entry's graph data
                st.session_state.graph_data = entry["graph_data"]
                if entry.get("detailed_analysis"):
                    st.session_state.detailed_analysis = entry["detailed_analysis"]
                st.success("Graph loaded from history!")
                st.rerun()
        
        with col_z:
            if st.button(f"📋 Copy Query #{len(st.session_state.conversation_history) - index}", key=f"copy_{index}"):
                # Show the query for copying
                if entry.get("cypher_query"):
                    st.code(entry["cypher_query"], language="cypher")
                    st.success("Query displayed above for copying!")

def generate_detailed_analysis(graph_data: dict, question: str, api_response: str) -> dict:
    """Generate comprehensive analysis of the query results"""
    if not graph_data:
        return {
            "summary": "No graph data was returned from the query.",
            "insights": [],
            "recommendations": []
        }
    
    nodes = graph_data.get("nodes", [])
    relationships = graph_data.get("relationships", [])
    
    # Analyze nodes
    node_analysis = analyze_nodes(nodes)
    
    # Analyze relationships
    relationship_analysis = analyze_relationships(relationships, nodes)
    
    # Generate insights
    insights = generate_insights(nodes, relationships, question)
    
    # Generate recommendations
    recommendations = generate_recommendations(nodes, relationships, question)
    
    # Create summary
    summary = create_detailed_summary(nodes, relationships, question, api_response)
    
    return {
        "summary": summary,
        "node_analysis": node_analysis,
        "relationship_analysis": relationship_analysis,
        "insights": insights,
        "recommendations": recommendations,
        "raw_stats": {
            "total_nodes": len(nodes),
            "total_relationships": len(relationships),
            "node_types": len(set(node.get("labels", [None])[0] for node in nodes if node.get("labels"))),
            "relationship_types": len(set(rel.get("type") for rel in relationships)),
            "connectivity": len(relationships) / max(len(nodes), 1)
        }
    }

def analyze_nodes(nodes):
    """Detailed node analysis"""
    if not nodes:
        return {"message": "No nodes found in the results."}
    
    # Count by type
    node_types = {}
    properties_analysis = {}
    
    for node in nodes:
        labels = node.get("labels", ["Unknown"])
        label = labels[0] if labels else "Unknown"
        
        # Count types
        node_types[label] = node_types.get(label, 0) + 1
        
        # Analyze properties
        props = node.get("properties", {})
        for prop_name in props.keys():
            if prop_name not in properties_analysis:
                properties_analysis[prop_name] = {"count": 0, "sample_values": []}
            properties_analysis[prop_name]["count"] += 1
            if len(properties_analysis[prop_name]["sample_values"]) < 3:
                properties_analysis[prop_name]["sample_values"].append(str(props[prop_name]))
    
    return {
        "total_nodes": len(nodes),
        "node_types": node_types,
        "properties_analysis": properties_analysis,
        "most_common_type": max(node_types.items(), key=lambda x: x[1]) if node_types else None
    }

def analyze_relationships(relationships, nodes):
    """Detailed relationship analysis"""
    if not relationships:
        return {"message": "No relationships found in the results."}
    
    # Count by type
    rel_types = {}
    connection_patterns = {}
    
    for rel in relationships:
        rel_type = rel.get("type", "Unknown")
        rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
        
        # Analyze connection patterns (if we have node data)
        start_node = find_node_by_id(rel.get("startNode"), nodes)
        end_node = find_node_by_id(rel.get("endNode"), nodes)
        
        if start_node and end_node:
            start_type = start_node.get("labels", ["Unknown"])[0]
            end_type = end_node.get("labels", ["Unknown"])[0]
            pattern = f"{start_type} -> {end_type}"
            connection_patterns[pattern] = connection_patterns.get(pattern, 0) + 1
    
    return {
        "total_relationships": len(relationships),
        "relationship_types": rel_types,
        "connection_patterns": connection_patterns,
        "most_common_relationship": max(rel_types.items(), key=lambda x: x[1]) if rel_types else None
    }

def find_node_by_id(node_id, nodes):
    """Find a node by its ID"""
    for node in nodes:
        if str(node.get("id")) == str(node_id):
            return node
    return None

def generate_insights(nodes, relationships, question):
    """Generate smart insights about the data"""
    insights = []
    
    if not nodes:
        insights.append("🔍 **No Data Found**: Your query didn't return any nodes. Try broadening your search criteria.")
        return insights
    
    # Network density insight
    if relationships:
        density = len(relationships) / len(nodes)
        if density > 2:
            insights.append(f"🕸️ **Highly Connected Network**: Your data shows a dense network with {density:.1f} connections per node on average. This suggests strong interconnectedness.")
        elif density > 1:
            insights.append(f"🔗 **Well Connected**: Each node has about {density:.1f} connections on average, indicating good connectivity in your network.")
        else:
            insights.append(f"📊 **Sparse Network**: Your network has {density:.1f} connections per node, suggesting more isolated or loosely connected data.")
    
    # Node diversity insight
    node_types = set(node.get("labels", [None])[0] for node in nodes if node.get("labels"))
    if len(node_types) > 3:
        insights.append(f"🎨 **Diverse Data Types**: Your results include {len(node_types)} different types of entities: {', '.join(sorted(node_types))}. This shows a rich, varied dataset.")
    elif len(node_types) == 1:
        insights.append(f"🎯 **Focused Data**: All your results are of the same type ({list(node_types)[0]}), indicating a targeted query result.")
    
    # Relationship diversity insight
    if relationships:
        rel_types = set(rel.get("type") for rel in relationships)
        if len(rel_types) > 2:
            insights.append(f"🔀 **Multiple Relationship Types**: Your network contains {len(rel_types)} different types of relationships: {', '.join(sorted(rel_types))}. This indicates complex interconnections.")
    
    # Property richness insight
    total_properties = sum(len(node.get("properties", {})) for node in nodes)
    avg_properties = total_properties / len(nodes)
    if avg_properties > 3:
        insights.append(f"📋 **Rich Data**: Your nodes contain an average of {avg_properties:.1f} properties each, indicating detailed, information-rich data.")
    elif avg_properties < 1:
        insights.append("📝 **Simple Structure**: Your nodes have minimal property data, focusing on relationships rather than detailed attributes.")
    
    return insights

def generate_recommendations(nodes, relationships, question):
    """Generate actionable recommendations"""
    recommendations = []
    
    if not nodes:
        recommendations.append("🔍 **Try a broader search**: Your query returned no results. Consider using less specific criteria or checking if the data exists.")
        return recommendations
    
    # Based on connectivity
    if not relationships:
        recommendations.append("🔗 **Explore connections**: Your nodes appear isolated. Try queries like 'Show relationships for these nodes' to discover connections.")
    
    # Based on data types
    node_types = set(node.get("labels", [None])[0] for node in nodes if node.get("labels"))
    if "Person" in node_types:
        recommendations.append("👥 **Explore social connections**: You have Person nodes. Try 'Show who knows whom' or 'Find friendship networks' to explore social relationships.")
    
    if "Company" in node_types:
        recommendations.append("🏢 **Business analysis**: You have Company data. Consider exploring 'Company relationships', 'Who works where', or 'Business partnerships'.")
    
    # Based on network size
    if len(nodes) > 20:
        recommendations.append("📊 **Use filters**: Your result set is large. Consider filtering by specific properties or node types for clearer visualization.")
    elif len(nodes) < 5:
        recommendations.append("🔍 **Expand your view**: Your result set is small. Try broader queries or explore connected nodes to see more of the network.")
    
    # Based on question context  
    if "all" in question.lower():
        recommendations.append("🎯 **Focus your query**: You asked for 'all' data. Try more specific queries like 'Person nodes older than 30' or 'Companies in Technology sector'.")
    
    return recommendations

def create_detailed_summary(nodes, relationships, question, api_response):
    """Create a comprehensive summary"""
    if not nodes:
        return "Your query didn't return any data. This could mean the data doesn't exist or your search criteria were too specific."
    
    # Extract key metrics
    node_count = len(nodes)
    rel_count = len(relationships)
    node_types = list(set(node.get("labels", [None])[0] for node in nodes if node.get("labels")))
    rel_types = list(set(rel.get("type") for rel in relationships)) if relationships else []
    
    summary = f"""
**🎯 Query Execution Summary**

Your question "{question}" returned {node_count} nodes and {rel_count} relationships from the Neo4j database.

**📊 Data Composition:**
- **Node Types Found:** {', '.join(node_types) if node_types else 'Mixed types'}
- **Relationship Types:** {', '.join(rel_types) if rel_types else 'No relationships found'}
- **Network Density:** {rel_count / node_count:.2f} connections per node

**🔍 What This Means:**
{_interpret_results(node_count, rel_count, node_types, rel_types, question)}

**⚡ Performance:**
- Query executed successfully in the reported timeframe
- Data retrieved and processed for visualization
- Graph network successfully constructed with all relationships preserved
    """.strip()
    
    return summary

def _interpret_results(node_count, rel_count, node_types, rel_types, question):
    """Interpret what the results mean in context"""
    interpretations = []
    
    if node_count == 0:
        return "No data matched your query criteria. Consider broadening your search or checking if the data exists in the database."
    
    if rel_count == 0:
        interpretations.append("The nodes exist but appear to be isolated (no relationships found)")
    elif rel_count > node_count:
        interpretations.append("This is a highly connected network with multiple relationships per node")
    else:
        interpretations.append("This shows a moderately connected network structure")
    
    if len(node_types) == 1:
        interpretations.append(f"All results are of type '{node_types[0]}', indicating a focused query")
    elif len(node_types) > 3:
        interpretations.append("Multiple entity types suggest a diverse, interconnected dataset")
    
    if "person" in question.lower() or "people" in question.lower():
        interpretations.append("This appears to be social network data showing people and their connections")
    elif "company" in question.lower():
        interpretations.append("This looks like business/organizational data showing company relationships")
    
    return ". ".join(interpretations) + "."

def get_node_color(labels):
    """Get vibrant colors for nodes"""
    if not labels:
        return "#95afc0"
    
    label = labels[0] if isinstance(labels, list) else str(labels)
    
    colors = {
        "Person": "#FF6B6B",       # Bright Red
        "Movie": "#4ECDC4",        # Turquoise
        "Company": "#45B7D1",      # Blue
        "Product": "#96CEB4",      # Green
        "Location": "#FECA57",     # Yellow
        "Event": "#FF9FF3",        # Pink
        "User": "#A55EEA",         # Purple
        "Order": "#26DE81",        # Emerald
        "Category": "#FD79A8",     # Rose
        "Department": "#6C5CE7",   # Indigo
        "Project": "#FDCB6E",      # Orange
        "Actor": "#00CEC9",        # Cyan
        "Director": "#E84393",     # Magenta
        "Producer": "#00B894"      # Teal
    }
    
    return colors.get(label, "#95afc0")

def get_relationship_color(rel_type):
    """Get colors for relationships"""
    colors = {
        "KNOWS": "#e74c3c",
        "FRIEND_OF": "#e74c3c",
        "WORKS_FOR": "#3498db",
        "MANAGES": "#9b59b6",
        "LOCATED_IN": "#f39c12",
        "BELONGS_TO": "#27ae60",
        "CREATED": "#e91e63",
        "OWNS": "#673ab7",
        "USES": "#009688",
        "ACTED_IN": "#2196f3",
        "DIRECTED": "#ff9800",
        "PRODUCED": "#4caf50",
        "LOVES": "#e91e63",
        "MARRIED_TO": "#fd79a8"
    }
    
    return colors.get(rel_type, "#666666")

def safe_extract_node_name(node):
    """Safely extract display name from node"""
    try:
        props = node.get("properties", {})
        labels = node.get("labels", ["Unknown"])
        node_id = str(node.get("id", ""))
        
        # Debug: Print what we're working with
        print(f"🔍 Extracting name from node: {node}")
        print(f"🔍 Properties: {props}")
        
        # Try different name properties in order of preference
        name_options = [
            props.get("name"),           # Most common
            props.get("title"),          # For movies, books, etc.
            props.get("displayName"),    # Alternative display name
            props.get("username"),       # For user accounts
            props.get("label"),          # Generic label
            props.get("fullName"),       # Full name variant
            props.get("firstName"),      # First name as fallback
        ]
        
        for name in name_options:
            if name and str(name).strip():
                extracted_name = str(name).strip()[:30]  # Increased length
                print(f"✅ Found name: '{extracted_name}'")
                return extracted_name
        
        # If no name property found, try to create meaningful name from label
        if labels and labels[0] != "Unknown":
            # Try to use label + part of ID for uniqueness
            short_id = node_id.split(":")[-1][-4:] if ":" in node_id else node_id[-4:]
            fallback_name = f"{labels[0]}_{short_id}"
            print(f"⚠️ Using fallback name: '{fallback_name}'")
            return fallback_name
        
        # Last resort: use Node_ + short ID
        final_fallback = f"Node_{node_id[-6:] if len(node_id) > 6 else node_id}"
        print(f"❌ Using final fallback: '{final_fallback}'")
        return final_fallback
        
    except Exception as e:
        error_name = f"Node_{hash(str(node)) % 10000}"
        print(f"💥 Error extracting name: {e}, using: '{error_name}'")
        return error_name

def create_simple_legend(nodes, relationships):
    """Create a simple legend"""
    try:
        # Get unique node types
        node_types = {}
        for node in nodes:
            labels = node.get("labels", ["Unknown"])
            if labels:
                label = labels[0]
                node_types[label] = get_node_color([label])
        
        # Get unique relationship types
        rel_types = {}
        for rel in relationships:
            rel_type = rel.get("type", "CONNECTED")
            rel_types[rel_type] = get_relationship_color(rel_type)
        
        legend_html = '<div class="legend-box">'
        legend_html += '<h4 style="margin-top: 0;">🎨 Graph Legend</h4>'
        
        if node_types:
            legend_html += '<p><strong>📊 Node Types:</strong><br>'
            for label, color in sorted(node_types.items()):
                legend_html += f'<span style="background: {color}; color: white; padding: 2px 8px; border-radius: 10px; margin: 2px; font-size: 12px;">{label}</span> '
            legend_html += '</p>'
        
        if rel_types:
            legend_html += '<p><strong>🔗 Relationship Types:</strong><br>'
            for rel_type, color in sorted(rel_types.items()):
                legend_html += f'<span style="background: {color}; color: white; padding: 2px 8px; border-radius: 10px; margin: 2px; font-size: 12px;">{rel_type}</span> '
            legend_html += '</p>'
        
        legend_html += '</div>'
        return legend_html
        
    except Exception as e:
        return f'<div class="legend-box">Legend error: {str(e)}</div>'

def render_working_graph(graph_data: dict) -> bool:
    """Render graph with simplified, working configuration"""
    
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
        st.markdown(f'<div class="success-box">🎨 <strong>Processing:</strong> {len(nodes)} nodes, {len(relationships)} relationships</div>', unsafe_allow_html=True)
        
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
        
        # Create Pyvis network with SIMPLE settings
        net = Network(
            height="650px",
            width="100%", 
            bgcolor="#ffffff",
            font_color="#333333"
        )
        
        # Add nodes safely
        added_nodes = set()
        node_details = []
        
        for i, node in enumerate(nodes):
            try:
                # Create safe node ID
                raw_id = str(node.get("id", f"node_{i}"))
                node_id = f"node_{i}"  # Use simple sequential IDs for internal reference
                
                # Extract name safely - this is the key fix
                display_name = safe_extract_node_name(node)
                node_details.append(display_name)
                
                # Debug: Show what names we're extracting
                if i < 3:  # Show first 3 for debugging
                    st.write(f"🔍 Debug: Node {i} → Name: '{display_name}' from {node.get('properties', {}).get('name', 'NO NAME FOUND')}")
                
                # Get colors
                labels = node.get("labels", ["Unknown"])
                color = get_node_color(labels)
                
                # Create better tooltip with actual data
                props = node.get("properties", {})
                tooltip_parts = [f"Name: {display_name}"]
                if labels:
                    tooltip_parts.append(f"Type: {labels[0]}")
                for key, value in list(props.items())[:3]:  # Show first 3 properties
                    if key != 'name':  # Don't repeat name
                        tooltip_parts.append(f"{key}: {value}")
                tooltip = "\\n".join(tooltip_parts)
                
                # Add node with the DISPLAY NAME as the label
                net.add_node(
                    node_id,                    # Internal ID for Pyvis
                    label=display_name,         # This should show the actual name
                    color=color,
                    size=30,                    # Slightly larger
                    title=tooltip,
                    font={'size': 14, 'color': '#000000'},  # Better font
                    borderWidth=2,
                    borderWidthSelected=4
                )
                
                # Store mapping for relationships
                added_nodes.add((raw_id, node_id))
                
            except Exception as e:
                st.warning(f"⚠️ Skipped node {i}: {str(e)}")
                continue
        
        # Create ID mapping for relationships
        id_mapping = dict(added_nodes)
        simple_nodes = {node_id for _, node_id in added_nodes}
        
        # Add relationships safely
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
                    
                    # Add edge with SIMPLE configuration
                    net.add_edge(
                        start_id,
                        end_id,
                        label=rel_type,
                        color=color,
                        width=3
                    )
                    
                    added_edges += 1
                    relationship_details.append(f"{rel_type}: {start_id} → {end_id}")
                    
            except Exception as e:
                st.warning(f"⚠️ Skipped relationship {i}: {str(e)}")
                continue
        
        st.write(f"✅ **Successfully created:** {len(simple_nodes)} nodes, {added_edges} relationships")
        
        # Show details with checkboxes instead of nested expanders
        node_details_key = f"node_details_{hash(str(node_details)) % 1000}"
        show_node_details = st.checkbox(f"👥 Show Node Names ({len(node_details)})", key=node_details_key)
        
        if show_node_details:
            for name in node_details[:10]:
                st.write(f"• {name}")
            if len(node_details) > 10:
                st.write(f"... and {len(node_details) - 10} more")
        
        rel_details_key = f"rel_details_{hash(str(relationship_details)) % 1000}"
        show_rel_details = st.checkbox(f"🔗 Show Relationships ({len(relationship_details)})", key=rel_details_key)
        
        if show_rel_details:
            for rel in relationship_details[:10]:
                st.write(f"• {rel}")
            if len(relationship_details) > 10:
                st.write(f"... and {len(relationship_details) - 10} more")
        
        # Use VERY SIMPLE options that won't cause JSON errors
        try:
            net.barnes_hut()  # Use built-in physics method instead of complex JSON
        except:
            pass  # If this fails, just use default physics
        
        # Generate HTML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            net.save_graph(f.name)
            html_file = f.name
        
        # Read and display
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Simple wrapper
        wrapped_html = f"""
        <div style="
            border: 2px solid #667eea; 
            border-radius: 10px; 
            overflow: hidden; 
            background: white;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        ">
            <div style="
                background: linear-gradient(90deg, #667eea, #764ba2);
                color: white;
                padding: 10px 20px;
                font-weight: bold;
            ">
                🕸️ Network Graph | {len(simple_nodes)} Named Nodes | {added_edges} Colored Relationships
            </div>
            {html_content}
        </div>
        """
        
        # Display
        components.html(wrapped_html, height=700, scrolling=False)
        
        # Cleanup
        try:
            os.unlink(html_file)
        except:
            pass
        
        return True
        
    except Exception as e:
        st.error(f"❌ Graph rendering failed: {str(e)}")
        
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

# Main layout
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown("### 💬 Chat Interface")
    
    # Status
    status_colors = {"connected": "🟢", "disconnected": "🔴", "unknown": "⚪"}
    st.markdown(f'<div class="success-box"><strong>API Status:</strong> {status_colors.get(st.session_state.connection_status, "⚪")} {st.session_state.connection_status}</div>', unsafe_allow_html=True)
    
    # Prompt suggestions
    st.markdown("#### 💡 Quick Prompt Suggestions")
    st.info("Click any suggestion to populate the search field below:")
    
    # Initialize selected_prompt in session state if not exists
    if 'selected_prompt' not in st.session_state:
        st.session_state.selected_prompt = ""
    
    # Create suggestion buttons in a grid
    col_s1, col_s2 = st.columns(2)
    
    suggestions = [
        "Show me all Person nodes",
        "Display the database schema", 
        "Find all relationships",
        "Show network connections",
        "Count all nodes in database",
        "Show companies and employees",
        "Find isolated nodes",
        "Display node properties"
    ]
    
    for i, suggestion in enumerate(suggestions):
        if i % 2 == 0:
            with col_s1:
                if st.button(f"💭 {suggestion}", key=f"suggest_{i}", use_container_width=True, help="Click to use this prompt"):
                    st.session_state.selected_prompt = suggestion
                    st.rerun()
        else:
            with col_s2:
                if st.button(f"💭 {suggestion}", key=f"suggest_{i}", use_container_width=True, help="Click to use this prompt"):
                    st.session_state.selected_prompt = suggestion
                    st.rerun()
    
    st.divider()
    

    
    # Question input
    st.markdown("#### ✍️ Ask Your Question")
    st.info("💡 All queries are tracked with full details including Cypher queries and comprehensive analysis!")
    
    # Clear button for selected prompt
    if st.session_state.selected_prompt:
        col_clear1, col_clear2 = st.columns([3, 1])
        with col_clear1:
            st.success(f"📝 Selected: {st.session_state.selected_prompt}")
        with col_clear2:
            if st.button("🗑️ Clear", key="clear_prompt"):
                st.session_state.selected_prompt = ""
                st.rerun()
    
    with st.form("question_form", clear_on_submit=True):
        user_question = st.text_area(
            "Your question:",
            value=st.session_state.selected_prompt,
            placeholder="e.g., Show me all Person nodes, Display database schema, Find all relationships...",
            height=100
        )
        
        node_limit = st.selectbox(
            "Analysis scope:",
            [10, 25, 50, 75],
            index=1,
            help="Smaller scope = more detailed analysis and tracking"
        )
        
        submit_button = st.form_submit_button("🚀 Execute & Track Analysis", use_container_width=True)
    
    # Clear selected prompt after form submission
    if submit_button:
        st.session_state.selected_prompt = ""
    
    if submit_button and user_question.strip():
        # Clear selected prompt when submitting
        st.session_state.selected_prompt = ""
        
        result = call_agent_api(user_question.strip(), node_limit)
        
        if result:
            # Generate detailed analysis
            detailed_analysis = None
            if result.get("graph_data"):
                detailed_analysis = generate_detailed_analysis(
                    result["graph_data"], 
                    user_question.strip(), 
                    result.get("answer", "")
                )
                st.session_state.detailed_analysis = detailed_analysis
            
            # Create comprehensive history entry
            history_entry = create_detailed_history_entry(user_question.strip(), result, detailed_analysis)
            st.session_state.conversation_history.append(history_entry)
            
            if result.get("graph_data"):
                st.session_state.graph_data = result["graph_data"]
                st.success("✅ Query executed and tracked in detailed history!")
            
            st.session_state.last_response = result
            st.rerun()
    
    st.divider()
    
    # Test data
    if st.button("🧪 Load Test Dataset", use_container_width=True):
        test_data = {
            "nodes": [
                {"id": "person1", "labels": ["Person"], "properties": {"name": "Alice Johnson", "age": 30, "department": "Engineering", "role": "Senior Developer", "experience": 8, "salary": 120000}},
                {"id": "person2", "labels": ["Person"], "properties": {"name": "Bob Smith", "age": 25, "department": "Marketing", "role": "Designer", "experience": 3, "salary": 85000}},
                {"id": "person3", "labels": ["Person"], "properties": {"name": "Carol Brown", "age": 35, "department": "Engineering", "role": "Manager", "experience": 12, "salary": 150000}},
                {"id": "person4", "labels": ["Person"], "properties": {"name": "David Wilson", "age": 28, "department": "Sales", "role": "Account Executive", "experience": 5, "salary": 95000}},
                {"id": "company1", "labels": ["Company"], "properties": {"name": "TechCorp Inc.", "industry": "Technology", "employees": 500, "founded": 2010, "revenue": "50M"}},
                {"id": "location1", "labels": ["Location"], "properties": {"name": "New York", "country": "USA", "population": 8000000, "timezone": "EST"}},
                {"id": "project1", "labels": ["Project"], "properties": {"name": "AI Innovation", "status": "Active", "budget": 2000000, "duration": "12 months"}}
            ],
            "relationships": [
                {"startNode": "person1", "endNode": "person2", "type": "KNOWS", "properties": {"since": "2020", "relationship": "colleague"}},
                {"startNode": "person2", "endNode": "person4", "type": "FRIEND_OF", "properties": {"since": "2019", "closeness": "high"}},  
                {"startNode": "person3", "endNode": "person1", "type": "MANAGES", "properties": {"since": "2021", "team": "Backend"}},
                {"startNode": "person1", "endNode": "company1", "type": "WORKS_FOR", "properties": {"position": "Senior Developer", "salary": 120000}},
                {"startNode": "person2", "endNode": "company1", "type": "WORKS_FOR", "properties": {"position": "Designer", "salary": 85000}},
                {"startNode": "person3", "endNode": "company1", "type": "WORKS_FOR", "properties": {"position": "Engineering Manager", "salary": 150000}},
                {"startNode": "person4", "endNode": "company1", "type": "WORKS_FOR", "properties": {"position": "Sales Executive", "salary": 95000}},
                {"startNode": "company1", "endNode": "location1", "type": "LOCATED_IN", "properties": {"headquarters": True, "offices": 3}},
                {"startNode": "person1", "endNode": "project1", "type": "ASSIGNED_TO", "properties": {"role": "Technical Lead", "allocation": "100%"}},
                {"startNode": "person3", "endNode": "project1", "type": "MANAGES", "properties": {"responsibility": "Budget and Timeline"}}
            ]
        }
        
        # Debug: Verify test data names
        st.info("🔍 **Debug:** Test data loaded with the following names:")
        for i, node in enumerate(test_data["nodes"]):
            name = node["properties"].get("name", "NO NAME")
            st.write(f"• Node {i}: **{name}** (Type: {node['labels'][0]})")
        
        st.session_state.graph_data = test_data
        
        # Generate analysis for test data
        detailed_analysis = generate_detailed_analysis(test_data, "Test dataset analysis", "Test data loaded")
        st.session_state.detailed_analysis = detailed_analysis
        
        # Create mock result for history
        mock_result = {
            "answer": "Test dataset loaded with comprehensive analysis",
            "graph_data": test_data,
            "tool": "test_data_loader",
            "query": "MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 50",
            "trace": "Test data loading operation",
            "execution_time_ms": 15.5,
            "success": True
        }
        
        # Add to detailed history
        history_entry = create_detailed_history_entry("Load test dataset", mock_result, detailed_analysis)
        st.session_state.conversation_history.append(history_entry)
        
        st.success("✅ Test dataset loaded with complete tracking!")
        st.rerun()
    
    # History Section
    st.markdown("#### 📊 Query History")
    st.markdown(f"**{len(st.session_state.conversation_history)} queries tracked** with complete details")
    
    if st.session_state.conversation_history:
        # History controls
        col_hist1, col_hist2 = st.columns(2)
        with col_hist1:
            show_count = st.selectbox("Show last:", [3, 5, 10, "All"], index=0)
        with col_hist2:
            if st.button("📥 Export History", use_container_width=True):
                # Create downloadable history
                history_json = json.dumps(st.session_state.conversation_history, indent=2, default=str)
                st.download_button(
                    label="💾 Download JSON",
                    data=history_json,
                    file_name=f"neo4j_analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Display history entries
        display_count = len(st.session_state.conversation_history) if show_count == "All" else int(show_count)
        displayed_entries = st.session_state.conversation_history[-display_count:]
        
        for i, entry in enumerate(reversed(displayed_entries)):
            display_detailed_history_entry(entry, i)
        
        # History statistics - using checkbox instead of nested expander
        stats_key = f"history_stats_{len(st.session_state.conversation_history)}"
        show_stats = st.checkbox("📈 Show History Statistics", key=stats_key)
        
        if show_stats:
            total_queries = len(st.session_state.conversation_history)
            total_nodes_analyzed = sum(entry.get("nodes_count", 0) for entry in st.session_state.conversation_history)
            total_relationships_analyzed = sum(entry.get("relationships_count", 0) for entry in st.session_state.conversation_history)
            avg_execution_time = sum(entry.get("execution_time_ms", 0) for entry in st.session_state.conversation_history) / max(total_queries, 1)
            
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Total Queries", total_queries)
            with col_s2:
                st.metric("Nodes Analyzed", total_nodes_analyzed)
            with col_s3:
                st.metric("Relationships", total_relationships_analyzed)
            with col_s4:
                st.metric("Avg Time (ms)", f"{avg_execution_time:.1f}")
            
            # Show most used tools
            tool_usage = {}
            for entry in st.session_state.conversation_history:
                tool = entry.get("tool_used", "unknown")
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
            
            if tool_usage:
                st.write("**Most Used Tools:**")
                for tool, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"• {tool}: {count} times")
    else:
        st.info("💡 No analysis history yet. Run some queries to build your detailed tracking history!")
    
    if st.button("🗑️ Clear All History & Data", use_container_width=True):
        for key in ["conversation_history", "graph_data", "last_response", "detailed_analysis"]:
            st.session_state[key] = [] if key == "conversation_history" else None
        st.success("🧹 All history and data cleared!")
        st.rerun()

with col2:
    st.markdown("### 🎨 Analysis & Visualization")
    
    # Show comprehensive detailed response
    if st.session_state.detailed_analysis:
        analysis = st.session_state.detailed_analysis
        
        st.markdown("#### 🤖 AI Analysis")
        
        # Main summary
        st.markdown(f'<div class="detailed-response-box">{analysis["summary"]}</div>', unsafe_allow_html=True)
        
        # Insights section
        if analysis.get("insights"):
            st.markdown('<div class="response-section"><h4>🔍 Key Insights</h4>', unsafe_allow_html=True)
            for insight in analysis["insights"]:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendations section
        if analysis.get("recommendations"):
            st.markdown('<div class="response-section"><h4>💡 Smart Recommendations</h4>', unsafe_allow_html=True)
            for rec in analysis["recommendations"]:
                st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Technical details - using checkbox instead of nested expander
        if analysis.get("raw_stats"):
            tech_analysis_key = f"tech_analysis_{hash(str(analysis['raw_stats'])) % 1000}"
            show_tech_analysis = st.checkbox("📊 Show Detailed Technical Analysis", key=tech_analysis_key)
            
            if show_tech_analysis:
                stats = analysis["raw_stats"]
                st.write(f"**Network Metrics:**")
                st.write(f"• Total Nodes: {stats['total_nodes']}")
                st.write(f"• Total Relationships: {stats['total_relationships']}")
                st.write(f"• Node Type Diversity: {stats['node_types']} different types")
                st.write(f"• Relationship Type Diversity: {stats['relationship_types']} different types")
                st.write(f"• Network Connectivity: {stats['connectivity']:.2f} connections per node")
                
                if analysis.get("node_analysis"):
                    st.write(f"**Node Analysis:**")
                    node_analysis = analysis["node_analysis"]
                    if node_analysis.get("node_types"):
                        st.write("Node Type Distribution:")
                        for node_type, count in node_analysis["node_types"].items():
                            st.write(f"  • {node_type}: {count} nodes")
                
                if analysis.get("relationship_analysis"):
                    st.write(f"**Relationship Analysis:**")
                    rel_analysis = analysis["relationship_analysis"]
                    if rel_analysis.get("relationship_types"):
                        st.write("Relationship Type Distribution:")
                        for rel_type, count in rel_analysis["relationship_types"].items():
                            st.write(f"  • {rel_type}: {count} connections")
    
    elif st.session_state.last_response:
        # Fallback to basic response if no detailed analysis
        answer = st.session_state.last_response.get("answer", "")
        if answer:
            st.markdown("#### 🤖 AI Response")
            clean_answer = answer.replace("**", "").replace("#", "").strip()
            st.markdown(f'<div class="detailed-response-box">{clean_answer}</div>', unsafe_allow_html=True)
    
    # Graph section
    if st.session_state.graph_data:
        nodes = st.session_state.graph_data.get("nodes", [])
        relationships = st.session_state.graph_data.get("relationships", [])
        
        # Enhanced stats with more details
        col2_1, col2_2, col2_3 = st.columns(3)
        with col2_1:
            st.markdown(f'<div class="metric-container"><h2>{len(nodes)}</h2><p>Named Nodes</p></div>', unsafe_allow_html=True)
        with col2_2:
            st.markdown(f'<div class="metric-container"><h2>{len(relationships)}</h2><p>Relationships</p></div>', unsafe_allow_html=True)
        with col2_3:
            connections = len(relationships) / max(len(nodes), 1)
            st.markdown(f'<div class="metric-container"><h2>{connections:.1f}</h2><p>Connectivity</p></div>', unsafe_allow_html=True)
        
        # Legend
        if nodes or relationships:
            legend = create_simple_legend(nodes, relationships)
            st.markdown(legend, unsafe_allow_html=True)
        
        # Render graph
        st.markdown("#### 🎨 Interactive Network Visualization")
        success = render_working_graph(st.session_state.graph_data)
        
        if success:
            if len(relationships) > 0:
                st.markdown(f'<div class="success-box">🎉 <strong>Success!</strong> Interactive graph displays {len(nodes)} named nodes connected by {len(relationships)} colored relationship lines! All details tracked in analysis history.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">ℹ️ Graph shows isolated nodes - no relationships found in current dataset</div>', unsafe_allow_html=True)
            
            # Enhanced controls
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("🔄 Refresh Analysis", use_container_width=True):
                    if st.session_state.graph_data:
                        detailed_analysis = generate_detailed_analysis(
                            st.session_state.graph_data, 
                            "Manual refresh analysis", 
                            "Analysis refreshed"
                        )
                        st.session_state.detailed_analysis = detailed_analysis
                    st.rerun()
            with col_b:
                if st.button("🌐 Full Network Study", use_container_width=True):
                    result = call_agent_api("Conduct comprehensive network analysis with detailed insights", node_limit=50)
                    if result and result.get("graph_data"):
                        st.session_state.graph_data = result["graph_data"]
                        detailed_analysis = generate_detailed_analysis(
                            result["graph_data"], 
                            "Full network study", 
                            result.get("answer", "")
                        )
                        st.session_state.detailed_analysis = detailed_analysis
                        
                        # Add to history
                        history_entry = create_detailed_history_entry("Full network study", result, detailed_analysis)
                        st.session_state.conversation_history.append(history_entry)
                        st.rerun()
            with col_c:
                if st.button("📊 Export Analysis", use_container_width=True):
                    if st.session_state.detailed_analysis:
                        analysis_json = json.dumps(st.session_state.detailed_analysis, indent=2, default=str)
                        st.download_button(
                            label="📥 Download Analysis",
                            data=analysis_json,
                            file_name=f"neo4j_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
        else:
            st.error("❌ Graph rendering failed. Check the debug information above.")
    
    else:
        # Welcome screen with fixed HTML
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin: 2rem 0;">
            <h2>🎯 Neo4j Graph Explorer</h2>
            <p><strong>Complete Query Tracking & Analysis</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #a8edea, #fed6e3); color: #2c3e50; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
            <h3 style="text-align: center; margin-top: 0;">📊 What's Tracked in History:</h3>
            <div style="text-align: left;">
                <p>🔧 <strong>Cypher Queries</strong> - Exact queries executed</p>
                <p>📊 <strong>Detailed Metrics</strong> - Nodes, relationships, connectivity</p>
                <p>🧠 <strong>AI Analysis</strong> - Comprehensive insights and recommendations</p>
                <p>⚡ <strong>Performance Data</strong> - Execution times and success rates</p>
                <p>🎯 <strong>Key Insights</strong> - Smart observations about patterns</p>
                <p>🔄 <strong>Repeatable Actions</strong> - Re-run or load any previous query</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #ffecd2, #fcb69f); color: #8b4513; border-radius: 10px; margin: 1rem 0;">
            <p><em>Click "Load Test Dataset" to see complete tracking in action!</em></p>
        </div>
        """, unsafe_allow_html=True)

# Enhanced footer
st.markdown("---")
st.markdown("""
<div style="
    text-align: center; 
    color: #6c757d; 
    padding: 1rem;
    background: linear-gradient(90deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
    border-radius: 10px;
    margin-top: 2rem;
">
    <h4 style="margin: 0; background: linear-gradient(90deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        🚀 Neo4j Graph Explorer
    </h4>
    <p style="margin: 0.5rem 0;">🔧 Cypher Queries • 📊 Detailed History • 🧠 AI Analysis • 💾 Export Capabilities</p>
</div>
""", unsafe_allow_html=True)
