# streamlit_mcp_neo4j_client.py
"""
Streamlit client for Enhanced Neo4j MCP Server
User selects tools and provides prompts to execute against the MCP server
"""

import streamlit as st
import asyncio
import json
import subprocess
import threading
import time
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="🧠 Neo4j MCP Tool Executor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .tool-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .tool-selected {
        background-color: #e3f2fd;
        border: 2px solid #2196f3;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .result-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .result-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .prompt-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MCP Tools Configuration
# ============================================================================

MCP_TOOLS = {
    "intelligent_neo4j_query": {
        "name": "🧠 Intelligent Neo4j Query",
        "description": "AI-powered natural language query processing with intent analysis and business insights",
        "params": {"user_input": "text"},
        "example": "Show me the top 10 most popular fitness apps with their ratings",
        "category": "AI-Powered"
    },
    "get_neo4j_schema": {
        "name": "📊 Get Database Schema", 
        "description": "Retrieve complete database schema including nodes, relationships, and properties",
        "params": {},
        "example": "Get the database schema",
        "category": "Schema"
    },
    "read_neo4j_cypher": {
        "name": "🔍 Execute Read Query",
        "description": "Execute read-only Cypher queries against the Neo4j database",
        "params": {"query": "cypher", "params": "json"},
        "example": "MATCH (a:Apps) WHERE a.rating > 4.0 RETURN a.name, a.rating ORDER BY a.rating DESC LIMIT 10",
        "category": "Direct Query"
    },
    "write_neo4j_cypher": {
        "name": "✏️ Execute Write Query",
        "description": "Execute write Cypher queries (CREATE, MERGE, DELETE, SET, REMOVE)",
        "params": {"query": "cypher", "params": "json"},
        "example": "CREATE (a:App {name: 'MyApp', rating: 4.5, category: 'Fitness'})",
        "category": "Direct Query"
    },
    "analyze_query_performance": {
        "name": "⚡ Query Performance Analysis",
        "description": "AI-powered analysis of Cypher query performance with optimization suggestions",
        "params": {"query": "cypher", "suggest_improvements": "boolean"},
        "example": "MATCH (a:Apps) WHERE a.rating > 4.0 RETURN a",
        "category": "Optimization"
    },
    "system_health_check": {
        "name": "🏥 System Health Check",
        "description": "Comprehensive system status including database connectivity and component health",
        "params": {},
        "example": "Check system health",
        "category": "Monitoring"
    }
}

# ============================================================================
# MCP Server Manager
# ============================================================================

class MCPServerManager:
    """Manages the FastMCP Neo4j server process"""
    
    def __init__(self):
        self.process = None
        self.is_running = False
        
    def start_server(self):
        """Start the MCP server in background"""
        if self.is_running:
            return True
            
        try:
            # Check if server file exists
            if not Path("fastmcp_neo4j_langgraph.py").exists():
                st.error("❌ FastMCP server file not found: fastmcp_neo4j_langgraph.py")
                return False
            
            # Set environment variables
            env = os.environ.copy()
            env.update(NEO4J_CONFIG)
            
            # Start server process
            self.process = subprocess.Popen([
                "python", "fastmcp_neo4j_langgraph.py",
                "--transport", "stdio"
            ], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.is_running = True
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to start MCP server: {e}")
            return False
    
    def stop_server(self):
        """Stop the MCP server"""
        if self.process:
            self.process.terminate()
            self.process = None
            self.is_running = False

# ============================================================================
# Tool Executor
# ============================================================================

async def execute_mcp_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Execute MCP tool with given parameters"""
    try:
        # For this demo, we'll simulate the MCP call
        # In production, you'd use actual MCP client here
        
        # Simulate processing time
        await asyncio.sleep(1)
        
        # Mock responses based on tool type
        if tool_name == "intelligent_neo4j_query":
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "request": {
                    "user_input": parameters.get("user_input", ""),
                    "intent": "READ_QUERY",
                    "confidence": 0.95,
                    "analysis": "User wants to retrieve popular apps with high ratings"
                },
                "execution": {
                    "tool_used": "read_neo4j_cypher",
                    "cypher_query": "MATCH (a:Apps) WHERE a.rating > 4.0 RETURN a.name, a.rating, a.install_count ORDER BY a.install_count DESC LIMIT 10"
                },
                "raw_result": [
                    {"a.name": "FitnessTracker Pro", "a.rating": 4.8, "a.install_count": 15000},
                    {"a.name": "RunMaster", "a.rating": 4.7, "a.install_count": 12000},
                    {"a.name": "CycleCompanion", "a.rating": 4.6, "a.install_count": 10000}
                ],
                "insights": {
                    "interpretation": "The top fitness apps show excellent user satisfaction with ratings above 4.5. FitnessTracker Pro leads with 15,000 installs and 4.8 rating, indicating strong market position.",
                    "business_value": "Focus on partnerships with top-rated apps for maximum user engagement."
                }
            }
        
        elif tool_name == "get_neo4j_schema":
            return {
                "status": "success",
                "schema": {
                    "Apps": {
                        "type": "node",
                        "count": 1250,
                        "properties": {
                            "name": {"type": "string", "indexed": True},
                            "rating": {"type": "float"},
                            "install_count": {"type": "integer"},
                            "category": {"type": "string"}
                        },
                        "relationships": {
                            "BELONGS_TO": {"direction": "out", "labels": ["Categories"]},
                            "COMPATIBLE_WITH": {"direction": "out", "labels": ["Devices"]}
                        }
                    },
                    "Devices": {
                        "type": "node", 
                        "count": 45,
                        "properties": {
                            "name": {"type": "string", "indexed": True},
                            "device_type": {"type": "string"},
                            "manufacturer": {"type": "string"}
                        }
                    }
                }
            }
            
        elif tool_name == "read_neo4j_cypher":
            return {
                "status": "success",
                "query": parameters.get("query", ""),
                "result": [
                    {"a.name": "App1", "a.rating": 4.5},
                    {"a.name": "App2", "a.rating": 4.3},
                    {"a.name": "App3", "a.rating": 4.7}
                ]
            }
            
        elif tool_name == "system_health_check":
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "neo4j_database": {"status": "connected", "uri": NEO4J_CONFIG["NEO4J_URI"]},
                    "langgraph_workflow": {"status": "active"},
                    "cortex_llm": {"status": "connected"},
                    "fastmcp_server": {"status": "running", "tools_count": 6}
                }
            }
            
        else:
            return {
                "status": "success",
                "tool": tool_name,
                "parameters": parameters,
                "result": f"Tool {tool_name} executed successfully"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "tool": tool_name
        }

# ============================================================================
# UI Components
# ============================================================================

def render_tool_selector():
    """Render tool selection interface"""
    st.markdown("### 🛠️ Select MCP Tool")
    
    # Group tools by category
    categories = {}
    for tool_id, tool_info in MCP_TOOLS.items():
        category = tool_info["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append((tool_id, tool_info))
    
    # Create tabs for categories
    category_tabs = st.tabs(list(categories.keys()))
    
    selected_tool = None
    
    for i, (category, tools) in enumerate(categories.items()):
        with category_tabs[i]:
            for tool_id, tool_info in tools:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**{tool_info['name']}**")
                        st.markdown(f"*{tool_info['description']}*")
                        if tool_info.get('example'):
                            st.code(tool_info['example'], language='text' if 'cypher' not in tool_info.get('params', {}).values() else 'cypher')
                    
                    with col2:
                        if st.button(f"Select", key=f"select_{tool_id}"):
                            st.session_state.selected_tool = tool_id
                            st.rerun()
    
    return st.session_state.get('selected_tool')

def render_parameter_inputs(tool_id: str):
    """Render parameter input interface for selected tool"""
    if not tool_id or tool_id not in MCP_TOOLS:
        return {}
    
    tool_info = MCP_TOOLS[tool_id]
    params = tool_info.get("params", {})
    
    st.markdown(f"### 📝 Configure: {tool_info['name']}")
    
    user_params = {}
    
    if not params:
        st.info("This tool doesn't require any parameters.")
        return user_params
    
    for param_name, param_type in params.items():
        if param_type == "text":
            user_params[param_name] = st.text_area(
                f"{param_name.replace('_', ' ').title()}:",
                placeholder=f"Enter {param_name}...",
                height=100
            )
        elif param_type == "cypher":
            user_params[param_name] = st.text_area(
                f"{param_name.replace('_', ' ').title()}:",
                placeholder="Enter Cypher query...",
                height=150
            )
        elif param_type == "json":
            json_input = st.text_area(
                f"{param_name.replace('_', ' ').title()} (JSON):",
                placeholder='{"key": "value"}',
                height=100
            )
            if json_input.strip():
                try:
                    user_params[param_name] = json.loads(json_input)
                except json.JSONDecodeError:
                    st.error(f"Invalid JSON format for {param_name}")
            else:
                user_params[param_name] = None
        elif param_type == "boolean":
            user_params[param_name] = st.checkbox(
                f"{param_name.replace('_', ' ').title()}",
                value=True
            )
    
    return user_params

def render_results(result: Dict[str, Any]):
    """Render execution results"""
    st.markdown("### 📊 Execution Results")
    
    if result.get("status") == "error":
        st.markdown('<div class="result-error">', unsafe_allow_html=True)
        st.error(f"❌ **Error:** {result.get('error', 'Unknown error')}")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Success results
    st.markdown('<div class="result-success">', unsafe_allow_html=True)
    st.success("✅ **Tool executed successfully!**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle different result types
    if "intelligent_neo4j_query" in str(result):
        render_intelligent_query_results(result)
    elif "schema" in result:
        render_schema_results(result)
    elif "components" in result:
        render_health_check_results(result)
    else:
        render_generic_results(result)

def render_intelligent_query_results(result: Dict[str, Any]):
    """Render intelligent query results with insights"""
    
    # Request Analysis
    if "request" in result:
        st.markdown("#### 🧠 AI Analysis")
        request = result["request"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Intent", request.get("intent", "N/A"))
        with col2:
            st.metric("Confidence", f"{request.get('confidence', 0):.2%}")
        with col3:
            st.metric("Status", "Analyzed")
        
        st.info(f"**Analysis:** {request.get('analysis', 'No analysis available')}")
    
    # Execution Details
    if "execution" in result:
        st.markdown("#### ⚙️ Query Execution")
        execution = result["execution"]
        
        st.markdown("**Generated Cypher Query:**")
        st.code(execution.get("cypher_query", "No query"), language="cypher")
        
        st.markdown(f"**Tool Used:** `{execution.get('tool_used', 'Unknown')}`")
    
    # Raw Results
    if "raw_result" in result:
        st.markdown("#### 📋 Query Results")
        raw_result = result["raw_result"]
        
        if isinstance(raw_result, list) and len(raw_result) > 0:
            # Convert to DataFrame for better display
            df = pd.DataFrame(raw_result)
            st.dataframe(df, use_container_width=True)
            
            # Create visualization if numeric columns exist
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_columns) > 0:
                st.markdown("#### 📈 Data Visualization")
                
                # Create bar chart if possible
                if len(df) <= 20:  # Only for reasonable number of rows
                    if 'name' in df.columns or any('name' in col.lower() for col in df.columns):
                        name_col = next((col for col in df.columns if 'name' in col.lower()), df.columns[0])
                        if len(numeric_columns) > 0:
                            fig = px.bar(df, x=name_col, y=numeric_columns[0], 
                                       title=f"{numeric_columns[0]} by {name_col}")
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(raw_result)
    
    # Business Insights
    if "insights" in result:
        st.markdown("#### 💡 Business Insights")
        insights = result["insights"]
        
        st.markdown('<div class="prompt-box">', unsafe_allow_html=True)
        st.markdown(f"**Interpretation:** {insights.get('interpretation', 'No interpretation available')}")
        st.markdown(f"**Business Value:** {insights.get('business_value', 'No business value identified')}")
        st.markdown('</div>', unsafe_allow_html=True)

def render_schema_results(result: Dict[str, Any]):
    """Render database schema results"""
    st.markdown("#### 🗄️ Database Schema")
    
    schema = result.get("schema", {})
    
    # Schema overview
    node_count = len([k for k, v in schema.items() if v.get("type") == "node"])
    total_records = sum(v.get("count", 0) for v in schema.values() if isinstance(v.get("count"), int))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Node Types", node_count)
    with col2:
        st.metric("Total Records", f"{total_records:,}")
    with col3:
        st.metric("Schema Objects", len(schema))
    
    # Detailed schema
    for name, info in schema.items():
        with st.expander(f"📦 {name} ({info.get('type', 'unknown')})"):
            if info.get("count"):
                st.write(f"**Count:** {info['count']:,} records")
            
            if "properties" in info:
                st.write("**Properties:**")
                props_df = pd.DataFrame.from_dict(info["properties"], orient="index")
                st.dataframe(props_df, use_container_width=True)
            
            if "relationships" in info:
                st.write("**Relationships:**")
                for rel_name, rel_info in info["relationships"].items():
                    st.write(f"- `{rel_name}` → {rel_info.get('labels', [])}")

def render_health_check_results(result: Dict[str, Any]):
    """Render system health check results"""
    st.markdown("#### 🏥 System Health Status")
    
    overall_status = result.get("status", "unknown")
    status_color = "🟢" if overall_status == "healthy" else "🔴"
    
    st.markdown(f"**Overall Status:** {status_color} {overall_status.upper()}")
    
    if "components" in result:
        components = result["components"]
        
        for comp_name, comp_info in components.items():
            with st.expander(f"🔧 {comp_name.replace('_', ' ').title()}"):
                comp_status = comp_info.get("status", "unknown")
                comp_color = "🟢" if comp_status in ["connected", "active", "running"] else "🔴"
                
                st.markdown(f"**Status:** {comp_color} {comp_status}")
                
                # Show additional info
                for key, value in comp_info.items():
                    if key != "status":
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")

def render_generic_results(result: Dict[str, Any]):
    """Render generic results as JSON"""
    st.markdown("#### 📄 Raw Results")
    st.json(result)

# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Neo4j MCP Tool Executor</h1>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Graph Database Interface**")
    
    # Initialize session state
    if "execution_history" not in st.session_state:
        st.session_state.execution_history = []
    if "server_manager" not in st.session_state:
        st.session_state.server_manager = MCPServerManager()
    
    # Sidebar - Server Status & Configuration
    with st.sidebar:
        st.markdown("### 🔧 Server Configuration")
        
        # Server status
        server_status = st.session_state.server_manager.is_running
        status_color = "🟢" if server_status else "🔴"
        st.markdown(f"**Server Status:** {status_color} {'Running' if server_status else 'Stopped'}")
        
        # Start/Stop server
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Start Server"):
                with st.spinner("Starting MCP server..."):
                    if st.session_state.server_manager.start_server():
                        st.success("✅ Server started!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to start server")
        
        with col2:
            if st.button("🛑 Stop Server"):
                st.session_state.server_manager.stop_server()
                st.success("✅ Server stopped!")
                st.rerun()
        
        # Configuration display
        st.markdown("### ⚙️ Neo4j Configuration")
        st.code(f"""
URI: {NEO4J_CONFIG['NEO4J_URI']}
Database: {NEO4J_CONFIG['NEO4J_DATABASE']}
Username: {NEO4J_CONFIG['NEO4J_USERNAME']}
Namespace: {NEO4J_CONFIG['NEO4J_NAMESPACE']}
        """)
        
        # Clear history
        if st.button("🗑️ Clear History"):
            st.session_state.execution_history = []
            st.rerun()
    
    # Main content area
    if not server_status:
        st.warning("⚠️ Please start the MCP server to use the tools.")
        st.info("Click the '🚀 Start Server' button in the sidebar to begin.")
        return
    
    # Tool selection
    selected_tool = render_tool_selector()
    
    if selected_tool:
        st.markdown("---")
        
        # Parameter configuration
        parameters = render_parameter_inputs(selected_tool)
        
        # Execute button
        st.markdown("### 🚀 Execute Tool")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("▶️ Execute Tool", type="primary", use_container_width=True):
                # Validate required parameters
                tool_info = MCP_TOOLS[selected_tool]
                required_params = tool_info.get("params", {})
                
                missing_params = []
                for param_name, param_type in required_params.items():
                    if param_type != "boolean" and not parameters.get(param_name):
                        missing_params.append(param_name)
                
                if missing_params:
                    st.error(f"❌ Missing required parameters: {', '.join(missing_params)}")
                else:
                    # Execute the tool
                    with st.spinner(f"Executing {tool_info['name']}..."):
                        try:
                            # Run async function
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            result = loop.run_until_complete(
                                execute_mcp_tool(selected_tool, parameters)
                            )
                            loop.close()
                            
                            # Store in history
                            execution_record = {
                                "timestamp": datetime.now(),
                                "tool": selected_tool,
                                "tool_name": tool_info["name"],
                                "parameters": parameters,
                                "result": result
                            }
                            st.session_state.execution_history.append(execution_record)
                            
                            # Display results
                            st.markdown("---")
                            render_results(result)
                            
                        except Exception as e:
                            st.error(f"❌ Execution failed: {str(e)}")
        
        with col2:
            if st.button("📋 Use Example", use_container_width=True):
                # Load example parameters
                example = tool_info.get("example", "")
                if example and "params" in tool_info:
                    param_names = list(tool_info["params"].keys())
                    if len(param_names) == 1:
                        st.session_state[f"param_{param_names[0]}"] = example
                        st.rerun()
        
        with col3:
            if st.button("🔄 Reset Form", use_container_width=True):
                # Clear form
                for param_name in tool_info.get("params", {}):
                    if f"param_{param_name}" in st.session_state:
                        del st.session_state[f"param_{param_name}"]
                st.rerun()
    
    # Execution history
    if st.session_state.execution_history:
        st.markdown("---")
        st.markdown("### 📚 Execution History")
        
        for i, record in enumerate(reversed(st.session_state.execution_history[-5:])):  # Show last 5
            with st.expander(f"🕒 {record['timestamp'].strftime('%H:%M:%S')} - {record['tool_name']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Parameters:**")
                    st.json(record["parameters"])
                
                with col2:
                    st.markdown("**Result Status:**")
                    status = record["result"].get("status", "unknown")
                    st.write(f"Status: {'✅' if status == 'success' else '❌'} {status}")
                
                if st.button(f"🔄 Re-run", key=f"rerun_{i}"):
                    st.session_state.selected_tool = record["tool"]
                    st.rerun()

if __name__ == "__main__":
    main()
