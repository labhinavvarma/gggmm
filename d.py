# standalone_streamlit_neo4j_ui.py
"""
Standalone Streamlit UI for Neo4j MCP Server with LangGraph Intelligence
All configurations hard-coded - just run with: streamlit run standalone_streamlit_neo4j_ui.py
"""

import streamlit as st
import asyncio
import json
import subprocess
import threading
import time
import os
import sys
import uuid
import requests
import urllib3
from datetime import datetime
from typing import Dict, Any, Optional, List, TypedDict, Literal
from pathlib import Path

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================================
# HARD-CODED CONFIGURATION (Edit these values as needed)
# ============================================================================

# Neo4j Configuration
NEO4J_URI = "neo4j://10.189.116.237:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "Vkg5d$F!pLq2@9vRwE="
NEO4J_DATABASE = "connectiq"
NEO4J_NAMESPACE = "connectiq"

# Cortex LLM Configuration
CORTEX_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
CORTEX_API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
CORTEX_APP_ID = "edadip"
CORTEX_APLCTN_CD = "edagnai"
CORTEX_MODEL = "llama3.1-70b"

# MCP Server Configuration
MCP_SERVER_FILE = "standalone_neo4j_mcp_server.py"
MCP_PORT = 8000

# ============================================================================
# AUTO-INSTALL DEPENDENCIES
# ============================================================================

def install_dependencies():
    """Install required packages if not available"""
    required_packages = [
        "streamlit", "pandas", "plotly", "requests", "asyncio-mqtt"
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            st.info(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install dependencies
try:
    install_dependencies()
except Exception as e:
    st.error(f"Failed to install dependencies: {e}")

# Import after installation
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🧠 Neo4j AI Assistant",
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
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .tool-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .tool-card:hover {
        background-color: #e3f2fd;
        border-color: #2196f3;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
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
    .insight-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-green { background-color: #28a745; }
    .status-red { background-color: #dc3545; }
    .status-yellow { background-color: #ffc107; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MCP TOOLS CONFIGURATION
# ============================================================================

MCP_TOOLS = {
    "intelligent_neo4j_query": {
        "name": "🧠 AI-Powered Query",
        "description": "Natural language processing with intent analysis, query generation, and business insights",
        "params": {"user_input": "text"},
        "example": "Show me the top 10 most popular fitness apps with their ratings and device compatibility",
        "category": "🤖 AI-Powered",
        "color": "#4CAF50"
    },
    "get_neo4j_schema": {
        "name": "📊 Database Schema", 
        "description": "Complete database structure including nodes, relationships, properties, and indexes",
        "params": {},
        "example": "Get the complete database schema",
        "category": "📋 Schema",
        "color": "#2196F3"
    },
    "read_neo4j_cypher": {
        "name": "🔍 Read Query",
        "description": "Execute read-only Cypher queries with results formatting",
        "params": {"query": "cypher", "params": "json"},
        "example": "MATCH (a:Apps)-[:BELONGS_TO]->(c:Categories {name: 'Fitness'}) RETURN a.name, a.rating, a.install_count ORDER BY a.rating DESC LIMIT 10",
        "category": "🔧 Direct Query",
        "color": "#FF9800"
    },
    "write_neo4j_cypher": {
        "name": "✏️ Write Query",
        "description": "Execute write operations (CREATE, MERGE, DELETE, SET, REMOVE)",
        "params": {"query": "cypher", "params": "json"},
        "example": "CREATE (a:App {name: 'MyNewApp', rating: 4.5, category: 'Fitness', release_date: date()})",
        "category": "🔧 Direct Query",
        "color": "#F44336"
    },
    "system_health_check": {
        "name": "🏥 System Health",
        "description": "Comprehensive system status, connectivity tests, and performance metrics",
        "params": {},
        "example": "Check system health and connectivity",
        "category": "🔧 Monitoring",
        "color": "#9C27B0"
    }
}

# ============================================================================
# MCP SERVER MANAGER
# ============================================================================

class MCPServerManager:
    """Manages the MCP server process"""
    
    def __init__(self):
        self.process = None
        self.is_running = False
        
    def start_server(self):
        """Start the MCP server"""
        if self.is_running:
            return True
            
        try:
            # Check if server file exists
            if not Path(MCP_SERVER_FILE).exists():
                st.error(f"❌ MCP server file not found: {MCP_SERVER_FILE}")
                st.info("💡 Make sure 'standalone_neo4j_mcp_server.py' is in the same directory")
                return False
            
            # Start server process
            self.process = subprocess.Popen([
                sys.executable, MCP_SERVER_FILE
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
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
# MCP CLIENT FOR COMMUNICATION
# ============================================================================

class MCPClient:
    """Simple MCP client for communication with server"""
    
    def __init__(self, server_manager):
        self.server_manager = server_manager
        
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool via JSON-RPC over STDIO"""
        if not self.server_manager.is_running or not self.server_manager.process:
            return {"status": "error", "error": "MCP server not running"}
        
        try:
            # Prepare JSON-RPC request
            request = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tools/call",
                "params": {
                    "name": f"{NEO4J_NAMESPACE}-{tool_name}" if NEO4J_NAMESPACE else tool_name,
                    "arguments": parameters
                }
            }
            
            # Send request
            request_json = json.dumps(request) + "\n"
            self.server_manager.process.stdin.write(request_json.encode())
            self.server_manager.process.stdin.flush()
            
            # Read response (with timeout)
            response_line = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.server_manager.process.stdout.readline
                ), timeout=30.0
            )
            
            if response_line:
                response = json.loads(response_line.decode().strip())
                
                if "result" in response:
                    result = response["result"]
                    if "content" in result and result["content"]:
                        content = result["content"][0].get("text", "")
                        try:
                            return json.loads(content)
                        except json.JSONDecodeError:
                            return {"status": "success", "result": content}
                    else:
                        return {"status": "success", "result": "No content"}
                elif "error" in response:
                    return {"status": "error", "error": response["error"].get("message", "Unknown error")}
                else:
                    return {"status": "error", "error": "Invalid response format"}
            else:
                return {"status": "error", "error": "No response from server"}
                
        except asyncio.TimeoutError:
            return {"status": "error", "error": "Request timeout"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================

def render_status_indicator(status: bool, text: str):
    """Render a status indicator with color"""
    color = "status-green" if status else "status-red"
    st.markdown(f'<span class="{color} status-indicator"></span>{text}', unsafe_allow_html=True)

def render_tool_selector():
    """Render the tool selection interface"""
    st.markdown("### 🛠️ Select Neo4j Tool")
    
    # Group tools by category
    categories = {}
    for tool_id, tool_info in MCP_TOOLS.items():
        category = tool_info["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append((tool_id, tool_info))
    
    # Create expandable sections for each category
    selected_tool = None
    
    for category, tools in categories.items():
        with st.expander(f"{category} ({len(tools)} tools)", expanded=True):
            for tool_id, tool_info in tools:
                with st.container():
                    # Tool card
                    card_class = "tool-selected" if st.session_state.get('selected_tool') == tool_id else "tool-card"
                    
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="{card_class}">
                            <h4 style="color: {tool_info['color']}; margin: 0;">{tool_info['name']}</h4>
                            <p style="margin: 0.5rem 0; color: #666;">{tool_info['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if tool_info.get('example'):
                            with st.expander("💡 Example"):
                                if 'cypher' in str(tool_info.get('params', {})).lower():
                                    st.code(tool_info['example'], language='cypher')
                                else:
                                    st.code(tool_info['example'], language='text')
                    
                    with col2:
                        if st.button(f"Select", key=f"select_{tool_id}", type="primary" if st.session_state.get('selected_tool') == tool_id else "secondary"):
                            st.session_state.selected_tool = tool_id
                            st.rerun()
    
    return st.session_state.get('selected_tool')

def render_parameter_inputs(tool_id: str):
    """Render parameter input interface"""
    if not tool_id or tool_id not in MCP_TOOLS:
        return {}
    
    tool_info = MCP_TOOLS[tool_id]
    params = tool_info.get("params", {})
    
    st.markdown(f"### 📝 Configure: {tool_info['name']}")
    
    if not params:
        st.info("ℹ️ This tool doesn't require any parameters.")
        return {}
    
    user_params = {}
    
    for param_name, param_type in params.items():
        if param_type == "text":
            user_params[param_name] = st.text_area(
                f"🔤 {param_name.replace('_', ' ').title()}:",
                placeholder=f"Enter your {param_name.replace('_', ' ')}...",
                height=120,
                help=f"Natural language input for {param_name}"
            )
        elif param_type == "cypher":
            user_params[param_name] = st.text_area(
                f"⚙️ {param_name.replace('_', ' ').title()}:",
                placeholder="MATCH (n) RETURN n LIMIT 10",
                height=150,
                help="Enter your Cypher query"
            )
        elif param_type == "json":
            json_input = st.text_area(
                f"📋 {param_name.replace('_', ' ').title()} (JSON):",
                placeholder='{"key": "value"}',
                height=100,
                help="Enter JSON parameters for the query"
            )
            if json_input.strip():
                try:
                    user_params[param_name] = json.loads(json_input)
                except json.JSONDecodeError:
                    st.error(f"❌ Invalid JSON format for {param_name}")
                    user_params[param_name] = None
            else:
                user_params[param_name] = None
    
    return user_params

def render_results(result: Dict[str, Any]):
    """Render execution results with enhanced formatting"""
    st.markdown("### 📊 Execution Results")
    
    if result.get("status") == "error":
        st.markdown('<div class="result-error">', unsafe_allow_html=True)
        st.error(f"❌ **Error:** {result.get('error', 'Unknown error')}")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Success indicator
    st.markdown('<div class="result-success">', unsafe_allow_html=True)
    st.success("✅ **Tool executed successfully!**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle different result types
    if "request" in result and "execution" in result and "insights" in result:
        render_intelligent_query_results(result)
    elif "schema" in str(result).lower():
        render_schema_results(result)
    elif "components" in result:
        render_health_check_results(result)
    else:
        render_generic_results(result)

def render_intelligent_query_results(result: Dict[str, Any]):
    """Render AI-powered query results with insights"""
    
    # AI Analysis Section
    if "request" in result:
        st.markdown("#### 🧠 AI Analysis")
        request = result["request"]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Intent", request.get("intent", "N/A"))
        with col2:
            confidence = request.get("confidence", 0)
            st.metric("Confidence", f"{confidence:.1%}")
        with col3:
            st.metric("Status", "✅ Analyzed")
        with col4:
            st.metric("Timestamp", datetime.now().strftime("%H:%M:%S"))
        
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(f"**🔍 Analysis:** {request.get('analysis', 'No analysis available')}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Query Execution Section
    if "execution" in result:
        st.markdown("#### ⚙️ Query Execution")
        execution = result["execution"]
        
        st.markdown("**Generated Cypher Query:**")
        cypher_query = execution.get("cypher_query", "No query")
        st.code(cypher_query, language="cypher")
        
        st.markdown(f"**🔧 Tool Used:** `{execution.get('tool_used', 'Unknown')}`")
    
    # Data Results Section
    if "execution" in result and "raw_result" in result["execution"]:
        st.markdown("#### 📋 Query Results")
        raw_result = result["execution"]["raw_result"]
        
        try:
            # Try to parse as JSON and create DataFrame
            if isinstance(raw_result, str):
                data = json.loads(raw_result)
            else:
                data = raw_result
                
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                
                # Display data table
                st.dataframe(df, use_container_width=True)
                
                # Create visualizations for numeric data
                numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_columns) > 0 and len(df) <= 50:
                    st.markdown("#### 📈 Data Visualization")
                    
                    # Find name column for x-axis
                    name_cols = [col for col in df.columns if 'name' in col.lower()]
                    if name_cols and len(numeric_columns) > 0:
                        fig = px.bar(
                            df.head(20),  # Limit to top 20 for readability
                            x=name_cols[0], 
                            y=numeric_columns[0],
                            title=f"{numeric_columns[0]} by {name_cols[0]}",
                            color=numeric_columns[0],
                            color_continuous_scale="viridis"
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Additional chart if multiple numeric columns
                        if len(numeric_columns) > 1:
                            fig2 = px.scatter(
                                df,
                                x=numeric_columns[0],
                                y=numeric_columns[1],
                                title=f"{numeric_columns[1]} vs {numeric_columns[0]}",
                                hover_name=name_cols[0] if name_cols else None
                            )
                            st.plotly_chart(fig2, use_container_width=True)
            else:
                st.write(raw_result)
                
        except (json.JSONDecodeError, ValueError, KeyError):
            st.code(str(raw_result), language="json")
    
    # Business Insights Section
    if "insights" in result:
        st.markdown("#### 💡 Business Insights")
        insights = result["insights"]
        
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(f"**📊 Interpretation:**")
        st.markdown(insights.get('interpretation', 'No interpretation available'))
        st.markdown(f"**💼 Business Value:**")
        st.markdown(insights.get('business_value', 'No business value identified'))
        st.markdown('</div>', unsafe_allow_html=True)

def render_schema_results(result: Dict[str, Any]):
    """Render database schema results"""
    st.markdown("#### 🗄️ Database Schema Overview")
    
    # Extract schema data
    if isinstance(result.get("result"), str):
        try:
            schema_data = json.loads(result["result"])
        except json.JSONDecodeError:
            schema_data = {}
    else:
        schema_data = result.get("result", {})
    
    if not schema_data:
        st.warning("No schema data available")
        return
    
    # Overview metrics
    node_types = len([k for k, v in schema_data.items() if v.get("type") == "node"])
    total_records = sum(v.get("count", 0) for v in schema_data.values() if isinstance(v.get("count"), (int, float)))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏷️ Node Types", node_types)
    with col2:
        st.metric("📊 Total Records", f"{total_records:,}")
    with col3:
        st.metric("🔗 Schema Objects", len(schema_data))
    
    # Detailed schema breakdown
    for name, info in schema_data.items():
        with st.expander(f"📦 {name} ({info.get('type', 'unknown')})", expanded=False):
            if info.get("count"):
                st.write(f"**📊 Count:** {info['count']:,} records")
            
            if "properties" in info and info["properties"]:
                st.write("**🔧 Properties:**")
                props_data = []
                for prop_name, prop_info in info["properties"].items():
                    props_data.append({
                        "Property": prop_name,
                        "Type": prop_info.get("type", "unknown"),
                        "Indexed": "✅" if prop_info.get("indexed") else "❌"
                    })
                
                if props_data:
                    props_df = pd.DataFrame(props_data)
                    st.dataframe(props_df, use_container_width=True)
            
            if "relationships" in info and info["relationships"]:
                st.write("**🔗 Relationships:**")
                for rel_name, rel_info in info["relationships"].items():
                    direction = rel_info.get("direction", "unknown")
                    labels = rel_info.get("labels", [])
                    st.write(f"- `{rel_name}` ({direction}) → {', '.join(labels)}")

def render_health_check_results(result: Dict[str, Any]):
    """Render system health check results"""
    st.markdown("#### 🏥 System Health Dashboard")
    
    overall_status = result.get("status", "unknown")
    status_emoji = "🟢" if overall_status == "healthy" else "🔴"
    
    st.markdown(f"**Overall Status:** {status_emoji} {overall_status.upper()}")
    
    if "components" in result:
        components = result["components"]
        
        # Component status overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🔧 Core Components")
            for comp_name, comp_info in components.items():
                if comp_name in ["neo4j_database", "fastmcp_server"]:
                    status = comp_info.get("status", "unknown")
                    emoji = "🟢" if status in ["connected", "running"] else "🔴"
                    st.write(f"{emoji} **{comp_name.replace('_', ' ').title()}:** {status}")
        
        with col2:
            st.markdown("##### 🤖 AI Components")
            for comp_name, comp_info in components.items():
                if comp_name in ["langgraph_workflow", "cortex_llm"]:
                    status = comp_info.get("status", "unknown")
                    emoji = "🟢" if status in ["active", "connected"] else "🔴"
                    st.write(f"{emoji} **{comp_name.replace('_', ' ').title()}:** {status}")
        
        # Detailed component information
        for comp_name, comp_info in components.items():
            with st.expander(f"🔍 {comp_name.replace('_', ' ').title()} Details"):
                for key, value in comp_info.items():
                    if key != "status":
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")

def render_generic_results(result: Dict[str, Any]):
    """Render generic results"""
    st.markdown("#### 📄 Result Data")
    
    # Try to format nicely
    if "result" in result:
        data = result["result"]
        if isinstance(data, (list, dict)):
            st.json(data)
        else:
            st.code(str(data), language="json")
    else:
        st.json(result)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Neo4j AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown("**Standalone AI-Powered Graph Database Interface with LangGraph Intelligence**")
    
    # Initialize session state
    if "execution_history" not in st.session_state:
        st.session_state.execution_history = []
    if "server_manager" not in st.session_state:
        st.session_state.server_manager = MCPServerManager()
    if "mcp_client" not in st.session_state:
        st.session_state.mcp_client = MCPClient(st.session_state.server_manager)
    
    # Sidebar - Server Management & Configuration
    with st.sidebar:
        st.markdown("### 🔧 Server Control")
        
        # Server status
        server_status = st.session_state.server_manager.is_running
        render_status_indicator(server_status, f"MCP Server {'Running' if server_status else 'Stopped'}")
        
        # Start/Stop controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Start", disabled=server_status):
                with st.spinner("Starting MCP server..."):
                    if st.session_state.server_manager.start_server():
                        st.success("✅ Server started!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to start server")
        
        with col2:
            if st.button("🛑 Stop", disabled=not server_status):
                st.session_state.server_manager.stop_server()
                st.success("✅ Server stopped!")
                st.rerun()
        
        # Configuration display
        st.markdown("### ⚙️ Configuration")
        with st.expander("📊 Neo4j Settings"):
            st.code(f"""
URI: {NEO4J_URI}
Database: {NEO4J_DATABASE}
Username: {NEO4J_USERNAME}
Namespace: {NEO4J_NAMESPACE}
            """)
        
        with st.expander("🤖 AI Settings"):
            st.code(f"""
Model: {CORTEX_MODEL}
API: Cortex LLM
Features: Intent Analysis, Query Generation, Business Insights
            """)
        
        # Statistics
        st.markdown("### 📊 Statistics")
        st.metric("Tools Available", len(MCP_TOOLS))
        st.metric("Executions", len(st.session_state.execution_history))
        
        # Clear history
        if st.button("🗑️ Clear History", help="Clear execution history"):
            st.session_state.execution_history = []
            st.rerun()
    
    # Main content area
    if not server_status:
        st.warning("⚠️ Please start the MCP server to use the AI-powered tools.")
        st.info("Click the '🚀 Start' button in the sidebar to begin.")
        
        # Quick setup guide
        with st.expander("📋 Quick Setup Guide"):
            st.markdown("""
            **Required Files:**
            1. `standalone_streamlit_neo4j_ui.py` (this file) ✅
            2. `standalone_neo4j_mcp_server.py` (MCP server) ⚠️
            
            **Steps:**
            1. Make sure both files are in the same directory
            2. Click '🚀 Start' in the sidebar
            3. Select tools and enter prompts
            4. Enjoy AI-powered Neo4j interactions!
            """)
        return
    
    # Tool selection interface
    selected_tool = render_tool_selector()
    
    if selected_tool:
        st.markdown("---")
        
        # Parameter configuration
        parameters = render_parameter_inputs(selected_tool)
        
        # Execution controls
        st.markdown("### 🚀 Execute Tool")
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            if st.button("▶️ Execute Tool", type="primary", use_container_width=True):
                # Validate parameters
                tool_info = MCP_TOOLS[selected_tool]
                required_params = tool_info.get("params", {})
                
                missing_params = []
                for param_name, param_type in required_params.items():
                    if not parameters.get(param_name) and param_type != "json":
                        missing_params.append(param_name)
                
                if missing_params:
                    st.error(f"❌ Missing required parameters: {', '.join(missing_params)}")
                else:
                    # Execute the tool
                    with st.spinner(f"🤖 Executing {tool_info['name']}..."):
                        try:
                            # Run async tool execution
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            result = loop.run_until_complete(
                                st.session_state.mcp_client.call_tool(selected_tool, parameters)
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
            if st.button("📋 Example", use_container_width=True):
                example = tool_info.get("example", "")
                if example:
                    st.code(example)
                    st.info("💡 Copy this example to the parameter field above")
        
        with col3:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.selected_tool = None
                st.rerun()
    
    # Execution history
    if st.session_state.execution_history:
        st.markdown("---")
        st.markdown("### 📚 Recent Executions")
        
        # Show last 3 executions
        for i, record in enumerate(reversed(st.session_state.execution_history[-3:])):
            with st.expander(f"🕒 {record['timestamp'].strftime('%H:%M:%S')} - {record['tool_name']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Parameters:**")
                    st.json(record["parameters"])
                
                with col2:
                    st.markdown("**Status:**")
                    status = record["result"].get("status", "unknown")
                    emoji = "✅" if status == "success" else "❌"
                    st.write(f"{emoji} {status}")
                
                if st.button(f"🔄 Re-execute", key=f"rerun_{i}"):
                    st.session_state.selected_tool = record["tool"]
                    st.rerun()

if __name__ == "__main__":
    # App info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    **Neo4j AI Assistant**
    
    🧠 **Features:**
    - AI-powered natural language queries
    - Smart intent analysis
    - Automatic Cypher generation
    - Business insights extraction
    - Interactive data visualization
    
    🔧 **Technologies:**
    - FastMCP for tool protocols
    - LangGraph for AI orchestration
    - Neo4j for graph database
    - Streamlit for UI
    - Cortex for LLM processing
    """)
    
    main()
