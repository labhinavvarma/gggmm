# clean_standalone_streamlit_ui.py
"""
Clean Standalone Streamlit UI for Neo4j MCP Server
NO EXPANDERS - Simple and clean interface
"""

import streamlit as st

# PAGE CONFIGURATION - MUST BE FIRST!
st.set_page_config(
    page_title="🧠 Neo4j AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

import asyncio
import json
import subprocess
import time
import os
import sys
import uuid
import requests
import urllib3
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# Suppress warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Install dependencies
def install_deps():
    packages = ["pandas", "plotly", "requests"]
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

install_deps()
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# CONFIGURATION
# ============================================================================

NEO4J_URI = "neo4j://10.189.116.237:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "Vkg5d$F!pLq2@9vRwE="
NEO4J_DATABASE = "connectiq"
NEO4J_NAMESPACE = "connectiq"

CORTEX_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
CORTEX_API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
CORTEX_APP_ID = "edadip"
CORTEX_APLCTN_CD = "edagnai"
CORTEX_MODEL = "llama3.1-70b"

MCP_SERVER_FILE = "standalone_neo4j_mcp_server.py"

# ============================================================================
# TOOLS CONFIGURATION
# ============================================================================

TOOLS = {
    "intelligent_neo4j_query": {
        "name": "🧠 AI-Powered Query",
        "description": "Natural language processing with AI analysis and business insights",
        "params": {"user_input": "text"},
        "example": "Show me the top 10 most popular fitness apps with ratings above 4.0",
        "category": "AI"
    },
    "get_neo4j_schema": {
        "name": "📊 Database Schema", 
        "description": "Get complete database structure and information",
        "params": {},
        "example": "No parameters needed - just click execute",
        "category": "Schema"
    },
    "read_neo4j_cypher": {
        "name": "🔍 Read Query",
        "description": "Execute read-only Cypher queries",
        "params": {"query": "cypher", "params": "json"},
        "example": "MATCH (a:Apps) WHERE a.rating > 4.0 RETURN a.name, a.rating ORDER BY a.rating DESC LIMIT 10",
        "category": "Query"
    },
    "write_neo4j_cypher": {
        "name": "✏️ Write Query",
        "description": "Execute write operations (CREATE, MERGE, DELETE, etc.)",
        "params": {"query": "cypher", "params": "json"},
        "example": "CREATE (a:App {name: 'TestApp', rating: 4.5})",
        "category": "Query"
    },
    "system_health_check": {
        "name": "🏥 Health Check",
        "description": "Check system and database connectivity",
        "params": {},
        "example": "No parameters needed - just click execute",
        "category": "System"
    }
}

# ============================================================================
# MCP SERVER MANAGER
# ============================================================================

class MCPServerManager:
    def __init__(self):
        self.process = None
        self.is_running = False
        
    def start_server(self):
        if self.is_running:
            return True
            
        try:
            if not Path(MCP_SERVER_FILE).exists():
                st.error(f"❌ Server file not found: {MCP_SERVER_FILE}")
                return False
            
            self.process = subprocess.Popen([
                sys.executable, MCP_SERVER_FILE
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.is_running = True
            return True
        except Exception as e:
            st.error(f"❌ Server start failed: {e}")
            return False
    
    def stop_server(self):
        if self.process:
            self.process.terminate()
            self.process = None
            self.is_running = False

# ============================================================================
# MCP CLIENT
# ============================================================================

class MCPClient:
    def __init__(self, server_manager):
        self.server_manager = server_manager
        
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if not self.server_manager.is_running:
            return {"status": "error", "error": "Server not running"}
        
        try:
            request = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tools/call",
                "params": {
                    "name": f"{NEO4J_NAMESPACE}-{tool_name}" if NEO4J_NAMESPACE else tool_name,
                    "arguments": parameters
                }
            }
            
            request_json = json.dumps(request) + "\n"
            self.server_manager.process.stdin.write(request_json.encode())
            self.server_manager.process.stdin.flush()
            
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
            
            return {"status": "error", "error": "No response"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}

# ============================================================================
# UI FUNCTIONS
# ============================================================================

def render_tool_selector():
    """Simple tool selector without any expanders"""
    st.markdown("### 🛠️ Select Tool")
    
    # Simple selectbox for tool selection
    tool_options = {}
    for tool_id, tool_info in TOOLS.items():
        display_name = f"{tool_info['category']} - {tool_info['name']}"
        tool_options[display_name] = tool_id
    
    selected_display = st.selectbox(
        "Choose a tool:",
        [""] + list(tool_options.keys())
    )
    
    if not selected_display:
        return None
    
    selected_tool = tool_options[selected_display]
    tool_info = TOOLS[selected_tool]
    
    # Show tool information
    st.markdown(f"**{tool_info['name']}**")
    st.markdown(f"*{tool_info['description']}*")
    
    # Show example
    if tool_info.get('example'):
        st.markdown("**💡 Example:**")
        st.code(tool_info['example'])
    
    return selected_tool

def render_parameters(tool_id: str):
    """Render parameter inputs"""
    if not tool_id:
        return {}
    
    tool_info = TOOLS[tool_id]
    params = tool_info.get("params", {})
    
    if not params:
        st.info("This tool requires no parameters.")
        return {}
    
    st.markdown("### 📝 Parameters")
    user_params = {}
    
    for param_name, param_type in params.items():
        if param_type == "text":
            user_params[param_name] = st.text_area(
                f"{param_name.title()}:",
                height=100,
                key=f"param_{param_name}"
            )
        elif param_type == "cypher":
            user_params[param_name] = st.text_area(
                f"{param_name.title()}:",
                height=150,
                key=f"param_{param_name}"
            )
        elif param_type == "json":
            json_input = st.text_area(
                f"{param_name.title()} (JSON):",
                height=100,
                key=f"param_{param_name}"
            )
            if json_input.strip():
                try:
                    user_params[param_name] = json.loads(json_input)
                except json.JSONDecodeError:
                    st.error(f"Invalid JSON for {param_name}")
                    user_params[param_name] = None
            else:
                user_params[param_name] = None
    
    return user_params

def render_results(result: Dict[str, Any]):
    """Render results without expanders"""
    st.markdown("### 📊 Results")
    
    if result.get("status") == "error":
        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
        return
    
    st.success("✅ Success!")
    
    # Handle different result types
    if "request" in result and "execution" in result:
        # AI-powered query result
        st.markdown("#### 🧠 AI Analysis")
        
        request = result.get("request", {})
        if request:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Intent", request.get("intent", "N/A"))
            with col2:
                st.metric("Confidence", f"{request.get('confidence', 0):.1%}")
        
        execution = result.get("execution", {})
        if execution.get("cypher_query"):
            st.markdown("#### ⚙️ Generated Query")
            st.code(execution["cypher_query"], language="cypher")
        
        if execution.get("raw_result"):
            st.markdown("#### 📋 Data")
            try:
                raw_data = execution["raw_result"]
                if isinstance(raw_data, str):
                    data = json.loads(raw_data)
                else:
                    data = raw_data
                
                if isinstance(data, list) and data:
                    df = pd.DataFrame(data)
                    st.dataframe(df)
                    
                    # Simple chart
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0 and len(df) <= 20:
                        fig = px.bar(df.head(10), y=numeric_cols[0], title="Data Visualization")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write(data)
            except:
                st.code(str(execution["raw_result"]))
        
        insights = result.get("insights", {})
        if insights.get("interpretation"):
            st.markdown("#### 💡 Insights")
            st.markdown(insights["interpretation"])
    
    elif "components" in result:
        # Health check result
        st.markdown("#### 🏥 System Status")
        components = result["components"]
        
        for comp_name, comp_info in components.items():
            status = comp_info.get("status", "unknown")
            emoji = "🟢" if status in ["connected", "running", "active"] else "🔴"
            st.write(f"{emoji} **{comp_name.replace('_', ' ').title()}**: {status}")
    
    else:
        # Generic result
        st.json(result)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown("# 🧠 Neo4j AI Assistant")
    st.markdown("**Simple AI-Powered Graph Database Interface**")
    
    # Initialize session state
    if "server_manager" not in st.session_state:
        st.session_state.server_manager = MCPServerManager()
    if "mcp_client" not in st.session_state:
        st.session_state.mcp_client = MCPClient(st.session_state.server_manager)
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Server Control")
        
        # Server status
        is_running = st.session_state.server_manager.is_running
        status_text = "🟢 Running" if is_running else "🔴 Stopped"
        st.markdown(f"**Status:** {status_text}")
        
        # Start/Stop buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Start", disabled=is_running):
                if st.session_state.server_manager.start_server():
                    st.success("Started!")
                    time.sleep(1)
                    st.rerun()
        
        with col2:
            if st.button("🛑 Stop", disabled=not is_running):
                st.session_state.server_manager.stop_server()
                st.success("Stopped!")
                st.rerun()
        
        # Configuration
        st.markdown("### ⚙️ Configuration")
        st.code(f"Database: {NEO4J_DATABASE}\nURI: {NEO4J_URI}")
        
        # History
        st.markdown("### 📊 Statistics")
        st.metric("Executions", len(st.session_state.history))
        
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    
    # Main content
    if not is_running:
        st.warning("⚠️ Please start the MCP server first.")
        st.info("Click '🚀 Start' in the sidebar to begin.")
        return
    
    # Tool selection
    selected_tool = render_tool_selector()
    
    if selected_tool:
        st.markdown("---")
        
        # Parameters
        parameters = render_parameters(selected_tool)
        
        # Execute button
        st.markdown("### 🚀 Execute")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("▶️ Execute Tool", type="primary"):
                # Validate required parameters
                tool_info = TOOLS[selected_tool]
                required_params = tool_info.get("params", {})
                
                missing = []
                for param_name, param_type in required_params.items():
                    if not parameters.get(param_name) and param_type != "json":
                        missing.append(param_name)
                
                if missing:
                    st.error(f"Missing: {', '.join(missing)}")
                else:
                    with st.spinner("Executing..."):
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            result = loop.run_until_complete(
                                st.session_state.mcp_client.call_tool(selected_tool, parameters)
                            )
                            loop.close()
                            
                            # Store in history
                            st.session_state.history.append({
                                "timestamp": datetime.now(),
                                "tool": selected_tool,
                                "tool_name": tool_info["name"],
                                "result": result
                            })
                            
                            # Show results
                            st.markdown("---")
                            render_results(result)
                            
                        except Exception as e:
                            st.error(f"Execution failed: {e}")
        
        with col2:
            if st.button("🔄 Reset"):
                st.rerun()
    
    # History
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📚 Recent Executions")
        
        for record in reversed(st.session_state.history[-3:]):
            timestamp = record["timestamp"].strftime("%H:%M:%S")
            tool_name = record["tool_name"]
            status = record["result"].get("status", "unknown")
            emoji = "✅" if status == "success" else "❌"
            
            st.write(f"{timestamp} - {tool_name} {emoji}")

if __name__ == "__main__":
    main()
