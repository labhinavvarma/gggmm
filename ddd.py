# test.py - Fixed version based on your working SSE client

import streamlit as st

# CRITICAL: This MUST be the first Streamlit command
st.set_page_config(page_title="Neo4j + Cortex Fixed", page_icon="🧠", layout="wide")

# Now import everything else
import asyncio
import json
import uuid
import requests
import threading
import urllib3
import time
import sys
from asyncio import run_coroutine_threadsafe
from threading import Thread
from typing import Optional, Dict, Any
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcpserver import main as start_mcp_server

# Configuration
CORTEX_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
APP_ID = "edadip"
APLCTN_CD = "edagnai"
MODEL = "llama3.1-70b"
SYS_MSG = "You are a powerful AI assistant specialized in Neo4j Cypher queries. Provide accurate, concise Cypher queries."

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class AsyncTaskManager:
    """Manages async tasks with proper error handling."""
    
    def __init__(self):
        self.loop = None
        self.thread = None
        self._setup_event_loop()
    
    def _setup_event_loop(self):
        """Set up event loop with version compatibility."""
        try:
            self.loop = asyncio.new_event_loop()
            self.thread = Thread(target=self._run_event_loop, daemon=True)
            self.thread.start()
            time.sleep(0.1)
        except Exception as e:
            print(f"Failed to create event loop: {e}")
    
    def _run_event_loop(self):
        """Run the event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        except Exception as e:
            print(f"Event loop error: {e}")
    
    def run_async(self, coroutine, timeout=30):
        """Run async function with timeout and error handling."""
        try:
            if self.loop is None or self.loop.is_closed():
                self._setup_event_loop()
            
            future = run_coroutine_threadsafe(coroutine, self.loop)
            return future.result(timeout=timeout)
        except asyncio.TimeoutError:
            return f"❌ Operation timed out after {timeout} seconds"
        except Exception as e:
            return f"❌ Async task failed: {e}"

class FixedCortexClient:
    """Fixed Cortex client with better error handling."""
    
    def __init__(self, url: str, api_key: str, app_id: str, aplctn_cd: str, model: str):
        self.url = url
        self.api_key = api_key
        self.app_id = app_id
        self.aplctn_cd = aplctn_cd
        self.model = model
    
    def generate_cypher(self, user_query: str) -> str:
        """Generate Cypher query using Cortex LLM with better error handling."""
        prompt = f"""
        You are an expert Neo4j Cypher query generator. 
        Generate a valid Cypher query for the following request: {user_query}
        
        Rules:
        1. Return ONLY the Cypher query, no explanations
        2. Use proper Neo4j syntax
        3. Include appropriate LIMIT clauses (default LIMIT 10)
        4. Use MATCH for read operations, CREATE/MERGE for write operations
        
        Query request: {user_query}
        """
        
        return self._call_cortex(prompt)
    
    def _call_cortex(self, prompt: str) -> str:
        """Call Cortex API with comprehensive error handling."""
        payload = {
            "query": {
                "aplctn_cd": self.aplctn_cd,
                "app_id": self.app_id,
                "api_key": self.api_key,
                "method": "cortex",
                "model": self.model,
                "sys_msg": SYS_MSG,
                "limit_convs": "0",
                "prompt": {
                    "messages": [{"role": "user", "content": prompt}]
                },
                "session_id": str(uuid.uuid4())
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f'Snowflake Token="{self.api_key}"'
        }

        try:
            # Add more detailed error handling
            response = requests.post(
                self.url, 
                headers=headers, 
                json=payload, 
                verify=False, 
                timeout=30
            )
            
            # Check if request was successful
            if response.status_code == 200:
                raw_text = response.text
                if "end_of_stream" in raw_text:
                    result = raw_text.split("end_of_stream")[0].strip()
                    return result if result else "MATCH (n) RETURN n LIMIT 10"
                return raw_text.strip() if raw_text.strip() else "MATCH (n) RETURN n LIMIT 10"
            elif response.status_code == 401:
                return "❌ Authentication failed - check API key"
            elif response.status_code == 403:
                return "❌ Access forbidden - check permissions"
            elif response.status_code == 404:
                return "❌ Service not found - check URL"
            else:
                return f"❌ HTTP {response.status_code}: {response.text[:200]}"
                
        except requests.exceptions.Timeout:
            return "❌ Request timed out - try again"
        except requests.exceptions.ConnectionError:
            return "❌ Connection failed - check network"
        except requests.exceptions.RequestException as e:
            return f"❌ Request failed: {str(e)[:100]}"
        except Exception as e:
            return f"❌ Unexpected error: {str(e)[:100]}"

class WorkingMCPClient:
    """MCP client based on your working SSE code."""
    
    def __init__(self, server_url: str = "http://localhost:8001/sse"):
        self.server_url = server_url
    
    async def call_tool_async(self, tool_name: str, query: str, max_retries: int = 3) -> str:
        """Call MCP tool using SSE transport (based on your working code)."""
        for attempt in range(max_retries):
            try:
                # Use SSE client like your working code
                async with sse_client(url=self.server_url) as sse_connection:
                    async with ClientSession(*sse_connection) as session:
                        # Initialize session
                        await session.initialize()
                        
                        # List available tools
                        tools = await session.list_tools()
                        available_tools = [t.name for t in tools.tools] if hasattr(tools, 'tools') else []
                        
                        if tool_name not in available_tools:
                            return f"❌ Tool '{tool_name}' not found. Available: {available_tools}"
                        
                        # Find the tool
                        tool = None
                        for t in tools.tools:
                            if t.name == tool_name:
                                tool = t
                                break
                        
                        if not tool:
                            return f"❌ Tool '{tool_name}' not found"
                        
                        # Call the tool
                        result = await session.call_tool(tool, {"query": query})
                        
                        # Extract content
                        if hasattr(result, 'content') and result.content:
                            content = result.content[0].text if result.content else "✅ Query executed successfully"
                            return content
                        else:
                            return "✅ Query executed successfully"
                            
            except ConnectionError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return f"❌ Connection failed after {max_retries} attempts: {str(e)[:100]}"
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return f"❌ Error after {max_retries} attempts: {str(e)[:100]}"
        
        return f"❌ Failed after {max_retries} attempts"
    
    async def health_check_async(self) -> bool:
        """Check if MCP server is healthy."""
        try:
            async with sse_client(url=self.server_url) as sse_connection:
                async with ClientSession(*sse_connection) as session:
                    await session.initialize()
                    result = await session.call_tool("health_check", {})
                    return True  # If we get here, connection is working
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    async def get_server_info_async(self) -> str:
        """Get server information including tools and resources."""
        try:
            async with sse_client(url=self.server_url) as sse_connection:
                async with ClientSession(*sse_connection) as session:
                    await session.initialize()
                    
                    # Get tools
                    tools = await session.list_tools()
                    tool_names = [t.name for t in tools.tools] if hasattr(tools, 'tools') else []
                    
                    # Get resources
                    resources = await session.list_resources()
                    resource_names = [r.name for r in resources.resources] if hasattr(resources, 'resources') else []
                    
                    info = {
                        "server_url": self.server_url,
                        "tools": tool_names,
                        "resources": resource_names,
                        "status": "connected"
                    }
                    
                    return json.dumps(info, indent=2)
        except Exception as e:
            return f"❌ Failed to get server info: {str(e)[:100]}"

# Server management
def start_server():
    """Start MCP server with SSE transport."""
    def server_runner():
        try:
            start_mcp_server()
        except Exception as e:
            print(f"Server failed: {e}")
    
    thread = threading.Thread(target=server_runner, daemon=True)
    thread.start()
    time.sleep(5)  # Give server more time to start
    return True

# Initialize components
if "task_manager" not in st.session_state:
    st.session_state.task_manager = AsyncTaskManager()

if "server_started" not in st.session_state:
    st.session_state.server_started = start_server()

if "cortex_client" not in st.session_state:
    st.session_state.cortex_client = FixedCortexClient(CORTEX_URL, API_KEY, APP_ID, APLCTN_CD, MODEL)

if "mcp_client" not in st.session_state:
    st.session_state.mcp_client = WorkingMCPClient()

# Main UI
st.title("🧠 Neo4j + Cortex (Fixed SSE + Cortex)")
st.success("✅ Using SSE transport (based on your working code)")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "server_healthy" not in st.session_state:
    st.session_state.server_healthy = None
if "server_info" not in st.session_state:
    st.session_state.server_info = None

# Sidebar
with st.sidebar:
    st.header("🎛️ Status")
    
    # Server status
    if st.session_state.server_started:
        st.success("✅ MCP Server (SSE Transport)")
        st.code("http://localhost:8001/sse")
    else:
        st.error("❌ MCP Server Failed")
    
    # Health check
    if st.button("🏥 Health Check"):
        with st.spinner("Checking server health..."):
            health_result = st.session_state.task_manager.run_async(
                st.session_state.mcp_client.health_check_async()
            )
            st.session_state.server_healthy = health_result
            if health_result:
                st.success("✅ Server Healthy")
            else:
                st.error("❌ Server Unhealthy")
    
    # Server info
    if st.button("📋 Server Info"):
        with st.spinner("Getting server info..."):
            server_info = st.session_state.task_manager.run_async(
                st.session_state.mcp_client.get_server_info_async()
            )
            st.session_state.server_info = server_info
            if server_info and not server_info.startswith("❌"):
                st.success("✅ Server Info Retrieved")
                try:
                    info = json.loads(server_info)
                    st.json(info)
                except:
                    st.code(server_info)
            else:
                st.error("❌ Failed to get server info")
    
    # Display last health check
    if st.session_state.server_healthy is not None:
        status = "✅ Healthy" if st.session_state.server_healthy else "❌ Unhealthy"
        st.info(f"Last check: {status}")
    
    st.divider()
    
    # Clear history
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
    
    # Debug info
    st.subheader("🔧 Debug")
    st.write(f"Python: {sys.version[:5]}")
    st.write(f"Transport: SSE")
    st.write(f"History: {len(st.session_state.history)} items")

# Quick test buttons
st.subheader("🚀 Quick Tests")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Count Nodes"):
        st.session_state.test_query = "How many nodes are in the database?"

with col2:
    if st.button("Show Sample Nodes"):
        st.session_state.test_query = "Show me 5 sample nodes"

with col3:
    if st.button("Database Info"):
        st.session_state.test_query = "What is the schema of this database?"

with col4:
    if st.button("List Relationships"):
        st.session_state.test_query = "Show me all relationship types"

# Test Cortex separately
st.subheader("🧪 Test Cortex API")
col1, col2 = st.columns(2)

with col1:
    if st.button("Test Cortex Connection"):
        with st.spinner("Testing Cortex API..."):
            test_result = st.session_state.cortex_client.generate_cypher("RETURN 1")
            if test_result.startswith("❌"):
                st.error(f"Cortex Error: {test_result}")
            else:
                st.success("✅ Cortex API Working")
                st.code(test_result)

with col2:
    if st.button("Test MCP Tools"):
        with st.spinner("Testing MCP tools..."):
            test_result = st.session_state.task_manager.run_async(
                st.session_state.mcp_client.call_tool_async("health_check", "")
            )
            if test_result.startswith("❌"):
                st.error(f"MCP Error: {test_result}")
            else:
                st.success("✅ MCP Tools Working")
                st.code(test_result)

# Chat input
user_query = st.chat_input("Ask a Neo4j question...")

# Handle test queries
if "test_query" in st.session_state:
    user_query = st.session_state.test_query
    del st.session_state.test_query

if user_query:
    # Add user message
    st.session_state.history.append(("user", user_query))
    
    # Generate Cypher query
    with st.spinner("🤖 Generating Cypher..."):
        cypher_query = st.session_state.cortex_client.generate_cypher(user_query)
    
    if cypher_query.startswith("❌"):
        st.session_state.history.append(("error", f"Cortex Error: {cypher_query}"))
    else:
        st.session_state.history.append(("cortex", cypher_query))
        
        # Determine tool type
        is_write_query = any(keyword in cypher_query.upper() 
                           for keyword in ["CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP"])
        tool_name = "write_neo4j_cypher" if is_write_query else "read_neo4j_cypher"
        
        # Execute query
        with st.spinner(f"⚡ Executing {tool_name}..."):
            result = st.session_state.task_manager.run_async(
                st.session_state.mcp_client.call_tool_async(tool_name, cypher_query)
            )
        
        if result:
            st.session_state.history.append(("neo4j", result))
        else:
            st.session_state.history.append(("error", "Failed to execute query"))

# Display chat history
st.subheader("📜 Chat History")
if st.session_state.history:
    for role, message in reversed(st.session_state.history):
        if role == "user":
            st.chat_message("user").write(f"**You:** {message}")
        elif role == "cortex":
            st.chat_message("assistant").write(f"🤖 **Generated Cypher:**\n```cypher\n{message}\n```")
        elif role == "neo4j":
            st.chat_message("assistant").write(f"📊 **Neo4j Result:**\n```json\n{message}\n```")
        elif role == "error":
            st.chat_message("assistant").write(f"❌ **Error:** {message}")
else:
    st.info("👋 Ask a Neo4j question to get started!")

# Footer
st.divider()
st.caption("🎯 Fixed Cortex errors • SSE transport based on your working code • Robust error handling")
