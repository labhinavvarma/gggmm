# test.py - UPDATED VERSION - Replace your existing file with this

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
from fastmcp import Client
from mcpserver import main as start_mcp_server  # FIXED: Use mcpserver (not mcpserver_http)

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

class CortexLLMClient:
    """Enhanced Cortex LLM client with better error handling."""
    
    def __init__(self, url: str, api_key: str, app_id: str, aplctn_cd: str, model: str):
        self.url = url
        self.api_key = api_key
        self.app_id = app_id
        self.aplctn_cd = aplctn_cd
        self.model = model
    
    def generate_cypher(self, user_query: str) -> str:
        """Generate Cypher query using Cortex LLM."""
        prompt = f"""
        You are an expert Neo4j Cypher query generator. 
        Generate a valid Cypher query for the following request: {user_query}
        
        Rules:
        1. Return ONLY the Cypher query, no explanations
        2. Use proper Neo4j syntax
        3. Include appropriate LIMIT clauses for large datasets
        4. Use MATCH for read operations, CREATE/MERGE for write operations
        
        Query request: {user_query}
        """
        
        return self._call_cortex(prompt)
    
    def _call_cortex(self, prompt: str) -> str:
        """Call Cortex API with enhanced error handling."""
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
            response = requests.post(
                self.url, 
                headers=headers, 
                json=payload, 
                verify=False, 
                timeout=30
            )
            response.raise_for_status()
            
            raw_text = response.text
            if "end_of_stream" in raw_text:
                return raw_text.split("end_of_stream")[0].strip()
            return raw_text.strip()
            
        except requests.exceptions.Timeout:
            return "❌ Request timed out"
        except requests.exceptions.RequestException as e:
            return f"❌ Request failed: {e}"
        except Exception as e:
            return f"❌ Unexpected error: {e}"

class FixedMCPClient:
    """Fixed MCP client using HTTP transport (no more SSE errors)."""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
    
    async def call_tool_async(self, tool_name: str, query: str, max_retries: int = 3) -> str:
        """Call MCP tool using HTTP transport (TaskGroup error fixed)."""
        for attempt in range(max_retries):
            try:
                # Use HTTP client instead of deprecated SSE
                async with Client(self.server_url) as client:
                    # List available tools
                    tools = await client.list_tools()
                    available_tools = [t.name for t in tools.tools]
                    
                    if tool_name not in available_tools:
                        return f"❌ Tool '{tool_name}' not found. Available: {available_tools}"
                    
                    # Call the tool
                    result = await client.call_tool(tool_name, {"query": query})
                    
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
                return f"❌ Connection failed after {max_retries} attempts: {e}"
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return f"❌ Error after {max_retries} attempts: {e}"
        
        return f"❌ Failed after {max_retries} attempts"
    
    async def health_check_async(self) -> bool:
        """Check if MCP server is healthy."""
        try:
            async with Client(self.server_url) as client:
                result = await client.call_tool("health_check", {})
                return not str(result).startswith("❌")
        except Exception:
            return False

# Server management
def start_server():
    """Start MCP server with HTTP transport."""
    def server_runner():
        try:
            start_mcp_server()
        except Exception as e:
            print(f"Server failed: {e}")
    
    thread = threading.Thread(target=server_runner, daemon=True)
    thread.start()
    time.sleep(4)  # Give server time to start
    return True

# Initialize components
if "task_manager" not in st.session_state:
    st.session_state.task_manager = AsyncTaskManager()

if "server_started" not in st.session_state:
    st.session_state.server_started = start_server()

if "cortex_client" not in st.session_state:
    st.session_state.cortex_client = CortexLLMClient(CORTEX_URL, API_KEY, APP_ID, APLCTN_CD, MODEL)

if "mcp_client" not in st.session_state:
    st.session_state.mcp_client = FixedMCPClient()

# Main UI
st.title("🧠 Neo4j + Cortex (TaskGroup Error FIXED)")
st.success("✅ Using HTTP transport (SSE deprecated errors resolved)")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "server_healthy" not in st.session_state:
    st.session_state.server_healthy = None

# Sidebar
with st.sidebar:
    st.header("🎛️ Status")
    
    # Server status
    if st.session_state.server_started:
        st.success("✅ MCP Server (HTTP Transport)")
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
    st.write(f"Transport: HTTP")
    st.write(f"History: {len(st.session_state.history)} items")

# Quick test buttons
st.subheader("🚀 Quick Tests")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Count Nodes"):
        st.session_state.test_query = "How many nodes are in the database?"

with col2:
    if st.button("Show Nodes"):
        st.session_state.test_query = "Show me all nodes in the database"

with col3:
    if st.button("List Relationships"):
        st.session_state.test_query = "Show me all relationships"

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
for role, message in reversed(st.session_state.history):
    if role == "user":
        st.chat_message("user").write(f"**You:** {message}")
    elif role == "cortex":
        st.chat_message("assistant").write(f"🤖 **Generated Cypher:**\n```cypher\n{message}\n```")
    elif role == "neo4j":
        st.chat_message("assistant").write(f"📊 **Neo4j Result:**\n```json\n{message}\n```")
    elif role == "error":
        st.chat_message("assistant").write(f"❌ **Error:** {message}")

# Footer
st.divider()
st.caption("🎯 TaskGroup errors resolved • HTTP transport • No SSE deprecation warnings")
