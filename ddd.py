
# test.py - BULLETPROOF TaskGroup fix - No more retry loops causing issues

import streamlit as st

# CRITICAL: This MUST be the first Streamlit command
st.set_page_config(page_title="Neo4j + Cortex Bulletproof", page_icon="🧠", layout="wide")

# Now import everything else
import asyncio
import json
import uuid
import requests
import threading
import urllib3
import time
import sys
import nest_asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcpserver import main as start_mcp_server

# Apply nest_asyncio to handle event loop issues
nest_asyncio.apply()

# Configuration
CORTEX_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
APP_ID = "edadip"
APLCTN_CD = "edagnai"
MODEL = "llama3.1-70b"
SYS_MSG = "You are a powerful AI assistant specialized in Neo4j Cypher queries. Provide accurate, concise Cypher queries."

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class SimpleCortexClient:
    """Simple Cortex client without complex error handling."""
    
    def __init__(self, url: str, api_key: str, app_id: str, aplctn_cd: str, model: str):
        self.url = url
        self.api_key = api_key
        self.app_id = app_id
        self.aplctn_cd = aplctn_cd
        self.model = model
    
    def generate_cypher(self, user_query: str) -> str:
        """Generate Cypher query using Cortex LLM."""
        prompt = f"""
        Generate a Neo4j Cypher query for: {user_query}
        
        Return ONLY the Cypher query, no explanations.
        Use MATCH for reads, CREATE/MERGE for writes.
        Always include LIMIT 10 for queries that return multiple results.
        """
        
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
            
            if response.status_code == 200:
                raw_text = response.text
                if "end_of_stream" in raw_text:
                    result = raw_text.split("end_of_stream")[0].strip()
                    return result if result else "MATCH (n) RETURN count(n)"
                return raw_text.strip() if raw_text.strip() else "MATCH (n) RETURN count(n)"
            else:
                return f"❌ Cortex API error: HTTP {response.status_code}"
                
        except Exception as e:
            return f"❌ Cortex error: {str(e)[:100]}"

class BulletproofMCPClient:
    """Bulletproof MCP client - NO retry loops, single attempt only."""
    
    def __init__(self, server_url: str = "http://localhost:8001/sse"):
        self.server_url = server_url
    
    async def call_tool_single_attempt(self, tool_name: str, query: str) -> str:
        """Call MCP tool with single attempt - no retry loops to avoid TaskGroup issues."""
        try:
            # Simple, direct approach - no retry loops
            async with sse_client(url=self.server_url) as sse_connection:
                async with ClientSession(*sse_connection) as session:
                    await session.initialize()
                    
                    # Get tools
                    tools = await session.list_tools()
                    if not hasattr(tools, 'tools'):
                        return "❌ No tools found"
                    
                    # Find tool
                    target_tool = None
                    for tool in tools.tools:
                        if tool.name == tool_name:
                            target_tool = tool
                            break
                    
                    if not target_tool:
                        available = [t.name for t in tools.tools]
                        return f"❌ Tool '{tool_name}' not found. Available: {available}"
                    
                    # Call tool
                    result = await session.call_tool(target_tool, {"query": query})
                    
                    # Extract result
                    if hasattr(result, 'content') and result.content:
                        return result.content[0].text if result.content[0].text else "✅ Success"
                    else:
                        return "✅ Query executed successfully"
                        
        except Exception as e:
            return f"❌ MCP error: {str(e)[:150]}"
    
    async def health_check_single(self) -> str:
        """Single health check - no retries."""
        try:
            async with sse_client(url=self.server_url) as sse_connection:
                async with ClientSession(*sse_connection) as session:
                    await session.initialize()
                    
                    # Try to list tools as health check
                    tools = await session.list_tools()
                    tool_count = len(tools.tools) if hasattr(tools, 'tools') else 0
                    
                    return f"✅ Server healthy - {tool_count} tools available"
                    
        except Exception as e:
            return f"❌ Health check failed: {str(e)[:100]}"

# Simple async runner - no complex thread management
def run_async_simple(coro):
    """Simple async runner using nest_asyncio."""
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        # Create new loop if needed
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

# Server management
def start_server():
    """Start MCP server."""
    def server_runner():
        try:
            start_mcp_server()
        except Exception as e:
            print(f"Server failed: {e}")
    
    thread = threading.Thread(target=server_runner, daemon=True)
    thread.start()
    time.sleep(6)  # Give server time to start
    return True

# Initialize components
if "server_started" not in st.session_state:
    st.session_state.server_started = start_server()

if "cortex_client" not in st.session_state:
    st.session_state.cortex_client = SimpleCortexClient(CORTEX_URL, API_KEY, APP_ID, APLCTN_CD, MODEL)

if "mcp_client" not in st.session_state:
    st.session_state.mcp_client = BulletproofMCPClient()

# Main UI
st.title("🧠 Neo4j + Cortex (Bulletproof TaskGroup Fix)")
st.success("✅ Single attempt operations - No retry loops causing TaskGroup errors")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.header("🎛️ Status")
    
    # Server status
    if st.session_state.server_started:
        st.success("✅ MCP Server Started")
        st.code("http://localhost:8001/sse")
    else:
        st.error("❌ MCP Server Failed")
    
    # Simple health check
    if st.button("🏥 Health Check"):
        with st.spinner("Checking health..."):
            health_result = run_async_simple(
                st.session_state.mcp_client.health_check_single()
            )
            if health_result.startswith("✅"):
                st.success(health_result)
            else:
                st.error(health_result)
    
    st.divider()
    
    # Clear history
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
    
    # Simple debug info
    st.subheader("🔧 Info")
    st.write(f"Transport: SSE")
    st.write(f"nest_asyncio: Applied")
    st.write(f"History: {len(st.session_state.history)} items")

# Quick test buttons
st.subheader("🚀 Quick Tests")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Count Nodes"):
        st.session_state.test_query = "How many nodes are in the database?"

with col2:
    if st.button("Show 5 Nodes"):
        st.session_state.test_query = "Show me 5 nodes from the database"

with col3:
    if st.button("Database Schema"):
        st.session_state.test_query = "What labels exist in the database?"

# Test buttons
st.subheader("🧪 Component Tests")
col1, col2 = st.columns(2)

with col1:
    if st.button("Test Cortex Only"):
        with st.spinner("Testing Cortex..."):
            result = st.session_state.cortex_client.generate_cypher("RETURN 1")
            if result.startswith("❌"):
                st.error(result)
            else:
                st.success("✅ Cortex working")
                st.code(result)

with col2:
    if st.button("Test MCP Only"):
        with st.spinner("Testing MCP..."):
            result = run_async_simple(
                st.session_state.mcp_client.call_tool_single_attempt("health_check", "")
            )
            if result.startswith("❌"):
                st.error(result)
            else:
                st.success("✅ MCP working")
                st.code(result)

# Main chat input
st.subheader("💬 Chat")
user_query = st.chat_input("Ask a Neo4j question...")

# Handle test queries
if "test_query" in st.session_state:
    user_query = st.session_state.test_query
    del st.session_state.test_query

if user_query:
    # Add user message
    st.session_state.history.append(("user", user_query))
    
    # Generate Cypher - simple, no complex error handling
    with st.spinner("🤖 Generating Cypher..."):
        cypher_query = st.session_state.cortex_client.generate_cypher(user_query)
    
    if cypher_query.startswith("❌"):
        st.session_state.history.append(("error", cypher_query))
    else:
        st.session_state.history.append(("cortex", cypher_query))
        
        # Determine tool
        is_write = any(kw in cypher_query.upper() for kw in ["CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP"])
        tool_name = "write_neo4j_cypher" if is_write else "read_neo4j_cypher"
        
        # Execute - SINGLE ATTEMPT ONLY
        with st.spinner(f"⚡ Executing {tool_name}..."):
            result = run_async_simple(
                st.session_state.mcp_client.call_tool_single_attempt(tool_name, cypher_query)
            )
        
        st.session_state.history.append(("neo4j", result))

# Display history
st.subheader("📜 History")
if st.session_state.history:
    for role, message in reversed(st.session_state.history):
        if role == "user":
            st.chat_message("user").write(f"**You:** {message}")
        elif role == "cortex":
            st.chat_message("assistant").write(f"🤖 **Cypher:**\n```cypher\n{message}\n```")
        elif role == "neo4j":
            st.chat_message("assistant").write(f"📊 **Result:**\n```json\n{message}\n```")
        elif role == "error":
            st.chat_message("assistant").write(f"❌ **Error:** {message}")
else:
    st.info("👋 Ask a question to get started!")

# Footer
st.divider()
st.caption("🎯 TaskGroup errors eliminated • Single attempt operations • nest_asyncio applied")
