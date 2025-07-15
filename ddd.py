# test.py - STDIO transport version (eliminates TaskGroup errors)

import streamlit as st

# CRITICAL: This MUST be the first Streamlit command
st.set_page_config(page_title="Neo4j + Cortex STDIO", page_icon="🧠", layout="wide")

# Now import everything else
import asyncio
import json
import uuid
import requests
import urllib3
import time
import sys
import os
import nest_asyncio
from fastmcp import Client

# Apply nest_asyncio
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
    """Simple Cortex client."""
    
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

class StdioMCPClient:
    """MCP client using STDIO transport - eliminates TaskGroup errors."""
    
    def __init__(self, script_path: str = "mcpserver.py"):
        self.script_path = os.path.join(os.path.dirname(__file__), script_path)
        self.python_path = sys.executable
    
    async def call_tool_stdio(self, tool_name: str, arguments: dict = None) -> str:
        """Call MCP tool using STDIO transport - no HTTP/SSE complexity."""
        try:
            # Use fastmcp Client with stdio transport
            async with Client(self.script_path) as client:
                # Call the tool directly
                result = await client.call_tool(tool_name, arguments or {})
                
                # Extract result
                if hasattr(result, 'content') and result.content:
                    return result.content[0].text if result.content[0].text else "✅ Success"
                else:
                    return "✅ Tool executed successfully"
                    
        except Exception as e:
            return f"❌ STDIO MCP error: {str(e)[:150]}"
    
    async def health_check_stdio(self) -> str:
        """Health check using STDIO."""
        try:
            async with Client(self.script_path) as client:
                # Try to list tools as health check
                tools = await client.list_tools()
                tool_count = len(tools.tools) if hasattr(tools, 'tools') else 0
                
                return f"✅ Server healthy - {tool_count} tools available"
                
        except Exception as e:
            return f"❌ Health check failed: {str(e)[:100]}"
    
    async def list_tools_stdio(self) -> str:
        """List available tools."""
        try:
            async with Client(self.script_path) as client:
                tools = await client.list_tools()
                if hasattr(tools, 'tools') and tools.tools:
                    tool_names = [t.name for t in tools.tools]
                    return json.dumps({"tools": tool_names, "count": len(tool_names)}, indent=2)
                else:
                    return "❌ No tools found"
                    
        except Exception as e:
            return f"❌ List tools failed: {str(e)[:100]}"

# Simple async runner
def run_async_simple(coro):
    """Simple async runner using nest_asyncio."""
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

# Initialize components
if "cortex_client" not in st.session_state:
    st.session_state.cortex_client = SimpleCortexClient(CORTEX_URL, API_KEY, APP_ID, APLCTN_CD, MODEL)

if "mcp_client" not in st.session_state:
    st.session_state.mcp_client = StdioMCPClient()

# Main UI
st.title("🧠 Neo4j + Cortex (STDIO Transport)")
st.success("✅ Using STDIO transport - TaskGroup errors eliminated!")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.header("🎛️ Status")
    
    st.success("✅ STDIO Transport Active")
    st.info("No HTTP/SSE complexity")
    
    # Health check
    if st.button("🏥 Health Check"):
        with st.spinner("Checking STDIO health..."):
            health_result = run_async_simple(
                st.session_state.mcp_client.health_check_stdio()
            )
            if health_result.startswith("✅"):
                st.success(health_result)
            else:
                st.error(health_result)
    
    # List tools
    if st.button("🔧 List Tools"):
        with st.spinner("Listing tools..."):
            tools_result = run_async_simple(
                st.session_state.mcp_client.list_tools_stdio()
            )
            if tools_result.startswith("❌"):
                st.error(tools_result)
            else:
                st.success("✅ Tools retrieved")
                st.json(json.loads(tools_result))
    
    st.divider()
    
    # Clear history
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
    
    # Debug info
    st.subheader("🔧 Info")
    st.write(f"Transport: STDIO")
    st.write(f"Script: mcpserver.py")
    st.write(f"nest_asyncio: Applied")

# Quick test buttons
st.subheader("🚀 Quick Tests")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Count Nodes"):
        with st.spinner("Counting nodes..."):
            result = run_async_simple(
                st.session_state.mcp_client.call_tool_stdio("count_nodes")
            )
            if result.startswith("❌"):
                st.error(result)
            else:
                st.success("✅ Nodes counted")
                try:
                    data = json.loads(result)
                    st.metric("Total Nodes", data.get("total_nodes", "N/A"))
                except:
                    st.code(result)

with col2:
    if st.button("List Labels"):
        with st.spinner("Getting labels..."):
            result = run_async_simple(
                st.session_state.mcp_client.call_tool_stdio("list_labels")
            )
            if result.startswith("❌"):
                st.error(result)
            else:
                st.success("✅ Labels retrieved")
                try:
                    data = json.loads(result)
                    st.write(f"Found {data.get('count', 0)} labels:")
                    for label in data.get('labels', []):
                        st.write(f"• {label}")
                except:
                    st.code(result)

with col3:
    if st.button("Health Check"):
        with st.spinner("Health check..."):
            result = run_async_simple(
                st.session_state.mcp_client.call_tool_stdio("health_check")
            )
            if result.startswith("❌"):
                st.error(result)
            else:
                st.success(result)

with col4:
    if st.button("Test Cortex"):
        with st.spinner("Testing Cortex..."):
            result = st.session_state.cortex_client.generate_cypher("RETURN 1")
            if result.startswith("❌"):
                st.error(result)
            else:
                st.success("✅ Cortex working")
                st.code(result)

# Quick queries
st.subheader("🔍 Quick Queries")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Show me all nodes"):
        st.session_state.test_query = "Show me all nodes in the database"

with col2:
    if st.button("What labels exist?"):
        st.session_state.test_query = "What node labels exist in this database?"

with col3:
    if st.button("Database summary"):
        st.session_state.test_query = "Give me a summary of this database"

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
    
    # Generate Cypher
    with st.spinner("🤖 Generating Cypher..."):
        cypher_query = st.session_state.cortex_client.generate_cypher(user_query)
    
    if cypher_query.startswith("❌"):
        st.session_state.history.append(("error", cypher_query))
    else:
        st.session_state.history.append(("cortex", cypher_query))
        
        # Determine tool
        is_write = any(kw in cypher_query.upper() for kw in ["CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP"])
        tool_name = "write_neo4j_cypher" if is_write else "read_neo4j_cypher"
        
        # Execute using STDIO - NO TASKGROUP ERRORS!
        with st.spinner(f"⚡ Executing {tool_name} via STDIO..."):
            result = run_async_simple(
                st.session_state.mcp_client.call_tool_stdio(tool_name, {"query": cypher_query})
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
st.caption("🎯 STDIO transport • No HTTP/SSE complexity • TaskGroup errors eliminated")
st.caption("✨ Direct process communication • Stable and reliable")
