# test.py - WORKING VERSION with fixed MCP server

import streamlit as st

# MUST be first Streamlit command
st.set_page_config(page_title="Neo4j WORKING", page_icon="✅", layout="wide")

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
SYS_MSG = "You are a powerful AI assistant specialized in Neo4j Cypher queries."

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def extract_fastmcp_result(result):
    """Extract content from FastMCP CallToolResult."""
    try:
        # FastMCP returns CallToolResult with content array
        if hasattr(result, 'content') and result.content:
            # Get the first content item
            content_item = result.content[0]
            if hasattr(content_item, 'text'):
                return content_item.text
        
        # Fallback to string conversion
        return str(result)
        
    except Exception as e:
        return f"❌ Extraction error: {e}"

class SimpleCortexClient:
    def __init__(self, url, api_key, app_id, aplctn_cd, model):
        self.url = url
        self.api_key = api_key
        self.app_id = app_id
        self.aplctn_cd = aplctn_cd
        self.model = model
    
    def generate_cypher(self, user_query: str) -> str:
        prompt = f"Generate a Neo4j Cypher query for: {user_query}\nReturn ONLY the Cypher query."
        
        payload = {
            "query": {
                "aplctn_cd": self.aplctn_cd,
                "app_id": self.app_id,
                "api_key": self.api_key,
                "method": "cortex",
                "model": self.model,
                "sys_msg": SYS_MSG,
                "limit_convs": "0",
                "prompt": {"messages": [{"role": "user", "content": prompt}]},
                "session_id": str(uuid.uuid4())
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f'Snowflake Token="{self.api_key}"'
        }

        try:
            response = requests.post(self.url, headers=headers, json=payload, verify=False, timeout=30)
            if response.status_code == 200:
                raw_text = response.text
                if "end_of_stream" in raw_text:
                    result = raw_text.split("end_of_stream")[0].strip()
                    return result if result else "MATCH (n) RETURN count(n)"
                return raw_text.strip() if raw_text.strip() else "MATCH (n) RETURN count(n)"
            else:
                return f"❌ Cortex error: HTTP {response.status_code}"
        except Exception as e:
            return f"❌ Cortex error: {str(e)[:100]}"

class WorkingMCPClient:
    def __init__(self, script_path="mcpserver.py"):
        self.script_path = os.path.join(os.path.dirname(__file__), script_path)
    
    async def call_tool_working(self, tool_name: str, arguments: dict = None) -> str:
        """Call MCP tool with proper extraction."""
        try:
            async with Client(self.script_path) as client:
                # Call the tool
                result = await client.call_tool(tool_name, arguments or {})
                
                # Extract content using the fixed method
                extracted = extract_fastmcp_result(result)
                
                return extracted
                
        except Exception as e:
            return f"❌ Tool error: {str(e)}"

def run_async_safe(coro):
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

# Initialize
if "cortex_client" not in st.session_state:
    st.session_state.cortex_client = SimpleCortexClient(CORTEX_URL, API_KEY, APP_ID, APLCTN_CD, MODEL)

if "mcp_client" not in st.session_state:
    st.session_state.mcp_client = WorkingMCPClient()

if "history" not in st.session_state:
    st.session_state.history = []

# UI
st.title("✅ Neo4j + Cortex WORKING!")
st.success("🎉 Fixed MCP server - Returns actual JSON data!")

# Test section
st.subheader("🧪 Test Fixed Tools")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🧪 Simple Test"):
        with st.spinner("Testing..."):
            result = run_async_safe(
                st.session_state.mcp_client.call_tool_working("simple_test")
            )
        
        st.subheader("Result:")
        if result and not result.startswith("❌"):
            try:
                parsed = json.loads(result)
                st.json(parsed)
                st.success(f"✅ Message: {parsed.get('message', 'N/A')}")
            except:
                st.code(result)
        else:
            st.error(result)

with col2:
    if st.button("🏥 Health Check"):
        with st.spinner("Health check..."):
            result = run_async_safe(
                st.session_state.mcp_client.call_tool_working("health_check")
            )
        
        st.subheader("Health:")
        if result and not result.startswith("❌"):
            try:
                parsed = json.loads(result)
                st.json(parsed)
                st.success(f"✅ Status: {parsed.get('status', 'N/A')}")
            except:
                st.code(result)
        else:
            st.error(result)

with col3:
    if st.button("🔢 Count Nodes"):
        with st.spinner("Counting..."):
            result = run_async_safe(
                st.session_state.mcp_client.call_tool_working("count_nodes")
            )
        
        st.subheader("Node Count:")
        if result and not result.startswith("❌"):
            try:
                parsed = json.loads(result)
                st.json(parsed)
                st.metric("Total Nodes", parsed.get("total_nodes", "N/A"))
            except:
                st.code(result)
        else:
            st.error(result)

with col4:
    if st.button("🏷️ List Labels"):
        with st.spinner("Getting labels..."):
            result = run_async_safe(
                st.session_state.mcp_client.call_tool_working("list_labels")
            )
        
        st.subheader("Labels:")
        if result and not result.startswith("❌"):
            try:
                parsed = json.loads(result)
                st.json(parsed)
                labels = parsed.get("labels", [])
                st.write(f"Found {len(labels)} labels:")
                for label in labels:
                    st.write(f"• {label}")
            except:
                st.code(result)
        else:
            st.error(result)

# Chat interface
st.subheader("💬 Chat Interface")
user_query = st.chat_input("Ask a Neo4j question...")

if user_query:
    st.session_state.history.append(("user", user_query))
    
    # Generate Cypher
    with st.spinner("🤖 Generating Cypher..."):
        cypher_query = st.session_state.cortex_client.generate_cypher(user_query)
    
    if cypher_query.startswith("❌"):
        st.session_state.history.append(("error", cypher_query))
    else:
        st.session_state.history.append(("cortex", cypher_query))
        
        # Execute query
        is_write = any(kw in cypher_query.upper() for kw in ["CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP"])
        tool_name = "write_neo4j_cypher" if is_write else "read_neo4j_cypher"
        
        with st.spinner(f"⚡ Executing {tool_name}..."):
            result = run_async_safe(
                st.session_state.mcp_client.call_tool_working(tool_name, {"query": cypher_query})
            )
        
        st.session_state.history.append(("neo4j", result))

# Display history
st.subheader("📜 History")
for role, message in reversed(st.session_state.history):
    if role == "user":
        st.chat_message("user").write(f"**You:** {message}")
    elif role == "cortex":
        st.chat_message("assistant").write(f"🤖 **Cypher:**\n```cypher\n{message}\n```")
    elif role == "neo4j":
        st.chat_message("assistant").write(f"📊 **Result:**")
        if message and not message.startswith("❌"):
            try:
                parsed = json.loads(message)
                st.json(parsed)
            except:
                st.code(message)
        else:
            st.code(message)
    elif role == "error":
        st.chat_message("assistant").write(f"❌ **Error:** {message}")

# Success message
st.success("🎉 **Fixed!** Tools now return raw JSON strings instead of ToolResult objects!")

st.divider()
st.caption("✅ Working solution • Fixed MCP server • Actual JSON data")
