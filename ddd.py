# test_fixed.py - Fixed result extraction

import streamlit as st

# CRITICAL: This MUST be the first Streamlit command
st.set_page_config(page_title="Neo4j + Cortex FIXED", page_icon="🧠", layout="wide")

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

class FixedStdioMCPClient:
    """Fixed MCP client with proper result extraction."""
    
    def __init__(self, script_path: str = "mcpserver.py"):
        self.script_path = os.path.join(os.path.dirname(__file__), script_path)
    
    def extract_content(self, result) -> str:
        """Extract actual content from MCP result."""
        try:
            # Debug logging
            print(f"🔍 DEBUG - Result type: {type(result)}")
            print(f"🔍 DEBUG - Result repr: {repr(result)}")
            
            # Handle list of ToolResult objects
            if isinstance(result, list):
                print(f"🔍 DEBUG - List with {len(result)} items")
                if len(result) > 0:
                    first_item = result[0]
                    print(f"🔍 DEBUG - First item type: {type(first_item)}")
                    
                    # Check if it's a ToolResult object
                    if hasattr(first_item, 'content'):
                        print(f"🔍 DEBUG - Has content attribute")
                        if first_item.content and len(first_item.content) > 0:
                            content_item = first_item.content[0]
                            print(f"🔍 DEBUG - Content item type: {type(content_item)}")
                            if hasattr(content_item, 'text'):
                                actual_text = content_item.text
                                print(f"🔍 DEBUG - Extracted text: {actual_text[:100]}...")
                                return actual_text if actual_text else "✅ No content"
                    
                    # Fallback: try to get string representation
                    return str(first_item)
            
            # Handle single ToolResult object
            elif hasattr(result, 'content'):
                print(f"🔍 DEBUG - Single result with content")
                if result.content and len(result.content) > 0:
                    content_item = result.content[0]
                    if hasattr(content_item, 'text'):
                        return content_item.text if content_item.text else "✅ No content"
            
            # Fallback
            return str(result) if result else "✅ Empty result"
            
        except Exception as e:
            print(f"🔍 DEBUG - Extraction error: {e}")
            return f"❌ Content extraction error: {e}"
    
    async def call_tool_fixed(self, tool_name: str, arguments: dict = None) -> str:
        """Call MCP tool with fixed result extraction."""
        try:
            print(f"🔧 Calling tool: {tool_name} with args: {arguments}")
            
            async with Client(self.script_path) as client:
                # Call the tool
                result = await client.call_tool(tool_name, arguments or {})
                
                # Extract the actual content
                extracted_content = self.extract_content(result)
                
                print(f"🔧 Extracted content: {extracted_content[:200]}...")
                return extracted_content
                    
        except Exception as e:
            error_msg = f"❌ Tool call error: {str(e)}"
            print(f"🔧 {error_msg}")
            return error_msg

# Simple async runner
def run_async_simple(coro):
    """Simple async runner."""
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
    st.session_state.mcp_client = FixedStdioMCPClient()

# Main UI
st.title("🧠 Neo4j + Cortex (FIXED Result Extraction)")
st.success("✅ Fixed ToolResult object extraction")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Debug section
with st.expander("🔍 Debug Console"):
    st.write("Check the terminal/console for debug output when you run queries.")
    if st.button("Clear Debug History"):
        st.session_state.history = []
        st.rerun()

# Test buttons
st.subheader("🧪 Test Fixed Extraction")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Test Count Nodes"):
        with st.spinner("Testing count nodes..."):
            result = run_async_simple(
                st.session_state.mcp_client.call_tool_fixed("count_nodes")
            )
            st.code(result)

with col2:
    if st.button("Test Health Check"):
        with st.spinner("Testing health check..."):
            result = run_async_simple(
                st.session_state.mcp_client.call_tool_fixed("health_check")
            )
            st.code(result)

with col3:
    if st.button("Test Simple Query"):
        with st.spinner("Testing simple query..."):
            result = run_async_simple(
                st.session_state.mcp_client.call_tool_fixed("read_neo4j_cypher", {"query": "RETURN 1 as test"})
            )
            st.code(result)

# Main chat input
st.subheader("💬 Chat")
user_query = st.chat_input("Ask a Neo4j question...")

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
        
        # Execute with fixed extraction
        with st.spinner(f"⚡ Executing {tool_name}..."):
            result = run_async_simple(
                st.session_state.mcp_client.call_tool_fixed(tool_name, {"query": cypher_query})
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
st.caption("🔧 Fixed ToolResult extraction • Debug output in console")
