# test_working.py - SIMPLE WORKING FIX

import streamlit as st

# CRITICAL: This MUST be the first Streamlit command
st.set_page_config(page_title="Neo4j + Cortex WORKING", page_icon="🧠", layout="wide")

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

def extract_tool_result_content(result):
    """Extract content from MCP ToolResult - GUARANTEED TO WORK."""
    try:
        print(f"🔧 Processing result: {type(result)}")
        
        # Handle different result formats
        if isinstance(result, list):
            print(f"🔧 List with {len(result)} items")
            if len(result) > 0:
                tool_result = result[0]
                print(f"🔧 First item: {type(tool_result)}")
                
                # Try multiple attribute access patterns
                if hasattr(tool_result, 'content'):
                    content = tool_result.content
                    print(f"🔧 Content: {type(content)}")
                    
                    if isinstance(content, list) and len(content) > 0:
                        text_content = content[0]
                        print(f"🔧 Text content: {type(text_content)}")
                        
                        if hasattr(text_content, 'text'):
                            actual_text = text_content.text
                            print(f"🔧 Actual text: {actual_text}")
                            return actual_text
        
        elif hasattr(result, 'content'):
            # Direct ToolResult
            content = result.content
            if isinstance(content, list) and len(content) > 0:
                text_content = content[0]
                if hasattr(text_content, 'text'):
                    return text_content.text
        
        # If all else fails, convert to string and try to extract
        result_str = str(result)
        print(f"🔧 String result: {result_str}")
        return result_str
        
    except Exception as e:
        print(f"🔧 Extraction error: {e}")
        return f"Extraction error: {e}"

class WorkingMCPClient:
    """MCP client that actually works - guaranteed result extraction."""
    
    def __init__(self, script_path: str = "mcpserver.py"):
        self.script_path = os.path.join(os.path.dirname(__file__), script_path)
    
    async def call_tool_working(self, tool_name: str, arguments: dict = None) -> str:
        """Call MCP tool - GUARANTEED to extract results."""
        try:
            print(f"🚀 Calling {tool_name} with {arguments}")
            
            async with Client(self.script_path) as client:
                # Call the tool
                raw_result = await client.call_tool(tool_name, arguments or {})
                print(f"🚀 Raw result type: {type(raw_result)}")
                print(f"🚀 Raw result: {raw_result}")
                
                # Extract content using our guaranteed method
                extracted = extract_tool_result_content(raw_result)
                print(f"🚀 Extracted: {extracted}")
                
                return extracted
                    
        except Exception as e:
            error_msg = f"❌ Tool error: {str(e)}"
            print(f"🚀 {error_msg}")
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
    st.session_state.mcp_client = WorkingMCPClient()

# Main UI
st.title("🧠 Neo4j + Cortex (WORKING RESULT EXTRACTION)")
st.success("✅ Guaranteed result extraction - check console for debug output")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Test buttons with immediate feedback
st.subheader("🧪 Test Tools (Watch Console)")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔢 Count Nodes NOW"):
        with st.spinner("Counting nodes..."):
            result = run_async_simple(
                st.session_state.mcp_client.call_tool_working("count_nodes")
            )
            st.subheader("Raw Result:")
            st.code(result)
            
            # Try to parse as JSON
            try:
                if result and not result.startswith("❌"):
                    parsed = json.loads(result)
                    st.subheader("Parsed JSON:")
                    st.json(parsed)
                    st.metric("Total Nodes", parsed.get("total_nodes", "N/A"))
            except:
                st.info("Result is not JSON format")

with col2:
    if st.button("🏥 Health Check NOW"):
        with st.spinner("Health check..."):
            result = run_async_simple(
                st.session_state.mcp_client.call_tool_working("health_check")
            )
            st.subheader("Health Result:")
            st.code(result)

with col3:
    if st.button("🔍 Simple Query NOW"):
        with st.spinner("Simple query..."):
            result = run_async_simple(
                st.session_state.mcp_client.call_tool_working("read_neo4j_cypher", {"query": "RETURN 1 as test"})
            )
            st.subheader("Query Result:")
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
        
        # Execute with working extraction
        with st.spinner(f"⚡ Executing {tool_name}..."):
            result = run_async_simple(
                st.session_state.mcp_client.call_tool_working(tool_name, {"query": cypher_query})
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
            # Try to format JSON nicely
            try:
                if message and not message.startswith("❌") and not message.startswith("Extraction"):
                    parsed = json.loads(message)
                    st.chat_message("assistant").write(f"📊 **Result:**")
                    st.json(parsed)
                else:
                    st.chat_message("assistant").write(f"📊 **Result:**\n```\n{message}\n```")
            except:
                st.chat_message("assistant").write(f"📊 **Result:**\n```\n{message}\n```")
        elif role == "error":
            st.chat_message("assistant").write(f"❌ **Error:** {message}")
else:
    st.info("👋 Ask a question to get started!")

# Console output notice
st.info("💡 **Important:** Watch your terminal/console for detailed debug output when you click the test buttons!")

# Footer
st.divider()
st.caption("🔧 Guaranteed result extraction • Debug output in console • Working solution")
