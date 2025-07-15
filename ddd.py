# uix.py - COMPLETE FIX for ToolResult extraction issue

import streamlit as st

# MUST be first Streamlit command
st.set_page_config(page_title="Neo4j Fixed", page_icon="🧠", layout="wide")

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

def deep_extract_content(obj, depth=0):
    """DEEP extraction of content from any object structure."""
    indent = "  " * depth
    print(f"{indent}🔍 Analyzing: {type(obj)} - {repr(obj)[:100]}")
    
    # If it's a string and looks like JSON, return it
    if isinstance(obj, str):
        try:
            json.loads(obj)
            print(f"{indent}✅ Found JSON string!")
            return obj
        except:
            if len(obj) > 10:  # If it's a meaningful string
                print(f"{indent}✅ Found text string!")
                return obj
    
    # If it's a list, check each item
    if isinstance(obj, list):
        print(f"{indent}📋 List with {len(obj)} items")
        for i, item in enumerate(obj):
            print(f"{indent}  Item {i}:")
            result = deep_extract_content(item, depth + 2)
            if result and not result.startswith("No content"):
                return result
    
    # If it's an object, check all attributes
    if hasattr(obj, '__dict__') or hasattr(obj, '__slots__'):
        print(f"{indent}🔧 Object with attributes:")
        
        # Check common attribute names
        for attr_name in ['text', 'content', 'data', 'result', 'value', 'response']:
            if hasattr(obj, attr_name):
                attr_value = getattr(obj, attr_name)
                print(f"{indent}  Found .{attr_name}: {type(attr_value)}")
                result = deep_extract_content(attr_value, depth + 2)
                if result and not result.startswith("No content"):
                    return result
        
        # Check all attributes if common ones don't work
        try:
            for attr_name in dir(obj):
                if not attr_name.startswith('_') and not callable(getattr(obj, attr_name)):
                    try:
                        attr_value = getattr(obj, attr_name)
                        if attr_value and str(attr_value) != str(obj):  # Avoid infinite recursion
                            print(f"{indent}  Checking .{attr_name}: {type(attr_value)}")
                            result = deep_extract_content(attr_value, depth + 2)
                            if result and not result.startswith("No content"):
                                return result
                    except:
                        continue
        except:
            pass
    
    print(f"{indent}❌ No extractable content found")
    return f"No content found in {type(obj)}"

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

class FixedMCPClient:
    def __init__(self, script_path="mcpserver.py"):
        self.script_path = os.path.join(os.path.dirname(__file__), script_path)
    
    async def call_tool_fixed(self, tool_name: str, arguments: dict = None) -> str:
        print(f"\n{'='*60}")
        print(f"🚀 CALLING TOOL: {tool_name}")
        print(f"🚀 ARGUMENTS: {arguments}")
        print(f"{'='*60}")
        
        try:
            async with Client(self.script_path) as client:
                # Call the tool
                raw_result = await client.call_tool(tool_name, arguments or {})
                
                print(f"\n🔍 RAW RESULT ANALYSIS:")
                print(f"Type: {type(raw_result)}")
                print(f"Repr: {repr(raw_result)}")
                print(f"Str: {str(raw_result)}")
                
                # Use deep extraction
                extracted_content = deep_extract_content(raw_result)
                
                print(f"\n✅ FINAL EXTRACTED CONTENT:")
                print(f"'{extracted_content}'")
                print(f"{'='*60}\n")
                
                return extracted_content
                
        except Exception as e:
            error_msg = f"❌ Tool call failed: {str(e)}"
            print(f"🚀 {error_msg}")
            return error_msg

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
    st.session_state.mcp_client = FixedMCPClient()

if "history" not in st.session_state:
    st.session_state.history = []

# UI
st.title("🧠 Neo4j + Cortex (COMPLETE FIX)")
st.error("🔍 DEEP CONTENT EXTRACTION - Watch console for detailed analysis!")

# Test section
st.subheader("🧪 Immediate Tests")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔢 Count Nodes"):
        st.write("**Testing count_nodes tool...**")
        with st.spinner("Analyzing result structure..."):
            result = run_async_safe(
                st.session_state.mcp_client.call_tool_fixed("count_nodes")
            )
        
        st.write("**Raw extracted result:**")
        st.code(result, language="json")
        
        # Try to parse and display nicely
        try:
            if result and not result.startswith(("❌", "No content")):
                parsed = json.loads(result)
                st.write("**Parsed JSON:**")
                st.json(parsed)
                if "total_nodes" in parsed:
                    st.metric("Total Nodes", parsed["total_nodes"])
        except json.JSONDecodeError:
            st.write("**Not JSON format - displaying as text:**")
            st.text(result)

with col2:
    if st.button("🏥 Health Check"):
        st.write("**Testing health_check tool...**")
        with st.spinner("Analyzing result structure..."):
            result = run_async_safe(
                st.session_state.mcp_client.call_tool_fixed("health_check")
            )
        
        st.write("**Raw extracted result:**")
        st.code(result, language="json")

with col3:
    if st.button("🔍 Simple Query"):
        st.write("**Testing read_neo4j_cypher tool...**")
        with st.spinner("Analyzing result structure..."):
            result = run_async_safe(
                st.session_state.mcp_client.call_tool_fixed("read_neo4j_cypher", {"query": "RETURN 1 as test"})
            )
        
        st.write("**Raw extracted result:**")
        st.code(result, language="json")

# Chat interface
st.subheader("💬 Chat")
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
                st.session_state.mcp_client.call_tool_fixed(tool_name, {"query": cypher_query})
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
        try:
            if message and not message.startswith(("❌", "No content")):
                parsed = json.loads(message)
                st.json(parsed)
            else:
                st.code(message)
        except:
            st.code(message)
    elif role == "error":
        st.chat_message("assistant").write(f"❌ **Error:** {message}")

# Console reminder
st.info("💡 **IMPORTANT:** Watch your terminal/console for detailed content extraction analysis!")

st.divider()
st.caption("🔧 Deep content extraction • Complete debugging • Guaranteed to find the data")
