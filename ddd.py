# uix.py - Debug version using the debug MCP server

import streamlit as st

# MUST be first Streamlit command
st.set_page_config(page_title="Neo4j Debug", page_icon="🔧", layout="wide")

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

def extract_mcp_result(result):
    """Extract content from MCP result with detailed logging."""
    print(f"\n🔍 EXTRACTING MCP RESULT:")
    print(f"  Type: {type(result)}")
    print(f"  Repr: {repr(result)[:200]}...")
    
    try:
        # Handle list results
        if isinstance(result, list):
            print(f"  📋 List with {len(result)} items")
            if len(result) > 0:
                first_item = result[0]
                print(f"  📋 First item type: {type(first_item)}")
                
                # Check if it has content attribute
                if hasattr(first_item, 'content'):
                    content = first_item.content
                    print(f"  📋 Content type: {type(content)}")
                    print(f"  📋 Content: {content}")
                    
                    if isinstance(content, list) and len(content) > 0:
                        content_item = content[0]
                        print(f"  📋 Content item type: {type(content_item)}")
                        
                        if hasattr(content_item, 'text'):
                            text = content_item.text
                            print(f"  ✅ FOUND TEXT: {text}")
                            return text
                        
                        # Check other possible attributes
                        for attr in ['data', 'value', 'result']:
                            if hasattr(content_item, attr):
                                value = getattr(content_item, attr)
                                print(f"  📋 Found .{attr}: {value}")
                                return str(value)
                
                # Check if the item itself has text
                if hasattr(first_item, 'text'):
                    text = first_item.text
                    print(f"  ✅ DIRECT TEXT: {text}")
                    return text
        
        # Handle direct object
        elif hasattr(result, 'content'):
            content = result.content
            if isinstance(content, list) and len(content) > 0:
                content_item = content[0]
                if hasattr(content_item, 'text'):
                    text = content_item.text
                    print(f"  ✅ DIRECT OBJECT TEXT: {text}")
                    return text
        
        # Fallback to string conversion
        result_str = str(result)
        print(f"  ⚠️  FALLBACK STRING: {result_str}")
        return result_str
        
    except Exception as e:
        error_msg = f"❌ Extraction error: {e}"
        print(f"  {error_msg}")
        return error_msg

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

class DebugMCPClient:
    def __init__(self, script_path="mcpserver_debug.py"):
        self.script_path = os.path.join(os.path.dirname(__file__), script_path)
    
    async def call_tool_debug(self, tool_name: str, arguments: dict = None) -> str:
        print(f"\n🚀 CALLING TOOL: {tool_name}")
        print(f"🚀 ARGUMENTS: {arguments}")
        print(f"🚀 SCRIPT PATH: {self.script_path}")
        
        try:
            async with Client(self.script_path) as client:
                print(f"🚀 Client connected")
                
                # Call the tool
                raw_result = await client.call_tool(tool_name, arguments or {})
                print(f"🚀 Raw result received: {type(raw_result)}")
                
                # Extract content
                extracted = extract_mcp_result(raw_result)
                print(f"🚀 Final extracted result: {extracted}")
                
                return extracted
                
        except Exception as e:
            error_msg = f"❌ Tool call error: {str(e)}"
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
    st.session_state.mcp_client = DebugMCPClient()

if "history" not in st.session_state:
    st.session_state.history = []

# UI
st.title("🔧 Neo4j Debug Version")
st.error("🔍 Using DEBUG MCP server with detailed logging - Watch console!")

# File check
debug_server_path = os.path.join(os.path.dirname(__file__), "mcpserver_debug.py")
if not os.path.exists(debug_server_path):
    st.error(f"❌ Debug server not found: {debug_server_path}")
    st.info("Make sure mcpserver_debug.py is in the same directory")
else:
    st.success(f"✅ Debug server found: {debug_server_path}")

# Test section
st.subheader("🧪 Debug Tests")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🧪 Simple Test"):
        st.write("**Testing simple_test tool...**")
        with st.spinner("Running simple test..."):
            result = run_async_safe(
                st.session_state.mcp_client.call_tool_debug("simple_test")
            )
        
        st.write("**Result:**")
        st.code(result)
        
        # Try to parse as JSON
        try:
            if result and not result.startswith("❌"):
                parsed = json.loads(result)
                st.json(parsed)
        except:
            st.write("Not JSON format")

with col2:
    if st.button("🏥 Health Check"):
        st.write("**Testing health_check tool...**")
        with st.spinner("Health check..."):
            result = run_async_safe(
                st.session_state.mcp_client.call_tool_debug("health_check")
            )
        
        st.write("**Result:**")
        st.code(result)
        
        try:
            if result and not result.startswith("❌"):
                parsed = json.loads(result)
                st.json(parsed)
        except:
            st.write("Not JSON format")

with col3:
    if st.button("🔢 Count Nodes"):
        st.write("**Testing count_nodes tool...**")
        with st.spinner("Counting nodes..."):
            result = run_async_safe(
                st.session_state.mcp_client.call_tool_debug("count_nodes")
            )
        
        st.write("**Result:**")
        st.code(result)
        
        try:
            if result and not result.startswith("❌"):
                parsed = json.loads(result)
                st.json(parsed)
                if "total_nodes" in parsed:
                    st.metric("Total Nodes", parsed["total_nodes"])
        except:
            st.write("Not JSON format")

with col4:
    if st.button("🔍 Test Query"):
        st.write("**Testing read_neo4j_cypher tool...**")
        with st.spinner("Test query..."):
            result = run_async_safe(
                st.session_state.mcp_client.call_tool_debug("read_neo4j_cypher", {"query": "RETURN 1 as test"})
            )
        
        st.write("**Result:**")
        st.code(result)
        
        try:
            if result and not result.startswith("❌"):
                parsed = json.loads(result)
                st.json(parsed)
        except:
            st.write("Not JSON format")

# Direct test option
st.subheader("🔧 Direct Server Test")
if st.button("🔬 Run Direct MCP Test"):
    st.info("Running direct MCP test script - check your console for detailed output!")
    with st.spinner("Running direct test..."):
        import subprocess
        import sys
        try:
            # Run the direct test script
            result = subprocess.run([sys.executable, "test_mcp_direct.py"], 
                                  capture_output=True, text=True, timeout=30)
            st.code(result.stdout)
            if result.stderr:
                st.error(result.stderr)
        except Exception as e:
            st.error(f"Direct test failed: {e}")

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
                st.session_state.mcp_client.call_tool_debug(tool_name, {"query": cypher_query})
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
            if message and not message.startswith("❌"):
                parsed = json.loads(message)
                st.json(parsed)
            else:
                st.code(message)
        except:
            st.code(message)
    elif role == "error":
        st.chat_message("assistant").write(f"❌ **Error:** {message}")

# Instructions
st.info("💡 **How to debug:**\n1. First save mcpserver_debug.py and test_mcp_direct.py\n2. Run direct test: `python test_mcp_direct.py`\n3. Then use this UI and watch console output")

st.divider()
st.caption("🔧 Debug version • Detailed logging • Watch console for analysis")
