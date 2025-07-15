# ui.py

import streamlit as st
import asyncio
import json
import uuid
import requests
import threading
import urllib3
import time
from asyncio import run_coroutine_threadsafe
from threading import Thread
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcpserver import run_mcp_server

# Cortex Config
CORTEX_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
APP_ID = "edadip"
APLCTN_CD = "edagnai"
MODEL = "llama3.1-70b"
SYS_MSG = "You are a powerful AI assistant. Provide accurate, concise answers based on context."

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create a dedicated event loop for async operations
@st.cache_resource(show_spinner=False)
def create_event_loop():
    """Create a dedicated event loop in a worker thread."""
    loop = asyncio.new_event_loop()
    thread = Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop, thread

EVENT_LOOP, WORKER_THREAD = create_event_loop()

def run_async(coroutine):
    """Run a coroutine in the worker thread and return the result."""
    try:
        return run_coroutine_threadsafe(coroutine, EVENT_LOOP).result(timeout=30)
    except Exception as e:
        st.error(f"Async operation failed: {str(e)}")
        return None

# Start MCP server once
@st.cache_resource(show_spinner=False)
def start_mcp_server():
    """Start the MCP server in a separate thread."""
    def server_runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_mcp_server())
        except Exception as e:
            st.error(f"MCP server failed to start: {e}")
    
    thread = threading.Thread(target=server_runner, daemon=True)
    thread.start()
    
    # Wait a bit for server to start
    time.sleep(2)
    return True

# Initialize MCP server
MCP_SERVER_STARTED = start_mcp_server()

# Streamlit UI
st.set_page_config(page_title="Neo4j + Cortex", page_icon="🧠")
st.title("🧠 Neo4j Cypher + Cortex LLM Chat")

if "history" not in st.session_state:
    st.session_state.history = []

def call_cortex_llm(prompt: str) -> str:
    """Call Cortex LLM with proper error handling."""
    payload = {
        "query": {
            "aplctn_cd": APLCTN_CD,
            "app_id": APP_ID,
            "api_key": API_KEY,
            "method": "cortex",
            "model": MODEL,
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
        "Authorization": f'Snowflake Token="{API_KEY}"'
    }

    try:
        res = requests.post(CORTEX_URL, headers=headers, json=payload, verify=False, timeout=30)
        res.raise_for_status()
        raw = res.text
        return raw.split("end_of_stream")[0].strip() if "end_of_stream" in raw else raw.strip()
    except requests.exceptions.RequestException as e:
        return f"❌ Cortex API error: {e}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"

async def call_mcp_async(tool_name: str, query: str) -> str:
    """Call MCP tool with proper async handling and error recovery."""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            # Use SSE client to connect to MCP server
            async with sse_client("http://localhost:8000/messages/") as sse:
                async with ClientSession(*sse) as session:
                    await session.initialize()
                    
                    # Get available tools
                    tools = await session.list_tools()
                    tool = None
                    for t in tools.tools:
                        if t.name == tool_name:
                            tool = t
                            break
                    
                    if not tool:
                        return f"❌ Tool '{tool_name}' not found. Available tools: {[t.name for t in tools.tools]}"
                    
                    # Call the tool
                    result = await session.call_tool(tool, {"query": query})
                    
                    # Extract result text
                    if result.content and len(result.content) > 0:
                        return result.content[0].text
                    else:
                        return "✅ Tool executed successfully but returned no content."
                        
        except ConnectionError as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue
            return f"❌ Connection error after {max_retries} attempts: {e}"
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
            return f"❌ MCP error after {max_retries} attempts: {e}"
    
    return f"❌ Failed to execute tool after {max_retries} attempts"

def call_mcp(tool_name: str, query: str) -> str:
    """Wrapper function to call MCP from Streamlit."""
    return run_async(call_mcp_async(tool_name, query))

# Main UI
user_query = st.chat_input("Ask a Neo4j question...")

if user_query:
    st.session_state.history.append(("user", user_query))

    # Step 1: Generate Cypher query using Cortex
    with st.spinner("💡 Cortex generating Cypher..."):
        cypher_prompt = f"""
        Generate a Neo4j Cypher query for: {user_query}
        
        Please provide only the Cypher query without any explanation or formatting.
        Make sure the query is valid and follows Neo4j syntax.
        """
        cypher = call_cortex_llm(cypher_prompt)
    
    if cypher.startswith("❌"):
        st.error("Failed to generate Cypher query")
        st.session_state.history.append(("error", cypher))
    else:
        st.session_state.history.append(("llm", cypher))

        # Step 2: Determine tool type
        tool = "read_neo4j_cypher" if "MATCH" in cypher.upper() else "write_neo4j_cypher"

        # Step 3: Execute MCP tool
        with st.spinner(f"⚙️ Running {tool}..."):
            result = call_mcp(tool, cypher)
        
        if result:
            st.session_state.history.append(("neo4j", result))
        else:
            st.session_state.history.append(("error", "Failed to execute MCP tool"))

# Display chat history
st.subheader("Chat History")
for role, msg in reversed(st.session_state.history):
    if role == "user":
        st.chat_message("user").markdown(f"**You:** {msg}")
    elif role == "llm":
        st.chat_message("assistant").markdown(f"💡 **Cortex Cypher:**\n```cypher\n{msg}\n```")
    elif role == "neo4j":
        st.chat_message("assistant").markdown(f"📦 **Neo4j Result:**\n```json\n{msg}\n```")
    elif role == "error":
        st.chat_message("assistant").markdown(f"❌ **Error:** {msg}")

# Sidebar with controls
st.sidebar.header("Controls")
if st.sidebar.button("Clear History"):
    st.session_state.history = []
    st.rerun()

# Server status
st.sidebar.header("Server Status")
if MCP_SERVER_STARTED:
    st.sidebar.success("✅ MCP Server Running")
else:
    st.sidebar.error("❌ MCP Server Failed to Start")

# Debug info
with st.sidebar.expander("Debug Info"):
    st.write(f"Event Loop: {EVENT_LOOP}")
    st.write(f"Worker Thread: {WORKER_THREAD}")
    st.write(f"History Length: {len(st.session_state.history)}")

# Health check
if st.sidebar.button("Test MCP Connection"):
    with st.spinner("Testing MCP connection..."):
        test_result = call_mcp("read_neo4j_cypher", "MATCH (n) RETURN count(n) as total_nodes LIMIT 1")
        if test_result and not test_result.startswith("❌"):
            st.sidebar.success("✅ MCP Connection OK")
        else:
            st.sidebar.error(f"❌ MCP Connection Failed: {test_result}")
