# streamlit_app.py

import streamlit as st
import asyncio
import json
import uuid
import requests
import threading
import urllib3
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

# Start MCP server once
if "mcp_started" not in st.session_state:
    def start_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_mcp_server())

    thread = threading.Thread(target=start_server, daemon=True)
    thread.start()
    st.session_state["mcp_started"] = True

# Streamlit UI
st.set_page_config(page_title="Neo4j + Cortex", page_icon="🧠")
st.title("🧠 Neo4j Cypher + Cortex LLM Chat")

if "history" not in st.session_state:
    st.session_state.history = []

user_query = st.chat_input("Ask a Neo4j question...")

def call_cortex_llm(prompt: str) -> str:
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
        res = requests.post(CORTEX_URL, headers=headers, json=payload, verify=False)
        res.raise_for_status()
        raw = res.text
        return raw.split("end_of_stream")[0].strip() if "end_of_stream" in raw else raw.strip()
    except Exception as e:
        return f"❌ Cortex error: {e}"

async def call_mcp(tool_name: str, query: str) -> str:
    try:
        async with sse_client("http://localhost:8000/messages/") as sse:
            async with ClientSession(*sse) as session:
                await session.initialize()
                tool = next(t for t in (await session.list_tools()).tools if t.name == tool_name)
                result = await session.query_tool(tool, {"query": query})
                return result.content[0].text if result.content else "✅ No results."
    except Exception as e:
        return f"❌ MCP error: {e}"

# Run the chat flow
if user_query:
    st.session_state.history.append(("user", user_query))

    with st.spinner("💡 Cortex generating Cypher..."):
        cypher = call_cortex_llm(f"Write Cypher to: {user_query}")
    st.session_state.history.append(("llm", cypher))

    tool = "read_neo4j_cypher" if "MATCH" in cypher.upper() else "write_neo4j_cypher"

    with st.spinner(f"⚙️ Running {tool}..."):
        result = asyncio.get_event_loop().run_until_complete(call_mcp(tool, cypher))
    st.session_state.history.append(("neo4j", result))

# Show history
for role, msg in reversed(st.session_state.history):
    if role == "user":
        st.chat_message("user").markdown(msg)
    elif role == "llm":
        st.chat_message("assistant").markdown(f"💡 **Cortex Suggestion:**\n```\n{msg}\n```")
    else:
        st.chat_message("assistant").markdown(f"📦 **Neo4j Result:**\n```\n{msg}\n```")
