import streamlit as st
st.set_page_config(page_title="Milliman MCP Chat", page_icon="🤖")

import asyncio
import threading
from fastmcp import Client
from fastmcp.client.transports import SSETransport

# Title
st.title("🤖 Milliman MCP Chat")

# State
if "history" not in st.session_state:
    st.session_state.history = []
if "client" not in st.session_state:
    st.session_state.client = None

# Initialize client once
def init_client():
    transport = SSETransport("http://localhost:8000/sse")
    st.session_state.client = Client(transport)
    # Connect
    asyncio.run(st.session_state.client.__aenter__())

init_client_thread = threading.Thread(target=init_client, daemon=True)
init_client_thread.start()

# Ensure client is ready before continuing
while st.session_state.client is None or init_client_thread.is_alive():
    st.write("🚧 Connecting to MCP server...")
    asyncio.sleep(0.1)

# Form for input
with st.form("user_form"):
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    ssn = st.text_input("SSN")
    dob = st.text_input("DOB (YYYY-MM-DD)")
    gender = st.selectbox("Gender", ["M", "F", "O"])
    zip_code = st.text_input("Zip Code")
    tool = st.selectbox(
        "Tool",
        ["medical_submit", "pharmacy_submit", "mcid_search", "get_all_data"]
    )
    submitted = st.form_submit_button("Submit")

async def call_tool(tool_name, args):
    return await st.session_state.client.call_tool(tool_name, args)

# On submit
if submitted:
    args = {
        "first_name": first_name,
        "last_name": last_name,
        "ssn": ssn,
        "date_of_birth": dob,
        "gender": gender,
        "zip_code": zip_code
    }
    st.session_state.history.append(("user", f"Called {tool}"))
    try:
        result = asyncio.run(call_tool(tool, args))
        st.session_state.history.append(("tool", f"{tool} output: {result}"))
    except Exception as e:
        st.session_state.history.append(("error", f"Error: {e}"))

# Display history
st.divider()
for role, msg in st.session_state.history:
    icon = {"user": "🧑", "tool": "⚙️", "error": "❌"}.get(role, "")
    st.markdown(f"{icon} {msg}")
