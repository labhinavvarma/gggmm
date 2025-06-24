import streamlit as st
st.set_page_config(page_title="Milliman MCP Chat", page_icon="🤖")

import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# Title
st.title("🤖 Milliman MCP Chat")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Create and cache the MCP client using Streamable HTTP
@st.cache_resource
def get_client():
    transport = StreamableHttpTransport(url="http://localhost:8000/sse")
    return Client(transport)

client = get_client()

# Async helper to call tool
async def call_tool(tool_name: str, args: dict):
    async with client:
        return await client.call_tool(tool_name, args)

# Form inputs
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

if submitted:
    args = {
        "first_name": first_name,
        "last_name": last_name,
        "ssn": ssn,
        "date_of_birth": dob,
        "gender": gender,
        "zip_code": zip_code
    }
    st.session_state.history.append(("user", f"Called `{tool}`"))
    try:
        # Run the async call synchronously
        result = asyncio.run(call_tool(tool, args))
        st.session_state.history.append(("tool", f"{tool} output: {result}"))
    except Exception as e:
        st.session_state.history.append(("error", f"Error: {e}"))

# Display chat history
st.divider()
for role, msg in st.session_state.history:
    icon = {"user": "🧑", "tool": "⚙️", "error": "❌"}.get(role, "")
    st.markdown(f"{icon} {msg}")
