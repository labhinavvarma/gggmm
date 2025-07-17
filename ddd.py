import streamlit as st
import requests
import uuid

# Try to import config, fallback to hardcoded values
try:
    from config import SERVER_CONFIG
    APP_PORT = SERVER_CONFIG["app_port"]
except ImportError:
    APP_PORT = 8081

st.set_page_config(page_title="Neo4j LLM Agent", page_icon="🧠")

st.title("🧠 Neo4j LangGraph MCP Agent (Cortex LLM)")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.form("chat_form", clear_on_submit=True):
    user_query = st.text_input("Ask a question:", key="chat_input", placeholder="e.g. How many nodes are in the graph?")
    submitted = st.form_submit_button("Send")

if submitted and user_query:
    session_id = str(uuid.uuid4())
    payload = {
        "question": user_query,
        "session_id": session_id
    }
    
    try:
        response = requests.post(f"http://localhost:{APP_PORT}/chat", json=payload)
        if response.status_code == 200:
            result = response.json()
            agent_reply = f"**Tool:** {result['tool']}\n\n**Query:** {result['query']}\n\n**Trace:** {result['trace']}\n\n**Answer:** {result['answer']}"
        else:
            agent_reply = f"❌ Error {response.status_code}: {response.text}"
        
        st.session_state.messages.append(("user", user_query))
        st.session_state.messages.append(("bot", agent_reply))
    except Exception as e:
        st.error(f"❌ Request failed: {str(e)}")

st.divider()

for role, message in reversed(st.session_state.messages):
    if role == "user":
        st.markdown(f"🧑 **You:** {message}")
    else:
        st.markdown(f"🤖 **Agent:** {message}")

# Add a sidebar with some info
st.sidebar.title("🔧 System Info")
st.sidebar.markdown(f"**App Port:** {APP_PORT}")
st.sidebar.markdown("**Services:**")
st.sidebar.markdown("- MCP Server: Port 8000")
st.sidebar.markdown("- Main App: Port 8081")
st.sidebar.markdown("- Streamlit UI: Port 8501")

# Add some example queries
st.sidebar.title("📝 Example Queries")
examples = [
    "How many nodes are in the graph?",
    "Show me the schema",
    "List all Person nodes",
    "Find nodes with most relationships",
    "Create a new Person node named John",
    "Count all relationships"
]

for example in examples:
    if st.sidebar.button(example, key=f"example_{example}"):
        st.session_state.example_query = example
        st.rerun()

# Handle example query selection
if hasattr(st.session_state, 'example_query'):
    st.info(f"Example query selected: {st.session_state.example_query}")
    del st.session_state.example_query
