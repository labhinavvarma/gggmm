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
 
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
 
def send_query(query):
    """Send query to backend and handle response"""
    payload = {
        "question": query,
        "session_id": st.session_state.session_id
    }
   
    try:
        with st.spinner("Processing your query..."):
            response = requests.post(f"http://localhost:{APP_PORT}/chat", json=payload)
            
        if response.status_code == 200:
            result = response.json()
            agent_reply = f"**Tool:** {result['tool']}\n\n**Query:** {result['query']}\n\n**Answer:** {result['answer']}"
            
            # Add messages to session state
            st.session_state.messages.append(("user", query))
            st.session_state.messages.append(("bot", agent_reply))
            
            st.success("✅ Query processed successfully!")
            return True
        else:
            error_msg = f"❌ Error {response.status_code}: {response.text}"
            st.session_state.messages.append(("user", query))
            st.session_state.messages.append(("bot", error_msg))
            st.error("❌ Query failed")
            return False
            
    except Exception as e:
        error_msg = f"❌ Request failed: {str(e)}"
        st.session_state.messages.append(("user", query))
        st.session_state.messages.append(("bot", error_msg))
        st.error(f"❌ Request failed: {str(e)}")
        return False

# Handle example query selection first
if hasattr(st.session_state, 'example_query'):
    selected_query = st.session_state.example_query
    st.info(f"Running example: {selected_query}")
    send_query(selected_query)
    del st.session_state.example_query
    st.rerun()
 
# Chat input form
with st.form("chat_form", clear_on_submit=True):
    user_query = st.text_input(
        "Ask a question:", 
        key="chat_input", 
        placeholder="e.g. How many nodes are in the graph?"
    )
    submitted = st.form_submit_button("Send")
 
# Handle form submission
if submitted and user_query:
    send_query(user_query)
    st.rerun()
 
st.divider()
 
# Display chat messages
if st.session_state.messages:
    st.markdown("### 💬 Chat History")
    for role, message in reversed(st.session_state.messages):
        if role == "user":
            st.markdown(f"🧑 **You:** {message}")
        else:
            st.markdown(f"🤖 **Agent:** {message}")
else:
    st.info("👋 Welcome! Ask a question or try an example from the sidebar.")
 
# Sidebar with system info
st.sidebar.title("🔧 System Info")
st.sidebar.markdown(f"**App Port:** {APP_PORT}")
st.sidebar.markdown(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
st.sidebar.markdown("**Services:**")
st.sidebar.markdown("- MCP Server: Port 8000")
st.sidebar.markdown("- Main App: Port 8081")
st.sidebar.markdown("- Streamlit UI: Port 8501")

# Check service status
st.sidebar.markdown("**Status Check:**")
try:
    health_response = requests.get(f"http://localhost:{APP_PORT}/health", timeout=3)
    if health_response.status_code == 200:
        st.sidebar.success("🟢 Backend Online")
    else:
        st.sidebar.warning("🟡 Backend Issues")
except:
    st.sidebar.error("🔴 Backend Offline")
 
# Example queries
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

# Clear chat button
st.sidebar.title("🗑️ Actions")
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.success("Chat history cleared!")
    st.rerun()
