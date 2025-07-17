import streamlit as st
import requests
import uuid

# Try to import config, fallback to hardcoded values
try:
    from config import SERVER_CONFIG
    APP_PORT = SERVER_CONFIG["app_port"]
except ImportError:
    APP_PORT = 8081

# Page configuration
st.set_page_config(
    page_title="Neo4j Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Simple styling
st.markdown("""
<style>
    .user-msg {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 4px solid #2196f3;
    }
    .bot-msg {
        background-color: #f3e5f5;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Title
st.title("🤖 Neo4j Chatbot")

# Chat input
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Ask me about your Neo4j database:",
        placeholder="How many nodes are in the graph?"
    )
    submitted = st.form_submit_button("Send")

# Process message
if submitted and user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Send to backend
    try:
        payload = {
            "question": user_input,
            "session_id": st.session_state.session_id
        }
        
        response = requests.post(f"http://localhost:{APP_PORT}/chat", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "No answer received")
            
            # Add bot response
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Sorry, I couldn't process your request."
            })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Error: {str(e)}"
        })

# Display messages
st.markdown("### Chat History")

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="user-msg">
            <strong>You:</strong> {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="bot-msg">
            <strong>Bot:</strong> {message["content"]}
        </div>
        """, unsafe_allow_html=True)

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
