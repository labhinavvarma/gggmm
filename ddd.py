import streamlit as st
import requests
import uuid

st.set_page_config(page_title="Neo4j LLM Agent", page_icon="🧠")
st.title("🧠 Neo4j + LangGraph + Cortex Chat")

API_URL = "http://localhost:8081/chat"

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Ask your question:",
        placeholder="e.g. How many nodes are there? (Try: Create a Person node named Bob)"
    )
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    session_id = str(uuid.uuid4())
    try:
        response = requests.post(API_URL, json={"question": user_input, "session_id": session_id})
        if response.status_code == 200:
            result = response.json()
            st.session_state.messages.append(("user", user_input))
            st.session_state.messages.append(("agent", result.get("answer", "No answer")))
            if result.get("trace"):
                st.session_state.messages.append(("trace", result["trace"]))
        else:
            st.error(f"Something went wrong: {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")

st.divider()
for role, message in reversed(st.session_state.messages):
    if role == "user":
        st.markdown(f"🧑 **You:** {message}")
    elif role == "agent":
        st.markdown(f"🤖 **Agent:** {message}")
    elif role == "trace":
        st.markdown(
            f"<span style='color:grey;font-size:12px'><b>Trace:</b> {message}</span>",
            unsafe_allow_html=True,
        )
