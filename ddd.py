
import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8081/chat"

st.set_page_config(page_title="Neo4j LLM Agent", page_icon="🧠")
st.title("🧠 Neo4j + LangGraph + Cortex Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask your question:", placeholder="e.g. How many nodes are there?")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    session_id = str(uuid.uuid4())
    response = requests.post(API_URL, json={"question": user_input, "session_id": session_id})
    if response.status_code == 200:
        result = response.json().get("answer", "No response.")
        st.session_state.messages.append(("user", user_input))
        st.session_state.messages.append(("agent", result))
    else:
        st.error("Something went wrong: " + response.text)

st.divider()
for role, message in reversed(st.session_state.messages):
    if role == "user":
        st.markdown(f"🧑 **You:** {message}")
    else:
        st.markdown(f"🤖 **Agent:** {message}")
