
import streamlit as st
import requests
import json
import uuid

# === Cortex LLM Config ===
API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
APP_ID = "edadip"
APLCTN_CD = "edagnai"
MODEL = "llama3.1-70b"
SYS_MSG = "You are a powerful AI assistant. Provide accurate, concise answers based on context."

# --- App UI ---
st.title("🧠 De-ID Health Assist")
st.markdown("""
Upload one or more JSON medical records, or paste JSON(s) below.  
Ask questions or give instructions. The conversation context is preserved.
""")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context_window" not in st.session_state:
    st.session_state.context_window = []

# --- Reset Button ---
if st.button("🔄 Reset Conversation"):
    st.session_state.messages = []
    st.session_state.context_window = []
    st.success("Conversation reset.")

# --- Input Method ---
option = st.radio("Choose how to provide the Medical Record:", ("Paste JSON", "Upload .json file(s)"))
parsed_jsons = []

if option == "Paste JSON":
    json_data = st.text_area("Paste JSON (single object or list):", height=200)
    if json_data.strip():
        try:
            parsed = json.loads(json_data)
            parsed_jsons = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            st.warning("Invalid JSON format.")
elif option == "Upload .json file(s)":
    uploaded_files = st.file_uploader("Upload .json files", type="json", accept_multiple_files=True)
    for file in uploaded_files:
        try:
            content = file.read().decode("utf-8")
            parsed = json.loads(content)
            parsed_jsons.extend(parsed if isinstance(parsed, list) else [parsed])
        except Exception:
            st.warning(f"Invalid file: {file.name}")

# --- User Input ---
user_input = st.text_input("Ask a question or provide instructions:")
if st.button("🔍 Run Analysis / Continue Conversation"):
    if not parsed_jsons and not st.session_state.messages:
        st.warning("Please provide valid JSON medical records.")
    else:
        # Initial context
        if not st.session_state.messages:
            st.session_state.messages.append({
                "role": "system",
                "content": (
                    "You are a medical assistant. Extract and answer health-related information "
                    "from the provided medical record JSON(s) including Medical and Pharmacy Claims. "
                    "Focus on diagnoses, medications, procedures, and relevant clinical details."
                )
            })
            context_str = f"Here {'is the JSON:' if len(parsed_jsons)==1 else 'are multiple JSONs:'}\n" + json.dumps(parsed_jsons, indent=2)
            st.session_state.messages.append({"role": "user", "content": context_str})

        # Add user query
        if user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input.strip()})
            st.session_state.context_window.append(user_input.strip())

            # --- Call Cortex LLM ---
            def call_cortex_llm(text, context_window):
                session_id = str(uuid.uuid4())
                history = "\n".join(context_window[-5:])
                full_prompt = f"{SYS_MSG}\n{history}\nUser: {text}"
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
                            "messages": [{"role": "user", "content": full_prompt}]
                        },
                        "app_lvl_prefix": "",
                        "user_id": "",
                        "session_id": session_id
                    }
                }
                headers = {
                    "Content-Type": "application/json; charset=utf-8",
                    "Accept": "application/json",
                    "Authorization": f'Snowflake Token="{API_KEY}"'
                }
                try:
                    response = requests.post(API_URL, headers=headers, json=payload, verify=False)
                    if response.status_code == 200:
                        raw = response.text
                        if "end_of_stream" in raw:
                            answer, _, _ = raw.partition("end_of_stream")
                            return answer.strip()
                        return raw.strip()
                    return f"❌ Cortex Error {response.status_code}: {response.text}"
                except Exception as e:
                    return f"❌ Cortex Exception: {str(e)}"

            # Get LLM response
            assistant_reply = call_cortex_llm(user_input, st.session_state.context_window)
            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
            st.markdown("**Assistant:**")
            st.write(assistant_reply)

# --- Display History ---
if st.session_state.messages:
    st.markdown("---")
    st.markdown("### Conversation History")
    skip_prefixes = ["Here is the JSON:", "Here are multiple JSONs:"]
    for msg in st.session_state.messages:
        if msg["role"] == "user" and any(msg["content"].startswith(p) for p in skip_prefixes):
            continue
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        elif msg["role"] == "assistant":
            st.markdown(f"**Assistant:** {msg['content']}")
