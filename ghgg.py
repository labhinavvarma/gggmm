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

# === Streamlit UI ===
st.set_page_config(page_title="🧠 De-ID Health Assist", layout="wide")
st.title("🧠 De-ID Health Assist")

st.markdown("""
Upload one or more JSON medical records, or paste JSON(s) below.  
Ask questions or give instructions. The assistant will keep the JSONs in context.
""")

# === Session Initialization ===
if "messages" not in st.session_state:
    st.session_state.messages = []

if "context_window" not in st.session_state:
    st.session_state.context_window = []

if "jsons" not in st.session_state:
    st.session_state.jsons = []

# === Reset Button ===
if st.button("🔄 Reset Conversation"):
    st.session_state.messages = []
    st.session_state.context_window = []
    st.session_state.jsons = []
    st.success("Conversation reset.")

# === Input Option: Upload or Paste ===
option = st.radio("How would you like to provide the Medical Record?", ("Paste JSON", "Upload .json file(s)"))
parsed_jsons = []

if option == "Paste JSON":
    json_text = st.text_area("Paste your JSON here (object or list):", height=200)
    if json_text.strip():
        try:
            data = json.loads(json_text)
            parsed_jsons = data if isinstance(data, list) else [data]
        except Exception:
            st.warning("Invalid JSON format.")
elif option == "Upload .json file(s)":
    uploaded_files = st.file_uploader("Upload JSON file(s)", type="json", accept_multiple_files=True)
    for file in uploaded_files:
        try:
            content = file.read().decode("utf-8")
            data = json.loads(content)
            parsed_jsons.extend(data if isinstance(data, list) else [data])
        except Exception:
            st.warning(f"Invalid file: {file.name}")

# === Save valid parsed JSONs ===
if parsed_jsons:
    st.session_state.jsons = parsed_jsons
    st.success("✅ Medical records added to context.")

# === LLM Call Function ===
def call_cortex_llm(user_query, context_window, json_context):
    session_id = str(uuid.uuid4())
    history = "\n".join(context_window[-5:])
    
    json_blob = ""
    if json_context:
        json_blob = f"\nThese are the relevant medical records:\n{json.dumps(json_context, indent=2)}\n"

    full_prompt = f"{SYS_MSG}\n{json_blob}\n{history}\nUser: {user_query}"

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

# === Chat Input ===
user_input = st.chat_input("💬 Ask a question about the medical records...")

if user_input:
    st.session_state.context_window.append(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    assistant_reply = call_cortex_llm(
        user_input,
        st.session_state.context_window,
        st.session_state.jsons
    )

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

# === Chat History Display ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
