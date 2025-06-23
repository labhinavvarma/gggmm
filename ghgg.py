import streamlit as st
import requests
import json
import uuid
import pickle
import pandas as pd

# === Cortex LLM Config ===
API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
APP_ID = "edadip"
APLCTN_CD = "edagnai"
MODEL = "llama3.1-70b"
SYS_MSG = "You are a powerful AI assistant. Provide accurate, concise answers based on context."

st.set_page_config(page_title="🧠 De-ID Health + ML Chatbot", layout="wide")
st.title("🧠 De-ID Health + ML Chatbot")

# === Session Initialization ===
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context_window" not in st.session_state:
    st.session_state.context_window = []
if "jsons" not in st.session_state:
    st.session_state.jsons = []
if "ml_prediction" not in st.session_state:
    st.session_state.ml_prediction = None

# === Reset Button ===
if st.button("🔄 Reset Conversation"):
    st.session_state.messages = []
    st.session_state.context_window = []
    st.session_state.jsons = []
    st.session_state.ml_prediction = None
    st.success("Conversation reset.")

# === JSON Medical Input ===
option = st.radio("Provide Medical Records:", ("Paste JSON", "Upload .json file(s)"))
parsed_jsons = []

if option == "Paste JSON":
    json_text = st.text_area("Paste your JSON (object or list):", height=200)
    if json_text.strip():
        try:
            data = json.loads(json_text)
            parsed_jsons = data if isinstance(data, list) else [data]
        except Exception:
            st.warning("Invalid JSON format.")
elif option == "Upload .json file(s)":
    uploaded_files = st.file_uploader("Upload .json files", type="json", accept_multiple_files=True)
    for file in uploaded_files:
        try:
            content = file.read().decode("utf-8")
            data = json.loads(content)
            parsed_jsons.extend(data if isinstance(data, list) else [data])
        except Exception:
            st.warning(f"Invalid file: {file.name}")

# === Store JSONs Persistently ===
if parsed_jsons:
    st.session_state.jsons = parsed_jsons
    st.success("✅ Medical records added to context.")

# === Upload ML Model ===
st.sidebar.markdown("### 🧠 ML Model Upload + Predict")
pkl_file = st.sidebar.file_uploader("Upload `.pkl` model", type=["pkl"])
input_file = st.sidebar.file_uploader("Upload Input for Model (.json or .csv)", type=["json", "csv"])
loaded_model = None
input_df = None

if pkl_file:
    try:
        loaded_model = pickle.load(pkl_file)
        st.sidebar.success("✅ Model loaded")
    except Exception as e:
        st.sidebar.error(f"Model load failed: {e}")

if input_file:
    try:
        if input_file.name.endswith(".json"):
            input_df = pd.read_json(input_file)
        else:
            input_df = pd.read_csv(input_file)
        st.sidebar.success("✅ Input data loaded")
    except Exception as e:
        st.sidebar.error(f"Input read error: {e}")

if loaded_model and input_df is not None:
    try:
        predictions = loaded_model.predict(input_df)
        pred_text = predictions.tolist()
        st.session_state.ml_prediction = {"predictions": pred_text}
        st.success("✅ Prediction added to LLM context")
        st.write("### 🔢 ML Predictions")
        st.write(pred_text)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# === LLM Call Function ===
def call_cortex_llm(user_query, context_window, json_context, ml_pred):
    session_id = str(uuid.uuid4())
    history = "\n".join(context_window[-5:])

    json_blob = (
        f"\n📁 Medical Records:\n{json.dumps(json_context, indent=2)}\n"
        if json_context else ""
    )

    ml_blob = (
        f"\n🤖 ML Prediction Output:\n{json.dumps(ml_pred, indent=2)}\n"
        if ml_pred else ""
    )

    full_prompt = f"{SYS_MSG}\n{json_blob}{ml_blob}{history}\nUser: {user_query}"

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
user_input = st.chat_input("💬 Ask about JSON records, ML predictions, or health data...")

if user_input:
    st.session_state.context_window.append(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    assistant_reply = call_cortex_llm(
        user_input,
        st.session_state.context_window,
        st.session_state.jsons,
        st.session_state.ml_prediction
    )

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

# === Display Chat History ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

