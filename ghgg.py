import streamlit as st
import requests
import json
import uuid
import pickle
import pandas as pd

# === Cortex LLM Config ===
API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "xxxxxx"
APP_ID = "edadip"
APLCTN_CD = "edagnai"
MODEL = "llama3.1-70b"
SYS_MSG = "You are a powerful AI assistant. Provide accurate, concise answers based on context."

st.set_page_config(page_title="🧠 Health + ML Chatbot", layout="wide")
st.title("🧠 Health + ML Chatbot with Multi-Model Support")

# === Session State Init ===
for key in ["messages", "context_window", "jsons", "ml_predictions"]:
    if key not in st.session_state:
        st.session_state[key] = []

# === Reset Chat Button ===
if st.button("🔄 Reset Chat"):
    for key in ["messages", "context_window", "jsons", "ml_predictions"]:
        st.session_state[key] = []
    st.success("Reset successful!")

# === Medical JSON Upload ===
option = st.radio("Provide Medical Records:", ("Paste JSON", "Upload .json file(s)"))
parsed_jsons = []

if option == "Paste JSON":
    json_text = st.text_area("Paste your JSON:", height=200)
    if json_text.strip():
        try:
            parsed = json.loads(json_text)
            parsed_jsons = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            st.warning("Invalid JSON format.")
elif option == "Upload .json file(s)":
    files = st.file_uploader("Upload .json files", type="json", accept_multiple_files=True)
    for file in files:
        try:
            data = json.load(file)
            parsed_jsons.extend(data if isinstance(data, list) else [data])
        except Exception:
            st.warning(f"Invalid file: {file.name}")

if parsed_jsons:
    st.session_state.jsons = parsed_jsons
    st.success("✅ Medical JSONs loaded")

# === ML Model Upload ===
st.sidebar.markdown("### 🧠 Upload ML Models")
model_files = st.sidebar.file_uploader("Upload `.pkl` files", type="pkl", accept_multiple_files=True)

loaded_models = {}
for model_file in model_files:
    try:
        model = pickle.load(model_file)
        loaded_models[model_file.name] = model
        st.sidebar.success(f"Loaded: {model_file.name}")
    except Exception as e:
        st.sidebar.error(f"Error loading {model_file.name}: {e}")

# === Input Data for Model ===
input_file = st.sidebar.file_uploader("📂 Upload prediction input (.json/.csv)", type=["json", "csv"])
input_df = None
if input_file:
    try:
        if input_file.name.endswith(".json"):
            input_df = pd.read_json(input_file)
        else:
            input_df = pd.read_csv(input_file)
        st.sidebar.success("✅ Input data loaded")
    except Exception as e:
        st.sidebar.error(f"Input read error: {e}")

# === Select model to run prediction ===
if loaded_models and input_df is not None:
    selected_model_name = st.sidebar.selectbox("Select a model to run", list(loaded_models.keys()))
    if selected_model_name:
        model = loaded_models[selected_model_name]
        try:
            predictions = model.predict(input_df)
            result = {
                "model_name": selected_model_name,
                "predictions": predictions.tolist()
            }
            st.session_state.ml_predictions.append(result)
            st.success(f"✅ Prediction done with {selected_model_name}")
            st.write(f"### 🔢 Predictions from `{selected_model_name}`")
            st.write(result["predictions"])
        except Exception as e:
            st.error(f"Prediction error: {e}")

# === Cortex LLM Call ===
def call_cortex_llm(user_query, context_window, json_context, model_preds):
    session_id = str(uuid.uuid4())
    history = "\n".join(context_window[-5:])
    
    json_blob = (
        f"\n📁 Medical Records:\n{json.dumps(json_context, indent=2)}\n"
        if json_context else ""
    )

    model_blob = ""
    for mp in model_preds:
        model_blob += f"\n🔬 Model `{mp['model_name']}` Predictions:\n{json.dumps(mp['predictions'], indent=2)}\n"

    full_prompt = f"{SYS_MSG}\n{json_blob}{model_blob}{history}\nUser: {user_query}"

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
user_input = st.chat_input("💬 Ask about ML predictions or medical records...")
if user_input:
    st.session_state.context_window.append(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    assistant_reply = call_cortex_llm(
        user_input,
        st.session_state.context_window,
        st.session_state.jsons,
        st.session_state.ml_predictions
    )

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

# === Chat Display ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

