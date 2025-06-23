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

st.set_page_config(page_title="🧠 Heart Disease Chatbot + ML", layout="wide")
st.title("🧠 Heart Disease Chatbot + ML (.pkl Chain)")

# === Init session state ===
for key in ["messages", "context_window", "jsons", "ml_prediction"]:
    if key not in st.session_state:
        st.session_state[key] = []

# === Reset Button ===
if st.button("🔄 Reset"):
    for key in ["messages", "context_window", "jsons", "ml_prediction"]:
        st.session_state[key] = []
    st.success("Conversation reset.")

# === JSON Input ===
option = st.radio("Provide Medical Records:", ("Paste JSON", "Upload .json file(s)"))
parsed_jsons = []

if option == "Paste JSON":
    json_text = st.text_area("Paste JSON:", height=200)
    if json_text.strip():
        try:
            parsed = json.loads(json_text)
            parsed_jsons = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            st.warning("Invalid JSON.")
elif option == "Upload .json file(s)":
    files = st.file_uploader("Upload .json files", type="json", accept_multiple_files=True)
    for f in files:
        try:
            content = f.read().decode("utf-8")
            data = json.loads(content)
            parsed_jsons.extend(data if isinstance(data, list) else [data])
        except Exception:
            st.warning(f"Invalid JSON in {f.name}")

if parsed_jsons:
    st.session_state.jsons = parsed_jsons
    st.success("✅ Medical JSONs loaded.")

# === Upload all 3 PKL files ===
st.sidebar.markdown("### 📦 Upload 3 Model Parts")
pkl_files = st.sidebar.file_uploader("Upload: preprocessor → model → calibrator", type="pkl", accept_multiple_files=True)

preprocessor = model = calibrator = None

if len(pkl_files) == 3:
    try:
        preprocessor = pickle.load(pkl_files[0])
        model = pickle.load(pkl_files[1])
        calibrator = pickle.load(pkl_files[2])
        st.sidebar.success("✅ All 3 components loaded.")
    except Exception as e:
        st.sidebar.error(f"❌ Load error: {e}")
else:
    st.sidebar.info("Upload exactly 3 `.pkl` files in order.")

# === Input Data for Model ===
input_file = st.sidebar.file_uploader("📂 Upload input data (.csv/.json)", type=["csv", "json"])
input_df = None

if input_file:
    try:
        if input_file.name.endswith(".json"):
            input_df = pd.read_json(input_file)
        else:
            input_df = pd.read_csv(input_file)
        st.sidebar.success("✅ Input data loaded.")
    except Exception as e:
        st.sidebar.error(f"❌ Input read error: {e}")

# === Predict using full chain: preprocessor → model → calibrator
if preprocessor and model and calibrator and input_df is not None:
    try:
        transformed = preprocessor.transform(input_df)
        raw_preds = model.predict_proba(transformed)
        final_preds = calibrator.transform(raw_preds)
        final_preds = final_preds.tolist()
        st.session_state.ml_prediction = {"model": "HeartDiseaseAdaBoost", "predictions": final_preds}
        st.success("✅ Prediction complete")
        st.write("### 🔢 Final Predictions")
        st.write(final_preds)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# === LLM Call ===
def call_cortex_llm(query, ctx_window, json_ctx, pred_ctx):
    session_id = str(uuid.uuid4())
    history = "\n".join(ctx_window[-5:])
    json_blob = f"\n📁 Medical Records:\n{json.dumps(json_ctx, indent=2)}\n" if json_ctx else ""
    pred_blob = f"\n🔬 ML Predictions:\n{json.dumps(pred_ctx, indent=2)}\n" if pred_ctx else ""
    prompt = f"{SYS_MSG}\n{json_blob}{pred_blob}{history}\nUser: {query}"

    payload = {
        "query": {
            "aplctn_cd": APLCTN_CD,
            "app_id": APP_ID,
            "api_key": API_KEY,
            "method": "cortex",
            "model": MODEL,
            "sys_msg": SYS_MSG,
            "limit_convs": "0",
            "prompt": {"messages": [{"role": "user", "content": prompt}]},
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
        res = requests.post(API_URL, headers=headers, json=payload, verify=False)
        if res.status_code == 200:
            text = res.text
            return text.partition("end_of_stream")[0].strip() if "end_of_stream" in text else text.strip()
        return f"❌ Cortex Error {res.status_code}: {res.text}"
    except Exception as e:
        return f"❌ Cortex Exception: {str(e)}"

# === Chat Input ===
user_input = st.chat_input("💬 Ask anything about medical data or ML predictions...")
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

# === Chat Display ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
