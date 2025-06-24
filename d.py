
import streamlit as st
import json
from langgraph_agent import LangGraphRAGAgent # Import the stable agent

st.set_page_config(page_title="LangGraph Health Agent UI", layout="wide")
st.title("👨‍⚕️ Health Record RAG Agent")
st.markdown("This interface uses a stable, LangGraph-powered agent in the backend. Paste a patient's JSON record, ask a question, and the agent will process it and provide an answer.")

# --- Initialize or retrieve the agent from session state ---
if 'agent' not in st.session_state:
    # Instantiate the agent only once per session
    st.session_state.agent = LangGraphRAGAgent()
    st.session_state.messages = []
    st.session_state.processed_record = None

# --- UI Layout ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Step 1: Provide Patient Data")
    default_record = {
      "first_name": "Eleanor",
      "last_name": "Vance",
      "date_of_birth": "1955-09-23",
      "gender": "F"
    }
    json_text = st.text_area(
        "Paste Patient JSON Record Here:",
        value=json.dumps(default_record, indent=2),
        height=200
    )

with col2:
    st.subheader("Step 2: Ask a Question")
    question = st.text_input(
        "Ask a question about this record:",
        value="What preventive care is recommended for this patient?"
    )

if st.button("🚀 Run Agent Workflow", type="primary"):
    if json_text.strip() and question.strip():
        try:
            patient_record = json.loads(json_text)
            st.session_state.processed_record = patient_record
            st.session_state.messages = [] # Clear previous chat
            
            st.session_state.messages.append({"role": "user", "content": question})

            with st.spinner("The agent is running its LangGraph workflow..."):
                # Call the agent's single public method
                response = st.session_state.agent.invoke(patient_record, question)
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            st.rerun() # Rerun to display the new messages

        except json.JSONDecodeError:
            st.error("Invalid JSON format. Please check the patient record.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please provide both a patient record and a question.")

st.divider()

# --- Chat History Display ---
st.subheader("Conversation")
if not st.session_state.messages:
    st.info("Run the agent workflow to see the conversation here.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
