import streamlit as st
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
import logging

# Configure Streamlit page FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="🏥 Enhanced Health Analysis + Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import the enhanced agent
try:
    from enhanced_health_agent_with_chatbot import HealthAnalysisAgentWithChatbot, Config, ChatbotConfig
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    import_error = str(e)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    
    if "chatbot_session" not in st.session_state:
        st.session_state.chatbot_session = None
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    
    if "chatbot_ready" not in st.session_state:
        st.session_state.chatbot_ready = False

def create_agent():
    """Create and cache the enhanced health analysis agent"""
    try:
        with st.spinner("🔧 Initializing Enhanced Health Analysis Agent with Chatbot..."):
            agent = HealthAnalysisAgentWithChatbot()
            
            # Test both connections
            analysis_test = agent.test_llm_connection()
            chatbot_test = agent.test_chatbot_connection()
            
            if analysis_test["success"] and chatbot_test["success"]:
                st.success("✅ Both Analysis and Chatbot connections successful!")
                st.info(f"🤖 Analysis Model: {agent.config.model}")
                st.info(f"💬 Chatbot Model: {agent.chatbot_config.model}")
                return agent
            else:
                if not analysis_test["success"]:
                    st.error(f"❌ Analysis API Error: {analysis_test['error']}")
                if not chatbot_test["success"]:
                    st.error(f"❌ Chatbot API Error: {chatbot_test['error']}")
                return None
                
    except Exception as e:
        st.error(f"❌ Failed to initialize agent: {str(e)}")
        return None

def render_patient_input_form():
    """Render the patient data input form"""
    st.header("📋 Patient Information")
    
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            first_name = st.text_input("First Name", value="John", help="Patient's first name")
            last_name = st.text_input("Last Name", value="Doe", help="Patient's last name")
            ssn = st.text_input("SSN", value="123-45-6789", help="Social Security Number")
        
        with col2:
            date_of_birth = st.date_input("Date of Birth", help="Patient's date of birth")
            gender = st.selectbox("Gender", ["M", "F", "Other"], help="Patient's gender")
            zip_code = st.text_input("Zip Code", value="12345", help="Patient's zip code")
        
        submitted = st.form_submit_button("🚀 Run Health Analysis", use_container_width=True)
        
        if submitted:
            # Validate required fields
            if not all([first_name, last_name, ssn, date_of_birth, gender, zip_code]):
                st.error("❌ Please fill in all required fields")
                return None
            
            patient_data = {
                "first_name": first_name,
                "last_name": last_name,
                "ssn": ssn,
                "date_of_birth": date_of_birth.strftime("%Y-%m-%d"),
                "gender": gender,
                "zip_code": zip_code
            }
            
            return patient_data
    
    return None

def run_health_analysis(agent, patient_data):
    """Run the health analysis workflow"""
    try:
        with st.spinner("🔄 Running Enhanced Health Analysis Workflow..."):
            # Show progress
            progress_container = st.container()
            with progress_container:
                st.info("📡 Step 1/6: Fetching API data...")
                st.info("🔒 Step 2/6: Deidentifying data...")
                st.info("🔍 Step 3/6: Extracting medical/pharmacy fields...")
                st.info("🎯 Step 4/6: Extracting health entities...")
                st.info("📈 Step 5/6: Analyzing health trajectory...")
                st.info("📋 Step 6/6: Generating summary...")
            
            # Run analysis
            results = agent.run_analysis_with_chatbot(patient_data)
            
            # Clear progress and show results
            progress_container.empty()
            
            if results["success"]:
                st.success("✅ Health Analysis Completed Successfully!")
                st.session_state.analysis_results = results
                st.session_state.analysis_complete = True
                
                # Initialize chatbot session
                chatbot_session = agent.start_chatbot_session(results)
                if chatbot_session["success"]:
                    st.session_state.chatbot_session = chatbot_session
                    st.session_state.conversation_history = chatbot_session["conversation_history"]
                    st.session_state.chatbot_ready = True
                    st.success("💬 Chatbot ready for questions about deidentified data!")
                else:
                    st.warning(f"⚠️ Analysis completed but chatbot initialization failed: {chatbot_session.get('error', 'Unknown error')}")
                
                return results
            else:
                st.error(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
                if results.get("errors"):
                    for error in results["errors"]:
                        st.error(f"• {error}")
                return None
                
    except Exception as e:
        st.error(f"❌ Fatal error during analysis: {str(e)}")
        return None

def render_analysis_results(results):
    """Render the analysis results in tabs"""
    if not results:
        return
    
    st.header("📊 Analysis Results")
    
    # Create tabs for different result views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Summary", 
        "🔒 Deidentified Data", 
        "🔍 Extractions", 
        "🎯 Entities", 
        "📈 Trajectory", 
        "⚙️ Processing Details"
    ])
    
    with tab1:
        st.subheader("Executive Summary")
        if results.get("final_summary"):
            st.markdown(results["final_summary"])
        else:
            st.warning("No summary available")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            medical_records = len(results.get("structured_extractions", {}).get("medical", {}).get("hlth_srvc_records", []))
            st.metric("Medical Records", medical_records)
        
        with col2:
            pharmacy_records = len(results.get("structured_extractions", {}).get("pharmacy", {}).get("ndc_records", []))
            st.metric("Pharmacy Records", pharmacy_records)
        
        with col3:
            conditions = len(results.get("entity_extraction", {}).get("medical_conditions", []))
            st.metric("Medical Conditions", conditions)
    
    with tab2:
        st.subheader("Deidentified Medical Data")
        if results.get("deidentified_data", {}).get("medical"):
            st.json(results["deidentified_data"]["medical"])
        
        st.subheader("Deidentified Pharmacy Data")
        if results.get("deidentified_data", {}).get("pharmacy"):
            st.json(results["deidentified_data"]["pharmacy"])
    
    with tab3:
        st.subheader("Medical Field Extractions")
        medical_extraction = results.get("structured_extractions", {}).get("medical", {})
        if medical_extraction.get("hlth_srvc_records"):
            df_medical = pd.DataFrame(medical_extraction["hlth_srvc_records"])
            st.dataframe(df_medical, use_container_width=True)
        
        st.subheader("Pharmacy Field Extractions")
        pharmacy_extraction = results.get("structured_extractions", {}).get("pharmacy", {})
        if pharmacy_extraction.get("ndc_records"):
            df_pharmacy = pd.DataFrame(pharmacy_extraction["ndc_records"])
            st.dataframe(df_pharmacy, use_container_width=True)
    
    with tab4:
        st.subheader("Extracted Health Entities")
        entities = results.get("entity_extraction", {})
        
        # Health status indicators
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Diabetes", entities.get("diabetics", "unknown"))
            st.metric("Blood Pressure", entities.get("blood_pressure", "unknown"))
        
        with col2:
            st.metric("Age Group", entities.get("age_group", "unknown"))
            st.metric("Smoking", entities.get("smoking", "unknown"))
        
        # Medical conditions
        if entities.get("medical_conditions"):
            st.subheader("Medical Conditions Identified")
            for condition in entities["medical_conditions"]:
                st.info(f"• {condition}")
        
        # Medications
        if entities.get("medications_identified"):
            st.subheader("Medications Identified")
            df_meds = pd.DataFrame(entities["medications_identified"])
            st.dataframe(df_meds, use_container_width=True)
    
    with tab5:
        st.subheader("Health Trajectory Analysis")
        if results.get("health_trajectory"):
            st.markdown(results["health_trajectory"])
        else:
            st.warning("No trajectory analysis available")
    
    with tab6:
        st.subheader("Processing Status")
        step_status = results.get("step_status", {})
        for step, status in step_status.items():
            if status == "completed":
                st.success(f"✅ {step}: {status}")
            elif status == "error":
                st.error(f"❌ {step}: {status}")
            else:
                st.info(f"ℹ️ {step}: {status}")
        
        st.subheader("Processing Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Steps Completed", results.get("processing_steps_completed", 0))
            st.metric("LangGraph Used", "Yes" if results.get("langgraph_used") else "No")
        
        with col2:
            st.metric("Chatbot Enabled", "Yes" if results.get("chatbot_enabled") else "No")
            st.metric("Enhancement Version", results.get("enhancement_version", "Unknown"))

def render_chatbot_interface(agent):
    """Render the chatbot interface for deidentified data interaction - similar to working chatbot"""
    if not st.session_state.chatbot_ready:
        st.info("💬 Complete the health analysis first to enable chatbot")
        return
    
    st.header("🧠 De-ID Health Assist")
    st.markdown("""
    Ask questions about the deidentified medical and pharmacy data.  
    The conversation context is preserved.
    """)
    
    # --- Reset Conversation Button ---
    if st.button("🔄 Reset Conversation"):
        if st.session_state.analysis_results:
            reset_result = agent.reset_chatbot_conversation(st.session_state.analysis_results)
            if reset_result["success"]:
                st.session_state.conversation_history = reset_result["conversation_history"]
                st.success("Conversation reset.")
                st.rerun()
    
    # --- User Input for Conversation ---
    user_input = st.text_input("Ask a question or give an instruction about the medical record(s):")
    
    if st.button("🔍 Extract De-ID Data / Continue Conversation"):
        if not user_input.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("🤖 Analyzing your question..."):
                chat_result = agent.chat_with_data(
                    st.session_state.conversation_history, 
                    user_input.strip()
                )
                
                if chat_result["success"]:
                    st.session_state.conversation_history = chat_result["conversation_history"]
                    st.rerun()
                else:
                    st.error(f"❌ Chat error: {chat_result.get('error', 'Unknown error')}")
    
    # --- Display Conversation History (Questions & Answers Only) ---
    if st.session_state.conversation_history:
        st.markdown("---")
        st.markdown("### Conversation History")
        
        # Skip the initial context message(s)
        skip_prefix_single = "Here is the deidentified medical and pharmacy data"
        
        for msg in st.session_state.conversation_history:
            # Skip the initial context message(s)
            if msg["role"] == "user" and msg["content"].strip().startswith(skip_prefix_single):
                continue
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            elif msg["role"] == "assistant":
                st.markdown(f"**Assistant:** {msg['content']}")

def render_sidebar():
    """Render the sidebar with configuration and controls"""
    with st.sidebar:
        st.title("🏥 Health Analysis + Chatbot")
        st.markdown("---")
        
        # Configuration display
        st.subheader("⚙️ Configuration")
        
        if AGENT_AVAILABLE:
            config = Config()
            chatbot_config = ChatbotConfig()
            
            with st.expander("📊 Analysis Settings"):
                st.code(f"""
Model: {config.model}
API: {config.api_url[:50]}...
App ID: {config.app_id}
Timeout: {config.timeout}s
                """)
            
            with st.expander("💬 Chatbot Settings (Hardcoded)"):
                st.code(f"""
Model: {chatbot_config.model}
Host: {chatbot_config.host[:30]}...
User: {chatbot_config.user}
Warehouse: {chatbot_config.warehouse}
Role: {chatbot_config.role}
                """)
        else:
            st.error(f"❌ Agent not available: {import_error}")
        
        st.markdown("---")
        
        # Status indicators
        st.subheader("📊 Status")
        
        if st.session_state.agent:
            st.success("🤖 Agent: Ready")
        else:
            st.error("🤖 Agent: Not initialized")
        
        if st.session_state.analysis_complete:
            st.success("📈 Analysis: Complete")
        else:
            st.info("📈 Analysis: Pending")
        
        if st.session_state.chatbot_ready:
            st.success("💬 Chatbot: Ready")
        else:
            st.info("💬 Chatbot: Waiting for analysis")
        
        st.markdown("---")
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        
        if st.session_state.analysis_results:
            if st.button("📥 Download Results", use_container_width=True):
                results_json = json.dumps(st.session_state.analysis_results, indent=2, default=str)
                st.download_button(
                    label="💾 Download JSON",
                    data=results_json,
                    file_name=f"health_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

def main():
    """Main Streamlit application"""
    # Initialize session state
    initialize_session_state()
    
    # Check agent availability
    if not AGENT_AVAILABLE:
        st.error(f"❌ Failed to import Enhanced Health Analysis Agent: {import_error}")
        
        with st.expander("🔧 Installation Guide"):
            st.markdown("""
            **Install Requirements:**
            ```bash
            pip install langgraph langchain-core streamlit requests urllib3 pandas snowflake-connector-python
            ```
            
            **Required Files:**
            - `enhanced_health_agent_with_chatbot.py` (the Enhanced LangGraph agent)
            - `streamlit_enhanced_ui_with_chatbot.py` (this file)
            """)
        st.stop()
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    st.title("🏥 Enhanced Health Analysis + Chatbot")
    st.markdown("### Complete workflow: Data Analysis + Interactive Chatbot with Deidentified Medical Data")
    
    # Initialize agent if not already done
    if st.session_state.agent is None:
        st.session_state.agent = create_agent()
    
    if st.session_state.agent is None:
        st.error("❌ Unable to initialize the health analysis agent. Please check your configuration.")
        st.stop()
    
    # Main workflow
    if not st.session_state.analysis_complete:
        # Step 1: Patient input and analysis
        patient_data = render_patient_input_form()
        
        if patient_data:
            results = run_health_analysis(st.session_state.agent, patient_data)
            if results:
                st.rerun()
    
    else:
        # Step 2: Display results and chatbot
        col1, col2 = st.columns([2, 1])
        
        with col1:
            render_analysis_results(st.session_state.analysis_results)
        
        with col2:
            render_chatbot_interface(st.session_state.agent)

if __name__ == "__main__":
    main()
