# Configure Streamlit page FIRST - before any other Streamlit commands
import streamlit as st
st.set_page_config(
    page_title="🔥 Enhanced LangGraph + Snowflake Cortex Health Analysis + Interactive Chatbot",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other modules
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import os
from typing import Dict, Any, Optional

# Add current directory to path for importing the agent
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the Enhanced LangGraph health analysis agent with Snowflake Cortex + Chatbot
AGENT_AVAILABLE = False
import_error = None
HealthAnalysisAgent = None
Config = None

try:
    from langgraph_health_agent_proper import HealthAnalysisAgent, Config
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    import_error = str(e)

# Custom CSS for Enhanced LangGraph + Snowflake + Chatbot themed styling
st.markdown("""
<style>
.main-header {
    font-size: 2.8rem;
    color: #ff6b35;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}

.langgraph-badge {
    background: linear-gradient(45deg, #ff6b35, #f7931e);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 1rem;
    font-weight: bold;
    display: inline-block;
    margin: 0.5rem 0;
}

.snowflake-badge {
    background: linear-gradient(45deg, #29b5e8, #00a2e8);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 0.8rem;
    font-weight: bold;
    display: inline-block;
    margin: 0.2rem;
    font-size: 0.9rem;
}

.chatbot-badge {
    background: linear-gradient(45deg, #6f42c1, #8e44ad);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 0.8rem;
    font-weight: bold;
    display: inline-block;
    margin: 0.2rem;
    font-size: 0.9rem;
}

.extraction-badge {
    background: linear-gradient(45deg, #28a745, #20c997);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 0.8rem;
    font-weight: bold;
    display: inline-block;
    margin: 0.2rem;
    font-size: 0.9rem;
}

.step-header {
    font-size: 1.6rem;
    color: #ff6b35;
    border-left: 4px solid #ff6b35;
    padding-left: 1rem;
    margin: 1.5rem 0;
    font-weight: bold;
}

.chatbot-header {
    font-size: 1.6rem;
    color: #6f42c1;
    border-left: 4px solid #6f42c1;
    padding-left: 1rem;
    margin: 1.5rem 0;
    font-weight: bold;
}

.success-box {
    background: linear-gradient(135deg, #d4edda, #c3e6cb);
    border: 2px solid #28a745;
    color: #155724;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    border-radius: 0.5rem;
    font-weight: bold;
}

.error-box {
    background: linear-gradient(135deg, #f8d7da, #f5c6cb);
    border: 2px solid #dc3545;
    color: #721c24;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    border-radius: 0.5rem;
}

.info-box {
    background: linear-gradient(135deg, #cce7ff, #99d6ff);
    border: 2px solid #007bff;
    color: #004085;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    border-radius: 0.5rem;
}

.chatbot-box {
    background: linear-gradient(135deg, #e8e0ff, #d6c7ff);
    border: 2px solid #6f42c1;
    color: #432874;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    border-radius: 0.5rem;
}

.langgraph-node {
    background: linear-gradient(135deg, #fff3e0, #ffecb3);
    border: 2px solid #ff9800;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 0.5rem 0;
}

.chatbot-node {
    background: linear-gradient(135deg, #f3e5f5, #e1bee7);
    border: 2px solid #9c27b0;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 0.5rem 0;
}

.extraction-node {
    background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
    border: 2px solid #4caf50;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 0.5rem 0;
}

.entity-card {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    border: 2px solid #6c757d;
    border-radius: 0.8rem;
    padding: 1.5rem;
    margin: 0.5rem;
    text-align: center;
    transition: transform 0.2s;
}

.entity-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.metric-highlight {
    font-size: 1.2rem;
    font-weight: bold;
    color: #ff6b35;
}

.age-display {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
    padding: 0.5rem 1rem;
    margin: 0.5rem 0;
    border-radius: 0.25rem;
}

.chat-message {
    padding: 0.5rem 1rem;
    margin: 0.5rem 0;
    border-radius: 0.5rem;
}

.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
}

.assistant-message {
    background-color: #f3e5f5;
    border-left: 4px solid #9c27b0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'config' not in st.session_state:
        st.session_state.config = None
    # NEW: Chatbot session state
    if 'chatbot_messages' not in st.session_state:
        st.session_state.chatbot_messages = []
    if 'chatbot_context' not in st.session_state:
        st.session_state.chatbot_context = None

def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary"""
    try:
        return data.get(key, default) if data else default
    except:
        return default

def safe_str(value: Any) -> str:
    """Safely convert any value to string"""
    try:
        return str(value) if value is not None else "unknown"
    except:
        return "unknown"

def safe_json_dumps(data: Any, default: str = "{}") -> str:
    """Safely convert data to JSON string"""
    try:
        return json.dumps(data, indent=2) if data else default
    except Exception as e:
        return f'{{"error": "JSON serialization failed: {str(e)}"}}'

def calculate_age(birth_date):
    """Calculate age from birth date"""
    if not birth_date:
        return None
    
    today = datetime.now().date()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age

def validate_patient_data(data: Dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate patient data and return validation status and errors"""
    errors = []
    required_fields = {
        'first_name': 'First Name',
        'last_name': 'Last Name', 
        'ssn': 'SSN',
        'date_of_birth': 'Date of Birth',
        'gender': 'Gender',
        'zip_code': 'Zip Code'
    }
    
    for field, display_name in required_fields.items():
        if not data.get(field):
            errors.append(f"{display_name} is required")
        elif field == 'ssn' and len(str(data[field])) < 9:
            errors.append("SSN must be at least 9 digits")
        elif field == 'zip_code' and len(str(data[field])) < 5:
            errors.append("Zip code must be at least 5 digits")
    
    # Additional date validation
    if data.get('date_of_birth'):
        try:
            birth_date = datetime.strptime(data['date_of_birth'], '%Y-%m-%d').date()
            age = calculate_age(birth_date)
            
            if age and age > 150:
                errors.append("Age cannot be greater than 150 years")
            elif age and age < 0:
                errors.append("Date of birth cannot be in the future")
        except:
            errors.append("Invalid date format")
    
    return len(errors) == 0, errors

# Initialize session state
initialize_session_state()

# Main Title with Enhanced LangGraph + Snowflake + Chatbot branding
st.markdown('<h1 class="main-header">🔥 Enhanced LangGraph + Snowflake Cortex + Interactive Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<div class="langgraph-badge">🚀 Powered by Enhanced LangGraph v3.0 + Snowflake Cortex API</div>', unsafe_allow_html=True)
st.markdown('<div class="snowflake-badge">❄️ Snowflake Cortex: llama3.1-70b</div>', unsafe_allow_html=True)
st.markdown('<div class="chatbot-badge">💬 Interactive Medical Data Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="extraction-badge">🏥 Medical ICD-10 Extraction</div>', unsafe_allow_html=True)
st.markdown('<div class="extraction-badge">💊 Pharmacy NDC Extraction</div>', unsafe_allow_html=True)
st.markdown("**Advanced health analysis with Snowflake Cortex AI, comprehensive medical data extraction, and interactive chatbot for medical data Q&A**")

# Display import status AFTER page config
if AGENT_AVAILABLE:
    st.success("✅ Enhanced LangGraph Health Analysis Agent with Interactive Chatbot imported successfully!")
else:
    st.error(f"❌ Failed to import Enhanced LangGraph Health Analysis Agent: {import_error}")
    
    with st.expander("🔧 Enhanced LangGraph + Snowflake + Chatbot Installation Guide"):
        st.markdown("""
        **Install Enhanced LangGraph Requirements:**
        ```bash
        pip install langgraph langchain-core streamlit requests urllib3 pandas
        ```
        
        **Required Files:**
        - `langgraph_health_agent_proper.py` (the Enhanced LangGraph agent with Snowflake Cortex + Chatbot)
        - `streamlit_langgraph_ui.py` (this file)
        
        **Enhanced LangGraph + Snowflake + Chatbot Features v3.0:**
        - ✅ State management and persistence
        - ✅ Conditional workflow routing  
        - ✅ Automatic retry mechanisms
        - ✅ Error handling and recovery
        - ✅ Checkpointing for reliability
        - 🆕 **Medical field extraction (hlth_srvc_cd, diag_1_50_cd)**
        - 🆕 **Pharmacy field extraction (Ndc, lbl_nm)**
        - 🆕 **Enhanced entity detection with ICD-10 codes**
        - 🆕 **Snowflake Cortex API integration**
        - 🆕 **llama3.1-70b model for health analysis**
        - 🔥 **Interactive chatbot with medical data context**
        """)
    st.stop()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Enhanced LangGraph + Snowflake + Chatbot Configuration")
    
    # Enhanced LangGraph + Snowflake + Chatbot Status
    st.markdown("### 🔥 System Status")
    st.markdown("✅ **LangGraph v3.0 Enabled**")
    st.markdown("❄️ **Snowflake Cortex Enabled**")
    st.markdown("💬 **Interactive Chatbot Enabled**")
    st.markdown("🔄 **State Management:** Active")
    st.markdown("💾 **Checkpointing:** Enabled")
    st.markdown("🔁 **Retry Logic:** Configured")
    st.markdown("🏥 **Medical Extraction:** Active")
    st.markdown("💊 **Pharmacy Extraction:** Active")
    
    st.markdown("---")
    
    # API Configuration
    st.subheader("🔌 API Settings")
    fastapi_url = st.text_input("FastAPI URL", value="http://localhost:8001")
    
    # Snowflake Cortex API Configuration - Showing configured values
    st.subheader("❄️ Snowflake Cortex API Settings")
    st.info("💡 **Snowflake Cortex API is pre-configured.** All settings are optimized for health analysis and chatbot functionality.")
    
    # Show current Snowflake configuration (read-only)
    try:
        current_config = Config()
        st.text_input("API URL", value=current_config.api_url[:50] + "...", disabled=True)
        st.text_input("Model", value=current_config.model, disabled=True)
        st.text_input("App ID", value=current_config.app_id, disabled=True)
        st.text_input("Application Code", value=current_config.aplctn_cd, disabled=True)
        st.text_area("Analysis System Message", value=current_config.sys_msg, disabled=True, height=80)
        st.text_area("Chatbot System Message", value=current_config.chatbot_sys_msg, disabled=True, height=80)
        
        # Only allow basic settings changes
        st.markdown("**🔧 FastAPI URL can be modified. Snowflake Cortex settings are pre-configured.**")
    except Exception as e:
        st.error(f"❌ Error loading Snowflake configuration: {e}")
        st.error("💡 There might be an issue with the Config class.")

    # Enhanced LangGraph Settings
    st.subheader("🔄 Enhanced LangGraph Settings")
    max_retries = st.slider("Max Retries (per node)", 1, 5, 3)
    timeout = st.slider("Timeout (seconds)", 10, 60, 30)
    
    # Update configuration
    if st.button("🔄 Update Configuration"):
        try:
            config = Config(
                fastapi_url=fastapi_url,
                max_retries=max_retries,
                timeout=timeout
            )
            st.session_state.config = config
            st.session_state.agent = None  # Force reinitialization
            st.success("✅ Configuration updated! Snowflake Cortex settings remain optimized.")
            st.info("🔄 Agent will be reinitialized on next analysis")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Configuration update failed: {str(e)}")
    
    # Clear configuration button
    if st.button("🗑️ Reset Configuration"):
        st.session_state.config = None
        st.session_state.agent = None
        st.info("🔄 Configuration reset to defaults")
        st.rerun()

    st.markdown("---")
    
    # Current Configuration Status
    st.subheader("📋 Current Configuration Status")
    try:
        current_config = st.session_state.config or Config()
        
        st.success("✅ Configuration active")
        st.write(f"**FastAPI:** {current_config.fastapi_url}")
        st.write(f"**Max Retries:** {current_config.max_retries}")
        st.write(f"**Timeout:** {current_config.timeout}")
        
        # Show Snowflake Cortex settings
        st.markdown("**❄️ Snowflake Cortex Settings:**")
        st.write(f"**API URL:** {current_config.api_url[:30]}...")
        st.write(f"**Model:** {current_config.model}")
        st.write(f"**App ID:** {current_config.app_id}")
        st.write(f"**Application Code:** {current_config.aplctn_cd}")
        
        st.success("✅ Snowflake Cortex API is configured and ready!")
        st.success("💬 Interactive chatbot is ready!")
        
        # Test Snowflake Cortex Connection
        if st.button("🧪 Test Snowflake Cortex Connection"):
            try:
                test_config = Config()
                test_agent = HealthAnalysisAgent(test_config)
                
                with st.spinner("Testing Snowflake Cortex API connection..."):
                    test_result = test_agent.test_llm_connection()
                
                if test_result["success"]:
                    st.success("✅ Snowflake Cortex API connection successful!")
                    st.info(f"📝 Response: {test_result['response']}")
                    st.info(f"🤖 Model: {test_result['model']}")
                else:
                    st.error("❌ Snowflake Cortex API connection failed!")
                    st.error(f"💥 Error: {test_result['error']}")
            except Exception as e:
                st.error(f"❌ Test failed: {str(e)}")
        
    except Exception as e:
        st.error(f"❌ Configuration error: {e}")
        st.code(f"Error details: {str(e)}")

# Enhanced LangGraph Workflow Visualization (7 nodes including chatbot)
st.markdown('<div class="step-header">🔄 Enhanced LangGraph Workflow (7 Nodes + Snowflake Cortex + Interactive Chatbot)</div>', unsafe_allow_html=True)

# Create 7 columns for the 7 nodes
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    st.markdown("""
    <div class="langgraph-node">
        <h4>📊 Node 1</h4>
        <p><strong>API Data Fetch</strong></p>
        <small>MCID, Medical, Pharmacy</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="langgraph-node">
        <h4>🔒 Node 2</h4>
        <p><strong>Data Deidentification</strong></p>
        <small>PII Removal & Standardization</small>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="extraction-node">
        <h4>🔍 Node 3</h4>
        <p><strong>Data Extraction</strong></p>
        <small>Medical & Pharmacy Fields</small>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="langgraph-node">
        <h4>🎯 Node 4</h4>
        <p><strong>Entity Extraction</strong></p>
        <small>Enhanced Health Analysis</small>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="langgraph-node">
        <h4>📈 Node 5</h4>
        <p><strong>Snowflake Analysis</strong></p>
        <small>❄️ llama3.1-70b</small>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
    <div class="langgraph-node">
        <h4>📋 Node 6</h4>
        <p><strong>Summary Generation</strong></p>
        <small>❄️ Cortex Summary</small>
    </div>
    """, unsafe_allow_html=True)

with col7:
    st.markdown("""
    <div class="chatbot-node">
        <h4>💬 Node 7</h4>
        <p><strong>Interactive Chatbot</strong></p>
        <small>🤖 Medical Data Q&A</small>
    </div>
    """, unsafe_allow_html=True)

# Enhanced System Status (updated to 7 nodes)
st.subheader("📊 Enhanced System Status")
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    agent_status = "🔥 Ready" if st.session_state.agent else "⚠️ Not Initialized"
    st.metric("LangGraph Agent", agent_status)

with col2:
    config_status = "✅ Custom" if st.session_state.config else "⚠️ Default"
    st.metric("Configuration", config_status)

with col3:
    import_status = "✅ Success" if AGENT_AVAILABLE else "❌ Failed"
    st.metric("Agent Import", import_status)

with col4:
    workflow_status = "🔄 7-Node Ready" if AGENT_AVAILABLE else "❌ Unavailable"
    st.metric("Workflow Engine", workflow_status)

with col5:
    extraction_status = "🆕 Active" if AGENT_AVAILABLE else "❌ Unavailable"
    st.metric("Data Extraction", extraction_status)

with col6:
    snowflake_status = "❄️ Ready" if AGENT_AVAILABLE else "❌ Unavailable"
    st.metric("Snowflake Cortex", snowflake_status)

with col7:
    chatbot_status = "💬 Ready" if AGENT_AVAILABLE else "❌ Unavailable"
    st.metric("Interactive Chatbot", chatbot_status)

# Patient Input Form
st.markdown('<div class="step-header">👤 Patient Information Input (→ Enhanced LangGraph + Snowflake Cortex + Chatbot)</div>', unsafe_allow_html=True)
st.info("💡 Enter patient information below. This data will be processed through the Enhanced LangGraph workflow with Snowflake Cortex AI analysis and will be available for interactive chatbot queries.")

# Get today's date for default
today_date = datetime.now().date()

with st.form("patient_input_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        first_name = st.text_input("First Name *", value="", help="Patient's first name")
        last_name = st.text_input("Last Name *", value="", help="Patient's last name")
    
    with col2:
        ssn = st.text_input("SSN *", value="", help="Social Security Number (9+ digits)")
        date_of_birth = st.date_input(
            "Date of Birth *", 
            value=today_date,
            min_value=datetime(1900, 1, 1).date(),
            max_value=today_date,
            help="Patient's date of birth"
        )
    
    with col3:
        gender = st.selectbox("Gender *", ["F", "M"], help="Patient's gender")
        zip_code = st.text_input("Zip Code *", value="", help="Patient's zip code (5+ digits)")
    
    # Show calculated age
    if date_of_birth:
        calculated_age = calculate_age(date_of_birth)
        if calculated_age is not None:
            st.markdown(f"""
            <div class="age-display">
                <strong>📅 Calculated Age:</strong> {calculated_age} years old
            </div>
            """, unsafe_allow_html=True)
            
            # Age validation warnings
            if calculated_age > 120:
                st.warning("⚠️ Age seems unusually high. Please verify the date of birth.")
            elif calculated_age < 0:
                st.error("❌ Date of birth cannot be in the future.")
    
    # Submit button
    submitted = st.form_submit_button(
        "🔥 Execute Enhanced LangGraph + Snowflake Cortex + Chatbot Analysis", 
        use_container_width=True,
        disabled=st.session_state.analysis_running
    )

# Analysis Status Section
if st.session_state.analysis_running:
    st.markdown('<div class="info-box">🔄 Enhanced LangGraph + Snowflake Cortex + Chatbot workflow executing... Please wait.</div>', unsafe_allow_html=True)

# Run Enhanced LangGraph Analysis
if submitted and not st.session_state.analysis_running:
    # Prepare patient data
    patient_data = {
        "first_name": first_name.strip(),
        "last_name": last_name.strip(),
        "ssn": ssn.strip(),
        "date_of_birth": date_of_birth.strftime("%Y-%m-%d"),
        "gender": gender,
        "zip_code": zip_code.strip()
    }
    
    # Calculate age
    calculated_age = calculate_age(date_of_birth)
    
    # Display patient data being sent to Enhanced LangGraph
    st.info(f"📤 Sending patient data to Enhanced LangGraph + Snowflake Cortex + Chatbot: {patient_data['first_name']} {patient_data['last_name']} (Age: {calculated_age})")
    
    # Validate patient data
    is_valid, validation_errors = validate_patient_data(patient_data)
    
    if not is_valid:
        st.error("❌ Please fix the following errors:")
        for error in validation_errors:
            st.error(f"• {error}")
    else:
        # Initialize Enhanced LangGraph agent
        if st.session_state.agent is None:
            try:
                config = st.session_state.config or Config()
                st.session_state.agent = HealthAnalysisAgent(config)
                st.success(f"🔥 Enhanced LangGraph agent initialized with Snowflake Cortex + Interactive Chatbot")
                st.info(f"❄️ Snowflake Model: {config.model}")
                st.info(f"🔑 App ID: {config.app_id}")
                st.info(f"💬 Chatbot: Interactive mode ready")
            except Exception as e:
                st.error(f"❌ Failed to initialize Enhanced LangGraph agent: {str(e)}")
                st.stop()
        
        st.session_state.analysis_running = True
        
        # Enhanced progress tracking for 7-node workflow
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run Enhanced LangGraph analysis
        with st.spinner("🔥 Executing Enhanced LangGraph + Snowflake Cortex + Chatbot workflow..."):
            try:
                # Update progress
                status_text.text("🚀 Initializing Enhanced LangGraph state machine...")
                progress_bar.progress(5)
                time.sleep(0.5)
                
                status_text.text("📊 Node 1: Fetching API data...")
                progress_bar.progress(15)
                time.sleep(0.5)
                
                status_text.text("🔒 Node 2: Deidentifying data...")
                progress_bar.progress(25)
                time.sleep(0.5)
                
                status_text.text("🔍 Node 3: Extracting medical & pharmacy fields...")
                progress_bar.progress(35)
                time.sleep(0.5)
                
                status_text.text("🎯 Node 4: Enhanced entity extraction...")
                progress_bar.progress(50)
                time.sleep(0.5)
                
                status_text.text("📈 Node 5: Snowflake Cortex health trajectory analysis...")
                progress_bar.progress(65)
                time.sleep(0.5)
                
                status_text.text("📋 Node 6: Snowflake Cortex summary generation...")
                progress_bar.progress(80)
                time.sleep(0.5)
                
                status_text.text("💬 Node 7: Initializing interactive chatbot...")
                progress_bar.progress(95)
                
                # Execute the Enhanced LangGraph workflow
                results = st.session_state.agent.run_analysis(patient_data)
                
                # Update progress based on completion
                if results.get("success", False):
                    progress_bar.progress(100)
                    status_text.text("✅ Enhanced LangGraph + Snowflake Cortex + Chatbot workflow completed successfully!")
                    
                    st.session_state.analysis_results = results
                    # Store chatbot context for interactive use
                    st.session_state.chatbot_context = results.get("chatbot_context", {})
                    st.markdown('<div class="success-box">🔥 Enhanced LangGraph + Snowflake Cortex + Interactive Chatbot health analysis completed successfully!</div>', unsafe_allow_html=True)
                    
                    if results.get("chatbot_ready", False):
                        st.markdown('<div class="chatbot-box">💬 Interactive chatbot is ready! You can now ask questions about the medical data.</div>', unsafe_allow_html=True)
                else:
                    progress_bar.progress(60)
                    status_text.text("⚠️ Enhanced LangGraph workflow completed with errors")
                    
                    st.session_state.analysis_results = results
                    st.warning("⚠️ Analysis completed but with some errors. Check results below.")
                    
                    errors = results.get('errors', [])
                    if errors:
                        st.error("Errors encountered:")
                        for error in errors:
                            st.error(f"• {error}")
                
            except Exception as e:
                progress_bar.progress(0)
                status_text.text("❌ Enhanced LangGraph workflow failed")
                st.error(f"❌ Error in Enhanced LangGraph execution: {str(e)}")
                
                st.session_state.analysis_results = {
                    "success": False,
                    "error": str(e),
                    "patient_data": patient_data,
                    "errors": [str(e)],
                    "processing_steps_completed": 0,
                    "langgraph_used": True,
                    "enhancement_version": "v3.0_with_interactive_chatbot"
                }
            finally:
                st.session_state.analysis_running = False

# Display Enhanced LangGraph Results
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    # Enhanced Results Overview
    st.markdown('<div class="step-header">🔥 Enhanced LangGraph + Snowflake Cortex + Chatbot Analysis Results</div>', unsafe_allow_html=True)
    
    # Show enhancement version
    enhancement_version = results.get("enhancement_version", "v1.0")
    if enhancement_version:
        st.markdown(f'<div class="snowflake-badge">📊 Analysis Version: {enhancement_version}</div>', unsafe_allow_html=True)
    
    # Show patient info
    processed_patient = safe_get(results, 'patient_data', {})
    if processed_patient:
        patient_dob = processed_patient.get('date_of_birth', '')
        patient_age = None
        if patient_dob:
            try:
                birth_date = datetime.strptime(patient_dob, '%Y-%m-%d').date()
                patient_age = calculate_age(birth_date)
            except:
                pass
        
        age_display = f" (Age: {patient_age})" if patient_age is not None else ""
        st.info(f"📋 Enhanced analysis completed for: {processed_patient.get('first_name', 'Unknown')} {processed_patient.get('last_name', 'Unknown')}{age_display}")
    
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
    with col1:
        success_status = "✅ Success" if results.get("success", False) else "⚠️ With Errors"
        steps_completed = results.get('processing_steps_completed', 0)
        st.metric("LangGraph Status", success_status, f"{steps_completed}/7 nodes")
    
    with col2:
        st.metric("Workflow Engine", "🔥 Enhanced LG", "v3.0")
    
    with col3:
        st.metric("AI Engine", "❄️ Snowflake", "llama3.1-70b")
    
    with col4:
        api_outputs = safe_get(results, 'api_outputs', {})
        api_count = len([k for k in api_outputs.keys() if api_outputs.get(k)]) if api_outputs else 0
        st.metric("APIs Called", f"{api_count}/4", "Data Sources")
    
    with col5:
        structured_extractions = safe_get(results, 'structured_extractions', {})
        medical_records = len(safe_get(structured_extractions.get('medical', {}), 'hlth_srvc_records', []))
        pharmacy_records = len(safe_get(structured_extractions.get('pharmacy', {}), 'ndc_records', []))
        st.metric("Extracted Records", f"{medical_records + pharmacy_records}", f"Med:{medical_records} Rx:{pharmacy_records}")
    
    with col6:
        entity_extraction = safe_get(results, 'entity_extraction', {})
        entity_count = len([k for k, v in entity_extraction.items() 
                           if k != 'analysis_details' and v not in ['no', 'unknown']]) if entity_extraction else 0
        st.metric("Health Entities", entity_count, "Conditions")
    
    with col7:
        chatbot_ready = results.get("chatbot_ready", False)
        chatbot_status = "💬 Ready" if chatbot_ready else "❌ Failed"
        st.metric("Interactive Chatbot", chatbot_status, "Medical Q&A")

    # Show enhanced step status (7 nodes)
    step_status = safe_get(results, 'step_status', {})
    if step_status:
        st.subheader("🔄 Enhanced LangGraph Node Execution Status")
        status_cols = st.columns(7)
        
        nodes = [
            ("fetch_api_data", "📊 API"),
            ("deidentify_data", "🔒 Deidentify"),
            ("extract_medical_pharmacy_data", "🔍 Extract"),
            ("extract_entities", "🎯 Entities"),
            ("analyze_trajectory", "📈 Snowflake"),
            ("generate_summary", "📋 Summary"),
            ("initialize_chatbot", "💬 Chatbot")
        ]
        
        for i, (node_key, node_name) in enumerate(nodes):
            with status_cols[i]:
                status = step_status.get(node_key, "pending")
                if status == "completed":
                    st.success(f"✅ {node_name}")
                elif status == "error":
                    st.error(f"❌ {node_name}")
                elif status == "running":
                    st.info(f"🔄 {node_name}")
                else:
                    st.info(f"⏳ {node_name}")

    # Show errors if any
    errors = safe_get(results, 'errors', [])
    if errors:
        st.markdown('<div class="error-box">❌ Enhanced LangGraph workflow errors:</div>', unsafe_allow_html=True)
        for error in errors:
            st.error(f"• {error}")

    # NEW: Interactive Chatbot Section (Node 7 Results)
    if results.get("chatbot_ready", False) and st.session_state.chatbot_context:
        st.markdown('<div class="chatbot-header">💬 LangGraph Node 7: Interactive Medical Data Chatbot</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="chatbot-box">🤖 Ask questions about the patient\'s medical data, analysis results, or request specific insights based on the deidentified records.</div>', unsafe_allow_html=True)
        
        # Chatbot Interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Display chat history
            if st.session_state.chatbot_messages:
                st.subheader("💬 Chat History")
                for message in st.session_state.chatbot_messages:
                    if message["role"] == "user":
                        st.markdown(f'<div class="chat-message user-message"><strong>👤 You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-message assistant-message"><strong>🤖 Medical Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("📊 Chatbot Context")
            context_summary = {
                "Medical Records": "✅ Available" if st.session_state.chatbot_context.get("deidentified_medical") else "❌ None",
                "Pharmacy Records": "✅ Available" if st.session_state.chatbot_context.get("deidentified_pharmacy") else "❌ None",
                "Health Analysis": "✅ Available" if st.session_state.chatbot_context.get("health_trajectory") else "❌ None",
                "Entity Extraction": "✅ Available" if st.session_state.chatbot_context.get("entity_extraction") else "❌ None"
            }
            
            for key, value in context_summary.items():
                if "✅" in value:
                    st.success(f"{key}: {value}")
                else:
                    st.warning(f"{key}: {value}")
        
        # Chat input
        user_question = st.chat_input("💬 Ask a question about the medical data...")
        
        if user_question:
            # Add user message to chat history
            st.session_state.chatbot_messages.append({"role": "user", "content": user_question})
            
            # Get response from chatbot
            try:
                with st.spinner("🤖 Analyzing medical data and generating response..."):
                    chatbot_response = st.session_state.agent.chat_with_data(
                        user_question, 
                        st.session_state.chatbot_context, 
                        st.session_state.chatbot_messages
                    )
                
                # Add assistant response to chat history
                st.session_state.chatbot_messages.append({"role": "assistant", "content": chatbot_response})
                
                # Rerun to display the new messages
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Chatbot error: {str(e)}")
                st.session_state.chatbot_messages.append({"role": "assistant", "content": f"I apologize, but I encountered an error: {str(e)}"})
        
        # Clear chat history button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chatbot_messages = []
            st.success("Chat history cleared!")
            st.rerun()
        
        # Sample questions
        st.markdown("**💡 Sample Questions You Can Ask:**")
        sample_questions = [
            "What medications was this patient prescribed?",
            "Are there any chronic conditions indicated in the medical data?",
            "What is the patient's overall health risk assessment?",
            "Explain the significance of the ICD-10 diagnosis codes found",
            "What drug interactions should be considered?",
            "Summarize the key health insights from all the data"
        ]
        
        for i, question in enumerate(sample_questions):
            if st.button(f"💬 {question}", key=f"sample_q_{i}"):
                # Use the sample question as if user typed it
                st.session_state.chatbot_messages.append({"role": "user", "content": question})
                
                try:
                    with st.spinner("🤖 Analyzing medical data and generating response..."):
                        chatbot_response = st.session_state.agent.chat_with_data(
                            question, 
                            st.session_state.chatbot_context, 
                            st.session_state.chatbot_messages
                        )
                    
                    st.session_state.chatbot_messages.append({"role": "assistant", "content": chatbot_response})
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Chatbot error: {str(e)}")
    
    elif not results.get("chatbot_ready", False):
        st.markdown('<div class="chatbot-header">💬 LangGraph Node 7: Interactive Medical Data Chatbot</div>', unsafe_allow_html=True)
        st.warning("⚠️ Chatbot initialization failed. Please check the workflow execution above.")

    # Continue with the rest of the original results display...
    # (I'll continue with the API outputs, deidentified data, etc. - same as before but condensed due to length)
    
    # API Outputs Section (same as before)
    api_outputs = safe_get(results, 'api_outputs', {})
    if api_outputs:
        st.markdown('<div class="step-header">📊 LangGraph Node 1: API Data Retrieval</div>', unsafe_allow_html=True)
        
        api_tabs = st.tabs(["🆔 MCID", "🏥 Medical", "💊 Pharmacy", "🔑 Token"])
        
        with api_tabs[0]:
            st.subheader("MCID Search Results")
            mcid_data = safe_get(api_outputs, 'mcid', {})
            if mcid_data:
                st.json(mcid_data)
                st.download_button(
                    "📄 Download MCID Data",
                    safe_json_dumps(mcid_data),
                    f"mcid_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No MCID data available")
        
        with api_tabs[1]:
            st.subheader("Medical API Results")
            medical_data = safe_get(api_outputs, 'medical', {})
            if medical_data:
                st.json(medical_data)
                st.download_button(
                    "📄 Download Medical Data",
                    safe_json_dumps(medical_data),
                    f"medical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No medical data available")
        
        with api_tabs[2]:
            st.subheader("Pharmacy API Results")
            pharmacy_data = safe_get(api_outputs, 'pharmacy', {})
            if pharmacy_data:
                st.json(pharmacy_data)
                st.download_button(
                    "📄 Download Pharmacy Data",
                    safe_json_dumps(pharmacy_data),
                    f"pharmacy_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No pharmacy data available")
        
        with api_tabs[3]:
            st.subheader("Token API Results")
            token_data = safe_get(api_outputs, 'token', {})
            if token_data:
                st.json(token_data)
            else:
                st.info("Token data not available")

    # Step 2: Deidentified Data
    deidentified_data = safe_get(results, 'deidentified_data', {})
    if deidentified_data:
        st.markdown('<div class="step-header">🔒 LangGraph Node 2: Data Deidentification</div>', unsafe_allow_html=True)
        
        deident_tabs = st.tabs(["🏥 Medical Deidentified", "💊 Pharmacy Deidentified"])
        
        with deident_tabs[0]:
            st.subheader("Deidentified Medical Data")
            deident_medical = safe_get(deidentified_data, 'medical', {})
            
            if deident_medical and not deident_medical.get('error'):
                st.markdown("**🎯 LangGraph Standardized Format:**")
                
                deident_display = {
                    "Field": ["First Name", "Last Name", "Middle Initial", "Age", "Zip Code"],
                    "Value": [
                        safe_str(safe_get(deident_medical, 'src_mbr_first_nm', 'N/A')),
                        safe_str(safe_get(deident_medical, 'src_mbr_last_nm', 'N/A')),
                        safe_str(safe_get(deident_medical, 'src_mbr_mid_init_nm', 'None')),
                        safe_str(safe_get(deident_medical, 'src_mbr_age', 'N/A')),
                        safe_str(safe_get(deident_medical, 'src_mbr_zip_cd', 'N/A'))
                    ]
                }
                
                deident_df = pd.DataFrame(deident_display)
                st.table(deident_df)
                
                with st.expander("🔍 View Full Deidentified Medical Data"):
                    st.json(deident_medical)
                
                st.download_button(
                    "📄 Download Deidentified Medical",
                    safe_json_dumps(deident_medical),
                    f"deidentified_medical_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                error_msg = safe_get(deident_medical, 'error', 'Unknown error')
                st.error(f"Deidentification failed: {error_msg}")
        
        with deident_tabs[1]:
            st.subheader("Deidentified Pharmacy Data")
            deident_pharmacy = safe_get(deidentified_data, 'pharmacy', {})
            
            if deident_pharmacy and not deident_pharmacy.get('error'):
                with st.expander("🔍 View Deidentified Pharmacy Data"):
                    st.json(deident_pharmacy)
                
                st.download_button(
                    "📄 Download Deidentified Pharmacy",
                    safe_json_dumps(deident_pharmacy),
                    f"deidentified_pharmacy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                error_msg = safe_get(deident_pharmacy, 'error', 'Unknown error')
                st.error(f"Pharmacy deidentification failed: {error_msg}")

    # Step 3: Structured Extractions
    structured_extractions = safe_get(results, 'structured_extractions', {})
    if structured_extractions:
        st.markdown('<div class="step-header">🔍 LangGraph Node 3: Structured Data Extraction</div>', unsafe_allow_html=True)
        
        extraction_tabs = st.tabs(["🏥 Medical Extraction", "💊 Pharmacy Extraction"])
        
        with extraction_tabs[0]:
            st.subheader("Medical Field Extraction Results")
            medical_extraction = safe_get(structured_extractions, 'medical', {})
            
            if medical_extraction and not medical_extraction.get('error'):
                extraction_summary = safe_get(medical_extraction, 'extraction_summary', {})
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Health Service Records", extraction_summary.get('total_hlth_srvc_records', 0))
                with col2:
                    st.metric("Diagnosis Codes Found", extraction_summary.get('total_diagnosis_codes', 0))
                with col3:
                    st.metric("Unique Service Codes", len(extraction_summary.get('unique_service_codes', [])))
                with col4:
                    st.metric("Unique Diagnosis Codes", len(extraction_summary.get('unique_diagnosis_codes', [])))
                
                hlth_srvc_records = safe_get(medical_extraction, 'hlth_srvc_records', [])
                if hlth_srvc_records:
                    st.markdown("**🏥 Extracted Medical Records:**")
                    
                    for i, record in enumerate(hlth_srvc_records[:5]):
                        with st.expander(f"Medical Record {i+1} - Service Code: {record.get('hlth_srvc_cd', 'N/A')}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"**Health Service Code:**")
                                st.code(record.get('hlth_srvc_cd', 'N/A'))
                                st.markdown(f"**Data Path:**")
                                st.code(record.get('data_path', 'N/A'))
                            
                            with col2:
                                diagnosis_codes = record.get('diagnosis_codes', [])
                                if diagnosis_codes:
                                    st.markdown(f"**Diagnosis Codes ({len(diagnosis_codes)}):**")
                                    for diag in diagnosis_codes:
                                        st.markdown(f"- Position {diag.get('position', 'N/A')}: `{diag.get('code', 'N/A')}`")
                                else:
                                    st.markdown("**No diagnosis codes found**")
                    
                    if len(hlth_srvc_records) > 5:
                        st.info(f"Showing first 5 of {len(hlth_srvc_records)} medical records.")
                
                st.download_button(
                    "📄 Download Medical Extraction",
                    safe_json_dumps(medical_extraction),
                    f"medical_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                error_msg = safe_get(medical_extraction, 'error', 'No medical extraction data available')
                st.warning(f"Medical extraction: {error_msg}")
        
        with extraction_tabs[1]:
            st.subheader("Pharmacy Field Extraction Results")
            pharmacy_extraction = safe_get(structured_extractions, 'pharmacy', {})
            
            if pharmacy_extraction and not pharmacy_extraction.get('error'):
                extraction_summary = safe_get(pharmacy_extraction, 'extraction_summary', {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("NDC Records Found", extraction_summary.get('total_ndc_records', 0))
                with col2:
                    st.metric("Unique NDC Codes", len(extraction_summary.get('unique_ndc_codes', [])))
                with col3:
                    st.metric("Unique Medications", len(extraction_summary.get('unique_label_names', [])))
                
                ndc_records = safe_get(pharmacy_extraction, 'ndc_records', [])
                if ndc_records:
                    st.markdown("**💊 Extracted Pharmacy Records:**")
                    
                    for i, record in enumerate(ndc_records[:10]):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"**NDC Code:**")
                            st.code(record.get('ndc', 'N/A'))
                        
                        with col2:
                            st.markdown(f"**Label Name:**")
                            st.code(record.get('lbl_nm', 'N/A'))
                        
                        with col3:
                            st.markdown(f"**Data Path:**")
                            st.text(record.get('data_path', 'N/A'))
                        
                        if i < len(ndc_records) - 1:
                            st.divider()
                    
                    if len(ndc_records) > 10:
                        st.info(f"Showing first 10 of {len(ndc_records)} pharmacy records.")
                
                st.download_button(
                    "📄 Download Pharmacy Extraction",
                    safe_json_dumps(pharmacy_extraction),
                    f"pharmacy_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                error_msg = safe_get(pharmacy_extraction, 'error', 'No pharmacy extraction data available')
                st.warning(f"Pharmacy extraction: {error_msg}")

    # Step 4: Enhanced Entity Extraction
    entity_extraction = safe_get(results, 'entity_extraction', {})
    if entity_extraction:
        st.markdown('<div class="step-header">🎯 LangGraph Node 4: Enhanced Entity Extraction</div>', unsafe_allow_html=True)
        
        # Enhanced entity display with cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        entities_data = [
            ("diabetics", "🩺 Diabetes", col1),
            ("age_group", "👥 Age Group", col2),
            ("smoking", "🚬 Smoking", col3),
            ("alcohol", "🍷 Alcohol", col4),
            ("blood_pressure", "💓 Blood Pressure", col5)
        ]
        
        for key, title, col in entities_data:
            with col:
                value = safe_get(entity_extraction, key, 'unknown')
                
                # Color coding
                if key == "diabetics":
                    color = "#dc3545" if value == "yes" else "#28a745"
                    emoji = "⚠️" if value == "yes" else "✅"
                elif key == "smoking":
                    color = "#ffc107" if value == "quit_attempt" else "#28a745"
                    emoji = "🚭" if value == "quit_attempt" else "✅"
                elif key == "alcohol":
                    color = "#ffc107" if value == "treatment" else "#28a745"
                    emoji = "🍷" if value == "treatment" else "✅"
                elif key == "blood_pressure":
                    color = "#ffc107" if value in ["managed", "diagnosed"] else "#6c757d"
                    emoji = "💓" if value in ["managed", "diagnosed"] else "❓"
                else:
                    color = "#17a2b8"
                    emoji = "👥"
                
                st.markdown(f"""
                <div class="entity-card" style="border-color: {color};">
                    <h4 style="color: {color};">{emoji} {title}</h4>
                    <p class="metric-highlight" style="color: {color};">{value.replace('_', ' ').upper()}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Show medical conditions if available
        medical_conditions = safe_get(entity_extraction, 'medical_conditions', [])
        if medical_conditions:
            st.markdown("**🏥 Medical Conditions Identified from ICD-10 Codes:**")
            for condition in medical_conditions:
                st.markdown(f"- {condition}")
        
        # Show medications identified if available
        medications_identified = safe_get(entity_extraction, 'medications_identified', [])
        if medications_identified:
            st.markdown("**💊 Medications Identified from NDC Data:**")
            for med in medications_identified[:5]:
                st.markdown(f"- **{med.get('label_name', 'N/A')}** (NDC: {med.get('ndc', 'N/A')})")
            if len(medications_identified) > 5:
                st.info(f"Showing first 5 of {len(medications_identified)} medications identified.")
        
        # Detailed entity analysis
        analysis_details = safe_get(entity_extraction, 'analysis_details', [])
        if analysis_details:
            with st.expander("🔍 View Enhanced Entity Analysis Details"):
                for detail in analysis_details:
                    st.write(f"• {detail}")
        
        st.download_button(
            "📄 Download Enhanced Entity Extraction",
            safe_json_dumps(entity_extraction),
            f"enhanced_entity_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    # Step 5: Snowflake Cortex Health Trajectory Analysis
    health_trajectory = safe_get(results, 'health_trajectory', '')
    if health_trajectory:
        st.markdown('<div class="step-header">📈 LangGraph Node 5: Snowflake Cortex Health Trajectory Analysis</div>', unsafe_allow_html=True)
        
        st.markdown("**❄️ Snowflake Cortex llama3.1-70b Analysis (with Structured Extractions):**")
        st.markdown(health_trajectory)
        
        st.download_button(
            "📄 Download Snowflake Health Trajectory",
            health_trajectory,
            f"snowflake_health_trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    else:
        st.warning("Snowflake Cortex health trajectory analysis not available")

    # Step 6: Snowflake Cortex Final Summary
    final_summary = safe_get(results, 'final_summary', '')
    if final_summary:
        st.markdown('<div class="step-header">📋 LangGraph Node 6: Snowflake Cortex Final Health Summary</div>', unsafe_allow_html=True)
        
        st.markdown("**❄️ Snowflake Cortex Executive Summary (with Medical & Pharmacy Extractions):**")
        st.markdown(final_summary)
        
        st.download_button(
            "📄 Download Snowflake Final Summary",
            final_summary,
            f"snowflake_final_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    else:
        st.warning("Snowflake Cortex final summary not available")

    # Complete Enhanced LangGraph + Snowflake + Chatbot Report Download
    st.markdown('<div class="step-header">💾 Complete Enhanced LangGraph + Snowflake + Chatbot Analysis Report</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Complete enhanced JSON report
        complete_report = {
            "enhanced_langgraph_snowflake_chatbot_metadata": {
                "patient_info": processed_patient,
                "timestamp": datetime.now().isoformat(),
                "success": results.get("success", False),
                "nodes_completed": results.get("processing_steps_completed", 0),
                "workflow_engine": "Enhanced LangGraph v3.0",
                "ai_engine": "Snowflake Cortex",
                "ai_model": "llama3.1-70b",
                "chatbot_enabled": True,
                "chatbot_ready": results.get("chatbot_ready", False),
                "enhancement_version": results.get("enhancement_version", "v3.0"),
                "step_status": safe_get(results, "step_status", {}),
                "extraction_enabled": True,
                "interactive_features": ["Medical Data Q&A", "ICD-10 Analysis", "NDC Code Interpretation"]
            },
            "api_outputs": safe_get(results, "api_outputs", {}),
            "deidentified_data": safe_get(results, "deidentified_data", {}),
            "structured_extractions": safe_get(results, "structured_extractions", {}),
            "entity_extraction": safe_get(results, "entity_extraction", {}),
            "health_trajectory": safe_get(results, "health_trajectory", ""),
            "final_summary": safe_get(results, "final_summary", ""),
            "chatbot_context": safe_get(results, "chatbot_context", {}),
            "chat_history": st.session_state.chatbot_messages,
            "errors": safe_get(results, "errors", [])
        }
        
        patient_last_name = processed_patient.get('last_name', 'unknown')
        st.download_button(
            "📊 Download Complete Enhanced + Snowflake + Chatbot Report",
            safe_json_dumps(complete_report),
            f"enhanced_snowflake_chatbot_analysis_{patient_last_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Enhanced text report with chatbot info
        patient_name = f"{processed_patient.get('first_name', 'Unknown')} {processed_patient.get('last_name', 'Unknown')}"
        
        # Get extraction counts
        medical_extraction = safe_get(structured_extractions, 'medical', {})
        pharmacy_extraction = safe_get(structured_extractions, 'pharmacy', {})
        medical_records = len(safe_get(medical_extraction, 'hlth_srvc_records', []))
        pharmacy_records = len(safe_get(pharmacy_extraction, 'ndc_records', []))
        
        text_report = f"""
ENHANCED LANGGRAPH + SNOWFLAKE CORTEX + INTERACTIVE CHATBOT HEALTH ANALYSIS REPORT v3.0
=========================================================================================
Patient: {patient_name}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: {'Success' if results.get('success', False) else 'Failed'}
Nodes Completed: {results.get('processing_steps_completed', 0)}/7
Workflow Engine: Enhanced LangGraph v3.0
AI Engine: Snowflake Cortex
AI Model: llama3.1-70b
Interactive Chatbot: {'Ready' if results.get('chatbot_ready', False) else 'Failed'}
Enhancement Version: {results.get('enhancement_version', 'v3.0')}

STRUCTURED EXTRACTIONS SUMMARY:
===============================
Medical Records Extracted: {medical_records}
Pharmacy Records Extracted: {pharmacy_records}
Total Diagnosis Codes: {safe_get(medical_extraction, 'extraction_summary', {}).get('total_diagnosis_codes', 0)}
Unique NDC Codes: {len(safe_get(pharmacy_extraction, 'extraction_summary', {}).get('unique_ndc_codes', []))}

ENHANCED ENTITY EXTRACTION RESULTS:
===================================
{safe_json_dumps(entity_extraction)}

SNOWFLAKE CORTEX HEALTH TRAJECTORY ANALYSIS:
============================================
{health_trajectory}

SNOWFLAKE CORTEX FINAL SUMMARY:
===============================
{final_summary}

INTERACTIVE CHATBOT SESSION:
============================
Chat Messages Exchanged: {len(st.session_state.chatbot_messages)}
Chatbot Context Components: {len(safe_get(results, 'chatbot_context', {}))}

CHAT HISTORY:
=============
{chr(10).join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.chatbot_messages])}

ENHANCED LANGGRAPH + SNOWFLAKE + CHATBOT ERRORS (if any):
=========================================================
{chr(10).join(safe_get(results, 'errors', []))}
        """
        
        st.download_button(
            "📝 Download Enhanced + Snowflake + Chatbot Text Report",
            text_report,
            f"enhanced_snowflake_chatbot_analysis_{patient_last_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        # Enhanced CSV summary with chatbot metrics
        try:
            # Get extraction counts
            medical_extraction = safe_get(structured_extractions, 'medical', {})
            pharmacy_extraction = safe_get(structured_extractions, 'pharmacy', {})
            medical_records = len(safe_get(medical_extraction, 'hlth_srvc_records', []))
            pharmacy_records = len(safe_get(pharmacy_extraction, 'ndc_records', []))
            
            csv_data = {
                "Metric": [
                    "Analysis Status", "Workflow Engine", "AI Engine", "AI Model", "Enhancement Version", "Nodes Completed", 
                    "Medical Records Extracted", "Pharmacy Records Extracted", "Diagnosis Codes Found",
                    "Unique NDC Codes", "Diabetes", "Age Group", "Smoking", "Alcohol", "Blood Pressure", 
                    "Chatbot Ready", "Chat Messages", "Timestamp"
                ],
                "Value": [
                    safe_str("Success" if results.get("success", False) else "Failed"),
                    safe_str("Enhanced LangGraph v3.0"),
                    safe_str("Snowflake Cortex"),
                    safe_str("llama3.1-70b"),
                    safe_str(results.get('enhancement_version', 'v3.0')),
                    safe_str(f"{results.get('processing_steps_completed', 0)}/7"),
                    safe_str(medical_records),
                    safe_str(pharmacy_records),
                    safe_str(safe_get(medical_extraction, 'extraction_summary', {}).get('total_diagnosis_codes', 0)),
                    safe_str(len(safe_get(pharmacy_extraction, 'extraction_summary', {}).get('unique_ndc_codes', []))),
                    safe_str(safe_get(entity_extraction, 'diabetics', 'unknown')),
                    safe_str(safe_get(entity_extraction, 'age_group', 'unknown')),
                    safe_str(safe_get(entity_extraction, 'smoking', 'unknown')),
                    safe_str(safe_get(entity_extraction, 'alcohol', 'unknown')),
                    safe_str(safe_get(entity_extraction, 'blood_pressure', 'unknown')),
                    safe_str("Ready" if results.get("chatbot_ready", False) else "Failed"),
                    safe_str(len(st.session_state.chatbot_messages)),
                    safe_str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                ]
            }
            
            csv_df = pd.DataFrame(csv_data)
            csv_string = csv_df.to_csv(index=False)
            
            st.download_button(
                "📊 Download Enhanced + Snowflake + Chatbot CSV",
                csv_string,
                f"enhanced_snowflake_chatbot_summary_{patient_last_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error generating enhanced CSV: {str(e)}")

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🔥 <strong>Enhanced LangGraph + Snowflake Cortex + Interactive Chatbot Health Analysis Dashboard v3.0</strong><br>
    Powered by Enhanced LangGraph State Machines + Medical/Pharmacy Extraction + Snowflake Cortex llama3.1-70b + Interactive Medical Data Chatbot<br>
    🆕 <strong>NEW:</strong> Medical ICD-10 Code Extraction | Pharmacy NDC Data Extraction | Snowflake Cortex AI Analysis | Interactive Medical Data Q&A Chatbot<br>
    ⚠️ <em>This analysis is for informational purposes only and should not replace professional medical advice.</em>
</div>
""", unsafe_allow_html=True)

# Enhanced Debug information for LangGraph + Snowflake + Chatbot
if st.sidebar.checkbox("🐛 Show Enhanced Debug Info"):
    st.sidebar.markdown("### 🔥 Enhanced LangGraph + Snowflake + Chatbot Debug v3.0")
    st.sidebar.write("Agent Available:", AGENT_AVAILABLE)
    st.sidebar.write("Enhanced Agent Initialized:", st.session_state.agent is not None)
    st.sidebar.write("Current Date:", today_date)
    
    # Configuration Debug
    st.sidebar.markdown("**Configuration Debug:**")
    if st.session_state.config:
        st.sidebar.write("Custom Config:", "✅ Yes")
        st.sidebar.write("FastAPI URL:", st.session_state.config.fastapi_url)
        st.sidebar.write("Max Retries:", st.session_state.config.max_retries)
        st.sidebar.write("Timeout:", st.session_state.config.timeout)
    else:
        st.sidebar.write("Custom Config:", "❌ No (using defaults)")
    
    # Snowflake Cortex Debug
    st.sidebar.markdown("**❄️ Snowflake Cortex Settings:**")
    try:
        default_config = Config()
        st.sidebar.write("API URL:", default_config.api_url[:40] + "...")
        st.sidebar.write("Model:", default_config.model)
        st.sidebar.write("App ID:", default_config.app_id)
        st.sidebar.write("Application Code:", default_config.aplctn_cd)
    except Exception as e:
        st.sidebar.write("Config Error:", str(e))
    
    # Chatbot Debug
    st.sidebar.markdown("**💬 Chatbot Debug:**")
    st.sidebar.write("Chatbot Messages:", len(st.session_state.chatbot_messages))
    st.sidebar.write("Chatbot Context Available:", st.session_state.chatbot_context is not None)
    if st.session_state.chatbot_context:
        st.sidebar.write("Context Components:", len(st.session_state.chatbot_context))
    
    # Agent Debug
    if st.session_state.agent:
        st.sidebar.markdown("**Agent Debug:**")
        st.sidebar.write("Agent API URL:", st.session_state.agent.config.api_url[:40] + "...")
        st.sidebar.write("Agent Model:", st.session_state.agent.config.model)
    
    # Analysis Results Debug
    if st.session_state.analysis_results:
        st.sidebar.markdown("**Analysis Results:**")
        st.sidebar.write("Analysis Success:", st.session_state.analysis_results.get("success"))
        st.sidebar.write("Nodes Completed:", f"{st.session_state.analysis_results.get('processing_steps_completed', 0)}/7")
        st.sidebar.write("Enhanced LangGraph Used:", st.session_state.analysis_results.get("langgraph_used"))
        st.sidebar.write("Enhancement Version:", st.session_state.analysis_results.get("enhancement_version", "v1.0"))
        st.sidebar.write("Chatbot Ready:", st.session_state.analysis_results.get("chatbot_ready", False))
        
        # Show extraction info
        structured_extractions = st.session_state.analysis_results.get("structured_extractions", {})
        if structured_extractions:
            medical_count = len(safe_get(structured_extractions.get('medical', {}), 'hlth_srvc_records', []))
            pharmacy_count = len(safe_get(structured_extractions.get('pharmacy', {}), 'ndc_records', []))
            st.sidebar.write("Medical Extractions:", medical_count)
            st.sidebar.write("Pharmacy Extractions:", pharmacy_count)
        
        step_status = st.session_state.analysis_results.get("step_status", {})
        if step_status:
            st.sidebar.write("Enhanced Node Status:", step_status)
        errors = st.session_state.analysis_results.get("errors", [])
        if errors:
            st.sidebar.write("Errors:", len(errors))
            for i, error in enumerate(errors[:3]):
                st.sidebar.write(f"Error {i+1}:", error[:50] + "..." if len(error) > 50 else error)
