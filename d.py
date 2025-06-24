
# Configure Streamlit page FIRST - before any other Streamlit commands
import streamlit as st
st.set_page_config(
    page_title="Health Agent with Heart Attack Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
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

# Import the Enhanced LangGraph health analysis agent with Combined Heart Attack Prediction
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

# Custom CSS for Enhanced LangGraph + Snowflake + Chatbot + Combined Heart Attack Prediction themed styling
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

.heart-badge {
    background: linear-gradient(45deg, #dc3545, #c82333);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 0.8rem;
    font-weight: bold;
    display: inline-block;
    margin: 0.2rem;
    font-size: 0.9rem;
}

.combined-badge {
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

.heart-header {
    font-size: 1.6rem;
    color: #dc3545;
    border-left: 4px solid #dc3545;
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

.heart-box {
    background: linear-gradient(135deg, #ffe6e6, #ffcccc);
    border: 2px solid #dc3545;
    color: #721c24;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    border-radius: 0.5rem;
}

.combined-box {
    background: linear-gradient(135deg, #e6ffe6, #ccffcc);
    border: 2px solid #28a745;
    color: #155724;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    border-radius: 0.5rem;
}

.risk-high {
    background: linear-gradient(135deg, #f8d7da, #f5c6cb);
    border: 3px solid #dc3545;
    color: #721c24;
    padding: 1.5rem;
    border-radius: 0.8rem;
    font-weight: bold;
}

.risk-moderate {
    background: linear-gradient(135deg, #fff3cd, #ffeaa7);
    border: 3px solid #ffc107;
    color: #856404;
    padding: 1.5rem;
    border-radius: 0.8rem;
    font-weight: bold;
}

.risk-low {
    background: linear-gradient(135deg, #d4edda, #c3e6cb);
    border: 3px solid #28a745;
    color: #155724;
    padding: 1.5rem;
    border-radius: 0.8rem;
    font-weight: bold;
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
    # Chatbot session state
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

# Main Title with Health Agent branding
st.markdown('<h1 class="main-header">🏥 Health Agent with Combined Heart Attack Prediction</h1>', unsafe_allow_html=True)
st.markdown("**Advanced health analysis with comprehensive medical data extraction, interactive chatbot, and combined ML-powered heart attack risk prediction**")

# Display import status AFTER page config
if AGENT_AVAILABLE:
    st.success("✅ Health Agent with Combined Heart Attack Prediction imported successfully!")
else:
    st.error(f"❌ Failed to import Health Agent: {import_error}")
    
    with st.expander("🔧 Health Agent Installation Guide"):
        st.markdown("""
        **Install Health Agent Requirements:**
        ```bash
        pip install langgraph langchain-core streamlit requests urllib3 pandas numpy scikit-learn
        ```
        
        **Required Files:**
        - `langgraph_health_agent_proper.py` (the Health Agent)
        - `streamlit_langgraph_ui.py` (this file)
        - `heart_attack_combined_model.pkl` (your combined trained ML model)
        
        **Combined PKL Structure:**
        Your PKL file should contain:
        - Model: AdaBoostClassifier or similar
        - Scaler: StandardScaler (optional)
        - Features: ['Age', 'Gender', 'Diabetes', 'High_BP', 'Smoking']
        
        **Health Agent Features:**
        - ✅ State management and persistence
        - ✅ Conditional workflow routing  
        - ✅ Automatic retry mechanisms
        - ✅ Error handling and recovery
        - ✅ Checkpointing for reliability
        - ✅ **Medical field extraction (hlth_srvc_cd, diag_1_50_cd)**
        - ✅ **Pharmacy field extraction (Ndc, lbl_nm)**
        - ✅ **Enhanced entity detection with ICD-10 codes**
        - ✅ **Combined Heart Attack Risk Prediction using single ML model PKL**
        - ✅ **Interactive chatbot with medical data context**
        """)
    st.stop()

# Enhanced Sidebar Configuration with Combined Heart Attack Prediction
with st.sidebar:
    st.header("⚙️ Health Agent Configuration")
    
    # Health Agent Status
    st.markdown("### 📊 System Status")
    st.markdown("✅ **Health Agent Enabled**")
    st.markdown("🔄 **State Management:** Active")
    st.markdown("💾 **Checkpointing:** Enabled")
    st.markdown("🔁 **Retry Logic:** Configured")
    st.markdown("🏥 **Medical Extraction:** Active")
    st.markdown("💊 **Pharmacy Extraction:** Active")
    st.markdown("❤️ **Combined Heart Attack Prediction:** Active")  # NEW
    st.markdown("💬 **Interactive Chatbot:** Ready")
    
    st.markdown("---")
    
    # Enhanced API Configuration
    st.subheader("🔌 API Settings")
    fastapi_url = st.text_input("FastAPI URL", value="http://localhost:8001")
    
    # NEW: Combined Heart Attack Prediction Configuration
    st.subheader("❤️ Combined Heart Attack Prediction Settings")
    st.markdown('<div class="combined-badge">📦 Single Combined PKL File</div>', unsafe_allow_html=True)
    
    heart_attack_combined_pkl_path = st.text_input(
        "Combined Heart Attack Model PKL Path *", 
        value="heart_attack_combined_model.pkl",
        help="Path to the combined trained heart attack prediction model (PKL file containing model, scaler, and features)"
    )
    
    heart_attack_threshold = st.slider(
        "Risk Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Threshold for high/low risk classification"
    )
    
    # File upload for combined PKL file
    st.markdown("**📁 Or Upload Combined PKL File:**")
    uploaded_combined_model = st.file_uploader(
        "Upload Combined Heart Attack Model", 
        type=['pkl'], 
        key="combined_model_upload",
        help="Upload your combined trained heart attack prediction model (contains model + scaler + features)"
    )
    
    # Combined model info display
    if os.path.exists(heart_attack_combined_pkl_path):
        st.success(f"✅ Combined model file found: {heart_attack_combined_pkl_path}")
        
        # Try to show model info
        try:
            import pickle
            with open(heart_attack_combined_pkl_path, 'rb') as f:
                combined_data = pickle.load(f)
            
            if isinstance(combined_data, dict):
                model_type = type(combined_data.get('model', 'Unknown')).__name__
                has_scaler = 'scaler' in combined_data
                features = combined_data.get('feature_names', ['Age', 'Gender', 'Diabetes', 'High_BP', 'Smoking'])
            elif hasattr(combined_data, 'model'):
                model_type = type(combined_data.model).__name__
                has_scaler = hasattr(combined_data, 'scaler')
                features = getattr(combined_data, 'feature_names', ['Age', 'Gender', 'Diabetes', 'High_BP', 'Smoking'])
            else:
                model_type = type(combined_data).__name__
                has_scaler = False
                features = ['Age', 'Gender', 'Diabetes', 'High_BP', 'Smoking']
            
            st.info(f"🤖 Model Type: {model_type}")
            st.info(f"🔧 Scaler Included: {'Yes' if has_scaler else 'No'}")
            st.info(f"📊 Features: {len(features)}")
            
            with st.expander("🔍 View Model Features"):
                for i, feature in enumerate(features):
                    st.write(f"{i+1}. {feature}")
        except Exception as e:
            st.warning(f"⚠️ Could not read model info: {str(e)}")
    else:
        st.warning(f"⚠️ Combined model file not found: {heart_attack_combined_pkl_path}")
    
    # API Configuration - Showing configured values
    st.subheader("🔧 Snowflake API Settings")
    st.info("💡 **API settings are pre-configured.** All settings are optimized for health analysis.")
    
    # Show current configuration (read-only)
    try:
        current_config = Config()
        st.text_input("API URL", value=current_config.api_url[:50] + "...", disabled=True)
        st.text_input("Model", value=current_config.model, disabled=True)
        st.text_input("App ID", value=current_config.app_id, disabled=True)
        st.text_input("Application Code", value=current_config.aplctn_cd, disabled=True)
        st.text_area("Analysis System Message", value=current_config.sys_msg, disabled=True, height=80)
        st.text_area("Chatbot System Message", value=current_config.chatbot_sys_msg, disabled=True, height=80)
        
        st.markdown("**🔧 FastAPI URL and Combined Heart Attack Prediction settings can be modified. API settings are pre-configured.**")
    except Exception as e:
        st.error(f"❌ Error loading configuration: {e}")

    # Settings
    st.subheader("🔄 Agent Settings")
    max_retries = st.slider("Max Retries (per node)", 1, 5, 3)
    timeout = st.slider("Timeout (seconds)", 10, 60, 30)
    
    # Update configuration with combined heart attack prediction
    if st.button("🔄 Update Configuration"):
        try:
            # Handle uploaded combined file
            if uploaded_combined_model:
                # Save uploaded combined model temporarily
                combined_model_path = f"temp_heart_combined_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                with open(combined_model_path, "wb") as f:
                    f.write(uploaded_combined_model.read())
                heart_attack_combined_pkl_path = combined_model_path
                st.success(f"✅ Combined model uploaded and saved as: {combined_model_path}")
            
            config = Config(
                fastapi_url=fastapi_url,
                max_retries=max_retries,
                timeout=timeout,
                heart_attack_combined_pkl_path=heart_attack_combined_pkl_path,
                heart_attack_threshold=heart_attack_threshold
            )
            st.session_state.config = config
            st.session_state.agent = None  # Force reinitialization
            st.success("✅ Configuration updated including combined heart attack prediction!")
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
    
    # Current Configuration Status with Combined Heart Attack Prediction
    st.subheader("📋 Current Configuration Status")
    try:
        current_config = st.session_state.config or Config()
        
        st.success("✅ Configuration active")
        st.write(f"**FastAPI:** {current_config.fastapi_url}")
        st.write(f"**Max Retries:** {current_config.max_retries}")
        st.write(f"**Timeout:** {current_config.timeout}")
        
        # Show API settings
        st.markdown("**🔧 API Settings:**")
        st.write(f"**API URL:** {current_config.api_url[:30]}...")
        st.write(f"**Model:** {current_config.model}")
        st.write(f"**App ID:** {current_config.app_id}")
        
        # NEW: Show Combined Heart Attack Prediction settings
        st.markdown("**❤️ Combined Heart Attack Prediction:**")
        st.write(f"**Combined Model Path:** {current_config.heart_attack_combined_pkl_path}")
        st.write(f"**Risk Threshold:** {current_config.heart_attack_threshold}")
        st.write(f"**Expected Features:** Age, Gender, Diabetes, High_BP, Smoking")
        
        st.success("✅ API is configured and ready!")
        st.success("❤️ Combined Heart Attack Prediction is configured!")
        st.success("💬 Interactive chatbot is ready!")
        
        # Test API Connection
        if st.button("🧪 Test API Connection"):
            try:
                test_config = Config()
                test_agent = HealthAnalysisAgent(test_config)
                
                with st.spinner("Testing API connection..."):
                    test_result = test_agent.test_llm_connection()
                
                if test_result["success"]:
                    st.success("✅ API connection successful!")
                    st.info(f"📝 Response: {test_result['response']}")
                    st.info(f"🤖 Model: {test_result['model']}")
                else:
                    st.error("❌ API connection failed!")
                    st.error(f"💥 Error: {test_result['error']}")
            except Exception as e:
                st.error(f"❌ Test failed: {str(e)}")
        
    except Exception as e:
        st.error(f"❌ Configuration error: {e}")
        st.code(f"Error details: {str(e)}")

# Patient Input Form
st.markdown('<div class="step-header">👤 Patient Information Input</div>', unsafe_allow_html=True)
st.info("💡 Enter patient information below. This data will be processed through the Health Agent workflow with AI analysis and combined heart attack risk prediction using your ML model.")

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
        "🚀 Execute Health Agent Analysis with Combined Heart Attack Prediction", 
        use_container_width=True,
        disabled=st.session_state.analysis_running
    )

# Analysis Status Section
if st.session_state.analysis_running:
    st.markdown('<div class="info-box">🔄 Health Agent workflow with combined heart attack prediction executing... Please wait.</div>', unsafe_allow_html=True)

# Run Enhanced LangGraph Analysis with Combined Heart Attack Prediction
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
    
    # Display patient data being sent to Health Agent
    st.info(f"📤 Sending patient data to Health Agent with Combined Heart Attack Prediction: {patient_data['first_name']} {patient_data['last_name']} (Age: {calculated_age})")
    
    # Validate patient data
    is_valid, validation_errors = validate_patient_data(patient_data)
    
    if not is_valid:
        st.error("❌ Please fix the following errors:")
        for error in validation_errors:
            st.error(f"• {error}")
    else:
        # Initialize Health Agent with combined heart attack prediction
        if st.session_state.agent is None:
            try:
                config = st.session_state.config or Config()
                st.session_state.agent = HealthAnalysisAgent(config)
                st.success(f"✅ Health Agent initialized with combined heart attack prediction")
                st.info(f"🤖 Model: {config.model}")
                st.info(f"🔑 App ID: {config.app_id}")
                st.info(f"❤️ Combined Heart Attack Model: {config.heart_attack_combined_pkl_path}")
                st.info(f"💬 Chatbot: Interactive mode ready")
            except Exception as e:
                st.error(f"❌ Failed to initialize Health Agent: {str(e)}")
                st.stop()
        
        st.session_state.analysis_running = True
        
        # Progress tracking for 8-node workflow
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run Health Agent analysis
        with st.spinner("🚀 Executing Enhanced Health Agent workflow with Combined Heart Attack Prediction..."):
            try:
                # Update progress for 8 nodes
                status_text.text("🚀 Initializing Health Agent state machine...")
                progress_bar.progress(5)
                time.sleep(0.5)
                
                status_text.text("📊 Node 1: Data Retrieval...")
                progress_bar.progress(12)
                time.sleep(0.5)
                
                status_text.text("🔒 Node 2: Data Deidentification...")
                progress_bar.progress(24)
                time.sleep(0.5)
                
                status_text.text("🔍 Node 3: Medical/Pharmacy Extraction...")
                progress_bar.progress(36)
                time.sleep(0.5)
                
                status_text.text("🎯 Node 4: Enhanced Entity Extraction...")
                progress_bar.progress(48)
                time.sleep(0.5)
                
                status_text.text("📈 Node 5: Health Trajectory Analysis...")
                progress_bar.progress(60)
                time.sleep(0.5)
                
                status_text.text("📋 Node 6: Summary Generation...")
                progress_bar.progress(72)
                time.sleep(0.5)
                
                status_text.text("❤️ Node 8: Combined Heart Attack Risk Prediction...")  # NEW
                progress_bar.progress(84)
                time.sleep(0.5)
                
                status_text.text("💬 Node 9: Chatbot Initialization...")
                progress_bar.progress(96)
                
                # Execute the Enhanced Health Agent workflow
                results = st.session_state.agent.run_analysis(patient_data)
                
                # Update progress based on completion
                if results.get("success", False):
                    progress_bar.progress(100)
                    status_text.text("✅ Enhanced Health Agent workflow with Combined Heart Attack Prediction completed successfully!")
                    
                    st.session_state.analysis_results = results
                    st.session_state.chatbot_context = results.get("chatbot_context", {})
                    st.markdown('<div class="success-box">✅ Enhanced Health Agent analysis with Combined Heart Attack Prediction completed successfully!</div>', unsafe_allow_html=True)
                    
                    # Show heart attack prediction result prominently
                    heart_attack_prediction = results.get("heart_attack_prediction", {})
                    if heart_attack_prediction and not heart_attack_prediction.get("error"):
                        risk_level = heart_attack_prediction.get("risk_level", "UNKNOWN")
                        risk_score = heart_attack_prediction.get("risk_score", 0.0)
                        risk_icon = heart_attack_prediction.get("risk_icon", "❓")
                        risk_percentage = heart_attack_prediction.get("risk_percentage", "0.0%")
                        model_source = heart_attack_prediction.get("model_info", {}).get("model_source", "combined_pkl")
                        
                        # Display risk with appropriate styling
                        if risk_level == "HIGH":
                            st.markdown(f'<div class="risk-high">{risk_icon} <strong>Combined Heart Attack Risk Assessment: {risk_level}</strong><br>Risk Score: {risk_percentage}<br>Model Source: {model_source}<br>⚠️ Immediate medical consultation recommended</div>', unsafe_allow_html=True)
                        elif risk_level == "MODERATE":
                            st.markdown(f'<div class="risk-moderate">{risk_icon} <strong>Combined Heart Attack Risk Assessment: {risk_level}</strong><br>Risk Score: {risk_percentage}<br>Model Source: {model_source}<br>📋 Regular monitoring advised</div>', unsafe_allow_html=True)
                        elif risk_level == "LOW":
                            st.markdown(f'<div class="risk-low">{risk_icon} <strong>Combined Heart Attack Risk Assessment: {risk_level}</strong><br>Risk Score: {risk_percentage}<br>Model Source: {model_source}<br>✅ Continue healthy lifestyle practices</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="heart-box">{risk_icon} <strong>Combined Heart Attack Risk Assessment: {risk_level}</strong><br>Risk Score: {risk_percentage}<br>Model Source: {model_source}</div>', unsafe_allow_html=True)
                    elif heart_attack_prediction and heart_attack_prediction.get("error"):
                        st.warning(f"⚠️ Combined Heart Attack Prediction: {heart_attack_prediction.get('error', 'Unknown error')}")
                    
                    if results.get("chatbot_ready", False):
                        st.markdown('<div class="chatbot-box">💬 Interactive chatbot is ready with combined heart attack prediction context!</div>', unsafe_allow_html=True)
                else:
                    progress_bar.progress(70)
                    status_text.text("⚠️ Enhanced Health Agent workflow completed with errors")
                    
                    st.session_state.analysis_results = results
                    st.warning("⚠️ Analysis completed but with some errors. Check results below.")
                    
                    errors = results.get('errors', [])
                    if errors:
                        st.error("Errors encountered:")
                        for error in errors:
                            st.error(f"• {error}")
                
            except Exception as e:
                progress_bar.progress(0)
                status_text.text("❌ Enhanced Health Agent workflow failed")
                st.error(f"❌ Error in Health Agent execution: {str(e)}")
                
                st.session_state.analysis_results = {
                    "success": False,
                    "error": str(e),
                    "patient_data": patient_data,
                    "errors": [str(e)],
                    "processing_steps_completed": 0,
                    "langgraph_used": True,
                    "enhancement_version": "v4.0_with_combined_heart_attack_prediction"
                }
            finally:
                st.session_state.analysis_running = False

# Display Health Agent Results with Combined Heart Attack Prediction
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    # Health Agent Results Overview
    st.markdown('<div class="step-header">📊 Health Agent Analysis Results</div>', unsafe_allow_html=True)
    
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
        st.info(f"📋 Enhanced analysis with combined heart attack prediction completed for: {processed_patient.get('first_name', 'Unknown')} {processed_patient.get('last_name', 'Unknown')}{age_display}")

    # Show errors if any
    errors = safe_get(results, 'errors', [])
    if errors:
        st.markdown('<div class="error-box">❌ Enhanced LangGraph workflow errors:</div>', unsafe_allow_html=True)
        for error in errors:
            st.error(f"• {error}")

    # API Outputs Section (keeping same structure for brevity)
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

    # Deidentified Data Section (keeping same structure)
    deidentified_data = safe_get(results, 'deidentified_data', {})
    if deidentified_data:
        st.markdown('<div class="step-header">🔒 Node 2: Data Deidentification</div>', unsafe_allow_html=True)
        
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

    # Structured Extractions Section (keeping same structure for brevity)
    structured_extractions = safe_get(results, 'structured_extractions', {})
    if structured_extractions:
        st.markdown('<div class="step-header">🔍 Node 3: Medical/Pharmacy Data Extraction</div>', unsafe_allow_html=True)
        
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

    # Entity Extraction Section (keeping same structure for brevity)
    entity_extraction = safe_get(results, 'entity_extraction', {})
    if entity_extraction:
        st.markdown('<div class="step-header">🎯 Node 4: Enhanced Entity Extraction</div>', unsafe_allow_html=True)
        
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
        
        # Show medical conditions and medications if available
        medical_conditions = safe_get(entity_extraction, 'medical_conditions', [])
        if medical_conditions:
            st.markdown("**🏥 Medical Conditions Identified from ICD-10 Codes:**")
            for condition in medical_conditions:
                st.markdown(f"- {condition}")
        
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

    # Health Trajectory and Summary Sections (keeping same structure)
    health_trajectory = safe_get(results, 'health_trajectory', '')
    if health_trajectory:
        st.markdown('<div class="step-header">📈 Node 5: Health Trajectory Analysis</div>', unsafe_allow_html=True)
        
        st.markdown("**🤖 AI Analysis (with Structured Extractions):**")
        st.markdown(health_trajectory)
        
        st.download_button(
            "📄 Download Health Trajectory",
            health_trajectory,
            f"health_trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    else:
        st.warning("Health trajectory analysis not available")

    final_summary = safe_get(results, 'final_summary', '')
    if final_summary:
        st.markdown('<div class="step-header">📋 Node 6: Final Summary</div>', unsafe_allow_html=True)
        
        st.markdown("**🤖 AI Executive Summary (with Medical & Pharmacy Extractions):**")
        st.markdown(final_summary)
        
        st.download_button(
            "📄 Download Final Summary",
            final_summary,
            f"final_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    else:
        st.warning("Final summary not available")

    # NEW: Combined Heart Attack Prediction Section (Node 8)
    heart_attack_prediction = safe_get(results, 'heart_attack_prediction', {})
    heart_attack_risk_score = safe_get(results, 'heart_attack_risk_score', 0.0)
    heart_attack_features = safe_get(results, 'heart_attack_features', {})
    
    if heart_attack_prediction:
        st.markdown('<div class="heart-header">❤️ Node 8: Combined Heart Attack Risk Prediction</div>', unsafe_allow_html=True)
        
        # Show combined model badge
        st.markdown('<div class="combined-box">📦 <strong>Using Combined PKL Model</strong> - Model, Scaler, and Features in single file</div>', unsafe_allow_html=True)
        
        if not heart_attack_prediction.get('error'):
            # Risk Assessment Display
            risk_level = heart_attack_prediction.get("risk_level", "UNKNOWN")
            risk_score = heart_attack_prediction.get("risk_score", 0.0)
            risk_icon = heart_attack_prediction.get("risk_icon", "❓")
            risk_color = heart_attack_prediction.get("risk_color", "gray")
            risk_percentage = heart_attack_prediction.get("risk_percentage", "0.0%")
            model_info = heart_attack_prediction.get("model_info", {})
            
            # Main risk display with combined model info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="entity-card" style="border-color: {risk_color}; background: linear-gradient(135deg, #fff, #f8f9fa);">
                    <h3 style="color: {risk_color};">{risk_icon} Risk Level</h3>
                    <p class="metric-highlight" style="color: {risk_color}; font-size: 1.5rem;">{risk_level}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="entity-card" style="border-color: {risk_color}; background: linear-gradient(135deg, #fff, #f8f9fa);">
                    <h3 style="color: {risk_color};">📊 Risk Score</h3>
                    <p class="metric-highlight" style="color: {risk_color}; font-size: 1.5rem;">{risk_percentage}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="entity-card" style="border-color: {risk_color}; background: linear-gradient(135deg, #fff, #f8f9fa);">
                    <h3 style="color: {risk_color};">🤖 Model Type</h3>
                    <p class="metric-highlight" style="color: {risk_color}; font-size: 1.2rem;">{model_info.get('model_type', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="entity-card" style="border-color: {risk_color}; background: linear-gradient(135deg, #fff, #f8f9fa);">
                    <h3 style="color: {risk_color};">📦 Source</h3>
                    <p class="metric-highlight" style="color: {risk_color}; font-size: 1.2rem;">{model_info.get('model_source', 'combined_pkl').upper()}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Prediction Interpretation
            prediction_interpretation = heart_attack_prediction.get("prediction_interpretation", {})
            if prediction_interpretation:
                st.markdown("**🔍 Combined Model Risk Assessment Interpretation:**")
                st.info(f"📋 **Assessment:** {prediction_interpretation.get('risk_assessment', 'N/A')}")
                st.info(f"📊 **Confidence:** {prediction_interpretation.get('confidence', 'N/A')}")
                st.info(f"💡 **Recommendation:** {prediction_interpretation.get('recommendation', 'N/A')}")
                
                # Risk factors
                risk_factors = prediction_interpretation.get("risk_factors", [])
                if risk_factors:
                    st.markdown("**⚠️ Identified Risk Factors:**")
                    for factor in risk_factors:
                        st.markdown(f"- {factor}")
            
            # Feature Analysis for Combined Model
            if heart_attack_features and not heart_attack_features.get('error'):
                with st.expander("🔍 View Combined Model Features Analysis"):
                    extracted_features = heart_attack_features.get("extracted_features", {})
                    feature_interpretation = heart_attack_features.get("feature_interpretation", {})
                    model_info_features = heart_attack_features.get("model_info", {})
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📊 Extracted Features (Combined Model):**")
                        if extracted_features:
                            for feature, value in extracted_features.items():
                                st.markdown(f"- **{feature}:** {value}")
                    
                    with col2:
                        st.markdown("**🔍 Feature Interpretation:**")
                        if feature_interpretation:
                            for feature, interpretation in feature_interpretation.items():
                                st.markdown(f"- **{feature.replace('_', ' ').title()}:** {interpretation}")
                    
                    # Show model-specific info
                    if model_info_features:
                        st.markdown("**🤖 Combined Model Information:**")
                        st.markdown(f"- **Model Type:** {model_info_features.get('model_type', 'Unknown')}")
                        st.markdown(f"- **Expected Features:** {model_info_features.get('features_expected', [])}")
                        st.markdown(f"- **Features Count:** {model_info_features.get('features_count', 0)}")
            
            # Combined Model Information
            if model_info:
                with st.expander("🤖 Combined Model Information"):
                    st.json(model_info)
            
            # Download combined heart attack prediction
            st.download_button(
                "📄 Download Combined Heart Attack Prediction",
                safe_json_dumps(heart_attack_prediction),
                f"combined_heart_attack_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
        else:
            # Display error information
            error_msg = heart_attack_prediction.get('error', 'Unknown error')
            st.error(f"❌ Combined Heart Attack Prediction Error: {error_msg}")
            
            model_status = heart_attack_prediction.get('model_status', '')
            if model_status:
                st.warning(f"⚠️ Model Status: {model_status}")
            
            st.info("💡 Combined heart attack prediction requires a trained ML model (combined PKL file). Please check the configuration in the sidebar.")

    # Interactive Chatbot Section (updated to include combined heart attack prediction context)
    if results.get("chatbot_ready", False) and st.session_state.chatbot_context:
        st.markdown('<div class="chatbot-header">💬 Node 9: Interactive Chatbot with Combined Heart Attack Prediction</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="chatbot-box">🤖 Ask questions about the patient\'s medical data, analysis results, combined heart attack risk assessment, or request specific insights based on the deidentified records.</div>', unsafe_allow_html=True)
        
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
                "Entity Extraction": "✅ Available" if st.session_state.chatbot_context.get("entity_extraction") else "❌ None",
                "Combined Heart Attack Prediction": "✅ Available" if st.session_state.chatbot_context.get("heart_attack_prediction") else "❌ None"  # NEW
            }
            
            for key, value in context_summary.items():
                if "✅" in value:
                    st.success(f"{key}: {value}")
                else:
                    st.warning(f"{key}: {value}")
        
        # Chat input
        user_question = st.chat_input("💬 Ask a question about the medical data or combined heart attack risk...")
        
        if user_question:
            # Add user message to chat history
            st.session_state.chatbot_messages.append({"role": "user", "content": user_question})
            
            # Get response from chatbot
            try:
                with st.spinner("🤖 Analyzing medical data and combined heart attack prediction, generating response..."):
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
        
        # Enhanced Sample questions including combined heart attack prediction
        st.markdown("**💡 Sample Questions You Can Ask:**")
        sample_questions = [
            "What medications was this patient prescribed?",
            "Are there any chronic conditions indicated in the medical data?",
            "What is the patient's combined heart attack risk assessment?",  # NEW
            "What factors contribute to the combined heart attack risk score?",  # NEW
            "Explain the significance of the ICD-10 diagnosis codes found",
            "What drug interactions should be considered?",
            "How do the medications relate to the combined heart attack risk prediction?",  # NEW
            "Summarize the key health insights including combined heart attack risk"  # NEW
        ]
        
        for i, question in enumerate(sample_questions):
            if st.button(f"💬 {question}", key=f"sample_q_{i}"):
                # Use the sample question as if user typed it
                st.session_state.chatbot_messages.append({"role": "user", "content": question})
                
                try:
                    with st.spinner("🤖 Analyzing medical data and combined heart attack prediction, generating response..."):
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
        st.markdown('<div class="chatbot-header">💬 Node 9: Interactive Chatbot</div>', unsafe_allow_html=True)
        st.warning("⚠️ Chatbot initialization failed. Please check the workflow execution above.")

    # Complete Health Agent Analysis Report Download (updated with combined heart attack prediction)
    st.markdown('<div class="step-header">💾 Complete Health Agent Analysis Report</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Complete JSON report with combined heart attack prediction
        complete_report = {
            "health_agent_metadata": {
                "patient_info": processed_patient,
                "timestamp": datetime.now().isoformat(),
                "success": results.get("success", False),
                "nodes_completed": results.get("processing_steps_completed", 0),
                "workflow_engine": "Health Agent",
                "ai_engine": "AI Analysis",
                "ai_model": "llama3.1-70b",
                "chatbot_enabled": True,
                "chatbot_ready": results.get("chatbot_ready", False),
                "combined_heart_attack_prediction_enabled": True,  # NEW
                "heart_attack_risk_score": results.get("heart_attack_risk_score", 0.0),  # NEW
                "enhancement_version": results.get("enhancement_version", "v4.0"),
                "step_status": safe_get(results, "step_status", {}),
                "extraction_enabled": True,
                "interactive_features": ["Medical Data Q&A", "ICD-10 Analysis", "NDC Code Interpretation", "Combined Heart Attack Risk Assessment"]  # NEW
            },
            "api_outputs": safe_get(results, "api_outputs", {}),
            "deidentified_data": safe_get(results, "deidentified_data", {}),
            "structured_extractions": safe_get(results, "structured_extractions", {}),
            "entity_extraction": safe_get(results, "entity_extraction", {}),
            "health_trajectory": safe_get(results, "health_trajectory", ""),
            "final_summary": safe_get(results, "final_summary", ""),
            # NEW: Combined heart attack prediction data
            "combined_heart_attack_prediction": safe_get(results, "heart_attack_prediction", {}),
            "heart_attack_risk_score": safe_get(results, "heart_attack_risk_score", 0.0),
            "heart_attack_features": safe_get(results, "heart_attack_features", {}),
            "chatbot_context": safe_get(results, "chatbot_context", {}),
            "chat_history": st.session_state.chatbot_messages,
            "errors": safe_get(results, "errors", [])
        }
        
        patient_last_name = processed_patient.get('last_name', 'unknown')
        st.download_button(
            "📊 Download Complete Health Agent Report",
            safe_json_dumps(complete_report),
            f"health_agent_analysis_combined_heart_prediction_{patient_last_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Text report with combined heart attack prediction info
        patient_name = f"{processed_patient.get('first_name', 'Unknown')} {processed_patient.get('last_name', 'Unknown')}"
        
        # Get extraction counts
        medical_extraction = safe_get(structured_extractions, 'medical', {})
        pharmacy_extraction = safe_get(structured_extractions, 'pharmacy', {})
        medical_records = len(safe_get(medical_extraction, 'hlth_srvc_records', []))
        pharmacy_records = len(safe_get(pharmacy_extraction, 'ndc_records', []))
        
        # Combined heart attack prediction info
        heart_risk_score = results.get("heart_attack_risk_score", 0.0)
        heart_risk_level = safe_get(heart_attack_prediction, "risk_level", "Unknown")
        model_source = safe_get(heart_attack_prediction, "model_info", {}).get("model_source", "combined_pkl")
        
        text_report = f"""
HEALTH AGENT ANALYSIS REPORT WITH COMBINED HEART ATTACK PREDICTION
===============================================================
Patient: {patient_name}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: {'Success' if results.get('success', False) else 'Failed'}
Nodes Completed: {results.get('processing_steps_completed', 0)}/8
Workflow Engine: Health Agent
AI Engine: AI Analysis
AI Model: llama3.1-70b
Interactive Chatbot: {'Ready' if results.get('chatbot_ready', False) else 'Failed'}
Combined Heart Attack Prediction: {'Enabled' if heart_attack_prediction else 'Disabled'}
Enhancement Version: {results.get('enhancement_version', 'v4.0')}

COMBINED HEART ATTACK RISK ASSESSMENT:
====================================
Risk Score: {heart_risk_score:.3f} ({heart_risk_score*100:.1f}%)
Risk Level: {heart_risk_level}
Model Source: {model_source}
Model Used: {safe_get(heart_attack_prediction, 'model_info', {}).get('model_type', 'Unknown')}
Features Used: {safe_get(heart_attack_prediction, 'model_info', {}).get('features_used', 0)}

STRUCTURED EXTRACTIONS SUMMARY:
===============================
Medical Records Extracted: {medical_records}
Pharmacy Records Extracted: {pharmacy_records}
Total Diagnosis Codes: {safe_get(medical_extraction, 'extraction_summary', {}).get('total_diagnosis_codes', 0)}
Unique NDC Codes: {len(safe_get(pharmacy_extraction, 'extraction_summary', {}).get('unique_ndc_codes', []))}

ENHANCED ENTITY EXTRACTION RESULTS:
===================================
{safe_json_dumps(entity_extraction)}

HEALTH TRAJECTORY ANALYSIS:
===========================
{health_trajectory}

FINAL SUMMARY:
==============
{final_summary}

COMBINED HEART ATTACK PREDICTION DETAILS:
========================================
{safe_json_dumps(heart_attack_prediction)}

INTERACTIVE CHATBOT SESSION:
============================
Chat Messages Exchanged: {len(st.session_state.chatbot_messages)}
Chatbot Context Components: {len(safe_get(results, 'chatbot_context', {}))}

CHAT HISTORY:
=============
{chr(10).join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.chatbot_messages])}

HEALTH AGENT ERRORS (if any):
=============================
{chr(10).join(safe_get(results, 'errors', []))}
        """
        
        st.download_button(
            "📝 Download Health Agent Text Report",
            text_report,
            f"health_agent_analysis_combined_heart_prediction_{patient_last_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        # CSV summary with combined heart attack prediction metrics
        try:
            # Get extraction counts
            medical_extraction = safe_get(structured_extractions, 'medical', {})
            pharmacy_extraction = safe_get(structured_extractions, 'pharmacy', {})
            medical_records = len(safe_get(medical_extraction, 'hlth_srvc_records', []))
            pharmacy_records = len(safe_get(pharmacy_extraction, 'ndc_records', []))
            
            # Combined heart attack prediction metrics
            heart_risk_score = results.get("heart_attack_risk_score", 0.0)
            heart_risk_level = safe_get(heart_attack_prediction, "risk_level", "Unknown")
            model_source = safe_get(heart_attack_prediction, "model_info", {}).get("model_source", "combined_pkl")
            
            csv_data = {
                "Metric": [
                    "Analysis Status", "Workflow Engine", "AI Engine", "AI Model", "Enhancement Version", "Nodes Completed", 
                    "Medical Records Extracted", "Pharmacy Records Extracted", "Diagnosis Codes Found",
                    "Unique NDC Codes", "Diabetes", "Age Group", "Smoking", "Alcohol", "Blood Pressure", 
                    "Combined Heart Attack Risk Score", "Combined Heart Attack Risk Level", "Model Source", "Chatbot Ready", "Chat Messages", "Timestamp"  # NEW
                ],
                "Value": [
                    safe_str("Success" if results.get("success", False) else "Failed"),
                    safe_str("Health Agent"),
                    safe_str("AI Analysis"),
                    safe_str("llama3.1-70b"),
                    safe_str(results.get('enhancement_version', 'v4.0')),
                    safe_str(f"{results.get('processing_steps_completed', 0)}/8"),
                    safe_str(medical_records),
                    safe_str(pharmacy_records),
                    safe_str(safe_get(medical_extraction, 'extraction_summary', {}).get('total_diagnosis_codes', 0)),
                    safe_str(len(safe_get(pharmacy_extraction, 'extraction_summary', {}).get('unique_ndc_codes', []))),
                    safe_str(safe_get(entity_extraction, 'diabetics', 'unknown')),
                    safe_str(safe_get(entity_extraction, 'age_group', 'unknown')),
                    safe_str(safe_get(entity_extraction, 'smoking', 'unknown')),
                    safe_str(safe_get(entity_extraction, 'alcohol', 'unknown')),
                    safe_str(safe_get(entity_extraction, 'blood_pressure', 'unknown')),
                    safe_str(f"{heart_risk_score:.3f}"),  # NEW
                    safe_str(heart_risk_level),  # NEW
                    safe_str(model_source),  # NEW
                    safe_str("Ready" if results.get("chatbot_ready", False) else "Failed"),
                    safe_str(len(st.session_state.chatbot_messages)),
                    safe_str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                ]
            }
            
            csv_df = pd.DataFrame(csv_data)
            csv_string = csv_df.to_csv(index=False)
            
            st.download_button(
                "📊 Download Health Agent CSV",
                csv_string,
                f"health_agent_summary_combined_heart_prediction_{patient_last_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error generating CSV: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🏥 <strong>Health Agent Dashboard with Combined Heart Attack Prediction</strong><br>
    Powered by Advanced AI with Medical/Pharmacy Extraction, Combined ML Heart Attack Risk Assessment, and Interactive Medical Data Chatbot<br>
    ✅ <strong>Features:</strong> Medical ICD-10 Code Extraction | Pharmacy NDC Data Extraction | AI Analysis | Combined Heart Attack Risk Prediction | Interactive Medical Data Q&A Chatbot<br>
    📦 <strong>Combined PKL:</strong> Single file containing Model + Scaler + Features (Age, Gender, Diabetes, High_BP, Smoking)<br>
    ⚠️ <em>This analysis is for informational purposes only and should not replace professional medical advice.</em>
</div>
""", unsafe_allow_html=True)

# Enhanced Debug information in sidebar
if st.sidebar.checkbox("🐛 Show Debug Info"):
    st.sidebar.markdown("### 🔧 Health Agent Debug")
    st.sidebar.write("Agent Available:", AGENT_AVAILABLE)
    st.sidebar.write("Agent Initialized:", st.session_state.agent is not None)
    st.sidebar.write("Current Date:", today_date)
    
    # Configuration Debug
    st.sidebar.markdown("**Configuration Debug:**")
    if st.session_state.config:
        st.sidebar.write("Custom Config:", "✅ Yes")
        st.sidebar.write("FastAPI URL:", st.session_state.config.fastapi_url)
        st.sidebar.write("Max Retries:", st.session_state.config.max_retries)
        st.sidebar.write("Timeout:", st.session_state.config.timeout)
        # NEW: Combined heart attack prediction debug
        st.sidebar.write("Combined Heart Attack Model:", st.session_state.config.heart_attack_combined_pkl_path)
        st.sidebar.write("Heart Attack Threshold:", st.session_state.config.heart_attack_threshold)
        
        # Try to show model debug info
        try:
            if hasattr(st.session_state.agent, 'heart_attack_model') and st.session_state.agent.heart_attack_model:
                st.sidebar.write("Model Loaded:", "✅ Yes")
                st.sidebar.write("Model Type:", type(st.session_state.agent.heart_attack_model).__name__)
                if hasattr(st.session_state.agent, 'heart_attack_scaler'):
                    st.sidebar.write("Scaler Available:", "✅ Yes" if st.session_state.agent.heart_attack_scaler else "❌ No")
                if hasattr(st.session_state.agent, 'heart_attack_features'):
                    st.sidebar.write("Features:", st.session_state.agent.heart_attack_features)
            else:
                st.sidebar.write("Model Loaded:", "❌ No")
        except Exception as e:
            st.sidebar.write("Model Debug Error:", str(e))
    else:
        st.sidebar.write("Custom Config:", "❌ Using defaults")
    
    # Session State Debug
    st.sidebar.markdown("**Session State Debug:**")
    st.sidebar.write("Analysis Results:", "✅ Available" if st.session_state.analysis_results else "❌ None")
    st.sidebar.write("Analysis Running:", st.session_state.analysis_running)
    st.sidebar.write("Chatbot Messages:", len(st.session_state.chatbot_messages))
    st.sidebar.write("Chatbot Context:", "✅ Available" if st.session_state.chatbot_context else "❌ None")
    
    # Results Debug
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        st.sidebar.markdown("**Results Debug:**")
        st.sidebar.write("Success:", results.get("success", False))
        st.sidebar.write("Steps Completed:", results.get("processing_steps_completed", 0))
        st.sidebar.write("Errors Count:", len(results.get("errors", [])))
        st.sidebar.write("Heart Attack Prediction:", "✅ Available" if results.get("heart_attack_prediction") else "❌ None")
        st.sidebar.write("Risk Score:", results.get("heart_attack_risk_score", 0.0))
        st.sidebar.write("Chatbot Ready:", results.get("chatbot_ready", False))

# Instructions for Users
if st.sidebar.checkbox("📚 Show Usage Instructions"):
    st.sidebar.markdown("### 📚 Usage Instructions")
    st.sidebar.markdown("""
    **🏥 Health Agent Workflow:**
    1. Enter patient information in the form
    2. Click "Execute Health Agent Analysis"
    3. Wait for 8-node workflow to complete
    4. Review results in each section
    5. Use chatbot for interactive Q&A
    
    **❤️ Combined Heart Attack Prediction:**
    - Uses single PKL file with model + scaler + features
    - Expected features: Age, Gender, Diabetes, High_BP, Smoking
    - Predicts risk score and classification
    - Provides detailed interpretation
    
    **💬 Interactive Chatbot:**
    - Ask questions about medical data
    - Get insights on heart attack risk
    - Request specific analysis
    - Review medication interactions
    
    **📊 Download Options:**
    - Complete JSON report
    - Text summary report
    - CSV metrics summary
    - Individual component data
    """)

# Model Information Panel
if st.sidebar.checkbox("🤖 Show Model Information"):
    st.sidebar.markdown("### 🤖 Model Information")
    
    try:
        if st.session_state.agent and hasattr(st.session_state.agent, 'heart_attack_model'):
            if st.session_state.agent.heart_attack_model:
                model = st.session_state.agent.heart_attack_model
                st.sidebar.success("✅ Combined Model Loaded")
                st.sidebar.write(f"**Type:** {type(model).__name__}")
                
                # Try to get model parameters
                if hasattr(model, 'n_estimators'):
                    st.sidebar.write(f"**N Estimators:** {model.n_estimators}")
                if hasattr(model, 'learning_rate'):
                    st.sidebar.write(f"**Learning Rate:** {model.learning_rate}")
                if hasattr(model, 'algorithm'):
                    st.sidebar.write(f"**Algorithm:** {model.algorithm}")
                
                # Show scaler info
                if st.session_state.agent.heart_attack_scaler:
                    scaler = st.session_state.agent.heart_attack_scaler
                    st.sidebar.write(f"**Scaler:** {type(scaler).__name__}")
                    if hasattr(scaler, 'mean_'):
                        st.sidebar.write(f"**Features Scaled:** {len(scaler.mean_)}")
                
                # Show features
                if st.session_state.agent.heart_attack_features:
                    st.sidebar.write(f"**Features:** {len(st.session_state.agent.heart_attack_features)}")
                    with st.sidebar.expander("View Features"):
                        for i, feature in enumerate(st.session_state.agent.heart_attack_features):
                            st.sidebar.write(f"{i+1}. {feature}")
            else:
                st.sidebar.error("❌ No Model Loaded")
        else:
            st.sidebar.warning("⚠️ Agent not initialized")
    except Exception as e:
        st.sidebar.error(f"❌ Model info error: {str(e)}")

# Performance Metrics
if st.sidebar.checkbox("📈 Show Performance Metrics"):
    st.sidebar.markdown("### 📈 Performance Metrics")
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Analysis metrics
        st.sidebar.metric("Nodes Completed", f"{results.get('processing_steps_completed', 0)}/8")
        st.sidebar.metric("Errors", len(results.get('errors', [])))
        
        # Data extraction metrics
        medical_extraction = safe_get(results.get('structured_extractions', {}), 'medical', {})
        pharmacy_extraction = safe_get(results.get('structured_extractions', {}), 'pharmacy', {})
        
        st.sidebar.metric("Medical Records", len(safe_get(medical_extraction, 'hlth_srvc_records', [])))
        st.sidebar.metric("Pharmacy Records", len(safe_get(pharmacy_extraction, 'ndc_records', [])))
        st.sidebar.metric("Diagnosis Codes", safe_get(medical_extraction, 'extraction_summary', {}).get('total_diagnosis_codes', 0))
        
        # Heart attack prediction metrics
        if results.get('heart_attack_prediction'):
            risk_score = results.get('heart_attack_risk_score', 0.0)
            st.sidebar.metric("Heart Attack Risk Score", f"{risk_score:.3f}")
            
        # Chatbot metrics
        st.sidebar.metric("Chat Messages", len(st.session_state.chatbot_messages))
    else:
        st.sidebar.info("No analysis results available yet")
