# Configure Streamlit page FIRST - before any other Streamlit commands
import streamlit as st
st.set_page_config(
    page_title="Health Agent",
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
import asyncio

# Add current directory to path for importing the agent
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the Enhanced LangGraph health analysis agent
AGENT_AVAILABLE = False
import_error = None
HealthAnalysisAgent = None
Config = None

try:
    from fixed_health_agent import HealthAnalysisAgent, Config  # Updated import
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    import_error = str(e)

# Custom CSS for Health Agent themed styling
st.markdown("""
<style>
.main-header {
    font-size: 2.8rem;
    color: #ff6b35;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}

.health-badge {
    background: linear-gradient(45deg, #ff6b35, #f7931e);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 1rem;
    font-weight: bold;
    display: inline-block;
    margin: 0.5rem 0;
}

.analysis-badge {
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

.fastapi-badge {
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

.fastapi-header {
    font-size: 1.6rem;
    color: #28a745;
    border-left: 4px solid #28a745;
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

.fastapi-box {
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
st.markdown('<h1 class="main-header">🏥 Health Agent</h1>', unsafe_allow_html=True)
st.markdown("**Advanced health analysis with comprehensive medical data extraction, interactive chatbot, and heart attack risk prediction**")

# Display import status AFTER page config
if AGENT_AVAILABLE:
    st.success("✅ Health Agent imported successfully!")
else:
    st.error(f"❌ Failed to import Health Agent: {import_error}")
    
    with st.expander("🔧 Health Agent Installation Guide"):
        st.markdown("""
        **Install Health Agent Requirements:**
        ```bash
        pip install langgraph langchain-core streamlit requests urllib3 pandas numpy aiohttp
        ```
        
        **Required Files:**
        - `fixed_health_agent.py` (the Fixed Health Agent)
        - `streamlit_langgraph_ui.py` (this file)
        - FastAPI server for heart attack prediction
        
        **FastAPI Server Setup:**
        1. Start the FastAPI server: `python heart_attack_fastapi_server.py`
        2. Server runs on http://localhost:8002
        3. Features: Age, Gender, Diabetes, High_BP, Smoking
        4. Endpoints: /health and /predict
        
        **Health Agent Features:**
        - ✅ State management and persistence
        - ✅ Conditional workflow routing  
        - ✅ Automatic retry mechanisms
        - ✅ Error handling and recovery
        - ✅ Checkpointing for reliability
        - ✅ **Medical field extraction (hlth_srvc_cd, diag_1_50_cd)**
        - ✅ **Pharmacy field extraction (Ndc, lbl_nm)**
        - ✅ **Enhanced entity detection with ICD-10 codes**
        - ✅ **Heart Attack Risk Prediction using FastAPI ML model**
        - ✅ **Interactive chatbot with medical data context**
        """)
    st.stop()

# Enhanced Sidebar Configuration
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
    st.markdown("❤️ **Heart Attack Prediction:** Active")
    st.markdown("💬 **Interactive Chatbot:** Ready")
    
    st.markdown("---")
    
    # API Configuration
    st.subheader("🔌 Server Settings")
    server_url = st.text_input("Backend Server URL", value="http://localhost:8001")
    
    # Heart Attack Prediction Configuration
    st.subheader("❤️ Heart Attack Prediction Settings")
    st.markdown('<div class="fastapi-badge">🔗 FastAPI Server Integration</div>', unsafe_allow_html=True)
    
    fastapi_server_url = st.text_input(
        "FastAPI Server URL *", 
        value="http://localhost:8002",
        help="URL of the FastAPI server running the heart attack prediction model"
    )
    
    heart_attack_threshold = st.slider(
        "Risk Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Threshold for high/low risk classification"
    )
    
    # FastAPI Server Connection Test
    st.markdown("**🧪 FastAPI Server Connection:**")
    if st.button("🔍 Test FastAPI Connection", key="test_fastapi"):
        try:
            # Test FastAPI connection with a simple request
            import requests
            health_url = f"{fastapi_server_url}/health"
            
            with st.spinner("Testing FastAPI server connection..."):
                # Test health endpoint
                health_response = requests.get(health_url, timeout=10)
                
                if health_response.status_code == 200:
                    st.success("✅ FastAPI Health endpoint successful!")
                    health_data = health_response.json()
                    st.info(f"📊 Health response: {health_data}")
                    
                    # Test prediction endpoint
                    predict_url = f"{fastapi_server_url}/predict"
                    test_params = {
                        "age": 50,
                        "gender": 1,
                        "diabetes": 0,
                        "high_bp": 0,
                        "smoking": 0
                    }
                    
                    predict_response = requests.post(predict_url, params=test_params, timeout=10)
                    
                    if predict_response.status_code == 200:
                        st.success("✅ FastAPI Prediction endpoint successful!")
                        prediction_data = predict_response.json()
                        st.info(f"📊 Test prediction: {prediction_data}")
                    else:
                        st.error(f"❌ Prediction endpoint error: {predict_response.status_code}")
                        
                else:
                    st.error(f"❌ FastAPI Health endpoint error: {health_response.status_code}")
                    
        except Exception as e:
            st.error(f"❌ FastAPI connection failed: {str(e)}")
    
    # FastAPI Server Information
    with st.expander("🔧 FastAPI Server Information"):
        st.markdown("""
        **Heart Attack Prediction Model:**
        - **Model Type:** Machine Learning Model (AdaBoost, Random Forest, etc.)
        - **Features:** Age, Gender, Diabetes, High_BP, Smoking (5 features)
        - **Input Format:** Query parameters (integers)
        - **Output:** Risk probability and prediction
        
        **To start FastAPI server:**
        ```bash
        python heart_attack_fastapi_server.py
        ```
        
        **Server Requirements:**
        - Trained ML model (pickle file)
        - FastAPI and uvicorn libraries
        - Server runs on port 8002 by default
        
        **Endpoints:**
        - GET /health - Health check
        - POST /predict?age=X&gender=X&diabetes=X&high_bp=X&smoking=X - Prediction
        """)
        
        # Show current FastAPI URL
        st.code(f"Current FastAPI URL: {fastapi_server_url}")
    
    # API Configuration - Showing configured values
    st.subheader("🔧 Analysis Settings")
    st.info("💡 **Analysis settings are pre-configured.** All settings are optimized for health analysis.")
    
    # Show current configuration (read-only)
    try:
        current_config = Config()
        st.text_input("Analysis API URL", value=current_config.api_url[:50] + "...", disabled=True)
        st.text_input("AI Model", value=current_config.model, disabled=True)
        st.text_input("App ID", value=current_config.app_id, disabled=True)
        st.text_input("Application Code", value=current_config.aplctn_cd, disabled=True)
        st.text_area("Analysis System Message", value=current_config.sys_msg, disabled=True, height=80)
        st.text_area("Chatbot System Message", value=current_config.chatbot_sys_msg, disabled=True, height=80)
        
        st.markdown("**🔧 Server URL and heart attack prediction settings can be modified. Analysis settings are pre-configured.**")
    except Exception as e:
        st.error(f"❌ Error loading configuration: {e}")

    # Settings
    st.subheader("🔄 Agent Settings")
    max_retries = st.slider("Max Retries (per node)", 1, 5, 3)
    timeout = st.slider("Timeout (seconds)", 10, 60, 30)
    
    # Update configuration
    if st.button("🔄 Update Configuration"):
        try:
            config = Config(
                fastapi_url=server_url,
                max_retries=max_retries,
                timeout=timeout,
                heart_attack_api_url=fastapi_server_url,
                heart_attack_threshold=heart_attack_threshold
            )
            st.session_state.config = config
            st.session_state.agent = None  # Force reinitialization
            st.success("✅ Configuration updated!")
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
        st.write(f"**Backend Server:** {current_config.fastapi_url}")
        st.write(f"**Max Retries:** {current_config.max_retries}")
        st.write(f"**Timeout:** {current_config.timeout}")
        
        # Show analysis settings
        st.markdown("**🔧 Analysis Settings:**")
        st.write(f"**API URL:** {current_config.api_url[:30]}...")
        st.write(f"**AI Model:** {current_config.model}")
        st.write(f"**App ID:** {current_config.app_id}")
        
        # Show heart attack prediction settings
        st.markdown("**❤️ Heart Attack Prediction:**")
        st.write(f"**FastAPI Server URL:** {current_config.heart_attack_api_url}")
        st.write(f"**Risk Threshold:** {current_config.heart_attack_threshold}")
        st.write(f"**Expected Features:** Age, Gender, Diabetes, High_BP, Smoking")
        
        st.success("✅ Analysis engine is configured and ready!")
        st.success("❤️ Heart attack prediction is configured!")
        st.success("💬 Interactive chatbot is ready!")
        
        # Test API Connection
        if st.button("🧪 Test Analysis Connection"):
            try:
                test_config = Config()
                test_agent = HealthAnalysisAgent(test_config)
                
                with st.spinner("Testing analysis connection..."):
                    test_result = test_agent.test_llm_connection()
                
                if test_result["success"]:
                    st.success("✅ Analysis connection successful!")
                    st.info(f"📝 Response: {test_result['response']}")
                    st.info(f"🤖 Model: {test_result['model']}")
                else:
                    st.error("❌ Analysis connection failed!")
                    st.error(f"💥 Error: {test_result['error']}")
            except Exception as e:
                st.error(f"❌ Test failed: {str(e)}")
        
        # Test FastAPI Connection from agent
        if st.button("🧪 Test FastAPI from Agent"):
            try:
                if st.session_state.agent is None:
                    test_config = Config(heart_attack_api_url=fastapi_server_url)
                    test_agent = HealthAnalysisAgent(test_config)
                else:
                    test_agent = st.session_state.agent
                
                with st.spinner("Testing FastAPI connection from agent..."):
                    # Run async test in sync context
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    fastapi_result = loop.run_until_complete(test_agent.test_fastapi_connection())
                    loop.close()
                
                if fastapi_result["success"]:
                    st.success("✅ FastAPI connection from agent successful!")
                    st.info(f"📝 Health Check: {fastapi_result['health_check']}")
                    st.info(f"📝 Prediction Test: {fastapi_result['prediction_test']}")
                    st.info(f"🔗 FastAPI Server: {fastapi_result['server_url']}")
                else:
                    st.error("❌ FastAPI connection from agent failed!")
                    st.error(f"💥 Error: {fastapi_result['error']}")
            except Exception as e:
                st.error(f"❌ FastAPI test failed: {str(e)}")
        
    except Exception as e:
        st.error(f"❌ Configuration error: {e}")
        st.code(f"Error details: {str(e)}")

# Patient Input Form
st.markdown('<div class="step-header">👤 Patient Information Input</div>', unsafe_allow_html=True)
st.info("💡 Enter patient information below. This data will be processed through the Health Agent workflow with AI analysis and heart attack risk prediction using FastAPI ML model.")

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
        "🚀 Execute Health Agent Analysis", 
        use_container_width=True,
        disabled=st.session_state.analysis_running
    )

# Analysis Status Section
if st.session_state.analysis_running:
    st.markdown('<div class="info-box">🔄 Health Agent workflow executing... Please wait.</div>', unsafe_allow_html=True)

# Run Health Agent Analysis
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
    st.info(f"📤 Sending patient data to Health Agent: {patient_data['first_name']} {patient_data['last_name']} (Age: {calculated_age})")
    
    # Validate patient data
    is_valid, validation_errors = validate_patient_data(patient_data)
    
    if not is_valid:
        st.error("❌ Please fix the following errors:")
        for error in validation_errors:
            st.error(f"• {error}")
    else:
        # Initialize Health Agent
        if st.session_state.agent is None:
            try:
                config = st.session_state.config or Config()
                st.session_state.agent = HealthAnalysisAgent(config)
                st.success(f"✅ Health Agent initialized")
                st.info(f"🤖 AI Model: {config.model}")
                st.info(f"🔑 App ID: {config.app_id}")
                st.info(f"❤️ FastAPI Server: {config.heart_attack_api_url}")
                st.info(f"💬 Chatbot: Interactive mode ready")
            except Exception as e:
                st.error(f"❌ Failed to initialize Health Agent: {str(e)}")
                st.stop()
        
        st.session_state.analysis_running = True
        
        # Progress tracking for 8-node workflow
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run Health Agent analysis
        with st.spinner("🚀 Executing Health Agent workflow..."):
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
                
                status_text.text("❤️ Node 8: Heart Attack Risk Prediction (FastAPI)...")
                progress_bar.progress(84)
                time.sleep(0.5)
                
                status_text.text("💬 Node 9: Chatbot Initialization...")
                progress_bar.progress(96)
                
                # Execute the Health Agent workflow
                results = st.session_state.agent.run_analysis(patient_data)
                
                # Update progress based on completion
                if results.get("success", False):
                    progress_bar.progress(100)
                    status_text.text("✅ Health Agent workflow completed successfully!")
                    
                    st.session_state.analysis_results = results
                    st.session_state.chatbot_context = results.get("chatbot_context", {})
                    st.markdown('<div class="success-box">✅ Health Agent analysis completed successfully!</div>', unsafe_allow_html=True)
                    
                    # Show heart attack prediction result prominently
                    heart_attack_prediction = results.get("heart_attack_prediction", {})
                    if heart_attack_prediction and not heart_attack_prediction.get("error"):
                        risk_level = heart_attack_prediction.get("risk_level", "UNKNOWN")
                        risk_score = heart_attack_prediction.get("risk_score", 0.0)
                        risk_icon = heart_attack_prediction.get("risk_icon", "❓")
                        risk_percentage = heart_attack_prediction.get("risk_percentage", "0.0%")
                        model_source = heart_attack_prediction.get("model_info", {}).get("model_source", "fastapi_server")
                        fastapi_url = heart_attack_prediction.get("model_info", {}).get("fastapi_server_url", "unknown")
                        
                        # Display risk with appropriate styling
                        if risk_level == "HIGH":
                            st.markdown(f'<div class="risk-high">{risk_icon} <strong>Heart Attack Risk Assessment: {risk_level}</strong><br>Risk Score: {risk_percentage}<br>Model Source: {model_source}<br>FastAPI Server: {fastapi_url}<br>⚠️ Immediate medical consultation recommended</div>', unsafe_allow_html=True)
                        elif risk_level == "MODERATE":
                            st.markdown(f'<div class="risk-moderate">{risk_icon} <strong>Heart Attack Risk Assessment: {risk_level}</strong><br>Risk Score: {risk_percentage}<br>Model Source: {model_source}<br>FastAPI Server: {fastapi_url}<br>📋 Regular monitoring advised</div>', unsafe_allow_html=True)
                        elif risk_level == "LOW":
                            st.markdown(f'<div class="risk-low">{risk_icon} <strong>Heart Attack Risk Assessment: {risk_level}</strong><br>Risk Score: {risk_percentage}<br>Model Source: {model_source}<br>FastAPI Server: {fastapi_url}<br>✅ Continue healthy lifestyle practices</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="fastapi-box">{risk_icon} <strong>Heart Attack Risk Assessment: {risk_level}</strong><br>Risk Score: {risk_percentage}<br>Model Source: {model_source}<br>FastAPI Server: {fastapi_url}</div>', unsafe_allow_html=True)
                    elif heart_attack_prediction and heart_attack_prediction.get("error"):
                        st.warning(f"⚠️ Heart Attack Prediction: {heart_attack_prediction.get('error', 'Unknown error')}")
                    
                    if results.get("chatbot_ready", False):
                        st.markdown('<div class="chatbot-box">💬 Interactive chatbot is ready with heart attack prediction context!</div>', unsafe_allow_html=True)
                else:
                    progress_bar.progress(70)
                    status_text.text("⚠️ Health Agent workflow completed with errors")
                    
                    st.session_state.analysis_results = results
                    st.warning("⚠️ Analysis completed but with some errors. Check results below.")
                    
                    errors = results.get('errors', [])
                    if errors:
                        st.error("Errors encountered:")
                        for error in errors:
                            st.error(f"• {error}")
                
            except Exception as e:
                progress_bar.progress(0)
                status_text.text("❌ Health Agent workflow failed")
                st.error(f"❌ Error in Health Agent execution: {str(e)}")
                
                st.session_state.analysis_results = {
                    "success": False,
                    "error": str(e),
                    "patient_data": patient_data,
                    "errors": [str(e)],
                    "processing_steps_completed": 0,
                    "langgraph_used": True,
                    "enhancement_version": "v4.0_with_fastapi_heart_attack_prediction"
                }
            finally:
                st.session_state.analysis_running = False

# Display Health Agent Results
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
        st.info(f"📋 Enhanced analysis completed for: {processed_patient.get('first_name', 'Unknown')} {processed_patient.get('last_name', 'Unknown')}{age_display}")

    # Show errors if any
    errors = safe_get(results, 'errors', [])
    if errors:
        st.markdown('<div class="error-box">❌ Health Agent workflow errors:</div>', unsafe_allow_html=True)
        for error in errors:
            st.error(f"• {error}")

    # API Outputs Section
    api_outputs = safe_get(results, 'api_outputs', {})
    if api_outputs:
        st.markdown('<div class="step-header">📊 Node 1: API Data Retrieval</div>', unsafe_allow_html=True)
        
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

    # Deidentified Data Section
    deidentified_data = safe_get(results, 'deidentified_data', {})
    if deidentified_data:
        st.markdown('<div class="step-header">🔒 Node 2: Data Deidentification</div>', unsafe_allow_html=True)
        
        deident_tabs = st.tabs(["🏥 Medical Deidentified", "💊 Pharmacy Deidentified"])
        
        with deident_tabs[0]:
            st.subheader("Deidentified Medical Data")
            deident_medical = safe_get(deidentified_data, 'medical', {})
            
            if deident_medical and not deident_medical.get('error'):
                st.markdown("**🎯 Health Agent Standardized Format:**")
                
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

    # Structured Extractions Section
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

    # Entity Extraction Section
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

    # Health Trajectory and Summary Sections
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

    # Heart Attack Prediction Section (Node 8)
    heart_attack_prediction = safe_get(results, 'heart_attack_prediction', {})
    heart_attack_risk_score = safe_get(results, 'heart_attack_risk_score', 0.0)
    heart_attack_features = safe_get(results, 'heart_attack_features', {})
    
    if heart_attack_prediction:
        st.markdown('<div class="fastapi-header">❤️ Node 8: Heart Attack Risk Prediction</div>', unsafe_allow_html=True)
        
        # Show FastAPI model badge
        st.markdown('<div class="fastapi-box">🔗 <strong>Using FastAPI Server</strong> - ML model via FastAPI HTTP API</div>', unsafe_allow_html=True)
        
        if not heart_attack_prediction.get('error'):
            # Risk Assessment Display
            risk_level = heart_attack_prediction.get("risk_level", "UNKNOWN")
            risk_score = heart_attack_prediction.get("risk_score", 0.0)
            risk_icon = heart_attack_prediction.get("risk_icon", "❓")
            risk_color = heart_attack_prediction.get("risk_color", "gray")
            risk_percentage = heart_attack_prediction.get("risk_percentage", "0.0%")
            model_info = heart_attack_prediction.get("model_info", {})
            
            # Main risk display with FastAPI model info
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
                    <h3 style="color: {risk_color};">🤖 Model Source</h3>
                    <p class="metric-highlight" style="color: {risk_color}; font-size: 1.2rem;">{model_info.get('model_source', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="entity-card" style="border-color: {risk_color}; background: linear-gradient(135deg, #fff, #f8f9fa);">
                    <h3 style="color: {risk_color};">🔗 FastAPI</h3>
                    <p class="metric-highlight" style="color: {risk_color}; font-size: 1.2rem;">ENABLED</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Prediction Interpretation
            prediction_interpretation = heart_attack_prediction.get("prediction_interpretation", {})
            if prediction_interpretation:
                st.markdown("**🔍 Risk Assessment Interpretation:**")
                st.info(f"📋 **Assessment:** {prediction_interpretation.get('risk_assessment', 'N/A')}")
                st.info(f"📊 **Confidence:** {prediction_interpretation.get('confidence', 'N/A')}")
                st.info(f"💡 **Recommendation:** {prediction_interpretation.get('recommendation', 'N/A')}")
                
                # Risk factors
                risk_factors = prediction_interpretation.get("risk_factors", [])
                if risk_factors:
                    st.markdown("**⚠️ Identified Risk Factors:**")
                    for factor in risk_factors:
                        st.markdown(f"- {factor}")
            
            # Feature Analysis
            if heart_attack_features and not heart_attack_features.get('error'):
                with st.expander("🔍 View Model Features Analysis"):
                    extracted_features = heart_attack_features.get("extracted_features", {})
                    feature_interpretation = heart_attack_features.get("feature_interpretation", {})
                    model_info_features = heart_attack_features.get("model_info", {})
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📊 Extracted Features:**")
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
                        st.markdown("**🤖 Model Information:**")
                        st.markdown(f"- **Model Type:** {model_info_features.get('model_type', 'Unknown')}")
                        st.markdown(f"- **Expected Features:** {model_info_features.get('features_expected', [])}")
                        st.markdown(f"- **Features Count:** {model_info_features.get('features_count', 0)}")
                        st.markdown(f"- **FastAPI Server URL:** {model_info_features.get('fastapi_server_url', 'Unknown')}")
                        st.markdown(f"- **Data Types:** {model_info_features.get('data_types', {})}")
            
            # Model Information
            if model_info:
                with st.expander("🤖 Model Information"):
                    st.json(model_info)
            
            # Download heart attack prediction
            st.download_button(
                "📄 Download Heart Attack Prediction",
                safe_json_dumps(heart_attack_prediction),
                f"heart_attack_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
        else:
            # Display error information
            error_msg = heart_attack_prediction.get('error', 'Unknown error')
            st.error(f"❌ Heart Attack Prediction Error: {error_msg}")
            
            model_info = heart_attack_prediction.get('model_info', {})
            if model_info and 'fastapi_server_url' in model_info:
                st.warning(f"⚠️ FastAPI Server: {model_info['fastapi_server_url']}")
            
            st.info("💡 Heart attack prediction requires a running FastAPI server. Please check the configuration in the sidebar.")

    # Interactive Chatbot Section
    if results.get("chatbot_ready", False) and st.session_state.chatbot_context:
        st.markdown('<div class="chatbot-header">💬 Node 9: Interactive Chatbot</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="chatbot-box">🤖 Ask questions about the patient\'s medical data, analysis results, heart attack risk assessment, or request specific insights based on the deidentified records.</div>', unsafe_allow_html=True)
        
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
                "Heart Attack Prediction": "✅ Available" if st.session_state.chatbot_context.get("heart_attack_prediction") else "❌ None"
            }
            
            for key, value in context_summary.items():
                if "✅" in value:
                    st.success(f"{key}: {value}")
                else:
                    st.warning(f"{key}: {value}")
        
        # Chat input
        user_question = st.chat_input("💬 Ask a question about the medical data or heart attack risk...")
        
        if user_question:
            # Add user message to chat history
            st.session_state.chatbot_messages.append({"role": "user", "content": user_question})
            
            # Get response from chatbot
            try:
                with st.spinner("🤖 Analyzing medical data and heart attack prediction, generating response..."):
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
            "What is the patient's heart attack risk assessment?",
            "What factors contribute to the heart attack risk score?",
            "Explain the significance of the ICD-10 diagnosis codes found",
            "What drug interactions should be considered?",
            "How do the medications relate to the heart attack risk prediction?",
            "Summarize the key health insights including heart attack risk"
        ]
        
        for i, question in enumerate(sample_questions):
            if st.button(f"💬 {question}", key=f"sample_q_{i}"):
                # Use the sample question as if user typed it
                st.session_state.chatbot_messages.append({"role": "user", "content": question})
                
                try:
                    with st.spinner("🤖 Analyzing medical data and heart attack prediction, generating response..."):
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

    # Complete Health Agent Analysis Report Download
    st.markdown('<div class="step-header">💾 Complete Health Agent Analysis Report</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Complete JSON report
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
                "heart_attack_prediction_enabled": True,
                "heart_attack_risk_score": results.get("heart_attack_risk_score", 0.0),
                "enhancement_version": results.get("enhancement_version", "v4.0"),
                "step_status": safe_get(results, "step_status", {}),
                "extraction_enabled": True,
                "interactive_features": ["Medical Data Q&A", "ICD-10 Analysis", "NDC Code Interpretation", "Heart Attack Risk Assessment"]
            },
            "api_outputs": safe_get(results, "api_outputs", {}),
            "deidentified_data": safe_get(results, "deidentified_data", {}),
            "structured_extractions": safe_get(results, "structured_extractions", {}),
            "entity_extraction": safe_get(results, "entity_extraction", {}),
            "health_trajectory": safe_get(results, "health_trajectory", ""),
            "final_summary": safe_get(results, "final_summary", ""),
            "heart_attack_prediction": safe_get(results, "heart_attack_prediction", {}),
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
            f"health_agent_analysis_{patient_last_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Text report
        patient_name = f"{processed_patient.get('first_name', 'Unknown')} {processed_patient.get('last_name', 'Unknown')}"
        
        # Get extraction counts
        medical_extraction = safe_get(structured_extractions, 'medical', {})
        pharmacy_extraction = safe_get(structured_extractions, 'pharmacy', {})
        medical_records = len(safe_get(medical_extraction, 'hlth_srvc_records', []))
        pharmacy_records = len(safe_get(pharmacy_extraction, 'ndc_records', []))
        
        # Heart attack prediction info
        heart_risk_score = results.get("heart_attack_risk_score", 0.0)
        heart_risk_level = safe_get(heart_attack_prediction, "risk_level", "Unknown")
        model_source = safe_get(heart_attack_prediction, "model_info", {}).get("model_source", "fastapi_server")
        fastapi_server_url = safe_get(heart_attack_prediction, "model_info", {}).get("fastapi_server_url", "unknown")
        
        text_report = f"""
HEALTH AGENT ANALYSIS REPORT
============================
Patient: {patient_name}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: {'Success' if results.get('success', False) else 'Failed'}
Nodes Completed: {results.get('processing_steps_completed', 0)}/8
Workflow Engine: Health Agent
AI Engine: AI Analysis
AI Model: llama3.1-70b
Interactive Chatbot: {'Ready' if results.get('chatbot_ready', False) else 'Failed'}
Heart Attack Prediction: {'Enabled' if heart_attack_prediction else 'Disabled'}
Enhancement Version: {results.get('enhancement_version', 'v4.0')}

HEART ATTACK RISK ASSESSMENT:
============================
Risk Score: {heart_risk_score:.3f} ({heart_risk_score*100:.1f}%)
Risk Level: {heart_risk_level}
Model Source: {model_source}
FastAPI Server: {fastapi_server_url}
Features Used: {safe_get(heart_attack_prediction, 'model_info', {}).get('
