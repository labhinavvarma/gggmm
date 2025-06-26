# Configure Streamlit page FIRST - before any other Streamlit commands
import streamlit as st
st.set_page_config(
    page_title="Health Analysis Agent",
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
    from fixed_health_agent import HealthAnalysisAgent, Config
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    import_error = str(e)

# Custom CSS for clean, modern styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 600;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.status-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #3498db;
    margin: 1rem 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.success-card {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #28a745;
    margin: 1rem 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.error-card {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #dc3545;
    margin: 1rem 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #e9ecef;
    margin: 0.5rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    transition: transform 0.2s;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}

.risk-high {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border: 2px solid #dc3545;
    color: #721c24;
    padding: 1.5rem;
    border-radius: 10px;
    font-weight: bold;
    margin: 1rem 0;
}

.risk-moderate {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    border: 2px solid #ffc107;
    color: #856404;
    padding: 1.5rem;
    border-radius: 10px;
    font-weight: bold;
    margin: 1rem 0;
}

.risk-low {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border: 2px solid #28a745;
    color: #155724;
    padding: 1.5rem;
    border-radius: 10px;
    font-weight: bold;
    margin: 1rem 0;
}

.section-header {
    font-size: 1.4rem;
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
    font-weight: 600;
}

.chatbot-container {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #dee2e6;
    margin: 1rem 0;
}

.chat-message {
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0;
    border-radius: 10px;
}

.user-message {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left: 3px solid #2196f3;
}

.assistant-message {
    background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
    border-left: 3px solid #9c27b0;
}

.input-section {
    background: white;
    padding: 2rem;
    border-radius: 10px;
    border: 1px solid #e9ecef;
    margin: 1rem 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.hidden {
    display: none;
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

# Main Title
st.markdown('<h1 class="main-header">🏥 Health Analysis Agent</h1>', unsafe_allow_html=True)

# Display import status
if AGENT_AVAILABLE:
    st.markdown('<div class="success-card">✅ Health Analysis Agent is ready for processing!</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="error-card">❌ Failed to import Health Agent: {import_error}</div>', unsafe_allow_html=True)
    st.stop()

# Patient Input Form
st.markdown('<div class="section-header">👤 Patient Information</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    with st.form("patient_input_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            first_name = st.text_input("First Name *", value="", help="Patient's first name")
            last_name = st.text_input("Last Name *", value="", help="Patient's last name")
        
        with col2:
            ssn = st.text_input("SSN *", value="", help="Social Security Number (9+ digits)")
            date_of_birth = st.date_input(
                "Date of Birth *", 
                value=datetime.now().date(),
                min_value=datetime(1900, 1, 1).date(),
                max_value=datetime.now().date(),
                help="Patient's date of birth"
            )
        
        with col3:
            gender = st.selectbox("Gender *", ["F", "M"], help="Patient's gender")
            zip_code = st.text_input("Zip Code *", value="", help="Patient's zip code (5+ digits)")
        
        # Show calculated age
        if date_of_birth:
            calculated_age = calculate_age(date_of_birth)
            if calculated_age is not None:
                st.info(f"📅 **Calculated Age:** {calculated_age} years old")
                
                if calculated_age > 120:
                    st.warning("⚠️ Age seems unusually high. Please verify the date of birth.")
                elif calculated_age < 0:
                    st.error("❌ Date of birth cannot be in the future.")
        
        # Submit button
        submitted = st.form_submit_button(
            "🚀 Run Health Analysis", 
            use_container_width=True,
            disabled=st.session_state.analysis_running
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Analysis Status Section
if st.session_state.analysis_running:
    st.markdown('<div class="status-card">🔄 Health analysis workflow executing... Please wait.</div>', unsafe_allow_html=True)

# Run Health Analysis
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
    
    calculated_age = calculate_age(date_of_birth)
    st.info(f"📤 Processing: {patient_data['first_name']} {patient_data['last_name']} (Age: {calculated_age})")
    
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
                st.success("✅ Health Analysis Agent initialized successfully")
            except Exception as e:
                st.error(f"❌ Failed to initialize Health Agent: {str(e)}")
                st.stop()
        
        st.session_state.analysis_running = True
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🚀 Executing health analysis workflow..."):
            try:
                status_text.text("🚀 Initializing workflow...")
                progress_bar.progress(10)
                time.sleep(0.5)
                
                status_text.text("📊 Fetching medical data...")
                progress_bar.progress(25)
                time.sleep(0.5)
                
                status_text.text("🔒 Deidentifying data...")
                progress_bar.progress(40)
                time.sleep(0.5)
                
                status_text.text("🔍 Extracting medical information...")
                progress_bar.progress(55)
                time.sleep(0.5)
                
                status_text.text("📈 Analyzing health trajectory...")
                progress_bar.progress(70)
                time.sleep(0.5)
                
                status_text.text("❤️ Predicting heart attack risk...")
                progress_bar.progress(85)
                time.sleep(0.5)
                
                status_text.text("💬 Initializing chatbot...")
                progress_bar.progress(95)
                
                # Execute the analysis
                results = st.session_state.agent.run_analysis(patient_data)
                
                if results.get("success", False):
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis completed successfully!")
                    
                    st.session_state.analysis_results = results
                    st.session_state.chatbot_context = results.get("chatbot_context", {})
                    st.markdown('<div class="success-card">✅ Health analysis completed successfully!</div>', unsafe_allow_html=True)
                else:
                    progress_bar.progress(70)
                    status_text.text("⚠️ Analysis completed with errors")
                    st.session_state.analysis_results = results
                    st.warning("⚠️ Analysis completed but with some errors. Check results below.")
                
            except Exception as e:
                progress_bar.progress(0)
                status_text.text("❌ Analysis failed")
                st.error(f"❌ Error in analysis execution: {str(e)}")
                
                st.session_state.analysis_results = {
                    "success": False,
                    "error": str(e),
                    "patient_data": patient_data,
                    "errors": [str(e)],
                    "processing_steps_completed": 0
                }
            finally:
                st.session_state.analysis_running = False

# Display Results
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    # Patient Summary
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
        st.info(f"📋 Analysis completed for: {processed_patient.get('first_name', 'Unknown')} {processed_patient.get('last_name', 'Unknown')}{age_display}")

    # Show errors if any
    errors = safe_get(results, 'errors', [])
    if errors:
        st.markdown('<div class="error-card">❌ Analysis errors:</div>', unsafe_allow_html=True)
        for error in errors:
            st.error(f"• {error}")

    # Heart Attack Risk Assessment
    heart_attack_prediction = safe_get(results, 'heart_attack_prediction', {})
    if heart_attack_prediction:
        st.markdown('<div class="section-header">❤️ Heart Attack Risk Assessment</div>', unsafe_allow_html=True)
        
        if not heart_attack_prediction.get('error'):
            risk_level = heart_attack_prediction.get("risk_level", "UNKNOWN")
            risk_score = heart_attack_prediction.get("risk_score", 0.0)
            risk_icon = heart_attack_prediction.get("risk_icon", "❓")
            risk_percentage = heart_attack_prediction.get("risk_percentage", "0.0%")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-card"><h3>{risk_icon}</h3><p><strong>Risk Level</strong></p><h4>{risk_level}</h4></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h3>📊</h3><p><strong>Risk Score</strong></p><h4>{risk_percentage}</h4></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><h3>🤖</h3><p><strong>Model</strong></p><h4>FastAPI</h4></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-card"><h3>🎯</h3><p><strong>Features</strong></p><h4>5 Used</h4></div>', unsafe_allow_html=True)
            
            # Risk interpretation
            prediction_interpretation = heart_attack_prediction.get("prediction_interpretation", {})
            if prediction_interpretation:
                recommendation = prediction_interpretation.get('recommendation', 'N/A')
                if risk_level == "HIGH":
                    st.markdown(f'<div class="risk-high">⚠️ <strong>{recommendation}</strong></div>', unsafe_allow_html=True)
                elif risk_level == "MODERATE":
                    st.markdown(f'<div class="risk-moderate">📋 <strong>{recommendation}</strong></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-low">✅ <strong>{recommendation}</strong></div>', unsafe_allow_html=True)
        else:
            st.error(f"❌ Heart Attack Prediction Error: {heart_attack_prediction.get('error', 'Unknown error')}")

    # Patient Summary (Deidentified Data)
    deidentified_data = safe_get(results, 'deidentified_data', {})
    if deidentified_data:
        st.markdown('<div class="section-header">🔒 Patient Summary</div>', unsafe_allow_html=True)
        
        deident_medical = safe_get(deidentified_data, 'medical', {})
        if deident_medical and not deident_medical.get('error'):
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                name = f"{safe_str(safe_get(deident_medical, 'src_mbr_first_nm', 'N/A'))} {safe_str(safe_get(deident_medical, 'src_mbr_last_nm', 'N/A'))}"
                st.markdown(f'<div class="metric-card"><h3>👤</h3><p><strong>Name</strong></p><h4>{name}</h4></div>', unsafe_allow_html=True)
            
            with col2:
                age = safe_str(safe_get(deident_medical, 'src_mbr_age', 'N/A'))
                st.markdown(f'<div class="metric-card"><h3>📅</h3><p><strong>Age</strong></p><h4>{age}</h4></div>', unsafe_allow_html=True)
            
            with col3:
                zip_code = safe_str(safe_get(deident_medical, 'src_mbr_zip_cd', 'N/A'))
                st.markdown(f'<div class="metric-card"><h3>📍</h3><p><strong>Zip Code</strong></p><h4>{zip_code}</h4></div>', unsafe_allow_html=True)
            
            with col4:
                entity_extraction = safe_get(results, 'entity_extraction', {})
                diabetes_status = safe_get(entity_extraction, 'diabetics', 'unknown')
                diabetes_display = "Yes" if diabetes_status == "yes" else "No"
                st.markdown(f'<div class="metric-card"><h3>🩺</h3><p><strong>Diabetes</strong></p><h4>{diabetes_display}</h4></div>', unsafe_allow_html=True)
            
            with col5:
                bp_status = safe_get(entity_extraction, 'blood_pressure', 'unknown')
                bp_display = "Managed" if bp_status in ["managed", "diagnosed"] else "Unknown"
                st.markdown(f'<div class="metric-card"><h3>💓</h3><p><strong>Blood Pressure</strong></p><h4>{bp_display}</h4></div>', unsafe_allow_html=True)

        # Download deidentified data (only download option available)
        if st.button("📄 Download Deidentified Data", use_container_width=True):
            patient_last_name = processed_patient.get('last_name', 'unknown')
            deidentified_report = {
                "deidentified_data": deidentified_data,
                "timestamp": datetime.now().isoformat(),
                "patient_reference": f"{processed_patient.get('first_name', 'Unknown')} {processed_patient.get('last_name', 'Unknown')}"
            }
            
            st.download_button(
                "💾 Download JSON",
                safe_json_dumps(deidentified_report),
                f"deidentified_data_{patient_last_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

    # Interactive Chatbot
    if results.get("chatbot_ready", False) and st.session_state.chatbot_context:
        st.markdown('<div class="section-header">💬 Medical Data Assistant</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="chatbot-container">', unsafe_allow_html=True)
        st.markdown("🤖 **Ask questions about the medical data, medications, diagnoses, or heart attack risk assessment.**")
        
        # Chat input
        user_question = st.chat_input("💬 Ask a question about the medical analysis...")
        
        # Display recent chat history
        if st.session_state.chatbot_messages:
            st.markdown("**Recent Conversation:**")
            recent_messages = st.session_state.chatbot_messages[-6:]  # Last 3 exchanges
            for message in recent_messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message"><strong>👤 You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant-message"><strong>🤖 Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        
        # Handle chat input
        if user_question:
            st.session_state.chatbot_messages.append({"role": "user", "content": user_question})
            
            try:
                with st.spinner("🤖 Analyzing medical data..."):
                    chatbot_response = st.session_state.agent.chat_with_data(
                        user_question, 
                        st.session_state.chatbot_context, 
                        st.session_state.chatbot_messages
                    )
                
                st.session_state.chatbot_messages.append({"role": "assistant", "content": chatbot_response})
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Chatbot error: {str(e)}")
        
        # Quick action buttons
        col1, col2, col3, col4 = st.columns(4)
        quick_questions = [
            "What medications was this patient prescribed?",
            "What diagnosis codes were found?", 
            "What factors contribute to heart attack risk?",
            "Summarize key health insights"
        ]
        
        for i, (col, question) in enumerate(zip([col1, col2, col3, col4], quick_questions)):
            with col:
                button_text = question.split('?')[0].replace('What ', '').replace('Summarize ', 'Summary')
                if st.button(f"💬 {button_text}", key=f"quick_q_{i}", use_container_width=True):
                    st.session_state.chatbot_messages.append({"role": "user", "content": question})
                    
                    try:
                        with st.spinner("🤖 Processing..."):
                            chatbot_response = st.session_state.agent.chat_with_data(
                                question, 
                                st.session_state.chatbot_context, 
                                st.session_state.chatbot_messages
                            )
                        
                        st.session_state.chatbot_messages.append({"role": "assistant", "content": chatbot_response})
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chatbot_messages = []
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Expandable detailed sections
    if st.button("📖 Show Detailed Analysis", use_container_width=True):
        # FIXED: Get structured_extractions from results
        structured_extractions = safe_get(results, 'structured_extractions', {})
        
        if structured_extractions:
            st.markdown('<div class="section-header">🔍 Detailed Medical Analysis</div>', unsafe_allow_html=True)
            
            # Medical extraction details
            medical_extraction = safe_get(structured_extractions, 'medical', {})
            if medical_extraction and not medical_extraction.get('error'):
                with st.expander("🏥 Medical Records Analysis", expanded=True):
                    extraction_summary = safe_get(medical_extraction, 'extraction_summary', {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Health Service Records", extraction_summary.get('total_hlth_srvc_records', 0))
                    with col2:
                        st.metric("Diagnosis Codes Found", extraction_summary.get('total_diagnosis_codes', 0))
                    with col3:
                        st.metric("Unique Service Codes", len(extraction_summary.get('unique_service_codes', [])))
                    
                    hlth_srvc_records = safe_get(medical_extraction, 'hlth_srvc_records', [])
                    if hlth_srvc_records:
                        st.markdown("**📋 Medical Records Found:**")
                        for i, record in enumerate(hlth_srvc_records[:3]):  # Show first 3
                            st.markdown(f"**Record {i+1}:**")
                            st.write(f"- Service Code: `{record.get('hlth_srvc_cd', 'N/A')}`")
                            diagnosis_codes = record.get('diagnosis_codes', [])
                            if diagnosis_codes:
                                codes_list = [f"`{d.get('code', 'N/A')}`" for d in diagnosis_codes[:5]]
                                st.write(f"- Diagnosis Codes: {', '.join(codes_list)}")
                        
                        if len(hlth_srvc_records) > 3:
                            st.info(f"Showing first 3 of {len(hlth_srvc_records)} medical records.")
            
            # Pharmacy extraction details
            pharmacy_extraction = safe_get(structured_extractions, 'pharmacy', {})
            if pharmacy_extraction and not pharmacy_extraction.get('error'):
                with st.expander("💊 Pharmacy Records Analysis", expanded=True):
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
                        st.markdown("**💊 Medications Found:**")
                        for i, record in enumerate(ndc_records[:5]):  # Show first 5
                            st.write(f"**{i+1}.** {record.get('lbl_nm', 'N/A')} (NDC: `{record.get('ndc', 'N/A')}`)")
                        
                        if len(ndc_records) > 5:
                            st.info(f"Showing first 5 of {len(ndc_records)} pharmacy records.")
        
        # Health trajectory and summary
        health_trajectory = safe_get(results, 'health_trajectory', '')
        if health_trajectory:
            with st.expander("📈 Health Trajectory Analysis", expanded=True):
                st.markdown(health_trajectory)
        
        final_summary = safe_get(results, 'final_summary', '')
        if final_summary:
            with st.expander("📋 Clinical Summary", expanded=True):
                st.markdown(final_summary)
