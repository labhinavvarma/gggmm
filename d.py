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

# Custom CSS for boxed modules
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

.section-box {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #e9ecef;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.section-title {
    font-size: 1.3rem;
    color: #2c3e50;
    font-weight: 600;
    margin-bottom: 1rem;
    border-bottom: 2px solid #3498db;
    padding-bottom: 0.5rem;
}

.status-success {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #28a745;
    margin: 1rem 0;
}

.status-error {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #dc3545;
    margin: 1rem 0;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.metric-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    border: 1px solid #dee2e6;
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

.chat-container {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 10px;
    padding: 1rem;
    border: 1px solid #dee2e6;
}

.chat-message {
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0;
    border-radius: 8px;
}

.user-message {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left: 3px solid #2196f3;
}

.assistant-message {
    background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
    border-left: 3px solid #9c27b0;
}

.example-questions {
    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #ffcc02;
    margin-top: 1rem;
}

.json-container {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #dee2e6;
    max-height: 400px;
    overflow-y: auto;
    font-family: monospace;
    font-size: 0.85rem;
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
if not AGENT_AVAILABLE:
    st.markdown(f'<div class="status-error">❌ Failed to import Health Agent: {import_error}</div>', unsafe_allow_html=True)
    st.stop()

# 1. PATIENT INFORMATION BOX
st.markdown("""
<div class="section-box">
    <div class="section-title">👤 Patient Information</div>
</div>
""", unsafe_allow_html=True)

with st.container():
    with st.form("patient_input_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            first_name = st.text_input("First Name *", value="")
            last_name = st.text_input("Last Name *", value="")
        
        with col2:
            ssn = st.text_input("SSN *", value="")
            date_of_birth = st.date_input(
                "Date of Birth *", 
                value=datetime.now().date(),
                min_value=datetime(1900, 1, 1).date(),
                max_value=datetime.now().date()
            )
        
        with col3:
            gender = st.selectbox("Gender *", ["F", "M"])
            zip_code = st.text_input("Zip Code *", value="")
        
        # Show calculated age
        if date_of_birth:
            calculated_age = calculate_age(date_of_birth)
            if calculated_age is not None:
                st.info(f"📅 **Calculated Age:** {calculated_age} years old")
        
        # 2. RUN HEALTH ANALYSIS BUTTON
        submitted = st.form_submit_button(
            "🚀 Run Health Analysis", 
            use_container_width=True,
            disabled=st.session_state.analysis_running
        )

# Analysis Status
if st.session_state.analysis_running:
    st.markdown('<div class="status-success">🔄 Health analysis workflow executing... Please wait.</div>', unsafe_allow_html=True)

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
                st.success("✅ Health Analysis Agent initialized")
            except Exception as e:
                st.error(f"❌ Failed to initialize Health Agent: {str(e)}")
                st.stop()
        
        st.session_state.analysis_running = True
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🚀 Executing health analysis..."):
            try:
                # Progress updates
                for i, step in enumerate([
                    "Initializing workflow...",
                    "Fetching medical data...", 
                    "Deidentifying data...",
                    "Extracting medical information...",
                    "Analyzing health trajectory...",
                    "Predicting heart attack risk...",
                    "Initializing chatbot..."
                ]):
                    status_text.text(f"🔄 {step}")
                    progress_bar.progress(int((i + 1) * 14))
                    time.sleep(0.3)
                
                # Execute analysis
                results = st.session_state.agent.run_analysis(patient_data)
                
                if results.get("success", False):
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis completed successfully!")
                    st.session_state.analysis_results = results
                    st.session_state.chatbot_context = results.get("chatbot_context", {})
                    st.markdown('<div class="status-success">✅ Health analysis completed successfully!</div>', unsafe_allow_html=True)
                else:
                    st.session_state.analysis_results = results
                    st.warning("⚠️ Analysis completed with some errors.")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.session_state.analysis_results = {
                    "success": False,
                    "error": str(e),
                    "patient_data": patient_data,
                    "errors": [str(e)]
                }
            finally:
                st.session_state.analysis_running = False

# Display Results if Available
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    # Show errors if any
    errors = safe_get(results, 'errors', [])
    if errors:
        st.markdown('<div class="status-error">❌ Analysis errors occurred</div>', unsafe_allow_html=True)

    # 3. CHATBOT IN BOX
    if results.get("chatbot_ready", False) and st.session_state.chatbot_context:
        st.markdown("""
        <div class="section-box">
            <div class="section-title">💬 Medical Data Assistant</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Chat input
            user_question = st.chat_input("💬 Ask a question about the medical analysis...")
            
            # Display chat history
            if st.session_state.chatbot_messages:
                st.markdown("**Recent Conversation:**")
                for message in st.session_state.chatbot_messages[-6:]:
                    if message["role"] == "user":
                        st.markdown(f'<div class="chat-message user-message"><strong>👤:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-message assistant-message"><strong>🤖:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            
            # Handle chat input
            if user_question:
                st.session_state.chatbot_messages.append({"role": "user", "content": user_question})
                
                try:
                    with st.spinner("🤖 Processing..."):
                        chatbot_response = st.session_state.agent.chat_with_data(
                            user_question, 
                            st.session_state.chatbot_context, 
                            st.session_state.chatbot_messages
                        )
                    
                    st.session_state.chatbot_messages.append({"role": "assistant", "content": chatbot_response})
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chatbot_messages = []
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="example-questions">
                <h4>💡 Example Questions</h4>
                <p><strong>Medications:</strong></p>
                <ul>
                    <li>What medications was prescribed?</li>
                    <li>Any diabetes medications?</li>
                </ul>
                <p><strong>Diagnoses:</strong></p>
                <ul>
                    <li>What diagnosis codes were found?</li>
                    <li>Any chronic conditions?</li>
                </ul>
                <p><strong>Risk Assessment:</strong></p>
                <ul>
                    <li>What's the heart attack risk?</li>
                    <li>Key risk factors?</li>
                </ul>
                <p><strong>Summary:</strong></p>
                <ul>
                    <li>Summarize key findings</li>
                    <li>Health recommendations?</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # 4. MILLIMAN DATA BUTTON
    if st.button("📊 Milliman Data", use_container_width=True):
        st.markdown("""
        <div class="section-box">
            <div class="section-title">📊 Milliman Deidentified Data</div>
        </div>
        """, unsafe_allow_html=True)
        
        deidentified_data = safe_get(results, 'deidentified_data', {})
        
        if deidentified_data:
            tab1, tab2 = st.tabs(["🏥 Medical Data", "💊 Pharmacy Data"])
            
            with tab1:
                medical_data = safe_get(deidentified_data, 'medical', {})
                if medical_data:
                    st.markdown('<div class="json-container">', unsafe_allow_html=True)
                    st.json(medical_data)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.download_button(
                        "📥 Download Medical Data JSON",
                        safe_json_dumps(medical_data),
                        f"medical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    st.warning("No medical data available")
            
            with tab2:
                pharmacy_data = safe_get(deidentified_data, 'pharmacy', {})
                if pharmacy_data:
                    st.markdown('<div class="json-container">', unsafe_allow_html=True)
                    st.json(pharmacy_data)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.download_button(
                        "📥 Download Pharmacy Data JSON",
                        safe_json_dumps(pharmacy_data),
                        f"pharmacy_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    st.warning("No pharmacy data available")

    # 5. MEDICAL/PHARMACY DATA EXTRACTION BUTTON
    if st.button("🔍 Medical/Pharmacy Data Extraction", use_container_width=True):
        st.markdown("""
        <div class="section-box">
            <div class="section-title">🔍 Medical/Pharmacy Data Extraction</div>
        </div>
        """, unsafe_allow_html=True)
        
        structured_extractions = safe_get(results, 'structured_extractions', {})
        
        if structured_extractions:
            tab1, tab2 = st.tabs(["🏥 Medical Extraction", "💊 Pharmacy Extraction"])
            
            with tab1:
                medical_extraction = safe_get(structured_extractions, 'medical', {})
                if medical_extraction and not medical_extraction.get('error'):
                    extraction_summary = safe_get(medical_extraction, 'extraction_summary', {})
                    
                    st.markdown(f"""
                    <div class="metric-grid">
                        <div class="metric-card">
                            <h3>{extraction_summary.get('total_hlth_srvc_records', 0)}</h3>
                            <p>Health Service Records</p>
                        </div>
                        <div class="metric-card">
                            <h3>{extraction_summary.get('total_diagnosis_codes', 0)}</h3>
                            <p>Diagnosis Codes</p>
                        </div>
                        <div class="metric-card">
                            <h3>{len(extraction_summary.get('unique_service_codes', []))}</h3>
                            <p>Unique Service Codes</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    hlth_srvc_records = safe_get(medical_extraction, 'hlth_srvc_records', [])
                    if hlth_srvc_records:
                        st.markdown("**📋 All Medical Records:**")
                        for i, record in enumerate(hlth_srvc_records, 1):
                            with st.expander(f"Medical Record {i} - Service Code: {record.get('hlth_srvc_cd', 'N/A')}"):
                                st.write(f"**Service Code:** `{record.get('hlth_srvc_cd', 'N/A')}`")
                                st.write(f"**Data Path:** `{record.get('data_path', 'N/A')}`")
                                
                                diagnosis_codes = record.get('diagnosis_codes', [])
                                if diagnosis_codes:
                                    st.write("**Diagnosis Codes:**")
                                    for idx, diag in enumerate(diagnosis_codes, 1):
                                        source_info = f" (from {diag.get('source', 'individual field')})" if diag.get('source') else ""
                                        st.write(f"  {idx}. `{diag.get('code', 'N/A')}`{source_info}")
                else:
                    st.warning("No medical extraction data available")
            
            with tab2:
                pharmacy_extraction = safe_get(structured_extractions, 'pharmacy', {})
                if pharmacy_extraction and not pharmacy_extraction.get('error'):
                    extraction_summary = safe_get(pharmacy_extraction, 'extraction_summary', {})
                    
                    st.markdown(f"""
                    <div class="metric-grid">
                        <div class="metric-card">
                            <h3>{extraction_summary.get('total_ndc_records', 0)}</h3>
                            <p>NDC Records</p>
                        </div>
                        <div class="metric-card">
                            <h3>{len(extraction_summary.get('unique_ndc_codes', []))}</h3>
                            <p>Unique NDC Codes</p>
                        </div>
                        <div class="metric-card">
                            <h3>{len(extraction_summary.get('unique_label_names', []))}</h3>
                            <p>Unique Medications</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    ndc_records = safe_get(pharmacy_extraction, 'ndc_records', [])
                    if ndc_records:
                        st.markdown("**💊 All Pharmacy Records:**")
                        for i, record in enumerate(ndc_records, 1):
                            with st.expander(f"Pharmacy Record {i} - {record.get('lbl_nm', 'N/A')}"):
                                st.write(f"**NDC Code:** `{record.get('ndc', 'N/A')}`")
                                st.write(f"**Label Name:** `{record.get('lbl_nm', 'N/A')}`")
                                st.write(f"**Data Path:** `{record.get('data_path', 'N/A')}`")
                else:
                    st.warning("No pharmacy extraction data available")

    # 6. ENHANCED ENTITY EXTRACTION BUTTON
    if st.button("🎯 Enhanced Entity Extraction", use_container_width=True):
        st.markdown("""
        <div class="section-box">
            <div class="section-title">🎯 Enhanced Entity Extraction</div>
        </div>
        """, unsafe_allow_html=True)
        
        entity_extraction = safe_get(results, 'entity_extraction', {})
        if entity_extraction:
            # Entity cards
            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>🩺</h3>
                    <p><strong>Diabetes</strong></p>
                    <h4>{entity_extraction.get('diabetics', 'unknown').upper()}</h4>
                </div>
                <div class="metric-card">
                    <h3>👥</h3>
                    <p><strong>Age Group</strong></p>
                    <h4>{entity_extraction.get('age_group', 'unknown').upper()}</h4>
                </div>
                <div class="metric-card">
                    <h3>🚬</h3>
                    <p><strong>Smoking</strong></p>
                    <h4>{entity_extraction.get('smoking', 'unknown').upper()}</h4>
                </div>
                <div class="metric-card">
                    <h3>🍷</h3>
                    <p><strong>Alcohol</strong></p>
                    <h4>{entity_extraction.get('alcohol', 'unknown').upper()}</h4>
                </div>
                <div class="metric-card">
                    <h3>💓</h3>
                    <p><strong>Blood Pressure</strong></p>
                    <h4>{entity_extraction.get('blood_pressure', 'unknown').upper()}</h4>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Medical conditions
            medical_conditions = safe_get(entity_extraction, 'medical_conditions', [])
            if medical_conditions:
                st.markdown("**🏥 Medical Conditions Identified:**")
                for condition in medical_conditions:
                    st.write(f"• {condition}")
            
            # Medications identified
            medications_identified = safe_get(entity_extraction, 'medications_identified', [])
            if medications_identified:
                st.markdown("**💊 Medications Identified:**")
                for med in medications_identified:
                    st.write(f"• **{med.get('label_name', 'N/A')}** (NDC: {med.get('ndc', 'N/A')})")

    # 7. HEALTH TRAJECTORY BUTTON
    if st.button("📈 Health Trajectory", use_container_width=True):
        st.markdown("""
        <div class="section-box">
            <div class="section-title">📈 Health Trajectory Analysis</div>
        </div>
        """, unsafe_allow_html=True)
        
        health_trajectory = safe_get(results, 'health_trajectory', '')
        if health_trajectory:
            st.markdown(health_trajectory)
        else:
            st.warning("Health trajectory analysis not available")

    # 8. FINAL SUMMARY BUTTON
    if st.button("📋 Final Summary", use_container_width=True):
        st.markdown("""
        <div class="section-box">
            <div class="section-title">📋 Clinical Summary</div>
        </div>
        """, unsafe_allow_html=True)
        
        final_summary = safe_get(results, 'final_summary', '')
        if final_summary:
            st.markdown(final_summary)
        else:
            st.warning("Final summary not available")

    # 9. HEART ATTACK RISK PREDICTION BUTTON
    if st.button("❤️ Heart Attack Risk Prediction", use_container_width=True):
        st.markdown("""
        <div class="section-box">
            <div class="section-title">❤️ Heart Attack Risk Assessment</div>
        </div>
        """, unsafe_allow_html=True)
        
        heart_attack_prediction = safe_get(results, 'heart_attack_prediction', {})
        if heart_attack_prediction and not heart_attack_prediction.get('error'):
            risk_level = heart_attack_prediction.get("risk_level", "UNKNOWN")
            risk_score = heart_attack_prediction.get("risk_score", 0.0)
            risk_icon = heart_attack_prediction.get("risk_icon", "❓")
            risk_percentage = heart_attack_prediction.get("risk_percentage", "0.0%")
            
            # Risk display
            if risk_level == "HIGH":
                st.markdown(f'<div class="risk-high">{risk_icon} <strong>Risk Level: {risk_level}</strong><br>Risk Score: {risk_percentage}<br>⚠️ Immediate medical consultation recommended</div>', unsafe_allow_html=True)
            elif risk_level == "MODERATE":
                st.markdown(f'<div class="risk-moderate">{risk_icon} <strong>Risk Level: {risk_level}</strong><br>Risk Score: {risk_percentage}<br>📋 Regular monitoring advised</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">{risk_icon} <strong>Risk Level: {risk_level}</strong><br>Risk Score: {risk_percentage}<br>✅ Continue healthy lifestyle practices</div>', unsafe_allow_html=True)
            
            # Features used
            heart_attack_features = safe_get(results, 'heart_attack_features', {})
            if heart_attack_features:
                extracted_features = heart_attack_features.get("extracted_features", {})
                feature_interpretation = heart_attack_features.get("feature_interpretation", {})
                
                if extracted_features:
                    st.markdown("**🎯 Model Features Used:**")
                    for feature, value in extracted_features.items():
                        interpretation = feature_interpretation.get(feature, str(value))
                        st.write(f"• **{feature}:** {interpretation}")
                
                # Risk factors
                prediction_interpretation = heart_attack_prediction.get("prediction_interpretation", {})
                risk_factors = prediction_interpretation.get("risk_factors", [])
                if risk_factors:
                    st.markdown("**⚠️ Identified Risk Factors:**")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
        else:
            error_msg = heart_attack_prediction.get('error', 'Heart attack prediction not available')
            st.error(f"❌ {error_msg}")
