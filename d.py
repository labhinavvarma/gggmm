# Configure Streamlit page FIRST - before any other Streamlit commands
import streamlit as st

# Determine sidebar state based on chatbot readiness
if 'analysis_results' in st.session_state and st.session_state.get('analysis_results') and st.session_state.analysis_results.get("chatbot_ready", False):
    sidebar_state = "expanded"
else:
    sidebar_state = "collapsed"

st.set_page_config(
    page_title="Enhanced Health Analysis Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state=sidebar_state
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

# Import the Enhanced Health Analysis Agent
AGENT_AVAILABLE = False
import_error = None
HealthAnalysisAgent = None
Config = None

try:
    from health_agent_core import HealthAnalysisAgent, Config
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    import_error = str(e)

# Custom CSS for enhanced layout
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

.json-container {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #dee2e6;
    max-height: 500px;
    overflow-y: auto;
    font-family: monospace;
    font-size: 0.85rem;
}

.complete-data-info {
    background: linear-gradient(135deg, #e8f5e8 0%, #d4edd4 100%);
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #28a745;
    margin: 1rem 0;
}

.chatbot-ready {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #2196f3;
    margin: 1rem 0;
}

/* Sidebar styling */
.css-1d391kg {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.css-1d391kg .css-10trblm {
    color: white;
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

def get_complete_data_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    """Get statistics about complete data including MCID, deidentified medical, and raw pharmacy"""
    stats = {
        "mcid_fields_processed": 0,
        "medical_fields_processed": 0,
        "pharmacy_data_type": "raw",
        "medical_records": 0,
        "pharmacy_records": 0,
        "total_diagnosis_codes": 0,
        "total_ndc_codes": 0,
        "medical_dates_extracted": 0,
        "pharmacy_dates_extracted": 0,
        "llm_enhanced": False
    }
    
    try:
        # MCID data stats
        deidentified_mcid = safe_get(results, 'deidentified_data', {}).get('mcid', {})
        stats["mcid_fields_processed"] = deidentified_mcid.get('total_fields_processed', 0)
        
        # Medical data stats (deidentified)
        deidentified_medical = safe_get(results, 'deidentified_data', {}).get('medical', {})
        stats["medical_fields_processed"] = deidentified_medical.get('total_fields_processed', 0)
        
        medical_extraction = safe_get(results, 'structured_extractions', {}).get('medical', {})
        stats["medical_records"] = len(medical_extraction.get('hlth_srvc_records', []))
        stats["total_diagnosis_codes"] = medical_extraction.get('extraction_summary', {}).get('total_diagnosis_codes', 0)
        stats["medical_dates_extracted"] = medical_extraction.get('extraction_summary', {}).get('dates_extracted', 0)
        stats["llm_enhanced"] = medical_extraction.get('llm_enhanced', False)
        
        # Pharmacy data stats (raw - no field processing count)
        pharmacy_extraction = safe_get(results, 'structured_extractions', {}).get('pharmacy', {})
        stats["pharmacy_records"] = len(pharmacy_extraction.get('ndc_records', []))
        stats["total_ndc_codes"] = len(pharmacy_extraction.get('extraction_summary', {}).get('unique_ndc_codes', []))
        stats["pharmacy_dates_extracted"] = pharmacy_extraction.get('extraction_summary', {}).get('dates_extracted', 0)
        
    except Exception as e:
        pass
    
    return stats

# Initialize session state
initialize_session_state()

# Main Title
st.markdown('<h1 class="main-header">🏥 Enhanced Health Analysis Agent</h1>', unsafe_allow_html=True)

# Display import status
if not AGENT_AVAILABLE:
    st.markdown(f'<div class="status-error">❌ Failed to import Enhanced Health Agent: {import_error}</div>', unsafe_allow_html=True)
    st.stop()

# ENHANCED SIDEBAR CHATBOT WITH COMPLETE DATA ACCESS
with st.sidebar:
    if st.session_state.analysis_results and st.session_state.analysis_results.get("chatbot_ready", False) and st.session_state.chatbot_context:
        st.title("💬 Enhanced Medical Assistant")
        st.markdown("---")
        
        # Display complete data access info including MCID and raw pharmacy
        data_stats = get_complete_data_stats(st.session_state.analysis_results)
        st.markdown(f"""
        <div class="complete-data-info">
        <strong>📊 Complete Data Access (Enhanced v6.1):</strong><br>
        • MCID Fields: {data_stats['mcid_fields_processed']:,} (deidentified)<br>
        • Medical Fields: {data_stats['medical_fields_processed']:,} (deidentified)<br>
        • Pharmacy Data: Raw (no deidentification needed)<br>
        • Medical Records: {data_stats['medical_records']}<br>
        • Pharmacy Records: {data_stats['pharmacy_records']}<br>
        • Diagnosis Codes: {data_stats['total_diagnosis_codes']}<br>
        • NDC Codes: {data_stats['total_ndc_codes']}<br>
        • Medical Dates: {data_stats['medical_dates_extracted']}<br>
        • Pharmacy Dates: {data_stats['pharmacy_dates_extracted']}<br>
        • LLM Enhanced: {'✅' if data_stats['llm_enhanced'] else '❌'}
        </div>
        """, unsafe_allow_html=True)
        
        # Chat history at top
        chat_container = st.container()
        with chat_container:
            if st.session_state.chatbot_messages:
                for message in st.session_state.chatbot_messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
            else:
                st.markdown("""
                <div class="chatbot-ready">
                <strong>👋 Enhanced Medical Assistant Ready! (v6.0)</strong><br>
                I have access to the COMPLETE deidentified MCID, medical, and pharmacy data with LLM-enhanced meanings.<br><br>
                <strong>Ask me about:</strong><br>
                • Any specific medical codes with AI-generated explanations<br>
                • Medication details, NDC codes, AI-powered descriptions<br>
                • MCID member information and identifiers<br>
                • Dates (CLM_RCVD_DT, RX_FILLED_DT), timelines, service details<br>
                • Any field or value in the complete JSON data<br>
                • Medical history analysis with enhanced insights<br>
                • Heart attack risk factors (gets both LLM + ML analysis)<br>
                • LLM-generated code meanings and explanations
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input at bottom (always visible)
        st.markdown("---")
        user_question = st.chat_input("Ask about ANY data in the complete medical records...")
        
        # Handle chat input
        if user_question:
            # Add user message
            st.session_state.chatbot_messages.append({"role": "user", "content": user_question})
            
            # Get bot response using complete deidentified data
            try:
                with st.spinner("Processing with complete data access..."):
                    chatbot_response = st.session_state.agent.chat_with_data(
                        user_question, 
                        st.session_state.chatbot_context, 
                        st.session_state.chatbot_messages
                    )
                
                # Add assistant response
                st.session_state.chatbot_messages.append({"role": "assistant", "content": chatbot_response})
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing with complete data: {str(e)}")
        
        # Clear chat button at bottom
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chatbot_messages = []
            st.rerun()
    
    else:
        # Show placeholder when chatbot is not ready
        st.title("💬 Enhanced Medical Assistant v6.0")
        st.info("💤 Enhanced chatbot will be available after running health analysis")
        st.markdown("---")
        st.markdown("**Enhanced Features v6.0:**")
        st.markdown("• Complete deidentified data access (MCID + Medical + Pharmacy)")
        st.markdown("• Query ANY field in medical/pharmacy/MCID JSON")
        st.markdown("• LLM-enhanced code meanings and descriptions")
        st.markdown("• Date extraction (CLM_RCVD_DT, RX_FILLED_DT)")
        st.markdown("• Comprehensive nested data analysis") 
        st.markdown("• Heart attack prediction (LLM + ML model)")
        st.markdown("• Detailed medical code explanations")
        st.markdown("• Complete medication and diagnosis details with AI insights")

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
        
        # 2. RUN ENHANCED HEALTH ANALYSIS BUTTON
        submitted = st.form_submit_button(
            "🚀 Run Enhanced Health Analysis", 
            use_container_width=True,
            disabled=st.session_state.analysis_running
        )

# Analysis Status
if st.session_state.analysis_running:
    st.markdown('<div class="status-success">🔄 Enhanced health analysis workflow executing... Please wait.</div>', unsafe_allow_html=True)

# Run Enhanced Health Analysis
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
        # Initialize Enhanced Health Agent
        if st.session_state.agent is None:
            try:
                config = st.session_state.config or Config()
                st.session_state.agent = HealthAnalysisAgent(config)
                st.success("✅ Enhanced Health Analysis Agent initialized")
            except Exception as e:
                st.error(f"❌ Failed to initialize Enhanced Health Agent: {str(e)}")
                st.stop()
        
        st.session_state.analysis_running = True
        
        # Enhanced Progress tracking with detailed research agent-style updates
        progress_container = st.container()
        progress_bar = st.progress(0)
        status_text = st.empty()
        detailed_progress = st.empty()
        
        with st.spinner("🚀 Executing enhanced health analysis with deep research capabilities..."):
            try:
                # Enhanced progress updates with research agent-style details
                workflow_steps = [
                    {
                        "step": "Initializing enhanced MCP-compatible workflow...",
                        "details": "🔧 Setting up LangGraph nodes, initializing API integrators, preparing comprehensive data processors",
                        "progress": 8
                    },
                    {
                        "step": "Fetching MCP-compatible data sources...",
                        "details": "📡 Connecting to MCP server, retrieving MCID data, medical claims, pharmacy records, authentication tokens",
                        "progress": 16
                    },
                    {
                        "step": "Comprehensive nested JSON deidentification...",
                        "details": "🔒 Processing MCID fields, medical data structures, pharmacy records with PII pattern detection and removal",
                        "progress": 24
                    },
                    {
                        "step": "LLM-enhanced medical information extraction...",
                        "details": "🤖 Extracting service codes, diagnosis codes, CLM_RCVD_DT dates, generating AI-powered code meanings",
                        "progress": 32
                    },
                    {
                        "step": "LLM-enhanced pharmacy data processing...",
                        "details": "💊 Processing NDC codes, medication labels, RX_FILLED_DT dates, generating AI descriptions for medications",
                        "progress": 40
                    },
                    {
                        "step": "Comprehensive health entity extraction...",
                        "details": "🎯 Analyzing diabetes indicators, smoking status, blood pressure, chronic conditions from all data sources",
                        "progress": 48
                    },
                    {
                        "step": "Advanced health trajectory analysis...",
                        "details": "📈 Synthesizing medical history, medication patterns, LLM-enhanced insights for comprehensive health assessment",
                        "progress": 56
                    },
                    {
                        "step": "Generating comprehensive clinical summary...",
                        "details": "📋 Creating executive summary with actionable insights, risk factors, recommendations based on complete analysis",
                        "progress": 64
                    },
                    {
                        "step": "Enhanced heart attack risk prediction...",
                        "details": "❤️ Running FastAPI ML model, extracting risk features, preparing dual LLM+ML analysis capabilities",
                        "progress": 72
                    },
                    {
                        "step": "Initializing comprehensive chatbot...",
                        "details": "💬 Loading complete deidentified context, MCID data, LLM meanings, preparing heart attack special handling",
                        "progress": 80
                    },
                    {
                        "step": "Finalizing enhanced analysis pipeline...",
                        "details": "✅ Completing workflow, validating all data processing, preparing comprehensive results with full context access",
                        "progress": 88
                    },
                    {
                        "step": "Analysis completed successfully!",
                        "details": "🎉 All enhanced features ready: MCID + Medical + Pharmacy deidentification, LLM meanings, dates, dual heart attack analysis",
                        "progress": 100
                    }
                ]
                
                for i, step_info in enumerate(workflow_steps):
                    status_text.text(f"🔄 {step_info['step']}")
                    detailed_progress.markdown(f"""
                    <div style="background: #f0f8ff; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; border-left: 3px solid #007acc;">
                    <strong>Step {i+1}/12:</strong> {step_info['step']}<br>
                    <small style="color: #555;">{step_info['details']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    progress_bar.progress(step_info['progress'])
                    time.sleep(0.8)  # Longer pause to show research agent-style processing
                
                # Execute enhanced analysis
                results = st.session_state.agent.run_analysis(patient_data)
                
                if results.get("success", False):
                    progress_bar.progress(100)
                    status_text.text("✅ Enhanced analysis completed successfully!")
                    detailed_progress.markdown("""
                    <div style="background: #d4edda; padding: 1rem; border-radius: 5px; margin: 1rem 0; border-left: 4px solid #28a745;">
                    <strong>🎉 Enhanced Health Analysis Complete!</strong><br>
                    All advanced features are now ready: MCID deidentification, LLM-enhanced code meanings, date extraction, and dual heart attack prediction analysis.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.session_state.analysis_results = results
                    st.session_state.chatbot_context = results.get("chatbot_context", {})
                    
                    # Display enhanced success information
                    data_stats = get_complete_data_stats(results)
                    
                    total_deidentified_fields = data_stats['mcid_fields_processed'] + data_stats['medical_fields_processed']
                    total_dates = data_stats['medical_dates_extracted'] + data_stats['pharmacy_dates_extracted']
                    
                    st.markdown(f"""
                    <div class="status-success">
                    ✅ Enhanced health analysis v6.1 completed successfully!<br>
                    🆔 MCID fields processed: {data_stats['mcid_fields_processed']:,} (deidentified)<br>
                    🔒 Medical deidentification: {data_stats['medical_fields_processed']:,} fields processed<br>
                    💊 Pharmacy data: Raw format preserved (no deidentification needed)<br>
                    📅 Dates extracted: {total_dates} (CLM_RCVD_DT + RX_FILLED_DT)<br>
                    🤖 LLM-enhanced: {'✅ Code meanings generated' if data_stats['llm_enhanced'] else '❌ Basic extraction only'}<br>
                    📊 Complete data ready for enhanced chatbot access
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Ensure enhanced chatbot is properly loaded
                    if results.get("chatbot_ready", False) and st.session_state.chatbot_context:
                        st.markdown(f"""
                        <div class="chatbot-ready">
                        💬 <strong>Enhanced Medical Assistant v6.1 is now ready!</strong><br>
                        The chatbot has complete access to ALL enhanced data:<br>
                        🆔 {data_stats['mcid_fields_processed']:,} MCID fields with member information (deidentified)<br>
                        📋 {data_stats['medical_records']} medical records with {data_stats['total_diagnosis_codes']} diagnosis codes + LLM meanings (deidentified)<br>
                        💊 {data_stats['pharmacy_records']} pharmacy records with {data_stats['total_ndc_codes']} NDC codes + AI descriptions (raw data)<br>
                        📅 {total_dates} dates extracted from claims and prescriptions<br>
                        🔍 Can answer questions about ANY field, code, or value in the complete data<br>
                        ❤️ Heart attack risk prediction with dual LLM + ML model analysis<br>
                        🤖 AI-powered code explanations and medication descriptions<br>
                        🔒 Privacy: MCID + Medical deidentified, Pharmacy kept raw as requested
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Force page refresh to open sidebar
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning("⚠️ Enhanced chatbot initialization incomplete.")
                else:
                    st.session_state.analysis_results = results
                    st.warning("⚠️ Enhanced analysis completed with some errors.")
                
            except Exception as e:
                st.error(f"❌ Enhanced analysis failed: {str(e)}")
                st.session_state.analysis_results = {
                    "success": False,
                    "error": str(e),
                    "patient_data": patient_data,
                    "errors": [str(e)]
                }
            finally:
                st.session_state.analysis_running = False

# Display Enhanced Results if Available
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    # Show errors if any
    errors = safe_get(results, 'errors', [])
    if errors:
        st.markdown('<div class="status-error">❌ Some analysis errors occurred</div>', unsafe_allow_html=True)

    # Display complete data access information including MCID and raw pharmacy
    if results.get("success", False):
        data_stats = get_complete_data_stats(results)
        total_deidentified_fields = data_stats['mcid_fields_processed'] + data_stats['medical_fields_processed']
        total_dates = data_stats['medical_dates_extracted'] + data_stats['pharmacy_dates_extracted']
        
        st.markdown(f"""
        <div class="complete-data-info">
        <strong>📊 Complete Enhanced Data Processing Results v6.1:</strong><br>
        • Total MCID Fields Processed: {data_stats['mcid_fields_processed']:,} (deidentified)<br>
        • Total Medical Fields Processed: {data_stats['medical_fields_processed']:,} (deidentified)<br>
        • Pharmacy Data: Raw format (no deidentification applied)<br>
        • Medical Records Extracted: {data_stats['medical_records']} (with LLM meanings)<br>
        • Pharmacy Records Extracted: {data_stats['pharmacy_records']} (with AI descriptions from raw data)<br>
        • Diagnosis Codes Found: {data_stats['total_diagnosis_codes']}<br>
        • NDC Codes Found: {data_stats['total_ndc_codes']}<br>
        • Medical Dates Extracted: {data_stats['medical_dates_extracted']} (CLM_RCVD_DT)<br>
        • Pharmacy Dates Extracted: {data_stats['pharmacy_dates_extracted']} (RX_FILLED_DT)<br>
        • Deidentification Applied: MCID + Medical only (Pharmacy kept raw)<br>
        • LLM Enhancement: {'✅ Active' if data_stats['llm_enhanced'] else '❌ Not Available'}<br>
        • Chatbot Data Access: Complete (MCID + Medical Deidentified + Pharmacy Raw) + Heart Attack Special Handling
        </div>
        """, unsafe_allow_html=True)

    # 3. COMPLETE DEIDENTIFIED DATA BUTTON (Enhanced with MCID and raw pharmacy)
    if st.button("📊 Complete Data (MCID + Medical Deidentified + Pharmacy Raw)", use_container_width=True):
        st.markdown("""
        <div class="section-box">
            <div class="section-title">📊 Complete Data (MCID Deidentified + Medical Deidentified + Pharmacy Raw)</div>
        </div>
        """, unsafe_allow_html=True)
        
        deidentified_data = safe_get(results, 'deidentified_data', {})
        
        if deidentified_data:
            tab1, tab2, tab3 = st.tabs(["🆔 MCID Data (Deidentified)", "🏥 Medical Data (Deidentified)", "💊 Pharmacy Data (Raw)"])
            
            with tab1:
                mcid_data = safe_get(deidentified_data, 'mcid', {})
                if mcid_data:
                    st.markdown("**Complete Deidentified MCID JSON Structure:**")
                    
                    # Show MCID processing stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MCID Fields Processed", mcid_data.get('total_fields_processed', 0))
                    with col2:
                        st.metric("Deidentification Level", mcid_data.get('deidentification_level', 'standard'))
                    with col3:
                        st.metric("Processing Status", "✅ Deidentified")
                    
                    st.markdown('<div class="json-container">', unsafe_allow_html=True)
                    st.json(mcid_data)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.download_button(
                        "📥 Download Complete MCID Data JSON",
                        safe_json_dumps(mcid_data),
                        f"complete_mcid_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    st.warning("No complete MCID data available")
            
            with tab2:
                medical_data = safe_get(deidentified_data, 'medical', {})
                if medical_data:
                    st.markdown("**Complete Deidentified Medical JSON Structure:**")
                    
                    # Show processing stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Fields Processed", medical_data.get('total_fields_processed', 0))
                    with col2:
                        st.metric("Deidentification Level", medical_data.get('deidentification_level', 'standard'))
                    with col3:
                        st.metric("Patient Age", medical_data.get('src_mbr_age', 'unknown'))
                    
                    st.markdown('<div class="json-container">', unsafe_allow_html=True)
                    st.json(medical_data)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.download_button(
                        "📥 Download Complete Medical Data JSON",
                        safe_json_dumps(medical_data),
                        f"complete_medical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    st.warning("No complete medical data available")
            
            with tab3:
                pharmacy_data = safe_get(deidentified_data, 'pharmacy', {})
                if pharmacy_data:
                    st.markdown("**Complete Raw Pharmacy JSON Structure (No Deidentification Applied):**")
                    
                    # Show raw pharmacy info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Data Type", "Raw Pharmacy Data")
                    with col2:
                        st.metric("Deidentification", "❌ None Applied")
                    
                    st.info("💡 Pharmacy data is kept in raw format as requested - no deidentification processing applied.")
                    
                    st.markdown('<div class="json-container">', unsafe_allow_html=True)
                    st.json(pharmacy_data)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.download_button(
                        "📥 Download Complete Raw Pharmacy Data JSON",
                        safe_json_dumps(pharmacy_data),
                        f"complete_raw_pharmacy_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    st.warning("No complete pharmacy data available")

    # 4. ENHANCED DATA EXTRACTION WITH LLM MEANINGS BUTTON
    if st.button("🔍 Enhanced Data Extraction (LLM Meanings + Dates)", use_container_width=True):
        st.markdown("""
        <div class="section-box">
            <div class="section-title">🔍 Enhanced Medical/Pharmacy Data Extraction with AI-Powered Meanings</div>
        </div>
        """, unsafe_allow_html=True)
        
        structured_extractions = safe_get(results, 'structured_extractions', {})
        
        if structured_extractions:
            tab1, tab2 = st.tabs(["🏥 Enhanced Medical Extraction", "💊 Enhanced Pharmacy Extraction"])
            
            with tab1:
                medical_extraction = safe_get(structured_extractions, 'medical', {})
                if medical_extraction and not medical_extraction.get('error'):
                    extraction_summary = safe_get(medical_extraction, 'extraction_summary', {})
                    
                    # Enhanced metrics including dates and LLM status
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
                            <h3>{extraction_summary.get('dates_extracted', 0)}</h3>
                            <p>Dates Extracted (CLM_RCVD_DT)</p>
                        </div>
                        <div class="metric-card">
                            <h3>{'✅' if medical_extraction.get('llm_enhanced', False) else '❌'}</h3>
                            <p>LLM Enhanced</p>
                        </div>
                        <div class="metric-card">
                            <h3>{len(extraction_summary.get('unique_service_codes', []))}</h3>
                            <p>Unique Service Codes</p>
                        </div>
                        <div class="metric-card">
                            <h3>{len(extraction_summary.get('unique_diagnosis_codes', []))}</h3>
                            <p>Unique Diagnosis Codes</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    hlth_srvc_records = safe_get(medical_extraction, 'hlth_srvc_records', [])
                    if hlth_srvc_records:
                        st.markdown("**📋 All Enhanced Medical Records with LLM Meanings:**")
                        for i, record in enumerate(hlth_srvc_records, 1):
                            service_code = record.get('hlth_srvc_cd', 'N/A')
                            service_meaning = record.get('hlth_srvc_cd_meaning', 'No meaning available')
                            claim_date = record.get('claim_received_date', 'No date available')
                            
                            with st.expander(f"Medical Record {i} - Service: {service_code} | Date: {claim_date}"):
                                st.write(f"**Service Code:** `{service_code}`")
                                st.write(f"**🤖 AI-Generated Meaning:** {service_meaning}")
                                st.write(f"**📅 Claim Received Date:** `{claim_date}`")
                                st.write(f"**Data Path:** `{record.get('data_path', 'N/A')}`")
                                
                                diagnosis_codes = record.get('diagnosis_codes', [])
                                if diagnosis_codes:
                                    st.write("**Diagnosis Codes with LLM Meanings:**")
                                    for idx, diag in enumerate(diagnosis_codes, 1):
                                        source_info = f" (from {diag.get('source', 'individual field')})" if diag.get('source') else ""
                                        llm_meaning = diag.get('llm_meaning', 'No meaning available')
                                        st.write(f"  {idx}. **Code:** `{diag.get('code', 'N/A')}`{source_info}")
                                        st.write(f"      **🤖 AI Meaning:** {llm_meaning}")
                else:
                    st.warning("No enhanced medical extraction data available")
            
            with tab2:
                pharmacy_extraction = safe_get(structured_extractions, 'pharmacy', {})
                if pharmacy_extraction and not pharmacy_extraction.get('error'):
                    extraction_summary = safe_get(pharmacy_extraction, 'extraction_summary', {})
                    
                    # Enhanced pharmacy metrics including dates and LLM status
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
                        <div class="metric-card">
                            <h3>{extraction_summary.get('dates_extracted', 0)}</h3>
                            <p>Dates Extracted (RX_FILLED_DT)</p>
                        </div>
                        <div class="metric-card">
                            <h3>{'✅' if pharmacy_extraction.get('llm_enhanced', False) else '❌'}</h3>
                            <p>LLM Enhanced</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    ndc_records = safe_get(pharmacy_extraction, 'ndc_records', [])
                    if ndc_records:
                        st.markdown("**💊 All Enhanced Pharmacy Records with AI Descriptions:**")
                        for i, record in enumerate(ndc_records, 1):
                            ndc_code = record.get('ndc', 'N/A')
                            label_name = record.get('lbl_nm', 'N/A')
                            ndc_meaning = record.get('ndc_llm_meaning', 'No meaning available')
                            label_description = record.get('lbl_nm_llm_description', 'No description available')
                            rx_date = record.get('prescription_filled_date', 'No date available')
                            
                            with st.expander(f"Pharmacy Record {i} - {label_name} | Date: {rx_date}"):
                                st.write(f"**NDC Code:** `{ndc_code}`")
                                st.write(f"**🤖 AI-Generated NDC Meaning:** {ndc_meaning}")
                                st.write(f"**Label Name:** `{label_name}`")
                                st.write(f"**🤖 AI-Generated Description:** {label_description}")
                                st.write(f"**📅 Prescription Filled Date:** `{rx_date}`")
                                st.write(f"**Data Path:** `{record.get('data_path', 'N/A')}`")
                                
                                # Show additional fields if available
                                additional_fields = {k: v for k, v in record.items() 
                                                   if k not in ['ndc', 'lbl_nm', 'data_path', 'ndc_llm_meaning', 'lbl_nm_llm_description', 'prescription_filled_date']}
                                if additional_fields:
                                    st.write("**Additional Pharmacy Fields:**")
                                    for field, value in additional_fields.items():
                                        st.write(f"  • **{field}:** `{value}`")
                else:
                    st.warning("No enhanced pharmacy extraction data available")

    # 5. COMPREHENSIVE ENTITY EXTRACTION BUTTON
    if st.button("🎯 Comprehensive Entity Extraction", use_container_width=True):
        st.markdown("""
        <div class="section-box">
            <div class="section-title">🎯 Comprehensive Health Entity Extraction</div>
        </div>
        """, unsafe_allow_html=True)
        
        entity_extraction = safe_get(results, 'entity_extraction', {})
        if entity_extraction:
            # Enhanced entity cards
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
            
            # Enhanced medical conditions
            medical_conditions = safe_get(entity_extraction, 'medical_conditions', [])
            chronic_conditions = safe_get(entity_extraction, 'chronic_conditions', [])
            risk_factors = safe_get(entity_extraction, 'risk_factors', [])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if medical_conditions:
                    st.markdown("**🏥 Medical Conditions Identified:**")
                    for condition in medical_conditions:
                        st.write(f"• {condition}")
            
            with col2:
                if chronic_conditions:
                    st.markdown("**🔄 Chronic Conditions:**")
                    for condition in chronic_conditions:
                        st.write(f"• {condition}")
            
            with col3:
                if risk_factors:
                    st.markdown("**⚠️ Risk Factors:**")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
            
            # Enhanced medications identified
            medications_identified = safe_get(entity_extraction, 'medications_identified', [])
            if medications_identified:
                st.markdown("**💊 Medications Identified:**")
                for med in medications_identified:
                    additional_fields = med.get('additional_fields', {})
                    additional_info = f" | {additional_fields}" if additional_fields else ""
                    st.write(f"• **{med.get('label_name', 'N/A')}** (NDC: {med.get('ndc', 'N/A')}){additional_info}")

    # 6. HEALTH TRAJECTORY BUTTON
    if st.button("📈 Comprehensive Health Trajectory", use_container_width=True):
        st.markdown("""
        <div class="section-box">
            <div class="section-title">📈 Comprehensive Health Trajectory Analysis</div>
        </div>
        """, unsafe_allow_html=True)
        
        health_trajectory = safe_get(results, 'health_trajectory', '')
        if health_trajectory:
            st.markdown(health_trajectory)
        else:
            st.warning("Comprehensive health trajectory analysis not available")

    # 7. FINAL SUMMARY BUTTON
    if st.button("📋 Comprehensive Final Summary", use_container_width=True):
        st.markdown("""
        <div class="section-box">
            <div class="section-title">📋 Comprehensive Clinical Summary</div>
        </div>
        """, unsafe_allow_html=True)
        
        final_summary = safe_get(results, 'final_summary', '')
        if final_summary:
            st.markdown(final_summary)
        else:
            st.warning("Comprehensive final summary not available")

    # 8. ENHANCED HEART ATTACK RISK PREDICTION BUTTON
    if st.button("❤️ Enhanced Heart Attack Risk Prediction", use_container_width=True):
        st.markdown("""
        <div class="section-box">
            <div class="section-title">❤️ Enhanced Heart Attack Risk Assessment</div>
        </div>
        """, unsafe_allow_html=True)
        
        heart_attack_prediction = safe_get(results, 'heart_attack_prediction', {})
        if heart_attack_prediction and not heart_attack_prediction.get('error'):
            # Display enhanced prediction format
            combined_display = heart_attack_prediction.get("combined_display", "Heart Disease Risk: Not available")
            risk_category = heart_attack_prediction.get("risk_category", "Unknown")
            prediction_method = heart_attack_prediction.get("prediction_method", "unknown")
            
            # Enhanced display with more information
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 2rem; border-radius: 10px; border: 1px solid #dee2e6; margin: 1rem 0; text-align: center;">
                <h3 style="color: #2c3e50; margin-bottom: 1rem;">Enhanced Heart Attack Risk Prediction</h3>
                <h4 style="color: #495057; font-weight: 600;">{combined_display}</h4>
                <p style="color: #6c757d; margin-top: 1rem; font-size: 0.9rem;">
                    Enhanced Prediction Method: {prediction_method}<br>
                    FastAPI Server: {heart_attack_prediction.get('fastapi_server_url', 'Unknown')}<br>
                    Model Enhanced: {heart_attack_prediction.get('model_enhanced', False)}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show enhanced features used
            enhanced_features = heart_attack_prediction.get("enhanced_features_used", {})
            if enhanced_features:
                st.markdown("**🎯 Enhanced Features Used for Prediction:**")
                for feature, value in enhanced_features.items():
                    st.write(f"• **{feature}:** {value}")
            
        else:
            error_msg = heart_attack_prediction.get('error', 'Enhanced heart attack prediction not available')
            st.error(f"❌ Enhanced FastAPI Server Error: {error_msg}")
            
            # Show enhanced connection info for debugging
            st.info(f"💡 Expected Enhanced FastAPI Server: {st.session_state.config.heart_attack_api_url if st.session_state.config else 'http://localhost:8080'}")
            st.info("💡 Make sure enhanced FastAPI server is running with heart attack prediction model")

# Footer with enhanced information including all v6.1 features
if st.session_state.analysis_results and st.session_state.analysis_results.get("success", False):
    st.markdown("---")
    data_stats = get_complete_data_stats(st.session_state.analysis_results)
    total_deidentified_fields = data_stats['mcid_fields_processed'] + data_stats['medical_fields_processed']
    total_dates = data_stats['medical_dates_extracted'] + data_stats['pharmacy_dates_extracted']
    
    st.markdown(f"""
    <div class="complete-data-info">
    <strong>🔍 Enhanced Analysis Complete v6.1:</strong><br>
    The Enhanced Medical Assistant in the sidebar has complete access to ALL enhanced data including:<br>
    🆔 MCID Data: {data_stats['mcid_fields_processed']:,} fields processed (deidentified)<br>
    📋 Medical Data: {data_stats['medical_records']} records with {data_stats['total_diagnosis_codes']} diagnosis codes + AI meanings (deidentified)<br>
    💊 Pharmacy Data: {data_stats['pharmacy_records']} records with {data_stats['total_ndc_codes']} NDC codes + AI descriptions (raw format)<br>
    📅 Date Extraction: {total_dates} dates from CLM_RCVD_DT and RX_FILLED_DT fields<br>
    🤖 LLM Enhancement: {'✅ Code meanings and descriptions generated' if data_stats['llm_enhanced'] else '❌ Basic extraction only'}<br>
    ❤️ Heart Attack Prediction: Dual LLM + ML model analysis available<br>
    🔒 Privacy Applied: MCID + Medical deidentified, Pharmacy kept raw as requested<br>
    🔍 Ask specific questions about any codes, medications, dates, or fields in the complete medical records.<br>
    📊 No data truncation - complete JSON structures (MCID deidentified + Medical deidentified + Pharmacy raw) available for comprehensive queries.
    </div>
    """, unsafe_allow_html=True)
