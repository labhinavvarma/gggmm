# Configure Streamlit page FIRST - before any other Streamlit commands
import streamlit as st

# Determine sidebar state based on chatbot readiness
if 'analysis_results' in st.session_state and st.session_state.get('analysis_results') and st.session_state.analysis_results.get("chatbot_ready", False):
    sidebar_state = "expanded"
else:
    sidebar_state = "collapsed"

st.set_page_config(
    page_title="⚡ Enhanced Health Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state=sidebar_state
)

# Core imports with proper error handling
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import os
import logging
import traceback
from typing import Dict, Any, Optional, Union, List, Tuple
import io
import base64
import re
import numpy as np

# Matplotlib imports with safety
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for stability
import matplotlib.pyplot as plt

# Plotly imports with error handling
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    make_subplots = None

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Set up enhanced logging to catch all errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('enhanced_health_app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# ===== ENHANCED UTILITY FUNCTIONS =====

def safe_get(data: Union[Dict[str, Any], Any], key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary or object"""
    try:
        if data is None:
            logger.warning(f"safe_get: data is None for key '{key}'")
            return default
        if isinstance(data, dict):
            return data.get(key, default)
        elif hasattr(data, key):
            return getattr(data, key, default)
        else:
            logger.warning(f"safe_get: data type {type(data)} doesn't support key '{key}'")
            return default
    except Exception as e:
        logger.error(f"Error in safe_get for key '{key}': {e}")
        return default

def safe_execute(func, *args, **kwargs) -> Tuple[bool, Any]:
    """Safely execute a function and return success status and result"""
    try:
        logger.info(f"Executing function: {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"Function {func.__name__} executed successfully")
        return True, result
    except Exception as e:
        error_msg = f"Error executing {func.__name__}: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return False, error_msg

def safe_str(value: Any, default: str = "Unknown") -> str:
    """Safely convert value to string"""
    try:
        if value is None:
            return default
        return str(value)
    except Exception as e:
        logger.warning(f"Error converting to string: {e}")
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to integer"""
    try:
        if value is None:
            return default
        return int(float(str(value)))
    except (ValueError, TypeError) as e:
        logger.warning(f"Error converting to int: {e}")
        return default

def safe_len(obj: Any, default: int = 0) -> int:
    """Safely get length of an object"""
    try:
        if obj is None:
            return default
        return len(obj)
    except (TypeError, AttributeError) as e:
        logger.warning(f"Error getting length: {e}")
        return default

def log_error(error_msg: str, context: str = ""):
    """Log error to session state and logger"""
    timestamp = datetime.now().isoformat()
    error_entry = {
        'timestamp': timestamp,
        'message': error_msg,
        'context': context
    }
    
    if 'error_log' not in st.session_state:
        st.session_state.error_log = []
    
    st.session_state.error_log.append(error_entry)
    logger.error(f"[{context}] {error_msg}")

# Import the Enhanced health analysis agent with error handling
AGENT_AVAILABLE = False
import_error = None
EnhancedHealthAnalysisAgent = None
EnhancedConfig = None

try:
    from health_agent_core_enhanced import EnhancedHealthAnalysisAgent, EnhancedConfig
    AGENT_AVAILABLE = True
    logger.info("✅ Enhanced Health Analysis Agent imported successfully")
except ImportError as e:
    AGENT_AVAILABLE = False
    import_error = str(e)
    logger.error(f"❌ Failed to import Enhanced Health Analysis Agent: {e}")

# Enhanced CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.main-header {
    font-size: 3.2rem;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.section-box {
    background: white;
    padding: 1.8rem;
    border-radius: 15px;
    border: 1px solid #e9ecef;
    margin: 1.2rem 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}

.section-title {
    font-size: 1.4rem;
    color: #2c3e50;
    font-weight: 600;
    margin-bottom: 1rem;
    border-bottom: 3px solid #3498db;
    padding-bottom: 0.6rem;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.2rem;
    margin: 1.2rem 0;
}

.metric-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #dee2e6;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.status-success {
    background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
    color: #2e7d32;
    padding: 1rem;
    border-radius: 12px;
    border: 2px solid #4caf50;
    margin: 1rem 0;
    font-weight: 500;
}

.status-error {
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    color: #c62828;
    padding: 1rem;
    border-radius: 12px;
    border: 2px solid #f44336;
    margin: 1rem 0;
    font-weight: 500;
}

/* Enhanced table styling */
.dataframe {
    border: none !important;
}

.dataframe th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px !important;
    text-align: center !important;
}

.dataframe td {
    padding: 10px !important;
    text-align: center !important;
    border-bottom: 1px solid #e9ecef !important;
}

.dataframe tr:hover {
    background-color: #f8f9fa !important;
}

/* Button styling */
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 25px rgba(40, 167, 69, 0.4) !important;
    transition: all 0.3s ease !important;
}

.stButton button[kind="primary"]:hover {
    background: linear-gradient(135deg, #218838 0%, #1abc9c 100%) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 35px rgba(40, 167, 69, 0.5) !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables for enhanced processing"""
    defaults = {
        'analysis_results': None,
        'analysis_running': False,
        'agent': None,
        'config': None,
        'chatbot_messages': [],
        'chatbot_context': None,
        'show_claims_data': False,
        'show_batch_extraction': False,
        'show_entity_extraction': False,
        'show_enhanced_trajectory': False,
        'show_heart_attack': False,
        'error_log': [],
        'last_update': datetime.now().isoformat()
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def calculate_age(birth_date):
    """Calculate age from birth date"""
    if not birth_date:
        return None
    
    try:
        today = datetime.now().date()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return age
    except Exception as e:
        logger.error(f"Age calculation error: {e}")
        return None

def validate_patient_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate patient data"""
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
        field_value = safe_get(data, field)
        if not field_value:
            errors.append(f"{display_name} is required")
        elif field == 'ssn' and safe_len(safe_str(field_value)) < 9:
            errors.append("SSN must be at least 9 digits")
        elif field == 'zip_code' and safe_len(safe_str(field_value)) < 5:
            errors.append("Zip code must be at least 5 digits")
    
    return len(errors) == 0, errors

def run_analysis_workflow(patient_data: Dict[str, Any]):
    """Run the analysis workflow with enhanced error handling and logging"""
    if not AGENT_AVAILABLE:
        error_msg = "Enhanced Health Analysis Agent not available"
        log_error(error_msg, "workflow_init")
        st.error(f"❌ {error_msg}")
        return None
        
    try:
        logger.info("Starting analysis workflow...")
        
        # Initialize agent and config
        config = EnhancedConfig()
        agent = EnhancedHealthAnalysisAgent(config)
        
        # Store in session state
        st.session_state.agent = agent
        st.session_state.config = config
        
        logger.info("Agent initialized successfully, starting analysis...")
        
        # Run analysis with proper error handling
        with st.spinner("🔬 Running enhanced healthcare analysis..."):
            success, results = safe_execute(agent.run_enhanced_analysis, patient_data)
            
            if success:
                logger.info("Analysis completed successfully")
                st.session_state.analysis_results = results
                st.session_state.chatbot_context = safe_get(results, 'chatbot_context')
                return results
            else:
                error_msg = f"Analysis execution failed: {results}"
                log_error(error_msg, "workflow_execution")
                st.error(f"❌ {error_msg}")
                return None
                
    except Exception as e:
        error_msg = f"Analysis workflow error: {str(e)}"
        log_error(error_msg, "workflow_exception")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        st.error(f"❌ {error_msg}")
        return None

def display_claims_data_viewer():
    """Display claims data viewer with only JSON views for Medical, Pharmacy, and MCID"""
    st.markdown("""
    <div class="section-box">
        <div class="section-title">📊 Claims Data</div>
    </div>
    """, unsafe_allow_html=True)
    
    analysis_results = st.session_state.get('analysis_results')
    if not analysis_results:
        st.warning("No analysis results available")
        return
    
    deidentified_data = safe_get(analysis_results, 'deidentified_data', {})
    api_outputs = safe_get(analysis_results, 'api_outputs', {})
    
    if deidentified_data or api_outputs:
        # Three tabs as requested: Medical Claims, Pharmacy Claims, MCID
        tab1, tab2, tab3 = st.tabs(["🏥 Medical Claims", "💊 Pharmacy Claims", "🆔 MCID"])
        
        with tab1:
            medical_data = safe_get(deidentified_data, 'medical', {})
            if medical_data and not safe_get(medical_data, 'error'):
                st.json(medical_data)
            else:
                error_msg = safe_get(medical_data, 'error', 'No medical claims data available')
                st.warning(f"⚠️ {error_msg}")
        
        with tab2:
            pharmacy_data = safe_get(deidentified_data, 'pharmacy', {})
            if pharmacy_data and not safe_get(pharmacy_data, 'error'):
                st.json(pharmacy_data)
            else:
                error_msg = safe_get(pharmacy_data, 'error', 'No pharmacy claims data available')
                st.warning(f"⚠️ {error_msg}")
        
        with tab3:
            mcid_data = safe_get(api_outputs, 'mcid', {})
            if mcid_data:
                st.json(mcid_data)
            else:
                st.warning("⚠️ No MCID data available")
    else:
        st.error("❌ No claims data available for display")

def display_batch_extraction_tabular():
    """Display batch extraction results in tabular format with unique codes and meanings"""
    st.markdown("""
    <div class="section-box">
        <div class="section-title">🚀 Batch Code Processing Results</div>
    </div>
    """, unsafe_allow_html=True)
    
    analysis_results = st.session_state.get('analysis_results')
    if not analysis_results:
        st.warning("No analysis results available")
        return
    
    structured_extractions = safe_get(analysis_results, 'structured_extractions', {})
    
    if structured_extractions:
        tab1, tab2 = st.tabs(["🏥 Medical Extraction", "💊 Pharmacy Extraction"])
        
        with tab1:
            medical_extraction = safe_get(structured_extractions, 'medical', {})
            if medical_extraction:
                st.markdown("### 📊 Medical Extraction Summary")
                
                # Create tabular display for the 4 main attributes
                batch_stats = safe_get(medical_extraction, 'batch_stats', {})
                extraction_summary = safe_get(medical_extraction, 'extraction_summary', {})
                
                # Main metrics table
                metrics_data = {
                    'Metric': [
                        'Total Health Service Records',
                        'Total Diagnosis Codes', 
                        'API Calls Made',
                        'Individual Calls Saved'
                    ],
                    'Value': [
                        safe_int(extraction_summary.get('total_hlth_srvc_records', 0)),
                        safe_int(extraction_summary.get('total_diagnosis_codes', 0)),
                        safe_int(batch_stats.get('api_calls_made', 0)),
                        safe_int(batch_stats.get('individual_calls_saved', 0))
                    ],
                    'Processing Time (seconds)': [
                        safe_get(batch_stats, 'processing_time_seconds', 'N/A'),
                        'N/A',
                        'N/A', 
                        'N/A'
                    ],
                    'Enhancement Status': [
                        'Enhanced' if medical_extraction.get('enhanced_analysis', False) else 'Standard',
                        'Enhanced' if medical_extraction.get('enhanced_analysis', False) else 'Standard',
                        'Batch Processing',
                        'Efficiency Optimization'
                    ]
                }
                
                df_metrics = pd.DataFrame(metrics_data)
                st.dataframe(df_metrics, use_container_width=True, hide_index=True)
                
                # Unique codes with meanings
                st.markdown("### 🔍 Unique Codes with Meanings")
                
                code_meanings = safe_get(medical_extraction, 'code_meanings', {})
                service_meanings = safe_get(code_meanings, 'service_code_meanings', {})
                diagnosis_meanings = safe_get(code_meanings, 'diagnosis_code_meanings', {})
                
                if service_meanings:
                    st.markdown("**Service Codes:**")
                    service_data = []
                    for code, meaning in service_meanings.items():
                        service_data.append({
                            'Code': code,
                            'Type': 'Service Code',
                            'Meaning': meaning[:100] + '...' if len(meaning) > 100 else meaning
                        })
                    
                    if service_data:
                        df_service = pd.DataFrame(service_data)
                        st.dataframe(df_service, use_container_width=True, hide_index=True)
                
                if diagnosis_meanings:
                    st.markdown("**Diagnosis Codes:**")
                    diagnosis_data = []
                    for code, meaning in diagnosis_meanings.items():
                        diagnosis_data.append({
                            'Code': code,
                            'Type': 'Diagnosis Code',
                            'Meaning': meaning[:100] + '...' if len(meaning) > 100 else meaning
                        })
                    
                    if diagnosis_data:
                        df_diagnosis = pd.DataFrame(diagnosis_data)
                        st.dataframe(df_diagnosis, use_container_width=True, hide_index=True)
                
                if not service_meanings and not diagnosis_meanings:
                    st.info("No code meanings available. Codes processed but meanings not generated.")
            else:
                st.warning("No medical extraction data available")
        
        with tab2:
            pharmacy_extraction = safe_get(structured_extractions, 'pharmacy', {})
            if pharmacy_extraction:
                st.markdown("### 📊 Pharmacy Extraction Summary")
                
                # Create tabular display for the 4 main attributes
                batch_stats = safe_get(pharmacy_extraction, 'batch_stats', {})
                extraction_summary = safe_get(pharmacy_extraction, 'extraction_summary', {})
                
                # Main metrics table
                metrics_data = {
                    'Metric': [
                        'Total NDC Records',
                        'Unique NDC Codes',
                        'API Calls Made', 
                        'Individual Calls Saved'
                    ],
                    'Value': [
                        safe_int(extraction_summary.get('total_ndc_records', 0)),
                        safe_len(extraction_summary.get('unique_ndc_codes', [])),
                        safe_int(batch_stats.get('api_calls_made', 0)),
                        safe_int(batch_stats.get('individual_calls_saved', 0))
                    ],
                    'Processing Time (seconds)': [
                        safe_get(batch_stats, 'processing_time_seconds', 'N/A'),
                        'N/A',
                        'N/A',
                        'N/A'
                    ],
                    'Enhancement Status': [
                        'Enhanced' if pharmacy_extraction.get('enhanced_analysis', False) else 'Standard',
                        'Enhanced' if pharmacy_extraction.get('enhanced_analysis', False) else 'Standard', 
                        'Batch Processing',
                        'Efficiency Optimization'
                    ]
                }
                
                df_metrics = pd.DataFrame(metrics_data)
                st.dataframe(df_metrics, use_container_width=True, hide_index=True)
                
                # Unique codes with meanings
                st.markdown("### 🔍 Unique Codes with Meanings")
                
                code_meanings = safe_get(pharmacy_extraction, 'code_meanings', {})
                ndc_meanings = safe_get(code_meanings, 'ndc_code_meanings', {})
                medication_meanings = safe_get(code_meanings, 'medication_meanings', {})
                
                if ndc_meanings:
                    st.markdown("**NDC Codes:**")
                    ndc_data = []
                    for code, meaning in ndc_meanings.items():
                        ndc_data.append({
                            'Code': code,
                            'Type': 'NDC Code',
                            'Meaning': meaning[:100] + '...' if len(meaning) > 100 else meaning
                        })
                    
                    if ndc_data:
                        df_ndc = pd.DataFrame(ndc_data)
                        st.dataframe(df_ndc, use_container_width=True, hide_index=True)
                
                if medication_meanings:
                    st.markdown("**Medications:**")
                    med_data = []
                    for med, meaning in medication_meanings.items():
                        med_data.append({
                            'Code': med,
                            'Type': 'Medication',
                            'Meaning': meaning[:100] + '...' if len(meaning) > 100 else meaning
                        })
                    
                    if med_data:
                        df_med = pd.DataFrame(med_data)
                        st.dataframe(df_med, use_container_width=True, hide_index=True)
                
                if not ndc_meanings and not medication_meanings:
                    st.info("No code meanings available. Codes processed but meanings not generated.")
            else:
                st.warning("No pharmacy extraction data available")
    else:
        st.warning("No structured extraction data available")

def display_entity_extraction_five_boxes():
    """Display entity extraction results using the exact format from the uploaded code"""
    st.markdown("""
    <div class="section-box">
        <div class="section-title">🎯 Enhanced Entity Extraction</div>
    </div>
    """, unsafe_allow_html=True)
    
    analysis_results = st.session_state.get('analysis_results')
    if not analysis_results:
        st.warning("No analysis results available")
        return
    
    entity_extraction = safe_get(analysis_results, 'entity_extraction', {})
    if entity_extraction:
        # Five boxes in the exact format from the uploaded code
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <h3>🩺</h3>
                <p><strong>Diabetes</strong></p>
                <h4>{safe_get(entity_extraction, 'diabetics', 'unknown').upper()}</h4>
            </div>
            <div class="metric-card">
                <h3>👥</h3>
                <p><strong>Age Group</strong></p>
                <h4>{safe_get(entity_extraction, 'age_group', 'unknown').upper()}</h4>
            </div>
            <div class="metric-card">
                <h3>🚬</h3>
                <p><strong>Smoking</strong></p>
                <h4>{safe_get(entity_extraction, 'smoking', 'unknown').upper()}</h4>
            </div>
            <div class="metric-card">
                <h3>🍷</h3>
                <p><strong>Alcohol</strong></p>
                <h4>{safe_get(entity_extraction, 'alcohol', 'unknown').upper()}</h4>
            </div>
            <div class="metric-card">
                <h3>💓</h3>
                <p><strong>Blood Pressure</strong></p>
                <h4>{safe_get(entity_extraction, 'blood_pressure', 'unknown').upper()}</h4>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional entity information if available
        enhanced_analysis = safe_get(entity_extraction, 'enhanced_clinical_analysis', False)
        if enhanced_analysis:
            st.success("✅ Enhanced clinical analysis completed")
        
        # Show clinical complexity if available
        complexity_score = safe_get(entity_extraction, 'clinical_complexity_score', 0)
        if complexity_score > 0:
            st.info(f"🔬 Clinical Complexity Score: {complexity_score}")
        
        # Show additional details in expandable section
        with st.expander("🔍 View Detailed Entity Extraction Results"):
            st.json(entity_extraction)
    else:
        st.warning("No entity extraction data available")

def display_enhanced_trajectory_results():
    """Display enhanced trajectory analysis results"""
    st.markdown("""
    <div class="section-box">
        <div class="section-title">📈 Enhanced Health Trajectory Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    analysis_results = st.session_state.get('analysis_results')
    if not analysis_results:
        st.warning("No analysis results available")
        return
    
    enhanced_trajectory = safe_get(analysis_results, 'enhanced_health_trajectory', '')
    
    if enhanced_trajectory:
        st.markdown("### 📈 Comprehensive Health Trajectory Analysis")
        st.markdown(enhanced_trajectory)
    else:
        st.warning("No enhanced trajectory data available")

def display_heart_attack_prediction_results():
    """Display heart attack prediction results"""
    st.markdown("""
    <div class="section-box">
        <div class="section-title">❤️ Enhanced Heart Attack Risk Prediction</div>
    </div>
    """, unsafe_allow_html=True)
    
    analysis_results = st.session_state.get('analysis_results')
    if not analysis_results:
        st.warning("No analysis results available")
        return
    
    heart_attack_prediction = safe_get(analysis_results, 'heart_attack_prediction', {})
    heart_attack_features = safe_get(analysis_results, 'heart_attack_features', {})
    heart_attack_risk_score = safe_get(analysis_results, 'heart_attack_risk_score', 0.0)
    
    if heart_attack_prediction:
        # Display prediction results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_display = safe_get(heart_attack_prediction, 'risk_display', 'Risk assessment unavailable')
            st.metric("Risk Assessment", safe_str(risk_display))
        
        with col2:
            confidence_display = safe_get(heart_attack_prediction, 'confidence_display', 'Confidence unknown')
            st.metric("Prediction Confidence", safe_str(confidence_display))
        
        with col3:
            try:
                risk_score = float(heart_attack_risk_score)
                st.metric("Risk Score", f"{risk_score:.1%}")
            except:
                st.metric("Risk Score", safe_str(heart_attack_risk_score))
        
        # Clinical interpretation
        clinical_interpretation = safe_get(heart_attack_prediction, 'clinical_interpretation', '')
        if clinical_interpretation:
            st.markdown("### 🔬 Clinical Interpretation")
            st.markdown(clinical_interpretation)
        
        # Features used
        if heart_attack_features:
            with st.expander("📊 View Risk Assessment Features"):
                st.json(heart_attack_features)
    else:
        st.warning("❌ No heart attack prediction data available")

def display_chatbot_interface():
    """Display enhanced chatbot interface in sidebar"""
    analysis_results = st.session_state.get('analysis_results')
    chatbot_ready = analysis_results and safe_get(analysis_results, "chatbot_ready", False)
    chatbot_context = st.session_state.get('chatbot_context')
    
    if chatbot_ready and chatbot_context:
        st.title("💬 Enhanced AI Healthcare Assistant")
        
        # Chat history display
        chat_container = st.container()
        with chat_container:
            chatbot_messages = st.session_state.get('chatbot_messages', [])
            if chatbot_messages:
                for i, message in enumerate(chatbot_messages):
                    with st.chat_message(safe_get(message, "role", "user")):
                        st.write(safe_get(message, "content", ""))
            else:
                st.info("👋 Hello! I'm your Enhanced Healthcare AI Assistant!")
                st.info("🎯 Ask me about your health analysis results!")
        
        # Chat input
        st.markdown("---")
        user_question = st.chat_input("Ask detailed healthcare questions...")
        
        # Handle chat input
        if user_question:
            st.session_state.chatbot_messages.append({"role": "user", "content": user_question})
            
            try:
                with st.spinner("🤖 Processing your healthcare question..."):
                    agent = st.session_state.get('agent')
                    
                    if agent and chatbot_context:
                        success, result = safe_execute(
                            agent.chat_with_enhanced_graphs,
                            user_question, 
                            chatbot_context, 
                            st.session_state.chatbot_messages
                        )
                        if success:
                            response, _, _ = result
                            st.session_state.chatbot_messages.append({"role": "assistant", "content": response})
                        else:
                            log_error(f"Chat error: {result}", "chatbot")
                            st.error(f"Chat error: {result}")
                    else:
                        st.error("Agent or context not available")
                
                st.rerun()
                
            except Exception as e:
                log_error(f"Chat input error: {str(e)}", "chatbot")
                st.error(f"Error: {str(e)}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chatbot_messages = []
            st.rerun()
    
    else:
        st.title("💬 Enhanced AI Healthcare Assistant")
        st.info("💤 Assistant available after analysis completion")
        st.markdown("---")
        st.markdown("**🚀 Enhanced Features:**")
        st.markdown("• 📊 **Advanced Analytics** - Comprehensive health insights")
        st.markdown("• 🎯 **Specialized Analysis** - Healthcare-specific AI")
        st.markdown("• ❤️ **Risk Assessment** - Advanced health modeling")
        st.markdown("• 💡 **Smart Insights** - Evidence-based recommendations")

# Initialize session state
initialize_session_state()

# Enhanced Main Title
st.markdown('<h1 class="main-header">🚀 Enhanced Health Analysis Agent</h1>', unsafe_allow_html=True)

# Display import status
if not AGENT_AVAILABLE:
    st.markdown(f'<div class="status-error">❌ Failed to import Enhanced Health Agent: {import_error}</div>', unsafe_allow_html=True)
    st.markdown("""
    **Troubleshooting Steps:**
    1. Ensure all required dependencies are installed
    2. Check that all Python files are in the same directory
    3. Verify the health_agent_core_enhanced.py file exists and is properly formatted
    4. Check the Python environment and package versions
    """)
    st.stop()

# ENHANCED SIDEBAR CHATBOT
with st.sidebar:
    display_chatbot_interface()

# 1. PATIENT INFORMATION
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
        
        # ENHANCED RUN ANALYSIS BUTTON
        submitted = st.form_submit_button(
            "🚀 Run Enhanced Healthcare Analysis", 
            use_container_width=True,
            disabled=st.session_state.get('analysis_running', False),
            type="primary"
        )

# Handle form submission
if submitted:
    # Prepare patient data
    patient_data = {
        "first_name": first_name,
        "last_name": last_name,
        "ssn": ssn,
        "date_of_birth": date_of_birth.strftime('%Y-%m-%d') if date_of_birth else "",
        "gender": gender,
        "zip_code": zip_code
    }
    
    # Validate data
    is_valid, validation_errors = validate_patient_data(patient_data)
    
    if not is_valid:
        st.markdown('<div class="status-error">❌ Please fix the following errors:</div>', unsafe_allow_html=True)
        for error in validation_errors:
            st.error(f"• {error}")
    else:
        # Set analysis running state
        st.session_state.analysis_running = True
        st.session_state.analysis_results = None
        st.session_state.chatbot_messages = []
        
        # Run analysis
        results = run_analysis_workflow(patient_data)
        
        # Clear running state
        st.session_state.analysis_running = False
        
        if results:
            st.success("✅ Enhanced healthcare analysis completed successfully!")
            st.rerun()
        else:
            st.error("❌ Analysis failed. Please check the error logs for more details.")

# ENHANCED RESULTS SECTION
analysis_results = st.session_state.get('analysis_results')
analysis_running = st.session_state.get('analysis_running', False)

if analysis_results and not analysis_running:
    st.markdown("---")
    st.markdown("## 📊 Enhanced Healthcare Analysis Results")
    
    # Show errors if any
    errors = safe_get(analysis_results, 'errors', [])
    if errors:
        st.markdown('<div class="status-error">❌ Analysis errors occurred</div>', unsafe_allow_html=True)
        with st.expander("🐛 Debug Information"):
            st.write("**Errors:**")
            for error in errors:
                st.write(f"• {error}")

    # 2. CLAIMS DATA VIEWER (Fixed with 3 tabs and JSON only)
    if st.button("📊 Claims Data", use_container_width=True, key="claims_data_btn"):
        st.session_state.show_claims_data = not st.session_state.get('show_claims_data', False)
    
    if st.session_state.get('show_claims_data', False):
        display_claims_data_viewer()

    # 3. BATCH CODE PROCESSING (Fixed with tabular format)
    if st.button("🚀 Batch Code Processing Results", use_container_width=True, key="batch_extraction_btn"):
        st.session_state.show_batch_extraction = not st.session_state.get('show_batch_extraction', False)
    
    if st.session_state.get('show_batch_extraction', False):
        display_batch_extraction_tabular()

    # 4. ENTITY EXTRACTION (Fixed with 5 boxes)
    if st.button("🎯 Entity Extraction", use_container_width=True, key="entity_extraction_btn"):
        st.session_state.show_entity_extraction = not st.session_state.get('show_entity_extraction', False)
    
    if st.session_state.get('show_entity_extraction', False):
        display_entity_extraction_five_boxes()

    # 5. ENHANCED TRAJECTORY RESULTS
    if st.button("📈 Enhanced Health Trajectory Analysis", use_container_width=True, key="enhanced_trajectory_btn"):
        st.session_state.show_enhanced_trajectory = not st.session_state.get('show_enhanced_trajectory', False)
    
    if st.session_state.get('show_enhanced_trajectory', False):
        display_enhanced_trajectory_results()

    # 6. HEART ATTACK PREDICTION RESULTS
    if st.button("❤️ Enhanced Heart Attack Risk Prediction", use_container_width=True, key="enhanced_heart_attack_btn"):
        st.session_state.show_heart_attack = not st.session_state.get('show_heart_attack', False)
    
    if st.session_state.get('show_heart_attack', False):
        display_heart_attack_prediction_results()

# Error Log & Debugging
if st.session_state.get('error_log'):
    with st.expander("🐛 Error Log & Debugging", expanded=False):
        st.markdown("### Recent Errors")
        for error in st.session_state.error_log[-10:]:  # Show last 10 errors
            timestamp = safe_get(error, 'timestamp', 'Unknown time')
            message = safe_get(error, 'message', 'Unknown error')
            context = safe_get(error, 'context', 'Unknown context')
            
            st.error(f"**{timestamp}** [{context}]: {message}")
        
        if st.button("Clear Error Log"):
            st.session_state.error_log = []
            st.rerun()

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin: 2rem 0;">
    🚀 Enhanced Health Analysis Agent v4.0 | 
    Fixed Claims Data Viewer | Tabular Batch Processing | Five-Box Entity Extraction
</div>
""", unsafe_allow_html=True)
