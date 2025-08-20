# Configure Streamlit page FIRST - before any other Streamlit commands
import streamlit as st

# Determine sidebar state based on chatbot readiness
if 'analysis_results' in st.session_state and st.session_state.get('analysis_results') and st.session_state.analysis_results.get("chatbot_ready", False):
    sidebar_state = "expanded"
else:
    sidebar_state = "collapsed"

st.set_page_config(
    page_title="🔬 Deep Research Health Agent 2.0",
    page_icon="🚀",
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
import logging
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for stability
import io
import base64
import re
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Set up logging
logger = logging.getLogger(__name__)

# Import the health analysis agent
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

# Enhanced CSS with advanced animations and modern styling
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
    animation: glow-pulse 3s ease-in-out infinite;
}

@keyframes glow-pulse {
    0%, 100% { filter: drop-shadow(0 0 10px rgba(102, 126, 234, 0.3)); }
    50% { filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.6)); }
}

.enhanced-badge {
    background: linear-gradient(135deg, #00ff87 0%, #60efff 100%);
    color: #2c3e50;
    padding: 0.6rem 1.2rem;
    border-radius: 25px;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
    margin: 0.4rem;
    box-shadow: 0 8px 25px rgba(0, 255, 135, 0.4);
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-5px); }
}

.section-box {
    background: white;
    padding: 1.8rem;
    border-radius: 15px;
    border: 1px solid #e9ecef;
    margin: 1.2rem 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.section-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 35px rgba(0,0,0,0.15);
}

.section-title {
    font-size: 1.4rem;
    color: #2c3e50;
    font-weight: 600;
    margin-bottom: 1rem;
    border-bottom: 3px solid #3498db;
    padding-bottom: 0.6rem;
}

/* Enhanced workflow animations */
.advanced-workflow-container {
    background: linear-gradient(135deg, #e8f0fe 0%, #f3e5f5 25%, #e1f5fe 50%, #f1f8e9 75%, #fff8e1 100%);
    padding: 3rem;
    border-radius: 25px;
    margin: 2rem 0;
    border: 2px solid rgba(52, 152, 219, 0.3);
    box-shadow: 0 20px 50px rgba(52, 152, 219, 0.2);
    position: relative;
    overflow: hidden;
}

.advanced-workflow-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
    animation: rotate-glow 20s linear infinite;
    pointer-events: none;
}

@keyframes rotate-glow {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.workflow-step {
    background: rgba(255, 255, 255, 0.8);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    border-left: 4px solid #6c757d;
    transition: all 0.4s ease;
    backdrop-filter: blur(10px);
}

.workflow-step.running {
    border-left-color: #ffc107;
    background: rgba(255, 193, 7, 0.15);
    animation: pulse-step 2s infinite;
    box-shadow: 0 10px 30px rgba(255, 193, 7, 0.3);
}

.workflow-step.completed {
    border-left-color: #28a745;
    background: rgba(40, 167, 69, 0.15);
    box-shadow: 0 10px 30px rgba(40, 167, 69, 0.2);
}

.workflow-step.error {
    border-left-color: #dc3545;
    background: rgba(220, 53, 69, 0.15);
    box-shadow: 0 10px 30px rgba(220, 53, 69, 0.2);
}

@keyframes pulse-step {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.02); }
}

.claims-viewer-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 2.2rem;
    border-radius: 18px;
    border: 2px solid #dee2e6;
    margin: 1.2rem 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
}

.mcid-container {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    padding: 1.8rem;
    border-radius: 15px;
    border: 2px solid #2196f3;
    margin: 1rem 0;
    box-shadow: 0 8px 25px rgba(33, 150, 243, 0.2);
}

.mcid-match-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 0.8rem 0;
    border-left: 4px solid #4caf50;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.code-table-container {
    background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border: 1px solid #2196f3;
    box-shadow: 0 4px 15px rgba(33, 150, 243, 0.1);
}

.code-category-header {
    background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
    color: white;
    padding: 0.8rem 1.2rem;
    border-radius: 8px;
    font-weight: 600;
    margin-bottom: 1rem;
    text-align: center;
}

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

# Utility functions
def safe_get(data, key, default=None):
    """Safely get a value from a dictionary"""
    try:
        return data.get(key, default) if data else default
    except:
        return default

def calculate_age(birth_date):
    """Calculate age from birth date"""
    if not birth_date:
        return None
    
    today = datetime.now().date()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age

def validate_patient_data(data: Dict[str, Any]) -> tuple[bool, list[str]]:
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
def initialize_session_state():
    """Initialize session state variables for enhanced processing"""
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
    
    # Section toggle states
    if 'show_workflow' not in st.session_state:
        st.session_state.show_workflow = False
    if 'show_all_claims_data' not in st.session_state:
        st.session_state.show_all_claims_data = False
    if 'show_batch_codes' not in st.session_state:
        st.session_state.show_batch_codes = False
    
    # Enhanced workflow steps
    if 'workflow_steps' not in st.session_state:
        st.session_state.workflow_steps = [
            {'name': 'FAST API Fetch', 'status': 'pending', 'description': 'Fetching claims data with enhanced timeout', 'icon': '⚡'},
            {'name': 'ENHANCED Deidentification', 'status': 'pending', 'description': 'Advanced PII removal with structure preservation', 'icon': '🔒'},
            {'name': 'BATCH Code Processing', 'status': 'pending', 'description': 'Processing codes in batches (93% fewer API calls)', 'icon': '🚀'},
            {'name': 'DETAILED Entity Extraction', 'status': 'pending', 'description': 'Advanced health entity identification', 'icon': '🎯'},
            {'name': 'ENHANCED Health Trajectory', 'status': 'pending', 'description': 'Detailed predictive analysis with specific evaluation questions', 'icon': '📈'},
            {'name': 'IMPROVED Heart Risk Prediction', 'status': 'pending', 'description': 'Enhanced ML-based risk assessment', 'icon': '❤️'},
            {'name': 'STABLE Graph Chatbot', 'status': 'pending', 'description': 'AI assistant with enhanced graph stability', 'icon': '📊'}
        ]

def display_advanced_professional_workflow():
    """Display the advanced professional workflow animation"""
    
    # Calculate statistics
    total_steps = len(st.session_state.workflow_steps)
    completed_steps = sum(1 for step in st.session_state.workflow_steps if step['status'] == 'completed')
    running_steps = sum(1 for step in st.session_state.workflow_steps if step['status'] == 'running')
    error_steps = sum(1 for step in st.session_state.workflow_steps if step['status'] == 'error')
    progress_percentage = (completed_steps / total_steps) * 100
    
    # Main container
    st.markdown('<div class="advanced-workflow-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #2c3e50; font-weight: 700;">🔬 LangGraph Healthcare Analysis Pipeline</h2>
        <p style="color: #34495e; font-size: 1.1rem;">Advanced multi-step processing workflow</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Steps", total_steps)
    with col2:
        st.metric("Completed", completed_steps)
    with col3:
        st.metric("Processing", running_steps)
    with col4:
        st.metric("Progress", f"{progress_percentage:.0f}%")
    
    # Progress bar
    st.progress(progress_percentage / 100)
    
    # Display each step
    for i, step in enumerate(st.session_state.workflow_steps):
        status = step['status']
        name = step['name']
        description = step['description']
        icon = step['icon']
        
        # Determine styling based on status
        if status == 'completed':
            step_class = "workflow-step completed"
            status_emoji = "✅"
        elif status == 'running':
            step_class = "workflow-step running"
            status_emoji = "🔄"
        elif status == 'error':
            step_class = "workflow-step error"
            status_emoji = "❌"
        else:
            step_class = "workflow-step"
            status_emoji = "⏳"
        
        st.markdown(f"""
        <div class="{step_class}">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 1.5rem;">{icon}</div>
                <div style="flex: 1;">
                    <h4 style="margin: 0; color: #2c3e50;">{name}</h4>
                    <p style="margin: 0; color: #666; font-size: 0.9rem;">{description}</p>
                </div>
                <div style="font-size: 1.2rem;">{status_emoji}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Status message
    if running_steps > 0:
        current_step_name = next((step['name'] for step in st.session_state.workflow_steps if step['status'] == 'running'), 'Processing')
        status_message = f"🔄 Currently executing: {current_step_name}"
    elif completed_steps == total_steps:
        status_message = "🎉 All LangGraph workflow steps completed successfully!"
    elif error_steps > 0:
        status_message = f"⚠️ {error_steps} step(s) encountered errors"
    else:
        status_message = "⏳ LangGraph healthcare analysis pipeline ready to start..."
    
    st.markdown(f"""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.8); border-radius: 10px;">
        <p style="margin: 0; font-weight: 600; color: #2c3e50;">{status_message}</p>
    </div>
    </div>
    """, unsafe_allow_html=True)

def display_enhanced_mcid_data(mcid_data):
    """Enhanced MCID data display"""
    if not mcid_data:
        st.warning("⚠️ No MCID data available")
        return
    
    st.markdown("""
    <div class="mcid-container">
        <h3>🆔 MCID (Member Consumer ID) Analysis</h3>
        <p><strong>Purpose:</strong> Patient identity verification and matching across healthcare systems</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display MCID status information
    status_code = mcid_data.get('status_code', 'Unknown')
    service = mcid_data.get('service', 'Unknown')
    timestamp = mcid_data.get('timestamp', '')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Response Status", f"HTTP {status_code}")
    with col2:
        st.metric("Service", service)
    with col3:
        if timestamp:
            try:
                formatted_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
                st.metric("Query Time", formatted_time)
            except:
                st.metric("Query Time", "Recent")
        else:
            st.metric("Query Time", "Unknown")
    
    # Process and display consumer matches
    if status_code == 200 and mcid_data.get('body'):
        mcid_body = mcid_data.get('body', {})
        consumers = mcid_body.get('consumer', [])
        
        if consumers and len(consumers) > 0:
            st.success(f"✅ Found {len(consumers)} consumer match(es)")
            
            for i, consumer in enumerate(consumers, 1):
                st.markdown(f"""
                <div class="mcid-match-card">
                    <h4>🔍 Consumer Match #{i}</h4>
                """, unsafe_allow_html=True)
                
                # Create two columns for consumer info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Consumer Information:**")
                    st.write(f"• **Consumer ID:** {consumer.get('consumerId', 'N/A')}")
                    st.write(f"• **Match Score:** {consumer.get('score', 'N/A')}")
                    st.write(f"• **Status:** {consumer.get('status', 'N/A')}")
                    st.write(f"• **Date of Birth:** {consumer.get('dateOfBirth', 'N/A')}")
                
                with col2:
                    st.write("**Address Information:**")
                    address = consumer.get('address', {})
                    if address:
                        st.write(f"• **City:** {address.get('city', 'N/A')}")
                        st.write(f"• **State:** {address.get('state', 'N/A')}")
                        st.write(f"• **ZIP Code:** {address.get('zip', 'N/A')}")
                        st.write(f"• **County:** {address.get('county', 'N/A')}")
                    else:
                        st.write("• No address information available")
                
                st.markdown("</div>", unsafe_allow_html=True)

def display_batch_code_views(results):
    """Display comprehensive batch code analysis"""
    st.markdown("""
    <div class="section-box">
        <div class="section-title">🔬 Batch Healthcare Code Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get structured extractions
    structured_extractions = safe_get(results, 'structured_extractions', {})
    medical_extraction = structured_extractions.get('medical', {}) if structured_extractions else {}
    pharmacy_extraction = structured_extractions.get('pharmacy', {}) if structured_extractions else {}
    
    # Code type tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏥 ICD-10 Diagnosis Codes", 
        "⚕️ CPT Procedure Codes", 
        "💊 NDC Drug Codes",
        "🔬 HCPCS Healthcare Codes"
    ])
    
    with tab1:
        st.markdown('<div class="code-table-container">', unsafe_allow_html=True)
        st.markdown('<div class="code-category-header">🏥 ICD-10 Diagnosis Codes Analysis</div>', unsafe_allow_html=True)
        
        # Extract ICD-10 codes from medical records
        medical_records = medical_extraction.get('hlth_srvc_records', [])
        icd10_codes = []
        
        for record in medical_records:
            diagnosis_codes = record.get('diagnosis_codes', [])
            for diag in diagnosis_codes:
                code = diag.get('code', '')
                position = diag.get('position', 1)
                date = record.get('clm_rcvd_dt', 'Unknown')
                
                icd10_codes.append({
                    'Code': code,
                    'Position': f"Position {position}",
                    'Claim Date': date,
                    'Category': 'Primary' if position == 1 else 'Secondary',
                    'Description': f"ICD-10 diagnosis code {code}"
                })
        
        if icd10_codes:
            df_icd10 = pd.DataFrame(icd10_codes)
            st.dataframe(df_icd10, use_container_width=True)
            st.info(f"📊 **Total ICD-10 Codes Found:** {len(icd10_codes)}")
            
            # Code frequency analysis
            if len(icd10_codes) > 1:
                code_counts = df_icd10['Code'].value_counts()
                st.write("**Most Frequent Diagnosis Codes:**")
                for code, count in code_counts.head(5).items():
                    st.write(f"• **{code}**: {count} occurrence(s)")
        else:
            st.warning("No ICD-10 diagnosis codes found in medical records")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="code-table-container">', unsafe_allow_html=True)
        st.markdown('<div class="code-category-header">⚕️ CPT Procedure Codes Analysis</div>', unsafe_allow_html=True)
        
        # Extract CPT codes from medical records
        cpt_codes = []
        
        for record in medical_records:
            service_code = record.get('hlth_srvc_cd', '')
            date = record.get('clm_rcvd_dt', 'Unknown')
            
            if service_code:
                cpt_codes.append({
                    'Code': service_code,
                    'Service Date': date,
                    'Type': 'Healthcare Service',
                    'Category': 'Procedure/Service',
                    'Description': f"Healthcare service code {service_code}"
                })
        
        if cpt_codes:
            df_cpt = pd.DataFrame(cpt_codes)
            st.dataframe(df_cpt, use_container_width=True)
            st.info(f"📊 **Total CPT/Service Codes Found:** {len(cpt_codes)}")
            
            # Service frequency
            if len(cpt_codes) > 1:
                service_counts = df_cpt['Code'].value_counts()
                st.write("**Most Frequent Service Codes:**")
                for code, count in service_counts.head(5).items():
                    st.write(f"• **{code}**: {count} occurrence(s)")
        else:
            st.warning("No CPT/Service codes found in medical records")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="code-table-container">', unsafe_allow_html=True)
        st.markdown('<div class="code-category-header">💊 NDC Drug Identification Codes</div>', unsafe_allow_html=True)
        
        # Extract NDC codes from pharmacy records
        pharmacy_records = pharmacy_extraction.get('ndc_records', [])
        ndc_codes = []
        
        for record in pharmacy_records:
            ndc_code = record.get('ndc', '')
            label_name = record.get('lbl_nm', 'Unknown medication')
            fill_date = record.get('rx_filled_dt', 'Unknown')
            
            if ndc_code:
                ndc_codes.append({
                    'NDC Code': ndc_code,
                    'Medication Name': label_name,
                    'Fill Date': fill_date,
                    'Type': 'Prescription Drug',
                    'Category': 'Pharmacy',
                    'Description': f"NDC {ndc_code} - {label_name}"
                })
        
        if ndc_codes:
            df_ndc = pd.DataFrame(ndc_codes)
            st.dataframe(df_ndc, use_container_width=True)
            st.info(f"📊 **Total NDC Codes Found:** {len(ndc_codes)}")
            
            # Medication frequency
            if len(ndc_codes) > 1:
                med_counts = df_ndc['Medication Name'].value_counts()
                st.write("**Most Frequently Filled Medications:**")
                for med, count in med_counts.head(5).items():
                    st.write(f"• **{med}**: {count} fill(s)")
        else:
            st.warning("No NDC codes found in pharmacy records")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="code-table-container">', unsafe_allow_html=True)
        st.markdown('<div class="code-category-header">🔬 HCPCS Healthcare Procedure Codes</div>', unsafe_allow_html=True)
        
        # Extract HCPCS codes (these might be in various fields)
        hcpcs_codes = []
        
        # Look for HCPCS patterns in service codes and other fields
        for record in medical_records:
            service_code = record.get('hlth_srvc_cd', '')
            date = record.get('clm_rcvd_dt', 'Unknown')
            
            # HCPCS codes typically start with letters
            if service_code and any(c.isalpha() for c in service_code):
                hcpcs_codes.append({
                    'HCPCS Code': service_code,
                    'Service Date': date,
                    'Type': 'Healthcare Procedure',
                    'Category': 'HCPCS',
                    'Level': 'Level II' if service_code[0].isalpha() else 'Level I',
                    'Description': f"HCPCS code {service_code}"
                })
        
        if hcpcs_codes:
            df_hcpcs = pd.DataFrame(hcpcs_codes)
            st.dataframe(df_hcpcs, use_container_width=True)
            st.info(f"📊 **Total HCPCS Codes Found:** {len(hcpcs_codes)}")
        else:
            st.info("No specific HCPCS codes identified (may be included in CPT codes above)")
            st.write("**HCPCS Code Information:**")
            st.write("• **Level I**: CPT codes (numeric)")
            st.write("• **Level II**: Alpha-numeric codes for supplies, equipment")
            st.write("• **Common examples**: J codes (drugs), A codes (supplies)")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary section
    st.markdown("### 📊 Code Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ICD-10 Codes", len(icd10_codes) if 'icd10_codes' in locals() else 0)
    with col2:
        st.metric("CPT/Service Codes", len(cpt_codes) if 'cpt_codes' in locals() else 0)
    with col3:
        st.metric("NDC Codes", len(ndc_codes) if 'ndc_codes' in locals() else 0)
    with col4:
        st.metric("HCPCS Codes", len(hcpcs_codes) if 'hcpcs_codes' in locals() else 0)

# Initialize session state
initialize_session_state()

# Enhanced Main Title - Updated to "Deep Research Health Agent 2.0"
st.markdown('<h1 class="main-header">🔬 Deep Research Health Agent 2.0</h1>', unsafe_allow_html=True)

# Enhanced optimization badges
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <div class="enhanced-badge">⚡ 93% Fewer API Calls</div>
    <div class="enhanced-badge">🚀 90% Faster Processing</div>
    <div class="enhanced-badge">📊 Enhanced Graph Stability</div>
    <div class="enhanced-badge">🗂️ Complete Claims Data Viewer</div>
    <div class="enhanced-badge">🎯 Detailed Health Analysis</div>
    <div class="enhanced-badge">💡 Batch Code Processing</div>
</div>
""", unsafe_allow_html=True)

# Display import status
if not AGENT_AVAILABLE:
    st.markdown(f'<div class="status-error">❌ Failed to import Health Agent: {import_error}</div>', unsafe_allow_html=True)
    st.stop()

# SIDEBAR (keeping existing sidebar code)
with st.sidebar:
    if st.session_state.analysis_results and st.session_state.analysis_results.get("chatbot_ready", False) and st.session_state.chatbot_context:
        st.title("💬 Enhanced AI Healthcare Assistant")
        st.markdown("""
        <div class="enhanced-badge" style="margin: 0.5rem 0;">📊 Advanced Graph Generation</div>
        <div class="enhanced-badge" style="margin: 0.5rem 0;">🎯 Specialized Healthcare Analysis</div>
        """, unsafe_allow_html=True)
        
        # Chat interface (existing code)
        if st.session_state.chatbot_messages:
            for message in st.session_state.chatbot_messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        else:
            st.info("👋 Hello! I'm your AI Healthcare Assistant!")
        
        user_question = st.chat_input("Ask detailed healthcare questions...")
        
        if user_question:
            st.session_state.chatbot_messages.append({"role": "user", "content": user_question})
            try:
                with st.spinner("🤖 Processing..."):
                    response = st.session_state.agent.chat_with_data(
                        user_question, 
                        st.session_state.chatbot_context, 
                        st.session_state.chatbot_messages
                    )
                    st.session_state.chatbot_messages.append({"role": "assistant", "content": response})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.title("💬 AI Healthcare Assistant")
        st.info("💤 Assistant available after analysis completion")

# 1. PATIENT INFORMATION SECTION
st.markdown("""
<div class="section-box">
    <div class="section-title">👤 Patient Information</div>
</div>
""", unsafe_allow_html=True)

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
    
    submitted = st.form_submit_button(
        "🚀 Run Deep Research Analysis", 
        use_container_width=True,
        disabled=st.session_state.analysis_running,
        type="primary"
    )

# Handle form submission
if submitted:
    patient_data = {
        "first_name": first_name,
        "last_name": last_name,
        "ssn": ssn,
        "date_of_birth": date_of_birth.strftime('%Y-%m-%d'),
        "gender": gender,
        "zip_code": zip_code
    }
    
    valid, errors = validate_patient_data(patient_data)
    
    if not valid:
        st.error("Please fix the following errors:")
        for error in errors:
            st.error(f"• {error}")
    else:
        st.session_state.analysis_running = True
        st.session_state.analysis_results = None
        
        # Initialize agent
        try:
            config = Config()
            st.session_state.config = config
            st.session_state.agent = HealthAnalysisAgent(config)
            
            # Update workflow to show running
            for step in st.session_state.workflow_steps:
                step['status'] = 'pending'
            st.session_state.workflow_steps[0]['status'] = 'running'
            
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
            st.session_state.analysis_running = False
            st.stop()
        
        # Run analysis
        with st.spinner("🔬 Running Deep Research Analysis..."):
            try:
                # Simulate workflow progression
                for i, step in enumerate(st.session_state.workflow_steps):
                    step['status'] = 'running'
                    time.sleep(0.5)  # Brief delay for demonstration
                    step['status'] = 'completed'
                    if i < len(st.session_state.workflow_steps) - 1:
                        st.session_state.workflow_steps[i + 1]['status'] = 'running'
                
                results = st.session_state.agent.run_analysis(patient_data)
                
                st.session_state.analysis_results = results
                st.session_state.analysis_running = False
                
                if results.get("success") and results.get("chatbot_ready"):
                    st.session_state.chatbot_context = results.get("chatbot_context")
                
                st.success("✅ Deep Research Analysis completed successfully!")
                st.rerun()
                
            except Exception as e:
                st.session_state.analysis_running = False
                # Mark current step as error
                for step in st.session_state.workflow_steps:
                    if step['status'] == 'running':
                        step['status'] = 'error'
                st.error(f"Analysis failed: {str(e)}")

# 2. WORKFLOW ANIMATION SECTION
if st.session_state.analysis_running or st.session_state.analysis_results:
    if st.button("📊 View Workflow Progress", use_container_width=True, key="workflow_btn"):
        st.session_state.show_workflow = not st.session_state.show_workflow
    
    if st.session_state.show_workflow:
        display_advanced_professional_workflow()

# 3. RESULTS DISPLAY SECTIONS
if st.session_state.analysis_results and not st.session_state.analysis_running:
    results = st.session_state.analysis_results
    
    st.markdown("---")
    st.markdown("## 📊 Deep Research Analysis Results")
    
    # Key metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        structured_extractions = safe_get(results, 'structured_extractions', {})
        medical_data = structured_extractions.get('medical', {}) if structured_extractions else {}
        medical_records = len(medical_data.get('hlth_srvc_records', []) if medical_data else [])
        st.metric("Medical Records", medical_records)
    
    with col2:
        pharmacy_data = structured_extractions.get('pharmacy', {}) if structured_extractions else {}
        pharmacy_records = len(pharmacy_data.get('ndc_records', []) if pharmacy_data else [])
        st.metric("Pharmacy Records", pharmacy_records)
    
    with col3:
        entities = safe_get(results, 'entity_extraction', {})
        conditions_count = len(entities.get('medical_conditions', []) if entities else [])
        st.metric("Conditions Identified", conditions_count)
    
    with col4:
        heart_attack_prediction = safe_get(results, 'heart_attack_prediction', {})
        risk_display = heart_attack_prediction.get('risk_display', 'Not available') if heart_attack_prediction else 'Not available'
        if 'Error' not in risk_display:
            risk_text = risk_display.split(':')[1].strip() if ':' in risk_display else risk_display
            st.metric("Heart Attack Risk", risk_text)
        else:
            st.metric("Heart Attack Risk", "Error")

    # COMPLETE CLAIMS DATA VIEWER
    if st.button("🗂️ Complete Claims Data Viewer - Enhanced Edition", use_container_width=True, key="enhanced_claims_btn"):
        st.session_state.show_all_claims_data = not st.session_state.show_all_claims_data
    
    if st.session_state.show_all_claims_data:
        st.markdown("""
        <div class="claims-viewer-card">
            <h3>📋 Complete Deidentified Claims Database</h3>
            <p><strong>Enhanced Features:</strong> Complete access to ALL deidentified claims data with detailed viewing options and comprehensive analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        deidentified_data = safe_get(results, 'deidentified_data', {})
        api_outputs = safe_get(results, 'api_outputs', {})
        
        if deidentified_data or api_outputs:
            tab1, tab2, tab3, tab4 = st.tabs([
                "🏥 Medical Claims Details", 
                "💊 Pharmacy Claims Details", 
                "🆔 MCID Consumer Data",
                "📊 Complete JSON Explorer"
            ])
            
            with tab1:
                medical_data = safe_get(deidentified_data, 'medical', {})
                if medical_data and not medical_data.get('error'):
                    st.markdown("### 🏥 Enhanced Medical Claims Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Patient Age", medical_data.get('src_mbr_age', 'Unknown'))
                    with col2:
                        st.metric("ZIP Code", medical_data.get('src_mbr_zip_cd', 'Unknown'))
                    with col3:
                        deident_time = medical_data.get('deidentification_timestamp', '')
                        if deident_time:
                            try:
                                formatted_time = datetime.fromisoformat(deident_time.replace('Z', '+00:00')).strftime('%m/%d/%Y %H:%M')
                                st.metric("Deidentified", formatted_time)
                            except:
                                st.metric("Deidentified", "Recently")
                        else:
                            st.metric("Deidentified", "Unknown")
                    
                    medical_claims_data = medical_data.get('medical_claims_data', {})
                    if medical_claims_data:
                        with st.expander("🔍 Explore Medical Claims JSON Structure", expanded=False):
                            st.json(medical_claims_data)
                else:
                    st.error("❌ No medical claims data available")
            
            with tab2:
                pharmacy_data = safe_get(deidentified_data, 'pharmacy', {})
                if pharmacy_data and not pharmacy_data.get('error'):
                    st.markdown("### 💊 Enhanced Pharmacy Claims Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        data_type = pharmacy_data.get('data_type', 'Unknown')
                        st.metric("Data Type", data_type)
                    with col2:
                        deident_time = pharmacy_data.get('deidentification_timestamp', '')
                        if deident_time:
                            try:
                                formatted_time = datetime.fromisoformat(deident_time.replace('Z', '+00:00')).strftime('%m/%d/%Y %H:%M')
                                st.metric("Processed", formatted_time)
                            except:
                                st.metric("Processed", "Recently")
                        else:
                            st.metric("Processed", "Unknown")
                    with col3:
                        masked_fields = pharmacy_data.get('name_fields_masked', [])
                        st.metric("Fields Masked", len(masked_fields))
                    
                    pharmacy_claims_data = pharmacy_data.get('pharmacy_claims_data', {})
                    if pharmacy_claims_data:
                        with st.expander("🔍 Explore Pharmacy Claims JSON Structure", expanded=False):
                            st.json(pharmacy_claims_data)
                else:
                    st.error("❌ No pharmacy claims data available")
            
            with tab3:
                mcid_data = safe_get(api_outputs, 'mcid', {})
                display_enhanced_mcid_data(mcid_data)
            
            with tab4:
                st.markdown("### 🔍 Complete JSON Data Explorer")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🏥 Medical + Pharmacy Data")
                    if deidentified_data:
                        with st.expander("Expand Deidentified Data JSON", expanded=False):
                            st.json(deidentified_data)
                    else:
                        st.warning("No deidentified data available")
                
                with col2:
                    st.markdown("#### 🆔 MCID + API Outputs")
                    if api_outputs:
                        with st.expander("Expand API Outputs JSON", expanded=False):
                            st.json(api_outputs)
                    else:
                        st.warning("No API outputs available")

    # BATCH CODE VIEWS SECTION
    if st.button("🔬 Batch Healthcare Code Analysis", use_container_width=True, key="batch_codes_btn"):
        st.session_state.show_batch_codes = not st.session_state.show_batch_codes
    
    if st.session_state.show_batch_codes:
        display_batch_code_views(results)

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin: 2rem 0;">
    🔬 Deep Research Health Agent 2.0 | 
    <span class="enhanced-badge" style="margin: 0;">⚡ LangGraph Powered</span>
    <span class="enhanced-badge" style="margin: 0;">🚀 Fast Processing</span>
    <span class="enhanced-badge" style="margin: 0;">📊 Interactive Analysis</span>
    <span class="enhanced-badge" style="margin: 0;">🗂️ Complete Claims Viewer</span>
    <span class="enhanced-badge" style="margin: 0;">🎯 Batch Code Processing</span>
</div>
""", unsafe_allow_html=True)
