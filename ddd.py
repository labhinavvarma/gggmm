# Configure Streamlit page FIRST
import streamlit as st

# Determine sidebar state - COLLAPSED by default with larger width
sidebar_state = "collapsed"

st.set_page_config(
    page_title="⚡ Enhanced Health Agent with Graph Generation",
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
from typing import Dict, Any, Optional, Callable
import matplotlib.pyplot as plt
import matplotlib
import io
import base64
import re
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import threading
from queue import Queue, Empty

# ENHANCED MATPLOTLIB CONFIGURATION FOR STREAMLIT
matplotlib.use('Agg')  # Use non-interactive backend
plt.ioff()  # Turn off interactive mode

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*FigureCanvasAgg is non-interactive.*')

# Set default style
plt.style.use('default')

# Configure matplotlib parameters for better Streamlit integration
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'figure.figsize': (10, 6),
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'text.usetex': False,
})

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

# ORIGINAL Enhanced CSS with advanced animations and modern styling
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
    position: relative;
    z-index: 2;
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

.analysis-complete-banner {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    padding: 2rem;
    border-radius: 20px;
    border: 2px solid #28a745;
    margin: 2rem 0;
    text-align: center;
    box-shadow: 0 15px 40px rgba(40, 167, 69, 0.2);
}

.entity-card-enhanced {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    border: 2px solid transparent;
    transition: all 0.4s ease;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    position: relative;
    overflow: hidden;
}

.entity-card-enhanced::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #007bff, #28a745, #ffc107, #dc3545);
    transition: all 0.3s ease;
}

.entity-card-enhanced:hover {
    transform: translateY(-8px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    border-color: #007bff;
}

.entity-card-enhanced:hover::before {
    height: 8px;
}

.entity-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    display: block;
    animation: float-icon 3s ease-in-out infinite;
}

@keyframes float-icon {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-5px); }
}

.entity-label {
    font-size: 1rem;
    color: #6c757d;
    margin-bottom: 0.5rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.entity-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #2c3e50;
    margin-bottom: 1rem;
}

.entity-value.positive { color: #dc3545; }
.entity-value.negative { color: #28a745; }
.entity-value.unknown { color: #6c757d; }

.metric-summary-box {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 0.8rem;
    border: 2px solid #dee2e6;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
}

.metric-summary-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    border-color: #007bff;
}

.status-error {
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    color: #d32f2f;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    border-left: 4px solid #f44336;
    font-weight: 600;
}

.live-sync-indicator {
    background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 4px solid #28a745;
    font-weight: 600;
    animation: live-pulse 1.5s infinite;
}

@keyframes live-pulse {
    0%, 100% { opacity: 0.9; }
    50% { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# Helper functions
def safe_get(dictionary, keys, default=None):
    """Safely get nested dictionary values"""
    if isinstance(keys, str):
        keys = [keys]
    
    current = dictionary
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

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

def display_advanced_professional_workflow():
    """ORIGINAL Display the advanced professional workflow animation - WITH REAL LANGGRAPH SYNC"""
    
    # Calculate statistics
    total_steps = len(st.session_state.workflow_steps)
    completed_steps = sum(1 for step in st.session_state.workflow_steps if step['status'] == 'completed')
    running_steps = sum(1 for step in st.session_state.workflow_steps if step['status'] == 'running')
    error_steps = sum(1 for step in st.session_state.workflow_steps if step['status'] == 'error')
    progress_percentage = (completed_steps / total_steps) * 100
    
    # Main container with ORIGINAL styling
    st.markdown('<div class="advanced-workflow-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; position: relative; z-index: 2;">
        <h2 style="color: #2c3e50; font-weight: 700;">🧠 LangGraph Healthcare Analysis Pipeline</h2>
        <p style="color: #34495e; font-size: 1.1rem;">Live synchronized workflow with real LangGraph execution</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Live sync indicator - shows REAL status
    if st.session_state.analysis_running:
        current_node = next((step['name'] for step in st.session_state.workflow_steps if step['status'] == 'running'), 'Initializing')
        st.markdown(f"""
        <div class="live-sync-indicator" style="position: relative; z-index: 2;">
            🟢 <strong>LIVE:</strong> LangGraph node executing | 
            <strong>Current:</strong> {current_node} | 
            <strong>Real Progress:</strong> {progress_percentage:.0f}%
        </div>
        """, unsafe_allow_html=True)
    
    # Progress metrics with ORIGINAL styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("LangGraph Nodes", total_steps)
    with col2:
        st.metric("Completed", completed_steps)
    with col3:
        st.metric("Processing", running_steps)
    with col4:
        st.metric("Real Progress", f"{progress_percentage:.0f}%")
    
    # Progress bar
    st.progress(progress_percentage / 100)
    
    # Display each step with ORIGINAL animations
    for i, step in enumerate(st.session_state.workflow_steps):
        status = step['status']
        name = step['name']
        description = step['description']
        icon = step['icon']
        
        # Map to LangGraph nodes
        langgraph_node_map = {
            'API Fetch': 'fetch_api_data',
            'Deidentification': 'deidentify_claims_data', 
            'Field Extraction': 'extract_claims_fields',
            'Entity Extraction': 'extract_entities',
            'Health Trajectory': 'analyze_trajectory',
            'Heart Risk Prediction': 'predict_heart_attack',
            'Chatbot Initialization': 'initialize_chatbot'
        }
        
        langgraph_node = langgraph_node_map.get(name, name)
        
        # Determine styling based on REAL status from LangGraph
        if status == 'completed':
            step_class = "workflow-step completed"
            status_emoji = "✅"
            status_text = "Node Complete"
        elif status == 'running':
            step_class = "workflow-step running"
            status_emoji = "🔄"
            status_text = "Node Executing"
        elif status == 'error':
            step_class = "workflow-step error"
            status_emoji = "❌"
            status_text = "Node Failed"
        else:
            step_class = "workflow-step"
            status_emoji = "⏳"
            status_text = "Pending"
        
        st.markdown(f"""
        <div class="{step_class}">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 1.5rem;">{icon}</div>
                <div style="flex: 1;">
                    <h4 style="margin: 0; color: #2c3e50;">{name}</h4>
                    <p style="margin: 0; color: #666; font-size: 0.9rem;">{description}</p>
                    <small style="color: #888; font-size: 0.8rem;">LangGraph Node: <code>{langgraph_node}</code></small>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.2rem;">{status_emoji}</div>
                    <small style="color: #666; font-size: 0.8rem;">{status_text}</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Status message with REAL LangGraph status
    if running_steps > 0:
        current_step_name = next((step['name'] for step in st.session_state.workflow_steps if step['status'] == 'running'), 'Processing')
        status_message = f"🧠 LIVE: LangGraph executing {current_step_name}"
    elif completed_steps == total_steps:
        status_message = "🎉 All LangGraph nodes completed successfully!"
    elif error_steps > 0:
        status_message = f"❌ {error_steps} LangGraph node(s) failed"
    else:
        status_message = "🚀 LangGraph workflow ready to execute..."
    
    st.markdown(f"""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.8); border-radius: 10px; position: relative; z-index: 2;">
        <p style="margin: 0; font-weight: 600; color: #2c3e50;">{status_message}</p>
    </div>
    </div>
    """, unsafe_allow_html=True)

# REAL LANGGRAPH PROGRESS TRACKER
class LangGraphProgressTracker:
    """Track real LangGraph progress and update UI accordingly"""
    
    def __init__(self):
        self.progress_queue = Queue()
        self.node_mapping = {
            'fetch_api_data': 'API Fetch',
            'deidentify_claims_data': 'Deidentification',
            'extract_claims_fields': 'Field Extraction', 
            'extract_entities': 'Entity Extraction',
            'analyze_trajectory': 'Health Trajectory',
            'predict_heart_attack': 'Heart Risk Prediction',
            'initialize_chatbot': 'Chatbot Initialization'
        }
    
    def create_progress_callback(self):
        """Create a progress callback for the LangGraph agent"""
        def progress_callback(node_name: str, status: str, data: dict = None):
            """Callback function to track LangGraph node progress"""
            ui_name = self.node_mapping.get(node_name, node_name)
            self.progress_queue.put({
                'node_name': node_name,
                'ui_name': ui_name,
                'status': status,
                'data': data,
                'timestamp': datetime.now()
            })
        
        return progress_callback
    
    def update_workflow_status(self, node_name: str, status: str):
        """Update workflow step status based on LangGraph node"""
        ui_name = self.node_mapping.get(node_name, node_name)
        
        for i, step in enumerate(st.session_state.workflow_steps):
            if step['name'] == ui_name:
                st.session_state.workflow_steps[i]['status'] = status
                break

# ENHANCED LANGGRAPH WRAPPER with REAL PROGRESS
class LangGraphAgentWrapper:
    """Wrapper around HealthAnalysisAgent to provide real progress updates"""
    
    def __init__(self, agent):
        self.agent = agent
        self.progress_tracker = LangGraphProgressTracker()
        
    def run_analysis_with_live_updates(self, patient_data: Dict[str, Any], 
                                     workflow_placeholder, progress_placeholder):
        """Run LangGraph analysis with real live progress updates"""
        
        try:
            # Set up progress tracking
            progress_callback = self.progress_tracker.create_progress_callback()
            
            # Start monitoring progress in a separate thread
            def monitor_progress():
                while st.session_state.analysis_running:
                    try:
                        # Check for progress updates
                        progress_update = self.progress_tracker.progress_queue.get(timeout=0.1)
                        
                        # Update workflow status based on real LangGraph progress
                        self.progress_tracker.update_workflow_status(
                            progress_update['node_name'], 
                            progress_update['status']
                        )
                        
                        # Update UI immediately
                        with workflow_placeholder.container():
                            display_advanced_professional_workflow()
                        
                        with progress_placeholder.container():
                            if progress_update['status'] == 'running':
                                st.info(f"🧠 LIVE: Executing LangGraph node `{progress_update['node_name']}`")
                            elif progress_update['status'] == 'completed':
                                st.success(f"✅ Completed: {progress_update['ui_name']}")
                            elif progress_update['status'] == 'error':
                                st.error(f"❌ Failed: {progress_update['ui_name']}")
                        
                    except Empty:
                        continue
                    except Exception as e:
                        logger.warning(f"Progress monitoring error: {e}")
                        continue
            
            # Start progress monitoring thread
            progress_thread = threading.Thread(target=monitor_progress, daemon=True)
            progress_thread.start()
            
            # Patch the agent to send progress updates
            original_methods = {}
            
            # Hook into LangGraph node methods to send real progress
            if hasattr(self.agent, 'fetch_api_data'):
                original_methods['fetch_api_data'] = self.agent.fetch_api_data
                def wrapped_fetch_api_data(state):
                    progress_callback('fetch_api_data', 'running')
                    result = original_methods['fetch_api_data'](state)
                    status = 'completed' if not state.get('errors') else 'error'
                    progress_callback('fetch_api_data', status)
                    return result
                self.agent.fetch_api_data = wrapped_fetch_api_data
            
            if hasattr(self.agent, 'deidentify_claims_data'):
                original_methods['deidentify_claims_data'] = self.agent.deidentify_claims_data
                def wrapped_deidentify_claims_data(state):
                    progress_callback('deidentify_claims_data', 'running')
                    result = original_methods['deidentify_claims_data'](state)
                    status = 'completed' if not state.get('errors') else 'error'
                    progress_callback('deidentify_claims_data', status)
                    return result
                self.agent.deidentify_claims_data = wrapped_deidentify_claims_data
            
            if hasattr(self.agent, 'extract_claims_fields'):
                original_methods['extract_claims_fields'] = self.agent.extract_claims_fields
                def wrapped_extract_claims_fields(state):
                    progress_callback('extract_claims_fields', 'running')
                    result = original_methods['extract_claims_fields'](state)
                    status = 'completed' if not state.get('errors') else 'error'
                    progress_callback('extract_claims_fields', status)
                    return result
                self.agent.extract_claims_fields = wrapped_extract_claims_fields
            
            if hasattr(self.agent, 'extract_entities'):
                original_methods['extract_entities'] = self.agent.extract_entities
                def wrapped_extract_entities(state):
                    progress_callback('extract_entities', 'running')
                    result = original_methods['extract_entities'](state)
                    status = 'completed' if not state.get('errors') else 'error'
                    progress_callback('extract_entities', status)
                    return result
                self.agent.extract_entities = wrapped_extract_entities
            
            if hasattr(self.agent, 'analyze_trajectory'):
                original_methods['analyze_trajectory'] = self.agent.analyze_trajectory
                def wrapped_analyze_trajectory(state):
                    progress_callback('analyze_trajectory', 'running')
                    result = original_methods['analyze_trajectory'](state)
                    status = 'completed' if not state.get('errors') else 'error'
                    progress_callback('analyze_trajectory', status)
                    return result
                self.agent.analyze_trajectory = wrapped_analyze_trajectory
            
            if hasattr(self.agent, 'predict_heart_attack'):
                original_methods['predict_heart_attack'] = self.agent.predict_heart_attack
                def wrapped_predict_heart_attack(state):
                    progress_callback('predict_heart_attack', 'running')
                    result = original_methods['predict_heart_attack'](state)
                    status = 'completed' if not state.get('errors') else 'error'
                    progress_callback('predict_heart_attack', status)
                    return result
                self.agent.predict_heart_attack = wrapped_predict_heart_attack
            
            if hasattr(self.agent, 'initialize_chatbot'):
                original_methods['initialize_chatbot'] = self.agent.initialize_chatbot
                def wrapped_initialize_chatbot(state):
                    progress_callback('initialize_chatbot', 'running')
                    result = original_methods['initialize_chatbot'](state)
                    status = 'completed' if not state.get('errors') else 'error'
                    progress_callback('initialize_chatbot', status)
                    return result
                self.agent.initialize_chatbot = wrapped_initialize_chatbot
            
            # Execute the actual LangGraph analysis
            results = self.agent.run_analysis(patient_data)
            
            # Restore original methods
            for method_name, original_method in original_methods.items():
                setattr(self.agent, method_name, original_method)
            
            return results
            
        except Exception as e:
            logger.error(f"LangGraph execution failed: {e}")
            # Mark all remaining steps as error
            for step in st.session_state.workflow_steps:
                if step['status'] not in ['completed', 'error']:
                    step['status'] = 'error'
            
            with workflow_placeholder.container():
                display_advanced_professional_workflow()
            
            raise e

def display_batch_code_meanings_enhanced(results):
    """Enhanced batch processed code meanings display"""
    st.markdown("### 📋 Claims Data Analysis")
    
    medical_extraction = safe_get(results, 'structured_extractions', {}).get('medical', {})
    pharmacy_extraction = safe_get(results, 'structured_extractions', {}).get('pharmacy', {})
    
    if medical_extraction or pharmacy_extraction:
        tab1, tab2 = st.tabs(["Medical Claims", "Pharmacy Claims"])
        
        with tab1:
            medical_records = medical_extraction.get("hlth_srvc_records", [])
            unique_service_codes = set()
            unique_diagnosis_codes = set()
            
            for record in medical_records:
                service_code = record.get("hlth_srvc_cd", "")
                if service_code:
                    unique_service_codes.add(service_code)
                
                for diag in record.get("diagnosis_codes", []):
                    code = diag.get("code", "")
                    if code:
                        unique_diagnosis_codes.add(code)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Service Codes", len(unique_service_codes))
            with col2:
                st.metric("Diagnosis Codes", len(unique_diagnosis_codes))
            with col3:
                st.metric("Medical Records", len(medical_records))
            
            if medical_records:
                with st.expander("Sample Medical Records"):
                    st.dataframe(pd.DataFrame(medical_records[:5]), use_container_width=True)
        
        with tab2:
            pharmacy_records = pharmacy_extraction.get("ndc_records", [])
            unique_ndc_codes = set()
            unique_medications = set()
            
            for record in pharmacy_records:
                ndc_code = record.get("ndc", "")
                if ndc_code:
                    unique_ndc_codes.add(ndc_code)
                
                med_name = record.get("lbl_nm", "")
                if med_name:
                    unique_medications.add(med_name)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("NDC Codes", len(unique_ndc_codes))
            with col2:
                st.metric("Medications", len(unique_medications))
            with col3:
                st.metric("Pharmacy Records", len(pharmacy_records))
            
            if pharmacy_records:
                with st.expander("Sample Pharmacy Records"):
                    st.dataframe(pd.DataFrame(pharmacy_records[:5]), use_container_width=True)
    else:
        st.warning("No claims data analysis available")

# Initialize session state
def initialize_session_state():
    """Initialize session state variables for enhanced processing"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'agent_wrapper' not in st.session_state:
        st.session_state.agent_wrapper = None
    if 'config' not in st.session_state:
        st.session_state.config = None
    if 'chatbot_context' not in st.session_state:
        st.session_state.chatbot_context = None
    if 'calculated_age' not in st.session_state:
        st.session_state.calculated_age = None
    
    # Enhanced workflow steps - REAL LANGGRAPH NODE MAPPING
    if 'workflow_steps' not in st.session_state:
        st.session_state.workflow_steps = [
            {'name': 'API Fetch', 'status': 'pending', 'description': 'Fetching comprehensive claims data', 'icon': '⚡', 'node': 'fetch_api_data'},
            {'name': 'Deidentification', 'status': 'pending', 'description': 'Advanced PII removal with clinical preservation', 'icon': '🔒', 'node': 'deidentify_claims_data'},
            {'name': 'Field Extraction', 'status': 'pending', 'description': 'Extracting medical and pharmacy fields', 'icon': '🚀', 'node': 'extract_claims_fields'},
            {'name': 'Entity Extraction', 'status': 'pending', 'description': 'Advanced health entity identification', 'icon': '🎯', 'node': 'extract_entities'},
            {'name': 'Health Trajectory', 'status': 'pending', 'description': 'Comprehensive predictive health analysis', 'icon': '📈', 'node': 'analyze_trajectory'},
            {'name': 'Heart Risk Prediction', 'status': 'pending', 'description': 'ML-based cardiovascular assessment', 'icon': '❤️', 'node': 'predict_heart_attack'},
            {'name': 'Chatbot Initialization', 'status': 'pending', 'description': 'AI assistant with graph generation', 'icon': '💬', 'node': 'initialize_chatbot'}
        ]

def reset_workflow():
    """Reset workflow to initial state"""
    for step in st.session_state.workflow_steps:
        step['status'] = 'pending'

initialize_session_state()

# Enhanced Main Title
st.markdown('<h1 class="main-header">Deep Research Health Agent 2.0</h1>', unsafe_allow_html=True)

# Display import status
if not AGENT_AVAILABLE:
    st.markdown(f'<div class="status-error">Failed to import Health Agent: {import_error}</div>', unsafe_allow_html=True)
    st.stop()

# Sidebar with chatbot info
with st.sidebar:
    st.title("Medical Assistant")
    st.info("Medical Assistant will be available after running health analysis")
    st.markdown("---")
    st.markdown("**REAL LangGraph Sync Features:**")
    st.markdown("• **Live Node Updates:** Real-time progress from LangGraph execution")
    st.markdown("• **Node Status Tracking:** Actual node completion status") 
    st.markdown("• **Error Handling:** Real failure detection and reporting")
    st.markdown("• **Progress Monitoring:** True synchronization with workflow")
    st.markdown("• **Beautiful Animations:** Original styling with real data")

# 1. PATIENT INFORMATION
st.markdown("""
<div class="section-box">
    <div class="section-title">Patient Information</div>
</div>
""", unsafe_allow_html=True)

with st.container():
    with st.form("patient_input_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            first_name = st.text_input("First Name *", value="", type="password")
            last_name = st.text_input("Last Name *", value="", type="password")
        
        with col2:
            ssn = st.text_input("SSN *", value="", type="password")
            date_of_birth = st.date_input(
                "Date of Birth *", 
                value=datetime.now().date(),
                min_value=datetime(1900, 1, 1).date(),
                max_value=datetime.now().date()
            )
        
        with col3:
            gender = st.selectbox("Gender *", ["F", "M"])
            zip_code = st.text_input("Zip Code *", value="", type="password")
        
        # Show calculated age
        if date_of_birth:
            calculated_age = calculate_age(date_of_birth)
            if calculated_age is not None:
                st.session_state.calculated_age = calculated_age
                st.info(f"**Calculated Age:** {calculated_age} years old")
        elif st.session_state.calculated_age is not None:
            st.info(f"**Calculated Age:** {st.session_state.calculated_age} years old")
        
        # RUN ANALYSIS BUTTON
        submitted = st.form_submit_button(
            "🚀 Run LIVE LangGraph Analysis", 
            use_container_width=True,
            disabled=st.session_state.analysis_running,
            type="primary"
        )

# Handle form submission
if submitted:
    # Validate form data
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
        # Start analysis
        reset_workflow()
        st.session_state.analysis_running = True
        st.session_state.analysis_results = None
        st.session_state.chatbot_context = None
        st.session_state.calculated_age = None
        
        # Initialize agent and wrapper
        try:
            config = Config()
            st.session_state.config = config
            st.session_state.agent = HealthAnalysisAgent(config)
            st.session_state.agent_wrapper = LangGraphAgentWrapper(st.session_state.agent)
            
            # DEBUG: Show agent setup
            with st.expander("🔍 **LIVE LangGraph Agent Configuration**", expanded=False):
                st.write("**Real-Time LangGraph Integration:**")
                
                if hasattr(st.session_state.agent, 'graph'):
                    st.success("✅ LangGraph StateGraph compiled and ready")
                if hasattr(st.session_state.agent, 'run_analysis'):
                    st.success("✅ run_analysis method available")
                if hasattr(st.session_state.agent_wrapper, 'run_analysis_with_live_updates'):
                    st.success("✅ Live progress tracking wrapper ready")
                
                st.write("**LangGraph Node → UI Mapping:**")
                for step in st.session_state.workflow_steps:
                    st.write(f"• `{step['node']}` → **{step['name']}**")
                
                st.write("**Live Update Features:**")
                st.write("• ✅ Real-time node status updates")
                st.write("• ✅ Progress tracking with callbacks")
                st.write("• ✅ Error detection and reporting")
                st.write("• ✅ Animation sync with actual execution")
            
        except Exception as e:
            st.error(f"Failed to initialize LangGraph agent: {str(e)}")
            st.session_state.analysis_running = False
            st.stop()
        
        st.rerun()

# Display LIVE synchronized workflow animation and execute LangGraph
if st.session_state.analysis_running:
    st.markdown("## 🧠 LIVE LangGraph Execution")
    
    # Create placeholders for REAL-TIME updates
    workflow_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    # Show initial workflow state
    with workflow_placeholder.container():
        display_advanced_professional_workflow()
    
    with progress_placeholder.container():
        st.info("🚀 Starting LIVE LangGraph healthcare analysis with real-time progress...")
    
    # Execute LangGraph with REAL LIVE UPDATES
    try:
        patient_data = {
            "first_name": first_name,
            "last_name": last_name,
            "ssn": ssn,
            "date_of_birth": date_of_birth.strftime('%Y-%m-%d'),
            "gender": gender,
            "zip_code": zip_code
        }
        
        # Run LangGraph with REAL live updates
        results = st.session_state.agent_wrapper.run_analysis_with_live_updates(
            patient_data, 
            workflow_placeholder,
            progress_placeholder
        )
        
        # Store results
        st.session_state.analysis_results = results
        st.session_state.analysis_running = False
        
        if results and results.get("success") and results.get("chatbot_ready"):
            st.session_state.chatbot_context = results.get("chatbot_context")
        
        # Final status update
        with progress_placeholder.container():
            if results and results.get('success'):
                st.success("🎉 LangGraph workflow completed successfully with LIVE tracking!")
            else:
                st.error("❌ LangGraph workflow encountered errors")
                # Mark failed steps
                for step in st.session_state.workflow_steps:
                    if step['status'] not in ['completed']:
                        step['status'] = 'error'
        
        # Final workflow update
        with workflow_placeholder.container():
            display_advanced_professional_workflow()
        
        st.rerun()
        
    except Exception as e:
        st.session_state.analysis_running = False
        
        with progress_placeholder.container():
            st.error(f"❌ LangGraph execution failed: {str(e)}")
        
        # Mark all steps as error
        for step in st.session_state.workflow_steps:
            step['status'] = 'error'
        
        with workflow_placeholder.container():
            display_advanced_professional_workflow()

# Display results after completion
if st.session_state.analysis_results and not st.session_state.analysis_running:
    results = st.session_state.analysis_results
    
    if results.get("success"):
        # Success banner
        st.markdown("""
        <div class="analysis-complete-banner">
            <h2 style="margin: 0; color: #28a745; font-weight: 700;">🎉 LIVE LangGraph Analysis Complete!</h2>
            <p style="margin: 0.5rem 0; color: #155724; font-size: 1.1rem;">
                All LangGraph nodes executed successfully with real-time progress tracking.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Chatbot launch button
        if results.get("chatbot_ready", False) and st.session_state.chatbot_context:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(
                    "🚀 Launch Medical Assistant", 
                    key="launch_chatbot_main",
                    use_container_width=True,
                    help="Open the Medical Assistant with full LangGraph analysis data"
                ):
                    st.switch_page("pages/chatbot.py")
        
        # Analysis Summary
        st.markdown("## 📊 LangGraph Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            medical_records = len(safe_get(results, 'structured_extractions', {}).get('medical', {}).get('hlth_srvc_records', []))
            st.metric("Medical Records", medical_records)
        with col2:
            pharmacy_records = len(safe_get(results, 'structured_extractions', {}).get('pharmacy', {}).get('ndc_records', []))
            st.metric("Pharmacy Records", pharmacy_records)
        with col3:
            heart_risk = safe_get(results, 'heart_attack_risk_score', 0.0)
            st.metric("Heart Risk Score", f"{heart_risk:.1%}")
        with col4:
            steps_completed = results.get('processing_steps_completed', 0)
            st.metric("LangGraph Nodes", f"{steps_completed}/8")
        
    else:
        st.error("❌ LangGraph analysis encountered errors")
        if results.get('errors'):
            for error in results['errors']:
                st.error(f"• {error}")
    
    # Detailed Results (Expandable)
    with st.expander("📋 Detailed LangGraph Results", expanded=False):
        tab1, tab2, tab3, tab4 = st.tabs(["Claims Data", "Entity Extraction", "Health Trajectory", "Heart Risk"])
        
        with tab1:
            display_batch_code_meanings_enhanced(results)
        
        with tab2:
            entity_extraction = safe_get(results, 'entity_extraction', {})
            if entity_extraction:
                entities_data = [
                    {'icon': '🩺', 'label': 'Diabetes', 'value': entity_extraction.get('diabetics', 'unknown')},
                    {'icon': '👥', 'label': 'Age Group', 'value': entity_extraction.get('age_group', 'unknown')},
                    {'icon': '🚬', 'label': 'Smoking', 'value': entity_extraction.get('smoking', 'unknown')},
                    {'icon': '💓', 'label': 'Blood Pressure', 'value': entity_extraction.get('blood_pressure', 'unknown')}
                ]
                
                cols = st.columns(len(entities_data))
                for i, (col, entity) in enumerate(zip(cols, entities_data)):
                    with col:
                        value = entity['value']
                        st.markdown(f"""
                        <div class="entity-card-enhanced">
                            <span class="entity-icon">{entity['icon']}</span>
                            <div class="entity-label">{entity['label']}</div>
                            <div class="entity-value">{value.upper()}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Entity extraction data not available")
        
        with tab3:
            health_trajectory = safe_get(results, 'health_trajectory', '')
            if health_trajectory:
                st.markdown("### Predictive Health Analysis")
                st.markdown(health_trajectory)
            else:
                st.warning("Health trajectory analysis not available")
        
        with tab4:
            heart_attack_prediction = safe_get(results, 'heart_attack_prediction', {})
            if heart_attack_prediction and not heart_attack_prediction.get('error'):
                risk_category = heart_attack_prediction.get("risk_category", "Unknown")
                combined_display = heart_attack_prediction.get("combined_display", "Not available")
                
                st.markdown("### ML Heart Risk Assessment")
                st.info(combined_display)
                
                if risk_category == 'High Risk':
                    st.error(f"**Risk Category: {risk_category}**")
                elif risk_category == 'Medium Risk':
                    st.warning(f"**Risk Category: {risk_category}**")
                else:
                    st.success(f"**Risk Category: {risk_category}**")
            else:
                st.warning("Heart attack risk prediction not available")

if __name__ == "__main__":
    pass
