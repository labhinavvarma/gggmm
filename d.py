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
from typing import Dict, Any, Optional
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
import uuid

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

# Safe marker styles for matplotlib
SAFE_MARKERS = ['o', 's', '^', 'v', '<', '>', 'd', 'p', '*', '+', 'x', 'D', 'h', 'H']
SAFE_LINESTYLES = ['-', '--', '-.', ':']
SAFE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

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

# Enhanced CSS with modern styling
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

/* SECTION AVAILABILITY STATES */
.section-available {
    border: 2px solid #28a745;
    box-shadow: 0 8px 25px rgba(40, 167, 69, 0.2);
}

.section-loading {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    border: 2px solid #ffc107;
    animation: pulse-loading 2s infinite;
}

.section-disabled {
    background: #f8f9fa;
    border: 2px solid #e9ecef;
    opacity: 0.6;
}

@keyframes pulse-loading {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; }
}

.workflow-container {
    background: linear-gradient(135deg, #e8f0fe 0%, #f3e5f5 25%, #e1f5fe 50%, #f1f8e9 75%, #fff8e1 100%);
    padding: 2rem;
    border-radius: 20px;
    margin: 2rem 0;
    border: 2px solid rgba(52, 152, 219, 0.3);
    box-shadow: 0 15px 40px rgba(52, 152, 219, 0.2);
}

.workflow-step {
    background: rgba(255, 255, 255, 0.9);
    padding: 1.2rem;
    border-radius: 12px;
    margin: 0.8rem 0;
    border-left: 4px solid #6c757d;
    transition: all 0.4s ease;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.workflow-step.running {
    border-left-color: #ffc107;
    background: rgba(255, 193, 7, 0.15);
    animation: pulse-step 2s infinite;
}

.workflow-step.completed {
    border-left-color: #28a745;
    background: rgba(40, 167, 69, 0.15);
}

.workflow-step.error {
    border-left-color: #dc3545;
    background: rgba(220, 53, 69, 0.15);
}

@keyframes pulse-step {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.02); }
}

.chatbot-window-btn {
    background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    padding: 1.2rem 2.5rem !important;
    border-radius: 15px !important;
    box-shadow: 0 10px 30px rgba(108, 92, 231, 0.4) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-size: 1.1rem !important;
}

.chatbot-window-btn:hover {
    background: linear-gradient(135deg, #5f3dc4 0%, #9775fa 100%) !important;
    transform: translateY(-5px) !important;
    box-shadow: 0 15px 40px rgba(108, 92, 231, 0.6) !important;
}

.entity-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.2rem;
    margin: 1.5rem 0;
}

.entity-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    border: 2px solid transparent;
    transition: all 0.3s ease;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
}

.entity-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.15);
    border-color: #007bff;
}

.entity-icon {
    font-size: 2.5rem;
    margin-bottom: 0.8rem;
    display: block;
}

.entity-label {
    font-size: 0.9rem;
    color: #6c757d;
    margin-bottom: 0.5rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.entity-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #2c3e50;
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

.trajectory-container {
    background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
    padding: 2rem;
    border-radius: 15px;
    border: 2px solid #28a745;
    margin: 1.5rem 0;
    box-shadow: 0 12px 30px rgba(40, 167, 69, 0.2);
}

.heart-risk-container {
    background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
    padding: 2rem;
    border-radius: 15px;
    border: 2px solid #dc3545;
    margin: 1.5rem 0;
    box-shadow: 0 12px 30px rgba(220, 53, 69, 0.2);
}

.chatbot-modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8);
    z-index: 10000;
    display: flex;
    justify-content: center;
    align-items: center;
}

.chatbot-modal-content {
    background: white;
    width: 90%;
    height: 85%;
    border-radius: 20px;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
    display: flex;
    overflow: hidden;
}

.chatbot-modal-sidebar {
    width: 350px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    color: white;
    overflow-y: auto;
}

.chatbot-modal-main {
    flex: 1;
    background: white;
    display: flex;
    flex-direction: column;
    padding: 2rem;
}

.sidebar-category {
    background: rgba(255, 255, 255, 0.1);
    padding: 0.8rem;
    border-radius: 8px;
    margin: 0.8rem 0;
    border-left: 3px solid #00ff87;
}

.category-prompt-btn {
    background: rgba(255, 255, 255, 0.2);
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 6px;
    padding: 0.5rem 0.8rem;
    margin: 0.3rem 0;
    font-size: 0.85rem;
    transition: all 0.3s ease;
    width: 100%;
    cursor: pointer;
}

.category-prompt-btn:hover {
    background: rgba(255, 255, 255, 0.3);
    transform: translateX(5px);
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

.step-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-left: 0.5rem;
}

.step-pending { background: #6c757d; }
.step-running { 
    background: #ffc107; 
    animation: pulse-indicator 1s infinite;
}
.step-completed { background: #28a745; }
.step-error { background: #dc3545; }

@keyframes pulse-indicator {
    0%, 100% { opacity: 0.7; }
    50% { opacity: 1; }
}

/* Enhanced sidebar sizing */
.css-1d391kg {
    width: 450px !important;
    min-width: 450px !important;
    max-width: 450px !important;
}

@media (max-width: 1200px) {
    .css-1d391kg {
        width: 400px !important;
        min-width: 400px !important;
        max-width: 400px !important;
    }
}

.loading-spinner {
    width: 30px;
    height: 30px;
    border: 3px solid #e3f2fd;
    border-top: 3px solid #2196f3;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 10px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.enhanced-section-btn {
    background: linear-gradient(135deg, #007bff 0%, #0056b3 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 1rem 2rem !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 25px rgba(0, 123, 255, 0.4) !important;
    transition: all 0.3s ease !important;
}

.enhanced-section-btn:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 35px rgba(0, 123, 255, 0.5) !important;
}
</style>
""", unsafe_allow_html=True)

# Helper function for safe data access
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

def extract_matplotlib_code(response: str) -> str:
    """Extract matplotlib code from response"""
    try:
        patterns = [
            r'```python\s*(.*?)```',
            r'```matplotlib\s*(.*?)```', 
            r'```\s*(.*?)```'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                for match in matches:
                    code = match.strip()
                    if any(keyword in code.lower() for keyword in ['matplotlib', 'plt.', 'pyplot']):
                        return code
        return None
    except Exception as e:
        logger.error(f"Error extracting matplotlib code: {e}")
        return None

def execute_matplotlib_code(code: str):
    """Execute matplotlib code safely"""
    try:
        plt.clf()
        plt.close('all')
        
        # Create namespace with sample data
        namespace = {
            'plt': plt,
            'np': np,
            'pd': pd,
            'patient_age': 45,
            'heart_risk_score': 0.25,
            'medications_count': 3,
            'medical_records_count': 8,
            'diabetes_status': 'yes',
            'smoking_status': 'no',
            'bp_status': 'managed',
            'risk_factors': {'Age': 45, 'Diabetes': 1, 'Smoking': 0, 'High_BP': 1},
            'medication_list': ['Metformin', 'Lisinopril', 'Atorvastatin'],
            'diagnosis_codes': ['I10', 'E11.9', 'E78.5']
        }
        
        # Add context data if available
        if st.session_state.chatbot_context:
            context = st.session_state.chatbot_context
            patient_overview = context.get('patient_overview', {})
            entity_extraction = context.get('entity_extraction', {})
            
            namespace.update({
                'patient_age': patient_overview.get('age', 45),
                'heart_risk_score': context.get('heart_attack_risk_score', 0.25),
                'diabetes_status': entity_extraction.get('diabetics', 'no'),
                'smoking_status': entity_extraction.get('smoking', 'no'),
                'bp_status': entity_extraction.get('blood_pressure', 'unknown')
            })
        
        # Execute code
        exec(code, namespace)
        fig = plt.gcf()
        
        # Convert to image
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', 
                   dpi=150, facecolor='white', pad_inches=0.3)
        img_buffer.seek(0)
        
        plt.clf()
        plt.close('all')
        
        return img_buffer
        
    except Exception as e:
        plt.clf()
        plt.close('all')
        logger.error(f"Matplotlib execution error: {e}")
        return None

def create_sample_dashboard():
    """Create sample health dashboard"""
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk Factors', 'Health Metrics', 'Timeline', 'Summary'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "pie"}]]
        )
        
        # Risk factors
        risk_data = [1, 0, 1, 0, 1]
        risk_labels = ['Age', 'Diabetes', 'Smoking', 'BP', 'Family History']
        colors = ['#dc3545' if x == 1 else '#28a745' for x in risk_data]
        
        fig.add_trace(
            go.Bar(x=risk_labels, y=risk_data, marker_color=colors, name="Risk Factors"),
            row=1, col=1
        )
        
        # Health metrics over time
        dates = pd.date_range('2023-01-01', periods=12, freq='ME')
        health_scores = np.random.uniform(60, 90, 12)
        
        fig.add_trace(
            go.Scatter(x=dates, y=health_scores, mode='lines+markers', 
                      name="Health Score", line=dict(color='#007bff')),
            row=1, col=2
        )
        
        # Timeline
        timeline_dates = pd.date_range('2023-01-01', periods=6, freq='2M')
        events = np.random.uniform(1, 10, 6)
        
        fig.add_trace(
            go.Scatter(x=timeline_dates, y=events, mode='markers', 
                      marker=dict(size=12, color='#ffc107'), name="Events"),
            row=2, col=1
        )
        
        # Summary pie
        summary_labels = ['Low Risk', 'Medium Risk', 'High Risk']
        summary_values = [45, 35, 20]
        summary_colors = ['#28a745', '#ffc107', '#dc3545']
        
        fig.add_trace(
            go.Pie(labels=summary_labels, values=summary_values, 
                  marker_colors=summary_colors, name="Risk Distribution"),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="Healthcare Analysis Dashboard",
            title_x=0.5,
            showlegend=False
        )
        
        return fig
    except Exception as e:
        logger.error(f"Dashboard creation error: {e}")
        return None

# Check workflow step status
def check_step_status(step_name: str) -> str:
    """Check status of a workflow step"""
    if hasattr(st.session_state, 'progressive_results') and st.session_state.progressive_results:
        step_mapping = {
            'API Fetch': 'api_outputs',
            'Deidentification': 'deidentified_data', 
            'Field Extraction': 'structured_extractions',
            'Entity Extraction': 'entity_extraction',
            'Health Trajectory': 'health_trajectory',
            'Heart Risk Prediction': 'heart_attack_prediction',
            'Chatbot Initialization': 'chatbot_ready'
        }
        
        result_key = step_mapping.get(step_name)
        if result_key and result_key in st.session_state.progressive_results:
            return 'completed'
    
    return 'pending'

def get_section_availability(section_name: str) -> str:
    """Determine section availability"""
    if not st.session_state.analysis_results:
        return 'disabled'
    
    section_requirements = {
        'claims_data': ['API Fetch', 'Deidentification'],
        'code_analysis': ['Field Extraction'],
        'entity_extraction': ['Entity Extraction'],
        'health_trajectory': ['Health Trajectory'], 
        'heart_risk': ['Heart Risk Prediction'],
        'chatbot': ['Chatbot Initialization']
    }
    
    required_steps = section_requirements.get(section_name, [])
    
    for step_name in required_steps:
        step_status = check_step_status(step_name)
        if step_status != 'completed':
            return 'disabled'
    
    return 'available'

def display_workflow():
    """Display workflow with real-time status"""
    st.markdown("""
    <div class="workflow-container">
        <h3 style="text-align: center; color: #2c3e50; margin-bottom: 2rem;">Healthcare Analysis Pipeline</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate progress
    total_steps = len(st.session_state.workflow_steps)
    completed_steps = sum(1 for step in st.session_state.workflow_steps if step['status'] == 'completed')
    running_steps = sum(1 for step in st.session_state.workflow_steps if step['status'] == 'running')
    progress_percentage = (completed_steps / total_steps) * 100
    
    # Progress metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Steps", total_steps)
    with col2:
        st.metric("Completed", completed_steps)
    with col3:
        st.metric("Running", running_steps)
    with col4:
        st.metric("Progress", f"{progress_percentage:.0f}%")
    
    # Progress bar
    st.progress(progress_percentage / 100)
    
    # Display workflow steps
    for step in st.session_state.workflow_steps:
        status = step['status']
        name = step['name']
        description = step['description']
        icon = step['icon']
        
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
            <div style="font-size: 1.5rem;">{icon}</div>
            <div style="flex: 1;">
                <h4 style="margin: 0; color: #2c3e50;">{name}</h4>
                <p style="margin: 0; color: #666; font-size: 0.9rem;">{description}</p>
            </div>
            <div style="font-size: 1.2rem;">{status_emoji}</div>
            <span class="step-indicator step-{status}"></span>
        </div>
        """, unsafe_allow_html=True)

def run_progressive_analysis(patient_data: Dict[str, Any]):
    """Run analysis with progressive updates"""
    try:
        steps_data = [
            ('API Fetch', {'api_outputs': {'status': 'success', 'medical_data': {}, 'pharmacy_data': {}}}),
            ('Deidentification', {'deidentified_data': {'medical': {}, 'pharmacy': {}}}),
            ('Field Extraction', {'structured_extractions': {'medical': {}, 'pharmacy': {}}}),
            ('Entity Extraction', {'entity_extraction': {
                'diabetics': 'yes', 'smoking': 'no', 'blood_pressure': 'managed',
                'alcohol': 'no', 'age_group': 'middle_aged'
            }}),
            ('Health Trajectory', {'health_trajectory': 'Comprehensive health analysis shows stable trajectory with managed risk factors.'}),
            ('Heart Risk Prediction', {
                'heart_attack_prediction': {
                    'combined_display': 'Heart Disease Risk: 25%',
                    'risk_category': 'Medium Risk'
                },
                'heart_attack_risk_score': 0.25
            }),
            ('Chatbot Initialization', {
                'chatbot_ready': True,
                'chatbot_context': {
                    'patient_overview': {'age': calculate_age(datetime.strptime(patient_data['date_of_birth'], '%Y-%m-%d').date())},
                    'entity_extraction': {'diabetics': 'yes', 'smoking': 'no'},
                    'heart_attack_risk_score': 0.25
                }
            })
        ]
        
        for i, (step_name, step_data) in enumerate(steps_data):
            # Set step to running
            st.session_state.workflow_steps[i]['status'] = 'running'
            st.rerun()
            time.sleep(1.5)  # Simulate processing
            
            # Add step data to progressive results
            st.session_state.progressive_results.update(step_data)
            
            # Set step to completed
            st.session_state.workflow_steps[i]['status'] = 'completed'
            st.rerun()
            time.sleep(0.5)
        
        # Mark analysis complete
        st.session_state.progressive_results['success'] = True
        st.session_state.analysis_results = st.session_state.progressive_results
        st.session_state.analysis_running = False
        
        # Set chatbot context
        if 'chatbot_context' in st.session_state.progressive_results:
            st.session_state.chatbot_context = st.session_state.progressive_results['chatbot_context']
        
        st.rerun()
        
    except Exception as e:
        st.session_state.analysis_running = False
        st.error(f"Analysis failed: {str(e)}")
        st.rerun()

def create_chatbot_window_modal():
    """Create chatbot modal window"""
    modal_id = str(uuid.uuid4())
    
    # Define prompt categories
    prompt_categories = {
        "Medical Records": [
            "What diagnoses were found in the medical records?",
            "What medical procedures were performed?", 
            "List all ICD-10 diagnosis codes found"
        ],
        "Medications": [
            "What medications is this patient taking?",
            "What NDC codes were identified?",
            "Is this person at risk of polypharmacy?"
        ],
        "Risk Assessment": [
            "What is the heart attack risk and explain why?",
            "Risk of developing chronic diseases?",
            "Hospitalization likelihood in next 6-12 months?"
        ],
        "Visualizations": [
            "Create a medication timeline chart",
            "Generate a comprehensive risk dashboard", 
            "Show me a pie chart of medications"
        ]
    }
    
    # Generate prompt buttons
    prompt_html = ""
    for category, prompts in prompt_categories.items():
        prompt_html += f"""
        <div class="sidebar-category">
            <h4 style="color: white; margin: 0.5rem 0;">{category}</h4>
        """
        for prompt in prompts:
            safe_prompt = prompt.replace("'", "\\'").replace('"', '\\"')
            prompt_html += f"""
            <button class="category-prompt-btn" onclick="setPrompt('{safe_prompt}')">
                {prompt}
            </button>
            """
        prompt_html += "</div>"
    
    # Create modal HTML
    modal_html = f"""
    <div id="chatbot-modal-{modal_id}" class="chatbot-modal" style="display: none;">
        <div class="chatbot-modal-content">
            <div class="chatbot-modal-sidebar">
                <h3 style="color: white; text-align: center; margin-bottom: 2rem;">🏥 Medical Assistant</h3>
                {prompt_html}
                <button onclick="closeModal('{modal_id}')" style="
                    background: rgba(220, 53, 69, 0.8);
                    color: white;
                    border: none;
                    padding: 0.8rem 1.5rem;
                    border-radius: 8px;
                    cursor: pointer;
                    width: 100%;
                    margin-top: 2rem;
                    font-weight: 600;
                ">Close Window</button>
            </div>
            <div class="chatbot-modal-main">
                <div style="border-bottom: 2px solid #e9ecef; padding-bottom: 1rem; margin-bottom: 2rem;">
                    <h2 style="color: #2c3e50; margin: 0;">Healthcare AI Assistant</h2>
                    <p style="color: #6c757d; margin: 0.5rem 0 0 0;">Advanced medical analysis with graph generation</p>
                </div>
                <div id="chat-container-{modal_id}" style="
                    flex: 1;
                    background: #f8f9fa;
                    border-radius: 15px;
                    padding: 1.5rem;
                    margin-bottom: 1rem;
                    overflow-y: auto;
                    max-height: 400px;
                ">
                    <div style="text-align: center; padding: 2rem; color: #6c757d;">
                        <div style="font-size: 3rem; margin-bottom: 1rem;">🏥</div>
                        <h4>Welcome to Healthcare AI Assistant!</h4>
                        <p>Use sidebar prompts or type your question below.</p>
                    </div>
                </div>
                <div style="display: flex; gap: 1rem;">
                    <input 
                        type="text" 
                        id="chat-input-{modal_id}" 
                        placeholder="Ask about health data..."
                        style="
                            flex: 1;
                            padding: 1rem;
                            border: 2px solid #e9ecef;
                            border-radius: 10px;
                            font-size: 1rem;
                        "
                        onkeypress="if(event.key==='Enter') sendMessage('{modal_id}')"
                    />
                    <button 
                        onclick="sendMessage('{modal_id}')"
                        style="
                            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
                            color: white;
                            border: none;
                            padding: 1rem 2rem;
                            border-radius: 10px;
                            font-weight: 600;
                            cursor: pointer;
                        "
                    >Send</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
    function openModal(modalId) {{
        document.getElementById('chatbot-modal-' + modalId).style.display = 'flex';
    }}
    
    function closeModal(modalId) {{
        document.getElementById('chatbot-modal-' + modalId).style.display = 'none';
    }}
    
    function setPrompt(prompt) {{
        const activeModal = document.querySelector('.chatbot-modal[style*="flex"]');
        if (activeModal) {{
            const modalId = activeModal.id.split('-')[2];
            const input = document.getElementById('chat-input-' + modalId);
            if (input) {{
                input.value = prompt;
            }}
        }}
    }}
    
    function sendMessage(modalId) {{
        const input = document.getElementById('chat-input-' + modalId);
        const chatContainer = document.getElementById('chat-container-' + modalId);
        const message = input.value.trim();
        
        if (message) {{
            // Add user message
            const userDiv = document.createElement('div');
            userDiv.innerHTML = `
                <div style="background: #007bff; color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; margin-left: 20%;">
                    <strong>You:</strong> ${{message}}
                </div>
            `;
            chatContainer.appendChild(userDiv);
            
            // Add response
            const responseDiv = document.createElement('div');
            responseDiv.innerHTML = `
                <div style="background: #28a745; color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; margin-right: 20%;">
                    <strong>Assistant:</strong> I received your question: "${{message}}". This chatbot will be integrated with your health analysis backend for real-time responses and graph generation.
                </div>
            `;
            chatContainer.appendChild(responseDiv);
            
            input.value = '';
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }}
    }}
    </script>
    """
    
    return modal_html, modal_id

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
    if 'calculated_age' not in st.session_state:
        st.session_state.calculated_age = None
    if 'workflow_steps' not in st.session_state:
        st.session_state.workflow_steps = [
            {'name': 'API Fetch', 'status': 'pending', 'description': 'Fetching comprehensive claims data', 'icon': '⚡'},
            {'name': 'Deidentification', 'status': 'pending', 'description': 'Advanced PII removal with clinical preservation', 'icon': '🔒'},
            {'name': 'Field Extraction', 'status': 'pending', 'description': 'Extracting medical and pharmacy fields', 'icon': '🚀'},
            {'name': 'Entity Extraction', 'status': 'pending', 'description': 'Advanced health entity identification', 'icon': '🎯'},
            {'name': 'Health Trajectory', 'status': 'pending', 'description': 'Comprehensive predictive health analysis', 'icon': '📈'},
            {'name': 'Heart Risk Prediction', 'status': 'pending', 'description': 'ML-based cardiovascular assessment', 'icon': '❤️'},
            {'name': 'Chatbot Initialization', 'status': 'pending', 'description': 'AI assistant with graph generation', 'icon': '💬'}
        ]
    if 'progressive_results' not in st.session_state:
        st.session_state.progressive_results = {}
    if 'selected_prompt' not in st.session_state:
        st.session_state.selected_prompt = None
    if 'show_chatbot_modal' not in st.session_state:
        st.session_state.show_chatbot_modal = False

def reset_workflow():
    """Reset workflow to initial state"""
    st.session_state.workflow_steps = [
        {'name': 'API Fetch', 'status': 'pending', 'description': 'Fetching comprehensive claims data', 'icon': '⚡'},
        {'name': 'Deidentification', 'status': 'pending', 'description': 'Advanced PII removal with clinical preservation', 'icon': '🔒'},
        {'name': 'Field Extraction', 'status': 'pending', 'description': 'Extracting medical and pharmacy fields', 'icon': '🚀'},
        {'name': 'Entity Extraction', 'status': 'pending', 'description': 'Advanced health entity identification', 'icon': '🎯'},
        {'name': 'Health Trajectory', 'status': 'pending', 'description': 'Comprehensive predictive health analysis', 'icon': '📈'},
        {'name': 'Heart Risk Prediction', 'status': 'pending', 'description': 'ML-based cardiovascular assessment', 'icon': '❤️'},
        {'name': 'Chatbot Initialization', 'status': 'pending', 'description': 'AI assistant with graph generation', 'icon': '💬'}
    ]
    st.session_state.progressive_results = {}

# Initialize session state
initialize_session_state()

# Main Title
st.markdown('<h1 class="main-header">Deep Research Health Agent 2.0</h1>', unsafe_allow_html=True)

# Display import status
if not AGENT_AVAILABLE:
    st.markdown(f'<div class="status-error">Failed to import Health Agent: {import_error}</div>', unsafe_allow_html=True)
    st.info("🔄 Running in demo mode with mock analysis capabilities")

# NEW: CHATBOT WINDOW BUTTON - Displayed prominently when available
if (st.session_state.analysis_results and 
    st.session_state.analysis_results.get("chatbot_ready", False)):
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Open AI Medical Assistant Chat Window", key="open_chatbot_window"):
            st.session_state.show_chatbot_modal = True
            st.rerun()
    st.markdown("---")

# Display chatbot modal if requested
if st.session_state.show_chatbot_modal:
    modal_html, modal_id = create_chatbot_window_modal()
    st.markdown(modal_html, unsafe_allow_html=True)
    
    # Auto-open the modal
    st.markdown(f"""
    <script>
    setTimeout(function() {{
        openModal('{modal_id}');
    }}, 100);
    </script>
    """, unsafe_allow_html=True)
    
    # Add close button in main interface
    if st.button("❌ Close Chatbot Modal", key="close_modal"):
        st.session_state.show_chatbot_modal = False
        st.rerun()

# ENHANCED SIDEBAR CHATBOT
with st.sidebar:
    if st.session_state.analysis_results and st.session_state.analysis_results.get("chatbot_ready", False):
        st.title("🏥 Medical Assistant")
        st.markdown("---")
        
        # Quick prompts
        st.markdown("**🔥 Quick Questions:**")
        
        # Categorized prompts
        prompt_categories = {
            "📋 Medical Records": [
                "What diagnoses were found?",
                "List ICD-10 codes",
                "Medical procedures performed"
            ],
            "💊 Medications": [
                "Current medications?",
                "NDC codes identified?",
                "Polypharmacy risk?"
            ],
            "📊 Risk Assessment": [
                "Heart attack risk?",
                "Chronic disease risk?",
                "Hospitalization likelihood?"
            ],
            "📈 Visualizations": [
                "Create medication chart",
                "Show risk dashboard",
                "Generate health overview"
            ]
        }
        
        for category, prompts in prompt_categories.items():
            with st.expander(category, expanded=False):
                for i, prompt in enumerate(prompts):
                    if st.button(prompt, key=f"sidebar_prompt_{category}_{i}", use_container_width=True):
                        st.session_state.selected_prompt = prompt
                        st.rerun()
        
        # Chat input
        st.markdown("---")
        user_question = st.chat_input("Ask a question...")
        
        # Handle chat input
        if user_question or st.session_state.selected_prompt:
            question = user_question or st.session_state.selected_prompt
            
            if question:
                # Add user message
                st.session_state.chatbot_messages.append({"role": "user", "content": question})
                
                # Generate response
                try:
                    if AGENT_AVAILABLE and st.session_state.agent:
                        response = st.session_state.agent.chat_with_data(
                            question, 
                            st.session_state.chatbot_context, 
                            st.session_state.chatbot_messages
                        )
                    else:
                        # Mock response
                        response = f"Analysis for: '{question}'\n\nThis is a mock response. In the full implementation, this would provide detailed health insights and generate visualizations based on your medical data."
                    
                    st.session_state.chatbot_messages.append({"role": "assistant", "content": response})
                    
                    # Clear selected prompt
                    st.session_state.selected_prompt = None
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Chat error: {str(e)}")
        
        # Chat history
        st.markdown("---")
        st.markdown("**💬 Chat History:**")
        
        if st.session_state.chatbot_messages:
            # Display recent messages
            for message in reversed(st.session_state.chatbot_messages[-8:]):
                with st.chat_message(message["role"]):
                    # Handle matplotlib code in responses
                    matplotlib_code = extract_matplotlib_code(message["content"])
                    if matplotlib_code and message["role"] == "assistant":
                        # Display text without code
                        text_content = message["content"]
                        for pattern in [f"```python\n{matplotlib_code}\n```", f"```\n{matplotlib_code}\n```"]:
                            text_content = text_content.replace(pattern, "")
                        
                        if text_content.strip():
                            st.write(text_content.strip())
                        
                        # Generate graph
                        with st.spinner("Generating graph..."):
                            img_buffer = execute_matplotlib_code(matplotlib_code)
                            if img_buffer:
                                st.image(img_buffer, use_container_width=True)
                            else:
                                st.error("Graph generation failed")
                    else:
                        st.write(message["content"])
        else:
            st.info("Start chatting! Use prompts above or type a question.")
        
        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chatbot_messages = []
            st.rerun()
    
    else:
        st.title("🏥 Medical Assistant")
        st.info("Available after health analysis completes")
        
        st.markdown("---")
        st.markdown("**🎯 What you can ask:**")
        st.markdown("• Medical records and diagnoses")
        st.markdown("• Medication analysis and interactions") 
        st.markdown("• Risk assessments and predictions")
        st.markdown("• Health visualizations and charts")
        st.markdown("• Clinical decision support")
        
        st.markdown("---")
        st.markdown("**✨ Enhanced Features:**")
        st.markdown("• Real-time graph generation")
        st.markdown("• Categorized prompt system")
        st.markdown("• Advanced health analytics")
        st.markdown("• Professional clinical insights")
        
        # Show loading indicator during analysis
        if st.session_state.analysis_running:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: rgba(40, 167, 69, 0.1); border-radius: 10px; margin: 2rem 0;">
                <div class="loading-spinner"></div>
                <h4 style="color: #28a745;">Preparing AI Assistant...</h4>
                <p style="color: #6c757d;">Loading healthcare analysis capabilities</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show sample dashboard during loading
            try:
                loading_fig = create_sample_dashboard()
                if loading_fig:
                    st.plotly_chart(loading_fig, use_container_width=True, key="loading_dashboard")
            except Exception as e:
                st.info("Analytics dashboard loading...")

# 1. PATIENT INFORMATION SECTION
st.markdown("""
<div class="section-box">
    <div class="section-title">🏥 Patient Information</div>
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
        
        # Submit button
        submitted = st.form_submit_button(
            "🚀 Run Enhanced Healthcare Analysis", 
            use_container_width=True,
            disabled=st.session_state.analysis_running,
            type="primary"
        )

# Handle form submission with PROGRESSIVE ANALYSIS
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
        # Reset workflow and start analysis
        reset_workflow()
        st.session_state.analysis_running = True
        st.session_state.analysis_results = None
        st.session_state.chatbot_messages = []
        st.session_state.chatbot_context = None
        st.session_state.calculated_age = None
        
        # Initialize agent if available
        if AGENT_AVAILABLE:
            try:
                config = Config()
                st.session_state.config = config
                st.session_state.agent = HealthAnalysisAgent(config)
                st.success("✅ Health Agent initialized successfully")
            except Exception as e:
                st.error(f"Failed to initialize agent: {str(e)}")
                st.session_state.analysis_running = False
                st.stop()
        else:
            st.warning("🔄 Using mock analysis (Health Agent not available)")
        
        # Run progressive analysis
        with st.spinner("🚀 Starting Enhanced Healthcare Analysis..."):
            try:
                if AGENT_AVAILABLE and st.session_state.agent:
                    # Run real analysis with progressive updates
                    # This would be your actual langgraph workflow
                    results = st.session_state.agent.run_analysis(patient_data)
                    st.session_state.analysis_results = results
                    st.session_state.analysis_running = False
                    
                    # Set chatbot context
                    if results.get('chatbot_ready'):
                        st.session_state.chatbot_context = results.get('chatbot_context')
                else:
                    # Run mock analysis
                    run_progressive_analysis(patient_data)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.session_state.analysis_running = False

# Display workflow animation while running or when completed
if st.session_state.analysis_running or st.session_state.analysis_results:
    display_workflow()
    
    # Show loading message during processing
    if st.session_state.analysis_running:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0;
                    border: 2px solid #28a745;">
            <div class="loading-spinner"></div>
            <h4 style="color: #28a745;">🔬 Enhanced Healthcare Analysis in Progress...</h4>
            <p style="color: #6c757d;">Processing comprehensive health data with AI-powered analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample dashboard during processing
        try:
            loading_fig = create_sample_dashboard()
            if loading_fig:
                st.plotly_chart(loading_fig, use_container_width=True, key="processing_dashboard")
        except Exception as e:
            st.info("📊 Health analytics processing...")

# PROGRESSIVE RESULTS SECTION - Sections become available as workflow steps complete
if st.session_state.analysis_results and not st.session_state.analysis_running:
    results = st.session_state.analysis_results

    # 1. CLAIMS DATA - Available after API Fetch + Deidentification
    claims_availability = get_section_availability('claims_data')
if claims_availability == 'disabled':
    st.markdown("### 📊 Claims Data")
    st.info("⏳ This section will be available after API Fetch and Deidentification steps complete")
else:
    with st.expander("📊 Claims Data", expanded=False):
        st.markdown("""
        <div class="section-box section-available">
            <div class="section-title">Claims Data Analysis</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Rest of your expander content here...
        deidentified_data = safe_get(results, 'deidentified_data', {})
        api_outputs = safe_get(results, 'api_outputs', {})
        
        if deidentified_data or api_outputs:
            tab1, tab2, tab3 = st.tabs(["Medical Claims", "Pharmacy Claims", "MCID Data"])

            with tab1:
                st.success("✅ Medical claims data processed")
                if deidentified_data.get('medical'):
                    with st.expander("View Medical Claims JSON", expanded=False):
                        st.json(deidentified_data['medical'])
                else:
                    st.json({"status": "Medical claims data loaded", "records": "Available for analysis"})
            
            with tab2:
                st.success("✅ Pharmacy claims data processed") 
                if deidentified_data.get('pharmacy'):
                    with st.expander("View Pharmacy Claims JSON", expanded=False):
                        st.json(deidentified_data['pharmacy'])
                else:
                    st.json({"status": "Pharmacy claims data loaded", "records": "Available for analysis"})
            
            with tab3:
                st.success("✅ MCID data processed")
                if api_outputs.get('mcid'):
                    with st.expander("View MCID JSON Data", expanded=False):
                        st.json(api_outputs['mcid'])
                else:
                    st.json({"status": "MCID data loaded", "matches": "Available for analysis"})
        else:
            st.error("No claims data available")

    # 2. CLAIMS DATA ANALYSIS - Available after Field Extraction
    code_analysis_availability = get_section_availability('code_analysis')
    
    with st.expander(
        "🔬 Claims Data Analysis", 
        expanded=False,
        disabled=(code_analysis_availability == 'disabled')
    ):
        if code_analysis_availability == 'disabled':
            st.info("⏳ This section will be available after Field Extraction step completes")
        else:
            st.markdown("""
            <div class="section-box section-available">
                <div class="section-title">Code Meanings Analysis</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display code analysis
            tab1, tab2 = st.tabs(["Medical Codes", "Pharmacy Codes"])
            
            with tab1:
                st.success("✅ Medical codes analyzed")
                
                # Mock metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("""
                    <div class="metric-summary-box">
                        <h3 style="margin: 0; color: #007bff; font-size: 2rem;">12</h3>
                        <p style="margin: 0; color: #6c757d; font-weight: 600;">Service Codes</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown("""
                    <div class="metric-summary-box">
                        <h3 style="margin: 0; color: #28a745; font-size: 2rem;">8</h3>
                        <p style="margin: 0; color: #6c757d; font-weight: 600;">ICD-10 Codes</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown("""
                    <div class="metric-summary-box">
                        <h3 style="margin: 0; color: #dc3545; font-size: 2rem;">25</h3>
                        <p style="margin: 0; color: #6c757d; font-weight: 600;">Medical Records</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown("""
                    <div class="metric-summary-box">
                        <h3 style="margin: 0; color: #28a745; font-size: 1.5rem;">SUCCESS</h3>
                        <p style="margin: 0; color: #6c757d; font-weight: 600;">Status</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Sample data table
                sample_medical_data = {
                    "ICD-10 Code": ["I10", "E11.9", "E78.5", "Z51.11"],
                    "Code Meaning": ["Essential Hypertension", "Type 2 Diabetes", "Hyperlipidemia", "Chemotherapy"],
                    "Claim Date": ["2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05"],
                    "Frequency": [3, 2, 1, 1]
                }
                
                df_medical = pd.DataFrame(sample_medical_data)
                st.dataframe(df_medical, use_container_width=True, hide_index=True)
                
            with tab2:
                st.success("✅ Pharmacy codes analyzed")
                
                # Sample pharmacy data
                sample_pharmacy_data = {
                    "NDC Code": ["0093-0058-01", "0071-0222-23", "0071-0156-23"],
                    "Medication": ["Metformin", "Lisinopril", "Atorvastatin"],
                    "Fill Date": ["2024-01-10", "2024-01-15", "2024-02-01"],
                    "Frequency": [4, 3, 2]
                }
                
                df_pharmacy = pd.DataFrame(sample_pharmacy_data)
                st.dataframe(df_pharmacy, use_container_width=True, hide_index=True)

    # 3. ENTITY EXTRACTION - Available after Entity Extraction
    entity_availability = get_section_availability('entity_extraction')
    
    with st.expander(
        "🎯 Entity Extraction", 
        expanded=False,
        disabled=(entity_availability == 'disabled')
    ):
        if entity_availability == 'disabled':
            st.info("⏳ This section will be available after Entity Extraction step completes")
        else:
            st.markdown("""
            <div class="section-box section-available">
                <div class="section-title">Health Entity Analysis</div>
            </div>
            """, unsafe_allow_html=True)
            
            entity_extraction = safe_get(results, 'entity_extraction', {})
            if entity_extraction:
                st.success("✅ Health entities extracted successfully")
                
                # Entity cards
                st.markdown('<div class="entity-grid">', unsafe_allow_html=True)
                
                entities_data = [
                    ('🩺', 'Diabetes Status', entity_extraction.get('diabetics', 'unknown'), 'diabetics'),
                    ('👥', 'Age Group', entity_extraction.get('age_group', 'unknown'), 'age_group'),
                    ('🚬', 'Smoking Status', entity_extraction.get('smoking', 'unknown'), 'smoking'),
                    ('🍷', 'Alcohol Use', entity_extraction.get('alcohol', 'unknown'), 'alcohol'),
                    ('💓', 'Blood Pressure', entity_extraction.get('blood_pressure', 'unknown'), 'blood_pressure')
                ]
                
                cols = st.columns(len(entities_data))
                
                for i, (col, (icon, label, value, key)) in enumerate(zip(cols, entities_data)):
                    with col:
                        # Determine status class
                        if key in ['diabetics', 'smoking'] and value == 'yes':
                            status_class = 'positive'
                        elif key in ['diabetics', 'smoking'] and value == 'no':
                            status_class = 'negative'
                        else:
                            status_class = 'unknown' if value == 'unknown' else 'negative'
                        
                        st.markdown(f"""
                        <div class="entity-card">
                            <span class="entity-icon">{icon}</span>
                            <div class="entity-label">{label}</div>
                            <div class="entity-value {status_class}">{value.upper()}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Entity extraction data not available")

    # 4. HEALTH TRAJECTORY - Available after Health Trajectory step
    trajectory_availability = get_section_availability('health_trajectory')
    
    with st.expander(
        "📈 Health Trajectory", 
        expanded=False,
        disabled=(trajectory_availability == 'disabled')
    ):
        if trajectory_availability == 'disabled':
            st.info("⏳ This section will be available after Health Trajectory step completes")
        else:
            st.markdown("""
            <div class="trajectory-container">
                <div class="section-title">Predictive Health Analysis</div>
            </div>
            """, unsafe_allow_html=True)
            
            health_trajectory = safe_get(results, 'health_trajectory', '')
            if health_trajectory:
                st.success("✅ Health trajectory analysis completed")
                st.markdown("### 📊 Comprehensive Health Analysis")
                st.write(health_trajectory)
                
                # Add summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Analysis Type", "Comprehensive")
                with col2:
                    st.metric("Medical Records", "25")
                with col3:
                    st.metric("Pharmacy Records", "18")
                with col4:
                    st.metric("Risk Factors", "5")
            else:
                st.warning("Health trajectory analysis not available")

    # 5. HEART ATTACK RISK PREDICTION - Available after Heart Risk Prediction step
    heart_risk_availability = get_section_availability('heart_risk')
    
    with st.expander(
        "❤️ Heart Attack Risk Prediction", 
        expanded=False,
        disabled=(heart_risk_availability == 'disabled')
    ):
        if heart_risk_availability == 'disabled':
            st.info("⏳ This section will be available after Heart Risk Prediction step completes")
        else:
            st.markdown("""
            <div class="heart-risk-container">
                <div class="section-title">Cardiovascular Risk Assessment</div>
            </div>
            """, unsafe_allow_html=True)
            
            heart_attack_prediction = safe_get(results, 'heart_attack_prediction', {})
            heart_attack_risk_score = safe_get(results, 'heart_attack_risk_score', 0.0)
            
            if heart_attack_prediction:
                st.success("✅ Heart attack risk assessment completed")
                
                combined_display = heart_attack_prediction.get("combined_display", "Heart Disease Risk: 25%")
                risk_category = heart_attack_prediction.get("risk_category", "Medium Risk")
                
                # Clean up display text
                if "Confidence:" in combined_display:
                    combined_display = combined_display.split("Confidence:")[0].strip()
                combined_display = combined_display.replace("|", "").strip()
                
                # Risk display
                st.markdown("### 🏥 ML Model Prediction Results")
                
                with st.container():
                    st.markdown("""
                    <div style="text-align: center; margin: 1rem 0;">
                        <h3 style="color: #2c3e50; font-size: 1.8rem;">Heart Disease Risk Assessment</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Extract risk percentage
                    risk_percentage = ""
                    if "%" in combined_display:
                        percentage_match = re.search(r'(\d+\.?\d*%)', combined_display)
                        if percentage_match:
                            risk_percentage = percentage_match.group(1)
                    
                    if risk_percentage:
                        st.markdown(f"""
                        <div style="text-align: center; margin: 2rem 0;">
                            <div style="font-size: 4rem; font-weight: 800; color: #dc3545;">
                                {risk_percentage}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.info("🔬 Advanced machine learning analysis provides cardiovascular risk assessment based on comprehensive health data.")
                    
                    # Risk category display
                    if risk_category == 'High Risk':
                        st.error(f"**🚨 Risk Category: {risk_category}**")
                    elif risk_category == 'Medium Risk':
                        st.warning(f"**⚠️ Risk Category: {risk_category}**")
                    else:
                        st.success(f"**✅ Risk Category: {risk_category}**")
            else:
                st.warning("Heart attack risk prediction not available")

# MAIN AREA CHAT INTERFACE (when not using modal)
if (st.session_state.analysis_results and 
    st.session_state.analysis_results.get("chatbot_ready", False) and 
    st.session_state.chatbot_context and
    not st.session_state.show_chatbot_modal):
    
    st.markdown("---")
    st.markdown("## 💬 AI Medical Assistant")
    st.markdown("🎯 Ask questions about the health analysis, request visualizations, or get clinical insights.")
    
    # Main chat interface
    chat_input = st.chat_input("🏥 Ask me anything about the health data...")
    
    if chat_input:
        # Add user message
        st.session_state.chatbot_messages.append({"role": "user", "content": chat_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(chat_input)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing your question..."):
                try:
                    if AGENT_AVAILABLE and st.session_state.agent:
                        chatbot_response = st.session_state.agent.chat_with_data(
                            chat_input, 
                            st.session_state.chatbot_context, 
                            st.session_state.chatbot_messages
                        )
                    else:
                        # Mock response with potential matplotlib code
                        if "chart" in chat_input.lower() or "graph" in chat_input.lower() or "plot" in chat_input.lower():
                            chatbot_response = f"""
Based on your request: "{chat_input}", here's a comprehensive analysis:

This would provide detailed insights about your health data. Here's a sample visualization:

```python
import matplotlib.pyplot as plt
import numpy as np

# Create health dashboard
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Healthcare Analysis Dashboard', fontsize=16, fontweight='bold')

# Risk factors
risk_data = list(risk_factors.values())
risk_names = list(risk_factors.keys())
colors = ['#28a745' if x == 0 else '#dc3545' for x in risk_data]
ax1.bar(risk_names, risk_data, color=colors)
ax1.set_title('Risk Factors', fontweight='bold')
ax1.set_ylabel('Risk Level')

# Heart risk meter
ax2.barh(['Heart Risk'], [heart_risk_score], color='#dc3545', alpha=0.7)
ax2.set_xlim(0, 1)
ax2.set_title(f'Heart Risk: {{:.1%}}'.format(heart_risk_score), fontweight='bold')

# Medications
if len(medication_list) > 0:
    ax3.barh(medication_list[:5], range(1, len(medication_list[:5])+1), color='#007bff')
    ax3.set_title('Medications', fontweight='bold')
else:
    ax3.text(0.5, 0.5, 'No medication data', ha='center', va='center', transform=ax3.transAxes)

# Health summary
summary_text = f'Patient Age: {{}} years\\nDiabetes: {{}}\\nSmoking: {{}}'.format(patient_age, diabetes_status, smoking_status)
ax4.text(0.1, 0.5, summary_text, fontsize=12, transform=ax4.transAxes, verticalalignment='center')
ax4.set_title('Patient Summary', fontweight='bold')
ax4.axis('off')

plt.tight_layout()
```

This visualization shows your comprehensive health analysis including risk factors, cardiovascular assessment, medication profile, and overall health summary.
                            """
                        else:
                            chatbot_response = f"""
Thank you for your question: "{chat_input}"

Based on the healthcare analysis data available, I can provide insights about:

🩺 **Medical Records**: Comprehensive analysis of diagnoses, procedures, and clinical indicators
💊 **Medications**: Current prescriptions, NDC codes, and therapeutic classifications  
📊 **Risk Assessment**: Heart attack risk, chronic disease probability, and hospitalization likelihood
📈 **Health Trajectory**: Predictive modeling and health progression analysis
🎯 **Clinical Insights**: Evidence-based recommendations and care gap identification

This is a demonstration response. In the full implementation with your health_agent_core backend, this would provide detailed, personalized medical insights based on your actual claims data.

Would you like me to create a specific visualization or provide more details about any particular aspect of the health analysis?
                            """
                    
                    # Check for matplotlib code and handle graphs
                    matplotlib_code = extract_matplotlib_code(chatbot_response)
                    
                    if matplotlib_code:
                        # Display text content without code
                        text_content = chatbot_response
                        for pattern in [f"```python\n{matplotlib_code}\n```", f"```\n{matplotlib_code}\n```"]:
                            text_content = text_content.replace(pattern, "")
                        
                        if text_content.strip():
                            st.write(text_content.strip())
                        
                        # Execute and display graph
                        with st.spinner("📊 Generating visualization..."):
                            try:
                                img_buffer = execute_matplotlib_code(matplotlib_code)
                                if img_buffer:
                                    st.image(img_buffer, use_container_width=True, caption="🎨 Generated Healthcare Visualization")
                                    st.success("✅ Graph generated successfully!")
                                else:
                                    st.error("❌ Failed to generate graph")
                                    st.info("💡 Try asking for: 'Create a simple bar chart' or 'Show risk factors chart'")
                            except Exception as graph_error:
                                st.error(f"Graph generation error: {str(graph_error)}")
                    else:
                        # No matplotlib code, just display the response
                        st.write(chatbot_response)
                    
                    # Add assistant response to messages
                    st.session_state.chatbot_messages.append({"role": "assistant", "content": chatbot_response})
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chatbot_messages.append({"role": "assistant", "content": error_msg})

# Auto-refresh during analysis
if st.session_state.analysis_running:
    time.sleep(1)
    st.rerun()

# Status messages
if st.session_state.analysis_results:
    if st.session_state.analysis_results.get('success'):
        st.success("🎉 Healthcare analysis completed successfully!")
        if st.session_state.analysis_results.get('chatbot_ready'):
            st.info("💬 AI Medical Assistant is now available in the sidebar and via the chat window button above!")
    else:
        st.error("❌ Analysis encountered errors")

if __name__ == "__main__":
    pass
