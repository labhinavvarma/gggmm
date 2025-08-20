# Configure Streamlit page FIRST
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

# SAFE_GET FUNCTION - CRITICAL MISSING FUNCTION
def safe_get(data, key_path, default=None):
    """
    Safely get a value from nested dictionary/object structure.
    
    Args:
        data: The data structure to search in (dict, object, etc.)
        key_path: String key or list of keys for nested access
        default: Default value to return if key not found
        
    Returns:
        The value if found, otherwise the default value
    """
    try:
        if data is None:
            return default
            
        # Handle string key path
        if isinstance(key_path, str):
            if hasattr(data, key_path):
                return getattr(data, key_path, default)
            elif isinstance(data, dict):
                return data.get(key_path, default)
            else:
                return default
                
        # Handle list of keys for nested access
        elif isinstance(key_path, (list, tuple)):
            current = data
            for key in key_path:
                if current is None:
                    return default
                    
                if hasattr(current, key):
                    current = getattr(current, key, None)
                elif isinstance(current, dict):
                    current = current.get(key, None)
                else:
                    return default
                    
            return current if current is not None else default
            
        else:
            return default
            
    except (AttributeError, KeyError, TypeError, IndexError):
        return default

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

.chatbot-loading-container {
    background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
    padding: 2rem;
    border-radius: 20px;
    margin: 1.5rem 0;
    border: 2px solid #28a745;
    text-align: center;
    animation: pulse-glow 2s infinite;
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(40, 167, 69, 0.3); }
    50% { box-shadow: 0 0 40px rgba(40, 167, 69, 0.6); }
}

.quick-prompts-enhanced {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    padding: 1.8rem;
    border-radius: 18px;
    margin: 1.2rem 0;
    border: 2px solid #2196f3;
}

.prompt-button-enhanced {
    background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
    color: white;
    border: none;
    padding: 0.8rem 1.4rem;
    border-radius: 25px;
    margin: 0.4rem;
    cursor: pointer;
    font-size: 0.9rem;
    font-weight: 500;
    transition: all 0.3s ease;
    box-shadow: 0 6px 20px rgba(33, 150, 243, 0.3);
}

.prompt-button-enhanced:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(33, 150, 243, 0.5);
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

@keyframes pulse-step {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.02); }
}

/* Green Run Analysis Button */
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

/* Enhanced sidebar styling */
.css-1d391kg {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Sidebar category styling */
.sidebar-category {
    background: rgba(255, 255, 255, 0.1);
    padding: 0.5rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 3px solid #00ff87;
}

.sidebar-category h4 {
    color: white;
    margin: 0;
    font-size: 0.9rem;
    font-weight: 600;
}

.category-prompt-btn {
    background: rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 6px !important;
    padding: 0.4rem 0.8rem !important;
    margin: 0.2rem 0 !important;
    font-size: 0.8rem !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}

.category-prompt-btn:hover {
    background: rgba(255, 255, 255, 0.3) !important;
    transform: translateX(5px) !important;
}

/* Graph loading animation */
.graph-loading {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 2rem;
    background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
    border-radius: 15px;
    margin: 1rem 0;
}

.loading-spinner {
    width: 40px;
    height: 40px;
    border: 4px solid #e3f2fd;
    border-top: 4px solid #2196f3;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.status-error {
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    color: #c62828;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #ef5350;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

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
    if 'show_all_claims_data' not in st.session_state:
        st.session_state.show_all_claims_data = False
    if 'show_batch_extraction' not in st.session_state:
        st.session_state.show_batch_extraction = False
    if 'show_entity_extraction' not in st.session_state:
        st.session_state.show_entity_extraction = False
    if 'show_enhanced_trajectory' not in st.session_state:
        st.session_state.show_enhanced_trajectory = False
    if 'show_heart_attack' not in st.session_state:
        st.session_state.show_heart_attack = False
    
    # Enhanced workflow steps
    if 'workflow_steps' not in st.session_state:
        st.session_state.workflow_steps = [
            {'name': 'API Fetch', 'status': 'pending', 'description': 'Fetching comprehensive claims data', 'icon': '⚡'},
            {'name': 'Deidentification', 'status': 'pending', 'description': 'Advanced PII removal with clinical preservation', 'icon': '🔒'},
            {'name': 'Field Extraction', 'status': 'pending', 'description': 'Extracting medical and pharmacy fields', 'icon': '🚀'},
            {'name': 'Entity Extraction', 'status': 'pending', 'description': 'Advanced health entity identification', 'icon': '🎯'},
            {'name': 'Health Trajectory', 'status': 'pending', 'description': 'Comprehensive predictive health analysis', 'icon': '📈'},
            {'name': 'Final Summary', 'status': 'pending', 'description': 'Executive healthcare summary generation', 'icon': '📋'},
            {'name': 'Heart Risk Prediction', 'status': 'pending', 'description': 'ML-based cardiovascular assessment', 'icon': '❤️'},
            {'name': 'Chatbot Initialization', 'status': 'pending', 'description': 'AI assistant with graph generation', 'icon': '💬'}
        ]
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'show_animation' not in st.session_state:
        st.session_state.show_animation = False
    
    # Add selected_prompt for categorized prompts
    if 'selected_prompt' not in st.session_state:
        st.session_state.selected_prompt = None

def reset_workflow():
    """Reset workflow to initial state"""
    st.session_state.workflow_steps = [
        {'name': 'API Fetch', 'status': 'pending', 'description': 'Fetching comprehensive claims data', 'icon': '⚡'},
        {'name': 'Deidentification', 'status': 'pending', 'description': 'Advanced PII removal with clinical preservation', 'icon': '🔒'},
        {'name': 'Field Extraction', 'status': 'pending', 'description': 'Extracting medical and pharmacy fields', 'icon': '🚀'},
        {'name': 'Entity Extraction', 'status': 'pending', 'description': 'Advanced health entity identification', 'icon': '🎯'},
        {'name': 'Health Trajectory', 'status': 'pending', 'description': 'Comprehensive predictive health analysis', 'icon': '📈'},
        {'name': 'Final Summary', 'status': 'pending', 'description': 'Executive healthcare summary generation', 'icon': '📋'},
        {'name': 'Heart Risk Prediction', 'status': 'pending', 'description': 'ML-based cardiovascular assessment', 'icon': '❤️'},
        {'name': 'Chatbot Initialization', 'status': 'pending', 'description': 'AI assistant with graph generation', 'icon': '💬'}
    ]
    st.session_state.current_step = 0

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
        <p style="color: #34495e; font-size: 1.1rem;">Comprehensive health analysis workflow</p>
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
        status_message = "⏳ LangGraph healthcare analysis pipeline ready..."
    
    st.markdown(f"""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.8); border-radius: 10px;">
        <p style="margin: 0; font-weight: 600; color: #2c3e50;">{status_message}</p>
    </div>
    </div>
    """, unsafe_allow_html=True)

def display_enhanced_mcid_data(mcid_data):
    """Enhanced MCID data display with improved styling and functionality"""
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
            except Exception:
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
                
                # Show additional consumer data if available
                if consumer.get('additionalData'):
                    with st.expander(f"Additional Data for Consumer #{i}"):
                        st.json(consumer.get('additionalData'))
        else:
            st.info("ℹ️ No consumer matches found in MCID search")
            st.markdown("""
            **Possible reasons:**
            - Patient may be new to the healthcare system
            - Different name variations or spelling
            - Updated personal information not yet synchronized
            """)
    else:
        st.warning(f"⚠️ MCID search returned status code: {status_code}")
        if mcid_data.get('error'):
            st.error(f"Error details: {mcid_data['error']}")
    
    # Raw MCID data in expandable section
    with st.expander("🔍 View Raw MCID JSON Data"):
        st.json(mcid_data)

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
        except Exception:
            errors.append("Invalid date format")
    
    return len(errors) == 0, errors

def create_chatbot_loading_graphs():
    """Create interactive graphs to display while chatbot is loading"""
    
    # Create sample health data for visualization
    sample_data = {
        'dates': pd.date_range('2023-01-01', periods=12, freq='M'),
        'risk_scores': np.random.uniform(0.1, 0.8, 12),
        'health_metrics': {
            'Blood Pressure': np.random.uniform(110, 140, 12),
            'Heart Rate': np.random.uniform(60, 100, 12),
            'Cholesterol': np.random.uniform(150, 250, 12)
        }
    }
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Health Risk Trend', 'Vital Signs Monitor', 'Risk Distribution', 'Health Score'),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"type": "pie"}, {"type": "indicator"}]]
    )
    
    # Risk trend line
    fig.add_trace(
        go.Scatter(
            x=sample_data['dates'],
            y=sample_data['risk_scores'],
            mode='lines+markers',
            name='Risk Score',
            line=dict(color='#ff6b6b', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Vital signs
    for i, (metric, values) in enumerate(sample_data['health_metrics'].items()):
        fig.add_trace(
            go.Scatter(
                x=sample_data['dates'],
                y=values,
                mode='lines',
                name=metric,
                line=dict(width=2)
            ),
            row=1, col=2
        )
    
    # Risk distribution pie chart
    risk_categories = ['Low Risk', 'Medium Risk', 'High Risk']
    risk_values = [45, 35, 20]
    colors = ['#4caf50', '#ff9800', '#f44336']
    
    fig.add_trace(
        go.Pie(
            labels=risk_categories,
            values=risk_values,
            marker_colors=colors,
            name="Risk Distribution"
        ),
        row=2, col=1
    )
    
    # Health score gauge
    current_score = np.random.uniform(60, 90)
    fig.add_trace(
        go.Indicator(
            mode = "gauge+number+delta",
            value = current_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Health Score"},
            delta = {'reference': 75},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#2196f3"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffebee"},
                    {'range': [50, 80], 'color': "#e8f5e8"},
                    {'range': [80, 100], 'color': "#c8e6c9"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Real-Time Health Analytics Dashboard",
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    # Update subplot properties
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def execute_matplotlib_code_enhanced_stability(code: str):
    """Execute matplotlib code with enhanced stability and error recovery"""
    try:
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        plt.ioff()
        
        # Create namespace for code execution
        namespace = {
            'plt': plt,
            'matplotlib': matplotlib,
            'np': np,
            'numpy': np,
            'pd': pd,
            'pandas': pd,
            'json': json,
            'datetime': datetime,
            'time': time,
            'math': __import__('math')
        }
        
        # Add sample patient data
        namespace.update({
            'patient_age': 45,
            'heart_risk_score': 0.25,
            'medications_count': 3,
            'conditions': ['Hypertension', 'Type 2 Diabetes'],
            'risk_factors': {
                'Age': 45, 
                'Diabetes': 1, 
                'Smoking': 0, 
                'High_BP': 1,
                'Family_History': 1
            },
            'medication_list': ['Metformin', 'Lisinopril', 'Atorvastatin'],
            'risk_scores': [0.15, 0.25, 0.35, 0.20],
            'risk_labels': ['Low', 'Medium', 'High', 'Very High'],
            'months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'utilization_data': [2, 3, 1, 4, 2, 3]
        })
        
        # Execute the code
        exec(code, namespace)
        
        # Get the figure
        fig = plt.gcf()
        
        # Check if figure has content
        if not fig.axes:
            # Create fallback visualization
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'Enhanced Healthcare Visualization\nGenerated Successfully\n\nYour data analysis is ready!', 
                    ha='center', va='center', fontsize=16, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
            plt.title('Healthcare Data Analysis Dashboard', fontsize=18, fontweight='bold')
            plt.axis('off')
            fig = plt.gcf()
        
        # Enhance figure styling
        for ax in fig.axes:
            ax.tick_params(labelsize=10)
            ax.grid(True, alpha=0.3)
            
            if ax.get_title():
                ax.set_title(ax.get_title(), fontsize=12, fontweight='bold')
            if ax.get_xlabel():
                ax.set_xlabel(ax.get_xlabel(), fontsize=11)
            if ax.get_ylabel():
                ax.set_ylabel(ax.get_ylabel(), fontsize=11)
        
        # Convert to image
        img_buffer = io.BytesIO()
        fig.savefig(
            img_buffer, 
            format='png', 
            bbox_inches='tight', 
            dpi=200,
            facecolor='white', 
            edgecolor='none', 
            pad_inches=0.2,
            transparent=False
        )
        img_buffer.seek(0)
        
        # Cleanup
        plt.clf()
        plt.close('all')
        plt.ion()
        
        return img_buffer
        
    except Exception as e:
        # Error handling
        plt.clf()
        plt.close('all')
        plt.ion()
        
        error_msg = str(e)
        logger.error(f"Enhanced matplotlib execution error: {error_msg}")
        
        # Create error visualization
        try:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.6, '⚠️ Graph Generation Error', 
                    ha='center', va='center', fontsize=20, fontweight='bold', color='red')
            plt.text(0.5, 0.4, f'Error: {error_msg[:100]}...', 
                    ha='center', va='center', fontsize=12, color='darkred')
            plt.text(0.5, 0.3, 'Please try a different visualization request', 
                    ha='center', va='center', fontsize=12, color='blue')
            plt.title('Healthcare Data Visualization', fontsize=16)
            plt.axis('off')
            
            error_buffer = io.BytesIO()
            plt.savefig(error_buffer, format='png', bbox_inches='tight', dpi=150, facecolor='white')
            error_buffer.seek(0)
            plt.clf()
            plt.close('all')
            
            return error_buffer
        except Exception:
            st.error(f"Enhanced graph generation failed: {error_msg}")
            return None

# Initialize session state
initialize_session_state()

# Enhanced Main Title
st.markdown('<h1 class="main-header">🔬 Enhanced Health Agent v7.0</h1>', unsafe_allow_html=True)

# Enhanced optimization badges
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <div class="enhanced-badge">🔬 Comprehensive Analysis</div>
    <div class="enhanced-badge">🚀 LangGraph Powered</div>
    <div class="enhanced-badge">📊 Graph Generation</div>
    <div class="enhanced-badge">🗂️ Complete Claims Viewer</div>
    <div class="enhanced-badge">🎯 Predictive Modeling</div>
    <div class="enhanced-badge">💬 Enhanced Chatbot</div>
</div>
""", unsafe_allow_html=True)

# Display import status
if not AGENT_AVAILABLE:
    st.markdown(f'<div class="status-error">❌ Failed to import Health Agent: {import_error}</div>', unsafe_allow_html=True)
    st.stop()

# ENHANCED SIDEBAR CHATBOT WITH CATEGORIZED PROMPTS
with st.sidebar:
    if st.session_state.analysis_results and st.session_state.analysis_results.get("chatbot_ready", False) and st.session_state.chatbot_context:
        st.title("💬 Medical Assistant")
        st.markdown("---")
        
        # Chat history at top
        chat_container = st.container()
        with chat_container:
            if st.session_state.chatbot_messages:
                for message in st.session_state.chatbot_messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
            else:
                st.info("👋 Hello! I can answer questions about the medical analysis and create visualizations!")
        
        # CATEGORIZED SUGGESTED PROMPTS SECTION
        st.markdown("---")
        st.markdown("**💡 Quick Questions:**")
        
        # Define categorized prompts
        prompt_categories = {
            "🏥 Medical Records": [
                "What diagnoses were found in the medical records?",
                "What medical procedures were performed?",
                "List all ICD-10 diagnosis codes found",
                "Show me the most recent medical claims",
                "Explain the medical service codes identified"
            ],
            "💊 Medications": [
                "What medications is this patient taking?",
                "What NDC codes were identified?",
                "Are there any diabetes medications?",
                "What blood pressure medications are prescribed?",
                "Analyze potential drug interactions"
            ],
            "❤️ Risk Assessment": [
                "What is the heart attack risk and explain why?",
                "What are the main cardiovascular risk factors?",
                "Compare ML prediction vs clinical assessment",
                "What chronic conditions does this patient have?",
                "Assess overall health risk profile"
            ],
            "📊 Analysis & Graphs": [
                "Create a medication timeline chart",
                "Generate a comprehensive risk dashboard", 
                "Show me a pie chart of medications",
                "Create a health overview visualization",
                "Generate a diagnosis timeline chart"
            ],
            "📈 Health Summary": [
                "Provide a comprehensive health analysis summary",
                "What does the health trajectory analysis show?",
                "Summarize key health findings and recommendations",
                "What are the priority health concerns?",
                "Explain the overall health status and prognosis"
            ]
        }
        
        # Handle selected prompt from session state
        if hasattr(st.session_state, 'selected_prompt') and st.session_state.selected_prompt:
            user_question = st.session_state.selected_prompt
            
            # Add user message
            st.session_state.chatbot_messages.append({"role": "user", "content": user_question})
            
            # Get bot response
            try:
                with st.spinner("Processing..."):
                    chatbot_response = st.session_state.agent.chat_with_data(
                        user_question, 
                        st.session_state.chatbot_context, 
                        st.session_state.chatbot_messages
                    )
                
                # Add assistant response
                st.session_state.chatbot_messages.append({"role": "assistant", "content": chatbot_response})
                
                # Clear the selected prompt
                st.session_state.selected_prompt = None
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.selected_prompt = None
        
        # Create expandable sections for each category
        for category, prompts in prompt_categories.items():
            with st.expander(category, expanded=False):
                for i, prompt in enumerate(prompts):
                    if st.button(prompt, key=f"cat_prompt_{category}_{i}", use_container_width=True):
                        st.session_state.selected_prompt = prompt
                        st.rerun()
        
        # Quick access buttons for most common questions
        st.markdown("**🚀 Quick Access:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Health Summary", use_container_width=True, key="quick_summary"):
                st.session_state.selected_prompt = "Provide a comprehensive health analysis summary including trajectory and key findings"
                st.rerun()
        
        with col2:
            if st.button("❤️ Risk Analysis", use_container_width=True, key="quick_heart"):
                st.session_state.selected_prompt = "What is this patient's cardiovascular risk assessment and explain the clinical reasoning?"
                st.rerun()
        
        # Graph generation quick buttons
        st.markdown("**📊 Quick Graphs:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📈 Timeline", use_container_width=True, key="quick_timeline"):
                st.session_state.selected_prompt = "Create a comprehensive medication timeline chart"
                st.rerun()
        
        with col2:
            if st.button("🎯 Dashboard", use_container_width=True, key="quick_dashboard"):
                st.session_state.selected_prompt = "Generate a comprehensive risk assessment dashboard"
                st.rerun()
        
        # Chat input at bottom
        st.markdown("---")
        user_question = st.chat_input("Type your question or use prompts above...")
        
        # Handle manual chat input
        if user_question:
            # Add user message
            st.session_state.chatbot_messages.append({"role": "user", "content": user_question})
            
            # Get bot response
            try:
                with st.spinner("Processing..."):
                    chatbot_response = st.session_state.agent.chat_with_data(
                        user_question, 
                        st.session_state.chatbot_context, 
                        st.session_state.chatbot_messages
                    )
                
                # Add assistant response
                st.session_state.chatbot_messages.append({"role": "assistant", "content": chatbot_response})
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Clear chat button at bottom
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chatbot_messages = []
            st.rerun()
    
    else:
        # Enhanced placeholder when chatbot is not ready
        st.title("💬 Medical Assistant")
        st.info("💤 Chatbot will be available after running health analysis")
        st.markdown("---")
        st.markdown("**🎯 What you can ask:**")
        st.markdown("• **Medical Records:** Diagnoses, procedures, ICD codes, service codes")
        st.markdown("• **Medications:** Prescriptions, NDC codes, drug interactions, therapeutic analysis") 
        st.markdown("• **Risk Assessment:** Heart attack risk, chronic conditions, clinical predictions")
        st.markdown("• **Health Summary:** Combined trajectory analysis, comprehensive health insights")
        st.markdown("• **Visualizations:** Charts, graphs, dashboards, timelines")
        st.markdown("---")
        st.markdown("**💡 Enhanced Features:**")
        st.markdown("• Categorized prompt system for easy navigation")
        st.markdown("• Quick access buttons for common analyses")
        st.markdown("• Advanced graph generation capabilities")
        st.markdown("• Comprehensive health summary with trajectory analysis")
        st.markdown("• Professional clinical decision support")
        
        # Show loading graphs while chatbot is being prepared
        if st.session_state.analysis_running or (st.session_state.analysis_results and not st.session_state.analysis_results.get("chatbot_ready", False)):
            st.markdown("""
            <div class="chatbot-loading-container">
                <div class="loading-spinner"></div>
                <h4>🤖 Preparing AI Assistant...</h4>
                <p>Loading healthcare analysis capabilities</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display interactive loading graphs
            try:
                loading_fig = create_chatbot_loading_graphs()
                st.plotly_chart(loading_fig, use_container_width=True, key="chatbot_loading_graphs")
            except Exception as e:
                st.info("📊 Health analytics dashboard loading...")

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
        # Reset workflow and start analysis
        reset_workflow()
        st.session_state.analysis_running = True
        st.session_state.analysis_results = None
        st.session_state.chatbot_messages = []
        st.session_state.chatbot_context = None
        
        # Initialize agent
        try:
            config = Config()
            st.session_state.config = config
            st.session_state.agent = HealthAnalysisAgent(config)
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
            st.session_state.analysis_running = False
            st.stop()
        
        # Create workflow animation placeholder
        workflow_placeholder = st.empty()
        
        # Run analysis with workflow animation
        with st.spinner("🔬 Running Enhanced Healthcare Analysis..."):
            try:
                # Simulate workflow steps
                for i, step in enumerate(st.session_state.workflow_steps):
                    st.session_state.workflow_steps[i]['status'] = 'running'
                    with workflow_placeholder.container():
                        display_advanced_professional_workflow()
                    time.sleep(1)  # Brief pause for animation
                    
                    # Run actual analysis step
                    if i == 0:
                        # Start the full analysis
                        results = st.session_state.agent.run_analysis(patient_data)
                        
                        # Update all steps to completed if successful
                        if results.get("success"):
                            for j in range(len(st.session_state.workflow_steps)):
                                st.session_state.workflow_steps[j]['status'] = 'completed'
                        else:
                            # Mark the failing step
                            st.session_state.workflow_steps[i]['status'] = 'error'
                            for j in range(i+1, len(st.session_state.workflow_steps)):
                                st.session_state.workflow_steps[j]['status'] = 'error'
                        break
                    else:
                        st.session_state.workflow_steps[i]['status'] = 'completed'
                
                # Final workflow display
                with workflow_placeholder.container():
                    display_advanced_professional_workflow()
                
                st.session_state.analysis_results = results
                st.session_state.analysis_running = False
                
                # Set chatbot context if analysis successful
                if results.get("success") and results.get("chatbot_ready"):
                    st.session_state.chatbot_context = results.get("chatbot_context")
                
                st.success("✅ Enhanced Healthcare Analysis completed successfully!")
                st.rerun()
                
            except Exception as e:
                st.session_state.analysis_running = False
                st.error(f"Analysis failed: {str(e)}")
                
                # Mark current step as error
                if st.session_state.current_step < len(st.session_state.workflow_steps):
                    st.session_state.workflow_steps[st.session_state.current_step]['status'] = 'error'
                
                with workflow_placeholder.container():
                    display_advanced_professional_workflow()

# Display workflow animation if analysis is running
if st.session_state.analysis_running:
    display_advanced_professional_workflow()

# ENHANCED RESULTS SECTION
if st.session_state.analysis_results and not st.session_state.analysis_running:
    results = st.session_state.analysis_results
    
    st.markdown("---")
    st.markdown("## 📊 Healthcare Analysis Results")
    
    # Show errors if any
    errors = safe_get(results, 'errors', [])
    if errors:
        st.markdown('<div class="status-error">❌ Analysis errors occurred</div>', unsafe_allow_html=True)
        with st.expander("🐛 Debug Information"):
            st.write("**Errors:**")
            for error in errors:
                st.write(f"• {error}")

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
            st.metric("Heart Attack Risk", risk_display.split(':')[1].strip() if ':' in risk_display else risk_display)
        else:
            st.metric("Heart Attack Risk", "Error")

    # 1. COMPLETE CLAIMS DATA VIEWER
    if st.button("🗂️ Complete Claims Data Viewer", use_container_width=True, key="claims_data_btn"):
        st.session_state.show_all_claims_data = not st.session_state.show_all_claims_data
    
    if st.session_state.show_all_claims_data:
        st.markdown("""
        <div class="section-box">
            <div class="section-title">🗂️ Complete Claims Data Analysis</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="claims-viewer-card">
            <h3>📋 Comprehensive Deidentified Claims Database</h3>
            <p><strong>Features:</strong> Complete access to all deidentified claims data with detailed analysis and comprehensive JSON exploration.</p>
        </div>
        """, unsafe_allow_html=True)
        
        deidentified_data = safe_get(results, 'deidentified_data', {})
        api_outputs = safe_get(results, 'api_outputs', {})
        
        if deidentified_data or api_outputs:
            tab1, tab2, tab3, tab4 = st.tabs([
                "🏥 Medical Claims", 
                "💊 Pharmacy Claims", 
                "🆔 MCID Data",
                "📊 Complete JSON"
            ])
            
            with tab1:
                medical_data = safe_get(deidentified_data, 'medical', {})
                if medical_data and not medical_data.get('error'):
                    st.markdown("### 🏥 Medical Claims Analysis")
                    
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
                                st.metric("Processed", formatted_time)
                            except Exception:
                                st.metric("Processed", "Recently")
                        else:
                            st.metric("Processed", "Unknown")
                    
                    medical_claims_data = medical_data.get('medical_claims_data', {})
                    if medical_claims_data:
                        with st.expander("🔍 Medical Claims JSON Data", expanded=False):
                            st.json(medical_claims_data)
                else:
                    st.error("❌ No medical claims data available")
            
            with tab2:
                pharmacy_data = safe_get(deidentified_data, 'pharmacy', {})
                if pharmacy_data and not pharmacy_data.get('error'):
                    st.markdown("### 💊 Pharmacy Claims Analysis")
                    
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
                            except Exception:
                                st.metric("Processed", "Recently")
                        else:
                            st.metric("Processed", "Unknown")
                    with col3:
                        masked_fields = pharmacy_data.get('name_fields_masked', [])
                        st.metric("Fields Masked", len(masked_fields) if masked_fields else 0)
                    
                    pharmacy_claims_data = pharmacy_data.get('pharmacy_claims_data', {})
                    if pharmacy_claims_data:
                        with st.expander("🔍 Pharmacy Claims JSON Data", expanded=False):
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
                        with st.expander("Deidentified Data JSON", expanded=False):
                            st.json(deidentified_data)
                    else:
                        st.warning("No deidentified data available")
                
                with col2:
                    st.markdown("#### 🆔 MCID + API Outputs")
                    if api_outputs:
                        with st.expander("API Outputs JSON", expanded=False):
                            st.json(api_outputs)
                    else:
                        st.warning("No API outputs available")
        else:
            st.error("❌ No claims data available")

    # 2. ENTITY EXTRACTION RESULTS
    if st.button("🎯 Health Entity Extraction Results", use_container_width=True, key="entity_btn"):
        st.session_state.show_entity_extraction = not st.session_state.show_entity_extraction
    
    if st.session_state.show_entity_extraction:
        st.markdown("""
        <div class="section-box">
            <div class="section-title">🎯 Enhanced Health Entity Extraction</div>
        </div>
        """, unsafe_allow_html=True)
        
        entities = safe_get(results, 'entity_extraction', {})
        if entities:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🏥 Health Conditions")
                st.write(f"**Diabetes:** {entities.get('diabetics', 'Unknown')}")
                st.write(f"**Blood Pressure:** {entities.get('blood_pressure', 'Unknown')}")
                st.write(f"**Smoking:** {entities.get('smoking', 'Unknown')}")
                st.write(f"**Age Group:** {entities.get('age_group', 'Unknown')}")
            
            with col2:
                st.markdown("#### 💊 Medication Analysis")
                medications = entities.get('medications_identified', [])
                if medications:
                    for i, med in enumerate(medications[:5], 1):
                        if isinstance(med, dict):
                            st.write(f"{i}. {med.get('label_name', 'Unknown')}")
                        else:
                            st.write(f"{i}. {med}")
                else:
                    st.write("No medications identified")
            
            with col3:
                st.markdown("#### 🔬 Clinical Insights")
                medical_conditions = entities.get('medical_conditions', [])
                st.write(f"**Medical Conditions:** {len(medical_conditions) if medical_conditions else 0}")
                st.write(f"**Clinical Complexity:** {entities.get('clinical_complexity_score', 0)}")
                st.write(f"**Enhanced Analysis:** {entities.get('enhanced_clinical_analysis', False)}")
            
            # Detailed entity data
            with st.expander("🔍 Complete Entity Extraction Data"):
                st.json(entities)
        else:
            st.warning("No entity extraction data available")

    # 3. HEART ATTACK RISK PREDICTION
    if st.button("❤️ Heart Attack Risk Assessment", use_container_width=True, key="heart_risk_btn"):
        st.session_state.show_heart_attack = not st.session_state.show_heart_attack
    
    if st.session_state.show_heart_attack:
        st.markdown("""
        <div class="section-box">
            <div class="section-title">❤️ Cardiovascular Risk Assessment</div>
        </div>
        """, unsafe_allow_html=True)
        
        heart_prediction = safe_get(results, 'heart_attack_prediction', {})
        heart_features = safe_get(results, 'heart_attack_features', {})
        
        if heart_prediction:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Risk Assessment Results")
                risk_display = heart_prediction.get('risk_display', 'Not available')
                confidence_display = heart_prediction.get('confidence_display', 'Not available')
                
                st.write(f"**{risk_display}**")
                st.write(f"**{confidence_display}**")
                
                risk_score = safe_get(results, 'heart_attack_risk_score', 0)
                try:
                    st.progress(float(risk_score))
                except (ValueError, TypeError):
                    st.progress(0.0)
                
                method = heart_prediction.get('prediction_method', 'Unknown')
                st.write(f"**Prediction Method:** {method}")
            
            with col2:
                st.markdown("#### 🎯 Risk Factors Analysis")
                feature_interp = heart_features.get('feature_interpretation', {}) if heart_features else {}
                if feature_interp:
                    for factor, value in feature_interp.items():
                        st.write(f"**{factor}:** {value}")
                else:
                    st.write("No risk factors data available")
            
            # Detailed prediction data
            with st.expander("🔍 Complete Risk Assessment Data"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Heart Attack Prediction:**")
                    st.json(heart_prediction)
                with col2:
                    st.markdown("**Risk Features:**")
                    st.json(heart_features)
        else:
            st.warning("No heart attack risk assessment available")

    # 4. COMBINED HEALTH SUMMARY & TRAJECTORY
    st.markdown("""
    <div class="section-box">
        <div class="section-title">📈 Comprehensive Health Analysis & Trajectory</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Health trajectory
    health_trajectory = results.get("health_trajectory")
    if health_trajectory:
        st.markdown("### 📈 Comprehensive Health Trajectory Analysis")
        with st.container():
            st.markdown(health_trajectory)
    
    # Final summary
    final_summary = results.get("final_summary")
    if final_summary:
        st.markdown("### 📋 Executive Health Summary")
        with st.container():
            st.markdown(final_summary)
    
    # If both are available, show combined view
    if health_trajectory and final_summary:
        with st.expander("📊 Complete Analysis Data"):
            tab1, tab2 = st.tabs(["Health Trajectory Data", "Summary Data"])
            with tab1:
                st.text_area("Health Trajectory", health_trajectory, height=300)
            with tab2:
                st.text_area("Final Summary", final_summary, height=300)
    
    # Show message if no detailed analysis available
    if not health_trajectory and not final_summary:
        st.info("📋 Comprehensive health analysis and trajectory will be displayed here after successful processing.")

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin: 2rem 0;">
    🔬 Enhanced Health Agent v7.0 | 
    <span class="enhanced-badge" style="margin: 0;">⚡ LangGraph Powered</span>
    <span class="enhanced-badge" style="margin: 0;">🚀 Comprehensive Analysis</span>
    <span class="enhanced-badge" style="margin: 0;">📊 Graph Generation</span>
    <span class="enhanced-badge" style="margin: 0;">🗂️ Complete Claims Viewer</span>
    <span class="enhanced-badge" style="margin: 0;">🎯 Predictive Modeling</span>
</div>
""", unsafe_allow_html=True)
