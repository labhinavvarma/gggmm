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
    # FIXED: Font configuration for unicode handling
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

# Enhanced CSS with advanced animations and modern styling + new sections
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

.batch-meanings-card {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    padding: 2.2rem;
    border-radius: 18px;
    border: 2px solid #2196f3;
    margin: 1.2rem 0;
    box-shadow: 0 10px 30px rgba(33, 150, 243, 0.2);
}

.medical-codes-section {
    background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c8 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border-left: 4px solid #4caf50;
}

.pharmacy-codes-section {
    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border-left: 4px solid #ff9800;
}

.code-table-container {
    background: white;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

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

.graph-container {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border: 2px solid #e3f2fd;
}

.graph-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #1976d2;
    margin-bottom: 1rem;
    text-align: center;
}

/* Enhanced Health Trajectory Section */
.health-trajectory-container {
    background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
    padding: 2.5rem;
    border-radius: 20px;
    border: 2px solid #28a745;
    margin: 1.5rem 0;
    box-shadow: 0 15px 40px rgba(40, 167, 69, 0.2);
    position: relative;
    overflow: hidden;
}

.health-trajectory-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(40, 167, 69, 0.1) 0%, transparent 70%);
    animation: pulse-health 4s ease-in-out infinite;
}

@keyframes pulse-health {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 0.6; }
}

.trajectory-content {
    position: relative;
    z-index: 2;
    background: rgba(255, 255, 255, 0.9);
    padding: 2rem;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}

/* Enhanced Heart Attack Prediction Section */
.heart-attack-container {
    background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
    padding: 2.5rem;
    border-radius: 20px;
    border: 2px solid #dc3545;
    margin: 1.5rem 0;
    box-shadow: 0 15px 40px rgba(220, 53, 69, 0.2);
    position: relative;
    overflow: hidden;
}

.heart-attack-container::before {
    content: '❤️';
    position: absolute;
    top: 20px;
    right: 20px;
    font-size: 3rem;
    opacity: 0.3;
    animation: heartbeat 2s ease-in-out infinite;
}

@keyframes heartbeat {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.1); }
}

/* Enhanced Entity Extraction with Graphs */
.entity-grid-enhanced {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
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

.status-error {
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    color: #d32f2f;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    border-left: 4px solid #f44336;
    font-weight: 600;
}

.chatbot-ready-button {
    background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
    padding: 1.2rem 3rem !important;
    border-radius: 25px !important;
    box-shadow: 0 10px 30px rgba(40, 167, 69, 0.4) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    animation: pulse-glow 2s infinite !important;
}

.chatbot-ready-button:hover {
    background: linear-gradient(135deg, #218838 0%, #1abc9c 100%) !important;
    transform: translateY(-5px) !important;
    box-shadow: 0 15px 40px rgba(40, 167, 69, 0.6) !important;
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 10px 30px rgba(40, 167, 69, 0.4); }
    50% { box-shadow: 0 15px 40px rgba(40, 167, 69, 0.8); }
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

def create_chatbot_loading_graphs():
    """Create interactive graphs to display while chatbot is loading"""
    
    # Create sample health data for visualization
    sample_data = {
        'dates': pd.date_range('2023-01-01', periods=12, freq='ME'),
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
    if 'chatbot_context' not in st.session_state:
        st.session_state.chatbot_context = None
    if 'calculated_age' not in st.session_state:
        st.session_state.calculated_age = None
    
    # Enhanced workflow steps - REMOVED FINAL SUMMARY
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
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'show_animation' not in st.session_state:
        st.session_state.show_animation = False

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
        <h2 style="color: #2c3e50; font-weight: 700;">Enhanced Healthcare Analysis Pipeline</h2>
        <p style="color: #34495e; font-size: 1.1rem;">Comprehensive health analysis workflow with graph generation</p>
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
        status_message = f"Currently executing: {current_step_name}"
    elif completed_steps == total_steps:
        status_message = "All healthcare workflow steps completed successfully!"
    elif error_steps > 0:
        status_message = f"{error_steps} step(s) encountered errors"
    else:
        status_message = "Healthcare analysis pipeline ready..."
    
    st.markdown(f"""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.8); border-radius: 10px;">
        <p style="margin: 0; font-weight: 600; color: #2c3e50;">{status_message}</p>
    </div>
    </div>
    """, unsafe_allow_html=True)

def display_enhanced_mcid_data(mcid_data):
    """Enhanced MCID data display with improved styling and functionality"""
    if not mcid_data:
        st.warning("No MCID data available")
        return
    # Raw MCID data in expandable section 
    with st.expander("View Raw MCID JSON Data"):
        st.json(mcid_data)

def display_batch_code_meanings_enhanced(results):
    """Enhanced batch processed code meanings in organized tabular format with proper subdivisions and FIXED METRICS"""
    st.markdown("""
    <div class="batch-meanings-card">
        <h3>Claims Data Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Get extraction results
    medical_extraction = safe_get(results, 'structured_extractions', {}).get('medical', {})
    pharmacy_extraction = safe_get(results, 'structured_extractions', {}).get('pharmacy', {})
    
    # Create main tabs for Medical and Pharmacy
    tab1, tab2 = st.tabs(["Medical Code Meanings", "Pharmacy Code Meanings"])
    
    with tab1:
        st.markdown('<div class="medical-codes-section">', unsafe_allow_html=True)
        st.markdown("### Medical Code Meanings Analysis")
        
        medical_meanings = medical_extraction.get("code_meanings", {})
        service_meanings = medical_meanings.get("service_code_meanings", {})
        diagnosis_meanings = medical_meanings.get("diagnosis_code_meanings", {})
        medical_records = medical_extraction.get("hlth_srvc_records", [])
        
        # FIXED METRICS CALCULATION - Count unique codes from actual data
        unique_service_codes = set()
        unique_diagnosis_codes = set()
        total_medical_records = len(medical_records)
        
        # Count unique codes from medical records
        for record in medical_records:
            # Count service codes
            service_code = record.get("hlth_srvc_cd", "")
            if service_code:
                unique_service_codes.add(service_code)
            
            # Count diagnosis codes
            for diag in record.get("diagnosis_codes", []):
                code = diag.get("code", "")
                if code:
                    unique_diagnosis_codes.add(code)
        
        # Medical summary metrics with CORRECTED VALUES and PROPER STYLING
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'''
            <div class="metric-summary-box">
                <h3 style="margin: 0; color: #007bff; font-size: 2rem; font-weight: bold;">{len(unique_service_codes)}</h3>
                <p style="margin: 0; color: #6c757d; font-weight: 600;">Service Codes</p>
            </div>
            ''', unsafe_allow_html=True)
        with col2:
            st.markdown(f'''
            <div class="metric-summary-box">
                <h3 style="margin: 0; color: #28a745; font-size: 2rem; font-weight: bold;">{len(unique_diagnosis_codes)}</h3>
                <p style="margin: 0; color: #6c757d; font-weight: 600;">ICD-10 Codes</p>
            </div>
            ''', unsafe_allow_html=True)
        with col3:
            st.markdown(f'''
            <div class="metric-summary-box">
                <h3 style="margin: 0; color: #dc3545; font-size: 2rem; font-weight: bold;">{total_medical_records}</h3>
                <p style="margin: 0; color: #6c757d; font-weight: 600;">Medical Records</p>
            </div>
            ''', unsafe_allow_html=True)
        with col4:
            batch_status = medical_extraction.get("llm_call_status", "unknown")
            status_color = "#28a745" if batch_status in ["success", "completed"] else "#ffc107" if batch_status == "pending" else "#dc3545"
            st.markdown(f'''
            <div class="metric-summary-box">
                <h3 style="margin: 0; color: {status_color}; font-size: 1.5rem; font-weight: bold;">{batch_status.upper()}</h3>
                <p style="margin: 0; color: #6c757d; font-weight: 600;">Batch Status</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Create sub-tabs for different medical code types
        med_tab1, med_tab2 = st.tabs(["ICD-10 Diagnosis Codes", "Medical Service Codes"])
        
        with med_tab1:
            st.markdown('<div class="code-table-container">', unsafe_allow_html=True)
            st.markdown("#### ICD-10 Diagnosis Codes with Dates and Meanings")
            
            if diagnosis_meanings and medical_records:
                # Prepare data for enhanced table display
                diagnosis_data = []
                for record in medical_records:
                    claim_date = record.get("clm_rcvd_dt", "Unknown")
                    record_path = record.get("data_path", "")
                    
                    for diag in record.get("diagnosis_codes", []):
                        code = diag.get("code", "")
                        if code in diagnosis_meanings:
                            diagnosis_data.append({
                                "ICD-10 Code": code,
                                "Code Meaning": diagnosis_meanings[code],
                                "Claim Date": claim_date,
                                "Position": diag.get("position", ""),
                                "Source Field": diag.get("source", ""),
                                "Record Path": record_path
                            })
                
                if diagnosis_data:
                    # Display unique code count
                    unique_codes = len(set(item["ICD-10 Code"] for item in diagnosis_data))
                    st.info(f"**Unique ICD-10 Codes Found:** {unique_codes}")
                    
                    # Create DataFrame and display as enhanced table
                    df_diagnosis = pd.DataFrame(diagnosis_data)
                    
                    # Sort by claim date (most recent first)
                    df_diagnosis_sorted = df_diagnosis.sort_values('Claim Date', ascending=False, na_position='last')
                    
                    st.dataframe(
                        df_diagnosis_sorted, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "ICD-10 Code": st.column_config.TextColumn("ICD-10 Code", width="small"),
                            "Code Meaning": st.column_config.TextColumn("Medical Meaning", width="large"),
                            "Claim Date": st.column_config.DateColumn("Claim Date", width="small"),
                            "Position": st.column_config.NumberColumn("Position", width="small"),
                            "Source Field": st.column_config.TextColumn("Source", width="small"),
                            "Record Path": st.column_config.TextColumn("Record Path", width="small")
                        }
                    )
                    
                    st.info("ICD-10 diagnosis data processed successfully")
                    
                    # Show code frequency analysis
                    with st.expander("ICD-10 Code Frequency Analysis"):
                        code_counts = df_diagnosis['ICD-10 Code'].value_counts()
                        st.bar_chart(code_counts)
                        st.write("**Most Frequent Diagnosis Codes:**")
                        for code, count in code_counts.head(5).items():
                            meaning = diagnosis_meanings.get(code, "Unknown")
                            st.write(f"• **{code}** ({count}x): {meaning}")
                else:
                    st.info("No ICD-10 diagnosis codes found in medical records")
            else:
                st.warning("No ICD-10 diagnosis code meanings available")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with med_tab2:
            st.markdown('<div class="code-table-container">', unsafe_allow_html=True)
            st.markdown("#### Medical Service Codes with Service End Dates and Meanings")
            
            if service_meanings and medical_records:
                # Prepare data for enhanced table display
                service_data = []
                for record in medical_records:
                    service_end_date = record.get("clm_line_srvc_end_dt", "Unknown")
                    service_code = record.get("hlth_srvc_cd", "")
                    record_path = record.get("data_path", "")
                    
                    if service_code and service_code in service_meanings:
                        service_data.append({
                            "Service Code": service_code,
                            "Service Meaning": service_meanings[service_code],
                            "Service End Date": service_end_date,
                            "Record Path": record_path
                        })
                
                if service_data:
                    # Display unique code count
                    unique_codes = len(set(item["Service Code"] for item in service_data))
                    st.info(f"**Unique Service Codes Found:** {unique_codes}")
                    
                    # Create DataFrame and display as enhanced table
                    df_service = pd.DataFrame(service_data)
                    
                    # Sort by service end date (most recent first)
                    df_service_sorted = df_service.sort_values('Service End Date', ascending=False, na_position='last')
                    
                    st.dataframe(
                        df_service_sorted, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "Service Code": st.column_config.TextColumn("Service Code", width="small"),
                            "Service Meaning": st.column_config.TextColumn("Service Description", width="large"),
                            "Service End Date": st.column_config.DateColumn("Service End Date", width="medium"),
                            "Record Path": st.column_config.TextColumn("Record Path", width="small")
                        }
                    )
                    
                    st.info("Medical service codes processed successfully")
                    
                    # Show code frequency analysis
                    with st.expander("Service Code Frequency Analysis"):
                        code_counts = df_service['Service Code'].value_counts()
                        st.bar_chart(code_counts)
                        st.write("**Most Frequent Service Codes:**")
                        for code, count in code_counts.head(5).items():
                            meaning = service_meanings.get(code, "Unknown")
                            st.write(f"• **{code}** ({count}x): {meaning}")
                else:
                    st.info("No medical service codes found in records")
            else:
                st.warning("No medical service code meanings available")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="pharmacy-codes-section">', unsafe_allow_html=True)
        st.markdown("### Pharmacy Code Meanings Analysis")
        
        pharmacy_meanings = pharmacy_extraction.get("code_meanings", {})
        ndc_meanings = pharmacy_meanings.get("ndc_code_meanings", {})
        med_meanings = pharmacy_meanings.get("medication_meanings", {})
        pharmacy_records = pharmacy_extraction.get("ndc_records", [])
        
        # FIXED METRICS CALCULATION - Count unique codes from actual pharmacy data
        unique_ndc_codes = set()
        unique_medications = set()
        total_pharmacy_records = len(pharmacy_records)
        
        # Count unique codes from pharmacy records
        for record in pharmacy_records:
            # Count NDC codes
            ndc_code = record.get("ndc", "")
            if ndc_code:
                unique_ndc_codes.add(ndc_code)
            
            # Count medications
            med_name = record.get("lbl_nm", "")
            if med_name:
                unique_medications.add(med_name)
        
        # Pharmacy summary metrics with CORRECTED VALUES and PROPER STYLING
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'''
            <div class="metric-summary-box">
                <h3 style="margin: 0; color: #007bff; font-size: 2rem; font-weight: bold;">{len(unique_ndc_codes)}</h3>
                <p style="margin: 0; color: #6c757d; font-weight: 600;">NDC Codes</p>
            </div>
            ''', unsafe_allow_html=True)
        with col2:
            st.markdown(f'''
            <div class="metric-summary-box">
                <h3 style="margin: 0; color: #28a745; font-size: 2rem; font-weight: bold;">{len(unique_medications)}</h3>
                <p style="margin: 0; color: #6c757d; font-weight: 600;">Medications</p>
            </div>
            ''', unsafe_allow_html=True)
        with col3:
            st.markdown(f'''
            <div class="metric-summary-box">
                <h3 style="margin: 0; color: #dc3545; font-size: 2rem; font-weight: bold;">{total_pharmacy_records}</h3>
                <p style="margin: 0; color: #6c757d; font-weight: 600;">Pharmacy Records</p>
            </div>
            ''', unsafe_allow_html=True)
        with col4:
            batch_status = pharmacy_extraction.get("llm_call_status", "unknown")
            status_color = "#28a745" if batch_status in ["success", "completed"] else "#ffc107" if batch_status == "pending" else "#dc3545"
            st.markdown(f'''
            <div class="metric-summary-box">
                <h3 style="margin: 0; color: {status_color}; font-size: 1.5rem; font-weight: bold;">{batch_status.upper()}</h3>
                <p style="margin: 0; color: #6c757d; font-weight: 600;">Batch Status</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Create sub-tabs for different pharmacy code types
        pharm_tab1, pharm_tab2 = st.tabs(["NDC Codes", "Medication Names"])
        
        with pharm_tab1:
            st.markdown('<div class="code-table-container">', unsafe_allow_html=True)
            st.markdown("#### NDC Codes with Fill Dates and Meanings")
            
            if pharmacy_records:
                # Prepare data for enhanced table display
                ndc_data = []
                for record in pharmacy_records:
                    fill_date = record.get("rx_filled_dt", "Unknown")
                    ndc_code = record.get("ndc", "")
                    label_name = record.get("lbl_nm", "")
                    record_path = record.get("data_path", "")
                    
                    if ndc_code:  # Just check if NDC code exists
                        ndc_meaning = ndc_meanings.get(ndc_code, f"NDC code {ndc_code}")  # Use fallback if no meaning
                        ndc_data.append({
                            "NDC Code": ndc_code,
                            "NDC Meaning": ndc_meaning,
                            "Medication Name": label_name,
                            "Fill Date": fill_date,
                            "Record Path": record_path
                        })
                
                if ndc_data:
                    # Display unique code count
                    unique_codes = len(set(item["NDC Code"] for item in ndc_data))
                    st.info(f"**Unique NDC Codes Found:** {unique_codes}")
                    
                    # Create DataFrame and display as enhanced table
                    df_ndc = pd.DataFrame(ndc_data)
                    
                    # Sort by fill date (most recent first)
                    df_ndc_sorted = df_ndc.sort_values('Fill Date', ascending=False, na_position='last')
                    
                    st.dataframe(
                        df_ndc_sorted, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "NDC Code": st.column_config.TextColumn("NDC Code", width="small"),
                            "NDC Meaning": st.column_config.TextColumn("NDC Description", width="large"),
                            "Medication Name": st.column_config.TextColumn("Medication", width="medium"),
                            "Fill Date": st.column_config.DateColumn("Fill Date", width="small"),
                            "Record Path": st.column_config.TextColumn("Record Path", width="small")
                        }
                    )
                    
                    st.info("NDC codes data processed successfully")
                    
                    # Show code frequency analysis
                    with st.expander("NDC Code Frequency Analysis"):
                        code_counts = df_ndc['NDC Code'].value_counts()
                        st.bar_chart(code_counts)
                        st.write("**Most Frequent NDC Codes:**")
                        for code, count in code_counts.head(5).items():
                            meaning = ndc_meanings.get(code, f"NDC code {code}")
                            st.write(f"• **{code}** ({count}x): {meaning}")
                else:
                    st.info("No NDC codes found in pharmacy records")
            else:
                st.warning("No pharmacy records available for NDC analysis")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with pharm_tab2:
            st.markdown('<div class="code-table-container">', unsafe_allow_html=True)
            st.markdown("#### Medication Names with Fill Dates and Meanings")
            
            if pharmacy_records:
                # Prepare data for enhanced table display
                medication_data = []
                for record in pharmacy_records:
                    fill_date = record.get("rx_filled_dt", "Unknown")
                    med_name = record.get("lbl_nm", "")
                    ndc_code = record.get("ndc", "")
                    record_path = record.get("data_path", "")
                    
                    if med_name:  # Just check if medication name exists
                        med_meaning = med_meanings.get(med_name, f"Medication: {med_name}")  # Use fallback if no meaning
                        medication_data.append({
                            "Medication Name": med_name,
                            "Medication Meaning": med_meaning,
                            "NDC Code": ndc_code,
                            "Fill Date": fill_date,
                            "Record Path": record_path
                        })
                
                if medication_data:
                    # Display unique medication count
                    unique_meds = len(set(item["Medication Name"] for item in medication_data))
                    st.info(f"**Unique Medications Found:** {unique_meds}")
                    
                    # Create DataFrame and display as enhanced table
                    df_medication = pd.DataFrame(medication_data)
                    
                    # Sort by fill date (most recent first)
                    df_medication_sorted = df_medication.sort_values('Fill Date', ascending=False, na_position='last')
                    
                    st.dataframe(
                        df_medication_sorted, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "Medication Name": st.column_config.TextColumn("Medication", width="medium"),
                            "Medication Meaning": st.column_config.TextColumn("Therapeutic Description", width="large"),
                            "NDC Code": st.column_config.TextColumn("NDC Code", width="small"),
                            "Fill Date": st.column_config.DateColumn("Fill Date", width="small"),
                            "Record Path": st.column_config.TextColumn("Record Path", width="small")
                        }
                    )
                    
                    st.info("Medication data processed successfully")
                    
                    # Show medication frequency analysis
                    with st.expander("Medication Frequency Analysis"):
                        med_counts = df_medication['Medication Name'].value_counts()
                        st.bar_chart(med_counts)
                        st.write("**Most Frequent Medications:**")
                        for med, count in med_counts.head(5).items():
                            meaning = med_meanings.get(med, f"Medication: {med}")
                            st.write(f"• **{med}** ({count}x): {meaning}")
                else:
                    st.info("No medication names found in pharmacy records")
            else:
                st.warning("No pharmacy records available for medication analysis")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
initialize_session_state()

# Enhanced Main Title
st.markdown('<h1 class="main-header">Deep Research Health Agent 2.0</h1>', unsafe_allow_html=True)

# Display import status
if not AGENT_AVAILABLE:
    st.markdown(f'<div class="status-error">Failed to import Health Agent: {import_error}</div>', unsafe_allow_html=True)
    st.stop()

# REMOVED SIDEBAR CHATBOT - Only show placeholder when chatbot is NOT ready
with st.sidebar:
    st.title("Medical Assistant")
    st.info("Medical Assistant will be available after running health analysis")
    st.markdown("---")
    st.markdown("**What you can ask:**")
    st.markdown("• **Medical Records:** Diagnoses, procedures, ICD codes, service codes")
    st.markdown("• **Medications:** Prescriptions, NDC codes, drug interactions, therapeutic analysis") 
    st.markdown("• **Risk Assessment:** Heart attack risk, chronic conditions, clinical predictions")
    st.markdown("• **Health Summary:** Combined trajectory analysis, comprehensive health insights")
    st.markdown("• **Visualizations:** Charts, graphs, dashboards, timelines with matplotlib")
    st.markdown("---")
    st.markdown("**Enhanced Features:**")
    st.markdown("• Categorized prompt system for easy navigation")
    st.markdown("• Quick access buttons for common analyses")
    st.markdown("• **Advanced graph generation with matplotlib**")
    st.markdown("• **Real-time chart display in chat**")
    st.markdown("• Comprehensive health summary with trajectory analysis")
    st.markdown("• Professional clinical decision support")
    st.markdown("• **Batch code meanings with LLM explanations**")
    
    # Show loading graphs while chatbot is being prepared
    if st.session_state.analysis_running or (st.session_state.analysis_results and not st.session_state.analysis_results.get("chatbot_ready", False)):
        st.markdown("**Preparing AI Assistant...**")
        st.info("Loading healthcare analysis capabilities with graph generation")
        
        # Display interactive loading graphs
        try:
            loading_fig = create_chatbot_loading_graphs()
            st.plotly_chart(loading_fig, use_container_width=True, key="chatbot_loading_graphs")
        except Exception as e:
            st.info("Health analytics dashboard loading...")

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
            first_name = st.text_input("First Name *", value="",type="password")
            last_name = st.text_input("Last Name *", value="",type="password")
        
        with col2:
            ssn = st.text_input("SSN *", value="",type="password")
            date_of_birth = st.date_input(
                "Date of Birth *", 
                value=datetime.now().date(),
                min_value=datetime(1900, 1, 1).date(),
                max_value=datetime.now().date()
            )
        
        with col3:
            gender = st.selectbox("Gender *", ["F", "M"])
            zip_code = st.text_input("Zip Code *", value="",type="password")
        
        # Show calculated age - persists until new analysis
        if date_of_birth:
            calculated_age = calculate_age(date_of_birth)
            if calculated_age is not None:
                st.session_state.calculated_age = calculated_age
                st.info(f"**Calculated Age:** {calculated_age} years old")
        elif st.session_state.calculated_age is not None:
            st.info(f"**Calculated Age:** {st.session_state.calculated_age} years old")
        
        # ENHANCED RUN ANALYSIS BUTTON
        submitted = st.form_submit_button(
            "Run Enhanced Healthcare Analysis", 
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
        st.session_state.chatbot_context = None
        st.session_state.calculated_age = None  # Reset age for new patient
        
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
        
        # ENHANCED WORKFLOW: Run graphics animation FIRST, then actual processing
        with st.spinner("Running Enhanced Healthcare Analysis..."):
            try:
                # Display initial workflow state
                with workflow_placeholder.container():
                    display_advanced_professional_workflow()
                
                # PHASE 1: Run the visual workflow animation first (shortened to 15 seconds)
                st.info("Initializing workflow visualization...")
                
                total_steps = len(st.session_state.workflow_steps)
                # OPTIMIZED TIMING: 15 seconds total instead of 67 seconds
                step_running_time = 1.0   # 1 second per step running
                step_pause_time = 0.8     # 0.8 seconds pause between steps
                final_pause_time = 2.0    # 2 seconds final pause
                # Total: 7 steps * (1 + 0.8) = 12.6 + 2 = ~15 seconds
                
                for i, step in enumerate(st.session_state.workflow_steps):
                    # Set step to running
                    st.session_state.workflow_steps[i]['status'] = 'running'
                    with workflow_placeholder.container():
                        display_advanced_professional_workflow()
                    time.sleep(step_running_time)
                    
                    # Set step to completed
                    st.session_state.workflow_steps[i]['status'] = 'completed'
                    with workflow_placeholder.container():
                        display_advanced_professional_workflow()
                    time.sleep(step_pause_time)
                
                # All steps are now green - shorter pause to show complete workflow
                st.success("Workflow visualization complete! Starting actual processing...")
                time.sleep(final_pause_time)
                
                # PHASE 2: Now run the actual analysis
                st.info("Running actual healthcare analysis...")
                
                # Reset workflow for actual processing
                for step in st.session_state.workflow_steps:
                    step['status'] = 'pending'
                
                # Actually run the analysis
                try:
                    results = st.session_state.agent.run_analysis(patient_data)
                    analysis_success = results.get("success", False)
                    
                    if analysis_success:
                        # Mark all steps as completed
                        for step in st.session_state.workflow_steps:
                            step['status'] = 'completed'
                    else:
                        # Mark all steps as error
                        for step in st.session_state.workflow_steps:
                            step['status'] = 'error'
                        raise Exception("Analysis failed")
                    
                except Exception as analysis_error:
                    st.error(f"Analysis failed: {str(analysis_error)}")
                    analysis_success = False
                    # Mark all steps as error
                    for step in st.session_state.workflow_steps:
                        step['status'] = 'error'
                
                # Final workflow display
                with workflow_placeholder.container():
                    display_advanced_professional_workflow()
                
                st.session_state.analysis_results = results
                st.session_state.analysis_running = False
                
                # Set chatbot context if analysis successful
                if results and results.get("success") and results.get("chatbot_ready"):
                    st.session_state.chatbot_context = results.get("chatbot_context")
                
                if analysis_success:
                    st.success("Enhanced Healthcare Analysis completed successfully!")
                    st.balloons()  # Add celebration effect
                else:
                    st.error("Healthcare Analysis encountered errors!")
                
                st.rerun()
                
            except Exception as e:
                st.session_state.analysis_running = False
                st.error(f"Analysis failed: {str(e)}")
                
                # Mark all steps as error
                for step in st.session_state.workflow_steps:
                    step['status'] = 'error'
                
                with workflow_placeholder.container():
                    display_advanced_professional_workflow()

# ENHANCED TIMING: Display workflow animation while analysis is running
if st.session_state.analysis_running:
    display_advanced_professional_workflow()
    
    # ENHANCED: Show loading graphs during processing
    st.markdown("**Enhanced Healthcare Analysis in Progress...**")
    st.info("Processing comprehensive health data with AI-powered analysis")
    
    # Display interactive loading graphs during processing
    try:
        loading_fig = create_chatbot_loading_graphs()
        st.plotly_chart(loading_fig, use_container_width=True, key="processing_graphs")
    except Exception as e:
        st.info("Health analytics processing...")

# NEW: CHATBOT READY BANNER AND BUTTON
if st.session_state.analysis_results and st.session_state.analysis_results.get("chatbot_ready", False) and st.session_state.chatbot_context:
    # Analysis Complete Banner
    st.markdown("""
    <div class="analysis-complete-banner">
        <h2 style="margin: 0; color: #28a745; font-weight: 700;">🎉 Healthcare Analysis Complete!</h2>
        <p style="margin: 0.5rem 0; color: #155724; font-size: 1.1rem;">Your comprehensive health analysis is ready. Launch the Medical Assistant to explore insights and generate visualizations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "🚀 Launch Medical Assistant", 
            key="launch_chatbot",
            use_container_width=True,
            help="Open the dedicated Medical Assistant window with full analysis capabilities"
        ):
            # Switch to chatbot page
            st.switch_page("pages/chatbot.py")

# ENHANCED RESULTS SECTION - CHANGED TO EXPANDABLE SECTIONS
if st.session_state.analysis_results and not st.session_state.analysis_running:
    results = st.session_state.analysis_results

    # 1. CLAIMS DATA - CHANGED TO EXPANDABLE
    with st.expander("Claims Data", expanded=False):
        st.markdown("""
        <div class="section-box">
            <div class="section-title">Claims Data</div>
        </div>
        """, unsafe_allow_html=True)
        
        deidentified_data = safe_get(results, 'deidentified_data', {})
        api_outputs = safe_get(results, 'api_outputs', {})
        
        if deidentified_data or api_outputs:
            tab1, tab2, tab3 = st.tabs([
                "Medical Claims", 
                "Pharmacy Claims", 
                "MCID Data"
            ])
            
            with tab1:
                medical_data = safe_get(deidentified_data, 'medical', {})
                if medical_data and not medical_data.get('error'):
                    st.markdown("### Medical Claims Analysis")
                    
                    medical_claims_data = medical_data.get('medical_claims_data', {})
                    if medical_claims_data:
                        with st.expander("Medical Claims JSON Data", expanded=False):
                            st.json(medical_claims_data)
                else:
                    st.error("No medical claims data available")
            
            with tab2:
                pharmacy_data = safe_get(deidentified_data, 'pharmacy', {})
                if pharmacy_data and not pharmacy_data.get('error'):
                    st.markdown("### Pharmacy Claims Analysis")
                    
                    pharmacy_claims_data = pharmacy_data.get('pharmacy_claims_data', {})
                    if pharmacy_claims_data:
                        with st.expander("Pharmacy Claims JSON Data", expanded=False):
                            st.json(pharmacy_claims_data)
                else:
                    st.error("No pharmacy claims data available")
            
            with tab3:
                mcid_data = safe_get(api_outputs, 'mcid', {})
                display_enhanced_mcid_data(mcid_data)
        else:
            st.error("No claims data available")

    # 2. CLAIMS DATA ANALYSIS - CHANGED TO EXPANDABLE
    with st.expander("Claims Data Analysis", expanded=False):
        display_batch_code_meanings_enhanced(results)

    # 3. ENTITY EXTRACTION - CHANGED TO EXPANDABLE
    with st.expander("Entity Extraction", expanded=False):
        st.markdown("""
        <div class="section-box">
            <div class="section-title">Enhanced Entity Extraction & Health Analytics</div>
        </div>
        """, unsafe_allow_html=True)
        
        entity_extraction = safe_get(results, 'entity_extraction', {})
        if entity_extraction:
            # Enhanced Entity Cards with Status Colors
            st.markdown("""
            <div class="entity-grid-enhanced">
            """, unsafe_allow_html=True)
            
            # Define entity data with enhanced styling
            entities_data = [
                {
                    'icon': '🩺',
                    'label': 'Diabetes Status',
                    'value': entity_extraction.get('diabetics', 'unknown'),
                    'key': 'diabetics'
                },
                {
                    'icon': '👥',
                    'label': 'Age Group',
                    'value': entity_extraction.get('age_group', 'unknown'),
                    'key': 'age_group'
                },
                {
                    'icon': '🚬',
                    'label': 'Smoking Status',
                    'value': entity_extraction.get('smoking', 'unknown'),
                    'key': 'smoking'
                },
                {
                    'icon': '🍷',
                    'label': 'Alcohol Use',
                    'value': entity_extraction.get('alcohol', 'unknown'),
                    'key': 'alcohol'
                },
                {
                    'icon': '💓',
                    'label': 'Blood Pressure',
                    'value': entity_extraction.get('blood_pressure', 'unknown'),
                    'key': 'blood_pressure'
                }
            ]
            
            # Create columns for entity cards
            cols = st.columns(len(entities_data))
            
            for i, (col, entity) in enumerate(zip(cols, entities_data)):
                with col:
                    value = entity['value']
                    # Determine status class
                    if entity['key'] in ['diabetics', 'smoking'] and value == 'yes':
                        status_class = 'positive'
                    elif entity['key'] in ['diabetics', 'smoking'] and value == 'no':
                        status_class = 'negative'
                    elif value == 'unknown':
                        status_class = 'unknown'
                    else:
                        status_class = 'negative'
                    
                    st.markdown(f"""
                    <div class="entity-card-enhanced">
                        <span class="entity-icon">{entity['icon']}</span>
                        <div class="entity-label">{entity['label']}</div>
                        <div class="entity-value {status_class}">{value.upper()}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

    # 4. HEALTH TRAJECTORY - CHANGED TO EXPANDABLE  
    with st.expander("Health Trajectory", expanded=False):
        st.markdown("""
        <div class="health-trajectory-container">
            <div class="section-title">Health Trajectory</div>
        </div>
        """, unsafe_allow_html=True)
        
        health_trajectory = safe_get(results, 'health_trajectory', '')
        if health_trajectory:
            st.markdown("""
            <div class="trajectory-content">
            """, unsafe_allow_html=True)
            
            # Add enhanced formatting
            st.markdown("### Predictive Health Analysis")
            st.markdown(health_trajectory)
            
            # Add trajectory summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Analysis Type", "Comprehensive")
            with col2:
                medical_records = len(safe_get(results, 'structured_extractions', {}).get('medical', {}).get('hlth_srvc_records', []))
                st.metric("Medical Records", medical_records)
            with col3:
                pharmacy_records = len(safe_get(results, 'structured_extractions', {}).get('pharmacy', {}).get('ndc_records', []))
                st.metric("Pharmacy Records", pharmacy_records)
            with col4:
                entity_count = len(safe_get(results, 'entity_extraction', {}).get('medical_conditions', []))
                st.metric("Conditions", entity_count)
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Health trajectory analysis not available. Please run the analysis first.")

    # 5. HEART ATTACK RISK PREDICTION - CHANGED TO EXPANDABLE AND ENHANCED BOX
    with st.expander("Heart Attack Risk Prediction", expanded=False):
        st.markdown("""
        <div class="heart-attack-container">
            <div class="section-title">Cardiovascular Risk Assessment</div>
        </div>
        """, unsafe_allow_html=True)
        
        heart_attack_prediction = safe_get(results, 'heart_attack_prediction', {})
        heart_attack_features = safe_get(results, 'heart_attack_features', {})
        heart_attack_risk_score = safe_get(results, 'heart_attack_risk_score', 0.0)
        
        if heart_attack_prediction and not heart_attack_prediction.get('error'):
            # Main risk display - USING STREAMLIT COMPONENTS INSTEAD OF HTML
            combined_display = heart_attack_prediction.get("combined_display", "Heart Disease Risk: Not available")
            risk_category = heart_attack_prediction.get("risk_category", "Unknown")
            
            # Remove "Confidence:" and "|" from the display and clean up text
            if "Confidence:" in combined_display:
                combined_display = combined_display.split("Confidence:")[0].strip()
            
            # Remove any "|" characters
            combined_display = combined_display.replace("|", "").strip()
            
            # Extract just the risk percentage if available
            risk_percentage = ""
            if "%" in combined_display:
                import re
                percentage_match = re.search(r'(\d+\.?\d*%)', combined_display)
                if percentage_match:
                    risk_percentage = percentage_match.group(1)
            
            # Enhanced risk display using Streamlit components
            st.markdown("### ML Model Prediction Results")
            
            # Create a clean display box
            with st.container():
                # Risk title
                st.markdown("""
                <div style="text-align: center; margin: 1rem 0;">
                    <h3 style="color: #2c3e50; font-size: 1.8rem; margin-bottom: 1rem;">Heart Disease Risk</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk percentage in large text
                if risk_percentage:
                    st.markdown(f"""
                    <div style="text-align: center; margin: 2rem 0;">
                        <div style="font-size: 4rem; font-weight: 800; color: #dc3545; text-shadow: 0 2px 4px rgba(220, 53, 69, 0.3);">
                            {risk_percentage}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="text-align: center; margin: 2rem 0;">
                        <div style="font-size: 3rem; font-weight: 800; color: #dc3545;">
                            Assessment Complete
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Description text
                st.info("Advanced machine learning analysis provides cardiovascular risk assessment based on comprehensive health data evaluation and clinical indicators.")
                
                # Risk category
                if risk_category == 'High Risk':
                    st.error(f"**{risk_category}**")
                elif risk_category == 'Medium Risk':
                    st.warning(f"**{risk_category}**")
                else:
                    st.success(f"**{risk_category}**")
                    
        else:
            st.warning("Heart attack risk prediction not available.")

if __name__ == "__main__":
    pass
