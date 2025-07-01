# Configure Streamlit page FIRST - before any other Streamlit commands
import streamlit as st

# Determine sidebar state based on chatbot readiness
if 'analysis_results' in st.session_state and st.session_state.get('analysis_results') and st.session_state.analysis_results.get("chatbot_ready", False):
    sidebar_state = "expanded"
else:
    sidebar_state = "collapsed"

st.set_page_config(
    page_title="Deep Research Health Agent",
    page_icon="🔬",
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
import asyncio

# Add current directory to path for importing the agent
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Set up logging
logger = logging.getLogger(__name__)

# Import the Enhanced Modular LangGraph health analysis agent
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

# Advanced CSS for sophisticated professional animation with DARK TEXT + CODE EXPLANATIONS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.main-header {
    font-size: 2.8rem;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
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
    max-height: 400px;
    overflow-y: auto;
    font-family: monospace;
    font-size: 0.85rem;
}

.code-explanation {
    background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
    padding: 0.8rem;
    border-radius: 6px;
    border-left: 3px solid #28a745;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    font-style: italic;
    color: #155724;
}

.code-container {
    background: #f1f3f4;
    padding: 0.5rem;
    border-radius: 4px;
    font-family: monospace;
    font-weight: bold;
    display: inline-block;
    margin: 0.2rem;
}

/* Green Run Analysis Button */
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3) !important;
    transition: all 0.3s ease !important;
}

.stButton button[kind="primary"]:hover {
    background: linear-gradient(135deg, #218838 0%, #1abc9c 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4) !important;
}

/* Advanced Professional Workflow Container with Lighter Background */
.advanced-workflow-container {
    background: linear-gradient(135deg, #e8f0fe 0%, #f3e5f5 25%, #e1f5fe 50%, #f1f8e9 75%, #fff8e1 100%);
    padding: 3rem;
    border-radius: 25px;
    margin: 2rem 0;
    color: #2c3e50;
    box-shadow: 
        0 25px 50px rgba(52, 152, 219, 0.2),
        0 0 0 1px rgba(0, 0, 0, 0.1),
        inset 0 1px 0 rgba(255, 255, 255, 0.8);
    position: relative;
    overflow: hidden;
    border: 2px solid rgba(0, 0, 0, 0.1);
}

.advanced-workflow-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
    animation: rotate 20s linear infinite;
    pointer-events: none;
}

.advanced-workflow-container::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, 
        transparent 30%, 
        rgba(255,255,255,0.4) 50%, 
        transparent 70%);
    animation: shimmer 3s ease-in-out infinite;
    pointer-events: none;
}

@keyframes rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

@keyframes shimmer {
    0%, 100% { transform: translateX(-100%) translateY(-100%); }
    50% { transform: translateX(100%) translateY(100%); }
}

.workflow-header {
    text-align: center;
    margin-bottom: 3rem;
    position: relative;
    z-index: 10;
}

.workflow-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(255,255,255,0.8);
    color: #2c3e50;
}

.workflow-subtitle {
    font-size: 1.2rem;
    margin-bottom: 2rem;
    font-weight: 500;
    text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
    color: #34495e;
}

.progress-dashboard {
    background: rgba(255, 255, 255, 0.7);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 0, 0, 0.1);
    position: relative;
    z-index: 10;
    box-shadow: 
        0 15px 35px rgba(0, 0, 0, 0.1),
        0 5px 15px rgba(0, 0, 0, 0.05);
}

.progress-header {
    text-align: center;
    margin-bottom: 2rem;
}

.progress-title {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: #2c3e50;
    text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
}

.stat-card {
    background: rgba(255, 255, 255, 0.8);
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    border: 1px solid rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.stat-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
    transition: left 0.5s ease;
}

.stat-card:hover::before {
    left: 100%;
}

.stat-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
}

.stat-number {
    font-size: 2.5rem;
    font-weight: 700;
    color: #2c3e50;
    text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
    margin-bottom: 0.5rem;
}

.stat-label {
    font-size: 1rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #34495e;
}

.advanced-progress-container {
    margin: 2rem 0;
}

.progress-bar-wrapper {
    background: rgba(255, 255, 255, 0.6);
    border-radius: 25px;
    padding: 8px;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(0, 0, 0, 0.1);
}

.progress-bar-fill {
    height: 20px;
    background: linear-gradient(90deg, #00ff87, #60efff, #ff6b9d, #ffd93d);
    border-radius: 20px;
    position: relative;
    overflow: hidden;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 5px 15px rgba(0, 255, 135, 0.4);
}

.progress-bar-fill::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(90deg, 
        transparent, 
        rgba(255,255,255,0.6), 
        transparent);
    animation: progress-shine 2s infinite;
}

@keyframes progress-shine {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.steps-section {
    background: rgba(255, 255, 255, 0.5);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(0, 0, 0, 0.1);
    position: relative;
    z-index: 10;
}

.steps-title {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 2rem;
    text-align: center;
    color: #2c3e50;
    text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
}

.step-item {
    background: rgba(255, 255, 255, 0.6);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(0, 0, 0, 0.1);
}

.step-item::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.3);
    border-radius: 15px;
    transition: all 0.3s ease;
    z-index: -1;
}

.step-pending {
    border-left: 4px solid #6c757d;
}

.step-running {
    border-left: 4px solid #ffc107;
    background: rgba(255, 193, 7, 0.15);
    animation: pulse-step 2s infinite;
    box-shadow: 0 10px 30px rgba(255, 193, 7, 0.3);
}

.step-completed {
    border-left: 4px solid #28a745;
    background: rgba(40, 167, 69, 0.15);
    box-shadow: 0 10px 30px rgba(40, 167, 69, 0.2);
}

.step-error {
    border-left: 4px solid #dc3545;
    background: rgba(220, 53, 69, 0.15);
    animation: shake-step 0.5s ease-in-out;
}

@keyframes pulse-step {
    0%, 100% { 
        transform: scale(1);
        box-shadow: 0 10px 30px rgba(255, 193, 7, 0.3);
    }
    50% { 
        transform: scale(1.02);
        box-shadow: 0 15px 40px rgba(255, 193, 7, 0.5);
    }
}

@keyframes shake-step {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-10px); }
    75% { transform: translateX(10px); }
}

.step-content {
    display: flex;
    align-items: center;
    gap: 1.5rem;
}

.step-icon {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    font-weight: 700;
    position: relative;
    transition: all 0.4s ease;
    flex-shrink: 0;
}

.step-icon-pending {
    background: rgba(108, 117, 125, 0.2);
    color: #6c757d;
    border: 2px solid #6c757d;
}

.step-icon-running {
    background: linear-gradient(135deg, #ffc107, #ff8f00);
    color: #000;
    border: 2px solid #ffca28;
    animation: spin-icon 2s linear infinite;
    box-shadow: 0 0 25px rgba(255, 193, 7, 0.6);
}

.step-icon-completed {
    background: linear-gradient(135deg, #28a745, #20c997);
    color: white;
    border: 2px solid #34ce57;
    box-shadow: 0 0 20px rgba(40, 167, 69, 0.5);
}

.step-icon-error {
    background: linear-gradient(135deg, #dc3545, #ff6b6b);
    color: white;
    border: 2px solid #e74c3c;
    box-shadow: 0 0 20px rgba(220, 53, 69, 0.5);
}

@keyframes spin-icon {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.step-details {
    flex: 1;
}

.step-name {
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: #2c3e50;
    text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
}

.step-description {
    font-size: 1rem;
    line-height: 1.5;
    color: #34495e;
}

.step-status {
    padding: 0.5rem 1.5rem;
    border-radius: 25px;
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    flex-shrink: 0;
}

.status-pending {
    background: rgba(108, 117, 125, 0.2);
    color: #6c757d;
}

.status-running {
    background: linear-gradient(135deg, #ffc107, #ff8f00);
    color: #000;
    animation: pulse-status 1.5s infinite;
}

.status-completed {
    background: linear-gradient(135deg, #28a745, #20c997);
    color: white;
}

.status-error {
    background: linear-gradient(135deg, #dc3545, #ff6b6b);
    color: white;
}

@keyframes pulse-status {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.8; transform: scale(1.05); }
}

.workflow-footer {
    text-align: center;
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid rgba(0, 0, 0, 0.1);
    position: relative;
    z-index: 10;
}

.footer-status {
    font-size: 1.2rem;
    font-weight: 500;
    color: #2c3e50;
    text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
    animation: breathe 3s infinite;
}

@keyframes breathe {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; }
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
    
    # Section toggle states
    if 'show_claims_data' not in st.session_state:
        st.session_state.show_claims_data = False
    if 'show_claims_extraction' not in st.session_state:
        st.session_state.show_claims_extraction = False
    if 'show_entity_extraction' not in st.session_state:
        st.session_state.show_entity_extraction = False
    if 'show_health_trajectory' not in st.session_state:
        st.session_state.show_health_trajectory = False
    if 'show_final_summary' not in st.session_state:
        st.session_state.show_final_summary = False
    if 'show_heart_attack' not in st.session_state:
        st.session_state.show_heart_attack = False
    
    # Advanced workflow steps with enhanced details
    if 'workflow_steps' not in st.session_state:
        st.session_state.workflow_steps = [
            {'name': 'Fetching Claims Data', 'status': 'pending', 'description': 'Retrieving medical and pharmacy claims from secure APIs', 'icon': '📡', 'emoji': '🌐'},
            {'name': 'Deidentifying Claims Data', 'status': 'pending', 'description': 'Removing personal identifiers while preserving clinical value', 'icon': '🔒', 'emoji': '🛡️'},
            {'name': 'Extracting Claims Fields', 'status': 'pending', 'description': 'Parsing medical codes, NDC numbers, and structured data', 'icon': '🔍', 'emoji': '⚙️'},
            {'name': 'Extracting Health Entities', 'status': 'pending', 'description': 'Identifying conditions, medications, and risk factors', 'icon': '🎯', 'emoji': '🧬'},
            {'name': 'Analyzing Health Trajectory', 'status': 'pending', 'description': 'Computing longitudinal health patterns and trends', 'icon': '📈', 'emoji': '📊'},
            {'name': 'Generating Summary', 'status': 'pending', 'description': 'Creating comprehensive clinical assessment report', 'icon': '📋', 'emoji': '📄'},
            {'name': 'Predicting Heart Attack Risk', 'status': 'pending', 'description': 'Running advanced ML risk assessment algorithms', 'icon': '❤️', 'emoji': '🤖'},
            {'name': 'Initializing Assistant', 'status': 'pending', 'description': 'Setting up AI medical assistant with full context', 'icon': '🤖', 'emoji': '💬'}
        ]
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'show_animation' not in st.session_state:
        st.session_state.show_animation = False

def reset_workflow():
    """Reset workflow to initial state"""
    st.session_state.workflow_steps = [
        {'name': 'Fetching Claims Data', 'status': 'pending', 'description': 'Retrieving medical and pharmacy claims from secure APIs', 'icon': '📡', 'emoji': '🌐'},
        {'name': 'Deidentifying Claims Data', 'status': 'pending', 'description': 'Removing personal identifiers while preserving clinical value', 'icon': '🔒', 'emoji': '🛡️'},
        {'name': 'Extracting Claims Fields', 'status': 'pending', 'description': 'Parsing medical codes, NDC numbers, and structured data', 'icon': '🔍', 'emoji': '⚙️'},
        {'name': 'Extracting Health Entities', 'status': 'pending', 'description': 'Identifying conditions, medications, and risk factors', 'icon': '🎯', 'emoji': '🧬'},
        {'name': 'Analyzing Health Trajectory', 'status': 'pending', 'description': 'Computing longitudinal health patterns and trends', 'icon': '📈', 'emoji': '📊'},
        {'name': 'Generating Summary', 'status': 'pending', 'description': 'Creating comprehensive clinical assessment report', 'icon': '📋', 'emoji': '📄'},
        {'name': 'Predicting Heart Attack Risk', 'status': 'pending', 'description': 'Running advanced ML risk assessment algorithms', 'icon': '❤️', 'emoji': '🤖'},
        {'name': 'Initializing Assistant', 'status': 'pending', 'description': 'Setting up AI medical assistant with full context', 'icon': '🤖', 'emoji': '💬'}
    ]
    st.session_state.current_step = 0

def display_advanced_professional_workflow():
    """Display the most advanced and professional workflow animation"""
    
    # Calculate comprehensive statistics
    total_steps = len(st.session_state.workflow_steps)
    completed_steps = sum(1 for step in st.session_state.workflow_steps if step['status'] == 'completed')
    running_steps = sum(1 for step in st.session_state.workflow_steps if step['status'] == 'running')
    error_steps = sum(1 for step in st.session_state.workflow_steps if step['status'] == 'error')
    progress_percentage = (completed_steps / total_steps) * 100
    
    # Main advanced container
    st.markdown('<div class="advanced-workflow-container">', unsafe_allow_html=True)
    
    # Header Section
    st.markdown("""
    <div class="workflow-header">
        <div class="workflow-title">🔬 Deep Research Analysis</div>
        <div class="workflow-subtitle">Advanced Healthcare Data Processing Pipeline</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress Dashboard
    st.markdown('<div class="progress-dashboard">', unsafe_allow_html=True)
    st.markdown('<div class="progress-title">Real-Time Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Advanced stats grid using Streamlit columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{total_steps}</div>
            <div class="stat-label">Total Steps</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{completed_steps}</div>
            <div class="stat-label">Completed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{running_steps}</div>
            <div class="stat-label">Processing</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{progress_percentage:.0f}%</div>
            <div class="stat-label">Progress</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced Progress Bar
    st.markdown(f"""
    <div class="advanced-progress-container">
        <div class="progress-bar-wrapper">
            <div class="progress-bar-fill" style="width: {progress_percentage}%;"></div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Steps Section
    st.markdown('<div class="steps-section">', unsafe_allow_html=True)
    st.markdown('<div class="steps-title">Workflow Execution Pipeline</div>', unsafe_allow_html=True)
    
    # Display each step with advanced styling
    for i, step in enumerate(st.session_state.workflow_steps):
        step_number = i + 1
        name = step['name']
        status = step['status']
        description = step['description']
        icon = step['icon']
        emoji = step['emoji']
        
        # Determine CSS classes and content based on status
        if status == 'pending':
            step_class = "step-pending"
            icon_class = "step-icon-pending"
            status_class = "status-pending"
            icon_content = str(step_number)
            status_text = "Waiting"
        elif status == 'running':
            step_class = "step-running"
            icon_class = "step-icon-running"
            status_class = "status-running"
            icon_content = "●"
            status_text = "Processing"
        elif status == 'completed':
            step_class = "step-completed"
            icon_class = "step-icon-completed"
            status_class = "status-completed"
            icon_content = "✓"
            status_text = "Complete"
        elif status == 'error':
            step_class = "step-error"
            icon_class = "step-icon-error"
            status_class = "status-error"
            icon_content = "✗"
            status_text = "Failed"
        else:
            step_class = "step-pending"
            icon_class = "step-icon-pending"
            status_class = "status-pending"
            icon_content = str(step_number)
            status_text = "Waiting"
        
        # Create the step item
        st.markdown(f"""
        <div class="step-item {step_class}">
            <div class="step-content">
                <div class="step-icon {icon_class}">
                    {icon_content}
                </div>
                <div class="step-details">
                    <div class="step-name">{icon} {name} {emoji}</div>
                    <div class="step-description">{description}</div>
                </div>
                <div class="step-status {status_class}">
                    {status_text}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close steps-section
    
    # Advanced Footer with dynamic status
    if running_steps > 0:
        current_step_name = next((step['name'] for step in st.session_state.workflow_steps if step['status'] == 'running'), 'Processing')
        status_message = f"🔄 Currently executing: {current_step_name}"
    elif completed_steps == total_steps:
        status_message = "🎉 All workflow steps completed successfully!"
    elif error_steps > 0:
        status_message = f"⚠️ {error_steps} step(s) encountered errors"
    else:
        status_message = "⏳ Comprehensive healthcare data analysis in progress..."
    
    st.markdown(f"""
    <div class="workflow-footer">
        <div class="footer-status">{status_message}</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

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

def display_code_with_explanation(code: str, explanation: str, code_type: str = "Code"):
    """Display a code with its explanation in a formatted way"""
    if code and explanation and explanation != "Explanation unavailable" and explanation != "Explanation not available" and not explanation.startswith("Click"):
        st.markdown(f"""
        <div style="margin: 0.5rem 0;">
            <span class="code-container">{code_type}: {code}</span>
            <div class="code-explanation">
                💡 <strong>Explanation:</strong> {explanation}
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif explanation.startswith("Click"):
        st.markdown(f"""
        <div style="margin: 0.5rem 0;">
            <span class="code-container">{code_type}: {code}</span>
            <div style="background: #fff3cd; padding: 0.8rem; border-radius: 6px; border-left: 3px solid #ffc107; margin: 0.5rem 0; font-size: 0.9rem; color: #856404;">
                🔮 <strong>LLM Explanation:</strong> {explanation}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="margin: 0.5rem 0;">
            <span class="code-container">{code_type}: {code}</span>
        </div>
        """, unsafe_allow_html=True)
        if explanation == "Explanation unavailable":
            st.caption("⚠️ Explanation temporarily unavailable")

def display_enhanced_medical_extraction(structured_extractions, agent):
    """Display medical extraction with LLM code explanations"""
    medical_extraction = safe_get(structured_extractions, 'medical', {})
    if medical_extraction and not medical_extraction.get('error'):
        extraction_summary = safe_get(medical_extraction, 'extraction_summary', {})
        
        st.markdown("**📊 Medical Claims Extraction Summary:**")
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
            st.markdown("**📋 Medical Records with LLM Code Explanations:**")
            st.info("💡 Each medical code and diagnosis code now includes a 1-2 line explanation from our LLM!")
            
            # Initialize explanation cache in session state
            if 'medical_explanations_cache' not in st.session_state:
                st.session_state.medical_explanations_cache = {}
            
            # Add "Generate All Explanations" button
            if st.button("🔮 Generate All LLM Explanations", key="generate_medical_explanations"):
                with st.spinner("🤖 Generating LLM explanations for all medical codes... Please wait."):
                    progress_bar = st.progress(0)
                    for i, record in enumerate(hlth_srvc_records):
                        record_key = f"medical_record_{i}"
                        if record_key not in st.session_state.medical_explanations_cache:
                            if agent:
                                try:
                                    explanations = agent.get_code_explanations_for_record(record, "medical")
                                    st.session_state.medical_explanations_cache[record_key] = explanations
                                except Exception as e:
                                    st.session_state.medical_explanations_cache[record_key] = {"error": str(e)}
                        progress_bar.progress((i + 1) / len(hlth_srvc_records))
                    
                    st.success("✅ All LLM explanations generated successfully!")
                    st.balloons()
            
            # Display records with cached explanations
            for i, record in enumerate(hlth_srvc_records, 1):
                service_code = record.get('hlth_srvc_cd', 'N/A')
                record_key = f"medical_record_{i-1}"
                
                with st.expander(f"Medical Record {i} - Service Code: {service_code}"):
                    # Get explanations from cache or show placeholder
                    if record_key in st.session_state.medical_explanations_cache:
                        explanations = st.session_state.medical_explanations_cache[record_key]
                        if "error" in explanations:
                            st.warning(f"Could not get explanations: {explanations['error']}")
                            explanations = {}
                    else:
                        explanations = {}
                        st.info("🔮 Click 'Generate All LLM Explanations' button above to get AI explanations for this record.")
                    
                    # Display service code with explanation
                    if service_code != 'N/A':
                        service_explanation = explanations.get("service_code_explanation", "Click 'Generate All LLM Explanations' to get explanation")
                        display_code_with_explanation(service_code, service_explanation, "Service Code")
                    
                    st.write(f"**Data Path:** `{record.get('data_path', 'N/A')}`")
                    
                    # Display claim received date if available
                    clm_rcvd_dt = record.get('clm_rcvd_dt')
                    if clm_rcvd_dt:
                        st.write(f"**Claim Received Date:** `{clm_rcvd_dt}`")
                    
                    # Display diagnosis codes with explanations
                    diagnosis_codes = record.get('diagnosis_codes', [])
                    if diagnosis_codes:
                        st.write("**Diagnosis Codes with LLM Explanations:**")
                        diagnosis_explanations = explanations.get("diagnosis_explanations", [])
                        
                        for idx, diag in enumerate(diagnosis_codes, 1):
                            diag_code = diag.get('code', 'N/A')
                            source_info = f" (from {diag.get('source', 'individual field')})" if diag.get('source') else ""
                            
                            # Find matching explanation
                            diag_explanation = "Click 'Generate All LLM Explanations' to get explanation"
                            for expl in diagnosis_explanations:
                                if expl.get('code') == diag_code:
                                    diag_explanation = expl.get('explanation', "Explanation unavailable")
                                    break
                            
                            st.write(f"**{idx}. Diagnosis Code {source_info}:**")
                            display_code_with_explanation(diag_code, diag_explanation, "ICD-10")
    else:
        st.warning("No medical claims extraction data available")

def display_enhanced_pharmacy_extraction(structured_extractions, agent):
    """Display pharmacy extraction with LLM code explanations"""
    pharmacy_extraction = safe_get(structured_extractions, 'pharmacy', {})
    if pharmacy_extraction and not pharmacy_extraction.get('error'):
        extraction_summary = safe_get(pharmacy_extraction, 'extraction_summary', {})
        
        st.markdown("**📊 Pharmacy Claims Extraction Summary:**")
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
            st.markdown("**💊 Pharmacy Records with LLM Code Explanations:**")
            st.info("💡 Each NDC code and medication name now includes a 1-2 line explanation from our LLM!")
            
            # Initialize explanation cache in session state
            if 'pharmacy_explanations_cache' not in st.session_state:
                st.session_state.pharmacy_explanations_cache = {}
            
            # Add "Generate All Explanations" button
            if st.button("🔮 Generate All LLM Explanations", key="generate_pharmacy_explanations"):
                with st.spinner("🤖 Generating LLM explanations for all pharmacy codes... Please wait."):
                    progress_bar = st.progress(0)
                    for i, record in enumerate(ndc_records):
                        record_key = f"pharmacy_record_{i}"
                        if record_key not in st.session_state.pharmacy_explanations_cache:
                            if agent:
                                try:
                                    explanations = agent.get_code_explanations_for_record(record, "pharmacy")
                                    st.session_state.pharmacy_explanations_cache[record_key] = explanations
                                except Exception as e:
                                    st.session_state.pharmacy_explanations_cache[record_key] = {"error": str(e)}
                        progress_bar.progress((i + 1) / len(ndc_records))
                    
                    st.success("✅ All LLM explanations generated successfully!")
                    st.balloons()
            
            # Display records with cached explanations
            for i, record in enumerate(ndc_records, 1):
                medication_name = record.get('lbl_nm', 'N/A')
                ndc_code = record.get('ndc', 'N/A')
                record_key = f"pharmacy_record_{i-1}"
                
                with st.expander(f"Pharmacy Record {i} - {medication_name}"):
                    # Get explanations from cache or show placeholder
                    if record_key in st.session_state.pharmacy_explanations_cache:
                        explanations = st.session_state.pharmacy_explanations_cache[record_key]
                        if "error" in explanations:
                            st.warning(f"Could not get explanations: {explanations['error']}")
                            explanations = {}
                    else:
                        explanations = {}
                        st.info("🔮 Click 'Generate All LLM Explanations' button above to get AI explanations for this record.")
                    
                    # Display NDC code with explanation
                    if ndc_code != 'N/A':
                        ndc_explanation = explanations.get("ndc_explanation", "Click 'Generate All LLM Explanations' to get explanation")
                        display_code_with_explanation(ndc_code, ndc_explanation, "NDC Code")
                    
                    # Display medication with explanation
                    if medication_name != 'N/A':
                        medication_explanation = explanations.get("medication_explanation", "Click 'Generate All LLM Explanations' to get explanation")
                        display_code_with_explanation(medication_name, medication_explanation, "Medication")
                    
                    st.write(f"**Data Path:** `{record.get('data_path', 'N/A')}`")
                    
                    # Show prescription date if available
                    rx_filled_dt = record.get('rx_filled_dt')
                    if rx_filled_dt:
                        st.write(f"**Prescription Filled Date:** `{rx_filled_dt}`")
    else:
        st.warning("No pharmacy claims extraction data available")

# Initialize session state
initialize_session_state()

# Main Title
st.markdown('<h1 class="main-header">🔬 Deep Research Health Agent</h1>', unsafe_allow_html=True)

# Display import status
if not AGENT_AVAILABLE:
    st.markdown(f'<div class="status-error">❌ Failed to import Health Agent: {import_error}</div>', unsafe_allow_html=True)
    st.stop()

# SIDEBAR CHATBOT
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
                st.info("👋 Hello! I can answer questions about the claims data analysis. Ask me anything!")
                st.info("💡 **Special Feature:** Ask about heart attack risk and I'll provide both ML model predictions and comprehensive LLM analysis for comparison!")
        
        # Chat input at bottom (always visible)
        st.markdown("---")
        user_question = st.chat_input("Ask about the claims data...")
        
        # Handle chat input
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
        # Show placeholder when chatbot is not ready
        st.title("💬 Medical Assistant")
        st.info("💤 Chatbot will be available after running health analysis")
        st.markdown("---")
        st.markdown("**Features:**")
        st.markdown("• Answer questions about claims data")
        st.markdown("• Analyze diagnoses and medications") 
        st.markdown("• Heart attack risk analysis (ML + LLM comparison)")
        st.markdown("• Extract specific dates and codes")
        st.markdown("• Provide detailed medical insights")

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
        
        # 2. RUN DEEP RESEARCH ANALYSIS BUTTON (GREEN)
        submitted = st.form_submit_button(
            "🔬 Run Deep Research Analysis", 
            use_container_width=True,
            disabled=st.session_state.analysis_running,
            type="primary"
        )

# Advanced Animation container
animation_container = st.empty()

# Show advanced professional animation when running
if st.session_state.analysis_running and st.session_state.show_animation:
    with animation_container.container():
        display_advanced_professional_workflow()

# Run Deep Research Analysis
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
                st.success("✅ Deep Research Health Agent initialized successfully")
                        
            except Exception as e:
                st.error(f"❌ Failed to initialize Deep Research Health Agent: {str(e)}")
                st.error("💡 Please check that all required modules are installed and services are running")
                st.stop()
        
        # Start analysis with advanced professional workflow
        st.session_state.analysis_running = True
        st.session_state.show_animation = True
        
        # Reset workflow
        reset_workflow()
        
        st.info("🔬 Starting Advanced Deep Research Analysis - Experience the sophisticated workflow:")
        
        try:
            # ADVANCED PROFESSIONAL STEP-BY-STEP EXECUTION
            for step_idx in range(len(st.session_state.workflow_steps)):
                st.session_state.current_step = step_idx + 1
                
                # Set current step to running
                st.session_state.workflow_steps[step_idx]['status'] = 'running'
                
                # Update display with advanced professional animation
                with animation_container.container():
                    display_advanced_professional_workflow()
                
                # Simulate processing time for visual effect
                time.sleep(2.5)  # Extended time to appreciate the advanced animations
                
                # Mark step as completed
                st.session_state.workflow_steps[step_idx]['status'] = 'completed'
                
                # Update display to show completion
                with animation_container.container():
                    display_advanced_professional_workflow()
                
                # Brief pause before next step
                time.sleep(0.8)
            
            # Execute actual analysis after animation
            with st.spinner("🔬 Executing advanced deep research analysis..."):
                results = st.session_state.agent.run_analysis(patient_data)
            
            # Store results
            st.session_state.analysis_results = results
            st.session_state.chatbot_context = results.get("chatbot_context", {})
            
            # Clear animation
            animation_container.empty()
            st.session_state.show_animation = False
            
            # Show completion
            if results.get("success", False):
                st.success("🎉 All advanced workflow steps completed successfully!")
                st.markdown('<div class="status-success">✅ Advanced deep research analysis completed successfully!</div>', unsafe_allow_html=True)
                
                if results.get("chatbot_ready", False):
                    st.success("💬 Advanced Medical Assistant is now available in the sidebar!")
                    st.info("🎯 You can ask detailed questions about the comprehensive analysis results!")
                    
                    # Force sidebar to expand
                    time.sleep(1)
                    st.rerun()
            else:
                st.warning("⚠️ Analysis completed with some issues.")
                
        except Exception as e:
            # Mark current step as error
            if st.session_state.current_step > 0:
                current_idx = st.session_state.current_step - 1
                st.session_state.workflow_steps[current_idx]['status'] = 'error'
            
            st.error(f"❌ Advanced analysis failed: {str(e)}")
            st.session_state.analysis_results = {
                "success": False,
                "error": str(e),
                "patient_data": patient_data,
                "errors": [str(e)]
            }
            animation_container.empty()
        
        finally:
            st.session_state.analysis_running = False
            st.session_state.show_animation = False

# RESULTS SECTION - Only show when analysis is complete and not running
if st.session_state.analysis_results and not st.session_state.analysis_running:
    results = st.session_state.analysis_results
    
    # Add separator
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # Show errors if any
    errors = safe_get(results, 'errors', [])
    if errors:
        st.markdown('<div class="status-error">❌ Analysis errors occurred</div>', unsafe_allow_html=True)
        with st.expander("🐛 Debug Information"):
            st.write("**Errors:**")
            for error in errors:
                st.write(f"• {error}")

    # 3. CLAIMS DATA BUTTON (NOW INCLUDING MCID)
    if st.button("📊 Claims Data", use_container_width=True, key="claims_data_btn"):
        st.session_state.show_claims_data = not st.session_state.show_claims_data
    
    if st.session_state.show_claims_data:
        st.markdown("""
        <div class="section-box">
            <div class="section-title">📊 Deidentified Claims Data</div>
        </div>
        """, unsafe_allow_html=True)
        
        deidentified_data = safe_get(results, 'deidentified_data', {})
        api_outputs = safe_get(results, 'api_outputs', {})
        
        if deidentified_data or api_outputs:
            # Updated tabs to include MCID
            tab1, tab2, tab3 = st.tabs(["🏥 Medical Claims", "💊 Pharmacy Claims", "🆔 MCID Claims"])
            
            with tab1:
                medical_data = safe_get(deidentified_data, 'medical', {})
                if medical_data:
                    st.markdown("**🏥 Deidentified Medical Claims Data:**")
                    st.markdown('<div class="json-container">', unsafe_allow_html=True)
                    st.json(medical_data)
                    st.markdown('</div>', unsafe_allow_html=True)
                    

                else:
                    st.warning("No medical claims data available")
            
            with tab2:
                pharmacy_data = safe_get(deidentified_data, 'pharmacy', {})
                if pharmacy_data:
                    st.markdown("**💊 Deidentified Pharmacy Claims Data:**")
                    st.markdown('<div class="json-container">', unsafe_allow_html=True)
                    st.json(pharmacy_data)
                    st.markdown('</div>', unsafe_allow_html=True)
                    

                else:
                    st.warning("No pharmacy claims data available")
            
            with tab3:
                # MCID data from API outputs
                mcid_data = safe_get(api_outputs, 'mcid', {})
                if mcid_data:
                    st.markdown("**🆔 MCID (Member Consumer ID) Claims Data:**")
                    st.markdown('<div class="json-container">', unsafe_allow_html=True)
                    st.json(mcid_data)
                    st.markdown('</div>', unsafe_allow_html=True)
                    

                    
                    # Show MCID summary if available
                    if mcid_data.get('status_code') == 200 and mcid_data.get('body'):
                        mcid_body = mcid_data.get('body', {})
                        st.markdown("**📋 MCID Search Summary:**")
                        
                        # Extract consumer information if available
                        consumers = mcid_body.get('consumer', [])
                        if consumers and len(consumers) > 0:
                            consumer = consumers[0]
                            st.write(f"**Consumer ID:** {consumer.get('consumerId', 'N/A')}")
                            st.write(f"**Match Score:** {consumer.get('score', 'N/A')}")
                            st.write(f"**Status:** {consumer.get('status', 'N/A')}")
                        else:
                            st.info("No consumer matches found in MCID search")
                else:
                    st.warning("No MCID claims data available")

    # 4. CLAIMS DATA EXTRACTION BUTTON WITH LLM CODE EXPLANATIONS
    if st.button("🔍 Claims Data Extraction", use_container_width=True, key="claims_extraction_btn"):
        st.session_state.show_claims_extraction = not st.session_state.show_claims_extraction
    
    if st.session_state.show_claims_extraction:
        st.markdown("""
        <div class="section-box">
            <div class="section-title">🔍 Claims Data Extraction with LLM Code Explanations</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add Clear Cache buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Medical Explanations Cache", key="clear_medical_cache"):
                if 'medical_explanations_cache' in st.session_state:
                    del st.session_state.medical_explanations_cache
                st.success("Medical explanations cache cleared!")
        
        with col2:
            if st.button("🗑️ Clear Pharmacy Explanations Cache", key="clear_pharmacy_cache"):
                if 'pharmacy_explanations_cache' in st.session_state:
                    del st.session_state.pharmacy_explanations_cache
                st.success("Pharmacy explanations cache cleared!")
        
        # Show cache status
        medical_cache_count = len(st.session_state.get('medical_explanations_cache', {}))
        pharmacy_cache_count = len(st.session_state.get('pharmacy_explanations_cache', {}))
        
        if medical_cache_count > 0 or pharmacy_cache_count > 0:
            st.info(f"📋 Cached explanations: {medical_cache_count} medical records, {pharmacy_cache_count} pharmacy records")
        
        structured_extractions = safe_get(results, 'structured_extractions', {})
        
        if structured_extractions:
            tab1, tab2 = st.tabs(["🏥 Medical Claims Extraction", "💊 Pharmacy Claims Extraction"])
            
            with tab1:
                display_enhanced_medical_extraction(structured_extractions, st.session_state.agent)
            
            with tab2:
                display_enhanced_pharmacy_extraction(structured_extractions, st.session_state.agent)

    # 5. ENHANCED ENTITY EXTRACTION BUTTON
    if st.button("🎯 Enhanced Entity Extraction", use_container_width=True, key="entity_extraction_btn"):
        st.session_state.show_entity_extraction = not st.session_state.show_entity_extraction
    
    if st.session_state.show_entity_extraction:
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


    # 6. HEALTH TRAJECTORY BUTTON
    if st.button("📈 Health Trajectory", use_container_width=True, key="health_trajectory_btn"):
        st.session_state.show_health_trajectory = not st.session_state.show_health_trajectory
    
    if st.session_state.show_health_trajectory:
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

    # 7. FINAL SUMMARY BUTTON
    if st.button("📋 Final Summary", use_container_width=True, key="final_summary_btn"):
        st.session_state.show_final_summary = not st.session_state.show_final_summary
    
    if st.session_state.show_final_summary:
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

    # 8. HEART ATTACK RISK PREDICTION BUTTON
    if st.button("❤️ Heart Attack Risk Prediction", use_container_width=True, key="heart_attack_btn"):
        st.session_state.show_heart_attack = not st.session_state.show_heart_attack
    
    if st.session_state.show_heart_attack:
        st.markdown("""
        <div class="section-box">
            <div class="section-title">❤️ Heart Attack Risk Assessment</div>
        </div>
        """, unsafe_allow_html=True)
        
        heart_attack_prediction = safe_get(results, 'heart_attack_prediction', {})
        if heart_attack_prediction and not heart_attack_prediction.get('error'):
            # Display simplified format
            combined_display = heart_attack_prediction.get("combined_display", "Heart Disease Risk: Not available")
            
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 2rem; border-radius: 10px; border: 1px solid #dee2e6; margin: 1rem 0; text-align: center;">
                <h3 style="color: #2c3e50; margin-bottom: 1rem;">Heart Attack Risk Prediction</h3>
                <h4 style="color: #495057; font-weight: 600;">{combined_display}</h4>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            error_msg = heart_attack_prediction.get('error', 'Heart attack prediction not available')
            st.error(f"❌ Server Error: {error_msg}")
            
            # Show connection info for debugging
            st.info(f"💡 Expected Server: {st.session_state.config.heart_attack_api_url if st.session_state.config else 'http://localhost:8080'}")
            st.info("💡 Make sure server is running: `python app.py`")
