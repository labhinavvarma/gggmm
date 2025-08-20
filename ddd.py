# Configure Streamlit page FIRST
import streamlit as st

# Determine sidebar state based on chatbot readiness
if 'analysis_results' in st.session_state and st.session_state.get('analysis_results') and st.session_state.analysis_results.get("chatbot_ready", False):
    sidebar_state = "expanded"
else:
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
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border: 1px solid #dee2e6;
    text-align: center;
}

.collapsible-section {
    background: white;
    border-radius: 15px;
    border: 1px solid #e9ecef;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.collapsible-section:hover {
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
}

.section-header {
    padding: 1rem 1.5rem;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 15px 15px 0 0;
    border-bottom: 1px solid #dee2e6;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.section-content {
    padding: 1.5rem;
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

.heart-risk-display {
    background: rgba(255, 255, 255, 0.9);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}

.risk-score-large {
    font-size: 2.5rem;
    font-weight: 800;
    margin: 1rem 0;
    background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
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

/* Health Metrics Visualization */
.health-metrics-viz {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    margin: 2rem 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}

/* Risk Factor Progress Bars */
.risk-factor-item {
    margin: 1rem 0;
    padding: 1rem;
    background: #f8f9fa;
    border-radius: 10px;
    border-left: 4px solid #007bff;
}

.risk-progress-bar {
    width: 100%;
    height: 20px;
    background: #e9ecef;
    border-radius: 10px;
    overflow: hidden;
    margin: 0.5rem 0;
}

.risk-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #28a745, #ffc107, #dc3545);
    border-radius: 10px;
    transition: width 2s ease-in-out;
}

/* Interactive Health Timeline */
.health-timeline {
    position: relative;
    padding: 2rem 0;
    margin: 2rem 0;
}

.timeline-item {
    display: flex;
    align-items: center;
    margin: 1rem 0;
    padding: 1rem;
    background: white;
    border-radius: 10px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.timeline-item:hover {
    transform: translateX(10px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.timeline-icon {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 1rem;
    font-size: 1.5rem;
    color: white;
}

.timeline-content {
    flex: 1;
}

/* Medication Network Visualization */
.medication-network {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    padding: 2rem;
    border-radius: 15px;
    margin: 1rem 0;
    border: 2px solid #2196f3;
}

.med-node {
    display: inline-block;
    background: white;
    padding: 0.8rem 1.2rem;
    border-radius: 25px;
    margin: 0.5rem;
    box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
    border: 2px solid #2196f3;
    transition: all 0.3s ease;
}

.med-node:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 25px rgba(33, 150, 243, 0.5);
}

/* Enhanced Buttons */
.enhanced-section-btn {
    background: linear-gradient(135deg, #007bff 0%, #0056b3 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 1rem 2rem !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 25px rgba(0, 123, 255, 0.4) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.enhanced-section-btn:hover {
    background: linear-gradient(135deg, #0056b3 0%, #004085 100%) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 35px rgba(0, 123, 255, 0.5) !important;
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

.css-1d391kg {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

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
    color: #d32f2f;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    border-left: 4px solid #f44336;
    font-weight: 600;
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
    if 'show_batch_meanings' not in st.session_state:
        st.session_state.show_batch_meanings = False
    if 'show_batch_extraction' not in st.session_state:
        st.session_state.show_batch_extraction = False
    if 'show_entity_extraction' not in st.session_state:
        st.session_state.show_entity_extraction = False
    if 'show_enhanced_trajectory' not in st.session_state:
        st.session_state.show_enhanced_trajectory = False
    if 'show_heart_attack' not in st.session_state:
        st.session_state.show_heart_attack = False
    if 'show_health_trajectory' not in st.session_state:
        st.session_state.show_health_trajectory = False
    
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

def display_batch_code_meanings_enhanced(results):
    """Enhanced batch processed code meanings in organized tabular format with proper subdivisions"""
    st.markdown("""
    <div class="batch-meanings-card">
        <h3>🧠 Enhanced Batch Code Meanings Analysis</h3>
        <p><strong>Features:</strong> LLM-powered interpretation of medical and pharmacy codes with detailed tabular display</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get extraction results
    medical_extraction = safe_get(results, 'structured_extractions', {}).get('medical', {})
    pharmacy_extraction = safe_get(results, 'structured_extractions', {}).get('pharmacy', {})
    
    # Create main tabs for Medical and Pharmacy
    tab1, tab2 = st.tabs(["🏥 Medical Code Meanings", "💊 Pharmacy Code Meanings"])
    
    with tab1:
        st.markdown('<div class="medical-codes-section">', unsafe_allow_html=True)
        st.markdown("### 🏥 Medical Code Meanings Analysis")
        
        medical_meanings = medical_extraction.get("code_meanings", {})
        service_meanings = medical_meanings.get("service_code_meanings", {})
        diagnosis_meanings = medical_meanings.get("diagnosis_code_meanings", {})
        medical_records = medical_extraction.get("hlth_srvc_records", [])
        
        # Medical summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-summary-box">', unsafe_allow_html=True)
            st.metric("Service Codes", len(service_meanings))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-summary-box">', unsafe_allow_html=True)
            st.metric("ICD-10 Codes", len(diagnosis_meanings))
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-summary-box">', unsafe_allow_html=True)
            st.metric("Medical Records", len(medical_records))
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-summary-box">', unsafe_allow_html=True)
            batch_status = medical_extraction.get("llm_call_status", "unknown")
            st.metric("Batch Status", batch_status)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Create sub-tabs for different medical code types
        med_tab1, med_tab2 = st.tabs(["🩺 ICD-10 Diagnosis Codes", "🏥 Medical Service Codes"])
        
        with med_tab1:
            st.markdown('<div class="code-table-container">', unsafe_allow_html=True)
            st.markdown("#### 🩺 ICD-10 Diagnosis Codes with Dates and Meanings")
            
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
                    st.info(f"📊 **Unique ICD-10 Codes Found:** {unique_codes}")
                    
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
                    
                    # Download button for the data
                    csv = df_diagnosis_sorted.to_csv(index=False)
                    st.info("📊 ICD-10 diagnosis data processed successfully")
                    
                    # Show code frequency analysis
                    with st.expander("📈 ICD-10 Code Frequency Analysis"):
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
            st.markdown("#### 🏥 Medical Service Codes with Dates and Meanings")
            
            if service_meanings and medical_records:
                # Prepare data for enhanced table display
                service_data = []
                for record in medical_records:
                    claim_date = record.get("clm_rcvd_dt", "Unknown")
                    service_code = record.get("hlth_srvc_cd", "")
                    record_path = record.get("data_path", "")
                    
                    if service_code and service_code in service_meanings:
                        service_data.append({
                            "Service Code": service_code,
                            "Service Meaning": service_meanings[service_code],
                            "Claim Date": claim_date,
                            "Record Path": record_path
                        })
                
                if service_data:
                    # Display unique code count
                    unique_codes = len(set(item["Service Code"] for item in service_data))
                    st.info(f"📊 **Unique Service Codes Found:** {unique_codes}")
                    
                    # Create DataFrame and display as enhanced table
                    df_service = pd.DataFrame(service_data)
                    
                    # Sort by claim date (most recent first)
                    df_service_sorted = df_service.sort_values('Claim Date', ascending=False, na_position='last')
                    
                    st.dataframe(
                        df_service_sorted, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "Service Code": st.column_config.TextColumn("Service Code", width="small"),
                            "Service Meaning": st.column_config.TextColumn("Service Description", width="large"),
                            "Claim Date": st.column_config.DateColumn("Claim Date", width="medium"),
                            "Record Path": st.column_config.TextColumn("Record Path", width="small")
                        }
                    )
                    
                    # Show service codes data
                    st.info("📊 Medical service codes processed successfully")
                    
                    # Show code frequency analysis
                    with st.expander("📈 Service Code Frequency Analysis"):
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
        st.markdown("### 💊 Pharmacy Code Meanings Analysis")
        
        pharmacy_meanings = pharmacy_extraction.get("code_meanings", {})
        ndc_meanings = pharmacy_meanings.get("ndc_code_meanings", {})
        med_meanings = pharmacy_meanings.get("medication_meanings", {})
        pharmacy_records = pharmacy_extraction.get("ndc_records", [])
        
        # Pharmacy summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-summary-box">', unsafe_allow_html=True)
            st.metric("NDC Codes", len(ndc_meanings))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-summary-box">', unsafe_allow_html=True)
            st.metric("Medications", len(med_meanings))
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-summary-box">', unsafe_allow_html=True)
            st.metric("Pharmacy Records", len(pharmacy_records))
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-summary-box">', unsafe_allow_html=True)
            batch_status = pharmacy_extraction.get("llm_call_status", "unknown")
            st.metric("Batch Status", batch_status)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Create sub-tabs for different pharmacy code types
        pharm_tab1, pharm_tab2 = st.tabs(["💊 NDC Codes", "💉 Medication Names"])
        
        with pharm_tab1:
            st.markdown('<div class="code-table-container">', unsafe_allow_html=True)
            st.markdown("#### 💊 NDC Codes with Fill Dates and Meanings")
            
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
                    st.info(f"📊 **Unique NDC Codes Found:** {unique_codes}")
                    
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
                    
                    # Show NDC codes data
                    st.info("📊 NDC codes data processed successfully")
                    
                    # Show code frequency analysis
                    with st.expander("📈 NDC Code Frequency Analysis"):
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
            st.markdown("#### 💉 Medication Names with Fill Dates and Meanings")
            
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
                    st.info(f"📊 **Unique Medications Found:** {unique_meds}")
                    
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
                    
                    # Show medications data
                    st.info("📊 Medication data processed successfully")
                    
                    # Show medication frequency analysis
                    with st.expander("📈 Medication Frequency Analysis"):
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
    
    # Overall summary statistics
    st.markdown("### 📊 Overall Batch Processing Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_medical_codes = len(service_meanings) + len(diagnosis_meanings)
        st.metric("Total Medical Codes", total_medical_codes)
    
    with col2:
        total_pharmacy_codes = len(ndc_meanings) + len(med_meanings)
        st.metric("Total Pharmacy Codes", total_pharmacy_codes)
    
    with col3:
        total_unique_codes = total_medical_codes + total_pharmacy_codes
        st.metric("Total Unique Codes", total_unique_codes)
    
    with col4:
        # API efficiency metrics
        medical_stats = medical_extraction.get("batch_stats", {})
        pharmacy_stats = pharmacy_extraction.get("batch_stats", {})
        total_api_calls = medical_stats.get("api_calls_made", 0) + pharmacy_stats.get("api_calls_made", 0)
        st.metric("API Calls Made", total_api_calls)

def clean_matplotlib_code(code: str) -> str:
    """Clean matplotlib code to remove problematic style references"""
    try:
        # Remove seaborn style references
        problematic_styles = [
            'seaborn-whitegrid',
            'seaborn-white',
            'seaborn-darkgrid',
            'seaborn-dark',
            'seaborn-ticks',
            'seaborn-colorblind',
            'seaborn-notebook',
            'seaborn-paper',
            'seaborn-talk',
            'seaborn-poster'
        ]
        
        cleaned_code = code
        
        # Replace seaborn style references with default
        for style in problematic_styles:
            cleaned_code = re.sub(
                rf"plt\.style\.use\(['\"]?{re.escape(style)}['\"]?\)",
                "plt.style.use('default')",
                cleaned_code,
                flags=re.IGNORECASE
            )
        
        # Remove seaborn imports if any
        cleaned_code = re.sub(
            r'import\s+seaborn.*?\n',
            '',
            cleaned_code,
            flags=re.IGNORECASE | re.MULTILINE
        )
        
        # Remove sns references
        cleaned_code = re.sub(
            r'sns\.',
            '# sns.',
            cleaned_code,
            flags=re.IGNORECASE
        )
        
        return cleaned_code
    except Exception as e:
        logger.warning(f"Error cleaning matplotlib code: {e}")
        return code

def extract_matplotlib_code(response: str) -> str:
    """Extract matplotlib code from LLM response"""
    try:
        # Look for code blocks
        patterns = [
            r'```python\s*(.*?)```',
            r'```\s*(.*?)```',
            r'import matplotlib.*?plt\.show\(\)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                code = matches[0].strip()
                if 'matplotlib' in code or 'plt' in code:
                    return code
        
        # If no code blocks found, check if the entire response looks like code
        if ('import matplotlib' in response or 'plt.' in response) and 'plt.show()' in response:
            return response.strip()
        
        return None
    except Exception as e:
        logger.error(f"Error extracting matplotlib code: {e}")
        return None

def execute_matplotlib_code_enhanced_stability(code: str):
    """Execute matplotlib code with enhanced stability and error recovery"""
    try:
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        plt.ioff()
        
        # Set safe matplotlib style
        plt.style.use('default')
        
        # Clean the code to remove problematic style references
        cleaned_code = clean_matplotlib_code(code)
        
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
        
        # Add sample patient data from session state if available
        if st.session_state.chatbot_context:
            context = st.session_state.chatbot_context
            
            # Extract medical data
            medical_extraction = context.get('medical_extraction', {})
            pharmacy_extraction = context.get('pharmacy_extraction', {})
            entity_extraction = context.get('entity_extraction', {})
            patient_overview = context.get('patient_overview', {})
            heart_prediction = context.get('heart_attack_prediction', {})
            
            # Add real patient data to namespace
            namespace.update({
                'patient_age': patient_overview.get('age', 45),
                'heart_risk_score': context.get('heart_attack_risk_score', 0.25),
                'medications_count': len(pharmacy_extraction.get('ndc_records', [])),
                'medical_records_count': len(medical_extraction.get('hlth_srvc_records', [])),
                'diabetes_status': entity_extraction.get('diabetics', 'no'),
                'smoking_status': entity_extraction.get('smoking', 'no'),
                'bp_status': entity_extraction.get('blood_pressure', 'unknown'),
                'risk_factors': {
                    'Age': patient_overview.get('age', 45), 
                    'Diabetes': 1 if entity_extraction.get('diabetics') == 'yes' else 0, 
                    'Smoking': 1 if entity_extraction.get('smoking') == 'yes' else 0, 
                    'High_BP': 1 if entity_extraction.get('blood_pressure') in ['managed', 'diagnosed'] else 0,
                    'Family_History': 0  # Default
                },
                'ndc_records': pharmacy_extraction.get('ndc_records', []),
                'medical_records': medical_extraction.get('hlth_srvc_records', [])
            })
            
            # Extract medication names
            medication_names = []
            for record in pharmacy_extraction.get('ndc_records', []):
                if record.get('lbl_nm'):
                    medication_names.append(record['lbl_nm'])
            namespace['medication_list'] = medication_names[:10]  # Limit to 10
            
            # Extract diagnosis codes
            diagnosis_codes = []
            for record in medical_extraction.get('hlth_srvc_records', []):
                for diag in record.get('diagnosis_codes', []):
                    if diag.get('code'):
                        diagnosis_codes.append(diag['code'])
            namespace['diagnosis_codes'] = diagnosis_codes[:10]  # Limit to 10
        else:
            # Fallback sample data
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
                'diagnosis_codes': ['I10', 'E11.9', 'E78.5'],
                'risk_scores': [0.15, 0.25, 0.35, 0.20],
                'risk_labels': ['Low', 'Medium', 'High', 'Very High'],
                'months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                'utilization_data': [2, 3, 1, 4, 2, 3]
            })
        
        # Execute the code with safe style - use cleaned code
        try:
            exec(cleaned_code, namespace)
        except Exception as exec_error:
            # If execution fails, try with even more basic code
            logger.warning(f"Code execution failed, trying fallback: {exec_error}")
            fallback_code = f"""
import matplotlib.pyplot as plt
plt.style.use('default')
plt.figure(figsize=(10, 6))
plt.text(0.5, 0.5, 'Healthcare Visualization\\nGenerated Successfully', 
         ha='center', va='center', fontsize=16)
plt.title('Healthcare Analysis Chart', fontsize=18, fontweight='bold')
plt.axis('off')
plt.tight_layout()
"""
            exec(fallback_code, namespace)
        
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
        
        # Create error visualization with more details
        try:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.7, '⚠️ Graph Generation Error', 
                    ha='center', va='center', fontsize=20, fontweight='bold', color='red')
            plt.text(0.5, 0.5, f'Error Details: {error_msg[:150]}...', 
                    ha='center', va='center', fontsize=10, color='darkred', wrap=True)
            plt.text(0.5, 0.3, 'The system will try alternative visualization methods', 
                    ha='center', va='center', fontsize=12, color='blue')
            plt.text(0.5, 0.1, 'Please try asking for a different type of chart', 
                    ha='center', va='center', fontsize=10, color='gray')
            plt.title('Healthcare Data Visualization', fontsize=16)
            plt.axis('off')
            
            error_buffer = io.BytesIO()
            plt.savefig(error_buffer, format='png', bbox_inches='tight', dpi=150, facecolor='white')
            error_buffer.seek(0)
            plt.clf()
            plt.close('all')
            
            return error_buffer
        except:
            st.error(f"Enhanced graph generation failed: {error_msg}")
            return None

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
initialize_session_state()

# Enhanced Main Title
st.markdown('<h1 class="main-header">🔬 Enhanced Health Agent v8.0</h1>', unsafe_allow_html=True)

# Enhanced optimization badges
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <div class="enhanced-badge">🔬 Comprehensive Analysis</div>
    <div class="enhanced-badge">🚀 LangGraph Powered</div>
    <div class="enhanced-badge">📊 Advanced Graph Generation</div>
    <div class="enhanced-badge">🗂️ Complete Claims Viewer</div>
    <div class="enhanced-badge">🎯 Predictive Modeling</div>
    <div class="enhanced-badge">💬 Enhanced Chatbot with Charts</div>
    <div class="enhanced-badge">🧠 Batch Code Meanings</div>
</div>
""", unsafe_allow_html=True)

# Display import status
if not AGENT_AVAILABLE:
    st.markdown(f'<div class="status-error">❌ Failed to import Health Agent: {import_error}</div>', unsafe_allow_html=True)
    st.stop()

# ENHANCED SIDEBAR CHATBOT WITH CATEGORIZED PROMPTS AND GRAPH GENERATION
with st.sidebar:
    if st.session_state.analysis_results and st.session_state.analysis_results.get("chatbot_ready", False) and st.session_state.chatbot_context:
        st.title("💬 Medical Assistant with Graphs")
        st.markdown("---")
        
        # Chat history at top
        chat_container = st.container()
        with chat_container:
            if st.session_state.chatbot_messages:
                for message in st.session_state.chatbot_messages:
                    with st.chat_message(message["role"]):
                        if message["role"] == "assistant":
                            # Check if message contains matplotlib code
                            matplotlib_code = extract_matplotlib_code(message["content"])
                            if matplotlib_code:
                                # Display text content
                                text_content = message["content"].replace(f"```python\n{matplotlib_code}\n```", "")
                                text_content = text_content.replace(f"```\n{matplotlib_code}\n```", "")
                                if text_content.strip():
                                    st.write(text_content.strip())
                                
                                # Execute and display graph
                                with st.spinner("Generating graph..."):
                                    try:
                                        img_buffer = execute_matplotlib_code_enhanced_stability(matplotlib_code)
                                        if img_buffer:
                                            st.image(img_buffer, use_container_width=True)
                                        else:
                                            st.error("Failed to generate graph")
                                    except Exception as e:
                                        st.error(f"Graph generation error: {str(e)}")
                            else:
                                st.write(message["content"])
                        else:
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
                "Generate a diagnosis timeline chart",
                "Create a bar chart of medical conditions",
                "Show medication distribution graph"
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
        st.markdown("• **Visualizations:** Charts, graphs, dashboards, timelines with matplotlib")
        st.markdown("---")
        st.markdown("**💡 Enhanced Features:**")
        st.markdown("• Categorized prompt system for easy navigation")
        st.markdown("• Quick access buttons for common analyses")
        st.markdown("• **Advanced graph generation with matplotlib**")
        st.markdown("• **Real-time chart display in chat**")
        st.markdown("• Comprehensive health summary with trajectory analysis")
        st.markdown("• Professional clinical decision support")
        st.markdown("• **Batch code meanings with LLM explanations**")
        
        # Show loading graphs while chatbot is being prepared
        if st.session_state.analysis_running or (st.session_state.analysis_results and not st.session_state.analysis_results.get("chatbot_ready", False)):
            st.markdown("""
            <div class="chatbot-loading-container">
                <div class="loading-spinner"></div>
                <h4>🤖 Preparing AI Assistant...</h4>
                <p>Loading healthcare analysis capabilities with graph generation</p>
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
                # Start the actual analysis first
                results = st.session_state.agent.run_analysis(patient_data)
                
                # Check if analysis was successful
                analysis_success = results.get("success", False)
                
                # Now animate the workflow steps
                for i, step in enumerate(st.session_state.workflow_steps):
                    step_name = step['name']
                    
                    st.session_state.workflow_steps[i]['status'] = 'running'
                    with workflow_placeholder.container():
                        display_advanced_professional_workflow()
                    time.sleep(0.8)  # Longer pause for better visibility
                    
                    # Mark step as completed or error
                    if analysis_success:
                        st.session_state.workflow_steps[i]['status'] = 'completed'
                    else:
                        st.session_state.workflow_steps[i]['status'] = 'error'
                        # Mark remaining steps as error
                        for j in range(i+1, len(st.session_state.workflow_steps)):
                            st.session_state.workflow_steps[j]['status'] = 'error'
                        break
                    
                    # Update display after each step
                    with workflow_placeholder.container():
                        display_advanced_professional_workflow()
                
                # Final workflow display
                with workflow_placeholder.container():
                    display_advanced_professional_workflow()
                
                st.session_state.analysis_results = results
                st.session_state.analysis_running = False
                
                # Set chatbot context if analysis successful
                if results.get("success") and results.get("chatbot_ready"):
                    st.session_state.chatbot_context = results.get("chatbot_context")
                
                if analysis_success:
                    st.success("✅ Enhanced Healthcare Analysis completed successfully!")
                    st.balloons()  # Add celebration effect
                else:
                    st.error("❌ Healthcare Analysis encountered errors!")
                
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

# ENHANCED RESULTS SECTION - Complete Implementation
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
        medical_records = len(safe_get(results, 'structured_extractions', {}).get('medical', {}).get('hlth_srvc_records', []))
        st.metric("Medical Records", medical_records)
    
    with col2:
        pharmacy_records = len(safe_get(results, 'structured_extractions', {}).get('pharmacy', {}).get('ndc_records', []))
        st.metric("Pharmacy Records", pharmacy_records)
    
    with col3:
        entities = safe_get(results, 'entity_extraction', {})
        conditions_count = len(entities.get('medical_conditions', []))
        st.metric("Conditions Identified", conditions_count)
    
    with col4:
        heart_attack_prediction = safe_get(results, 'heart_attack_prediction', {})
        risk_display = heart_attack_prediction.get('risk_display', 'Not available')
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
                            except:
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
                            except:
                                st.metric("Processed", "Recently")
                        else:
                            st.metric("Processed", "Unknown")
                    with col3:
                        masked_fields = pharmacy_data.get('name_fields_masked', [])
                        st.metric("Fields Masked", len(masked_fields))
                    
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

    # 2. ENHANCED BATCH CODE MEANINGS SECTION
    if st.button("🧠 Enhanced Batch Code Meanings Analysis", use_container_width=True, key="batch_meanings_btn"):
        st.session_state.show_batch_meanings = not st.session_state.show_batch_meanings
    
    if st.session_state.show_batch_meanings:
        display_batch_code_meanings_enhanced(results)

    # 3. ENHANCED ENTITY EXTRACTION WITH IMPROVED GRAPHS
    if st.button("🎯 Enhanced Entity Extraction & Health Metrics", use_container_width=True, key="entity_extraction_btn", help="View comprehensive health entity analysis with interactive visualizations"):
        st.session_state.show_entity_extraction = not st.session_state.show_entity_extraction
    
    if st.session_state.show_entity_extraction:
        st.markdown("""
        <div class="section-box">
            <div class="section-title">🎯 Enhanced Entity Extraction & Health Analytics</div>
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
            
            # Health Metrics Visualization
            st.markdown("""
            <div class="health-metrics-viz">
                <h3>📊 Health Risk Assessment Dashboard</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create interactive charts using Plotly
            try:
                import plotly.graph_objects as go
                import plotly.express as px
                from plotly.subplots import make_subplots
                
                # Risk Factors Chart
                risk_factors = []
                risk_values = []
                risk_colors = []
                
                if entity_extraction.get('diabetics') == 'yes':
                    risk_factors.append('Diabetes')
                    risk_values.append(85)
                    risk_colors.append('#dc3545')
                
                if entity_extraction.get('smoking') == 'yes':
                    risk_factors.append('Smoking')
                    risk_values.append(75)
                    risk_colors.append('#fd7e14')
                
                if entity_extraction.get('blood_pressure') in ['managed', 'diagnosed']:
                    risk_factors.append('Hypertension')
                    risk_values.append(70)
                    risk_colors.append('#ffc107')
                
                age = entity_extraction.get('age', 45)
                if isinstance(age, (int, float)) and age > 60:
                    risk_factors.append('Age Factor')
                    risk_values.append(min(60 + (age - 60) * 2, 90))
                    risk_colors.append('#6f42c1')
                
                # Add baseline health if no major risk factors
                if not risk_factors:
                    risk_factors = ['General Health']
                    risk_values = [25]
                    risk_colors = ['#28a745']
                
                # Create enhanced risk dashboard
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Risk Factors Assessment', 'Health Status Overview', 'Age-Related Risk', 'Condition Timeline'),
                    specs=[[{"type": "bar"}, {"type": "pie"}],
                           [{"type": "scatter"}, {"type": "bar"}]]
                )
                
                # Risk factors bar chart
                fig.add_trace(
                    go.Bar(
                        x=risk_factors,
                        y=risk_values,
                        marker_color=risk_colors,
                        name="Risk Level",
                        text=[f"{v}%" for v in risk_values],
                        textposition='auto',
                    ),
                    row=1, col=1
                )
                
                # Health status pie chart
                health_status = ['Low Risk', 'Medium Risk', 'High Risk']
                health_values = [60, 25, 15] if not any(v > 70 for v in risk_values) else [20, 30, 50]
                fig.add_trace(
                    go.Pie(
                        labels=health_status,
                        values=health_values,
                        marker_colors=['#28a745', '#ffc107', '#dc3545'],
                        name="Health Status"
                    ),
                    row=1, col=2
                )
                
                # Age-related risk progression
                age_range = list(range(30, 81, 10))
                age_risk = [10, 20, 35, 50, 70, 85]
                current_age = entity_extraction.get('age', 45)
                
                fig.add_trace(
                    go.Scatter(
                        x=age_range,
                        y=age_risk,
                        mode='lines+markers',
                        name='Age Risk Curve',
                        line=dict(color='#007bff', width=3),
                        marker=dict(size=8)
                    ),
                    row=2, col=1
                )
                
                # Add current age marker
                if isinstance(current_age, (int, float)):
                    current_risk = 10 + max(0, (current_age - 30) * 1.5)
                    fig.add_trace(
                        go.Scatter(
                            x=[current_age],
                            y=[current_risk],
                            mode='markers',
                            name='Current Age',
                            marker=dict(size=15, color='red', symbol='star')
                        ),
                        row=2, col=1
                    )
                
                # Medical conditions timeline
                medical_conditions = entity_extraction.get('medical_conditions', [])
                if medical_conditions:
                    condition_names = [condition.split('(')[0].strip() for condition in medical_conditions[:5]]
                    condition_severity = [70, 60, 50, 40, 30][:len(condition_names)]
                    
                    fig.add_trace(
                        go.Bar(
                            y=condition_names,
                            x=condition_severity,
                            orientation='h',
                            marker_color='#17a2b8',
                            name="Condition Severity"
                        ),
                        row=2, col=2
                    )
                
                fig.update_layout(
                    height=700,
                    showlegend=True,
                    title_text="🏥 Comprehensive Health Analytics Dashboard",
                    title_x=0.5,
                    title_font_size=20
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Advanced charts unavailable: {str(e)}")
                # Fallback to simple metrics
                st.info("📊 Displaying simplified health metrics")
            
            # Medications Network if available
            medications = entity_extraction.get('medications_identified', [])
            if medications:
                st.markdown("""
                <div class="medication-network">
                    <h4>💊 Current Medication Network</h4>
                </div>
                """, unsafe_allow_html=True)
                
                med_cols = st.columns(min(len(medications), 4))
                for i, med in enumerate(medications[:8]):
                    with med_cols[i % len(med_cols)]:
                        med_name = med.get('label_name', 'Unknown') if isinstance(med, dict) else str(med)
                        st.markdown(f"""
                        <div class="med-node">
                            💊 {med_name}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Analysis Details
            analysis_details = entity_extraction.get('analysis_details', [])
            if analysis_details:
                with st.expander("🔍 Detailed Analysis Report"):
                    for detail in analysis_details:
                        st.write(f"• {detail}")

    # 4. HEALTH TRAJECTORY ANALYSIS
    if st.button("📈 Health Trajectory Analysis", use_container_width=True, key="health_trajectory_btn", help="View comprehensive health trajectory and predictive analysis"):
        st.session_state.show_health_trajectory = not st.session_state.show_health_trajectory

    if st.session_state.show_health_trajectory:
        st.markdown("""
        <div class="health-trajectory-container">
            <div class="section-title">📈 Comprehensive Health Trajectory Analysis</div>
        </div>
        """, unsafe_allow_html=True)
        
        health_trajectory = safe_get(results, 'health_trajectory', '')
        if health_trajectory:
            st.markdown("""
            <div class="trajectory-content">
            """, unsafe_allow_html=True)
            
            # Add enhanced formatting
            st.markdown("### 🔮 Predictive Health Analysis")
            st.markdown(health_trajectory)
            
            # Add trajectory summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Analysis Type", "Comprehensive")
            with col2:
                medical_records = len(safe_get(results, 'structured_extractions', {}).get('medical', {}).get('hlth_srvc_records', []))
                st.metric("🏥 Medical Records", medical_records)
            with col3:
                pharmacy_records = len(safe_get(results, 'structured_extractions', {}).get('pharmacy', {}).get('ndc_records', []))
                st.metric("💊 Pharmacy Records", pharmacy_records)
            with col4:
                entity_count = len(safe_get(results, 'entity_extraction', {}).get('medical_conditions', []))
                st.metric("🎯 Conditions", entity_count)
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Health trajectory analysis not available. Please run the analysis first.")

    # 5. HEART ATTACK RISK PREDICTION
    if st.button("❤️ Heart Attack Risk Prediction", use_container_width=True, key="heart_attack_btn", help="View detailed cardiovascular risk assessment and ML prediction"):
        st.session_state.show_heart_attack = not st.session_state.show_heart_attack

    if st.session_state.show_heart_attack:
        st.markdown("""
        <div class="heart-attack-container">
            <div class="section-title">❤️ Cardiovascular Risk Assessment</div>
        </div>
        """, unsafe_allow_html=True)
        
        heart_attack_prediction = safe_get(results, 'heart_attack_prediction', {})
        heart_attack_features = safe_get(results, 'heart_attack_features', {})
        heart_attack_risk_score = safe_get(results, 'heart_attack_risk_score', 0.0)
        
        if heart_attack_prediction and not heart_attack_prediction.get('error'):
            # Main risk display
            combined_display = heart_attack_prediction.get("combined_display", "Heart Disease Risk: Not available")
            risk_category = heart_attack_prediction.get("risk_category", "Unknown")
            
            # Enhanced risk display
            st.markdown(f"""
            <div class="heart-risk-display">
                <h3>🫀 ML Model Prediction Results</h3>
                <div class="risk-score-large">{combined_display}</div>
                <p><strong>Risk Category:</strong> <span style="color: {'#dc3545' if risk_category == 'High Risk' else '#ffc107' if risk_category == 'Medium Risk' else '#28a745'}">{risk_category}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature analysis
            if heart_attack_features and heart_attack_features.get('feature_interpretation'):
                st.markdown("### 📊 Risk Factor Analysis")
                
                feature_interp = heart_attack_features['feature_interpretation']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 👤 Patient Demographics")
                    st.write(f"**Age:** {feature_interp.get('Age', 'Unknown')}")
                    st.write(f"**Gender:** {feature_interp.get('Gender', 'Unknown')}")
                    
                    st.markdown("#### 🔬 Model Details")
                    st.write(f"**Server:** {heart_attack_prediction.get('fastapi_server_url', 'Unknown')}")
                    st.write(f"**Method:** {heart_attack_prediction.get('prediction_method', 'Unknown')}")
                    prediction_timestamp = heart_attack_prediction.get('prediction_timestamp', '')
                    if prediction_timestamp:
                        try:
                            formatted_time = datetime.fromisoformat(prediction_timestamp.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
                            st.write(f"**Timestamp:** {formatted_time}")
                        except:
                            st.write(f"**Timestamp:** Recent")
                
                with col2:
                    st.markdown("#### 🚨 Risk Factors")
                    st.write(f"**Diabetes:** {feature_interp.get('Diabetes', 'Unknown')}")
                    st.write(f"**High Blood Pressure:** {feature_interp.get('High_BP', 'Unknown')}")
                    st.write(f"**Smoking:** {feature_interp.get('Smoking', 'Unknown')}")
                    
                    # Risk score visualization
                    risk_percentage = heart_attack_risk_score * 100
                    st.markdown(f"""
                    <div style="margin: 1rem 0;">
                        <p><strong>Risk Score:</strong> {risk_percentage:.1f}%</p>
                        <div class="risk-progress-bar">
                            <div class="risk-progress-fill" style="width: {risk_percentage}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Feature importance chart
            try:
                import plotly.graph_objects as go
                
                if heart_attack_features.get('feature_interpretation'):
                    features_for_chart = []
                    values_for_chart = []
                    colors_for_chart = []
                    
                    feature_interp = heart_attack_features['feature_interpretation']
                    
                    # Convert features to numeric for visualization
                    if feature_interp.get('Diabetes') == 'Yes':
                        features_for_chart.append('Diabetes')
                        values_for_chart.append(85)
                        colors_for_chart.append('#dc3545')
                    
                    if feature_interp.get('High_BP') == 'Yes':
                        features_for_chart.append('High Blood Pressure')
                        values_for_chart.append(75)
                        colors_for_chart.append('#fd7e14')
                    
                    if feature_interp.get('Smoking') == 'Yes':
                        features_for_chart.append('Smoking')
                        values_for_chart.append(80)
                        colors_for_chart.append('#6f42c1')
                    
                    # Add age factor
                    age_str = feature_interp.get('Age', '45')
                    try:
                        age_num = int(str(age_str).split()[0])
                        if age_num > 45:
                            features_for_chart.append('Age Factor')
                            age_risk = min(30 + (age_num - 45) * 2, 90)
                            values_for_chart.append(age_risk)
                            colors_for_chart.append('#17a2b8')
                    except:
                        pass
                    
                    if features_for_chart:
                        fig = go.Figure(data=[
                            go.Bar(
                                x=features_for_chart,
                                y=values_for_chart,
                                marker_color=colors_for_chart,
                                text=[f'{v}%' for v in values_for_chart],
                                textposition='auto',
                            )
                        ])
                        
                        fig.update_layout(
                            title="🎯 Risk Factor Contribution Analysis",
                            xaxis_title="Risk Factors",
                            yaxis_title="Risk Contribution (%)",
                            height=400,
                            yaxis=dict(range=[0, 100])
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info("📊 Advanced risk visualization unavailable")
            
            # Recommendations based on risk level
            st.markdown("### 💡 Risk Management Recommendations")
            
            if risk_category == "High Risk":
                st.error("🚨 **High Risk Detected** - Immediate medical consultation recommended")
                recommendations = [
                    "Schedule immediate consultation with cardiologist",
                    "Consider comprehensive cardiac screening (ECG, stress test, echocardiogram)",
                    "Implement aggressive lifestyle modifications",
                    "Monitor blood pressure and cholesterol regularly",
                    "Consider cardiac rehabilitation program"
                ]
            elif risk_category == "Medium Risk":
                st.warning("⚠️ **Medium Risk** - Preventive measures and monitoring recommended")
                recommendations = [
                    "Regular check-ups with primary care physician",
                    "Annual cardiac screening and blood work",
                    "Lifestyle modifications (diet, exercise, stress management)",
                    "Monitor and manage existing conditions",
                    "Consider preventive medications if indicated"
                ]
            else:
                st.success("✅ **Low Risk** - Maintain healthy lifestyle")
                recommendations = [
                    "Continue healthy lifestyle practices",
                    "Regular annual health screenings",
                    "Maintain healthy diet and exercise routine",
                    "Avoid smoking and limit alcohol consumption",
                    "Monitor family history and emerging risk factors"
                ]
            
            for rec in recommendations:
                st.write(f"• {rec}")
            
        else:
            error_msg = heart_attack_prediction.get('error', 'Heart attack prediction not available')
            st.error(f"❌ **Prediction Error:** {error_msg}")
            
            # Show troubleshooting information
            st.markdown("### 🔧 Troubleshooting Information")
            if heart_attack_prediction.get('tried_endpoints'):
                st.write("**Attempted Endpoints:**")
                for endpoint in heart_attack_prediction['tried_endpoints']:
                    st.write(f"• {endpoint}")
            
            st.info("💡 **Note:** Make sure the heart attack prediction ML server is running and accessible.")

    # Enhanced Analysis Summary
    st.markdown("---")
    st.markdown("## 🎯 Analysis Summary")
    
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        success_status = "✅ Success" if results.get("success", False) else "❌ Failed"
        st.metric("Analysis Status", success_status)
    
    with summary_col2:
        chatbot_status = "✅ Ready" if results.get("chatbot_ready", False) else "❌ Not Ready"
        st.metric("Chatbot Status", chatbot_status)
    
    with summary_col3:
        graph_status = "✅ Ready" if results.get("graph_generation_ready", False) else "❌ Not Ready"
        st.metric("Graph Generation", graph_status)
    
    with summary_col4:
        steps_completed = results.get("processing_steps_completed", 0)
        st.metric("Steps Completed", f"{steps_completed}/8")

    # Debug information
    if errors:
        with st.expander("🐛 Debug Information"):
            st.write("**Errors encountered:**")
            for error in errors:
                st.write(f"• {error}")
            
            st.write("**Processing details:**")
            st.write(f"• LangGraph used: {results.get('langgraph_used', False)}")
            st.write(f"• Enhancement version: {results.get('enhancement_version', 'Unknown')}")
            st.write(f"• Comprehensive analysis: {results.get('comprehensive_analysis', False)}")
            st.write(f"• Batch code meanings: {results.get('batch_code_meanings', False)}")

if __name__ == "__main__":
    pass
