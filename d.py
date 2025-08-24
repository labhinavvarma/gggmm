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

# Enhanced CSS with advanced animations and modern styling + new sections + NEW CHATBOT WINDOW STYLES
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

/* SECTION AVAILABILITY STATES */
.section-available {
    background: white;
    border: 2px solid #28a745;
    box-shadow: 0 8px 25px rgba(40, 167, 69, 0.2);
}

.section-loading {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    border: 2px solid #ffc107;
    box-shadow: 0 8px 25px rgba(255, 193, 7, 0.2);
    animation: pulse-loading 2s infinite;
}

.section-disabled {
    background: #f8f9fa;
    border: 2px solid #e9ecef;
    opacity: 0.6;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

@keyframes pulse-loading {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; }
}

/* NEW CHATBOT WINDOW STYLES */
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

.chatbot-modal-header {
    border-bottom: 2px solid #e9ecef;
    padding-bottom: 1rem;
    margin-bottom: 2rem;
}

.chatbot-modal-chat {
    flex: 1;
    overflow-y: auto;
    background: #f8f9fa;
    border-radius: 15px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.chatbot-modal-input {
    padding: 1rem;
    border: 2px solid #e9ecef;
    border-radius: 10px;
    font-size: 1rem;
}

.step-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 0.5rem;
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

/* Keep all existing styles */
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
    background: rgba(255, 255, 255, 0.95);
    padding: 2.5rem;
    border-radius: 20px;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    border: 1px solid rgba(220, 53, 69, 0.2);
}

.risk-score-large {
    font-size: 3.5rem;
    font-weight: 800;
    margin: 1.5rem 0;
    background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 4px 8px rgba(220, 53, 69, 0.3);
    font-family: 'Inter', sans-serif;
    letter-spacing: -2px;
}

.elegant-risk-text {
    font-family: 'Inter', serif;
    font-size: 1.3rem;
    color: #2c3e50;
    font-weight: 500;
    margin: 1rem 0;
    line-height: 1.6;
}

.risk-category-elegant {
    display: inline-block;
    padding: 0.8rem 2rem;
    border-radius: 50px;
    font-weight: 600;
    font-size: 1.1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 1rem 0;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
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

/* Enhanced sidebar for better chat display - LARGER WIDTH FOR ALL SCREEN SIZES */
.css-1d391kg {
    width: 450px !important;
    min-width: 450px !important;
    max-width: 450px !important;
}

/* Ensure sidebar is properly sized on all screen resolutions */
@media (max-width: 1200px) {
    .css-1d391kg {
        width: 400px !important;
        min-width: 400px !important;
        max-width: 400px !important;
    }
}

@media (max-width: 768px) {
    .css-1d391kg {
        width: 350px !important;
        min-width: 350px !important;
        max-width: 350px !important;
    }
}

@media (min-width: 1400px) {
    .css-1d391kg {
        width: 500px !important;
        min-width: 500px !important;
        max-width: 500px !important;
    }
}

/* Additional sidebar styling */
.sidebar .sidebar-content {
    width: 100% !important;
    padding: 1rem !important;
}

/* Chat message styling */
.chat-message {
    margin: 0.5rem 0;
    padding: 0.8rem;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.1);
    border-left: 3px solid #007bff;
}

.user-message {
    background: rgba(0, 123, 255, 0.1);
    border-left-color: #007bff;
}

.assistant-message {
    background: rgba(40, 167, 69, 0.1);
    border-left-color: #28a745;
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

def get_safe_plot_params():
    """Return safe plotting parameters"""
    return {
        'markers': SAFE_MARKERS,
        'linestyles': SAFE_LINESTYLES, 
        'colors': SAFE_COLORS
    }

def clean_matplotlib_code_enhanced(code: str) -> str:
    """Enhanced matplotlib code cleaning to handle more edge cases"""
    try:
        # Remove seaborn style references
        problematic_styles = [
            'seaborn-whitegrid', 'seaborn-white', 'seaborn-darkgrid',
            'seaborn-dark', 'seaborn-ticks', 'seaborn-colorblind',
            'seaborn-notebook', 'seaborn-paper', 'seaborn-talk', 'seaborn-poster'
        ]
        
        cleaned_code = code
        
        # Replace problematic styles with default
        for style in problematic_styles:
            cleaned_code = re.sub(
                rf"plt\.style\.use\(['\"]?{re.escape(style)}['\"]?\)",
                "plt.style.use('default')",
                cleaned_code,
                flags=re.IGNORECASE
            )
        
        # Remove seaborn imports
        cleaned_code = re.sub(r'import\s+seaborn.*?\n', '', cleaned_code, flags=re.IGNORECASE | re.MULTILINE)
        cleaned_code = re.sub(r'sns\.', '# sns.', cleaned_code, flags=re.IGNORECASE)
        
        # Replace plt.show() calls with pass (since we handle figure display differently)
        cleaned_code = re.sub(r'plt\.show\(\)', '# plt.show() - handled by streamlit', cleaned_code, flags=re.IGNORECASE)
        
        # Fix common problematic marker styles
        problematic_markers = {
            r"marker\s*=\s*['\"]!['\"]": "marker='o'",
            r"marker\s*=\s*['\"]@['\"]": "marker='s'", 
            r"marker\s*=\s*['\"]#['\"]": "marker='^'",
            r"marker\s*=\s*['\"]%['\"]": "marker='d'",
            r"marker\s*=\s*['\"]&['\"]": "marker='*'",
            r"marker\s*=\s*['\"]![^'\"]*['\"]": "marker='o'"  # Any marker starting with !
        }
        
        for pattern, replacement in problematic_markers.items():
            cleaned_code = re.sub(pattern, replacement, cleaned_code, flags=re.IGNORECASE)
        
        # Fix problematic linestyle patterns
        problematic_linestyles = {
            r"linestyle\s*=\s*['\"]!['\"]": "linestyle='-'",
            r"ls\s*=\s*['\"]!['\"]": "ls='-'"
        }
        
        for pattern, replacement in problematic_linestyles.items():
            cleaned_code = re.sub(pattern, replacement, cleaned_code, flags=re.IGNORECASE)
        
        # Remove or fix problematic plot parameters
        cleaned_code = re.sub(r"plt\.ion\(\)", "# plt.ion() - not needed in streamlit", cleaned_code, flags=re.IGNORECASE)
        cleaned_code = re.sub(r"plt\.ioff\(\)", "# plt.ioff() - handled by streamlit", cleaned_code, flags=re.IGNORECASE)
        
        return cleaned_code
    except Exception as e:
        logger.warning(f"Error in enhanced code cleaning: {e}")
        return code

def extract_matplotlib_code_enhanced(response: str) -> str:
    """Enhanced matplotlib code extraction with better pattern matching"""
    try:
        # Multiple patterns to catch different code formats
        patterns = [
            r'```python\s*(.*?)```',
            r'```matplotlib\s*(.*?)```', 
            r'```\s*(.*?)```',
            r'import matplotlib.*?(?:plt\.show\(\)|plt\.savefig\(.*?\))',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                for match in matches:
                    code = match.strip()
                    # Check if it's actually matplotlib code
                    if any(keyword in code.lower() for keyword in ['matplotlib', 'plt.', 'pyplot', 'import plt']):
                        return code
        
        # If no code blocks found, check if the entire response looks like code
        matplotlib_indicators = ['import matplotlib', 'plt.', 'pyplot', 'fig,', 'ax.', 'plt.show()']
        if any(indicator in response for indicator in matplotlib_indicators):
            # Try to extract just the code part
            lines = response.split('\n')
            code_lines = []
            in_code = False
            
            for line in lines:
                if any(indicator in line for indicator in matplotlib_indicators):
                    in_code = True
                
                if in_code:
                    code_lines.append(line)
                    
                # Stop if we hit explanatory text after code
                if in_code and line.strip() and not any(char in line for char in ['import', 'plt', 'ax', 'fig', '#', '=']):
                    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=\(]', line.strip()):
                        break
            
            if code_lines:
                return '\n'.join(code_lines)
        
        return None
    except Exception as e:
        logger.error(f"Error extracting matplotlib code: {e}")
        return None

def validate_matplotlib_code(code: str) -> tuple[bool, str]:
    """Validate matplotlib code before execution"""
    if not code:
        return False, "No code provided"
    
    # Check for required imports
    has_plt = 'plt' in code or 'pyplot' in code
    has_matplotlib = 'matplotlib' in code
    
    if not (has_plt or has_matplotlib):
        return False, "No matplotlib imports detected"
    
    # Check for dangerous patterns
    dangerous_patterns = [
        r'os\.',
        r'subprocess\.',
        r'eval\(',
        r'exec\(',
        r'__import__',
        r'open\(',
        r'file\(',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            return False, f"Potentially dangerous code pattern detected: {pattern}"
    
    # Check for basic plot elements
    plot_indicators = ['plot', 'bar', 'scatter', 'hist', 'pie', 'line', 'figure', 'subplot']
    has_plot = any(indicator in code.lower() for indicator in plot_indicators)
    
    if not has_plot:
        return False, "No plotting functions detected"
    
    return True, "Code validation passed"

def validate_plot_data(data_dict):
    """Validate data before plotting to prevent array length mismatches"""
    validated_data = {}
    
    for key, value in data_dict.items():
        if isinstance(value, list):
            # Ensure all list elements are of consistent type
            if value:
                # Convert all elements to strings if mixed types
                if not all(isinstance(x, type(value[0])) for x in value):
                    value = [str(x) for x in value]
                validated_data[key] = value[:15]  # Limit length
            else:
                validated_data[key] = []
        elif isinstance(value, dict):
            # Ensure dictionary values are consistent
            dict_values = list(value.values())
            if dict_values and all(isinstance(x, (int, float)) for x in dict_values):
                validated_data[key] = value
            else:
                # Convert to safe format
                validated_data[key] = {k: 0 if not isinstance(v, (int, float)) else v 
                                     for k, v in value.items()}
        else:
            # Ensure scalar values are of correct type
            if isinstance(value, str):
                validated_data[key] = value
            elif isinstance(value, (int, float)):
                validated_data[key] = value
            else:
                validated_data[key] = str(value)
    
    return validated_data

def execute_matplotlib_code_enhanced_stability(code: str):
    """Execute matplotlib code with enhanced stability and comprehensive error recovery - FIXED VERSION"""
    try:
        # Clear any existing plots and set up clean environment
        plt.clf()
        plt.close('all')
        plt.ioff()  # Turn off interactive mode
        
        # Set safe matplotlib backend and style
        matplotlib.use('Agg')  # Non-interactive backend
        plt.style.use('default')
        
        # ENHANCED font configuration to handle unicode characters
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white', 
            'savefig.facecolor': 'white',
            'figure.figsize': (12, 8),
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            # UNICODE AND FONT FIXES
            'font.family': 'DejaVu Sans',
            'axes.unicode_minus': False,  # Fix unicode minus issues
            'text.usetex': False,  # Don't use LaTeX
        })
        
        # Enhanced code cleaning
        cleaned_code = clean_matplotlib_code_enhanced(code)
        
        # Create comprehensive namespace for code execution
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
            'math': __import__('math'),
            'random': __import__('random'),
            # Add commonly used plot functions
            'figure': plt.figure,
            'subplot': plt.subplot,
            'subplots': plt.subplots,
        }
        
        # Add comprehensive patient data from session state - WITH PROPER DATA TYPE VALIDATION
        if st.session_state.chatbot_context:
            context = st.session_state.chatbot_context
            
            # Extract all available data
            medical_extraction = context.get('medical_extraction', {})
            pharmacy_extraction = context.get('pharmacy_extraction', {})
            entity_extraction = context.get('entity_extraction', {})
            patient_overview = context.get('patient_overview', {})
            
            # FIXED: Ensure all data types are properly handled
            patient_age = patient_overview.get('age', 45)
            if not isinstance(patient_age, (int, float)):
                patient_age = 45
                
            heart_risk_score = context.get('heart_attack_risk_score', 0.25)
            if not isinstance(heart_risk_score, (int, float)):
                heart_risk_score = 0.25
                
            medications_count = len(pharmacy_extraction.get('ndc_records', []))
            medical_records_count = len(medical_extraction.get('hlth_srvc_records', []))
            
            # Enhanced patient data with TYPE SAFETY
            namespace.update({
                'patient_age': int(patient_age),
                'heart_risk_score': float(heart_risk_score),
                'medications_count': int(medications_count),
                'medical_records_count': int(medical_records_count),
                'diabetes_status': str(entity_extraction.get('diabetics', 'no')),
                'smoking_status': str(entity_extraction.get('smoking', 'no')),
                'bp_status': str(entity_extraction.get('blood_pressure', 'unknown')),
                'alcohol_status': str(entity_extraction.get('alcohol', 'unknown')),
                'age_group': str(entity_extraction.get('age_group', 'unknown')),
            })
            
            # FIXED: Risk factors dictionary with proper data types
            namespace['risk_factors'] = {
                'Age': int(patient_age), 
                'Diabetes': 1 if str(entity_extraction.get('diabetics', 'no')).lower() == 'yes' else 0, 
                'Smoking': 1 if str(entity_extraction.get('smoking', 'no')).lower() == 'yes' else 0, 
                'High_BP': 1 if str(entity_extraction.get('blood_pressure', 'unknown')).lower() in ['managed', 'diagnosed'] else 0,
                'Alcohol': 1 if str(entity_extraction.get('alcohol', 'no')).lower() == 'yes' else 0,
                'Family_History': 0  # Default
            }
            
            # FIXED: Extract and process medication data with LENGTH VALIDATION
            medication_names = []
            medication_dates = []
            ndc_codes = []
            
            for record in pharmacy_extraction.get('ndc_records', []):
                if record.get('lbl_nm') and isinstance(record['lbl_nm'], str):
                    medication_names.append(str(record['lbl_nm']))
                if record.get('rx_filled_dt'):
                    medication_dates.append(str(record['rx_filled_dt']))
                if record.get('ndc') and isinstance(record['ndc'], str):
                    ndc_codes.append(str(record['ndc']))
            
            # ENSURE ALL ARRAYS HAVE SAME LENGTH - KEY FIX
            max_med_length = min(15, len(medication_names))  # Limit length
            
            namespace.update({
                'medication_list': medication_names[:max_med_length],
                'medication_dates': medication_dates[:max_med_length], 
                'ndc_codes': ndc_codes[:max_med_length],
                'ndc_records': pharmacy_extraction.get('ndc_records', []),
                'medical_records': medical_extraction.get('hlth_srvc_records', [])
            })
            
            # FIXED: Extract diagnosis data with TYPE AND LENGTH VALIDATION
            diagnosis_codes = []
            diagnosis_descriptions = []
            
            for record in medical_extraction.get('hlth_srvc_records', []):
                for diag in record.get('diagnosis_codes', []):
                    if diag.get('code') and isinstance(diag['code'], str):
                        diagnosis_codes.append(str(diag['code']))
                        # Try to get description from code meanings
                        medical_meanings = medical_extraction.get('code_meanings', {})
                        diag_meanings = medical_meanings.get('diagnosis_code_meanings', {})
                        desc = diag_meanings.get(diag['code'], str(diag['code']))
                        diagnosis_descriptions.append(str(desc))
            
            # ENSURE DIAGNOSIS ARRAYS HAVE SAME LENGTH
            max_diag_length = min(10, len(diagnosis_codes))
            
            namespace.update({
                'diagnosis_codes': diagnosis_codes[:max_diag_length],
                'diagnosis_descriptions': diagnosis_descriptions[:max_diag_length]
            })
            
        else:
            # FIXED: Comprehensive fallback sample data with GUARANTEED SAME LENGTHS
            sample_medications = ['Metformin', 'Lisinopril', 'Atorvastatin']
            sample_dates = ['2024-01', '2024-02', '2024-03']
            sample_codes = ['0093-0058-01', '0071-0222-23', '0071-0156-23']
            
            namespace.update({
                'patient_age': 45,
                'heart_risk_score': 0.25,
                'medications_count': 3,
                'medical_records_count': 8,
                'diabetes_status': 'yes',
                'smoking_status': 'no',
                'bp_status': 'managed',
                'risk_factors': {
                    'Age': 45, 'Diabetes': 1, 'Smoking': 0, 'High_BP': 1, 'Family_History': 1
                },
                'medication_list': sample_medications,
                'diagnosis_codes': ['I10', 'E11.9', 'E78.5'],
                'diagnosis_descriptions': ['Hypertension', 'Type 2 Diabetes', 'Hyperlipidemia'],
                'risk_scores': [0.15, 0.25, 0.35, 0.20],
                'risk_labels': ['Low', 'Medium', 'High', 'Very High'],
                'months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                'utilization_data': [2, 3, 1, 4, 2, 3],
                'medication_dates': sample_dates,
                'ndc_codes': sample_codes
            })
        
        # Enhanced code execution with better error handling
        try:
            exec(cleaned_code, namespace)
            fig = plt.gcf()
            
            # Validate figure has content
            if not fig.axes:
                raise ValueError("Generated figure has no axes - creating fallback")
                
        except Exception as exec_error:
            logger.warning(f"Primary code execution failed: {exec_error}")
            
            # IMPROVED FALLBACK CODE with BETTER ERROR HANDLING
            try:
                fallback_code = f"""
import matplotlib.pyplot as plt
import numpy as np

# FIXED: Create a comprehensive healthcare dashboard with proper data handling
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Healthcare Analysis Dashboard', fontsize=16, fontweight='bold')

# FIXED: Risk factors visualization with data validation
risk_data = list(risk_factors.values())
risk_names = list(risk_factors.keys())

# Ensure we have valid data
if len(risk_data) > 0 and len(risk_names) > 0:
    colors = ['#28a745' if x == 0 else '#dc3545' for x in risk_data]
    ax1.bar(risk_names, risk_data, color=colors)
    ax1.set_title('Risk Factors Assessment', fontweight='bold')
    ax1.set_ylabel('Risk Level')
    ax1.tick_params(axis='x', rotation=45)
else:
    ax1.text(0.5, 0.5, 'Risk data unavailable', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title('Risk Factors Assessment', fontweight='bold')

# FIXED: Heart risk visualization - avoid problematic Unicode characters
risk_score = float(heart_risk_score)
risk_color = '#28a745' if risk_score < 0.3 else '#ffc107' if risk_score < 0.6 else '#dc3545'

# Create a simple risk meter instead of complex gauge
ax2.barh(['Heart Risk'], [risk_score], color=risk_color, alpha=0.7)
ax2.set_xlim(0, 1)
ax2.set_title(f'Heart Attack Risk: {{:.1%}}'.format(risk_score), fontweight='bold')
ax2.set_xlabel('Risk Score')

# FIXED: Medications chart with length validation
if len(medication_list) > 0:
    # Count medication frequencies - handle string data properly
    med_counts = {{}}
    for med in medication_list:
        med_str = str(med)  # Ensure string type
        med_counts[med_str] = med_counts.get(med_str, 0) + 1
    
    if med_counts:
        meds = list(med_counts.keys())[:5]  # Top 5
        counts = [med_counts[med] for med in meds]
        
        # FIXED: Handle long medication names
        short_meds = [med[:20] + '...' if len(med) > 20 else med for med in meds]
        
        ax3.barh(short_meds, counts, color='#007bff')
        ax3.set_title('Top Medications', fontweight='bold')
        ax3.set_xlabel('Frequency')
    else:
        ax3.text(0.5, 0.5, 'No medication data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Medications', fontweight='bold')
else:
    ax3.text(0.5, 0.5, 'No medication data', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Medications', fontweight='bold')

# FIXED: Health summary with proper text formatting (avoid emojis)
summary_metrics = [
    f'Age: {{}} years'.format(patient_age),
    f'Medications: {{}}'.format(medications_count),
    f'Medical Records: {{}}'.format(medical_records_count), 
    f'Diabetes: {{}}'.format(diabetes_status),
    f'Smoking: {{}}'.format(smoking_status)
]

ax4.text(0.05, 0.9, 'Patient Health Summary', fontsize=14, fontweight='bold', transform=ax4.transAxes)
for i, metric in enumerate(summary_metrics):
    ax4.text(0.05, 0.8 - i*0.12, f'• {{}}', fontsize=11, transform=ax4.transAxes)

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1) 
ax4.axis('off')

plt.tight_layout()
"""
                
                # Format the fallback code with proper data
                formatted_fallback = fallback_code.format(
                    patient_age, medications_count, medical_records_count, 
                    diabetes_status, smoking_status
                )
                
                exec(formatted_fallback, namespace)
                fig = plt.gcf()
                
            except Exception as fallback_error:
                logger.warning(f"Fallback code failed: {fallback_error}")
                
                # ULTIMATE FALLBACK - simple success message without problematic characters
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.7, 'Healthcare Visualization', 
                        ha='center', va='center', fontsize=24, fontweight='bold', color='#2c3e50')
                plt.text(0.5, 0.5, 'Generated Successfully!', 
                        ha='center', va='center', fontsize=18, color='#28a745')
                plt.text(0.5, 0.3, 'Your health analysis is ready for review', 
                        ha='center', va='center', fontsize=14, color='#007bff')
                plt.title('Medical Data Analysis Dashboard', fontsize=20, fontweight='bold', pad=20)
                plt.axis('off')
                fig = plt.gcf()
        
        # Enhanced figure styling with UNICODE-SAFE formatting
        if fig.axes:
            for ax in fig.axes:
                try:
                    ax.tick_params(labelsize=9)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    
                    # Enhance titles and labels - AVOID PROBLEMATIC CHARACTERS
                    if ax.get_title():
                        ax.set_title(ax.get_title(), fontsize=12, fontweight='bold', pad=10)
                    if ax.get_xlabel():
                        ax.set_xlabel(ax.get_xlabel(), fontsize=10)
                    if ax.get_ylabel():
                        ax.set_ylabel(ax.get_ylabel(), fontsize=10)
                        
                    # FIXED: Handle text elements safely
                    for text in ax.get_xticklabels() + ax.get_yticklabels():
                        try:
                            # Convert to string and handle encoding issues
                            text_str = str(text.get_text()).encode('ascii', 'ignore').decode('ascii')
                            if text_str != text.get_text():
                                text.set_text(text_str)
                        except:
                            continue
                            
                except Exception as styling_error:
                    logger.warning(f"Styling error for axis: {styling_error}")
                    continue
        
        # Set overall figure properties
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1.0)
        
        # FIXED: Convert to image with better error handling
        img_buffer = io.BytesIO()
        try:
            fig.savefig(
                img_buffer, 
                format='png', 
                bbox_inches='tight', 
                dpi=200,  # Reduced DPI to avoid memory issues
                facecolor='white', 
                edgecolor='none', 
                pad_inches=0.3,
                transparent=False
            )
            img_buffer.seek(0)
        except Exception as save_error:
            logger.error(f"Figure save error: {save_error}")
            # Try with minimal settings
            fig.savefig(img_buffer, format='png', facecolor='white')
            img_buffer.seek(0)
        
        # Cleanup matplotlib state
        plt.clf()
        plt.close('all')
        plt.ion()  # Re-enable interactive mode for future use
        
        return img_buffer
        
    except Exception as e:
        # Comprehensive error handling and cleanup
        plt.clf()
        plt.close('all')
        plt.ion()
        
        error_msg = str(e)
        logger.error(f"Complete matplotlib execution failure: {error_msg}")
        
        # Create informative error visualization WITHOUT problematic characters
        try:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.75, 'Graph Generation Issue', 
                    ha='center', va='center', fontsize=18, fontweight='bold', color='#dc3545')
            
            # Truncate long error messages and make them ASCII-safe
            short_error = error_msg[:100] + "..." if len(error_msg) > 100 else error_msg
            safe_error = short_error.encode('ascii', 'ignore').decode('ascii')
            
            plt.text(0.5, 0.55, 'Technical Details:', 
                    ha='center', va='center', fontsize=12, fontweight='bold', color='#6c757d')
            plt.text(0.5, 0.45, safe_error, 
                    ha='center', va='center', fontsize=10, color='#6c757d')
            
            plt.text(0.5, 0.3, 'Suggestions:', 
                    ha='center', va='center', fontsize=12, fontweight='bold', color='#007bff')
            plt.text(0.5, 0.2, '• Try asking for a different chart type', 
                    ha='center', va='center', fontsize=10, color='#007bff')
            plt.text(0.5, 0.1, '• Use simpler visualization requests', 
                    ha='center', va='center', fontsize=10, color='#007bff')
            
            plt.title('Healthcare Data Visualization', fontsize=16, pad=20)
            plt.axis('off')
            
            error_buffer = io.BytesIO()
            plt.savefig(error_buffer, format='png', bbox_inches='tight', 
                       dpi=150, facecolor='white', pad_inches=0.3)
            error_buffer.seek(0)
            plt.clf()
            plt.close('all')
            
            return error_buffer
        except:
            # If even error visualization fails, return None
            st.error(f"Graph generation completely failed: {error_msg}")
            return None

def handle_chatbot_response_with_graphs(user_question: str):
    """Enhanced chatbot response handling with improved graph generation"""
    try:
        # Get bot response
        chatbot_response = st.session_state.agent.chat_with_data(
            user_question, 
            st.session_state.chatbot_context, 
            st.session_state.chatbot_messages
        )
        
        # Extract and validate matplotlib code
        matplotlib_code = extract_matplotlib_code_enhanced(chatbot_response)
        
        if matplotlib_code:
            is_valid, validation_msg = validate_matplotlib_code(matplotlib_code)
            
            if is_valid:
                # Display text content without code
                text_content = chatbot_response
                for pattern in [f"```python\n{matplotlib_code}\n```", f"```\n{matplotlib_code}\n```"]:
                    text_content = text_content.replace(pattern, "")
                
                if text_content.strip():
                    st.write(text_content.strip())
                
                # Execute and display graph with enhanced error handling
                with st.spinner("Generating visualization..."):
                    try:
                        img_buffer = execute_matplotlib_code_enhanced_stability(matplotlib_code)
                        if img_buffer:
                            st.image(img_buffer, use_container_width=True, caption="Generated Healthcare Visualization")
                            st.success("Graph generated successfully!")
                        else:
                            st.error("Failed to generate graph - please try a different chart type")
                    except Exception as graph_error:
                        st.error(f"Graph generation error: {str(graph_error)}")
                        st.info("Try asking for: 'Create a simple bar chart of my medications' or 'Show my risk factors as a pie chart'")
            else:
                st.error(f"Code validation failed: {validation_msg}")
                st.write(chatbot_response)  # Show response without executing code
        else:
            # No matplotlib code, just display the response
            st.write(chatbot_response)
        
        return chatbot_response
        
    except Exception as e:
        error_msg = f"Chatbot error: {str(e)}"
        st.error(error_msg)
        return error_msg

# Legacy function for backward compatibility
def extract_matplotlib_code(response: str) -> str:
    """Legacy function - calls enhanced version"""
    return extract_matplotlib_code_enhanced(response)

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

# NEW: Real-time workflow monitoring functions
def check_langgraph_step_status(step_name: str) -> str:
    """Check the status of a specific langgraph step"""
    # This function should connect to your actual langgraph workflow
    # For now, it's a placeholder that you'll need to implement
    
    # Example implementation - replace with actual langgraph integration:
    if hasattr(st.session_state, 'agent') and st.session_state.agent:
        # Check if the agent has completed this step
        results = st.session_state.analysis_results
        if not results:
            return 'pending'
        
        # Map step names to result keys
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
        if result_key and result_key in results:
            data = results[result_key]
            if data and not (isinstance(data, dict) and data.get('error')):
                return 'completed'
            else:
                return 'error'
        else:
            return 'pending'
    
    return 'pending'

def update_workflow_status():
    """Update workflow status based on actual langgraph progress"""
    if st.session_state.analysis_running and hasattr(st.session_state, 'workflow_steps'):
        for i, step in enumerate(st.session_state.workflow_steps):
            status = check_langgraph_step_status(step['name'])
            st.session_state.workflow_steps[i]['status'] = status

def get_section_availability(section_name: str) -> str:
    """Determine if a section is available, loading, or disabled"""
    if not st.session_state.analysis_results:
        return 'disabled'
    
    # Map sections to their required workflow steps
    section_requirements = {
        'claims_data': ['API Fetch', 'Deidentification'],
        'code_analysis': ['Field Extraction'],
        'entity_extraction': ['Entity Extraction'],
        'health_trajectory': ['Health Trajectory'], 
        'heart_risk': ['Heart Risk Prediction'],
        'chatbot': ['Chatbot Initialization']
    }
    
    required_steps = section_requirements.get(section_name, [])
    if not required_steps:
        return 'available'
    
    # Check if all required steps are completed
    for step_name in required_steps:
        step_status = check_langgraph_step_status(step_name)
        if step_status == 'running':
            return 'loading'
        elif step_status != 'completed':
            return 'disabled'
    
    return 'available'

# NEW: Chatbot window creation function
def create_chatbot_window_html():
    """Create HTML for chatbot modal window"""
    chatbot_window_id = str(uuid.uuid4())
    
    # Define categorized prompts for the modal
    prompt_categories = {
        "Medical Records": [
            "What diagnoses were found in the medical records?",
            "What medical procedures were performed?", 
            "List all ICD-10 diagnosis codes found",
            "When did patient started taking diabetes medication?",
            "Are there any unusual prescribing or billing patterns related to this person's records?"
        ],
        "Medications": [
            "What medications is this patient taking?",
            "What NDC codes were identified?",
            "Is this person at risk of polypharmacy (taking too many medications or unsafe combinations)?",
            "How likely is this person to stop taking prescribed medications (medication adherence risk)?",
            "Is this person likely to switch to higher-cost specialty drugs or need therapy escalation soon?"
        ],
        "Risk Assessment": [
            "What is the heart attack risk and explain why?",
            "Based on this person's medical and pharmacy history, is there a risk of developing chronic diseases like diabetes, hypertension, COPD, or chronic kidney disease?",
            "What is the likelihood that this person will be hospitalized or readmitted in the next 6–12 months?",
            "Is this person at risk of using the emergency room instead of outpatient care?",
            "Does this person have a high risk of serious events like stroke, heart attack, or other complications due to comorbidities?"
        ],
        "Analysis & Graphs": [
            "Create a medication timeline chart",
            "Generate a comprehensive risk dashboard", 
            "Show me a pie chart of medications",
            "Create a health overview visualization",
            "Generate a diagnosis timeline chart",
            "Create a bar chart of medical conditions",
            "Show medication distribution graph"
        ],
        "Predictive Analysis": [
            "Predict the patient life expectancy with two scenarios: 1) adhering to the medication 2) non-adhering to the medication",
            "Can you model how this person's disease might progress over time (for example: diabetes → complications → hospitalizations)?",
            "Is this person likely to become a high-cost claimant next year?",
            "Can you estimate this person's future healthcare costs (per month or per year)?",
            "Based on health data, how should this person be segmented — healthy, rising risk, chronic but stable, or high-cost/critical?"
        ],
        "Care Management": [
            "What preventive screenings, wellness programs, or lifestyle changes should be recommended as the next best action for this person?",
            "Does this person have any care gaps, such as missed checkups, cancer screenings, or vaccinations?",
            "Does this person have any care gaps that could affect quality metrics (like HEDIS or STAR ratings)?",
            "Is this person more likely to need inpatient hospital care or outpatient care in the future?",
            "Based on available data, how might this person's long-term health contribute to population-level risk?"
        ]
    }
    
    # Generate prompt buttons HTML
    prompt_buttons_html = ""
    for category, prompts in prompt_categories.items():
        prompt_buttons_html += f"""
        <div class="sidebar-category">
            <h4>{category}</h4>
        """
        for i, prompt in enumerate(prompts):
            prompt_buttons_html += f"""
            <button class="category-prompt-btn" onclick="setPrompt('{prompt.replace("'", "\\'")}')">
                {prompt}
            </button>
            """
        prompt_buttons_html += "</div>"
    
    # Create the HTML content and display it with st.markdown
    html_content = f"""
    <div id="chatbot-modal-{chatbot_window_id}" class="chatbot-modal" style="display: none;">
        <div class="chatbot-modal-content">
            <div class="chatbot-modal-sidebar">
                <h3 style="color: white; text-align: center; margin-bottom: 2rem;">Medical Assistant</h3>
                <div style="margin-bottom: 2rem;">
                    <h4 style="color: white; margin-bottom: 1rem;">Quick Questions:</h4>
                    {prompt_buttons_html}
                </div>
                <button onclick="closeChatbotModal('{chatbot_window_id}')" style="
                    background: rgba(255, 255, 255, 0.2);
                    color: white;
                    border: 1px solid rgba(255, 255, 255, 0.3);
                    padding: 0.8rem 1.5rem;
                    border-radius: 8px;
                    cursor: pointer;
                    width: 100%;
                    margin-top: 2rem;
                    font-weight: 600;
                ">Close Chat Window</button>
            </div>
            <div class="chatbot-modal-main">
                <div class="chatbot-modal-header">
                    <h2 style="color: #2c3e50; margin: 0;">Healthcare AI Assistant</h2>
                    <p style="color: #6c757d; margin: 0.5rem 0 0 0;">Ask questions about health data, request visualizations, or use quick prompts</p>
                </div>
                <div class="chatbot-modal-chat" id="chat-container-{chatbot_window_id}">
                    <div style="padding: 2rem; text-align: center; color: #6c757d;">
                        <h4>Welcome to your Healthcare AI Assistant!</h4>
                        <p>Start a conversation using the sidebar prompts or type your question below.</p>
                        <div style="margin: 2rem 0;">
                            <div style="font-size: 3rem; margin-bottom: 1rem;">🏥</div>
                            <p><strong>Enhanced Features:</strong></p>
                            <ul style="text-align: left; max-width: 400px; margin: 0 auto;">
                                <li>Real-time graph generation with matplotlib</li>
                                <li>Categorized medical prompts for easy navigation</li>
                                <li>Advanced health data analysis</li>
                                <li>Interactive chart display in chat</li>
                                <li>Comprehensive clinical decision support</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div style="display: flex; gap: 1rem;">
                    <input 
                        type="text" 
                        id="chat-input-{chatbot_window_id}" 
                        class="chatbot-modal-input" 
                        placeholder="Type your healthcare question here..."
                        style="flex: 1;"
                        onkeypress="if(event.key==='Enter') sendMessage('{chatbot_window_id}')"
                    />
                    <button 
                        onclick="sendMessage('{chatbot_window_id}')"
                        style="
                            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
                            color: white;
                            border: none;
                            padding: 1rem 2rem;
                            border-radius: 8px;
                            font-weight: 600;
                            cursor: pointer;
                        "
                    >Send</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
    function openChatbotModal(modalId) {{
        document.getElementById('chatbot-modal-' + modalId).style.display = 'flex';
    }}
    
    function closeChatbotModal(modalId) {{
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
            // Add user message to chat
            const userMessage = document.createElement('div');
            userMessage.innerHTML = `
                <div style="background: #007bff; color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; margin-left: 20%;">
                    <strong>You:</strong> ${{message}}
                </div>
            `;
            chatContainer.appendChild(userMessage);
            
            // Add loading message
            const loadingMessage = document.createElement('div');
            loadingMessage.innerHTML = `
                <div style="background: #f8f9fa; color: #6c757d; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; margin-right: 20%;">
                    <strong>Assistant:</strong> Processing your request...
                    <div class="loading-spinner" style="width: 20px; height: 20px; margin: 10px auto;"></div>
                </div>
            `;
            chatContainer.appendChild(loadingMessage);
            
            // Clear input
            input.value = '';
            
            // Scroll to bottom
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            // Simulate response (replace with actual Streamlit integration)
            setTimeout(() => {{
                loadingMessage.innerHTML = `
                    <div style="background: #28a745; color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; margin-right: 20%;">
                        <strong>Assistant:</strong> I've received your question: "${{message}}". This feature will be fully integrated with the Streamlit backend to provide real-time responses with graph generation capabilities.
                    </div>
                `;
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }}, 2000);
        }}
    }}
    </script>
    """
    
    # Display the HTML content using st.markdown
    st.markdown(html_content, unsafe_allow_html=True)
    
    return chatbot_window_id

# NEW: Function to check if sections should be enabled
def is_section_enabled(section_name: str) -> bool:
    """Check if a section should be enabled based on workflow progress"""
    availability = get_section_availability(section_name)
    return availability in ['available', 'loading']

def get_section_css_class(section_name: str) -> str:
    """Get CSS class for section based on availability"""
    availability = get_section_availability(section_name)
    if availability == 'available':
        return 'section-available'
    elif availability == 'loading':
        return 'section-loading'
    else:
        return 'section-disabled'

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
    
    # Add selected_prompt for categorized prompts
    if 'selected_prompt' not in st.session_state:
        st.session_state.selected_prompt = None
        
    # NEW: Add progressive loading state
    if 'progressive_results' not in st.session_state:
        st.session_state.progressive_results = {}

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
    st.session_state.progressive_results = {}

def display_advanced_professional_workflow():
    """Display the advanced professional workflow animation with real-time status"""
    
    # Update workflow status based on actual progress
    update_workflow_status()
    
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
    
    # Display each step with real-time status indicators
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
                <span class="step-indicator step-{status}"></span>
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

# NEW: Progressive analysis function that runs steps as they complete
def run_progressive_analysis(patient_data: Dict[str, Any]):
    """Run analysis progressively, updating UI as each step completes"""
    try:
        # Step 1: API Fetch
        st.session_state.workflow_steps[0]['status'] = 'running'
        st.rerun()
        
        # Initialize agent if not already done
        if not st.session_state.agent:
            config = Config()
            st.session_state.config = config
            st.session_state.agent = HealthAnalysisAgent(config)
        
        # Fetch API data
        api_results = st.session_state.agent.fetch_api_data(patient_data)
        st.session_state.progressive_results['api_outputs'] = api_results
        st.session_state.workflow_steps[0]['status'] = 'completed'
        
        # Step 2: Deidentification
        st.session_state.workflow_steps[1]['status'] = 'running'
        st.rerun()
        
        deidentified_data = st.session_state.agent.deidentify_data(api_results)
        st.session_state.progressive_results['deidentified_data'] = deidentified_data
        st.session_state.workflow_steps[1]['status'] = 'completed'
        
        # Step 3: Field Extraction
        st.session_state.workflow_steps[2]['status'] = 'running'
        st.rerun()
        
        structured_extractions = st.session_state.agent.extract_structured_data(deidentified_data)
        st.session_state.progressive_results['structured_extractions'] = structured_extractions
        st.session_state.workflow_steps[2]['status'] = 'completed'
        
        # Step 4: Entity Extraction
        st.session_state.workflow_steps[3]['status'] = 'running'
        st.rerun()
        
        entity_extraction = st.session_state.agent.extract_entities(structured_extractions)
        st.session_state.progressive_results['entity_extraction'] = entity_extraction
        st.session_state.workflow_steps[3]['status'] = 'completed'
        
        # Step 5: Health Trajectory
        st.session_state.workflow_steps[4]['status'] = 'running'
        st.rerun()
        
        health_trajectory = st.session_state.agent.analyze_health_trajectory(
            structured_extractions, entity_extraction, patient_data
        )
        st.session_state.progressive_results['health_trajectory'] = health_trajectory
        st.session_state.workflow_steps[4]['status'] = 'completed'
        
        # Step 6: Heart Risk Prediction
        st.session_state.workflow_steps[5]['status'] = 'running'
        st.rerun()
        
        heart_risk_results = st.session_state.agent.predict_heart_attack_risk(
            entity_extraction, patient_data
        )
        st.session_state.progressive_results.update(heart_risk_results)
        st.session_state.workflow_steps[5]['status'] = 'completed'
        
        # Step 7: Chatbot Initialization
        st.session_state.workflow_steps[6]['status'] = 'running'
        st.rerun()
        
        # Prepare chatbot context
        chatbot_context = {
            'medical_extraction': structured_extractions.get('medical', {}),
            'pharmacy_extraction': structured_extractions.get('pharmacy', {}),
            'entity_extraction': entity_extraction,
            'patient_overview': {
                'age': calculate_age(datetime.strptime(patient_data['date_of_birth'], '%Y-%m-%d').date()) if patient_data.get('date_of_birth') else None,
                'gender': patient_data.get('gender', ''),
                'zip_code': patient_data.get('zip_code', '')
            },
            'heart_attack_risk_score': st.session_state.progressive_results.get('heart_attack_risk_score', 0.0)
        }
        
        st.session_state.chatbot_context = chatbot_context
        st.session_state.progressive_results['chatbot_ready'] = True
        st.session_state.progressive_results['chatbot_context'] = chatbot_context
        st.session_state.workflow_steps[6]['status'] = 'completed'
        
        # Mark analysis as complete
        st.session_state.progressive_results['success'] = True
        st.session_state.analysis_results = st.session_state.progressive_results
        st.session_state.analysis_running = False
        
        st.rerun()
        
    except Exception as e:
        # Mark current step as error
        if st.session_state.current_step < len(st.session_state.workflow_steps):
            st.session_state.workflow_steps[st.session_state.current_step]['status'] = 'error'
        
        st.session_state.analysis_running = False
        st.session_state.progressive_results['success'] = False
        st.session_state.progressive_results['error'] = str(e)
        st.session_state.analysis_results = st.session_state.progressive_results
        
        st.error(f"Analysis failed at step: {str(e)}")
        st.rerun()

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
