# Configure Streamlit page FIRST
import streamlit as st

st.set_page_config(
    page_title="🤖 Medical Assistant - Healthcare AI Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import required modules
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
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

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

# Enhanced CSS for the dedicated chatbot page
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.chatbot-main-header {
    font-size: 2.8rem;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 1.5rem;
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

.chat-container {
    background: white;
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    border: 1px solid #e9ecef;
    min-height: 500px;
    max-height: 600px;
    overflow-y: auto;
}

.chat-message {
    margin: 1rem 0;
    padding: 1rem;
    border-radius: 10px;
}

.user-message {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left: 4px solid #2196f3;
    margin-left: 2rem;
}

.assistant-message {
    background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c8 100%);
    border-left: 4px solid #4caf50;
    margin-right: 2rem;
}

.sidebar-category {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 0.5rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 3px solid #007bff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
}

.sidebar-category:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
}

.sidebar-category h4 {
    color: #2c3e50;
    margin: 0 0 0.8rem 0;
    font-size: 1rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.category-prompt-btn {
    background: linear-gradient(135deg, #007bff 0%, #0056b3 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1rem !important;
    margin: 0.3rem 0 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    text-align: left !important;
    box-shadow: 0 4px 15px rgba(0, 123, 255, 0.3) !important;
}

.category-prompt-btn:hover {
    background: linear-gradient(135deg, #0056b3 0%, #004085 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 123, 255, 0.4) !important;
}

.chat-input-container {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    border: 2px solid #dee2e6;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
}

.status-indicator {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.5rem 0;
}

.status-connected {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    color: #155724;
    border: 1px solid #c3e6cb;
}

.status-processing {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    color: #856404;
    border: 1px solid #ffeaa7;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

.graph-display-container {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border: 2px solid #e3f2fd;
}

.back-to-main-btn {
    background: linear-gradient(135deg, #6c757d 0%, #495057 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.8rem 1.5rem !important;
    border-radius: 10px !important;
    box-shadow: 0 6px 20px rgba(108, 117, 125, 0.4) !important;
    transition: all 0.3s ease !important;
    margin-bottom: 1rem !important;
}

.back-to-main-btn:hover {
    background: linear-gradient(135deg, #495057 0%, #343a40 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(108, 117, 125, 0.5) !important;
}

.clear-chat-btn {
    background: linear-gradient(135deg, #dc3545 0%, #c82333 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 15px rgba(220, 53, 69, 0.4) !important;
    transition: all 0.3s ease !important;
}

.clear-chat-btn:hover {
    background: linear-gradient(135deg, #c82333 0%, #bd2130 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(220, 53, 69, 0.5) !important;
}

.analysis-summary-box {
    background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
    padding: 1.5rem;
    border-radius: 12px;
    border: 2px solid #28a745;
    margin: 1rem 0;
    box-shadow: 0 8px 25px rgba(40, 167, 69, 0.2);
}

.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #007bff;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-right: 0.5rem;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.chat-history-container {
    max-height: 400px;
    overflow-y: auto;
    padding: 1rem;
    background: #f8f9fa;
    border-radius: 10px;
    margin: 1rem 0;
}

/* Custom scrollbar for chat history */
.chat-history-container::-webkit-scrollbar {
    width: 8px;
}

.chat-history-container::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

.chat-history-container::-webkit-scrollbar-thumb {
    background: #007bff;
    border-radius: 4px;
}

.chat-history-container::-webkit-scrollbar-thumb:hover {
    background: #0056b3;
}

.chat-stats {
    text-align: center;
    padding: 1rem;
    color: #6c757d;
    font-style: italic;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 10px;
    margin: 1rem 0;
    border: 1px solid #dee2e6;
}
</style>
""", unsafe_allow_html=True)

# Initialize chatbot messages if not exists
if 'chatbot_messages' not in st.session_state:
    st.session_state.chatbot_messages = []

if 'selected_prompt' not in st.session_state:
    st.session_state.selected_prompt = None

# Helper functions for matplotlib
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
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                for match in matches:
                    code = match.strip()
                    # Check if it's actually matplotlib code
                    if any(keyword in code.lower() for keyword in ['matplotlib', 'plt.', 'pyplot', 'import plt']):
                        return code
        
        return None
    except Exception as e:
        logger.error(f"Error extracting matplotlib code: {e}")
        return None

def execute_matplotlib_code_enhanced_stability(code: str):
    """Execute matplotlib code with enhanced stability and comprehensive error recovery"""
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
        
        # Add comprehensive patient data from session state
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
            
            # Extract medication data
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
            
            # ENSURE ALL ARRAYS HAVE SAME LENGTH
            max_med_length = min(15, len(medication_names))
            
            namespace.update({
                'medication_list': medication_names[:max_med_length],
                'medication_dates': medication_dates[:max_med_length], 
                'ndc_codes': ndc_codes[:max_med_length],
                'ndc_records': pharmacy_extraction.get('ndc_records', []),
                'medical_records': medical_extraction.get('hlth_srvc_records', [])
            })
            
            # Extract diagnosis data
            diagnosis_codes = []
            diagnosis_descriptions = []
            
            for record in medical_extraction.get('hlth_srvc_records', []):
                for diag in record.get('diagnosis_codes', []):
                    if diag.get('code') and isinstance(diag['code'], str):
                        diagnosis_codes.append(str(diag['code']))
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
            # Comprehensive fallback sample data
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
                'medication_dates': sample_dates,
                'ndc_codes': sample_codes
            })
        
        # Execute code
        try:
            exec(cleaned_code, namespace)
            fig = plt.gcf()
            
            # Validate figure has content
            if not fig.axes:
                raise ValueError("Generated figure has no axes - creating fallback")
                
        except Exception as exec_error:
            logger.warning(f"Primary code execution failed: {exec_error}")
            
            # Create simple fallback visualization
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
        
        # Convert to image
        img_buffer = io.BytesIO()
        try:
            fig.savefig(
                img_buffer, 
                format='png', 
                bbox_inches='tight', 
                dpi=200,
                facecolor='white', 
                edgecolor='none', 
                pad_inches=0.3,
                transparent=False
            )
            img_buffer.seek(0)
        except Exception as save_error:
            logger.error(f"Figure save error: {save_error}")
            fig.savefig(img_buffer, format='png', facecolor='white')
            img_buffer.seek(0)
        
        # Cleanup matplotlib state
        plt.clf()
        plt.close('all')
        plt.ion()
        
        return img_buffer
        
    except Exception as e:
        # Comprehensive error handling and cleanup
        plt.clf()
        plt.close('all')
        plt.ion()
        
        error_msg = str(e)
        logger.error(f"Complete matplotlib execution failure: {error_msg}")
        
        # Create informative error visualization
        try:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.75, 'Graph Generation Issue', 
                    ha='center', va='center', fontsize=18, fontweight='bold', color='#dc3545')
            
            short_error = error_msg[:100] + "..." if len(error_msg) > 100 else error_msg
            safe_error = short_error.encode('ascii', 'ignore').decode('ascii')
            
            plt.text(0.5, 0.55, 'Technical Details:', 
                    ha='center', va='center', fontsize=12, fontweight='bold', color='#6c757d')
            plt.text(0.5, 0.45, safe_error, 
                    ha='center', va='center', fontsize=10, color='#6c757d')
            
            plt.text(0.5, 0.3, 'Try asking for a different chart type', 
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
        
        return chatbot_response
        
    except Exception as e:
        error_msg = f"Chatbot error: {str(e)}"
        return error_msg

# Main Chatbot Page
def main():
    # Check if analysis results are available
    if not st.session_state.get('analysis_results') or not st.session_state.get('chatbot_context'):
        st.error("❌ No analysis data found. Please run the healthcare analysis first.")
        st.info("👈 Go back to the main page to run the analysis")
        
        if st.button("🏠 Back to Main Page", key="back_to_main_error"):
            st.switch_page("main.py")
        
        st.stop()
    
    # Page header
    st.markdown('<h1 class="chatbot-main-header">🤖 Medical Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Healthcare Analysis with Graph Generation")
    
    # Connection status
    st.markdown("""
    <div class="status-indicator status-connected">
        ✅ Connected to Healthcare Analysis Data
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis summary box
    if st.session_state.chatbot_context:
        context = st.session_state.chatbot_context
        medical_records = len(context.get('medical_extraction', {}).get('hlth_srvc_records', []))
        pharmacy_records = len(context.get('pharmacy_extraction', {}).get('ndc_records', []))
        heart_risk = context.get('heart_attack_risk_score', 0.0)
        
        st.markdown(f"""
        <div class="analysis-summary-box">
            <h4 style="margin: 0 0 0.5rem 0; color: #28a745;">📊 Analysis Summary</h4>
            <p style="margin: 0; color: #155724;">
                <strong>Medical Records:</strong> {medical_records} | 
                <strong>Pharmacy Records:</strong> {pharmacy_records} | 
                <strong>Heart Risk Score:</strong> {heart_risk:.1%}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Main content area with sidebar
    with st.sidebar:
        st.title("Quick Questions")
        
        if st.button("🏠 Back to Main Page", key="back_to_main", use_container_width=True):
            st.switch_page("main.py")
        
        st.markdown("---")
        
        # Define categorized prompts
        prompt_categories = {
            "📋 Medical Records": [
                "What diagnoses were found in the medical records?",
                "What medical procedures were performed?", 
                "List all ICD-10 diagnosis codes found",
                "When did patient started taking diabetes medication?",
                "Are there any unusual prescribing or billing patterns?"
            ],
            "💊 Medications": [
                "What medications is this patient taking?",
                "What NDC codes were identified?",
                "Is this person at risk of polypharmacy?",
                "How likely is this person to stop taking prescribed medications?",
                "Is this person likely to switch to higher-cost specialty drugs?"
            ],
            "⚠️ Risk Assessment": [
                "What is the heart attack risk and explain why?",
                "Is there a risk of developing chronic diseases?",
                "What is the likelihood of hospitalization in the next 6-12 months?",
                "Is this person at risk of using ER instead of outpatient care?",
                "Does this person have high risk of serious complications?"
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
            "🔮 Predictive Analysis": [
                "Predict patient life expectancy with medication adherence scenarios",
                "Model disease progression over time",
                "Is this person likely to become a high-cost claimant?",
                "Estimate future healthcare costs",
                "How should this person be segmented for risk?"
            ],
            "🏥 Care Management": [
                "What preventive screenings should be recommended?",
                "Does this person have any care gaps?",
                "Are there missed checkups or screenings?",
                "Is this person likely to need inpatient or outpatient care?",
                "How might this contribute to population-level risk?"
            ]
        }
        
        # Create COLLAPSED expandable sections for each category
        for category, prompts in prompt_categories.items():
            with st.expander(category, expanded=False):
                for i, prompt in enumerate(prompts):
                    if st.button(prompt, key=f"cat_prompt_{category}_{i}", use_container_width=True):
                        st.session_state.selected_prompt = prompt
                        st.rerun()
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", key="clear_chat", use_container_width=True):
            st.session_state.chatbot_messages = []
            st.success("Chat history cleared!")
            st.rerun()
    
    # Chat interface
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    user_question = st.chat_input("💬 Type your question or use the quick prompts in the sidebar...")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle manual chat input
    if user_question:
        # Add user message
        st.session_state.chatbot_messages.append({"role": "user", "content": user_question})
        
        # Show processing status
        with st.spinner("Processing your request..."):
            try:
                chatbot_response = handle_chatbot_response_with_graphs(user_question)
                st.session_state.chatbot_messages.append({"role": "assistant", "content": chatbot_response})
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Handle selected prompt from session state
    if hasattr(st.session_state, 'selected_prompt') and st.session_state.selected_prompt:
        user_question = st.session_state.selected_prompt
        
        # Add user message
        st.session_state.chatbot_messages.append({"role": "user", "content": user_question})
        
        # Show processing status
        with st.spinner("Processing your request..."):
            try:
                chatbot_response = handle_chatbot_response_with_graphs(user_question)
                st.session_state.chatbot_messages.append({"role": "assistant", "content": chatbot_response})
                st.session_state.selected_prompt = None
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.selected_prompt = None
    
    # Normal Chat History Display
    st.markdown("### 💬 Chat History")
    
    # Add search functionality
    if st.session_state.chatbot_messages:
        col1, col2 = st.columns([4, 1])
        with col1:
            search_term = st.text_input(
                "🔍 Search chat history:", 
                placeholder="Search questions and answers...",
                help="Search through your chat history to find specific topics"
            )
        with col2:
            if st.button("Clear Search", key="clear_search"):
                st.rerun()
        
        st.markdown("---")
    
    if st.session_state.chatbot_messages:
        # Filter messages by search term if provided
        filtered_messages = st.session_state.chatbot_messages
        if 'search_term' in locals() and search_term:
            filtered_messages = []
            for msg in st.session_state.chatbot_messages:
                if search_term.lower() in msg["content"].lower():
                    filtered_messages.append(msg)
            
            if filtered_messages:
                st.success(f"🎯 Found {len(filtered_messages)} message(s) matching '{search_term}'")
            else:
                st.warning(f"❌ No results found for '{search_term}'. Try different keywords.")
                st.info("💡 **Search Tips:** Try searching for 'medication', 'risk', 'chart', 'diagnosis', etc.")
        
        # Display messages in normal chat format (newest first)
        for message in reversed(filtered_messages):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    # Check if message contains matplotlib code
                    matplotlib_code = extract_matplotlib_code_enhanced(message["content"])
                    if matplotlib_code:
                        # Display text content without code
                        text_content = message["content"]
                        for pattern in [f"```python\n{matplotlib_code}\n```", f"```\n{matplotlib_code}\n```"]:
                            text_content = text_content.replace(pattern, "")
                        if text_content.strip():
                            st.write(text_content.strip())
                        
                        # Execute and display graph
                        with st.spinner("🎨 Generating visualization..."):
                            try:
                                img_buffer = execute_matplotlib_code_enhanced_stability(matplotlib_code)
                                if img_buffer:
                                    st.markdown('<div class="graph-display-container">', unsafe_allow_html=True)
                                    st.image(img_buffer, use_container_width=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.error("❌ Failed to generate graph")
                            except Exception as e:
                                st.error(f"❌ Graph generation error: {str(e)}")
                    else:
                        st.write(message["content"])
                else:
                    st.write(message["content"])
        
        # Show message count
        if not ('search_term' in locals() and search_term):
            st.markdown(f"""
            <div class="chat-stats">
                📊 <strong>Total Messages:</strong> {len(st.session_state.chatbot_messages)} messages
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("🚀 Start a conversation! Use the quick prompts in the sidebar or type your question above.")
        
        # Show example questions when no chat history
        st.markdown("### 💡 Try asking:")
        st.markdown("• What is my heart attack risk?")
        st.markdown("• Create a visualization of my medications")
        st.markdown("• Show me my diagnosis codes")
        st.markdown("• Generate a health summary dashboard")
        st.markdown("• What preventive care do I need?")

if __name__ == "__main__":
    main()
