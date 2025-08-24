# complete_health_agent.py - Complete Integrated Health Agent Application
# Run with: streamlit run complete_health_agent.py

import streamlit as st
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import os
import logging
from typing import Dict, Any, Optional, List, Callable
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
import webbrowser
from urllib.parse import urlencode
import threading
from dataclasses import dataclass, asdict
from datetime import date
import requests

# Configure Streamlit page FIRST
st.set_page_config(
    page_title="⚡ Complete Health Agent with Separate Chatbot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ENHANCED MATPLOTLIB CONFIGURATION
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONFIGURATION CLASS
@dataclass
class Config:
    fastapi_url: str = "http://localhost:8000"
    api_url: str = "https://example-api.com/complete"  # Replace with your API
    api_key: str = "demo-api-key-replace-with-real"  # Replace with your API key
    app_id: str = "health-agent"
    model: str = "demo-model"
    timeout: int = 30
    heart_attack_api_url: str = "http://localhost:8000"
    heart_attack_threshold: float = 0.5

    def to_dict(self):
        return asdict(self)

# ENHANCED CSS WITH COMPLETE STYLING
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

/* Chatbot Launch Button */
.chatbot-launch-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 1rem 2rem;
    border-radius: 50px;
    font-size: 1.2rem;
    font-weight: 600;
    cursor: pointer;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 1px;
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1000;
    animation: pulse-glow 2s infinite;
}

.chatbot-launch-btn:hover {
    transform: translateY(-3px) scale(1.05);
    box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

@keyframes pulse-glow {
    0%, 100% { 
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
    50% { 
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.8);
    }
}

/* Status Indicator */
.status-indicator {
    position: fixed;
    top: 80px;
    right: 20px;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 600;
    z-index: 999;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.status-ready {
    background: #28a745;
    color: white;
}

.status-processing {
    background: #ffc107;
    color: #856404;
    animation: pulse-status 2s infinite;
}

.status-waiting {
    background: #6c757d;
    color: white;
}

@keyframes pulse-status {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

/* Workflow Styling */
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

.workflow-step {
    background: rgba(255, 255, 255, 0.8);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    border-left: 4px solid #6c757d;
    transition: all 0.4s ease;
    backdrop-filter: blur(10px);
}

.workflow-step.completed {
    border-left-color: #28a745;
    background: rgba(40, 167, 69, 0.15);
    box-shadow: 0 10px 30px rgba(40, 167, 69, 0.2);
}

.workflow-step.running {
    border-left-color: #ffc107;
    background: rgba(255, 193, 7, 0.15);
    animation: pulse-step 2s infinite;
    box-shadow: 0 10px 30px rgba(255, 193, 7, 0.3);
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

/* Section availability indicators */
.section-expandable {
    position: relative;
    margin: 1rem 0;
    border: 2px solid #e9ecef;
    border-radius: 15px;
    background: white;
    transition: all 0.3s ease;
}

.section-expandable.available {
    border: 2px solid #28a745;
    box-shadow: 0 0 20px rgba(40, 167, 69, 0.3);
}

.section-expandable.available::before {
    content: "✅ Available";
    position: absolute;
    top: -10px;
    right: 20px;
    background: #28a745;
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 15px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* Chatbot Window Styling */
.chatbot-window-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
}

.chatbot-window-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Chat message styling */
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    max-width: 80%;
}

.user-message {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left: 4px solid #2196f3;
    margin-left: auto;
}

.assistant-message {
    background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c8 100%);
    border-left: 4px solid #4caf50;
}

/* Graph display styling */
.graph-display {
    background: white;
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    border: 2px solid #e3f2fd;
}

/* Metric cards */
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

/* Enhanced buttons */
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

/* Sidebar styling */
.css-1d391kg {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.sidebar .sidebar-content {
    width: 100% !important;
    padding: 1rem !important;
}

/* Loading animations */
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
</style>
""", unsafe_allow_html=True)

# HELPER FUNCTIONS

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

# GRAPH GENERATION FUNCTIONS

def extract_matplotlib_code(response: str) -> str:
    """Extract matplotlib code from response"""
    try:
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
                    if any(keyword in code.lower() for keyword in ['matplotlib', 'plt.', 'pyplot', 'import plt']):
                        return code
        return None
    except Exception as e:
        logger.error(f"Error extracting matplotlib code: {e}")
        return None

def clean_matplotlib_code_enhanced(code: str) -> str:
    """Enhanced matplotlib code cleaning"""
    try:
        cleaned_code = code
        
        # Remove problematic imports and calls
        problematic_patterns = {
            r'import\s+seaborn.*?\n': '',
            r'sns\.': '# sns.',
            r'plt\.show\(\)': '# plt.show() - handled by streamlit',
            r'plt\.ion\(\)': '# plt.ion() - not needed',
            r'plt\.ioff\(\)': '# plt.ioff() - handled',
        }
        
        for pattern, replacement in problematic_patterns.items():
            cleaned_code = re.sub(pattern, replacement, cleaned_code, flags=re.IGNORECASE | re.MULTILINE)
        
        return cleaned_code
    except Exception as e:
        logger.warning(f"Error in code cleaning: {e}")
        return code

def execute_matplotlib_code_enhanced_stability(code: str):
    """Execute matplotlib code with enhanced stability"""
    try:
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        plt.ioff()
        
        # Set safe matplotlib backend
        matplotlib.use('Agg')
        plt.style.use('default')
        
        # Enhanced font configuration
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white', 
            'savefig.facecolor': 'white',
            'figure.figsize': (12, 8),
            'font.size': 10,
            'font.family': 'DejaVu Sans',
            'axes.unicode_minus': False,
            'text.usetex': False,
        })
        
        # Clean the code
        cleaned_code = clean_matplotlib_code_enhanced(code)
        
        # Create namespace for execution
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
            'figure': plt.figure,
            'subplot': plt.subplot,
            'subplots': plt.subplots,
        }
        
        # Add patient data from session state if available
        if hasattr(st, 'session_state') and st.session_state.get('chatbot_context'):
            context = st.session_state.chatbot_context
            
            # Extract patient data safely
            patient_age = context.get('patient_overview', {}).get('age', 45)
            if not isinstance(patient_age, (int, float)):
                patient_age = 45
                
            heart_risk_score = context.get('heart_attack_risk_score', 0.25)
            if not isinstance(heart_risk_score, (int, float)):
                heart_risk_score = 0.25
            
            # Add to namespace
            namespace.update({
                'patient_age': int(patient_age),
                'heart_risk_score': float(heart_risk_score),
                'medications_count': len(context.get('pharmacy_extraction', {}).get('ndc_records', [])),
                'medical_records_count': len(context.get('medical_extraction', {}).get('hlth_srvc_records', [])),
                'diabetes_status': str(context.get('entity_extraction', {}).get('diabetics', 'no')),
                'smoking_status': str(context.get('entity_extraction', {}).get('smoking', 'no')),
                'bp_status': str(context.get('entity_extraction', {}).get('blood_pressure', 'unknown')),
            })
            
            # Risk factors
            entity_extraction = context.get('entity_extraction', {})
            namespace['risk_factors'] = {
                'Age': int(patient_age), 
                'Diabetes': 1 if str(entity_extraction.get('diabetics', 'no')).lower() == 'yes' else 0, 
                'Smoking': 1 if str(entity_extraction.get('smoking', 'no')).lower() == 'yes' else 0, 
                'High_BP': 1 if str(entity_extraction.get('blood_pressure', 'unknown')).lower() in ['managed', 'diagnosed'] else 0,
                'Family_History': 0
            }
            
            # Medication data
            pharmacy_extraction = context.get('pharmacy_extraction', {})
            medication_list = []
            for record in pharmacy_extraction.get('ndc_records', [])[:10]:  # Limit to 10
                if record.get('lbl_nm'):
                    medication_list.append(str(record['lbl_nm']))
            
            namespace.update({
                'medication_list': medication_list,
                'diagnosis_codes': ['I10', 'E11.9', 'E78.5'],  # Sample codes
                'diagnosis_descriptions': ['Hypertension', 'Type 2 Diabetes', 'Hyperlipidemia'],
            })
        else:
            # Fallback sample data
            namespace.update({
                'patient_age': 45,
                'heart_risk_score': 0.25,
                'medications_count': 3,
                'medical_records_count': 8,
                'diabetes_status': 'yes',
                'smoking_status': 'no',
                'bp_status': 'managed',
                'risk_factors': {'Age': 45, 'Diabetes': 1, 'Smoking': 0, 'High_BP': 1, 'Family_History': 1},
                'medication_list': ['Metformin', 'Lisinopril', 'Atorvastatin'],
                'diagnosis_codes': ['I10', 'E11.9', 'E78.5'],
                'diagnosis_descriptions': ['Hypertension', 'Type 2 Diabetes', 'Hyperlipidemia'],
            })
        
        # Execute the code
        try:
            exec(cleaned_code, namespace)
            fig = plt.gcf()
            
            if not fig.axes:
                raise ValueError("No axes in figure")
                
        except Exception as exec_error:
            logger.warning(f"Code execution failed: {exec_error}")
            
            # Create fallback visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Healthcare Analysis Dashboard', fontsize=16, fontweight='bold')
            
            # Risk factors
            risk_data = list(namespace['risk_factors'].values())
            risk_names = list(namespace['risk_factors'].keys())
            colors = ['#28a745' if x == 0 else '#dc3545' for x in risk_data]
            ax1.bar(risk_names, risk_data, color=colors)
            ax1.set_title('Risk Factors Assessment', fontweight='bold')
            ax1.set_ylabel('Risk Level')
            
            # Heart risk
            risk_score = namespace['heart_risk_score']
            risk_color = '#28a745' if risk_score < 0.3 else '#ffc107' if risk_score < 0.6 else '#dc3545'
            ax2.barh(['Heart Risk'], [risk_score], color=risk_color, alpha=0.7)
            ax2.set_xlim(0, 1)
            ax2.set_title(f'Heart Attack Risk: {risk_score:.1%}', fontweight='bold')
            
            # Medications
            if namespace['medication_list']:
                med_counts = {}
                for med in namespace['medication_list']:
                    med_counts[med] = med_counts.get(med, 0) + 1
                
                meds = list(med_counts.keys())[:5]
                counts = [med_counts[med] for med in meds]
                
                ax3.barh(meds, counts, color='#007bff')
                ax3.set_title('Top Medications', fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'No medication data', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Medications', fontweight='bold')
            
            # Health summary
            summary_text = f"""Patient Health Summary
Age: {namespace['patient_age']} years
Medications: {namespace['medications_count']}
Medical Records: {namespace['medical_records_count']}
Diabetes: {namespace['diabetes_status']}
Smoking: {namespace['smoking_status']}"""
            
            ax4.text(0.05, 0.9, summary_text, transform=ax4.transAxes, fontsize=10, verticalalignment='top')
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            
            plt.tight_layout()
            fig = plt.gcf()
        
        # Style the figure
        if fig.axes:
            for ax in fig.axes:
                try:
                    ax.tick_params(labelsize=9)
                    ax.grid(True, alpha=0.3, linestyle='--')
                except:
                    continue
        
        # Convert to image
        img_buffer = io.BytesIO()
        fig.savefig(
            img_buffer, 
            format='png', 
            bbox_inches='tight', 
            dpi=200,
            facecolor='white', 
            pad_inches=0.3,
            transparent=False
        )
        img_buffer.seek(0)
        
        # Cleanup
        plt.clf()
        plt.close('all')
        plt.ion()
        
        return img_buffer
        
    except Exception as e:
        plt.clf()
        plt.close('all')
        plt.ion()
        
        logger.error(f"Graph generation failed: {e}")
        return None

# DEMO HEALTH ANALYSIS AGENT CLASS

class DemoHealthAnalysisAgent:
    """Demo Health Analysis Agent for the complete integrated application"""
    
    def __init__(self, config: Config, ui_callback: Optional[Callable] = None):
        self.config = config
        self.ui_callback = ui_callback
        logger.info("🔧 Demo Health Analysis Agent initialized")
    
    def _notify_ui_callback(self, step_id: str, status: str, data: Dict = None):
        """Notify UI callback of step progress"""
        if self.ui_callback:
            try:
                self.ui_callback(step_id, status, data or {})
            except Exception as e:
                logger.warning(f"UI callback error: {e}")
    
    def run_analysis(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run demo analysis with realistic workflow steps"""
        
        # Simulate step-by-step analysis
        steps = [
            ('fetch_api_data', 'Fetching claims data', 2.0),
            ('deidentify_claims_data', 'Deidentifying data', 1.5),
            ('extract_claims_fields', 'Extracting fields', 2.5),
            ('extract_entities', 'Extracting entities', 2.0),
            ('analyze_trajectory', 'Analyzing trajectory', 3.0),
            ('predict_heart_attack', 'Predicting heart risk', 2.0),
            ('initialize_chatbot', 'Initializing chatbot', 1.0)
        ]
        
        for step_id, description, duration in steps:
            # Set to running
            self._notify_ui_callback(step_id, 'running', {'description': description})
            
            # Simulate processing time
            time.sleep(duration)
            
            # Set to completed
            self._notify_ui_callback(step_id, 'completed', {'description': f'{description} completed'})
        
        # Create comprehensive demo results
        demo_results = {
            "success": True,
            "patient_data": patient_data,
            
            # Demo API outputs
            "api_outputs": {
                "mcid": {"member_id": "DEMO123", "status": "active"},
                "medical": {"records_found": 8, "status": "success"},
                "pharmacy": {"records_found": 12, "status": "success"},
                "token": {"access_token": "demo_token", "expires_in": 3600}
            },
            
            # Demo deidentified data
            "deidentified_data": {
                "medical": {
                    "src_mbr_age": calculate_age(datetime.strptime(patient_data['date_of_birth'], '%Y-%m-%d').date()) if patient_data.get('date_of_birth') else 45,
                    "src_mbr_zip_cd": patient_data.get('zip_code', '12345'),
                    "medical_claims_data": {"records": [{"claim_id": f"CLM00{i}", "diagnosis": f"DIAG{i}"} for i in range(8)]}
                },
                "pharmacy": {
                    "pharmacy_claims_data": {"records": [{"rx_id": f"RX00{i}", "medication": f"MED{i}"} for i in range(12)]}
                },
                "mcid": {"deidentified_id": "DEID123"}
            },
            
            # Demo structured extractions
            "structured_extractions": {
                "medical": {
                    "hlth_srvc_records": [
                        {
                            "hlth_srvc_cd": "99213",
                            "clm_rcvd_dt": "2024-01-15",
                            "diagnosis_codes": [
                                {"code": "I10", "position": 1, "source": "primary"},
                                {"code": "E11.9", "position": 2, "source": "secondary"}
                            ]
                        },
                        {
                            "hlth_srvc_cd": "80053",
                            "clm_rcvd_dt": "2024-02-20",
                            "diagnosis_codes": [
                                {"code": "E78.5", "position": 1, "source": "primary"}
                            ]
                        }
                    ],
                    "code_meanings": {
                        "service_code_meanings": {
                            "99213": "Office/outpatient visit, established patient, level 3",
                            "80053": "Comprehensive metabolic panel"
                        },
                        "diagnosis_code_meanings": {
                            "I10": "Essential hypertension",
                            "E11.9": "Type 2 diabetes mellitus without complications",
                            "E78.5": "Hyperlipidemia"
                        }
                    },
                    "llm_call_status": "completed"
                },
                "pharmacy": {
                    "ndc_records": [
                        {
                            "ndc": "0093-0058-01",
                            "lbl_nm": "Metformin HCl 500mg",
                            "rx_filled_dt": "2024-01-10"
                        },
                        {
                            "ndc": "0071-0222-23", 
                            "lbl_nm": "Lisinopril 10mg",
                            "rx_filled_dt": "2024-01-15"
                        },
                        {
                            "ndc": "0071-0156-23",
                            "lbl_nm": "Atorvastatin 20mg", 
                            "rx_filled_dt": "2024-02-01"
                        }
                    ],
                    "code_meanings": {
                        "ndc_code_meanings": {
                            "0093-0058-01": "Metformin HCl 500mg tablets - diabetes medication",
                            "0071-0222-23": "Lisinopril 10mg tablets - ACE inhibitor for hypertension",
                            "0071-0156-23": "Atorvastatin 20mg tablets - statin for cholesterol"
                        },
                        "medication_meanings": {
                            "Metformin HCl 500mg": "First-line medication for type 2 diabetes",
                            "Lisinopril 10mg": "ACE inhibitor used to treat high blood pressure",
                            "Atorvastatin 20mg": "Statin medication to lower cholesterol"
                        }
                    },
                    "llm_call_status": "completed"
                }
            },
            
            # Demo entity extraction
            "entity_extraction": {
                "diabetics": "yes",
                "age_group": "middle-aged",
                "smoking": "no",
                "alcohol": "unknown", 
                "blood_pressure": "managed",
                "medical_conditions": ["hypertension", "diabetes", "hyperlipidemia"],
                "medications_identified": ["metformin", "lisinopril", "atorvastatin"],
                "stable_analysis": True
            },
            
            # Demo health trajectory
            "health_trajectory": """
## Comprehensive Health Trajectory Analysis

### Current Health Status
This patient presents with a well-managed chronic disease profile including type 2 diabetes, hypertension, and hyperlipidemia. Current medication regimen suggests good clinical management.

### Risk Predictions
- **Chronic Disease Risk**: Moderate - existing conditions are being treated
- **Hospitalization Risk**: Low to moderate in next 6-12 months with current management
- **Cost Projection**: Estimated annual healthcare costs of $8,000-12,000

### Care Management Recommendations
1. Continue current diabetes management with Metformin
2. Monitor blood pressure control with Lisinopril
3. Regular lipid monitoring while on Atorvastatin  
4. Schedule annual diabetic eye exam and foot care

### Quality Metrics
Patient shows good adherence to evidence-based care for diabetes and hypertension management.
""",
            
            # Demo heart attack prediction
            "heart_attack_prediction": {
                "risk_display": "Heart Disease Risk: 35.2% (Moderate Risk)",
                "confidence_display": "Confidence: 87.3%",
                "combined_display": "Heart Disease Risk: 35.2% (Moderate Risk) | Confidence: 87.3%",
                "raw_risk_score": 0.352,
                "raw_prediction": 0,
                "risk_category": "Moderate Risk",
                "prediction_method": "Enhanced ML Model",
                "prediction_timestamp": datetime.now().isoformat(),
                "model_enhanced": True
            },
            
            "heart_attack_risk_score": 0.352,
            
            "heart_attack_features": {
                "extracted_features": {
                    "Age": calculate_age(datetime.strptime(patient_data['date_of_birth'], '%Y-%m-%d').date()) if patient_data.get('date_of_birth') else 45,
                    "Gender": 1 if patient_data.get('gender') == 'M' else 0,
                    "Diabetes": 1,
                    "High_BP": 1,
                    "Smoking": 0
                },
                "feature_interpretation": {
                    "Age": f"{calculate_age(datetime.strptime(patient_data['date_of_birth'], '%Y-%m-%d').date()) if patient_data.get('date_of_birth') else 45} years old",
                    "Gender": "Male" if patient_data.get('gender') == 'M' else "Female",
                    "Diabetes": "Yes",
                    "High_BP": "Yes", 
                    "Smoking": "No"
                }
            },
            
            # Chatbot readiness
            "chatbot_ready": True,
            "graph_generation_ready": True,
            "processing_steps_completed": 7,
            "langgraph_used": True,
            "comprehensive_analysis": True,
            "enhanced_chatbot": True,
            "batch_code_meanings": True,
            "real_time_callbacks": True,
            "enhancement_version": "v10.0_complete_integrated_demo"
        }
        
        # Create chatbot context
        demo_results["chatbot_context"] = {
            "deidentified_medical": demo_results["deidentified_data"]["medical"],
            "deidentified_pharmacy": demo_results["deidentified_data"]["pharmacy"], 
            "deidentified_mcid": demo_results["deidentified_data"]["mcid"],
            "medical_extraction": demo_results["structured_extractions"]["medical"],
            "pharmacy_extraction": demo_results["structured_extractions"]["pharmacy"],
            "entity_extraction": demo_results["entity_extraction"],
            "health_trajectory": demo_results["health_trajectory"],
            "heart_attack_prediction": demo_results["heart_attack_prediction"],
            "heart_attack_risk_score": demo_results["heart_attack_risk_score"],
            "heart_attack_features": demo_results["heart_attack_features"],
            "patient_overview": {
                "age": demo_results["deidentified_data"]["medical"]["src_mbr_age"],
                "zip": demo_results["deidentified_data"]["medical"]["src_mbr_zip_cd"],
                "analysis_timestamp": datetime.now().isoformat(),
                "heart_attack_risk_level": demo_results["heart_attack_prediction"]["risk_category"],
                "model_type": "demo_comprehensive_analysis",
                "deidentification_level": "complete_demo",
                "claims_data_types": ["medical", "pharmacy", "mcid"],
                "graph_generation_supported": True,
                "batch_code_meanings_available": True
            }
        }
        
        return demo_results
    
    def chat_with_data(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Demo chat functionality with graph generation"""
        
        # Check if this is a graph request
        graph_keywords = ['chart', 'graph', 'plot', 'visualization', 'dashboard', 'timeline']
        is_graph_request = any(keyword in user_query.lower() for keyword in graph_keywords)
        
        if is_graph_request:
            return self._handle_graph_request_demo(user_query, chat_context)
        else:
            return self._handle_general_question_demo(user_query, chat_context)
    
    def _handle_graph_request_demo(self, user_query: str, chat_context: Dict[str, Any]) -> str:
        """Handle graph generation requests"""
        
        # Determine graph type
        if 'medication' in user_query.lower() and ('timeline' in user_query.lower() or 'chart' in user_query.lower()):
            graph_type = "medication_timeline"
        elif 'risk' in user_query.lower() and 'dashboard' in user_query.lower():
            graph_type = "risk_dashboard"
        elif 'pie' in user_query.lower() and 'medication' in user_query.lower():
            graph_type = "medication_pie"
        else:
            graph_type = "comprehensive_dashboard"
        
        # Generate appropriate matplotlib code
        if graph_type == "medication_timeline":
            matplotlib_code = """
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Medication timeline data
medications = medication_list
dates = ['2024-01-10', '2024-01-15', '2024-02-01']

fig, ax = plt.subplots(figsize=(12, 8))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i, (med, date) in enumerate(zip(medications, dates)):
    ax.scatter(pd.to_datetime(date), i, color=colors[i % len(colors)], s=150, alpha=0.7)
    ax.text(pd.to_datetime(date), i+0.1, med, fontsize=10, ha='center')

ax.set_yticks(range(len(medications)))
ax.set_yticklabels(medications)
ax.set_xlabel('Fill Date')
ax.set_title('Medication Fill Timeline', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
"""
        
        elif graph_type == "risk_dashboard":
            matplotlib_code = """
import matplotlib.pyplot as plt
import numpy as np

# Risk dashboard
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comprehensive Health Risk Dashboard', fontsize=16, fontweight='bold')

# Risk factors
risk_data = list(risk_factors.values())
risk_names = list(risk_factors.keys())
colors = ['#dc3545' if x == 1 else '#28a745' for x in risk_data]
ax1.bar(risk_names, risk_data, color=colors, alpha=0.7)
ax1.set_title('Risk Factors Present', fontweight='bold')
ax1.set_ylabel('Present (1) / Absent (0)')
ax1.tick_params(axis='x', rotation=45)

# Heart risk pie
heart_risk = heart_risk_score
ax2.pie([heart_risk, 1-heart_risk], 
        labels=['Risk', 'No Risk'],
        colors=['#dc3545', '#28a745'],
        autopct='%1.1f%%',
        startangle=90)
ax2.set_title(f'Heart Attack Risk: {heart_risk:.1%}', fontweight='bold')

# Risk trend
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
risk_trend = [0.2, 0.22, 0.25, 0.23, 0.24, heart_risk]
ax3.plot(months, risk_trend, marker='o', linewidth=3, markersize=8, color='#ff6b6b')
ax3.fill_between(months, risk_trend, alpha=0.3, color='#ff6b6b')
ax3.set_title('Risk Trend Over Time', fontweight='bold')
ax3.set_ylabel('Risk Score')
ax3.grid(True, alpha=0.3)

# Overall assessment
overall_risk = np.mean(risk_data + [heart_risk])
risk_color = '#28a745' if overall_risk < 0.33 else '#ffc107' if overall_risk < 0.67 else '#dc3545'
ax4.barh(['Overall Risk'], [overall_risk], color=risk_color, alpha=0.7)
ax4.set_xlim(0, 1)
ax4.set_title(f'Overall Risk Score: {overall_risk:.1%}', fontweight='bold')

plt.tight_layout()
"""
        
        elif graph_type == "medication_pie":
            matplotlib_code = """
import matplotlib.pyplot as plt

# Medication pie chart
medications = medication_list
med_counts = [1] * len(medications)  # Each medication once
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']

fig, ax = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax.pie(med_counts, labels=medications, colors=colors[:len(medications)],
                                  autopct='%1.1f%%', startangle=90)

ax.set_title('Current Medication Distribution', fontsize=16, fontweight='bold')

# Make percentage text more readable
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_weight('bold')

plt.tight_layout()
"""
        
        else:  # comprehensive_dashboard
            matplotlib_code = """
import matplotlib.pyplot as plt
import numpy as np

# Comprehensive dashboard
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Complete Healthcare Analysis Dashboard', fontsize=16, fontweight='bold')

# Risk factors
risk_data = list(risk_factors.values())
risk_names = list(risk_factors.keys())
colors = ['#28a745' if x == 0 else '#dc3545' for x in risk_data]
ax1.bar(risk_names, risk_data, color=colors)
ax1.set_title('Risk Factors Assessment', fontweight='bold')
ax1.tick_params(axis='x', rotation=45)

# Heart risk gauge
heart_risk = heart_risk_score
ax2.barh(['Heart Risk'], [heart_risk], color='#dc3545', alpha=0.7)
ax2.set_xlim(0, 1)
ax2.set_title(f'Heart Attack Risk: {heart_risk:.1%}', fontweight='bold')

# Medications
if medication_list:
    med_counts = {med: 1 for med in medication_list}
    meds = list(med_counts.keys())[:5]
    counts = list(med_counts.values())[:5]
    ax3.barh(meds, counts, color='#007bff')
    ax3.set_title('Current Medications', fontweight='bold')
else:
    ax3.text(0.5, 0.5, 'No medication data', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Medications', fontweight='bold')

# Health summary
summary_text = f'''Patient Health Summary
Age: {patient_age} years
Medications: {medications_count}
Records: {medical_records_count}
Diabetes: {diabetes_status}
Smoking: {smoking_status}'''

ax4.text(0.05, 0.9, summary_text, transform=ax4.transAxes, fontsize=10, verticalalignment='top')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

plt.tight_layout()
"""
        
        response = f"""## Healthcare Data Visualization

I'll create a {graph_type.replace('_', ' ')} visualization for your healthcare data.

```python
{matplotlib_code}
```

This visualization uses your actual patient data including medical records, pharmacy claims, and risk assessments. The chart provides clinical insights based on your comprehensive healthcare analysis."""
        
        return response
    
    def _handle_general_question_demo(self, user_query: str, chat_context: Dict[str, Any]) -> str:
        """Handle general questions about health data"""
        
        # Extract relevant information from context
        entity_extraction = chat_context.get('entity_extraction', {})
        heart_risk = chat_context.get('heart_attack_risk_score', 0.0)
        medical_extraction = chat_context.get('medical_extraction', {})
        pharmacy_extraction = chat_context.get('pharmacy_extraction', {})
        
        # Generate contextual response based on query type
        if any(keyword in user_query.lower() for keyword in ['heart', 'cardiac', 'cardiovascular']):
            response = f"""## ❤️ Cardiovascular Risk Assessment

Based on your comprehensive health analysis:

**Current Heart Attack Risk: {heart_risk:.1%}**
- Risk Category: {chat_context.get('heart_attack_prediction', {}).get('risk_category', 'Moderate Risk')}
- This assessment is based on multiple factors including age, diabetes status, blood pressure, and smoking history

**Key Risk Factors Identified:**
- Diabetes: {entity_extraction.get('diabetics', 'unknown')}
- Blood Pressure: {entity_extraction.get('blood_pressure', 'unknown')}
- Smoking: {entity_extraction.get('smoking', 'unknown')}

**Clinical Recommendations:**
1. Continue current medication regimen for diabetes and hypertension
2. Regular monitoring of blood pressure and glucose levels
3. Annual cardiovascular risk assessment
4. Lifestyle modifications including diet and exercise

**Medications Contributing to Risk Management:**
- Lisinopril: Helps control blood pressure
- Metformin: Manages diabetes effectively
- Atorvastatin: Reduces cholesterol and cardiovascular risk

Your current risk level suggests good management of existing conditions with room for continued monitoring and care."""
        
        elif any(keyword in user_query.lower() for keyword in ['medication', 'drug', 'prescription']):
            medications = pharmacy_extraction.get('ndc_records', [])
            response = f"""## 💊 Current Medication Analysis

**Active Medications ({len(medications)} total):**

"""
            for med in medications[:5]:  # Show top 5
                med_name = med.get('lbl_nm', 'Unknown')
                ndc_code = med.get('ndc', 'Unknown')
                fill_date = med.get('rx_filled_dt', 'Unknown')
                response += f"• **{med_name}**\n  - NDC: {ndc_code}\n  - Last Fill: {fill_date}\n\n"
            
            response += f"""**Medication Safety Analysis:**
- No major drug interactions identified in current regimen
- Good coverage for diabetes, hypertension, and cholesterol management
- Adherence appears consistent based on fill dates

**Therapeutic Classes Covered:**
- Diabetes Management: Metformin (first-line therapy)
- Hypertension Control: ACE inhibitor (Lisinopril)
- Cholesterol Management: Statin therapy (Atorvastatin)

**Care Management Recommendations:**
1. Continue current medication schedule
2. Monitor for side effects and efficacy
3. Regular lab work to assess medication effectiveness
4. Medication therapy management consultation if needed"""
        
        elif any(keyword in user_query.lower() for keyword in ['diagnosis', 'condition', 'icd']):
            diagnoses = medical_extraction.get('hlth_srvc_records', [])
            response = f"""## 🏥 Medical Diagnosis Analysis

**Primary Diagnoses Identified:**

"""
            
            # Extract unique diagnosis codes
            diagnosis_codes = set()
            for record in diagnoses:
                for diag in record.get('diagnosis_codes', []):
                    diagnosis_codes.add(diag.get('code', ''))
            
            diagnosis_meanings = medical_extraction.get('code_meanings', {}).get('diagnosis_code_meanings', {})
            
            for code in sorted(diagnosis_codes):
                if code:
                    meaning = diagnosis_meanings.get(code, 'Medical condition')
                    response += f"• **{code}**: {meaning}\n"
            
            response += f"""
**Clinical Assessment:**
- Total unique diagnoses: {len(diagnosis_codes)}
- Chronic conditions are well-documented and managed
- No acute or emergency conditions noted in recent claims

**Condition Management Status:**
- Diabetes (E11.9): Active management with Metformin
- Hypertension (I10): Controlled with Lisinopril
- Hyperlipidemia (E78.5): Managed with Atorvastatin

**Care Coordination:**
- Regular monitoring recommended for all chronic conditions
- Preventive care up to date based on diagnosis patterns
- Good integration between pharmacy and medical management"""
        
        else:
            # General health summary
            response = f"""## 📋 Comprehensive Health Analysis Summary

**Patient Overview:**
- Age: {chat_context.get('patient_overview', {}).get('age', 'Unknown')} years
- Risk Level: {chat_context.get('heart_attack_prediction', {}).get('risk_category', 'Moderate')}
- Active Conditions: {len(entity_extraction.get('medical_conditions', []))} chronic conditions
- Current Medications: {len(pharmacy_extraction.get('ndc_records', []))} active prescriptions

**Key Health Indicators:**
- Diabetes Status: {entity_extraction.get('diabetics', 'unknown')}
- Blood Pressure: {entity_extraction.get('blood_pressure', 'unknown')}  
- Smoking Status: {entity_extraction.get('smoking', 'unknown')}
- Heart Attack Risk: {heart_risk:.1%}

**Recent Healthcare Utilization:**
- Medical Claims: {len(medical_extraction.get('hlth_srvc_records', []))} recent records
- Pharmacy Claims: {len(pharmacy_extraction.get('ndc_records', []))} prescriptions filled

**Overall Assessment:**
Your health profile shows well-managed chronic conditions with appropriate medication therapy. Continue current care plan with regular monitoring and follow-up appointments.

**Next Steps:**
1. Maintain current medication regimen
2. Schedule regular follow-up appointments
3. Monitor key health indicators
4. Continue lifestyle modifications as recommended

Would you like me to dive deeper into any specific aspect of your health analysis or create a visualization to better understand your data?"""
        
        return response

# SESSION STATE INITIALIZATION

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
    if 'calculated_age' not in st.session_state:
        st.session_state.calculated_age = None
    
    # Enhanced workflow steps tracking
    if 'workflow_steps' not in st.session_state:
        st.session_state.workflow_steps = [
            {'name': 'API Fetch', 'status': 'pending', 'description': 'Fetching comprehensive claims data', 'icon': '⚡', 'step_id': 'fetch_api_data'},
            {'name': 'Deidentification', 'status': 'pending', 'description': 'Advanced PII removal with clinical preservation', 'icon': '🔒', 'step_id': 'deidentify_claims_data'},
            {'name': 'Field Extraction', 'status': 'pending', 'description': 'Extracting medical and pharmacy fields', 'icon': '🚀', 'step_id': 'extract_claims_fields'},
            {'name': 'Entity Extraction', 'status': 'pending', 'description': 'Advanced health entity identification', 'icon': '🎯', 'step_id': 'extract_entities'},
            {'name': 'Health Trajectory', 'status': 'pending', 'description': 'Comprehensive predictive health analysis', 'icon': '📈', 'step_id': 'analyze_trajectory'},
            {'name': 'Heart Risk Prediction', 'status': 'pending', 'description': 'ML-based cardiovascular assessment', 'icon': '❤️', 'step_id': 'predict_heart_attack'},
            {'name': 'Chatbot Initialization', 'status': 'pending', 'description': 'AI assistant with graph generation', 'icon': '💬', 'step_id': 'initialize_chatbot'}
        ]
    
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'selected_prompt' not in st.session_state:
        st.session_state.selected_prompt = None
    if 'completed_steps' not in st.session_state:
        st.session_state.completed_steps = set()
    if 'available_sections' not in st.session_state:
        st.session_state.available_sections = set()

# WORKFLOW AND UI FUNCTIONS

def get_step_status(step_id: str) -> str:
    """Get current status of a workflow step"""
    for step in st.session_state.workflow_steps:
        if step['step_id'] == step_id:
            return step['status']
    return 'pending'

def is_step_completed(step_id: str) -> bool:
    """Check if a step is completed"""
    return get_step_status(step_id) == 'completed'

def update_step_status(step_id: str, status: str):
    """Update the status of a workflow step"""
    for step in st.session_state.workflow_steps:
        if step['step_id'] == step_id:
            step['status'] = status
            if status == 'completed':
                st.session_state.completed_steps.add(step_id)
            break

def get_available_sections() -> set:
    """Get sections that should be available based on completed steps"""
    available = set()
    
    if is_step_completed('fetch_api_data'):
        available.add('claims_data')
    if is_step_completed('extract_claims_fields'):
        available.add('claims_analysis')
    if is_step_completed('extract_entities'):
        available.add('entity_extraction')
    if is_step_completed('analyze_trajectory'):
        available.add('health_trajectory')
    if is_step_completed('predict_heart_attack'):
        available.add('heart_risk')
    if is_step_completed('initialize_chatbot'):
        available.add('chatbot_ready')
    
    return available

def display_advanced_professional_workflow():
    """Display the advanced professional workflow animation with real-time updates"""
    
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
        <p style="color: #34495e; font-size: 1.1rem;">Real-time workflow with synchronized UI updates</p>
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
    
    # Display each step with real-time status
    for i, step in enumerate(st.session_state.workflow_steps):
        status = step['status']
        name = step['name']
        description = step['description']
        icon = step['icon']
        
        # Determine styling based on status
        if status == 'completed':
            step_class = "workflow-step completed"
            status_emoji = "✅"
            status_color = "#28a745"
        elif status == 'running':
            step_class = "workflow-step running"
            status_emoji = "🔄"
            status_color = "#ffc107"
        elif status == 'error':
            step_class = "workflow-step error"
            status_emoji = "❌"
            status_color = "#dc3545"
        else:
            step_class = "workflow-step"
            status_emoji = "⏳"
            status_color = "#6c757d"
        
        st.markdown(f"""
        <div class="{step_class}">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 1.5rem;">{icon}</div>
                <div style="flex: 1;">
                    <h4 style="margin: 0; color: #2c3e50;">{name}</h4>
                    <p style="margin: 0; color: #666; font-size: 0.9rem;">{description}</p>
                </div>
                <div style="display: flex; flex-direction: column; align-items: center;">
                    <div style="font-size: 1.2rem;">{status_emoji}</div>
                    <div style="font-size: 0.8rem; color: {status_color}; font-weight: 600; text-transform: uppercase;">{status}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Status message
    if running_steps > 0:
        current_step_name = next((step['name'] for step in st.session_state.workflow_steps if step['status'] == 'running'), 'Processing')
        status_message = f"🔄 Currently executing: {current_step_name}"
        status_color = "#ffc107"
    elif completed_steps == total_steps:
        status_message = "✅ All healthcare workflow steps completed successfully!"
        status_color = "#28a745"
    elif error_steps > 0:
        status_message = f"❌ {error_steps} step(s) encountered errors"
        status_color = "#dc3545"
    else:
        status_message = "⏳ Healthcare analysis pipeline ready..."
        status_color = "#007bff"
    
    st.markdown(f"""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.8); border-radius: 10px; border-left: 4px solid {status_color};">
        <p style="margin: 0; font-weight: 600; color: #2c3e50;">{status_message}</p>
    </div>
    </div>
    """, unsafe_allow_html=True)

# CHATBOT FUNCTIONS

def create_launch_button_with_javascript():
    """Create JavaScript launch button for chatbot window"""
    
    # Get current URL and modify for chatbot
    chatbot_params = {"chatbot": "true"}
    chatbot_url = f"?{urlencode(chatbot_params)}"
    
    js_code = f"""
    <script>
    function openChatbotWindow() {{
        const chatbotWindow = window.open(
            '{chatbot_url}', 
            'MedicalAIChatbot',
            'width=1200,height=800,scrollbars=yes,resizable=yes,status=no,toolbar=no,menubar=no'
        );
        
        if (chatbotWindow) {{
            chatbotWindow.focus();
        }} else {{
            alert('Please allow popups for this site to open the AI Assistant');
        }}
    }}
    </script>
    
    <button class="chatbot-launch-btn" onclick="openChatbotWindow()">
        🤖 Open AI Assistant
    </button>
    """
    
    return js_code

def display_chatbot_status_indicator():
    """Display status indicator for chatbot availability"""
    
    if st.session_state.get('chatbot_context') and st.session_state.get('analysis_results'):
        status = "ready"
        message = "AI Assistant Ready!"
        color = "#28a745"
        icon = "✅"
    elif st.session_state.get('analysis_running'):
        status = "processing"
        message = "Analysis in Progress..."
        color = "#ffc107"
        icon = "🔄"
    else:
        status = "waiting"
        message = "Run Analysis First"
        color = "#6c757d"
        icon = "⏳"
    
    st.markdown(f"""
    <div class="status-indicator status-{status}" style="background: {color};">
        {icon} {message}
    </div>
    """, unsafe_allow_html=True)

def display_chatbot_interface():
    """Display the complete chatbot interface for separate window"""
    
    # Hide Streamlit elements for cleaner chatbot interface
    st.markdown("""
    <style>
    header[data-testid="stHeader"] {
        height: 0px;
        display: none;
    }
    
    .main .block-container {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Chatbot header
    st.markdown('''
    <div class="chatbot-window-header">
        <h1>🤖 Medical AI Assistant</h1>
        <p>Comprehensive healthcare analysis with real-time visualizations</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Check data availability
    if not st.session_state.get('chatbot_context'):
        st.error("⚠️ No health analysis data available. Please run an analysis in the main window first.")
        
        st.markdown(f"""
        <div style="text-align: center; margin: 2rem 0;">
            <a href="?" style="
                background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
                color: white;
                padding: 1rem 2rem;
                border-radius: 25px;
                text-decoration: none;
                font-weight: 600;
                display: inline-block;
                box-shadow: 0 8px 25px rgba(0, 123, 255, 0.4);
            ">
                🏠 Return to Main Application
            </a>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Display metrics
    if st.session_state.chatbot_context:
        medical_records = len(st.session_state.chatbot_context.get('medical_extraction', {}).get('hlth_srvc_records', []))
        pharmacy_records = len(st.session_state.chatbot_context.get('pharmacy_extraction', {}).get('ndc_records', []))
        heart_risk = st.session_state.chatbot_context.get('heart_attack_risk_score', 0.0)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Medical Records", medical_records)
        with col2:
            st.metric("Medications", pharmacy_records)
        with col3:
            st.metric("Heart Risk", f"{heart_risk:.1%}")
        with col4:
            st.metric("Messages", len(st.session_state.chatbot_messages))
    
    # Main chat interface
    st.markdown("### 💬 Chat with Your Health Data")
    
    # Chat input
    user_question = st.chat_input("Ask about your health analysis or request visualizations...")
    
    # Handle chat input
    if user_question:
        # Add user message
        st.session_state.chatbot_messages.append({"role": "user", "content": user_question})
        
        # Get bot response
        try:
            with st.spinner("🤖 Analyzing your health data..."):
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
    
    # Handle selected prompt from sidebar
    if st.session_state.selected_prompt:
        # Add user message
        st.session_state.chatbot_messages.append({"role": "user", "content": st.session_state.selected_prompt})
        
        # Get bot response
        try:
            with st.spinner("🤖 Processing your request..."):
                chatbot_response = st.session_state.agent.chat_with_data(
                    st.session_state.selected_prompt, 
                    st.session_state.chatbot_context, 
                    st.session_state.chatbot_messages
                )
            
            # Add assistant response
            st.session_state.chatbot_messages.append({"role": "assistant", "content": chatbot_response})
            
            # Clear selected prompt
            st.session_state.selected_prompt = None
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.selected_prompt = None
    
    # Display chat history
    st.markdown("### 📜 Chat History")
    
    if st.session_state.chatbot_messages:
        # Display messages in reverse order (newest first)
        for message in reversed(st.session_state.chatbot_messages):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    # Check for matplotlib code and handle graphs
                    matplotlib_code = extract_matplotlib_code(message["content"])
                    if matplotlib_code:
                        # Display text content without code
                        text_content = message["content"]
                        for pattern in [f"```python\n{matplotlib_code}\n```", f"```\n{matplotlib_code}\n```"]:
                            text_content = text_content.replace(pattern, "")
                        
                        if text_content.strip():
                            st.write(text_content.strip())
                        
                        # Execute and display graph
                        st.markdown('<div class="graph-display">', unsafe_allow_html=True)
                        st.markdown("#### 📊 Generated Healthcare Visualization")
                        
                        try:
                            with st.spinner("Generating visualization..."):
                                img_buffer = execute_matplotlib_code_enhanced_stability(matplotlib_code)
                                if img_buffer:
                                    st.image(img_buffer, use_container_width=True)
                                    st.success("✅ Graph generated successfully!")
                                else:
                                    st.error("❌ Failed to generate graph")
                        except Exception as e:
                            st.error(f"❌ Graph generation error: {str(e)}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.write(message["content"])
                else:
                    st.write(message["content"])
    else:
        st.info("Start a conversation! Use the prompts in the sidebar or type your question.")
    
    # Clear chat button
    if st.session_state.chatbot_messages:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chatbot_messages = []
            st.success("✅ Chat history cleared!")
            st.rerun()

def display_categorized_prompts_sidebar():
    """Display categorized prompts in sidebar"""
    
    # Define categorized prompts
    prompt_categories = {
        "📋 Medical Records": [
            "What diagnoses were found in the medical records?",
            "What medical procedures were performed?", 
            "List all ICD-10 diagnosis codes found",
            "Show me the medical claims timeline"
        ],
        "💊 Medications": [
            "What medications is this patient taking?",
            "What NDC codes were identified?",
            "Create a medication timeline chart",
            "Show medication distribution by class"
        ],
        "🎯 Risk Assessment": [
            "What is the heart attack risk and explain why?",
            "Risk of developing chronic diseases?",
            "Generate a comprehensive risk dashboard",
            "Show risk factors as a visualization"
        ],
        "📊 Visualizations": [
            "Create a medication timeline chart",
            "Generate a comprehensive risk dashboard", 
            "Show me a pie chart of medications",
            "Create a health overview visualization"
        ],
        "🔮 Predictions": [
            "Predict patient life expectancy scenarios",
            "Is this person likely to be high-cost next year?",
            "Estimate future healthcare costs",
            "Model disease progression over time"
        ]
    }
    
    # Create expandable sections for each category
    for category, prompts in prompt_categories.items():
        with st.expander(category, expanded=False):
            for i, prompt in enumerate(prompts):
                if st.button(prompt, key=f"cat_prompt_{category}_{i}", use_container_width=True):
                    st.session_state.selected_prompt = prompt
                    st.rerun()

# RESULTS DISPLAY FUNCTIONS

def display_dynamic_results_sections(available_sections: set, results: Dict):
    """Display results sections dynamically as they become available"""
    
    if 'claims_data' in available_sections:
        with st.expander("📊 Claims Data", expanded=False):
            st.markdown('<div class="section-expandable available">', unsafe_allow_html=True)
            display_claims_data_section(results)
            st.markdown('</div>', unsafe_allow_html=True)
    
    if 'claims_analysis' in available_sections:
        with st.expander("🔬 Claims Data Analysis", expanded=False):
            st.markdown('<div class="section-expandable available">', unsafe_allow_html=True)
            display_claims_analysis_section(results)
            st.markdown('</div>', unsafe_allow_html=True)
    
    if 'entity_extraction' in available_sections:
        with st.expander("🎯 Entity Extraction", expanded=False):
            st.markdown('<div class="section-expandable available">', unsafe_allow_html=True)
            display_entity_extraction_section(results)
            st.markdown('</div>', unsafe_allow_html=True)
    
    if 'health_trajectory' in available_sections:
        with st.expander("📈 Health Trajectory", expanded=False):
            st.markdown('<div class="section-expandable available">', unsafe_allow_html=True)
            display_health_trajectory_section(results)
            st.markdown('</div>', unsafe_allow_html=True)
    
    if 'heart_risk' in available_sections:
        with st.expander("❤️ Heart Attack Risk Prediction", expanded=False):
            st.markdown('<div class="section-expandable available">', unsafe_allow_html=True)
            display_heart_risk_section(results)
            st.markdown('</div>', unsafe_allow_html=True)
    
    if 'chatbot_ready' in available_sections:
        st.success("🎉 Analysis complete! Click the '🤖 Open AI Assistant' button to start chatting with your health data.")

def display_claims_data_section(results):
    """Display claims data section"""
    deidentified_data = safe_get(results, 'deidentified_data', {})
    api_outputs = safe_get(results, 'api_outputs', {})
    
    if deidentified_data or api_outputs:
        tab1, tab2, tab3 = st.tabs(["📋 Medical Claims", "💊 Pharmacy Claims", "🆔 MCID Data"])
        
        with tab1:
            medical_data = safe_get(deidentified_data, 'medical', {})
            if medical_data:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Patient Age", medical_data.get('src_mbr_age', 'Unknown'))
                with col2:
                    st.metric("Zip Code", medical_data.get('src_mbr_zip_cd', 'Unknown'))
                with col3:
                    records_count = len(medical_data.get('medical_claims_data', {}).get('records', []))
                    st.metric("Medical Records", records_count)
                
                with st.expander("Medical Claims JSON Data", expanded=False):
                    st.json(medical_data)
            else:
                st.warning("No medical claims data available")
        
        with tab2:
            pharmacy_data = safe_get(deidentified_data, 'pharmacy', {})
            if pharmacy_data:
                col1, col2 = st.columns(2)
                with col1:
                    records_count = len(pharmacy_data.get('pharmacy_claims_data', {}).get('records', []))
                    st.metric("Pharmacy Records", records_count)
                with col2:
                    st.metric("Data Status", "Available")
                
                with st.expander("Pharmacy Claims JSON Data", expanded=False):
                    st.json(pharmacy_data)
            else:
                st.warning("No pharmacy claims data available")
        
        with tab3:
            mcid_data = safe_get(api_outputs, 'mcid', {})
            if mcid_data:
                st.metric("MCID Status", "Available")
                with st.expander("MCID JSON Data", expanded=False):
                    st.json(mcid_data)
            else:
                st.warning("MCID data not available")

def display_claims_analysis_section(results):
    """Display claims analysis section"""
    st.markdown("### 🔬 Enhanced Code Analysis")
    
    medical_extraction = safe_get(results, 'structured_extractions', {}).get('medical', {})
    pharmacy_extraction = safe_get(results, 'structured_extractions', {}).get('pharmacy', {})
    
    if medical_extraction:
        st.markdown("#### 🏥 Medical Codes Analysis")
        medical_meanings = medical_extraction.get("code_meanings", {})
        service_meanings = medical_meanings.get("service_code_meanings", {})
        diagnosis_meanings = medical_meanings.get("diagnosis_code_meanings", {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Service Codes", len(service_meanings))
        with col2:
            st.metric("Diagnosis Codes", len(diagnosis_meanings))
        
        if diagnosis_meanings:
            st.markdown("**Sample ICD-10 Codes:**")
            for i, (code, meaning) in enumerate(list(diagnosis_meanings.items())[:3]):
                st.write(f"• **{code}**: {meaning}")
    
    if pharmacy_extraction:
        st.markdown("#### 💊 Pharmacy Codes Analysis")
        pharmacy_meanings = pharmacy_extraction.get("code_meanings", {})
        ndc_meanings = pharmacy_meanings.get("ndc_code_meanings", {})
        med_meanings = pharmacy_meanings.get("medication_meanings", {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("NDC Codes", len(ndc_meanings))
        with col2:
            st.metric("Medications", len(med_meanings))
        
        if med_meanings:
            st.markdown("**Sample Medications:**")
            for i, (med, meaning) in enumerate(list(med_meanings.items())[:3]):
                st.write(f"• **{med}**: {meaning}")

def display_entity_extraction_section(results):
    """Display entity extraction section"""
    st.markdown("### 🎯 Health Entity Analysis")
    
    entity_extraction = safe_get(results, 'entity_extraction', {})
    
    if entity_extraction:
        entities_data = [
            {'icon': '🩺', 'label': 'Diabetes Status', 'value': entity_extraction.get('diabetics', 'unknown'), 'key': 'diabetics'},
            {'icon': '👥', 'label': 'Age Group', 'value': entity_extraction.get('age_group', 'unknown'), 'key': 'age_group'},
            {'icon': '🚬', 'label': 'Smoking Status', 'value': entity_extraction.get('smoking', 'unknown'), 'key': 'smoking'},
            {'icon': '🍷', 'label': 'Alcohol Use', 'value': entity_extraction.get('alcohol', 'unknown'), 'key': 'alcohol'},
            {'icon': '💓', 'label': 'Blood Pressure', 'value': entity_extraction.get('blood_pressure', 'unknown'), 'key': 'blood_pressure'}
        ]
        
        cols = st.columns(len(entities_data))
        
        for i, (col, entity) in enumerate(zip(cols, entities_data)):
            with col:
                value = entity['value']
                color = "🔴" if entity['key'] in ['diabetics', 'smoking'] and value == 'yes' else "🟢" if entity['key'] in ['diabetics', 'smoking'] and value == 'no' else "🟡"
                
                st.markdown(f"""
                <div class="metric-summary-box">
                    <div style="font-size: 2rem;">{entity['icon']}</div>
                    <div style="font-size: 0.9rem; color: #6c757d; font-weight: 600;">{entity['label']}</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #2c3e50;">{color} {value.upper()}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Entity extraction data not available")

def display_health_trajectory_section(results):
    """Display health trajectory section"""
    st.markdown("### 📈 Predictive Health Analysis")
    
    health_trajectory = safe_get(results, 'health_trajectory', '')
    
    if health_trajectory:
        st.markdown(health_trajectory)
        
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
    else:
        st.warning("Health trajectory analysis not available")

def display_heart_risk_section(results):
    """Display heart attack risk prediction section"""
    st.markdown("### ❤️ Cardiovascular Risk Assessment")
    
    heart_attack_prediction = safe_get(results, 'heart_attack_prediction', {})
    
    if heart_attack_prediction:
        combined_display = heart_attack_prediction.get("combined_display", "Heart Disease Risk: Not available")
        risk_category = heart_attack_prediction.get("risk_category", "Unknown")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_score = heart_attack_prediction.get("raw_risk_score", 0.0)
            st.metric("Risk Score", f"{risk_score:.1%}")
        with col2:
            st.metric("Risk Category", risk_category)
        with col3:
            st.metric("Assessment", "Complete")
        
        # Risk visualization
        risk_value = heart_attack_prediction.get("raw_risk_score", 0.0) * 100
        st.progress(risk_value / 100)
        
        # Clinical interpretation
        if risk_category == "High Risk":
            st.error("⚠️ High cardiovascular risk detected. Recommend immediate clinical consultation.")
        elif risk_category == "Moderate Risk":
            st.warning("⚠️ Moderate cardiovascular risk. Consider preventive measures and monitoring.")
        else:
            st.success("✅ Low cardiovascular risk. Continue current health maintenance.")
    else:
        st.warning("Heart attack risk prediction not available")

# MAIN APPLICATION LOGIC

def run_enhanced_analysis_with_realtime_updates(patient_data: Dict[str, Any]):
    """Run analysis with enhanced real-time UI updates"""
    
    # Initialize analysis state
    st.session_state.analysis_running = True
    st.session_state.analysis_results = None
    st.session_state.chatbot_messages = []
    st.session_state.chatbot_context = None
    st.session_state.completed_steps = set()
    
    # Reset workflow steps
    for step in st.session_state.workflow_steps:
        step['status'] = 'pending'
    
    # Create UI callback function
    def ui_callback(step_id: str, status: str, data: Dict = None):
        """Callback function to update UI in real-time"""
        update_step_status(step_id, status)
        
        # Update available sections based on completed steps
        if status == 'completed':
            st.session_state.completed_steps.add(step_id)
            st.session_state.available_sections = get_available_sections()
    
    # Initialize agent with callback
    try:
        config = Config()
        st.session_state.agent = DemoHealthAnalysisAgent(config, ui_callback=ui_callback)
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        st.session_state.analysis_running = False
        return
    
    # Create placeholders for dynamic updates
    workflow_placeholder = st.empty()
    results_placeholder = st.empty()
    
    # Run analysis with real-time updates
    with st.spinner("🚀 Running Enhanced Healthcare Analysis..."):
        try:
            # Display initial workflow
            with workflow_placeholder.container():
                display_advanced_professional_workflow()
            
            # Run the actual analysis
            results = st.session_state.agent.run_analysis(patient_data)
            
            # Update session state
            st.session_state.analysis_results = results
            st.session_state.analysis_running = False
            
            # Set chatbot context if successful
            if results and results.get("success") and results.get("chatbot_ready"):
                st.session_state.chatbot_context = results.get("chatbot_context")
            
            # Display final results
            with results_placeholder.container():
                if results.get("success"):
                    st.success("✅ Enhanced Healthcare Analysis completed successfully!")
                    st.balloons()
                    st.info("🤖 AI Assistant is now ready! Click the floating button to open in a new window.")
                else:
                    st.error("❌ Healthcare Analysis encountered errors!")
            
            # Final workflow update
            with workflow_placeholder.container():
                display_advanced_professional_workflow()
            
            # Force rerun to show results
            st.rerun()
            
        except Exception as e:
            st.session_state.analysis_running = False
            st.error(f"Analysis failed: {str(e)}")
            
            # Mark all steps as error
            for step in st.session_state.workflow_steps:
                step['status'] = 'error'

# MAIN APPLICATION

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Check if this is chatbot mode
    query_params = st.query_params
    if query_params.get("chatbot") == "true":
        display_chatbot_interface()
        return
    
    # Enhanced Main Title
    st.markdown('<h1 class="main-header">Complete Health Agent with Separate Chatbot</h1>', unsafe_allow_html=True)
    
    # Add chatbot launch button if ready
    if st.session_state.get('chatbot_context'):
        js_code = create_launch_button_with_javascript()
        st.markdown(js_code, unsafe_allow_html=True)
    
    # Display status indicator
    display_chatbot_status_indicator()
    
    # PATIENT INFORMATION SECTION
    st.markdown("""
    <div class="section-box">
        <div class="section-title">Patient Information</div>
    </div>
    """, unsafe_allow_html=True)
    
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
        
        # Enhanced run analysis button
        submitted = st.form_submit_button(
            "🚀 Run Complete Healthcare Analysis", 
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
            run_enhanced_analysis_with_realtime_updates(patient_data)
    
    # Display workflow if analysis is running
    if st.session_state.analysis_running:
        display_advanced_professional_workflow()
    
    # Display results sections dynamically
    if st.session_state.analysis_results:
        available_sections = get_available_sections()
        display_dynamic_results_sections(available_sections, st.session_state.analysis_results)

# SIDEBAR WITH CHATBOT PROMPTS

with st.sidebar:
    if st.session_state.get('chatbot_context'):
        st.title("🤖 Medical Assistant")
        st.markdown("---")
        
        # Status
        if st.session_state.get('chatbot_context'):
            st.success("✅ AI Assistant Ready!")
        else:
            st.info("⏳ Complete analysis to enable chatbot")
        
        st.markdown("---")
        
        # Display categorized prompts in sidebar
        display_categorized_prompts_sidebar()
        
        # Additional tools
        st.markdown("---")
        st.markdown("### 🛠️ Tools")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            if st.session_state.chatbot_context:
                st.success("✅ Data refreshed!")
            else:
                st.warning("⚠️ No data to refresh")
        
        if st.button("📊 Generate Sample Chart", use_container_width=True):
            if st.session_state.chatbot_context:
                st.session_state.chatbot_messages.append({
                    "role": "user",
                    "content": "Create a comprehensive health dashboard"
                })
                st.rerun()
            else:
                st.warning("⚠️ Complete analysis first")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #6c757d; font-size: 0.8rem;">
            <p>🤖 Complete Health Agent v2.0</p>
            <p>Real-time Analysis & Separate Chatbot</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Placeholder when chatbot is not ready
        st.title("🤖 Medical Assistant")
        st.info("Complete health analysis to enable AI assistant")
        st.markdown("---")
        
        st.markdown("**What you can ask when ready:**")
        st.markdown("• **Medical Records:** Diagnoses, procedures, ICD codes")
        st.markdown("• **Medications:** Prescriptions, NDC codes, interactions") 
        st.markdown("• **Risk Assessment:** Heart attack risk, chronic conditions")
        st.markdown("• **Visualizations:** Charts, graphs, dashboards")
        st.markdown("• **Predictions:** Health outcomes, cost analysis")
        st.markdown("---")
        
        st.markdown("**Enhanced Features:**")
        st.markdown("• Separate chatbot window")
        st.markdown("• Real-time graph generation")
        st.markdown("• Categorized prompt system")
        st.markdown("• Professional visualizations")
        st.markdown("• Comprehensive health analysis")

# MAIN EXECUTION
if __name__ == "__main__":
    main()
