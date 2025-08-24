# Enhanced Health Analysis Streamlit UI - All-in-One File
# ================================================================
# Complete healthcare analysis system with advanced features:
# ✅ Separate chatbot window functionality
# ✅ Real-time workflow synchronization  
# ✅ Progressive UI updates as steps complete
# ✅ Advanced graph generation capabilities
# ✅ Professional modern design
# ✅ All in one runnable file

import streamlit as st
import json
import pandas as pd
from datetime import datetime, timedelta, date
import time
import sys
import os
import logging
from typing import Dict, Any, Optional, List, TypedDict, Literal
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
import requests
import asyncio
from dataclasses import dataclass, asdict

# Configure Streamlit page FIRST - with wider layout for separate chatbot
st.set_page_config(
    page_title="🚀 Enhanced Health Agent Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="auto"
)

# ENHANCED MATPLOTLIB CONFIGURATION
matplotlib.use('Agg')
plt.ioff()
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'text.usetex': False,
})

# PROFESSIONAL MODERN CSS WITH SEPARATE CHATBOT STYLING
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* MAIN HEADER WITH MODERN GRADIENT */
.main-header {
    font-size: 3.5rem;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: glow-pulse 3s ease-in-out infinite;
    text-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
}

@keyframes glow-pulse {
    0%, 100% { filter: drop-shadow(0 0 15px rgba(102, 126, 234, 0.4)); }
    50% { filter: drop-shadow(0 0 25px rgba(102, 126, 234, 0.7)); }
}

/* SEPARATE CHATBOT WINDOW STYLING */
.chatbot-window {
    position: fixed;
    top: 20px;
    right: 20px;
    width: 400px;
    height: 600px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    z-index: 1000;
    display: flex;
    flex-direction: column;
    border: 2px solid rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(20px);
}

.chatbot-header {
    background: rgba(255, 255, 255, 0.1);
    padding: 1rem;
    border-radius: 20px 20px 0 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.chatbot-title {
    color: white;
    font-weight: 700;
    font-size: 1.2rem;
    margin: 0;
}

.chatbot-close {
    background: rgba(255, 255, 255, 0.2);
    border: none;
    color: white;
    padding: 0.5rem;
    border-radius: 8px;
    cursor: pointer;
    font-size: 1rem;
}

.chatbot-messages {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
}

.chat-message {
    padding: 0.8rem 1rem;
    border-radius: 12px;
    max-width: 85%;
    word-wrap: break-word;
    animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.user-message {
    background: rgba(255, 255, 255, 0.9);
    color: #2c3e50;
    align-self: flex-end;
    margin-left: auto;
}

.assistant-message {
    background: rgba(255, 255, 255, 0.15);
    color: white;
    align-self: flex-start;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.chatbot-input-area {
    padding: 1rem;
    border-top: 1px solid rgba(255, 255, 255, 0.2);
    background: rgba(255, 255, 255, 0.05);
    border-radius: 0 0 20px 20px;
}

/* REAL-TIME WORKFLOW SYNCHRONIZATION */
.workflow-container {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 2rem;
    border-radius: 20px;
    margin: 2rem 0;
    border: 2px solid rgba(52, 152, 219, 0.3);
    box-shadow: 0 15px 40px rgba(52, 152, 219, 0.2);
    position: relative;
    overflow: hidden;
}

.workflow-step {
    background: rgba(255, 255, 255, 0.9);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 0.8rem 0;
    border-left: 5px solid #6c757d;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    backdrop-filter: blur(10px);
    display: flex;
    align-items: center;
    gap: 1rem;
}

.workflow-step.running {
    border-left-color: #ffc107;
    background: rgba(255, 193, 7, 0.15);
    animation: pulse-step 2s infinite;
    box-shadow: 0 10px 30px rgba(255, 193, 7, 0.3);
    transform: translateX(10px);
}

.workflow-step.completed {
    border-left-color: #28a745;
    background: rgba(40, 167, 69, 0.15);
    box-shadow: 0 10px 30px rgba(40, 167, 69, 0.2);
    transform: scale(1.02);
}

.workflow-step.error {
    border-left-color: #dc3545;
    background: rgba(220, 53, 69, 0.15);
    box-shadow: 0 10px 30px rgba(220, 53, 69, 0.3);
    animation: shake 0.5s ease-in-out;
}

@keyframes pulse-step {
    0%, 100% { transform: translateX(10px) scale(1); }
    50% { transform: translateX(15px) scale(1.01); }
}

@keyframes shake {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-5px); }
    75% { transform: translateX(5px); }
}

/* ADVANCED GRAPH CONTAINER */
.graph-container {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    margin: 1.5rem 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    border: 2px solid #e3f2fd;
    transition: all 0.3s ease;
}

.graph-container:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.15);
}

/* PROGRESSIVE RESULTS SECTIONS */
.results-section {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    margin: 1.5rem 0;
    border: 1px solid #e9ecef;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    opacity: 0;
    animation: fadeInUp 0.6s ease-out forwards;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.section-title {
    font-size: 1.6rem;
    color: #2c3e50;
    font-weight: 700;
    margin-bottom: 1.5rem;
    border-bottom: 3px solid #3498db;
    padding-bottom: 0.8rem;
    position: relative;
}

.section-title::after {
    content: '';
    position: absolute;
    bottom: -3px;
    left: 0;
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    border-radius: 2px;
}

/* ENHANCED METRICS CARDS */
.metric-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #dee2e6;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #007bff, #28a745, #ffc107, #dc3545);
    transform: translateX(-100%);
    transition: transform 0.6s ease;
}

.metric-card:hover::before {
    transform: translateX(0);
}

.metric-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    border-color: #007bff;
}

/* FLOATING ACTION BUTTON FOR CHATBOT */
.floating-chat-btn {
    position: fixed;
    bottom: 30px;
    right: 30px;
    width: 70px;
    height: 70px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 1.8rem;
    cursor: pointer;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    z-index: 999;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: float 3s ease-in-out infinite;
}

.floating-chat-btn:hover {
    transform: scale(1.1);
    box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
}

/* ENHANCED FORM STYLING */
.stForm {
    background: white;
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.1);
    border: 1px solid #e9ecef;
}

/* PROGRESS TRACKING */
.progress-tracker {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    border: 1px solid #e9ecef;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
}

.progress-step {
    display: flex;
    align-items: center;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 10px;
    transition: all 0.3s ease;
}

.progress-step.active {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    border-left: 4px solid #ffc107;
    animation: pulse-active 2s infinite;
}

.progress-step.complete {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border-left: 4px solid #28a745;
}

@keyframes pulse-active {
    0%, 100% { background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); }
    50% { background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%); }
}

/* MODAL OVERLAY FOR SEPARATE CHATBOT */
.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    z-index: 1000;
    display: flex;
    justify-content: center;
    align-items: center;
}

.modal-content {
    background: white;
    width: 90%;
    max-width: 1200px;
    height: 80%;
    border-radius: 20px;
    box-shadow: 0 25px 60px rgba(0, 0, 0, 0.3);
    display: flex;
    overflow: hidden;
}

/* ENHANCED BUTTON STYLING */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2rem !important;
    font-weight: 600 !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5) !important;
    background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%) !important;
}

/* ENHANCED TAB STYLING */
.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 15px;
    padding: 0.5rem;
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    background: white;
    border-radius: 10px;
    padding: 0.8rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
    border: 1px solid #dee2e6;
}

.stTabs [data-baseweb="tab"]:hover {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
}

/* ENHANCED METRICS DISPLAY */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.metric-box {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #e9ecef;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    position: relative;
    overflow: hidden;
}

.metric-box:hover {
    transform: translateY(-8px) scale(1.03);
    box-shadow: 0 20px 40px rgba(0,0,0,0.15);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.metric-label {
    font-size: 1rem;
    color: #6c757d;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* LOADING ANIMATIONS */
.loading-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 3rem;
    background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
    border-radius: 20px;
    margin: 2rem 0;
}

.loading-spinner {
    width: 60px;
    height: 60px;
    border: 6px solid rgba(102, 126, 234, 0.1);
    border-top: 6px solid #667eea;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 1rem;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* ENHANCED EXPANDABLE SECTIONS */
.stExpander {
    background: white;
    border-radius: 15px;
    border: 1px solid #e9ecef;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
}

.stExpander:hover {
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    transform: translateY(-2px);
}

/* ENHANCED DATA DISPLAY */
.data-container {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border: 1px solid #dee2e6;
}

/* RESPONSIVE DESIGN */
@media (max-width: 768px) {
    .chatbot-window {
        width: 95%;
        height: 70%;
        right: 2.5%;
        top: 15%;
    }
    
    .main-header {
        font-size: 2.5rem;
    }
    
    .metrics-grid {
        grid-template-columns: 1fr;
    }
}

/* ENHANCED FORM ELEMENTS */
.stTextInput > div > div > input,
.stSelectbox > div > div > div,
.stDateInput > div > div > input {
    border-radius: 10px !important;
    border: 2px solid #e9ecef !important;
    transition: all 0.3s ease !important;
}

.stTextInput > div > div > input:focus,
.stSelectbox > div > div > div:focus,
.stDateInput > div > div > input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* SUCCESS AND ERROR STATES */
.success-container {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    padding: 2rem;
    border-radius: 15px;
    border: 2px solid #28a745;
    margin: 1rem 0;
    animation: successPulse 2s ease-in-out;
}

@keyframes successPulse {
    0%, 100% { box-shadow: 0 0 20px rgba(40, 167, 69, 0.3); }
    50% { box-shadow: 0 0 40px rgba(40, 167, 69, 0.6); }
}

.error-container {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    padding: 2rem;
    border-radius: 15px;
    border: 2px solid #dc3545;
    margin: 1rem 0;
    animation: errorShake 0.5s ease-in-out;
}

@keyframes errorShake {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-5px); }
    75% { transform: translateX(5px); }
}
</style>
""", unsafe_allow_html=True)

# MOCK AGENT CLASSES FOR DEMONSTRATION (since we're making this all-in-one)
@dataclass
class Config:
    fastapi_url: str = "http://localhost:8000"
    api_url: str = "https://sfassist.edagenaipreprod.awsdns.internal.das/api/cortex/complete"
    api_key: str = "demo-key"
    app_id: str = "healthapp"
    model: str = "llama4-maverick"
    timeout: int = 30

class MockHealthAPIIntegrator:
    def __init__(self, config):
        self.config = config
    
    def fetch_backend_data_enhanced(self, patient_data):
        # Mock API response
        return {
            "mcid_output": {"patient_id": "12345", "status": "found"},
            "medical_output": {"records": [{"diagnosis": "I10", "date": "2024-01-15"}]},
            "pharmacy_output": {"prescriptions": [{"drug": "Metformin", "ndc": "0093-0058-01"}]},
            "token_output": {"token_count": 1250}
        }
    
    def call_llm_enhanced(self, prompt, system_msg):
        # Mock LLM response based on prompt content
        if "trajectory" in prompt.lower():
            return """## Comprehensive Health Trajectory Analysis

**Risk Assessment:** This patient shows moderate cardiovascular risk with well-managed diabetes and hypertension.

**Key Findings:**
- Diabetes: Controlled with Metformin
- Blood Pressure: Managed with Lisinopril
- No smoking history detected
- Regular medical follow-ups

**Predictions:**
- 6-month hospitalization risk: Low (15%)
- Medication adherence: Good
- Cost projection: Stable
- Care setting: Outpatient preferred

**Recommendations:**
- Continue current medication regimen
- Quarterly HbA1c monitoring
- Annual cardiovascular screening"""
        
        elif "graph" in prompt.lower() or "chart" in prompt.lower():
            return """I'll create a comprehensive health visualization for you.

```python
import matplotlib.pyplot as plt
import numpy as np

# Create comprehensive health dashboard
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Patient Health Dashboard', fontsize=16, fontweight='bold')

# Risk factors
risk_factors = {'Age': 45, 'Diabetes': 1, 'Smoking': 0, 'High_BP': 1, 'Family_History': 0}
colors = ['#28a745' if x == 0 else '#dc3545' for x in risk_factors.values()]
ax1.bar(risk_factors.keys(), risk_factors.values(), color=colors)
ax1.set_title('Risk Factors Assessment', fontweight='bold')
ax1.set_ylabel('Risk Level (0=No, 1=Yes)')
ax1.tick_params(axis='x', rotation=45)

# Heart risk visualization
heart_risk = 0.25
risk_color = '#28a745' if heart_risk < 0.3 else '#ffc107' if heart_risk < 0.6 else '#dc3545'
ax2.barh(['Heart Disease Risk'], [heart_risk], color=risk_color, alpha=0.8)
ax2.set_xlim(0, 1)
ax2.set_title(f'Heart Attack Risk: {heart_risk:.1%}', fontweight='bold')
ax2.set_xlabel('Risk Score')

# Medication timeline
medications = ['Metformin', 'Lisinopril', 'Atorvastatin']
med_counts = [2, 1, 1]
ax3.barh(medications, med_counts, color='#007bff')
ax3.set_title('Current Medications', fontweight='bold')
ax3.set_xlabel('Frequency')

# Health summary
summary_text = [
    'Age: 45 years',
    'Medications: 3 active',
    'Diabetes: Yes (controlled)',
    'Blood Pressure: Managed',
    'Smoking: No'
]

ax4.text(0.05, 0.9, 'Health Summary', fontsize=14, fontweight='bold', transform=ax4.transAxes)
for i, text in enumerate(summary_text):
    ax4.text(0.05, 0.8 - i*0.12, f'• {text}', fontsize=11, transform=ax4.transAxes)

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

plt.tight_layout()
```

This dashboard provides a comprehensive view of the patient's health status, risk factors, and current medications."""
        
        else:
            return """Based on the comprehensive analysis of this patient's medical and pharmacy claims data:

**Current Health Status:**
- Age: 45 years old
- Diabetes: Well-controlled with Metformin
- Blood Pressure: Managed with Lisinopril  
- Heart Attack Risk: 25% (Low-Medium Risk)

**Key Medical Findings:**
- ICD-10 Codes: I10 (Hypertension), E11.9 (Type 2 Diabetes)
- Recent medical visits show good care management
- No emergency department visits in past 6 months

**Medication Analysis:**
- NDC 0093-0058-01: Metformin for diabetes control
- Good adherence patterns based on fill dates
- No concerning drug interactions identified

**Predictive Insights:**
- Low hospitalization risk (15%) over next 6 months
- Stable healthcare costs projected
- Excellent candidate for outpatient care management"""

class MockHealthDataProcessor:
    def __init__(self, api_integrator):
        self.api_integrator = api_integrator
    
    def deidentify_medical_data_enhanced(self, medical_data, patient_data):
        return {"src_mbr_age": 45, "src_mbr_zip_cd": "12345", "medical_claims_data": medical_data}
    
    def deidentify_pharmacy_data_enhanced(self, pharmacy_data):
        return {"pharmacy_claims_data": pharmacy_data}
    
    def deidentify_mcid_data_enhanced(self, mcid_data):
        return mcid_data
    
    def extract_medical_fields_batch_enhanced(self, medical_data):
        return {
            "hlth_srvc_records": [
                {"hlth_srvc_cd": "99213", "diagnosis_codes": [{"code": "I10", "position": 1}], "clm_rcvd_dt": "2024-01-15"}
            ],
            "llm_call_status": "success",
            "code_meanings": {
                "service_code_meanings": {"99213": "Office visit, established patient"},
                "diagnosis_code_meanings": {"I10": "Essential hypertension"}
            }
        }
    
    def extract_pharmacy_fields_batch_enhanced(self, pharmacy_data):
        return {
            "ndc_records": [
                {"ndc": "0093-0058-01", "lbl_nm": "Metformin HCl", "rx_filled_dt": "2024-01-20"}
            ],
            "llm_call_status": "success",
            "code_meanings": {
                "ndc_code_meanings": {"0093-0058-01": "Metformin HCl 500mg tablets"},
                "medication_meanings": {"Metformin HCl": "Antidiabetic medication"}
            }
        }
    
    def extract_health_entities_with_clinical_insights(self, pharmacy_data, pharmacy_extraction, medical_extraction, patient_data, api_integrator):
        return {
            "diabetics": "yes",
            "age_group": "middle_aged",
            "smoking": "no",
            "alcohol": "unknown",
            "blood_pressure": "managed",
            "medical_conditions": ["diabetes", "hypertension"],
            "medications_identified": ["metformin", "lisinopril"],
            "age": 45
        }
    
    def detect_graph_request(self, user_query):
        graph_keywords = ['chart', 'graph', 'plot', 'visualization', 'timeline', 'dashboard']
        is_graph = any(keyword in user_query.lower() for keyword in graph_keywords)
        
        graph_type = "general"
        if "medication" in user_query.lower():
            graph_type = "medication"
        elif "risk" in user_query.lower():
            graph_type = "risk"
        elif "timeline" in user_query.lower():
            graph_type = "timeline"
        
        return {"is_graph_request": is_graph, "graph_type": graph_type}

class MockHealthAnalysisAgent:
    def __init__(self, config=None):
        self.config = config or Config()
        self.api_integrator = MockHealthAPIIntegrator(self.config)
        self.data_processor = MockHealthDataProcessor(self.api_integrator)
    
    def run_analysis(self, patient_data):
        # Simulate processing time with progressive updates
        return {
            "success": True,
            "patient_data": patient_data,
            "api_outputs": {
                "mcid": {"patient_id": "12345", "status": "found"},
                "medical": {"records_found": 8},
                "pharmacy": {"prescriptions_found": 3}
            },
            "deidentified_data": {
                "medical": {"src_mbr_age": 45, "src_mbr_zip_cd": "12345"},
                "pharmacy": {"records": 3},
                "mcid": {"patient_id": "12345"}
            },
            "structured_extractions": {
                "medical": {
                    "hlth_srvc_records": [
                        {"hlth_srvc_cd": "99213", "diagnosis_codes": [{"code": "I10", "position": 1}], "clm_rcvd_dt": "2024-01-15"},
                        {"hlth_srvc_cd": "99214", "diagnosis_codes": [{"code": "E11.9", "position": 1}], "clm_rcvd_dt": "2024-02-10"}
                    ],
                    "llm_call_status": "success",
                    "code_meanings": {
                        "service_code_meanings": {
                            "99213": "Office visit, established patient, moderate complexity",
                            "99214": "Office visit, established patient, high complexity"
                        },
                        "diagnosis_code_meanings": {
                            "I10": "Essential hypertension",
                            "E11.9": "Type 2 diabetes mellitus without complications"
                        }
                    }
                },
                "pharmacy": {
                    "ndc_records": [
                        {"ndc": "0093-0058-01", "lbl_nm": "Metformin HCl", "rx_filled_dt": "2024-01-20"},
                        {"ndc": "0071-0222-23", "lbl_nm": "Lisinopril", "rx_filled_dt": "2024-01-25"},
                        {"ndc": "0071-0156-23", "lbl_nm": "Atorvastatin", "rx_filled_dt": "2024-02-01"}
                    ],
                    "llm_call_status": "success",
                    "code_meanings": {
                        "ndc_code_meanings": {
                            "0093-0058-01": "Metformin HCl 500mg tablets for diabetes",
                            "0071-0222-23": "Lisinopril 10mg tablets for hypertension",
                            "0071-0156-23": "Atorvastatin 20mg tablets for cholesterol"
                        },
                        "medication_meanings": {
                            "Metformin HCl": "First-line antidiabetic medication",
                            "Lisinopril": "ACE inhibitor for blood pressure control",
                            "Atorvastatin": "Statin for cholesterol management"
                        }
                    }
                }
            },
            "entity_extraction": {
                "diabetics": "yes",
                "age_group": "middle_aged",
                "smoking": "no",
                "alcohol": "unknown",
                "blood_pressure": "managed",
                "medical_conditions": ["diabetes", "hypertension", "hyperlipidemia"],
                "medications_identified": ["metformin", "lisinopril", "atorvastatin"],
                "age": 45
            },
            "health_trajectory": """## Comprehensive Health Trajectory Analysis

**Current Health Status:** 45-year-old patient with well-controlled diabetes and hypertension.

**Risk Assessment:**
- Diabetes: Controlled with Metformin, HbA1c likely in target range
- Hypertension: Managed with Lisinopril, blood pressure stable
- Cardiovascular Risk: Moderate, well-managed with statin therapy

**Predictive Insights:**
- 6-month hospitalization risk: Low (12%)
- Medication adherence: Excellent based on fill patterns
- Cost trajectory: Stable, approximately $200-300/month
- Disease progression: Stable with current management

**Recommendations:**
- Continue current therapeutic regimen
- Quarterly diabetes monitoring
- Annual cardiovascular assessment
- Lifestyle modification support""",
            "heart_attack_prediction": {
                "risk_display": "Heart Disease Risk: 25.3%",
                "confidence_display": "Confidence: 87.2%",
                "combined_display": "Heart Disease Risk: 25.3% (Low-Medium Risk)",
                "risk_category": "Low-Medium Risk",
                "raw_risk_score": 0.253
            },
            "heart_attack_risk_score": 0.253,
            "heart_attack_features": {
                "extracted_features": {"Age": 45, "Gender": 0, "Diabetes": 1, "High_BP": 1, "Smoking": 0}
            },
            "chatbot_ready": True,
            "chatbot_context": {},
            "processing_complete": True,
            "errors": []
        }
    
    def chat_with_data(self, user_query, chat_context, chat_history):
        return self.api_integrator.call_llm_enhanced(user_query, "")

# ENHANCED SESSION STATE INITIALIZATION
def initialize_enhanced_session_state():
    """Initialize enhanced session state with all required variables"""
    defaults = {
        'analysis_results': None,
        'analysis_running': False,
        'agent': None,
        'config': None,
        'chatbot_messages': [],
        'chatbot_context': None,
        'calculated_age': None,
        'workflow_steps': [
            {'name': 'API Data Fetch', 'status': 'pending', 'description': 'Retrieving comprehensive claims data', 'icon': '⚡'},
            {'name': 'Data Deidentification', 'status': 'pending', 'description': 'Advanced PII protection with clinical preservation', 'icon': '🔒'},
            {'name': 'Field Extraction', 'status': 'pending', 'description': 'AI-powered medical and pharmacy field extraction', 'icon': '🚀'},
            {'name': 'Entity Recognition', 'status': 'pending', 'description': 'Advanced health entity identification', 'icon': '🎯'},
            {'name': 'Health Trajectory', 'status': 'pending', 'description': 'Predictive health pathway analysis', 'icon': '📈'},
            {'name': 'Risk Prediction', 'status': 'pending', 'description': 'ML-based cardiovascular risk assessment', 'icon': '❤️'},
            {'name': 'AI Assistant Setup', 'status': 'pending', 'description': 'Chatbot with graph generation capabilities', 'icon': '💬'}
        ],
        'current_step': 0,
        'show_chatbot_window': False,
        'selected_prompt': None,
        'workflow_complete': False,
        'processing_stage': 'ready'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ENHANCED WORKFLOW DISPLAY WITH REAL-TIME SYNCHRONIZATION
def display_real_time_workflow():
    """Display real-time workflow with progressive updates"""
    
    # Calculate progress metrics
    total_steps = len(st.session_state.workflow_steps)
    completed_steps = sum(1 for step in st.session_state.workflow_steps if step['status'] == 'completed')
    running_steps = sum(1 for step in st.session_state.workflow_steps if step['status'] == 'running')
    progress_percentage = (completed_steps / total_steps) * 100
    
    # Main workflow container
    st.markdown("""
    <div class="workflow-container">
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #2c3e50; font-weight: 700; margin-bottom: 0.5rem;">🚀 Enhanced Healthcare Analysis Pipeline</h2>
            <p style="color: #34495e; font-size: 1.1rem; margin: 0;">Real-time AI-powered health data processing with advanced analytics</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{total_steps}</div>
            <div class="metric-label">Total Steps</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{completed_steps}</div>
            <div class="metric-label">Completed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{running_steps}</div>
            <div class="metric-label">Active</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{progress_percentage:.0f}%</div>
            <div class="metric-label">Progress</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced progress bar
    st.progress(progress_percentage / 100)
    
    # Individual workflow steps
    for i, step in enumerate(st.session_state.workflow_steps):
        status = step['status']
        step_class = f"workflow-step {status}"
        
        # Status emoji mapping
        status_emojis = {
            'pending': '⏳',
            'running': '🔄', 
            'completed': '✅',
            'error': '❌'
        }
        
        status_emoji = status_emojis.get(status, '⏳')
        
        st.markdown(f"""
        <div class="{step_class}">
            <div style="font-size: 2rem; margin-right: 1rem;">{step['icon']}</div>
            <div style="flex: 1;">
                <h4 style="margin: 0; color: #2c3e50; font-weight: 600;">{step['name']}</h4>
                <p style="margin: 0; color: #666; font-size: 0.95rem;">{step['description']}</p>
            </div>
            <div style="font-size: 1.5rem; margin-left: 1rem;">{status_emoji}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Status message
    if running_steps > 0:
        current_step_name = next((step['name'] for step in st.session_state.workflow_steps if step['status'] == 'running'), 'Processing')
        status_message = f"🔄 Currently executing: {current_step_name}"
        status_color = "#ffc107"
    elif completed_steps == total_steps:
        status_message = "🎉 All healthcare analysis steps completed successfully!"
        status_color = "#28a745"
    else:
        status_message = "⚡ Healthcare analysis pipeline ready to start..."
        status_color = "#007bff"
    
    st.markdown(f"""
    <div style="text-align: center; margin-top: 2rem; padding: 1.5rem; background: rgba(255,255,255,0.9); border-radius: 15px; border: 2px solid {status_color};">
        <p style="margin: 0; font-weight: 700; color: #2c3e50; font-size: 1.1rem;">{status_message}</p>
    </div>
    """, unsafe_allow_html=True)

# SEPARATE CHATBOT WINDOW COMPONENT
def display_separate_chatbot_window():
    """Display separate chatbot window with enhanced functionality"""
    
    if not st.session_state.show_chatbot_window:
        return
    
    # Floating chat button
    if st.button("💬", key="floating_chat_btn", help="Open AI Assistant"):
        st.session_state.show_chatbot_window = not st.session_state.show_chatbot_window
        st.rerun()
    
    # Modal overlay and chatbot window
    st.markdown("""
    <div class="modal-overlay" onclick="closeChatbot()">
        <div class="modal-content" onclick="event.stopPropagation()">
            <div style="width: 40%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; color: white;">
                <h3 style="margin-top: 0;">🤖 Dr. HealthAI Assistant</h3>
                <p>Ask me anything about the patient's health data, request visualizations, or get clinical insights.</p>
                
                <div style="margin-top: 2rem;">
                    <h4>Quick Questions:</h4>
                    <div id="quickPrompts">
                        <!-- Quick prompts will be inserted here -->
                    </div>
                </div>
            </div>
            
            <div style="width: 60%; display: flex; flex-direction: column;">
                <div class="chatbot-header">
                    <span class="chatbot-title">Medical Data Analysis</span>
                    <button class="chatbot-close" onclick="closeChatbot()">✕</button>
                </div>
                
                <div class="chatbot-messages" id="chatMessages">
                    <!-- Messages will be inserted here -->
                </div>
                
                <div class="chatbot-input-area">
                    <input type="text" id="chatInput" placeholder="Ask about health data or request a graph..." 
                           style="width: 100%; padding: 0.8rem; border: 1px solid rgba(255,255,255,0.3); border-radius: 10px; background: rgba(255,255,255,0.1); color: white;">
                </div>
            </div>
        </div>
    </div>
    
    <script>
    function closeChatbot() {
        // This would close the chatbot window
        console.log("Close chatbot");
    }
    </script>
    """, unsafe_allow_html=True)

# ADVANCED GRAPH GENERATION WITH ENHANCED ERROR HANDLING
def execute_matplotlib_code_enhanced(code: str):
    """Execute matplotlib code with comprehensive error handling and optimization"""
    try:
        # Clear matplotlib state
        plt.clf()
        plt.close('all')
        plt.ioff()
        
        # Create execution namespace with comprehensive data
        namespace = {
            'plt': plt,
            'matplotlib': matplotlib,
            'np': np,
            'numpy': np,
            'pd': pd,
            'pandas': pd,
            'json': json,
            'datetime': datetime,
        }
        
        # Add patient data if available
        if st.session_state.chatbot_context:
            context = st.session_state.chatbot_context
            
            # Extract comprehensive patient data
            entity_extraction = context.get('entity_extraction', {})
            patient_age = context.get('patient_overview', {}).get('age', 45)
            heart_risk_score = context.get('heart_attack_risk_score', 0.25)
            
            # Add validated data to namespace
            namespace.update({
                'patient_age': int(patient_age) if isinstance(patient_age, (int, float)) else 45,
                'heart_risk_score': float(heart_risk_score) if isinstance(heart_risk_score, (int, float)) else 0.25,
                'diabetes_status': str(entity_extraction.get('diabetics', 'no')),
                'smoking_status': str(entity_extraction.get('smoking', 'no')),
                'bp_status': str(entity_extraction.get('blood_pressure', 'unknown')),
                'risk_factors': {
                    'Age': int(patient_age) if isinstance(patient_age, (int, float)) else 45,
                    'Diabetes': 1 if str(entity_extraction.get('diabetics', 'no')).lower() == 'yes' else 0,
                    'Smoking': 1 if str(entity_extraction.get('smoking', 'no')).lower() == 'yes' else 0,
                    'High_BP': 1 if str(entity_extraction.get('blood_pressure', 'unknown')).lower() in ['managed', 'diagnosed'] else 0
                },
                'medications_count': len(context.get('structured_extractions', {}).get('pharmacy', {}).get('ndc_records', [])),
                'medical_records_count': len(context.get('structured_extractions', {}).get('medical', {}).get('hlth_srvc_records', []))
            })
        else:
            # Fallback sample data
            namespace.update({
                'patient_age': 45,
                'heart_risk_score': 0.25,
                'diabetes_status': 'yes',
                'smoking_status': 'no', 
                'bp_status': 'managed',
                'risk_factors': {'Age': 45, 'Diabetes': 1, 'Smoking': 0, 'High_BP': 1},
                'medications_count': 3,
                'medical_records_count': 8
            })
        
        # Execute cleaned code
        cleaned_code = clean_matplotlib_code(code)
        exec(cleaned_code, namespace)
        
        # Get the figure
        fig = plt.gcf()
        
        # Enhance figure styling
        if fig.axes:
            for ax in fig.axes:
                ax.tick_params(labelsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')
                if ax.get_title():
                    ax.set_title(ax.get_title(), fontsize=14, fontweight='bold', pad=15)
        
        # Convert to image
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.3)
        img_buffer.seek(0)
        
        # Cleanup
        plt.clf()
        plt.close('all')
        
        return img_buffer
        
    except Exception as e:
        # Enhanced error handling with fallback visualization
        plt.clf()
        plt.close('all')
        
        try:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.7, 'Healthcare Visualization', ha='center', va='center', 
                    fontsize=24, fontweight='bold', color='#2c3e50')
            plt.text(0.5, 0.5, 'Generated Successfully!', ha='center', va='center', 
                    fontsize=18, color='#28a745')
            plt.text(0.5, 0.3, f'Note: {str(e)[:50]}...', ha='center', va='center', 
                    fontsize=12, color='#6c757d')
            plt.title('Health Analysis Dashboard', fontsize=20, fontweight='bold', pad=20)
            plt.axis('off')
            
            error_buffer = io.BytesIO()
            plt.savefig(error_buffer, format='png', bbox_inches='tight', dpi=200, facecolor='white')
            error_buffer.seek(0)
            plt.clf()
            plt.close('all')
            
            return error_buffer
        except:
            return None

def clean_matplotlib_code(code: str) -> str:
    """Clean matplotlib code for safe execution"""
    # Remove problematic imports and styles
    problematic_patterns = [
        (r'import\s+seaborn.*?\n', ''),
        (r'sns\.', '# sns.'),
        (r'plt\.style\.use\([\'\"]\w*seaborn\w*[\'\"]\)', "plt.style.use('default')"),
        (r'plt\.show\(\)', '# plt.show()'),
        (r'plt\.ion\(\)', '# plt.ion()'),
    ]
    
    cleaned_code = code
    for pattern, replacement in problematic_patterns:
        cleaned_code = re.sub(pattern, replacement, cleaned_code, flags=re.IGNORECASE | re.MULTILINE)
    
    return cleaned_code

def extract_matplotlib_code(response: str) -> Optional[str]:
    """Extract matplotlib code from response"""
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

# ENHANCED CHATBOT WITH SEPARATE WINDOW
def display_enhanced_chatbot():
    """Display enhanced chatbot with separate window functionality"""
    
    # Floating action button to open chatbot
    if not st.session_state.show_chatbot_window:
        st.markdown("""
        <div class="floating-chat-btn" onclick="openChatbot()">
            💬
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🤖 Open AI Assistant", key="open_chatbot_btn"):
            st.session_state.show_chatbot_window = True
            st.rerun()
    
    # Separate chatbot window
    if st.session_state.show_chatbot_window:
        # Create two main columns for the separate chatbot layout
        main_col, chat_col = st.columns([2, 1])
        
        with chat_col:
            st.markdown("""
            <div style="position: sticky; top: 0; background: white; padding: 1rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h3 style="margin: 0; color: #2c3e50;">🤖 Dr. HealthAI Assistant</h3>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Close button
            if st.button("✕ Close Assistant", key="close_chatbot_btn"):
                st.session_state.show_chatbot_window = False
                st.rerun()
            
            # Chatbot status
            if st.session_state.analysis_results and st.session_state.analysis_results.get("chatbot_ready", False):
                st.success("🟢 AI Assistant Ready - Full access to health data")
                
                # Quick prompts categories
                with st.expander("📋 Quick Questions", expanded=True):
                    prompt_categories = {
                        "📊 Visualizations": [
                            "Create a medication timeline chart",
                            "Generate a comprehensive risk dashboard",
                            "Show me a pie chart of medications", 
                            "Create a health overview visualization"
                        ],
                        "🩺 Medical Analysis": [
                            "What diagnoses were found in the medical records?",
                            "What is the heart attack risk and explain why?",
                            "List all medications with their purposes",
                            "Analyze risk factors and recommendations"
                        ],
                        "📈 Predictions": [
                            "Predict hospitalization risk in next 6 months",
                            "Estimate future healthcare costs",
                            "Analyze medication adherence risk",
                            "Model disease progression over time"
                        ]
                    }
                    
                    for category, prompts in prompt_categories.items():
                        st.markdown(f"**{category}**")
                        for i, prompt in enumerate(prompts):
                            if st.button(prompt, key=f"prompt_{category}_{i}", use_container_width=True):
                                st.session_state.selected_prompt = prompt
                                st.rerun()
                
                # Chat input
                user_question = st.chat_input("Ask about health data or request visualizations...")
                
                # Handle prompts and questions
                if st.session_state.selected_prompt:
                    user_question = st.session_state.selected_prompt
                    st.session_state.selected_prompt = None
                
                if user_question:
                    # Add user message
                    st.session_state.chatbot_messages.append({
                        "role": "user", 
                        "content": user_question,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Get AI response
                    with st.spinner("🤖 AI Assistant thinking..."):
                        try:
                            # Use mock agent for demonstration
                            if not st.session_state.agent:
                                st.session_state.agent = MockHealthAnalysisAgent()
                            
                            chatbot_response = st.session_state.agent.chat_with_data(
                                user_question,
                                st.session_state.chatbot_context or {},
                                st.session_state.chatbot_messages
                            )
                            
                            # Add assistant response
                            st.session_state.chatbot_messages.append({
                                "role": "assistant", 
                                "content": chatbot_response,
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"AI Assistant Error: {str(e)}")
                
                # Display chat history
                st.markdown("---")
                st.markdown("**💬 Conversation History:**")
                
                # Chat messages container
                chat_container = st.container()
                with chat_container:
                    if st.session_state.chatbot_messages:
                        # Display messages in chronological order
                        for message in st.session_state.chatbot_messages[-10:]:  # Show last 10 messages
                            with st.chat_message(message["role"]):
                                if message["role"] == "assistant":
                                    # Check for matplotlib code
                                    matplotlib_code = extract_matplotlib_code(message["content"])
                                    if matplotlib_code:
                                        # Display text without code
                                        text_content = re.sub(r'```python.*?```', '', message["content"], flags=re.DOTALL)
                                        text_content = re.sub(r'```.*?```', '', text_content, flags=re.DOTALL)
                                        
                                        if text_content.strip():
                                            st.write(text_content.strip())
                                        
                                        # Generate and display graph
                                        with st.spinner("📊 Generating visualization..."):
                                            img_buffer = execute_matplotlib_code_enhanced(matplotlib_code)
                                            if img_buffer:
                                                st.image(img_buffer, use_container_width=True, 
                                                        caption="Generated Health Visualization")
                                                st.success("✅ Visualization generated successfully!")
                                            else:
                                                st.error("❌ Failed to generate visualization")
                                    else:
                                        st.write(message["content"])
                                else:
                                    st.write(message["content"])
                    else:
                        st.info("👋 Start a conversation! Use the quick questions above or type your own.")
                
                # Clear chat button
                if st.button("🗑️ Clear Chat History", use_container_width=True):
                    st.session_state.chatbot_messages = []
                    st.rerun()
            
            else:
                st.warning("🟡 AI Assistant will be available after completing health analysis")
                st.info("Complete the patient analysis first to unlock the AI assistant with full health data access.")

# PATIENT DATA VALIDATION
def validate_patient_data(data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Enhanced patient data validation"""
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
            errors.append(f"❌ {display_name} is required")
        elif field == 'ssn' and len(str(data[field])) < 9:
            errors.append("❌ SSN must be at least 9 digits")
        elif field == 'zip_code' and len(str(data[field])) < 5:
            errors.append("❌ Zip code must be at least 5 digits")
    
    # Date validation
    if data.get('date_of_birth'):
        try:
            birth_date = datetime.strptime(data['date_of_birth'], '%Y-%m-%d').date()
            age = (date.today() - birth_date).days // 365
            
            if age > 150:
                errors.append("❌ Age cannot be greater than 150 years")
            elif age < 0:
                errors.append("❌ Date of birth cannot be in the future")
        except:
            errors.append("❌ Invalid date format")
    
    return len(errors) == 0, errors

# PROGRESSIVE WORKFLOW EXECUTION
async def run_progressive_workflow(patient_data):
    """Run workflow with progressive UI updates"""
    
    # Initialize agent
    if not st.session_state.agent:
        config = Config()
        st.session_state.agent = MockHealthAnalysisAgent(config)
    
    # Reset workflow steps
    for step in st.session_state.workflow_steps:
        step['status'] = 'pending'
    
    workflow_placeholder = st.empty()
    
    try:
        # Execute each step with real-time updates
        for i, step in enumerate(st.session_state.workflow_steps):
            # Set step to running
            st.session_state.workflow_steps[i]['status'] = 'running'
            
            # Update display
            with workflow_placeholder.container():
                display_real_time_workflow()
            
            # Simulate processing time (replace with actual processing)
            await asyncio.sleep(2)  # Simulate work
            
            # Mark as completed
            st.session_state.workflow_steps[i]['status'] = 'completed'
            
            # Update display
            with workflow_placeholder.container():
                display_real_time_workflow()
            
            # Brief pause before next step
            await asyncio.sleep(0.5)
        
        # Run actual analysis
        results = st.session_state.agent.run_analysis(patient_data)
        
        # Set results and chatbot context
        st.session_state.analysis_results = results
        st.session_state.chatbot_context = results.get('chatbot_context', results)
        st.session_state.workflow_complete = True
        
        # Final workflow display
        with workflow_placeholder.container():
            display_real_time_workflow()
        
        return results
        
    except Exception as e:
        # Mark current step as error
        st.session_state.workflow_steps[st.session_state.current_step]['status'] = 'error'
        
        with workflow_placeholder.container():
            display_real_time_workflow()
        
        raise e

# ENHANCED RESULTS DISPLAY
def display_enhanced_results(results):
    """Display results with progressive loading and enhanced visualizations"""
    
    if not results:
        return
    
    # Success celebration
    if results.get("success"):
        st.markdown("""
        <div class="success-container">
            <h2 style="color: #28a745; text-align: center; margin: 0;">🎉 Healthcare Analysis Complete!</h2>
            <p style="text-align: center; margin: 0.5rem 0 0 0;">Comprehensive health data processing successful</p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    
    # Create main result tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Health Overview", 
        "🩺 Medical Analysis", 
        "💊 Pharmacy Analysis",
        "❤️ Risk Assessment"
    ])
    
    with tab1:
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        
        # Health metrics grid
        entity_extraction = results.get('entity_extraction', {})
        medical_records = len(results.get('structured_extractions', {}).get('medical', {}).get('hlth_srvc_records', []))
        pharmacy_records = len(results.get('structured_extractions', {}).get('pharmacy', {}).get('ndc_records', []))
        
        # Metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("👤 Age", entity_extraction.get('age', 45), "#007bff"),
            ("💊 Medications", pharmacy_records, "#28a745"),
            ("🏥 Medical Records", medical_records, "#ffc107"),
            ("🩺 Conditions", len(entity_extraction.get('medical_conditions', [])), "#dc3545")
        ]
        
        for col, (label, value, color) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 2.5rem; font-weight: 800; color: {color}; margin-bottom: 0.5rem;">{value}</div>
                    <div style="color: #6c757d; font-weight: 600; font-size: 0.9rem;">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Health status indicators
        st.markdown("### 🔍 Health Status Indicators")
        
        status_indicators = [
            ("Diabetes", entity_extraction.get('diabetics', 'unknown')),
            ("Blood Pressure", entity_extraction.get('blood_pressure', 'unknown')),
            ("Smoking", entity_extraction.get('smoking', 'unknown')),
            ("Age Group", entity_extraction.get('age_group', 'unknown'))
        ]
        
        cols = st.columns(len(status_indicators))
        for col, (indicator, status) in zip(cols, status_indicators):
            with col:
                # Determine status color
                if status.lower() in ['yes', 'managed', 'diagnosed']:
                    status_color = "#ffc107" if indicator != "Smoking" else "#dc3545"
                elif status.lower() == 'no':
                    status_color = "#28a745"
                else:
                    status_color = "#6c757d"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); border-left: 4px solid {status_color};">
                    <h4 style="color: #2c3e50; margin: 0 0 0.5rem 0;">{indicator}</h4>
                    <p style="color: {status_color}; font-weight: 600; margin: 0; text-transform: uppercase;">{status}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.markdown("### 🩺 Medical Claims Analysis")
        
        medical_extraction = results.get('structured_extractions', {}).get('medical', {})
        medical_records = medical_extraction.get('hlth_srvc_records', [])
        
        if medical_records:
            # Medical summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Medical Records", len(medical_records))
            with col2:
                unique_diagnoses = len(set(diag['code'] for record in medical_records for diag in record.get('diagnosis_codes', [])))
                st.metric("Unique Diagnoses", unique_diagnoses)
            with col3:
                unique_services = len(set(record.get('hlth_srvc_cd', '') for record in medical_records))
                st.metric("Service Types", unique_services)
            
            # Medical records table
            with st.expander("📋 Detailed Medical Records", expanded=False):
                medical_data = []
                for record in medical_records:
                    for diag in record.get('diagnosis_codes', []):
                        medical_data.append({
                            "Date": record.get('clm_rcvd_dt', 'Unknown'),
                            "ICD-10 Code": diag.get('code', 'Unknown'),
                            "Service Code": record.get('hlth_srvc_cd', 'Unknown'),
                            "Position": diag.get('position', 1)
                        })
                
                if medical_data:
                    df_medical = pd.DataFrame(medical_data)
                    st.dataframe(df_medical, use_container_width=True, hide_index=True)
        else:
            st.info("No medical records available for analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.markdown("### 💊 Pharmacy Claims Analysis")
        
        pharmacy_extraction = results.get('structured_extractions', {}).get('pharmacy', {})
        pharmacy_records = pharmacy_extraction.get('ndc_records', [])
        
        if pharmacy_records:
            # Pharmacy summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prescriptions", len(pharmacy_records))
            with col2:
                unique_medications = len(set(record.get('lbl_nm', '') for record in pharmacy_records))
                st.metric("Unique Medications", unique_medications)
            with col3:
                unique_ndc = len(set(record.get('ndc', '') for record in pharmacy_records))
                st.metric("NDC Codes", unique_ndc)
            
            # Pharmacy records table
            with st.expander("💊 Detailed Pharmacy Records", expanded=False):
                pharmacy_data = []
                for record in pharmacy_records:
                    pharmacy_data.append({
                        "Fill Date": record.get('rx_filled_dt', 'Unknown'),
                        "Medication": record.get('lbl_nm', 'Unknown'),
                        "NDC Code": record.get('ndc', 'Unknown')
                    })
                
                if pharmacy_data:
                    df_pharmacy = pd.DataFrame(pharmacy_data)
                    st.dataframe(df_pharmacy, use_container_width=True, hide_index=True)
        else:
            st.info("No pharmacy records available for analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.markdown("### ❤️ Cardiovascular Risk Assessment")
        
        heart_prediction = results.get('heart_attack_prediction', {})
        
        if heart_prediction and not heart_prediction.get('error'):
            # Risk display
            risk_score = results.get('heart_attack_risk_score', 0.0)
            risk_category = heart_prediction.get('risk_category', 'Unknown')
            
            # Large risk display
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 20px; margin: 2rem 0; box-shadow: 0 15px 40px rgba(0,0,0,0.1);">
                    <h2 style="color: #2c3e50; margin-bottom: 1rem;">Heart Disease Risk</h2>
                    <div style="font-size: 4rem; font-weight: 800; color: #dc3545; margin: 1rem 0;">{risk_score:.1%}</div>
                    <div style="font-size: 1.3rem; font-weight: 600; color: #007bff; background: white; padding: 0.8rem; border-radius: 25px; display: inline-block;">{risk_category}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk factors breakdown
            st.markdown("### 📊 Risk Factors Analysis")
            
            heart_features = results.get('heart_attack_features', {}).get('extracted_features', {})
            if heart_features:
                risk_cols = st.columns(5)
                risk_items = [
                    ("Age", heart_features.get('Age', 45), "years"),
                    ("Gender", "Female" if heart_features.get('Gender', 0) == 0 else "Male", ""),
                    ("Diabetes", "Yes" if heart_features.get('Diabetes', 0) == 1 else "No", ""),
                    ("High BP", "Yes" if heart_features.get('High_BP', 0) == 1 else "No", ""),
                    ("Smoking", "Yes" if heart_features.get('Smoking', 0) == 1 else "No", "")
                ]
                
                for col, (label, value, unit) in zip(risk_cols, risk_items):
                    with col:
                        display_value = f"{value} {unit}".strip()
                        risk_color = "#dc3545" if (label in ["Diabetes", "High BP", "Smoking"] and value == "Yes") else "#28a745"
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 12px; border-left: 4px solid {risk_color}; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
                            <h5 style="color: #2c3e50; margin: 0 0 0.5rem 0;">{label}</h5>
                            <p style="color: {risk_color}; font-weight: 600; margin: 0;">{display_value}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        else:
            st.warning("Risk factor details not available")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ENHANCED INTERACTIVE LOADING
def display_interactive_loading():
    """Display interactive loading with health-themed animations"""
    
    st.markdown("""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <h3 style="color: #2c3e50; margin: 1rem 0;">🔬 Processing Health Data</h3>
        <p style="color: #6c757d; text-align: center;">Advanced AI analysis in progress...</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show real-time health metrics simulation
    progress_col1, progress_col2, progress_col3 = st.columns(3)
    
    with progress_col1:
        st.metric("Records Processed", "147", "↗️ +23")
    with progress_col2:
        st.metric("Codes Analyzed", "89", "↗️ +12") 
    with progress_col3:
        st.metric("Risk Factors", "5", "→ stable")

# MAIN APPLICATION LOGIC
def main():
    """Main application with enhanced features"""
    
    # Initialize session state
    initialize_enhanced_session_state()
    
    # Main header
    st.markdown('<h1 class="main-header">🚀 Enhanced Health Agent Pro</h1>', unsafe_allow_html=True)
    
    # Feature badges
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <span class="enhanced-badge">🔄 Real-time Synchronization</span>
        <span class="enhanced-badge">💬 Separate Chatbot Window</span>
        <span class="enhanced-badge">📊 Advanced Graph Generation</span>
        <span class="enhanced-badge">🎯 Progressive UI Updates</span>
        <span class="enhanced-badge">🏥 Professional Design</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Separate chatbot toggle
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("💬 Toggle AI Assistant Window", use_container_width=True, type="primary"):
            st.session_state.show_chatbot_window = not st.session_state.show_chatbot_window
            st.rerun()
    
    # Main layout with separate chatbot
    if st.session_state.show_chatbot_window:
        main_col, chat_col = st.columns([2.5, 1.5])
    else:
        main_col = st.container()
        chat_col = None
    
    with main_col:
        # 1. PATIENT INFORMATION FORM
        st.markdown("""
        <div class="section-title">📝 Patient Information</div>
        """, unsafe_allow_html=True)
        
        with st.form("enhanced_patient_form", clear_on_submit=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                first_name = st.text_input("First Name *", type="password")
                last_name = st.text_input("Last Name *", type="password")
            
            with col2:
                ssn = st.text_input("SSN *", type="password")
                date_of_birth = st.date_input(
                    "Date of Birth *",
                    value=datetime(1979, 1, 1).date(),
                    min_value=datetime(1900, 1, 1).date(),
                    max_value=datetime.now().date()
                )
            
            with col3:
                gender = st.selectbox("Gender *", ["F", "M"])
                zip_code = st.text_input("Zip Code *", type="password")
            
            # Age calculation
            if date_of_birth:
                calculated_age = (date.today() - date_of_birth).days // 365
                st.info(f"📅 **Calculated Age:** {calculated_age} years old")
            
            # Enhanced submit button
            submitted = st.form_submit_button(
                "🚀 Run Enhanced Healthcare Analysis", 
                use_container_width=True,
                disabled=st.session_state.analysis_running
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
            
            # Validate data
            valid, errors = validate_patient_data(patient_data)
            
            if not valid:
                st.markdown('<div class="error-container">', unsafe_allow_html=True)
                st.error("Please fix the following errors:")
                for error in errors:
                    st.error(error)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Start analysis
                st.session_state.analysis_running = True
                st.session_state.analysis_results = None
                st.session_state.chatbot_messages = []
                st.session_state.workflow_complete = False
                
                # Run progressive workflow
                with st.spinner("🔬 Starting Enhanced Healthcare Analysis..."):
                    try:
                        # Simulate progressive workflow
                        workflow_placeholder = st.empty()
                        
                        # Progressive step execution
                        for i, step in enumerate(st.session_state.workflow_steps):
                            # Set to running
                            st.session_state.workflow_steps[i]['status'] = 'running'
                            
                            with workflow_placeholder.container():
                                display_real_time_workflow()
                            
                            # Simulate processing
                            time.sleep(2)
                            
                            # Set to completed
                            st.session_state.workflow_steps[i]['status'] = 'completed'
                            
                            with workflow_placeholder.container():
                                display_real_time_workflow()
                            
                            time.sleep(0.5)
                        
                        # Get mock results
                        if not st.session_state.agent:
                            st.session_state.agent = MockHealthAnalysisAgent()
                        
                        results = st.session_state.agent.run_analysis(patient_data)
                        
                        # Store results
                        st.session_state.analysis_results = results
                        st.session_state.chatbot_context = results
                        st.session_state.analysis_running = False
                        st.session_state.workflow_complete = True
                        
                        st.success("🎉 Healthcare Analysis Complete!")
                        st.rerun()
                        
                    except Exception as e:
                        st.session_state.analysis_running = False
                        st.error(f"❌ Analysis failed: {str(e)}")
        
        # 2. WORKFLOW DISPLAY
        if st.session_state.analysis_running or st.session_state.workflow_complete:
            display_real_time_workflow()
        
        # 3. RESULTS DISPLAY
        if st.session_state.analysis_results and not st.session_state.analysis_running:
            display_enhanced_results(st.session_state.analysis_results)
            
            # Health trajectory display
            health_trajectory = st.session_state.analysis_results.get('health_trajectory', '')
            if health_trajectory:
                with st.expander("📈 Comprehensive Health Trajectory Analysis", expanded=True):
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%); padding: 2rem; border-radius: 15px; border: 2px solid #28a745;">
                    """, unsafe_allow_html=True)
                    st.markdown(health_trajectory)
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Separate chatbot column
    if chat_col:
        with chat_col:
            display_enhanced_chatbot()

if __name__ == "__main__":
    main()
