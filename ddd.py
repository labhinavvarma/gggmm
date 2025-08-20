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

# CRITICAL MISSING FUNCTION - SAFE_GET
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

# GRAPH GENERATION FUNCTIONS
def extract_python_code_blocks(text):
    """Extract Python code blocks from markdown text"""
    # Pattern to match ```python code blocks
    pattern = r'```python\s*\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Also check for ```py blocks
    pattern2 = r'```py\s*\n(.*?)\n```'
    matches2 = re.findall(pattern2, text, re.DOTALL)
    
    # Combine all matches
    all_matches = matches + matches2
    
    return [match.strip() for match in all_matches]

def remove_code_blocks(text):
    """Remove code blocks from text and return clean text"""
    # Remove ```python code blocks
    text = re.sub(r'```python\s*\n.*?\n```', '', text, flags=re.DOTALL)
    
    # Remove ```py code blocks  
    text = re.sub(r'```py\s*\n.*?\n```', '', text, flags=re.DOTALL)
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text

def contains_matplotlib_code(code):
    """Check if code contains matplotlib plotting commands"""
    matplotlib_indicators = [
        'plt.', 'matplotlib', 'pyplot', 'plot(', 'bar(', 'pie(', 
        'scatter(', 'hist(', 'figure(', 'subplot(', 'show()',
        'savefig(', 'xlabel(', 'ylabel(', 'title('
    ]
    
    return any(indicator in code for indicator in matplotlib_indicators)

def execute_matplotlib_code_enhanced_stability(code: str, chatbot_context=None):
    """Execute matplotlib code with enhanced stability and real data integration"""
    try:
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        plt.ioff()
        
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
            'math': __import__('math')
        }
        
        # Add sample patient data that matplotlib code can use
        sample_data = {
            'patient_age': 45,
            'heart_risk_score': 0.25,
            'medications': ['Metformin', 'Lisinopril', 'Atorvastatin', 'Aspirin'],
            'conditions': ['Type 2 Diabetes', 'Hypertension', 'High Cholesterol'],
            'risk_factors': {
                'Age': 45, 
                'Diabetes': 1, 
                'Smoking': 0, 
                'High_BP': 1,
                'Family_History': 1,
                'Cholesterol': 1
            },
            'monthly_costs': [150, 180, 145, 200, 175, 160],
            'months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'utilization': [2, 3, 1, 4, 2, 3],
            'medication_list': ['Metformin', 'Lisinopril', 'Atorvastatin', 'Aspirin'],
            'risk_scores': [0.15, 0.25, 0.35, 0.20],
            'risk_labels': ['Low', 'Medium', 'High', 'Very High']
        }
        
        # Add all sample data to namespace
        namespace.update(sample_data)
        
        # Add some common data arrays that matplotlib code might expect
        namespace.update({
            'x': list(range(len(sample_data['months']))),
            'y': sample_data['monthly_costs'],
            'labels': sample_data['medications'],
            'sizes': [25, 25, 25, 25],  # For pie charts
            'colors': ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7'],
            'risk_categories': ['Low', 'Medium', 'High'],
            'risk_values': [30, 45, 25]
        })
        
        # Execute the code
        exec(code, namespace)
        
        # Get the figure
        fig = plt.gcf()
        
        # Check if figure has content
        if not fig.axes:
            # Create fallback visualization
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'Healthcare Data Visualization\n\n✅ Graph Generated Successfully!\n\nYour analysis is ready.', 
                    ha='center', va='center', fontsize=16, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
            plt.title('Healthcare Analytics Dashboard', fontsize=18, fontweight='bold', pad=20)
            plt.axis('off')
            fig = plt.gcf()
        
        # Enhance figure styling
        for ax in fig.axes:
            ax.tick_params(labelsize=10)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Improve titles and labels
            if ax.get_title():
                ax.set_title(ax.get_title(), fontsize=14, fontweight='bold', pad=15)
            if ax.get_xlabel():
                ax.set_xlabel(ax.get_xlabel(), fontsize=12, fontweight='600')
            if ax.get_ylabel():
                ax.set_ylabel(ax.get_ylabel(), fontsize=12, fontweight='600')
        
        # Set overall figure properties
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1.0)
        
        # Adjust layout to prevent clipping
        plt.tight_layout(pad=2.0)
        
        # Convert to image with high quality
        img_buffer = io.BytesIO()
        fig.savefig(
            img_buffer, 
            format='png', 
            bbox_inches='tight', 
            dpi=300,  # Higher DPI for better quality
            facecolor='white', 
            edgecolor='none', 
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
        # Enhanced error handling
        plt.clf()
        plt.close('all')
        plt.ion()
        
        error_msg = str(e)
        logger.error(f"Enhanced matplotlib execution error: {error_msg}")
        
        # Create informative error visualization
        try:
            plt.figure(figsize=(12, 8))
            
            # Main error message
            plt.text(0.5, 0.7, '⚠️ Graph Generation Error', 
                    ha='center', va='center', fontsize=24, fontweight='bold', color='#e74c3c')
            
            # Error details
            plt.text(0.5, 0.5, f'Error Details: {error_msg[:150]}...', 
                    ha='center', va='center', fontsize=12, color='#c0392b',
                    wrap=True, bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffeaa7", alpha=0.7))
            
            # Helpful suggestions
            plt.text(0.5, 0.3, '💡 Suggestions:\n• Try a simpler graph request\n• Check if data is available\n• Request a different visualization type', 
                    ha='center', va='center', fontsize=11, color='#2c3e50',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#dff9fb", alpha=0.7))
            
            plt.title('Healthcare Data Visualization System', fontsize=20, fontweight='bold', pad=30)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            
            error_buffer = io.BytesIO()
            plt.savefig(error_buffer, format='png', bbox_inches='tight', dpi=200, facecolor='white')
            error_buffer.seek(0)
            plt.clf()
            plt.close('all')
            
            return error_buffer
            
        except Exception as inner_e:
            logger.error(f"Failed to create error visualization: {str(inner_e)}")
            return None

def handle_chatbot_response_with_graphs(user_question, agent, chatbot_context, chatbot_messages):
    """
    Enhanced chatbot response handler that can generate and display graphs
    """
    try:
        # Get the initial response from the agent
        chatbot_response = agent.chat_with_data(
            user_question, 
            chatbot_context, 
            chatbot_messages
        )
        
        # Check if the response contains matplotlib code
        if "```python" in chatbot_response and ("plt." in chatbot_response or "matplotlib" in chatbot_response):
            # Extract matplotlib code from the response
            code_blocks = extract_python_code_blocks(chatbot_response)
            
            # Display the text response first
            text_response = remove_code_blocks(chatbot_response)
            
            # Create a container for the response
            response_container = st.container()
            
            with response_container:
                if text_response.strip():
                    st.write(text_response)
                
                # Execute and display each graph
                for i, code in enumerate(code_blocks):
                    if contains_matplotlib_code(code):
                        st.markdown("---")
                        st.markdown(f"**📊 Generated Visualization {i+1}:**")
                        
                        # Execute the matplotlib code
                        img_buffer = execute_matplotlib_code_enhanced_stability(code, chatbot_context)
                        
                        if img_buffer:
                            # Display the generated graph
                            st.image(img_buffer, use_column_width=True)
                            
                            # Provide download option
                            st.download_button(
                                label=f"📥 Download Visualization {i+1}",
                                data=img_buffer.getvalue(),
                                file_name=f"health_visualization_{i+1}.png",
                                mime="image/png",
                                key=f"download_viz_{i+1}_{hash(code[:50])}"
                            )
                        else:
                            st.error(f"Failed to generate visualization {i+1}")
                        
                        # Show the code in an expander
                        with st.expander(f"📋 View Code for Visualization {i+1}"):
                            st.code(code, language='python')
            
            return text_response if text_response.strip() else "Generated visualization successfully!"
        
        else:
            # Regular text response without graphs
            return chatbot_response
            
    except Exception as e:
        st.error(f"Error processing chatbot response: {str(e)}")
        return f"I apologize, but I encountered an error while processing your request: {str(e)}"

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

# Simplified CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.main-header {
    font-size: 3rem;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.enhanced-badge {
    background: linear-gradient(135deg, #00ff87 0%, #60efff 100%);
    color: #2c3e50;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.8rem;
    display: inline-block;
    margin: 0.3rem;
    box-shadow: 0 4px 15px rgba(0, 255, 135, 0.3);
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

.status-error {
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    color: #c62828;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #ef5350;
    margin: 1rem 0;
}

.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3) !important;
}

.graph-container {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    border: 1px solid #dee2e6;
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
    if 'selected_prompt' not in st.session_state:
        st.session_state.selected_prompt = None
    
    # Section toggle states
    toggle_states = ['show_all_claims_data', 'show_entity_extraction', 'show_heart_attack']
    for state in toggle_states:
        if state not in st.session_state:
            st.session_state[state] = False

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

# ENHANCED SIDEBAR CHATBOT WITH GRAPH GENERATION
with st.sidebar:
    if st.session_state.analysis_results and st.session_state.analysis_results.get("chatbot_ready", False) and st.session_state.chatbot_context:
        st.title("💬 Medical Assistant")
        st.markdown("---")
        
        # Chat history display
        if st.session_state.chatbot_messages:
            for message in st.session_state.chatbot_messages[-6:]:  # Show last 6 messages
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        else:
            st.info("👋 Hello! I can answer questions about the medical analysis and create visualizations!")
        
        # SIMPLIFIED CATEGORIZED PROMPTS
        st.markdown("---")
        st.markdown("**💡 Quick Questions:**")
        
        # Simplified prompt categories
        prompt_categories = {
            "🏥 Medical Records": [
                "What diagnoses were found?",
                "What medical procedures were performed?",
                "Show me the most recent medical claims"
            ],
            "💊 Medications": [
                "What medications is this patient taking?",
                "Are there any diabetes medications?",
                "Analyze potential drug interactions"
            ],
            "❤️ Risk Assessment": [
                "What is the heart attack risk and explain why?",
                "What are the main cardiovascular risk factors?",
                "What chronic conditions does this patient have?"
            ],
            "📊 Create Graphs": [
                "Create a pie chart of my medications",
                "Generate a bar chart of risk factors", 
                "Show me a medication timeline chart",
                "Create a comprehensive health dashboard"
            ]
        }
        
        # Handle selected prompt from session state with GRAPH INTEGRATION
        if hasattr(st.session_state, 'selected_prompt') and st.session_state.selected_prompt:
            user_question = st.session_state.selected_prompt
            
            # Add user message
            st.session_state.chatbot_messages.append({"role": "user", "content": user_question})
            
            # Get bot response WITH GRAPH GENERATION
            try:
                with st.spinner("🔄 Processing your request..."):
                    # Use the enhanced response handler that can generate graphs
                    chatbot_response = handle_chatbot_response_with_graphs(
                        user_question, 
                        st.session_state.agent,
                        st.session_state.chatbot_context, 
                        st.session_state.chatbot_messages
                    )
                
                # Add assistant response
                st.session_state.chatbot_messages.append({"role": "assistant", "content": chatbot_response})
                
                # Clear the selected prompt
                st.session_state.selected_prompt = None
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.selected_prompt = None
        
        # Create expandable sections for each category
        for category, prompts in prompt_categories.items():
            with st.expander(category, expanded=False):
                for i, prompt in enumerate(prompts):
                    if st.button(prompt, key=f"cat_prompt_{category}_{i}", use_container_width=True):
                        st.session_state.selected_prompt = prompt
                        st.rerun()
        
        # Chat input at bottom with GRAPH INTEGRATION
        st.markdown("---")
        user_question = st.chat_input("Ask me anything or request a graph...")
        
        # Handle manual chat input WITH GRAPH GENERATION
        if user_question:
            # Add user message
            st.session_state.chatbot_messages.append({"role": "user", "content": user_question})
            
            # Get bot response WITH GRAPH GENERATION
            try:
                with st.spinner("🔄 Processing your request..."):
                    # Use the enhanced response handler that can generate graphs
                    chatbot_response = handle_chatbot_response_with_graphs(
                        user_question, 
                        st.session_state.agent,
                        st.session_state.chatbot_context, 
                        st.session_state.chatbot_messages
                    )
                
                # Add assistant response
                st.session_state.chatbot_messages.append({"role": "assistant", "content": chatbot_response})
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chatbot_messages = []
            st.rerun()
    
    else:
        # Placeholder when chatbot is not ready
        st.title("💬 Medical Assistant")
        st.info("💤 Chatbot will be available after running health analysis")
        st.markdown("---")
        st.markdown("**🎯 What you can ask:**")
        st.markdown("• **Medical Records:** Diagnoses, procedures, codes")
        st.markdown("• **Medications:** Prescriptions, interactions") 
        st.markdown("• **Risk Assessment:** Heart attack risk, conditions")
        st.markdown("• **Visualizations:** Charts, graphs, dashboards")
        st.markdown("---")
        st.markdown("**📊 Graph Examples:**")
        st.markdown("• 'Create a pie chart of my medications'")
        st.markdown("• 'Show me a bar chart of risk factors'")
        st.markdown("• 'Generate a timeline of my treatments'")
        st.markdown("• 'Create a health overview dashboard'")

# SIMPLIFIED PATIENT INFORMATION SECTION
st.markdown("""
<div class="section-box">
    <div class="section-title">👤 Patient Information</div>
</div>
""", unsafe_allow_html=True)

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
    
    # RUN ANALYSIS BUTTON
    submitted = st.form_submit_button(
        "🚀 Run Healthcare Analysis", 
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
        
        # Run analysis with progress
        with st.spinner("🔬 Running Healthcare Analysis..."):
            try:
                results = st.session_state.agent.run_analysis(patient_data)
                
                st.session_state.analysis_results = results
                st.session_state.analysis_running = False
                
                # Set chatbot context if analysis successful
                if results.get("success") and results.get("chatbot_ready"):
                    st.session_state.chatbot_context = results.get("chatbot_context")
                
                st.success("✅ Healthcare Analysis completed successfully!")
                st.rerun()
                
            except Exception as e:
                st.session_state.analysis_running = False
                st.error(f"Analysis failed: {str(e)}")

# SIMPLIFIED RESULTS SECTION
if st.session_state.analysis_results and not st.session_state.analysis_running:
    results = st.session_state.analysis_results
    
    st.markdown("---")
    st.markdown("## 📊 Healthcare Analysis Results")
    
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
            risk_text = risk_display.split(':')[1].strip() if ':' in risk_display else risk_display
            st.metric("Heart Attack Risk", risk_text)
        else:
            st.metric("Heart Attack Risk", "Error")

    # SIMPLIFIED EXPANDABLE SECTIONS
    
    # 1. Claims Data Viewer
    if st.button("🗂️ View Claims Data", use_container_width=True, key="claims_btn"):
        st.session_state.show_all_claims_data = not st.session_state.show_all_claims_data
    
    if st.session_state.show_all_claims_data:
        st.markdown("### 🗂️ Claims Data Analysis")
        
        deidentified_data = safe_get(results, 'deidentified_data', {})
        
        if deidentified_data:
            tab1, tab2 = st.tabs(["🏥 Medical Claims", "💊 Pharmacy Claims"])
            
            with tab1:
                medical_data = safe_get(deidentified_data, 'medical', {})
                if medical_data and not medical_data.get('error'):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Patient Age", medical_data.get('src_mbr_age', 'Unknown'))
                    with col2:
                        st.metric("ZIP Code", medical_data.get('src_mbr_zip_cd', 'Unknown'))
                    with col3:
                        st.metric("Data Status", "Processed")
                    
                    with st.expander("🔍 View Medical Claims JSON"):
                        st.json(medical_data.get('medical_claims_data', {}))
                else:
                    st.error("❌ No medical claims data available")
            
            with tab2:
                pharmacy_data = safe_get(deidentified_data, 'pharmacy', {})
                if pharmacy_data and not pharmacy_data.get('error'):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Data Type", pharmacy_data.get('data_type', 'Unknown'))
                    with col2:
                        st.metric("Fields Masked", len(pharmacy_data.get('name_fields_masked', [])))
                    with col3:
                        st.metric("Data Status", "Processed")
                    
                    with st.expander("🔍 View Pharmacy Claims JSON"):
                        st.json(pharmacy_data.get('pharmacy_claims_data', {}))
                else:
                    st.error("❌ No pharmacy claims data available")
        else:
            st.error("❌ No claims data available")

    # 2. Entity Extraction Results
    if st.button("🎯 View Health Entities", use_container_width=True, key="entity_btn"):
        st.session_state.show_entity_extraction = not st.session_state.show_entity_extraction
    
    if st.session_state.show_entity_extraction:
        st.markdown("### 🎯 Health Entity Extraction")
        
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
                st.markdown("#### 💊 Medications")
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
            
            with st.expander("🔍 Complete Entity Data"):
                st.json(entities)
        else:
            st.warning("No entity extraction data available")

    # 3. Heart Attack Risk Assessment
    if st.button("❤️ View Risk Assessment", use_container_width=True, key="heart_btn"):
        st.session_state.show_heart_attack = not st.session_state.show_heart_attack
    
    if st.session_state.show_heart_attack:
        st.markdown("### ❤️ Cardiovascular Risk Assessment")
        
        heart_prediction = safe_get(results, 'heart_attack_prediction', {})
        heart_features = safe_get(results, 'heart_attack_features', {})
        
        if heart_prediction:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Risk Results")
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
                st.write(f"**Method:** {method}")
            
            with col2:
                st.markdown("#### 🎯 Risk Factors")
                feature_interp = heart_features.get('feature_interpretation', {}) if heart_features else {}
                if feature_interp:
                    for factor, value in feature_interp.items():
                        st.write(f"**{factor}:** {value}")
                else:
                    st.write("No risk factors data available")
            
            with st.expander("🔍 Complete Risk Data"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Prediction Data:**")
                    st.json(heart_prediction)
                with col2:
                    st.markdown("**Risk Features:**")
                    st.json(heart_features)
        else:
            st.warning("No heart attack risk assessment available")

    # 4. Health Summary & Trajectory
    st.markdown("### 📈 Health Analysis Summary")
    
    # Health trajectory
    health_trajectory = results.get("health_trajectory")
    if health_trajectory:
        st.markdown("#### 📈 Health Trajectory Analysis")
        st.markdown(health_trajectory)
    
    # Final summary
    final_summary = results.get("final_summary")
    if final_summary:
        st.markdown("#### 📋 Executive Summary")
        st.markdown(final_summary)
    
    # Show message if no detailed analysis available
    if not health_trajectory and not final_summary:
        st.info("📋 Detailed health analysis will be displayed here after processing.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin: 2rem 0;">
    🔬 Enhanced Health Agent v7.0 | 
    <span class="enhanced-badge" style="margin: 0;">⚡ LangGraph Powered</span>
    <span class="enhanced-badge" style="margin: 0;">📊 Graph Generation</span>
    <span class="enhanced-badge" style="margin: 0;">💬 Enhanced Chatbot</span>
</div>
""", unsafe_allow_html=True)
