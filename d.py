# Configure Streamlit page FIRST - before any other Streamlit commands
import streamlit as st

# Determine sidebar state based on chatbot readiness
if 'analysis_results' in st.session_state and st.session_state.get('analysis_results') and st.session_state.analysis_results.get("chatbot_ready", False):
    sidebar_state = "expanded"
else:
    sidebar_state = "collapsed"

st.set_page_config(
    page_title="Health Analysis Agent",
    page_icon="🏥",
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
from typing import Dict, Any, Optional
import asyncio

# Add current directory to path for importing the agent
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

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

# Enhanced Custom CSS for clean layout and advanced chatbot
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 600;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
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

/* Advanced Chatbot Styles */
.chatbot-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px 10px 0 0;
    margin: -1rem -1rem 0 -1rem;
    text-align: center;
    font-weight: 600;
    font-size: 1.1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.chatbot-status {
    background: rgba(255,255,255,0.1);
    padding: 0.5rem;
    border-radius: 5px;
    margin-top: 0.5rem;
    font-size: 0.85rem;
    text-align: center;
}

.chat-container {
    height: 400px;
    overflow-y: auto;
    padding: 1rem 0;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    background: #fafafa;
    margin: 1rem 0;
}

.chat-message {
    margin: 0.8rem 0;
    padding: 0;
    display: flex;
    flex-direction: column;
}

.user-message {
    align-items: flex-end;
}

.assistant-message {
    align-items: flex-start;
}

.message-bubble {
    max-width: 85%;
    padding: 0.8rem 1rem;
    border-radius: 18px;
    word-wrap: break-word;
    position: relative;
    margin: 0.2rem 0;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.user-bubble {
    background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
    color: white;
    border-bottom-right-radius: 4px;
    margin-left: auto;
}

.assistant-bubble {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    color: #2c3e50;
    border: 1px solid #e9ecef;
    border-bottom-left-radius: 4px;
    margin-right: auto;
}

.message-timestamp {
    font-size: 0.7rem;
    opacity: 0.7;
    margin: 0.2rem 0.5rem;
    text-align: right;
}

.user-timestamp {
    text-align: right;
}

.assistant-timestamp {
    text-align: left;
}

.typing-indicator {
    display: flex;
    align-items: center;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 18px;
    border-bottom-left-radius: 4px;
    max-width: 85%;
    border: 1px solid #e9ecef;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.typing-dots {
    display: flex;
    gap: 0.3rem;
}

.typing-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #6c757d;
    animation: typing 1.4s infinite;
}

.typing-dot:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-dot:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes typing {
    0%, 60%, 100% {
        transform: translateY(0);
        opacity: 0.5;
    }
    30% {
        transform: translateY(-10px);
        opacity: 1;
    }
}

.quick-actions {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin: 1rem 0;
}

.quick-action-btn {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border: 1px solid #2196f3;
    color: #1976d2;
    padding: 0.5rem;
    border-radius: 8px;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.3s ease;
    text-align: center;
}

.quick-action-btn:hover {
    background: linear-gradient(135deg, #bbdefb 0%, #90caf9 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(33,150,243,0.3);
}

.chat-stats {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 0.8rem;
    border-radius: 8px;
    margin: 1rem 0;
    border: 1px solid #dee2e6;
    font-size: 0.85rem;
}

.context-indicator {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border: 1px solid #28a745;
    padding: 0.8rem;
    border-radius: 8px;
    margin: 1rem 0;
    font-size: 0.85rem;
}

.chat-input-container {
    position: sticky;
    bottom: 0;
    background: white;
    padding: 1rem 0;
    border-top: 1px solid #e9ecef;
    margin-top: 1rem;
}

.welcome-message {
    text-align: center;
    padding: 2rem 1rem;
    color: #6c757d;
    font-style: italic;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 10px;
    margin: 1rem 0;
    border: 1px dashed #dee2e6;
}

.error-message {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    color: #721c24;
    padding: 0.8rem;
    border-radius: 8px;
    border: 1px solid #f1aeb5;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}

/* Sidebar enhancements */
.css-1d391kg {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
}

.css-1d391kg .css-10trblm {
    color: white;
}

/* Scrollbar styling for chat container */
.chat-container::-webkit-scrollbar {
    width: 6px;
}

.chat-container::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
}

.chat-container::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
}

.chat-container::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
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
    if 'chatbot_typing' not in st.session_state:
        st.session_state.chatbot_typing = False
    if 'total_messages' not in st.session_state:
        st.session_state.total_messages = 0

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

def format_timestamp():
    """Format current timestamp for chat messages"""
    return datetime.now().strftime("%H:%M")

def render_chat_message(message, index):
    """Render a single chat message with advanced styling"""
    timestamp = message.get('timestamp', format_timestamp())
    
    if message["role"] == "user":
        st.markdown(f'''
        <div class="chat-message user-message">
            <div class="message-bubble user-bubble">
                {message["content"]}
            </div>
            <div class="message-timestamp user-timestamp">
                👤 {timestamp}
            </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="chat-message assistant-message">
            <div class="message-bubble assistant-bubble">
                🤖 {message["content"]}
            </div>
            <div class="message-timestamp assistant-timestamp">
                {timestamp}
            </div>
        </div>
        ''', unsafe_allow_html=True)

def show_typing_indicator():
    """Show typing indicator animation"""
    st.markdown('''
    <div class="typing-indicator">
        <span style="margin-right: 0.5rem;">🤖 Medical Assistant is typing</span>
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

def get_context_summary(context):
    """Get summary of available context data"""
    summary = []
    
    if context and context.get('medical_extraction', {}).get('hlth_srvc_records'):
        medical_count = len(context['medical_extraction']['hlth_srvc_records'])
        summary.append(f"📋 {medical_count} medical records")
    
    if context and context.get('pharmacy_extraction', {}).get('ndc_records'):
        pharmacy_count = len(context['pharmacy_extraction']['ndc_records'])
        summary.append(f"💊 {pharmacy_count} pharmacy records")
    
    if context and context.get('heart_attack_prediction'):
        summary.append("❤️ heart attack prediction")
    
    if context and context.get('health_trajectory'):
        summary.append("📈 health trajectory analysis")
    
    if context and context.get('final_summary'):
        summary.append("📋 clinical summary")
    
    return summary

# Initialize session state
initialize_session_state()

# Main Title
st.markdown('<h1 class="main-header">🏥 Health Analysis Agent</h1>', unsafe_allow_html=True)

# Display import status
if not AGENT_AVAILABLE:
    st.markdown(f'<div class="status-error">❌ Failed to import Health Agent: {import_error}</div>', unsafe_allow_html=True)
    st.stop()

# ENHANCED SIDEBAR CHATBOT
with st.sidebar:
    if st.session_state.analysis_results and st.session_state.analysis_results.get("chatbot_ready", False) and st.session_state.chatbot_context:
        # Advanced Chatbot Header
        context_summary = get_context_summary(st.session_state.chatbot_context)
        patient_info = st.session_state.chatbot_context.get('patient_overview', {})
        
        st.markdown(f'''
        <div class="chatbot-header">
            <div>🤖 Medical Assistant</div>
            <div class="chatbot-status">
                🟢 Online • Patient Age: {patient_info.get('age', 'Unknown')}
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Context Indicator
        if context_summary:
            st.markdown(f'''
            <div class="context-indicator">
                <strong>📊 Available Data:</strong><br>
                {" • ".join(context_summary)}
            </div>
            ''', unsafe_allow_html=True)
        
        # Quick Action Buttons
        st.markdown('<div style="margin: 1rem 0;"><strong>🚀 Quick Actions:</strong></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💊 Medications", key="quick_meds", help="Ask about medications"):
                quick_question = "What medications has this patient been prescribed? Include NDC codes and drug names."
                st.session_state.chatbot_messages.append({
                    "role": "user", 
                    "content": quick_question,
                    "timestamp": format_timestamp()
                })
                st.rerun()
            
            if st.button("🩺 Diagnoses", key="quick_diag", help="Ask about diagnoses"):
                quick_question = "What medical diagnoses and ICD-10 codes are documented for this patient?"
                st.session_state.chatbot_messages.append({
                    "role": "user", 
                    "content": quick_question,
                    "timestamp": format_timestamp()
                })
                st.rerun()
        
        with col2:
            if st.button("❤️ Heart Risk", key="quick_heart", help="Ask about heart attack risk"):
                quick_question = "What is this patient's heart attack risk assessment and what factors contribute to it?"
                st.session_state.chatbot_messages.append({
                    "role": "user", 
                    "content": quick_question,
                    "timestamp": format_timestamp()
                })
                st.rerun()
            
            if st.button("📊 Summary", key="quick_summary", help="Get patient summary"):
                quick_question = "Provide a comprehensive summary of this patient's health status including key conditions, medications, and risk factors."
                st.session_state.chatbot_messages.append({
                    "role": "user", 
                    "content": quick_question,
                    "timestamp": format_timestamp()
                })
                st.rerun()
        
        # Chat Statistics
        if st.session_state.chatbot_messages:
            user_msgs = len([m for m in st.session_state.chatbot_messages if m["role"] == "user"])
            bot_msgs = len([m for m in st.session_state.chatbot_messages if m["role"] == "assistant"])
            
            st.markdown(f'''
            <div class="chat-stats">
                <strong>💬 Conversation Stats:</strong><br>
                👤 You: {user_msgs} messages • 🤖 Assistant: {bot_msgs} responses
            </div>
            ''', unsafe_allow_html=True)
        
        # Chat Container with Custom Styling
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        if st.session_state.chatbot_messages:
            for i, message in enumerate(st.session_state.chatbot_messages):
                render_chat_message(message, i)
        else:
            st.markdown('''
            <div class="welcome-message">
                <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">👋 Welcome!</div>
                <div>I'm your AI medical assistant with access to comprehensive patient data.</div>
                <div style="margin-top: 0.5rem; font-size: 0.9rem;">Ask me about diagnoses, medications, risk factors, or any medical insights!</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Show typing indicator if processing
        if st.session_state.chatbot_typing:
            show_typing_indicator()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat Input with Enhanced Container
        st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
        
        user_question = st.chat_input("💬 Ask about medical data, diagnoses, medications...")
        
        # Handle chat input with enhanced processing
        if user_question:
            # Add user message with timestamp
            st.session_state.chatbot_messages.append({
                "role": "user", 
                "content": user_question,
                "timestamp": format_timestamp()
            })
            
            # Set typing indicator
            st.session_state.chatbot_typing = True
            st.rerun()
        
        # Process pending user question
        if st.session_state.chatbot_typing and st.session_state.chatbot_messages and st.session_state.chatbot_messages[-1]["role"] == "user":
            try:
                # Get the latest user question
                latest_question = st.session_state.chatbot_messages[-1]["content"]
                
                # Process with agent
                chatbot_response = st.session_state.agent.chat_with_data(
                    latest_question, 
                    st.session_state.chatbot_context, 
                    st.session_state.chatbot_messages[:-1]  # Exclude the current question
                )
                
                # Add assistant response with timestamp
                st.session_state.chatbot_messages.append({
                    "role": "assistant", 
                    "content": chatbot_response,
                    "timestamp": format_timestamp()
                })
                
                # Update stats
                st.session_state.total_messages += 2
                
                # Clear typing indicator
                st.session_state.chatbot_typing = False
                st.rerun()
                
            except Exception as e:
                st.markdown(f'''
                <div class="error-message">
                    <strong>❌ Error:</strong> {str(e)}
                </div>
                ''', unsafe_allow_html=True)
                st.session_state.chatbot_typing = False
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Advanced Action Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True, help="Clear all messages"):
                st.session_state.chatbot_messages = []
                st.session_state.chatbot_typing = False
                st.rerun()
        
        with col2:
            if st.button("📥 Export Chat", use_container_width=True, help="Download conversation"):
                if st.session_state.chatbot_messages:
                    chat_export = {
                        "conversation": st.session_state.chatbot_messages,
                        "exported_at": datetime.now().isoformat(),
                        "total_messages": len(st.session_state.chatbot_messages),
                        "patient_context": st.session_state.chatbot_context.get('patient_overview', {})
                    }
                    
                    st.download_button(
                        "💾 Download JSON",
                        safe_json_dumps(chat_export),
                        f"medical_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
    
    else:
        # Enhanced placeholder when chatbot is not ready
        st.markdown('''
        <div class="chatbot-header">
            <div>💤 Medical Assistant</div>
            <div class="chatbot-status">
                🔴 Offline • Waiting for analysis
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="welcome-message">
            <div style="font-size: 1.1rem; margin-bottom: 1rem;">🚀 AI Medical Assistant</div>
            <div style="margin-bottom: 0.5rem;"><strong>Features:</strong></div>
            <div style="text-align: left; margin: 0.5rem 0;">
                • 💊 Medication analysis with NDC codes<br>
                • 🩺 Diagnosis interpretation with ICD-10<br>
                • ❤️ Heart attack risk assessment<br>
                • 📊 Comprehensive health insights<br>
                • 🔍 Interactive medical data exploration<br>
                • 📈 Health trajectory analysis
            </div>
            <div style="margin-top: 1rem; padding: 0.8rem; background: rgba(33,150,243,0.1); border-radius: 8px; border: 1px solid #2196f3;">
                <strong>🎯 Get Started:</strong><br>
                Complete the health analysis below to activate the AI assistant
            </div>
        </div>
        ''', unsafe_allow_html=True)

# Continue with the rest of the original code...
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
        
        # 2. RUN HEALTH ANALYSIS BUTTON
        submitted = st.form_submit_button(
            "🚀 Run Health Analysis", 
            use_container_width=True,
            disabled=st.session_state.analysis_running
        )

# Analysis Status
if st.session_state.analysis_running:
    st.markdown('<div class="status-success">🔄 Health analysis workflow executing... Please wait.</div>', unsafe_allow_html=True)

# Run Health Analysis
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
                st.success("✅ Health Analysis Agent initialized")
            except Exception as e:
                st.error(f"❌ Failed to initialize Health Agent: {str(e)}")
                st.stop()
        
        st.session_state.analysis_running = True
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🚀 Executing health analysis..."):
            try:
                # Progress updates
                for i, step in enumerate([
                    "Initializing workflow...",
                    "Fetching medical data...", 
                    "Deidentifying data...",
                    "Extracting medical information...",
                    "Analyzing health trajectory...",
                    "Predicting heart attack risk...",
                    "Initializing chatbot..."
                ]):
                    status_text.text(f"🔄 {step}")
                    progress_bar.progress(int((i + 1) * 14))
                    time.sleep(0.3)
                
                # Execute analysis
                results = st.session_state.agent.run_analysis(patient_data)
                
                if results.get("success", False):
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis completed successfully!")
                    st.session_state.analysis_results = results
                    st.session_state.chatbot_context = results.get("chatbot_context", {})
                    st.markdown('<div class="status-success">✅ Health analysis completed successfully!</div>', unsafe_allow_html=True)
                    
                    # Ensure chatbot is properly loaded with comprehensive context
                    if results.get("chatbot_ready", False) and st.session_state.chatbot_context:
                        st.success("💬 Advanced Medical Assistant is now available in the sidebar!")
                        st.info("🎯 Try the quick action buttons or ask detailed questions about medical data!")
                        
                        # Display brief summary of available data for chatbot
                        context_summary = get_context_summary(st.session_state.chatbot_context)
                        if context_summary:
                            st.info(f"📊 AI Assistant loaded with: {', '.join(context_summary)}")
                        
                        # Force page refresh to open sidebar
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning("⚠️ Advanced chatbot initialization incomplete. Some features may not be available.")
                else:
                    st.session_state.analysis_results = results
                    st.warning("⚠️ Analysis completed with some errors.")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.session_state.analysis_results = {
                    "success": False,
                    "error": str(e),
                    "patient_data": patient_data,
                    "errors": [str(e)]
                }
            finally:
                st.session_state.analysis_running = False

# Display Results if Available
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    # Show errors if any
    errors = safe_get(results, 'errors', [])
    if errors:
        st.markdown('<div class="status-error">❌ Analysis errors occurred</div>', unsafe_allow_html=True)

    # 3. MILLIMAN DATA BUTTON
    if st.button("📊 Milliman Data", use_container_width=True):
        st.markdown("""
        <div class="section-box">
            <div class="section-title">📊 Milliman Deidentified Data</div>
        </div>
        """, unsafe_allow_html=True)
        
        deidentified_data = safe_get(results, 'deidentified_data', {})
        
        if deidentified_data:
            tab1, tab2 = st.tabs(["🏥 Medical Data", "💊 Pharmacy Data"])
            
            with tab1:
                medical_data = safe_get(deidentified_data, 'medical', {})
                if medical_data:
                    st.markdown('<div class="json-container">', unsafe_allow_html=True)
                    st.json(medical_data)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.download_button(
                        "📥 Download Medical Data JSON",
                        safe_json_dumps(medical_data),
                        f"medical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    st.warning("No medical data available")
            
            with tab2:
                pharmacy_data = safe_get(deidentified_data, 'pharmacy', {})
                if pharmacy_data:
                    st.markdown('<div class="json-container">', unsafe_allow_html=True)
                    st.json(pharmacy_data)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.download_button(
                        "📥 Download Pharmacy Data JSON",
                        safe_json_dumps(pharmacy_data),
                        f"pharmacy_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    st.warning("No pharmacy data available")

    # Continue with other buttons... (rest of the original code remains the same)
    # [The rest of the buttons and functionality would continue here exactly as in the original code]
