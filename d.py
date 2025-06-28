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

# Custom CSS for clean layout and stable progressive animation
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

/* Stable Progressive Deep Research Animation */
.deep-research-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 12px;
    margin: 2rem 0;
    color: white;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    border: 2px solid rgba(255, 255, 255, 0.2);
    animation: containerPulse 2s ease-in-out infinite;
}

.progress-header {
    text-align: center;
    margin-bottom: 2rem;
}

.progress-bar-container {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    height: 8px;
    margin: 1rem 0;
    overflow: hidden;
}

.progress-bar-fill {
    background: linear-gradient(90deg, #28a745, #20c997);
    height: 100%;
    border-radius: 10px;
    transition: width 0.5s ease;
    box-shadow: 0 2px 10px rgba(40, 167, 69, 0.5);
}

.research-step {
    display: flex;
    align-items: center;
    padding: 1rem 0;
    margin: 0.5rem 0;
    border-radius: 8px;
    transition: all 0.6s ease;
    opacity: 1;
    transform: translateX(0);
}

.research-step.slide-in {
    animation: slideInStep 0.6s ease forwards;
}

.research-step-icon {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    margin-right: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    font-weight: bold;
    transition: all 0.4s ease;
    border: 2px solid transparent;
    flex-shrink: 0;
}

.step-pending {
    background: rgba(255, 255, 255, 0.1);
    color: rgba(255, 255, 255, 0.5);
    border-color: rgba(255, 255, 255, 0.2);
}

.step-running {
    background: #ffc107;
    color: #000;
    border-color: #ffca2c;
    animation: pulse-strong 1.2s infinite;
    box-shadow: 0 0 20px rgba(255, 193, 7, 0.8);
}

.step-completed {
    background: #28a745;
    color: white;
    border-color: #34ce57;
    box-shadow: 0 0 15px rgba(40, 167, 69, 0.6);
}

.step-error {
    background: #dc3545;
    color: white;
    border-color: #e4606d;
    animation: errorShake 0.5s ease-in-out;
}

.research-step-text {
    flex: 1;
    font-weight: 500;
    transition: all 0.3s ease;
    font-size: 1.1rem;
}

.step-running .research-step-text {
    font-weight: 600;
    color: #fff;
}

.step-completed .research-step-text {
    opacity: 0.9;
}

.next-step-preview {
    background: rgba(255, 255, 255, 0.1);
    border: 1px dashed rgba(255, 255, 255, 0.3);
    border-radius: 8px;
    padding: 0.8rem;
    margin: 0.5rem 0;
    color: rgba(255, 255, 255, 0.7);
    font-style: italic;
    text-align: center;
    animation: fadeInOut 2s ease-in-out infinite;
}

.step-counter {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    padding: 0.5rem 1rem;
    display: inline-block;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.results-section {
    margin-top: 2rem;
    opacity: 1;
    transition: opacity 0.5s ease;
}

@keyframes slideInStep {
    from {
        opacity: 0;
        transform: translateX(-30px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes pulse-strong {
    0% { 
        transform: scale(1); 
        box-shadow: 0 0 20px rgba(255, 193, 7, 0.8);
    }
    50% { 
        transform: scale(1.1); 
        box-shadow: 0 0 30px rgba(255, 193, 7, 1);
    }
    100% { 
        transform: scale(1); 
        box-shadow: 0 0 20px rgba(255, 193, 7, 0.8);
    }
}

@keyframes containerPulse {
    0% { 
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    50% { 
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
    }
    100% { 
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
}

@keyframes errorShake {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-5px); }
    75% { transform: translateX(5px); }
}

@keyframes fadeInOut {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 0.8; }
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
    
    # Stable Progressive Animation Workflow Steps
    if 'workflow_steps' not in st.session_state:
        st.session_state.workflow_steps = {
            1: {'name': 'Fetching Claims Data', 'status': 'pending', 'description': 'Retrieving medical and pharmacy claims'},
            2: {'name': 'Deidentifying Claims Data', 'status': 'pending', 'description': 'Removing personal identifiers'},
            3: {'name': 'Extracting Claims Fields', 'status': 'pending', 'description': 'Parsing medical codes and data'},
            4: {'name': 'Extracting Health Entities', 'status': 'pending', 'description': 'Identifying health conditions'},
            5: {'name': 'Analyzing Health Trajectory', 'status': 'pending', 'description': 'Computing health trends'},
            6: {'name': 'Generating Summary', 'status': 'pending', 'description': 'Creating clinical summary'},
            7: {'name': 'Predicting Heart Attack Risk', 'status': 'pending', 'description': 'Running ML risk assessment'},
            8: {'name': 'Initializing Assistant', 'status': 'pending', 'description': 'Setting up medical chatbot'}
        }
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'show_animation' not in st.session_state:
        st.session_state.show_animation = False
    if 'steps_revealed' not in st.session_state:
        st.session_state.steps_revealed = 0
    if 'animation_complete' not in st.session_state:
        st.session_state.animation_complete = False

def update_workflow_step(step_number: int, status: str):
    """Safely update workflow step status and reveal steps progressively"""
    try:
        if step_number in st.session_state.workflow_steps:
            st.session_state.workflow_steps[step_number]['status'] = status
            st.session_state.current_step = step_number
            
            # Progressive reveal: show current step and previous steps
            if status == 'running':
                st.session_state.steps_revealed = max(st.session_state.steps_revealed, step_number)
            elif status == 'completed':
                st.session_state.steps_revealed = max(st.session_state.steps_revealed, step_number + 1)
                
    except Exception as e:
        logger.warning(f"Could not update workflow step {step_number}: {e}")

def reset_workflow_steps():
    """Reset all workflow steps to pending and hide all steps"""
    try:
        for step_num in st.session_state.workflow_steps:
            st.session_state.workflow_steps[step_num]['status'] = 'pending'
        st.session_state.current_step = 0
        st.session_state.steps_revealed = 0
        st.session_state.animation_complete = False
    except Exception as e:
        logger.warning(f"Could not reset workflow steps: {e}")

def complete_animation():
    """Mark animation as complete and prepare for results display"""
    try:
        st.session_state.animation_complete = True
        st.session_state.show_animation = False
        # Ensure all steps are marked as completed
        for step_num in st.session_state.workflow_steps:
            if st.session_state.workflow_steps[step_num]['status'] != 'error':
                st.session_state.workflow_steps[step_num]['status'] = 'completed'
    except Exception as e:
        logger.warning(f"Could not complete animation: {e}")

def display_progressive_workflow_animation():
    """Display stable progressive workflow animation that reveals steps one by one"""
    try:
        # Calculate progress percentage
        completed_steps = sum(1 for step in st.session_state.workflow_steps.values() if step['status'] == 'completed')
        progress_percentage = (completed_steps / len(st.session_state.workflow_steps)) * 100
        
        # Build the progressive steps HTML
        steps_html = ""
        steps_revealed = st.session_state.steps_revealed
        
        for step_num in range(1, min(steps_revealed + 2, 9)):  # Show current + 1 next step
            if step_num in st.session_state.workflow_steps:
                step_info = st.session_state.workflow_steps[step_num]
                status = step_info['status']
                name = step_info['name']
                description = step_info.get('description', '')
                
                if status == 'pending':
                    icon_class = "step-pending"
                    icon_text = str(step_num)
                elif status == 'running':
                    icon_class = "step-running"
                    icon_text = "●"
                elif status == 'completed':
                    icon_class = "step-completed" 
                    icon_text = "✓"
                elif status == 'error':
                    icon_class = "step-error"
                    icon_text = "✗"
                else:
                    icon_class = "step-pending"
                    icon_text = str(step_num)
                
                # Add slide-in class for new steps
                slide_class = "slide-in" if step_num <= steps_revealed else ""
                
                steps_html += f"""
                <div class="research-step {slide_class}">
                    <div class="research-step-icon {icon_class}">{icon_text}</div>
                    <div style="flex: 1;">
                        <div class="research-step-text">{name}</div>
                        <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 0.2rem;">{description}</div>
                    </div>
                </div>
                """
        
        # Show preview of next step if there is one
        next_step_preview = ""
        if steps_revealed < len(st.session_state.workflow_steps) and st.session_state.current_step > 0:
            next_step_num = steps_revealed + 1
            if next_step_num <= len(st.session_state.workflow_steps):
                next_step_name = st.session_state.workflow_steps[next_step_num]['name']
                next_step_preview = f"""
                <div class="next-step-preview">
                    Next: {next_step_name}...
                </div>
                """
        
        # Create the complete progressive animation HTML
        animation_html = f"""
        <div class="deep-research-container">
            <div class="progress-header">
                <h3 style="margin-bottom: 1rem; color: white;">🔬 Deep Research Analysis</h3>
                <div class="step-counter">Step {st.session_state.current_step} of 8</div>
                <div class="progress-bar-container">
                    <div class="progress-bar-fill" style="width: {progress_percentage}%;"></div>
                </div>
            </div>
            
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
                {steps_html}
                {next_step_preview}
                
                <div style="text-align: center; margin-top: 1.5rem; font-style: italic; opacity: 0.9; font-size: 0.9rem;">
                    Comprehensive analysis in progress...
                </div>
            </div>
        </div>
        """
        
        return animation_html
        
    except Exception as e:
        logger.warning(f"Error creating progressive animation HTML: {e}")
        return """
        <div class="deep-research-container">
            <h3 style="margin-bottom: 1rem; color: white; text-align: center;">🔬 Deep Research Analysis</h3>
            <div style="text-align: center; padding: 2rem;">
                Processing comprehensive claims data analysis...
            </div>
        </div>
        """

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
            type="primary"  # This enables the green styling from CSS
        )

# Analysis Status Display with Stable Progressive Animation
animation_container = st.empty()

# Only show animation if currently running and animation is enabled
if st.session_state.analysis_running and st.session_state.show_animation:
    with animation_container.container():
        st.markdown(display_progressive_workflow_animation(), unsafe_allow_html=True)

# Clear animation when not running
elif not st.session_state.analysis_running and st.session_state.animation_complete:
    animation_container.empty()

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
        
        # Start analysis
        st.session_state.analysis_running = True
        st.session_state.show_animation = True
        st.session_state.animation_complete = False
        
        # Reset and initialize progressive workflow animation
        reset_workflow_steps()
        
        # Show starting messages
        st.info("🔬 Starting Deep Research Analysis - Watch the progressive workflow below:")
        st.warning("⏳ This may take 30-60 seconds. Steps will appear one by one as they execute.")
        
        # Progressive workflow simulation
        try:
            # Progressive step execution with real-time animation updates
            for step_num in range(1, 9):
                # Start step
                update_workflow_step(step_num, 'running')
                
                # Update animation display in real-time
                with animation_container.container():
                    st.markdown(display_progressive_workflow_animation(), unsafe_allow_html=True)
                
                # Simulate step processing time
                time.sleep(0.8)  # Delay to see progression
                
                # Complete step
                update_workflow_step(step_num, 'completed')
                
                # Update animation to show completion
                with animation_container.container():
                    st.markdown(display_progressive_workflow_animation(), unsafe_allow_html=True)
                
                time.sleep(0.2)  # Brief pause
            
            # Execute the actual analysis
            with st.spinner("🔬 Executing deep research analysis..."):
                results = st.session_state.agent.run_analysis(patient_data)
            
            # Complete animation and prepare for results
            complete_animation()
            
            # Process results
            if results.get("success", False):
                # Store successful results
                st.session_state.analysis_results = results
                st.session_state.chatbot_context = results.get("chatbot_context", {})
                
                # Clear animation container
                animation_container.empty()
                
                # Show completion message
                st.success("🎉 All 8 workflow steps completed successfully!")
                st.markdown('<div class="status-success">✅ Deep research analysis completed successfully!</div>', unsafe_allow_html=True)
                
                # Ensure chatbot is properly loaded
                if results.get("chatbot_ready", False) and st.session_state.chatbot_context:
                    st.success("💬 Medical Assistant is now available in the sidebar with full access to all claims data!")
                    st.info("🎯 You can ask detailed questions about diagnoses, medications, dates, medical codes, and more!")
                    
                    # Display brief summary
                    context_summary = []
                    if safe_get(results, 'structured_extractions', {}).get('medical', {}).get('hlth_srvc_records'):
                        medical_count = len(safe_get(results, 'structured_extractions', {})['medical']['hlth_srvc_records'])
                        context_summary.append(f"📋 {medical_count} medical records")
                    
                    if safe_get(results, 'structured_extractions', {}).get('pharmacy', {}).get('ndc_records'):
                        pharmacy_count = len(safe_get(results, 'structured_extractions', {})['pharmacy']['ndc_records'])
                        context_summary.append(f"💊 {pharmacy_count} pharmacy records")
                    
                    if safe_get(results, 'heart_attack_prediction', {}):
                        context_summary.append("❤️ heart attack prediction")
                    
                    if context_summary:
                        st.info(f"📊 Chatbot has access to: {', '.join(context_summary)}")
                    
                    # Force page refresh to open sidebar
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("⚠️ Chatbot initialization incomplete. Some features may not be available.")
            else:
                # Handle analysis failure
                if st.session_state.current_step > 0:
                    update_workflow_step(st.session_state.current_step, 'error')
                st.session_state.analysis_results = results
                animation_container.empty()
                st.warning("⚠️ Analysis completed with some errors.")
            
        except Exception as e:
            # Mark current step as error
            if st.session_state.current_step > 0:
                update_workflow_step(st.session_state.current_step, 'error')
            
            logger.error(f"Deep research analysis failed: {str(e)}")
            st.error(f"❌ Deep research analysis failed: {str(e)}")
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

# RESULTS SECTION - STABLE DISPLAY
# This section is separate from animation and always displays when results are available
if st.session_state.analysis_results and not st.session_state.analysis_running:
    results = st.session_state.analysis_results
    
    # Create a dedicated results container
    results_container = st.container()
    
    with results_container:
        # Add a separator
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

        # 3. CLAIMS DATA BUTTON
        if st.button("📊 Claims Data", use_container_width=True, key="claims_data_btn"):
            st.session_state.show_claims_data = not st.session_state.show_claims_data
        
        if st.session_state.show_claims_data:
            st.markdown("""
            <div class="section-box">
                <div class="section-title">📊 Deidentified Claims Data</div>
            </div>
            """, unsafe_allow_html=True)
            
            deidentified_data = safe_get(results, 'deidentified_data', {})
            
            if deidentified_data:
                tab1, tab2 = st.tabs(["🏥 Medical Claims", "💊 Pharmacy Claims"])
                
                with tab1:
                    medical_data = safe_get(deidentified_data, 'medical', {})
                    if medical_data:
                        st.markdown("**🏥 Deidentified Medical Claims Data:**")
                        st.markdown('<div class="json-container">', unsafe_allow_html=True)
                        st.json(medical_data)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.download_button(
                            "📥 Download Medical Claims Data JSON",
                            safe_json_dumps(medical_data),
                            f"medical_claims_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    else:
                        st.warning("No medical claims data available")
                
                with tab2:
                    pharmacy_data = safe_get(deidentified_data, 'pharmacy', {})
                    if pharmacy_data:
                        st.markdown("**💊 Deidentified Pharmacy Claims Data:**")
                        st.markdown('<div class="json-container">', unsafe_allow_html=True)
                        st.json(pharmacy_data)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.download_button(
                            "📥 Download Pharmacy Claims Data JSON",
                            safe_json_dumps(pharmacy_data),
                            f"pharmacy_claims_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    else:
                        st.warning("No pharmacy claims data available")

        # 4. CLAIMS DATA EXTRACTION BUTTON
        if st.button("🔍 Claims Data Extraction", use_container_width=True, key="claims_extraction_btn"):
            st.session_state.show_claims_extraction = not st.session_state.show_claims_extraction
        
        if st.session_state.show_claims_extraction:
            st.markdown("""
            <div class="section-box">
                <div class="section-title">🔍 Claims Data Field Extraction</div>
            </div>
            """, unsafe_allow_html=True)
            
            structured_extractions = safe_get(results, 'structured_extractions', {})
            
            if structured_extractions:
                tab1, tab2 = st.tabs(["🏥 Medical Claims Extraction", "💊 Pharmacy Claims Extraction"])
                
                with tab1:
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
                            st.markdown("**📋 Extracted Medical Claims Records:**")
                            for i, record in enumerate(hlth_srvc_records, 1):
                                with st.expander(f"Medical Record {i} - Service Code: {record.get('hlth_srvc_cd', 'N/A')}"):
                                    st.write(f"**Service Code:** `{record.get('hlth_srvc_cd', 'N/A')}`")
                                    st.write(f"**Data Path:** `{record.get('data_path', 'N/A')}`")
                                    
                                    diagnosis_codes = record.get('diagnosis_codes', [])
                                    if diagnosis_codes:
                                        st.write("**Diagnosis Codes:**")
                                        for idx, diag in enumerate(diagnosis_codes, 1):
                                            source_info = f" (from {diag.get('source', 'individual field')})" if diag.get('source') else ""
                                            st.write(f"  {idx}. `{diag.get('code', 'N/A')}`{source_info}")
                    else:
                        st.warning("No medical claims extraction data available")
                
                with tab2:
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
                            st.markdown("**💊 Extracted Pharmacy Claims Records:**")
                            for i, record in enumerate(ndc_records, 1):
                                with st.expander(f"Pharmacy Record {i} - {record.get('lbl_nm', 'N/A')}"):
                                    st.write(f"**NDC Code:** `{record.get('ndc', 'N/A')}`")
                                    st.write(f"**Label Name:** `{record.get('lbl_nm', 'N/A')}`")
                                    st.write(f"**Data Path:** `{record.get('data_path', 'N/A')}`")
                    else:
                        st.warning("No pharmacy claims extraction data available")

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
                
                # Medical conditions
                medical_conditions = safe_get(entity_extraction, 'medical_conditions', [])
                if medical_conditions:
                    st.markdown("**🏥 Medical Conditions Identified:**")
                    for condition in medical_conditions:
                        st.write(f"• {condition}")
                
                # Medications identified
                medications_identified = safe_get(entity_extraction, 'medications_identified', [])
                if medications_identified:
                    st.markdown("**💊 Medications Identified:**")
                    for med in medications_identified:
                        st.write(f"• **{med.get('label_name', 'N/A')}** (NDC: {med.get('ndc', 'N/A')})")

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
                    <p style="color: #6c757d; margin-top: 1rem; font-size: 0.9rem;">
                        Prediction from ML Server: {heart_attack_prediction.get('fastapi_server_url', 'Unknown')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                error_msg = heart_attack_prediction.get('error', 'Heart attack prediction not available')
                st.error(f"❌ Server Error: {error_msg}")
                
                # Show connection info for debugging
                st.info(f"💡 Expected Server: {st.session_state.config.heart_attack_api_url if st.session_state.config else 'http://localhost:8080'}")
                st.info("💡 Make sure server is running: `python app.py`")
