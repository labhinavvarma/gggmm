# Configure Streamlit page FIRST
import streamlit as st
st.set_page_config(
    page_title="Health Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Now import other modules
import json
import pandas as pd
from datetime import datetime
import time
import sys
import os
from typing import Dict, Any, Optional

# Add current directory to path for importing the agent
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the MCP Health Agent
AGENT_AVAILABLE = False
import_error = None
MCPHealthAgent = None
Config = None
create_mcp_tools = None

try:
    from mcp_health_agent import MCPHealthAgent, Config, create_mcp_tools
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    import_error = str(e)

# Custom CSS for clean chat interface with better visual separation
st.markdown("""
<style>
.main-header {
    font-size: 2.8rem;
    color: #2e7d8b;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: bold;
}

.claims-section {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    border: 2px solid #dee2e6;
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0 2rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.claims-header {
    font-size: 1.4rem;
    color: #495057;
    margin-bottom: 1rem;
    font-weight: bold;
    border-bottom: 2px solid #dee2e6;
    padding-bottom: 0.5rem;
}

.claims-tabs {
    margin-top: 1rem;
}

.chat-section {
    background: linear-gradient(135deg, #ffffff, #f8f9fa);
    border: 3px solid #007bff;
    border-radius: 20px;
    padding: 0;
    margin: 2rem 0;
    box-shadow: 0 8px 16px rgba(0, 123, 255, 0.2);
    overflow: hidden;
}

.chat-header {
    background: linear-gradient(135deg, #007bff, #0056b3);
    color: white;
    padding: 1rem 1.5rem;
    margin: 0;
    font-size: 1.3rem;
    font-weight: bold;
    border-bottom: 2px solid #0056b3;
}

.chat-container {
    background: #ffffff;
    padding: 1.5rem;
    margin: 0;
    max-height: 500px;
    overflow-y: auto;
    border-bottom: 2px solid #e9ecef;
}

.chat-input-area {
    background: #f8f9fa;
    padding: 1rem 1.5rem;
    margin: 0;
}

.user-message {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    border: 1px solid #2196f3;
    border-left: 5px solid #2196f3;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    border-radius: 12px;
    box-shadow: 0 2px 4px rgba(33, 150, 243, 0.1);
}

.assistant-message {
    background: linear-gradient(135deg, #f3e5f5, #e1bee7);
    border: 1px solid #9c27b0;
    border-left: 5px solid #9c27b0;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    border-radius: 12px;
    box-shadow: 0 2px 4px rgba(156, 39, 176, 0.1);
}

.system-message {
    background: linear-gradient(135deg, #fff3e0, #ffecb3);
    border: 1px solid #ff9800;
    border-left: 5px solid #ff9800;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    border-radius: 12px;
    font-style: italic;
    box-shadow: 0 2px 4px rgba(255, 152, 0, 0.1);
}

.patient-info {
    background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
    border: 2px solid #4caf50;
    padding: 1.5rem;
    margin: 1rem 0;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(76, 175, 80, 0.1);
}

.progress-container {
    background: linear-gradient(135deg, #ffffff, #f8f9fa);
    border: 2px solid #17a2b8;
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(23, 162, 184, 0.1);
}

.node-status {
    display: inline-block;
    padding: 0.4rem 1rem;
    margin: 0.3rem;
    border-radius: 25px;
    font-size: 0.9rem;
    font-weight: bold;
    transition: all 0.3s ease;
}

.node-completed {
    background: linear-gradient(135deg, #d4edda, #c3e6cb);
    color: #155724;
    border: 2px solid #28a745;
    box-shadow: 0 2px 4px rgba(40, 167, 69, 0.2);
}

.node-running {
    background: linear-gradient(135deg, #cce7ff, #99d6ff);
    color: #004085;
    border: 2px solid #007bff;
    box-shadow: 0 2px 4px rgba(0, 123, 255, 0.2);
    animation: pulse 1.5s infinite;
}

.node-error {
    background: linear-gradient(135deg, #f8d7da, #f5c6cb);
    color: #721c24;
    border: 2px solid #dc3545;
    box-shadow: 0 2px 4px rgba(220, 53, 69, 0.2);
}

.node-pending {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    color: #6c757d;
    border: 2px solid #dee2e6;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.refresh-button {
    background: linear-gradient(45deg, #dc3545, #c82333);
    color: white;
    border: none;
    padding: 0.8rem 1.5rem;
    border-radius: 25px;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s;
    box-shadow: 0 4px 6px rgba(220, 53, 69, 0.3);
}

.refresh-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(220, 53, 69, 0.4);
}

.example-message {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    border: 2px solid #dee2e6;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.example-message:hover {
    background: linear-gradient(135deg, #e9ecef, #dee2e6);
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    border-color: #007bff;
}

.results-section {
    background: linear-gradient(135deg, #f8f9fa, #ffffff);
    border: 2px solid #28a745;
    border-radius: 15px;
    padding: 1.5rem;
    margin: 2rem 0;
    box-shadow: 0 6px 12px rgba(40, 167, 69, 0.1);
}

.section-divider {
    height: 3px;
    background: linear-gradient(90deg, #007bff, #28a745, #ffc107);
    margin: 2rem 0;
    border-radius: 3px;
}

.claims-card {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.claims-card:hover {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    transform: translateY(-1px);
    transition: all 0.3s ease;
}

.medical-claim {
    border-left: 4px solid #dc3545;
}

.pharmacy-claim {
    border-left: 4px solid #28a745;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'current_patient' not in st.session_state:
        st.session_state.current_patient = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}
    if 'chatbot_ready' not in st.session_state:
        st.session_state.chatbot_ready = False

def safe_json_dumps(data: Any, default: str = "{}") -> str:
    """Safely convert data to JSON string"""
    try:
        return json.dumps(data, indent=2) if data else default
    except Exception as e:
        return f'{{"error": "JSON serialization failed: {str(e)}"}}'

def refresh_patient():
    """Clear all patient data and start fresh"""
    st.session_state.current_patient = None
    st.session_state.analysis_results = None
    st.session_state.processing_status = {}
    st.session_state.chatbot_ready = False
    st.session_state.chat_messages = []
    st.rerun()

def display_node_status(step_status: Dict[str, str]):
    """Display the status of all nodes"""
    nodes = [
        ("parse_patient_data", "📝 Parse Data"),
        ("fetch_mcp_data", "🔗 MCP Tools"),
        ("deidentify_data", "🔒 Deidentify"),
        ("extract_medical_pharmacy_data", "🔍 Extract"),
        ("extract_entities", "🎯 Entities"),
        ("analyze_trajectory", "📈 Trajectory"),
        ("generate_summary", "📋 Summary"),
        ("initialize_chatbot", "💬 Chatbot")
    ]
    
    st.markdown("**🔄 Analysis Progress:**")
    status_html = ""
    for node_key, node_name in nodes:
        status = step_status.get(node_key, "pending")
        if status == "completed":
            css_class = "node-completed"
            icon = "✅"
        elif status == "running":
            css_class = "node-running"
            icon = "🔄"
        elif status == "error":
            css_class = "node-error"
            icon = "❌"
        else:
            css_class = "node-pending"
            icon = "⏳"
        
        status_html += f'<span class="node-status {css_class}">{icon} {node_name}</span>'
    
    st.markdown(f'<div>{status_html}</div>', unsafe_allow_html=True)

def parse_patient_command(message: str) -> bool:
    """Check if message is a patient analysis command"""
    keywords = ["analyze patient", "analyze", "patient", "new patient", "process patient"]
    return any(keyword in message.lower() for keyword in keywords)

def display_patient_info(patient_data: Dict[str, Any]):
    """Display current patient information"""
    if patient_data:
        st.markdown(f"""
        <div class="patient-info">
            <h4>👤 Current Patient</h4>
            <p><strong>Name:</strong> {patient_data.get('first_name', 'Unknown')} {patient_data.get('last_name', 'Unknown')}</p>
            <p><strong>DOB:</strong> {patient_data.get('date_of_birth', 'Unknown')}</p>
            <p><strong>Gender:</strong> {patient_data.get('gender', 'Unknown')}</p>
            <p><strong>Zip:</strong> {patient_data.get('zip_code', 'Unknown')}</p>
        </div>
        """, unsafe_allow_html=True)

# Initialize session state
initialize_session_state()

# Main Title
st.markdown('<h1 class="main-header">🏥 Health Agent</h1>', unsafe_allow_html=True)
st.markdown("**Chat-driven health analysis with MCP tool integration**")

# Display import status
if AGENT_AVAILABLE:
    st.success("✅ Health Agent ready!")
else:
    st.error(f"❌ Failed to import Health Agent: {import_error}")
    st.stop()

# Top section: Deidentified Medical and Pharmacy Claims
if st.session_state.analysis_results and st.session_state.chatbot_ready:
    st.markdown('<div class="claims-section">', unsafe_allow_html=True)
    st.markdown('<div class="claims-header">📊 Deidentified Medical & Pharmacy Claims</div>', unsafe_allow_html=True)
    
    results = st.session_state.analysis_results
    deidentified_data = results.get("chatbot_context", {})
    
    # Create tabs for medical and pharmacy claims
    claims_tab1, claims_tab2 = st.tabs(["🏥 Medical Claims", "💊 Pharmacy Claims"])
    
    with claims_tab1:
        medical_data = deidentified_data.get("deidentified_medical", {})
        medical_extraction = deidentified_data.get("medical_extraction", {})
        
        if medical_data and not medical_data.get("error"):
            # Patient overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Patient Age", medical_data.get("src_mbr_age", "Unknown"))
            with col2:
                st.metric("Records Found", len(medical_extraction.get("hlth_srvc_records", [])))
            with col3:
                st.metric("Diagnosis Codes", medical_extraction.get("extraction_summary", {}).get("total_diagnosis_codes", 0))
            with col4:
                st.metric("Service Codes", len(medical_extraction.get("extraction_summary", {}).get("unique_service_codes", [])))
            
            # Display medical records
            hlth_records = medical_extraction.get("hlth_srvc_records", [])
            if hlth_records:
                st.markdown("**🏥 Medical Records:**")
                for i, record in enumerate(hlth_records[:3]):  # Show first 3
                    st.markdown(f"""
                    <div class="claims-card medical-claim">
                        <strong>Record {i+1}</strong><br>
                        <strong>Service Code:</strong> {record.get('hlth_srvc_cd', 'N/A')}<br>
                        <strong>Diagnosis Codes:</strong> {', '.join([d.get('code', '') for d in record.get('diagnosis_codes', [])][:3])}
                        {' (+more)' if len(record.get('diagnosis_codes', [])) > 3 else ''}
                    </div>
                    """, unsafe_allow_html=True)
                
                if len(hlth_records) > 3:
                    st.info(f"Showing 3 of {len(hlth_records)} medical records")
            else:
                st.info("No medical records found")
        else:
            st.warning("Medical claims data not available")
    
    with claims_tab2:
        pharmacy_data = deidentified_data.get("deidentified_pharmacy", {})
        pharmacy_extraction = deidentified_data.get("pharmacy_extraction", {})
        
        if pharmacy_data and not pharmacy_data.get("error"):
            # Pharmacy overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prescriptions", len(pharmacy_extraction.get("ndc_records", [])))
            with col2:
                st.metric("Unique NDC Codes", len(pharmacy_extraction.get("extraction_summary", {}).get("unique_ndc_codes", [])))
            with col3:
                st.metric("Unique Medications", len(pharmacy_extraction.get("extraction_summary", {}).get("unique_label_names", [])))
            
            # Display pharmacy records
            ndc_records = pharmacy_extraction.get("ndc_records", [])
            if ndc_records:
                st.markdown("**💊 Pharmacy Records:**")
                for i, record in enumerate(ndc_records[:3]):  # Show first 3
                    st.markdown(f"""
                    <div class="claims-card pharmacy-claim">
                        <strong>Prescription {i+1}</strong><br>
                        <strong>NDC:</strong> {record.get('ndc', 'N/A')}<br>
                        <strong>Medication:</strong> {record.get('lbl_nm', 'N/A')}<br>
                        <strong>Generic:</strong> {record.get('generic_name', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)
                
                if len(ndc_records) > 3:
                    st.info(f"Showing 3 of {len(ndc_records)} pharmacy records")
            else:
                st.info("No pharmacy records found")
        else:
            st.warning("Pharmacy claims data not available")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Section divider
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Sidebar for configuration and debug
with st.sidebar:
    st.header("⚙️ Health Agent Configuration")
    
    # Initialize agent if not already done
    if st.session_state.agent is None:
        try:
            config = Config()
            mcp_tools = create_mcp_tools()
            st.session_state.agent = MCPHealthAgent(config, mcp_tools)
            st.success("✅ Agent initialized with MCP tools")
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {str(e)}")
    
    st.markdown("### 📊 System Status")
    st.markdown("✅ **MCP Tools Ready**")
    st.markdown("🤖 **AI Analysis Ready**")
    st.markdown("💬 **Chat Interface Active**")
    
    if st.session_state.current_patient:
        st.markdown("### 👤 Current Patient")
        patient = st.session_state.current_patient
        st.write(f"**Name:** {patient.get('first_name', '')} {patient.get('last_name', '')}")
        st.write(f"**Status:** {'Analysis Complete' if st.session_state.chatbot_ready else 'Processing...'}")
    
    # Debug information
    if st.checkbox("🐛 Show Debug Info"):
        st.markdown("### 🔧 Debug Information")
        st.write(f"Agent Ready: {st.session_state.agent is not None}")
        st.write(f"Current Patient: {st.session_state.current_patient is not None}")
        st.write(f"Chatbot Ready: {st.session_state.chatbot_ready}")
        st.write(f"Chat Messages: {len(st.session_state.chat_messages)}")
        
        if st.session_state.processing_status:
            st.markdown("**Node Status:**")
            for node, status in st.session_state.processing_status.items():
                st.write(f"- {node}: {status}")

# Chat Interface Section
st.markdown('<div class="chat-section">', unsafe_allow_html=True)
st.markdown('<div class="chat-header">💬 Health Agent Chat Interface</div>', unsafe_allow_html=True)

# Refresh Patient Button (inside chat section)
st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔄 Refresh Patient & Chat", use_container_width=True, type="secondary"):
        refresh_patient()
st.markdown('</div>', unsafe_allow_html=True)

# Display current patient info if available
if st.session_state.current_patient:
    st.markdown('<div style="padding: 0 1.5rem;">', unsafe_allow_html=True)
    display_patient_info(st.session_state.current_patient)
    st.markdown('</div>', unsafe_allow_html=True)

# Display processing status if analysis is running
if st.session_state.processing_status:
    st.markdown('<div style="padding: 0 1.5rem;">', unsafe_allow_html=True)
    st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    display_node_status(st.session_state.processing_status)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Chat messages display
if st.session_state.chat_messages:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.chat_messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>👤 You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
        elif message["role"] == "assistant":
            st.markdown(f'<div class="assistant-message"><strong>🤖 Health Agent:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
        elif message["role"] == "system":
            st.markdown(f'<div class="system-message"><strong>⚙️ System:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Show example commands when no messages
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown("### 💡 How to Use")
    st.info("Start by typing a patient analysis command. The Health Agent will automatically process the patient through all analysis nodes, then you can ask questions about the results.")
    
    st.markdown("**📝 Example Commands:**")
    
    examples = [
        "Analyze patient John Doe, SSN 123456789, DOB 1990-05-15, Gender M, Zip 12345",
        "Process patient Mary Smith, SSN 987654321, DOB 1985-12-03, Gender F, Zip 54321", 
        "New patient analysis: Robert Johnson, SSN 456789123, DOB 1978-08-22, Gender M, Zip 67890"
    ]
    
    for i, example in enumerate(examples):
        if st.button(f"💬 {example}", key=f"example_{i}"):
            # Use the example as input
            st.session_state.chat_messages.append({"role": "user", "content": example})
            
            # Process the command
            with st.spinner("🔄 Processing patient analysis..."):
                try:
                    # Add system message
                    st.session_state.chat_messages.append({
                        "role": "system", 
                        "content": "🚀 Starting patient analysis through all nodes..."
                    })
                    
                    # Run analysis
                    results = st.session_state.agent.run_patient_analysis(example)
                    
                    if results["success"]:
                        st.session_state.current_patient = results["patient_data"]
                        st.session_state.analysis_results = results
                        st.session_state.chatbot_ready = results["chatbot_ready"]
                        st.session_state.processing_status = results["step_status"]
                        
                        # Add success message
                        patient_name = f"{results['patient_data'].get('first_name', 'Unknown')} {results['patient_data'].get('last_name', 'Unknown')}"
                        success_msg = f"✅ Analysis complete for {patient_name}! You can now ask questions about this patient's health data."
                        st.session_state.chat_messages.append({
                            "role": "assistant", 
                            "content": success_msg
                        })
                        
                    else:
                        error_msg = f"❌ Analysis failed. Errors: {', '.join(results.get('errors', ['Unknown error']))}"
                        st.session_state.chat_messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
                
                except Exception as e:
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": f"❌ Error during analysis: {str(e)}"
                    })
            
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Chat input area
st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
user_input = st.chat_input("💬 Type your message here...")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # End of chat-section

if user_input:
    # Add user message to chat
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    
    # Check if this is a patient analysis command
    if parse_patient_command(user_input) and not st.session_state.chatbot_ready:
        # Process patient analysis
        with st.spinner("🔄 Processing patient analysis through all nodes..."):
            try:
                # Add system message
                st.session_state.chat_messages.append({
                    "role": "system", 
                    "content": "🚀 Starting patient analysis: Parse Data → MCP Tools → Deidentify → Extract → Entities → Trajectory → Summary → Chatbot"
                })
                
                # Update progress in real-time
                progress_placeholder = st.empty()
                
                # Run analysis
                results = st.session_state.agent.run_patient_analysis(user_input)
                
                if results["success"]:
                    st.session_state.current_patient = results["patient_data"]
                    st.session_state.analysis_results = results
                    st.session_state.chatbot_ready = results["chatbot_ready"]
                    st.session_state.processing_status = results["step_status"]
                    
                    # Add success message with summary
                    patient_name = f"{results['patient_data'].get('first_name', 'Unknown')} {results['patient_data'].get('last_name', 'Unknown')}"
                    
                    success_response = f"""✅ **Analysis Complete for {patient_name}**

**📋 Health Summary:**
{results.get('final_summary', 'Summary not available')}

**🎯 Key Health Entities:**
{safe_json_dumps(results.get('entity_extraction', {}), 'No entities extracted')}

You can now ask specific questions about this patient's medical data, medications, conditions, or any other health-related insights!
"""
                    
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": success_response
                    })
                    
                else:
                    error_msg = f"❌ **Analysis Failed**\n\nErrors encountered: {', '.join(results.get('errors', ['Unknown error']))}\n\nPlease check the patient information and try again."
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
                    
                    # Also update processing status to show where it failed
                    st.session_state.processing_status = results.get("step_status", {})
            
            except Exception as e:
                error_response = f"❌ **System Error**\n\nAn unexpected error occurred during analysis: {str(e)}\n\nPlease try again or refresh the patient data."
                st.session_state.chat_messages.append({
                    "role": "assistant", 
                    "content": error_response
                })
    
    elif st.session_state.chatbot_ready and st.session_state.analysis_results:
        # Handle follow-up questions about the patient
        with st.spinner("🤖 Analyzing patient data to answer your question..."):
            try:
                chatbot_context = st.session_state.analysis_results.get("chatbot_context", {})
                response = st.session_state.agent.chat_with_patient_data(user_input, chatbot_context)
                
                st.session_state.chat_messages.append({
                    "role": "assistant", 
                    "content": response
                })
            
            except Exception as e:
                error_response = f"❌ **Error Processing Question**\n\nI encountered an error while analyzing the patient data: {str(e)}\n\nPlease try rephrasing your question."
                st.session_state.chat_messages.append({
                    "role": "assistant", 
                    "content": error_response
                })
    
    else:
        # No patient analyzed yet
        help_response = """ℹ️ **No Patient Data Available**

To start, please provide a patient analysis command with the following information:
- Patient name
- SSN
- Date of birth (YYYY-MM-DD)
- Gender (M/F)
- Zip code

**Example:**
"Analyze patient John Doe, SSN 123456789, DOB 1990-05-15, Gender M, Zip 12345"

Once the analysis is complete, you can ask questions about the patient's health data!"""
        
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": help_response
        })
    
    st.rerun()

# Analysis Results Section (if available)
if st.session_state.analysis_results and st.session_state.chatbot_ready:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    st.markdown("## 📊 Analysis Results")
    
    results = st.session_state.analysis_results
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Analysis Status", "✅ Complete")
    with col2:
        patient_name = f"{results['patient_data'].get('first_name', '')} {results['patient_data'].get('last_name', '')}"
        st.metric("Patient", patient_name)
    with col3:
        st.metric("Chatbot Status", "💬 Ready")
    with col4:
        st.metric("Total Messages", len(st.session_state.chat_messages))
    
    # Expandable sections for detailed results
    with st.expander("🎯 Health Entities"):
        st.json(results.get("entity_extraction", {}))
    
    with st.expander("📈 Health Trajectory Analysis"):
        st.markdown(results.get("health_trajectory", "No trajectory analysis available"))
    
    with st.expander("📋 Executive Summary"):
        st.markdown(results.get("final_summary", "No summary available"))
    
    # Download section
    st.markdown("### 💾 Download Results")
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON download
        complete_report = {
            "patient_data": results.get("patient_data", {}),
            "health_trajectory": results.get("health_trajectory", ""),
            "final_summary": results.get("final_summary", ""),
            "entity_extraction": results.get("entity_extraction", {}),
            "chat_history": st.session_state.chat_messages,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        st.download_button(
            "📊 Download Complete Report (JSON)",
            safe_json_dumps(complete_report),
            f"health_agent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Text report
        patient_name = f"{results['patient_data'].get('first_name', 'Unknown')} {results['patient_data'].get('last_name', 'Unknown')}"
        
        text_report = f"""
HEALTH AGENT ANALYSIS REPORT
============================
Patient: {patient_name}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

HEALTH TRAJECTORY:
==================
{results.get('health_trajectory', 'Not available')}

EXECUTIVE SUMMARY:
==================
{results.get('final_summary', 'Not available')}

CHAT HISTORY:
=============
{chr(10).join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.chat_messages])}
        """
        
        st.download_button(
            "📝 Download Text Report",
            text_report,
            f"health_agent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 15px; margin: 1rem 0;'>
    🏥 <strong>Health Agent</strong> - MCP-Powered Chat Interface<br>
    Advanced health analysis through conversational AI and MCP tool integration<br>
    ⚠️ <em>This analysis is for informational purposes only and should not replace professional medical advice.</em>
</div>
""", unsafe_allow_html=True)
