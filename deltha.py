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

# Custom CSS for clean chat interface
st.markdown("""
<style>
.main-header {
    font-size: 2.8rem;
    color: #2e7d8b;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}

.chat-container {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    max-height: 400px;
    overflow-y: auto;
}

.user-message {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    border-left: 4px solid #2196f3;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 10px;
}

.assistant-message {
    background: linear-gradient(135deg, #f3e5f5, #e1bee7);
    border-left: 4px solid #9c27b0;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 10px;
}

.system-message {
    background: linear-gradient(135deg, #fff3e0, #ffecb3);
    border-left: 4px solid #ff9800;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 10px;
    font-style: italic;
}

.patient-info {
    background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
    border: 2px solid #4caf50;
    padding: 1.5rem;
    margin: 1rem 0;
    border-radius: 10px;
}

.progress-container {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}

.node-status {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    margin: 0.2rem;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: bold;
}

.node-completed {
    background: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}

.node-running {
    background: #cce7ff;
    color: #004085;
    border: 1px solid #99d6ff;
}

.node-error {
    background: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}

.node-pending {
    background: #f8f9fa;
    color: #6c757d;
    border: 1px solid #dee2e6;
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
}

.refresh-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(220, 53, 69, 0.3);
}

.example-message {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    padding: 0.8rem;
    margin: 0.3rem 0;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.2s;
}

.example-message:hover {
    background: #e9ecef;
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

# Main chat interface
st.markdown("## 💬 Chat Interface")

# Refresh Patient Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔄 Refresh Patient & Chat", use_container_width=True, type="secondary"):
        refresh_patient()

# Display current patient info if available
if st.session_state.current_patient:
    display_patient_info(st.session_state.current_patient)

# Display processing status if analysis is running
if st.session_state.processing_status:
    st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    display_node_status(st.session_state.processing_status)
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
            st.markdown(f'<div class="system-message">{message["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Show example commands when no messages
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

# Chat input
user_input = st.chat_input("💬 Type your message here...")

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
    st.markdown("---")
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

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🏥 <strong>Health Agent</strong> - MCP-Powered Chat Interface<br>
    Advanced health analysis through conversational AI and MCP tool integration<br>
    ⚠️ <em>This analysis is for informational purposes only and should not replace professional medical advice.</em>
</div>
""", unsafe_allow_html=True)
