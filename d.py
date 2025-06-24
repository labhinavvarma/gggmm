import streamlit as st
import json
import pandas as pd
from datetime import datetime
import time
import uuid
from typing import Dict, Any, List

# Import the enhanced agent
try:
    from enhanced_langgraph_agent import EnhancedHealthAnalysisAgent, Config
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    import_error = str(e)

# Configure Streamlit page
st.set_page_config(
    page_title="🏥 Enhanced Healthcare Analysis with MCP Integration",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #2E8B57;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: bold;
}

.section-header {
    font-size: 1.8rem;
    color: #4682B4;
    border-left: 4px solid #4682B4;
    padding-left: 1rem;
    margin: 1.5rem 0;
    font-weight: bold;
}

.success-box {
    background: linear-gradient(135deg, #d4edda, #c3e6cb);
    border: 2px solid #28a745;
    color: #155724;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    font-weight: bold;
}

.error-box {
    background: linear-gradient(135deg, #f8d7da, #f5c6cb);
    border: 2px solid #dc3545;
    color: #721c24;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}

.info-box {
    background: linear-gradient(135deg, #cce7ff, #99d6ff);
    border: 2px solid #007bff;
    color: #004085;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}

.json-container {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 0.5rem 0;
    max-height: 400px;
    overflow-y: auto;
}

.chat-container {
    background: #ffffff;
    border: 2px solid #007bff;
    border-radius: 1rem;
    padding: 1rem;
    margin: 1rem 0;
    min-height: 400px;
}

.chat-message {
    padding: 0.5rem 1rem;
    margin: 0.5rem 0;
    border-radius: 1rem;
}

.chat-user {
    background: #007bff;
    color: white;
    margin-left: 20%;
}

.chat-assistant {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    margin-right: 20%;
}

.entity-card {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    border: 2px solid #6c757d;
    border-radius: 0.8rem;
    padding: 1rem;
    margin: 0.5rem;
    text-align: center;
}

.refresh-button {
    background: linear-gradient(45deg, #dc3545, #c82333);
    color: white;
    border: none;
    border-radius: 0.5rem;
    padding: 0.5rem 1rem;
    font-weight: bold;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    if 'raw_json_data' not in st.session_state:
        st.session_state.raw_json_data = {}
    if 'config' not in st.session_state:
        st.session_state.config = None

def refresh_everything():
    """Refresh all data and start fresh"""
    st.session_state.analysis_results = None
    st.session_state.analysis_running = False
    st.session_state.chat_history = []
    st.session_state.current_session_id = None
    st.session_state.raw_json_data = {}
    
    # Refresh agent session
    if st.session_state.agent:
        st.session_state.agent.refresh_session()
    
    st.success("🔄 All data refreshed! Ready for new analysis.")
    st.rerun()

def display_collapsible_json(title: str, data: Dict[str, Any], key: str, default_expanded: bool = False):
    """Display JSON data in a collapsible expander"""
    with st.expander(f"📄 {title}", expanded=default_expanded):
        if data and not data.get("error"):
            st.json(data)
            
            # Download button
            json_str = json.dumps(data, indent=2)
            st.download_button(
                f"💾 Download {title}",
                json_str,
                f"{title.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key=f"download_{key}"
            )
        else:
            error_msg = data.get("error", "No data available") if data else "No data available"
            st.error(f"❌ {error_msg}")

def display_entity_cards(entities: Dict[str, Any]):
    """Display entity extraction results as cards"""
    if not entities:
        st.warning("No entity extraction data available")
        return
    
    st.markdown("### 🎯 Health Entity Extraction Results")
    
    # Main health indicators
    col1, col2, col3, col4, col5 = st.columns(5)
    
    indicators = [
        ("diabetes", "🩺 Diabetes", col1),
        ("age_group", "👥 Age Group", col2),
        ("blood_pressure", "💓 Blood Pressure", col3),
        ("smoking", "🚬 Smoking", col4),
        ("alcohol", "🍷 Alcohol", col5)
    ]
    
    for key, title, col in indicators:
        with col:
            value = entities.get(key, "unknown")
            
            # Color coding
            if key == "diabetes" and value == "yes":
                color = "#dc3545"
                emoji = "⚠️"
            elif key in ["smoking", "alcohol"] and value in ["quit_attempt", "treatment"]:
                color = "#ffc107"
                emoji = "🟡"
            elif value == "unknown":
                color = "#6c757d"
                emoji = "❓"
            else:
                color = "#28a745"
                emoji = "✅"
            
            st.markdown(f"""
            <div class="entity-card" style="border-color: {color};">
                <h4 style="color: {color};">{emoji} {title}</h4>
                <p style="color: {color}; font-weight: bold;">{value.replace('_', ' ').upper()}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Additional findings
    col1, col2 = st.columns(2)
    
    with col1:
        medical_conditions = entities.get("medical_conditions", [])
        if medical_conditions:
            st.markdown("**🏥 Medical Conditions Found:**")
            for condition in medical_conditions:
                st.markdown(f"- {condition}")
    
    with col2:
        medications = entities.get("medications_identified", [])
        if medications:
            st.markdown("**💊 Medications Identified:**")
            for med in medications:
                st.markdown(f"- {med}")
    
    # Analysis details
    analysis_details = entities.get("analysis_details", [])
    if analysis_details:
        with st.expander("🔍 Analysis Details"):
            for detail in analysis_details:
                st.write(f"• {detail}")

def display_chatbot_interface():
    """Display the chatbot interface"""
    st.markdown('<div class="section-header">🤖 Ask Questions About the Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.analysis_results or not st.session_state.analysis_results.get("success"):
        st.warning("⚠️ Please run a successful analysis first to enable the chatbot.")
        return
    
    # Chat history display
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                timestamp = message.get("timestamp", "")
                
                if role == "user":
                    st.markdown(f"""
                    <div class="chat-message chat-user">
                        <strong>You ({timestamp.split('T')[1].split('.')[0] if timestamp else 'Unknown'}):</strong><br>
                        {content}
                    </div>
                    """, unsafe_allow_html=True)
                elif role == "assistant":
                    st.markdown(f"""
                    <div class="chat-message chat-assistant">
                        <strong>🤖 AI Assistant ({timestamp.split('T')[1].split('.')[0] if timestamp else 'Unknown'}):</strong><br>
                        {content}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("💬 No conversation yet. Ask a question about the analysis!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick question buttons
    st.markdown("**💡 Quick Questions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_questions = [
        "What medications were found?",
        "Explain the diabetes findings", 
        "What are the key health risks?",
        "Show me the blood pressure data"
    ]
    
    for i, (col, question) in enumerate(zip([col1, col2, col3, col4], quick_questions)):
        with col:
            if st.button(question, key=f"quick_q_{i}"):
                ask_chatbot_question(question)
    
    # Chat input
    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_question = st.text_input(
            "Ask a question about the analysis:",
            placeholder="e.g., What conditions were identified?",
            key="chat_input"
        )
    
    with col2:
        if st.button("🚀 Ask", key="ask_button"):
            if user_question.strip():
                ask_chatbot_question(user_question)
                st.rerun()

def ask_chatbot_question(question: str):
    """Ask a question to the chatbot"""
    if not st.session_state.agent:
        st.error("❌ No agent available")
        return
    
    try:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get response from agent
        with st.spinner("🤖 AI is thinking..."):
            response = st.session_state.agent.ask_chatbot(question)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Clear input
        if 'chat_input' in st.session_state:
            st.session_state.chat_input = ""
            
    except Exception as e:
        st.error(f"❌ Error asking question: {str(e)}")

# Initialize session state
initialize_session_state()

# Main header
st.markdown('<h1 class="main-header">🏥 Enhanced Healthcare Analysis with MCP Integration</h1>', unsafe_allow_html=True)

# Top-level controls
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.markdown("**🔗 MCP Server Integration • 🤖 LangGraph Workflow • 💬 AI Chatbot**")

with col2:
    if st.session_state.analysis_results:
        success_status = "✅ Complete" if st.session_state.analysis_results.get("success") else "⚠️ With Errors"
        st.markdown(f"**Analysis Status:** {success_status}")

with col3:
    if st.button("🔄 Refresh All", key="refresh_button", help="Clear all data and start fresh"):
        refresh_everything()

# Check agent availability
if not AGENT_AVAILABLE:
    st.error(f"❌ Enhanced LangGraph Agent not available: {import_error}")
    st.info("💡 Please ensure the enhanced_langgraph_agent.py file is available and all dependencies are installed.")
    st.stop()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # MCP Server Configuration
    st.subheader("🔗 MCP Server Settings")
    mcp_server_url = st.text_input("MCP Server URL", value="http://localhost:8000")
    
    # Snowflake Configuration
    st.subheader("❄️ Snowflake Cortex")
    snowflake_model = st.selectbox("Model", ["llama3.1-70b", "llama3.1-8b"], index=0)
    
    # Initialize agent
    if st.button("🔧 Initialize Agent"):
        try:
            config = Config(
                mcp_server_url=mcp_server_url,
                model=snowflake_model
            )
            st.session_state.config = config
            st.session_state.agent = EnhancedHealthAnalysisAgent(config)
            st.success("✅ Agent initialized successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Agent initialization failed: {str(e)}")
    
    # Agent status
    st.markdown("---")
    st.subheader("📊 Status")
    
    agent_status = "🟢 Ready" if st.session_state.agent else "🔴 Not Initialized"
    st.markdown(f"**Agent:** {agent_status}")
    
    if st.session_state.analysis_results:
        session_id = st.session_state.analysis_results.get("session_id", "None")
        st.markdown(f"**Session ID:** {session_id[:12]}..." if session_id and session_id != "None" else "**Session ID:** None")
    
    analysis_status = "🟢 Available" if st.session_state.analysis_results else "⚪ None"
    st.markdown(f"**Analysis:** {analysis_status}")
    
    chat_messages = len(st.session_state.chat_history)
    st.markdown(f"**Chat Messages:** {chat_messages}")

# Main content area
if not st.session_state.agent:
    st.info("🔧 Please initialize the agent in the sidebar to begin.")
    st.stop()

# Patient Input Form
st.markdown('<div class="section-header">👤 Patient Information</div>', unsafe_allow_html=True)

with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        first_name = st.text_input("First Name *", value="John")
        last_name = st.text_input("Last Name *", value="Smith")
    
    with col2:
        ssn = st.text_input("SSN *", value="123456789")
        date_of_birth = st.date_input("Date of Birth *", value=datetime(1980, 1, 15))
    
    with col3:
        gender = st.selectbox("Gender *", ["M", "F"])
        zip_code = st.text_input("Zip Code *", value="12345")
    
    # Submit button
    submitted = st.form_submit_button(
        "🚀 Run Complete Analysis with MCP Integration",
        disabled=st.session_state.analysis_running
    )

# Run Analysis
if submitted and not st.session_state.analysis_running:
    patient_data = {
        "first_name": first_name.strip(),
        "last_name": last_name.strip(),
        "ssn": ssn.strip(),
        "date_of_birth": date_of_birth.strftime("%Y-%m-%d"),
        "gender": gender,
        "zip_code": zip_code.strip()
    }
    
    st.session_state.analysis_running = True
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🚀 Initializing Enhanced LangGraph workflow...")
        progress_bar.progress(10)
        time.sleep(0.5)
        
        status_text.text("📡 Fetching data from MCP server (MCID, Medical, Pharmacy, Token, All)...")
        progress_bar.progress(30)
        time.sleep(1)
        
        status_text.text("🔒 Deidentifying medical and pharmacy data...")
        progress_bar.progress(50)
        time.sleep(0.5)
        
        status_text.text("🎯 Extracting health entities (diabetes, BP, smoking, alcohol)...")
        progress_bar.progress(70)
        time.sleep(0.5)
        
        status_text.text("🤖 Setting up AI chatbot with analysis context...")
        progress_bar.progress(90)
        time.sleep(0.5)
        
        # Run the analysis
        results = st.session_state.agent.run_analysis(patient_data)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis completed successfully!")
        
        st.session_state.analysis_results = results
        st.session_state.current_session_id = results.get("session_id")
        
        # Store chat history from results
        if results.get("chat_history"):
            st.session_state.chat_history = results["chat_history"]
        
        if results.get("success"):
            st.markdown('<div class="success-box">🔥 Enhanced health analysis completed successfully!</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Analysis completed with some errors. Check results below.")
            
    except Exception as e:
        progress_bar.progress(0)
        status_text.text("❌ Analysis failed")
        st.error(f"❌ Error in analysis: {str(e)}")
        
    finally:
        st.session_state.analysis_running = False

# Display Results
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    # Success/Error Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        success_status = "✅ Success" if results.get("success") else "⚠️ With Errors"
        st.metric("Analysis Status", success_status)
    
    with col2:
        step_status = results.get("step_status", {})
        completed_steps = len([s for s in step_status.values() if s == "completed"])
        total_steps = 4  # fetch_mcp_data, deidentify_data, extract_entities, setup_chatbot
        st.metric("Workflow Steps", f"{completed_steps}/{total_steps}")
    
    with col3:
        raw_data = results.get("raw_api_data", {})
        api_calls = len([k for k, v in raw_data.items() if v and not v.get("error")])
        st.metric("API Calls", f"{api_calls}/5")
    
    with col4:
        entities = results.get("entity_extraction", {})
        conditions_found = len(entities.get("medical_conditions", []))
        st.metric("Conditions Found", conditions_found)
    
    # Display Errors (if any)
    errors = results.get("errors", [])
    if errors:
        st.markdown('<div class="section-header">❌ Errors</div>', unsafe_allow_html=True)
        for error in errors:
            st.error(f"• {error}")
    
    # Raw JSON Data Section
    st.markdown('<div class="section-header">📄 Raw MCP API Data (Collapsible)</div>', unsafe_allow_html=True)
    st.info("💡 All raw JSON responses from MCP server endpoints are displayed below (collapsed by default)")
    
    # Individual API endpoint data
    display_collapsible_json("MCID Search Results", results.get("mcid_raw", {}), "mcid")
    display_collapsible_json("Medical API Results", results.get("medical_raw", {}), "medical")
    display_collapsible_json("Pharmacy API Results", results.get("pharmacy_raw", {}), "pharmacy")
    display_collapsible_json("Token API Results", results.get("token_raw", {}), "token")
    display_collapsible_json("All Endpoints Results", results.get("all_raw", {}), "all")
    
    # Deidentified Data Section
    st.markdown('<div class="section-header">🔒 Deidentified Data</div>', unsafe_allow_html=True)
    
    deidentified_data = results.get("deidentified_data", {})
    if deidentified_data:
        col1, col2 = st.columns(2)
        
        with col1:
            display_collapsible_json("Deidentified Medical Data", deidentified_data.get("medical", {}), "deident_med")
        
        with col2:
            display_collapsible_json("Deidentified Pharmacy Data", deidentified_data.get("pharmacy", {}), "deident_pharm")
    
    # Entity Extraction Results
    st.markdown('<div class="section-header">🎯 Enhanced Entity Extraction</div>', unsafe_allow_html=True)
    entities = results.get("entity_extraction", {})
    display_entity_cards(entities)
    
    # Chatbot Interface
    display_chatbot_interface()
    
    # Download Complete Report
    st.markdown('<div class="section-header">💾 Download Complete Report</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        complete_report = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "patient_info": results.get("patient_data", {}),
                "success": results.get("success", False),
                "session_id": results.get("session_id"),
                "workflow_steps": results.get("step_status", {})
            },
            "raw_api_data": results.get("raw_api_data", {}),
            "deidentified_data": results.get("deidentified_data", {}),
            "entity_extraction": results.get("entity_extraction", {}),
            "chat_history": st.session_state.chat_history,
            "errors": results.get("errors", [])
        }
        
        st.download_button(
            "📊 Download Complete Analysis Report",
            json.dumps(complete_report, indent=2),
            f"healthcare_analysis_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Text summary
        patient_name = f"{results.get('patient_data', {}).get('first_name', 'Unknown')} {results.get('patient_data', {}).get('last_name', 'Unknown')}"
        
        text_summary = f"""
ENHANCED HEALTHCARE ANALYSIS REPORT WITH MCP INTEGRATION
=======================================================
Patient: {patient_name}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session ID: {results.get('session_id', 'Unknown')}
Status: {'Success' if results.get('success', False) else 'Failed'}

MCP API ENDPOINTS CALLED:
========================
- MCID Search: {'✅' if results.get('mcid_raw') and not results.get('mcid_raw', {}).get('error') else '❌'}
- Medical API: {'✅' if results.get('medical_raw') and not results.get('medical_raw', {}).get('error') else '❌'}
- Pharmacy API: {'✅' if results.get('pharmacy_raw') and not results.get('pharmacy_raw', {}).get('error') else '❌'}
- Token API: {'✅' if results.get('token_raw') and not results.get('token_raw', {}).get('error') else '❌'}
- All Endpoints: {'✅' if results.get('all_raw') and not results.get('all_raw', {}).get('error') else '❌'}

ENTITY EXTRACTION RESULTS:
==========================
{json.dumps(entities, indent=2)}

CHAT CONVERSATION:
==================
{chr(10).join([f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}" for msg in st.session_state.chat_history])}

ERRORS (if any):
================
{chr(10).join(results.get('errors', []))}
"""
        
        st.download_button(
            "📝 Download Text Summary",
            text_summary,
            f"healthcare_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with col3:
        # CSV summary
        try:
            csv_data = {
                "Metric": [
                    "Analysis Status", "Session ID", "Workflow Steps Completed",
                    "API Calls Successful", "Medical Conditions Found", "Medications Identified",
                    "Diabetes", "Age Group", "Blood Pressure", "Smoking", "Alcohol",
                    "Chat Messages", "Timestamp"
                ],
                "Value": [
                    "Success" if results.get("success", False) else "Failed",
                    results.get("session_id", "Unknown")[:12] + "..." if results.get("session_id") else "Unknown",
                    f"{completed_steps}/{total_steps}",
                    f"{api_calls}/5",
                    len(entities.get("medical_conditions", [])),
                    len(entities.get("medications_identified", [])),
                    entities.get("diabetes", "unknown"),
                    entities.get("age_group", "unknown"),
                    entities.get("blood_pressure", "unknown"),
                    entities.get("smoking", "unknown"),
                    entities.get("alcohol", "unknown"),
                    len(st.session_state.chat_history),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            }
            
            csv_df = pd.DataFrame(csv_data)
            csv_string = csv_df.to_csv(index=False)
            
            st.download_button(
                "📊 Download CSV Summary",
                csv_string,
                f"healthcare_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error generating CSV: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🔥 <strong>Enhanced Healthcare Analysis with MCP Integration</strong><br>
    🔗 MCP Server • 🤖 LangGraph Workflow • 🔒 Data Deidentification • 💬 AI Chatbot<br>
    ⚠️ <em>This analysis is for informational purposes only and should not replace professional medical advice.</em>
</div>
""", unsafe_allow_html=True)

# Debug information
if st.sidebar.checkbox("🐛 Show Debug Info"):
    st.sidebar.markdown("### 🔧 Debug Information")
    st.sidebar.write("Agent Available:", AGENT_AVAILABLE)
    st.sidebar.write("Agent Initialized:", st.session_state.agent is not None)
    st.sidebar.write("Analysis Results:", st.session_state.analysis_results is not None)
    st.sidebar.write("Chat History Length:", len(st.session_state.chat_history))
    st.sidebar.write("Current Session ID:", st.session_state.current_session_id)
    
    if st.session_state.analysis_results:
        st.sidebar.json(st.session_state.analysis_results.get("step_status", {}))
