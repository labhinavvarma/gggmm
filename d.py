import streamlit as st
import json
import pandas as pd
from datetime import datetime
import time
from typing import Dict, Any, List

# Import the chatbot-first agent
try:
    from chatbot_first_agent import ChatbotFirstHealthAgent, Config
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    import_error = str(e)

# Configure Streamlit page
st.set_page_config(
    page_title="🤖 Healthcare Analysis Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for chatbot interface
st.markdown("""
<style>
.main-title {
    font-size: 3rem;
    color: #2E8B57;
    text-align: center;
    margin-bottom: 0.5rem;
    font-weight: bold;
}

.subtitle {
    font-size: 1.2rem;
    color: #666;
    text-align: center;
    margin-bottom: 2rem;
}

.chat-container {
    background: #ffffff;
    border: 2px solid #007bff;
    border-radius: 1rem;
    padding: 1.5rem;
    margin: 1rem 0;
    min-height: 600px;
    max-height: 600px;
    overflow-y: auto;
}

.chat-message {
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    border-radius: 1rem;
    animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.chat-user {
    background: linear-gradient(45deg, #007bff, #0056b3);
    color: white;
    margin-left: 15%;
    border-bottom-right-radius: 0.3rem;
}

.chat-assistant {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    margin-right: 15%;
    border-bottom-left-radius: 0.3rem;
}

.chat-system {
    background: linear-gradient(45deg, #28a745, #1e7e34);
    color: white;
    text-align: center;
    border-radius: 0.5rem;
    font-weight: bold;
}

.chat-input-container {
    position: sticky;
    bottom: 0;
    background: white;
    padding: 1rem 0;
    border-top: 2px solid #dee2e6;
}

.example-commands {
    background: #e3f2fd;
    border: 1px solid #2196f3;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
}

.json-section {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 0.5rem;
    margin: 1rem 0;
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
    width: 100%;
}

.typing-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 1rem 1.5rem;
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 1rem;
    margin: 1rem 0;
    margin-right: 15%;
    border-bottom-left-radius: 0.3rem;
}

.typing-dots {
    display: flex;
    gap: 4px;
}

.typing-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #007bff;
    animation: typing 1.4s infinite;
}

.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-10px); }
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    if 'is_typing' not in st.session_state:
        st.session_state.is_typing = False
    if 'show_json_data' not in st.session_state:
        st.session_state.show_json_data = False

def refresh_everything():
    """Refresh all data and conversation"""
    # Clear chat history and analysis data
    st.session_state.chat_history = []
    st.session_state.current_analysis = None
    st.session_state.show_json_data = False
    
    # Reset agent session if agent exists
    if st.session_state.agent:
        st.session_state.agent.refresh_session()
    
    # Add welcome message back
    add_system_message("""🔄 **Session Refreshed!** 

I'm ready to help with new healthcare analysis. 

**Give me a command like:**
- "Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345"
- "Evaluate patient data for Sarah Johnson, born 1975-08-22, female, SSN 987654321, zip 90210"

**Ready to analyze patient data!** 🏥""")
    
    st.success("🔄 Session refreshed! Ready for new patient analysis.")
    st.rerun()

def add_system_message(message: str):
    """Add a system message to chat history"""
    st.session_state.chat_history.append({
        "role": "system",
        "content": message,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

def display_chat_message(message: Dict[str, Any]):
    """Display a single chat message"""
    role = message.get("role", "unknown")
    content = message.get("content", "")
    timestamp = message.get("timestamp", "")
    
    if role == "user":
        st.markdown(f"""
        <div class="chat-message chat-user">
            <strong>You ({timestamp}):</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    elif role == "assistant":
        st.markdown(f"""
        <div class="chat-message chat-assistant">
            <strong>🤖 Healthcare AI ({timestamp}):</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    elif role == "system":
        st.markdown(f"""
        <div class="chat-message chat-system">
            {content}
        </div>
        """, unsafe_allow_html=True)

def display_typing_indicator():
    """Display typing indicator"""
    st.markdown("""
    <div class="typing-indicator">
        <span>🤖 AI is thinking</span>
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_json_section(title: str, data: Dict[str, Any], key: str):
    """Display JSON data in collapsible expander"""
    with st.expander(f"📄 {title}", expanded=False):
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
        return
    
    st.markdown("### 🎯 Health Entity Extraction")
    
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
    medical_conditions = entities.get("medical_conditions", [])
    medications = entities.get("medications_identified", [])
    
    if medical_conditions or medications:
        col1, col2 = st.columns(2)
        
        with col1:
            if medical_conditions:
                st.markdown("**🏥 Medical Conditions:**")
                for condition in medical_conditions:
                    st.markdown(f"- {condition}")
        
        with col2:
            if medications:
                st.markdown("**💊 Medications:**")
                for med in medications:
                    st.markdown(f"- {med}")

def send_message(user_input: str):
    """Send message to the chatbot agent"""
    if not st.session_state.agent:
        st.session_state.chat_history.append({
            "role": "system",
            "content": "❌ Agent not initialized. Please initialize the agent first.",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        return
    
    if not user_input.strip():
        return
    
    # Add user message to chat history immediately
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input.strip(),
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    
    # Show typing indicator
    st.session_state.is_typing = True
    
    # Process message in try-catch to ensure typing indicator is cleared
    try:
        # Process message with agent
        with st.spinner("🤖 Processing..."):
            result = st.session_state.agent.chat(user_input.strip())
        
        # Add assistant response to chat history
        assistant_response = result.get("response", "I couldn't process your request.")
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": assistant_response,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Handle analysis completion
        if result.get("analysis_ready"):
            st.session_state.current_analysis = result
            st.session_state.show_json_data = True
            
            # Add analysis completion message
            analysis_info = result.get('entity_extraction', {})
            api_responses = result.get('raw_api_responses', {})
            successful_apis = len([k for k, v in api_responses.items() if v and not v.get('error')])
            
            completion_message = f"""📊 **Analysis Complete!** 

✅ **API Results:** {successful_apis}/5 MCP endpoints successful
🔒 **Data Processing:** Medical & pharmacy data deidentified
🎯 **Entities Extracted:** 
- Diabetes: {analysis_info.get('diabetes', 'unknown').title()}
- Age Group: {analysis_info.get('age_group', 'unknown').title()}  
- Blood Pressure: {analysis_info.get('blood_pressure', 'unknown').title()}

💬 **I can now answer detailed questions about the analysis data!**

Try asking:
- "How many medical claims were found?"
- "What medications were identified?"  
- "Show me the API status"
- "Count the pharmacy claims"
"""
            
            st.session_state.chat_history.append({
                "role": "system",
                "content": completion_message,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
        
        # Show any errors
        if result.get("errors"):
            for error in result["errors"]:
                st.session_state.chat_history.append({
                    "role": "system",
                    "content": f"⚠️ {error}",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
        
    except Exception as e:
        # Add error message to chat
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"❌ I encountered an error: {str(e)}",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    
    finally:
        # Always clear typing indicator
        st.session_state.is_typing = False

# Initialize session state
initialize_session_state()

# Main title
st.markdown('<h1 class="main-title">🤖 Healthcare Analysis Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Natural Language Healthcare Data Analysis • Give commands to analyze patient data</p>', unsafe_allow_html=True)

# Check agent availability
if not AGENT_AVAILABLE:
    st.error(f"❌ Chatbot agent not available: {import_error}")
    st.info("💡 Please ensure chatbot_first_agent.py is available and dependencies are installed.")
    st.stop()

# Top controls
col1, col2 = st.columns([3, 1])

with col1:
    # Initialize agent button
    if not st.session_state.agent:
        if st.button("🚀 Initialize Healthcare AI Agent", key="init_agent"):
            try:
                st.session_state.agent = ChatbotFirstHealthAgent()
                add_system_message("""✅ **Healthcare AI Agent Initialized!** 

I can analyze patient data through natural language commands and answer detailed questions about the results.

**Ready to process your commands!** Just tell me what you need. 🤖

**Example:** "Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345" """)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to initialize agent: {str(e)}")

with col2:
    # Refresh button
    if st.button("🔄 Refresh All", key="refresh_all"):
        refresh_everything()

# Agent status
if st.session_state.agent:
    st.success("✅ Healthcare AI Agent Ready")
else:
    st.warning("⚠️ Please initialize the agent to begin")
    st.stop()

# Add welcome message if chat is empty
if not st.session_state.chat_history:
    add_system_message("""🤖 **Welcome to Healthcare Analysis Chatbot!** 

I can analyze patient healthcare data and answer detailed questions about the results.

**🔄 How it works:**
1. **Give me a patient analysis command** (see examples below)
2. **I extract data from your command** using AI
3. **I call MCP server** to get medical and pharmacy data  
4. **I process and deidentify** the data for privacy
5. **You can ask specific questions** about the analysis results

**📝 Example analysis commands:**
- "Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345"

**💬 After analysis, ask me questions like:**
- "How many medical claims were found?"
- "Count the pharmacy claims"
- "What medications were identified?"
- "Show me the medical conditions"
- "What's the API status?"
- "Give me diabetes details"

**🔄 Use the Refresh button to start a new analysis.**

**Ready to analyze patient data!** 🏥""")

# Chat interface
st.markdown("### 💬 Healthcare Analysis Conversation")

# Chat container with proper scrolling
chat_container = st.container()
with chat_container:
    # Create a scrollable chat area
    chat_messages_placeholder = st.empty()
    
    with chat_messages_placeholder.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display all chat messages
        for i, message in enumerate(st.session_state.chat_history):
            display_chat_message(message)
        
        # Show typing indicator if agent is processing
        if st.session_state.is_typing:
            display_typing_indicator()
        
        st.markdown('</div>', unsafe_allow_html=True)

# Chat input
st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)

# Main chat input - moved to be more prominent
col1, col2, col3 = st.columns([6, 1, 1])

with col1:
    # Use session state for input to control clearing
    if 'user_input_key' not in st.session_state:
        st.session_state.user_input_key = 0
    
    user_input = st.text_input(
        "Type your command or question:",
        placeholder="e.g., Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345",
        key=f"chat_input_{st.session_state.user_input_key}",
        label_visibility="collapsed"
    )

with col2:
    send_clicked = st.button("📤 Send", key="send_button")

with col3:
    refresh_clicked = st.button("🔄 Refresh", key="refresh_button", help="Reset conversation and analysis")

# Handle button clicks
if send_clicked and user_input:
    send_message(user_input)
    # Clear input by incrementing key
    st.session_state.user_input_key += 1
    st.rerun()

if refresh_clicked:
    refresh_everything()

# Handle Enter key simulation (show instruction)
if user_input:
    st.caption("💡 Click 'Send' or press Ctrl+Enter to send your message")

# Example commands section - moved below input
with st.expander("💡 Example Commands & Quick Actions", expanded=False):
    # Show different examples based on whether we have analysis data
    if st.session_state.current_analysis:
        st.markdown("""
        **🔍 Ask Questions About Current Analysis:**
        - `How many medical claims were found?`
        - `Count the pharmacy claims`
        - `What medications were identified?`
        - `Show me the medical conditions`
        - `What's the API status?`
        - `Give me the diabetes details`
        - `Explain the blood pressure findings`
        """)
        
        # Contextual quick buttons
        st.markdown("**🚀 Quick Questions:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Medical Claims Count", key="medical_claims_q"):
                send_message("How many medical claims were found?")
                st.session_state.user_input_key += 1
                st.rerun()
        
        with col2:
            if st.button("💊 Pharmacy Claims Count", key="pharmacy_claims_q"):
                send_message("How many pharmacy claims were found?")
                st.session_state.user_input_key += 1
                st.rerun()
        
        with col3:
            if st.button("💊 Medications Found", key="medications_q"):
                send_message("What medications were identified?")
                st.session_state.user_input_key += 1
                st.rerun()
        
        with col4:
            if st.button("📡 API Status", key="api_status_q"):
                send_message("What's the API status?")
                st.session_state.user_input_key += 1
                st.rerun()
    
    else:
        st.markdown("""
        **📝 Patient Analysis Commands:**
        - `Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345`
        - `Evaluate health data for Sarah Johnson, born 1975-08-22, female, SSN 987654321, zip 90210`  
        - `Check patient Michael Brown, DOB 1990-12-05, male, SSN 111223333, zip 77001`
        
        **ℹ️ After analysis, you can ask:**
        - `How many medical claims were found?`
        - `Count the pharmacy claims`
        - `What medications were identified?`
        - `Show me the medical conditions`
        - `What's the API status?`
        """)
        
        # Quick command buttons for starting analysis
        st.markdown("**🚀 Quick Start:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("👤 Sample Patient Analysis", key="sample_analysis"):
                sample_command = "Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345"
                send_message(sample_command)
                st.session_state.user_input_key += 1
                st.rerun()
        
        with col2:
            if st.button("❓ What can you do?", key="help_command"):
                help_command = "What can you help me with?"
                send_message(help_command)
                st.session_state.user_input_key += 1
                st.rerun()
        
        with col3:
            if st.button("📋 System Capabilities", key="capabilities_command"):
                capabilities_command = "Tell me about your capabilities"
                send_message(capabilities_command)
                st.session_state.user_input_key += 1
                st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Display analysis results section - only show JSON data, not duplicated analysis
if st.session_state.show_json_data and st.session_state.current_analysis:
    st.markdown("---")
    st.markdown("## 📊 Analysis Data Available")
    st.info("💡 The analysis is complete. All data is available for the chatbot to reference. Ask questions in the chat above!")
    
    analysis = st.session_state.current_analysis
    
    # Analysis summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        success_status = "✅ Success" if analysis.get("success") else "⚠️ With Errors"
        st.metric("Analysis Status", success_status)
    
    with col2:
        raw_responses = analysis.get("raw_api_responses", {})
        api_calls = len([k for k, v in raw_responses.items() if v and not v.get("error")])
        st.metric("MCP API Calls", f"{api_calls}/5")
    
    with col3:
        entities = analysis.get("entity_extraction", {})
        conditions = len(entities.get("medical_conditions", []))
        st.metric("Conditions Found", conditions)
    
    with col4:
        medications = len(entities.get("medications_identified", []))
        st.metric("Medications Found", medications)
    
    # Raw JSON Data Section - Collapsed by default
    st.markdown("### 📄 Raw MCP Server JSON Responses")
    st.info("💡 All raw API responses from MCP server endpoints (collapsed by default):")
    
    raw_responses = analysis.get("raw_api_responses", {})
    if raw_responses:
        display_json_section("MCID Search Results", raw_responses.get("mcid", {}), "mcid")
        display_json_section("Medical API Results", raw_responses.get("medical", {}), "medical")
        display_json_section("Pharmacy API Results", raw_responses.get("pharmacy", {}), "pharmacy")
        display_json_section("Token API Results", raw_responses.get("token", {}), "token")
        display_json_section("All Endpoints Results", raw_responses.get("all", {}), "all")
    
    # Deidentified Data Section
    st.markdown("### 🔒 Deidentified Data")
    st.info("💬 The chatbot has access to this deidentified data. Ask questions about it in the chat above!")
    deidentified_data = analysis.get("deidentified_data", {})
    
    col1, col2 = st.columns(2)
    with col1:
        display_json_section("Deidentified Medical Data", deidentified_data.get("medical", {}), "deident_med")
    with col2:
        display_json_section("Deidentified Pharmacy Data", deidentified_data.get("pharmacy", {}), "deident_pharm")
    
    # Entity Extraction Results
    st.markdown("### 🎯 Entity Extraction Results")
    display_entity_cards(analysis.get("entity_extraction", {}))
    
    # Download section
    st.markdown("### 💾 Download Complete Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Complete JSON report
        complete_report = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "patient_info": analysis.get("patient_data", {}),
                "success": analysis.get("success", False),
                "session_id": analysis.get("session_id")
            },
            "raw_api_responses": analysis.get("raw_api_responses", {}),
            "deidentified_data": analysis.get("deidentified_data", {}),
            "entity_extraction": analysis.get("entity_extraction", {}),
            "conversation_history": st.session_state.chat_history,
            "errors": analysis.get("errors", [])
        }
        
        st.download_button(
            "📊 Complete JSON Report",
            json.dumps(complete_report, indent=2),
            f"healthcare_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # CSV summary
        try:
            entities = analysis.get("entity_extraction", {})
            csv_data = {
                "Metric": [
                    "Analysis Status", "API Calls Successful", "Conditions Found", "Medications Found",
                    "Diabetes", "Age Group", "Blood Pressure", "Smoking", "Alcohol", "Timestamp"
                ],
                "Value": [
                    "Success" if analysis.get("success") else "Failed",
                    f"{api_calls}/5",
                    conditions,
                    medications,
                    entities.get("diabetes", "unknown"),
                    entities.get("age_group", "unknown"),
                    entities.get("blood_pressure", "unknown"),
                    entities.get("smoking", "unknown"),
                    entities.get("alcohol", "unknown"),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            }
            
            csv_df = pd.DataFrame(csv_data)
            csv_string = csv_df.to_csv(index=False)
            
            st.download_button(
                "📊 CSV Summary",
                csv_string,
                f"healthcare_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error generating CSV: {str(e)}")
    
    with col3:
        # Text report
        patient_info = analysis.get("patient_data", {})
        patient_name = f"{patient_info.get('first_name', 'Unknown')} {patient_info.get('last_name', 'Unknown')}"
        
        text_report = f"""
HEALTHCARE ANALYSIS CHATBOT REPORT
=================================
Patient: {patient_name}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session ID: {analysis.get('session_id', 'Unknown')}
Status: {'Success' if analysis.get('success') else 'Failed'}

MCP SERVER API CALLS:
====================
Successful Calls: {api_calls}/5

ENTITY EXTRACTION:
=================
{json.dumps(entities, indent=2)}

CONVERSATION HISTORY:
====================
{chr(10).join([f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}" for msg in st.session_state.chat_history])}

ERRORS (if any):
===============
{chr(10).join(analysis.get('errors', []))}
"""
        
        st.download_button(
            "📝 Text Report",
            text_report,
            f"healthcare_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🤖 <strong>Healthcare Analysis Chatbot</strong><br>
    Natural Language Processing • MCP Server Integration • AI-Powered Analysis<br>
    ⚠️ <em>This analysis is for informational purposes only and should not replace professional medical advice.</em>
</div>
""", unsafe_allow_html=True)

# Sidebar info (collapsed by default)
with st.sidebar:
    st.header("ℹ️ System Info")
    
    if st.session_state.agent:
        st.success("🤖 Agent: Ready")
    else:
        st.error("🤖 Agent: Not initialized")
    
    if st.session_state.current_analysis:
        st.success("📊 Analysis: Available")
        session_id = st.session_state.current_analysis.get("session_id", "")
        if session_id:
            st.text(f"Session: {session_id[:12]}...")
    else:
        st.info("📊 Analysis: None")
    
    st.text(f"💬 Messages: {len(st.session_state.chat_history)}")
    
    st.markdown("---")
    st.markdown("### 🎯 How to Use")
    st.markdown("""
    1. **Initialize Agent** (if not done)
    2. **Give Command** like:
       "Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345"
    3. **View Results** - JSON data and analysis
    4. **Ask Questions** about the findings
    5. **Refresh All** to start new analysis
    """)
    
    if st.checkbox("🐛 Debug Mode"):
        st.text(f"Agent initialized: {st.session_state.agent is not None}")
        st.text(f"Show JSON: {st.session_state.show_json_data}")
        st.text(f"Is typing: {st.session_state.is_typing}")
        if st.session_state.current_analysis:
            st.text(f"Analysis success: {st.session_state.current_analysis.get('success')}")
