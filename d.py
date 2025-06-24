import streamlit as st
import json
import pandas as pd
from datetime import datetime
import time
from typing import Dict, Any, List

# Import the enhanced RAG agent
try:
    from enhanced_rag_chatbot_agent import EnhancedRAGHealthAgent, Config
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    import_error = str(e)

# Configure Streamlit page
st.set_page_config(
    page_title="🧠 RAG Healthcare Analysis Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for RAG mode visualization
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

.rag-mode-banner {
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    color: white;
    padding: 1rem;
    border-radius: 1rem;
    text-align: center;
    font-weight: bold;
    font-size: 1.2rem;
    margin: 1rem 0;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.8; }
    100% { opacity: 1; }
}

.normal-mode-banner {
    background: linear-gradient(45deg, #007bff, #0056b3);
    color: white;
    padding: 1rem;
    border-radius: 1rem;
    text-align: center;
    font-weight: bold;
    font-size: 1.2rem;
    margin: 1rem 0;
}

.knowledge-base-status {
    background: #e8f5e8;
    border: 2px solid #28a745;
    border-radius: 0.8rem;
    padding: 1rem;
    margin: 1rem 0;
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

.rag-chat-container {
    background: #ffffff;
    border: 3px solid #ff6b6b;
    border-radius: 1rem;
    padding: 1.5rem;
    margin: 1rem 0;
    min-height: 600px;
    max-height: 600px;
    overflow-y: auto;
    box-shadow: 0 0 20px rgba(255, 107, 107, 0.3);
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

.chat-assistant-rag {
    background: linear-gradient(135deg, #fff5f5, #ffe6e6);
    border: 2px solid #ff6b6b;
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

.chat-system-rag {
    background: linear-gradient(45deg, #ff6b6b, #ff8e8e);
    color: white;
    text-align: center;
    border-radius: 0.5rem;
    font-weight: bold;
}

.rag-quick-actions {
    background: linear-gradient(135deg, #fff0f0, #ffe0e0);
    border: 2px solid #ff6b6b;
    border-radius: 1rem;
    padding: 1rem;
    margin: 1rem 0;
}

.mode-indicator {
    position: fixed;
    top: 10px;
    right: 10px;
    z-index: 1000;
    padding: 0.5rem 1rem;
    border-radius: 2rem;
    font-weight: bold;
    font-size: 0.9rem;
}

.mode-normal {
    background: #007bff;
    color: white;
}

.mode-rag {
    background: #ff6b6b;
    color: white;
    animation: blink 1s infinite;
}

@keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0.7; }
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

.rag-stats {
    background: linear-gradient(135deg, #f0f8ff, #e6f3ff);
    border: 2px solid #4ecdc4;
    border-radius: 0.8rem;
    padding: 1rem;
    margin: 1rem 0;
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
    if 'rag_mode' not in st.session_state:
        st.session_state.rag_mode = False
    if 'rag_knowledge_base' not in st.session_state:
        st.session_state.rag_knowledge_base = None

def refresh_everything():
    """Enhanced refresh to properly reset RAG mode"""
    # Clear all session state
    st.session_state.chat_history = []
    st.session_state.current_analysis = None
    st.session_state.show_json_data = False
    st.session_state.rag_mode = False
    st.session_state.rag_knowledge_base = None
    
    # Reset agent session if agent exists
    if st.session_state.agent:
        st.session_state.agent.refresh_session()
    
    # Add welcome message
    add_system_message("""🔄 **Session Refreshed - RAG Mode Deactivated!** 

I'm ready to help with new healthcare analysis. 

**Give me a command like:**
- "Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345"

**After analysis, I'll automatically enter RAG mode to answer detailed questions!**

**Ready to analyze patient data!** 🏥""", is_rag=False)
    
    st.success("🔄 Session refreshed! RAG mode deactivated. Ready for new patient analysis.")
    st.rerun()

def add_system_message(message: str, is_rag: bool = False):
    """Add a system message to chat history"""
    st.session_state.chat_history.append({
        "role": "system",
        "content": message,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "is_rag": is_rag
    })

def display_chat_message(message: Dict[str, Any], rag_mode: bool = False):
    """Display a single chat message with RAG mode awareness"""
    role = message.get("role", "unknown")
    content = message.get("content", "")
    timestamp = message.get("timestamp", "")
    is_rag_message = message.get("is_rag", False)
    
    if role == "user":
        st.markdown(f"""
        <div class="chat-message chat-user">
            <strong>You ({timestamp}):</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    elif role == "assistant":
        css_class = "chat-assistant-rag" if (rag_mode or is_rag_message) else "chat-assistant"
        icon = "🧠" if (rag_mode or is_rag_message) else "🤖"
        mode_text = "RAG AI" if (rag_mode or is_rag_message) else "Healthcare AI"
        
        st.markdown(f"""
        <div class="chat-message {css_class}">
            <strong>{icon} {mode_text} ({timestamp}):</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    elif role == "system":
        css_class = "chat-system-rag" if (rag_mode or is_rag_message) else "chat-system"
        st.markdown(f"""
        <div class="chat-message {css_class}">
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

def display_mode_banner():
    """Display current mode banner"""
    if st.session_state.rag_mode:
        st.markdown("""
        <div class="rag-mode-banner">
            🧠 RAG MODE ACTIVE - Answering questions based on patient analysis data
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="normal-mode-banner">
            🤖 ANALYSIS MODE - Ready to analyze patient data
        </div>
        """, unsafe_allow_html=True)

def display_rag_status():
    """Display RAG knowledge base status"""
    if st.session_state.agent:
        rag_status = st.session_state.agent.get_rag_status()
        
        if rag_status["rag_active"]:
            st.markdown(f"""
            <div class="knowledge-base-status">
                <h4>🧠 RAG Knowledge Base Status</h4>
                <p><strong>Status:</strong> ✅ Active and Ready</p>
                <p><strong>Data Size:</strong> {rag_status['knowledge_base_size']:,} characters</p>
                <p><strong>Session ID:</strong> {rag_status['session_id'][:12] if rag_status['session_id'] else 'None'}...</p>
                <p><strong>Capabilities:</strong> Can answer detailed questions about patient analysis data</p>
            </div>
            """, unsafe_allow_html=True)

def display_rag_quick_actions():
    """Display RAG mode quick action buttons"""
    if not st.session_state.rag_mode:
        return
    
    st.markdown("""
    <div class="rag-quick-actions">
        <h4>🔍 RAG Mode Quick Questions</h4>
        <p>Click any button below to ask common questions about the analysis data:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Medical Claims", key="rag_medical_claims", help="Count medical claims found"):
            send_message("How many medical claims were found?")
            st.session_state.user_input_key += 1
            st.rerun()
    
    with col2:
        if st.button("💊 Pharmacy Claims", key="rag_pharmacy_claims", help="Count pharmacy claims found"):
            send_message("How many pharmacy claims were found?")
            st.session_state.user_input_key += 1
            st.rerun()
    
    with col3:
        if st.button("💊 Medications", key="rag_medications", help="List identified medications"):
            send_message("What medications were identified?")
            st.session_state.user_input_key += 1
            st.rerun()
    
    with col4:
        if st.button("🏥 Conditions", key="rag_conditions", help="Show medical conditions"):
            send_message("Show me the medical conditions")
            st.session_state.user_input_key += 1
            st.rerun()
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        if st.button("📡 API Status", key="rag_api_status", help="Check MCP server status"):
            send_message("What's the API status?")
            st.session_state.user_input_key += 1
            st.rerun()
    
    with col6:
        if st.button("🎯 Entities", key="rag_entities", help="Show entity extraction results"):
            send_message("Give me the entity extraction results")
            st.session_state.user_input_key += 1
            st.rerun()
    
    with col7:
        if st.button("🩺 Diabetes", key="rag_diabetes", help="Diabetes-related findings"):
            send_message("Tell me about diabetes findings")
            st.session_state.user_input_key += 1
            st.rerun()
    
    with col8:
        if st.button("📋 Summary", key="rag_summary", help="Overall analysis summary"):
            send_message("Give me an overall summary of the analysis")
            st.session_state.user_input_key += 1
            st.rerun()

def send_message(user_input: str):
    """Send message to the enhanced RAG agent"""
    if not st.session_state.agent:
        st.session_state.chat_history.append({
            "role": "system",
            "content": "❌ Agent not initialized. Please initialize the agent first.",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "is_rag": False
        })
        return
    
    if not user_input.strip():
        return
    
    # Add user message to chat history immediately
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input.strip(),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "is_rag": st.session_state.rag_mode
    })
    
    # Show typing indicator
    st.session_state.is_typing = True
    
    try:
        # Process message with enhanced RAG agent
        with st.spinner("🤖 Processing..."):
            result = st.session_state.agent.chat(user_input.strip())
        
        # Update RAG mode state
        st.session_state.rag_mode = result.get("rag_mode", False)
        st.session_state.rag_knowledge_base = result.get("rag_knowledge_base", {})
        
        # Add assistant response to chat history
        assistant_response = result.get("response", "I couldn't process your request.")
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": assistant_response,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "is_rag": result.get("rag_mode", False)
        })
        
        # Handle analysis completion and RAG mode activation
        if result.get("analysis_ready") and result.get("rag_mode"):
            st.session_state.current_analysis = result
            st.session_state.show_json_data = True
            
            # Add RAG mode activation message
            rag_activation_message = f"""🧠 **RAG MODE ACTIVATED!** 

✅ **Knowledge Base Created:** Your patient analysis data is now my knowledge base
🔍 **Enhanced Capabilities:** I can answer detailed questions about:
- Medical claims counts and details
- Pharmacy claims and medications
- Health conditions and risk factors  
- API status and data quality
- Entity extraction results
- Cross-data analysis and insights

**I'm now a specialized RAG chatbot for this patient's analysis!**

Try the quick action buttons below or ask me anything about the analysis data.

🔄 **Use Refresh to exit RAG mode and analyze a new patient.**"""
            
            add_system_message(rag_activation_message, is_rag=True)
        
        # Show any errors
        if result.get("errors"):
            for error in result["errors"]:
                st.session_state.chat_history.append({
                    "role": "system",
                    "content": f"⚠️ {error}",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "is_rag": result.get("rag_mode", False)
                })
        
    except Exception as e:
        # Add error message to chat
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"❌ I encountered an error: {str(e)}",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "is_rag": False
        })
    
    finally:
        # Always clear typing indicator
        st.session_state.is_typing = False

# Initialize session state
initialize_session_state()

# Mode indicator (fixed position)
if st.session_state.rag_mode:
    st.markdown("""
    <div class="mode-indicator mode-rag">
        🧠 RAG MODE
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="mode-indicator mode-normal">
        🤖 ANALYSIS MODE
    </div>
    """, unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-title">🧠 RAG Healthcare Analysis Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Retrieval-Augmented Generation • Context-Aware Healthcare Data Analysis</p>', unsafe_allow_html=True)

# Check agent availability
if not AGENT_AVAILABLE:
    st.error(f"❌ Enhanced RAG agent not available: {import_error}")
    st.info("💡 Please ensure enhanced_rag_chatbot_agent.py is available and dependencies are installed.")
    st.stop()

# Display mode banner
display_mode_banner()

# Top controls
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    # Initialize agent button
    if not st.session_state.agent:
        if st.button("🚀 Initialize Enhanced RAG Agent", key="init_agent"):
            try:
                st.session_state.agent = EnhancedRAGHealthAgent()
                add_system_message("""✅ **Enhanced RAG Healthcare Agent Initialized!** 

I can analyze patient data and automatically enter RAG mode to answer detailed questions using the analysis results as my knowledge base.

**🔄 How it works:**
1. **Analysis Mode:** Give me patient data to analyze
2. **RAG Mode:** I become a specialized chatbot using your analysis data
3. **Knowledge-Based Answers:** Ask detailed questions about the results

**Ready to process your commands!** 🤖

**Example:** "Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345" """)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to initialize agent: {str(e)}")

with col2:
    # RAG status indicator
    if st.session_state.agent and st.session_state.rag_mode:
        st.success("🧠 RAG Active")
    elif st.session_state.agent:
        st.info("🤖 Analysis Ready")
    else:
        st.warning("⚠️ Agent Not Init")

with col3:
    # Enhanced refresh button
    if st.button("🔄 Refresh & Exit RAG", key="refresh_all"):
        refresh_everything()

# Agent status
if st.session_state.agent:
    st.success("✅ Enhanced RAG Healthcare Agent Ready")
    # Display RAG status if active
    if st.session_state.rag_mode:
        display_rag_status()
else:
    st.warning("⚠️ Please initialize the agent to begin")
    st.stop()

# Add welcome message if chat is empty
if not st.session_state.chat_history:
    add_system_message("""🧠 **Welcome to Enhanced RAG Healthcare Analysis Chatbot!** 

I'm an advanced AI that combines healthcare data analysis with RAG (Retrieval-Augmented Generation) capabilities.

**🔄 How I work:**
1. **Analysis Phase:** Give me patient data and I'll fetch & analyze it from MCP server
2. **RAG Activation:** I automatically enter RAG mode using the analysis as my knowledge base
3. **Expert Q&A:** Ask me detailed questions - I'll answer using the specific patient data

**📝 Start with a patient analysis command:**
- "Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345"

**🧠 Once in RAG mode, ask me:**
- "How many medical claims were found?"
- "What medications were identified?"
- "Show me the API status"
- "Count pharmacy claims"
- "Give me diabetes details"
- "What conditions were found?"

**🔄 Use Refresh to exit RAG mode and start fresh.**

**Ready to analyze and enter RAG mode!** 🏥""")

# RAG Quick Actions (only show in RAG mode)
if st.session_state.rag_mode:
    display_rag_quick_actions()

# Chat interface
st.markdown("### 💬 Healthcare Analysis Conversation")

# Chat container with RAG mode styling
chat_container = st.container()
with chat_container:
    chat_messages_placeholder = st.empty()
    
    with chat_messages_placeholder.container():
        # Use different container styling based on RAG mode
        container_class = "rag-chat-container" if st.session_state.rag_mode else "chat-container"
        st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
        
        # Display all chat messages with RAG mode awareness
        for message in st.session_state.chat_history:
            display_chat_message(message, st.session_state.rag_mode)
        
        # Show typing indicator if agent is processing
        if st.session_state.is_typing:
            display_typing_indicator()
        
        st.markdown('</div>', unsafe_allow_html=True)

# Chat input section
st.markdown("### 💬 Chat Input")

# Main chat input
col1, col2, col3 = st.columns([6, 1, 1])

with col1:
    # Use session state for input to control clearing
    if 'user_input_key' not in st.session_state:
        st.session_state.user_input_key = 0
    
    # Change placeholder based on mode
    if st.session_state.rag_mode:
        placeholder = "Ask me about the analysis data: e.g., How many medical claims were found?"
    else:
        placeholder = "e.g., Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345"
    
    user_input = st.text_input(
        "Type your command or question:",
        placeholder=placeholder,
        key=f"chat_input_{st.session_state.user_input_key}",
        label_visibility="collapsed"
    )

with col2:
    send_clicked = st.button("📤 Send", key="send_button")

with col3:
    refresh_clicked = st.button("🔄 Exit RAG", key="refresh_button", help="Exit RAG mode and reset")

# Handle button clicks
if send_clicked and user_input:
    send_message(user_input)
    st.session_state.user_input_key += 1
    st.rerun()

if refresh_clicked:
    refresh_everything()

# Handle Enter key instruction
if user_input:
    if st.session_state.rag_mode:
        st.caption("💡 Ask me anything about the patient analysis data!")
    else:
        st.caption("💡 Give me a patient analysis command to start")

# Example commands section
with st.expander("💡 Example Commands & RAG Mode Guide", expanded=False):
    if st.session_state.rag_mode:
        st.markdown("""
        **🧠 You're in RAG Mode! Ask me detailed questions:**
        
        **📊 Data Analysis Questions:**
        - `How many medical claims were found?`
        - `Count the pharmacy claims`
        - `What's the total number of records?`
        
        **💊 Medical & Pharmacy Questions:**
        - `What medications were identified?`
        - `Show me the medical conditions`
        - `List all diabetes-related findings`
        - `What pharmacy data is available?`
        
        **🔍 Technical Questions:**
        - `What's the API status?`
        - `Show me entity extraction results`
        - `What endpoints were successful?`
        - `Give me the data quality summary`
        
        **📋 Analysis Questions:**
        - `Summarize the key findings`
        - `What are the health risk factors?`
        - `Compare medical vs pharmacy data`
        - `What patterns do you see?`
        """)
        
        st.info("🧠 **RAG Mode**: I'm answering based on the specific patient analysis data loaded in my knowledge base.")
        
    else:
        st.markdown("""
        **📝 Patient Analysis Commands (to enter RAG mode):**
        - `Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345`
        - `Evaluate health data for Sarah Johnson, born 1975-08-22, female, SSN 987654321, zip 90210`  
        - `Check patient Michael Brown, DOB 1990-12-05, male, SSN 111223333, zip 77001`
        
        **🤖 General Questions:**
        - `What can you help me with?`
        - `What are your capabilities?`
        - `How does RAG mode work?`
        
        **🧠 After analysis → Automatic RAG mode activation:**
        - I'll use your analysis data as my knowledge base
        - Ask detailed questions about the specific patient
        - Get precise answers based on the actual data
        """)
        
        # Quick command buttons for starting analysis
        st.markdown("**🚀 Quick Start Commands:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("👤 Sample Analysis", key="sample_analysis"):
                sample_command = "Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345"
                send_message(sample_command)
                st.session_state.user_input_key += 1
                st.rerun()
        
        with col2:
            if st.button("❓ Capabilities", key="help_command"):
                help_command = "What can you help me with?"
                send_message(help_command)
                st.session_state.user_input_key += 1
                st.rerun()
        
        with col3:
            if st.button("🧠 RAG Info", key="rag_info_command"):
                rag_command = "How does your RAG mode work?"
                send_message(rag_command)
                st.session_state.user_input_key += 1
                st.rerun()

# Display analysis results section (only show if analysis available)
if st.session_state.show_json_data and st.session_state.current_analysis:
    st.markdown("---")
    st.markdown("## 📊 Analysis Data & RAG Knowledge Base")
    
    # RAG Statistics
    if st.session_state.rag_mode:
        st.markdown("""
        <div class="rag-stats">
            <h4>🧠 RAG Knowledge Base Statistics</h4>
            <p>This data is actively being used to answer your questions in RAG mode.</p>
        </div>
        """, unsafe_allow_html=True)
    
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
    
    # RAG Knowledge Base Info
    if st.session_state.rag_mode:
        st.info("🧠 **RAG Mode Active**: The chatbot above is using this analysis data as its knowledge base. Ask questions about any aspect of this data!")
    
    # Collapsible data sections
    with st.expander("📄 Raw MCP Server JSON Responses", expanded=False):
        raw_responses = analysis.get("raw_api_responses", {})
        if raw_responses:
            tabs = st.tabs(["MCID", "Medical", "Pharmacy", "Token", "All"])
            
            with tabs[0]:
                st.json(raw_responses.get("mcid", {}))
            with tabs[1]:
                st.json(raw_responses.get("medical", {}))
            with tabs[2]:
                st.json(raw_responses.get("pharmacy", {}))
            with tabs[3]:
                st.json(raw_responses.get("token", {}))
            with tabs[4]:
                st.json(raw_responses.get("all", {}))
    
    with st.expander("🔒 Deidentified Data (RAG Knowledge Base)", expanded=False):
        deidentified_data = analysis.get("deidentified_data", {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Medical Data")
            st.json(deidentified_data.get("medical", {}))
        with col2:
            st.subheader("Pharmacy Data")
            st.json(deidentified_data.get("pharmacy", {}))
    
    with st.expander("🎯 Entity Extraction Results", expanded=False):
        st.json(analysis.get("entity_extraction", {}))
    
    # Download section
    st.markdown("### 💾 Download Analysis Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Complete analysis report
        complete_report = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "patient_info": analysis.get("patient_data", {}),
                "success": analysis.get("success", False),
                "rag_mode": analysis.get("rag_mode", False),
                "session_id": analysis.get("session_id")
            },
            "raw_api_responses": analysis.get("raw_api_responses", {}),
            "deidentified_data": analysis.get("deidentified_data", {}),
            "entity_extraction": analysis.get("entity_extraction", {}),
            "rag_knowledge_base": analysis.get("rag_knowledge_base", {}),
            "conversation_history": st.session_state.chat_history,
            "errors": analysis.get("errors", [])
        }
        
        st.download_button(
            "📊 Complete RAG Report",
            json.dumps(complete_report, indent=2),
            f"rag_healthcare_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # RAG Knowledge Base only
        rag_kb = analysis.get("rag_knowledge_base", {})
        if rag_kb:
            st.download_button(
                "🧠 RAG Knowledge Base",
                json.dumps(rag_kb, indent=2),
                f"rag_knowledge_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        # Conversation history
        convo_data = {
            "session_info": {
                "rag_mode": st.session_state.rag_mode,
                "timestamp": datetime.now().isoformat(),
                "total_messages": len(st.session_state.chat_history)
            },
            "conversation": st.session_state.chat_history
        }
        
        st.download_button(
            "💬 Conversation Log",
            json.dumps(convo_data, indent=2),
            f"rag_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    🧠 <strong>Enhanced RAG Healthcare Analysis Chatbot</strong><br>
    Retrieval-Augmented Generation • Context-Aware Analysis • Knowledge-Based Q&A<br>
    <strong>Current Mode:</strong> {'🧠 RAG Mode (Knowledge-Based)' if st.session_state.rag_mode else '🤖 Analysis Mode (Data Processing)'}<br>
    ⚠️ <em>This analysis is for informational purposes only and should not replace professional medical advice.</em>
</div>
""", unsafe_allow_html=True)

# Enhanced sidebar with RAG information
with st.sidebar:
    st.header("🧠 RAG System Info")
    
    # Agent status
    if st.session_state.agent:
        st.success("🤖 Agent: Ready")
        
        # RAG status
        if st.session_state.rag_mode:
            st.success("🧠 RAG: Active")
            if st.session_state.agent:
                rag_status = st.session_state.agent.get_rag_status()
                st.metric("KB Size", f"{rag_status['knowledge_base_size']:,}")
                if rag_status['session_id']:
                    st.text(f"Session: {rag_status['session_id'][:12]}...")
        else:
            st.info("🧠 RAG: Inactive")
            
        # Analysis status
        if st.session_state.current_analysis:
            st.success("📊 Analysis: Available")
        else:
            st.info("📊 Analysis: None")
    else:
        st.error("🤖 Agent: Not initialized")
    
    st.text(f"💬 Messages: {len(st.session_state.chat_history)}")
    
    st.markdown("---")
    st.markdown("### 🎯 RAG Mode Guide")
    
    if st.session_state.rag_mode:
        st.markdown("""
        **🧠 You're in RAG Mode!**
        
        **What this means:**
        - I'm using your analysis data as my knowledge base
        - All answers are grounded in your specific patient data  
        - I can provide precise, data-driven responses
        - Ask me anything about the analysis results
        
        **Try asking:**
        - Specific data counts  
        - Medical conditions found
        - Medication details
        - API status information
        - Cross-data comparisons
        """)
        
        if st.button("🔄 Exit RAG Mode", key="sidebar_exit_rag"):
            refresh_everything()
            
    else:
        st.markdown("""
        **🤖 Analysis Mode Active**
        
        **How to enter RAG mode:**
        1. Give me patient analysis command
        2. I'll fetch & process the data
        3. RAG mode automatically activates
        4. Ask detailed questions about results
        
        **RAG Benefits:**
        - Answers based on actual patient data
        - Precise, context-aware responses  
        - Deep data analysis capabilities
        - Knowledge-based conversation
        """)
    
    st.markdown("---")
    
    # Debug info
    if st.checkbox("🐛 Debug Info"):
        st.text(f"Agent init: {st.session_state.agent is not None}")
        st.text(f"RAG mode: {st.session_state.rag_mode}")
        st.text(f"Show JSON: {st.session_state.show_json_data}")
        st.text(f"Is typing: {st.session_state.is_typing}")
        st.text(f"KB available: {st.session_state.rag_knowledge_base is not None}")
        if st.session_state.current_analysis:
            st.text(f"Analysis success: {st.session_state.current_analysis.get('success')}")
            st.text(f"Analysis RAG: {st.session_state.current_analysis.get('rag_mode')}")
