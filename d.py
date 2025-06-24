import streamlit as st
import json
from datetime import datetime
from typing import Dict, Any, List

# Import the simple RAG agent
try:
    from simple_rag_agent import SimpleRAGHealthAgent, Config
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    import_error = str(e)

# === Streamlit UI ===
st.set_page_config(
    page_title="🧠 Simple RAG Health Assistant", 
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Simple RAG Health Assistant")

st.markdown("""
**Simple workflow:** Analyze patient → Get MCP data → Deidentify → RAG mode with JSON context.  
Ask questions about the medical records. The assistant keeps the deidentified JSONs in context.
""")

# === Session Initialization ===
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "rag_active" not in st.session_state:
    st.session_state.rag_active = False

if "json_context" not in st.session_state:
    st.session_state.json_context = []

# === Check Agent Availability ===
if not AGENT_AVAILABLE:
    st.error(f"❌ Simple RAG agent not available: {import_error}")
    st.info("💡 Please ensure simple_rag_agent.py is available and dependencies are installed.")
    st.stop()

# === Initialize Agent ===
if not st.session_state.agent:
    try:
        st.session_state.agent = SimpleRAGHealthAgent()
        st.success("✅ Simple RAG agent initialized!")
    except Exception as e:
        st.error(f"❌ Failed to initialize agent: {str(e)}")
        st.stop()

# === Reset Button ===
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    if st.session_state.rag_active:
        st.success("🧠 RAG Mode Active - Medical records loaded in context")
    else:
        st.info("🤖 Analysis Mode - Ready to analyze patient data")

with col2:
    if st.button("🔄 Reset Session"):
        st.session_state.messages = []
        st.session_state.rag_active = False
        st.session_state.json_context = []
        if st.session_state.agent:
            st.session_state.agent.refresh_session()
        st.success("Session reset.")
        st.rerun()

with col3:
    if st.session_state.rag_active:
        st.metric("RAG Status", "Active", "JSON Context Loaded")
    else:
        st.metric("RAG Status", "Inactive", "No Context")

# === RAG Context Display ===
if st.session_state.rag_active and st.session_state.json_context:
    with st.expander("🧠 RAG Context - Deidentified Medical Records", expanded=False):
        st.info("These JSON records are being used as context for RAG responses.")
        
        for i, record in enumerate(st.session_state.json_context):
            record_type = record.get("type", f"Record {i+1}")
            st.subheader(f"📄 {record_type.title()}")
            st.json(record)
        
        # Download button
        context_json = json.dumps(st.session_state.json_context, indent=2)
        st.download_button(
            "💾 Download RAG Context",
            context_json,
            f"rag_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# === Quick Actions for RAG Mode ===
if st.session_state.rag_active:
    st.markdown("### 🚀 Quick Questions (RAG Mode)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Claims Count", help="How many medical claims?"):
            st.session_state.messages.append({"role": "user", "content": "How many medical claims were found?"})
            st.rerun()
    
    with col2:
        if st.button("💊 Medications", help="What medications are listed?"):
            st.session_state.messages.append({"role": "user", "content": "What medications are listed in the records?"})
            st.rerun()
    
    with col3:
        if st.button("🏥 Conditions", help="What medical conditions?"):
            st.session_state.messages.append({"role": "user", "content": "What medical conditions were found?"})
            st.rerun()
    
    with col4:
        if st.button("📡 API Status", help="MCP server status"):
            st.session_state.messages.append({"role": "user", "content": "What's the API status?"})
            st.rerun()

# === Example Commands ===
with st.expander("💡 Example Commands", expanded=not st.session_state.rag_active):
    if st.session_state.rag_active:
        st.markdown("""
        **🧠 RAG Mode - Ask questions about the medical records:**
        
        📊 **Data Questions:**
        - "How many medical claims were found?"
        - "What's the total number of records?"
        - "Show me the pharmacy data"
        
        💊 **Medical Questions:**
        - "What medications are listed?"
        - "What medical conditions were found?"
        - "Are there any diabetes medications?"
        
        🔍 **Analysis Questions:**
        - "Summarize the medical records"
        - "What patterns do you see in the data?"
        - "Analyze the pharmacy records"
        """)
    else:
        st.markdown("""
        **🤖 Analysis Mode - Start with patient analysis:**
        
        📝 **Patient Analysis Commands:**
        - "Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345"
        - "Evaluate patient Sarah Johnson, born 1975-08-22, female, SSN 987654321, zip 90210"
        
        **After analysis completes, I'll enter RAG mode and you can ask detailed questions!**
        """)
        
        # Quick start button
        if st.button("👤 Try Sample Analysis", key="sample_patient"):
            sample_cmd = "Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345"
            st.session_state.messages.append({"role": "user", "content": sample_cmd})
            st.rerun()

# === Process Pending Messages ===
if st.session_state.messages:
    # Get the last message if it's from user and not processed
    last_message = st.session_state.messages[-1]
    if last_message["role"] == "user" and not any(msg["role"] == "assistant" for msg in st.session_state.messages[st.session_state.messages.index(last_message)+1:]):
        
        with st.spinner("🤖 Processing..."):
            try:
                # Call the agent
                result = st.session_state.agent.chat(last_message["content"])
                
                # Update RAG status
                st.session_state.rag_active = result.get("rag_active", False)
                st.session_state.json_context = result.get("json_context", [])
                
                # Add assistant response
                assistant_response = result.get("response", "I couldn't process your request.")
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                
                # Show errors if any
                if result.get("errors"):
                    for error in result["errors"]:
                        st.error(f"⚠️ {error}")
                
                # Rerun to update the UI
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing message: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})

# === Chat History Display ===
st.markdown("### 💬 Conversation")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === Chat Input ===
user_input = st.chat_input("💬 Ask about medical records or give patient analysis command...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()

# === Sidebar Info ===
with st.sidebar:
    st.header("🧠 RAG System Info")
    
    # Status indicators
    if st.session_state.agent:
        st.success("🤖 Agent: Ready")
    else:
        st.error("🤖 Agent: Not Ready")
    
    if st.session_state.rag_active:
        st.success("🧠 RAG: Active")
        st.metric("Context Records", len(st.session_state.json_context))
    else:
        st.info("🧠 RAG: Inactive")
    
    st.metric("Messages", len(st.session_state.messages))
    
    st.markdown("---")
    
    # System explanation
    st.markdown("### 🔄 How It Works")
    
    if st.session_state.rag_active:
        st.markdown("""
        **🧠 RAG Mode Active**
        
        ✅ Medical records loaded  
        ✅ Data deidentified  
        ✅ JSON context ready  
        ✅ Ready for questions  
        
        **Current Context:**
        - Deidentified medical data
        - Deidentified pharmacy data  
        - MCP API status info
        
        Ask me anything about the records!
        """)
    else:
        st.markdown("""
        **🤖 Analysis Mode**
        
        **Steps:**
        1. Give patient analysis command
        2. I extract patient data
        3. I call MCP server APIs
        4. I deidentify the data
        5. I load JSON into RAG context
        6. You ask questions!
        
        **Ready for patient analysis command.**
        """)
    
    st.markdown("---")
    
    # Controls
    st.markdown("### ⚙️ Controls")
    
    if st.session_state.rag_active:
        if st.button("🔄 Exit RAG Mode", key="exit_rag"):
            st.session_state.rag_active = False
            st.session_state.json_context = []
            if st.session_state.agent:
                st.session_state.agent.refresh_session()
            st.success("Exited RAG mode")
            st.rerun()
    
    # Debug info
    if st.checkbox("🐛 Debug Info"):
        st.json({
            "agent_ready": st.session_state.agent is not None,
            "rag_active": st.session_state.rag_active,
            "context_records": len(st.session_state.json_context),
            "total_messages": len(st.session_state.messages),
            "context_types": [r.get("type", "unknown") for r in st.session_state.json_context] if st.session_state.json_context else []
        })

# === Footer ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🧠 <strong>Simple RAG Health Assistant</strong><br>
    MCP Server Integration • Auto-Deidentification • JSON Context RAG<br>
    <em>Simplified workflow for healthcare data analysis</em>
</div>
""", unsafe_allow_html=True)
