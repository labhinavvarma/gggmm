#!/usr/bin/env python3
"""
Streamlit Web Interface for Milliman MCP Chatbot
================================================

A professional web interface for the Milliman MCP chatbot with real-time 
patient data processing and API integration.

Usage:
    streamlit run streamlit_mcp_chatbot.py
"""

import streamlit as st
import asyncio
import json
import pandas as pd
from datetime import datetime
import logging
import traceback
from typing import Dict, Any, List, Optional
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the chatbot components
try:
    from milliman_mcp_chatbot import (
        MillimanMCPChatbot, 
        SnowflakeCortexConfig, 
        MCPConfig, 
        PatientData,
        PatientDataExtractor
    )
except ImportError as e:
    st.error(f"❌ Could not import chatbot components: {e}")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Milliman MCP Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional healthcare styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #2E86C1;
    text-align: center;
    margin-bottom: 1.5rem;
    font-weight: bold;
}

.mcp-badge {
    background: linear-gradient(45deg, #2E86C1, #3498DB);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 0.8rem;
    font-weight: bold;
    display: inline-block;
    margin: 0.2rem;
    font-size: 0.9rem;
}

.cortex-badge {
    background: linear-gradient(45deg, #E67E22, #F39C12);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 0.8rem;
    font-weight: bold;
    display: inline-block;
    margin: 0.2rem;
    font-size: 0.9rem;
}

.success-box {
    background: linear-gradient(135deg, #d4edda, #c3e6cb);
    border: 2px solid #28a745;
    color: #155724;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    border-radius: 0.5rem;
    font-weight: bold;
}

.error-box {
    background: linear-gradient(135deg, #f8d7da, #f5c6cb);
    border: 2px solid #dc3545;
    color: #721c24;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    border-radius: 0.5rem;
}

.info-box {
    background: linear-gradient(135deg, #cce7ff, #99d6ff);
    border: 2px solid #007bff;
    color: #004085;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    border-radius: 0.5rem;
}

.chat-message {
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 0.5rem;
    border-left: 4px solid;
}

.user-message {
    background-color: #f0f8ff;
    border-left-color: #2E86C1;
}

.assistant-message {
    background-color: #f8fff0;
    border-left-color: #28a745;
}

.patient-data-card {
    background: linear-gradient(135deg, #fff3e0, #ffecb3);
    border: 2px solid #ff9800;
    border-radius: 0.8rem;
    padding: 1.5rem;
    margin: 1rem 0;
}

.api-response-card {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    border: 2px solid #2196f3;
    border-radius: 0.8rem;
    padding: 1.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'is_initialized' not in st.session_state:
        st.session_state.is_initialized = False
    if 'config_cortex' not in st.session_state:
        st.session_state.config_cortex = None
    if 'config_mcp' not in st.session_state:
        st.session_state.config_mcp = None

def safe_run_async(coro):
    """Safely run async code in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

# Initialize session state
initialize_session_state()

# Main title
st.markdown('<h1 class="main-header">🏥 Milliman MCP Chatbot</h1>', unsafe_allow_html=True)
st.markdown("**Advanced healthcare AI assistant with MCP integration and Snowflake Cortex LLM**")

# Show system badges
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="mcp-badge">🔌 MCP Client</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="cortex-badge">🧠 Snowflake Cortex</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="mcp-badge">🏥 Healthcare APIs</div>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Snowflake Cortex Configuration
    st.subheader("🧠 Snowflake Cortex Settings")
    
    cortex_api_url = st.text_input(
        "API URL",
        value="https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete",
        help="Snowflake Cortex API endpoint"
    )
    
    cortex_api_key = st.text_input(
        "API Key",
        value="78a799ea-a0f6-11ef-a0ce-15a449f7a8b0",
        type="password",
        help="Your Snowflake Cortex API key"
    )
    
    cortex_model = st.selectbox(
        "Model",
        ["llama3.1-70b", "llama3.1-8b", "mixtral-8x7b"],
        index=0,
        help="Snowflake Cortex model to use"
    )
    
    cortex_app_id = st.text_input("App ID", value="edadip")
    cortex_aplctn_cd = st.text_input("Application Code", value="edagnai")
    
    # MCP Configuration
    st.subheader("🔌 MCP Server Settings")
    
    mcp_server_url = st.text_input(
        "MCP Server URL",
        value="http://localhost:8000/sse",
        help="URL of your Milliman MCP server"
    )
    
    mcp_server_name = st.text_input("Server Name", value="MillimanServer")
    
    # Connection settings
    st.subheader("🔧 Connection Settings")
    timeout = st.slider("Timeout (seconds)", 10, 60, 30)
    max_retries = st.slider("Max Retries", 1, 5, 3)
    
    # Initialize button
    if st.button("🚀 Initialize Chatbot", use_container_width=True):
        try:
            # Create configurations
            cortex_config = SnowflakeCortexConfig(
                api_url=cortex_api_url,
                api_key=cortex_api_key,
                model=cortex_model,
                app_id=cortex_app_id,
                aplctn_cd=cortex_aplctn_cd,
                timeout=timeout,
                max_retries=max_retries
            )
            
            mcp_config = MCPConfig(
                server_name=mcp_server_name,
                server_url=mcp_server_url,
                max_retries=max_retries
            )
            
            # Store in session state
            st.session_state.config_cortex = cortex_config
            st.session_state.config_mcp = mcp_config
            
            # Create chatbot
            chatbot = MillimanMCPChatbot(cortex_config, mcp_config)
            
            # Initialize
            with st.spinner("Initializing chatbot..."):
                success = safe_run_async(chatbot.initialize())
            
            if success:
                st.session_state.chatbot = chatbot
                st.session_state.is_initialized = True
                st.success("✅ Chatbot initialized successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to initialize chatbot. Check your settings and MCP server.")
                
        except Exception as e:
            st.error(f"❌ Initialization error: {str(e)}")
            st.code(traceback.format_exc())
    
    # Status display
    st.subheader("📊 Status")
    if st.session_state.is_initialized:
        st.success("✅ Chatbot Active")
        st.info(f"🤖 Model: {st.session_state.config_cortex.model if st.session_state.config_cortex else 'Unknown'}")
        st.info(f"🔌 MCP Server: Connected")
    else:
        st.warning("⏳ Chatbot Not Initialized")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        if st.session_state.chatbot:
            st.session_state.chatbot.clear_chat_history()
        st.success("Chat history cleared!")
        st.rerun()

# Main content area
if not st.session_state.is_initialized:
    st.markdown('<div class="info-box">🔧 Please configure and initialize the chatbot using the sidebar.</div>', unsafe_allow_html=True)
    
    # Show example configurations and usage
    st.markdown("### 💡 Getting Started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 Prerequisites:**")
        st.markdown("""
        1. **MCP Server Running:** Ensure your Milliman MCP server is running on port 8000
        2. **Snowflake Cortex Access:** Valid API key and endpoint
        3. **Network Access:** Connection to both MCP server and Snowflake Cortex
        """)
    
    with col2:
        st.markdown("**🚀 Quick Start:**")
        st.markdown("""
        1. **Configure Settings:** Update API credentials in the sidebar
        2. **Initialize Chatbot:** Click the "Initialize Chatbot" button
        3. **Start Chatting:** Enter patient queries or commands
        """)
    
    st.markdown("### 📖 Example Commands")
    
    examples = [
        "Get medical data for John Smith, SSN 123456789, DOB 1980-01-15, Male, Zip 12345",
        "Search MCID database for patient information",
        "Get authentication token for API access",
        "Retrieve comprehensive patient data including medical and pharmacy records"
    ]
    
    for i, example in enumerate(examples, 1):
        st.markdown(f"**{i}.** {example}")

else:
    # Chat interface
    st.markdown("### 💬 Chat Interface")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("**Recent Conversation:**")
        
        for msg in st.session_state.chat_history[-10:]:  # Show last 10 messages
            if msg['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🗣️ You:</strong> {msg['content']}
                    <br><small>⏰ {msg.get('timestamp', 'Unknown time')}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong> {msg['content']}
                    <br><small>⏰ {msg.get('timestamp', 'Unknown time')}</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("💬 Enter your message or patient query...")
    
    if user_input:
        try:
            # Add user message to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Process with chatbot
            with st.spinner("🤖 Processing your request..."):
                response = safe_run_async(st.session_state.chatbot.process_command(user_input))
            
            # Add assistant response to history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing message: {str(e)}")
            st.code(traceback.format_exc())

# Patient Data Entry Form
with st.expander("📝 Quick Patient Data Entry", expanded=False):
    st.markdown("**Enter patient information for structured API calls:**")
    
    with st.form("patient_data_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            ssn = st.text_input("SSN (9 digits)")
        
        with col2:
            date_of_birth = st.date_input("Date of Birth")
            gender = st.selectbox("Gender", ["M", "F"])
            zip_code = st.text_input("Zip Code")
        
        operation = st.selectbox(
            "Operation",
            ["get_all_data", "medical_submit", "mcid_search", "get_token"],
            help="Select the API operation to perform"
        )
        
        if st.form_submit_button("🚀 Submit Patient Data"):
            try:
                # Create patient data object
                patient_data = PatientData(
                    first_name=first_name,
                    last_name=last_name,
                    ssn=ssn,
                    date_of_birth=date_of_birth.strftime('%Y-%m-%d'),
                    gender=gender,
                    zip_code=zip_code
                )
                
                # Validate
                is_valid, errors = patient_data.validate()
                
                if not is_valid:
                    st.error("❌ Invalid patient data:")
                    for error in errors:
                        st.error(f"• {error}")
                else:
                    # Create structured command
                    command = f"""
                    Please use the {operation} API with this patient information:
                    
                    Patient: {first_name} {last_name}
                    SSN: {ssn}
                    Date of Birth: {date_of_birth.strftime('%Y-%m-%d')}
                    Gender: {gender}
                    Zip Code: {zip_code}
                    
                    Execute the {operation} operation and provide a detailed response.
                    """
                    
                    if st.session_state.is_initialized:
                        # Process command
                        with st.spinner("🤖 Processing patient data..."):
                            response = safe_run_async(st.session_state.chatbot.process_command(command))
                        
                        # Display response
                        st.markdown('<div class="api-response-card">', unsafe_allow_html=True)
                        st.markdown("**🔍 API Response:**")
                        st.markdown(response)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Add to chat history
                        st.session_state.chat_history.extend([
                            {
                                'role': 'user',
                                'content': f"Patient data form submission: {operation} for {first_name} {last_name}",
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            },
                            {
                                'role': 'assistant',
                                'content': response,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                        ])
                    else:
                        st.warning("⚠️ Please initialize the chatbot first using the sidebar.")
                        
            except Exception as e:
                st.error(f"❌ Error processing patient data: {str(e)}")

# API Tools Information
with st.expander("🔧 Available API Tools", expanded=False):
    st.markdown("**Milliman API Tools accessible through the chatbot:**")
    
    tools_info = [
        {
            "Tool": "get_token",
            "Description": "Get authentication token for API access",
            "Parameters": "None",
            "Use Case": "Authentication and session management"
        },
        {
            "Tool": "medical_submit",
            "Description": "Submit medical record request",
            "Parameters": "Patient information (name, SSN, DOB, gender, zip)",
            "Use Case": "Retrieve medical records and health data"
        },
        {
            "Tool": "mcid_search",
            "Description": "Search MCID database",
            "Parameters": "Patient information (name, SSN, DOB, gender, zip)",
            "Use Case": "Patient identification and matching"
        },
        {
            "Tool": "get_all_data",
            "Description": "Get comprehensive patient data",
            "Parameters": "Patient information (name, SSN, DOB, gender, zip)",
            "Use Case": "Complete patient record retrieval"
        }
    ]
    
    df = pd.DataFrame(tools_info)
    st.table(df)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🏥 <strong>Milliman MCP Chatbot</strong><br>
    Powered by Snowflake Cortex LLM and Model Context Protocol<br>
    ⚠️ <em>This system processes healthcare data. Ensure HIPAA compliance in production environments.</em>
</div>
""", unsafe_allow_html=True)

# Debug information (only in development)
if st.checkbox("🐛 Show Debug Information"):
    st.subheader("Debug Information")
    
    debug_info = {
        "Session State": {
            "is_initialized": st.session_state.is_initialized,
            "chat_history_length": len(st.session_state.chat_history),
            "chatbot_exists": st.session_state.chatbot is not None
        },
        "Configuration": {
            "cortex_config": st.session_state.config_cortex.__dict__ if st.session_state.config_cortex else None,
            "mcp_config": st.session_state.config_mcp.__dict__ if st.session_state.config_mcp else None
        },
        "Environment": {
            "python_version": sys.version,
            "current_directory": current_dir
        }
    }
    
    st.json(debug_info)
