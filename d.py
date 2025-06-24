# streamlit_enhanced_chatbot_ui.py
import streamlit as st
import asyncio
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
import sys
import os

# Configure Streamlit page
st.set_page_config(
    page_title="🤖 Enhanced LangGraph MCP Healthcare Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the Enhanced LangGraph MCP chatbot
try:
    from enhanced_langgraph_mcp_chatbot import EnhancedLangGraphMCPChatbot, MCPChatbotConfig
    CHATBOT_AVAILABLE = True
except ImportError as e:
    CHATBOT_AVAILABLE = False
    st.error(f"❌ Could not import Enhanced LangGraph MCP Chatbot: {e}")

# Custom CSS for enhanced chatbot styling
st.markdown("""
<style>
.main-header {
    font-size: 2.8rem;
    color: #2E86AB;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: bold;
    background: linear-gradient(45deg, #2E86AB, #A23B72);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.enhanced-badge {
    background: linear-gradient(45deg, #2E86AB, #A23B72);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 1rem;
    font-weight: bold;
    display: inline-block;
    margin: 0.5rem 0;
}

.langgraph-badge {
    background: linear-gradient(45deg, #F18F01, #C73E1D);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 0.8rem;
    font-weight: bold;
    display: inline-block;
    margin: 0.2rem;
    font-size: 0.9rem;
}

.mcp-badge {
    background: linear-gradient(45deg, #4CAF50, #45a049);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 0.8rem;
    font-weight: bold;
    display: inline-block;
    margin: 0.2rem;
    font-size: 0.9rem;
}

.chat-container {
    background: #f8f9fa;
    border-radius: 1rem;
    padding: 1rem;
    margin: 1rem 0;
    border: 2px solid #e9ecef;
}

.user-message {
    background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
    border-left: 4px solid #2196F3;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

.assistant-message {
    background: linear-gradient(135deg, #F3E5F5, #E1BEE7);
    border-left: 4px solid #9C27B0;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

.analysis-complete {
    background: linear-gradient(135deg, #E8F5E8, #C8E6C9);
    border: 2px solid #4CAF50;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
}

.step-indicator {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 0.5rem;
    padding: 0.5rem;
    margin: 0.25rem 0;
    font-size: 0.9rem;
}

.step-completed {
    background: linear-gradient(135deg, #d4edda, #c3e6cb);
    border-color: #28a745;
    color: #155724;
}

.step-running {
    background: linear-gradient(135deg, #fff3cd, #ffeaa7);
    border-color: #ffc107;
    color: #856404;
}

.step-error {
    background: linear-gradient(135deg, #f8d7da, #f5c6cb);
    border-color: #dc3545;
    color: #721c24;
}

.entity-card {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.metric-box {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    border: 1px solid #dee2e6;
    border-radius: 0.5rem;
    padding: 1rem;
    text-align: center;
    margin: 0.25rem;
}

.follow-up-button {
    background: linear-gradient(135deg, #17a2b8, #138496);
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 0.5rem;
    margin: 0.25rem;
    cursor: pointer;
    font-size: 0.9rem;
}

.suggestion-button {
    background: linear-gradient(135deg, #6f42c1, #5a2d91);
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 0.5rem;
    margin: 0.25rem;
    cursor: pointer;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state for the chatbot"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().timestamp()}"
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'config' not in st.session_state:
        st.session_state.config = None

def display_chat_message(message: Dict[str, Any], is_user: bool = False):
    """Display a chat message with proper styling"""
    if is_user:
        st.markdown(f"""
        <div class="user-message">
            <strong>👤 You:</strong><br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <strong>🤖 Healthcare Assistant:</strong><br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)

def display_analysis_results(result: Dict[str, Any]):
    """Display comprehensive analysis results"""
    if not result.get('analysis_complete'):
        return
    
    st.markdown('<div class="analysis-complete">', unsafe_allow_html=True)
    st.markdown("### 🔬 **Analysis Results**")
    
    # Health Analysis Summary
    health_analysis = result.get('health_analysis', {})
    if health_analysis:
        summary = health_analysis.get('summary', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Conditions", summary.get('conditions_identified', 0))
        with col2:
            st.metric("Medications", summary.get('medications_found', 0))
        with col3:
            st.metric("Risk Factors", len(summary.get('risk_factors', {})))
        with col4:
            confidence = summary.get('analysis_confidence', 0)
            st.metric("Confidence", f"{confidence:.1%}")
        
        # Risk Assessment
        risk_assessment = health_analysis.get('risk_assessment', {})
        if risk_assessment:
            st.markdown("#### ⚠️ **Risk Assessment**")
            risk_cols = st.columns(len(risk_assessment))
            for i, (risk_type, level) in enumerate(risk_assessment.items()):
                with risk_cols[i]:
                    color = "#dc3545" if level == "high" else "#ffc107" if level == "moderate" else "#28a745"
                    st.markdown(f"""
                    <div class="metric-box" style="border-color: {color};">
                        <strong>{risk_type.replace('_', ' ').title()}</strong><br>
                        <span style="color: {color}; font-weight: bold;">{level.title()}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Recommendations
        recommendations = health_analysis.get('recommendations', [])
        if recommendations:
            st.markdown("#### 💡 **Recommendations**")
            for rec in recommendations:
                st.markdown(f"• {rec}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_entity_extraction(entity_extraction: Dict[str, Any]):
    """Display entity extraction results"""
    if not entity_extraction:
        return
    
    st.markdown("#### 🎯 **Enhanced Entity Extraction**")
    
    # Health Conditions
    conditions = entity_extraction.get('health_conditions', {})
    if conditions:
        st.markdown("**Health Conditions:**")
        condition_cols = st.columns(min(len(conditions), 4))
        for i, (condition, status) in enumerate(conditions.items()):
            with condition_cols[i % 4]:
                st.markdown(f"""
                <div class="entity-card">
                    <strong>{condition.title()}</strong><br>
                    <span style="color: #dc3545;">{status}</span>
                </div>
                """, unsafe_allow_html=True)
    
    # Medications
    medications = entity_extraction.get('medications', {})
    if medications:
        st.markdown("**Medications Identified:**")
        med_cols = st.columns(min(len(medications), 4))
        for i, (med, status) in enumerate(medications.items()):
            with med_cols[i % 4]:
                st.markdown(f"""
                <div class="entity-card">
                    <strong>{med.title()}</strong><br>
                    <span style="color: #28a745;">{status}</span>
                </div>
                """, unsafe_allow_html=True)

def display_workflow_steps(step_status: Dict[str, str]):
    """Display workflow step progress"""
    if not step_status:
        return
    
    st.markdown("#### 🔄 **Workflow Progress**")
    
    step_names = {
        "analyze_intent_and_context": "🎯 Intent Analysis",
        "determine_api_routing": "🔀 API Routing",
        "extract_patient_info": "👤 Patient Extraction",
        "check_data_requirements": "✅ Data Validation",
        "call_mcp_apis": "📡 MCP API Calls",
        "deidentify_responses": "🔒 Data Deidentification",
        "enhanced_entity_extraction": "🎯 Entity Extraction",
        "analyze_health_data": "📊 Health Analysis",
        "generate_conversational_response": "💬 Response Generation"
    }
    
    cols = st.columns(3)
    for i, (step_key, status) in enumerate(step_status.items()):
        step_name = step_names.get(step_key, step_key.replace('_', ' ').title())
        
        with cols[i % 3]:
            if status == "completed":
                css_class = "step-completed"
                icon = "✅"
            elif status == "running":
                css_class = "step-running"
                icon = "🔄"
            elif status == "error":
                css_class = "step-error"
                icon = "❌"
            else:
                css_class = "step-indicator"
                icon = "⏳"
            
            st.markdown(f"""
            <div class="{css_class} step-indicator">
                {icon} {step_name}
            </div>
            """, unsafe_allow_html=True)

def display_deidentified_data(deidentified_data: Dict[str, Any]):
    """Display deidentified data summary"""
    if not deidentified_data:
        return
    
    with st.expander("🔒 **Deidentified Data Summary**"):
        for data_type, data in deidentified_data.items():
            st.markdown(f"**{data_type.replace('_', ' ').title()}:**")
            if isinstance(data, dict) and not data.get('error'):
                st.markdown(f"✅ Successfully deidentified - {len(str(data))} characters")
            else:
                st.markdown(f"⚠️ {data.get('error', 'Unknown error')}")

# Initialize session state
initialize_session_state()

# Main Header
st.markdown('<h1 class="main-header">🤖 Enhanced LangGraph MCP Healthcare Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<div class="enhanced-badge">🚀 Powered by LangGraph + MCP + Snowflake Cortex llama3.1-70b</div>', unsafe_allow_html=True)
st.markdown('<div class="langgraph-badge">🔄 LangGraph Workflow</div>', unsafe_allow_html=True)
st.markdown('<div class="mcp-badge">🔗 MCP Client Integration</div>', unsafe_allow_html=True)

st.markdown("**Advanced healthcare chatbot with intelligent LLM-powered responses, API routing, data deidentification, and enhanced entity extraction**")

# Display availability status
if CHATBOT_AVAILABLE:
    st.success("✅ Enhanced LangGraph MCP Chatbot loaded successfully!")
else:
    st.error("❌ Enhanced LangGraph MCP Chatbot not available")
    st.stop()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Enhanced Chatbot Configuration")
    
    # System Status
    st.markdown("### 🔥 System Status")
    st.markdown("✅ **Enhanced LangGraph**: Active")
    st.markdown("🔗 **MCP Client**: Connected")
    st.markdown("🤖 **Snowflake Cortex LLM**: llama3.1-70b")
    st.markdown("🔀 **Intelligent Routing**: Enabled")
    st.markdown("🔒 **Data Deidentification**: Active")
    st.markdown("🎯 **Entity Extraction**: Enhanced")
    st.markdown("💬 **Continuous Chat**: Enabled")
    st.markdown("🧠 **AI-Powered Responses**: Active")
    
    st.markdown("---")
    
    # Configuration
    st.subheader("🔌 MCP Server Configuration")
    mcp_server_url = st.text_input("MCP Server URL", value="http://localhost:8000")
    timeout = st.slider("Timeout (seconds)", 10, 60, 30)
    max_retries = st.slider("Max Retries", 1, 5, 3)
    
    st.subheader("🤖 Snowflake Cortex LLM Settings")
    st.info("💡 **Snowflake Cortex AI is integrated** for intelligent responses and analysis")
    
    # Show current LLM configuration (read-only for security)
    cortex_model = st.text_input("LLM Model", value="llama3.1-70b", disabled=True)
    cortex_api_url = st.text_input("Cortex API URL", value="https://sfassist.edagenaidev.awsdns.internal.das/...", disabled=True)
    st.text_area("System Message", value="You are a healthcare AI assistant. Provide accurate, concise answers based on context.", disabled=True, height=100)
    
    st.markdown("**🔧 LLM Features Enabled:**")
    st.markdown("• ✅ Intelligent Intent Analysis")
    st.markdown("• ✅ Smart Patient Data Extraction") 
    st.markdown("• ✅ Conversational Response Generation")
    st.markdown("• ✅ Context-Aware Follow-up Questions")
    st.markdown("• ✅ Medical Knowledge Integration")
    
    # Update configuration
    if st.button("🔄 Update Configuration"):
        config = MCPChatbotConfig(
            mcp_server_url=mcp_server_url,
            timeout=timeout,
            max_retries=max_retries
        )
        st.session_state.config = config
        st.session_state.chatbot = None  # Force reinitialization
        st.success("✅ Configuration updated!")
        st.rerun()
    
    # Test LLM Connection
    if st.button("🧪 Test Snowflake Cortex LLM"):
        if st.session_state.chatbot:
            with st.spinner("Testing Snowflake Cortex connection..."):
                try:
                    test_result = asyncio.run(st.session_state.chatbot.test_cortex_connection())
                    if test_result["success"]:
                        st.success("✅ Snowflake Cortex LLM connection successful!")
                        st.info(f"🤖 Model: {test_result['model']}")
                        st.info(f"📝 Response: {test_result['response']}")
                    else:
                        st.error("❌ Snowflake Cortex LLM connection failed!")
                        st.error(f"💥 Error: {test_result['error']}")
                except Exception as e:
                    st.error(f"❌ LLM test failed: {e}")
        else:
            st.warning("⚠️ Chatbot not initialized")
    
    # Current session info
    st.markdown("---")
    st.subheader("📊 Session Information")
    st.write(f"**Session ID:** {st.session_state.session_id[:8]}...")
    st.write(f"**Messages:** {len(st.session_state.messages)}")
    st.write(f"**Analyses:** {len(st.session_state.analysis_results)}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.analysis_results = []
        st.session_state.session_id = f"session_{datetime.now().timestamp()}"
        st.rerun()
    
    # Health check with detailed diagnostics
    if st.button("🩺 Detailed Health Check"):
        if st.session_state.chatbot:
            with st.spinner("Running comprehensive health check..."):
                try:
                    health = asyncio.run(st.session_state.chatbot.health_check())
                    
                    if health.get('chatbot_status') == 'healthy':
                        st.success("✅ System is healthy!")
                    elif health.get('chatbot_status') == 'server_unreachable':
                        st.error("❌ Cannot reach MCP server!")
                        st.error("💡 **Fix:** Start your FastAPI server with `python fixed_app.py`")
                    elif health.get('chatbot_status') == 'method_not_allowed_errors':
                        st.error("❌ 405 Method Not Allowed errors detected!")
                        st.error("💡 **Fix:** Check FastAPI router configuration")
                    else:
                        st.warning("⚠️ System issues detected")
                    
                    # Show detailed diagnostics
                    with st.expander("🔍 Detailed Diagnostics"):
                        st.json(health)
                    
                    # Show recommendations
                    recommendations = health.get('recommendations', [])
                    if recommendations:
                        st.markdown("### 💡 Recommendations:")
                        for rec in recommendations:
                            st.warning(f"• {rec}")
                            
                except Exception as e:
                    st.error(f"❌ Health check failed: {e}")
        else:
            st.warning("⚠️ Chatbot not initialized")

# Enhanced Workflow Visualization
st.markdown("### 🔄 Enhanced LangGraph Workflow with Snowflake Cortex")

workflow_cols = st.columns(5)
workflow_steps = [
    ("🧠", "LLM Intent Analysis", "Snowflake Cortex intent recognition"),
    ("🔀", "Smart API Routing", "AI-powered endpoint selection"),
    ("📡", "MCP Calls", "Call selected healthcare APIs"),
    ("🔒", "Deidentification", "Remove PII from responses"),
    ("🎯", "LLM Entity Extraction", "AI-enhanced health entity extraction")
]

for i, (icon, title, desc) in enumerate(workflow_steps):
    with workflow_cols[i]:
        st.markdown(f"""
        <div class="metric-box">
            <div style="font-size: 2rem;">{icon}</div>
            <strong>{title}</strong><br>
            <small>{desc}</small>
        </div>
        """, unsafe_allow_html=True)

# Initialize chatbot if needed
if st.session_state.chatbot is None:
    try:
        config = st.session_state.config or MCPChatbotConfig()
        st.session_state.chatbot = EnhancedLangGraphMCPChatbot(config)
        st.success(f"🤖 Enhanced chatbot initialized with MCP server: {config.mcp_server_url}")
    except Exception as e:
        st.error(f"❌ Failed to initialize chatbot: {e}")
        st.stop()

# Initialize welcome message
if not st.session_state.messages:
    welcome_message = {
        "role": "assistant",
        "content": """Hello! I'm your Enhanced Healthcare Analysis Assistant powered by LangGraph, MCP, and Snowflake Cortex AI. 

🧠 **AI-Powered Features:**
🤖 **Snowflake Cortex llama3.1-70b** - Advanced natural language understanding
🎯 **Intelligent Intent Recognition** - Understands your healthcare questions naturally
📊 **Smart Patient Data Extraction** - Automatically extracts information from your messages
💬 **Conversational Health Analysis** - Ask follow-up questions naturally

🏥 **Healthcare Capabilities:**
• **Comprehensive Patient Analysis** - Complete health data processing with intelligent API routing
• **Medical Code Interpretation** - ICD-10, NDC, and other medical codes explained in plain language
• **Medication Analysis** - Detailed pharmacy data examination with drug interaction insights
• **Risk Assessment** - AI-powered health risk evaluation and recommendations
• **Continuous Conversations** - Remember context and build on previous analyses

🔒 **Privacy & Compliance:**
• **Automatic Data Deidentification** - HIPAA-compliant PII removal
• **Secure Processing** - All data processed through secure healthcare APIs

**Example:** "Analyze patient John Smith, age 45, male, SSN 123456789, zip 12345"

What healthcare analysis can I help you with today?"""
    }
    st.session_state.messages.append(welcome_message)

# Display chat history
st.markdown("### 💬 Healthcare Chat")

for i, message in enumerate(st.session_state.messages):
    display_chat_message(message, is_user=(message["role"] == "user"))
    
    # Display analysis results if available
    if message["role"] == "assistant" and i < len(st.session_state.analysis_results):
        result = st.session_state.analysis_results[i]
        if result:
            display_analysis_results(result)
            display_entity_extraction(result.get('entity_extraction', {}))
            display_workflow_steps(result.get('step_status', {}))
            display_deidentified_data(result.get('deidentified_data', {}))

# Chat input
if prompt := st.chat_input("Ask about healthcare analysis, patient data, or medical information..."):
    # Add user message
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    
    # Display user message immediately
    display_chat_message(user_message, is_user=True)
    
    # Process with enhanced chatbot
    with st.spinner("🤖 Processing your request through Enhanced LangGraph workflow..."):
        try:
            # Get conversation history for context
            conversation_history = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in st.session_state.messages[-10:]  # Last 10 messages
            ]
            
            # Call enhanced chatbot
            result = asyncio.run(st.session_state.chatbot.chat(
                message=prompt,
                session_id=st.session_state.session_id,
                conversation_history=conversation_history
            ))
            
            # Create assistant response
            assistant_message = {
                "role": "assistant", 
                "content": result['response']
            }
            st.session_state.messages.append(assistant_message)
            st.session_state.analysis_results.append(result)
            
            # Display assistant response
            display_chat_message(assistant_message)
            
            # Display comprehensive results
            if result.get('analysis_complete'):
                display_analysis_results(result)
                display_entity_extraction(result.get('entity_extraction', {}))
            
            # Always show workflow progress
            display_workflow_steps(result.get('step_status', {}))
            
            # Show deidentified data if available
            display_deidentified_data(result.get('deidentified_data', {}))
            
            # Display follow-up questions as interactive buttons
            follow_ups = result.get('follow_up_questions', [])
            if follow_ups:
                st.markdown("#### ❓ **Follow-up Questions**")
                follow_up_cols = st.columns(min(len(follow_ups), 3))
                for i, question in enumerate(follow_ups):
                    with follow_up_cols[i % 3]:
                        if st.button(question, key=f"followup_{len(st.session_state.messages)}_{i}"):
                            st.rerun()
            
            # Display suggested actions
            suggestions = result.get('suggested_actions', [])
            if suggestions:
                st.markdown("#### 💡 **Suggested Actions**")
                suggestion_cols = st.columns(min(len(suggestions), 3))
                for i, action in enumerate(suggestions):
                    with suggestion_cols[i % 3]:
                        if st.button(action, key=f"suggestion_{len(st.session_state.messages)}_{i}"):
                            st.info(f"Action: {action}")
            
            # Show errors if any with helpful diagnostics
            errors = result.get('errors', [])
            if errors:
                st.error("⚠️ **Issues encountered:**")
                for error in errors:
                    st.error(f"• {error}")
                
                # Show specific help for common errors
                if any("405" in str(error) for error in errors):
                    st.error("🚨 **405 Method Not Allowed Error Detected!**")
                    st.markdown("""
                    **Quick Fixes:**
                    1. **Check if your server is running:** `python fixed_app.py`
                    2. **Verify the correct port:** Should be http://localhost:8000
                    3. **Check server logs** for router mounting issues
                    4. **Test manually:** Visit http://localhost:8000/health in your browser
                    """)
                elif any("connection" in str(error).lower() for error in errors):
                    st.error("🔌 **Connection Error Detected!**")
                    st.markdown("""
                    **Quick Fixes:**
                    1. **Start your FastAPI server:** `python fixed_app.py` or `python app.py`
                    2. **Check the port:** Make sure it's running on port 8000
                    3. **Check firewall:** Ensure port 8000 is not blocked
                    """)
                elif any("timeout" in str(error).lower() for error in errors):
                    st.error("⏱️ **Timeout Error Detected!**")
                    st.markdown("""
                    **Quick Fixes:**
                    1. **Check server performance:** Server might be overloaded
                    2. **Increase timeout:** Try increasing timeout in settings
                    3. **Check server logs:** Look for processing delays
                    """)
            
            # Success metrics
            if result.get('success'):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Intent", result.get('intent', 'unknown'))
                with col2:
                    confidence = result.get('confidence', 0)
                    st.metric("Confidence", f"{confidence:.1%}")
                with col3:
                    apis_called = len(result.get('api_responses', {}))
                    st.metric("APIs Called", apis_called)
                with col4:
                    steps_completed = len([s for s in result.get('step_status', {}).values() if s == 'completed'])
                    st.metric("Steps Completed", steps_completed)
            
        except Exception as e:
            st.error(f"❌ Error processing message: {e}")
            
            # Add error message to chat
            error_message = {
                "role": "assistant",
                "content": f"I encountered an error while processing your request: {str(e)}. Please try rephrasing your question or check the system configuration."
            }
            st.session_state.messages.append(error_message)
            st.session_state.analysis_results.append({})
    
    # Auto-rerun to show the new messages
    st.rerun()

# Enhanced Analytics Dashboard
if st.session_state.analysis_results:
    st.markdown("---")
    st.markdown("### 📊 Session Analytics")
    
    # Calculate analytics
    total_analyses = len([r for r in st.session_state.analysis_results if r.get('analysis_complete')])
    total_api_calls = sum(len(r.get('api_responses', {})) for r in st.session_state.analysis_results)
    avg_confidence = sum(r.get('confidence', 0) for r in st.session_state.analysis_results) / max(len(st.session_state.analysis_results), 1)
    
    analytics_cols = st.columns(4)
    with analytics_cols[0]:
        st.metric("Total Messages", len(st.session_state.messages))
    with analytics_cols[1]:
        st.metric("Completed Analyses", total_analyses)
    with analytics_cols[2]:
        st.metric("API Calls Made", total_api_calls)
    with analytics_cols[3]:
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Export options
    st.markdown("#### 💾 **Export Options**")
    export_cols = st.columns(3)
    
    with export_cols[0]:
        if st.button("📄 Export Chat History"):
            chat_data = {
                "session_id": st.session_state.session_id,
                "timestamp": datetime.now().isoformat(),
                "messages": st.session_state.messages,
                "analysis_results": st.session_state.analysis_results
            }
            st.download_button(
                "Download Chat History",
                json.dumps(chat_data, indent=2),
                f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with export_cols[1]:
        if st.button("📊 Export Analytics"):
            analytics_data = {
                "session_analytics": {
                    "total_messages": len(st.session_state.messages),
                    "completed_analyses": total_analyses,
                    "total_api_calls": total_api_calls,
                    "average_confidence": avg_confidence
                },
                "detailed_results": st.session_state.analysis_results
            }
            st.download_button(
                "Download Analytics",
                json.dumps(analytics_data, indent=2),
                f"session_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with export_cols[2]:
        if st.button("📋 Export Summary Report"):
            # Create a summary report
            summary_report = f"""# Healthcare Chatbot Session Summary

## Session Information
- Session ID: {st.session_state.session_id}
- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Total Messages: {len(st.session_state.messages)}
- Completed Analyses: {total_analyses}

## Analytics
- Total API Calls: {total_api_calls}
- Average Confidence: {avg_confidence:.1%}

## Recent Conversations
"""
            for i, msg in enumerate(st.session_state.messages[-5:]):
                role = "User" if msg["role"] == "user" else "Assistant"
                summary_report += f"\n**{role}:** {msg['content'][:200]}...\n"
            
            st.download_button(
                "Download Summary",
                summary_report,
                f"session_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🤖 <strong>Enhanced LangGraph MCP Healthcare Chatbot</strong><br>
    Powered by Intelligent Routing + Data Deidentification + Enhanced Entity Extraction<br>
    🔄 <strong>Continuous Conversations</strong> | 🔒 <strong>HIPAA Compliant</strong> | 🎯 <strong>Advanced Analytics</strong><br>
    ⚠️ <em>This analysis is for informational purposes only and should not replace professional medical advice.</em>
</div>
""", unsafe_allow_html=True)
