import streamlit as st
import requests
import uuid
import time
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Any
import re

# Try to import config, fallback to hardcoded values
try:
    from config import SERVER_CONFIG
    APP_PORT = SERVER_CONFIG["app_port"]
    MCP_PORT = SERVER_CONFIG["mcp_port"]
except ImportError:
    APP_PORT = 8081
    MCP_PORT = 8000

# Page configuration
st.set_page_config(
    page_title="Neo4j AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for elegant styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
    }
    
    .user-message {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-left: 4px solid #4c63d2;
    }
    
    .bot-message {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-left: 4px solid #e91e63;
    }
    
    .status-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .highlight {
        background: linear-gradient(120deg, #a8edea 0%, #fed6e3 100%);
        padding: 0.2rem 0.4rem;
        border-radius: 0.3rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "successful_queries" not in st.session_state:
    st.session_state.successful_queries = 0
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = None

def check_service_health():
    """Check the health of backend services"""
    services = {
        "Main App": f"http://localhost:{APP_PORT}/health",
        "MCP Server": f"http://localhost:{MCP_PORT}/health"
    }
    
    status = {}
    for service, url in services.items():
        try:
            response = requests.get(url, timeout=3)
            status[service] = "🟢 Online" if response.status_code == 200 else "🟡 Issues"
        except requests.exceptions.RequestException:
            status[service] = "🔴 Offline"
    
    return status

def get_system_stats():
    """Get system statistics"""
    try:
        response = requests.get(f"http://localhost:{APP_PORT}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return {}

def format_query_result(result: Dict[str, Any]) -> str:
    """Format query result for display"""
    if not result.get("success", True):
        return f"❌ **Error:** {result.get('error', 'Unknown error')}"
    
    answer = result.get("answer", "No answer provided")
    
    # Add some visual formatting
    if "**Result:**" in answer:
        answer = answer.replace("**Result:**", "📊 **Result:**")
    
    if answer.startswith("✅"):
        answer = f"🎉 {answer}"
    
    return answer

def create_query_stats_chart():
    """Create a chart showing query statistics"""
    if not st.session_state.query_history:
        return None
    
    # Create DataFrame from query history
    df = pd.DataFrame(st.session_state.query_history)
    
    if len(df) > 0:
        # Group by tool used
        tool_counts = df.groupby('tool').size().reset_index(name='count')
        
        # Create pie chart
        fig = px.pie(
            tool_counts, 
            values='count', 
            names='tool',
            title="Query Distribution by Tool",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(
            height=300,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    
    return None

def create_response_time_chart():
    """Create a chart showing response times"""
    if len(st.session_state.query_history) < 2:
        return None
    
    df = pd.DataFrame(st.session_state.query_history)
    
    if len(df) > 0 and 'response_time' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df['response_time'],
            mode='lines+markers',
            name='Response Time',
            line=dict(color='#667eea', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Response Time Trends",
            xaxis_title="Query Number",
            yaxis_title="Response Time (seconds)",
            height=300,
            showlegend=False
        )
        return fig
    
    return None

def send_query(question: str) -> Dict[str, Any]:
    """Send query to the backend and return response"""
    payload = {
        "question": question,
        "session_id": st.session_state.session_id
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"http://localhost:{APP_PORT}/chat", 
            json=payload,
            timeout=60
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            result["response_time"] = response_time
            
            # Update statistics
            st.session_state.query_count += 1
            if result.get("success", True):
                st.session_state.successful_queries += 1
            
            # Add to history
            st.session_state.query_history.append({
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "tool": result.get("tool", "unknown"),
                "success": result.get("success", True),
                "response_time": response_time
            })
            
            st.session_state.last_query_time = response_time
            
            return result
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "trace": "",
                "tool": "",
                "query": "",
                "answer": f"❌ Server error: {response.status_code}"
            }
    
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timeout",
            "trace": "",
            "tool": "",
            "query": "",
            "answer": "⏰ Request timed out. Please try again."
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Connection error",
            "trace": "",
            "tool": "",
            "query": "",
            "answer": "🔌 Cannot connect to backend. Please check if services are running."
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "trace": "",
            "tool": "",
            "query": "",
            "answer": f"❌ Unexpected error: {str(e)}"
        }

# Header
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="color: #667eea; margin-bottom: 0.5rem;">🧠 Neo4j AI Assistant</h1>
    <p style="color: #666; font-size: 1.1rem;">Natural Language Interface for Graph Database Queries</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    
    # Service Status
    st.markdown("### 🔧 Service Status")
    services = check_service_health()
    
    for service, status in services.items():
        st.markdown(f"**{service}:** {status}")
    
    # System Stats
    st.markdown("### 📊 System Statistics")
    stats = get_system_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Queries", st.session_state.query_count)
    with col2:
        st.metric("Successful", st.session_state.successful_queries)
    
    if st.session_state.query_count > 0:
        success_rate = (st.session_state.successful_queries / st.session_state.query_count) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    if st.session_state.last_query_time:
        st.metric("Last Response Time", f"{st.session_state.last_query_time:.2f}s")
    
    # Query Examples
    st.markdown("### 💡 Quick Examples")
    
    example_queries = [
        "How many nodes are in the graph?",
        "Show me the database schema",
        "List all Person nodes",
        "Find nodes with most relationships",
        "Count all relationships",
        "What node types exist?",
        "Show me recent activity",
        "Create a test node",
        "Delete temporary data"
    ]
    
    selected_example = st.selectbox(
        "Choose an example query:",
        ["Select an example..."] + example_queries
    )
    
    if selected_example != "Select an example...":
        if st.button("🚀 Run Example", key="run_example"):
            st.session_state.selected_query = selected_example
            st.rerun()
    
    # Export Options
    st.markdown("### 📥 Export Options")
    
    if st.button("📊 Export Chat History"):
        if st.session_state.messages:
            export_data = {
                "session_id": st.session_state.session_id,
                "timestamp": datetime.now().isoformat(),
                "messages": st.session_state.messages,
                "statistics": {
                    "query_count": st.session_state.query_count,
                    "successful_queries": st.session_state.successful_queries,
                    "query_history": st.session_state.query_history
                }
            }
            
            st.download_button(
                label="💾 Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"neo4j_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.warning("No chat history to export")
    
    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.query_history = []
        st.session_state.query_count = 0
        st.session_state.successful_queries = 0
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Chat Interface
    st.markdown("### 💬 Chat Interface")
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask me anything about your Neo4j database:",
            placeholder="e.g., How many nodes are in the graph?",
            key="user_input"
        )
        
        col_submit, col_voice = st.columns([3, 1])
        
        with col_submit:
            submitted = st.form_submit_button("🚀 Send Query", use_container_width=True)
        
        with col_voice:
            voice_input = st.form_submit_button("🎤 Voice", use_container_width=True)
    
    # Handle form submission or example selection
    query_to_process = None
    
    if submitted and user_input:
        query_to_process = user_input
    elif hasattr(st.session_state, 'selected_query'):
        query_to_process = st.session_state.selected_query
        delattr(st.session_state, 'selected_query')
    
    if query_to_process:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": query_to_process,
            "timestamp": datetime.now().isoformat()
        })
        
        # Show loading spinner
        with st.spinner("🤔 Processing your query..."):
            result = send_query(query_to_process)
        
        # Add bot response
        st.session_state.messages.append({
            "role": "assistant",
            "content": result,
            "timestamp": datetime.now().isoformat()
        })
        
        # Show success or error message
        if result.get("success", True):
            st.success(f"✅ Query processed successfully in {result.get('response_time', 0):.2f}s")
        else:
            st.error(f"❌ Query failed: {result.get('error', 'Unknown error')}")
    
    # Chat Messages Display
    st.markdown("### 📝 Conversation History")
    
    if st.session_state.messages:
        for i, message in enumerate(reversed(st.session_state.messages)):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🧑 You:</strong><br>
                    {message["content"]}
                    <br><small style="opacity: 0.7;">⏰ {message.get("timestamp", "")}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                result = message["content"]
                formatted_answer = format_query_result(result)
                
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 AI Assistant:</strong><br>
                    {formatted_answer}
                    <br><small style="opacity: 0.7;">⏰ {message.get("timestamp", "")}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Show technical details in expandable section
                if result.get("tool") or result.get("query"):
                    with st.expander("🔍 Technical Details"):
                        if result.get("tool"):
                            st.markdown(f"**Tool Used:** `{result['tool']}`")
                        if result.get("query"):
                            st.markdown(f"**Cypher Query:**")
                            st.code(result["query"], language="cypher")
                        if result.get("trace"):
                            st.markdown(f"**Trace:**")
                            st.text(result["trace"])
    else:
        st.info("👋 Welcome! Start by asking a question about your Neo4j database.")

with col2:
    # Analytics and Visualizations
    st.markdown("### 📈 Analytics")
    
    # Query distribution chart
    chart = create_query_stats_chart()
    if chart:
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("📊 Query statistics will appear here after you make some queries.")
    
    # Response time chart
    time_chart = create_response_time_chart()
    if time_chart:
        st.plotly_chart(time_chart, use_container_width=True)
    
    # Recent Activity
    st.markdown("### ⚡ Recent Activity")
    
    if st.session_state.query_history:
        recent_queries = st.session_state.query_history[-5:]  # Last 5 queries
        
        for query in reversed(recent_queries):
            status_icon = "✅" if query.get("success", True) else "❌"
            st.markdown(f"""
            <div class="status-card">
                {status_icon} <strong>{query.get('tool', 'Unknown')}</strong><br>
                <small>{query.get('question', 'No question')[:50]}...</small><br>
                <small>⏱️ {query.get('response_time', 0):.2f}s</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("🔄 Recent activity will appear here.")
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🔍 Check Database Health", use_container_width=True):
        st.session_state.selected_query = "How many nodes are in the graph?"
        st.rerun()
    
    if st.button("📊 Show Schema", use_container_width=True):
        st.session_state.selected_query = "Show me the database schema"
        st.rerun()
    
    if st.button("🔢 Count Relationships", use_container_width=True):
        st.session_state.selected_query = "Count all relationships"
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🧠 <strong>Neo4j AI Assistant</strong> | Powered by LangGraph + Snowflake Cortex</p>
    <p>Session ID: <code>{}</code></p>
</div>
""".format(st.session_state.session_id), unsafe_allow_html=True)
