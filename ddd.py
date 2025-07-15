# streamlit_detailed_neo4j.py - Complete Streamlit app with detailed LangGraph agent

import streamlit as st
import asyncio
import nest_asyncio
import json
import time
from detailed_langgraph_neo4j import DetailedNeo4jAgent

nest_asyncio.apply()

def run_async_safe(coro):
    """Run async function safely in Streamlit."""
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

# Page configuration
st.set_page_config(
    page_title="Neo4j LangGraph Agent", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .error-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "detailed_agent" not in st.session_state:
    with st.spinner("🧠 Initializing Intelligent Neo4j Agent..."):
        st.session_state.detailed_agent = DetailedNeo4jAgent()

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "agent_stats" not in st.session_state:
    st.session_state.agent_stats = {
        "total_questions": 0,
        "successful_answers": 0,
        "failed_answers": 0,
        "query_fixes_applied": 0
    }

# Sidebar for agent controls and stats
with st.sidebar:
    st.markdown("## 🧠 Agent Controls")
    
    # Agent status
    st.markdown("### 📊 Agent Status")
    if st.button("🔍 Test Agent Health"):
        with st.spinner("Testing..."):
            result = run_async_safe(
                st.session_state.detailed_agent.call_mcp_tool("health_check")
            )
        
        if result.startswith("❌"):
            st.error(f"❌ Health check failed: {result}")
        else:
            try:
                health_data = json.loads(result)
                st.success(f"✅ Agent healthy! Database: {health_data.get('database', 'N/A')}")
            except:
                st.success("✅ Agent is responding")
    
    # Statistics
    st.markdown("### 📈 Performance Stats")
    stats = st.session_state.agent_stats
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions", stats["total_questions"])
        st.metric("Successful", stats["successful_answers"])
    with col2:
        st.metric("Failed", stats["failed_answers"])
        st.metric("Fixes Applied", stats["query_fixes_applied"])
    
    # Success rate
    if stats["total_questions"] > 0:
        success_rate = (stats["successful_answers"] / stats["total_questions"]) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Clear history
    if st.button("🗑️ Clear History"):
        st.session_state.conversation_history = []
        st.session_state.agent_stats = {
            "total_questions": 0,
            "successful_answers": 0,
            "failed_answers": 0,
            "query_fixes_applied": 0
        }
        st.rerun()

# Main interface
st.title("🧠 Intelligent Neo4j Assistant")
st.markdown("**Powered by LangGraph + Your MCP Server**")

# Feature highlights
st.markdown("""
<div class="success-card">
    <h4>🚀 Advanced Features</h4>
    <ul>
        <li><strong>Smart Error Recovery:</strong> Automatically fixes Neo4j syntax issues</li>
        <li><strong>Context Awareness:</strong> Uses your database schema for better queries</li>
        <li><strong>Multi-step Reasoning:</strong> Breaks down complex questions</li>
        <li><strong>Adaptive Responses:</strong> Formats answers based on question type</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Example questions section
st.markdown("## 💡 Try These Intelligent Questions")

example_categories = {
    "🔗 Connectivity Analysis": [
        "Show me the most connected nodes in the database",
        "Find nodes with the highest degree centrality", 
        "Which entities have the most relationships?"
    ],
    "📊 Database Analytics": [
        "What's the overall structure of this database?",
        "Give me statistics about node and relationship counts",
        "Analyze the distribution of different node types"
    ],
    "🔍 Smart Exploration": [
        "Find interesting patterns in the data",
        "Show me a sample of different node types",
        "What are the main categories of information stored here?"
    ],
    "🏗️ Schema Investigation": [
        "What properties do the different node types have?",
        "List all the relationship types and their usage",
        "Explain the database schema structure"
    ]
}

# Create tabs for different question categories
tabs = st.tabs(list(example_categories.keys()))

for tab, (category, questions) in zip(tabs, example_categories.items()):
    with tab:
        for i, question in enumerate(questions):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**{i+1}.** {question}")
            with col2:
                if st.button("Ask", key=f"{category}_{i}"):
                    st.session_state.conversation_history.append(("user", question))
                    
                    # Process with agent
                    with st.spinner("🧠 Agent thinking..."):
                        progress_container = st.container()
                        with progress_container:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Show progress
                            status_text.text("🔍 Analyzing question...")
                            progress_bar.progress(20)
                            time.sleep(0.5)
                            
                            status_text.text("📊 Gathering schema info...")
                            progress_bar.progress(40)
                            time.sleep(0.5)
                            
                            status_text.text("🤖 Generating optimized query...")
                            progress_bar.progress(60)
                            time.sleep(0.5)
                            
                            status_text.text("⚡ Executing and formatting...")
                            progress_bar.progress(80)
                            
                            # Run the agent
                            answer = run_async_safe(st.session_state.detailed_agent.run(question))
                            
                            progress_bar.progress(100)
                            status_text.text("✅ Complete!")
                            time.sleep(0.5)
                            
                            # Clear progress
                            progress_container.empty()
                    
                    # Update stats
                    st.session_state.agent_stats["total_questions"] += 1
                    if not answer.startswith("❌"):
                        st.session_state.agent_stats["successful_answers"] += 1
                    else:
                        st.session_state.agent_stats["failed_answers"] += 1
                    
                    st.session_state.conversation_history.append(("agent", answer))
                    st.rerun()

# Main chat interface
st.markdown("## 💬 Intelligent Chat")

# Chat input
user_question = st.chat_input("Ask any complex Neo4j question...")

if user_question:
    # Add to history
    st.session_state.conversation_history.append(("user", user_question))
    
    # Process with detailed progress tracking
    with st.spinner("🧠 Intelligent Agent Processing..."):
        
        # Create progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Initialize
            status_text.text("🚀 Initializing agent...")
            progress_bar.progress(10)
            time.sleep(0.3)
            
            # Step 2: Question analysis
            status_text.text("🧠 Analyzing question complexity...")
            progress_bar.progress(25)
            time.sleep(0.4)
            
            # Step 3: Schema gathering
            status_text.text("📊 Gathering database schema...")
            progress_bar.progress(45)
            time.sleep(0.5)
            
            # Step 4: Query generation
            status_text.text("🤖 Generating optimized Cypher...")
            progress_bar.progress(65)
            time.sleep(0.4)
            
            # Step 5: Execution
            status_text.text("⚡ Executing query...")
            progress_bar.progress(80)
            time.sleep(0.3)
            
            # Step 6: Formatting
            status_text.text("📝 Formatting intelligent response...")
            progress_bar.progress(95)
            
            # Actually run the agent
            answer = run_async_safe(st.session_state.detailed_agent.run(user_question))
            
            # Complete
            progress_bar.progress(100)
            status_text.text("✅ Response ready!")
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_container.empty()
    
    # Update statistics
    st.session_state.agent_stats["total_questions"] += 1
    if not answer.startswith("❌"):
        st.session_state.agent_stats["successful_answers"] += 1
        if "🔧" in answer or "syntax fix" in answer.lower():
            st.session_state.agent_stats["query_fixes_applied"] += 1
    else:
        st.session_state.agent_stats["failed_answers"] += 1
    
    # Add to conversation
    st.session_state.conversation_history.append(("agent", answer))
    st.rerun()

# Display conversation history
if st.session_state.conversation_history:
    st.markdown("## 📜 Conversation History")
    
    # Show last 10 exchanges
    recent_history = st.session_state.conversation_history[-20:]  # Last 10 exchanges (20 messages)
    
    for i, (role, message) in enumerate(reversed(recent_history)):
        if role == "user":
            st.chat_message("user").write(f"**You:** {message}")
        elif role == "agent":
            # Determine message type for styling
            if message.startswith("❌"):
                st.chat_message("assistant").error(message)
            elif message.startswith("🔄"):
                st.chat_message("assistant").warning(message)
            else:
                st.chat_message("assistant").success(message)

# Advanced features section
st.markdown("---")
st.markdown("## 🔧 Advanced Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h4>🛠️ Error Recovery</h4>
        <p>Automatically detects and fixes:</p>
        <ul>
            <li>Deprecated size() syntax</li>
            <li>Missing LIMIT clauses</li>
            <li>Old Neo4j patterns</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h4>🧠 Smart Context</h4>
        <p>Uses your database info:</p>
        <ul>
            <li>Available node labels</li>
            <li>Relationship types</li>
            <li>Database statistics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h4>🎯 Adaptive Responses</h4>
        <p>Intelligent formatting for:</p>
        <ul>
            <li>Connectivity analysis</li>
            <li>Schema exploration</li>
            <li>Statistical queries</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Debug information
if st.checkbox("🔧 Show Debug Information"):
    st.markdown("### 🐛 Debug Info")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Agent Configuration:**")
        st.json({
            "MCP Script": "mcpserver.py",
            "Max Attempts": 3,
            "Cortex Model": "llama3.1-70b",
            "Available Tools": [
                "read_neo4j_cypher",
                "write_neo4j_cypher", 
                "list_labels",
                "list_relationships",
                "database_summary",
                "health_check"
            ]
        })
    
    with col2:
        st.markdown("**Session Statistics:**")
        st.json(st.session_state.agent_stats)

# Footer
st.markdown("---")
st.markdown("""
<div class="success-card">
    <h4>✨ You're using the most advanced Neo4j assistant!</h4>
    <p>This system combines your existing MCP server with LangGraph's intelligent workflow 
    to provide context-aware, error-recovering, and adaptively formatted responses to any Neo4j question.</p>
</div>
""", unsafe_allow_html=True)
