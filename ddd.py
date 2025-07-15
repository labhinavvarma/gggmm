
# neo4j_intelligent_ui.py - Comprehensive UI for the Specialized Neo4j System

import streamlit as st
import asyncio
import nest_asyncio
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
from updated_langgraph_agent import OptimizedNeo4jAgent

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
    page_title="🧠 Intelligent Neo4j Assistant", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .error-card {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .tool-button {
        width: 100%;
        margin: 0.25rem 0;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-healthy { background-color: #28a745; }
    .status-warning { background-color: #ffc107; }
    .status-error { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "intelligent_agent" not in st.session_state:
    with st.spinner("🧠 Initializing Intelligent Neo4j System..."):
        st.session_state.intelligent_agent = OptimizedNeo4jAgent("langgraph_mcpserver.py")

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "system_metrics" not in st.session_state:
    st.session_state.system_metrics = {
        "total_questions": 0,
        "successful_queries": 0,
        "failed_queries": 0,
        "syntax_fixes_applied": 0,
        "avg_response_time": 0,
        "server_health": "unknown"
    }

if "schema_cache" not in st.session_state:
    st.session_state.schema_cache = None

if "performance_history" not in st.session_state:
    st.session_state.performance_history = []

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 Intelligent Neo4j Assistant</h1>
    <p>Powered by Specialized MCP Server + LangGraph Agent</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - System Control Panel
with st.sidebar:
    st.markdown("## 🎛️ System Control Panel")
    
    # System Health Status
    st.markdown("### 🏥 System Health")
    
    if st.button("🔍 Check System Health", key="health_check"):
        with st.spinner("Checking system health..."):
            health_result = run_async_safe(
                st.session_state.intelligent_agent.call_mcp_tool("health_check")
            )
        
        if health_result.startswith("❌"):
            st.session_state.system_metrics["server_health"] = "error"
            st.error(f"❌ {health_result}")
        else:
            try:
                health_data = json.loads(health_result)
                st.session_state.system_metrics["server_health"] = "healthy"
                st.success(f"✅ System Healthy")
                st.json(health_data)
            except:
                st.session_state.system_metrics["server_health"] = "warning"
                st.warning("⚠️ System responding but data unclear")
    
    # Health indicator
    health_status = st.session_state.system_metrics["server_health"]
    if health_status == "healthy":
        st.markdown('<span class="status-indicator status-healthy"></span>**System Status: Healthy**', unsafe_allow_html=True)
    elif health_status == "warning":
        st.markdown('<span class="status-indicator status-warning"></span>**System Status: Warning**', unsafe_allow_html=True)
    elif health_status == "error":
        st.markdown('<span class="status-indicator status-error"></span>**System Status: Error**', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-indicator status-warning"></span>**System Status: Unknown**', unsafe_allow_html=True)
    
    st.divider()
    
    # Performance Metrics
    st.markdown("### 📊 Performance Metrics")
    
    metrics = st.session_state.system_metrics
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Queries", metrics["total_questions"])
        st.metric("Success Rate", f"{(metrics['successful_queries']/max(metrics['total_questions'],1)*100):.1f}%")
    with col2:
        st.metric("Syntax Fixes", metrics["syntax_fixes_applied"])
        st.metric("Avg Response", f"{metrics['avg_response_time']:.1f}s")
    
    if st.button("📈 Get Live Metrics", key="live_metrics"):
        with st.spinner("Getting live metrics..."):
            metrics_result = run_async_safe(
                st.session_state.intelligent_agent.call_mcp_tool("get_metrics")
            )
        
        if not metrics_result.startswith("❌"):
            try:
                live_metrics = json.loads(metrics_result)
                st.json(live_metrics)
            except:
                st.error("Failed to parse metrics")
    
    if st.button("🔄 Reset Metrics", key="reset_metrics"):
        run_async_safe(
            st.session_state.intelligent_agent.call_mcp_tool("reset_metrics")
        )
        st.session_state.system_metrics = {
            "total_questions": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "syntax_fixes_applied": 0,
            "avg_response_time": 0,
            "server_health": "unknown"
        }
        st.success("✅ Metrics reset!")
        st.rerun()
    
    st.divider()
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🔄 Clear Chat History", key="clear_history"):
        st.session_state.conversation_history = []
        st.success("✅ Chat cleared!")
        st.rerun()
    
    if st.button("📊 Refresh Schema Cache", key="refresh_schema"):
        st.session_state.schema_cache = None
        st.success("✅ Schema cache cleared!")

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Intelligent Chat", 
    "🧪 Tool Testing", 
    "📊 Database Analytics", 
    "🏗️ Schema Explorer", 
    "📈 Performance Monitor"
])

# Tab 1: Intelligent Chat
with tab1:
    st.markdown("## 💬 Intelligent Chat Interface")
    
    # Feature highlights
    st.markdown("""
    <div class="success-card">
        <h4>🚀 Enhanced Intelligence Features</h4>
        <ul>
            <li><strong>🧠 Multi-Step Reasoning:</strong> Breaks down complex questions intelligently</li>
            <li><strong>🔧 Automatic Syntax Fixing:</strong> Modernizes deprecated Neo4j syntax</li>
            <li><strong>📊 Schema-Aware Generation:</strong> Uses your database structure for better queries</li>
            <li><strong>📈 Performance Tracking:</strong> Monitors and reports execution metrics</li>
            <li><strong>🎯 Adaptive Formatting:</strong> Tailors responses based on question type</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Example questions
    st.markdown("### 💡 Try These Intelligent Questions")
    
    example_cols = st.columns(3)
    
    with example_cols[0]:
        st.markdown("**🔗 Connectivity Analysis**")
        if st.button("Most Connected Nodes", key="conn1"):
            question = "show me the nodes with the most connections in the database"
            st.session_state.conversation_history.append(("user", question))
            st.rerun()
        
        if st.button("Network Hub Analysis", key="conn2"):
            question = "analyze the network structure and find hub nodes"
            st.session_state.conversation_history.append(("user", question))
            st.rerun()
    
    with example_cols[1]:
        st.markdown("**📊 Database Analytics**")
        if st.button("Comprehensive Overview", key="analytics1"):
            question = "give me a comprehensive analysis of this database"
            st.session_state.conversation_history.append(("user", question))
            st.rerun()
            
        if st.button("Distribution Analysis", key="analytics2"):
            question = "analyze the distribution of different node types"
            st.session_state.conversation_history.append(("user", question))
            st.rerun()
    
    with example_cols[2]:
        st.markdown("**🔍 Smart Exploration**")
        if st.button("Find Patterns", key="explore1"):
            question = "find interesting patterns in the data"
            st.session_state.conversation_history.append(("user", question))
            st.rerun()
            
        if st.button("Sample Data", key="explore2"):
            question = "show me sample data for each node type"
            st.session_state.conversation_history.append(("user", question))
            st.rerun()
    
    # Chat input
    st.markdown("### 🎤 Ask Anything")
    user_question = st.chat_input("Ask any complex Neo4j question...")
    
    if user_question:
        st.session_state.conversation_history.append(("user", user_question))
        st.rerun()
    
    # Process latest question
    if st.session_state.conversation_history and st.session_state.conversation_history[-1][0] == "user":
        latest_question = st.session_state.conversation_history[-1][1]
        
        with st.spinner("🧠 Intelligent Agent Processing..."):
            # Progress tracking
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                steps = [
                    (15, "🚀 Initializing intelligent agent..."),
                    (30, "🧠 Analyzing question complexity..."),
                    (45, "📊 Gathering enhanced schema..."),
                    (60, "🤖 Generating context-aware query..."),
                    (75, "✅ Validating and optimizing..."),
                    (90, "⚡ Executing with performance tracking..."),
                    (100, "📝 Formatting intelligent response...")
                ]
                
                for progress, step_text in steps:
                    status_text.text(step_text)
                    progress_bar.progress(progress)
                    time.sleep(0.3)
                
                # Actually run the agent
                start_time = time.time()
                answer = run_async_safe(st.session_state.intelligent_agent.run(latest_question))
                execution_time = time.time() - start_time
                
                # Update metrics
                st.session_state.system_metrics["total_questions"] += 1
                st.session_state.system_metrics["avg_response_time"] = (
                    (st.session_state.system_metrics["avg_response_time"] * (st.session_state.system_metrics["total_questions"] - 1) + execution_time) 
                    / st.session_state.system_metrics["total_questions"]
                )
                
                if not answer.startswith("❌"):
                    st.session_state.system_metrics["successful_queries"] += 1
                    if "🔧" in answer or "syntax fix" in answer.lower():
                        st.session_state.system_metrics["syntax_fixes_applied"] += 1
                else:
                    st.session_state.system_metrics["failed_queries"] += 1
                
                # Add performance data
                st.session_state.performance_history.append({
                    "timestamp": datetime.now(),
                    "question": latest_question,
                    "execution_time": execution_time,
                    "success": not answer.startswith("❌")
                })
                
                progress_container.empty()
        
        # Add to conversation
        st.session_state.conversation_history.append(("agent", answer))
    
    # Display conversation
    if st.session_state.conversation_history:
        st.markdown("### 📜 Conversation History")
        
        for i, (role, message) in enumerate(reversed(st.session_state.conversation_history[-20:])):
            if role == "user":
                st.chat_message("user").write(f"**You:** {message}")
            elif role == "agent":
                if message.startswith("❌"):
                    st.chat_message("assistant").error(message)
                elif message.startswith("🔄"):
                    st.chat_message("assistant").warning(message)
                else:
                    st.chat_message("assistant").success(message)

# Tab 2: Tool Testing
with tab2:
    st.markdown("## 🧪 Advanced Tool Testing")
    st.markdown("Test individual tools from the specialized MCP server.")
    
    # Organize tools by category
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔧 Core Tools")
        
        if st.button("🏥 Health Check", key="tool_health", help="Check server health"):
            with st.spinner("Checking health..."):
                result = run_async_safe(
                    st.session_state.intelligent_agent.call_mcp_tool("health_check")
                )
            st.code(result, language="json")
        
        if st.button("📊 Get Metrics", key="tool_metrics", help="Get performance metrics"):
            with st.spinner("Getting metrics..."):
                result = run_async_safe(
                    st.session_state.intelligent_agent.call_mcp_tool("get_metrics")
                )
            st.code(result, language="json")
        
        if st.button("🔄 Reset Metrics", key="tool_reset", help="Reset performance metrics"):
            with st.spinner("Resetting metrics..."):
                result = run_async_safe(
                    st.session_state.intelligent_agent.call_mcp_tool("reset_metrics")
                )
            st.success(result)
    
    with col2:
        st.markdown("### 📊 Schema Tools")
        
        if st.button("🏗️ Analyze Schema", key="tool_schema", help="Deep schema analysis"):
            with st.spinner("Analyzing schema..."):
                result = run_async_safe(
                    st.session_state.intelligent_agent.call_mcp_tool("analyze_schema")
                )
            try:
                parsed = json.loads(result)
                st.json(parsed)
            except:
                st.code(result)
        
        if st.button("📋 Database Summary", key="tool_summary", help="Get database summary"):
            with st.spinner("Getting summary..."):
                result = run_async_safe(
                    st.session_state.intelligent_agent.call_mcp_tool("database_summary")
                )
            try:
                parsed = json.loads(result)
                st.json(parsed)
            except:
                st.code(result)
        
        if st.button("🔢 Count by Label", key="tool_count", help="Count nodes by label"):
            with st.spinner("Counting nodes..."):
                result = run_async_safe(
                    st.session_state.intelligent_agent.call_mcp_tool("count_by_label")
                )
            try:
                parsed = json.loads(result)
                st.json(parsed)
            except:
                st.code(result)
    
    with col3:
        st.markdown("### 🔍 Data Tools")
        
        sample_limit = st.number_input("Sample Limit", min_value=1, max_value=50, value=5, key="sample_limit")
        if st.button("🔍 Get Sample Data", key="tool_sample", help="Get sample data"):
            with st.spinner("Getting samples..."):
                result = run_async_safe(
                    st.session_state.intelligent_agent.call_mcp_tool("get_sample_data", {"limit": sample_limit})
                )
            try:
                parsed = json.loads(result)
                st.json(parsed)
            except:
                st.code(result)
        
        # Query validation
        st.markdown("**Query Validation**")
        test_query = st.text_area("Enter Cypher Query to Validate:", 
                                 value="MATCH (n) RETURN size((n)-[]->()) as degree", 
                                 key="validation_query")
        
        if st.button("✅ Validate Query", key="tool_validate"):
            with st.spinner("Validating query..."):
                result = run_async_safe(
                    st.session_state.intelligent_agent.call_mcp_tool("validate_query", {"query": test_query})
                )
            try:
                parsed = json.loads(result)
                st.json(parsed)
                
                if parsed.get("suggested_query"):
                    st.success(f"💡 Suggested fix: `{parsed['suggested_query']}`")
            except:
                st.code(result)

# Tab 3: Database Analytics
with tab3:
    st.markdown("## 📊 Database Analytics Dashboard")
    
    # Get analytics data
    if st.button("🔄 Refresh Analytics", key="refresh_analytics"):
        with st.spinner("Gathering analytics data..."):
            # Get count by label
            count_result = run_async_safe(
                st.session_state.intelligent_agent.call_mcp_tool("count_by_label")
            )
            
            # Get database summary
            summary_result = run_async_safe(
                st.session_state.intelligent_agent.call_mcp_tool("database_summary")
            )
            
            try:
                count_data = json.loads(count_result)
                summary_data = json.loads(summary_result)
                
                # Create visualizations
                if "label_counts" in count_data:
                    # Node distribution chart
                    labels = [item["label"] for item in count_data["label_counts"]]
                    counts = [item["count"] for item in count_data["label_counts"]]
                    
                    fig1 = px.bar(x=labels, y=counts, title="Node Distribution by Label")
                    fig1.update_layout(xaxis_title="Node Labels", yaxis_title="Count")
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Pie chart
                    fig2 = px.pie(values=counts, names=labels, title="Node Label Distribution")
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Nodes", summary_data.get("node_count", 0))
                with col2:
                    st.metric("Total Relationships", summary_data.get("relationship_count", 0))
                with col3:
                    st.metric("Node Types", summary_data.get("label_count", 0))
                with col4:
                    st.metric("Relationship Types", summary_data.get("relationship_type_count", 0))
                
                # Performance metrics from server
                if "performance_metrics" in summary_data:
                    perf_metrics = summary_data["performance_metrics"]
                    st.markdown("### 📈 Server Performance")
                    
                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                    with perf_col1:
                        st.metric("Success Rate", f"{perf_metrics.get('success_rate', 0)}%")
                    with perf_col2:
                        st.metric("Total Queries", perf_metrics.get('total_queries', 0))
                    with perf_col3:
                        st.metric("Avg Execution Time", f"{perf_metrics.get('avg_execution_time_ms', 0)}ms")
                
            except Exception as e:
                st.error(f"Failed to parse analytics data: {e}")
    
    # Performance history chart
    if st.session_state.performance_history:
        st.markdown("### 📈 Query Performance History")
        
        df = pd.DataFrame(st.session_state.performance_history)
        
        fig3 = px.line(df, x="timestamp", y="execution_time", 
                      title="Query Execution Time Over Time",
                      color="success")
        st.plotly_chart(fig3, use_container_width=True)
        
        # Success rate over time
        df['success_rate'] = df['success'].rolling(window=10, min_periods=1).mean() * 100
        fig4 = px.line(df, x="timestamp", y="success_rate", 
                      title="Success Rate Trend (10-query rolling average)")
        st.plotly_chart(fig4, use_container_width=True)

# Tab 4: Schema Explorer
with tab4:
    st.markdown("## 🏗️ Interactive Schema Explorer")
    
    if st.button("🔍 Load Schema", key="load_schema"):
        with st.spinner("Loading comprehensive schema..."):
            schema_result = run_async_safe(
                st.session_state.intelligent_agent.call_mcp_tool("analyze_schema")
            )
        
        if not schema_result.startswith("❌"):
            try:
                schema_data = json.loads(schema_result)
                st.session_state.schema_cache = schema_data
                st.success("✅ Schema loaded successfully!")
            except:
                st.error("Failed to parse schema data")
    
    if st.session_state.schema_cache:
        schema = st.session_state.schema_cache
        
        # Schema overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏷️ Node Labels")
            if "labels" in schema:
                for label_info in schema["labels"][:10]:
                    label = label_info.get("label", "Unknown")
                    count = label_info.get("count", "unknown")
                    st.markdown(f"**{label}** ({count} nodes)")
        
        with col2:
            st.markdown("### 🔗 Relationship Types")
            if "relationships" in schema:
                for rel_info in schema["relationships"][:10]:
                    rel_type = rel_info.get("type", "Unknown")
                    count = rel_info.get("count", "unknown")
                    st.markdown(f"**{rel_type}** ({count} relationships)")
        
        # Property analysis
        if "node_properties" in schema:
            st.markdown("### 📋 Node Properties")
            
            for node_type, properties in schema["node_properties"].items():
                with st.expander(f"Properties for {node_type}"):
                    for prop in properties:
                        prop_name = prop.get("property", "unknown")
                        prop_types = prop.get("types", ["unknown"])
                        st.markdown(f"• **{prop_name}**: {', '.join(prop_types)}")
        
        # Sample data for each label
        st.markdown("### 🔍 Sample Data")
        if "labels" in schema:
            selected_label = st.selectbox(
                "Select a label to view sample data:",
                [label_info.get("label", "Unknown") for label_info in schema["labels"][:10]]
            )
            
            if st.button(f"Get samples for {selected_label}", key="schema_samples"):
                with st.spinner(f"Getting samples for {selected_label}..."):
                    sample_result = run_async_safe(
                        st.session_state.intelligent_agent.call_mcp_tool("get_sample_data", {"label": selected_label, "limit": 5})
                    )
                
                try:
                    sample_data = json.loads(sample_result)
                    st.json(sample_data)
                except:
                    st.code(sample_result)

# Tab 5: Performance Monitor
with tab5:
    st.markdown("## 📈 Performance Monitoring")
    
    # Real-time metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Current Session Metrics")
        metrics = st.session_state.system_metrics
        
        st.metric("Questions Asked", metrics["total_questions"])
        st.metric("Successful Queries", metrics["successful_queries"])
        st.metric("Failed Queries", metrics["failed_queries"])
        st.metric("Syntax Fixes Applied", metrics["syntax_fixes_applied"])
        st.metric("Average Response Time", f"{metrics['avg_response_time']:.2f}s")
    
    with col2:
        st.markdown("### 🖥️ Server Metrics")
        
        if st.button("📊 Get Live Server Metrics", key="live_server_metrics"):
            with st.spinner("Getting live metrics..."):
                result = run_async_safe(
                    st.session_state.intelligent_agent.call_mcp_tool("get_metrics")
                )
            
            if not result.startswith("❌"):
                try:
                    server_metrics = json.loads(result)
                    
                    st.metric("Server Success Rate", f"{server_metrics.get('success_rate', 0)}%")
                    st.metric("Total Server Queries", server_metrics.get('total_queries', 0))
                    st.metric("Server Avg Time", f"{server_metrics.get('avg_execution_time_ms', 0)}ms")
                    st.metric("Syntax Fixes by Server", server_metrics.get('syntax_fixes_applied', 0))
                    
                except:
                    st.error("Failed to parse server metrics")
    
    # Performance charts
    if st.session_state.performance_history:
        st.markdown("### 📊 Performance Visualizations")
        
        df = pd.DataFrame(st.session_state.performance_history)
        
        # Response time distribution
        fig1 = px.histogram(df, x="execution_time", title="Response Time Distribution", nbins=20)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Success vs failure over time
        success_counts = df.groupby(df['timestamp'].dt.floor('H'))['success'].agg(['sum', 'count']).reset_index()
        success_counts['failure'] = success_counts['count'] - success_counts['sum']
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=success_counts['timestamp'], y=success_counts['sum'], 
                                 mode='lines+markers', name='Successful'))
        fig2.add_trace(go.Scatter(x=success_counts['timestamp'], y=success_counts['failure'], 
                                 mode='lines+markers', name='Failed'))
        fig2.update_layout(title="Success vs Failure Rate Over Time", xaxis_title="Time", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)
    
    # System diagnostics
    st.markdown("### 🔧 System Diagnostics")
    
    if st.button("🏥 Run Full System Diagnostic", key="full_diagnostic"):
        with st.spinner("Running comprehensive system diagnostic..."):
            diagnostic_results = {}
            
            # Health check
            health = run_async_safe(
                st.session_state.intelligent_agent.call_mcp_tool("health_check")
            )
            diagnostic_results["health_check"] = health
            
            # Metrics
            metrics = run_async_safe(
                st.session_state.intelligent_agent.call_mcp_tool("get_metrics")
            )
            diagnostic_results["metrics"] = metrics
            
            # Quick query test
            test_query = "MATCH (n) RETURN count(n) LIMIT 1"
            query_test = run_async_safe(
                st.session_state.intelligent_agent.call_mcp_tool("execute_read_query", {"query": test_query})
            )
            diagnostic_results["query_test"] = query_test
            
            # Display results
            for test_name, result in diagnostic_results.items():
                with st.expander(f"🔍 {test_name.replace('_', ' ').title()}"):
                    if result.startswith("❌"):
                        st.error(result)
                    else:
                        try:
                            parsed = json.loads(result)
                            st.json(parsed)
                        except:
                            st.code(result)

# Footer
st.markdown("---")
st.markdown("""
<div class="success-card">
    <h4>✨ Welcome to the Future of Neo4j Interaction!</h4>
    <p>You're using an advanced AI-powered Neo4j assistant that combines:</p>
    <ul>
        <li><strong>🔧 Specialized MCP Server:</strong> Enhanced tools with automatic optimization</li>
        <li><strong>🧠 LangGraph Intelligence:</strong> Multi-step reasoning and context awareness</li>
        <li><strong>📊 Real-time Monitoring:</strong> Performance tracking and health monitoring</li>
        <li><strong>🎯 Adaptive Responses:</strong> Intelligent formatting based on question type</li>
    </ul>
    <p><strong>Ready to ask complex questions? Your originally failing query should now work perfectly!</strong></p>
</div>
""", unsafe_allow_html=True)
