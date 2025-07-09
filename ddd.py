# 1. ADD THIS IMPORT at the top of your file
import streamlit.components.v1 as components

# 2. UPDATE YOUR CSS - Add these styles to your existing CSS
# ADD this to your st.markdown CSS section:

"""
<style>
/* Your existing styles... */

.chat-message {
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    border-radius: 8px;
}

.user-message {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left: 3px solid #2196f3;
}

.assistant-message {
    background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
    border-left: 3px solid #9c27b0;
    max-width: 100%;
    overflow-x: auto;
}

/* NEW: Graph container styling */
.graph-container {
    background: white;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border: 1px solid #e0e0e0;
}

/* NEW: Adjust assistant message for graphs */
.assistant-message-with-graph {
    background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
    border-left: 3px solid #9c27b0;
    max-width: 100%;
    overflow-x: auto;
    padding: 0.8rem 0.5rem; /* Reduced padding for graphs */
}

/* NEW: Text content in graph messages */
.graph-text-content {
    padding: 0.5rem;
    margin-bottom: 1rem;
}
</style>
"""

# 3. REPLACE YOUR CHAT DISPLAY SECTION with this enhanced version:

with st.sidebar:
    if st.session_state.analysis_results and st.session_state.analysis_results.get("chatbot_ready", False) and st.session_state.chatbot_context:
        st.title("💬 Medical Assistant")
        st.markdown("---")
        
        # Chat history at top
        chat_container = st.container()
        with chat_container:
            if st.session_state.chatbot_messages:
                for message in st.session_state.chatbot_messages:
                    if message['role'] == 'user':
                        st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        content = message["content"]
                        
                        # Check if content contains graphs (HTML)
                        if "<html>" in content and "</html>" in content:
                            # Handle full HTML graphs
                            st.markdown('<div class="assistant-message-with-graph"><strong>Assistant:</strong></div>', unsafe_allow_html=True)
                            
                            # Extract text content before the HTML
                            html_start = content.find("<html>")
                            text_before = content[:html_start].strip()
                            
                            if text_before:
                                st.markdown(f'<div class="graph-text-content">{text_before}</div>', unsafe_allow_html=True)
                            
                            # Display the graph in a container
                            st.markdown('<div class="graph-container">', unsafe_allow_html=True)
                            
                            # Extract and display HTML
                            html_content = content[html_start:]
                            components.html(html_content, height=600, scrolling=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        elif any(indicator in content for indicator in ["plotly", "fig.to_html", "📊", "💊", "🏥", "❤️"]) and "##" in content:
                            # Handle mixed content with potential embedded graphs
                            st.markdown('<div class="assistant-message-with-graph">', unsafe_allow_html=True)
                            st.markdown(f'<strong>Assistant:</strong><div class="graph-text-content">{content}</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        else:
                            # Regular text message
                            st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {content}</div>', unsafe_allow_html=True)
            else:
                st.info("👋 Hello! I can answer questions about the claims data analysis. Ask me anything!")
                st.info("💡 **Special Feature:** Ask about heart attack risk and I'll provide both ML model predictions and comprehensive LLM analysis for comparison!")
                st.info("🎨 **New:** Ask me to 'show medication timeline' or 'generate risk dashboard' for interactive visualizations!")
        
        # Chat input at bottom (always visible)
        st.markdown("---")
        user_question = st.chat_input("Ask about the claims data...")
        
        # Handle chat input
        if user_question:
            # Add user message
            st.session_state.chatbot_messages.append({"role": "user", "content": user_question})
            
            # Get bot response
            try:
                with st.spinner("Processing..."):
                    chatbot_response = st.session_state.agent.chat_with_data(
                        user_question, 
                        st.session_state.chatbot_context, 
                        st.session_state.chatbot_messages
                    )
                
                # Add assistant response
                st.session_state.chatbot_messages.append({"role": "assistant", "content": chatbot_response})
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # NEW: Quick Graph Buttons
        st.markdown("---")
        st.markdown("**🎨 Quick Visualizations:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💊 Med Timeline", use_container_width=True, key="quick_med_timeline"):
                # Auto-generate medication timeline
                st.session_state.chatbot_messages.append({"role": "user", "content": "show me a medication timeline"})
                
                try:
                    with st.spinner("🎨 Creating timeline..."):
                        response = st.session_state.agent.chat_with_data(
                            "show me a medication timeline", 
                            st.session_state.chatbot_context, 
                            st.session_state.chatbot_messages
                        )
                    st.session_state.chatbot_messages.append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("📊 Risk Dashboard", use_container_width=True, key="quick_risk_dash"):
                # Auto-generate risk dashboard
                st.session_state.chatbot_messages.append({"role": "user", "content": "generate a risk assessment dashboard"})
                
                try:
                    with st.spinner("📊 Creating dashboard..."):
                        response = st.session_state.agent.chat_with_data(
                            "generate a risk assessment dashboard", 
                            st.session_state.chatbot_context, 
                            st.session_state.chatbot_messages
                        )
                    st.session_state.chatbot_messages.append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        col3, col4 = st.columns(2)
        
        with col3:
            if st.button("🏥 Diagnosis Chart", use_container_width=True, key="quick_diag_chart"):
                # Auto-generate diagnosis timeline
                st.session_state.chatbot_messages.append({"role": "user", "content": "create a diagnosis timeline chart"})
                
                try:
                    with st.spinner("🏥 Creating chart..."):
                        response = st.session_state.agent.chat_with_data(
                            "create a diagnosis timeline chart", 
                            st.session_state.chatbot_context, 
                            st.session_state.chatbot_messages
                        )
                    st.session_state.chatbot_messages.append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with col4:
            if st.button("🥧 Med Distribution", use_container_width=True, key="quick_med_pie"):
                # Auto-generate medication pie chart
                st.session_state.chatbot_messages.append({"role": "user", "content": "show me a pie chart of medications"})
                
                try:
                    with st.spinner("🥧 Creating chart..."):
                        response = st.session_state.agent.chat_with_data(
                            "show me a pie chart of medications", 
                            st.session_state.chatbot_context, 
                            st.session_state.chatbot_messages
                        )
                    st.session_state.chatbot_messages.append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Clear chat button at bottom
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chatbot_messages = []
            st.rerun()
    
    else:
        # Show placeholder when chatbot is not ready
        st.title("💬 Medical Assistant")
        st.info("💤 Chatbot will be available after running health analysis")
        st.markdown("---")
        st.markdown("**Features:**")
        st.markdown("• Answer questions about claims data")
        st.markdown("• Analyze diagnoses and medications") 
        st.markdown("• Heart attack risk analysis (ML + LLM comparison)")
        st.markdown("• Extract specific dates and codes")
        st.markdown("• Provide detailed medical insights")
        st.markdown("• **🎨 Generate interactive visualizations**")
        st.markdown("• **📊 Create charts and graphs on demand**")
