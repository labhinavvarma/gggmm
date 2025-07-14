
# ENHANCED SIDEBAR CHATBOT WITH CATEGORIZED PROMPTS
with st.sidebar:
    if st.session_state.analysis_results and st.session_state.analysis_results.get("chatbot_ready", False) and st.session_state.chatbot_context:
        st.title("💬 Medical Assistant")
        st.markdown("---")
        
        # Chat history at top
        chat_container = st.container()
        with chat_container:
            if st.session_state.chatbot_messages:
                for message in st.session_state.chatbot_messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
            else:
                st.info("👋 Hello! I can answer questions about the medical analysis. Ask me anything!")
        
        # CATEGORIZED SUGGESTED PROMPTS SECTION
        st.markdown("---")
        st.markdown("**💡 Quick Questions:**")
        
        # Define categorized prompts
        prompt_categories = {
            "🏥 Medical Records": [
                "What diagnoses were found in the medical records?",
                "What medical procedures were performed?",
                "List all ICD-10 diagnosis codes found",
                "Show me the most recent medical claims"
            ],
            "💊 Medications": [
                "What medications is this patient taking?",
                "What NDC codes were identified?",
                "Are there any diabetes medications?",
                "What blood pressure medications are prescribed?"
            ],
            "❤️ Risk Assessment": [
                "What is the heart attack risk and why?",
                "What are the main risk factors?",
                "Compare ML prediction vs clinical assessment",
                "What chronic conditions does this patient have?"
            ],
            "📊 Analysis": [
                "Summarize the patient's health trajectory",
                "How has the patient's health changed over time?",
                "What patterns do you see in the claims data?",
                "Provide a comprehensive health overview"
            ]
        }
        
        # Handle selected prompt from session state
        if hasattr(st.session_state, 'selected_prompt') and st.session_state.selected_prompt:
            user_question = st.session_state.selected_prompt
            
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
                
                # Clear the selected prompt
                st.session_state.selected_prompt = None
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.selected_prompt = None
        
        # Create expandable sections for each category
        for category, prompts in prompt_categories.items():
            with st.expander(category, expanded=False):
                for i, prompt in enumerate(prompts):
                    if st.button(prompt, key=f"cat_prompt_{category}_{i}", use_container_width=True):
                        st.session_state.selected_prompt = prompt
                        st.rerun()
        
        # Quick access buttons for most common questions
        st.markdown("**🚀 Quick Access:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Summary", use_container_width=True, key="quick_summary"):
                st.session_state.selected_prompt = "Provide a comprehensive summary of this patient's health status"
                st.rerun()
        
        with col2:
            if st.button("❤️ Heart Risk", use_container_width=True, key="quick_heart"):
                st.session_state.selected_prompt = "What is this patient's heart attack risk and explain the reasoning?"
                st.rerun()
        
        # Chat input at bottom
        st.markdown("---")
        user_question = st.chat_input("Type your question or use prompts above...")
        
        # Handle manual chat input
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
        
        # Clear chat button at bottom
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chatbot_messages = []
            st.rerun()
    
    else:
        # Enhanced placeholder when chatbot is not ready
        st.title("💬 Medical Assistant")
        st.info("💤 Chatbot will be available after running health analysis")
        st.markdown("---")
        st.markdown("**🎯 What you can ask:**")
        st.markdown("• **Medical Records:** Diagnoses, procedures, ICD codes")
        st.markdown("• **Medications:** Prescriptions, NDC codes, drug interactions") 
        st.markdown("• **Risk Assessment:** Heart attack risk, chronic conditions")
        st.markdown("• **Analysis:** Health trends, comprehensive summaries")
        st.markdown("---")
        st.markdown("**💡 Features:**")
        st.markdown("• Click suggested prompts for instant questions")
        st.markdown("• Categorized quick access buttons")
        st.markdown("• Full conversational AI capabilities")
