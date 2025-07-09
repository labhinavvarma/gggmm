# REPLACE WITH THIS ENHANCED VERSION:
with chat_container:
    if st.session_state.chatbot_messages:
        for message in st.session_state.chatbot_messages:
            with st.chat_message(message["role"]):
                content = message["content"]
                
                # Check if content contains HTML (graphs)
                if "<html>" in content and "</html>" in content:
                    # Display as HTML component for graphs
                    st.components.v1.html(content, height=800, scrolling=True)
                elif "##" in content and ("<div>" in content or "fig.to_html" in content):
                    # Alternative: Use markdown with HTML enabled
                    st.markdown(content, unsafe_allow_html=True)
                else:
                    # Regular text content
                    st.write(content)
