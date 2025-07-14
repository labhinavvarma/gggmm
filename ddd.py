# Add this CSS to your existing st.markdown CSS section

st.markdown("""
<style>
/* ... existing CSS ... */

/* Prompt Button Styling */
.stButton > button[key^="prompt_"], 
.stButton > button[key^="cat_prompt_"],
.stButton > button[key^="quick_"] {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
    border: 1px solid #dee2e6 !important;
    color: #495057 !important;
    font-weight: 500 !important;
    padding: 0.5rem 0.75rem !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
    line-height: 1.2 !important;
    transition: all 0.2s ease !important;
    margin-bottom: 0.5rem !important;
    text-align: left !important;
}

.stButton > button[key^="prompt_"]:hover,
.stButton > button[key^="cat_prompt_"]:hover,
.stButton > button[key^="quick_"]:hover {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%) !important;
    border-color: #2196f3 !important;
    color: #1565c0 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 8px rgba(33, 150, 243, 0.2) !important;
}

/* Quick Access Buttons */
.stButton > button[key^="quick_"] {
    background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%) !important;
    border-color: #4caf50 !important;
    color: #2e7d32 !important;
    font-weight: 600 !important;
}

.stButton > button[key^="quick_"]:hover {
    background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%) !important;
    color: white !important;
}

/* Sidebar Expander Styling */
.streamlit-expanderHeader {
    background-color: rgba(255, 255, 255, 0.1) !important;
    border-radius: 4px !important;
    margin-bottom: 0.5rem !important;
}

/* Chat Message Styling */
.stChatMessage {
    margin: 0.5rem 0 !important;
    padding: 0.75rem !important;
    border-radius: 8px !important;
}

/* Clear Chat Button */
.stButton > button:contains("Clear Chat") {
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%) !important;
    border-color: #f44336 !important;
    color: #c62828 !important;
}

.stButton > button:contains("Clear Chat"):hover {
    background: linear-gradient(135deg, #f44336 0%, #e53935 100%) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)
