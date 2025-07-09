def chat_with_data(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
    """Enhanced chatbot with COMPLETE deidentified claims data access, heart attack analysis, and graph generation capabilities"""
    try:
        # FIRST: Check if this is a graph request
        graph_request = self.graph_generator.detect_graph_request(user_query)
        
        if graph_request.get("is_graph_request", False):
            return self._handle_graph_request(user_query, chat_context, chat_history, graph_request)
        
        # SECOND: Check if this is a heart attack related question
        heart_attack_keywords = ['heart attack', 'heart disease', 'cardiac', 'cardiovascular', 'heart risk', 'coronary', 'myocardial', 'cardiac risk']
        is_heart_attack_question = any(keyword in user_query.lower() for keyword in heart_attack_keywords)
        
        if is_heart_attack_question:
            return self._handle_heart_attack_question(user_query, chat_context, chat_history)
        else:
            return self._handle_general_question(user_query, chat_context, chat_history)
        
    except Exception as e:
        logger.error(f"Error in enhanced chatbot with complete deidentified claims data and graph capabilities: {str(e)}")
        return "I encountered an error processing your question. Please try again. I have access to the complete deidentified claims JSON data and can generate visualizations for comprehensive analysis of any aspect of the patient's records."
