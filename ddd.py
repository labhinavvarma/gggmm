def chat_with_data(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
    """Enhanced chatbot with graph generation capabilities"""
    try:
        # Check if this is a graph request FIRST
        graph_request = self.graph_generator.detect_graph_request(user_query)
        
        if graph_request.get("is_graph_request", False):
            return self._handle_graph_request(user_query, chat_context, chat_history, graph_request)
        
        # Check if this is a heart attack related question
        heart_attack_keywords = ['heart attack', 'heart disease', 'cardiac', 'cardiovascular', 'heart risk', 'coronary', 'myocardial', 'cardiac risk']
        is_heart_attack_question = any(keyword in user_query.lower() for keyword in heart_attack_keywords)
        
        if is_heart_attack_question:
            return self._handle_heart_attack_question(user_query, chat_context, chat_history)
        else:
            return self._handle_general_question(user_query, chat_context, chat_history)
        
    except Exception as e:
        logger.error(f"Error in enhanced chatbot with graph capabilities: {str(e)}")
        return "I encountered an error processing your question. Please try again. I have access to comprehensive claims data and can generate visualizations for your analysis."
