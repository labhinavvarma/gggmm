# Changes to make in your health_agent_core.py file

# 1. ADD THIS IMPORT at the top of your file (with other imports)
from health_graph_generator import HealthGraphGenerator

# 2. MODIFY your HealthAnalysisAgent.__init__ method
# Add this line after self.data_processor = HealthDataProcessor():
def __init__(self, custom_config: Optional[Config] = None):
    # Use provided config or create default
    self.config = custom_config or Config()
    
    # Initialize enhanced components
    self.api_integrator = HealthAPIIntegrator(self.config)
    self.data_processor = HealthDataProcessor()
    self.graph_generator = HealthGraphGenerator()  # ADD THIS LINE
    
    logger.info("🔧 Enhanced HealthAnalysisAgent initialized with Graph Capabilities")
    logger.info(f"🎨 Graph generation ready for medical data visualizations")
    
    self.setup_enhanced_langgraph()

# 3. ADD these helper methods to your HealthAnalysisAgent class

def _handle_graph_request(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]], graph_request: Dict[str, Any]) -> str:
    """Handle graph generation requests"""
    try:
        graph_type = graph_request.get("graph_type", "timeline")
        
        logger.info(f"📊 Generating {graph_type} visualization for user query: {user_query[:50]}...")
        
        # Generate appropriate graph based on type
        if graph_type == "medication_timeline":
            return self.graph_generator.generate_medication_timeline(chat_context)
        elif graph_type == "diagnosis_timeline":
            return self.graph_generator.generate_diagnosis_timeline(chat_context)
        elif graph_type == "risk_dashboard":
            return self.graph_generator.generate_risk_dashboard(chat_context)
        elif graph_type == "pie":
            return self.graph_generator.generate_medication_distribution(chat_context)
        elif graph_type == "timeline":
            # Default to medication timeline
            return self.graph_generator.generate_medication_timeline(chat_context)
        else:
            # Generate a comprehensive overview
            return self.graph_generator.generate_comprehensive_health_overview(chat_context)
            
    except Exception as e:
        logger.error(f"Error handling graph request: {str(e)}")
        return f"""
## 📊 Graph Generation Error

I encountered an error while generating your requested visualization: {str(e)}

**Available Graph Types:**
- **Medication Timeline**: `show me a medication timeline`
- **Diagnosis Timeline**: `create a diagnosis timeline chart`
- **Risk Dashboard**: `generate a risk assessment dashboard`
- **Medication Distribution**: `show me a pie chart of medications`
- **Health Overview**: `show comprehensive health overview`

Please try rephrasing your request with one of these specific graph types.
"""

# 4. REPLACE your existing chat_with_data method with this enhanced version
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

# 5. MODIFY your existing _handle_heart_attack_question method
# Add this at the end of the method (before the return statement):
def _handle_heart_attack_question(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
    """Handle heart attack related questions with potential graph generation"""
    try:
        # Check if they want a graph for heart attack analysis
        graph_request = self.graph_generator.detect_graph_request(user_query)
        
        if graph_request.get("is_graph_request", False):
            return self.graph_generator.generate_risk_dashboard(chat_context)
        
        # YOUR EXISTING HEART ATTACK CODE HERE...
        # [Keep all your existing heart attack handling logic]
        
        # At the end, before returning the response, add:
        if not response.startswith("Error"):
            response += "\n\n💡 **Tip:** You can also ask me to 'generate a risk assessment dashboard' for visual analysis!"
        
        return response
        
    except Exception as e:
        logger.error(f"Error in heart attack question handling: {str(e)}")
        return "I encountered an error analyzing cardiovascular risk. Please try again. I can provide comprehensive analysis and generate visualizations including risk dashboards and timelines."

# 6. MODIFY your existing _handle_general_question method
# Add this at the end of the method (before the return statement):
def _handle_general_question(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
    """Handle general questions with graph generation suggestions"""
    try:
        # Check if they want a graph for general analysis
        graph_request = self.graph_generator.detect_graph_request(user_query)
        
        if graph_request.get("is_graph_request", False):
            graph_type = graph_request.get("graph_type", "timeline")
            
            if graph_type == "medication_timeline":
                return self.graph_generator.generate_medication_timeline(chat_context)
            elif graph_type == "diagnosis_timeline":
                return self.graph_generator.generate_diagnosis_timeline(chat_context)
            else:
                return self.graph_generator.generate_risk_dashboard(chat_context)
        
        # YOUR EXISTING GENERAL QUESTION CODE HERE...
        # [Keep all your existing general question handling logic]
        
        # At the end, before returning the response, add:
        if not response.startswith("Error"):
            response += "\n\n📊 **Available Visualizations:** Ask me to 'show medication timeline', 'create diagnosis chart', or 'generate risk dashboard' for visual insights!"
        
        return response
        
    except Exception as e:
        logger.error(f"Error in general question handling: {str(e)}")
        return "I encountered an error processing your question. Please try again. I have access to the complete deidentified claims JSON data and can generate various visualizations."
