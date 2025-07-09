# FIXED HELPER METHODS - Replace the ones I provided earlier

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
            # Generate a general dashboard
            return self.graph_generator.generate_risk_dashboard(chat_context)
            
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

Please try rephrasing your request with one of these specific graph types.
"""

def _handle_heart_attack_question(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
    """Handle heart attack related questions with potential graph generation"""
    try:
        # Check if they want a graph for heart attack analysis
        graph_request = self.graph_generator.detect_graph_request(user_query)
        
        if graph_request.get("is_graph_request", False):
            return self.graph_generator.generate_risk_dashboard(chat_context)
        
        # Use your existing heart attack handling logic
        # Get FastAPI prediction from context
        heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
        entity_extraction = chat_context.get("entity_extraction", {})
        
        # Prepare comprehensive context for LLM analysis
        complete_context = self._prepare_heart_attack_context(chat_context)
        
        # Build conversation history
        history_text = ""
        if chat_history:
            recent_history = chat_history[-5:]
            history_text = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in recent_history
            ])
        
        # Create enhanced prompt for heart attack analysis
        heart_attack_prompt = f"""You are an expert cardiologist and data analyst with access to COMPLETE deidentified patient claims data. Analyze the heart attack/cardiovascular risk based on the comprehensive medical and pharmacy claims data provided.

COMPLETE DEIDENTIFIED CLAIMS DATA FOR HEART ATTACK ANALYSIS:
{complete_context}

CURRENT FASTAPI ML MODEL PREDICTION:
{json.dumps(heart_attack_prediction, indent=2)}

EXTRACTED HEALTH ENTITIES:
{json.dumps(entity_extraction, indent=2)}

RECENT CONVERSATION HISTORY:
{history_text}

USER QUESTION: {user_query}

CRITICAL ANALYSIS INSTRUCTIONS:
- Provide a comprehensive heart attack/cardiovascular risk assessment in PERCENTAGE format
- Analyze ALL available medical codes (ICD-10), medications (NDC), and claims data
- Consider age, gender, diabetes status, blood pressure, smoking, and medication patterns
- Compare your analysis with the FastAPI ML model prediction provided above
- Provide specific percentages for cardiovascular risk based on the complete claims data
- Reference specific medical codes, medications, dates, and clinical indicators from the JSON data
- Explain discrepancies between your analysis and the FastAPI model if any
- Include risk factors found in the claims data that support your percentage assessment

PROVIDE YOUR RESPONSE IN THIS FORMAT:

**🤖 LLM CARDIOVASCULAR RISK ANALYSIS:**
- **Risk Percentage:** [Your calculated percentage]% 
- **Risk Category:** [Low/Medium/High Risk]
- **Key Risk Factors:** [List specific factors from claims data]
- **Supporting Evidence:** [Specific codes, medications, dates from JSON]

**⚖️ COMPARISON WITH FASTAPI MODEL:**
- **FastAPI Prediction:** [FastAPI percentage and category]
- **LLM Analysis:** [Your percentage and category]  
- **Agreement/Discrepancy:** [Comparison and explanation]
- **Confidence:** [Your confidence in the assessment]

**📊 DETAILED CARDIOVASCULAR ASSESSMENT:**
[Provide detailed analysis based on complete claims data]

Use the complete deidentified claims data to provide the most accurate cardiovascular risk assessment possible."""

        logger.info(f"💬 Processing heart attack question with comprehensive analysis: {user_query[:50]}...")
        
        # Use enhanced API integrator for LLM call
        enhanced_system_msg = """You are an expert cardiologist with access to COMPLETE deidentified claims data. Provide detailed cardiovascular risk analysis with specific percentages based on comprehensive medical and pharmacy claims data. Compare your analysis with ML model predictions and explain your reasoning using specific data from the claims records."""
        
        response = self.api_integrator.call_llm(heart_attack_prompt, enhanced_system_msg)
        
        if response.startswith("Error"):
            return "I encountered an error analyzing cardiovascular risk. Please try rephrasing your question. I can provide detailed heart attack risk analysis using both ML predictions and comprehensive claims data analysis."
        
        # Add visualization suggestion
        response += "\n\n💡 **Tip:** You can also ask me to 'generate a risk assessment dashboard' for visual analysis!"
        
        return response
        
    except Exception as e:
        logger.error(f"Error in heart attack question handling: {str(e)}")
        return "I encountered an error analyzing cardiovascular risk. Please try again. I can compare ML model predictions with comprehensive claims data analysis for heart attack risk assessment."

def _handle_general_question(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
    """Handle general questions with complete claims data access"""
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
        
        # Use enhanced data processor to prepare COMPLETE context with both medical and pharmacy data
        complete_context = self.data_processor.prepare_chunked_context(chat_context)
        
        # Build conversation history for continuity
        history_text = ""
        if chat_history:
            recent_history = chat_history[-10:]
            history_text = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in recent_history
            ])
        
        # Create COMPLETE prompt with ENTIRE deidentified claims data
        complete_prompt = f"""You are an expert medical claims data assistant with access to the COMPLETE, ENTIRE deidentified patient claims records (medical, pharmacy, and MCID). You have the FULL JSON data structures available. Answer the user's question with specific, detailed information from ANY part of the complete claims data provided.

COMPLETE DEIDENTIFIED CLAIMS DATA (ENTIRE JSON STRUCTURES - MEDICAL & PHARMACY):
{complete_context}

RECENT CONVERSATION HISTORY:
{history_text}

USER QUESTION: {user_query}

CRITICAL INSTRUCTIONS:
- You have access to the COMPLETE deidentified medical, pharmacy, and MCID claims JSON data
- Search through the ENTIRE JSON structure to find relevant information
- Include specific dates (clm_rcvd_dt, rx_filled_dt), codes, medications, diagnoses, and values from ANY part of the JSON
- Reference exact field names, values, and nested structures from the data
- If user asks about ANY specific field, code, medication, or data point, find it in the complete JSON
- Include ICD-10 codes, NDC codes, service codes, dates, quantities, and any other specific data
- Access all nested levels of the JSON structure to answer questions
- Be thorough and cite specific data points from the complete deidentified records
- If data exists in the JSON, you can find and reference it
- Use the conversation history to understand follow-up questions and context
- Explain medical codes and terminology when relevant
- For numerical values, dates, codes - provide exact values from the JSON data
- Include both medical claims data AND pharmacy claims data in your analysis

DETAILED ANSWER USING COMPLETE DEIDENTIFIED CLAIMS DATA:"""

        logger.info(f"💬 Processing general query with COMPLETE deidentified claims data access: {user_query[:50]}...")
        logger.info(f"📊 Complete context length: {len(complete_context)} characters")
        
        # Use enhanced API integrator for LLM call with extended system message
        enhanced_system_msg = """You are a powerful healthcare AI assistant with access to COMPLETE deidentified claims records including both medical and pharmacy data. You can search through and reference ANY part of the provided JSON data structures. Provide accurate, detailed analysis based on the ENTIRE medical, pharmacy, and MCID claims data provided. Always maintain patient privacy and provide professional medical insights using the complete available data."""
        
        response = self.api_integrator.call_llm(complete_prompt, enhanced_system_msg)
        
        if response.startswith("Error"):
            return "I encountered an error processing your question. Please try rephrasing your question. I have access to the complete deidentified claims data including both medical and pharmacy records and can answer questions about any specific codes, medications, dates, or other data points."
        
        # Add visualization suggestion
        response += "\n\n📊 **Available Visualizations:** Ask me to 'show medication timeline', 'create diagnosis chart', or 'generate risk dashboard' for visual insights!"
        
        return response
        
    except Exception as e:
        logger.error(f"Error in general question handling: {str(e)}")
        return "I encountered an error processing your question. Please try again. I have access to the complete deidentified claims JSON data and can answer detailed questions about any aspect of the patient's records."

def _prepare_heart_attack_context(self, chat_context: Dict[str, Any]) -> str:
    """Prepare comprehensive context specifically for heart attack analysis"""
    try:
        context_sections = []
        
        # 1. Patient Overview
        patient_overview = chat_context.get("patient_overview", {})
        if patient_overview:
            context_sections.append(f"PATIENT OVERVIEW:\n{json.dumps(patient_overview, indent=2)}")
        
        # 2. Complete Medical Claims Data
        deidentified_medical = chat_context.get("deidentified_medical", {})
        if deidentified_medical:
            medical_claims_data = deidentified_medical.get('medical_claims_data', {})
            if medical_claims_data:
                context_sections.append(f"COMPLETE MEDICAL CLAIMS DATA:\n{json.dumps(medical_claims_data, indent=2)}")
        
        # 3. Complete Pharmacy Claims Data  
        deidentified_pharmacy = chat_context.get("deidentified_pharmacy", {})
        if deidentified_pharmacy:
            pharmacy_claims_data = deidentified_pharmacy.get('pharmacy_claims_data', {})
            if pharmacy_claims_data:
                context_sections.append(f"COMPLETE PHARMACY CLAIMS DATA:\n{json.dumps(pharmacy_claims_data, indent=2)}")
        
        # 4. Medical Extractions with dates
        medical_extraction = chat_context.get("medical_extraction", {})
        if medical_extraction and not medical_extraction.get('error'):
            context_sections.append(f"MEDICAL EXTRACTIONS (including clm_rcvd_dt):\n{json.dumps(medical_extraction, indent=2)}")
        
        # 5. Pharmacy Extractions with dates
        pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
        if pharmacy_extraction and not pharmacy_extraction.get('error'):
            context_sections.append(f"PHARMACY EXTRACTIONS (including rx_filled_dt):\n{json.dumps(pharmacy_extraction, indent=2)}")
        
        # 6. Entity Extraction
        entity_extraction = chat_context.get("entity_extraction", {})
        if entity_extraction:
            context_sections.append(f"HEALTH ENTITIES:\n{json.dumps(entity_extraction, indent=2)}")
        
        # 7. Heart Attack Features
        heart_attack_features = chat_context.get("heart_attack_features", {})
        if heart_attack_features:
            context_sections.append(f"HEART ATTACK FEATURES:\n{json.dumps(heart_attack_features, indent=2)}")
        
        return "\n\n".join(context_sections)
        
    except Exception as e:
        logger.error(f"Error preparing heart attack context: {e}")
        return "Patient claims data available for cardiovascular analysis."
