# Add this method to your HealthAnalysisAgent class

def run_analysis(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run the enhanced health analysis workflow using LangGraph with Claims Data Processing"""
    
    # Initialize enhanced state for LangGraph
    initial_state = HealthAnalysisState(
        patient_data=patient_data,
        mcid_output={},
        medical_output={},
        pharmacy_output={},
        token_output={},
        deidentified_medical={},
        deidentified_pharmacy={},
        deidentified_mcid={},
        medical_extraction={},
        pharmacy_extraction={},
        entity_extraction={},
        health_trajectory="",
        final_summary="",
        heart_attack_prediction={},
        heart_attack_risk_score=0.0,
        heart_attack_features={},
        chatbot_ready=False,
        chatbot_context={},
        chat_history=[],
        current_step="",
        errors=[],
        retry_count=0,
        processing_complete=False,
        step_status={}
    )
    
    try:
        config_dict = {"configurable": {"thread_id": f"enhanced_health_analysis_{datetime.now().timestamp()}"}}
        
        logger.info("🚀 Starting Enhanced Claims Data Processing LangGraph workflow...")
        
        # Execute the workflow without step simulation
        final_state = self.graph.invoke(initial_state, config=config_dict)
        
        # Prepare enhanced results with comprehensive information
        results = {
            "success": final_state["processing_complete"] and not final_state["errors"],
            "patient_data": final_state["patient_data"],
            "api_outputs": {
                "mcid": final_state["mcid_output"],
                "medical": final_state["medical_output"], 
                "pharmacy": final_state["pharmacy_output"],
                "token": final_state["token_output"]
            },
            "deidentified_data": {
                "medical": final_state["deidentified_medical"],
                "pharmacy": final_state["deidentified_pharmacy"],
                "mcid": final_state["deidentified_mcid"]
            },
            "structured_extractions": {
                "medical": final_state["medical_extraction"],
                "pharmacy": final_state["pharmacy_extraction"]
            },
            "entity_extraction": final_state["entity_extraction"],
            "health_trajectory": final_state["health_trajectory"],
            "final_summary": final_state["final_summary"],
            "heart_attack_prediction": final_state["heart_attack_prediction"],
            "heart_attack_risk_score": final_state["heart_attack_risk_score"],
            "heart_attack_features": final_state["heart_attack_features"],
            "chatbot_ready": final_state["chatbot_ready"],
            "chatbot_context": final_state["chatbot_context"],
            "chat_history": final_state["chat_history"],
            "errors": final_state["errors"],
            "processing_steps_completed": self._count_completed_steps(final_state),
            "step_status": final_state["step_status"],
            "langgraph_used": True,
            "mcp_compatible": True,
            "comprehensive_deidentification": True,
            "enhanced_chatbot": True,
            "claims_data_processing": True,
            "enhancement_version": "v6.0_claims_data_processing_with_graphs"
        }
        
        if results["success"]:
            logger.info("✅ Enhanced Claims Data Processing LangGraph analysis completed successfully!")
            logger.info(f"🔒 Comprehensive claims deidentification: {results['comprehensive_deidentification']}")
            logger.info(f"💬 Enhanced chatbot ready: {results['chatbot_ready']}")
            logger.info(f"🎨 Graph capabilities enabled: True")
        else:
            logger.error(f"❌ Enhanced LangGraph analysis failed with errors: {final_state['errors']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Fatal error in Enhanced Claims Data Processing LangGraph workflow: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "patient_data": patient_data,
            "errors": [str(e)],
            "processing_steps_completed": 0,
            "langgraph_used": True,
            "mcp_compatible": True,
            "comprehensive_deidentification": False,
            "enhanced_chatbot": False,
            "claims_data_processing": False,
            "enhancement_version": "v6.0_claims_data_processing_with_graphs"
        }

def _count_completed_steps(self, state: HealthAnalysisState) -> int:
    """Count enhanced processing steps completed"""
    steps = 0
    if state.get("mcid_output"): steps += 1
    if state.get("deidentified_medical") and not state.get("deidentified_medical", {}).get("error"): steps += 1
    if state.get("medical_extraction") or state.get("pharmacy_extraction"): steps += 1
    if state.get("entity_extraction"): steps += 1
    if state.get("health_trajectory"): steps += 1
    if state.get("final_summary"): steps += 1
    if state.get("heart_attack_prediction"): steps += 1
    if state.get("chatbot_ready"): steps += 1
    return steps
