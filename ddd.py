def predict_heart_attack(self, state: HealthAnalysisState) -> HealthAnalysisState:
    """Enhanced LangGraph Node 7: Enhanced heart attack prediction with FastAPI compatibility"""
    logger.info("❤️ Enhanced Node 7: Starting enhanced heart attack prediction...")
    state["current_step"] = "predict_heart_attack"
    state["step_status"]["predict_heart_attack"] = "running"
    
    try:
        # Step 1: Extract features using enhanced feature extraction
        logger.info("🔍 Extracting heart attack features...")
        features = self._extract_enhanced_heart_attack_features(state)
        state["heart_attack_features"] = features
        
        if not features or "error" in features:
            error_msg = "Failed to extract enhanced features for heart attack prediction"
            state["errors"].append(error_msg)
            state["step_status"]["predict_heart_attack"] = "error"
            logger.error(error_msg)
            return state
        
        # Step 2: Prepare feature vector for enhanced FastAPI call
        logger.info("⚙️ Preparing features for FastAPI call...")
        fastapi_features = self._prepare_enhanced_fastapi_features(features)
        
        if fastapi_features is None:
            error_msg = "Failed to prepare enhanced feature vector for prediction"
            state["errors"].append(error_msg)
            state["step_status"]["predict_heart_attack"] = "error"
            logger.error(error_msg)
            return state
        
        # Step 3: Make prediction using synchronous method
        logger.info("🚀 Making heart attack prediction call...")
        prediction_result = self._call_heart_attack_prediction_sync(fastapi_features)
        
        if prediction_result is None:
            error_msg = "Heart attack prediction returned None"
            state["errors"].append(error_msg)
            state["step_status"]["predict_heart_attack"] = "error"
            logger.error(error_msg)
            return state
        
        # Step 4: Process prediction result
        if prediction_result.get("success", False):
            logger.info("✅ Processing successful prediction result...")
            
            # Extract prediction data
            prediction_data = prediction_result.get("prediction_data", {})
            
            # Get risk probability and prediction
            risk_probability = prediction_data.get("probability", 0.0)
            binary_prediction = prediction_data.get("prediction", 0)
            
            # Convert to percentage
            risk_percentage = risk_probability * 100
            confidence_percentage = (1 - risk_probability) * 100 if binary_prediction == 0 else risk_probability * 100
            
            # Determine risk level
            if risk_percentage >= 70:
                risk_category = "High Risk"
            elif risk_percentage >= 50:
                risk_category = "Medium Risk"
            else:
                risk_category = "Low Risk"
            
            # Create prediction result
            enhanced_prediction = {
                "risk_display": f"Heart Disease Risk: {risk_percentage:.1f}% ({risk_category})",
                "confidence_display": f"Confidence: {confidence_percentage:.1f}%",
                "combined_display": f"Heart Disease Risk: {risk_percentage:.1f}% ({risk_category}) | Confidence: {confidence_percentage:.1f}%",
                "raw_risk_score": risk_probability,
                "raw_prediction": binary_prediction,
                "risk_category": risk_category,
                "fastapi_server_url": self.config.heart_attack_api_url,
                "prediction_method": prediction_result.get("method", "unknown"),
                "prediction_endpoint": prediction_result.get("endpoint", "unknown"),
                "prediction_timestamp": datetime.now().isoformat(),
                "enhanced_features_used": features.get("feature_interpretation", {}),
                "model_enhanced": True
            }
            
            state["heart_attack_prediction"] = enhanced_prediction
            state["heart_attack_risk_score"] = float(risk_probability)
            
            logger.info(f"✅ Enhanced FastAPI heart attack prediction completed successfully")
            logger.info(f"❤️ Display: {enhanced_prediction['combined_display']}")
            
        else:
            # Handle prediction failure
            error_msg = prediction_result.get("error", "Unknown FastAPI error")
            logger.warning(f"⚠️ Enhanced FastAPI heart attack prediction failed: {error_msg}")
            
            state["heart_attack_prediction"] = {
                "error": error_msg,
                "risk_display": "Heart Disease Risk: Error",
                "confidence_display": "Confidence: Error",
                "combined_display": f"Heart Disease Risk: Error - {error_msg}",
                "fastapi_server_url": self.config.heart_attack_api_url,
                "error_details": error_msg,
                "tried_endpoints": prediction_result.get("tried_endpoints", []),
                "model_enhanced": True
            }
            state["heart_attack_risk_score"] = 0.0
        
        state["step_status"]["predict_heart_attack"] = "completed"
        
    except Exception as e:
        error_msg = f"Error in enhanced FastAPI heart attack prediction: {str(e)}"
        state["errors"].append(error_msg)
        state["step_status"]["predict_heart_attack"] = "error"
        logger.error(error_msg)
    
    return state
