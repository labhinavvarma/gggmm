def predict_heart_attack(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 7: Enhanced heart attack prediction with FastAPI compatibility"""
        logger.info("❤️ Enhanced Node 7: Starting enhanced heart attack prediction...")
        state["current_step"] = "predict_heart_attack"
        state["step_status"]["predict_heart_attack"] = "running"
        
        try:
            # Extract features using enhanced feature extraction
            features = self._extract_enhanced_heart_attack_features(state)
            state["heart_attack_features"] = features
            
            if not features or "error" in features:
                state["errors"].append("Failed to extract enhanced features for heart attack prediction")
                state["step_status"]["predict_heart_attack"] = "error"
                return state
            
            # Prepare feature vector for enhanced FastAPI call
            fastapi_features = self._prepare_enhanced_fastapi_features(features)
            
            if fastapi_features is None:
                state["errors"].append("Failed to prepare enhanced feature vector for prediction")
                state["step_status"]["predict_heart_attack"] = "error"
                return state
            
            # Make prediction using synchronous method (avoid event loop conflict)
            try:
                prediction_result = self._call_heart_attack_prediction_sync(fastapi_features)
            except Exception as sync_error:
                logger.error(f"Enhanced prediction call failed: {sync_error}")
                state["errors"].append(f"Enhanced FastAPI prediction call failed: {str(sync_error)}")
                state["step_status"]["predict_heart_attack"] = "error"
                return state
            
            if prediction_result.get("success", False):
                # Process successful enhanced FastAPI prediction
                prediction_data = prediction_result.get("prediction_data", {})
                
                # Extract key values from enhanced FastAPI response
                risk_probability = prediction_data.get("probability", 0.0)
                binary_prediction = prediction_data.get("prediction", 0)
                
                # Convert to percentage
                risk_percentage = risk_probability * 100
                confidence_percentage = (1 - risk_probability) * 100 if binary_prediction == 0 else risk_probability * 100
                
                # Determine enhanced risk level
                if risk_percentage >= 70:
                    risk_category = "High Risk"
                elif risk_percentage >= 50:
                    risk_category = "Medium Risk"
                else:
                    risk_category = "Low Risk"
                
                # Create enhanced prediction result
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
                # Handle enhanced FastAPI prediction failure
                error_msg = prediction_result.get("error", "Unknown enhanced FastAPI error")
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
                logger.warning(f"⚠️ Enhanced FastAPI heart attack prediction failed: {error_msg}")
            
            state["step_status"]["predict_heart_attack"] = "completed"
            
        except Exception as e:
            error_msg = f"Error in enhanced FastAPI heart attack prediction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["predict_heart_attack"] = "error"
            logger.error(error_msg)
        
        return state
