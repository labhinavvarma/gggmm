def _extract_enhanced_heart_attack_features(self, state):
        """Enhanced feature extraction for heart attack prediction"""
        try:
            logger.info("🔍 Starting feature extraction for heart attack prediction...")
            
            extracted_features = {}
            
            # Extract patient age
            deidentified_medical = state.get("deidentified_medical", {})
            patient_age = deidentified_medical.get("src_mbr_age", None)
            
            if patient_age and patient_age != "unknown":
                try:
                    age_value = int(float(str(patient_age)))
                    if 0 <= age_value <= 120:
                        extracted_features["Age"] = age_value
                        logger.info(f"✅ Age extracted: {age_value}")
                    else:
                        extracted_features["Age"] = 50  # Default age
                        logger.warning(f"⚠️ Age {age_value} out of range, using default 50")
                except:
                    extracted_features["Age"] = 50
                    logger.warning("⚠️ Could not parse age, using default 50")
            else:
                extracted_features["Age"] = 50
                logger.warning("⚠️ No age found, using default 50")
            
            # Extract gender
            patient_data = state.get("patient_data", {})
            gender = str(patient_data.get("gender", "F")).upper()
            extracted_features["Gender"] = 1 if gender in ["M", "MALE", "1"] else 0
            logger.info(f"✅ Gender extracted: {gender} -> {extracted_features['Gender']}")
            
            # Extract health conditions from entity extraction
            entity_extraction = state.get("entity_extraction", {})
            logger.info(f"📊 Entity extraction data: {entity_extraction}")
            
            # Extract diabetes
            diabetes = str(entity_extraction.get("diabetics", "no")).lower()
            extracted_features["Diabetes"] = 1 if diabetes in ["yes", "true", "1"] else 0
            logger.info(f"✅ Diabetes extracted: {diabetes} -> {extracted_features['Diabetes']}")
            
            # Extract blood pressure
            blood_pressure = str(entity_extraction.get("blood_pressure", "unknown")).lower()
            extracted_features["High_BP"] = 1 if blood_pressure in ["managed", "diagnosed", "yes", "true", "1"] else 0
            logger.info(f"✅ Blood pressure extracted: {blood_pressure} -> {extracted_features['High_BP']}")
            
            # Extract smoking
            smoking = str(entity_extraction.get("smoking", "no")).lower()
            extracted_features["Smoking"] = 1 if smoking in ["yes", "true", "1"] else 0
            logger.info(f"✅ Smoking extracted: {smoking} -> {extracted_features['Smoking']}")
            
            # Validate all features are integers
            for key in extracted_features:
                try:
                    extracted_features[key] = int(extracted_features[key])
                except:
                    if key == "Age":
                        extracted_features[key] = 50
                    else:
                        extracted_features[key] = 0
                    logger.warning(f"⚠️ Could not convert {key} to integer, using default")
            
            # Create feature summary
            feature_summary = {
                "extracted_features": extracted_features,
                "feature_interpretation": {
                    "Age": f"{extracted_features['Age']} years old",
                    "Gender": "Male" if extracted_features["Gender"] == 1 else "Female",
                    "Diabetes": "Yes" if extracted_features["Diabetes"] == 1 else "No",
                    "High_BP": "Yes" if extracted_features["High_BP"] == 1 else "No",
                    "Smoking": "Yes" if extracted_features["Smoking"] == 1 else "No"
                },
                "extraction_success": True
            }
            
            logger.info(f"✅ Feature extraction completed successfully")
            logger.info(f"📊 Features: {feature_summary['feature_interpretation']}")
            
            return feature_summary
            
        except Exception as e:
            logger.error(f"❌ Error in feature extraction: {e}")
            return {
                "error": f"Feature extraction failed: {str(e)}",
                "extraction_success": False
            }

    def _prepare_enhanced_fastapi_features(self, feature_summary):
        """Prepare enhanced feature data for FastAPI server call"""
        try:
            logger.info(f"⚙️ Preparing features for FastAPI call...")
            logger.info(f"📊 Input feature_summary: {feature_summary}")
            
            # Extract the features from the feature extraction result
            if "extracted_features" not in feature_summary:
                logger.error("❌ No extracted_features found in feature_summary")
                return None
                
            extracted_features = feature_summary["extracted_features"]
            logger.info(f"📋 Extracted features: {extracted_features}")
            
            # Convert to FastAPI format with proper parameter names
            fastapi_features = {
                "age": int(extracted_features.get("Age", 50)),
                "gender": int(extracted_features.get("Gender", 0)),
                "diabetes": int(extracted_features.get("Diabetes", 0)),
                "high_bp": int(extracted_features.get("High_BP", 0)),
                "smoking": int(extracted_features.get("Smoking", 0))
            }
            
            # Validate age range
            if fastapi_features["age"] < 0 or fastapi_features["age"] > 120:
                logger.warning(f"Age {fastapi_features['age']} out of range, using default 50")
                fastapi_features["age"] = 50
            
            # Validate binary features (0 or 1 only)
            binary_features = ["gender", "diabetes", "high_bp", "smoking"]
            for key in binary_features:
                if fastapi_features[key] not in [0, 1]:
                    logger.warning(f"{key} value {fastapi_features[key]} invalid, using 0")
                    fastapi_features[key] = 0
            
            logger.info(f"✅ FastAPI features prepared: {fastapi_features}")
            return fastapi_features
            
        except Exception as e:
            logger.error(f"❌ Error preparing FastAPI features: {e}")
            return None

    def _call_heart_attack_prediction_sync(self, fastapi_features):
        """Synchronous heart attack prediction call to avoid event loop conflicts"""
        try:
            import requests
            
            # Log the features we received
            logger.info(f"🔍 Received fastapi_features for prediction: {fastapi_features}")
            
            # Validate that we have features
            if not fastapi_features:
                logger.error("❌ No fastapi_features provided for prediction")
                return {
                    "success": False,
                    "error": "No features provided for heart attack prediction"
                }
            
            # Get the heart attack API URL from config
            heart_attack_url = self.config.heart_attack_api_url
            logger.info(f"🌐 Using heart attack API URL: {heart_attack_url}")
            
            # Try multiple endpoint formats for compatibility
            endpoints = [
                f"{heart_attack_url}/predict",
                f"{heart_attack_url}/predict-simple"
            ]
            
            # Use the fastapi_features directly (they should already be integers)
            params = {
                "age": int(fastapi_features.get("age", 50)),
                "gender": int(fastapi_features.get("gender", 0)),
                "diabetes": int(fastapi_features.get("diabetes", 0)),
                "high_bp": int(fastapi_features.get("high_bp", 0)),
                "smoking": int(fastapi_features.get("smoking", 0))
            }
            
            logger.info(f"📤 Sending prediction request to {endpoints[0]}")
            logger.info(f"📊 Parameters: {params}")
            
            # Try POST with JSON body first
            try:
                response = requests.post(endpoints[0], json=params, timeout=30)
                logger.info(f"📡 Response status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ Prediction successful: {result}")
                    return {
                        "success": True,
                        "prediction_data": result,
                        "method": "POST_JSON_SYNC",
                        "endpoint": endpoints[0]
                    }
                else:
                    logger.warning(f"❌ First endpoint failed with status {response.status_code}")
                    logger.warning(f"Response: {response.text}")
            
            except requests.exceptions.ConnectionError as conn_error:
                logger.error(f"❌ Connection failed to {endpoints[0]}: {conn_error}")
                return {
                    "success": False,
                    "error": f"Cannot connect to heart attack prediction server at {endpoints[0]}. Make sure the server is running."
                }
            except requests.exceptions.Timeout as timeout_error:
                logger.error(f"❌ Timeout connecting to {endpoints[0]}: {timeout_error}")
                return {
                    "success": False,
                    "error": f"Timeout connecting to heart attack prediction server"
                }
            except Exception as request_error:
                logger.warning(f"❌ JSON method failed: {str(request_error)}")
            
            # Try POST with query parameters as fallback
            try:
                logger.info(f"🔄 Trying fallback endpoint: {endpoints[1]}")
                response = requests.post(endpoints[1], params=params, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ Fallback prediction successful: {result}")
                    return {
                        "success": True,
                        "prediction_data": result,
                        "method": "POST_PARAMS_SYNC",
                        "endpoint": endpoints[1]
                    }
                else:
                    error_text = response.text
                    logger.error(f"❌ All endpoints failed. Status {response.status_code}: {error_text}")
                    return {
                        "success": False,
                        "error": f"Heart attack prediction server error {response.status_code}: {error_text}",
                        "tried_endpoints": endpoints
                    }
            
            except Exception as fallback_error:
                logger.error(f"❌ All prediction methods failed: {str(fallback_error)}")
                return {
                    "success": False,
                    "error": f"All prediction methods failed. Error: {str(fallback_error)}",
                    "tried_endpoints": endpoints
                }
                
        except ImportError:
            logger.error("❌ requests library not found")
            return {
                "success": False,
                "error": "requests library not installed. Run: pip install requests"
            }
        except Exception as general_error:
            logger.error(f"❌ Unexpected error in heart attack prediction: {general_error}")
            return {
                "success": False,
                "error": f"Heart attack prediction failed: {str(general_error)}"
            }

    def predict_heart_attack(self, state):
        """Enhanced LangGraph Node 7: Enhanced heart attack prediction with FastAPI compatibility"""
        logger.info("❤️ Enhanced Node 7: Starting enhanced heart attack prediction...")
        state["current_step"] = "predict_heart_attack"
        state["step_status"]["predict_heart_attack"] = "running"
        
        try:
            # Step 1: Extract features using enhanced feature extraction
            logger.info("🔍 Step 1: Extracting heart attack features...")
            feature_summary = self._extract_enhanced_heart_attack_features(state)
            state["heart_attack_features"] = feature_summary
            
            if not feature_summary or "error" in feature_summary:
                error_msg = "Failed to extract enhanced features for heart attack prediction"
                state["errors"].append(error_msg)
                state["step_status"]["predict_heart_attack"] = "error"
                logger.error(error_msg)
                return state
            
            # Step 2: Prepare feature vector for enhanced FastAPI call
            logger.info("⚙️ Step 2: Preparing features for FastAPI call...")
            fastapi_features = self._prepare_enhanced_fastapi_features(feature_summary)
            
            if fastapi_features is None:
                error_msg = "Failed to prepare enhanced feature vector for prediction"
                state["errors"].append(error_msg)
                state["step_status"]["predict_heart_attack"] = "error"
                logger.error(error_msg)
                return state
            
            # Step 3: Make prediction using synchronous method
            logger.info("🚀 Step 3: Making heart attack prediction call...")
            prediction_result = self._call_heart_attack_prediction_sync(fastapi_features)
            
            if prediction_result is None:
                error_msg = "Heart attack prediction returned None"
                state["errors"].append(error_msg)
                state["step_status"]["predict_heart_attack"] = "error"
                logger.error(error_msg)
                return state
            
            # Step 4: Process prediction result
            if prediction_result.get("success", False):
                logger.info("✅ Step 4: Processing successful prediction result...")
                
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
                    "enhanced_features_used": feature_summary.get("feature_interpretation", {}),
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
