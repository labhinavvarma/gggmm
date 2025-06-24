async def test_fastapi_connection(self) -> Dict[str, Any]:
        """Test the FastAPI server connection - FIXED FOR QUERY PARAMETERS"""
        try:
            logger.info(f"🧪 Testing FastAPI server connection at {self.config.heart_attack_api_url}...")
            
            # Test health endpoint first
            health_url = f"{self.config.heart_attack_api_url}/health"
            
            timeout = aiohttp.ClientTimeout(total=10)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_url) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        
                        # Test prediction endpoint with sample data using query parameters - FIXED
                        test_params = {
                            "age": 50,
                            "gender": 1,
                            "diabetes": 0,
                            "high_bp": 0,
                            "smoking": 0
                        }
                        
                        predict_url = f"{self.config.heart_attack_api_url}/predict"
                        # FIXED: Use query parameters instead of JSON
                        async with session.post(predict_url, params=test_params) as pred_response:
                            if pred_response.status == 200:
                                pred_data = await pred_response.json()
                                return {
                                    "success": True,
                                    "health_check": health_data,
                                    "prediction_test": pred_data,
                                    "server_url": self.config.heart_attack_api_url,
                                    "test_params": test_params
                                }
                            else:
                                error_text = await pred_response.text()
                                return {
                                    "success": False,
                                    "error": f"Prediction endpoint error {pred_response.status}: {error_text}",
                                    "server_url": self.config.heart_attack_api_url,
                                    "test_params": test_params
                                }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"Health endpoint error {response.status}: {error_text}",
                            "server_url": self.config.heart_attack_api_url
                        }
                        
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "FastAPI server timeout",
                "server_url": self.config.heart_attack_api_url
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"FastAPI connection test failed: {str(e)}",
                "server_url": self.config.heart_attack_api_url
            }

    async def _call_fastapi_heart_attack_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Call FastAPI server for heart attack prediction - FIXED FOR QUERY PARAMETERS"""
        try:
            logger.info(f"🔗 Calling FastAPI server for heart attack prediction...")
            logger.info(f"📊 Features: {features}")
            
            # Prepare the request to FastAPI server
            predict_url = f"{self.config.heart_attack_api_url}/predict"
            
            # FIXED: FastAPI expects query parameters, not JSON body
            # Make sure all values are integers as required by the server
            params = {
                "age": int(features.get("age", 50)),
                "gender": int(features.get("gender", 0)),
                "diabetes": int(features.get("diabetes", 0)),
                "high_bp": int(features.get("high_bp", 0)),
                "smoking": int(features.get("smoking", 0))
            }
            
            logger.info(f"📤 Sending query params: {params}")
            
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # FIXED: Use POST with query parameters (not JSON body)
                async with session.post(predict_url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ FastAPI prediction successful: {result}")
                        return {
                            "success": True,
                            "prediction_data": result
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ FastAPI server error {response.status}: {error_text}")
                        return {
                            "success": False,
                            "error": f"FastAPI server error {response.status}: {error_text}"
                        }
                        
        except asyncio.TimeoutError:
            logger.error("❌ FastAPI server timeout")
            return {
                "success": False,
                "error": "FastAPI server timeout"
            }
        except Exception as e:
            logger.error(f"Error calling FastAPI server: {e}")
            return {
                "success": False,
                "error": f"FastAPI call failed: {str(e)}"
            }

    def _prepare_fastapi_features(self, features: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """Prepare feature data for FastAPI server call - FIXED TO ENSURE INTEGERS"""
        try:
            extracted_features = features.get("extracted_features", {})
            
            # Prepare features for FastAPI server: age, gender, diabetes, high_bp, smoking
            # FIXED: Ensure all values are integers
            fastapi_features = {
                "age": int(extracted_features.get("Age", 50)),
                "gender": int(extracted_features.get("Gender", 0)),
                "diabetes": int(extracted_features.get("Diabetes", 0)),
                "high_bp": int(extracted_features.get("High_BP", 0)),
                "smoking": int(extracted_features.get("Smoking", 0))
            }
            
            # Validate ranges
            if fastapi_features["age"] < 0 or fastapi_features["age"] > 120:
                fastapi_features["age"] = 50  # Default safe age
            
            for key in ["gender", "diabetes", "high_bp", "smoking"]:
                if fastapi_features[key] not in [0, 1]:
                    fastapi_features[key] = 0  # Default to 0 for binary features
            
            logger.info(f"✅ FastAPI features prepared: {fastapi_features}")
            return fastapi_features
            
        except Exception as e:
            logger.error(f"Error preparing FastAPI features: {e}")
            return None

    def _extract_heart_attack_features_for_fastapi(self, state: HealthAnalysisState) -> Dict[str, Any]:
        """Extract features specifically for FastAPI model: Age, Gender, Diabetes, High_BP, Smoking - FIXED"""
        try:
            logger.info("🔍 Extracting features for FastAPI heart attack prediction model...")
            
            features = {}
            
            # Get patient age from deidentified medical data
            deidentified_medical = state.get("deidentified_medical", {})
            patient_age = deidentified_medical.get("src_mbr_age", None)
            
            if patient_age and patient_age != "unknown":
                try:
                    # FIXED: Ensure integer conversion
                    age_value = int(float(str(patient_age)))  # Handle various formats
                    if 0 <= age_value <= 120:
                        features["Age"] = age_value
                    else:
                        features["Age"] = 50  # Default age if out of range
                except:
                    features["Age"] = 50  # Default age if conversion fails
            else:
                features["Age"] = 50  # Default age
            
            # Get gender from patient data - convert to 0/1 for model
            patient_data = state.get("patient_data", {})
            gender = str(patient_data.get("gender", "F")).upper()
            features["Gender"] = 1 if gender in ["M", "MALE", "1"] else 0  # 1 for Male, 0 for Female
            
            # Extract features from entity extraction
            entity_extraction = state.get("entity_extraction", {})
            
            # Diabetes indicator - FIXED: Ensure integer
            diabetes = str(entity_extraction.get("diabetics", "no")).lower()
            features["Diabetes"] = 1 if diabetes in ["yes", "true", "1"] else 0
            
            # High Blood Pressure indicator - FIXED: Ensure integer
            blood_pressure = str(entity_extraction.get("blood_pressure", "unknown")).lower()
            features["High_BP"] = 1 if blood_pressure in ["managed", "diagnosed", "yes", "true", "1"] else 0
            
            # Smoking indicator - FIXED: Ensure integer
            smoking = str(entity_extraction.get("smoking", "no")).lower()
            features["Smoking"] = 1 if smoking in ["yes", "true", "1"] else 0
            
            # Enhance feature extraction from medical codes
            medical_extraction = state.get("medical_extraction", {})
            hlth_srvc_records = medical_extraction.get("hlth_srvc_records", [])
            
            # Check for diabetes-related ICD-10 codes
            diabetes_codes = ["E10", "E11", "E12", "E13", "E14"]
            hypertension_codes = ["I10", "I11", "I12", "I13", "I15"]
            smoking_codes = ["Z72.0", "F17"]
            
            for record in hlth_srvc_records:
                diagnosis_codes = record.get("diagnosis_codes", [])
                for diag in diagnosis_codes:
                    code = str(diag.get("code", ""))
                    if code:
                        # Check for diabetes
                        if any(code.startswith(d_code) for d_code in diabetes_codes):
                            features["Diabetes"] = 1
                        # Check for hypertension
                        if any(code.startswith(h_code) for h_code in hypertension_codes):
                            features["High_BP"] = 1
                        # Check for smoking
                        if any(code.startswith(s_code) for s_code in smoking_codes):
                            features["Smoking"] = 1
            
            # Enhance from pharmacy data
            pharmacy_extraction = state.get("pharmacy_extraction", {})
            ndc_records = pharmacy_extraction.get("ndc_records", [])
            
            for record in ndc_records:
                lbl_nm = str(record.get("lbl_nm", "")).lower()
                if lbl_nm:
                    # Check for diabetes medications
                    diabetes_meds = ["insulin", "metformin", "glipizide", "glucophage", "lantus", "humalog"]
                    if any(med in lbl_nm for med in diabetes_meds):
                        features["Diabetes"] = 1
                    
                    # Check for blood pressure medications
                    bp_meds = ["lisinopril", "amlodipine", "metoprolol", "losartan", "hydrochlorothiazide"]
                    if any(med in lbl_nm for med in bp_meds):
                        features["High_BP"] = 1
            
            # FIXED: Final validation - ensure all values are integers
            for key in features:
                try:
                    features[key] = int(features[key])
                except:
                    if key == "Age":
                        features[key] = 50
                    else:
                        features[key] = 0
            
            # Create feature summary
            feature_summary = {
                "extracted_features": features,
                "feature_sources": {
                    "Age": "deidentified_medical_data",
                    "Gender": "patient_data",
                    "Diabetes": "entity_extraction + medical_codes + pharmacy_data",
                    "High_BP": "entity_extraction + medical_codes + pharmacy_data",
                    "Smoking": "entity_extraction + medical_codes"
                },
                "feature_interpretation": {
                    "Age": f"{features['Age']} years old",
                    "Gender": "Male" if features["Gender"] == 1 else "Female",
                    "Diabetes": "Yes" if features["Diabetes"] == 1 else "No",
                    "High_BP": "Yes" if features["High_BP"] == 1 else "No",
                    "Smoking": "Yes" if features["Smoking"] == 1 else "No"
                },
                "model_info": {
                    "model_type": "fastapi_server",
                    "features_expected": ["Age", "Gender", "Diabetes", "High_BP", "Smoking"],
                    "features_count": 5,
                    "fastapi_server_url": self.config.heart_attack_api_url,
                    "data_types": {key: type(value).__name__ for key, value in features.items()}
                }
            }
            
            logger.info(f"✅ Extracted {len(features)} features for FastAPI heart attack prediction")
            logger.info(f"📊 Features: Age={features['Age']}, Gender={'M' if features['Gender']==1 else 'F'}, Diabetes={'Y' if features['Diabetes']==1 else 'N'}, High_BP={'Y' if features['High_BP']==1 else 'N'}, Smoking={'Y' if features['Smoking']==1 else 'N'}")
            logger.info(f"🔍 Data types: {feature_summary['model_info']['data_types']}")
            
            return feature_summary
            
        except Exception as e:
            logger.error(f"Error extracting heart attack features for FastAPI model: {e}")
            return {"error": f"Feature extraction failed: {str(e)}"}
