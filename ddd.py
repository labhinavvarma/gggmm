def call_fastapi_heart_attack_prediction_sync(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous FastAPI heart attack prediction to avoid event loop conflicts"""
        try:
            logger.info(f"🔗 Calling FastAPI server for heart attack prediction (sync)...")
            logger.info(f"📊 Features: {features}")
            
            # Try multiple endpoint formats for compatibility
            endpoints = [
                f"{self.config.heart_attack_api_url}/predict",
                f"{self.config.heart_attack_api_url}/predict-simple"
            ]
            
            # Ensure all values are integers as required by the server
            params = {
                "age": int(features.get("age", 50)),
                "gender": int(features.get("gender", 0)),
                "diabetes": int(features.get("diabetes", 0)),
                "high_bp": int(features.get("high_bp", 0)),
                "smoking": int(features.get("smoking", 0))
            }
            
            logger.info(f"📤 Sending parameters: {params}")
            
            # Try POST with JSON body first
            try:
                response = requests.post(endpoints[0], json=params, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ FastAPI prediction successful (JSON): {result}")
                    return {
                        "success": True,
                        "prediction_data": result,
                        "method": "POST_JSON",
                        "endpoint": endpoints[0]
                    }
                else:
                    logger.warning(f"JSON method failed with status {response.status_code}")
            except Exception as e:
                logger.warning(f"JSON method failed: {str(e)}")
            
            # Try POST with query parameters as fallback
            try:
                response = requests.post(endpoints[1], params=params, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ FastAPI prediction successful (params): {result}")
                    return {
                        "success": True,
                        "prediction_data": result,
                        "method": "POST_PARAMS",
                        "endpoint": endpoints[1]
                    }
                else:
                    error_text = response.text
                    logger.error(f"❌ All FastAPI methods failed. Status {response.status_code}: {error_text}")
                    return {
                        "success": False,
                        "error": f"FastAPI server error {response.status_code}: {error_text}",
                        "tried_endpoints": endpoints
                    }
            except Exception as e:
                logger.error(f"Parameters method also failed: {str(e)}")
                return {
                    "success": False,
                    "error": f"All prediction methods failed. Last error: {str(e)}",
                    "tried_endpoints": endpoints
                }
                        
        except Exception as e:
            logger.error(f"Error calling FastAPI server: {e}")
            return {
                "success": False,
                "error": f"FastAPI call failed: {str(e)}"
            }

    def test_fastapi_connection_sync(self) -> Dict[str, Any]:
        """Synchronous FastAPI server connection test"""
        try:
            logger.info(f"🧪 Testing FastAPI server connection at {self.config.heart_attack_api_url}...")
            
            health_url = f"{self.config.heart_attack_api_url}/health"
            
            # Test health endpoint
            response = requests.get(health_url, timeout=15)
            if response.status_code == 200:
                health_data = response.json()
                
                # Test prediction endpoint with sample data
                test_features = {
                    "age": 50,
                    "gender": 1,
                    "diabetes": 0,
                    "high_bp": 0,
                    "smoking": 0
                }
                
                predict_url = f"{self.config.heart_attack_api_url}/predict"
                pred_response = requests.post(predict_url, json=test_features, timeout=15)
                
                if pred_response.status_code == 200:
                    pred_data = pred_response.json()
                    return {
                        "success": True,
                        "health_check": health_data,
                        "prediction_test": pred_data,
                        "server_url": self.config.heart_attack_api_url,
                        "test_features": test_features,
                        "connection_method": "synchronous"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Prediction endpoint error {pred_response.status_code}: {pred_response.text}",
                        "server_url": self.config.heart_attack_api_url
                    }
            else:
                return {
                    "success": False,
                    "error": f"Health endpoint error {response.status_code}: {response.text}",
                    "server_url": self.config.heart_attack_api_url
                }
                        
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "FastAPI server timeout - server may be down",
                "server_url": self.config.heart_attack_api_url
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"FastAPI connection test failed: {str(e)}",
                "server_url": self.config.heart_attack_api_url
            }

    def test_llm_connection(self) -> Dict[str, Any]:
