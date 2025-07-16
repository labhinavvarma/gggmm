def _call_heart_attack_prediction_sync(self, features: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous heart attack prediction call to avoid event loop conflicts"""
    try:
        import requests
        
        # Log the features we received
        logger.info(f"🔍 Received features for prediction: {features}")
        
        # Validate that we have features
        if not features:
            logger.error("❌ No features provided for prediction")
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
        
        # Prepare parameters with validation
        params = {
            "age": int(features.get("age", 50)),
            "gender": int(features.get("gender", 0)),
            "diabetes": int(features.get("diabetes", 0)),
            "high_bp": int(features.get("high_bp", 0)),
            "smoking": int(features.get("smoking", 0))
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
