def _call_heart_attack_prediction_sync(self, features: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous heart attack prediction call to avoid event loop conflicts"""
    try:
        import requests  # Import requests library
        
        # Try multiple endpoint formats for compatibility
        endpoints = [
            f"{self.config.heart_attack_api_url}/predict",
            f"{self.config.heart_attack_api_url}/predict-simple"
        ]
        
        # Use the features directly (they should already be integers)
        params = {
            "age": int(features.get("age", 50)),
            "gender": int(features.get("gender", 0)),
            "diabetes": int(features.get("diabetes", 0)),
            "high_bp": int(features.get("high_bp", 0)),
            "smoking": int(features.get("smoking", 0))
        }
        
        logger.info(f"📤 Sending synchronous prediction request: {params}")
        
        # Try POST with JSON body first
        try:
            response = requests.post(endpoints[0], json=params, timeout=30)
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Synchronous FastAPI prediction successful (JSON): {result}")
                return {
                    "success": True,
                    "prediction_data": result,
                    "method": "POST_JSON_SYNC",
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
                logger.info(f"✅ Synchronous FastAPI prediction successful (params): {result}")
                return {
                    "success": True,
                    "prediction_data": result,
                    "method": "POST_PARAMS_SYNC",
                    "endpoint": endpoints[1]
                }
            else:
                error_text = response.text
                logger.error(f"❌ All synchronous FastAPI methods failed. Status {response.status_code}: {error_text}")
                return {
                    "success": False,
                    "error": f"FastAPI server error {response.status_code}: {error_text}",
                    "tried_endpoints": endpoints
                }
        except Exception as e:
            logger.error(f"Parameters method also failed: {str(e)}")
            return {
                "success": False,
                "error": f"All synchronous prediction methods failed. Last error: {str(e)}",
                "tried_endpoints": endpoints
            }
            
    except ImportError:
        logger.error("requests library not found. Please install: pip install requests")
        return {
            "success": False,
            "error": "requests library not installed. Run: pip install requests"
        }
    except Exception as e:
        logger.error(f"Error in synchronous FastAPI call: {e}")
        return {
            "success": False,
            "error": f"Synchronous FastAPI call failed: {str(e)}"
        }
