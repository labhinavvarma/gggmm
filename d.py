
import json
import requests
import urllib3
import uuid
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthAPIIntegrator:
    """Enhanced API integrator compatible with MCP server and FastAPI heart attack prediction"""
    
    def __init__(self, config):
        self.config = config
        logger.info("🔗 Enhanced HealthAPIIntegrator initialized")
        logger.info(f"🌐 Snowflake API URL: {self.config.api_url}")
        logger.info(f"📡 Backend API URL: {self.config.fastapi_url}")
        logger.info(f"❤️ Heart Attack API URL: {self.config.heart_attack_api_url}")
    
    def call_llm(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Enhanced Snowflake Cortex API call with better error handling"""
        try:
            session_id = str(uuid.uuid4())
            sys_msg = system_message or self.config.sys_msg
            
            logger.info(f"🤖 Calling Snowflake Cortex API")
            logger.info(f"🤖 Message length: {len(user_message)} characters")
            
            payload = {
                "query": {
                    "aplctn_cd": self.config.aplctn_cd,
                    "app_id": self.config.app_id,
                    "api_key": self.config.api_key,
                    "method": "cortex",
                    "model": self.config.model,
                    "sys_msg": sys_msg,
                    "limit_convs": "0",
                    "prompt": {
                        "messages": [
                            {
                                "role": "user",
                                "content": user_message
                            }
                        ]
                    },
                    "app_lvl_prefix": "",
                    "user_id": "",
                    "session_id": session_id
                }
            }
            
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "Authorization": f'Snowflake Token="{self.config.api_key}"'
            }
            
            response = requests.post(
                self.config.api_url, 
                headers=headers, 
                json=payload, 
                verify=False,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                try:
                    raw = response.text
                    if "end_of_stream" in raw:
                        answer, _, _ = raw.partition("end_of_stream")
                        bot_reply = answer.strip()
                    else:
                        bot_reply = raw.strip()
                    
                    logger.info("✅ Snowflake Cortex API call successful")
                    return bot_reply
                    
                except Exception as e:
                    error_msg = f"Error parsing Snowflake response: {e}"
                    logger.error(error_msg)
                    return f"Parse Error: {error_msg}"
            else:
                error_msg = f"Snowflake Cortex API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                return f"API Error {response.status_code}: {response.text[:500]}"
                
        except requests.exceptions.Timeout:
            error_msg = f"Snowflake Cortex API timeout after {self.config.timeout} seconds"
            logger.error(error_msg)
            return f"Timeout Error: {error_msg}"
        except requests.exceptions.ConnectionError:
            error_msg = f"Cannot connect to Snowflake Cortex API: {self.config.api_url}"
            logger.error(error_msg)
            return f"Connection Error: {error_msg}"
        except Exception as e:
            error_msg = f"Unexpected error calling Snowflake Cortex API: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
    
    def fetch_backend_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced backend data fetch compatible with MCP server structure"""
        try:
            logger.info(f"📡 Calling MCP-compatible Backend API: {self.config.fastapi_url}/all")
            
            # Enhanced payload validation
            required_fields = ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"]
            for field in required_fields:
                if not patient_data.get(field):
                    return {"error": f"Missing required field: {field}"}
            
            response = requests.post(
                f"{self.config.fastapi_url}/all", 
                json=patient_data, 
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                api_data = response.json()
                
                # Enhanced result mapping for MCP server compatibility
                result = {
                    "mcid_output": self._process_api_response(api_data.get('mcid_search', {}), 'mcid'),
                    "medical_output": self._process_api_response(api_data.get('medical_submit', {}), 'medical'),
                    "pharmacy_output": self._process_api_response(api_data.get('pharmacy_submit', {}), 'pharmacy'),
                    "token_output": self._process_api_response(api_data.get('get_token', {}), 'token')
                }
                
                logger.info("✅ Successfully fetched all MCP-compatible API data")
                return result
                
            else:
                error_msg = f"Backend API call failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                return {"error": error_msg}
                
        except Exception as e:
            error_msg = f"Error fetching backend data: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _process_api_response(self, response_data: Dict[str, Any], service_name: str) -> Dict[str, Any]:
        """Process API response to ensure compatibility"""
        if not response_data:
            return {"error": f"No {service_name} data received", "service": service_name}
        
        # Handle error responses
        if "error" in response_data:
            return {
                "error": response_data["error"],
                "service": service_name,
                "status_code": response_data.get("status_code", 500)
            }
        
        # Handle successful responses
        if response_data.get("status_code") == 200 and "body" in response_data:
            return {
                "status_code": 200,
                "body": response_data["body"],
                "service": service_name,
                "timestamp": response_data.get("timestamp", datetime.now().isoformat())
            }
        
        # Handle other response formats
        return {
            "status_code": response_data.get("status_code", 200),
            "body": response_data,
            "service": service_name,
            "timestamp": datetime.now().isoformat()
        }
    
    async def test_fastapi_connection(self) -> Dict[str, Any]:
        """Enhanced FastAPI server connection test"""
        try:
            logger.info(f"🧪 Testing Enhanced FastAPI server connection at {self.config.heart_attack_api_url}...")
            
            health_url = f"{self.config.heart_attack_api_url}/health"
            timeout = aiohttp.ClientTimeout(total=15)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Test health endpoint
                async with session.get(health_url) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        
                        # Test prediction endpoint with sample data
                        test_features = {
                            "age": 50,
                            "gender": 1,
                            "diabetes": 0,
                            "high_bp": 0,
                            "smoking": 0
                        }
                        
                        predict_url = f"{self.config.heart_attack_api_url}/predict"
                        async with session.post(predict_url, json=test_features) as pred_response:
                            if pred_response.status == 200:
                                pred_data = await pred_response.json()
                                return {
                                    "success": True,
                                    "health_check": health_data,
                                    "prediction_test": pred_data,
                                    "server_url": self.config.heart_attack_api_url,
                                    "test_features": test_features,
                                    "connection_method": "enhanced"
                                }
                            else:
                                error_text = await pred_response.text()
                                return {
                                    "success": False,
                                    "error": f"Prediction endpoint error {pred_response.status}: {error_text}",
                                    "server_url": self.config.heart_attack_api_url
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
                "error": "FastAPI server timeout - server may be down",
                "server_url": self.config.heart_attack_api_url
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"FastAPI connection test failed: {str(e)}",
                "server_url": self.config.heart_attack_api_url
            }

    async def call_fastapi_heart_attack_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced FastAPI heart attack prediction with multiple endpoint support"""
        try:
            logger.info(f"🔗 Calling Enhanced FastAPI server for heart attack prediction...")
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
            
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Try POST with JSON body first
                try:
                    async with session.post(endpoints[0], json=params) as response:
                        if response.status == 200:
                            result = await response.json()
                            logger.info(f"✅ FastAPI prediction successful (JSON): {result}")
                            return {
                                "success": True,
                                "prediction_data": result,
                                "method": "POST_JSON",
                                "endpoint": endpoints[0]
                            }
                        else:
                            logger.warning(f"JSON method failed with status {response.status}")
                except Exception as e:
                    logger.warning(f"JSON method failed: {str(e)}")
                
                # Try POST with query parameters as fallback
                try:
                    async with session.post(endpoints[1], params=params) as response:
                        if response.status == 200:
                            result = await response.json()
                            logger.info(f"✅ FastAPI prediction successful (params): {result}")
                            return {
                                "success": True,
                                "prediction_data": result,
                                "method": "POST_PARAMS",
                                "endpoint": endpoints[1]
                            }
                        else:
                            error_text = await response.text()
                            logger.error(f"❌ All FastAPI methods failed. Status {response.status}: {error_text}")
                            return {
                                "success": False,
                                "error": f"FastAPI server error {response.status}: {error_text}",
                                "tried_endpoints": endpoints
                            }
                except Exception as e:
                    logger.error(f"Parameters method also failed: {str(e)}")
                    return {
                        "success": False,
                        "error": f"All prediction methods failed. Last error: {str(e)}",
                        "tried_endpoints": endpoints
                    }
                        
        except asyncio.TimeoutError:
            logger.error("❌ FastAPI server timeout")
            return {
                "success": False,
                "error": "FastAPI server timeout - check if server is running"
            }
        except Exception as e:
            logger.error(f"Error calling FastAPI server: {e}")
            return {
                "success": False,
                "error": f"FastAPI call failed: {str(e)}"
            }

    def test_llm_connection(self) -> Dict[str, Any]:
        """Enhanced Snowflake Cortex API connection test"""
        try:
            logger.info("🧪 Testing Enhanced Snowflake Cortex API connection...")
            test_response = self.call_llm("Hello, please respond with 'Snowflake Cortex connection successful'")
            
            if test_response.startswith("Error"):
                return {
                    "success": False,
                    "error": test_response,
                    "endpoint": self.config.api_url
                }
            else:
                return {
                    "success": True,
                    "response": test_response,
                    "endpoint": self.config.api_url,
                    "model": self.config.model,
                    "connection_enhanced": True
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Enhanced connection test failed: {str(e)}",
                "endpoint": self.config.api_url
            }

    def test_backend_connection(self) -> Dict[str, Any]:
        """Test backend MCP server connection"""
        try:
            logger.info("🧪 Testing MCP Backend API connection...")
            
            # Test health endpoint
            health_url = f"{self.config.fastapi_url}/health"
            response = requests.get(health_url, timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                return {
                    "success": True,
                    "health_data": health_data,
                    "backend_url": self.config.fastapi_url,
                    "mcp_compatible": True
                }
            else:
                return {
                    "success": False,
                    "error": f"Backend health check failed: {response.status_code}",
                    "backend_url": self.config.fastapi_url
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Backend connection test failed: {str(e)}",
                "backend_url": self.config.fastapi_url
            }
