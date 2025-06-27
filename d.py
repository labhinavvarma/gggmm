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
    """Handles all API integrations including Snowflake Cortex and FastAPI server"""
    
    def __init__(self, config):
        self.config = config
        logger.info("🔗 HealthAPIIntegrator initialized")
        logger.info(f"🌐 Snowflake API URL: {self.config.api_url}")
        logger.info(f"❤️ FastAPI Server: {self.config.heart_attack_api_url}")
    
    def call_llm(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Call Snowflake Cortex API with the user message"""
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
        """Fetch data from backend APIs (MCID, Medical, Pharmacy)"""
        try:
            logger.info(f"📡 Calling Backend API: {self.config.fastapi_url}/all")
            
            response = requests.post(
                f"{self.config.fastapi_url}/all", 
                json=patient_data, 
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                api_data = response.json()
                
                result = {
                    "mcid_output": api_data.get('mcid_search', {}),
                    "medical_output": api_data.get('medical_submit', {}),
                    "pharmacy_output": api_data.get('pharmacy_submit', {}),
                    "token_output": api_data.get('get_token', {})
                }
                
                logger.info("✅ Successfully fetched all API data")
                return result
                
            else:
                error_msg = f"Backend API call failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                return {"error": error_msg}
                
        except Exception as e:
            error_msg = f"Error fetching backend data: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    async def test_fastapi_connection(self) -> Dict[str, Any]:
        """Test the FastAPI server connection"""
        try:
            logger.info(f"🧪 Testing FastAPI server connection at {self.config.heart_attack_api_url}...")
            
            health_url = f"{self.config.heart_attack_api_url}/health"
            timeout = aiohttp.ClientTimeout(total=10)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_url) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        
                        # Test prediction endpoint with sample data
                        test_params = {
                            "age": 50,
                            "gender": 1,
                            "diabetes": 0,
                            "high_bp": 0,
                            "smoking": 0
                        }
                        
                        predict_url = f"{self.config.heart_attack_api_url}/predict"
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
                "error": "FastAPI server timeout",
                "server_url": self.config.heart_attack_api_url
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"FastAPI connection test failed: {str(e)}",
                "server_url": self.config.heart_attack_api_url
            }

    async def call_fastapi_heart_attack_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Call FastAPI server for heart attack prediction"""
        try:
            logger.info(f"🔗 Calling FastAPI server for heart attack prediction...")
            logger.info(f"📊 Features: {features}")
            
            predict_url = f"{self.config.heart_attack_api_url}/predict"
            
            # Ensure all values are integers as required by the server
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

    def test_llm_connection(self) -> Dict[str, Any]:
        """Test the Snowflake Cortex API connection with a simple query"""
        try:
            logger.info("🧪 Testing Snowflake Cortex API connection...")
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
                    "model": self.config.model
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Connection test failed: {str(e)}",
                "endpoint": self.config.api_url
            }
