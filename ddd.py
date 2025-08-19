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

class OptimizedHealthAPIIntegrator:
    """OPTIMIZED API integrator with FAST batch processing and reduced timeouts"""

    def __init__(self, config):
        self.config = config
        logger.info("🚀 OptimizedHealthAPIIntegrator initialized")
        logger.info(f"⚡ Reduced timeout: {self.config.timeout}s for faster processing")
        logger.info(f"🌐 Snowflake API: {self.config.api_url}")
        logger.info(f"📡 Backend API: {self.config.fastapi_url}")
        logger.info(f"❤️ Heart Attack ML API: {self.config.heart_attack_api_url}")

    def call_llm_fast(self, user_message: str, system_message: Optional[str] = None) -> str:
        """FAST Snowflake Cortex API call with reduced timeout"""
        try:
            session_id = str(uuid.uuid4())
            sys_msg = system_message or self.config.sys_msg

            logger.info(f"⚡ FAST LLM call - {len(user_message)} chars")

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
                    "app_lvl_prefix": "edadip",
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
                timeout=self.config.timeout  # Reduced timeout for faster processing
            )

            if response.status_code == 200:
                try:
                    raw = response.text
                    if "end_of_stream" in raw:
                        answer, _, _ = raw.partition("end_of_stream")
                        bot_reply = answer.strip()
                    else:
                        bot_reply = raw.strip()

                    logger.info("✅ FAST LLM call successful")
                    return bot_reply

                except Exception as e:
                    error_msg = f"Fast LLM parse error: {e}"
                    logger.error(error_msg)
                    return f"Parse Error: {error_msg}"
            else:
                error_msg = f"Fast LLM API error {response.status_code}: {response.text[:200]}"
                logger.error(error_msg)
                return f"API Error {response.status_code}: {response.text[:500]}"

        except requests.exceptions.Timeout:
            error_msg = f"Fast LLM timeout after {self.config.timeout}s"
            logger.error(error_msg)
            return f"Timeout Error: {error_msg}"
        except requests.exceptions.ConnectionError:
            error_msg = f"Fast LLM connection failed: {self.config.api_url}"
            logger.error(error_msg)
            return f"Connection Error: {error_msg}"
        except Exception as e:
            error_msg = f"Fast LLM unexpected error: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

    def call_llm_isolated(self, user_message: str, system_message: Optional[str] = None) -> str:
        """OPTIMIZED isolated LLM call for BATCH code explanations - CRITICAL for batch processing"""
        try:
            session_id = str(uuid.uuid4()) + "_batch_isolated"
            sys_msg = system_message or "You are a medical coding expert. Return only valid JSON with brief explanations."

            logger.info(f"🚀 BATCH ISOLATED LLM: {user_message[:100]}...")

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
                    "app_lvl_prefix": "edadip",
                    "user_id": "batch_isolated",
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
                timeout=25  # Slightly longer for batch processing
            )

            if response.status_code == 200:
                try:
                    raw = response.text
                    if "end_of_stream" in raw:
                        answer, _, _ = raw.partition("end_of_stream")
                        bot_reply = answer.strip()
                    else:
                        bot_reply = raw.strip()

                    logger.info(f"✅ BATCH ISOLATED SUCCESS: {len(bot_reply)} chars returned")
                    return bot_reply

                except Exception as e:
                    error_msg = f"Batch isolated parse error: {e}"
                    logger.warning(error_msg)
                    return "Explanation unavailable"
            else:
                logger.warning(f"❌ BATCH ISOLATED API error {response.status_code}")
                return "Explanation unavailable"

        except Exception as e:
            logger.warning(f"❌ BATCH ISOLATED call failed: {str(e)}")
            return "Explanation unavailable"

    def call_llm(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Wrapper for backward compatibility - uses fast call"""
        return self.call_llm_fast(user_message, system_message)

    def fetch_backend_data_fast(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """FAST backend data fetch with reduced timeout"""
        try:
            logger.info(f"📡 FAST Backend API call: {self.config.fastapi_url}/all")

            # Quick validation
            required_fields = ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"]
            missing_fields = [field for field in required_fields if not patient_data.get(field)]
            
            if missing_fields:
                return {"error": f"Missing required fields: {', '.join(missing_fields)}"}

            response = requests.post(
                f"{self.config.fastapi_url}/all",
                json=patient_data,
                timeout=self.config.timeout  # Reduced timeout
            )

            if response.status_code == 200:
                api_data = response.json()

                # Fast result mapping
                result = {
                    "mcid_output": self._fast_process_response(api_data.get('mcid_search', {}), 'mcid'),
                    "medical_output": self._fast_process_response(api_data.get('medical_submit', {}), 'medical'),
                    "pharmacy_output": self._fast_process_response(api_data.get('pharmacy_submit', {}), 'pharmacy'),
                    "token_output": self._fast_process_response(api_data.get('get_token', {}), 'token')
                }

                logger.info("✅ FAST backend API data fetch successful")
                return result

            else:
                error_msg = f"Fast backend API failed {response.status_code}: {response.text[:200]}"
                logger.error(error_msg)
                return {"error": error_msg}

        except Exception as e:
            error_msg = f"Fast backend fetch error: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    def fetch_backend_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper for backward compatibility - uses fast fetch"""
        return self.fetch_backend_data_fast(patient_data)

    def _fast_process_response(self, response_data: Dict[str, Any], service_name: str) -> Dict[str, Any]:
        """Fast API response processing"""
        if not response_data:
            return {"error": f"No {service_name} data", "service": service_name}

        # Handle error responses quickly
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

        # Handle other formats
        return {
            "status_code": response_data.get("status_code", 200),
            "body": response_data,
            "service": service_name,
            "timestamp": datetime.now().isoformat()
        }

    async def test_ml_connection_fast(self) -> Dict[str, Any]:
        """FAST ML API server connection test with reduced timeout"""
        try:
            logger.info(f"⚡ FAST ML API test: {self.config.heart_attack_api_url}")

            health_url = f"{self.config.heart_attack_api_url}/health"
            timeout = aiohttp.ClientTimeout(total=10)  # Reduced timeout

            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Quick health check
                async with session.get(health_url) as response:
                    if response.status == 200:
                        health_data = await response.json()

                        # Quick prediction test
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
                                    "connection_method": "fast_optimized"
                                }
                            else:
                                error_text = await pred_response.text()
                                return {
                                    "success": False,
                                    "error": f"Fast prediction endpoint error {pred_response.status}: {error_text[:200]}",
                                    "server_url": self.config.heart_attack_api_url
                                }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"Fast health endpoint error {response.status}: {error_text[:200]}",
                            "server_url": self.config.heart_attack_api_url
                        }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Fast ML API timeout - server may be down",
                "server_url": self.config.heart_attack_api_url
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Fast ML API test failed: {str(e)}",
                "server_url": self.config.heart_attack_api_url
            }

    async def call_ml_heart_attack_prediction_fast(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """FAST ML API heart attack prediction with reduced timeout"""
        try:
            logger.info(f"🚀 FAST ML prediction call...")
            logger.info(f"📊 Features: {features}")

            endpoints = [
                f"{self.config.heart_attack_api_url}/predict",
                f"{self.config.heart_attack_api_url}/predict-simple"
            ]

            # Fast integer conversion
            params = {
                "age": int(features.get("age", 50)),
                "gender": int(features.get("gender", 0)),
                "diabetes": int(features.get("diabetes", 0)),
                "high_bp": int(features.get("high_bp", 0)),
                "smoking": int(features.get("smoking", 0))
            }

            logger.info(f"📤 Fast params: {params}")

            timeout = aiohttp.ClientTimeout(total=15)  # Reduced timeout

            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Try first endpoint quickly
                try:
                    async with session.post(endpoints[0], json=params) as response:
                        if response.status == 200:
                            result = await response.json()
                            logger.info(f"✅ FAST ML prediction successful: {result}")
                            return {
                                "success": True,
                                "prediction_data": result,
                                "method": "FAST_POST_JSON",
                                "endpoint": endpoints[0]
                            }
                        else:
                            logger.warning(f"Fast first endpoint failed: {response.status}")
                except Exception as e:
                    logger.warning(f"Fast first method failed: {str(e)}")

                # Try second endpoint as fallback
                try:
                    async with session.post(endpoints[1], params=params) as response:
                        if response.status == 200:
                            result = await response.json()
                            logger.info(f"✅ FAST ML prediction fallback successful: {result}")
                            return {
                                "success": True,
                                "prediction_data": result,
                                "method": "FAST_POST_PARAMS",
                                "endpoint": endpoints[1]
                            }
                        else:
                            error_text = await response.text()
                            logger.error(f"❌ All fast ML endpoints failed. Status {response.status}: {error_text[:200]}")
                            return {
                                "success": False,
                                "error": f"Fast ML server error {response.status}: {error_text[:200]}",
                                "tried_endpoints": endpoints
                            }
                except Exception as e:
                    logger.error(f"Fast fallback method failed: {str(e)}")
                    return {
                        "success": False,
                        "error": f"All fast prediction methods failed: {str(e)}",
                        "tried_endpoints": endpoints
                    }

        except asyncio.TimeoutError:
            logger.error("❌ Fast ML API timeout")
            return {
                "success": False,
                "error": "Fast ML API timeout - check server status"
            }
        except Exception as e:
            logger.error(f"Fast ML API error: {e}")
            return {
                "success": False,
                "error": f"Fast ML API failed: {str(e)}"
            }

    def test_llm_connection_fast(self) -> Dict[str, Any]:
        """FAST Snowflake Cortex API connection test"""
        try:
            logger.info("⚡ FAST LLM connection test...")
            test_response = self.call_llm_fast("Hello, respond with 'Fast Snowflake connection successful'")

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
                    "connection_optimized": True
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Fast LLM test failed: {str(e)}",
                "endpoint": self.config.api_url
            }

    def test_isolated_llm_connection(self) -> Dict[str, Any]:
        """Test isolated LLM connection for BATCH processing - CRITICAL TEST"""
        try:
            logger.info("🚀 Testing BATCH ISOLATED LLM connection...")
            test_prompt = "Explain ICD-10 code 'I10' in JSON format: {\"I10\": \"brief explanation\"}"
            test_response = self.call_llm_isolated(test_prompt)

            if test_response and test_response != "Explanation unavailable":
                return {
                    "success": True,
                    "message": "BATCH isolated LLM connection successful",
                    "test_prompt": test_prompt,
                    "response": test_response[:200],
                    "endpoint": self.config.api_url,
                    "model": self.config.model,
                    "batch_processing_enabled": True
                }
            else:
                return {
                    "success": False,
                    "error": f"BATCH isolated LLM returned: {test_response}",
                    "endpoint": self.config.api_url,
                    "model": self.config.model,
                    "batch_processing_enabled": False
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"BATCH isolated LLM test failed: {str(e)}",
                "endpoint": self.config.api_url,
                "batch_processing_enabled": False
            }

    def test_backend_connection_fast(self) -> Dict[str, Any]:
        """FAST backend server connection test"""
        try:
            logger.info("⚡ FAST backend API test...")

            health_url = f"{self.config.fastapi_url}/health"
            response = requests.get(health_url, timeout=5)  # Reduced timeout

            if response.status_code == 200:
                health_data = response.json()
                return {
                    "success": True,
                    "health_data": health_data,
                    "backend_url": self.config.fastapi_url,
                    "connection_optimized": True
                }
            else:
                return {
                    "success": False,
                    "error": f"Fast backend health check failed: {response.status_code}",
                    "backend_url": self.config.fastapi_url
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Fast backend test failed: {str(e)}",
                "backend_url": self.config.fastapi_url
            }

    def test_all_connections_fast(self) -> Dict[str, Any]:
        """FAST test of all API connections"""
        logger.info("⚡ FAST testing ALL connections...")

        results = {
            "llm_connection": self.test_llm_connection_fast(),
            "isolated_llm_connection": self.test_isolated_llm_connection(),  # CRITICAL for batch processing
            "backend_connection": self.test_backend_connection_fast()
        }

        # Fast ML connection test
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results["ml_connection"] = loop.run_until_complete(self.test_ml_connection_fast())
            loop.close()
        except Exception as e:
            results["ml_connection"] = {
                "success": False,
                "error": f"Fast ML test failed: {str(e)}"
            }

        # Fast summary
        all_success = all(result.get("success", False) for result in results.values())
        batch_processing_ready = results["isolated_llm_connection"].get("success", False)
        
        results["overall_status"] = {
            "all_connections_successful": all_success,
            "successful_connections": sum(1 for result in results.values() if result.get("success", False)),
            "total_connections": len(results) - 1,
            "batch_processing_ready": batch_processing_ready,
            "optimization_level": "fast_processing_enabled"
        }

        logger.info(f"⚡ FAST connection test complete: {results['overall_status']['successful_connections']}/{results['overall_status']['total_connections']} successful")
        logger.info(f"🚀 Batch processing ready: {batch_processing_ready}")

        return results

    # Backward compatibility methods
    def test_llm_connection(self) -> Dict[str, Any]:
        """Backward compatibility - uses fast test"""
        return self.test_llm_connection_fast()

    def test_backend_connection(self) -> Dict[str, Any]:
        """Backward compatibility - uses fast test"""
        return self.test_backend_connection_fast()

    async def test_ml_connection(self) -> Dict[str, Any]:
        """Backward compatibility - uses fast test"""
        return await self.test_ml_connection_fast()

    def test_all_connections(self) -> Dict[str, Any]:
        """Backward compatibility - uses fast test"""
        return self.test_all_connections_fast()
