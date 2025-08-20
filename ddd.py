import json
import requests
import urllib3
import uuid
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, Any, Optional
import logging
import time

# Disable SSL warnings for internal APIs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthAPIIntegrator:
    """Comprehensive Health API Integrator for Healthcare Analysis System"""

    def __init__(self, config):
        self.config = config
        logger.info("🔧 HealthAPIIntegrator initialized")
        logger.info(f"🌐 Snowflake Cortex API: {self.config.api_url}")
        logger.info(f"📡 Backend FastAPI: {self.config.fastapi_url}")
        logger.info(f"❤️ Heart Attack ML API: {self.config.heart_attack_api_url}")
        logger.info(f"🤖 Model: {self.config.model}")
        logger.info(f"⏱️ Timeout: {self.config.timeout}s")

    # ===== SNOWFLAKE CORTEX LLM API METHODS =====

    def call_llm(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Primary LLM call method for healthcare analysis"""
        try:
            session_id = str(uuid.uuid4()) + "_healthcare_analysis"
            sys_msg = system_message or self.config.sys_msg

            logger.info(f"🤖 Healthcare LLM call - {len(user_message)} chars")

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
                "Authorization": f'Snowflake Token="{self.config.api_key}"',
                "X-Healthcare-Analysis": "comprehensive",
                "X-Clinical-Context": "enabled"
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
                    raw_response = response.text
                    if "end_of_stream" in raw_response:
                        answer, _, _ = raw_response.partition("end_of_stream")
                        bot_reply = answer.strip()
                    else:
                        bot_reply = raw_response.strip()

                    logger.info("✅ Healthcare LLM call successful")
                    return bot_reply

                except Exception as e:
                    error_msg = f"Healthcare LLM response parsing error: {e}"
                    logger.error(error_msg)
                    return f"Error: {error_msg}"
            else:
                error_msg = f"Healthcare LLM API error {response.status_code}: {response.text[:200]}"
                logger.error(error_msg)
                return f"Error: {error_msg}"

        except requests.exceptions.Timeout:
            error_msg = f"Healthcare LLM timeout after {self.config.timeout}s"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        except requests.exceptions.ConnectionError:
            error_msg = "Healthcare LLM connection failed"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Healthcare LLM unexpected error: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

    def call_llm_isolated_enhanced(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Enhanced isolated LLM call for code explanations and batch processing"""
        try:
            session_id = str(uuid.uuid4()) + "_isolated_batch"
            sys_msg = system_message or """You are Dr. CodeAI, a medical coding expert specializing in healthcare terminology. Provide clear, concise explanations of medical codes, medications, and healthcare terminology. Keep responses brief but informative and clinically accurate."""

            logger.info(f"🔬 Isolated batch healthcare call: {user_message[:100]}...")

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
                    "app_lvl_prefix": "isolated_batch",
                    "user_id": "batch_processor",
                    "session_id": session_id
                }
            }

            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "Authorization": f'Snowflake Token="{self.config.api_key}"',
                "X-Healthcare-Batch": "isolated",
                "X-Medical-Coding": "enabled"
            }

            response = requests.post(
                self.config.api_url,
                headers=headers,
                json=payload,
                verify=False,
                timeout=35  # Longer timeout for batch processing
            )

            if response.status_code == 200:
                try:
                    raw_response = response.text
                    if "end_of_stream" in raw_response:
                        answer, _, _ = raw_response.partition("end_of_stream")
                        bot_reply = answer.strip()
                    else:
                        bot_reply = raw_response.strip()

                    logger.info(f"✅ Isolated batch SUCCESS: {len(bot_reply)} chars returned")
                    return bot_reply

                except Exception as e:
                    logger.warning(f"Isolated batch parse error: {e}")
                    return "Detailed explanation unavailable"
            else:
                logger.warning(f"❌ Isolated batch API error {response.status_code}")
                return "Detailed explanation unavailable"

        except Exception as e:
            logger.warning(f"❌ Isolated batch call failed: {str(e)}")
            return "Detailed explanation unavailable"

    def call_llm_fast(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Fast LLM call with reduced timeout for quick responses"""
        try:
            # Temporarily reduce timeout for fast calls
            original_timeout = self.config.timeout
            self.config.timeout = 15
            
            result = self.call_llm(user_message, system_message)
            
            # Restore original timeout
            self.config.timeout = original_timeout
            
            return result
            
        except Exception as e:
            # Restore timeout even on error
            self.config.timeout = getattr(self.config, 'timeout', 30)
            logger.error(f"Fast LLM call error: {e}")
            return f"Error: {str(e)}"

    # ===== BACKEND FASTAPI METHODS =====

    def fetch_backend_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch comprehensive healthcare data from backend APIs"""
        try:
            logger.info(f"📡 Healthcare Backend API call: {self.config.fastapi_url}/all")

            # Validate required patient data fields
            required_fields = {
                "first_name": "Patient identification",
                "last_name": "Patient identification", 
                "ssn": "Unique patient identifier",
                "date_of_birth": "Age verification and risk assessment",
                "gender": "Gender-based risk calculation",
                "zip_code": "Geographic and demographic analysis"
            }
            
            missing_fields = []
            validation_errors = []
            
            for field, purpose in required_fields.items():
                if not patient_data.get(field):
                    missing_fields.append(f"{field}: {purpose}")
                else:
                    # Enhanced validation
                    if field == "ssn" and len(str(patient_data[field]).replace("-", "")) != 9:
                        validation_errors.append(f"SSN format invalid (must be 9 digits)")
                    elif field == "zip_code" and len(str(patient_data[field])) < 5:
                        validation_errors.append(f"ZIP code incomplete (minimum 5 digits)")
                    elif field == "date_of_birth":
                        try:
                            from datetime import datetime
                            birth_date = datetime.strptime(patient_data[field], '%Y-%m-%d')
                            age = (datetime.now() - birth_date).days // 365
                            if age < 0 or age > 150:
                                validation_errors.append(f"Invalid age calculated from date of birth")
                        except:
                            validation_errors.append(f"Invalid date format for date of birth")
            
            if missing_fields or validation_errors:
                all_errors = missing_fields + validation_errors
                return {"error": f"Patient data validation failed: {'; '.join(all_errors)}"}

            # Enhanced API call with comprehensive headers
            headers = {
                "Content-Type": "application/json",
                "X-Healthcare-Request": "comprehensive",
                "X-Clinical-Analysis": "enabled",
                "X-Patient-Lookup": "multi-source",
                "X-Request-Timestamp": datetime.now().isoformat()
            }

            response = requests.post(
                f"{self.config.fastapi_url}/all",
                json=patient_data,
                headers=headers,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                api_data = response.json()

                # Enhanced result processing with comprehensive error handling
                result = {
                    "mcid_output": self._process_api_response(api_data.get('mcid_search', {}), 'mcid_search'),
                    "medical_output": self._process_api_response(api_data.get('medical_submit', {}), 'medical_submit'),
                    "pharmacy_output": self._process_api_response(api_data.get('pharmacy_submit', {}), 'pharmacy_submit'),
                    "token_output": self._process_api_response(api_data.get('get_token', {}), 'get_token')
                }

                logger.info("✅ Healthcare backend API successful")
                logger.info(f"🏥 Medical data: {'Available' if result['medical_output'].get('status_code') == 200 else 'Limited/Error'}")
                logger.info(f"💊 Pharmacy data: {'Available' if result['pharmacy_output'].get('status_code') == 200 else 'Limited/Error'}")
                logger.info(f"🆔 MCID data: {'Available' if result['mcid_output'].get('status_code') == 200 else 'Limited/Error'}")
                logger.info(f"🎫 Token data: {'Available' if result['token_output'].get('status_code') == 200 else 'Limited/Error'}")
                
                return result

            elif response.status_code == 404:
                error_msg = f"Backend API endpoint not found: {self.config.fastapi_url}/all"
                logger.error(error_msg)
                return {"error": error_msg}
            elif response.status_code == 500:
                error_msg = f"Backend API server error: {response.text[:300]}"
                logger.error(error_msg)
                return {"error": error_msg}
            else:
                error_msg = f"Backend API failed {response.status_code}: {response.text[:200]}"
                logger.error(error_msg)
                return {"error": error_msg}

        except requests.exceptions.ConnectionError:
            error_msg = f"Cannot connect to backend API at {self.config.fastapi_url}. Please ensure the server is running."
            logger.error(error_msg)
            return {"error": error_msg}
        except requests.exceptions.Timeout:
            error_msg = f"Backend API timeout after {self.config.timeout}s"
            logger.error(error_msg)
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Backend fetch error: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    def _process_api_response(self, response_data: Dict[str, Any], service_name: str) -> Dict[str, Any]:
        """Process and standardize API response data"""
        if not response_data:
            return {
                "error": f"No {service_name} data received", 
                "service": service_name,
                "status": "no_data"
            }

        # Handle error responses
        if "error" in response_data:
            return {
                "error": response_data["error"],
                "service": service_name,
                "status_code": response_data.get("status_code", 500),
                "status": "error"
            }

        # Handle successful responses with status code 200
        if response_data.get("status_code") == 200 and "body" in response_data:
            return {
                "status_code": 200,
                "body": response_data["body"],
                "service": service_name,
                "timestamp": response_data.get("timestamp", datetime.now().isoformat()),
                "status": "success"
            }

        # Handle other response formats
        return {
            "status_code": response_data.get("status_code", 200),
            "body": response_data,
            "service": service_name,
            "timestamp": datetime.now().isoformat(),
            "status": "partial"
        }

    # ===== CONNECTION TESTING METHODS =====

    def test_llm_connection(self) -> Dict[str, Any]:
        """Test Snowflake Cortex LLM API connection"""
        try:
            logger.info("🧪 Testing Snowflake Cortex LLM connection...")
            
            test_prompt = """You are performing a healthcare system connectivity test.

Please respond with exactly: "Healthcare LLM connection successful and ready for clinical analysis."

This tests basic API connectivity and response formatting for healthcare applications."""

            test_response = self.call_llm(test_prompt, self.config.sys_msg)

            # Validate response
            if test_response and not test_response.startswith("Error") and "successful" in test_response:
                return {
                    "success": True,
                    "response": test_response[:200] + "..." if len(test_response) > 200 else test_response,
                    "endpoint": self.config.api_url,
                    "model": self.config.model,
                    "healthcare_ready": True,
                    "test_timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": test_response,
                    "endpoint": self.config.api_url,
                    "healthcare_ready": False,
                    "test_timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"LLM connection test failed: {str(e)}",
                "endpoint": self.config.api_url,
                "healthcare_ready": False,
                "test_timestamp": datetime.now().isoformat()
            }

    def test_backend_connection(self) -> Dict[str, Any]:
        """Test backend FastAPI server connection"""
        try:
            logger.info(f"🧪 Testing backend server at {self.config.fastapi_url}...")

            # Test health endpoint first
            health_url = f"{self.config.fastapi_url}/health"
            
            headers = {
                "X-Healthcare-Test": "connection",
                "X-Test-Timestamp": datetime.now().isoformat()
            }
            
            response = requests.get(health_url, headers=headers, timeout=10)

            if response.status_code == 200:
                health_data = response.json()
                
                # Validate health response structure
                service_status = "Healthy" if health_data.get("status") == "healthy" else "Limited"
                
                return {
                    "success": True,
                    "health_data": health_data,
                    "backend_url": self.config.fastapi_url,
                    "service_status": service_status,
                    "endpoints_available": health_data.get("endpoints", []),
                    "test_timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": f"Backend health check failed: {response.status_code} - {response.text[:200]}",
                    "backend_url": self.config.fastapi_url,
                    "test_timestamp": datetime.now().isoformat()
                }

        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": f"Cannot connect to backend server at {self.config.fastapi_url}",
                "backend_url": self.config.fastapi_url,
                "test_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Backend connection test failed: {str(e)}",
                "backend_url": self.config.fastapi_url,
                "test_timestamp": datetime.now().isoformat()
            }

    async def test_ml_connection(self) -> Dict[str, Any]:
        """Test ML API server connection (Heart Attack Prediction)"""
        try:
            logger.info(f"🧪 Testing ML API server at {self.config.heart_attack_api_url}...")

            timeout = aiohttp.ClientTimeout(total=15)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Test health endpoint
                health_url = f"{self.config.heart_attack_api_url}/health"
                
                try:
                    async with session.get(health_url) as response:
                        if response.status == 200:
                            health_data = await response.json()

                            # Test prediction endpoint with sample data
                            test_features = {
                                "age": 45,
                                "gender": 0,
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
                                        "test_prediction": pred_data,
                                        "server_url": self.config.heart_attack_api_url,
                                        "ml_model_ready": True,
                                        "test_features": test_features,
                                        "test_timestamp": datetime.now().isoformat()
                                    }
                                else:
                                    error_text = await pred_response.text()
                                    return {
                                        "success": False,
                                        "error": f"ML prediction test failed {pred_response.status}: {error_text[:200]}",
                                        "server_url": self.config.heart_attack_api_url,
                                        "ml_model_ready": False,
                                        "test_timestamp": datetime.now().isoformat()
                                    }
                        else:
                            error_text = await response.text()
                            return {
                                "success": False,
                                "error": f"ML health endpoint error {response.status}: {error_text[:200]}",
                                "server_url": self.config.heart_attack_api_url,
                                "ml_model_ready": False,
                                "test_timestamp": datetime.now().isoformat()
                            }
                            
                except aiohttp.ClientConnectorError:
                    return {
                        "success": False,
                        "error": f"Cannot connect to ML server at {self.config.heart_attack_api_url}",
                        "server_url": self.config.heart_attack_api_url,
                        "ml_model_ready": False,
                        "test_timestamp": datetime.now().isoformat()
                    }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "ML API timeout - server may be down or slow",
                "server_url": self.config.heart_attack_api_url,
                "ml_model_ready": False,
                "test_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"ML API test failed: {str(e)}",
                "server_url": self.config.heart_attack_api_url,
                "ml_model_ready": False,
                "test_timestamp": datetime.now().isoformat()
            }

    def test_all_connections(self) -> Dict[str, Any]:
        """Test all API connections comprehensively"""
        logger.info("🧪 Testing ALL healthcare API connections...")

        start_time = time.time()

        results = {
            "llm_connection": self.test_llm_connection(),
            "backend_connection": self.test_backend_connection()
        }

        # Test ML connection using asyncio
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results["ml_connection"] = loop.run_until_complete(self.test_ml_connection())
            loop.close()
        except Exception as e:
            results["ml_connection"] = {
                "success": False,
                "error": f"ML connection test failed: {str(e)}",
                "ml_model_ready": False,
                "test_timestamp": datetime.now().isoformat()
            }

        # Calculate overall summary
        all_success = all(result.get("success", False) for result in results.values())
        successful_connections = sum(1 for result in results.values() if result.get("success", False))
        total_connections = len(results)
        
        # Determine healthcare readiness
        llm_ready = results["llm_connection"].get("healthcare_ready", False)
        backend_ready = results["backend_connection"].get("success", False)
        ml_ready = results["ml_connection"].get("ml_model_ready", False)
        
        healthcare_ready = llm_ready and backend_ready  # ML is optional
        
        test_duration = time.time() - start_time
        
        results["overall_status"] = {
            "all_connections_successful": all_success,
            "successful_connections": successful_connections,
            "total_connections": total_connections,
            "healthcare_system_ready": healthcare_ready,
            "llm_ready": llm_ready,
            "backend_ready": backend_ready,
            "ml_ready": ml_ready,
            "readiness_level": "full" if all_success else "partial" if successful_connections >= 2 else "limited",
            "test_duration_seconds": round(test_duration, 2),
            "test_timestamp": datetime.now().isoformat()
        }

        logger.info(f"🧪 Connection test complete: {successful_connections}/{total_connections} successful")
        logger.info(f"🏥 Healthcare system ready: {healthcare_ready}")
        logger.info(f"📊 Readiness level: {results['overall_status']['readiness_level']}")
        logger.info(f"⏱️ Test duration: {test_duration:.2f}s")

        return results

    # ===== UTILITY METHODS =====

    def get_api_status(self) -> Dict[str, Any]:
        """Get current API configuration status"""
        return {
            "snowflake_cortex": {
                "url": self.config.api_url,
                "model": self.config.model,
                "timeout": self.config.timeout
            },
            "backend_fastapi": {
                "url": self.config.fastapi_url,
                "timeout": self.config.timeout
            },
            "ml_api": {
                "url": self.config.heart_attack_api_url,
                "threshold": self.config.heart_attack_threshold
            },
            "configuration_timestamp": datetime.now().isoformat()
        }

    def update_timeout(self, new_timeout: int) -> bool:
        """Update API timeout configuration"""
        try:
            old_timeout = self.config.timeout
            self.config.timeout = new_timeout
            logger.info(f"⏱️ Timeout updated: {old_timeout}s → {new_timeout}s")
            return True
        except Exception as e:
            logger.error(f"Failed to update timeout: {e}")
            return False

    # ===== BACKWARD COMPATIBILITY METHODS =====

    def call_llm_enhanced(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Backward compatibility - uses primary call_llm method"""
        return self.call_llm(user_message, system_message)

    def fetch_backend_data_enhanced(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility - uses primary fetch_backend_data method"""
        return self.fetch_backend_data(patient_data)

    def test_llm_connection_enhanced(self) -> Dict[str, Any]:
        """Backward compatibility - uses primary test_llm_connection method"""
        return self.test_llm_connection()

    def test_backend_connection_enhanced(self) -> Dict[str, Any]:
        """Backward compatibility - uses primary test_backend_connection method"""
        return self.test_backend_connection()

    def test_all_connections_enhanced(self) -> Dict[str, Any]:
        """Backward compatibility - uses primary test_all_connections method"""
        return self.test_all_connections()

def main():
    """Example usage and testing of HealthAPIIntegrator"""
    
    print("🔧 Health API Integrator - Comprehensive Healthcare API Management")
    print("✅ Features:")
    print("   🤖 Snowflake Cortex LLM API integration")
    print("   📡 Backend FastAPI data fetching")
    print("   ❤️ ML API heart attack prediction")
    print("   🧪 Comprehensive connection testing")
    print("   🔄 Enhanced error handling and retry logic")
    print()
    
    # Example configuration (would normally come from Config class)
    class MockConfig:
        api_url = "https://sfassist.edagenaipreprod.awsdns.internal.das/api/cortex/complete"
        api_key = "test-key"
        app_id = "test-app"
        aplctn_cd = "test-code"
        model = "llama4-maverick"
        fastapi_url = "http://localhost:8000"
        heart_attack_api_url = "http://localhost:8000"
        heart_attack_threshold = 0.5
        timeout = 30
        sys_msg = "You are a healthcare AI assistant."
    
    config = MockConfig()
    integrator = HealthAPIIntegrator(config)
    
    print("📋 Configuration:")
    print(f"   🌐 LLM API: {config.api_url}")
    print(f"   📡 Backend: {config.fastapi_url}")
    print(f"   ❤️ ML API: {config.heart_attack_api_url}")
    print(f"   🤖 Model: {config.model}")
    print()
    print("✅ Health API Integrator ready for healthcare operations!")
    
    return "Health API Integrator initialized successfully"

if __name__ == "__main__":
    main()
