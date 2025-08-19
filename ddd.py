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

class EnhancedHealthAPIIntegrator:
    """Enhanced API integrator with HEALTHCARE SPECIALIZATION and detailed clinical prompts"""

    def __init__(self, config):
        self.config = config
        logger.info("🔬 EnhancedHealthAPIIntegrator initialized with healthcare specialization")
        logger.info(f"⚡ Enhanced timeout: {self.config.timeout}s for detailed analysis")
        logger.info(f"🌐 Snowflake Clinical API: {self.config.api_url}")
        logger.info(f"📡 Enhanced Backend API: {self.config.fastapi_url}")
        logger.info(f"❤️ Enhanced Heart Attack ML API: {self.config.heart_attack_api_url}")
        logger.info(f"🔬 Healthcare specialization: Activated")

    def call_llm_enhanced(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Enhanced Snowflake Cortex API call with healthcare specialization"""
        try:
            session_id = str(uuid.uuid4()) + "_healthcare_enhanced"
            sys_msg = system_message or self.config.sys_msg

            logger.info(f"🔬 Enhanced Healthcare LLM call - {len(user_message)} chars")

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
                    "app_lvl_prefix": "edadip_healthcare_enhanced",
                    "user_id": "healthcare_specialist",
                    "session_id": session_id
                }
            }

            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "Authorization": f'Snowflake Token="{self.config.api_key}"',
                "X-Healthcare-Analysis": "enhanced",
                "X-Clinical-Context": "detailed"
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

                    logger.info("✅ Enhanced Healthcare LLM call successful")
                    return bot_reply

                except Exception as e:
                    error_msg = f"Enhanced Healthcare LLM parse error: {e}"
                    logger.error(error_msg)
                    return f"Parse Error: {error_msg}"
            else:
                error_msg = f"Enhanced Healthcare LLM API error {response.status_code}: {response.text[:200]}"
                logger.error(error_msg)
                return f"API Error {response.status_code}: {response.text[:500]}"

        except requests.exceptions.Timeout:
            error_msg = f"Enhanced Healthcare LLM timeout after {self.config.timeout}s"
            logger.error(error_msg)
            return f"Timeout Error: {error_msg}"
        except requests.exceptions.ConnectionError:
            error_msg = f"Enhanced Healthcare LLM connection failed: {self.config.api_url}"
            logger.error(error_msg)
            return f"Connection Error: {error_msg}"
        except Exception as e:
            error_msg = f"Enhanced Healthcare LLM unexpected error: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

    def call_llm_isolated_enhanced(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Enhanced isolated LLM call for DETAILED healthcare code explanations"""
        try:
            session_id = str(uuid.uuid4()) + "_batch_healthcare_enhanced"
            sys_msg = system_message or """You are Dr. CodeAI, a medical coding expert with comprehensive knowledge of:
• ICD-10 diagnosis codes and clinical terminology
• CPT procedure codes and healthcare services
• NDC medication codes and pharmaceutical therapeutics
• Clinical context and evidence-based medicine

Return detailed, clinically accurate JSON explanations with therapeutic context, contraindications, and clinical significance where appropriate."""

            logger.info(f"🔬 Enhanced BATCH Healthcare LLM: {user_message[:100]}...")

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
                    "app_lvl_prefix": "edadip_healthcare",
                    "user_id": "batch_healthcare_enhanced",
                    "session_id": session_id
                }
            }

            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "Authorization": f'Snowflake Token="{self.config.api_key}"',
                "X-Healthcare-Batch": "enhanced",
                "X-Medical-Coding": "detailed"
            }

            response = requests.post(
                self.config.api_url,
                headers=headers,
                json=payload,
                verify=False,
                timeout=30  # Longer timeout for detailed healthcare analysis
            )

            if response.status_code == 200:
                try:
                    raw = response.text
                    if "end_of_stream" in raw:
                        answer, _, _ = raw.partition("end_of_stream")
                        bot_reply = answer.strip()
                    else:
                        bot_reply = raw.strip()

                    logger.info(f"✅ Enhanced BATCH Healthcare SUCCESS: {len(bot_reply)} chars returned")
                    return bot_reply

                except Exception as e:
                    error_msg = f"Enhanced batch healthcare parse error: {e}"
                    logger.warning(error_msg)
                    return "Detailed explanation unavailable"
            else:
                logger.warning(f"❌ Enhanced BATCH Healthcare API error {response.status_code}")
                return "Detailed explanation unavailable"

        except Exception as e:
            logger.warning(f"❌ Enhanced BATCH Healthcare call failed: {str(e)}")
            return "Detailed explanation unavailable"

    def call_llm_fast(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Backward compatibility wrapper - uses enhanced call"""
        return self.call_llm_enhanced(user_message, system_message)

    def call_llm_isolated(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Backward compatibility wrapper - uses enhanced isolated call"""
        return self.call_llm_isolated_enhanced(user_message, system_message)

    def call_llm(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Backward compatibility wrapper - uses enhanced call"""
        return self.call_llm_enhanced(user_message, system_message)

    def fetch_backend_data_enhanced(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced backend data fetch with detailed healthcare validation"""
        try:
            logger.info(f"🔬 Enhanced Healthcare Backend API call: {self.config.fastapi_url}/all")

            # Enhanced validation with clinical context
            required_fields = {
                "first_name": "Patient identification for healthcare record matching",
                "last_name": "Patient identification for healthcare record matching", 
                "ssn": "Unique patient identifier for secure medical record access",
                "date_of_birth": "Patient age verification for age-specific medical analysis",
                "gender": "Gender-specific healthcare risk assessment and clinical guidelines",
                "zip_code": "Geographic health pattern analysis and provider network assessment"
            }
            
            missing_fields = []
            validation_errors = []
            
            for field, clinical_purpose in required_fields.items():
                if not patient_data.get(field):
                    missing_fields.append(f"{field}: {clinical_purpose}")
                else:
                    # Enhanced field validation
                    if field == "ssn" and len(str(patient_data[field]).replace("-", "")) != 9:
                        validation_errors.append(f"SSN format invalid - required for secure healthcare record access")
                    elif field == "zip_code" and len(str(patient_data[field])) < 5:
                        validation_errors.append(f"ZIP code incomplete - needed for geographic health analysis")
                    elif field == "gender" and str(patient_data[field]).upper() not in ["M", "F", "MALE", "FEMALE"]:
                        validation_errors.append(f"Gender specification unclear - required for clinical risk assessment")
            
            if missing_fields or validation_errors:
                all_errors = missing_fields + validation_errors
                return {"error": f"Enhanced healthcare validation failed: {'; '.join(all_errors)}"}

            # Enhanced API call with healthcare context
            headers = {
                "Content-Type": "application/json",
                "X-Healthcare-Request": "enhanced",
                "X-Clinical-Analysis": "comprehensive"
            }

            response = requests.post(
                f"{self.config.fastapi_url}/all",
                json=patient_data,
                headers=headers,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                api_data = response.json()

                # Enhanced result mapping with healthcare context
                result = {
                    "mcid_output": self._enhanced_process_response(api_data.get('mcid_search', {}), 'mcid'),
                    "medical_output": self._enhanced_process_response(api_data.get('medical_submit', {}), 'medical'),
                    "pharmacy_output": self._enhanced_process_response(api_data.get('pharmacy_submit', {}), 'pharmacy'),
                    "token_output": self._enhanced_process_response(api_data.get('get_token', {}), 'token')
                }

                logger.info("✅ Enhanced healthcare backend API data fetch successful")
                logger.info(f"🔬 Medical data: {'Available' if result['medical_output'].get('status_code') == 200 else 'Limited'}")
                logger.info(f"💊 Pharmacy data: {'Available' if result['pharmacy_output'].get('status_code') == 200 else 'Limited'}")
                logger.info(f"🆔 MCID data: {'Available' if result['mcid_output'].get('status_code') == 200 else 'Limited'}")
                
                return result

            else:
                error_msg = f"Enhanced healthcare backend API failed {response.status_code}: {response.text[:200]}"
                logger.error(error_msg)
                return {"error": error_msg}

        except Exception as e:
            error_msg = f"Enhanced healthcare backend fetch error: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    def fetch_backend_data_fast(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility wrapper - uses enhanced fetch"""
        return self.fetch_backend_data_enhanced(patient_data)

    def fetch_backend_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility wrapper - uses enhanced fetch"""
        return self.fetch_backend_data_enhanced(patient_data)

    def _enhanced_process_response(self, response_data: Dict[str, Any], service_name: str) -> Dict[str, Any]:
        """Enhanced API response processing with healthcare context"""
        if not response_data:
            return {
                "error": f"No {service_name} healthcare data available", 
                "service": service_name,
                "clinical_impact": self._get_clinical_impact(service_name)
            }

        # Enhanced error handling with clinical context
        if "error" in response_data:
            return {
                "error": response_data["error"],
                "service": service_name,
                "status_code": response_data.get("status_code", 500),
                "clinical_impact": self._get_clinical_impact(service_name)
            }

        # Enhanced successful response handling
        if response_data.get("status_code") == 200 and "body" in response_data:
            return {
                "status_code": 200,
                "body": response_data["body"],
                "service": service_name,
                "timestamp": response_data.get("timestamp", datetime.now().isoformat()),
                "clinical_impact": self._get_clinical_impact(service_name),
                "data_quality": "High - Complete healthcare record available"
            }

        # Enhanced other format handling
        return {
            "status_code": response_data.get("status_code", 200),
            "body": response_data,
            "service": service_name,
            "timestamp": datetime.now().isoformat(),
            "clinical_impact": self._get_clinical_impact(service_name),
            "data_quality": "Standard - Partial healthcare record available"
        }

    def _get_clinical_impact(self, service_name: str) -> str:
        """Get clinical impact description for each service"""
        clinical_impacts = {
            "medical": "Critical for diagnosis analysis, treatment history, and clinical decision support",
            "pharmacy": "Essential for medication management, therapeutic monitoring, and drug interaction analysis",
            "mcid": "Important for patient identity verification and care coordination across providers",
            "token": "Required for secure healthcare data access and API authentication"
        }
        return clinical_impacts.get(service_name, "Important for comprehensive healthcare analysis")

    async def test_ml_connection_enhanced(self) -> Dict[str, Any]:
        """Enhanced ML API server connection test with healthcare validation"""
        try:
            logger.info(f"🔬 Enhanced Healthcare ML API test: {self.config.heart_attack_api_url}")

            health_url = f"{self.config.heart_attack_api_url}/health"
            timeout = aiohttp.ClientTimeout(total=15)  # Enhanced timeout for detailed analysis

            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Enhanced health check with healthcare context
                async with session.get(health_url) as response:
                    if response.status == 200:
                        health_data = await response.json()

                        # Enhanced prediction test with clinical scenarios
                        clinical_test_scenarios = [
                            {
                                "name": "Low Risk Patient",
                                "features": {"age": 30, "gender": 0, "diabetes": 0, "high_bp": 0, "smoking": 0}
                            },
                            {
                                "name": "High Risk Patient", 
                                "features": {"age": 65, "gender": 1, "diabetes": 1, "high_bp": 1, "smoking": 1}
                            }
                        ]

                        test_results = []
                        
                        for scenario in clinical_test_scenarios:
                            predict_url = f"{self.config.heart_attack_api_url}/predict"
                            async with session.post(predict_url, json=scenario["features"]) as pred_response:
                                if pred_response.status == 200:
                                    pred_data = await pred_response.json()
                                    test_results.append({
                                        "scenario": scenario["name"],
                                        "features": scenario["features"],
                                        "prediction": pred_data,
                                        "status": "Success"
                                    })
                                else:
                                    error_text = await pred_response.text()
                                    test_results.append({
                                        "scenario": scenario["name"],
                                        "error": f"Prediction failed {pred_response.status}: {error_text[:200]}",
                                        "status": "Failed"
                                    })

                        # Determine overall ML service health
                        successful_tests = sum(1 for result in test_results if result.get("status") == "Success")
                        ml_service_health = "Excellent" if successful_tests == len(clinical_test_scenarios) else "Limited"

                        return {
                            "success": True,
                            "health_check": health_data,
                            "clinical_test_scenarios": test_results,
                            "ml_service_health": ml_service_health,
                            "server_url": self.config.heart_attack_api_url,
                            "connection_method": "enhanced_healthcare_validation",
                            "clinical_readiness": successful_tests == len(clinical_test_scenarios)
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"Enhanced ML health endpoint error {response.status}: {error_text[:200]}",
                            "server_url": self.config.heart_attack_api_url,
                            "clinical_readiness": False
                        }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Enhanced ML API timeout - cardiovascular prediction service may be down",
                "server_url": self.config.heart_attack_api_url,
                "clinical_readiness": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Enhanced ML API test failed: {str(e)}",
                "server_url": self.config.heart_attack_api_url,
                "clinical_readiness": False
            }

    async def call_ml_heart_attack_prediction_enhanced(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced ML API heart attack prediction with clinical validation"""
        try:
            logger.info(f"🔬 Enhanced cardiovascular risk prediction call...")
            logger.info(f"📊 Clinical features: {features}")

            endpoints = [
                f"{self.config.heart_attack_api_url}/predict",
                f"{self.config.heart_attack_api_url}/predict-simple"
            ]

            # Enhanced clinical parameter validation
            params = {
                "age": int(features.get("age", 50)),
                "gender": int(features.get("gender", 0)),
                "diabetes": int(features.get("diabetes", 0)),
                "high_bp": int(features.get("high_bp", 0)),
                "smoking": int(features.get("smoking", 0))
            }

            # Clinical validation of parameters
            clinical_warnings = []
            if params["age"] < 18 or params["age"] > 100:
                clinical_warnings.append("Age outside typical clinical range")
            if sum([params["diabetes"], params["high_bp"], params["smoking"]]) >= 3:
                clinical_warnings.append("Multiple high-risk factors present")

            logger.info(f"📤 Enhanced clinical parameters: {params}")
            if clinical_warnings:
                logger.info(f"⚠️ Clinical considerations: {clinical_warnings}")

            timeout = aiohttp.ClientTimeout(total=20)  # Enhanced timeout

            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Try enhanced primary endpoint
                try:
                    headers = {
                        "Content-Type": "application/json",
                        "X-Clinical-Prediction": "enhanced",
                        "X-Cardiovascular-Assessment": "comprehensive"
                    }
                    
                    async with session.post(endpoints[0], json=params, headers=headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            # Enhanced result validation
                            if "probability" in result and "prediction" in result:
                                logger.info(f"✅ Enhanced cardiovascular prediction successful: {result}")
                                return {
                                    "success": True,
                                    "prediction_data": result,
                                    "method": "ENHANCED_CLINICAL_POST_JSON",
                                    "endpoint": endpoints[0],
                                    "clinical_warnings": clinical_warnings,
                                    "cardiovascular_assessment": "comprehensive"
                                }
                            else:
                                logger.warning(f"Enhanced prediction response incomplete: {result}")
                        else:
                            logger.warning(f"Enhanced primary endpoint failed: {response.status}")
                except Exception as e:
                    logger.warning(f"Enhanced primary method failed: {str(e)}")

                # Try enhanced fallback endpoint
                try:
                    async with session.post(endpoints[1], params=params) as response:
                        if response.status == 200:
                            result = await response.json()
                            logger.info(f"✅ Enhanced cardiovascular prediction fallback successful: {result}")
                            return {
                                "success": True,
                                "prediction_data": result,
                                "method": "ENHANCED_CLINICAL_POST_PARAMS",
                                "endpoint": endpoints[1],
                                "clinical_warnings": clinical_warnings,
                                "cardiovascular_assessment": "standard"
                            }
                        else:
                            error_text = await response.text()
                            logger.error(f"❌ All enhanced cardiovascular endpoints failed. Status {response.status}: {error_text[:200]}")
                            return {
                                "success": False,
                                "error": f"Enhanced cardiovascular prediction server error {response.status}: {error_text[:200]}",
                                "tried_endpoints": endpoints,
                                "clinical_warnings": clinical_warnings
                            }
                except Exception as e:
                    logger.error(f"Enhanced fallback method failed: {str(e)}")
                    return {
                        "success": False,
                        "error": f"All enhanced cardiovascular prediction methods failed: {str(e)}",
                        "tried_endpoints": endpoints,
                        "clinical_warnings": clinical_warnings
                    }

        except asyncio.TimeoutError:
            logger.error("❌ Enhanced cardiovascular ML API timeout")
            return {
                "success": False,
                "error": "Enhanced cardiovascular ML API timeout - check prediction service status",
                "clinical_impact": "Unable to assess cardiovascular risk - manual clinical assessment recommended"
            }
        except Exception as e:
            logger.error(f"Enhanced cardiovascular ML API error: {e}")
            return {
                "success": False,
                "error": f"Enhanced cardiovascular ML API failed: {str(e)}",
                "clinical_impact": "Cardiovascular risk assessment unavailable"
            }

    def test_healthcare_llm_connection(self) -> Dict[str, Any]:
        """Test healthcare-specialized LLM connection"""
        try:
            logger.info("🔬 Testing Enhanced Healthcare LLM connection...")
            
            healthcare_test_prompt = """You are Dr. TestAI performing a healthcare system validation.

CLINICAL SCENARIO: A 45-year-old patient with ICD-10 code I10 (Essential hypertension) and NDC code 0002-0145-02 (Lisinopril).

Please provide:
1. Clinical significance of I10
2. Therapeutic purpose of the NDC code
3. Clinical correlation between the diagnosis and medication

Respond in structured format demonstrating healthcare knowledge."""

            test_response = self.call_llm_enhanced(healthcare_test_prompt, self.config.sys_msg)

            # Enhanced validation of healthcare response
            healthcare_indicators = [
                "hypertension", "blood pressure", "lisinopril", "ace inhibitor", 
                "cardiovascular", "clinical", "therapeutic", "medication"
            ]
            
            healthcare_score = sum(1 for indicator in healthcare_indicators if indicator.lower() in test_response.lower())
            
            if test_response.startswith("Error"):
                return {
                    "success": False,
                    "error": test_response,
                    "endpoint": self.config.api_url,
                    "healthcare_specialization": False
                }
            elif healthcare_score >= 4:
                return {
                    "success": True,
                    "response": test_response[:300] + "...",
                    "endpoint": self.config.api_url,
                    "model": self.config.model,
                    "healthcare_specialization": True,
                    "healthcare_knowledge_score": f"{healthcare_score}/8",
                    "clinical_readiness": True
                }
            else:
                return {
                    "success": True,
                    "response": test_response[:300] + "...",
                    "endpoint": self.config.api_url,
                    "model": self.config.model,
                    "healthcare_specialization": False,
                    "healthcare_knowledge_score": f"{healthcare_score}/8",
                    "clinical_readiness": False,
                    "warning": "Limited healthcare knowledge detected"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Enhanced Healthcare LLM test failed: {str(e)}",
                "endpoint": self.config.api_url,
                "healthcare_specialization": False
            }

    def test_llm_connection_enhanced(self) -> Dict[str, Any]:
        """Enhanced Snowflake Cortex API connection test with healthcare context"""
        try:
            logger.info("🔬 Enhanced Healthcare LLM connection test...")
            test_response = self.call_llm_enhanced("Hello Dr. AI, please confirm your healthcare analysis capabilities are operational.")

            if test_response.startswith("Error"):
                return {
                    "success": False,
                    "error": test_response,
                    "endpoint": self.config.api_url,
                    "healthcare_enhanced": False
                }
            else:
                return {
                    "success": True,
                    "response": test_response,
                    "endpoint": self.config.api_url,
                    "model": self.config.model,
                    "healthcare_enhanced": True,
                    "connection_optimized": True
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Enhanced Healthcare LLM test failed: {str(e)}",
                "endpoint": self.config.api_url,
                "healthcare_enhanced": False
            }

    def test_isolated_llm_connection_enhanced(self) -> Dict[str, Any]:
        """Test enhanced isolated LLM connection for DETAILED healthcare batch processing"""
        try:
            logger.info("🔬 Testing Enhanced BATCH Healthcare LLM connection...")
            
            test_prompt = """Provide detailed clinical explanations for these medical codes in JSON format:
            
Medical Codes:
- I10: ICD-10 diagnosis code
- 99213: CPT procedure code  
- 0002-0145-02: NDC medication code

Return JSON: {"I10": "detailed clinical explanation", "99213": "detailed procedure explanation", "0002-0145-02": "detailed medication explanation"}"""

            test_response = self.call_llm_isolated_enhanced(test_prompt)

            if test_response and test_response != "Detailed explanation unavailable":
                # Enhanced validation for healthcare content
                try:
                    # Try to parse as JSON to validate structure
                    test_json = json.loads(test_response.strip())
                    
                    # Check for healthcare-specific content
                    healthcare_terms = ["hypertension", "clinical", "medication", "diagnosis", "therapeutic", "treatment"]
                    healthcare_content_score = sum(1 for term in healthcare_terms if term.lower() in test_response.lower())
                    
                    return {
                        "success": True,
                        "message": "Enhanced BATCH Healthcare LLM connection successful",
                        "test_prompt": test_prompt[:100] + "...",
                        "response": test_response[:300] + "...",
                        "endpoint": self.config.api_url,
                        "model": self.config.model,
                        "enhanced_batch_processing_enabled": True,
                        "healthcare_content_score": f"{healthcare_content_score}/6",
                        "json_structure_valid": True,
                        "clinical_analysis_ready": healthcare_content_score >= 3
                    }
                except json.JSONDecodeError:
                    return {
                        "success": True,
                        "message": "Enhanced BATCH Healthcare LLM partially functional",
                        "response": test_response[:200] + "...",
                        "endpoint": self.config.api_url,
                        "model": self.config.model,
                        "enhanced_batch_processing_enabled": True,
                        "json_structure_valid": False,
                        "warning": "JSON structure needs improvement"
                    }
            else:
                return {
                    "success": False,
                    "error": f"Enhanced BATCH Healthcare LLM returned: {test_response}",
                    "endpoint": self.config.api_url,
                    "model": self.config.model,
                    "enhanced_batch_processing_enabled": False
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Enhanced BATCH Healthcare LLM test failed: {str(e)}",
                "endpoint": self.config.api_url,
                "enhanced_batch_processing_enabled": False
            }

    def test_backend_connection_enhanced(self) -> Dict[str, Any]:
        """Enhanced backend server connection test with healthcare validation"""
        try:
            logger.info("🔬 Enhanced Healthcare Backend API test...")

            health_url = f"{self.config.fastapi_url}/health"
            
            headers = {
                "X-Healthcare-Test": "enhanced",
                "X-Clinical-Validation": "comprehensive"
            }
            
            response = requests.get(health_url, headers=headers, timeout=10)

            if response.status_code == 200:
                health_data = response.json()
                
                # Enhanced health check validation
                service_status = "Excellent" if "status" in health_data and health_data["status"] == "healthy" else "Limited"
                
                return {
                    "success": True,
                    "health_data": health_data,
                    "backend_url": self.config.fastapi_url,
                    "service_status": service_status,
                    "healthcare_enhanced": True,
                    "connection_optimized": True,
                    "clinical_data_access": True
                }
            else:
                return {
                    "success": False,
                    "error": f"Enhanced healthcare backend health check failed: {response.status_code}",
                    "backend_url": self.config.fastapi_url,
                    "clinical_data_access": False
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Enhanced healthcare backend test failed: {str(e)}",
                "backend_url": self.config.fastapi_url,
                "clinical_data_access": False
            }

    def test_all_connections_enhanced(self) -> Dict[str, Any]:
        """Enhanced test of all API connections with healthcare specialization"""
        logger.info("🔬 Enhanced testing ALL connections with healthcare specialization...")

        results = {
            "llm_connection": self.test_llm_connection_enhanced(),
            "healthcare_llm_connection": self.test_healthcare_llm_connection(),
            "isolated_llm_connection": self.test_isolated_llm_connection_enhanced(),
            "backend_connection": self.test_backend_connection_enhanced()
        }

        # Enhanced ML connection test
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results["ml_connection"] = loop.run_until_complete(self.test_ml_connection_enhanced())
            loop.close()
        except Exception as e:
            results["ml_connection"] = {
                "success": False,
                "error": f"Enhanced cardiovascular ML test failed: {str(e)}",
                "clinical_readiness": False
            }

        # Enhanced comprehensive summary
        all_success = all(result.get("success", False) for result in results.values())
        healthcare_ready = (
            results["healthcare_llm_connection"].get("success", False) and
            results["isolated_llm_connection"].get("success", False) and
            results["ml_connection"].get("success", False)
        )
        clinical_analysis_ready = (
            results["healthcare_llm_connection"].get("clinical_readiness", False) and
            results["isolated_llm_connection"].get("clinical_analysis_ready", False) and
            results["ml_connection"].get("clinical_readiness", False)
        )
        
        results["enhanced_overall_status"] = {
            "all_connections_successful": all_success,
            "successful_connections": sum(1 for result in results.values() if result.get("success", False)),
            "total_connections": len(results) - 1,
            "healthcare_specialization_ready": healthcare_ready,
            "clinical_analysis_ready": clinical_analysis_ready,
            "batch_processing_ready": results["isolated_llm_connection"].get("success", False),
            "cardiovascular_prediction_ready": results["ml_connection"].get("clinical_readiness", False),
            "enhancement_level": "healthcare_specialized_enhanced"
        }

        logger.info(f"🔬 Enhanced healthcare connection test complete: {results['enhanced_overall_status']['successful_connections']}/{results['enhanced_overall_status']['total_connections']} successful")
        logger.info(f"🏥 Healthcare specialization ready: {healthcare_ready}")
        logger.info(f"🔬 Clinical analysis ready: {clinical_analysis_ready}")

        return results

    # Backward compatibility methods
    def test_llm_connection_fast(self) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced test"""
        return self.test_llm_connection_enhanced()

    def test_llm_connection(self) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced test"""
        return self.test_llm_connection_enhanced()

    def test_backend_connection_fast(self) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced test"""
        return self.test_backend_connection_enhanced()

    def test_backend_connection(self) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced test"""
        return self.test_backend_connection_enhanced()

    async def test_ml_connection_fast(self) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced test"""
        return await self.test_ml_connection_enhanced()

    async def test_ml_connection(self) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced test"""
        return await self.test_ml_connection_enhanced()

    def test_all_connections_fast(self) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced test"""
        return self.test_all_connections_enhanced()

    def test_all_connections(self) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced test"""
        return self.test_all_connections_enhanced()
