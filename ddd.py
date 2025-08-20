# Enhanced Health API Integrator with reliable connectivity and graph generation support
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
    """Enhanced API integrator with reliable healthcare connectivity and graph generation support"""

    def __init__(self, config):
        self.config = config
        logger.info("🔬 Enhanced HealthAPIIntegrator initialized with graph generation support")
        logger.info(f"⚡ Enhanced timeout: {self.config.timeout}s")
        logger.info(f"🌐 Snowflake API: {self.config.api_url}")
        logger.info(f"📡 Backend API: {self.config.fastapi_url}")
        logger.info(f"❤️ Heart Attack ML API: {self.config.heart_attack_api_url}")
        logger.info(f"📊 Graph generation: Ready for matplotlib code generation")

    def call_llm_enhanced(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Enhanced Snowflake Cortex API call with robust error handling"""
        try:
            session_id = str(uuid.uuid4()) + "_enhanced_healthcare"
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
                    "app_lvl_prefix": "enhanced_healthcare",
                    "user_id": "healthcare_enhanced",
                    "session_id": session_id
                }
            }

            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "Authorization": f'Snowflake Token="{self.config.api_key}"',
                "X-Healthcare-Analysis": "enhanced",
                "X-Clinical-Context": "comprehensive",
                "X-Graph-Generation": "enabled"
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
                    return f"I apologize, but I encountered a processing error while analyzing your healthcare data. Please try rephrasing your question."
            else:
                error_msg = f"Enhanced Healthcare LLM API error {response.status_code}"
                logger.error(error_msg)
                return f"I'm experiencing connectivity issues with the healthcare analysis service. Please try again in a moment."

        except requests.exceptions.Timeout:
            error_msg = f"Enhanced Healthcare LLM timeout after {self.config.timeout}s"
            logger.error(error_msg)
            return "The healthcare analysis request is taking longer than expected. Please try again with a simpler question."
        except requests.exceptions.ConnectionError:
            error_msg = f"Enhanced Healthcare LLM connection failed"
            logger.error(error_msg)
            return "I'm having trouble connecting to the healthcare analysis service. Please check your connection and try again."
        except Exception as e:
            error_msg = f"Enhanced Healthcare LLM unexpected error: {str(e)}"
            logger.error(error_msg)
            return "I encountered an unexpected error during healthcare analysis. Please try rephrasing your question or contact support if the issue persists."

    def call_llm_isolated_enhanced(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Enhanced isolated LLM call for reliable code explanations and graph generation"""
        try:
            session_id = str(uuid.uuid4()) + "_enhanced_batch"
            sys_msg = system_message or """You are Dr. CodeAI, a medical coding and visualization expert. Provide clear, concise explanations of medical codes and healthcare terminology. Support graph generation requests with matplotlib code. Keep responses brief but informative."""

            logger.info(f"🔬 Enhanced batch healthcare call: {user_message[:100]}...")

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
                    "app_lvl_prefix": "enhanced_batch",
                    "user_id": "batch_enhanced",
                    "session_id": session_id
                }
            }

            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "Authorization": f'Snowflake Token="{self.config.api_key}"',
                "X-Healthcare-Batch": "enhanced",
                "X-Medical-Coding": "comprehensive",
                "X-Graph-Generation": "matplotlib"
            }

            response = requests.post(
                self.config.api_url,
                headers=headers,
                json=payload,
                verify=False,
                timeout=30  # Longer timeout for batch processing
            )

            if response.status_code == 200:
                try:
                    raw = response.text
                    if "end_of_stream" in raw:
                        answer, _, _ = raw.partition("end_of_stream")
                        bot_reply = answer.strip()
                    else:
                        bot_reply = raw.strip()

                    logger.info(f"✅ Enhanced batch healthcare SUCCESS: {len(bot_reply)} chars returned")
                    return bot_reply

                except Exception as e:
                    logger.warning(f"Enhanced batch parse error: {e}")
                    return "Brief explanation unavailable"
            else:
                logger.warning(f"❌ Enhanced batch API error {response.status_code}")
                return "Brief explanation unavailable"

        except Exception as e:
            logger.warning(f"❌ Enhanced batch call failed: {str(e)}")
            return "Brief explanation unavailable"

    def call_llm_for_graph_generation(self, user_message: str, chat_context: Dict[str, Any]) -> str:
        """Specialized LLM call for graph generation with matplotlib code"""
        try:
            session_id = str(uuid.uuid4()) + "_graph_generation"
            
            graph_system_msg = """You are Dr. GraphAI, a healthcare data visualization expert specializing in matplotlib code generation. 

CAPABILITIES:
- Generate working matplotlib code for healthcare data visualizations
- Create medical timeline charts, risk dashboards, medication distributions
- Use actual patient data when available from the provided context
- Provide complete, executable Python code with proper imports

RESPONSE FORMAT:
Always respond with:
1. Brief explanation of the visualization
2. Complete matplotlib code block
3. Any data insights or observations

IMPORTANT: Generate only working matplotlib code that can be executed directly."""

            # Prepare context summary for graph generation
            context_summary = self._prepare_graph_context_summary(chat_context)
            
            enhanced_prompt = f"""Healthcare Data Visualization Request:

USER REQUEST: {user_message}

AVAILABLE PATIENT DATA CONTEXT:
{context_summary}

Generate appropriate matplotlib code for this healthcare visualization request. Use the available patient data when possible, or create representative sample data if specific data is not available.

Provide:
1. Brief description of what the visualization shows
2. Complete executable matplotlib code
3. Any relevant healthcare insights"""

            logger.info(f"📊 Enhanced graph generation LLM call: {user_message[:50]}...")

            payload = {
                "query": {
                    "aplctn_cd": self.config.aplctn_cd,
                    "app_id": self.config.app_id,
                    "api_key": self.config.api_key,
                    "method": "cortex",
                    "model": self.config.model,
                    "sys_msg": graph_system_msg,
                    "limit_convs": "0",
                    "prompt": {
                        "messages": [
                            {
                                "role": "user",
                                "content": enhanced_prompt
                            }
                        ]
                    },
                    "app_lvl_prefix": "enhanced_graph",
                    "user_id": "graph_enhanced",
                    "session_id": session_id
                }
            }

            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "Authorization": f'Snowflake Token="{self.config.api_key}"',
                "X-Healthcare-Graph": "matplotlib",
                "X-Visualization": "enhanced"
            }

            response = requests.post(
                self.config.api_url,
                headers=headers,
                json=payload,
                verify=False,
                timeout=45  # Longer timeout for graph generation
            )

            if response.status_code == 200:
                try:
                    raw = response.text
                    if "end_of_stream" in raw:
                        answer, _, _ = raw.partition("end_of_stream")
                        bot_reply = answer.strip()
                    else:
                        bot_reply = raw.strip()

                    logger.info(f"✅ Enhanced graph generation SUCCESS: {len(bot_reply)} chars returned")
                    return bot_reply

                except Exception as e:
                    logger.error(f"Enhanced graph generation parse error: {e}")
                    return "Graph generation failed during response parsing. Please try a simpler visualization request."
            else:
                logger.error(f"❌ Enhanced graph generation API error {response.status_code}")
                return "Graph generation service temporarily unavailable. Please try again or request a different visualization."

        except Exception as e:
            logger.error(f"❌ Enhanced graph generation call failed: {str(e)}")
            return f"Graph generation encountered an error: {str(e)}. Please try a simpler visualization request."

    def _prepare_graph_context_summary(self, chat_context: Dict[str, Any]) -> str:
        """Prepare a summary of available data for graph generation"""
        try:
            summary_parts = []
            
            # Patient overview
            patient_overview = chat_context.get("patient_overview", {})
            if patient_overview:
                age = patient_overview.get("age", "unknown")
                risk_level = patient_overview.get("heart_attack_risk_level", "unknown")
                summary_parts.append(f"Patient: Age {age}, Risk Level: {risk_level}")
            
            # Medical data summary
            medical_extraction = chat_context.get("medical_extraction", {})
            if medical_extraction and medical_extraction.get("hlth_srvc_records"):
                medical_records = len(medical_extraction["hlth_srvc_records"])
                diagnosis_codes = medical_extraction.get("extraction_summary", {}).get("total_diagnosis_codes", 0)
                summary_parts.append(f"Medical: {medical_records} service records, {diagnosis_codes} diagnosis codes")
                
                # Add sample codes for visualization
                if medical_extraction.get("code_meanings", {}).get("diagnosis_code_meanings"):
                    sample_diagnoses = list(medical_extraction["code_meanings"]["diagnosis_code_meanings"].keys())[:3]
                    summary_parts.append(f"Sample diagnoses: {', '.join(sample_diagnoses)}")
            
            # Pharmacy data summary
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                pharmacy_records = len(pharmacy_extraction["ndc_records"])
                summary_parts.append(f"Pharmacy: {pharmacy_records} medication records")
                
                # Add sample medications for visualization
                if pharmacy_extraction.get("code_meanings", {}).get("medication_meanings"):
                    sample_meds = list(pharmacy_extraction["code_meanings"]["medication_meanings"].keys())[:3]
                    summary_parts.append(f"Sample medications: {', '.join(sample_meds)}")
            
            # Risk assessment data
            entity_extraction = chat_context.get("entity_extraction", {})
            if entity_extraction:
                diabetes = entity_extraction.get("diabetics", "unknown")
                bp = entity_extraction.get("blood_pressure", "unknown")
                smoking = entity_extraction.get("smoking", "unknown")
                summary_parts.append(f"Health factors: Diabetes={diabetes}, BP={bp}, Smoking={smoking}")
            
            # Heart attack prediction
            heart_prediction = chat_context.get("heart_attack_prediction", {})
            if heart_prediction and not heart_prediction.get("error"):
                risk_score = chat_context.get("heart_attack_risk_score", 0)
                summary_parts.append(f"Heart attack risk: {risk_score:.2f}")
            
            return "\n".join(summary_parts) if summary_parts else "Limited patient data available for visualization"
            
        except Exception as e:
            logger.warning(f"Error preparing graph context: {e}")
            return "Patient healthcare data available for visualization"

    def fetch_backend_data_enhanced(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced backend data fetch with comprehensive validation"""
        try:
            logger.info(f"🔬 Enhanced Healthcare Backend API call: {self.config.fastapi_url}/all")

            # Enhanced validation
            required_fields = {
                "first_name": "Patient identification",
                "last_name": "Patient identification", 
                "ssn": "Unique patient identifier",
                "date_of_birth": "Age verification",
                "gender": "Risk assessment",
                "zip_code": "Geographic analysis"
            }
            
            missing_fields = []
            validation_errors = []
            
            for field, purpose in required_fields.items():
                if not patient_data.get(field):
                    missing_fields.append(f"{field}: {purpose}")
                else:
                    # Basic validation
                    if field == "ssn" and len(str(patient_data[field]).replace("-", "")) != 9:
                        validation_errors.append(f"SSN format invalid")
                    elif field == "zip_code" and len(str(patient_data[field])) < 5:
                        validation_errors.append(f"ZIP code incomplete")
            
            if missing_fields or validation_errors:
                all_errors = missing_fields + validation_errors
                return {"error": f"Validation failed: {'; '.join(all_errors)}"}

            # Enhanced API call
            headers = {
                "Content-Type": "application/json",
                "X-Healthcare-Request": "enhanced",
                "X-Clinical-Analysis": "comprehensive",
                "X-Graph-Generation": "supported"
            }

            response = requests.post(
                f"{self.config.fastapi_url}/all",
                json=patient_data,
                headers=headers,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                api_data = response.json()

                # Enhanced result mapping
                result = {
                    "mcid_output": self._enhanced_process_response(api_data.get('mcid_search', {}), 'mcid'),
                    "medical_output": self._enhanced_process_response(api_data.get('medical_submit', {}), 'medical'),
                    "pharmacy_output": self._enhanced_process_response(api_data.get('pharmacy_submit', {}), 'pharmacy'),
                    "token_output": self._enhanced_process_response(api_data.get('get_token', {}), 'token')
                }

                logger.info("✅ Enhanced healthcare backend API successful")
                logger.info(f"🔬 Medical data: {'Available' if result['medical_output'].get('status_code') == 200 else 'Limited'}")
                logger.info(f"💊 Pharmacy data: {'Available' if result['pharmacy_output'].get('status_code') == 200 else 'Limited'}")
                logger.info(f"🆔 MCID data: {'Available' if result['mcid_output'].get('status_code') == 200 else 'Limited'}")
                logger.info(f"📊 Graph generation: Ready for healthcare visualizations")
                
                return result

            else:
                error_msg = f"Backend API failed {response.status_code}: {response.text[:200]}"
                logger.error(error_msg)
                return {"error": error_msg}

        except Exception as e:
            error_msg = f"Backend fetch error: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    def _enhanced_process_response(self, response_data: Dict[str, Any], service_name: str) -> Dict[str, Any]:
        """Enhanced API response processing with graph generation support"""
        if not response_data:
            return {
                "error": f"No {service_name} data available", 
                "service": service_name,
                "status": "unavailable",
                "graph_ready": False
            }

        # Enhanced error handling
        if "error" in response_data:
            return {
                "error": response_data["error"],
                "service": service_name,
                "status_code": response_data.get("status_code", 500),
                "status": "error",
                "graph_ready": False
            }

        # Enhanced successful response handling
        if response_data.get("status_code") == 200 and "body" in response_data:
            return {
                "status_code": 200,
                "body": response_data["body"],
                "service": service_name,
                "timestamp": response_data.get("timestamp", datetime.now().isoformat()),
                "status": "success",
                "graph_ready": True,
                "visualization_supported": True
            }

        # Enhanced other format handling
        return {
            "status_code": response_data.get("status_code", 200),
            "body": response_data,
            "service": service_name,
            "timestamp": datetime.now().isoformat(),
            "status": "partial",
            "graph_ready": False
        }

    def test_healthcare_llm_connection(self) -> Dict[str, Any]:
        """Test enhanced healthcare LLM connection with graph generation support"""
        try:
            logger.info("🔬 Testing enhanced healthcare LLM connection...")
            
            test_prompt = """You are performing a healthcare system test with graph generation capabilities.

Please respond with: "Enhanced Healthcare LLM connection ready for clinical analysis and matplotlib graph generation."

This tests basic connectivity, response formatting, and visualization capabilities."""

            test_response = self.call_llm_enhanced(test_prompt, self.config.sys_msg)

            # Enhanced validation
            if test_response and not test_response.startswith("Error") and not test_response.startswith("I apologize"):
                return {
                    "success": True,
                    "response": test_response[:200] + "...",
                    "endpoint": self.config.api_url,
                    "model": self.config.model,
                    "healthcare_ready": True,
                    "graph_generation_ready": True,
                    "enhanced_connection": True
                }
            else:
                return {
                    "success": False,
                    "error": test_response,
                    "endpoint": self.config.api_url,
                    "healthcare_ready": False,
                    "graph_generation_ready": False,
                    "enhanced_connection": False
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Enhanced Healthcare LLM test failed: {str(e)}",
                "endpoint": self.config.api_url,
                "healthcare_ready": False,
                "graph_generation_ready": False,
                "enhanced_connection": False
            }

    def test_graph_generation_capability(self) -> Dict[str, Any]:
        """Test graph generation capability"""
        try:
            logger.info("📊 Testing enhanced graph generation capability...")
            
            sample_context = {
                "patient_overview": {"age": "45", "heart_attack_risk_level": "medium"},
                "medical_extraction": {"hlth_srvc_records": [{"diagnosis_codes": [{"code": "I10"}]}]},
                "pharmacy_extraction": {"ndc_records": [{"lbl_nm": "Metformin"}]}
            }
            
            test_request = "Create a simple bar chart showing patient risk factors"
            
            test_response = self.call_llm_for_graph_generation(test_request, sample_context)
            
            if test_response and "matplotlib" in test_response.lower() and not test_response.startswith("Graph generation failed"):
                return {
                    "success": True,
                    "response": test_response[:300] + "...",
                    "graph_generation_ready": True,
                    "matplotlib_supported": True
                }
            else:
                return {
                    "success": False,
                    "error": test_response,
                    "graph_generation_ready": False,
                    "matplotlib_supported": False
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Graph generation test failed: {str(e)}",
                "graph_generation_ready": False,
                "matplotlib_supported": False
            }

    def test_backend_connection_enhanced(self) -> Dict[str, Any]:
        """Test enhanced backend server connection"""
        try:
            logger.info("🔬 Testing enhanced healthcare backend...")

            health_url = f"{self.config.fastapi_url}/health"
            
            headers = {
                "X-Healthcare-Test": "enhanced",
                "X-Clinical-Validation": "comprehensive",
                "X-Graph-Support": "matplotlib"
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
                    "enhanced_connection": True,
                    "healthcare_ready": True,
                    "graph_support": True
                }
            else:
                return {
                    "success": False,
                    "error": f"Backend health check failed: {response.status_code}",
                    "backend_url": self.config.fastapi_url,
                    "enhanced_connection": False,
                    "healthcare_ready": False,
                    "graph_support": False
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Backend test failed: {str(e)}",
                "backend_url": self.config.fastapi_url,
                "enhanced_connection": False,
                "healthcare_ready": False,
                "graph_support": False
            }

    async def test_ml_connection_enhanced(self) -> Dict[str, Any]:
        """Test enhanced ML API server connection"""
        try:
            logger.info(f"🔬 Testing enhanced ML API: {self.config.heart_attack_api_url}")

            health_url = f"{self.config.heart_attack_api_url}/health"
            timeout = aiohttp.ClientTimeout(total=15)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Enhanced health check
                async with session.get(health_url) as response:
                    if response.status == 200:
                        health_data = await response.json()

                        # Enhanced prediction test
                        test_features = {"age": 45, "gender": 0, "diabetes": 0, "high_bp": 0, "smoking": 0}
                        
                        predict_url = f"{self.config.heart_attack_api_url}/predict"
                        async with session.post(predict_url, json=test_features) as pred_response:
                            if pred_response.status == 200:
                                pred_data = await pred_response.json()
                                
                                return {
                                    "success": True,
                                    "health_check": health_data,
                                    "test_prediction": pred_data,
                                    "server_url": self.config.heart_attack_api_url,
                                    "enhanced_connection": True,
                                    "ml_ready": True,
                                    "prediction_ready": True
                                }
                            else:
                                error_text = await pred_response.text()
                                return {
                                    "success": False,
                                    "error": f"ML prediction test failed {pred_response.status}: {error_text[:200]}",
                                    "server_url": self.config.heart_attack_api_url,
                                    "enhanced_connection": False,
                                    "ml_ready": False,
                                    "prediction_ready": False
                                }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"ML health endpoint error {response.status}: {error_text[:200]}",
                            "server_url": self.config.heart_attack_api_url,
                            "enhanced_connection": False,
                            "ml_ready": False,
                            "prediction_ready": False
                        }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "ML API timeout - service may be down",
                "server_url": self.config.heart_attack_api_url,
                "enhanced_connection": False,
                "ml_ready": False,
                "prediction_ready": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"ML API test failed: {str(e)}",
                "server_url": self.config.heart_attack_api_url,
                "enhanced_connection": False,
                "ml_ready": False,
                "prediction_ready": False
            }

    def test_all_connections_enhanced(self) -> Dict[str, Any]:
        """Test all connections with enhanced reporting and graph generation support"""
        logger.info("🔬 Testing ALL enhanced connections...")

        results = {
            "llm_connection": self.test_healthcare_llm_connection(),
            "backend_connection": self.test_backend_connection_enhanced(),
            "graph_generation": self.test_graph_generation_capability()
        }

        # Test ML connection
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results["ml_connection"] = loop.run_until_complete(self.test_ml_connection_enhanced())
            loop.close()
        except Exception as e:
            results["ml_connection"] = {
                "success": False,
                "error": f"ML test failed: {str(e)}",
                "enhanced_connection": False,
                "ml_ready": False,
                "prediction_ready": False
            }

        # Enhanced summary
        all_success = all(result.get("success", False) for result in results.values())
        enhanced_connections = sum(1 for result in results.values() if result.get("enhanced_connection", False))
        healthcare_ready = all(result.get("healthcare_ready", False) or result.get("ml_ready", False) for result in results.values())
        graph_ready = results.get("graph_generation", {}).get("graph_generation_ready", False)
        
        results["enhanced_overall_status"] = {
            "all_connections_successful": all_success,
            "successful_connections": sum(1 for result in results.values() if result.get("success", False)),
            "total_connections": len(results),
            "enhanced_connections": enhanced_connections,
            "healthcare_ready": healthcare_ready,
            "graph_generation_ready": graph_ready,
            "matplotlib_supported": results.get("graph_generation", {}).get("matplotlib_supported", False),
            "enhancement_level": "high" if enhanced_connections >= 3 else "moderate" if enhanced_connections >= 2 else "low"
        }

        logger.info(f"🔬 Enhanced connection test complete: {results['enhanced_overall_status']['successful_connections']}/{results['enhanced_overall_status']['total_connections']} successful")
        logger.info(f"🏥 Healthcare ready: {healthcare_ready}")
        logger.info(f"📊 Graph generation ready: {graph_ready}")
        logger.info(f"📈 Enhancement level: {results['enhanced_overall_status']['enhancement_level']}")

        return results

    # Backward compatibility methods
    def call_llm_fast(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Backward compatibility - uses enhanced call"""
        return self.call_llm_enhanced(user_message, system_message)

    def call_llm_isolated(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Backward compatibility - uses enhanced isolated call"""
        return self.call_llm_isolated_enhanced(user_message, system_message)

    def call_llm(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Backward compatibility - uses enhanced call"""
        return self.call_llm_enhanced(user_message, system_message)

    def fetch_backend_data_fast(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced fetch"""
        return self.fetch_backend_data_enhanced(patient_data)

    def fetch_backend_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced fetch"""
        return self.fetch_backend_data_enhanced(patient_data)

    def test_llm_connection_enhanced(self) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced test"""
        return self.test_healthcare_llm_connection()

    def test_backend_connection(self) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced test"""
        return self.test_backend_connection_enhanced()

    def test_all_connections(self) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced test"""
        return self.test_all_connections_enhanced()
