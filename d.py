import json
import re
import requests
import urllib3
import uuid
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Dict, Any, List, TypedDict, Literal, Optional
from dataclasses import dataclass, asdict
import logging
import warnings
import asyncio
import aiohttp

# LangGraph imports - these are required
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Disable SSL warnings and ML warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced Configuration with FastAPI Heart Attack Prediction
@dataclass
class Config:
    fastapi_url: str = "http://localhost:8001"
    # Snowflake Cortex API Configuration
    api_url: str = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
    api_key: str = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
    app_id: str = "edadip"
    aplctn_cd: str = "edagnai"
    model: str = "llama3.1-70b"
    sys_msg: str = "You are a healthcare AI assistant. Provide accurate, concise answers based on context."
    chatbot_sys_msg: str = "You are a powerful healthcare AI assistant with access to deidentified medical records and heart attack risk predictions. Provide accurate, detailed analysis based on the medical and pharmacy data provided. Always maintain patient privacy and provide professional medical insights."
    max_retries: int = 3
    timeout: int = 30
    
    # UPDATED: FastAPI Heart Attack Prediction Configuration
    heart_attack_api_url: str = "http://localhost:8002"
    heart_attack_threshold: float = 0.5
    
    def to_dict(self):
        return asdict(self)

# Enhanced State Definition for LangGraph (8 nodes including heart attack prediction)
class HealthAnalysisState(TypedDict):
    # Input data
    patient_data: Dict[str, Any]
    
    # API outputs
    mcid_output: Dict[str, Any]
    medical_output: Dict[str, Any]
    pharmacy_output: Dict[str, Any]
    token_output: Dict[str, Any]
    
    # Processed data
    deidentified_medical: Dict[str, Any]
    deidentified_pharmacy: Dict[str, Any]
    
    # Extracted structured data
    medical_extraction: Dict[str, Any]
    pharmacy_extraction: Dict[str, Any]
    
    entity_extraction: Dict[str, Any]
    
    # Analysis results
    health_trajectory: str
    final_summary: str
    
    # UPDATED: Heart Attack Prediction via FastAPI
    heart_attack_prediction: Dict[str, Any]
    heart_attack_risk_score: float
    heart_attack_features: Dict[str, Any]
    
    # Chatbot functionality
    chatbot_ready: bool
    chatbot_context: Dict[str, Any]
    chat_history: List[Dict[str, str]]
    
    # Control flow
    current_step: str
    errors: List[str]
    retry_count: int
    processing_complete: bool
    step_status: Dict[str, str]

class HealthAnalysisAgent:
    def __init__(self, custom_config: Optional[Config] = None):
        # Use provided config or create default
        if custom_config:
            self.config = custom_config
        else:
            self.config = Config()
        
        logger.info("🔧 HealthAnalysisAgent initialized with Snowflake Cortex API + Interactive Chatbot + FastAPI Heart Attack Prediction")
        logger.info(f"🌐 API URL: {self.config.api_url}")
        logger.info(f"🤖 Model: {self.config.model}")
        logger.info(f"🔑 App ID: {self.config.app_id}")
        logger.info(f"💬 Chatbot: Enhanced Interactive Mode")
        logger.info(f"❤️ Heart Attack Prediction: FastAPI Server Enabled")
        logger.info(f"🔗 FastAPI Server: {self.config.heart_attack_api_url}")
        
        self.setup_langgraph()
    
    def setup_langgraph(self):
        """Setup LangGraph workflow - 8 node enhanced workflow with heart attack prediction"""
        logger.info("🔧 Setting up Enhanced LangGraph workflow with 8 nodes (including FastAPI heart attack prediction)...")
        
        # Create the StateGraph
        workflow = StateGraph(HealthAnalysisState)
        
        # Add all 8 processing nodes
        workflow.add_node("fetch_api_data", self.fetch_api_data)
        workflow.add_node("deidentify_data", self.deidentify_data)
        workflow.add_node("extract_medical_pharmacy_data", self.extract_medical_pharmacy_data)
        workflow.add_node("extract_entities", self.extract_entities)
        workflow.add_node("analyze_trajectory", self.analyze_trajectory)
        workflow.add_node("generate_summary", self.generate_summary)
        workflow.add_node("predict_heart_attack", self.predict_heart_attack)  # UPDATED: FastAPI prediction
        workflow.add_node("initialize_chatbot", self.initialize_chatbot)
        workflow.add_node("handle_error", self.handle_error)
        
        # Define the enhanced workflow edges (8 nodes)
        workflow.add_edge(START, "fetch_api_data")
        
        # Conditional edges with retry logic
        workflow.add_conditional_edges(
            "fetch_api_data",
            self.should_continue_after_api,
            {
                "continue": "deidentify_data",
                "retry": "fetch_api_data", 
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "deidentify_data",
            self.should_continue_after_deidentify,
            {
                "continue": "extract_medical_pharmacy_data",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "extract_medical_pharmacy_data",
            self.should_continue_after_extraction_step,
            {
                "continue": "extract_entities",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "extract_entities", 
            self.should_continue_after_entity_extraction,
            {
                "continue": "analyze_trajectory",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "analyze_trajectory",
            self.should_continue_after_trajectory,
            {
                "continue": "generate_summary",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_summary",
            self.should_continue_after_summary,
            {
                "continue": "predict_heart_attack",  # UPDATED: Go to FastAPI heart attack prediction
                "error": "handle_error"
            }
        )
        
        # UPDATED: FastAPI heart attack prediction node routing
        workflow.add_conditional_edges(
            "predict_heart_attack",
            self.should_continue_after_heart_attack_prediction,
            {
                "continue": "initialize_chatbot",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("initialize_chatbot", END)
        workflow.add_edge("handle_error", END)
        
        # Compile with checkpointer for persistence and reliability
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)
        
        logger.info("✅ Enhanced LangGraph workflow compiled successfully with 8 nodes including FastAPI heart attack prediction!")
    
    def call_llm(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Call Snowflake Cortex API with the user message"""
        try:
            session_id = str(uuid.uuid4())
            sys_msg = system_message or self.config.sys_msg
            
            logger.info(f"🤖 Calling Snowflake Cortex API: {self.config.api_url}")
            logger.info(f"🤖 Model: {self.config.model}")
            logger.info(f"🤖 Message length: {len(user_message)} characters")
            logger.info(f"🔑 Session ID: {session_id}")
            
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
                    error_msg = f"Error parsing Snowflake response: {e}. Raw response: {response.text[:500]}"
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
    
    # ===== FASTAPI CONNECTION METHODS (FIXED) =====
    
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
    
    # ===== LANGGRAPH NODES (8 NODES INCLUDING HEART ATTACK PREDICTION) =====
    
    def fetch_api_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """LangGraph Node 1: Fetch data from MCID, Medical, and Pharmacy APIs"""
        logger.info("🚀 LangGraph Node 1: Starting API data fetch...")
        state["current_step"] = "fetch_api_data"
        state["step_status"]["fetch_api_data"] = "running"
        
        try:
            patient_data = state["patient_data"]
            
            # Validate patient data
            required_fields = ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"]
            for field in required_fields:
                if not patient_data.get(field):
                    state["errors"].append(f"Missing required field: {field}")
                    state["step_status"]["fetch_api_data"] = "error"
                    return state
            
            logger.info(f"📡 Calling FastAPI: {self.config.fastapi_url}/all")
            
            response = requests.post(
                f"{self.config.fastapi_url}/all", 
                json=patient_data, 
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                api_data = response.json()
                
                state["mcid_output"] = api_data.get('mcid_search', {})
                state["medical_output"] = api_data.get('medical_submit', {})
                state["pharmacy_output"] = api_data.get('pharmacy_submit', {})
                state["token_output"] = api_data.get('get_token', {})
                
                state["step_status"]["fetch_api_data"] = "completed"
                logger.info("✅ Successfully fetched all API data")
                
            else:
                error_msg = f"API call failed with status {response.status_code}: {response.text}"
                state["errors"].append(error_msg)
                state["step_status"]["fetch_api_data"] = "error"
                logger.error(error_msg)
                
        except Exception as e:
            error_msg = f"Error fetching API data: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["fetch_api_data"] = "error"
            logger.error(error_msg)
        
        return state
    
    def deidentify_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """LangGraph Node 2: Deidentify medical and pharmacy data"""
        logger.info("🔒 LangGraph Node 2: Starting data deidentification...")
        state["current_step"] = "deidentify_data"
        state["step_status"]["deidentify_data"] = "running"
        
        try:
            medical_data = state.get("medical_output", {})
            deidentified_medical = self._deidentify_medical_data(medical_data, state["patient_data"])
            state["deidentified_medical"] = deidentified_medical
            
            pharmacy_data = state.get("pharmacy_output", {})
            deidentified_pharmacy = self._deidentify_pharmacy_data(pharmacy_data)
            state["deidentified_pharmacy"] = deidentified_pharmacy
            
            state["step_status"]["deidentify_data"] = "completed"
            logger.info("✅ Successfully deidentified medical and pharmacy data")
            
        except Exception as e:
            error_msg = f"Error deidentifying data: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["deidentify_data"] = "error"
            logger.error(error_msg)
        
        return state
    
    def extract_medical_pharmacy_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """LangGraph Node 3: Extract specific fields from deidentified medical and pharmacy data"""
        logger.info("🔍 LangGraph Node 3: Starting medical and pharmacy data extraction...")
        state["current_step"] = "extract_medical_pharmacy_data"
        state["step_status"]["extract_medical_pharmacy_data"] = "running"
        
        try:
            medical_extraction = self._extract_medical_fields(state.get("deidentified_medical", {}))
            state["medical_extraction"] = medical_extraction
            logger.info(f"📋 Medical extraction completed: {len(medical_extraction.get('hlth_srvc_records', []))} health service records found")
            
            pharmacy_extraction = self._extract_pharmacy_fields(state.get("deidentified_pharmacy", {}))
            state["pharmacy_extraction"] = pharmacy_extraction
            logger.info(f"💊 Pharmacy extraction completed: {len(pharmacy_extraction.get('ndc_records', []))} NDC records found")
            
            state["step_status"]["extract_medical_pharmacy_data"] = "completed"
            logger.info("✅ Successfully extracted medical and pharmacy structured data")
            
        except Exception as e:
            error_msg = f"Error extracting medical/pharmacy data: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_medical_pharmacy_data"] = "error"
            logger.error(error_msg)
        
        return state
    
    def extract_entities(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """LangGraph Node 4: Extract health entities using both pharmacy data and new extractions"""
        logger.info("🎯 LangGraph Node 4: Starting enhanced entity extraction...")
        state["current_step"] = "extract_entities"
        state["step_status"]["extract_entities"] = "running"
        
        try:
            pharmacy_data = state.get("pharmacy_output", {})
            pharmacy_extraction = state.get("pharmacy_extraction", {})
            medical_extraction = state.get("medical_extraction", {})
            
            entities = self._extract_health_entities_enhanced(
                pharmacy_data, pharmacy_extraction, medical_extraction
            )
            state["entity_extraction"] = entities
            
            state["step_status"]["extract_entities"] = "completed"
            logger.info("✅ Successfully extracted enhanced health entities")
            logger.info(f"🔍 Entities found: {entities}")
            
        except Exception as e:
            error_msg = f"Error extracting entities: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_entities"] = "error"
            logger.error(error_msg)
        
        return state
    
    def analyze_trajectory(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """LangGraph Node 5: Analyze health trajectory using Snowflake Cortex with enhanced data"""
        logger.info("📈 LangGraph Node 5: Starting enhanced health trajectory analysis...")
        state["current_step"] = "analyze_trajectory"
        state["step_status"]["analyze_trajectory"] = "running"
        
        try:
            deidentified_medical = state.get("deidentified_medical", {})
            deidentified_pharmacy = state.get("deidentified_pharmacy", {})
            medical_extraction = state.get("medical_extraction", {})
            pharmacy_extraction = state.get("pharmacy_extraction", {})
            entities = state.get("entity_extraction", {})
            
            trajectory_prompt = self._create_enhanced_trajectory_prompt(
                deidentified_medical, deidentified_pharmacy, 
                medical_extraction, pharmacy_extraction, entities
            )
            
            logger.info("🤖 Calling Snowflake Cortex for enhanced health trajectory analysis...")
            
            response = self.call_llm(trajectory_prompt)
            
            if response.startswith("Error"):
                state["errors"].append(f"Snowflake Cortex analysis failed: {response}")
                state["step_status"]["analyze_trajectory"] = "error"
            else:
                state["health_trajectory"] = response
                state["step_status"]["analyze_trajectory"] = "completed"
                logger.info("✅ Successfully analyzed enhanced health trajectory")
            
        except Exception as e:
            error_msg = f"Error analyzing trajectory: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["analyze_trajectory"] = "error"
            logger.error(error_msg)
        
        return state
    
    def generate_summary(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """LangGraph Node 6: Generate final health summary with enhanced data"""
        logger.info("📋 LangGraph Node 6: Generating enhanced final health summary...")
        state["current_step"] = "generate_summary"
        state["step_status"]["generate_summary"] = "running"
        
        try:
            summary_prompt = self._create_enhanced_summary_prompt(
                state.get("health_trajectory", ""), 
                state.get("entity_extraction", {}),
                state.get("medical_extraction", {}),
                state.get("pharmacy_extraction", {})
            )
            
            logger.info("🤖 Calling Snowflake Cortex for enhanced final summary generation...")
            
            response = self.call_llm(summary_prompt)
            
            if response.startswith("Error"):
                state["errors"].append(f"Summary generation failed: {response}")
                state["step_status"]["generate_summary"] = "error"
            else:
                state["final_summary"] = response
                state["step_status"]["generate_summary"] = "completed"
                logger.info("✅ Successfully generated enhanced final summary")
            
        except Exception as e:
            error_msg = f"Error generating summary: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["generate_summary"] = "error"
            logger.error(error_msg)
        
        return state
    
    def predict_heart_attack(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """LangGraph Node 8: Predict heart attack risk using FastAPI server - SIMPLIFIED OUTPUT"""
        logger.info("❤️ LangGraph Node 8: Starting heart attack prediction with FastAPI server...")
        state["current_step"] = "predict_heart_attack"
        state["step_status"]["predict_heart_attack"] = "running"
        
        try:
            # Extract features from health data for FastAPI model
            features = self._extract_heart_attack_features_for_fastapi(state)
            state["heart_attack_features"] = features
            
            if not features or "error" in features:
                state["errors"].append("Failed to extract features for FastAPI heart attack prediction")
                state["step_status"]["predict_heart_attack"] = "error"
                return state
            
            # Prepare feature vector for FastAPI call
            fastapi_features = self._prepare_fastapi_features(features)
            
            if fastapi_features is None:
                state["errors"].append("Failed to prepare feature vector for FastAPI prediction")
                state["step_status"]["predict_heart_attack"] = "error"
                return state
            
            # Make async prediction using FastAPI server
            try:
                # Run async call in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                prediction_result = loop.run_until_complete(
                    self._call_fastapi_heart_attack_prediction(fastapi_features)
                )
                loop.close()
            except Exception as async_error:
                logger.error(f"Async prediction call failed: {async_error}")
                state["errors"].append(f"FastAPI prediction call failed: {str(async_error)}")
                state["step_status"]["predict_heart_attack"] = "error"
                return state
            
            if prediction_result.get("success", False):
                # Process successful FastAPI prediction - SIMPLIFIED
                prediction_data = prediction_result.get("prediction_data", {})
                
                # Extract key values from FastAPI response
                risk_probability = prediction_data.get("probability", 0.0)
                binary_prediction = prediction_data.get("prediction", 0)
                
                # Convert to percentage
                risk_percentage = risk_probability * 100
                confidence_percentage = (1 - risk_probability) * 100 if binary_prediction == 0 else risk_probability * 100
                
                # Determine risk level for display
                if risk_percentage >= 70:
                    risk_category = "high risk"
                elif risk_percentage >= 50:
                    risk_category = "medium risk"
                else:
                    risk_category = "low risk"
                
                # Create SIMPLIFIED prediction result as requested
                simplified_prediction = {
                    "risk_display": f"Heart Disease Risk: {risk_percentage:.1f}%({risk_category})",
                    "confidence_display": f"Confidence: {confidence_percentage:.1f}%",
                    "combined_display": f"Heart Disease Risk: {risk_percentage:.1f}%({risk_category}) and Confidence: {confidence_percentage:.1f}%",
                    "raw_risk_score": risk_probability,
                    "raw_prediction": binary_prediction,
                    "fastapi_server_url": self.config.heart_attack_api_url,
                    "prediction_timestamp": datetime.now().isoformat()
                }
                
                state["heart_attack_prediction"] = simplified_prediction
                state["heart_attack_risk_score"] = float(risk_probability)
                
                logger.info(f"✅ FastAPI heart attack prediction completed successfully")
                logger.info(f"❤️ Display: {simplified_prediction['combined_display']}")
                
            else:
                # Handle FastAPI prediction failure
                error_msg = prediction_result.get("error", "Unknown FastAPI error")
                state["heart_attack_prediction"] = {
                    "error": error_msg,
                    "risk_display": "Heart Disease Risk: Error",
                    "confidence_display": "Confidence: Error",
                    "combined_display": f"Heart Disease Risk: Error - {error_msg}",
                    "fastapi_server_url": self.config.heart_attack_api_url,
                    "error_details": error_msg
                }
                state["heart_attack_risk_score"] = 0.0
                logger.warning(f"⚠️ FastAPI heart attack prediction failed: {error_msg}")
            
            state["step_status"]["predict_heart_attack"] = "completed"
            
        except Exception as e:
            error_msg = f"Error in FastAPI heart attack prediction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["predict_heart_attack"] = "error"
            logger.error(error_msg)
        
        return state
    
    def initialize_chatbot(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """LangGraph Node 9: Initialize interactive chatbot with all context including heart attack prediction"""
        logger.info("💬 LangGraph Node 9: Initializing interactive chatbot with FastAPI heart attack prediction...")
        state["current_step"] = "initialize_chatbot"
        state["step_status"]["initialize_chatbot"] = "running"
        
        try:
            # Prepare chatbot context with all data including heart attack prediction
            chatbot_context = {
                "deidentified_medical": state.get("deidentified_medical", {}),
                "deidentified_pharmacy": state.get("deidentified_pharmacy", {}),
                "medical_extraction": state.get("medical_extraction", {}),
                "pharmacy_extraction": state.get("pharmacy_extraction", {}),
                "entity_extraction": state.get("entity_extraction", {}),
                "health_trajectory": state.get("health_trajectory", ""),
                "final_summary": state.get("final_summary", ""),
                # UPDATED: FastAPI heart attack prediction context
                "heart_attack_prediction": state.get("heart_attack_prediction", {}),
                "heart_attack_risk_score": state.get("heart_attack_risk_score", 0.0),
                "heart_attack_features": state.get("heart_attack_features", {}),
                "patient_overview": {
                    "age": state.get("deidentified_medical", {}).get("src_mbr_age", "unknown"),
                    "zip": state.get("deidentified_medical", {}).get("src_mbr_zip_cd", "unknown"),
                    "analysis_timestamp": datetime.now().isoformat(),
                    "heart_attack_risk_level": state.get("heart_attack_prediction", {}).get("risk_level", "unknown"),
                    "model_type": "fastapi_server"
                }
            }
            
            state["chat_history"] = []
            state["chatbot_context"] = chatbot_context
            state["chatbot_ready"] = True
            state["processing_complete"] = True
            state["step_status"]["initialize_chatbot"] = "completed"
            
            logger.info("✅ Successfully initialized interactive chatbot with FastAPI heart attack prediction context")
            logger.info(f"💬 Chatbot ready with {len(chatbot_context)} data components including FastAPI heart attack prediction")
            
        except Exception as e:
            error_msg = f"Error initializing chatbot: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["initialize_chatbot"] = "error"
            logger.error(error_msg)
        
        return state
    
    def handle_error(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """LangGraph Node: Error handling"""
        logger.error(f"🚨 LangGraph Error Handler: {state['current_step']}")
        logger.error(f"Errors: {state['errors']}")
        
        state["processing_complete"] = True
        current_step = state.get("current_step", "unknown")
        state["step_status"][current_step] = "error"
        return state
    
    # ===== LANGGRAPH CONDITIONAL EDGES =====
    
    def should_continue_after_api(self, state: HealthAnalysisState) -> Literal["continue", "retry", "error"]:
        """LangGraph Conditional: Decide what to do after API fetch"""
        if state["errors"]:
            if state["retry_count"] < self.config.max_retries:
                state["retry_count"] += 1
                logger.warning(f"🔄 Retrying API fetch (attempt {state['retry_count']}/{self.config.max_retries})")
                state["errors"] = []
                return "retry"
            else:
                logger.error(f"❌ Max retries ({self.config.max_retries}) exceeded for API fetch")
                return "error"
        return "continue"
    
    def should_continue_after_deidentify(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        """LangGraph Conditional: Decide what to do after deidentification"""
        return "error" if state["errors"] else "continue"
    
    def should_continue_after_extraction_step(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        """LangGraph Conditional: Decide what to do after medical/pharmacy extraction"""
        return "error" if state["errors"] else "continue"
    
    def should_continue_after_entity_extraction(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        """LangGraph Conditional: Decide what to do after entity extraction"""
        return "error" if state["errors"] else "continue"
    
    def should_continue_after_trajectory(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        """LangGraph Conditional: Decide what to do after trajectory analysis"""
        return "error" if state["errors"] else "continue"
    
    def should_continue_after_summary(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        """LangGraph Conditional: Decide what to do after summary generation"""
        return "error" if state["errors"] else "continue"
    
    def should_continue_after_heart_attack_prediction(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        """LangGraph Conditional: Decide what to do after heart attack prediction"""
        return "error" if state["errors"] else "continue"
    
    # ===== FASTAPI HEART ATTACK PREDICTION HELPER METHODS =====
    
    def _get_risk_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on risk level"""
        recommendations = {
            "HIGH": "Immediate medical consultation recommended. Consider cardiac evaluation and risk factor modification.",
            "MODERATE": "Regular monitoring advised. Discuss with healthcare provider about preventive measures.",
            "LOW": "Continue healthy lifestyle practices. Regular check-ups as per medical advice."
        }
        return recommendations.get(risk_level, "Consult healthcare provider for personalized advice.")
    
    def _identify_risk_factors_for_fastapi(self, state: HealthAnalysisState) -> List[str]:
        """Identify key risk factors from the FastAPI model analysis"""
        risk_factors = []
        
        # Get extracted features
        heart_attack_features = state.get("heart_attack_features", {})
        extracted_features = heart_attack_features.get("extracted_features", {})
        
        # Age risk
        age = extracted_features.get("Age", 0)
        if age > 65:
            risk_factors.append(f"Advanced age ({age} years)")
        elif age > 55:
            risk_factors.append(f"Moderate age risk ({age} years)")
        
        # Gender risk
        if extracted_features.get("Gender", 0) == 1:
            risk_factors.append("Male gender")
        
        # Diabetes
        if extracted_features.get("Diabetes", 0) == 1:
            risk_factors.append("Diabetes mellitus")
        
        # Hypertension
        if extracted_features.get("High_BP", 0) == 1:
            risk_factors.append("High blood pressure")
        
        # Smoking
        if extracted_features.get("Smoking", 0) == 1:
            risk_factors.append("Smoking history")
        
        return risk_factors
    
    # ===== CHATBOT FUNCTIONALITY =====
    
    def chat_with_data(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Handle chatbot conversation with comprehensive medical data access"""
        try:
            # Prepare comprehensive context with ALL available data in chunks
            comprehensive_context = self._prepare_chunked_context(chat_context)
            
            # Build conversation history for continuity (last 6 messages)
            history_text = ""
            if chat_history:
                recent_history = chat_history[-6:]
                history_text = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in recent_history
                ])
            
            # Create comprehensive prompt with all medical data
            comprehensive_prompt = f"""You are an expert medical data assistant with access to complete patient health records. Answer the user's question with specific, detailed information from the medical data provided.

COMPLETE PATIENT DATA AVAILABLE:
{comprehensive_context}

RECENT CONVERSATION HISTORY:
{history_text}

USER QUESTION: {user_query}

INSTRUCTIONS:
- Provide detailed, specific answers based on the medical data
- Include relevant dates, codes, medications, diagnoses, and values when available
- Use conversation history to understand context and follow-up questions
- For heart attack risk questions, use the FastAPI prediction results
- Be thorough but focused on what the user is asking
- Include specific data points, codes, and numbers when relevant
- If asking about medications, include NDC codes and medication names
- If asking about diagnoses, include ICD-10 codes and descriptions
- If asking about dates or timeline, provide specific dates from the data

DETAILED ANSWER:"""

            logger.info(f"💬 Processing comprehensive query: {user_query[:50]}...")
            
            # Call Snowflake Cortex with comprehensive context
            response = self.call_llm(comprehensive_prompt)
            
            if response.startswith("Error"):
                return f"I encountered an error processing your question. Please try rephrasing your question."
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chatbot conversation: {str(e)}")
            return f"I encountered an error processing your question. Please try again."

    def _prepare_chunked_context(self, chat_context: Dict[str, Any]) -> str:
        """Prepare comprehensive context in chunks to avoid payload issues"""
        try:
            context_sections = []
            
            # 1. Patient Overview (small)
            patient_overview = chat_context.get("patient_overview", {})
            if patient_overview:
                context_sections.append(f"PATIENT OVERVIEW:\n{json.dumps(patient_overview, indent=2)}")
            
            # 2. Deidentified Medical Data (chunked)
            deidentified_medical = chat_context.get("deidentified_medical", {})
            if deidentified_medical:
                # Summarize large medical data
                medical_summary = {
                    "patient_info": {
                        "name": f"{deidentified_medical.get('src_mbr_first_nm', 'N/A')} {deidentified_medical.get('src_mbr_last_nm', 'N/A')}",
                        "age": deidentified_medical.get('src_mbr_age', 'N/A'),
                        "zip": deidentified_medical.get('src_mbr_zip_cd', 'N/A')
                    },
                    "medical_data_available": "Yes" if deidentified_medical.get('medical_data') else "No"
                }
                context_sections.append(f"DEIDENTIFIED MEDICAL DATA:\n{json.dumps(medical_summary, indent=2)}")
            
            # 3. Medical Extractions (detailed)
            medical_extraction = chat_context.get("medical_extraction", {})
            if medical_extraction and not medical_extraction.get('error'):
                extraction_summary = medical_extraction.get('extraction_summary', {})
                hlth_records = medical_extraction.get('hlth_srvc_records', [])
                
                medical_details = {
                    "summary": extraction_summary,
                    "health_service_records": hlth_records[:10] if len(hlth_records) > 10 else hlth_records  # Limit to first 10
                }
                context_sections.append(f"MEDICAL EXTRACTIONS:\n{json.dumps(medical_details, indent=2)}")
            
            # 4. Pharmacy Extractions (detailed)
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            if pharmacy_extraction and not pharmacy_extraction.get('error'):
                extraction_summary = pharmacy_extraction.get('extraction_summary', {})
                ndc_records = pharmacy_extraction.get('ndc_records', [])
                
                pharmacy_details = {
                    "summary": extraction_summary,
                    "ndc_records": ndc_records[:15] if len(ndc_records) > 15 else ndc_records  # Limit to first 15
                }
                context_sections.append(f"PHARMACY EXTRACTIONS:\n{json.dumps(pharmacy_details, indent=2)}")
            
            # 5. Entity Extraction (small)
            entity_extraction = chat_context.get("entity_extraction", {})
            if entity_extraction:
                context_sections.append(f"HEALTH ENTITIES:\n{json.dumps(entity_extraction, indent=2)}")
            
            # 6. Heart Attack Prediction (small)
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            if heart_attack_prediction:
                context_sections.append(f"HEART ATTACK PREDICTION:\n{json.dumps(heart_attack_prediction, indent=2)}")
            
            # 7. Health Analysis (text summaries)
            health_trajectory = chat_context.get("health_trajectory", "")
            if health_trajectory:
                # Truncate if too long
                trajectory_text = health_trajectory[:1000] + "..." if len(health_trajectory) > 1000 else health_trajectory
                context_sections.append(f"HEALTH TRAJECTORY ANALYSIS:\n{trajectory_text}")
            
            final_summary = chat_context.get("final_summary", "")
            if final_summary:
                # Truncate if too long
                summary_text = final_summary[:1000] + "..." if len(final_summary) > 1000 else final_summary
                context_sections.append(f"CLINICAL SUMMARY:\n{summary_text}")
            
            # Join all sections
            return "\n\n" + "\n\n".join(context_sections)
            
        except Exception as e:
            logger.error(f"Error preparing chunked context: {e}")
            return "Patient medical data available for analysis."
    
    # ===== HELPER METHODS - FIXED DIAGNOSIS EXTRACTION =====
    
    def _extract_medical_fields(self, deidentified_medical: Dict[str, Any]) -> Dict[str, Any]:
        """Extract hlth_srvc_cd and diag_1_50_cd fields from deidentified medical data - FIXED for comma-separated diagnosis codes"""
        extraction_result = {
            "hlth_srvc_records": [],
            "extraction_summary": {
                "total_hlth_srvc_records": 0,
                "total_diagnosis_codes": 0,
                "unique_service_codes": set(),
                "unique_diagnosis_codes": set()
            }
        }
        
        try:
            logger.info("🔍 Starting medical field extraction...")
            
            medical_data = deidentified_medical.get("medical_data", {})
            if not medical_data:
                logger.warning("No medical data found in deidentified medical data")
                return extraction_result
            
            self._recursive_medical_extraction(medical_data, extraction_result)
            
            extraction_result["extraction_summary"]["unique_service_codes"] = list(
                extraction_result["extraction_summary"]["unique_service_codes"]
            )
            extraction_result["extraction_summary"]["unique_diagnosis_codes"] = list(
                extraction_result["extraction_summary"]["unique_diagnosis_codes"]
            )
            
            logger.info(f"📋 Medical extraction completed: "
                       f"{extraction_result['extraction_summary']['total_hlth_srvc_records']} health service records, "
                       f"{extraction_result['extraction_summary']['total_diagnosis_codes']} diagnosis codes")
            
        except Exception as e:
            logger.error(f"Error in medical field extraction: {e}")
            extraction_result["error"] = f"Medical extraction failed: {str(e)}"
        
        return extraction_result
    
    def _recursive_medical_extraction(self, data: Any, result: Dict[str, Any], path: str = ""):
        """Recursively search for medical fields in nested data structures - FIXED for comma-separated diagnosis codes"""
        if isinstance(data, dict):
            current_record = {}
            
            # Extract health service code
            if "hlth_srvc_cd" in data:
                current_record["hlth_srvc_cd"] = data["hlth_srvc_cd"]
                result["extraction_summary"]["unique_service_codes"].add(str(data["hlth_srvc_cd"]))
            
            diagnosis_codes = []
            
            # FIXED: Handle comma-separated diagnosis codes in diag_1_50_cd field
            if "diag_1_50_cd" in data and data["diag_1_50_cd"]:
                diag_value = str(data["diag_1_50_cd"]).strip()
                if diag_value and diag_value.lower() != 'null':
                    # Split by comma and process each diagnosis code
                    individual_codes = [code.strip() for code in diag_value.split(',') if code.strip()]
                    for i, code in enumerate(individual_codes, 1):
                        if code:  # Only add non-empty codes
                            diagnosis_codes.append({
                                "code": code,
                                "position": i,
                                "source": "diag_1_50_cd (comma-separated)"
                            })
                            result["extraction_summary"]["unique_diagnosis_codes"].add(code)
            
            # Also handle individual diagnosis fields (diag_1_cd, diag_2_cd, etc.) for backwards compatibility
            for i in range(1, 51):
                diag_key = f"diag_{i}_cd"
                if diag_key in data and data[diag_key]:
                    diag_code = str(data[diag_key]).strip()
                    if diag_code and diag_code.lower() != 'null':
                        diagnosis_codes.append({
                            "code": diag_code,
                            "position": i,
                            "source": f"individual field ({diag_key})"
                        })
                        result["extraction_summary"]["unique_diagnosis_codes"].add(diag_code)
            
            if diagnosis_codes:
                current_record["diagnosis_codes"] = diagnosis_codes
                result["extraction_summary"]["total_diagnosis_codes"] += len(diagnosis_codes)
            
            if current_record:
                current_record["data_path"] = path
                result["hlth_srvc_records"].append(current_record)
                result["extraction_summary"]["total_hlth_srvc_records"] += 1
            
            # Continue recursive search
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self._recursive_medical_extraction(value, result, new_path)
                
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                self._recursive_medical_extraction(item, result, new_path)
    
    def _extract_pharmacy_fields(self, deidentified_pharmacy: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Ndc and lbl_nm fields from deidentified pharmacy data"""
        extraction_result = {
            "ndc_records": [],
            "extraction_summary": {
                "total_ndc_records": 0,
                "unique_ndc_codes": set(),
                "unique_label_names": set()
            }
        }
        
        try:
            logger.info("🔍 Starting pharmacy field extraction...")
            
            pharmacy_data = deidentified_pharmacy.get("pharmacy_data", {})
            if not pharmacy_data:
                logger.warning("No pharmacy data found in deidentified pharmacy data")
                return extraction_result
            
            self._recursive_pharmacy_extraction(pharmacy_data, extraction_result)
            
            extraction_result["extraction_summary"]["unique_ndc_codes"] = list(
                extraction_result["extraction_summary"]["unique_ndc_codes"]
            )
            extraction_result["extraction_summary"]["unique_label_names"] = list(
                extraction_result["extraction_summary"]["unique_label_names"]
            )
            
            logger.info(f"💊 Pharmacy extraction completed: "
                       f"{extraction_result['extraction_summary']['total_ndc_records']} NDC records")
            
        except Exception as e:
            logger.error(f"Error in pharmacy field extraction: {e}")
            extraction_result["error"] = f"Pharmacy extraction failed: {str(e)}"
        
        return extraction_result
    
    def _recursive_pharmacy_extraction(self, data: Any, result: Dict[str, Any], path: str = ""):
        """Recursively search for pharmacy fields in nested data structures"""
        if isinstance(data, dict):
            current_record = {}
            
            for key in data:
                if key.lower() in ['ndc', 'ndc_code', 'ndc_number']:
                    current_record["ndc"] = data[key]
                    result["extraction_summary"]["unique_ndc_codes"].add(str(data[key]))
                    break
            
            for key in data:
                if key.lower() in ['lbl_nm', 'label_name', 'drug_name', 'medication_name']:
                    current_record["lbl_nm"] = data[key]
                    result["extraction_summary"]["unique_label_names"].add(str(data[key]))
                    break
            
            if current_record:
                current_record["data_path"] = path
                result["ndc_records"].append(current_record)
                result["extraction_summary"]["total_ndc_records"] += 1
            
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self._recursive_pharmacy_extraction(value, result, new_path)
                
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                self._recursive_pharmacy_extraction(item, result, new_path)
    
    def _deidentify_medical_data(self, medical_data: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deidentify medical data with specific field transformations"""
        try:
            if not medical_data:
                return {"error": "No medical data to deidentify"}
            
            try:
                dob_str = patient_data.get('date_of_birth', '')
                if dob_str:
                    dob = datetime.strptime(dob_str, '%Y-%m-%d').date()
                    today = date.today()
                    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                else:
                    age = "unknown"
            except Exception as e:
                logger.warning(f"Error calculating age: {e}")
                age = "unknown"
            
            deidentified = {
                "src_mbr_first_nm": "john",
                "src_mbr_last_nm": "smith", 
                "src_mbr_mid_init_nm": None,
                "src_mbr_age": age,
                "src_mbr_zip_cd": "12345",
                "medical_data": self._remove_pii_from_data(medical_data.get('body', medical_data))
            }
            
            return deidentified
            
        except Exception as e:
            logger.error(f"Error in medical deidentification: {e}")
            return {"error": f"Deidentification failed: {str(e)}"}
    
    def _deidentify_pharmacy_data(self, pharmacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deidentify pharmacy data by removing PII"""
        try:
            if not pharmacy_data:
                return {"error": "No pharmacy data to deidentify"}
            
            return {
                "pharmacy_data": self._remove_pii_from_data(pharmacy_data.get('body', pharmacy_data))
            }
            
        except Exception as e:
            logger.error(f"Error in pharmacy deidentification: {e}")
            return {"error": f"Deidentification failed: {str(e)}"}
    
    def _remove_pii_from_data(self, data: Any) -> Any:
        """Remove PII from data structure"""
        try:
            if isinstance(data, dict):
                return {k: self._remove_pii_from_data(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [self._remove_pii_from_data(item) for item in data]
            elif isinstance(data, str):
                data = re.sub(r'\b\d{3}-?\d{2}-?\d{4}\b', '[SSN_MASKED]', data)
                data = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[NAME_MASKED]', data)
                data = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_MASKED]', data)
                data = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_MASKED]', data)
                return data
            else:
                return data
        except Exception as e:
            logger.warning(f"Error removing PII: {e}")
            return data
    
    def _extract_health_entities_enhanced(self, pharmacy_data: Dict[str, Any], 
                                        pharmacy_extraction: Dict[str, Any],
                                        medical_extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced health entity extraction using pharmacy data, extractions, and medical codes"""
        entities = {
            "diabetics": "no",
            "age_group": "unknown", 
            "smoking": "no",
            "alcohol": "no",
            "blood_pressure": "unknown",
            "analysis_details": [],
            "medical_conditions": [],
            "medications_identified": []
        }
        
        try:
            if pharmacy_data:
                data_str = json.dumps(pharmacy_data).lower()
                self._analyze_pharmacy_for_entities(data_str, entities)
            
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                self._analyze_pharmacy_extraction_for_entities(pharmacy_extraction, entities)
            
            if medical_extraction and medical_extraction.get("hlth_srvc_records"):
                self._analyze_medical_extraction_for_entities(medical_extraction, entities)
            
            entities["analysis_details"].append(f"Total analysis sources: Pharmacy data, {len(pharmacy_extraction.get('ndc_records', []))} pharmacy records, {len(medical_extraction.get('hlth_srvc_records', []))} medical records")
            
        except Exception as e:
            logger.error(f"Error in enhanced entity extraction: {e}")
            entities["analysis_details"].append(f"Error in entity extraction: {str(e)}")
        
        return entities
    
    def _analyze_pharmacy_for_entities(self, data_str: str, entities: Dict[str, Any]):
        """Original pharmacy data analysis for entities"""
        diabetes_keywords = [
            'insulin', 'metformin', 'glipizide', 'diabetes', 'diabetic', 
            'glucophage', 'lantus', 'humalog', 'novolog', 'levemir'
        ]
        for keyword in diabetes_keywords:
            if keyword in data_str:
                entities["diabetics"] = "yes"
                entities["analysis_details"].append(f"Diabetes indicator found in pharmacy data: {keyword}")
                break
        
        senior_medications = [
            'aricept', 'warfarin', 'lisinopril', 'atorvastatin', 'metoprolol',
            'furosemide', 'amlodipine', 'simvastatin'
        ]
        adult_medications = [
            'adderall', 'vyvanse', 'accutane', 'birth control'
        ]
        
        for med in senior_medications:
            if med in data_str:
                entities["age_group"] = "senior"
                entities["analysis_details"].append(f"Senior medication found: {med}")
                break
        
        if entities["age_group"] == "unknown":
            for med in adult_medications:
                if med in data_str:
                    entities["age_group"] = "adult"
                    entities["analysis_details"].append(f"Adult medication found: {med}")
                    break
    
    def _analyze_pharmacy_extraction_for_entities(self, pharmacy_extraction: Dict[str, Any], entities: Dict[str, Any]):
        """Analyze structured pharmacy extraction for health entities"""
        ndc_records = pharmacy_extraction.get("ndc_records", [])
        
        for record in ndc_records:
            ndc = record.get("ndc", "")
            lbl_nm = record.get("lbl_nm", "")
            
            if lbl_nm:
                entities["medications_identified"].append({
                    "ndc": ndc,
                    "label_name": lbl_nm,
                    "path": record.get("data_path", "")
                })
                
                lbl_lower = lbl_nm.lower()
                
                if any(word in lbl_lower for word in ['insulin', 'metformin', 'glucophage', 'diabetes']):
                    entities["diabetics"] = "yes"
                    entities["analysis_details"].append(f"Diabetes medication found in extraction: {lbl_nm}")
                
                if any(word in lbl_lower for word in ['lisinopril', 'amlodipine', 'metoprolol', 'blood pressure']):
                    entities["blood_pressure"] = "managed"
                    entities["analysis_details"].append(f"Blood pressure medication found in extraction: {lbl_nm}")
    
    def _analyze_medical_extraction_for_entities(self, medical_extraction: Dict[str, Any], entities: Dict[str, Any]):
        """Analyze medical codes for health conditions"""
        hlth_srvc_records = medical_extraction.get("hlth_srvc_records", [])
        
        condition_mappings = {
            "diabetes": ["E10", "E11", "E12", "E13", "E14"],
            "hypertension": ["I10", "I11", "I12", "I13", "I15"],
            "smoking": ["Z72.0", "F17"],
            "alcohol": ["F10", "Z72.1"],
        }
        
        for record in hlth_srvc_records:
            diagnosis_codes = record.get("diagnosis_codes", [])
            for diag in diagnosis_codes:
                diag_code = diag.get("code", "")
                if diag_code:
                    for condition, code_prefixes in condition_mappings.items():
                        if any(diag_code.startswith(prefix) for prefix in code_prefixes):
                            if condition == "diabetes":
                                entities["diabetics"] = "yes"
                                entities["medical_conditions"].append(f"Diabetes (ICD-10: {diag_code})")
                            elif condition == "hypertension":
                                entities["blood_pressure"] = "diagnosed"
                                entities["medical_conditions"].append(f"Hypertension (ICD-10: {diag_code})")
                            elif condition == "smoking":
                                entities["smoking"] = "yes"
                                entities["medical_conditions"].append(f"Smoking (ICD-10: {diag_code})")
                            
                            entities["analysis_details"].append(f"Medical condition identified from ICD-10 {diag_code}: {condition}")
    
    def _create_enhanced_trajectory_prompt(self, medical_data: Dict, pharmacy_data: Dict, 
                                         medical_extraction: Dict, pharmacy_extraction: Dict, 
                                         entities: Dict) -> str:
        """Create enhanced prompt for health trajectory analysis"""
        return f"""
You are a healthcare AI assistant analyzing a patient's health trajectory. Based on the following comprehensive deidentified data, provide a detailed health trajectory analysis.

DEIDENTIFIED MEDICAL DATA:
{json.dumps(medical_data, indent=2)}

DEIDENTIFIED PHARMACY DATA:
{json.dumps(pharmacy_data, indent=2)}

STRUCTURED MEDICAL EXTRACTION:
{json.dumps(medical_extraction, indent=2)}

STRUCTURED PHARMACY EXTRACTION:
{json.dumps(pharmacy_extraction, indent=2)}

EXTRACTED HEALTH ENTITIES:
{json.dumps(entities, indent=2)}

Please analyze this patient's health trajectory focusing on:

1. **Current Health Status**: Overall assessment based on medical codes, pharmacy data, and extracted entities
2. **Risk Factors**: Identified health risks from ICD-10 codes and medication patterns
3. **Medication Analysis**: NDC codes, drug names, and therapeutic areas identified
4. **Chronic Conditions**: Long-term health management needs from medical service codes
5. **Health Trends**: Trajectory of health over time based on service utilization
6. **Care Recommendations**: Suggested areas for medical attention based on comprehensive data analysis

Provide a detailed analysis (400-500 words) that synthesizes all the available structured and unstructured information into a coherent health trajectory assessment.
"""
    
    def _create_enhanced_summary_prompt(self, trajectory_analysis: str, entities: Dict, 
                                      medical_extraction: Dict, pharmacy_extraction: Dict) -> str:
        """Create enhanced prompt for final health summary"""
        return f"""
Based on the detailed health trajectory analysis below and the comprehensive data extractions, create a concise executive summary of this patient's health status.

DETAILED HEALTH TRAJECTORY ANALYSIS:
{trajectory_analysis}

KEY HEALTH ENTITIES:
- Diabetes: {entities.get('diabetics', 'unknown')}
- Age Group: {entities.get('age_group', 'unknown')}
- Smoking Status: {entities.get('smoking', 'unknown')}
- Alcohol Status: {entities.get('alcohol', 'unknown')}
- Blood Pressure: {entities.get('blood_pressure', 'unknown')}
- Medical Conditions Identified: {len(entities.get('medical_conditions', []))}
- Medications Identified: {len(entities.get('medications_identified', []))}

MEDICAL DATA SUMMARY:
- Health Service Records: {len(medical_extraction.get('hlth_srvc_records', []))}
- Total Diagnosis Codes: {medical_extraction.get('extraction_summary', {}).get('total_diagnosis_codes', 0)}
- Unique Service Codes: {len(medical_extraction.get('extraction_summary', {}).get('unique_service_codes', []))}

PHARMACY DATA SUMMARY:
- NDC Records: {len(pharmacy_extraction.get('ndc_records', []))}
- Unique NDC Codes: {len(pharmacy_extraction.get('extraction_summary', {}).get('unique_ndc_codes', []))}
- Unique Medications: {len(pharmacy_extraction.get('extraction_summary', {}).get('unique_label_names', []))}

Create a final summary that includes:

1. **Health Status Overview** (2-3 sentences)
2. **Key Risk Factors** (bullet points based on ICD-10 codes and medications)
3. **Priority Recommendations** (3-4 actionable items based on comprehensive analysis)
4. **Follow-up Needs** (timing and type of care based on service codes and medication patterns)

Keep the summary under 250 words and focus on actionable insights for healthcare providers based on the comprehensive data analysis.
"""
    
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
    
    def run_analysis(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the enhanced health analysis workflow using LangGraph with FastAPI heart attack prediction"""
        
        # Initialize enhanced state for LangGraph (8 nodes)
        initial_state = HealthAnalysisState(
            patient_data=patient_data,
            mcid_output={},
            medical_output={},
            pharmacy_output={},
            token_output={},
            deidentified_medical={},
            deidentified_pharmacy={},
            medical_extraction={},
            pharmacy_extraction={},
            entity_extraction={},
            health_trajectory="",
            final_summary="",
            # UPDATED: Heart attack prediction via FastAPI
            heart_attack_prediction={},
            heart_attack_risk_score=0.0,
            heart_attack_features={},
            chatbot_ready=False,
            chatbot_context={},
            chat_history=[],
            current_step="",
            errors=[],
            retry_count=0,
            processing_complete=False,
            step_status={}
        )
        
        try:
            config_dict = {"configurable": {"thread_id": f"health_analysis_{datetime.now().timestamp()}"}}
            
            logger.info("🚀 Starting Enhanced LangGraph health analysis workflow with FastAPI heart attack prediction...")
            logger.info(f"🔧 Snowflake Model: {self.config.model}")
            logger.info(f"🔧 FastAPI: {self.config.fastapi_url}")
            logger.info(f"💬 Chatbot: Interactive mode enabled")
            logger.info(f"❤️ Heart Attack Prediction: FastAPI server enabled")
            logger.info(f"🔗 FastAPI Server: {self.config.heart_attack_api_url}")
            
            final_state = self.graph.invoke(initial_state, config=config_dict)
            
            # Prepare enhanced results
            results = {
                "success": final_state["processing_complete"] and not final_state["errors"],
                "patient_data": final_state["patient_data"],
                "api_outputs": {
                    "mcid": final_state["mcid_output"],
                    "medical": final_state["medical_output"], 
                    "pharmacy": final_state["pharmacy_output"],
                    "token": final_state["token_output"]
                },
                "deidentified_data": {
                    "medical": final_state["deidentified_medical"],
                    "pharmacy": final_state["deidentified_pharmacy"]
                },
                "structured_extractions": {
                    "medical": final_state["medical_extraction"],
                    "pharmacy": final_state["pharmacy_extraction"]
                },
                "entity_extraction": final_state["entity_extraction"],
                "health_trajectory": final_state["health_trajectory"],
                "final_summary": final_state["final_summary"],
                # UPDATED: FastAPI heart attack prediction results
                "heart_attack_prediction": final_state["heart_attack_prediction"],
                "heart_attack_risk_score": final_state["heart_attack_risk_score"],
                "heart_attack_features": final_state["heart_attack_features"],
                "chatbot_ready": final_state["chatbot_ready"],
                "chatbot_context": final_state["chatbot_context"],
                "chat_history": final_state["chat_history"],
                "errors": final_state["errors"],
                "processing_steps_completed": self._count_completed_steps(final_state),
                "step_status": final_state["step_status"],
                "langgraph_used": True,
                "enhancement_version": "v4.0_with_fastapi_heart_attack_prediction"
            }
            
            if results["success"]:
                logger.info("✅ Enhanced LangGraph health analysis with FastAPI heart attack prediction completed successfully!")
                logger.info(f"📊 Medical records extracted: {len(final_state.get('medical_extraction', {}).get('hlth_srvc_records', []))}")
                logger.info(f"💊 Pharmacy records extracted: {len(final_state.get('pharmacy_extraction', {}).get('ndc_records', []))}")
                logger.info(f"❤️ Heart attack risk score: {final_state.get('heart_attack_risk_score', 0.0):.3f}")
                logger.info(f"💬 Chatbot ready: {final_state.get('chatbot_ready', False)}")
            else:
                logger.error(f"❌ Enhanced LangGraph health analysis failed with errors: {final_state['errors']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Fatal error in Enhanced LangGraph workflow: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "patient_data": patient_data,
                "api_outputs": {},
                "deidentified_data": {},
                "structured_extractions": {},
                "entity_extraction": {},
                "health_trajectory": "",
                "final_summary": "",
                "heart_attack_prediction": {"error": "Workflow failed before prediction"},
                "heart_attack_risk_score": 0.0,
                "heart_attack_features": {},
                "chatbot_ready": False,
                "chatbot_context": {},
                "chat_history": [],
                "errors": [str(e)],
                "processing_steps_completed": 0,
                "step_status": {"workflow": "error"},
                "langgraph_used": True,
                "enhancement_version": "v4.0_with_fastapi_heart_attack_prediction"
            }
    
    def _count_completed_steps(self, state: HealthAnalysisState) -> int:
        """Count how many processing steps were completed (8 nodes)"""
        steps = 0
        if state.get("mcid_output"): steps += 1
        if state.get("deidentified_medical") and not state.get("deidentified_medical", {}).get("error"): steps += 1
        if state.get("medical_extraction") or state.get("pharmacy_extraction"): steps += 1
        if state.get("entity_extraction"): steps += 1
        if state.get("health_trajectory"): steps += 1
        if state.get("final_summary"): steps += 1
        if state.get("heart_attack_prediction"): steps += 1  # UPDATED: Count FastAPI heart attack prediction
        if state.get("chatbot_ready"): steps += 1
        return steps

def main():
    """Example usage of the Enhanced LangGraph Health Analysis Agent with FastAPI Heart Attack Prediction"""
    
    print("🏥 Enhanced LangGraph Health Analysis Agent v4.0")
    print("✅ Agent is ready with Snowflake Cortex API + Interactive Chatbot + FastAPI Heart Attack Prediction")
    print("❤️ Heart Attack Prediction using FastAPI server with ML model")
    print("🔧 To use this agent, run: streamlit run streamlit_langgraph_ui.py")
    print()
    
    # Show configuration including FastAPI heart attack prediction
    config = Config(
        heart_attack_api_url="http://localhost:8002"  # User will specify this
    )
    
    print("📋 Enhanced Configuration:")
    print(f"   🌐 API URL: {config.api_url}")
    print(f"   🤖 Model: {config.model}")
    print(f"   🔑 App ID: {config.app_id}")
    print(f"   ❤️ FastAPI Heart Attack Server: {config.heart_attack_api_url}")
    print(f"   📊 Risk Threshold: {config.heart_attack_threshold}")
    print(f"   🎯 Expected Features: Age, Gender, Diabetes, High_BP, Smoking")
    print()
    print("✅ Enhanced Health Agent with FastAPI Heart Attack Prediction ready!")
    print("🚀 Run Streamlit to start the complete health analysis workflow")
    
    return "Enhanced Agent with FastAPI Heart Attack Prediction ready for Streamlit integration"

if __name__ == "__main__":
    main()
