# Updated imports for the synchronized components
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, TypedDict, Literal, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import date
import requests

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import our enhanced synchronized components
# Note: Replace these imports with your actual file names
from health_api_integrator_enhanced import HealthAPIIntegrator  # Use the clean version
from health_data_processor_debug import HealthDataProcessor     # Use the debug version

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    fastapi_url: str = "http://localhost:8000"  # MCP server URL
    # Snowflake Cortex API Configuration
    api_url: str = "https://sfassist.edagenaipreprod.awsdns.internal.das/api/cortex/complete"
    api_key: str = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
    app_id: str = "edadip"
    aplctn_cd: str = "edagnai"
    model: str = "llama4-maverick"
    sys_msg: str = """You are an expert healthcare AI analyst with full access to comprehensive deidentified medical and pharmacy claims data WITH CODE MEANINGS. You have been provided with:1. Complete deidentified medical records including ICD-10 diagnosis codes WITH their medical meanings2. Complete deidentified pharmacy records including NDC codes and medication details WITH their therapeutic meanings3. Structured extractions of all medical and pharmacy fields WITH code explanations4. Enhanced entity analysis results based on code meaningsYour role is to provide detailed clinical insights, risk assessments, predictive analytics, and evidence-based analysis using ALL the available data including the code meanings. Always reference specific data points and code meanings from the provided records when making assessments. You HAVE ACCESS to the complete dataset with comprehensive code explanations and should use it comprehensively for advanced healthcare predictions."""
    chatbot_sys_msg: str = """You are a specialized healthcare AI assistant with COMPLETE ACCESS to this patient's deidentified medical and pharmacy claims data WITH COMPREHENSIVE CODE MEANINGS. You have been provided with:AVAILABLE DATA SOURCES:✅ Deidentified medical records with ICD-10 diagnosis codes AND their clinical meanings✅ Deidentified pharmacy records with NDC medication codes AND their therapeutic meanings✅ Health service utilization patterns (hlth_srvc_cd) WITH procedure explanations✅ Complete medication history with NDC codes, label names AND therapeutic explanations✅ Diagnosis codes (diag_1_50_cd) from medical claims WITH condition meanings✅ Enhanced entity extraction results including chronic conditions based on code meanings✅ Comprehensive health trajectory analysis with predictive insights✅ Code meanings for ALL medical and pharmacy codesYou can analyze the complete medical context because you have both the codes AND their professional meanings."""
    timeout: int = 30

    # Heart Attack Prediction API Configuration (separate from MCP server)
    heart_attack_api_url: str = "http://localhost:8000"  # Heart attack ML server
    heart_attack_threshold: float = 0.5
    max_retries: int = 3

    def to_dict(self):
        return asdict(self)

# Enhanced State Definition for LangGraph with MCP compatibility - REMOVED final_summary
class HealthAnalysisState(TypedDict):
    # Input data
    patient_data: Dict[str, Any]

    # Enhanced API outputs with MCP compatibility
    mcid_output: Dict[str, Any]
    medical_output: Dict[str, Any]
    pharmacy_output: Dict[str, Any]
    token_output: Dict[str, Any]

    # Enhanced processed data with comprehensive deidentification
    deidentified_medical: Dict[str, Any]
    deidentified_pharmacy: Dict[str, Any]
    deidentified_mcid: Dict[str, Any]

    # Enhanced extracted structured data WITH CODE MEANINGS
    medical_extraction: Dict[str, Any]
    pharmacy_extraction: Dict[str, Any]

    entity_extraction: Dict[str, Any]

    # Enhanced analysis results - REMOVED final_summary, enhanced health_trajectory
    health_trajectory: str

    # Enhanced Heart Attack Prediction via ML API
    heart_attack_prediction: Dict[str, Any]
    heart_attack_risk_score: float
    heart_attack_features: Dict[str, Any]

    # Enhanced chatbot functionality with comprehensive context
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
    """Enhanced Health Analysis Agent with Synchronized Components and Code Meanings"""

    def __init__(self, custom_config: Optional[Config] = None):
        # Use provided config or create default
        self.config = custom_config or Config()

        # Initialize enhanced synchronized components
        logger.info("🔧 Initializing Enhanced HealthAnalysisAgent with synchronized components...")
        
        # Initialize API integrator first
        self.api_integrator = HealthAPIIntegrator(self.config)
        
        # Pass API integrator to data processor for LLM explanations
        self.data_processor = HealthDataProcessor(self.api_integrator)

        logger.info("✅ Enhanced HealthAnalysisAgent initialized with synchronized components")
        logger.info(f"🌐 Snowflake API URL: {self.config.api_url}")
        logger.info(f"🤖 Model: {self.config.model}")
        logger.info(f"📡 MCP Server URL: {self.config.fastapi_url}")
        logger.info(f"❤️ Heart Attack ML API: {self.config.heart_attack_api_url}")
        logger.info("🎯 Enhanced entity extraction with synchronized code meanings enabled")
        
        # Test connections immediately
        self._test_connections_on_startup()

        self.setup_enhanced_langgraph()

    def _test_connections_on_startup(self):
        """Test all connections during startup"""
        try:
            logger.info("🧪 Testing connections during startup...")
            
            # Test isolated LLM connection (critical for meaning generation)
            isolated_test = self.api_integrator.test_isolated_llm_connection()
            if isolated_test.get("success"):
                logger.info("✅ Isolated LLM connection successful - meaning generation will work")
            else:
                logger.error(f"❌ Isolated LLM connection failed: {isolated_test.get('error')}")
                logger.error("❌ CODE MEANING GENERATION WILL NOT WORK!")
            
            # Test main LLM connection
            llm_test = self.api_integrator.test_llm_connection()
            if llm_test.get("success"):
                logger.info("✅ Main LLM connection successful")
            else:
                logger.warning(f"⚠️ Main LLM connection failed: {llm_test.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ Connection test during startup failed: {e}")

    def setup_enhanced_langgraph(self):
        """Setup enhanced LangGraph workflow with synchronized components"""
        logger.info("🔧 Setting up Enhanced LangGraph workflow with synchronized components...")

        # Create the StateGraph
        workflow = StateGraph(HealthAnalysisState)

        # Add all processing nodes - REMOVED generate_summary
        workflow.add_node("fetch_api_data", self.fetch_api_data)
        workflow.add_node("deidentify_claims_data", self.deidentify_claims_data)
        workflow.add_node("extract_claims_fields", self.extract_claims_fields)
        workflow.add_node("extract_entities", self.extract_entities)
        workflow.add_node("analyze_trajectory", self.analyze_trajectory)
        workflow.add_node("predict_heart_attack", self.predict_heart_attack)
        workflow.add_node("initialize_chatbot", self.initialize_chatbot)
        workflow.add_node("handle_error", self.handle_error)

        # Define the enhanced workflow edges - REMOVED generate_summary step
        workflow.add_edge(START, "fetch_api_data")

        # Conditional edges with enhanced retry logic
        workflow.add_conditional_edges(
            "fetch_api_data",
            self.should_continue_after_api,
            {
                "continue": "deidentify_claims_data",
                "retry": "fetch_api_data",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "deidentify_claims_data",
            self.should_continue_after_deidentify,
            {
                "continue": "extract_claims_fields",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "extract_claims_fields",
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
                "continue": "predict_heart_attack",  # DIRECT TO HEART ATTACK PREDICTION
                "error": "handle_error"
            }
        )

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

        logger.info("✅ Enhanced LangGraph workflow compiled successfully with synchronized components!")

    # ===== ENHANCED LANGGRAPH NODES =====

    def fetch_api_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 1: Fetch claims data from MCP-compatible APIs"""
        logger.info("🚀 Enhanced Node 1: Starting MCP-compatible Claims API data fetch...")
        state["current_step"] = "fetch_api_data"
        state["step_status"]["fetch_api_data"] = "running"

        try:
            patient_data = state["patient_data"]

            # Enhanced validation
            required_fields = ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"]
            for field in required_fields:
                if not patient_data.get(field):
                    state["errors"].append(f"Missing required field: {field}")
                    state["step_status"]["fetch_api_data"] = "error"
                    return state

            # Use enhanced API integrator to fetch data from MCP server
            api_result = self.api_integrator.fetch_backend_data(patient_data)

            if "error" in api_result:
                state["errors"].append(f"MCP Claims API Error: {api_result['error']}")
                state["step_status"]["fetch_api_data"] = "error"
            else:
                state["mcid_output"] = api_result.get("mcid_output", {})
                state["medical_output"] = api_result.get("medical_output", {})
                state["pharmacy_output"] = api_result.get("pharmacy_output", {})
                state["token_output"] = api_result.get("token_output", {})

                state["step_status"]["fetch_api_data"] = "completed"
                logger.info("✅ Successfully fetched all MCP-compatible Claims API data")

        except Exception as e:
            error_msg = f"Error fetching MCP Claims API data: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["fetch_api_data"] = "error"
            logger.error(error_msg)

        return state

    def deidentify_claims_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 2: Comprehensive deidentification using synchronized processor"""
        logger.info("🔒 Enhanced Node 2: Starting comprehensive claims data deidentification...")
        state["current_step"] = "deidentify_claims_data"
        state["step_status"]["deidentify_claims_data"] = "running"

        try:
            # Deidentify Medical Claims Data
            medical_data = state.get("medical_output", {})
            deidentified_medical = self.data_processor.deidentify_medical_data(medical_data, state["patient_data"])
            state["deidentified_medical"] = deidentified_medical

            # Deidentify Pharmacy Claims Data
            pharmacy_data = state.get("pharmacy_output", {})
            deidentified_pharmacy = self.data_processor.deidentify_pharmacy_data(pharmacy_data)
            state["deidentified_pharmacy"] = deidentified_pharmacy

            # Deidentify MCID Claims Data
            mcid_data = state.get("mcid_output", {})
            deidentified_mcid = self.data_processor.deidentify_mcid_data(mcid_data)
            state["deidentified_mcid"] = deidentified_mcid

            state["step_status"]["deidentify_claims_data"] = "completed"

            logger.info("✅ Successfully completed comprehensive claims data deidentification")
            logger.info(f"📊 Medical claims processed: {deidentified_medical.get('data_type', 'unknown')}")
            logger.info(f"📊 Pharmacy claims processed: {deidentified_pharmacy.get('data_type', 'unknown')}")
            logger.info(f"📊 MCID claims processed: {deidentified_mcid.get('data_type', 'unknown')}")

        except Exception as e:
            error_msg = f"Error in comprehensive claims data deidentification: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["deidentify_claims_data"] = "error"
            logger.error(error_msg)

        return state

    def extract_claims_fields(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 3: Extract fields WITH CODE MEANINGS using synchronized processor"""
        logger.info("🔍 Enhanced Node 3: Starting claims field extraction WITH CODE MEANINGS...")
        state["current_step"] = "extract_claims_fields"
        state["step_status"]["extract_claims_fields"] = "running"

        try:
            # Enhanced extraction WITH LLM-generated code meanings using synchronized processor
            logger.info("📋 Extracting medical fields with meanings...")
            medical_extraction = self.data_processor.extract_medical_fields(state.get("deidentified_medical", {}))
            state["medical_extraction"] = medical_extraction
            
            logger.info(f"📋 Medical extraction results:")
            logger.info(f"  📊 Records: {len(medical_extraction.get('hlth_srvc_records', []))}")
            logger.info(f"  🔤 Meanings added: {medical_extraction.get('code_meanings_added', False)}")
            logger.info(f"  🤖 LLM status: {medical_extraction.get('llm_call_status', 'unknown')}")

            logger.info("💊 Extracting pharmacy fields with meanings...")
            pharmacy_extraction = self.data_processor.extract_pharmacy_fields(state.get("deidentified_pharmacy", {}))
            state["pharmacy_extraction"] = pharmacy_extraction
            
            logger.info(f"💊 Pharmacy extraction results:")
            logger.info(f"  📊 Records: {len(pharmacy_extraction.get('ndc_records', []))}")
            logger.info(f"  🔤 Meanings added: {pharmacy_extraction.get('code_meanings_added', False)}")
            logger.info(f"  🤖 LLM status: {pharmacy_extraction.get('llm_call_status', 'unknown')}")

            state["step_status"]["extract_claims_fields"] = "completed"
            logger.info("✅ Successfully completed enhanced claims field extraction WITH CODE MEANINGS")

        except Exception as e:
            error_msg = f"Error in enhanced claims field extraction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_claims_fields"] = "error"
            logger.error(error_msg)

        return state

    def extract_entities(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 4: Extract health entities using CODE MEANINGS from synchronized processor"""
        logger.info("🎯 Enhanced Node 4: Starting LLM-powered health entity extraction WITH CODE MEANINGS...")
        state["current_step"] = "extract_entities"
        state["step_status"]["extract_entities"] = "running"
       
        try:
            pharmacy_data = state.get("pharmacy_output", {})
            pharmacy_extraction = state.get("pharmacy_extraction", {})
            medical_extraction = state.get("medical_extraction", {})
            patient_data = state.get("patient_data", {})
           
            # Calculate age from date of birth and add to patient data
            if patient_data.get('date_of_birth'):
                try:
                    dob = datetime.strptime(patient_data['date_of_birth'], '%Y-%m-%d').date()
                    today = date.today()
                    calculated_age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                    patient_data['calculated_age'] = calculated_age
                    logger.info(f"📅 Calculated age from DOB: {calculated_age} years")
                except Exception as e:
                    logger.warning(f"Could not calculate age from DOB: {e}")
           
            # Enhanced entity extraction WITH CODE MEANINGS using synchronized processor
            entities = self.data_processor.extract_health_entities_enhanced(
                pharmacy_data,
                pharmacy_extraction,
                medical_extraction,
                patient_data,  # Pass patient data for age calculation
                self.api_integrator  # Pass API integrator for LLM calls
            )
           
            state["entity_extraction"] = entities
            state["step_status"]["extract_entities"] = "completed"
           
            # Enhanced logging with code meanings status
            conditions_count = len(entities.get("medical_conditions", []))
            medications_count = len(entities.get("medications_identified", []))
            llm_status = entities.get("llm_analysis", "not_performed")
            code_meanings_used = entities.get("enhanced_with_code_meanings", False)
            age_info = f"Age: {entities.get('age', 'unknown')} ({entities.get('age_group', 'unknown')})"
           
            logger.info(f"✅ Successfully extracted health entities using synchronized processor:")
            logger.info(f"  📊 Conditions: {conditions_count}, Medications: {medications_count}")
            logger.info(f"  📊 Results: Diabetes={entities.get('diabetics')}, Smoking={entities.get('smoking')}, BP={entities.get('blood_pressure')}")
            logger.info(f"  📅 {age_info}")
            logger.info(f"  🤖 LLM analysis: {llm_status}")
            logger.info(f"  🔤 Code meanings enhanced: {code_meanings_used}")
           
        except Exception as e:
            error_msg = f"Error in LLM-powered entity extraction with synchronized processor: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_entities"] = "error"
            logger.error(error_msg)
       
        return state

    def analyze_trajectory(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 5: Analyze health trajectory using synchronized components"""
        logger.info("📈 Enhanced Node 5: Starting comprehensive health trajectory analysis...")
        state["current_step"] = "analyze_trajectory"
        state["step_status"]["analyze_trajectory"] = "running"

        try:
            deidentified_medical = state.get("deidentified_medical", {})
            deidentified_pharmacy = state.get("deidentified_pharmacy", {})
            deidentified_mcid = state.get("deidentified_mcid", {})
            medical_extraction = state.get("medical_extraction", {})
            pharmacy_extraction = state.get("pharmacy_extraction", {})
            entities = state.get("entity_extraction", {})

            trajectory_prompt = self._create_comprehensive_predictive_trajectory_prompt(
                deidentified_medical, deidentified_pharmacy, deidentified_mcid,
                medical_extraction, pharmacy_extraction, entities
            )

            logger.info("🤖 Calling Snowflake Cortex for comprehensive predictive trajectory analysis...")

            # Use API integrator for LLM call
            response = self.api_integrator.call_llm(trajectory_prompt)

            if response.startswith("Error"):
                state["errors"].append(f"Comprehensive predictive trajectory analysis failed: {response}")
                state["step_status"]["analyze_trajectory"] = "error"
            else:
                state["health_trajectory"] = response
                state["step_status"]["analyze_trajectory"] = "completed"
                logger.info("✅ Successfully completed comprehensive predictive trajectory analysis")

        except Exception as e:
            error_msg = f"Error in comprehensive predictive trajectory analysis: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["analyze_trajectory"] = "error"
            logger.error(error_msg)

        return state

    def predict_heart_attack(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 6: Heart attack prediction using synchronized components"""
        logger.info("❤️ Enhanced Node 6: Starting enhanced heart attack prediction...")
        state["current_step"] = "predict_heart_attack"
        state["step_status"]["predict_heart_attack"] = "running"

        try:
            # Step 1: Extract features using enhanced feature extraction
            logger.info("🔍 Extracting heart attack features...")
            features = self._extract_enhanced_heart_attack_features(state)
            state["heart_attack_features"] = features

            if not features or "error" in features:
                error_msg = "Failed to extract enhanced features for heart attack prediction"
                state["errors"].append(error_msg)
                state["step_status"]["predict_heart_attack"] = "error"
                logger.error(error_msg)
                return state

            # Step 2: Prepare feature vector for enhanced FastAPI call
            logger.info("⚙️ Preparing features for FastAPI call...")
            fastapi_features = self._prepare_enhanced_fastapi_features(features)

            if fastapi_features is None:
                error_msg = "Failed to prepare enhanced feature vector for prediction"
                state["errors"].append(error_msg)
                state["step_status"]["predict_heart_attack"] = "error"
                logger.error(error_msg)
                return state

            # Step 3: Make prediction using synchronous method
            logger.info("🚀 Making heart attack prediction call...")
            prediction_result = self._call_heart_attack_prediction_sync(fastapi_features)

            if prediction_result is None:
                error_msg = "Heart attack prediction returned None"
                state["errors"].append(error_msg)
                state["step_status"]["predict_heart_attack"] = "error"
                logger.error(error_msg)
                return state

            # Step 4: Process prediction result
            if prediction_result.get("success", False):
                logger.info("✅ Processing successful prediction result...")

                # Extract prediction data
                prediction_data = prediction_result.get("prediction_data", {})

                # Get risk probability and prediction
                risk_probability = prediction_data.get("probability", 0.0)
                binary_prediction = prediction_data.get("prediction", 0)

                # Convert to percentage
                risk_percentage = risk_probability * 100
                confidence_percentage = (1 - risk_probability) * 100 if binary_prediction == 0 else risk_probability * 100

                # Determine risk level
                if risk_percentage >= 70:
                    risk_category = "High Risk"
                elif risk_percentage >= 50:
                    risk_category = "Medium Risk"
                else:
                    risk_category = "Low Risk"

                # Create prediction result
                enhanced_prediction = {
                    "risk_display": f"Heart Disease Risk: {risk_percentage:.1f}% ({risk_category})",
                    "confidence_display": f"Confidence: {confidence_percentage:.1f}%",
                    "combined_display": f"Heart Disease Risk: {risk_percentage:.1f}% ({risk_category}) | Confidence: {confidence_percentage:.1f}%",
                    "raw_risk_score": risk_probability,
                    "raw_prediction": binary_prediction,
                    "risk_category": risk_category,
                    "fastapi_server_url": self.config.heart_attack_api_url,
                    "prediction_method": prediction_result.get("method", "unknown"),
                    "prediction_endpoint": prediction_result.get("endpoint", "unknown"),
                    "prediction_timestamp": datetime.now().isoformat(),
                    "enhanced_features_used": features.get("feature_interpretation", {}),
                    "model_enhanced": True
                }

                state["heart_attack_prediction"] = enhanced_prediction
                state["heart_attack_risk_score"] = float(risk_probability)

                logger.info(f"✅ Enhanced heart attack prediction completed successfully")
                logger.info(f"❤️ Display: {enhanced_prediction['combined_display']}")

            else:
                # Handle prediction failure
                error_msg = prediction_result.get("error", "Unknown FastAPI error")
                logger.warning(f"⚠️ Enhanced heart attack prediction failed: {error_msg}")

                state["heart_attack_prediction"] = {
                    "error": error_msg,
                    "risk_display": "Heart Disease Risk: Error",
                    "confidence_display": "Confidence: Error",
                    "combined_display": f"Heart Disease Risk: Error - {error_msg}",
                    "fastapi_server_url": self.config.heart_attack_api_url,
                    "error_details": error_msg,
                    "tried_endpoints": prediction_result.get("tried_endpoints", []),
                    "model_enhanced": True
                }
                state["heart_attack_risk_score"] = 0.0
            
            state["step_status"]["predict_heart_attack"] = "completed"

        except Exception as e:
            error_msg = f"Error in enhanced heart attack prediction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["predict_heart_attack"] = "error"
            logger.error(error_msg)

        return state

    def initialize_chatbot(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 7: Initialize chatbot with synchronized context including code meanings"""
        logger.info("💬 Enhanced Node 7: Initializing comprehensive chatbot with synchronized context...")
        state["current_step"] = "initialize_chatbot"
        state["step_status"]["initialize_chatbot"] = "running"

        try:
            # Prepare comprehensive chatbot context with all data INCLUDING CODE MEANINGS
            comprehensive_chatbot_context = {
                "deidentified_medical": state.get("deidentified_medical", {}),
                "deidentified_pharmacy": state.get("deidentified_pharmacy", {}),
                "deidentified_mcid": state.get("deidentified_mcid", {}),
                "medical_extraction": state.get("medical_extraction", {}),  # WITH CODE MEANINGS
                "pharmacy_extraction": state.get("pharmacy_extraction", {}),  # WITH CODE MEANINGS
                "entity_extraction": state.get("entity_extraction", {}),  # ENHANCED WITH CODE MEANINGS
                "health_trajectory": state.get("health_trajectory", ""),  # ENHANCED PREDICTIVE ANALYSIS
                "heart_attack_prediction": state.get("heart_attack_prediction", {}),
                "heart_attack_risk_score": state.get("heart_attack_risk_score", 0.0),
                "heart_attack_features": state.get("heart_attack_features", {}),
                "patient_overview": {
                    "age": state.get("deidentified_medical", {}).get("src_mbr_age", "unknown"),
                    "zip": state.get("deidentified_medical", {}).get("src_mbr_zip_cd", "unknown"),  # REAL ZIP CODE PRESERVED
                    "analysis_timestamp": datetime.now().isoformat(),
                    "heart_attack_risk_level": state.get("heart_attack_prediction", {}).get("risk_category", "unknown"),
                    "model_type": "enhanced_synchronized_components_with_code_meanings",
                    "deidentification_level": "comprehensive_claims_data_with_real_zip_preserved",
                    "claims_data_types": ["medical", "pharmacy", "mcid"],
                    "code_meanings_available": True,
                    "enhanced_entity_extraction": True,
                    "predictive_analysis_included": True,
                    "real_zip_code_available": True,
                    "comprehensive_analysis_enabled": True,
                    "synchronized_components": True
                }
            }

            state["chat_history"] = []
            state["chatbot_context"] = comprehensive_chatbot_context
            state["chatbot_ready"] = True
            state["processing_complete"] = True
            state["step_status"]["initialize_chatbot"] = "completed"

            # Log comprehensive chatbot initialization with synchronized components
            medical_records = len(state.get("medical_extraction", {}).get("hlth_srvc_records", []))
            pharmacy_records = len(state.get("pharmacy_extraction", {}).get("ndc_records", []))
            medical_meanings = state.get("medical_extraction", {}).get("code_meanings_added", False)
            pharmacy_meanings = state.get("pharmacy_extraction", {}).get("code_meanings_added", False)
            entity_meanings = state.get("entity_extraction", {}).get("enhanced_with_code_meanings", False)

            logger.info("✅ Successfully initialized comprehensive chatbot with synchronized components")
            logger.info(f"📊 Chatbot context includes: {medical_records} medical records, {pharmacy_records} pharmacy records")
            logger.info(f"🔤 Code meanings available: Medical={medical_meanings}, Pharmacy={pharmacy_meanings}, Entities={entity_meanings}")
            logger.info(f"📍 Real ZIP code preserved: {state.get('deidentified_medical', {}).get('src_mbr_zip_cd', 'unknown')}")
            logger.info(f"🔗 Synchronized components enabled with complete code meanings integration")

        except Exception as e:
            error_msg = f"Error initializing chatbot with synchronized components: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["initialize_chatbot"] = "error"
            logger.error(error_msg)

        return state

    def handle_error(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node: Enhanced error handling"""
        logger.error(f"🚨 Enhanced LangGraph Error Handler: {state['current_step']}")
        logger.error(f"Enhanced Errors: {state['errors']}")

        state["processing_complete"] = True
        current_step = state.get("current_step", "unknown")
        state["step_status"][current_step] = "error"
        return state

    # ===== ENHANCED LANGGRAPH CONDITIONAL EDGES =====

    def should_continue_after_api(self, state: HealthAnalysisState) -> Literal["continue", "retry", "error"]:
        if state["errors"]:
            if state["retry_count"] < self.config.max_retries:
                state["retry_count"] += 1
                logger.warning(f"🔄 Retrying enhanced API fetch (attempt {state['retry_count']}/{self.config.max_retries})")
                state["errors"] = []
                return "retry"
            else:
                logger.error(f"❌ Max retries ({self.config.max_retries}) exceeded for enhanced API fetch")
                return "error"
        return "continue"

    def should_continue_after_deidentify(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_extraction_step(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_entity_extraction(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_trajectory(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_heart_attack_prediction(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    # ===== ENHANCED CHATBOT FUNCTIONALITY WITH SYNCHRONIZED COMPONENTS =====
    def chat_with_data(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Enhanced chatbot with synchronized components and code meanings"""
        try:
            # Check if this is a heart attack related question
            heart_attack_keywords = ['heart attack', 'heart disease', 'cardiac', 'cardiovascular', 'heart risk', 'coronary', 'myocardial', 'cardiac risk']
            is_heart_attack_question = any(keyword in user_query.lower() for keyword in heart_attack_keywords)

            if is_heart_attack_question:
                return self._handle_heart_attack_question(user_query, chat_context, chat_history)
            else:
                return self._handle_general_question(user_query, chat_context, chat_history)

        except Exception as e:
            logger.error(f"Error in enhanced chatbot with synchronized components: {str(e)}")
            return "I encountered an error processing your question. Please try again. I have access to the complete deidentified claims data with comprehensive code meanings for analysis."

    def _handle_heart_attack_question(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Handle heart attack questions with synchronized components"""
        try:
            # Use the synchronized data processor to prepare context
            complete_context = self.data_processor.prepare_chunked_context(chat_context)
            
            # Build conversation history
            history_text = "No previous conversation"
            if chat_history:
                recent_history = chat_history[-5:]
                history_lines = []
                for msg in recent_history:
                    role = "User" if msg['role'] == 'user' else "Assistant"
                    content = msg['content'][:200]
                    history_lines.append(f"{role}: {content}")
                history_text = "\n".join(history_lines)

            # Create comprehensive heart attack analysis prompt
            heart_attack_prompt = f"""You are a cardiologist analyzing cardiovascular risk with COMPLETE synchronized access to patient data WITH CODE MEANINGS.

COMPLETE PATIENT DATA WITH CODE MEANINGS (from synchronized components):
{complete_context}

CONVERSATION HISTORY:
{history_text}

USER QUESTION: {user_query}

COMPREHENSIVE CARDIOVASCULAR ANALYSIS INSTRUCTIONS:
- You have COMPLETE ACCESS to deidentified medical and pharmacy data WITH CODE MEANINGS from synchronized components
- Use ICD-10 code meanings to understand cardiovascular conditions and comorbidities
- Use NDC code meanings to understand cardiovascular medications and effectiveness
- Use health service code meanings to understand cardiac procedures and interventions
- Provide evidence-based cardiovascular risk assessment with percentage scoring
- Include specific dates, codes, medications, diagnoses, and values from the complete data
- Use conversation history to understand follow-up questions and context
- Compare your analysis with any ML model predictions available
- Provide comprehensive cardiovascular risk scoring and analysis

Use ALL available synchronized data WITH CODE MEANINGS to provide the most comprehensive cardiovascular risk assessment possible."""

            logger.info(f"Processing comprehensive heart attack question with synchronized components: {user_query[:50]}...")

            # Use synchronized API integrator for LLM call
            response = self.api_integrator.call_llm(heart_attack_prompt, self.config.chatbot_sys_msg)

            if response.startswith("Error"):
                return "I encountered an error analyzing cardiovascular risk. Please try rephrasing your question. I have complete access to all patient data with code meanings for comprehensive cardiovascular analysis."

            return response

        except Exception as e:
            logger.error(f"Error in comprehensive heart attack question with synchronized components: {str(e)}")
            return "I encountered an error. Please try again with your cardiovascular question. I have complete access to all patient data including location data for comprehensive analysis."

    def _handle_general_question(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Handle general questions with synchronized components and code meanings"""
        try:
            # Use synchronized data processor to prepare complete context
            complete_context = self.data_processor.prepare_chunked_context(chat_context)
            
            # Build conversation history
            history_text = "No previous conversation"
            if chat_history:
                recent_history = chat_history[-10:]
                history_lines = []
                for msg in recent_history:
                    role = "User" if msg['role'] == 'user' else "Assistant"
                    content = msg['content'][:300]
                    history_lines.append(f"{role}: {content}")
                history_text = "\n".join(history_lines)

            # Create comprehensive prompt with synchronized context
            comprehensive_prompt = f"""You are a specialized healthcare AI analyst with COMPLETE synchronized access to patient data WITH CODE MEANINGS.

COMPLETE PATIENT DATA WITH CODE MEANINGS (from synchronized components):
{complete_context}

CONVERSATION HISTORY:
{history_text}

USER QUESTION: {user_query}

COMPREHENSIVE ANALYSIS CAPABILITIES with synchronized components:
- Complete access to deidentified medical, pharmacy, and MCID claims WITH CODE MEANINGS
- ICD-10 codes with their clinical meanings for understanding patient conditions
- NDC codes with their therapeutic meanings for understanding medications
- Health service codes with their procedure meanings for understanding care received
- Enhanced entity extraction based on code meanings
- Comprehensive health trajectory analysis
- Heart attack risk assessment
- Real ZIP code for location-based analysis

Use ALL available synchronized data and code meanings to provide comprehensive healthcare analysis."""

            logger.info(f"Processing comprehensive general query with synchronized components: {user_query[:50]}...")

            # Use synchronized API integrator for LLM call
            response = self.api_integrator.call_llm(comprehensive_prompt, self.config.chatbot_sys_msg)

            if response.startswith("Error"):
                return "I encountered an error processing your question. Please try rephrasing it. I have complete access to all patient data with code meanings for comprehensive analysis."

            return response

        except Exception as e:
            logger.error(f"Error in comprehensive general question with synchronized components: {str(e)}")
            return "I encountered an error. Please try again. I have complete access to all patient healthcare data including code meanings for comprehensive analysis and scoring."

    # ===== HELPER METHODS =====

    def _extract_enhanced_heart_attack_features(self, state: HealthAnalysisState) -> Dict[str, Any]:
        """Enhanced feature extraction for heart attack prediction"""
        try:
            features = {}

            # Enhanced patient age extraction
            deidentified_medical = state.get("deidentified_medical", {})
            patient_age = deidentified_medical.get("src_mbr_age", None)

            if patient_age and patient_age != "unknown":
                try:
                    age_value = int(float(str(patient_age)))
                    if 0 <= age_value <= 120:
                        features["Age"] = age_value
                    else:
                        features["Age"] = 50  # Default age
                except:
                    features["Age"] = 50
            else:
                features["Age"] = 50

            # Enhanced gender extraction
            patient_data = state.get("patient_data", {})
            gender = str(patient_data.get("gender", "F")).upper()
            features["Gender"] = 1 if gender in ["M", "MALE", "1"] else 0

            # Enhanced feature extraction from comprehensive entity extraction WITH CODE MEANINGS
            entity_extraction = state.get("entity_extraction", {})

            # Enhanced diabetes detection
            diabetes = str(entity_extraction.get("diabetics", "no")).lower()
            features["Diabetes"] = 1 if diabetes in ["yes", "true", "1"] else 0

            # Enhanced blood pressure detection
            blood_pressure = str(entity_extraction.get("blood_pressure", "unknown")).lower()
            features["High_BP"] = 1 if blood_pressure in ["managed", "diagnosed", "yes", "true", "1"] else 0

            # Enhanced smoking detection
            smoking = str(entity_extraction.get("smoking", "no")).lower()
            features["Smoking"] = 1 if smoking in ["yes", "true", "1"] else 0

            # Validate all features are integers
            for key in features:
                try:
                    features[key] = int(features[key])
                except:
                    if key == "Age":
                        features[key] = 50
                    else:
                        features[key] = 0

            # Create enhanced feature summary
            enhanced_feature_summary = {
                "extracted_features": features,
                "feature_interpretation": {
                    "Age": f"{features['Age']} years old",
                    "Gender": "Male" if features["Gender"] == 1 else "Female",
                    "Diabetes": "Yes" if features["Diabetes"] == 1 else "No",
                    "High_BP": "Yes" if features["High_BP"] == 1 else "No",
                    "Smoking": "Yes" if features["Smoking"] == 1 else "No"
                },
                "data_sources": {
                    "age_source": "deidentified_medical.src_mbr_age",
                    "gender_source": "patient_data.gender",
                    "diabetes_source": "entity_extraction.diabetics (code meanings based)",
                    "bp_source": "entity_extraction.blood_pressure (code meanings based)",
                    "smoking_source": "entity_extraction.smoking (code meanings based)"
                },
                "extraction_enhanced": True,
                "synchronized_components_used": True
            }

            logger.info(f"✅ Enhanced heart attack features extracted using synchronized components: {enhanced_feature_summary['feature_interpretation']}")
            return enhanced_feature_summary

        except Exception as e:
            logger.error(f"Error in enhanced heart attack feature extraction: {e}")
            return {"error": f"Enhanced feature extraction failed: {str(e)}"}

    def _prepare_enhanced_fastapi_features(self, features: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """Prepare enhanced feature data for FastAPI server call"""
        try:
            # Extract the features from the feature extraction result
            extracted_features = features.get("extracted_features", {})

            # Convert to FastAPI format with proper parameter names
            fastapi_features = {
                "age": int(extracted_features.get("Age", 50)),
                "gender": int(extracted_features.get("Gender", 0)),
                "diabetes": int(extracted_features.get("Diabetes", 0)),
                "high_bp": int(extracted_features.get("High_BP", 0)),
                "smoking": int(extracted_features.get("Smoking", 0))
            }

            # Validate age range
            if fastapi_features["age"] < 0 or fastapi_features["age"] > 120:
                logger.warning(f"Age {fastapi_features['age']} out of range, using default 50")
                fastapi_features["age"] = 50

            # Validate binary features (0 or 1 only)
            binary_features = ["gender", "diabetes", "high_bp", "smoking"]
            for key in binary_features:
                if fastapi_features[key] not in [0, 1]:
                    logger.warning(f"{key} value {fastapi_features[key]} invalid, using 0")
                    fastapi_features[key] = 0

            logger.info(f"✅ FastAPI features prepared: {fastapi_features}")
            return fastapi_features

        except Exception as e:
            logger.error(f"Error preparing FastAPI features: {e}")
            return None

    def _call_heart_attack_prediction_sync(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous heart attack prediction call using synchronized components"""
        try:
            import requests

            logger.info(f"🔍 Received features for prediction: {features}")

            if not features:
                logger.error("❌ No features provided for prediction")
                return {
                    "success": False,
                    "error": "No features provided for heart attack prediction"
                }

            heart_attack_url = self.config.heart_attack_api_url
            logger.info(f"🌐 Using heart attack API URL: {heart_attack_url}")

            endpoints = [
                f"{heart_attack_url}/predict",
                f"{heart_attack_url}/predict-simple"
            ]

            params = {
                "age": int(features.get("age", 50)),
                "gender": int(features.get("gender", 0)),
                "diabetes": int(features.get("diabetes", 0)),
                "high_bp": int(features.get("high_bp", 0)),
                "smoking": int(features.get("smoking", 0))
            }

            logger.info(f"📤 Sending prediction request to {endpoints[0]}")
            logger.info(f"📊 Parameters: {params}")

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

    def _create_comprehensive_predictive_trajectory_prompt(self, medical_data: Dict, pharmacy_data: Dict, mcid_data: Dict,
                                              medical_extraction: Dict, pharmacy_extraction: Dict,
                                              entities: Dict) -> str:
        """Create comprehensive predictive trajectory analysis with evaluation questions"""

        return f"""You are a healthcare AI analyst performing comprehensive predictive analysis using complete deidentified claims data WITH CODE MEANINGS from synchronized components.

ENHANCED EXTRACTED MEDICAL DATA WITH CODE MEANINGS:
{json.dumps(medical_extraction, indent=2)}

ENHANCED EXTRACTED PHARMACY DATA WITH CODE MEANINGS:
{json.dumps(pharmacy_extraction, indent=2)}

ENHANCED HEALTH ENTITIES (BASED ON CODE MEANINGS):
{json.dumps(entities, indent=2)}

PATIENT DEMOGRAPHICS:
- Age: {entities.get('age', 'unknown')} years
- Age Group: {entities.get('age_group', 'unknown')}

Using the complete deidentified claims data WITH CODE MEANINGS from synchronized components, provide a detailed analysis that addresses comprehensive evaluation questions including risk prediction, cost prediction, fraud detection, care management, pharmacy predictions, and strategic planning.

DELIVER A COMPREHENSIVE 800-1000 WORD ANALYSIS using the complete synchronized data with code meanings."""

    # ===== CONNECTION TESTING METHODS =====

    def test_llm_connection(self) -> Dict[str, Any]:
        """Test enhanced Snowflake Cortex API connection"""
        return self.api_integrator.test_llm_connection()

    def test_isolated_llm_connection(self) -> Dict[str, Any]:
        """Test isolated LLM connection for code meanings"""
        return self.api_integrator.test_isolated_llm_connection()

    async def test_ml_connection(self) -> Dict[str, Any]:
        """Test enhanced ML API server connection"""
        return await self.api_integrator.test_ml_connection()

    def test_backend_connection(self) -> Dict[str, Any]:
        """Test MCP backend server connection"""
        return self.api_integrator.test_backend_connection()

    def test_all_connections(self) -> Dict[str, Any]:
        """Test all connections including critical isolated LLM"""
        return self.api_integrator.test_all_connections()

    def run_analysis(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the enhanced health analysis workflow using synchronized components"""

        # Initialize enhanced state for LangGraph
        initial_state = HealthAnalysisState(
            patient_data=patient_data,
            mcid_output={},
            medical_output={},
            pharmacy_output={},
            token_output={},
            deidentified_medical={},
            deidentified_pharmacy={},
            deidentified_mcid={},
            medical_extraction={},
            pharmacy_extraction={},
            entity_extraction={},
            health_trajectory="",
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
            config_dict = {"configurable": {"thread_id": f"enhanced_health_analysis_{datetime.now().timestamp()}"}}

            logger.info("🚀 Starting Enhanced Claims Data Processing with Synchronized Components LangGraph workflow...")

            # Execute the workflow
            final_state = self.graph.invoke(initial_state, config=config_dict)

            # Prepare enhanced results with comprehensive information
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
                    "pharmacy": final_state["deidentified_pharmacy"],
                    "mcid": final_state["deidentified_mcid"]
                },
                "structured_extractions": {
                    "medical": final_state["medical_extraction"],  # WITH CODE MEANINGS
                    "pharmacy": final_state["pharmacy_extraction"]  # WITH CODE MEANINGS
                },
                "entity_extraction": final_state["entity_extraction"],  # ENHANCED WITH CODE MEANINGS
                "health_trajectory": final_state["health_trajectory"],
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
                "mcp_compatible": True,
                "comprehensive_deidentification": True,
                "enhanced_chatbot": True,
                "claims_data_processing": True,
                "code_meanings_integration": True,
                "predictive_analysis_included": True,
                "real_zip_code_access": True,
                "comprehensive_analysis_enabled": True,
                "synchronized_components": True,
                "enhancement_version": "v9.0_synchronized_components_with_complete_code_meanings"
            }

            if results["success"]:
                logger.info("✅ Enhanced Claims Data Processing with Synchronized Components completed successfully!")
                logger.info(f"🔗 Synchronized components: {results['synchronized_components']}")
                logger.info(f"🔒 Comprehensive deidentification: {results['comprehensive_deidentification']}")
                logger.info(f"💬 Enhanced chatbot ready: {results['chatbot_ready']}")
                logger.info(f"🔤 Code meanings integration: {results['code_meanings_integration']}")
                logger.info(f"🎯 Predictive analysis included: {results['predictive_analysis_included']}")
                logger.info(f"📍 Real ZIP code access: {results['real_zip_code_access']}")
                logger.info(f"📈 Comprehensive analysis enabled: {results['comprehensive_analysis_enabled']}")
            else:
                logger.error(f"❌ Enhanced LangGraph analysis failed with errors: {final_state['errors']}")

            return results

        except Exception as e:
            logger.error(f"Fatal error in Enhanced Claims Data Processing with Synchronized Components: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "patient_data": patient_data,
                "errors": [str(e)],
                "processing_steps_completed": 0,
                "langgraph_used": True,
                "mcp_compatible": True,
                "synchronized_components": False,
                "enhancement_version": "v9.0_synchronized_components_with_complete_code_meanings"
            }

    def _count_completed_steps(self, state: HealthAnalysisState) -> int:
        """Count enhanced processing steps completed"""
        steps = 0
        if state.get("mcid_output"): steps += 1
        if state.get("deidentified_medical") and not state.get("deidentified_medical", {}).get("error"): steps += 1
        if state.get("medical_extraction") or state.get("pharmacy_extraction"): steps += 1
        if state.get("entity_extraction"): steps += 1
        if state.get("health_trajectory"): steps += 1
        if state.get("heart_attack_prediction"): steps += 1
        if state.get("chatbot_ready"): steps += 1
        return steps

def main():
    """Example usage of the Enhanced Health Analysis Agent with Synchronized Components"""

    print("🏥 Enhanced Health Analysis Agent v9.0 - Synchronized Components")
    print("✅ Enhanced synchronized architecture with comprehensive features:")
    print("   📡 HealthAPIIntegrator - Clean, enhanced API calls with isolated LLM support")
    print("   🔧 HealthDataProcessor - Comprehensive debug logging + meaning generation")
    print("   🏗️ HealthAnalysisAgent - Synchronized workflow orchestration")
    print("   💬 Enhanced chatbot - Complete access with code meanings")
    print("   🎯 Predictive Analytics - Comprehensive evaluation questions")
    print("   📍 Real ZIP Code Access - Geographic analysis capabilities")
    print("   📊 Comprehensive Analysis - Healthcare scoring and risk assessment")
    print("   🔗 Synchronized Components - All parts working together perfectly")
    print()

    config = Config()
    print("📋 Enhanced Configuration:")
    print(f"   🌐 Snowflake API: {config.api_url}")
    print(f"   🤖 Model: {config.model}")
    print(f"   📡 MCP Server: {config.fastapi_url}")
    print(f"   ❤️ Heart Attack ML API: {config.heart_attack_api_url}")
    print()
    print("✅ Enhanced Health Agent with Synchronized Components ready!")
    print("🚀 Features: Complete code meanings, real zip codes, comprehensive analysis")
    print("🚀 Run: from health_agent_core_updated import HealthAnalysisAgent, Config")

    return "Enhanced Health Agent with Synchronized Components ready for integration"

if __name__ == "__main__":
    main()
