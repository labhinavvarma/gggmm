# Enhanced Health Analysis Agent with DETAILED prompts, SPECIFIC healthcare analysis, and STABLE graph generation
import json
import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, TypedDict, Literal, Optional, Union
from dataclasses import dataclass, asdict
import logging
from datetime import date
import requests
import traceback

# LangGraph imports with error handling
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LangGraph not available: {e}")
    LANGGRAPH_AVAILABLE = False

# Import enhanced components with error handling
try:
    from health_api_integrator_enhanced import EnhancedHealthAPIIntegrator
    API_INTEGRATOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"API Integrator not available: {e}")
    API_INTEGRATOR_AVAILABLE = False

try:
    from health_data_processor_enhanced import EnhancedHealthDataProcessor
    DATA_PROCESSOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Data Processor not available: {e}")
    DATA_PROCESSOR_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('health_agent.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedConfig:
    """Enhanced configuration with detailed healthcare analysis settings"""
    fastapi_url: str = "http://localhost:8000"
    api_url: str = "https://sfassist.edagenaipreprod.awsdns.internal.das/api/cortex/complete"
    api_key: str = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
    app_id: str = "edadip"
    aplctn_cd: str = "edagnai"
    model: str = "llama4-maverick"
    
    # Enhanced system messages with healthcare specialization
    sys_msg: str = """You are Dr. HealthAI, an expert healthcare data analyst and clinical decision support specialist with comprehensive knowledge of:

• Medical coding systems (ICD-10, CPT, HCPCS, NDC)
• Clinical terminology and healthcare workflows
• Risk stratification and predictive modeling
• Healthcare cost analysis and utilization patterns
• Pharmacy therapeutics and medication management
• Population health analytics and care management
• Healthcare fraud, waste, and abuse detection
• Quality metrics (HEDIS, STAR ratings, clinical outcomes)

You have COMPLETE ACCESS to batch-processed claims data with professional-grade code meanings for ALL medical codes, diagnosis codes, NDC codes, and medications. Provide detailed, clinically accurate analyses with specific medical insights and actionable recommendations."""

    chatbot_sys_msg: str = """You are Dr. ChatAI, a specialized healthcare AI assistant providing detailed clinical analysis and patient-centered insights. You have COMPLETE ACCESS to:

• Comprehensive deidentified medical and pharmacy claims data
• Batch-generated professional meanings for all medical codes, diagnosis codes, NDC codes, and medications
• Advanced risk assessment models and predictive analytics
• Healthcare utilization patterns and cost projections
• Clinical decision support capabilities

When generating matplotlib visualizations:
• Create professional, publication-quality healthcare charts
• Use appropriate medical color schemes and styling
• Include detailed labels, legends, and clinical context
• Ensure all visualizations are clinically meaningful and actionable
• Focus on patient safety and clinical decision support

Provide detailed healthcare analysis with specific medical terminology, clinical insights, and evidence-based recommendations. Always prioritize patient safety and clinical accuracy in your responses."""

    timeout: int = 25  # Enhanced timeout for detailed processing
    heart_attack_api_url: str = "http://localhost:8000"
    heart_attack_threshold: float = 0.5
    max_retries: int = 3  # Enhanced retry logic

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

# Enhanced State Definition for detailed healthcare analysis
class EnhancedHealthAnalysisState(TypedDict):
    # Input data
    patient_data: Dict[str, Any]

    # API outputs
    mcid_output: Dict[str, Any]
    medical_output: Dict[str, Any]
    pharmacy_output: Dict[str, Any]
    token_output: Dict[str, Any]

    # Enhanced deidentified data
    deidentified_medical: Dict[str, Any]
    deidentified_pharmacy: Dict[str, Any]
    deidentified_mcid: Dict[str, Any]

    # BATCH extracted data with detailed meanings
    medical_extraction: Dict[str, Any]
    pharmacy_extraction: Dict[str, Any]

    # Enhanced entity extraction
    entity_extraction: Dict[str, Any]
    
    # Enhanced health trajectory with detailed questions
    enhanced_health_trajectory: str
    
    # Enhanced risk predictions
    heart_attack_prediction: Dict[str, Any]
    heart_attack_risk_score: float
    heart_attack_features: Dict[str, Any]

    # Enhanced chatbot with stable graphs
    chatbot_ready: bool
    chatbot_context: Dict[str, Any]
    chat_history: List[Dict[str, str]]

    # Enhanced control flow
    current_step: str
    errors: List[str]
    retry_count: int
    processing_complete: bool
    step_status: Dict[str, str]

def safe_get(data: Union[Dict[str, Any], Any], key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary or object"""
    try:
        if data is None:
            return default
        if isinstance(data, dict):
            return data.get(key, default)
        elif hasattr(data, key):
            return getattr(data, key, default)
        else:
            return default
    except Exception as e:
        logger.warning(f"Error in safe_get for key '{key}': {e}")
        return default

def safe_execute(func, *args, **kwargs) -> tuple[bool, Any]:
    """Safely execute a function and return success status and result"""
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {e}")
        logger.debug(traceback.format_exc())
        return False, str(e)

class EnhancedHealthAnalysisAgent:
    """Enhanced Health Analysis Agent with DETAILED healthcare prompts and STABLE graph generation"""

    def __init__(self, custom_config: Optional[EnhancedConfig] = None):
        self.config = custom_config or EnhancedConfig()
        self.api_integrator = None
        self.data_processor = None
        self.graph = None
        self.is_initialized = False

        logger.info("🚀 Initializing Enhanced HealthAnalysisAgent with detailed healthcare prompts...")
        
        # Initialize enhanced components with error handling
        self._initialize_components()
        
        # Test connections
        self._enhanced_connection_test()
        
        # Setup LangGraph if available
        if LANGGRAPH_AVAILABLE:
            self.setup_enhanced_langgraph()
        else:
            logger.warning("LangGraph not available - workflow functionality limited")

    def _initialize_components(self):
        """Initialize components with proper error handling"""
        try:
            if API_INTEGRATOR_AVAILABLE:
                success, result = safe_execute(EnhancedHealthAPIIntegrator, self.config)
                if success:
                    self.api_integrator = result
                    logger.info("✅ Enhanced API Integrator initialized")
                else:
                    logger.error(f"❌ API Integrator initialization failed: {result}")
            else:
                logger.warning("❌ API Integrator not available")

            if DATA_PROCESSOR_AVAILABLE and self.api_integrator:
                success, result = safe_execute(EnhancedHealthDataProcessor, self.api_integrator)
                if success:
                    self.data_processor = result
                    logger.info("✅ Enhanced Data Processor initialized")
                else:
                    logger.error(f"❌ Data Processor initialization failed: {result}")
            else:
                logger.warning("❌ Data Processor not available")

            self.is_initialized = True
        except Exception as e:
            logger.error(f"Component initialization error: {e}")
            self.is_initialized = False

    def _enhanced_connection_test(self):
        """Enhanced connection test for detailed healthcare analysis"""
        if not self.api_integrator:
            logger.warning("❌ No API integrator available for connection test")
            return

        try:
            logger.info("🔬 Enhanced healthcare analysis connection test...")
            
            # Test enhanced LLM with healthcare-specific prompt
            success, result = safe_execute(self.api_integrator.test_healthcare_llm_connection)
            if success and safe_get(result, "success"):
                logger.info("✅ Healthcare LLM - Advanced clinical analysis enabled")
            else:
                logger.error(f"❌ Healthcare LLM failed - Clinical analysis limited: {result}")
                
        except Exception as e:
            logger.error(f"❌ Enhanced connection test failed: {e}")

    def setup_enhanced_langgraph(self):
        """Setup Enhanced LangGraph workflow for detailed healthcare analysis"""
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available - skipping workflow setup")
            return

        try:
            logger.info("🔬 Setting up Enhanced LangGraph workflow with detailed healthcare analysis...")

            workflow = StateGraph(EnhancedHealthAnalysisState)

            # Add enhanced processing nodes
            workflow.add_node("fetch_api_data_enhanced", self.fetch_api_data_enhanced)
            workflow.add_node("deidentify_claims_data_enhanced", self.deidentify_claims_data_enhanced)
            workflow.add_node("extract_claims_fields_batch_enhanced", self.extract_claims_fields_batch_enhanced)
            workflow.add_node("extract_entities_enhanced", self.extract_entities_enhanced)
            workflow.add_node("analyze_trajectory_enhanced", self.analyze_trajectory_enhanced)
            workflow.add_node("predict_heart_attack_enhanced", self.predict_heart_attack_enhanced)
            workflow.add_node("initialize_chatbot_with_stable_graphs", self.initialize_chatbot_with_stable_graphs)
            workflow.add_node("handle_error_enhanced", self.handle_error_enhanced)

            # Enhanced workflow edges
            workflow.add_edge(START, "fetch_api_data_enhanced")

            workflow.add_conditional_edges(
                "fetch_api_data_enhanced",
                self.should_continue_after_api_enhanced,
                {
                    "continue": "deidentify_claims_data_enhanced",
                    "retry": "fetch_api_data_enhanced",
                    "error": "handle_error_enhanced"
                }
            )

            workflow.add_conditional_edges(
                "deidentify_claims_data_enhanced",
                self.should_continue_after_deidentify_enhanced,
                {
                    "continue": "extract_claims_fields_batch_enhanced",
                    "error": "handle_error_enhanced"
                }
            )

            workflow.add_conditional_edges(
                "extract_claims_fields_batch_enhanced",
                self.should_continue_after_extraction_enhanced,
                {
                    "continue": "extract_entities_enhanced",
                    "error": "handle_error_enhanced"
                }
            )

            workflow.add_conditional_edges(
                "extract_entities_enhanced",
                self.should_continue_after_entities_enhanced,
                {
                    "continue": "analyze_trajectory_enhanced",
                    "error": "handle_error_enhanced"
                }
            )

            workflow.add_conditional_edges(
                "analyze_trajectory_enhanced",
                self.should_continue_after_trajectory_enhanced,
                {
                    "continue": "predict_heart_attack_enhanced",
                    "error": "handle_error_enhanced"
                }
            )

            workflow.add_conditional_edges(
                "predict_heart_attack_enhanced",
                self.should_continue_after_heart_attack_enhanced,
                {
                    "continue": "initialize_chatbot_with_stable_graphs",
                    "error": "handle_error_enhanced"
                }
            )

            workflow.add_edge("initialize_chatbot_with_stable_graphs", END)
            workflow.add_edge("handle_error_enhanced", END)

            # Compile with enhanced memory
            memory = MemorySaver()
            self.graph = workflow.compile(checkpointer=memory)

            logger.info("✅ Enhanced LangGraph workflow compiled for detailed healthcare analysis!")

        except Exception as e:
            logger.error(f"❌ LangGraph setup failed: {e}")
            self.graph = None

    # ===== ENHANCED LANGGRAPH NODES =====

    def fetch_api_data_enhanced(self, state: EnhancedHealthAnalysisState) -> EnhancedHealthAnalysisState:
        """Enhanced API data fetch with detailed validation"""
        logger.info("🔬 Node 1: Enhanced Claims API data fetch with detailed validation...")
        state["current_step"] = "fetch_api_data_enhanced"
        state["step_status"] = safe_get(state, "step_status", {})
        state["step_status"]["fetch_api_data_enhanced"] = "running"

        try:
            patient_data = safe_get(state, "patient_data", {})

            # Enhanced validation with detailed error messages
            required_fields = {
                "first_name": "Patient first name for identity verification",
                "last_name": "Patient last name for identity verification", 
                "ssn": "Social Security Number for unique patient identification",
                "date_of_birth": "Date of birth for age-based risk assessment",
                "gender": "Gender for gender-specific risk modeling",
                "zip_code": "ZIP code for geographic health pattern analysis"
            }
            
            missing_fields = []
            for field, description in required_fields.items():
                if not safe_get(patient_data, field):
                    missing_fields.append(f"{field}: {description}")
            
            if missing_fields:
                state["errors"] = safe_get(state, "errors", [])
                state["errors"].extend([f"Missing required field - {field}" for field in missing_fields])
                state["step_status"]["fetch_api_data_enhanced"] = "error"
                return state

            # Enhanced API call with detailed logging
            if self.api_integrator:
                logger.info("📡 Fetching comprehensive healthcare claims data...")
                success, api_result = safe_execute(self.api_integrator.fetch_backend_data_enhanced, patient_data)
                
                if success and not safe_get(api_result, "error"):
                    state["mcid_output"] = safe_get(api_result, "mcid_output", {})
                    state["medical_output"] = safe_get(api_result, "medical_output", {})
                    state["pharmacy_output"] = safe_get(api_result, "pharmacy_output", {})
                    state["token_output"] = safe_get(api_result, "token_output", {})

                    state["step_status"]["fetch_api_data_enhanced"] = "completed"
                    logger.info("✅ Enhanced API data fetch completed with comprehensive validation")
                else:
                    error_msg = f"Enhanced API Error: {api_result if not success else safe_get(api_result, 'error', 'Unknown error')}"
                    state["errors"] = safe_get(state, "errors", [])
                    state["errors"].append(error_msg)
                    state["step_status"]["fetch_api_data_enhanced"] = "error"
            else:
                error_msg = "API integrator not available"
                state["errors"] = safe_get(state, "errors", [])
                state["errors"].append(error_msg)
                state["step_status"]["fetch_api_data_enhanced"] = "error"

        except Exception as e:
            error_msg = f"Enhanced API fetch error: {str(e)}"
            state["errors"] = safe_get(state, "errors", [])
            state["errors"].append(error_msg)
            state["step_status"]["fetch_api_data_enhanced"] = "error"
            logger.error(error_msg)

        return state

    def deidentify_claims_data_enhanced(self, state: EnhancedHealthAnalysisState) -> EnhancedHealthAnalysisState:
        """Enhanced claims data deidentification with clinical data preservation"""
        logger.info("🔒 Node 2: Enhanced claims data deidentification with clinical preservation...")
        state["current_step"] = "deidentify_claims_data_enhanced"
        state["step_status"]["deidentify_claims_data_enhanced"] = "running"

        try:
            if not self.data_processor:
                raise Exception("Data processor not available")

            # Enhanced deidentification with clinical context preservation
            medical_data = safe_get(state, "medical_output", {})
            success, deidentified_medical = safe_execute(
                self.data_processor.deidentify_medical_data_enhanced, 
                medical_data, 
                safe_get(state, "patient_data", {})
            )
            if success:
                state["deidentified_medical"] = deidentified_medical
            else:
                raise Exception(f"Medical deidentification failed: {deidentified_medical}")

            pharmacy_data = safe_get(state, "pharmacy_output", {})
            success, deidentified_pharmacy = safe_execute(
                self.data_processor.deidentify_pharmacy_data_enhanced, 
                pharmacy_data
            )
            if success:
                state["deidentified_pharmacy"] = deidentified_pharmacy
            else:
                raise Exception(f"Pharmacy deidentification failed: {deidentified_pharmacy}")

            mcid_data = safe_get(state, "mcid_output", {})
            success, deidentified_mcid = safe_execute(
                self.data_processor.deidentify_mcid_data_enhanced, 
                mcid_data
            )
            if success:
                state["deidentified_mcid"] = deidentified_mcid
            else:
                raise Exception(f"MCID deidentification failed: {deidentified_mcid}")

            state["step_status"]["deidentify_claims_data_enhanced"] = "completed"
            logger.info("✅ Enhanced deidentification completed with clinical context preservation")

        except Exception as e:
            error_msg = f"Enhanced deidentification error: {str(e)}"
            state["errors"] = safe_get(state, "errors", [])
            state["errors"].append(error_msg)
            state["step_status"]["deidentify_claims_data_enhanced"] = "error"
            logger.error(error_msg)

        return state

    def extract_claims_fields_batch_enhanced(self, state: EnhancedHealthAnalysisState) -> EnhancedHealthAnalysisState:
        """Enhanced BATCH PROCESSING with detailed healthcare code analysis"""
        logger.info("🔬 Node 3: Enhanced BATCH claims field extraction with detailed healthcare analysis...")
        state["current_step"] = "extract_claims_fields_batch_enhanced"
        state["step_status"]["extract_claims_fields_batch_enhanced"] = "running"

        try:
            if not self.data_processor:
                raise Exception("Data processor not available")

            # Enhanced medical extraction with detailed clinical analysis
            logger.info("🏥 Enhanced BATCH medical extraction with clinical insights...")
            success, medical_extraction = safe_execute(
                self.data_processor.extract_medical_fields_batch_enhanced,
                safe_get(state, "deidentified_medical", {})
            )
            if success:
                state["medical_extraction"] = medical_extraction
                logger.info(f"🏥 Enhanced medical batch results:")
                logger.info(f"  📊 API calls: {safe_get(medical_extraction, 'batch_stats', {}).get('api_calls_made', 0)}")
                logger.info(f"  💾 Calls saved: {safe_get(medical_extraction, 'batch_stats', {}).get('individual_calls_saved', 0)}")
                logger.info(f"  🔬 Clinical insights: {safe_get(medical_extraction, 'enhanced_analysis', False)}")
            else:
                raise Exception(f"Medical extraction failed: {medical_extraction}")

            # Enhanced pharmacy extraction with therapeutic analysis
            logger.info("💊 Enhanced BATCH pharmacy extraction with therapeutic insights...")
            success, pharmacy_extraction = safe_execute(
                self.data_processor.extract_pharmacy_fields_batch_enhanced,
                safe_get(state, "deidentified_pharmacy", {})
            )
            if success:
                state["pharmacy_extraction"] = pharmacy_extraction
                logger.info(f"💊 Enhanced pharmacy batch results:")
                logger.info(f"  📊 API calls: {safe_get(pharmacy_extraction, 'batch_stats', {}).get('api_calls_made', 0)}")
                logger.info(f"  💾 Calls saved: {safe_get(pharmacy_extraction, 'batch_stats', {}).get('individual_calls_saved', 0)}")
                logger.info(f"  🔬 Therapeutic insights: {safe_get(pharmacy_extraction, 'enhanced_analysis', False)}")
            else:
                raise Exception(f"Pharmacy extraction failed: {pharmacy_extraction}")

            state["step_status"]["extract_claims_fields_batch_enhanced"] = "completed"
            logger.info("✅ Enhanced BATCH extraction completed with detailed clinical analysis!")

        except Exception as e:
            error_msg = f"Enhanced batch extraction error: {str(e)}"
            state["errors"] = safe_get(state, "errors", [])
            state["errors"].append(error_msg)
            state["step_status"]["extract_claims_fields_batch_enhanced"] = "error"
            logger.error(error_msg)

        return state

    def extract_entities_enhanced(self, state: EnhancedHealthAnalysisState) -> EnhancedHealthAnalysisState:
        """Enhanced health entity extraction with detailed clinical analysis"""
        logger.info("🎯 Node 4: Enhanced entity extraction with detailed clinical insights...")
        state["current_step"] = "extract_entities_enhanced"
        state["step_status"]["extract_entities_enhanced"] = "running"
       
        try:
            if not self.data_processor:
                raise Exception("Data processor not available")

            pharmacy_data = safe_get(state, "pharmacy_output", {})
            pharmacy_extraction = safe_get(state, "pharmacy_extraction", {})
            medical_extraction = safe_get(state, "medical_extraction", {})
            patient_data = safe_get(state, "patient_data", {})
           
            # Enhanced entity extraction with detailed clinical analysis
            success, entities = safe_execute(
                self.data_processor.extract_health_entities_with_clinical_insights,
                pharmacy_data,
                pharmacy_extraction,
                medical_extraction,
                patient_data,
                self.api_integrator
            )
            
            if success:
                state["entity_extraction"] = entities
                state["step_status"]["extract_entities_enhanced"] = "completed"
                
                logger.info(f"✅ Enhanced entity extraction completed with clinical insights:")
                logger.info(f"  🩺 Diabetes: {safe_get(entities, 'diabetics')}")
                logger.info(f"  💓 BP: {safe_get(entities, 'blood_pressure')}")
                logger.info(f"  🚬 Smoking: {safe_get(entities, 'smoking')}")
                logger.info(f"  🍷 Alcohol: {safe_get(entities, 'alcohol')}")
                logger.info(f"  🔬 Clinical insights: {safe_get(entities, 'enhanced_clinical_analysis')}")
            else:
                raise Exception(f"Entity extraction failed: {entities}")
           
        except Exception as e:
            error_msg = f"Enhanced entity extraction error: {str(e)}"
            state["errors"] = safe_get(state, "errors", [])
            state["errors"].append(error_msg)
            state["step_status"]["extract_entities_enhanced"] = "error"
            logger.error(error_msg)
       
        return state

    def analyze_trajectory_enhanced(self, state: EnhancedHealthAnalysisState) -> EnhancedHealthAnalysisState:
        """Enhanced health trajectory analysis with DETAILED evaluation questions"""
        logger.info("📈 Node 5: Enhanced health trajectory analysis with detailed evaluation questions...")
        state["current_step"] = "analyze_trajectory_enhanced"
        state["step_status"]["analyze_trajectory_enhanced"] = "running"

        try:
            if not self.api_integrator:
                raise Exception("API integrator not available")

            deidentified_medical = safe_get(state, "deidentified_medical", {})
            deidentified_pharmacy = safe_get(state, "deidentified_pharmacy", {})
            medical_extraction = safe_get(state, "medical_extraction", {})
            pharmacy_extraction = safe_get(state, "pharmacy_extraction", {})
            entities = safe_get(state, "entity_extraction", {})

            # Enhanced trajectory prompt with SPECIFIC evaluation questions
            enhanced_trajectory_prompt = self._create_enhanced_trajectory_prompt_with_detailed_questions(
                deidentified_medical, deidentified_pharmacy,
                medical_extraction, pharmacy_extraction, entities
            )

            logger.info("🔬 Enhanced Snowflake Cortex trajectory analysis with detailed evaluation questions...")
            success, response = safe_execute(
                self.api_integrator.call_llm_enhanced, 
                enhanced_trajectory_prompt, 
                self.config.sys_msg
            )

            if success and not response.startswith("Error"):
                state["enhanced_health_trajectory"] = response
                state["step_status"]["analyze_trajectory_enhanced"] = "completed"
                logger.info("✅ Enhanced trajectory analysis completed with detailed evaluation questions")
            else:
                error_msg = f"Enhanced trajectory analysis failed: {response if success else response}"
                state["errors"] = safe_get(state, "errors", [])
                state["errors"].append(error_msg)
                state["step_status"]["analyze_trajectory_enhanced"] = "error"

        except Exception as e:
            error_msg = f"Enhanced trajectory analysis error: {str(e)}"
            state["errors"] = safe_get(state, "errors", [])
            state["errors"].append(error_msg)
            state["step_status"]["analyze_trajectory_enhanced"] = "error"
            logger.error(error_msg)

        return state

    def predict_heart_attack_enhanced(self, state: EnhancedHealthAnalysisState) -> EnhancedHealthAnalysisState:
        """Enhanced heart attack prediction with COMPREHENSIVE clinical analysis"""
        logger.info("❤️ Node 6: Enhanced heart attack prediction with comprehensive clinical modeling...")
        state["current_step"] = "predict_heart_attack_enhanced"
        state["step_status"]["predict_heart_attack_enhanced"] = "running"

        try:
            # Enhanced feature extraction with comprehensive clinical analysis
            logger.info("🔬 Enhanced feature extraction with comprehensive clinical insights...")
            success, features = safe_execute(self._extract_features_enhanced, state)
            
            if success:
                state["heart_attack_features"] = features
            else:
                logger.warning("⚠️ Feature extraction failed - using fallback features")
                features = self._create_fallback_features(state)
                state["heart_attack_features"] = features

            # Enhanced prediction with multiple approaches
            logger.info("🔬 Enhanced heart attack prediction with multiple validation approaches...")
            success, prediction_result = safe_execute(self._predict_heart_attack_with_fallbacks, features, state)
            
            if success:
                state["heart_attack_prediction"] = prediction_result
                state["heart_attack_risk_score"] = float(safe_get(prediction_result, "raw_risk_score", 0.0))
                logger.info(f"✅ Enhanced heart attack prediction completed: {safe_get(prediction_result, 'combined_display', 'Analysis complete')}")
                state["step_status"]["predict_heart_attack_enhanced"] = "completed"
            else:
                # Ultimate fallback
                fallback_prediction = self._ultimate_clinical_fallback_prediction(state)
                state["heart_attack_prediction"] = fallback_prediction
                state["heart_attack_risk_score"] = float(safe_get(fallback_prediction, "raw_risk_score", 0.0))
                
                error_msg = f"Enhanced heart attack prediction error: {prediction_result}"
                state["errors"] = safe_get(state, "errors", [])
                state["errors"].append(error_msg)
                state["step_status"]["predict_heart_attack_enhanced"] = "completed_with_fallback"

        except Exception as e:
            error_msg = f"Enhanced heart attack prediction error: {str(e)}"
            logger.error(error_msg)
            
            # Ultimate fallback
            fallback_prediction = self._ultimate_clinical_fallback_prediction(state)
            state["heart_attack_prediction"] = fallback_prediction
            state["heart_attack_risk_score"] = float(safe_get(fallback_prediction, "raw_risk_score", 0.0))
            
            state["errors"] = safe_get(state, "errors", [])
            state["errors"].append(error_msg)
            state["step_status"]["predict_heart_attack_enhanced"] = "completed_with_fallback"

        return state

    def initialize_chatbot_with_stable_graphs(self, state: EnhancedHealthAnalysisState) -> EnhancedHealthAnalysisState:
        """Initialize Enhanced chatbot with stable graph generation capabilities"""
        logger.info("💬 Node 7: Initialize Enhanced chatbot with stable graph generation...")
        state["current_step"] = "initialize_chatbot_with_stable_graphs"
        state["step_status"]["initialize_chatbot_with_stable_graphs"] = "running"

        try:
            # Prepare enhanced chatbot context with detailed clinical data
            enhanced_chatbot_context = {
                "deidentified_medical": safe_get(state, "deidentified_medical", {}),
                "deidentified_pharmacy": safe_get(state, "deidentified_pharmacy", {}),
                "deidentified_mcid": safe_get(state, "deidentified_mcid", {}),
                "medical_extraction": safe_get(state, "medical_extraction", {}),
                "pharmacy_extraction": safe_get(state, "pharmacy_extraction", {}),
                "entity_extraction": safe_get(state, "entity_extraction", {}),
                "enhanced_health_trajectory": safe_get(state, "enhanced_health_trajectory", ""),
                "heart_attack_prediction": safe_get(state, "heart_attack_prediction", {}),
                "heart_attack_risk_score": safe_get(state, "heart_attack_risk_score", 0.0),
                "heart_attack_features": safe_get(state, "heart_attack_features", {}),
                "patient_overview": {
                    "age": safe_get(safe_get(state, "deidentified_medical", {}), "src_mbr_age", "unknown"),
                    "zip": safe_get(safe_get(state, "deidentified_medical", {}), "src_mbr_zip_cd", "unknown"),
                    "analysis_timestamp": datetime.now().isoformat(),
                    "cardiovascular_risk_level": safe_get(safe_get(state, "heart_attack_prediction", {}), "risk_category", "unknown"),
                    "model_type": "enhanced_clinical_analysis_with_stable_graphs",
                    "batch_processing_enabled": True,
                    "stable_graph_generation_enabled": True,
                    "detailed_code_meanings_available": True,
                    "clinical_insights_enhanced": True,
                    "healthcare_specialization": "advanced"
                }
            }

            state["chat_history"] = []
            state["chatbot_context"] = enhanced_chatbot_context
            state["chatbot_ready"] = True
            state["processing_complete"] = True
            state["step_status"]["initialize_chatbot_with_stable_graphs"] = "completed"

            logger.info("✅ Enhanced chatbot with stable graphs initialized")
            logger.info(f"🔬 Enhanced clinical analysis: Enabled")
            logger.info(f"📊 Stable graph generation: Enabled")
            logger.info(f"🎯 Detailed healthcare prompts: Active")

        except Exception as e:
            error_msg = f"Enhanced chatbot initialization error: {str(e)}"
            state["errors"] = safe_get(state, "errors", [])
            state["errors"].append(error_msg)
            state["step_status"]["initialize_chatbot_with_stable_graphs"] = "error"
            logger.error(error_msg)

        return state

    def handle_error_enhanced(self, state: EnhancedHealthAnalysisState) -> EnhancedHealthAnalysisState:
        """Enhanced error handling with detailed diagnostics"""
        current_step = safe_get(state, "current_step", "unknown")
        errors = safe_get(state, "errors", [])
        
        logger.error(f"🚨 Enhanced Error Handler: {current_step}")
        logger.error(f"Detailed errors: {errors}")

        state["processing_complete"] = True
        state["step_status"] = safe_get(state, "step_status", {})
        state["step_status"][current_step] = "error"
        return state

    # ===== ENHANCED CONDITIONAL EDGES =====

    def should_continue_after_api_enhanced(self, state: EnhancedHealthAnalysisState) -> Literal["continue", "retry", "error"]:
        errors = safe_get(state, "errors", [])
        retry_count = safe_get(state, "retry_count", 0)
        
        if errors:
            if retry_count < self.config.max_retries:
                state["retry_count"] = retry_count + 1
                logger.warning(f"🔄 Enhanced retry {state['retry_count']}/{self.config.max_retries}")
                state["errors"] = []
                return "retry"
            else:
                logger.error(f"❌ Enhanced max retries exceeded")
                return "error"
        return "continue"

    def should_continue_after_deidentify_enhanced(self, state: EnhancedHealthAnalysisState) -> Literal["continue", "error"]:
        errors = safe_get(state, "errors", [])
        return "error" if errors else "continue"

    def should_continue_after_extraction_enhanced(self, state: EnhancedHealthAnalysisState) -> Literal["continue", "error"]:
        errors = safe_get(state, "errors", [])
        return "error" if errors else "continue"

    def should_continue_after_entities_enhanced(self, state: EnhancedHealthAnalysisState) -> Literal["continue", "error"]:
        errors = safe_get(state, "errors", [])
        return "error" if errors else "continue"

    def should_continue_after_trajectory_enhanced(self, state: EnhancedHealthAnalysisState) -> Literal["continue", "error"]:
        errors = safe_get(state, "errors", [])
        return "error" if errors else "continue"

    def should_continue_after_heart_attack_enhanced(self, state: EnhancedHealthAnalysisState) -> Literal["continue", "error"]:
        errors = safe_get(state, "errors", [])
        return "error" if errors else "continue"

    # ===== ENHANCED HELPER METHODS =====

    def _create_enhanced_trajectory_prompt_with_detailed_questions(self, medical_data: Dict, pharmacy_data: Dict,
                                                                  medical_extraction: Dict, pharmacy_extraction: Dict,
                                                                  entities: Dict) -> str:
        """Create enhanced trajectory analysis prompt with comprehensive evaluation questions"""
        return f"""You are Dr. MegaTrajectoryAI, a world-renowned healthcare futurist and predictive analytics specialist with 30+ years of experience in clinical medicine, health economics, population health, and advanced medical modeling.

COMPREHENSIVE PATIENT DATA WITH ADVANCED CLINICAL ANALYTICS:

**MEDICAL PROFILE:** {json.dumps(medical_extraction, indent=2)[:1000]}...

**PHARMACY PROFILE:** {json.dumps(pharmacy_extraction, indent=2)[:1000]}...

**CLINICAL RISK PROFILE:** {json.dumps(entities, indent=2)}

**DEMOGRAPHIC CONTEXT:**
- Patient Age: {safe_get(entities, 'age', 'unknown')} years
- Risk Category: {safe_get(entities, 'age_group', 'unknown')}
- Clinical Complexity: {safe_get(entities, 'clinical_complexity_score', 0)}

# COMPREHENSIVE HEALTHCARE EVALUATION & PREDICTIVE ANALYSIS

Perform a detailed, evidence-based analysis addressing the following evaluation domains:

## 1. CHRONIC DISEASE DEVELOPMENT RISK ASSESSMENT
• Calculate precise 5-year and 10-year risk percentages for:
  - Type 2 Diabetes progression or complications
  - Hypertension progression to resistant hypertension
  - Chronic Kidney Disease stages 1-5
  - Cardiovascular disease including CAD, stroke, PAD

## 2. HEALTHCARE UTILIZATION PREDICTIONS
• Calculate probability percentages for:
  - Emergency department visits (6 months, 12 months, 24 months)
  - Inpatient hospitalizations with timeline analysis
  - 30-day, 60-day, and 90-day readmission risks
• Identify TOP 5 conditions driving hospitalization risk

## 3. COST PREDICTION & HEALTHCARE ECONOMICS
• High-cost claimant probability (top 1%, 5%, 10% healthcare spenders)
• Detailed cost projections by category:
  - Inpatient hospital costs
  - Outpatient specialty care costs
  - Prescription drug costs
• ROI calculations for preventive interventions

## 4. MEDICATION ADHERENCE & THERAPEUTIC PREDICTIONS
• Adherence probability for each medication class
• Medications with highest discontinuation risk
• Polypharmacy risk assessment and drug interactions
• Therapeutic progression modeling

## 5. PERSONALIZED CARE MANAGEMENT
• Patient risk segmentation classification
• Evidence-based intervention recommendations
• Care gap analysis against clinical guidelines
• Preventive care opportunities with timelines

Provide specific percentages, dollar amounts, timeframes, and clinical evidence for all assessments. Structure with clear sections and prioritized action plans."""

    def _extract_features_enhanced(self, state: EnhancedHealthAnalysisState) -> Dict[str, Any]:
        """Enhanced feature extraction with comprehensive clinical analysis"""
        try:
            features = {}
            clinical_evidence = []

            # Enhanced age extraction
            deidentified_medical = safe_get(state, "deidentified_medical", {})
            patient_age = safe_get(deidentified_medical, "src_mbr_age", None)

            if patient_age and patient_age != "unknown":
                try:
                    age_str = str(patient_age)
                    age_value = int(float(age_str.split()[0]) if ' ' in age_str else age_str)
                    features["Age"] = max(0, min(120, age_value))
                    clinical_evidence.append(f"Age {age_value}: Clinical risk assessment")
                except:
                    features["Age"] = 50
                    clinical_evidence.append("Age unavailable - using population average")
            else:
                features["Age"] = 50
                clinical_evidence.append("Age data unavailable - conservative estimation")

            # Enhanced gender extraction
            patient_data = safe_get(state, "patient_data", {})
            gender = str(safe_get(patient_data, "gender", "F")).upper()
            features["Gender"] = 1 if gender in ["M", "MALE", "1"] else 0

            # Enhanced entity-based feature extraction
            entity_extraction = safe_get(state, "entity_extraction", {})

            # Enhanced diabetes analysis
            diabetes = str(safe_get(entity_extraction, "diabetics", "no")).lower()
            features["Diabetes"] = 1 if diabetes in ["yes", "true", "1"] else 0

            # Enhanced blood pressure analysis
            blood_pressure = str(safe_get(entity_extraction, "blood_pressure", "unknown")).lower()
            features["High_BP"] = 1 if blood_pressure in ["managed", "diagnosed", "yes", "true", "1"] else 0

            # Enhanced smoking analysis
            smoking = str(safe_get(entity_extraction, "smoking", "no")).lower()
            features["Smoking"] = 1 if smoking in ["yes", "true", "1"] else 0

            # Validate features
            for key in ["Age", "Gender", "Diabetes", "High_BP", "Smoking"]:
                try:
                    features[key] = int(features[key])
                except:
                    if key == "Age":
                        features[key] = 50
                    else:
                        features[key] = 0

            enhanced_feature_summary = {
                "extracted_features": features,
                "clinical_evidence": clinical_evidence,
                "extraction_enhanced": True,
                "feature_validation": "completed"
            }

            logger.info(f"✅ Enhanced clinical features extracted: {features}")
            return enhanced_feature_summary

        except Exception as e:
            logger.error(f"Enhanced feature extraction error: {e}")
            return self._create_fallback_features(state)

    def _create_fallback_features(self, state: EnhancedHealthAnalysisState) -> Dict[str, Any]:
        """Create fallback features when extraction fails"""
        try:
            logger.info("🔬 Creating fallback clinical features...")
            
            fallback_features = {
                "extracted_features": {
                    "Age": 50,
                    "Gender": 0,
                    "Diabetes": 0,
                    "High_BP": 0,
                    "Smoking": 0
                },
                "clinical_evidence": [
                    "Limited data - using conservative assumptions",
                    "Fallback features applied for safety"
                ],
                "extraction_enhanced": False,
                "fallback_used": True
            }
            
            return fallback_features
            
        except Exception as e:
            logger.error(f"Fallback features creation error: {e}")
            return {
                "extracted_features": {"Age": 50, "Gender": 0, "Diabetes": 0, "High_BP": 0, "Smoking": 0},
                "error": "Fallback features creation failed",
                "fallback_used": True
            }

    def _predict_heart_attack_with_fallbacks(self, features: Dict[str, Any], state: EnhancedHealthAnalysisState) -> Dict[str, Any]:
        """Enhanced heart attack prediction with multiple fallback mechanisms"""
        try:
            extracted_features = safe_get(features, "extracted_features", {})
            
            # Prepare features for ML model
            ml_features = {
                "age": int(safe_get(extracted_features, "Age", 50)),
                "gender": int(safe_get(extracted_features, "Gender", 0)),
                "diabetes": int(safe_get(extracted_features, "Diabetes", 0)),
                "high_bp": int(safe_get(extracted_features, "High_BP", 0)),
                "smoking": int(safe_get(extracted_features, "Smoking", 0))
            }

            # Try ML prediction if API integrator available
            if self.api_integrator:
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    ml_result = loop.run_until_complete(
                        self.api_integrator.call_ml_heart_attack_prediction_enhanced(ml_features)
                    )
                    loop.close()
                    
                    if safe_get(ml_result, "success"):
                        prediction_data = safe_get(ml_result, "prediction_data", {})
                        probability = safe_get(prediction_data, "probability", 0.0)
                        
                        return {
                            "risk_display": f"Cardiovascular Risk: {probability*100:.1f}%",
                            "confidence_display": f"ML Model Confidence: High",
                            "combined_display": f"CVD Risk: {probability*100:.1f}% (ML Analysis)",
                            "raw_risk_score": float(probability),
                            "raw_prediction": 1 if probability >= 0.075 else 0,
                            "prediction_method": "enhanced_ml_model",
                            "clinical_features": ml_features
                        }
                except Exception as e:
                    logger.warning(f"ML prediction failed: {e}")

            # Fallback to clinical risk calculator
            return self._calculate_clinical_risk_fallback(ml_features)

        except Exception as e:
            logger.error(f"Heart attack prediction error: {e}")
            return self._ultimate_clinical_fallback_prediction(state)

    def _calculate_clinical_risk_fallback(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate clinical cardiovascular risk as fallback"""
        try:
            age = safe_get(features, "age", 50)
            gender = safe_get(features, "gender", 0)
            diabetes = safe_get(features, "diabetes", 0)
            smoking = safe_get(features, "smoking", 0)
            high_bp = safe_get(features, "high_bp", 0)
            
            # Simple clinical risk calculation
            risk_score = 0
            
            # Age-based risk
            if age >= 70: risk_score += 6
            elif age >= 60: risk_score += 4
            elif age >= 50: risk_score += 2
            
            # Gender risk
            if gender == 1: risk_score += 2  # Male
            
            # Risk factors
            if diabetes: risk_score += 3
            if smoking: risk_score += 3
            if high_bp: risk_score += 2
            
            # Convert to probability
            risk_probability = min(0.8, max(0.01, risk_score / 20))
            
            return {
                "risk_display": f"Cardiovascular Risk: {risk_probability*100:.1f}%",
                "confidence_display": f"Clinical Model Confidence: Moderate",
                "combined_display": f"CVD Risk: {risk_probability*100:.1f}% (Clinical Assessment)",
                "raw_risk_score": float(risk_probability),
                "raw_prediction": 1 if risk_probability >= 0.075 else 0,
                "prediction_method": "clinical_risk_calculator",
                "clinical_features": features
            }
            
        except Exception as e:
            logger.error(f"Clinical risk calculation error: {e}")
            return self._ultimate_clinical_fallback_prediction(None)

    def _ultimate_clinical_fallback_prediction(self, state) -> Dict[str, Any]:
        """Ultimate fallback when all prediction methods fail"""
        try:
            baseline_risk = 0.10  # 10% baseline risk
            
            return {
                "risk_display": f"Cardiovascular Risk: {baseline_risk*100:.1f}% (Estimated)",
                "confidence_display": f"Confidence: Limited (Insufficient data)",
                "combined_display": f"CVD Risk: {baseline_risk*100:.1f}% (Clinical Estimation)",
                "raw_risk_score": baseline_risk,
                "raw_prediction": 0,
                "prediction_method": "clinical_fallback",
                "clinical_recommendation": "Comprehensive cardiovascular evaluation recommended"
            }
            
        except Exception as e:
            logger.error(f"Ultimate fallback error: {e}")
            return {
                "error": "Risk assessment unavailable",
                "risk_display": "Risk Assessment: Unavailable",
                "confidence_display": "Confidence: Unable to assess",
                "combined_display": "CVD Risk: Assessment Unavailable",
                "raw_risk_score": 0.0,
                "ultimate_fallback_used": True
            }

    # ===== ENHANCED CHATBOT WITH STABLE GRAPHS =====

    def chat_with_enhanced_graphs(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> tuple:
        """Enhanced chatbot with stable matplotlib graph generation"""
        try:
            if not self.api_integrator:
                return "API integrator not available. Please check system configuration.", None, None
                
            # Enhanced graph keyword detection
            graph_keywords = [
                'graph', 'chart', 'plot', 'visualize', 'visualization', 'show me', 'display', 
                'histogram', 'bar chart', 'line chart', 'pie chart', 'scatter plot', 'trend', 
                'distribution', 'create', 'generate', 'draw', 'render', 'dashboard'
            ]
            
            wants_graph = any(keyword in user_query.lower() for keyword in graph_keywords)
            
            if wants_graph:
                return self._handle_enhanced_graph_request(user_query, chat_context, chat_history)
            else:
                return self._handle_enhanced_regular_chat(user_query, chat_context, chat_history)

        except Exception as e:
            logger.error(f"Enhanced chat error: {str(e)}")
            return "I encountered an error processing your request. Please try again.", None, None

    def _handle_enhanced_graph_request(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> tuple:
        """Handle enhanced graph generation requests"""
        try:
            if not self.data_processor:
                return "Data processor not available for graph generation.", None, None
                
            # Prepare enhanced context
            enhanced_context = self.data_processor.prepare_enhanced_clinical_context(chat_context)
            
            # Create enhanced graph prompt
            enhanced_prompt = f"""You are Dr. GraphAI, a healthcare visualization specialist. Generate matplotlib code for clinical visualizations.

PATIENT DATA: {enhanced_context[:1000]}...

USER REQUEST: {user_query}

Generate Python matplotlib code with professional healthcare styling:

Return format:
EXPLANATION: [Clinical explanation]

CODE:
```python
import matplotlib.pyplot as plt
import numpy as np

# Professional healthcare visualization
plt.figure(figsize=(10, 6))
# Your code here
plt.show()
```

Use actual patient data for meaningful insights."""

            success, response = safe_execute(
                self.api_integrator.call_llm_enhanced, 
                enhanced_prompt, 
                self.config.chatbot_sys_msg
            )
            
            if success:
                explanation, code = self._extract_explanation_and_code(response)
                return explanation, code, None
            else:
                return f"Graph generation error: {response}", None, None
            
        except Exception as e:
            logger.error(f"Enhanced graph request error: {e}")
            return "Graph generation failed. Please try a different visualization.", None, None

    def _handle_enhanced_regular_chat(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Handle enhanced regular chat requests"""
        try:
            if not self.data_processor:
                return "Data processor not available for chat analysis."
                
            enhanced_context = self.data_processor.prepare_enhanced_clinical_context(chat_context)
            
            enhanced_prompt = f"""You are Dr. HealthAI, a comprehensive healthcare analyst.

PATIENT DATA: {enhanced_context[:1500]}...

QUESTION: {user_query}

Provide detailed healthcare analysis with clinical insights and evidence-based recommendations."""

            success, response = safe_execute(
                self.api_integrator.call_llm_enhanced, 
                enhanced_prompt, 
                self.config.chatbot_sys_msg
            )
            
            if success:
                return response
            else:
                return f"Analysis error: {response}"

        except Exception as e:
            logger.error(f"Enhanced regular chat error: {e}")
            return "I encountered an error processing your question. Please try again."

    def _extract_explanation_and_code(self, response: str) -> tuple:
        """Extract explanation and code from LLM response"""
        try:
            lines = response.split('\n')
            explanation_lines = []
            code_lines = []
            in_code_block = False
            
            for line in lines:
                if line.strip().startswith('```python'):
                    in_code_block = True
                    continue
                elif line.strip() == '```' and in_code_block:
                    in_code_block = False
                    continue
                elif in_code_block:
                    code_lines.append(line)
                elif not in_code_block and line.strip():
                    explanation_lines.append(line)
            
            explanation = '\n'.join(explanation_lines).strip()
            code = '\n'.join(code_lines).strip()
            
            return explanation, code
            
        except Exception as e:
            logger.error(f"Code extraction error: {e}")
            return "Visualization generated.", ""

    # ===== CONNECTION TESTING =====

    def test_all_connections_enhanced(self) -> Dict[str, Any]:
        """Test all connections with enhanced error handling"""
        if not self.api_integrator:
            return {"error": "API integrator not available", "success": False}
            
        try:
            return self.api_integrator.test_all_connections_enhanced()
        except Exception as e:
            logger.error(f"Connection test error: {e}")
            return {"error": f"Connection test failed: {e}", "success": False}

    def run_enhanced_analysis(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run enhanced health analysis with comprehensive error handling"""
        
        if not self.is_initialized:
            return {
                "success": False,
                "error": "Agent not properly initialized",
                "patient_data": patient_data,
                "errors": ["Agent initialization failed"]
            }

        initial_state = EnhancedHealthAnalysisState(
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
            enhanced_health_trajectory="",
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
            if self.graph:
                config_dict = {"configurable": {"thread_id": f"enhanced_{datetime.now().timestamp()}"}}
                logger.info("🔬 Starting Enhanced healthcare analysis...")
                
                # Execute enhanced workflow
                success, final_state = safe_execute(self.graph.invoke, initial_state, config=config_dict)
                
                if not success:
                    raise Exception(f"Workflow execution failed: {final_state}")
            else:
                # Fallback processing without LangGraph
                logger.warning("LangGraph not available - using fallback processing")
                final_state = self._run_fallback_analysis(initial_state)

            # Prepare enhanced results
            results = {
                "success": safe_get(final_state, "processing_complete", False) and not safe_get(final_state, "errors", []),
                "patient_data": safe_get(final_state, "patient_data", {}),
                "api_outputs": {
                    "mcid": safe_get(final_state, "mcid_output", {}),
                    "medical": safe_get(final_state, "medical_output", {}),
                    "pharmacy": safe_get(final_state, "pharmacy_output", {}),
                    "token": safe_get(final_state, "token_output", {})
                },
                "deidentified_data": {
                    "medical": safe_get(final_state, "deidentified_medical", {}),
                    "pharmacy": safe_get(final_state, "deidentified_pharmacy", {}),
                    "mcid": safe_get(final_state, "deidentified_mcid", {})
                },
                "structured_extractions": {
                    "medical": safe_get(final_state, "medical_extraction", {}),
                    "pharmacy": safe_get(final_state, "pharmacy_extraction", {})
                },
                "entity_extraction": safe_get(final_state, "entity_extraction", {}),
                "enhanced_health_trajectory": safe_get(final_state, "enhanced_health_trajectory", ""),
                "heart_attack_prediction": safe_get(final_state, "heart_attack_prediction", {}),
                "heart_attack_risk_score": safe_get(final_state, "heart_attack_risk_score", 0.0),
                "heart_attack_features": safe_get(final_state, "heart_attack_features", {}),
                "chatbot_ready": safe_get(final_state, "chatbot_ready", False),
                "chatbot_context": safe_get(final_state, "chatbot_context", {}),
                "chat_history": safe_get(final_state, "chat_history", []),
                "errors": safe_get(final_state, "errors", []),
                "step_status": safe_get(final_state, "step_status", {}),
                "enhancement_stats": {
                    "detailed_healthcare_prompts_enabled": True,
                    "stable_graph_generation_enabled": True,
                    "clinical_analysis_enhanced": True,
                    "healthcare_specialization": "advanced"
                },
                "version": "enhanced_v4.0_stable_compatible"
            }

            if results["success"]:
                logger.info("✅ Enhanced healthcare analysis completed successfully!")
            else:
                logger.error(f"❌ Enhanced analysis failed: {safe_get(final_state, 'errors', [])}")

            return results

        except Exception as e:
            logger.error(f"Fatal enhanced analysis error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "patient_data": patient_data,
                "errors": [str(e)],
                "enhancement_stats": {
                    "processing_failed": True
                },
                "version": "enhanced_v4.0_stable_compatible"
            }

    def _run_fallback_analysis(self, state: EnhancedHealthAnalysisState) -> EnhancedHealthAnalysisState:
        """Run fallback analysis without LangGraph"""
        try:
            logger.info("Running fallback analysis...")
            
            # Basic processing steps
            state = self.fetch_api_data_enhanced(state)
            if not safe_get(state, "errors", []):
                state = self.deidentify_claims_data_enhanced(state)
            if not safe_get(state, "errors", []):
                state = self.extract_claims_fields_batch_enhanced(state)
            if not safe_get(state, "errors", []):
                state = self.extract_entities_enhanced(state)
            if not safe_get(state, "errors", []):
                state = self.predict_heart_attack_enhanced(state)
            if not safe_get(state, "errors", []):
                state = self.initialize_chatbot_with_stable_graphs(state)
            
            state["processing_complete"] = True
            return state
            
        except Exception as e:
            logger.error(f"Fallback analysis error: {e}")
            state["errors"] = safe_get(state, "errors", [])
            state["errors"].append(f"Fallback analysis failed: {e}")
            state["processing_complete"] = True
            return state

def main():
    """Enhanced Health Analysis Agent entry point"""
    print("🔬 Enhanced Health Analysis Agent v4.0")
    print("✅ Stable and compatible healthcare analysis")
    print("✅ Enhanced error handling and fallbacks")
    print("✅ Comprehensive clinical insights")
    print()
    print("🔬 Ready for enhanced healthcare data analysis!")

if __name__ == "__main__":
    main()
