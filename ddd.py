import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, TypedDict, Literal, Optional
from dataclasses import dataclass, asdict
import logging
from  datetime import date
import requests
 
# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
 
# Import our enhanced modular components
from health_api_integrator import HealthAPIIntegrator
from health_data_processor_work import HealthDataProcessor
from health_graph_generator import HealthGraphGenerator
 
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
    model: str =  "llama4-maverick"
    sys_msg: str = """You are an expert healthcare AI analyst with full access to comprehensive deidentified medical and pharmacy claims data. You have been provided with:1. Complete deidentified medical records including ICD-10 diagnosis codes and health service codes2. Complete deidentified pharmacy records including NDC codes and medication details  3. Structured extractions of all medical and pharmacy fields4. Enhanced entity analysis resultsYour role is to provide detailed clinical insights, risk assessments, and evidence-based analysis using ALL the available data. Always reference specific data points from the provided records when making assessments. You HAVE ACCESS to the complete dataset and should use it comprehensively."""
    chatbot_sys_msg: str = """You are a specialized healthcare AI assistant with COMPLETE ACCESS to this patient's deidentified medical and pharmacy claims data. You have been provided with:AVAILABLE DATA SOURCES:✅ Complete deidentified medical records with ICD-10 diagnosis codes✅ Complete deidentified pharmacy records with NDC medication codes  ✅ Health service utilization patterns (hlth_srvc_cd)✅ Complete medication history with NDC codes and label names✅ Diagnosis codes (diag_1_50_cd) from medical claims✅ Enhanced entity extraction results including chronic conditions✅ Health trajectory analysis from previous processingCAPABILITIES:- Analyze specific ICD-10 codes for disease progression and prognosis- Interpret NDC medication codes for treatment adherence and effectiveness- Assess comorbidity burden from diagnosis patterns- Evaluate medication interactions and therapeutic pathways- Provide risk stratification based on available clinical indicators- Estimate health outcomes using evidence-based medical literatureINSTRUCTIONS:1. Always use the specific data provided in your analysis2. Reference exact ICD-10 codes, NDC codes, and medical findings3. Provide evidence-based insights based on the available clinical data4. When asked about prognosis or life expectancy, use available comorbidities, medications, and service utilization patterns5. Be specific about which data points support your conclusions6. Maintain professional medical analysis standards while working with available dataYou have comprehensive access to this patient's healthcare data - use it to provide detailed, professional medical insights."""
    timeout: int = 30
    max_retries: int = 3
 
    # Heart Attack Prediction API Configuration (separate from MCP server)
    heart_attack_api_url: str = "http://localhost:8000"  # Heart attack ML server
    heart_attack_threshold: float = 0.5
 
    def to_dict(self):
        return asdict(self)
 
 
 
# Enhanced State Definition for LangGraph with MCP compatibility
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
 
    # Enhanced extracted structured data
    medical_extraction: Dict[str, Any]
    pharmacy_extraction: Dict[str, Any]
 
    entity_extraction: Dict[str, Any]
 
    # Analysis results (no LLM calls for basic extraction)
    health_trajectory: str
    final_summary: str
 
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
    """Enhanced Health Analysis Agent with Claims Data Processing"""
 
    def __init__(self, custom_config: Optional[Config] = None):
        # Use provided config or create default
        self.config = custom_config or Config()
 
        # Initialize enhanced components
        self.api_integrator = HealthAPIIntegrator(self.config)
        # Pass API integrator to data processor for LLM explanations
        self.data_processor = HealthDataProcessor(self.api_integrator)
        self.graph_generator = HealthGraphGenerator()
 
        logger.info("🔧 Enhanced HealthAnalysisAgent initialized with Claims Data Processing")
        logger.info(f"🌐 Snowflake API URL: {self.config.api_url}")
        logger.info(f"🤖 Model: {self.config.model}")
        logger.info(f"📡 MCP Server URL: {self.config.fastapi_url}")
        logger.info(f"❤️ Heart Attack ML API: {self.config.heart_attack_api_url}")
        logger.info("🔧 Enhanced HealthAnalysisAgent initialized with Graph Capabilities")
        logger.info(f"🎨 Graph generation ready for medical data visualizations")
 
        self.setup_enhanced_langgraph()
 
    def setup_enhanced_langgraph(self):
        """Setup enhanced LangGraph workflow with Claims Data Processing"""
        logger.info("🔧 Setting up Enhanced LangGraph workflow with Claims Data Processing...")
 
        # Create the StateGraph
        workflow = StateGraph(HealthAnalysisState)
 
        # Add all processing nodes
        workflow.add_node("fetch_api_data", self.fetch_api_data)
        workflow.add_node("deidentify_claims_data", self.deidentify_claims_data)
        workflow.add_node("extract_claims_fields", self.extract_claims_fields)
        workflow.add_node("extract_entities", self.extract_entities)
        workflow.add_node("analyze_trajectory", self.analyze_trajectory)
        workflow.add_node("generate_summary", self.generate_summary)
        workflow.add_node("predict_heart_attack", self.predict_heart_attack)
        workflow.add_node("initialize_chatbot", self.initialize_chatbot)
        workflow.add_node("handle_error", self.handle_error)
 
        # Define the enhanced workflow edges
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
                "continue": "generate_summary",
                "error": "handle_error"
            }
        )
 
        workflow.add_conditional_edges(
            "generate_summary",
            self.should_continue_after_summary,
            {
                "continue": "predict_heart_attack",
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
 
        logger.info("✅ Enhanced LangGraph workflow compiled successfully with Claims Data Processing!")
 
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
        """Enhanced LangGraph Node 2: Comprehensive deidentification of all claims data"""
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
        """Enhanced LangGraph Node 3: Extract specific fields from claims data (NO LLM)"""
        logger.info("🔍 Enhanced Node 3: Starting direct claims field extraction (NO LLM)...")
        state["current_step"] = "extract_claims_fields"
        state["step_status"]["extract_claims_fields"] = "running"
 
        try:
            # Direct extraction without LLM calls
            medical_extraction = self.data_processor.extract_medical_fields(state.get("deidentified_medical", {}))
            state["medical_extraction"] = medical_extraction
            logger.info(f"📋 Direct medical extraction: {len(medical_extraction.get('hlth_srvc_records', []))} health service records")
 
            pharmacy_extraction = self.data_processor.extract_pharmacy_fields(state.get("deidentified_pharmacy", {}))
            state["pharmacy_extraction"] = pharmacy_extraction
            logger.info(f"💊 Direct pharmacy extraction: {len(pharmacy_extraction.get('ndc_records', []))} NDC records")
 
            state["step_status"]["extract_claims_fields"] = "completed"
            logger.info("✅ Successfully completed direct claims field extraction (NO LLM)")
 
        except Exception as e:
            error_msg = f"Error in direct claims field extraction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_claims_fields"] = "error"
            logger.error(error_msg)
 
        return state
 
    def extract_entities(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 4: Extract comprehensive health entities using LLM"""
        logger.info("🎯 Enhanced Node 4: Starting LLM-powered health entity extraction...")
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
           
            # Enhanced entity extraction WITH LLM
            entities = self.data_processor.extract_health_entities_enhanced(
                pharmacy_data,
                pharmacy_extraction,
                medical_extraction,
                patient_data,  # Pass patient data for age calculation
                self.api_integrator  # Pass API integrator for LLM calls
            )
           
            state["entity_extraction"] = entities
            state["step_status"]["extract_entities"] = "completed"
           
            # Enhanced logging
            conditions_count = len(entities.get("medical_conditions", []))
            medications_count = len(entities.get("medications_identified", []))
            llm_status = entities.get("llm_analysis", "not_performed")
            age_info = f"Age: {entities.get('age', 'unknown')} ({entities.get('age_group', 'unknown')})"
           
            logger.info(f"✅ Successfully extracted health entities using LLM: {conditions_count} conditions, {medications_count} medications")
            logger.info(f"📊 Entity results: Diabetes={entities.get('diabetics')}, Smoking={entities.get('smoking')}, BP={entities.get('blood_pressure')}")
            logger.info(f"📅 {age_info}")
            logger.info(f"🤖 LLM analysis: {llm_status}")
           
        except Exception as e:
            error_msg = f"Error in LLM-powered entity extraction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_entities"] = "error"
            logger.error(error_msg)
       
        return state
 
    def analyze_trajectory(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 5: Analyze health trajectory with comprehensive claims data"""
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
 
            trajectory_prompt = self._create_comprehensive_trajectory_prompt(
                deidentified_medical, deidentified_pharmacy, deidentified_mcid,
                medical_extraction, pharmacy_extraction, entities
            )
 
            logger.info("🤖 Calling Snowflake Cortex for comprehensive claims trajectory analysis...")
 
            # Use API integrator for LLM call
            response = self.api_integrator.call_llm(trajectory_prompt)
 
            if response.startswith("Error"):
                state["errors"].append(f"Comprehensive trajectory analysis failed: {response}")
                state["step_status"]["analyze_trajectory"] = "error"
            else:
                state["health_trajectory"] = response
                state["step_status"]["analyze_trajectory"] = "completed"
                logger.info("✅ Successfully completed comprehensive claims trajectory analysis")
 
        except Exception as e:
            error_msg = f"Error in comprehensive trajectory analysis: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["analyze_trajectory"] = "error"
            logger.error(error_msg)
 
        return state
 
    def generate_summary(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 6: Generate comprehensive final claims summary"""
        logger.info("📋 Enhanced Node 6: Generating comprehensive final claims summary...")
        state["current_step"] = "generate_summary"
        state["step_status"]["generate_summary"] = "running"
 
        try:
            summary_prompt = self._create_comprehensive_summary_prompt(
                state.get("health_trajectory", ""),
                state.get("entity_extraction", {}),
                state.get("medical_extraction", {}),
                state.get("pharmacy_extraction", {})
            )
 
            logger.info("🤖 Calling Snowflake Cortex for comprehensive final summary...")
 
            # Use API integrator for LLM call
            response = self.api_integrator.call_llm(summary_prompt)
 
            if response.startswith("Error"):
                state["errors"].append(f"Comprehensive summary generation failed: {response}")
                state["step_status"]["generate_summary"] = "error"
            else:
                state["final_summary"] = response
                state["step_status"]["generate_summary"] = "completed"
                logger.info("✅ Successfully generated comprehensive final summary")
 
        except Exception as e:
            error_msg = f"Error in comprehensive summary generation: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["generate_summary"] = "error"
            logger.error(error_msg)
 
        return state  
   
       
   
    def predict_heart_attack(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 7: Enhanced heart attack prediction with FastAPI compatibility"""
        logger.info("❤️ Enhanced Node 7: Starting enhanced heart attack prediction...")
        state["current_step"] = "predict_heart_attack"
        state["step_status"]["predict_heart_attack"] = "running"
 
        try:
             #  Step 1: Extract features using enhanced feature extraction
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
           
                logger.info(f"✅ Enhanced FastAPI heart attack prediction completed successfully")
                logger.info(f"❤️ Display: {enhanced_prediction['combined_display']}")
           
             else:
                  # Handle prediction failure
                error_msg = prediction_result.get("error", "Unknown FastAPI error")
                logger.warning(f"⚠️ Enhanced FastAPI heart attack prediction failed: {error_msg}")
           
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
           error_msg = f"Error in enhanced FastAPI heart attack prediction: {str(e)}"
           state["errors"].append(error_msg)
           state["step_status"]["predict_heart_attack"] = "error"
           logger.error(error_msg)
   
        return state
 
 


 
    def initialize_chatbot(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 8: Initialize comprehensive chatbot with complete deidentified claims context"""
        logger.info("💬 Enhanced Node 8: Initializing comprehensive chatbot with complete claims context...")
        state["current_step"] = "initialize_chatbot"
        state["step_status"]["initialize_chatbot"] = "running"
 
        try:
            # Prepare comprehensive chatbot context with all deidentified claims data
            comprehensive_chatbot_context = {
                "deidentified_medical": state.get("deidentified_medical", {}),
                "deidentified_pharmacy": state.get("deidentified_pharmacy", {}),
                "deidentified_mcid": state.get("deidentified_mcid", {}),
                "medical_extraction": state.get("medical_extraction", {}),
                "pharmacy_extraction": state.get("pharmacy_extraction", {}),
                "entity_extraction": state.get("entity_extraction", {}),
                "health_trajectory": state.get("health_trajectory", ""),
                "final_summary": state.get("final_summary", ""),
                "heart_attack_prediction": state.get("heart_attack_prediction", {}),
                "heart_attack_risk_score": state.get("heart_attack_risk_score", 0.0),
                "heart_attack_features": state.get("heart_attack_features", {}),
                "patient_overview": {
                    "age": state.get("deidentified_medical", {}).get("src_mbr_age", "unknown"),
                    "zip": state.get("deidentified_medical", {}).get("src_mbr_zip_cd", "unknown"),
                    "analysis_timestamp": datetime.now().isoformat(),
                    "heart_attack_risk_level": state.get("heart_attack_prediction", {}).get("risk_category", "unknown"),
                    "model_type": "enhanced_ml_api_mcp_compatible",
                    "deidentification_level": "comprehensive_claims_data",
                    "claims_data_types": ["medical", "pharmacy", "mcid"]
                }
            }
 
            state["chat_history"] = []
            state["chatbot_context"] = comprehensive_chatbot_context
            state["chatbot_ready"] = True
            state["processing_complete"] = True
            state["step_status"]["initialize_chatbot"] = "completed"
 
            # Log comprehensive chatbot initialization
            medical_records = len(state.get("medical_extraction", {}).get("hlth_srvc_records", []))
            pharmacy_records = len(state.get("pharmacy_extraction", {}).get("ndc_records", []))
 
            logger.info("✅ Successfully initialized comprehensive chatbot with complete deidentified claims context")
            logger.info(f"📊 Chatbot context includes: {medical_records} medical records, {pharmacy_records} pharmacy records")
            logger.info(f"🔒 Deidentification level: comprehensive claims data processing")
 
        except Exception as e:
            error_msg = f"Error initializing comprehensive claims chatbot: {str(e)}"
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
 
    def should_continue_after_summary(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"
 
    def should_continue_after_heart_attack_prediction(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"
 
    def _handle_graph_request(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]], graph_request: Dict[str, Any]) -> str:
        """Handle graph generation requests"""
        try:
            graph_type = graph_request.get("graph_type", "timeline")
       
            logger.info(f"📊 Generating {graph_type} visualization for user query: {user_query[:50]}...")
       
        # Generate appropriate graph based on type
            if graph_type == "medication_timeline":
               return self.graph_generator.generate_medication_timeline(chat_context)
            elif graph_type == "diagnosis_timeline":
                return self.graph_generator.generate_diagnosis_timeline(chat_context)
            elif graph_type == "risk_dashboard":
               return self.graph_generator.generate_risk_dashboard(chat_context)
            elif graph_type == "pie":
                return self.graph_generator.generate_medication_distribution(chat_context)
            elif graph_type == "timeline":
            # Default to medication timeline
               return self.graph_generator.generate_medication_timeline(chat_context)
            else:
            # Generate a comprehensive overview
                return self.graph_generator.generate_comprehensive_health_overview(chat_context)
           
        except Exception as e:
          logger.error(f"Error handling graph request: {str(e)}")
          return f"""
##  Graph Generation Error
 
I encountered an error while generating your requested visualization: {str(e)}
 
Available Graph Types:
- Medication Timeline: show me a medication timeline
- Diagnosis Timeline: create a diagnosis timeline chart
- Risk Dashboard: generate a risk assessment dashboard
- Medication Distribution: `show me a pie chart of medications
- Health Overview: show comprehensive health overview
Please try rephrasing your request with one of these specific graph types.
"""

    # ===== ENHANCED CHATBOT FUNCTIONALITY WITH FULL JSON =====
    def chat_with_data(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Enhanced chatbot with COMPLETE deidentified claims data access, heart attack analysis, and graph generation capabilities"""
        try:
                  # FIRST: Check if this is a graph request
                graph_request = self.graph_generator.detect_graph_request(user_query)
       
                if graph_request.get("is_graph_request", False):
                   return self._handle_graph_request(user_query, chat_context, chat_history, graph_request)
       
                # SECOND: Check if this is a heart attack related question
                heart_attack_keywords = ['heart attack', 'heart disease', 'cardiac', 'cardiovascular', 'heart risk', 'coronary', 'myocardial', 'cardiac risk']
                is_heart_attack_question = any(keyword in user_query.lower() for keyword in heart_attack_keywords)
       
                if is_heart_attack_question:
                  return self._handle_heart_attack_question(user_query, chat_context, chat_history)
                else:
                  return self._handle_general_question(user_query, chat_context, chat_history)
       
        except Exception as e:
            logger.error(f"Error in enhanced chatbot with complete deidentified claims data and graph capabilities: {str(e)}")
            return "I encountered an error processing your question. Please try again. I have access to the complete deidentified claims JSON data and can generate visualizations for comprehensive analysis of any aspect of the patient's records."

    def _handle_heart_attack_question(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Handle heart attack related questions with FULL deidentified JSON data"""
        try:
            # Prepare FULL deidentified JSON context
            full_context = self._prepare_full_deidentified_context(chat_context)
            
            # Get prediction data
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            patient_age = chat_context.get("patient_overview", {}).get("age", "unknown")
            risk_display = heart_attack_prediction.get("risk_display", "Not available")

            # Build conversation history
            history_text = self._build_conversation_history(chat_history, limit=3)

            # Create prompt with FULL deidentified JSON
            heart_attack_prompt = f"""You are a cardiologist analyzing heart attack risk with COMPLETE access to deidentified medical and pharmacy claims JSON data.

COMPLETE DEIDENTIFIED CLAIMS DATA:
{full_context}

CURRENT ML MODEL PREDICTION:
{risk_display}

RECENT CONVERSATION:
{history_text}

USER QUESTION: {user_query}

CRITICAL INSTRUCTIONS:
- You have COMPLETE access to all deidentified medical and pharmacy claims JSON data
- Search through the ENTIRE JSON structure to find relevant cardiovascular information
- Look for specific ICD-10 codes, medications, dates, and service codes related to heart disease
- Include exact field names, values, and nested structures from the JSON data
- Reference specific dates (clm_rcvd_dt, rx_filled_dt), NDC codes, ICD codes
- Analyze ALL available data points for comprehensive cardiovascular assessment
- Give precise answers with specific data references from the complete JSON

CARDIOVASCULAR ANALYSIS REQUIREMENTS:
- Provide risk percentage based on complete medical history in JSON
- Compare with ML model prediction
- Reference specific medications, diagnosis codes, and dates from JSON
- Use your medical knowledge to interpret ALL codes and medications found

RESPONSE FORMAT:

🔍 COMPREHENSIVE CARDIOVASCULAR ANALYSIS (Based on Complete Deidentified JSON):
- Risk Percentage: [Your calculated %]%
- Risk Category: [Low/Medium/High Risk]
- Key Risk Factors: [From complete JSON analysis]
- Supporting Evidence: [Specific JSON paths, codes, medications, dates]

📊 COMPARISON WITH ML MODEL:
- ML Prediction: {risk_display}
- LLM Analysis: [Your assessment from complete data]
- Data Concordance: [Agreement/differences and reasoning]

🩺 DETAILED FINDINGS FROM COMPLETE JSON:
[Comprehensive analysis using all available deidentified data]
"""

            logger.info(f"Processing heart attack question with FULL JSON: {user_query[:50]}...")
            logger.info(f"📊 Full context length: {len(full_context)} characters")

            # Check if context is too large and use chunked approach if needed
            if len(full_context) > 15000:  # If too large, use chunked processing
                return self._process_large_context_question(heart_attack_prompt, full_context, "heart_attack")
            else:
                response = self.api_integrator.call_llm(heart_attack_prompt)
                return response if not response.startswith("Error") else "Error analyzing heart attack risk with complete data."

        except Exception as e:
            logger.error(f"Error in heart attack question with full JSON: {str(e)}")
            return "I encountered an error accessing the complete deidentified claims data."

    def _handle_general_question(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Handle general questions with FULL deidentified JSON data"""
        try:
            # Prepare FULL deidentified JSON context
            full_context = self._prepare_full_deidentified_context(chat_context)
            
            # Build conversation history
            history_text = self._build_conversation_history(chat_history, limit=5)

            # Create prompt with FULL deidentified JSON
            general_prompt = f"""You are a medical assistant with COMPLETE access to deidentified medical and pharmacy claims JSON data.

COMPLETE DEIDENTIFIED CLAIMS DATA:
{full_context}

CONVERSATION HISTORY:
{history_text}

USER QUESTION: {user_query}

CRITICAL INSTRUCTIONS:
- You have COMPLETE access to all deidentified medical and pharmacy claims JSON data
- Search through the ENTIRE JSON structure to find information relevant to the question
- Include specific dates, codes, medications, diagnoses, and values from ANY part of the JSON
- Reference exact field names, values, and nested structures from the complete data
- Navigate all levels: medical_claims_data, pharmacy_claims_data, diagnosis_codes, service_codes
- For diagnoses: Reference specific ICD-10 codes, positions, sources, and claim dates
- For medications: Reference NDC codes, label names, fill dates, and prescription details
- For procedures: Reference service codes, dates, and related diagnostic information
- Include exact JSON paths when citing specific data points
- Be thorough and comprehensive using ALL available deidentified data

JSON NAVIGATION GUIDE:
- Medical data: medical_claims_data → [nested structures] → hlth_srvc_cd, diag_codes, clm_rcvd_dt
- Pharmacy data: pharmacy_claims_data → [nested structures] → ndc, lbl_nm, rx_filled_dt
- Patient info: src_mbr_age, src_mbr_zip_cd (all deidentified)
- Diagnosis details: diagnosis_codes array with code, position, source information
- Service records: hlth_srvc_records with complete service and diagnosis information

ANSWER USING COMPLETE DEIDENTIFIED JSON DATA:
Provide a comprehensive answer using the complete deidentified claims data. Reference specific codes, dates, medications, and diagnoses from the JSON structure.
"""

            logger.info(f"Processing general question with FULL JSON: {user_query[:50]}...")
            logger.info(f"📊 Full context length: {len(full_context)} characters")

            # Check if context is too large and use chunked approach if needed
            if len(full_context) > 15000:  # If too large, use chunked processing
                return self._process_large_context_question(general_prompt, full_context, "general")
            else:
                response = self.api_integrator.call_llm(general_prompt)
                return response if not response.startswith("Error") else "Error processing question with complete data."

        except Exception as e:
            logger.error(f"Error in general question with full JSON: {str(e)}")
            return "I encountered an error accessing the complete deidentified claims data."

    def _prepare_full_deidentified_context(self, chat_context: Dict[str, Any]) -> str:
        """Prepare COMPLETE deidentified JSON context for chatbot"""
        try:
            context_sections = []

            # 1. Complete Deidentified Medical Claims JSON
            deidentified_medical = chat_context.get("deidentified_medical", {})
            if deidentified_medical and not deidentified_medical.get("error"):
                # Optimize JSON formatting (remove extra whitespace)
                medical_json = self._optimize_json_formatting(deidentified_medical)
                context_sections.append(f"COMPLETE_DEIDENTIFIED_MEDICAL_CLAIMS:\n{medical_json}")

            # 2. Complete Deidentified Pharmacy Claims JSON  
            deidentified_pharmacy = chat_context.get("deidentified_pharmacy", {})
            if deidentified_pharmacy and not deidentified_pharmacy.get("error"):
                # Optimize JSON formatting (remove extra whitespace)
                pharmacy_json = self._optimize_json_formatting(deidentified_pharmacy)
                context_sections.append(f"COMPLETE_DEIDENTIFIED_PHARMACY_CLAIMS:\n{pharmacy_json}")

            # 3. Complete Medical Extractions (for easy navigation)
            medical_extraction = chat_context.get("medical_extraction", {})
            if medical_extraction and not medical_extraction.get("error"):
                extraction_json = self._optimize_json_formatting(medical_extraction)
                context_sections.append(f"MEDICAL_EXTRACTIONS_STRUCTURED:\n{extraction_json}")

            # 4. Complete Pharmacy Extractions (for easy navigation)
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            if pharmacy_extraction and not pharmacy_extraction.get("error"):
                extraction_json = self._optimize_json_formatting(pharmacy_extraction)
                context_sections.append(f"PHARMACY_EXTRACTIONS_STRUCTURED:\n{extraction_json}")

            # 5. Health Entities (for context)
            entity_extraction = chat_context.get("entity_extraction", {})
            if entity_extraction:
                entity_json = self._optimize_json_formatting(entity_extraction)
                context_sections.append(f"HEALTH_ENTITIES_ANALYSIS:\n{entity_json}")

            # 6. Patient Overview (deidentified)
            patient_overview = chat_context.get("patient_overview", {})
            if patient_overview:
                overview_json = self._optimize_json_formatting(patient_overview)
                context_sections.append(f"PATIENT_OVERVIEW_DEIDENTIFIED:\n{overview_json}")

            return "\n\n".join(context_sections)

        except Exception as e:
            logger.error(f"Error preparing full deidentified context: {e}")
            return "Complete deidentified claims data available but could not be formatted."

    def _optimize_json_formatting(self, data: Dict[str, Any]) -> str:
        """Optimize JSON formatting to reduce size while maintaining readability"""
        try:
            # Use compact JSON formatting (no extra spaces, minimal indentation)
            return json.dumps(data, separators=(',', ':'), ensure_ascii=False)
        except Exception as e:
            logger.warning(f"JSON optimization failed: {e}")
            return json.dumps(data, indent=1)  # Fallback to minimal indentation

    def _build_conversation_history(self, chat_history: List[Dict[str, str]], limit: int = 5) -> str:
        """Build formatted conversation history"""
        if not chat_history:
            return "No previous conversation"
        
        recent_history = chat_history[-limit:]
        history_lines = []
        for msg in recent_history:
            role = "User" if msg['role'] == 'user' else "Assistant"
            content = msg['content'][:200]  # Limit individual message length
            history_lines.append(f"{role}: {content}")
        
        return "\n".join(history_lines)

    def _process_large_context_question(self, prompt: str, full_context: str, question_type: str) -> str:
        """Process questions with large context using chunked approach"""
        try:
            logger.info(f"🔄 Using chunked processing for large context ({len(full_context)} chars)")
            
            # Split context into chunks
            chunks = self._split_context_into_chunks(full_context, max_chunk_size=10000)
            
            # Process each chunk to find relevant information
            relevant_findings = []
            for i, chunk in enumerate(chunks):
                chunk_prompt = f"""Analyze this section of deidentified claims data for information relevant to the user question.

CLAIMS DATA SECTION {i+1}/{len(chunks)}:
{chunk}

USER QUESTION: {prompt.split('USER QUESTION:')[1].split('CRITICAL')[0].strip()}

Extract ONLY relevant findings from this section:
- Relevant medications with NDC codes and dates
- Relevant diagnosis codes with dates  
- Relevant service codes with dates
- Any other pertinent information

If no relevant information in this section, respond with "No relevant data in this section."
"""
                
                chunk_response = self.api_integrator.call_llm(chunk_prompt)
                if chunk_response and not chunk_response.startswith("Error") and "No relevant data" not in chunk_response:
                    relevant_findings.append(f"Section {i+1}: {chunk_response}")

            # Synthesize findings
            if relevant_findings:
                synthesis_prompt = f"""Based on the relevant findings from the complete deidentified claims data, provide a comprehensive answer.

RELEVANT FINDINGS FROM COMPLETE CLAIMS DATA:
{chr(10).join(relevant_findings)}

USER QUESTION: {prompt.split('USER QUESTION:')[1].split('CRITICAL')[0].strip()}

Provide a comprehensive answer using all the relevant findings from the complete deidentified claims data.
"""
                
                return self.api_integrator.call_llm(synthesis_prompt)
            else:
                return "No relevant information found in the complete deidentified claims data for your question."

        except Exception as e:
            logger.error(f"Error in chunked processing: {e}")
            return "Error processing your question with the complete claims data."

    def _split_context_into_chunks(self, context: str, max_chunk_size: int = 10000) -> List[str]:
        """Split large context into manageable chunks"""
        try:
            chunks = []
            lines = context.split('\n')
            current_chunk = ""
            
            for line in lines:
                if len(current_chunk + line + '\n') > max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = line + '\n'
                    else:
                        # Single line too long, split it
                        chunks.append(line[:max_chunk_size])
                        current_chunk = line[max_chunk_size:] + '\n'
                else:
                    current_chunk += line + '\n'
            
            if current_chunk:
                chunks.append(current_chunk)
            
            logger.info(f"📦 Split context into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error splitting context: {e}")
            return [context[:max_chunk_size]]  # Fallback

    def _create_comprehensive_trajectory_prompt(self, medical_data: Dict, pharmacy_data: Dict, mcid_data: Dict,
                                              medical_extraction: Dict, pharmacy_extraction: Dict,
                                              entities: Dict) -> str:
        """Create trajectory analysis prompt with FULL deidentified JSON"""
        
        # Prepare complete deidentified context
        complete_context = {
            "deidentified_medical_claims": medical_data,
            "deidentified_pharmacy_claims": pharmacy_data,
            "deidentified_mcid_claims": mcid_data,
            "medical_extractions": medical_extraction,
            "pharmacy_extractions": pharmacy_extraction,
            "health_entities": entities
        }
        
        # Optimize formatting
        context_json = self._optimize_json_formatting(complete_context)

        return f"""You are a healthcare AI assistant analyzing a patient's health trajectory with COMPLETE access to deidentified claims data.

COMPLETE DEIDENTIFIED CLAIMS DATA:
{context_json}

ANALYSIS REQUIREMENTS:
You have complete access to all deidentified medical and pharmacy claims JSON data. Analyze this patient's health trajectory by examining:

1. **Current Health Status**: Comprehensive assessment using all ICD-10 codes, medications, and service codes
2. **Risk Factors**: Complete analysis of all diagnosis codes and medication patterns
3. **Medication Analysis**: All NDC codes, drug names, fill dates, and therapeutic patterns
4. **Chronic Conditions**: Long-term conditions identified from complete claims history
5. **Health Trends**: Trajectory analysis using all available dates and service utilization
6. **Care Coordination**: Integration of medical and pharmacy claims patterns
7. **Comprehensive Assessment**: Synthesis of all available deidentified data

CRITICAL INSTRUCTIONS:
- Use ALL available data in the complete deidentified JSON structure
- Reference specific ICD-10 codes, NDC codes, service codes, and dates
- Navigate through all nested JSON structures for comprehensive analysis
- Include exact dates (clm_rcvd_dt, rx_filled_dt) and code sequences
- Analyze patterns across all available claims data
- Provide evidence-based assessment using the complete dataset

Provide a detailed trajectory analysis (600-800 words) using the complete deidentified claims data.
"""

    def _create_comprehensive_summary_prompt(self, trajectory_analysis: str, entities: Dict,
                                           medical_extraction: Dict, pharmacy_extraction: Dict) -> str:
        """Create summary prompt with FULL extraction data"""

        # Include complete extraction data
        complete_extractions = {
            "health_entities": entities,
            "medical_extractions": medical_extraction,
            "pharmacy_extractions": pharmacy_extraction
        }
        
        extractions_json = self._optimize_json_formatting(complete_extractions)

        return f"""Based on the health trajectory analysis and complete extraction data, create a comprehensive executive summary.

HEALTH TRAJECTORY ANALYSIS:
{trajectory_analysis}

COMPLETE EXTRACTION DATA:
{extractions_json}

SUMMARY REQUIREMENTS:
Create a comprehensive summary using ALL available extraction data:

1. **Health Status Overview**: Based on complete entity and extraction analysis
2. **Key Risk Factors**: From all diagnosis codes and medication patterns
3. **Priority Recommendations**: Using complete claims analysis
4. **Follow-up Needs**: Based on comprehensive service utilization patterns
5. **Care Coordination**: Integration recommendations from complete data

CRITICAL INSTRUCTIONS:
- Use the complete extraction data provided
- Reference specific counts, codes, and patterns from the full dataset
- Integrate findings from medical and pharmacy extractions
- Provide actionable insights based on comprehensive analysis

Create a detailed summary (400-500 words) using the complete extraction data.
"""
 
    # ===== ENHANCED HELPER METHODS =====
 
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
 
            # Enhanced feature extraction from comprehensive entity extraction
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
                    "diabetes_source": "entity_extraction.diabetics",
                    "bp_source": "entity_extraction.blood_pressure",
                    "smoking_source": "entity_extraction.smoking"
                },
                "extraction_enhanced": True
            }
 
            logger.info(f"✅ Enhanced heart attack features extracted: {enhanced_feature_summary['feature_interpretation']}")
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
        """Synchronous heart attack prediction call to avoid event loop conflicts"""
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

    def test_llm_connection(self) -> Dict[str, Any]:
        """Test enhanced Snowflake Cortex API connection"""
        return self.api_integrator.test_llm_connection()
 
    async def test_ml_connection(self) -> Dict[str, Any]:
        """Test enhanced ML API server connection"""
        return await self.api_integrator.test_ml_connection()
 
    def test_backend_connection(self) -> Dict[str, Any]:
        """Test MCP backend server connection"""
        return self.api_integrator.test_backend_connection()
 
    def run_analysis(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the enhanced health analysis workflow using LangGraph with Claims Data Processing"""
 
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
            final_summary="",
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
 
            logger.info("🚀 Starting Enhanced Claims Data Processing LangGraph workflow...")
 
            # Execute the workflow without step simulation
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
                    "medical": final_state["medical_extraction"],
                    "pharmacy": final_state["pharmacy_extraction"]
                },
                "entity_extraction": final_state["entity_extraction"],
                "health_trajectory": final_state["health_trajectory"],
                "final_summary": final_state["final_summary"],
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
                "full_json_chat_enabled": True,
                "enhancement_version": "v6.1_full_json_chat"
            }
 
            if results["success"]:
                logger.info("✅ Enhanced Claims Data Processing LangGraph analysis completed successfully!")
                logger.info(f"🔒 Comprehensive claims deidentification: {results['comprehensive_deidentification']}")
                logger.info(f"💬 Enhanced chatbot with full JSON access ready: {results['chatbot_ready']}")
            else:
                logger.error(f"❌ Enhanced LangGraph analysis failed with errors: {final_state['errors']}")
 
            return results
 
        except Exception as e:
            logger.error(f"Fatal error in Enhanced Claims Data Processing LangGraph workflow: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "patient_data": patient_data,
                "errors": [str(e)],
                "processing_steps_completed": 0,
                "langgraph_used": True,
                "mcp_compatible": True,
                "comprehensive_deidentification": False,
                "enhanced_chatbot": False,
                "claims_data_processing": False,
                "full_json_chat_enabled": False,
                "enhancement_version": "v6.1_full_json_chat"
            }
 
    def _count_completed_steps(self, state: HealthAnalysisState) -> int:
        """Count enhanced processing steps completed"""
        steps = 0
        if state.get("mcid_output"): steps += 1
        if state.get("deidentified_medical") and not state.get("deidentified_medical", {}).get("error"): steps += 1
        if state.get("medical_extraction") or state.get("pharmacy_extraction"): steps += 1
        if state.get("entity_extraction"): steps += 1
        if state.get("health_trajectory"): steps += 1
        if state.get("final_summary"): steps += 1
        if state.get("heart_attack_prediction"): steps += 1
        if state.get("chatbot_ready"): steps += 1
        return steps
 
def main():
    """Example usage of the Enhanced Claims Data Processing Health Analysis Agent with Full JSON Chat"""
 
    print("🏥 Enhanced Claims Data Processing Health Analysis Agent v6.1")
    print("✅ Enhanced modular architecture with comprehensive features:")
    print("   📡 HealthAPIIntegrator - MCP server compatible API calls")
    print("   🔧 HealthDataProcessor - Comprehensive claims data deidentification + Code Explanations")
    print("   🏗️ HealthAnalysisAgent - Enhanced workflow orchestration")
    print("   💬 Enhanced chatbot - COMPLETE deidentified claims JSON data access")
    print("   📊 Graph generation - Medical data visualizations")
    print()
 
    config = Config()
    print("📋 Enhanced Configuration:")
    print(f"   🌐 Snowflake API: {config.api_url}")
    print(f"   🤖 Model: {config.model}")
    print(f"   📡 MCP Server: {config.fastapi_url}")
    print(f"   ❤️ Heart Attack ML API: {config.heart_attack_api_url}")
    print()
    print("✅ Enhanced Claims Data Processing Health Agent with Full JSON Chat ready!")
    print("🚀 Run: from health_agent_core import HealthAnalysisAgent, Config")
 
    return "Enhanced Claims Data Processing Health Agent with Full JSON Chat ready for integration"
 
if __name__ == "__main__":
    main()
