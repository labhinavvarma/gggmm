import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, TypedDict, Literal, Optional
from dataclasses import dataclass, asdict
import logging

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import our enhanced modular components
from health_api_integrator import HealthAPIIntegrator
from health_data_processor import HealthDataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    fastapi_url: str = "http://localhost:8001"  # MCP server URL
    # Snowflake Cortex API Configuration
    api_url: str = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
    api_key: str = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
    app_id: str = "edadip"
    aplctn_cd: str = "edagnai"
    model: str = "llama3.1-70b"
    sys_msg: str = "You are a healthcare AI assistant. Provide accurate, concise answers based on context."
    chatbot_sys_msg: str = "You are a powerful healthcare AI assistant with access to comprehensive deidentified medical records and heart attack risk predictions. Provide accurate, detailed analysis based on the complete medical and pharmacy data provided. Always maintain patient privacy and provide professional medical insights."
    max_retries: int = 3
    timeout: int = 30
    
    # Heart Attack Prediction API Configuration (separate from MCP server)
    heart_attack_api_url: str = "http://localhost:8080"  # Heart attack FastAPI server
    heart_attack_threshold: float = 0.5
    
    def to_dict(self):
        return asdict(self)

# Enhanced State Definition for LangGraph with MCP compatibility
class HealthAnalysisState(TypedDict):
    # Input data
    patient_data: Dict[str, Any]
    
    # Enhanced API outputs with MCP compatibility including MCID
    mcid_output: Dict[str, Any]
    medical_output: Dict[str, Any]
    pharmacy_output: Dict[str, Any]
    token_output: Dict[str, Any]
    
    # Enhanced processed data with comprehensive deidentification including MCID
    deidentified_mcid: Dict[str, Any]
    deidentified_medical: Dict[str, Any]
    raw_pharmacy: Dict[str, Any]  # Pharmacy data kept raw, no deidentification needed
    
    # Enhanced extracted structured data
    medical_extraction: Dict[str, Any]
    pharmacy_extraction: Dict[str, Any]
    
    entity_extraction: Dict[str, Any]
    
    # Analysis results
    health_trajectory: str
    final_summary: str
    
    # Enhanced Heart Attack Prediction via FastAPI
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
    """Enhanced Health Analysis Agent with MCP server compatibility and comprehensive features"""
    
    def __init__(self, custom_config: Optional[Config] = None):
        # Use provided config or create default
        self.config = custom_config or Config()
        
        # Initialize enhanced components
        self.api_integrator = HealthAPIIntegrator(self.config)
        self.data_processor = HealthDataProcessor()
        
        logger.info("🔧 Enhanced HealthAnalysisAgent initialized with MCP compatibility")
        logger.info(f"🌐 Snowflake API URL: {self.config.api_url}")
        logger.info(f"🤖 Model: {self.config.model}")
        logger.info(f"📡 MCP Server URL: {self.config.fastapi_url}")
        logger.info(f"❤️ Heart Attack API: {self.config.heart_attack_api_url}")
        
        self.setup_enhanced_langgraph()
    
    def setup_enhanced_langgraph(self):
        """Setup enhanced LangGraph workflow with MCP compatibility"""
        logger.info("🔧 Setting up Enhanced LangGraph workflow with MCP compatibility...")
        
        # Create the StateGraph
        workflow = StateGraph(HealthAnalysisState)
        
        # Add all processing nodes
        workflow.add_node("fetch_api_data", self.fetch_api_data)
        workflow.add_node("deidentify_data", self.deidentify_data)
        workflow.add_node("extract_medical_pharmacy_data", self.extract_medical_pharmacy_data)
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
        
        logger.info("✅ Enhanced LangGraph workflow compiled successfully with MCP compatibility!")
    
    # ===== ENHANCED LANGGRAPH NODES =====
    
    def fetch_api_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 1: Fetch data from MCP-compatible APIs"""
        logger.info("🚀 Enhanced Node 1: Starting MCP-compatible API data fetch...")
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
                state["errors"].append(f"MCP API Error: {api_result['error']}")
                state["step_status"]["fetch_api_data"] = "error"
            else:
                state["mcid_output"] = api_result.get("mcid_output", {})
                state["medical_output"] = api_result.get("medical_output", {})
                state["pharmacy_output"] = api_result.get("pharmacy_output", {})
                state["token_output"] = api_result.get("token_output", {})
                
                state["step_status"]["fetch_api_data"] = "completed"
                logger.info("✅ Successfully fetched all MCP-compatible API data")
                
        except Exception as e:
            error_msg = f"Error fetching MCP API data: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["fetch_api_data"] = "error"
            logger.error(error_msg)
        
        return state
    
    def deidentify_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 2: Comprehensive deidentification for MCID and Medical data only"""
        logger.info("🔒 Enhanced Node 2: Starting comprehensive nested JSON deidentification (MCID + Medical only)...")
        state["current_step"] = "deidentify_data"
        state["step_status"]["deidentify_data"] = "running"
        
        try:
            # Deidentify MCID data
            mcid_data = state.get("mcid_output", {})
            deidentified_mcid = self.data_processor.deidentify_mcid_data(mcid_data)
            state["deidentified_mcid"] = deidentified_mcid
            
            # Deidentify medical data
            medical_data = state.get("medical_output", {})
            deidentified_medical = self.data_processor.deidentify_medical_data(medical_data, state["patient_data"])
            state["deidentified_medical"] = deidentified_medical
            
            # Keep pharmacy data raw - no deidentification needed
            pharmacy_data = state.get("pharmacy_output", {})
            state["raw_pharmacy"] = pharmacy_data
            
            state["step_status"]["deidentify_data"] = "completed"
            
            # Log comprehensive deidentification stats
            mcid_stats = deidentified_mcid.get("deidentification_stats", {})
            medical_stats = deidentified_medical.get("deidentification_stats", {})
            
            total_fields = (
                deidentified_mcid.get('total_fields_processed', 0) +
                deidentified_medical.get('total_fields_processed', 0)
            )
            
            logger.info("✅ Successfully completed comprehensive nested JSON deidentification (MCID + Medical)")
            logger.info(f"📊 MCID fields processed: {deidentified_mcid.get('total_fields_processed', 0)}")
            logger.info(f"📊 Medical fields processed: {deidentified_medical.get('total_fields_processed', 0)}")
            logger.info(f"💊 Pharmacy data kept raw (no deidentification needed)")
            logger.info(f"📊 Total deidentified fields: {total_fields}")
            
        except Exception as e:
            error_msg = f"Error in comprehensive deidentification: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["deidentify_data"] = "error"
            logger.error(error_msg)
        
        return state
    
    def extract_medical_pharmacy_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 3: Extract fields with LLM-powered meanings and dates"""
        logger.info("🔍 Enhanced Node 3: Starting LLM-enhanced medical and pharmacy data extraction with dates...")
        state["current_step"] = "extract_medical_pharmacy_data"
        state["step_status"]["extract_medical_pharmacy_data"] = "running"
        
        try:
            # Use LLM-enhanced data processor for medical extraction with meanings and dates
            medical_extraction = self.data_processor.extract_medical_fields_with_llm_meanings(
                state.get("deidentified_medical", {}), 
                self.api_integrator.call_llm
            )
            state["medical_extraction"] = medical_extraction
            
            medical_records = len(medical_extraction.get('hlth_srvc_records', []))
            dates_extracted = medical_extraction.get('extraction_summary', {}).get('dates_extracted', 0)
            logger.info(f"📋 LLM-enhanced medical extraction: {medical_records} records, {dates_extracted} dates, LLM meanings added")
            
            # Use LLM-enhanced data processor for pharmacy extraction with raw pharmacy data
            pharmacy_extraction = self.data_processor.extract_pharmacy_fields_with_llm_meanings_from_raw(
                state.get("raw_pharmacy", {}), 
                self.api_integrator.call_llm
            )
            state["pharmacy_extraction"] = pharmacy_extraction
            
            pharmacy_records = len(pharmacy_extraction.get('ndc_records', []))
            rx_dates_extracted = pharmacy_extraction.get('extraction_summary', {}).get('dates_extracted', 0)
            logger.info(f"💊 LLM-enhanced pharmacy extraction (from raw data): {pharmacy_records} records, {rx_dates_extracted} dates, LLM meanings added")
            
            state["step_status"]["extract_medical_pharmacy_data"] = "completed"
            logger.info("✅ Successfully completed LLM-enhanced medical and pharmacy data extraction with dates")
            
        except Exception as e:
            error_msg = f"Error in LLM-enhanced medical/pharmacy extraction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_medical_pharmacy_data"] = "error"
            logger.error(error_msg)
        
        return state
    
    def extract_entities(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 4: Extract comprehensive health entities"""
        logger.info("🎯 Enhanced Node 4: Starting comprehensive health entity extraction...")
        state["current_step"] = "extract_entities"
        state["step_status"]["extract_entities"] = "running"
        
        try:
            raw_pharmacy_data = state.get("raw_pharmacy", {})
            pharmacy_extraction = state.get("pharmacy_extraction", {})
            medical_extraction = state.get("medical_extraction", {})
            
            # Use enhanced data processor for comprehensive entity extraction with raw pharmacy data
            entities = self.data_processor.extract_health_entities_enhanced(
                raw_pharmacy_data, pharmacy_extraction, medical_extraction
            )
            state["entity_extraction"] = entities
            
            state["step_status"]["extract_entities"] = "completed"
            
            # Log extraction results
            conditions_count = len(entities.get("medical_conditions", []))
            medications_count = len(entities.get("medications_identified", []))
            logger.info(f"✅ Successfully extracted comprehensive health entities: {conditions_count} conditions, {medications_count} medications")
            
        except Exception as e:
            error_msg = f"Error in comprehensive entity extraction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_entities"] = "error"
            logger.error(error_msg)
        
        return state
    
    def analyze_trajectory(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 5: Analyze health trajectory with comprehensive data"""
        logger.info("📈 Enhanced Node 5: Starting comprehensive health trajectory analysis...")
        state["current_step"] = "analyze_trajectory"
        state["step_status"]["analyze_trajectory"] = "running"
        
        try:
            deidentified_medical = state.get("deidentified_medical", {})
            deidentified_pharmacy = state.get("deidentified_pharmacy", {})
            medical_extraction = state.get("medical_extraction", {})
            pharmacy_extraction = state.get("pharmacy_extraction", {})
            entities = state.get("entity_extraction", {})
            
            trajectory_prompt = self._create_comprehensive_trajectory_prompt(
                deidentified_medical, deidentified_pharmacy, 
                medical_extraction, pharmacy_extraction, entities
            )
            
            logger.info("🤖 Calling Snowflake Cortex for comprehensive health trajectory analysis...")
            
            # Use API integrator for LLM call
            response = self.api_integrator.call_llm(trajectory_prompt)
            
            if response.startswith("Error"):
                state["errors"].append(f"Comprehensive trajectory analysis failed: {response}")
                state["step_status"]["analyze_trajectory"] = "error"
            else:
                state["health_trajectory"] = response
                state["step_status"]["analyze_trajectory"] = "completed"
                logger.info("✅ Successfully completed comprehensive health trajectory analysis")
            
        except Exception as e:
            error_msg = f"Error in comprehensive trajectory analysis: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["analyze_trajectory"] = "error"
            logger.error(error_msg)
        
        return state
    
    def generate_summary(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 6: Generate comprehensive final health summary"""
        logger.info("📋 Enhanced Node 6: Generating comprehensive final health summary...")
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
            # Extract features using enhanced feature extraction
            features = self._extract_enhanced_heart_attack_features(state)
            state["heart_attack_features"] = features
            
            if not features or "error" in features:
                state["errors"].append("Failed to extract enhanced features for heart attack prediction")
                state["step_status"]["predict_heart_attack"] = "error"
                return state
            
            # Prepare feature vector for enhanced FastAPI call
            fastapi_features = self._prepare_enhanced_fastapi_features(features)
            
            if fastapi_features is None:
                state["errors"].append("Failed to prepare enhanced feature vector for prediction")
                state["step_status"]["predict_heart_attack"] = "error"
                return state
            
            # Make async prediction using enhanced API integrator
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                prediction_result = loop.run_until_complete(
                    self.api_integrator.call_fastapi_heart_attack_prediction(fastapi_features)
                )
                loop.close()
            except Exception as async_error:
                logger.error(f"Enhanced async prediction call failed: {async_error}")
                state["errors"].append(f"Enhanced FastAPI prediction call failed: {str(async_error)}")
                state["step_status"]["predict_heart_attack"] = "error"
                return state
            
            if prediction_result.get("success", False):
                # Process successful enhanced FastAPI prediction
                prediction_data = prediction_result.get("prediction_data", {})
                
                # Extract key values from enhanced FastAPI response
                risk_probability = prediction_data.get("probability", 0.0)
                binary_prediction = prediction_data.get("prediction", 0)
                
                # Convert to percentage
                risk_percentage = risk_probability * 100
                confidence_percentage = (1 - risk_probability) * 100 if binary_prediction == 0 else risk_probability * 100
                
                # Determine enhanced risk level
                if risk_percentage >= 70:
                    risk_category = "High Risk"
                elif risk_percentage >= 50:
                    risk_category = "Medium Risk"
                else:
                    risk_category = "Low Risk"
                
                # Create enhanced prediction result
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
                # Handle enhanced FastAPI prediction failure
                error_msg = prediction_result.get("error", "Unknown enhanced FastAPI error")
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
                logger.warning(f"⚠️ Enhanced FastAPI heart attack prediction failed: {error_msg}")
            
            state["step_status"]["predict_heart_attack"] = "completed"
            
        except Exception as e:
            error_msg = f"Error in enhanced FastAPI heart attack prediction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["predict_heart_attack"] = "error"
            logger.error(error_msg)
        
        return state
    
    def initialize_chatbot(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 8: Initialize comprehensive chatbot with complete context including MCID and raw pharmacy"""
        logger.info("💬 Enhanced Node 8: Initializing comprehensive chatbot with complete context including MCID and raw pharmacy...")
        state["current_step"] = "initialize_chatbot"
        state["step_status"]["initialize_chatbot"] = "running"
        
        try:
            # Prepare comprehensive chatbot context with deidentified medical/MCID and raw pharmacy data
            comprehensive_chatbot_context = {
                "deidentified_mcid": state.get("deidentified_mcid", {}),
                "deidentified_medical": state.get("deidentified_medical", {}),
                "raw_pharmacy": state.get("raw_pharmacy", {}),  # Pharmacy data kept raw
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
                    "model_type": "enhanced_fastapi_mcp_compatible_with_mcid_raw_pharmacy",
                    "deidentification_level": "comprehensive_nested_with_mcid_raw_pharmacy",
                    "total_mcid_fields": state.get("deidentified_mcid", {}).get("total_fields_processed", 0),
                    "total_medical_fields": state.get("deidentified_medical", {}).get("total_fields_processed", 0),
                    "pharmacy_data_type": "raw",
                    "llm_enhanced_extractions": True
                }
            }
            
            state["chat_history"] = []
            state["chatbot_context"] = comprehensive_chatbot_context
            state["chatbot_ready"] = True
            state["processing_complete"] = True
            state["step_status"]["initialize_chatbot"] = "completed"
            
            # Log comprehensive chatbot initialization including MCID and raw pharmacy
            medical_records = len(state.get("medical_extraction", {}).get("hlth_srvc_records", []))
            pharmacy_records = len(state.get("pharmacy_extraction", {}).get("ndc_records", []))
            mcid_fields = state.get("deidentified_mcid", {}).get("total_fields_processed", 0)
            
            logger.info("✅ Successfully initialized comprehensive chatbot with complete context including MCID and raw pharmacy")
            logger.info(f"📊 Chatbot context includes: {mcid_fields} MCID fields, {medical_records} medical records, {pharmacy_records} pharmacy records")
            logger.info(f"🔒 Deidentification: MCID + Medical deidentified, Pharmacy kept raw")
            logger.info(f"🤖 LLM-enhanced extractions: code meanings and dates included")
            
        except Exception as e:
            error_msg = f"Error initializing comprehensive chatbot with MCID and raw pharmacy: {str(e)}"
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
    
    # ===== ENHANCED CHATBOT FUNCTIONALITY =====
    
    def chat_with_data(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Enhanced chatbot with COMPLETE deidentified data access and special heart attack handling"""
        try:
            # Check if this is a heart attack prediction question
            heart_attack_keywords = [
                'heart attack', 'cardiac risk', 'heart disease', 'cardiovascular risk',
                'heart health', 'cardiac health', 'heart attack risk', 'heart attack prediction'
            ]
            
            is_heart_attack_question = any(keyword in user_query.lower() for keyword in heart_attack_keywords)
            
            if is_heart_attack_question:
                return self._handle_heart_attack_prediction_question(user_query, chat_context, chat_history)
            
            # Use enhanced data processor to prepare COMPLETE context - ENTIRE JSON data including MCID
            complete_context = self.data_processor.prepare_complete_deidentified_context(chat_context)
            
            # Build conversation history for continuity
            history_text = ""
            if chat_history:
                recent_history = chat_history[-10:]
                history_text = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in recent_history
                ])
            
            # Create COMPLETE prompt with ENTIRE deidentified data including MCID
            complete_prompt = f"""You are an expert medical data assistant with access to the COMPLETE, ENTIRE deidentified patient health records including MCID, medical, and pharmacy data. You have the FULL JSON data structures available. Answer the user's question with specific, detailed information from ANY part of the complete medical data provided.

COMPLETE DEIDENTIFIED PATIENT DATA (ENTIRE JSON STRUCTURES INCLUDING MCID - NO TRUNCATION):
{complete_context}

RECENT CONVERSATION HISTORY:
{history_text}

USER QUESTION: {user_query}

CRITICAL INSTRUCTIONS:
- You have access to the COMPLETE deidentified MCID, medical, and pharmacy JSON data with LLM-enhanced meanings
- Search through the ENTIRE JSON structure including MCID data to find relevant information
- Include specific dates (CLM_RCVD_DT, RX_FILLED_DT), codes, medications, diagnoses, and values from ANY part of the JSON
- Reference exact field names, values, nested structures, and LLM-generated meanings from the data
- If user asks about ANY specific field, code, medication, or data point, find it in the complete JSON
- Include ICD-10 codes with LLM meanings, NDC codes with descriptions, service codes with explanations
- Access all nested levels of the JSON structure including MCID member information
- Use LLM-enhanced code meanings to provide better explanations
- Include dates from CLM_RCVD_DT (medical) and RX_FILLED_DT (pharmacy) when relevant
- Be thorough and cite specific data points from the complete deidentified records
- Use the conversation history to understand follow-up questions and context

DETAILED ANSWER USING COMPLETE DEIDENTIFIED DATA INCLUDING MCID:"""

            logger.info(f"💬 Processing query with COMPLETE deidentified data access including MCID: {user_query[:50]}...")
            logger.info(f"📊 Complete context length: {len(complete_context)} characters")
            
            # Use enhanced API integrator for LLM call
            enhanced_system_msg = """You are a powerful healthcare AI assistant with access to COMPLETE deidentified medical records including MCID, medical, and pharmacy data with LLM-enhanced code meanings. You can search through and reference ANY part of the provided JSON data structures. Provide accurate, detailed analysis based on the ENTIRE medical and pharmacy data provided."""
            
            response = self.api_integrator.call_llm(complete_prompt, enhanced_system_msg)
            
            if response.startswith("Error"):
                return "I encountered an error processing your question. Please try rephrasing your question. I have access to the complete deidentified MCID, medical, and pharmacy data with enhanced code meanings and can answer questions about any specific codes, medications, dates, or other data points."
            
            return response
            
        except Exception as e:
            logger.error(f"Error in complete deidentified data chatbot with MCID: {str(e)}")
            return "I encountered an error processing your question. Please try again. I have access to the complete deidentified MCID, medical, and pharmacy JSON data with enhanced meanings and can answer detailed questions about any aspect of the patient's records."
    
    def _handle_heart_attack_prediction_question(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Special handler for heart attack prediction questions - shows both LLM analysis and ML model results"""
        try:
            logger.info("❤️ Handling heart attack prediction question with both LLM and ML model analysis...")
            
            # Get ML model prediction results
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            heart_attack_features = chat_context.get("heart_attack_features", {})
            
            # Get complete context for LLM analysis
            complete_context = self.data_processor.prepare_complete_deidentified_context(chat_context)
            
            # Build conversation history
            history_text = ""
            if chat_history:
                recent_history = chat_history[-8:]
                history_text = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in recent_history
                ])
            
            # Create special heart attack analysis prompt
            heart_attack_prompt = f"""You are an expert cardiovascular risk assessment specialist with access to complete patient data AND machine learning model results. Provide a comprehensive heart attack risk analysis combining both clinical data review and ML model insights.

COMPLETE PATIENT DATA INCLUDING MCID, MEDICAL, AND PHARMACY RECORDS:
{complete_context}

MACHINE LEARNING MODEL PREDICTION RESULTS:
{json.dumps(heart_attack_prediction, indent=2)}

FEATURES USED IN ML MODEL:
{json.dumps(heart_attack_features, indent=2)}

CONVERSATION HISTORY:
{history_text}

USER QUESTION: {user_query}

PROVIDE A COMPREHENSIVE HEART ATTACK RISK ASSESSMENT THAT INCLUDES:

1. **CLINICAL DATA ANALYSIS** (Your LLM Analysis):
   - Review the complete medical history, diagnoses, and medications
   - Analyze risk factors from the deidentified data
   - Clinical interpretation of findings
   - Evidence-based risk assessment

2. **MACHINE LEARNING MODEL RESULTS**:
   - ML model prediction: {heart_attack_prediction.get('combined_display', 'Not available')}
   - Model confidence and methodology
   - Features that influenced the prediction

3. **INTEGRATED ASSESSMENT**:
   - How the clinical data supports or contrasts with the ML prediction
   - Comprehensive risk evaluation
   - Recommendations based on both analyses

Format your response clearly with sections for LLM Clinical Analysis and ML Model Results, then provide an integrated conclusion."""

            logger.info("🤖 Calling LLM for comprehensive heart attack risk analysis...")
            
            response = self.api_integrator.call_llm(heart_attack_prompt, self.config.chatbot_sys_msg)
            
            if response.startswith("Error"):
                # Fallback response with just ML model results
                ml_display = heart_attack_prediction.get('combined_display', 'Heart attack prediction not available')
                return f"""**Heart Attack Risk Assessment**

**Machine Learning Model Results:**
{ml_display}

**Error:** Unable to perform detailed clinical analysis at this time. The ML model prediction above is based on the extracted patient features.

Please try rephrasing your question for a more detailed analysis."""
            
            return response
            
        except Exception as e:
            logger.error(f"Error in heart attack prediction analysis: {str(e)}")
            # Fallback with just ML results
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            ml_display = heart_attack_prediction.get('combined_display', 'Heart attack prediction not available')
            
            return f"""**Heart Attack Risk Assessment**

**Machine Learning Model Results:**
{ml_display}

**Note:** Detailed clinical analysis unavailable due to processing error. The ML model prediction above provides the cardiovascular risk assessment based on extracted patient features."""
    
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
            extracted_features = features.get("extracted_features", {})
            
            enhanced_fastapi_features = {
                "age": int(extracted_features.get("Age", 50)),
                "gender": int(extracted_features.get("Gender", 0)),
                "diabetes": int(extracted_features.get("Diabetes", 0)),
                "high_bp": int(extracted_features.get("High_BP", 0)),
                "smoking": int(extracted_features.get("Smoking", 0))
            }
            
            # Enhanced validation with logging
            if enhanced_fastapi_features["age"] < 0 or enhanced_fastapi_features["age"] > 120:
                logger.warning(f"Age {enhanced_fastapi_features['age']} out of range, using default 50")
                enhanced_fastapi_features["age"] = 50
            
            for key in ["gender", "diabetes", "high_bp", "smoking"]:
                if enhanced_fastapi_features[key] not in [0, 1]:
                    logger.warning(f"{key} value {enhanced_fastapi_features[key]} invalid, using 0")
                    enhanced_fastapi_features[key] = 0
            
            logger.info(f"✅ Enhanced FastAPI features prepared: {enhanced_fastapi_features}")
            return enhanced_fastapi_features
            
        except Exception as e:
            logger.error(f"Error preparing enhanced FastAPI features: {e}")
            return None
    
    def _create_comprehensive_trajectory_prompt(self, medical_data: Dict, pharmacy_data: Dict, 
                                              medical_extraction: Dict, pharmacy_extraction: Dict, 
                                              entities: Dict) -> str:
        """Create comprehensive prompt for health trajectory analysis"""
        return f"""
You are a healthcare AI assistant analyzing a patient's comprehensive health trajectory. Based on the following complete deidentified data with comprehensive nested JSON processing, provide a detailed health trajectory analysis.

COMPLETE DEIDENTIFIED MEDICAL DATA (Comprehensive Processing):
{json.dumps(medical_data, indent=2, default=str)}

COMPLETE DEIDENTIFIED PHARMACY DATA (Comprehensive Processing):
{json.dumps(pharmacy_data, indent=2, default=str)}

ENHANCED MEDICAL DATA EXTRACTIONS:
{json.dumps(medical_extraction, indent=2)}

ENHANCED PHARMACY DATA EXTRACTIONS:
{json.dumps(pharmacy_extraction, indent=2)}

COMPREHENSIVE HEALTH ENTITIES:
{json.dumps(entities, indent=2)}

Please analyze this patient's health trajectory focusing on:

1. **Current Health Status**: Comprehensive assessment based on medical codes, pharmacy data, and extracted entities
2. **Risk Factors**: Identified health risks from ICD-10 codes and medication patterns  
3. **Medication Analysis**: Complete NDC codes, drug names, and therapeutic areas identified
4. **Chronic Conditions**: Long-term health management needs from medical service codes
5. **Health Trends**: Trajectory of health over time based on comprehensive service utilization
6. **Care Recommendations**: Suggested areas for medical attention based on complete data analysis
7. **Deidentification Impact**: How comprehensive deidentification preserves clinical utility

Provide a detailed analysis (500-600 words) that synthesizes all the available structured and unstructured information into a coherent health trajectory assessment using the comprehensive nested JSON processing results.
"""
    
    def _create_comprehensive_summary_prompt(self, trajectory_analysis: str, entities: Dict, 
                                           medical_extraction: Dict, pharmacy_extraction: Dict) -> str:
        """Create comprehensive prompt for final health summary"""
        return f"""
Based on the detailed health trajectory analysis below and the comprehensive data extractions with enhanced processing, create a concise executive summary of this patient's health status.

COMPREHENSIVE HEALTH TRAJECTORY ANALYSIS:
{trajectory_analysis}

ENHANCED HEALTH ENTITIES:
- Diabetes: {entities.get('diabetics', 'unknown')}
- Age Group: {entities.get('age_group', 'unknown')}
- Smoking Status: {entities.get('smoking', 'unknown')}
- Alcohol Status: {entities.get('alcohol', 'unknown')}
- Blood Pressure: {entities.get('blood_pressure', 'unknown')}
- Medical Conditions Identified: {len(entities.get('medical_conditions', []))}
- Medications Identified: {len(entities.get('medications_identified', []))}
- Risk Factors: {len(entities.get('risk_factors', []))}
- Chronic Conditions: {len(entities.get('chronic_conditions', []))}

ENHANCED MEDICAL DATA SUMMARY:
- Health Service Records: {len(medical_extraction.get('hlth_srvc_records', []))}
- Total Diagnosis Codes: {medical_extraction.get('extraction_summary', {}).get('total_diagnosis_codes', 0)}
- Unique Service Codes: {len(medical_extraction.get('extraction_summary', {}).get('unique_service_codes', []))}

ENHANCED PHARMACY DATA SUMMARY:
- NDC Records: {len(pharmacy_extraction.get('ndc_records', []))}
- Unique NDC Codes: {len(pharmacy_extraction.get('extraction_summary', {}).get('unique_ndc_codes', []))}
- Unique Medications: {len(pharmacy_extraction.get('extraction_summary', {}).get('unique_label_names', []))}

Create a comprehensive final summary that includes:

1. **Health Status Overview** (2-3 sentences based on comprehensive data)
2. **Key Risk Factors** (bullet points based on ICD-10 codes and enhanced medication analysis)
3. **Priority Recommendations** (3-4 actionable items based on comprehensive analysis)
4. **Follow-up Needs** (timing and type of care based on service codes and medication patterns)
5. **Data Processing Notes** (brief note on comprehensive deidentification and extraction)

Keep the summary under 300 words and focus on actionable insights for healthcare providers based on the comprehensive data analysis with enhanced nested JSON processing.
"""
    
    def test_llm_connection(self) -> Dict[str, Any]:
        """Test enhanced Snowflake Cortex API connection"""
        return self.api_integrator.test_llm_connection()
    
    async def test_fastapi_connection(self) -> Dict[str, Any]:
        """Test enhanced FastAPI server connection"""
        return await self.api_integrator.test_fastapi_connection()
    
    def test_backend_connection(self) -> Dict[str, Any]:
        """Test MCP backend server connection"""
        return self.api_integrator.test_backend_connection()
    
    def run_analysis(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the enhanced health analysis workflow using LangGraph with MCP compatibility"""
        
        # Initialize enhanced state for LangGraph including MCID and raw pharmacy
        initial_state = HealthAnalysisState(
            patient_data=patient_data,
            mcid_output={},
            medical_output={},
            pharmacy_output={},
            token_output={},
            deidentified_mcid={},
            deidentified_medical={},
            raw_pharmacy={},
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
            
            logger.info("🚀 Starting Enhanced MCP-Compatible LangGraph health analysis workflow...")
            
            final_state = self.graph.invoke(initial_state, config=config_dict)
            
            # Prepare enhanced results with comprehensive information including MCID and raw pharmacy
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
                    "mcid": final_state["deidentified_mcid"],
                    "medical": final_state["deidentified_medical"],
                    "pharmacy": final_state["raw_pharmacy"]  # Pharmacy kept raw
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
                "mcid_included": True,
                "comprehensive_deidentification": True,
                "pharmacy_data_type": "raw",
                "llm_enhanced_extractions": True,
                "enhanced_chatbot": True,
                "heart_attack_special_handling": True,
                "enhancement_version": "v6.1_mcid_raw_pharmacy_llm_enhanced"
            }
            
            if results["success"]:
                logger.info("✅ Enhanced MCP-Compatible LangGraph health analysis completed successfully!")
                logger.info(f"🔒 Comprehensive deidentification: {results['comprehensive_deidentification']}")
                logger.info(f"💬 Enhanced chatbot ready: {results['chatbot_ready']}")
            else:
                logger.error(f"❌ Enhanced LangGraph health analysis failed with errors: {final_state['errors']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Fatal error in Enhanced MCP-Compatible LangGraph workflow: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "patient_data": patient_data,
                "errors": [str(e)],
                "processing_steps_completed": 0,
                "langgraph_used": True,
                "mcp_compatible": True,
                "mcid_included": False,
                "comprehensive_deidentification": False,
                "pharmacy_data_type": "raw",
                "llm_enhanced_extractions": False,
                "enhanced_chatbot": False,
                "heart_attack_special_handling": False,
                "enhancement_version": "v6.1_mcid_raw_pharmacy_llm_enhanced"
            }
    
    def _count_completed_steps(self, state: HealthAnalysisState) -> int:
        """Count enhanced processing steps completed including MCID and raw pharmacy"""
        steps = 0
        if state.get("mcid_output"): steps += 1
        if state.get("deidentified_mcid") and not state.get("deidentified_mcid", {}).get("error"): steps += 1
        if state.get("deidentified_medical") and not state.get("deidentified_medical", {}).get("error"): steps += 1
        if state.get("raw_pharmacy"): steps += 1  # Pharmacy kept raw
        if state.get("medical_extraction") or state.get("pharmacy_extraction"): steps += 1
        if state.get("entity_extraction"): steps += 1
        if state.get("health_trajectory"): steps += 1
        if state.get("final_summary"): steps += 1
        if state.get("heart_attack_prediction"): steps += 1
        if state.get("chatbot_ready"): steps += 1
        return steps

def main():
    """Example usage of the Enhanced MCP-Compatible Health Analysis Agent with MCID and LLM enhancements"""
    
    print("🏥 Enhanced MCP-Compatible Health Analysis Agent v6.0")
    print("✅ Comprehensive modular architecture with advanced features:")
    print("   📡 HealthAPIIntegrator - MCP server compatible API calls")
    print("   🔧 HealthDataProcessor - Comprehensive nested JSON deidentification with MCID")
    print("   🏗️ HealthAnalysisAgent - Enhanced workflow orchestration with LLM integration")
    print("   💬 Enhanced chatbot - Complete deidentified data access with heart attack special handling")
    print("   🆔 MCID Integration - Member Consumer ID data included in analysis")
    print("   🤖 LLM-Enhanced Extractions - Code meanings and descriptions powered by AI")
    print("   📅 Date Extraction - CLM_RCVD_DT and RX_FILLED_DT processing")
    print("   ❤️ Heart Attack Prediction - Dual LLM + ML model analysis")
    print()
    
    config = Config()
    print("📋 Enhanced Configuration:")
    print(f"   🌐 Snowflake API: {config.api_url}")
    print(f"   🤖 Model: {config.model}")
    print(f"   📡 MCP Server: {config.fastapi_url}")
    print(f"   ❤️ Heart Attack API: {config.heart_attack_api_url}")
    print()
    print("✅ Enhanced MCP-Compatible Health Agent ready with all advanced features!")
    print("🚀 Features: MCID + Medical + Pharmacy deidentification, LLM code meanings, date extraction, heart attack dual analysis")
    print("🚀 Run: from health_agent_core import HealthAnalysisAgent, Config")
    
    return "Enhanced MCP-Compatible Health Agent v6.0 ready for integration"

if __name__ == "__main__":
    main()
