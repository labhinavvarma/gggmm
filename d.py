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
    chatbot_sys_msg: str = "You are a powerful healthcare AI assistant with access to comprehensive deidentified claims data. Provide accurate, detailed analysis based on the complete medical, pharmacy, and MCID claims data provided. Always maintain patient privacy and provide professional medical insights."
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
    """Enhanced Health Analysis Agent with Claims Data Processing"""
    
    def __init__(self, custom_config: Optional[Config] = None):
        # Use provided config or create default
        self.config = custom_config or Config()
        
        # Initialize enhanced components
        self.api_integrator = HealthAPIIntegrator(self.config)
        self.data_processor = HealthDataProcessor()
        
        logger.info("🔧 Enhanced HealthAnalysisAgent initialized with Claims Data Processing")
        logger.info(f"🌐 Snowflake API URL: {self.config.api_url}")
        logger.info(f"🤖 Model: {self.config.model}")
        logger.info(f"📡 MCP Server URL: {self.config.fastapi_url}")
        logger.info(f"❤️ Heart Attack API: {self.config.heart_attack_api_url}")
        
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
        """Enhanced LangGraph Node 4: Extract comprehensive health entities (NO LLM)"""
        logger.info("🎯 Enhanced Node 4: Starting direct health entity extraction (NO LLM)...")
        state["current_step"] = "extract_entities"
        state["step_status"]["extract_entities"] = "running"
        
        try:
            pharmacy_data = state.get("pharmacy_output", {})
            pharmacy_extraction = state.get("pharmacy_extraction", {})
            medical_extraction = state.get("medical_extraction", {})
            
            # Direct entity extraction without LLM
            entities = self.data_processor.extract_health_entities_enhanced(
                pharmacy_data, pharmacy_extraction, medical_extraction
            )
            state["entity_extraction"] = entities
            
            state["step_status"]["extract_entities"] = "completed"
            
            # Log extraction results
            conditions_count = len(entities.get("medical_conditions", []))
            medications_count = len(entities.get("medications_identified", []))
            logger.info(f"✅ Successfully extracted health entities (NO LLM): {conditions_count} conditions, {medications_count} medications")
            
        except Exception as e:
            error_msg = f"Error in direct entity extraction: {str(e)}"
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
                    "model_type": "enhanced_fastapi_mcp_compatible",
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
    
    # ===== ENHANCED CHATBOT FUNCTIONALITY =====
    
    def chat_with_data(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Enhanced chatbot with COMPLETE deidentified claims data access and heart attack analysis"""
        try:
            # Check if this is a heart attack related question
            heart_attack_keywords = ['heart attack', 'heart disease', 'cardiac', 'cardiovascular', 'heart risk', 'coronary', 'myocardial', 'cardiac risk']
            is_heart_attack_question = any(keyword in user_query.lower() for keyword in heart_attack_keywords)
            
            if is_heart_attack_question:
                return self._handle_heart_attack_question(user_query, chat_context, chat_history)
            else:
                return self._handle_general_question(user_query, chat_context, chat_history)
            
        except Exception as e:
            logger.error(f"Error in complete deidentified claims data chatbot: {str(e)}")
            return "I encountered an error processing your question. Please try again. I have access to the complete deidentified claims JSON data and can answer detailed questions about any aspect of the patient's records."
    
    def _handle_heart_attack_question(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Handle heart attack related questions with both FastAPI and LLM analysis"""
        try:
            # Get FastAPI prediction from context
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            entity_extraction = chat_context.get("entity_extraction", {})
            
            # Prepare comprehensive context for LLM analysis
            complete_context = self._prepare_heart_attack_context(chat_context)
            
            # Build conversation history
            history_text = ""
            if chat_history:
                recent_history = chat_history[-5:]
                history_text = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in recent_history
                ])
            
            # Create enhanced prompt for heart attack analysis
            heart_attack_prompt = f"""You are an expert cardiologist and data analyst with access to COMPLETE deidentified patient claims data. Analyze the heart attack/cardiovascular risk based on the comprehensive medical and pharmacy claims data provided.

COMPLETE DEIDENTIFIED CLAIMS DATA FOR HEART ATTACK ANALYSIS:
{complete_context}

CURRENT FASTAPI ML MODEL PREDICTION:
{json.dumps(heart_attack_prediction, indent=2)}

EXTRACTED HEALTH ENTITIES:
{json.dumps(entity_extraction, indent=2)}

RECENT CONVERSATION HISTORY:
{history_text}

USER QUESTION: {user_query}

CRITICAL ANALYSIS INSTRUCTIONS:
- Provide a comprehensive heart attack/cardiovascular risk assessment in PERCENTAGE format
- Analyze ALL available medical codes (ICD-10), medications (NDC), and claims data
- Consider age, gender, diabetes status, blood pressure, smoking, and medication patterns
- Compare your analysis with the FastAPI ML model prediction provided above
- Provide specific percentages for cardiovascular risk based on the complete claims data
- Reference specific medical codes, medications, dates, and clinical indicators from the JSON data
- Explain discrepancies between your analysis and the FastAPI model if any
- Include risk factors found in the claims data that support your percentage assessment

PROVIDE YOUR RESPONSE IN THIS FORMAT:

**🤖 LLM CARDIOVASCULAR RISK ANALYSIS:**
- **Risk Percentage:** [Your calculated percentage]% 
- **Risk Category:** [Low/Medium/High Risk]
- **Key Risk Factors:** [List specific factors from claims data]
- **Supporting Evidence:** [Specific codes, medications, dates from JSON]

**⚖️ COMPARISON WITH FASTAPI MODEL:**
- **FastAPI Prediction:** [FastAPI percentage and category]
- **LLM Analysis:** [Your percentage and category]  
- **Agreement/Discrepancy:** [Comparison and explanation]
- **Confidence:** [Your confidence in the assessment]

**📊 DETAILED CARDIOVASCULAR ASSESSMENT:**
[Provide detailed analysis based on complete claims data]

Use the complete deidentified claims data to provide the most accurate cardiovascular risk assessment possible."""

            logger.info(f"💬 Processing heart attack question with comprehensive analysis: {user_query[:50]}...")
            
            # Use enhanced API integrator for LLM call
            enhanced_system_msg = """You are an expert cardiologist with access to COMPLETE deidentified claims data. Provide detailed cardiovascular risk analysis with specific percentages based on comprehensive medical and pharmacy claims data. Compare your analysis with ML model predictions and explain your reasoning using specific data from the claims records."""
            
            response = self.api_integrator.call_llm(heart_attack_prompt, enhanced_system_msg)
            
            if response.startswith("Error"):
                return "I encountered an error analyzing cardiovascular risk. Please try rephrasing your question. I can provide detailed heart attack risk analysis using both ML predictions and comprehensive claims data analysis."
            
            return response
            
        except Exception as e:
            logger.error(f"Error in heart attack question handling: {str(e)}")
            return "I encountered an error analyzing cardiovascular risk. Please try again. I can compare ML model predictions with comprehensive claims data analysis for heart attack risk assessment."
    
    def _handle_general_question(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Handle general questions with complete claims data access"""
        try:
            # Use enhanced data processor to prepare COMPLETE context with both medical and pharmacy data
            complete_context = self.data_processor.prepare_chunked_context(chat_context)
            
            # Build conversation history for continuity
            history_text = ""
            if chat_history:
                recent_history = chat_history[-10:]
                history_text = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in recent_history
                ])
            
            # Create COMPLETE prompt with ENTIRE deidentified claims data
            complete_prompt = f"""You are an expert medical claims data assistant with access to the COMPLETE, ENTIRE deidentified patient claims records (medical, pharmacy, and MCID). You have the FULL JSON data structures available. Answer the user's question with specific, detailed information from ANY part of the complete claims data provided.

COMPLETE DEIDENTIFIED CLAIMS DATA (ENTIRE JSON STRUCTURES - MEDICAL & PHARMACY):
{complete_context}

RECENT CONVERSATION HISTORY:
{history_text}

USER QUESTION: {user_query}

CRITICAL INSTRUCTIONS:
- You have access to the COMPLETE deidentified medical, pharmacy, and MCID claims JSON data
- Search through the ENTIRE JSON structure to find relevant information
- Include specific dates (clm_rcvd_dt, rx_filled_dt), codes, medications, diagnoses, and values from ANY part of the JSON
- Reference exact field names, values, and nested structures from the data
- If user asks about ANY specific field, code, medication, or data point, find it in the complete JSON
- Include ICD-10 codes, NDC codes, service codes, dates, quantities, and any other specific data
- Access all nested levels of the JSON structure to answer questions
- Be thorough and cite specific data points from the complete deidentified records
- If data exists in the JSON, you can find and reference it
- Use the conversation history to understand follow-up questions and context
- Explain medical codes and terminology when relevant
- For numerical values, dates, codes - provide exact values from the JSON data
- Include both medical claims data AND pharmacy claims data in your analysis

DETAILED ANSWER USING COMPLETE DEIDENTIFIED CLAIMS DATA:"""

            logger.info(f"💬 Processing general query with COMPLETE deidentified claims data access: {user_query[:50]}...")
            logger.info(f"📊 Complete context length: {len(complete_context)} characters")
            
            # Use enhanced API integrator for LLM call with extended system message
            enhanced_system_msg = """You are a powerful healthcare AI assistant with access to COMPLETE deidentified claims records including both medical and pharmacy data. You can search through and reference ANY part of the provided JSON data structures. Provide accurate, detailed analysis based on the ENTIRE medical, pharmacy, and MCID claims data provided. Always maintain patient privacy and provide professional medical insights using the complete available data."""
            
            response = self.api_integrator.call_llm(complete_prompt, enhanced_system_msg)
            
            if response.startswith("Error"):
                return "I encountered an error processing your question. Please try rephrasing your question. I have access to the complete deidentified claims data including both medical and pharmacy records and can answer questions about any specific codes, medications, dates, or other data points."
            
            return response
            
        except Exception as e:
            logger.error(f"Error in general question handling: {str(e)}")
            return "I encountered an error processing your question. Please try again. I have access to the complete deidentified claims JSON data and can answer detailed questions about any aspect of the patient's records."
    
    def _prepare_heart_attack_context(self, chat_context: Dict[str, Any]) -> str:
        """Prepare comprehensive context specifically for heart attack analysis"""
        try:
            context_sections = []
            
            # 1. Patient Overview
            patient_overview = chat_context.get("patient_overview", {})
            if patient_overview:
                context_sections.append(f"PATIENT OVERVIEW:\n{json.dumps(patient_overview, indent=2)}")
            
            # 2. Complete Medical Claims Data
            deidentified_medical = chat_context.get("deidentified_medical", {})
            if deidentified_medical:
                medical_claims_data = deidentified_medical.get('medical_claims_data', {})
                if medical_claims_data:
                    context_sections.append(f"COMPLETE MEDICAL CLAIMS DATA:\n{json.dumps(medical_claims_data, indent=2)}")
            
            # 3. Complete Pharmacy Claims Data  
            deidentified_pharmacy = chat_context.get("deidentified_pharmacy", {})
            if deidentified_pharmacy:
                pharmacy_claims_data = deidentified_pharmacy.get('pharmacy_claims_data', {})
                if pharmacy_claims_data:
                    context_sections.append(f"COMPLETE PHARMACY CLAIMS DATA:\n{json.dumps(pharmacy_claims_data, indent=2)}")
            
            # 4. Medical Extractions with dates
            medical_extraction = chat_context.get("medical_extraction", {})
            if medical_extraction and not medical_extraction.get('error'):
                context_sections.append(f"MEDICAL EXTRACTIONS (including clm_rcvd_dt):\n{json.dumps(medical_extraction, indent=2)}")
            
            # 5. Pharmacy Extractions with dates
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            if pharmacy_extraction and not pharmacy_extraction.get('error'):
                context_sections.append(f"PHARMACY EXTRACTIONS (including rx_filled_dt):\n{json.dumps(pharmacy_extraction, indent=2)}")
            
            # 6. Entity Extraction
            entity_extraction = chat_context.get("entity_extraction", {})
            if entity_extraction:
                context_sections.append(f"HEALTH ENTITIES:\n{json.dumps(entity_extraction, indent=2)}")
            
            # 7. Heart Attack Features
            heart_attack_features = chat_context.get("heart_attack_features", {})
            if heart_attack_features:
                context_sections.append(f"HEART ATTACK FEATURES:\n{json.dumps(heart_attack_features, indent=2)}")
            
            return "\n\n".join(context_sections)
            
        except Exception as e:
            logger.error(f"Error preparing heart attack context: {e}")
            return "Patient claims data available for cardiovascular analysis."
    
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
    
    def _create_comprehensive_trajectory_prompt(self, medical_data: Dict, pharmacy_data: Dict, mcid_data: Dict,
                                              medical_extraction: Dict, pharmacy_extraction: Dict, 
                                              entities: Dict) -> str:
        """Create comprehensive prompt for health trajectory analysis"""
        return f"""
You are a healthcare AI assistant analyzing a patient's comprehensive health trajectory. Based on the following complete deidentified claims data with comprehensive nested JSON processing, provide a detailed health trajectory analysis.

COMPLETE DEIDENTIFIED MEDICAL CLAIMS DATA:
{json.dumps(medical_data, indent=2, default=str)}

COMPLETE DEIDENTIFIED PHARMACY CLAIMS DATA:
{json.dumps(pharmacy_data, indent=2, default=str)}

COMPLETE DEIDENTIFIED MCID CLAIMS DATA:
{json.dumps(mcid_data, indent=2, default=str)}

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
7. **Claims Data Integration**: How medical, pharmacy, and MCID data provide comprehensive view

Provide a detailed analysis (500-600 words) that synthesizes all the available structured and unstructured information into a coherent health trajectory assessment using the comprehensive claims data processing results.
"""
    
    def _create_comprehensive_summary_prompt(self, trajectory_analysis: str, entities: Dict, 
                                           medical_extraction: Dict, pharmacy_extraction: Dict) -> str:
        """Create comprehensive prompt for final health summary"""
        return f"""
Based on the detailed health trajectory analysis below and the comprehensive claims data extractions with enhanced processing, create a concise executive summary of this patient's health status.

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

ENHANCED MEDICAL CLAIMS DATA SUMMARY:
- Health Service Records: {len(medical_extraction.get('hlth_srvc_records', []))}
- Total Diagnosis Codes: {medical_extraction.get('extraction_summary', {}).get('total_diagnosis_codes', 0)}
- Unique Service Codes: {len(medical_extraction.get('extraction_summary', {}).get('unique_service_codes', []))}

ENHANCED PHARMACY CLAIMS DATA SUMMARY:
- NDC Records: {len(pharmacy_extraction.get('ndc_records', []))}
- Unique NDC Codes: {len(pharmacy_extraction.get('extraction_summary', {}).get('unique_ndc_codes', []))}
- Unique Medications: {len(pharmacy_extraction.get('extraction_summary', {}).get('unique_label_names', []))}

Create a comprehensive final summary that includes:

1. **Health Status Overview** (2-3 sentences based on comprehensive claims data)
2. **Key Risk Factors** (bullet points based on ICD-10 codes and enhanced medication analysis)
3. **Priority Recommendations** (3-4 actionable items based on comprehensive analysis)
4. **Follow-up Needs** (timing and type of care based on service codes and medication patterns)
5. **Claims Data Processing Notes** (brief note on comprehensive deidentification and extraction)

Keep the summary under 300 words and focus on actionable insights for healthcare providers based on the comprehensive claims data analysis with enhanced nested JSON processing.
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
                "enhancement_version": "v6.0_claims_data_processing"
            }
            
            if results["success"]:
                logger.info("✅ Enhanced Claims Data Processing LangGraph analysis completed successfully!")
                logger.info(f"🔒 Comprehensive claims deidentification: {results['comprehensive_deidentification']}")
                logger.info(f"💬 Enhanced chatbot ready: {results['chatbot_ready']}")
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
                "enhancement_version": "v6.0_claims_data_processing"
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
    """Example usage of the Enhanced Claims Data Processing Health Analysis Agent"""
    
    print("🏥 Enhanced Claims Data Processing Health Analysis Agent v6.0")
    print("✅ Enhanced modular architecture with comprehensive features:")
    print("   📡 HealthAPIIntegrator - MCP server compatible API calls")
    print("   🔧 HealthDataProcessor - Comprehensive claims data deidentification")
    print("   🏗️ HealthAnalysisAgent - Enhanced workflow orchestration")
    print("   💬 Enhanced chatbot - Complete deidentified claims data access")
    print()
    
    config = Config()
    print("📋 Enhanced Configuration:")
    print(f"   🌐 Snowflake API: {config.api_url}")
    print(f"   🤖 Model: {config.model}")
    print(f"   📡 MCP Server: {config.fastapi_url}")
    print(f"   ❤️ Heart Attack API: {config.heart_attack_api_url}")
    print()
    print("✅ Enhanced Claims Data Processing Health Agent ready!")
    print("🚀 Run: from health_agent_core import HealthAnalysisAgent, Config")
    
    return "Enhanced Claims Data Processing Health Agent ready for integration"

if __name__ == "__main__":
    main()
