import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, TypedDict, Literal, Optional
from dataclasses import dataclass, asdict
import logging

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import our modular components
from health_api_integrator import HealthAPIIntegrator
from health_data_processor import HealthDataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    # FastAPI Heart Attack Prediction Configuration
    heart_attack_api_url: str = "http://localhost:8002"
    heart_attack_threshold: float = 0.5
    
    def to_dict(self):
        return asdict(self)

# Enhanced State Definition for LangGraph
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
    
    # Heart Attack Prediction via FastAPI
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
    """Main Health Analysis Agent orchestrating all components"""
    
    def __init__(self, custom_config: Optional[Config] = None):
        # Use provided config or create default
        self.config = custom_config or Config()
        
        # Initialize components
        self.api_integrator = HealthAPIIntegrator(self.config)
        self.data_processor = HealthDataProcessor()
        
        logger.info("🔧 HealthAnalysisAgent initialized")
        logger.info(f"🌐 API URL: {self.config.api_url}")
        logger.info(f"🤖 Model: {self.config.model}")
        logger.info(f"❤️ FastAPI Server: {self.config.heart_attack_api_url}")
        
        self.setup_langgraph()
    
    def setup_langgraph(self):
        """Setup LangGraph workflow - 8 node enhanced workflow"""
        logger.info("🔧 Setting up Enhanced LangGraph workflow with 8 nodes...")
        
        # Create the StateGraph
        workflow = StateGraph(HealthAnalysisState)
        
        # Add all 8 processing nodes
        workflow.add_node("fetch_api_data", self.fetch_api_data)
        workflow.add_node("deidentify_data", self.deidentify_data)
        workflow.add_node("extract_medical_pharmacy_data", self.extract_medical_pharmacy_data)
        workflow.add_node("extract_entities", self.extract_entities)
        workflow.add_node("analyze_trajectory", self.analyze_trajectory)
        workflow.add_node("generate_summary", self.generate_summary)
        workflow.add_node("predict_heart_attack", self.predict_heart_attack)
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
        
        logger.info("✅ Enhanced LangGraph workflow compiled successfully!")
    
    # ===== LANGGRAPH NODES =====
    
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
            
            # Use API integrator to fetch data
            api_result = self.api_integrator.fetch_backend_data(patient_data)
            
            if "error" in api_result:
                state["errors"].append(api_result["error"])
                state["step_status"]["fetch_api_data"] = "error"
            else:
                state["mcid_output"] = api_result.get("mcid_output", {})
                state["medical_output"] = api_result.get("medical_output", {})
                state["pharmacy_output"] = api_result.get("pharmacy_output", {})
                state["token_output"] = api_result.get("token_output", {})
                
                state["step_status"]["fetch_api_data"] = "completed"
                logger.info("✅ Successfully fetched all API data")
                
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
            # Use data processor for deidentification
            medical_data = state.get("medical_output", {})
            deidentified_medical = self.data_processor.deidentify_medical_data(medical_data, state["patient_data"])
            state["deidentified_medical"] = deidentified_medical
            
            pharmacy_data = state.get("pharmacy_output", {})
            deidentified_pharmacy = self.data_processor.deidentify_pharmacy_data(pharmacy_data)
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
            # Use data processor for extraction
            medical_extraction = self.data_processor.extract_medical_fields(state.get("deidentified_medical", {}))
            state["medical_extraction"] = medical_extraction
            logger.info(f"📋 Medical extraction completed: {len(medical_extraction.get('hlth_srvc_records', []))} health service records found")
            
            pharmacy_extraction = self.data_processor.extract_pharmacy_fields(state.get("deidentified_pharmacy", {}))
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
        """LangGraph Node 4: Extract health entities using enhanced extraction"""
        logger.info("🎯 LangGraph Node 4: Starting enhanced entity extraction...")
        state["current_step"] = "extract_entities"
        state["step_status"]["extract_entities"] = "running"
        
        try:
            pharmacy_data = state.get("pharmacy_output", {})
            pharmacy_extraction = state.get("pharmacy_extraction", {})
            medical_extraction = state.get("medical_extraction", {})
            
            # Use data processor for entity extraction
            entities = self.data_processor.extract_health_entities_enhanced(
                pharmacy_data, pharmacy_extraction, medical_extraction
            )
            state["entity_extraction"] = entities
            
            state["step_status"]["extract_entities"] = "completed"
            logger.info("✅ Successfully extracted enhanced health entities")
            
        except Exception as e:
            error_msg = f"Error extracting entities: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_entities"] = "error"
            logger.error(error_msg)
        
        return state
    
    def analyze_trajectory(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """LangGraph Node 5: Analyze health trajectory using Snowflake Cortex"""
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
            
            # Use API integrator for LLM call
            response = self.api_integrator.call_llm(trajectory_prompt)
            
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
        """LangGraph Node 6: Generate final health summary"""
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
            
            # Use API integrator for LLM call
            response = self.api_integrator.call_llm(summary_prompt)
            
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
        """LangGraph Node 8: Predict heart attack risk using FastAPI server"""
        logger.info("❤️ LangGraph Node 8: Starting heart attack prediction with FastAPI server...")
        state["current_step"] = "predict_heart_attack"
        state["step_status"]["predict_heart_attack"] = "running"
        
        try:
            # Extract features from health data
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
            
            # Make async prediction using API integrator
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                prediction_result = loop.run_until_complete(
                    self.api_integrator.call_fastapi_heart_attack_prediction(fastapi_features)
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
                
                # Create SIMPLIFIED prediction result
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
        """LangGraph Node 9: Initialize interactive chatbot with all context"""
        logger.info("💬 LangGraph Node 9: Initializing interactive chatbot...")
        state["current_step"] = "initialize_chatbot"
        state["step_status"]["initialize_chatbot"] = "running"
        
        try:
            # Prepare chatbot context with all data
            chatbot_context = {
                "deidentified_medical": state.get("deidentified_medical", {}),
                "deidentified_pharmacy": state.get("deidentified_pharmacy", {}),
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
                    "heart_attack_risk_level": state.get("heart_attack_prediction", {}).get("risk_level", "unknown"),
                    "model_type": "fastapi_server"
                }
            }
            
            state["chat_history"] = []
            state["chatbot_context"] = chatbot_context
            state["chatbot_ready"] = True
            state["processing_complete"] = True
            state["step_status"]["initialize_chatbot"] = "completed"
            
            logger.info("✅ Successfully initialized interactive chatbot")
            
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
    
    # ===== CHATBOT FUNCTIONALITY =====
    
    def chat_with_data(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Handle chatbot conversation with comprehensive medical data access"""
        try:
            # Use data processor to prepare chunked context
            comprehensive_context = self.data_processor.prepare_chunked_context(chat_context)
            
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
            
            # Use API integrator for LLM call
            response = self.api_integrator.call_llm(comprehensive_prompt)
            
            if response.startswith("Error"):
                return "I encountered an error processing your question. Please try rephrasing your question."
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chatbot conversation: {str(e)}")
            return "I encountered an error processing your question. Please try again."
    
    # ===== HELPER METHODS =====
    
    def _extract_heart_attack_features_for_fastapi(self, state: HealthAnalysisState) -> Dict[str, Any]:
        """Extract features specifically for FastAPI model"""
        try:
            features = {}
            
            # Get patient age
            deidentified_medical = state.get("deidentified_medical", {})
            patient_age = deidentified_medical.get("src_mbr_age", None)
            
            if patient_age and patient_age != "unknown":
                try:
                    age_value = int(float(str(patient_age)))
                    if 0 <= age_value <= 120:
                        features["Age"] = age_value
                    else:
                        features["Age"] = 50
                except:
                    features["Age"] = 50
            else:
                features["Age"] = 50
            
            # Get gender
            patient_data = state.get("patient_data", {})
            gender = str(patient_data.get("gender", "F")).upper()
            features["Gender"] = 1 if gender in ["M", "MALE", "1"] else 0
            
            # Extract features from entity extraction
            entity_extraction = state.get("entity_extraction", {})
            
            diabetes = str(entity_extraction.get("diabetics", "no")).lower()
            features["Diabetes"] = 1 if diabetes in ["yes", "true", "1"] else 0
            
            blood_pressure = str(entity_extraction.get("blood_pressure", "unknown")).lower()
            features["High_BP"] = 1 if blood_pressure in ["managed", "diagnosed", "yes", "true", "1"] else 0
            
            smoking = str(entity_extraction.get("smoking", "no")).lower()
            features["Smoking"] = 1 if smoking in ["yes", "true", "1"] else 0
            
            # Final validation - ensure all values are integers
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
                "feature_interpretation": {
                    "Age": f"{features['Age']} years old",
                    "Gender": "Male" if features["Gender"] == 1 else "Female",
                    "Diabetes": "Yes" if features["Diabetes"] == 1 else "No",
                    "High_BP": "Yes" if features["High_BP"] == 1 else "No",
                    "Smoking": "Yes" if features["Smoking"] == 1 else "No"
                }
            }
            
            return feature_summary
            
        except Exception as e:
            logger.error(f"Error extracting heart attack features: {e}")
            return {"error": f"Feature extraction failed: {str(e)}"}

    def _prepare_fastapi_features(self, features: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """Prepare feature data for FastAPI server call"""
        try:
            extracted_features = features.get("extracted_features", {})
            
            fastapi_features = {
                "age": int(extracted_features.get("Age", 50)),
                "gender": int(extracted_features.get("Gender", 0)),
                "diabetes": int(extracted_features.get("Diabetes", 0)),
                "high_bp": int(extracted_features.get("High_BP", 0)),
                "smoking": int(extracted_features.get("Smoking", 0))
            }
            
            # Validate ranges
            if fastapi_features["age"] < 0 or fastapi_features["age"] > 120:
                fastapi_features["age"] = 50
            
            for key in ["gender", "diabetes", "high_bp", "smoking"]:
                if fastapi_features[key] not in [0, 1]:
                    fastapi_features[key] = 0
            
            return fastapi_features
            
        except Exception as e:
            logger.error(f"Error preparing FastAPI features: {e}")
            return None
    
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
        """Test the Snowflake Cortex API connection"""
        return self.api_integrator.test_llm_connection()
    
    async def test_fastapi_connection(self) -> Dict[str, Any]:
        """Test the FastAPI server connection"""
        return await self.api_integrator.test_fastapi_connection()
    
    def run_analysis(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the enhanced health analysis workflow using LangGraph"""
        
        # Initialize enhanced state for LangGraph
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
            
            logger.info("🚀 Starting Enhanced LangGraph health analysis workflow...")
            
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
                "enhancement_version": "v4.0_modular_architecture"
            }
            
            if results["success"]:
                logger.info("✅ Enhanced LangGraph health analysis completed successfully!")
            else:
                logger.error(f"❌ Enhanced LangGraph health analysis failed with errors: {final_state['errors']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Fatal error in Enhanced LangGraph workflow: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "patient_data": patient_data,
                "errors": [str(e)],
                "processing_steps_completed": 0,
                "langgraph_used": True,
                "enhancement_version": "v4.0_modular_architecture"
            }
    
    def _count_completed_steps(self, state: HealthAnalysisState) -> int:
        """Count how many processing steps were completed"""
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
    """Example usage of the Enhanced Modular Health Analysis Agent"""
    
    print("🏥 Enhanced Modular Health Analysis Agent v4.0")
    print("✅ Modular architecture with separated concerns:")
    print("   📡 HealthAPIIntegrator - All API calls")
    print("   🔧 HealthDataProcessor - Data processing & extraction")
    print("   🏗️ HealthAnalysisAgent - Core workflow orchestration")
    print()
    
    config = Config()
    print("📋 Configuration:")
    print(f"   🌐 Snowflake API: {config.api_url}")
    print(f"   🤖 Model: {config.model}")
    print(f"   ❤️ FastAPI Server: {config.heart_attack_api_url}")
    print()
    print("✅ Modular Health Agent ready!")
    print("🚀 Run: from health_agent_core import HealthAnalysisAgent, Config")
    
    return "Modular Health Agent ready for integration"

if __name__ == "__main__":
    main()
