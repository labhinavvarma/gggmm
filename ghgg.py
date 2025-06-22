import json
import re
import requests
import urllib3
import uuid
from datetime import datetime, date
from typing import Dict, Any, List, TypedDict, Literal, Optional
from dataclasses import dataclass, asdict
import logging

# LangGraph imports - these are required
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Disable SSL warnings (only do this in internal/dev environments)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration with Snowflake Cortex API
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
    chatbot_sys_msg: str = "You are a powerful healthcare AI assistant with access to deidentified medical records. Provide accurate, detailed analysis based on the medical and pharmacy data provided. Always maintain patient privacy and provide professional medical insights."
    max_retries: int = 3
    timeout: int = 30
    
    def to_dict(self):
        return asdict(self)

# Enhanced State Definition for LangGraph (7 nodes including chatbot)
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
    
    # NEW: Chatbot functionality
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
        
        logger.info("🔧 HealthAnalysisAgent initialized with Snowflake Cortex API + Interactive Chatbot")
        logger.info(f"🌐 API URL: {self.config.api_url}")
        logger.info(f"🤖 Model: {self.config.model}")
        logger.info(f"🔑 App ID: {self.config.app_id}")
        logger.info(f"💬 Chatbot: Enhanced Interactive Mode")
        
        self.setup_langgraph()
    
    def setup_langgraph(self):
        """Setup LangGraph workflow - 7 node enhanced workflow with chatbot"""
        logger.info("🔧 Setting up Enhanced LangGraph workflow with 7 nodes (including chatbot)...")
        
        # Create the StateGraph
        workflow = StateGraph(HealthAnalysisState)
        
        # Add all 7 processing nodes (6 original + 1 new chatbot node)
        workflow.add_node("fetch_api_data", self.fetch_api_data)
        workflow.add_node("deidentify_data", self.deidentify_data)
        workflow.add_node("extract_medical_pharmacy_data", self.extract_medical_pharmacy_data)
        workflow.add_node("extract_entities", self.extract_entities)
        workflow.add_node("analyze_trajectory", self.analyze_trajectory)
        workflow.add_node("generate_summary", self.generate_summary)
        workflow.add_node("initialize_chatbot", self.initialize_chatbot)  # NEW NODE 7
        workflow.add_node("handle_error", self.handle_error)
        
        # Define the enhanced workflow edges (7 nodes)
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
                "continue": "initialize_chatbot",  # Continue to chatbot after summary
                "error": "handle_error"
            }
        )
        
        # NEW: Chatbot node routing
        workflow.add_edge("initialize_chatbot", END)  # Chatbot is the final step
        workflow.add_edge("handle_error", END)
        
        # Compile with checkpointer for persistence and reliability
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)
        
        logger.info("✅ Enhanced LangGraph workflow compiled successfully with 7 nodes including chatbot!")
    
    def call_llm(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Call Snowflake Cortex API with the user message"""
        try:
            session_id = str(uuid.uuid4())  # Unique session ID per message
            
            # Use custom system message if provided, otherwise use default
            sys_msg = system_message or self.config.sys_msg
            
            logger.info(f"🤖 Calling Snowflake Cortex API: {self.config.api_url}")
            logger.info(f"🤖 Model: {self.config.model}")
            logger.info(f"🤖 Message length: {len(user_message)} characters")
            logger.info(f"🔑 Session ID: {session_id}")
            
            # Build payload according to Snowflake Cortex API format
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
            
            logger.info(f"🤖 Payload preview: {str(payload)[:200]}...")
            
            # Headers for Snowflake Cortex API
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "Authorization": f'Snowflake Token="{self.config.api_key}"'
            }
            
            logger.info(f"🌐 Making POST request to: {self.config.api_url}")
            
            # Make POST request to Snowflake Cortex API
            response = requests.post(
                self.config.api_url, 
                headers=headers, 
                json=payload, 
                verify=False,  # As specified in original code
                timeout=self.config.timeout
            )
            
            logger.info(f"🤖 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    raw = response.text
                    logger.info(f"🤖 Raw response length: {len(raw)} characters")
                    
                    # Parse response - handle end_of_stream marker as in original code
                    if "end_of_stream" in raw:
                        answer, _, _ = raw.partition("end_of_stream")
                        bot_reply = answer.strip()
                        logger.info("✅ Snowflake Cortex API call successful - found end_of_stream marker")
                    else:
                        bot_reply = raw.strip()
                        logger.info("✅ Snowflake Cortex API call successful - no end_of_stream marker")
                    
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
    
    # ===== LANGGRAPH NODES (7 NODES INCLUDING CHATBOT) =====
    
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
            
            # Call the FastAPI endpoint to get all data
            response = requests.post(
                f"{self.config.fastapi_url}/all", 
                json=patient_data, 
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                api_data = response.json()
                
                # Extract individual API outputs
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
            # Deidentify Medical Data with specific field transformations
            medical_data = state.get("medical_output", {})
            deidentified_medical = self._deidentify_medical_data(medical_data, state["patient_data"])
            state["deidentified_medical"] = deidentified_medical
            
            # Deidentify Pharmacy Data (standard PII removal)
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
            # Extract medical data fields
            medical_extraction = self._extract_medical_fields(state.get("deidentified_medical", {}))
            state["medical_extraction"] = medical_extraction
            logger.info(f"📋 Medical extraction completed: {len(medical_extraction.get('hlth_srvc_records', []))} health service records found")
            
            # Extract pharmacy data fields
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
            # Use both original pharmacy data and new extractions for entity detection
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
            # Prepare enhanced data for Snowflake Cortex analysis
            deidentified_medical = state.get("deidentified_medical", {})
            deidentified_pharmacy = state.get("deidentified_pharmacy", {})
            medical_extraction = state.get("medical_extraction", {})
            pharmacy_extraction = state.get("pharmacy_extraction", {})
            entities = state.get("entity_extraction", {})
            
            # Create enhanced trajectory prompt
            trajectory_prompt = self._create_enhanced_trajectory_prompt(
                deidentified_medical, deidentified_pharmacy, 
                medical_extraction, pharmacy_extraction, entities
            )
            
            logger.info("🤖 Calling Snowflake Cortex for enhanced health trajectory analysis...")
            
            # Get Snowflake Cortex analysis
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
            # Create enhanced summary prompt
            summary_prompt = self._create_enhanced_summary_prompt(
                state.get("health_trajectory", ""), 
                state.get("entity_extraction", {}),
                state.get("medical_extraction", {}),
                state.get("pharmacy_extraction", {})
            )
            
            logger.info("🤖 Calling Snowflake Cortex for enhanced final summary generation...")
            
            # Get final summary from Snowflake Cortex
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
    
    def initialize_chatbot(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """LangGraph Node 7: Initialize interactive chatbot with deidentified data context"""
        logger.info("💬 LangGraph Node 7: Initializing interactive chatbot...")
        state["current_step"] = "initialize_chatbot"
        state["step_status"]["initialize_chatbot"] = "running"
        
        try:
            # Prepare chatbot context with all deidentified data
            chatbot_context = {
                "deidentified_medical": state.get("deidentified_medical", {}),
                "deidentified_pharmacy": state.get("deidentified_pharmacy", {}),
                "medical_extraction": state.get("medical_extraction", {}),
                "pharmacy_extraction": state.get("pharmacy_extraction", {}),
                "entity_extraction": state.get("entity_extraction", {}),
                "health_trajectory": state.get("health_trajectory", ""),
                "final_summary": state.get("final_summary", ""),
                "patient_overview": {
                    "age": state.get("deidentified_medical", {}).get("src_mbr_age", "unknown"),
                    "zip": state.get("deidentified_medical", {}).get("src_mbr_zip_cd", "unknown"),
                    "analysis_timestamp": datetime.now().isoformat()
                }
            }
            
            # Initialize chat history
            state["chat_history"] = []
            state["chatbot_context"] = chatbot_context
            state["chatbot_ready"] = True
            state["processing_complete"] = True
            state["step_status"]["initialize_chatbot"] = "completed"
            
            logger.info("✅ Successfully initialized interactive chatbot with deidentified data context")
            logger.info(f"💬 Chatbot ready with {len(chatbot_context)} data components")
            
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
                state["errors"] = []  # Clear errors for retry
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
    
    # ===== CHATBOT FUNCTIONALITY =====
    
    def chat_with_data(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Handle chatbot conversation with deidentified medical data context"""
        try:
            # Prepare context with medical and pharmacy data
            json_context = {
                "deidentified_medical_data": chat_context.get("deidentified_medical", {}),
                "deidentified_pharmacy_data": chat_context.get("deidentified_pharmacy", {}),
                "medical_extractions": chat_context.get("medical_extraction", {}),
                "pharmacy_extractions": chat_context.get("pharmacy_extraction", {}),
                "health_entities": chat_context.get("entity_extraction", {}),
                "health_trajectory_analysis": chat_context.get("health_trajectory", ""),
                "clinical_summary": chat_context.get("final_summary", ""),
                "patient_overview": chat_context.get("patient_overview", {})
            }
            
            # Build conversation history
            history_text = ""
            if chat_history:
                recent_history = chat_history[-5:]  # Last 5 exchanges
                history_text = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in recent_history
                ])
            
            # Create comprehensive prompt with medical context
            full_prompt = f"""{self.config.chatbot_sys_msg}

Here are the complete deidentified medical records and analysis for this patient:

{json.dumps(json_context, indent=2)}

Previous conversation:
{history_text}

User Question: {user_query}

Please provide a detailed, professional medical analysis based on the deidentified data. Focus on:
1. Relevant medical findings from the data
2. Clinical interpretation of the extracted information
3. Potential health implications
4. Professional medical insights based on the available data

Always maintain patient privacy and provide evidence-based responses using the medical data provided."""

            logger.info(f"💬 Processing chatbot query: {user_query[:100]}...")
            
            # Call Snowflake Cortex with chatbot system message
            response = self.call_llm(full_prompt, self.config.chatbot_sys_msg)
            
            if response.startswith("Error"):
                return f"I apologize, but I encountered an error processing your question: {response}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chatbot conversation: {str(e)}")
            return f"I apologize, but I encountered an error processing your question: {str(e)}"
    
    # ===== HELPER METHODS (same as before, keeping them for completeness) =====
    
    def _extract_medical_fields(self, deidentified_medical: Dict[str, Any]) -> Dict[str, Any]:
        """Extract hlth_srvc_cd and diag_1_50_cd fields from deidentified medical data"""
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
            
            # Get the medical data section
            medical_data = deidentified_medical.get("medical_data", {})
            if not medical_data:
                logger.warning("No medical data found in deidentified medical data")
                return extraction_result
            
            # Recursively search for hlth_srvc_cd and diag_1_50_cd fields
            self._recursive_medical_extraction(medical_data, extraction_result)
            
            # Convert sets to lists for JSON serialization
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
        """Recursively search for medical fields in nested data structures"""
        if isinstance(data, dict):
            current_record = {}
            
            # Look for hlth_srvc_cd
            if "hlth_srvc_cd" in data:
                current_record["hlth_srvc_cd"] = data["hlth_srvc_cd"]
                result["extraction_summary"]["unique_service_codes"].add(str(data["hlth_srvc_cd"]))
            
            # Look for diag_1_50_cd (nested diagnosis codes)
            diagnosis_codes = []
            for i in range(1, 51):  # diag_1_cd through diag_50_cd
                diag_key = f"diag_{i}_cd"
                if diag_key in data and data[diag_key]:
                    diagnosis_codes.append({
                        "code": data[diag_key],
                        "position": i
                    })
                    result["extraction_summary"]["unique_diagnosis_codes"].add(str(data[diag_key]))
            
            if diagnosis_codes:
                current_record["diagnosis_codes"] = diagnosis_codes
                result["extraction_summary"]["total_diagnosis_codes"] += len(diagnosis_codes)
            
            # If we found either service codes or diagnosis codes, add to results
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
            
            # Get the pharmacy data section
            pharmacy_data = deidentified_pharmacy.get("pharmacy_data", {})
            if not pharmacy_data:
                logger.warning("No pharmacy data found in deidentified pharmacy data")
                return extraction_result
            
            # Recursively search for Ndc and lbl_nm fields
            self._recursive_pharmacy_extraction(pharmacy_data, extraction_result)
            
            # Convert sets to lists for JSON serialization
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
            
            # Look for Ndc (case insensitive)
            for key in data:
                if key.lower() in ['ndc', 'ndc_code', 'ndc_number']:
                    current_record["ndc"] = data[key]
                    result["extraction_summary"]["unique_ndc_codes"].add(str(data[key]))
                    break
            
            # Look for lbl_nm (case insensitive)
            for key in data:
                if key.lower() in ['lbl_nm', 'label_name', 'drug_name', 'medication_name']:
                    current_record["lbl_nm"] = data[key]
                    result["extraction_summary"]["unique_label_names"].add(str(data[key]))
                    break
            
            # If we found either NDC or label name, add to results
            if current_record:
                current_record["data_path"] = path
                result["ndc_records"].append(current_record)
                result["extraction_summary"]["total_ndc_records"] += 1
            
            # Continue recursive search
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
            
            # Calculate age from date of birth
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
            
            # Create deidentified structure
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
                # Remove common PII patterns
                data = re.sub(r'\b\d{3}-?\d{2}-?\d{4}\b', '[SSN_MASKED]', data)  # SSN
                data = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[NAME_MASKED]', data)  # Names
                data = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_MASKED]', data)  # Phone
                data = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_MASKED]', data)  # Email
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
            # Original pharmacy data analysis
            if pharmacy_data:
                data_str = json.dumps(pharmacy_data).lower()
                self._analyze_pharmacy_for_entities(data_str, entities)
            
            # Enhanced analysis using pharmacy extraction
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                self._analyze_pharmacy_extraction_for_entities(pharmacy_extraction, entities)
            
            # Medical code analysis
            if medical_extraction and medical_extraction.get("hlth_srvc_records"):
                self._analyze_medical_extraction_for_entities(medical_extraction, entities)
            
            entities["analysis_details"].append(f"Total analysis sources: Pharmacy data, {len(pharmacy_extraction.get('ndc_records', []))} pharmacy records, {len(medical_extraction.get('hlth_srvc_records', []))} medical records")
            
        except Exception as e:
            logger.error(f"Error in enhanced entity extraction: {e}")
            entities["analysis_details"].append(f"Error in entity extraction: {str(e)}")
        
        return entities
    
    def _analyze_pharmacy_for_entities(self, data_str: str, entities: Dict[str, Any]):
        """Original pharmacy data analysis for entities"""
        # Diabetes indicators
        diabetes_keywords = [
            'insulin', 'metformin', 'glipizide', 'diabetes', 'diabetic', 
            'glucophage', 'lantus', 'humalog', 'novolog', 'levemir'
        ]
        for keyword in diabetes_keywords:
            if keyword in data_str:
                entities["diabetics"] = "yes"
                entities["analysis_details"].append(f"Diabetes indicator found in pharmacy data: {keyword}")
                break
        
        # Age group estimation from medications
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
                
                # Analyze label name for conditions
                lbl_lower = lbl_nm.lower()
                
                # Enhanced diabetes detection
                if any(word in lbl_lower for word in ['insulin', 'metformin', 'glucophage', 'diabetes']):
                    entities["diabetics"] = "yes"
                    entities["analysis_details"].append(f"Diabetes medication found in extraction: {lbl_nm}")
                
                # Enhanced blood pressure detection
                if any(word in lbl_lower for word in ['lisinopril', 'amlodipine', 'metoprolol', 'blood pressure']):
                    entities["blood_pressure"] = "managed"
                    entities["analysis_details"].append(f"Blood pressure medication found in extraction: {lbl_nm}")
    
    def _analyze_medical_extraction_for_entities(self, medical_extraction: Dict[str, Any], entities: Dict[str, Any]):
        """Analyze medical codes for health conditions"""
        hlth_srvc_records = medical_extraction.get("hlth_srvc_records", [])
        
        # ICD-10 diagnosis code mappings for common conditions
        condition_mappings = {
            "diabetes": ["E10", "E11", "E12", "E13", "E14"],  # Diabetes mellitus codes
            "hypertension": ["I10", "I11", "I12", "I13", "I15"],  # Hypertension codes
            "smoking": ["Z72.0", "F17"],  # Tobacco use and dependence
            "alcohol": ["F10", "Z72.1"],  # Alcohol related disorders
        }
        
        for record in hlth_srvc_records:
            # Analyze diagnosis codes
            diagnosis_codes = record.get("diagnosis_codes", [])
            for diag in diagnosis_codes:
                diag_code = diag.get("code", "")
                if diag_code:
                    # Check against condition mappings
                    for condition, code_prefixes in condition_mappings.items():
                        if any(diag_code.startswith(prefix) for prefix in code_prefixes):
                            if condition == "diabetes":
                                entities["diabetics"] = "yes"
                                entities["medical_conditions"].append(f"Diabetes (ICD-10: {diag_code})")
                            elif condition == "hypertension":
                                entities["blood_pressure"] = "diagnosed"
                                entities["medical_conditions"].append(f"Hypertension (ICD-10: {diag_code})")
                            
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
        """Run the enhanced health analysis workflow using LangGraph with chatbot"""
        
        # Initialize enhanced state for LangGraph (7 nodes)
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
            # Configure for thread safety
            config_dict = {"configurable": {"thread_id": f"health_analysis_{datetime.now().timestamp()}"}}
            
            # Run the enhanced LangGraph workflow
            logger.info("🚀 Starting Enhanced LangGraph health analysis workflow with chatbot...")
            logger.info(f"🔧 Snowflake Model: {self.config.model}")
            logger.info(f"🔧 FastAPI: {self.config.fastapi_url}")
            logger.info(f"💬 Chatbot: Interactive mode enabled")
            
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
                "chatbot_ready": final_state["chatbot_ready"],
                "chatbot_context": final_state["chatbot_context"],
                "chat_history": final_state["chat_history"],
                "errors": final_state["errors"],
                "processing_steps_completed": self._count_completed_steps(final_state),
                "step_status": final_state["step_status"],
                "langgraph_used": True,
                "enhancement_version": "v3.0_with_interactive_chatbot"
            }
            
            if results["success"]:
                logger.info("✅ Enhanced LangGraph health analysis with chatbot completed successfully!")
                logger.info(f"📊 Medical records extracted: {len(final_state.get('medical_extraction', {}).get('hlth_srvc_records', []))}")
                logger.info(f"💊 Pharmacy records extracted: {len(final_state.get('pharmacy_extraction', {}).get('ndc_records', []))}")
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
                "chatbot_ready": False,
                "chatbot_context": {},
                "chat_history": [],
                "errors": [str(e)],
                "processing_steps_completed": 0,
                "step_status": {"workflow": "error"},
                "langgraph_used": True,
                "enhancement_version": "v3.0_with_interactive_chatbot"
            }
    
    def _count_completed_steps(self, state: HealthAnalysisState) -> int:
        """Count how many processing steps were completed (7 nodes)"""
        steps = 0
        if state.get("mcid_output"): steps += 1
        if state.get("deidentified_medical") and not state.get("deidentified_medical", {}).get("error"): steps += 1
        if state.get("medical_extraction") or state.get("pharmacy_extraction"): steps += 1
        if state.get("entity_extraction"): steps += 1
        if state.get("health_trajectory"): steps += 1
        if state.get("final_summary"): steps += 1
        if state.get("chatbot_ready"): steps += 1  # NEW: Count chatbot initialization
        return steps

def main():
    """Example usage of the Enhanced LangGraph Health Analysis Agent with Interactive Chatbot"""
    
    print("🏥 Enhanced LangGraph Health Analysis Agent v3.0")
    print("✅ Agent is ready with Snowflake Cortex API + Interactive Chatbot")
    print("🔧 To use this agent, run: streamlit run streamlit_langgraph_ui.py")
    print()
    
    # Show Snowflake Cortex configuration
    config = Config()
    print("📋 Snowflake Cortex Configuration:")
    print(f"   🌐 API URL: {config.api_url}")
    print(f"   🤖 Model: {config.model}")
    print(f"   🔑 App ID: {config.app_id}")
    print(f"   🏢 Application Code: {config.aplctn_cd}")
    print(f"   💬 System Message: {config.sys_msg[:50]}...")
    print(f"   🤖 Chatbot Message: {config.chatbot_sys_msg[:50]}...")
    print(f"   ⏱️  Timeout: {config.timeout}s")
    print()
    print("✅ Snowflake Cortex API is configured and ready to use!")
    print("💬 Interactive chatbot enabled for medical data Q&A!")
    print("🚀 Run Streamlit to start the health analysis workflow")
    
    return "Enhanced Agent with Interactive Chatbot ready for Streamlit integration"

if __name__ == "__main__":
    main()
