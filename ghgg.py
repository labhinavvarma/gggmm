
import json
import re
import requests
import urllib3
import uuid
import snowflake.connector
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
    max_retries: int = 3
    timeout: int = 30
    
    def to_dict(self):
        return asdict(self)

# HARDCODED CHATBOT CONFIGURATION - Using exact settings from working chatbot
@dataclass
class ChatbotConfig:
    # Hardcoded Snowflake connection settings - DO NOT CHANGE
    host: str = "Delata-m567-nonprod.privatelink.snowflakecomputing.com"
    user: str = "DElta -2"
    password: str = "DElta -2"
    account: str = "DElta -2.privatelink"
    port: int = 443
    warehouse: str = "D01_EDA_GENAI_USER_WH_S"
    role: str = "NOGBD"
    authenticator: str = "https://p.com//okta"
    
    # Hardcoded LLM settings - DO NOT CHANGE
    model: str = "claude-4-sonnet"
    system_message: str = """You are a medical assistant specialized in analyzing deidentified medical and pharmacy data. 
    Extract and answer health-related questions from the provided deidentified medical record JSON(s) which consists of Medical Claims and Pharmacy Claims data. 
    Focus on diagnoses, medications, procedures, ICD-10 codes, NDC codes, and any other relevant clinical details.
    Always maintain patient privacy and work only with deidentified data."""

# Enhanced State Definition for LangGraph (6 nodes + chatbot)
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
    chatbot_enabled: bool
    conversation_history: List[Dict[str, str]]
    
    # Control flow
    current_step: str
    errors: List[str]
    retry_count: int
    processing_complete: bool
    step_status: Dict[str, str]

class HealthAnalysisAgentWithChatbot:
    def __init__(self, custom_config: Optional[Config] = None):
        # Use provided config or create default
        if custom_config:
            self.config = custom_config
        else:
            self.config = Config()
        
        # Initialize hardcoded chatbot config
        self.chatbot_config = ChatbotConfig()
        
        # Initialize Snowflake connection for chatbot
        self.snowflake_conn = None
        self._init_snowflake_connection()
        
        logger.info("🔧 HealthAnalysisAgent with Chatbot initialized")
        logger.info(f"🌐 Original API URL: {self.config.api_url}")
        logger.info(f"🤖 Original Model: {self.config.model}")
        logger.info(f"💬 Chatbot Model: {self.chatbot_config.model}")
        logger.info(f"🏔️ Snowflake Host: {self.chatbot_config.host}")
        
        self.setup_langgraph()
    
    def _init_snowflake_connection(self):
        """Initialize Snowflake connection for chatbot - using exact settings from working code"""
        try:
            logger.info("🏔️ Initializing Snowflake connection for chatbot...")
            self.snowflake_conn = snowflake.connector.connect(
                user=self.chatbot_config.user,
                password=self.chatbot_config.password,
                account=self.chatbot_config.account,
                host=self.chatbot_config.host,
                port=self.chatbot_config.port,
                warehouse=self.chatbot_config.warehouse,
                role=self.chatbot_config.role,
                authenticator=self.chatbot_config.authenticator
            )
            logger.info("✅ Snowflake connection established successfully")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Snowflake: {e}")
            self.snowflake_conn = None
    
    def setup_langgraph(self):
        """Setup LangGraph workflow - 6 node enhanced workflow + chatbot"""
        logger.info("🔧 Setting up Enhanced LangGraph workflow with 6 nodes + chatbot...")
        
        # Create the StateGraph
        workflow = StateGraph(HealthAnalysisState)
        
        # Add all 6 processing nodes
        workflow.add_node("fetch_api_data", self.fetch_api_data)
        workflow.add_node("deidentify_data", self.deidentify_data)
        workflow.add_node("extract_medical_pharmacy_data", self.extract_medical_pharmacy_data)
        workflow.add_node("extract_entities", self.extract_entities)
        workflow.add_node("analyze_trajectory", self.analyze_trajectory)
        workflow.add_node("generate_summary", self.generate_summary)
        workflow.add_node("handle_error", self.handle_error)
        
        # Define the enhanced workflow edges (6 nodes)
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
        
        # Final edges
        workflow.add_edge("generate_summary", END)
        workflow.add_edge("handle_error", END)
        
        # Compile with checkpointer for persistence and reliability
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)
        
        logger.info("✅ Enhanced LangGraph workflow compiled successfully with 6 nodes + chatbot!")
    
    def call_llm(self, user_message: str) -> str:
        """Call Snowflake Cortex API with the user message (original method)"""
        try:
            session_id = str(uuid.uuid4())
            
            logger.info(f"🤖 Calling Snowflake Cortex API: {self.config.api_url}")
            logger.info(f"🤖 Model: {self.config.model}")
            
            payload = {
                "query": {
                    "aplctn_cd": self.config.aplctn_cd,
                    "app_id": self.config.app_id,
                    "api_key": self.config.api_key,
                    "method": "cortex",
                    "model": self.config.model,
                    "sys_msg": self.config.sys_msg,
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
                raw = response.text
                if "end_of_stream" in raw:
                    answer, _, _ = raw.partition("end_of_stream")
                    bot_reply = answer.strip()
                else:
                    bot_reply = raw.strip()
                return bot_reply
            else:
                return f"API Error {response.status_code}: {response.text[:500]}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def call_chatbot_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call Snowflake Cortex for chatbot - using exact settings from working chatbot"""
        try:
            if not self.snowflake_conn:
                return "❌ Snowflake connection not available"
            
            logger.info(f"💬 Calling Snowflake Cortex for chatbot with {self.chatbot_config.model}")
            
            # Prepare request body using exact format from working chatbot
            request_body = {
                "model": self.chatbot_config.model,
                "messages": messages
            }
            
            # Get token from Snowflake connection
            token = self.snowflake_conn.rest.token
            
            # Make API call using exact format from working chatbot
            resp = requests.post(
                url=f"https://{self.chatbot_config.host}/api/v2/cortex/inference:complete",
                json=request_body,
                headers={
                    "Authorization": f'Snowflake Token="{token}"',
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )
            
            resp.raise_for_status()
            response_chunks = []
            
            if resp.status_code == 200:
                for line in resp.iter_lines():
                    if line:
                        line_content = line.decode('utf-8').replace('data: ', '')
                        try:
                            json_line = json.loads(line_content)
                            response_chunks.append(json_line)
                        except json.JSONDecodeError:
                            logger.warning(f"Error decoding JSON line: {line_content}")
                
                # Extract assistant response from chunks
                assistant_reply = ''.join(
                    chunk["choices"][0]["delta"]["text"]
                    for chunk in response_chunks
                    if "choices" in chunk and chunk["choices"][0].get("delta")
                )
                
                return assistant_reply if assistant_reply else "No response received"
            else:
                return f"❌ Chatbot API Error: {resp.status_code} - {resp.text}"
                
        except Exception as e:
            logger.error(f"❌ Chatbot API Error: {str(e)}")
            return f"❌ Chatbot Error: {str(e)}"
    
    def start_chatbot_session(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Start a chatbot session with deidentified medical and pharmacy data"""
        try:
            logger.info("💬 Starting chatbot session with deidentified data...")
            
            if not self.snowflake_conn:
                return {
                    "success": False,
                    "error": "Snowflake connection not available",
                    "conversation_history": []
                }
            
            # Prepare deidentified data for chatbot context
            deidentified_data = {
                "deidentified_medical": analysis_results.get("deidentified_data", {}).get("medical", {}),
                "deidentified_pharmacy": analysis_results.get("deidentified_data", {}).get("pharmacy", {}),
                "medical_extraction": analysis_results.get("structured_extractions", {}).get("medical", {}),
                "pharmacy_extraction": analysis_results.get("structured_extractions", {}).get("pharmacy", {}),
                "entity_extraction": analysis_results.get("entity_extraction", {}),
                "health_trajectory": analysis_results.get("health_trajectory", ""),
                "final_summary": analysis_results.get("final_summary", "")
            }
            
            # Initialize conversation with system message and context
            conversation_history = [
                {
                    "role": "system",
                    "content": self.chatbot_config.system_message
                },
                {
                    "role": "user",
                    "content": f"Here is the deidentified medical and pharmacy data for analysis:\n{json.dumps(deidentified_data, indent=2)}"
                }
            ]
            
            return {
                "success": True,
                "conversation_history": conversation_history,
                "deidentified_data": deidentified_data,
                "chatbot_ready": True
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting chatbot session: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to start chatbot session: {str(e)}",
                "conversation_history": []
            }
    
    def chat_with_data(self, conversation_history: List[Dict[str, str]], user_question: str) -> Dict[str, Any]:
        """Chat with the deidentified medical/pharmacy data"""
        try:
            logger.info(f"💬 Processing user question: {user_question[:100]}...")
            
            # Add user question to conversation
            conversation_history.append({
                "role": "user",
                "content": user_question
            })
            
            # Get response from chatbot LLM
            assistant_response = self.call_chatbot_llm(conversation_history)
            
            # Add assistant response to conversation
            conversation_history.append({
                "role": "assistant",
                "content": assistant_response
            })
            
            return {
                "success": True,
                "response": assistant_response,
                "conversation_history": conversation_history,
                "question": user_question
            }
            
        except Exception as e:
            logger.error(f"❌ Error in chat: {str(e)}")
            return {
                "success": False,
                "error": f"Chat error: {str(e)}",
                "response": f"❌ Error processing question: {str(e)}",
                "conversation_history": conversation_history
            }
    
    def reset_chatbot_conversation(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Reset the chatbot conversation to start fresh"""
        logger.info("🔄 Resetting chatbot conversation...")
        return self.start_chatbot_session(analysis_results)
    
    # ===== LANGGRAPH NODES (Same as before) =====
    
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
                state["processing_complete"] = True
                state["step_status"]["generate_summary"] = "completed"
                logger.info("✅ Successfully generated enhanced final summary")
            
        except Exception as e:
            error_msg = f"Error generating summary: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["generate_summary"] = "error"
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
    
    # ===== LANGGRAPH CONDITIONAL EDGES (Same as before) =====
    
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
        return "error" if state["errors"] else "continue"
    
    def should_continue_after_extraction_step(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"
    
    def should_continue_after_entity_extraction(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"
    
    def should_continue_after_trajectory(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"
    
    # ===== HELPER METHODS (Same as before) =====
    
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
        """Recursively search for medical fields in nested data structures"""
        if isinstance(data, dict):
            current_record = {}
            
            if "hlth_srvc_cd" in data:
                current_record["hlth_srvc_cd"] = data["hlth_srvc_cd"]
                result["extraction_summary"]["unique_service_codes"].add(str(data["hlth_srvc_cd"]))
            
            diagnosis_codes = []
            for i in range(1, 51):
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
            
            if current_record:
                current_record["data_path"] = path
                result["hlth_srvc_records"].append(current_record)
                result["extraction_summary"]["total_hlth_srvc_records"] += 1
            
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
    
    def test_chatbot_connection(self) -> Dict[str, Any]:
        """Test the Snowflake connection for chatbot functionality"""
        try:
            logger.info("🧪 Testing Snowflake chatbot connection...")
            
            if not self.snowflake_conn:
                return {
                    "success": False,
                    "error": "Snowflake connection not established",
                    "host": self.chatbot_config.host
                }
            
            test_messages = [
                {
                    "role": "system", 
                    "content": "You are a test assistant."
                },
                {
                    "role": "user", 
                    "content": "Please respond with 'Chatbot connection successful'"
                }
            ]
            
            test_response = self.call_chatbot_llm(test_messages)
            
            if test_response.startswith("❌"):
                return {
                    "success": False,
                    "error": test_response,
                    "host": self.chatbot_config.host,
                    "model": self.chatbot_config.model
                }
            else:
                return {
                    "success": True,
                    "response": test_response,
                    "host": self.chatbot_config.host,
                    "model": self.chatbot_config.model
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Chatbot connection test failed: {str(e)}",
                "host": self.chatbot_config.host
            }
    
    def run_analysis_with_chatbot(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the enhanced health analysis workflow with chatbot capability"""
        
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
            chatbot_enabled=True,
            conversation_history=[],
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
            logger.info(f"🔧 Analysis Model: {self.config.model}")
            logger.info(f"💬 Chatbot Model: {self.chatbot_config.model}")
            logger.info(f"🔧 FastAPI: {self.config.fastapi_url}")
            
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
                "errors": final_state["errors"],
                "processing_steps_completed": self._count_completed_steps(final_state),
                "step_status": final_state["step_status"],
                "langgraph_used": True,
                "chatbot_enabled": final_state["chatbot_enabled"],
                "enhancement_version": "v3.0_with_integrated_chatbot"
            }
            
            if results["success"]:
                logger.info("✅ Enhanced LangGraph health analysis with chatbot completed successfully!")
                logger.info(f"📊 Medical records extracted: {len(final_state.get('medical_extraction', {}).get('hlth_srvc_records', []))}")
                logger.info(f"💊 Pharmacy records extracted: {len(final_state.get('pharmacy_extraction', {}).get('ndc_records', []))}")
                logger.info("💬 Chatbot ready for deidentified data interaction")
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
                "errors": [str(e)],
                "processing_steps_completed": 0,
                "step_status": {"workflow": "error"},
                "langgraph_used": True,
                "chatbot_enabled": False,
                "enhancement_version": "v3.0_with_integrated_chatbot"
            }
    
    def _count_completed_steps(self, state: HealthAnalysisState) -> int:
        """Count how many processing steps were completed (6 nodes)"""
        steps = 0
        if state.get("mcid_output"): steps += 1
        if state.get("deidentified_medical") and not state.get("deidentified_medical", {}).get("error"): steps += 1
        if state.get("medical_extraction") or state.get("pharmacy_extraction"): steps += 1
        if state.get("entity_extraction"): steps += 1
        if state.get("health_trajectory"): steps += 1
        if state.get("final_summary"): steps += 1
        return steps

def main():
    """Example usage of the Enhanced LangGraph Health Analysis Agent with Integrated Chatbot"""
    
    print("🏥 Enhanced LangGraph Health Analysis Agent v3.0 with Integrated Chatbot")
    print("✅ Agent is ready with Snowflake Cortex API + Claude-4-Sonnet Chatbot")
    print("🔧 To use this agent, run: streamlit run streamlit_enhanced_ui.py")
    print()
    
    # Show configuration
    config = Config()
    chatbot_config = ChatbotConfig()
    
    print("📋 Analysis Configuration:")
    print(f"   🌐 API URL: {config.api_url}")
    print(f"   🤖 Model: {config.model}")
    print(f"   🔑 App ID: {config.app_id}")
    print()
    
    print("📋 Chatbot Configuration (Hardcoded):")
    print(f"   🏔️ Snowflake Host: {chatbot_config.host}")
    print(f"   🤖 Model: {chatbot_config.model}")
    print(f"   👤 User: {chatbot_config.user}")
    print(f"   🏢 Warehouse: {chatbot_config.warehouse}")
    print(f"   🔑 Role: {chatbot_config.role}")
    print()
    
    print("✅ Both Analysis and Chatbot APIs are configured and ready!")
    print("💬 Chatbot will use deidentified medical and pharmacy data")
    print("🚀 Run Streamlit to start the enhanced health analysis + chatbot workflow")
    
    return "Enhanced Agent with Integrated Chatbot ready for Streamlit integration"

if __name__ == "__main__":
    main()
