import json
import re
import requests
import urllib3
import uuid
from datetime import datetime, date
from typing import Dict, Any, List, TypedDict, Literal, Optional, Callable
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

# Enhanced State Definition for MCP-powered LangGraph
class HealthAnalysisState(TypedDict):
    # Chat interface data
    user_message: str
    parsed_patient_data: Dict[str, Any]
    
    # MCP Tool results
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
    
    # Chatbot functionality
    chatbot_ready: bool
    chatbot_context: Dict[str, Any]
    analysis_complete: bool
    
    # Control flow
    current_step: str
    errors: List[str]
    retry_count: int
    processing_complete: bool
    step_status: Dict[str, str]

class MCPHealthAgent:
    def __init__(self, custom_config: Optional[Config] = None, mcp_tools: Optional[Dict[str, Callable]] = None):
        # Use provided config or create default
        if custom_config:
            self.config = custom_config
        else:
            self.config = Config()
        
        # Store MCP tools
        self.mcp_tools = mcp_tools or {}
        
        logger.info("🔧 MCP Health Agent initialized with Snowflake Cortex API + MCP Tools")
        logger.info(f"🌐 API URL: {self.config.api_url}")
        logger.info(f"🤖 Model: {self.config.model}")
        logger.info(f"🔑 App ID: {self.config.app_id}")
        logger.info(f"💬 Chatbot: MCP-powered chat interface")
        logger.info(f"🛠️ MCP Tools: {list(self.mcp_tools.keys())}")
        
        self.setup_langgraph()
    
    def setup_langgraph(self):
        """Setup LangGraph workflow - MCP-powered 7 node workflow"""
        logger.info("🔧 Setting up MCP-powered LangGraph workflow with 7 nodes...")
        
        # Create the StateGraph
        workflow = StateGraph(HealthAnalysisState)
        
        # Add all 7 processing nodes
        workflow.add_node("parse_patient_data", self.parse_patient_data)
        workflow.add_node("fetch_mcp_data", self.fetch_mcp_data)
        workflow.add_node("deidentify_data", self.deidentify_data)
        workflow.add_node("extract_medical_pharmacy_data", self.extract_medical_pharmacy_data)
        workflow.add_node("extract_entities", self.extract_entities)
        workflow.add_node("analyze_trajectory", self.analyze_trajectory)
        workflow.add_node("generate_summary", self.generate_summary)
        workflow.add_node("initialize_chatbot", self.initialize_chatbot)
        workflow.add_node("handle_error", self.handle_error)
        
        # Define the workflow edges
        workflow.add_edge(START, "parse_patient_data")
        
        # Conditional edges with retry logic
        workflow.add_conditional_edges(
            "parse_patient_data",
            self.should_continue_after_parsing,
            {
                "continue": "fetch_mcp_data",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "fetch_mcp_data",
            self.should_continue_after_mcp,
            {
                "continue": "deidentify_data",
                "retry": "fetch_mcp_data", 
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
                "continue": "initialize_chatbot",
                "error": "handle_error"
            }
        )
        
        # Chatbot is the final step
        workflow.add_edge("initialize_chatbot", END)
        workflow.add_edge("handle_error", END)
        
        # Compile with checkpointer for persistence and reliability
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)
        
        logger.info("✅ MCP-powered LangGraph workflow compiled successfully with 7 nodes!")
    
    def call_llm(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Call Snowflake Cortex API with the user message"""
        try:
            session_id = str(uuid.uuid4())
            sys_msg = system_message or self.config.sys_msg
            
            logger.info(f"🤖 Calling Snowflake Cortex API: {self.config.api_url}")
            
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
                raw = response.text
                if "end_of_stream" in raw:
                    answer, _, _ = raw.partition("end_of_stream")
                    bot_reply = answer.strip()
                else:
                    bot_reply = raw.strip()
                return bot_reply
            else:
                return f"API Error {response.status_code}: {response.text}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    # ===== MCP-POWERED LANGGRAPH NODES =====
    
    def parse_patient_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 1: Parse patient data from chat message using LLM"""
        logger.info("🔍 Node 1: Parsing patient data from chat message...")
        state["current_step"] = "parse_patient_data"
        state["step_status"]["parse_patient_data"] = "running"
        
        try:
            user_message = state["user_message"]
            
            # Use LLM to extract patient data from natural language
            extraction_prompt = f"""
Extract patient information from this message and return a JSON object with these exact fields:
- first_name (string)
- last_name (string) 
- ssn (string, numbers only)
- date_of_birth (string, YYYY-MM-DD format)
- gender (string, "M" or "F")
- zip_code (string, 5 digits)

Message: {user_message}

Return only the JSON object, no other text. If any information is missing, use null for that field.
"""
            
            response = self.call_llm(extraction_prompt)
            
            try:
                # Parse the JSON response
                patient_data = json.loads(response)
                
                # Validate required fields
                required_fields = ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"]
                missing_fields = [field for field in required_fields if not patient_data.get(field)]
                
                if missing_fields:
                    state["errors"].append(f"Missing patient information: {', '.join(missing_fields)}")
                    state["step_status"]["parse_patient_data"] = "error"
                else:
                    state["parsed_patient_data"] = patient_data
                    state["step_status"]["parse_patient_data"] = "completed"
                    logger.info(f"✅ Successfully parsed patient data: {patient_data['first_name']} {patient_data['last_name']}")
                
            except json.JSONDecodeError:
                state["errors"].append(f"Failed to parse patient data from message. LLM response: {response}")
                state["step_status"]["parse_patient_data"] = "error"
                
        except Exception as e:
            error_msg = f"Error parsing patient data: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["parse_patient_data"] = "error"
            logger.error(error_msg)
        
        return state
    
    def fetch_mcp_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 2: Fetch data using MCP tools"""
        logger.info("🔗 Node 2: Fetching data using MCP tools...")
        state["current_step"] = "fetch_mcp_data"
        state["step_status"]["fetch_mcp_data"] = "running"
        
        try:
            patient_data = state["parsed_patient_data"]
            
            # Call MCP tools to get all data
            if "get_token" in self.mcp_tools:
                logger.info("🔑 Calling MCP tool: get_token")
                token_result = self.mcp_tools["get_token"]()
                state["token_output"] = token_result
            
            if "mcid_search" in self.mcp_tools:
                logger.info("🆔 Calling MCP tool: mcid_search")
                mcid_result = self.mcp_tools["mcid_search"](patient_data)
                state["mcid_output"] = mcid_result
            
            if "medical_submit" in self.mcp_tools:
                logger.info("🏥 Calling MCP tool: medical_submit")
                medical_result = self.mcp_tools["medical_submit"](patient_data)
                state["medical_output"] = medical_result
            
            if "pharmacy_submit" in self.mcp_tools:
                logger.info("💊 Calling MCP tool: pharmacy_submit")
                pharmacy_result = self.mcp_tools["pharmacy_submit"](patient_data)
                state["pharmacy_output"] = pharmacy_result
            
            state["step_status"]["fetch_mcp_data"] = "completed"
            logger.info("✅ Successfully fetched all MCP data")
                
        except Exception as e:
            error_msg = f"Error fetching MCP data: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["fetch_mcp_data"] = "error"
            logger.error(error_msg)
        
        return state
    
    def deidentify_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 3: Deidentify medical and pharmacy data"""
        logger.info("🔒 Node 3: Starting data deidentification...")
        state["current_step"] = "deidentify_data"
        state["step_status"]["deidentify_data"] = "running"
        
        try:
            # Deidentify Medical Data
            medical_data = state.get("medical_output", {})
            deidentified_medical = self._deidentify_medical_data(medical_data, state["parsed_patient_data"])
            state["deidentified_medical"] = deidentified_medical
            
            # Deidentify Pharmacy Data
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
        """Node 4: Extract specific fields from deidentified medical and pharmacy data"""
        logger.info("🔍 Node 4: Starting medical and pharmacy data extraction...")
        state["current_step"] = "extract_medical_pharmacy_data"
        state["step_status"]["extract_medical_pharmacy_data"] = "running"
        
        try:
            # Extract medical data fields
            medical_extraction = self._extract_medical_fields(state.get("deidentified_medical", {}))
            state["medical_extraction"] = medical_extraction
            
            # Extract pharmacy data fields
            pharmacy_extraction = self._extract_pharmacy_fields(state.get("deidentified_pharmacy", {}))
            state["pharmacy_extraction"] = pharmacy_extraction
            
            state["step_status"]["extract_medical_pharmacy_data"] = "completed"
            logger.info("✅ Successfully extracted medical and pharmacy structured data")
            
        except Exception as e:
            error_msg = f"Error extracting medical/pharmacy data: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_medical_pharmacy_data"] = "error"
            logger.error(error_msg)
        
        return state
    
    def extract_entities(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 5: Extract health entities"""
        logger.info("🎯 Node 5: Starting enhanced entity extraction...")
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
            
        except Exception as e:
            error_msg = f"Error extracting entities: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_entities"] = "error"
            logger.error(error_msg)
        
        return state
    
    def analyze_trajectory(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 6: Analyze health trajectory using Snowflake Cortex"""
        logger.info("📈 Node 6: Starting health trajectory analysis...")
        state["current_step"] = "analyze_trajectory"
        state["step_status"]["analyze_trajectory"] = "running"
        
        try:
            # Prepare data for analysis
            deidentified_medical = state.get("deidentified_medical", {})
            deidentified_pharmacy = state.get("deidentified_pharmacy", {})
            medical_extraction = state.get("medical_extraction", {})
            pharmacy_extraction = state.get("pharmacy_extraction", {})
            entities = state.get("entity_extraction", {})
            
            # Create trajectory prompt
            trajectory_prompt = self._create_trajectory_prompt(
                deidentified_medical, deidentified_pharmacy, 
                medical_extraction, pharmacy_extraction, entities
            )
            
            response = self.call_llm(trajectory_prompt)
            
            if response.startswith("Error"):
                state["errors"].append(f"Health trajectory analysis failed: {response}")
                state["step_status"]["analyze_trajectory"] = "error"
            else:
                state["health_trajectory"] = response
                state["step_status"]["analyze_trajectory"] = "completed"
                logger.info("✅ Successfully analyzed health trajectory")
            
        except Exception as e:
            error_msg = f"Error analyzing trajectory: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["analyze_trajectory"] = "error"
            logger.error(error_msg)
        
        return state
    
    def generate_summary(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 7: Generate final health summary"""
        logger.info("📋 Node 7: Generating final health summary...")
        state["current_step"] = "generate_summary"
        state["step_status"]["generate_summary"] = "running"
        
        try:
            # Create summary prompt
            summary_prompt = self._create_summary_prompt(
                state.get("health_trajectory", ""), 
                state.get("entity_extraction", {}),
                state.get("medical_extraction", {}),
                state.get("pharmacy_extraction", {})
            )
            
            response = self.call_llm(summary_prompt)
            
            if response.startswith("Error"):
                state["errors"].append(f"Summary generation failed: {response}")
                state["step_status"]["generate_summary"] = "error"
            else:
                state["final_summary"] = response
                state["step_status"]["generate_summary"] = "completed"
                state["analysis_complete"] = True
                logger.info("✅ Successfully generated final summary")
            
        except Exception as e:
            error_msg = f"Error generating summary: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["generate_summary"] = "error"
            logger.error(error_msg)
        
        return state
    
    def initialize_chatbot(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 8: Initialize interactive chatbot with data context"""
        logger.info("💬 Node 8: Initializing interactive chatbot...")
        state["current_step"] = "initialize_chatbot"
        state["step_status"]["initialize_chatbot"] = "running"
        
        try:
            # Prepare chatbot context with all data
            chatbot_context = {
                "patient_data": state.get("parsed_patient_data", {}),
                "deidentified_medical": state.get("deidentified_medical", {}),
                "deidentified_pharmacy": state.get("deidentified_pharmacy", {}),
                "medical_extraction": state.get("medical_extraction", {}),
                "pharmacy_extraction": state.get("pharmacy_extraction", {}),
                "entity_extraction": state.get("entity_extraction", {}),
                "health_trajectory": state.get("health_trajectory", ""),
                "final_summary": state.get("final_summary", ""),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
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
        """Error handling node"""
        logger.error(f"🚨 Error Handler: {state['current_step']}")
        logger.error(f"Errors: {state['errors']}")
        
        state["processing_complete"] = True
        current_step = state.get("current_step", "unknown")
        state["step_status"][current_step] = "error"
        return state
    
    # ===== CONDITIONAL EDGES =====
    
    def should_continue_after_parsing(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"
    
    def should_continue_after_mcp(self, state: HealthAnalysisState) -> Literal["continue", "retry", "error"]:
        if state["errors"]:
            if state["retry_count"] < self.config.max_retries:
                state["retry_count"] += 1
                state["errors"] = []
                return "retry"
            else:
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
    
    # ===== CHATBOT FUNCTIONALITY =====
    
    def chat_with_patient_data(self, user_query: str, chat_context: Dict[str, Any]) -> str:
        """Handle chatbot conversation with patient data context"""
        try:
            # Create comprehensive prompt with patient context
            full_prompt = f"""{self.config.chatbot_sys_msg}

Here are the complete patient medical records and analysis:

PATIENT INFO:
{json.dumps(chat_context.get('patient_data', {}), indent=2)}

MEDICAL DATA:
{json.dumps(chat_context.get('deidentified_medical', {}), indent=2)}

PHARMACY DATA:
{json.dumps(chat_context.get('deidentified_pharmacy', {}), indent=2)}

MEDICAL EXTRACTIONS:
{json.dumps(chat_context.get('medical_extraction', {}), indent=2)}

PHARMACY EXTRACTIONS:
{json.dumps(chat_context.get('pharmacy_extraction', {}), indent=2)}

HEALTH ENTITIES:
{json.dumps(chat_context.get('entity_extraction', {}), indent=2)}

HEALTH TRAJECTORY ANALYSIS:
{chat_context.get('health_trajectory', '')}

CLINICAL SUMMARY:
{chat_context.get('final_summary', '')}

User Question: {user_query}

Please provide a detailed, professional medical analysis based on the patient data. Focus on relevant medical findings and provide evidence-based insights."""

            response = self.call_llm(full_prompt, self.config.chatbot_sys_msg)
            
            if response.startswith("Error"):
                return f"I apologize, but I encountered an error: {response}"
            
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"
    
    # ===== HELPER METHODS =====
    
    def _deidentify_medical_data(self, medical_data: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deidentify medical data"""
        try:
            if not medical_data:
                return {"error": "No medical data to deidentify"}
            
            # Calculate age
            try:
                dob_str = patient_data.get('date_of_birth', '')
                if dob_str:
                    dob = datetime.strptime(dob_str, '%Y-%m-%d').date()
                    today = date.today()
                    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                else:
                    age = "unknown"
            except:
                age = "unknown"
            
            return {
                "src_mbr_first_nm": "john",
                "src_mbr_last_nm": "smith", 
                "src_mbr_mid_init_nm": None,
                "src_mbr_age": age,
                "src_mbr_zip_cd": "12345",
                "medical_data": self._remove_pii_from_data(medical_data)
            }
            
        except Exception as e:
            return {"error": f"Deidentification failed: {str(e)}"}
    
    def _deidentify_pharmacy_data(self, pharmacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deidentify pharmacy data"""
        try:
            if not pharmacy_data:
                return {"error": "No pharmacy data to deidentify"}
            
            return {
                "pharmacy_data": self._remove_pii_from_data(pharmacy_data)
            }
            
        except Exception as e:
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
                data = re.sub(r'\b\d{3}-?\d{2}-?\d{4}\b', '[SSN_MASKED]', data)
                data = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[NAME_MASKED]', data)
                data = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_MASKED]', data)
                data = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_MASKED]', data)
                return data
            else:
                return data
        except:
            return data
    
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
            
            # Extract from medical records if available
            medical_records = medical_data.get("medical_records", [])
            
            for record in medical_records:
                current_record = {}
                
                # Extract health service code
                if "hlth_srvc_cd" in record:
                    current_record["hlth_srvc_cd"] = record["hlth_srvc_cd"]
                    extraction_result["extraction_summary"]["unique_service_codes"].add(str(record["hlth_srvc_cd"]))
                
                # Extract diagnosis codes (diag_1_cd through diag_50_cd)
                diagnosis_codes = []
                for i in range(1, 51):
                    diag_key = f"diag_{i}_cd"
                    if diag_key in record and record[diag_key]:
                        diagnosis_codes.append({
                            "code": record[diag_key],
                            "position": i
                        })
                        extraction_result["extraction_summary"]["unique_diagnosis_codes"].add(str(record[diag_key]))
                
                if diagnosis_codes:
                    current_record["diagnosis_codes"] = diagnosis_codes
                    extraction_result["extraction_summary"]["total_diagnosis_codes"] += len(diagnosis_codes)
                
                # Add other medical fields
                for field in ["service_date", "provider_id", "facility_code"]:
                    if field in record:
                        current_record[field] = record[field]
                
                if current_record:
                    extraction_result["hlth_srvc_records"].append(current_record)
                    extraction_result["extraction_summary"]["total_hlth_srvc_records"] += 1
            
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
            
            # Extract from pharmacy records if available
            pharmacy_records = pharmacy_data.get("pharmacy_records", [])
            
            for record in pharmacy_records:
                current_record = {}
                
                # Extract NDC code
                if "ndc" in record:
                    current_record["ndc"] = record["ndc"]
                    extraction_result["extraction_summary"]["unique_ndc_codes"].add(str(record["ndc"]))
                
                # Extract label name
                if "lbl_nm" in record:
                    current_record["lbl_nm"] = record["lbl_nm"]
                    extraction_result["extraction_summary"]["unique_label_names"].add(str(record["lbl_nm"]))
                
                # Add other pharmacy fields
                for field in ["generic_name", "brand_name", "strength", "dosage_form", "quantity_dispensed", "days_supply", "fill_date"]:
                    if field in record:
                        current_record[field] = record[field]
                
                if current_record:
                    extraction_result["ndc_records"].append(current_record)
                    extraction_result["extraction_summary"]["total_ndc_records"] += 1
            
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
            # Analyze pharmacy data for medication patterns
            ndc_records = pharmacy_extraction.get("ndc_records", [])
            for record in ndc_records:
                lbl_nm = record.get("lbl_nm", "").lower()
                generic_name = record.get("generic_name", "").lower()
                
                # Store medication info
                entities["medications_identified"].append({
                    "ndc": record.get("ndc", ""),
                    "label_name": record.get("lbl_nm", ""),
                    "generic_name": record.get("generic_name", ""),
                    "brand_name": record.get("brand_name", "")
                })
                
                # Diabetes detection
                diabetes_keywords = ['metformin', 'insulin', 'diabetes', 'glucophage', 'glipizide']
                if any(keyword in lbl_nm or keyword in generic_name for keyword in diabetes_keywords):
                    entities["diabetics"] = "yes"
                    entities["analysis_details"].append(f"Diabetes medication found: {record.get('lbl_nm', '')}")
                
                # Blood pressure detection
                bp_keywords = ['lisinopril', 'amlodipine', 'metoprolol', 'atenolol', 'losartan']
                if any(keyword in lbl_nm or keyword in generic_name for keyword in bp_keywords):
                    entities["blood_pressure"] = "managed"
                    entities["analysis_details"].append(f"Blood pressure medication found: {record.get('lbl_nm', '')}")
            
            # Analyze medical diagnosis codes
            hlth_srvc_records = medical_extraction.get("hlth_srvc_records", [])
            for record in hlth_srvc_records:
                diagnosis_codes = record.get("diagnosis_codes", [])
                for diag in diagnosis_codes:
                    diag_code = diag.get("code", "")
                    
                    # ICD-10 diabetes codes (E10-E14)
                    if diag_code.startswith(('E10', 'E11', 'E12', 'E13', 'E14')):
                        entities["diabetics"] = "yes"
                        entities["medical_conditions"].append(f"Diabetes (ICD-10: {diag_code})")
                        entities["analysis_details"].append(f"Diabetes diagnosis code found: {diag_code}")
                    
                    # ICD-10 hypertension codes (I10-I15)
                    elif diag_code.startswith(('I10', 'I11', 'I12', 'I13', 'I15')):
                        entities["blood_pressure"] = "diagnosed"
                        entities["medical_conditions"].append(f"Hypertension (ICD-10: {diag_code})")
                        entities["analysis_details"].append(f"Hypertension diagnosis code found: {diag_code}")
                    
                    # Tobacco use (Z72.0, F17)
                    elif diag_code.startswith(('Z72.0', 'F17')):
                        entities["smoking"] = "yes"
                        entities["medical_conditions"].append(f"Tobacco use (ICD-10: {diag_code})")
                        entities["analysis_details"].append(f"Smoking/tobacco diagnosis code found: {diag_code}")
                    
                    # Alcohol related (F10, Z72.1)
                    elif diag_code.startswith(('F10', 'Z72.1')):
                        entities["alcohol"] = "yes"
                        entities["medical_conditions"].append(f"Alcohol related (ICD-10: {diag_code})")
                        entities["analysis_details"].append(f"Alcohol diagnosis code found: {diag_code}")
            
            entities["analysis_details"].append(f"Analysis complete: {len(ndc_records)} medications, {len(hlth_srvc_records)} medical records analyzed")
            
        except Exception as e:
            logger.error(f"Error in enhanced entity extraction: {e}")
            entities["analysis_details"].append(f"Error in entity extraction: {str(e)}")
        
        return entities
    
    def _create_trajectory_prompt(self, medical_data: Dict, pharmacy_data: Dict, 
                                medical_extraction: Dict, pharmacy_extraction: Dict, 
                                entities: Dict) -> str:
        """Create trajectory analysis prompt"""
        return f"""
Analyze this patient's health trajectory based on the following data:

MEDICAL DATA: {json.dumps(medical_data, indent=2)}
PHARMACY DATA: {json.dumps(pharmacy_data, indent=2)}
EXTRACTED ENTITIES: {json.dumps(entities, indent=2)}

Provide a comprehensive health trajectory analysis focusing on:
1. Current health status
2. Risk factors identified
3. Medication patterns
4. Chronic conditions
5. Health trends
6. Care recommendations
"""
    
    def _create_summary_prompt(self, trajectory: str, entities: Dict, 
                             medical_extraction: Dict, pharmacy_extraction: Dict) -> str:
        """Create summary prompt"""
        return f"""
Create a concise executive summary based on:

TRAJECTORY ANALYSIS: {trajectory}
HEALTH ENTITIES: {json.dumps(entities, indent=2)}

Include:
1. Health status overview (2-3 sentences)
2. Key risk factors
3. Priority recommendations
4. Follow-up needs

Keep under 250 words.
"""
    
    def run_patient_analysis(self, user_message: str) -> Dict[str, Any]:
        """Run the full patient analysis workflow from chat message"""
        
        # Initialize state
        initial_state = HealthAnalysisState(
            user_message=user_message,
            parsed_patient_data={},
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
            analysis_complete=False,
            current_step="",
            errors=[],
            retry_count=0,
            processing_complete=False,
            step_status={}
        )
        
        try:
            # Configure for thread safety
            config_dict = {"configurable": {"thread_id": f"mcp_analysis_{datetime.now().timestamp()}"}}
            
            logger.info("🚀 Starting MCP-powered patient analysis...")
            
            # Run the workflow
            final_state = self.graph.invoke(initial_state, config=config_dict)
            
            # Prepare results
            results = {
                "success": final_state["processing_complete"] and not final_state["errors"] and final_state.get("analysis_complete", False),
                "patient_data": final_state["parsed_patient_data"],
                "chatbot_ready": final_state["chatbot_ready"],
                "chatbot_context": final_state["chatbot_context"],
                "health_trajectory": final_state["health_trajectory"],
                "final_summary": final_state["final_summary"],
                "entity_extraction": final_state["entity_extraction"],
                "errors": final_state["errors"],
                "step_status": final_state["step_status"]
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Fatal error in MCP workflow: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "patient_data": {},
                "chatbot_ready": False,
                "chatbot_context": {},
                "errors": [str(e)]
            }

# Example MCP tool functions - these show how to make structured payload calls
def create_mcp_tools():
    """Create MCP tools that make structured payload calls to MCP server"""
    
    def get_token():
        """MCP tool to get authentication token"""
        # Structured payload for MCP server
        payload = {
            "method": "get_token",
            "params": {
                "service": "health_agent",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # This would call the actual MCP server
        # response = mcp_client.call(payload)
        # For now, return example response
        return {
            "token": "Bearer_eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9",
            "expires_in": 3600,
            "status": "success"
        }
    
    def mcid_search(patient_data):
        """MCP tool for MCID search with structured patient payload"""
        # Structured payload for MCP server
        payload = {
            "method": "mcid_search",
            "params": {
                "patient": {
                    "first_name": patient_data.get("first_name", ""),
                    "last_name": patient_data.get("last_name", ""),
                    "ssn": patient_data.get("ssn", ""),
                    "date_of_birth": patient_data.get("date_of_birth", ""),
                    "gender": patient_data.get("gender", ""),
                    "zip_code": patient_data.get("zip_code", "")
                },
                "search_type": "comprehensive",
                "include_history": True
            }
        }
        
        # This would call the actual MCP server
        # response = mcp_client.call(payload)
        # For now, return example response
        return {
            "mcid": f"MCID_{patient_data.get('ssn', 'UNKNOWN')}_2024",
            "patient_match": True,
            "confidence_score": 0.95,
            "matched_records": 3,
            "status": "success"
        }
    
    def medical_submit(patient_data):
        """MCP tool for medical data submission with structured payload"""
        # Structured payload for MCP server
        payload = {
            "method": "medical_submit",
            "params": {
                "patient_identifier": {
                    "first_name": patient_data.get("first_name", ""),
                    "last_name": patient_data.get("last_name", ""),
                    "ssn": patient_data.get("ssn", ""),
                    "date_of_birth": patient_data.get("date_of_birth", ""),
                    "gender": patient_data.get("gender", ""),
                    "zip_code": patient_data.get("zip_code", "")
                },
                "request_type": "full_medical_history",
                "include_fields": [
                    "hlth_srvc_cd",
                    "diag_1_cd", "diag_2_cd", "diag_3_cd", "diag_4_cd", "diag_5_cd",
                    "diag_6_cd", "diag_7_cd", "diag_8_cd", "diag_9_cd", "diag_10_cd",
                    "diag_11_cd", "diag_12_cd", "diag_13_cd", "diag_14_cd", "diag_15_cd",
                    "diag_16_cd", "diag_17_cd", "diag_18_cd", "diag_19_cd", "diag_20_cd",
                    "diag_21_cd", "diag_22_cd", "diag_23_cd", "diag_24_cd", "diag_25_cd",
                    "diag_26_cd", "diag_27_cd", "diag_28_cd", "diag_29_cd", "diag_30_cd",
                    "diag_31_cd", "diag_32_cd", "diag_33_cd", "diag_34_cd", "diag_35_cd",
                    "diag_36_cd", "diag_37_cd", "diag_38_cd", "diag_39_cd", "diag_40_cd",
                    "diag_41_cd", "diag_42_cd", "diag_43_cd", "diag_44_cd", "diag_45_cd",
                    "diag_46_cd", "diag_47_cd", "diag_48_cd", "diag_49_cd", "diag_50_cd",
                    "service_date", "provider_id", "facility_code"
                ],
                "date_range": {
                    "start_date": "2020-01-01",
                    "end_date": datetime.now().strftime("%Y-%m-%d")
                }
            }
        }
        
        # This would call the actual MCP server
        # response = mcp_client.call(payload)
        # For now, return example medical data
        return {
            "body": {
                "medical_records": [
                    {
                        "hlth_srvc_cd": "99213",
                        "diag_1_cd": "E11.9",
                        "diag_2_cd": "I10",
                        "diag_3_cd": "Z79.4",
                        "service_date": "2024-01-15",
                        "provider_id": "PROV001",
                        "facility_code": "FAC001"
                    },
                    {
                        "hlth_srvc_cd": "99214",
                        "diag_1_cd": "E11.65",
                        "diag_2_cd": "I10",
                        "service_date": "2024-03-22",
                        "provider_id": "PROV002", 
                        "facility_code": "FAC001"
                    }
                ],
                "total_records": 2,
                "patient_id": f"PID_{patient_data.get('ssn', 'UNKNOWN')}"
            },
            "status": "success"
        }
    
    def pharmacy_submit(patient_data):
        """MCP tool for pharmacy data submission with structured payload"""
        # Structured payload for MCP server
        payload = {
            "method": "pharmacy_submit",
            "params": {
                "patient_identifier": {
                    "first_name": patient_data.get("first_name", ""),
                    "last_name": patient_data.get("last_name", ""),
                    "ssn": patient_data.get("ssn", ""),
                    "date_of_birth": patient_data.get("date_of_birth", ""),
                    "gender": patient_data.get("gender", ""),
                    "zip_code": patient_data.get("zip_code", "")
                },
                "request_type": "full_pharmacy_history",
                "include_fields": [
                    "ndc",
                    "lbl_nm",
                    "generic_name",
                    "brand_name",
                    "strength",
                    "dosage_form",
                    "quantity_dispensed",
                    "days_supply",
                    "fill_date",
                    "prescriber_id",
                    "pharmacy_id"
                ],
                "date_range": {
                    "start_date": "2020-01-01",
                    "end_date": datetime.now().strftime("%Y-%m-%d")
                }
            }
        }
        
        # This would call the actual MCP server
        # response = mcp_client.call(payload)
        # For now, return example pharmacy data
        return {
            "body": {
                "pharmacy_records": [
                    {
                        "ndc": "0378-6055-77",
                        "lbl_nm": "METFORMIN HCL 500MG TABLETS",
                        "generic_name": "Metformin Hydrochloride",
                        "brand_name": "Glucophage",
                        "strength": "500mg",
                        "dosage_form": "Tablet",
                        "quantity_dispensed": 60,
                        "days_supply": 30,
                        "fill_date": "2024-01-15",
                        "prescriber_id": "PRES001",
                        "pharmacy_id": "PHARM001"
                    },
                    {
                        "ndc": "0071-0222-23",
                        "lbl_nm": "LISINOPRIL 10MG TABLETS",
                        "generic_name": "Lisinopril",
                        "brand_name": "Prinivil",
                        "strength": "10mg",
                        "dosage_form": "Tablet",
                        "quantity_dispensed": 90,
                        "days_supply": 30,
                        "fill_date": "2024-01-15",
                        "prescriber_id": "PRES001",
                        "pharmacy_id": "PHARM001"
                    },
                    {
                        "ndc": "0378-6055-77",
                        "lbl_nm": "METFORMIN HCL 500MG TABLETS",
                        "generic_name": "Metformin Hydrochloride",
                        "brand_name": "Glucophage",
                        "strength": "500mg",
                        "dosage_form": "Tablet",
                        "quantity_dispensed": 60,
                        "days_supply": 30,
                        "fill_date": "2024-02-15",
                        "prescriber_id": "PRES001",
                        "pharmacy_id": "PHARM001"
                    }
                ],
                "total_records": 3,
                "patient_id": f"PID_{patient_data.get('ssn', 'UNKNOWN')}"
            },
            "status": "success"
        }
    
    return {
        "get_token": get_token,
        "mcid_search": mcid_search,
        "medical_submit": medical_submit,
        "pharmacy_submit": pharmacy_submit
    }

def main():
    """Example usage of MCP Health Agent"""
    print("🏥 MCP-Powered Health Agent")
    print("✅ Chat-driven interface with MCP tool integration")
    
    # Create MCP tools
    mcp_tools = create_mcp_tools()
    
    # Initialize agent
    config = Config()
    agent = MCPHealthAgent(config, mcp_tools)
    
    print("🚀 MCP Health Agent ready for chat-based patient analysis!")
    return agent

if __name__ == "__main__":
    main()
