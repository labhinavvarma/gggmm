import json
import re
import requests
import urllib3
import uuid
import asyncio
from datetime import datetime, date
from typing import Dict, Any, List, TypedDict, Literal, Optional
from dataclasses import dataclass, asdict
import logging

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
class Constants:
    MAX_CONVERSATION_HISTORY = 8
    MAX_CONTENT_LENGTH = 400
    REQUIRED_PATIENT_FIELDS = ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"]
    MCP_ENDPOINTS = {
        "mcid": "/mcid/search",
        "medical": "/medical/submit",
        "pharmacy": "/pharmacy/submit", 
        "token": "/token",
        "all": "/all"
    }

@dataclass
class Config:
    mcp_server_url: str = "http://localhost:8000"
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

# Enhanced State Definition for RAG Mode
class EnhancedChatbotState(TypedDict):
    # Core conversation
    user_message: str
    conversation_history: List[Dict[str, Any]]
    assistant_response: str
    
    # Patient data and analysis
    patient_data: Optional[Dict[str, Any]]
    raw_api_responses: Dict[str, Any]
    deidentified_medical: Dict[str, Any]
    deidentified_pharmacy: Dict[str, Any]
    entity_extraction: Dict[str, Any]
    
    # RAG mode state
    rag_mode: bool
    rag_knowledge_base: Dict[str, Any]
    rag_context: str
    
    # Control flow
    current_step: str
    mode: Literal["analysis", "rag", "general"]
    analysis_ready: bool
    errors: List[str]
    processing_complete: bool

class EnhancedRAGHealthAgent:
    def __init__(self, custom_config: Optional[Config] = None):
        self.config = custom_config or Config()
        logger.info("🤖 Enhanced RAG Healthcare Agent initialized")
        logger.info(f"🔗 MCP Server: {self.config.mcp_server_url}")
        
        self.setup_langgraph()
        
        # Session management
        self.session_conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.current_session_id: Optional[str] = None
        self.rag_knowledge_base: Optional[Dict[str, Any]] = None
        self.rag_mode_active: bool = False
        
    def setup_langgraph(self):
        """Setup enhanced LangGraph workflow with RAG capabilities"""
        logger.info("🔧 Setting up Enhanced RAG LangGraph workflow...")
        
        workflow = StateGraph(EnhancedChatbotState)
        
        # Add processing nodes
        workflow.add_node("process_user_input", self.process_user_input)
        workflow.add_node("extract_patient_data", self.extract_patient_data)
        workflow.add_node("call_mcp_server", self.call_mcp_server)
        workflow.add_node("process_analysis_data", self.process_analysis_data)
        workflow.add_node("setup_rag_mode", self.setup_rag_mode)
        workflow.add_node("rag_query", self.rag_query)
        workflow.add_node("generate_response", self.generate_response)
        
        # Define workflow edges
        workflow.add_edge(START, "process_user_input")
        
        # Enhanced conditional routing
        workflow.add_conditional_edges(
            "process_user_input",
            self.route_user_input,
            {
                "extract_data": "extract_patient_data",
                "rag_query": "rag_query",
                "general_response": "generate_response"
            }
        )
        
        # Analysis flow
        workflow.add_edge("extract_patient_data", "call_mcp_server")
        workflow.add_edge("call_mcp_server", "process_analysis_data")
        workflow.add_edge("process_analysis_data", "setup_rag_mode")
        workflow.add_edge("setup_rag_mode", "generate_response")
        
        # RAG flow
        workflow.add_edge("rag_query", "generate_response")
        
        # All paths end at response generation
        workflow.add_edge("generate_response", END)
        
        # Compile workflow
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)
        
        logger.info("✅ Enhanced RAG LangGraph workflow compiled!")
    
    def call_llm(self, user_message: str, system_message: str = None) -> str:
        """Enhanced LLM call with custom system message"""
        try:
            session_id = str(uuid.uuid4())
            
            sys_msg = system_message or self.config.sys_msg
            
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
                        "messages": [{"role": "user", "content": user_message}]
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
                    return answer.strip()
                return raw.strip()
            else:
                return f"API Error {response.status_code}: {response.text[:500]}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    # ===== LANGGRAPH NODES =====
    
    def process_user_input(self, state: EnhancedChatbotState) -> EnhancedChatbotState:
        """Process user input and determine mode"""
        logger.info("🔄 Processing user input...")
        state["current_step"] = "process_user_input"
        
        user_message = state["user_message"]
        
        # Initialize conversation history if needed
        if "conversation_history" not in state:
            state["conversation_history"] = []
        
        # Add user message to history
        state["conversation_history"].append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Determine mode based on current state
        if self.rag_mode_active and self.rag_knowledge_base:
            state["mode"] = "rag"
            state["rag_mode"] = True
            logger.info("🔍 RAG mode active - user query will be processed against knowledge base")
        else:
            state["mode"] = "analysis" if self._is_analysis_request(user_message) else "general"
            state["rag_mode"] = False
        
        logger.info(f"📝 User message: {user_message[:100]}... | Mode: {state['mode']}")
        return state
    
    def extract_patient_data(self, state: EnhancedChatbotState) -> EnhancedChatbotState:
        """Extract patient data from natural language"""
        logger.info("🔍 Extracting patient data...")
        state["current_step"] = "extract_patient_data"
        
        try:
            user_message = state["user_message"]
            
            extraction_prompt = f"""
You are a healthcare data extraction specialist. Extract patient information from the following message and return it as a valid JSON object.

User message: "{user_message}"

Extract the following fields if available:
- first_name (string)
- last_name (string) 
- ssn (string, numbers only)
- date_of_birth (string, format: YYYY-MM-DD)
- gender (string, "M" or "F")
- zip_code (string)

If any field is missing or unclear, use null for that field.

Return ONLY a valid JSON object with these exact field names. Do not include any other text or explanation.

Example format:
{{
    "first_name": "John",
    "last_name": "Smith", 
    "ssn": "123456789",
    "date_of_birth": "1980-01-15",
    "gender": "M",
    "zip_code": "12345"
}}
"""
            
            extracted_json = self.call_llm(extraction_prompt)
            
            try:
                patient_data = json.loads(extracted_json)
                
                # Validate required fields
                missing_fields = []
                for field in Constants.REQUIRED_PATIENT_FIELDS:
                    if not patient_data.get(field):
                        missing_fields.append(field)
                
                if missing_fields:
                    state["errors"].append(f"Missing required fields: {', '.join(missing_fields)}")
                    state["assistant_response"] = f"I couldn't extract all required patient information. Missing: {', '.join(missing_fields)}. Please provide: first name, last name, SSN, date of birth (YYYY-MM-DD), gender (M/F), and zip code."
                    state["processing_complete"] = True
                    return state
                
                state["patient_data"] = patient_data
                logger.info(f"✅ Extracted patient data: {patient_data['first_name']} {patient_data['last_name']}")
                
            except json.JSONDecodeError:
                state["errors"].append("Failed to parse patient data from LLM response")
                state["assistant_response"] = "I couldn't understand the patient information format. Please provide patient details like: 'Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345'"
                state["processing_complete"] = True
                
        except Exception as e:
            error_msg = f"Error extracting patient data: {str(e)}"
            state["errors"].append(error_msg)
            state["assistant_response"] = "I encountered an error processing the patient information. Please try again with clear patient details."
            state["processing_complete"] = True
            logger.error(error_msg)
        
        return state
    
    def call_mcp_server(self, state: EnhancedChatbotState) -> EnhancedChatbotState:
        """Call all MCP server endpoints"""
        logger.info("📡 Calling MCP server endpoints...")
        state["current_step"] = "call_mcp_server"
        
        try:
            patient_data = state["patient_data"]
            if not patient_data:
                state["errors"].append("No patient data available for MCP calls")
                return state
            
            state["raw_api_responses"] = {}
            successful_calls = 0
            
            for endpoint_name, endpoint_path in Constants.MCP_ENDPOINTS.items():
                try:
                    logger.info(f"📞 Calling {endpoint_name} endpoint...")
                    
                    if endpoint_name == "token":
                        response = requests.post(
                            f"{self.config.mcp_server_url}{endpoint_path}",
                            timeout=self.config.timeout
                        )
                    else:
                        response = requests.post(
                            f"{self.config.mcp_server_url}{endpoint_path}",
                            json=patient_data,
                            timeout=self.config.timeout
                        )
                    
                    if response.status_code == 200:
                        raw_data = response.json()
                        state["raw_api_responses"][endpoint_name] = raw_data
                        successful_calls += 1
                        logger.info(f"✅ {endpoint_name} call successful")
                    else:
                        error_data = {
                            "error": f"HTTP {response.status_code}",
                            "message": response.text[:500]
                        }
                        state["raw_api_responses"][endpoint_name] = error_data
                        logger.warning(f"⚠️ {endpoint_name} call failed: {response.status_code}")
                        
                except Exception as e:
                    error_data = {
                        "error": "Request failed",
                        "message": str(e)
                    }
                    state["raw_api_responses"][endpoint_name] = error_data
                    logger.error(f"❌ {endpoint_name} call error: {str(e)}")
            
            logger.info(f"📊 MCP calls completed: {successful_calls}/5 successful")
            
            if successful_calls == 0:
                state["errors"].append("All MCP server calls failed")
                state["assistant_response"] = "I couldn't connect to the healthcare data services. Please check if the MCP server is running."
                state["processing_complete"] = True
            
        except Exception as e:
            error_msg = f"Error calling MCP server: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    def process_analysis_data(self, state: EnhancedChatbotState) -> EnhancedChatbotState:
        """Process and deidentify the raw API data"""
        logger.info("🔒 Processing and deidentifying analysis data...")
        state["current_step"] = "process_analysis_data"
        
        try:
            raw_responses = state.get("raw_api_responses", {})
            patient_data = state.get("patient_data", {})
            
            # Deidentify medical data
            medical_raw = raw_responses.get("medical", {})
            if medical_raw and not medical_raw.get("error"):
                state["deidentified_medical"] = self._deidentify_medical_data(medical_raw, patient_data)
                logger.info("✅ Medical data deidentified")
            else:
                state["deidentified_medical"] = {"error": "No valid medical data"}
            
            # Deidentify pharmacy data
            pharmacy_raw = raw_responses.get("pharmacy", {})
            if pharmacy_raw and not pharmacy_raw.get("error"):
                state["deidentified_pharmacy"] = self._deidentify_pharmacy_data(pharmacy_raw)
                logger.info("✅ Pharmacy data deidentified")
            else:
                state["deidentified_pharmacy"] = {"error": "No valid pharmacy data"}
            
            # Extract health entities
            entities = self._extract_health_entities(
                state["deidentified_medical"],
                state["deidentified_pharmacy"],
                patient_data
            )
            state["entity_extraction"] = entities
            logger.info("✅ Health entities extracted")
            
        except Exception as e:
            error_msg = f"Error processing analysis data: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    def setup_rag_mode(self, state: EnhancedChatbotState) -> EnhancedChatbotState:
        """Setup RAG mode with the deidentified data as knowledge base"""
        logger.info("🧠 Setting up RAG mode with analysis data...")
        state["current_step"] = "setup_rag_mode"
        
        try:
            # Create comprehensive RAG knowledge base
            rag_knowledge = {
                "patient_info": {
                    "name": f"{state['patient_data'].get('first_name', 'Unknown')} {state['patient_data'].get('last_name', 'Unknown')}",
                    "analysis_timestamp": datetime.now().isoformat()
                },
                "deidentified_medical_data": state["deidentified_medical"],
                "deidentified_pharmacy_data": state["deidentified_pharmacy"],
                "entity_extraction_results": state["entity_extraction"],
                "raw_api_responses": state["raw_api_responses"],
                "api_status": self._get_api_status_summary(state["raw_api_responses"])
            }
            
            # Store in both state and instance variables
            state["rag_knowledge_base"] = rag_knowledge
            state["rag_mode"] = True
            state["analysis_ready"] = True
            
            # Update instance variables for persistence across calls
            self.rag_knowledge_base = rag_knowledge
            self.rag_mode_active = True
            
            # Create RAG context string for better LLM understanding
            state["rag_context"] = self._create_rag_context_string(rag_knowledge)
            
            logger.info("✅ RAG mode setup complete - Ready for knowledge-based queries")
            
        except Exception as e:
            error_msg = f"Error setting up RAG mode: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    def rag_query(self, state: EnhancedChatbotState) -> EnhancedChatbotState:
        """Process queries in RAG mode using the knowledge base"""
        logger.info("🔍 Processing RAG query...")
        state["current_step"] = "rag_query"
        
        try:
            user_question = state["user_message"]
            
            # First try direct data analysis for common queries
            direct_answer = self._try_direct_data_analysis(user_question)
            
            if direct_answer:
                state["assistant_response"] = direct_answer
                logger.info("✅ Direct data analysis answer provided")
            else:
                # Use LLM with RAG context for complex queries
                rag_answer = self._generate_rag_response(user_question, state["conversation_history"])
                state["assistant_response"] = rag_answer
                logger.info("✅ RAG LLM response generated")
            
            state["processing_complete"] = True
            
        except Exception as e:
            error_msg = f"Error in RAG query processing: {str(e)}"
            state["errors"].append(error_msg)
            state["assistant_response"] = f"I encountered an error processing your question: {str(e)}"
            state["processing_complete"] = True
            logger.error(error_msg)
        
        return state
    
    def generate_response(self, state: EnhancedChatbotState) -> EnhancedChatbotState:
        """Generate appropriate response based on mode and state"""
        logger.info("💬 Generating response...")
        state["current_step"] = "generate_response"
        
        try:
            mode = state.get("mode", "general")
            
            if state.get("analysis_ready") and not state.get("assistant_response"):
                # Analysis just completed - provide summary
                state["assistant_response"] = self._generate_analysis_complete_response(state)
            elif not state.get("assistant_response"):
                # No response set yet - generate general response
                if mode == "general":
                    state["assistant_response"] = self._generate_general_response(state)
            
            # Add assistant response to conversation history
            if state.get("assistant_response"):
                state["conversation_history"].append({
                    "role": "assistant",
                    "content": state["assistant_response"],
                    "timestamp": datetime.now().isoformat()
                })
            
            state["processing_complete"] = True
            logger.info("✅ Response generated")
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            state["errors"].append(error_msg)
            state["assistant_response"] = f"I apologize, but I encountered an error: {str(e)}"
            state["processing_complete"] = True
            logger.error(error_msg)
        
        return state
    
    # ===== ROUTING LOGIC =====
    
    def route_user_input(self, state: EnhancedChatbotState) -> Literal["extract_data", "rag_query", "general_response"]:
        """Enhanced routing logic for different modes"""
        user_message = state["user_message"].lower()
        
        # Priority 1: If in RAG mode, route to RAG unless explicitly asking for new analysis
        if self.rag_mode_active and self.rag_knowledge_base:
            # Check if explicitly asking for new patient analysis
            new_analysis_indicators = [
                "new patient", "different patient", "another patient", 
                "analyze patient", "fresh analysis"
            ]
            
            is_new_analysis = any(indicator in user_message for indicator in new_analysis_indicators)
            has_patient_data = self._has_patient_identifiers(state["user_message"])
            
            if is_new_analysis and has_patient_data:
                logger.info("🔄 Routing to new analysis (explicit request while in RAG mode)")
                # Reset RAG mode for new analysis
                self.rag_mode_active = False
                self.rag_knowledge_base = None
                return "extract_data"
            else:
                logger.info("🔍 Routing to RAG query (RAG mode active)")
                return "rag_query"
        
        # Priority 2: Check if this is a patient analysis request
        if self._is_analysis_request(user_message) and self._has_patient_identifiers(state["user_message"]):
            logger.info("🔄 Routing to patient data extraction")
            return "extract_data"
        
        # Priority 3: General response
        logger.info("🔄 Routing to general response")
        return "general_response"
    
    # ===== HELPER METHODS =====
    
    def _is_analysis_request(self, message: str) -> bool:
        """Check if message is requesting patient analysis"""
        analysis_keywords = [
            "analyze", "analysis", "patient", "evaluate", "assess", "check",
            "dob", "date of birth", "ssn", "social security", "zip code"
        ]
        return any(keyword in message.lower() for keyword in analysis_keywords)
    
    def _has_patient_identifiers(self, message: str) -> bool:
        """Check if message contains patient identifiers"""
        # Check for names (capitalized words)
        has_name = any(word.istitle() for word in message.split())
        # Check for numbers (likely SSN, DOB, zip)
        has_numbers = any(char.isdigit() for char in message)
        return has_name and has_numbers
    
    def _create_rag_context_string(self, knowledge_base: Dict[str, Any]) -> str:
        """Create a comprehensive context string for RAG queries"""
        try:
            context_parts = []
            
            # Patient info
            patient_info = knowledge_base.get("patient_info", {})
            context_parts.append(f"PATIENT: {patient_info.get('name', 'Unknown')}")
            
            # API status
            api_status = knowledge_base.get("api_status", {})
            context_parts.append(f"API_CALLS_SUCCESSFUL: {api_status.get('successful', 0)}/5")
            
            # Medical data summary
            medical_data = knowledge_base.get("deidentified_medical_data", {})
            if medical_data and not medical_data.get("error"):
                context_parts.append("MEDICAL_DATA: Available")
            
            # Pharmacy data summary
            pharmacy_data = knowledge_base.get("deidentified_pharmacy_data", {})
            if pharmacy_data and not pharmacy_data.get("error"):
                context_parts.append("PHARMACY_DATA: Available")
            
            # Entity extraction summary
            entities = knowledge_base.get("entity_extraction_results", {})
            if entities:
                context_parts.append(f"ENTITIES_EXTRACTED: {len(entities)} categories analyzed")
            
            return " | ".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error creating RAG context: {e}")
            return "RAG_CONTEXT: Analysis data available"
    
    def _try_direct_data_analysis(self, user_question: str) -> Optional[str]:
        """Try to answer questions by directly analyzing the data"""
        if not self.rag_knowledge_base:
            return None
        
        try:
            question_lower = user_question.lower()
            
            # Get data from knowledge base
            medical_data = self.rag_knowledge_base.get('deidentified_medical_data', {})
            pharmacy_data = self.rag_knowledge_base.get('deidentified_pharmacy_data', {})
            entities = self.rag_knowledge_base.get('entity_extraction_results', {})
            raw_responses = self.rag_knowledge_base.get('raw_api_responses', {})
            
            # Medical claims questions
            if any(phrase in question_lower for phrase in ["medical claims", "number of medical", "how many medical", "count medical"]):
                medical_count = self._count_medical_claims(medical_data, raw_responses.get('medical', {}))
                return f"""📊 **Medical Claims Analysis (RAG Mode):**

**Total Medical Claims Found:** {medical_count['total_claims']}

**Detailed Breakdown:**
- Deidentified medical records: {medical_count['deident_records']}
- Raw medical response records: {medical_count['raw_records']}
- Service entries identified: {medical_count['service_entries']}

**Analysis Details:**
{chr(10).join(medical_count['details'])}

*Source: RAG analysis of deidentified MCP server data*"""
            
            # Pharmacy claims questions
            elif any(phrase in question_lower for phrase in ["pharmacy claims", "number of pharmacy", "how many pharmacy", "count pharmacy"]):
                pharmacy_count = self._count_pharmacy_claims(pharmacy_data, raw_responses.get('pharmacy', {}))
                return f"""💊 **Pharmacy Claims Analysis (RAG Mode):**

**Total Pharmacy Claims Found:** {pharmacy_count['total_claims']}

**Detailed Breakdown:**
- Deidentified pharmacy records: {pharmacy_count['deident_records']}
- Raw pharmacy response records: {pharmacy_count['raw_records']}
- Medication entries: {pharmacy_count['medication_entries']}

**Analysis Details:**
{chr(10).join(pharmacy_count['details'])}

*Source: RAG analysis of deidentified MCP server data*"""
            
            # Medications questions
            elif any(phrase in question_lower for phrase in ["medications found", "number of medications", "how many medications", "list medications"]):
                medications = entities.get('medications_identified', [])
                return f"""💊 **Medications Analysis (RAG Mode):**

**Total Medications Found:** {len(medications)}

**Identified Medications:**
{chr(10).join([f"- {med}" for med in medications]) if medications else "- No specific medications identified in entity extraction"}

**Data Sources Analyzed:**
- Deidentified pharmacy data
- Deidentified medical records  
- Entity extraction patterns

*Source: RAG analysis of deidentified MCP server data*"""
            
            # Medical conditions questions
            elif any(phrase in question_lower for phrase in ["medical conditions", "conditions found", "health conditions", "diagnoses"]):
                conditions = entities.get('medical_conditions', [])
                return f"""🏥 **Medical Conditions Analysis (RAG Mode):**

**Total Conditions Found:** {len(conditions)}

**Identified Conditions:**
{chr(10).join([f"- {condition}" for condition in conditions]) if conditions else "- No specific conditions identified in entity extraction"}

**Health Indicators:**
- Diabetes: {entities.get('diabetes', 'unknown').title()}
- Blood Pressure: {entities.get('blood_pressure', 'unknown').title()}
- Age Group: {entities.get('age_group', 'unknown').title()}

*Source: RAG analysis of deidentified MCP server data*"""
            
            # API status questions
            elif any(phrase in question_lower for phrase in ["api status", "server status", "mcp status", "connection status"]):
                api_status = self.rag_knowledge_base.get('api_status', {})
                return f"""📡 **MCP Server Status (RAG Mode):**

**Overall Status:** {api_status.get('overall_status', 'Unknown')}
**Successful API Calls:** {api_status.get('successful', 0)}/5

**Individual Endpoint Status:**
{chr(10).join(api_status.get('details', ['No status details available']))}

*Source: RAG analysis of MCP server responses*"""
            
            # Entity extraction questions
            elif any(phrase in question_lower for phrase in ["entity extraction", "entities found", "health entities", "extracted data"]):
                return f"""🎯 **Entity Extraction Results (RAG Mode):**

**Health Indicators Analyzed:**
- **Diabetes:** {entities.get('diabetes', 'unknown').upper()}
- **Age Group:** {entities.get('age_group', 'unknown').upper()}  
- **Blood Pressure:** {entities.get('blood_pressure', 'unknown').upper()}
- **Smoking Status:** {entities.get('smoking', 'unknown').upper()}
- **Alcohol Status:** {entities.get('alcohol', 'unknown').upper()}

**Extracted Items:**
- Medical Conditions: {len(entities.get('medical_conditions', []))}
- Medications Identified: {len(entities.get('medications_identified', []))}

**Analysis Details:**
{chr(10).join(entities.get('analysis_details', ['No analysis details available']))}

*Source: RAG analysis of deidentified healthcare data*"""
            
            return None  # No direct match found
            
        except Exception as e:
            logger.error(f"Error in direct data analysis: {e}")
            return f"I encountered an error analyzing the data: {str(e)}"
    
    def _generate_rag_response(self, user_question: str, conversation_history: List[Dict[str, Any]]) -> str:
        """Generate RAG response using LLM with knowledge base context"""
        try:
            # Create comprehensive RAG prompt
            rag_system_message = """You are a healthcare AI assistant operating in RAG (Retrieval-Augmented Generation) mode. You have access to complete deidentified patient healthcare analysis data from MCP server APIs. 

IMPORTANT INSTRUCTIONS:
1. Answer questions ONLY based on the provided deidentified healthcare data
2. Reference specific data points, numbers, and findings from the JSON data
3. Always mention this is based on "deidentified analysis data" for privacy
4. If asked about something not in the data, clearly state it's not available in the current analysis
5. Provide specific, detailed answers with exact numbers and findings when available
6. Maintain professional healthcare communication standards

You are in RAG MODE - all responses should be grounded in the provided healthcare data."""
            
            # Get recent conversation for context
            recent_messages = conversation_history[-Constants.MAX_CONVERSATION_HISTORY:] if len(conversation_history) > Constants.MAX_CONVERSATION_HISTORY else conversation_history
            history_text = ""
            
            for msg in recent_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:Constants.MAX_CONTENT_LENGTH]
                history_text += f"{role.upper()}: {content}\n"
            
            # Create comprehensive data context
            knowledge_context = ""
            if self.rag_knowledge_base:
                knowledge_context = f"""
COMPLETE DEIDENTIFIED HEALTHCARE DATA AVAILABLE FOR RAG ANALYSIS:

DEIDENTIFIED MEDICAL DATA:
{json.dumps(self.rag_knowledge_base.get('deidentified_medical_data', {}), indent=2)}

DEIDENTIFIED PHARMACY DATA:
{json.dumps(self.rag_knowledge_base.get('deidentified_pharmacy_data', {}), indent=2)}

ENTITY EXTRACTION RESULTS:
{json.dumps(self.rag_knowledge_base.get('entity_extraction_results', {}), indent=2)}

API STATUS SUMMARY:
{json.dumps(self.rag_knowledge_base.get('api_status', {}), indent=2)}

RAW MCP SERVER RESPONSES SUMMARY:
{json.dumps({k: "Available" if v and not v.get('error') else f"Error: {v.get('error', 'No data')}" for k, v in self.rag_knowledge_base.get('raw_api_responses', {}).items()}, indent=2)}
"""
            
            rag_prompt = f"""RECENT CONVERSATION CONTEXT:
{history_text}

{knowledge_context}

CURRENT USER QUESTION: {user_question}

Based on the complete deidentified healthcare data provided above, answer the user's question with specific details, numbers, and findings from the data. Always reference that this is based on deidentified analysis data for privacy."""
            
            # Generate response using LLM
            response = self.call_llm(rag_prompt, rag_system_message)
            
            if response.startswith("Error"):
                return "I'm having trouble processing your question in RAG mode. Please try rephrasing or ask something specific about the analysis data."
            
            # Add RAG mode indicator
            return f"{response}\n\n*🧠 RAG Mode: Response based on deidentified healthcare analysis data*"
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return f"I encountered an error in RAG mode: {str(e)}"
    
    def _generate_analysis_complete_response(self, state: EnhancedChatbotState) -> str:
        """Generate response when analysis is complete and RAG mode is activated"""
        try:
            patient_data = state.get("patient_data", {})
            patient_name = f"{patient_data.get('first_name', 'Unknown')} {patient_data.get('last_name', 'Unknown')}"
            entities = state.get("entity_extraction", {})
            
            # Count successful API calls
            raw_responses = state.get("raw_api_responses", {})
            successful_calls = len([k for k, v in raw_responses.items() if v and not v.get("error")])
            
            # Count findings
            conditions = len(entities.get("medical_conditions", []))
            medications = len(entities.get("medications_identified", []))
            
            response = f"""🏥 **Healthcare Analysis Complete - RAG Mode Activated**

**Patient:** {patient_name}  
**Analysis Status:** ✅ Complete and Ready

📊 **MCP Server Results:**
- API Calls Successful: {successful_calls}/5
- Data Retrieved & Processed: ✅ Medical, ✅ Pharmacy, ✅ MCID, ✅ Token, ✅ All

🔒 **Data Processing Complete:**
- Medical data deidentified ✅
- Pharmacy data deidentified ✅  
- Health entities extracted ✅
- **RAG knowledge base created ✅**

🎯 **Key Health Indicators:**
- Age Group: {entities.get('age_group', 'unknown').title()}
- Diabetes: {entities.get('diabetes', 'unknown').title()}
- Blood Pressure: {entities.get('blood_pressure', 'unknown').title()}
- Smoking Status: {entities.get('smoking', 'unknown').title()}
- Alcohol Status: {entities.get('alcohol', 'unknown').title()}

📋 **Analysis Summary:**
- Medical Conditions Identified: {conditions}
- Medications Identified: {medications}

🧠 **RAG MODE NOW ACTIVE!**
I can now answer detailed questions about this specific patient's analysis using the deidentified data as my knowledge base.

**Ask me anything about the analysis:**
- "How many medical claims were found?"
- "What medications were identified?"
- "Show me the medical conditions"
- "What's the API status?"
- "Explain the pharmacy findings"
- "Give me diabetes details"
- "Count the pharmacy claims"

**🔄 Use the Refresh button to exit RAG mode and start a new analysis.**

Ready for your questions about {patient_name}'s healthcare data! 🤖"""

            return response
            
        except Exception as e:
            return f"Analysis completed and RAG mode activated, but I had trouble generating the summary: {str(e)}"
    
    def _generate_general_response(self, state: EnhancedChatbotState) -> str:
        """Generate general response for non-analysis queries"""
        user_message = state.get("user_message", "").lower()
        
        # Check if asking about capabilities
        if any(phrase in user_message for phrase in ["what can you", "capabilities", "what do you do", "help me"]):
            if self.rag_mode_active:
                return """🧠 **I'm in RAG Mode with Patient Analysis Data Loaded!**

I can answer detailed questions about the current patient's healthcare analysis using the deidentified data as my knowledge base.

🔍 **Ask me specific questions like:**
- "How many medical claims were found?"
- "Count the pharmacy claims" 
- "What medications were identified?"
- "Show me the medical conditions"
- "What's the API status?"
- "Give me diabetes details"
- "Explain the pharmacy findings"
- "List all health conditions"
- "What does the entity extraction show?"

📊 **Data Sources I Have Access To:**
- Deidentified medical records
- Deidentified pharmacy data  
- Entity extraction results
- MCP server API responses
- Claims analysis and counts

🔄 **To exit RAG mode and start a new analysis, use the Refresh button.**

**What would you like to know about the current patient's data?**"""
            else:
                return """🤖 **Healthcare Analysis Chatbot - Ready!**

I can analyze patient healthcare data and then answer detailed questions about the results using RAG (Retrieval-Augmented Generation) mode.

📝 **To Start Analysis:**
Give me a patient analysis command like:
- "Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345"

🔄 **My Process:**
1. **Extract patient info** from your command using AI
2. **Call MCP server** (medical, pharmacy, MCID APIs)  
3. **Deidentify data** for privacy compliance
4. **Extract health entities** (diabetes, medications, conditions)
5. **Activate RAG mode** - I become a specialized chatbot for your analysis data
6. **Answer your questions** using the analysis as my knowledge base

🧠 **In RAG Mode, Ask Me:**
- "How many medical claims were found?"
- "Count pharmacy claims"
- "What medications were identified?"
- "Show me medical conditions"
- "What's the API status?"

**Ready to analyze patient data!** 🏥"""
        
        elif self.rag_mode_active:
            return """🧠 **RAG Mode Active - Analysis Data Available**

I have patient analysis data loaded and ready to answer your questions!

**Try asking:**
- "How many medical claims were found?"
- "What medications were identified?"
- "Show me the medical conditions" 
- "Count the pharmacy claims"
- "What's the API status?"
- "Give me entity extraction results"

**Or give me a new patient analysis command to start fresh!**

🔄 **Use Refresh to exit RAG mode**"""
        else:
            return """Hello! I'm your healthcare analysis assistant. 

**To get started:**
- "Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345"

**After analysis, I'll enter RAG mode where you can ask detailed questions about:**
- Medical claims counts
- Pharmacy claims counts  
- Medications identified
- Medical conditions found
- API response status
- Health risk indicators

**What would you like me to analyze?** 🏥"""
    
    def _get_api_status_summary(self, raw_responses: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive API status summary"""
        endpoints = ['mcid', 'medical', 'pharmacy', 'token', 'all']
        successful = 0
        details = []
        
        for endpoint in endpoints:
            response = raw_responses.get(endpoint, {})
            if response and not response.get('error'):
                successful += 1
                details.append(f"✅ {endpoint.upper()}: Success")
            else:
                error_msg = response.get('error', 'No response') if response else 'No response'
                details.append(f"❌ {endpoint.upper()}: {error_msg}")
        
        if successful == 5:
            overall_status = "All endpoints successful"
        elif successful > 0:
            overall_status = f"Partial success ({successful}/5 endpoints)"
        else:
            overall_status = "All endpoints failed"
        
        return {
            'successful': successful,
            'total_endpoints': len(endpoints),
            'details': details,
            'overall_status': overall_status
        }
    
    # Include the existing helper methods for data processing
    def _deidentify_medical_data(self, medical_data: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deidentify medical data"""
        try:
            # Calculate age
            age = "unknown"
            if patient_data.get('date_of_birth'):
                try:
                    dob = datetime.strptime(patient_data['date_of_birth'], '%Y-%m-%d').date()
                    today = date.today()
                    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                except:
                    pass
            
            deidentified = {
                "patient_info": {
                    "first_name": "john",
                    "last_name": "smith",
                    "age": age,
                    "zip_code": "12345"
                },
                "medical_data": self._remove_pii_from_data(medical_data),
                "deidentification_timestamp": datetime.now().isoformat()
            }
            
            return deidentified
            
        except Exception as e:
            logger.error(f"Error in medical deidentification: {e}")
            return {"error": f"Medical deidentification failed: {str(e)}"}
    
    def _deidentify_pharmacy_data(self, pharmacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deidentify pharmacy data"""
        try:
            deidentified = {
                "pharmacy_data": self._remove_pii_from_data(pharmacy_data),
                "deidentification_timestamp": datetime.now().isoformat()
            }
            return deidentified
            
        except Exception as e:
            logger.error(f"Error in pharmacy deidentification: {e}")
            return {"error": f"Pharmacy deidentification failed: {str(e)}"}
    
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
                return data
            else:
                return data
        except:
            return data
    
    def _extract_health_entities(self, medical_data: Dict[str, Any], pharmacy_data: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract health entities from deidentified data"""
        entities = {
            "diabetes": "unknown",
            "age_group": "unknown",
            "blood_pressure": "unknown", 
            "smoking": "unknown",
            "alcohol": "unknown",
            "analysis_details": [],
            "medical_conditions": [],
            "medications_identified": []
        }
        
        try:
            # Calculate age group
            if patient_data.get("date_of_birth"):
                try:
                    dob = datetime.strptime(patient_data["date_of_birth"], '%Y-%m-%d').date()
                    today = date.today()
                    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                    
                    if age < 18:
                        entities["age_group"] = "child"
                    elif age < 65:
                        entities["age_group"] = "adult"
                    else:
                        entities["age_group"] = "senior"
                    
                    entities["analysis_details"].append(f"Age calculated: {age} years")
                except:
                    entities["analysis_details"].append("Could not calculate age")
            
            # Analyze medical data
            if medical_data and not medical_data.get("error"):
                medical_str = json.dumps(medical_data).lower()
                
                # Diabetes indicators
                diabetes_keywords = ['diabetes', 'diabetic', 'insulin', 'glucose', 'a1c', 'metformin']
                for keyword in diabetes_keywords:
                    if keyword in medical_str:
                        entities["diabetes"] = "yes"
                        entities["medical_conditions"].append(f"Diabetes indicator: {keyword}")
                        break
                
                # Blood pressure indicators
                bp_keywords = ['hypertension', 'blood pressure', 'systolic', 'diastolic']
                for keyword in bp_keywords:
                    if keyword in medical_str:
                        entities["blood_pressure"] = "diagnosed"
                        entities["medical_conditions"].append(f"Blood pressure indicator: {keyword}")
                        break
            
            # Analyze pharmacy data
            if pharmacy_data and not pharmacy_data.get("error"):
                pharmacy_str = json.dumps(pharmacy_data).lower()
                
                # Diabetes medications
                diabetes_meds = ['insulin', 'metformin', 'glipizide', 'lantus']
                for med in diabetes_meds:
                    if med in pharmacy_str:
                        entities["diabetes"] = "yes"
                        entities["medications_identified"].append(f"Diabetes medication: {med}")
                
                # Blood pressure medications
                bp_meds = ['lisinopril', 'amlodipine', 'metoprolol', 'losartan']
                for med in bp_meds:
                    if med in pharmacy_str:
                        entities["blood_pressure"] = "managed"
                        entities["medications_identified"].append(f"BP medication: {med}")
                
                # Smoking cessation
                smoking_meds = ['chantix', 'varenicline', 'nicotine']
                for med in smoking_meds:
                    if med in pharmacy_str:
                        entities["smoking"] = "quit_attempt"
                        entities["medications_identified"].append(f"Smoking cessation: {med}")
            
        except Exception as e:
            entities["analysis_details"].append(f"Error in entity extraction: {str(e)}")
        
        return entities
    
    def _count_medical_claims(self, deident_medical: Dict, raw_medical: Dict) -> Dict[str, Any]:
        """Count medical claims from the data"""
        result = {
            'total_claims': 0,
            'deident_records': 0,
            'raw_records': 0,
            'service_entries': 0,
            'details': []
        }
        
        try:
            # Count from deidentified medical data
            if deident_medical and not deident_medical.get('error'):
                medical_data = deident_medical.get('medical_data', {})
                deident_count = self._recursive_count_records(medical_data)
                result['deident_records'] = deident_count
                result['details'].append(f"Deidentified medical records: {deident_count}")
            
            # Count from raw medical data
            if raw_medical and not raw_medical.get('error'):
                raw_count = self._recursive_count_records(raw_medical)
                result['raw_records'] = raw_count
                result['details'].append(f"Raw medical response records: {raw_count}")
            
            # Calculate total
            result['total_claims'] = max(result['deident_records'], result['raw_records'])
            
            if result['total_claims'] == 0:
                result['details'].append("No medical claims found in the data")
            
        except Exception as e:
            result['details'].append(f"Error counting medical claims: {str(e)}")
        
        return result
    
    def _count_pharmacy_claims(self, deident_pharmacy: Dict, raw_pharmacy: Dict) -> Dict[str, Any]:
        """Count pharmacy claims from the data"""
        result = {
            'total_claims': 0,
            'deident_records': 0,
            'raw_records': 0,
            'medication_entries': 0,
            'details': []
        }
        
        try:
            # Count from deidentified pharmacy data
            if deident_pharmacy and not deident_pharmacy.get('error'):
                pharmacy_data = deident_pharmacy.get('pharmacy_data', {})
                deident_count = self._recursive_count_records(pharmacy_data)
                result['deident_records'] = deident_count
                result['details'].append(f"Deidentified pharmacy records: {deident_count}")
            
            # Count from raw pharmacy data
            if raw_pharmacy and not raw_pharmacy.get('error'):
                raw_count = self._recursive_count_records(raw_pharmacy)
                result['raw_records'] = raw_count
                result['details'].append(f"Raw pharmacy response records: {raw_count}")
            
            # Calculate total
            result['total_claims'] = max(result['deident_records'], result['raw_records'])
            
            if result['total_claims'] == 0:
                result['details'].append("No pharmacy claims found in the data")
            
        except Exception as e:
            result['details'].append(f"Error counting pharmacy claims: {str(e)}")
        
        return result
    
    def _recursive_count_records(self, data: Any, record_types: list = None) -> int:
        """Recursively count records in nested data structures"""
        if record_types is None:
            record_types = ['claims', 'records', 'entries', 'items', 'data', 'results']
        
        count = 0
        try:
            if isinstance(data, dict):
                # Count if this dict represents records
                for key in record_types:
                    if key in data and isinstance(data[key], list):
                        count += len(data[key])
                
                # Recursively count in nested structures
                for value in data.values():
                    count += self._recursive_count_records(value, record_types)
                    
            elif isinstance(data, list):
                count += len(data)  # Count list items
                # Also check nested structures
                for item in data:
                    count += self._recursive_count_records(item, record_types)
        except:
            pass
        
        return count
    
    # ===== PUBLIC METHODS =====
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """Main chat interface with enhanced RAG capabilities"""
        try:
            # Create session ID if needed
            if not self.current_session_id:
                self.current_session_id = str(uuid.uuid4())
                self.session_conversations[self.current_session_id] = []
            
            # Initialize state
            initial_state = EnhancedChatbotState(
                user_message=user_message,
                conversation_history=[],
                assistant_response="",
                patient_data=None,
                raw_api_responses={},
                deidentified_medical={},
                deidentified_pharmacy={},
                entity_extraction={},
                rag_mode=self.rag_mode_active,
                rag_knowledge_base=self.rag_knowledge_base or {},
                rag_context="",
                current_step="",
                mode="general",
                analysis_ready=False,
                errors=[],
                processing_complete=False
            )
            
            # Run the enhanced workflow
            config_dict = {"configurable": {"thread_id": f"chat_{self.current_session_id}"}}
            final_state = self.graph.invoke(initial_state, config=config_dict)
            
            # Store conversation in session
            self.session_conversations[self.current_session_id].extend(final_state["conversation_history"])
            
            # Prepare enhanced response
            result = {
                "success": final_state["processing_complete"] and not final_state["errors"],
                "response": final_state["assistant_response"],
                "rag_mode": final_state.get("rag_mode", False),
                "analysis_ready": final_state.get("analysis_ready", False),
                "patient_data": final_state.get("patient_data"),
                "raw_api_responses": final_state.get("raw_api_responses", {}),
                "deidentified_data": {
                    "medical": final_state.get("deidentified_medical", {}),
                    "pharmacy": final_state.get("deidentified_pharmacy", {})
                },
                "entity_extraction": final_state.get("entity_extraction", {}),
                "rag_knowledge_base": final_state.get("rag_knowledge_base", {}),
                "errors": final_state.get("errors", []),
                "session_id": self.current_session_id,
                "conversation_history": self.session_conversations[self.current_session_id],
                "mode": final_state.get("mode", "general")
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in chat processing: {str(e)}")
            return {
                "success": False,
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "rag_mode": self.rag_mode_active,
                "analysis_ready": False,
                "errors": [str(e)],
                "session_id": self.current_session_id,
                "mode": "error"
            }
    
    def refresh_session(self):
        """Enhanced refresh to properly reset RAG mode"""
        # Clear session conversations
        if self.current_session_id and self.current_session_id in self.session_conversations:
            del self.session_conversations[self.current_session_id]
        
        # Reset all instance variables
        self.current_session_id = None
        self.rag_knowledge_base = None
        self.rag_mode_active = False
        
        logger.info("🔄 Enhanced session refreshed - RAG mode deactivated")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get current conversation history"""
        if self.current_session_id and self.current_session_id in self.session_conversations:
            return self.session_conversations[self.current_session_id]
        return []
    
    def get_rag_status(self) -> Dict[str, Any]:
        """Get current RAG mode status"""
        return {
            "rag_active": self.rag_mode_active,
            "has_knowledge_base": self.rag_knowledge_base is not None,
            "knowledge_base_size": len(str(self.rag_knowledge_base)) if self.rag_knowledge_base else 0,
            "session_id": self.current_session_id
        }

def main():
    """Test the enhanced RAG chatbot agent"""
    print("🧠 Enhanced RAG Healthcare Analysis Agent")
    print("=" * 50)
    
    agent = EnhancedRAGHealthAgent()
    
    # Test messages
    test_messages = [
        "Hello, what can you help me with?",
        "Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345",
        "How many medical claims were found?",
        "What medications were identified?",
        "Give me the API status"
    ]
    
    for i, message in enumerate(test_messages):
        print(f"\n👤 User: {message}")
        result = agent.chat(message)
        print(f"🤖 Assistant: {result['response']}")
        
        if result.get("rag_mode"):
            print("🧠 RAG Mode: ACTIVE")
        if result.get("analysis_ready"):
            print("✅ Analysis ready for RAG queries!")
        
        # Show RAG status after each interaction
        rag_status = agent.get_rag_status()
        print(f"📊 RAG Status: {rag_status}")
    
    return agent

if __name__ == "__main__":
    main()
