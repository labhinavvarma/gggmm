
# enhanced_langgraph_mcp_chatbot.py
import asyncio
import json
import logging
import re
import uuid
import requests
from datetime import datetime, date
from typing import Dict, Any, List, TypedDict, Literal, Optional
from dataclasses import dataclass
import httpx

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable SSL warnings for internal/dev environments
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@dataclass
class MCPChatbotConfig:
    """Configuration for MCP chatbot"""
    mcp_server_url: str = "http://localhost:8000"  # Same port as router
    timeout: int = 30
    max_retries: int = 3
    
    # Snowflake Cortex API Configuration
    cortex_api_url: str = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
    cortex_api_key: str = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
    cortex_app_id: str = "edadip"
    cortex_aplctn_cd: str = "edagnai"
    cortex_model: str = "llama3.1-70b"
    cortex_sys_msg: str = "You are a healthcare AI assistant. Provide accurate, concise answers based on context."

# Enhanced LangGraph State for Continuous Healthcare Chat
class ContinuousHealthChatState(TypedDict):
    # User interaction & context
    user_message: str
    conversation_history: List[Dict[str, str]]
    session_id: str
    user_context: Dict[str, Any]  # Persistent user context
    
    # Intent analysis & routing
    intent: str
    confidence: float
    required_apis: List[str]  # Which APIs to call
    extracted_patient_data: Dict[str, Any]
    missing_fields: List[str]
    
    # API orchestration
    api_responses: Dict[str, Any]
    api_call_status: Dict[str, str]
    
    # Data processing pipeline
    deidentified_data: Dict[str, Any]
    entity_extraction: Dict[str, Any]
    health_analysis: Dict[str, Any]
    
    # Response generation
    conversational_response: str
    follow_up_questions: List[str]
    suggested_actions: List[str]
    
    # Control flow & state
    current_step: str
    errors: List[str]
    needs_clarification: bool
    analysis_complete: bool
    can_answer_question: bool
    step_status: Dict[str, str]

class EnhancedLangGraphMCPChatbot:
    """
    Enhanced LangGraph chatbot that intelligently routes to MCP server endpoints,
    deidentifies data, performs entity extraction, and supports continuous conversations
    """
    
    def __init__(self, config: MCPChatbotConfig = None):
        self.config = config or MCPChatbotConfig()
        self.session_contexts = {}  # Store conversation contexts
        
        logger.info("🤖 Initializing Enhanced LangGraph MCP Healthcare Chatbot")
        logger.info(f"🔗 MCP Server URL: {self.config.mcp_server_url}")
        
        self.setup_enhanced_workflow()
    
    def setup_enhanced_workflow(self):
        """Setup enhanced LangGraph workflow with intelligent routing and processing"""
        logger.info("🔧 Setting up Enhanced LangGraph workflow...")
        
        workflow = StateGraph(ContinuousHealthChatState)
        
        # Enhanced workflow nodes
        workflow.add_node("analyze_intent_and_context", self.analyze_intent_and_context_node)
        workflow.add_node("determine_api_routing", self.determine_api_routing_node)
        workflow.add_node("extract_patient_info", self.extract_patient_info_node)
        workflow.add_node("check_data_requirements", self.check_data_requirements_node)
        workflow.add_node("call_mcp_apis", self.call_mcp_apis_node)
        workflow.add_node("deidentify_responses", self.deidentify_responses_node)
        workflow.add_node("enhanced_entity_extraction", self.enhanced_entity_extraction_node)
        workflow.add_node("analyze_health_data", self.analyze_health_data_node)
        workflow.add_node("generate_conversational_response", self.generate_conversational_response_node)
        workflow.add_node("request_clarification", self.request_clarification_node)
        workflow.add_node("handle_error", self.handle_error_node)
        
        # Define enhanced workflow edges
        workflow.add_edge(START, "analyze_intent_and_context")
        
        workflow.add_conditional_edges(
            "analyze_intent_and_context",
            self.route_after_intent_analysis,
            {
                "need_patient_data": "extract_patient_info",
                "direct_question": "generate_conversational_response",
                "clarification": "request_clarification",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("extract_patient_info", "determine_api_routing")
        workflow.add_edge("determine_api_routing", "check_data_requirements")
        
        workflow.add_conditional_edges(
            "check_data_requirements",
            self.route_after_data_check,
            {
                "ready": "call_mcp_apis",
                "incomplete": "request_clarification",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "call_mcp_apis",
            self.route_after_api_calls,
            {
                "success": "deidentify_responses",
                "partial": "deidentify_responses",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("deidentify_responses", "enhanced_entity_extraction")
        workflow.add_edge("enhanced_entity_extraction", "analyze_health_data")
        workflow.add_edge("analyze_health_data", "generate_conversational_response")
        workflow.add_edge("generate_conversational_response", END)
        workflow.add_edge("request_clarification", END)
        workflow.add_edge("handle_error", END)
        
        # Compile with memory for conversation persistence
        memory = MemorySaver()
        self.workflow = workflow.compile(checkpointer=memory)
        
        logger.info("✅ Enhanced LangGraph workflow compiled successfully!")
    
    # ===== SNOWFLAKE CORTEX LLM INTEGRATION =====
    
    async def call_cortex_llm(self, user_message: str, system_message: str = None) -> str:
        """Call Snowflake Cortex API for LLM responses"""
        try:
            import uuid
            import requests
            
            session_id = str(uuid.uuid4())
            
            # Use provided system message or default
            sys_msg = system_message or self.config.cortex_sys_msg
            
            # Build payload for Snowflake Cortex API
            payload = {
                "query": {
                    "aplctn_cd": self.config.cortex_aplctn_cd,
                    "app_id": self.config.cortex_app_id,
                    "api_key": self.config.cortex_api_key,
                    "method": "cortex",
                    "model": self.config.cortex_model,
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
                "Authorization": f'Snowflake Token="{self.config.cortex_api_key}"'
            }
            
            logger.info(f"🤖 Calling Snowflake Cortex: {self.config.cortex_model}")
            
            response = requests.post(
                self.config.cortex_api_url, 
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
                
                logger.info("✅ Snowflake Cortex response received")
                return bot_reply
            else:
                error_msg = f"Cortex API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                return f"LLM Error: {error_msg}"
                
        except Exception as e:
            error_msg = f"Error calling Cortex LLM: {str(e)}"
            logger.error(error_msg)
            return f"LLM Error: {error_msg}"
    
    # ===== ENHANCED WORKFLOW NODES =====
    
    async def analyze_intent_and_context_node(self, state: ContinuousHealthChatState) -> ContinuousHealthChatState:
        """Analyze user intent and conversation context using Snowflake Cortex LLM"""
        logger.info("🎯 Node: Analyzing intent and context with Snowflake Cortex...")
        state["current_step"] = "analyze_intent_and_context"
        state["step_status"]["analyze_intent_and_context"] = "running"
        
        try:
            user_message = state["user_message"]
            session_id = state["session_id"]
            conversation_history = state["conversation_history"]
            
            # Get or create session context
            if session_id not in self.session_contexts:
                self.session_contexts[session_id] = {
                    "patient_data": {},
                    "previous_analyses": [],
                    "topics_discussed": [],
                    "user_preferences": {}
                }
            
            user_context = self.session_contexts[session_id]
            state["user_context"] = user_context
            
            # Create context-aware prompt for intent analysis
            context_info = ""
            if conversation_history:
                recent_context = conversation_history[-3:]  # Last 3 messages
                context_info = f"Recent conversation context: {recent_context}"
            
            existing_patient_data = user_context.get("patient_data", {})
            if existing_patient_data:
                context_info += f"\nKnown patient information: {list(existing_patient_data.keys())}"
            
            intent_prompt = f"""
Analyze this healthcare message and determine the user's intent and extract information:

Message: "{user_message}"
{context_info}

Classify the intent as one of:
1. "patient_analysis" - User wants to analyze patient health data
2. "medical_question" - User asking about medical terms, codes, conditions  
3. "medication_inquiry" - User asking about medications or pharmacy data
4. "follow_up_question" - User asking about previous analysis results
5. "data_request" - User providing patient information
6. "clarification" - Message unclear or off-topic

Also extract any patient information mentioned (names, ages, conditions, medications, SSN, DOB, etc.).
Determine if we can answer with existing data or need new analysis.

Return JSON format:
{{
    "intent": "...",
    "confidence": 0.95,
    "extracted_patient_data": {{"first_name": "...", "age": 45, ...}},
    "can_answer_with_existing": true/false,
    "requires_new_analysis": true/false,
    "reasoning": "Brief explanation of the intent classification"
}}
"""
            
            # Call Snowflake Cortex for intelligent intent analysis
            cortex_response = await self.call_cortex_llm(
                intent_prompt,
                "You are a healthcare AI assistant specialized in analyzing user intents and extracting patient information. Always respond with valid JSON."
            )
            
            # Parse the LLM response
            try:
                intent_analysis = json.loads(cortex_response)
            except json.JSONDecodeError:
                # Fallback to simple pattern matching if JSON parsing fails
                logger.warning("Could not parse LLM intent analysis, using fallback")
                intent_analysis = await self._fallback_intent_analysis(user_message)
            
            state["intent"] = intent_analysis.get("intent", "clarification")
            state["confidence"] = intent_analysis.get("confidence", 0.5)
            state["can_answer_question"] = intent_analysis.get("can_answer_with_existing", False)
            
            # Extract and merge patient data
            extracted_data = intent_analysis.get("extracted_patient_data", {})
            if extracted_data:
                existing_patient_data.update(extracted_data)
                state["extracted_patient_data"] = existing_patient_data
            
            # Update conversation topics
            user_context["topics_discussed"].append({
                "message": user_message,
                "intent": intent_analysis.get("intent"),
                "reasoning": intent_analysis.get("reasoning"),
                "timestamp": datetime.now().isoformat()
            })
            
            state["step_status"]["analyze_intent_and_context"] = "completed"
            logger.info(f"🎯 Intent: {state['intent']} (confidence: {state['confidence']})")
            
        except Exception as e:
            error_msg = f"Error analyzing intent: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["analyze_intent_and_context"] = "error"
            logger.error(error_msg)
        
        return state
    
    async def determine_api_routing_node(self, state: ContinuousHealthChatState) -> ContinuousHealthChatState:
        """Intelligently determine which API endpoints to call"""
        logger.info("🔀 Node: Determining API routing...")
        state["current_step"] = "determine_api_routing"
        state["step_status"]["determine_api_routing"] = "running"
        
        try:
            intent = state["intent"]
            user_message = state["user_message"].lower()
            user_context = state["user_context"]
            
            # Intelligent API routing based on intent and keywords
            required_apis = []
            
            if intent == "patient_analysis":
                # Full analysis - call all APIs
                required_apis = ["medical", "pharmacy", "mcid", "all"]
                
            elif intent == "medical_question":
                if any(keyword in user_message for keyword in ["medication", "drug", "prescription", "pharmacy"]):
                    required_apis = ["pharmacy"]
                elif any(keyword in user_message for keyword in ["medical", "diagnosis", "icd", "claim"]):
                    required_apis = ["medical"]
                elif any(keyword in user_message for keyword in ["member", "id", "mcid"]):
                    required_apis = ["mcid"]
                else:
                    # Default to comprehensive analysis
                    required_apis = ["all"]
                    
            elif intent == "follow_up_question":
                # Check if we have existing data to answer the question
                if user_context.get("previous_analyses"):
                    required_apis = []  # Can answer from existing data
                else:
                    required_apis = ["all"]  # Need fresh data
                    
            elif intent == "medication_inquiry":
                required_apis = ["pharmacy", "token"]
                
            elif intent == "diagnosis_inquiry":
                required_apis = ["medical", "token"]
                
            else:
                # Default comprehensive analysis
                required_apis = ["all"]
            
            state["required_apis"] = required_apis
            state["step_status"]["determine_api_routing"] = "completed"
            
            logger.info(f"🔀 API routing determined: {required_apis}")
            
        except Exception as e:
            error_msg = f"Error determining API routing: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["determine_api_routing"] = "error"
            logger.error(error_msg)
        
        return state
    
    async def extract_patient_info_node(self, state: ContinuousHealthChatState) -> ContinuousHealthChatState:
        """Extract and merge patient information using Snowflake Cortex LLM"""
        logger.info("👤 Node: Extracting patient information with Snowflake Cortex...")
        state["current_step"] = "extract_patient_info"
        state["step_status"]["extract_patient_info"] = "running"
        
        try:
            user_message = state["user_message"]
            user_context = state["user_context"]
            
            # Create smart extraction prompt with correct field names
            extraction_prompt = f"""
Extract patient information from this healthcare message:

Message: "{user_message}"

Extract any patient data mentioned including:
- first_name (first name)
- last_name (last name/surname) 
- age (convert to approximate date_of_birth if only age given)
- date_of_birth (YYYY-MM-DD format)
- gender (F for female, M for male)
- ssn (Social Security Number)
- zip_code (postal code)
- medical_conditions (any health conditions mentioned)
- medications (any drugs or prescriptions mentioned)

IMPORTANT: Use "last_name" for the last name/surname in the JSON output.

Return JSON format with only the fields that are explicitly mentioned:
{{
    "first_name": "...",
    "last_name": "...",
    "age": 45,
    "date_of_birth": "1979-01-01",
    "gender": "M",
    "ssn": "123456789",
    "zip_code": "12345",
    "medical_conditions": ["diabetes", "hypertension"],
    "medications": ["insulin", "lisinopril"]
}}

Important:
- Only include fields that are clearly mentioned
- Convert age to approximate birth year (current year - age)
- Use M/F for gender
- Remove dashes from SSN
- Use "last_name" for last name/surname
- Return empty JSON {{}} if no patient data found
"""
            
            # Call Snowflake Cortex for intelligent extraction
            cortex_response = await self.call_cortex_llm(
                extraction_prompt,
                "You are a healthcare data extraction specialist. Extract patient information accurately and return valid JSON only."
            )
            
            # Parse extraction results
            try:
                extracted_data = json.loads(cortex_response)
                
                # Process age to date_of_birth conversion
                if "age" in extracted_data and "date_of_birth" not in extracted_data:
                    age = extracted_data["age"]
                    current_year = datetime.now().year
                    birth_year = current_year - int(age)
                    extracted_data["date_of_birth"] = f"{birth_year}-01-01"
                
                # Clean up SSN format
                if "ssn" in extracted_data:
                    extracted_data["ssn"] = str(extracted_data["ssn"]).replace("-", "").replace(" ", "")
                
                # Ensure gender is F or M
                if "gender" in extracted_data:
                    gender = str(extracted_data["gender"]).upper()
                    if gender in ["FEMALE", "WOMAN"]:
                        extracted_data["gender"] = "F"
                    elif gender in ["MALE", "MAN"]:
                        extracted_data["gender"] = "M"
                    else:
                        extracted_data["gender"] = gender[0] if gender else "M"
                
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM extraction, using fallback")
                extracted_data = await self._extract_patient_info_advanced(user_message)
            
            # Merge with existing patient data from context
            existing_patient_data = user_context.get("patient_data", {})
            merged_patient_data = {**existing_patient_data, **extracted_data}
            
            # Update both state and persistent context
            state["extracted_patient_data"] = merged_patient_data
            user_context["patient_data"] = merged_patient_data
            
            state["step_status"]["extract_patient_info"] = "completed"
            logger.info(f"👤 Patient data extracted: {list(extracted_data.keys())}")
            
        except Exception as e:
            error_msg = f"Error extracting patient info: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_patient_info"] = "error"
            logger.error(error_msg)
        
        return state
    
    async def check_data_requirements_node(self, state: ContinuousHealthChatState) -> ContinuousHealthChatState:
        """Check if we have sufficient data for the required API calls"""
        logger.info("✅ Node: Checking data requirements...")
        state["current_step"] = "check_data_requirements"
        state["step_status"]["check_data_requirements"] = "running"
        
        try:
            required_apis = state["required_apis"]
            extracted_data = state["extracted_patient_data"]
            
            # Define requirements for different API types with correct field names
            api_requirements = {
                "medical": ["first_name", "last_name", "date_of_birth", "gender"],
                "pharmacy": ["first_name", "last_name", "date_of_birth", "gender"],
                "mcid": ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"],
                "all": ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"]
            }
            
            missing_fields = set()
            
            for api in required_apis:
                if api in api_requirements:
                    for field in api_requirements[api]:
                        if not extracted_data.get(field):
                            missing_fields.add(field)
            
            state["missing_fields"] = list(missing_fields)
            state["needs_clarification"] = len(missing_fields) > 0
            
            state["step_status"]["check_data_requirements"] = "completed"
            
            if missing_fields:
                logger.warning(f"⚠️ Missing required fields: {missing_fields}")
            else:
                logger.info("✅ All required data available")
            
        except Exception as e:
            error_msg = f"Error checking data requirements: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["check_data_requirements"] = "error"
            logger.error(error_msg)
        
        return state
    
    async def call_mcp_apis_node(self, state: ContinuousHealthChatState) -> ContinuousHealthChatState:
        """Call the determined MCP server endpoints"""
        logger.info("📡 Node: Calling MCP APIs...")
        state["current_step"] = "call_mcp_apis"
        state["step_status"]["call_mcp_apis"] = "running"
        
        try:
            required_apis = state["required_apis"]
            patient_data = state["extracted_patient_data"]
            
            api_responses = {}
            api_call_status = {}
            
            # Call the specific APIs based on routing decision
            for api_name in required_apis:
                try:
                    response = await self._call_specific_mcp_api(api_name, patient_data)
                    api_responses[api_name] = response
                    api_call_status[api_name] = "success" if response.get("status_code") == 200 else "failed"
                    
                except Exception as api_error:
                    logger.error(f"❌ Error calling {api_name} API: {api_error}")
                    api_responses[api_name] = {"status_code": 500, "error": str(api_error)}
                    api_call_status[api_name] = "error"
            
            state["api_responses"] = api_responses
            state["api_call_status"] = api_call_status
            
            # Update user context with API responses
            state["user_context"]["previous_analyses"].append({
                "timestamp": datetime.now().isoformat(),
                "apis_called": required_apis,
                "responses": api_responses
            })
            
            successful_calls = sum(1 for status in api_call_status.values() if status == "success")
            total_calls = len(required_apis)
            
            state["step_status"]["call_mcp_apis"] = "completed"
            logger.info(f"📡 API calls completed: {successful_calls}/{total_calls} successful")
            
        except Exception as e:
            error_msg = f"Error calling MCP APIs: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["call_mcp_apis"] = "error"
            logger.error(error_msg)
        
        return state
    
    async def deidentify_responses_node(self, state: ContinuousHealthChatState) -> ContinuousHealthChatState:
        """Deidentify the API responses"""
        logger.info("🔒 Node: Deidentifying API responses...")
        state["current_step"] = "deidentify_responses"
        state["step_status"]["deidentify_responses"] = "running"
        
        try:
            api_responses = state["api_responses"]
            patient_data = state["extracted_patient_data"]
            
            deidentified_data = {}
            
            for api_name, response in api_responses.items():
                if response.get("status_code") == 200 and response.get("body"):
                    # Deidentify based on API type
                    if api_name in ["medical", "all"]:
                        deidentified_data[f"{api_name}_deidentified"] = await self._deidentify_medical_data(
                            response["body"], patient_data
                        )
                    elif api_name == "pharmacy":
                        deidentified_data[f"{api_name}_deidentified"] = await self._deidentify_pharmacy_data(
                            response["body"]
                        )
                    elif api_name == "mcid":
                        deidentified_data[f"{api_name}_deidentified"] = await self._deidentify_mcid_data(
                            response["body"]
                        )
                    else:
                        # Generic deidentification
                        deidentified_data[f"{api_name}_deidentified"] = await self._deidentify_generic_data(
                            response["body"]
                        )
            
            state["deidentified_data"] = deidentified_data
            state["step_status"]["deidentify_responses"] = "completed"
            
            logger.info(f"🔒 Deidentification completed for {len(deidentified_data)} datasets")
            
        except Exception as e:
            error_msg = f"Error deidentifying responses: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["deidentify_responses"] = "error"
            logger.error(error_msg)
        
        return state
    
    async def enhanced_entity_extraction_node(self, state: ContinuousHealthChatState) -> ContinuousHealthChatState:
        """Perform enhanced entity extraction on deidentified data"""
        logger.info("🎯 Node: Enhanced entity extraction...")
        state["current_step"] = "enhanced_entity_extraction"
        state["step_status"]["enhanced_entity_extraction"] = "running"
        
        try:
            deidentified_data = state["deidentified_data"]
            
            # Enhanced entity extraction
            entity_extraction = {
                "health_conditions": {},
                "medications": {},
                "demographics": {},
                "risk_factors": {},
                "medical_codes": {},
                "analysis_confidence": 0.0
            }
            
            # Extract entities from each deidentified dataset
            for data_type, data in deidentified_data.items():
                if "medical" in data_type:
                    medical_entities = await self._extract_medical_entities(data)
                    entity_extraction["health_conditions"].update(medical_entities.get("conditions", {}))
                    entity_extraction["medical_codes"].update(medical_entities.get("codes", {}))
                    
                elif "pharmacy" in data_type:
                    pharmacy_entities = await self._extract_pharmacy_entities(data)
                    entity_extraction["medications"].update(pharmacy_entities.get("medications", {}))
                    entity_extraction["risk_factors"].update(pharmacy_entities.get("risk_factors", {}))
                    
                elif "mcid" in data_type:
                    demographic_entities = await self._extract_demographic_entities(data)
                    entity_extraction["demographics"].update(demographic_entities)
            
            # Calculate overall confidence
            entity_extraction["analysis_confidence"] = self._calculate_extraction_confidence(entity_extraction)
            
            state["entity_extraction"] = entity_extraction
            state["step_status"]["enhanced_entity_extraction"] = "completed"
            
            logger.info(f"🎯 Entity extraction completed with confidence: {entity_extraction['analysis_confidence']}")
            
        except Exception as e:
            error_msg = f"Error in entity extraction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["enhanced_entity_extraction"] = "error"
            logger.error(error_msg)
        
        return state
    
    async def analyze_health_data_node(self, state: ContinuousHealthChatState) -> ContinuousHealthChatState:
        """Analyze the health data and create insights"""
        logger.info("📊 Node: Analyzing health data...")
        state["current_step"] = "analyze_health_data"
        state["step_status"]["analyze_health_data"] = "running"
        
        try:
            entity_extraction = state["entity_extraction"]
            deidentified_data = state["deidentified_data"]
            api_responses = state["api_responses"]
            
            # Create comprehensive health analysis
            health_analysis = {
                "summary": {},
                "key_findings": [],
                "risk_assessment": {},
                "recommendations": [],
                "data_quality": {}
            }
            
            # Analyze health conditions
            conditions = entity_extraction.get("health_conditions", {})
            medications = entity_extraction.get("medications", {})
            risk_factors = entity_extraction.get("risk_factors", {})
            
            # Generate summary
            health_analysis["summary"] = {
                "conditions_identified": len(conditions),
                "medications_found": len(medications),
                "risk_factors": len(risk_factors),
                "analysis_confidence": entity_extraction.get("analysis_confidence", 0.0)
            }
            
            # Key findings
            if conditions:
                health_analysis["key_findings"].append(f"Identified {len(conditions)} health conditions")
            if medications:
                health_analysis["key_findings"].append(f"Found {len(medications)} medications")
            
            # Risk assessment
            health_analysis["risk_assessment"] = await self._assess_health_risks(
                conditions, medications, risk_factors
            )
            
            # Generate recommendations
            health_analysis["recommendations"] = await self._generate_health_recommendations(
                conditions, medications, risk_factors
            )
            
            # Data quality assessment
            health_analysis["data_quality"] = self._assess_data_quality(api_responses, deidentified_data)
            
            state["health_analysis"] = health_analysis
            state["analysis_complete"] = True
            state["step_status"]["analyze_health_data"] = "completed"
            
            logger.info("📊 Health data analysis completed")
            
        except Exception as e:
            error_msg = f"Error analyzing health data: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["analyze_health_data"] = "error"
            logger.error(error_msg)
        
        return state
    
    async def generate_conversational_response_node(self, state: ContinuousHealthChatState) -> ContinuousHealthChatState:
        """Generate intelligent conversational response"""
        logger.info("💬 Node: Generating conversational response...")
        state["current_step"] = "generate_conversational_response"
        state["step_status"]["generate_conversational_response"] = "running"
        
        try:
            intent = state["intent"]
            user_message = state["user_message"]
            user_context = state["user_context"]
            
            if state.get("analysis_complete"):
                # Generate response based on completed analysis
                response = await self._generate_analysis_based_response(state)
            elif state.get("can_answer_question") and user_context.get("previous_analyses"):
                # Answer question using existing data
                response = await self._generate_context_based_response(state)
            else:
                # Generate general conversational response
                response = await self._generate_general_response(state)
            
            # Generate follow-up questions and suggestions
            follow_ups = await self._generate_intelligent_follow_ups(state)
            suggestions = await self._generate_action_suggestions(state)
            
            state["conversational_response"] = response
            state["follow_up_questions"] = follow_ups
            state["suggested_actions"] = suggestions
            
            state["step_status"]["generate_conversational_response"] = "completed"
            logger.info("💬 Conversational response generated")
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["generate_conversational_response"] = "error"
            logger.error(error_msg)
        
        return state
    
    async def request_clarification_node(self, state: ContinuousHealthChatState) -> ContinuousHealthChatState:
        """Request clarification for missing information using Snowflake Cortex"""
        logger.info("❓ Node: Requesting clarification with Snowflake Cortex...")
        
        try:
            missing_fields = state.get("missing_fields", [])
            intent = state["intent"]
            user_message = state["user_message"]
            extracted_data = state.get("extracted_patient_data", {})
            
            # Create intelligent clarification prompt with correct field names
            clarification_prompt = f"""
The user wants healthcare analysis but we're missing some required information.

User's Message: "{user_message}"
Intent: {intent}
Data We Have: {list(extracted_data.keys())}
Missing Required Fields: {missing_fields}

Field descriptions (use these exact names):
- first_name: Patient's first name
- second_name: Patient's last name/surname  
- ssn: Social Security Number
- date_of_birth: Date of birth (YYYY-MM-DD format)
- gender: Gender (F for female, M for male)
- zip_code: ZIP code

Generate a helpful, conversational request for the missing information that:
1. Acknowledges what they want to do
2. Explains what specific information is still needed
3. Gives a clear example of how to provide it
4. Is encouraging and helpful in tone
5. Keeps it under 150 words

Make it feel like a helpful healthcare assistant, not a form to fill out.
"""
            
            clarification_response = await self.call_cortex_llm(
                clarification_prompt,
                "You are a helpful healthcare assistant requesting missing patient information. Be conversational and encouraging."
            )
            
            state["conversational_response"] = clarification_response
            state["needs_clarification"] = True
            
            # Generate helpful follow-up questions
            state["follow_up_questions"] = [
                "What type of healthcare analysis are you looking for?",
                "Do you have all the patient information available?",
                "Would you like me to explain what each field is used for?"
            ]
            
            logger.info("❓ Intelligent clarification request generated")
            
        except Exception as e:
            error_msg = f"Error generating clarification: {str(e)}"
            state["errors"].append(error_msg)
            
            # Fallback clarification with correct field names
            missing_fields = state.get("missing_fields", [])
            if missing_fields:
                field_names = {
                    "first_name": "first name",
                    "second_name": "last name/surname",
                    "ssn": "Social Security Number",
                    "date_of_birth": "date of birth (YYYY-MM-DD)",
                    "gender": "gender (F for female, M for male)",
                    "zip_code": "zip code"
                }
                
                missing_descriptions = [field_names.get(field, field) for field in missing_fields]
                
                if len(missing_descriptions) == 1:
                    clarification = f"To analyze the patient's health data, I need their {missing_descriptions[0]}. Could you please provide this information?"
                else:
                    clarification = f"To analyze the patient's health data, I need: {', '.join(missing_descriptions[:-1])}, and {missing_descriptions[-1]}. Could you please provide these details?"
                
                clarification += "\n\n💡 **Example:** 'Analyze patient John Smith, last name Doe, age 45, male, SSN 123456789, zip 12345'"
            else:
                clarification = "I'd be happy to help with healthcare analysis! Could you please provide patient details for analysis?"
            
            state["conversational_response"] = clarification
            logger.error(error_msg)
        
        return state
    
    async def handle_error_node(self, state: ContinuousHealthChatState) -> ContinuousHealthChatState:
        """Handle workflow errors"""
        logger.error("🚨 Node: Handling errors...")
        
        errors = state.get("errors", [])
        if errors:
            error_response = "I encountered some issues while processing your request:\n\n"
            for i, error in enumerate(errors, 1):
                error_response += f"{i}. {error}\n"
            error_response += "\n💡 Please try rephrasing your request or provide more specific information."
        else:
            error_response = "I'm sorry, but I encountered an unexpected error. Please try again with your request."
        
        state["conversational_response"] = error_response
        state["follow_up_questions"] = [
            "Would you like to try again?",
            "Do you need help with the correct format?",
            "Should I explain what information I need?"
        ]
        
        return state
    
    # ===== CONDITIONAL ROUTING METHODS =====
    
    def route_after_intent_analysis(self, state: ContinuousHealthChatState) -> str:
        """Route after intent analysis"""
        if state.get("errors"):
            return "error"
        elif state["intent"] in ["patient_analysis", "medical_question", "medication_inquiry", "diagnosis_inquiry"]:
            return "need_patient_data"
        elif state["intent"] == "follow_up_question" and state.get("can_answer_question"):
            return "direct_question"
        elif state["intent"] == "unclear":
            return "clarification"
        else:
            return "need_patient_data"
    
    def route_after_data_check(self, state: ContinuousHealthChatState) -> str:
        """Route after checking data requirements"""
        if state.get("errors"):
            return "error"
        elif state["needs_clarification"]:
            return "incomplete"
        else:
            return "ready"
    
    def route_after_api_calls(self, state: ContinuousHealthChatState) -> str:
        """Route after API calls"""
        if state.get("errors"):
            return "error"
        
        api_call_status = state.get("api_call_status", {})
        successful_calls = sum(1 for status in api_call_status.values() if status == "success")
        total_calls = len(api_call_status)
        
        if successful_calls == 0:
            return "error"
        elif successful_calls < total_calls:
            return "partial"
        else:
            return "success"
    
    # ===== HELPER METHODS =====
    
    async def test_cortex_connection(self) -> Dict[str, Any]:
        """Test Snowflake Cortex API connection"""
        try:
            test_response = await self.call_cortex_llm(
                "Hello, please respond with 'Snowflake Cortex connection successful'",
                "You are a test assistant. Respond exactly as requested."
            )
            
            if "successful" in test_response.lower():
                return {
                    "success": True,
                    "response": test_response,
                    "model": self.config.cortex_model,
                    "api_url": self.config.cortex_api_url
                }
            else:
                return {
                    "success": False,
                    "error": f"Unexpected response: {test_response}",
                    "model": self.config.cortex_model
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": self.config.cortex_model
            }
    
    async def _analyze_intent_with_context(self, message: str, history: List[Dict[str, str]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced intent analysis with conversation context"""
        message_lower = message.lower()
        
        # Enhanced intent patterns
        intent_patterns = {
            "patient_analysis": [
                r"analyze|analysis|check|examine|review.*patient",
                r"patient.*analysis",
                r"health.*analysis|medical.*analysis",
                r"run.*analysis"
            ],
            "medical_question": [
                r"what.*(?:icd|diagnosis|medical|condition)",
                r"explain.*(?:medical|diagnosis|condition)",
                r"tell me about.*(?:medical|health)"
            ],
            "medication_inquiry": [
                r"what.*(?:medication|drug|prescription)",
                r"explain.*(?:medication|drug|prescription)",
                r"tell me about.*(?:medication|drug)"
            ],
            "follow_up_question": [
                r"what about|what.*found|more.*about",
                r"can you.*explain|tell me more",
                r"what does.*mean|what is"
            ]
        }
        
        # Check for intent patterns
        detected_intent = "unclear"
        confidence = 0.5
        
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    detected_intent = intent
                    confidence = 0.8
                    break
            if detected_intent != "unclear":
                break
        
        # Check if we can answer with existing data
        can_answer_with_existing = False
        if detected_intent == "follow_up_question" and context.get("previous_analyses"):
            can_answer_with_existing = True
            confidence = 0.9
        
        return {
            "intent": detected_intent,
            "confidence": confidence,
            "can_answer_with_existing_data": can_answer_with_existing
        }
    
    async def _extract_patient_info_advanced(self, message: str) -> Dict[str, Any]:
        """Advanced patient information extraction with correct field names"""
        extracted = {}
        
        # Enhanced patterns for patient data extraction with correct field names
        patterns = {
            "first_name": r"(?:patient|name|first name)\s+(?:is\s+)?([A-Z][a-z]+)",
            "second_name": r"(?:patient|name)\s+(?:is\s+)?[A-Z][a-z]+\s+([A-Z][a-z]+)",
            "full_name": r"(?:patient|name)\s+(?:is\s+)?([A-Z][a-z]+)\s+([A-Z][a-z]+)",
            "age": r"(?:age|years?\s+old|y/?o)\s*(?:is\s*)?(\d{1,3})",
            "date_of_birth": r"(?:dob|birth|born)\s*(?:is\s*)?(\d{4}-\d{2}-\d{2})",
            "ssn": r"(?:ssn|social)\s*(?:security)?\s*(?:number)?\s*(?:is\s*)?(\d{3}-?\d{2}-?\d{4})",
            "zip_code": r"(?:zip|postal)\s*(?:code)?\s*(?:is\s*)?(\d{5})",
            "gender_male": r"\b(?:male|man|boy|he|him|his)\b",
            "gender_female": r"\b(?:female|woman|girl|she|her)\b"
        }
        
        # Extract based on patterns
        for field, pattern in patterns.items():
            match = re.search(pattern, message, re.IGNORECASE)
            
            if match:
                if field == "full_name":
                    extracted["first_name"] = match.group(1)
                    extracted["second_name"] = match.group(2)  # Use second_name instead of last_name
                elif field == "first_name" and "second_name" not in extracted:
                    extracted["first_name"] = match.group(1)
                elif field == "second_name" and "first_name" not in extracted:
                    extracted["second_name"] = match.group(1)  # Use second_name instead of last_name
                elif field == "age":
                    age = int(match.group(1))
                    # Convert age to approximate birth date
                    current_year = datetime.now().year
                    birth_year = current_year - age
                    extracted["date_of_birth"] = f"{birth_year}-01-01"
                elif field == "ssn":
                    extracted["ssn"] = match.group(1).replace("-", "")
                elif field == "gender_male":
                    extracted["gender"] = "M"
                elif field == "gender_female":
                    extracted["gender"] = "F"
                elif field in ["date_of_birth", "zip_code"]:
                    extracted[field] = match.group(1)
        
        return extracted
    
    async def _call_specific_mcp_api(self, api_name: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call specific MCP server API endpoint with enhanced error handling and correct payload format"""
        base_url = self.config.mcp_server_url.rstrip('/')
        
        endpoints = {
            "medical": f"{base_url}/medical/submit",
            "pharmacy": f"{base_url}/pharmacy/submit",
            "mcid": f"{base_url}/mcid/search",
            "token": f"{base_url}/token",
            "all": f"{base_url}/all"
        }
        
        if api_name not in endpoints:
            return {"status_code": 404, "error": f"Unknown API: {api_name}"}
        
        endpoint_url = endpoints[api_name]
        logger.info(f"📡 Calling {api_name} API: {endpoint_url}")
        
        # Prepare the correct payload format for MCP server
        mcp_payload = {}
        if patient_data:
            # Map to the correct field names for MCP server
            if "first_name" in patient_data:
                mcp_payload["first_name"] = patient_data["first_name"]
            if "last_name" in patient_data:
                mcp_payload["last_name"] = patient_data["last_name"]
            if "ssn" in patient_data:
                mcp_payload["ssn"] = patient_data["ssn"]
            if "date_of_birth" in patient_data:
                mcp_payload["date_of_birth"] = patient_data["date_of_birth"]
            if "gender" in patient_data:
                mcp_payload["gender"] = patient_data["gender"]
            if "zip_code" in patient_data:
                mcp_payload["zip_code"] = patient_data["zip_code"]
        
        logger.info(f"📤 MCP Payload: {mcp_payload}")
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            try:
                # First, test if the server is reachable
                try:
                    health_response = await client.get(f"{base_url}/health")
                    logger.info(f"🏥 Health check: {health_response.status_code}")
                except:
                    logger.warning("⚠️ Health check failed, continuing with API call")
                
                # Make the actual API call
                if api_name == "token":
                    # Token endpoint doesn't need patient data
                    logger.info(f"🔑 Calling token endpoint: POST {endpoint_url}")
                    response = await client.post(endpoint_url)
                else:
                    logger.info(f"📤 Calling {api_name} endpoint: POST {endpoint_url}")
                    logger.info(f"📋 MCP payload keys: {list(mcp_payload.keys())}")
                    
                    response = await client.post(
                        endpoint_url,
                        json=mcp_payload,
                        headers={"Content-Type": "application/json"}
                    )
                
                logger.info(f"📥 {api_name} API response: {response.status_code}")
                
                if response.status_code == 405:
                    # Method not allowed - try to get more info
                    logger.error(f"❌ 405 Method Not Allowed for {endpoint_url}")
                    return {
                        "status_code": 405,
                        "error": f"Method POST not allowed for {endpoint_url}. Check if endpoint exists and accepts POST requests.",
                        "endpoint": endpoint_url,
                        "suggestion": "Verify the server is running and the router is properly configured"
                    }
                elif response.status_code == 404:
                    logger.error(f"❌ 404 Not Found for {endpoint_url}")
                    return {
                        "status_code": 404,
                        "error": f"Endpoint {endpoint_url} not found. Check if the router is properly mounted.",
                        "endpoint": endpoint_url
                    }
                elif response.status_code >= 400:
                    error_text = response.text
                    logger.error(f"❌ HTTP {response.status_code} error: {error_text}")
                    return {
                        "status_code": response.status_code,
                        "error": error_text,
                        "endpoint": endpoint_url,
                        "payload_sent": mcp_payload
                    }
                else:
                    # Success case
                    try:
                        response_body = response.json()
                    except:
                        response_body = response.text
                    
                    logger.info(f"✅ {api_name} API successful response received")
                    return {
                        "status_code": response.status_code,
                        "body": response_body,
                        "endpoint": endpoint_url,
                        "payload_sent": mcp_payload
                    }
                
            except httpx.ConnectError as e:
                error_msg = f"Cannot connect to MCP server at {base_url}. Is the server running?"
                logger.error(f"🔌 Connection error: {error_msg}")
                return {
                    "status_code": 503,
                    "error": error_msg,
                    "endpoint": endpoint_url,
                    "suggestion": "Check if the MCP server is running on the correct port"
                }
            except httpx.TimeoutException as e:
                error_msg = f"Request timeout after {self.config.timeout} seconds"
                logger.error(f"⏱️ Timeout error: {error_msg}")
                return {
                    "status_code": 408,
                    "error": error_msg,
                    "endpoint": endpoint_url
                }
            except Exception as e:
                error_msg = f"Unexpected error calling {api_name} API: {str(e)}"
                logger.error(f"💥 Unexpected error: {error_msg}")
                return {
                    "status_code": 500,
                    "error": error_msg,
                    "endpoint": endpoint_url
                }
    
    async def _deidentify_medical_data(self, medical_data: Any, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deidentify medical data"""
        try:
            # Calculate age for deidentification
            age = self._calculate_age_from_dob(patient_data.get("date_of_birth"))
            
            deidentified = {
                "patient_info": {
                    "age": age,
                    "gender": patient_data.get("gender", "unknown"),
                    "zip_code": "12345"  # Generic zip
                },
                "medical_data": self._remove_pii_recursive(medical_data)
            }
            
            return deidentified
            
        except Exception as e:
            return {"error": f"Medical deidentification failed: {str(e)}"}
    
    async def _deidentify_pharmacy_data(self, pharmacy_data: Any) -> Dict[str, Any]:
        """Deidentify pharmacy data"""
        try:
            return {
                "pharmacy_data": self._remove_pii_recursive(pharmacy_data)
            }
        except Exception as e:
            return {"error": f"Pharmacy deidentification failed: {str(e)}"}
    
    async def _deidentify_mcid_data(self, mcid_data: Any) -> Dict[str, Any]:
        """Deidentify MCID data"""
        try:
            return {
                "member_data": self._remove_pii_recursive(mcid_data)
            }
        except Exception as e:
            return {"error": f"MCID deidentification failed: {str(e)}"}
    
    async def _deidentify_generic_data(self, data: Any) -> Dict[str, Any]:
        """Generic data deidentification"""
        try:
            return {
                "deidentified_data": self._remove_pii_recursive(data)
            }
        except Exception as e:
            return {"error": f"Generic deidentification failed: {str(e)}"}
    
    def _remove_pii_recursive(self, data: Any) -> Any:
        """Recursively remove PII from data structures"""
        if isinstance(data, dict):
            return {k: self._remove_pii_recursive(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._remove_pii_recursive(item) for item in data]
        elif isinstance(data, str):
            # Apply PII removal patterns
            data = re.sub(r'\b\d{3}-?\d{2}-?\d{4}\b', '[SSN_MASKED]', data)  # SSN
            data = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[NAME_MASKED]', data)  # Names
            data = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_MASKED]', data)  # Phone
            return data
        else:
            return data
    
    async def _extract_medical_entities(self, medical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract medical entities from deidentified medical data"""
        entities = {"conditions": {}, "codes": {}}
        
        try:
            # Look for ICD codes and conditions
            data_str = json.dumps(medical_data).lower()
            
            # Common ICD-10 patterns
            icd_patterns = {
                "diabetes": ["e10", "e11", "e12", "e13", "e14"],
                "hypertension": ["i10", "i11", "i12", "i13", "i15"],
                "smoking": ["z72.0", "f17"],
                "alcohol": ["f10", "z72.1"]
            }
            
            for condition, codes in icd_patterns.items():
                for code in codes:
                    if code in data_str:
                        entities["conditions"][condition] = "detected"
                        entities["codes"][code] = condition
            
        except Exception as e:
            entities["error"] = str(e)
        
        return entities
    
    async def _extract_pharmacy_entities(self, pharmacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract pharmacy entities from deidentified pharmacy data"""
        entities = {"medications": {}, "risk_factors": {}}
        
        try:
            data_str = json.dumps(pharmacy_data).lower()
            
            # Medication patterns
            medication_patterns = {
                "insulin": ["insulin", "lantus", "humalog", "novolog"],
                "metformin": ["metformin", "glucophage"],
                "lisinopril": ["lisinopril", "prinivil", "zestril"],
                "atorvastatin": ["atorvastatin", "lipitor"]
            }
            
            for med_class, keywords in medication_patterns.items():
                for keyword in keywords:
                    if keyword in data_str:
                        entities["medications"][med_class] = "found"
                        
                        # Infer risk factors
                        if med_class in ["insulin", "metformin"]:
                            entities["risk_factors"]["diabetes"] = "indicated"
                        elif med_class == "lisinopril":
                            entities["risk_factors"]["hypertension"] = "indicated"
            
        except Exception as e:
            entities["error"] = str(e)
        
        return entities
    
    async def _extract_demographic_entities(self, mcid_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract demographic entities from MCID data"""
        entities = {}
        
        try:
            # Extract demographic information that's not PII
            if isinstance(mcid_data, dict):
                entities["member_found"] = bool(mcid_data.get("member_data"))
                entities["data_quality"] = "high" if mcid_data.get("member_data") else "low"
            
        except Exception as e:
            entities["error"] = str(e)
        
        return entities
    
    def _calculate_extraction_confidence(self, entity_extraction: Dict[str, Any]) -> float:
        """Calculate confidence score for entity extraction"""
        total_entities = 0
        found_entities = 0
        
        for category, entities in entity_extraction.items():
            if isinstance(entities, dict) and category != "analysis_confidence":
                total_entities += len(entities)
                found_entities += sum(1 for v in entities.values() if v not in ["unknown", "error"])
        
        return found_entities / max(total_entities, 1)
    
    async def _assess_health_risks(self, conditions: Dict, medications: Dict, risk_factors: Dict) -> Dict[str, Any]:
        """Assess health risks based on extracted entities"""
        risk_assessment = {
            "diabetes_risk": "low",
            "cardiovascular_risk": "low", 
            "overall_risk": "low"
        }
        
        # Assess diabetes risk
        if conditions.get("diabetes") or medications.get("insulin") or medications.get("metformin"):
            risk_assessment["diabetes_risk"] = "high"
        
        # Assess cardiovascular risk
        if conditions.get("hypertension") or medications.get("lisinopril") or medications.get("atorvastatin"):
            risk_assessment["cardiovascular_risk"] = "moderate"
        
        # Overall risk
        high_risks = sum(1 for risk in risk_assessment.values() if risk == "high")
        if high_risks > 0:
            risk_assessment["overall_risk"] = "high"
        elif any(risk == "moderate" for risk in risk_assessment.values()):
            risk_assessment["overall_risk"] = "moderate"
        
        return risk_assessment
    
    async def _generate_health_recommendations(self, conditions: Dict, medications: Dict, risk_factors: Dict) -> List[str]:
        """Generate health recommendations"""
        recommendations = []
        
        if conditions.get("diabetes") or medications.get("insulin"):
            recommendations.append("Regular blood glucose monitoring recommended")
            recommendations.append("Consider diabetes education and lifestyle counseling")
        
        if conditions.get("hypertension") or medications.get("lisinopril"):
            recommendations.append("Regular blood pressure monitoring advised")
            recommendations.append("Consider dietary modifications and exercise")
        
        if not recommendations:
            recommendations.append("Continue regular healthcare check-ups")
            recommendations.append("Maintain healthy lifestyle practices")
        
        return recommendations
    
    def _assess_data_quality(self, api_responses: Dict, deidentified_data: Dict) -> Dict[str, Any]:
        """Assess quality of the data retrieved"""
        quality_assessment = {
            "completeness": 0.0,
            "accuracy": "unknown",
            "freshness": "recent"
        }
        
        successful_apis = sum(1 for response in api_responses.values() 
                            if response.get("status_code") == 200)
        total_apis = len(api_responses)
        
        quality_assessment["completeness"] = successful_apis / max(total_apis, 1)
        
        if quality_assessment["completeness"] > 0.8:
            quality_assessment["accuracy"] = "high"
        elif quality_assessment["completeness"] > 0.5:
            quality_assessment["accuracy"] = "moderate"
        else:
            quality_assessment["accuracy"] = "low"
        
        return quality_assessment
    
    async def _generate_analysis_based_response(self, state: ContinuousHealthChatState) -> str:
        """Generate response based on completed analysis using Snowflake Cortex"""
        health_analysis = state["health_analysis"]
        entity_extraction = state["entity_extraction"]
        patient_data = state["extracted_patient_data"]
        deidentified_data = state["deidentified_data"]
        user_message = state["user_message"]
        
        # Create comprehensive prompt for analysis-based response
        analysis_prompt = f"""
Based on the completed healthcare analysis, provide a conversational response to the user's question.

User's Question: "{user_message}"

Patient Information:
- Name: {patient_data.get('first_name', 'Unknown')} {patient_data.get('last_name', '')}
- Age: {self._calculate_age_from_dob(patient_data.get('date_of_birth', ''))} years
- Gender: {patient_data.get('gender', 'Unknown')}

Health Analysis Summary:
{json.dumps(health_analysis.get('summary', {}), indent=2)}

Entity Extraction Results:
- Health Conditions: {entity_extraction.get('health_conditions', {})}
- Medications Found: {entity_extraction.get('medications', {})}
- Risk Factors: {entity_extraction.get('risk_factors', {})}

Key Findings:
{chr(10).join('• ' + finding for finding in health_analysis.get('key_findings', []))}

Risk Assessment:
{json.dumps(health_analysis.get('risk_assessment', {}), indent=2)}

Recommendations:
{chr(10).join('• ' + rec for rec in health_analysis.get('recommendations', []))}

Please provide a conversational, empathetic response that:
1. Addresses the user's specific question
2. Explains the key findings in simple terms
3. Highlights any important health insights
4. Provides actionable recommendations
5. Uses a warm, professional healthcare assistant tone
6. Asks if they want more details on specific areas

Keep the response comprehensive but accessible, around 200-300 words.
"""
        
        try:
            response = await self.call_cortex_llm(
                analysis_prompt,
                "You are a healthcare AI assistant providing analysis results. Be conversational, empathetic, and provide clear explanations of medical findings."
            )
            return response
        except Exception as e:
            logger.error(f"Error generating analysis response: {e}")
            return f"I've completed the healthcare analysis but encountered an error generating the response: {str(e)}"
    
    async def _generate_context_based_response(self, state: ContinuousHealthChatState) -> str:
        """Generate response using existing context/analysis with Snowflake Cortex"""
        user_message = state["user_message"]
        user_context = state["user_context"]
        
        # Get recent analysis data
        previous_analyses = user_context.get("previous_analyses", [])
        patient_data = user_context.get("patient_data", {})
        topics_discussed = user_context.get("topics_discussed", [])
        
        context_prompt = f"""
Answer this follow-up question based on previous healthcare analysis:

Current Question: "{user_message}"

Patient Information:
{json.dumps(patient_data, indent=2)}

Previous Analysis Context:
{json.dumps(previous_analyses[-1] if previous_analyses else {}, indent=2)}

Recent Topics Discussed:
{json.dumps(topics_discussed[-3:], indent=2)}

Provide a conversational response that:
1. Directly answers their question using the available data
2. References specific findings from the previous analysis
3. Explains medical terms in simple language
4. Offers to provide more details if needed
5. Maintains continuity with the ongoing conversation

If the question asks about something not covered in the previous analysis, suggest running a new analysis.
"""
        
        try:
            response = await self.call_cortex_llm(
                context_prompt,
                "You are a healthcare AI assistant answering follow-up questions based on previous analysis. Be helpful and reference specific previous findings."
            )
            return response
        except Exception as e:
            logger.error(f"Error generating context response: {e}")
            return f"I have previous analysis data available, but encountered an error: {str(e)}"
    
    async def _generate_general_response(self, state: ContinuousHealthChatState) -> str:
        """Generate general conversational response with Snowflake Cortex"""
        intent = state["intent"]
        user_message = state["user_message"]
        conversation_history = state["conversation_history"]
        
        general_prompt = f"""
Respond to this healthcare-related message:

User Message: "{user_message}"
Detected Intent: {intent}

Recent Conversation Context:
{json.dumps(conversation_history[-3:] if conversation_history else [], indent=2)}

Provide a helpful response that:
1. Acknowledges their request
2. Explains what information you need to help them
3. Gives examples of how they can provide patient data
4. Offers specific help based on their apparent intent
5. Uses a friendly, professional healthcare assistant tone

If they're asking about medical information without patient data, explain that you can provide general information but need specific patient data for detailed analysis.

Keep the response encouraging and helpful, around 100-150 words.
"""
        
        try:
            response = await self.call_cortex_llm(
                general_prompt,
                "You are a helpful healthcare AI assistant guiding users on how to get healthcare analysis. Be encouraging and specific."
            )
            return response
        except Exception as e:
            logger.error(f"Error generating general response: {e}")
            return "I'm here to help with healthcare analysis! Could you provide more details about what you'd like me to analyze?"
    
    async def _generate_intelligent_follow_ups(self, state: ContinuousHealthChatState) -> List[str]:
        """Generate intelligent follow-up questions using Snowflake Cortex"""
        try:
            user_message = state["user_message"]
            intent = state["intent"]
            analysis_complete = state.get("analysis_complete", False)
            entity_extraction = state.get("entity_extraction", {})
            health_analysis = state.get("health_analysis", {})
            
            followup_prompt = f"""
Generate 3 relevant follow-up questions for this healthcare conversation:

User's Last Message: "{user_message}"
Intent: {intent}
Analysis Complete: {analysis_complete}

Current Context:
- Conditions Found: {entity_extraction.get('health_conditions', {})}
- Medications Found: {entity_extraction.get('medications', {})}
- Risk Factors: {entity_extraction.get('risk_factors', {})}

Generate follow-up questions that:
1. Are relevant to what was just discussed
2. Help the user explore the analysis results further
3. Provide actionable next steps
4. Are phrased as natural questions a user might ask

Return as JSON array of 3 questions:
["Question 1?", "Question 2?", "Question 3?"]

Make the questions specific and helpful based on the current context.
"""
            
            cortex_response = await self.call_cortex_llm(
                followup_prompt,
                "Generate helpful follow-up questions for healthcare conversations. Return only a JSON array of questions."
            )
            
            try:
                follow_ups = json.loads(cortex_response)
                if isinstance(follow_ups, list):
                    return follow_ups[:3]  # Limit to 3
            except json.JSONDecodeError:
                pass
            
        except Exception as e:
            logger.error(f"Error generating follow-ups: {e}")
        
        # Fallback follow-ups
        if state.get("analysis_complete"):
            return [
                "Would you like me to explain any specific findings?",
                "Do you want recommendations based on these results?",
                "Should I analyze another patient?"
            ]
        else:
            return [
                "What type of healthcare analysis are you looking for?",
                "Do you have patient information to analyze?",
                "Would you like me to explain what data I need?"
            ]
    
    async def _generate_action_suggestions(self, state: ContinuousHealthChatState) -> List[str]:
        """Generate action suggestions using Snowflake Cortex"""
        try:
            analysis_complete = state.get("analysis_complete", False)
            needs_clarification = state.get("needs_clarification", False)
            intent = state["intent"]
            
            suggestions_prompt = f"""
Generate 3 actionable suggestions for this healthcare conversation:

Analysis Complete: {analysis_complete}
Needs Clarification: {needs_clarification}
Intent: {intent}

Generate practical actions the user can take, such as:
- Export or download results
- Get more detailed information
- Start new analysis
- Provide missing information
- Get explanations

Return as JSON array:
["Action 1", "Action 2", "Action 3"]
"""
            
            cortex_response = await self.call_cortex_llm(
                suggestions_prompt,
                "Generate practical action suggestions for healthcare analysis. Return only a JSON array."
            )
            
            try:
                suggestions = json.loads(cortex_response)
                if isinstance(suggestions, list):
                    return suggestions[:3]
            except json.JSONDecodeError:
                pass
                
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
        
        # Fallback suggestions
        if state.get("analysis_complete"):
            return ["Export analysis results", "Generate detailed report", "Analyze another patient"]
        elif state.get("needs_clarification"):
            return ["Provide patient information", "View data requirements", "See example format"]
        else:
            return ["Start new analysis", "Ask about medical codes", "Get help with format"]
    
    async def _fallback_intent_analysis(self, message: str) -> Dict[str, Any]:
        """Fallback intent analysis using simple patterns"""
        message_lower = message.lower()
        
        # Enhanced intent patterns
        intent_patterns = {
            "patient_analysis": [
                r"analyze|analysis|check|examine|review.*patient",
                r"patient.*analysis",
                r"health.*analysis|medical.*analysis",
                r"run.*analysis"
            ],
            "medical_question": [
                r"what.*(?:icd|diagnosis|medical|condition)",
                r"explain.*(?:medical|diagnosis|condition)",
                r"tell me about.*(?:medical|health)"
            ],
            "medication_inquiry": [
                r"what.*(?:medication|drug|prescription)",
                r"explain.*(?:medication|drug|prescription)",
                r"tell me about.*(?:medication|drug)"
            ],
            "follow_up_question": [
                r"what about|what.*found|more.*about",
                r"can you.*explain|tell me more",
                r"what does.*mean|what is"
            ]
        }
        
        # Check for intent patterns
        detected_intent = "clarification"
        confidence = 0.5
        
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    detected_intent = intent
                    confidence = 0.7
                    break
            if detected_intent != "clarification":
                break
        
        return {
            "intent": detected_intent,
            "confidence": confidence,
            "extracted_patient_data": {},
            "can_answer_with_existing": False,
            "requires_new_analysis": detected_intent == "patient_analysis",
            "reasoning": "Pattern-based fallback analysis"
        }
    
    def _calculate_age_from_dob(self, dob_str: str) -> int:
        """Calculate age from date of birth string"""
        try:
            if dob_str:
                dob = datetime.strptime(dob_str, '%Y-%m-%d').date()
                today = date.today()
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                return age
        except:
            pass
        return 0
    
    # ===== PUBLIC INTERFACE =====
    
    async def chat(self, message: str, session_id: str = None, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Main chat interface - process user message and return conversational response
        """
        session_id = session_id or f"session_{datetime.now().timestamp()}"
        conversation_history = conversation_history or []
        
        # Initialize state
        initial_state = ContinuousHealthChatState(
            user_message=message,
            conversation_history=conversation_history,
            session_id=session_id,
            user_context={},
            intent="",
            confidence=0.0,
            required_apis=[],
            extracted_patient_data={},
            missing_fields=[],
            api_responses={},
            api_call_status={},
            deidentified_data={},
            entity_extraction={},
            health_analysis={},
            conversational_response="",
            follow_up_questions=[],
            suggested_actions=[],
            current_step="",
            errors=[],
            needs_clarification=False,
            analysis_complete=False,
            can_answer_question=False,
            step_status={}
        )
        
        try:
            # Configure for session persistence
            config = {"configurable": {"thread_id": session_id}}
            
            # Run enhanced workflow
            logger.info(f"🤖 Processing chat message: {message[:50]}...")
            final_state = await self.workflow.ainvoke(initial_state, config=config)
            
            # Return comprehensive response
            return {
                "success": not bool(final_state.get("errors")),
                "response": final_state.get("conversational_response", "I'm sorry, I couldn't process your message."),
                "follow_up_questions": final_state.get("follow_up_questions", []),
                "suggested_actions": final_state.get("suggested_actions", []),
                "needs_clarification": final_state.get("needs_clarification", False),
                "analysis_complete": final_state.get("analysis_complete", False),
                "patient_data": final_state.get("extracted_patient_data", {}),
                "health_analysis": final_state.get("health_analysis", {}),
                "entity_extraction": final_state.get("entity_extraction", {}),
                "api_responses": final_state.get("api_responses", {}),
                "deidentified_data": final_state.get("deidentified_data", {}),
                "errors": final_state.get("errors", []),
                "step_status": final_state.get("step_status", {}),
                "session_id": session_id,
                "intent": final_state.get("intent", ""),
                "confidence": final_state.get("confidence", 0.0)
            }
            
        except Exception as e:
            logger.error(f"❌ Error in chat workflow: {e}")
            return {
                "success": False,
                "response": f"I encountered an error while processing your message: {str(e)}",
                "follow_up_questions": ["Would you like to try rephrasing your question?"],
                "suggested_actions": ["Try again", "Get help"],
                "errors": [str(e)],
                "session_id": session_id
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check with detailed diagnostics"""
        try:
            base_url = self.config.mcp_server_url.rstrip('/')
            
            # Test different endpoints to diagnose issues
            diagnostics = {
                "chatbot_status": "healthy",
                "mcp_server_url": base_url,
                "workflow_ready": self.workflow is not None,
                "active_sessions": len(self.session_contexts),
                "timestamp": datetime.now().isoformat(),
                "endpoint_tests": {},
                "recommendations": []
            }
            
            async with httpx.AsyncClient(timeout=5) as client:
                # Test 1: Basic connectivity
                try:
                    response = await client.get(f"{base_url}/")
                    diagnostics["endpoint_tests"]["root"] = {
                        "status": response.status_code,
                        "reachable": True
                    }
                except Exception as e:
                    diagnostics["endpoint_tests"]["root"] = {
                        "status": "unreachable",
                        "error": str(e),
                        "reachable": False
                    }
                    diagnostics["recommendations"].append(f"Check if server is running on {base_url}")
                
                # Test 2: Health endpoint
                try:
                    response = await client.get(f"{base_url}/health")
                    diagnostics["endpoint_tests"]["health"] = {
                        "status": response.status_code,
                        "reachable": True
                    }
                except Exception as e:
                    diagnostics["endpoint_tests"]["health"] = {
                        "status": "unreachable", 
                        "error": str(e),
                        "reachable": False
                    }
                
                # Test 3: Check if router endpoints exist
                test_endpoints = ["/medical/submit", "/pharmacy/submit", "/mcid/search", "/token", "/all"]
                
                for endpoint in test_endpoints:
                    try:
                        # Try OPTIONS method first to see if endpoint exists
                        response = await client.options(f"{base_url}{endpoint}")
                        diagnostics["endpoint_tests"][endpoint] = {
                            "options_status": response.status_code,
                            "exists": response.status_code != 404
                        }
                        
                        # If OPTIONS works, the endpoint exists but might not accept POST
                        if response.status_code == 405:
                            diagnostics["endpoint_tests"][endpoint]["post_allowed"] = False
                            diagnostics["recommendations"].append(f"Endpoint {endpoint} exists but doesn't accept POST method")
                        
                    except Exception as e:
                        diagnostics["endpoint_tests"][endpoint] = {
                            "error": str(e),
                            "exists": False
                        }
                
                # Test 4: Try a simple token call
                try:
                    response = await client.post(f"{base_url}/token")
                    diagnostics["endpoint_tests"]["token_post"] = {
                        "status": response.status_code,
                        "works": response.status_code == 200
                    }
                    
                    if response.status_code == 405:
                        diagnostics["recommendations"].append("Token endpoint exists but doesn't accept POST - check router configuration")
                    elif response.status_code == 404:
                        diagnostics["recommendations"].append("Token endpoint not found - check if router is mounted correctly")
                        
                except Exception as e:
                    diagnostics["endpoint_tests"]["token_post"] = {
                        "error": str(e)
                    }
            
            # Overall status assessment
            if not diagnostics["endpoint_tests"].get("root", {}).get("reachable"):
                diagnostics["chatbot_status"] = "server_unreachable"
            elif any(test.get("status") == 405 for test in diagnostics["endpoint_tests"].values()):
                diagnostics["chatbot_status"] = "method_not_allowed_errors"
                diagnostics["recommendations"].append("405 errors detected - check FastAPI router configuration")
            elif not any(test.get("works") or test.get("status") == 200 for test in diagnostics["endpoint_tests"].values()):
                diagnostics["chatbot_status"] = "endpoints_not_working"
            
            return diagnostics
            
        except Exception as e:
            return {
                "chatbot_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "recommendations": ["Check chatbot configuration and server connectivity"]
            }

# Example usage and testing
async def main():
    """Test the Enhanced LangGraph MCP Chatbot"""
    
    print("🤖 Enhanced LangGraph MCP Healthcare Chatbot Test")
    print("=" * 60)
    
    # Initialize chatbot
    config = MCPChatbotConfig(mcp_server_url="http://localhost:8000")
    chatbot = EnhancedLangGraphMCPChatbot(config)
    
    # Health check
    health = await chatbot.health_check()
    print(f"🏥 Health Check: {health}")
    print()
    
    # Test conversation flow
    test_messages = [
        "Hi, I need to analyze a patient",
        "Patient John Smith, age 45, male, SSN 123456789, zip 12345",
        "What medications did you find?",
        "Explain the diabetes findings",
        "Any recommendations for this patient?"
    ]
    
    session_id = "test_session"
    conversation_history = []
    
    for i, message in enumerate(test_messages, 1):
        print(f"👤 User: {message}")
        print("-" * 40)
        
        result = await chatbot.chat(
            message=message,
            session_id=session_id,
            conversation_history=conversation_history
        )
        
        print(f"🤖 Assistant: {result['response']}\n")
        
        if result['follow_up_questions']:
            print("❓ Follow-up Questions:")
            for q in result['follow_up_questions']:
                print(f"   • {q}")
            print()
        
        if result['suggested_actions']:
            print("💡 Suggested Actions:")
            for action in result['suggested_actions']:
                print(f"   • {action}")
            print()
        
        if result.get('analysis_complete'):
            print("✅ Analysis completed!")
            health_analysis = result.get('health_analysis', {})
            if health_analysis:
                print(f"📊 Summary: {health_analysis.get('summary', {})}")
            print()
        
        # Update conversation history
        conversation_history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": result['response']}
        ])
        
        print("=" * 60 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
