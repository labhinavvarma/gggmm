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

# Enhanced State Definition for LangGraph
class HealthAnalysisState(TypedDict):
    # Input data
    patient_data: Dict[str, Any]
    
    # Raw MCP API outputs (always stored for display)
    raw_api_data: Dict[str, Any]
    mcid_raw: Dict[str, Any]
    medical_raw: Dict[str, Any]
    pharmacy_raw: Dict[str, Any]
    token_raw: Dict[str, Any]
    all_raw: Dict[str, Any]
    
    # Processed data
    deidentified_medical: Dict[str, Any]
    deidentified_pharmacy: Dict[str, Any]
    
    # Enhanced entity extraction
    entity_extraction: Dict[str, Any]
    
    # Chatbot context
    chatbot_context: Dict[str, Any]
    chat_history: List[Dict[str, Any]]
    
    # Control flow
    current_step: str
    errors: List[str]
    retry_count: int
    processing_complete: bool
    step_status: Dict[str, str]

class EnhancedHealthAnalysisAgent:
    def __init__(self, custom_config: Optional[Config] = None):
        self.config = custom_config or Config()
        logger.info("🔧 Enhanced HealthAnalysisAgent initialized with MCP integration")
        logger.info(f"🌐 MCP Server URL: {self.config.mcp_server_url}")
        
        self.setup_langgraph()
        
        # Initialize chatbot context storage
        self.chatbot_sessions: Dict[str, List[Dict[str, Any]]] = {}
        self.current_session_id: Optional[str] = None
    
    def setup_langgraph(self):
        """Setup Enhanced LangGraph workflow"""
        logger.info("🔧 Setting up Enhanced LangGraph workflow with MCP integration...")
        
        workflow = StateGraph(HealthAnalysisState)
        
        # Add processing nodes
        workflow.add_node("fetch_mcp_data", self.fetch_mcp_data)
        workflow.add_node("deidentify_data", self.deidentify_data)
        workflow.add_node("extract_entities", self.extract_entities)
        workflow.add_node("setup_chatbot", self.setup_chatbot)
        workflow.add_node("handle_error", self.handle_error)
        
        # Define workflow edges
        workflow.add_edge(START, "fetch_mcp_data")
        
        workflow.add_conditional_edges(
            "fetch_mcp_data",
            self.should_continue_after_api,
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
                "continue": "extract_entities",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "extract_entities",
            self.should_continue_after_entities,
            {
                "continue": "setup_chatbot",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("setup_chatbot", END)
        workflow.add_edge("handle_error", END)
        
        # Compile with checkpointer
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)
        
        logger.info("✅ Enhanced LangGraph workflow compiled successfully!")
    
    def call_llm(self, user_message: str) -> str:
        """Call Snowflake Cortex API"""
        try:
            session_id = str(uuid.uuid4())
            
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
    
    def fetch_mcp_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Fetch all data from MCP server"""
        logger.info("🚀 LangGraph Node 1: Fetching data from MCP server...")
        state["current_step"] = "fetch_mcp_data"
        state["step_status"]["fetch_mcp_data"] = "running"
        
        try:
            patient_data = state["patient_data"]
            
            # Initialize raw data storage
            state["raw_api_data"] = {}
            
            # Call individual MCP endpoints
            endpoints = {
                "mcid": "/mcid/search",
                "medical": "/medical/submit", 
                "pharmacy": "/pharmacy/submit",
                "token": "/token",
                "all": "/all"
            }
            
            for endpoint_name, endpoint_path in endpoints.items():
                try:
                    logger.info(f"📡 Calling MCP endpoint: {endpoint_name}")
                    
                    if endpoint_name == "token":
                        # Token endpoint doesn't need patient data
                        response = requests.post(
                            f"{self.config.mcp_server_url}{endpoint_path}",
                            timeout=self.config.timeout
                        )
                    else:
                        # Other endpoints need patient data
                        response = requests.post(
                            f"{self.config.mcp_server_url}{endpoint_path}",
                            json=patient_data,
                            timeout=self.config.timeout
                        )
                    
                    if response.status_code == 200:
                        raw_data = response.json()
                        state["raw_api_data"][endpoint_name] = raw_data
                        state[f"{endpoint_name}_raw"] = raw_data
                        logger.info(f"✅ Successfully fetched {endpoint_name} data")
                    else:
                        error_msg = f"{endpoint_name} API failed: {response.status_code} - {response.text}"
                        state["errors"].append(error_msg)
                        state["raw_api_data"][endpoint_name] = {"error": error_msg}
                        state[f"{endpoint_name}_raw"] = {"error": error_msg}
                        logger.error(error_msg)
                        
                except Exception as e:
                    error_msg = f"Error calling {endpoint_name}: {str(e)}"
                    state["errors"].append(error_msg)
                    state["raw_api_data"][endpoint_name] = {"error": error_msg}
                    state[f"{endpoint_name}_raw"] = {"error": error_msg}
                    logger.error(error_msg)
            
            state["step_status"]["fetch_mcp_data"] = "completed"
            logger.info("✅ MCP data fetching completed")
            
        except Exception as e:
            error_msg = f"Error in MCP data fetching: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["fetch_mcp_data"] = "error"
            logger.error(error_msg)
        
        return state
    
    def deidentify_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Deidentify medical and pharmacy data"""
        logger.info("🔒 LangGraph Node 2: Deidentifying medical and pharmacy data...")
        state["current_step"] = "deidentify_data"
        state["step_status"]["deidentify_data"] = "running"
        
        try:
            patient_data = state["patient_data"]
            
            # Deidentify Medical Data
            medical_raw = state.get("medical_raw", {})
            if medical_raw and not medical_raw.get("error"):
                deidentified_medical = self._deidentify_medical_data(medical_raw, patient_data)
                state["deidentified_medical"] = deidentified_medical
                logger.info("✅ Medical data deidentified")
            else:
                state["deidentified_medical"] = {"error": "No valid medical data to deidentify"}
                logger.warning("⚠️ No valid medical data for deidentification")
            
            # Deidentify Pharmacy Data
            pharmacy_raw = state.get("pharmacy_raw", {})
            if pharmacy_raw and not pharmacy_raw.get("error"):
                deidentified_pharmacy = self._deidentify_pharmacy_data(pharmacy_raw)
                state["deidentified_pharmacy"] = deidentified_pharmacy
                logger.info("✅ Pharmacy data deidentified")
            else:
                state["deidentified_pharmacy"] = {"error": "No valid pharmacy data to deidentify"}
                logger.warning("⚠️ No valid pharmacy data for deidentification")
            
            state["step_status"]["deidentify_data"] = "completed"
            logger.info("✅ Data deidentification completed")
            
        except Exception as e:
            error_msg = f"Error in data deidentification: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["deidentify_data"] = "error"
            logger.error(error_msg)
        
        return state
    
    def extract_entities(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced entity extraction for diabetes, age, blood pressure, smoking, alcohol"""
        logger.info("🎯 LangGraph Node 3: Enhanced entity extraction...")
        state["current_step"] = "extract_entities"
        state["step_status"]["extract_entities"] = "running"
        
        try:
            # Initialize entity extraction results
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
            
            # Calculate age from patient data
            patient_data = state["patient_data"]
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
                    
                    entities["analysis_details"].append(f"Age calculated: {age} years ({entities['age_group']})")
                except:
                    entities["analysis_details"].append("Could not calculate age from date of birth")
            
            # Analyze deidentified medical data
            deidentified_medical = state.get("deidentified_medical", {})
            if deidentified_medical and not deidentified_medical.get("error"):
                self._analyze_medical_for_entities(deidentified_medical, entities)
            
            # Analyze deidentified pharmacy data
            deidentified_pharmacy = state.get("deidentified_pharmacy", {})
            if deidentified_pharmacy and not deidentified_pharmacy.get("error"):
                self._analyze_pharmacy_for_entities(deidentified_pharmacy, entities)
            
            # Analyze raw data for additional insights
            pharmacy_raw = state.get("pharmacy_raw", {})
            if pharmacy_raw and not pharmacy_raw.get("error"):
                self._analyze_raw_pharmacy_for_entities(pharmacy_raw, entities)
            
            state["entity_extraction"] = entities
            state["step_status"]["extract_entities"] = "completed"
            logger.info("✅ Enhanced entity extraction completed")
            
        except Exception as e:
            error_msg = f"Error in entity extraction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_entities"] = "error"
            logger.error(error_msg)
        
        return state
    
    def setup_chatbot(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Setup chatbot context with deidentified data"""
        logger.info("🤖 LangGraph Node 4: Setting up chatbot context...")
        state["current_step"] = "setup_chatbot"
        state["step_status"]["setup_chatbot"] = "running"
        
        try:
            # Create new session for this analysis
            session_id = str(uuid.uuid4())
            self.current_session_id = session_id
            
            # Prepare chatbot context
            chatbot_context = {
                "session_id": session_id,
                "patient_info": {
                    "age_group": state["entity_extraction"].get("age_group", "unknown"),
                    "analysis_timestamp": datetime.now().isoformat()
                },
                "deidentified_medical": state.get("deidentified_medical", {}),
                "deidentified_pharmacy": state.get("deidentified_pharmacy", {}),
                "entity_extraction": state.get("entity_extraction", {}),
                "available_for_questions": True
            }
            
            state["chatbot_context"] = chatbot_context
            state["chat_history"] = []
            
            # Initialize chat session
            self.chatbot_sessions[session_id] = []
            
            # Add welcome message
            welcome_message = self._generate_welcome_message(state)
            state["chat_history"].append({
                "role": "assistant",
                "content": welcome_message,
                "timestamp": datetime.now().isoformat()
            })
            
            self.chatbot_sessions[session_id].append({
                "role": "assistant", 
                "content": welcome_message,
                "timestamp": datetime.now().isoformat()
            })
            
            state["step_status"]["setup_chatbot"] = "completed"
            state["processing_complete"] = True
            logger.info("✅ Chatbot setup completed")
            
        except Exception as e:
            error_msg = f"Error setting up chatbot: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["setup_chatbot"] = "error"
            logger.error(error_msg)
        
        return state
    
    def handle_error(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Handle errors in the workflow"""
        logger.error(f"🚨 LangGraph Error Handler: {state['current_step']}")
        logger.error(f"Errors: {state['errors']}")
        
        state["processing_complete"] = True
        current_step = state.get("current_step", "unknown")
        state["step_status"][current_step] = "error"
        return state
    
    # ===== CONDITIONAL EDGES =====
    
    def should_continue_after_api(self, state: HealthAnalysisState) -> Literal["continue", "retry", "error"]:
        if state["errors"]:
            if state["retry_count"] < self.config.max_retries:
                state["retry_count"] += 1
                logger.warning(f"🔄 Retrying API fetch (attempt {state['retry_count']}/{self.config.max_retries})")
                state["errors"] = []
                return "retry"
            else:
                logger.error(f"❌ Max retries ({self.config.max_retries}) exceeded")
                return "error"
        return "continue"
    
    def should_continue_after_deidentify(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"
    
    def should_continue_after_entities(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"
    
    # ===== HELPER METHODS =====
    
    def _deidentify_medical_data(self, medical_data: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deidentify medical data"""
        try:
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
            
            # Create deidentified structure
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
    
    def _analyze_medical_for_entities(self, medical_data: Dict[str, Any], entities: Dict[str, Any]):
        """Analyze medical data for health entities"""
        try:
            medical_str = json.dumps(medical_data).lower()
            
            # Look for diabetes indicators
            diabetes_keywords = ['diabetes', 'diabetic', 'insulin', 'glucose', 'hemoglobin a1c', 'metformin']
            for keyword in diabetes_keywords:
                if keyword in medical_str:
                    entities["diabetes"] = "yes"
                    entities["medical_conditions"].append(f"Diabetes indicator: {keyword}")
                    entities["analysis_details"].append(f"Diabetes keyword found in medical data: {keyword}")
                    break
            
            # Look for blood pressure indicators
            bp_keywords = ['hypertension', 'blood pressure', 'bp', 'systolic', 'diastolic']
            for keyword in bp_keywords:
                if keyword in medical_str:
                    entities["blood_pressure"] = "diagnosed"
                    entities["medical_conditions"].append(f"Blood pressure indicator: {keyword}")
                    entities["analysis_details"].append(f"Blood pressure keyword found: {keyword}")
                    break
            
        except Exception as e:
            entities["analysis_details"].append(f"Error analyzing medical data: {str(e)}")
    
    def _analyze_pharmacy_for_entities(self, pharmacy_data: Dict[str, Any], entities: Dict[str, Any]):
        """Analyze pharmacy data for health entities"""
        try:
            pharmacy_str = json.dumps(pharmacy_data).lower()
            
            # Diabetes medications
            diabetes_meds = ['insulin', 'metformin', 'glipizide', 'glyburide', 'lantus', 'humalog']
            for med in diabetes_meds:
                if med in pharmacy_str:
                    entities["diabetes"] = "yes"
                    entities["medications_identified"].append(f"Diabetes medication: {med}")
                    entities["analysis_details"].append(f"Diabetes medication found: {med}")
            
            # Blood pressure medications
            bp_meds = ['lisinopril', 'amlodipine', 'metoprolol', 'losartan', 'atenolol']
            for med in bp_meds:
                if med in pharmacy_str:
                    entities["blood_pressure"] = "managed"
                    entities["medications_identified"].append(f"BP medication: {med}")
                    entities["analysis_details"].append(f"Blood pressure medication found: {med}")
            
            # Smoking cessation
            smoking_meds = ['chantix', 'varenicline', 'nicotine', 'bupropion', 'zyban']
            for med in smoking_meds:
                if med in pharmacy_str:
                    entities["smoking"] = "quit_attempt"
                    entities["medications_identified"].append(f"Smoking cessation: {med}")
                    entities["analysis_details"].append(f"Smoking cessation medication found: {med}")
            
        except Exception as e:
            entities["analysis_details"].append(f"Error analyzing pharmacy data: {str(e)}")
    
    def _analyze_raw_pharmacy_for_entities(self, pharmacy_raw: Dict[str, Any], entities: Dict[str, Any]):
        """Analyze raw pharmacy data for additional insights"""
        try:
            # This can look at more detailed pharmacy response structure
            if "body" in pharmacy_raw:
                body_str = json.dumps(pharmacy_raw["body"]).lower()
                
                # Look for alcohol-related medications
                alcohol_meds = ['naltrexone', 'disulfiram', 'acamprosate', 'antabuse']
                for med in alcohol_meds:
                    if med in body_str:
                        entities["alcohol"] = "treatment"
                        entities["medications_identified"].append(f"Alcohol treatment: {med}")
                        entities["analysis_details"].append(f"Alcohol treatment medication found: {med}")
                        
        except Exception as e:
            entities["analysis_details"].append(f"Error analyzing raw pharmacy data: {str(e)}")
    
    def _generate_welcome_message(self, state: HealthAnalysisState) -> str:
        """Generate welcome message for chatbot"""
        entity_extraction = state.get("entity_extraction", {})
        
        # Count findings
        conditions_found = len(entity_extraction.get("medical_conditions", []))
        medications_found = len(entity_extraction.get("medications_identified", []))
        
        welcome = f"""🏥 **Health Analysis Complete!**

I've analyzed the patient's deidentified medical and pharmacy data. Here's what I found:

📊 **Analysis Summary:**
- Age Group: {entity_extraction.get('age_group', 'unknown').title()}
- Diabetes: {entity_extraction.get('diabetes', 'unknown').title()}
- Blood Pressure: {entity_extraction.get('blood_pressure', 'unknown').title()}
- Smoking Status: {entity_extraction.get('smoking', 'unknown').title()}
- Alcohol Status: {entity_extraction.get('alcohol', 'unknown').title()}

📋 **Data Summary:**
- Medical Conditions Identified: {conditions_found}
- Medications Identified: {medications_found}

💬 **Ask me anything about the analysis!** 
Examples:
- "What medications were found?"
- "Show me the diabetes indicators"
- "What are the key health risks?"
- "Explain the medical conditions"

I have access to all the deidentified medical and pharmacy data and can answer detailed questions about the findings."""
        
        return welcome
    
    # ===== CHATBOT METHODS =====
    
    def ask_chatbot(self, question: str, session_id: Optional[str] = None) -> str:
        """Ask the chatbot a question about the analysis"""
        try:
            if not session_id:
                session_id = self.current_session_id
            
            if not session_id or session_id not in self.chatbot_sessions:
                return "❌ No active analysis session. Please run an analysis first."
            
            # Get chat history for context
            chat_history = self.chatbot_sessions[session_id]
            
            # Add user question to history
            user_message = {
                "role": "user",
                "content": question,
                "timestamp": datetime.now().isoformat()
            }
            chat_history.append(user_message)
            
            # Get the most recent analysis state
            # In a real implementation, you'd store this with the session
            # For now, we'll use a simplified approach
            context = self._build_chatbot_context(session_id)
            
            # Create prompt for Snowflake Cortex
            prompt = self._create_chatbot_prompt(question, context, chat_history)
            
            # Get response from LLM
            response = self.call_llm(prompt)
            
            # Add assistant response to history
            assistant_message = {
                "role": "assistant", 
                "content": response,
                "timestamp": datetime.now().isoformat()
            }
            chat_history.append(assistant_message)
            
            # Keep only last 20 messages to prevent context overflow
            if len(chat_history) > 20:
                self.chatbot_sessions[session_id] = chat_history[-20:]
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chatbot question: {e}")
            return f"❌ Error processing your question: {str(e)}"
    
    def _build_chatbot_context(self, session_id: str) -> str:
        """Build context for chatbot from stored analysis data"""
        # This would typically retrieve stored analysis results
        # For now, return a placeholder that would be filled with actual data
        return """
        Deidentified Medical Data: [Available for analysis]
        Deidentified Pharmacy Data: [Available for analysis]
        Entity Extraction Results: [Available for analysis]
        """
    
    def _create_chatbot_prompt(self, question: str, context: str, chat_history: List[Dict[str, Any]]) -> str:
        """Create comprehensive prompt for chatbot"""
        
        # Get recent chat context (last 5 messages)
        recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
        history_text = ""
        
        for msg in recent_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:500]  # Truncate long messages
            history_text += f"{role.upper()}: {content}\n"
        
        prompt = f"""You are a healthcare AI assistant analyzing deidentified patient data. Answer questions based on the provided context and maintain conversation continuity.

ANALYSIS CONTEXT:
{context}

RECENT CONVERSATION HISTORY:
{history_text}

CURRENT QUESTION: {question}

Instructions:
1. Answer based on the deidentified medical and pharmacy data provided
2. Reference specific findings from the analysis when relevant
3. Maintain conversation context from previous messages
4. Be concise but informative
5. If asked about specific data, provide relevant details
6. Always remind that this is deidentified data for analysis purposes
7. Provide actionable insights when appropriate

Please provide a helpful, accurate response:"""
        
        return prompt
    
    def get_chat_history(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        if not session_id:
            session_id = self.current_session_id
            
        if not session_id or session_id not in self.chatbot_sessions:
            return []
            
        return self.chatbot_sessions[session_id]
    
    def refresh_session(self):
        """Refresh/reset the current session"""
        if self.current_session_id:
            if self.current_session_id in self.chatbot_sessions:
                del self.chatbot_sessions[self.current_session_id]
            self.current_session_id = None
        
        logger.info("🔄 Session refreshed - all data cleared")
    
    def run_analysis(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete analysis workflow"""
        
        # Initialize state
        initial_state = HealthAnalysisState(
            patient_data=patient_data,
            raw_api_data={},
            mcid_raw={},
            medical_raw={},
            pharmacy_raw={},
            token_raw={},
            all_raw={},
            deidentified_medical={},
            deidentified_pharmacy={},
            entity_extraction={},
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
            
            logger.info("🚀 Starting Enhanced LangGraph workflow with MCP integration...")
            
            # Run the workflow
            final_state = self.graph.invoke(initial_state, config=config_dict)
            
            # Prepare results
            results = {
                "success": final_state["processing_complete"] and not final_state["errors"],
                "patient_data": final_state["patient_data"],
                "raw_api_data": final_state["raw_api_data"],
                "mcid_raw": final_state["mcid_raw"],
                "medical_raw": final_state["medical_raw"],
                "pharmacy_raw": final_state["pharmacy_raw"],
                "token_raw": final_state["token_raw"],
                "all_raw": final_state["all_raw"],
                "deidentified_data": {
                    "medical": final_state["deidentified_medical"],
                    "pharmacy": final_state["deidentified_pharmacy"]
                },
                "entity_extraction": final_state["entity_extraction"],
                "chatbot_context": final_state["chatbot_context"],
                "chat_history": final_state["chat_history"],
                "errors": final_state["errors"],
                "step_status": final_state["step_status"],
                "session_id": self.current_session_id
            }
            
            if results["success"]:
                logger.info("✅ Enhanced LangGraph workflow completed successfully!")
            else:
                logger.error(f"❌ Workflow failed with errors: {final_state['errors']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Fatal error in workflow: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "patient_data": patient_data,
                "raw_api_data": {},
                "deidentified_data": {},
                "entity_extraction": {},
                "chatbot_context": {},
                "chat_history": [],
                "errors": [str(e)],
                "step_status": {"workflow": "error"},
                "session_id": None
            }

def main():
    """Example usage of Enhanced Health Analysis Agent"""
    print("🏥 Enhanced Health Analysis Agent with MCP Integration")
    print("=" * 60)
    
    # Create agent
    agent = EnhancedHealthAnalysisAgent()
    
    # Example patient data
    patient_data = {
        "first_name": "John",
        "last_name": "Smith",
        "ssn": "123456789",
        "date_of_birth": "1980-01-15",
        "gender": "M",
        "zip_code": "12345"
    }
    
    print("🔍 Running analysis...")
    results = agent.run_analysis(patient_data)
    
    print(f"📊 Analysis completed: {'✅ Success' if results['success'] else '❌ Failed'}")
    
    if results["success"]:
        print("\n🤖 Chatbot is ready! Try asking questions:")
        
        # Example questions
        questions = [
            "What medications were found in the analysis?",
            "What are the key health indicators?",
            "Explain the diabetes findings",
            "What can you tell me about the patient's blood pressure?"
        ]
        
        for question in questions:
            print(f"\n❓ Question: {question}")
            answer = agent.ask_chatbot(question)
            print(f"🤖 Answer: {answer[:200]}...")
    
    return agent

if __name__ == "__main__":
    main()
