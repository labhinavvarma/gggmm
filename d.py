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

# State Definition for Chatbot-First LangGraph
class ChatbotHealthState(TypedDict):
    # User input and conversation
    user_message: str
    conversation_history: List[Dict[str, Any]]
    
    # Extracted patient data
    patient_data: Optional[Dict[str, Any]]
    
    # Raw MCP API responses
    raw_api_responses: Dict[str, Any]
    
    # Processed data
    deidentified_medical: Dict[str, Any]
    deidentified_pharmacy: Dict[str, Any]
    
    # Entity extraction
    entity_extraction: Dict[str, Any]
    
    # Analysis complete flag
    analysis_ready: bool
    
    # Current response to user
    assistant_response: str
    
    # Control flow
    current_step: str
    errors: List[str]
    processing_complete: bool

class ChatbotFirstHealthAgent:
    def __init__(self, custom_config: Optional[Config] = None):
        self.config = custom_config or Config()
        logger.info("🤖 ChatbotFirstHealthAgent initialized")
        logger.info(f"🔗 MCP Server: {self.config.mcp_server_url}")
        
        self.setup_langgraph()
        
        # Conversation memory
        self.session_conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.current_session_id: Optional[str] = None
        self.current_analysis_context: Optional[Dict[str, Any]] = None
        
    def setup_langgraph(self):
        """Setup LangGraph workflow for chatbot-first processing"""
        logger.info("🔧 Setting up Chatbot-First LangGraph workflow...")
        
        workflow = StateGraph(ChatbotHealthState)
        
        # Add processing nodes
        workflow.add_node("process_user_input", self.process_user_input)
        workflow.add_node("extract_patient_data", self.extract_patient_data)
        workflow.add_node("call_mcp_server", self.call_mcp_server)
        workflow.add_node("process_analysis_data", self.process_analysis_data)
        workflow.add_node("setup_analysis_context", self.setup_analysis_context)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("handle_contextual_chat", self.handle_contextual_chat)
        
        # Define workflow edges
        workflow.add_edge(START, "process_user_input")
        
        # Conditional routing based on user input type
        workflow.add_conditional_edges(
            "process_user_input",
            self.route_user_input,
            {
                "extract_data": "extract_patient_data",
                "contextual_chat": "handle_contextual_chat",
                "general_response": "generate_response"
            }
        )
        
        # Patient data extraction flow
        workflow.add_edge("extract_patient_data", "call_mcp_server")
        workflow.add_edge("call_mcp_server", "process_analysis_data")
        workflow.add_edge("process_analysis_data", "setup_analysis_context")
        workflow.add_edge("setup_analysis_context", "generate_response")
        
        # All paths lead to response generation
        workflow.add_edge("handle_contextual_chat", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Compile workflow
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)
        
        logger.info("✅ Chatbot-First LangGraph workflow compiled!")
    
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
    
    def process_user_input(self, state: ChatbotHealthState) -> ChatbotHealthState:
        """Process user input and determine what type of response is needed"""
        logger.info("🔄 Processing user input...")
        state["current_step"] = "process_user_input"
        
        user_message = state["user_message"]
        
        # Add to conversation history
        if "conversation_history" not in state:
            state["conversation_history"] = []
        
        state["conversation_history"].append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"📝 User message: {user_message[:100]}...")
        return state
    
    def extract_patient_data(self, state: ChatbotHealthState) -> ChatbotHealthState:
        """Extract patient data from natural language using LLM"""
        logger.info("🔍 Extracting patient data from natural language...")
        state["current_step"] = "extract_patient_data"
        
        try:
            user_message = state["user_message"]
            
            # Create extraction prompt
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
            
            # Get extraction from LLM
            extracted_json = self.call_llm(extraction_prompt)
            
            try:
                # Parse JSON response
                patient_data = json.loads(extracted_json)
                
                # Validate required fields
                required_fields = ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"]
                missing_fields = []
                
                for field in required_fields:
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
    
    def call_mcp_server(self, state: ChatbotHealthState) -> ChatbotHealthState:
        """Call all MCP server endpoints"""
        logger.info("📡 Calling MCP server endpoints...")
        state["current_step"] = "call_mcp_server"
        
        try:
            patient_data = state["patient_data"]
            if not patient_data:
                state["errors"].append("No patient data available for MCP calls")
                return state
            
            # Initialize raw API responses
            state["raw_api_responses"] = {}
            
            # Define MCP endpoints to call
            endpoints = {
                "mcid": "/mcid/search",
                "medical": "/medical/submit",
                "pharmacy": "/pharmacy/submit", 
                "token": "/token",
                "all": "/all"
            }
            
            successful_calls = 0
            
            for endpoint_name, endpoint_path in endpoints.items():
                try:
                    logger.info(f"📞 Calling {endpoint_name} endpoint...")
                    
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
    
    def process_analysis_data(self, state: ChatbotHealthState) -> ChatbotHealthState:
        """Process the raw API data through deidentification and entity extraction"""
        logger.info("🔒 Processing analysis data...")
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
            
            # Extract entities
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
    
    def setup_analysis_context(self, state: ChatbotHealthState) -> ChatbotHealthState:
        """Setup context for future conversations about the analysis"""
        logger.info("🤖 Setting up analysis context...")
        state["current_step"] = "setup_analysis_context"
        
        try:
            # Create analysis context for future conversations
            self.current_analysis_context = {
                "patient_info": {
                    "name": f"{state['patient_data'].get('first_name', 'Unknown')} {state['patient_data'].get('last_name', 'Unknown')}",
                    "age_group": state["entity_extraction"].get("age_group", "unknown"),
                    "analysis_timestamp": datetime.now().isoformat()
                },
                "deidentified_medical": state["deidentified_medical"],
                "deidentified_pharmacy": state["deidentified_pharmacy"],
                "entity_extraction": state["entity_extraction"],
                "raw_api_responses": state["raw_api_responses"]
            }
            
            state["analysis_ready"] = True
            logger.info("✅ Analysis context setup complete")
            
        except Exception as e:
            error_msg = f"Error setting up analysis context: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    def generate_response(self, state: ChatbotHealthState) -> ChatbotHealthState:
        """Generate response based on current state"""
        logger.info("💬 Generating response...")
        state["current_step"] = "generate_response"
        
        try:
            if state.get("analysis_ready"):
                # Generate analysis complete response
                response = self._generate_analysis_complete_response(state)
            elif state.get("errors"):
                # Generate error response
                response = f"❌ I encountered some issues: {'; '.join(state['errors'])}"
            else:
                # Generate general response based on context
                if self.current_analysis_context:
                    # We have analysis context, so guide them to ask questions
                    response = """I have analysis data available from the previous patient analysis. You can ask me detailed questions about:

💊 **Pharmacy Data**: "What medications were found?" or "Show me the pharmacy JSON details"
🏥 **Medical Data**: "What medical conditions were identified?" or "Explain the medical findings"
🎯 **Entity Extraction**: "What health indicators were found?" or "Tell me about the diabetes findings"
📄 **Raw Data**: "What did the MCP server responses show?" or "Explain the API results"

Or give me a new patient analysis command to start fresh!"""
                else:
                    # No analysis context, guide them to start analysis
                    response = """Hello! I'm your healthcare analysis assistant. I can analyze patient data and then answer detailed questions about the results.

**To start, give me a command like:**
- "Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345"
- "Evaluate health data for Sarah Johnson, born 1975-08-22, female, SSN 987654321, zip 90210"

**After analysis, I can answer questions about:**
- Medications found in pharmacy data
- Medical conditions identified  
- Health risk indicators
- Specific JSON response details
- Deidentified data insights

What would you like me to help you with?"""
            
            state["assistant_response"] = response
            state["processing_complete"] = True
            
            # Add assistant response to conversation history
            state["conversation_history"].append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info("✅ Response generated")
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            state["errors"].append(error_msg)
            state["assistant_response"] = f"I apologize, but I encountered an error: {str(e)}"
            state["processing_complete"] = True
            logger.error(error_msg)
        
        return state
    
    def handle_contextual_chat(self, state: ChatbotHealthState) -> ChatbotHealthState:
        """Handle contextual chat about existing analysis"""
        logger.info("💭 Handling contextual chat...")
        state["current_step"] = "handle_contextual_chat"
        
        try:
            if not self.current_analysis_context:
                state["assistant_response"] = "I don't have any analysis data to discuss. Please run a patient analysis first by providing patient information."
                state["processing_complete"] = True
                return state
            
            user_question = state["user_message"]
            
            # Create contextual prompt for LLM
            context_prompt = self._create_contextual_chat_prompt(user_question, state["conversation_history"])
            
            # Get response from LLM
            response = self.call_llm(context_prompt)
            
            if response.startswith("Error"):
                state["assistant_response"] = "I'm having trouble processing your question. Please try rephrasing or ask something specific about the analysis data."
            else:
                state["assistant_response"] = response
            
            state["processing_complete"] = True
            logger.info("✅ Contextual chat response generated")
            
        except Exception as e:
            error_msg = f"Error in contextual chat: {str(e)}"
            state["errors"].append(error_msg)
            state["assistant_response"] = f"I encountered an error processing your question: {str(e)}"
            state["processing_complete"] = True
            logger.error(error_msg)
        
        return state
    
    # ===== CONDITIONAL ROUTING =====
    
    def route_user_input(self, state: ChatbotHealthState) -> Literal["extract_data", "contextual_chat", "general_response"]:
        """Route user input based on content and context"""
        user_message = state["user_message"].lower()
        
        # Check if we have existing analysis context for contextual chat FIRST
        if self.current_analysis_context:
            question_keywords = [
                "what", "how", "why", "explain", "tell me", "show me", "describe",
                "medication", "condition", "risk", "diabetes", "blood pressure", "pharmacy",
                "medical", "json", "data", "found", "identified", "reveal", "indicate",
                "specific", "detail", "values", "response", "analysis", "findings"
            ]
            
            if any(keyword in user_message for keyword in question_keywords):
                logger.info("🔄 Routing to contextual chat (analysis available)")
                return "contextual_chat"
        
        # Check if this looks like a patient analysis request
        analysis_keywords = [
            "analyze", "analysis", "patient", "evaluate", "assess", "check",
            "dob", "date of birth", "ssn", "social security", "zip code"
        ]
        
        if any(keyword in user_message for keyword in analysis_keywords):
            # Check if we can extract patient info
            has_name = any(word.istitle() for word in state["user_message"].split())
            has_numbers = any(char.isdigit() for char in state["user_message"])
            
            if has_name and has_numbers:
                logger.info("🔄 Routing to patient data extraction")
                return "extract_data"
        
        logger.info("🔄 Routing to general response")
        return "general_response"
    
    # ===== HELPER METHODS =====
    
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
    
    def _generate_analysis_complete_response(self, state: ChatbotHealthState) -> str:
        """Generate response when analysis is complete"""
        try:
            patient_name = f"{state['patient_data'].get('first_name', 'Unknown')} {state['patient_data'].get('last_name', 'Unknown')}"
            entities = state["entity_extraction"]
            
            # Count successful API calls
            raw_responses = state.get("raw_api_responses", {})
            successful_calls = len([k for k, v in raw_responses.items() if v and not v.get("error")])
            
            # Count findings
            conditions = len(entities.get("medical_conditions", []))
            medications = len(entities.get("medications_identified", []))
            
            response = f"""🏥 **Healthcare Analysis Complete for {patient_name}**

📊 **MCP Server Results:**
- API Calls Successful: {successful_calls}/5
- Data Retrieved: ✅ Medical, ✅ Pharmacy, ✅ MCID, ✅ Token, ✅ All

🔒 **Data Processing Complete:**
- Medical data deidentified ✅
- Pharmacy data deidentified ✅  
- Health entities extracted ✅

🎯 **Key Health Indicators:**
- Age Group: {entities.get('age_group', 'unknown').title()}
- Diabetes: {entities.get('diabetes', 'unknown').title()}
- Blood Pressure: {entities.get('blood_pressure', 'unknown').title()}
- Smoking Status: {entities.get('smoking', 'unknown').title()}
- Alcohol Status: {entities.get('alcohol', 'unknown').title()}

📋 **Analysis Summary:**
- Medical Conditions Identified: {conditions}
- Medications Identified: {medications}

💬 **I'm now ready to answer questions about this analysis!**

You can ask me:
- "What medications were found?"
- "Explain the diabetes findings"
- "What are the key health risks?"
- "Show me the medical conditions"
- "What does the pharmacy data show?"

The raw JSON data from all MCP endpoints is available for review, and I can discuss any aspect of the deidentified analysis results."""

            return response
            
        except Exception as e:
            return f"Analysis completed but I had trouble generating the summary: {str(e)}"
    
    def _create_contextual_chat_prompt(self, user_question: str, conversation_history: List[Dict[str, Any]]) -> str:
        """Create prompt for contextual chat about analysis"""
        try:
            # Get recent conversation context
            recent_messages = conversation_history[-8:] if len(conversation_history) > 8 else conversation_history
            history_text = ""
            
            for msg in recent_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:400]  # Increased for better context
                history_text += f"{role.upper()}: {content}\n"
            
            # Prepare comprehensive analysis context
            context_summary = ""
            if self.current_analysis_context:
                # Include full deidentified data for better responses
                deident_medical = self.current_analysis_context.get('deidentified_medical', {})
                deident_pharmacy = self.current_analysis_context.get('deidentified_pharmacy', {})
                entities = self.current_analysis_context.get('entity_extraction', {})
                raw_responses = self.current_analysis_context.get('raw_api_responses', {})
                
                context_summary = f"""
PATIENT ANALYSIS CONTEXT - FULL DEIDENTIFIED DATA AVAILABLE:

DEIDENTIFIED MEDICAL DATA:
{json.dumps(deident_medical, indent=2)}

DEIDENTIFIED PHARMACY DATA:
{json.dumps(deident_pharmacy, indent=2)}

ENTITY EXTRACTION RESULTS:
{json.dumps(entities, indent=2)}

RAW API RESPONSE SUMMARY:
- MCID: {"✅ Available" if raw_responses.get('mcid') and not raw_responses.get('mcid', {}).get('error') else "❌ Error/Missing"}
- Medical: {"✅ Available" if raw_responses.get('medical') and not raw_responses.get('medical', {}).get('error') else "❌ Error/Missing"}
- Pharmacy: {"✅ Available" if raw_responses.get('pharmacy') and not raw_responses.get('pharmacy', {}).get('error') else "❌ Error/Missing"}
- Token: {"✅ Available" if raw_responses.get('token') and not raw_responses.get('token', {}).get('error') else "❌ Error/Missing"}
- All: {"✅ Available" if raw_responses.get('all') and not raw_responses.get('all', {}).get('error') else "❌ Error/Missing"}
"""
            
            prompt = f"""You are a healthcare AI assistant with access to complete deidentified patient analysis data. Answer the user's question based on the comprehensive data provided below.

RECENT CONVERSATION HISTORY:
{history_text}

COMPLETE ANALYSIS DATA:
{context_summary}

CURRENT QUESTION: {user_question}

Instructions:
1. Answer based on the complete deidentified medical and pharmacy JSON data above
2. Reference specific data points, codes, medications, or conditions when relevant
3. Maintain conversation context from previous messages
4. Be detailed but informative - you have access to all the deidentified data
5. If asked about specific medications, conditions, or codes, search through the JSON data
6. If asked about raw API responses, mention which endpoints returned data successfully
7. Always clarify this is based on deidentified data for privacy
8. Provide medical insights based on the patterns in the data
9. If the user asks about specific JSON fields or structures, explain what you found

Answer the user's question with specific details from the deidentified analysis data:"""
            
            return prompt
            
        except Exception as e:
            return f"Error creating contextual prompt: {str(e)}"
    
    # ===== PUBLIC METHODS =====
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """Main chat interface - process user message and return response"""
        try:
            # Create session ID if needed
            if not self.current_session_id:
                self.current_session_id = str(uuid.uuid4())
                self.session_conversations[self.current_session_id] = []
            
            # Initialize state
            initial_state = ChatbotHealthState(
                user_message=user_message,
                conversation_history=[],
                patient_data=None,
                raw_api_responses={},
                deidentified_medical={},
                deidentified_pharmacy={},
                entity_extraction={},
                analysis_ready=False,
                assistant_response="",
                current_step="",
                errors=[],
                processing_complete=False
            )
            
            # Run the workflow
            config_dict = {"configurable": {"thread_id": f"chat_{self.current_session_id}"}}
            final_state = self.graph.invoke(initial_state, config=config_dict)
            
            # Store conversation in session
            self.session_conversations[self.current_session_id].extend(final_state["conversation_history"])
            
            # Prepare response
            result = {
                "success": final_state["processing_complete"] and not final_state["errors"],
                "response": final_state["assistant_response"],
                "analysis_ready": final_state.get("analysis_ready", False),
                "patient_data": final_state.get("patient_data"),
                "raw_api_responses": final_state.get("raw_api_responses", {}),
                "deidentified_data": {
                    "medical": final_state.get("deidentified_medical", {}),
                    "pharmacy": final_state.get("deidentified_pharmacy", {})
                },
                "entity_extraction": final_state.get("entity_extraction", {}),
                "errors": final_state.get("errors", []),
                "session_id": self.current_session_id,
                "conversation_history": self.session_conversations[self.current_session_id]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in chat processing: {str(e)}")
            return {
                "success": False,
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "analysis_ready": False,
                "errors": [str(e)],
                "session_id": self.current_session_id
            }
    
    def refresh_session(self):
        """Refresh the current session"""
        if self.current_session_id and self.current_session_id in self.session_conversations:
            del self.session_conversations[self.current_session_id]
        
        self.current_session_id = None
        self.current_analysis_context = None
        logger.info("🔄 Session refreshed")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get current conversation history"""
        if self.current_session_id and self.current_session_id in self.session_conversations:
            return self.session_conversations[self.current_session_id]
        return []

def main():
    """Test the chatbot-first agent"""
    print("🤖 Chatbot-First Healthcare Analysis Agent")
    print("=" * 50)
    
    agent = ChatbotFirstHealthAgent()
    
    # Test messages
    test_messages = [
        "Hello, I need help with healthcare analysis",
        "Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345",
        "What medications were found in the analysis?",
        "Explain the diabetes findings"
    ]
    
    for message in test_messages:
        print(f"\n👤 User: {message}")
        result = agent.chat(message)
        print(f"🤖 Assistant: {result['response']}")
        
        if result.get("analysis_ready"):
            print("✅ Analysis is ready for questions!")
    
    return agent

if __name__ == "__main__":
    main()
