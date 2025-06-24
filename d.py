import json
import re
import requests
import urllib3
import uuid
from datetime import datetime, date
from typing import Dict, Any, List, TypedDict, Literal, Optional
from dataclasses import dataclass
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
    model: str = "llama4-maverick"
    sys_msg: str = "You are a healthcare AI assistant. Provide accurate, concise answers based on the medical record context provided."
    timeout: int = 30

# Simple State Definition
class SimpleRAGState(TypedDict):
    user_message: str
    conversation_history: List[str]
    assistant_response: str
    
    # Patient data
    patient_data: Optional[Dict[str, Any]]
    
    # RAG Context - the key part!
    rag_active: bool
    json_context: List[Dict[str, Any]]  # Deidentified medical/pharmacy JSONs
    
    # Control
    mode: Literal["analysis", "rag"]
    errors: List[str]
    processing_complete: bool

class SimpleRAGHealthAgent:
    def __init__(self, custom_config: Optional[Config] = None):
        self.config = custom_config or Config()
        logger.info("🤖 Simple RAG Healthcare Agent initialized")
        
        self.setup_langgraph()
        
        # Simple session management
        self.session_id: Optional[str] = None
        self.rag_json_context: List[Dict[str, Any]] = []  # Our RAG knowledge base
        self.conversation_history: List[str] = []
        
    def setup_langgraph(self):
        """Setup simple LangGraph workflow"""
        logger.info("🔧 Setting up Simple RAG LangGraph...")
        
        workflow = StateGraph(SimpleRAGState)
        
        # Simple nodes
        workflow.add_node("process_input", self.process_input)
        workflow.add_node("analyze_patient", self.analyze_patient)
        workflow.add_node("rag_chat", self.rag_chat)
        workflow.add_node("generate_response", self.generate_response)
        
        # Simple flow
        workflow.add_edge(START, "process_input")
        
        # Route based on mode
        workflow.add_conditional_edges(
            "process_input", 
            self.route_request,
            {
                "analyze": "analyze_patient",
                "rag": "rag_chat", 
                "general": "generate_response"
            }
        )
        
        # Analysis flow -> RAG activation
        workflow.add_edge("analyze_patient", "generate_response")
        workflow.add_edge("rag_chat", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Compile
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)
        
        logger.info("✅ Simple RAG workflow ready!")
    
    def call_cortex_llm(self, user_query: str, json_context: List[Dict] = None) -> str:
        """Call Cortex LLM with JSON context (same as your example)"""
        try:
            session_id = str(uuid.uuid4())
            
            # Build context like your example
            history = "\n".join(self.conversation_history[-5:])  # Last 5 messages
            
            json_blob = ""
            if json_context:
                json_blob = f"\nThese are the relevant deidentified medical records:\n{json.dumps(json_context, indent=2)}\n"
            
            full_prompt = f"{self.config.sys_msg}\n{json_blob}\n{history}\nUser: {user_query}"
            
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
                        "messages": [{"role": "user", "content": full_prompt}]
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
                return f"❌ Cortex Error {response.status_code}: {response.text[:200]}"
                
        except Exception as e:
            return f"❌ Cortex Exception: {str(e)}"
    
    # ===== LANGGRAPH NODES =====
    
    def process_input(self, state: SimpleRAGState) -> SimpleRAGState:
        """Process user input and determine mode"""
        logger.info("🔄 Processing input...")
        
        user_message = state["user_message"]
        
        # Add to conversation history
        self.conversation_history.append(f"User: {user_message}")
        
        # Simple mode detection
        if self.rag_json_context:  # We have JSON context = RAG mode
            state["rag_active"] = True
            state["json_context"] = self.rag_json_context
            state["mode"] = "rag"
        elif self._is_analysis_request(user_message):
            state["rag_active"] = False
            state["mode"] = "analysis"
        else:
            state["rag_active"] = False
            state["mode"] = "general"
        
        logger.info(f"📝 Mode: {state['mode']}, RAG Active: {state['rag_active']}")
        return state
    
    def analyze_patient(self, state: SimpleRAGState) -> SimpleRAGState:
        """Analyze patient - fetch from MCP and setup RAG context"""
        logger.info("🔍 Analyzing patient...")
        
        try:
            # Extract patient data using LLM
            patient_data = self._extract_patient_data(state["user_message"])
            if not patient_data:
                state["errors"].append("Could not extract patient data")
                state["assistant_response"] = "Please provide patient details like: 'Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345'"
                state["processing_complete"] = True
                return state
            
            state["patient_data"] = patient_data
            
            # Call MCP server
            raw_responses = self._call_mcp_server(patient_data)
            
            # Deidentify and create JSON context
            json_context = []
            
            # Add deidentified medical data
            medical_data = raw_responses.get("medical", {})
            if medical_data and not medical_data.get("error"):
                deidentified_medical = self._deidentify_data(medical_data, patient_data)
                json_context.append({
                    "type": "medical_record",
                    "patient_id": "DEIDENTIFIED",
                    "data": deidentified_medical
                })
            
            # Add deidentified pharmacy data  
            pharmacy_data = raw_responses.get("pharmacy", {})
            if pharmacy_data and not pharmacy_data.get("error"):
                deidentified_pharmacy = self._deidentify_data(pharmacy_data, patient_data)
                json_context.append({
                    "type": "pharmacy_record",
                    "patient_id": "DEIDENTIFIED", 
                    "data": deidentified_pharmacy
                })
            
            # Add API status info
            api_status = self._get_api_status(raw_responses)
            json_context.append({
                "type": "api_status",
                "mcp_server_responses": api_status
            })
            
            # Store in RAG context
            self.rag_json_context = json_context
            state["json_context"] = json_context
            state["rag_active"] = True
            
            # Generate completion message
            patient_name = f"{patient_data.get('first_name', 'Unknown')} {patient_data.get('last_name', 'Unknown')}"
            state["assistant_response"] = f"""✅ **Analysis Complete for {patient_name}**

🔍 **Data Retrieved & Deidentified:**
- Medical records: {'✅' if medical_data and not medical_data.get('error') else '❌'}
- Pharmacy records: {'✅' if pharmacy_data and not pharmacy_data.get('error') else '❌'}
- MCP API calls: {api_status['successful']}/5 successful

🧠 **RAG Mode Activated!**
I now have the deidentified medical records in my context. Ask me anything about:
- Medical claims and conditions
- Pharmacy records and medications  
- API response details
- Data analysis and insights

**Try asking:**
- "How many medical claims were found?"
- "What medications are listed?" 
- "Show me the pharmacy data"
- "What conditions were identified?"

Ready for your questions! 🤖"""
            
            logger.info("✅ Analysis complete, RAG mode activated")
            
        except Exception as e:
            error_msg = f"Error in patient analysis: {str(e)}"
            state["errors"].append(error_msg)
            state["assistant_response"] = f"❌ Analysis failed: {error_msg}"
            logger.error(error_msg)
        
        state["processing_complete"] = True
        return state
    
    def rag_chat(self, state: SimpleRAGState) -> SimpleRAGState:
        """RAG chat using JSON context"""
        logger.info("🧠 RAG chat...")
        
        try:
            user_query = state["user_message"]
            json_context = state["json_context"]
            
            # Call LLM with JSON context (like your example)
            response = self.call_cortex_llm(user_query, json_context)
            
            state["assistant_response"] = f"{response}\n\n*🧠 RAG Mode: Answer based on deidentified medical records*"
            
            # Add to conversation history
            self.conversation_history.append(f"Assistant: {response}")
            
            logger.info("✅ RAG response generated")
            
        except Exception as e:
            error_msg = f"Error in RAG chat: {str(e)}"
            state["errors"].append(error_msg)
            state["assistant_response"] = f"❌ RAG chat failed: {error_msg}"
            logger.error(error_msg)
        
        state["processing_complete"] = True
        return state
    
    def generate_response(self, state: SimpleRAGState) -> SimpleRAGState:
        """Generate general response"""
        logger.info("💬 Generating response...")
        
        if not state.get("assistant_response"):
            user_message = state["user_message"].lower()
            
            if "help" in user_message or "what can you" in user_message:
                if self.rag_json_context:
                    state["assistant_response"] = """🧠 **RAG Mode Active - Medical Records Loaded**

I have deidentified medical records in my context. Ask me questions like:

📊 **Data Analysis:**
- "How many medical claims were found?"
- "What's the total number of records?"
- "Show me the API status"

💊 **Medical Information:**
- "What medications are listed?"
- "What medical conditions were found?"
- "Show me the pharmacy records"

🔍 **Detailed Questions:**
- "Analyze the medical data"
- "What patterns do you see?"
- "Summarize the key findings"

**🔄 Use Refresh to exit RAG mode and analyze new patient**"""
                else:
                    state["assistant_response"] = """🤖 **Healthcare Analysis Assistant**

I can analyze patient data and enter RAG mode to answer questions.

**To start:**
- "Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345"

**My process:**
1. Extract patient info from your command
2. Fetch data from MCP server
3. Deidentify medical/pharmacy records
4. Enter RAG mode with JSON context
5. Answer your questions using the data

**After analysis, ask me detailed questions about the medical records!**"""
            else:
                state["assistant_response"] = "Hello! Give me a patient analysis command or ask 'help' to see what I can do."
        
        state["processing_complete"] = True
        return state
    
    # ===== ROUTING =====
    
    def route_request(self, state: SimpleRAGState) -> Literal["analyze", "rag", "general"]:
        """Simple routing logic"""
        mode = state.get("mode", "general")
        logger.info(f"🔄 Routing to: {mode}")
        return mode
    
    # ===== HELPER METHODS =====
    
    def _is_analysis_request(self, message: str) -> bool:
        """Check if requesting patient analysis"""
        analysis_keywords = ["analyze", "patient", "evaluate", "check"]
        message_lower = message.lower()
        
        # Must have analysis keyword AND patient identifiers
        has_keyword = any(keyword in message_lower for keyword in analysis_keywords)
        has_name = any(word.istitle() for word in message.split())
        has_numbers = any(char.isdigit() for char in message)
        
        return has_keyword and has_name and has_numbers
    
    def _extract_patient_data(self, message: str) -> Optional[Dict[str, Any]]:
        """Extract patient data using LLM"""
        try:
            extraction_prompt = f"""
Extract patient information from this message and return ONLY a JSON object:
"{message}"

Required format:
{{
    "first_name": "string",
    "last_name": "string",
    "ssn": "digits only",
    "date_of_birth": "YYYY-MM-DD",
    "gender": "M or F",
    "zip_code": "string"
}}

Return only the JSON, no other text."""
            
            response = self.call_cortex_llm(extraction_prompt)
            
            # Try to parse JSON from response
            try:
                # Clean up response to get just JSON
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    patient_data = json.loads(json_str)
                    
                    # Validate required fields
                    required = ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"]
                    if all(patient_data.get(field) for field in required):
                        return patient_data
                        
            except json.JSONDecodeError:
                pass
                
            return None
            
        except Exception as e:
            logger.error(f"Error extracting patient data: {e}")
            return None
    
    def _call_mcp_server(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP server endpoints"""
        endpoints = {
            "medical": "/medical/submit",
            "pharmacy": "/pharmacy/submit",
            "mcid": "/mcid/search",
            "token": "/token", 
            "all": "/all"
        }
        
        responses = {}
        
        for name, path in endpoints.items():
            try:
                if name == "token":
                    response = requests.post(f"{self.config.mcp_server_url}{path}", timeout=self.config.timeout)
                else:
                    response = requests.post(f"{self.config.mcp_server_url}{path}", json=patient_data, timeout=self.config.timeout)
                
                if response.status_code == 200:
                    responses[name] = response.json()
                else:
                    responses[name] = {"error": f"HTTP {response.status_code}", "message": response.text[:200]}
                    
            except Exception as e:
                responses[name] = {"error": "Request failed", "message": str(e)}
        
        return responses
    
    def _deidentify_data(self, data: Any, patient_data: Dict[str, Any]) -> Any:
        """Simple deidentification"""
        if isinstance(data, dict):
            return {k: self._deidentify_data(v, patient_data) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._deidentify_data(item, patient_data) for item in data]
        elif isinstance(data, str):
            # Replace PII patterns
            data = re.sub(r'\b\d{3}-?\d{2}-?\d{4}\b', '[SSN_REMOVED]', data)
            data = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[NAME_REMOVED]', data)
            data = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REMOVED]', data)
            return data
        else:
            return data
    
    def _get_api_status(self, raw_responses: Dict[str, Any]) -> Dict[str, Any]:
        """Get API status summary"""
        successful = 0
        details = []
        
        for endpoint, response in raw_responses.items():
            if response and not response.get("error"):
                successful += 1
                details.append(f"✅ {endpoint}: Success")
            else:
                error = response.get("error", "No response") if response else "No response"
                details.append(f"❌ {endpoint}: {error}")
        
        return {
            "successful": successful,
            "total": len(raw_responses),
            "details": details
        }
    
    # ===== PUBLIC METHODS =====
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """Main chat interface"""
        try:
            if not self.session_id:
                self.session_id = str(uuid.uuid4())
            
            # Initialize state
            initial_state = SimpleRAGState(
                user_message=user_message,
                conversation_history=self.conversation_history.copy(),
                assistant_response="",
                patient_data=None,
                rag_active=bool(self.rag_json_context),
                json_context=self.rag_json_context.copy(),
                mode="general",
                errors=[],
                processing_complete=False
            )
            
            # Run workflow
            config = {"configurable": {"thread_id": self.session_id}}
            final_state = self.graph.invoke(initial_state, config=config)
            
            # Return result
            return {
                "success": final_state["processing_complete"] and not final_state["errors"],
                "response": final_state["assistant_response"],
                "rag_active": final_state["rag_active"],
                "json_context": final_state["json_context"],
                "patient_data": final_state.get("patient_data"),
                "errors": final_state["errors"],
                "session_id": self.session_id
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {
                "success": False,
                "response": f"❌ Error: {str(e)}",
                "rag_active": bool(self.rag_json_context),
                "errors": [str(e)],
                "session_id": self.session_id
            }
    
    def refresh_session(self):
        """Reset everything"""
        self.session_id = None
        self.rag_json_context = []
        self.conversation_history = []
        logger.info("🔄 Session refreshed")
    
    def get_rag_context(self) -> List[Dict[str, Any]]:
        """Get current RAG context"""
        return self.rag_json_context.copy()

def main():
    """Test the simple RAG agent"""
    print("🧠 Simple RAG Healthcare Agent")
    print("=" * 40)
    
    agent = SimpleRAGHealthAgent()
    
    test_messages = [
        "What can you help me with?",
        "Analyze patient John Smith, DOB 1980-01-15, male, SSN 123456789, zip code 12345",
        "How many medical claims were found?",
        "What medications are listed?"
    ]
    
    for message in test_messages:
        print(f"\n👤 User: {message}")
        result = agent.chat(message)
        print(f"🤖 Assistant: {result['response']}")
        print(f"RAG Active: {result['rag_active']}")
    
    return agent

if __name__ == "__main__":
    main()
