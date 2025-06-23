#!/usr/bin/env python3
"""
Updated MCP Client for FastMCP Server
=====================================

Fixed client that properly connects to the FastMCP server and works with
the Snowflake Cortex LLM for natural language healthcare queries.

Usage:
    python updated_mcp_client.py
"""

import asyncio
import json
import logging
import re
import sys
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict

# MCP and LangGraph imports
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.language_models.base import BaseLanguageModel
    from langchain_core.tools import BaseTool
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("📦 Install with: pip install langchain-mcp-adapters langgraph langchain-core")
    sys.exit(1)

# HTTP client for Snowflake Cortex
import requests
import urllib3

# Disable SSL warnings for development (remove in production)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcp_client.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SnowflakeCortexConfig:
    """Configuration for Snowflake Cortex API"""
    api_url: str = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
    api_key: str = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
    app_id: str = "edadip"
    aplctn_cd: str = "edagnai"
    model: str = "llama3.1-70b"
    sys_msg: str = """You are a healthcare AI assistant that helps users interact with Milliman medical APIs through natural language. 

You have access to these tools:
- get_token: Get authentication token for API access
- medical_submit: Submit medical record requests  
- mcid_search: Search MCID database for patient coverage
- get_all_data: Get comprehensive patient data from all sources

When users provide patient information, use the appropriate tools to retrieve their data. Always validate patient information and provide clear explanations of the results."""
    max_retries: int = 3
    timeout: int = 30

@dataclass
class MCPConfig:
    """Configuration for MCP server connection"""
    server_name: str = "MillimanServer"
    server_url: str = "http://localhost:8000/sse"
    transport: str = "sse"
    max_retries: int = 3
    retry_delay: float = 2.0

@dataclass
class PatientData:
    """Patient data structure for API calls"""
    first_name: str
    last_name: str
    ssn: str
    date_of_birth: str  # YYYY-MM-DD format
    gender: str
    zip_code: str
    
    def to_dict(self) -> Dict[str, str]:
        return asdict(self)
    
    def validate(self) -> tuple[bool, List[str]]:
        """Validate patient data"""
        errors = []
        
        if not self.first_name or len(self.first_name.strip()) < 2:
            errors.append("First name must be at least 2 characters")
        
        if not self.last_name or len(self.last_name.strip()) < 2:
            errors.append("Last name must be at least 2 characters")
        
        if not self.ssn or len(re.sub(r'\D', '', self.ssn)) != 9:
            errors.append("SSN must be exactly 9 digits")
        
        if not self.gender or self.gender.upper() not in ['M', 'F']:
            errors.append("Gender must be 'M' or 'F'")
        
        if not self.zip_code or len(re.sub(r'\D', '', self.zip_code)) < 5:
            errors.append("Zip code must be at least 5 digits")
        
        # Validate date format
        try:
            datetime.strptime(self.date_of_birth, '%Y-%m-%d')
        except ValueError:
            errors.append("Date of birth must be in YYYY-MM-DD format")
        
        return len(errors) == 0, errors

class SnowflakeCortexLLM(BaseLanguageModel):
    """Snowflake Cortex LLM implementation for LangChain compatibility"""
    
    def __init__(self, config: SnowflakeCortexConfig):
        super().__init__()
        self.config = config
        logger.info(f"🔧 Initialized Snowflake Cortex LLM: {config.model}")
    
    @property
    def _llm_type(self) -> str:
        return "snowflake_cortex"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Synchronous call (not typically used by LangGraph)"""
        return asyncio.run(self._acall(prompt, stop, **kwargs))
    
    async def _acall(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Asynchronous call to Snowflake Cortex API"""
        try:
            return await self._call_cortex_api(prompt)
        except Exception as e:
            logger.error(f"❌ Error in _acall: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    async def _call_cortex_api(self, user_message: str) -> str:
        """Call Snowflake Cortex API with retry logic"""
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
        
        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"🤖 Calling Snowflake Cortex API (attempt {attempt + 1})")
                
                response = requests.post(
                    self.config.api_url,
                    headers=headers,
                    json=payload,
                    verify=False,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    raw_response = response.text
                    
                    # Parse response - handle end_of_stream marker
                    if "end_of_stream" in raw_response:
                        answer, _, _ = raw_response.partition("end_of_stream")
                        return answer.strip()
                    else:
                        return raw_response.strip()
                
                else:
                    logger.warning(f"⚠️ API error {response.status_code}: {response.text}")
                    if attempt == self.config.max_retries - 1:
                        return f"API Error {response.status_code}: {response.text[:200]}"
                
            except requests.exceptions.Timeout:
                logger.warning(f"⏱️ Timeout on attempt {attempt + 1}")
                if attempt == self.config.max_retries - 1:
                    return "Request timed out. Please try again."
                    
            except Exception as e:
                logger.error(f"❌ Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    return f"Unexpected error: {str(e)}"
                
            # Wait before retry
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return "Failed to get response after multiple attempts."

class PatientDataExtractor:
    """Extract patient data from natural language input"""
    
    @staticmethod
    def extract_patient_data(text: str) -> Optional[PatientData]:
        """Extract patient information from natural language text"""
        try:
            logger.info(f"🔍 Extracting patient data from: {text[:100]}...")
            
            # Initialize patient data with defaults
            patient_info = {
                'first_name': '',
                'last_name': '',
                'ssn': '',
                'date_of_birth': '',
                'gender': '',
                'zip_code': ''
            }
            
            # Extract SSN (9 digits with optional formatting)
            ssn_pattern = r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b'
            ssn_match = re.search(ssn_pattern, text)
            if ssn_match:
                patient_info['ssn'] = re.sub(r'\D', '', ssn_match.group())
                logger.info(f"✅ Found SSN: {patient_info['ssn'][:3]}******")
            
            # Extract date (YYYY-MM-DD or MM/DD/YYYY)
            date_patterns = [
                r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
                r'\b\d{1,2}-\d{1,2}-\d{4}\b'   # MM-DD-YYYY
            ]
            
            for pattern in date_patterns:
                date_match = re.search(pattern, text)
                if date_match:
                    date_str = date_match.group()
                    # Convert to YYYY-MM-DD format
                    if '/' in date_str:
                        parts = date_str.split('/')
                        if len(parts) == 3:
                            patient_info['date_of_birth'] = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
                    elif '-' in date_str and len(date_str.split('-')[0]) == 4:
                        patient_info['date_of_birth'] = date_str
                    elif '-' in date_str:
                        parts = date_str.split('-')
                        if len(parts) == 3:
                            patient_info['date_of_birth'] = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
                    logger.info(f"✅ Found DOB: {patient_info['date_of_birth']}")
                    break
            
            # Extract gender
            gender_pattern = r'\b(male|female|M|F)\b'
            gender_match = re.search(gender_pattern, text, re.IGNORECASE)
            if gender_match:
                gender = gender_match.group().upper()
                patient_info['gender'] = 'M' if gender in ['MALE', 'M'] else 'F'
                logger.info(f"✅ Found Gender: {patient_info['gender']}")
            
            # Extract zip code (5-9 digits)
            zip_pattern = r'\b\d{5}(?:-\d{4})?\b'
            zip_match = re.search(zip_pattern, text)
            if zip_match:
                patient_info['zip_code'] = zip_match.group()
                logger.info(f"✅ Found Zip: {patient_info['zip_code']}")
            
            # Extract names (look for common patterns)
            name_patterns = [
                r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b',  # First Last
                r'name[:\s]+([A-Z][a-z]+)\s+([A-Z][a-z]+)',  # name: First Last
                r'patient[:\s]+([A-Z][a-z]+)\s+([A-Z][a-z]+)'  # patient: First Last
            ]
            
            for pattern in name_patterns:
                name_match = re.search(pattern, text)
                if name_match:
                    patient_info['first_name'] = name_match.group(1)
                    patient_info['last_name'] = name_match.group(2)
                    logger.info(f"✅ Found Name: {patient_info['first_name']} {patient_info['last_name']}")
                    break
            
            # Check if we have minimum required information
            required_fields = ['ssn', 'date_of_birth']
            if all(patient_info.get(field) for field in required_fields):
                logger.info("✅ Sufficient patient data extracted")
                return PatientData(**patient_info)
            else:
                missing = [field for field in required_fields if not patient_info.get(field)]
                logger.warning(f"⚠️ Missing required fields: {missing}")
                return None
            
        except Exception as e:
            logger.error(f"❌ Error extracting patient data: {e}")
            return None

class MillimanMCPClient:
    """Main MCP client class integrating with Snowflake Cortex LLM"""
    
    def __init__(self, cortex_config: SnowflakeCortexConfig, mcp_config: MCPConfig):
        self.cortex_config = cortex_config
        self.mcp_config = mcp_config
        self.llm = SnowflakeCortexLLM(cortex_config)
        self.mcp_client = None
        self.agent = None
        self.chat_history: List[Dict[str, str]] = []
        self.patient_extractor = PatientDataExtractor()
        self.is_connected = False
        
        logger.info("🤖 Milliman MCP Client initialized")
    
    async def connect(self):
        """Connect to MCP server and initialize agent"""
        try:
            logger.info(f"🔌 Connecting to MCP server: {self.mcp_config.server_url}")
            
            # Initialize MCP client with proper configuration
            self.mcp_client = MultiServerMCPClient({
                self.mcp_config.server_name: {
                    "url": self.mcp_config.server_url,
                    "transport": self.mcp_config.transport,
                }
            })
            
            # Enter the client context
            await self.mcp_client.__aenter__()
            
            # Give the server a moment to be ready
            await asyncio.sleep(2)
            
            # Get available tools
            tools = self.mcp_client.get_tools()
            if not tools:
                logger.error("❌ No tools received from MCP server")
                return False
            
            logger.info(f"✅ Connected to MCP server. Available tools: {[tool.name for tool in tools]}")
            
            # Create ReAct agent with Snowflake Cortex LLM
            self.agent = create_react_agent(
                model=self.llm,
                tools=tools
            )
            
            self.is_connected = True
            logger.info("🚀 MCP Client ready!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MCP server: {e}")
            traceback.print_exc()
            return False
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        try:
            if self.mcp_client:
                await self.mcp_client.__aexit__(None, None, None)
                self.is_connected = False
                logger.info("🔌 Disconnected from MCP server")
        except Exception as e:
            logger.error(f"❌ Error during disconnect: {e}")
    
    async def process_command(self, user_input: str) -> str:
        """Process user command and return response"""
        try:
            if not self.is_connected:
                return "❌ Not connected to MCP server. Please check connection."
            
            # Add to chat history
            self.chat_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })
            
            # Check if this is a patient data query
            patient_data = self.patient_extractor.extract_patient_data(user_input)
            
            if patient_data:
                # Validate patient data
                is_valid, errors = patient_data.validate()
                if not is_valid:
                    error_msg = "❌ Invalid patient data:\n" + "\n".join(f"• {error}" for error in errors)
                    self.chat_history.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "timestamp": datetime.now().isoformat()
                    })
                    return error_msg
                
                # Enhance prompt with patient context
                enhanced_prompt = f"""
I have extracted the following patient information from the user's request:

Patient Information:
- Name: {patient_data.first_name} {patient_data.last_name}
- SSN: {patient_data.ssn}
- Date of Birth: {patient_data.date_of_birth}
- Gender: {patient_data.gender}
- Zip Code: {patient_data.zip_code}

Original User Request: {user_input}

Please use the appropriate Milliman API tools to process this request. Available tools:
- get_token: Get authentication token
- medical_submit: Submit medical record request
- mcid_search: Search MCID database 
- get_all_data: Get comprehensive patient data

Choose the most appropriate tool(s) based on the user's request and provide a comprehensive response with the results.
"""
            else:
                # No patient data found - provide guidance or handle general queries
                enhanced_prompt = f"""
User Request: {user_input}

I am a healthcare AI assistant that can help you interact with Milliman medical APIs. 

Available tools:
- get_token: Get authentication token for API access
- medical_submit: Submit medical record request (requires patient info)
- mcid_search: Search MCID database (requires patient info)  
- get_all_data: Get comprehensive patient data (requires patient info)

If you need to search medical records or get patient data, please provide:
- Patient's first and last name
- SSN (9 digits)
- Date of birth (YYYY-MM-DD format)
- Gender (M/F)
- Zip code (5+ digits)

How can I help you with healthcare data today?
"""
            
            # Process with agent
            logger.info("🤖 Processing with MCP agent...")
            response = await self.agent.ainvoke({"messages": [HumanMessage(content=enhanced_prompt)]})
            
            # Extract response content
            if hasattr(response, 'messages') and response.messages:
                assistant_response = response.messages[-1].content
            elif isinstance(response, dict) and 'messages' in response:
                assistant_response = response['messages'][-1].content
            else:
                assistant_response = str(response)
            
            # Add to chat history
            self.chat_history.append({
                "role": "assistant",
                "content": assistant_response,
                "timestamp": datetime.now().isoformat()
            })
            
            return assistant_response
            
        except Exception as e:
            error_msg = f"❌ Error processing command: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            self.chat_history.append({
                "role": "assistant",
                "content": error_msg,
                "timestamp": datetime.now().isoformat()
            })
            
            return error_msg
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get formatted chat history"""
        return self.chat_history.copy()
    
    def clear_chat_history(self):
        """Clear chat history"""
        self.chat_history = []
        logger.info("🗑️ Chat history cleared")

async def main():
    """Main function to run the MCP client"""
    print("🏥 Milliman MCP Client with Snowflake Cortex")
    print("=" * 50)
    
    # Initialize configurations
    cortex_config = SnowflakeCortexConfig()
    mcp_config = MCPConfig()
    
    # Create client
    client = MillimanMCPClient(cortex_config, mcp_config)
    
    try:
        # Connect to MCP server
        print("🔌 Connecting to MCP server...")
        connected = await client.connect()
        
        if not connected:
            print("❌ Failed to connect to MCP server.")
            print("💡 Make sure the FastMCP server is running:")
            print("   python working_fastmcp_server.py")
            return
        
        print("\n✅ Successfully connected to MCP server!")
        print("\n💡 Example commands:")
        print("• 'Get medical data for John Smith, SSN 123456789, DOB 1980-01-15, Male, Zip 12345'")
        print("• 'Search MCID for patient Jane Doe'")
        print("• 'Get authentication token'")
        print("• 'Get all data for patient'")
        print("• 'help' - Show available commands")
        print("• 'history' - Show chat history")
        print("• 'clear' - Clear chat history")
        print("• 'exit' - Quit the client")
        print("\n" + "=" * 50)
        
        # Chat loop
        while True:
            try:
                user_input = input("\n🗣️  You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    client.clear_chat_history()
                    print("🗑️ Chat history cleared!")
                    continue
                elif user_input.lower() == 'history':
                    history = client.get_chat_history()
                    if history:
                        print("\n📜 Chat History:")
                        for msg in history[-10:]:  # Show last 10 messages
                            role = "🗣️  You" if msg['role'] == 'user' else "🤖 Assistant"
                            print(f"{role}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
                    else:
                        print("📭 No chat history available")
                    continue
                elif user_input.lower() == 'help':
                    print("\n🆘 Available Commands:")
                    print("• Patient data requests - Include patient info (name, SSN, DOB, gender, zip)")
                    print("• 'get token' - Get API authentication token")
                    print("• 'medical submit' - Submit medical record request")
                    print("• 'mcid search' - Search MCID database")
                    print("• 'get all data' - Get comprehensive patient data")
                    print("• 'history' - Show recent chat history")
                    print("• 'clear' - Clear chat history")
                    print("• 'exit' - Quit the client")
                    continue
                
                # Process command
                print("🤖 Assistant: ", end="", flush=True)
                response = await client.process_command(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                logger.error(f"Unexpected error in main loop: {e}")
    
    finally:
        # Cleanup
        await client.disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 MCP Client stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
