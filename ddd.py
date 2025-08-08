import json
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)
import asyncio
import httpx
import re
import base64
 
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    get_pydantic_field_names,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.utils import _build_model_kwargs
from pydantic import Field, SecretStr, model_validator
 
SUPPORTED_ROLES: List[str] = [
    "system",
    "user",
    "assistant",
]
 
class ChatSnowflakeCortexError(Exception):
    """Error with Snowpark client."""
 
def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary."""
    message_dict: Dict[str, Any] = {
        "content": message.content,
    }
 
    # Populate role and additional message data
    if isinstance(message, ChatMessage) and message.role in SUPPORTED_ROLES:
        message_dict["role"] = message.role
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict
 
def _truncate_at_stop_tokens(text: str, stop: Optional[List[str]]) -> str:
    """Truncates text at the earliest stop token found."""
    if stop is None:
        return text
 
    for stop_token in stop:
        stop_token_idx = text.find(stop_token)
        if stop_token_idx != -1:
            text = text[:stop_token_idx]
    return text
 
def _escape_for_sql(text: str) -> str:
    """Properly escape text for SQL string literals."""
    # Replace single quotes with double single quotes (SQL standard)
    escaped = text.replace("'", "''")
    # Replace backslashes
    escaped = escaped.replace("\\", "\\\\")
    return escaped
 
def _safe_json_for_sql(data: Any) -> str:
    """Create a JSON string that's safe for SQL embedding."""
    json_str = json.dumps(data, ensure_ascii=True)
    # Escape for SQL
    escaped = _escape_for_sql(json_str)
    return escaped
 
class ChatSnowflakeCortex(BaseChatModel):
    """Enhanced Snowflake Cortex Chat model with MCP tool integration and Brave Search"""
   
    # MCP server configuration
    mcp_server_url: str = Field(default="http://localhost:8081/sse")
    """URL of the MCP server"""
    
    # Brave Search API key
    brave_api_key: str = Field(default="BSAQIFoBulbULfcL6RMBxRWCtopFY0E")
    """Brave Search API key"""
   
    test_tools: Dict[str, Union[Dict[str, Any], Type, Callable, BaseTool]] = Field(
        default_factory=dict
    )
 
    session: Any = None
    """Snowpark session object."""
 
    model: str = "mistral-large"
    """Snowflake cortex hosted LLM model name, defaulted to `mistral-large`."""
 
    cortex_function: str = "complete"
    """Cortex function to use, defaulted to `complete`."""
 
    temperature: float = 0
    """Model temperature. Value should be >= 0 and <= 1.0"""
 
    max_tokens: Optional[int] = None
    """The maximum number of output tokens in the response."""
 
    top_p: Optional[float] = 0
    """top_p adjusts the number of choices for each predicted tokens based on
        cumulative probabilities. Value should be ranging between 0.0 and 1.0.
    """
 
    snowflake_username: Optional[str] = Field(default=None, alias="username")
    """Automatically inferred from env var `SNOWFLAKE_USERNAME` if not provided."""
    snowflake_password: Optional[SecretStr] = Field(default=None, alias="password")
    """Automatically inferred from env var `SNOWFLAKE_PASSWORD` if not provided."""
    snowflake_account: Optional[str] = Field(default=None, alias="account")
    """Automatically inferred from env var `SNOWFLAKE_ACCOUNT` if not provided."""
    snowflake_database: Optional[str] = Field(default=None, alias="database")
    """Automatically inferred from env var `SNOWFLAKE_DATABASE` if not provided."""
    snowflake_schema: Optional[str] = Field(default=None, alias="schema")
    """Automatically inferred from env var `SNOWFLAKE_SCHEMA` if not provided."""
    snowflake_warehouse: Optional[str] = Field(default=None, alias="warehouse")
    """Automatically inferred from env var `SNOWFLAKE_WAREHOUSE` if not provided."""
    snowflake_role: Optional[str] = Field(default=None, alias="role")
    """Automatically inferred from env var `SNOWFLAKE_ROLE` if not provided."""
 
    def __init__(self, **kwargs):
        """Initialize the ChatSnowflakeCortex model."""
        super().__init__(**kwargs)
        print(f"🔑 Brave API key loaded: {self.brave_api_key[:8]}...")
    
    def _configure_brave_api_key_sync(self):
        """Synchronously configure Brave API key."""
        try:
            print(f"🔑 Configuring Brave API key...")
            
            import requests
            
            # Try multiple possible endpoints
            endpoints_to_try = [
                f"{self.mcp_server_url.rstrip('/sse')}/configure_brave_key",
                f"{self.mcp_server_url.rstrip('/sse')}/api/v1/configure_brave_key",
                f"{self.mcp_server_url.replace('/sse', '')}/configure_brave_key"
            ]
            
            for endpoint in endpoints_to_try:
                try:
                    print(f"🔄 Trying endpoint: {endpoint}")
                    response = requests.post(
                        endpoint,
                        json={"api_key": self.brave_api_key},
                        headers={"Content-Type": "application/json"},
                        timeout=10
                    )
                    
                    print(f"📡 Response status: {response.status_code}")
                    print(f"📄 Response text: {response.text}")
                    
                    if response.status_code == 200:
                        print("✅ Brave API key configured successfully")
                        return True
                    else:
                        print(f"⚠️ Endpoint {endpoint} returned status: {response.status_code}")
                        continue
                        
                except requests.exceptions.RequestException as e:
                    print(f"❌ Request failed for {endpoint}: {e}")
                    continue
            
            print("❌ All configuration endpoints failed")
            return False
                    
        except Exception as e:
            print(f"❌ Could not configure Brave API key: {e}")
            return False
 
    def test_brave_configuration(self):
        """Test Brave API configuration with detailed debugging."""
        print("🧪 Testing Brave API Configuration...")
        print(f"🔑 API Key: {self.brave_api_key[:8]}...{self.brave_api_key[-4:]}")
        print(f"🌐 MCP Server URL: {self.mcp_server_url}")
        
        # Test server connectivity first
        import requests
        base_url = self.mcp_server_url.replace('/sse', '')
        
        print(f"\n🔗 Testing server connectivity to: {base_url}")
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            print(f"✅ Server health check: {response.status_code}")
        except Exception as e:
            print(f"❌ Server health check failed: {e}")
            return False
        
        # Test configuration
        print(f"\n🔧 Testing Brave API configuration...")
        return self._configure_brave_api_key_sync()
    
    def get_debug_info(self):
        """Get debugging information for troubleshooting."""
        return {
            "mcp_server_url": self.mcp_server_url,
            "brave_api_key_prefix": self.brave_api_key[:8] + "...",
            "brave_api_key_length": len(self.brave_api_key),
            "possible_endpoints": [
                f"{self.mcp_server_url.rstrip('/sse')}/configure_brave_key",
                f"{self.mcp_server_url.rstrip('/sse')}/api/v1/configure_brave_key",
                f"{self.mcp_server_url.replace('/sse', '')}/configure_brave_key"
            ]
        }

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]
        ] = "auto",
        **kwargs: Any,
    ) -> "ChatSnowflakeCortex":
        """Bind tool-like objects to this chat model."""
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        formatted_tools_dict = {
            tool["name"]: tool for tool in formatted_tools if "name" in tool
        }
        self.test_tools.update(formatted_tools_dict)
 
        print(f"🔧 Tools bound to chat model: {len(formatted_tools_dict)} tools")
        for tool_name in formatted_tools_dict.keys():
            print(f"- {tool_name}")
        return self
 
    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        values = _build_model_kwargs(values, all_required_field_names)
        return values
 
    def __del__(self) -> None:
        if getattr(self, "session", None) is not None:
            self.session.close()
 
    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return f"snowflake-cortex-{self.model}"
 
    async def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call MCP tool via HTTP API with Brave Search support"""
        try:
            print(f"🔧 Calling MCP tool: {tool_name} with args: {arguments}")
            
            # Ensure Brave API key is configured for Brave search tools
            if tool_name in ["brave_web_search", "brave_local_search"]:
                print(f"🔍 Brave search tool detected, ensuring configuration...")
                await self._ensure_brave_configured()
           
            tool_call_data = {
                "tool_name": tool_name,
                "arguments": arguments
            }
           
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Try multiple possible endpoints
                endpoints_to_try = [
                    f"{self.mcp_server_url.rstrip('/sse')}/tool_call",
                    f"{self.mcp_server_url.rstrip('/sse')}/api/v1/tool_call",
                    f"{self.mcp_server_url.replace('/sse', '')}/tool_call"
                ]
                
                for endpoint in endpoints_to_try:
                    try:
                        print(f"🔄 Trying tool endpoint: {endpoint}")
                        response = await client.post(
                            endpoint,
                            json=tool_call_data,
                            headers={"Content-Type": "application/json"}
                        )
                        
                        print(f"📡 Tool call response status: {response.status_code}")
                       
                        if response.status_code == 200:
                            result = response.json()
                            print(f"📄 Tool call response: {result}")
                            if result.get('success'):
                                return str(result.get('result', 'No result returned'))
                            else:
                                return f"Tool error: {result.get('error', 'Unknown error')}"
                        else:
                            print(f"❌ Tool endpoint {endpoint} failed with status {response.status_code}: {response.text}")
                            continue
                            
                    except Exception as http_error:
                        print(f"❌ HTTP request failed for {endpoint}: {http_error}")
                        continue
                
                # If all endpoints failed, try fallback
                print("❌ All tool call endpoints failed, using fallback")
                return await self._fallback_tool_call(tool_name, arguments)
                   
        except Exception as e:
            print(f"❌ MCP tool call error: {e}")
            return f"Error calling tool {tool_name}: {str(e)}"
    
    async def _ensure_brave_configured(self):
        """Ensure Brave API key is configured before making Brave search calls."""
        try:
            print(f"🔍 Verifying Brave API configuration...")
            
            # Try synchronous configuration first
            if self._configure_brave_api_key_sync():
                return True
            
            # If sync failed, try async
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Try multiple endpoints
                endpoints_to_try = [
                    f"{self.mcp_server_url.rstrip('/sse')}/configure_brave_key",
                    f"{self.mcp_server_url.rstrip('/sse')}/api/v1/configure_brave_key",
                    f"{self.mcp_server_url.replace('/sse', '')}/configure_brave_key"
                ]
                
                for endpoint in endpoints_to_try:
                    try:
                        print(f"🔄 Async trying endpoint: {endpoint}")
                        response = await client.post(
                            endpoint,
                            json={"api_key": self.brave_api_key},
                            headers={"Content-Type": "application/json"}
                        )
                        
                        print(f"📡 Async response status: {response.status_code}")
                        
                        if response.status_code == 200:
                            print("✅ Brave API key configured successfully (async)")
                            return True
                        else:
                            print(f"⚠️ Async endpoint {endpoint} returned: {response.status_code}")
                            continue
                            
                    except Exception as e:
                        print(f"❌ Async request failed for {endpoint}: {e}")
                        continue
                
                print("❌ All async configuration endpoints failed")
                return False
                
        except Exception as e:
            print(f"⚠️ Could not verify Brave configuration: {e}")
            return False
 
    async def _fallback_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Fallback method to call tools directly with Brave Search support"""
        try:
            print(f"🔄 Using fallback for tool: {tool_name}")
           
            if tool_name == "calculator":
                expression = arguments.get("expression", "")
                if expression:
                    try:
                        allowed_chars = "0123456789+-*/(). "
                        if all(char in allowed_chars for char in expression):
                            result = eval(expression)
                            return f"Result: {result}"
                        else:
                            return "Invalid characters in expression."
                    except Exception as e:
                        return f"Error: {str(e)}"
                       
            elif tool_name == "test_tool":
                message = arguments.get("message", "test")
                from datetime import datetime
                current_time = datetime.now().isoformat()
                return f"✅ SUCCESS: Test tool called with message '{message}' at {current_time}"
               
            elif tool_name == "get_weather":
                place = arguments.get("place", "")
                return f"🌤️ Weather service unavailable in fallback mode. Please check MCP server for location: {place}"
            
            elif tool_name in ["brave_web_search", "brave_local_search"]:
                query = arguments.get("query", "")
                return f"🔍 Brave Search service unavailable in fallback mode. Please check MCP server connection for query: {query}"
               
            else:
                return f"Tool {tool_name} not available in fallback mode. Please check MCP server connection."
               
        except Exception as e:
            return f"Fallback tool call failed: {str(e)}"
 
    def _detect_tool_calls(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Enhanced tool call detection from messages with Brave Search"""
        tool_calls = []
       
        for message in messages:
            content = str(message.content)
            content_lower = content.lower()
           
            # Enhanced detection patterns with Brave Search
            tool_patterns = {
                "calculator": [
                    r"use.*calculator.*(?:calculate|expression|compute).*?[:=]\s*([^\n]+)",
                    r"calculate\s*[:=]\s*([^\n]+)",
                    r"calculator.*tool.*(?:with|expression).*?[:=]\s*([^\n]+)",
                    # Direct calculation requests
                    r"(?:what\s+is|calculate|compute)\s+([0-9+\-*/().\s]+)(?:\?|$)",
                    # Weather queries that need calculation
                    r"weather.*(?:calculate|temperature|convert)"
                ],
                "get_weather": [
                    r"(?:weather|temperature|forecast|conditions?).*in\s+([^?\n.]+)",
                    r"(?:what.*weather|current.*weather|weather.*like).*in\s+([^?\n.]+)",
                    r"(?:get|show|find).*weather.*(?:for|in)\s+([^?\n.]+)",
                    r"weather.*(?:for|in)\s+([^?\n.]+)",
                    # Simple weather requests
                    r"weather\s+([a-zA-Z\s,]+)",
                    # Current weather patterns
                    r"current.*weather.*([a-zA-Z\s,]+)"
                ],
                "wikipedia_search": [
                    r"(?:search|find|look.*up).*wikipedia.*(?:for|about)\s+([^?\n.]+)",
                    r"wikipedia.*(?:search|information|article).*(?:for|about|on)\s+([^?\n.]+)",
                    r"(?:what.*according.*wikipedia|wikipedia.*says).*about\s+([^?\n.]+)"
                ],
                "brave_web_search": [
                    # General web search patterns
                    r"(?:search|find|look.*up).*(?:web|internet|online|news).*(?:for|about)\s+([^?\n.]+)",
                    r"(?:latest|recent|current).*(?:news|information|updates?).*about\s+([^?\n.]+)",
                    r"web.*search.*(?:for|about)\s+([^?\n.]+)",
                    r"(?:find|search).*(?:latest|current|recent).*([^?\n.]+)",
                    # Brave specific patterns
                    r"brave.*search.*(?:for|about)\s+([^?\n.]+)",
                    r"(?:search|find).*brave.*([^?\n.]+)",
                    # General search that should use Brave
                    r"(?:search|google|find).*(?:for|about)\s+([^?\n.]+)",
                    r"(?:what.*is|tell.*me.*about|information.*about)\s+([^?\n.]+)(?:\s+(?:news|latest|recent|today))?",
                    # News and current events
                    r"(?:news|latest|breaking|current).*(?:about|on)\s+([^?\n.]+)",
                    r"(?:what.*happening|what.*new).*(?:with|about)\s+([^?\n.]+)"
                ],
                "brave_local_search": [
                    # Local business search patterns
                    r"(?:find|search|locate).*(?:restaurants?|food|dining).*(?:near|in)\s+([^?\n.]+)",
                    r"(?:restaurants?|cafes?|bars?).*(?:near|in)\s+([^?\n.]+)",
                    r"(?:find|search).*(?:local|nearby).*(?:business|services?).*(?:in|near)\s+([^?\n.]+)",
                    r"(?:pizza|coffee|gas|pharmacy|hotel).*(?:near|in)\s+([^?\n.]+)",
                    r"(?:where.*can.*find|find.*nearby).*([^?\n.]+)",
                    # Generic local search
                    r"(?:local|nearby).*search.*(?:for|about)\s+([^?\n.]+)",
                    r"(?:businesses?|places?).*(?:near|in)\s+([^?\n.]+)"
                ]
            }
           
            # Check each tool pattern
            for tool_name, patterns in tool_patterns.items():
                for pattern in patterns:
                    match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
                    if match:
                        argument_value = match.group(1).strip().strip('"\'.,!?')
                       
                        # Skip if argument is too short or generic
                        if len(argument_value) < 2:
                            continue
                           
                        # Determine the argument name based on tool
                        if tool_name == "calculator":
                            arg_name = "expression"
                            # For weather queries that need calculation, redirect to weather tool
                            if any(w in content_lower for w in ["weather", "temperature", "forecast"]):
                                continue  # Let weather pattern handle it
                        elif tool_name == "get_weather":
                            arg_name = "place"
                            # Clean up location names
                            argument_value = re.sub(r'^(?:the\s+)?', '', argument_value, flags=re.IGNORECASE)
                            argument_value = re.sub(r'\s*\?.*$', '', argument_value)
                        elif tool_name in ["wikipedia_search", "brave_web_search"]:
                            arg_name = "query"
                        elif tool_name == "brave_local_search":
                            arg_name = "query"
                            # Add additional arguments for local search
                            tool_calls.append({
                                "tool_name": tool_name,
                                "arguments": {
                                    arg_name: argument_value,
                                    "count": 5  # Default count for local search
                                }
                            })
                            print(f"🎯 Detected local search: {tool_name}({arg_name}='{argument_value}')")
                            break
                        else:
                            arg_name = "query"
                       
                        if tool_name != "brave_local_search":  # Already handled above
                            # Add additional arguments for web search
                            if tool_name == "brave_web_search":
                                tool_calls.append({
                                    "tool_name": tool_name,
                                    "arguments": {
                                        arg_name: argument_value,
                                        "count": 10,  # Default count for web search
                                        "offset": 0
                                    }
                                })
                            else:
                                tool_calls.append({
                                    "tool_name": tool_name,
                                    "arguments": {arg_name: argument_value}
                                })
                            print(f"🎯 Detected tool call: {tool_name}({arg_name}='{argument_value}')")
                        break  # Found a match for this tool, move to next tool
               
                if tool_calls:  # If we found a tool call, we can break early
                    break
       
        return tool_calls
 
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        print(f"🚀 Starting generation with {len(messages)} messages")
       
        # Detect tool calls in messages
        tool_calls = self._detect_tool_calls(messages)
       
        # Execute tool calls if detected
        tool_results = []
        if tool_calls:
            print(f"🔧 Detected {len(tool_calls)} tool calls")
            for tool_call in tool_calls:
                try:
                    # Use asyncio to call async tool function
                    loop = None
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        pass
                   
                    if loop is not None:
                        # Create a new event loop in a thread if one is already running
                        import concurrent.futures
                        import threading
                       
                        def run_in_thread():
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                return new_loop.run_until_complete(
                                    self._call_mcp_tool(tool_call["tool_name"], tool_call["arguments"])
                                )
                            finally:
                                new_loop.close()
                       
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(run_in_thread)
                            result = future.result(timeout=60)
                    else:
                        # No running loop, we can use asyncio.run
                        result = asyncio.run(
                            self._call_mcp_tool(tool_call["tool_name"], tool_call["arguments"])
                        )
                   
                    tool_results.append({
                        "tool_name": tool_call["tool_name"],
                        "result": result
                    })
                    print(f"✅ Tool {tool_call['tool_name']} executed successfully")
                   
                except Exception as e:
                    print(f"❌ Tool {tool_call['tool_name']} failed: {e}")
                    tool_results.append({
                        "tool_name": tool_call["tool_name"],
                        "result": f"Tool execution failed: {str(e)}"
                    })
 
        # If we have tool results, return them directly without Snowflake
        if tool_results:
            content = "🔧 **Tool Execution Results:**\n\n"
            for tool_result in tool_results:
                content += f"**{tool_result['tool_name']}**:\n{tool_result['result']}\n\n"
           
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
 
        # If no tools were called, proceed with Snowflake Cortex
        try:
            if not self.session:
                return ChatResult(generations=[ChatGeneration(
                    message=AIMessage(content="❌ No Snowflake session available and no tools were called")
                )])
 
            # Prepare messages for Snowflake with better JSON handling
            message_dicts = [_convert_message_to_dict(m) for m in messages]
           
            # Use safe JSON encoding for SQL
            message_json = _safe_json_for_sql(message_dicts)
 
            options = {
                "temperature": self.temperature,
                "top_p": self.top_p if self.top_p is not None else 1.0,
                "max_tokens": self.max_tokens if self.max_tokens is not None else 2048,
            }
            options_json = _safe_json_for_sql(options)
 
            # Use $$ delimiter for complex JSON to avoid escaping issues
            sql_stmt = f"""
                select snowflake.cortex.{self.cortex_function}(
                    '{self.model}',
                    parse_json($${message_json}$$),
                    parse_json($${options_json}$$)
                ) as llm_stream_response;
            """
 
            print(f"🗃️ Executing SQL query...")
           
            # Use the Snowflake Cortex Complete function
            self.session.sql(
                f"USE WAREHOUSE {self.session.get_current_warehouse()};"
            ).collect()
            l_rows = self.session.sql(sql_stmt).collect()
           
            response = json.loads(l_rows[0]["LLM_STREAM_RESPONSE"])
            ai_message_content = response["choices"][0]["messages"]
           
            content = _truncate_at_stop_tokens(ai_message_content, stop)
           
            message = AIMessage(
                content=content,
                response_metadata=response.get("usage", {}),
            )
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
               
        except Exception as e:
            print(f"❌ Snowflake Cortex error: {e}")
           
            # Provide helpful error message
            error_content = f"❌ **Snowflake Cortex Error**: {str(e)}\n\n"
            error_content += "💡 **Possible solutions**:\n"
            error_content += "- Check your Snowflake connection and permissions\n"
            error_content += "- Verify the model name is correct\n"
            error_content += "- Try a simpler query\n"
            error_content += "- Use tool-based queries (weather, Brave search, calculator)\n"
           
            message = AIMessage(content=error_content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
 
    def _stream_content(
        self, content: str, stop: Optional[List[str]]
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model in chunks."""
        chunk_size = 50
        truncated_content = _truncate_at_stop_tokens(content, stop)
 
        for i in range(0, len(truncated_content), chunk_size):
            chunk_content = truncated_content[i : i + chunk_size]
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk_content))
 
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model."""
        try:
            result = self._generate(messages, stop, run_manager, **kwargs)
            content = result.generations[0].message.content
           
            for chunk in self._stream_content(content, stop):
                yield chunk
               
        except Exception as e:
            error_content = f"Streaming error: {str(e)}"
            yield ChatGenerationChunk(message=AIMessageChunk(content=error_content))
