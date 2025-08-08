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
    """Robust Snowflake Cortex Chat model with comprehensive tool integration"""
   
    # MCP server configuration
    mcp_server_url: str = Field(default="http://localhost:8081/sse")
    """URL of the MCP server"""
   
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
        """Call MCP tool via HTTP API with robust error handling"""
        try:
            print(f"🔧 Calling MCP tool: {tool_name} with args: {arguments}")
           
            tool_call_data = {
                "tool_name": tool_name,
                "arguments": arguments
            }
           
            async with httpx.AsyncClient(timeout=60.0) as client:
                try:
                    response = await client.post(
                        f"{self.mcp_server_url.rstrip('/sse')}/api/v1/tool_call",
                        json=tool_call_data,
                        headers={"Content-Type": "application/json"}
                    )
                   
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('success'):
                            return str(result.get('result', 'No result returned'))
                        else:
                            error_msg = result.get('error', 'Unknown error')
                            print(f"❌ Tool returned error: {error_msg}")
                            return await self._fallback_tool_call(tool_name, arguments)
                    else:
                        print(f"❌ HTTP {response.status_code}: {response.text}")
                        return await self._fallback_tool_call(tool_name, arguments)
                       
                except httpx.TimeoutException:
                    print(f"⏰ Tool call timed out for {tool_name}")
                    return await self._fallback_tool_call(tool_name, arguments)
                except httpx.ConnectError:
                    print(f"🔌 Connection failed to MCP server for {tool_name}")
                    return await self._fallback_tool_call(tool_name, arguments)
                except Exception as http_error:
                    print(f"❌ HTTP error for {tool_name}: {http_error}")
                    return await self._fallback_tool_call(tool_name, arguments)
                   
        except Exception as e:
            print(f"❌ Unexpected error calling {tool_name}: {e}")
            return await self._fallback_tool_call(tool_name, arguments)
 
    async def _fallback_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Comprehensive fallback for all tools"""
        try:
            print(f"🔄 Using fallback for tool: {tool_name}")
           
            if tool_name == "calculator":
                expression = arguments.get("expression", "")
                if expression:
                    try:
                        # Enhanced calculator for compound interest and exponents
                        if "^" in expression:
                            expression = expression.replace("^", "**")
                        
                        # Allow more characters for advanced math
                        allowed_chars = "0123456789+-*/().* "
                        if all(char in allowed_chars for char in expression):
                            result = eval(expression)
                            return f"✅ **Calculator Result:** {result}\n\n*Calculated locally (MCP server unavailable)*"
                        else:
                            return f"❌ **Calculator Error:** Invalid characters in expression '{expression}'"
                    except Exception as e:
                        return f"❌ **Calculator Error:** {str(e)}"
                else:
                    return "❌ **Calculator Error:** No expression provided"
                       
            elif tool_name == "test_tool":
                message = arguments.get("message", "test")
                from datetime import datetime
                current_time = datetime.now().isoformat()
                return f"✅ **Test Tool Success:** Message '{message}' processed at {current_time}\n\n*Executed locally (MCP server unavailable)*"
               
            elif tool_name == "get_weather":
                place = arguments.get("place", "unknown location")
                return f"🌤️ **Weather Service Unavailable**\n\nSorry, I cannot get weather information for **{place}** right now.\n\n**Possible reasons:**\n- MCP server is not running\n- Weather service is temporarily down\n- Network connectivity issues\n\n**Suggestions:**\n- Check your local weather app\n- Try again in a few minutes\n- Verify MCP server status"
               
            elif tool_name == "brave_web_search":
                query = arguments.get("query", "unknown query")
                return f"🔍 **Web Search Unavailable**\n\nSorry, I cannot search the web for **'{query}'** right now.\n\n**Possible reasons:**\n- MCP server is not running\n- Search service is temporarily down\n- API limits may have been reached\n\n**Suggestions:**\n- Try using a search engine directly\n- Check MCP server status\n- Try a simpler search query later"
               
            elif tool_name == "brave_local_search":
                query = arguments.get("query", "unknown query")
                return f"📍 **Local Search Unavailable**\n\nSorry, I cannot search for local businesses/places for **'{query}'** right now.\n\n**Possible reasons:**\n- MCP server is not running\n- Local search service is temporarily down\n- Location services may be unavailable\n\n**Suggestions:**\n- Try using Google Maps or similar apps\n- Check MCP server status\n- Try again in a few minutes"
               
            elif tool_name in ["DFWAnalyst", "DFWSearch"]:
                action = "convert to SQL" if tool_name == "DFWAnalyst" else "search HEDIS documents"
                content = arguments.get("prompt" if tool_name == "DFWAnalyst" else "query", "")
                return f"🏥 **HEDIS Tool Unavailable**\n\nSorry, I cannot {action} for **'{content}'** right now.\n\n**Possible reasons:**\n- MCP server is not running\n- Snowflake connection issues\n- HEDIS services are temporarily down\n\n**Suggestions:**\n- Check MCP server and Snowflake connectivity\n- Try again in a few minutes\n- Contact system administrator if issue persists"
               
            elif tool_name == "diagnostic":
                test_type = arguments.get("test_type", "basic")
                return f"🔧 **Diagnostic Tool - Local Mode**\n\n**Test Type:** {test_type}\n**Status:** MCP server unavailable\n**Timestamp:** {datetime.now().isoformat()}\n\n**Local System Check:**\n- ✅ LLM wrapper is functioning\n- ❌ MCP server connection failed\n- ⚠️ Running in fallback mode\n\n**Recommendation:** Check MCP server status and network connectivity"
               
            else:
                return f"🛠️ **Tool '{tool_name}' Unavailable**\n\nThis tool requires MCP server connection which is currently unavailable.\n\n**Suggestions:**\n- Check if MCP server is running\n- Verify network connectivity\n- Try again in a few minutes\n- Contact system administrator if issue persists"
               
        except Exception as e:
            return f"❌ **Fallback Error:** Unable to execute {tool_name} - {str(e)}"
 
    def _detect_tool_calls(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Comprehensive tool call detection with smart routing"""
        tool_calls = []
       
        for message in messages:
            content = str(message.content)
            content_lower = content.lower()
           
            # Comprehensive tool patterns
            tool_patterns = {
                "calculator": [
                    r"(?:calculate|compute|what\s+is)\s*[:=]?\s*([0-9+\-*/().\s^]+)(?:\?|$)",
                    r"calculator.*(?:expression|calculate|compute).*?[:=]\s*([^\n]+)",
                    r"compound.*interest.*\$?(\d+).*(\d+\.?\d*)%.*(\d+).*year",
                    r"math.*(?:problem|calculation|expression).*?[:=]\s*([^\n]+)",
                ],
                "get_weather": [
                    r"(?:weather|temperature|forecast|climate|conditions?).*(?:in|for|at)\s+([^?\n.]+)",
                    r"(?:what.*weather|current.*weather|weather.*like).*(?:in|for|at)\s+([^?\n.]+)",
                    r"(?:get|show|find|check).*weather.*(?:for|in|at)\s+([^?\n.]+)",
                    r"weather\s+([a-zA-Z\s,]+)(?:\?|$)",
                    r"(?:temperature|forecast).*(?:in|for|at)\s+([^?\n.]+)",
                ],
                "brave_web_search": [
                    # General search patterns
                    r"(?:search|find|look.*up).*(?:for|about)\s+([^?\n.]+)",
                    r"(?:latest|recent|current|new).*(?:news|information|updates?|developments?).*(?:about|on)\s+([^?\n.]+)",
                    r"(?:what|who|when|where|how).*is\s+([^?\n.]+)",
                    r"tell.*me.*about\s+([^?\n.]+)",
                    r"information.*(?:about|on)\s+([^?\n.]+)",
                    # News and trends
                    r"(?:AI|technology|tech|science).*(?:news|developments?|trends?|updates?)\s*([^?\n.]*)",
                    r"latest\s+([^?\n.]+)",
                    r"recent\s+([^?\n.]+)",
                    r"current\s+([^?\n.]+)",
                    # Direct search requests
                    r"search.*web.*(?:for|about)\s+([^?\n.]+)",
                    r"google.*(?:search|for)\s+([^?\n.]+)",
                    r"web.*search.*(?:for|about)\s+([^?\n.]+)",
                ],
                "brave_local_search": [
                    # Local business patterns
                    r"find.*(?:restaurant|coffee|hotel|gas.*station|shop|store|pharmacy|hospital|bank|atm).*(?:near|in|around)\s+([^?\n.]+)",
                    r"(?:restaurant|coffee|hotel|shop|store|pharmacy|hospital|bank).*(?:near|in|around)\s+([^?\n.]+)",
                    r"(?:pizza|food|italian|chinese|mexican|sushi|thai).*(?:restaurant|place).*(?:near|in|around)\s+([^?\n.]+)",
                    r"(?:where.*can.*find|where.*is.*nearest).*(?:restaurant|coffee|hotel|shop|store|gas.*station).*(?:near|in|around)\s+([^?\n.]+)",
                    r"local.*(?:business|restaurant|shop|store).*(?:near|in|around)\s+([^?\n.]+)",
                    # Location-based queries
                    r"(?:coffee|pizza|gas|hotel|pharmacy|hospital).*(?:near|in|around)\s+([^?\n.]+)",
                    r"best.*(?:restaurant|coffee|hotel).*(?:near|in|around)\s+([^?\n.]+)",
                ],
                # HEDIS and specialized tools
                "DFWAnalyst": [
                    r"DFWAnalyst.*[:=]\s*([^\n]+)",
                    r"(?:convert|translate|transform).*(?:to|into)\s*SQL.*[:=]?\s*([^\n]+)",
                    r"HEDIS.*(?:SQL|query|analyst).*[:=]?\s*([^\n]+)",
                    r"text.*to.*SQL.*[:=]?\s*([^\n]+)",
                ],
                "DFWSearch": [
                    r"DFWSearch.*[:=]\s*([^\n]+)",
                    r"search.*HEDIS.*(?:for|about|documents?)\s+([^\n]+)",
                    r"HEDIS.*(?:measure|specification|document).*[:=]?\s*([^\n]+)",
                    r"find.*HEDIS.*(?:information|document).*[:=]?\s*([^\n]+)",
                ],
                "test_tool": [
                    r"test.*tool.*[:=]\s*([^\n]+)",
                    r"run.*test.*[:=]\s*([^\n]+)",
                    r"test.*(?:message|function).*[:=]\s*([^\n]+)",
                ],
                "diagnostic": [
                    r"diagnostic.*[:=]\s*([^\n]+)",
                    r"run.*diagnostic.*[:=]\s*([^\n]+)",
                    r"system.*(?:check|diagnostic|test).*[:=]?\s*([^\n]*)",
                    r"check.*(?:system|server|connection).*[:=]?\s*([^\n]*)",
                ]
            }
           
            # Process each tool pattern
            for tool_name, patterns in tool_patterns.items():
                for pattern in patterns:
                    match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
                    if match:
                        # Special handling for compound interest
                        if tool_name == "calculator" and "compound.*interest" in pattern:
                            principal_match = re.search(r'\$?(\d+)', content)
                            rate_match = re.search(r'(\d+\.?\d*)%', content)
                            time_match = re.search(r'(\d+)\s*year', content)
                            
                            if principal_match and rate_match and time_match:
                                principal = principal_match.group(1)
                                rate = rate_match.group(1)
                                time = time_match.group(1)
                                argument_value = f"{principal} * (1 + {rate}/100)^{time}"
                            else:
                                continue
                        else:
                            argument_value = match.group(1).strip().strip('"\'.,!?')
                       
                        # Skip very short or empty arguments
                        if len(argument_value) < 1:
                            continue
                           
                        # Clean up and validate arguments
                        if tool_name == "calculator":
                            arg_name = "expression"
                        elif tool_name == "get_weather":
                            arg_name = "place"
                            # Clean up location names
                            argument_value = re.sub(r'^(?:the\s+)?', '', argument_value, flags=re.IGNORECASE)
                            argument_value = re.sub(r'\s*[?!.]*$', '', argument_value)
                            if len(argument_value) < 2:
                                continue
                        elif tool_name in ["brave_web_search", "brave_local_search"]:
                            arg_name = "query"
                            
                            # Smart routing between web and local search
                            location_indicators = ["near", "in", "around", "restaurant", "coffee", "hotel", "shop", "store", "gas station", "pharmacy", "hospital", "pizza", "food"]
                            
                            # If web search but has location indicators, switch to local
                            if tool_name == "brave_web_search" and any(indicator in content_lower for indicator in location_indicators):
                                tool_name = "brave_local_search"
                            
                            # Add count parameter
                            count = 5 if tool_name == "brave_web_search" else 3
                            tool_calls.append({
                                "tool_name": tool_name,
                                "arguments": {arg_name: argument_value, "count": count}
                            })
                            print(f"🎯 Detected: {tool_name}(query='{argument_value}', count={count})")
                            break
                        elif tool_name in ["DFWAnalyst", "DFWSearch"]:
                            arg_name = "prompt" if tool_name == "DFWAnalyst" else "query"
                        elif tool_name == "test_tool":
                            arg_name = "message"
                        elif tool_name == "diagnostic":
                            arg_name = "test_type"
                            if not argument_value or argument_value.isspace():
                                argument_value = "basic"
                        else:
                            arg_name = "query"
                        
                        # Add non-search tools
                        if tool_name not in ["brave_web_search", "brave_local_search"]:
                            tool_calls.append({
                                "tool_name": tool_name,
                                "arguments": {arg_name: argument_value}
                            })
                            print(f"🎯 Detected: {tool_name}({arg_name}='{argument_value}')")
                        
                        break
               
                if tool_calls:
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
                    # Robust async execution
                    loop = None
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        pass
                   
                    if loop is not None:
                        import concurrent.futures
                       
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
                        result = asyncio.run(
                            self._call_mcp_tool(tool_call["tool_name"], tool_call["arguments"])
                        )
                   
                    tool_results.append({
                        "tool_name": tool_call["tool_name"],
                        "result": result
                    })
                    print(f"✅ Tool {tool_call['tool_name']} completed")
                   
                except Exception as e:
                    print(f"❌ Tool {tool_call['tool_name']} failed: {e}")
                    # Still try fallback
                    try:
                        fallback_result = asyncio.run(
                            self._fallback_tool_call(tool_call["tool_name"], tool_call["arguments"])
                        )
                        tool_results.append({
                            "tool_name": tool_call["tool_name"],
                            "result": fallback_result
                        })
                    except Exception as fallback_error:
                        tool_results.append({
                            "tool_name": tool_call["tool_name"],
                            "result": f"❌ Tool completely failed: {str(e)} | Fallback error: {str(fallback_error)}"
                        })
 
        # Return tool results
        if tool_results:
            content = "🔧 **Tool Execution Results:**\n\n"
            for tool_result in tool_results:
                content += f"**{tool_result['tool_name']}:**\n{tool_result['result']}\n\n"
           
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
 
        # No tools called - use Snowflake or provide helpful message
        try:
            if not self.session:
                helpful_msg = """❌ **No Tools Available & No Snowflake Session**

I couldn't detect any tool requests in your message, and there's no Snowflake session available for general AI responses.

**Available Tools:**
- 🧮 **Calculator:** "Calculate 10 + 5" or "What is 2^8?"
- 🌤️ **Weather:** "Weather in New York" or "Current temperature in London"
- 🔍 **Web Search:** "Latest AI news" or "Recent technology trends"
- 📍 **Local Search:** "Pizza restaurants near Times Square"
- 🏥 **HEDIS Tools:** "DFWAnalyst: convert to SQL" or "DFWSearch: BCS measure"
- 🔧 **Test/Diagnostic:** "Run diagnostic test" or "Test tool with message"

**Try asking:** "What's the weather in Paris?" or "Calculate compound interest on $1000 at 5% for 3 years"
"""
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=helpful_msg))])
 
            # Use Snowflake Cortex
            message_dicts = [_convert_message_to_dict(m) for m in messages]
            message_json = _safe_json_for_sql(message_dicts)
 
            options = {
                "temperature": self.temperature,
                "top_p": self.top_p if self.top_p is not None else 1.0,
                "max_tokens": self.max_tokens if self.max_tokens is not None else 2048,
            }
            options_json = _safe_json_for_sql(options)
 
            sql_stmt = f"""
                select snowflake.cortex.{self.cortex_function}(
                    '{self.model}',
                    parse_json($${message_json}$$),
                    parse_json($${options_json}$$)
                ) as llm_stream_response;
            """
 
            print(f"🗃️ Executing SQL query...")
           
            self.session.sql(f"USE WAREHOUSE {self.session.get_current_warehouse()};").collect()
            l_rows = self.session.sql(sql_stmt).collect()
           
            response = json.loads(l_rows[0]["LLM_STREAM_RESPONSE"])
            ai_message_content = response["choices"][0]["messages"]
           
            content = _truncate_at_stop_tokens(ai_message_content, stop)
           
            message = AIMessage(content=content, response_metadata=response.get("usage", {}))
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
               
        except Exception as e:
            print(f"❌ Snowflake Cortex error: {e}")
            error_content = f"""❌ **Snowflake Cortex Error**

{str(e)}

**Suggestions:**
- Check Snowflake connection and permissions
- Try using specific tools instead:
  - Calculator: "Calculate 25 * 4"
  - Weather: "Weather in Boston"  
  - Search: "Latest news about AI"
- Verify model name and warehouse settings
"""
            message = AIMessage(content=error_content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
 
    def _stream_content(self, content: str, stop: Optional[List[str]]) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model in chunks."""
        chunk_size = 50
        truncated_content = _truncate_at_stop_tokens(content, stop)
 
        for i in range(0, len(truncated_content), chunk_size):
            chunk_content = truncated_content[i : i + chunk_size]
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk_content))
 
    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model."""
        try:
            result = self._generate(messages, stop, run_manager, **kwargs)
            content = result.generations[0].message.content
           
            for chunk in self._stream_content(content, stop):
                yield chunk
               
        except Exception as e:
            error_content = f"Streaming error: {str(e)}"
            yield ChatGenerationChunk(message=AIMessageChunk(content=error_content))
