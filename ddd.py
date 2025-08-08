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
    """Enhanced Snowflake Cortex Chat model with Brave Search integration only"""
   
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
        """Call MCP tool via HTTP API"""
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
                            return f"Tool error: {result.get('error', 'Unknown error')}"
                    else:
                        print(f"❌ MCP tool call failed with status {response.status_code}: {response.text}")
                        return await self._fallback_tool_call(tool_name, arguments)
                       
                except Exception as http_error:
                    print(f"❌ HTTP request to MCP server failed: {http_error}")
                    return await self._fallback_tool_call(tool_name, arguments)
                   
        except Exception as e:
            print(f"❌ MCP tool call error: {e}")
            return f"Error calling tool {tool_name}: {str(e)}"
 
    async def _fallback_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Fallback method to call tools directly"""
        try:
            print(f"🔄 Using fallback for tool: {tool_name}")
           
            if tool_name == "calculator":
                expression = arguments.get("expression", "")
                if expression:
                    try:
                        # Enhanced calculator for compound interest and exponents
                        if "^" in expression:
                            # Replace ^ with ** for Python exponentiation
                            expression = expression.replace("^", "**")
                        
                        allowed_chars = "0123456789+-*/(). *"  # Added * for **
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
                search_type = "web" if tool_name == "brave_web_search" else "local"
                query = arguments.get("query", "")
                return f"🔍 Brave {search_type} search for '{query}' requires MCP server connection. Please check server status and Brave API key configuration."
               
            elif tool_name in ["DFWAnalyst", "DFWSearch"]:
                return f"🏥 HEDIS tool '{tool_name}' requires MCP server connection for Snowflake integration. Please check server status."
               
            else:
                return f"Tool {tool_name} not available in fallback mode. Please check MCP server connection."
               
        except Exception as e:
            return f"Fallback tool call failed: {str(e)}"
 
    def _detect_tool_calls(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Enhanced tool call detection - Brave Search Only"""
        tool_calls = []
       
        for message in messages:
            content = str(message.content)
            content_lower = content.lower()
           
            # Enhanced detection patterns - BRAVE SEARCH ONLY
            tool_patterns = {
                "calculator": [
                    r"use.*calculator.*(?:calculate|expression|compute).*?[:=]\s*([^\n]+)",
                    r"calculate\s*[:=]?\s*([^\n]+)",
                    r"calculator.*tool.*(?:with|expression).*?[:=]\s*([^\n]+)",
                    # Direct calculation requests
                    r"(?:what\s+is|calculate|compute)\s+([0-9+\-*/().\s^]+)(?:\?|$)",
                    # Compound interest detection
                    r"compound.*interest.*\$?(\d+).*(\d+\.?\d*)%.*(\d+).*year",
                ],
                "get_weather": [
                    r"(?:weather|temperature|forecast|conditions?).*in\s+([^?\n.]+)",
                    r"(?:what.*weather|current.*weather|weather.*like).*in\s+([^?\n.]+)",
                    r"(?:get|show|find).*weather.*(?:for|in)\s+([^?\n.]+)",
                    r"weather.*(?:for|in)\s+([^?\n.]+)",
                    r"weather\s+([a-zA-Z\s,]+)",
                    r"current.*weather.*([a-zA-Z\s,]+)"
                ],
                "brave_web_search": [
                    # General web search patterns
                    r"(?:search|find|look.*up).*(?:web|internet|online|news).*(?:for|about)\s+([^?\n.]+)",
                    r"(?:latest|recent|current).*(?:news|information|updates?).*about\s+([^?\n.]+)",
                    r"web.*search.*(?:for|about)\s+([^?\n.]+)",
                    r"(?:find|search).*(?:latest|current|recent).*([^?\n.]+)",
                    # Direct search requests
                    r"search.*for\s+([^?\n.]+)",
                    r"find.*information.*about\s+([^?\n.]+)",
                    r"look.*up\s+([^?\n.]+)",
                    r"what.*is\s+([^?\n.]+)",
                    r"tell.*me.*about\s+([^?\n.]+)",
                    # Latest/current patterns
                    r"latest\s+([^?\n.]+)",
                    r"recent\s+([^?\n.]+)",
                    r"current\s+([^?\n.]+)",
                    # News and developments
                    r"(?:news|developments|updates).*about\s+([^?\n.]+)",
                    r"(?:AI|technology|tech).*(?:news|developments|trends)\s*([^?\n.]*)",
                ],
                "brave_local_search": [
                    # Local business and place patterns
                    r"find.*(?:restaurant|coffee|hotel|gas station|shop|store|pharmacy|hospital).*(?:near|in)\s+([^?\n.]+)",
                    r"(?:restaurant|coffee|hotel|shop|store|pharmacy|hospital).*near\s+([^?\n.]+)",
                    r"(?:pizza|food|italian|chinese|mexican).*(?:restaurant|place).*(?:near|in)\s+([^?\n.]+)",
                    r"local.*search.*(?:for|about)\s+([^?\n.]+)",
                    r"(?:where.*can.*find|where.*is).*(?:restaurant|coffee|hotel|shop|store).*(?:near|in)\s+([^?\n.]+)",
                    # Business type + location patterns
                    r"(?:coffee|pizza|gas|hotel|pharmacy).*(?:near|in)\s+([^?\n.]+)",
                ],
                # HEDIS and other tools
                "DFWAnalyst": [
                    r"DFWAnalyst.*[:=]\s*([^\n]+)",
                    r"(?:convert|translate).*(?:to|into)\s*SQL.*[:=]?\s*([^\n]+)",
                    r"HEDIS.*SQL.*[:=]?\s*([^\n]+)",
                ],
                "DFWSearch": [
                    r"DFWSearch.*[:=]\s*([^\n]+)",
                    r"search.*HEDIS.*(?:for|about)\s+([^\n]+)",
                    r"HEDIS.*(?:measure|specification).*[:=]?\s*([^\n]+)",
                ],
                "test_tool": [
                    r"test.*tool.*[:=]\s*([^\n]+)",
                    r"run.*test.*[:=]\s*([^\n]+)",
                ],
                "diagnostic": [
                    r"diagnostic.*[:=]\s*([^\n]+)",
                    r"run.*diagnostic.*[:=]\s*([^\n]+)",
                ]
            }
           
            # Check each tool pattern
            for tool_name, patterns in tool_patterns.items():
                for pattern in patterns:
                    match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
                    if match:
                        # Special handling for compound interest
                        if tool_name == "calculator" and "compound.*interest" in pattern:
                            # Extract principal, rate, time from the content
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
                        elif tool_name in ["brave_web_search", "brave_local_search"]:
                            arg_name = "query"
                            # Check if this should be local vs web search
                            location_keywords = ["near", "restaurant", "coffee", "hotel", "gas station", "shop", "store", "pharmacy", "hospital"]
                            business_keywords = ["pizza", "food", "italian", "chinese", "mexican"]
                            
                            # If we detected web search but it contains location keywords, switch to local
                            if tool_name == "brave_web_search" and any(keyword in content_lower for keyword in location_keywords + business_keywords):
                                tool_name = "brave_local_search"
                            
                            # Add count parameter for Brave searches
                            count = 5 if tool_name == "brave_web_search" else 3  # Different defaults
                            tool_calls.append({
                                "tool_name": tool_name,
                                "arguments": {arg_name: argument_value, "count": count}
                            })
                            print(f"🎯 Detected tool call: {tool_name}({arg_name}='{argument_value}', count={count})")
                            break  # Found a match for this tool, move to next tool
                        elif tool_name in ["DFWAnalyst", "DFWSearch"]:
                            arg_name = "prompt" if tool_name == "DFWAnalyst" else "query"
                        elif tool_name == "test_tool":
                            arg_name = "message"
                        elif tool_name == "diagnostic":
                            arg_name = "test_type"
                        else:
                            arg_name = "query"
                        
                        # Only add non-Brave tools here (Brave tools were added above)
                        if tool_name not in ["brave_web_search", "brave_local_search"]:
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
            error_content += "- Use tool-based queries (weather, brave search, calculator)\n"
           
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
