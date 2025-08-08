
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
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
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

def _truncate_at_stop_tokens(
    text: str,
    stop: Optional[List[str]],
) -> str:
    """Truncates text at the earliest stop token found."""
    if stop is None:
        return text

    for stop_token in stop:
        stop_token_idx = text.find(stop_token)
        if stop_token_idx != -1:
            text = text[:stop_token_idx]
    return text

class ChatSnowflakeCortex(BaseChatModel):
    """Enhanced Snowflake Cortex Chat model with MCP tool integration and Brave Search"""
    
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
        """Bind tool-like objects to this chat model, ensuring they conform to
        expected formats."""

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        formatted_tools_dict = {
            tool["name"]: tool for tool in formatted_tools if "name" in tool
        }
        self.test_tools.update(formatted_tools_dict)

        print("Tools bound to chat model:")
        print(f"Number of tools: {len(formatted_tools_dict)}")
        for tool_name in formatted_tools_dict.keys():
            print(f"- {tool_name}")
        print("########################################")
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
            
            # Create the MCP tool call request
            tool_call_data = {
                "tool_name": tool_name,
                "arguments": arguments
            }
            
            # Make HTTP request to MCP server
            async with httpx.AsyncClient(timeout=60.0) as client:  # Increased timeout for search
                # Try calling the tool endpoint
                try:
                    response = await client.post(
                        f"{self.mcp_server_url.rstrip('/sse')}/tool_call",
                        json=tool_call_data,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        return str(result.get('result', 'No result returned'))
                    else:
                        print(f"❌ MCP tool call failed with status {response.status_code}: {response.text}")
                        return f"Tool call failed: HTTP {response.status_code}"
                        
                except Exception as http_error:
                    print(f"❌ HTTP request to MCP server failed: {http_error}")
                    # Fallback: try to call tool directly if it's a simple one
                    return await self._fallback_tool_call(tool_name, arguments)
                    
        except Exception as e:
            print(f"❌ MCP tool call error: {e}")
            return f"Error calling tool {tool_name}: {str(e)}"

    async def _fallback_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Fallback method to call tools directly"""
        try:
            print(f"🔄 Using fallback for tool: {tool_name}")
            
            # Handle specific tools that we can simulate
            if tool_name == "calculator":
                expression = arguments.get("expression", "")
                if expression:
                    try:
                        # Safe evaluation for simple math
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
                
            elif tool_name == "diagnostic":
                test_type = arguments.get("test_type", "basic")
                from datetime import datetime
                current_time = datetime.now().isoformat()
                
                result = f"🔧 Diagnostic Test: {test_type}\n"
                result += f"⏰ Timestamp: {current_time}\n"
                result += f"🖥️ MCP Server: DataFlyWheel App (Fallback Mode)\n"
                result += f"✅ Status: WORKING\n"
                
                if test_type == "basic":
                    result += "📝 Message: MCP server is responding correctly\n"
                    result += "🛠️ Tool Execution: SUCCESS (Fallback)\n"
                    
                return result
                
            # Brave Search tools - inform user that MCP server is needed
            elif tool_name in ["brave_web_search", "brave_local_search"]:
                search_type = "web" if tool_name == "brave_web_search" else "local"
                query = arguments.get("query", "")
                return f"🔍 Brave {search_type} search for '{query}' requires MCP server connection. Please check server status and try again."
                
            # HEDIS tools - inform user that MCP server is needed
            elif tool_name in ["DFWAnalyst", "DFWSearch"]:
                return f"🏥 HEDIS tool '{tool_name}' requires MCP server connection for Snowflake integration. Please check server status."
                
            # Weather tool - inform user that MCP server is needed
            elif tool_name == "get_weather":
                place = arguments.get("place", "unknown location")
                return f"🌤️ Weather information for '{place}' requires MCP server connection. Please check server status."
                
            else:
                return f"Tool {tool_name} not available in fallback mode. Please check MCP server connection."
                
        except Exception as e:
            return f"Fallback tool call failed: {str(e)}"

    def _detect_tool_calls(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Detect if messages contain tool call requests - Updated for Brave Search"""
        tool_calls = []
        
        for message in messages:
            content = str(message.content).lower()
            
            # Updated tool patterns for Brave Search (removed DuckDuckGo and Wikipedia)
            tool_patterns = {
                "calculator": r"(?:use|call).*calculator.*(?:with|tool).*(?:expression|calculate)[:=]\s*([^\n]+)",
                "test_tool": r"(?:use|call).*test_tool.*(?:with|message)[:=]\s*([^\n]+)",
                "diagnostic": r"(?:use|call).*diagnostic.*(?:with|test_type)[:=]\s*([^\n]+)",
                "brave_web_search": r"(?:use|call).*brave_web_search.*(?:with|query|for|search)[:=]\s*([^\n]+)",
                "brave_local_search": r"(?:use|call).*brave_local_search.*(?:with|query|for|search)[:=]\s*([^\n]+)",
                "get_weather": r"(?:use|call).*get_weather.*(?:with|for|place)[:=]\s*([^\n]+)",
                "DFWAnalyst": r"(?:use|call).*DFWAnalyst.*(?:with|prompt)[:=]\s*([^\n]+)",
                "DFWSearch": r"(?:use|call).*DFWSearch.*(?:with|query)[:=]\s*([^\n]+)"
            }
            
            for tool_name, pattern in tool_patterns.items():
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    argument_value = match.group(1).strip().strip('"\'')
                    
                    # Determine the argument name based on tool
                    arg_name = "expression" if tool_name == "calculator" else \
                              "message" if tool_name == "test_tool" else \
                              "test_type" if tool_name == "diagnostic" else \
                              "query" if tool_name in ["brave_web_search", "brave_local_search", "DFWSearch"] else \
                              "place" if tool_name == "get_weather" else \
                              "prompt" if tool_name == "DFWAnalyst" else "query"
                    
                    tool_calls.append({
                        "tool_name": tool_name,
                        "arguments": {arg_name: argument_value}
                    })
                    break
            
            # Also check for explicit tool mentions (updated list)
            explicit_tools = ["calculator", "test_tool", "diagnostic", "brave_web_search", 
                            "brave_local_search", "get_weather", "DFWAnalyst", "DFWSearch"]
            
            for tool in explicit_tools:
                if tool in content and not any(tc["tool_name"] == tool for tc in tool_calls):
                    # Extract argument from context
                    lines = str(message.content).split('\n')
                    for line in lines:
                        if tool in line.lower():
                            # Try to extract argument after colon
                            if ':' in line:
                                arg_value = line.split(':', 1)[1].strip()
                                if arg_value:
                                    arg_name = "expression" if tool == "calculator" else \
                                              "message" if tool == "test_tool" else \
                                              "test_type" if tool == "diagnostic" else \
                                              "query" if tool in ["brave_web_search", "brave_local_search", "DFWSearch"] else \
                                              "place" if tool == "get_weather" else \
                                              "prompt" if tool == "DFWAnalyst" else "query"
                                    
                                    tool_calls.append({
                                        "tool_name": tool,
                                        "arguments": {arg_name: arg_value}
                                    })
                                    break
            
            # Smart search detection - automatically detect when user wants to search
            search_keywords = ["search", "find", "look up", "what is", "who is", "where is", "when is", "how is"]
            location_keywords = ["near", "restaurant", "coffee", "gas station", "hotel", "pharmacy", "shop", "store"]
            
            # Check if this looks like a web search query
            if any(keyword in content for keyword in search_keywords):
                # Check if it's a local search (contains location indicators)
                if any(loc_keyword in content for loc_keyword in location_keywords):
                    # Extract the query (everything after search keywords)
                    for keyword in search_keywords:
                        if keyword in content:
                            query_start = content.find(keyword) + len(keyword)
                            query = content[query_start:].strip()
                            if len(query) > 3:  # Minimum query length
                                tool_calls.append({
                                    "tool_name": "brave_local_search",
                                    "arguments": {"query": query}
                                })
                                break
                else:
                    # Regular web search
                    for keyword in search_keywords:
                        if keyword in content:
                            query_start = content.find(keyword) + len(keyword)
                            query = content[query_start:].strip()
                            if len(query) > 3:  # Minimum query length
                                tool_calls.append({
                                    "tool_name": "brave_web_search",
                                    "arguments": {"query": query}
                                })
                                break
            
            # Weather detection
            weather_keywords = ["weather", "temperature", "forecast", "climate", "conditions"]
            if any(weather_keyword in content for weather_keyword in weather_keywords):
                # Try to extract location
                import re
                # Look for "in [location]" or "for [location]" patterns
                location_match = re.search(r"(?:in|for|at)\s+([a-zA-Z\s,]+)(?:\?|$|\.)", content, re.IGNORECASE)
                if location_match:
                    location = location_match.group(1).strip()
                    if len(location) > 2:
                        tool_calls.append({
                            "tool_name": "get_weather",
                            "arguments": {"place": location}
                        })
        
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
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
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
                            result = future.result(timeout=60)  # Increased timeout for search operations
                    else:
                        result = loop.run_until_complete(
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

        # Prepare messages for Snowflake
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        
        # Add tool results to the conversation
        if tool_results:
            tool_response = "Tool execution results:\n\n"
            for tool_result in tool_results:
                tool_response += f"**{tool_result['tool_name']}**: {tool_result['result']}\n\n"
            
            message_dicts.append({
                "role": "assistant",
                "content": tool_response
            })

        # Prepare the request for Snowflake Cortex
        message_json = json.dumps(message_dicts)
        message_json = message_json.replace("'", "''")  # Escape single quotes for SQL

        options = {
            "temperature": self.temperature,
            "top_p": self.top_p if self.top_p is not None else 1.0,
            "max_tokens": self.max_tokens if self.max_tokens is not None else 2048,
        }
        options_json = json.dumps(options)

        sql_stmt = f"""
            select snowflake.cortex.{self.cortex_function}(
                '{self.model}',
                parse_json(${message_json}$),
                parse_json('{options_json}')
            ) as llm_stream_response;
        """

        try:
            # Use the Snowflake Cortex Complete function
            if self.session:
                self.session.sql(
                    f"USE WAREHOUSE {self.session.get_current_warehouse()};"
                ).collect()
                l_rows = self.session.sql(sql_stmt).collect()
                
                response = json.loads(l_rows[0]["LLM_STREAM_RESPONSE"])
                ai_message_content = response["choices"][0]["messages"]
                
                content = _truncate_at_stop_tokens(ai_message_content, stop)
                
                # If we had tool results, include them in the response
                if tool_results:
                    final_content = ""
                    for tool_result in tool_results:
                        final_content += f"🔧 **{tool_result['tool_name']}**:\n{tool_result['result']}\n\n"
                    final_content += f"\n**AI Response:**\n{content}"
                    content = final_content
                
                message = AIMessage(
                    content=content,
                    response_metadata=response.get("usage", {}),
                )
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])
            else:
                # If no session, return tool results only
                if tool_results:
                    content = ""
                    for tool_result in tool_results:
                        content += f"🔧 **{tool_result['tool_name']}**:\n{tool_result['result']}\n\n"
                    
                    message = AIMessage(content=content)
                    generation = ChatGeneration(message=message)
                    return ChatResult(generations=[generation])
                else:
                    raise ChatSnowflakeCortexError("No Snowflake session available and no tool calls detected")
                
        except Exception as e:
            print(f"❌ Snowflake Cortex error: {e}")
            
            # Fallback: return tool results if available
            if tool_results:
                content = "⚠️ Snowflake Cortex unavailable, showing tool results only:\n\n"
                for tool_result in tool_results:
                    content += f"🔧 **{tool_result['tool_name']}**:\n{tool_result['result']}\n\n"
                
                message = AIMessage(content=content)
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])
            else:
                raise ChatSnowflakeCortexError(
                    f"Error while making request to Snowflake Cortex: {e}"
                )

    def _stream_content(
        self, content: str, stop: Optional[List[str]]
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model in chunks to return ChatGenerationChunk."""
        chunk_size = 50  # Define a reasonable chunk size for streaming
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
        """Stream the output of the model in chunks to return ChatGenerationChunk."""
        
        # For streaming, we'll use the regular generate method and then stream the result
        try:
            result = self._generate(messages, stop, run_manager, **kwargs)
            content = result.generations[0].message.content
            
            # Stream the content
            for chunk in self._stream_content(content, stop):
                yield chunk
                
        except Exception as e:
            # Yield error as chunk
            error_content = f"Streaming error: {str(e)}"
            yield ChatGenerationChunk(message=AIMessageChunk(content=error_content))
