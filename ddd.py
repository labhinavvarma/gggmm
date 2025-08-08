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
import threading
import concurrent.futures

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

def _escape_sql_string(text: str) -> str:
    """Properly escape string for SQL"""
    # Replace single quotes with double single quotes for SQL
    return text.replace("'", "''")

class ChatSnowflakeCortex(BaseChatModel):
    """Fixed Snowflake Cortex Chat model with proper async and SQL handling"""
    
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
        try:
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
        except Exception as e:
            print(f"Error binding tools: {e}")
        
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

    def _call_mcp_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Synchronous wrapper for MCP tool calls"""
        try:
            print(f"🔧 Calling MCP tool: {tool_name} with args: {arguments}")
            
            # Create the MCP tool call request
            tool_call_data = {
                "tool_name": tool_name,
                "arguments": arguments
            }
            
            # Use requests instead of httpx for synchronous call
            import requests
            
            base_url = self.mcp_server_url.rstrip('/sse')
            
            response = requests.post(
                f"{base_url}/tool_call",
                json=tool_call_data,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                tool_result = result.get('result', 'No result returned')
                print(f"✅ Tool {tool_name} executed successfully")
                return str(tool_result)
            else:
                error_msg = f"Tool call failed: HTTP {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                return error_msg
                    
        except Exception as e:
            error_msg = f"Error calling tool {tool_name}: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    def _simple_tool_detection(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Simplified tool detection with basic patterns"""
        tool_calls = []
        
        # Combine all message content
        all_content = " ".join([str(msg.content).lower() for msg in messages])
        
        try:
            # Calculator detection - Enhanced patterns
            calc_patterns = [
                r"calculate\s+(.+)",
                r"calculator.*(?:expression|calculate)[:\s]+(.+)",
                r"use.*calculator.*[:\s]+(.+)",
                r"compute\s+(.+)",
                r"what.*is\s+(.+\d+.*[\+\-\*/].*\d+.*)",  # Match mathematical expressions
                r"compound.*interest.*\$?(\d+).*(\d+\.?\d*)%.*(\d+).*year"  # Compound interest pattern
            ]
            
            for pattern in calc_patterns:
                match = re.search(pattern, all_content, re.IGNORECASE)
                if match:
                    if "compound" in pattern and "interest" in pattern:
                        # Special handling for compound interest
                        if "compound interest" in all_content:
                            # Extract principal, rate, time from the message
                            principal_match = re.search(r'\$?(\d+)', all_content)
                            rate_match = re.search(r'(\d+\.?\d*)%', all_content)
                            time_match = re.search(r'(\d+)\s*year', all_content)
                            
                            if principal_match and rate_match and time_match:
                                principal = principal_match.group(1)
                                rate = rate_match.group(1)
                                time = time_match.group(1)
                                
                                # Compound interest formula: A = P(1 + r/100)^t
                                expression = f"{principal} * (1 + {rate}/100)^{time}"
                                tool_calls.append({
                                    "tool_name": "calculator",
                                    "arguments": {"expression": expression}
                                })
                                break
                    else:
                        expression = match.group(1).strip().strip('"\'')
                        tool_calls.append({
                            "tool_name": "calculator",
                            "arguments": {"expression": expression}
                        })
                        break
            
            # Weather detection
            weather_patterns = [
                r"weather.*(?:in|for|at)\s+([a-zA-Z\s,]+)",
                r"get.*weather.*[:\s]+([a-zA-Z\s,]+)",
                r"(?:temperature|forecast|climate).*(?:in|for|at)\s+([a-zA-Z\s,]+)",
                r"current.*weather.*in\s+([a-zA-Z\s,]+)"
            ]
            for pattern in weather_patterns:
                match = re.search(pattern, all_content, re.IGNORECASE)
                if match:
                    place = match.group(1).strip().strip('"\'.,?!')
                    if len(place) > 2:  # Minimum place name length
                        tool_calls.append({
                            "tool_name": "get_weather",
                            "arguments": {"place": place}
                        })
                        break
            
            # Brave Web Search detection
            web_search_patterns = [
                r"search.*(?:for|about)\s+(.+)",
                r"brave.*search.*[:\s]+(.+)",
                r"find.*information.*about\s+(.+)",
                r"look.*up\s+(.+)",
                r"latest\s+(.+)",  # "latest AI developments"
                r"(?:what|who|when|where|how).*is\s+(.+)",
                r"tell.*me.*about\s+(.+)"
            ]
            for pattern in web_search_patterns:
                match = re.search(pattern, all_content, re.IGNORECASE)
                if match:
                    query = match.group(1).strip().strip('"\'.,?!')
                    # Check if it's not a location-based query
                    location_keywords = ["near", "restaurant", "coffee", "hotel", "gas station", "shop"]
                    if not any(keyword in query.lower() for keyword in location_keywords):
                        if len(query) > 3:
                            tool_calls.append({
                                "tool_name": "brave_web_search",
                                "arguments": {"query": query, "count": 5}
                            })
                            break
            
            # Local Search detection  
            local_search_keywords = ["near", "restaurant", "coffee", "hotel", "gas station", "shop", "store"]
            if any(keyword in all_content for keyword in local_search_keywords):
                # Try to extract local search query
                local_patterns = [
                    r"find.*(?:restaurant|coffee|hotel|gas station|shop|store).*(?:near|in)\s+(.+)",
                    r"(?:restaurant|coffee|hotel|shop).*near\s+(.+)",
                    r"search.*(?:restaurant|coffee|hotel).*[:\s]+(.+)"
                ]
                for pattern in local_patterns:
                    match = re.search(pattern, all_content, re.IGNORECASE)
                    if match:
                        query = match.group(1).strip().strip('"\'.,?!')
                        if len(query) > 3:
                            tool_calls.append({
                                "tool_name": "brave_local_search", 
                                "arguments": {"query": query, "count": 5}
                            })
                            break
            
            # Test tool detection
            if "test_tool" in all_content or "test tool" in all_content:
                test_patterns = [
                    r"test.*tool.*[:\s]+(.+)",
                    r"use.*test.*tool.*message[:\s]+(.+)"
                ]
                message = "test"
                for pattern in test_patterns:
                    match = re.search(pattern, all_content, re.IGNORECASE)
                    if match:
                        message = match.group(1).strip().strip('"\'')
                        break
                
                tool_calls.append({
                    "tool_name": "test_tool",
                    "arguments": {"message": message}
                })
            
            # Diagnostic tool detection
            if "diagnostic" in all_content:
                diag_patterns = [
                    r"diagnostic.*[:\s]+(.+)",
                    r"run.*diagnostic.*test[:\s]+(.+)"
                ]
                test_type = "basic"
                for pattern in diag_patterns:
                    match = re.search(pattern, all_content, re.IGNORECASE)
                    if match:
                        test_type = match.group(1).strip().strip('"\'')
                        break
                
                tool_calls.append({
                    "tool_name": "diagnostic",
                    "arguments": {"test_type": test_type}
                })
            
            # HEDIS tools detection
            if "DFWAnalyst" in str(messages) or "hedis" in all_content:
                for msg in messages:
                    content = str(msg.content)
                    if "DFWAnalyst" in content:
                        # Extract prompt after DFWAnalyst
                        parts = content.split("DFWAnalyst")
                        if len(parts) > 1:
                            prompt = parts[1].strip().strip('":').strip()
                            if prompt:
                                tool_calls.append({
                                    "tool_name": "DFWAnalyst",
                                    "arguments": {"prompt": prompt}
                                })
            
            if "DFWSearch" in str(messages):
                for msg in messages:
                    content = str(msg.content)
                    if "DFWSearch" in content:
                        # Extract query after DFWSearch
                        parts = content.split("DFWSearch")
                        if len(parts) > 1:
                            query = parts[1].strip().strip('":').strip()
                            if query:
                                tool_calls.append({
                                    "tool_name": "DFWSearch",
                                    "arguments": {"query": query}
                                })
            
        except Exception as e:
            print(f"Error in tool detection: {e}")
            # Return empty list if detection fails
            return []
        
        print(f"🔍 Detected {len(tool_calls)} tool calls: {[tc['tool_name'] for tc in tool_calls]}")
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
        tool_calls = self._simple_tool_detection(messages)
        
        # Execute tool calls if detected
        tool_results = []
        if tool_calls:
            print(f"🔧 Executing {len(tool_calls)} tool calls")
            
            for tool_call in tool_calls:
                try:
                    # Use synchronous call to avoid event loop issues
                    result = self._call_mcp_tool_sync(tool_call["tool_name"], tool_call["arguments"])
                    
                    tool_results.append({
                        "tool_name": tool_call["tool_name"],
                        "result": result
                    })
                    
                except Exception as e:
                    error_msg = f"Tool execution failed: {str(e)}"
                    print(f"❌ Tool {tool_call['tool_name']} failed: {e}")
                    tool_results.append({
                        "tool_name": tool_call["tool_name"],
                        "result": error_msg
                    })

        # If we have tool results, format them nicely
        if tool_results:
            final_content = "Tool Execution Results:\n\n"
            for tool_result in tool_results:
                final_content += f"🔧 **{tool_result['tool_name']}**:\n"
                final_content += f"{tool_result['result']}\n\n"
            
            # Try to get AI response from Snowflake if session is available
            try:
                if self.session:
                    # Prepare messages for Snowflake with proper escaping
                    message_dicts = [_convert_message_to_dict(m) for m in messages]
                    
                    # Create a simple query without tool results to avoid JSON complexity
                    simple_message = f"Based on the tool execution results: {final_content}, provide a helpful summary or analysis."
                    
                    simple_messages = [{"role": "user", "content": simple_message}]
                    
                    # Properly escape and format for SQL
                    message_json = json.dumps(simple_messages)
                    
                    options = {
                        "temperature": self.temperature,
                        "top_p": self.top_p if self.top_p is not None else 1.0,
                        "max_tokens": self.max_tokens if self.max_tokens is not None else 2048,
                    }
                    options_json = json.dumps(options)

                    # Use SQL parameters to avoid escaping issues
                    sql_stmt = f"""
                        select snowflake.cortex.{self.cortex_function}(
                            '{self.model}',
                            parse_json('{_escape_sql_string(message_json)}'),
                            parse_json('{_escape_sql_string(options_json)}')
                        ) as llm_stream_response;
                    """

                    self.session.sql(
                        f"USE WAREHOUSE {self.session.get_current_warehouse()};"
                    ).collect()
                    l_rows = self.session.sql(sql_stmt).collect()
                    
                    response = json.loads(l_rows[0]["LLM_STREAM_RESPONSE"])
                    ai_response = response["choices"][0]["messages"]
                    
                    # Combine tool results with AI response
                    final_content += f"\n**AI Analysis:**\n{ai_response}"
                    
            except Exception as e:
                print(f"⚠️ Snowflake Cortex error: {e}")
                final_content += f"\n*Note: AI analysis unavailable - {str(e)}*"
            
            message = AIMessage(content=final_content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        
        else:
            # No tool calls detected, use regular Snowflake processing
            try:
                if not self.session:
                    raise ChatSnowflakeCortexError("No Snowflake session available")
                
                # Prepare messages for Snowflake with proper escaping
                message_dicts = [_convert_message_to_dict(m) for m in messages]
                message_json = json.dumps(message_dicts)

                options = {
                    "temperature": self.temperature,
                    "top_p": self.top_p if self.top_p is not None else 1.0,
                    "max_tokens": self.max_tokens if self.max_tokens is not None else 2048,
                }
                options_json = json.dumps(options)

                sql_stmt = f"""
                    select snowflake.cortex.{self.cortex_function}(
                        '{self.model}',
                        parse_json('{_escape_sql_string(message_json)}'),
                        parse_json('{_escape_sql_string(options_json)}')
                    ) as llm_stream_response;
                """

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
                error_msg = f"Error processing request: {str(e)}"
                print(f"❌ Generation error: {e}")
                message = AIMessage(content=error_msg)
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
        """Stream the output of the model in chunks."""
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
