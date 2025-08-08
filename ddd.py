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

import re

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
    """Snowflake Cortex based Chat model - FIXED to work with LangGraph

    This version fixes the bind_tools method to properly work with LangGraph's create_react_agent.
    """

    test_tools: Dict[str, Union[Dict[str, Any], Type, Callable, BaseTool]] = Field(
        default_factory=dict
    )

    session: Any = None
    """Snowpark session object."""

    model: str = "claude-4-sonnet"  # Changed from mistral-large to claude-4-sonnet
    """Snowflake cortex hosted LLM model name."""

    cortex_function: str = "complete"
    """Cortex function to use, defaulted to `complete`."""

    temperature: float = 0.7  # Changed from 0 to 0.7 for better responses
    """Model temperature. Value should be >= 0 and <= 1.0"""

    max_tokens: Optional[int] = 2048  # Changed from None to 2048
    """The maximum number of output tokens in the response."""

    top_p: Optional[float] = 0.9  # Changed from 0 to 0.9
    """top_p adjusts the number of choices for each predicted tokens."""

    # Snowflake connection parameters
    snowflake_username: Optional[str] = Field(default=None, alias="username")
    snowflake_password: Optional[SecretStr] = Field(default=None, alias="password")
    snowflake_account: Optional[str] = Field(default=None, alias="account")
    snowflake_database: Optional[str] = Field(default=None, alias="database")
    snowflake_schema: Optional[str] = Field(default=None, alias="schema")
    snowflake_warehouse: Optional[str] = Field(default=None, alias="warehouse")
    snowflake_role: Optional[str] = Field(default=None, alias="role")

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]
        ] = "auto",
        **kwargs: Any,
    ) -> "ChatSnowflakeCortex":
        """FIXED: Bind tool-like objects to this chat model - handles MCP tools properly."""

        print(f"🔧 ChatSnowflakeCortex: Attempting to bind {len(tools)} tools")

        try:
            formatted_tools_dict = {}
            
            for i, tool in enumerate(tools):
                try:
                    print(f"🔍 Processing tool {i+1}: {type(tool)}")
                    
                    # Handle different tool types
                    if hasattr(tool, 'name') and hasattr(tool, 'description'):
                        # This looks like an MCP tool or similar
                        tool_name = tool.name
                        print(f"   - Tool name: {tool_name}")
                        
                        # Store the tool directly without conversion
                        formatted_tools_dict[tool_name] = tool
                        print(f"   ✅ Stored tool directly: {tool_name}")
                        
                    elif isinstance(tool, dict):
                        # Already formatted tool
                        if "name" in tool:
                            formatted_tools_dict[tool["name"]] = tool
                            print(f"   ✅ Stored dict tool: {tool['name']}")
                        
                    else:
                        # Try to convert using OpenAI format (but handle errors)
                        try:
                            formatted_tool = convert_to_openai_tool(tool)
                            if "function" in formatted_tool and "name" in formatted_tool["function"]:
                                tool_name = formatted_tool["function"]["name"]
                                formatted_tools_dict[tool_name] = formatted_tool
                                print(f"   ✅ Converted and stored: {tool_name}")
                            else:
                                print(f"   ⚠️ Converted tool missing expected structure")
                        except Exception as conv_error:
                            print(f"   ⚠️ Conversion failed: {conv_error}")
                            # Try to extract name and store directly
                            if hasattr(tool, 'name'):
                                formatted_tools_dict[tool.name] = tool
                                print(f"   ✅ Stored unconverted tool: {tool.name}")
                        
                except Exception as tool_error:
                    print(f"   ❌ Error processing individual tool: {tool_error}")
                    continue

            print(f"✅ Successfully processed tools: {list(formatted_tools_dict.keys())}")
            
            # Create a new instance with bound tools - SIMPLIFIED
            # Instead of creating a new instance (which can cause issues), 
            # just update the current instance and return it
            self.test_tools.update(formatted_tools_dict)
            
            print(f"🎯 Total tools now available: {list(self.test_tools.keys())}")
            
            return self  # Return self instead of new instance to avoid constructor issues

        except Exception as e:
            print(f"❌ Error in bind_tools: {e}")
            print(f"   Returning self without binding tools")
            # Don't fail completely - return self so LangGraph can continue
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
            try:
                self.session.close()
            except:
                pass

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return f"snowflake-cortex-{self.model}"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response with improved MCP tool handling."""
        
        print(f"🧠 ChatSnowflakeCortex: Generating response for {len(messages)} messages")
        
        try:
            message_dicts = [_convert_message_to_dict(m) for m in messages]

            # Enhanced tool invocation handling for MCP tools
            tool_output = None
            for message in messages:
                # Check for various tool invocation patterns
                if isinstance(message, SystemMessage):
                    # Pattern 1: JSON tool invocation in content
                    if (isinstance(message.content, dict) and "invoke_tool" in message.content):
                        tool_info = json.loads(message.content.get("invoke_tool"))
                        tool_name = tool_info.get("tool_name")
                        if tool_name in self.test_tools:
                            tool_args = tool_info.get("args", {})
                            print(f"🔧 Invoking tool (pattern 1): {tool_name} with args: {tool_args}")
                            
                            # Try to invoke the tool
                            tool = self.test_tools[tool_name]
                            if hasattr(tool, 'invoke'):
                                tool_output = tool.invoke(tool_args)
                            elif callable(tool):
                                tool_output = tool(**tool_args)
                            else:
                                print(f"⚠️ Tool {tool_name} is not callable")
                            break
                
                # Pattern 2: Check for tool calls in AI messages (LangGraph pattern)
                elif isinstance(message, AIMessage) and hasattr(message, 'tool_calls'):
                    if message.tool_calls:
                        for tool_call in message.tool_calls:
                            tool_name = tool_call.get('name') or tool_call.get('function', {}).get('name')
                            tool_args = tool_call.get('args') or tool_call.get('function', {}).get('arguments', {})
                            
                            if tool_name in self.test_tools:
                                print(f"🔧 Invoking tool (pattern 2): {tool_name} with args: {tool_args}")
                                
                                tool = self.test_tools[tool_name]
                                if hasattr(tool, 'invoke'):
                                    tool_output = tool.invoke(tool_args)
                                elif callable(tool):
                                    tool_output = tool(**tool_args)
                                break

            # Add tool output to messages if we got one
            if tool_output:
                print(f"✅ Tool executed successfully, output length: {len(str(tool_output))}")
                message_dicts.append({
                    "role": "system",
                    "content": f"Tool output: {str(tool_output)}"
                })

            # JSON dump the message_dicts and options
            message_json = json.dumps(message_dicts)
            message_json = message_json.replace("'", "''")  # Escape single quotes for SQL

            options = {
                "temperature": self.temperature,
                "top_p": self.top_p if self.top_p is not None else 0.9,
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

            print("🔄 Executing Snowflake Cortex query...")

            # Execute SQL
            self.session.sql(f"USE WAREHOUSE {self.session.get_current_warehouse()};").collect()
            l_rows = self.session.sql(sql_stmt).collect()

            print("✅ Snowflake query executed successfully")

            # Parse response
            response = json.loads(l_rows[0]["LLM_STREAM_RESPONSE"])
            ai_message_content = response["choices"][0]["messages"]

            content = _truncate_at_stop_tokens(ai_message_content, stop)
            
            print(f"📤 Generated response: {len(content)} characters")
            
            message = AIMessage(
                content=content,
                response_metadata=response.get("usage", {}),
            )
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

        except Exception as e:
            print(f"❌ ChatSnowflakeCortex generation error: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            raise ChatSnowflakeCortexError(f"Error while making request to Snowflake Cortex: {e}")

    def _stream_content(
        self, content: str, stop: Optional[List[str]]
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model in chunks to return ChatGenerationChunk."""
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
        
        # For simplicity, use _generate and stream the result
        try:
            result = self._generate(messages, stop, run_manager, **kwargs)
            content = result.generations[0].message.content
            
            for chunk in self._stream_content(content, stop):
                yield chunk
                
        except Exception as e:
            print(f"❌ ChatSnowflakeCortex streaming error: {e}")
            raise ChatSnowflakeCortexError(f"Error while streaming from Snowflake Cortex: {e}")
