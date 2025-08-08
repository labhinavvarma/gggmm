import json
import re
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
    ToolMessage,
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
    "tool"  # Added tool role support
]

class ChatSnowflakeCortexError(Exception):
    """Error with Snowpark client."""

def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary with enhanced tool support."""
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
        # Handle tool calls in AI messages
        if hasattr(message, 'tool_calls') and message.tool_calls:
            message_dict["tool_calls"] = message.tool_calls
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        if hasattr(message, 'tool_call_id'):
            message_dict["tool_call_id"] = message.tool_call_id
    else:
        raise TypeError(f"Got unknown message type {type(message)}")
    
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

def _extract_serpapi_insights(json_data: Dict[str, Any]) -> str:
    """Extract key insights from SerpApi JSON response for LLM analysis."""
    insights = []
    
    # Extract answer box (direct answers)
    if "answer_box" in json_data:
        answer_box = json_data["answer_box"]
        if isinstance(answer_box, dict):
            if "answer" in answer_box:
                insights.append(f"Direct Answer: {answer_box['answer']}")
            if "title" in answer_box:
                insights.append(f"Answer Title: {answer_box['title']}")
    
    # Extract knowledge graph info
    if "knowledge_graph" in json_data:
        kg = json_data["knowledge_graph"]
        if isinstance(kg, dict):
            if "title" in kg:
                insights.append(f"Entity: {kg['title']}")
            if "type" in kg:
                insights.append(f"Type: {kg['type']}")
            if "description" in kg:
                insights.append(f"Description: {kg['description']}")
    
    # Extract organic results
    if "organic_results" in json_data and isinstance(json_data["organic_results"], list):
        for i, result in enumerate(json_data["organic_results"][:3]):  # Top 3 results
            if isinstance(result, dict):
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No description")
                insights.append(f"Result {i+1}: {title} - {snippet}")
    
    # Extract news results
    if "news_results" in json_data and isinstance(json_data["news_results"], list):
        for i, result in enumerate(json_data["news_results"][:2]):  # Top 2 news
            if isinstance(result, dict):
                title = result.get("title", "No title")
                date = result.get("date", "No date")
                insights.append(f"News {i+1}: {title} ({date})")
    
    return "\n".join(insights) if insights else "No relevant information found in search results."

class ChatSnowflakeCortex(BaseChatModel):
    """Enhanced Snowflake Cortex Chat model with SerpApi JSON processing capabilities.
    
    This version includes:
    - Enhanced tool binding and calling
    - SerpApi JSON response processing
    - Improved MCP integration
    - Better error handling
    """

    # Store tools with enhanced metadata
    mcp_tools: Dict[str, Union[Dict[str, Any], Type, Callable, BaseTool]] = Field(
        default_factory=dict
    )

    session: Any = None
    """Snowpark session object."""

    model: str = "claude-4-sonnet"  # Changed to Claude for better tool usage
    """Snowflake cortex hosted LLM model name."""

    cortex_function: str = "complete"
    """Cortex function to use, defaulted to `complete`."""

    temperature: float = 0.7  # Increased for more creative responses
    """Model temperature. Value should be >= 0 and <= 1.0"""

    max_tokens: Optional[int] = 2048
    """The maximum number of output tokens in the response."""

    top_p: Optional[float] = 0.9
    """top_p adjusts the number of choices for each predicted tokens."""

    # Snowflake connection parameters (same as before)
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
        """Enhanced tool binding for MCP tools with SerpApi support."""

        print(f"🔧 Binding {len(tools)} tools to ChatSnowflakeCortex")
        
        formatted_tools = []
        for tool in tools:
            try:
                formatted_tool = convert_to_openai_tool(tool)
                formatted_tools.append(formatted_tool)
                
                # Store tool reference for execution
                tool_name = formatted_tool.get("function", {}).get("name", "unknown")
                self.mcp_tools[tool_name] = tool
                
                print(f"✅ Tool bound: {tool_name}")
                
            except Exception as e:
                print(f"❌ Failed to bind tool: {e}")

        print(f"🎯 Total tools available: {list(self.mcp_tools.keys())}")
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

    def _process_tool_calls(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Process tool calls and execute MCP tools, with special SerpApi handling."""
        processed_messages = []
        
        for message in messages:
            message_dict = _convert_message_to_dict(message)
            
            # Handle tool calls in AI messages
            if isinstance(message, AIMessage) and hasattr(message, 'tool_calls'):
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", "unknown")
                    
                    print(f"🔧 Executing tool: {tool_name} with args: {tool_args}")
                    
                    if tool_name in self.mcp_tools:
                        try:
                            # Execute the tool
                            tool_result = self.mcp_tools[tool_name].invoke(tool_args)
                            
                            # Special handling for SerpApi responses
                            if tool_name == "SerpApiSearch" and isinstance(tool_result, str):
                                # Extract JSON from the tool result
                                if "COMPLETE JSON RESPONSE:" in tool_result:
                                    json_start = tool_result.find("```json\n") + 8
                                    json_end = tool_result.find("```", json_start)
                                    if json_start > 7 and json_end > json_start:
                                        json_str = tool_result[json_start:json_end]
                                        try:
                                            serpapi_data = json.loads(json_str)
                                            # Extract insights for better LLM processing
                                            insights = _extract_serpapi_insights(serpapi_data)
                                            tool_result = f"SerpApi Search Results:\n{insights}\n\nFull JSON available for detailed analysis."
                                            print(f"🌐 Processed SerpApi response with insights")
                                        except json.JSONDecodeError:
                                            print("⚠️ Could not parse SerpApi JSON, using raw response")
                            
                            # Add tool result as a message
                            processed_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "name": tool_name,
                                "content": str(tool_result)
                            })
                            
                            print(f"✅ Tool {tool_name} executed successfully")
                            
                        except Exception as e:
                            error_msg = f"Tool execution failed: {str(e)}"
                            processed_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "name": tool_name,
                                "content": error_msg
                            })
                            print(f"❌ Tool {tool_name} failed: {e}")
            
            processed_messages.append(message_dict)
        
        return processed_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Enhanced generation with tool calling and SerpApi support."""
        
        print(f"🧠 Generating response for {len(messages)} messages")
        
        # Process messages and handle tool calls
        message_dicts = self._process_tool_calls(messages)

        # Prepare enhanced system message for better tool usage
        enhanced_messages = []
        
        # Add tool information to system context
        if self.mcp_tools:
            tool_names = list(self.mcp_tools.keys())
            system_context = f"""You are an AI assistant with access to the following tools: {', '.join(tool_names)}.

When using SerpApiSearch, you will receive structured search results. Analyze the JSON response and provide accurate, cited answers based on the search data.

For HEDIS queries, use DFWAnalyst for SQL generation and DFWSearch for document searches.

For weather queries, use the get_weather tool with location names.

Always use the most appropriate tool for the user's query and provide helpful, accurate responses."""
            
            enhanced_messages.append({"role": "system", "content": system_context})
        
        enhanced_messages.extend(message_dicts)

        # JSON dump the message_dicts and options
        message_json = json.dumps(enhanced_messages)
        message_json = message_json.replace("'", "''")  # Escape for SQL

        options = {
            "temperature": self.temperature,
            "top_p": self.top_p if self.top_p is not None else 0.9,
            "max_tokens": self.max_tokens if self.max_tokens is not None else 2048,
        }
        options_json = json.dumps(options)

        sql_stmt = f"""
            select snowflake.cortex.{self.cortex_function}(
                '{self.model}',
                parse_json($${message_json}$$),
                parse_json('{options_json}')
            ) as llm_stream_response;
        """

        try:
            # Execute SQL query
            self.session.sql(
                f"USE WAREHOUSE {self.session.get_current_warehouse()};"
            ).collect()
            l_rows = self.session.sql(sql_stmt).collect()
            
            print("✅ Snowflake Cortex query executed successfully")
            
        except Exception as e:
            raise ChatSnowflakeCortexError(
                f"Error while making request to Snowflake Cortex: {e}"
            )

        # Parse response
        response = json.loads(l_rows[0]["LLM_STREAM_RESPONSE"])
        ai_message_content = response["choices"][0]["messages"]

        content = _truncate_at_stop_tokens(ai_message_content, stop)
        message = AIMessage(
            content=content,
            response_metadata=response.get("usage", {}),
        )
        generation = ChatGeneration(message=message)
        
        print(f"📤 Generated response: {len(content)} characters")
        
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
        """Stream the output with enhanced tool support."""
        
        # Use the same enhanced message processing as _generate
        message_dicts = self._process_tool_calls(messages)

        # Add system context for tools
        if self.mcp_tools:
            tool_names = list(self.mcp_tools.keys())
            system_context = f"You have access to these tools: {', '.join(tool_names)}. Use them appropriately."
            message_dicts.insert(0, {"role": "system", "content": system_context})

        message_json = json.dumps(message_dicts)
        message_json = message_json.replace("'", "''")

        options = {
            "temperature": self.temperature,
            "top_p": self.top_p if self.top_p is not None else 0.9,
            "max_tokens": self.max_tokens if self.max_tokens is not None else 2048,
        }
        options_json = json.dumps(options)

        sql_stmt = f"""
            select snowflake.cortex.{self.cortex_function}(
                '{self.model}',
                parse_json($${message_json}$$),
                parse_json('{options_json}')
            ) as llm_stream_response;
        """

        try:
            self.session.sql(
                f"USE WAREHOUSE {self.session.get_current_warehouse()};"
            ).collect()
            result = self.session.sql(sql_stmt).collect()

            for row in result:
                response = json.loads(row["LLM_STREAM_RESPONSE"])
                ai_message_content = response["choices"][0]["messages"]

                for chunk in self._stream_content(ai_message_content, stop):
                    yield chunk

        except Exception as e:
            raise ChatSnowflakeCortexError(
                f"Error while making request to Snowflake Cortex stream: {e}"
            )

# Helper function for MCP integration
def create_enhanced_cortex_model(session, model_name: str = "claude-4-sonnet") -> ChatSnowflakeCortex:
    """Create an enhanced ChatSnowflakeCortex model optimized for MCP tool usage."""
    return ChatSnowflakeCortex(
        session=session,
        model=model_name,
        temperature=0.7,  # Good balance for tool usage and creativity
        max_tokens=2048,
        top_p=0.9
    )
