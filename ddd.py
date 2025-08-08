import json
from typing import Any, Dict, List, Optional, Iterator
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
from pydantic import Field

class ChatSnowflakeCortexError(Exception):
    """Error with Snowpark client."""

class SimpleChatSnowflakeCortex(BaseChatModel):
    """
    Simplified Snowflake Cortex Chat model that works reliably with LangGraph.
    
    Removes complex tool binding and focuses on basic LLM functionality.
    LangGraph will handle tool calling separately.
    """

    session: Any = None
    """Snowpark session object."""

    model: str = "claude-4-sonnet"
    """Snowflake cortex hosted LLM model name."""

    cortex_function: str = "complete" 
    """Cortex function to use."""

    temperature: float = 0.7
    """Model temperature."""

    max_tokens: Optional[int] = 2048
    """Maximum number of output tokens."""

    top_p: Optional[float] = 0.9
    """top_p parameter."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f"🧠 SimpleChatSnowflakeCortex initialized with model: {self.model}")

    @property
    def _llm_type(self) -> str:
        """Get the type of language model."""
        return f"simple-snowflake-cortex-{self.model}"

    def _convert_message_to_dict(self, message: BaseMessage) -> dict:
        """Convert a LangChain message to a simple dictionary."""
        
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": str(message.content)}
        elif isinstance(message, HumanMessage):
            return {"role": "user", "content": str(message.content)}
        elif isinstance(message, AIMessage):
            return {"role": "assistant", "content": str(message.content)}
        elif isinstance(message, ChatMessage):
            return {"role": message.role, "content": str(message.content)}
        else:
            # Fallback for any other message type
            return {"role": "user", "content": str(message.content)}

    def _truncate_at_stop_tokens(self, text: str, stop: Optional[List[str]]) -> str:
        """Truncate text at stop tokens."""
        if stop is None:
            return text
        
        for stop_token in stop:
            stop_token_idx = text.find(stop_token)
            if stop_token_idx != -1:
                text = text[:stop_token_idx]
        return text

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response using Snowflake Cortex - SIMPLIFIED VERSION."""
        
        print(f"🧠 SimpleChatSnowflakeCortex: Processing {len(messages)} messages")
        
        try:
            # Convert messages to simple format
            message_dicts = [self._convert_message_to_dict(m) for m in messages]
            
            print(f"📋 Converted {len(message_dicts)} messages for Snowflake")
            
            # Create JSON for Snowflake - SIMPLIFIED
            message_json = json.dumps(message_dicts)
            message_json = message_json.replace("'", "''")  # Escape single quotes for SQL

            # Set options
            options = {
                "temperature": self.temperature,
                "top_p": self.top_p if self.top_p is not None else 0.9,
                "max_tokens": self.max_tokens if self.max_tokens is not None else 2048,
            }
            options_json = json.dumps(options)

            print(f"⚙️ Using options: temperature={self.temperature}, max_tokens={self.max_tokens}")

            # Build SQL query
            sql_stmt = f"""
                SELECT snowflake.cortex.{self.cortex_function}(
                    '{self.model}',
                    parse_json($${message_json}$$),
                    parse_json('{options_json}')
                ) as llm_response;
            """

            print("🔄 Executing Snowflake Cortex query...")

            # Execute the query
            self.session.sql(f"USE WAREHOUSE {self.session.get_current_warehouse()};").collect()
            result_rows = self.session.sql(sql_stmt).collect()

            print("✅ Snowflake query executed successfully")

            # Parse the response - SIMPLIFIED
            response_data = json.loads(result_rows[0]["LLM_RESPONSE"])
            ai_content = response_data["choices"][0]["messages"]

            print(f"📤 Generated response: {len(ai_content)} characters")

            # Apply stop token truncation
            content = self._truncate_at_stop_tokens(ai_content, stop)

            # Create AI message
            message = AIMessage(
                content=content,
                response_metadata=response_data.get("usage", {}),
            )

            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

        except Exception as e:
            print(f"❌ SimpleChatSnowflakeCortex error: {e}")
            raise ChatSnowflakeCortexError(f"Failed to generate response: {e}")

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream response (simplified version)."""
        
        print("🌊 Starting streaming response...")
        
        # For simplicity, use _generate and split into chunks
        result = self._generate(messages, stop, run_manager, **kwargs)
        content = result.generations[0].message.content

        # Split into chunks for streaming
        chunk_size = 50
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))

    def __del__(self) -> None:
        if getattr(self, "session", None) is not None:
            try:
                self.session.close()
            except:
                pass

def create_simple_cortex_model(session, model_name: str = "claude-4-sonnet") -> SimpleChatSnowflakeCortex:
    """Create a simple, reliable Cortex model."""
    
    print(f"🏗️ Creating SimpleChatSnowflakeCortex with model: {model_name}")
    
    model = SimpleChatSnowflakeCortex(
        session=session,
        model=model_name,
        temperature=0.7,
        max_tokens=2048,
        top_p=0.9
    )
    
    print(f"✅ SimpleChatSnowflakeCortex created successfully")
    return model
