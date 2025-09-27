"""
Updated Snowflake Cortex AI Client for DSA Agent
Enterprise AI integration for corporate environments
"""

import requests
import json
import uuid
import urllib3
import time
from typing import List, Dict, Any, Iterator

# Disable insecure request warnings for corporate environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class SnowflakeCortexClient:
    """
    Snowflake Cortex AI client for enterprise data science applications
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Snowflake Cortex client
        
        Args:
            config: Configuration dictionary containing all Snowflake settings
        """
        # Use your specific configuration
        self.api_key = config.get('api_key', '78a799ea-a0f6-11ef-a0ce-15a449f7a8b0')
        self.base_url = config.get('base_url_conv_model', 'https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete')
        self.app_id = config.get('app_id', 'edadip')
        self.aplctn_cd = config.get('aplctn_cd', 'edagnai')
        self.model = config.get('conv_model', 'llama3.1-70b')
        
        # HTTP headers for your specific Snowflake API
        self.headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "Authorization": f'Snowflake Token="{self.api_key}"'
        }
        
        # Generate session ID for conversation tracking
        self.session_id = str(uuid.uuid4())
    
    def _build_payload(self, messages: List[Dict[str, str]], sys_msg: str = None) -> Dict[str, Any]:
        """
        Build Snowflake Cortex API payload matching your exact format
        """
        # Default system message if not provided
        if not sys_msg and messages and messages[0].get('role') == 'system':
            sys_msg = messages[0]['content']
            messages = messages[1:]  # Remove system message from messages list
        elif not sys_msg:
            sys_msg = "You are a powerful AI assistant. Provide accurate, concise answers based on context."
        
        # Get the latest user message
        user_message = ""
        if messages:
            # Find the last user message
            for msg in reversed(messages):
                if msg.get('role') == 'user':
                    user_message = msg['content']
                    break
        
        # Use your exact payload structure
        payload = {
            "query": {
                "aplctn_cd": self.aplctn_cd,
                "app_id": self.app_id,
                "api_key": self.api_key,
                "method": "cortex",
                "model": self.model,
                "sys_msg": sys_msg,
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
                "session_id": self.session_id
            }
        }
        
        return payload
    
    def _make_request(self, payload: Dict[str, Any]) -> requests.Response:
        """
        Make HTTP request to your Snowflake Cortex API
        
        Args:
            payload: Request payload
            
        Returns:
            HTTP response object
        """
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload, verify=False)
            return response
        except Exception as e:
            print(f"❌ Request Failed: {str(e)}")
            raise
    
    def _parse_response(self, response_text: str) -> str:
        """
        Parse response text handling the end_of_stream format
        """
        if "end_of_stream" in response_text:
            answer, _, _ = response_text.partition("end_of_stream")
            return answer.strip()
        else:
            return response_text.strip()
    
    class ChatCompletion:
        """Chat completion interface for Snowflake Cortex"""
        
        def __init__(self, client):
            self.client = client
        
        def create(self, model: str, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> 'CompletionResponse':
            """
            Create chat completion using Snowflake Cortex
            
            Args:
                model: Model name (will use client's configured model)
                messages: List of messages
                stream: Whether to stream the response
                **kwargs: Additional parameters
                
            Returns:
                CompletionResponse object or streaming generator
            """
            # Extract system message if present
            sys_msg = None
            if messages and messages[0].get('role') == 'system':
                sys_msg = messages[0]['content']
            
            # Build payload
            payload = self.client._build_payload(messages, sys_msg)
            
            if stream:
                return self._create_streaming_response(payload)
            else:
                return self._create_regular_response(payload)
        
        def _create_regular_response(self, payload):
            """Create regular (non-streaming) response"""
            # Make request
            response = self.client._make_request(payload)
            
            # Handle response
            if response.status_code == 200:
                try:
                    raw_text = response.text
                    content = self.client._parse_response(raw_text)
                    
                    # Create usage stats
                    usage = UsageStats(
                        prompt_tokens=len(str(payload)) // 4,  # Rough estimate
                        completion_tokens=len(content) // 4,    # Rough estimate
                        total_tokens=len(str(payload) + content) // 4
                    )
                    
                    return CompletionResponse(content, usage)
                    
                except Exception as e:
                    raise Exception(f"⚠️ Failed to parse response: {str(e)}. Raw response: {response.text}")
            else:
                try:
                    error_data = response.json()
                    raise Exception(f"⚠️ Error Response: {json.dumps(error_data, indent=2)}")
                except:
                    raise Exception(f"⚠️ Error Response: {response.text}")
        
        def _create_streaming_response(self, payload):
            """Create streaming response generator"""
            # Make request
            response = self.client._make_request(payload)
            
            if response.status_code == 200:
                raw_text = response.text
                content = self.client._parse_response(raw_text)
                
                # Return streaming response that simulates gradual output
                return StreamingResponse(content)
            else:
                raise Exception(f"⚠️ Streaming Error: {response.text}")
    
    @property
    def chat(self):
        """Provide chat.completions interface"""
        if not hasattr(self, '_chat'):
            self._chat = type('chat', (), {})()
            self._chat.completions = self.ChatCompletion(self)
        return self._chat


class CompletionResponse:
    """Response object for chat completions"""
    
    def __init__(self, content: str, usage: 'UsageStats'):
        self.choices = [
            type('choice', (), {
                'message': type('message', (), {'content': content})()
            })()
        ]
        self.usage = usage


class UsageStats:
    """Usage statistics for API calls"""
    
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


# Streaming support for UI compatibility
class StreamingResponse:
    """Streaming response simulation for gradual text display"""
    
    def __init__(self, content: str):
        self.content = content
        self.words = content.split()
    
    def __iter__(self):
        """Simulate streaming by yielding words progressively"""
        accumulated = ""
        for word in self.words:
            accumulated += word + " "
            # Create chunk object with delta content
            chunk = type('chunk', (), {
                'choices': [
                    type('choice', (), {
                        'delta': type('delta', (), {'content': word + " "})()
                    })()
                ]
            })()
            yield chunk
            time.sleep(0.01)  # Small delay to simulate streaming


def create_snowflake_client(config: Dict[str, Any]) -> SnowflakeCortexClient:
    """
    Factory function to create Snowflake Cortex client
    
    Args:
        config: Configuration dictionary containing Snowflake settings
        
    Returns:
        SnowflakeCortexClient instance
    """
    return SnowflakeCortexClient(config)


# Test function
if __name__ == "__main__":
    # Test configuration with your settings
    test_config = {
        'api_key': '78a799ea-a0f6-11ef-a0ce-15a449f7a8b0',
        'base_url_conv_model': 'https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete',
        'app_id': 'edadip',
        'aplctn_cd': 'edagnai',
        'conv_model': 'llama3.1-70b'
    }
    
    # Test the client
    client = SnowflakeCortexClient(test_config)
    
    # Test chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="llama3.1-70b",
            messages=messages
        )
        print("✅ Test successful!")
        print(f"Response: {response.choices[0].message.content}")
        print(f"Usage: {response.usage.total_tokens} tokens")
    except Exception as e:
        print(f"❌ Test failed: {e}")
