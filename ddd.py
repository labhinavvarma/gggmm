import streamlit as st
import asyncio
import json
import traceback

from mcp.client.sse import sse_client
from mcp import ClientSession
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dependencies import SnowFlakeConnector
from llmobjectwrapper import create_simple_cortex_model  # Import simple model
from snowflake.snowpark import Session

# Page config
st.set_page_config(page_title="Reliable MCP Demo", page_icon="✅", layout="wide")
st.title("✅ Reliable MCP Demo - Calculator & Web Search")

# Configuration
server_url = st.sidebar.text_input("MCP Server URL", "http://10.126.192.183:8001/sse")

# Simple connection functions
@st.cache_resource
def get_snowflake_connection():
    """Get Snowflake connection."""
    try:
        conn = SnowFlakeConnector.get_conn('aedl', '')
        print("✅ Snowflake connection established")
        return conn
    except Exception as e:
        print(f"❌ Snowflake connection failed: {e}")
        raise

@st.cache_resource 
def get_simple_model():
    """Get simple Cortex model."""
    try:
        sf_conn = get_snowflake_connection()
        session = Session.builder.configs({"connection": sf_conn}).getOrCreate()
        model = create_simple_cortex_model(session, "claude-4-sonnet")
        print("✅ Simple model created successfully")
        return model
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        raise

# Simple mode selection
prompt_type = st.sidebar.radio("Select Mode", [
    "🧮 Calculator",
    "🌐 Web Search", 
    "🌤️ Weather",
    "💬 Direct Chat"
])

# Simple prompt mapping - matches MCP server exactly
prompt_map = {
    "🧮 Calculator": "calculator-prompt",
    "🌐 Web Search": "serpapi-prompt", 
    "🌤️ Weather": "weather-prompt",
    "💬 Direct Chat": None
}

# Simple examples
examples = {
    "🧮 Calculator": [
        "What is 15% of 85,000?",
        "Calculate 25 * 45",
        "What is the square root of 144?",
        "Calculate (100 + 50) / 3"
    ],
    "🌐 Web Search": [
        "Who is the current prime minister of India?",
        "Latest AI news today",
        "Current stock price of Apple",
        "Recent developments in healthcare"
    ],
    "🌤️ Weather": [
        "Weather in Atlanta today",
        "Current weather in Boston",
        "Temperature in Miami"
    ],
    "💬 Direct Chat": [
        "Explain quantum computing",
        "What is machine learning?",
        "Benefits of renewable energy"
    ]
}

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Layout
col1, col2 = st.columns([2, 1])

# Chat area
with col1:
    st.subheader(f"Chat - {prompt_type}")
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Sidebar examples and status
with col2:
    st.subheader("💡 Examples")
    for example in examples[prompt_type]:
        if st.button(example, key=f"ex_{hash(example)}", use_container_width=True):
            st.session_state.query_input = example

    st.subheader("📊 System Status")
    
    # Check MCP server status
    async def check_mcp_status():
        try:
            async with sse_client(url=server_url) as sse_connection:
                async with ClientSession(*sse_connection) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    tool_count = len([t for t in tools.tools if t.name not in {"add-frequent-questions", "add-prompts", "suggested_top_prompts"}])
                    return True, tool_count
        except Exception as e:
            return False, str(e)
    
    try:
        server_online, tool_info = asyncio.run(check_mcp_status())
        if server_online:
            st.success(f"🟢 MCP Server: Online ({tool_info} tools)")
        else:
            st.error(f"🔴 MCP Server: Offline - {tool_info}")
    except:
        st.warning("⚠️ MCP Server: Status unknown")
    
    # Check model status
    try:
        model = get_simple_model()
        st.success("🟢 LLM Model: Ready")
    except Exception as e:
        st.error(f"🔴 LLM Model: Error - {str(e)[:50]}...")

def extract_response_from_langgraph(response):
    """
    ROBUST response extraction specifically designed for LangGraph AddableValuesDict.
    This function tries multiple methods to extract meaningful content.
    """
    
    print(f"\n🔍 RESPONSE EXTRACTION DEBUG:")
    print(f"   Type: {type(response)}")
    print(f"   String representation length: {len(str(response))}")
    
    try:
        # Method 1: Direct dictionary-like access
        if hasattr(response, 'get') or hasattr(response, '__getitem__'):
            print("📋 Trying dictionary-like access...")
            
            # Check for common keys that contain the final response
            for key in ['messages', 'output', 'result', 'content']:
                try:
                    if key in response:
                        value = response[key]
                        print(f"🔍 Found key '{key}': {type(value)}")
                        
                        if key == 'messages' and isinstance(value, list) and value:
                            # Get the last message
                            last_msg = value[-1]
                            print(f"🔍 Last message type: {type(last_msg)}")
                            
                            # Try to extract content from the message
                            if hasattr(last_msg, 'content'):
                                content = last_msg.content
                                print(f"✅ Extracted from {key}[{len(value)-1}].content: {len(content)} chars")
                                return content
                            elif isinstance(last_msg, dict) and 'content' in last_msg:
                                content = last_msg['content']
                                print(f"✅ Extracted from {key}[{len(value)-1}]['content']: {len(content)} chars")
                                return content
                            else:
                                # Last resort: convert message to string
                                content = str(last_msg)
                                if len(content) > 10 and 'object at 0x' not in content:
                                    print(f"✅ Converted message to string: {len(content)} chars")
                                    return content
                        
                        elif isinstance(value, str) and value.strip():
                            print(f"✅ Extracted string from key '{key}': {len(value)} chars")
                            return value
                            
                except Exception as e:
                    print(f"⚠️ Error accessing key '{key}': {e}")
                    continue
        
        # Method 2: Check for direct messages attribute
        if hasattr(response, 'messages'):
            print("📋 Trying direct messages attribute...")
            messages = response.messages
            
            if messages and len(messages) > 0:
                last_msg = messages[-1]
                if hasattr(last_msg, 'content'):
                    content = last_msg.content
                    print(f"✅ Extracted from .messages attribute: {len(content)} chars")
                    return content
        
        # Method 3: Try to iterate through the object if it's iterable
        if hasattr(response, 'items'):
            print("📋 Trying iteration through items...")
            for key, value in response.items():
                print(f"🔍 Iterating: {key} = {type(value)}")
                
                # Look for lists that might contain messages
                if isinstance(value, list) and value:
                    for item in value:
                        if hasattr(item, 'content'):
                            content = item.content
                            if isinstance(content, str) and len(content) > 10:
                                print(f"✅ Found content in list item: {len(content)} chars")
                                return content
                
                # Look for string values
                elif isinstance(value, str) and len(value) > 10:
                    print(f"✅ Found string value: {len(value)} chars")
                    return value
        
        # Method 4: String parsing as last resort
        print("📋 Trying string parsing...")
        str_response = str(response)
        
        # Look for patterns that indicate actual content
        patterns_to_find = [
            "content='",
            "content\":",
            "content: '",
            "content: \"",
            "'content': '",
            "\"content\": \""
        ]
        
        for pattern in patterns_to_find:
            if pattern in str_response:
                # Try to extract content after the pattern
                start_idx = str_response.find(pattern) + len(pattern)
                if start_idx < len(str_response):
                    # Find the end quote
                    for end_char in ["'", "\""]:
                        end_idx = str_response.find(end_char, start_idx)
                        if end_idx > start_idx:
                            content = str_response[start_idx:end_idx]
                            if len(content) > 10 and not content.startswith('object at'):
                                print(f"✅ Extracted via string parsing: {len(content)} chars")
                                return content
        
        # Method 5: If we find any meaningful text in the string representation
        if len(str_response) < 2000 and len(str_response) > 20:
            # Filter out object references and technical jargon
            if not any(x in str_response.lower() for x in ['object at 0x', 'addablevaluesdict', '<class']):
                print(f"✅ Using cleaned string representation: {len(str_response)} chars")
                return str_response
        
        print("❌ Could not extract meaningful content")
        return None
        
    except Exception as e:
        print(f"❌ Response extraction error: {e}")
        return None

# MAIN CHAT INPUT PROCESSING
with col1:
    if query := st.chat_input("Type your query here...") or "query_input" in st.session_state:

        if "query_input" in st.session_state:
            query = st.session_state.query_input
            del st.session_state.query_input
       
        with st.chat_message("user"):
            st.markdown(query)
       
        st.session_state.messages.append({"role": "user", "content": query})

        async def reliable_processing(query_text):
            """RELIABLE processing with robust response extraction."""
            
            with st.chat_message("assistant"):
                placeholder = st.empty()
                
                try:
                    placeholder.text("🔄 Step 1: Connecting to MCP server...")
                    
                    # Initialize MCP client
                    client = MultiServerMCPClient(
                        {"DataFlyWheelServer": {"url": server_url, "transport": "sse"}}
                    )
                    
                    placeholder.text("🧠 Step 2: Loading model and tools...")
                    
                    # Get model and tools
                    model = get_simple_model()
                    tools = await client.get_tools()
                    
                    print(f"🔧 Retrieved {len(tools)} tools from MCP server:")
                    for tool in tools:
                        if tool.name not in {"add-frequent-questions", "add-prompts", "suggested_top_prompts"}:
                            print(f"   - {tool.name}")
                    
                    placeholder.text("🤖 Step 3: Creating AI agent...")
                    
                    # Create agent with simple model
                    agent = create_react_agent(model=model, tools=tools)
                    
                    placeholder.text("📋 Step 4: Getting prompt from server...")
                    
                    # Get appropriate prompt
                    prompt_name = prompt_map[prompt_type]
                    
                    if prompt_name is None:
                        # Direct chat mode
                        messages = [{"role": "user", "content": query_text}]
                        print("💬 Using direct chat mode")
                    else:
                        # Get prompt from MCP server
                        print(f"📋 Requesting prompt: {prompt_name}")
                        server_prompt = await client.get_prompt(
                            server_name="DataFlyWheelServer",
                            prompt_name=prompt_name,
                            arguments={"query": query_text}
                        )
                        messages = server_prompt
                        print(f"✅ Received prompt with {len(messages)} messages")
                    
                    placeholder.text("🚀 Step 5: Processing with AI agent...")
                    
                    # Run the agent
                    response = await agent.ainvoke({"messages": messages})
                    
                    placeholder.text("📤 Step 6: Extracting response...")
                    
                    # ROBUST response extraction
                    result = extract_response_from_langgraph(response)
                    
                    if result and isinstance(result, str) and result.strip():
                        # Clean up the result
                        result = result.strip()
                        
                        # Display the result
                        placeholder.markdown(result)
                        st.session_state.messages.append({"role": "assistant", "content": result})
                        
                        # Show success
                        with col2:
                            st.success("✅ Query completed successfully!")
                        
                        print(f"✅ Successfully processed query and extracted {len(result)} character response")
                        
                    else:
                        # Create detailed debug information
                        debug_info = f"""
❌ **Response Extraction Failed**

**Query:** "{query_text}"
**Mode:** {prompt_type}

**Technical Details:**
- Response Type: `{type(response)}`
- Response String Length: {len(str(response))}
- Extraction Result: {type(result)} = {result}

**Raw Response (first 800 characters):**
```
{str(response)[:800]}{'...' if len(str(response)) > 800 else ''}
```

**Troubleshooting Steps:**
1. Try a simpler query
2. Check if the tool is working (try calculator: "2+2")
3. Verify MCP server logs
4. Try a different mode

**Note:** The agent executed but we couldn't extract the final response content.
This might be a response format issue rather than a functional problem.
"""
                        
                        placeholder.markdown(debug_info)
                        st.session_state.messages.append({"role": "assistant", "content": debug_info})
                        
                        with col2:
                            st.error("❌ Response extraction failed")
                        
                        print(f"❌ Response extraction failed for query: {query_text}")

                except Exception as e:
                    error_msg = f"""
❌ **Processing Error**

**Error:** {str(e)}

**Query:** "{query_text}"
**Mode:** {prompt_type}
**Server:** {server_url}

**Debug Information:**
```
{traceback.format_exc()}
```

**Common Solutions:**
1. Check MCP server is running at {server_url}
2. Verify Snowflake connection is working
3. Ensure tools are properly registered
4. Try restarting the MCP server
"""
                    placeholder.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
                    with col2:
                        st.error("❌ Processing failed")
                    
                    print(f"❌ Processing error: {e}")

        if query:
            asyncio.run(reliable_processing(query))

# Control buttons
with col1:
    col_clear, col_refresh, col_test = st.columns(3)
    
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    with col_test:
        if st.button("🧪 Test", use_container_width=True):
            st.session_state.query_input = "Calculate 2 + 2"

# Information and testing section
with col2:
    with st.expander("🔧 System Information", expanded=False):
        st.markdown("""
        **✅ Reliable MCP Demo Features:**
        
        🔧 **Simplified LLM Wrapper:**
        - Removes complex tool binding
        - Focuses on basic Snowflake Cortex
        - Better error handling
        
        🤖 **Robust Response Extraction:**
        - Multiple extraction methods
        - Handles AddableValuesDict properly
        - Detailed debugging information
        
        📊 **Real-time Status:**
        - MCP server connectivity
        - Tool availability
        - Model readiness
        
        🛠️ **Available Tools:**
        - Calculator: Basic math operations
        - SerpApiSearch: Web search
        - Weather: Location-based forecasts
        """)
    
    with st.expander("🧪 Test Connection", expanded=False):
        if st.button("Test MCP Connection", use_container_width=True):
            async def test_connection():
                try:
                    async with sse_client(url=server_url) as sse_connection:
                        async with ClientSession(*sse_connection) as session:
                            await session.initialize()
                            
                            # Test tools
                            tools = await session.list_tools() 
                            tool_names = [t.name for t in tools.tools if t.name not in {"add-frequent-questions", "add-prompts", "suggested_top_prompts"}]
                            
                            # Test prompts
                            prompts = await session.list_prompts()
                            prompt_names = [p.name for p in prompts.prompts]
                            
                            return True, tool_names, prompt_names
                except Exception as e:
                    return False, str(e), []
            
            success, tools_or_error, prompts = asyncio.run(test_connection())
            
            if success:
                st.success("✅ Connection successful!")
                st.write(f"**Tools:** {', '.join(tools_or_error)}")
                st.write(f"**Prompts:** {', '.join(prompts)}")
            else:
                st.error(f"❌ Connection failed: {tools_or_error}")

# Footer
st.markdown("---")
col1_footer, col2_footer, col3_footer = st.columns(3)

with col1_footer:
    st.markdown("**🔧 Simplified:** Reliable LLM wrapper")

with col2_footer:
    st.markdown("**🤖 Robust:** Advanced response extraction")

with col3_footer:
    st.markdown("**✅ Tested:** Calculator, Search, Weather")
