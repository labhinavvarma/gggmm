import streamlit as st
import asyncio
import json
import yaml
import traceback

from mcp.client.sse import sse_client
from mcp import ClientSession

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dependencies import SnowFlakeConnector
from llmobjectwrapper import ChatSnowflakeCortex  # Use the fixed version
from snowflake.snowpark import Session

# Page config
st.set_page_config(page_title="MCP DEMO - Fixed")
st.title("MCP DEMO - Fixed")

server_url = st.sidebar.text_input("MCP Server URL", "http://10.126.192.183:8022/sse")
show_server_info = st.sidebar.checkbox("🛡 Show MCP Server Info", value=False)

# === Server Info Section ===
if show_server_info:
    async def fetch_mcp_info():
        result = {"resources": [], "tools": [], "prompts": [], "yaml": [], "search": []}
        try:
            async with sse_client(url=server_url) as sse_connection:
                async with ClientSession(*sse_connection) as session:
                    await session.initialize()

                    # --- Resources ---
                    resources = await session.list_resources()
                    if hasattr(resources, 'resources'):
                        for r in resources.resources:
                            result["resources"].append({"name": r.name})
                   
                    # --- Tools (filtered) ---
                    tools = await session.list_tools()
                    hidden_tools = {"add-frequent-questions", "add-prompts", "suggested_top_prompts"}
                    if hasattr(tools, 'tools'):
                        for t in tools.tools:
                            if t.name not in hidden_tools:
                                result["tools"].append({
                                    "name": t.name,
                                    "description": getattr(t, 'description', '')
                                })

                    # --- Prompts ---
                    prompts = await session.list_prompts()
                    if hasattr(prompts, 'prompts'):
                        for p in prompts.prompts:
                            args = []
                            if hasattr(p, 'arguments'):
                                for arg in p.arguments:
                                    args.append(f"{arg.name} ({'Required' if arg.required else 'Optional'}): {arg.description}")
                            result["prompts"].append({
                                "name": p.name,
                                "description": getattr(p, 'description', ''),
                                "args": args
                            })

        except Exception as e:
            st.sidebar.error(f"❌ MCP Connection Error: {e}")
        return result

    mcp_data = asyncio.run(fetch_mcp_info())

    # Display server info in sidebar
    with st.sidebar.expander("📦 Resources", expanded=False):
        for r in mcp_data["resources"]:
            if "cortex_search/search_obj/list" in r["name"]:
                display_name = "Cortex Search"
            else:
                display_name = r["name"]
            st.markdown(f"**{display_name}**")

    with st.sidebar.expander("🛠 Tools", expanded=False):
        for t in mcp_data["tools"]:
            st.markdown(f"**{t['name']}**")
            if t.get('description'):
                st.caption(t['description'][:100] + "..." if len(t['description']) > 100 else t['description'])

    with st.sidebar.expander("🧐 Prompts", expanded=False):
        for p in mcp_data["prompts"]:
            st.markdown(f"**{p['name']}**")

else:
    # === Main Application ===
    @st.cache_resource
    def get_snowflake_connection():
        return SnowFlakeConnector.get_conn('aedl', '')

    @st.cache_resource
    def get_model():
        sf_conn = get_snowflake_connection()
        return ChatSnowflakeCortex(
            model="claude-4-sonnet",
            cortex_function="complete",
            session=Session.builder.configs({"connection": sf_conn}).getOrCreate()
        )
    
    # Fixed prompt selection
    prompt_type = st.sidebar.radio("Select Prompt Type", [
        "Calculator", 
        "HEDIS Expert", 
        "Weather", 
        "Web Search",  # Added Web Search
        "No Context"
    ])
    
    # Fixed prompt mapping
    prompt_map = {
        "Calculator": "calculator-prompt",  # Fixed typo
        "HEDIS Expert": "hedis-prompt",
        "Weather": "weather-prompt",
        "Web Search": "serpapi-prompt",  # Added SerpApi
        "No Context": None
    }

    # Enhanced examples
    examples = {
        "Calculator": [
            "What is 15% of 85,000?",  # Main test case
            "Calculate the expression (4+5)/2.0", 
            "Calculate the math function sqrt(16) + 7", 
            "Calculate the expression 3^4 - 12"
        ],
        "Web Search": [
            "Who is the current prime minister of India?",  # SerpApi test
            "Latest AI news today",
            "Current stock price of Apple",
            "Recent developments in healthcare"
        ],
        "HEDIS Expert": [],
        "Weather": [
            "What is the present weather in Richmond?",
            "What's the weather forecast for Atlanta?",
            "Is it raining in New York City today?"
        ],
        "No Context": [
            "Who won the world cup in 2022?", 
            "Summarize climate change impact on oceans"
        ]
    }

    # Load HEDIS examples
    if prompt_type == "HEDIS Expert":
        try:
            async def fetch_hedis_examples():
                async with sse_client(url=server_url) as sse_connection:
                    async with ClientSession(*sse_connection) as session:
                        await session.initialize()
                        content = await session.read_resource("genaiplatform://hedis/frequent_questions/Initialization")
                        if hasattr(content, "contents"):
                            for item in content.contents:
                                if hasattr(item, "text"):
                                    examples["HEDIS Expert"].extend(json.loads(item.text))
            
            asyncio.run(fetch_hedis_examples())
        except Exception as e:
            examples["HEDIS Expert"] = [f"⚠️ Failed to load examples: {e}"]

    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Example queries
    with st.sidebar.expander("Example Queries", expanded=True):
        for example in examples[prompt_type]:
            if st.button(example, key=example):
                st.session_state.query_input = example

    # FIXED response extraction function
    def extract_response_from_langgraph(response):
        """
        ROBUST response extraction for LangGraph AddableValuesDict.
        Tries multiple methods to get the actual response content.
        """
        
        print(f"\n🔍 RESPONSE EXTRACTION:")
        print(f"   Response type: {type(response)}")
        print(f"   Response str length: {len(str(response))}")
        
        try:
            # Method 1: Try dictionary-like access for AddableValuesDict
            if hasattr(response, 'get') or hasattr(response, '__getitem__'):
                print("📋 Trying dictionary access...")
                
                for key in ['messages', 'output', 'result', 'content']:
                    try:
                        if key in response:
                            value = response[key]
                            print(f"🔍 Found key '{key}': {type(value)}")
                            
                            if key == 'messages' and isinstance(value, list) and value:
                                last_msg = value[-1]
                                print(f"🔍 Last message: {type(last_msg)}")
                                
                                if hasattr(last_msg, 'content'):
                                    result = last_msg.content
                                    print(f"✅ Extracted from {key}[{len(value)-1}].content")
                                    return result
                                elif isinstance(last_msg, dict) and 'content' in last_msg:
                                    result = last_msg['content']
                                    print(f"✅ Extracted from {key}[{len(value)-1}]['content']")
                                    return result
                                else:
                                    # Convert message object to string
                                    result = str(last_msg)
                                    if len(result) > 20 and 'object at 0x' not in result:
                                        print(f"✅ Converted message to string")
                                        return result
                            
                            elif isinstance(value, str) and value.strip():
                                print(f"✅ Extracted string from key '{key}'")
                                return value
                    except Exception as e:
                        print(f"⚠️ Error with key '{key}': {e}")
                        continue

            # Method 2: Try direct attributes
            if hasattr(response, 'messages'):
                print("📋 Trying direct messages attribute...")
                messages = response.messages
                if messages and len(messages) > 0:
                    last_msg = messages[-1]
                    if hasattr(last_msg, 'content'):
                        result = last_msg.content
                        print(f"✅ Extracted from messages attribute")
                        return result

            # Method 3: Iterate through response if possible
            if hasattr(response, 'items'):
                print("📋 Trying iteration...")
                for key, value in response.items():
                    if isinstance(value, list) and value:
                        for item in value:
                            if hasattr(item, 'content') and isinstance(item.content, str):
                                if len(item.content) > 10:
                                    print(f"✅ Found content in iteration")
                                    return item.content

            # Method 4: String parsing as fallback
            print("📋 Trying string parsing...")
            str_response = str(response)
            
            # Look for content patterns
            patterns = [
                r"content='([^']+)'",
                r'content="([^"]+)"',
                r"content: '([^']+)'",
                r'content: "([^"]+)"'
            ]
            
            for pattern in patterns:
                import re
                match = re.search(pattern, str_response)
                if match:
                    content = match.group(1)
                    if len(content) > 10:
                        print(f"✅ Extracted via regex: {pattern}")
                        return content

            print("❌ Could not extract response")
            return None

        except Exception as e:
            print(f"❌ Response extraction error: {e}")
            return None

    # MAIN CHAT INPUT
    if query := st.chat_input("Type your query here...") or "query_input" in st.session_state:

        if "query_input" in st.session_state:
            query = st.session_state.query_input
            del st.session_state.query_input
       
        with st.chat_message("user"):
            st.markdown(query, unsafe_allow_html=True)
       
        st.session_state.messages.append({"role": "user", "content": query})

        async def process_query_fixed(query_text):
            """FIXED query processing with robust response extraction."""
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.text("🔄 Processing...")
                
                try:
                    print(f"🚀 Starting query processing for: '{query_text}'")
                    
                    # Step 1: Initialize MCP client
                    message_placeholder.text("🔗 Connecting to MCP server...")
                    client = MultiServerMCPClient(
                        {"DataFlyWheelServer": {"url": server_url, "transport": "sse"}}
                    )
                    
                    # Step 2: Get model
                    message_placeholder.text("🧠 Loading model...")
                    model = get_model()
                    print("✅ Model loaded successfully")
                    
                    # Step 3: Get tools and create agent
                    message_placeholder.text("🛠️ Loading tools...")
                    tools = await client.get_tools()
                    print(f"✅ Loaded {len(tools)} tools")
                    
                    message_placeholder.text("🤖 Creating agent...")
                    agent = create_react_agent(model=model, tools=tools)
                    print("✅ Agent created successfully")
                    
                    # Step 4: Get prompt
                    message_placeholder.text("📋 Getting prompt...")
                    prompt_name = prompt_map[prompt_type]
                    
                    if prompt_name is None:
                        # No context mode
                        messages_for_agent = [{"role": "user", "content": query_text}]
                        print("💬 Using no context mode")
                    else:
                        # Get prompt from server
                        prompt_from_server = await client.get_prompt(
                            server_name="DataFlyWheelServer",
                            prompt_name=prompt_name,
                            arguments={"query": query_text}
                        )
                        messages_for_agent = prompt_from_server
                        print(f"✅ Got prompt '{prompt_name}' with {len(messages_for_agent)} messages")
                    
                    # Step 5: Run agent
                    message_placeholder.text("🚀 AI processing...")
                    response = await agent.ainvoke({"messages": messages_for_agent})
                    print("✅ Agent completed processing")
                    
                    # Step 6: ROBUST response extraction
                    message_placeholder.text("📤 Extracting response...")
                    result = extract_response_from_langgraph(response)
                    
                    if result and isinstance(result, str) and result.strip():
                        # Success!
                        message_placeholder.markdown(result)
                        st.session_state.messages.append({"role": "assistant", "content": result})
                        print(f"✅ SUCCESS: Extracted {len(result)} character response")
                        
                    else:
                        # Detailed failure information
                        debug_msg = f"""
⚠️ **Response Extraction Failed**

**Query:** "{query_text}"
**Mode:** {prompt_type}

**Technical Details:**
- Response Type: `{type(response)}`
- Response String Length: {len(str(response))}
- Extraction Result: `{result}`

**Raw Response Sample:**
```
{str(response)[:600]}{'...' if len(str(response)) > 600 else ''}
```

**Next Steps:**
1. Try a simpler query like "2+2"
2. Check server logs for errors
3. Verify the tool is working

**Note:** The AI agent ran but response extraction failed. This is likely a format issue, not a functional problem.
"""
                        message_placeholder.markdown(debug_msg)
                        st.session_state.messages.append({"role": "assistant", "content": debug_msg})
                        print("❌ Response extraction failed")

                except Exception as e:
                    error_message = f"""
❌ **Processing Error**

**Error:** {str(e)}

**Query:** "{query_text}"
**Mode:** {prompt_type}
**Server:** {server_url}

**Troubleshooting:**
1. Verify MCP server is running
2. Check Snowflake connection
3. Try restarting server
4. Use simpler query

**Debug Info:**
```
{traceback.format_exc()}
```
"""
                    message_placeholder.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    print(f"❌ Processing error: {e}")

        if query:
            asyncio.run(process_query_fixed(query))

    # Controls
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()  # Fixed deprecated function

    # Status indicator
    st.sidebar.markdown("---")
    
    async def check_status():
        try:
            async with sse_client(url=server_url) as sse_connection:
                async with ClientSession(*sse_connection) as session:
                    await session.initialize()
                    return "🟢 Online"
        except:
            return "🔴 Offline"
    
    try:
        status = asyncio.run(check_status())
        st.sidebar.markdown(f"**Server Status:** {status}")
    except:
        st.sidebar.markdown("**Server Status:** ⚠️ Unknown")

# Footer
st.markdown("---")
st.markdown("**🔧 Fixed Version:** Improved response extraction and tool binding")
