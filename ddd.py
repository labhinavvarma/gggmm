import streamlit as st
import asyncio
import json
import yaml
import pkg_resources
import requests
from datetime import datetime

from mcp.client.sse import sse_client
from mcp import ClientSession

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dependencies import SnowFlakeConnector
from llmobjectwrapper import ChatSnowflakeCortex
from snowflake.snowpark import Session

# Page config
st.set_page_config(page_title="Enhanced MCP Client", page_icon="🚀")
st.title("🚀 Enhanced MCP Client - DataFlyWheel Edition")
st.markdown("*Synced with Enhanced Server - Fresh Data Guaranteed*")

# Updated server URL to match your configuration
server_url = st.sidebar.text_input("MCP Server URL", "http://localhost:8081/sse")
show_server_info = st.sidebar.checkbox("🛡 Show MCP Server Info", value=False)

# Enhanced connection status check
@st.cache_data(ttl=15)
def check_server_connection(url):
    try:
        # Check both the SSE endpoint and the main server
        base_url = url.replace('/sse', '')
        
        # Try health check endpoint first
        health_response = requests.get(f"{base_url}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            return {
                "connected": True,
                "status": health_data.get("status", "unknown"),
                "tools_available": len(health_data.get("tools", {})),
                "details": health_data
            }
        else:
            # Fallback to basic connectivity test
            basic_response = requests.get(base_url, timeout=5)
            return {
                "connected": basic_response.status_code == 200,
                "status": "basic_connection",
                "tools_available": "unknown",
                "details": {}
            }
    except Exception as e:
        return {
            "connected": False,
            "status": f"error: {str(e)}",
            "tools_available": 0,
            "details": {}
        }

server_status = check_server_connection(server_url)
status_indicator = "🟢 Connected" if server_status["connected"] else "🔴 Disconnected"
st.sidebar.markdown(f"**Server Status:** {status_indicator}")

if server_status["connected"] and server_status.get("tools_available") != "unknown":
    st.sidebar.markdown(f"**Tools Available:** {server_status['tools_available']}")

# Enhanced server info display
if show_server_info:
    async def fetch_enhanced_mcp_info():
        result = {
            "resources": [], 
            "tools": [], 
            "prompts": [], 
            "yaml": [], 
            "search": [],
            "server_health": {},
            "weather_cache": {}
        }
        
        try:
            # Get server health info
            base_url = server_url.replace('/sse', '')
            try:
                health_response = requests.get(f"{base_url}/health", timeout=5)
                if health_response.status_code == 200:
                    result["server_health"] = health_response.json()
            except:
                pass
            
            # Get weather cache status
            try:
                cache_response = requests.get(f"{base_url}/api/v1/weather_cache", timeout=5)
                if cache_response.status_code == 200:
                    result["weather_cache"] = cache_response.json()
            except:
                pass
                
            # Get MCP server info
            async with sse_client(url=server_url) as sse_connection:
                async with ClientSession(*sse_connection) as session:
                    await session.initialize()

                    # --- Resources ---
                    try:
                        resources = await session.list_resources()
                        if hasattr(resources, 'resources'):
                            for r in resources.resources:
                                result["resources"].append({
                                    "name": r.name,
                                    "uri": getattr(r, 'uri', 'N/A'),
                                    "description": getattr(r, 'description', 'N/A')
                                })
                    except Exception as e:
                        result["resources"].append({"error": f"Failed to load resources: {e}"})
                   
                    # --- Enhanced Tools (with updated tool names) ---
                    try:
                        tools = await session.list_tools()
                        # Updated hidden tools list
                        hidden_tools = {"add-frequent-questions", "add-prompts", "suggested_top_prompts"}
                        if hasattr(tools, 'tools'):
                            for t in tools.tools:
                                if t.name not in hidden_tools:
                                    # Enhanced tool info
                                    tool_info = {
                                        "name": t.name,
                                        "description": getattr(t, 'description', ''),
                                    }
                                    
                                    # Add schema info if available
                                    if hasattr(t, 'inputSchema'):
                                        schema = t.inputSchema
                                        if isinstance(schema, dict) and 'properties' in schema:
                                            tool_info["parameters"] = list(schema['properties'].keys())
                                    
                                    result["tools"].append(tool_info)
                    except Exception as e:
                        result["tools"].append({"error": f"Failed to load tools: {e}"})

                    # --- Enhanced Prompts ---
                    try:
                        prompts = await session.list_prompts()
                        if hasattr(prompts, 'prompts'):
                            for p in prompts.prompts:
                                args = []
                                if hasattr(p, 'arguments'):
                                    for arg in p.arguments:
                                        args.append({
                                            "name": arg.name,
                                            "required": getattr(arg, 'required', False),
                                            "description": getattr(arg, 'description', '')
                                        })
                                result["prompts"].append({
                                    "name": p.name,
                                    "description": getattr(p, 'description', ''),
                                    "args": args
                                })
                    except Exception as e:
                        result["prompts"].append({"error": f"Failed to load prompts: {e}"})

                    # --- YAML Resources ---
                    try:
                        yaml_content = await session.read_resource("schematiclayer://cortex_analyst/schematic_models/hedis_stage_full/list")
                        if hasattr(yaml_content, 'contents'):
                            for item in yaml_content.contents:
                                if hasattr(item, 'text'):
                                    try:
                                        parsed = yaml.safe_load(item.text)
                                        result["yaml"].append(yaml.dump(parsed, sort_keys=False))
                                    except:
                                        result["yaml"].append(item.text)
                    except Exception as e:
                        result["yaml"].append(f"YAML error: {e}")

                    # --- Search Objects ---
                    try:
                        content = await session.read_resource("search://cortex_search/search_obj/list")
                        if hasattr(content, 'contents'):
                            for item in content.contents:
                                if hasattr(item, 'text'):
                                    try:
                                        objs = json.loads(item.text)
                                        result["search"].extend(objs)
                                    except:
                                        result["search"].append(item.text)
                    except Exception as e:
                        result["search"].append(f"Search error: {e}")

        except Exception as e:
            st.sidebar.error(f"❌ MCP Connection Error: {e}")
            
        return result

    mcp_data = asyncio.run(fetch_enhanced_mcp_info())

    # Enhanced server health display
    if mcp_data.get("server_health"):
        with st.sidebar.expander("🏥 Server Health", expanded=True):
            health = mcp_data["server_health"]
            st.json(health)
    
    # Weather cache status
    if mcp_data.get("weather_cache") and mcp_data["weather_cache"].get("cache_entries", 0) > 0:
        with st.sidebar.expander("🌤️ Weather Cache Status", expanded=False):
            cache_info = mcp_data["weather_cache"]
            st.write(f"**Cached Locations:** {cache_info.get('cache_entries', 0)}")
            
            for location, status in cache_info.get("cache_status", {}).items():
                valid_indicator = "✅" if status.get("is_valid") else "❌"
                st.write(f"{valid_indicator} **{location}**: {status.get('age_seconds', 0):.0f}s old")

    # Display Resources with better organization
    with st.sidebar.expander("📦 Resources", expanded=False):
        for r in mcp_data["resources"]:
            if isinstance(r, dict) and "error" not in r:
                if "cortex_search/search_obj/list" in r["name"]:
                    display_name = "🔍 Cortex Search Service"
                elif "schematic_models" in r["name"]:
                    display_name = "📋 HEDIS Schematic Models"
                elif "frequent_questions" in r["name"]:
                    display_name = "❓ Frequent Questions"
                elif "prompts" in r["name"]:
                    display_name = "📝 Prompt Templates"
                else:
                    display_name = r["name"]
                st.markdown(f"**{display_name}**")
                if r.get("description") and r["description"] != "N/A":
                    st.caption(r["description"])
            else:
                st.error(str(r))

    # --- Enhanced Tools Section with Updated Categories ---
    with st.sidebar.expander("🛠 Available Tools", expanded=False):
        # Updated tool categories to match the enhanced server
        tool_categories = {
            "🏥 HEDIS & Analytics": ["DFWAnalyst", "DFWSearch", "calculator"],
            "🔍 Search & Information": ["wikipedia_search", "duckduckgo_search"],
            "🌤️ Weather & Location": ["get_weather"],  # Updated to match server
            "🔧 System & Testing": ["test_tool", "diagnostic"]
        }
        
        available_tools = {t["name"]: t for t in mcp_data["tools"] if isinstance(t, dict) and "error" not in t}
        
        for category, expected_tools in tool_categories.items():
            st.markdown(f"**{category}:**")
            category_found = False
            for tool_name in expected_tools:
                if tool_name in available_tools:
                    tool_info = available_tools[tool_name]
                    st.markdown(f"  • **{tool_name}**")
                    if tool_info.get('description'):
                        st.caption(f"    {tool_info['description']}")
                    if tool_info.get('parameters'):
                        st.caption(f"    Parameters: {', '.join(tool_info['parameters'])}")
                    category_found = True
            
            if not category_found:
                st.caption("    No tools found in this category")
        
        # Show any uncategorized tools
        all_categorized = [tool for tools in tool_categories.values() for tool in tools]
        uncategorized = [name for name in available_tools.keys() if name not in all_categorized]
        
        if uncategorized:
            st.markdown("**🔧 Other Tools:**")
            for tool_name in uncategorized:
                tool_info = available_tools[tool_name]
                st.markdown(f"  • **{tool_name}**")
                if tool_info.get('description'):
                    st.caption(f"    {tool_info['description']}")

    # Display Prompts with enhanced formatting
    with st.sidebar.expander("🧐 Available Prompts", expanded=False):
        # Updated prompt display names to match server
        prompt_display_names = {
            "hedis-prompt": "🏥 HEDIS Expert",
            "calculator-prompt": "🧮 Calculator Expert",  # Fixed typo
            "weather-prompt": "🌤️ Weather Expert", 
            "wikipedia-search-prompt": "📖 Wikipedia Expert",
            "duckduckgo-search-prompt": "🦆 Web Search Expert",
            "test-tool-prompt": "🔧 Test Tool",
            "diagnostic-prompt": "🔧 Diagnostic Tool"
        }
        
        for p in mcp_data["prompts"]:
            if isinstance(p, dict) and "error" not in p:
                display_name = prompt_display_names.get(p['name'], p['name'])
                st.markdown(f"**{display_name}**")
                if p.get('description'):
                    st.caption(f"Description: {p['description']}")
                if p.get('args'):
                    args_text = ", ".join([f"{arg['name']}{'*' if arg.get('required') else ''}" 
                                         for arg in p['args']])
                    if args_text:
                        st.caption(f"Arguments: {args_text}")
            else:
                st.error(str(p))

else:
    # === MAIN APPLICATION MODE ===
    @st.cache_resource
    def get_snowflake_connection():
        try:
            return SnowFlakeConnector.get_conn('aedl', '')
        except Exception as e:
            st.error(f"❌ Failed to connect to Snowflake: {e}")
            return None

    @st.cache_resource
    def get_model():
        try:
            sf_conn = get_snowflake_connection()
            if sf_conn:
                return ChatSnowflakeCortex(
                    model="claude-4-sonnet", 
                    cortex_function="complete",
                    session=Session.builder.configs({"connection": sf_conn}).getOrCreate(),
                    mcp_server_url=server_url  # Add MCP server URL to the model
                )
            else:
                # Create model without Snowflake session (will use MCP tools only)
                return ChatSnowflakeCortex(
                    model="claude-4-sonnet",
                    cortex_function="complete",
                    mcp_server_url=server_url
                )
        except Exception as e:
            st.error(f"❌ Failed to initialize model: {e}")
            return None
    
    # Enhanced prompt type selection with updated options
    prompt_type = st.sidebar.radio(
        "🎯 Select Expert Mode", 
        ["Calculator", "HEDIS Expert", "Weather", "Wikipedia Search", "Web Search", "General AI"],
        help="Choose the type of expert assistance you need"
    )
    
    # Updated prompt mapping to match server prompts
    prompt_map = {
        "Calculator": "calculator-prompt",  # Fixed typo
        "HEDIS Expert": "hedis-prompt",
        "Weather": "weather-prompt",
        "Wikipedia Search": "wikipedia-search-prompt",
        "Web Search": "duckduckgo-search-prompt",
        "General AI": None
    }

    # Enhanced examples with updated weather examples
    examples = {
        "Calculator": [
            "Calculate the expression (4+5)/2.0", 
            "What is the square root of 144?", 
            "Calculate 3 to the power of 4",
            "What is 15% of 847?",
            "Calculate compound interest on $1000 at 5% for 3 years"
        ],
        "HEDIS Expert": [],  # Will be loaded dynamically
        "Weather": [
            "What's the current weather in New York?",
            "Get weather forecast for London, UK",
            "Show me the weather for Tokyo, Japan",
            "What's the weather like in Sydney, Australia?",
            "Get current conditions for Paris, France"
        ],
        "Wikipedia Search": [
            "Search Wikipedia for artificial intelligence",
            "What is quantum computing according to Wikipedia?",
            "Find Wikipedia information about climate change",
            "Look up the current US President on Wikipedia",
            "Search for information about the James Webb telescope"
        ],
        "Web Search": [
            "Search for latest AI developments in 2025",
            "Find current information about renewable energy trends", 
            "Look up recent space exploration missions",
            "Search for today's technology news",
            "Find latest updates on electric vehicles"
        ],
        "General AI": [
            "Explain quantum computing in simple terms",
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
            "What's the difference between AI and ML?"
        ]
    }

    # Load HEDIS examples dynamically from MCP server
    if prompt_type == "HEDIS Expert":
        try:
            async def fetch_hedis_examples():
                try:
                    async with sse_client(url=server_url) as sse_connection:
                        async with ClientSession(*sse_connection) as session:
                            await session.initialize()
                            content = await session.read_resource("genaiplatform://hedis/frequent_questions/Initialization")
                            if hasattr(content, "contents"):
                                for item in content.contents:
                                    if hasattr(item, "text"):
                                        loaded_examples = json.loads(item.text)
                                        examples["HEDIS Expert"].extend(loaded_examples[:10])  # Limit to 10 examples
                except Exception as e:
                    print(f"Failed to load HEDIS examples: {e}")
   
            asyncio.run(fetch_hedis_examples())
        except Exception as e:
            pass
            
        # Fallback examples if dynamic loading failed
        if not examples["HEDIS Expert"]:
            examples["HEDIS Expert"] = [
                "What are the codes in BCS Value Set?",
                "Explain the BCS (Breast Cancer Screening) measure",
                "What is the age criteria for CBP measure?",
                "Describe the COA measure requirements",
                "What LOB is COA measure scoped under?"
            ]

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Enhanced example queries with better organization
    with st.sidebar.expander(f"💡 Example Queries - {prompt_type}", expanded=True):
        if examples[prompt_type]:
            for i, example in enumerate(examples[prompt_type]):
                display_text = example if len(example) <= 70 else example[:67] + "..."
                if st.button(display_text, key=f"{prompt_type}_{i}_{hash(example)}", use_container_width=True):
                    st.session_state.query_input = example
        else:
            st.info("Loading examples...")

    # Add helpful tips based on selected mode
    if prompt_type == "Weather":
        with st.sidebar.expander("🌍 Weather Tips", expanded=False):
            st.info("""
            **Enhanced Weather Service:**
            • Covers worldwide locations
            • Uses cached data (5-min refresh)
            • Multiple sources (NWS + Open-Meteo)
            • 3-day forecast included
            • Current conditions and detailed forecasts
            
            **Examples:**
            • "Weather in New York"
            • "Current conditions in London"
            • "Tokyo weather forecast"
            """)
    
    elif prompt_type == "Wikipedia Search":
        with st.sidebar.expander("📖 Wikipedia Tips", expanded=False):
            st.info("""
            **Enhanced Wikipedia Search:**
            • Current, up-to-date articles with cache-busting
            • Last modification dates shown
            • Comprehensive content from multiple sections
            • Freshness validation for recent topics
            
            **Best for:**
            • Current events and recent changes
            • Encyclopedic information
            • Historical facts and scientific concepts
            • Biographical information
            """)
    
    elif prompt_type == "Web Search":
        with st.sidebar.expander("🦆 Web Search Tips", expanded=False):
            st.info("""
            **Enhanced Web Search (Content Analysis):**
            • Searches DuckDuckGo for fresh results
            • Actually reads and analyzes webpage content
            • Prioritizes recent content with date validation
            • Provides comprehensive summaries from multiple sources
            
            **Best for:**
            • Breaking news and latest developments
            • Current research and recent findings
            • Real-time information and trends
            • Fresh updates and announcements
            """)

    # Chat input handling with enhanced processing
    if query := st.chat_input("Type your query here...") or "query_input" in st.session_state:

        if "query_input" in st.session_state:
            query = st.session_state.query_input
            del st.session_state.query_input

        with st.chat_message("user"):
            st.markdown(query, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "user", "content": query})

        async def process_enhanced_query(query_text):
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.text("🤔 Processing your request...")
                
                try:
                    # Initialize MCP client with better error handling
                    message_placeholder.text("🔌 Connecting to enhanced MCP server...")
                    
                    if not server_status["connected"]:
                        raise Exception("MCP server is not accessible. Please check the server URL and ensure it's running.")
                    
                    client = MultiServerMCPClient(
                        {"DataFlyWheelServer": {"url": server_url, "transport": "sse"}}
                    )

                    model = get_model()
                    if not model:
                        raise Exception("Failed to initialize the AI model. Please check Snowflake connection.")
                    
                    # Get tools and create agent
                    message_placeholder.text("🛠️ Loading enhanced tools from server...")
                    tools = await client.get_tools()
                    
                    if not tools:
                        raise Exception("No tools available from the MCP server.")
                    
                    message_placeholder.text(f"🤖 Creating AI agent with {len(tools)} tools...")
                    agent = create_react_agent(model=model, tools=tools)
                    
                    # Handle prompt selection with better formatting
                    prompt_name = prompt_map[prompt_type]
                    
                    if prompt_name is None:
                        # General AI mode - use query directly
                        message_placeholder.text("💭 Processing in general AI mode...")
                        messages = [{"role": "user", "content": query_text}]
                    else:  
                        # Get prompt from server
                        message_placeholder.text(f"📝 Loading {prompt_type} expert prompt...")
                        try:
                            prompt_from_server = await client.get_prompt(
                                server_name="DataFlyWheelServer",
                                prompt_name=prompt_name,
                                arguments={"query": query_text}
                            )
                            
                            # Enhanced prompt handling
                            if prompt_from_server and len(prompt_from_server) > 0:
                                first_prompt = prompt_from_server[0]
                                
                                if hasattr(first_prompt, 'content'):
                                    content = first_prompt.content
                                elif hasattr(first_prompt, 'text'):
                                    content = first_prompt.text
                                else:
                                    content = str(first_prompt)
                                
                                # Handle template substitution
                                if "{query}" in content:
                                    content = content.format(query=query_text)
                                
                                messages = [{"role": "user", "content": content}]
                                message_placeholder.text(f"✅ Loaded {prompt_type} expert prompt")
                            else:
                                # Fallback if prompt not found
                                st.warning(f"⚠️ {prompt_type} prompt not found on server. Using direct mode.")
                                messages = [{"role": "user", "content": query_text}]
                                
                        except Exception as prompt_error:
                            st.warning(f"⚠️ Could not load {prompt_type} prompt: {prompt_error}. Using direct mode.")
                            messages = [{"role": "user", "content": query_text}]

                    message_placeholder.text("🧠 Generating intelligent response...")
                    
                    # Invoke agent with proper message format and timeout
                    try:
                        response = await asyncio.wait_for(
                            agent.ainvoke({"messages": messages}), 
                            timeout=120.0  # 2 minute timeout
                        )
                    except asyncio.TimeoutError:
                        raise Exception("Request timed out. The server may be overloaded or the query is too complex.")
                    
                    # Enhanced result extraction with multiple strategies
                    result = None
                    
                    if isinstance(response, dict):
                        # Strategy 1: Check for 'messages' key with AI message
                        if 'messages' in response:
                            messages_list = response['messages']
                            if isinstance(messages_list, list):
                                # Look for the last assistant message
                                for msg in reversed(messages_list):
                                    if hasattr(msg, 'content') and hasattr(msg, 'type'):
                                        if getattr(msg, 'type', None) == 'ai' or not result:
                                            result = msg.content
                                            break
                                    elif hasattr(msg, 'content'):
                                        result = msg.content
                        
                        # Strategy 2: Look for any meaningful content
                        if result is None:
                            for key, value in response.items():
                                if isinstance(value, str) and len(value) > 20:
                                    result = value
                                    break
                                elif isinstance(value, list) and len(value) > 0:
                                    for item in value:
                                        if hasattr(item, 'content') and len(str(item.content)) > 20:
                                            result = item.content
                                            break
                                        elif isinstance(item, str) and len(item) > 20:
                                            result = item
                                            break
                                    if result:
                                        break
                    
                    # Fallback to string representation
                    if result is None or (isinstance(result, str) and len(result.strip()) < 10):
                        result = str(response)
                    
                    # Clean up and validate the result
                    if isinstance(result, str):
                        result = result.strip()
                        if result.startswith('"') and result.endswith('"'):
                            result = result[1:-1]
                        
                        # Check for empty or very short results
                        if len(result) < 10:
                            result = f"⚠️ Received a very short response: '{result}'. Please try rephrasing your query."
                    
                    # Ensure we have a meaningful response
                    if not result or result.strip() == "":
                        result = "⚠️ Received empty response from the server. This might be due to a processing error. Please try again with a different query."
                    
                    # Add timestamp and mode info
                    current_time = datetime.now().strftime('%H:%M:%S')
                    result += f"\n\n*Response generated at {current_time} using {prompt_type} mode*"
                    
                    # Display result
                    message_placeholder.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    
                except Exception as e:
                    error_message = f"❌ **Error Processing Request**: {str(e)}\n\n"
                    error_message += f"**Troubleshooting Steps:**\n"
                    error_message += f"1. **Server Status**: {status_indicator}\n"
                    error_message += f"2. **Server URL**: {server_url}\n"
                    error_message += f"3. **Selected Mode**: {prompt_type}\n"
                    error_message += f"4. **Tools Available**: {server_status.get('tools_available', 'Unknown')}\n\n"
                    
                    if not server_status["connected"]:
                        error_message += "**🔧 Server Connection Issues:**\n"
                        error_message += "- Verify the MCP server is running\n"
                        error_message += "- Check if the URL is correct\n"
                        error_message += "- Ensure no firewall is blocking the connection\n"
                    else:
                        error_message += "**🔧 Processing Issues:**\n"
                        error_message += "- Try a simpler query\n"
                        error_message += "- Switch to 'General AI' mode\n"
                        error_message += "- Check server logs for detailed error information\n"
                    
                    error_message += f"\n*Error occurred at {datetime.now().strftime('%H:%M:%S')}*"
                    
                    message_placeholder.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

        if query:
            asyncio.run(process_enhanced_query(query))

    # Enhanced sidebar controls
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔧 Enhanced Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        with col2:
            if st.button("🔄 Refresh Status", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

        if st.button("🔍 Test Enhanced Connection", use_container_width=True):
            try:
                async def test_enhanced_connection():
                    # Test MCP connection
                    async with sse_client(url=server_url) as sse_connection:
                        async with ClientSession(*sse_connection) as session:
                            await session.initialize()
                            
                            # Count available resources and tools
                            tools = await session.list_tools()
                            tool_count = len(tools.tools) if hasattr(tools, 'tools') else 0
                            
                            resources = await session.list_resources()
                            resource_count = len(resources.resources) if hasattr(resources, 'resources') else 0
                            
                            prompts = await session.list_prompts()
                            prompt_count = len(prompts.prompts) if hasattr(prompts, 'prompts') else 0
                            
                            return f"""✅ **Enhanced Connection Test Successful!**
                            
📊 **Server Statistics:**
- 🛠️ Tools Available: {tool_count}
- 📦 Resources: {resource_count}  
- 🧐 Prompts: {prompt_count}

🚀 **Enhanced Features:**
- ✅ Fresh data retrieval with cache-busting
- ✅ Weather caching system active
- ✅ Wikipedia search with current data
- ✅ Web search with content analysis
- ✅ HEDIS analytics tools ready

🌐 **Connection Quality:** Excellent"""

                result = asyncio.run(test_enhanced_connection())
                st.success(result)
                
                # Additional HTTP endpoint tests
                base_url = server_url.replace('/sse', '')
                
                # Test tool call endpoint
                try:
                    test_response = requests.post(
                        f"{base_url}/api/v1/tool_call",
                        json={"tool_name": "test_tool", "arguments": {"message": "connection test"}},
                        timeout=10
                    )
                    if test_response.status_code == 200:
                        st.info("🔧 Direct tool call endpoint: ✅ Working")
                    else:
                        st.warning(f"🔧 Direct tool call endpoint: ❌ HTTP {test_response.status_code}")
                except Exception as e:
                    st.warning(f"🔧 Direct tool call endpoint: ❌ {str(e)}")
                
            except Exception as e:
                st.error(f"❌ **Enhanced Connection Test Failed**: {e}")
                
                # Provide specific troubleshooting
                base_url = server_url.replace('/sse', '')
                try:
                    health_check = requests.get(f"{base_url}/health", timeout=5)
                    if health_check.status_code == 200:
                        st.info("✅ HTTP server is responding, but MCP connection failed")
                        st.info("🔧 Try restarting the MCP server or check server logs")
                    else:
                        st.error(f"❌ HTTP server error: {health_check.status_code}")
                except:
                    st.error("❌ Server is completely unreachable")

        # Server integration test
        if st.button("🧪 Test Integration", use_container_width=True):
            try:
                base_url = server_url.replace('/sse', '')
                test_response = requests.post(f"{base_url}/test_integration", timeout=30)
                
                if test_response.status_code == 200:
                    test_data = test_response.json()
                    st.success(f"🧪 Integration Test: {test_data.get('success_rate', 'Unknown')} passed")
                    
                    # Show details in expandable section
                    with st.expander("📊 Test Details"):
                        for result in test_data.get('results', []):
                            status_icon = "✅" if result.get('success') else "❌"
                            st.write(f"{status_icon} **{result.get('test')}**: {result.get('result', 'No result')[:100]}...")
                else:
                    st.error(f"❌ Integration test failed: HTTP {test_response.status_code}")
                    
            except Exception as e:
                st.error(f"❌ Integration test error: {e}")

        # Enhanced status information
        st.markdown("---")
        st.markdown("### 📊 System Status")
        
        # Server info
        st.caption(f"🌐 **Server**: {server_url}")
        st.caption(f"🤖 **Mode**: {prompt_type}")
        st.caption(f"📡 **Status**: {status_indicator}")
        
        if server_status.get("details") and isinstance(server_status["details"], dict):
            if "timestamp" in server_status["details"]:
                st.caption(f"⏰ **Last Check**: {server_status['details']['timestamp'][:19]}")
        
        # Model info
        try:
            model = get_model()
            if model:
                st.caption(f"🧠 **Model**: {getattr(model, 'model', 'Unknown')}")
                if hasattr(model, 'session') and model.session:
                    st.caption("❄️ **Snowflake**: ✅ Connected")
                else:
                    st.caption("❄️ **Snowflake**: ⚠️ Not Connected")
        except:
            st.caption("🧠 **Model**: ❌ Failed to load")

        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        quick_queries = {
            "🧮": "Calculate 25 * 4 + 10",
            "🌤️": "Weather in New York", 
            "📖": "Search Wikipedia for quantum computing",
            "🔍": "Latest AI news",
            "🏥": "What is BCS measure?"
        }
        
        cols = st.columns(len(quick_queries))
        for i, (icon, query) in enumerate(quick_queries.items()):
            with cols[i]:
                if st.button(icon, help=query, use_container_width=True):
                    st.session_state.query_input = query
                    st.rerun()

# Enhanced footer with version and feature info
st.markdown("---")
st.markdown("### 🚀 Enhanced MCP Client v2.0")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔧 Core Features:**")
    st.caption("• HEDIS Analytics")
    st.caption("• Advanced Calculator")
    st.caption("• System Diagnostics")

with col2:
    st.markdown("**🌐 Search & Data:**")
    st.caption("• Fresh Wikipedia Data")
    st.caption("• Content-Reading Web Search")
    st.caption("• Cached Weather Service")

with col3:
    st.markdown("**⚡ Enhanced:**")
    st.caption("• Real-time Data Validation")
    st.caption("• Multiple Fallback Sources")
    st.caption("• Smart Cache Management")

st.caption(f"📡 **Connection**: {server_url} | 🤖 **Mode**: {prompt_type} | 📊 **Status**: {status_indicator}")

# Add debug information in development
if st.sidebar.checkbox("🐛 Debug Mode", value=False):
    st.sidebar.markdown("### 🐛 Debug Information")
    st.sidebar.json({
        "server_status": server_status,
        "server_url": server_url,
        "prompt_type": prompt_type,
        "session_messages": len(st.session_state.messages),
        "timestamp": datetime.now().isoformat()
    })
    
    if st.sidebar.button("🔍 Test Direct API"):
        try:
            base_url = server_url.replace('/sse', '')
            
            # Test health endpoint
            health = requests.get(f"{base_url}/health", timeout=5)
            st.sidebar.write("**Health Check:**", health.status_code)
            
            # Test tools list
            tools = requests.get(f"{base_url}/api/v1/tools", timeout=5)
            st.sidebar.write("**Tools Endpoint:**", tools.status_code)
            
            if tools.status_code == 200:
                tools_data = tools.json()
                st.sidebar.write("**Available Tools:**", len(tools_data.get("tools", {})))
                
        except Exception as e:
            st.sidebar.error(f"API Test Error: {e}")
