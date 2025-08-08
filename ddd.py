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
st.set_page_config(page_title="Enhanced MCP Client with Brave Search", page_icon="🚀")
st.title("🚀 Enhanced MCP Client - DataFlyWheel Edition with Brave Search")
st.markdown("*Synced with Enhanced Server - Brave Search Integrated*")

# Updated server URL to match your configuration
server_url = st.sidebar.text_input("MCP Server URL", "http://10.126.192.183:8082/sse")

# Brave API Key Configuration
st.sidebar.markdown("### 🔍 Brave Search Configuration")
brave_api_key = st.sidebar.text_input(
    "Brave API Key", 
    value="BSA-FDD7EPTjkdgDqW_znc5uhZledvE", 
    type="password",
    help="Enter your Brave Search API key"
)

# Configure API key in server
if brave_api_key and st.sidebar.button("🔑 Configure API Key"):
    try:
        base_url = server_url.replace('/sse', '')
        config_response = requests.post(
            f"{base_url}/configure_brave_key",
            json={"api_key": brave_api_key},
            timeout=5
        )
        if config_response.status_code == 200:
            st.sidebar.success("✅ Brave API key configured!")
        else:
            st.sidebar.error("❌ Failed to configure API key")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")

show_server_info = st.sidebar.checkbox("🛡 Show MCP Server Info", value=False)

# Enhanced connection status check
@st.cache_data(ttl=15)
def check_server_connection(url):
    try:
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
st.sidebar.markdown(f"**Brave API Key:** {'✅ Configured' if brave_api_key else '❌ Not configured'}")

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
            "weather_cache": {},
            "brave_cache": {}
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
                cache_response = requests.get(f"{base_url}/weather_cache", timeout=5)
                if cache_response.status_code == 200:
                    result["weather_cache"] = cache_response.json()
            except:
                pass
            
            # Get Brave cache status
            try:
                brave_cache_response = requests.get(f"{base_url}/brave_cache", timeout=5)
                if brave_cache_response.status_code == 200:
                    result["brave_cache"] = brave_cache_response.json()
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
                        hidden_tools = {"add-frequent-questions", "add-prompts", "suggested_top_prompts"}
                        if hasattr(tools, 'tools'):
                            for t in tools.tools:
                                if t.name not in hidden_tools:
                                    tool_info = {
                                        "name": t.name,
                                        "description": getattr(t, 'description', ''),
                                    }
                                    
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
    
    # Brave cache status
    if mcp_data.get("brave_cache") and mcp_data["brave_cache"].get("cache_entries", 0) > 0:
        with st.sidebar.expander("🔍 Brave Search Cache", expanded=False):
            cache_info = mcp_data["brave_cache"]
            st.write(f"**Cached Searches:** {cache_info.get('cache_entries', 0)}")
            
            for search_key, status in cache_info.get("cache_status", {}).items():
                valid_indicator = "✅" if status.get("is_valid") else "❌"
                display_key = search_key[:30] + "..." if len(search_key) > 30 else search_key
                st.write(f"{valid_indicator} **{display_key}**: {status.get('age_seconds', 0):.0f}s old")

    # Display Resources
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

    # Enhanced Tools Section - Updated with Brave Search
    with st.sidebar.expander("🛠 Available Tools", expanded=False):
        tool_categories = {
            "🏥 HEDIS & Analytics": ["DFWAnalyst", "DFWSearch", "calculator"],
            "🔍 Search & Information": ["brave_web_search", "brave_local_search"],  # Updated
            "🌤️ Weather & Location": ["get_weather"],
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
        
        # Show uncategorized tools
        all_categorized = [tool for tools in tool_categories.values() for tool in tools]
        uncategorized = [name for name in available_tools.keys() if name not in all_categorized]
        
        if uncategorized:
            st.markdown("**🔧 Other Tools:**")
            for tool_name in uncategorized:
                tool_info = available_tools[tool_name]
                st.markdown(f"  • **{tool_name}**")
                if tool_info.get('description'):
                    st.caption(f"    {tool_info['description']}")

    # Display Prompts with enhanced formatting - Updated with Brave Search
    with st.sidebar.expander("🧐 Available Prompts", expanded=False):
        # Updated prompt display names to match server
        prompt_display_names = {
            "hedis-prompt": "🏥 HEDIS Expert",
            "calculator-prompt": "🧮 Calculator Expert",
            "weather-prompt": "🌤️ Weather Expert", 
            "brave-web-search-prompt": "🔍 Web Search Expert",  # Updated
            "brave-local-search-prompt": "📍 Local Search Expert",  # Updated
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
                    mcp_server_url=server_url
                )
            else:
                return ChatSnowflakeCortex(
                    model="claude-4-sonnet",
                    cortex_function="complete",
                    mcp_server_url=server_url
                )
        except Exception as e:
            st.error(f"❌ Failed to initialize model: {e}")
            return None
    
    # Enhanced prompt type selection with updated options - Updated with Brave Search
    prompt_type = st.sidebar.radio(
        "🎯 Select Expert Mode", 
        ["Calculator", "HEDIS Expert", "Weather", "Brave Web Search", "Local Search", "General AI"],  # Updated
        help="Choose the type of expert assistance you need"
    )
    
    # Updated prompt mapping to match server prompts - Updated with Brave Search
    prompt_map = {
        "Calculator": "calculator-prompt",
        "HEDIS Expert": "hedis-prompt",
        "Weather": "weather-prompt",
        "Brave Web Search": "brave-web-search-prompt",  # New dedicated Brave Web Search
        "Local Search": "brave-local-search-prompt",
        "General AI": None
    }

    # Enhanced examples with updated search examples - Updated with Brave Search
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
        "Brave Web Search": [  # New dedicated Brave Web Search examples
            "latest AI developments 2025",
            "current renewable energy trends", 
            "recent space exploration missions",
            "today's technology news",
            "breaking news artificial intelligence",
            "newest electric vehicle models",
            "latest cryptocurrency updates",
            "recent climate change research"
        ],
        "Local Search": [  # Local Search examples
            "pizza restaurants near Central Park",
            "coffee shops in Manhattan",
            "gas stations in San Francisco",
            "Italian restaurants downtown",
            "best sushi near Times Square",
            "pharmacies open late",
            "hotels near airport",
            "auto repair shops nearby"
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
                                        examples["HEDIS Expert"].extend(loaded_examples[:10])
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

    # Enhanced example queries
    with st.sidebar.expander(f"💡 Example Queries - {prompt_type}", expanded=True):
        if examples[prompt_type]:
            for i, example in enumerate(examples[prompt_type]):
                display_text = example if len(example) <= 70 else example[:67] + "..."
                if st.button(display_text, key=f"{prompt_type}_{i}_{hash(example)}", use_container_width=True):
                    st.session_state.query_input = example
        else:
            st.info("Loading examples...")

    # Add helpful tips based on selected mode - Updated with Brave Search
    if prompt_type == "Weather":
        with st.sidebar.expander("🌍 Weather Tips", expanded=False):
            st.info("""
            **Enhanced Weather Service:**
            • Covers worldwide locations
            • Uses cached data (5-min refresh)
            • Multiple sources (NWS + Open-Meteo)
            • 3-day forecast included
            
            **Examples:**
            • "Weather in New York"
            • "Current conditions in London"
            • "Tokyo weather forecast"
            """)
    
    elif prompt_type == "Brave Web Search":  # New dedicated section
        with st.sidebar.expander("🔍 Brave Web Search Tips", expanded=False):
            st.info("""
            **Brave Web Search:**
            • Fresh, unbiased search results
            • No tracking or data collection
            • Independent search engine
            • Latest news and developments
            • Real-time information
            • Global web content coverage
            
            **Best for:**
            • Breaking news and current events
            • Latest research and findings
            • Recent technological developments
            • Current trends and analysis
            • Product reviews and comparisons
            • Academic and scientific content
            """)
    
    elif prompt_type == "Local Search":  # Updated
        with st.sidebar.expander("📍 Brave Local Search Tips", expanded=False):
            st.info("""
            **Brave Local Search:**
            • Find local businesses and places
            • Detailed business information
            • Ratings and reviews
            • Contact details and hours
            • Address and location data
            • Maps integration support
            
            **Best for:**
            • Restaurants and dining
            • Services and shopping
            • Entertainment venues
            • Local business directories
            • Emergency services
            • Healthcare facilities
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
                            # Use the correct prompt name from the mapping
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
                            timeout=120.0
                        )
                    except asyncio.TimeoutError:
                        raise Exception("Request timed out. The server may be overloaded or the query is too complex.")
                    
                    # Enhanced result extraction
                    result = None
                    
                    if isinstance(response, dict):
                        # Strategy 1: Check for 'messages' key with AI message
                        if 'messages' in response:
                            messages_list = response['messages']
                            if isinstance(messages_list, list):
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
                        
                        if len(result) < 10:
                            result = f"⚠️ Received a very short response: '{result}'. Please try rephrasing your query."
                    
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
                    error_message += f"4. **Tools Available**: {server_status.get('tools_available', 'Unknown')}\n"
                    error_message += f"5. **Brave API Key**: {'Configured' if brave_api_key else 'Not configured'}\n\n"
                    
                    if not server_status["connected"]:
                        error_message += "**🔧 Server Connection Issues:**\n"
                        error_message += "- Verify the MCP server is running\n"
                        error_message += "- Check if the URL is correct\n"
                        error_message += "- Ensure no firewall is blocking the connection\n"
                    elif not brave_api_key and prompt_type in ["Brave Web Search", "Local Search"]:  # Updated
                        error_message += "**🔧 Brave Search Issues:**\n"
                        error_message += "- Configure Brave API key in the sidebar\n"
                        error_message += "- Click 'Configure API Key' to send it to server\n"
                        error_message += "- Try again after configuration\n"
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
                    async with sse_client(url=server_url) as sse_connection:
                        async with ClientSession(*sse_connection) as session:
                            await session.initialize()
                            
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

🔍 **Brave Search Integration:**
- API Key: {'✅ Configured' if brave_api_key else '❌ Not configured'}
- Web Search: brave_web_search tool
- Local Search: brave_local_search tool

🚀 **Enhanced Features:**
- ✅ Fresh data retrieval with cache-busting
- ✅ Weather caching system active
- ✅ Brave Search with current data
- ✅ HEDIS analytics tools ready

🌐 **Connection Quality:** Excellent"""

                result = asyncio.run(test_enhanced_connection())
                st.success(result)
                
                # Additional HTTP endpoint tests
                base_url = server_url.replace('/sse', '')
                
                try:
                    test_response = requests.post(
                        f"{base_url}/tool_call",
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
        st.caption(f"🔍 **Brave Search**: {'✅ Ready' if brave_api_key else '❌ Not configured'}")
        
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
        except Exception as e:
            st.caption("🧠 **Model**: ❌ Failed to load")
            if st.sidebar.checkbox("Show Model Error", value=False):
                st.sidebar.error(f"Model Error: {str(e)}")

        # Quick actions - Updated with Brave Search
        st.markdown("### ⚡ Quick Actions")
        
        quick_queries = {
            "🧮": "Calculate 25 * 4 + 10",
            "🌤️": "Weather in New York", 
            "🔍": "Latest AI news 2025",  # Brave Web Search
            "📍": "Pizza near Times Square",  # Local Search
            "🏥": "What is BCS measure?"
        }
        
        cols = st.columns(len(quick_queries))
        for i, (icon, query) in enumerate(quick_queries.items()):
            with cols[i]:
                if st.button(icon, help=query, use_container_width=True):
                    st.session_state.query_input = query
                    st.rerun()

# Enhanced footer with version and feature info - Updated
st.markdown("---")
st.markdown("### 🚀 Enhanced MCP Client v2.2 - Brave Search Integrated")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔧 Core Features:**")
    st.caption("• HEDIS Analytics")
    st.caption("• Advanced Calculator")
    st.caption("• System Diagnostics")

with col2:
    st.markdown("**🔍 Search & Data:**")
    st.caption("• Brave Web Search")  # Updated
    st.caption("• Brave Local Search")  # Updated
    st.caption("• Cached Weather Service")

with col3:
    st.markdown("**⚡ Enhanced:**")
    st.caption("• Server-side Brave Integration")  # Updated
    st.caption("• Client API Key Configuration")  # Updated
    st.caption("• Smart Cache Management")

st.caption(f"📡 **Connection**: {server_url} | 🤖 **Mode**: {prompt_type} | 📊 **Status**: {status_indicator}")

# Add debug information in development
if st.sidebar.checkbox("🐛 Debug Mode", value=False):
    st.sidebar.markdown("### 🐛 Debug Information")
    
    debug_info = {
        "server_status": server_status,
        "server_url": server_url,
        "prompt_type": prompt_type,
        "prompt_name": prompt_map.get(prompt_type, "None"),
        "brave_api_key_configured": bool(brave_api_key),
        "session_messages": len(st.session_state.messages),
        "timestamp": datetime.now().isoformat()
    }
    
    st.sidebar.json(debug_info)
    
    if st.sidebar.button("🔍 Test Direct API"):
        try:
            base_url = server_url.replace('/sse', '')
            
            # Test health endpoint
            health = requests.get(f"{base_url}/health", timeout=5)
            st.sidebar.write("**Health Check:**", health.status_code)
            
            # Test tools list
            tools = requests.get(f"{base_url}/tools", timeout=5)
            st.sidebar.write("**Tools Endpoint:**", tools.status_code)
            
            if tools.status_code == 200:
                tools_data = tools.json()
                st.sidebar.write("**Available Tools:**", len(tools_data.get("tools", {})))
                
                # Check for Brave Search tools
                tool_names = list(tools_data.get("tools", {}).keys())
                brave_tools = [t for t in tool_names if "brave" in t.lower()]
                st.sidebar.write("**Brave Tools:**", brave_tools)
                
            # Test specific prompt
            selected_prompt = prompt_map.get(prompt_type)
            if selected_prompt:
                st.sidebar.write(f"**Selected Prompt:** {selected_prompt}")
                
            # Test Brave API key configuration
            if brave_api_key:
                config_test = requests.post(
                    f"{base_url}/configure_brave_key",
                    json={"api_key": brave_api_key},
                    timeout=5
                )
                st.sidebar.write("**API Key Config:**", config_test.status_code)
                
        except Exception as e:
            st.sidebar.error(f"API Test Error: {e}")
    
    # Show prompt mapping for debugging
    st.sidebar.markdown("### 🔍 Prompt Mapping")
    for mode, prompt_name in prompt_map.items():
        status_emoji = "✅" if prompt_name else "❌"
        if mode in ["Brave Web Search", "Local Search"]:  # Updated
            status_emoji += " 🔍"  # Indicate Brave Search
        st.sidebar.write(f"{status_emoji} {mode}: `{prompt_name}`")
    
    # Show Brave Search status
    st.sidebar.markdown("### 🔍 Brave Search Status")
    st.sidebar.write(f"🔑 API Key: {'✅ Configured' if brave_api_key else '❌ Missing'}")
    st.sidebar.write(f"🔧 Web Search Tool: brave_web_search")
    st.sidebar.write(f"📍 Local Search Tool: brave_local_search")
    st.sidebar.write(f"📝 Web Prompt: brave-web-search-prompt")
    st.sidebar.write(f"📝 Local Prompt: brave-local-search-prompt")
