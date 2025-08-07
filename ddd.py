import streamlit as st
import asyncio
import json
import yaml
import pkg_resources
import asyncio
import os

from mcp.client.sse import sse_client
from mcp import ClientSession

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dependencies import SnowFlakeConnector
from llmobjectwrapper import ChatSnowflakeCortex
from snowflake.snowpark import Session
from mcp.client.sse import sse_client
from mcp import ClientSession
from langchain_mcp_adapters.client import MultiServerMCPClient

# Page config
st.set_page_config(page_title="MCP DEMO - Weather & Brave Search", page_icon="🔍")
st.title("🔍 MCP DEMO - Weather & Brave Search")
st.markdown("*Enhanced with Weather (Nominatim+NWS), Brave Search & HEDIS Analytics*")

# === BRAVE API KEY CONFIGURATION (HARDCODED) ===
# Hardcoded Brave API Key
BRAVE_API_KEY = "BSA-FDD7EPTjkdgDqW_znc5uhZledvE"
os.environ['BRAVE_API_KEY'] = BRAVE_API_KEY

with st.sidebar.expander("🔑 Brave Search API Status", expanded=False):
    st.success("✅ Brave API key is configured and ready")
    st.markdown("**Brave Search Features Enabled:**")
    st.markdown("• Web search for articles and news")
    st.markdown("• Local business and place search")
    st.markdown("• Current information retrieval")
    st.markdown("• Privacy-focused search results")

server_url = st.sidebar.text_input("MCP Server URL", "http://10.126.192.183:8001/sse")
show_server_info = st.sidebar.checkbox("🛡 Show MCP Server Info", value=False)

# Connection status check
@st.cache_data(ttl=30)
def check_server_connection(url):
    try:
        import requests
        response = requests.get(url.replace('/sse', ''), timeout=5)
        return True
    except:
        return False

server_status = check_server_connection(server_url)
status_indicator = "🟢 Connected" if server_status else "🔴 Disconnected"
st.sidebar.markdown(f"**Server Status:** {status_indicator}")

# === MOCK LLM ===
def mock_llm_response(prompt_text: str) -> str:
    return f"🤖 Mock LLM Response to: '{prompt_text}'"

# === Server Info ===
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

                    # --- YAML Resources ---
                    try:
                        yaml_content = await session.read_resource("schematiclayer://cortex_analyst/schematic_models/hedis_stage_full/list")
                        if hasattr(yaml_content, 'contents'):
                            for item in yaml_content.contents:
                                if hasattr(item, 'text'):
                                    parsed = yaml.safe_load(item.text)
                                    result["yaml"].append(yaml.dump(parsed, sort_keys=False))
                    except Exception as e:
                        result["yaml"].append(f"YAML error: {e}")

                    # --- Search Objects ---
                    try:
                        content = await session.read_resource("search://cortex_search/search_obj/list")
                        if hasattr(content, 'contents'):
                            for item in content.contents:
                                if hasattr(item, 'text'):
                                    objs = json.loads(item.text)
                                    result["search"].extend(objs)
                    except Exception as e:
                        result["search"].append(f"Search error: {e}")

        except Exception as e:
            st.sidebar.error(f"❌ MCP Connection Error: {e}")
        return result

    mcp_data = asyncio.run(fetch_mcp_info())

    # Display Resources with better organization
    with st.sidebar.expander("📦 Resources", expanded=False):
        for r in mcp_data["resources"]:
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

    # --- YAML Section ---
    with st.sidebar.expander("📋 Schematic Layer", expanded=False):
        for y in mcp_data["yaml"]:
            st.code(y, language="yaml")

    # --- Enhanced Tools Section ---
    with st.sidebar.expander("🛠 Available Tools", expanded=False):
        # Updated tool categories for new tools
        tool_categories = {
            "🏥 HEDIS & Analytics": ["DFWAnalyst", "DFWSearch", "calculator"],
            "🔍 Search & Web": ["brave_web_search", "brave_local_search"],
            "🌤️ Weather": ["get_weather"],
        }
        
        for category, expected_tools in tool_categories.items():
            st.markdown(f"**{category}:**")
            category_found = False
            for t in mcp_data["tools"]:
                if t['name'] in expected_tools:
                    st.markdown(f"  • **{t['name']}**")
                    if t.get('description'):
                        st.caption(f"    {t['description']}")
                    category_found = True
            
            if not category_found:
                st.caption("    No tools found in this category")

    # Display Prompts with enhanced formatting
    with st.sidebar.expander("🧐 Available Prompts", expanded=False):
        prompt_display_names = {
            "hedis-prompt": "🏥 HEDIS Expert",
            "caleculator-promt": "🧮 Calculator Expert",
            "weather-prompt": "🌤️ Weather Expert (Nominatim+NWS)", 
            "brave-web-search-prompt": "🔍 Brave Web Search Expert",
        }
        
        for p in mcp_data["prompts"]:
            display_name = prompt_display_names.get(p['name'], p['name'])
            st.markdown(f"**{display_name}**")
            if p.get('description'):
                st.caption(f"Description: {p['description']}")

else:
    # === MAIN APPLICATION MODE ===
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
    
    # Enhanced prompt type selection with new capabilities
    prompt_type = st.sidebar.radio(
        "🎯 Select Expert Mode", 
        ["Calculator", "HEDIS Expert", "Weather", "Web Search", "Local Search", "No Context"],
        help="Choose the type of expert assistance you need"
    )
    
    # Updated prompt mapping for new tools
    prompt_map = {
        "Calculator": "caleculator-promt",
        "HEDIS Expert": "hedis-prompt",
        "Weather": "weather-prompt",
        "Web Search": "brave-web-search-prompt",
        "Local Search": "brave-web-search-prompt",  # Uses same prompt but different tool logic
        "No Context": None
    }

    # Enhanced examples for all prompt types
    examples = {
        "Calculator": [
            "Calculate the expression (4+5)/2.0", 
            "Calculate the math function sqrt(16) + 7", 
            "Calculate the expression 3^4 - 12",
            "What is 15% of 847?",
            "Calculate compound interest on $1000 at 5% for 3 years"
        ],
        "HEDIS Expert": [],  # Will be loaded dynamically
        "Weather": [
            "What's the weather in New York?",
            "Get weather for Los Angeles, California",
            "Weather forecast for Miami, Florida",
            "Current conditions in Chicago, Illinois",
            "Show me the weather in Seattle, Washington"
        ],
        "Web Search": [
            "Latest news about artificial intelligence 2024",
            "Recent developments in renewable energy",
            "Current stock market trends",
            "New scientific discoveries in 2024",
            "Technology breakthroughs this year"
        ],
        "Local Search": [
            "Pizza restaurants near Central Park New York",
            "Coffee shops in downtown Seattle",
            "Gas stations near Los Angeles airport",
            "Hospitals in Miami, Florida",
            "Best restaurants in Chicago"
        ],
        "No Context": [
            "Who won the World Cup in 2022?", 
            "Calculate 25 * 4",
            "What's the weather in Denver?",
            "Find pizza places near me",
            "Latest AI news"
        ]
    }

    # Load HEDIS examples dynamically from MCP server
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
            examples["HEDIS Expert"] = [
                "What are the codes in BCS Value Set?",
                "Explain the BCS (Breast Cancer Screening) measure",
                "What is the age criteria for CBP measure?",
                f"⚠️ Failed to load dynamic examples: {e}"
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
                display_text = example if len(example) <= 60 else example[:57] + "..."
                if st.button(display_text, key=f"{prompt_type}_{i}_{hash(example)}", use_container_width=True):
                    st.session_state.query_input = example
        else:
            st.info("No examples available for this prompt type")

    # Add helpful tips based on selected mode
    if prompt_type == "Weather":
        with st.sidebar.expander("🌤️ Weather Tips", expanded=False):
            st.info("""
            **Weather Service (Nominatim + NWS):**
            • Works best with US locations
            • Uses National Weather Service data
            • Automatically finds coordinates for cities
            • No API key required
            
            **Examples:**
            • "New York" or "New York, NY"
            • "Los Angeles, California"  
            • "Miami, FL"
            • "Chicago, Illinois"
            """)
    
    elif prompt_type == "Web Search":
        with st.sidebar.expander("🔍 Web Search Tips", expanded=False):
            st.info("""
            **Brave Web Search:**
            • General web search for articles and news
            • Current information and recent developments
            • Privacy-focused search engine
            • Requires Brave API key (configured above)
            
            **Best for:**
            • News articles and current events
            • Research and information gathering
            • Technology and science updates
            • General web content
            """)
    
    elif prompt_type == "Local Search":
        with st.sidebar.expander("📍 Local Search Tips", expanded=False):
            st.info("""
            **Brave Local Search:**
            • Find local businesses and services
            • Restaurant, shop, and service information
            • Ratings, hours, and contact details
            • Address and location data
            
            **Examples:**
            • "Pizza near Central Park"
            • "Gas stations in Los Angeles"
            • "Coffee shops downtown Seattle"
            • "Hospitals in Miami"
            """)

    # Check if Brave search modes are selected (API key is now always available)
    if prompt_type in ["Web Search", "Local Search"]:
        st.info("🔍 Brave Search is ready! Enter your query to search the web or find local businesses.")

    # Chat input handling with enhanced processing
    if query := st.chat_input("Type your query here...") or "query_input" in st.session_state:

        if "query_input" in st.session_state:
            query = st.session_state.query_input
            del st.session_state.query_input

        with st.chat_message("user"):
            st.markdown(query, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "user", "content": query})

        async def process_query(query_text):
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.text("🤔 Processing your request...")
                
                try:
                    # Initialize MCP client
                    message_placeholder.text("🔌 Connecting to MCP server...")
                    client = MultiServerMCPClient(
                        {"DataFlyWheelServer": {"url": server_url, "transport": "sse"}}
                    )

                    model = get_model()
                    
                    # Get tools and create agent
                    message_placeholder.text("🛠️ Loading tools from server...")
                    tools = await client.get_tools()
                    message_placeholder.text("🤖 Creating AI agent...")
                    agent = create_react_agent(model=model, tools=tools)
                    
                    # Handle prompt selection with better formatting
                    prompt_name = prompt_map[prompt_type]
                    
                    if prompt_name is None:
                        # No context mode - use query directly
                        message_placeholder.text("💭 Processing without specific context...")
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
                            
                            # Handle different prompt response formats
                            if prompt_from_server and len(prompt_from_server) > 0:
                                if hasattr(prompt_from_server[0], 'content'):
                                    content = prompt_from_server[0].content
                                    # Format the prompt content if needed
                                    if "{query}" in content:
                                        content = content.format(query=query_text)
                                    messages = [{"role": "user", "content": content}]
                                else:
                                    # Handle case where prompt_from_server[0] is not a message object
                                    messages = [{"role": "user", "content": str(prompt_from_server[0])}]
                            else:
                                # Fallback if prompt not found
                                messages = [{"role": "user", "content": query_text}]
                        except Exception as prompt_error:
                            st.warning(f"⚠️ Could not load expert prompt: {prompt_error}. Using basic mode.")
                            messages = [{"role": "user", "content": query_text}]

                    message_placeholder.text("🧠 Generating response...")
                    
                    # Invoke agent with proper message format
                    response = await agent.ainvoke({"messages": messages})
                    
                    # Enhanced result extraction with better error handling
                    result = None
                    
                    if isinstance(response, dict):
                        # Try multiple strategies to extract the response
                        
                        # Strategy 1: Check for 'messages' key
                        if 'messages' in response:
                            messages_list = response['messages']
                            if isinstance(messages_list, list) and len(messages_list) > 0:
                                last_message = messages_list[-1]
                                if hasattr(last_message, 'content'):
                                    result = last_message.content
                                elif hasattr(last_message, 'text'):
                                    result = last_message.text
                                else:
                                    result = str(last_message)
                        
                        # Strategy 2: Try the original method as fallback
                        if result is None:
                            try:
                                response_values = list(response.values())
                                if response_values and len(response_values) > 0:
                                    if isinstance(response_values[0], list) and len(response_values[0]) > 1:
                                        if hasattr(response_values[0][1], 'content'):
                                            result = response_values[0][1].content
                                        else:
                                            result = str(response_values[0][1])
                                    else:
                                        result = str(response_values[0])
                            except:
                                pass
                        
                        # Strategy 3: Look for any text content in the response
                        if result is None:
                            for key, value in response.items():
                                if isinstance(value, str) and len(value) > 10:
                                    result = value
                                    break
                                elif isinstance(value, list):
                                    for item in value:
                                        if hasattr(item, 'content') and len(item.content) > 10:
                                            result = item.content
                                            break
                                        elif isinstance(item, str) and len(item) > 10:
                                            result = item
                                            break
                                    if result:
                                        break
                    
                    # If we still don't have a result, use the entire response
                    if result is None:
                        result = str(response)
                    
                    # Clean up the result if needed
                    if result is None or result.strip() == "":
                        result = "⚠️ Received empty response from the server. Please try again."
                    
                    # Format the result nicely
                    if isinstance(result, str):
                        # Clean up any formatting issues
                        result = result.strip()
                        if result.startswith('"') and result.endswith('"'):
                            result = result[1:-1]
                    
                    # Display result
                    message_placeholder.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    
                except Exception as e:
                    error_message = f"❌ **Error**: {str(e)}\n\n**Troubleshooting:**\n"
                    error_message += f"- Check if MCP server is running at: {server_url}\n"
                    error_message += f"- Verify server URL is correct\n"
                    error_message += f"- Ensure all required tools are available\n"
                    error_message += f"- Server Status: {status_indicator}"
                    
                    message_placeholder.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

        if query:
            asyncio.run(process_query(query))

    # Enhanced sidebar controls
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔧 Controls")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("🔍 Test MCP Connection", use_container_width=True):
            try:
                async def test_connection():
                    async with sse_client(url=server_url) as sse_connection:
                        async with ClientSession(*sse_connection) as session:
                            await session.initialize()
                            tools = await session.list_tools()
                            tool_count = len(tools.tools) if hasattr(tools, 'tools') else 0
                            return f"✅ Connection successful! Found {tool_count} tools available."

                result = asyncio.run(test_connection())
                st.success(result)
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")

        # Status info
        st.caption(f"🌐 **Server:** {server_url}")
        st.caption(f"🤖 **Mode:** {prompt_type}")
        st.caption(f"📊 **Status:** {status_indicator}")
        st.caption("🔑 **Brave API:** ✅ Configured")

# Footer with feature info
st.markdown("---")
st.caption("🚀 **Enhanced MCP Demo** - Weather (Nominatim+NWS) & Brave Search with HEDIS Analytics")
st.caption("📋 **Available Modes:** Calculator | HEDIS Expert | Weather | Web Search | Local Search | No Context")
