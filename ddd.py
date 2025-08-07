import streamlit as st
import asyncio
import json
import yaml
import pkg_resources
import asyncio
 
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
st.set_page_config(page_title="MCP DEMO - Enhanced Search", page_icon="🔍")
st.title("🔍 MCP DEMO - Enhanced Search & Analytics")
st.markdown("*Compatible with Wikipedia, DuckDuckGo, Weather & HEDIS Analytics*")

server_url = st.sidebar.text_input("MCP Server URL", "http://10.126.192.183:8001/sse")
show_server_info = st.sidebar.checkbox("🛡 Show MCP Server Info", value=False)

# Connection status check
@st.cache_data(ttl=30)  # Cache for 30 seconds
def check_server_connection(url):
    try:
        import requests
        # Try a simple HTTP request to check if server is accessible
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
                    # Hide internal/admin tools but show all search and functional tools
                    hidden_tools = {"add-frequent-questions", "add-prompts", "suggested_top_prompts"}
                    if hasattr(tools, 'tools'):
                        for t in tools.tools:
                            if t.name not in hidden_tools:
                                result["tools"].append({
                                    "name": t.name,
                                    "description": getattr(t, 'description', '')
                                })

                    # --- Prompts (include all available prompts) ---
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

    # Display Resources
    with st.sidebar.expander("📦 Resources", expanded=False):
        for r in mcp_data["resources"]:
            # Match based on pattern inside the name
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

    # --- Tools Section (Enhanced with categories) ---
    with st.sidebar.expander("🛠 Available Tools", expanded=False):
        # Categorize tools for better organization
        tool_categories = {
            "🏥 HEDIS & Analytics": ["DFWAnalyst", "DFWSearch", "calculator"],
            "🔍 Search & Web": ["wikipedia_search", "duckduckgo_search"], 
            "🌤️ Weather": ["get_weather"],
            "🔧 System": ["test_tool", "diagnostic"]
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
                st.caption(f"    No tools found in this category")

    # Display Prompts with better formatting
    with st.sidebar.expander("🧐 Available Prompts", expanded=False):
        prompt_display_names = {
            "hedis-prompt": "🏥 HEDIS Expert",
            "caleculator-promt": "🧮 Calculator Expert", 
            "weather-prompt": "🌤️ Weather Expert",
            "wikipedia-search-prompt": "📖 Wikipedia Search Expert",
            "duckduckgo-search-prompt": "🦆 Web Search Expert",
            "test-tool-prompt": "🔧 Test Tool",
            "diagnostic-prompt": "🔧 Diagnostic Tool"
        }
        
        for p in mcp_data["prompts"]:
            display_name = prompt_display_names.get(p['name'], p['name'])
            st.markdown(f"**{display_name}**")
            if p.get('description'):
                st.caption(f"Description: {p['description']}")
            if p.get('args'):
                st.caption(f"Arguments: {', '.join(p['args'])}")

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
    
    # Enhanced prompt type selection with new search capabilities
    prompt_type = st.sidebar.radio(
        "🎯 Select Expert Mode", 
        ["Calculator", "HEDIS Expert", "Weather", "Wikipedia Search", "Web Search", "No Context"],
        help="Choose the type of expert assistance you need"
    )
    
    # Map prompt types to server prompt names
    prompt_map = {
        "Calculator": "caleculator-promt",
        "HEDIS Expert": "hedis-prompt", 
        "Weather": "weather-prompt",
        "Wikipedia Search": "wikipedia-search-prompt",
        "Web Search": "duckduckgo-search-prompt",
        "No Context": None
    }

    # Enhanced examples for each prompt type
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
            "What is the current weather in Richmond, Virginia? (Coordinates: 37.5407, -77.4360)",
            "Get weather forecast for Atlanta, Georgia (33.7490, -84.3880)",
            "What's the weather like in New York City? (40.7128, -74.0060)",
            "Show me the weather for Denver, Colorado (39.7392, -104.9903)",
            "Current conditions in Miami, Florida (25.7617, -80.1918)"
        ],
        "Wikipedia Search": [
            "Search Wikipedia for artificial intelligence",
            "What is quantum computing according to Wikipedia?",
            "Find Wikipedia information about climate change",
            "Look up the history of the Internet on Wikipedia",
            "Search for information about machine learning"
        ],
        "Web Search": [
            "Search for latest news about AI developments", 
            "Find current information about renewable energy trends",
            "Look up recent space exploration missions",
            "Search for today's stock market news",
            "Find articles about electric vehicle adoption"
        ],
        "No Context": [
            "Who won the World Cup in 2022?", 
            "Summarize climate change impact on oceans",
            "Calculate 25 * 4",
            "What's the weather in Denver?",
            "Define machine learning"
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
                "Describe COA Measure requirements",
                f"⚠️ Failed to load dynamic examples: {e}"
            ]

    # Initialize chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Enhanced example queries with better organization
    with st.sidebar.expander(f"💡 Example Queries - {prompt_type}", expanded=True):
        if examples[prompt_type]:
            for i, example in enumerate(examples[prompt_type]):
                # Create unique keys and handle long examples
                display_text = example if len(example) <= 60 else example[:57] + "..."
                if st.button(display_text, key=f"{prompt_type}_{i}_{hash(example)}", use_container_width=True):
                    st.session_state.query_input = example
        else:
            st.info("No examples available for this prompt type")

    # Add helpful tips based on selected mode
    if prompt_type == "Weather":
        with st.sidebar.expander("🌍 Weather Tips", expanded=False):
            st.info("""
            **Weather queries require coordinates:**
            • Richmond, VA: 37.5407, -77.4360
            • Atlanta, GA: 33.7490, -84.3880  
            • New York, NY: 40.7128, -74.0060
            • Denver, CO: 39.7392, -104.9903
            • Miami, FL: 25.7617, -80.1918
            
            You can also ask the assistant to look up coordinates for other cities.
            """)
    
    elif prompt_type == "Wikipedia Search":
        with st.sidebar.expander("📖 Wikipedia Tips", expanded=False):
            st.info("""
            **Wikipedia search capabilities:**
            • Comprehensive encyclopedia articles
            • Reliable, well-sourced information
            • Historical and factual data
            • Academic and scientific topics
            
            Best for: definitions, historical facts, scientific concepts
            """)
    
    elif prompt_type == "Web Search":
        with st.sidebar.expander("🦆 Web Search Tips", expanded=False):
            st.info("""
            **Web search capabilities:**
            • Current news and developments
            • Real-time information
            • Multiple web sources
            • Privacy-focused search
            
            Best for: recent events, current trends, news updates
            """)

    elif prompt_type == "HEDIS Expert":
        with st.sidebar.expander("🏥 HEDIS Tips", expanded=False):
            st.info("""
            **HEDIS capabilities:**
            • SQL generation for value sets
            • Measure specification search
            • Code set queries
            • Analytics and reporting
            
            Try asking about specific measures like BCS, COA, CBP, or EED.
            """)

    # Chat input handling
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
                    
                    # Handle prompt selection
                    prompt_name = prompt_map[prompt_type]
                    prompt_from_server = None
                    
                    if prompt_name is None:
                        # No context mode - use query directly
                        message_placeholder.text("💭 Processing without specific context...")
                        prompt_from_server = [{"role": "user", "content": query_text}]
                    else:  
                        # Get prompt from server
                        message_placeholder.text(f"📝 Loading {prompt_type} expert prompt...")
                        try:
                            prompt_from_server = await client.get_prompt(
                                server_name="DataFlyWheelServer",
                                prompt_name=prompt_name,
                                arguments={"query": query_text}
                            )
                            
                            # Handle prompt formatting
                            if prompt_from_server and len(prompt_from_server) > 0:
                                if hasattr(prompt_from_server[0], 'content'):
                                    if "{query}" in prompt_from_server[0].content:
                                        prompt_from_server[0].content = prompt_from_server[0].content.format(query=query_text)
                                else:
                                    # Handle different prompt formats
                                    prompt_from_server = [{"role": "user", "content": str(prompt_from_server[0])}]
                            else:
                                # Fallback if prompt not found
                                prompt_from_server = [{"role": "user", "content": query_text}]
                        except Exception as prompt_error:
                            st.warning(f"⚠️ Could not load expert prompt: {prompt_error}. Using basic mode.")
                            prompt_from_server = [{"role": "user", "content": query_text}]
                    
                    message_placeholder.text("🧠 Generating response...")
                    
                    # Invoke agent
                    response = await agent.ainvoke({"messages": prompt_from_server})
                    
                    # Extract result with better error handling
                    result = None
                    if isinstance(response, dict):
                        # Try different possible keys for the response
                        for key in ['messages', 'output', 'result']:
                            if key in response:
                                if isinstance(response[key], list) and len(response[key]) > 0:
                                    last_message = response[key][-1]
                                    if hasattr(last_message, 'content'):
                                        result = last_message.content
                                    else:
                                        result = str(last_message)
                                    break
                                elif isinstance(response[key], str):
                                    result = response[key]
                                    break
                        
                        # If still no result, try to extract from response structure
                        if result is None:
                            response_values = list(response.values())
                            if response_values and len(response_values) > 0:
                                if isinstance(response_values[0], list) and len(response_values[0]) > 1:
                                    if hasattr(response_values[0][1], 'content'):
                                        result = response_values[0][1].content
                                    else:
                                        result = str(response_values[0][1])
                                else:
                                    result = str(response_values[0])
                    else:
                        result = str(response)
                    
                    if result is None:
                        result = "⚠️ Received empty response from the server."
                    
                    # Display result with formatting
                    message_placeholder.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    
                except Exception as e:
                    error_message = f"❌ **Error**: {str(e)}\n\n**Troubleshooting Steps:**\n"
                    error_message += f"- ✅ Check if MCP server is running at: {server_url}\n"
                    error_message += f"- ✅ Verify server URL is correct\n"
                    error_message += f"- ✅ Ensure Snowflake connection is active\n"
                    error_message += f"- ✅ Check server logs for more details\n\n"
                    error_message += f"**Selected Mode:** {prompt_type}\n"
                    error_message += f"**Server Status:** {status_indicator}"
                    
                    message_placeholder.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
   
        if query:
            asyncio.run(process_query(query))
   
        # Enhanced clear chat with confirmation
        if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
    # Add connection status and controls
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔧 Connection Controls")
        
        if st.button("🔍 Test MCP Connection", use_container_width=True):
            try:
                async def test_connection():
                    async with sse_client(url=server_url) as sse_connection:
                        async with ClientSession(*sse_connection) as session:
                            await session.initialize()
                            # Test listing tools to verify full functionality
                            tools = await session.list_tools()
                            tool_count = len(tools.tools) if hasattr(tools, 'tools') else 0
                            return f"✅ Connection successful! Found {tool_count} tools available."
                
                result = asyncio.run(test_connection())
                st.success(result)
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")
        
        # Server info display
        st.caption(f"🌐 **Server:** {server_url}")
        st.caption(f"🤖 **Mode:** {prompt_type}")
        st.caption(f"📊 **Status:** {status_indicator}")
        
        # Add quick server health check
        if server_status:
            st.success("🟢 Server is reachable")
        else:
            st.error("🔴 Server appears to be down")
            st.caption("Check server URL and ensure MCP server is running")

# Add footer with version info
st.markdown("---")
st.caption("🚀 **Enhanced MCP Demo** - Compatible with Wikipedia, DuckDuckGo, Weather & HEDIS Analytics")
st.caption("📋 **Available Modes:** Calculator | HEDIS Expert | Weather | Wikipedia Search | Web Search | No Context")
