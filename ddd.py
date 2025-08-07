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
st.set_page_config(page_title="MCP DEMO - Enhanced Web Research")
st.title("MCP DEMO - Enhanced with Advanced Web Research & Weather")

server_url = st.sidebar.text_input("MCP Server URL", "http://10.126.192.183:8001/sse")
show_server_info = st.sidebar.checkbox("🛡 Show MCP Server Info", value=False)

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
                    # Hide internal/admin tools but show all research and weather tools
                    hidden_tools = {"add-frequent-questions", "add-prompts", "suggested_top_prompts"}
                    if hasattr(tools, 'tools'):
                        for t in tools.tools:
                            if t.name not in hidden_tools:
                                result["tools"].append({
                                    "name": t.name,
                                    "description": getattr(t, 'description', '')
                                })

                    # --- Prompts (filtered to match client expectations) ---
                    prompts = await session.list_prompts()
                    # Include new web-research-prompt
                    expected_prompts = {"hedis-prompt", "caleculator-promt", "weather-prompt", "web-research-prompt", "search-prompt", "test-tool-prompt", "diagnostic-prompt"}
                    if hasattr(prompts, 'prompts'):
                        for p in prompts.prompts:
                            if p.name in expected_prompts:
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
                display_name = "Cortex Search"
            elif "schematic_models" in r["name"]:
                display_name = "HEDIS Schematic Models"
            elif "frequent_questions" in r["name"]:
                display_name = "Frequent Questions"
            elif "prompts" in r["name"]:
                display_name = "Prompt Templates"
            else:
                display_name = r["name"]
            st.markdown(f"**{display_name}**")
    
    # --- YAML Section ---
    with st.sidebar.expander("📋 Schematic Layer", expanded=False):
        for y in mcp_data["yaml"]:
            st.code(y, language="yaml")

    # --- Tools Section (Enhanced for Web Research) ---
    with st.sidebar.expander("🛠 Tools", expanded=False):
        tool_categories = {
            "🏥 HEDIS & Analytics": ["DFWAnalyst", "DFWSearch", "calculator"],
            "🔍 Web Research (NEW)": ["web_research", "focused_web_search"],
            "🌐 Legacy Search": ["get_current_info", "debug_web_scraping", "mandatory_web_search", "verify_web_scraping", "block_old_data_responses", "force_current_search", "search_and_analyze", "duckduckgo_search", "fetch_content", "web_search", "real_search"],
            "🌤️ Weather": ["get_weather"],
            "⚙️ System & Testing": ["test_tool", "diagnostic"]
        }
        
        for category, tool_names in tool_categories.items():
            st.markdown(f"**{category}:**")
            category_tools = [t for t in mcp_data["tools"] if t['name'] in tool_names]
            if category_tools:
                for t in category_tools:
                    st.markdown(f"  • **{t['name']}**")
                    if t.get('description'):
                        st.caption(f"    {t['description'][:100]}{'...' if len(t['description']) > 100 else ''}")
            else:
                st.caption("  No tools found in this category")

    # Display Prompts
    with st.sidebar.expander("🧐 Prompts", expanded=False):
        for p in mcp_data["prompts"]:
            prompt_display_names = {
                "hedis-prompt": "🏥 HEDIS Expert",
                "caleculator-promt": "🧮 Calculator", 
                "weather-prompt": "🌤️ Weather Expert",
                "web-research-prompt": "🔍 Web Research Expert (NEW)",
                "search-prompt": "🔍 Search Expert (Legacy)",
                "test-tool-prompt": "🧪 Test Tool",
                "diagnostic-prompt": "🔧 Diagnostic"
            }
            display_name = prompt_display_names.get(p['name'], p['name'])
            st.markdown(f"**{display_name}**")
            if p.get('description'):
                st.caption(p['description'])

else:
    # Re-enable Snowflake and LLM chatbot features
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
    
    # Enhanced prompt type selection with new web research capabilities
    prompt_type = st.sidebar.radio(
        "Select Prompt Type", 
        ["Web Research Expert", "Calculator", "HEDIS Expert", "Weather", "Legacy Search", "No Context"],
        help="Choose the type of expert assistance you need. Web Research Expert uses advanced multi-engine search with content analysis."
    )
    
    prompt_map = {
        "Web Research Expert": "web-research-prompt",
        "Calculator": "caleculator-promt",
        "HEDIS Expert": "hedis-prompt", 
        "Weather": "weather-prompt",
        "Legacy Search": "search-prompt",
        "No Context": None
    }

    examples = {
        "Web Research Expert": [
            "Research the latest developments in artificial intelligence and machine learning from 2024",
            "Find current information about climate change policies and recent environmental legislation",
            "What are the most recent breakthroughs in quantum computing and their potential applications?", 
            "Research current electric vehicle market trends and adoption rates worldwide",
            "Find the latest information about renewable energy technology and recent innovations",
            "What are the current developments in space exploration and recent missions?",
            "Research recent advances in medical technology and healthcare innovations",
            "Find current information about cryptocurrency market trends and regulatory changes",
            "What are the latest developments in robotics and automation technology?",
            "Research current cybersecurity threats and recent security breaches"
        ],
        "Calculator": [
            "Calculate the expression (4+5)/2.0", 
            "Calculate the math function sqrt(16) + 7", 
            "Calculate the expression 3^4 - 12",
            "What is 15% of 847?",
            "Calculate compound interest on $1000 at 5% for 3 years"
        ],
        "HEDIS Expert": [],
        "Weather": [
            "What is the current weather in Richmond, Virginia? (Latitude: 37.5407, Longitude: -77.4360)",
            "Get weather forecast for Atlanta, Georgia (33.7490, -84.3880)",
            "What's the weather like in New York City? (40.7128, -74.0060)",
            "Show me the weather for Denver, Colorado (39.7392, -104.9903)",
            "Current conditions in Miami, Florida (25.7617, -80.1918)"
        ],
        "Legacy Search": [
            "Who is the current president of the United States?",
            "What are the latest developments in artificial intelligence this week?",
            "What's happening with climate change policies in 2024?", 
            "Find recent renewable energy technology breakthroughs and explain them",
            "What are today's stock market trends and current market conditions?",
            "🔧 DEBUG: Use debug_web_scraping to diagnose why web scraping might be failing",
            "🔧 TEST: Use get_current_info to test if current information retrieval is working"
        ],
        "No Context": [
            "Who won the world cup in 2022?", 
            "Summarize climate change impact on oceans",
            "Explain quantum computing basics",
            "What are the benefits of renewable energy?"
        ]
    }

    # Load HEDIS examples dynamically
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
                "List all HEDIS measures for 2024",
                f"⚠️ Failed to load dynamic examples: {e}"
            ]

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
                if st.button(display_text, key=f"{prompt_type}_{i}_{example[:20]}"):
                    st.session_state.query_input = example
        else:
            st.info("No examples available for this prompt type")
    
    # Add helpful tips for certain prompt types
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
    
    elif prompt_type == "Web Research Expert":
        with st.sidebar.expander("🔍 Web Research Expert Features", expanded=False):
            st.info("""
            **🚀 ADVANCED WEB RESEARCH ENGINE:**
            
            **🔍 Multi-Engine Search:**
            • Searches across Bing, Yahoo, and Searx simultaneously
            • Removes duplicates and ranks by relevance
            • Handles rate limiting automatically
            
            **📄 Content Analysis:**
            • Fetches full webpage content
            • Extracts meaningful text from HTML
            • Generates intelligent summaries
            • Calculates relevance scores
            
            **🎯 Smart Features:**
            • Relevance scoring based on query matching
            • Content enrichment with summaries
            • Timestamp tracking for freshness
            • Error handling and debugging
            
            **⚡ Tools Available:**
            • `web_research` - Comprehensive multi-engine research
            • `focused_web_search` - Quick targeted search
            
            **✅ Best For:** 
            Research projects, current events analysis, comprehensive information gathering, academic research, market analysis.
            """)
    
    elif prompt_type == "Legacy Search":
        with st.sidebar.expander("🔍 Legacy Search Tips", expanded=False):
            st.info("""
            **🌐 LEGACY SEARCH MODE:**
            Uses the original DuckDuckGo-based search tools.
            
            **Process:**
            1. 🔍 Searches DuckDuckGo for current information
            2. 📄 Attempts to scrape content from websites  
            3. 📊 Analyzes content when available
            4. 📝 Provides answers with available data
            
            **Tools Used:**
            • force_current_search (enforces live data)
            • search_and_analyze (comprehensive analysis)
            • debug_web_scraping (troubleshooting)
            
            **⚠️ Note:** 
            Legacy search may encounter more scraping failures due to bot detection. Use Web Research Expert for better reliability.
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
                
                # Show progress steps based on mode
                if prompt_type == "Web Research Expert":
                    progress_steps = [
                        "🤔 Processing your research request...",
                        "🔗 Connecting to advanced web research server...", 
                        "🛠️ Loading multi-engine web research tools...",
                        "🧠 Creating intelligent research agent...",
                        "📝 Loading web research expert prompt...",
                        "🌐 Preparing multi-engine search (Bing, Yahoo, Searx)...",
                        "📄 Ready to analyze and score content relevance...",
                        "✍️ Generating comprehensive research report..."
                    ]
                else:
                    progress_steps = [
                        "🤔 Processing your request...",
                        "🔗 Connecting to MCP server...", 
                        "🛠️ Loading tools...",
                        "🧠 Creating intelligent agent...",
                        "📝 Loading expert prompt...",
                        "🌐 Preparing to search for information...",
                        "📄 Ready to analyze content...",
                        "✍️ Generating response..."
                    ]
                
                step_index = 0
                message_placeholder.text(progress_steps[step_index])
                
                try:
                    # Initialize MCP client
                    step_index += 1
                    message_placeholder.text(progress_steps[step_index])
                    client = MultiServerMCPClient(
                        {"DataFlyWheelServer": {"url": server_url, "transport": "sse"}}
                    )
                       
                    model = get_model()
                    
                    # Get available tools from MCP server
                    step_index += 1
                    message_placeholder.text(progress_steps[step_index])
                    tools = await client.get_tools()
                    
                    # Show which tools are available based on mode
                    if prompt_type == "Web Research Expert":
                        research_tools = [t for t in tools if t.name in ['web_research', 'focused_web_search']]
                        if research_tools:
                            message_placeholder.text(f"🔍 Advanced Research Tools Loaded: {', '.join([t.name for t in research_tools])}")
                        else:
                            # Fallback to legacy search tools
                            legacy_tools = [t for t in tools if 'search' in t.name.lower() or 'scraping' in t.name.lower()]
                            if legacy_tools:
                                message_placeholder.text(f"⚠️ Using Legacy Search Tools: {', '.join([t.name for t in legacy_tools[:3]])}")
                            else:
                                message_placeholder.text("❌ No web research tools found")
                    else:
                        # Show relevant tools for other modes
                        if prompt_type == "Legacy Search":
                            search_tools = [t for t in tools if 'search' in t.name.lower() or 'scraping' in t.name.lower() or 'current' in t.name.lower()]
                            if search_tools:
                                message_placeholder.text(f"🔍 Legacy Search Tools: {', '.join([t.name for t in search_tools[:3]])}")
                        elif prompt_type == "Weather":
                            weather_tools = [t for t in tools if 'weather' in t.name.lower()]
                            if weather_tools:
                                message_placeholder.text(f"🌤️ Weather Tools: {', '.join([t.name for t in weather_tools])}")
                        elif prompt_type == "HEDIS Expert":
                            hedis_tools = [t for t in tools if 'DFW' in t.name]
                            if hedis_tools:
                                message_placeholder.text(f"🏥 HEDIS Tools: {', '.join([t.name for t in hedis_tools])}")
                    
                    await asyncio.sleep(1)  # Brief pause to show tool info
                    
                    # Create agent with tools
                    step_index += 1
                    message_placeholder.text(progress_steps[step_index])
                    agent = create_react_agent(model=model, tools=tools)
                    
                    # Handle prompt selection
                    prompt_name = prompt_map[prompt_type]
                    prompt_from_server = None
                    
                    if prompt_name is None:
                        # No context mode - use query directly
                        prompt_from_server = [{"role": "user", "content": query_text}]
                    else:  
                        # Get prompt from server
                        step_index += 1
                        message_placeholder.text(progress_steps[step_index])
                        prompt_from_server = await client.get_prompt(
                            server_name="DataFlyWheelServer",
                            prompt_name=prompt_name,
                            arguments={"query": query_text}
                        )
                        
                        # Handle prompt formatting
                        if prompt_from_server and len(prompt_from_server) > 0:
                            if "{query}" in prompt_from_server[0].content:
                                formatted_content = prompt_from_server[0].content.format(query=query_text)
                                prompt_from_server[0].content = formatted_content
                        else:
                            # Fallback if prompt not found
                            prompt_from_server = [{"role": "user", "content": query_text}]
                    
                    # Show specific mode activation
                    if prompt_type == "Web Research Expert":
                        step_index = min(step_index + 1, len(progress_steps) - 1)
                        message_placeholder.text(f"🚀 Web Research Expert Mode: Activating advanced multi-engine search with content analysis...")
                        await asyncio.sleep(1)
                    elif prompt_type == "Legacy Search":
                        step_index = min(step_index + 1, len(progress_steps) - 1)
                        message_placeholder.text(f"🔒 Legacy Search Mode: Activating DuckDuckGo-based search with data verification...")
                        await asyncio.sleep(1)
                    
                    step_index = min(step_index + 1, len(progress_steps) - 1)  
                    message_placeholder.text(progress_steps[step_index])
                    
                    # Invoke agent
                    response = await agent.ainvoke({"messages": prompt_from_server})
                    
                    # Extract result
                    if isinstance(response, dict):
                        # Try different possible keys for the response
                        result = None
                        for key in ['messages', 'output', 'result']:
                            if key in response:
                                if isinstance(response[key], list) and len(response[key]) > 0:
                                    if hasattr(response[key][-1], 'content'):
                                        result = response[key][-1].content
                                    else:
                                        result = str(response[key][-1])
                                    break
                                elif isinstance(response[key], str):
                                    result = response[key]
                                    break
                        
                        if result is None:
                            # Fallback: try to get any meaningful content
                            result = str(list(response.values())[0])
                            if isinstance(list(response.values())[0], list) and len(list(response.values())[0]) > 1:
                                result = list(response.values())[0][1].content
                    else:
                        result = str(response)
                    
                    # Enhanced result analysis for Web Research Expert
                    if prompt_type == "Web Research Expert":
                        # Look for indicators of advanced web research
                        advanced_indicators = [
                            "Web Research Results",
                            "sources analyzed",
                            "Relevance Score:",
                            "Multi-engine search", 
                            "Content Analysis:",
                            "Comprehensive research",
                            "web_research",
                            "focused_web_search",
                            "Result 1:",
                            "Result 2:",
                            "multiple sources"
                        ]
                        
                        failure_indicators = [
                            "No web research results found",
                            "Web research error", 
                            "failed to find",
                            "search engines are unavailable"
                        ]
                        
                        advanced_detected = any(indicator in result for indicator in advanced_indicators)
                        failure_detected = any(indicator in result for indicator in failure_indicators)
                        
                        if advanced_detected and not failure_detected:
                            result = f"🚀 **ADVANCED WEB RESEARCH COMPLETED** 🚀\n\n{result}"
                        elif failure_detected:
                            result = f"""🚨 **WEB RESEARCH FAILED** 🚨

The advanced web research engine encountered issues.

**🔧 POSSIBLE CAUSES:**
- Search engines may be blocking requests
- Network connectivity issues
- Rate limiting by search providers
- Server-side configuration problems

**🛠️ TROUBLESHOOTING:**
- Try with a different search query
- Switch to "Legacy Search" mode
- Check server logs for detailed error information
- Verify MCP server has internet access

---

**ERROR DETAILS:**

{result}"""
                        else:
                            result = f"ℹ️ **RESEARCH RESPONSE** (method unclear)\n\n{result}"
                    
                    # Enhanced analysis for Legacy Search mode  
                    elif prompt_type == "Legacy Search":
                        # Legacy search analysis (same as before)
                        current_info_indicators = [
                            "CURRENT INFORMATION RETRIEVED",
                            "CURRENT WEB CONTENT", 
                            "MANDATORY WEB SEARCH COMPLETED",
                            "LIVE SCRAPED CONTENT",
                            "get_current_info",
                            "mandatory_web_search",
                            "Scraped at:",
                            "characters scraped"
                        ]
                        
                        failure_indicators = [
                            "SCRAPING FAILED",
                            "Could not scrape",
                            "No results found",
                            "blocking our scraper"
                        ]
                        
                        current_info_detected = any(indicator in result for indicator in current_info_indicators)
                        failure_detected = any(indicator in result for indicator in failure_indicators)
                        
                        if current_info_detected and not failure_detected:
                            result = f"✅ **LEGACY SEARCH: CURRENT INFORMATION RETRIEVED** ✅\n\n{result}"
                        elif failure_detected:
                            result = f"""🚨 **LEGACY SEARCH: WEB SCRAPING FAILED** 🚨

The legacy DuckDuckGo-based search encountered issues.

**💡 RECOMMENDATION:** Switch to "Web Research Expert" mode for better reliability.

**ERROR DETAILS:**

{result}"""
                    
                    message_placeholder.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    
                except Exception as e:
                    error_message = f"❌ **Error**: {str(e)}\n\n**Troubleshooting:**\n- Check if MCP server is running at {server_url}\n- Verify server URL is correct\n- Ensure Snowflake connection is active\n- For Web Research Expert mode, verify web research tools are properly configured"
                    message_placeholder.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
   
        if query:
            asyncio.run(process_query(query))
   
        # Enhanced clear chat with confirmation
        if st.sidebar.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
            
    # Add connection status indicator and tool testing
    with st.sidebar:
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Test Connection"):
                try:
                    async def test_connection():
                        async with sse_client(url=server_url) as sse_connection:
                            async with ClientSession(*sse_connection) as session:
                                await session.initialize()
                                return "✅ Connection successful!"
                    
                    result = asyncio.run(test_connection())
                    st.success(result)
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")
        
        with col2:
            if st.button("🧪 Test Tools"):
                try:
                    async def test_tools():
                        async with sse_client(url=server_url) as sse_connection:
                            async with ClientSession(*sse_connection) as session:
                                await session.initialize()
                                tools = await session.list_tools()
                                if hasattr(tools, 'tools'):
                                    web_research_tools = [t.name for t in tools.tools if t.name in ['web_research', 'focused_web_search']]
                                    legacy_tools = [t.name for t in tools.tools if 'search' in t.name.lower() and t.name not in web_research_tools]
                                    
                                    result = "🧪 **Tool Status:**\n"
                                    if web_research_tools:
                                        result += f"✅ Web Research: {', '.join(web_research_tools)}\n"
                                    else:
                                        result += "❌ Web Research: Not available\n"
                                    
                                    if legacy_tools:
                                        result += f"✅ Legacy Search: {len(legacy_tools)} tools\n"
                                    else:
                                        result += "❌ Legacy Search: Not available\n"
                                    
                                    return result
                                else:
                                    return "❌ No tools found"
                    
                    result = asyncio.run(test_tools())
                    st.success(result)
                except Exception as e:
                    st.error(f"❌ Tool test failed: {e}")
        
        # Server info
        st.caption(f"🌐 Server: {server_url}")
        st.caption(f"🤖 Mode: {prompt_type}")
        
        # Add mode recommendations
        if prompt_type == "Web Research Expert":
            st.success("🚀 Using Advanced Multi-Engine Research")
        elif prompt_type == "Legacy Search":
            st.warning("⚠️ Using Legacy DuckDuckGo Search")
