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
st.set_page_config(page_title="MCP DEMO - Enhanced Bing USA Web Research")
st.title("MCP DEMO - Enhanced with Bing USA Web Search & Weather")

server_url = st.sidebar.text_input("MCP Server URL", "http://localhost:8081/sse")
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
                    # Updated expected prompts to match our server
                    expected_prompts = {
                        "hedis-prompt", "caleculator-promt", "weather-prompt", 
                        "bing-search-prompt", "web-research-prompt", 
                        "test-tool-prompt", "diagnostic-prompt"
                    }
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

    # --- Tools Section (Updated for Bing USA Search) ---
    with st.sidebar.expander("🛠 Tools", expanded=False):
        tool_categories = {
            "🏥 HEDIS & Analytics": ["DFWAnalyst", "DFWSearch", "calculator"],
            "🔍 Bing USA Web Search": ["bing_search", "fetch_webpage", "web_research"],
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
                "bing-search-prompt": "🔍 Bing Search Expert",
                "web-research-prompt": "🌐 Web Research Expert",
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
    
    # Enhanced prompt type selection with Bing USA web search
    prompt_type = st.sidebar.radio(
        "Select Prompt Type", 
        ["Bing USA Web Search", "Web Research Expert", "Calculator", "HEDIS Expert", "Weather", "No Context"],
        help="Choose the type of expert assistance you need. Bing USA Web Search provides current information from USA Bing search engine."
    )
    
    prompt_map = {
        "Bing USA Web Search": "bing-search-prompt",
        "Web Research Expert": "web-research-prompt",
        "Calculator": "caleculator-promt",
        "HEDIS Expert": "hedis-prompt", 
        "Weather": "weather-prompt",
        "No Context": None
    }

    examples = {
        "Bing USA Web Search": [
            "Who is the current president of the United States in 2024-2025?",
            "What are the latest AI developments and breakthroughs from this week?", 
            "Find current information about climate change policies enacted in 2024",
            "What are the most recent developments in quantum computing technology?",
            "Research current electric vehicle sales data and market trends for 2024-2025",
            "What are today's stock market conditions and recent economic indicators?",
            "Find the latest information about space exploration missions currently active",
            "Research recent advances in medical technology and healthcare innovations",
            "What are the current developments in renewable energy technology this year?",
            "Find recent cybersecurity threats and data breaches from 2024",
            "What are the latest developments in cryptocurrency regulations in the US?", 
            "Research current smartphone technology trends and recent releases",
            "Find current information about social media platform changes and policies",
            "What are the recent developments in autonomous vehicle technology?",
            "Research current inflation rates and economic conditions in the USA"
        ],
        "Web Research Expert": [
            "Conduct comprehensive research on artificial intelligence trends in 2024",
            "Analyze multiple sources about climate change impacts this year",
            "Research and compare electric vehicle market data from various sources",
            "Find and analyze information about current space exploration missions",
            "Research renewable energy adoption rates from multiple reliable sources",
            "Analyze current cryptocurrency market conditions and regulations",
            "Research latest developments in quantum computing from academic sources",
            "Find comprehensive information about current economic indicators",
            "Research and analyze social media platform policy changes in 2024",
            "Analyze current cybersecurity threat landscape from multiple sources"
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
    
    elif prompt_type == "Bing USA Web Search":
        with st.sidebar.expander("🔍 Bing USA Search Features", expanded=False):
            st.info("""
            **🇺🇸 BING USA WEB SEARCH ENGINE:**
            
            **🔍 Search Features:**
            • Uses official Bing.com USA search
            • English-language optimized results
            • US-focused content prioritization
            • Real-time search capabilities
            
            **📄 Content Features:**
            • Full webpage content extraction
            • Smart HTML parsing and cleanup
            • Title and snippet extraction
            • Content length optimization
            
            **🎯 Search Tools:**
            • `bing_search` - Direct Bing USA search with results
            • `fetch_webpage` - Get full content from search results
            • Rate limiting to prevent blocking
            
            **⚡ Best For:** 
            Current events, news, US-specific information, recent developments, market data, technology updates.
            
            **🌐 Search Scope:**
            Prioritizes US English content and American websites for most relevant results to US users.
            """)
    
    elif prompt_type == "Web Research Expert":
        with st.sidebar.expander("🌐 Web Research Expert Features", expanded=False):
            st.info("""
            **🌐 COMPREHENSIVE WEB RESEARCH:**
            
            **🔍 Research Capabilities:**
            • Multi-source information gathering
            • Content analysis and summarization
            • Relevance scoring and ranking
            • Cross-reference validation
            
            **📊 Analysis Features:**
            • Compare information across sources
            • Identify reliable vs questionable sources
            • Extract key insights and trends
            • Generate comprehensive summaries
            
            **🛠️ Research Tools:**
            • `web_research` - Comprehensive multi-source research
            • `bing_search` - Targeted search queries
            • `fetch_webpage` - Deep content analysis
            
            **✅ Best For:** 
            Academic research, market analysis, competitive intelligence, trend analysis, fact-checking, comprehensive reports.
            
            **🎯 Research Process:**
            1. Multi-query search strategy
            2. Source credibility assessment
            3. Content analysis and extraction
            4. Synthesis and insight generation
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
                if prompt_type == "Bing USA Web Search":
                    progress_steps = [
                        "🤔 Processing your search request...",
                        "🔗 Connecting to MCP server...", 
                        "🛠️ Loading Bing USA search tools...",
                        "🧠 Creating intelligent search agent...",
                        "📝 Loading Bing search expert prompt...",
                        "🌐 Preparing Bing.com USA search...",
                        "📄 Ready to retrieve current USA information...",
                        "✍️ Generating response with Bing search data..."
                    ]
                elif prompt_type == "Web Research Expert":
                    progress_steps = [
                        "🤔 Processing your research request...",
                        "🔗 Connecting to MCP server...", 
                        "🛠️ Loading comprehensive research tools...",
                        "🧠 Creating intelligent research agent...",
                        "📝 Loading web research expert prompt...",
                        "🌐 Preparing multi-source research...",
                        "📄 Ready to analyze multiple sources...",
                        "✍️ Generating comprehensive research response..."
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
                    if prompt_type == "Bing USA Web Search":
                        bing_tools = [t for t in tools if t.name in ['bing_search', 'fetch_webpage']]
                        if bing_tools:
                            message_placeholder.text(f"🔍 Bing USA Search Tools Loaded: {', '.join([t.name for t in bing_tools])}")
                        else:
                            message_placeholder.text("❌ No Bing search tools found")
                            
                    elif prompt_type == "Web Research Expert":
                        research_tools = [t for t in tools if t.name in ['web_research', 'bing_search', 'fetch_webpage']]
                        if research_tools:
                            message_placeholder.text(f"🌐 Web Research Tools Loaded: {', '.join([t.name for t in research_tools])}")
                        else:
                            message_placeholder.text("❌ No web research tools found")
                            
                    else:
                        # Show relevant tools for other modes
                        if prompt_type == "Weather":
                            weather_tools = [t for t in tools if 'weather' in t.name.lower()]
                            if weather_tools:
                                message_placeholder.text(f"🌤️ Weather Tools: {', '.join([t.name for t in weather_tools])}")
                        elif prompt_type == "HEDIS Expert":
                            hedis_tools = [t for t in tools if 'DFW' in t.name]
                            if hedis_tools:
                                message_placeholder.text(f"🏥 HEDIS Tools: {', '.join([t.name for t in hedis_tools])}")
                        elif prompt_type == "Calculator":
                            calc_tools = [t for t in tools if 'calculator' in t.name.lower()]
                            if calc_tools:
                                message_placeholder.text(f"🧮 Calculator Tools: {', '.join([t.name for t in calc_tools])}")
                    
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
                            if hasattr(prompt_from_server[0], 'content'):
                                if "{query}" in prompt_from_server[0].content:
                                    formatted_content = prompt_from_server[0].content.format(query=query_text)
                                    prompt_from_server[0].content = formatted_content
                        else:
                            # Fallback if prompt not found
                            prompt_from_server = [{"role": "user", "content": query_text}]
                    
                    # Show specific mode activation
                    if prompt_type == "Bing USA Web Search":
                        step_index = min(step_index + 1, len(progress_steps) - 1)
                        message_placeholder.text(f"🇺🇸 Bing USA Search Mode: Activating Bing.com search API...")
                        await asyncio.sleep(1)
                    elif prompt_type == "Web Research Expert":
                        step_index = min(step_index + 1, len(progress_steps) - 1)
                        message_placeholder.text(f"🌐 Web Research Mode: Activating comprehensive research tools...")
                        await asyncio.sleep(1)
                    
                    step_index = min(step_index + 1, len(progress_steps) - 1)  
                    message_placeholder.text(progress_steps[step_index])
                    
                    # Invoke agent with retry logic
                    max_retries = 2
                    for attempt in range(max_retries):
                        try:
                            response = await agent.ainvoke({"messages": prompt_from_server})
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                message_placeholder.text(f"⚠️ Retry attempt {attempt + 1}/{max_retries}...")
                                await asyncio.sleep(2)
                                continue
                            else:
                                raise e
                    
                    # Extract result with improved handling
                    result = None
                    if isinstance(response, dict):
                        # Try different possible keys for the response
                        for key in ['messages', 'output', 'result', 'content']:
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
                        
                        if result is None:
                            # Try to find any meaningful content in the response
                            for value in response.values():
                                if isinstance(value, list) and len(value) > 0:
                                    if hasattr(value[-1], 'content'):
                                        result = value[-1].content
                                        break
                                    elif isinstance(value[-1], str):
                                        result = value[-1]
                                        break
                                elif isinstance(value, str) and len(value) > 10:
                                    result = value
                                    break
                    else:
                        result = str(response)
                    
                    # If still no result, provide fallback
                    if not result or result.strip() == "":
                        result = "⚠️ The search completed but returned no readable content. Please try rephrasing your query or check the server connection."
                    
                    # Enhanced result analysis for Bing USA Search
                    if prompt_type == "Bing USA Web Search":
                        # Look for indicators of successful Bing search
                        bing_indicators = [
                            "Bing Search Results",
                            "bing.com",
                            "Starting Bing search",
                            "Bing search completed",
                            "search results found",
                            "Result 1:", "Result 2:", "Result 3:",
                            "Link:", "Summary:",
                            "ID:"
                        ]
                        
                        failure_indicators = [
                            "Bing search error",
                            "No search results found", 
                            "search failed",
                            "Connection failed",
                            "HTTP error"
                        ]
                        
                        bing_detected = any(indicator in result for indicator in bing_indicators)
                        failure_detected = any(indicator in result for indicator in failure_indicators)
                        
                        if bing_detected and not failure_detected:
                            result = f"🇺🇸 **BING USA SEARCH: CURRENT INFORMATION RETRIEVED** 🇺🇸\n\n{result}"
                        elif failure_detected:
                            result = f"""🚨 **BING USA SEARCH FAILURE** 🚨

The Bing USA search encountered issues.

**🔧 COMMON SOLUTIONS:**
- **Network Issues**: Check internet connectivity to Bing.com
- **Rate Limiting**: Wait a moment and try again (Bing may temporarily block requests)
- **Server Configuration**: Verify MCP server tools are properly configured
- **Query Issues**: Try rephrasing your search query

**💡 TROUBLESHOOTING:**
1. Test with a simple query like "current weather"
2. Check MCP server logs for detailed errors
3. Verify server URL is accessible
4. Try "Web Research Expert" mode as alternative

**🔄 ALTERNATIVE:** Switch to "Web Research Expert" mode for comprehensive search

---

**TECHNICAL ERROR DETAILS:**

{result}"""
                        else:
                            result = f"ℹ️ **BING USA SEARCH RESPONSE**\n\n{result}\n\n💡 **Note:** Search completed successfully. Results may contain both current web data and general knowledge."
                    
                    # Enhanced analysis for Web Research Expert mode  
                    elif prompt_type == "Web Research Expert":
                        research_indicators = [
                            "Web Research Results",
                            "sources analyzed",
                            "Research Source",
                            "comprehensive research",
                            "multiple sources",
                            "analysis completed",
                            "research completed"
                        ]
                        
                        failure_indicators = [
                            "Web research error",
                            "research failed",
                            "No web research results",
                            "research tools unavailable"
                        ]
                        
                        research_detected = any(indicator in result for indicator in research_indicators)
                        failure_detected = any(indicator in result for indicator in failure_indicators)
                        
                        if research_detected and not failure_detected:
                            result = f"🌐 **WEB RESEARCH EXPERT: COMPREHENSIVE ANALYSIS COMPLETE** 🌐\n\n{result}"
                        elif failure_detected:
                            result = f"""🚨 **WEB RESEARCH FAILURE** 🚨

The comprehensive web research encountered issues.

**ERROR DETAILS:**

{result}"""
                    
                    message_placeholder.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    
                except Exception as e:
                    error_message = f"""❌ **Error**: {str(e)}

**🔧 Troubleshooting Steps:**

1. **Server Connection**: Check if MCP server is running at `{server_url}`
2. **Server URL**: Verify the URL is correct and accessible
3. **Snowflake Connection**: Ensure Snowflake connection is active
4. **Tool Configuration**: Verify Bing search tools are properly configured on server
5. **Network Issues**: Check internet connectivity for external searches

**💡 Quick Fixes:**
- Try testing connection with the "🔍 Test Connection" button
- Switch to a different prompt type to isolate the issue
- Restart MCP server if needed
- Check server logs for detailed error information"""
                    
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
                                    bing_tools = [t.name for t in tools.tools if t.name in ['bing_search', 'fetch_webpage', 'web_research']]
                                    hedis_tools = [t.name for t in tools.tools if 'DFW' in t.name]
                                    weather_tools = [t.name for t in tools.tools if 'weather' in t.name.lower()]
                                    calc_tools = [t.name for t in tools.tools if 'calculator' in t.name.lower()]
                                    system_tools = [t.name for t in tools.tools if t.name in ['test_tool', 'diagnostic']]
                                    
                                    result = "🧪 **Tool Status:**\n"
                                    if bing_tools:
                                        result += f"✅ Bing USA Search: {', '.join(bing_tools)}\n"
                                    else:
                                        result += "❌ Bing USA Search: Not available\n"
                                    
                                    if hedis_tools:
                                        result += f"✅ HEDIS Analytics: {', '.join(hedis_tools)}\n"
                                    
                                    if weather_tools:
                                        result += f"✅ Weather: {', '.join(weather_tools)}\n"
                                        
                                    if calc_tools:
                                        result += f"✅ Calculator: {', '.join(calc_tools)}\n"
                                    
                                    if system_tools:
                                        result += f"✅ System Tools: {', '.join(system_tools)}\n"
                                    
                                    # Test diagnostic tool
                                    try:
                                        diagnostic_result = await session.call_tool("diagnostic", {"test_type": "search"})
                                        if diagnostic_result and hasattr(diagnostic_result, 'content'):
                                            result += f"\n🔧 **Diagnostic Check:**\n{diagnostic_result.content[0].text}"
                                    except Exception as diag_error:
                                        result += f"\n⚠️ Diagnostic test error: {str(diag_error)}"
                                    
                                    # Test basic Bing search
                                    try:
                                        search_result = await session.call_tool("bing_search", {
                                            "query": "test search",
                                            "num_results": 1
                                        })
                                        if search_result and hasattr(search_result, 'content'):
                                            if "Bing Search Results" in search_result.content[0].text:
                                                result += f"\n✅ **Bing Search Test:** PASSED"
                                            else:
                                                result += f"\n⚠️ **Bing Search Test:** Unexpected response format"
                                        else:
                                            result += f"\n❌ **Bing Search Test:** No response"
                                    except Exception as search_error:
                                        result += f"\n❌ **Bing Search Test:** {str(search_error)}"
                                    
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
        if prompt_type == "Bing USA Web Search":
            st.success("🇺🇸 Using Bing USA Search Engine")
            st.caption("✅ Current USA information, English-optimized")
        elif prompt_type == "Web Research Expert":
            st.info("🌐 Using Comprehensive Web Research")
            st.caption("🔍 Multi-source analysis and validation")
        
        # Additional testing options
        st.markdown("---")
        st.markdown("**🔬 Advanced Testing:**")
        
        col3, col4 = st.columns(2)
        
        with col3:
            if st.button("🔍 Test Search", help="Test Bing search functionality"):
                try:
                    async def test_search():
                        async with sse_client(url=server_url) as sse_connection:
                            async with ClientSession(*sse_connection) as session:
                                await session.initialize()
                                result = await session.call_tool("bing_search", {
                                    "query": "current time",
                                    "num_results": 2
                                })
                                if result and hasattr(result, 'content'):
                                    return f"✅ Search test successful!\n\nResults preview:\n{result.content[0].text[:200]}..."
                                else:
                                    return "❌ Search test failed - no results"
                    
                    result = asyncio.run(test_search())
                    st.success(result)
                except Exception as e:
                    st.error(f"❌ Search test failed: {e}")
        
        with col4:
            if st.button("🌡️ Test Weather", help="Test weather functionality"):
                try:
                    async def test_weather():
                        async with sse_client(url=server_url) as sse_connection:
                            async with ClientSession(*sse_connection) as session:
                                await session.initialize()
                                result = await session.call_tool("get_weather", {
                                    "latitude": 40.7128,
                                    "longitude": -74.0060
                                })
                                if result and hasattr(result, 'content'):
                                    return f"✅ Weather test successful!\n\nWeather preview:\n{result.content[0].text[:200]}..."
                                else:
                                    return "❌ Weather test failed - no results"
                    
                    result = asyncio.run(test_weather())
                    st.success(result)
                except Exception as e:
                    st.error(f"❌ Weather test failed: {e}")

        # Server status indicators
        st.markdown("---")
        st.markdown("**📊 Server Status:**")
        
        # Real-time server status check
        try:
            async def check_server_status():
                try:
                    async with sse_client(url=server_url) as sse_connection:
                        async with ClientSession(*sse_connection) as session:
                            await session.initialize()
                            
                            # Quick tool count
                            tools = await session.list_tools()
                            tool_count = len(tools.tools) if hasattr(tools, 'tools') else 0
                            
                            # Quick prompt count
                            prompts = await session.list_prompts()
                            prompt_count = len(prompts.prompts) if hasattr(prompts, 'prompts') else 0
                            
                            return {
                                "status": "online",
                                "tools": tool_count,
                                "prompts": prompt_count
                            }
                except Exception:
                    return {"status": "offline", "tools": 0, "prompts": 0}
            
            status = asyncio.run(check_server_status())
            
            if status["status"] == "online":
                st.success(f"🟢 Server Online")
                st.caption(f"Tools: {status['tools']} | Prompts: {status['prompts']}")
            else:
                st.error("🔴 Server Offline")
                st.caption("Check server URL and connection")
                
        except Exception:
            st.warning("🟡 Status Unknown")
            st.caption("Unable to check server status")

        # Help and documentation
        with st.sidebar.expander("📚 Help & Documentation", expanded=False):
            st.markdown("""
            **🔧 Troubleshooting:**
            
            **Connection Issues:**
            • Verify MCP server is running
            • Check server URL format: `http://host:port/sse`
            • Test connection with the "🔍 Test Connection" button
            
            **Search Issues:**
            • Use "🔍 Test Search" to verify search functionality
            • Try different search terms if no results
            • Check internet connectivity for external searches
            
            **Tool Issues:**
            • Use "🧪 Test Tools" to check tool availability
            • Restart MCP server if tools are missing
            • Check server logs for detailed error information
            
            **Performance Tips:**
            • Use specific search terms for better results
            • Try "Web Research Expert" for comprehensive research
            • Use "Bing USA Search" for current US information
            
            **🆘 Support:**
            If issues persist, check:
            1. MCP server logs
            2. Network connectivity  
            3. Snowflake connection status
            4. Server configuration
            """)

        # Quick action shortcuts
        st.markdown("---")
        st.markdown("**⚡ Quick Actions:**")
        
        quick_actions = {
            "🌍 Current News": "What are the top news stories today in the United States?",
            "📈 Stock Market": "What are the current stock market conditions and major market movements today?",
            "🌤️ Weather NYC": "What is the current weather in New York City? (40.7128, -74.0060)",
            "🧮 Quick Math": "Calculate 25% of 847 + 123",
            "🔧 Server Test": "test_tool:connectivity test"
        }
        
        for action_name, action_query in quick_actions.items():
            if st.button(action_name, key=f"quick_{action_name}"):
                if action_query.startswith("test_tool:"):
                    # Special handling for test tool
                    try:
                        async def run_test():
                            async with sse_client(url=server_url) as sse_connection:
                                async with ClientSession(*sse_connection) as session:
                                    await session.initialize()
                                    result = await session.call_tool("test_tool", {
                                        "message": action_query.split(":", 1)[1]
                                    })
                                    return result.content[0].text if result and hasattr(result, 'content') else "Test completed"
                        
                        result = asyncio.run(run_test())
                        st.success(result)
                    except Exception as e:
                        st.error(f"Test failed: {e}")
                else:
                    # Set query for regular processing
                    st.session_state.query_input = action_query
                    st.rerun()

# Footer with version and connection info
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🚀 MCP Client v2.0")
    
with col2:
    st.caption("🔍 Bing USA Search")
    
with col3:
    st.caption("🏥 HEDIS Analytics")

# Connection status in footer
if server_url:
    try:
        # Quick async check for footer status
        async def footer_status():
            try:
                async with sse_client(url=server_url) as sse_connection:
                    async with ClientSession(*sse_connection) as session:
                        await session.initialize()
                        return True
            except:
                return False
        
        is_connected = asyncio.run(footer_status())
        status_text = "🟢 Connected" if is_connected else "🔴 Disconnected"
        st.caption(f"Status: {status_text} | Server: {server_url}")
    except:
        st.caption(f"Status: 🟡 Unknown | Server: {server_url}")
else:
    st.caption("Status: ⚪ No Server URL")
