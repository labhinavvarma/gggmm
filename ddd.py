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
st.set_page_config(page_title="MCP DEMO - Enhanced")
st.title("MCP DEMO - Enhanced with Search & Weather")
 
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
                    # Hide internal/admin tools but show new search and weather tools
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
                    # Only show prompts that the client knows how to handle
                    expected_prompts = {"hedis-prompt", "caleculator-promt", "weather-prompt", "search-prompt"}
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
 
    # --- Tools Section (Enhanced) ---
    with st.sidebar.expander("🛠 Tools", expanded=False):
        tool_categories = {
            "HEDIS & Analytics": ["DFWAnalyst", "DFWSearch", "calculator"],
            "Search & Web": ["get_current_info", "debug_web_scraping", "mandatory_web_search", "verify_web_scraping", "block_old_data_responses", "force_current_search", "search_and_analyze", "duckduckgo_search", "fetch_content", "web_search", "real_search"],
            "Weather": ["get_weather"],
            "System": ["test_tool", "diagnostic"]
        }
        
        for category, tool_names in tool_categories.items():
            st.markdown(f"**{category}:**")
            for t in mcp_data["tools"]:
                if t['name'] in tool_names:
                    st.markdown(f"  • {t['name']}")
                    if t.get('description'):
                        st.caption(f"    {t['description']}")
 
    # Display Prompts
    with st.sidebar.expander("🧐 Prompts", expanded=False):
        for p in mcp_data["prompts"]:
            prompt_display_names = {
                "hedis-prompt": "🏥 HEDIS Expert",
                "caleculator-promt": "🧮 Calculator",
                "weather-prompt": "🌤️ Weather Expert",
                "search-prompt": "🔍 Search Expert"
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
    
    # Enhanced prompt type selection with search capabilities
    prompt_type = st.sidebar.radio(
        "Select Prompt Type", 
        ["Calculator", "HEDIS Expert", "Weather", "Search Expert", "No Context"],
        help="Choose the type of expert assistance you need"
    )
    
    prompt_map = {
        "Calculator": "caleculator-promt",
        "HEDIS Expert": "hedis-prompt",
        "Weather": "weather-prompt",
        "Search Expert": "search-prompt",
        "No Context": None
    }
 
    examples = {
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
        "Search Expert": [
            "Who is the current president of the United States?",
            "What are the latest developments in artificial intelligence this week?",
            "What's happening with climate change policies in 2024?", 
            "Find recent renewable energy technology breakthroughs and explain them",
            "What are today's stock market trends and current market conditions?",
            "Find current machine learning applications in healthcare and recent breakthroughs",
            "Research the latest quantum computing developments and newest breakthroughs",
            "What are the current electric vehicle sales data and adoption rates this year?",
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
    
        elif prompt_type == "Search Expert":
        with st.sidebar.expander("🔍 Search Tips", expanded=False):
            st.info("""
            **🌐 LIVE WEB SEARCH MODE:**
            This mode FORCES current web searches and refuses to use outdated training data.
            
            **Process:**
            1. 🔍 Searches live web for TODAY'S information
            2. 📄 Fetches current content from websites  
            3. 📊 Analyzes fresh content only
            4. 📝 Provides answers with current data and citations
            
            **Tools Used:**
            • force_current_search (enforces live data)
            • search_and_analyze (comprehensive current analysis)
            • Live content fetching from websites
            • Timestamp verification for freshness
            
            **🎯 Best For:** Current events, recent news, latest developments, today's trends, current statistics.
            
            **✅ Guaranteed:** You will get TODAY'S information, not old training data!
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
                
                # Show progress steps
                progress_steps = [
                    "🤔 Processing your request...",
                    "🔗 Connecting to MCP server...", 
                    "🛠️ Loading current information retrieval tools...",
                    "🧠 Creating intelligent agent...",
                    "📝 Loading expert prompt for current information...",
                    "🌐 Preparing to request current web data...",
                    "📄 Ready to search and analyze current content...",
                    "✍️ Generating comprehensive response..."
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
                    
                    # Show which search tools are available
                    search_tools = [t for t in tools if 'search' in t.name.lower() or 'scraping' in t.name.lower() or 'mandatory' in t.name.lower() or 'block' in t.name.lower() or 'current' in t.name.lower()]
                    if search_tools:
                        current_tools = [t for t in search_tools if 'current' in t.name.lower()]
                        if current_tools:
                            message_placeholder.text(f"🌐 Found CURRENT INFO tools: {', '.join([t.name for t in current_tools])}")
                        else:
                            blocking_tools = [t for t in search_tools if 'block' in t.name.lower()]
                            mandatory_tools = [t for t in search_tools if 'mandatory' in t.name.lower()]
                            if blocking_tools:
                                message_placeholder.text(f"🔒 Found DATA BLOCKING tools: {', '.join([t.name for t in blocking_tools])}")
                            elif mandatory_tools:
                                message_placeholder.text(f"🔒 Found MANDATORY web scraping tools: {', '.join([t.name for t in mandatory_tools])}")
                            else:
                                message_placeholder.text(f"🔍 Found {len(search_tools)} search tools: {', '.join([t.name for t in search_tools])}")
                        await asyncio.sleep(1)  # Brief pause to show this info
                    
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
                    
                    # Show which expert mode is being used
                    if prompt_type == "Search Expert":
                        step_index = min(step_index + 1, len(progress_steps) - 1)
                        message_placeholder.text(f"🔒 Search Expert Mode: Activating data security protocol to block old training data...")
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
                    
                    # Check if web scraping actually happened for Search Expert mode
                    if prompt_type == "Search Expert":
                        # Look for indicators that current information was retrieved
                        current_info_indicators = [
                            "CURRENT INFORMATION RETRIEVED",
                            "CURRENT WEB CONTENT",
                            "MANDATORY WEB SEARCH COMPLETED",
                            "LIVE SCRAPED CONTENT", 
                            "get_current_info",
                            "mandatory_web_search",
                            "Scraped at:",
                            "characters scraped",
                            "Content length:",
                            "Retrieved:",
                            "Information retrieved on:"
                        ]
                        
                        # Look for failure indicators
                        failure_indicators = [
                            "SCRAPING FAILED",
                            "Could not scrape",
                            "No results found",
                            "blocking our scraper",
                            "CRITICAL FAILURE"
                        ]
                        
                        # Look for signs the LLM rejected the approach
                        rejection_indicators = [
                            "fake",
                            "don't have access to any such tool",
                            "isn't how I actually work",
                            "I can tell you directly",
                            "part of my training data"
                        ]
                        
                        current_info_detected = any(indicator in result for indicator in current_info_indicators)
                        failure_detected = any(indicator in result for indicator in failure_indicators)
                        rejection_detected = any(indicator in result for indicator in rejection_indicators)
                        
                        if rejection_detected:
                            result = f"""🧠 **LLM REJECTED TOOL USAGE** 🧠

The LLM chose not to use web search tools and answered directly from training data.

**🔍 WHAT HAPPENED:**
The AI recognized that it has relevant information in its training data and decided to answer directly rather than searching the web first.

**🎯 WHY THIS OCCURRED:**
- The LLM is designed to be helpful and saw it could answer immediately
- It didn't perceive the need for current information for this query
- The request wasn't framed as requiring absolutely current data

**🛠️ TO GET CURRENT INFORMATION:**
1. Be more specific: "Get current information about who is president right now"
2. Try: "Use get_current_info to find current information about [topic]"
3. Ask: "What's the latest information about [topic] from the web?"
4. Use diagnostic: "Use debug_web_scraping to test web search with [topic]"

**💡 NOTE:** 
For basic facts like "president of USA", the LLM may prefer to answer directly since it's confident in its knowledge. For recent events, breaking news, or rapidly changing information, it's more likely to use web search.

---

**RESPONSE FROM LLM:**

{result}

---

🔄 **Try asking for "current" or "latest" information to encourage web search usage.**"""
                        
                        elif current_info_detected and not failure_detected:
                            result = f"✅ **CURRENT INFORMATION SUCCESSFULLY RETRIEVED** ✅\n\n{result}"
                        
                        elif failure_detected:
                            result = f"""🚨 **WEB SCRAPING TECHNICAL FAILURE** 🚨

The system attempted to get current information but web scraping failed.

**🔧 TECHNICAL ISSUES DETECTED:**
- Web search may have found results but content extraction failed
- Websites might be blocking scraper requests
- Network connectivity or timeout issues
- Bot detection systems preventing access

**🛠️ TROUBLESHOOTING STEPS:**
1. Try: "Use debug_web_scraping to diagnose the exact problem"
2. Use the "🔧 Full Debug" button in the sidebar
3. Test with different topics or search terms
4. Check if the MCP server has proper internet access

---

**DETAILED ERROR INFORMATION:**

{result}

---

🔧 **Run the debug tool to identify the specific failure point.**"""
                        
                        else:
                            # No clear indicators - might be training data
                            result = f"""⚠️ **UNCLEAR: POSSIBLE TRAINING DATA USAGE** ⚠️

The response doesn't show clear evidence of current web information retrieval.

**🤔 WHAT THIS MIGHT MEAN:**
- The LLM answered from training data without using web tools
- Web search was attempted but indicators weren't captured
- The response combines training knowledge with some current data

**🔧 TO VERIFY CURRENT INFORMATION:**
- Try: "Use get_current_info to find current information about [your topic]"
- Ask: "What's the most recent information about [topic] from today's web?"
- Use diagnostic: "Use debug_web_scraping to test current information retrieval"

---

**RESPONSE (source unclear):**

{result}

---

💡 **For definitely current information, explicitly request web search in your query.**"""
                    
                    message_placeholder.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    
                except Exception as e:
                    error_message = f"❌ **Error**: {str(e)}\n\n**Troubleshooting:**\n- Check if MCP server is running at {server_url}\n- Verify server URL is correct\n- Ensure Snowflake connection is active\n- Check if search tools are properly configured"
                    message_placeholder.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
   
        if query:
            asyncio.run(process_query(query))
   
        # Enhanced clear chat with confirmation
        if st.sidebar.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
            
    # Add connection status indicator
    with st.sidebar:
        st.markdown("---")
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
        
        # Server info
        st.caption(f"🌐 Server: {server_url}")
        st.caption(f"🤖 Mode: {prompt_type}")
