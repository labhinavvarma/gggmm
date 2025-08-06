import streamlit as st
import asyncio
#import nest_asyncio
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
st.set_page_config(page_title="MCP DEMO")
st.title("MCP DEMO")
 
#nest_asyncio.apply()
 
server_url = st.sidebar.text_input("MCP Server URL", "http://10.126.192.183:8001/sse")
show_server_info = st.sidebar.checkbox("🛡 Show MCP Server Info", value=False)
 
# === MOCK LLM ===
def mock_llm_response(prompt_text: str) -> str:
    return f"🤖 Mock LLM Response to: '{prompt_text}'"
 
# === Server Info ===
# --- Show Server Information ---
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
 
   
    #--------------resource----------------------------
     # Display Resources
    with st.sidebar.expander("📦 Resources", expanded=False):
         for r in mcp_data["resources"]:
 
          # Match based on pattern inside the name
 
              if "cortex_search/search_obj/list" in r["name"]:
 
                   display_name = "Cortex Search"
 
              else:
 
                display_name = r["name"]
 
              st.markdown(f"**{display_name}**")
         # --- YAML Section ---
    with st.sidebar.expander("Schematic Layer", expanded=False):
        for y in mcp_data["yaml"]:
            st.code(y, language="yaml")
 
    # --- Tools Section (Filtered) ---
    with st.sidebar.expander("🛠 Tools", expanded=False):
        for t in mcp_data["tools"]:
            st.markdown(f"**{t['name']}**")
 
 # Display Prompts
     # Display Prompts
    with st.sidebar.expander("🧐 Prompts", expanded=False):
       for p in mcp_data["prompts"]:
        st.markdown(f"**{p['name']}**")
else:
    # Show compatibility status
    with st.sidebar.expander("🔗 Server Compatibility", expanded=True):
        if show_server_info:
            st.write("✅ **MCP Server Info Loaded**")
            # Check if all expected prompts are available
            if 'prompts' in locals() and mcp_data.get('prompts'):
                available_prompts = [p['name'] for p in mcp_data['prompts']]
                missing_prompts = [p for p in expected_prompts if p not in available_prompts]
                
                if not missing_prompts:
                    st.success("✅ All prompts compatible")
                else:
                    st.warning(f"⚠️ Missing prompts: {missing_prompts}")
                    st.error("🔄 Restart app.py to fix")
            
            # Check tools compatibility  
            if 'tools' in locals() and mcp_data.get('tools'):
                tool_count = len(mcp_data['tools'])
                st.info(f"🔧 {tool_count} tools available")
        else:
            st.write("❓ Enable 'Show MCP Server Info' to check compatibility")

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
    
    # Updated prompt types with all search options
    prompt_type = st.sidebar.radio("Select Prompt Type", [
        "Calculator", 
        "HEDIS Expert", 
        "Wikipedia Search",
        "Weather Search",
        "Web Search",
        "Test Tool", 
        "Diagnostic", 
        "No Context"
    ])
    
    # STANDARDIZED prompt mapping - exactly matching server registration
    prompt_map = {
        "Calculator": "caleculator-promt",      # Existing working prompt
        "HEDIS Expert": "hedis-prompt",         # Existing working prompt  
        "Weather Search": "weather-prompt",     # New standardized prompt
        "Web Search": "web-search-prompt",      # New standardized prompt
        "Wikipedia Search": "wikipedia-prompt", # New standardized prompt
        "Test Tool": "test-prompt",             # New standardized prompt
        "Diagnostic": "diagnostic-prompt",      # New standardized prompt
        "No Context": None
    }
    
    # Validate that we have all the prompts we need
    expected_prompts = [
        "caleculator-promt", "hedis-prompt", "weather-prompt", 
        "web-search-prompt", "wikipedia-prompt", "test-prompt", "diagnostic-prompt"
    ]
 
    # Updated examples with all search types
    examples = {
        "Calculator": ["Caleculate the expression (4+5)/2.0", "Caleculate the math function sqrt(16) + 7", "Caleculate the expression 3^4 - 12"],
        "HEDIS Expert": [],
        "Wikipedia Search": [
            "Prime Minister of Canada",
            "President of the United States", 
            "Artificial Intelligence",
            "Climate Change",
            "Python programming language",
            "Quantum Computing"
        ],
        "Weather Search": [
            "New York",
            "London, UK", 
            "Tokyo, Japan",
            "Atlanta, Georgia",
            "Paris, France",
            "Sydney, Australia"
        ],
        "Web Search": [
            "Latest news about AI technology",
            "Current stock market trends",
            "Recent developments in renewable energy",
            "Today's cryptocurrency prices",
            "Latest smartphone releases 2025"
        ],
        "Test Tool": [
            "Test connectivity",
            "Hello world test", 
            "Tool functionality check"
        ],
        "Diagnostic": [
            "Run basic diagnostic",
            "Test search capability",
            "Check current time"
        ],
        "No Context": ["Who won the world cup in 2022?", "Summarize climate change impact on oceans"]
    }
 
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
   
            #examples["HEDIS Expert"] = asyncio.run(fetch_hedis_examples())
            asyncio.run(fetch_hedis_examples())
        except Exception as e:
            examples["HEDIS Expert"] = [f"⚠️ Failed to load examples: {e}"]
 
    if "messages" not in st.session_state:
        st.session_state.messages = []
 
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
 
    with st.sidebar.expander("Example Queries", expanded=True):
        for example in examples[prompt_type]:
            if st.button(example, key=example):
                st.session_state.query_input = example
 
    if query := st.chat_input("Type your query here...") or  "query_input" in st.session_state:
 
        if "query_input" in st.session_state:
            query = st.session_state.query_input
            del st.session_state.query_input
       
        with st.chat_message("user"):
            st.markdown(query,unsafe_allow_html=True)
       
        st.session_state.messages.append({"role": "user", "content": query})
   
        async def process_query(query_text):
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.text("Processing...")
                try:
                    client = MultiServerMCPClient(
                        {"DataFlyWheelServer": {"url": server_url, "transport": "sse"}}
                    )
                    
                    # DEBUG: Check what tools are available
                    available_tools = await client.get_tools()
                    tool_names = [tool.name for tool in available_tools]
                    st.write(f"🔧 **Available tools**: {tool_names}")  # Debug output
                    
                    # DEBUG: Check if all search tools are available
                    wikipedia_search_available = 'wikipedia_search' in tool_names
                    weather_search_available = 'weather_search' in tool_names
                    web_search_available = 'web_search' in tool_names
                    
                    st.write(f"📖 **wikipedia_search available**: {wikipedia_search_available}")
                    st.write(f"🌤️ **weather_search available**: {weather_search_available}")
                    st.write(f"🌐 **web_search available**: {web_search_available}")
                    
                    # Get model and bind tools properly
                    model = get_model()
                    
                    # DEBUG: Try binding tools to model
                    if available_tools:
                        st.write("🔗 **Binding tools to ChatSnowflakeCortex model...**")
                        try:
                            model_with_tools = model.bind_tools(available_tools)
                            st.write("✅ **Tools bound successfully**")
                            # Check if tools were actually bound
                            if hasattr(model_with_tools, 'test_tools') and model_with_tools.test_tools:
                                st.write(f"📋 **Bound tools**: {list(model_with_tools.test_tools.keys())}")
                            else:
                                st.write("⚠️ **No tools found in bound model**")
                        except Exception as bind_error:
                            st.write(f"❌ **Tool binding failed**: {bind_error}")
                            model_with_tools = model
                    else:
                        st.write("⚠️ **No tools available to bind**")
                        model_with_tools = model
                       
                    agent = create_react_agent(model=model_with_tools, tools=available_tools)
                    prompt_name = prompt_map[prompt_type]
                    prompt_from_server = None
                    
                    if prompt_name == None:
                        # No Context mode - direct user query
                        prompt_from_server = [{"role": "user", "content": query_text}]
                        st.write("📝 **Using direct query (No Context mode)**")
                    else:  
                        # Try to get prompt from server with error handling
                        try:
                            st.write(f"🔍 **Requesting prompt**: {prompt_name}")
                            prompt_from_server = await client.get_prompt(
                                server_name="DataFlyWheelServer",
                                prompt_name=prompt_name,
                                arguments={"query": query_text}
                            )
                            st.write(f"✅ **Prompt '{prompt_name}' found and loaded successfully**")
                        except Exception as prompt_error:
                            # Handle unknown prompt error gracefully
                            error_msg = str(prompt_error)
                            if "unknown prompt" in error_msg.lower() or "not found" in error_msg.lower():
                                st.error(f"❌ **Prompt Error**: '{prompt_name}' not found on server")
                                st.error("💡 **Solution**: Restart your app.py server to register new prompts")
                                st.error("🔄 **Command**: Stop app.py (Ctrl+C) and run: python app.py")
                                
                                # Fallback to direct query
                                st.write("🔄 **Fallback**: Using direct query instead")
                                prompt_from_server = [{"role": "user", "content": f"Please help with: {query_text}"}]
                            else:
                                st.error(f"❌ **Prompt Error**: {error_msg}")
                                prompt_from_server = [{"role": "user", "content": query_text}]
                    
                    # Validate prompt format
                    if prompt_from_server and len(prompt_from_server) > 0:
                        if hasattr(prompt_from_server[0], 'content') and "{query}" in prompt_from_server[0].content:
                            formatted_prompt = prompt_from_server[0].content.format(query=query_text)
                        else:
                            formatted_prompt = prompt_from_server[0].content if hasattr(prompt_from_server[0], 'content') else str(prompt_from_server[0])
                    
                    # DEBUG: Show the prompt being sent for tool-related prompts
                    if prompt_type in ["Wikipedia Search", "Weather Search", "Web Search", "Test Tool", "Diagnostic"]:
                        st.write(f"🧠 **{prompt_type} Prompt being sent:**")
                        if hasattr(prompt_from_server[0], 'content'):
                            content = prompt_from_server[0].content
                            if isinstance(content, dict):
                                st.json(content)
                            else:
                                st.code(str(content)[:800] + "..." if len(str(content)) > 800 else str(content))
                        else:
                            st.write("Prompt format not recognized")
                    
                    # Enhanced agent monitoring for tool calls
                    if prompt_type in ["Wikipedia Search", "Weather Search", "Web Search", "Test Tool", "Diagnostic"]:
                        st.write("🔍 **Agent starting with tool format...**")
                        response = await agent.ainvoke({"messages": prompt_from_server})
                        
                        # Check if response contains tool output
                        if isinstance(response, dict) and 'messages' in response:
                            messages = response['messages']
                            for msg in messages:
                                if hasattr(msg, 'content') and 'tool_output' in str(msg.content).lower():
                                    st.write("✅ **Tool output detected in response!**")
                    else:
                        response = await agent.ainvoke({"messages": prompt_from_server})
                    
                    # DEBUG: Show agent response details for tool prompts
                    if prompt_type in ["Wikipedia Search", "Weather Search", "Web Search", "Test Tool", "Diagnostic"]:
                        st.write("🤖 **Agent response details:**")
                        st.write(f"Response type: {type(response)}")
                        st.write(f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
                        
                        # Check for tool execution evidence
                        result = list(response.values())[0][1].content if isinstance(response, dict) and response else "No content"
                        
                        if 'tool_output' in str(result).lower():
                            st.write("✅ **Tool execution detected in result!**")
                        elif 'success' in str(result).lower() and prompt_type == "Test Tool":
                            st.write("✅ **Test tool appears to have executed!**")
                        elif any(keyword in str(result).lower() for keyword in ['wikipedia', 'found', 'information', 'summary']):
                            st.write("✅ **Wikipedia results detected!**")
                        elif any(keyword in str(result).lower() for keyword in ['weather', 'temperature', 'humidity', 'forecast']):
                            st.write("✅ **Weather results detected!**")
                        elif any(keyword in str(result).lower() for keyword in ['web search', 'search results', 'duckduckgo', 'internet']):
                            st.write("✅ **Web search results detected!**")
                        else:
                            st.write("⚠️ **No clear tool execution evidence found**")
                            st.write("Raw result preview:")
                            st.text(str(result)[:300] + "..." if len(str(result)) > 300 else str(result))
                    
                    result = list(response.values())[0][1].content
                    message_placeholder.text(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    message_placeholder.text(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
   
        if query:
            asyncio.run(process_query(query))
   
        if st.sidebar.button("Clear Chat"):
            st.session_state.messages = []
            st.experimental_rerun()
