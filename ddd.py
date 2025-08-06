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
    
    # Updated prompt types with Wikipedia Search
    prompt_type = st.sidebar.radio("Select Prompt Type", [
        "Calculator", 
        "HEDIS Expert", 
        "Wikipedia Search", 
        "Test Tool", 
        "Diagnostic", 
        "Weather", 
        "No Context"
    ])
    
    # Updated prompt mapping with Wikipedia search
    prompt_map = {
        "Calculator": "caleculator-promt",
        "HEDIS Expert": "hedis-prompt",
        "Wikipedia Search": "wikipedia-search-analysis",
        "Test Tool": "test-tool-prompt",
        "Diagnostic": "diagnostic-prompt", 
        "Weather": "weather-prompt",
        "No Context": None
    }
 
    # Updated examples with Wikipedia search examples
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
        "Weather": [
            "What is the present weather in Richmond?",
            "What's the weather forecast for Atlanta?",
            "Is it raining in New York City today?"
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
                    
                    # DEBUG: Check if wikipedia_search is in tools
                    wikipedia_search_available = 'wikipedia_search' in tool_names
                    st.write(f"📖 **wikipedia_search available**: {wikipedia_search_available}")
                    
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
                       prompt_from_server = [{"role": "user", "content": query_text}]
                    else:  
                        prompt_from_server = await client.get_prompt(
                            server_name="DataFlyWheelServer",
                            prompt_name=prompt_name,
                            arguments={"query": query_text}
                        )
                        if "{query}" in prompt_from_server[0].content:
                            formatted_prompt = prompt_from_server[0].content.format(query=query_text)
                        else:
                            formatted_prompt = prompt_from_server[0].content
                    
                    # DEBUG: Show the prompt being sent for tool-related prompts
                    if prompt_type in ["Wikipedia Search", "Test Tool", "Diagnostic"]:
                        st.write(f"🧠 **{prompt_type} Prompt being sent:**")
                        if hasattr(prompt_from_server[0], 'content'):
                            content = prompt_from_server[0].content
                            if isinstance(content, dict):
                                st.json(content)
                            else:
                                st.code(str(content)[:800] + "..." if len(str(content)) > 800 else str(content))
                        else:
                            st.write("Prompt format not recognized")
                    
                    # DEBUG: Test wikipedia_search tool directly if available
                    if prompt_type == "Wikipedia Search" and wikipedia_search_available:
                        st.write("🧪 **Testing wikipedia_search tool directly:**")
                        try:
                            # Get the tool object directly
                            wikipedia_search_tool = next((tool for tool in available_tools if tool.name == 'wikipedia_search'), None)
                            if wikipedia_search_tool:
                                st.write("✅ **Found wikipedia_search tool object**")
                                # Try to invoke it directly
                                try:
                                    # Check if tool has a direct invoke method
                                    if hasattr(wikipedia_search_tool, 'invoke'):
                                        test_result = await wikipedia_search_tool.invoke({"query": query_text, "limit": 3})
                                        st.write("✅ **Direct tool invocation successful:**")
                                        st.text(str(test_result)[:500] + "..." if len(str(test_result)) > 500 else str(test_result))
                                    else:
                                        st.write("⚠️ **Tool object doesn't have invoke method**")
                                        st.write(f"Tool methods: {[method for method in dir(wikipedia_search_tool) if not method.startswith('_')]}")
                                except Exception as direct_error:
                                    st.write(f"⚠️ **Direct invocation failed**: {direct_error}")
                            else:
                                st.write("❌ **Could not find wikipedia_search tool object**")
                        except Exception as tool_error:
                            st.write(f"❌ **Tool test setup failed**: {tool_error}")
                    
                    # Enhanced agent monitoring for tool calls
                    if prompt_type in ["Wikipedia Search", "Test Tool", "Diagnostic"]:
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
                    if prompt_type in ["Wikipedia Search", "Test Tool", "Diagnostic"]:
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
