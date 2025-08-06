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
    
    # Updated prompt types with Web Search added
    prompt_type = st.sidebar.radio("Select Prompt Type", ["Calculator", "HEDIS Expert", "Web Search", "Weather", "No Context"])
    
    # Updated prompt mapping with web search
    prompt_map = {
        "Calculator": "caleculator-promt",
        "HEDIS Expert": "hedis-prompt",
        "Web Search": "web-search-analysis",
        "Weather": "weather-prompt",
        "No Context": None
    }
 
    # Updated examples with web search examples
    examples = {
        "Calculator": ["Caleculate the expression (4+5)/2.0", "Caleculate the math function sqrt(16) + 7", "Caleculate the expression 3^4 - 12"],
        "HEDIS Expert": [],
        "Web Search": [
            "Search for latest AI research papers",
            "Find current weather forecast for Atlanta",
            "Look up recent news about healthcare technology",
            "Search for Python programming tutorials",
            "Find information about climate change impact on oceans"
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
                    
                    # DEBUG: Check if web_search is in tools
                    web_search_available = 'web_search' in tool_names
                    st.write(f"🔍 **web_search available**: {web_search_available}")
                       
                    model = get_model()
                    agent = create_react_agent(model=model, tools=available_tools)
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
                    
                    # DEBUG: Show the prompt being sent for Web Search
                    if prompt_type == "Web Search":
                        st.write("🧠 **Prompt being sent:**")
                        st.code(prompt_from_server[0].content[:800] + "..." if len(prompt_from_server[0].content) > 800 else prompt_from_server[0].content)
                    
                    # DEBUG: Test web_search tool directly if available
                    if prompt_type == "Web Search" and web_search_available:
                        st.write("🧪 **Testing web_search tool directly:**")
                        try:
                            # Test the tool directly
                            test_result = await client.call_tool(
                                server_name="DataFlyWheelServer",
                                tool_name="web_search",
                                arguments={"query": query_text, "limit": 3}
                            )
                            st.write("✅ **Direct tool test successful:**")
                            st.json(test_result[:500] + "..." if len(str(test_result)) > 500 else test_result)
                        except Exception as tool_error:
                            st.write(f"❌ **Direct tool test failed**: {tool_error}")
                    
                    # Run the agent
                    response = await agent.ainvoke({"messages": prompt_from_server})
                    
                    # DEBUG: Show agent response details
                    if prompt_type == "Web Search":
                        st.write("🤖 **Agent response details:**")
                        st.write(f"Response type: {type(response)}")
                        st.write(f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
                    
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
