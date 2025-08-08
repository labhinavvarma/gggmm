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

# Hardcoded SerpApi Key
SERPAPI_API_KEY = "28009a3e8f74ab4680e232c4ed5ae4f0e5d1bf849d052100ce3f7f74be9d4e54"
 
# Page config
st.set_page_config(page_title="MCP DEMO")
st.title("MCP DEMO")

# Display SerpApi status in sidebar
if SERPAPI_API_KEY:
    st.sidebar.success("🔍 SerpApi Loaded Successfully")
    st.sidebar.caption(f"Key: {SERPAPI_API_KEY[:8]}...{SERPAPI_API_KEY[-4:]}")
else:
    st.sidebar.error("❌ SerpApi Key Not Found")
 
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
    
    # Updated prompt types to include Web Search
    prompt_type = st.sidebar.radio("Select Prompt Type", ["Calculator", "HEDIS Expert", "Weather", "Web Search", "No Context"])
    
    # Show additional info for Web Search
    if prompt_type == "Web Search":
        st.sidebar.info("🌐 Web Search uses SerpApi for real-time results")
        st.sidebar.caption("💡 Tip: For current events, include '2025', 'current', 'latest', or 'recent' in your query for best results")
    
    # Updated prompt map to include serpapi-prompt
    prompt_map = {
        "Calculator": "caleculator-promt",
        "HEDIS Expert": "hedis-prompt",
        "Weather": "weather-prompt",
        "Web Search": "serpapi-prompt",
        "No Context": None
    }
 
    # Updated examples to include Web Search examples
    examples = {
        "Calculator": ["Caleculate the expression (4+5)/2.0", "Caleculate the math function sqrt(16) + 7", "Caleculate the expression 3^4 - 12"],
        "HEDIS Expert": [],
        "Weather": [
            "What is the present weather in Richmond?",
            "What's the weather forecast for Atlanta?",
            "Is it raining in New York City today?"
        ],
        "Web Search": [
            "current US president 2025",
            "latest AI developments 2025", 
            "recent healthcare technology trends",
            "breaking news today",
            "Donald Trump president January 2025",
            "latest HEDIS measure updates 2025"
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
                       
                    model = get_model()
                    agent = create_react_agent(model=model, tools=await client.get_tools())
                    prompt_name = prompt_map[prompt_type]
                    
                    # If using Web Search, modify the prompt to include API key instruction
                    if prompt_type == "Web Search":
                        prompt_from_server = [
                            {
                                "role": "user",
                                "content": f"""You are a web search expert. You have been provided with the `SerpApiSearch` tool to perform current internet searches. 
                                
                                When using the SerpApiSearch tool, you MUST provide these parameters:
                                - query: "{query_text}"
                                - api_key: "{SERPAPI_API_KEY}"
                                - location: (optional) for location-specific results
                                - num_results: (optional) number of results to return (default 5)
                                
                                Always use the SerpApiSearch tool to get current information. Provide comprehensive and up-to-date search results."""
                            }
                        ]
                    elif prompt_name == None:
                       prompt_from_server = [{"role": "user", "content": query_text}]
                    else:  
                        prompt_from_server = await client.get_prompt(
                            server_name="DataFlyWheelServer",
                            prompt_name=prompt_name,
                            arguments={"query": query_text}
                        )
                        
                    response = await agent.ainvoke({"messages": prompt_from_server})
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
