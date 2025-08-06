import streamlit as st
import asyncio
import json
import yaml

from mcp.client.sse import sse_client
from mcp import ClientSession
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dependencies import SnowFlakeConnector
from llmobjectwrapper import ChatSnowflakeCortex
from snowflake.snowpark import Session

# --- Page config ---
st.set_page_config(page_title="MCP DEMO")
st.title("MCP DEMO")

# --- Sidebar: Server URL & Info ---
server_url = st.sidebar.text_input("MCP Server URL", "http://10.126.192.183:8001/sse")
show_server_info = st.sidebar.checkbox("🛡 Show MCP Server Info", value=False)

# --- Mock LLM (fallback) ---
def mock_llm_response(prompt_text: str) -> str:
    return f"🤖 Mock LLM Response to: '{prompt_text}'"

# --- Fetch and display server info ---
if show_server_info:
    async def fetch_mcp_info():
        result = {"resources": [], "tools": [], "prompts": [], "yaml": [], "search": []}
        try:
            async with sse_client(url=server_url) as sse_connection:
                async with ClientSession(*sse_connection) as session:
                    await session.initialize()
                    # Resources
                    res = await session.list_resources()
                    for r in getattr(res, 'resources', []):
                        result["resources"].append(r.name)
                    # Tools
                    tools = await session.list_tools()
                    hidden = {"add-frequent-questions", "add-prompts"}
                    for t in getattr(tools, 'tools', []):
                        if t.name not in hidden:
                            result["tools"].append(t.name)
                    # Prompts
                    pr = await session.list_prompts()
                    for p in getattr(pr, 'prompts', []):
                        args = []
                        for arg in getattr(p, 'arguments', []):
                            req = 'Required' if arg.required else 'Optional'
                            args.append(f"{arg.name} ({req}): {arg.description}")
                        result["prompts"].append({
                            "name": p.name,
                            "description": getattr(p, 'description', ''),
                            "args": args
                        })
                    # YAML models
                    try:
                        y = await session.read_resource(
                            "schematiclayer://cortex_analyst/schematic_models/hedis_stage_full/list"
                        )
                        for item in getattr(y, 'contents', []):
                            result["yaml"].append(item.text)
                    except Exception as e:
                        result["yaml"].append(f"YAML error: {e}")
                    # Search services
                    try:
                        s = await session.read_resource("search://cortex_search/search_obj/list")
                        for item in getattr(s, 'contents', []):
                            result["search"].extend(json.loads(item.text))
                    except Exception as e:
                        result["search"].append(f"Search error: {e}")
        except Exception as e:
            st.sidebar.error(f"❌ MCP Connection Error: {e}")
        return result

    mcp_data = asyncio.run(fetch_mcp_info())

    # Display
    with st.sidebar.expander("📦 Resources", expanded=False):
        for name in mcp_data["resources"]:
            st.markdown(f"**{name}**")
    with st.sidebar.expander("📜 YAML Models", expanded=False):
        for y in mcp_data["yaml"]:
            st.code(y, language="yaml")
    with st.sidebar.expander("🛠 Tools", expanded=False):
        for t in mcp_data["tools"]:
            st.markdown(f"**{t}**")
    with st.sidebar.expander("🧐 Prompts", expanded=False):
        for p in mcp_data["prompts"]:
            st.markdown(f"**{p['name']}**: {p['description']}")

else:
    # --- Cached resources ---
    @st.cache_resource
    def get_snowflake_connection():
        return SnowFlakeConnector.get_conn('aedl', '')

    @st.cache_resource
    def get_model():
        sf = get_snowflake_connection()
        sess = Session.builder.configs({"connection": sf}).getOrCreate()
        return ChatSnowflakeCortex(
            model="claude-4-sonnet",
            cortex_function="complete",
            session=sess
        )

    # --- Prompt selection ---
    prompt_type = st.sidebar.radio(
        "Select Prompt Type",
        ["Calculator", "HEDIS Expert", "Weather", "Web Search", "Wikipedia Summary", "No Context"]
    )

    prompt_map = {
        "Calculator": "calculator_prompt",
        "HEDIS Expert": "hedis_prompt",
        "Weather": "weather_prompt",
        "Web Search": "web_search_prompt",
        "Wikipedia Summary": "wikipedia_prompt",
        "No Context": None
    }

    examples = {
        "Calculator": ["(4+5)/2.0", "sqrt(16) + 7", "3**4 - 12"],
        "HEDIS Expert": [],
        "Weather": [
            "What is the present weather in Richmond?",
            "Weather forecast for Atlanta",
            "Is it raining in New York City today?"
        ],
        "Web Search": [
            "Latest AI breakthroughs",
            "Python streamlit tutorial",
            "Top 10 travel destinations 2025"
        ],
        "Wikipedia Summary": [
            "Python (programming language)",
            "Machine learning",
            "Apollo 11 mission"
        ],
        "No Context": [
            "Who won the world cup in 2022?",
            "Summarize climate change impact on oceans"
        ]
    }

    # Fetch HEDIS examples
    if prompt_type == "HEDIS Expert":
        try:
            async def load_hedis():
                async with sse_client(url=server_url) as sse_conn:
                    async with ClientSession(*sse_conn) as sess:
                        await sess.initialize()
                        res = await sess.read_resource(
                            "genaiplatform://aedl/frequent_questions/Initialization"
                        )
                        for item in getattr(res, 'contents', []):
                            examples["HEDIS Expert"].extend(json.loads(item.text))
            asyncio.run(load_hedis())
        except Exception:
            examples["HEDIS Expert"] = ["⚠️ Failed to load HEDIS examples"]

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Example buttons
    with st.sidebar.expander("Example Queries", expanded=True):
        for ex in examples[prompt_type]:
            if st.button(ex, key=ex):
                st.session_state.query_input = ex

    # Input and processing
    query = st.chat_input("Type your query here...")
    if not query and "query_input" in st.session_state:
        query = st.session_state.pop("query_input")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.text("Processing...")
            try:
                client = MultiServerMCPClient({
                    "DataFlyWheelServer": {"url": server_url, "transport": "sse"}
                })
                tools = asyncio.run(client.get_tools())
                model = get_model()
                agent = create_react_agent(model=model, tools=tools)

                prompt_name = prompt_map[prompt_type]
                if prompt_name:
                    prompt_msgs = asyncio.run(
                        client.get_prompt(
                            server_name="DataFlyWheelServer",
                            prompt_name=prompt_name,
                            arguments={"query": query}
                        )
                    )
                else:
                    prompt_msgs = [{"role": "user", "content": query}]

                response = asyncio.run(agent.ainvoke({"messages": prompt_msgs}))
                result = list(response.values())[0][1].content
                placeholder.text(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
            except Exception as e:
                err = f"Error: {e}"
                placeholder.text(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

    # Clear chat
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

