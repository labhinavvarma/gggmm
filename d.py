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

# Page config
st.set_page_config(page_title="DataFlyWheel MCP Demo", page_icon="🌐")
st.title("🌐 DataFlyWheel MCP Demo")

# Sidebar configuration
server_url = st.sidebar.text_input("MCP Server URL", "http://10.126.192.183:8001/sse")
show_server_info = st.sidebar.checkbox("🛡 Show MCP Server Info", value=False)

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

    # Display server info in sidebar (same as before)
    with st.sidebar.expander("📦 Resources", expanded=False):
        for r in mcp_data["resources"]:
            if "cortex_search/search_obj/list" in r["name"]:
                display_name = "🔍 Cortex Search"
            elif "schematic_models" in r["name"]:
                display_name = "📊 Hedis Schematic"
            elif "frequent_questions" in r["name"]:
                display_name = "❓ Frequent Questions"
            elif "prompts" in r["name"]:
                display_name = "💭 Prompt Templates"
            else:
                display_name = r["name"]
            st.markdown(f"**{display_name}**")

    with st.sidebar.expander("🛠 Available Tools", expanded=False):
        for t in mcp_data["tools"]:
            st.markdown(f"**{t['name']}**")
            if t.get('description'):
                st.caption(t['description'][:100] + "..." if len(t['description']) > 100 else t['description'])

    with st.sidebar.expander("🧐 Available Prompts", expanded=False):
        for p in mcp_data["prompts"]:
            st.markdown(f"**{p['name']}**")
            if p.get('description'):
                st.caption(p['description'])

else:
    # Main application
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

    # Fixed prompt selection
    prompt_type = st.sidebar.radio("Select Prompt Type", [
        "Calculator", 
        "HEDIS Expert", 
        "Weather Expert", 
        "Web Search Expert",
        "No Context"
    ])
    
    # Fixed prompt mapping
    prompt_map = {
        "Calculator": "calculator-prompt",        # Fixed typo
        "HEDIS Expert": "hedis-prompt",
        "Weather Expert": "weather-prompt",
        "Web Search Expert": "serpapi-prompt",   # This is the key one for SerpApi
        "No Context": None
    }

    # Example queries
    examples = {
        "Calculator": [
            "Calculate the expression (4+5)/2.0", 
            "Calculate the expression 3^4 - 12",
            "What is 15% of 250?"
        ],
        "HEDIS Expert": [
            "What are the codes in BCS Value Set?",
            "What is the age criteria for BCS Measure?",
            "Generate SQL to get all diabetes measures"
        ],
        "Weather Expert": [
            "What is the present weather in Richmond?",
            "What's the weather forecast for Atlanta?",
            "Weather conditions in Boston"
        ],
        "Web Search Expert": [
            "Who is the current prime minister of India?",
            "Latest AI news today",
            "Current stock price of Apple",
            "Recent developments in healthcare",
            "Who won the latest election?"
        ],
        "No Context": [
            "Who won the world cup in 2022?", 
            "Explain quantum computing"
        ]
    }

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Example queries sidebar
    with st.sidebar.expander("💡 Example Queries", expanded=True):
        st.markdown(f"**{prompt_type} Examples:**")
        for example in examples[prompt_type]:
            if st.button(example, key=example):
                st.session_state.query_input = example

    # Chat input - FIXED WORKFLOW
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
                message_placeholder.text("🔄 Connecting to MCP server...")
                
                try:
                    # Step 1: Initialize MCP client
                    client = MultiServerMCPClient(
                        {"DataFlyWheelServer": {"url": server_url, "transport": "sse"}}
                    )
                    
                    message_placeholder.text("🧠 Loading AI model...")
                    
                    # Step 2: Get Snowflake model
                    model = get_model()
                    
                    message_placeholder.text("🛠️ Preparing tools and agent...")
                    
                    # Step 3: Create agent with MCP tools
                    tools = await client.get_tools()
                    agent = create_react_agent(model=model, tools=tools)
                    
                    # Step 4: Get prompt from server or use direct query
                    prompt_name = prompt_map[prompt_type]
                    
                    if prompt_name is None:
                        # No context - direct query
                        messages_for_agent = [{"role": "user", "content": query_text}]
                        message_placeholder.text("💭 Processing direct query...")
                    else:  
                        message_placeholder.text(f"📋 Loading {prompt_type} prompt from server...")
                        
                        # Step 5: Get prompt from MCP server
                        prompt_from_server = await client.get_prompt(
                            server_name="DataFlyWheelServer",
                            prompt_name=prompt_name,
                            arguments={"query": query_text}
                        )
                        
                        messages_for_agent = prompt_from_server
                    
                    message_placeholder.text("🤖 AI agent is thinking...")
                    
                    # Step 6: Process with agent - FIXED RESPONSE HANDLING
                    response = await agent.ainvoke({"messages": messages_for_agent})
                    
                    # Step 7: Extract result more safely
                    result = None
                    
                    # Try different ways to extract the result
                    if hasattr(response, 'messages') and response.messages:
                        # LangGraph response with messages
                        last_message = response.messages[-1]
                        if hasattr(last_message, 'content'):
                            result = last_message.content
                        else:
                            result = str(last_message)
                    elif isinstance(response, dict):
                        # Dictionary response
                        if 'messages' in response and response['messages']:
                            last_message = response['messages'][-1]
                            if isinstance(last_message, dict) and 'content' in last_message:
                                result = last_message['content']
                            else:
                                result = str(last_message)
                        else:
                            # Fallback: try to get any content
                            result = str(response)
                    else:
                        # Fallback
                        result = str(response)
                    
                    # Step 8: Display result
                    if result:
                        message_placeholder.markdown(result)
                        st.session_state.messages.append({"role": "assistant", "content": result})
                    else:
                        error_msg = "❌ Could not extract response from agent"
                        message_placeholder.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
                except Exception as e:
                    error_message = f"❌ **Error Processing Request:**\n\n```\n{str(e)}\n```"
                    message_placeholder.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    
                    # Show debug info in debug mode
                    if st.sidebar.checkbox("🐛 Show Debug Info"):
                        st.sidebar.error(f"Full error: {repr(e)}")
   
        if query:
            asyncio.run(process_query(query))

    # Controls
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()  # Fixed deprecated function
    
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()  # Fixed deprecated function

    # Server status indicator
    st.sidebar.markdown("---")
    
    async def check_server_status():
        try:
            async with sse_client(url=server_url) as sse_connection:
                async with ClientSession(*sse_connection) as session:
                    await session.initialize()
                    return True
        except:
            return False
    
    try:
        server_online = asyncio.run(check_server_status())
        status_icon = "🟢" if server_online else "🔴"
        status_text = "Online" if server_online else "Offline"
        st.sidebar.markdown(f"**Server Status:** {status_icon} {status_text}")
    except:
        st.sidebar.markdown("**Server Status:** 🔴 Offline")

    # Debug section
    with st.sidebar.expander("🐛 Debug Info", expanded=False):
        debug_mode = st.checkbox("Enable Debug Mode", value=False)
        if debug_mode:
            st.write("🔍 Debug mode enabled")
            if st.button("Test Server Connection"):
                try:
                    test_result = asyncio.run(check_server_status())
                    st.success(f"Connection test: {'✅ Success' if test_result else '❌ Failed'}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    # Information section
    with st.sidebar.expander("ℹ️ SerpApi Workflow", expanded=False):
        st.markdown("""
        **🌐 Web Search Expert Workflow:**
        
        1. **User Query** → *"Who is prime minister of India?"*
        2. **Client** → Server: *Request "serpapi-prompt"*
        3. **Server** → Client: *Returns prompt with instructions*
        4. **Client** → LLM: *Sends prompt to AI agent*
        5. **LLM** → Server: *Calls SerpApiSearch tool*
        6. **Server** → SerpApi: *HTTP request with formatted URL*
        7. **SerpApi** → Server: *Returns JSON search results*
        8. **Server** → LLM: *Raw JSON + analysis instructions*
        9. **LLM** → Client: *Final analyzed answer*
        10. **Client** → User: *Displays final answer*
        
        **🔗 URL Format:**
        ```
        https://serpapi.com/search.json?engine=google&q=who+is+prime+minister+of+india&google_domain=google.com&gl=us&hl=en&api_key=...
        ```
        """)

    # Advanced options
    with st.sidebar.expander("⚙️ Advanced Options", expanded=False):
        max_tokens = st.slider("Max Response Tokens", 100, 2000, 1000)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        
        st.markdown("**Workflow Debug:**")
        show_workflow_steps = st.checkbox("Show workflow steps", value=False)
        show_json_response = st.checkbox("Show raw JSON responses", value=False)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Powered by MCP & Snowflake Cortex*")


    # Add workflow visualization with fixed mermaid
if not show_server_info:
    with st.expander("🔄 Workflow Visualization", expanded=False):
        try:
            st.graphviz_chart("""
            digraph {
                A [label="User Query"]
                B [label="Client: Select Prompt"]
                C [label="Request serpapi-prompt"]
                D [label="Server: Return Prompt"]
                E [label="LLM: Call SerpApiSearch"]
                F [label="Server: HTTP to SerpApi"]
                G [label="SerpApi: Return JSON"]
                H [label="LLM: Generate Answer"]
                I [label="Client: Display Result"]
                
                A -> B -> C -> D -> E -> F -> G -> H -> I
            }
            """)
        except:
            st.text("Workflow diagram not available")
