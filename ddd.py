import streamlit as st
import asyncio
import json
import requests
from datetime import datetime

from mcp.client.sse import sse_client
from mcp import ClientSession

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dependencies import SnowFlakeConnector
from llmobjectwrapper import ChatSnowflakeCortex
from snowflake.snowpark import Session

# Page config
st.set_page_config(page_title="MCP Client with Brave Search", page_icon="🚀")
st.title("🚀 MCP Client - DataFlyWheel Edition with Brave Search")
st.markdown("*Configure Brave API key and use the enhanced MCP server*")

# Server URL configuration
server_url = st.sidebar.text_input("MCP Server URL", "http://10.126.192.183:8082/sse")

# Brave API Key Configuration
st.sidebar.markdown("### 🔍 Brave Search Configuration")
brave_api_key = st.sidebar.text_input(
    "Brave API Key", 
    value="BSA-FDD7EPTjkdgDqW_znc5uhZledvE", 
    type="password",
    help="Enter your Brave Search API key"
)

# Configure API key in server
if brave_api_key and st.sidebar.button("🔑 Configure API Key"):
    try:
        # Send API key to server (you'll need to implement this endpoint)
        base_url = server_url.replace('/sse', '')
        config_response = requests.post(
            f"{base_url}/configure_brave_key",
            json={"api_key": brave_api_key},
            timeout=5
        )
        if config_response.status_code == 200:
            st.sidebar.success("✅ Brave API key configured!")
        else:
            st.sidebar.error("❌ Failed to configure API key")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")

# Enhanced connection status check
@st.cache_data(ttl=15)
def check_server_connection(url):
    try:
        base_url = url.replace('/sse', '')
        health_response = requests.get(f"{base_url}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            return {
                "connected": True,
                "status": health_data.get("status", "unknown"),
                "tools_available": len(health_data.get("tools", {})),
                "details": health_data
            }
        else:
            basic_response = requests.get(base_url, timeout=5)
            return {
                "connected": basic_response.status_code == 200,
                "status": "basic_connection",
                "tools_available": "unknown",
                "details": {}
            }
    except Exception as e:
        return {
            "connected": False,
            "status": f"error: {str(e)}",
            "tools_available": 0,
            "details": {}
        }

server_status = check_server_connection(server_url)
status_indicator = "🟢 Connected" if server_status["connected"] else "🔴 Disconnected"
st.sidebar.markdown(f"**Server Status:** {status_indicator}")

if server_status["connected"] and server_status.get("tools_available") != "unknown":
    st.sidebar.markdown(f"**Tools Available:** {server_status['tools_available']}")

# Get model functions
@st.cache_resource
def get_snowflake_connection():
    try:
        return SnowFlakeConnector.get_conn('aedl', '')
    except Exception as e:
        st.error(f"❌ Failed to connect to Snowflake: {e}")
        return None

@st.cache_resource
def get_model():
    try:
        sf_conn = get_snowflake_connection()
        if sf_conn:
            return ChatSnowflakeCortex(
                model="claude-4-sonnet", 
                cortex_function="complete",
                session=Session.builder.configs({"connection": sf_conn}).getOrCreate(),
                mcp_server_url=server_url
            )
        else:
            return ChatSnowflakeCortex(
                model="claude-4-sonnet",
                cortex_function="complete",
                mcp_server_url=server_url
            )
    except Exception as e:
        st.error(f"❌ Failed to initialize model: {e}")
        return None

# Expert mode selection
prompt_type = st.sidebar.radio(
    "🎯 Select Expert Mode", 
    ["Calculator", "HEDIS Expert", "Weather", "Web Search", "Local Search", "General AI"],
    help="Choose the type of expert assistance you need"
)

# Prompt mapping
prompt_map = {
    "Calculator": "calculator-prompt",
    "HEDIS Expert": "hedis-prompt",
    "Weather": "weather-prompt",
    "Web Search": "brave-web-search-prompt",
    "Local Search": "brave-local-search-prompt",
    "General AI": None
}

# Example queries
examples = {
    "Calculator": [
        "Calculate 25 * 4 + 10",
        "What is 15% of 847?",
        "Calculate compound interest on $1000 at 5% for 3 years"
    ],
    "HEDIS Expert": [
        "What are the codes in BCS Value Set?",
        "Explain the BCS measure",
        "What is the age criteria for CBP measure?"
    ],
    "Weather": [
        "What's the current weather in New York?",
        "Get weather forecast for London, UK",
        "Show me the weather for Tokyo, Japan"
    ],
    "Web Search": [
        "latest AI developments 2025",
        "current renewable energy trends", 
        "recent space exploration missions"
    ],
    "Local Search": [
        "pizza restaurants near Central Park",
        "coffee shops in Manhattan",
        "gas stations in San Francisco"
    ],
    "General AI": [
        "Explain quantum computing",
        "What are the benefits of renewable energy?",
        "How does machine learning work?"
    ]
}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Example queries sidebar
with st.sidebar.expander(f"💡 Example Queries - {prompt_type}", expanded=True):
    if examples[prompt_type]:
        for i, example in enumerate(examples[prompt_type]):
            if st.button(example, key=f"{prompt_type}_{i}", use_container_width=True):
                st.session_state.query_input = example

# Chat input handling
if query := st.chat_input("Type your query here...") or "query_input" in st.session_state:
    
    if "query_input" in st.session_state:
        query = st.session_state.query_input
        del st.session_state.query_input

    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append({"role": "user", "content": query})

    async def process_query(query_text):
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.text("🤔 Processing your request...")
            
            try:
                # Check server connection
                if not server_status["connected"]:
                    raise Exception("MCP server is not accessible. Please check the server URL.")
                
                message_placeholder.text("🔌 Connecting to MCP server...")
                
                client = MultiServerMCPClient(
                    {"DataFlyWheelServer": {"url": server_url, "transport": "sse"}}
                )

                model = get_model()
                if not model:
                    raise Exception("Failed to initialize the AI model.")
                
                # Get tools and create agent
                message_placeholder.text("🛠️ Loading tools from server...")
                tools = await client.get_tools()
                
                if not tools:
                    raise Exception("No tools available from the MCP server.")
                
                message_placeholder.text(f"🤖 Creating AI agent with {len(tools)} tools...")
                agent = create_react_agent(model=model, tools=tools)
                
                # Handle prompt selection
                prompt_name = prompt_map[prompt_type]
                
                if prompt_name is None:
                    # General AI mode
                    message_placeholder.text("💭 Processing in general AI mode...")
                    messages = [{"role": "user", "content": query_text}]
                else:  
                    # Get specific prompt from server
                    message_placeholder.text(f"📝 Loading {prompt_type} expert prompt...")
                    try:
                        prompt_from_server = await client.get_prompt(
                            server_name="DataFlyWheelServer",
                            prompt_name=prompt_name,
                            arguments={"query": query_text}
                        )
                        
                        if prompt_from_server and len(prompt_from_server) > 0:
                            first_prompt = prompt_from_server[0]
                            
                            if hasattr(first_prompt, 'content'):
                                content = first_prompt.content
                            elif hasattr(first_prompt, 'text'):
                                content = first_prompt.text
                            else:
                                content = str(first_prompt)
                            
                            # Handle template substitution
                            if "{query}" in content:
                                content = content.format(query=query_text)
                            
                            messages = [{"role": "user", "content": content}]
                            message_placeholder.text(f"✅ Loaded {prompt_type} expert prompt")
                        else:
                            st.warning(f"⚠️ {prompt_type} prompt not found. Using direct mode.")
                            messages = [{"role": "user", "content": query_text}]
                            
                    except Exception as prompt_error:
                        st.warning(f"⚠️ Could not load {prompt_type} prompt: {prompt_error}. Using direct mode.")
                        messages = [{"role": "user", "content": query_text}]

                message_placeholder.text("🧠 Generating response...")
                
                # Invoke agent
                try:
                    response = await asyncio.wait_for(
                        agent.ainvoke({"messages": messages}), 
                        timeout=120.0
                    )
                except asyncio.TimeoutError:
                    raise Exception("Request timed out. Please try again.")
                
                # Extract result
                result = None
                
                if isinstance(response, dict):
                    if 'messages' in response:
                        messages_list = response['messages']
                        if isinstance(messages_list, list):
                            for msg in reversed(messages_list):
                                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                                    if getattr(msg, 'type', None) == 'ai' or not result:
                                        result = msg.content
                                        break
                                elif
