import streamlit as st
import asyncio
import json
import yaml
import pkg_resources
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
st.set_page_config(page_title="Enhanced MCP Client with Brave Search", page_icon="🚀")
st.title("🚀 Enhanced MCP Client - DataFlyWheel Edition with Brave Search")
st.markdown("*Synced with Enhanced Server - Brave Search Integrated*")

# Updated server URL to match your configuration
server_url = st.sidebar.text_input("MCP Server URL", "http://10.126.192.183:8082/sse")

# Brave API Key Configuration
st.sidebar.markdown("### 🔍 Brave Search Configuration")
brave_api_key = st.sidebar.text_input(
    "Brave API Key", 
    value="BSAQIFoBulbULfcL6RMBxRWCtopFY0E", 
    type="password",
    help="Enter your Brave Search API key"
)

def test_server_connectivity(server_url):
    """Simple server connectivity test."""
    try:
        base_url = server_url.replace('/sse', '')
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200, {"status": "healthy" if response.status_code == 200 else "error"}
    except:
        return False, {"error": "Connection failed"}

# Configure API key in server with cleaner interface
if brave_api_key and st.sidebar.button("🔑 Configure API Key"):
    with st.spinner("Configuring Brave API key..."):
        # Test server connectivity first
        base_url = server_url.replace('/sse', '')
        try:
            health_response = requests.get(f"{base_url}/health", timeout=5)
            if health_response.status_code != 200:
                st.error("❌ Server not reachable")
                st.stop()
        except:
            st.error("❌ Cannot connect to server")
            st.stop()
        
        # Try to configure API key
        success = False
        endpoints_to_try = [
            f"{base_url}/api/v1/configure_brave_key",
            f"{base_url}/configure_brave_key"
        ]
        
        for endpoint in endpoints_to_try:
            try:
                response = requests.post(
                    endpoint,
                    json={"api_key": brave_api_key},
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    st.success("✅ Brave API key configured successfully!")
                    success = True
                    break
            except:
                continue
        
        if not success:
            st.error("❌ Failed to configure API key")
            with st.expander("Show troubleshooting info"):
                st.write("Tried endpoints:", endpoints_to_try)
                st.write("Check that your MCP server is running and accessible")

# Enhanced connection status check
@st.cache_data(ttl=15)
def check_server_connection(url):
    try:
        base_url = url.replace('/sse', '')
        
        # Try health check endpoint first
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
            # Fallback to basic connectivity test
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
st.sidebar.markdown(f"**Brave API Key:** {'✅ Configured' if brave_api_key else '❌ Not configured'}")

if server_status["connected"] and server_status.get("tools_available") != "unknown":
    st.sidebar.markdown(f"**Tools Available:** {server_status['tools_available']}")

# Test Brave Configuration Button
if st.sidebar.button("🧪 Test Brave Config", help="Test if Brave API is properly configured"):
    st.markdown("### 🧪 Testing Brave Configuration")
    
    # Test server connectivity
    server_ok, server_info = test_server_connectivity(server_url)
    
    if server_ok:
        # Test if Brave search tools are available
        try:
            base_url = server_url.replace('/sse', '')
            tools_response = requests.get(f"{base_url}/tools", timeout=10)
            
            if tools_response.status_code == 200:
                tools_data = tools_response.json()
                available_tools = tools_data.get("tools", {})
                
                brave_tools = [tool for tool in available_tools.keys() if "brave" in tool.lower()]
                
                if brave_tools:
                    st.success(f"✅ Brave tools found: {brave_tools}")
                    
                    # Test actual Brave search call
                    test_payload = {
                        "tool_name": "brave_web_search",
                        "arguments": {"query": "test search", "count": 1}
                    }
                    
                    test_response = requests.post(
                        f"{base_url}/tool_call",
                        json=test_payload,
                        timeout=15
                    )
                    
                    if test_response.status_code == 200:
                        test_result = test_response.json()
                        if test_result.get("success"):
                            st.success("✅ Brave search test successful!")
                            st.json(test_result)
                        else:
                            st.error(f"❌ Brave search test failed: {test_result.get('error')}")
                    else:
                        st.error(f"❌ Brave search test failed: HTTP {test_response.status_code}")
                else:
                    st.warning("⚠️ No Brave tools found in server tools list")
                    st.json(available_tools)
            else:
                st.error(f"❌ Could not get tools list: HTTP {tools_response.status_code}")
                
        except Exception as e:
            st.error(f"❌ Error testing Brave configuration: {e}")
    else:
        st.error("❌ Server not reachable - cannot test Brave configuration")

show_server_info = st.sidebar.checkbox("🛡 Show MCP Server Info", value=False)

# Enhanced server info display
if show_server_info:
    async def fetch_mcp_info():
        result = {"tools": [], "server_health": {}}
        
        try:
            # Get server health info
            base_url = server_url.replace('/sse', '')
            try:
                health_response = requests.get(f"{base_url}/health", timeout=5)
                if health_response.status_code == 200:
                    result["server_health"] = health_response.json()
            except:
                pass
                
            # Get MCP server info
            async with sse_client(url=server_url) as sse_connection:
                async with ClientSession(*sse_connection) as session:
                    await session.initialize()

                    # --- Tools ---
                    try:
                        tools = await session.list_tools()
                        if hasattr(tools, 'tools'):
                            for t in tools.tools:
                                tool_info = {
                                    "name": t.name,
                                    "description": getattr(t, 'description', ''),
                                }
                                result["tools"].append(tool_info)
                    except Exception as e:
                        result["tools"].append({"error": f"Failed to load tools: {e}"})

        except Exception as e:
            st.sidebar.error(f"❌ MCP Connection Error: {e}")
            
        return result

    mcp_data = asyncio.run(fetch_mcp_info())

    # Server health display
    if mcp_data.get("server_health"):
        with st.sidebar.expander("🏥 Server Health", expanded=False):
            health = mcp_data["server_health"]
            st.json(health)

    # Tools display
    with st.sidebar.expander("🛠 Available Tools", expanded=False):
        available_tools = [t for t in mcp_data["tools"] if isinstance(t, dict) and "error" not in t]
        
        for tool in available_tools:
            st.markdown(f"**{tool['name']}**")
            if tool.get('description'):
                st.caption(tool['description'])

else:
    # === MAIN APPLICATION MODE ===
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
                    mcp_server_url=server_url,
                    brave_api_key=brave_api_key
                )
            else:
                return ChatSnowflakeCortex(
                    model="claude-4-sonnet",
                    cortex_function="complete",
                    mcp_server_url=server_url,
                    brave_api_key=brave_api_key
                )
        except Exception as e:
            st.error(f"❌ Failed to initialize model: {e}")
            return None
    
    # Enhanced prompt type selection with Brave Search
    prompt_type = st.sidebar.radio(
        "🎯 Select Expert Mode", 
        ["Calculator", "HEDIS Expert", "Weather", "Brave Web Search", "Local Search", "General AI"],
        help="Choose the type of expert assistance you need"
    )
    
    # Enhanced examples with Brave Search
    examples = {
        "Calculator": [
            "Calculate the expression (4+5)/2.0", 
            "What is the square root of 144?", 
            "Calculate 3 to the power of 4",
            "What is 15% of 847?",
            "Calculate compound interest on $1000 at 5% for 3 years"
        ],
        "HEDIS Expert": [
            "What are the codes in BCS Value Set?",
            "Explain the BCS (Breast Cancer Screening) measure",
            "What is the age criteria for CBP measure?",
            "Describe the COA measure requirements",
            "What LOB is COA measure scoped under?",
            "List all value sets for diabetes measures",
            "What are the exclusions for HbA1c testing?",
            "Explain the numerator criteria for blood pressure control",
            "What is the measurement period for HEDIS measures?",
            "Define the eligible population for colorectal cancer screening"
        ],
        "Weather": [
            "What's the current weather in New York?",
            "Get weather forecast for London, UK",
            "Show me the weather for Tokyo, Japan",
            "What's the weather like in Sydney, Australia?",
            "Get current conditions for Paris, France"
        ],
        "Brave Web Search": [
            "latest AI developments 2025",
            "current renewable energy trends", 
            "recent space exploration missions",
            "today's technology news",
            "breaking news artificial intelligence",
            "newest electric vehicle models",
            "latest cryptocurrency updates",
            "recent climate change research"
        ],
        "Local Search": [
            "pizza restaurants near Central Park",
            "coffee shops in Manhattan",
            "gas stations in San Francisco",
            "Italian restaurants downtown",
            "best sushi near Times Square",
            "pharmacies open late",
            "hotels near airport",
            "auto repair shops nearby"
        ],
        "General AI": [
            "Explain quantum computing in simple terms",
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
            "What's the difference between AI and ML?"
        ]
    }

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Enhanced example queries
    with st.sidebar.expander(f"💡 Example Queries - {prompt_type}", expanded=True):
        if examples[prompt_type]:
            for i, example in enumerate(examples[prompt_type]):
                display_text = example if len(example) <= 70 else example[:67] + "..."
                if st.button(display_text, key=f"{prompt_type}_{i}_{hash(example)}", use_container_width=True):
                    st.session_state.query_input = example
                    st.session_state.selected_mode = prompt_type
        else:
            st.info("Loading examples...")

    # Show current mode
    if debug_mode:
        st.sidebar.info(f"🐛 Current Mode: {prompt_type}")
        if "selected_mode" in st.session_state:
            st.sidebar.info(f"🐛 Last Selected Mode: {st.session_state.selected_mode}")

    # Control buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Chat input handling
    if query := st.chat_input("Type your query here...") or "query_input" in st.session_state:
        if "query_input" in st.session_state:
            query = st.session_state.query_input
            del st.session_state.query_input

        with st.chat_message("user"):
            st.markdown(query)

        st.session_state.messages.append({"role": "user", "content": query})

        if debug_mode:
            st.info(f"🐛 Original query: {query}")
            st.info(f"🐛 Current mode: {prompt_type}")

        async def process_query(query_text):
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.text("Processing...")
                
                try:
                    # Initialize MCP client
                    client = MultiServerMCPClient(
                        {"DataFlyWheelServer": {"url": server_url, "transport": "sse"}}
                    )

                    model = get_model()
                    if not model:
                        raise Exception("Failed to initialize AI model")
                    
                    # Get tools and create agent
                    tools = await client.get_tools()
                    if not tools:
                        raise Exception("No tools available")
                    
                    # Create agent with tools
                    agent = create_react_agent(model=model, tools=tools)
                    
                    # Determine what type of query this is and force tool usage
                    query_lower = query_text.lower()
                    
                    # First check the selected expert mode
                    if prompt_type == "Brave Web Search":
                        message_placeholder.text("🔍 Using Brave Web Search...")
                        query_text = f"Use the brave_web_search tool to search for: {query_text}"
                    
                    elif prompt_type == "Local Search":
                        message_placeholder.text("📍 Using Brave Local Search...")
                        query_text = f"Use the brave_local_search tool to find local businesses for: {query_text}"
                    
                    elif prompt_type == "Weather":
                        message_placeholder.text("🌤️ Getting weather information...")
                        query_text = f"Use the get_weather tool to get weather information for: {query_text}"
                    
                    elif prompt_type == "Calculator":
                        message_placeholder.text("🧮 Calculating...")
                        query_text = f"Use the calculator tool to calculate: {query_text}"
                    
                    elif prompt_type == "HEDIS Expert":
                        message_placeholder.text("🏥 Searching HEDIS data...")
                        if "search" in query_lower or "find" in query_lower or "list" in query_lower:
                            query_text = f"Use the DFWSearch tool to search HEDIS documentation for: {query_text}"
                        else:
                            query_text = f"Use the DFWAnalyst tool to analyze HEDIS data for: {query_text}"
                    
                    # If General AI mode, still try to detect tool usage from keywords
                    elif prompt_type == "General AI":
                        if any(keyword in query_lower for keyword in ["search", "latest", "news", "find", "recent", "current", "what is", "tell me about"]):
                            # This should use Brave Web Search
                            if not any(local_keyword in query_lower for local_keyword in ["restaurant", "near", "gas station", "hotel", "pizza", "coffee"]):
                                message_placeholder.text("🔍 Using Brave Web Search...")
                                query_text = f"Use the brave_web_search tool to search for: {query_text}"
                        
                        elif any(keyword in query_lower for keyword in ["restaurant", "near", "gas station", "hotel", "pizza", "coffee", "local", "nearby"]):
                            # This should use Brave Local Search
                            message_placeholder.text("📍 Using Brave Local Search...")
                            query_text = f"Use the brave_local_search tool to find: {query_text}"
                        
                        elif any(keyword in query_lower for keyword in ["weather", "temperature", "forecast", "conditions"]):
                            # This should use Weather tool
                            message_placeholder.text("🌤️ Getting weather information...")
                            query_text = f"Use the get_weather tool for: {query_text}"
                        
                        elif any(keyword in query_lower for keyword in ["calculate", "compute", "math", "+", "-", "*", "/", "="]):
                            # This should use Calculator
                            message_placeholder.text("🧮 Calculating...")
                            query_text = f"Use the calculator tool to calculate: {query_text}"
                        
                        elif any(keyword in query_lower for keyword in ["hedis", "bcs", "cbp", "measure", "value set", "codes"]):
                            # This should use HEDIS tools
                            message_placeholder.text("🏥 Searching HEDIS data...")
                            if "search" in query_lower or "find" in query_lower:
                                query_text = f"Use the DFWSearch tool to search: {query_text}"
                            else:
                                query_text = f"Use the DFWAnalyst tool to analyze: {query_text}"
                    
                    # Process the modified query
                    messages = [{"role": "user", "content": query_text}]
                    
                    if debug_mode:
                        st.info(f"🐛 Debug - Modified query: {query_text}")
                        st.info(f"🐛 Debug - Selected mode: {prompt_type}")
                    
                    message_placeholder.text("🤖 Processing with AI agent...")
                    response = await asyncio.wait_for(
                        agent.ainvoke({"messages": messages}), 
                        timeout=60.0
                    )
                    
                    if debug_mode:
                        st.info(f"🐛 Debug - Raw response type: {type(response)}")
                        with st.expander("🐛 Debug - Raw Response"):
                            st.json(str(response)[:1000] + "..." if len(str(response)) > 1000 else str(response))
                    
                    # Extract result
                    result = None
                    if isinstance(response, dict):
                        if 'messages' in response:
                            messages_list = response['messages']
                            if isinstance(messages_list, list):
                                for msg in reversed(messages_list):
                                    if hasattr(msg, 'content') and hasattr(msg, 'type'):
                                        if getattr(msg, 'type', None) == 'ai':
                                            result = msg.content
                                            break
                                    elif hasattr(msg, 'content'):
                                        result = msg.content
                    
                    if not result:
                        result = str(response)
                    
                    if isinstance(result, str):
                        result = result.strip()
                        if len(result) < 10:
                            result = "Response was too short. Please try rephrasing your query."
                    
                    # Display result
                    message_placeholder.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    
                except Exception as e:
                    error_message = f"❌ Error: {str(e)}"
                    message_placeholder.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

        if query:
            asyncio.run(process_query(query))

# Footer
st.markdown("---")
st.caption(f"🚀 Enhanced MCP Client with Brave Search | 📡 {server_url} | 📊 {status_indicator}")
