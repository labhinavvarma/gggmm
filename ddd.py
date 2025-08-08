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

def configure_brave_api_key(server_url, api_key):
    """Configure Brave API key with multiple endpoint attempts and detailed error reporting."""
    base_url = server_url.replace('/sse', '')
    
    # Try multiple possible endpoints
    endpoints_to_try = [
        f"{base_url}/configure_brave_key",
        f"{base_url}/api/v1/configure_brave_key",
        f"{base_url}/api/configure_brave_key",
        f"{server_url.rstrip('/sse')}/configure_brave_key"
    ]
    
    success_info = None
    all_errors = []
    
    st.info(f"🔧 Attempting to configure Brave API key...")
    st.info(f"🌐 Base URL: {base_url}")
    st.info(f"🔑 API Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 8 else '***'}")
    
    for i, endpoint in enumerate(endpoints_to_try, 1):
        try:
            st.info(f"🔄 Attempt {i}/{len(endpoints_to_try)}: {endpoint}")
            
            response = requests.post(
                endpoint,
                json={"api_key": api_key},
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            
            st.info(f"📡 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    st.success(f"✅ SUCCESS: Brave API key configured!")
                    st.json(response_data)
                    return True, f"Configured successfully via {endpoint}"
                except json.JSONDecodeError:
                    st.success(f"✅ SUCCESS: Brave API key configured (no JSON response)")
                    return True, f"Configured successfully via {endpoint}"
            elif response.status_code == 404:
                error_msg = f"❌ Endpoint not found: {endpoint}"
                st.warning(error_msg)
                all_errors.append(error_msg)
            elif response.status_code == 500:
                error_msg = f"❌ Server error at {endpoint}: {response.text[:200]}"
                st.error(error_msg)
                all_errors.append(error_msg)
            else:
                error_msg = f"❌ HTTP {response.status_code} at {endpoint}: {response.text[:200]}"
                st.warning(error_msg)
                all_errors.append(error_msg)
                
        except requests.exceptions.ConnectionError as e:
            error_msg = f"❌ Connection failed to {endpoint}: {str(e)}"
            st.error(error_msg)
            all_errors.append(error_msg)
        except requests.exceptions.Timeout as e:
            error_msg = f"❌ Timeout for {endpoint}: {str(e)}"
            st.warning(error_msg)
            all_errors.append(error_msg)
        except Exception as e:
            error_msg = f"❌ Unexpected error for {endpoint}: {str(e)}"
            st.error(error_msg)
            all_errors.append(error_msg)
    
    return False, all_errors

def test_server_connectivity(server_url):
    """Test basic server connectivity."""
    base_url = server_url.replace('/sse', '')
    
    # Test health endpoint
    try:
        st.info(f"🏥 Testing server health: {base_url}/health")
        health_response = requests.get(f"{base_url}/health", timeout=10)
        st.success(f"✅ Health check: {health_response.status_code}")
        
        if health_response.status_code == 200:
            try:
                health_data = health_response.json()
                st.json(health_data)
                return True, health_data
            except json.JSONDecodeError:
                return True, {"status": "healthy", "details": "No JSON response"}
        return False, {"error": f"HTTP {health_response.status_code}"}
        
    except Exception as e:
        st.error(f"❌ Server connectivity failed: {e}")
        return False, {"error": str(e)}

# Configure API key in server with enhanced error handling
if brave_api_key and st.sidebar.button("🔑 Configure API Key"):
    st.markdown("### 🔧 API Key Configuration Process")
    
    # Step 1: Test server connectivity
    st.markdown("#### Step 1: Testing Server Connectivity")
    server_ok, server_info = test_server_connectivity(server_url)
    
    if server_ok:
        st.success("✅ Server is reachable")
        
        # Step 2: Configure API key
        st.markdown("#### Step 2: Configuring Brave API Key")
        config_success, config_result = configure_brave_api_key(server_url, brave_api_key)
        
        if config_success:
            st.balloons()
            st.success("🎉 Brave API key configured successfully!")
        else:
            st.error("❌ Failed to configure Brave API key")
            st.markdown("**All attempted endpoints failed:**")
            for error in config_result:
                st.text(f"• {error}")
            
            st.markdown("### 🛠️ Troubleshooting Steps:")
            st.markdown("""
            1. **Check Server URL**: Verify your MCP server URL is correct
            2. **Server Running**: Ensure your MCP server is running and accessible
            3. **Port Access**: Check if port 8082 is open and accessible
            4. **Endpoint Path**: The server might use a different API endpoint path
            5. **Server Logs**: Check your MCP server logs for error details
            """)
            
            st.markdown("### 🔍 Debug Information:")
            debug_info = {
                "server_url": server_url,
                "base_url": server_url.replace('/sse', ''),
                "api_key_length": len(brave_api_key),
                "api_key_preview": f"{brave_api_key[:8]}...{brave_api_key[-4:] if len(brave_api_key) > 8 else '***'}",
                "attempted_endpoints": [
                    f"{server_url.replace('/sse', '')}/configure_brave_key",
                    f"{server_url.replace('/sse', '')}/api/v1/configure_brave_key",
                    f"{server_url.replace('/sse', '')}/api/configure_brave_key",
                    f"{server_url.rstrip('/sse')}/configure_brave_key"
                ]
            }
            st.json(debug_info)
    else:
        st.error("❌ Cannot reach server - configuration skipped")
        st.markdown("### 🛠️ Server Connection Issues:")
        st.markdown("""
        1. **Check URL**: Verify the server URL is correct
        2. **Server Status**: Ensure your MCP server is running
        3. **Network**: Check network connectivity to the server
        4. **Firewall**: Verify no firewall is blocking the connection
        """)

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
    async def fetch_enhanced_mcp_info():
        result = {
            "resources": [], 
            "tools": [], 
            "prompts": [], 
            "yaml": [], 
            "search": [],
            "server_health": {},
            "weather_cache": {},
            "brave_cache": {}
        }
        
        try:
            # Get server health info
            base_url = server_url.replace('/sse', '')
            try:
                health_response = requests.get(f"{base_url}/health", timeout=5)
                if health_response.status_code == 200:
                    result["server_health"] = health_response.json()
            except:
                pass
            
            # Get weather cache status
            try:
                cache_response = requests.get(f"{base_url}/weather_cache", timeout=5)
                if cache_response.status_code == 200:
                    result["weather_cache"] = cache_response.json()
            except:
                pass
            
            # Get Brave cache status
            try:
                brave_cache_response = requests.get(f"{base_url}/brave_cache", timeout=5)
                if brave_cache_response.status_code == 200:
                    result["brave_cache"] = brave_cache_response.json()
            except:
                pass
                
            # Get MCP server info
            async with sse_client(url=server_url) as sse_connection:
                async with ClientSession(*sse_connection) as session:
                    await session.initialize()

                    # --- Resources ---
                    try:
                        resources = await session.list_resources()
                        if hasattr(resources, 'resources'):
                            for r in resources.resources:
                                result["resources"].append({
                                    "name": r.name,
                                    "uri": getattr(r, 'uri', 'N/A'),
                                    "description": getattr(r, 'description', 'N/A')
                                })
                    except Exception as e:
                        result["resources"].append({"error": f"Failed to load resources: {e}"})
                   
                    # --- Enhanced Tools (with Brave Search) ---
                    try:
                        tools = await session.list_tools()
                        hidden_tools = {"add-frequent-questions", "add-prompts", "suggested_top_prompts"}
                        if hasattr(tools, 'tools'):
                            for t in tools.tools:
                                if t.name not in hidden_tools:
                                    tool_info = {
                                        "name": t.name,
                                        "description": getattr(t, 'description', ''),
                                    }
                                    
                                    if hasattr(t, 'inputSchema'):
                                        schema = t.inputSchema
                                        if isinstance(schema, dict) and 'properties' in schema:
                                            tool_info["parameters"] = list(schema['properties'].keys())
                                    
                                    result["tools"].append(tool_info)
                    except Exception as e:
                        result["tools"].append({"error": f"Failed to load tools: {e}"})

                    # --- Enhanced Prompts ---
                    try:
                        prompts = await session.list_prompts()
                        if hasattr(prompts, 'prompts'):
                            for p in prompts.prompts:
                                args = []
                                if hasattr(p, 'arguments'):
                                    for arg in p.arguments:
                                        args.append({
                                            "name": arg.name,
                                            "required": getattr(arg, 'required', False),
                                            "description": getattr(arg, 'description', '')
                                        })
                                result["prompts"].append({
                                    "name": p.name,
                                    "description": getattr(p, 'description', ''),
                                    "args": args
                                })
                    except Exception as e:
                        result["prompts"].append({"error": f"Failed to load prompts: {e}"})

        except Exception as e:
            st.sidebar.error(f"❌ MCP Connection Error: {e}")
            
        return result

    mcp_data = asyncio.run(fetch_enhanced_mcp_info())

    # Enhanced server health display
    if mcp_data.get("server_health"):
        with st.sidebar.expander("🏥 Server Health", expanded=True):
            health = mcp_data["server_health"]
            st.json(health)
    
    # Weather cache status
    if mcp_data.get("weather_cache") and mcp_data["weather_cache"].get("cache_entries", 0) > 0:
        with st.sidebar.expander("🌤️ Weather Cache Status", expanded=False):
            cache_info = mcp_data["weather_cache"]
            st.write(f"**Cached Locations:** {cache_info.get('cache_entries', 0)}")
            
            for location, status in cache_info.get("cache_status", {}).items():
                valid_indicator = "✅" if status.get("is_valid") else "❌"
                st.write(f"{valid_indicator} **{location}**: {status.get('age_seconds', 0):.0f}s old")
    
    # Brave cache status
    if mcp_data.get("brave_cache") and mcp_data["brave_cache"].get("cache_entries", 0) > 0:
        with st.sidebar.expander("🔍 Brave Search Cache", expanded=False):
            cache_info = mcp_data["brave_cache"]
            st.write(f"**Cached Searches:** {cache_info.get('cache_entries', 0)}")
            
            for search_key, status in cache_info.get("cache_status", {}).items():
                valid_indicator = "✅" if status.get("is_valid") else "❌"
                display_key = search_key[:30] + "..." if len(search_key) > 30 else search_key
                st.write(f"{valid_indicator} **{display_key}**: {status.get('age_seconds', 0):.0f}s old")

    # Enhanced Tools Section - Updated with Brave Search
    with st.sidebar.expander("🛠 Available Tools", expanded=False):
        tool_categories = {
            "🏥 HEDIS & Analytics": ["DFWAnalyst", "DFWSearch", "calculator"],
            "🔍 Search & Information": ["brave_web_search", "brave_local_search"],
            "🌤️ Weather & Location": ["get_weather"],
            "🔧 System & Testing": ["test_tool", "diagnostic"]
        }
        
        available_tools = {t["name"]: t for t in mcp_data["tools"] if isinstance(t, dict) and "error" not in t}
        
        for category, expected_tools in tool_categories.items():
            st.markdown(f"**{category}:**")
            category_found = False
            for tool_name in expected_tools:
                if tool_name in available_tools:
                    tool_info = available_tools[tool_name]
                    st.markdown(f"  • **{tool_name}**")
                    if tool_info.get('description'):
                        st.caption(f"    {tool_info['description']}")
                    if tool_info.get('parameters'):
                        st.caption(f"    Parameters: {', '.join(tool_info['parameters'])}")
                    category_found = True
            
            if not category_found:
                st.caption("    No tools found in this category")

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
        "HEDIS Expert": [],  # Will be loaded dynamically
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
        else:
            st.info("Loading examples...")

    # Chat input handling
    if query := st.chat_input("Type your query here...") or "query_input" in st.session_state:
        if "query_input" in st.session_state:
            query = st.session_state.query_input
            del st.session_state.query_input

        with st.chat_message("user"):
            st.markdown(query, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "user", "content": query})

        # Simple direct processing for now
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.text("🤔 Processing your request...")
            
            try:
                model = get_model()
                if model:
                    # Test the model's Brave configuration
                    if hasattr(model, 'test_brave_configuration'):
                        brave_ok = model.test_brave_configuration()
                        if not brave_ok:
                            st.warning("⚠️ Brave API configuration may have issues")
                    
                    # Simple response for now
                    response = f"Received your query: {query}\n\nThis is a test response. The enhanced MCP integration with Brave Search is being processed."
                    
                    message_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    error_msg = "❌ Could not initialize the AI model"
                    message_placeholder.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except Exception as e:
                error_msg = f"❌ Error processing request: {str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Enhanced footer
st.markdown("---")
st.markdown("### 🚀 Enhanced MCP Client v2.3 - Brave Search Only")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔧 Core Features:**")
    st.caption("• HEDIS Analytics")
    st.caption("• Advanced Calculator")
    st.caption("• System Diagnostics")

with col2:
    st.markdown("**🔍 Search Features:**")
    st.caption("• Brave Web Search Only")
    st.caption("• Brave Local Search Only")
    st.caption("• Cached Weather Service")

with col3:
    st.markdown("**⚡ Enhanced:**")
    st.caption("• Pure Brave Search Integration")
    st.caption("• No Tracking or Data Collection")
    st.caption("• Smart Cache Management")

st.caption(f"📡 **Connection**: {server_url} | 🤖 **Mode**: {prompt_type if 'prompt_type' in locals() else 'Not Selected'} | 📊 **Status**: {status_indicator}")
