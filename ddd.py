import streamlit as st
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional

try:
    from mcp.client.sse import sse_client
    from mcp import ClientSession
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.prebuilt import create_react_agent
    from dependencies import SnowFlakeConnector
    from llmobjectwrapper import ChatSnowflakeCortex
    from snowflake.snowpark import Session
except ImportError as e:
    st.error(f"Missing required packages. Install with: pip install mcp langchain-mcp-adapters langgraph snowflake-snowpark-python")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit-mcp-client")

# Page config
st.set_page_config(
    page_title="Brave Search MCP Demo", 
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Brave Search MCP Demo")
st.markdown("*Powered by Model Context Protocol*")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
server_url = st.sidebar.text_input(
    "MCP Server URL", 
    value="http://localhost:8081/sse",
    help="URL of your MCP server SSE endpoint"
)

show_server_info = st.sidebar.checkbox("🛡 Show MCP Server Info", value=False)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "tools_cache" not in st.session_state:
    st.session_state.tools_cache = []

if "prompts_cache" not in st.session_state:
    st.session_state.prompts_cache = []

# === Snowflake LLM Setup ===
@st.cache_resource
def get_snowflake_connection():
    """Get Snowflake connection"""
    return SnowFlakeConnector.get_conn('aedl', '')

@st.cache_resource
def get_model():
    """Get Snowflake Cortex LLM model"""
    sf_conn = get_snowflake_connection()
    return ChatSnowflakeCortex(
        model="claude-4-sonnet",
        cortex_function="complete",
        session=Session.builder.configs({"connection": sf_conn}).getOrCreate()
    )

# === MCP Server Info Functions ===
async def fetch_mcp_info():
    """Fetch information from the MCP server"""
    result = {"tools": [], "prompts": [], "connection_status": "disconnected"}
    
    try:
        st.sidebar.info("🔌 Connecting to MCP server...")
        
        async with sse_client(url=server_url) as sse_connection:
            async with ClientSession(*sse_connection) as session:
                await session.initialize()
                result["connection_status"] = "connected"
                
                # Fetch Tools
                try:
                    tools = await session.list_tools()
                    if hasattr(tools, 'tools'):
                        for t in tools.tools:
                            tool_info = {
                                "name": t.name,
                                "description": getattr(t, 'description', 'No description available')
                            }
                            
                            # Get parameters from input schema
                            if hasattr(t, 'inputSchema') and hasattr(t.inputSchema, 'properties'):
                                tool_info["parameters"] = list(t.inputSchema.properties.keys())
                            
                            result["tools"].append(tool_info)
                except Exception as e:
                    st.sidebar.error(f"❌ Failed to fetch tools: {e}")
                
                # Fetch Prompts
                try:
                    prompts = await session.list_prompts()
                    if hasattr(prompts, 'prompts'):
                        for p in prompts.prompts:
                            prompt_info = {
                                "name": p.name,
                                "description": getattr(p, 'description', 'No description available'),
                                "arguments": []
                            }
                            
                            if hasattr(p, 'arguments'):
                                for arg in p.arguments:
                                    prompt_info["arguments"].append({
                                        "name": arg.name,
                                        "required": getattr(arg, 'required', False),
                                        "description": getattr(arg, 'description', '')
                                    })
                            
                            result["prompts"].append(prompt_info)
                except Exception as e:
                    st.sidebar.error(f"❌ Failed to fetch prompts: {e}")
                
        st.sidebar.success("✅ Connected to MCP server")
        
    except Exception as e:
        st.sidebar.error(f"❌ MCP Connection Error: {e}")
        result["connection_status"] = "failed"
    
    return result

# === Mock LLM for demonstration ===
def mock_llm_response(prompt_text: str) -> str:
    """Mock LLM response for demo purposes"""
    return f"🤖 Mock LLM Response to: '{prompt_text[:100]}...'"

# === Server Info Display ===
if show_server_info:
    mcp_data = asyncio.run(fetch_mcp_info())
    
    # Cache the results for use in the main interface
    st.session_state.tools_cache = mcp_data["tools"]
    st.session_state.prompts_cache = mcp_data["prompts"]
    
    # Connection Status
    if mcp_data["connection_status"] == "connected":
        st.sidebar.success("🟢 MCP Server Connected")
    elif mcp_data["connection_status"] == "failed":
        st.sidebar.error("🔴 MCP Server Disconnected")
    else:
        st.sidebar.warning("🟡 MCP Server Status Unknown")
    
    # Display Tools
    with st.sidebar.expander("🛠 Available Tools", expanded=False):
        if mcp_data["tools"]:
            for tool in mcp_data["tools"]:
                st.markdown(f"**{tool['name']}**")
                st.markdown(f"*{tool['description']}*")
                if 'parameters' in tool:
                    st.markdown(f"Parameters: `{', '.join(tool['parameters'])}`")
                st.markdown("---")
        else:
            st.warning("No tools available")
    
    # Display Prompts
    with st.sidebar.expander("📝 Available Prompts", expanded=False):
        if mcp_data["prompts"]:
            for prompt in mcp_data["prompts"]:
                st.markdown(f"**{prompt['name']}**")
                st.markdown(f"*{prompt['description']}*")
                if prompt['arguments']:
                    for arg in prompt['arguments']:
                        req_text = "Required" if arg['required'] else "Optional"
                        st.markdown(f"• `{arg['name']}` ({req_text}): {arg['description']}")
                st.markdown("---")
        else:
            st.warning("No prompts available")

# === Main Chat Interface ===
else:
    # Prompt Type Selection
    st.sidebar.header("🎯 Prompt Types")
    prompt_type = st.sidebar.radio(
        "Select Prompt Type", 
        ["Calculator", "Weather", "Web Search", "Local Search", "No Context"]
    )
    
    # Prompt mapping for server prompts
    prompt_map = {
        "Calculator": "calculation_helper",
        "Weather": "weather_assistant", 
        "Web Search": "brave_search_expert",
        "Local Search": "local_business_finder",
        "No Context": None
    }
    
    # Example queries based on prompt type
    examples = {
        "Calculator": [
            "Calculate the expression (4+5)/2.0", 
            "Calculate the math function sqrt(16) + 7", 
            "Calculate the expression 3^4 - 12"
        ],
        "Weather": [
            "What is the present weather in Richmond?",
            "What's the weather forecast for Atlanta?",
            "Is it raining in New York City today?"
        ],
        "Web Search": [
            "Latest developments in artificial intelligence",
            "Python programming best practices 2024",
            "Climate change recent research",
            "Cryptocurrency market trends"
        ],
        "Local Search": [
            "Pizza restaurants in New York City",
            "Coffee shops near San Francisco", 
            "Gas stations in Atlanta",
            "Bookstores in Boston"
        ],
        "No Context": [
            "Who won the world cup in 2022?", 
            "Summarize climate change impact on oceans"
        ]
    }
    
    # Display example queries
    with st.sidebar.expander("💡 Example Queries", expanded=True):
        for example in examples[prompt_type]:
            if st.button(example, key=f"example_{example}", use_container_width=True):
                st.session_state.query_input = example

# === Chat Interface ===
st.header("💬 Chat Interface")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle query input
query = st.chat_input("Type your query here...") or st.session_state.get("query_input")

if query:
    # Clear the query_input from session state if it was used
    if "query_input" in st.session_state:
        del st.session_state.query_input
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Process the query
    async def process_query(query_text, prompt_type):
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.text("🤔 Processing your request...")
            
            try:
                # Create MCP client
                client = MultiServerMCPClient(
                    {"BraveSearchServer": {"url": server_url, "transport": "sse"}}
                )
                
                # Get model and create agent
                model = get_model()
                tools = await client.get_tools()
                agent = create_react_agent(model=model, tools=tools)
                
                # Get prompt from server or use query directly
                prompt_name = prompt_map[prompt_type]
                
                if prompt_name is None:
                    # No context - just use the query as-is
                    prompt_from_server = query_text
                    formatted_prompt = query_text
                else:
                    # Get prompt from server
                    try:
                        prompt_from_server = await client.get_prompt(
                            server_name="BraveSearchServer",
                            prompt_name=prompt_name,
                            arguments={"query": query_text} if prompt_name == "brave_search_expert" else
                                     {"location": query_text} if prompt_name == "weather_assistant" else
                                     {"problem": query_text} if prompt_name == "calculation_helper" else
                                     {"location": query_text.split(" in ")[-1] if " in " in query_text else "Unknown", 
                                      "business_type": query_text.split(" in ")[0] if " in " in query_text else query_text} if prompt_name == "local_business_finder" else
                                     {"query": query_text}
                        )
                        
                        if prompt_from_server and len(prompt_from_server) > 0:
                            # Format the prompt if it contains {query} placeholder
                            if hasattr(prompt_from_server[0], 'content'):
                                prompt_content = prompt_from_server[0].content
                                if "{query}" in prompt_content:
                                    formatted_prompt = prompt_content.format(query=query_text)
                                else:
                                    formatted_prompt = prompt_content
                            else:
                                formatted_prompt = str(prompt_from_server[0])
                        else:
                            formatted_prompt = query_text
                            
                    except Exception as e:
                        st.warning(f"Could not retrieve prompt '{prompt_name}': {e}")
                        formatted_prompt = query_text
                
                # Invoke the agent
                if prompt_name is None:
                    # For "No Context", just return a simple response
                    result = f"I understand you're asking about: {query_text}. However, I'm configured to use MCP tools for responses. Please try one of the other prompt types for tool-assisted answers."
                else:
                    # Use the agent with the formatted prompt
                    response = await agent.ainvoke({"messages": [{"role": "user", "content": formatted_prompt}]})
                    
                    # Extract the result from the response
                    if isinstance(response, dict):
                        # Handle different response formats
                        if "messages" in response:
                            messages = response["messages"]
                            if messages and len(messages) > 0:
                                last_message = messages[-1]
                                if hasattr(last_message, 'content'):
                                    result = last_message.content
                                else:
                                    result = str(last_message)
                            else:
                                result = "No response generated"
                        else:
                            # Try to get the last value if it's a dict
                            values = list(response.values())
                            if values:
                                last_value = values[-1]
                                if isinstance(last_value, list) and len(last_value) > 1:
                                    result = last_value[1].content if hasattr(last_value[1], 'content') else str(last_value[1])
                                else:
                                    result = str(last_value)
                            else:
                                result = str(response)
                    else:
                        result = str(response)
                
                message_placeholder.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
                        
            except Exception as e:
                error_message = f"❌ **Error**: {str(e)}\n\n💡 Make sure the MCP server is running at `{server_url}` and Snowflake connection is configured."
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Run the async function
    if query:
        asyncio.run(process_query(query, prompt_type))

# === Sidebar Actions ===
st.sidebar.markdown("---")
st.sidebar.header("🎛️ Actions")

if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

if st.sidebar.button("🔄 Refresh Server Info", use_container_width=True):
    st.session_state.tools_cache = []
    st.session_state.prompts_cache = []
    st.rerun()

# === Footer ===
st.sidebar.markdown("---")
st.sidebar.markdown("**🔍 Brave Search MCP Demo**")
st.sidebar.markdown("*Powered by Snowflake Cortex (Claude-4-Sonnet)*")
st.sidebar.markdown("*Model Context Protocol + LangGraph*")
st.sidebar.markdown(f"Server: `{server_url}`")

# === Instructions ===
with st.expander("📖 How to Use", expanded=False):
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Start the MCP Server**: Make sure your MCP server is running on the configured URL
    2. **Enable Server Info**: Check "Show MCP Server Info" to see available tools and prompts
    3. **Select Interaction Mode**: Choose how you want to interact with the server
    4. **Try Examples**: Click on example queries in the sidebar
    5. **Ask Questions**: Type your own questions in the chat input
    
    ### 🛠️ Available Modes
    
    - **Calculator**: Perform mathematical calculations using prompts and tools
    - **Weather**: Get weather information with AI assistance
    - **Web Search**: Search the internet using AI-guided queries
    - **Local Search**: Find local businesses with intelligent assistance
    - **No Context**: Basic chat without MCP tools (limited functionality)
    
    ### 🧠 AI Integration
    
    This app uses **Snowflake Cortex (Claude-4-Sonnet)** with **LangGraph agents** to:
    - Understand your queries intelligently
    - Select appropriate MCP tools automatically  
    - Provide contextual, helpful responses
    - Chain multiple tool calls when needed
    
    ### 💡 Tips
    
    - Use specific, clear queries for better results
    - Check server info to see what tools are available
    - Try different interaction modes for different types of questions
    - Clear chat history with the sidebar button
    """)
