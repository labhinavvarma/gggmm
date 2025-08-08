import streamlit as st
import asyncio
import json
import yaml
import traceback

from mcp.client.sse import sse_client
from mcp import ClientSession
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dependencies import SnowFlakeConnector
from llmobjectwrapper import create_enhanced_cortex_model  # Import our enhanced model
from snowflake.snowpark import Session

# Page config
st.set_page_config(page_title="DataFlyWheel MCP Demo", page_icon="🌐", layout="wide")
st.title("🌐 DataFlyWheel MCP Demo - Full Integration")

# Sidebar configuration
server_url = st.sidebar.text_input("MCP Server URL", "http://10.126.192.183:8001/sse")
show_server_info = st.sidebar.checkbox("🛡 Show MCP Server Info", value=False)

# Server Info Section
if show_server_info:
    async def fetch_mcp_info():
        result = {"resources": [], "tools": [], "prompts": [], "yaml": [], "search": []}
        try:
            async with sse_client(url=server_url) as sse_connection:
                async with ClientSession(*sse_connection) as session:
                    await session.initialize()

                    # Resources
                    resources = await session.list_resources()
                    if hasattr(resources, 'resources'):
                        for r in resources.resources:
                            result["resources"].append({"name": r.name})
                   
                    # Tools
                    tools = await session.list_tools()
                    hidden_tools = {"add-frequent-questions", "add-prompts", "suggested_top_prompts"}
                    if hasattr(tools, 'tools'):
                        for t in tools.tools:
                            if t.name not in hidden_tools:
                                result["tools"].append({
                                    "name": t.name,
                                    "description": getattr(t, 'description', '')
                                })

                    # Prompts
                    prompts = await session.list_prompts()
                    if hasattr(prompts, 'prompts'):
                        for p in prompts.prompts:
                            result["prompts"].append({
                                "name": p.name,
                                "description": getattr(p, 'description', ''),
                            })

        except Exception as e:
            st.sidebar.error(f"❌ MCP Connection Error: {e}")
        return result

    mcp_data = asyncio.run(fetch_mcp_info())

    # Display server information in sidebar
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
                st.caption(t['description'][:150] + "..." if len(t['description']) > 150 else t['description'])

    with st.sidebar.expander("🧐 Available Prompts", expanded=False):
        for p in mcp_data["prompts"]:
            st.markdown(f"**{p['name']}**")
            if p.get('description'):
                st.caption(p['description'])

else:
    # Main Application
    @st.cache_resource
    def get_snowflake_connection():
        """Get Snowflake connection for the LLM wrapper."""
        return SnowFlakeConnector.get_conn('aedl', '')

    @st.cache_resource
    def get_enhanced_model():
        """Get enhanced Cortex model with SerpApi support."""
        sf_conn = get_snowflake_connection()
        session = Session.builder.configs({"connection": sf_conn}).getOrCreate()
        return create_enhanced_cortex_model(session, "claude-4-sonnet")

    # Prompt selection with all available prompts
    prompt_type = st.sidebar.radio("Select Expert Mode", [
        "🧮 Calculator", 
        "🏥 HEDIS Expert", 
        "🌤️ Weather Expert", 
        "🌐 Web Search Expert",  # SerpApi integration
        "💬 General Chat"
    ])
    
    # Updated prompt mapping
    prompt_map = {
        "🧮 Calculator": "calculator-prompt",
        "🏥 HEDIS Expert": "hedis-prompt",
        "🌤️ Weather Expert": "weather-prompt",
        "🌐 Web Search Expert": "serpapi-prompt",  # Key SerpApi integration
        "💬 General Chat": None
    }

    # Comprehensive example queries
    examples = {
        "🌐 Web Search Expert": [
            "Who is the current prime minister of India?",  # Main test case
            "Latest developments in artificial intelligence",
            "Current stock price of Tesla",
            "Recent news about climate change",
            "Who won the latest US election?",
            "Best restaurants in New York City",
            "Latest iPhone features and release date"
        ],
        "🧮 Calculator": [
            "Calculate the expression (4+5)/2.0",
            "What is 15% of 85,000?",
            "Calculate compound interest on $10,000 at 5.5% for 3 years",
            "Find the square root of 144",
            "What is 2^10?"
        ],
        "🏥 HEDIS Expert": [
            "What are the codes in BCS Value Set?",
            "What is the age criteria for BCS Measure?",
            "Generate SQL to get all diabetes measures",
            "Describe the COA Measure requirements",
            "What LOB is the EED measure scoped under?"
        ],
        "🌤️ Weather Expert": [
            "What's the weather forecast for Atlanta today?",
            "Current weather conditions in Boston",
            "Is it raining in Seattle right now?",
            "Weather forecast for Miami this weekend",
            "Temperature in Chicago today"
        ],
        "💬 General Chat": [
            "Explain quantum computing in simple terms",
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
            "Tell me about the history of the internet"
        ]
    }

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Chat with {prompt_type}")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

    with col2:
        st.subheader("💡 Example Queries")
        
        # Example queries with better organization
        for example in examples[prompt_type]:
            if st.button(example, key=f"example_{hash(example)}", use_container_width=True):
                st.session_state.query_input = example

        # Workflow status
        st.subheader("📊 Integration Status")
        
        async def check_integration_status():
            status = {"server": False, "tools": 0, "model": False}
            try:
                # Test server connection
                async with sse_client(url=server_url) as sse_connection:
                    async with ClientSession(*sse_connection) as session:
                        await session.initialize()
                        status["server"] = True
                        
                        # Count tools
                        tools = await session.list_tools()
                        if hasattr(tools, 'tools'):
                            status["tools"] = len([t for t in tools.tools 
                                                 if t.name not in {"add-frequent-questions", "add-prompts", "suggested_top_prompts"}])
                
                # Test model
                try:
                    model = get_enhanced_model()
                    status["model"] = model is not None
                except:
                    pass
                    
            except:
                pass
            return status
        
        try:
            status = asyncio.run(check_integration_status())
            
            server_status = "🟢 Online" if status["server"] else "🔴 Offline"
            st.write(f"**MCP Server:** {server_status}")
            st.write(f"**Available Tools:** {status['tools']}")
            
            model_status = "🟢 Ready" if status["model"] else "🔴 Not Ready"
            st.write(f"**LLM Model:** {model_status}")
            
        except:
            st.write("**Status:** ⚠️ Checking...")

    # Chat input - ENHANCED WORKFLOW
    with col1:
        if query := st.chat_input("Type your query here...") or "query_input" in st.session_state:

            if "query_input" in st.session_state:
                query = st.session_state.query_input
                del st.session_state.query_input
           
            with st.chat_message("user"):
                st.markdown(query, unsafe_allow_html=True)
           
            st.session_state.messages.append({"role": "user", "content": query})
       
            async def enhanced_workflow_processing(query_text):
                """Enhanced workflow with better integration and error handling."""
                
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    
                    try:
                        # Step 1: Initialize components
                        message_placeholder.text("🔄 Initializing MCP client and enhanced LLM...")
                        
                        client = MultiServerMCPClient(
                            {"DataFlyWheelServer": {"url": server_url, "transport": "sse"}}
                        )
                        
                        # Get enhanced model with SerpApi support
                        model = get_enhanced_model()
                        
                        message_placeholder.text("🛠️ Loading tools and creating agent...")
                        
                        # Get tools and create agent
                        tools = await client.get_tools()
                        print(f"🔧 Retrieved {len(tools)} tools from MCP server")
                        
                        # Create agent with enhanced model
                        agent = create_react_agent(model=model, tools=tools)
                        
                        message_placeholder.text("📋 Processing prompt...")
                        
                        # Get prompt from server
                        prompt_name = prompt_map[prompt_type]
                        
                        if prompt_name is None:
                            # Direct query
                            messages_for_agent = [{"role": "user", "content": query_text}]
                        else:
                            # Get specific prompt from MCP server
                            prompt_from_server = await client.get_prompt(
                                server_name="DataFlyWheelServer",
                                prompt_name=prompt_name,
                                arguments={"query": query_text}
                            )
                            messages_for_agent = prompt_from_server
                        
                        message_placeholder.text("🧠 AI agent processing with enhanced LLM wrapper...")
                        
                        # Process with enhanced agent
                        response = await agent.ainvoke({"messages": messages_for_agent})
                        
                        # Enhanced response extraction
                        result = None
                        
                        try:
                            # Method 1: Standard LangGraph response
                            if hasattr(response, 'messages') and response.messages:
                                last_message = response.messages[-1]
                                if hasattr(last_message, 'content'):
                                    result = last_message.content
                                elif isinstance(last_message, dict) and 'content' in last_message:
                                    result = last_message['content']
                            
                            # Method 2: Dictionary response
                            elif isinstance(response, dict):
                                if 'messages' in response and response['messages']:
                                    last_message = response['messages'][-1]
                                    if isinstance(last_message, dict):
                                        result = last_message.get('content', str(last_message))
                                elif 'output' in response:
                                    result = response['output']
                                elif 'content' in response:
                                    result = response['content']
                            
                            # Method 3: Direct response
                            elif hasattr(response, 'content'):
                                result = response.content
                            
                            # Method 4: String response
                            elif isinstance(response, str):
                                result = response
                            
                            # Fallback
                            else:
                                result = str(response)
                                
                        except Exception as extraction_error:
                            result = f"⚠️ Response processing completed but extraction had issues: {extraction_error}\n\nAttempting fallback extraction..."
                            try:
                                # Aggressive fallback
                                result = str(response)
                            except:
                                result = "❌ Could not extract response content"
                        
                        # Display result
                        if result and result.strip():
                            message_placeholder.markdown(result)
                            st.session_state.messages.append({"role": "assistant", "content": result})
                            
                            # Show success in sidebar
                            with col2:
                                st.success("✅ Query processed successfully!")
                                
                        else:
                            error_msg = f"⚠️ No response content generated\n\nResponse type: {type(response)}"
                            message_placeholder.markdown(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        
                    except Exception as e:
                        error_details = f"""
❌ **Workflow Error:**

**Error:** {str(e)}

**Query:** {query_text}
**Mode:** {prompt_type}
**Server:** {server_url}

**Troubleshooting:**
1. Check MCP server is running
2. Verify Snowflake connection
3. Ensure tools are properly registered
4. Check server logs for detailed errors

**Debug Info:**
```
{traceback.format_exc()}
```
"""
                        message_placeholder.markdown(error_details)
                        st.session_state.messages.append({"role": "assistant", "content": error_details})
                        
                        # Show error in sidebar
                        with col2:
                            st.error("❌ Query processing failed")

            if query:
                asyncio.run(enhanced_workflow_processing(query))

    # Control buttons
    with col1:
        col_clear, col_refresh = st.columns(2)
        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        with col_refresh:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

    # Workflow visualization
    with col2:
        with st.expander("🔄 Enhanced Workflow", expanded=False):
            st.markdown("""
            **🌐 Full Integration Workflow:**
            
            ```
            1. User Query → Streamlit Client
            2. Client → MCP Server (Request Prompt)
            3. Server → Client (Return Enhanced Prompt)  
            4. Client → Enhanced LLM Wrapper
            5. LLM → MCP Server (Call SerpApiSearch)
            6. Server → SerpApi (Formatted HTTP Request)
            7. SerpApi → Server (JSON Response)
            8. Server → LLM (Processed JSON + Insights)
            9. Enhanced LLM → Snowflake Cortex (Query)
            10. Cortex → LLM (AI Response)
            11. LLM → Client (Final Answer)
            12. Client → User (Display Result)
            ```
            
            **🔧 Enhancements:**
            - ✅ SerpApi JSON insight extraction
            - ✅ Enhanced tool binding
            - ✅ Robust error handling  
            - ✅ Better response parsing
            - ✅ Real-time status monitoring
            """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🌐 MCP Integration:** Full workflow support")
with col2:
    st.markdown("**🧠 Enhanced LLM:** SerpApi JSON processing")
with col3:
    st.markdown("**⚡ Real-time:** Live tool execution")
