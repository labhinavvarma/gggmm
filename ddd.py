# langgraph_supervisor.py

import asyncio
import json
import uuid
import requests
import urllib3
import time
import threading
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import tools_condition
from mcp import ClientSession
from mcp.client.sse import sse_client
import streamlit as st
from mcpserver import run_mcp_server

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Cortex Config
CORTEX_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
APP_ID = "edadip"
APLCTN_CD = "edagnai"
MODEL = "llama3.1-70b"
MCP_URL = "http://localhost:8000/messages/"

# Global MCP server state
_mcp_server_ready = False
_mcp_server_thread = None
_mcp_server_process = None

def start_mcp_server():
    """Start MCP server in background thread without Streamlit context"""
    global _mcp_server_ready
    try:
        # Clear any existing Streamlit context to avoid warnings
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        from streamlit.runtime.scriptrunner import add_script_run_ctx
        
        # Remove context to avoid threading warnings
        ctx = get_script_run_ctx()
        if ctx:
            ctx = None
        
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Set server ready flag
        _mcp_server_ready = True
        
        # Start the MCP server (disable signals since we're in a background thread)
        loop.run_until_complete(run_mcp_server(enable_signals=False))
        
    except Exception as e:
        print(f"❌ Failed to start MCP server: {e}")
        _mcp_server_ready = False

def ensure_mcp_server():
    """Ensure MCP server is running with improved error handling"""
    global _mcp_server_thread, _mcp_server_ready
    
    if "mcp_initialized" not in st.session_state:
        st.session_state.mcp_initialized = True
        
        # Show initialization status
        status_placeholder = st.empty()
        status_placeholder.info("🚀 Initializing MCP Server...")
        
        # Start server thread
        _mcp_server_thread = threading.Thread(target=start_mcp_server, daemon=True)
        _mcp_server_thread.start()
        
        # Wait for server to be ready with better feedback
        max_wait = 15  # increased timeout
        wait_time = 0
        
        while wait_time < max_wait and not _mcp_server_ready:
            time.sleep(0.5)
            wait_time += 0.5
            # Update status every 2 seconds
            if wait_time % 2 == 0:
                status_placeholder.info(f"⏳ Starting MCP server... ({wait_time:.0f}s)")
        
        if not _mcp_server_ready:
            status_placeholder.error("❌ MCP server failed to start within timeout")
            st.error("Please try refreshing the page or check the console for errors.")
            return False
        
        # Test connection with retries
        status_placeholder.info("🔍 Testing MCP connection...")
        connection_ok = False
        for attempt in range(3):
            if test_mcp_connection():
                connection_ok = True
                break
            time.sleep(2)  # Wait between attempts
            status_placeholder.info(f"🔍 Testing connection... (attempt {attempt + 1}/3)")
        
        if connection_ok:
            status_placeholder.success("✅ MCP Server connected and ready!")
            time.sleep(1)  # Show success message briefly
            status_placeholder.empty()
            return True
        else:
            status_placeholder.error("❌ MCP server started but connection test failed")
            st.error("Please check your Neo4j database connection settings.")
            return False
    
    return _mcp_server_ready

def test_mcp_connection():
    """Test MCP server connection with better error handling"""
    try:
        # Give the server a moment to fully start
        time.sleep(2)
        
        # Test connection in a separate event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(test_mcp_async())
        loop.close()
        return result
    except Exception as e:
        print(f"MCP connection test failed: {e}")
        return False

async def test_mcp_async():
    """Async MCP connection test with timeout"""
    try:
        # Test with timeout (use asyncio.wait_for for compatibility)
        async def test_connection():
            async with sse_client(MCP_URL) as sse:
                async with ClientSession(*sse) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    return len(tools.tools) > 0
        
        result = await asyncio.wait_for(test_connection(), timeout=10.0)
        return result
    except asyncio.TimeoutError:
        print("MCP connection test timed out")
        return False
    except Exception as e:
        print(f"MCP async test error: {e}")
        return False

# Define the state
class GraphState(TypedDict):
    messages: list
    user_query: str
    cypher_query: str
    query_result: str
    interpretation: str
    next_action: str
    error_message: str
    step_count: int

class CortexLLM:
    """Custom LLM wrapper for Cortex API"""
    
    def __init__(self):
        self.api_key = API_KEY
        self.url = CORTEX_URL
        
    async def ainvoke(self, messages, system_message=""):
        """Async invoke method for LangGraph compatibility"""
        if isinstance(messages, list):
            prompt = "\n".join([msg.content if hasattr(msg, 'content') else str(msg) for msg in messages])
        else:
            prompt = str(messages)
            
        return await self._call_cortex(prompt, system_message)
    
    async def _call_cortex(self, prompt: str, system_message: str = "") -> str:
        """Call Cortex API with improved error handling"""
        full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
        
        payload = {
            "query": {
                "aplctn_cd": APLCTN_CD,
                "app_id": APP_ID,
                "api_key": API_KEY,
                "method": "cortex",
                "model": MODEL,
                "sys_msg": system_message or "You are a helpful AI assistant for Neo4j database operations.",
                "limit_convs": "0",
                "prompt": {
                    "messages": [{"role": "user", "content": full_prompt}]
                },
                "session_id": str(uuid.uuid4())
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f'Snowflake Token="{API_KEY}"'
        }

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(self.url, headers=headers, json=payload, verify=False, timeout=30)
            )
            response.raise_for_status()
            raw = response.text
            
            # Clean response
            if "end_of_stream" in raw:
                result = raw.split("end_of_stream")[0].strip()
            else:
                result = raw.strip()
            
            # Remove any JSON wrapper if present
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict) and 'content' in parsed:
                    result = parsed['content']
                elif isinstance(parsed, dict) and 'response' in parsed:
                    result = parsed['response']
            except json.JSONDecodeError:
                pass
            
            return result
        except requests.exceptions.Timeout:
            return "❌ Cortex API timeout - please try again"
        except requests.exceptions.RequestException as e:
            return f"❌ Cortex API error: {str(e)}"
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"

# Initialize LLM
llm = CortexLLM()

# Enhanced MCP Client Functions
async def call_mcp_with_retry(tool_name: str, query: str, max_retries: int = 3) -> str:
    """Call MCP server tools with retry logic"""
    for attempt in range(max_retries):
        try:
            result = await call_mcp(tool_name, query)
            if not result.startswith("❌"):
                return result
        except Exception as e:
            if attempt == max_retries - 1:
                return f"❌ MCP error after {max_retries} attempts: {str(e)}"
            await asyncio.sleep(1)  # Wait before retry
    return "❌ MCP failed after all retry attempts"

async def call_mcp(tool_name: str, query: str) -> str:
    """Call MCP server tools with improved error handling"""
    try:
        async with sse_client(MCP_URL) as sse:
            async with ClientSession(*sse) as session:
                await session.initialize()
                tools = await session.list_tools()
                tool = next((t for t in tools.tools if t.name == tool_name), None)
                
                if not tool:
                    available_tools = [t.name for t in tools.tools]
                    return f"❌ Tool '{tool_name}' not found. Available tools: {available_tools}"
                
                result = await session.call_tool(tool.name, {"query": query})
                
                if result and result.content:
                    content = result.content[0].text
                    # Try to parse and format JSON results
                    try:
                        if content.startswith('[') or content.startswith('{'):
                            parsed = json.loads(content)
                            return json.dumps(parsed, indent=2)
                        return content
                    except json.JSONDecodeError:
                        return content
                else:
                    return "✅ Query executed successfully (no results returned)"
                    
    except ConnectionError:
        return "❌ Cannot connect to MCP server - please ensure it's running"
    except Exception as e:
        return f"❌ MCP error: {str(e)}"

# Enhanced Agent Prompts
SUPERVISOR_PROMPT = """You are a Supervisor Agent coordinating Neo4j database operations.

Your specialized team:
1. **Cypher Generator** - Creates optimized Cypher queries
2. **Query Executor** - Executes queries against Neo4j database
3. **Result Interpreter** - Analyzes results and provides insights

Current workflow state:
- User Query: {user_query}
- Has Cypher Query: {has_cypher}
- Has Results: {has_results}
- Step: {step_count}

Decision Rules:
- If no Cypher query exists: route to "cypher_generator"
- If Cypher exists but no results: route to "query_executor"
- If results exist but no interpretation: route to "result_interpreter"
- If interpretation exists: route to "FINISH"

Respond with ONLY one word: cypher_generator, query_executor, result_interpreter, or FINISH"""

CYPHER_GENERATOR_PROMPT = """You are an Expert Cypher Query Generator for Neo4j ConnectIQ database.

Database Schema:
- **Nodes**: Apps, Devices, Users, Categories, Versions, Reviews
- **Relationships**: COMPATIBLE_WITH, BELONGS_TO, HAS_VERSION, REVIEWED_BY, DEVELOPED_BY
- **Properties**: name, version, rating, install_count, category, device_type, release_date

Query Guidelines:
1. Use efficient patterns with proper WHERE clauses
2. Include LIMIT for large datasets (default: 50)
3. Return meaningful properties
4. Use OPTIONAL MATCH when relationships might not exist
5. Consider case-insensitive searches with toLower()

User Request: "{user_query}"

Generate ONLY the Cypher query (no explanations, no markdown):"""

QUERY_EXECUTOR_PROMPT = """Execute the Cypher query using the appropriate MCP tool.

Query Analysis:
- READ operations: MATCH, RETURN, WITH, WHERE
- WRITE operations: CREATE, MERGE, DELETE, SET, REMOVE, DROP

Query: {cypher_query}

Execute and return results."""

RESULT_INTERPRETER_PROMPT = """Analyze Neo4j query results and provide comprehensive insights.

Context:
- User Question: {user_query}
- Executed Query: {cypher_query}
- Raw Results: {query_result}

Provide a clear analysis including:
1. **Summary**: What the data shows
2. **Key Findings**: Important patterns or insights
3. **Recommendations**: Actionable next steps
4. **Data Quality**: Any observations about the data

Make it conversational and business-focused."""

# Enhanced Agent Functions
async def supervisor_agent(state: GraphState) -> GraphState:
    """Enhanced supervisor with better decision logic"""
    has_cypher = bool(state.get("cypher_query", "").strip())
    has_results = bool(state.get("query_result", "").strip())
    has_interpretation = bool(state.get("interpretation", "").strip())
    
    # Increment step counter
    state["step_count"] = state.get("step_count", 0) + 1
    
    prompt = SUPERVISOR_PROMPT.format(
        user_query=state.get("user_query", ""),
        has_cypher=has_cypher,
        has_results=has_results,
        step_count=state["step_count"]
    )
    
    response = await llm.ainvoke(prompt)
    next_action = response.strip().lower()
    
    # Validate response
    valid_actions = ["cypher_generator", "query_executor", "result_interpreter", "finish"]
    if next_action not in valid_actions:
        # Fallback logic
        if not has_cypher:
            next_action = "cypher_generator"
        elif not has_results:
            next_action = "query_executor"
        elif not has_interpretation:
            next_action = "result_interpreter"
        else:
            next_action = "finish"
    
    state["next_action"] = next_action
    state["messages"].append(AIMessage(content=f"🎯 Supervisor: Routing to {next_action.replace('_', ' ').title()}"))
    
    return state

async def cypher_generator_agent(state: GraphState) -> GraphState:
    """Enhanced Cypher generation with validation"""
    prompt = CYPHER_GENERATOR_PROMPT.format(user_query=state["user_query"])
    
    response = await llm.ainvoke(prompt)
    
    # Clean the response
    cypher_query = response.strip()
    
    # Remove code block formatting if present
    if "```" in cypher_query:
        lines = cypher_query.split("\n")
        cypher_lines = []
        in_code_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block or (not in_code_block and line.strip()):
                cypher_lines.append(line)
        cypher_query = "\n".join(cypher_lines).strip()
    
    # Basic validation
    if not cypher_query or len(cypher_query) < 5:
        cypher_query = f"MATCH (n) RETURN n LIMIT 10 // Generated for: {state['user_query']}"
    
    state["cypher_query"] = cypher_query
    state["messages"].append(AIMessage(content=f"🔍 **Generated Cypher Query:**\n```cypher\n{cypher_query}\n```"))
    
    return state

async def query_executor_agent(state: GraphState) -> GraphState:
    """Enhanced query execution with better error handling"""
    cypher_query = state["cypher_query"]
    
    # Determine tool based on query type
    write_keywords = ["CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP"]
    is_write = any(keyword in cypher_query.upper() for keyword in write_keywords)
    tool_name = "write_neo4j_cypher" if is_write else "read_neo4j_cypher"
    
    state["messages"].append(AIMessage(content=f"⚙️ Executing query using {tool_name}..."))
    
    # Execute with retry
    result = await call_mcp_with_retry(tool_name, cypher_query)
    
    state["query_result"] = result
    
    if result.startswith("❌"):
        state["messages"].append(AIMessage(content=f"❌ **Execution Error:**\n{result}"))
    else:
        state["messages"].append(AIMessage(content=f"📊 **Query Results:**\n```json\n{result}\n```"))
    
    return state

async def result_interpreter_agent(state: GraphState) -> GraphState:
    """Enhanced result interpretation"""
    prompt = RESULT_INTERPRETER_PROMPT.format(
        user_query=state["user_query"],
        cypher_query=state["cypher_query"],
        query_result=state["query_result"]
    )
    
    response = await llm.ainvoke(prompt)
    
    state["interpretation"] = response
    state["messages"].append(AIMessage(content=f"💡 **Analysis & Insights:**\n\n{response}"))
    
    return state

# Enhanced routing function
def route_agent(state: GraphState) -> Literal["cypher_generator", "query_executor", "result_interpreter", "end"]:
    """Enhanced routing with validation"""
    next_action = state.get("next_action", "cypher_generator")
    
    # Safety check for infinite loops
    if state.get("step_count", 0) > 10:
        return "end"
    
    if next_action == "finish":
        return "end"
    
    return next_action

# Build the enhanced graph
def create_supervisor_graph():
    """Create enhanced LangGraph supervisor workflow"""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("cypher_generator", cypher_generator_agent)
    workflow.add_node("query_executor", query_executor_agent)
    workflow.add_node("result_interpreter", result_interpreter_agent)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "supervisor",
        route_agent,
        {
            "cypher_generator": "cypher_generator",
            "query_executor": "query_executor", 
            "result_interpreter": "result_interpreter",
            "end": END
        }
    )
    
    # Return to supervisor after each agent
    workflow.add_edge("cypher_generator", "supervisor")
    workflow.add_edge("query_executor", "supervisor")
    workflow.add_edge("result_interpreter", "supervisor")
    
    return workflow.compile()

# Enhanced Streamlit Application
class Neo4jCortexApp:
    """Enhanced application with proper MCP integration"""
    
    def __init__(self):
        self.graph = None
        self.mcp_ready = False
        
    def initialize(self):
        """Initialize the application"""
        # Ensure MCP server is running
        self.mcp_ready = ensure_mcp_server()
        
        if self.mcp_ready:
            self.graph = create_supervisor_graph()
            return True
        return False
    
    async def process_query(self, user_query: str) -> dict:
        """Process user query through the supervisor graph"""
        if not self.mcp_ready or not self.graph:
            return {
                "error_message": "MCP server not ready. Please restart the application.",
                "messages": []
            }
        
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "cypher_query": "",
            "query_result": "",
            "interpretation": "",
            "next_action": "",
            "error_message": "",
            "step_count": 0
        }
        
        try:
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)
            return final_state
        except Exception as e:
            return {
                **initial_state,
                "error_message": f"Processing error: {str(e)}",
                "messages": [AIMessage(content=f"❌ Error: {str(e)}")]
            }
    
    def run_streamlit_app(self):
        """Run the enhanced Streamlit application"""
        st.set_page_config(
            page_title="Neo4j LangGraph Supervisor", 
            page_icon="🧠",
            layout="wide"
        )
        
        st.title("🧠 Neo4j LangGraph Supervisor Agent")
        st.markdown("### Powered by Cortex LLM + MCP + LangGraph")
        
        # Initialize the application
        if "app_initialized" not in st.session_state:
            # Show a loading screen while initializing
            with st.container():
                st.markdown("#### 🚀 Initializing System...")
                
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Initialize MCP Server
                status_text.text("Step 1/3: Starting MCP Server...")
                progress_bar.progress(33)
                
                if self.initialize():
                    # Step 2: Validate connections
                    status_text.text("Step 2/3: Validating connections...")
                    progress_bar.progress(67)
                    time.sleep(1)
                    
                    # Step 3: Ready
                    status_text.text("Step 3/3: System ready!")
                    progress_bar.progress(100)
                    time.sleep(1)
                    
                    # Clear loading screen
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.session_state.app_initialized = True
                    st.success("✅ Application ready!")
                    time.sleep(1)
                    st.rerun()  # Refresh to show main interface
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ Failed to initialize application")
                    st.markdown("**Troubleshooting steps:**")
                    st.markdown("1. Check Neo4j database connection")
                    st.markdown("2. Verify Cortex API credentials")
                    st.markdown("3. Ensure port 8000 is available")
                    st.markdown("4. Try refreshing the page")
                    st.stop()
        
        # Main application interface (only shown after initialization)
        if st.session_state.get("app_initialized", False):
            # Initialize session state
            if "history" not in st.session_state:
                st.session_state.history = []
            
            # Sidebar with status
            with st.sidebar:
                st.markdown("### 🔧 System Status")
                if self.mcp_ready:
                    st.success("✅ MCP Server: Connected")
                    st.success("✅ Cortex LLM: Ready")
                    st.success("✅ LangGraph: Active")
                else:
                    st.error("❌ System not ready")
                
                st.markdown("---")
                st.markdown("### 📊 Quick Stats")
                if st.button("🔍 Test Database"):
                    with st.spinner("Testing..."):
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            result = loop.run_until_complete(call_mcp("health_check", ""))
                            st.code(result)
                        except Exception as e:
                            st.error(f"Test failed: {e}")
                
                st.markdown("---")
                if st.button("🔄 Reset Conversation"):
                    st.session_state.history = []
                    st.rerun()
            
            # Main chat interface
            st.markdown("### 💬 Ask your Neo4j questions:")
            
            # Example queries
            with st.expander("💡 Example Queries"):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📱 Popular Apps"):
                        st.session_state.example_query = "Show me the top 10 most popular apps"
                        st.rerun()
                    if st.button("🏃 Fitness Apps"):
                        st.session_state.example_query = "What are the highest rated fitness apps?"
                        st.rerun()
                with col2:
                    if st.button("📊 Device Compatibility"):
                        st.session_state.example_query = "Which apps are compatible with Garmin devices?"
                        st.rerun()
                    if st.button("👨‍💻 Top Developers"):
                        st.session_state.example_query = "Show me apps by the most successful developers"
                        st.rerun()
            
            # Chat input
            default_query = st.session_state.pop("example_query", "")
            user_query = st.chat_input("Ask me anything about your Neo4j ConnectIQ database...", value=default_query)
            
            if user_query:
                # Add user message to history
                st.session_state.history.append(("user", user_query))
                
                # Process query
                with st.spinner("🤖 Supervisor agents are working..."):
                    try:
                        # Run async function
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(self.process_query(user_query))
                        loop.close()
                        
                        # Process results
                        if result.get("error_message"):
                            st.session_state.history.append(("error", result["error_message"]))
                        else:
                            # Add all messages from the workflow
                            for msg in result.get("messages", []):
                                if isinstance(msg, AIMessage):
                                    content = msg.content
                                    if "Generated Cypher Query" in content:
                                        st.session_state.history.append(("cypher", content))
                                    elif "Query Results" in content:
                                        st.session_state.history.append(("result", content))
                                    elif "Analysis & Insights" in content:
                                        st.session_state.history.append(("interpretation", content))
                                    else:
                                        st.session_state.history.append(("agent", content))
                        
                    except Exception as e:
                        st.session_state.history.append(("error", f"Application error: {str(e)}"))
                
                # Auto-scroll to bottom
                st.rerun()
            
            # Display chat history
            if st.session_state.history:
                st.markdown("### 📝 Conversation History")
                for i, (role, message) in enumerate(reversed(st.session_state.history)):
                    if role == "user":
                        st.chat_message("user").markdown(f"**You:** {message}")
                    elif role == "agent":
                        st.chat_message("assistant").markdown(message)
                    elif role == "cypher":
                        st.chat_message("assistant").markdown(message)
                    elif role == "result":
                        st.chat_message("assistant").markdown(message)
                    elif role == "interpretation":
                        st.chat_message("assistant").markdown(message)
                    elif role == "error":
                        st.chat_message("assistant").markdown(f"❌ **Error:** {message}")
            else:
                st.info("👋 Welcome! Ask me anything about your Neo4j ConnectIQ database to get started.")

# Main execution
def main():
    """Main function"""
    app = Neo4jCortexApp()
    app.run_streamlit_app()

if __name__ == "__main__":
    main()
