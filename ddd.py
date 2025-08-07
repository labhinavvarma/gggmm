import streamlit as st
import asyncio
import json
import yaml
import requests
import httpx
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# MCP imports with fallback handling
try:
    from mcp.client.sse import sse_client
    from mcp import ClientSession
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.prebuilt import create_react_agent
    MCP_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ MCP libraries not available: {e}. Running in standalone mode.")
    MCP_AVAILABLE = False

from dependencies import SnowFlakeConnector
from llmobjectwrapper import ChatSnowflakeCortex
from snowflake.snowpark import Session
from loguru import logger

# Page config
st.set_page_config(page_title="MCP Enhanced Demo", page_icon="🔍")
st.title("🔍 MCP Enhanced Demo")

# === CONFIGURATION ===
server_url = st.sidebar.text_input("MCP Server URL", "http://10.126.192.183:8001/sse")
use_mcp = st.sidebar.checkbox("🔌 Use MCP Server", value=MCP_AVAILABLE, disabled=not MCP_AVAILABLE)
show_server_info = st.sidebar.checkbox("🛡️ Show MCP Server Info", value=False)

if not MCP_AVAILABLE:
    st.sidebar.warning("⚠️ MCP libraries not installed. Using standalone mode.")
    use_mcp = False

# Connection status
mcp_connection_status = "disconnected"
if use_mcp:
    try:
        # Quick connection test
        async def test_mcp_connection():
            try:
                async with sse_client(url=server_url) as sse_connection:
                    async with ClientSession(*sse_connection) as session:
                        await session.initialize()
                        return True
            except:
                return False
        
        # Only test connection if MCP is enabled
        if use_mcp:
            mcp_connection_status = "connected" if asyncio.run(test_mcp_connection()) else "failed"
    except:
        mcp_connection_status = "failed"

# Display connection status
status_colors = {"connected": "🟢", "failed": "🔴", "disconnected": "⚪"}
st.sidebar.markdown(f"{status_colors[mcp_connection_status]} **MCP Status:** {mcp_connection_status.title()}")

# === RATE LIMITER ===
class RateLimiter:
    """Rate limiter to prevent overwhelming external services"""
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    async def acquire(self):
        now = datetime.now()
        self.requests = [req for req in self.requests if now - req < timedelta(minutes=1)]
        if len(self.requests) >= self.requests_per_minute:
            wait_time = 60 - (now - self.requests[0]).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        self.requests.append(now)

@st.cache_resource
def get_rate_limiter():
    return RateLimiter(requests_per_minute=20)

rate_limiter = get_rate_limiter()

# === STANDALONE TOOL IMPLEMENTATIONS ===

async def standalone_wikipedia_search(query: str, max_results: int = 3) -> str:
    """Standalone Wikipedia search implementation"""
    try:
        await rate_limiter.acquire()
        headers = {"User-Agent": "Streamlit Wikipedia Client (demo@example.com)"}
        search_url = "https://en.wikipedia.org/api/rest_v1/page/search"
        search_params = {"q": query, "limit": max_results}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            search_response = await client.get(search_url, params=search_params, headers=headers)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            if not search_data.get('pages'):
                return f"❌ No Wikipedia results found for: {query}"
            
            results = [f"📖 **Wikipedia Search Results for '{query}':**\n"]
            
            for i, page in enumerate(search_data['pages'][:max_results], 1):
                title = page.get('title', 'Unknown')
                description = page.get('description', 'No description available')
                excerpt = page.get('excerpt', 'No excerpt available')
                
                article_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
                try:
                    article_response = await client.get(article_url, headers=headers)
                    if article_response.status_code == 200:
                        article_data = article_response.json()
                        extract = article_data.get('extract', excerpt)
                        page_url = article_data.get('content_urls', {}).get('desktop', {}).get('page', '')
                    else:
                        extract = excerpt
                        page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                except:
                    extract = excerpt
                    page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                
                results.extend([
                    f"**{i}. {title}**",
                    f"*Description:* {description}",
                    f"*URL:* {page_url}",
                    f"*Content:* {extract}",
                    ""
                ])
            
            return "\n".join(results)
    except Exception as e:
        return f"❌ Wikipedia search error: {str(e)}"

async def standalone_duckduckgo_search(query: str, max_results: int = 10) -> str:
    """Standalone DuckDuckGo search implementation with current results"""
    try:
        await rate_limiter.acquire()
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        
        results = [f"🦆 **DuckDuckGo Search Results for '{query}':**\n"]
        
        # Enhanced query with current year for recent results
        current_year = datetime.now().year
        enhanced_query = f"{query} {current_year}"
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Method 1: Instant Answer API
            search_url = "https://api.duckduckgo.com/"
            search_params = {
                "q": enhanced_query,
                "format": "json",
                "no_redirect": "1",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            try:
                response = await client.get(search_url, params=search_params, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                if data.get('Abstract'):
                    results.extend([f"**Definition:** {data['Abstract']}", f"**Source:** {data.get('AbstractURL', '')}", ""])
                
                if data.get('Answer'):
                    results.extend([f"**Direct Answer:** {data['Answer']}", f"**Type:** {data.get('AnswerType', '')}", ""])
                
                if data.get('RelatedTopics'):
                    results.append("**Related Information:**")
                    for i, topic in enumerate(data['RelatedTopics'][:max_results//2], 1):
                        if isinstance(topic, dict) and topic.get('Text'):
                            results.append(f"{i}. {topic['Text']}")
                            if topic.get('FirstURL'):
                                results.append(f"   Source: {topic['FirstURL']}")
                    results.append("")
            except:
                pass
            
            # Method 2: HTML Search for current results
            try:
                html_search_url = "https://html.duckduckgo.com/html/"
                html_params = {"q": enhanced_query, "s": "0", "dc": str(max_results), "v": "l"}
                
                html_response = await client.get(html_search_url, params=html_params, headers=headers)
                if html_response.status_code == 200:
                    html_content = html_response.text
                    
                    # Extract links and titles
                    patterns = [
                        r'<a[^>]+href="([^"]+)"[^>]*class="[^"]*result[^"]*"[^>]*>([^<]+)</a>',
                        r'<a[^>]+href="(https?://[^"]+)"[^>]*>([^<]+)</a>'
                    ]
                    
                    web_results = []
                    urls_seen = set()
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, html_content, re.IGNORECASE | re.DOTALL)
                        
                        for url, title in matches:
                            if (url.startswith('http') and url not in urls_seen and 
                                len(title.strip()) > 10 and not url.startswith('https://duckduckgo.com')):
                                
                                urls_seen.add(url)
                                web_results.append(f"• **{title.strip()}**\n  URL: {url}")
                                
                                if len(web_results) >= max_results:
                                    break
                        
                        if len(web_results) >= max_results:
                            break
                    
                    if web_results:
                        results.append("**Current Web Search Results:**")
                        results.extend(web_results)
            except:
                pass
            
            return "\n".join(results) if len(results) > 1 else f"❌ No results found for: {query}"
    except Exception as e:
        return f"❌ DuckDuckGo search error: {str(e)}"

def standalone_get_weather(latitude: float, longitude: float) -> str:
    """Standalone weather implementation"""
    try:
        headers = {"User-Agent": "Streamlit Weather Client", "Accept": "application/geo+json"}
        points_url = f"https://api.weather.gov/points/{latitude},{longitude}"
        points_response = requests.get(points_url, headers=headers, timeout=10)
        points_response.raise_for_status()
        points_data = points_response.json()
        
        forecast_url = points_data['properties']['forecast']
        location_info = points_data['properties']['relativeLocation']['properties']
        location_name = f"{location_info['city']}, {location_info['state']}"
        
        forecast_response = requests.get(forecast_url, headers=headers, timeout=10)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()
        
        periods = forecast_data['properties']['periods']
        current_period = periods[0] if periods else None
        
        if current_period:
            result = f"🌤️ **Weather for {location_name}:**\n\n"
            result += f"**Period:** {current_period['name']}\n"
            result += f"**Temperature:** {current_period['temperature']}°{current_period['temperatureUnit']}\n"
            result += f"**Forecast:** {current_period['detailedForecast']}"
            
            if len(periods) > 1:
                result += f"\n\n**Next Period ({periods[1]['name']}):**\n"
                result += f"**Temperature:** {periods[1]['temperature']}°{periods[1]['temperatureUnit']}\n"
                result += f"**Forecast:** {periods[1]['shortForecast']}"
            
            return result
        else:
            return f"❌ Weather data unavailable for {location_name}"
    except Exception as e:
        return f"❌ Weather error: {str(e)}"

def standalone_calculate(expression: str) -> str:
    """Standalone calculator implementation"""
    try:
        allowed_chars = "0123456789+-*/(). "
        if any(char not in allowed_chars for char in expression):
            return "❌ Invalid characters in expression. Only numbers and basic operators allowed."
        result = eval(expression)
        return f"🧮 **Calculation Result:**\n\nExpression: `{expression}`\nResult: **{result}**"
    except Exception as e:
        return f"❌ Calculation error: {str(e)}"

# === HEDIS TOOLS ===
@st.cache_resource
def get_hedis_connection():
    return snowflake_conn(logger, aplctn_cd="aedl", env="preprod", region_name="us-east-1", warehouse_size_suffix="", prefix="")

def standalone_hedis_text2sql(prompt: str) -> str:
    """Standalone HEDIS text2sql implementation"""
    try:
        HOST = "carelon-eda-preprod.privatelink.snowflakecomputing.com"
        conn = get_hedis_connection()
        db, schema = 'POC_SPC_SNOWPARK_DB', 'HEDIS_SCHEMA'
        stage_name, file_name = "hedis_stage_full", "hedis_semantic_model_complete.yaml"
        
        request_body = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "semantic_model_file": f"@{db}.{schema}.{stage_name}/{file_name}",
        }
        
        resp = requests.post(
            url=f"https://{HOST}/api/v2/cortex/analyst/message",
            json=request_body,
            headers={"Authorization": f'Snowflake Token="{conn.rest.token}"', "Content-Type": "application/json"},
        )
        
        result = resp.json()
        if 'message' in result and 'content' in result['message']:
            content = result['message']['content']
            if isinstance(content, list):
                sql_query = explanation = ""
                for item in content:
                    if item.get('type') == 'sql':
                        sql_query = item.get('statement', '')
                    elif item.get('type') == 'text':
                        explanation = item.get('text', '')
                
                formatted_result = f"🏥 **HEDIS SQL Analysis:**\n\n"
                if explanation:
                    formatted_result += f"**Explanation:** {explanation}\n\n"
                if sql_query:
                    formatted_result += f"**Generated SQL:**\n```sql\n{sql_query}\n```"
                return formatted_result
        
        return f"🏥 **HEDIS Response:**\n\n{json.dumps(result, indent=2)}"
    except Exception as e:
        return f"❌ HEDIS Analyst error: {str(e)}"

def standalone_hedis_search(query: str) -> str:
    """Standalone HEDIS search implementation"""
    try:
        conn = get_hedis_connection()
        root = Root(conn)
        search_service = root.databases['POC_SPC_SNOWPARK_DB'].schemas['HEDIS_SCHEMA'].cortex_search_services['CS_HEDIS_FULL_2024']
        response = search_service.search(query=query, columns=['chunk'], limit=2)
        result_data = json.loads(response.to_json())
        
        formatted_result = f"🔍 **HEDIS Document Search Results for '{query}':**\n\n"
        if 'results' in result_data:
            for i, item in enumerate(result_data['results'], 1):
                chunk = item.get('chunk', 'No content available')
                formatted_result += f"**Result {i}:**\n{chunk}\n\n"
        else:
            formatted_result += "No results found in HEDIS documents."
        return formatted_result
    except Exception as e:
        return f"❌ HEDIS Search error: {str(e)}"

# === MCP SERVER INFO ===
if show_server_info and use_mcp and mcp_connection_status == "connected":
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
                                result["tools"].append({"name": t.name, "description": getattr(t, 'description', '')})

                    # Prompts
                    prompts = await session.list_prompts()
                    if hasattr(prompts, 'prompts'):
                        for p in prompts.prompts:
                            args = []
                            if hasattr(p, 'arguments'):
                                for arg in p.arguments:
                                    args.append(f"{arg.name} ({'Required' if arg.required else 'Optional'}): {arg.description}")
                            result["prompts"].append({"name": p.name, "description": getattr(p, 'description', ''), "args": args})

                    # YAML Resources
                    try:
                        yaml_content = await session.read_resource("schematiclayer://cortex_analyst/schematic_models/hedis_stage_full/list")
                        if hasattr(yaml_content, 'contents'):
                            for item in yaml_content.contents:
                                if hasattr(item, 'text'):
                                    parsed = yaml.safe_load(item.text)
                                    result["yaml"].append(yaml.dump(parsed, sort_keys=False))
                    except Exception as e:
                        result["yaml"].append(f"YAML error: {e}")

                    # Search Objects
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

    # Display MCP Server Info
    with st.sidebar.expander("📦 MCP Resources", expanded=False):
        for r in mcp_data["resources"]:
            if "cortex_search/search_obj/list" in r["name"]:
                display_name = "Cortex Search"
            elif "schematic_models" in r["name"]:
                display_name = "HEDIS Schematic Models"
            elif "frequent_questions" in r["name"]:
                display_name = "Frequent Questions"
            elif "prompts" in r["name"]:
                display_name = "Prompt Templates"
            else:
                display_name = r["name"]
            st.markdown(f"**{display_name}**")

    with st.sidebar.expander("📋 Schematic Layer", expanded=False):
        for y in mcp_data["yaml"]:
            st.code(y, language="yaml")

    with st.sidebar.expander("🛠 MCP Tools", expanded=False):
        tool_categories = {
            "HEDIS & Analytics": ["DFWAnalyst", "DFWSearch", "calculator"],
            "Search & Web": ["wikipedia_search", "duckduckgo_search"],
            "Weather": ["get_weather"],
            "System": ["test_tool", "diagnostic"]
        }
        
        for category, tool_names in tool_categories.items():
            st.markdown(f"**{category}:**")
            for t in mcp_data["tools"]:
                if t['name'] in tool_names:
                    st.markdown(f"  • {t['name']}")
                    if t.get('description'):
                        st.caption(f"    {t['description']}")

    with st.sidebar.expander("🧐 MCP Prompts", expanded=False):
        for p in mcp_data["prompts"]:
            prompt_display_names = {
                "hedis-prompt": "🏥 HEDIS Expert",
                "caleculator-promt": "🧮 Calculator", 
                "weather-prompt": "🌤️ Weather Expert",
                "wikipedia-search-prompt": "📖 Wikipedia Expert",
                "duckduckgo-search-prompt": "🦆 Web Search Expert"
            }
            display_name = prompt_display_names.get(p['name'], p['name'])
            st.markdown(f"**{display_name}**")
            if p.get('description'):
                st.caption(p['description'])

# === MAIN APPLICATION ===
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

# UI Configuration
prompt_type = st.sidebar.radio(
    "Select Expert Mode", 
    ["Calculator", "HEDIS Expert", "Weather", "Wikipedia Search", "Web Search", "No Context"],
    help="Choose the type of expert assistance you need"
)

prompt_map = {
    "Calculator": "caleculator-promt",
    "HEDIS Expert": "hedis-prompt", 
    "Weather": "weather-prompt",
    "Wikipedia Search": "wikipedia-search-prompt",
    "Web Search": "duckduckgo-search-prompt",
    "No Context": None
}

# Examples
examples = {
    "Calculator": ["Calculate 15% of 847", "(25 + 75) * 2.5", "3^2 + 4^2", "What is 1000 * 1.05^3"],
    "HEDIS Expert": ["What are the codes in BCS Value Set?", "Explain the BCS measure requirements", "What is the age criteria for CBP measure?", "List HEDIS measures for diabetes"],
    "Weather": ["Weather for Richmond, Virginia (37.5407, -77.4360)", "Current conditions in Atlanta, Georgia (33.7490, -84.3880)", "Weather forecast for New York City (40.7128, -74.0060)", "What's the weather in Denver, Colorado (39.7392, -104.9903)"],
    "Wikipedia Search": ["Search Wikipedia for artificial intelligence", "What is quantum computing according to Wikipedia?", "Find Wikipedia information about climate change", "Look up machine learning on Wikipedia"],
    "Web Search": ["Latest developments in AI 2024", "Current renewable energy trends", "Recent space exploration missions", "Today's stock market news"],
    "No Context": ["Who won the 2022 World Cup?", "Calculate 25 * 4", "Weather in Denver", "Define machine learning"]
}

# Load HEDIS examples dynamically if using MCP
if prompt_type == "HEDIS Expert" and use_mcp and mcp_connection_status == "connected":
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
        asyncio.run(fetch_hedis_examples())
    except Exception as e:
        examples["HEDIS Expert"].extend([f"⚠️ Failed to load dynamic examples: {e}"])

# Display mode info
mode_text = "🔌 MCP Mode" if (use_mcp and mcp_connection_status == "connected") else "🔧 Standalone Mode"
st.sidebar.markdown(f"**Current Mode:** {mode_text}")

# Example queries sidebar
with st.sidebar.expander(f"💡 Example Queries - {prompt_type}", expanded=True):
    if examples[prompt_type]:
        for i, example in enumerate(examples[prompt_type]):
            display_text = example if len(example) <= 60 else example[:57] + "..."
            if st.button(display_text, key=f"{prompt_type}_{i}_{example[:20]}", use_container_width=True):
                st.session_state.query_input = example
    else:
        st.info("No examples available for this prompt type")

# === QUERY PROCESSING ===

async def process_query_with_mcp(query: str, prompt_type: str) -> str:
    """Process query using MCP server with proper agent wiring"""
    try:
        # Initialize MCP client
        client = MultiServerMCPClient(
            {"DataFlyWheelServer": {"url": server_url, "transport": "sse"}}
        )
        
        model = get_model()
        
        # Get tools and create React agent
        tools = await client.get_tools()
        agent = create_react_agent(model=model, tools=tools)
        
        # Handle prompt selection
        prompt_name = prompt_map[prompt_type]
        
        if prompt_name is None:
            # No context mode - use query directly
            messages = [{"role": "user", "content": query}]
        else:  
            # Get prompt from server
            try:
                prompt_from_server = await client.get_prompt(
                    server_name="DataFlyWheelServer",
                    prompt_name=prompt_name,
                    arguments={"query": query}
                )
                
                # Handle prompt formatting
                if prompt_from_server and len(prompt_from_server) > 0:
                    if hasattr(prompt_from_server[0], 'content'):
                        content = prompt_from_server[0].content
                        if "{query}" in content:
                            content = content.format(query=query)
                        messages = [{"role": "user", "content": content}]
                    else:
                        messages = [{"role": "user", "content": str(prompt_from_server[0])}]
                else:
                    # Fallback if prompt not found
                    messages = [{"role": "user", "content": query}]
            except Exception as prompt_error:
                # Fallback to direct query if prompt fails
                messages = [{"role": "user", "content": query}]
        
        # Invoke agent with proper message format
        response = await agent.ainvoke({"messages": messages})
        
        # Extract result with better error handling
        if isinstance(response, dict):
            # Try different possible keys for the response
            for key in ['messages', 'output', 'result']:
                if key in response:
                    if isinstance(response[key], list) and len(response[key]) > 0:
                        last_message = response[key][-1]
                        if hasattr(last_message, 'content'):
                            return last_message.content
                        else:
                            return str(last_message)
                    elif isinstance(response[key], str):
                        return response[key]
            
            # Fallback: try to get any meaningful content
            response_values = list(response.values())
            if response_values and len(response_values) > 0:
                if isinstance(response_values[0], list) and len(response_values[0]) > 1:
                    if hasattr(response_values[0][1], 'content'):
                        return response_values[0][1].content
                    else:
                        return str(response_values[0][1])
                else:
                    return str(response_values[0])
        
        return str(response) if response else "⚠️ Received empty response from the server."
        
    except Exception as e:
        raise Exception(f"MCP processing failed: {str(e)}")

async def process_query_standalone(query: str, prompt_type: str) -> str:
    """Process query using standalone implementations (no HEDIS - MCP only)"""
    
    query_lower = query.lower()
    
    # HEDIS Expert Logic - Requires MCP server
    if prompt_type == "HEDIS Expert":
        return "🏥 **HEDIS functionality requires MCP server connection.** Please enable MCP mode and ensure the server is running to access HEDIS tools (DFWAnalyst and DFWSearch)."
    
    # Calculator Logic
    elif prompt_type == "Calculator":
        # Extract mathematical expressions
        math_patterns = [
            r'[\d\+\-\*\/\(\)\.\s]+',
            r'calculate\s+(.+)',
            r'what\s+is\s+(.+)',
            r'compute\s+(.+)'
        ]
        
        for pattern in math_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if pattern == r'[\d\+\-\*\/\(\)\.\s]+':
                    # Direct mathematical expression
                    if re.match(r'^[\d\+\-\*\/\(\)\.\s]+

async def process_query_hybrid(query: str, prompt_type: str) -> str:
    """Process query using MCP if available, fallback to standalone"""
    
    # Try MCP first if enabled and connected
    if use_mcp and mcp_connection_status == "connected":
        try:
            return await process_query_with_mcp(query, prompt_type)
        except Exception as e:
            st.warning(f"⚠️ MCP processing failed, falling back to standalone: {str(e)}")
    
    # Fallback to standalone processing
    return await process_query_standalone(query, prompt_type)

# === STREAMLIT UI ===

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if query := st.chat_input("Type your query here...") or st.session_state.get("query_input"):
    
    if "query_input" in st.session_state:
        query = st.session_state.query_input
        del st.session_state.query_input
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Process and display assistant response
    with st.chat_message("assistant"):
        with st.spinner(f"🤔 Processing with {prompt_type} mode..."):
            try:
                response = asyncio.run(process_query_hybrid(query, prompt_type))
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"❌ **Error**: {str(e)}"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar controls
with st.sidebar:
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Connection test
    if st.button("🔍 Test Connections", use_container_width=True):
        test_results = []
        
        # Test MCP connection
        if use_mcp:
            try:
                async def test_mcp():
                    async with sse_client(url=server_url) as sse_connection:
                        async with ClientSession(*sse_connection) as session:
                            await session.initialize()
                            tools = await session.list_tools()
                            tool_count = len(tools.tools) if hasattr(tools, 'tools') else 0
                            return f"✅ MCP connection: OK ({tool_count} tools)"
                
                result = asyncio.run(test_mcp())
                test_results.append(result)
            except Exception as e:
                test_results.append(f"❌ MCP connection: Failed - {str(e)}")
        
        # Test external APIs
        try:
            response = requests.get("https://httpbin.org/get", timeout=5)
            if response.status_code == 200:
                test_results.append("✅ Internet connectivity: OK")
            else:
                test_results.append("❌ Internet connectivity: Failed")
        except:
            test_results.append("❌ Internet connectivity: Failed")
        
        for result in test_results:
            if "✅" in result:
                st.success(result)
            else:
                st.error(result)
    
    st.caption(f"🤖 Mode: {prompt_type}")
    st.caption(f"🔧 Status: {mode_text}")
, query.strip()):
                        return standalone_calculate(query.strip())
                else:
                    # Extract expression from sentence
                    expression = match.group(1).strip()
                    return standalone_calculate(expression)
        
        return f"🧮 **Calculator Ready**\n\nI can help you calculate mathematical expressions. Please provide a mathematical expression like:\n- `3 + 4 * 5`\n- `(10 + 5) / 3`\n- `2.5 * 8`\n\nYour query: {query}"
    
    # Weather Logic
    elif prompt_type == "Weather":
        # Extract coordinates if present
        coord_pattern = r'(-?\d+\.?\d*),?\s*(-?\d+\.?\d*)'
        coord_match = re.search(coord_pattern, query)
        
        if coord_match:
            lat, lon = float(coord_match.group(1)), float(coord_match.group(2))
            return standalone_get_weather(lat, lon)
        
        # Common city coordinates
        city_coords = {
            'richmond': (37.5407, -77.4360),
            'atlanta': (33.7490, -84.3880),
            'new york': (40.7128, -74.0060),
            'denver': (39.7392, -104.9903),
            'miami': (25.7617, -80.1918)
        }
        
        for city, (lat, lon) in city_coords.items():
            if city in query_lower:
                return standalone_get_weather(lat, lon)
        
        return f"🌤️ **Weather Service**\n\nTo get weather information, please provide:\n1. Coordinates: `latitude, longitude`\n2. Or mention a major city like Richmond, Atlanta, New York, Denver, or Miami\n\nExample: 'Weather for Richmond, Virginia (37.5407, -77.4360)'\n\nYour query: {query}"
    
    # Wikipedia Search Logic
    elif prompt_type == "Wikipedia Search":
        return await standalone_wikipedia_search(query, max_results=3)
    
    # Web Search Logic
    elif prompt_type == "Web Search":
        return await standalone_duckduckgo_search(query, max_results=10)
    
    # No Context - Smart routing
    else:
        # Check for calculator patterns
        if re.match(r'^[\d\+\-\*\/\(\)\.\s]+

async def process_query_hybrid(query: str, prompt_type: str) -> str:
    """Process query using MCP if available, fallback to standalone"""
    
    # Try MCP first if enabled and connected
    if use_mcp and mcp_connection_status == "connected":
        try:
            return await process_query_with_mcp(query, prompt_type)
        except Exception as e:
            st.warning(f"⚠️ MCP processing failed, falling back to standalone: {str(e)}")
    
    # Fallback to standalone processing
    return await process_query_standalone(query, prompt_type)

# === STREAMLIT UI ===

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if query := st.chat_input("Type your query here...") or st.session_state.get("query_input"):
    
    if "query_input" in st.session_state:
        query = st.session_state.query_input
        del st.session_state.query_input
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Process and display assistant response
    with st.chat_message("assistant"):
        with st.spinner(f"🤔 Processing with {prompt_type} mode..."):
            try:
                response = asyncio.run(process_query_hybrid(query, prompt_type))
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"❌ **Error**: {str(e)}"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar controls
with st.sidebar:
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Connection test
    if st.button("🔍 Test Connections", use_container_width=True):
        test_results = []
        
        # Test MCP connection
        if use_mcp:
            try:
                async def test_mcp():
                    async with sse_client(url=server_url) as sse_connection:
                        async with ClientSession(*sse_connection) as session:
                            await session.initialize()
                            tools = await session.list_tools()
                            tool_count = len(tools.tools) if hasattr(tools, 'tools') else 0
                            return f"✅ MCP connection: OK ({tool_count} tools)"
                
                result = asyncio.run(test_mcp())
                test_results.append(result)
            except Exception as e:
                test_results.append(f"❌ MCP connection: Failed - {str(e)}")
        
        # Test external APIs
        try:
            response = requests.get("https://httpbin.org/get", timeout=5)
            if response.status_code == 200:
                test_results.append("✅ Internet connectivity: OK")
            else:
                test_results.append("❌ Internet connectivity: Failed")
        except:
            test_results.append("❌ Internet connectivity: Failed")
        
        # Test Snowflake (for HEDIS)
        try:
            conn = get_hedis_connection()
            if conn:
                test_results.append("✅ Snowflake HEDIS: Connected")
            else:
                test_results.append("❌ Snowflake HEDIS: Failed")
        except Exception as e:
            test_results.append(f"❌ Snowflake HEDIS: {str(e)}")
        
        for result in test_results:
            if "✅" in result:
                st.success(result)
            else:
                st.error(result)
    
    st.caption(f"🤖 Mode: {prompt_type}")
    st.caption(f"🔧 Status: {mode_text}")
, query.strip()):
            return standalone_calculate(query.strip())
        
        # Check for weather patterns
        if any(keyword in query_lower for keyword in ["weather", "temperature", "forecast"]):
            return f"🌤️ For weather information, please switch to 'Weather' mode or provide coordinates."
        
        # Check for HEDIS patterns
        if any(keyword in query_lower for keyword in ["hedis", "measure", "bcs", "coa", "eed"]):
            return f"🏥 For HEDIS queries, please enable MCP mode and ensure the server is running."
        
        # Default to search
        if any(keyword in query_lower for keyword in ["what", "who", "when", "where", "how", "define"]):
            return await standalone_wikipedia_search(query, max_results=2)
        else:
            return await standalone_duckduckgo_search(query, max_results=5)

async def process_query_hybrid(query: str, prompt_type: str) -> str:
    """Process query using MCP if available, fallback to standalone"""
    
    # Try MCP first if enabled and connected
    if use_mcp and mcp_connection_status == "connected":
        try:
            return await process_query_with_mcp(query, prompt_type)
        except Exception as e:
            st.warning(f"⚠️ MCP processing failed, falling back to standalone: {str(e)}")
    
    # Fallback to standalone processing
    return await process_query_standalone(query, prompt_type)

# === STREAMLIT UI ===

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if query := st.chat_input("Type your query here...") or st.session_state.get("query_input"):
    
    if "query_input" in st.session_state:
        query = st.session_state.query_input
        del st.session_state.query_input
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Process and display assistant response
    with st.chat_message("assistant"):
        with st.spinner(f"🤔 Processing with {prompt_type} mode..."):
            try:
                response = asyncio.run(process_query_hybrid(query, prompt_type))
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"❌ **Error**: {str(e)}"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar controls
with st.sidebar:
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Connection test
    if st.button("🔍 Test Connections", use_container_width=True):
        test_results = []
        
        # Test MCP connection
        if use_mcp:
            try:
                async def test_mcp():
                    async with sse_client(url=server_url) as sse_connection:
                        async with ClientSession(*sse_connection) as session:
                            await session.initialize()
                            tools = await session.list_tools()
                            tool_count = len(tools.tools) if hasattr(tools, 'tools') else 0
                            return f"✅ MCP connection: OK ({tool_count} tools)"
                
                result = asyncio.run(test_mcp())
                test_results.append(result)
            except Exception as e:
                test_results.append(f"❌ MCP connection: Failed - {str(e)}")
        
        # Test external APIs
        try:
            response = requests.get("https://httpbin.org/get", timeout=5)
            if response.status_code == 200:
                test_results.append("✅ Internet connectivity: OK")
            else:
                test_results.append("❌ Internet connectivity: Failed")
        except:
            test_results.append("❌ Internet connectivity: Failed")
        
        # Test Snowflake (for HEDIS)
        try:
            conn = get_hedis_connection()
            if conn:
                test_results.append("✅ Snowflake HEDIS: Connected")
            else:
                test_results.append("❌ Snowflake HEDIS: Failed")
        except Exception as e:
            test_results.append(f"❌ Snowflake HEDIS: {str(e)}")
        
        for result in test_results:
            if "✅" in result:
                st.success(result)
            else:
                st.error(result)
    
    st.caption(f"🤖 Mode: {prompt_type}")
    st.caption(f"🔧 Status: {mode_text}")
