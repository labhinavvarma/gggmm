import streamlit as st
import asyncio
import json
import yaml
import pkg_resources
import requests
import httpx
from datetime import datetime
import time

from mcp.client.sse import sse_client
from mcp import ClientSession

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dependencies import SnowFlakeConnector
from llmobjectwrapper import ChatSnowflakeCortex
from snowflake.snowpark import Session

# Brave Search Integration
BRAVE_API_KEY = "BSA-FDD7EPTjkdgDqW_znc5uhZledvE"

# Brave Search cache for the client
brave_search_cache = {}
BRAVE_CACHE_DURATION = 180  # 3 minutes in seconds

def is_brave_cache_valid(cache_entry):
    """Check if cached Brave search data is still valid"""
    if not cache_entry:
        return False
    return (time.time() - cache_entry['timestamp']) < BRAVE_CACHE_DURATION

class BraveSearchClient:
    """Brave Search client integrated into Streamlit"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limit_requests = []
        self.max_requests_per_minute = 60
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        now = time.time()
        # Remove requests older than 1 minute
        self.rate_limit_requests = [
            req for req in self.rate_limit_requests 
            if now - req < 60
        ]
        
        if len(self.rate_limit_requests) >= self.max_requests_per_minute:
            wait_time = 60 - (now - self.rate_limit_requests[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        self.rate_limit_requests.append(now)
    
    async def web_search(self, query: str, count: int = 10, offset: int = 0) -> str:
        """Perform web search using Brave Search API"""
        try:
            # Check cache first
            cache_key = f"web_{query}_{count}_{offset}".lower().strip()
            if cache_key in brave_search_cache and is_brave_cache_valid(brave_search_cache[cache_key]):
                return brave_search_cache[cache_key]['data']
            
            await self._check_rate_limit()
            
            # Validate inputs
            if len(query) > 400:
                query = query[:400]
            count = max(1, min(count, 20))
            offset = max(0, min(offset, 9))
            
            headers = {
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip',
                'X-Subscription-Token': self.api_key,
                'User-Agent': 'DataFlyWheel-Streamlit-Client/1.0'
            }
            
            url = "https://api.search.brave.com/res/v1/web/search"
            params = {
                'q': query,
                'count': count,
                'offset': offset,
                'search_lang': 'en',
                'country': 'US',
                'safesearch': 'moderate',
                'freshness': 'pd'  # Past day for freshest results
            }
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, headers=headers, params=params)
                
                if response.status_code == 429:
                    return "⚠️ Brave Search rate limit reached. Please try again in a moment."
                elif response.status_code != 200:
                    raise Exception(f"Brave API error: {response.status_code} - {response.text}")
                
                data = response.json()
                
                # Extract web results
                web_results = data.get('web', {}).get('results', [])
                
                if not web_results:
                    return f"❌ No web results found for: {query}"
                
                # Format results with enhanced information
                results = []
                results.append(f"🔍 **Brave Web Search Results for '{query}' (Fresh Data - {datetime.now().strftime('%B %d, %Y')}):**\n")
                
                for i, result in enumerate(web_results, 1):
                    title = result.get('title', 'No title')
                    description = result.get('description', 'No description')
                    url = result.get('url', 'No URL')
                    published = result.get('published', '')
                    language = result.get('language', 'en')
                    
                    results.append(f"## {i}. {title}")
                    results.append(f"**URL:** {url}")
                    if published:
                        try:
                            # Parse and format publish date
                            pub_date = datetime.fromisoformat(published.replace('Z', '+00:00'))
                            results.append(f"**Published:** {pub_date.strftime('%B %d, %Y at %I:%M %p UTC')}")
                        except:
                            results.append(f"**Published:** {published}")
                    
                    results.append(f"**Language:** {language.upper()}")
                    results.append(f"**Description:** {description}")
                    results.append("")
                
                # Add metadata
                query_data = data.get('query', {})
                if query_data:
                    results.append(f"**Search Metadata:**")
                    results.append(f"- Original Query: {query_data.get('original', query)}")
                    results.append(f"- Results Count: {len(web_results)}")
                    if offset > 0:
                        results.append(f"- Page Offset: {offset}")
                    results.append("")
                
                current_time = datetime.now().strftime('%B %d, %Y at %I:%M %p')
                results.append(f"*Fresh search completed at: {current_time} via Brave Search API*")
                
                formatted_result = "\n".join(results)
                
                # Cache the result
                brave_search_cache[cache_key] = {
                    'data': formatted_result,
                    'timestamp': time.time()
                }
                
                return formatted_result
                
        except Exception as e:
            return f"❌ Brave web search error: {str(e)}"
    
    async def local_search(self, query: str, count: int = 5) -> str:
        """Perform local search using Brave Search API"""
        try:
            # Check cache first
            cache_key = f"local_{query}_{count}".lower().strip()
            if cache_key in brave_search_cache and is_brave_cache_valid(brave_search_cache[cache_key]):
                return brave_search_cache[cache_key]['data']
            
            await self._check_rate_limit()
            
            # Validate inputs
            count = max(1, min(count, 20))
            
            headers = {
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip',
                'X-Subscription-Token': self.api_key,
                'User-Agent': 'DataFlyWheel-Streamlit-Client/1.0'
            }
            
            # Step 1: Get location IDs from web search
            web_url = "https://api.search.brave.com/res/v1/web/search"
            web_params = {
                'q': query,
                'search_lang': 'en',
                'result_filter': 'locations',
                'count': count,
                'country': 'US'
            }
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                web_response = await client.get(web_url, headers=headers, params=web_params)
                
                if web_response.status_code == 429:
                    return "⚠️ Brave Search rate limit reached. Please try again in a moment."
                elif web_response.status_code != 200:
                    raise Exception(f"Brave API error: {web_response.status_code} - {web_response.text}")
                
                web_data = web_response.json()
                location_results = web_data.get('locations', {}).get('results', [])
                
                if not location_results:
                    # Fallback to regular web search
                    return await self.web_search(query, count)
                
                # Extract location IDs
                location_ids = [loc['id'] for loc in location_results if loc.get('id')]
                
                if not location_ids:
                    return await self.web_search(query, count)
                
                # Step 2: Get POI details
                poi_url = "https://api.search.brave.com/res/v1/local/pois"
                poi_params = {'ids': location_ids}
                
                poi_response = await client.get(poi_url, headers=headers, params=poi_params)
                
                if poi_response.status_code != 200:
                    return await self.web_search(query, count)
                
                poi_data = poi_response.json()
                
                # Step 3: Get descriptions
                desc_url = "https://api.search.brave.com/res/v1/local/descriptions"
                desc_params = {'ids': location_ids}
                
                desc_response = await client.get(desc_url, headers=headers, params=desc_params)
                descriptions = {}
                
                if desc_response.status_code == 200:
                    desc_data = desc_response.json()
                    descriptions = desc_data.get('descriptions', {})
                
                # Format results
                results = []
                results.append(f"📍 **Brave Local Search Results for '{query}' ({datetime.now().strftime('%B %d, %Y')}):**\n")
                
                poi_results = poi_data.get('results', [])
                
                for i, poi in enumerate(poi_results, 1):
                    name = poi.get('name', 'Unknown Business')
                    
                    # Format address
                    address_parts = []
                    address = poi.get('address', {})
                    if address.get('streetAddress'):
                        address_parts.append(address['streetAddress'])
                    if address.get('addressLocality'):
                        address_parts.append(address['addressLocality'])
                    if address.get('addressRegion'):
                        address_parts.append(address['addressRegion'])
                    if address.get('postalCode'):
                        address_parts.append(address['postalCode'])
                    
                    formatted_address = ', '.join(address_parts) if address_parts else 'Address not available'
                    
                    results.append(f"## {i}. {name}")
                    results.append(f"**Address:** {formatted_address}")
                    
                    # Phone
                    phone = poi.get('phone', 'Not available')
                    results.append(f"**Phone:** {phone}")
                    
                    # Rating
                    rating = poi.get('rating', {})
                    rating_value = rating.get('ratingValue', 'N/A')
                    rating_count = rating.get('ratingCount', 0)
                    results.append(f"**Rating:** {rating_value} ({rating_count} reviews)")
                    
                    # Price range
                    price_range = poi.get('priceRange', 'Not specified')
                    results.append(f"**Price Range:** {price_range}")
                    
                    # Opening hours
                    hours = poi.get('openingHours', [])
                    if hours:
                        results.append(f"**Hours:** {', '.join(hours)}")
                    else:
                        results.append(f"**Hours:** Not specified")
                    
                    # Description
                    poi_id = poi.get('id', '')
                    description = descriptions.get(poi_id, 'No description available')
                    results.append(f"**Description:** {description}")
                    
                    # Coordinates (if available)
                    coordinates = poi.get('coordinates', {})
                    if coordinates.get('latitude') and coordinates.get('longitude'):
                        lat = coordinates['latitude']
                        lng = coordinates['longitude']
                        results.append(f"**Location:** {lat:.6f}, {lng:.6f}")
                    
                    results.append("")
                
                current_time = datetime.now().strftime('%B %d, %Y at %I:%M %p')
                results.append(f"*Local search completed at: {current_time} via Brave Local Search API*")
                
                formatted_result = "\n".join(results)
                
                # Cache the result
                brave_search_cache[cache_key] = {
                    'data': formatted_result,
                    'timestamp': time.time()
                }
                
                return formatted_result
                
        except Exception as e:
            # Fallback to web search on error
            return await self.web_search(query, count)

# Initialize Brave Search client
brave_client = BraveSearchClient(BRAVE_API_KEY)

# Page config
st.set_page_config(page_title="Enhanced MCP Client with Brave Search", page_icon="🚀")
st.title("🚀 Enhanced MCP Client - DataFlyWheel Edition with Brave Search")
st.markdown("*Integrated Brave Search API - Fresh, Unbiased Results*")

# Updated server URL to match your configuration
server_url = st.sidebar.text_input("MCP Server URL", "http://localhost:8081/sse")
show_server_info = st.sidebar.checkbox("🛡 Show MCP Server Info", value=False)

# Enhanced connection status check
@st.cache_data(ttl=15)
def check_server_connection(url):
    try:
        # Check both the SSE endpoint and the main server
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

if server_status["connected"] and server_status.get("tools_available") != "unknown":
    st.sidebar.markdown(f"**Tools Available:** {server_status['tools_available']}")

# Brave Search Status
st.sidebar.markdown("**Brave Search:** 🟢 Integrated")
if brave_search_cache:
    st.sidebar.markdown(f"**Cached Searches:** {len(brave_search_cache)}")

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
            "weather_cache": {}
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
                cache_response = requests.get(f"{base_url}/api/v1/weather_cache", timeout=5)
                if cache_response.status_code == 200:
                    result["weather_cache"] = cache_response.json()
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
                   
                    # --- Enhanced Tools (excluding Brave Search which is now client-side) ---
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
                                    
                                    # Add schema info if available
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

                    # --- YAML Resources ---
                    try:
                        yaml_content = await session.read_resource("schematiclayer://cortex_analyst/schematic_models/hedis_stage_full/list")
                        if hasattr(yaml_content, 'contents'):
                            for item in yaml_content.contents:
                                if hasattr(item, 'text'):
                                    try:
                                        parsed = yaml.safe_load(item.text)
                                        result["yaml"].append(yaml.dump(parsed, sort_keys=False))
                                    except:
                                        result["yaml"].append(item.text)
                    except Exception as e:
                        result["yaml"].append(f"YAML error: {e}")

                    # --- Search Objects ---
                    try:
                        content = await session.read_resource("search://cortex_search/search_obj/list")
                        if hasattr(content, 'contents'):
                            for item in content.contents:
                                if hasattr(item, 'text'):
                                    try:
                                        objs = json.loads(item.text)
                                        result["search"].extend(objs)
                                    except:
                                        result["search"].append(item.text)
                    except Exception as e:
                        result["search"].append(f"Search error: {e}")

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

    # Brave Search cache status
    if brave_search_cache:
        with st.sidebar.expander("🔍 Brave Search Cache", expanded=False):
            st.write(f"**Cached Searches:** {len(brave_search_cache)}")
            
            for search_key, cache_data in brave_search_cache.items():
                valid_indicator = "✅" if is_brave_cache_valid(cache_data) else "❌"
                age_seconds = time.time() - cache_data['timestamp']
                display_key = search_key[:30] + "..." if len(search_key) > 30 else search_key
                st.write(f"{valid_indicator} **{display_key}**: {age_seconds:.0f}s old")

    # Display Resources with better organization
    with st.sidebar.expander("📦 Resources", expanded=False):
        for r in mcp_data["resources"]:
            if isinstance(r, dict) and "error" not in r:
                if "cortex_search/search_obj/list" in r["name"]:
                    display_name = "🔍 Cortex Search Service"
                elif "schematic_models" in r["name"]:
                    display_name = "📋 HEDIS Schematic Models"
                elif "frequent_questions" in r["name"]:
                    display_name = "❓ Frequent Questions"
                elif "prompts" in r["name"]:
                    display_name = "📝 Prompt Templates"
                else:
                    display_name = r["name"]
                st.markdown(f"**{display_name}**")
                if r.get("description") and r["description"] != "N/A":
                    st.caption(r["description"])
            else:
                st.error(str(r))

    # --- Enhanced Tools Section with Updated Categories (including Brave Search) ---
    with st.sidebar.expander("🛠 Available Tools", expanded=False):
        # Updated tool categories to include Brave Search (client-side)
        tool_categories = {
            "🏥 HEDIS & Analytics": ["DFWAnalyst", "DFWSearch", "calculator"],
            "🔍 Search & Information": ["brave_web_search", "brave_local_search"],  # Client-side tools
            "🌤️ Weather & Location": ["get_weather"],
            "🔧 System & Testing": ["test_tool", "diagnostic"]
        }
        
        available_tools = {t["name"]: t for t in mcp_data["tools"] if isinstance(t, dict) and "error" not in t}
        
        # Add client-side Brave Search tools
        available_tools["brave_web_search"] = {
            "name": "brave_web_search", 
            "description": "Search the web using Brave Search API for fresh, unbiased results"
        }
        available_tools["brave_local_search"] = {
            "name": "brave_local_search", 
            "description": "Search for local businesses and places using Brave Search API"
        }
        
        for category, expected_tools in tool_categories.items():
            st.markdown(f"**{category}:**")
            category_found = False
            for tool_name in expected_tools:
                if tool_name in available_tools:
                    tool_info = available_tools[tool_name]
                    if tool_name.startswith("brave_"):
                        st.markdown(f"  • **{tool_name}** (Client-side)")
                    else:
                        st.markdown(f"  • **{tool_name}**")
                    if tool_info.get('description'):
                        st.caption(f"    {tool_info['description']}")
                    if tool_info.get('parameters'):
                        st.caption(f"    Parameters: {', '.join(tool_info['parameters'])}")
                    category_found = True
            
            if not category_found:
                st.caption("    No tools found in this category")
        
        # Show any uncategorized tools
        all_categorized = [tool for tools in tool_categories.values() for tool in tools]
        uncategorized = [name for name in available_tools.keys() if name not in all_categorized]
        
        if uncategorized:
            st.markdown("**🔧 Other Tools:**")
            for tool_name in uncategorized:
                tool_info = available_tools[tool_name]
                st.markdown(f"  • **{tool_name}**")
                if tool_info.get('description'):
                    st.caption(f"    {tool_info['description']}")

    # Display Prompts with enhanced formatting
    with st.sidebar.expander("🧐 Available Prompts", expanded=False):
        # Updated prompt display names to include Brave Search prompts
        prompt_display_names = {
            "hedis-prompt": "🏥 HEDIS Expert",
            "calculator-prompt": "🧮 Calculator Expert",
            "weather-prompt": "🌤️ Weather Expert", 
            "brave-web-search-prompt": "🔍 Web Search Expert",
            "brave-local-search-prompt": "📍 Local Search Expert",
            "test-tool-prompt": "🔧 Test Tool",
            "diagnostic-prompt": "🔧 Diagnostic Tool"
        }
        
        # Add client-side Brave Search prompts info
        brave_prompts = [
            {
                "name": "brave-web-search-prompt",
                "description": "Web search expert using Brave Search API",
                "args": [{"name": "query", "required": True, "description": "Search query"}]
            },
            {
                "name": "brave-local-search-prompt", 
                "description": "Local business search expert using Brave Search API",
                "args": [{"name": "query", "required": True, "description": "Local search query"}]
            }
        ]
        
        all_prompts = mcp_data["prompts"] + brave_prompts
        
        for p in all_prompts:
            if isinstance(p, dict) and "error" not in p:
                display_name = prompt_display_names.get(p['name'], p['name'])
                if p['name'].startswith('brave-'):
                    display_name += " (Client-side)"
                st.markdown(f"**{display_name}**")
                if p.get('description'):
                    st.caption(f"Description: {p['description']}")
                if p.get('args'):
                    args_text = ", ".join([f"{arg['name']}{'*' if arg.get('required') else ''}" 
                                         for arg in p['args']])
                    if args_text:
                        st.caption(f"Arguments: {args_text}")
            else:
                st.error(str(p))

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
                    mcp_server_url=server_url  # Add MCP server URL to the model
                )
            else:
                # Create model without Snowflake session (will use MCP tools only)
                return ChatSnowflakeCortex(
                    model="claude-4-sonnet",
                    cortex_function="complete",
                    mcp_server_url=server_url
                )
        except Exception as e:
            st.error(f"❌ Failed to initialize model: {e}")
            return None
    
    # Enhanced prompt type selection with updated options including Brave Search
    prompt_type = st.sidebar.radio(
        "🎯 Select Expert Mode", 
        ["Calculator", "HEDIS Expert", "Weather", "Web Search", "Local Search", "General AI"],
        help="Choose the type of expert assistance you need"
    )
    
    # Updated prompt mapping to include Brave Search (client-side)
    prompt_map = {
        "Calculator": "calculator-prompt",
        "HEDIS Expert": "hedis-prompt",
        "Weather": "weather-prompt",
        "Web Search": "brave-web-search",  # Client-side
        "Local Search": "brave-local-search",  # Client-side
        "General AI": None
    }

    # Enhanced examples with updated search examples
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
        "Web Search": [
            "latest AI developments 2025",
            "current renewable energy trends", 
            "recent space exploration missions",
            "today's technology news",
            "latest electric vehicle updates"
        ],
        "Local Search": [
            "pizza restaurants near Central Park",
            "gas stations in San Francisco",
            "coffee shops downtown Portland",
            "restaurants near Times Square",
            "best sushi in Manhattan"
        ],
        "General AI": [
            "Explain quantum computing in simple terms",
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
            "What's the difference between AI and ML?"
        ]
    }

    # Load HEDIS examples dynamically from MCP server
    if prompt_type == "HEDIS Expert":
        try:
            async def fetch_hedis_examples():
                try:
                    async with sse_client(url=server_url) as sse_connection:
                        async with ClientSession(*sse_connection) as session:
                            await session.initialize()
                            content = await session.read_resource("genaiplatform://hedis/frequent_questions/Initialization")
                            if hasattr(content, "contents"):
                                for item in content.contents:
                                    if hasattr(item, "text"):
                                        loaded_examples = json.loads(item.text)
                                        examples["HEDIS Expert"].extend(loaded_examples[:10])  # Limit to 10 examples
                except Exception as e:
                    print(f"Failed to load HEDIS examples: {e}")
   
            asyncio.run(fetch_hedis_examples())
        except Exception as e:
            pass
            
        # Fallback examples if dynamic loading failed
        if not examples["HEDIS Expert"]:
            examples["HEDIS Expert"] = [
                "What are the codes in BCS Value Set?",
                "Explain the BCS (Breast Cancer Screening) measure",
                "What is the age criteria for CBP measure?",
                "Describe the COA measure requirements",
                "What LOB is COA measure scoped under?"
            ]

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Enhanced example queries with better organization
    with st.sidebar.expander(f"💡 Example Queries - {prompt_type}", expanded=True):
        if examples[prompt_type]:
            for i, example in enumerate(examples[prompt_type]):
                display_text = example if len(example) <= 70 else example[:67] + "..."
                if st.button(display_text, key=f"{prompt_type}_{i}_{hash(example)}", use_container_width=True):
                    st.session_state.query_input = example
        else:
            st.info("Loading examples...")

    # Add helpful tips based on selected mode
    if prompt_type == "Weather":
        with st.sidebar.expander("🌍 Weather Tips", expanded=False):
            st.info("""
            **Enhanced Weather Service:**
            • Covers worldwide locations
            • Uses cached data (5-min refresh)
            • Multiple sources (NWS + Open-Meteo)
            • 3-day forecast included
            • Current conditions and detailed forecasts
            
            **Examples:**
            • "Weather in New York"
            • "Current conditions in London"
            • "Tokyo weather forecast"
            """)
    
    elif prompt_type == "Web Search":
        with st.sidebar.expander("🔍 Web Search Tips", expanded=False):
            st.info("""
            **Brave Web Search (Client-side):**
            • Fresh, unbiased search results
            • No tracking or data collection
            • Prioritizes recent content
            • Fast and reliable API
            • Independent search engine
            
            **Best for:**
            • Latest news and developments
            • Current research and findings
            • Real-time information
            • Recent events and trends
            """)
    
    elif prompt_type == "Local Search":
        with st.sidebar.expander("📍 Local Search Tips", expanded=False):
            st.info("""
            **Brave Local Search (Client-side):**
            • Find local businesses and places
            • Detailed business information
            • Ratings and reviews
            • Contact details and hours
            • Maps and directions
            
            **Best for:**
            • Restaurants and dining
            • Services and shopping
            • Entertainment venues
            • Business directories
            """)

    # Chat input handling with enhanced processing including Brave Search
    if query := st.chat_input("Type your query here...") or "query_input" in st.session_state:

        if "query_input" in st.session_state:
            query = st.session_state.query_input
            del st.session_state.query_input

        with st.chat_message("user"):
            st.markdown(query, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "user", "content": query})

        async def process_enhanced_query(query_text):
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.text("🤔 Processing your request...")
                
                try:
                    # Handle Brave Search requests directly in client
                    if prompt_type in ["Web Search", "Local Search"]:
                        message_placeholder.text(f"🔍 Performing {prompt_type.lower()} using Brave Search API...")
                        
                        if prompt_type == "Web Search":
                            result = await brave_client.web_search(query_text)
                        else:  # Local Search
                            result = await brave_client.local_search(query_text)
                        
                        message_placeholder.markdown(result)
                        st.session_state.messages.append({"role": "assistant", "content": result})
                        return
                    
                    # For other requests, use MCP server
                    message_placeholder.text("🔌 Connecting to enhanced MCP server...")
                    
                    if not server_status["connected"]:
                        raise Exception("MCP server is not accessible. Please check the server URL and ensure it's running.")
                    
                    client = MultiServerMCPClient(
                        {"DataFlyWheelServer": {"url": server_url, "transport": "sse"}}
                    )

                    model = get_model()
                    if not model:
                        raise Exception("Failed to initialize the AI model. Please check Snowflake connection.")
                    
                    # Get tools and create agent
                    message_placeholder.text("🛠️ Loading enhanced tools from server...")
                    tools = await client.get_tools()
                    
                    if not tools:
                        raise Exception("No tools available from the MCP server.")
                    
                    message_placeholder.text(f"🤖 Creating AI agent with {len(tools)} tools...")
                    agent = create_react_agent(model=model, tools=tools)
                    
                    # Handle prompt selection with better formatting
                    prompt_name = prompt_map[prompt_type]
                    
                    if prompt_name is None:
                        # General AI mode - use query directly
                        message_placeholder.text("💭 Processing in general AI mode...")
                        messages = [{"role": "user", "content": query_text}]
                    else:  
                        # Get prompt from server
                        message_placeholder.text(f"📝 Loading {prompt_type} expert prompt...")
                        try:
                            prompt_from_server = await client.get_prompt(
                                server_name="DataFlyWheelServer",
                                prompt_name=prompt_name,
                                arguments={"query": query_text}
                            )
                            
                            # Enhanced prompt handling
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
                                # Fallback if prompt not found
                                st.warning(f"⚠️ {prompt_type} prompt not found on server. Using direct mode.")
                                messages = [{"role": "user", "content": query_text}]
                                
                        except Exception as prompt_error:
                            st.warning(f"⚠️ Could not load {prompt_type} prompt: {prompt_error}. Using direct mode.")
                            messages = [{"role": "user", "content": query_text}]

                    message_placeholder.text("🧠 Generating intelligent response...")
                    
                    # Invoke agent with proper message format and timeout
                    try:
                        response = await asyncio.wait_for(
                            agent.ainvoke({"messages": messages}), 
                            timeout=120.0  # 2 minute timeout
                        )
                    except asyncio.TimeoutError:
                        raise Exception("Request timed out. The server may be overloaded or the query is too complex.")
                    
                    # Enhanced result extraction with multiple strategies
                    result = None
                    
                    if isinstance(response, dict):
                        # Strategy 1: Check for 'messages' key with AI message
                        if 'messages' in response:
                            messages_list = response['messages']
                            if isinstance(messages_list, list):
                                # Look for the last assistant message
                                for msg in reversed(messages_list):
                                    if hasattr(msg, 'content') and hasattr(msg, 'type'):
                                        if getattr(msg, 'type', None) == 'ai' or not result:
                                            result = msg.content
                                            break
                                    elif hasattr(msg, 'content'):
                                        result = msg.content
                        
                        # Strategy 2: Look for any meaningful content
                        if result is None:
                            for key, value in response.items():
                                if isinstance(value, str) and len(value) > 20:
                                    result = value
                                    break
                                elif isinstance(value, list) and len(value) > 0:
                                    for item in value:
                                        if hasattr(item, 'content') and len(str(item.content)) > 20:
                                            result = item.content
                                            break
                                        elif isinstance(item, str) and len(item) > 20:
                                            result = item
                                            break
                                    if result:
                                        break
                    
                    # Fallback to string representation
                    if result is None or (isinstance(result, str) and len(result.strip()) < 10):
                        result = str(response)
                    
                    # Clean up and validate the result
                    if isinstance(result, str):
                        result = result.strip()
                        if result.startswith('"') and result.endswith('"'):
                            result = result[1:-1]
                        
                        # Check for empty or very short results
                        if len(result) < 10:
                            result = f"⚠️ Received a very short response: '{result}'. Please try rephrasing your query."
                    
                    # Ensure we have a meaningful response
                    if not result or result.strip() == "":
                        result = "⚠️ Received empty response from the server. This might be due to a processing error. Please try again with a different query."
                    
                    # Add timestamp and mode info
                    current_time = datetime.now().strftime('%H:%M:%S')
                    if prompt_type in ["Web Search", "Local Search"]:
                        result += f"\n\n*Response generated at {current_time} using {prompt_type} mode via Brave Search API*"
                    else:
                        result += f"\n\n*Response generated at {current_time} using {prompt_type} mode*"
                    
                    # Display result
                    message_placeholder.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    
                except Exception as e:
                    error_message = f"❌ **Error Processing Request**: {str(e)}\n\n"
                    error_message += f"**Troubleshooting Steps:**\n"
                    error_message += f"1. **Server Status**: {status_indicator}\n"
                    error_message += f"2. **Server URL**: {server_url}\n"
                    error_message += f"3. **Selected Mode**: {prompt_type}\n"
                    error_message += f"4. **Tools Available**: {server_status.get('tools_available', 'Unknown')}\n"
                    if prompt_type in ["Web Search", "Local Search"]:
                        error_message += f"5. **Brave Search**: Integrated in client\n"
                    error_message += "\n"
                    
                    if not server_status["connected"] and prompt_type not in ["Web Search", "Local Search"]:
                        error_message += "**🔧 Server Connection Issues:**\n"
                        error_message += "- Verify the MCP server is running\n"
                        error_message += "- Check if the URL is correct\n"
                        error_message += "- Ensure no firewall is blocking the connection\n"
                    else:
                        error_message += "**🔧 Processing Issues:**\n"
                        error_message += "- Try a simpler query\n"
                        error_message += "- Switch to 'General AI' mode\n"
                        if prompt_type in ["Web Search", "Local Search"]:
                            error_message += "- Check your internet connection for Brave Search\n"
                        else:
                            error_message += "- Check server logs for detailed error information\n"
                    
                    error_message += f"\n*Error occurred at {datetime.now().strftime('%H:%M:%S')}*"
                    
                    message_placeholder.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

        if query:
            asyncio.run(process_enhanced_query(query))

    # Enhanced sidebar controls
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔧 Enhanced Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        with col2:
            if st.button("🔄 Refresh Status", use_container_width=True):
                st.cache_data.clear()
                brave_search_cache.clear()
                st.rerun()

        if st.button("🔍 Test Enhanced Connection", use_container_width=True):
            try:
                async def test_enhanced_connection():
                    # Test MCP connection
                    async with sse_client(url=server_url) as sse_connection:
                        async with ClientSession(*sse_connection) as session:
                            await session.initialize()
                            
                            # Count available resources and tools
                            tools = await session.list_tools()
                            tool_count = len(tools.tools) if hasattr(tools, 'tools') else 0
                            
                            resources = await session.list_resources()
                            resource_count = len(resources.resources) if hasattr(resources, 'resources') else 0
                            
                            prompts = await session.list_prompts()
                            prompt_count = len(prompts.prompts) if hasattr(prompts, 'prompts') else 0
                            
                            # Test Brave Search
                            brave_test = await brave_client.web_search("test query", 1)
                            brave_status = "✅ Working" if "Brave Web Search Results" in brave_test else "❌ Failed"
                            
                            return f"""✅ **Enhanced Connection Test Successful!**
                            
📊 **Server Statistics:**
- 🛠️ MCP Tools Available: {tool_count}
- 📦 Resources: {resource_count}  
- 🧐 Prompts: {prompt_count}

🔍 **Brave Search Integration:**
- Status: {brave_status}
- Cache Entries: {len(brave_search_cache)}

🚀 **Enhanced Features:**
- ✅ Fresh data retrieval with cache-busting
- ✅ Weather caching system active
- ✅ Brave Search integrated (client-side)
- ✅ HEDIS analytics tools ready

🌐 **Connection Quality:** Excellent"""

                result = asyncio.run(test_enhanced_connection())
                st.success(result)
                
                # Additional HTTP endpoint tests
                base_url = server_url.replace('/sse', '')
                
                # Test tool call endpoint
                try:
                    test_response = requests.post(
                        f"{base_url}/api/v1/tool_call",
                        json={"tool_name": "test_tool", "arguments": {"message": "connection test"}},
                        timeout=10
                    )
                    if test_response.status_code == 200:
                        st.info("🔧 Direct tool call endpoint: ✅ Working")
                    else:
                        st.warning(f"🔧 Direct tool call endpoint: ❌ HTTP {test_response.status_code}")
                except Exception as e:
                    st.warning(f"🔧 Direct tool call endpoint: ❌ {str(e)}")
                
            except Exception as e:
                st.error(f"❌ **Enhanced Connection Test Failed**: {e}")
                
                # Provide specific troubleshooting
                base_url = server_url.replace('/sse', '')
                try:
                    health_check = requests.get(f"{base_url}/health", timeout=5)
                    if health_check.status_code == 200:
                        st.info("✅ HTTP server is responding, but MCP connection failed")
                        st.info("🔧 Try restarting the MCP server or check server logs")
                    else:
                        st.error(f"❌ HTTP server error: {health_check.status_code}")
                except:
                    st.error("❌ Server is completely unreachable")

        # Server integration test
        if st.button("🧪 Test Integration", use_container_width=True):
            try:
                base_url = server_url.replace('/sse', '')
                test_response = requests.post(f"{base_url}/test_integration", timeout=30)
                
                if test_response.status_code == 200:
                    test_data = test_response.json()
                    st.success(f"🧪 Integration Test: {test_data.get('success_rate', 'Unknown')} passed")
                    
                    # Show details in expandable section
                    with st.expander("📊 Test Details"):
                        for result in test_data.get('results', []):
                            status_icon = "✅" if result.get('success') else "❌"
                            st.write(f"{status_icon} **{result.get('test')}**: {result.get('result', 'No result')[:100]}...")
                else:
                    st.error(f"❌ Integration test failed: HTTP {test_response.status_code}")
                    
            except Exception as e:
                st.error(f"❌ Integration test error: {e}")

        # Enhanced status information
        st.markdown("---")
        st.markdown("### 📊 System Status")
        
        # Server info
        st.caption(f"🌐 **Server**: {server_url}")
        st.caption(f"🤖 **Mode**: {prompt_type}")
        st.caption(f"📡 **Status**: {status_indicator}")
        
        if server_status.get("details") and isinstance(server_status["details"], dict):
            if "timestamp" in server_status["details"]:
                st.caption(f"⏰ **Last Check**: {server_status['details']['timestamp'][:19]}")
        
        # Brave Search info
        st.caption(f"🔍 **Brave Search**: ✅ Integrated")
        st.caption(f"🔑 **API Key**: {BRAVE_API_KEY[:10]}...")
        
        # Model info
        try:
            model = get_model()
            if model:
                st.caption(f"🧠 **Model**: {getattr(model, 'model', 'Unknown')}")
                if hasattr(model, 'session') and model.session:
                    st.caption("❄️ **Snowflake**: ✅ Connected")
                else:
                    st.caption("❄️ **Snowflake**: ⚠️ Not Connected")
        except:
            st.caption("🧠 **Model**:
