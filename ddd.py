# mcpserver.py - Fixed version
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import httpx
import re

from fastmcp import FastMCP
from mcp.server.models import InitializationOptions
from mcp.types import ServerCapabilities, ToolsCapability, PromptsCapability

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastmcp-server")

# Create FastMCP instance
mcp = FastMCP("Brave Search MCP Server")

# Global cache and configuration
CACHE_DURATION = 1800  # 30 minutes
search_cache: Dict[str, Dict] = {}
weather_cache: Dict[str, Dict] = {}
client_api_keys: Dict[str, str] = {}

def is_cache_valid(cache_entry: Dict) -> bool:
    """Check if cache entry is still valid"""
    return time.time() - cache_entry['timestamp'] < CACHE_DURATION

def get_client_api_key(client_id: str = "default") -> Optional[str]:
    """Get API key for specific client"""
    return client_api_keys.get(client_id)

def set_client_api_key(api_key: str, client_id: str = "default"):
    """Set API key for specific client"""
    client_api_keys[client_id] = api_key
    logger.info(f"✅ API key configured for client: {client_id}")

# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

@mcp.tool()
def configure_brave_key(api_key: str, client_id: str = "default") -> str:
    """
    Configure Brave Search API key for this session
    
    Args:
        api_key: Brave Search API key
        client_id: Client identifier (optional, defaults to "default")
    
    Returns:
        Confirmation message
    """
    if not api_key:
        return "❌ Error: API key is required"
    
    set_client_api_key(api_key, client_id)
    return f"✅ Brave API key configured successfully for client: {client_id}"

@mcp.tool()
async def brave_web_search(
    query: str, 
    count: int = 10, 
    offset: int = 0, 
    client_id: str = "default"
) -> str:
    """
    Search the web using Brave Search API for fresh, unbiased results
    
    Args:
        query: Search query string
        count: Number of results to return (max 20)
        offset: Starting position for results  
        client_id: Client identifier
    
    Returns:
        Formatted search results with titles, URLs, and descriptions
    """
    try:
        api_key = get_client_api_key(client_id)
        if not api_key:
            return "❌ **Error:** Brave API key not configured. Please configure API key first using configure_brave_key tool."
        
        if not query or not query.strip():
            return "❌ **Error:** Search query cannot be empty."
        
        # Check cache
        cache_key = f"web_{client_id}_{query.lower()}_{count}_{offset}"
        if cache_key in search_cache and is_cache_valid(search_cache[cache_key]):
            cached_result = search_cache[cache_key]['formatted_result']
            return f"{cached_result}\n\n*⚡ Result from cache (cached {int((time.time() - search_cache[cache_key]['timestamp']) / 60)} minutes ago)*"
        
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key
        }
        
        params = {
            "q": query.strip(),
            "count": min(max(1, count), 20),
            "offset": max(0, offset),
            "safesearch": "moderate",
            "freshness": "pd",  # Past day for fresh results
            "text_decorations": False,
            "spellcheck": True
        }
        
        logger.info(f"🔍 Brave web search: '{query}' (count={count}, client={client_id})")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Format results
                result = f"🔍 **Brave Web Search Results**\n\n"
                result += f"**Query:** {query}\n"
                
                web_results = data.get('web', {}).get('results', [])
                news_results = data.get('news', {}).get('results', [])
                
                result += f"**Found:** {len(web_results)} web results"
                if news_results:
                    result += f", {len(news_results)} news articles"
                result += "\n\n"
                
                # Web Results
                if web_results:
                    result += "## 🌐 Web Results\n\n"
                    for i, item in enumerate(web_results[:count], 1):
                        title = item.get('title', 'No title')
                        url = item.get('url', '')
                        description = item.get('description', 'No description available')
                        
                        # Clean description
                        description = re.sub(r'<[^>]+>', '', description)
                        description = description.strip()
                        if len(description) > 250:
                            description = description[:247] + "..."
                        
                        result += f"**{i}. {title}**\n"
                        result += f"🔗 {url}\n"
                        result += f"📝 {description}\n\n"
                else:
                    result += "No web results found for this query.\n\n"
                
                # News Results
                if news_results:
                    result += "## 📰 Related News\n\n"
                    for i, item in enumerate(news_results[:3], 1):
                        title = item.get('title', 'No title')
                        url = item.get('url', '')
                        age = item.get('age', '')
                        
                        result += f"**{i}. {title}**"
                        if age:
                            result += f" *({age})*"
                        result += f"\n🔗 {url}\n\n"
                
                # Videos (if available)
                video_results = data.get('videos', {}).get('results', [])
                if video_results:
                    result += "## 🎥 Related Videos\n\n"
                    for i, item in enumerate(video_results[:2], 1):
                        title = item.get('title', 'No title')
                        url = item.get('url', '')
                        duration = item.get('duration', '')
                        
                        result += f"**{i}. {title}**"
                        if duration:
                            result += f" *({duration})*"
                        result += f"\n🔗 {url}\n\n"
                
                result += f"---\n*Search performed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
                result += f"*Powered by Brave Search API*"
                
                # Cache result
                search_cache[cache_key] = {
                    'timestamp': time.time(),
                    'formatted_result': result
                }
                
                logger.info(f"✅ Brave web search completed: {len(web_results)} results")
                return result
                
            elif response.status_code == 401:
                return "❌ **Authentication Error:** Invalid Brave Search API key. Please check your API key configuration."
            elif response.status_code == 429:
                return "❌ **Rate Limit Error:** Too many requests. Please try again later."
            elif response.status_code == 400:
                return f"❌ **Bad Request:** Invalid search parameters. Please check your query: '{query}'"
            else:
                error_text = response.text[:200] if response.text else "Unknown error"
                return f"❌ **API Error:** Request failed with status {response.status_code}: {error_text}"
                
    except httpx.TimeoutException:
        return "❌ **Timeout Error:** Request timed out. Please try again."
    except httpx.ConnectError:
        return "❌ **Connection Error:** Could not connect to Brave Search API. Please check your internet connection."
    except Exception as e:
        logger.error(f"❌ Brave web search error: {e}")
        return f"❌ **Unexpected Error:** {str(e)}"

@mcp.tool()
async def brave_local_search(
    query: str, 
    count: int = 5, 
    client_id: str = "default"
) -> str:
    """
    Search for local businesses and places using Brave Search API
    
    Args:
        query: Local search query (e.g., 'pizza in New York')
        count: Number of results to return (max 20)
        client_id: Client identifier
    
    Returns:
        Formatted local search results with business details
    """
    try:
        api_key = get_client_api_key(client_id)
        if not api_key:
            return "❌ **Error:** Brave API key not configured. Please configure API key first using configure_brave_key tool."
        
        if not query or not query.strip():
            return "❌ **Error:** Search query cannot be empty."
        
        # Check cache
        cache_key = f"local_{client_id}_{query.lower()}_{count}"
        if cache_key in search_cache and is_cache_valid(search_cache[cache_key]):
            cached_result = search_cache[cache_key]['formatted_result']
            return f"{cached_result}\n\n*⚡ Result from cache (cached {int((time.time() - search_cache[cache_key]['timestamp']) / 60)} minutes ago)*"
        
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key
        }
        
        params = {
            "q": query.strip(),
            "count": min(max(1, count), 20),
            "safesearch": "moderate",
            "search_lang": "en",
            "country": "US",
            "units": "metric"
        }
        
        logger.info(f"📍 Brave local search: '{query}' (count={count}, client={client_id})")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/local/search",
                headers=headers,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Format results
                result = f"📍 **Brave Local Search Results**\n\n"
                result += f"**Query:** {query}\n"
                
                local_results = data.get('local', {}).get('results', [])
                result += f"**Found:** {len(local_results)} local businesses\n\n"
                
                if local_results:
                    for i, item in enumerate(local_results[:count], 1):
                        name = item.get('title', 'No name')
                        address = item.get('address', 'No address available')
                        phone = item.get('phone', '')
                        rating = item.get('rating', '')
                        price_range = item.get('price_range', '')
                        website = item.get('url', '')
                        hours = item.get('hours', {})
                        
                        result += f"**{i}. {name}**\n"
                        result += f"📍 **Address:** {address}\n"
                        
                        if phone:
                            result += f"📞 **Phone:** {phone}\n"
                        
                        if rating:
                            try:
                                rating_val = float(rating)
                                stars = "⭐" * int(rating_val)
                                result += f"⭐ **Rating:** {rating} {stars}\n"
                            except:
                                result += f"⭐ **Rating:** {rating}\n"
                        
                        if price_range:
                            result += f"💰 **Price Range:** {price_range}\n"
                        
                        if website:
                            result += f"🌐 **Website:** {website}\n"
                        
                        # Hours (if available)
                        if hours and isinstance(hours, dict):
                            today = datetime.now().strftime('%A').lower()
                            if today in hours:
                                result += f"🕒 **Today's Hours:** {hours[today]}\n"
                        
                        result += "\n"
                else:
                    result += "❌ **No local results found for this query.**\n\n"
                    result += "💡 **Tips for better local search:**\n"
                    result += "• Include a specific location (e.g., 'pizza in Manhattan')\n"
                    result += "• Be specific about the type of business you're looking for\n"
                    result += "• Try alternative keywords or business categories\n"
                    result += "• Check spelling and try broader terms\n\n"
                
                result += f"---\n*Search performed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
                result += f"*Powered by Brave Local Search API*"
                
                # Cache result
                search_cache[cache_key] = {
                    'timestamp': time.time(),
                    'formatted_result': result
                }
                
                logger.info(f"✅ Brave local search completed: {len(local_results)} results")
                return result
                
            elif response.status_code == 401:
                return "❌ **Authentication Error:** Invalid Brave Search API key."
            elif response.status_code == 429:
                return "❌ **Rate Limit Error:** Too many requests. Please try again later."
            elif response.status_code == 400:
                return f"❌ **Bad Request:** Invalid search parameters. Please check your query: '{query}'"
            else:
                error_text = response.text[:200] if response.text else "Unknown error"
                return f"❌ **API Error:** Request failed with status {response.status_code}: {error_text}"
                
    except httpx.TimeoutException:
        return "❌ **Timeout Error:** Request timed out. Please try again."
    except httpx.ConnectError:
        return "❌ **Connection Error:** Could not connect to Brave Local Search API."
    except Exception as e:
        logger.error(f"❌ Brave local search error: {e}")
        return f"❌ **Unexpected Error:** {str(e)}"

@mcp.tool()
def calculator(expression: str) -> str:
    """
    Calculate mathematical expressions safely
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2 * 3")
    
    Returns:
        Calculation result with the expression and answer
    """
    try:
        if not expression or not expression.strip():
            return "❌ Error: Expression cannot be empty."
        
        expression = expression.strip()
        
        # Allow only safe characters
        allowed_chars = "0123456789+-*/(). "
        if not all(char in allowed_chars for char in expression):
            return "❌ Error: Invalid characters in expression. Only numbers, +, -, *, /, (, ), and spaces are allowed."
        
        # Evaluate the expression
        result = eval(expression)
        
        # Format the result nicely
        if isinstance(result, float):
            if result.is_integer():
                result = int(result)
            else:
                result = round(result, 6)
        
        response = f"🧮 **Calculation Result**\n\n"
        response += f"**Expression:** {expression}\n"
        response += f"**Result:** {result}\n\n"
        response += f"*Calculated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        logger.info(f"🧮 Calculator: '{expression}' = {result}")
        return response
        
    except ZeroDivisionError:
        return "❌ Error: Division by zero is not allowed."
    except SyntaxError:
        return f"❌ Error: Invalid mathematical expression: '{expression}'"
    except Exception as e:
        return f"❌ Error: Could not calculate '{expression}'. {str(e)}"

@mcp.tool()
async def get_weather(place: str) -> str:
    """
    Get current weather information for a location
    
    Args:
        place: Location name (e.g., "New York", "London, UK")
    
    Returns:
        Current weather conditions and forecast
    """
    try:
        if not place or not place.strip():
            return "❌ Error: Location cannot be empty."
        
        place = place.strip().title()
        
        # Check cache
        cache_key = place.lower()
        if cache_key in weather_cache and is_cache_valid(weather_cache[cache_key]):
            cached_result = weather_cache[cache_key]['formatted_result']
            return f"{cached_result}\n\n*⚡ Result from cache (cached {int((time.time() - weather_cache[cache_key]['timestamp']) / 60)} minutes ago)*"
        
        # Mock weather data (replace with real weather API if needed)
        # This is a simplified implementation - you can integrate with OpenWeatherMap, WeatherAPI, etc.
        weather_data = {
            "location": place,
            "temperature": "22°C",
            "condition": "Partly Cloudy",
            "humidity": "65%",
            "wind": "10 km/h NW",
            "pressure": "1013 mb",
            "visibility": "10 km",
            "uv_index": "5 (Moderate)"
        }
        
        result = f"🌤️ **Weather Report for {place}**\n\n"
        result += f"**Current Conditions:**\n"
        result += f"• 🌡️ Temperature: {weather_data['temperature']}\n"
        result += f"• ☁️ Condition: {weather_data['condition']}\n"
        result += f"• 💧 Humidity: {weather_data['humidity']}\n"
        result += f"• 💨 Wind: {weather_data['wind']}\n"
        result += f"• 📊 Pressure: {weather_data['pressure']}\n"
        result += f"• 👁️ Visibility: {weather_data['visibility']}\n"
        result += f"• ☀️ UV Index: {weather_data['uv_index']}\n\n"
        result += f"**3-Day Forecast:**\n"
        result += f"• Today: Partly cloudy, high 25°C, low 18°C\n"
        result += f"• Tomorrow: Sunny, high 27°C, low 20°C\n"
        result += f"• Day 3: Light rain, high 23°C, low 16°C\n\n"
        result += f"---\n*Weather updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        result += f"*Note: This is a demo implementation. For production use, integrate with a real weather API.*"
        
        # Cache result
        weather_cache[cache_key] = {
            'timestamp': time.time(),
            'formatted_result': result
        }
        
        logger.info(f"🌤️ Weather lookup for: {place}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Weather error: {e}")
        return f"❌ **Weather Error:** Could not get weather for '{place}': {str(e)}"

@mcp.tool()
def get_cache_stats() -> str:
    """
    Get cache statistics and server status
    
    Returns:
        Current cache statistics and server information
    """
    try:
        total_search_entries = len(search_cache)
        valid_search_entries = sum(1 for entry in search_cache.values() if is_cache_valid(entry))
        
        total_weather_entries = len(weather_cache)
        valid_weather_entries = sum(1 for entry in weather_cache.values() if is_cache_valid(entry))
        
        configured_clients = len(client_api_keys)
        
        result = f"📊 **MCP Server Statistics**\n\n"
        result += f"**Server Information:**\n"
        result += f"• Server: Brave Search MCP Server\n"
        result += f"• Running since: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        result += f"• Configured clients: {configured_clients}\n\n"
        
        result += f"**Search Cache:**\n"
        result += f"• Total entries: {total_search_entries}\n"
        result += f"• Valid entries: {valid_search_entries}\n"
        result += f"• Expired entries: {total_search_entries - valid_search_entries}\n\n"
        
        result += f"**Weather Cache:**\n"
        result += f"• Total entries: {total_weather_entries}\n"
        result += f"• Valid entries: {valid_weather_entries}\n"
        result += f"• Expired entries: {total_weather_entries - valid_weather_entries}\n\n"
        
        result += f"**Configuration:**\n"
        result += f"• Cache duration: {CACHE_DURATION // 60} minutes\n"
        result += f"• Available tools: 6 tools\n\n"
        
        if total_search_entries > 0:
            result += f"**Recent Searches:**\n"
            recent_keys = list(search_cache.keys())[-5:]  # Last 5 searches
            for key in recent_keys:
                search_type = "Web" if key.startswith("web_") else "Local"
                result += f"• {search_type} search\n"
        
        result += f"\n---\n*Statistics generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return result
        
    except Exception as e:
        return f"❌ **Error:** Could not generate cache statistics: {str(e)}"

@mcp.tool()
def clear_cache() -> str:
    """
    Clear all cached search and weather results
    
    Returns:
        Confirmation message with cleared cache counts
    """
    try:
        search_count = len(search_cache)
        weather_count = len(weather_cache)
        
        search_cache.clear()
        weather_cache.clear()
        
        result = f"🗑️ **Cache Cleared Successfully**\n\n"
        result += f"**Cleared:**\n"
        result += f"• Search cache: {search_count} entries\n"
        result += f"• Weather cache: {weather_count} entries\n"
        result += f"• Total: {search_count + weather_count} entries\n\n"
        result += f"*Cache cleared at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        logger.info(f"🗑️ Cache cleared: {search_count + weather_count} entries")
        return result
        
    except Exception as e:
        return f"❌ **Error:** Could not clear cache: {str(e)}"

# ============================================================================
# PROMPTS
# ============================================================================

@mcp.prompt()
def brave_search_expert(query: str) -> str:
    """
    Expert prompt for Brave web search with optimized search strategies
    
    Args:
        query: The search query to optimize and execute
    
    Returns:
        Optimized search prompt for web search
    """
    return f"""You are a Brave Search expert. Help the user search for: "{query}"

First, use the brave_web_search tool with this optimized query to get current, unbiased results:

Search Query: {query}
Search Parameters: Use count=10 for comprehensive results

Then provide:
1. **Summary**: Key findings from the search results
2. **Key Sources**: Most relevant and authoritative sources found
3. **Additional Context**: Any important background information
4. **Follow-up**: Suggest related search terms if the user wants to explore further

Focus on providing factual, unbiased information from the search results. If the search results are insufficient, suggest alternative search terms or approaches."""

@mcp.prompt()
def local_business_finder(location: str, business_type: str) -> str:
    """
    Expert prompt for finding local businesses using Brave local search
    
    Args:
        location: The location to search in
        business_type: Type of business to find
    
    Returns:
        Optimized local search prompt
    """
    return f"""You are a local business discovery expert. Help the user find {business_type} in {location}.

First, use the brave_local_search tool with this optimized query:

Search Query: {business_type} in {location}
Search Parameters: Use count=5 for focused local results

Then provide:
1. **Top Recommendations**: Best businesses found based on ratings and reviews
2. **Business Details**: Key information like hours, contact, and specialties
3. **Location Tips**: How to get there, parking, nearby landmarks
4. **Alternatives**: Similar businesses if the first options don't meet needs

Focus on providing practical, actionable information to help the user make the best choice for their needs."""

@mcp.prompt()
def weather_assistant(location: str) -> str:
    """
    Expert prompt for weather information and advice
    
    Args:
        location: Location to get weather for
    
    Returns:
        Weather-focused prompt with advice
    """
    return f"""You are a weather expert assistant. Help the user with weather information for {location}.

First, use the get_weather tool to get current conditions:

Location: {location}

Then provide:
1. **Current Conditions**: Summary of current weather
2. **What to Expect**: Practical implications for today's activities
3. **Clothing Advice**: What to wear based on conditions
4. **Activity Suggestions**: Indoor/outdoor activity recommendations
5. **Travel Considerations**: How weather might affect travel plans

Make the weather information practical and actionable for daily planning."""

@mcp.prompt()
def calculation_helper(problem: str) -> str:
    """
    Expert prompt for mathematical calculations and problem solving
    
    Args:
        problem: Mathematical problem or expression to solve
    
    Returns:
        Calculation-focused prompt with explanation
    """
    return f"""You are a mathematics expert. Help the user solve: "{problem}"

First, use the calculator tool to compute the result:

Expression: {problem}

Then provide:
1. **Solution**: The calculated result
2. **Step-by-Step**: Break down how the calculation works
3. **Verification**: Double-check the result makes sense
4. **Related Concepts**: Explain any mathematical concepts involved
5. **Similar Problems**: Suggest related calculations if helpful

Make mathematics accessible and educational while providing accurate results."""

# ============================================================================
# CUSTOM MCP SERVER CLASS WITH PROPER INITIALIZATION
# ============================================================================

class CustomMCPServer:
    """Custom MCP Server wrapper to handle initialization options properly"""
    
    def __init__(self, fastmcp_instance):
        self._fastmcp = fastmcp_instance
        self.name = "Brave Search MCP Server"
        self.version = "1.0.0"
    
    def create_initialization_options(self):
        """Create proper initialization options with required capabilities"""
        return InitializationOptions(
            server_name=self.name,
            server_version=self.version,
            capabilities=ServerCapabilities(
                tools=ToolsCapability(),
                prompts=PromptsCapability()
            )
        )
    
    async def run(self, read_stream, write_stream, init_options):
        """Run the MCP server with proper options"""
        return await self._fastmcp.run(read_stream, write_stream, init_options)

# ============================================================================
# INITIALIZATION
# ============================================================================

# Configure default API key if provided
# Note: In production, you should get this from environment variables or secure configuration
DEFAULT_API_KEY = "BSAQIFoBulbULfcL6RMBxRWCtopFY0E"
if DEFAULT_API_KEY:
    set_client_api_key(DEFAULT_API_KEY, "default")
    logger.info("🚀 FastMCP Server initialized with default API key")

# Create the custom MCP server instance
_mcp_server = CustomMCPServer(mcp)

logger.info("🚀 FastMCP Server ready")
logger.info("🛠️  Available tools: configure_brave_key, brave_web_search, brave_local_search, calculator, get_weather, get_cache_stats, clear_cache")
logger.info("📝 Available prompts: brave_search_expert, local_business_finder, weather_assistant, calculation_helper")

# Export the custom MCP server instance
mcp_server = _mcp_server
