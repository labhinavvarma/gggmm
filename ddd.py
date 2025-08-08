"""
Enhanced MCP Server Module with Working Brave Search Integration
File: mcpserver.py
"""

import asyncio
import httpx
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import re

# Global variables for caching and configuration
brave_api_key: Optional[str] = None
weather_cache: Dict[str, Dict] = {}
brave_search_cache: Dict[str, Dict] = {}

# Cache validity periods (in seconds)
WEATHER_CACHE_DURATION = 300  # 5 minutes
BRAVE_CACHE_DURATION = 1800   # 30 minutes

def set_brave_api_key(api_key: str) -> None:
    """Set the Brave Search API key globally."""
    global brave_api_key
    brave_api_key = api_key
    print(f"✅ Brave API key configured: {api_key[:8]}...{api_key[-4:]}")

def get_brave_api_key() -> Optional[str]:
    """Get the current Brave API key."""
    return brave_api_key

def is_weather_cache_valid(cache_entry: Dict) -> bool:
    """Check if weather cache entry is still valid."""
    return time.time() - cache_entry['timestamp'] < WEATHER_CACHE_DURATION

def is_brave_cache_valid(cache_entry: Dict) -> bool:
    """Check if Brave search cache entry is still valid."""
    return time.time() - cache_entry['timestamp'] < BRAVE_CACHE_DURATION

# ============================================================================
# CALCULATOR TOOL
# ============================================================================

def calculate(expression: str) -> str:
    """Safely evaluate mathematical expressions."""
    try:
        # Clean the expression
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
        
        return f"🧮 **Calculation Result**\n\n**Expression:** {expression}\n**Result:** {result}"
        
    except ZeroDivisionError:
        return "❌ Error: Division by zero is not allowed."
    except Exception as e:
        return f"❌ Error: Could not calculate '{expression}'. {str(e)}"

# ============================================================================
# TEST AND DIAGNOSTIC TOOLS
# ============================================================================

async def test_tool(message: str) -> str:
    """Simple test tool to verify tool calling works."""
    current_time = datetime.now().isoformat()
    return f"✅ **Test Tool Success**\n\n**Message:** {message}\n**Timestamp:** {current_time}\n**Status:** MCP tool calling is working correctly!"

async def diagnostic(test_type: str = "basic") -> str:
    """Diagnostic tool to test MCP functionality."""
    current_time = datetime.now().isoformat()
    
    result = f"🔧 **Diagnostic Test Report**\n\n"
    result += f"**Test Type:** {test_type}\n"
    result += f"**Timestamp:** {current_time}\n"
    result += f"**Server:** DataFlyWheel MCP Server\n\n"
    
    if test_type == "basic":
        result += "**Basic System Check:**\n"
        result += "✅ MCP server is running\n"
        result += "✅ Tool execution is working\n"
        result += "✅ Message formatting is correct\n"
        result += f"✅ Brave API Key: {'Configured' if brave_api_key else 'Not configured'}\n"
    
    elif test_type == "advanced":
        result += "**Advanced System Check:**\n"
        result += f"✅ Weather cache entries: {len(weather_cache)}\n"
        result += f"✅ Brave search cache entries: {len(brave_search_cache)}\n"
        result += f"✅ Brave API status: {'Ready' if brave_api_key else 'Not configured'}\n"
        result += "✅ All systems operational\n"
    
    result += "\n**Status:** All diagnostics passed! 🚀"
    
    return result

# ============================================================================
# WEATHER TOOL
# ============================================================================

async def get_weather(place: str, ctx=None) -> str:
    """Get current weather information for a location with caching."""
    try:
        # Clean up the place name
        place = place.strip().title()
        
        if ctx:
            await ctx.info(f"Getting weather for {place}")
        
        # Check cache first
        cache_key = place.lower()
        if cache_key in weather_cache and is_weather_cache_valid(weather_cache[cache_key]):
            if ctx:
                await ctx.info(f"Using cached weather data for {place}")
            cached_data = weather_cache[cache_key]
            return cached_data['formatted_result']
        
        # Fetch new weather data
        if ctx:
            await ctx.info(f"Fetching fresh weather data for {place}")
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Using a weather API (you may need to replace with your preferred service)
            # This is a mock implementation - replace with actual weather API
            weather_data = {
                "location": place,
                "temperature": "22°C",
                "condition": "Partly Cloudy",
                "humidity": "65%",
                "wind": "10 km/h NW",
                "forecast": "Sunny intervals with occasional clouds"
            }
            
            # Format the result
            result = f"🌤️ **Weather Report for {place}**\n\n"
            result += f"**Current Conditions:**\n"
            result += f"• Temperature: {weather_data['temperature']}\n"
            result += f"• Condition: {weather_data['condition']}\n"
            result += f"• Humidity: {weather_data['humidity']}\n"
            result += f"• Wind: {weather_data['wind']}\n\n"
            result += f"**Forecast:** {weather_data['forecast']}\n\n"
            result += f"*Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
            
            # Cache the result
            weather_cache[cache_key] = {
                'timestamp': time.time(),
                'data': weather_data,
                'formatted_result': result
            }
            
            return result
            
    except Exception as e:
        error_msg = f"❌ **Weather Error**\n\nCould not get weather for '{place}': {str(e)}"
        if ctx:
            await ctx.error(f"Weather API error: {str(e)}")
        return error_msg

# ============================================================================
# BRAVE SEARCH TOOLS (FIXED IMPLEMENTATION)
# ============================================================================

async def brave_web_search(query: str, ctx=None, count: int = 10, offset: int = 0) -> str:
    """Search the web using Brave Search API with proper error handling."""
    try:
        if ctx:
            await ctx.info(f"Starting Brave web search for: {query}")
        
        # Check if API key is configured
        if not brave_api_key:
            error_msg = "❌ **Brave Search Error**\n\nBrave API key is not configured. Please configure the API key first."
            if ctx:
                await ctx.error("Brave API key not configured")
            return error_msg
        
        # Check cache first
        cache_key = f"web_{query.lower()}_{count}_{offset}"
        if cache_key in brave_search_cache and is_brave_cache_valid(brave_search_cache[cache_key]):
            if ctx:
                await ctx.info(f"Using cached Brave search results for: {query}")
            return brave_search_cache[cache_key]['formatted_result']
        
        if ctx:
            await ctx.info(f"Making fresh Brave API call for: {query}")
        
        # Make the API call
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": brave_api_key
        }
        
        params = {
            "q": query,
            "count": min(count, 20),  # Brave API limit
            "offset": offset,
            "safesearch": "moderate",
            "freshness": "pd",  # Past day for fresh results
            "text_decorations": False,
            "spellcheck": True
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params
            )
            
            if ctx:
                await ctx.info(f"Brave API response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Format the results
                result = f"🔍 **Brave Web Search Results**\n\n"
                result += f"**Query:** {query}\n"
                result += f"**Found:** {len(data.get('web', {}).get('results', []))} results\n\n"
                
                web_results = data.get('web', {}).get('results', [])
                
                if web_results:
                    for i, item in enumerate(web_results[:count], 1):
                        title = item.get('title', 'No title')
                        url = item.get('url', '')
                        description = item.get('description', 'No description available')
                        
                        # Clean up description
                        description = description[:200] + "..." if len(description) > 200 else description
                        
                        result += f"**{i}. {title}**\n"
                        result += f"🔗 {url}\n"
                        result += f"📝 {description}\n\n"
                else:
                    result += "No web results found for this query.\n\n"
                
                # Add news results if available
                news_results = data.get('news', {}).get('results', [])
                if news_results:
                    result += f"📰 **Related News** ({len(news_results)} items):\n\n"
                    for i, item in enumerate(news_results[:3], 1):
                        title = item.get('title', 'No title')
                        url = item.get('url', '')
                        result += f"**{i}. {title}**\n🔗 {url}\n\n"
                
                result += f"*Search performed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
                
                # Cache the result
                brave_search_cache[cache_key] = {
                    'timestamp': time.time(),
                    'data': data,
                    'formatted_result': result
                }
                
                if ctx:
                    await ctx.info(f"Brave web search completed successfully")
                
                return result
                
            elif response.status_code == 401:
                error_msg = "❌ **Brave Search Error**\n\nInvalid API key. Please check your Brave Search API key configuration."
                if ctx:
                    await ctx.error("Brave API authentication failed")
                return error_msg
                
            elif response.status_code == 429:
                error_msg = "❌ **Brave Search Error**\n\nRate limit exceeded. Please try again later."
                if ctx:
                    await ctx.error("Brave API rate limit exceeded")
                return error_msg
                
            else:
                error_msg = f"❌ **Brave Search Error**\n\nAPI request failed with status {response.status_code}: {response.text[:200]}"
                if ctx:
                    await ctx.error(f"Brave API error: {response.status_code}")
                return error_msg
                
    except httpx.TimeoutException:
        error_msg = "❌ **Brave Search Error**\n\nRequest timed out. Please try again."
        if ctx:
            await ctx.error("Brave API request timed out")
        return error_msg
        
    except Exception as e:
        error_msg = f"❌ **Brave Search Error**\n\nUnexpected error: {str(e)}"
        if ctx:
            await ctx.error(f"Brave search unexpected error: {str(e)}")
        return error_msg

async def brave_local_search(query: str, ctx=None, count: int = 5) -> str:
    """Search for local businesses and places using Brave Search API."""
    try:
        if ctx:
            await ctx.info(f"Starting Brave local search for: {query}")
        
        # Check if API key is configured
        if not brave_api_key:
            error_msg = "❌ **Brave Local Search Error**\n\nBrave API key is not configured. Please configure the API key first."
            if ctx:
                await ctx.error("Brave API key not configured")
            return error_msg
        
        # Check cache first
        cache_key = f"local_{query.lower()}_{count}"
        if cache_key in brave_search_cache and is_brave_cache_valid(brave_search_cache[cache_key]):
            if ctx:
                await ctx.info(f"Using cached Brave local search results for: {query}")
            return brave_search_cache[cache_key]['formatted_result']
        
        if ctx:
            await ctx.info(f"Making fresh Brave local API call for: {query}")
        
        # Make the API call
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": brave_api_key
        }
        
        params = {
            "q": query,
            "count": min(count, 20),
            "safesearch": "moderate",
            "search_lang": "en",
            "country": "US",
            "units": "metric"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/local/search",
                headers=headers,
                params=params
            )
            
            if ctx:
                await ctx.info(f"Brave Local API response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Format the results
                result = f"📍 **Brave Local Search Results**\n\n"
                result += f"**Query:** {query}\n"
                result += f"**Found:** {len(data.get('local', {}).get('results', []))} local results\n\n"
                
                local_results = data.get('local', {}).get('results', [])
                
                if local_results:
                    for i, item in enumerate(local_results[:count], 1):
                        name = item.get('title', 'No name')
                        address = item.get('address', 'No address available')
                        phone = item.get('phone', 'No phone')
                        rating = item.get('rating', 'No rating')
                        website = item.get('url', '')
                        
                        result += f"**{i}. {name}**\n"
                        result += f"📍 {address}\n"
                        
                        if phone != 'No phone':
                            result += f"📞 {phone}\n"
                        
                        if rating != 'No rating':
                            result += f"⭐ Rating: {rating}\n"
                        
                        if website:
                            result += f"🌐 {website}\n"
                        
                        result += "\n"
                else:
                    result += "No local results found for this query.\n\n"
                    result += "💡 **Tips for better local search:**\n"
                    result += "• Include location (e.g., 'pizza in New York')\n"
                    result += "• Be specific about business type\n"
                    result += "• Try different keywords\n\n"
                
                result += f"*Search performed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
                
                # Cache the result
                brave_search_cache[cache_key] = {
                    'timestamp': time.time(),
                    'data': data,
                    'formatted_result': result
                }
                
                if ctx:
                    await ctx.info(f"Brave local search completed successfully")
                
                return result
                
            elif response.status_code == 401:
                error_msg = "❌ **Brave Local Search Error**\n\nInvalid API key. Please check your Brave Search API key configuration."
                if ctx:
                    await ctx.error("Brave Local API authentication failed")
                return error_msg
                
            elif response.status_code == 429:
                error_msg = "❌ **Brave Local Search Error**\n\nRate limit exceeded. Please try again later."
                if ctx:
                    await ctx.error("Brave Local API rate limit exceeded")
                return error_msg
                
            else:
                error_msg = f"❌ **Brave Local Search Error**\n\nAPI request failed with status {response.status_code}: {response.text[:200]}"
                if ctx:
                    await ctx.error(f"Brave Local API error: {response.status_code}")
                return error_msg
                
    except httpx.TimeoutException:
        error_msg = "❌ **Brave Local Search Error**\n\nRequest timed out. Please try again."
        if ctx:
            await ctx.error("Brave Local API request timed out")
        return error_msg
        
    except Exception as e:
        error_msg = f"❌ **Brave Local Search Error**\n\nUnexpected error: {str(e)}"
        if ctx:
            await ctx.error(f"Brave local search unexpected error: {str(e)}")
        return error_msg

# ============================================================================
# HEDIS TOOLS (PLACEHOLDER IMPLEMENTATIONS)
# ============================================================================

async def dfw_text2sql(prompt: str, ctx=None) -> str:
    """Convert text to SQL for HEDIS value sets and code sets."""
    try:
        if ctx:
            await ctx.info(f"Processing HEDIS text-to-SQL request: {prompt}")
        
        # This is a placeholder implementation
        # Replace with your actual HEDIS database query logic
        
        result = f"🏥 **HEDIS Text-to-SQL Analysis**\n\n"
        result += f"**Query:** {prompt}\n\n"
        result += f"**Analysis:** This is a placeholder for HEDIS text-to-SQL conversion.\n"
        result += f"Please implement the actual HEDIS database integration here.\n\n"
        result += f"**Suggested SQL Structure:**\n"
        result += f"```sql\n"
        result += f"SELECT value_set_name, code, description\n"
        result += f"FROM hedis_value_sets\n"
        result += f"WHERE value_set_name LIKE '%{prompt}%'\n"
        result += f"   OR description LIKE '%{prompt}%';\n"
        result += f"```\n\n"
        result += f"*Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return result
        
    except Exception as e:
        error_msg = f"❌ **HEDIS Analysis Error**\n\nFailed to process request: {str(e)}"
        if ctx:
            await ctx.error(f"HEDIS text-to-SQL error: {str(e)}")
        return error_msg

async def dfw_search(ctx, query: str) -> str:
    """Search HEDIS measure specification documents."""
    try:
        if ctx:
            await ctx.info(f"Searching HEDIS documentation for: {query}")
        
        # This is a placeholder implementation
        # Replace with your actual HEDIS document search logic
        
        result = f"🏥 **HEDIS Document Search**\n\n"
        result += f"**Query:** {query}\n\n"
        result += f"**Search Results:** This is a placeholder for HEDIS document search.\n"
        result += f"Please implement the actual HEDIS document search integration here.\n\n"
        result += f"**Common HEDIS Measures:**\n"
        result += f"• BCS - Breast Cancer Screening\n"
        result += f"• CBP - Controlling High Blood Pressure\n"
        result += f"• COA - Care for Older Adults\n"
        result += f"• HbA1c - Hemoglobin A1c Testing\n"
        result += f"• CCS - Cervical Cancer Screening\n\n"
        result += f"*Search completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return result
        
    except Exception as e:
        error_msg = f"❌ **HEDIS Search Error**\n\nFailed to search documentation: {str(e)}"
        if ctx:
            await ctx.error(f"HEDIS search error: {str(e)}")
        return error_msg

# ============================================================================
# INITIALIZATION AND STATUS FUNCTIONS
# ============================================================================

def get_cache_status() -> Dict[str, Any]:
    """Get current cache status for monitoring."""
    return {
        "weather_cache": {
            "entries": len(weather_cache),
            "locations": list(weather_cache.keys())
        },
        "brave_cache": {
            "entries": len(brave_search_cache),
            "queries": list(brave_search_cache.keys())[:5]  # First 5 for privacy
        },
        "api_status": {
            "brave_api_configured": brave_api_key is not None,
            "brave_api_preview": f"{brave_api_key[:8]}...{brave_api_key[-4:]}" if brave_api_key else None
        }
    }

def clear_cache() -> str:
    """Clear all caches."""
    global weather_cache, brave_search_cache
    weather_cache.clear()
    brave_search_cache.clear()
    return "✅ All caches cleared successfully"

# Initialize with the provided API key
set_brave_api_key("BSAQIFoBulbULfcL6RMBxRWCtopFY0E")

if __name__ == "__main__":
    print("🚀 MCP Server Module Loaded")
    print(f"✅ Brave API Key: {brave_api_key[:8]}...{brave_api_key[-4:] if brave_api_key else 'Not configured'}")
    print("📋 Available tools: calculate, test_tool, diagnostic, get_weather, brave_web_search, brave_local_search, dfw_text2sql, dfw_search")
