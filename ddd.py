from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import httpx
from dataclasses import dataclass
from urllib.parse import urlparse
from pathlib import Path
import json
import asyncio
import snowflake.connector
import requests
import os
from loguru import logger
import logging
import re
import urllib.parse
from datetime import datetime, timedelta
from snowflake.connector import SnowflakeConnection
from ReduceReuseRecycleGENAI.snowflake import snowflake_conn
from snowflake.connector.errors import DatabaseError
from snowflake.core import Root
from typing import Optional, List, Dict
from fastapi import (
    HTTPException,
    status,
)
from mcp.server.fastmcp.prompts.base import Message
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts import Prompt
import mcp.types as types
from functools import partial
import sys
import traceback
import time

# Create a named server
mcp = FastMCP("DataFlyWheel App")

@dataclass
class AppContext:
    conn : SnowflakeConnection
    db: str
    schema: str
    host: str

class RateLimiter:
    """Rate limiter to prevent overwhelming external services"""
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    async def acquire(self):
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [
            req for req in self.requests if now - req < timedelta(minutes=1)
        ]

        if len(self.requests) >= self.requests_per_minute:
            # Wait until we can make another request
            wait_time = 60 - (now - self.requests[0]).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.requests.append(now)

# Initialize rate limiter
rate_limiter = RateLimiter(requests_per_minute=20)

#Stag name may need to be determined; requires code change
#Resources; Have access to resources required for the server; Cortex Search; Cortex stage schematic config; stage area should be fully qualified name

@mcp.resource(uri="schematiclayer://cortex_analyst/schematic_models/{stagename}/list",name="hedis_schematic_models",description="Hedis Schematic models")
async def get_schematic_model(stagename: str):
    """Cortex analyst scematic layer model, model is in yaml format"""
    #ctx = mcp.get_context()
    HOST = "carelon-eda-preprod.privatelink.snowflakecomputing.com"
    conn = snowflake_conn(
           logger,
           aplctn_cd="aedl",
           env="preprod",
           region_name="us-east-1",
           warehouse_size_suffix="",
           prefix=""
        )
    #conn = ctx.request_context.lifespan_context.conn
    db = 'POC_SPC_SNOWPARK_DB'
    schema = 'HEDIS_SCHEMA'
    cursor = conn.cursor()
    snfw_model_list = cursor.execute("LIST @{db}.{schema}.{stagename}".format(db=db, schema=schema, stagename=stagename))
    return [stg_nm[0].split("/")[-1] for stg_nm in snfw_model_list if stg_nm[0].endswith('yaml')]

@mcp.resource("search://cortex_search/search_obj/list")
async def get_search_service():
    """Cortex search service"""
    HOST = "carelon-eda-preprod.privatelink.snowflakecomputing.com"
    conn = snowflake_conn(
           logger,
           aplctn_cd="aedl",
           env="preprod",
           region_name="us-east-1",
           warehouse_size_suffix="",
           prefix=""
        )
    #conn = ctx.request_context.lifespan_context.conn
    db = 'POC_SPC_SNOWPARK_DB'
    schema = 'HEDIS_SCHEMA'
    cursor = conn.cursor()
    snfw_search_objs = cursor.execute("SHOW CORTEX SEARCH SERVICES IN SCHEMA {db}.{schema}".format(db=db, schema=schema))
    result = [search_obj[1] for search_obj in snfw_search_objs.fetchall()]
    return result

@mcp.resource("genaiplatform://{aplctn_cd}/frequent_questions/{user_context}")
async def frequent_questions(aplctn_cd: str, user_context: str) -> List[str]:
    resource_name = aplctn_cd + "_freq_questions.json"
    freq_questions = json.load(open(resource_name))
    aplcn_question = freq_questions.get(aplctn_cd)
    return [rec["prompt"] for rec in aplcn_question if rec["user_context"] == user_context]

@mcp.resource("genaiplatform://{aplctn_cd}/prompts/{prompt_name}")
async def prompt_templates(aplctn_cd: str, prompt_name: str) -> List[str]:
    resource_name = aplctn_cd + "_prompts.json"
    prompt_data = json.load(open(resource_name))
    aplcn_prompts = prompt_data.get(aplctn_cd)
    return [rec["content"] for rec in aplcn_prompts if rec["prompt_name"] == prompt_name]

@mcp.tool(
        name="add-frequent-questions"
       ,description="""
        Tool to add frequent questions to MCP server
        Example inputs:
        {
           "uri"
        }
        Args:
               uri (str):  text to be passed
               questions (list):
               [
                 {
                   "user_context" (str): "User context for the prompt"
                   "prompt" (str): "prompt"
                 }
               ]
        """
)
async def add_frequent_questions(ctx: Context,uri: str,questions: list) -> list:
    #Parse and extract aplctn_cd and user_context (urllib)
    url_path = urlparse(uri)
    aplctn_cd = url_path.netloc
    user_context = Path(url_path.path).name
    file_data = {}
    file_name = aplctn_cd + "_freq_questions.json"
    if Path(file_name).exists():
        file_data  = json.load(open(file_name,'r'))
        file_data[aplctn_cd].extend(questions)
    else:
        file_data[aplctn_cd] =  questions
    index_dict = {
        user_context: set()
    }
    result = []
    #Remove duplicates
    for elm in file_data[aplctn_cd]:
        if elm["user_context"] == user_context and elm['prompt'] not in index_dict[user_context]:
            result.append(elm)
            index_dict[user_context].add(elm['prompt'])
    file_data[aplctn_cd] = result
    file = open(file_name,'w')
    file.write(json.dumps(file_data))
    return file_data[aplctn_cd]

@mcp.tool(
        name="add-prompts"
       ,description="""
        Tool to add prompts to MCP server
        Example inputs:
        {
           ""
        }
        Args:
               uri (str):  text to be passed
               prompts (dict):
                 {
                   "prompt_name" (str): "Unique name assigned to prompt for a application"
                   "description" (str): "Prompt description"
                   "content" (str): "Prompt content"
                 }
        """
)
async def add_prompts(ctx: Context,uri: str,prompt: dict) -> dict:
    #Parse and extract aplctn_cd and user_context (urllib)
    url_path = urlparse(uri)
    aplctn_cd = url_path.netloc
    prompt_name = Path(url_path.path).name
    #Before adding the prompt to file add to the server
    ##Add prompts to server
    def func1(query: str ):
        return [
            {
                "role": "user",
                "content": prompt["content"] + f"\n  {query}"
            }
        ]
    ctx.fastmcp.add_prompt(
        Prompt.from_function(
            func1,name = prompt["prompt_name"],description=prompt["description"])
    )
    file_data = {}
    file_name = aplctn_cd + "_prompts.json"
    if Path(file_name).exists():
        file = open(file_name,'r')
        file_data  = json.load(file)
        file_data[aplctn_cd].append(prompt)
    else:
        file_data[aplctn_cd] =  [prompt]
    file = open(file_name,'w')
    file.write(json.dumps(file_data))
    return prompt

#Tools; corex Analyst; Cortex Search; Cortex Complete

@mcp.tool(
        name="DFWAnalyst"
       ,description="""
        Coneverts text to valid SQL which can be executed on HEDIS value sets and code sets.
        Example inputs:
           What are the codes in <some value> Value Set?
        Returns valid sql to retive data from underlying value sets and code sets.
        Args:
               prompt (str):  text to be passed
        """
)
async def dfw_text2sql(prompt:str,ctx: Context) -> dict:
    """Tool to convert natural language text to snowflake sql for hedis system, text should be passed as 'prompt' input perameter"""
    HOST = "carelon-eda-preprod.privatelink.snowflakecomputing.com"
    conn = snowflake_conn(
           logger,
           aplctn_cd="aedl",
           env="preprod",
           region_name="us-east-1",
           warehouse_size_suffix="",
           prefix=""
        )
    #conn = ctx.request_context.lifespan_context.conn
    db = 'POC_SPC_SNOWPARK_DB'
    schema = 'HEDIS_SCHEMA'
    host = HOST
    stage_name = "hedis_stage_full"
    file_name = "hedis_semantic_model_complete.yaml"
    request_body = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "semantic_model_file": f"@{db}.{schema}.{stage_name}/{file_name}",
    }
    token = conn.rest.token
    resp = requests.post(
        url=f"https://{host}/api/v2/cortex/analyst/message",
        json=request_body,
        headers={
            "Authorization": f'Snowflake Token="{token}"',
            "Content-Type": "application/json",
        },
    )
    return resp.json()

#Need to change the type of serch, implimented in the below code; Revisit
@mcp.tool(
        name="DFWSearch"
       ,description= """
        Searches HEDIS measure specification documents.
        Example inputs:
        What is the age criteria for  BCS Measure ?
        What is EED Measure in HEDIS?
        Describe COA Measure?
        What LOB is COA measure scoped under?
        Returns information utilizing HEDIS measure speficication documents.
        Args:
              query (str): text to be passed
       """
)
async def dfw_search(ctx: Context,query: str):
    """Tool to provide search againest HEDIS business documents for the year 2024, search string should be provided as 'query' perameter"""
    HOST = "carelon-eda-preprod.privatelink.snowflakecomputing.com"
    conn = snowflake_conn(
           logger,
           aplctn_cd="aedl",
           env="preprod",
           region_name="us-east-1",
           warehouse_size_suffix="",
           prefix=""
        )
    #conn = ctx.request_context.lifespan_context.conn
    db = 'POC_SPC_SNOWPARK_DB'
    schema = 'HEDIS_SCHEMA'
    search_service = 'CS_HEDIS_FULL_2024'
    columns = ['chunk']
    limit = 2
    root = Root(conn)
    search_service = root.databases[db].schemas[schema].cortex_search_services[search_service]
    response = search_service.search(
        query=query,
        columns=columns,
        limit=limit
    )
    return response.to_json()

@mcp.tool(
        name="calculator",
        description="""
        Evaluates a basic arithmetic expression.
        Supports: +, -, *, /, parentheses, decimals.
        Example inputs:
        3+4/5
        3.0/6*8
        Returns decimal result
        Args:
             expression (str): Arthamatic expression input
        """
)
def calculate(expression: str) -> str:
    """
    Evaluates a basic arithmetic expression.
    Supports: +, -, *, /, parentheses, decimals.
    """
    print(f" calculate() called with expression: {expression}", flush=True)
    try:
        allowed_chars = "0123456789+-*/(). "
        if any(char not in allowed_chars for char in expression):
            return " Invalid characters in expression."
        result = eval(expression)
        return f" Result: {result}"
    except Exception as e:
        print(" Error in calculate:", str(e), flush=True)
        return f" Error: {str(e)}"

#This may required to be integrated in main agent
@mcp.tool(
        name="suggested_top_prompts",
        description="""
        Suggests requested number of prompts with given context.
        Example Input:
        {
          top_n_suggestions: 3,
          context: "Initialization" | "The age criteria for the BCS (Breast Cancer Screening) measure is 50-74 years of age."
          aplctn_cd: "hedis"
        }
        Returns List of string values.
        Args:
            top_n_suggestions (int): how many suggestions to be generated.
            context (str): context that need to be used for the promt suggestions.
            aplctn_cd (str): application code.
        """
)
async def question_suggestions(ctx: Context,aplctn_cd: str, app_lvl_prefix: str, session_id: str, top_n: int = 3,context: str="Initialization",llm_flg: bool = False):
    """Tool to suggest aditional prompts with in the provided context, context should be passed as 'context' input perameter"""
    if  not llm_flg:
        return ctx.read_resource(f"genaiplatform://{aplctn_cd}/frequent_questions/{context}")
    try:
        sf_conn = SnowFlakeConnector.get_conn(
            aplctn_cd,
            app_lvl_prefix,
            session_id,
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User not authorized to resources"
        )
    clnt = httpx.AsyncClient(verify=False)
    request_body = {
        "model": "llama3.1-70b-elevance",
        "messages": [
            {
                "role": "user",
                "content": f"""
                You are an expert in suggesting hypothetical questions.
                Suggest a list of {top_n} hypothetical questions that the below context could be used to answer:
                {context}
                Return List with hypothetical questions
                """
            }
        ]
    }
    headers = {
        "Authorization": f'Snowflake Token="{sf_conn.rest.token}"',
        "Content-Type": "application/json",
        "Accept": "application/json",
        "method":"cortex",
    }
    url = "https://jib90126.us-east-1.privatelink.snowflakecomputing.com/api/v2/cortex/inference:complete"
    response_text = []
    async with clnt.stream('POST', url, headers=headers, json=request_body) as response:
        if response.is_client_error:
            error_message = await response.aread()
            raise HTTPException(
                status_code=response.status_code,
                detail=error_message.decode("utf-8")
            )
        if response.is_server_error:
            error_message = await response.aread()
            raise HTTPException(
                status_code=response.status_code,
                detail=error_message.decode("utf-8")
            )
        # Stream the response content
        async for result_chunk in response.aiter_bytes():
            for elem in result_chunk.split(b'\n\n'):
                if b'content' in elem:  # Check for data presence
                    chunk_dict = json.loads(elem.replace(b'data: ', b''))
                    print(chunk_dict)
                    full_response = chunk_dict['choices'][0]['delta']['text']
                    full_response = full_response
                    response_text.append(full_response)
    return json.loads(response_text)

# === WEATHER TOOL (NOMINATIM + NWS) ===
@mcp.tool(
        name="get_weather",
        description="""
        Get weather forecast for a place (e.g., 'New York') without needing an API key.
        Uses Nominatim for geocoding and National Weather Service for forecasts.
        Works best with US locations.
        Args:
            place (str): Location name (e.g., 'New York', 'Los Angeles, CA')
        """
)
async def get_weather(ctx: Context, place: str) -> str:
    """Get weather forecast for a place using Nominatim + NWS APIs"""
    try:
        await ctx.info(f"🌤️ Getting weather for: {place}")
        
        # Step 1: Get coordinates using Nominatim (no key needed)
        nominatim_url = f"https://nominatim.openstreetmap.org/search?q={place}&format=json&limit=1&countrycodes=us"
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(nominatim_url, headers={"User-Agent": "MCP Weather Tool"})
            response.raise_for_status()
            data = response.json()

            if not data:
                return f"❌ Could not find location: {place}. Please try a more specific city name."

            latitude = data[0]["lat"]
            longitude = data[0]["lon"]
            display_name = data[0].get("display_name", place)
            
            await ctx.info(f"📍 Found coordinates: {latitude}, {longitude} for {display_name}")

            # Step 2: Use NWS API to get forecast
            nws_url = f"https://api.weather.gov/points/{latitude},{longitude}"
            headers = {"User-Agent": "MCP Weather Tool"}
            
            points_resp = await client.get(nws_url, headers=headers)
            
            if points_resp.status_code == 404:
                return f"❌ Weather service not available for {place}. The National Weather Service only covers US locations."
            
            points_resp.raise_for_status()
            points_data = points_resp.json()

            forecast_url = points_data["properties"]["forecast"]
            city = points_data["properties"]["relativeLocation"]["properties"]["city"]
            state = points_data["properties"]["relativeLocation"]["properties"]["state"]

            forecast_resp = await client.get(forecast_url, headers=headers)
            forecast_resp.raise_for_status()
            forecast_data = forecast_resp.json()

            period = forecast_data["properties"]["periods"][0]
            
            result = f"🌤️ **Weather for {city}, {state}:**\n\n"
            result += f"📅 **{period['name']}**\n"
            result += f"🌡️ **Temperature:** {period['temperature']}°{period['temperatureUnit']}\n"
            result += f"☁️ **Conditions:** {period['shortForecast']}\n"
            result += f"💨 **Wind:** {period['windSpeed']} {period['windDirection']}\n"
            result += f"📝 **Detailed Forecast:** {period['detailedForecast']}"
            
            await ctx.info("✅ Weather data retrieved successfully")
            return result

    except Exception as e:
        await ctx.error(f"Weather request failed: {e}")
        return f"❌ Error fetching weather: {str(e)}"

# === BRAVE SEARCH TOOLS ===
# Note: BRAVE_API_KEY will be provided by the client

@mcp.tool(
        name="brave_web_search",
        description="""
        Performs a web search using the Brave Search API, ideal for general queries, news, articles, and online content.
        Use this for broad information gathering, recent events, or when you need diverse web sources.
        Supports pagination, content filtering, and freshness controls.
        Maximum 20 results per request, with offset for pagination.
        Args:
            query (str): Search query (max 400 chars, 50 words)
            count (int): Number of results (1-20, default 10)
            offset (int): Pagination offset (max 9, default 0)
        """
)
async def brave_web_search(ctx: Context, query: str, count: int = 10, offset: int = 0) -> str:
    """Perform web search using Brave Search API"""
    try:
        await ctx.info(f"🔍 Brave web search for: {query}")
        
        # Get API key from environment or context (will be set by client)
        brave_api_key = os.environ.get('BRAVE_API_KEY')
        if not brave_api_key:
            return "❌ Brave API key not configured. Please set BRAVE_API_KEY environment variable."
        
        url = "https://api.search.brave.com/res/v1/web/search"
        params = {
            'q': query,
            'count': min(count, 20),  # API limit
            'offset': offset
        }
        
        headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': brave_api_key
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Extract web results
            web_results = data.get('web', {}).get('results', [])
            
            if not web_results:
                return f"❌ No web results found for: {query}"
            
            results = []
            results.append(f"🔍 **Brave Web Search Results for '{query}':**\n")
            
            for i, result in enumerate(web_results, 1):
                title = result.get('title', 'No title')
                description = result.get('description', 'No description')
                url = result.get('url', 'No URL')
                
                results.append(f"## {i}. {title}")
                results.append(f"**Description:** {description}")
                results.append(f"**URL:** {url}")
                results.append("")
            
            await ctx.info(f"✅ Found {len(web_results)} web results")
            return "\n".join(results)
            
    except Exception as e:
        await ctx.error(f"Brave web search failed: {e}")
        return f"❌ Brave web search error: {str(e)}"

@mcp.tool(
        name="brave_local_search",
        description="""
        Searches for local businesses and places using Brave's Local Search API.
        Best for queries related to physical locations, businesses, restaurants, services, etc.
        Returns detailed information including business names, addresses, ratings, phone numbers, and opening hours.
        Use this when the query implies 'near me' or mentions specific locations.
        Automatically falls back to web search if no local results are found.
        Args:
            query (str): Local search query (e.g. 'pizza near Central Park')
            count (int): Number of results (1-20, default 5)
        """
)
async def brave_local_search(ctx: Context, query: str, count: int = 5) -> str:
    """Perform local search using Brave Local Search API"""
    try:
        await ctx.info(f"📍 Brave local search for: {query}")
        
        # Get API key from environment or context (will be set by client)
        brave_api_key = os.environ.get('BRAVE_API_KEY')
        if not brave_api_key:
            return "❌ Brave API key not configured. Please set BRAVE_API_KEY environment variable."
        
        headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': brave_api_key
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Step 1: Search for locations
            web_url = "https://api.search.brave.com/res/v1/web/search"
            web_params = {
                'q': query,
                'search_lang': 'en',
                'result_filter': 'locations',
                'count': min(count, 20)
            }
            
            web_response = await client.get(web_url, params=web_params, headers=headers)
            web_response.raise_for_status()
            web_data = web_response.json()
            
            location_ids = []
            locations_results = web_data.get('locations', {}).get('results', [])
            for loc in locations_results:
                if loc.get('id'):
                    location_ids.append(loc['id'])
            
            if not location_ids:
                # Fallback to web search
                await ctx.info("📍 No local results found, falling back to web search")
                return await brave_web_search(ctx, query, count)
            
            # Step 2: Get POI details
            pois_url = "https://api.search.brave.com/res/v1/local/pois"
            pois_params = {'ids': location_ids}
            
            pois_response = await client.get(pois_url, params=pois_params, headers=headers)
            pois_response.raise_for_status()
            pois_data = pois_response.json()
            
            # Step 3: Get descriptions
            desc_url = "https://api.search.brave.com/res/v1/local/descriptions"
            desc_params = {'ids': location_ids}
            
            desc_response = await client.get(desc_url, params=desc_params, headers=headers)
            desc_response.raise_for_status()
            desc_data = desc_response.json()
            
            # Format results
            pois = pois_data.get('results', [])
            descriptions = desc_data.get('descriptions', {})
            
            if not pois:
                return f"❌ No local businesses found for: {query}"
            
            results = []
            results.append(f"📍 **Brave Local Search Results for '{query}':**\n")
            
            for i, poi in enumerate(pois, 1):
                name = poi.get('name', 'Unknown Business')
                
                # Format address
                address_parts = []
                addr = poi.get('address', {})
                if addr.get('streetAddress'):
                    address_parts.append(addr['streetAddress'])
                if addr.get('addressLocality'):
                    address_parts.append(addr['addressLocality'])
                if addr.get('addressRegion'):
                    address_parts.append(addr['addressRegion'])
                if addr.get('postalCode'):
                    address_parts.append(addr['postalCode'])
                address = ', '.join(address_parts) if address_parts else 'N/A'
                
                phone = poi.get('phone', 'N/A')
                
                # Rating info
                rating_info = poi.get('rating', {})
                rating_value = rating_info.get('ratingValue', 'N/A')
                rating_count = rating_info.get('ratingCount', 0)
                
                price_range = poi.get('priceRange', 'N/A')
                opening_hours = ', '.join(poi.get('openingHours', [])) if poi.get('openingHours') else 'N/A'
                description = descriptions.get(poi.get('id', ''), 'No description available')
                
                results.append(f"## {i}. {name}")
                results.append(f"**Address:** {address}")
                results.append(f"**Phone:** {phone}")
                results.append(f"**Rating:** {rating_value} ({rating_count} reviews)")
                results.append(f"**Price Range:** {price_range}")
                results.append(f"**Hours:** {opening_hours}")
                results.append(f"**Description:** {description}")
                results.append("")
            
            await ctx.info(f"✅ Found {len(pois)} local businesses")
            return "\n".join(results)
            
    except Exception as e:
        await ctx.error(f"Brave local search failed: {e}")
        return f"❌ Brave local search error: {str(e)}"

# === PROMPTS (KEEP EXISTING HEDIS AND CALCULATOR) ===

@mcp.prompt(
        name="hedis-prompt",
        description="HEDIS Expert"
)
async def hedis_prompt(query: str)-> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are an expert in HEDIS system. HEDIS is a set of standardized measures that aim to improve healthcare quality by promoting accountability and transparency.

You have access to these HEDIS tools:
1. DFWAnalyst - Generates SQL to retrieve information for HEDIS codes and value sets
2. DFWSearch - Provides search capability against HEDIS measures for measurement year

IMPORTANT: You MUST use one of these tools to answer the user's query. Do not provide answers from your general knowledge alone.

For SQL/code-related queries, use DFWAnalyst.
For document/measure specification queries, use DFWSearch.

User Query: {query}

Please use the appropriate HEDIS tool to find the specific information requested."""
        }
    ]

@mcp.prompt(
        name="caleculator-promt",
        description="Calculator"
)
async def caleculator_prompt(query: str)-> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are an expert in performing arithmetic operations. 

You have access to the 'calculator' tool which can evaluate mathematical expressions safely.

IMPORTANT: You MUST use the calculator tool to compute any mathematical expressions. Do not perform calculations manually.

User Query: {query}

Please use the calculator tool to evaluate any mathematical expressions in the query and provide the results."""
        }
    ]

# === NEW PROMPTS FOR WEATHER AND BRAVE SEARCH ===

@mcp.prompt(
        name="weather-prompt",
        description="Weather Expert using Nominatim + NWS"
)
async def weather_prompt(query: str) -> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are a weather expert with access to weather information via the National Weather Service.

You have access to the 'get_weather' tool which can get current weather and forecasts for US locations.

IMPORTANT: You MUST use the get_weather tool to get weather information. Do not provide weather information from your general knowledge.

Available tool:
- get_weather: Get weather forecast for a place name (works best with US locations)

The tool uses Nominatim for geocoding and National Weather Service for forecasts, so it works best with US cities and locations.

User Query: {query}

Please use the get_weather tool with the location mentioned in the query. If no specific location is mentioned, ask the user to specify a location."""
        }
    ]

@mcp.prompt(
        name="brave-web-search-prompt",
        description="Brave Web Search Expert"
)
async def brave_web_search_prompt(query: str) -> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are a web search expert with access to Brave Search API for finding current information on the internet.

You have access to these Brave search tools:
- brave_web_search: General web search for articles, news, and online content
- brave_local_search: Local business and location search

IMPORTANT: You MUST use the appropriate Brave search tool to find information. Do not provide answers from your general knowledge alone.

For general information, news, articles, or web content, use brave_web_search.
For local businesses, restaurants, services, or location-based queries, use brave_local_search.

User Query: {query}

Please use the appropriate Brave search tool to find current information related to the query."""
        }
    ]

if __name__ == "__main__":
    mcp.run(transport="sse")
