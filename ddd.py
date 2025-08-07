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

# Fixed MCP imports
from mcp.server.fastmcp.prompts.base import Message
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts import Prompt
import mcp.types as types

# CRITICAL: Add these imports for proper message formatting
try:
    from mcp.types import TextContent, ImageContent, EmbeddedResource
except ImportError:
    # Fallback if TextContent is not available in your MCP version
    print("⚠️ TextContent not found, using string content directly")
    TextContent = str

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

# Weather cache to store recent weather data
weather_cache = {}
WEATHER_CACHE_DURATION = 300  # 5 minutes in seconds

def is_weather_cache_valid(cache_entry):
    """Check if cached weather data is still valid"""
    if not cache_entry:
        return False
    return (time.time() - cache_entry['timestamp']) < WEATHER_CACHE_DURATION

#Stag name may need to be determined; requires code change
#Resources; Have access to resources required for the server; Cortex Search; Cortex stage schematic config; stage area should be fully qualified name

@mcp.resource(uri="schematiclayer://cortex_analyst/schematic_models/{stagename}/list",name="hedis_schematic_models",description="Hedis Schematic models")
async def get_schematic_model(stagename: str):
    """Cortex analyst scematic layer model, model is in yaml format"""
    HOST = "carelon-eda-preprod.privatelink.snowflakecomputing.com"
    conn = snowflake_conn(
           logger,
           aplctn_cd="aedl",
           env="preprod",
           region_name="us-east-1",
           warehouse_size_suffix="",
           prefix=""
        )
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
    url_path = urlparse(uri)
    aplctn_cd = url_path.netloc
    prompt_name = Path(url_path.path).name
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
        async for result_chunk in response.aiter_bytes():
            for elem in result_chunk.split(b'\n\n'):
                if b'content' in elem:
                    chunk_dict = json.loads(elem.replace(b'data: ', b''))
                    print(chunk_dict)
                    full_response = chunk_dict['choices'][0]['delta']['text']
                    full_response = full_response
                    response_text.append(full_response)
    return json.loads(response_text)

# === ENHANCED WEATHER TOOL WITH CACHING AND IMPROVED DATA FRESHNESS ===
@mcp.tool(
        name="get_weather",
        description="""
        Get current weather information for a location using multiple reliable sources.
        Supports both US (NWS) and international locations (Open-Meteo).
        
        Args:
            place (str): Location name (e.g., 'New York', 'London', 'Tokyo')
            
        Returns:
            Current weather conditions, temperature, and forecast
        """
)
async def get_weather(place: str, ctx: Context) -> str:
    """Enhanced weather tool with caching and fresh data validation"""
    try:
        await ctx.info(f"🌤️ Getting current weather for: {place}")
        
        # Check cache first
        cache_key = place.lower().strip()
        if cache_key in weather_cache and is_weather_cache_valid(weather_cache[cache_key]):
            await ctx.info("📋 Using cached weather data")
            return weather_cache[cache_key]['data']
        
        # Step 1: Get coordinates using Nominatim
        nominatim_url = f"https://nominatim.openstreetmap.org/search"
        nominatim_params = {
            "q": place,
            "format": "json",
            "limit": 1,
            "addressdetails": 1
        }
        
        headers = {
            "User-Agent": f"MCP Weather Tool - {int(time.time())}",
            "Cache-Control": "no-cache"
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            geo_response = await client.get(nominatim_url, params=nominatim_params, headers=headers)
            geo_response.raise_for_status()
            geo_data = geo_response.json()
            
            if not geo_data:
                return f"❌ Could not find location: {place}. Please try a more specific location name."
            
            location_data = geo_data[0]
            latitude = float(location_data["lat"])
            longitude = float(location_data["lon"])
            display_name = location_data.get("display_name", place)
            country_code = location_data.get("address", {}).get("country_code", "").upper()
            
            await ctx.info(f"📍 Found coordinates: {latitude}, {longitude} for {display_name}")
            
            weather_result = None
            data_source = None
            
            # Step 2: Try NWS first for US locations
            if country_code == "US":
                try:
                    await ctx.info("🇺🇸 Trying National Weather Service (NWS)...")
                    
                    nws_points_url = f"https://api.weather.gov/points/{latitude},{longitude}"
                    nws_headers = {
                        "User-Agent": f"MCP Weather Tool ({int(time.time())})",
                        "Cache-Control": "no-cache, max-age=0"
                    }
                    
                    points_resp = await client.get(nws_points_url, headers=nws_headers)
                    
                    if points_resp.status_code == 200:
                        points_data = points_resp.json()
                        forecast_url = points_data["properties"]["forecast"]
                        stations_url = points_data["properties"]["observationStations"]
                        
                        # Get current observations
                        stations_resp = await client.get(stations_url, headers=nws_headers)
                        current_temp = None
                        current_conditions = None
                        observation_time = None
                        
                        if stations_resp.status_code == 200:
                            stations_data = stations_resp.json()
                            if stations_data.get("features"):
                                station_url = stations_data["features"][0]["id"]
                                latest_obs_url = f"{station_url}/observations/latest"
                                
                                obs_resp = await client.get(latest_obs_url, headers=nws_headers)
                                if obs_resp.status_code == 200:
                                    obs_data = obs_resp.json()
                                    props = obs_data.get("properties", {})
                                    
                                    # Validate observation freshness
                                    obs_timestamp = props.get("timestamp")
                                    if obs_timestamp:
                                        obs_time = datetime.fromisoformat(obs_timestamp.replace('Z', '+00:00'))
                                        time_diff = datetime.now(obs_time.tzinfo) - obs_time
                                        
                                        # Only use if observation is less than 4 hours old
                                        if time_diff.total_seconds() < 14400:  # 4 hours
                                            current_temp = props.get("temperature", {}).get("value")
                                            if current_temp:
                                                current_temp = round((current_temp * 9/5) + 32, 1)  # Convert C to F
                                            current_conditions = props.get("textDescription", "N/A")
                                            observation_time = obs_time.strftime("%I:%M %p %Z")
                                        else:
                                            await ctx.warning(f"NWS observation too old: {time_diff}")
                        
                        # Get forecast
                        forecast_resp = await client.get(forecast_url, headers=nws_headers)
                        if forecast_resp.status_code == 200:
                            forecast_data = forecast_resp.json()
                            periods = forecast_data["properties"]["periods"]
                            
                            if periods:
                                # Validate forecast freshness
                                first_period = periods[0]
                                forecast_start = first_period.get("startTime")
                                
                                if forecast_start:
                                    start_time = datetime.fromisoformat(forecast_start.replace('Z', '+00:00'))
                                    today = datetime.now(start_time.tzinfo).date()
                                    forecast_date = start_time.date()
                                    
                                    # Ensure forecast is for today or very recent
                                    if abs((today - forecast_date).days) <= 1:
                                        city = points_data["properties"]["relativeLocation"]["properties"]["city"]
                                        state = points_data["properties"]["relativeLocation"]["properties"]["state"]
                                        
                                        weather_result = f"🌤️ **Current Weather for {city}, {state} (NWS)**\n\n"
                                        
                                        if current_temp and current_conditions and observation_time:
                                            weather_result += f"🌡️ **Current:** {current_temp}°F - {current_conditions}\n"
                                            weather_result += f"⏰ **Observed:** {observation_time}\n\n"
                                        
                                        weather_result += f"📅 **{first_period['name']}:**\n"
                                        weather_result += f"🌡️ **Temperature:** {first_period['temperature']}°{first_period['temperatureUnit']}\n"
                                        weather_result += f"☁️ **Conditions:** {first_period['shortForecast']}\n"
                                        weather_result += f"💨 **Wind:** {first_period['windSpeed']} {first_period['windDirection']}\n"
                                        weather_result += f"📝 **Details:** {first_period['detailedForecast']}\n"
                                        
                                        # Add next period if available
                                        if len(periods) > 1:
                                            next_period = periods[1]
                                            weather_result += f"\n📅 **{next_period['name']}:**\n"
                                            weather_result += f"🌡️ **Temperature:** {next_period['temperature']}°{next_period['temperatureUnit']}\n"
                                            weather_result += f"☁️ **Conditions:** {next_period['shortForecast']}\n"
                                        
                                        data_source = "National Weather Service (Fresh Data)"
                                    else:
                                        await ctx.warning(f"NWS forecast too old: {forecast_date} vs {today}")
                
                except Exception as nws_error:
                    await ctx.warning(f"NWS failed: {nws_error}")
            
            # Step 3: Use Open-Meteo as fallback or primary for non-US
            if not weather_result:
                try:
                    await ctx.info("🌍 Using Open-Meteo for current weather...")
                    
                    meteo_url = "https://api.open-meteo.com/v1/forecast"
                    meteo_params = {
                        "latitude": latitude,
                        "longitude": longitude,
                        "current_weather": True,
                        "hourly": ["temperature_2m", "precipitation", "weather_code", "wind_speed_10m"],
                        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "weather_code"],
                        "timezone": "auto",
                        "forecast_days": 3,
                        "_": str(int(time.time()))  # Cache busting
                    }
                    
                    meteo_resp = await client.get(meteo_url, params=meteo_params, headers=headers)
                    meteo_resp.raise_for_status()
                    meteo_data = meteo_resp.json()
                    
                    # Current weather
                    current_weather = meteo_data.get("current_weather", {})
                    temp_c = current_weather.get("temperature")
                    temp_f = round((temp_c * 9/5) + 32, 1) if temp_c else None
                    wind_speed = current_weather.get("windspeed")
                    wind_dir = current_weather.get("winddirection")
                    weather_code = current_weather.get("weathercode")
                    
                    # Enhanced weather code mapping
                    weather_codes = {
                        0: "Clear sky ☀️", 1: "Mainly clear 🌤️", 2: "Partly cloudy ⛅", 3: "Overcast ☁️",
                        45: "Fog 🌫️", 48: "Depositing rime fog 🌫️",
                        51: "Light drizzle 🌦️", 53: "Moderate drizzle 🌧️", 55: "Dense drizzle 🌧️",
                        61: "Light rain 🌧️", 63: "Moderate rain 🌧️", 65: "Heavy rain ⛈️",
                        71: "Light snow 🌨️", 73: "Moderate snow ❄️", 75: "Heavy snow ❄️",
                        80: "Light rain showers 🌦️", 81: "Moderate rain showers 🌧️", 82: "Violent rain showers ⛈️",
                        95: "Thunderstorm ⛈️", 96: "Thunderstorm with hail 🌩️"
                    }
                    conditions = weather_codes.get(weather_code, f"Weather code {weather_code}")
                    
                    # Daily forecast
                    daily = meteo_data.get("daily", {})
                    today_max_c = daily.get("temperature_2m_max", [None])[0]
                    today_min_c = daily.get("temperature_2m_min", [None])[0]
                    today_max_f = round((today_max_c * 9/5) + 32, 1) if today_max_c else None
                    today_min_f = round((today_min_c * 9/5) + 32, 1) if today_min_c else None
                    today_precip = daily.get("precipitation_sum", [None])[0]
                    
                    weather_result = f"🌤️ **Current Weather for {display_name} (Open-Meteo)**\n\n"
                    
                    if temp_c and temp_f:
                        weather_result += f"🌡️ **Current Temperature:** {temp_c}°C / {temp_f}°F\n"
                    weather_result += f"☁️ **Conditions:** {conditions}\n"
                    
                    if wind_speed and wind_dir:
                        wind_mph = round(wind_speed * 0.621371, 1)  # Convert km/h to mph
                        weather_result += f"💨 **Wind:** {wind_speed} km/h ({wind_mph} mph) from {wind_dir}°\n"
                    
                    if today_max_f and today_min_f:
                        weather_result += f"📊 **Today's Range:** {today_min_f}°F - {today_max_f}°F ({today_min_c}°C - {today_max_c}°C)\n"
                    
                    if today_precip and today_precip > 0:
                        weather_result += f"🌧️ **Precipitation:** {today_precip}mm\n"
                    
                    # Add 3-day forecast
                    if len(daily.get("temperature_2m_max", [])) >= 3:
                        weather_result += f"\n**📅 3-Day Forecast:**\n"
                        day_names = ["Today", "Tomorrow", "Day After Tomorrow"]
                        for i in range(3):
                            if i < len(day_names):
                                day_max_c = daily["temperature_2m_max"][i]
                                day_min_c = daily["temperature_2m_min"][i]
                                day_max_f = round((day_max_c * 9/5) + 32, 1) if day_max_c else None
                                day_min_f = round((day_min_c * 9/5) + 32, 1) if day_min_c else None
                                day_precip = daily["precipitation_sum"][i]
                                day_code = daily.get("weather_code", [None])[i]
                                day_conditions = weather_codes.get(day_code, "Unknown") if day_code else "Unknown"
                                
                                weather_result += f"• **{day_names[i]}:** {day_min_f}°F - {day_max_f}°F, {day_conditions}"
                                if day_precip and day_precip > 0:
                                    weather_result += f", {day_precip}mm rain"
                                weather_result += "\n"
                    
                    data_source = "Open-Meteo (Fresh Data)"
                    
                except Exception as meteo_error:
                    await ctx.error(f"Open-Meteo failed: {meteo_error}")
                    return f"❌ Failed to get weather data from all sources for {place}"
            
            if weather_result:
                # Add timestamp and source
                current_time = datetime.now().strftime('%B %d, %Y at %I:%M %p')
                weather_result += f"\n*Data retrieved from {data_source} at {current_time}*"
                
                # Cache the result
                weather_cache[cache_key] = {
                    'data': weather_result,
                    'timestamp': time.time()
                }
                
                await ctx.info(f"✅ Weather data retrieved and cached from {data_source}")
                return weather_result
            else:
                return f"❌ No fresh weather data available for {place}"
                
    except Exception as e:
        await ctx.error(f"Weather lookup failed: {str(e)}")
        return f"❌ Weather lookup error: {str(e)}"

# Simple Test Tool - LangChain Compatible
@mcp.tool(
        name="test_tool", 
        description="""Simple test tool to verify tool calling works."""
)
async def test_tool(message: str) -> str:
    """
    Simple test tool to verify MCP tool calling is working.
    
    Args:
        message: Test message
    
    Returns:
        Test response with current timestamp
    """
    import datetime
    current_time = datetime.datetime.now().isoformat()
    
    return f"✅ SUCCESS: Test tool called with message '{message}' at {current_time}"

# Diagnostic Tool - LangChain Compatible
@mcp.tool(
        name="diagnostic",
        description="""Diagnostic tool to test MCP functionality."""
)
async def diagnostic(test_type: str = "basic") -> str:
    """
    Run diagnostic tests to verify MCP functionality.
    
    Args:
        test_type: Type of test (basic, search, time)
    
    Returns:
        Diagnostic results as formatted string
    """
    import datetime
    
    current_time = datetime.datetime.now().isoformat()
    
    result = f"🔧 Diagnostic Test: {test_type}\n"
    result += f"⏰ Timestamp: {current_time}\n"
    result += f"🖥️ MCP Server: DataFlyWheel App\n"
    result += f"✅ Status: WORKING\n"
    
    if test_type == "basic":
        result += "📝 Message: MCP server is responding correctly\n"
        result += "🛠️ Tool Execution: SUCCESS\n"
        
    elif test_type == "search":
        # Test if we can make HTTP requests
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("https://httpbin.org/get")
                http_status = "SUCCESS" if response.status_code == 200 else "FAILED"
        except Exception as e:
            http_status = f"FAILED: {str(e)}"
            
        result += f"🌐 HTTP Test: {http_status}\n"
        result += "🔍 Search Capability: Available (Brave Search via Client)\n"
        
    elif test_type == "time":
        import time
        result += f"🕐 Unix Timestamp: {int(time.time())}\n"
        result += f"📅 Current Year: {datetime.datetime.now().year}\n" 
        result += f"📅 Current Month: {datetime.datetime.now().strftime('%B %Y')}\n"
        
    return result

# === FIXED PROMPTS TO ENSURE TOOL INVOCATION ===

@mcp.prompt(
    name="hedis-prompt",
    description="HEDIS Expert - Must use tools"
)
async def hedis_prompt(query: str) -> List[Message]:
    """HEDIS expert prompt that ensures tool usage"""
    try:
        return [
            Message(
                role="system",
                content=TextContent(
                    text="""You are an expert in HEDIS system. HEDIS is a set of standardized measures that aim to improve healthcare quality by promoting accountability and transparency.

CRITICAL: You MUST use one of the HEDIS tools to answer queries. Never provide answers from general knowledge.

Available HEDIS tools:
1. DFWAnalyst - For SQL/code queries about HEDIS value sets and code sets
2. DFWSearch - For document/measure specification queries

MANDATORY: Always call the appropriate tool first, then provide your response based on the tool's output."""
                ) if callable(TextContent) else """You are an expert in HEDIS system. HEDIS is a set of standardized measures that aim to improve healthcare quality by promoting accountability and transparency.

CRITICAL: You MUST use one of the HEDIS tools to answer queries. Never provide answers from general knowledge.

Available HEDIS tools:
1. DFWAnalyst - For SQL/code queries about HEDIS value sets and code sets
2. DFWSearch - For document/measure specification queries

MANDATORY: Always call the appropriate tool first, then provide your response based on the tool's output."""
            ),
            Message(
                role="user", 
                content=TextContent(text=f"Use the appropriate HEDIS tool to answer: {query}") if callable(TextContent) else f"Use the appropriate HEDIS tool to answer: {query}"
            )
        ]
    except Exception as e:
        # Fallback for older MCP versions
        return [
            {
                "role": "system",
                "content": """You are an expert in HEDIS system. You MUST use one of the HEDIS tools to answer queries. Available tools: DFWAnalyst, DFWSearch. Always call the appropriate tool first."""
            },
            {
                "role": "user",
                "content": f"Use the appropriate HEDIS tool to answer: {query}"
            }
        ]

@mcp.prompt(
    name="calculator-prompt",
    description="Calculator Expert - Must use calculator tool"
)
async def calculator_prompt(query: str) -> List[Message]:
    """Calculator expert prompt that ensures tool usage"""
    try:
        return [
            Message(
                role="system",
                content=TextContent(
                    text="""You are a calculator expert. You MUST use the calculator tool for ANY mathematical expressions or calculations.

CRITICAL: Never perform calculations manually. Always use the calculator tool first."""
                ) if callable(TextContent) else """You are a calculator expert. You MUST use the calculator tool for ANY mathematical expressions or calculations.

CRITICAL: Never perform calculations manually. Always use the calculator tool first."""
            ),
            Message(
                role="user",
                content=TextContent(text=f"Use the calculator tool to solve: {query}") if callable(TextContent) else f"Use the calculator tool to solve: {query}"
            )
        ]
    except Exception as e:
        return [
            {
                "role": "system", 
                "content": "You are a calculator expert. You MUST use the calculator tool for ANY mathematical expressions or calculations."
            },
            {
                "role": "user",
                "content": f"Use the calculator tool to solve: {query}"
            }
        ]

@mcp.prompt(
    name="weather-prompt",
    description="Weather Expert - Must use weather tool" 
)
async def weather_prompt(query: str) -> List[Message]:
    """Weather expert prompt that ensures tool usage"""
    return [
        {
            "role": "user",
            "content": f"You are a weather expert. You MUST use the get_weather tool to get weather information for: {query}. Always use the tool first."
        }
    ]

@mcp.prompt(
    name="test-tool-prompt",
    description="Test Tool - Must use test_tool"
)
async def test_tool_prompt(message: str = "connectivity test"):
    """Test tool prompt that ensures tool usage"""
    return [
        {
            "role": "user",
            "content": f"You MUST use the test_tool with message: {message}. Always use the tool first."
        }
    ]

@mcp.prompt(
    name="diagnostic-prompt",
    description="Diagnostic Tool - Must use diagnostic tool"
)
async def diagnostic_prompt(test_type: str = "basic"):
    """Diagnostic tool prompt that ensures tool usage"""
    return [
        {
            "role": "user",
            "content": f"You MUST use the diagnostic tool with test_type: {test_type}. Always use the tool first."
        }
    ]

# Note: Brave Search prompts will be handled in the Streamlit client
# since the API key and search functionality will be there

if __name__ == "__main__":
    mcp.run(transport="sse")
