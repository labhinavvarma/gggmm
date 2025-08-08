from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import httpx
from dataclasses import dataclass
from urllib.parse import urlparse
from pathlib import Path
import json
import snowflake.connector
import requests
import os
from loguru import logger
import logging
from snowflake.connector import SnowflakeConnection
from ReduceReuseRecycleGENAI.snowflake import snowflake_conn
from snowflake.connector.errors import DatabaseError
from snowflake.core import Root
from typing import Optional, List, Dict, Any
from fastapi import (
 HTTPException,
 status,
)
 
from mcp.server.fastmcp.prompts.base import Message
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts import Prompt
import mcp.types as types
from functools import partial

# SerpApi imports - with complete fallback
try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    try:
        import serpapi
        GoogleSearch = serpapi.GoogleSearch
        SERPAPI_AVAILABLE = True
    except (ImportError, AttributeError):
        try:
            from serpapi.google_search import GoogleSearch
            SERPAPI_AVAILABLE = True
        except ImportError:
            # Complete fallback - create dummy class
            class GoogleSearch:
                def __init__(self, params):
                    pass
                def get_dict(self):
                    return {"error": "SerpApi not available"}
            
            SERPAPI_AVAILABLE = False
            logger.warning("SerpApi library not found. Install with: pip install google-search-results")
 
 
# Create a named server
mcp = FastMCP("DataFlyWheel App")
 
@dataclass
class AppContext:
    conn : SnowflakeConnection
    db: str
    schema: str
    host: str
 
 
 
 
 
 
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
 
        Returns information utilizing HEDIS measure speficification documents.
 
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
        name="SerpApiSearch"
       ,description="""
        Performs web searches using SerpApi (Google search) with intelligent filtering for current information.
 
        Example inputs:
        "current US president 2025" (automatically uses news search + recent filters)
        "latest AI developments" (automatically filters for recent results)  
        "weather in New York" (standard search)
        "HEDIS measures 2024" (standard search)
        "breaking news today" (automatically uses news search)
 
        Automatically detects queries needing current information and applies:
        - Recent time filters (last month) for current events
        - Google News search for political/news queries  
        - Fresh results (no cache) for all searches
 
        Returns current web search results from Google with date information when available.
 
        Args:
               query (str): Search query text
               api_key (str): SerpApi API key
               location (str): Optional location for localized results (e.g., "Austin, TX")
               num_results (int): Number of results to return (default: 5, max: 10)
               search_type (str): Type of search - "web" (default) or "news"
        """
)
async def serpapi_search(query: str, api_key: str, location: str = "", num_results: int = 5, search_type: str = "web") -> str:
    """Tool to perform web searches using SerpApi, search query should be provided as 'query' parameter and api_key parameter"""
    
    if not SERPAPI_AVAILABLE:
        return "Error: SerpApi library is not installed. Please install with: pip install google-search-results"
    
    if not api_key or not api_key.strip():
        return "Error: SerpApi API key is required. Please provide a valid SERPAPI_API_KEY."
    
    # Limit number of results
    num_results = min(max(num_results, 1), 10)
    
    # Base parameters
    params = {
        "api_key": api_key.strip(),
        "engine": "google",
        "q": query,
        "num": num_results,
        "no_cache": "true",  # Force fresh results
        "gl": "us",          # Search from US
        "hl": "en",          # English language
    }
    
    # Add location if provided
    if location.strip():
        params["location"] = location.strip()
    
    # Detect if query needs current/recent information
    current_keywords = ["current", "latest", "recent", "today", "2025", "now", "president", "news"]
    needs_current_info = any(keyword in query.lower() for keyword in current_keywords)
    
    if needs_current_info:
        # For current events, use recent time filter
        params["tbs"] = "qdr:m"  # Last month results
        
        # For political/news queries, also try news search
        political_keywords = ["president", "election", "government", "politics", "news"]
        if any(keyword in query.lower() for keyword in political_keywords):
            params["tbm"] = "nws"  # News search for political queries
    
    try:
        search = GoogleSearch(params)
        data = search.get_dict()
        
        # Process organic search results if available
        results_key = "news_results" if params.get("tbm") == "nws" else "organic_results"
        
        if results_key in data and data[results_key]:
            formatted_results = []
            for i, result in enumerate(data.get(results_key, [])[:num_results], 1):
                title = result.get("title", "No title")
                link = result.get("link", "No link")
                snippet = result.get("snippet", "No snippet")
                
                # Include date for news results
                date_info = ""
                if "date" in result:
                    date_info = f" ({result['date']})"
                
                formatted_results.append(f"{i}. **{title}**{date_info}\n   Link: {link}\n   Summary: {snippet}\n")
            
            result_text = "\n".join(formatted_results) if formatted_results else "No search results found"
            
            # Add note about search type used
            search_note = ""
            if params.get("tbm") == "nws":
                search_note = "\n*Note: Used Google News search for current information*"
            elif params.get("tbs"):
                search_note = f"\n*Note: Filtered for recent results (last month)*"
                
            return result_text + search_note
        else:
            return "No search results found for the given query"
            
    except Exception as e:
        logger.error(f"SerpApi search error: {str(e)}")
        if "429" in str(e) or "rate limit" in str(e).lower():
            return "Error: Search rate limit exceeded. Please try again later."
        elif "401" in str(e) or "unauthorized" in str(e).lower():
            return "Error: Invalid SerpApi API key. Please check your SERPAPI_API_KEY."
        else:
            return f"Error performing search: {str(e)}"
 
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
 
@mcp.tool()
async def get_weather(place: str) -> str:
    """
    Get weather forecast for a place (e.g., 'New York') without needing an API key.
    """
    print(f" get_weather() called for location: {place}", flush=True)
 
    try:
        # Step 1: Get coordinates using Nominatim (no key needed)
        nominatim_url = f"https://nominatim.openstreetmap.org/search?q={place}&format=json&limit=1&countrycodes=us"
        response = requests.get(nominatim_url, headers={"User-Agent": "MCP Weather Tool"})
        response.raise_for_status()
        data = response.json()
 
        if not data:
            return f" Could not find location: {place}. Please try a more specific city name."
 
        latitude = data[0]["lat"]
        longitude = data[0]["lon"]
        display_name = data[0].get("display_name", place)
        print(f" Found coordinates: {latitude}, {longitude} for {display_name}", flush=True)
 
        # Step 2: Use NWS API to get forecast
        nws_url = f"https://api.weather.gov/points/{latitude},{longitude}"
        headers = {"User-Agent": "MCP Weather Tool"}
        points_resp = requests.get(nws_url, headers=headers)
       
        if points_resp.status_code == 404:
            return f" Weather service not available for {place}. The National Weather Service only covers US locations."
       
        points_resp.raise_for_status()
        points_data = points_resp.json()
 
        forecast_url = points_data["properties"]["forecast"]
        city = points_data["properties"]["relativeLocation"]["properties"]["city"]
        state = points_data["properties"]["relativeLocation"]["properties"]["state"]
 
        forecast_resp = requests.get(forecast_url, headers=headers)
        forecast_resp.raise_for_status()
        forecast_data = forecast_resp.json()
 
        period = forecast_data["properties"]["periods"][0]
        return (
            f" Weather for {city}, {state}:\n"
            f" - {period['name']}\n"
            f" - Temp: {period['temperature']}°{period['temperatureUnit']}\n"
            f" - Conditions: {period['shortForecast']}\n"
            f" - Wind: {period['windSpeed']} {period['windDirection']}\n"
            f" - Forecast: {period['detailedForecast']}"
        )
 
    except Exception as e:
        print(" Error:", str(e), flush=True)
        return f" Error fetching weather: {str(e)}"
 
@mcp.prompt(
        name="hedis-prompt",
        description="HEDIS Expert"
)
async def hedis_prompt(query: str)-> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are expert in HEDIS system, HEDIS is a set of standardized measures that aim to improve healthcare quality by promoting accountability and transparency.You are provided with below tools: 1) DFWAnalyst - Generates SQL to retrive information for hedis codes and value sets. 2) DFWSearch -  Provides search capability againest HEDIS measures for measurement year.You will respond with the results returned from right tool. {query}"""
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
            "content": f"""You are expert in performing arthametic operations.You are provided with the tool calculator to verify the results.You will respond with the results after verifying with the tool result. {query} """
        }
    ]

@mcp.prompt(
    name="weather-prompt",
    description="Weather Expert"
)
async def weather_prompt(query: str)-> List[Message]:
    """Weather expert who intakes the place as input and returns the present weather"""
    return [
        {
            "role": "user",
            "content": f"You are a weather expert. You have been provided with `get_weather` tool to get up to date weather information for: {query}. Always use the tool first."
        }
    ]

@mcp.prompt(
        name="serpapi-prompt",
        description="Web Search Expert with Current Information Focus"
)
async def serpapi_prompt(query: str)-> List[Message]:
    """Web search expert who performs internet searches using SerpApi with focus on current information"""
    return [
        {
            "role": "user", 
            "content": f"""You are a web search expert specializing in finding current, up-to-date information. You have access to the `SerpApiSearch` tool which automatically detects when queries need recent information and applies appropriate filters.

            IMPORTANT: When searching for current events, political information, or recent developments, formulate queries to be specific about the current timeframe:

            For political queries:
            - Instead of "who is the US president" → use "current US president 2025" or "US president January 2025"
            - Instead of "latest election results" → use "2024 US election results Trump"

            For current events:
            - Include current year: "AI developments 2025"
            - Use terms like "current", "latest", "recent", "today"
            - Be specific about timeframes when relevant

            The SerpApiSearch tool will automatically:
            - Apply recent time filters for current event queries
            - Use Google News search for political/news topics
            - Force fresh results (no cached data)
            - Return results with date information when available

            Parameters available:
            - query: Your search terms (be specific for current info)
            - api_key: Automatically provided
            - location: Optional for location-specific results  
            - num_results: Number of results (default 5)

            Your task: Search for current information about: {query}

            Always prioritize recent, authoritative sources for current events and provide comprehensive, up-to-date information."""
        }
    ]
 
if __name__ == "__main__":
    mcp.run(transport="sse")
