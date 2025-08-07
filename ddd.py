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
import json
from typing import Any, Dict, List, Optional, Union
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

# Constants
NWS_API_BASE = "https://api.weather.gov"

def select_nws_period_for_today(forecast_data):
    """
    Given NWS forecast_data, select the period whose startTime matches today's date (UTC),
    falling back to the first period if not found.
    """
    from datetime import datetime
    today = datetime.utcnow().date().isoformat()
    periods = forecast_data["properties"]["periods"]
    for p in periods:
        if p["startTime"].startswith(today):
            return p
    return periods[0]


def get_nws_current_observation(latitude, longitude, location_name="Location"):
    """
    Fetch the latest current weather observation from the NWS API for the given lat/lon.
    Returns a string with the temperature (°C) and text description.
    """
    import requests
    headers = {"User-Agent": "weather-app/1.0"}
    # 1. Get the points endpoint
    points_url = f"https://api.weather.gov/points/{latitude},{longitude}"
    points_data = requests.get(points_url, headers=headers).json()
    stations_url = points_data["properties"]["observationStations"]
    # 2. Fetch the first station
    stations = requests.get(stations_url, headers=headers).json()
    station_id = stations["features"][0]["properties"]["stationIdentifier"]
    # 3. Get latest observation
    obs_url = f"https://api.weather.gov/stations/{station_id}/observations/latest"
    obs = requests.get(obs_url, headers=headers).json()
    temp = obs["properties"]["temperature"]["value"]
    desc = obs["properties"]["textDescription"]
    return f"Current at {location_name}: {temp}°C, {desc}"


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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_server.log')
    ]
)
logger = logging.getLogger('mcp_server')

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

# === ENHANCED WIKIPEDIA MCP TOOL (CURRENT DATA) ===
@mcp.tool(
        name="wikipedia_search",
        description="""
        Search Wikipedia for current information on any topic with enhanced content retrieval.
        Example inputs:
        "artificial intelligence"
        "World War II"
        "Python programming language"
        Returns current Wikipedia article content and summary.
        Args:
             query (str): Search query for Wikipedia
             max_results (int): Maximum number of results to return (default: 3)
        """
)
async def wikipedia_search(query: str, ctx: Context, max_results: int = 3) -> str:
    """Tool to search Wikipedia for current information, query should be passed as 'query' input parameter"""
    try:
        await rate_limiter.acquire()
        await ctx.info(f"🔍 Searching Wikipedia for current data: {query}")
        
        current_timestamp = int(time.time())
        headers = {
            "User-Agent": f"MCP Wikipedia Client (mcp-server@example.com) - {current_timestamp}",
            "Cache-Control": "no-cache, max-age=0",
            "Accept": "application/json"
        }
        
        # Search Wikipedia API with cache busting
        search_url = "https://en.wikipedia.org/api/rest_v1/page/search"
        search_params = {
            "q": query,
            "limit": max_results,
            "_": current_timestamp  # Cache busting parameter
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            search_response = await client.get(search_url, params=search_params, headers=headers)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            if not search_data.get('pages'):
                return f"❌ No Wikipedia results found for: {query}"
            
            results = []
            results.append(f"📖 **Wikipedia Search Results for '{query}' (Current Data):**\n")
            
            for i, page in enumerate(search_data['pages'][:max_results], 1):
                title = page.get('title', 'Unknown')
                description = page.get('description', 'No description available')
                
                await ctx.info(f"📄 Fetching full content for: {title}")
                
                # Get full article content with multiple API calls for comprehensive data
                page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                
                try:
                    # Get detailed summary
                    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
                    summary_response = await client.get(summary_url, headers=headers, 
                                                      params={"_": current_timestamp})
                    
                    if summary_response.status_code == 200:
                        summary_data = summary_response.json()
                        extract = summary_data.get('extract', '')
                        last_modified = summary_data.get('timestamp', 'Unknown')
                        
                        # Get additional content sections
                        content_url = f"https://en.wikipedia.org/api/rest_v1/page/mobile-sections/{title.replace(' ', '_')}"
                        content_response = await client.get(content_url, headers=headers,
                                                          params={"_": current_timestamp})
                        
                        additional_content = ""
                        if content_response.status_code == 200:
                            content_data = content_response.json()
                            sections = content_data.get('sections', [])
                            # Get first few sections for more comprehensive content
                            for section in sections[:3]:
                                if section.get('text'):
                                    # Clean HTML tags
                                    section_text = re.sub(r'<[^>]+>', '', section.get('text', ''))
                                    additional_content += f" {section_text}"
                        
                        # Combine extract with additional content
                        full_content = extract
                        if additional_content:
                            full_content += f"\n\nAdditional Information: {additional_content[:1000]}..."
                        
                    else:
                        full_content = f"Unable to fetch detailed content for {title}"
                        last_modified = "Unknown"
                        
                except Exception as content_error:
                    await ctx.warning(f"Content fetch failed for {title}: {content_error}")
                    full_content = description
                    last_modified = "Unknown"
                
                # Format result with timestamp info
                results.append(f"## {i}. {title}")
                results.append(f"**Description:** {description}")
                results.append(f"**URL:** {page_url}")
                if last_modified != "Unknown":
                    try:
                        # Parse and format timestamp
                        mod_time = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                        results.append(f"**Last Modified:** {mod_time.strftime('%B %d, %Y')}")
                    except:
                        results.append(f"**Last Modified:** {last_modified}")
                results.append(f"**Content:** {full_content}")
                results.append("")
            
            await ctx.info(f"✅ Wikipedia search completed: {len(search_data['pages'])} current results found")
            return "\n".join(results)
            
    except Exception as e:
        await ctx.error(f"Wikipedia search failed: {str(e)}")
        return f"❌ Wikipedia search error: {str(e)}"

# === ENHANCED DUCKDUCKGO TOOL (READS LINKS FOR LATEST DATA) ===
@mcp.tool(
        name="duckduckgo_search",
        description="""
        Search the web using DuckDuckGo and fetch actual content from top results for latest information.
        This tool searches, finds links, and reads the actual webpage content to provide current data.
        Example inputs:
        "latest news about AI 2024"
        "current developments in renewable energy"
        "recent space exploration missions"
        Returns current information extracted from actual web pages.
        Args:
             query (str): Search query for DuckDuckGo
             max_results (int): Maximum number of links to read and analyze (default: 3)
        """
)
async def duckduckgo_search(query: str, ctx: Context, max_results: int = 3) -> str:
    """Tool to search web and read actual content from links for latest information"""
    try:
        await rate_limiter.acquire()
        await ctx.info(f"🦆 Searching DuckDuckGo and reading content for: {query}")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        # Enhanced search with current year for recent results
        current_year = datetime.now().year
        enhanced_query = f"{query} {current_year}"
        
        results = []
        results.append(f"🦆 **DuckDuckGo Web Search Results with Content Analysis for '{query}':**\n")
        
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            
            # Step 1: Get search results from DuckDuckGo
            search_urls = []
            
            # Try DuckDuckGo HTML search for actual links
            try:
                html_search_url = "https://html.duckduckgo.com/html/"
                html_params = {
                    "q": enhanced_query,
                    "s": "0",
                    "dc": str(max_results * 2),  # Get more results to filter from
                    "v": "l"
                }
                
                await ctx.info("🔍 Getting search results from DuckDuckGo...")
                html_response = await client.get(html_search_url, params=html_params, headers=headers)
                
                if html_response.status_code == 200:
                    html_content = html_response.text
                    
                    # Enhanced regex patterns for extracting URLs
                    patterns = [
                        r'<a[^>]+href="(https?://[^"]+)"[^>]*class="[^"]*result[^"]*"[^>]*>([^<]+)</a>',
                        r'<a[^>]+href="(https?://[^"]+)"[^>]*>([^<]+)</a>[^<]*<span[^>]*class="[^"]*snippet[^"]*"[^>]*>([^<]+)</span>',
                        r'<a[^>]+href="(https?://[^"]+)"[^>]*title="([^"]+)"[^>]*>',
                        r'href="(https?://[^"]+)"[^>]*>([^<]+)</a>'
                    ]
                    
                    found_urls = []
                    for pattern in patterns:
                        matches = re.findall(pattern, html_content, re.IGNORECASE | re.DOTALL)
                        
                        for match in matches:
                            if len(match) >= 2:
                                url = match[0].strip()
                                title = match[1].strip()
                                
                                # Filter for good URLs and avoid duplicates
                                if (url.startswith('http') and 
                                    url not in [item['url'] for item in found_urls] and
                                    not any(skip in url.lower() for skip in ['duckduckgo.com', 'youtube.com/redirect', 'facebook.com', 'twitter.com/home']) and
                                    len(title) > 5):
                                    
                                    found_urls.append({
                                        'url': url,
                                        'title': title
                                    })
                                    
                                    if len(found_urls) >= max_results:
                                        break
                        
                        if len(found_urls) >= max_results:
                            break
                    
                    search_urls = found_urls[:max_results]
                    await ctx.info(f"📊 Found {len(search_urls)} URLs to analyze")
                
            except Exception as search_error:
                await ctx.warning(f"Search step failed: {search_error}")
                return f"❌ Failed to get search results: {search_error}"
            
            # Step 2: Fetch and analyze content from each URL
            if not search_urls:
                return f"❌ No valid URLs found for query: {query}"
            
            content_results = []
            
            for i, url_data in enumerate(search_urls, 1):
                url = url_data['url']
                title = url_data['title']
                
                await ctx.info(f"📄 Reading content from {i}/{len(search_urls)}: {title[:50]}...")
                
                try:
                    # Fetch webpage content
                    page_response = await client.get(url, headers=headers, timeout=15.0)
                    
                    if page_response.status_code == 200:
                        page_content = page_response.text
                        
                        # Extract meaningful text content
                        # Remove scripts, styles, and other non-content tags
                        clean_content = re.sub(r'<script[^>]*>.*?</script>', '', page_content, flags=re.DOTALL)
                        clean_content = re.sub(r'<style[^>]*>.*?</style>', '', clean_content, flags=re.DOTALL)
                        clean_content = re.sub(r'<[^>]+>', ' ', clean_content)
                        clean_content = re.sub(r'\s+', ' ', clean_content).strip()
                        
                        # Extract the most relevant paragraphs (usually the first few contain the main content)
                        paragraphs = [p.strip() for p in clean_content.split('\n') if len(p.strip()) > 100]
                        
                        # Get the best content (first few substantial paragraphs)
                        extracted_content = ""
                        word_count = 0
                        for para in paragraphs[:10]:  # Check first 10 paragraphs
                            if word_count < 500:  # Limit to ~500 words per article
                                extracted_content += para + " "
                                word_count += len(para.split())
                            else:
                                break
                        
                        if len(extracted_content.strip()) > 200:
                            # Create summary of the content
                            content_summary = extracted_content[:800] + "..." if len(extracted_content) > 800 else extracted_content
                            
                            content_results.append({
                                'title': title,
                                'url': url,
                                'content': content_summary,
                                'word_count': len(extracted_content.split()),
                                'status': 'success'
                            })
                            
                            await ctx.info(f"✅ Successfully extracted {len(extracted_content.split())} words")
                        else:
                            await ctx.warning(f"⚠️ Insufficient content extracted from {url}")
                            content_results.append({
                                'title': title,
                                'url': url,
                                'content': 'Content too short or inaccessible',
                                'status': 'insufficient'
                            })
                    else:
                        await ctx.warning(f"⚠️ HTTP {page_response.status_code} for {url}")
                        content_results.append({
                            'title': title,
                            'url': url,
                            'content': f'Failed to access (HTTP {page_response.status_code})',
                            'status': 'failed'
                        })
                        
                except Exception as fetch_error:
                    await ctx.warning(f"⚠️ Failed to read {url}: {fetch_error}")
                    content_results.append({
                        'title': title,
                        'url': url,
                        'content': f'Error reading content: {str(fetch_error)}',
                        'status': 'error'
                    })
            
            # Step 3: Format results with actual content
            if content_results:
                results.append(f"📊 **Analyzed {len(content_results)} web sources for latest information:**\n")
                
                successful_results = [r for r in content_results if r['status'] == 'success']
                
                for i, result in enumerate(successful_results, 1):
                    results.append(f"## Source {i}: {result['title']}")
                    results.append(f"**URL:** {result['url']}")
                    results.append(f"**Content ({result.get('word_count', 'Unknown')} words):**")
                    results.append(result['content'])
                    results.append("")
                
                # Add summary of failed attempts
                failed_results = [r for r in content_results if r['status'] != 'success']
                if failed_results:
                    results.append(f"⚠️ **{len(failed_results)} sources could not be fully analyzed:**")
                    for result in failed_results:
                        results.append(f"• {result['title']}: {result['content']}")
                    results.append("")
                
                current_time = datetime.now().strftime('%B %d, %Y at %I:%M %p')
                results.append(f"*Content analysis completed at: {current_time}*")
                
                await ctx.info(f"✅ Web search and content analysis completed: {len(successful_results)} sources analyzed")
                return "\n".join(results)
            else:
                return f"❌ No content could be extracted from search results for: {query}"
            
    except Exception as e:
        await ctx.error(f"DuckDuckGo search and content analysis failed: {str(e)}")
        return f"❌ DuckDuckGo search error: {str(e)}"

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
        result += "🔍 Search Capability: Available\n"
        
    elif test_type == "time":
        import time
        result += f"🕐 Unix Timestamp: {int(time.time())}\n"
        result += f"📅 Current Year: {datetime.datetime.now().year}\n" 
        result += f"📅 Current Month: {datetime.datetime.now().strftime('%B %Y')}\n"
        
    return result

# === ENHANCED OPEN-METEO WEATHER TOOL ===
@mcp.tool(
    name="open_meteo_weather",
    description="""
    Get current weather (temperature, wind, precipitation) from Open-Meteo (no API key required).
    Args:
      latitude (float): Latitude of the location.
      longitude (float): Longitude of the location.
    Returns:
      A human-readable summary of current conditions.
    """
)
async def open_meteo_weather(ctx: Context, latitude: float, longitude: float) -> str:
    """Get current weather from Open-Meteo API"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": True,
        "hourly": ["temperature_2m", "precipitation", "weather_code"],
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "auto",
        "forecast_days": 3
    }
    try:
        await ctx.info(f"🌤️ Fetching current weather for {latitude}, {longitude}")
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        
        # Current weather
        cw = data.get("current_weather", {})
        temp = cw.get("temperature")
        wind = cw.get("windspeed")
        wind_dir = cw.get("winddirection")
        weather_code = cw.get("weathercode")
        
        # Enhanced weather code mapping
        codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Light rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Light snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Light rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            95: "Thunderstorm", 96: "Thunderstorm with hail"
        }
        desc = codes.get(weather_code, f"Weather code {weather_code}")
        
        # Get daily forecast
        daily = data.get("daily", {})
        today_max = daily.get("temperature_2m_max", [None])[0]
        today_min = daily.get("temperature_2m_min", [None])[0]
        today_precip = daily.get("precipitation_sum", [None])[0]
        
        result = f"🌤️ **Current Weather Report**\n\n"
        result += f"📍 **Location:** {latitude}°, {longitude}°\n"
        result += f"🌡️ **Current Temperature:** {temp}°C\n"
        result += f"☁️ **Conditions:** {desc}\n"
        result += f"💨 **Wind:** {wind} km/h from {wind_dir}°\n"
        
        if today_max and today_min:
            result += f"📊 **Today's Range:** {today_min}°C - {today_max}°C\n"
        
        if today_precip and today_precip > 0:
            result += f"🌧️ **Precipitation:** {today_precip}mm\n"
        
        # Add 3-day forecast
        if len(daily.get("temperature_2m_max", [])) >= 3:
            result += f"\n**📅 3-Day Forecast:**\n"
            for i in range(3):
                day_max = daily["temperature_2m_max"][i]
                day_min = daily["temperature_2m_min"][i]
                day_precip = daily["precipitation_sum"][i]
                day_name = ["Today", "Tomorrow", "Day After"][i]
                result += f"• **{day_name}:** {day_min}°C - {day_max}°C"
                if day_precip > 0:
                    result += f", {day_precip}mm rain"
                result += "\n"
        
        await ctx.info("✅ Weather data retrieved successfully")
        return result
        
    except Exception as e:
        await ctx.error(f"Open-Meteo request failed: {e}")
        return f"❌ Failed to fetch weather from Open-Meteo: {e}"

@mcp.tool(
    name="get_weather",
    description="""
    Get current weather conditions for a location using NWS (real-time observation, then forecast) with fallback to Open-Meteo. Avoids stale data. Args: latitude (float), longitude (float), location_name (str, optional). Returns: human-readable weather summary string.
    """
)
async def get_weather(ctx: Context, latitude: float, longitude: float, location_name: str = "Location") -> str:
    """Get current weather using NWS (observation, then forecast) with fallback to Open-Meteo."""
    import requests
    from datetime import datetime
    headers = {"User-Agent": "weather-app/1.0"}
    # Try NWS real-time observation
    try:
        obs_str = get_nws_current_observation(latitude, longitude, location_name)
        if obs_str and "Current at" in obs_str:
            await ctx.info("✅ NWS real-time observation used")
            return obs_str
    except Exception as e:
        await ctx.error(f"NWS observation fetch failed: {e}")
    # Try NWS forecast
    try:
        # 1. Get forecast URL
        points_url = f"https://api.weather.gov/points/{latitude},{longitude}"
        points_data = requests.get(points_url, headers=headers, timeout=8).json()
        forecast_url = points_data["properties"]["forecast"]
        forecast_data = requests.get(forecast_url, headers=headers, timeout=8).json()
        period = select_nws_period_for_today(forecast_data)
        if period:
            start_time = period.get("startTime", "")
            detailed = period.get("detailedForecast") or period.get("shortForecast")
            temp = period.get("temperature")
            unit = period.get("temperatureUnit", "°C")
            date_str = start_time.split("T")[0] if start_time else ""
            result = f"NWS Forecast for {location_name} ({date_str}): {temp}{unit}, {detailed}"
            await ctx.info("✅ NWS forecast used")
            return result
    except Exception as e:
        await ctx.error(f"NWS forecast fetch failed: {e}")
    # Fallback to Open-Meteo
    try:
        result = await open_meteo_weather(ctx, latitude, longitude)
        await ctx.info("✅ Open-Meteo fallback used")
        return result
    except Exception as e:
        await ctx.error(f"All weather sources failed: {e}")
        return f"❌ Failed to fetch weather from all sources: {e}"

# === FIXED PROMPTS TO ENSURE TOOL INVOCATION ===

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

@mcp.prompt(
        name="wikipedia-search-prompt",
        description="Wikipedia Search Expert"
)
async def wikipedia_search_prompt(query: str) -> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are a Wikipedia search expert specialized in finding accurate, encyclopedic information from Wikipedia.

You have access to the 'wikipedia_search' tool which can search Wikipedia articles for comprehensive information.

IMPORTANT: You MUST use the wikipedia_search tool to find information. Do not provide answers from your general knowledge alone.

Available tool:
- wikipedia_search: Search Wikipedia for detailed information on any topic

User Query: {query}

Please use the wikipedia_search tool to find relevant Wikipedia information and provide a comprehensive summary based on the search results."""
        }
    ]

@mcp.prompt(
        name="duckduckgo-search-prompt",
        description="DuckDuckGo Web Search Expert"
)
async def duckduckgo_search_prompt(query: str) -> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are a web search expert specialized in finding current, relevant information from the internet using DuckDuckGo.

You have access to the 'duckduckgo_search' tool which searches the web for current information.

IMPORTANT: You MUST use the duckduckgo_search tool to find current web information. Do not provide answers from your general knowledge alone.

Available tool:
- duckduckgo_search: Search the web using DuckDuckGo for current information

User Query: {query}

Please use the duckduckgo_search tool to find relevant current web information and provide insights from the search results. Focus on recent developments and current information."""
        }
    ]

@mcp.prompt(
        name="weather-prompt",
        description="Weather Information Expert using NWS and Open-Meteo"
)
async def weather_prompt(query: str) -> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are a weather information expert with access to multiple weather data sources.

You have access to the 'get_weather' tool which requires latitude and longitude coordinates to provide current weather conditions and forecasts. This tool will first try to get real-time data from the National Weather Service (NWS), then fall back to NWS forecasts, and finally to Open-Meteo if needed.

IMPORTANT: You MUST use the get_weather tool to get weather information. Do not provide weather information from your general knowledge.

Available tool:
- get_weather: Get current weather conditions using NWS (real-time observation, then forecast) with fallback to Open-Meteo. Requires latitude and longitude coordinates.

Common city coordinates for reference:
- Richmond, VA: 37.5407, -77.4360
- Atlanta, GA: 33.7490, -84.3880
- New York, NY: 40.7128, -74.0060
- Denver, CO: 39.7392, -104.9903
- Miami, FL: 25.7617, -80.1918

User Query: {query}

If the query mentions a city, extract or look up the coordinates and use the get_weather tool. If coordinates aren't clear from the query, ask the user to provide them or suggest coordinates for the nearest major city."""
        }
    ]

# Simplified Test Tool Prompt
@mcp.prompt(
        name="test-tool-prompt",
        description="Test Tool Caller"
)
async def test_tool_prompt(message: str = "connectivity test") -> List[Message]:
    """
    Simplified test prompt for ChatSnowflakeCortex.
    
    Args:
        message: Test message to send
    
    Returns:
        Simple formatted prompt messages
    """
    return [
        {
            "role": "user",
            "content": f"""Please test the tool system by calling the test_tool with message: "{message}"

IMPORTANT: You MUST use the test_tool to respond.

Use the test_tool now with the message: {message}"""
        }
    ]

# Simplified Diagnostic Prompt  
@mcp.prompt(
        name="diagnostic-prompt",
        description="Diagnostic Tool Caller"
)
async def diagnostic_prompt(test_type: str = "basic") -> List[Message]:
    """
    Simplified diagnostic prompt for ChatSnowflakeCortex.
    
    Args:
        test_type: Type of diagnostic test
    
    Returns:
        Simple formatted prompt messages
    """
    return [
        {
            "role": "user",
            "content": f"""Please run a diagnostic test of type: "{test_type}"

IMPORTANT: You MUST use the diagnostic tool to run the test.

Use the diagnostic tool now with test_type: {test_type}"""
        }
    ]

def log_tool_invocation(func):
    """Decorator to log tool invocations and their responses"""
    async def wrapper(*args, **kwargs):
        ctx = None
        for arg in args:
            if hasattr(arg, 'info') and hasattr(arg, 'error'):  # Check if it's a Context object
                ctx = arg
                break
        
        tool_name = func.__name__
        logger.info(f"🔧 TOOL INVOCATION: {tool_name} with args: {json.dumps(kwargs, default=str)}")
        
        try:
            result = await func(*args, **kwargs)
            logger.info(f"✅ TOOL RESULT ({tool_name}): {json.dumps(result, default=str)[:500]}...")
            return result
        except Exception as e:
            logger.error(f"❌ TOOL ERROR ({tool_name}): {str(e)}", exc_info=True)
            if ctx:
                await ctx.error(f"Error in {tool_name}: {str(e)}")
            raise
    
    # Copy the MCP tool attributes to the wrapper
    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    wrapper.__module__ = func.__module__
    wrapper.__doc__ = func.__doc__
    wrapper.__annotations__ = func.__annotations__
    
    # Copy MCP tool attributes if they exist
    if hasattr(func, '_mcp_tool'):
        wrapper._mcp_tool = func._mcp_tool
    
    return wrapper

# Apply the logging decorator to all MCP tools
for name, func in list(globals().items()):
    if hasattr(func, '_mcp_tool'):
        globals()[name] = log_tool_invocation(func)

if __name__ == "__main__":
    logger.info("Starting MCP server with enhanced logging...")
    mcp.run(transport="sse")
