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

# Constants
NWS_API_BASE = "https://api.weather.gov"

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

# === WIKIPEDIA MCP TOOL ===
@mcp.tool(
        name="wikipedia_search",
        description="""
        Search Wikipedia for information on any topic.
        Example inputs:
        "artificial intelligence"
        "World War II"
        "Python programming language"
        Returns Wikipedia article content and summary.
        Args:
             query (str): Search query for Wikipedia
             max_results (int): Maximum number of results to return (default: 3)
        """
)
async def wikipedia_search(query: str, ctx: Context, max_results: int = 3) -> str:
    """Tool to search Wikipedia for information, query should be passed as 'query' input parameter"""
    try:
        await rate_limiter.acquire()
        await ctx.info(f"🔍 Searching Wikipedia for: {query}")
        
        headers = {
            "User-Agent": "MCP Wikipedia Client (mcp-server@example.com)"
        }
        
        # Search Wikipedia API
        search_url = "https://en.wikipedia.org/api/rest_v1/page/search"
        search_params = {
            "q": query,
            "limit": max_results
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            search_response = await client.get(search_url, params=search_params, headers=headers)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            if not search_data.get('pages'):
                return f"❌ No Wikipedia results found for: {query}"
            
            results = []
            results.append(f"📖 Wikipedia Search Results for '{query}':\n")
            
            for i, page in enumerate(search_data['pages'][:max_results], 1):
                title = page.get('title', 'Unknown')
                description = page.get('description', 'No description available')
                excerpt = page.get('excerpt', 'No excerpt available')
                
                # Get full article content
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
                
                results.append(f"## {i}. {title}")
                results.append(f"**Description:** {description}")
                results.append(f"**URL:** {page_url}")
                results.append(f"**Content:** {extract}")
                results.append("")
            
            await ctx.info(f"✅ Wikipedia search completed: {len(search_data['pages'])} results found")
            return "\n".join(results)
            
    except Exception as e:
        await ctx.error(f"Wikipedia search failed: {str(e)}")
        return f"❌ Wikipedia search error: {str(e)}"

# === ENHANCED DUCKDUCKGO MCP TOOL (FIXED FOR CURRENT RESULTS) ===
@mcp.tool(
        name="duckduckgo_search",
        description="""
        Search the web using DuckDuckGo search engine for current information.
        Example inputs:
        "latest news about AI 2024"
        "current weather in New York"
        "recent developments in renewable energy"
        Returns current web search results with links and descriptions.
        Args:
             query (str): Search query for DuckDuckGo
             max_results (int): Maximum number of results to return (default: 10)
        """
)
async def duckduckgo_search(query: str, ctx: Context, max_results: int = 10) -> str:
    """Tool to search the web using DuckDuckGo for current information, query should be passed as 'query' input parameter"""
    try:
        await rate_limiter.acquire()
        await ctx.info(f"🦆 Searching DuckDuckGo for current information: {query}")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        # Enhanced search - try multiple approaches for better current results
        results = []
        results.append(f"🦆 DuckDuckGo Search Results for '{query}':\n")
        
        # Method 1: Try DuckDuckGo Instant Answer API first
        search_url = "https://api.duckduckgo.com/"
        search_params = {
            "q": query + " " + str(datetime.now().year),  # Add current year for more recent results
            "format": "json",
            "no_redirect": "1",
            "no_html": "1",
            "skip_disambig": "1"
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.get(search_url, params=search_params, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                # Abstract/Definition
                if data.get('Abstract'):
                    results.append(f"**Definition:** {data['Abstract']}")
                    if data.get('AbstractURL'):
                        results.append(f"**Source:** {data['AbstractURL']}")
                    results.append("")
                
                # Answer (for factual queries)
                if data.get('Answer'):
                    results.append(f"**Direct Answer:** {data['Answer']}")
                    if data.get('AnswerType'):
                        results.append(f"**Type:** {data['AnswerType']}")
                    results.append("")
                
                # Related Topics
                if data.get('RelatedTopics'):
                    results.append("**Related Information:**")
                    for i, topic in enumerate(data['RelatedTopics'][:max_results//2], 1):
                        if isinstance(topic, dict) and topic.get('Text'):
                            results.append(f"{i}. {topic['Text']}")
                            if topic.get('FirstURL'):
                                results.append(f"   Source: {topic['FirstURL']}")
                    results.append("")
                
                # Results from web search
                if data.get('Results'):
                    results.append("**Web Results:**")
                    for i, result in enumerate(data['Results'][:max_results//2], 1):
                        if result.get('Text') and result.get('FirstURL'):
                            results.append(f"{i}. {result['Text']}")
                            results.append(f"   URL: {result['FirstURL']}")
                    results.append("")
                
            except Exception as e:
                await ctx.warning(f"Instant API failed: {str(e)}, trying HTML search...")
            
            # Method 2: HTML Search for more comprehensive results
            await ctx.info("🔍 Searching HTML interface for comprehensive results...")
            
            try:
                # Use DuckDuckGo HTML search with current year
                html_search_url = "https://html.duckduckgo.com/html/"
                current_year = datetime.now().year
                enhanced_query = f"{query} {current_year} OR {current_year-1}"  # Include current and previous year
                html_params = {
                    "q": enhanced_query,
                    "s": "0",  # Start from first result
                    "dc": str(max_results),  # Number of results
                    "v": "l",  # Lite version
                    "api": "d.js",
                    "o": "json"
                }
                
                html_response = await client.get(html_search_url, params=html_params, headers=headers)
                if html_response.status_code == 200:
                    html_content = html_response.text
                    
                    # Enhanced regex patterns for better extraction
                    patterns = [
                        # Pattern for result links with titles
                        r'<a[^>]+href="([^"]+)"[^>]*class="[^"]*result[^"]*"[^>]*>([^<]+)</a>',
                        r'<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>[^<]*<span[^>]*class="[^"]*snippet[^"]*"[^>]*>([^<]+)</span>',
                        # Fallback pattern
                        r'<a[^>]+href="(https?://[^"]+)"[^>]*>([^<]+)</a>'
                    ]
                    
                    web_results = []
                    urls_seen = set()
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, html_content, re.IGNORECASE | re.DOTALL)
                        
                        for match in matches:
                            if len(match) >= 2:
                                url = match[0].strip()
                                title = match[1].strip()
                                description = match[2].strip() if len(match) > 2 else ""
                                
                                # Filter for relevant URLs and avoid duplicates
                                if (url.startswith('http') and 
                                    url not in urls_seen and 
                                    len(title) > 10 and 
                                    not url.startswith('https://duckduckgo.com')):
                                    
                                    urls_seen.add(url)
                                    result_text = f"• **{title}**"
                                    if description:
                                        result_text += f"\n  Description: {description}"
                                    result_text += f"\n  URL: {url}"
                                    web_results.append(result_text)
                                    
                                    if len(web_results) >= max_results:
                                        break
                        
                        if len(web_results) >= max_results:
                            break
                    
                    if web_results:
                        if not any("**Web Results:**" in r for r in results):
                            results.append("**Current Web Search Results:**")
                        results.extend(web_results[:max_results])
                    else:
                        results.append("⚠️ No current web results found. Try a more specific query.")
                
            except Exception as e:
                await ctx.warning(f"HTML search failed: {str(e)}")
                results.append(f"⚠️ Enhanced search partially failed: {str(e)}")
            
            await ctx.info(f"✅ DuckDuckGo search completed")
            return "\n".join(results) if len(results) > 1 else f"❌ No results found for: {query}"
            
    except Exception as e:
        await ctx.error(f"DuckDuckGo search failed: {str(e)}")
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

@mcp.tool(
        name="get_weather",
        description="""Get current weather forecast using the National Weather Service API."""
)
def get_weather(latitude: float, longitude: float) -> str:
    """
    Get current weather forecast using the National Weather Service API.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
    """
    try:
        headers = {
            "User-Agent": "MCP Weather Client (mcp-weather@example.com)",
            "Accept": "application/geo+json"
        }
        
        # Get grid point information
        points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
        points_response = requests.get(points_url, headers=headers, timeout=10)
        points_response.raise_for_status()
        points_data = points_response.json()
        
        # Extract forecast URL and location info
        forecast_url = points_data['properties']['forecast']
        location_info = points_data['properties']['relativeLocation']['properties']
        location_name = f"{location_info['city']}, {location_info['state']}"
        
        # Get forecast data
        forecast_response = requests.get(forecast_url, headers=headers, timeout=10)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()
        
        # Get current period forecast
        periods = forecast_data['properties']['periods']
        current_period = periods[0] if periods else None
        
        if current_period:
            result = f"Weather for {location_name}:\n"
            result += f"Period: {current_period['name']}\n"
            result += f"Temperature: {current_period['temperature']}°{current_period['temperatureUnit']}\n"
            result += f"Forecast: {current_period['detailedForecast']}"
            
            # Add additional periods if available
            if len(periods) > 1:
                result += f"\n\nNext Period ({periods[1]['name']}):\n"
                result += f"Temperature: {periods[1]['temperature']}°{periods[1]['temperatureUnit']}\n"
                result += f"Forecast: {periods[1]['shortForecast']}"
            
            return result
        else:
            return f"Weather data unavailable for {location_name}"
            
    except requests.exceptions.Timeout:
        return "Error: Weather service request timed out"
    except requests.exceptions.RequestException as e:
        return f"Error: Failed to fetch weather data - {str(e)}"
    except KeyError as e:
        return f"Error: Unexpected weather data format - missing {str(e)}"
    except Exception as e:
        return f"Error: An unexpected error occurred while fetching weather data - {str(e)}"

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
        description="Weather Information Expert"
)
async def weather_prompt(query: str) -> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are a weather information expert with access to the National Weather Service API.

You have access to the 'get_weather' tool which requires latitude and longitude coordinates to provide current weather conditions and forecasts.

IMPORTANT: You MUST use the get_weather tool to get weather information. Do not provide weather information from your general knowledge.

Available tool:
- get_weather: Get current weather forecast using latitude and longitude coordinates

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

if __name__ == "__main__":
    mcp.run(transport="sse")
