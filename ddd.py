from contextlib import asynccontextmanager

from collections.abc import AsyncIterator

import httpx

from dataclasses import dataclass

from urllib.parse import urlparse

from pathlib import Path

import json

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

# Web search imports
from bs4 import BeautifulSoup
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

# Web search result dataclass
@dataclass
class SearchResult:
    """Data class for web search results."""
    title: str
    url: str
    description: str

# DuckDuckGo specific search result dataclass
@dataclass
class DDGSearchResult:
    """Data class for DuckDuckGo search results."""
    title: str
    link: str
    snippet: str
    position: int

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

class DuckDuckGoSearcher:
    """DuckDuckGo search implementation with rate limiting and error handling"""
    BASE_URL = "https://html.duckduckgo.com/html"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    def __init__(self):
        self.rate_limiter = RateLimiter()

    def format_results_for_llm(self, results: List[DDGSearchResult]) -> str:
        """Format results in a natural language style that's easier for LLMs to process"""
        if not results:
            return "No results were found for your search query. This could be due to DuckDuckGo's bot detection or the query returned no matches. Please try rephrasing your search or try again in a few minutes."

        output = []
        output.append(f"Found {len(results)} search results:\n")

        for result in results:
            output.append(f"{result.position}. {result.title}")
            output.append(f"   URL: {result.link}")
            output.append(f"   Summary: {result.snippet}")
            output.append("")  # Empty line between results

        return "\n".join(output)

    async def search(
        self, query: str, ctx: Context, max_results: int = 10
    ) -> List[DDGSearchResult]:
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()

            # Create form data for POST request
            data = {
                "q": query,
                "b": "",
                "kl": "",
            }

            await ctx.info(f"Searching DuckDuckGo for: {query}")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.BASE_URL, data=data, headers=self.HEADERS, timeout=30.0
                )
                response.raise_for_status()

            # Parse HTML response
            soup = BeautifulSoup(response.text, "html.parser")
            if not soup:
                await ctx.error("Failed to parse HTML response")
                return []

            results = []
            for result in soup.select(".result"):
                title_elem = result.select_one(".result__title")
                if not title_elem:
                    continue

                link_elem = title_elem.find("a")
                if not link_elem:
                    continue

                title = link_elem.get_text(strip=True)
                link = link_elem.get("href", "")

                # Skip ad results
                if "y.js" in link:
                    continue

                # Clean up DuckDuckGo redirect URLs
                if link.startswith("//duckduckgo.com/l/?uddg="):
                    link = urllib.parse.unquote(link.split("uddg=")[1].split("&")[0])

                snippet_elem = result.select_one(".result__snippet")
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                results.append(
                    DDGSearchResult(
                        title=title,
                        link=link,
                        snippet=snippet,
                        position=len(results) + 1,
                    )
                )

                if len(results) >= max_results:
                    break

            await ctx.info(f"Successfully found {len(results)} results")
            return results

        except httpx.TimeoutException:
            await ctx.error("Search request timed out")
            return []
        except httpx.HTTPError as e:
            await ctx.error(f"HTTP error occurred: {str(e)}")
            return []
        except Exception as e:
            await ctx.error(f"Unexpected error during search: {str(e)}")
            traceback.print_exc(file=sys.stderr)
            return []

class WebContentFetcher:
    """Fetch and parse web content with rate limiting"""
    def __init__(self):
        self.rate_limiter = RateLimiter(requests_per_minute=20)

    async def fetch_and_parse(self, url: str, ctx: Context) -> str:
        """Fetch and parse content from a webpage"""
        try:
            await self.rate_limiter.acquire()

            await ctx.info(f"Fetching content from: {url}")

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                    follow_redirects=True,
                    timeout=30.0,
                )
                response.raise_for_status()

            # Parse the HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()

            # Get the text content
            text = soup.get_text()

            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text).strip()

            # Truncate if too long
            if len(text) > 8000:
                text = text[:8000] + "... [content truncated]"

            await ctx.info(
                f"Successfully fetched and parsed content ({len(text)} characters)"
            )
            return text

        except httpx.TimeoutException:
            await ctx.error(f"Request timed out for URL: {url}")
            return "Error: The request timed out while trying to fetch the webpage."
        except httpx.HTTPError as e:
            await ctx.error(f"HTTP error occurred while fetching {url}: {str(e)}")
            return f"Error: Could not access the webpage ({str(e)})"
        except Exception as e:
            await ctx.error(f"Error fetching content from {url}: {str(e)}")
            return f"Error: An unexpected error occurred while fetching the webpage ({str(e)})"

# Initialize search components
searcher = DuckDuckGoSearcher()
fetcher = WebContentFetcher()
 
 
 
 
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

# Real Search Tool - LangChain Compatible
@mcp.tool(
        name="real_search",
        description="""Search using DuckDuckGo API for current information."""
)
async def real_search(query: str) -> str:
    """
    Use DuckDuckGo Instant Answer API for real current data.
    
    Args:
        query: Search query string
    
    Returns:
        Search results from API as formatted string
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Try DuckDuckGo Instant Answer API first
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = await client.get(
                'https://api.duckduckgo.com/',
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for instant answer
                if data.get('Answer'):
                    return f"🔍 DuckDuckGo Instant Answer:\n{data['Answer']}\n📎 Source: {data.get('AnswerURL', 'N/A')}"
                
                # Check for abstract  
                if data.get('Abstract'):
                    return f"📋 DuckDuckGo Abstract:\n{data['Abstract']}\n📎 Source: {data.get('AbstractURL', 'N/A')}"
                
                # Check for definition
                if data.get('Definition'):
                    return f"📖 DuckDuckGo Definition:\n{data['Definition']}\n📎 Source: {data.get('DefinitionURL', 'N/A')}"
                
                # Check for related topics
                if data.get('RelatedTopics'):
                    topics = data['RelatedTopics'][:3]  # First 3 topics
                    results = []
                    for topic in topics:
                        if isinstance(topic, dict) and 'Text' in topic:
                            results.append(topic['Text'])
                    
                    if results:
                        return f"🔗 DuckDuckGo Related Topics:\n" + "\n".join(f"• {topic}" for topic in results)
            
            return f"❌ No current information found for: {query}"
            
    except Exception as e:
        logger.error(f"Real search failed for query '{query}': {str(e)}")
        return f"❌ Search error: {str(e)}"

# === NEW TOOLS: DuckDuckGo Search and Weather ===

@mcp.tool(
        name="duckduckgo_search",
        description="""Search DuckDuckGo for current web information and return formatted results."""
)
async def duckduckgo_search(query: str, ctx: Context, max_results: int = 10) -> str:
    """
    Search DuckDuckGo and return formatted results.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 10)
    """
    try:
        results = await searcher.search(query, ctx, max_results)
        return searcher.format_results_for_llm(results)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return f"An error occurred while searching: {str(e)}"

@mcp.tool(
        name="fetch_content",
        description="""Fetch and parse content from a webpage URL."""
)
async def fetch_content(url: str, ctx: Context) -> str:
    """
    Fetch and parse content from a webpage URL.

    Args:
        url: The webpage URL to fetch content from
    """
    return await fetcher.fetch_and_parse(url, ctx)

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

# Web Search Tool - LangChain Compatible Version
@mcp.tool(
        name="web_search",
        description="""Search the web for current information. Use this tool to get up-to-date data."""
)
async def web_search(query: str, limit: int = 5) -> str:
    """
    Search the web using multiple search engines for current results.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return (1-10, default: 5)
    
    Returns:
        String containing search results with title, url, and description
    """
    # Validate inputs
    if not query or not query.strip():
        return "Error: Query cannot be empty"
    
    if limit < 1 or limit > 10:
        return "Error: Limit must be between 1 and 10"
    
    try:
        results = await _perform_enhanced_search(query.strip(), limit)
        
        if not results:
            return f"No search results found for query: {query}"
        
        # Format results as a readable string for the LLM
        formatted_results = f"Search results for '{query}':\n\n"
        
        for i, result in enumerate(results, 1):
            formatted_results += f"{i}. **{result.title}**\n"
            formatted_results += f"   URL: {result.url}\n"
            formatted_results += f"   Description: {result.description}\n\n"
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Web search failed for query '{query}': {str(e)}")
        return f"Web search error: {str(e)}"

async def _perform_enhanced_search(query: str, limit: int) -> List[SearchResult]:
    """
    Perform enhanced web search using multiple strategies.
    Based on the TypeScript reference but with improvements for current data.
    
    Args:
        query: Search query string
        limit: Maximum number of results
        
    Returns:
        List of SearchResult objects
    """
    # Enhanced headers similar to TypeScript reference
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
    }
    
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        # Try multiple search strategies in order of preference
        search_strategies = [
            # Strategy 1: DuckDuckGo (often more current and less cached)
            {
                'name': 'DuckDuckGo',
                'url': 'https://html.duckduckgo.com/html/',
                'params': {'q': query},
                'parser': _parse_duckduckgo_results
            },
            
            # Strategy 2: Google with cache-busting parameters
            {
                'name': 'Google Fresh',
                'url': 'https://www.google.com/search',
                'params': {
                    'q': query, 
                    'tbs': 'qdr:w',  # Results from past week
                    'num': limit,
                    'safe': 'off',
                    'gl': 'us',
                    'hl': 'en'
                },
                'parser': _parse_google_results
            },
            
            # Strategy 3: Bing search
            {
                'name': 'Bing',
                'url': 'https://www.bing.com/search',
                'params': {'q': query, 'count': limit},
                'parser': _parse_bing_results
            },
            
            # Strategy 4: Google News for very current events
            {
                'name': 'Google News',
                'url': 'https://news.google.com/search',
                'params': {
                    'q': query,
                    'hl': 'en-US',
                    'gl': 'US',
                    'ceid': 'US:en'
                },
                'parser': _parse_google_news_results
            }
        ]
        
        for strategy in search_strategies:
            try:
                logger.info(f"Trying {strategy['name']} search for: {query}")
                
                response = await client.get(
                    strategy['url'],
                    params=strategy['params'],
                    headers=headers
                )
                
                if response.status_code == 200:
                    results = strategy['parser'](response.text, limit)
                    if results:
                        logger.info(f"Success: Found {len(results)} results from {strategy['name']}")
                        return results
                    else:
                        logger.warning(f"No results from {strategy['name']}")
                else:
                    logger.warning(f"{strategy['name']} returned status {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"{strategy['name']} search failed: {str(e)}")
                continue
        
        # If all strategies fail, return empty list
        logger.error("All search strategies failed")
        return []

def _parse_google_results(html_content: str, limit: int) -> List[SearchResult]:
    """Parse Google search results - similar to TypeScript reference."""
    soup = BeautifulSoup(html_content, 'html.parser')
    results = []
    
    # Google result containers (same as TypeScript reference)
    search_containers = soup.find_all('div', class_='g')
    
    for i, container in enumerate(search_containers[:limit]):
        try:
            # Extract title (same pattern as TypeScript)
            title_element = container.find('h3')
            if not title_element:
                continue
                
            title = title_element.get_text(strip=True)
            
            # Extract URL (same pattern as TypeScript)
            link_element = container.find('a')
            if not link_element or not link_element.get('href'):
                continue
                
            url = link_element.get('href')
            
            # Skip non-HTTP URLs (same as TypeScript)
            if not url.startswith('http'):
                continue
            
            # Extract description (same pattern as TypeScript)
            snippet_element = container.find(class_=['VwiC3b', 's3v9rd', 'st'])
            description = snippet_element.get_text(strip=True) if snippet_element else ''
            
            if title and url:
                results.append(SearchResult(
                    title=title,
                    url=url,
                    description=description
                ))
                
        except Exception as e:
            logger.warning(f"Failed to parse Google result {i}: {str(e)}")
            continue
    
    return results

def _parse_duckduckgo_results(html_content: str, limit: int) -> List[SearchResult]:
    """Parse DuckDuckGo search results."""
    soup = BeautifulSoup(html_content, 'html.parser')
    results = []
    
    # DuckDuckGo result containers
    search_containers = soup.find_all('div', class_=['web-result', 'result'])
    
    for i, container in enumerate(search_containers[:limit]):
        try:
            # Extract title and URL
            title_element = container.find('h2')
            if title_element:
                title_link = title_element.find('a')
            else:
                title_link = container.find('a', class_='result__a')
            
            if not title_link:
                continue
                
            title = title_link.get_text(strip=True)
            url = title_link.get('href', '')
            
            # Skip non-HTTP URLs
            if not url.startswith('http'):
                continue
            
            # Extract description
            snippet_element = container.find(class_=['result__snippet', 'result__body'])
            description = snippet_element.get_text(strip=True) if snippet_element else ''
            
            if title and url:
                results.append(SearchResult(
                    title=title,
                    url=url,
                    description=description
                ))
                
        except Exception as e:
            logger.warning(f"Failed to parse DuckDuckGo result {i}: {str(e)}")
            continue
    
    return results

def _parse_bing_results(html_content: str, limit: int) -> List[SearchResult]:
    """Parse Bing search results."""
    soup = BeautifulSoup(html_content, 'html.parser')
    results = []
    
    # Bing result containers
    search_containers = soup.find_all('li', class_='b_algo')
    
    for i, container in enumerate(search_containers[:limit]):
        try:
            # Extract title and URL
            title_element = container.find('h2')
            if not title_element:
                continue
                
            title_link = title_element.find('a')
            if not title_link:
                continue
                
            title = title_link.get_text(strip=True)
            url = title_link.get('href', '')
            
            # Skip non-HTTP URLs
            if not url.startswith('http'):
                continue
            
            # Extract description
            snippet_element = container.find(class_=['b_caption', 'b_snippetText'])
            description = snippet_element.get_text(strip=True) if snippet_element else ''
            
            if title and url:
                results.append(SearchResult(
                    title=title,
                    url=url,
                    description=description
                ))
                
        except Exception as e:
            logger.warning(f"Failed to parse Bing result {i}: {str(e)}")
            continue
    
    return results

def _parse_google_news_results(html_content: str, limit: int) -> List[SearchResult]:
    """Parse Google News search results."""
    soup = BeautifulSoup(html_content, 'html.parser')
    results = []
    
    # Google News result containers
    search_containers = soup.find_all('article') or soup.find_all('div', class_='xrnccd')
    
    for i, container in enumerate(search_containers[:limit]):
        try:
            # Extract title and URL
            title_element = container.find('h3') or container.find('h4')
            if not title_element:
                continue
                
            title_link = title_element.find('a') or container.find('a')
            if not title_link:
                continue
                
            title = title_element.get_text(strip=True)
            url = title_link.get('href', '')
            
            # Handle Google News URLs
            if url.startswith('./articles/'):
                url = f"https://news.google.com{url[1:]}"
            elif not url.startswith('http'):
                continue
            
            # Extract description
            snippet_element = container.find(class_=['st', 'snippet']) or container.find('p')
            description = snippet_element.get_text(strip=True) if snippet_element else ''
            
            if title and url:
                results.append(SearchResult(
                    title=title,
                    url=url,
                    description=description
                ))
                
        except Exception as e:
            logger.warning(f"Failed to parse Google News result {i}: {str(e)}")
            continue
    
    return results
 
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

# === NEW PROMPTS: DuckDuckGo Search and Weather ===

@mcp.prompt(
        name="search-prompt",
        description="Web Search Expert"
)
async def search_prompt(query: str) -> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are an expert at finding and analyzing information from the web using DuckDuckGo search.
            
            You have access to these search tools:
            1) duckduckgo_search - Search DuckDuckGo for current web information
            2) fetch_content - Retrieve and parse content from specific web pages
            3) web_search - Multi-engine web search with fallback strategies
            4) real_search - DuckDuckGo instant answers and quick facts
            
            Use these tools to find accurate, current information and provide comprehensive answers with proper source citations.
            Always verify information quality and cite your sources.
            
            Query: {query}"""
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
            
            You have access to the get_weather tool which requires latitude and longitude coordinates to provide:
            - Current weather conditions
            - Detailed forecasts  
            - Temperature information
            - Weather alerts and warnings
            
            When users ask about weather for a location, help them provide coordinates or use your knowledge to suggest approximate coordinates for major cities.
            
            Common city coordinates:
            - Richmond, VA: 37.5407, -77.4360
            - Atlanta, GA: 33.7490, -84.3880
            - New York, NY: 40.7128, -74.0060
            - Denver, CO: 39.7392, -104.9903
            - Miami, FL: 25.7617, -80.1918
            
            If coordinates aren't provided, ask the user for them or suggest coordinates for the nearest major city.
            Provide detailed, helpful weather information including current conditions and short-term forecasts.
            
            Query: {query}"""
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

To call the test_tool with ChatSnowflakeCortex, include this:
{{"invoke_tool": "{{\\"tool_name\\": \\"test_tool\\", \\"args\\": {{\\"message\\": \\"{message}\\"}}}}"}}

Execute the test_tool now."""
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

To call the diagnostic tool with ChatSnowflakeCortex, include this:
{{"invoke_tool": "{{\\"tool_name\\": \\"diagnostic\\", \\"args\\": {{\\"test_type\\": \\"{test_type}\\"}}}}"}}

Execute the diagnostic tool now."""
        }
    ]
 
 
if __name__ == "__main__":

    mcp.run(transport="sse")
