
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
 
 
# Create a named server

mcp = FastMCP("DataFlyWheel App")
 
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

# Web Search Tool - Enhanced Version
@mcp.tool(
        name="web_search",
        description="""
        Search the web using multiple search engines for current information.
        
        Example inputs:
        
        current president of USA
        latest news about technology
        weather forecast today
        
        Returns current web search results with title, url, and description.
        
        Args:
            query (str): Search query string
            limit (int): Maximum number of results to return (1-10, default: 5)
        """
)
async def web_search(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Enhanced web search using multiple search engines for current results.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return (1-10, default: 5)
    
    Returns:
        List of search results with title, url, and description
    """
    # Validate inputs
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    if limit < 1 or limit > 10:
        raise ValueError("Limit must be between 1 and 10")
    
    try:
        results = await _perform_enhanced_search(query.strip(), limit)
        return [
            {
                "title": result.title,
                "url": result.url, 
                "description": result.description
            }
            for result in results
        ]
    except Exception as e:
        logger.error(f"Web search failed for query '{query}': {str(e)}")
        raise Exception(f"Web search error: {str(e)}")

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

# Web Search Prompt - Force Tool Usage
@mcp.prompt(
        name="web-search-analysis",
        description="Current Web Search Expert"
)
async def web_search_analysis_prompt(query: str, results_count: int = 5) -> List[Message]:
    """
    Generate a prompt that forces web search tool usage.
    
    Args:
        query: The search query 
        results_count: Number of results to include
    
    Returns:
        Formatted prompt messages for current web search
    """
    return [
        {
            "role": "user",
            "content": f"""I need you to search for CURRENT information. Do NOT use your training data.

MANDATORY INSTRUCTIONS - FOLLOW EXACTLY:
1. You MUST call the web_search tool before answering
2. Do NOT provide any answer without first calling web_search
3. Your training data is outdated - only trust search results
4. If search results show old information, try different search terms

SEARCH REQUIRED FOR: "{query}"

RESPONSE FORMAT:
- First: Call web_search tool 
- Then: Answer based ONLY on search results
- Say: "Based on current search results: [answer]"

DO NOT ANSWER WITHOUT CALLING THE TOOL FIRST."""
        }
    ]
 
 
if __name__ == "__main__":

    mcp.run(transport="sse")
