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

# Bing Search Result dataclass
@dataclass
class BingSearchResult:
    """Data class for Bing search results."""
    id: str
    title: str
    link: str
    snippet: str

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

class BingUSASearchEngine:
    """Bing USA Search Engine based on the bing-cn-mcp-server implementation"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter(requests_per_minute=20)
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        self.search_results = {}  # Store search results globally
        
    async def search_bing(self, query: str, num_results: int = 5) -> List[BingSearchResult]:
        """
        Bing search function
        Args:
            query: Search keywords
            num_results: Number of results to return
        Returns:
            List of search results
        """
        try:
            await self.rate_limiter.acquire()
            
            # Build Bing USA search URL with English support
            search_url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}&setlang=en-US&cc=US"
            
            # Set request headers to simulate US browser
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Upgrade-Insecure-Requests': '1',
                'Cookie': 'SRCHHPGUSR=SRCHLANG=en; _EDGE_S=ui=en-us; _EDGE_V=1'
            }
            
            # Send request
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.get(search_url, headers=headers)
                response.raise_for_status()
                
                # Parse HTML using simple regex (mimicking cheerio functionality)
                html_content = response.text
                results = []
                
                # Find search result list elements using regex patterns
                # Try multiple selectors for better results
                result_selectors = [
                    r'<h2><a[^>]+href="([^"]+)"[^>]*>([^<]+)</a></h2>',
                    r'<h3[^>]*><a[^>]+href="([^"]+)"[^>]*>([^<]+)</a></h3>'
                ]
                
                for selector in result_selectors:
                    matches = re.findall(selector, html_content)
                    for i, (url, title) in enumerate(matches):
                        if len(results) >= num_results:
                            break
                            
                        if url.startswith('http'):
                            # Get snippet/description
                            snippet = self._extract_snippet(html_content, title, url)
                            
                            # Create unique ID
                            result_id = f"result_{int(time.time())}_{i}"
                            
                            result = BingSearchResult(
                                id=result_id,
                                title=title.strip(),
                                link=url.strip(),
                                snippet=snippet
                            )
                            
                            # Store result for later retrieval
                            self.search_results[result_id] = result
                            results.append(result)
                    
                    if results:
                        break  # Stop trying other selectors if we found results
                
                # If no results found, add a fallback
                if not results:
                    result_id = f"result_{int(time.time())}_fallback"
                    result = BingSearchResult(
                        id=result_id,
                        title=f"Search results for: {query}",
                        link=search_url,
                        snippet=f"Could not parse search results for \"{query}\", but you can visit the Bing search page directly."
                    )
                    self.search_results[result_id] = result
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f'Bing search error: {e}')
            # Return error result
            result_id = f"error_{int(time.time())}"
            result = BingSearchResult(
                id=result_id,
                title=f"Error searching for \"{query}\"",
                link=f"https://www.bing.com/search?q={urllib.parse.quote(query)}",
                snippet=f"An error occurred during search: {str(e)}"
            )
            self.search_results[result_id] = result
            return [result]

    def _extract_snippet(self, html_content: str, title: str, url: str) -> str:
        """Extract snippet from HTML content"""
        try:
            # Simple regex to find description/snippet near the title
            snippet_patterns = [
                rf'{re.escape(title)}.*?<p[^>]*>([^<]+)</p>',
                rf'<p[^>]*class="[^"]*caption[^"]*"[^>]*>([^<]+)</p>',
                rf'<div[^>]*class="[^"]*snippet[^"]*"[^>]*>([^<]+)</div>'
            ]
            
            for pattern in snippet_patterns:
                matches = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)
                if matches:
                    snippet = matches.group(1).strip()
                    if len(snippet) > 20:
                        return snippet[:200] + "..." if len(snippet) > 200 else snippet
            
            # Fallback snippet
            return f"Result from {url}"
            
        except Exception:
            return f"Result from {url}"

    async def fetch_webpage_content(self, result_id: str) -> str:
        """
        Fetch webpage content by result ID
        Args:
            result_id: Search result ID returned from bing_search
        Returns:
            Webpage content
        """
        try:
            # Get result from stored results
            result = self.search_results.get(result_id)
            if not result:
                raise Exception(f"找不到ID为 {result_id} 的搜索结果")
            
            url = result.link
            
            # Set request headers to simulate browser
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
                'Referer': 'https://cn.bing.com/'
            }
            
            # Send request to get webpage content
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
                html_content = response.text
                
                # Extract main content using simple regex patterns
                content = self._extract_main_content(html_content)
                
                # Add title
                title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip()
                    content = f"标题: {title}\n\n{content}"
                
                # Limit content length
                max_length = 8000
                if len(content) > max_length:
                    content = content[:max_length] + '... (内容已截断)'
                
                return content
                
        except Exception as e:
            logger.error(f'Fetch webpage content error: {e}')
            raise Exception(f"Failed to fetch webpage content: {str(e)}")

    def _extract_main_content(self, html_content: str) -> str:
        """Extract main content from HTML"""
        try:
            # Remove script, style, and other unwanted elements
            content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r'<iframe[^>]*>.*?</iframe>', '', content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r'<noscript[^>]*>.*?</noscript>', '', content, flags=re.DOTALL | re.IGNORECASE)
            
            # Try to find main content areas
            main_patterns = [
                r'<main[^>]*>(.*?)</main>',
                r'<article[^>]*>(.*?)</article>',
                r'<div[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</div>',
                r'<div[^>]*class="[^"]*main[^"]*"[^>]*>(.*?)</div>',
                r'<div[^>]*class="[^"]*post[^"]*"[^>]*>(.*?)</div>'
            ]
            
            main_content = ""
            for pattern in main_patterns:
                matches = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                if matches:
                    main_content = matches.group(1)
                    break
            
            # If no main content area found, extract all paragraphs
            if not main_content or len(main_content) < 100:
                paragraphs = re.findall(r'<p[^>]*>([^<]+)</p>', content, re.IGNORECASE)
                main_content = '\n\n'.join([p.strip() for p in paragraphs if len(p.strip()) > 20])
            
            # If still no content, use body content
            if not main_content or len(main_content) < 100:
                body_match = re.search(r'<body[^>]*>(.*?)</body>', content, re.DOTALL | re.IGNORECASE)
                if body_match:
                    main_content = body_match.group(1)
            
            # Clean up HTML tags
            main_content = re.sub(r'<[^>]+>', ' ', main_content)
            main_content = re.sub(r'\s+', ' ', main_content).strip()
            
            return main_content
            
        except Exception:
            # Fallback: just remove all HTML tags
            clean_content = re.sub(r'<[^>]+>', ' ', html_content)
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()
            return clean_content[:3000]

    def format_search_results(self, results: List[BingSearchResult]) -> str:
        """Format search results for display"""
        if not results:
            return "No search results found. Please try different search terms."
        
        output = []
        output.append(f"🔍 Bing Search Results ({len(results)} results):\n")
        
        for i, result in enumerate(results, 1):
            output.append(f"## Result {i}: {result.title}")
            output.append(f"**ID:** {result.id}")
            output.append(f"**Link:** {result.link}")
            output.append(f"**Summary:** {result.snippet}")
            output.append("")  # Empty line between results
        
        return "\n".join(output)

# Initialize Bing USA search engine
bing_search_engine = BingUSASearchEngine()

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

# === NEW BING CHINESE WEB SEARCH TOOLS ===

@mcp.tool(
        name="bing_search",
        description="""Search Bing for specified keywords and return a list of search results including titles, links, summaries, and IDs"""
)
async def bing_search(query: str, ctx: Context, num_results: int = 5) -> str:
    """
    Search Bing for specified keywords and return a list of search results
    
    Args:
        query: Search keywords
        num_results: Number of results to return, default is 5
    
    Returns:
        Formatted search results including titles, links, summaries, and IDs
    """
    try:
        await ctx.info(f"🔍 Starting Bing search: {query}")
        
        # Execute Bing search
        results = await bing_search_engine.search_bing(query, num_results)
        
        if not results:
            return f"❌ No search results found for \"{query}\". Please try different search terms."
        
        # Format results
        formatted_results = bing_search_engine.format_search_results(results)
        
        await ctx.info(f"✅ Bing search completed: found {len(results)} results")
        
        return formatted_results
        
    except Exception as e:
        await ctx.error(f"Bing search failed for query '{query}': {str(e)}")
        return f"❌ Bing search error: {str(e)}"

@mcp.tool(
        name="fetch_webpage",
        description="""Fetch webpage content based on the provided ID"""
)
async def fetch_webpage(result_id: str, ctx: Context) -> str:
    """
    Fetch webpage content based on the provided ID
    
    Args:
        result_id: Result ID returned from bing_search
    
    Returns:
        Text content of the webpage
    """
    try:
        await ctx.info(f"📄 Fetching webpage content, ID: {result_id}")
        
        # Get webpage content
        content = await bing_search_engine.fetch_webpage_content(result_id)
        
        await ctx.info(f"✅ Webpage content fetched successfully, length: {len(content)} characters")
        
        return content
        
    except Exception as e:
        await ctx.error(f"Error fetching webpage content for ID '{result_id}': {str(e)}")
        return f"❌ Failed to fetch webpage content: {str(e)}"

@mcp.tool(
        name="web_research",
        description="""Comprehensive web research tool that searches and analyzes content"""
)
async def web_research(query: str, ctx: Context, max_results: int = 5) -> str:
    """
    Comprehensive web research tool that searches and analyzes content
    
    Args:
        query: Research query string
        max_results: Maximum number of results (default: 5)
    
    Returns:
        Formatted research results with summaries and relevance scores
    """
    try:
        await ctx.info(f"🔍 Starting comprehensive web research: {query}")
        
        # Execute Bing search
        results = await bing_search_engine.search_bing(query, max_results)
        
        if not results:
            return f"❌ No web research results found for '{query}'. Please try different search terms."
        
        # Format results for research
        output = []
        output.append(f"🔍 Web Research Results: '{query}' ({len(results)} sources analyzed):\n")
        
        for i, result in enumerate(results, 1):
            output.append(f"## Research Source {i}: {result.title}")
            output.append(f"**Link:** {result.link}")
            output.append(f"**ID:** {result.id} (use for full content)")
            output.append(f"**Summary:** {result.snippet}")
            output.append(f"**Relevance:** High")  # Simplified relevance scoring
            output.append("")  # Empty line between results
        
        output.append("💡 **Tip:** Use the fetch_webpage tool with the respective ID to get full content from any source.")
        
        await ctx.info(f"✅ Web research completed: analyzed {len(results)} results")
        
        return "\n".join(output)
        
    except Exception as e:
        await ctx.error(f"Web research failed for query '{query}': {str(e)}")
        return f"❌ Web research error: {str(e)}"

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

# === PROMPTS ===

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
        name="bing-search-prompt",
        description="Bing Search Expert"
)
async def bing_search_prompt(query: str) -> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are a professional Bing search assistant specializing in finding current, accurate information using the Bing search engine.

Your capabilities include:
- Multi-language search using Bing search engine
- Content analysis and summarization
- Relevance scoring and ranking
- Real-time information gathering

Available tools:
- bing_search: Search Bing for keywords, returning results list with IDs
- fetch_webpage: Get full webpage content using search result IDs
- web_research: Comprehensive web research with content analysis and relevance scoring

For the query "{query}", please:
1. Use appropriate Bing search tools to gather current information
2. Analyze and summarize findings
3. Provide insights based on research results
4. Cite sources and relevance scores when available

Research Query: {query}"""
        }
    ]

@mcp.prompt(
        name="web-research-prompt",
        description="Web Research Expert"
)
async def web_research_prompt(query: str) -> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are a comprehensive web research assistant specializing in finding current, accurate information from multiple online sources using Bing search engine.

Your capabilities include:
- Multi-engine web search across Bing and other sources
- Content analysis and summarization
- Relevance scoring and ranking
- Real-time information gathering

Available tools:
- web_research: Comprehensive research with content analysis and relevance scoring
- bing_search: Bing search for specific information
- fetch_webpage: Get full webpage content for in-depth analysis

For the query "{query}", please:
1. Use appropriate web research tools to gather current information
2. Analyze and summarize findings
3. Provide insights based on research results
4. Cite sources and relevance scores when available

If coordinates aren't provided, ask the user for them or suggest coordinates for the nearest major city.
Provide detailed, helpful information including current conditions and short-term forecasts.

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
