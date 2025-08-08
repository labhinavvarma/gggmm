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

# SerpApi utilities - no API key variables
SERPAPI_AVAILABLE = True

def format_query_for_serpapi(query: str) -> str:
    """Format query for SerpApi by replacing spaces with + signs"""
    return query.replace(" ", "+")

def build_serpapi_url(query: str) -> str:
    """Build SerpApi URL using the exact hardcoded format specified"""
    formatted_query = format_query_for_serpapi(query)
    
    # Use exact hardcoded URL format with API key directly embedded
    url = f"https://serpapi.com/search.json?engine=google&q={formatted_query}&google_domain=google.com&gl=us&hl=en&api_key=28009a3e8f74ab4680e232c4ed5ae4f0e5d1bf849d052100ce3f7f74be9d4e54"
    
    return url

async def fetch_serpapi_json(query: str) -> dict:
    """Fetch JSON response from SerpApi using exact hardcoded URL format"""
    
    print(f"\n🌐 BUILDING SERPAPI URL FOR: '{query}'")
    
    url = build_serpapi_url(query)
    
    print(f"📡 SERPAPI URL: {url}")
    print(f"🔗 MAKING HTTP REQUEST TO SERPAPI...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        results = response.json()
        
        print(f"✅ HTTP REQUEST SUCCESSFUL")
        print(f"📊 RESPONSE STATUS: {response.status_code}")
        print(f"📊 RESPONSE KEYS: {list(results.keys())}")
        
        return results
        
    except requests.exceptions.RequestException as e:
        print(f"❌ HTTP REQUEST FAILED: {str(e)}")
        raise Exception(f"SerpApi request failed: {str(e)}")
    except Exception as e:
        print(f"❌ SERPAPI ERROR: {str(e)}")
        raise Exception(f"SerpApi error: {str(e)}")

# Create a named server
mcp = FastMCP("DataFlyWheel App")

@dataclass
class AppContext:
    conn: SnowflakeConnection
    db: str
    schema: str
    host: str

# Resources
@mcp.resource(uri="schematiclayer://cortex_analyst/schematic_models/{stagename}/list", name="hedis_schematic_models", description="Hedis Schematic models")
async def get_schematic_model(stagename: str):
    """Cortex analyst schematic layer model, model is in yaml format"""
    
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
    name="add-frequent-questions",
    description="""
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
async def add_frequent_questions(ctx: Context, uri: str, questions: list) -> list:
    # Parse and extract aplctn_cd and user_context (urllib)
    url_path = urlparse(uri)
    aplctn_cd = url_path.netloc
    user_context = Path(url_path.path).name
    file_data = {}
    file_name = aplctn_cd + "_freq_questions.json"
    if Path(file_name).exists():
        file_data = json.load(open(file_name, 'r'))
        file_data[aplctn_cd].extend(questions)
    else:
        file_data[aplctn_cd] = questions

    index_dict = {
        user_context: set()
    }
    result = []
    # Remove duplicates
    for elm in file_data[aplctn_cd]:
        if elm["user_context"] == user_context and elm['prompt'] not in index_dict[user_context]:
            result.append(elm)
            index_dict[user_context].add(elm['prompt'])

    file_data[aplctn_cd] = result

    with open(file_name, 'w') as file:
        file.write(json.dumps(file_data))

    return file_data[aplctn_cd]

@mcp.tool(
    name="add-prompts",
    description="""
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
async def add_prompts(ctx: Context, uri: str, prompt: dict) -> dict:
    # Parse and extract aplctn_cd and user_context (urllib)
    url_path = urlparse(uri)
    aplctn_cd = url_path.netloc
    prompt_name = Path(url_path.path).name
    
    # Before adding the prompt to file add to the server
    # Add prompts to server
    def func1(query: str):
        return [
            {
                "role": "user",
                "content": prompt["content"] + f"\n  {query}"
            }
        ]
    
    ctx.fastmcp.add_prompt(
        Prompt.from_function(
            func1, name=prompt["prompt_name"], description=prompt["description"])
    )

    file_data = {}
    file_name = aplctn_cd + "_prompts.json"
    if Path(file_name).exists():
        with open(file_name, 'r') as file:
            file_data = json.load(file)
        file_data[aplctn_cd].append(prompt)
    else:
        file_data[aplctn_cd] = [prompt]

    with open(file_name, 'w') as file:
        file.write(json.dumps(file_data))

    return prompt

# Tools: Cortex Analyst; Cortex Search; Cortex Complete

@mcp.tool(
    name="DFWAnalyst",
    description="""
    Converts text to valid SQL which can be executed on HEDIS value sets and code sets.

    Example inputs:
       What are the codes in <some value> Value Set?

    Returns valid sql to retrieve data from underlying value sets and code sets.

    Args:
           prompt (str):  text to be passed

    """
)
async def dfw_text2sql(prompt: str, ctx: Context) -> dict:
    """Tool to convert natural language text to snowflake sql for hedis system, text should be passed as 'prompt' input parameter"""

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
    name="DFWSearch",
    description="""
    Searches HEDIS measure specification documents.

    Example inputs:
    What is the age criteria for BCS Measure?
    What is EED Measure in HEDIS?
    Describe COA Measure?
    What LOB is COA measure scoped under?

    Returns information utilizing HEDIS measure specification documents.

    Args:
          query (str): text to be passed
   """
)
async def dfw_search(ctx: Context, query: str):
    """Tool to provide search against HEDIS business documents for the year 2024, search string should be provided as 'query' parameter"""

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
    name="SerpApiRawDebug",
    description="""
    DEBUG TOOL: Returns raw SerpApi JSON response using hardcoded URL format.
    Prints detailed logs to server console for debugging.

    Args:
           query (str): Search query text
    """
)
async def serpapi_raw_debug(query: str) -> str:
    """Debug tool that returns raw SerpApi JSON response using hardcoded URL format"""
    
    print(f"\n🔧 TOOL INVOKED: SerpApiRawDebug")
    print(f"📞 DEBUG QUERY: '{query}'")
    print(f"✅ TOOL CONNECTION CONFIRMED - PROCESSING DEBUG REQUEST...")
    
    if not SERPAPI_AVAILABLE:
        print(f"❌ ERROR: SerpApi not available")
        return "Error: SerpApi is not available."
    
    try:
        print(f"\n{'='*80}")
        print(f"SERPAPI DEBUG CALL FOR QUERY: '{query}'")
        print(f"{'='*80}")
        
        url = build_serpapi_url(query)
        print(f"📡 HARDCODED SERPAPI URL: {url}")
        print(f"{'='*80}")
        
        # Fetch results using hardcoded URL format
        results = await fetch_serpapi_json(query)
        
        # PRINT RAW JSON TO SERVER CONSOLE
        print(f"RAW SERPAPI JSON RESPONSE:")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"{'='*80}")
        print(f"END OF SERPAPI RESPONSE FOR: '{query}'")
        print(f"{'='*80}\n")
        
        # Also log specific sections for easy analysis
        if "answer_box" in results:
            print(f"ANSWER_BOX CONTENT:")
            print(json.dumps(results["answer_box"], indent=2))
            print(f"-" * 40)
        
        if "knowledge_graph" in results:
            print(f"KNOWLEDGE_GRAPH CONTENT:")
            print(json.dumps(results["knowledge_graph"], indent=2))
            print(f"-" * 40)
        
        if "organic_results" in results:
            print(f"FIRST 2 ORGANIC RESULTS:")
            print(json.dumps(results["organic_results"][:2], indent=2))
            print(f"-" * 40)
        
        # Return the JSON for debugging
        json_output = json.dumps(results, indent=2, ensure_ascii=False)
        
        print(f"✅ SERPAPI DEBUG TOOL COMPLETED SUCCESSFULLY")
        print(f"📤 RETURNING RAW JSON TO CLIENT")
        
        return f"""
SERPAPI RAW DEBUG FOR QUERY: "{query}"

HARDCODED URL USED:
{url}

RAW SERPAPI RESPONSE:
```json
{json_output}
```

END OF RAW DEBUG DATA - CHECK SERVER LOGS FOR DETAILED PRINTOUT
"""
        
    except Exception as e:
        print(f"❌ SERPAPI DEBUG ERROR: {str(e)}")
        return f"Debug error: {str(e)}"

@mcp.tool(
    name="SerpApiSearch",
    description="""
    Performs web searches using SerpApi with hardcoded URL format and returns raw JSON for LLM analysis.
    
    Uses exact URL format: https://serpapi.com/search.json?engine=google&q={query}&google_domain=google.com&gl=us&hl=en&api_key=...
    
    Process:
    1. Formats query (spaces become + signs)
    2. Fetches JSON from SerpApi
    3. Returns complete JSON response for LLM to analyze and answer user's question

    Example inputs:
    "who is prime minister of india"
    "latest AI news" 
    "weather in New York"
    "current stock price of Apple"

    Returns complete JSON response for LLM analysis.

    Args:
           query (str): Search query text
    """
)
async def serpapi_search(query: str) -> str:
    """Tool to perform web searches using SerpApi hardcoded URL format and return JSON for LLM analysis"""
    
    print(f"\n🔍 TOOL INVOKED: SerpApiSearch")
    print(f"📞 SEARCH QUERY: '{query}'")
    print(f"✅ TOOL CONNECTION CONFIRMED - BUILDING SERPAPI REQUEST...")
    
    if not SERPAPI_AVAILABLE:
        print(f"❌ ERROR: SerpApi not available")
        return "Error: SerpApi is not available."
    
    try:
        # Build the exact SerpApi URL format
        url = build_serpapi_url(query)
        
        print(f"📡 USING HARDCODED SERPAPI URL FORMAT")
        print(f"🔗 FORMATTED QUERY: {format_query_for_serpapi(query)}")
        
        # Fetch JSON results
        results = await fetch_serpapi_json(query)
        
        print(f"✅ SERPAPI JSON FETCHED SUCCESSFULLY")
        print(f"📤 RETURNING COMPLETE JSON TO LLM FOR ANALYSIS")
        
        # Return the complete JSON response for LLM analysis
        json_output = json.dumps(results, indent=2, ensure_ascii=False)
        
        return f"""
SERPAPI SEARCH RESULTS FOR: "{query}"

URL USED: {url}

COMPLETE JSON RESPONSE:
```json
{json_output}
```

LLM ANALYSIS INSTRUCTIONS:
Please analyze the above JSON response to answer the user's question: "{query}"

Key sections to examine:
- "answer_box": Direct answers or featured snippets
- "knowledge_graph": Entity information and key facts
- "organic_results": Main search results with titles, snippets, links
- "news_results": Recent news articles (if available)
- "related_questions": Additional relevant questions
- "shopping_results": Product information (if applicable)

Extract the most relevant information from these sections to provide a comprehensive answer.
"""
        
    except Exception as e:
        print(f"❌ SERPAPI SEARCH ERROR: {str(e)}")
        logger.error(f"SerpApi search error: {str(e)}")
        if "429" in str(e):
            return "Error: Rate limit exceeded. Please try again later."
        elif "401" in str(e):
            return "Error: Invalid API key in hardcoded URL."
        else:
            return f"Search error: {str(e)}"

@mcp.tool(
    name="SerpApiUrlTest", 
    description="""
    TEST TOOL: Shows the exact hardcoded SerpApi URL that will be called.
    Use this to verify URL construction with the hardcoded format.

    Args:
           query (str): Search query text
    """
)
async def serpapi_url_test(query: str) -> str:
    """Test tool that shows the exact hardcoded URL that will be called"""
    
    print(f"\n🔗 TOOL INVOKED: SerpApiUrlTest")
    print(f"📞 TESTING URL CONSTRUCTION FOR: '{query}'")
    
    # Build the hardcoded URL
    url = build_serpapi_url(query)
    
    print(f"🔗 CONSTRUCTED HARDCODED URL: {url}")
    
    return f"""
SerpApi URL Test for Query: "{query}"

Hardcoded URL Format:
{url}

Formatted Query: {format_query_for_serpapi(query)}

This is the exact URL that will be used to fetch JSON data from SerpApi.
You can test this URL directly in your browser to see the JSON response.
"""

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
         expression (str): Arithmetic expression input

    """
)
def calculate(expression: str) -> str:
    """
    Evaluates a basic arithmetic expression.
    Supports: +, -, *, /, parentheses, decimals.
    """
    print(f"calculate() called with expression: {expression}", flush=True)
    try:
        allowed_chars = "0123456789+-*/(). "
        if any(char not in allowed_chars for char in expression):
            return "Invalid characters in expression."

        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        print("Error in calculate:", str(e), flush=True)
        return f"Error: {str(e)}"

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
        context (str): context that need to be used for the prompt suggestions.
        aplctn_cd (str): application code.
    """
)
async def question_suggestions(ctx: Context, aplctn_cd: str, app_lvl_prefix: str, session_id: str, top_n: int = 3, context: str = "Initialization", llm_flg: bool = False):
    """Tool to suggest additional prompts within the provided context, context should be passed as 'context' input parameter"""

    if not llm_flg:
        return ctx.read_resource(f"genaiplatform://{aplctn_cd}/frequent_questions/{context}")

    try:
        # Note: SnowFlakeConnector is not defined in this code - needs to be imported or implemented
        # from your_module import SnowFlakeConnector  # TODO: Add proper import
        # For now, this will raise a NameError until SnowFlakeConnector is properly imported
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
        "method": "cortex",
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
                    response_text.append(full_response)

    return json.loads(''.join(response_text))

@mcp.tool()
async def get_weather(place: str) -> str:
    """
    Get weather forecast for a place (e.g., 'New York') without needing an API key.
    """
    print(f"get_weather() called for location: {place}", flush=True)

    try:
        # Step 1: Get coordinates using Nominatim (no key needed)
        nominatim_url = f"https://nominatim.openstreetmap.org/search?q={place}&format=json&limit=1&countrycodes=us"
        response = requests.get(nominatim_url, headers={"User-Agent": "MCP Weather Tool"})
        response.raise_for_status()
        data = response.json()

        if not data:
            return f"Could not find location: {place}. Please try a more specific city name."

        latitude = data[0]["lat"]
        longitude = data[0]["lon"]
        display_name = data[0].get("display_name", place)
        print(f"Found coordinates: {latitude}, {longitude} for {display_name}", flush=True)

        # Step 2: Use NWS API to get forecast
        nws_url = f"https://api.weather.gov/points/{latitude},{longitude}"
        headers = {"User-Agent": "MCP Weather Tool"}
        points_resp = requests.get(nws_url, headers=headers)
       
        if points_resp.status_code == 404:
            return f"Weather service not available for {place}. The National Weather Service only covers US locations."
       
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
            f"Weather for {city}, {state}:\n"
            f"- {period['name']}\n"
            f"- Temp: {period['temperature']}°{period['temperatureUnit']}\n"
            f"- Conditions: {period['shortForecast']}\n"
            f"- Wind: {period['windSpeed']} {period['windDirection']}\n"
            f"- Forecast: {period['detailedForecast']}"
        )

    except Exception as e:
        print("Error:", str(e), flush=True)
        return f"Error fetching weather: {str(e)}"

# Prompts
@mcp.prompt(
    name="hedis-prompt",
    description="HEDIS Expert"
)
async def hedis_prompt(query: str) -> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are expert in HEDIS system, HEDIS is a set of standardized measures that aim to improve healthcare quality by promoting accountability and transparency. You are provided with below tools: 1) DFWAnalyst - Generates SQL to retrieve information for hedis codes and value sets. 2) DFWSearch - Provides search capability against HEDIS measures for measurement year. You will respond with the results returned from right tool. {query}"""
        }
    ]

@mcp.prompt(
    name="calculator-prompt",
    description="Calculator"
)
async def calculator_prompt(query: str) -> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are expert in performing arithmetic operations. You are provided with the tool calculator to verify the results. You will respond with the results after verifying with the tool result. {query}"""
        }
    ]

@mcp.prompt(
    name="weather-prompt",
    description="Weather Expert"
)
async def weather_prompt(query: str) -> List[Message]:
    """Weather expert who intakes the place as input and returns the present weather"""
    return [
        {
            "role": "user",
            "content": f"You are a weather expert. You have been provided with `get_weather` tool to get up to date weather information for: {query}. Always use the tool first."
        }
    ]

@mcp.prompt(
    name="serpapi-prompt",
    description="Web Search Expert using SerpApi"
)
async def serpapi_prompt(query: str) -> List[Message]:
    """Web search expert who uses SerpApi through structured HTTP requests"""
    return [
        {
            "role": "user", 
            "content": f"""You are a web search expert. Use the SerpApiSearch tool to find current information through structured HTTP requests.

            The search system will:
            - Create structured HTTP requests to SerpApi
            - Apply appropriate filters automatically (recent results, news search)
            - Handle authentication and configuration server-side
            - Return formatted search results for interpretation

            Steps:
            1. Use SerpApiSearch to search for: {query}
            2. Read through the search results carefully  
            3. Provide a clear, accurate answer based on what you find
            4. Cite the sources when providing information

            Important: Base your answer ONLY on what the search results actually show. Do not make assumptions or add information not found in the search results.

            The system uses structured request configurations for reliable, secure search operations."""
        }
    ]

if __name__ == "__main__":
    mcp.run(transport="sse")
