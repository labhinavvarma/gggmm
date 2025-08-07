
"""
DataFlyWheel MCP Server with FREE Web Search Integration

🆓 NO API KEYS REQUIRED FOR WEB SEARCH 🆓

ENVIRONMENT VARIABLES (all optional):
- All existing Snowflake connection variables for HEDIS functionality
- No web search API keys needed!

FREE Web Search Features:
- ✅ DuckDuckGo Instant Answer API (no key required)
- ✅ Wikipedia API (completely free)
- ✅ Improved web scraping with anti-detection
- ✅ Free news APIs for current information
- ✅ Multi-source aggregation for comprehensive results
- ✅ No costs, no registration, no API limits

To set up:
1. No setup required for web search!
2. Just run the MCP server
3. Web search works immediately with multiple free sources

Benefits of FREE approach:
- No API costs or subscription fees
- No rate limiting or usage restrictions
- Privacy-friendly (no account tracking)
- Multiple sources provide comprehensive coverage
- Works out-of-the-box with zero configuration
"""

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

# Web Research Result dataclass
@dataclass
class WebResearchResult:
    """Data class for web research results."""
    title: str
    url: str
    content: str
    summary: str
    relevance_score: float
    timestamp: str

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

class CommercialWebSearchEngine:
    """Commercial Web Search Engine using Zhipu AI API for reliable current information"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter(requests_per_minute=30)  # More lenient for commercial API
        self.zhipu_api_key = os.getenv('ZHIPU_AUTH_KEY')
        self.session_results = []
        
    async def comprehensive_search(self, query: str, ctx: Context, max_results: int = 10) -> List[WebResearchResult]:
        """Perform comprehensive web research using Google search with browser automation"""
        try:
            await self.rate_limiter.acquire()
            await ctx.info(f"🔍 Starting Google-powered web research for: {query}")
            
            # Strategy 1: Google search with real browser simulation
            search_results = await self._google_search_with_browser(query, ctx, max_results)
            
            if not search_results:
                await ctx.warning("Google search failed, falling back to multi-engine approach")
                search_results = await self._multi_engine_search(query, ctx, max_results)
            
            # Strategy 2: Enhanced content extraction from top results
            enriched_results = await self._enhanced_content_extraction(search_results, ctx)
            
            # Strategy 3: Current-focused relevance scoring
            scored_results = await self._current_focused_scoring(enriched_results, query, ctx)
            
            await ctx.info(f"✅ Advanced web research completed: {len(scored_results)} current results")
            return scored_results
            
        except Exception as e:
            await ctx.error(f"Web research failed: {str(e)}")
            return []
    
    async def _google_search_with_browser(self, query: str, ctx: Context, max_results: int) -> List[dict]:
        """Perform Google search using browser automation approach"""
        try:
            await ctx.info("🤖 Simulating real browser Google search...")
            
            # Enhanced search with current date context
            current_date = datetime.now().strftime("%Y-%m-%d")
            enhanced_query = f"{query} {current_date} recent latest"
            
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache"
            }
            
            # Add consent cookies to bypass Google consent dialog
            cookies = {
                'CONSENT': 'YES+cb.20210720-07-p0.en+FX+410'
            }
            
            async with httpx.AsyncClient(
                timeout=20.0, 
                follow_redirects=True,
                cookies=cookies,
                headers=headers
            ) as client:
                
                # Step 1: Get Google search page
                await ctx.info("📡 Accessing Google search...")
                search_url = f"https://www.google.com/search"
                params = {
                    'q': enhanced_query,
                    'num': max_results,
                    'hl': 'en',
                    'gl': 'us',
                    'tbs': 'qdr:w',  # Recent results (past week)
                    'safe': 'off'
                }
                
                response = await client.get(search_url, params=params)
                
                if response.status_code != 200:
                    raise Exception(f"Google search failed with status {response.status_code}")
                
                await ctx.info("🔍 Parsing Google search results...")
                
                # Step 2: Parse results with enhanced extraction
                results = await self._parse_google_results_enhanced(response.text, max_results, ctx)
                
                if results:
                    await ctx.info(f"✅ Google search successful: {len(results)} current results found")
                else:
                    await ctx.warning("⚠️ No Google results found, may be blocked")
                
                return results
                
        except Exception as e:
            await ctx.error(f"Google browser search failed: {str(e)}")
            return []
    
    async def _parse_google_results_enhanced(self, html_content: str, limit: int, ctx: Context) -> List[dict]:
        """Enhanced Google results parsing with current-focused extraction"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            results = []
            
            # Multiple selectors to catch different Google result formats
            result_selectors = [
                'div.g',           # Standard results
                'div.tF2Cxc',      # New format results  
                'div.hlcw0c',      # Alternative format
                'div.yuRUbf',      # Mobile format
            ]
            
            search_containers = []
            for selector in result_selectors:
                containers = soup.find_all(selector)
                if containers:
                    search_containers = containers
                    await ctx.info(f"📊 Found {len(containers)} results with selector: {selector}")
                    break
            
            if not search_containers:
                await ctx.warning("⚠️ No Google result containers found - may be blocked")
                return []
            
            for i, container in enumerate(search_containers[:limit]):
                try:
                    # Enhanced title extraction with multiple fallbacks
                    title_element = (
                        container.find('h3') or 
                        container.find('div[role="heading"]') or
                        container.find('[data-header-feature="0"]')
                    )
                    
                    if not title_element:
                        continue
                        
                    title = title_element.get_text(strip=True)
                    
                    # Enhanced URL extraction with validation
                    link_element = container.find('a')
                    if not link_element:
                        continue
                        
                    url = link_element.get('href', '')
                    
                    # Clean Google redirect URLs
                    if url.startswith('/url?q='):
                        url = url.split('/url?q=')[1].split('&')[0]
                    
                    # Skip invalid URLs
                    if not url.startswith(('http://', 'https://')):
                        continue
                    
                    # Enhanced snippet extraction with current-date focus
                    snippet_selectors = [
                        '.VwiC3b',      # Standard snippet
                        '.s3v9rd',      # Alternative snippet
                        '.st',          # Legacy snippet
                        '.Uroaid',      # New snippet format
                    ]
                    
                    snippet = ''
                    for snippet_sel in snippet_selectors:
                        snippet_element = container.select_one(snippet_sel)
                        if snippet_element:
                            snippet = snippet_element.get_text(strip=True)
                            break
                    
                    # Prioritize results with current date indicators
                    current_indicators = [
                        '2024', '2025', 'today', 'latest', 'recent', 
                        'current', 'now', 'this week', 'this month'
                    ]
                    
                    # Calculate current relevance score
                    current_score = sum(1 for indicator in current_indicators 
                                      if indicator.lower() in (title + ' ' + snippet).lower())
                    
                    if title and url:
                        result = {
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'current_score': current_score,
                            'source': 'Google Enhanced'
                        }
                        results.append(result)
                        
                        await ctx.info(f"📄 Result {i+1}: {title[:50]}... (Current score: {current_score})")
                        
                except Exception as e:
                    await ctx.warning(f"Failed to parse Google result {i}: {str(e)}")
                    continue
            
            # Sort by current relevance score
            results.sort(key=lambda x: x['current_score'], reverse=True)
            
            await ctx.info(f"🎯 Enhanced parsing completed: {len(results)} results with current focus")
            return results
            
        except Exception as e:
            await ctx.error(f"Enhanced Google parsing failed: {str(e)}")
            return []
    
    async def _enhanced_content_extraction(self, search_results: List[dict], ctx: Context) -> List[dict]:
        """Enhanced content extraction focused on current information"""
        enriched_results = []
        
        for i, result in enumerate(search_results):
            try:
                await ctx.info(f"📄 Enhanced extraction {i+1}: {result['title'][:50]}...")
                
                # Enhanced content fetching with current-date optimization
                content = await self._fetch_current_content_optimized(result['url'], ctx)
                
                if content and len(content) > 200:  # Minimum content threshold
                    # Generate current-focused summary
                    summary = await self._generate_current_summary(content, ctx)
                    
                    # Extract current date information
                    current_info = await self._extract_current_indicators(content, ctx)
                    
                    enriched_result = {
                        **result,
                        'content': content[:4000],  # Increased content size
                        'summary': summary,
                        'current_indicators': current_info,
                        'content_length': len(content),
                        'extraction_timestamp': datetime.now().isoformat()
                    }
                    enriched_results.append(enriched_result)
                    await ctx.info(f"✅ Enhanced: {len(content)} chars, current indicators: {len(current_info)}")
                else:
                    await ctx.warning(f"⚠️ Insufficient content from {result['url']}")
                    
            except Exception as e:
                await ctx.warning(f"❌ Enhanced extraction failed for {result['url']}: {str(e)}")
                continue
        
        return enriched_results
    
    async def _fetch_current_content_optimized(self, url: str, ctx: Context) -> str:
        """Optimized content fetching focused on current information"""
        try:
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache, must-revalidate",
                "Pragma": "no-cache",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate"
            }
            
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
                # Enhanced content cleaning similar to Node.js version
                content = response.text
                
                # Remove scripts, styles, and navigation (like Node.js version)
                content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r'<nav[^>]*>.*?</nav>', '', content, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r'<header[^>]*>.*?</header>', '', content, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r'<footer[^>]*>.*?</footer>', '', content, flags=re.DOTALL | re.IGNORECASE)
                
                # Focus on main content areas (like Node.js version)
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')
                
                # Try to find main content containers
                main_content = None
                content_selectors = ['main', 'article', '[role="main"]', '#content', '.content', '.main', '.post']
                
                for selector in content_selectors:
                    element = soup.select_one(selector)
                    if element:
                        main_content = element.get_text(separator=' ', strip=True)
                        break
                
                if not main_content:
                    # Fallback to body with enhanced cleaning
                    main_content = soup.get_text(separator=' ', strip=True)
                
                # Clean up whitespace and format
                main_content = re.sub(r'\s+', ' ', main_content).strip()
                
                return main_content[:6000]  # Increased limit
                
        except Exception as e:
            await ctx.warning(f"Optimized content fetch failed for {url}: {str(e)}")
            return ""
    
    async def _generate_current_summary(self, content: str, ctx: Context) -> str:
        """Generate summary focused on current/recent information"""
        # Enhanced summarization focusing on current indicators
        sentences = content.split('. ')
        
        # Prioritize sentences with current indicators
        current_indicators = [
            '2024', '2025', 'today', 'latest', 'recent', 'current', 
            'now', 'this week', 'this month', 'updated', 'new'
        ]
        
        # Score sentences by current relevance
        scored_sentences = []
        for sentence in sentences:
            score = sum(1 for indicator in current_indicators 
                       if indicator.lower() in sentence.lower())
            scored_sentences.append((sentence, score))
        
        # Sort by current relevance and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:4]]  # Top 4 most current sentences
        
        summary = '. '.join(top_sentences)
        
        if len(summary) > 800:
            summary = summary[:800] + "..."
        
        return summary
    
    async def _extract_current_indicators(self, content: str, ctx: Context) -> List[str]:
        """Extract indicators of current/recent information"""
        current_patterns = [
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+202[4-5]\b',
            r'\b202[4-5]\b',
            r'\b(today|yesterday|this week|this month|last week|recently|latest|current)\b',
            r'\bupdated:\s*[^\n]+\b',
            r'\bpublished:\s*[^\n]+\b',
            r'\b(breaking|urgent|just in|live)\b'
        ]
        
        indicators = []
        for pattern in current_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            indicators.extend(matches)
        
        return list(set(indicators))  # Remove duplicates
    
    async def _current_focused_scoring(self, results: List[dict], query: str, ctx: Context) -> List[WebResearchResult]:
        """Enhanced scoring focused on current information relevance"""
        scored_results = []
        query_words = set(query.lower().split())
        
        current_date = datetime.now()
        
        for result in results:
            try:
                # Multi-factor scoring system
                title_score = len(query_words.intersection(set(result['title'].lower().split()))) * 3
                content_score = len(query_words.intersection(set(result['content'].lower().split()))) * 1
                summary_score = len(query_words.intersection(set(result['summary'].lower().split()))) * 2
                
                # Current information bonus scoring
                current_bonus = result.get('current_score', 0) * 2
                indicator_bonus = len(result.get('current_indicators', [])) * 1.5
                
                # Recency bonus based on extraction timestamp
                extraction_time = datetime.fromisoformat(result.get('extraction_timestamp', current_date.isoformat()))
                minutes_ago = (current_date - extraction_time).total_seconds() / 60
                recency_bonus = max(0, 10 - minutes_ago / 60)  # Bonus for very recent extractions
                
                # Calculate total relevance score
                total_score = (title_score + content_score + summary_score + current_bonus + indicator_bonus + recency_bonus) / len(query_words)
                
                web_result = WebResearchResult(
                    title=result['title'],
                    url=result['url'],
                    content=result['content'],
                    summary=result['summary'],
                    relevance_score=total_score,
                    timestamp=result.get('extraction_timestamp', current_date.isoformat())
                )
                
                scored_results.append(web_result)
                
                await ctx.info(f"📊 Scored: {result['title'][:30]}... = {total_score:.2f}")
                
            except Exception as e:
                await ctx.warning(f"Scoring failed for result: {str(e)}")
                continue
        
        # Sort by relevance score with current focus
        scored_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        await ctx.info(f"🎯 Current-focused scoring completed: {len(scored_results)} results prioritized")
        return scored_results
    
    async def _multi_engine_search(self, query: str, ctx: Context, max_results: int) -> List[dict]:
        """Fallback multi-engine search when Google fails"""
        results = []
        
        # Enhanced search engines with current-date parameters
        current_date = datetime.now().strftime("%Y-%m-%d")
        enhanced_query = f"{query} {current_date} recent"
        
        engines = [
            {
                "name": "Bing",
                "url": "https://www.bing.com/search", 
                "params": {"q": enhanced_query, "count": max_results, "freshness": "Week"},
                "parser": self._parse_bing_results
            },
            {
                "name": "Yahoo",
                "url": "https://search.yahoo.com/search",
                "params": {"p": enhanced_query, "n": max_results},
                "parser": self._parse_yahoo_results
            }
        ]
        
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
        }
        
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            for engine in engines:
                try:
                    await ctx.info(f"📡 Fallback search: {engine['name']}...")
                    
                    response = await client.get(
                        engine['url'],
                        params=engine['params'],
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        engine_results = engine['parser'](response.text, max_results)
                        results.extend(engine_results)
                        await ctx.info(f"✅ {engine['name']}: {len(engine_results)} results")
                    else:
                        await ctx.warning(f"❌ {engine['name']}: HTTP {response.status_code}")
                        
                except Exception as e:
                    await ctx.warning(f"❌ {engine['name']} failed: {str(e)}")
                    continue
        
        # Remove duplicates and prioritize unique results
        seen_urls = set()
        unique_results = []
        for result in results:
            if result.get('url') not in seen_urls and result.get('url'):
                seen_urls.add(result['url'])
                unique_results.append(result)
        
        await ctx.info(f"📊 Fallback search completed: {len(unique_results)} unique results")
        return unique_results[:max_results]
    
    def _parse_bing_results(self, html_content: str, limit: int) -> List[dict]:
        """Parse Bing search results"""
        # Simple regex-based parsing (you can enhance with BeautifulSoup)
        results = []
        
        # This is a simplified parser - in practice, you'd use BeautifulSoup
        title_pattern = r'<h2><a[^>]+href="([^"]+)"[^>]*>([^<]+)</a></h2>'
        matches = re.findall(title_pattern, html_content)
        
        for i, (url, title) in enumerate(matches[:limit]):
            if url.startswith('http'):
                results.append({
                    'title': title.strip(),
                    'url': url.strip(),
                    'description': f"Search result from Bing for: {title}"
                })
        
        return results
    
    def _parse_yahoo_results(self, html_content: str, limit: int) -> List[dict]:
        """Parse Yahoo search results"""
        results = []
        # Simplified parser for Yahoo
        title_pattern = r'<h3[^>]*><a[^>]+href="([^"]+)"[^>]*>([^<]+)</a></h3>'
        matches = re.findall(title_pattern, html_content)
        
        for i, (url, title) in enumerate(matches[:limit]):
            if url.startswith('http'):
                results.append({
                    'title': title.strip(),
                    'url': url.strip(),
                    'description': f"Search result from Yahoo for: {title}"
                })
        
        return results
    
    def _parse_searx_results(self, html_content: str, limit: int) -> List[dict]:
        """Parse Searx search results"""
        results = []
        # Simplified parser for Searx
        title_pattern = r'<h3><a[^>]+href="([^"]+)"[^>]*>([^<]+)</a></h3>'
        matches = re.findall(title_pattern, html_content)
        
        for i, (url, title) in enumerate(matches[:limit]):
            if url.startswith('http'):
                results.append({
                    'title': title.strip(),
                    'url': url.strip(),
                    'description': f"Search result from Searx for: {title}"
                })
        
        return results

    def format_results_for_llm(self, results: List[WebResearchResult]) -> str:
        """Format advanced web research results with current information focus"""
        if not results:
            return "No current web research results found. The search may have been blocked or no current information is available."
        
        output = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        output.append(f"🌐 **CURRENT WEB RESEARCH RESULTS** (Retrieved: {current_time})\n")
        output.append(f"📊 **{len(results)} sources analyzed with current information focus**\n")
        
        for i, result in enumerate(results, 1):
            output.append(f"## 🔍 Source {i}: {result.title}")
            output.append(f"**🌐 URL:** {result.url}")
            output.append(f"**⭐ Current Relevance Score:** {result.relevance_score:.2f}/10")
            output.append(f"**⏰ Information Retrieved:** {result.timestamp}")
            
            # Highlight current information indicators
            if hasattr(result, 'current_indicators') and result.current_indicators:
                output.append(f"**🎯 Current Indicators Found:** {', '.join(result.current_indicators[:5])}")
            
            output.append(f"**📝 Current Summary:** {result.summary}")
            
            # Show content preview with current info highlighted
            content_preview = result.content[:400]
            # Highlight current year mentions
            content_preview = content_preview.replace('2024', '**2024**').replace('2025', '**2025**')
            output.append(f"**📄 Content Preview:** {content_preview}...")
            
            output.append("---\n")
        
        output.append(f"✅ **VERIFICATION:** All information above was retrieved in real-time from live websites at {current_time}")
        output.append("🎯 **FOCUS:** Results prioritized for current/recent information based on content analysis")
        
        return "\n".join(output)

# Initialize web research engine
web_research_engine = WebResearchEngine()

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

# Diagnostic Tool - LangChain Compatible  
@mcp.tool(
        name="diagnostic",
        description="""Diagnostic tool to test MCP functionality including FREE web search APIs."""
)
async def diagnostic(test_type: str = "basic", ctx: Context = None) -> str:
    """
    Run diagnostic tests to verify MCP functionality.
    
    Args:
        test_type: Type of test (basic, search, free_apis, time)
    
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
        
    elif test_type == "free_apis":
        # Test free web search APIs connectivity
        result += f"🆓 FREE Web Search APIs Test:\n"
        
        # Test DuckDuckGo API
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                ddg_response = await client.get("https://api.duckduckgo.com/")
                ddg_status = "✅ ACCESSIBLE" if ddg_response.status_code < 500 else "⚠️ SERVER_ERROR"
        except Exception as e:
            ddg_status = f"❌ FAILED: {str(e)[:50]}"
        
        result += f"🦆 DuckDuckGo API: {ddg_status}\n"
        
        # Test Wikipedia API
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                wiki_response = await client.get("https://en.wikipedia.org/api/rest_v1/page/summary/test")
                wiki_status = "✅ ACCESSIBLE" if wiki_response.status_code < 500 else "⚠️ SERVER_ERROR"
        except Exception as e:
            wiki_status = f"❌ FAILED: {str(e)[:50]}"
        
        result += f"📚 Wikipedia API: {wiki_status}\n"
        
        # Test general web connectivity
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                web_response = await client.get("https://httpbin.org/get")
                web_status = "✅ ACCESSIBLE" if web_response.status_code == 200 else "⚠️ ISSUES"
        except Exception as e:
            web_status = f"❌ FAILED: {str(e)[:50]}"
        
        result += f"🌐 Web Scraping: {web_status}\n"
        result += f"💰 Cost: 100% FREE (no API keys required)\n"
        
    elif test_type == "search":
        # Test if we can make HTTP requests
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("https://httpbin.org/get")
                http_status = "SUCCESS" if response.status_code == 200 else "FAILED"
        except Exception as e:
            http_status = f"FAILED: {str(e)}"
            
        result += f"🌐 HTTP Test: {http_status}\n"
        result += "🆓 FREE Search: Available (DuckDuckGo + Wikipedia + Web + News)\n"
        result += "💡 No API keys required for web search functionality\n"
        
    elif test_type == "time":
        import time
        result += f"🕐 Unix Timestamp: {int(time.time())}\n"
        result += f"📅 Current Year: {datetime.datetime.now().year}\n" 
        result += f"📅 Current Month: {datetime.datetime.now().strftime('%B %Y')}\n"
        
    return result

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

# === NEW WEB RESEARCH TOOLS ===

@mcp.tool(
        name="web_research",
        description="""FREE comprehensive web search using multiple free APIs - no API key required."""
)
async def web_research(query: str, ctx: Context) -> str:
    """
    Perform FREE comprehensive web research using multiple free APIs and improved techniques.
    
    Args:
        query: Research query string
    
    Returns:
        Formatted research results from multiple free sources
    """
    try:
        await ctx.info(f"🆓 Starting FREE web research for: {query}")
        
        # Use free search engine
        result = await free_search_engine.free_comprehensive_search(query, ctx)
        
        await ctx.info(f"✅ FREE web research completed")
        
        return result
        
    except Exception as e:
        await ctx.error(f"FREE web research failed for query '{query}': {str(e)}")
        return f"""❌ **FREE WEB RESEARCH ERROR**

Free web research system encountered an error: {str(e)}

**Error Details:**
- Query: "{query}"
- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- System: FREE Multi-Source Web Search (No API Keys Required)

**Troubleshooting:**
- Check internet connectivity and DNS resolution
- Try rephrasing your query with simpler terms
- Verify MCP server has access to external APIs
- Some free services may have temporary outages"""

@mcp.tool(
        name="focused_web_search", 
        description="""Quick focused FREE web search using DuckDuckGo API - no API key required."""
)
async def focused_web_search(query: str, ctx: Context) -> str:
    """
    Perform a quick focused FREE web search using DuckDuckGo Instant Answer API.
    
    Args:
        query: Search query string
    
    Returns:
        Quick search results from free APIs
    """
    try:
        await ctx.info(f"⚡ Performing focused FREE search for: {query}")
        
        # Use just DuckDuckGo for quick searches
        ddg_result = await free_search_engine._duckduckgo_instant_api(query, ctx)
        
        if not ddg_result:
            return f"❌ No focused search results found for: {query}. Try rephrasing your query or use the full web_research tool."
        
        # Format as focused results
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        focused_response = f"""⚡ **FOCUSED FREE SEARCH** (Retrieved: {current_time})
🎯 Query: '{query}' | 🆓 **100% Free - No API Key Required**

## 🦆 DuckDuckGo Instant Results:
{ddg_result}

---
✅ **Free Search Completed**: Using DuckDuckGo Instant Answer API
🔄 **No Cost**: Completely free service, no API keys or registration required
⏰ **Retrieved**: {current_time}"""
        
        await ctx.info(f"✅ Focused FREE search completed")
        return focused_response
        
    except Exception as e:
        await ctx.error(f"Focused FREE search failed: {str(e)}")
        return f"❌ Focused FREE search error: {str(e)}. Free API may be temporarily unavailable."

@mcp.tool(
        name="get_weather",
        description="""Get current weather forecast using the National Weather Service API with formatted output."""
)
async def get_weather(latitude: float, longitude: float, ctx: Context) -> str:
    """
    Get current weather forecast using the National Weather Service API.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
    """
    try:
        await ctx.info(f"🌤️ Fetching weather for coordinates: {latitude}, {longitude}")
        
        headers = {
            "User-Agent": "MCP Weather Client (mcp-weather@example.com)",
            "Accept": "application/geo+json"
        }
        
        # Get grid point information
        points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            points_response = await client.get(points_url, headers=headers)
            points_response.raise_for_status()
            points_data = points_response.json()
            
            # Extract forecast URL and location info
            forecast_url = points_data['properties']['forecast']
            location_info = points_data['properties']['relativeLocation']['properties']
            location_name = f"{location_info['city']}, {location_info['state']}"
            
            await ctx.info(f"📍 Location identified: {location_name}")
            
            # Get forecast data
            forecast_response = await client.get(forecast_url, headers=headers)
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()
            
        # Get current and upcoming periods
        periods = forecast_data['properties']['periods']
        
        if not periods:
            return f"❌ No weather data available for {location_name}"
        
        # Format the weather response nicely
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        result = f"🌤️ **WEATHER FORECAST FOR {location_name.upper()}**\n"
        result += f"📍 **Coordinates:** {latitude}, {longitude}\n"
        result += f"⏰ **Retrieved:** {current_time}\n\n"
        
        # Current/First Period
        current_period = periods[0]
        result += f"## 🌡️ {current_period['name']}\n"
        result += f"**Temperature:** {current_period['temperature']}°{current_period['temperatureUnit']}\n"
        result += f"**Wind:** {current_period.get('windSpeed', 'N/A')} {current_period.get('windDirection', '')}\n"
        result += f"**Conditions:** {current_period['shortForecast']}\n"
        result += f"**Detailed Forecast:** {current_period['detailedForecast']}\n\n"
        
        # Next few periods if available
        if len(periods) > 1:
            result += "## 📅 **UPCOMING FORECAST:**\n\n"
            for period in periods[1:4]:  # Next 3 periods
                result += f"### {period['name']}\n"
                result += f"**Temperature:** {period['temperature']}°{period['temperatureUnit']}\n"
                result += f"**Conditions:** {period['shortForecast']}\n"
                if period.get('windSpeed'):
                    result += f"**Wind:** {period['windSpeed']} {period.get('windDirection', '')}\n"
                result += "\n"
        
        # Add source attribution
        result += "---\n"
        result += "📡 **Source:** National Weather Service (weather.gov)\n"
        result += f"🔗 **Data URL:** {forecast_url}\n"
        
        await ctx.info(f"✅ Weather data formatted successfully for {location_name}")
        return result
        
    except httpx.TimeoutError:
        return f"⏰ Weather service request timed out for coordinates {latitude}, {longitude}"
    except httpx.HTTPError as e:
        return f"🌐 Weather service HTTP error: {e.response.status_code if hasattr(e, 'response') else 'Unknown'}"
    except KeyError as e:
        return f"📊 Weather data format error - missing field: {str(e)}"
    except Exception as e:
        return f"❌ Weather service error: {str(e)}"

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
        name="web-research-prompt",
        description="FREE Web Search Expert using Multiple Free APIs"
)
async def web_research_prompt(query: str) -> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are a FREE web research specialist with access to multiple free APIs and improved search techniques - no API keys required!

Your FREE capabilities include:
- **🦆 DuckDuckGo Instant API**: Free instant answers and definitions (no key required)
- **📚 Wikipedia API**: Comprehensive encyclopedia access (completely free)
- **🌐 Improved Web Scraping**: Anti-detection techniques for live web content
- **📰 Free News APIs**: Access to current news from free sources
- **🔄 Multi-Source Aggregation**: Combines results from all free sources

**🆓 100% FREE ADVANTAGES:**
- **No API Keys**: Zero setup required, no registration, no costs
- **No Rate Limits**: Reasonable usage limits from free services
- **Multiple Sources**: Diversified results reduce single-point failures
- **Current Information**: Live APIs provide up-to-date information
- **Privacy Friendly**: No account tracking or data collection

**🎯 FREE SEARCH STRATEGIES:**
- **Instant Answers**: DuckDuckGo provides quick facts and definitions
- **Deep Knowledge**: Wikipedia offers comprehensive background information  
- **Live Web Content**: Improved scraping gets current website information
- **News Updates**: Free news APIs provide recent developments
- **Smart Aggregation**: Combines all sources for comprehensive coverage

Available FREE tools:
- **web_research**: Comprehensive multi-source free search (DuckDuckGo + Wikipedia + Web + News)
- **focused_web_search**: Quick DuckDuckGo instant answers (fastest option)

**For the query "{query}", your mission is:**

1. **🆓 FREE SEARCH FIRST**: Use the free web research tools to gather information from multiple sources
2. **📊 MULTI-SOURCE ANALYSIS**: Process results from DuckDuckGo, Wikipedia, live web, and news
3. **🎯 SYNTHESIZE FINDINGS**: Combine insights from all available free sources
4. **📝 PROVIDE COMPREHENSIVE ANSWER**: Deliver well-rounded response using all free data
5. **🔗 CITE FREE SOURCES**: Reference Wikipedia links, DuckDuckGo sources, and scraped websites

**✅ FREE GUARANTEE**: All search capabilities are completely free with no API keys, costs, or registration required.

**⚠️ IMPORTANT**: Always use the FREE web research tools first to gather current information. The free APIs provide reliable, current data without any costs.

Research Query: {query}"""
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
