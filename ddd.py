from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import httpx
from dataclasses import dataclass, field
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
import base64
import tempfile
import shutil
from datetime import datetime, timedelta
from snowflake.connector import SnowflakeConnection
from ReduceReuseRecycleGENAI.snowflake import snowflake_conn
from snowflake.connector.errors import DatabaseError
from snowflake.core import Root
from typing import Optional, List, Dict, Any
from fastapi import HTTPException, status, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp.prompts.base import Message
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts import Prompt
import mcp.types as types
from functools import partial
import sys
import traceback
import statistics
import ast
import time

# Playwright imports for browser automation
try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Browser = Any
    Page = Any
    BrowserContext = Any

# HTML to Markdown conversion
try:
    import markdownify
    MARKDOWNIFY_AVAILABLE = True
except ImportError:
    MARKDOWNIFY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
NWS_API_BASE = "https://api.weather.gov"
HOST = "carelon-eda-preprod.privatelink.snowflakecomputing.com"
MAX_RESULTS_PER_SESSION = 100
MAX_RETRIES = 3
RETRY_DELAY = 1000
SCREENSHOTS_DIR = None

@dataclass
class AppContext:
    conn: Optional[SnowflakeConnection] = None
    db: str = 'POC_SPC_SNOWPARK_DB'
    schema: str = 'HEDIS_SCHEMA'
    host: str = HOST

@dataclass
class ResearchResult:
    url: str
    title: str
    content: str
    timestamp: str
    screenshot_path: Optional[str] = None

@dataclass
class ResearchSession:
    query: str
    results: List[ResearchResult] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

class WebResearchBrowser:
    """Browser automation class using Playwright"""
    
    def __init__(self):
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.screenshots_dir = None
        
    async def initialize(self):
        """Initialize Playwright browser"""
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright not available. Install with: pip install playwright")
        
        try:
            # Create screenshots directory
            self.screenshots_dir = tempfile.mkdtemp(prefix='mcp-screenshots-')
            
            # Initialize Playwright
            self.playwright = await async_playwright().start()
            
            # Launch browser with optimized settings
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-features=VizDisplayCompositor',
                    '--disable-extensions',
                    '--disable-plugins',
                    '--disable-images',  # Faster loading
                    '--disable-javascript',  # Security and speed (we'll enable when needed)
                ]
            )
            
            # Create browser context with realistic user agent
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            
            # Create initial page
            self.page = await self.context.new_page()
            
            logger.info("✅ Web research browser initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize browser: {e}")
            raise

    async def cleanup(self):
        """Clean up browser resources"""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            
            # Clean up screenshots directory
            if self.screenshots_dir and os.path.exists(self.screenshots_dir):
                shutil.rmtree(self.screenshots_dir)
                
            logger.info("✅ Browser cleanup completed")
            
        except Exception as e:
            logger.error(f"⚠️ Error during browser cleanup: {e}")

    async def ensure_page(self) -> Page:
        """Ensure we have a valid page"""
        if not self.page or self.page.is_closed():
            if not self.context:
                await self.initialize()
            self.page = await self.context.new_page()
        return self.page

    async def dismiss_google_consent(self, page: Page):
        """Handle Google consent dialogs"""
        try:
            # Wait a bit for any consent dialogs to appear
            await asyncio.sleep(2)
            
            # Look for common consent dialog patterns
            consent_selectors = [
                'button[aria-label*="accept"]',
                'button[aria-label*="Accept"]',
                'button:has-text("Accept all")',
                'button:has-text("I agree")',
                'button:has-text("Accept")',
                'div[role="button"]:has-text("Accept")',
                '#L2AGLb',  # Google's "I agree" button
                'button[jsname="VfPpkd-LgbsSe"]'  # Google Material Design button
            ]
            
            for selector in consent_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element and await element.is_visible():
                        await element.click()
                        logger.info(f"✅ Clicked consent button: {selector}")
                        await asyncio.sleep(1)
                        break
                except:
                    continue
                    
        except Exception as e:
            logger.warning(f"⚠️ Consent handling failed: {e}")

    async def safe_navigation(self, page: Page, url: str, wait_for_load: bool = True):
        """Safely navigate to a URL with error handling"""
        try:
            # Set bypass cookies for common consent systems
            await page.context.add_cookies([
                {
                    'name': 'CONSENT',
                    'value': 'YES+',
                    'domain': '.google.com',
                    'path': '/'
                }
            ])
            
            response = await page.goto(url, timeout=30000, wait_until='domcontentloaded')
            
            if not response:
                raise Exception("No response received")
                
            if response.status >= 400:
                raise Exception(f"HTTP {response.status}: {response.status_text}")
            
            if wait_for_load:
                # Wait for page to be ready
                try:
                    await page.wait_for_load_state('networkidle', timeout=10000)
                except:
                    pass  # Continue even if networkidle times out
            
            # Handle consent dialogs
            await self.dismiss_google_consent(page)
            
            # Basic bot detection check
            page_title = await page.title()
            suspicious_titles = ['security check', 'ddos protection', 'please wait', 'just a moment']
            if any(phrase in page_title.lower() for phrase in suspicious_titles):
                raise Exception(f"Bot protection detected: {page_title}")
            
            logger.info(f"✅ Successfully navigated to: {url}")
            
        except Exception as e:
            logger.error(f"❌ Navigation failed for {url}: {e}")
            raise

    async def extract_content_as_markdown(self, page: Page, selector: Optional[str] = None) -> str:
        """Extract page content and convert to markdown"""
        try:
            # Execute content extraction in browser
            html_content = await page.evaluate("""
                (selector) => {
                    // If specific selector provided, use it
                    if (selector) {
                        const element = document.querySelector(selector);
                        return element ? element.outerHTML : '';
                    }
                    
                    // Try standard content containers
                    const contentSelectors = [
                        'main', 'article', '[role="main"]', 
                        '#content', '.content', '.main', 
                        '.post', '.article'
                    ];
                    
                    for (const contentSelector of contentSelectors) {
                        const element = document.querySelector(contentSelector);
                        if (element) {
                            return element.outerHTML;
                        }
                    }
                    
                    // Fallback: clean body content
                    const body = document.body.cloneNode(true);
                    
                    // Remove unwanted elements
                    const unwantedSelectors = [
                        'header', 'footer', 'nav', '[role="navigation"]',
                        'aside', '.sidebar', '[role="complementary"]',
                        '.nav', '.menu', '.header', '.footer',
                        '.advertisement', '.ads', '.cookie-notice',
                        'script', 'style', 'noscript'
                    ];
                    
                    unwantedSelectors.forEach(sel => {
                        body.querySelectorAll(sel).forEach(el => el.remove());
                    });
                    
                    return body.outerHTML;
                }
            """, selector)
            
            if not html_content:
                return ""
            
            # Convert HTML to Markdown
            if MARKDOWNIFY_AVAILABLE:
                markdown = markdownify.markdownify(
                    html_content, 
                    heading_style='atx',
                    bullets='-',
                    strip=['script', 'style']
                )
            else:
                # Simple fallback - extract text content
                text_content = await page.evaluate("""
                    (html) => {
                        const div = document.createElement('div');
                        div.innerHTML = html;
                        return div.textContent || div.innerText || '';
                    }
                """, html_content)
                markdown = text_content
            
            # Clean up markdown
            if markdown:
                markdown = re.sub(r'\n{3,}', '\n\n', markdown)  # Max 2 newlines
                markdown = re.sub(r'^- $', '', markdown, flags=re.MULTILINE)  # Remove empty list items
                markdown = re.sub(r'^\s+$', '', markdown, flags=re.MULTILINE)  # Remove whitespace-only lines
                markdown = markdown.strip()
            
            return markdown
            
        except Exception as e:
            logger.error(f"❌ Content extraction failed: {e}")
            # Fallback: get page text
            try:
                text_content = await page.inner_text('body')
                return text_content[:5000] + "..." if len(text_content) > 5000 else text_content
            except:
                return ""

    async def take_screenshot(self, page: Page, title: str = "screenshot") -> Optional[str]:
        """Take a screenshot and save it"""
        try:
            if not self.screenshots_dir:
                return None
                
            # Generate safe filename
            safe_title = re.sub(r'[^a-zA-Z0-9]', '_', title)[:50]
            timestamp = int(datetime.now().timestamp())
            filename = f"{safe_title}_{timestamp}.png"
            filepath = os.path.join(self.screenshots_dir, filename)
            
            # Take screenshot
            await page.screenshot(path=filepath, full_page=False)
            
            logger.info(f"✅ Screenshot saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Screenshot failed: {e}")
            return None

# Global browser instance
web_browser = WebResearchBrowser()
current_session: Optional[ResearchSession] = None

def add_research_result(result: ResearchResult):
    """Add result to current research session"""
    global current_session
    
    if not current_session:
        current_session = ResearchSession(
            query="Research Session",
            results=[],
            last_updated=datetime.now().isoformat()
        )
    
    if len(current_session.results) >= MAX_RESULTS_PER_SESSION:
        current_session.results.pop(0)  # Remove oldest
    
    current_session.results.append(result)
    current_session.last_updated = datetime.now().isoformat()

async def retry_operation(operation, max_retries: int = MAX_RETRIES, delay: float = 1.0):
    """Generic retry mechanism"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return await operation()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_retries} attempts failed")
    
    raise last_error

def is_valid_url(url_string: str) -> bool:
    """Validate URL format"""
    try:
        url = urllib.parse.urlparse(url_string)
        return url.scheme in ['http', 'https'] and url.netloc
    except:
        return False

def get_snowflake_connection():
    """Get Snowflake connection - simplified for demo"""
    try:
        # In production, use proper connection management
        conn = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database='POC_SPC_SNOWPARK_DB',
            schema='HEDIS_SCHEMA'
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to Snowflake: {e}")
        return None

# Initialize FastAPI app and MCP server
app = FastAPI(title="Web Research Enhanced MCP Server")
mcp = FastMCP("WebResearch-Enhanced-DataFlyWheel", app=app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Categorized Prompt Library
PROMPT_LIBRARY = {
    "hedis": [
        {"name": "Explain BCS Measure", "prompt": "Explain the purpose of the BCS (Breast Cancer Screening) HEDIS measure, including age criteria and methodology."},
        {"name": "List 2024 HEDIS Measures", "prompt": "List all HEDIS measures for the year 2024 with their abbreviations and brief descriptions."},
        {"name": "Age Criteria for CBP", "prompt": "What is the age criteria for the CBP (Controlling High Blood Pressure) measure?"},
        {"name": "HEDIS Measure Categories", "prompt": "What are the main categories of HEDIS measures and how are they organized?"},
        {"name": "Value Set Explanation", "prompt": "Explain what value sets are in HEDIS and how they're used in measure calculations."}
    ],
    "contract": [
        {"name": "Summarize Contract Performance", "prompt": "Summarize the key performance metrics for a specific contract."},
        {"name": "Compare Contract Performance", "prompt": "Compare performance metrics between multiple contracts."},
        {"name": "Contract Compliance Analysis", "prompt": "Analyze contract compliance rates and identify areas for improvement."}
    ],
    "analytics": [
        {"name": "Statistical Analysis", "prompt": "Perform statistical analysis on the provided dataset including mean, median, and standard deviation."},
        {"name": "Trend Analysis", "prompt": "Analyze trends in the data over time and identify patterns."},
        {"name": "Outlier Detection", "prompt": "Identify outliers in the dataset and provide recommendations."}
    ],
    "weather": [
        {"name": "Current Weather", "prompt": "Get current weather conditions for a specific location using coordinates."},
        {"name": "Weather Forecast", "prompt": "Provide detailed weather forecast for the next few days."},
        {"name": "Weather Impact Analysis", "prompt": "Analyze how weather conditions might impact operations or activities."}
    ],
    "web_research": [
        {"name": "Topic Research", "prompt": "Conduct comprehensive research on a specific topic using web sources."},
        {"name": "Company Analysis", "prompt": "Research and analyze a company's current status, news, and market position."},
        {"name": "Technology Trends", "prompt": "Research current trends and developments in a specific technology area."},
        {"name": "Market Research", "prompt": "Conduct market research on a specific industry or product category."}
    ]
}

# === RESOURCES ===

@mcp.resource(uri="schematiclayer://cortex_analyst/schematic_models/{stagename}/list", 
              name="hedis_schematic_models", 
              description="HEDIS Schematic models from Snowflake Cortex")
async def get_schematic_model(stagename: str):
    """Cortex analyst schematic layer model, model is in yaml format"""
    try:
        conn = get_snowflake_connection()
        if not conn:
            return {"error": "Failed to connect to Snowflake"}
        
        db = 'POC_SPC_SNOWPARK_DB'
        schema = 'HEDIS_SCHEMA'
        cursor = conn.cursor()
        
        snfw_model_list = cursor.execute(f"LIST @{db}.{schema}.{stagename}")
        models = [stg_nm[0].split("/")[-1] for stg_nm in snfw_model_list if stg_nm[0].endswith('yaml')]
        
        cursor.close()
        conn.close()
        
        return {"models": models, "stage": stagename}
    except Exception as e:
        logger.error(f"Error getting schematic models: {e}")
        return {"error": str(e)}

@mcp.resource("search://cortex_search/search_obj/list")
async def get_search_service():
    """Cortex search service"""
    try:
        conn = get_snowflake_connection()
        if not conn:
            return {"error": "Failed to connect to Snowflake"}
        
        db = 'POC_SPC_SNOWPARK_DB'
        schema = 'HEDIS_SCHEMA'
        cursor = conn.cursor()
        
        snfw_search_objs = cursor.execute(f"SHOW CORTEX SEARCH SERVICES IN SCHEMA {db}.{schema}")
        result = [search_obj[1] for search_obj in snfw_search_objs.fetchall()]
        
        cursor.close()
        conn.close()
        
        return {"search_services": result}
    except Exception as e:
        logger.error(f"Error getting search services: {e}")
        return {"error": str(e)}

@mcp.resource("research://current/summary")
async def get_research_summary():
    """Get current research session summary"""
    try:
        if not current_session:
            return {"error": "No active research session"}
        
        return {
            "query": current_session.query,
            "result_count": len(current_session.results),
            "last_updated": current_session.last_updated,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "timestamp": r.timestamp,
                    "has_screenshot": r.screenshot_path is not None
                }
                for r in current_session.results
            ]
        }
    except Exception as e:
        logger.error(f"Error getting research summary: {e}")
        return {"error": str(e)}

# === TOOLS ===

@mcp.tool(name="search_google", description="Search Google for information and return results")
async def search_google(query: str, ctx: Context, max_results: int = 10) -> str:
    """
    Search Google for current information.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
    """
    try:
        await ctx.info(f"🔍 Starting Google search for: {query}")
        
        page = await web_browser.ensure_page()
        
        async def perform_search():
            # Navigate to Google
            await web_browser.safe_navigation(page, 'https://www.google.com')
            
            # Find and interact with search input
            search_input = await page.wait_for_selector('input[name="q"], textarea[name="q"]', timeout=10000)
            await search_input.click()
            await search_input.fill(query)
            await search_input.press('Enter')
            
            # Wait for results
            await page.wait_for_selector('div.g', timeout=15000)
            
            # Extract search results
            results = await page.evaluate("""
                () => {
                    const elements = document.querySelectorAll('div.g');
                    return Array.from(elements).slice(0, 10).map(el => {
                        const titleEl = el.querySelector('h3');
                        const linkEl = el.querySelector('a');
                        const snippetEl = el.querySelector('div.VwiC3b, .IsZvec');
                        
                        if (!titleEl || !linkEl) return null;
                        
                        return {
                            title: titleEl.textContent || '',
                            url: linkEl.href || '',
                            snippet: snippetEl ? snippetEl.textContent || '' : ''
                        };
                    }).filter(result => result && result.url && result.title);
                }
            """)
            
            return results[:max_results]
        
        search_results = await retry_operation(perform_search)
        
        # Store results in session
        for result in search_results:
            add_research_result(ResearchResult(
                url=result['url'],
                title=result['title'],
                content=result['snippet'],
                timestamp=datetime.now().isoformat()
            ))
        
        await ctx.info(f"✅ Found {len(search_results)} search results")
        
        # Format results
        formatted_results = {
            "search_query": query,
            "results_count": len(search_results),
            "results": search_results,
            "timestamp": datetime.now().isoformat()
        }
        
        return json.dumps(formatted_results, indent=2)
        
    except Exception as e:
        error_msg = f"❌ Google search failed: {str(e)}"
        await ctx.error(error_msg)
        return json.dumps({"error": error_msg, "query": query})

@mcp.tool(name="visit_page", description="Visit a webpage and extract its content")
async def visit_page(url: str, ctx: Context, take_screenshot: bool = False) -> str:
    """
    Visit a webpage and extract its content.
    
    Args:
        url: URL to visit
        take_screenshot: Whether to take a screenshot
    """
    try:
        if not is_valid_url(url):
            return json.dumps({"error": f"Invalid URL: {url}"})
        
        await ctx.info(f"🌐 Visiting page: {url}")
        
        page = await web_browser.ensure_page()
        
        async def visit_and_extract():
            # Navigate to the page
            await web_browser.safe_navigation(page, url)
            
            # Get page title
            title = await page.title()
            
            # Extract content
            content = await web_browser.extract_content_as_markdown(page)
            
            # Take screenshot if requested
            screenshot_path = None
            if take_screenshot:
                screenshot_path = await web_browser.take_screenshot(page, title)
            
            return {
                "title": title,
                "content": content,
                "screenshot_path": screenshot_path
            }
        
        result = await retry_operation(visit_and_extract)
        
        # Create research result
        research_result = ResearchResult(
            url=url,
            title=result["title"],
            content=result["content"],
            timestamp=datetime.now().isoformat(),
            screenshot_path=result["screenshot_path"]
        )
        
        add_research_result(research_result)
        
        await ctx.info(f"✅ Successfully extracted content from: {url}")
        
        response = {
            "url": url,
            "title": result["title"],
            "content": result["content"],
            "content_length": len(result["content"]),
            "timestamp": research_result.timestamp,
            "screenshot_taken": result["screenshot_path"] is not None
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        error_msg = f"❌ Failed to visit page {url}: {str(e)}"
        await ctx.error(error_msg)
        return json.dumps({"error": error_msg, "url": url})

@mcp.tool(name="take_screenshot", description="Take a screenshot of the current page")
async def take_screenshot_tool(ctx: Context) -> str:
    """
    Take a screenshot of the current page.
    """
    try:
        await ctx.info("📸 Taking screenshot of current page")
        
        page = await web_browser.ensure_page()
        
        # Get current page info
        url = page.url
        title = await page.title()
        
        # Take screenshot
        screenshot_path = await web_browser.take_screenshot(page, title)
        
        if screenshot_path:
            # Add to research session
            add_research_result(ResearchResult(
                url=url,
                title=f"Screenshot: {title}",
                content="Screenshot captured",
                timestamp=datetime.now().isoformat(),
                screenshot_path=screenshot_path
            ))
            
            await ctx.info(f"✅ Screenshot saved successfully")
            
            return json.dumps({
                "success": True,
                "message": "Screenshot taken successfully",
                "url": url,
                "title": title,
                "screenshot_path": screenshot_path,
                "timestamp": datetime.now().isoformat()
            })
        else:
            raise Exception("Failed to save screenshot")
            
    except Exception as e:
        error_msg = f"❌ Screenshot failed: {str(e)}"
        await ctx.error(error_msg)
        return json.dumps({"error": error_msg})

@mcp.tool(name="ready-prompts", description="Return ready-made prompts by application category")
def get_ready_prompts(category: Optional[str] = None) -> dict:
    """
    Get ready-made prompts organized by category.
    
    Args:
        category: Optional category filter (hedis, contract, analytics, weather, web_research)
    """
    if category:
        category = category.lower()
        if category not in PROMPT_LIBRARY:
            return {"error": f"No prompts found for category '{category}'. Available categories: {', '.join(PROMPT_LIBRARY.keys())}"}
        return {"category": category, "prompts": PROMPT_LIBRARY[category]}
    else:
        return {"categories": list(PROMPT_LIBRARY.keys()), "prompts": PROMPT_LIBRARY}

@mcp.tool(name="secure-calculator", description="Safely evaluates basic arithmetic expressions")
def secure_calculate(expression: str) -> str:
    """
    Securely evaluates a basic arithmetic expression.
    Supports: +, -, *, /, parentheses, decimals.
    """
    try:
        # Parse and validate the expression
        node = ast.parse(expression, mode='eval')
        
        # Only allow specific node types for safety
        allowed_types = (ast.Expression, ast.Constant, ast.BinOp, 
                        ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, 
                        ast.Div, ast.USub, ast.UAdd, ast.Num)
        
        for node_type in ast.walk(node):
            if not isinstance(node_type, allowed_types):
                return f"Error: Invalid operation in expression: {type(node_type).__name__}"
        
        # Safely evaluate
        result = eval(compile(node, '<string>', 'eval'))
        return f"Result: {result}"
    except SyntaxError:
        return "Error: Invalid syntax in expression"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool(name="json-analyzer", description="Analyze JSON numeric data by performing operations like sum, mean, median, min, max")
def analyze_json(data: dict, operation: str) -> dict:
    """
    Analyze JSON numeric data with statistical operations.
    
    Args:
        data: Dictionary with keys mapping to lists of numbers
        operation: Operation to perform (sum, mean, median, min, max, all)
    """
    try:
        valid_operations = ["sum", "mean", "median", "min", "max", "all"]
        if operation not in valid_operations:
            return {"error": f"Invalid operation. Must be one of: {', '.join(valid_operations)}"}

        result = {}
        for key, values in data.items():
            if not isinstance(values, list):
                return {"error": f"'{key}' must be a list of numbers"}
            
            try:
                numbers = [float(n) for n in values]
            except (ValueError, TypeError):
                return {"error": f"All values in '{key}' must be numeric"}
            
            if not numbers:
                return {"error": f"No numbers provided for '{key}'"}
            
            stats = {
                "sum": sum(numbers),
                "mean": statistics.mean(numbers),
                "median": statistics.median(numbers),
                "min": min(numbers),
                "max": max(numbers),
                "count": len(numbers),
                "std_dev": statistics.stdev(numbers) if len(numbers) > 1 else 0
            }
            
            if operation == "all":
                result[key] = stats
            else:
                result[key] = stats[operation]
        
        return {"status": "success", "operation": operation, "result": result}
    except Exception as e:
        return {"error": f"Error analyzing data: {str(e)}"}

@mcp.tool()
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

@mcp.tool(name="DFWAnalyst", description="Convert text to valid SQL for HEDIS value sets and code sets")
async def dfw_text2sql(prompt: str, ctx: Context) -> dict:
    """
    Tool to convert natural language text to Snowflake SQL for HEDIS system.
    
    Args:
        prompt: Natural language text describing the query
    """
    try:
        conn = get_snowflake_connection()
        if not conn:
            return {"error": "Failed to connect to Snowflake"}
        
        db = 'POC_SPC_SNOWPARK_DB'
        schema = 'HEDIS_SCHEMA'
        stage_name = "hedis_stage_full"
        file_name = "hedis_semantic_model_complete.yaml"
        
        request_body = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "semantic_model_file": f"@{db}.{schema}.{stage_name}/{file_name}",
        }
        
        token = conn.rest.token
        
        response = requests.post(
            url=f"https://{HOST}/api/v2/cortex/analyst/message",
            json=request_body,
            headers={
                "Authorization": f'Snowflake Token="{token}"',
                "Content-Type": "application/json",
            },
            timeout=30
        )
        
        conn.close()
        
        if response.status_code == 200:
            return {"status": "success", "result": response.json()}
        else:
            return {"error": f"API request failed with status {response.status_code}: {response.text}"}
            
    except Exception as e:
        logger.error(f"DFW Analyst error: {e}")
        return {"error": f"Error in text-to-SQL conversion: {str(e)}"}

@mcp.tool(name="DFWSearch", description="Search HEDIS measure specification documents")
async def dfw_search(ctx: Context, query: str):
    """
    Tool to search HEDIS business documents for the year 2024.
    
    Args:
        query: Search string for HEDIS documents
    """
    try:
        conn = get_snowflake_connection()
        if not conn:
            return {"error": "Failed to connect to Snowflake"}
        
        db = 'POC_SPC_SNOWPARK_DB'
        schema = 'HEDIS_SCHEMA'
        search_service = 'CS_HEDIS_FULL_2024'
        columns = ['chunk']
        limit = 5
        
        root = Root(conn)
        search_service_obj = root.databases[db].schemas[schema].cortex_search_services[search_service]
        
        response = search_service_obj.search(
            query=query,
            columns=columns,
            limit=limit
        )
        
        conn.close()
        
        return {"status": "success", "query": query, "results": response.to_json()}
        
    except Exception as e:
        logger.error(f"DFW Search error: {e}")
        return {"error": f"Error searching HEDIS documents: {str(e)}"}

@mcp.tool(name="diagnostic", description="Run diagnostic tests to verify MCP functionality")
async def diagnostic(test_type: str = "basic") -> str:
    """
    Run diagnostic tests to verify MCP functionality.
    
    Args:
        test_type: Type of test (basic, browser, snowflake, all)
    """
    current_time = datetime.now().isoformat()
    
    result = f"🔧 Diagnostic Test: {test_type}\n"
    result += f"⏰ Timestamp: {current_time}\n"
    result += f"🖥️ MCP Server: Web Research Enhanced DataFlyWheel\n"
    
    if test_type in ["basic", "all"]:
        result += f"✅ Status: WORKING\n"
        result += f"📝 Message: MCP server is responding correctly\n"
        result += f"🛠️ Tool Execution: SUCCESS\n"
        
    if test_type in ["browser", "all"]:
        try:
            if not PLAYWRIGHT_AVAILABLE:
                result += f"❌ Playwright: NOT AVAILABLE (install with: pip install playwright)\n"
            else:
                # Test browser initialization
                await web_browser.ensure_page()
                result += f"✅ Playwright: AVAILABLE\n"
                result += f"✅ Browser: INITIALIZED\n"
        except Exception as e:
            result += f"❌ Browser Test: FAILED - {str(e)}\n"
        
    if test_type in ["snowflake", "all"]:
        try:
            conn = get_snowflake_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT CURRENT_VERSION()")
                version = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                result += f"❄️ Snowflake Connection: SUCCESS\n"
                result += f"❄️ Snowflake Version: {version}\n"
            else:
                result += f"❄️ Snowflake Connection: FAILED\n"
        except Exception as e:
            result += f"❄️ Snowflake Connection: FAILED - {str(e)}\n"
    
    return result

# === PROMPTS ===

@mcp.prompt(name="hedis-expert", description="HEDIS domain expert prompt")
async def hedis_prompt(query: str) -> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are an expert in the HEDIS (Healthcare Effectiveness Data and Information Set) system. 
            HEDIS is a set of standardized measures that aim to improve healthcare quality by promoting accountability and transparency.
            
            You have access to these specialized tools:
            1) DFWAnalyst - Generates SQL to retrieve information for HEDIS codes and value sets
            2) DFWSearch - Provides search capability against HEDIS measures for measurement year 2024
            3) ready-prompts - Get predefined HEDIS-related prompts and examples
            
            Please provide comprehensive, accurate information about HEDIS measures, methodologies, and requirements.
            
            Query: {query}"""
        }
    ]

@mcp.prompt(name="calculator-expert", description="Mathematical computation expert prompt")
async def calculator_prompt(query: str) -> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are an expert in performing mathematical and arithmetic operations.
            You have access to a secure calculator tool that can safely evaluate mathematical expressions.
            
            Always verify your calculations using the secure-calculator tool before providing final answers.
            Support operations: addition (+), subtraction (-), multiplication (*), division (/), parentheses for grouping.
            
            Query: {query}"""
        }
    ]

@mcp.prompt(name="weather-expert", description="Weather information expert prompt")
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

@mcp.prompt(name="web-research-expert", description="Web research expert prompt")
async def web_research_prompt(query: str) -> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are a comprehensive web research expert with access to advanced browser automation tools.
            
            Available Research Tools:
            1) search_google - Search Google for current information and news
            2) visit_page - Visit specific web pages and extract their content
            3) take_screenshot - Capture screenshots of web pages for reference
            
            Research Process:
            1. Start by using search_google to find relevant sources
            2. Visit the most promising pages using visit_page to get detailed content
            3. Take screenshots of important pages for documentation
            4. Synthesize information from multiple sources
            5. Provide comprehensive analysis with proper source citations
            
            Best Practices:
            - Always cite your sources with URLs
            - Cross-reference information from multiple sources
            - Focus on recent and authoritative sources
            - Extract key insights and provide analysis
            - Take screenshots of important findings
            
            Research Query: {query}
            
            Please conduct thorough research on this topic using the available tools."""
        }
    ]

@mcp.prompt(name="agentic-research", description="Conduct iterative web research on a topic")
async def agentic_research_prompt(topic: str) -> List[Message]:
    return [
        {
            "role": "assistant",
            "content": "I am ready to help you with your research. I will conduct thorough web research, explore topics deeply, and maintain a dialogue with you throughout the process using advanced browser automation."
        },
        {
            "role": "user",
            "content": f"""I'd like to research this topic: <topic>{topic}</topic>

Please help me explore it deeply, like you're a thoughtful, highly-trained research assistant with access to live web browsing capabilities.

Research Instructions:
1. Start by proposing your research approach - formulate an initial Google search query optimized for high-quality results
2. Get my input on whether to proceed with that query or refine it
3. Once approved, perform the Google search using search_google
4. Visit the most promising pages using visit_page to extract detailed content
5. Take screenshots of important findings using take_screenshot
6. Prioritize high-quality, authoritative sources and avoid low-quality content
7. Iteratively refine your research direction based on findings
8. Keep me informed and let me guide the research direction interactively
9. If you hit a dead end, search for new angles and explore different aspects
10. Always cite your sources with URLs and provide comprehensive analysis

Available Tools:
- search_google: Search for current information
- visit_page: Extract content from specific web pages  
- take_screenshot: Capture visual evidence

Only conclude when research goals are met. Provide thorough source citations throughout."""
        }
    ]

# === FASTAPI ENDPOINTS ===

@app.get("/")
async def root():
    return {
        "message": "Web Research Enhanced MCP Server",
        "version": "2.0.0",
        "features": [
            "Advanced Web Research with Playwright",
            "Google Search Automation",
            "Content Extraction & Screenshots", 
            "Weather Forecasting",
            "HEDIS Analytics",
            "Secure Calculations",
            "JSON Data Analysis"
        ],
        "status": "running",
        "browser_available": PLAYWRIGHT_AVAILABLE
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    browser_status = "operational" if PLAYWRIGHT_AVAILABLE else "unavailable"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "mcp_server": "operational",
            "web_research": browser_status,
            "weather": "operational",
            "analytics": "operational",
            "hedis": "operational"
        },
        "browser_capabilities": {
            "playwright_available": PLAYWRIGHT_AVAILABLE,
            "markdownify_available": MARKDOWNIFY_AVAILABLE
        }
    }

# Cleanup function
async def cleanup():
    """Cleanup browser resources on shutdown"""
    try:
        await web_browser.cleanup()
        logger.info("✅ Server cleanup completed")
    except Exception as e:
        logger.error(f"⚠️ Error during cleanup: {e}")

# Register cleanup
import atexit
atexit.register(lambda: asyncio.run(cleanup()))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Web Research Enhanced MCP Server")
    print("🌐 Features: Advanced Web Research, Google Search, Content Extraction, Screenshots")
    print("📊 Additional: HEDIS Analytics, Weather, Calculations, JSON Analysis")
    print("🖥️ Server URL: http://0.0.0.0:8000")
    
    if not PLAYWRIGHT_AVAILABLE:
        print("⚠️  WARNING: Playwright not available. Install with: pip install playwright")
        print("⚠️  After installation, run: playwright install chromium")
    
    mcp.run()
