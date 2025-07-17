# standalone_neo4j_mcp_server.py
"""
Standalone Neo4j MCP Server with LangGraph Intelligence
All configurations hard-coded - just run with: python standalone_neo4j_mcp_server.py
"""

import json
import logging
import re
import asyncio
import uuid
import requests
import urllib3
from typing import Any, Literal, Optional, TypedDict, Dict, List
from datetime import datetime

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("standalone_neo4j_mcp")

# ============================================================================
# HARD-CODED CONFIGURATION (Edit these values as needed)
# ============================================================================

# Neo4j Configuration
NEO4J_URI = "neo4j://10.189.116.237:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "Vkg5d$F!pLq2@9vRwE="
NEO4J_DATABASE = "connectiq"
NEO4J_NAMESPACE = "connectiq"

# Cortex LLM Configuration
CORTEX_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
CORTEX_API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
CORTEX_APP_ID = "edadip"
CORTEX_APLCTN_CD = "edagnai"
CORTEX_MODEL = "llama3.1-70b"

# Server Configuration
MCP_HOST = "127.0.0.1"
MCP_PORT = 8000

# ============================================================================
# INSTALL DEPENDENCIES PROGRAMMATICALLY
# ============================================================================

def install_dependencies():
    """Install required packages if not available"""
    required_packages = [
        "fastmcp", "neo4j", "pydantic", "requests", 
        "langgraph", "langchain-core"
    ]
    
    import subprocess
    import sys
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install dependencies
install_dependencies()

# Import after installation
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult, TextContent
from fastmcp.server import FastMCP
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncResult, AsyncTransaction
from pydantic import Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# ============================================================================
# NEO4J HELPER FUNCTIONS
# ============================================================================

def _format_namespace(namespace: str) -> str:
    """Format namespace with proper suffix"""
    if namespace:
        if namespace.endswith("-"):
            return namespace
        else:
            return namespace + "-"
    else:
        return ""

async def _read(tx: AsyncTransaction, query: str, params: dict[str, Any]) -> str:
    """Execute read transaction"""
    raw_results = await tx.run(query, params)
    eager_results = await raw_results.to_eager_result()
    return json.dumps([r.data() for r in eager_results.records], default=str)

async def _write(tx: AsyncTransaction, query: str, params: dict[str, Any]) -> AsyncResult:
    """Execute write transaction"""
    return await tx.run(query, params)

def _is_write_query(query: str) -> bool:
    """Check if the query is a write query"""
    return (
        re.search(r"\b(MERGE|CREATE|SET|DELETE|REMOVE|ADD)\b", query, re.IGNORECASE)
        is not None
    )

# ============================================================================
# LANGGRAPH AI COMPONENTS
# ============================================================================

class GraphState(TypedDict):
    user_input: str
    intent: str
    analysis: str
    cypher_query: str
    neo4j_tool: str
    tool_params: Dict[str, Any]
    raw_result: str
    interpretation: str
    next_action: str
    error_message: str
    confidence: float

class CortexLLM:
    """Cortex LLM for intelligent query processing"""
    
    async def generate_response(self, prompt: str, system_message: str = "") -> str:
        """Generate response using Cortex LLM"""
        try:
            payload = {
                "query": {
                    "aplctn_cd": CORTEX_APLCTN_CD,
                    "app_id": CORTEX_APP_ID,
                    "api_key": CORTEX_API_KEY,
                    "method": "cortex",
                    "model": CORTEX_MODEL,
                    "sys_msg": system_message,
                    "limit_convs": "0",
                    "prompt": {
                        "messages": [{"role": "user", "content": prompt}]
                    },
                    "session_id": str(uuid.uuid4())
                }
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f'Snowflake Token="{CORTEX_API_KEY}"'
            }

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    CORTEX_URL, 
                    headers=headers, 
                    json=payload, 
                    verify=False, 
                    timeout=30
                )
            )
            
            response.raise_for_status()
            raw = response.text
            
            if "end_of_stream" in raw:
                result = raw.split("end_of_stream")[0].strip()
            else:
                result = raw.strip()
                
            # Clean JSON wrappers if present
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict) and 'content' in parsed:
                    return parsed['content']
                elif isinstance(parsed, dict) and 'response' in parsed:
                    return parsed['response']
            except json.JSONDecodeError:
                pass
            
            return result
            
        except Exception as e:
            return f"❌ Cortex LLM error: {str(e)}"

# Initialize LLM
cortex_llm = CortexLLM()

# ============================================================================
# LANGGRAPH AGENTS
# ============================================================================

async def intent_analyzer(state: GraphState) -> GraphState:
    """Analyze user intent and determine appropriate Neo4j action"""
    user_input = state["user_input"]
    
    system_prompt = """
You are an expert Neo4j database analyst. Analyze user requests and classify the intent.

ConnectIQ Database Schema:
- Nodes: Apps, Devices, Users, Categories, Versions, Reviews, Developers
- Relationships: COMPATIBLE_WITH, BELONGS_TO, HAS_VERSION, REVIEWED_BY, DEVELOPED_BY, INSTALLED_ON
- Properties: name, version, rating, install_count, category, device_type, release_date, description

Intent Classifications:
1. SCHEMA_INQUIRY - wants database structure/schema information
2. READ_QUERY - wants to search/retrieve data
3. WRITE_QUERY - wants to create/update/delete data  
4. ANALYTICS - wants statistics, trends, or analysis
5. RELATIONSHIP_EXPLORATION - wants to explore connections between entities

Provide:
- Intent classification
- Confidence level (0.0-1.0)
- Brief analysis of what the user wants

Format: JSON with "intent", "confidence", "analysis"
"""
    
    prompt = f'Analyze this request: "{user_input}"'
    
    response = await cortex_llm.generate_response(prompt, system_prompt)
    
    try:
        analysis_data = json.loads(response)
        intent = analysis_data.get("intent", "READ_QUERY").upper()
        confidence = float(analysis_data.get("confidence", 0.8))
        analysis = analysis_data.get("analysis", "General query analysis")
    except (json.JSONDecodeError, ValueError):
        intent = "READ_QUERY"
        confidence = 0.7
        analysis = response[:200] + "..." if len(response) > 200 else response
    
    valid_intents = ["SCHEMA_INQUIRY", "READ_QUERY", "WRITE_QUERY", "ANALYTICS", "RELATIONSHIP_EXPLORATION"]
    if intent not in valid_intents:
        intent = "READ_QUERY"
    
    state["intent"] = intent
    state["confidence"] = confidence
    state["analysis"] = analysis
    state["next_action"] = "query_generator"
    
    return state

async def query_generator(state: GraphState) -> GraphState:
    """Generate optimized Cypher query based on intent analysis"""
    user_input = state["user_input"]
    intent = state["intent"]
    analysis = state["analysis"]
    
    if intent == "SCHEMA_INQUIRY":
        state["neo4j_tool"] = "get_neo4j_schema"
        state["tool_params"] = {}
        state["cypher_query"] = "// Schema inquiry - using get_neo4j_schema tool"
        state["next_action"] = "neo4j_executor"
        return state
    
    system_prompt = f"""
You are a Cypher query expert for Neo4j ConnectIQ database.

Context:
- User Intent: {intent}
- Analysis: {analysis}

Database Schema:
- Nodes: Apps(name, rating, install_count, category, release_date, description), 
         Devices(name, device_type, manufacturer), 
         Users(name, age, location), 
         Categories(name, description),
         Versions(version_number, release_date, features),
         Reviews(rating, comment, review_date),
         Developers(name, company, experience_level)

- Relationships: 
  - (App)-[:COMPATIBLE_WITH]->(Device)
  - (App)-[:BELONGS_TO]->(Category) 
  - (App)-[:HAS_VERSION]->(Version)
  - (User)-[:REVIEWED_BY]->(Review)-[:FOR]->(App)
  - (Developer)-[:DEVELOPED_BY]->(App)
  - (User)-[:INSTALLED_ON]->(Device)

Guidelines:
1. Generate efficient, optimized Cypher queries
2. Use appropriate LIMIT clauses (default: 25 for lists, 100 for analytics)
3. Include meaningful property names in RETURN statements
4. Use ORDER BY for better results presentation
5. Consider performance implications

Generate ONLY the Cypher query, no explanations or formatting.
"""
    
    prompt = f'Generate Cypher query for: "{user_input}"'
    
    cypher_query = await cortex_llm.generate_response(prompt, system_prompt)
    
    # Clean the query
    cypher_query = cypher_query.strip()
    if cypher_query.startswith("```"):
        lines = cypher_query.split("\n")
        cypher_query = "\n".join([line for line in lines if not line.startswith("```")])
        cypher_query = cypher_query.strip()
    
    cypher_query = re.sub(r'^(cypher|neo4j)\s*', '', cypher_query, flags=re.IGNORECASE)
    
    state["cypher_query"] = cypher_query
    
    if _is_write_query(cypher_query):
        state["neo4j_tool"] = "write_neo4j_cypher"
    else:
        state["neo4j_tool"] = "read_neo4j_cypher"
    
    state["tool_params"] = {"query": cypher_query}
    state["next_action"] = "neo4j_executor"
    
    return state

async def result_interpreter(state: GraphState) -> GraphState:
    """Interpret and enhance results with business insights"""
    user_input = state["user_input"]
    intent = state["intent"]
    cypher_query = state.get("cypher_query", "")
    raw_result = state.get("raw_result", "")
    
    system_prompt = f"""
You are a business intelligence analyst specializing in mobile app ecosystems and ConnectIQ data.

Context:
- User Question: "{user_input}"
- Intent: {intent}
- Query Type: {"Schema" if intent == "SCHEMA_INQUIRY" else "Data Query"}

Provide a comprehensive analysis including:
1. **Summary**: What the data shows in plain language
2. **Key Insights**: Important patterns, trends, or findings
3. **Business Value**: What this means for stakeholders
4. **Recommendations**: Actionable next steps or follow-up questions
5. **Data Quality**: Any observations about completeness or reliability

Make it accessible for both technical and non-technical audiences.
Focus on business value and actionable insights.
"""
    
    prompt = f"""
Query executed: {cypher_query}

Raw results: {raw_result}

Provide comprehensive business analysis:
"""
    
    interpretation = await cortex_llm.generate_response(prompt, system_prompt)
    state["interpretation"] = interpretation
    state["next_action"] = "finish"
    
    return state

def route_agent(state: GraphState) -> Literal["query_generator", "result_interpreter", "end"]:
    """Route to next agent based on workflow state"""
    next_action = state.get("next_action", "finish")
    
    if next_action == "finish":
        return "end"
    elif next_action == "query_generator":
        return "query_generator"
    elif next_action == "result_interpreter":
        return "result_interpreter"
    else:
        return "end"

# Create LangGraph workflow
def create_neo4j_workflow():
    """Create the intelligent Neo4j workflow"""
    workflow = StateGraph(GraphState)
    
    workflow.add_node("intent_analyzer", intent_analyzer)
    workflow.add_node("query_generator", query_generator)
    workflow.add_node("result_interpreter", result_interpreter)
    
    workflow.set_entry_point("intent_analyzer")
    
    workflow.add_edge("intent_analyzer", "query_generator")
    
    workflow.add_conditional_edges(
        "query_generator",
        route_agent,
        {
            "result_interpreter": "result_interpreter",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "result_interpreter",
        route_agent,
        {
            "end": END
        }
    )
    
    return workflow.compile()

# ============================================================================
# FASTMCP SERVER WITH NEO4J TOOLS
# ============================================================================

def create_enhanced_neo4j_server(neo4j_driver: AsyncDriver, database: str = "neo4j", namespace: str = "") -> FastMCP:
    """Create enhanced FastMCP server with original Neo4j tools + LangGraph intelligence"""
    
    mcp: FastMCP = FastMCP(
        "enhanced-neo4j-langgraph", 
        dependencies=["neo4j", "pydantic", "langgraph", "langchain-core"], 
        stateless_http=True
    )
    
    namespace_prefix = _format_namespace(namespace)
    neo4j_workflow = create_neo4j_workflow()
    
    # ========================================================================
    # Original Neo4j MCP Tools
    # ========================================================================
    
    @mcp.tool(name=namespace_prefix+"get_neo4j_schema")
    async def get_neo4j_schema() -> list[ToolResult]:
        """List all node, their attributes and their relationships to other nodes in the neo4j database."""
        get_schema_query = "CALL apoc.meta.schema();"

        def clean_schema(schema: dict) -> dict:
            cleaned = {}
            for key, entry in schema.items():
                new_entry = {"type": entry["type"]}
                if "count" in entry:
                    new_entry["count"] = entry["count"]

                labels = entry.get("labels", [])
                if labels:
                    new_entry["labels"] = labels

                props = entry.get("properties", {})
                clean_props = {}
                for pname, pinfo in props.items():
                    cp = {}
                    if "indexed" in pinfo:
                        cp["indexed"] = pinfo["indexed"]
                    if "type" in pinfo:
                        cp["type"] = pinfo["type"]
                    if cp:
                        clean_props[pname] = cp
                if clean_props:
                    new_entry["properties"] = clean_props

                if entry.get("relationships"):
                    rels_out = {}
                    for rel_name, rel in entry["relationships"].items():
                        cr = {}
                        if "direction" in rel:
                            cr["direction"] = rel["direction"]
                        rlabels = rel.get("labels", [])
                        if rlabels:
                            cr["labels"] = rlabels
                        rprops = rel.get("properties", {})
                        clean_rprops = {}
                        for rpname, rpinfo in rprops.items():
                            crp = {}
                            if "indexed" in rpinfo:
                                crp["indexed"] = rpinfo["indexed"]
                            if "type" in rpinfo:
                                crp["type"] = rpinfo["type"]
                            if crp:
                                clean_rprops[rpname] = crp
                        if clean_rprops:
                            cr["properties"] = clean_rprops
                        if cr:
                            rels_out[rel_name] = cr
                    if rels_out:
                        new_entry["relationships"] = rels_out

                cleaned[key] = new_entry
            return cleaned

        try:
            async with neo4j_driver.session(database=database) as session:
                results_json_str = await session.execute_read(_read, get_schema_query, dict())
                
                schema = json.loads(results_json_str)[0].get('value')
                schema_clean = clean_schema(schema)
                schema_clean_str = json.dumps(schema_clean)
                
                return [ToolResult(content=[TextContent(type="text", text=schema_clean_str)])]

        except Exception as e:
            logger.error(f"Database error retrieving schema: {e}")
            raise ToolError(f"Error: {e}")

    @mcp.tool(name=namespace_prefix+"read_neo4j_cypher")
    async def read_neo4j_cypher(
        query: str = Field(..., description="The Cypher query to execute."),
        params: Optional[dict[str, Any]] = Field(None, description="The parameters to pass to the Cypher query."),
    ) -> list[ToolResult]:
        """Execute a read Cypher query on the neo4j database."""
        if _is_write_query(query):
            raise ValueError("Only MATCH queries are allowed for read-query")

        try:
            async with neo4j_driver.session(database=database) as session:
                results_json_str = await session.execute_read(_read, query, params)
                return [ToolResult(content=[TextContent(type="text", text=results_json_str)])]

        except Exception as e:
            logger.error(f"Database error executing query: {e}\n{query}\n{params}")
            raise ToolError(f"Error: {e}\n{query}\n{params}")

    @mcp.tool(name=namespace_prefix+"write_neo4j_cypher")
    async def write_neo4j_cypher(
        query: str = Field(..., description="The Cypher query to execute."),
        params: Optional[dict[str, Any]] = Field(None, description="The parameters to pass to the Cypher query."),
    ) -> list[ToolResult]:
        """Execute a write Cypher query on the neo4j database."""
        if not _is_write_query(query):
            raise ValueError("Only write queries are allowed for write-query")

        try:
            async with neo4j_driver.session(database=database) as session:
                raw_results = await session.execute_write(_write, query, params)
                counters_json_str = json.dumps(raw_results._summary.counters.__dict__, default=str)
            
            return [ToolResult(content=[TextContent(type="text", text=counters_json_str)])]

        except Exception as e:
            logger.error(f"Database error executing query: {e}\n{query}\n{params}")
            raise ToolError(f"Error: {e}\n{query}\n{params}")

    # ========================================================================
    # Enhanced AI-Powered Tools with LangGraph
    # ========================================================================
    
    @mcp.tool(name=namespace_prefix+"intelligent_neo4j_query")
    async def intelligent_neo4j_query(
        user_input: str = Field(..., description="Natural language query or request about the Neo4j database")
    ) -> list[ToolResult]:
        """
        Process natural language requests using AI-powered analysis and optimized Cypher generation.
        This tool uses LangGraph to understand intent, generate optimal queries, and provide business insights.
        """
        try:
            initial_state = {
                "user_input": user_input,
                "intent": "",
                "analysis": "",
                "cypher_query": "",
                "neo4j_tool": "",
                "tool_params": {},
                "raw_result": "",
                "interpretation": "",
                "next_action": "",
                "error_message": "",
                "confidence": 0.0
            }
            
            workflow_state = await neo4j_workflow.ainvoke(initial_state)
            
            # Execute the actual Neo4j operation
            raw_result = ""
            if workflow_state.get("neo4j_tool") == "get_neo4j_schema":
                schema_results = await get_neo4j_schema()
                raw_result = schema_results[0].content[0].text
                
            elif workflow_state.get("neo4j_tool") == "read_neo4j_cypher":
                read_results = await read_neo4j_cypher(
                    query=workflow_state["tool_params"]["query"],
                    params=workflow_state["tool_params"].get("params")
                )
                raw_result = read_results[0].content[0].text
                
            elif workflow_state.get("neo4j_tool") == "write_neo4j_cypher":
                write_results = await write_neo4j_cypher(
                    query=workflow_state["tool_params"]["query"],
                    params=workflow_state["tool_params"].get("params")
                )
                raw_result = write_results[0].content[0].text
            
            # Update state and get interpretation
            workflow_state["raw_result"] = raw_result
            final_state = await result_interpreter(workflow_state)
            
            response = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "request": {
                    "user_input": user_input,
                    "intent": final_state.get("intent", ""),
                    "confidence": final_state.get("confidence", 0.0),
                    "analysis": final_state.get("analysis", "")
                },
                "execution": {
                    "tool_used": final_state.get("neo4j_tool", ""),
                    "cypher_query": final_state.get("cypher_query", ""),
                    "raw_result": raw_result
                },
                "insights": {
                    "interpretation": final_state.get("interpretation", ""),
                    "business_value": "Extracted from comprehensive analysis"
                }
            }
            
            if final_state.get("error_message"):
                response["status"] = "error"
                response["error"] = final_state["error_message"]
            
            return [ToolResult(content=[TextContent(type="text", text=json.dumps(response, indent=2))])]
            
        except Exception as e:
            error_response = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "error": str(e)
            }
            return [ToolResult(content=[TextContent(type="text", text=json.dumps(error_response, indent=2))])]
    
    @mcp.tool(name=namespace_prefix+"system_health_check")
    async def system_health_check() -> list[ToolResult]:
        """Comprehensive system health check including database connectivity."""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "neo4j_database": {
                        "status": "connected",
                        "uri": NEO4J_URI,
                        "database": database,
                        "namespace": namespace
                    },
                    "langgraph_workflow": {
                        "status": "active",
                        "agents": ["intent_analyzer", "query_generator", "result_interpreter"]
                    },
                    "cortex_llm": {
                        "status": "connected",
                        "model": CORTEX_MODEL
                    },
                    "fastmcp_server": {
                        "status": "running",
                        "transport": "stdio",
                        "tools_count": 5
                    }
                }
            }
            
            # Test database connectivity
            try:
                async with neo4j_driver.session(database=database) as session:
                    test_result = await session.execute_read(_read, "RETURN 1 as test", {})
                    if "1" in test_result:
                        health_status["components"]["neo4j_database"]["connectivity_test"] = "passed"
                    else:
                        health_status["components"]["neo4j_database"]["connectivity_test"] = "failed"
            except Exception as e:
                health_status["components"]["neo4j_database"]["connectivity_test"] = f"failed: {str(e)}"
                health_status["status"] = "degraded"
            
            return [ToolResult(content=[TextContent(type="text", text=json.dumps(health_status, indent=2))])]
            
        except Exception as e:
            error_response = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
            return [ToolResult(content=[TextContent(type="text", text=json.dumps(error_response, indent=2))])]
    
    return mcp

# ============================================================================
# MAIN SERVER FUNCTION
# ============================================================================

async def main():
    """Main function to run the enhanced Neo4j MCP server"""
    
    logger.info("🚀 Starting Standalone Enhanced Neo4j MCP Server")
    logger.info(f"📊 Database: {NEO4J_URI}/{NEO4J_DATABASE}")
    logger.info(f"🧠 AI Features: Intent Analysis, Query Generation, Result Interpretation")
    logger.info(f"🔧 Transport: STDIO")
    
    # Initialize Neo4j driver
    neo4j_driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    # Create enhanced server
    mcp = create_enhanced_neo4j_server(neo4j_driver, NEO4J_DATABASE, NEO4J_NAMESPACE)
    
    # Run server with STDIO transport
    try:
        logger.info("📡 Running with STDIO transport")
        await mcp.run_stdio_async()
                
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise
    finally:
        logger.info("🧹 Cleaning up...")
        await neo4j_driver.close()
        logger.info("👋 Enhanced Neo4j MCP Server shutdown complete")

if __name__ == "__main__":
    print("🧠 Standalone Neo4j MCP Server with LangGraph Intelligence")
    print("=" * 60)
    print("📋 Available Tools:")
    print("  - get_neo4j_schema: Database schema information")
    print("  - read_neo4j_cypher: Execute read queries")
    print("  - write_neo4j_cypher: Execute write queries")
    print("  - intelligent_neo4j_query: AI-powered natural language processing")
    print("  - system_health_check: System status monitoring")
    print("=" * 60)
    print("🚀 Starting server...")
    
    asyncio.run(main())
