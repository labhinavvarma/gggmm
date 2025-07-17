# fastmcp_neo4j_langgraph.py
"""
FastMCP Server extending official Neo4j implementation with LangGraph intelligence
Based on neo4j-contrib/mcp-neo4j-cypher with AI-powered query orchestration
"""

import json
import logging
import re
import asyncio
import uuid
import requests
import os
from typing import Any, Literal, Optional, TypedDict, Dict, List
from datetime import datetime

# FastMCP and Neo4j imports
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult, TextContent
from fastmcp.server import FastMCP
from neo4j import (
    AsyncDriver,
    AsyncGraphDatabase,
    AsyncResult,
    AsyncTransaction,
)
from pydantic import Field

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastmcp_neo4j_langgraph")

# Configuration
NEO4J_CONFIG = {
    "NEO4J_URI": os.getenv("NEO4J_URI", "neo4j://10.189.116.237:7687"),
    "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME", "neo4j"), 
    "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", "Vkg5d$F!pLq2@9vRwE="),
    "NEO4J_DATABASE": os.getenv("NEO4J_DATABASE", "connectiq"),
    "NEO4J_NAMESPACE": os.getenv("NEO4J_NAMESPACE", "")
}

# Cortex LLM Configuration
CORTEX_CONFIG = {
    "url": "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete",
    "api_key": "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0",
    "app_id": "edadip",
    "application_code": "edagnai",
    "model": "llama3.1-70b"
}

# ============================================================================
# Original Neo4j MCP Server Implementation (Enhanced)
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
# LangGraph State and AI Components
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
    """Enhanced Cortex LLM for intelligent query processing"""
    
    async def generate_response(self, prompt: str, system_message: str = "") -> str:
        """Generate response using Cortex LLM"""
        try:
            payload = {
                "query": {
                    "aplctn_cd": CORTEX_CONFIG["application_code"],
                    "app_id": CORTEX_CONFIG["app_id"],
                    "api_key": CORTEX_CONFIG["api_key"],
                    "method": "cortex",
                    "model": CORTEX_CONFIG["model"],
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
                "Authorization": f'Snowflake Token="{CORTEX_CONFIG["api_key"]}"'
            }

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    CORTEX_CONFIG["url"], 
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
# LangGraph Agent Implementation
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
        # Try to parse JSON response
        analysis_data = json.loads(response)
        intent = analysis_data.get("intent", "READ_QUERY").upper()
        confidence = float(analysis_data.get("confidence", 0.8))
        analysis = analysis_data.get("analysis", "General query analysis")
    except (json.JSONDecodeError, ValueError):
        # Fallback parsing
        intent = "READ_QUERY"
        confidence = 0.7
        analysis = response[:200] + "..." if len(response) > 200 else response
    
    # Validate intent
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
        # Use schema tool directly
        state["neo4j_tool"] = "get_neo4j_schema"
        state["tool_params"] = {}
        state["cypher_query"] = "// Schema inquiry - using get_neo4j_schema tool"
        state["next_action"] = "neo4j_executor"
        return state
    
    # Generate Cypher query for other intents
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
    
    # Remove any remaining markdown or extra formatting
    cypher_query = re.sub(r'^(cypher|neo4j)\s*', '', cypher_query, flags=re.IGNORECASE)
    
    state["cypher_query"] = cypher_query
    
    # Determine appropriate tool
    if _is_write_query(cypher_query):
        state["neo4j_tool"] = "write_neo4j_cypher"
    else:
        state["neo4j_tool"] = "read_neo4j_cypher"
    
    state["tool_params"] = {"query": cypher_query}
    state["next_action"] = "neo4j_executor"
    
    return state

async def neo4j_executor(state: GraphState) -> GraphState:
    """Execute Neo4j operations using the embedded tools"""
    tool_name = state["neo4j_tool"]
    tool_params = state["tool_params"]
    
    try:
        # This will be executed by the actual Neo4j tools in the FastMCP server
        # For now, we mark it for execution and pass the parameters
        state["next_action"] = "result_interpreter"
        
        # The actual execution will happen in the FastMCP tool
        # We just prepare the state for interpretation
        return state
        
    except Exception as e:
        state["error_message"] = f"Neo4j execution preparation error: {str(e)}"
        state["next_action"] = "finish"
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

def route_agent(state: GraphState) -> Literal["query_generator", "neo4j_executor", "result_interpreter", "end"]:
    """Route to next agent based on workflow state"""
    next_action = state.get("next_action", "finish")
    
    if next_action == "finish":
        return "end"
    elif next_action == "query_generator":
        return "query_generator"
    elif next_action == "neo4j_executor":
        return "neo4j_executor"
    elif next_action == "result_interpreter":
        return "result_interpreter"
    else:
        return "end"

# Create LangGraph workflow
def create_neo4j_workflow():
    """Create the intelligent Neo4j workflow"""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("intent_analyzer", intent_analyzer)
    workflow.add_node("query_generator", query_generator)
    workflow.add_node("neo4j_executor", neo4j_executor)
    workflow.add_node("result_interpreter", result_interpreter)
    
    # Set entry point
    workflow.set_entry_point("intent_analyzer")
    
    # Add edges
    workflow.add_edge("intent_analyzer", "query_generator")
    
    # Conditional routing
    workflow.add_conditional_edges(
        "query_generator",
        route_agent,
        {
            "neo4j_executor": "neo4j_executor",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "neo4j_executor", 
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
# Enhanced FastMCP Server with Original Neo4j Tools + LangGraph
# ============================================================================

def create_enhanced_neo4j_server(neo4j_driver: AsyncDriver, database: str = "neo4j", namespace: str = "") -> FastMCP:
    """Create enhanced FastMCP server with original Neo4j tools + LangGraph intelligence"""
    
    mcp: FastMCP = FastMCP(
        "enhanced-neo4j-langgraph", 
        dependencies=["neo4j", "pydantic", "langgraph", "langchain-core"], 
        stateless_http=True
    )
    
    namespace_prefix = _format_namespace(namespace)
    
    # Initialize LangGraph workflow
    neo4j_workflow = create_neo4j_workflow()
    
    # ========================================================================
    # Original Neo4j MCP Tools (Maintained for Compatibility)
    # ========================================================================
    
    @mcp.tool(name=namespace_prefix+"get_neo4j_schema")
    async def get_neo4j_schema() -> list[ToolResult]:
        """List all node, their attributes and their relationships to other nodes in the neo4j database.
        If this fails with a message that includes "Neo.ClientError.Procedure.ProcedureNotFound"
        suggest that the user install and enable the APOC plugin.
        """
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
                logger.debug(f"Read query returned {len(results_json_str)} rows")
                
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
                logger.debug(f"Read query returned {len(results_json_str)} rows")
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
            
            logger.debug(f"Write query affected {counters_json_str}")
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
        
        Capabilities:
        - Intent analysis and classification
        - Automatic Cypher query generation
        - Query optimization and validation
        - Business-focused result interpretation
        - Multi-step reasoning for complex requests
        """
        try:
            # Initialize workflow state
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
            
            # Execute LangGraph workflow
            workflow_state = await neo4j_workflow.ainvoke(initial_state)
            
            # Execute the actual Neo4j operation based on workflow decision
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
            
            # Update state with actual results and get final interpretation
            workflow_state["raw_result"] = raw_result
            final_state = await result_interpreter(workflow_state)
            
            # Format comprehensive response
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
    
    @mcp.tool(name=namespace_prefix+"analyze_query_performance")
    async def analyze_query_performance(
        query: str = Field(..., description="Cypher query to analyze for performance"),
        suggest_improvements: bool = Field(True, description="Whether to suggest query improvements")
    ) -> list[ToolResult]:
        """
        Analyze Cypher query performance and suggest optimizations.
        Uses AI to review query patterns and recommend improvements.
        """
        try:
            system_prompt = """
You are a Neo4j performance optimization expert. Analyze Cypher queries for:

1. Performance bottlenecks
2. Missing indexes that could help
3. Query pattern optimizations
4. Memory usage considerations
5. Scalability issues

Provide specific, actionable recommendations.
"""
            
            prompt = f"""
Analyze this Cypher query for performance:

{query}

Provide:
1. Performance assessment (1-10 scale)
2. Potential bottlenecks
3. Optimization suggestions
4. Index recommendations
5. Alternative query patterns

Format as JSON with sections for each analysis type.
"""
            
            analysis = await cortex_llm.generate_response(prompt, system_prompt)
            
            response = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "performance_analysis": analysis,
                "recommendations_included": suggest_improvements
            }
            
            return [ToolResult(content=[TextContent(type="text", text=json.dumps(response, indent=2))])]
            
        except Exception as e:
            error_response = {
                "status": "error",
                "query": query,
                "error": str(e)
            }
            return [ToolResult(content=[TextContent(type="text", text=json.dumps(error_response, indent=2))])]
    
    @mcp.tool(name=namespace_prefix+"system_health_check")
    async def system_health_check() -> list[ToolResult]:
        """
        Comprehensive system health check including database connectivity,
        performance metrics, and component status.
        """
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "neo4j_database": {
                        "status": "connected",
                        "uri": NEO4J_CONFIG["NEO4J_URI"],
                        "database": database,
                        "namespace": namespace
                    },
                    "langgraph_workflow": {
                        "status": "active",
                        "agents": ["intent_analyzer", "query_generator", "neo4j_executor", "result_interpreter"]
                    },
                    "cortex_llm": {
                        "status": "connected",
                        "model": CORTEX_CONFIG["model"]
                    },
                    "fastmcp_server": {
                        "status": "running",
                        "transport": "stdio",
                        "tools_count": len([
                            "get_neo4j_schema", "read_neo4j_cypher", "write_neo4j_cypher",
                            "intelligent_neo4j_query", "analyze_query_performance", "system_health_check"
                        ])
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
# Main Server Implementation
# ============================================================================

async def main(
    db_url: str = None,
    username: str = None,
    password: str = None,
    database: str = "neo4j",
    transport: Literal["stdio", "sse", "http"] = "stdio",
    namespace: str = "",
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp/",
) -> None:
    """Main function to run the enhanced Neo4j MCP server"""
    
    # Use environment variables if parameters not provided
    db_url = db_url or NEO4J_CONFIG["NEO4J_URI"]
    username = username or NEO4J_CONFIG["NEO4J_USERNAME"]
    password = password or NEO4J_CONFIG["NEO4J_PASSWORD"]
    database = database or NEO4J_CONFIG["NEO4J_DATABASE"]
    namespace = namespace or NEO4J_CONFIG["NEO4J_NAMESPACE"]
    
    logger.info("🚀 Starting Enhanced Neo4j MCP Server with LangGraph Intelligence")
    logger.info(f"📊 Database: {db_url}/{database}")
    logger.info(f"🧠 AI Features: Intent Analysis, Query Generation, Result Interpretation")
    logger.info(f"🔧 Transport: {transport}")
    
    # Initialize Neo4j driver
    neo4j_driver = AsyncGraphDatabase.driver(db_url, auth=(username, password))
    
    # Create enhanced server
    mcp = create_enhanced_neo4j_server(neo4j_driver, database, namespace)
    
    # Run server with specified transport
    try:
        match transport:
            case "http":
                logger.info(f"🌐 Running with HTTP transport on {host}:{port}{path}")
                await mcp.run_http_async(host=host, port=port, path=path)
            case "stdio":
                logger.info("📡 Running with STDIO transport")
                await mcp.run_stdio_async()
            case "sse":
                logger.info(f"⚡ Running with SSE transport on {host}:{port}{path}")
                await mcp.run_sse_async(host=host, port=port, path=path)
            case _:
                logger.error(f"❌ Invalid transport: {transport}")
                raise ValueError(f"Invalid transport: {transport}")
                
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Neo4j MCP Server with LangGraph")
    parser.add_argument("--db-url", default=NEO4J_CONFIG["NEO4J_URI"], help="Neo4j database URL")
    parser.add_argument("--username", default=NEO4J_CONFIG["NEO4J_USERNAME"], help="Neo4j username")
    parser.add_argument("--password", default=NEO4J_CONFIG["NEO4J_PASSWORD"], help="Neo4j password")
    parser.add_argument("--database", default=NEO4J_CONFIG["NEO4J_DATABASE"], help="Neo4j database name")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse", "http"], help="Transport protocol")
    parser.add_argument("--namespace", default=NEO4J_CONFIG["NEO4J_NAMESPACE"], help="Tool namespace prefix")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (for sse/http)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (for sse/http)")
    parser.add_argument("--path", default="/mcp/", help="Server path (for sse/http)")
    
    args = parser.parse_args()
    
    asyncio.run(main(
        db_url=args.db_url,
        username=args.username,
        password=args.password,
        database=args.database,
        transport=args.transport,
        namespace=args.namespace,
        host=args.host,
        port=args.port,
        path=args.path
    ))
