import requests
import urllib3
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import re
import json
import logging
from typing import Dict, List, Any, Optional, Literal
from enum import Enum

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("complete_langgraph_agent")

# Configuration
API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"  # Change this!
MODEL = "llama3.1-70b"
MCP_PORT = 8000

class ToolType(str, Enum):
    READ = "read_neo4j_cypher"
    WRITE = "write_neo4j_cypher"
    SCHEMA = "get_neo4j_schema"
    UNKNOWN = "unknown"

class AgentState(BaseModel):
    # Input
    question: str
    session_id: str
    
    # Processing state
    intent: str = ""
    tool: str = ""
    query: str = ""
    
    # LLM interactions
    analysis_prompt: str = ""
    analysis_response: str = ""
    query_prompt: str = ""
    query_response: str = ""
    
    # Execution
    execution_result: Any = None
    formatted_result: str = ""
    
    # Output
    trace: str = ""
    answer: str = ""
    
    # Error handling
    error_count: int = 0
    last_error: str = ""
    
    # Routing
    next_step: str = ""

# ============================================
# PROMPT TEMPLATES
# ============================================

INTENT_ANALYSIS_PROMPT = """You are an expert Neo4j database assistant. Analyze the user's question and determine their intent.

USER QUESTION: {question}

Your task is to classify the intent into one of these categories:

**READ INTENTS:**
- count: Counting nodes, relationships, or aggregations
- list: Listing nodes, relationships, or properties
- find: Finding specific nodes or relationships
- analyze: Analyzing patterns, connections, or statistics
- explore: General exploration of data

**WRITE INTENTS:**
- create: Creating new nodes or relationships
- update: Updating existing nodes or relationships  
- delete: Deleting nodes or relationships
- modify: Any modification to the graph structure

**SCHEMA INTENTS:**
- schema: Asking about database structure, labels, relationships
- meta: Asking about metadata, constraints, indexes

**EXAMPLES:**

User: "How many nodes are in the graph?"
Intent: count

User: "List all Person nodes"
Intent: list

User: "Create a Person named Alice"
Intent: create

User: "Show me the database schema"
Intent: schema

User: "Find nodes with the most relationships"
Intent: find

User: "What labels exist in the database?"
Intent: schema

User: "Delete all temporary nodes"
Intent: delete

User: "Update John's age to 30"
Intent: update

**RESPONSE FORMAT:**
Intent: [intent_name]
Reasoning: [brief explanation of why you chose this intent]

Analyze the user's question and respond:"""

TOOL_SELECTION_PROMPT = """You are a Neo4j tool selection expert. Based on the user's intent, select the appropriate tool.

USER QUESTION: {question}
DETECTED INTENT: {intent}

**TOOL SELECTION RULES:**

**Use read_neo4j_cypher for:**
- count, list, find, analyze, explore intents
- Any query that retrieves or analyzes data
- MATCH, RETURN, WHERE, COUNT, COLLECT, ORDER BY, LIMIT
- Aggregations, calculations, pattern matching

**Use write_neo4j_cypher for:**
- create, update, delete, modify intents  
- Any query that changes the graph
- CREATE, MERGE, SET, DELETE, DETACH DELETE, REMOVE
- Adding nodes, relationships, or properties

**Use get_neo4j_schema for:**
- schema, meta intents
- Questions about database structure
- Asking for labels, relationship types, properties
- Metadata queries

**EXAMPLES:**

Intent: count → Tool: read_neo4j_cypher
Intent: create → Tool: write_neo4j_cypher  
Intent: schema → Tool: get_neo4j_schema
Intent: list → Tool: read_neo4j_cypher
Intent: delete → Tool: write_neo4j_cypher

**RESPONSE FORMAT:**
Tool: [tool_name]
Reasoning: [brief explanation]

Select the tool:"""

CYPHER_GENERATION_PROMPT = """You are an expert Cypher query generator for Neo4j. Generate a precise Cypher query based on the user's question and selected tool.

USER QUESTION: {question}
INTENT: {intent}
SELECTED TOOL: {tool}

**CYPHER QUERY GUIDELINES:**

**For read_neo4j_cypher (READ-ONLY QUERIES):**
- Use MATCH, RETURN, WHERE, COUNT, COLLECT, ORDER BY, LIMIT
- For counting: MATCH (n) RETURN count(n)
- For listing: MATCH (n:Label) RETURN n LIMIT 10
- For finding patterns: MATCH (a)-[r]->(b) RETURN a, r, b
- For aggregations: MATCH (n) RETURN labels(n), count(n)

**For write_neo4j_cypher (WRITE QUERIES):**
- Use CREATE, MERGE, SET, DELETE, DETACH DELETE, REMOVE
- For creating nodes: CREATE (:Label {{property: 'value'}})
- For creating relationships: MATCH (a), (b) CREATE (a)-[:REL_TYPE]->(b)
- For updating: MATCH (n:Label) SET n.property = 'new_value'
- For deleting: MATCH (n:Label) DETACH DELETE n

**For get_neo4j_schema:**
- No Cypher query needed - this tool fetches schema automatically

**COMMON PATTERNS:**

Count nodes: `MATCH (n) RETURN count(n)`
Count by label: `MATCH (n) RETURN labels(n)[0] as label, count(n) ORDER BY count(n) DESC`
List all labels: Use get_neo4j_schema tool
Find connected nodes: `MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 10`
Most connected nodes: `MATCH (n) WITH n, size((n)--()) as degree ORDER BY degree DESC LIMIT 10 RETURN n, degree`
Create person: `CREATE (:Person {{name: '{name}', created: datetime()}})`
Update property: `MATCH (n:Person {{name: '{name}'}}) SET n.updated = datetime()`
Delete by label: `MATCH (n:TempLabel) DETACH DELETE n`

**RESPONSE FORMAT:**
Query: [cypher_query_on_single_line]

**IMPORTANT:** 
- Put the entire query on ONE line
- No code blocks or backticks
- For schema queries, respond with: Query: NO_QUERY_NEEDED

Generate the Cypher query:"""

RESULT_FORMATTING_PROMPT = """You are a result formatting expert. Format the Neo4j query result into a clear, user-friendly response.

USER QUESTION: {question}
TOOL USED: {tool}
CYPHER QUERY: {query}
RAW RESULT: {result}

**FORMATTING GUIDELINES:**

**For COUNT queries:**
- Format as: "📊 **Result:** [number]"
- Example: "📊 **Result:** 179 nodes found"

**For LIST queries:**
- Show up to 10 items clearly
- Format as numbered list for multiple items
- Example: "1. Person: Alice (age: 30)\n2. Person: Bob (age: 25)"

**For SCHEMA queries:**
- Group by categories: Labels, Relationships, Properties
- Use clear headers and bullet points
- Example: "**Node Labels:** Person, Company\n**Relationships:** WORKS_FOR, MANAGES"

**For WRITE operations:**
- Highlight what was changed
- Use success indicators
- Example: "✅ Created 1 node, Set 2 properties"

**For ANALYSIS queries:**
- Provide insights and context
- Highlight interesting findings
- Use charts/graphs concepts when relevant

**For ERROR cases:**
- Explain what went wrong clearly
- Suggest next steps or alternatives
- Use warning icons appropriately

**RESPONSE FORMAT:**
[formatted_user_friendly_response]

Format the result:"""

ERROR_ANALYSIS_PROMPT = """You are an error analysis expert for Neo4j queries. Analyze the error and suggest a solution.

USER QUESTION: {question}
TOOL: {tool}
QUERY: {query}
ERROR: {error}
ATTEMPT: {attempt}

**COMMON ERROR PATTERNS:**

**Syntax Errors:**
- Missing parentheses, brackets, or quotes
- Incorrect Cypher syntax
- Solution: Fix syntax and retry

**Connection Errors:**
- Neo4j database not available
- Network timeout issues
- Solution: Check database connection

**Permission Errors:**
- Insufficient privileges for operation
- Read-only mode restrictions
- Solution: Use appropriate permissions

**Data Errors:**
- Referenced nodes/relationships don't exist
- Constraint violations
- Solution: Modify query or create missing data

**Performance Errors:**
- Query too complex or slow
- Memory limitations
- Solution: Add LIMIT clause or optimize query

**RESPONSE FORMAT:**
Error_Type: [error_category]
Solution: [specific_solution]
Retry_Query: [corrected_query_if_applicable]
User_Message: [user-friendly_explanation]

Analyze the error:"""

# ============================================
# LLM COMMUNICATION
# ============================================

def call_cortex_llm(prompt: str, session_id: str, context: str = "") -> str:
    """Call Cortex LLM with the given prompt"""
    try:
        headers = {
            "Authorization": f'Snowflake Token="{API_KEY}"',
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": {
                "aplctn_cd": "edagnai",
                "app_id": "edadip",
                "api_key": API_KEY,
                "method": "cortex",
                "model": MODEL,
                "sys_msg": f"You are an expert Neo4j assistant. {context}",
                "limit_convs": "0",
                "prompt": {
                    "messages": [{"role": "user", "content": prompt}]
                },
                "session_id": session_id
            }
        }
        
        logger.info(f"Calling Cortex LLM: {prompt[:100]}...")
        response = requests.post(API_URL, headers=headers, json=payload, verify=False)
        
        if response.status_code == 200:
            result = response.text.partition("end_of_stream")[0].strip()
            logger.info(f"LLM response received: {len(result)} characters")
            return result
        else:
            logger.error(f"Cortex API error: {response.status_code}")
            return f"Error: Cortex API returned {response.status_code}"
            
    except Exception as e:
        logger.error(f"Cortex LLM call failed: {e}")
        return f"Error: Failed to call LLM - {str(e)}"

# ============================================
# MCP SERVER COMMUNICATION
# ============================================

def call_mcp_server(tool: str, query: str = None) -> Dict[str, Any]:
    """Call MCP server with the specified tool"""
    try:
        base_url = f"http://localhost:{MCP_PORT}"
        headers = {"Content-Type": "application/json"}
        
        logger.info(f"Calling MCP server - Tool: {tool}, Query: {query}")
        
        if tool == ToolType.SCHEMA:
            response = requests.post(f"{base_url}/get_neo4j_schema", headers=headers)
            
        elif tool == ToolType.READ:
            if not query:
                return {"error": "No query provided for read operation"}
            
            data = {"query": query, "params": {}}
            response = requests.post(f"{base_url}/read_neo4j_cypher", json=data, headers=headers)
            
        elif tool == ToolType.WRITE:
            if not query:
                return {"error": "No query provided for write operation"}
                
            data = {"query": query, "params": {}}
            response = requests.post(f"{base_url}/write_neo4j_cypher", json=data, headers=headers)
            
        else:
            return {"error": f"Unknown tool: {tool}"}
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"MCP server response: {type(result)}")
            return {"success": True, "data": result}
        else:
            logger.error(f"MCP server error: {response.status_code} - {response.text}")
            return {"error": f"MCP server error: {response.status_code}"}
            
    except Exception as e:
        logger.error(f"Failed to call MCP server: {e}")
        return {"error": f"Failed to call MCP server: {str(e)}"}

# ============================================
# UTILITY FUNCTIONS
# ============================================

def parse_llm_field(text: str, field: str) -> str:
    """Extract a specific field from LLM response"""
    try:
        pattern = f"{field}:\\s*(.+?)(?=\\n|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
        return ""
    except Exception:
        return ""

def clean_cypher_query(query: str) -> str:
    """Clean and format Cypher query"""
    if not query or query == "NO_QUERY_NEEDED":
        return ""
    
    # Remove code block markers
    query = re.sub(r'```[a-zA-Z]*', '', query)
    query = re.sub(r'```', '', query)
    
    # Clean whitespace
    query = re.sub(r'\s+', ' ', query)
    query = query.strip()
    
    return query

def should_retry(error_count: int, error: str) -> bool:
    """Determine if we should retry based on error"""
    if error_count >= 3:
        return False
    
    retry_errors = [
        "syntax error",
        "connection error", 
        "timeout",
        "network error"
    ]
    
    return any(retry_error in error.lower() for retry_error in retry_errors)

# ============================================
# LANGGRAPH NODES
# ============================================

def analyze_intent_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Analyze user intent"""
    logger.info(f"Node 1: Analyzing intent for: {state.question}")
    
    prompt = INTENT_ANALYSIS_PROMPT.format(question=state.question)
    response = call_cortex_llm(prompt, state.session_id, "Analyze user intent for Neo4j queries.")
    
    intent = parse_llm_field(response, "Intent")
    
    return {
        **state.dict(),
        "intent": intent,
        "analysis_prompt": prompt,
        "analysis_response": response,
        "next_step": "select_tool"
    }

def select_tool_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Select appropriate tool"""
    logger.info(f"Node 2: Selecting tool for intent: {state.intent}")
    
    prompt = TOOL_SELECTION_PROMPT.format(
        question=state.question,
        intent=state.intent
    )
    response = call_cortex_llm(prompt, state.session_id, "Select the appropriate Neo4j tool.")
    
    tool = parse_llm_field(response, "Tool")
    
    # Validate tool
    valid_tools = {ToolType.READ, ToolType.WRITE, ToolType.SCHEMA}
    if tool not in valid_tools:
        tool = ToolType.UNKNOWN
        
    next_step = "generate_query" if tool != ToolType.SCHEMA else "execute_tool"
    
    return {
        **state.dict(),
        "tool": tool,
        "next_step": next_step
    }

def generate_query_node(state: AgentState) -> Dict[str, Any]:
    """Node 3: Generate Cypher query"""
    logger.info(f"Node 3: Generating query for tool: {state.tool}")
    
    prompt = CYPHER_GENERATION_PROMPT.format(
        question=state.question,
        intent=state.intent,
        tool=state.tool
    )
    response = call_cortex_llm(prompt, state.session_id, "Generate precise Cypher queries.")
    
    query = parse_llm_field(response, "Query")
    query = clean_cypher_query(query)
    
    return {
        **state.dict(),
        "query": query,
        "query_prompt": prompt,
        "query_response": response,
        "next_step": "execute_tool"
    }

def execute_tool_node(state: AgentState) -> Dict[str, Any]:
    """Node 4: Execute tool on MCP server"""
    logger.info(f"Node 4: Executing tool: {state.tool}")
    
    result = call_mcp_server(state.tool, state.query)
    
    if "error" in result:
        return {
            **state.dict(),
            "execution_result": result,
            "last_error": result["error"],
            "error_count": state.error_count + 1,
            "next_step": "handle_error"
        }
    else:
        return {
            **state.dict(),
            "execution_result": result,
            "next_step": "format_result"
        }

def format_result_node(state: AgentState) -> Dict[str, Any]:
    """Node 5: Format result for user"""
    logger.info("Node 5: Formatting result")
    
    prompt = RESULT_FORMATTING_PROMPT.format(
        question=state.question,
        tool=state.tool,
        query=state.query,
        result=json.dumps(state.execution_result, default=str)
    )
    response = call_cortex_llm(prompt, state.session_id, "Format Neo4j results for users.")
    
    # Build trace
    trace_parts = []
    if state.intent:
        trace_parts.append(f"Intent: {state.intent}")
    if state.tool:
        trace_parts.append(f"Tool: {state.tool}")
    if state.query:
        trace_parts.append(f"Query: {state.query}")
    
    trace = " | ".join(trace_parts)
    
    return {
        **state.dict(),
        "formatted_result": response,
        "answer": response,
        "trace": trace,
        "next_step": "end"
    }

def handle_error_node(state: AgentState) -> Dict[str, Any]:
    """Node 6: Handle errors and retries"""
    logger.info(f"Node 6: Handling error (attempt {state.error_count})")
    
    if should_retry(state.error_count, state.last_error):
        prompt = ERROR_ANALYSIS_PROMPT.format(
            question=state.question,
            tool=state.tool,
            query=state.query,
            error=state.last_error,
            attempt=state.error_count
        )
        response = call_cortex_llm(prompt, state.session_id, "Analyze and fix Neo4j errors.")
        
        # Try to extract corrected query
        retry_query = parse_llm_field(response, "Retry_Query")
        if retry_query:
            retry_query = clean_cypher_query(retry_query)
            return {
                **state.dict(),
                "query": retry_query,
                "next_step": "execute_tool"
            }
    
    # No retry or max attempts reached
    user_message = parse_llm_field(response if 'response' in locals() else "", "User_Message")
    if not user_message:
        user_message = f"❌ Error: {state.last_error}"
    
    return {
        **state.dict(),
        "answer": user_message,
        "trace": f"Error after {state.error_count} attempts: {state.last_error}",
        "next_step": "end"
    }

def fallback_node(state: AgentState) -> Dict[str, Any]:
    """Fallback node for unknown states"""
    logger.warning(f"Fallback node reached with next_step: {state.next_step}")
    
    return {
        **state.dict(),
        "answer": "❌ I encountered an unexpected error. Please try rephrasing your question.",
        "trace": f"Fallback triggered - next_step was: {state.next_step}",
        "next_step": "end"
    }

# ============================================
# ROUTING FUNCTIONS
# ============================================

def route_after_analyze(state: AgentState) -> str:
    """Route after intent analysis"""
    if state.intent:
        return "select_tool"
    else:
        return "fallback"

def route_after_tool_selection(state: AgentState) -> str:
    """Route after tool selection"""
    if state.tool == ToolType.SCHEMA:
        return "execute_tool"
    elif state.tool in [ToolType.READ, ToolType.WRITE]:
        return "generate_query"
    else:
        return "fallback"

def route_after_execution(state: AgentState) -> str:
    """Route after tool execution"""
    if state.next_step == "handle_error":
        return "handle_error"
    elif state.next_step == "format_result":
        return "format_result"
    else:
        return "fallback"

def route_after_error(state: AgentState) -> str:
    """Route after error handling"""
    if state.next_step == "execute_tool":
        return "execute_tool"
    else:
        return END

def route_final(state: AgentState) -> str:
    """Final routing"""
    return END

# ============================================
# BUILD AGENT
# ============================================

def build_agent():
    """Build the complete LangGraph agent"""
    logger.info("Building complete LangGraph agent with all prompts...")
    
    # Create workflow
    workflow = StateGraph(state_schema=AgentState)
    
    # Add all nodes
    workflow.add_node("analyze_intent", RunnableLambda(analyze_intent_node))
    workflow.add_node("select_tool", RunnableLambda(select_tool_node))
    workflow.add_node("generate_query", RunnableLambda(generate_query_node))
    workflow.add_node("execute_tool", RunnableLambda(execute_tool_node))
    workflow.add_node("format_result", RunnableLambda(format_result_node))
    workflow.add_node("handle_error", RunnableLambda(handle_error_node))
    workflow.add_node("fallback", RunnableLambda(fallback_node))
    
    # Set entry point
    workflow.set_entry_point("analyze_intent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "analyze_intent",
        route_after_analyze,
        {
            "select_tool": "select_tool",
            "fallback": "fallback"
        }
    )
    
    workflow.add_conditional_edges(
        "select_tool", 
        route_after_tool_selection,
        {
            "generate_query": "generate_query",
            "execute_tool": "execute_tool",
            "fallback": "fallback"
        }
    )
    
    workflow.add_edge("generate_query", "execute_tool")
    
    workflow.add_conditional_edges(
        "execute_tool",
        route_after_execution,
        {
            "format_result": "format_result",
            "handle_error": "handle_error",
            "fallback": "fallback"
        }
    )
    
    workflow.add_edge("format_result", END)
    
    workflow.add_conditional_edges(
        "handle_error",
        route_after_error,
        {
            "execute_tool": "execute_tool",
            END: END
        }
    )
    
    workflow.add_edge("fallback", END)
    
    # Compile the workflow
    agent = workflow.compile()
    logger.info("✅ Complete LangGraph agent built successfully")
    
    return agent

# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    # Test the complete agent
    logger.info("Testing complete LangGraph agent...")
    
    agent = build_agent()
    
    test_questions = [
        "How many nodes are in the graph?",
        "Create a Person named TestUser", 
        "Show me the database schema",
        "List all Person nodes",
        "Find nodes with most relationships"
    ]
    
    for question in test_questions:
        print(f"\n{'='*50}")
        print(f"Testing: {question}")
        print(f"{'='*50}")
        
        test_state = AgentState(
            question=question,
            session_id="test_session"
        )
        
        try:
            result = agent.invoke(test_state)
            print(f"Tool: {result.get('tool', 'N/A')}")
            print(f"Query: {result.get('query', 'N/A')}")
            print(f"Answer: {result.get('answer', 'N/A')}")
        except Exception as e:
            print(f"Error: {e}")
