import requests
import urllib3
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import re
import json
import logging

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_langgraph_agent")

class AgentState(BaseModel):
    question: str
    session_id: str
    tool: str = ""
    query: str = ""
    trace: str = ""
    answer: str = ""

# Configuration
API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"  # Change this!
MODEL = "llama3.1-70b"
MCP_PORT = 8000

# Simple system message
SYS_MSG = """You are a Neo4j database assistant. Choose the right tool and generate a Cypher query.

TOOLS:
1. read_neo4j_cypher - For reading data (MATCH, RETURN, WHERE, COUNT, etc.)
2. write_neo4j_cypher - For writing data (CREATE, MERGE, SET, DELETE, etc.)  
3. get_neo4j_schema - For getting database structure info

RESPONSE FORMAT - Use this exact format:
Tool: [tool_name]
Query: [cypher_query]

EXAMPLES:

User: How many nodes are in the graph?
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN count(n)

User: Create a Person named Alice
Tool: write_neo4j_cypher
Query: CREATE (:Person {name: 'Alice'})

User: Show me the schema
Tool: get_neo4j_schema

IMPORTANT: Always use the exact format above. Put the entire Cypher query on one line after "Query: ".
"""

def clean_cypher_query(query: str) -> str:
    """Clean and format Cypher query"""
    if not query:
        return ""
    
    # Remove code block markers
    query = re.sub(r'```[a-zA-Z]*', '', query)
    query = re.sub(r'```', '', query)
    
    # Clean whitespace
    query = re.sub(r'\s+', ' ', query)
    query = query.strip()
    
    return query

def call_cortex_llm(prompt: str, session_id: str) -> str:
    """Call Cortex LLM API"""
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
                "sys_msg": SYS_MSG,
                "limit_convs": "0",
                "prompt": {
                    "messages": [{"role": "user", "content": prompt}]
                },
                "session_id": session_id
            }
        }
        
        logger.info(f"Calling Cortex LLM for: {prompt}")
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

def parse_llm_response(llm_output: str):
    """Parse LLM response to extract tool and query"""
    try:
        lines = llm_output.strip().split('\n')
        tool = ""
        query = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("Tool:"):
                tool = line.replace("Tool:", "").strip()
            elif line.startswith("Query:"):
                query = line.replace("Query:", "").strip()
                
        # Validate tool
        valid_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
        if tool not in valid_tools:
            logger.warning(f"Invalid tool: {tool}, trying to extract from text")
            # Try to find tool in the text
            for valid_tool in valid_tools:
                if valid_tool in llm_output.lower():
                    tool = valid_tool
                    break
        
        logger.info(f"Parsed - Tool: {tool}, Query: {query}")
        return tool, clean_cypher_query(query), llm_output
        
    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}")
        return "", "", llm_output

def call_mcp_server(tool: str, query: str = None):
    """Call MCP server with the specified tool"""
    try:
        base_url = f"http://localhost:{MCP_PORT}"
        headers = {"Content-Type": "application/json"}
        
        logger.info(f"Calling MCP server - Tool: {tool}, Query: {query}")
        
        if tool == "get_neo4j_schema":
            response = requests.post(f"{base_url}/get_neo4j_schema", headers=headers)
            
        elif tool == "read_neo4j_cypher":
            if not query:
                return {"error": "No query provided for read operation"}
            
            data = {"query": query, "params": {}}
            response = requests.post(f"{base_url}/read_neo4j_cypher", json=data, headers=headers)
            
        elif tool == "write_neo4j_cypher":
            if not query:
                return {"error": "No query provided for write operation"}
                
            data = {"query": query, "params": {}}
            response = requests.post(f"{base_url}/write_neo4j_cypher", json=data, headers=headers)
            
        else:
            return {"error": f"Unknown tool: {tool}"}
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"MCP server response: {type(result)} with {len(str(result))} chars")
            return result
        else:
            logger.error(f"MCP server error: {response.status_code} - {response.text}")
            return {"error": f"MCP server error: {response.status_code}"}
            
    except Exception as e:
        logger.error(f"Failed to call MCP server: {e}")
        return {"error": f"Failed to call MCP server: {str(e)}"}

def format_mcp_response(result, tool: str) -> str:
    """Format MCP server response for display"""
    try:
        if isinstance(result, dict) and "error" in result:
            return f"❌ Error: {result['error']}"
        
        if tool == "get_neo4j_schema":
            if isinstance(result, dict):
                parts = []
                if result.get("labels"):
                    parts.append(f"**Node Labels:** {', '.join(result['labels'])}")
                if result.get("relationship_types"):
                    parts.append(f"**Relationship Types:** {', '.join(result['relationship_types'])}")
                if result.get("property_keys"):
                    parts.append(f"**Property Keys:** {', '.join(result['property_keys'])}")
                return "\n".join(parts) if parts else "No schema information available"
            
        elif tool == "write_neo4j_cypher":
            if isinstance(result, dict):
                if result.get("success"):
                    parts = []
                    if result.get("nodes_created", 0) > 0:
                        parts.append(f"Created {result['nodes_created']} nodes")
                    if result.get("relationships_created", 0) > 0:
                        parts.append(f"Created {result['relationships_created']} relationships")
                    if result.get("properties_set", 0) > 0:
                        parts.append(f"Set {result['properties_set']} properties")
                    if result.get("nodes_deleted", 0) > 0:
                        parts.append(f"Deleted {result['nodes_deleted']} nodes")
                    if result.get("relationships_deleted", 0) > 0:
                        parts.append(f"Deleted {result['relationships_deleted']} relationships")
                    
                    return "✅ " + ", ".join(parts) if parts else "✅ Operation completed successfully"
                else:
                    return f"❌ Write operation failed"
                    
        elif tool == "read_neo4j_cypher":
            if isinstance(result, list):
                if not result:
                    return "No results found"
                
                # Handle single count results
                if len(result) == 1 and isinstance(result[0], dict) and len(result[0]) == 1:
                    key, value = list(result[0].items())[0]
                    if "count" in key.lower():
                        return f"📊 **Result:** {value}"
                    else:
                        return f"📊 **{key}:** {value}"
                
                # Handle multiple results
                if len(result) <= 10:
                    formatted_results = []
                    for i, item in enumerate(result, 1):
                        if isinstance(item, dict):
                            item_str = ", ".join([f"{k}: {v}" for k, v in item.items()])
                            formatted_results.append(f"{i}. {item_str}")
                        else:
                            formatted_results.append(f"{i}. {item}")
                    return "\n".join(formatted_results)
                else:
                    return f"Found {len(result)} results (showing first 10):\n" + format_mcp_response(result[:10], tool)
        
        # Fallback for any other format
        return str(result)
        
    except Exception as e:
        logger.error(f"Failed to format response: {e}")
        return f"❌ Error formatting response: {str(e)}"

def step_1_analyze_question(state: AgentState) -> dict:
    """Step 1: Analyze question and get tool/query from LLM"""
    logger.info(f"Step 1: Analyzing question: {state.question}")
    
    # Call LLM
    llm_response = call_cortex_llm(state.question, state.session_id)
    
    # Parse response
    tool, query, trace = parse_llm_response(llm_response)
    
    return {
        "question": state.question,
        "session_id": state.session_id,
        "tool": tool,
        "query": query,
        "trace": trace,
        "answer": ""
    }

def step_2_execute_tool(state: AgentState) -> dict:
    """Step 2: Execute the tool on MCP server"""
    logger.info(f"Step 2: Executing tool: {state.tool} with query: {state.query}")
    
    if not state.tool:
        answer = "❌ No valid tool selected. Please rephrase your question."
    else:
        # Call MCP server
        mcp_result = call_mcp_server(state.tool, state.query)
        
        # Format response
        answer = format_mcp_response(mcp_result, state.tool)
    
    return {
        "question": state.question,
        "session_id": state.session_id,
        "tool": state.tool,
        "query": state.query,
        "trace": state.trace,
        "answer": answer
    }

def build_agent():
    """Build the LangGraph agent"""
    logger.info("Building simple LangGraph agent...")
    
    # Create workflow
    workflow = StateGraph(state_schema=AgentState)
    
    # Add nodes
    workflow.add_node("analyze", RunnableLambda(step_1_analyze_question))
    workflow.add_node("execute", RunnableLambda(step_2_execute_tool))
    
    # Set entry point
    workflow.set_entry_point("analyze")
    
    # Add edges
    workflow.add_edge("analyze", "execute")
    workflow.add_edge("execute", END)
    
    # Compile
    agent = workflow.compile()
    logger.info("✅ Simple LangGraph agent built successfully")
    
    return agent

if __name__ == "__main__":
    # Test the agent
    agent = build_agent()
    
    test_state = AgentState(
        question="How many nodes are in the graph?",
        session_id="test_session"
    )
    
    print("Testing agent...")
    result = agent.invoke(test_state)
    print(f"Result: {result}")
