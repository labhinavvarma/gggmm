import re
import requests
import json
import logging
from langgraph.prebuilt import create_react_agent
from cypher_prompt import CYPHER_GENERATION_TEMPLATE
from dependencies import SnowFlakeConnector
from langchain_core.prompts.prompt import PromptTemplate
from llm_chat_wrapper import ChatSnowflakeCortex
from snowflake.snowpark import Session
from langchain_neo4j import Neo4jGraph
from typing import Any, Dict, List, Optional, Union
from neo4j_graphrag.schema import format_schema
from neo4j_graphrag.retrievers.text2cypher import extract_cypher
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field, BaseModel
from neo4j.exceptions import Neo4jError
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langgraph_neo4j_agent")

checkpointer = InMemorySaver()

# HTTP headers for Neo4j API calls
NEO4J_API_HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}
NEO4J_API_BASE_URL = "http://localhost:8000"

def clean_cypher_query(query: str) -> str:
    """Clean and format Cypher queries for execution"""
    query = re.sub(r'[\r\n]+', ' ', query)
    keywords = [
        "MATCH", "WITH", "RETURN", "ORDER BY", "UNWIND", "WHERE", "LIMIT",
        "SKIP", "CALL", "YIELD", "CREATE", "MERGE", "SET", "DELETE", "DETACH DELETE", "REMOVE"
    ]
    for kw in keywords:
        query = re.sub(rf'(?<!\s)({kw})', r' \1', query)
        query = re.sub(rf'({kw})([^\s\(])', r'\1 \2', query)
    query = re.sub(r'\s+', ' ', query)
    return query.strip()

def enhance_query_for_relationships(query: str) -> str:
    """Enhance queries to always include relationships for graph visualization"""
    query = query.strip()
    
    # If it's a simple MATCH (n) RETURN n, enhance it to include relationships
    if re.match(r'MATCH\s*\([^)]+\)\s*RETURN\s+[^L]+(?:LIMIT\s+\d+)?$', query, re.I):
        # Convert "MATCH (n:Person) RETURN n" to include relationships
        base_match = re.search(r'MATCH\s*\(([^)]+)\)', query, re.I)
        limit_match = re.search(r'LIMIT\s+(\d+)', query, re.I)
        
        if base_match:
            node_pattern = base_match.group(1)
            limit_clause = f" LIMIT {limit_match.group(1)}" if limit_match else " LIMIT 50"
            
            # Enhanced query that returns nodes and their relationships
            enhanced_query = f"""
            MATCH (n{':' + node_pattern if not node_pattern.startswith(':') and ':' in node_pattern else ''})
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n, r, m{limit_clause}
            """.strip()
            
            logger.info(f"Enhanced query from '{query}' to '{enhanced_query}'")
            return enhanced_query
    
    return query

def intelligent_keyword_fallback(text: str, fallback_query: str = None) -> tuple:
    """Intelligent keyword-based tool selection with relationship focus"""
    text_lower = text.lower()
    
    logger.info(f"🧠 Using intelligent fallback for: {text[:100]}...")
    
    # Schema-related keywords
    if any(word in text_lower for word in ["schema", "structure", "types", "labels", "properties", "what is in"]):
        return "get_neo4j_schema", ""
    
    # Write operation keywords
    elif any(word in text_lower for word in ["create", "add", "insert", "update", "set", "delete", "remove", "merge"]):
        if fallback_query and any(word in fallback_query.lower() for word in ["create", "merge", "set", "delete"]):
            return "write_neo4j_cypher", fallback_query
        else:
            return "write_neo4j_cypher", "CREATE (n {name: 'New Node'}) RETURN n"
    
    # Read operation keywords - ALWAYS include relationships
    else:
        if fallback_query:
            return "read_neo4j_cypher", enhance_query_for_relationships(fallback_query)
        elif "count" in text_lower or "how many" in text_lower:
            return "read_neo4j_cypher", "MATCH (n) RETURN count(n) AS node_count"
        else:
            # Default query that always includes relationships
            return "read_neo4j_cypher", "MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 50"

def execute_robust_http_request(endpoint: str, data: dict = None, timeout: int = 45) -> dict:
    """Execute robust HTTP request with error handling"""
    try:
        url = f"{NEO4J_API_BASE_URL}/{endpoint}"
        
        if data is None:
            response = requests.post(url, headers=NEO4J_API_HEADERS, timeout=timeout)
        else:
            response = requests.post(url, json=data, headers=NEO4J_API_HEADERS, timeout=timeout)
        
        if response.ok:
            return {"success": True, "data": response.json()}
        else:
            logger.error(f"❌ HTTP request failed: {response.text}")
            return {"success": False, "error": f"HTTP request failed: {response.text}"}
            
    except requests.exceptions.Timeout:
        logger.error("⏰ Request timed out")
        return {"success": False, "error": "Request timed out. Try a simpler query or reduce the data scope."}
    except requests.exceptions.ConnectionError:
        logger.error("🔌 Connection error") 
        return {"success": False, "error": "Cannot connect to the database server. Please check if all services are running."}
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

#Need to change the function to perform the async operations
def get_graph_ref() -> Neo4jGraph:
    graph = Neo4jGraph(
        url="neo4j://localhost:7687",
        username="neo4j",
        password="password",
        database="neo4j",
        enhanced_schema=True,
    )
    return graph

graph = get_graph_ref()

def _is_write_query(query: str) -> bool:
    """Check if the query is a write query."""
    return (
        re.search(r"\b(MERGE|CREATE|SET|DELETE|REMOVE|ADD)\b", query, re.IGNORECASE)
        is not None
    )

@tool(
    "read_neo4j_cypher",
    description="Executes a read Cypher queries on the Neo4j database and returns the dictionary of Cypher query index and corresponding results.",
    return_direct=True
)
def read_neo4j_cypher(
    query: str,
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Execute a read Cypher query via HTTP API with robust error handling."""
    if params is None:
        params = {}
    
    # Clean and enhance query
    query_clean = clean_cypher_query(query)
    query_enhanced = enhance_query_for_relationships(query_clean)
    
    data = {
        "query": query_enhanced,
        "params": params,
        "node_limit": 1000
    }
    
    logger.info(f"📖 Executing read query: {query_enhanced[:100]}...")
    
    result = execute_robust_http_request("read_neo4j_cypher", data)
    
    if result["success"]:
        result_data = result["data"]
        logger.info("✅ Read query executed successfully")
        
        # Extract the actual data from the response
        if "data" in result_data:
            return result_data["data"]
        return result_data
    else:
        logger.error(f"❌ Read query failed: {result['error']}")
        # Try fallback query
        fallback_tool, fallback_query = intelligent_keyword_fallback(query, query)
        if fallback_query and fallback_query != query_enhanced:
            logger.info("🔄 Trying fallback query...")
            fallback_data = {"query": fallback_query, "params": params, "node_limit": 1000}
            fallback_result = execute_robust_http_request("read_neo4j_cypher", fallback_data)
            if fallback_result["success"]:
                return fallback_result["data"].get("data", fallback_result["data"])
        
        return [{"error": result["error"]}]

@tool(
    "write_neo4j_cypher",
    description="Executes a write Cypher query (CREATE, MERGE, SET, DELETE) on the Neo4j database.",
    return_direct=True
)
def write_neo4j_cypher(
    query: str,
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Execute a write Cypher query via HTTP API with robust error handling."""
    if params is None:
        params = {}
    
    # Clean query
    query_clean = clean_cypher_query(query)
    
    data = {
        "query": query_clean,
        "params": params,
        "node_limit": 1000
    }
    
    logger.info(f"✏️ Executing write query: {query_clean[:100]}...")
    
    result = execute_robust_http_request("write_neo4j_cypher", data)
    
    if result["success"]:
        logger.info("✅ Write query executed successfully")
        return result["data"]
    else:
        logger.error(f"❌ Write query failed: {result['error']}")
        return [{"error": result["error"]}]

@tool(
    "get_neo4j_schema",
    description="Retrieves the Neo4j database schema information including node types, relationships, and properties.",
    return_direct=True
)
def get_neo4j_schema() -> Dict[str, Any]:
    """Get Neo4j schema via HTTP API with robust error handling."""
    logger.info("📋 Getting database schema...")
    
    result = execute_robust_http_request("get_neo4j_schema")
    
    if result["success"]:
        logger.info("✅ Schema retrieved successfully")
        return result["data"]
    else:
        logger.error(f"❌ Schema query failed: {result['error']}")
        return {"error": result["error"]}

def construct_schema(
    structured_schema: Dict[str, Any],
    include_types: List[str],
    exclude_types: List[str],
    is_enhanced: bool,
) -> str:
    """Filter the schema based on included or excluded types"""

    def filter_func(x: str) -> bool:
        return x in include_types if include_types else x not in exclude_types

    filtered_schema: Dict[str, Any] = {
        "node_props": {
            k: v
            for k, v in structured_schema.get("node_props", {}).items()
            if filter_func(k)
        },
        "rel_props": {
            k: v
            for k, v in structured_schema.get("rel_props", {}).items()
            if filter_func(k)
        },
        "relationships": [
            r
            for r in structured_schema.get("relationships", [])
            if all(filter_func(r[t]) for t in ["start", "end", "type"])
        ],
    }
    return format_schema(filtered_schema, is_enhanced)

def get_structured_schema_from_api() -> Dict[str, Any]:
    """Get structured schema from the HTTP API with robust error handling"""
    logger.info("🏗️ Getting structured schema from API...")
    
    result = execute_robust_http_request("get_neo4j_schema", timeout=30)
    
    if result["success"]:
        schema_data = result["data"]
        schema = schema_data.get("schema", {})
        
        # Transform to the expected structure
        structured_schema = {
            "node_props": {},
            "rel_props": {},
            "relationships": []
        }
        
        # Extract node properties and relationships from schema
        for node_type, info in schema.items():
            if isinstance(info, dict):
                structured_schema["node_props"][node_type] = info.get("properties", {})
                
                # Extract relationships
                relationships = info.get("relationships", {})
                for rel_type, rel_info in relationships.items():
                    if isinstance(rel_info, list):
                        for rel in rel_info:
                            structured_schema["relationships"].append({
                                "start": node_type,
                                "type": rel_type,
                                "end": rel.get("target", "Unknown")
                            })
        
        logger.info("✅ Structured schema retrieved successfully")
        return structured_schema
    else:
        logger.error(f"❌ Failed to get structured schema: {result['error']}")
        return {"node_props": {}, "rel_props": {}, "relationships": []}

sf_conn = SnowFlakeConnector.get_conn(
            'aedl',
            '',
)
 
#LLM Model  
model = ChatSnowflakeCortex(
    model="claude-4-sonnet",
    cortex_function="complete",
    session=Session.builder.configs({"connection": sf_conn}).getOrCreate()
)

async def run():
    logger.info("🚀 Starting Neo4j LangGraph Agent...")

    custom_tool_node = ToolNode(
        [read_neo4j_cypher, write_neo4j_cypher, get_neo4j_schema],
        handle_tool_errors="Tool error"
    )

    agent = create_react_agent(
        model=model,
        tools=custom_tool_node,
        prompt="You are expert in generating accurate neo4j cypher queries with robust error handling and relationship focus",
    )

    # Get structured schema from API with error handling
    try:
        structured_schema = get_structured_schema_from_api()
        logger.info("✅ Schema loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load schema: {e}")
        structured_schema = {"node_props": {}, "rel_props": {}, "relationships": []}

    args_var = {
        "schema": construct_schema(
            structured_schema,
            include_types=["Table","Column","Domain","System","Kw","HAS_CHILD","HAS_COLUMN","HAS_KNOWLEDGE","HAS_SYSTEM","HAS_TABLE","HAS_DOMAIN"],
            exclude_types=[],
            is_enhanced=True
        ),
        "question": "Identify rule runs and results for crem system"
    }

    CYPHER_GENERATION_PROMPT = CYPHER_GENERATION_TEMPLATE.format(**args_var)
   
    #Manage to send the latest conversation message received.
    messages = [
        {"role": "user", "content": CYPHER_GENERATION_PROMPT},
        {"role": "assistant", "content": "Hi there! I can help with modeling your data and get relation information between entities.\n\nLet me analyze your question: \"Identify rule runs and results for crem system\"\n\n**Identified entities:**\n- **System**: CREM\n- **Knowledge (Kw)**: RULE_RUNS, RULE_RESULT\n\nI\'ve identified the CREM system and two knowledge entities - RULE_RUNS (which describes rule runs by identifying rule execution instances) and RULE_RESULT (which captures the numerator and denominator for each rule run).\n\nCan you confirm that these are the correct System and Knowledge entities you\'re looking for, or would you like me to include any additional knowledge entities?"},
        {"role": "user", "content": "I confirm these are the System and Knowledge entities i am looking for."}
    ]

    try:
        logger.info("🤖 Invoking agent with conversation messages...")
        response = await agent.ainvoke(
            {"messages": messages},
            print_mode="messages"
        )
        
        logger.info("✅ Agent execution completed successfully")
        
        print("\n**********Printing the message state**********\n")
        print(response["messages"])
        print("\n**********End of response state **********\n")
        
    except Exception as e:
        logger.error(f"❌ Agent execution failed: {e}")
        print(f"Agent execution failed: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())

CYPHER_GENERATION_TEMPLATE = """
    Using the provided graph database schema, generate accurate Neo4j Cypher queries for the given user query.

    Tools:
      - 'read_neo4j_cypher': Executes a read Cypher query on the Neo4j database.
      - 'write_neo4j_cypher': Executes a write Cypher query on the Neo4j database.
      - 'get_neo4j_schema': Retrieves the database schema information.

   <workflow>
    1. Schema Exploration:
       - Analyze the provided schema to understand the graph structure.
       - Understand types of nodes and relationships in the schema.

    2. Identify System and Knowledge entities:
       - Parse user question
       - Identify the System and corresponding knowledge(Kw) entities according to the provided schema
       - Ensure that all identified elements map accurately to the schema definitions and structure.
       - If System missing from user question, return an error message indicating the missing System.
       - If Knowledge(Kw) missing from user question, return an error message indicating the missing Knowledge(Kw).

    3. Generate Cyphers:
         Task: For each Knowledge(Kw) identified, generate Neo4j Cypher queries by performing the following steps.
            1. Extract all the connected nodes with relation HAS_TABLE.
            2. Filter the results for a System identified.
            3. Ensure the query is syntactically correct.
         Repeat these steps, once for each Knowledge(Kw).
         
         input: output from Identify System and Knowledge entities

    4. Execute Cyphers:
         Task: For each generated Cypher query, perform the following steps.
            1. Tool 'read_neo4j_cypher' provided to execute the Cypher against the Graph database.
            2. Execute it against the Neo4j database utilizing the Tool provided.
            3. Make no assumption and ensure the Cypher queries passed from Generate Cyphers is executed as intended.
            4. Return the result from the tool execution.
            5. if tool execution fails with error, log the error and continue with the next Cypher.
         Repeat these steps, once for each Cypher.

         input: output from Generate Cyphers
   </workflow>

    <conversation-flow>
    1. Start with: 'Hi there i can help with modeling your data and get relation information between entities?'
    2. After Identifying System and Knowledge(Kw):
       - Confirm the identified System and Knowledge(Kw) entities with the user
       - If error encountered with missing System, ask the user to provide the missing System entity.
       - If error encountered with missing Knowledge(Kw), ask the user to provide the missing Knowledge(Kw) entity.
    3. After Cypher Generation:
       - Provide your interpretation of the ask.
    4. After Cypher Execution:
       - Present the results to the user, update the next step as finding the relation between entities.
       - If no results found, inform the user and suggest refining the query.
       - If there are errors in execution, provide feedback to the user.
    </conversation-flow>

    Do not:
      - Make assumptions about database structure.
      - Use any nodes, relationships, properties not provided in the schema.
      - Generate Cypher queries without confirming the identified System and Knowledge(Kw) entities with the user.

    Here is the Graph database schema you should be aware of:
    <schema>
    {schema}
    </schema>
   
    The question is:
    {question}"""
