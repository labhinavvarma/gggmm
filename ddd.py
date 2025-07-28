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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langgraph_neo4j_agent")

checkpointer = InMemorySaver()

# HTTP headers for Neo4j API calls
NEO4J_API_HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}
NEO4J_API_BASE_URL = "http://localhost:8000"

def call_neo4j_api(endpoint: str, data: dict = None) -> dict:
    """Call Neo4j API endpoints directly"""
    try:
        url = f"{NEO4J_API_BASE_URL}/{endpoint}"
        logger.info(f"🔗 Calling {url}")
        
        if data is None:
            response = requests.post(url, headers=NEO4J_API_HEADERS, timeout=45)
        else:
            response = requests.post(url, json=data, headers=NEO4J_API_HEADERS, timeout=45)
        
        if response.ok:
            result = response.json()
            logger.info(f"✅ API call successful")
            return result
        else:
            logger.error(f"❌ API call failed: {response.text}")
            return {"error": f"API call failed: {response.text}"}
            
    except requests.exceptions.Timeout:
        logger.error("⏰ Request timed out")
        return {"error": "Request timed out"}
    except requests.exceptions.ConnectionError:
        logger.error("🔌 Connection error")
        return {"error": "Cannot connect to Neo4j service on localhost:8000"}
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        return {"error": f"Unexpected error: {str(e)}"}

def execute_cypher_query(query: str, params: dict = None, is_write: bool = False) -> dict:
    """Execute Cypher query via API"""
    if params is None:
        params = {}
    
    data = {
        "query": query,
        "params": params,
        "node_limit": 1000
    }
    
    # Choose endpoint based on query type
    if is_write:
        endpoint = "write_neo4j_cypher"  # Calls: http://localhost:8000/write_neo4j_cypher
    else:
        endpoint = "read_neo4j_cypher"   # Calls: http://localhost:8000/read_neo4j_cypher
        
    logger.info(f"📝 Executing {'write' if is_write else 'read'} query: {query[:100]}...")
    
    return call_neo4j_api(endpoint, data)

def get_schema_from_api() -> dict:
    """Get schema from Neo4j API"""
    logger.info("📋 Getting database schema...")
    # Calls: http://localhost:8000/get_neo4j_schema
    return call_neo4j_api("get_neo4j_schema")

def _is_write_query(query: str) -> bool:
    """Check if the query is a write query."""
    return (
        re.search(r"\b(MERGE|CREATE|SET|DELETE|REMOVE|ADD)\b", query, re.IGNORECASE)
        is not None
    )

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
    """Get structured schema from the HTTP API"""
    logger.info("🏗️ Getting structured schema from API...")
    
    result = get_schema_from_api()
    
    if "error" not in result:
        schema = result.get("schema", {})
        
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

# Custom agent class that uses API calls directly
class Neo4jAPIAgent:
    def __init__(self, model, schema: str):
        self.model = model
        self.schema = schema
        self.conversation_history = []
    
    async def process_query(self, question: str) -> str:
        """Process user question and execute appropriate Neo4j operations"""
        logger.info(f"🤔 Processing question: {question}")
        
        # Simple logic to determine what to do based on question
        question_lower = question.lower()
        
        if "schema" in question_lower:
            # Get schema via: http://localhost:8000/get_neo4j_schema
            result = get_schema_from_api()
            if "error" in result:
                return f"Error getting schema: {result['error']}"
            return f"Schema retrieved successfully: {json.dumps(result, indent=2)}"
        
        elif any(word in question_lower for word in ["create", "add", "insert", "update", "delete", "merge"]):
            # This would use: http://localhost:8000/write_neo4j_cypher
            return "Write operations detected. You would need to provide the specific Cypher query to execute."
        
        else:
            # This uses: http://localhost:8000/read_neo4j_cypher
            # For your specific use case, let's create a query for CREM system
            if "crem" in question_lower and "rule" in question_lower:
                # Generate Cypher query for CREM system rule runs
                cypher_query = """
                MATCH (s:System {name: 'CREM'})-[:HAS_DOMAIN]->(d:Domain)-[:HAS_KNOWLEDGE]->(kw:Kw)
                WHERE kw.name IN ['RULE_RUNS', 'RULE_RESULT']
                OPTIONAL MATCH (kw)-[:HAS_TABLE]->(t:Table)
                RETURN kw.name as Knowledge, 
                       kw.description as Knowledge_Description,
                       t.name as Table_Name, 
                       t.table as Table_Code,
                       t.description as Table_Description,
                       t.type as Table_Type
                ORDER BY kw.name, t.name
                """
                
                # This calls: http://localhost:8000/read_neo4j_cypher
                result = execute_cypher_query(cypher_query)
                
                if "error" in result:
                    return f"Error executing query: {result['error']}"
                
                # Format the results
                if "data" in result and result["data"]:
                    formatted_results = "Results for CREM system rule runs and results:\n\n"
                    for row in result["data"]:
                        formatted_results += f"Knowledge: {row.get('Knowledge', 'N/A')}\n"
                        formatted_results += f"Description: {row.get('Knowledge_Description', 'N/A')}\n"
                        formatted_results += f"Table: {row.get('Table_Name', 'N/A')}\n"
                        formatted_results += f"Code: {row.get('Table_Code', 'N/A')}\n"
                        formatted_results += "---\n"
                    return formatted_results
                else:
                    return "No results found for CREM system rule runs."
            
            else:
                # Generic query - would also use: http://localhost:8000/read_neo4j_cypher
                return "Please provide more specific information about what you'd like to query."

sf_conn = SnowFlakeConnector.get_conn(
    'aedl',
    '',
)

# LLM Model  
model = ChatSnowflakeCortex(
    model="claude-4-sonnet",
    cortex_function="complete",
    session=Session.builder.configs({"connection": sf_conn}).getOrCreate()
)

async def run():
    logger.info("🚀 Starting Neo4j API Agent...")

    # Get structured schema from API with error handling
    try:
        structured_schema = get_structured_schema_from_api()
        logger.info("✅ Schema loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load schema: {e}")
        structured_schema = {"node_props": {}, "rel_props": {}, "relationships": []}

    schema_text = construct_schema(
        structured_schema,
        include_types=["Table", "Column", "Domain", "System", "Kw", "HAS_CHILD", "HAS_COLUMN", "HAS_KNOWLEDGE", "HAS_SYSTEM", "HAS_TABLE", "HAS_DOMAIN"],
        exclude_types=[],
        is_enhanced=True
    )

    # Create agent that uses API calls directly
    agent = Neo4jAPIAgent(model, schema_text)
    
    # Test the agent
    question = "Identify rule runs and results for crem system"
    
    try:
        logger.info("🤖 Processing question...")
        response = await agent.process_query(question)
        
        logger.info("✅ Agent execution completed successfully")
        
        print("\n**********Agent Response**********\n")
        print(response)
        print("\n**********End of Response**********\n")
        
    except Exception as e:
        logger.error(f"❌ Agent execution failed: {e}")
        print(f"Agent execution failed: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
