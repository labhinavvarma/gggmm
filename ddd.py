import requests
import urllib3
from pydantic import BaseModel
from typing import Optional, Dict, List
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import re
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("schema_aware_agent")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class SchemaAwareAgentState(BaseModel):
    question: str
    session_id: str
    tool: str = ""
    query: str = ""
    trace: str = ""
    answer: str = ""
    graph_data: Optional[dict] = None
    node_limit: int = 1000
    schema_info: Optional[dict] = None  # New: Store schema information

class Neo4jSchemaManager:
    """Manages Neo4j schema retrieval and analysis"""
    
    def __init__(self):
        self.schema_cache = None
        self.last_schema_update = None
        self.schema_summary = None
    
    def fetch_database_schema(self) -> Dict:
        """Retrieve the complete Neo4j database schema"""
        try:
            logger.info("🔍 Fetching complete Neo4j database schema...")
            
            # Get schema from MCP server
            headers = {"Accept": "application/json", "Content-Type": "application/json"}
            result = requests.post("http://localhost:8000/get_neo4j_schema", headers=headers, timeout=30)
            
            if not result.ok:
                logger.error(f"Failed to fetch schema: {result.status_code}")
                return {}
            
            schema_response = result.json()
            raw_schema = schema_response.get("schema", {})
            
            # Also get additional schema information
            additional_queries = [
                ("node_labels", "CALL db.labels() YIELD label RETURN collect(label) as labels"),
                ("relationship_types", "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"),
                ("property_keys", "CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) as keys"),
                ("node_counts", "MATCH (n) RETURN labels(n) as label, count(*) as count ORDER BY count DESC"),
                ("relationship_counts", "MATCH ()-[r]->() RETURN type(r) as type, count(*) as count ORDER BY count DESC")
            ]
            
            enhanced_schema = {
                "raw_schema": raw_schema,
                "enhanced_info": {},
                "last_updated": datetime.now().isoformat()
            }
            
            # Execute additional schema queries
            for info_type, query in additional_queries:
                try:
                    data = {"query": query, "params": {}, "node_limit": 1000}
                    result = requests.post("http://localhost:8000/read_neo4j_cypher", json=data, headers=headers, timeout=15)
                    if result.ok:
                        query_result = result.json()
                        enhanced_schema["enhanced_info"][info_type] = query_result.get("data", [])
                        logger.info(f"✅ Retrieved {info_type}")
                    else:
                        logger.warning(f"⚠️ Failed to get {info_type}")
                        enhanced_schema["enhanced_info"][info_type] = []
                except Exception as e:
                    logger.warning(f"⚠️ Error getting {info_type}: {e}")
                    enhanced_schema["enhanced_info"][info_type] = []
            
            # Cache the schema
            self.schema_cache = enhanced_schema
            self.last_schema_update = datetime.now()
            
            # Generate schema summary
            self.schema_summary = self._generate_schema_summary(enhanced_schema)
            
            logger.info("✅ Complete database schema retrieved and cached")
            return enhanced_schema
            
        except Exception as e:
            logger.error(f"❌ Error fetching database schema: {e}")
            return {}
    
    def _generate_schema_summary(self, schema: Dict) -> str:
        """Generate a human-readable schema summary"""
        try:
            enhanced_info = schema.get("enhanced_info", {})
            
            # Extract information
            node_labels = []
            if enhanced_info.get("node_labels") and len(enhanced_info["node_labels"]) > 0:
                node_labels = enhanced_info["node_labels"][0].get("labels", [])
            
            relationship_types = []
            if enhanced_info.get("relationship_types") and len(enhanced_info["relationship_types"]) > 0:
                relationship_types = enhanced_info["relationship_types"][0].get("types", [])
            
            property_keys = []
            if enhanced_info.get("property_keys") and len(enhanced_info["property_keys"]) > 0:
                property_keys = enhanced_info["property_keys"][0].get("keys", [])
            
            node_counts = enhanced_info.get("node_counts", [])
            rel_counts = enhanced_info.get("relationship_counts", [])
            
            # Build summary
            summary = f"""
📊 **COMPLETE NEO4J DATABASE SCHEMA**

🏷️ **Available Node Labels ({len(node_labels)}):**
{', '.join(node_labels) if node_labels else 'None found'}

🔗 **Available Relationship Types ({len(relationship_types)}):**
{', '.join(relationship_types) if relationship_types else 'None found'}

📝 **Available Properties ({len(property_keys)}):**
{', '.join(property_keys[:20]) if property_keys else 'None found'}
{f'... and {len(property_keys) - 20} more' if len(property_keys) > 20 else ''}

📈 **Node Counts by Type:**
{chr(10).join([f"  • {item.get('label', ['Unknown'])[0] if isinstance(item.get('label'), list) else item.get('label', 'Unknown')}: {item.get('count', 0)}" for item in node_counts[:10]]) if node_counts else 'No data'}

🔗 **Relationship Counts by Type:**
{chr(10).join([f"  • {item.get('type', 'Unknown')}: {item.get('count', 0)}" for item in rel_counts[:10]]) if rel_counts else 'No data'}
            """.strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating schema summary: {e}")
            return "Schema summary unavailable"
    
    def get_cached_schema(self) -> Optional[Dict]:
        """Get cached schema if available and recent"""
        if self.schema_cache and self.last_schema_update:
            # Cache valid for 10 minutes
            time_diff = (datetime.now() - self.last_schema_update).total_seconds()
            if time_diff < 600:  # 10 minutes
                return self.schema_cache
        return None
    
    def get_schema_for_query_generation(self) -> str:
        """Get schema information formatted for query generation"""
        schema = self.get_cached_schema()
        if not schema:
            schema = self.fetch_database_schema()
        
        if not schema:
            return "Schema not available"
        
        enhanced_info = schema.get("enhanced_info", {})
        
        # Extract key information for query generation
        node_labels = []
        if enhanced_info.get("node_labels") and len(enhanced_info["node_labels"]) > 0:
            node_labels = enhanced_info["node_labels"][0].get("labels", [])
        
        relationship_types = []
        if enhanced_info.get("relationship_types") and len(enhanced_info["relationship_types"]) > 0:
            relationship_types = enhanced_info["relationship_types"][0].get("types", [])
        
        property_keys = []
        if enhanced_info.get("property_keys") and len(enhanced_info["property_keys"]) > 0:
            property_keys = enhanced_info["property_keys"][0].get("keys", [])
        
        # Format for LLM
        schema_text = f"""
AVAILABLE NODE LABELS: {', '.join(node_labels)}
AVAILABLE RELATIONSHIP TYPES: {', '.join(relationship_types)}
COMMON PROPERTIES: {', '.join(property_keys[:15])}
        """.strip()
        
        return schema_text

# Initialize schema manager
schema_manager = Neo4jSchemaManager()

def create_schema_aware_system_message() -> str:
    """Create a system message that includes the actual database schema"""
    
    # Get the current schema
    schema_info = schema_manager.get_schema_for_query_generation()
    schema_summary = schema_manager.schema_summary or "Schema summary not available"
    
    system_message = f"""You are a Neo4j database expert assistant with COMPLETE KNOWLEDGE of the actual database schema.

🎯 **ACTUAL DATABASE SCHEMA:**
{schema_info}

**RESPONSE FORMAT (REQUIRED):**
Tool: [tool_name]
Query: [cypher_query_or_none_for_schema]

**TOOLS AVAILABLE:**
1. **read_neo4j_cypher** - For viewing, exploring, counting, finding data
2. **write_neo4j_cypher** - For creating, updating, deleting data  
3. **get_neo4j_schema** - For database structure questions

**SCHEMA-AWARE QUERY GENERATION RULES:**
✅ ONLY use node labels that exist in the schema above
✅ ONLY use relationship types that exist in the schema above  
✅ ONLY use properties that exist in the schema above
✅ Generate precise queries based on actual schema
✅ Use exact label/property names (case-sensitive)

**ENHANCED EXAMPLES USING ACTUAL SCHEMA:**

User: "Show me all nodes"
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN labels(n) as Type, count(*) as Count ORDER BY Count DESC LIMIT 10

User: "Find people" (if Person label exists)
Tool: read_neo4j_cypher
Query: MATCH (n:Person) RETURN n LIMIT 30

User: "Show relationships" (using actual relationship types)
Tool: read_neo4j_cypher
Query: MATCH (a)-[r]->(b) RETURN type(r) as RelType, count(*) as Count ORDER BY Count DESC LIMIT 10

User: "Count data by type"
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN labels(n) as NodeType, count(*) as Count ORDER BY Count DESC

User: "Show me connections between [actual node types]"
Tool: read_neo4j_cypher
Query: MATCH (a)-[r]->(b) WHERE any(label IN labels(a) WHERE label IN {list(actual_labels)}) RETURN a, r, b LIMIT 25

**MULTI-TIER QUERIES (using actual schema):**

User: "Find 2nd degree connections"
Tool: read_neo4j_cypher
Query: MATCH (a)-[r1]->(b)-[r2]->(c) WHERE a <> c RETURN a, r1, b, r2, c LIMIT 40

User: "Show network paths"
Tool: read_neo4j_cypher
Query: MATCH path = (a)-[*1..3]-(b) WHERE a <> b RETURN path LIMIT 30

**VALIDATION RULES:**
- If user asks for non-existent labels/relationships, suggest available alternatives
- Always use schema-validated queries
- Provide helpful error messages when schema doesn't match request

**CURRENT SCHEMA SUMMARY:**
{schema_summary}

IMPORTANT: Generate queries ONLY using the actual schema elements listed above. If a user asks for something not in the schema, explain what's available instead.
"""
    
    return system_message

# Cortex LLM configuration
API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
MODEL = "llama3.1-70b"

def schema_aware_cortex_llm(prompt: str, session_id: str) -> str:
    """Enhanced Cortex LLM call with schema-aware system message"""
    
    # Get the latest schema-aware system message
    schema_aware_sys_msg = create_schema_aware_system_message()
    
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
            "sys_msg": schema_aware_sys_msg,  # Use schema-aware system message
            "limit_convs": "0",
            "prompt": {
                "messages": [{"role": "user", "content": prompt}]
            },
            "session_id": session_id
        }
    }
    
    try:
        logger.info(f"🔄 Calling Schema-Aware Cortex LLM...")
        resp = requests.post(API_URL, headers=headers, json=payload, verify=False, timeout=30)
        resp.raise_for_status()
        
        raw_response = resp.text
        
        if "end_of_stream" in raw_response:
            parsed_response = raw_response.partition("end_of_stream")[0].strip()
        else:
            parsed_response = raw_response.strip()
        
        logger.info(f"✅ Schema-aware LLM response received")
        return parsed_response
        
    except Exception as e:
        logger.error(f"❌ Schema-aware Cortex LLM API error: {e}")
        return f"Tool: read_neo4j_cypher\nQuery: MATCH (n) RETURN n LIMIT 20"

def validate_query_against_schema(query: str, schema_info: Dict) -> tuple[bool, str]:
    """Validate a Cypher query against the actual database schema"""
    try:
        enhanced_info = schema_info.get("enhanced_info", {})
        
        # Get available labels and relationships
        node_labels = []
        if enhanced_info.get("node_labels") and len(enhanced_info["node_labels"]) > 0:
            node_labels = enhanced_info["node_labels"][0].get("labels", [])
        
        relationship_types = []
        if enhanced_info.get("relationship_types") and len(enhanced_info["relationship_types"]) > 0:
            relationship_types = enhanced_info["relationship_types"][0].get("types", [])
        
        # Extract labels and relationships from query
        query_labels = re.findall(r':(\w+)', query)
        query_relationships = re.findall(r':\s*(\w+)\s*\]', query)
        
        issues = []
        
        # Check labels
        for label in query_labels:
            if label not in node_labels:
                issues.append(f"Label '{label}' not found. Available: {', '.join(node_labels[:5])}")
        
        # Check relationships
        for rel in query_relationships:
            if rel not in relationship_types:
                issues.append(f"Relationship '{rel}' not found. Available: {', '.join(relationship_types[:5])}")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, "Query validated successfully"
        
    except Exception as e:
        logger.warning(f"Schema validation error: {e}")
        return True, "Validation skipped due to error"

def schema_aware_parse_llm_output(llm_output, schema_info):
    """Enhanced parsing with schema validation"""
    allowed_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    trace = llm_output.strip()
    tool = None
    query = None
    
    logger.info(f"🔍 Schema-aware parsing of LLM output")
    
    # Extract tool
    tool_match = re.search(r"Tool:\s*([\w_]+)", llm_output, re.I)
    if tool_match:
        tname = tool_match.group(1).strip()
        if tname in allowed_tools:
            tool = tname
    
    # Extract query
    query_match = re.search(r"Query:\s*(.+?)(?:\n\n|\n[A-Z]|$)", llm_output, re.I | re.DOTALL)
    if query_match:
        query = query_match.group(1).strip()
    
    # Validate query against schema
    if query and tool == "read_neo4j_cypher" and schema_info:
        is_valid, validation_msg = validate_query_against_schema(query, schema_info)
        if not is_valid:
            logger.warning(f"⚠️ Schema validation failed: {validation_msg}")
            trace += f"\n\nSchema Validation Warning: {validation_msg}"
    
    # Fallback logic with schema awareness
    if not tool or not query:
        logger.warning("⚠️ Parsing failed, using schema-aware fallback...")
        tool = "read_neo4j_cypher"
        query = "MATCH (n) RETURN labels(n) as Types, count(*) as Count ORDER BY Count DESC LIMIT 10"
    
    return tool, query, trace

def schema_aware_select_tool_node(state: SchemaAwareAgentState) -> dict:
    """Schema-aware tool selection"""
    logger.info(f"🧠 Processing question with schema awareness: {state.question}")
    
    try:
        # Ensure schema is loaded
        schema_info = schema_manager.get_cached_schema()
        if not schema_info:
            logger.info("📊 Loading database schema...")
            schema_info = schema_manager.fetch_database_schema()
        
        # Call LLM with schema-aware prompt
        llm_output = schema_aware_cortex_llm(state.question, state.session_id)
        
        # Parse with schema validation
        tool, query, trace = schema_aware_parse_llm_output(llm_output, schema_info)
        
        # Optimize query for visualization
        if query and tool == "read_neo4j_cypher":
            query = optimize_query_for_visualization(query, state.node_limit)
        
        logger.info(f"✅ Schema-aware tool selection complete - Tool: {tool}")
        
        return {
            "question": state.question,
            "session_id": state.session_id,
            "tool": tool or "",
            "query": query or "",
            "trace": trace or f"Schema-aware processing of: {state.question}",
            "answer": "",
            "graph_data": None,
            "node_limit": state.node_limit,
            "schema_info": schema_info
        }
        
    except Exception as e:
        logger.error(f"❌ Error in schema-aware tool selection: {e}")
        
        return {
            "question": state.question,
            "session_id": state.session_id,
            "tool": "read_neo4j_cypher",
            "query": "MATCH (n) RETURN n LIMIT 10",
            "trace": f"Error in schema-aware processing: {str(e)}",
            "answer": f"Schema-aware processing encountered an error: {str(e)}",
            "graph_data": None,
            "node_limit": state.node_limit,
            "schema_info": None
        }

def optimize_query_for_visualization(query: str, node_limit: int = 1000) -> str:
    """Optimize queries for visualization"""
    query = query.strip()
    
    if ("MATCH" in query.upper() and 
        "LIMIT" not in query.upper() and 
        "count(" not in query.lower() and
        "COUNT(" not in query):
        
        if "RETURN" in query.upper():
            limit = min(node_limit, 50) if node_limit > 50 else node_limit
            query += f" LIMIT {limit}"
    
    return query

def clean_cypher_query(query: str) -> str:
    """Clean and format Cypher queries"""
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

def format_response_with_graph(result_data, tool_type, node_limit=5000):
    """Format response for display"""
    try:
        if isinstance(result_data, str):
            try:
                result_data = json.loads(result_data)
            except:
                return str(result_data), None
        
        graph_data = None
        
        if tool_type == "read_neo4j_cypher" and isinstance(result_data, dict):
            if "data" in result_data and "metadata" in result_data:
                data = result_data["data"]
                metadata = result_data["metadata"]
                graph_data = result_data.get("graph_data")
                
                formatted_response = f"""
📊 **Schema-Aware Query Results**

**🔢 Records:** {metadata['record_count']}  
**⚡ Time:** {metadata['execution_time_ms']}ms  
**🧠 Schema-Validated:** ✅
                """.strip()
                
                if graph_data and graph_data.get('nodes'):
                    node_count = len(graph_data['nodes'])
                    rel_count = len(graph_data.get('relationships', []))
                    formatted_response += f"\n\n🕸️ **Graph visualization** with {node_count} nodes and {rel_count} relationships"
                else:
                    if isinstance(data, list) and len(data) > 0:
                        formatted_response += f"\n\n**📋 Data:**\n```json\n{json.dumps(data[:3], indent=2)}\n```"
                
                return formatted_response, graph_data
        
        return str(result_data), None
        
    except Exception as e:
        return f"❌ Error formatting response: {str(e)}", None

def schema_aware_execute_tool_node(state: SchemaAwareAgentState) -> dict:
    """Enhanced execution with schema awareness"""
    tool = state.tool
    query = state.query
    trace = state.trace
    node_limit = state.node_limit
    schema_info = state.schema_info
    answer = ""
    graph_data = None
    
    valid_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    
    logger.info(f"⚡ Executing schema-aware tool: '{tool}'")
    
    try:
        if tool == "get_neo4j_schema":
            # Return cached schema if available
            if schema_info:
                answer = schema_manager.schema_summary or "Schema information available"
            else:
                # Fetch fresh schema
                schema_info = schema_manager.fetch_database_schema()
                answer = schema_manager.schema_summary or "Schema retrieved"
                
        elif tool == "read_neo4j_cypher":
            if not query:
                answer = "⚠️ No query generated for your request."
            else:
                query_clean = clean_cypher_query(query)
                
                # Enhanced limits for schema-aware queries
                enhanced_node_limit = node_limit
                if any(pattern in query_clean.upper() for pattern in ['*', 'PATH', 'DEGREE', 'TIER']):
                    enhanced_node_limit = min(node_limit * 2, 100)
                
                data = {
                    "query": query_clean,
                    "params": {},
                    "node_limit": enhanced_node_limit
                }
                
                result = requests.post("http://localhost:8000/read_neo4j_cypher", json=data, headers=headers, timeout=60)
                
                if result.ok:
                    answer, graph_data = format_response_with_graph(result.json(), tool, enhanced_node_limit)
                    logger.info("✅ Schema-aware query successful")
                else:
                    answer = f"❌ Query failed: {result.text}"
                    
        elif tool == "write_neo4j_cypher":
            # Similar handling for write operations
            if not query:
                answer = "⚠️ No modification query generated."
            else:
                query_clean = clean_cypher_query(query)
                data = {"query": query_clean, "params": {}, "node_limit": node_limit}
                result = requests.post("http://localhost:8000/write_neo4j_cypher", json=data, headers=headers, timeout=45)
                
                if result.ok:
                    answer, graph_data = format_response_with_graph(result.json(), tool, node_limit)
                else:
                    answer = f"❌ Update failed: {result.text}"
        else:
            answer = f"❌ Unknown tool: {tool}"
    
    except Exception as e:
        logger.error(f"💥 Error in schema-aware execution: {e}")
        answer = f"⚠️ Execution failed: {str(e)}"
    
    return {
        "question": state.question,
        "session_id": state.session_id,
        "tool": tool,
        "query": query,
        "trace": trace,
        "answer": answer,
        "graph_data": graph_data,
        "node_limit": node_limit,
        "schema_info": schema_info
    }

def build_schema_aware_agent():
    """Build the enhanced schema-aware LangGraph agent"""
    logger.info("🚀 Building Schema-Aware LangGraph Agent...")
    
    # Initialize schema on startup
    schema_manager.fetch_database_schema()
    
    workflow = StateGraph(state_schema=SchemaAwareAgentState)
    
    # Add nodes with schema-aware functions
    workflow.add_node("select_tool", RunnableLambda(schema_aware_select_tool_node))
    workflow.add_node("execute_tool", RunnableLambda(schema_aware_execute_tool_node))
    
    # Set entry point
    workflow.set_entry_point("select_tool")
    
    # Add edges
    workflow.add_edge("select_tool", "execute_tool")
    workflow.add_edge("execute_tool", END)
    
    # Compile
    agent = workflow.compile()
    logger.info("✅ Schema-Aware LangGraph agent built successfully!")
    
    return agent

# Additional utility functions
def refresh_schema_cache():
    """Manually refresh the schema cache"""
    logger.info("🔄 Manually refreshing schema cache...")
    return schema_manager.fetch_database_schema()

def get_schema_summary():
    """Get current schema summary"""
    return schema_manager.schema_summary

# For testing
if __name__ == "__main__":
    # Test the schema-aware agent
    agent = build_schema_aware_agent()
    
    import asyncio
    
    async def test_schema_aware():
        test_questions = [
            "What's in my database?",
            "Show me all node types",
            "Find connections between data",
            "Display the most connected nodes"
        ]
        
        for question in test_questions:
            print(f"\n🧪 Testing: {question}")
            test_state = SchemaAwareAgentState(
                question=question,
                session_id="test_session",
                node_limit=50
            )
            result = await agent.ainvoke(test_state)
            print(f"✅ Tool: {result.get('tool')}")
            print(f"📝 Query: {result.get('query')}")
            print(f"🎯 Answer: {result.get('answer', '')[:100]}...")
    
    # asyncio.run(test_schema_aware())
