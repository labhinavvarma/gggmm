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

# ============================================================================
# STATE MODELS - Both Original and Schema-Aware
# ============================================================================

class AgentState(BaseModel):
    """Original agent state for backward compatibility"""
    question: str
    session_id: str
    tool: str = ""
    query: str = ""
    trace: str = ""
    answer: str = ""
    graph_data: Optional[dict] = None
    node_limit: int = 1000

class SchemaAwareAgentState(BaseModel):
    """Enhanced agent state with schema awareness"""
    question: str
    session_id: str
    tool: str = ""
    query: str = ""
    trace: str = ""
    answer: str = ""
    graph_data: Optional[dict] = None
    node_limit: int = 1000
    schema_info: Optional[dict] = None

# ============================================================================
# SCHEMA MANAGER
# ============================================================================

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
            
            # Get additional schema information
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

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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

def optimize_query_for_visualization(query: str, node_limit: int = 1000) -> str:
    """Optimize queries for better visualization performance"""
    query = query.strip()
    
    if ("MATCH" in query.upper() and 
        "LIMIT" not in query.upper() and 
        "count(" not in query.lower() and
        "COUNT(" not in query):
        
        if "RETURN" in query.upper():
            limit = min(node_limit, 50) if node_limit > 50 else node_limit
            query += f" LIMIT {limit}"
    
    return query

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

def format_response_with_graph(result_data, tool_type, node_limit=5000):
    """Format the response for split-screen display"""
    try:
        if isinstance(result_data, str):
            try:
                result_data = json.loads(result_data)
            except:
                return str(result_data), None
        
        graph_data = None
        
        if tool_type == "write_neo4j_cypher" and isinstance(result_data, dict):
            if "change_info" in result_data:
                change_info = result_data["change_info"]
                formatted_response = f"""
🔄 **Database Update Completed**

**⚡ Execution:** {change_info['execution_time_ms']}ms  
**🕐 Time:** {change_info['timestamp'][:19]}

**📝 Changes Made:**
{chr(10).join(f"{change}" for change in change_info['changes'])}

**🔧 Query:** `{change_info['query']}`
                """.strip()
                
                if result_data.get("graph_data"):
                    graph_data = result_data["graph_data"]
                    node_count = len(graph_data.get('nodes', []))
                    rel_count = len(graph_data.get('relationships', []))
                    if node_count > 0 or rel_count > 0:
                        formatted_response += f"\n\n🕸️ **Updated graph visualization** with {node_count} nodes and {rel_count} relationships"
                
                return formatted_response, graph_data
        
        elif tool_type == "read_neo4j_cypher" and isinstance(result_data, dict):
            if "data" in result_data and "metadata" in result_data:
                data = result_data["data"]
                metadata = result_data["metadata"]
                graph_data = result_data.get("graph_data")
                
                formatted_response = f"""
📊 **Query Results**

**🔢 Records:** {metadata['record_count']}  
**⚡ Time:** {metadata['execution_time_ms']}ms  
**🕐 Timestamp:** {metadata['timestamp'][:19]}
                """.strip()
                
                if not graph_data or not graph_data.get('nodes'):
                    if isinstance(data, list) and len(data) > 0:
                        if len(data) <= 3:
                            formatted_response += f"\n\n**📋 Data:**\n```json\n{json.dumps(data, indent=2)}\n```"
                        else:
                            formatted_response += f"\n\n**📋 Sample Data:**\n```json\n{json.dumps(data[:2], indent=2)}\n... and {len(data) - 2} more records\n```"
                    else:
                        formatted_response += "\n\n**📋 Data:** No records found"
                
                if graph_data and graph_data.get('nodes'):
                    node_count = len(graph_data['nodes'])
                    rel_count = len(graph_data.get('relationships', []))
                    
                    formatted_response += f"\n\n🕸️ **Graph visualization updated** with {node_count} nodes and {rel_count} relationships"
                    
                    if node_count > 0:
                        label_counts = {}
                        for node in graph_data['nodes']:
                            for label in node.get('labels', ['Unknown']):
                                label_counts[label] = label_counts.get(label, 0) + 1
                        
                        if len(label_counts) > 0:
                            label_summary = ", ".join([f"{label}({count})" for label, count in sorted(label_counts.items())])
                            formatted_response += f"\n**🏷️ Node Types:** {label_summary}"
                    
                    if graph_data.get('limited'):
                        formatted_response += f"\n**⚠️ Display limited to {node_limit} nodes for performance**"
                
                return formatted_response, graph_data
        
        elif tool_type == "get_neo4j_schema" and isinstance(result_data, dict):
            if "schema" in result_data:
                schema = result_data["schema"]
                metadata = result_data.get("metadata", {})
                
                schema_summary = []
                if isinstance(schema, dict):
                    for label, info in schema.items():
                        if isinstance(info, dict):
                            props = info.get('properties', {})
                            relationships = info.get('relationships', {})
                            schema_summary.append(f"**{label}**: {len(props)} props, {len(relationships)} rels")
                
                formatted_response = f"""
🏗️ **Database Schema**

**⚡ Time:** {metadata.get('execution_time_ms', 'N/A')}ms

**📊 Overview:**
{chr(10).join(f"{item}" for item in schema_summary[:10])}
{f"... and {len(schema_summary) - 10} more types" if len(schema_summary) > 10 else ""}
                """.strip()
                
                return formatted_response, None
        
        formatted_text = json.dumps(result_data, indent=2) if isinstance(result_data, (dict, list)) else str(result_data)
        return formatted_text, None
    
    except Exception as e:
        error_msg = f"❌ **Error formatting response:** {str(e)}"
        logger.error(error_msg)
        return error_msg, None

# ============================================================================
# SYSTEM MESSAGES
# ============================================================================

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

User: "Show me connections between different types"
Tool: read_neo4j_cypher
Query: MATCH (a)-[r]->(b) WHERE labels(a) <> labels(b) RETURN a, r, b LIMIT 25

User: "Find nodes with most connections"
Tool: read_neo4j_cypher
Query: MATCH (n)-[r]-() RETURN n, count(r) as connections ORDER BY connections DESC LIMIT 20

**MULTI-TIER QUERIES (using actual schema):**

User: "Find 2nd degree connections"
Tool: read_neo4j_cypher
Query: MATCH (a)-[r1]->(b)-[r2]->(c) WHERE a <> c RETURN a, r1, b, r2, c LIMIT 40

User: "Show network paths"
Tool: read_neo4j_cypher
Query: MATCH path = (a)-[*1..3]-(b) WHERE a <> b RETURN path LIMIT 30

User: "Friends of friends"
Tool: read_neo4j_cypher
Query: MATCH (p)-[*2]-(fof) WHERE p <> fof RETURN p, fof LIMIT 35

User: "Extended network of specific person"
Tool: read_neo4j_cypher
Query: MATCH path = (start {{name: "John"}})-[*1..3]-(end) WHERE start <> end RETURN path LIMIT 30

**VALIDATION RULES:**
- If user asks for non-existent labels/relationships, suggest available alternatives
- Always use schema-validated queries
- Provide helpful error messages when schema doesn't match request

**CURRENT SCHEMA SUMMARY:**
{schema_summary}

IMPORTANT: Generate queries ONLY using the actual schema elements listed above. If a user asks for something not in the schema, explain what's available instead.
"""
    
    return system_message

# Original system message for backward compatibility
ORIGINAL_SYS_MSG = """You are a Neo4j database expert assistant. For each user question, you must select ONE tool and provide a Cypher query (if needed).

REQUIRED OUTPUT FORMAT:
Tool: [tool_name]
Query: [cypher_query_if_needed]

AVAILABLE TOOLS:
1. read_neo4j_cypher - for reading data (MATCH, RETURN, WHERE, COUNT, etc.)
2. write_neo4j_cypher - for modifying data (CREATE, MERGE, SET, DELETE, etc.) 
3. get_neo4j_schema - for schema information (no query needed)

EXAMPLES:

User: Show me all Person nodes
Tool: read_neo4j_cypher
Query: MATCH (n:Person) RETURN n LIMIT 25

User: How many nodes are there?
Tool: read_neo4j_cypher  
Query: MATCH (n) RETURN count(n) AS node_count

User: Create a person named John
Tool: write_neo4j_cypher
Query: CREATE (n:Person {name: "John"}) RETURN n

User: What is the database schema?
Tool: get_neo4j_schema

Always respond with Tool: and Query: on separate lines."""

# ============================================================================
# LLM INTERACTION
# ============================================================================

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
            "sys_msg": schema_aware_sys_msg,
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

def cortex_llm(prompt: str, session_id: str) -> str:
    """Original Cortex LLM call for backward compatibility"""
    
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
            "sys_msg": ORIGINAL_SYS_MSG,
            "limit_convs": "0",
            "prompt": {
                "messages": [{"role": "user", "content": prompt}]
            },
            "session_id": session_id
        }
    }
    
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, verify=False, timeout=30)
        resp.raise_for_status()
        return resp.text.partition("end_of_stream")[0].strip()
    except Exception as e:
        logger.error(f"Cortex LLM API error: {e}")
        return f"Tool: read_neo4j_cypher\nQuery: MATCH (n) RETURN n LIMIT 20"

# ============================================================================
# PARSING FUNCTIONS
# ============================================================================

def parse_llm_output(llm_output):
    """Parse LLM output to extract tool and query (original)"""
    allowed_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    trace = llm_output.strip()
    tool = None
    query = None
    
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
    
    # Fallback
    if not tool:
        tool = "read_neo4j_cypher"
        query = "MATCH (n) RETURN n LIMIT 20"
    
    return tool, query, trace

def schema_aware_parse_llm_output(llm_output, schema_info):
    """Enhanced parsing with schema validation"""
    allowed_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    trace = llm_output.strip()
    tool = None
    query = None
    
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
    
    # Fallback
    if not tool or not query:
        tool = "read_neo4j_cypher"
        query = "MATCH (n) RETURN labels(n) as Types, count(*) as Count ORDER BY Count DESC LIMIT 10"
    
    return tool, query, trace

# ============================================================================
# NODE FUNCTIONS
# ============================================================================

def select_tool_node(state: AgentState) -> dict:
    """Original tool selection for backward compatibility"""
    logger.info(f"🤔 Processing question: {state.question}")
    
    try:
        llm_output = cortex_llm(state.question, state.session_id)
        tool, query, trace = parse_llm_output(llm_output)
        
        if query and tool == "read_neo4j_cypher":
            query = optimize_query_for_visualization(query, state.node_limit)
        
        return {
            "question": state.question,
            "session_id": state.session_id,
            "tool": tool or "",
            "query": query or "",
            "trace": trace or "",
            "answer": "",
            "graph_data": None,
            "node_limit": state.node_limit
        }
    except Exception as e:
        logger.error(f"Error in select_tool_node: {e}")
        return {
            "question": state.question,
            "session_id": state.session_id,
            "tool": "read_neo4j_cypher",
            "query": "MATCH (n) RETURN n LIMIT 10",
            "trace": f"Error: {str(e)}",
            "answer": f"❌ Error processing question: {str(e)}",
            "graph_data": None,
            "node_limit": state.node_limit
        }

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

def execute_tool_node(state: AgentState) -> dict:
    """Original tool execution for backward compatibility"""
    tool = state.tool
    query = state.query
    trace = state.trace
    node_limit = state.node_limit
    answer = ""
    graph_data = None
    
    valid_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    
    logger.info(f"⚡ Executing tool: '{tool}'")
    
    try:
        if tool == "get_neo4j_schema":
            result = requests.post("http://localhost:8000/get_neo4j_schema", headers=headers, timeout=30)
            if result.ok:
                answer, graph_data = format_response_with_graph(result.json(), tool, node_limit)
            else:
                answer = f"❌ Schema query failed: {result.text}"
                
        elif tool == "read_neo4j_cypher":
            if not query:
                answer = "⚠️ No query generated for your request."
            else:
                query_clean = clean_cypher_query(query)
                data = {"query": query_clean, "params": {}, "node_limit": node_limit}
                result = requests.post("http://localhost:8000/read_neo4j_cypher", json=data, headers=headers, timeout=45)
                
                if result.ok:
                    answer, graph_data = format_response_with_graph(result.json(), tool, node_limit)
                else:
                    answer = f"❌ Query failed: {result.text}"
                    
        elif tool == "write_neo4j_cypher":
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
        logger.error(f"Error in execute_tool_node: {e}")
        answer = f"⚠️ Execution failed: {str(e)}"
    
    return {
        "question": state.question,
        "session_id": state.session_id,
        "tool": tool,
        "query": query,
        "trace": trace,
        "answer": answer,
        "graph_data": graph_data,
        "node_limit": node_limit
    }

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
        logger.error(f"Error in schema-aware execution: {e}")
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

# ============================================================================
# AGENT BUILDERS
# ============================================================================

def build_agent():
    """Build the original LangGraph agent (backward compatibility)"""
    workflow = StateGraph(state_schema=AgentState)
    
    # Add nodes
    workflow.add_node("select_tool", RunnableLambda(select_tool_node))
    workflow.add_node("execute_tool", RunnableLambda(execute_tool_node))
    
    # Set entry point
    workflow.set_entry_point("select_tool")
    
    # Add edges
    workflow.add_edge("select_tool", "execute_tool")
    workflow.add_edge("execute_tool", END)
    
    # Compile and return
    agent = workflow.compile()
    logger.info("🚀 Original LangGraph agent built successfully")
    return agent

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

# ============================================================================
# UTILITY FUNCTIONS FOR EXTERNAL USE
# ============================================================================

def refresh_schema_cache():
    """Manually refresh the schema cache"""
    logger.info("🔄 Manually refreshing schema cache...")
    return schema_manager.fetch_database_schema()

def get_schema_summary():
    """Get current schema summary"""
    return schema_manager.schema_summary

# For testing
if __name__ == "__main__":
    # Test both agents
    print("Testing both agent types...")
    
    # Test original agent
    original_agent = build_agent()
    print("✅ Original agent built successfully")
    
    # Test schema-aware agent
    schema_agent = build_schema_aware_agent()
    print("✅ Schema-aware agent built successfully")
    
    print("🎉 All agents working!")
