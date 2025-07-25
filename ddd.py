import requests
import urllib3
from pydantic import BaseModel
from typing import Optional
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import re
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced_langgraph_agent")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class AgentState(BaseModel):
    question: str
    session_id: str
    tool: str = ""
    query: str = ""
    trace: str = ""
    answer: str = ""
    graph_data: Optional[dict] = None
    node_limit: int = 1000

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

def optimize_query_for_visualization(query: str, node_limit: int = 1000, unlimited_mode: bool = False) -> str:
    """Enhanced query optimization for visualization with unlimited display support"""
    query = query.strip()
    
    if unlimited_mode:
        # Remove artificial limits for unlimited display
        query = re.sub(r'\s+LIMIT\s+\d+', '', query, flags=re.IGNORECASE)
        
        # Enhance queries for better unlimited visualization
        if "MATCH (n)" in query.upper() and "RETURN n" in query.upper():
            # Add relationship context for better visualization
            if "OPTIONAL MATCH" not in query.upper():
                query = query.replace("RETURN n", "OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m")
        
        logger.info(f"🚀 Optimized query for unlimited display: {query[:100]}...")
        
    else:
        # Apply reasonable limits for standard display
        if ("MATCH" in query.upper() and 
            "LIMIT" not in query.upper() and 
            "count(" not in query.lower() and
            "COUNT(" not in query):
            
            if "RETURN" in query.upper():
                # Use smaller limits for better visualization
                limit = min(node_limit, 50) if node_limit > 50 else node_limit
                query += f" LIMIT {limit}"
    
    return query

def smart_question_preprocessor(question: str) -> str:
    """Preprocess questions to make them more specific and actionable"""
    
    # Common question mappings for vague or general questions
    question_mappings = {
        # Very general questions
        "what's in the database": "Show me a sample of all data in the database",
        "what do you have": "Display different types of nodes and their counts", 
        "show me something": "Show me sample data from the database",
        "what can i see": "Show me the different types of data available",
        "explore": "Show me a sample of nodes and relationships",
        "anything": "Display sample data from the database",
        "what's there": "Show me what data exists in the database",
        "database content": "Display the contents of the database",
        "what's here": "Show me what data is available in the database",
        "tell me about this database": "Show me a summary of the database structure and content",
        
        # Count and size questions
        "how much data": "How many total nodes are in the database",
        "size": "Count all nodes and relationships in the database",
        "data size": "Count all nodes in the database",
        "how big": "Show me the size of the database",
        "database size": "Count all nodes and relationships",
        
        # Type and structure questions  
        "what kinds": "Show me the different types of nodes available",
        "what sorts": "Display the database schema and available node types",
        "categories": "Show me the categories of data in the database",
        "what types": "Show me the different node types in the database",
        "structure": "Show me the database schema and structure",
        "schema": "Display the database schema",
        
        # Network and relationship questions
        "connections": "Show me how nodes are connected in the database",
        "network": "Display the network structure of the database",
        "relationships": "Show me the relationships between nodes",
        "graph": "Display the graph structure",
        
        # Exploration questions
        "overview": "Give me an overview of the database content",
        "summary": "Show me a summary of what's in the database",
        "tour": "Give me a tour of the database",
        "walkthrough": "Show me different parts of the database"
    }
    
    # Normalize the question
    normalized = question.lower().strip()
    
    # Check for direct mappings
    for pattern, replacement in question_mappings.items():
        if pattern in normalized:
            logger.info(f"🔄 Preprocessed question: '{question}' → '{replacement}'")
            return replacement
    
    # Expand abbreviated questions
    if len(normalized) < 10:
        if any(word in normalized for word in ['show', 'see', 'get', 'tell', 'what']):
            expanded = f"Show me sample data from the database: {question}"
            logger.info(f"🔄 Expanded short question: '{question}' → '{expanded}'")
            return expanded
    
    # Handle single word questions
    single_word_mappings = {
        "nodes": "Show me different types of nodes in the database",
        "data": "Show me sample data from the database",
        "graph": "Display the graph structure",
        "schema": "Show me the database schema",
        "structure": "Display the database structure",
        "overview": "Give me an overview of the database",
        "summary": "Show me a summary of the database content"
    }
    
    if normalized in single_word_mappings:
        expanded = single_word_mappings[normalized]
        logger.info(f"🔄 Expanded single word: '{question}' → '{expanded}'")
        return expanded
    
    return question

def generate_fallback_response(original_question: str) -> str:
    """Generate intelligent fallback response when LLM fails"""
    
    question_lower = original_question.lower()
    
    # Schema questions
    if any(word in question_lower for word in ['schema', 'structure', 'types', 'labels', 'what kinds', 'what types', 'properties']):
        return "Tool: get_neo4j_schema"
    
    # Write operations  
    elif any(word in question_lower for word in ['create', 'add', 'insert', 'new', 'make', 'build']):
        return "Tool: write_neo4j_cypher\nQuery: // Please specify what you want to create"
    
    # Specific entity types with common variations
    elif any(word in question_lower for word in ['person', 'people', 'user', 'users', 'individual', 'human']):
        return "Tool: read_neo4j_cypher\nQuery: MATCH (n:Person) RETURN n LIMIT 25"
    
    elif any(word in question_lower for word in ['company', 'organization', 'companies', 'business', 'corporation']):
        return "Tool: read_neo4j_cypher\nQuery: MATCH (n:Company) RETURN n LIMIT 25"
    
    elif any(word in question_lower for word in ['movie', 'film', 'movies', 'films', 'cinema']):
        return "Tool: read_neo4j_cypher\nQuery: MATCH (n:Movie) RETURN n LIMIT 25"
    
    elif any(word in question_lower for word in ['product', 'products', 'item', 'items']):
        return "Tool: read_neo4j_cypher\nQuery: MATCH (n:Product) RETURN n LIMIT 25"
    
    # Count questions
    elif any(word in question_lower for word in ['count', 'how many', 'number', 'total', 'size', 'amount']):
        return "Tool: read_neo4j_cypher\nQuery: MATCH (n) RETURN count(n) as TotalNodes"
    
    # Relationship questions
    elif any(word in question_lower for word in ['relationship', 'connection', 'link', 'connected', 'relationships', 'network', 'connections']):
        return "Tool: read_neo4j_cypher\nQuery: MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 20"
    
    # General exploration
    elif any(word in question_lower for word in ['explore', 'show', 'display', 'see', 'view', 'what', 'database', 'data', 'tell', 'overview']):
        return "Tool: read_neo4j_cypher\nQuery: MATCH (n) RETURN n LIMIT 20"
    
    # Default: show sample data
    else:
        return "Tool: read_neo4j_cypher\nQuery: MATCH (n) RETURN n LIMIT 20"

# Enhanced system message with comprehensive examples and better general question handling
SYS_MSG = """You are a Neo4j database expert assistant with enhanced question understanding capabilities. Your job is to help users explore and interact with their graph database.

For ANY user question, you MUST respond with a tool selection and appropriate query. Here are your tools:

**TOOLS AVAILABLE:**
1. **read_neo4j_cypher** - For viewing, exploring, counting, finding data
   - Use for: "show me", "find", "how many", "what are", "display", "list", "get", "explore", "tell me about"
   - ALWAYS generates MATCH queries with RETURN statements
   - Include relationships when users want to see connections

2. **write_neo4j_cypher** - For creating, updating, deleting data  
   - Use for: "create", "add", "update", "delete", "remove", "insert", "make", "build"
   - Generates CREATE, MERGE, SET, DELETE queries
   - Include RETURN clauses to show created/modified data

3. **get_neo4j_schema** - For database structure questions
   - Use for: "schema", "structure", "what types", "what labels", "what properties"
   - NO query needed - just returns database structure

**RESPONSE FORMAT (REQUIRED):**
Tool: [tool_name]
Query: [cypher_query_or_none_for_schema]

**COMPREHENSIVE EXAMPLES FOR COMMON QUESTIONS:**

User: "What's in my database?"
Tool: read_neo4j_cypher  
Query: MATCH (n) RETURN labels(n) as NodeType, count(*) as Count ORDER BY Count DESC LIMIT 10

User: "Show me some data"
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN n LIMIT 20

User: "Explore the database"
Tool: read_neo4j_cypher
Query: MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 25

User: "How many nodes do I have?"
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN count(n) as TotalNodes

User: "What types of nodes exist?"
Tool: get_neo4j_schema

User: "Show me all people"
Tool: read_neo4j_cypher
Query: MATCH (n:Person) RETURN n LIMIT 30

User: "Create a person named John"
Tool: write_neo4j_cypher
Query: CREATE (n:Person {name: "John"}) RETURN n

User: "What can I explore?"
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN DISTINCT labels(n) as AvailableNodeTypes LIMIT 20

User: "Show me connections"
Tool: read_neo4j_cypher
Query: MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 25

User: "Display the network"
Tool: read_neo4j_cypher
Query: MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 30

User: "Tell me about this database"
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN labels(n) as NodeTypes, count(*) as Count ORDER BY Count DESC

User: "Show me something"
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN n LIMIT 20

User: "Overview"
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN DISTINCT labels(n) as NodeTypes

User: "Database structure"
Tool: get_neo4j_schema

**IMPORTANT RULES:**
- EVERY question gets a tool + query response
- For vague questions, use read_neo4j_cypher with broad MATCH queries
- ALWAYS include RETURN clause in read queries
- Add LIMIT to prevent large results unless user wants everything
- For exploration, include relationships when possible: OPTIONAL MATCH (n)-[r]-(m)
- If unsure, default to showing sample data with read_neo4j_cypher
- Make queries inclusive and broad for general questions

**SPECIAL HANDLING:**
- "Show/Display/See" + general terms = show sample data with relationships
- "Count/How many/Total" = use count() functions
- "Types/Kinds/Categories" = either schema tool OR distinct labels query
- "Create/Add/Make" + specifics = write operations with RETURN
- Single word questions = interpret as exploration requests

RESPOND NOW with Tool: and Query: for the user's question."""

# Cortex LLM configuration
API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
MODEL = "llama3.1-70b"

def enhanced_cortex_llm(prompt: str, session_id: str) -> str:
    """Enhanced Cortex LLM call with preprocessing and comprehensive error handling"""
    
    # Preprocess the question for better understanding
    processed_prompt = smart_question_preprocessor(prompt)
    
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
                "messages": [{"role": "user", "content": processed_prompt}]
            },
            "session_id": session_id
        }
    }
    
    try:
        logger.info(f"🔄 Calling Enhanced Cortex LLM with processed prompt: {processed_prompt[:100]}...")
        resp = requests.post(API_URL, headers=headers, json=payload, verify=False, timeout=30)
        resp.raise_for_status()
        
        raw_response = resp.text
        logger.info(f"📥 Raw Cortex response length: {len(raw_response)}")
        
        # Enhanced response parsing
        if "end_of_stream" in raw_response:
            parsed_response = raw_response.partition("end_of_stream")[0].strip()
        else:
            parsed_response = raw_response.strip()
        
        # If response seems incomplete, try to fix it
        if len(parsed_response) < 20:
            logger.warning(f"⚠️ Short response detected, using fallback")
            return generate_fallback_response(prompt)
        
        logger.info(f"✅ Enhanced LLM response processed successfully")
        return parsed_response
        
    except requests.exceptions.Timeout:
        logger.error(f"⏰ Cortex LLM timeout, using intelligent fallback")
        return generate_fallback_response(prompt)
    except requests.exceptions.ConnectionError:
        logger.error(f"🔌 Cortex LLM connection error, using intelligent fallback")
        return generate_fallback_response(prompt)
    except Exception as e:
        logger.error(f"❌ Enhanced Cortex LLM API error: {e}")
        
        # Intelligent fallback based on original question
        fallback_response = generate_fallback_response(prompt)
        logger.info(f"🔄 Using intelligent fallback response")
        return fallback_response

def enhanced_parse_llm_output(llm_output):
    """Enhanced parsing with comprehensive fallback handling and multiple extraction patterns"""
    allowed_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    trace = llm_output.strip()
    tool = None
    query = None
    
    logger.info(f"🔍 Enhanced parsing of LLM output (length: {len(llm_output)})")
    logger.info(f"🔍 LLM output preview: {llm_output[:500]}...")
    
    # Try multiple tool extraction patterns with increased flexibility
    tool_patterns = [
        r"Tool:\s*([\w_]+)",                    # Standard
        r"**Tool:**\s*([\w_]+)",                # Bold
        r"Tool\s*[:=]\s*([\w_]+)",              # Flexible separator
        r"Selected tool:\s*([\w_]+)",           # Alternative phrasing
        r"Using:\s*([\w_]+)",                   # Informal
        r"I'll use\s*:\s*([\w_]+)",            # Natural language
        r"(?:Tool|TOOL)\s*[:=]\s*([\w_]+)",    # Case insensitive
        r"The tool is:\s*([\w_]+)",             # Descriptive
        r"Choose tool:\s*([\w_]+)",             # Command form
        r"Tool selection:\s*([\w_]+)"           # Formal selection
    ]
    
    for pattern in tool_patterns:
        tool_match = re.search(pattern, llm_output, re.I)
        if tool_match:
            tname = tool_match.group(1).strip()
            logger.info(f"🎯 Found tool candidate: '{tname}' using pattern: {pattern}")
            if tname in allowed_tools:
                tool = tname
                logger.info(f"✅ Valid tool found: {tool}")
                break
    
    # Try multiple query extraction patterns with enhanced flexibility
    query_patterns = [
        r"Query:\s*(.+?)(?:\n\n|\nTool|\nNote|\n$|$)",     # More flexible ending
        r"**Query:**\s*(.+?)(?:\n\n|\nTool|\nNote|\n$|$)", # Bold format
        r"Query\s*[:=]\s*(.+?)(?:\n\n|\nTool|\nNote|\n$|$)", # Flexible separator
        r"Cypher:\s*(.+?)(?:\n\n|\nTool|\nNote|\n$|$)",    # Alternative
        r"```cypher\s*(.+?)\s*```",                        # Code block
        r"```\s*(.+?)\s*```",                              # Generic code block
        r"The query is:\s*(.+?)(?:\n\n|\nTool|\nNote|\n$|$)", # Descriptive
        r"Query to execute:\s*(.+?)(?:\n\n|\nTool|\nNote|\n$|$)", # Formal
        r"MATCH\s+.+?(?=\n[A-Z]|\n\n|$)",                 # Direct MATCH detection
        r"CREATE\s+.+?(?=\n[A-Z]|\n\n|$)",                # Direct CREATE detection
        r"MERGE\s+.+?(?=\n[A-Z]|\n\n|$)",                 # Direct MERGE detection
        r"CALL\s+.+?(?=\n[A-Z]|\n\n|$)"                   # Direct CALL detection
    ]
    
    for pattern in query_patterns:
        query_match = re.search(pattern, llm_output, re.I | re.DOTALL)
        if query_match:
            query = query_match.group(1).strip() if query_match.groups() else query_match.group(0).strip()
            logger.info(f"🎯 Found query using pattern: {pattern}")
            logger.info(f"🎯 Query preview: {query[:100]}...")
            if query and len(query) > 3:
                # Clean up the query
                query = re.sub(r'\s+', ' ', query.strip())
                # Remove common trailing phrases
                query = re.sub(r'\s*(Note:|Explanation:|This query)', '', query, flags=re.I)
                logger.info(f"✅ Valid query found and cleaned")
                break
    
    # Enhanced fallback logic for common question types
    if not tool or not query:
        logger.warning("⚠️ Primary parsing failed, attempting enhanced intelligent fallback...")
        
        lower_output = llm_output.lower()
        
        # Analyze question intent with expanded keywords
        question_indicators = {
            'read': ['show', 'display', 'find', 'get', 'what', 'how many', 'count', 'list', 'see', 'view', 'explore', 'tell me', 'describe', 'look', 'check', 'search', 'browse'],
            'write': ['create', 'add', 'insert', 'make', 'new', 'update', 'set', 'change', 'delete', 'remove', 'build', 'establish'],
            'schema': ['schema', 'structure', 'types', 'labels', 'properties', 'what kind', 'what sorts', 'metadata', 'format', 'design']
        }
        
        # Score each category with weighted scoring
        scores = {'read': 0, 'write': 0, 'schema': 0}
        for category, keywords in question_indicators.items():
            for keyword in keywords:
                if keyword in lower_output:
                    # Weight more specific keywords higher
                    weight = 2 if keyword in ['schema', 'create', 'delete'] else 1
                    scores[category] += weight
        
        # Determine tool based on highest score
        if not tool:
            if scores['schema'] > 0:
                tool = "get_neo4j_schema"
                logger.info(f"🔄 Fallback: Inferred schema tool (score: {scores['schema']})")
            elif scores['write'] > 0:
                tool = "write_neo4j_cypher" 
                logger.info(f"🔄 Fallback: Inferred write tool (score: {scores['write']})")
            else:
                tool = "read_neo4j_cypher"  # Default to read
                logger.info(f"🔄 Fallback: Defaulting to read tool (scores: {scores})")
        
        # Generate appropriate fallback query with enhanced intelligence
        if not query and tool != "get_neo4j_schema":
            if tool == "read_neo4j_cypher":
                # Generate smart read queries based on question content
                if any(word in lower_output for word in ['count', 'how many', 'number', 'total', 'size']):
                    query = "MATCH (n) RETURN count(n) as TotalNodes"
                elif any(word in lower_output for word in ['person', 'people', 'user', 'human', 'individual']):
                    query = "MATCH (n:Person) RETURN n LIMIT 25"
                elif any(word in lower_output for word in ['company', 'organization', 'business', 'corporation']):
                    query = "MATCH (n:Company) RETURN n LIMIT 25"
                elif any(word in lower_output for word in ['movie', 'film', 'cinema']):
                    query = "MATCH (n:Movie) RETURN n LIMIT 25"
                elif any(word in lower_output for word in ['product', 'item', 'goods']):
                    query = "MATCH (n:Product) RETURN n LIMIT 25"
                elif any(word in lower_output for word in ['relationship', 'connection', 'link', 'network', 'connected']):
                    query = "MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 20"
                elif any(word in lower_output for word in ['all', 'everything', 'complete', 'entire']):
                    query = "MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 30"
                elif any(word in lower_output for word in ['type', 'kind', 'category', 'sort']):
                    query = "MATCH (n) RETURN DISTINCT labels(n) as NodeTypes"
                elif any(word in lower_output for word in ['explore', 'overview', 'summary', 'tour']):
                    query = "MATCH (n) RETURN n LIMIT 25"
                else:
                    # Very general fallback with relationships
                    query = "MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 20"
                
                logger.info(f"🔄 Generated enhanced fallback read query: {query}")
                
            elif tool == "write_neo4j_cypher":
                # For write operations, we need more specific info
                query = "// Unable to generate specific write query from the request - please provide more details"
                logger.info(f"🔄 Write operation needs more specific instructions")
    
    # Final validation and cleanup with enhanced rules
    if query and tool != "get_neo4j_schema":
        # Ensure query has proper structure
        query = query.strip().rstrip(';')  # Remove trailing semicolon
        
        # Fix incomplete read queries
        if tool == "read_neo4j_cypher" and "RETURN" not in query.upper():
            if query.upper().startswith("MATCH"):
                query += " RETURN n LIMIT 25"
            else:
                query = f"MATCH (n) WHERE {query} RETURN n LIMIT 25"
        
        # Add reasonable LIMIT if missing for read queries (but not for counts)
        if (tool == "read_neo4j_cypher" and 
            "LIMIT" not in query.upper() and 
            "count(" not in query.lower() and
            "COUNT(" not in query and
            "DISTINCT" not in query.upper()):
            query += " LIMIT 25"
        
        # Clean up any remaining formatting issues
        query = re.sub(r'\s+', ' ', query.strip())
    
    logger.info(f"🎯 Final parsing results - Tool: {tool}, Query: {query[:100] if query else 'None'}...")
    
    return tool, query, trace

def format_response_with_graph(result_data, tool_type, node_limit=5000):
    """Enhanced response formatting for split-screen display with better error handling"""
    try:
        if isinstance(result_data, str):
            try:
                result_data = json.loads(result_data)
            except json.JSONDecodeError:
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
{chr(10).join(f"  • {change}" for change in change_info['changes'])}

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
{chr(10).join(f"  • {item}" for item in schema_summary[:10])}
{f"  • ... and {len(schema_summary) - 10} more types" if len(schema_summary) > 10 else ""}
                """.strip()
                
                return formatted_response, None
        
        # Fallback for other formats
        formatted_text = json.dumps(result_data, indent=2) if isinstance(result_data, (dict, list)) else str(result_data)
        return formatted_text, None
    
    except Exception as e:
        error_msg = f"❌ **Error formatting response:** {str(e)}"
        logger.error(error_msg)
        return error_msg, None

def enhanced_select_tool_node(state: AgentState) -> dict:
    """Enhanced tool selection with comprehensive question handling and error recovery"""
    logger.info(f"🤔 Processing question: {state.question}")
    
    try:
        # Call LLM with enhanced prompt preprocessing
        llm_output = enhanced_cortex_llm(state.question, state.session_id)
        logger.info(f"📥 LLM Response received (length: {len(llm_output)})")
        
        # Use enhanced parsing with multiple fallback levels
        tool, query, trace = enhanced_parse_llm_output(llm_output)
        
        # Ultimate fallback if still no tool/query
        if not tool:
            logger.warning("🚨 Ultimate fallback: No tool detected, defaulting to exploration")
            tool = "read_neo4j_cypher"
            query = "MATCH (n) RETURN n LIMIT 20"
            trace = f"Ultimate fallback response for: {state.question}"
        
        # Optimize query for visualization if needed
        if query and tool == "read_neo4j_cypher":
            unlimited_mode = state.node_limit is None or state.node_limit >= 50000
            query = optimize_query_for_visualization(query, state.node_limit, unlimited_mode)
        
        logger.info(f"✅ Enhanced tool selection complete - Tool: {tool}, Query: {query[:100] if query else 'None'}")
        
        return {
            "question": state.question,
            "session_id": state.session_id,
            "tool": tool or "",
            "query": query or "",
            "trace": trace or f"Enhanced processing of: {state.question}",
            "answer": "",
            "graph_data": None,
            "node_limit": state.node_limit
        }
        
    except Exception as e:
        logger.error(f"❌ Error in enhanced_select_tool_node: {e}")
        
        # Emergency fallback for any error
        return {
            "question": state.question,
            "session_id": state.session_id,
            "tool": "read_neo4j_cypher",
            "query": "MATCH (n) RETURN n LIMIT 10",
            "trace": f"Emergency fallback due to error: {str(e)}",
            "answer": f"I'll show you some data from your database. Error details: {str(e)}",
            "graph_data": None,
            "node_limit": state.node_limit
        }

def execute_tool_node(state: AgentState) -> dict:
    """Enhanced tool execution with comprehensive error handling and logging"""
    tool = state.tool
    query = state.query
    trace = state.trace
    node_limit = state.node_limit
    answer = ""
    graph_data = None
    
    valid_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    
    logger.info(f"⚡ Executing tool: '{tool}' with node limit: {node_limit}")
    logger.info(f"🔧 Query: {query[:200] if query else 'None'}...")
    
    try:
        if not tool:
            logger.error("❌ No tool selected")
            answer = "⚠️ I couldn't determine the right tool for your question. Please try rephrasing your question or ask about viewing data, making changes, or exploring the database schema."
        elif tool not in valid_tools:
            logger.error(f"❌ Invalid tool: {tool}")
            answer = f"⚠️ Tool '{tool}' not recognized. Available tools: {', '.join(valid_tools)}"
        elif tool == "get_neo4j_schema":
            logger.info("📋 Calling get_neo4j_schema endpoint...")
            result = requests.post("http://localhost:8000/get_neo4j_schema", headers=headers, timeout=30)
            if result.ok:
                answer, graph_data = format_response_with_graph(result.json(), tool, node_limit)
                logger.info("✅ Schema retrieval successful")
            else:
                logger.error(f"❌ Schema query failed: {result.status_code} - {result.text}")
                answer = f"❌ Schema query failed: {result.text}"
        elif tool == "read_neo4j_cypher":
            if not query or not query.strip():
                logger.error("❌ No query provided for read operation")
                answer = "⚠️ I couldn't generate a valid query for your question. Try rephrasing or being more specific about what you want to see."
            else:
                logger.info("📖 Executing read query...")
                query_clean = clean_cypher_query(query)
                data = {
                    "query": query_clean, 
                    "params": {},
                    "node_limit": node_limit
                }
                result = requests.post("http://localhost:8000/read_neo4j_cypher", json=data, headers=headers, timeout=60)
                if result.ok:
                    answer, graph_data = format_response_with_graph(result.json(), tool, node_limit)
                    logger.info("✅ Read query successful")
                else:
                    logger.error(f"❌ Read query failed: {result.status_code} - {result.text}")
                    answer = f"❌ Query failed: {result.text}"
        elif tool == "write_neo4j_cypher":
            if not query or not query.strip():
                logger.error("❌ No query provided for write operation")
                answer = "⚠️ I couldn't generate a valid modification query. Please be more specific about what you want to create, update, or delete."
            else:
                logger.info("✏️ Executing write query...")
                query_clean = clean_cypher_query(query)
                data = {
                    "query": query_clean, 
                    "params": {},
                    "node_limit": node_limit
                }
                result = requests.post("http://localhost:8000/write_neo4j_cypher", json=data, headers=headers, timeout=60)
                if result.ok:
                    answer, graph_data = format_response_with_graph(result.json(), tool, node_limit)
                    logger.info("✅ Write query successful")
                else:
                    logger.error(f"❌ Write query failed: {result.status_code} - {result.text}")
                    answer = f"❌ Update failed: {result.text}"
        else:
            logger.error(f"❌ Unknown tool: {tool}")
            answer = f"❌ Unknown tool: {tool}"
    
    except requests.exceptions.Timeout:
        logger.error("⏰ Request timed out")
        answer = "⚠️ Query timed out. Try a simpler query or reduce the data scope."
    except requests.exceptions.ConnectionError:
        logger.error("🔌 Connection error")
        answer = "⚠️ Cannot connect to the database server. Please check if all services are running on localhost:8000."
    except Exception as e:
        logger.error(f"💥 Unexpected error in execute_tool_node: {e}")
        answer = f"⚠️ Execution failed: {str(e)}"
    
    logger.info(f"🏁 Tool execution completed. Graph data: {'Yes' if graph_data else 'No'}")
    
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

def build_agent():
    """Build and return the enhanced LangGraph agent with improved question handling"""
    workflow = StateGraph(state_schema=AgentState)
    
    # Add nodes with enhanced functions
    workflow.add_node("select_tool", RunnableLambda(enhanced_select_tool_node))
    workflow.add_node("execute_tool", RunnableLambda(execute_tool_node))
    
    # Set entry point
    workflow.set_entry_point("select_tool")
    
    # Add edges
    workflow.add_edge("select_tool", "execute_tool")
    workflow.add_edge("execute_tool", END)
    
    # Compile and return
    agent = workflow.compile()
    logger.info("🚀 Enhanced LangGraph agent built successfully with comprehensive question handling")
    return agent

# Utility functions for unlimited display support
def get_unlimited_display_queries():
    """Pre-optimized queries for unlimited display scenarios"""
    
    return {
        "complete_graph": """
            MATCH (n) 
            OPTIONAL MATCH (n)-[r]-(m) 
            RETURN n, r, m
        """,
        
        "all_connections": """
            MATCH (a)-[r]->(b) 
            RETURN a, r, b 
            UNION 
            MATCH (isolated) 
            WHERE NOT (isolated)--() 
            RETURN isolated, null as r, null as b
        """,
        
        "complete_network_paths": """
            MATCH path = (a)-[*1..3]-(b) 
            WHERE a <> b 
            RETURN path
        """,
        
        "schema_visualization": """
            MATCH (n)-[r]-(m) 
            RETURN DISTINCT labels(n) as StartLabels, type(r) as RelType, labels(m) as EndLabels
            UNION
            MATCH (isolated) 
            WHERE NOT (isolated)--() 
            RETURN DISTINCT labels(isolated) as StartLabels, null as RelType, null as EndLabels
        """,
        
        "connected_components": """
            MATCH (n) 
            OPTIONAL MATCH path = (n)-[*1..10]-(connected) 
            RETURN n, collect(DISTINCT connected) as component
        """
    }

# For testing purposes
if __name__ == "__main__":
    # Test the enhanced agent locally with various question types
    agent = build_agent()
    test_questions = [
        "What's in my database?",
        "Show me something",
        "Explore",
        "How many nodes do I have?",
        "Tell me about this database",
        "Overview",
        "Display connections",
        "Create a person named Alice",
        "What types exist?",
        "Schema"
    ]
    
    import asyncio
    
    async def test():
        for question in test_questions:
            print(f"\n🧪 Testing: {question}")
            test_state = AgentState(
                question=question,
                session_id="test_session",
                node_limit=50
            )
            result = await agent.ainvoke(test_state)
            print(f"✅ Result: Tool={result.get('tool')}, Query={result.get('query')}")
    
    # asyncio.run(test())
