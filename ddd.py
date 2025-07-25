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
logger = logging.getLogger("unlimited_langgraph_agent")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class AgentState(BaseModel):
    question: str
    session_id: str
    tool: str = ""
    query: str = ""
    trace: str = ""
    answer: str = ""
    graph_data: Optional[dict] = None
    node_limit: int = None  # Changed to None for unlimited by default

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

def optimize_query_for_unlimited_visualization(query: str) -> str:
    """
    Enhanced query optimization that REMOVES limits and optimizes for unlimited display
    """
    query = query.strip()
    
    # REMOVE any existing LIMIT clauses - we want unlimited display
    query = re.sub(r'\s+LIMIT\s+\d+', '', query, flags=re.IGNORECASE)
    
    # Enhance queries for better unlimited visualization by adding relationship context
    if "MATCH (n)" in query and "RETURN n" == query.split("RETURN")[-1].strip():
        # Add relationships for better graph visualization
        query = query.replace("RETURN n", "OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m")
        logger.info("🚀 Enhanced query with relationships for unlimited display")
    
    # For entity-specific queries, ensure we get relationships too
    entity_patterns = [
        (r"MATCH \(n:(\w+)\) RETURN n$", r"MATCH (n:\1) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m"),
        (r"MATCH \((\w+):(\w+)\) RETURN \1$", r"MATCH (\1:\2) OPTIONAL MATCH (\1)-[r]-(m) RETURN \1, r, m")
    ]
    
    for pattern, replacement in entity_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
            logger.info("🔗 Enhanced entity query with relationships")
            break
    
    logger.info(f"🚀 Optimized query for UNLIMITED display: {query}")
    
    return query

def smart_question_preprocessor(question: str) -> str:
    """Preprocess questions to make them more specific and actionable for unlimited display"""
    
    # Common question mappings for vague or general questions
    question_mappings = {
        # Very general questions - now optimized for unlimited display
        "what's in the database": "Show me all data in the database with relationships",
        "what do you have": "Display all types of nodes and their relationships", 
        "show me something": "Show me all data from the database",
        "what can i see": "Show me all the data available",
        "explore": "Show me all nodes and relationships",
        "anything": "Display all data from the database",
        "what's there": "Show me all data in the database",
        "database content": "Display all contents of the database",
        "what's here": "Show me all data available in the database",
        "tell me about this database": "Show me complete database structure and all content",
        
        # Count and size questions
        "how much data": "How many total nodes are in the database",
        "size": "Count all nodes and relationships in the database",
        "data size": "Count all nodes in the database",
        "how big": "Show me the size of the database",
        "database size": "Count all nodes and relationships",
        
        # Type and structure questions  
        "what kinds": "Show me all different types of nodes available",
        "what sorts": "Display the database schema and all available node types",
        "categories": "Show me all categories of data in the database",
        "what types": "Show me all different node types in the database",
        "structure": "Show me the complete database schema and structure",
        "schema": "Display the complete database schema",
        
        # Network and relationship questions - enhanced for unlimited display
        "connections": "Show me all connections between nodes in the database",
        "network": "Display the complete network structure of the database",
        "relationships": "Show me all relationships between all nodes",
        "graph": "Display the complete graph structure",
        
        # Exploration questions - enhanced for unlimited display
        "overview": "Give me a complete overview of all database content",
        "summary": "Show me a complete summary of everything in the database",
        "tour": "Give me a complete tour of the entire database",
        "walkthrough": "Show me all different parts of the database"
    }
    
    # Normalize the question
    normalized = question.lower().strip()
    
    # Check for direct mappings first
    for pattern, replacement in question_mappings.items():
        if pattern in normalized:
            logger.info(f"🔄 Preprocessed question for unlimited display: '{question}' → '{replacement}'")
            return replacement
    
    # Enhanced entity-specific preprocessing - keep specific entity requests intact
    entity_patterns = {
        r'\beda\b.*\b(group|team|department)\b': question,
        r'\beda\b.*\b(relationship|connection|network)\b': question,
        r'\b(person|people|user)\b.*\b(relationship|connection|network)\b': question,
        r'\b(company|organization)\b.*\b(relationship|connection|network)\b': question,
        r'\b(department|dept)\b.*\b(relationship|connection|network)\b': question,
        r'\b(group|team)\b.*\b(relationship|connection|network)\b': question,
    }
    
    # Check if this is a specific entity request that should NOT be preprocessed
    for pattern, keep_original in entity_patterns.items():
        if re.search(pattern, normalized, re.IGNORECASE):
            logger.info(f"🎯 Keeping specific entity request unchanged: '{question}'")
            return question
    
    # Expand abbreviated questions for unlimited display
    if len(normalized) < 10:
        if any(word in normalized for word in ['show', 'see', 'get', 'tell', 'what']):
            # Check if it mentions specific entities
            if any(entity in normalized for entity in ['eda', 'person', 'company', 'user', 'group', 'team']):
                logger.info(f"🎯 Keeping short specific entity request: '{question}'")
                return question
            else:
                expanded = f"Show me all data from the database: {question}"
                logger.info(f"🔄 Expanded short question for unlimited display: '{question}' → '{expanded}'")
                return expanded
    
    # Handle single word questions - enhanced for unlimited display
    single_word_mappings = {
        "nodes": "Show me all types of nodes in the database with their relationships",
        "data": "Show me all data from the database",
        "graph": "Display the complete graph structure with all nodes and relationships",
        "schema": "Show me the complete database schema",
        "structure": "Display the complete database structure with all relationships",
        "overview": "Give me a complete overview of all database content",
        "summary": "Show me a complete summary of all database content"
    }
    
    if normalized in single_word_mappings:
        expanded = single_word_mappings[normalized]
        logger.info(f"🔄 Expanded single word for unlimited display: '{question}' → '{expanded}'")
        return expanded
    
    # If it's a specific request mentioning entities, keep it unchanged
    if any(entity in normalized for entity in ['eda', 'person', 'people', 'company', 'user', 'group', 'team', 'department']):
        logger.info(f"🎯 Preserving specific entity request: '{question}'")
        return question
    
    logger.info(f"✅ Question unchanged: '{question}'")
    return question

def generate_fallback_response(original_question: str) -> str:
    """Generate intelligent fallback response when LLM fails - optimized for unlimited display"""
    
    question_lower = original_question.lower()
    
    # Schema questions
    if any(word in question_lower for word in ['schema', 'structure', 'types', 'labels', 'what kinds', 'what types', 'properties']):
        return "Tool: get_neo4j_schema"
    
    # Write operations  
    elif any(word in question_lower for word in ['create', 'add', 'insert', 'new', 'make', 'build']):
        return "Tool: write_neo4j_cypher\nQuery: // Please specify what you want to create"
    
    # Specific entity types with relationships (NO LIMITS)
    elif any(word in question_lower for word in ['person', 'people', 'user', 'users', 'individual', 'human']):
        return "Tool: read_neo4j_cypher\nQuery: MATCH (n:Person) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m"
    
    elif any(word in question_lower for word in ['company', 'organization', 'companies', 'business', 'corporation']):
        return "Tool: read_neo4j_cypher\nQuery: MATCH (n:Company) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m"
    
    elif any(word in question_lower for word in ['eda', 'eda group', 'eda team']):
        return "Tool: read_neo4j_cypher\nQuery: MATCH (n:EDA) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m"
    
    elif any(word in question_lower for word in ['movie', 'film', 'movies', 'films', 'cinema']):
        return "Tool: read_neo4j_cypher\nQuery: MATCH (n:Movie) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m"
    
    elif any(word in question_lower for word in ['product', 'products', 'item', 'items']):
        return "Tool: read_neo4j_cypher\nQuery: MATCH (n:Product) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m"
    
    # Count questions
    elif any(word in question_lower for word in ['count', 'how many', 'number', 'total', 'size', 'amount']):
        return "Tool: read_neo4j_cypher\nQuery: MATCH (n) RETURN count(n) as TotalNodes"
    
    # Relationship questions (NO LIMITS)
    elif any(word in question_lower for word in ['relationship', 'connection', 'link', 'connected', 'relationships', 'network', 'connections']):
        return "Tool: read_neo4j_cypher\nQuery: MATCH (a)-[r]->(b) RETURN a, r, b"
    
    # General exploration (NO LIMITS - show everything)
    elif any(word in question_lower for word in ['explore', 'show', 'display', 'see', 'view', 'what', 'database', 'data', 'tell', 'overview', 'all', 'everything']):
        return "Tool: read_neo4j_cypher\nQuery: MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m"
    
    # Default: show all data with relationships
    else:
        return "Tool: read_neo4j_cypher\nQuery: MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m"

# Enhanced system message for UNLIMITED display
SYS_MSG = """You are a Neo4j database expert assistant optimized for UNLIMITED graph visualization. Your job is to help users explore their entire graph database without any artificial limits.

**CRITICAL: NO NODE LIMITS** - Do NOT add LIMIT clauses unless specifically requested by the user. Show ALL data according to the command.

For ANY user question, you MUST respond with a tool selection and appropriate query. Here are your tools:

**TOOLS AVAILABLE:**
1. **read_neo4j_cypher** - For viewing, exploring, counting, finding data
   - Use for: "show me", "find", "how many", "what are", "display", "list", "get", "explore", "tell me about"
   - ALWAYS generates MATCH queries with RETURN statements
   - Include relationships when users want to see connections
   - DO NOT ADD LIMIT CLAUSES - show everything

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

**UNLIMITED DISPLAY EXAMPLES:**

User: "Show me EDA group with relationships"
Tool: read_neo4j_cypher
Query: MATCH (n:EDA) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m

User: "Display all nodes"
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN n

User: "Show me everything"
Tool: read_neo4j_cypher
Query: MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m

User: "Display all Person nodes with their relationships"
Tool: read_neo4j_cypher
Query: MATCH (p:Person) OPTIONAL MATCH (p)-[r]-(other) RETURN p, r, other

User: "Show me the complete network"
Tool: read_neo4j_cypher
Query: MATCH (a)-[r]->(b) RETURN a, r, b

User: "Find all connections"
Tool: read_neo4j_cypher
Query: MATCH (a)-[r]-(b) RETURN a, r, b

User: "Explore the database"
Tool: read_neo4j_cypher
Query: MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m

User: "Display Company network"
Tool: read_neo4j_cypher
Query: MATCH (c:Company) OPTIONAL MATCH (c)-[r]-(connected) RETURN c, r, connected

User: "Show all people"
Tool: read_neo4j_cypher
Query: MATCH (n:Person) RETURN n

User: "What's in my database?"
Tool: read_neo4j_cypher  
Query: MATCH (n) RETURN labels(n) as NodeType, count(*) as Count ORDER BY Count DESC

**IMPORTANT RULES FOR UNLIMITED DISPLAY:**
- NEVER add LIMIT clauses unless the user specifically asks for a limit
- When showing entities, include their relationships: OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m
- For "show all" or "display everything" commands, return complete data sets
- For specific entity queries, show ALL instances of that entity type
- For network/relationship queries, show ALL connections
- For exploration queries, show complete graph structure
- Only use count() for counting queries, not for limiting display

**ENTITY-SPECIFIC UNLIMITED PATTERNS:**
- "EDA" mentions → MATCH (n:EDA) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m
- "Person/People" → MATCH (n:Person) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m
- "Company/Companies" → MATCH (n:Company) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m
- "All/Everything" → MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m
- "Network/Connections" → MATCH (a)-[r]-(b) RETURN a, r, b

RESPOND NOW with Tool: and Query: for UNLIMITED display according to the user's command."""

# Cortex LLM configuration
API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
MODEL = "llama3.1-70b"

def enhanced_cortex_llm(prompt: str, session_id: str) -> str:
    """Enhanced Cortex LLM call optimized for unlimited display"""
    
    # Preprocess the question for unlimited display
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
        logger.info(f"🔄 Calling Cortex LLM for unlimited display: {processed_prompt[:100]}...")
        resp = requests.post(API_URL, headers=headers, json=payload, verify=False, timeout=30)
        resp.raise_for_status()
        
        raw_response = resp.text
        logger.info(f"📥 Raw Cortex response length: {len(raw_response)}")
        
        if "end_of_stream" in raw_response:
            parsed_response = raw_response.partition("end_of_stream")[0].strip()
        else:
            parsed_response = raw_response.strip()
        
        if len(parsed_response) < 20:
            logger.warning(f"⚠️ Short response detected, using unlimited fallback")
            return generate_fallback_response(prompt)
        
        logger.info(f"✅ LLM response processed for unlimited display")
        return parsed_response
        
    except Exception as e:
        logger.error(f"❌ Cortex LLM error: {e}")
        return generate_fallback_response(prompt)

def enhanced_parse_llm_output(llm_output):
    """Enhanced parsing optimized for unlimited display - removes any LIMIT additions"""
    allowed_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    trace = llm_output.strip()
    tool = None
    query = None
    
    logger.info(f"🔍 Parsing LLM output for unlimited display")
    
    # Extract tool
    tool_patterns = [
        r"Tool:\s*([\w_]+)",
        r"**Tool:**\s*([\w_]+)",
        r"Tool\s*[:=]\s*([\w_]+)",
        r"Selected tool:\s*([\w_]+)",
        r"Using:\s*([\w_]+)",
        r"I'll use\s*:\s*([\w_]+)",
        r"(?:Tool|TOOL)\s*[:=]\s*([\w_]+)",
        r"The tool is:\s*([\w_]+)",
        r"Choose tool:\s*([\w_]+)",
        r"Tool selection:\s*([\w_]+)"
    ]
    
    for pattern in tool_patterns:
        tool_match = re.search(pattern, llm_output, re.I)
        if tool_match:
            tname = tool_match.group(1).strip()
            if tname in allowed_tools:
                tool = tname
                logger.info(f"✅ Valid tool found: {tool}")
                break
    
    # Extract query
    query_patterns = [
        r"Query:\s*(.+?)(?:\n\n|\nTool|\nNote|\n$|$)",
        r"**Query:**\s*(.+?)(?:\n\n|\nTool|\nNote|\n$|$)",
        r"Query\s*[:=]\s*(.+?)(?:\n\n|\nTool|\nNote|\n$|$)",
        r"Cypher:\s*(.+?)(?:\n\n|\nTool|\nNote|\n$|$)",
        r"```cypher\s*(.+?)\s*```",
        r"```\s*(.+?)\s*```",
        r"The query is:\s*(.+?)(?:\n\n|\nTool|\nNote|\n$|$)",
        r"Query to execute:\s*(.+?)(?:\n\n|\nTool|\nNote|\n$|$)",
        r"MATCH\s+.+?(?=\n[A-Z]|\n\n|$)",
        r"CREATE\s+.+?(?=\n[A-Z]|\n\n|$)",
        r"MERGE\s+.+?(?=\n[A-Z]|\n\n|$)",
        r"CALL\s+.+?(?=\n[A-Z]|\n\n|$)"
    ]
    
    for pattern in query_patterns:
        query_match = re.search(pattern, llm_output, re.I | re.DOTALL)
        if query_match:
            query = query_match.group(1).strip() if query_match.groups() else query_match.group(0).strip()
            if query and len(query) > 3:
                query = re.sub(r'\s+', ' ', query.strip())
                query = re.sub(r'\s*(Note:|Explanation:|This query)', '', query, flags=re.I)
                logger.info(f"✅ Query found: {query[:100]}...")
                break
    
    # Fallback if parsing failed
    if not tool or not query:
        logger.warning("⚠️ Primary parsing failed, using unlimited fallback...")
        fallback_response = generate_fallback_response(llm_output)
        tool_match = re.search(r"Tool:\s*([\w_]+)", fallback_response)
        query_match = re.search(r"Query:\s*(.+)", fallback_response, re.DOTALL)
        
        if tool_match:
            tool = tool_match.group(1)
        if query_match:
            query = query_match.group(1).strip()
    
    # CRITICAL: Remove any LIMIT clauses that might have been added
    if query and tool == "read_neo4j_cypher":
        # Remove LIMIT clauses for unlimited display
        original_query = query
        query = re.sub(r'\s+LIMIT\s+\d+', '', query, flags=re.IGNORECASE)
        if original_query != query:
            logger.info(f"🚀 REMOVED LIMIT clause for unlimited display")
        
        # Ensure we have RETURN clause
        if "RETURN" not in query.upper():
            if query.upper().startswith("MATCH"):
                query += " RETURN n"
        
        # Clean up formatting
        query = re.sub(r'\s+', ' ', query.strip())
    
    logger.info(f"🎯 Final unlimited parsing - Tool: {tool}, Query: {query[:100] if query else 'None'}...")
    
    return tool, query, trace

def format_response_with_graph(result_data, tool_type, node_limit=None):
    """Enhanced response formatting for unlimited display"""
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
                        formatted_response += f"\n\n🕸️ **Complete graph visualization** with {node_count} nodes and {rel_count} relationships"
                
                return formatted_response, graph_data
        
        elif tool_type == "read_neo4j_cypher" and isinstance(result_data, dict):
            if "data" in result_data and "metadata" in result_data:
                data = result_data["data"]
                metadata = result_data["metadata"]
                graph_data = result_data.get("graph_data")
                
                formatted_response = f"""
📊 **Complete Query Results - No Limits Applied**

**🔢 Records:** {metadata['record_count']}  
**⚡ Time:** {metadata['execution_time_ms']}ms  
**🕐 Timestamp:** {metadata['timestamp'][:19]}
                """.strip()
                
                if not graph_data or not graph_data.get('nodes'):
                    if isinstance(data, list) and len(data) > 0:
                        if len(data) <= 5:
                            formatted_response += f"\n\n**📋 Complete Data:**\n```json\n{json.dumps(data, indent=2)}\n```"
                        else:
                            formatted_response += f"\n\n**📋 Data Sample (showing first 3 of {len(data)} total):**\n```json\n{json.dumps(data[:3], indent=2)}\n... and {len(data) - 3} more records\n```"
                    else:
                        formatted_response += "\n\n**📋 Data:** No records found"
                
                if graph_data and graph_data.get('nodes'):
                    node_count = len(graph_data['nodes'])
                    rel_count = len(graph_data.get('relationships', []))
                    
                    formatted_response += f"\n\n🕸️ **Complete graph visualization** with {node_count} nodes and {rel_count} relationships"
                    
                    if node_count > 0:
                        label_counts = {}
                        for node in graph_data['nodes']:
                            for label in node.get('labels', ['Unknown']):
                                label_counts[label] = label_counts.get(label, 0) + 1
                        
                        if len(label_counts) > 0:
                            label_summary = ", ".join([f"{label}({count})" for label, count in sorted(label_counts.items())])
                            formatted_response += f"\n**🏷️ Node Types:** {label_summary}"
                    
                    # Note that this is unlimited display
                    formatted_response += f"\n**🚀 Unlimited Display:** Showing ALL results without artificial limits"
                
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
🏗️ **Complete Database Schema**

**⚡ Time:** {metadata.get('execution_time_ms', 'N/A')}ms

**📊 Complete Overview:**
{chr(10).join(f"  • {item}" for item in schema_summary)}
                """.strip()
                
                return formatted_response, None
        
        # Fallback
        formatted_text = json.dumps(result_data, indent=2) if isinstance(result_data, (dict, list)) else str(result_data)
        return formatted_text, None
    
    except Exception as e:
        error_msg = f"❌ **Error formatting response:** {str(e)}"
        logger.error(error_msg)
        return error_msg, None

def enhanced_select_tool_node(state: AgentState) -> dict:
    """Enhanced tool selection optimized for unlimited display"""
    logger.info(f"🤔 Processing question for unlimited display: {state.question}")
    
    try:
        # Call LLM with unlimited display focus
        llm_output = enhanced_cortex_llm(state.question, state.session_id)
        logger.info(f"📥 LLM Response received for unlimited processing")
        
        # Use enhanced parsing that removes limits
        tool, query, trace = enhanced_parse_llm_output(llm_output)
        
        # Ultimate fallback if still no tool/query
        if not tool:
            logger.warning("🚨 Ultimate fallback: No tool detected, defaulting to unlimited exploration")
            tool = "read_neo4j_cypher"
            query = "MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m"
            trace = f"Ultimate unlimited fallback for: {state.question}"
        
        # Apply unlimited query optimization
        if query and tool == "read_neo4j_cypher":
            query = optimize_query_for_unlimited_visualization(query)
        
        logger.info(f"✅ Unlimited tool selection complete - Tool: {tool}, Query: {query[:100] if query else 'None'}")
        
        return {
            "question": state.question,
            "session_id": state.session_id,
            "tool": tool or "",
            "query": query or "",
            "trace": trace or f"Unlimited processing of: {state.question}",
            "answer": "",
            "graph_data": None,
            "node_limit": None  # Always None for unlimited
        }
        
    except Exception as e:
        logger.error(f"❌ Error in unlimited tool selection: {e}")
        
        # Emergency fallback for unlimited display
        return {
            "question": state.question,
            "session_id": state.session_id,
            "tool": "read_neo4j_cypher",
            "query": "MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m",
            "trace": f"Emergency unlimited fallback due to error: {str(e)}",
            "answer": f"I'll show you complete data from your database. Error details: {str(e)}",
            "graph_data": None,
            "node_limit": None
        }

def execute_tool_node(state: AgentState) -> dict:
    """Enhanced tool execution for unlimited display"""
    tool = state.tool
    query = state.query
    trace = state.trace
    answer = ""
    graph_data = None
    
    valid_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    
    logger.info(f"⚡ Executing unlimited tool: '{tool}'")
    logger.info(f"🔧 Unlimited query: {query[:200] if query else 'None'}...")
    
    try:
        if not tool:
            logger.error("❌ No tool selected")
            answer = "⚠️ I couldn't determine the right tool for your question."
        elif tool not in valid_tools:
            logger.error(f"❌ Invalid tool: {tool}")
            answer = f"⚠️ Tool '{tool}' not recognized. Available tools: {', '.join(valid_tools)}"
        elif tool == "get_neo4j_schema":
            logger.info("📋 Calling schema endpoint...")
            result = requests.post("http://localhost:8000/get_neo4j_schema", headers=headers, timeout=60)
            if result.ok:
                answer, graph_data = format_response_with_graph(result.json(), tool, None)
                logger.info("✅ Schema retrieval successful")
            else:
                logger.error(f"❌ Schema query failed: {result.status_code}")
                answer = f"❌ Schema query failed: {result.text}"
        elif tool == "read_neo4j_cypher":
            if not query or not query.strip():
                logger.error("❌ No query provided")
                answer = "⚠️ I couldn't generate a valid query for your question."
            else:
                logger.info("📖 Executing unlimited read query...")
                query_clean = clean_cypher_query(query)
                data = {
                    "query": query_clean, 
                    "params": {},
                    "node_limit": None  # CRITICAL: No limits
                }
                result = requests.post("http://localhost:8000/read_neo4j_cypher", json=data, headers=headers, timeout=120)
                if result.ok:
                    answer, graph_data = format_response_with_graph(result.json(), tool, None)
                    logger.info("✅ Unlimited read query successful")
                else:
                    logger.error(f"❌ Read query failed: {result.status_code}")
                    answer = f"❌ Query failed: {result.text}"
        elif tool == "write_neo4j_cypher":
            if not query or not query.strip():
                logger.error("❌ No query provided")
                answer = "⚠️ I couldn't generate a valid modification query."
            else:
                logger.info("✏️ Executing write query...")
                query_clean = clean_cypher_query(query)
                data = {
                    "query": query_clean, 
                    "params": {},
                    "node_limit": None  # No limits for write operations either
                }
                result = requests.post("http://localhost:8000/write_neo4j_cypher", json=data, headers=headers, timeout=120)
                if result.ok:
                    answer, graph_data = format_response_with_graph(result.json(), tool, None)
                    logger.info("✅ Write query successful")
                else:
                    logger.error(f"❌ Write query failed: {result.status_code}")
                    answer = f"❌ Update failed: {result.text}"
        else:
            logger.error(f"❌ Unknown tool: {tool}")
            answer = f"❌ Unknown tool: {tool}"
    
    except requests.exceptions.Timeout:
        logger.error("⏰ Request timed out")
        answer = "⚠️ Query timed out. The query may be too complex for unlimited display."
    except requests.exceptions.ConnectionError:
        logger.error("🔌 Connection error")
        answer = "⚠️ Cannot connect to the database server. Please check if services are running."
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        answer = f"⚠️ Execution failed: {str(e)}"
    
    logger.info(f"🏁 Unlimited tool execution completed. Graph data: {'Yes' if graph_data else 'No'}")
    
    return {
        "question": state.question,
        "session_id": state.session_id,
        "tool": tool,
        "query": query,
        "trace": trace,
        "answer": answer,
        "graph_data": graph_data,
        "node_limit": None  # Always None for unlimited
    }

def build_agent():
    """Build and return the unlimited LangGraph agent"""
    workflow = StateGraph(state_schema=AgentState)
    
    # Add nodes with unlimited functions
    workflow.add_node("select_tool", RunnableLambda(enhanced_select_tool_node))
    workflow.add_node("execute_tool", RunnableLambda(execute_tool_node))
    
    # Set entry point
    workflow.set_entry_point("select_tool")
    
    # Add edges
    workflow.add_edge("select_tool", "execute_tool")
    workflow.add_edge("execute_tool", END)
    
    # Compile and return
    agent = workflow.compile()
    logger.info("🚀 UNLIMITED LangGraph agent built successfully - NO NODE LIMITS")
    return agent

if __name__ == "__main__":
    agent = build_agent()
    logger.info("🚀 Unlimited Neo4j Graph Agent ready - displays everything according to commands!")
