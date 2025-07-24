# Enhanced system message with better general question handling
SYS_MSG = """You are a Neo4j database expert assistant. Your job is to help users explore and interact with their graph database.

For ANY user question, you MUST respond with a tool selection and appropriate query. Here are your tools:

**TOOLS AVAILABLE:**
1. **read_neo4j_cypher** - For viewing, exploring, counting, finding data
   - Use for: "show me", "find", "how many", "what are", "display", "list", "get"
   - ALWAYS generates MATCH queries with RETURN statements

2. **write_neo4j_cypher** - For creating, updating, deleting data  
   - Use for: "create", "add", "update", "delete", "remove", "insert"
   - Generates CREATE, MERGE, SET, DELETE queries

3. **get_neo4j_schema** - For database structure questions
   - Use for: "schema", "structure", "what types", "what labels", "what properties"
   - NO query needed - just returns database structure

**RESPONSE FORMAT (REQUIRED):**
```
Tool: [tool_name]
Query: [cypher_query_or_none_for_schema]
```

**EXAMPLES FOR COMMON QUESTIONS:**

User: "What's in my database?"
Tool: read_neo4j_cypher  
Query: MATCH (n) RETURN labels(n) as NodeType, count(*) as Count ORDER BY Count DESC LIMIT 10

User: "Show me some data"
Tool: read_neo4j_cypher
Query: MATCH (n) RETURN n LIMIT 20

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

**IMPORTANT RULES:**
- EVERY question gets a tool + query response
- For vague questions, use read_neo4j_cypher with broad MATCH queries
- ALWAYS include RETURN clause in read queries
- Add LIMIT to prevent large results
- If unsure, default to showing sample data with read_neo4j_cypher

RESPOND NOW with Tool: and Query: for the user's question."""

def enhanced_parse_llm_output(llm_output):
    """Enhanced parsing with better fallback handling for general questions"""
    allowed_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    trace = llm_output.strip()
    tool = None
    query = None
    
    logger.info(f"🔍 Enhanced parsing of LLM output (length: {len(llm_output)})")
    logger.info(f"🔍 LLM output preview: {llm_output[:500]}...")
    
    # Try multiple tool extraction patterns
    tool_patterns = [
        r"Tool:\s*([\w_]+)",                    # Standard
        r"**Tool:**\s*([\w_]+)",                # Bold
        r"Tool\s*[:=]\s*([\w_]+)",              # Flexible separator
        r"Selected tool:\s*([\w_]+)",           # Alternative phrasing
        r"Using:\s*([\w_]+)",                   # Informal
        r"I'll use\s*:\s*([\w_]+)",            # Natural language
        r"(?:Tool|TOOL)\s*[:=]\s*([\w_]+)",    # Case insensitive
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
    
    # Try multiple query extraction patterns
    query_patterns = [
        r"Query:\s*(.+?)(?:\n\n|\nTool|\nNote|\n$|$)",     # More flexible ending
        r"**Query:**\s*(.+?)(?:\n\n|\nTool|\nNote|\n$|$)", # Bold format
        r"Query\s*[:=]\s*(.+?)(?:\n\n|\nTool|\nNote|\n$|$)", # Flexible separator
        r"Cypher:\s*(.+?)(?:\n\n|\nTool|\nNote|\n$|$)",    # Alternative
        r"```cypher\s*(.+?)\s*```",                        # Code block
        r"```\s*(.+?)\s*```",                              # Generic code block
        r"MATCH\s+.+",                                     # Direct MATCH detection
        r"CREATE\s+.+",                                    # Direct CREATE detection
        r"MERGE\s+.+",                                     # Direct MERGE detection
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
                logger.info(f"✅ Valid query found and cleaned")
                break
    
    # Enhanced fallback logic for common question types
    if not tool or not query:
        logger.warning("⚠️ Primary parsing failed, attempting intelligent fallback...")
        
        lower_output = llm_output.lower()
        original_question = lower_output  # Assume the input contains the question
        
        # Analyze question intent
        question_indicators = {
            'read': ['show', 'display', 'find', 'get', 'what', 'how many', 'count', 'list', 'see', 'view', 'explore', 'tell me'],
            'write': ['create', 'add', 'insert', 'make', 'new', 'update', 'set', 'change', 'delete', 'remove'],
            'schema': ['schema', 'structure', 'types', 'labels', 'properties', 'what kind', 'what sorts']
        }
        
        # Score each category
        scores = {'read': 0, 'write': 0, 'schema': 0}
        for category, keywords in question_indicators.items():
            for keyword in keywords:
                if keyword in lower_output:
                    scores[category] += 1
        
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
        
        # Generate appropriate fallback query
        if not query and tool != "get_neo4j_schema":
            if tool == "read_neo4j_cypher":
                # Generate smart read queries based on question content
                if any(word in lower_output for word in ['count', 'how many', 'number']):
                    query = "MATCH (n) RETURN count(n) as TotalNodes"
                elif any(word in lower_output for word in ['person', 'people', 'user']):
                    query = "MATCH (n:Person) RETURN n LIMIT 25"
                elif any(word in lower_output for word in ['company', 'organization']):
                    query = "MATCH (n:Company) RETURN n LIMIT 25"
                elif any(word in lower_output for word in ['movie', 'film']):
                    query = "MATCH (n:Movie) RETURN n LIMIT 25"
                elif any(word in lower_output for word in ['relationship', 'connection', 'link']):
                    query = "MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 20"
                elif any(word in lower_output for word in ['all', 'everything', 'database', 'data']):
                    query = "MATCH (n) RETURN n LIMIT 30"
                elif any(word in lower_output for word in ['type', 'kind', 'category']):
                    query = "MATCH (n) RETURN DISTINCT labels(n) as NodeTypes"
                else:
                    # Very general fallback
                    query = "MATCH (n) RETURN n LIMIT 20"
                
                logger.info(f"🔄 Generated fallback read query: {query}")
                
            elif tool == "write_neo4j_cypher":
                # For write operations, we need more specific info
                query = "// Unable to generate specific write query from the request"
                logger.info(f"🔄 Write operation needs more specific instructions")
    
    # Final validation and cleanup
    if query and tool != "get_neo4j_schema":
        # Ensure query has proper structure
        query = query.strip().rstrip(';')  # Remove trailing semicolon
        if tool == "read_neo4j_cypher" and "RETURN" not in query.upper():
            if query.upper().startswith("MATCH"):
                query += " RETURN n LIMIT 25"
            else:
                query = f"MATCH (n) WHERE {query} RETURN n LIMIT 25"
        
        # Add LIMIT if missing for read queries
        if (tool == "read_neo4j_cypher" and 
            "LIMIT" not in query.upper() and 
            "count(" not in query.lower() and
            "COUNT(" not in query):
            query += " LIMIT 25"
    
    logger.info(f"🎯 Final parsing results - Tool: {tool}, Query: {query[:100] if query else 'None'}...")
    
    return tool, query, trace

def enhanced_select_tool_node(state: AgentState) -> dict:
    """Enhanced tool selection with better question handling"""
    logger.info(f"🤔 Processing question: {state.question}")
    
    try:
        # Call LLM with enhanced prompt
        llm_output = cortex_llm(state.question, state.session_id)
        logger.info(f"📥 LLM Response received (length: {len(llm_output)})")
        
        # Use enhanced parsing
        tool, query, trace = enhanced_parse_llm_output(llm_output)
        
        # If still no tool/query, provide ultimate fallback
        if not tool:
            logger.warning("🚨 Ultimate fallback: No tool detected, defaulting to exploration")
            tool = "read_neo4j_cypher"
            query = "MATCH (n) RETURN n LIMIT 20"
            trace = f"Fallback response for: {state.question}"
        
        # Optimize query for visualization if needed
        if query and tool == "read_neo4j_cypher":
            query = optimize_query_for_visualization(query, state.node_limit)
        
        logger.info(f"✅ Tool selection complete - Tool: {tool}, Query: {query[:100] if query else 'None'}")
        
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

def smart_question_preprocessor(question: str) -> str:
    """Preprocess questions to make them more specific and actionable"""
    
    # Common question mappings
    question_mappings = {
        # Very general questions
        "what's in the database": "Show me a sample of all data in the database",
        "what do you have": "Display different types of nodes and their counts", 
        "show me something": "Show me sample data from the database",
        "what can i see": "Show me the different types of data available",
        "explore": "Show me a sample of nodes and relationships",
        "anything": "Display sample data from the database",
        
        # Count questions
        "how much data": "How many total nodes are in the database",
        "size": "Count all nodes and relationships in the database",
        
        # Type questions  
        "what kinds": "Show me the different types of nodes available",
        "what sorts": "Display the database schema and available node types",
        "categories": "Show me the categories of data in the database",
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
        if any(word in normalized for word in ['show', 'see', 'get']):
            expanded = f"Show me sample data from the database: {question}"
            logger.info(f"🔄 Expanded short question: '{question}' → '{expanded}'")
            return expanded
    
    return question

# Updated cortex_llm function with better error handling
def enhanced_cortex_llm(prompt: str, session_id: str) -> str:
    """Enhanced Cortex LLM call with preprocessing and error handling"""
    
    # Preprocess the question
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
            return f"Tool: read_neo4j_cypher\nQuery: MATCH (n) RETURN n LIMIT 20"
        
        logger.info(f"✅ Enhanced LLM response processed successfully")
        return parsed_response
        
    except Exception as e:
        logger.error(f"❌ Enhanced Cortex LLM API error: {e}")
        
        # Intelligent fallback based on original question
        fallback_response = generate_fallback_response(prompt)
        logger.info(f"🔄 Using intelligent fallback response")
        return fallback_response

def generate_fallback_response(original_question: str) -> str:
    """Generate intelligent fallback response when LLM fails"""
    
    question_lower = original_question.lower()
    
    # Schema questions
    if any(word in question_lower for word in ['schema', 'structure', 'types', 'labels']):
        return "Tool: get_neo4j_schema"
    
    # Write operations  
    elif any(word in question_lower for word in ['create', 'add', 'insert', 'new']):
        return "Tool: write_neo4j_cypher\nQuery: // Please specify what you want to create"
    
    # Specific entity types
    elif any(word in question_lower for word in ['person', 'people', 'user']):
        return "Tool: read_neo4j_cypher\nQuery: MATCH (n:Person) RETURN n LIMIT 25"
    
    elif any(word in question_lower for word in ['company', 'organization']):
        return "Tool: read_neo4j_cypher\nQuery: MATCH (n:Company) RETURN n LIMIT 25"
    
    elif any(word in question_lower for word in ['movie', 'film']):
        return "Tool: read_neo4j_cypher\nQuery: MATCH (n:Movie) RETURN n LIMIT 25"
    
    # Count questions
    elif any(word in question_lower for word in ['count', 'how many', 'number']):
        return "Tool: read_neo4j_cypher\nQuery: MATCH (n) RETURN count(n) as TotalNodes"
    
    # Relationship questions
    elif any(word in question_lower for word in ['relationship', 'connection', 'link']):
        return "Tool: read_neo4j_cypher\nQuery: MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 20"
    
    # Default: show sample data
    else:
        return "Tool: read_neo4j_cypher\nQuery: MATCH (n) RETURN n LIMIT 20"
