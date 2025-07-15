# detailed_langgraph_neo4j.py - Production-ready LangGraph agent using your MCP server

import asyncio
import json
import re
import uuid
import requests
import urllib3
from typing import Dict, List, Any, Optional, TypedDict
from langgraph.graph import Graph, END
from langgraph.checkpoint.memory import MemorySaver
from fastmcp import Client
import nest_asyncio

nest_asyncio.apply()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class AgentState(TypedDict):
    """State structure for the LangGraph agent."""
    original_question: str
    current_query: str
    attempts: int
    max_attempts: int
    results: List[Dict]
    error_messages: List[str]
    schema_info: Dict
    final_answer: str
    question_type: str
    complexity_level: str
    cortex_attempts: int

class DetailedNeo4jAgent:
    """Production-ready Neo4j agent using LangGraph and your existing MCP server."""
    
    def __init__(self, mcp_script_path="mcpserver.py"):
        self.mcp_script_path = mcp_script_path
        
        # Your existing Cortex configuration
        self.cortex_config = {
            "url": "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete",
            "api_key": "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0",
            "app_id": "edadip",
            "aplctn_cd": "edagnai",
            "model": "llama3.1-70b",
            "sys_msg": "You are a powerful AI assistant specialized in Neo4j Cypher queries."
        }
        
        # Modern Neo4j syntax patterns and fixes
        self.syntax_fixes = {
            # Size function fixes
            r"size\(\s*\(([^)]+)\)\s*-\s*\[\s*\]\s*-\s*\(\s*\)\s*\)": r"COUNT { (\1)-[]-() }",
            r"size\(\s*\(([^)]+)\)\s*-\s*\[([^\]]*)\]\s*-\s*\(([^)]*)\)\s*\)": r"COUNT { (\1)-[\2]-(\3) }",
            
            # Length function fixes  
            r"length\(\s*\(([^)]+)\)\s*-\s*\[\s*\*\s*\]\s*-\s*\(([^)]*)\)\s*\)": r"COUNT { (\1)-[*]-(\2) }",
            
            # Common deprecated syntax
            r"has\(([^)]+)\)": r"\1 IS NOT NULL",
            r"\.(\w+)\s*=\s*": r".\1 = ",
        }
        
        # Question type classification
        self.question_patterns = {
            "connectivity": ["most connected", "highest degree", "centrality", "connections", "connected nodes"],
            "path_finding": ["shortest path", "path between", "route", "connected through"],
            "aggregation": ["count", "total", "average", "sum", "statistics", "how many"],
            "exploration": ["show me", "find", "list", "what", "which"],
            "schema": ["properties", "structure", "schema", "labels", "relationships"],
            "comparison": ["compare", "versus", "vs", "difference", "similar"],
            "temporal": ["recent", "latest", "trend", "over time", "history"],
            "write_operation": ["create", "add", "update", "delete", "remove", "insert"]
        }

    def extract_mcp_result(self, result) -> str:
        """Extract content from FastMCP CallToolResult using your existing method."""
        try:
            if hasattr(result, 'content') and result.content:
                content_item = result.content[0]
                if hasattr(content_item, 'text'):
                    return content_item.text
            return str(result)
        except Exception as e:
            return f"❌ Extraction error: {e}"

    async def call_mcp_tool(self, tool_name: str, arguments: Dict = None) -> str:
        """Call your MCP server tools with proper error handling."""
        try:
            async with Client(self.mcp_script_path) as client:
                result = await client.call_tool(tool_name, arguments or {})
                return self.extract_mcp_result(result)
        except Exception as e:
            error_msg = f"❌ MCP tool '{tool_name}' failed: {str(e)}"
            print(error_msg)
            return error_msg

    def generate_cypher_with_cortex(self, prompt: str) -> str:
        """Generate Cypher using your existing Cortex client."""
        payload = {
            "query": {
                "aplctn_cd": self.cortex_config["aplctn_cd"],
                "app_id": self.cortex_config["app_id"],
                "api_key": self.cortex_config["api_key"],
                "method": "cortex",
                "model": self.cortex_config["model"],
                "sys_msg": self.cortex_config["sys_msg"],
                "limit_convs": "0",
                "prompt": {"messages": [{"role": "user", "content": prompt}]},
                "session_id": str(uuid.uuid4())
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f'Snowflake Token="{self.cortex_config["api_key"]}"'
        }

        try:
            response = requests.post(
                self.cortex_config["url"], 
                headers=headers, 
                json=payload, 
                verify=False, 
                timeout=30
            )
            
            if response.status_code == 200:
                raw_text = response.text
                if "end_of_stream" in raw_text:
                    result = raw_text.split("end_of_stream")[0].strip()
                    return result if result else "MATCH (n) RETURN count(n)"
                return raw_text.strip() if raw_text.strip() else "MATCH (n) RETURN count(n)"
            else:
                return f"❌ Cortex error: HTTP {response.status_code}"
        except Exception as e:
            return f"❌ Cortex error: {str(e)[:100]}"

    def fix_cypher_syntax(self, cypher: str) -> str:
        """Apply modern Neo4j syntax fixes."""
        fixed_cypher = cypher.strip()
        
        for pattern, replacement in self.syntax_fixes.items():
            old_cypher = fixed_cypher
            fixed_cypher = re.sub(pattern, replacement, fixed_cypher, flags=re.IGNORECASE)
            if fixed_cypher != old_cypher:
                print(f"🔧 Applied syntax fix: {pattern} → {replacement}")
        
        return fixed_cypher

    def classify_question(self, question: str) -> tuple[str, str]:
        """Classify question type and complexity."""
        question_lower = question.lower()
        
        # Determine question type
        question_type = "general"
        for qtype, patterns in self.question_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                question_type = qtype
                break
        
        # Determine complexity
        complexity_indicators = {
            "simple": ["count", "total", "list", "show"],
            "medium": ["find", "where", "with", "having"],
            "complex": ["path", "connected through", "most", "compare", "analyze", "relationship between"]
        }
        
        complexity = "simple"
        for level, indicators in complexity_indicators.items():
            if any(indicator in question_lower for indicator in indicators):
                complexity = level
        
        return question_type, complexity

    def create_graph(self) -> Graph:
        """Create the detailed LangGraph workflow."""
        workflow = Graph()
        
        # Add all nodes
        workflow.add_node("initialize", self.initialize_state)
        workflow.add_node("classify_question", self.classify_question_node)
        workflow.add_node("gather_schema", self.gather_schema_info)
        workflow.add_node("generate_query", self.generate_cypher_query)
        workflow.add_node("validate_query", self.validate_query)
        workflow.add_node("execute_query", self.execute_query)
        workflow.add_node("handle_error", self.handle_execution_error)
        workflow.add_node("format_response", self.format_final_response)
        workflow.add_node("fallback_simple", self.fallback_to_simple_query)
        
        # Set entry point
        workflow.set_entry_point("initialize")
        
        # Define the flow
        workflow.add_edge("initialize", "classify_question")
        workflow.add_edge("classify_question", "gather_schema")
        workflow.add_edge("gather_schema", "generate_query")
        workflow.add_edge("generate_query", "validate_query")
        
        # Conditional routing after validation
        workflow.add_conditional_edges(
            "validate_query",
            self.should_execute_or_regenerate,
            {
                "execute": "execute_query",
                "regenerate": "generate_query",
                "fallback": "fallback_simple"
            }
        )
        
        workflow.add_edge("execute_query", "format_response")
        
        # Error handling flow
        workflow.add_conditional_edges(
            "format_response",
            self.check_execution_success,
            {
                "success": END,
                "retry": "handle_error",
                "fallback": "fallback_simple"
            }
        )
        
        workflow.add_edge("handle_error", "generate_query")
        workflow.add_edge("fallback_simple", "format_response")
        
        return workflow.compile(checkpointer=MemorySaver())

    async def initialize_state(self, state: AgentState) -> AgentState:
        """Initialize the agent state."""
        state.update({
            "attempts": 0,
            "max_attempts": 3,
            "results": [],
            "error_messages": [],
            "schema_info": {},
            "final_answer": "",
            "cortex_attempts": 0
        })
        
        print(f"🚀 Initialized agent for question: {state['original_question']}")
        return state

    async def classify_question_node(self, state: AgentState) -> AgentState:
        """Classify the user's question."""
        question_type, complexity = self.classify_question(state["original_question"])
        
        state["question_type"] = question_type
        state["complexity_level"] = complexity
        
        print(f"🧠 Classified as: {question_type} ({complexity} complexity)")
        return state

    async def gather_schema_info(self, state: AgentState) -> AgentState:
        """Gather database schema information using your MCP tools."""
        if not state["schema_info"]:
            print("📊 Gathering database schema information...")
            
            # Use your existing MCP tools
            schema_tasks = [
                ("labels", "list_labels"),
                ("relationships", "list_relationships"), 
                ("summary", "database_summary"),
                ("health", "health_check")
            ]
            
            schema_info = {}
            for key, tool_name in schema_tasks:
                try:
                    result = await self.call_mcp_tool(tool_name)
                    if not result.startswith("❌"):
                        schema_info[key] = json.loads(result)
                    else:
                        schema_info[key] = {"error": result}
                except json.JSONDecodeError:
                    schema_info[key] = {"raw": result}
            
            state["schema_info"] = schema_info
            print(f"✅ Schema gathered: {len(schema_info)} components")
        
        return state

    async def generate_cypher_query(self, state: AgentState) -> AgentState:
        """Generate Cypher query with enhanced context."""
        state["cortex_attempts"] += 1
        print(f"🤖 Generating Cypher query (attempt {state['cortex_attempts']})...")
        
        # Build comprehensive context
        context = self.build_query_context(state)
        
        # Generate query with Cortex
        raw_cypher = self.generate_cypher_with_cortex(context)
        
        if raw_cypher.startswith("❌"):
            state["error_messages"].append(f"Cortex generation failed: {raw_cypher}")
            state["current_query"] = "MATCH (n) RETURN count(n) LIMIT 1"  # Fallback
        else:
            # Apply syntax fixes
            fixed_cypher = self.fix_cypher_syntax(raw_cypher)
            state["current_query"] = fixed_cypher
            
            if fixed_cypher != raw_cypher:
                print(f"🔧 Applied syntax fixes")
        
        print(f"📝 Generated query: {state['current_query']}")
        return state

    def build_query_context(self, state: AgentState) -> str:
        """Build comprehensive context for Cypher generation."""
        schema_info = state.get("schema_info", {})
        
        # Extract schema details
        labels = []
        relationships = []
        
        if "labels" in schema_info and "labels" in schema_info["labels"]:
            labels = schema_info["labels"]["labels"]
        
        if "relationships" in schema_info and "relationship_types" in schema_info["relationships"]:
            relationships = schema_info["relationships"]["relationship_types"]
        
        # Build context based on question type
        context_parts = [
            "Generate a modern Neo4j Cypher query for the following request.",
            "",
            f"Database Schema:",
            f"- Available Node Labels: {', '.join(labels) if labels else 'Unknown'}",
            f"- Available Relationships: {', '.join(relationships) if relationships else 'Unknown'}",
            "",
            f"Question Type: {state.get('question_type', 'general')}",
            f"Complexity: {state.get('complexity_level', 'simple')}",
            ""
        ]
        
        # Add previous errors for learning
        if state.get("error_messages"):
            context_parts.extend([
                "Previous errors to avoid:",
                *[f"- {error}" for error in state["error_messages"][-2:]],  # Last 2 errors
                ""
            ])
        
        # Add syntax guidelines
        context_parts.extend([
            "IMPORTANT Syntax Rules:",
            "- Use COUNT { (n)-[]-() } instead of size((n)-[]->())",
            "- Use IS NOT NULL instead of has(property)",
            "- Always include LIMIT for large result sets",
            "- Use proper Neo4j 5.x syntax",
            "",
            f"User Question: {state['original_question']}",
            "",
            "Return ONLY the Cypher query, no explanations."
        ])
        
        return "\n".join(context_parts)

    async def validate_query(self, state: AgentState) -> AgentState:
        """Validate the generated Cypher query."""
        query = state["current_query"]
        
        # Basic validation checks
        validation_issues = []
        
        # Check for deprecated syntax
        deprecated_patterns = [
            (r"size\(\s*\([^)]+\)\s*-\s*\[\s*\]\s*-\s*\(\s*\)\s*\)", "size() function usage"),
            (r"has\(\s*\w+\s*\)", "has() function usage"),
            (r"length\(\s*path\s*\)", "length() function on paths")
        ]
        
        for pattern, issue in deprecated_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                validation_issues.append(issue)
        
        # Check for missing LIMIT on potentially large queries
        if "ORDER BY" in query.upper() and "LIMIT" not in query.upper():
            if state["complexity_level"] == "complex":
                validation_issues.append("Missing LIMIT clause for complex query")
        
        if validation_issues:
            state["error_messages"].extend(validation_issues)
            print(f"⚠️ Validation issues found: {', '.join(validation_issues)}")
        else:
            print(f"✅ Query validation passed")
        
        return state

    def should_execute_or_regenerate(self, state: AgentState) -> str:
        """Decide whether to execute, regenerate, or fallback."""
        # If we have validation issues and haven't tried too many times
        if state.get("error_messages") and state["cortex_attempts"] < 2:
            return "regenerate"
        
        # If we've tried too many times, fallback to simple query
        if state["attempts"] >= state["max_attempts"]:
            return "fallback"
        
        # Otherwise, execute the query
        return "execute"

    async def execute_query(self, state: AgentState) -> AgentState:
        """Execute the Cypher query using your MCP server."""
        state["attempts"] += 1
        print(f"⚡ Executing query (attempt {state['attempts']})...")
        
        query = state["current_query"]
        
        # Determine which MCP tool to use
        is_write = any(keyword in query.upper() for keyword in ["CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP"])
        tool_name = "write_neo4j_cypher" if is_write else "read_neo4j_cypher"
        
        # Execute the query
        result = await self.call_mcp_tool(tool_name, {"query": query})
        
        if result.startswith("❌"):
            state["error_messages"].append(result)
            print(f"❌ Query execution failed: {result}")
        else:
            try:
                parsed_result = json.loads(result)
                state["results"].append(parsed_result)
                print(f"✅ Query executed successfully, got {len(parsed_result)} records")
            except json.JSONDecodeError:
                # Handle non-JSON results
                state["results"].append({"raw_result": result})
                print(f"✅ Query executed, got raw result")
        
        return state

    async def handle_execution_error(self, state: AgentState) -> AgentState:
        """Handle execution errors intelligently."""
        if state["error_messages"]:
            last_error = state["error_messages"][-1]
            print(f"🛠️ Handling error: {last_error[:100]}...")
            
            # Analyze error and suggest fixes
            if "SyntaxError" in last_error:
                print("🔧 Syntax error detected - will regenerate with fixes")
            elif "timeout" in last_error.lower():
                print("🔧 Timeout detected - will add LIMIT clause")
            elif "size()" in last_error:
                print("🔧 Size function error - will use COUNT syntax")
        
        return state

    def check_execution_success(self, state: AgentState) -> str:
        """Check if execution was successful."""
        if state["results"] and not state["error_messages"]:
            return "success"
        elif state["attempts"] < state["max_attempts"] and state["cortex_attempts"] < 3:
            return "retry"
        else:
            return "fallback"

    async def fallback_to_simple_query(self, state: AgentState) -> AgentState:
        """Fallback to a simple, guaranteed-to-work query."""
        print("🔄 Falling back to simple query approach...")
        
        question_type = state.get("question_type", "general")
        
        # Map question types to simple, working queries
        fallback_queries = {
            "connectivity": "MATCH (n) RETURN n, COUNT { (n)-[]-() } as connections ORDER BY connections DESC LIMIT 10",
            "aggregation": "MATCH (n) RETURN count(n) as total_nodes",
            "schema": "CALL db.labels() YIELD label RETURN label LIMIT 20",
            "exploration": "MATCH (n) RETURN n LIMIT 10",
            "general": "MATCH (n) RETURN count(n) as node_count"
        }
        
        fallback_query = fallback_queries.get(question_type, fallback_queries["general"])
        
        # Execute fallback query
        result = await self.call_mcp_tool("read_neo4j_cypher", {"query": fallback_query})
        
        if not result.startswith("❌"):
            try:
                parsed_result = json.loads(result)
                state["results"] = [parsed_result]
                state["final_answer"] = f"🔄 **Fallback Result:** Used simplified query due to complexity.\n\nQuery: `{fallback_query}`"
            except json.JSONDecodeError:
                state["results"] = [{"raw_result": result}]
        
        return state

    async def format_final_response(self, state: AgentState) -> AgentState:
        """Format the final response based on question type and results."""
        if not state["results"]:
            state["final_answer"] = f"❌ Unable to answer '{state['original_question']}' after {state['attempts']} attempts."
            return state
        
        question_type = state.get("question_type", "general")
        results = state["results"]
        
        # Format based on question type
        if question_type == "connectivity":
            state["final_answer"] = self.format_connectivity_results(results, state["original_question"])
        elif question_type == "aggregation":
            state["final_answer"] = self.format_aggregation_results(results, state["original_question"])
        elif question_type == "schema":
            state["final_answer"] = self.format_schema_results(results, state["original_question"])
        elif question_type == "exploration":
            state["final_answer"] = self.format_exploration_results(results, state["original_question"])
        else:
            state["final_answer"] = self.format_general_results(results, state["original_question"])
        
        # Add query information if helpful
        if state.get("current_query") and not state["final_answer"].startswith("🔄"):
            state["final_answer"] += f"\n\n**Query used:** `{state['current_query']}`"
        
        return state

    def format_connectivity_results(self, results: List[Dict], question: str) -> str:
        """Format results for connectivity questions."""
        if not results or not results[0]:
            return "No connectivity data found."
        
        data = results[0]
        if not data:
            return "No nodes found in the database."
        
        response = "🔗 **Most Connected Nodes:**\n\n"
        
        # Handle different result formats
        for i, record in enumerate(data[:10], 1):
            if isinstance(record, dict):
                # Extract node information
                node_info = record.get('n', {})
                connections = record.get('connections', record.get('degree', 0))
                
                # Get node labels and properties
                if isinstance(node_info, dict):
                    labels = node_info.get('labels', ['Unknown'])
                    properties = node_info.get('properties', {})
                    name = properties.get('name', properties.get('title', f"Node {i}"))
                else:
                    labels = ['Node']
                    name = f"Item {i}"
                
                response += f"{i}. **{name}** ({labels[0] if labels else 'Node'})\n"
                response += f"   Connections: {connections}\n\n"
        
        return response

    def format_aggregation_results(self, results: List[Dict], question: str) -> str:
        """Format results for aggregation questions."""
        if not results or not results[0]:
            return "No aggregation data available."
        
        data = results[0]
        response = "📊 **Database Statistics:**\n\n"
        
        for record in data[:5]:  # Show top 5 results
            if isinstance(record, dict):
                for key, value in record.items():
                    response += f"• **{key.replace('_', ' ').title()}:** {value:,}\n"
        
        return response

    def format_schema_results(self, results: List[Dict], question: str) -> str:
        """Format results for schema questions."""
        if not results or not results[0]:
            return "No schema information available."
        
        data = results[0]
        response = "🏗️ **Database Schema:**\n\n"
        
        for record in data:
            if isinstance(record, dict):
                if 'label' in record:
                    response += f"• **Label:** {record['label']}\n"
                elif 'relationshipType' in record:
                    response += f"• **Relationship:** {record['relationshipType']}\n"
                else:
                    for key, value in record.items():
                        response += f"• **{key}:** {value}\n"
        
        return response

    def format_exploration_results(self, results: List[Dict], question: str) -> str:
        """Format results for exploration questions."""
        if not results or not results[0]:
            return "No data found to explore."
        
        data = results[0]
        response = f"🔍 **Found {len(data)} items:**\n\n"
        
        for i, record in enumerate(data[:5], 1):  # Show first 5
            if isinstance(record, dict):
                response += f"{i}. "
                if 'n' in record:
                    node = record['n']
                    if isinstance(node, dict) and 'properties' in node:
                        props = node['properties']
                        name = props.get('name', props.get('title', f"Item {i}"))
                        response += f"**{name}**"
                        if node.get('labels'):
                            response += f" ({node['labels'][0]})"
                    else:
                        response += f"**Node {i}**"
                else:
                    # Show first few properties
                    shown_props = list(record.items())[:3]
                    prop_strs = [f"{k}: {v}" for k, v in shown_props]
                    response += ", ".join(prop_strs)
                response += "\n"
        
        if len(data) > 5:
            response += f"\n... and {len(data) - 5} more items.\n"
        
        return response

    def format_general_results(self, results: List[Dict], question: str) -> str:
        """Format general results."""
        if not results:
            return "No results found."
        
        response = "📋 **Results:**\n\n"
        response += f"```json\n{json.dumps(results[0][:10] if results[0] else results, indent=2)}\n```"
        
        return response

    async def run(self, question: str) -> str:
        """Run the complete agent workflow."""
        initial_state = AgentState(
            original_question=question,
            current_query="",
            attempts=0,
            max_attempts=3,
            results=[],
            error_messages=[],
            schema_info={},
            final_answer="",
            question_type="general",
            complexity_level="simple",
            cortex_attempts=0
        )
        
        graph = self.create_graph()
        
        try:
            final_state = await graph.ainvoke(initial_state)
            return final_state["final_answer"]
        except Exception as e:
            return f"❌ Agent error: {str(e)}"

# Test and demonstration
async def test_detailed_agent():
    """Test the detailed agent with various question types."""
    agent = DetailedNeo4jAgent()
    
    test_questions = [
        "show me nodes with most connected nodes in the database?",
        "what properties does the user node have?", 
        "how many nodes are in the database?",
        "list all the different types of nodes",
        "find me some sample data from the database",
        "what's the structure of this database?",
    ]
    
    print("🧪 Testing Detailed Neo4j LangGraph Agent")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🤔 **Test {i}:** {question}")
        print("-" * 50)
        
        answer = await agent.run(question)
        print(f"\n🎯 **Answer:**\n{answer}\n")
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_detailed_agent())
