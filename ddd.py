# updated_langgraph_agent.py - LangGraph agent optimized for the new MCP server

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

class EnhancedAgentState(TypedDict):
    """Enhanced state structure for the LangGraph agent."""
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
    validation_result: Dict
    performance_metrics: Dict
    sample_data: Dict

class OptimizedNeo4jAgent:
    """Production-ready Neo4j agent using the specialized LangGraph MCP server."""
    
    def __init__(self, mcp_script_path="langgraph_mcpserver.py"):
        self.mcp_script_path = mcp_script_path
        
        # Your existing Cortex configuration
        self.cortex_config = {
            "url": "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete",
            "api_key": "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0",
            "app_id": "edadip",
            "aplctn_cd": "edagnai",
            "model": "llama3.1-70b",
            "sys_msg": "You are a powerful AI assistant specialized in Neo4j Cypher queries. Generate modern Neo4j 5.x compatible syntax."
        }
        
        # Enhanced question type classification
        self.question_patterns = {
            "connectivity": [
                "most connected", "highest degree", "centrality", "connections", 
                "connected nodes", "node degree", "network analysis", "hub nodes"
            ],
            "path_finding": [
                "shortest path", "path between", "route", "connected through", 
                "distance", "steps between", "reachable"
            ],
            "aggregation": [
                "count", "total", "average", "sum", "statistics", "how many", 
                "distribution", "metrics", "analyze"
            ],
            "exploration": [
                "show me", "find", "list", "what", "which", "sample", 
                "examples", "browse", "explore"
            ],
            "schema": [
                "properties", "structure", "schema", "labels", "relationships", 
                "types", "model", "design"
            ],
            "comparison": [
                "compare", "versus", "vs", "difference", "similar", 
                "between", "contrast"
            ],
            "temporal": [
                "recent", "latest", "trend", "over time", "history", 
                "when", "timeline", "changes"
            ],
            "write_operation": [
                "create", "add", "update", "delete", "remove", "insert", 
                "modify", "set", "merge"
            ]
        }

    def extract_mcp_result(self, result) -> str:
        """Extract content from FastMCP CallToolResult."""
        try:
            if hasattr(result, 'content') and result.content:
                content_item = result.content[0]
                if hasattr(content_item, 'text'):
                    return content_item.text
            return str(result)
        except Exception as e:
            return f"❌ Extraction error: {e}"

    async def call_mcp_tool(self, tool_name: str, arguments: Dict = None) -> str:
        """Call the specialized MCP server tools."""
        try:
            async with Client(self.mcp_script_path) as client:
                result = await client.call_tool(tool_name, arguments or {})
                return self.extract_mcp_result(result)
        except Exception as e:
            error_msg = f"❌ MCP tool '{tool_name}' failed: {str(e)}"
            print(error_msg)
            return error_msg

    def generate_cypher_with_cortex(self, prompt: str) -> str:
        """Generate Cypher using your existing Cortex client with enhanced prompting."""
        enhanced_prompt = f"""
        {prompt}
        
        CRITICAL SYNTAX REQUIREMENTS:
        - Use COUNT {{ (n)-[]-() }} instead of size((n)-[]->())
        - Use property IS NOT NULL instead of has(property)
        - Always include LIMIT for ORDER BY queries
        - Use modern Neo4j 5.x syntax only
        - Return ONLY the Cypher query, no explanations
        """
        
        payload = {
            "query": {
                "aplctn_cd": self.cortex_config["aplctn_cd"],
                "app_id": self.cortex_config["app_id"],
                "api_key": self.cortex_config["api_key"],
                "method": "cortex",
                "model": self.cortex_config["model"],
                "sys_msg": self.cortex_config["sys_msg"],
                "limit_convs": "0",
                "prompt": {"messages": [{"role": "user", "content": enhanced_prompt}]},
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

    def classify_question(self, question: str) -> tuple[str, str]:
        """Enhanced question classification."""
        question_lower = question.lower()
        
        # Determine question type with scoring
        type_scores = {}
        for qtype, patterns in self.question_patterns.items():
            score = sum(1 for pattern in patterns if pattern in question_lower)
            if score > 0:
                type_scores[qtype] = score
        
        # Get the highest scoring type
        question_type = max(type_scores, key=type_scores.get) if type_scores else "general"
        
        # Enhanced complexity analysis
        complexity_indicators = {
            "simple": ["count", "total", "list", "show", "one", "single"],
            "medium": ["find", "where", "with", "having", "some", "many"],
            "complex": [
                "path", "connected through", "most", "compare", "analyze", 
                "relationship between", "distribution", "pattern", "network"
            ]
        }
        
        complexity_scores = {}
        for level, indicators in complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in question_lower)
            complexity_scores[level] = score
        
        # Determine complexity
        if complexity_scores["complex"] > 0:
            complexity = "complex"
        elif complexity_scores["medium"] > complexity_scores["simple"]:
            complexity = "medium"
        else:
            complexity = "simple"
        
        return question_type, complexity

    def create_graph(self) -> Graph:
        """Create the enhanced LangGraph workflow."""
        workflow = Graph()
        
        # Add all nodes
        workflow.add_node("initialize", self.initialize_state)
        workflow.add_node("health_check", self.check_server_health)
        workflow.add_node("classify_question", self.classify_question_node)
        workflow.add_node("gather_enhanced_schema", self.gather_enhanced_schema)
        workflow.add_node("get_sample_data", self.get_sample_data)
        workflow.add_node("generate_query", self.generate_enhanced_query)
        workflow.add_node("validate_query", self.validate_query_with_server)
        workflow.add_node("execute_query", self.execute_enhanced_query)
        workflow.add_node("handle_error", self.handle_execution_error)
        workflow.add_node("format_response", self.format_enhanced_response)
        workflow.add_node("fallback_simple", self.fallback_to_simple_query)
        workflow.add_node("get_metrics", self.get_performance_metrics)
        
        # Set entry point
        workflow.set_entry_point("initialize")
        
        # Define the enhanced flow
        workflow.add_edge("initialize", "health_check")
        workflow.add_edge("health_check", "classify_question")
        workflow.add_edge("classify_question", "gather_enhanced_schema")
        
        # Conditional: get sample data for complex questions
        workflow.add_conditional_edges(
            "gather_enhanced_schema",
            self.should_get_sample_data,
            {
                "get_samples": "get_sample_data",
                "skip_samples": "generate_query"
            }
        )
        
        workflow.add_edge("get_sample_data", "generate_query")
        workflow.add_edge("generate_query", "validate_query")
        
        # Enhanced validation routing
        workflow.add_conditional_edges(
            "validate_query",
            self.should_execute_after_validation,
            {
                "execute": "execute_query",
                "regenerate": "generate_query",
                "fallback": "fallback_simple"
            }
        )
        
        workflow.add_edge("execute_query", "get_metrics")
        workflow.add_edge("get_metrics", "format_response")
        
        # Enhanced error handling
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

    async def initialize_state(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Initialize the enhanced agent state."""
        state.update({
            "attempts": 0,
            "max_attempts": 3,
            "results": [],
            "error_messages": [],
            "schema_info": {},
            "final_answer": "",
            "cortex_attempts": 0,
            "validation_result": {},
            "performance_metrics": {},
            "sample_data": {}
        })
        
        print(f"🚀 Initialized enhanced agent for: {state['original_question']}")
        return state

    async def check_server_health(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Check the specialized MCP server health."""
        print("🏥 Checking specialized MCP server health...")
        
        health_result = await self.call_mcp_tool("health_check")
        
        if health_result.startswith("❌"):
            print(f"⚠️ Server health check failed: {health_result}")
            state["error_messages"].append(f"Server health issue: {health_result}")
        else:
            try:
                health_data = json.loads(health_result)
                print(f"✅ Server healthy: {health_data.get('status', 'unknown')}")
                state["performance_metrics"]["server_health"] = health_data
            except json.JSONDecodeError:
                print("✅ Server responding")
        
        return state

    async def classify_question_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Enhanced question classification with the new system."""
        question_type, complexity = self.classify_question(state["original_question"])
        
        state["question_type"] = question_type
        state["complexity_level"] = complexity
        
        print(f"🧠 Enhanced classification: {question_type} ({complexity} complexity)")
        return state

    async def gather_enhanced_schema(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Gather comprehensive schema information using the enhanced MCP server."""
        print("📊 Gathering enhanced schema information...")
        
        # Use the enhanced schema analysis tool
        schema_result = await self.call_mcp_tool("analyze_schema")
        
        if not schema_result.startswith("❌"):
            try:
                schema_data = json.loads(schema_result)
                state["schema_info"] = schema_data
                print(f"✅ Enhanced schema gathered: {schema_data.get('total_labels', 0)} labels, {schema_data.get('total_relationship_types', 0)} relationship types")
            except json.JSONDecodeError:
                print("⚠️ Schema analysis returned non-JSON, using fallback")
                # Fallback to database summary
                summary_result = await self.call_mcp_tool("database_summary")
                if not summary_result.startswith("❌"):
                    try:
                        state["schema_info"] = json.loads(summary_result)
                    except json.JSONDecodeError:
                        state["schema_info"] = {"error": "Schema gathering failed"}
        else:
            print(f"⚠️ Schema gathering failed: {schema_result}")
            state["error_messages"].append(f"Schema gathering failed: {schema_result}")
        
        return state

    def should_get_sample_data(self, state: EnhancedAgentState) -> str:
        """Decide whether to get sample data based on question complexity."""
        if state["complexity_level"] == "complex" or state["question_type"] in ["exploration", "schema"]:
            return "get_samples"
        else:
            return "skip_samples"

    async def get_sample_data(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Get sample data to better understand the database structure."""
        print("🔍 Getting sample data for better context...")
        
        # Get general samples
        sample_result = await self.call_mcp_tool("get_sample_data", {"limit": 3})
        
        if not sample_result.startswith("❌"):
            try:
                sample_data = json.loads(sample_result)
                state["sample_data"] = sample_data
                print(f"✅ Sample data gathered: {sample_data.get('count', 0)} samples")
            except json.JSONDecodeError:
                print("⚠️ Sample data returned non-JSON")
        else:
            print(f"⚠️ Sample data gathering failed: {sample_result}")
        
        return state

    async def generate_enhanced_query(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Generate Cypher query with enhanced context from the specialized server."""
        state["cortex_attempts"] += 1
        print(f"🤖 Generating enhanced Cypher query (attempt {state['cortex_attempts']})...")
        
        # Build comprehensive context
        context = self.build_enhanced_query_context(state)
        
        # Generate query with Cortex
        raw_cypher = self.generate_cypher_with_cortex(context)
        
        if raw_cypher.startswith("❌"):
            state["error_messages"].append(f"Cortex generation failed: {raw_cypher}")
            state["current_query"] = "MATCH (n) RETURN count(n) LIMIT 1"  # Fallback
        else:
            state["current_query"] = raw_cypher.strip()
        
        print(f"📝 Generated query: {state['current_query']}")
        return state

    def build_enhanced_query_context(self, state: EnhancedAgentState) -> str:
        """Build comprehensive context using all available information."""
        schema_info = state.get("schema_info", {})
        sample_data = state.get("sample_data", {})
        
        # Extract enhanced schema details
        labels = []
        relationships = []
        
        if "labels" in schema_info:
            labels = [item.get("label", "") for item in schema_info["labels"] if item.get("label")]
        
        if "relationships" in schema_info:
            relationships = [item.get("type", "") for item in schema_info["relationships"] if item.get("type")]
        
        # Build context with sample data if available
        context_parts = [
            "Generate a modern Neo4j 5.x Cypher query for the following request.",
            "",
            f"DATABASE SCHEMA:",
            f"- Available Node Labels: {', '.join(labels[:10]) if labels else 'Unknown'}",  # Limit to avoid long prompts
            f"- Available Relationships: {', '.join(relationships[:10]) if relationships else 'Unknown'}",
        ]
        
        # Add sample data context if available
        if sample_data and "samples" in sample_data:
            context_parts.extend([
                "",
                "SAMPLE DATA STRUCTURE:",
            ])
            for i, sample in enumerate(sample_data["samples"][:2], 1):  # Show max 2 samples
                if isinstance(sample, dict) and "n" in sample:
                    node = sample["n"]
                    if isinstance(node, dict):
                        node_labels = node.get("labels", [])
                        properties = node.get("properties", {})
                        prop_names = list(properties.keys())[:5]  # Show first 5 properties
                        context_parts.append(f"- Sample {i}: {node_labels[0] if node_labels else 'Node'} with properties: {', '.join(prop_names)}")
        
        context_parts.extend([
            "",
            f"QUESTION ANALYSIS:",
            f"- Type: {state.get('question_type', 'general')}",
            f"- Complexity: {state.get('complexity_level', 'simple')}",
            ""
        ])
        
        # Add previous errors for learning
        if state.get("error_messages"):
            context_parts.extend([
                "PREVIOUS ERRORS TO AVOID:",
                *[f"- {error}" for error in state["error_messages"][-2:]],  # Last 2 errors
                ""
            ])
        
        # Add enhanced syntax guidelines
        context_parts.extend([
            "CRITICAL SYNTAX REQUIREMENTS:",
            "- Use COUNT { (n)-[]-() } instead of size((n)-[]->())",
            "- Use property IS NOT NULL instead of has(property)",
            "- Always include LIMIT for ORDER BY queries (suggest LIMIT 10-100)",
            "- Use proper Neo4j 5.x syntax only",
            "- For connectivity queries, return both node info and connection counts",
            "",
            f"USER QUESTION: {state['original_question']}",
            "",
            "Return ONLY the Cypher query, no explanations."
        ])
        
        return "\n".join(context_parts)

    async def validate_query_with_server(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Validate query using the specialized server's validation tool."""
        print("🔍 Validating query with enhanced server validation...")
        
        validation_result = await self.call_mcp_tool("validate_query", {
            "query": state["current_query"],
            "apply_fixes": True
        })
        
        if not validation_result.startswith("❌"):
            try:
                validation_data = json.loads(validation_result)
                state["validation_result"] = validation_data
                
                # Use suggested query if available
                if "suggested_query" in validation_data and validation_data["suggested_query"]:
                    old_query = state["current_query"]
                    state["current_query"] = validation_data["suggested_query"]
                    print(f"🔧 Applied server-suggested query improvements")
                
                issues = validation_data.get("issues", [])
                if issues:
                    print(f"⚠️ Validation found {len(issues)} issues: {', '.join(issues[:2])}")
                else:
                    print("✅ Query validation passed")
                    
            except json.JSONDecodeError:
                print("⚠️ Validation returned non-JSON")
        else:
            print(f"⚠️ Query validation failed: {validation_result}")
            state["error_messages"].append(f"Validation failed: {validation_result}")
        
        return state

    def should_execute_after_validation(self, state: EnhancedAgentState) -> str:
        """Enhanced decision making after validation."""
        validation_result = state.get("validation_result", {})
        
        # If validation found critical issues and we haven't tried many times
        if validation_result.get("issues") and state["cortex_attempts"] < 2:
            critical_issues = [issue for issue in validation_result.get("issues", []) if "deprecated" in issue.lower() or "syntax" in issue.lower()]
            if critical_issues:
                return "regenerate"
        
        # If we've tried too many times, fallback
        if state["attempts"] >= state["max_attempts"]:
            return "fallback"
        
        # Otherwise, execute
        return "execute"

    async def execute_enhanced_query(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Execute query using the enhanced MCP server."""
        state["attempts"] += 1
        print(f"⚡ Executing enhanced query (attempt {state['attempts']})...")
        
        query = state["current_query"]
        
        # Determine which enhanced tool to use
        is_write = any(keyword in query.upper() for keyword in ["CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP"])
        tool_name = "execute_write_query" if is_write else "execute_read_query"
        
        # Execute with the enhanced server
        result = await self.call_mcp_tool(tool_name, {
            "query": query,
            "apply_fixes": True  # Let the server apply additional fixes
        })
        
        if result.startswith("❌"):
            state["error_messages"].append(result)
            print(f"❌ Enhanced query execution failed: {result}")
        else:
            try:
                parsed_result = json.loads(result)
                
                # Enhanced result handling
                if "data" in parsed_result and "metadata" in parsed_result:
                    # New enhanced format
                    state["results"].append(parsed_result["data"])
                    state["performance_metrics"]["last_execution"] = parsed_result["metadata"]
                    print(f"✅ Enhanced execution successful: {parsed_result['metadata'].get('record_count', 'unknown')} records in {parsed_result['metadata'].get('execution_time_ms', 'unknown')}ms")
                else:
                    # Legacy format
                    state["results"].append(parsed_result)
                    print(f"✅ Query executed successfully")
                    
            except json.JSONDecodeError:
                # Handle non-JSON results
                state["results"].append({"raw_result": result})
                print(f"✅ Query executed, got raw result")
        
        return state

    async def get_performance_metrics(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Get performance metrics from the enhanced server."""
        print("📊 Gathering performance metrics...")
        
        metrics_result = await self.call_mcp_tool("get_metrics")
        
        if not metrics_result.startswith("❌"):
            try:
                metrics_data = json.loads(metrics_result)
                state["performance_metrics"]["server_metrics"] = metrics_data
                print(f"✅ Performance metrics gathered: {metrics_data.get('success_rate', 'unknown')}% success rate")
            except json.JSONDecodeError:
                print("⚠️ Metrics returned non-JSON")
        
        return state

    async def handle_execution_error(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Enhanced error handling with server insights."""
        if state["error_messages"]:
            last_error = state["error_messages"][-1]
            print(f"🛠️ Enhanced error handling: {last_error[:100]}...")
            
            # Enhanced error analysis
            if "syntax" in last_error.lower():
                print("🔧 Syntax error detected - will use server validation for fixes")
            elif "timeout" in last_error.lower():
                print("🔧 Timeout detected - will optimize query complexity")
            elif "deprecated" in last_error.lower():
                print("🔧 Deprecated syntax detected - will modernize query")
        
        return state

    def check_execution_success(self, state: EnhancedAgentState) -> str:
        """Enhanced success checking."""
        if state["results"] and not state["error_messages"]:
            return "success"
        elif state["attempts"] < state["max_attempts"] and state["cortex_attempts"] < 3:
            return "retry"
        else:
            return "fallback"

    async def fallback_to_simple_query(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Enhanced fallback using server tools."""
        print("🔄 Using enhanced fallback approach...")
        
        question_type = state.get("question_type", "general")
        
        # Enhanced fallback queries
        fallback_queries = {
            "connectivity": "MATCH (n) RETURN n, COUNT { (n)-[]-() } as connections ORDER BY connections DESC LIMIT 10",
            "aggregation": "CALL count_by_label() YIELD label, count RETURN label, count ORDER BY count DESC",
            "schema": "CALL analyze_schema() YIELD *",
            "exploration": "CALL get_sample_data({limit: 10}) YIELD *",
            "general": "CALL database_summary() YIELD *"
        }
        
        fallback_query = fallback_queries.get(question_type, "MATCH (n) RETURN count(n) as node_count")
        
        # Try to use enhanced server tools instead of raw queries
        if question_type == "aggregation":
            result = await self.call_mcp_tool("count_by_label")
        elif question_type == "schema":
            result = await self.call_mcp_tool("analyze_schema")
        elif question_type == "exploration":
            result = await self.call_mcp_tool("get_sample_data", {"limit": 10})
        else:
            # Execute simple query
            result = await self.call_mcp_tool("execute_read_query", {"query": fallback_query})
        
        if not result.startswith("❌"):
            try:
                parsed_result = json.loads(result)
                state["results"] = [parsed_result]
                state["final_answer"] = f"🔄 **Enhanced Fallback Result:** Used specialized server tools due to query complexity.\n\nApproach: {question_type} analysis"
            except json.JSONDecodeError:
                state["results"] = [{"raw_result": result}]
        
        return state

    async def format_enhanced_response(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Enhanced response formatting with performance metrics."""
        if not state["results"]:
            performance_info = ""
            if state.get("performance_metrics"):
                server_metrics = state["performance_metrics"].get("server_metrics", {})
                performance_info = f"\n\n📊 **Server Performance:** {server_metrics.get('success_rate', 'unknown')}% success rate, {server_metrics.get('total_queries', 0)} total queries processed"
            
            state["final_answer"] = f"❌ Unable to answer '{state['original_question']}' after {state['attempts']} attempts.{performance_info}"
            return state
        
        question_type = state.get("question_type", "general")
        results = state["results"]
        
        # Enhanced formatting based on question type
        if question_type == "connectivity":
            state["final_answer"] = self.format_connectivity_results_enhanced(results, state)
        elif question_type == "aggregation":
            state["final_answer"] = self.format_aggregation_results_enhanced(results, state)
        elif question_type == "schema":
            state["final_answer"] = self.format_schema_results_enhanced(results, state)
        elif question_type == "exploration":
            state["final_answer"] = self.format_exploration_results_enhanced(results, state)
        else:
            state["final_answer"] = self.format_general_results_enhanced(results, state)
        
        # Add performance metrics if available
        self.add_performance_info(state)
        
        return state

    def format_connectivity_results_enhanced(self, results: List[Dict], state: EnhancedAgentState) -> str:
        """Enhanced formatting for connectivity questions."""
        if not results or not results[0]:
            return "No connectivity data found."
        
        data = results[0]
        response = "🔗 **Network Connectivity Analysis:**\n\n"
        
        # Handle enhanced result format
        if isinstance(data, dict) and "data" in data:
            actual_data = data["data"]
            metadata = data.get("metadata", {})
            response += f"*Query executed in {metadata.get('execution_time_ms', 'unknown')}ms*\n\n"
        else:
            actual_data = data
        
        if not actual_data:
            return "No nodes found in the database."
        
        # Process connectivity data
        for i, record in enumerate(actual_data[:10], 1):
            if isinstance(record, dict):
                node_info = record.get('n', {})
                connections = record.get('connections', record.get('degree', 0))
                
                if isinstance(node_info, dict):
                    labels = node_info.get('labels', ['Unknown'])
                    properties = node_info.get('properties', {})
                    name = properties.get('name', properties.get('title', f"Node {i}"))
                else:
                    labels = ['Node']
                    name = f"Item {i}"
                
                response += f"{i}. **{name}** ({labels[0] if labels else 'Node'})\n"
                response += f"   🔗 Connections: **{connections}**\n\n"
        
        return response

    def format_aggregation_results_enhanced(self, results: List[Dict], state: EnhancedAgentState) -> str:
        """Enhanced formatting for aggregation questions."""
        if not results or not results[0]:
            return "No aggregation data available."
        
        data = results[0]
        response = "📊 **Database Analytics:**\n\n"
        
        # Handle different result formats
        if "label_counts" in data:
            # Count by label format
            response += "**Node Distribution by Label:**\n\n"
            for item in data["label_counts"][:10]:
                label = item.get("label", "Unknown")
                count = item.get("count", 0)
                response += f"• **{label}:** {count:,} nodes\n"
        elif isinstance(data, list):
            # Regular aggregation results
            for record in data[:5]:
                if isinstance(record, dict):
                    for key, value in record.items():
                        response += f"• **{key.replace('_', ' ').title()}:** {value:,}\n"
        else:
            # Single result
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    response += f"• **{key.replace('_', ' ').title()}:** {value:,}\n"
        
        return response

    def format_schema_results_enhanced(self, results: List[Dict], state: EnhancedAgentState) -> str:
        """Enhanced formatting for schema questions."""
        if not results or not results[0]:
            return "No schema information available."
        
        data = results[0]
        response = "🏗️ **Database Schema Analysis:**\n\n"
        
        # Handle enhanced schema format
        if "labels" in data and isinstance(data["labels"], list):
            response += f"**Node Labels ({len(data['labels'])}):**\n"
            for item in data["labels"][:10]:
                if isinstance(item, dict):
                    label = item.get("label", "Unknown")
                    count = item.get("count", "unknown")
                    response += f"• **{label}** ({count} nodes)\n"
            response += "\n"
        
        if "relationships" in data and isinstance(data["relationships"], list):
            response += f"**Relationship Types ({len(data['relationships'])}):**\n"
            for item in data["relationships"][:10]:
                if isinstance(item, dict):
                    rel_type = item.get("type", "Unknown")
                    count = item.get("count", "unknown")
                    response += f"• **{rel_type}** ({count} relationships)\n"
        
        return response

    def format_exploration_results_enhanced(self, results: List[Dict], state: EnhancedAgentState) -> str:
        """Enhanced formatting for exploration questions."""
        if not results or not results[0]:
            return "No data found to explore."
        
        data = results[0]
        
        # Handle sample data format
        if "samples" in data:
            samples = data["samples"]
            response = f"🔍 **Data Exploration ({len(samples)} samples):**\n\n"
            
            for i, sample in enumerate(samples[:5], 1):
                if isinstance(sample, dict) and "n" in sample:
                    node = sample["n"]
                    if isinstance(node, dict):
                        labels = node.get("labels", ["Unknown"])
                        properties = node.get("properties", {})
                        
                        response += f"{i}. **{labels[0]}**\n"
                        
                        # Show key properties
                        key_props = ["name", "title", "id", "type"]
                        shown_props = []
                        for prop in key_props:
                            if prop in properties:
                                shown_props.append(f"{prop}: {properties[prop]}")
                        
                        if shown_props:
                            response += f"   Properties: {', '.join(shown_props)}\n"
                        response += "\n"
        else:
            # Regular exploration format
            response = f"🔍 **Found {len(data) if isinstance(data, list) else 1} items:**\n\n"
            
            items_to_show = data[:5] if isinstance(data, list) else [data]
            for i, record in enumerate(items_to_show, 1):
                response += f"{i}. "
                if isinstance(record, dict):
                    # Show first few properties
                    shown_props = list(record.items())[:3]
                    prop_strs = [f"{k}: {v}" for k, v in shown_props]
                    response += ", ".join(prop_strs)
                response += "\n"
        
        return response

    def format_general_results_enhanced(self, results: List[Dict], state: EnhancedAgentState) -> str:
        """Enhanced formatting for general results."""
        if not results:
            return "No results found."
        
        data = results[0]
        response = "📋 **Query Results:**\n\n"
        
        # Handle enhanced result format
        if isinstance(data, dict) and "data" in data:
            actual_data = data["data"]
            metadata = data.get("metadata", {})
            response += f"*Executed in {metadata.get('execution_time_ms', 'unknown')}ms, returned {metadata.get('record_count', 'unknown')} records*\n\n"
            display_data = actual_data
        else:
            display_data = data
        
        # Limit displayed results for readability
        if isinstance(display_data, list):
            display_data = display_data[:10]
        
        response += f"```json\n{json.dumps(display_data, indent=2)}\n```"
        
        return response

    def add_performance_info(self, state: EnhancedAgentState):
        """Add performance metrics to the final answer."""
        performance_metrics = state.get("performance_metrics", {})
        
        if performance_metrics:
            perf_info = "\n\n📊 **Performance Metrics:**\n"
            
            # Last execution metrics
            if "last_execution" in performance_metrics:
                exec_metrics = performance_metrics["last_execution"]
                perf_info += f"• Execution time: {exec_metrics.get('execution_time_ms', 'unknown')}ms\n"
                if exec_metrics.get("syntax_fixes_applied"):
                    perf_info += f"• ✅ Automatic syntax fixes applied\n"
            
            # Server metrics
            if "server_metrics" in performance_metrics:
                server_metrics = performance_metrics["server_metrics"]
                perf_info += f"• Server success rate: {server_metrics.get('success_rate', 'unknown')}%\n"
                perf_info += f"• Total queries processed: {server_metrics.get('total_queries', 0)}\n"
            
            state["final_answer"] += perf_info

    async def run(self, question: str) -> str:
        """Run the complete enhanced agent workflow."""
        initial_state = EnhancedAgentState(
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
            cortex_attempts=0,
            validation_result={},
            performance_metrics={},
            sample_data={}
        )
        
        graph = self.create_graph()
        
        try:
            final_state = await graph.ainvoke(initial_state)
            return final_state["final_answer"]
        except Exception as e:
            return f"❌ Enhanced agent error: {str(e)}"

# Test and demonstration
async def test_enhanced_agent():
    """Test the enhanced agent with the specialized MCP server."""
    agent = OptimizedNeo4jAgent()
    
    test_questions = [
        "show me nodes with most connected nodes in the database?",  # Your original failing question
        "what properties does the user node have?",
        "how many nodes are in the database?",
        "give me a comprehensive database analysis",
        "find interesting patterns in the data",
        "what's the performance of this server?",
    ]
    
    print("🧪 Testing Enhanced Neo4j LangGraph Agent with Specialized MCP Server")
    print("=" * 80)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🤔 **Test {i}:** {question}")
        print("-" * 60)
        
        answer = await agent.run(question)
        print(f"\n🎯 **Answer:**\n{answer}\n")
        print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_enhanced_agent())
