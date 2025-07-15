# test_system.py - Comprehensive Testing System for Intelligent Neo4j Assistant

import asyncio
import json
import logging
import time
import sys
import traceback
import random
import string
from typing import Dict, Any, List, Tuple, Optional, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Import system components
try:
    from fastmcp import Client
    from neo4j import AsyncGraphDatabase
    import nest_asyncio
    from config import Config
    from updated_langgraph_agent import OptimizedNeo4jAgent
    import pandas as pd
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

# Apply nest_asyncio for Jupyter/nested event loops
nest_asyncio.apply()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestStatus(Enum):
    """Test result status."""
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"

@dataclass
class TestResult:
    """Represents the result of a single test."""
    name: str
    status: TestStatus
    execution_time: float
    details: Dict[str, Any]
    error: Optional[str] = None
    expected: Any = None
    actual: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "execution_time_ms": round(self.execution_time * 1000, 2),
            "details": self.details,
            "error": self.error,
            "expected": self.expected,
            "actual": self.actual,
            "timestamp": datetime.now().isoformat()
        }
    
    def __str__(self) -> str:
        status_icon = {
            TestStatus.PASS: "✅",
            TestStatus.FAIL: "❌", 
            TestStatus.SKIP: "⏭️",
            TestStatus.ERROR: "💥"
        }
        return f"{status_icon[self.status]} {self.name} ({self.execution_time:.3f}s)"

class TestSuite:
    """Base class for test suites."""
    
    def __init__(self, name: str):
        self.name = name
        self.tests: List[TestResult] = []
        self.setup_completed = False
        
    async def setup(self):
        """Setup before running tests."""
        pass
    
    async def teardown(self):
        """Cleanup after running tests."""
        pass
    
    async def run_test(self, test_func: Callable, test_name: str) -> TestResult:
        """Run a single test function."""
        start_time = time.time()
        
        try:
            result = await test_func()
            execution_time = time.time() - start_time
            
            if isinstance(result, TestResult):
                result.execution_time = execution_time
                return result
            else:
                # Assume success if test returns without exception
                return TestResult(
                    name=test_name,
                    status=TestStatus.PASS,
                    execution_time=execution_time,
                    details=result if isinstance(result, dict) else {"result": result}
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Test {test_name} failed with exception: {e}")
            return TestResult(
                name=test_name,
                status=TestStatus.ERROR,
                execution_time=execution_time,
                details={"exception": str(e), "traceback": traceback.format_exc()},
                error=str(e)
            )
    
    async def run_all_tests(self) -> List[TestResult]:
        """Run all tests in the suite."""
        logger.info(f"🧪 Running test suite: {self.name}")
        
        # Setup
        if not self.setup_completed:
            await self.setup()
            self.setup_completed = True
        
        # Get all test methods
        test_methods = [
            (name, getattr(self, name)) 
            for name in dir(self) 
            if name.startswith('test_') and callable(getattr(self, name))
        ]
        
        results = []
        for test_name, test_method in test_methods:
            logger.info(f"Running {test_name}...")
            result = await self.run_test(test_method, test_name)
            results.append(result)
            self.tests.append(result)
            
            # Log result
            status_text = result.status.value
            logger.info(f"  {test_name}: {status_text} ({result.execution_time:.3f}s)")
        
        # Teardown
        await self.teardown()
        
        return results

class MCPServerTestSuite(TestSuite):
    """Test suite for MCP Server functionality."""
    
    def __init__(self):
        super().__init__("MCP Server Tests")
        self.config = Config()
        
    async def setup(self):
        """Setup MCP server tests."""
        # Verify MCP server script exists
        script_path = Path(self.config.MCP_SERVER_SCRIPT)
        if not script_path.exists():
            raise FileNotFoundError(f"MCP server script not found: {self.config.MCP_SERVER_SCRIPT}")
    
    async def test_mcp_server_connection(self) -> TestResult:
        """Test basic MCP server connection."""
        try:
            async with Client(self.config.MCP_SERVER_SCRIPT) as client:
                # Test that we can connect
                details = {"connection_successful": True}
                return TestResult(
                    name="MCP Server Connection",
                    status=TestStatus.PASS,
                    execution_time=0,
                    details=details
                )
        except Exception as e:
            return TestResult(
                name="MCP Server Connection",
                status=TestStatus.FAIL,
                execution_time=0,
                details={"connection_successful": False},
                error=str(e)
            )
    
    async def test_health_check_tool(self) -> TestResult:
        """Test MCP server health check tool."""
        async with Client(self.config.MCP_SERVER_SCRIPT) as client:
            result = await client.call_tool("health_check")
            
            if hasattr(result, 'content') and result.content:
                content = result.content[0].text
                health_data = json.loads(content)
                
                expected_fields = ["status", "database", "message"]
                has_expected = all(field in health_data for field in expected_fields)
                is_healthy = health_data.get("status") == "healthy"
                
                status = TestStatus.PASS if (has_expected and is_healthy) else TestStatus.FAIL
                
                return TestResult(
                    name="Health Check Tool",
                    status=status,
                    execution_time=0,
                    details={
                        "has_expected_fields": has_expected,
                        "is_healthy": is_healthy,
                        "health_data": health_data
                    },
                    expected="healthy status with required fields",
                    actual=health_data
                )
            else:
                return TestResult(
                    name="Health Check Tool",
                    status=TestStatus.FAIL,
                    execution_time=0,
                    details={"has_content": False},
                    error="No content in health check response"
                )
    
    async def test_database_summary_tool(self) -> TestResult:
        """Test database summary tool."""
        async with Client(self.config.MCP_SERVER_SCRIPT) as client:
            result = await client.call_tool("database_summary")
            
            if hasattr(result, 'content') and result.content:
                content = result.content[0].text
                summary_data = json.loads(content)
                
                expected_fields = ["database", "node_count", "relationship_count", "labels"]
                has_expected = all(field in summary_data for field in expected_fields)
                has_reasonable_data = (
                    isinstance(summary_data.get("node_count"), int) and
                    isinstance(summary_data.get("relationship_count"), int) and
                    isinstance(summary_data.get("labels"), list)
                )
                
                status = TestStatus.PASS if (has_expected and has_reasonable_data) else TestStatus.FAIL
                
                return TestResult(
                    name="Database Summary Tool",
                    status=status,
                    execution_time=0,
                    details={
                        "has_expected_fields": has_expected,
                        "has_reasonable_data": has_reasonable_data,
                        "node_count": summary_data.get("node_count"),
                        "relationship_count": summary_data.get("relationship_count"),
                        "label_count": len(summary_data.get("labels", []))
                    }
                )
            else:
                return TestResult(
                    name="Database Summary Tool",
                    status=TestStatus.FAIL,
                    execution_time=0,
                    details={"has_content": False},
                    error="No content in database summary response"
                )
    
    async def test_query_validation_tool(self) -> TestResult:
        """Test query validation tool with deprecated syntax."""
        deprecated_query = "MATCH (n) RETURN size((n)-[]->()) as degree"
        
        async with Client(self.config.MCP_SERVER_SCRIPT) as client:
            result = await client.call_tool("validate_query", {
                "query": deprecated_query,
                "apply_fixes": True
            })
            
            if hasattr(result, 'content') and result.content:
                content = result.content[0].text
                validation_data = json.loads(content)
                
                # Check if validation detected the deprecated syntax
                issues = validation_data.get("issues", [])
                has_size_issue = any("size" in issue.lower() for issue in issues)
                has_suggestions = len(validation_data.get("suggestions", [])) > 0
                has_suggested_query = "suggested_query" in validation_data
                
                status = TestStatus.PASS if (has_size_issue and (has_suggestions or has_suggested_query)) else TestStatus.FAIL
                
                return TestResult(
                    name="Query Validation Tool",
                    status=status,
                    execution_time=0,
                    details={
                        "deprecated_query": deprecated_query,
                        "issues_detected": len(issues),
                        "has_size_issue": has_size_issue,
                        "has_suggestions": has_suggestions,
                        "has_suggested_query": has_suggested_query,
                        "issues": issues,
                        "suggested_query": validation_data.get("suggested_query")
                    },
                    expected="Detection of deprecated size() syntax with suggestions",
                    actual=f"Issues: {issues}, Suggestions: {validation_data.get('suggestions', [])}"
                )
            else:
                return TestResult(
                    name="Query Validation Tool",
                    status=TestStatus.FAIL,
                    execution_time=0,
                    details={"has_content": False},
                    error="No content in validation response"
                )
    
    async def test_enhanced_query_execution(self) -> TestResult:
        """Test enhanced query execution with automatic fixes."""
        # Test with a query that needs fixing
        problematic_query = "MATCH (n) RETURN size((n)-[]->()) as degree LIMIT 5"
        
        async with Client(self.config.MCP_SERVER_SCRIPT) as client:
            result = await client.call_tool("execute_read_query", {
                "query": problematic_query,
                "apply_fixes": True
            })
            
            if hasattr(result, 'content') and result.content:
                content = result.content[0].text
                
                try:
                    execution_data = json.loads(content)
                    
                    # Check if query executed successfully despite deprecated syntax
                    has_data = "data" in execution_data or isinstance(execution_data, list)
                    has_metadata = "metadata" in execution_data
                    syntax_fixes_applied = False
                    
                    if has_metadata:
                        metadata = execution_data.get("metadata", {})
                        syntax_fixes_applied = metadata.get("syntax_fixes_applied", False)
                    
                    status = TestStatus.PASS if has_data else TestStatus.FAIL
                    
                    return TestResult(
                        name="Enhanced Query Execution",
                        status=status,
                        execution_time=0,
                        details={
                            "original_query": problematic_query,
                            "has_data": has_data,
                            "has_metadata": has_metadata,
                            "syntax_fixes_applied": syntax_fixes_applied,
                            "execution_successful": has_data
                        },
                        expected="Successful execution with automatic syntax fixes",
                        actual="Success" if has_data else "Failure"
                    )
                    
                except json.JSONDecodeError:
                    # Query might have executed but returned non-JSON
                    return TestResult(
                        name="Enhanced Query Execution",
                        status=TestStatus.FAIL,
                        execution_time=0,
                        details={"json_decode_error": True, "raw_content": content[:200]},
                        error="Response is not valid JSON"
                    )
            else:
                return TestResult(
                    name="Enhanced Query Execution",
                    status=TestStatus.FAIL,
                    execution_time=0,
                    details={"has_content": False},
                    error="No content in execution response"
                )
    
    async def test_performance_metrics(self) -> TestResult:
        """Test performance metrics collection."""
        async with Client(self.config.MCP_SERVER_SCRIPT) as client:
            # Reset metrics first
            await client.call_tool("reset_metrics")
            
            # Execute a few queries to generate metrics
            for i in range(3):
                await client.call_tool("execute_read_query", {
                    "query": f"MATCH (n) RETURN count(n) as count LIMIT 1"
                })
            
            # Get metrics
            result = await client.call_tool("get_metrics")
            
            if hasattr(result, 'content') and result.content:
                content = result.content[0].text
                metrics_data = json.loads(content)
                
                expected_fields = ["total_queries", "successful_queries", "success_rate"]
                has_expected = any(field in metrics_data for field in expected_fields)
                has_query_data = metrics_data.get("total_queries", 0) > 0
                
                status = TestStatus.PASS if (has_expected and has_query_data) else TestStatus.FAIL
                
                return TestResult(
                    name="Performance Metrics",
                    status=status,
                    execution_time=0,
                    details={
                        "has_expected_fields": has_expected,
                        "has_query_data": has_query_data,
                        "metrics": metrics_data
                    }
                )
            else:
                return TestResult(
                    name="Performance Metrics",
                    status=TestStatus.FAIL,
                    execution_time=0,
                    details={"has_content": False},
                    error="No content in metrics response"
                )

class LangGraphAgentTestSuite(TestSuite):
    """Test suite for LangGraph Agent functionality."""
    
    def __init__(self):
        super().__init__("LangGraph Agent Tests")
        self.config = Config()
        self.agent = None
        
    async def setup(self):
        """Setup LangGraph agent tests."""
        self.agent = OptimizedNeo4jAgent(self.config.MCP_SERVER_SCRIPT)
    
    async def test_agent_initialization(self) -> TestResult:
        """Test agent initialization."""
        try:
            agent = OptimizedNeo4jAgent(self.config.MCP_SERVER_SCRIPT)
            
            # Check agent attributes
            has_mcp_script = hasattr(agent, 'mcp_script_path')
            has_cortex_config = hasattr(agent, 'cortex_config')
            has_question_patterns = hasattr(agent, 'question_patterns')
            
            initialization_successful = has_mcp_script and has_cortex_config and has_question_patterns
            
            status = TestStatus.PASS if initialization_successful else TestStatus.FAIL
            
            return TestResult(
                name="Agent Initialization",
                status=status,
                execution_time=0,
                details={
                    "has_mcp_script": has_mcp_script,
                    "has_cortex_config": has_cortex_config,
                    "has_question_patterns": has_question_patterns,
                    "initialization_successful": initialization_successful
                }
            )
            
        except Exception as e:
            return TestResult(
                name="Agent Initialization",
                status=TestStatus.FAIL,
                execution_time=0,
                details={"initialization_failed": True},
                error=str(e)
            )
    
    async def test_question_classification(self) -> TestResult:
        """Test question classification functionality."""
        test_questions = [
            ("show me nodes with most connected nodes", "connectivity", "complex"),
            ("how many nodes are there", "aggregation", "simple"),
            ("what are the node labels", "schema", "simple"),
            ("find interesting patterns", "exploration", "complex")
        ]
        
        classification_results = []
        all_correct = True
        
        for question, expected_type, expected_complexity in test_questions:
            question_type, complexity = self.agent.classify_question(question)
            
            type_correct = question_type == expected_type
            complexity_reasonable = complexity in ["simple", "medium", "complex"]
            
            classification_results.append({
                "question": question,
                "expected_type": expected_type,
                "actual_type": question_type,
                "expected_complexity": expected_complexity,
                "actual_complexity": complexity,
                "type_correct": type_correct,
                "complexity_reasonable": complexity_reasonable
            })
            
            if not (type_correct and complexity_reasonable):
                all_correct = False
        
        status = TestStatus.PASS if all_correct else TestStatus.FAIL
        
        return TestResult(
            name="Question Classification",
            status=status,
            execution_time=0,
            details={
                "all_classifications_correct": all_correct,
                "test_results": classification_results,
                "total_tests": len(test_questions)
            }
        )
    
    async def test_mcp_tool_communication(self) -> TestResult:
        """Test agent's communication with MCP tools."""
        # Test basic tool call
        health_result = await self.agent.call_mcp_tool("health_check")
        
        if health_result and not health_result.startswith("❌"):
            try:
                health_data = json.loads(health_result)
                is_healthy = health_data.get("status") == "healthy"
                
                status = TestStatus.PASS if is_healthy else TestStatus.FAIL
                
                return TestResult(
                    name="MCP Tool Communication",
                    status=status,
                    execution_time=0,
                    details={
                        "tool_call_successful": True,
                        "response_parseable": True,
                        "health_status": health_data.get("status"),
                        "is_healthy": is_healthy
                    }
                )
            except json.JSONDecodeError:
                return TestResult(
                    name="MCP Tool Communication",
                    status=TestStatus.FAIL,
                    execution_time=0,
                    details={
                        "tool_call_successful": True,
                        "response_parseable": False
                    },
                    error="Response is not valid JSON"
                )
        else:
            return TestResult(
                name="MCP Tool Communication",
                status=TestStatus.FAIL,
                execution_time=0,
                details={"tool_call_successful": False},
                error=health_result if health_result else "No response"
            )
    
    async def test_simple_query_workflow(self) -> TestResult:
        """Test simple query workflow."""
        simple_question = "how many nodes are in the database?"
        
        answer = await self.agent.run(simple_question)
        
        success = not answer.startswith("❌")
        has_reasonable_length = len(answer) > 20
        has_number = any(char.isdigit() for char in answer)
        
        workflow_successful = success and has_reasonable_length and has_number
        
        status = TestStatus.PASS if workflow_successful else TestStatus.FAIL
        
        return TestResult(
            name="Simple Query Workflow",
            status=status,
            execution_time=0,
            details={
                "question": simple_question,
                "answer_length": len(answer),
                "success": success,
                "has_reasonable_length": has_reasonable_length,
                "has_number": has_number,
                "workflow_successful": workflow_successful,
                "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer
            }
        )
    
    async def test_complex_query_workflow(self) -> TestResult:
        """Test complex query workflow with the originally failing query."""
        complex_question = "show me nodes with most connected nodes in the database?"
        
        answer = await self.agent.run(complex_question)
        
        success = not answer.startswith("❌")
        has_connectivity_terms = any(term in answer.lower() for term in ["connect", "degree", "relationship"])
        has_formatting = any(marker in answer for marker in ["🔗", "**", "#", "•"])
        has_reasonable_length = len(answer) > 100
        
        workflow_successful = success and has_connectivity_terms and has_reasonable_length
        
        status = TestStatus.PASS if workflow_successful else TestStatus.FAIL
        
        return TestResult(
            name="Complex Query Workflow (Original Failing Query)",
            status=status,
            execution_time=0,
            details={
                "question": complex_question,
                "answer_length": len(answer),
                "success": success,
                "has_connectivity_terms": has_connectivity_terms,
                "has_formatting": has_formatting,
                "has_reasonable_length": has_reasonable_length,
                "workflow_successful": workflow_successful,
                "answer_preview": answer[:300] + "..." if len(answer) > 300 else answer
            },
            expected="Successful connectivity analysis with formatted output",
            actual="Success with formatting" if workflow_successful else "Failed or poor formatting"
        )

class IntegrationTestSuite(TestSuite):
    """Integration tests for the complete system."""
    
    def __init__(self):
        super().__init__("Integration Tests")
        self.config = Config()
        self.agent = None
        
    async def setup(self):
        """Setup integration tests."""
        self.agent = OptimizedNeo4jAgent(self.config.MCP_SERVER_SCRIPT)
    
    async def test_end_to_end_originally_failing_query(self) -> TestResult:
        """Test the complete end-to-end workflow with the originally failing query."""
        original_question = "show me nodes with most connected nodes in the database?"
        
        # This is the key test - the query that was originally failing
        start_time = time.time()
        answer = await self.agent.run(original_question)
        execution_time = time.time() - start_time
        
        # Analyze the result comprehensively
        success = not answer.startswith("❌")
        has_connectivity_data = "connect" in answer.lower() and ("node" in answer.lower() or "degree" in answer.lower())
        has_intelligent_formatting = any(marker in answer for marker in ["🔗", "**", "#", "•", "Network", "Analysis"])
        has_reasonable_length = len(answer) > 100
        has_performance_info = "ms" in answer or "time" in answer.lower()
        includes_query_info = "query" in answer.lower() or "cypher" in answer.lower()
        
        # Overall success criteria
        comprehensive_success = (
            success and 
            has_connectivity_data and 
            has_intelligent_formatting and 
            has_reasonable_length
        )
        
        status = TestStatus.PASS if comprehensive_success else TestStatus.FAIL
        
        return TestResult(
            name="End-to-End Originally Failing Query",
            status=status,
            execution_time=execution_time,
            details={
                "original_question": original_question,
                "execution_time_seconds": execution_time,
                "answer_length": len(answer),
                "success": success,
                "has_connectivity_data": has_connectivity_data,
                "has_intelligent_formatting": has_intelligent_formatting,
                "has_reasonable_length": has_reasonable_length,
                "has_performance_info": has_performance_info,
                "includes_query_info": includes_query_info,
                "comprehensive_success": comprehensive_success,
                "answer_preview": answer[:400] + "..." if len(answer) > 400 else answer
            },
            expected="Successful connectivity analysis with intelligent formatting",
            actual="Success" if comprehensive_success else "Failed or insufficient quality"
        )
    
    async def test_multiple_query_types(self) -> TestResult:
        """Test multiple different query types to ensure versatility."""
        test_queries = [
            ("how many nodes are in the database?", "aggregation"),
            ("what are the node types?", "schema"),
            ("show me sample data", "exploration"),
            ("analyze the database structure", "schema")
        ]
        
        results = []
        all_successful = True
        
        for question, expected_type in test_queries:
            try:
                answer = await self.agent.run(question)
                success = not answer.startswith("❌")
                has_content = len(answer) > 20
                
                query_successful = success and has_content
                if not query_successful:
                    all_successful = False
                
                results.append({
                    "question": question,
                    "expected_type": expected_type,
                    "success": success,
                    "has_content": has_content,
                    "answer_length": len(answer),
                    "answer_preview": answer[:100] + "..." if len(answer) > 100 else answer
                })
                
            except Exception as e:
                all_successful = False
                results.append({
                    "question": question,
                    "expected_type": expected_type,
                    "success": False,
                    "error": str(e)
                })
        
        status = TestStatus.PASS if all_successful else TestStatus.FAIL
        
        return TestResult(
            name="Multiple Query Types",
            status=status,
            execution_time=0,
            details={
                "all_successful": all_successful,
                "total_queries": len(test_queries),
                "successful_queries": sum(1 for r in results if r.get("success", False)),
                "query_results": results
            }
        )
    
    async def test_error_recovery(self) -> TestResult:
        """Test error recovery and handling."""
        # Test with an intentionally problematic query
        problematic_question = "execute this broken cypher: MATCH (n) RETURN invalid_function(n)"
        
        answer = await self.agent.run(problematic_question)
        
        # The system should handle this gracefully
        handles_gracefully = not ("Exception" in answer and "Traceback" in answer)
        provides_feedback = len(answer) > 20
        not_completely_broken = not answer.startswith("❌ Enhanced agent error:")
        
        error_recovery_successful = handles_gracefully and provides_feedback
        
        status = TestStatus.PASS if error_recovery_successful else TestStatus.FAIL
        
        return TestResult(
            name="Error Recovery",
            status=status,
            execution_time=0,
            details={
                "problematic_question": problematic_question,
                "handles_gracefully": handles_gracefully,
                "provides_feedback": provides_feedback,
                "not_completely_broken": not_completely_broken,
                "error_recovery_successful": error_recovery_successful,
                "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer
            }
        )
    
    async def test_performance_benchmarks(self) -> TestResult:
        """Test performance benchmarks for different query types."""
        performance_tests = [
            ("MATCH (n) RETURN count(n)", "simple_count"),
            ("show me the database structure", "complex_analysis"),
            ("how many nodes are there?", "natural_language_simple")
        ]
        
        performance_results = []
        all_within_limits = True
        
        for query, test_type in performance_tests:
            start_time = time.time()
            
            if test_type == "simple_count":
                # Direct MCP call for comparison
                result = await self.agent.call_mcp_tool("execute_read_query", {"query": query})
                success = not result.startswith("❌")
            else:
                # Full agent workflow
                answer = await self.agent.run(query)
                success = not answer.startswith("❌")
            
            execution_time = time.time() - start_time
            
            # Performance thresholds (these are reasonable for a development environment)
            time_limit = 30.0 if test_type == "complex_analysis" else 10.0
            within_limit = execution_time <= time_limit
            
            if not within_limit:
                all_within_limits = False
            
            performance_results.append({
                "query": query,
                "test_type": test_type,
                "execution_time": round(execution_time, 3),
                "time_limit": time_limit,
                "within_limit": within_limit,
                "success": success
            })
        
        status = TestStatus.PASS if all_within_limits else TestStatus.FAIL
        
        return TestResult(
            name="Performance Benchmarks",
            status=status,
            execution_time=0,
            details={
                "all_within_limits": all_within_limits,
                "performance_results": performance_results,
                "average_time": sum(r["execution_time"] for r in performance_results) / len(performance_results)
            }
        )

class ComprehensiveTestRunner:
    """Main test runner for the comprehensive testing system."""
    
    def __init__(self):
        self.test_suites: List[TestSuite] = [
            MCPServerTestSuite(),
            LangGraphAgentTestSuite(), 
            IntegrationTestSuite()
        ]
        self.all_results: List[TestResult] = []
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and compile results."""
        logger.info("🧪 Starting Comprehensive Test Suite")
        logger.info("=" * 80)
        
        start_time = time.time()
        suite_results = {}
        
        for suite in self.test_suites:
            logger.info(f"\n🔬 Running {suite.name}...")
            suite_start = time.time()
            
            try:
                results = await suite.run_all_tests()
                suite_time = time.time() - suite_start
                
                # Analyze suite results
                total_tests = len(results)
                passed_tests = sum(1 for r in results if r.status == TestStatus.PASS)
                failed_tests = sum(1 for r in results if r.status == TestStatus.FAIL)
                error_tests = sum(1 for r in results if r.status == TestStatus.ERROR)
                
                suite_results[suite.name] = {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "error_tests": error_tests,
                    "success_rate": round((passed_tests / total_tests) * 100, 1) if total_tests > 0 else 0,
                    "execution_time": round(suite_time, 3),
                    "test_results": [r.to_dict() for r in results]
                }
                
                self.all_results.extend(results)
                
                logger.info(f"  ✅ {suite.name} completed: {passed_tests}/{total_tests} passed ({suite_time:.3f}s)")
                
            except Exception as e:
                logger.error(f"  ❌ {suite.name} failed with exception: {e}")
                suite_results[suite.name] = {
                    "total_tests": 0,
                    "passed_tests": 0,
                    "failed_tests": 0,
                    "error_tests": 1,
                    "success_rate": 0,
                    "execution_time": 0,
                    "exception": str(e)
                }
        
        total_time = time.time() - start_time
        
        # Compile overall results
        total_tests = sum(suite["total_tests"] for suite in suite_results.values())
        total_passed = sum(suite["passed_tests"] for suite in suite_results.values())
        total_failed = sum(suite["failed_tests"] for suite in suite_results.values())
        total_errors = sum(suite["error_tests"] for suite in suite_results.values())
        
        overall_results = {
            "test_summary": {
                "total_execution_time": round(total_time, 3),
                "total_tests": total_tests,
                "passed_tests": total_passed,
                "failed_tests": total_failed,
                "error_tests": total_errors,
                "overall_success_rate": round((total_passed / total_tests) * 100, 1) if total_tests > 0 else 0,
                "timestamp": datetime.now().isoformat()
            },
            "suite_results": suite_results,
            "system_assessment": self._assess_system_health(suite_results),
            "recommendations": self._generate_recommendations(suite_results)
        }
        
        logger.info("\n" + "=" * 80)
        logger.info(f"🧪 Testing completed in {total_time:.3f}s")
        logger.info(f"📊 Overall: {total_passed}/{total_tests} tests passed ({overall_results['test_summary']['overall_success_rate']}%)")
        
        return overall_results
    
    def _assess_system_health(self, suite_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system health based on test results."""
        assessment = {
            "overall_health": "unknown",
            "critical_issues": [],
            "warnings": [],
            "strengths": []
        }
        
        # Check MCP Server health
        mcp_results = suite_results.get("MCP Server Tests", {})
        mcp_success_rate = mcp_results.get("success_rate", 0)
        
        if mcp_success_rate >= 80:
            assessment["strengths"].append("MCP Server functioning well")
        elif mcp_success_rate >= 60:
            assessment["warnings"].append("MCP Server has some issues")
        else:
            assessment["critical_issues"].append("MCP Server has major problems")
        
        # Check LangGraph Agent health
        agent_results = suite_results.get("LangGraph Agent Tests", {})
        agent_success_rate = agent_results.get("success_rate", 0)
        
        if agent_success_rate >= 80:
            assessment["strengths"].append("LangGraph Agent working well")
        elif agent_success_rate >= 60:
            assessment["warnings"].append("LangGraph Agent has some issues")
        else:
            assessment["critical_issues"].append("LangGraph Agent has major problems")
        
        # Check Integration Tests (most important)
        integration_results = suite_results.get("Integration Tests", {})
        integration_success_rate = integration_results.get("success_rate", 0)
        
        # Look specifically for the originally failing query test
        integration_tests = integration_results.get("test_results", [])
        original_query_test = next(
            (t for t in integration_tests if "Originally Failing Query" in t.get("name", "")), 
            None
        )
        
        if original_query_test and original_query_test.get("status") == "PASS":
            assessment["strengths"].append("✅ Originally failing query now works!")
        elif original_query_test:
            assessment["critical_issues"].append("❌ Originally failing query still doesn't work")
        
        if integration_success_rate >= 75:
            assessment["strengths"].append("End-to-end integration working well")
        else:
            assessment["critical_issues"].append("End-to-end integration has problems")
        
        # Overall assessment
        if len(assessment["critical_issues"]) == 0:
            assessment["overall_health"] = "excellent" if len(assessment["strengths"]) >= 3 else "good"
        elif len(assessment["critical_issues"]) <= 1:
            assessment["overall_health"] = "fair"
        else:
            assessment["overall_health"] = "poor"
        
        return assessment
    
    def _generate_recommendations(self, suite_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on test results."""
        recommendations = []
        
        # Check each suite for specific issues
        for suite_name, results in suite_results.items():
            success_rate = results.get("success_rate", 0)
            
            if success_rate < 50:
                if "MCP Server" in suite_name:
                    recommendations.append("🔧 Check MCP server configuration and Neo4j connection")
                    recommendations.append("📝 Verify langgraph_mcpserver.py file exists and is correct")
                elif "LangGraph Agent" in suite_name:
                    recommendations.append("🧠 Check LangGraph agent configuration and dependencies")
                    recommendations.append("🔑 Verify Cortex API credentials in config.py")
                elif "Integration" in suite_name:
                    recommendations.append("🔗 Focus on fixing individual components before integration")
            
            elif success_rate < 80:
                recommendations.append(f"⚠️ {suite_name} needs attention - review failed tests")
        
        # Check for the key success indicator
        integration_results = suite_results.get("Integration Tests", {})
        integration_tests = integration_results.get("test_results", [])
        original_query_test = next(
            (t for t in integration_tests if "Originally Failing Query" in t.get("name", "")), 
            None
        )
        
        if original_query_test and original_query_test.get("status") != "PASS":
            recommendations.append("🎯 PRIORITY: Fix the originally failing query - this is the main goal")
            recommendations.append("🔍 Check if syntax fixes are being applied correctly")
        
        # General recommendations
        if not recommendations:
            recommendations.append("🎉 System is working well! Ready for production use.")
        else:
            recommendations.append("🔧 Run health check for more detailed diagnostics: python health_check.py")
            recommendations.append("📋 Check logs in logs/ directory for detailed error information")
        
        return recommendations
    
    def print_test_report(self, results: Dict[str, Any]):
        """Print a formatted test report."""
        print("\n" + "="*100)
        print("🧪 INTELLIGENT NEO4J ASSISTANT - COMPREHENSIVE TEST REPORT")
        print("="*100)
        
        # Overall summary
        summary = results["test_summary"]
        print(f"\n📊 OVERALL SUMMARY:")
        print(f"   • Total Tests: {summary['total_tests']}")
        print(f"   • Passed: {summary['passed_tests']} ✅")
        print(f"   • Failed: {summary['failed_tests']} ❌")
        print(f"   • Errors: {summary['error_tests']} 💥")
        print(f"   • Success Rate: {summary['overall_success_rate']}%")
        print(f"   • Total Time: {summary['total_execution_time']}s")
        
        # Suite breakdown
        print(f"\n🔬 TEST SUITE BREAKDOWN:")
        print("-" * 100)
        
        for suite_name, suite_data in results["suite_results"].items():
            success_rate = suite_data.get("success_rate", 0)
            status_icon = "✅" if success_rate >= 80 else "⚠️" if success_rate >= 60 else "❌"
            
            print(f"{suite_name:.<40} {status_icon} {suite_data['passed_tests']}/{suite_data['total_tests']} ({success_rate}%)")
            
            # Show failed tests
            failed_tests = [
                t for t in suite_data.get("test_results", []) 
                if t.get("status") in ["FAIL", "ERROR"]
            ]
            
            for failed_test in failed_tests:
                print(f"   ❌ {failed_test['name']}: {failed_test.get('error', 'Failed')}")
        
        print("-" * 100)
        
        # System assessment
        assessment = results["system_assessment"]
        health_icons = {
            "excellent": "🌟",
            "good": "✅", 
            "fair": "⚠️",
            "poor": "❌",
            "unknown": "❓"
        }
        
        health_icon = health_icons.get(assessment["overall_health"], "❓")
        print(f"\n{health_icon} SYSTEM HEALTH: {assessment['overall_health'].upper()}")
        
        if assessment["strengths"]:
            print(f"\n💪 STRENGTHS:")
            for strength in assessment["strengths"]:
                print(f"   • {strength}")
        
        if assessment["warnings"]:
            print(f"\n⚠️ WARNINGS:")
            for warning in assessment["warnings"]:
                print(f"   • {warning}")
        
        if assessment["critical_issues"]:
            print(f"\n🚨 CRITICAL ISSUES:")
            for issue in assessment["critical_issues"]:
                print(f"   • {issue}")
        
        # Recommendations
        recommendations = results["recommendations"]
        if recommendations:
            print(f"\n🔧 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Key success indicator
        integration_results = results["suite_results"].get("Integration Tests", {})
        integration_tests = integration_results.get("test_results", [])
        original_query_test = next(
            (t for t in integration_tests if "Originally Failing Query" in t.get("name", "")), 
            None
        )
        
        print(f"\n🎯 KEY SUCCESS INDICATOR:")
        if original_query_test:
            if original_query_test.get("status") == "PASS":
                print(f"   ✅ Originally failing query: WORKS! 🎉")
                print(f"   🚀 Your system transformation is SUCCESSFUL!")
            else:
                print(f"   ❌ Originally failing query: Still broken")
                print(f"   🔧 This needs to be fixed for the system to be considered successful")
        else:
            print(f"   ❓ Originally failing query test not found")
        
        print("="*100)

# Standalone functions
def run_all_tests() -> bool:
    """Run all tests and return success status."""
    try:
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)
        
        runner = ComprehensiveTestRunner()
        
        # Use existing event loop or create new one
        try:
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(runner.run_all_tests())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(runner.run_all_tests())
            finally:
                loop.close()
        
        # Print report
        runner.print_test_report(results)
        
        # Save detailed results
        try:
            with open("logs/test_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n📁 Detailed results saved to: logs/test_results.json")
        except Exception as e:
            logger.warning(f"Could not save results to file: {e}")
        
        # Determine overall success
        success_rate = results["test_summary"]["overall_success_rate"]
        return success_rate >= 75  # 75% threshold for overall success
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"\n💥 TEST EXECUTION FAILED: {e}")
        return False

async def run_quick_test() -> bool:
    """Run a quick subset of critical tests."""
    logger.info("🚀 Running quick test suite...")
    
    try:
        # Test just the critical functionality
        agent = OptimizedNeo4jAgent()
        
        # Test the originally failing query
        answer = await agent.run("show me nodes with most connected nodes in the database?")
        success = not answer.startswith("❌") and len(answer) > 50
        
        if success:
            print("✅ Quick test PASSED - Originally failing query works!")
        else:
            print("❌ Quick test FAILED - Originally failing query still broken")
        
        return success
        
    except Exception as e:
        print(f"❌ Quick test ERROR: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Intelligent Neo4j Assistant - Comprehensive Testing System")
    print("Starting comprehensive test suite...")
    
    success = run_all_tests()
    
    print(f"\nTesting {'✅ COMPLETED SUCCESSFULLY' if success else '❌ COMPLETED WITH ISSUES'}")
    
    # Exit with appropriate code
    exit_code = 0 if success else 1
    sys.exit(exit_code)
