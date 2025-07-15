# health_check.py - Comprehensive Health Monitoring for Intelligent Neo4j Assistant

import asyncio
import json
import logging
import time
import sys
import traceback
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

# Import system components
try:
    from fastmcp import Client
    from neo4j import AsyncGraphDatabase
    import requests
    import urllib3
    from config import Config
    from updated_langgraph_agent import OptimizedNeo4jAgent
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

# Suppress SSL warnings for development
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/health_check.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HealthCheckResult:
    """Represents the result of a health check."""
    
    def __init__(self, component: str, healthy: bool, details: Dict[str, Any], 
                 execution_time: float = 0.0, error: Optional[str] = None):
        self.component = component
        self.healthy = healthy
        self.details = details
        self.execution_time = execution_time
        self.error = error
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "component": self.component,
            "healthy": self.healthy,
            "details": self.details,
            "execution_time_ms": round(self.execution_time * 1000, 2),
            "error": self.error,
            "timestamp": self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        status = "✅ HEALTHY" if self.healthy else "❌ UNHEALTHY"
        return f"{self.component}: {status} ({self.execution_time:.3f}s)"

class ComprehensiveHealthMonitor:
    """Comprehensive health monitoring system for the Intelligent Neo4j Assistant."""
    
    def __init__(self):
        self.config = Config()
        self.results: List[HealthCheckResult] = []
        self.overall_health = True
        
    async def check_python_environment(self) -> HealthCheckResult:
        """Check Python environment and dependencies."""
        component = "Python Environment"
        start_time = time.time()
        
        try:
            details = {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform,
                "executable": sys.executable
            }
            
            # Check critical dependencies
            required_packages = [
                'streamlit', 'langgraph', 'langchain', 'fastmcp', 
                'neo4j', 'pandas', 'plotly', 'requests', 'nest_asyncio'
            ]
            
            missing_packages = []
            package_versions = {}
            
            for package in required_packages:
                try:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'unknown')
                    package_versions[package] = version
                except ImportError:
                    missing_packages.append(package)
            
            details["package_versions"] = package_versions
            details["missing_packages"] = missing_packages
            
            # Check Python version compatibility
            python_compatible = sys.version_info >= (3, 8)
            details["python_compatible"] = python_compatible
            
            healthy = python_compatible and len(missing_packages) == 0
            error = f"Missing packages: {missing_packages}" if missing_packages else None
            
            execution_time = time.time() - start_time
            return HealthCheckResult(component, healthy, details, execution_time, error)
            
        except Exception as e:
            execution_time = time.time() - start_time
            return HealthCheckResult(component, False, {}, execution_time, str(e))
    
    def check_file_system(self) -> HealthCheckResult:
        """Check file system and required files."""
        component = "File System"
        start_time = time.time()
        
        try:
            details = {}
            
            # Check required files
            required_files = [
                "langgraph_mcpserver.py",
                "updated_langgraph_agent.py",
                "neo4j_intelligent_ui.py",
                "config.py"
            ]
            
            missing_files = []
            existing_files = {}
            
            for file_path in required_files:
                path = Path(file_path)
                if path.exists():
                    existing_files[file_path] = {
                        "size_bytes": path.stat().st_size,
                        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                    }
                else:
                    missing_files.append(file_path)
            
            details["existing_files"] = existing_files
            details["missing_files"] = missing_files
            
            # Check required directories
            required_dirs = ["logs", "cache", "exports"]
            missing_dirs = []
            existing_dirs = {}
            
            for dir_path in required_dirs:
                path = Path(dir_path)
                if path.exists():
                    existing_dirs[dir_path] = {
                        "exists": True,
                        "writable": os.access(path, os.W_OK) if hasattr(os, 'access') else True
                    }
                else:
                    missing_dirs.append(dir_path)
                    # Try to create missing directory
                    try:
                        path.mkdir(exist_ok=True)
                        existing_dirs[dir_path] = {"exists": True, "created": True}
                    except Exception as e:
                        details[f"dir_creation_error_{dir_path}"] = str(e)
            
            details["existing_directories"] = existing_dirs
            details["missing_directories"] = missing_dirs
            
            healthy = len(missing_files) == 0
            error = f"Missing files: {missing_files}" if missing_files else None
            
            execution_time = time.time() - start_time
            return HealthCheckResult(component, healthy, details, execution_time, error)
            
        except Exception as e:
            execution_time = time.time() - start_time
            return HealthCheckResult(component, False, {}, execution_time, str(e))
    
    async def check_neo4j_connection(self) -> HealthCheckResult:
        """Check Neo4j database connection and basic functionality."""
        component = "Neo4j Database"
        start_time = time.time()
        
        try:
            neo4j_config = self.config.get_neo4j_config()
            details = {
                "uri": neo4j_config["uri"],
                "database": neo4j_config["database"],
                "username": neo4j_config["username"]
            }
            
            # Test connection
            driver = AsyncGraphDatabase.driver(
                neo4j_config["uri"],
                auth=(neo4j_config["username"], neo4j_config["password"])
            )
            
            # Test basic query
            async with driver.session(database=neo4j_config["database"]) as session:
                # Basic connectivity test
                result = await session.run("RETURN 1 as test, datetime() as timestamp")
                record = await result.single()
                details["connection_test"] = {
                    "test_value": record["test"],
                    "server_time": str(record["timestamp"])
                }
                
                # Database info
                db_info_result = await session.run("""
                    CALL dbms.components() YIELD name, versions, edition
                    RETURN name, versions[0] as version, edition
                """)
                db_info = await db_info_result.single()
                if db_info:
                    details["database_info"] = {
                        "name": db_info["name"],
                        "version": db_info["version"],
                        "edition": db_info["edition"]
                    }
                
                # Basic statistics
                node_count_result = await session.run("MATCH (n) RETURN count(n) as node_count")
                node_record = await node_count_result.single()
                details["statistics"] = {
                    "node_count": node_record["node_count"] if node_record else 0
                }
                
                # Test write capability (create and delete a test node)
                write_test_result = await session.run("""
                    CREATE (test:HealthCheck {created: datetime(), id: randomUUID()})
                    RETURN test.id as test_id
                """)
                write_record = await write_test_result.single()
                test_id = write_record["test_id"]
                
                # Clean up test node
                await session.run("MATCH (test:HealthCheck {id: $id}) DELETE test", {"id": test_id})
                
                details["write_test"] = {"success": True, "test_id": test_id}
            
            await driver.close()
            
            execution_time = time.time() - start_time
            return HealthCheckResult(component, True, details, execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            return HealthCheckResult(component, False, error_details, execution_time, str(e))
    
    async def check_mcp_server(self) -> HealthCheckResult:
        """Check MCP server functionality."""
        component = "MCP Server"
        start_time = time.time()
        
        try:
            details = {"script_path": self.config.MCP_SERVER_SCRIPT}
            
            # Check if MCP server script exists
            script_path = Path(self.config.MCP_SERVER_SCRIPT)
            if not script_path.exists():
                return HealthCheckResult(
                    component, False, details, time.time() - start_time,
                    f"MCP server script not found: {self.config.MCP_SERVER_SCRIPT}"
                )
            
            details["script_exists"] = True
            details["script_size"] = script_path.stat().st_size
            
            # Test MCP server connection
            async with Client(self.config.MCP_SERVER_SCRIPT) as client:
                # Test health check tool
                health_result = await client.call_tool("health_check")
                
                if hasattr(health_result, 'content') and health_result.content:
                    health_content = health_result.content[0].text
                    health_data = json.loads(health_content)
                    details["health_check"] = health_data
                    
                    # Test metrics tool
                    metrics_result = await client.call_tool("get_metrics")
                    if hasattr(metrics_result, 'content') and metrics_result.content:
                        metrics_content = metrics_result.content[0].text
                        metrics_data = json.loads(metrics_content)
                        details["metrics"] = metrics_data
                    
                    # Test schema analysis tool
                    try:
                        schema_result = await client.call_tool("database_summary")
                        if hasattr(schema_result, 'content') and schema_result.content:
                            schema_content = schema_result.content[0].text
                            schema_data = json.loads(schema_content)
                            details["database_summary"] = {
                                "node_count": schema_data.get("node_count", 0),
                                "relationship_count": schema_data.get("relationship_count", 0),
                                "label_count": schema_data.get("label_count", 0)
                            }
                    except Exception as schema_error:
                        details["schema_test_error"] = str(schema_error)
                    
                    healthy = health_data.get("status") == "healthy"
                else:
                    healthy = False
                    details["error"] = "No content in health check response"
            
            execution_time = time.time() - start_time
            return HealthCheckResult(component, healthy, details, execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            return HealthCheckResult(component, False, error_details, execution_time, str(e))
    
    async def check_cortex_api(self) -> HealthCheckResult:
        """Check Cortex API configuration and connectivity."""
        component = "Cortex API"
        start_time = time.time()
        
        try:
            cortex_config = self.config.get_cortex_config()
            
            # Don't expose sensitive information
            details = {
                "url": cortex_config["url"],
                "model": cortex_config["model"],
                "app_id": cortex_config["app_id"],
                "has_api_key": bool(cortex_config["api_key"]),
                "api_key_length": len(cortex_config["api_key"]) if cortex_config["api_key"] else 0
            }
            
            # Test basic connectivity (without making actual API call to avoid costs)
            import urllib.parse
            parsed_url = urllib.parse.urlparse(cortex_config["url"])
            details["url_parsed"] = {
                "scheme": parsed_url.scheme,
                "hostname": parsed_url.hostname,
                "port": parsed_url.port,
                "path": parsed_url.path
            }
            
            # Check if URL is reachable (basic connection test)
            try:
                response = requests.head(cortex_config["url"], timeout=10, verify=False)
                details["connectivity_test"] = {
                    "status_code": response.status_code,
                    "headers_available": bool(response.headers),
                    "connection_successful": True
                }
                connection_healthy = True
            except requests.exceptions.RequestException as e:
                details["connectivity_test"] = {
                    "connection_successful": False,
                    "error": str(e)
                }
                connection_healthy = False
            
            # Configuration completeness check
            required_fields = ["url", "api_key", "app_id", "aplctn_cd", "model"]
            missing_fields = [field for field in required_fields if not cortex_config.get(field)]
            
            details["configuration"] = {
                "required_fields": required_fields,
                "missing_fields": missing_fields,
                "configuration_complete": len(missing_fields) == 0
            }
            
            healthy = len(missing_fields) == 0 and connection_healthy
            error = f"Missing configuration: {missing_fields}" if missing_fields else None
            
            execution_time = time.time() - start_time
            return HealthCheckResult(component, healthy, details, execution_time, error)
            
        except Exception as e:
            execution_time = time.time() - start_time
            return HealthCheckResult(component, False, {}, execution_time, str(e))
    
    async def check_langgraph_agent(self) -> HealthCheckResult:
        """Check LangGraph agent functionality."""
        component = "LangGraph Agent"
        start_time = time.time()
        
        try:
            details = {}
            
            # Initialize agent
            agent = OptimizedNeo4jAgent(self.config.MCP_SERVER_SCRIPT)
            details["agent_initialized"] = True
            
            # Test basic MCP tool call through agent
            try:
                health_result = await agent.call_mcp_tool("health_check")
                if health_result and not health_result.startswith("❌"):
                    health_data = json.loads(health_result)
                    details["mcp_tool_test"] = {
                        "success": True,
                        "health_status": health_data.get("status", "unknown")
                    }
                    mcp_healthy = True
                else:
                    details["mcp_tool_test"] = {"success": False, "error": health_result}
                    mcp_healthy = False
            except Exception as e:
                details["mcp_tool_test"] = {"success": False, "error": str(e)}
                mcp_healthy = False
            
            # Test question classification
            try:
                question_type, complexity = agent.classify_question("show me nodes with most connected nodes")
                details["classification_test"] = {
                    "success": True,
                    "question_type": question_type,
                    "complexity": complexity
                }
                classification_healthy = True
            except Exception as e:
                details["classification_test"] = {"success": False, "error": str(e)}
                classification_healthy = False
            
            # Test agent workflow (simple query)
            try:
                simple_answer = await agent.run("how many nodes are in the database?")
                details["workflow_test"] = {
                    "success": not simple_answer.startswith("❌"),
                    "answer_length": len(simple_answer),
                    "has_error": simple_answer.startswith("❌")
                }
                workflow_healthy = not simple_answer.startswith("❌")
            except Exception as e:
                details["workflow_test"] = {"success": False, "error": str(e)}
                workflow_healthy = False
            
            healthy = mcp_healthy and classification_healthy and workflow_healthy
            
            execution_time = time.time() - start_time
            return HealthCheckResult(component, healthy, details, execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            return HealthCheckResult(component, False, error_details, execution_time, str(e))
    
    async def check_integration_test(self) -> HealthCheckResult:
        """Perform end-to-end integration test with the originally failing query."""
        component = "Integration Test"
        start_time = time.time()
        
        try:
            details = {"test_description": "End-to-end test with originally failing query"}
            
            # Initialize agent
            agent = OptimizedNeo4jAgent(self.config.MCP_SERVER_SCRIPT)
            
            # Test the originally failing query
            test_question = "show me nodes with most connected nodes in the database?"
            details["test_question"] = test_question
            
            # Run the full workflow
            answer = await agent.run(test_question)
            
            # Analyze the result
            success = not answer.startswith("❌")
            has_connectivity_data = "connect" in answer.lower() and "node" in answer.lower()
            has_formatting = "🔗" in answer or "**" in answer or "#" in answer
            answer_length = len(answer)
            
            details["test_results"] = {
                "success": success,
                "answer_length": answer_length,
                "has_connectivity_data": has_connectivity_data,
                "has_formatting": has_formatting,
                "answer_preview": answer[:200] + "..." if answer_length > 200 else answer
            }
            
            # Additional performance test
            performance_start = time.time()
            simple_answer = await agent.run("count nodes in database")
            performance_time = time.time() - performance_start
            
            details["performance_test"] = {
                "simple_query_time": round(performance_time, 3),
                "simple_query_success": not simple_answer.startswith("❌")
            }
            
            healthy = success and has_connectivity_data and answer_length > 50
            
            execution_time = time.time() - start_time
            return HealthCheckResult(component, healthy, details, execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            return HealthCheckResult(component, False, error_details, execution_time, str(e))
    
    async def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check of all system components."""
        logger.info("🏥 Starting comprehensive health check...")
        
        start_time = time.time()
        self.results = []
        self.overall_health = True
        
        # Define health check sequence
        health_checks = [
            ("Python Environment", self.check_python_environment()),
            ("File System", self.check_file_system()),
            ("Neo4j Database", self.check_neo4j_connection()),
            ("MCP Server", self.check_mcp_server()),
            ("Cortex API", self.check_cortex_api()),
            ("LangGraph Agent", self.check_langgraph_agent()),
            ("Integration Test", self.check_integration_test())
        ]
        
        # Execute health checks
        for check_name, check_coro in health_checks:
            logger.info(f"Checking {check_name}...")
            
            try:
                if asyncio.iscoroutine(check_coro):
                    result = await check_coro
                else:
                    result = check_coro
                
                self.results.append(result)
                self.overall_health &= result.healthy
                
                status = "✅" if result.healthy else "❌"
                logger.info(f"{status} {check_name}: {result.execution_time:.3f}s")
                
                if not result.healthy and result.error:
                    logger.warning(f"   Error: {result.error}")
                    
            except Exception as e:
                logger.error(f"❌ {check_name} check failed with exception: {e}")
                error_result = HealthCheckResult(check_name, False, {}, 0.0, str(e))
                self.results.append(error_result)
                self.overall_health = False
        
        total_time = time.time() - start_time
        
        # Compile final results
        results_summary = {
            "overall_health": self.overall_health,
            "total_execution_time": round(total_time, 3),
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform
            },
            "component_results": [result.to_dict() for result in self.results],
            "summary": {
                "total_components": len(self.results),
                "healthy_components": sum(1 for r in self.results if r.healthy),
                "unhealthy_components": sum(1 for r in self.results if not r.healthy),
                "health_percentage": round(sum(1 for r in self.results if r.healthy) / len(self.results) * 100, 1)
            }
        }
        
        logger.info(f"🏥 Health check completed in {total_time:.3f}s")
        logger.info(f"Overall health: {'✅ HEALTHY' if self.overall_health else '❌ ISSUES DETECTED'}")
        
        return results_summary
    
    def print_health_report(self, results: Dict[str, Any]):
        """Print a formatted health report."""
        print("\n" + "="*80)
        print("🏥 INTELLIGENT NEO4J ASSISTANT - HEALTH CHECK REPORT")
        print("="*80)
        
        # Overall status
        overall = results["overall_health"]
        status_icon = "✅" if overall else "❌"
        status_text = "SYSTEM HEALTHY" if overall else "SYSTEM ISSUES DETECTED"
        print(f"\n{status_icon} OVERALL STATUS: {status_text}")
        
        # Summary
        summary = results["summary"]
        print(f"\n📊 SUMMARY:")
        print(f"   • Total Components: {summary['total_components']}")
        print(f"   • Healthy: {summary['healthy_components']}")
        print(f"   • Issues: {summary['unhealthy_components']}")
        print(f"   • Health Score: {summary['health_percentage']}%")
        print(f"   • Total Time: {results['total_execution_time']}s")
        
        # Component details
        print(f"\n🔍 COMPONENT DETAILS:")
        print("-" * 80)
        
        for component_result in results["component_results"]:
            status = "✅ HEALTHY" if component_result["healthy"] else "❌ UNHEALTHY"
            name = component_result["component"]
            time_ms = component_result["execution_time_ms"]
            
            print(f"{name:.<25} {status} ({time_ms}ms)")
            
            if not component_result["healthy"] and component_result.get("error"):
                print(f"   ⚠️  Error: {component_result['error']}")
            
            # Show key details for each component
            details = component_result.get("details", {})
            if component_result["component"] == "Neo4j Database" and component_result["healthy"]:
                stats = details.get("statistics", {})
                print(f"   📊 Nodes: {stats.get('node_count', 'unknown')}")
            elif component_result["component"] == "Integration Test" and component_result["healthy"]:
                test_results = details.get("test_results", {})
                print(f"   🎯 Originally failing query: {'✅ WORKS' if test_results.get('success') else '❌ STILL FAILS'}")
        
        print("-" * 80)
        
        # Recommendations
        unhealthy_components = [r for r in results["component_results"] if not r["healthy"]]
        if unhealthy_components:
            print(f"\n🔧 RECOMMENDATIONS:")
            for component in unhealthy_components:
                name = component["component"]
                error = component.get("error", "Unknown error")
                
                if "Python Environment" in name:
                    print(f"   • Install missing packages: pip install -r requirements.txt")
                elif "Neo4j Database" in name:
                    print(f"   • Check Neo4j connection settings in config.py")
                    print(f"   • Verify Neo4j server is running and accessible")
                elif "MCP Server" in name:
                    print(f"   • Ensure langgraph_mcpserver.py exists and is correct")
                elif "Cortex API" in name:
                    print(f"   • Verify Cortex API credentials in config.py")
                elif "LangGraph Agent" in name:
                    print(f"   • Check that all core components are working first")
                elif "Integration Test" in name:
                    print(f"   • Your originally failing query still has issues")
                    print(f"   • Check MCP server and agent configuration")
        else:
            print(f"\n🎉 CONGRATULATIONS!")
            print(f"   • All components are healthy!")
            print(f"   • Your originally failing query should now work perfectly!")
            print(f"   • Ready to run: python run.py")
        
        print("="*80)

# Standalone functions for external use
async def quick_health_check() -> bool:
    """Quick health check returning simple boolean."""
    monitor = ComprehensiveHealthMonitor()
    results = await monitor.run_comprehensive_health_check()
    return results["overall_health"]

async def get_health_status() -> Dict[str, Any]:
    """Get detailed health status."""
    monitor = ComprehensiveHealthMonitor()
    return await monitor.run_comprehensive_health_check()

def run_health_check() -> bool:
    """Synchronous wrapper for health check."""
    try:
        import nest_asyncio
        nest_asyncio.apply()
        
        loop = asyncio.get_event_loop()
        monitor = ComprehensiveHealthMonitor()
        results = loop.run_until_complete(monitor.run_comprehensive_health_check())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            monitor = ComprehensiveHealthMonitor()
            results = loop.run_until_complete(monitor.run_comprehensive_health_check())
        finally:
            loop.close()
    
    # Print report
    monitor.print_health_report(results)
    
    # Save detailed results to file
    try:
        with open("logs/health_check_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Detailed results saved to: logs/health_check_results.json")
    except Exception as e:
        logger.warning(f"Could not save results to file: {e}")
    
    return results["overall_health"]

if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Add import fix for missing os module
    import os
    
    print("🏥 Intelligent Neo4j Assistant - Health Check")
    print("Starting comprehensive system health check...")
    
    success = run_health_check()
    
    # Exit with appropriate code
    exit_code = 0 if success else 1
    print(f"\nHealth check {'✅ PASSED' if success else '❌ FAILED'}")
    sys.exit(exit_code)
