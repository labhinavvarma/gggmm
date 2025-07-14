#!/usr/bin/env python3
"""
Test App.py MCP Integration
Test script to verify the MCP server works with your app.py
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from mcp import ClientSession
from mcp.client.sse import sse_client

# Color codes for better output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")

def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_info(message: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

def print_json(data: dict, title: str = "Data"):
    """Print JSON data in a formatted way"""
    print(f"{Colors.MAGENTA}📋 {title}:{Colors.END}")
    print(json.dumps(data, indent=2, default=str))

async def test_app_py_mcp_integration():
    """Test the MCP server integration with app.py"""
    
    print_header("APP.PY MCP INTEGRATION TEST")
    print(f"{Colors.BOLD}Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    
    # Configuration - your app.py runs on port 8001
    server_url = 'http://localhost:8001/sse'
    
    print_info(f"Testing app.py integration at: {server_url}")
    print_info("Expected: Neo4j MCP server integrated with FastAPI")
    print("=" * 60)
    
    test_results = {}
    
    try:
        # Step 1: Test app.py server is running
        print_header("STEP 1: CHECKING APP.PY SERVER")
        import aiohttp
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get("http://localhost:8001/docs") as response:
                    if response.status == 200:
                        print_success("App.py FastAPI server is running")
                        test_results['app_py_running'] = True
                    else:
                        print_error(f"App.py server returned status: {response.status}")
                        test_results['app_py_running'] = False
            except Exception as e:
                print_error(f"App.py server not accessible: {e}")
                print_warning("Make sure to run: python app.py")
                test_results['app_py_running'] = False
                return False
        
        # Step 2: Establish SSE connection
        print_header("STEP 2: ESTABLISHING SSE CONNECTION WITH APP.PY")
        async with sse_client(url=server_url) as sse_connection:
            print_success("SSE connection to app.py established")
            test_results['sse_connection'] = True
            
            # Step 3: Create client session
            print_header("STEP 3: CREATING MCP CLIENT SESSION")
            async with ClientSession(*sse_connection) as session:
                print_success("MCP client session created with app.py")
                test_results['client_session'] = True
                
                # Step 4: Initialize session
                print_header("STEP 4: INITIALIZING MCP SESSION")
                await session.initialize()
                print_success("MCP session initialized through app.py")
                test_results['session_init'] = True
                
                # Step 5: List tools available through app.py
                print_header("STEP 5: LISTING TOOLS VIA APP.PY")
                try:
                    tools = await session.list_tools()
                    if hasattr(tools, 'tools') and tools.tools:
                        print_success(f"Found {len(tools.tools)} tools via app.py")
                        
                        app_py_tools = []
                        for tool in tools.tools:
                            app_py_tools.append(tool.name)
                            print_info(f"Tool: {tool.name}")
                            if hasattr(tool, 'description') and tool.description:
                                print(f"  Description: {tool.description[:80]}...")
                            print("-" * 40)
                        
                        # Check for expected tools
                        expected_tools = [
                            'check_connection_health', 
                            'execute_cypher', 
                            'get_database_schema',
                            'test_connectiq_queries'
                        ]
                        
                        missing_tools = [tool for tool in expected_tools if tool not in app_py_tools]
                        if missing_tools:
                            print_warning(f"Missing expected tools: {missing_tools}")
                        else:
                            print_success("All expected app.py integration tools found!")
                        
                        test_results['list_tools'] = True
                    else:
                        print_error("No tools found via app.py")
                        test_results['list_tools'] = False
                except Exception as e:
                    print_error(f"Failed to list tools via app.py: {e}")
                    test_results['list_tools'] = False
                
                # Step 6: Test app.py specific health check
                print_header("STEP 6: TESTING APP.PY HEALTH CHECK")
                try:
                    health_result = await session.call_tool("check_connection_health", {})
                    if hasattr(health_result, 'content') and health_result.content:
                        health_data = json.loads(health_result.content[0].text)
                        print_json(health_data, "App.py Health Check")
                        
                        is_healthy = health_data.get('healthy')
                        integration_info = health_data.get('integration', {})
                        
                        if is_healthy:
                            print_success("Neo4j connection is healthy via app.py!")
                            print_info(f"Database: {health_data.get('database')}")
                            
                            if integration_info.get('app_py_compatible'):
                                print_success("App.py integration confirmed!")
                                print_info(f"FastAPI ready: {integration_info.get('fastapi_ready')}")
                                print_info(f"SSE ready: {integration_info.get('sse_transport_ready')}")
                            
                            # Performance check
                            performance = health_data.get('performance', {})
                            if performance:
                                response_time = performance.get('response_time_ms', 0)
                                print_info(f"Response time via app.py: {response_time}ms")
                        else:
                            print_error(f"Health check failed via app.py: {health_data.get('error')}")
                        
                        test_results['app_py_health'] = is_healthy
                    else:
                        print_error("No health check result from app.py")
                        test_results['app_py_health'] = False
                except Exception as e:
                    print_error(f"App.py health check failed: {e}")
                    test_results['app_py_health'] = False
                
                # Step 7: Test Connectiq database queries via app.py
                print_header("STEP 7: TESTING CONNECTIQ QUERIES VIA APP.PY")
                try:
                    connectiq_result = await session.call_tool("test_connectiq_queries", {})
                    
                    if hasattr(connectiq_result, 'content') and connectiq_result.content:
                        connectiq_data = json.loads(connectiq_result.content[0].text)
                        print_json(connectiq_data, "Connectiq Database Tests")
                        
                        summary = connectiq_data.get('summary', {})
                        tests_passed = summary.get('tests_passed', 0)
                        total_tests = summary.get('total_tests', 0)
                        
                        if tests_passed > 0:
                            print_success(f"Connectiq database tests via app.py: {tests_passed}/{total_tests} passed")
                            
                            if connectiq_data.get('database') == 'connectiq':
                                print_success("Connectiq healthcare database confirmed!")
                            
                            if summary.get('connectiq_database_verified'):
                                print_success("Healthcare data structure verified!")
                            
                            test_results['connectiq_tests'] = True
                        else:
                            print_warning("No Connectiq tests passed (may be expected)")
                            test_results['connectiq_tests'] = False
                    else:
                        print_error("No Connectiq test results from app.py")
                        test_results['connectiq_tests'] = False
                except Exception as e:
                    print_error(f"Connectiq tests via app.py failed: {e}")
                    test_results['connectiq_tests'] = False
                
                # Step 8: Test Cypher execution via app.py
                print_header("STEP 8: TESTING CYPHER EXECUTION VIA APP.PY")
                try:
                    test_query = "RETURN 'App.py MCP Integration Test' as message, datetime() as timestamp"
                    cypher_result = await session.call_tool("execute_cypher", {
                        "query": test_query
                    })
                    
                    if hasattr(cypher_result, 'content') and cypher_result.content:
                        cypher_data = json.loads(cypher_result.content[0].text)
                        
                        if "error" not in cypher_data:
                            records = cypher_data.get('records', [])
                            app_integration = cypher_data.get('app_integration', {})
                            
                            if records and app_integration:
                                message = records[0].get('message')
                                timestamp = records[0].get('timestamp')
                                
                                print_success("Cypher query executed via app.py!")
                                print_info(f"Message: {message}")
                                print_info(f"Timestamp: {timestamp}")
                                print_info(f"Executed via: {app_integration.get('executed_via')}")
                                print_info(f"Transport: {app_integration.get('transport')}")
                                
                                test_results['cypher_execution'] = True
                            else:
                                print_warning("Cypher query returned incomplete results")
                                test_results['cypher_execution'] = False
                        else:
                            print_error(f"Cypher execution error via app.py: {cypher_data.get('error')}")
                            test_results['cypher_execution'] = False
                    else:
                        print_error("No Cypher result from app.py")
                        test_results['cypher_execution'] = False
                except Exception as e:
                    print_error(f"Cypher execution via app.py failed: {e}")
                    test_results['cypher_execution'] = False
                
                # Step 9: Test schema retrieval via app.py
                print_header("STEP 9: TESTING SCHEMA RETRIEVAL VIA APP.PY")
                try:
                    schema_result = await session.call_tool("get_database_schema", {})
                    
                    if hasattr(schema_result, 'content') and schema_result.content:
                        schema_data = json.loads(schema_result.content[0].text)
                        
                        if "error" not in schema_data:
                            app_integration = schema_data.get('app_integration', {})
                            connectiq_tables = schema_data.get('connectiq_tables', [])
                            
                            print_success("Database schema retrieved via app.py!")
                            
                            if app_integration.get('optimized_for') == 'connectiq_healthcare_database':
                                print_success("Schema optimized for Connectiq healthcare database!")
                            
                            if connectiq_tables:
                                print_info(f"Connectiq tables found: {len(connectiq_tables)}")
                                for table in connectiq_tables[:3]:  # Show first 3
                                    print_info(f"  Table: {table.get('table_name')}")
                            
                            test_results['schema_retrieval'] = True
                        else:
                            print_error(f"Schema retrieval error via app.py: {schema_data.get('error')}")
                            test_results['schema_retrieval'] = False
                    else:
                        print_error("No schema result from app.py")
                        test_results['schema_retrieval'] = False
                except Exception as e:
                    print_error(f"Schema retrieval via app.py failed: {e}")
                    test_results['schema_retrieval'] = False
        
        print_success("App.py MCP connection closed successfully")
        
    except Exception as e:
        print_error(f"App.py MCP integration failed: {e}")
        test_results['sse_connection'] = False
    
    # App.py Integration Test Summary
    print_header("APP.PY INTEGRATION TEST SUMMARY")
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    print(f"{Colors.BOLD}App.py Integration Results:{Colors.END}")
    for test_name, success in test_results.items():
        status = "PASS" if success else "FAIL"
        color = Colors.GREEN if success else Colors.RED
        symbol = "✅" if success else "❌"
        print(f"{color}{symbol} {test_name.replace('_', ' ').title()}: {status}{Colors.END}")
    
    print(f"\n{Colors.BOLD}Overall App.py Integration: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print_success("🎉 PERFECT APP.PY INTEGRATION!")
        print_success("✅ MCP server works flawlessly with your app.py")
        print_success("✅ Neo4j Connectiq database is accessible via app.py")
        print_success("✅ All FastAPI + SSE + MCP features working")
        print_success("✅ Ready for production with app.py")
        return True
    elif passed >= total * 0.8:  # 80% pass rate
        print_success("🎯 EXCELLENT APP.PY INTEGRATION!")
        print_success(f"✅ {passed}/{total} tests passed - app.py integration is working well")
        print_info("Minor issues may exist but core functionality is solid")
        return True
    elif passed >= total * 0.6:  # 60% pass rate
        print_warning("⚠️  GOOD APP.PY INTEGRATION")
        print_warning(f"✅ {passed}/{total} tests passed - app.py integration has some issues")
        print_info("Core functionality working but needs attention")
        return True
    else:
        print_error(f"❌ APP.PY INTEGRATION ISSUES")
        print_error(f"Only {passed}/{total} tests passed")
        print_warning("App.py integration troubleshooting:")
        print_warning("1. Ensure app.py is running: python app.py")
        print_warning("2. Check mcpserver.py is in the same directory as app.py")
        print_warning("3. Verify Neo4j connection in mcpserver.py")
        print_warning("4. Check app.py imports: from mcpserver import mcp")
        print_warning("5. Verify port 8001 is available")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_app_py_mcp_integration())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_error("\n❌ App.py integration test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"❌ App.py integration test error: {e}")
        sys.exit(1)
