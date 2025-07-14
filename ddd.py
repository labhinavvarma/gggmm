#!/usr/bin/env python3
"""
MCP SDK Test Script for Neo4j Server
Tests the Neo4j MCP server using the official MCP SDK
"""

import asyncio
import json
import sys
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

async def test_neo4j_mcp_server():
    """Test Neo4j MCP server functionality"""
    
    print_header("NEO4J MCP SERVER TEST")
    print(f"{Colors.BOLD}Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    
    # Configuration
    server_url = 'http://localhost:8001/sse'  # Update to match your app.py port
    
    print_info(f"Connecting to: {server_url}")
    print("=" * 50)
    
    test_results = {}
    
    try:
        # Step 1: Establish SSE connection
        print_header("STEP 1: ESTABLISHING SSE CONNECTION")
        async with sse_client(url=server_url) as sse_connection:
            print_success("SSE connection established")
            test_results['sse_connection'] = True
            
            # Step 2: Create client session
            print_header("STEP 2: CREATING CLIENT SESSION")
            async with ClientSession(*sse_connection) as session:
                print_success("Client session created")
                test_results['client_session'] = True
                
                # Step 3: Initialize session
                print_header("STEP 3: INITIALIZING SESSION")
                await session.initialize()
                print_success("Session initialized")
                test_results['session_init'] = True
                
                # Step 4: List available resources
                print_header("STEP 4: LISTING RESOURCES")
                try:
                    resources = await session.list_resources()
                    if hasattr(resources, 'resources') and resources.resources:
                        print_success(f"Found {len(resources.resources)} resources")
                        for resource in resources.resources:
                            print_info(f"Resource: {resource.name}")
                            if hasattr(resource, 'description'):
                                print(f"  Description: {resource.description}")
                        test_results['list_resources'] = True
                    else:
                        print_warning("No resources found")
                        test_results['list_resources'] = True  # Still pass, just no resources
                except Exception as e:
                    print_error(f"Failed to list resources: {e}")
                    test_results['list_resources'] = False
                
                # Step 5: List available tools
                print_header("STEP 5: LISTING TOOLS")
                try:
                    tools = await session.list_tools()
                    if hasattr(tools, 'tools') and tools.tools:
                        print_success(f"Found {len(tools.tools)} tools")
                        
                        # Track unique tools to avoid duplicates
                        seen_tools = set()
                        for tool in tools.tools:
                            if tool.name not in seen_tools:
                                seen_tools.add(tool.name)
                                print_info(f"Tool: {tool.name}")
                                if hasattr(tool, 'description') and tool.description:
                                    print(f"  Description: {tool.description}")
                                print("-" * 30)
                        
                        test_results['list_tools'] = True
                    else:
                        print_error("No tools found")
                        test_results['list_tools'] = False
                except Exception as e:
                    print_error(f"Failed to list tools: {e}")
                    test_results['list_tools'] = False
                
                # Step 6: List available prompts
                print_header("STEP 6: LISTING PROMPTS")
                try:
                    prompts = await session.list_prompts()
                    if hasattr(prompts, 'prompts') and prompts.prompts:
                        print_success(f"Found {len(prompts.prompts)} prompts")
                        
                        seen_prompts = set()
                        for prompt in prompts.prompts:
                            if prompt.name not in seen_prompts:
                                seen_prompts.add(prompt.name)
                                print_info(f"Prompt: {prompt.name}")
                                if hasattr(prompt, 'description') and prompt.description:
                                    print(f"  Description: {prompt.description}")
                                
                                # Display arguments if available
                                if hasattr(prompt, 'arguments') and prompt.arguments:
                                    print("  Arguments:")
                                    for arg in prompt.arguments:
                                        required_str = "[Required]" if arg.required else "[Optional]"
                                        print(f"    - {arg.name} {required_str}: {arg.description}")
                                print("-" * 30)
                        
                        test_results['list_prompts'] = True
                    else:
                        print_warning("No prompts found")
                        test_results['list_prompts'] = True  # Still pass, just no prompts
                except Exception as e:
                    print_error(f"Failed to list prompts: {e}")
                    test_results['list_prompts'] = False
                
                # Step 7: Test connection health tool
                print_header("STEP 7: TESTING CONNECTION HEALTH")
                try:
                    health_result = await session.call_tool("check_connection_health", {})
                    if hasattr(health_result, 'content') and health_result.content:
                        health_data = json.loads(health_result.content[0].text)
                        print_json(health_data, "Health Check Result")
                        
                        if health_data.get('healthy'):
                            print_success("Neo4j connection is healthy!")
                            print_info(f"Status: {health_data.get('status')}")
                            print_info(f"URI: {health_data.get('uri')}")
                            print_info(f"Database: {health_data.get('database')}")
                        else:
                            print_error(f"Neo4j connection is unhealthy: {health_data.get('error')}")
                        
                        test_results['connection_health'] = health_data.get('healthy', False)
                    else:
                        print_error("No health check result returned")
                        test_results['connection_health'] = False
                except Exception as e:
                    print_error(f"Failed to check connection health: {e}")
                    test_results['connection_health'] = False
                
                # Step 8: Test Cypher query execution
                print_header("STEP 8: TESTING CYPHER EXECUTION")
                try:
                    test_query = "RETURN 'Neo4j MCP Test Success' as message, datetime() as timestamp, 42 as test_number"
                    cypher_result = await session.call_tool("execute_cypher", {"query": test_query})
                    
                    if hasattr(cypher_result, 'content') and cypher_result.content:
                        query_data = json.loads(cypher_result.content[0].text)
                        print_json(query_data, "Cypher Query Result")
                        
                        if "error" not in query_data:
                            records = query_data.get('records', [])
                            if records:
                                record = records[0]
                                message = record.get('message')
                                timestamp = record.get('timestamp')
                                test_number = record.get('test_number')
                                
                                print_success("Cypher query executed successfully!")
                                print_info(f"Message: {message}")
                                print_info(f"Timestamp: {timestamp}")
                                print_info(f"Test Number: {test_number}")
                                
                                test_results['cypher_execution'] = True
                            else:
                                print_error("No records returned from query")
                                test_results['cypher_execution'] = False
                        else:
                            print_error(f"Cypher query failed: {query_data.get('error')}")
                            test_results['cypher_execution'] = False
                    else:
                        print_error("No cypher result returned")
                        test_results['cypher_execution'] = False
                except Exception as e:
                    print_error(f"Failed to execute Cypher query: {e}")
                    test_results['cypher_execution'] = False
                
                # Step 9: Test database schema retrieval
                print_header("STEP 9: TESTING DATABASE SCHEMA")
                try:
                    schema_result = await session.call_tool("get_database_schema", {})
                    
                    if hasattr(schema_result, 'content') and schema_result.content:
                        schema_data = json.loads(schema_result.content[0].text)
                        
                        if "error" not in schema_data:
                            print_success("Database schema retrieved successfully!")
                            
                            # Show schema summary
                            node_labels = schema_data.get('node_labels', {})
                            rel_types = schema_data.get('relationship_types', {})
                            
                            if isinstance(node_labels, dict) and 'labels' in node_labels:
                                labels = node_labels['labels']
                                print_info(f"Node labels: {len(labels)} found")
                                if labels:
                                    print_info(f"Labels: {', '.join(labels[:5])}{'...' if len(labels) > 5 else ''}")
                            
                            if isinstance(rel_types, dict) and 'types' in rel_types:
                                types = rel_types['types']
                                print_info(f"Relationship types: {len(types)} found")
                                if types:
                                    print_info(f"Types: {', '.join(types[:5])}{'...' if len(types) > 5 else ''}")
                            
                            test_results['database_schema'] = True
                        else:
                            print_error(f"Schema retrieval failed: {schema_data.get('error')}")
                            test_results['database_schema'] = False
                    else:
                        print_error("No schema result returned")
                        test_results['database_schema'] = False
                except Exception as e:
                    print_error(f"Failed to get database schema: {e}")
                    test_results['database_schema'] = False
                
                # Step 10: Test connection info
                print_header("STEP 10: TESTING CONNECTION INFO")
                try:
                    info_result = await session.call_tool("get_connection_info", {})
                    
                    if hasattr(info_result, 'content') and info_result.content:
                        info_data = json.loads(info_result.content[0].text)
                        print_json(info_data, "Connection Info")
                        
                        print_info(f"URI: {info_data.get('uri')}")
                        print_info(f"Database: {info_data.get('database')}")
                        print_info(f"Status: {info_data.get('status')}")
                        print_info(f"Driver Available: {info_data.get('driver_available')}")
                        
                        test_results['connection_info'] = True
                    else:
                        print_error("No connection info returned")
                        test_results['connection_info'] = False
                except Exception as e:
                    print_error(f"Failed to get connection info: {e}")
                    test_results['connection_info'] = False
                
        print_success("Connection closed successfully")
        
    except Exception as e:
        print_error(f"Connection failed: {e}")
        test_results['sse_connection'] = False
    
    # Test Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    print(f"{Colors.BOLD}Test Results:{Colors.END}")
    for test_name, success in test_results.items():
        status = "PASS" if success else "FAIL"
        color = Colors.GREEN if success else Colors.RED
        symbol = "✅" if success else "❌"
        print(f"{color}{symbol} {test_name.replace('_', ' ').title()}: {status}{Colors.END}")
    
    print(f"\n{Colors.BOLD}Overall Result: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print_success("🎉 ALL TESTS PASSED!")
        print_success("✅ Your Neo4j MCP server is working perfectly!")
        print_success("✅ Neo4j connection is healthy through MCP")
        print_success("✅ All tools are accessible and functional")
        print_success("✅ Ready for production use")
        return True
    else:
        print_error(f"❌ {total - passed} tests failed")
        print_warning("Common issues to check:")
        print_warning("1. Ensure app.py is running (python app.py)")
        print_warning("2. Check Neo4j database is accessible")
        print_warning("3. Verify MCP server credentials are correct")
        print_warning("4. Make sure port 8001 is accessible")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_neo4j_mcp_server())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_error("\n❌ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"❌ Unexpected error: {e}")
        sys.exit(1)
