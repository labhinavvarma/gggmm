#!/usr/bin/env python3
"""
MCP Client Test Script
Tests the MCP server connection through SSE endpoint
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional
import aiohttp
import uuid

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


def print_json(data: dict, title: str = "Response"):
    """Print JSON data in a formatted way"""
    print(f"{Colors.MAGENTA}📋 {title}:{Colors.END}")
    print(json.dumps(data, indent=2, default=str))


class MCPClient:
    """MCP Client for testing SSE connection"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = None
        self.messages_url = f"{base_url}/messages/"
        self.sse_url = f"{base_url}/sse"
        self.session_id = str(uuid.uuid4())  # Generate session ID for this client
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_server_health(self) -> bool:
        """Test if the FastAPI server is running"""
        try:
            print_info("Testing FastAPI server health...")
            async with self.session.get(f"{self.base_url}/docs") as response:
                if response.status == 200:
                    print_success("FastAPI server is running")
                    return True
                else:
                    print_error(f"FastAPI server returned status: {response.status}")
                    return False
        except Exception as e:
            print_error(f"FastAPI server is not accessible: {e}")
            return False
    
    async def send_mcp_message(self, method: str, params: dict = None) -> dict:
        """Send an MCP message via POST to /messages"""
        try:
            message = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": method,
                "params": params or {}
            }
            
            print_info(f"Sending MCP message: {method}")
            print_json(message, "Request")
            
            # Add session_id as query parameter
            url_with_session = f"{self.messages_url}?session_id={self.session_id}"
            
            async with self.session.post(
                url_with_session,
                json=message,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    print_success(f"MCP message sent successfully")
                    print_json(result, "Response")
                    return result
                else:
                    error_text = await response.text()
                    print_error(f"MCP message failed with status {response.status}")
                    print_error(f"Error: {error_text}")
                    return {"error": f"HTTP {response.status}: {error_text}"}
                    
        except Exception as e:
            print_error(f"Error sending MCP message: {e}")
            return {"error": str(e)}
    
    async def test_mcp_initialize(self) -> bool:
        """Test MCP initialization"""
        try:
            print_info("Testing MCP initialization...")
            
            result = await self.send_mcp_message("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "mcp-test-client",
                    "version": "1.0.0"
                }
            })
            
            if "error" not in result:
                print_success("MCP initialization successful")
                return True
            else:
                print_error(f"MCP initialization failed: {result.get('error')}")
                return False
                
        except Exception as e:
            print_error(f"MCP initialization error: {e}")
            return False
    
    async def test_list_tools(self) -> bool:
        """Test listing MCP tools"""
        try:
            print_info("Testing MCP tools listing...")
            
            result = await self.send_mcp_message("tools/list")
            
            if "error" not in result and "result" in result:
                tools = result["result"].get("tools", [])
                print_success(f"Found {len(tools)} MCP tools")
                
                for tool in tools:
                    tool_name = tool.get("name", "Unknown")
                    tool_desc = tool.get("description", "No description")
                    print_info(f"Tool: {tool_name}")
                    print(f"   Description: {tool_desc}")
                
                return len(tools) > 0
            else:
                print_error(f"Failed to list tools: {result.get('error')}")
                return False
                
        except Exception as e:
            print_error(f"Error listing tools: {e}")
            return False
    
    async def test_connection_health_tool(self) -> bool:
        """Test the connection health tool"""
        try:
            print_info("Testing connection health tool...")
            
            result = await self.send_mcp_message("tools/call", {
                "name": "check_connection_health",
                "arguments": {}
            })
            
            if "error" not in result and "result" in result:
                content = result["result"].get("content", [])
                if content and len(content) > 0:
                    health_data = json.loads(content[0].get("text", "{}"))
                    
                    print_success("Connection health tool executed")
                    print_json(health_data, "Health Check Result")
                    
                    is_healthy = health_data.get("healthy", False)
                    if is_healthy:
                        print_success("Neo4j connection is healthy via MCP")
                        return True
                    else:
                        print_error(f"Neo4j connection is unhealthy: {health_data.get('error')}")
                        return False
                else:
                    print_error("No content returned from health check")
                    return False
            else:
                print_error(f"Health check tool failed: {result.get('error')}")
                return False
                
        except Exception as e:
            print_error(f"Error testing health tool: {e}")
            return False
    
    async def test_cypher_execution(self) -> bool:
        """Test Cypher query execution via MCP"""
        try:
            print_info("Testing Cypher execution via MCP...")
            
            test_query = "RETURN 1 as test_value, 'MCP Client Test' as message, datetime() as timestamp"
            
            result = await self.send_mcp_message("tools/call", {
                "name": "execute_cypher",
                "arguments": {
                    "query": test_query
                }
            })
            
            if "error" not in result and "result" in result:
                content = result["result"].get("content", [])
                if content and len(content) > 0:
                    query_result = json.loads(content[0].get("text", "{}"))
                    
                    print_success("Cypher query executed via MCP")
                    print_json(query_result, "Query Result")
                    
                    if "error" not in query_result:
                        records = query_result.get("records", [])
                        if records and len(records) > 0:
                            test_value = records[0].get("test_value")
                            message = records[0].get("message")
                            
                            if test_value == 1 and message == "MCP Client Test":
                                print_success("Cypher query returned expected results")
                                return True
                            else:
                                print_error("Cypher query returned unexpected results")
                                return False
                        else:
                            print_error("No records returned from query")
                            return False
                    else:
                        print_error(f"Query execution error: {query_result.get('error')}")
                        return False
                else:
                    print_error("No content returned from Cypher execution")
                    return False
            else:
                print_error(f"Cypher execution failed: {result.get('error')}")
                return False
                
        except Exception as e:
            print_error(f"Error testing Cypher execution: {e}")
            return False
    
    async def test_database_schema(self) -> bool:
        """Test database schema retrieval via MCP"""
        try:
            print_info("Testing database schema retrieval...")
            
            result = await self.send_mcp_message("tools/call", {
                "name": "get_database_schema",
                "arguments": {}
            })
            
            if "error" not in result and "result" in result:
                content = result["result"].get("content", [])
                if content and len(content) > 0:
                    schema_data = json.loads(content[0].get("text", "{}"))
                    
                    print_success("Database schema retrieved via MCP")
                    
                    if "error" not in schema_data:
                        # Show schema summary
                        node_labels = schema_data.get("node_labels", {})
                        rel_types = schema_data.get("relationship_types", {})
                        
                        if isinstance(node_labels, dict) and "labels" in node_labels:
                            labels = node_labels["labels"]
                            print_info(f"Node labels: {len(labels)} found")
                            if labels:
                                print_info(f"Labels: {', '.join(labels[:5])}{'...' if len(labels) > 5 else ''}")
                        
                        if isinstance(rel_types, dict) and "types" in rel_types:
                            types = rel_types["types"]
                            print_info(f"Relationship types: {len(types)} found")
                            if types:
                                print_info(f"Types: {', '.join(types[:5])}{'...' if len(types) > 5 else ''}")
                        
                        return True
                    else:
                        print_error(f"Schema retrieval error: {schema_data.get('error')}")
                        return False
                else:
                    print_error("No content returned from schema retrieval")
                    return False
            else:
                print_error(f"Schema retrieval failed: {result.get('error')}")
                return False
                
        except Exception as e:
            print_error(f"Error testing schema retrieval: {e}")
            return False
    
    async def test_connection_info(self) -> bool:
        """Test connection info retrieval"""
        try:
            print_info("Testing connection info retrieval...")
            
            result = await self.send_mcp_message("tools/call", {
                "name": "get_connection_info",
                "arguments": {}
            })
            
            if "error" not in result and "result" in result:
                content = result["result"].get("content", [])
                if content and len(content) > 0:
                    conn_info = json.loads(content[0].get("text", "{}"))
                    
                    print_success("Connection info retrieved via MCP")
                    print_json(conn_info, "Connection Info")
                    
                    # Show key connection details
                    print_info(f"URI: {conn_info.get('uri')}")
                    print_info(f"Database: {conn_info.get('database')}")
                    print_info(f"Status: {conn_info.get('status')}")
                    print_info(f"Driver Available: {conn_info.get('driver_available')}")
                    
                    return True
                else:
                    print_error("No content returned from connection info")
                    return False
            else:
                print_error(f"Connection info failed: {result.get('error')}")
                return False
                
        except Exception as e:
            print_error(f"Error testing connection info: {e}")
            return False


async def main():
    """Main test function"""
    print_header("MCP CLIENT TEST SUITE")
    print(f"{Colors.BOLD}Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    print_info("Testing MCP server connection via SSE endpoint")
    
    # Test configuration
    base_url = "http://localhost:8001"  # Change this if your app.py runs on different port
    
    results = {}
    
    async with MCPClient(base_url) as client:
        
        print_info(f"Using session ID: {client.session_id}")
        
        # Test 1: Server Health
        print_header("TEST 1: FASTAPI SERVER HEALTH")
        results['server_health'] = await client.test_server_health()
        
        if not results['server_health']:
            print_error("Cannot proceed - FastAPI server is not running")
            print_info("Make sure to start your app.py first: python app.py")
            return False
        
        # Test 2: MCP Initialize
        print_header("TEST 2: MCP INITIALIZATION")
        results['mcp_init'] = await client.test_mcp_initialize()
        
        # Test 3: List Tools
        print_header("TEST 3: MCP TOOLS LISTING")
        results['list_tools'] = await client.test_list_tools()
        
        # Test 4: Connection Health Tool
        print_header("TEST 4: CONNECTION HEALTH TOOL")
        results['health_tool'] = await client.test_connection_health_tool()
        
        # Test 5: Cypher Execution
        print_header("TEST 5: CYPHER EXECUTION VIA MCP")
        results['cypher_execution'] = await client.test_cypher_execution()
        
        # Test 6: Database Schema
        print_header("TEST 6: DATABASE SCHEMA VIA MCP")
        results['database_schema'] = await client.test_database_schema()
        
        # Test 7: Connection Info
        print_header("TEST 7: CONNECTION INFO VIA MCP")
        results['connection_info'] = await client.test_connection_info()
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"{Colors.BOLD}Test Results:{Colors.END}")
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        color = Colors.GREEN if success else Colors.RED
        symbol = "✅" if success else "❌"
        print(f"{color}{symbol} {test_name.replace('_', ' ').title()}: {status}{Colors.END}")
    
    print(f"\n{Colors.BOLD}Overall Result: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print_success("🎉 ALL TESTS PASSED!")
        print_success("✅ Your MCP server is working correctly via SSE")
        print_success("✅ Neo4j connection is healthy through MCP")
        print_success("✅ Ready for production use")
        return True
    else:
        print_error(f"❌ {total - passed} tests failed")
        print_warning("Check the errors above and ensure:")
        print_warning("1. app.py is running (python app.py)")
        print_warning("2. Neo4j database is accessible")
        print_warning("3. MCP server credentials are correct")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_error("\n❌ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"❌ Unexpected error: {e}")
        sys.exit(1)
