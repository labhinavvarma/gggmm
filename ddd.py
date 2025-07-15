# test_mcp_direct.py - Direct test of MCP server to see what's being returned

import asyncio
import nest_asyncio
import json
from fastmcp import Client

# Apply nest_asyncio
nest_asyncio.apply()

async def test_mcp_server_direct():
    """Test MCP server directly to see exactly what it returns."""
    print("=" * 80)
    print("DIRECT MCP SERVER TEST")
    print("=" * 80)
    
    try:
        print("🔧 Connecting to MCP server...")
        async with Client("mcpserver_debug.py") as client:
            print("✅ Connected successfully")
            
            print("\n" + "="*50)
            print("TEST 1: List Tools")
            print("="*50)
            tools = await client.list_tools()
            print(f"Tools type: {type(tools)}")
            if hasattr(tools, 'tools'):
                print(f"Number of tools: {len(tools.tools)}")
                for tool in tools.tools:
                    print(f"  - {tool.name}: {tool.description}")
            
            print("\n" + "="*50)
            print("TEST 2: Simple Test Tool")
            print("="*50)
            result = await client.call_tool("simple_test", {})
            print(f"Raw result type: {type(result)}")
            print(f"Raw result: {result}")
            print(f"Raw result repr: {repr(result)}")
            
            # Detailed analysis
            if isinstance(result, list):
                print(f"Result is a list with {len(result)} items")
                for i, item in enumerate(result):
                    print(f"  Item {i}: {type(item)}")
                    print(f"  Item {i} repr: {repr(item)}")
                    
                    if hasattr(item, 'content'):
                        print(f"  Item {i} has content: {type(item.content)}")
                        if item.content:
                            for j, content_item in enumerate(item.content):
                                print(f"    Content {j}: {type(content_item)}")
                                if hasattr(content_item, 'text'):
                                    print(f"    Content {j} text: {content_item.text}")
                                if hasattr(content_item, 'type'):
                                    print(f"    Content {j} type: {content_item.type}")
            
            print("\n" + "="*50)
            print("TEST 3: Health Check")
            print("="*50)
            result = await client.call_tool("health_check", {})
            print(f"Health check result type: {type(result)}")
            print(f"Health check result: {result}")
            
            # Extract actual content
            if isinstance(result, list) and len(result) > 0:
                tool_result = result[0]
                if hasattr(tool_result, 'content') and tool_result.content:
                    text_content = tool_result.content[0]
                    if hasattr(text_content, 'text'):
                        actual_text = text_content.text
                        print(f"✅ EXTRACTED TEXT: {actual_text}")
                        
                        # Try to parse as JSON
                        try:
                            parsed_json = json.loads(actual_text)
                            print(f"✅ PARSED JSON: {json.dumps(parsed_json, indent=2)}")
                        except:
                            print("❌ Not valid JSON")
            
            print("\n" + "="*50)
            print("TEST 4: Count Nodes")
            print("="*50)
            result = await client.call_tool("count_nodes", {})
            print(f"Count nodes result type: {type(result)}")
            
            # Extract and display
            if isinstance(result, list) and len(result) > 0:
                tool_result = result[0]
                if hasattr(tool_result, 'content') and tool_result.content:
                    text_content = tool_result.content[0]
                    if hasattr(text_content, 'text'):
                        actual_text = text_content.text
                        print(f"✅ COUNT NODES TEXT: {actual_text}")
                        
                        try:
                            parsed_json = json.loads(actual_text)
                            print(f"✅ COUNT NODES JSON: {json.dumps(parsed_json, indent=2)}")
                            print(f"✅ TOTAL NODES: {parsed_json.get('total_nodes', 'N/A')}")
                        except:
                            print("❌ Count nodes result not valid JSON")
            
            print("\n" + "="*50)
            print("TEST 5: Read Cypher Query")
            print("="*50)
            result = await client.call_tool("read_neo4j_cypher", {"query": "RETURN 1 as test"})
            print(f"Read cypher result type: {type(result)}")
            
            # Extract and display
            if isinstance(result, list) and len(result) > 0:
                tool_result = result[0]
                if hasattr(tool_result, 'content') and tool_result.content:
                    text_content = tool_result.content[0]
                    if hasattr(text_content, 'text'):
                        actual_text = text_content.text
                        print(f"✅ CYPHER RESULT TEXT: {actual_text}")
                        
                        try:
                            parsed_json = json.loads(actual_text)
                            print(f"✅ CYPHER RESULT JSON: {json.dumps(parsed_json, indent=2)}")
                        except:
                            print("❌ Cypher result not valid JSON")
            
            print("\n" + "=" * 80)
            print("✅ ALL TESTS COMPLETED")
            print("=" * 80)
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def run_test():
    """Run the test with proper async handling."""
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(test_mcp_server_direct())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_mcp_server_direct())
        finally:
            loop.close()

if __name__ == "__main__":
    print("Testing MCP server directly...")
    print("This will show exactly what the server returns.")
    print()
    run_test()
