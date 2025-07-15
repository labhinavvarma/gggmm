# verify_fix.py - Verify the MCP server fix works

import asyncio
import nest_asyncio
import json
from fastmcp import Client

nest_asyncio.apply()

async def verify_mcp_fix():
    """Verify that the MCP server now returns actual data."""
    print("=" * 60)
    print("VERIFYING MCP SERVER FIX")
    print("=" * 60)
    
    try:
        async with Client("mcpserver.py") as client:
            print("✅ Connected to fixed MCP server")
            
            # Test simple_test tool
            print("\n🧪 Testing simple_test tool:")
            result = await client.call_tool("simple_test", {})
            print(f"  Raw result type: {type(result)}")
            print(f"  Raw result: {result}")
            
            # Extract content
            if hasattr(result, 'content') and result.content:
                content_text = result.content[0].text
                print(f"  ✅ Extracted text: {content_text}")
                
                # Try to parse as JSON
                try:
                    parsed = json.loads(content_text)
                    print(f"  ✅ Parsed JSON: {json.dumps(parsed, indent=2)}")
                    print(f"  ✅ Message: {parsed.get('message', 'N/A')}")
                except:
                    print(f"  ❌ Not valid JSON: {content_text}")
            
            # Test count_nodes tool
            print("\n🔢 Testing count_nodes tool:")
            result = await client.call_tool("count_nodes", {})
            if hasattr(result, 'content') and result.content:
                content_text = result.content[0].text
                print(f"  ✅ Extracted text: {content_text}")
                
                try:
                    parsed = json.loads(content_text)
                    print(f"  ✅ Parsed JSON: {json.dumps(parsed, indent=2)}")
                    print(f"  ✅ Total nodes: {parsed.get('total_nodes', 'N/A')}")
                except:
                    print(f"  ❌ Not valid JSON: {content_text}")
            
            # Test health_check tool
            print("\n🏥 Testing health_check tool:")
            result = await client.call_tool("health_check", {})
            if hasattr(result, 'content') and result.content:
                content_text = result.content[0].text
                print(f"  ✅ Extracted text: {content_text}")
                
                try:
                    parsed = json.loads(content_text)
                    print(f"  ✅ Parsed JSON: {json.dumps(parsed, indent=2)}")
                    print(f"  ✅ Status: {parsed.get('status', 'N/A')}")
                except:
                    print(f"  ❌ Not valid JSON: {content_text}")
            
            print("\n" + "=" * 60)
            print("✅ VERIFICATION COMPLETE!")
            print("If you see actual JSON data above, the fix worked!")
            print("=" * 60)
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()

def run_verification():
    """Run verification with proper async handling."""
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(verify_mcp_fix())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(verify_mcp_fix())
        finally:
            loop.close()

if __name__ == "__main__":
    print("Verifying MCP server fix...")
    print("Expected: Actual JSON data instead of ToolResult object strings")
    print()
    run_verification()
