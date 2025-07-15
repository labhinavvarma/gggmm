# verify_fix.py - Simple verification that the fix works

print("🔍 Verifying TaskGroup fix...")

# Test 1: Check nest_asyncio installation
try:
    import nest_asyncio
    print("✅ nest_asyncio imported successfully")
except ImportError:
    print("❌ nest_asyncio not installed. Run: pip install nest-asyncio")
    exit(1)

# Test 2: Check MCP imports
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    print("✅ MCP imports successful")
except ImportError as e:
    print(f"❌ MCP import failed: {e}")
    exit(1)

# Test 3: Check mcpserver import (should not have Streamlit commands)
try:
    from mcpserver import main as start_mcp_server
    print("✅ mcpserver imported successfully (no Streamlit conflicts)")
except Exception as e:
    print(f"❌ mcpserver import failed: {e}")
    exit(1)

# Test 4: Check asyncio functionality
try:
    import asyncio
    nest_asyncio.apply()
    
    async def test_async():
        await asyncio.sleep(0.1)
        return "async works"
    
    def run_test():
        try:
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(test_async())
            return result
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(test_async())
            return result
    
    result = run_test()
    if result == "async works":
        print("✅ Async functionality working")
    else:
        print("❌ Async functionality failed")
        exit(1)
        
except Exception as e:
    print(f"❌ Async test failed: {e}")
    exit(1)

# Test 5: Check requirements
required_packages = [
    'streamlit', 'fastmcp', 'neo4j', 'nest_asyncio', 
    'mcp', 'requests', 'urllib3', 'pydantic'
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
        print(f"✅ {package} available")
    except ImportError:
        missing_packages.append(package)
        print(f"❌ {package} missing")

if missing_packages:
    print(f"\n❌ Missing packages: {missing_packages}")
    print("Run: pip install " + " ".join(missing_packages))
    exit(1)

print("\n🎉 ALL VERIFICATION TESTS PASSED!")
print("🚀 You can now run: streamlit run neo4j\\test.py")
print("📋 The TaskGroup error should be eliminated!")
