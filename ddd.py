import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession
import json


async def run():
    SERVER_URL = "http://localhost:8001/sse"
    print("=" * 60)
    print("🚀 MCP SSE CLIENT CONNECTION TEST")
    print("=" * 60)

    async with sse_client(url=SERVER_URL) as sse_connection:
        print("✓ Connected to SSE")

        async with ClientSession(*sse_connection) as session:
            await session.initialize()
            print("✓ Session Initialized\n")

            # === List Tools ===
            print("🔧 TOOLS")
            print("-" * 60)
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"• {tool.name} - {tool.description or 'No description'}")

            # === List Resources ===
            print("\n📦 RESOURCES")
            print("-" * 60)
            resources = await session.list_resources()
            for resource in resources.resources:
                print(f"• {resource.name} - {resource.description or 'No description'}")

            # === List Prompts ===
            print("\n💬 PROMPTS")
            print("-" * 60)
            try:
                prompts = await session.list_prompts()
                for prompt in prompts.prompts:
                    print(f"• {prompt.name} - {prompt.description or 'No description'}")
            except Exception as e:
                print(f"❌ Could not list prompts: {e}")

            # === Run Health Check ===
            print("\n🧠 TEST: Neo4j Health Check")
            print("-" * 60)
            result = await session.invoke_tool("check_connection_health", {})
            print(json.dumps(result.dict(), indent=2))

            # === Run Schema Fetch ===
            print("\n🧠 TEST: Get Database Schema")
            print("-" * 60)
            result = await session.invoke_tool("get_database_schema", {})
            print(json.dumps(result.dict(), indent=2))

            # === Run Sample Query ===
            print("\n🧠 TEST: Run Cypher Query")
            print("-" * 60)
            result = await session.invoke_tool("execute_cypher", {
                "query": "MATCH (n) RETURN n LIMIT 2"
            })
            print(json.dumps(result.dict(), indent=2))

            # === Run Connectiq Tests ===
            print("\n🧪 TEST: Connectiq Validation")
            print("-" * 60)
            result = await session.invoke_tool("test_connectiq_queries", {})
            print(json.dumps(result.dict(), indent=2))

    print("=" * 60)
    print("✅ COMPLETED MCP CLIENT TEST")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run())
