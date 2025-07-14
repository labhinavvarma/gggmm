import asyncio
import json
from mcp.client.sse import sse_client
from mcp import ClientSession


async def run():
    SERVER_URL = "http://localhost:8081/sse"

    print("=" * 60)
    print("🚀 MCP SSE CLIENT CONNECTION TEST")
    print("=" * 60)

    async with sse_client(url=SERVER_URL) as sse_connection:
        print("✓ Connected to SSE")

        async with ClientSession(*sse_connection) as session:
            await session.initialize()
            print("✓ Session Initialized\n")

            # === Run Tool Helper ===
            async def run_tool(tool_name, parameters=None):
                parameters = parameters or {}
                print(f"\n🛠️ Running tool: {tool_name}")
                print("-" * 60)
                try:
                    result = await session.query_tool(tool_name, parameters)
                    print(json.dumps(result.dict(), indent=2))
                except Exception as e:
                    print(f"❌ Failed to run {tool_name}: {e}")

            # === Call Neo4j Tools ===
            await run_tool("check_connection_health")
            await run_tool("get_connection_info")
            await run_tool("get_database_schema")
            await run_tool("execute_cypher", {
                "query": "MATCH (n) RETURN n LIMIT 2"
            })
            await run_tool("test_connectiq_queries")

    print("=" * 60)
    print("✅ COMPLETED MCP CLIENT TEST")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run())
