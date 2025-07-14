
import asyncio
import json
import uuid
import httpx

# MCP Server configuration
SERVER_URL = "http://localhost:8001"
MESSAGES_ENDPOINT = f"{SERVER_URL}/messages/"
SESSION_ID = str(uuid.uuid4())  # Unique session ID per run

# === Helper to post message ===
async def send_mcp_tool_request(tool_name: str, parameters: dict = None):
    message = {
        "session_id": SESSION_ID,
        "input": {
            "tool": tool_name,
            "parameters": parameters or {}
        }
    }

    async with httpx.AsyncClient(timeout=60) as client:
        print(f"\n🔄 Invoking MCP Tool: `{tool_name}`...")
        response = await client.post(MESSAGES_ENDPOINT, json=message)
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"✅ Tool `{tool_name}` Result:\n", json.dumps(data, indent=2))
            except Exception as e:
                print(f"❌ Failed to decode JSON: {e}")
                print(response.text)
        else:
            print(f"❌ Tool `{tool_name}` failed with status {response.status_code}")
            print(response.text)

# === Main Test Sequence ===
async def run_tests():
    print(f"🚀 Testing MCP Server at {SERVER_URL} with session {SESSION_ID}")

    # 1. Check Neo4j Health
    await send_mcp_tool_request("check_connection_health")

    # 2. Run Sample Cypher Query
    await send_mcp_tool_request("execute_cypher", {
        "query": "MATCH (n) RETURN n LIMIT 2"
    })

    # 3. Get Schema
    await send_mcp_tool_request("get_database_schema")

    # 4. Get Connection Info
    await send_mcp_tool_request("get_connection_info")

    # 5. Run Connectiq-Specific Tests
    await send_mcp_tool_request("test_connectiq_queries")


if __name__ == "__main__":
    asyncio.run(run_tests())
