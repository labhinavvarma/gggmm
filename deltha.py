from fastapi import FastAPI
from mcp import FastMCP, tool, prompt
from mcp.integrations.llms import LLMConfig
from pydantic import BaseModel
from typing import Optional, Dict, Any
import requests

# ------------------------------------------
# Configuration
# ------------------------------------------

BASE_URL = "https://milliman-api.example.com"
AUTH_TOKEN = "your-auth-token"

headers = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

# ------------------------------------------
# Request Models
# ------------------------------------------

class MemberInfo(BaseModel):
    first_name: str
    last_name: str
    ssn: str
    zip: str

# ------------------------------------------
# Tool: get_mcid
# ------------------------------------------

@tool(name="get_mcid", description="Get MCID using member info")
def get_mcid(info: MemberInfo) -> Dict[str, Any]:
    response = requests.post(f"{BASE_URL}/mcid", json=info.dict(), headers=headers)
    response.raise_for_status()
    return response.json()

# ------------------------------------------
# Tool: get_medical
# ------------------------------------------

class MCIDInput(BaseModel):
    mcid: str

@tool(name="get_medical", description="Get medical data for an MCID")
def get_medical(input: MCIDInput) -> Dict[str, Any]:
    response = requests.get(f"{BASE_URL}/medical/{input.mcid}", headers=headers)
    response.raise_for_status()
    return response.json()

# ------------------------------------------
# Tool: get_pharmacy
# ------------------------------------------

@tool(name="get_pharmacy", description="Get pharmacy data for an MCID")
def get_pharmacy(input: MCIDInput) -> Dict[str, Any]:
    response = requests.get(f"{BASE_URL}/pharmacy/{input.mcid}", headers=headers)
    response.raise_for_status()
    return response.json()

# ------------------------------------------
# Prompt: healthqa-prompt
# ------------------------------------------

@prompt(name="healthqa-prompt")
def health_qa_prompt(tools_output: Dict[str, Any], question: str) -> str:
    """
    Prompt to answer health questions using outputs from get_mcid, get_medical, and get_pharmacy.
    """
    return f"""
You are a health assistant.

Answer the question below using the following information:

MCID Info:
{tools_output.get("get_mcid", "N/A")}

Medical Info:
{tools_output.get("get_medical", "N/A")}

Pharmacy Info:
{tools_output.get("get_pharmacy", "N/A")}

User Question:
{question}
"""

# ------------------------------------------
# Server Setup
# ------------------------------------------

app = FastAPI()
mcp = FastMCP(app=app)

# Mount the MCP streaming endpoint
from mcp.transport.sse import SseServerTransport
mcp.include_transport(SseServerTransport(endpoint="/messages"))

# Run manually if needed
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

