# mcpserver.py
import os
import requests
import httpx
import asyncio
from typing import Dict, Any, List
from pydantic import BaseModel
from fastmcp import FastMCP
from mcp.types import Message

# Initialize MCP
mcp = FastMCP("Milliman MCP Server")

# === Config ===
TOKEN_URL = os.getenv("TOKEN_URL", "https://securefed.antheminc.com/as/token.oauth2")
TOKEN_PAYLOAD = {
    'grant_type': 'client_credentials',
    'client_id': os.getenv("CLIENT_ID", "MILLIMAN"),
    'client_secret': os.getenv("CLIENT_SECRET", "your-client-secret")
}
TOKEN_HEADERS = {'Content-Type': 'application/x-www-form-urlencoded'}
MEDICAL_URL = os.getenv("MEDICAL_URL", "https://your-medical-url")
MCID_URL = os.getenv("MCID_URL", "https://hix-clm-internaltesting-prod.anthem.com/medical")

# === Schema ===
class UserInput(BaseModel):
    first_name: str
    last_name: str
    ssn: str
    date_of_birth: str
    gender: str
    zip_code: str

# === Helpers ===
def get_access_token_sync():
    try:
        response = requests.post(TOKEN_URL, data=TOKEN_PAYLOAD, headers=TOKEN_HEADERS)
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception as e:
        print("Token error:", str(e))
        return None

async def async_get_token():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(TOKEN_URL, data=TOKEN_PAYLOAD, headers=TOKEN_HEADERS)
            return {'status_code': response.status_code, 'body': response.json()}
        except Exception as e:
            return {'status_code': 500, 'error': str(e)}

async def async_submit_medical_request(user: UserInput):
    access_token = await asyncio.to_thread(get_access_token_sync)
    if not access_token:
        return {'status_code': 500, 'error': 'Access token not found'}

    headers = {
        'Authorization': f"Bearer {access_token}",
        'Content-Type': 'application/json'
    }
    payload = {
        "requestId": "77554079",
        "firstName": user.first_name,
        "lastName": user.last_name,
        "ssn": user.ssn,
        "dateOfBirth": user.date_of_birth,
        "gender": user.gender,
        "zipCodes": [user.zip_code],
        "callerId": "Milliman-Test16"
    }
    response = requests.post(MEDICAL_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return {'status_code': response.status_code, 'error': response.text}
    return {'status_code': 200, 'body': response.json()}

async def async_mcid_search(user: UserInput):
    access_token = await asyncio.to_thread(get_access_token_sync)
    if not access_token:
        return {'status_code': 500, 'error': 'Access token not found'}

    headers = {
        'Content-Type': 'application/json',
        'Apiuser': 'MillimanUser',
        'Authorization': f"Bearer {access_token}"
    }
    payload = {
        "requestID": "1",
        "processStatus": {"completed": "false", "isMemput": "false"},
        "consumer": [{
            "fname": user.first_name,
            "lname": user.last_name,
            "sex": user.gender,
            "dob": user.date_of_birth.replace("-", ""),
            "addressList": [{"type": "P", "zip": user.zip_code}],
            "id": {"ssn": user.ssn}
        }],
        "searchSetting": {"minScore": "100", "maxResult": "1"}
    }
    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.post(MCID_URL, headers=headers, json=payload, timeout=30)
            return {'status_code': response.status_code, 'body': response.json()}
        except Exception as e:
            return {'status_code': 500, 'error': str(e)}

# === MCP Tools ===
@mcp.tool()
async def get_token() -> Dict[str, Any]:
    return await async_get_token()

@mcp.tool()
async def medical_submit(**kwargs) -> Dict[str, Any]:
    user = UserInput(**kwargs)
    return await async_submit_medical_request(user)

@mcp.tool()
async def mcid_search(**kwargs) -> Dict[str, Any]:
    user = UserInput(**kwargs)
    return await async_mcid_search(user)

@mcp.tool()
async def get_all_data(**kwargs) -> Dict[str, Any]:
    user = UserInput(**kwargs)
    token_task = async_get_token()
    mcid_task = async_mcid_search(user)
    medical_task = async_submit_medical_request(user)
    token_result, mcid_result, medical_result = await asyncio.gather(
        token_task, mcid_task, medical_task
    )
    return {
        "get_token": token_result,
        "mcid_search": mcid_result,
        "medical_submit": medical_result
    }

# === MCP Prompt ===
@mcp.prompt(name="milliman-summary", description="Summarize Milliman workflows and inputs")
async def summary_prompt(query: str) -> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""
You are an expert assistant for Milliman claims processing.
Help the user understand how to:
- Use tools like `mcid_search`, `medical_submit`, or `get_all_data`
- Clarify input fields like SSN, DOB, gender, zip code
Query: {query}
"""
        }
    ]
