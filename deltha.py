# router.py
from fastapi import APIRouter
from mcpserver import UserInput, async_submit_medical_request, async_mcid_search, async_get_token, get_all_data

route = APIRouter()

@route.post("/medical/submit")
async def http_submit_medical(user: UserInput):
    return await async_submit_medical_request(user)

@route.post("/mcid/search")
async def http_mcid_search(user: UserInput):
    return await async_mcid_search(user)

@route.post("/token")
async def http_token():
    return await async_get_token()

@route.post("/all")
async def http_all(user: UserInput):
    return await get_all_data(**user.dict())
