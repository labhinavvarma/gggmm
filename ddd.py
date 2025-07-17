from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router import router
from langgraph_agent import build_agent
import uvicorn

"""
FastAPI app that:
- Serves /chat (Streamlit calls this endpoint).
- Runs LangGraph agent and returns both reasoning and answer.
"""

app = FastAPI(title="Neo4j LLM LangGraph App")

# Allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
agent = build_agent()

@app.post("/chat")
async def chat(payload: dict):
    """
    Receives a user question, runs the LangGraph agent, and returns both the answer and trace.
    """
    question = payload.get("question", "")
    session_id = payload.get("session_id", "")
    result = await agent.ainvoke({"question": question, "session_id": session_id})
    return {"answer": result.get("answer", "No response"), "trace": result.get("trace")}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8081, reload=True)
