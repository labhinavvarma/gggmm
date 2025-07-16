# Updated Pydantic model for ChatRequest
class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1, max_length=1000)
    chat_history: Optional[List[Dict[str, str]]] = Field(default_factory=list)  # Default to empty list

# Fixed chat endpoint
@app.post("/chat")
async def chat_with_analysis(chat_request: ChatRequest):
    """Chat with medical analysis data - works with empty chat history"""
    check_agent_available()
    
    session_id = chat_request.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    if session_data["status"] != "completed" or not session_data.get("results"):
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    results = session_data["results"]
    if not results.get("chatbot_ready", False):
        raise HTTPException(status_code=400, detail="Chatbot not ready")
    
    try:
        chatbot_context = results.get("chatbot_context", {})
        
        # Ensure chat_history is always a list (handle None or missing)
        chat_history = chat_request.chat_history or []
        
        # Debug logging
        logger.info(f"💬 Chat question: {chat_request.question}")
        logger.info(f"📝 Chat history length: {len(chat_history)}")
        logger.info(f"📊 Context available: {bool(chatbot_context)}")
        
        # Validate chatbot context
        if not chatbot_context:
            logger.error("❌ No chatbot context available")
            return {
                "success": False,
                "session_id": session_id,
                "response": "Chat context not available. Please run analysis first.",
                "updated_chat_history": chat_history
            }
        
        # Call the health agent chat method
        response = health_agent.chat_with_data(
            chat_request.question,
            chatbot_context,
            chat_history  # This is now guaranteed to be a list
        )
        
        # Validate response
        if not response or response.strip() == "":
            logger.error("❌ Empty response from health agent")
            response = "I'm sorry, I couldn't generate a response. Please try rephrasing your question."
        elif '"detail"' in response or response.startswith("Error"):
            logger.error(f"❌ Error in response: {response}")
            response = "I encountered an error processing your question. Please try a different question about the medical analysis."
        
        # Create updated chat history
        updated_history = chat_history + [
            {"role": "user", "content": chat_request.question},
            {"role": "assistant", "content": response}
        ]
        
        logger.info(f"✅ Chat successful, response length: {len(response)}")
        
        return {
            "success": True,
            "session_id": session_id,
            "response": response,
            "updated_chat_history": updated_history
        }
        
    except Exception as e:
        logger.error(f"❌ Chat failed with exception: {str(e)}")
        
        # Provide helpful error response
        error_response = f"I encountered an error: {str(e)}. Please try a simpler question about the medical analysis."
        
        return {
            "success": False,
            "session_id": session_id,
            "response": error_response,
            "updated_chat_history": chat_history + [
                {"role": "user", "content": chat_request.question},
                {"role": "assistant", "content": error_response}
            ]
        }

# Alternative simple chat endpoint (no chat history required)
@app.post("/chat-simple")
async def simple_chat(session_id: str = Field(...), question: str = Field(...)):
    """Simple chat endpoint without chat history requirement"""
    try:
        # Create a ChatRequest object with empty chat history
        chat_request = ChatRequest(
            session_id=session_id,
            question=question,
            chat_history=[]  # Always empty for simple chat
        )
        
        # Use the main chat function
        return await chat_with_analysis(chat_request)
        
    except Exception as e:
        return {
            "success": False,
            "session_id": session_id,
            "response": f"Simple chat error: {str(e)}",
            "updated_chat_history": []
        }

# Test chat endpoint 
@app.get("/test-chat/{session_id}")
async def test_chat_endpoint(session_id: str):
    """Test if chat is working for a session"""
    try:
        test_request = ChatRequest(
            session_id=session_id,
            question="Hello, are you working?",
            chat_history=[]
        )
        
        result = await chat_with_analysis(test_request)
        return {
            "test_successful": result["success"],
            "session_id": session_id,
            "test_response": result["response"]
        }
        
    except Exception as e:
        return {
            "test_successful": False,
            "session_id": session_id,
            "error": str(e)
        }
