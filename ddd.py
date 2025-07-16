@app.post("/chat")
async def chat_with_analysis(chat_request: ChatRequest):
    """Chat with medical analysis data"""
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
        
        # Debug logging
        logger.info(f"💬 Chat request: {chat_request.question}")
        logger.info(f"📊 Context keys: {list(chatbot_context.keys())}")
        
        response = health_agent.chat_with_data(
            chat_request.question,
            chatbot_context,
            chat_request.chat_history
        )
        
        # Check if response is empty or error
        if not response or response.strip() == "" or '"detail"' in response:
            logger.error(f"❌ Empty or error response: {response}")
            response = "I'm having trouble processing your request. Please try a different question or check if the analysis completed successfully."
        
        updated_history = chat_request.chat_history + [
            {"role": "user", "content": chat_request.question},
            {"role": "assistant", "content": response}
        ]
        
        return {
            "success": True,
            "session_id": session_id,
            "response": response,
            "updated_chat_history": updated_history
        }
        
    except Exception as e:
        logger.error(f"❌ Chat failed: {str(e)}")
        return {
            "success": False,
            "session_id": session_id,
            "response": f"Chat error: {str(e)}",
            "updated_chat_history": chat_request.chat_history
        }
