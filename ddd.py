@router.post("/v2/complete")
async def llm_gateway(
        api_key: Annotated[str | None, Header()],
        query: Annotated[AiCompleteQryModel, Body(embed=True)],
        config: Annotated[GenAiEnvSettings, Depends(get_config)],
        logger: Annotated[Logger, Depends(get_logger)],
        background_tasks: BackgroundTasks,
        get_load_datetime: Annotated[datetime, Depends(get_load_timestamp)]          
):
    # Initialize log capture - this will store ALL logs throughout the process
    captured_logs = []
    
    # Helper function to capture logs - accessible throughout the function
    def capture_log(level, message):
        """Capture log entry and also log normally"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.upper(),
            "message": str(message)
        }
        captured_logs.append(log_entry)
        
        # Also log normally to system
        if level.upper() == "INFO":
            logger.info(message)
        elif level.upper() == "ERROR":
            logger.error(message)
        elif level.upper() == "DEBUG":
            logger.debug(message)
        elif level.upper() == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)
    
    # Start capturing logs immediately
    capture_log("INFO", "=== LLM Gateway Request Started ===")
    capture_log("INFO", f"Request received at {datetime.utcnow().isoformat()}")
    
    try:
        prompt = query.prompt.messages[-1].content
        messages_json = query.prompt.messages
        session_id = str(uuid.uuid4())  # Generate proper session ID instead of hardcoded
        
        capture_log("INFO", f"Generated session ID: {session_id}")
        capture_log("INFO", f"Processing request for model: {query.model.model}")
        capture_log("DEBUG", f"Prompt length: {len(prompt)} characters")
        capture_log("DEBUG", f"Application code: {query.application.aplctn_cd}")
        
        # The API key validation and generation has been pushed to backend; the api_validator will return True if API key is valid for the application.
        api_validator = ValidApiKey()
        
        capture_log("INFO", "Starting API key validation")
        
        if api_validator(api_key, query.application.aplctn_cd, query.application.app_id):
            capture_log("INFO", "✅ API key validation successful")
            
            try:
                capture_log("DEBUG", "Establishing Snowflake connection...")
                sf_conn = SnowFlakeConnector.get_conn(
                    query.application.aplctn_cd,
                    query.application.app_lvl_prefix,
                    session_id
                )
                capture_log("INFO", "✅ Snowflake connection established successfully")
            except DatabaseError as e:
                capture_log("ERROR", f"❌ Database connection failed: {str(e)}")
                
                error_response = {
                    "error": "Database connection failed",
                    "detail": "User not authorized to resources",
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "logs": captured_logs,  # Include all captured logs
                    "log_count": len(captured_logs)
                }
                
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=json.dumps(error_response, indent=2)
                )
            
            capture_log("DEBUG", "Preparing request configuration...")
            pre_response_format = {"response_format": query.response_format.model_dump()} if query.response_format.schema else {}
            
            # Add provisioned throughput if model is in provisioned models list
            provisioned_dict = {"provisioned_throughput_id": config.provisioned_id} if query.model.model in config.provisioned_models else {}
            
            if provisioned_dict:
                capture_log("INFO", "Using provisioned throughput")
            else:
                capture_log("DEBUG", "No provisioned throughput configured")
            
            request_body = {
                "model": query.model.model,
                "messages": [
                    {
                        "role": "user",
                        "content": get_conv_response(messages_json) + prompt
                    }
                ],
                **query.model.options,
                **pre_response_format,
                **provisioned_dict
            }
            
            capture_log("DEBUG", f"Request body prepared with {len(request_body)} fields")
            print("req", request_body)
            
            headers = {
                "Authorization": f'Snowflake Token="{sf_conn.rest.token}"',
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            url = getattr(config.COMPLETE, "{}_host".format(config.env))
            
            capture_log("INFO", f"Target URL configured: {url}")

            # ------------------------------------------------------------------
            # Preflight request BEFORE streaming (returns real 4xx/5xx)
            # ------------------------------------------------------------------
            capture_log("INFO", "🚀 Starting preflight check...")
            
            preflight_timeout = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=preflight_timeout, verify=True) as pre_client:
                try:
                    pre = await pre_client.post(url, headers=headers, json=request_body)
                    capture_log("INFO", f"✅ Preflight response received - Status: {pre.status_code}")
                except httpx.RequestError as e:
                    capture_log("ERROR", f"❌ Preflight request failed: {str(e)}")
                    
                    error_response = {
                        "error": "Service unavailable",
                        "detail": f"Service unavailable: {str(e)}",
                        "session_id": session_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "logs": captured_logs,
                        "log_count": len(captured_logs)
                    }
                    
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=json.dumps(error_response, indent=2)
                    )
                    
            if pre.is_client_error:
                capture_log("ERROR", f"❌ Preflight client error: {pre.status_code}")
                error_message = pre.text or ""
                status_code = pre.status_code
                
                if status_code == 429:
                    detail = "Too many requests. Please implement retry with backoff. Original message: " + error_message
                elif status_code == 402:
                    detail = "Budget exceeded. Check your Cortex token quota. " + error_message
                elif status_code == 400:
                    detail = "Bad request sent to Snowflake Cortex. " + error_message
                else:
                    detail = error_message
                
                capture_log("ERROR", f"Client error details: {detail}")
                
                error_response = {
                    "error": "Client error",
                    "detail": detail,
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "logs": captured_logs,
                    "log_count": len(captured_logs)
                }
                
                raise HTTPException(
                    status_code=status_code,
                    detail=json.dumps(error_response, indent=2)
                )
                
            if pre.is_server_error:
                capture_log("ERROR", f"❌ Preflight server error: {pre.status_code}")
                
                error_response = {
                    "error": "Server error",
                    "detail": pre.text or "",
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "logs": captured_logs,
                    "log_count": len(captured_logs)
                }
                
                raise HTTPException(
                    status_code=pre.status_code,
                    detail=json.dumps(error_response, indent=2)
                )
            
            capture_log("INFO", "✅ Preflight check passed successfully")
            
            response_text = []
            query_id = [None]
            fdbck_id = [str(uuid.uuid4())]
            
            capture_log("INFO", f"Generated feedback ID: {fdbck_id[0]}")
            capture_log("INFO", "Initializing streaming response...")
            
            async def data_streamer():
                """
                Collect all data internally and return only final JSON response with logs.
                """
                capture_log("INFO", "🔄 Data streaming started")
                
                vModel = query.model.model
                Created = datetime.utcnow().isoformat()
                usage_info = {
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0
                }
                
                stream_timeout = httpx.Timeout(connect=5.0, read=None, write=10.0, pool=10.0)
                async with httpx.AsyncClient(timeout=stream_timeout, verify=True) as clnt:
                    try:
                        async with clnt.stream('POST', url, headers=headers, json=request_body) as response:
                            capture_log("INFO", f"🌊 Stream connection established - Status: {response.status_code}")
                            
                            if response.is_client_error or response.is_server_error:
                                error_message = await response.aread()
                                decoded_error = error_message.decode("utf-8", errors="replace")
                                capture_log("ERROR", f"❌ Upstream stream error {response.status_code}: {decoded_error}")
                                
                                # Return error as final JSON with logs
                                error_response = {
                                    "error": "upstream_error", 
                                    "status": response.status_code, 
                                    "detail": decoded_error,
                                    "query_id": query_id[0],
                                    "fdbck_id": fdbck_id[0],
                                    "session_id": session_id,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "logs": captured_logs,
                                    "log_count": len(captured_logs)
                                }
                                yield json.dumps(error_response, indent=2)
                                return
                            
                            # Collect all streaming data internally - NO YIELDING DURING COLLECTION
                            capture_log("DEBUG", "📦 Starting to process streaming chunks...")
                            buffer = b""
                            chunk_count = 0
                            
                            async for result_chunk in response.aiter_bytes():
                                buffer += result_chunk
                                while b'\n\n' in buffer:
                                    elem, buffer = buffer.split(b'\n\n', 1)
                                    chunk_count += 1
                                    
                                    if b'content' in elem or b'text' in elem:
                                        try:
                                            chunk_dict = json.loads(elem.replace(b'data: ', b''))
                                            
                                            # Extract query ID if available
                                            if 'id' in chunk_dict:
                                                query_id[0] = chunk_dict['id']
                                                capture_log("DEBUG", f"🆔 Updated query ID: {query_id[0]}")
                                            
                                            # Update usage information if available
                                            if 'usage' in chunk_dict:
                                                usage_info.update(chunk_dict['usage'])
                                                capture_log("DEBUG", f"📊 Updated usage info: {usage_info}")
                                            
                                            # Extract content based on Snowflake Cortex format
                                            text_content = None
                                            if 'choices' in chunk_dict and len(chunk_dict['choices']) > 0:
                                                choice = chunk_dict['choices'][0]
                                                if 'delta' in choice:
                                                    delta = choice['delta']
                                                    if 'text' in delta:
                                                        text_content = delta['text']
                                                    elif 'content' in delta:
                                                        text_content = delta['content']
                                                elif 'message' in choice:
                                                    message = choice['message']
                                                    if 'content' in message:
                                                        text_content = message['content']
                                            
                                            # ONLY COLLECT - NO YIELDING OF TEXT
                                            if text_content:
                                                response_text.append(text_content)
                                                if chunk_count % 10 == 0:  # Log every 10th chunk to avoid spam
                                                    capture_log("DEBUG", f"📝 Processed chunk {chunk_count}, content length: {len(text_content)}")
                                        
                                        except json.JSONDecodeError as e:
                                            capture_log("ERROR", f"❌ Error decoding JSON chunk {chunk_count}: {e}")
                                            # Don't yield errors during collection - just log and continue
                                            continue
                            
                            capture_log("INFO", f"✅ Streaming completed! Processed {chunk_count} chunks, total response length: {len(''.join(response_text))}")
                    
                    except httpx.RequestError as e:
                        capture_log("ERROR", f"❌ Request error during streaming: {e}")
                        
                        error_response = {
                            "error": "Request error", 
                            "detail": str(e),
                            "query_id": query_id[0],
                            "fdbck_id": fdbck_id[0],
                            "session_id": session_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "logs": captured_logs,
                            "log_count": len(captured_logs)
                        }
                        yield json.dumps(error_response, indent=2)
                        return
                        
                    except Exception as e:
                        capture_log("ERROR", f"❌ Unexpected error during streaming: {e}")
                        
                        error_response = {
                            "error": "Unexpected error", 
                            "detail": str(e),
                            "query_id": query_id[0],
                            "fdbck_id": fdbck_id[0],
                            "session_id": session_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "logs": captured_logs,
                            "log_count": len(captured_logs)
                        }
                        yield json.dumps(error_response, indent=2)
                        return
                
                # ONLY YIELD THE FINAL JSON RESPONSE WITH LOGS - NOTHING ELSE
                full_final_response = "".join(response_text)
                capture_log("INFO", f"🎉 Processing completed successfully! Query ID: {query_id[0]}")
                capture_log("INFO", f"📋 Final response ready with {len(captured_logs)} log entries")
                
                # Create the final response with ALL captured logs
                final_response = {
                    "model": vModel,
                    "created": Created,
                    "query_id": query_id[0],
                    "fdbck_id": fdbck_id[0],
                    "session_id": session_id,
                    "content": full_final_response,
                    "usage": usage_info,
                    "tool_use": {},
                    "logs": captured_logs,  # THIS IS WHERE ALL LOGS ARE INCLUDED
                    "metadata": {
                        "total_log_entries": len(captured_logs),
                        "response_length": len(full_final_response),
                        "chunks_processed": chunk_count,
                        "log_levels": list(set(log.get("level", "UNKNOWN") for log in captured_logs)),
                        "processing_completed_at": datetime.utcnow().isoformat()
                    }
                }
                
                capture_log("INFO", "📤 Sending final response with complete log history")
                
                # SINGLE YIELD - FINAL JSON WITH ALL LOGS
                yield json.dumps(final_response, indent=2)
                
                # Background audit task
                capture_log("DEBUG", "🔧 Preparing background audit task")
                token_count = usage_info.get("total_tokens", 0)
                audit_rec = GenAiCortexAudit(
                    edl_load_dtm=get_load_datetime,
                    edl_run_id="0000",
                    edl_scrty_lvl_cd="NA",
                    edl_lob_cd="NA",
                    srvc_type="complete",
                    aplctn_cd=config.pltfrm_aplctn_cd,
                    user_id="Complete_User",
                    mdl_id=vModel,
                    cnvrstn_chat_lmt_txt="0",
                    sesn_id=session_id,
                    prmpt_txt=prompt.replace("'", "\\'"),
                    tkn_cnt=str(token_count),
                    feedbk_actn_txt="",
                    feedbk_cmnt_txt="",
                    feedbk_updt_dtm=get_load_datetime,
                )
                
                capture_log("INFO", "✅ Background audit task scheduled")
                
                background_tasks.add_task(
                    log_response,
                    audit_rec,
                    query_id,
                    str(full_final_response),
                    fdbck_id,
                    session_id
                )
            
            return StreamingResponse(data_streamer(), media_type='text/event-stream')
            
        else:
            capture_log("ERROR", "❌ API key validation failed")
            
            error_response = {
                "error": "Authentication failed",
                "detail": "unauthenticated user",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "logs": captured_logs,
                "log_count": len(captured_logs)
            }
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=json.dumps(error_response, indent=2)
            )
            
    except HTTPException as e:
        capture_log("ERROR", f"❌ HTTP Exception occurred: {e}")
        raise e
        
    except Exception as e:
        capture_log("ERROR", f"❌ Unexpected error in main handler: {e}")
        
        error_response = {
            "error": "Unexpected error",
            "detail": str(e),
            "session_id": session_id if 'session_id' in locals() else "unknown",
            "timestamp": datetime.utcnow().isoformat(),
            "logs": captured_logs,
            "log_count": len(captured_logs)
        }
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=json.dumps(error_response, indent=2)
        )
