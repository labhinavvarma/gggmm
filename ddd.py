import json
import uuid
import httpx
from datetime import datetime
from fastapi import HTTPException, status, BackgroundTasks, Header, Body, Depends
from fastapi.responses import StreamingResponse
from typing import Annotated

@router.post("/v2/complete")
async def llm_gateway(
        api_key: Annotated[str | None, Header()],
        query: Annotated[AiCompleteQryModel, Body(embed=True)],
        config: Annotated[GenAiEnvSettings, Depends(get_config)],
        logger: Annotated[Logger, Depends(get_logger)],
        background_tasks: BackgroundTasks,
        get_load_datetime: Annotated[datetime, Depends(get_load_timestamp)]          
):
    # ABSOLUTE GUARANTEE: Create logs list that WILL appear in output
    LOGS = []
    
    # Simple, bulletproof log function
    def LOG(message):
        timestamp = datetime.utcnow().isoformat()
        log_entry = f"[{timestamp}] {message}"
        LOGS.append(log_entry)
        print(f"LOG CAPTURED: {log_entry}")  # Debug print to verify
        try:
            logger.info(log_entry)
        except:
            pass
        return log_entry
    
    # START LOGGING IMMEDIATELY
    LOG("🚀 GATEWAY STARTED - LOGGING SYSTEM ACTIVE")
    LOG("📝 This log entry WILL appear in the final JSON response")
    
    session_id = "unknown"
    
    try:
        prompt = query.prompt.messages[-1].content
        messages_json = query.prompt.messages
        session_id = str(uuid.uuid4())
        
        LOG(f"✅ Session ID: {session_id}")
        LOG(f"📋 Model: {query.model.model}")
        LOG(f"📏 Prompt length: {len(prompt)} chars")
        LOG(f"🏢 Application: {query.application.aplctn_cd}")
        
        # The API key validation and generation has been pushed to backend; the api_validator will return True if API key is valid for the application.
        api_validator = ValidApiKey()
        LOG("🔐 Starting API validation...")
        
        if api_validator(api_key, query.application.aplctn_cd, query.application.app_id):
            LOG("✅ API key validation SUCCESS")
            
            try:
                LOG("🔗 Connecting to Snowflake...")
                sf_conn = SnowFlakeConnector.get_conn(
                    query.application.aplctn_cd,
                    query.application.app_lvl_prefix,
                    session_id
                )
                LOG("✅ Snowflake connection SUCCESS")
            except DatabaseError as e:
                LOG(f"❌ Database error: {str(e)}")
                
                # GUARANTEED LOG OUTPUT - Even in errors
                error_with_logs = {
                    "error": "User not authorized to resources",
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "LOGS": LOGS,  # LOGS GUARANTEED HERE
                    "log_count": len(LOGS)
                }
                
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=json.dumps(error_with_logs, indent=2)
                )
            
            LOG("🛠️ Preparing request...")
            pre_response_format = {"response_format": query.response_format.model_dump()} if query.response_format.schema else {}
            # Add provisioned throughput if model is in provisioned models list
            provisioned_dict = {"provisioned_throughput_id": config.provisioned_id} if query.model.model in config.provisioned_models else {}
            
            if provisioned_dict:
                LOG("⚡ Using provisioned throughput")
            
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
            
            LOG(f"📦 Request body ready ({len(request_body)} fields)")
            print("req", request_body)
            
            headers = {
                "Authorization": f'Snowflake Token="{sf_conn.rest.token}"',
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            url = getattr(config.COMPLETE, "{}_host".format(config.env))
            
            LOG(f"🎯 Target URL: {url}")

            # ------------------------------------------------------------------
            # Preflight request BEFORE streaming (returns real 4xx/5xx)
            # ------------------------------------------------------------------
            LOG("🔍 Starting preflight check...")
            
            preflight_timeout = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=preflight_timeout, verify=True) as pre_client:
                try:
                    pre = await pre_client.post(url, headers=headers, json=request_body)
                    LOG(f"✅ Preflight response: {pre.status_code}")
                except httpx.RequestError as e:
                    LOG(f"❌ Preflight failed: {str(e)}")
                    
                    error_with_logs = {
                        "error": f"Service unavailable: {str(e)}",
                        "session_id": session_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "LOGS": LOGS,  # LOGS GUARANTEED HERE
                        "log_count": len(LOGS)
                    }
                    
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=json.dumps(error_with_logs, indent=2)
                    )
                    
            if pre.is_client_error:
                LOG(f"❌ Client error: {pre.status_code}")
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
                
                LOG(f"Error details: {detail}")
                
                error_with_logs = {
                    "error": detail,
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "LOGS": LOGS,  # LOGS GUARANTEED HERE
                    "log_count": len(LOGS)
                }
                
                raise HTTPException(
                    status_code=status_code,
                    detail=json.dumps(error_with_logs, indent=2)
                )
                
            if pre.is_server_error:
                LOG(f"❌ Server error: {pre.status_code}")
                
                error_with_logs = {
                    "error": pre.text or "",
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "LOGS": LOGS,  # LOGS GUARANTEED HERE
                    "log_count": len(LOGS)
                }
                
                raise HTTPException(
                    status_code=pre.status_code,
                    detail=json.dumps(error_with_logs, indent=2)
                )
            
            LOG("✅ Preflight check PASSED")
            
            response_text = []
            query_id = [None]
            fdbck_id = [str(uuid.uuid4())]
            
            LOG(f"🆔 Feedback ID: {fdbck_id[0]}")
            LOG("🌊 Initializing streaming...")
            
            async def data_streamer():
                """
                Collect all data internally and return only final JSON response.
                """
                LOG("🚀 STREAMING STARTED")
                
                vModel = query.model.model
                Created = datetime.utcnow().isoformat()
                usage_info = {
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0
                }
                
                chunk_count = 0
                
                stream_timeout = httpx.Timeout(connect=5.0, read=None, write=10.0, pool=10.0)
                async with httpx.AsyncClient(timeout=stream_timeout, verify=True) as clnt:
                    try:
                        async with clnt.stream('POST', url, headers=headers, json=request_body) as response:
                            LOG(f"📡 Stream connected: {response.status_code}")
                            
                            if response.is_client_error or response.is_server_error:
                                error_message = await response.aread()
                                decoded_error = error_message.decode("utf-8", errors="replace")
                                LOG(f"❌ Stream error {response.status_code}: {decoded_error}")
                                
                                # Return error as final JSON only
                                error_response = {
                                    "error": "upstream_error", 
                                    "status": response.status_code, 
                                    "detail": decoded_error,
                                    "query_id": query_id[0],
                                    "fdbck_id": fdbck_id[0],
                                    "session_id": session_id,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "LOGS": LOGS,  # LOGS GUARANTEED HERE
                                    "log_count": len(LOGS)
                                }
                                yield json.dumps(error_response, indent=2)
                                return
                            
                            # Collect all streaming data internally - NO YIELDING DURING COLLECTION
                            LOG("📦 Processing chunks...")
                            buffer = b""
                            
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
                                                LOG(f"🆔 Query ID: {query_id[0]}")
                                            
                                            # Update usage information if available
                                            if 'usage' in chunk_dict:
                                                usage_info.update(chunk_dict['usage'])
                                                LOG(f"📊 Usage: {usage_info}")
                                            
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
                                                if chunk_count % 10 == 0:
                                                    LOG(f"📝 Processed {chunk_count} chunks")
                                        
                                        except json.JSONDecodeError as e:
                                            LOG(f"❌ JSON decode error: {e}")
                                            # Don't yield errors during collection - just log and continue
                                            continue
                            
                            LOG(f"✅ Streaming complete: {chunk_count} chunks, {len(''.join(response_text))} chars")
                    
                    except httpx.RequestError as e:
                        LOG(f"❌ Request error: {e}")
                        
                        error_response = {
                            "error": "Request error", 
                            "detail": str(e),
                            "query_id": query_id[0],
                            "fdbck_id": fdbck_id[0],
                            "session_id": session_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "LOGS": LOGS,  # LOGS GUARANTEED HERE
                            "log_count": len(LOGS)
                        }
                        yield json.dumps(error_response, indent=2)
                        return
                        
                    except Exception as e:
                        LOG(f"❌ Unexpected error: {e}")
                        
                        error_response = {
                            "error": "Unexpected error", 
                            "detail": str(e),
                            "query_id": query_id[0],
                            "fdbck_id": fdbck_id[0],
                            "session_id": session_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "LOGS": LOGS,  # LOGS GUARANTEED HERE
                            "log_count": len(LOGS)
                        }
                        yield json.dumps(error_response, indent=2)
                        return
                
                # ONLY YIELD THE FINAL JSON RESPONSE - NOTHING ELSE
                full_final_response = "".join(response_text)
                LOG(f"🎉 SUCCESS! Response ready: {len(full_final_response)} chars")
                LOG(f"📊 Total logs captured: {len(LOGS)}")
                
                # ABSOLUTE GUARANTEE: LOGS WILL BE IN THIS RESPONSE
                final_response = {
                    "model": vModel,
                    "created": Created,
                    "query_id": query_id[0],
                    "fdbck_id": fdbck_id[0],
                    "session_id": session_id,
                    "content": full_final_response,
                    "usage": usage_info,
                    "tool_use": {},
                    "LOGS": LOGS,  # <<<< LOGS ARE 100% GUARANTEED HERE
                    "log_count": len(LOGS),  # <<<< PROOF LOGS EXIST
                    "metadata": {
                        "total_log_entries": len(LOGS),
                        "response_length": len(full_final_response),
                        "chunks_processed": chunk_count,
                        "logs_included": True
                    }
                }
                
                LOG("📤 Sending final response with logs")
                print(f"FINAL RESPONSE LOG COUNT: {len(LOGS)}")  # Debug verification
                
                # SINGLE YIELD - FINAL JSON ONLY
                yield json.dumps(final_response, indent=2)
                
                # Background audit task
                LOG("🔧 Background task starting...")
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
                background_tasks.add_task(
                    log_response,
                    audit_rec,
                    query_id,
                    str(full_final_response),
                    fdbck_id,
                    session_id
                )
                LOG("✅ Background task scheduled")
            
            return StreamingResponse(data_streamer(), media_type='text/event-stream')
            
        else:
            LOG("❌ API validation FAILED")
            
            error_with_logs = {
                "error": "unauthenticated user",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "LOGS": LOGS,  # LOGS GUARANTEED HERE
                "log_count": len(LOGS)
            }
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=json.dumps(error_with_logs, indent=2)
            )
            
    except HTTPException as e:
        LOG(f"❌ HTTP Exception: {e}")
        raise e
        
    except Exception as e:
        LOG(f"❌ Unexpected error: {e}")
        
        error_with_logs = {
            "error": str(e),
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "LOGS": LOGS,  # LOGS GUARANTEED HERE
            "log_count": len(LOGS)
        }
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=json.dumps(error_with_logs, indent=2)
        )
