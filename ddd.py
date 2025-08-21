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
    # Create logs as a mutable list that will persist
    execution_logs = []
    
    def log_event(message):
        """Add log with timestamp"""
        timestamp = datetime.utcnow().isoformat()
        log_entry = f"[{timestamp}] {message}"
        execution_logs.append(log_entry)
        print(f"DEBUG LOG ADDED: {log_entry}")  # Verify logs are being added
        try:
            logger.info(log_entry)
        except:
            pass
    
    # Immediately test logging
    log_event("🚀 STARTED - Testing log system")
    log_event("📝 This message should appear in final response")
    print(f"DEBUG: Current log count: {len(execution_logs)}")  # Debug verification
    
    prompt = query.prompt.messages[-1].content
    messages_json = query.prompt.messages
    session_id = str(uuid.uuid4())
    
    log_event(f"Session: {session_id}")
    log_event(f"Model: {query.model.model}")
    log_event(f"Prompt chars: {len(prompt)}")
    
    # The API key validation and generation has been pushed to backend; the api_validator will return True if API key is valid for the application.
    api_validator = ValidApiKey()
    try:
        if api_validator(api_key, query.application.aplctn_cd, query.application.app_id):
            log_event("✅ API key valid")
            
            try:
                log_event("Connecting to Snowflake...")
                sf_conn = SnowFlakeConnector.get_conn(
                    query.application.aplctn_cd,
                    query.application.app_lvl_prefix,
                    session_id
                )
                log_event("✅ Snowflake connected")
            except DatabaseError as e:
                log_event(f"❌ DB error: {str(e)}")
                
                # Create response with logs
                response_with_logs = {
                    "error": "User not authorized to resources",
                    "session_id": session_id,
                    "logs": execution_logs,  # Direct inclusion
                    "log_count": len(execution_logs),
                    "query": {
                        "model": query.model.model,
                        "application": query.application.aplctn_cd
                    }
                }
                
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=json.dumps(response_with_logs)
                )
            
            log_event("Preparing request...")
            pre_response_format = {"response_format": query.response_format.model_dump()} if query.response_format.schema else {}
            # Add provisioned throughput if model is in provisioned models list
            provisioned_dict = {"provisioned_throughput_id": config.provisioned_id} if query.model.model in config.provisioned_models else {}
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
            log_event("Request body ready")
            print("req", request_body)
            headers = {
                "Authorization": f'Snowflake Token="{sf_conn.rest.token}"',
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            url = getattr(config.COMPLETE, "{}_host".format(config.env))

            # ------------------------------------------------------------------
            # Preflight request BEFORE streaming (returns real 4xx/5xx)
            # ------------------------------------------------------------------
            log_event("Starting preflight...")
            preflight_timeout = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=preflight_timeout, verify=True) as pre_client:
                try:
                    pre = await pre_client.post(url, headers=headers, json=request_body)
                    log_event(f"Preflight: {pre.status_code}")
                except httpx.RequestError as e:
                    log_event(f"❌ Preflight failed: {str(e)}")
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=json.dumps({
                            "error": f"Service unavailable: {str(e)}",
                            "logs": execution_logs,
                            "log_count": len(execution_logs)
                        })
                    )
            if pre.is_client_error:
                log_event(f"❌ Client error: {pre.status_code}")
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
                
                raise HTTPException(
                    status_code=status_code,
                    detail=json.dumps({
                        "error": detail,
                        "logs": execution_logs,
                        "log_count": len(execution_logs)
                    })
                )
            if pre.is_server_error:
                log_event(f"❌ Server error: {pre.status_code}")
                raise HTTPException(
                    status_code=pre.status_code,
                    detail=json.dumps({
                        "error": pre.text or "",
                        "logs": execution_logs,
                        "log_count": len(execution_logs)
                    })
                )
            
            log_event("✅ Preflight passed")
            
            response_text = []
            query_id = [None]
            fdbck_id = [str(uuid.uuid4())]
            
            log_event(f"Feedback ID: {fdbck_id[0]}")
            
            async def data_streamer():
                """
                Collect all data internally and return only final JSON response.
                """
                # CRITICAL: Copy logs to local variable to ensure they're accessible
                current_logs = execution_logs.copy()
                log_event("🌊 Streaming started")
                current_logs.append(f"[{datetime.utcnow().isoformat()}] 🌊 Streaming started")
                
                print(f"DEBUG STREAMER: Log count at start: {len(current_logs)}")
                
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
                            current_logs.append(f"[{datetime.utcnow().isoformat()}] Stream connected: {response.status_code}")
                            
                            if response.is_client_error or response.is_server_error:
                                error_message = await response.aread()
                                decoded_error = error_message.decode("utf-8", errors="replace")
                                current_logs.append(f"[{datetime.utcnow().isoformat()}] ❌ Stream error {response.status_code}: {decoded_error}")
                                
                                # Return error as final JSON only
                                error_response = {
                                    "error": "upstream_error", 
                                    "status": response.status_code, 
                                    "detail": decoded_error,
                                    "query_id": query_id[0],
                                    "fdbck_id": fdbck_id[0],
                                    "session_id": session_id,
                                    "logs": current_logs,  # Include logs in error
                                    "log_count": len(current_logs)
                                }
                                yield json.dumps(error_response)
                                return
                            
                            # Collect all streaming data internally - NO YIELDING DURING COLLECTION
                            current_logs.append(f"[{datetime.utcnow().isoformat()}] Processing chunks...")
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
                                            
                                            # Update usage information if available
                                            if 'usage' in chunk_dict:
                                                usage_info.update(chunk_dict['usage'])
                                            
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
                                        
                                        except json.JSONDecodeError as e:
                                            current_logs.append(f"[{datetime.utcnow().isoformat()}] ❌ JSON decode error: {e}")
                                            # Don't yield errors during collection - just log and continue
                                            continue
                            
                            current_logs.append(f"[{datetime.utcnow().isoformat()}] ✅ Processed {chunk_count} chunks")
                    
                    except httpx.RequestError as e:
                        current_logs.append(f"[{datetime.utcnow().isoformat()}] ❌ Request error: {e}")
                        error_response = {
                            "error": "Request error", 
                            "detail": str(e),
                            "query_id": query_id[0],
                            "fdbck_id": fdbck_id[0],
                            "session_id": session_id,
                            "logs": current_logs,
                            "log_count": len(current_logs)
                        }
                        yield json.dumps(error_response)
                        return
                    except Exception as e:
                        current_logs.append(f"[{datetime.utcnow().isoformat()}] ❌ Unexpected error: {e}")
                        error_response = {
                            "error": "Unexpected error", 
                            "detail": str(e),
                            "query_id": query_id[0],
                            "fdbck_id": fdbck_id[0],
                            "session_id": session_id,
                            "logs": current_logs,
                            "log_count": len(current_logs)
                        }
                        yield json.dumps(error_response)
                        return
                
                # ONLY YIELD THE FINAL JSON RESPONSE - NOTHING ELSE
                full_final_response = "".join(response_text)
                current_logs.append(f"[{datetime.utcnow().isoformat()}] 🎉 SUCCESS! Response ready")
                current_logs.append(f"[{datetime.utcnow().isoformat()}] Final log count: {len(current_logs)}")
                
                print(f"DEBUG FINAL: About to send {len(current_logs)} logs")
                
                final_response = {
                    "model": vModel,
                    "created": Created,
                    "query_id": query_id[0],
                    "fdbck_id": fdbck_id[0],
                    "session_id": session_id,
                    "content": full_final_response,
                    "usage": usage_info,
                    "tool_use": {},
                    "query": {
                        "prompt": prompt,
                        "messages": messages_json,
                        "model": {
                            "name": query.model.model,
                            "options": query.model.options
                        },
                        "application": {
                            "aplctn_cd": query.application.aplctn_cd,
                            "app_id": query.application.app_id,
                            "app_lvl_prefix": query.application.app_lvl_prefix
                        }
                    },
                    "logs": current_logs,  # LOGS GUARANTEED HERE
                    "log_count": len(current_logs),
                    "debug_info": {
                        "logs_captured": len(current_logs),
                        "response_length": len(full_final_response),
                        "processing_complete": True
                    }
                }
                
                # SINGLE YIELD - FINAL JSON ONLY
                yield json.dumps(final_response, indent=2)
                
                # Background audit task
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
            
            return StreamingResponse(data_streamer(), media_type='text/event-stream')
        else:
            log_event("❌ API validation failed")
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=json.dumps({
                    "error": "unauthenticated user",
                    "session_id": session_id,
                    "logs": execution_logs,
                    "log_count": len(execution_logs),
                    "query": {
                        "model": query.model.model,
                        "application": query.application.aplctn_cd
                    }
                })
            )
    except HTTPException as e:
        log_event(f"❌ HTTP Exception: {e}")
        raise e
    except Exception as e:
        log_event(f"❌ Unexpected error: {e}")
        raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=json.dumps({
                    "error": str(e),
                    "session_id": session_id,
                    "logs": execution_logs,
                    "log_count": len(execution_logs)
                })
            )
