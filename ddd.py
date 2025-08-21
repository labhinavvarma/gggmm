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
    # Create a shared logs container that will persist throughout the request
    request_logs = {
        "entries": [],
        "start_time": datetime.utcnow().isoformat()
    }
    
    def add_log(level, message):
        """Add log entry - guaranteed to work"""
        timestamp = datetime.utcnow().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "level": level.upper(),
            "message": str(message)
        }
        request_logs["entries"].append(log_entry)
        
        # Also log to system logger
        try:
            if level.upper() == "ERROR":
                logger.error(f"[{timestamp}] {message}")
            elif level.upper() == "DEBUG":
                logger.debug(f"[{timestamp}] {message}")
            elif level.upper() == "WARNING":
                logger.warning(f"[{timestamp}] {message}")
            else:
                logger.info(f"[{timestamp}] {message}")
        except:
            pass  # Don't let logging errors break the main flow
        
        return log_entry
    
    # Test log to verify logging works
    add_log("INFO", "🚀 LLM Gateway Started - Log System Active")
    add_log("INFO", f"Request timestamp: {request_logs['start_time']}")
    
    session_id = "unknown"
    
    try:
        prompt = query.prompt.messages[-1].content
        messages_json = query.prompt.messages
        session_id = str(uuid.uuid4())
        
        add_log("INFO", f"Session ID generated: {session_id}")
        add_log("INFO", f"Model requested: {query.model.model}")
        add_log("DEBUG", f"Prompt length: {len(prompt)} characters")
        add_log("DEBUG", f"Application: {query.application.aplctn_cd}")
        
        # API validation
        add_log("INFO", "Starting API key validation...")
        api_validator = ValidApiKey()
        
        if api_validator(api_key, query.application.aplctn_cd, query.application.app_id):
            add_log("INFO", "✅ API key validation passed")
            
            # Database connection
            try:
                add_log("DEBUG", "Connecting to Snowflake...")
                sf_conn = SnowFlakeConnector.get_conn(
                    query.application.aplctn_cd,
                    query.application.app_lvl_prefix,
                    session_id
                )
                add_log("INFO", "✅ Snowflake connection established")
            except DatabaseError as e:
                add_log("ERROR", f"❌ Database connection failed: {str(e)}")
                
                error_response = {
                    "error": "database_connection_failed",
                    "detail": "User not authorized to resources",
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "logs": request_logs["entries"],
                    "log_count": len(request_logs["entries"])
                }
                
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=json.dumps(error_response, indent=2)
                )
            
            # Request preparation
            add_log("DEBUG", "Preparing request body...")
            pre_response_format = {"response_format": query.response_format.model_dump()} if query.response_format.schema else {}
            provisioned_dict = {"provisioned_throughput_id": config.provisioned_id} if query.model.model in config.provisioned_models else {}
            
            if provisioned_dict:
                add_log("INFO", "Using provisioned throughput")
            
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
            
            add_log("DEBUG", f"Request body created with {len(request_body)} fields")
            print("req", request_body)
            
            headers = {
                "Authorization": f'Snowflake Token="{sf_conn.rest.token}"',
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            url = getattr(config.COMPLETE, "{}_host".format(config.env))
            
            add_log("INFO", f"Target URL: {url}")

            # Preflight check
            add_log("INFO", "🔍 Starting preflight check...")
            preflight_timeout = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=10.0)
            
            async with httpx.AsyncClient(timeout=preflight_timeout, verify=True) as pre_client:
                try:
                    pre = await pre_client.post(url, headers=headers, json=request_body)
                    add_log("INFO", f"Preflight status: {pre.status_code}")
                except httpx.RequestError as e:
                    add_log("ERROR", f"❌ Preflight failed: {str(e)}")
                    
                    error_response = {
                        "error": "service_unavailable",
                        "detail": f"Service unavailable: {str(e)}",
                        "session_id": session_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "logs": request_logs["entries"],
                        "log_count": len(request_logs["entries"])
                    }
                    
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=json.dumps(error_response, indent=2)
                    )
            
            # Handle preflight errors
            if pre.is_client_error:
                add_log("ERROR", f"❌ Preflight client error: {pre.status_code}")
                error_message = pre.text or ""
                
                detail_map = {
                    429: "Too many requests. Please implement retry with backoff.",
                    402: "Budget exceeded. Check your Cortex token quota.",
                    400: "Bad request sent to Snowflake Cortex."
                }
                detail = detail_map.get(pre.status_code, "Client error") + f" {error_message}"
                
                add_log("ERROR", f"Error details: {detail}")
                
                error_response = {
                    "error": "client_error",
                    "detail": detail,
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "logs": request_logs["entries"],
                    "log_count": len(request_logs["entries"])
                }
                
                raise HTTPException(
                    status_code=pre.status_code,
                    detail=json.dumps(error_response, indent=2)
                )
            
            if pre.is_server_error:
                add_log("ERROR", f"❌ Preflight server error: {pre.status_code}")
                
                error_response = {
                    "error": "server_error",
                    "detail": pre.text or "",
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "logs": request_logs["entries"],
                    "log_count": len(request_logs["entries"])
                }
                
                raise HTTPException(
                    status_code=pre.status_code,
                    detail=json.dumps(error_response, indent=2)
                )
            
            add_log("INFO", "✅ Preflight check passed")
            
            # Initialize streaming variables
            response_text = []
            query_id = [None]
            fdbck_id = [str(uuid.uuid4())]
            
            add_log("INFO", f"Feedback ID: {fdbck_id[0]}")
            add_log("INFO", "Preparing to start streaming...")
            
            async def data_streamer():
                """Stream data and return final response with logs"""
                add_log("INFO", "🌊 Data streaming initiated")
                
                vModel = query.model.model
                Created = datetime.utcnow().isoformat()
                usage_info = {
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0
                }
                
                chunk_count = 0
                
                try:
                    stream_timeout = httpx.Timeout(connect=5.0, read=None, write=10.0, pool=10.0)
                    async with httpx.AsyncClient(timeout=stream_timeout, verify=True) as clnt:
                        add_log("DEBUG", "HTTP client created for streaming")
                        
                        async with clnt.stream('POST', url, headers=headers, json=request_body) as response:
                            add_log("INFO", f"Stream connected - Status: {response.status_code}")
                            
                            if response.is_client_error or response.is_server_error:
                                error_message = await response.aread()
                                decoded_error = error_message.decode("utf-8", errors="replace")
                                add_log("ERROR", f"❌ Stream error {response.status_code}: {decoded_error}")
                                
                                error_response = {
                                    "error": "upstream_error", 
                                    "status": response.status_code, 
                                    "detail": decoded_error,
                                    "query_id": query_id[0],
                                    "fdbck_id": fdbck_id[0],
                                    "session_id": session_id,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "logs": request_logs["entries"],
                                    "log_count": len(request_logs["entries"])
                                }
                                yield json.dumps(error_response, indent=2)
                                return
                            
                            # Process streaming chunks
                            add_log("DEBUG", "Starting chunk processing...")
                            buffer = b""
                            
                            async for result_chunk in response.aiter_bytes():
                                buffer += result_chunk
                                while b'\n\n' in buffer:
                                    elem, buffer = buffer.split(b'\n\n', 1)
                                    chunk_count += 1
                                    
                                    if b'content' in elem or b'text' in elem:
                                        try:
                                            chunk_dict = json.loads(elem.replace(b'data: ', b''))
                                            
                                            if 'id' in chunk_dict:
                                                query_id[0] = chunk_dict['id']
                                                add_log("DEBUG", f"Query ID: {query_id[0]}")
                                            
                                            if 'usage' in chunk_dict:
                                                usage_info.update(chunk_dict['usage'])
                                                add_log("DEBUG", f"Usage updated: {usage_info}")
                                            
                                            # Extract content
                                            text_content = None
                                            if 'choices' in chunk_dict and len(chunk_dict['choices']) > 0:
                                                choice = chunk_dict['choices'][0]
                                                if 'delta' in choice:
                                                    delta = choice['delta']
                                                    text_content = delta.get('text') or delta.get('content')
                                                elif 'message' in choice:
                                                    text_content = choice['message'].get('content')
                                            
                                            if text_content:
                                                response_text.append(text_content)
                                                if chunk_count % 5 == 0:
                                                    add_log("DEBUG", f"Processed {chunk_count} chunks")
                                        
                                        except json.JSONDecodeError as e:
                                            add_log("ERROR", f"JSON decode error in chunk {chunk_count}: {e}")
                                            continue
                            
                            add_log("INFO", f"✅ Streaming completed - {chunk_count} chunks processed")
                            
                except httpx.RequestError as e:
                    add_log("ERROR", f"❌ Streaming request error: {e}")
                    
                    error_response = {
                        "error": "request_error", 
                        "detail": str(e),
                        "query_id": query_id[0],
                        "fdbck_id": fdbck_id[0],
                        "session_id": session_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "logs": request_logs["entries"],
                        "log_count": len(request_logs["entries"])
                    }
                    yield json.dumps(error_response, indent=2)
                    return
                    
                except Exception as e:
                    add_log("ERROR", f"❌ Unexpected streaming error: {e}")
                    
                    error_response = {
                        "error": "unexpected_error", 
                        "detail": str(e),
                        "query_id": query_id[0],
                        "fdbck_id": fdbck_id[0],
                        "session_id": session_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "logs": request_logs["entries"],
                        "log_count": len(request_logs["entries"])
                    }
                    yield json.dumps(error_response, indent=2)
                    return
                
                # Create final response
                full_final_response = "".join(response_text)
                add_log("INFO", f"🎉 Processing complete! Response length: {len(full_final_response)}")
                add_log("INFO", f"Final log count: {len(request_logs['entries'])}")
                
                final_response = {
                    "model": vModel,
                    "created": Created,
                    "query_id": query_id[0],
                    "fdbck_id": fdbck_id[0],
                    "session_id": session_id,
                    "content": full_final_response,
                    "usage": usage_info,
                    "tool_use": {},
                    "logs": request_logs["entries"],  # LOGS ARE DEFINITELY INCLUDED HERE
                    "metadata": {
                        "total_log_entries": len(request_logs["entries"]),
                        "response_length": len(full_final_response),
                        "chunks_processed": chunk_count,
                        "processing_start": request_logs["start_time"],
                        "processing_end": datetime.utcnow().isoformat()
                    }
                }
                
                add_log("INFO", "📤 Sending final response with logs")
                
                # Yield the final response
                yield json.dumps(final_response, indent=2)
                
                # Background audit task
                add_log("DEBUG", "Scheduling background audit task")
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
                
                add_log("INFO", "✅ Background task scheduled")
            
            return StreamingResponse(data_streamer(), media_type='text/event-stream')
            
        else:
            add_log("ERROR", "❌ API key validation failed")
            
            error_response = {
                "error": "authentication_failed",
                "detail": "unauthenticated user",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "logs": request_logs["entries"],
                "log_count": len(request_logs["entries"])
            }
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=json.dumps(error_response, indent=2)
            )
            
    except HTTPException as e:
        add_log("ERROR", f"❌ HTTPException: {e}")
        raise e
        
    except Exception as e:
        add_log("ERROR", f"❌ Unexpected main error: {e}")
        
        error_response = {
            "error": "unexpected_error",
            "detail": str(e),
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "logs": request_logs["entries"],
            "log_count": len(request_logs["entries"])
        }
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=json.dumps(error_response, indent=2)
        )
