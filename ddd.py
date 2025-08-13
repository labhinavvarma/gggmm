@route.post("/v2/complete")
async def llm_gateway(
        api_key: Annotated[str | None, Header()],
        query: Annotated[AiCompleteQryModel, Body(embed=True)],
        config: Annotated[GenAiEnvSettings, Depends(get_config)],
        logger: Annotated[Logger, Depends(get_logger)],
        background_tasks: BackgroundTasks,
        get_load_datetime: Annotated[datetime, Depends(get_load_timestamp)]          
):
    prompt = query.prompt.messages[-1].content
    messages_json = query.prompt.messages
    session_id = "4533"
    # The API key validation and generation has been pushed to backend; the api_validator will return True if API key is valid for the application.
    api_validator = ValidApiKey()
    try:
        if api_validator(api_key, query.application.aplctn_cd, query.application.app_id):
            try:
                sf_conn = SnowFlakeConnector.get_conn(
                    query.application.aplctn_cd,
                    query.application.app_lvl_prefix,
                    session_id
                )
            except DatabaseError as e:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User not authorized to resources"
                )
            clnt = httpx.AsyncClient(verify=False)
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
            print("req", request_body)
            headers = {
                "Authorization": f'Snowflake Token="{sf_conn.rest.token}"',
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            url = getattr(config.COMPLETE, "{}_host".format(config.env))
            response_text = []
            query_id = [None]
            fdbck_id = [str(uuid.uuid4())]
            
            async def data_streamer():
                """
                Stream data from the service and yield responses with proper exception handling.
                """
                vModel = query.model.model  # Model reference
                Created = datetime.utcnow().isoformat()  # Creation timestamp
                # Initialize response aggregation
                response_agg = {
                    "content": "",
                    "usage": {
                        "total_tokens": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0
                    },
                    "tool_use": {}
                }
                try:
                    async with clnt.stream('POST', url, headers=headers, json=request_body) as response:
                        if response.is_client_error:
                            error_message = await response.aread()
                            status_code = response.status_code
                            decoded_error = error_message.decode("utf-8")
 
                            if status_code == 429:
                                raise HTTPException(
                                    status_code=429,
                                    detail="Too many requests. Please implement retry with backoff. Original message: " + decoded_error
                                )
                            elif status_code == 402:
                                raise HTTPException(
                                    status_code=402,
                                    detail="Budget exceeded. Check your Cortex token quota. " + decoded_error
                                )
                            elif status_code == 400:
                                raise HTTPException(
                                    status_code=400,
                                    detail="Bad request sent to Snowflake Cortex. " + decoded_error
                                )
                            else:
                                raise HTTPException(
                                    status_code=status_code,
                                    detail=decoded_error
                                )
                        if response.is_server_error:
                            error_message = await response.aread()
                            raise HTTPException(
                                status_code=response.status_code,
                                detail=error_message.decode("utf-8")
                            )
                        
                        # Stream the response content
                        async for result_chunk in response.aiter_bytes():
                            chunk_str = result_chunk.decode('utf-8')
                            # Split by double newlines for SSE format
                            for elem in chunk_str.split('\n\n'):
                                elem = elem.strip()
                                if elem.startswith('data: ') and elem != 'data: [DONE]':
                                    try:
                                        # Remove 'data: ' prefix
                                        json_str = elem[6:]  # Remove 'data: '
                                        if json_str.strip():
                                            json_line = json.loads(json_str)
                                            
                                            # Update usage information if available
                                            if "usage" in json_line:
                                                response_agg["usage"].update(json_line["usage"])
                                            
                                            # Extract query ID if available
                                            if "id" in json_line:
                                                query_id[0] = json_line["id"]
                                            
                                            # Process choices if available
                                            if "choices" in json_line and len(json_line["choices"]) > 0:
                                                choice = json_line["choices"][0]
                                                
                                                # Handle delta chunk (streaming format)
                                                if "delta" in choice:
                                                    delta_chunk = choice["delta"]
                                                    
                                                    # Handle text content
                                                    if "content" in delta_chunk and delta_chunk["content"]:
                                                        content = delta_chunk["content"]
                                                        response_agg["content"] += content
                                                        response_text.append(content)
                                                        
                                                        # Yield intermediate chunk for real-time streaming
                                                        yield f"data: {json.dumps({'delta': {'content': content}})}\n\n"
                                                
                                                # Handle complete message (non-streaming format)
                                                elif "message" in choice:
                                                    message = choice["message"]
                                                    if "content" in message:
                                                        response_agg["content"] = message["content"]
                                                        response_text.append(message["content"])
                                            
                                    except json.JSONDecodeError as e:
                                        logger.error(f"Error decoding JSON: {e}, data: {elem}")
                                        continue
                                    except Exception as e:
                                        logger.error(f"Error processing chunk: {e}")
                                        continue
                                        
                except httpx.RequestError as e:
                    logger.error(f"Request error: {e}")
                    yield f"data: {json.dumps({'error': 'Request error', 'detail': str(e)})}\n\n"
                    return
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    yield f"data: {json.dumps({'error': 'Unexpected error', 'detail': str(e)})}\n\n"
                    return
                
                # Final response with all required fields
                full_final_response = response_agg["content"]
                
                # Yield the final aggregated response with query_id and fdbck_id
                final_response = {
                    "model": vModel,
                    "created": Created,
                    "query_id": query_id[0],
                    "fdbck_id": fdbck_id[0],
                    "content": response_agg["content"],
                    "usage": response_agg["usage"],
                    "tool_use": response_agg["tool_use"]
                }
                
                yield f"data: {json.dumps(final_response)}\n\n"
                yield "data: [DONE]\n\n"
                
                # Extract token count for audit
                token_count = response_agg["usage"].get("total_tokens", 0)
                
                # Update audit record
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
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="unauthenticated user"
            )
    except HTTPException as e:
        logger.error(f"Request error: {e}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
