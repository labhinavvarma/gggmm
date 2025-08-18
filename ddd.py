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
            # CHANGE: removed top-level client creation to avoid leaking connections
            # clnt = httpx.AsyncClient(verify=False)
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

            # ------------------------------------------------------------------
            # CHANGE 1: Preflight request BEFORE streaming (returns real 4xx/5xx)
            # ------------------------------------------------------------------
            preflight_timeout = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=10.0)  # CHANGE
            async with httpx.AsyncClient(timeout=preflight_timeout, verify=False) as pre_client:  # CHANGE
                try:
                    pre = await pre_client.post(url, headers=headers, json=request_body)  # CHANGE
                except httpx.RequestError as e:
                    logger.error(f"Request error during initial check: {e}")  # CHANGE
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,  # CHANGE
                        detail=f"Service unavailable: {str(e)}"  # CHANGE
                    )
            if pre.is_client_error:  # CHANGE
                error_message = pre.text or ""  # CHANGE
                status_code = pre.status_code  # CHANGE
                if status_code == 429:
                    raise HTTPException(
                        status_code=429,
                        detail="Too many requests. Please implement retry with backoff. Original message: " + error_message
                    )
                elif status_code == 402:
                    raise HTTPException(
                        status_code=402,
                        detail="Budget exceeded. Check your Cortex token quota. " + error_message
                    )
                elif status_code == 400:
                    raise HTTPException(
                        status_code=400,
                        detail="Bad request sent to Snowflake Cortex. " + error_message
                    )
                else:
                    raise HTTPException(
                        status_code=status_code,
                        detail=error_message
                    )
            if pre.is_server_error:  # CHANGE
                raise HTTPException(  # CHANGE
                    status_code=pre.status_code,
                    detail=pre.text or ""
                )
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
                usage_info = {
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0
                }
                # CHANGE 2: use a context-managed client for streaming + sane timeout
                stream_timeout = httpx.Timeout(connect=5.0, read=None, write=10.0, pool=10.0)  # CHANGE
                async with httpx.AsyncClient(timeout=stream_timeout, verify=False) as clnt:  # CHANGE
                    try:
                        async with clnt.stream('POST', url, headers=headers, json=request_body) as response:
                            # CHANGE 3: minimal in-stream guard (we're already 200—emit error frame & stop)
                            if response.is_client_error or response.is_server_error:  # CHANGE
                                error_message = await response.aread()  # CHANGE
                                decoded_error = error_message.decode("utf-8", errors="replace")  # CHANGE
                                logger.error(f"Upstream stream error {response.status_code}: {decoded_error}")  # CHANGE
                                yield json.dumps({"error": "upstream_error", "status": response.status_code, "detail": decoded_error})  # CHANGE
                                return  # CHANGE
                            # Stream the response content - Based on working pattern
                            # CHANGE 4: add a small buffer to avoid splitting SSE events across chunks
                            buffer = b""  # CHANGE
                            async for result_chunk in response.aiter_bytes():
                                buffer += result_chunk  # CHANGE
                                while b'\n\n' in buffer:  # CHANGE
                                    elem, buffer = buffer.split(b'\n\n', 1)  # CHANGE
                                    # keep your original filtering & parsing logic
                                    if b'content' in elem or b'text' in elem:
                                        try:
                                            chunk_dict = json.loads(elem.replace(b'data: ', b''))  # still allow both shapes
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
                                                # Try different response formats
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
                                            if text_content:
                                                response_text.append(text_content)
                                                # REMOVED: yield text_content - No streaming text output
                                        except json.JSONDecodeError as e:
                                            logger.error(f"Error decoding JSON: {e}")
                                            yield json.dumps({"error": "Error decoding JSON", "detail": str(e)})
                                            continue
                    except httpx.RequestError as e:
                        logger.error(f"Request error: {e}")
                        yield json.dumps({"error": "Request error", "detail": str(e)})
                        return
                    except Exception as e:
                        logger.error(f"Unexpected error: {e}")
                        yield json.dumps({"error": "Unexpected error", "detail": str(e)})
                        return
                
                # REMOVED: yield "end_of_stream" - No end signal needed
                
                # Final JSON response only
                responses = {
                    "model": vModel,
                    "created": Created,
                    "prompt": prompt,
                    "query_id": query_id[0],
                    "fdbck_id": fdbck_id[0],
                    "content": "".join(response_text),
                    "usage": usage_info,
                    "tool_use": {}
                }
                full_final_response = "".join(response_text)
                yield json.dumps(responses)
                
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
