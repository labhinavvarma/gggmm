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
    
    # The API key validation and generation has been pushed to backend
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
            
            pre_response_format = {"response_format": query.response_format.model_dump()} if query.response_format.schema else {}
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
            
            # REMOVED: print statement that was causing text output
            logger.debug(f"Request body: {request_body}")  # Use logger instead of print
            
            headers = {
                "Authorization": f'Snowflake Token="{sf_conn.rest.token}"',
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            url = getattr(config.COMPLETE, "{}_host".format(config.env))

            # Preflight request validation
            preflight_timeout = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=preflight_timeout, verify=False) as pre_client:
                try:
                    pre = await pre_client.post(url, headers=headers, json=request_body)
                except httpx.RequestError as e:
                    logger.error(f"Request error during initial check: {e}")
                    # Return JSON error instead of streaming
                    return JSONResponse(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        content={"error": "service_unavailable", "detail": str(e)}
                    )
            
            # Handle client errors with JSON response
            if pre.is_client_error:
                error_message = pre.text or ""
                status_code = pre.status_code
                error_details = {
                    429: "Too many requests. Please implement retry with backoff.",
                    402: "Budget exceeded. Check your Cortex token quota.",
                    400: "Bad request sent to Snowflake Cortex."
                }
                detail = error_details.get(status_code, "") + " " + error_message
                return JSONResponse(
                    status_code=status_code,
                    content={"error": "client_error", "detail": detail.strip()}
                )
            
            if pre.is_server_error:
                return JSONResponse(
                    status_code=pre.status_code,
                    content={"error": "server_error", "detail": pre.text or ""}
                )
            
            # Process the successful response
            response_text = []
            query_id = [None]
            fdbck_id = [str(uuid.uuid4())]
            
            # Initialize response data
            vModel = query.model.model
            Created = datetime.utcnow().isoformat()
            usage_info = {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0
            }
            
            # Stream and collect response
            stream_timeout = httpx.Timeout(connect=5.0, read=None, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=stream_timeout, verify=False) as clnt:
                try:
                    async with clnt.stream('POST', url, headers=headers, json=request_body) as response:
                        if response.is_client_error or response.is_server_error:
                            error_message = await response.aread()
                            decoded_error = error_message.decode("utf-8", errors="replace")
                            logger.error(f"Upstream stream error {response.status_code}: {decoded_error}")
                            return JSONResponse(
                                status_code=response.status_code,
                                content={"error": "upstream_error", "detail": decoded_error}
                            )
                        
                        # Process stream content
                        buffer = b""
                        async for result_chunk in response.aiter_bytes():
                            buffer += result_chunk
                            while b'\n\n' in buffer:
                                elem, buffer = buffer.split(b'\n\n', 1)
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
                                        
                                        if text_content:
                                            response_text.append(text_content)
                                            
                                    except json.JSONDecodeError as e:
                                        logger.error(f"Error decoding JSON: {e}")
                                        # Continue processing instead of returning error
                                        continue
                                        
                except httpx.RequestError as e:
                    logger.error(f"Request error: {e}")
                    return JSONResponse(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        content={"error": "request_error", "detail": str(e)}
                    )
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    return JSONResponse(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        content={"error": "unexpected_error", "detail": str(e)}
                    )
            
            # Build final response
            full_final_response = "".join(response_text)
            final_response = {
                "model": vModel,
                "created": Created,
                "prompt": prompt,
                "query_id": query_id[0],
                "fdbck_id": fdbck_id[0],
                "content": full_final_response,
                "usage": usage_info,
                "tool_use": {}
            }
            
            # Add background task for auditing
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
            
            # Return JSON response instead of streaming
            return JSONResponse(content=final_response)
            
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
