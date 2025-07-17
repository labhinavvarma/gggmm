def execute_tool_node(state: AgentState) -> dict:
    tool = state.tool
    query = state.query
    trace = state.trace
    answer = ""
    valid_tools = {"read_neo4j_cypher", "write_neo4j_cypher", "get_neo4j_schema"}
    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    try:
        if tool not in valid_tools:
            answer = f"⚠️ MCP tool not recognized: {tool}"

        # Special-case: Neo4j database name query (unsupported in community/most clouds)
        elif tool == "read_neo4j_cypher" and query.strip().lower() == "return db.name() as name":
            answer = "Your Neo4j does not support querying the database name via Cypher. Check your connection settings."

        elif tool == "get_neo4j_schema":
            result = requests.post("http://localhost:8000/get_neo4j_schema", headers=headers)
            answer = result.json() if result.ok else result.text

        elif tool == "read_neo4j_cypher":
            node_count_query = (
                query.strip().lower() == "match (n) return count(n)"
                or query.strip().lower() == "match (n) return count(n) as node_count"
            )
            data = {"query": query, "params": {}}
            result = requests.post("http://localhost:8000/read_neo4j_cypher", json=data, headers=headers)
            if node_count_query:
                try:
                    rjson = result.json() if result.ok else {}
                    reported_count = None
                    if isinstance(rjson, list) and rjson and isinstance(rjson[0], dict):
                        first_row = rjson[0]
                        if "count(n)" in first_row and first_row["count(n)"] is not None:
                            reported_count = int(first_row["count(n)"])
                        elif "node_count" in first_row and first_row["node_count"] is not None:
                            reported_count = int(first_row["node_count"])
                    if reported_count is not None and reported_count < 200:
                        alt_data = {"query": "CALL db.stats.retrieve('GRAPH COUNTS') YIELD data RETURN data['NodeCount'] AS node_count", "params": {}}
                        alt_result = requests.post("http://localhost:8000/read_neo4j_cypher", json=alt_data, headers=headers)
                        alt_json = alt_result.json() if alt_result.ok else {}
                        alt_count = None
                        if isinstance(alt_json, list) and alt_json and isinstance(alt_json[0], dict):
                            alt_row = alt_json[0]
                            if "node_count" in alt_row and alt_row["node_count"] is not None:
                                alt_count = int(alt_row["node_count"])
                        answer = {
                            "Simple MATCH count(n)": reported_count,
                            "Admin/Stats NodeCount": alt_count,
                            "Raw results": {"MATCH": rjson, "ADMIN": alt_json},
                            "note": "Tried fallback admin node count as result seemed low."
                        }
                    elif reported_count is not None:
                        answer = rjson
                    else:
                        answer = {
                            "error": "No node count found in result.",
                            "raw_result": rjson
                        }
                except Exception as exc:
                    answer = {"error": "Could not parse count result", "detail": str(exc)}
            else:
                answer = result.json() if result.ok else result.text

        elif tool == "write_neo4j_cypher":
            data = {"query": query, "params": {}}
            result = requests.post(f"http://localhost:8000/write_neo4j_cypher", json=data, headers=headers)
            answer = result.json() if result.ok else result.text

        else:
            answer = f"Unknown tool: {tool}"

    except Exception as e:
        answer = f"⚠️ MCP execution failed: {str(e)}"
    return {
        "question": state.question,
        "session_id": state.session_id,
        "tool": tool,
        "query": query,
        "trace": trace,
        "answer": answer
    }

