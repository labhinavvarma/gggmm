def clean_cypher_query(query: str) -> str:
    """
    Cleans up LLM-generated Cypher:
    - Joins lines together.
    - Fixes missing spaces between keywords and clauses.
    - Collapses multiple spaces and trims.
    """
    # Collapse newlines and extra spaces
    query = re.sub(r'[\r\n]+', ' ', query)
    keywords = [
        "MATCH", "WITH", "RETURN", "ORDER BY", "UNWIND", "WHERE", "LIMIT",
        "SKIP", "CALL", "YIELD", "CREATE", "MERGE", "SET", "DELETE", "DETACH DELETE", "REMOVE"
    ]
    for kw in keywords:
        query = re.sub(rf'(?<!\s)({kw})', r' \1', query)
        query = re.sub(rf'({kw})([^\s\(])', r'\1 \2', query)
    query = re.sub(r'\s+', ' ', query)
    return query.strip()
