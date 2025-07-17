TIMEOUT_CONFIG = {
    # UI to Backend timeouts
    "ui_request_timeout": 120,  # 2 minutes for UI requests
    "ui_health_check_timeout": 5,  # 5 seconds for health checks
    
    # Backend to Cortex API timeouts
    "cortex_api_timeout": 90,  # 1.5 minutes for Cortex API calls
    "cortex_connection_timeout": 30,  # 30 seconds for connection
    
    # Neo4j database timeouts
    "neo4j_query_timeout": 60,  # 1 minute for Neo4j queries
    "neo4j_connection_timeout": 10,  # 10 seconds for connection
    
    # Agent execution timeouts
    "agent_execution_timeout": 150,  # 2.5 minutes for total agent execution
    "tool_execution_timeout": 90,  # 1.5 minutes for individual tool execution
    
    # Retry configuration
    "max_retries": 3,
    "retry_delay": 2,  # seconds between retries
    "backoff_factor": 2,  # exponential backoff multiplier
}
