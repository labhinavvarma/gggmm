"""
Configuration file for Neo4j LangGraph MCP Agent
Modify these settings according to your environment.
"""

# Neo4j Configuration
NEO4J_CONFIG = {
    "uri": "neo4j://localhost:7687",
    "user": "neo4j",
    "password": "your_neo4j_password",  # Change this!
    "database": "neo4j"
}

# Cortex API Configuration
CORTEX_CONFIG = {
    "api_url": "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete",
    "api_key": "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0",  # Change this!
    "model": "llama3.1-70b",
    "app_code": "edagnai",
    "app_id": "edadip"
}

# Server Configuration
SERVER_CONFIG = {
    "mcp_port": 8000,
    "app_port": 8081,
    "ui_port": 8501,
    "host": "0.0.0.0"
}

# Debug Configuration
DEBUG_CONFIG = {
    "enable_debug_logging": True,
    "print_llm_output": True,
    "print_queries": True
}

# Timeouts and Retries
TIMEOUT_CONFIG = {
    "cortex_timeout": 30,  # seconds
    "neo4j_timeout": 10,   # seconds
    "max_retries": 3
}
