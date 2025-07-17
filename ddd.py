
# config.py
"""
Configuration settings for Neo4j LangGraph Supervisor Application
"""

import os
from typing import Dict, Any

# Neo4j Database Configuration
NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "neo4j://10.189.116.237:7687"),
    "username": os.getenv("NEO4J_USER", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", "Vkg5d$F!pLq2@9vRwE="),
    "database": os.getenv("NEO4J_DATABASE", "connectiq"),
    "max_connection_lifetime": 3600,
    "max_connection_pool_size": 50,
    "connection_acquisition_timeout": 60
}

# Cortex LLM Configuration
CORTEX_CONFIG = {
    "url": os.getenv("CORTEX_URL", "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"),
    "api_key": os.getenv("CORTEX_API_KEY", "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"),
    "app_id": os.getenv("CORTEX_APP_ID", "edadip"),
    "application_code": os.getenv("CORTEX_APLCTN_CD", "edagnai"),
    "model": os.getenv("CORTEX_MODEL", "llama3.1-70b"),
    "timeout": 30,
    "max_retries": 3
}

# MCP Server Configuration
MCP_CONFIG = {
    "host": os.getenv("MCP_HOST", "0.0.0.0"),
    "port": int(os.getenv("MCP_PORT", "8000")),
    "path": os.getenv("MCP_PATH", "/messages/"),
    "url": f"http://localhost:{os.getenv('MCP_PORT', '8000')}/messages/"
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "host": os.getenv("STREAMLIT_HOST", "localhost"),
    "port": int(os.getenv("STREAMLIT_PORT", "8501")),
    "title": "Neo4j LangGraph Supervisor",
    "icon": "🧠"
}

# LangGraph Configuration
LANGGRAPH_CONFIG = {
    "max_iterations": 10,
    "timeout": 300,  # 5 minutes
    "enable_debugging": True
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.getenv("LOG_FILE", "app.log")
}

# Agent Prompts Configuration
AGENT_PROMPTS = {
    "supervisor_system": """You are a Supervisor Agent coordinating Neo4j database operations.
Your team consists of specialized agents for different tasks.""",
    
    "cypher_generator_system": """You are an Expert Cypher Query Generator for Neo4j ConnectIQ database.
Focus on creating efficient, optimized queries.""",
    
    "query_executor_system": """You are a Query Execution Agent responsible for running queries safely and efficiently.""",
    
    "result_interpreter_system": """You are a Result Interpretation Agent that analyzes data and provides business insights."""
}

# Database Schema Information (for context)
CONNECTIQ_SCHEMA = {
    "nodes": [
        "Apps", "Devices", "Users", "Categories", 
        "Versions", "Reviews", "Developers"
    ],
    "relationships": [
        "COMPATIBLE_WITH", "BELONGS_TO", "HAS_VERSION", 
        "REVIEWED_BY", "DEVELOPED_BY", "INSTALLED_ON"
    ],
    "common_properties": [
        "name", "version", "rating", "install_count", 
        "category", "device_type", "release_date", "description"
    ]
}

# Security Configuration
SECURITY_CONFIG = {
    "allowed_operations": {
        "read": ["MATCH", "RETURN", "WITH", "WHERE", "ORDER BY", "LIMIT", "SKIP"],
        "write": ["CREATE", "MERGE", "SET", "DELETE", "REMOVE"]
    },
    "blocked_operations": [
        "CALL dbms.", "CALL db.", "LOAD CSV", "USING PERIODIC COMMIT"
    ],
    "max_query_length": 5000,
    "max_result_size": 1000
}

def get_config(section: str) -> Dict[str, Any]:
    """Get configuration for a specific section"""
    configs = {
        "neo4j": NEO4J_CONFIG,
        "cortex": CORTEX_CONFIG,
        "mcp": MCP_CONFIG,
        "streamlit": STREAMLIT_CONFIG,
        "langgraph": LANGGRAPH_CONFIG,
        "logging": LOGGING_CONFIG,
        "prompts": AGENT_PROMPTS,
        "schema": CONNECTIQ_SCHEMA,
        "security": SECURITY_CONFIG
    }
    return configs.get(section, {})

def validate_config() -> bool:
    """Validate all configuration settings"""
    required_settings = [
        (NEO4J_CONFIG, "uri"),
        (NEO4J_CONFIG, "username"),
        (NEO4J_CONFIG, "password"),
        (CORTEX_CONFIG, "url"),
        (CORTEX_CONFIG, "api_key")
    ]
    
    for config_dict, key in required_settings:
        if not config_dict.get(key):
            print(f"❌ Missing required configuration: {key}")
            return False
    
    return True

# Environment file template
ENV_TEMPLATE = """
# Neo4j Configuration
NEO4J_URI=neo4j://10.189.116.237:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=Vkg5d$F!pLq2@9vRwE=
NEO4J_DATABASE=connectiq

# Cortex LLM Configuration
CORTEX_URL=https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete
CORTEX_API_KEY=78a799ea-a0f6-11ef-a0ce-15a449f7a8b0
CORTEX_APP_ID=edadip
CORTEX_APLCTN_CD=edagnai
CORTEX_MODEL=llama3.1-70b

# MCP Server Configuration
MCP_HOST=0.0.0.0
MCP_PORT=8000
MCP_PATH=/messages/

# Streamlit Configuration
STREAMLIT_HOST=localhost
STREAMLIT_PORT=8501

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=app.log
"""

def create_env_file():
    """Create a .env file template"""
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(ENV_TEMPLATE)
        print("✅ Created .env file template")
    else:
        print("ℹ️  .env file already exists")

if __name__ == "__main__":
    print("🔧 Configuration validation:")
    if validate_config():
        print("✅ All configurations are valid")
    else:
        print("❌ Configuration validation failed")
    
    create_env_file()
