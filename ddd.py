# config.py - Configuration settings for the Intelligent Neo4j Assistant

import os
from typing import Dict, Any

class Config:
    """Configuration class for the intelligent Neo4j system."""
    
    # Neo4j Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://10.189.116.237:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Vkg5d$F!pLq2@9vRwE=")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "connectiq")
    
    # Cortex API Configuration
    CORTEX_URL = os.getenv("CORTEX_URL", "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete")
    CORTEX_API_KEY = os.getenv("CORTEX_API_KEY", "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0")
    CORTEX_APP_ID = os.getenv("CORTEX_APP_ID", "edadip")
    CORTEX_APLCTN_CD = os.getenv("CORTEX_APLCTN_CD", "edagnai")
    CORTEX_MODEL = os.getenv("CORTEX_MODEL", "llama3.1-70b")
    
    # MCP Server Configuration
    MCP_SERVER_SCRIPT = "langgraph_mcpserver.py"
    
    # Application Configuration
    APP_TITLE = "🧠 Intelligent Neo4j Assistant"
    APP_ICON = "🧠"
    
    # Performance Settings
    DEFAULT_QUERY_TIMEOUT = 30
    MAX_RETRIES = 3
    CACHE_TTL = 300  # 5 minutes
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = "logs/intelligent_neo4j.log"
    
    # UI Configuration
    STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
    STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "0.0.0.0")
    
    @classmethod
    def get_cortex_config(cls) -> Dict[str, Any]:
        """Get Cortex API configuration."""
        return {
            "url": cls.CORTEX_URL,
            "api_key": cls.CORTEX_API_KEY,
            "app_id": cls.CORTEX_APP_ID,
            "aplctn_cd": cls.CORTEX_APLCTN_CD,
            "model": cls.CORTEX_MODEL,
            "sys_msg": "You are a powerful AI assistant specialized in Neo4j Cypher queries. Generate modern Neo4j 5.x compatible syntax."
        }
    
    @classmethod
    def get_neo4j_config(cls) -> Dict[str, str]:
        """Get Neo4j connection configuration."""
        return {
            "uri": cls.NEO4J_URI,
            "username": cls.NEO4J_USERNAME,
            "password": cls.NEO4J_PASSWORD,
            "database": cls.NEO4J_DATABASE
        }
