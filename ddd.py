# requirements-minimal.txt - Minimal Production Dependencies

# ==================== CORE REQUIREMENTS ONLY ====================
# Essential dependencies for running the Intelligent Neo4j Assistant

# Web interface
streamlit>=1.28.0,<2.0.0

# AI agent framework
langgraph>=0.0.40,<1.0.0
langchain>=0.1.0,<1.0.0

# MCP server
fastmcp>=0.1.0,<1.0.0

# Async support
nest-asyncio>=1.5.8,<2.0.0

# Database
neo4j>=5.14.0,<6.0.0

# Data processing
pandas>=2.1.0,<3.0.0
pydantic>=2.4.0,<3.0.0

# Visualization
plotly>=5.17.0,<6.0.0

# HTTP/API
requests>=2.31.0,<3.0.0
urllib3>=2.0.0,<3.0.0

# Utilities
python-dateutil>=2.8.2,<3.0.0

# ==================== INSTALLATION COMMAND ====================
# pip install -r requirements-minimal.txt
