# Comprehensive Python Requirements with Snowflake & LangGraph Support
# Install with: pip install -r requirements.txt

# ================================
# SNOWFLAKE ECOSYSTEM
# ================================
snowflake-connector-python[pandas]>=3.6.0
snowflake-sqlalchemy>=1.5.0
snowflake-ml-python>=1.0.12
snowpark-python>=1.11.1
snowflake-ingest>=2.0.0
snowflake-kafka-connector>=2.0.0
snowflake-cli>=2.0.0

# ================================
# LANGCHAIN & LANGGRAPH ECOSYSTEM
# ================================
langchain>=0.1.0
langchain-core>=0.1.0
langchain-community>=0.0.13
langgraph>=0.0.26
langsmith>=0.0.77
langchain-openai>=0.0.5
langchain-anthropic>=0.1.0
langchain-google-genai>=0.0.6
langchain-huggingface>=0.0.1
langchain-experimental>=0.0.45

# ================================
# CORE DATA SCIENCE & ANALYTICS
# ================================
pandas>=2.1.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0
scipy>=1.11.0
scikit-learn>=1.3.0
jupyter>=1.0.0
ipython>=8.16.0
jupyterlab>=4.0.0
notebook>=7.0.0

# ================================
# DATABASE CONNECTORS & ORM
# ================================
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
pymongo>=4.6.0
redis>=5.0.0
mysql-connector-python>=8.2.0
cx-oracle>=8.3.0
pyodbc>=5.0.0
pymssql>=2.2.0
cassandra-driver>=3.28.0
elasticsearch>=8.11.0

# ================================
# MACHINE LEARNING & AI
# ================================
tensorflow>=2.15.0
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.36.0
huggingface-hub>=0.19.0
openai>=1.3.0
anthropic>=0.7.0
google-generativeai>=0.3.0
cohere>=4.37.0
sentence-transformers>=2.2.0
chromadb>=0.4.18
faiss-cpu>=1.7.4
pinecone-client>=2.2.4
weaviate-client>=3.25.0
qdrant-client>=1.7.0

# ================================
# COMPUTER VISION & IMAGE PROCESSING
# ================================
opencv-python>=4.8.0
pillow>=10.1.0
imageio>=2.33.0
scikit-image>=0.22.0
albumentations>=1.3.0

# ================================
# NATURAL LANGUAGE PROCESSING
# ================================
nltk>=3.8.0
spacy>=3.7.0
textblob>=0.17.0
gensim>=4.3.0
beautifulsoup4>=4.12.0

# ================================
# WEB DEVELOPMENT & APIs
# ================================
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
flask>=3.0.0
django>=4.2.0
requests>=2.31.0
httpx>=0.25.0
aiohttp>=3.9.0
streamlit>=1.28.0
gradio>=4.7.0
pydantic>=2.5.0
marshmallow>=3.20.0

# ================================
# ASYNC & CONCURRENCY
# ================================
asyncio  # Built-in
aiofiles>=23.2.0
asyncpg>=0.29.0
celery>=5.3.0
dramatiq>=1.15.0

# ================================
# DATA PROCESSING & ETL
# ================================
apache-airflow>=2.7.0
dask[complete]>=2023.11.0
ray[default]>=2.8.0
prefect>=2.14.0
great-expectations>=0.18.0
pandera>=0.17.0

# ================================
# FILE PROCESSING
# ================================
openpyxl>=3.1.0
xlsxwriter>=3.1.0
pypdf>=3.17.0
python-docx>=1.1.0
pyyaml>=6.0.0
toml>=0.10.0
configparser  # Built-in
python-dotenv>=1.0.0

# ================================
# CLOUD SERVICES
# ================================
boto3>=1.34.0  # AWS
botocore>=1.34.0
google-cloud-storage>=2.10.0  # Google Cloud
google-cloud-bigquery>=3.13.0
azure-storage-blob>=12.19.0  # Azure
azure-identity>=1.15.0

# ================================
# TESTING & QUALITY
# ================================
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-asyncio>=0.23.0
hypothesis>=6.92.0
mock>=5.1.0
factory-boy>=3.3.0

# ================================
# CODE QUALITY & FORMATTING
# ================================
black>=23.11.0
isort>=5.12.0
flake8>=6.1.0
mypy>=1.7.0
pylint>=3.0.0
bandit>=1.7.0
pre-commit>=3.6.0

# ================================
# MONITORING & LOGGING
# ================================
loguru>=0.7.0
structlog>=23.2.0
sentry-sdk>=1.38.0
prometheus-client>=0.19.0
psutil>=5.9.0
memory-profiler>=0.61.0

# ================================
# UTILITIES & CLI
# ================================
click>=8.1.0
typer>=0.9.0
rich>=13.7.0
tqdm>=4.66.0
arrow>=1.3.0
python-dateutil>=2.8.0
pytz>=2023.3
schedule>=1.2.0

# ================================
# CRYPTOGRAPHY & SECURITY
# ================================
cryptography>=41.0.0
pyjwt>=2.8.0
passlib>=1.7.0
bcrypt>=4.1.0
python-jose>=3.3.0

# ================================
# NETWORKING & SCRAPING
# ================================
scrapy>=2.11.0
selenium>=4.15.0
playwright>=1.40.0
websockets>=12.0.0

# ================================
# SCIENTIFIC COMPUTING
# ================================
sympy>=1.12.0
networkx>=3.2.0
igraph>=0.11.0
pyvis>=0.3.0

# ================================
# GEOSPATIAL
# ================================
geopandas>=0.14.0
folium>=0.15.0
geopy>=2.4.0
shapely>=2.0.0

# ================================
# VISUALIZATION ADVANCED
# ================================
bokeh>=3.3.0
altair>=5.2.0
dash>=2.14.0
holoviews>=1.18.0

# ================================
# TIME SERIES
# ================================
statsmodels>=0.14.0
prophet>=1.1.0
tslearn>=0.6.0

# ================================
# DEPLOYMENT & CONTAINERS
# ================================
docker>=6.1.0
kubernetes>=28.1.0
gunicorn>=21.2.0

# ================================
# COMMUNICATION
# ================================
twilio>=8.11.0
slack-sdk>=3.26.0
discord.py>=2.3.0
python-telegram-bot>=20.7.0

# ================================
# FINANCIAL & BUSINESS
# ================================
yfinance>=0.2.0
alpha-vantage>=2.3.0
quandl>=3.7.0

# ================================
# JUPYTER EXTENSIONS
# ================================
ipywidgets>=8.1.0
jupyter-dash>=0.4.0
nbconvert>=7.11.0

# ================================
# DEVELOPMENT TOOLS
# ================================
python-decouple>=3.8.0
invoke>=2.2.0
bump2version>=1.0.0
wheel>=0.42.0
setuptools>=69.0.0

# ================================
# SNOWFLAKE INTEGRATION LIBRARIES
# ================================
# Additional libraries that work well with Snowflake
dbt-snowflake>=1.6.0
great-expectations[snowflake]>=0.18.0
apache-airflow[snowflake]>=2.7.0
prefect-snowflake>=0.2.0
hightouch-events>=1.0.0

# ================================
# VECTOR DATABASES & EMBEDDINGS
# ================================
# For LangGraph/LangChain vector operations
tiktoken>=0.5.0
openai-whisper>=20231117
python-multipart>=0.0.6
