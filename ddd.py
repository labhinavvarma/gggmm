import streamlit as st
import requests
import json
import uuid
import pandas as pd
import numpy as np
import io
import base64
import re
import os
import logging
from typing import Dict, Any, List, Optional
import time
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', message='Data Validation extension is not supported')
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='Downcasting object dtype arrays')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

# Handle urllib3 warnings safely
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except (ImportError, AttributeError):
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Import optional libraries with error handling
try:
    import PyPDF2
    PDF_AVAILABLE = True
    logger.info("PyPDF2 available - PDF support enabled")
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PyPDF2 not available - PDF support disabled")

try:
    import docx
    DOCX_AVAILABLE = True
    logger.info("python-docx available - Word support enabled")
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("python-docx not available - Word support disabled")

# Configuration - Use environment variables for security
API_URL = os.getenv("API_URL", "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete")
API_KEY = os.getenv("API_KEY", "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0")  # Default for demo only
APP_ID = os.getenv("APP_ID", "edadip")
APLCTN_CD = os.getenv("APLCTN_CD", "edagnai")
MODEL = os.getenv("MODEL", "llama3.1-70b")

# Page configuration
st.set_page_config(
    page_title="Data Chat - AI File Analysis",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS similar to medical chatbot
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.main-header {
    font-size: 2.8rem;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 1.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: glow-pulse 3s ease-in-out infinite;
}

@keyframes glow-pulse {
    0%, 100% { filter: drop-shadow(0 0 10px rgba(102, 126, 234, 0.3)); }
    50% { filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.6)); }
}

.status-indicator {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.5rem 0;
}

.status-connected {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    color: #155724;
    border: 1px solid #c3e6cb;
}

.analysis-summary-box {
    background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
    padding: 1.5rem;
    border-radius: 12px;
    border: 2px solid #28a745;
    margin: 1rem 0;
    box-shadow: 0 8px 25px rgba(40, 167, 69, 0.2);
}

.sidebar-category {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 0.5rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 3px solid #007bff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
}

.sidebar-category:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
}

.category-prompt-btn {
    background: linear-gradient(135deg, #007bff 0%, #0056b3 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1rem !important;
    margin: 0.3rem 0 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    text-align: left !important;
    box-shadow: 0 4px 15px rgba(0, 123, 255, 0.3) !important;
}

.category-prompt-btn:hover {
    background: linear-gradient(135deg, #0056b3 0%, #004085 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 123, 255, 0.4) !important;
}

.file-status {
    background: linear-gradient(135deg, #e8f5e8 0%, #f0f9ff 100%); 
    border-left: 5px solid #10b981; 
    padding: 15px; 
    border-radius: 10px; 
    margin: 8px 0; 
    font-size: 13px; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.chat-stats {
    text-align: center;
    padding: 1rem;
    color: #6c757d;
    font-style: italic;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 10px;
    margin: 1rem 0;
    border: 1px solid #dee2e6;
}

/* Custom scrollbar */
.stChatMessage {
    margin-bottom: 1rem;
}

.stButton > button {
    border-radius: 8px;
    border: none;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* Enhanced Chat Input Styling */
.stChatInput > div {
    background: white;
    border-radius: 25px;
    border: 2px solid #e2e8f0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    padding: 8px;
}

.stChatInput input {
    font-size: 18px !important;
    padding: 20px 25px !important;
    border: none !important;
    border-radius: 20px !important;
    background: transparent !important;
    height: 60px !important;
    min-height: 60px !important;
}

.stChatInput input:focus {
    border: 2px solid #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
    outline: none !important;
}

.stChatInput button {
    height: 50px !important;
    width: 50px !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
}

.stChatInput button:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
}

/* Individual Chat Message Styling */
.user-message-container {
    display: flex;
    justify-content: flex-end;
    margin: 20px 0;
}

.user-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 18px 24px;
    border-radius: 20px 20px 5px 20px;
    max-width: 70%;
    font-size: 16px;
    line-height: 1.5;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    font-weight: 500;
}

.assistant-message-container {
    display: flex;
    justify-content: flex-start;
    margin: 20px 0;
    align-items: flex-start;
}

.assistant-avatar {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white;
    width: 45px;
    height: 45px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 15px;
    font-size: 20px;
    font-weight: bold;
    box-shadow: 0 3px 10px rgba(79, 70, 229, 0.3);
    flex-shrink: 0;
}

.assistant-message {
    background: white;
    color: #1f2937;
    padding: 20px 25px;
    border-radius: 20px 20px 20px 5px;
    max-width: 75%;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border-left: 4px solid #4f46e5;
    font-size: 16px;
    line-height: 1.6;
}

.assistant-message h1, .assistant-message h2, .assistant-message h3 {
    color: #2c3e50;
    margin-top: 1em;
    margin-bottom: 0.5em;
}

.assistant-message strong {
    color: #4f46e5;
    font-weight: 600;
}

.assistant-message ul, .assistant-message ol {
    margin: 10px 0;
    padding-left: 20px;
}

.assistant-message li {
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

class FileProcessor:
    """Handle different file types and extract content"""
    
    @staticmethod
    def process_excel(file) -> Dict[str, Any]:
        """Process Excel files and return structured data"""
        try:
            pd.set_option('future.no_silent_downcasting', True)
            
            excel_data = pd.read_excel(file, sheet_name=None)
            
            result = {
                "type": "excel",
                "filename": file.name,
                "sheets": {},
                "summary": "",
                "raw_data": {}
            }
            
            for sheet_name, df in excel_data.items():
                # Clean column names
                df.columns = df.columns.astype(str).str.strip()
                
                # Handle missing values properly
                df_clean = df.copy()
                for col in df_clean.columns:
                    if df_clean[col].dtype == 'object':
                        df_clean[col] = df_clean[col].fillna('').astype(str)
                    else:
                        df_clean[col] = df_clean[col].fillna(0)
                
                sheet_info = {
                    "rows": int(len(df)),
                    "columns": int(len(df.columns)),
                    "column_names": [str(col) for col in df.columns.tolist()],
                    "sample_data": df_clean.head(3).to_dict('records'),
                    "data_types": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
                    "summary_stats": {}
                }
                
                # Add summary statistics for numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    try:
                        # Ensure column is properly numeric and handle mixed types
                        col_data = pd.to_numeric(df[col], errors='coerce')
                        valid_data = col_data.dropna()
                        
                        if len(valid_data) > 0:
                            stats = {
                                "mean": float(valid_data.mean()),
                                "median": float(valid_data.median()),
                                "min": float(valid_data.min()),
                                "max": float(valid_data.max()),
                                "count": int(valid_data.count()),
                                "null_count": int(df[col].isnull().sum())
                            }
                            sheet_info["summary_stats"][str(col)] = {str(k): str(v) for k, v in stats.items()}
                    except Exception as e:
                        logger.warning(f"Error processing stats for column {col}: {e}")
                        continue
                
                result["sheets"][sheet_name] = sheet_info
                result["raw_data"][sheet_name] = df_clean.head(100).to_dict('records')
                
            total_rows = sum(info["rows"] for info in result["sheets"].values())
            total_sheets = len(result["sheets"])
            result["summary"] = f"Excel file '{file.name}' with {total_sheets} sheet(s) and {total_rows} total rows"
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing Excel file {file.name}: {e}")
            return {"type": "excel", "filename": file.name, "error": str(e)}
    
    @staticmethod
    def process_csv(file) -> Dict[str, Any]:
        """Process CSV files and return structured data"""
        try:
            pd.set_option('future.no_silent_downcasting', True)
            
            # Try different encodings if UTF-8 fails
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings_to_try:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=encoding)
                    logger.info(f"Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not read CSV file with any encoding")
            
            result = {
                "type": "csv",
                "filename": file.name,
                "summary": "",
                "raw_data": {}
            }
            
            # Clean column names
            df.columns = df.columns.astype(str).str.strip()
            
            # Handle missing values properly
            df_clean = df.copy()
            for col in df_clean.columns:
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].fillna('').astype(str)
                else:
                    df_clean[col] = df_clean[col].fillna(0)
            
            sheet_info = {
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "column_names": [str(col) for col in df.columns.tolist()],
                "sample_data": df_clean.head(3).to_dict('records'),
                "data_types": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
                "summary_stats": {}
            }
            
            # Add summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                try:
                    # Ensure column is properly numeric and handle mixed types
                    col_data = pd.to_numeric(df[col], errors='coerce')
                    valid_data = col_data.dropna()
                    
                    if len(valid_data) > 0:
                        stats = {
                            "mean": float(valid_data.mean()),
                            "median": float(valid_data.median()),
                            "min": float(valid_data.min()),
                            "max": float(valid_data.max()),
                            "count": int(valid_data.count()),
                            "null_count": int(df[col].isnull().sum())
                        }
                        sheet_info["summary_stats"][str(col)] = {str(k): str(v) for k, v in stats.items()}
                except Exception as e:
                    logger.warning(f"Error processing stats for column {col}: {e}")
                    continue
            
            result["sheet_info"] = sheet_info
            result["raw_data"]["main"] = df_clean.head(100).to_dict('records')
            result["summary"] = f"CSV file '{file.name}' with {len(df)} rows and {len(df.columns)} columns"
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing CSV file {file.name}: {e}")
            return {"type": "csv", "filename": file.name, "error": str(e)}
    
    @staticmethod
    def process_word(file) -> Dict[str, Any]:
        """Process Word documents and extract text"""
        if not DOCX_AVAILABLE:
            return {
                "type": "word", 
                "filename": file.name,
                "error": "python-docx library not available. Install with: pip install python-docx"
            }
            
        try:
            doc = docx.Document(file)
            
            paragraphs = []
            tables_data = []
            
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text.strip())
            
            for table in doc.tables:
                table_data = []
                for row in table.rows:
                    row_data = [cell.text.strip() for cell in row.cells]
                    table_data.append(row_data)
                tables_data.append(table_data)
            
            full_text = "\n".join(paragraphs)
            
            return {
                "type": "word",
                "filename": file.name,
                "paragraphs": paragraphs[:50],
                "tables": tables_data,
                "full_text": full_text[:5000],
                "summary": f"Word document '{file.name}' with {len(paragraphs)} paragraphs, {len(tables_data)} tables, and {len(full_text.split())} words",
                "word_count": len(full_text.split())
            }
            
        except Exception as e:
            logger.error(f"Error processing Word file {file.name}: {e}")
            return {"type": "word", "filename": file.name, "error": str(e)}
    
    @staticmethod
    def process_pdf(file) -> Dict[str, Any]:
        """Process PDF files and extract text"""
        if not PDF_AVAILABLE:
            return {
                "type": "pdf", 
                "filename": file.name,
                "error": "PyPDF2 library not available. Install with: pip install PyPDF2"
            }
            
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            pages_text = []
            full_text = ""
            
            # Limit to 10 pages to prevent memory issues
            pages_to_process = min(10, len(pdf_reader.pages))
            
            for page_num in range(pages_to_process):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    pages_text.append({
                        "page": page_num + 1,
                        "text": page_text.strip()[:1000]
                    })
                    full_text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                    pages_text.append({
                        "page": page_num + 1,
                        "text": f"Error extracting text: {str(e)}"
                    })
            
            return {
                "type": "pdf",
                "filename": file.name,
                "pages": pages_text,
                "full_text": full_text[:5000],
                "summary": f"PDF document '{file.name}' with {len(pdf_reader.pages)} pages and {len(full_text.split())} words",
                "page_count": len(pdf_reader.pages),
                "word_count": len(full_text.split())
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF file {file.name}: {e}")
            return {"type": "pdf", "filename": file.name, "error": str(e)}

class SFAssistAPI:
    """Enhanced API client with better error handling and security"""
    
    def __init__(self):
        self.api_url = API_URL
        self.api_key = API_KEY
        self.app_id = APP_ID
        self.aplctn_cd = APLCTN_CD
        self.model = MODEL
        self.session_id = str(uuid.uuid4())  # Maintain session consistency
    
    def create_system_message(self, files_data: List[Dict[str, Any]]) -> str:
        """Create system message with file context"""
        base_msg = """You are Data Chat - an expert AI assistant that analyzes uploaded files and provides insights, statistics, and recommendations.

CORE RESPONSIBILITIES:
• Analyze data comprehensively across all uploaded files
• Provide detailed statistical analysis, trends, and patterns
• Generate actionable insights and recommendations  
• Answer questions accurately based on the available data
• Explain findings in clear, understandable terms

IMPORTANT: DO NOT PROVIDE ANY CODE OR PROGRAMMING SYNTAX
• Never include Python, SQL, R, or any programming code
• Never show technical implementation details
• Never provide code snippets, functions, or scripts
• Focus only on business insights, analysis, and recommendations
• Use plain English explanations and business terminology

COMMUNICATION STYLE:
• Professional business language only
• Use structured formatting with bullet points and headers
• Include specific numbers, percentages, and data points
• Provide both summaries and detailed breakdowns
• Focus on business value and actionable insights
• Use chart descriptions instead of code to create charts

RESPONSE FORMAT:
• Start with key findings summary
• Provide detailed analysis with numbers
• End with actionable recommendations
• Use business terminology, not technical jargon
"""
        
        if not files_data:
            return base_msg + "\nNo files have been uploaded yet. Please ask the user to upload files to begin analysis."
        
        context = f"\n=== UPLOADED FILES ===\nYou have access to {len(files_data)} file(s):\n\n"
        
        for i, file_data in enumerate(files_data, 1):
            if "error" in file_data:
                context += f"FILE {i}: {file_data.get('filename', 'Unknown')} - Error: {file_data['error']}\n\n"
                continue
                
            try:
                filename = str(file_data.get('filename', 'Unknown'))
                file_type = str(file_data.get('type', 'Unknown')).upper()
                summary = str(file_data.get('summary', 'N/A'))
                
                context += f"FILE {i}: {filename}\n"
                context += f"Type: {file_type}\n"
                context += f"Summary: {summary}\n"
                
                if file_data.get('type') == 'excel':
                    sheets = file_data.get("sheets", {})
                    for sheet_name, sheet_info in sheets.items():
                        rows = str(sheet_info.get('rows', 0))
                        columns = str(sheet_info.get('columns', 0))
                        col_names = sheet_info.get('column_names', [])
                        col_names_str = ', '.join([str(col) for col in col_names[:5]])
                        if len(col_names) > 5:
                            col_names_str += "..."
                        
                        context += f"  Sheet '{str(sheet_name)}': {rows} rows × {columns} columns\n"
                        context += f"  Columns: {col_names_str}\n"
                
                elif file_data.get('type') == 'csv':
                    sheet_info = file_data.get("sheet_info", {})
                    rows = str(sheet_info.get('rows', 0))
                    columns = str(sheet_info.get('columns', 0))
                    col_names = sheet_info.get('column_names', [])
                    col_names_str = ', '.join([str(col) for col in col_names[:5]])
                    if len(col_names) > 5:
                        col_names_str += "..."
                    
                    context += f"  CSV Data: {rows} rows × {columns} columns\n"
                    context += f"  Columns: {col_names_str}\n"
                    
                    summary_stats = sheet_info.get('summary_stats', {})
                    if summary_stats:
                        context += f"  Numeric columns with stats: {', '.join(summary_stats.keys())}\n"
                            
                elif file_data.get('type') in ['word', 'pdf']:
                    word_count = str(file_data.get('word_count', 0))
                    context += f"  Content length: {word_count} words\n"
                    
                    full_text = str(file_data.get('full_text', ''))
                    if full_text and full_text.strip():
                        preview = full_text[:500] + "..." if len(full_text) > 500 else full_text
                        context += f"  Content preview: {preview}\n"
                
                context += "\n" + "-"*40 + "\n"
            except Exception as e:
                logger.error(f"Error processing file data for context: {e}")
                context += f"FILE {i}: Error processing file data\n\n"
                continue
        
        return (base_msg + context)[:8000]  # Reasonable limit for context
    
    def send_message(self, user_message: str, files_context: List[Dict[str, Any]] = None) -> Optional[str]:
        """Send message with proper error handling - no conversation history to avoid API errors"""
        try:
            sys_msg = self.create_system_message(files_context or [])
            
            # Build messages array - ONLY current message to avoid data type conflicts
            messages = [
                {
                    "role": "user",
                    "content": str(user_message).strip()
                }
            ]
            
            payload = {
                "query": {
                    "aplctn_cd": str(self.aplctn_cd),
                    "app_id": str(self.app_id),
                    "api_key": str(self.api_key),
                    "method": "cortex",
                    "model": str(self.model),
                    "sys_msg": str(sys_msg),
                    "limit_convs": "0",
                    "prompt": {
                        "messages": messages
                    },
                    "app_lvl_prefix": "enhanced_data_analyst",
                    "user_id": "data_analyst_enhanced",
                    "session_id": str(self.session_id)
                }
            }
            
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "Authorization": f'Snowflake Token="{str(self.api_key)}"'
            }
            
            logger.info(f"Sending request with {len(messages)} messages")
            
            # Use verify=True for production security
            verify_ssl = os.getenv("VERIFY_SSL", "false").lower() == "true"
            
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=payload, 
                verify=verify_ssl, 
                timeout=60
            )
            
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                raw = response.text
                if "end_of_stream" in raw:
                    answer, _, _ = raw.partition("end_of_stream")
                    return answer.strip()
                return raw.strip()
            else:
                error_msg = f"API Error {response.status_code}: {response.text[:500]}"
                logger.error(error_msg)
                st.error(error_msg)
                return None
                
        except requests.exceptions.Timeout:
            error_msg = "Request timed out. Please try again."
            logger.error(error_msg)
            st.error(error_msg)
            return None
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            return None
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            return None

def render_custom_message(role: str, content: str):
    """Render custom styled chat messages"""
    if role == "user":
        st.markdown(f"""
        <div class="user-message-container">
            <div class="user-message">
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Format assistant content
        formatted_content = content.replace('\n', '<br>')
        try:
            formatted_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_content)
            formatted_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', formatted_content)
            formatted_content = re.sub(r'^\s*[-•]\s*(.+)$', r'&nbsp;&nbsp;• \1', formatted_content, flags=re.MULTILINE)
        except Exception:
            formatted_content = content.replace('\n', '<br>')
        
        st.markdown(f"""
        <div class="assistant-message-container">
            <div class="assistant-avatar">
                AI
            </div>
            <div class="assistant-message">
                {formatted_content}
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_quick_questions():
    """Define categorized quick questions similar to medical chatbot"""
    return {
        "📊 Data Overview": [
            "What are the key insights from my data?",
            "Provide a comprehensive data summary",
            "What's the structure of my uploaded files?",
            "Show me the main statistics"
        ],
        "📈 Statistical Analysis": [
            "Show me summary statistics for all numeric columns",
            "What are the data distributions and patterns?",
            "Identify outliers and anomalies in the data",
            "Calculate correlation between key variables"
        ],
        "🔍 Data Quality": [
            "Check for missing values and data quality issues",
            "Are there any data inconsistencies?",
            "Validate data completeness and accuracy",
            "Suggest data cleaning recommendations"
        ],
        "📋 Detailed Insights": [
            "What trends do you see in the data?",
            "Compare different segments or categories",
            "Find patterns and relationships",
            "Generate actionable business recommendations"
        ],
        "🎯 Custom Analysis": [
            "Perform a deep dive analysis",
            "Create a comprehensive data report",
            "What story does this data tell?",
            "Provide strategic insights and next steps"
        ]
    }

def main():
    """Main application with improved structure and modern chat interface"""
    
    # Set pandas options
    pd.set_option('future.no_silent_downcasting', True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "files_data" not in st.session_state:
        st.session_state.files_data = []
    if "uploaded_files_names" not in st.session_state:
        st.session_state.uploaded_files_names = []
    if "api_client" not in st.session_state:
        st.session_state.api_client = SFAssistAPI()
    if "selected_prompt" not in st.session_state:
        st.session_state.selected_prompt = None
    
    # Header
    st.markdown('<h1 class="main-header">💬 Data Chat</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered File Analysis & Insights")
    
    # Connection status
    st.markdown("""
    <div class="status-indicator status-connected">
        ✅ Connected to Data Analysis Engine
    </div>
    """, unsafe_allow_html=True)
    
    # File summary box
    if st.session_state.files_data:
        total_files = len(st.session_state.files_data)
        file_types = list(set([f.get('type', 'unknown') for f in st.session_state.files_data if 'error' not in f]))
        
        st.markdown(f"""
        <div class="analysis-summary-box">
            <h4 style="margin: 0 0 0.5rem 0; color: #28a745;">📊 Files Loaded</h4>
            <p style="margin: 0; color: #155724;">
                <strong>Total Files:</strong> {total_files} | 
                <strong>Types:</strong> {', '.join(file_types).upper()} | 
                <strong>Status:</strong> Ready for Analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("📁 File Upload")
        
        # Determine supported file types
        supported_types = ['xlsx', 'xls', 'csv']
        type_descriptions = ["Excel (.xlsx, .xls)", "CSV (.csv)"]
        
        if DOCX_AVAILABLE:
            supported_types.extend(['docx'])
            type_descriptions.append("Word (.docx)")
        
        if PDF_AVAILABLE:
            supported_types.extend(['pdf'])
            type_descriptions.append("PDF (.pdf)")
        
        help_text = f"Supported: {', '.join(type_descriptions)}"
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files (multiple files supported)",
            type=supported_types,
            help=help_text,
            accept_multiple_files=True
        )
        
        # Process uploaded files
        if uploaded_files:
            current_file_names = [f.name for f in uploaded_files]
            
            if current_file_names != st.session_state.uploaded_files_names:
                st.session_state.uploaded_files_names = current_file_names
                st.session_state.files_data = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    try:
                        if file_extension in ['xlsx', 'xls']:
                            file_data = FileProcessor.process_excel(uploaded_file)
                        elif file_extension == 'csv':
                            file_data = FileProcessor.process_csv(uploaded_file)
                        elif file_extension == 'docx' and DOCX_AVAILABLE:
                            file_data = FileProcessor.process_word(uploaded_file)
                        elif file_extension == 'pdf' and PDF_AVAILABLE:
                            file_data = FileProcessor.process_pdf(uploaded_file)
                        else:
                            logger.warning(f"Unsupported file type: {file_extension}")
                            continue
                            
                        if file_data and "error" not in file_data:
                            st.session_state.files_data.append(file_data)
                        else:
                            error_msg = file_data.get("error", "Unknown error") if file_data else "No data returned"
                            st.error(f"Failed to process {uploaded_file.name}: {error_msg}")
                            
                    except Exception as e:
                        logger.error(f"Exception processing {uploaded_file.name}: {str(e)}")
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                progress_bar.empty()
                status_text.empty()
                
                if st.session_state.files_data:
                    st.success(f"✅ {len(st.session_state.files_data)} file(s) processed successfully!")
                else:
                    st.error("❌ No files were processed successfully. Please check file formats.")
        elif st.session_state.uploaded_files_names:
            st.session_state.uploaded_files_names = []
            st.session_state.files_data = []
        
        # Display uploaded files
        if st.session_state.files_data:
            st.markdown("---")
            st.markdown("### 📋 Uploaded Files")
            for file_data in st.session_state.files_data:
                if "error" not in file_data:
                    file_icon = {
                        "excel": "📊", 
                        "csv": "📈", 
                        "word": "📄", 
                        "pdf": "📕"
                    }.get(file_data.get('type'), "📄")
                    st.markdown(f"""
                    <div class="file-status">
                        <div style="font-weight: 700; font-size: 15px; color: #1f2937; margin-bottom: 8px;">
                            {file_icon} {file_data.get('filename', 'Unknown')}
                        </div>
                        <div style="color: #374151; font-size: 12px;">
                            {file_data.get('summary', 'No summary')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Upload files to start analysis")
        
        st.markdown("---")
        
        # Quick Questions
        st.title("🎯 Quick Questions")
        
        quick_questions = create_quick_questions()
        
        for category, questions in quick_questions.items():
            with st.expander(category, expanded=False):
                for i, question in enumerate(questions):
                    if st.button(question, key=f"q_{category}_{i}", use_container_width=True):
                        if not st.session_state.files_data:
                            st.warning("Upload files first!")
                        else:
                            st.session_state.selected_prompt = question
                            st.rerun()
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", key="clear_chat", use_container_width=True):
            st.session_state.messages = []
            st.success("Chat history cleared!")
            st.rerun()
    
    # Handle selected prompt
    if st.session_state.selected_prompt:
        user_question = st.session_state.selected_prompt
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        with st.spinner("Processing your request..."):
            try:
                response = st.session_state.api_client.send_message(
                    user_question,
                    st.session_state.files_data
                )
                if response:
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        st.session_state.selected_prompt = None
        st.rerun()
    
    # Chat interface
    st.markdown("### 💬 Chat with Your Data")
    
    # Display chat messages
    if st.session_state.messages:
        # Display messages in chat format (chronological order - oldest first)
        for message in st.session_state.messages:
            render_custom_message(message["role"], message["content"])
        
        # Show message count
        st.markdown(f"""
        <div class="chat-stats">
            📊 <strong>Total Messages:</strong> {len(st.session_state.messages)} messages
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("🚀 Start a conversation! Upload files and use the quick prompts in the sidebar or type your question below.")
        
        # Show example questions when no chat history
        st.markdown("### 💡 Try asking:")
        st.markdown("• What are the key insights from my data?")
        st.markdown("• Show me summary statistics")
        st.markdown("• What trends do you see?")
        st.markdown("• Check for data quality issues")
        st.markdown("• Generate a comprehensive report")
    
    # Chat input at bottom
    user_question = st.chat_input("💬 Ask a question about your data...")
    
    if user_question:
        if not st.session_state.files_data:
            st.warning("Please upload files first to start analysis!")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            with st.spinner("Analyzing your question..."):
                try:
                    # Send only current message to API (no conversation history)
                    response = st.session_state.api_client.send_message(
                        user_question, 
                        st.session_state.files_data
                    )
                    
                    if response:
                        # Store in UI history for display
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
