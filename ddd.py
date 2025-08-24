import streamlit as st
import requests
import json
import uuid
import pandas as pd
import numpy as np
import io
import base64
import re
from typing import Dict, Any, List, Optional
import time
from datetime import datetime
import warnings

# Import optional libraries with error handling
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Disable SSL warnings and other annoying warnings
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except (ImportError, AttributeError):
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Suppress common warnings
warnings.filterwarnings('ignore', message='Data Validation extension is not supported')
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='Downcasting object dtype arrays')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

# === Configuration ===
API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
APP_ID = "edadip"
APLCTN_CD = "edagnai"
MODEL = "llama3.1-70b"

class FileProcessor:
    """Handle different file types and extract content"""
    
    @staticmethod
    def process_excel(file) -> Dict[str, Any]:
        """Process Excel files and return structured data"""
        try:
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
                
                # Handle missing values properly to avoid FutureWarning
                df_clean = df.copy()
                for col in df_clean.columns:
                    if df_clean[col].dtype == 'object':
                        df_clean[col] = df_clean[col].fillna('').astype(str)
                    else:
                        df_clean[col] = df_clean[col].fillna(0)
                
                sheet_info = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "sample_data": df_clean.head(3).to_dict('records'),
                    "data_types": df.dtypes.to_dict(),
                    "summary_stats": {}
                }
                
                # Add summary statistics for numeric columns (with proper error handling)
                numeric_cols = df.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    try:
                        # Only calculate stats if column has valid numeric data
                        valid_data = df[col].dropna()
                        if len(valid_data) > 0 and not valid_data.empty:
                            sheet_info["summary_stats"][col] = {
                                "mean": float(valid_data.mean()),
                                "median": float(valid_data.median()),
                                "min": float(valid_data.min()),
                                "max": float(valid_data.max()),
                                "count": int(valid_data.count()),
                                "null_count": int(df[col].isnull().sum())
                            }
                        else:
                            sheet_info["summary_stats"][col] = {
                                "mean": 0,
                                "median": 0,
                                "min": 0,
                                "max": 0,
                                "count": 0,
                                "null_count": int(df[col].isnull().sum())
                            }
                    except (ValueError, TypeError):
                        # Skip columns that can't be processed
                        continue
                
                sheet_info["data_types"] = {k: str(v) for k, v in sheet_info["data_types"].items()}
                result["sheets"][sheet_name] = sheet_info
                
                # Clean data for JSON serialization
                df_for_json = df_clean.copy()
                result["raw_data"][sheet_name] = df_for_json.to_dict('records')
                
            total_rows = sum(info["rows"] for info in result["sheets"].values())
            total_sheets = len(result["sheets"])
            result["summary"] = f"Excel file '{file.name}' with {total_sheets} sheet(s) and {total_rows} total rows"
            
            return result
            
        except Exception as e:
            st.error(f"Error processing Excel file {file.name}: {str(e)}")
            return {"type": "excel", "filename": file.name, "error": str(e)}
    
    @staticmethod
    def process_word(file) -> Dict[str, Any]:
        """Process Word documents and extract text"""
        if not DOCX_AVAILABLE:
            return {
                "type": "word", 
                "filename": file.name,
                "error": "python-docx library not available"
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
                "paragraphs": paragraphs,
                "tables": tables_data,
                "full_text": full_text,
                "summary": f"Word document '{file.name}' with {len(paragraphs)} paragraphs, {len(tables_data)} tables, and {len(full_text.split())} words",
                "word_count": len(full_text.split())
            }
            
        except Exception as e:
            st.error(f"Error processing Word file {file.name}: {str(e)}")
            return {"type": "word", "filename": file.name, "error": str(e)}
    
    @staticmethod
    def process_pdf(file) -> Dict[str, Any]:
        """Process PDF files and extract text"""
        if not PDF_AVAILABLE:
            return {
                "type": "pdf", 
                "filename": file.name,
                "error": "PyPDF2 library not available"
            }
            
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            pages_text = []
            full_text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    pages_text.append({
                        "page": page_num + 1,
                        "text": page_text.strip()
                    })
                    full_text += page_text + "\n"
                except Exception as e:
                    pages_text.append({
                        "page": page_num + 1,
                        "text": f"Error extracting text: {str(e)}"
                    })
            
            return {
                "type": "pdf",
                "filename": file.name,
                "pages": pages_text,
                "full_text": full_text.strip(),
                "summary": f"PDF document '{file.name}' with {len(pdf_reader.pages)} pages and {len(full_text.split())} words",
                "page_count": len(pdf_reader.pages),
                "word_count": len(full_text.split())
            }
            
        except Exception as e:
            st.error(f"Error processing PDF file {file.name}: {str(e)}")
            return {"type": "pdf", "filename": file.name, "error": str(e)}

class SFAssistAPI:
    """Enhanced API client with better conversation handling"""
    
    def __init__(self):
        self.api_url = API_URL
        self.api_key = API_KEY
        self.app_id = APP_ID
        self.aplctn_cd = APLCTN_CD
        self.model = MODEL
    
    def create_enhanced_system_message(self, files_data: List[Dict[str, Any]]) -> str:
        """Create comprehensive system message with multiple file contexts"""
        base_msg = """You are an expert AI Data Analyst Assistant. You have been provided with the user's uploaded files and should act as their personal data analyst, providing comprehensive, accurate, and actionable insights.

CORE RESPONSIBILITIES:
• Analyze data comprehensively across all uploaded files
• Provide detailed statistical analysis, trends, and patterns
• Generate actionable business insights and recommendations  
• Answer questions accurately based on the available data
• Explain complex findings in clear, understandable terms
• Compare and contrast data across different files when relevant

ANALYSIS APPROACH:
• Always reference specific data points, metrics, and findings from the files
• Use statistical methods and business intelligence principles
• Identify correlations, trends, outliers, and anomalies
• Provide context and explain the significance of findings
• Suggest next steps or further analysis when appropriate
• Be transparent about data limitations or missing information

COMMUNICATION STYLE:
• Professional yet conversational tone
• Use bullet points and structured formatting for clarity
• Include specific numbers, percentages, and data points
• Explain technical concepts in business-friendly language
• Provide both high-level summaries and detailed breakdowns when requested

"""
        
        if not files_data:
            return base_msg + "\nNo files have been uploaded yet. Please ask the user to upload files to begin analysis."
        
        context = f"\n=== UPLOADED FILES CONTEXT ===\nYou have access to {len(files_data)} file(s):\n\n"
        
        for i, file_data in enumerate(files_data, 1):
            if "error" in file_data:
                continue
                
            context += f"FILE {i}: {file_data.get('filename', 'Unknown')}\n"
            context += f"Type: {file_data.get('type', 'Unknown').upper()}\n"
            context += f"Summary: {file_data.get('summary', 'N/A')}\n"
            
            if file_data.get('type') == 'excel':
                context += "EXCEL STRUCTURE:\n"
                for sheet_name, sheet_info in file_data.get("sheets", {}).items():
                    context += f"  Sheet '{sheet_name}': {sheet_info['rows']} rows × {sheet_info['columns']} columns\n"
                    context += f"  Columns: {', '.join(sheet_info['column_names'][:10])}{'...' if len(sheet_info['column_names']) > 10 else ''}\n"
                    
                    if sheet_info.get('summary_stats'):
                        context += f"  Numeric columns with stats: {', '.join(sheet_info['summary_stats'].keys())}\n"
                    
                    # Include sample data
                    if sheet_info.get('sample_data'):
                        context += f"  Sample data (first 2 rows): {json.dumps(sheet_info['sample_data'][:2], indent=2)}\n"
                        
            elif file_data.get('type') in ['word', 'pdf']:
                word_count = file_data.get('word_count', 0)
                context += f"Content length: {word_count} words\n"
                if file_data.get('type') == 'word' and file_data.get('tables'):
                    context += f"Contains {len(file_data['tables'])} table(s)\n"
                    
                # Include text preview
                full_text = file_data.get('full_text', '')
                if full_text:
                    preview = full_text[:1200] + "..." if len(full_text) > 1200 else full_text
                    context += f"Content preview:\n{preview}\n"
            
            context += "\n" + "-"*50 + "\n"
        
        return base_msg + context
    
    def send_message(self, user_message: str, files_context: List[Dict[str, Any]] = None, 
                    conversation_history: List[Dict[str, str]] = None) -> Optional[str]:
        """Send message with enhanced context handling"""
        try:
            session_id = str(uuid.uuid4())
            
            # Create enhanced system message
            sys_msg = self.create_enhanced_system_message(files_context or [])
            
            # Build conversation with better context management
            messages = []
            
            # Add recent conversation history (last 8 messages for better context)
            if conversation_history:
                messages.extend(conversation_history[-8:])
            
            # Add current user message
            messages.append({
                "role": "user", 
                "content": user_message
            })
            
            payload = {
                "query": {
                    "aplctn_cd": self.aplctn_cd,
                    "app_id": self.app_id,
                    "api_key": self.api_key,
                    "method": "cortex",
                    "model": self.model,
                    "sys_msg": sys_msg,
                    "limit_convs": "0",
                    "prompt": {"messages": messages},
                    "app_lvl_prefix": "",
                    "user_id": "",
                    "session_id": session_id
                }
            }
            
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "Authorization": f'Snowflake Token="{self.api_key}"'
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload, verify=False, timeout=60)
            
            if response.status_code == 200:
                raw = response.text
                if "end_of_stream" in raw:
                    answer, _, _ = raw.partition("end_of_stream")
                    return answer.strip()
                return raw.strip()
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Request failed: {str(e)}")
            return None

def create_quick_questions():
    """Define quick questions for different file types"""
    return {
        "📊 Data Analysis": {
            "What are the key insights from my data?": "Analyze the uploaded files and provide the top 5 key insights with supporting data points and business implications.",
            "Show me summary statistics": "Provide comprehensive summary statistics for all numeric columns including mean, median, min, max, and distribution insights.",
            "What trends do you see?": "Identify and explain the main trends, patterns, and correlations in the data with specific examples and metrics.",
            "Are there any outliers or anomalies?": "Find and analyze any outliers, anomalies, or unusual patterns in the data and explain their potential significance."
        },
        "💼 Business Intelligence": {
            "What's the overall performance?": "Provide an executive summary of overall performance metrics with key KPIs and business insights.",
            "Compare different segments": "Compare and contrast different segments, categories, or groups in the data and highlight significant differences.",
            "What recommendations do you have?": "Based on the data analysis, provide specific actionable recommendations for business improvement.",
            "Identify top and bottom performers": "Identify the best and worst performing elements in the data with detailed analysis of contributing factors."
        },
        "📈 Financial Analysis": {
            "Analyze revenue trends": "Perform detailed revenue analysis including growth rates, seasonality, and forecasting insights.",
            "What about profitability?": "Analyze profitability metrics, margins, and identify factors affecting financial performance.",
            "Show cost breakdown": "Provide comprehensive cost analysis with categorization and identification of cost-saving opportunities.",
            "Calculate key financial ratios": "Calculate and interpret important financial ratios and metrics from the available data."
        },
        "📋 Document Analysis": {
            "Summarize the main points": "Provide a comprehensive summary of the main points, findings, and conclusions from the document.",
            "Extract key data and metrics": "Extract all numerical data, statistics, and key metrics mentioned in the document.",
            "What are the recommendations?": "Identify and summarize all recommendations, action items, and next steps mentioned in the document.",
            "Find important dates and deadlines": "Extract all important dates, deadlines, milestones, and time-sensitive information from the document."
        }
    }

def render_chat_message(role: str, content: str, timestamp: str):
    """Render individual chat messages with enhanced styling"""
    if role == "user":
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin: 20px 0;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 20px 25px; border-radius: 25px 25px 8px 25px; 
                        max-width: 75%; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
                        font-size: 17px; line-height: 1.6; font-weight: 500;">
                {content}
                <div style="font-size: 13px; opacity: 0.85; margin-top: 8px; text-align: right; font-weight: 400;">
                    You • {timestamp}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Format the assistant message with better markdown rendering
        formatted_content = content.replace('\n', '<br>')
        
        # Handle bold text
        import re
        formatted_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_content)
        formatted_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', formatted_content)
        
        # Handle bullet points
        formatted_content = re.sub(r'^\s*[-•]\s*(.+)

def main():
    """Enhanced main application with ChatGPT-like interface"""
    
    # Set pandas options to avoid warnings
    pd.set_option('future.no_silent_downcasting', True)
    
    # Page configuration
    st.set_page_config(
        page_title="AI Data Analyst",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced chat appearance
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stChatInput > div {
        background: white;
    }
    .chat-container {
        height: 70vh;
        overflow-y: auto;
        padding: 40px 30px;
        background: linear-gradient(to bottom, #fafafa 0%, #f0f9ff 100%);
        border-radius: 20px;
        margin-bottom: 30px;
        border: 1px solid #e5e7eb;
    }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    .chat-container::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    .chat-container::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 10px;
    }
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    .file-upload-container {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 25px;
        border: 1px solid #e5e7eb;
    }
    .quick-questions {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.08);
        margin: 20px 0;
        border: 1px solid #e5e7eb;
    }
    .stExpander {
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stExpander > div > div > div > div {
        padding: 15px 20px;
    }
    .stExpander summary {
        font-weight: 600;
        font-size: 15px;
        padding: 15px 20px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Enhanced input box */
    .stTextInput > div > div > input {
        padding: 18px 24px;
        border-radius: 30px;
        border: 3px solid #e5e7eb;
        font-size: 17px;
        font-weight: 500;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15);
        outline: none;
    }
    .stTextInput > div > div > input::placeholder {
        color: #9ca3af;
        font-style: italic;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        border-radius: 25px;
        border: none;
        font-weight: 700;
        font-size: 15px;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Welcome message enhancement */
    .welcome-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* File info styling */
    .file-info-card {
        background: linear-gradient(135deg, #e8f5e8 0%, #f0f9ff 100%);
        border-left: 5px solid #10b981;
        padding: 15px;
        border-radius: 10px;
        margin: 8px 0;
        font-size: 13px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Header enhancement */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        padding: 30px 0;
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "files_data" not in st.session_state:
        st.session_state.files_data = []
    if "uploaded_files_names" not in st.session_state:
        st.session_state.uploaded_files_names = []
    
    # Initialize API client
    api_client = SFAssistAPI()
    
    # Enhanced Header
    st.markdown("""
    <div class="main-header">
        <div style="font-size: 50px; margin-bottom: 15px;">🤖</div>
        <h1 style="margin: 0; font-size: 48px; font-weight: 800; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            AI Data Analyst
        </h1>
        <p style="margin: 15px 0 0 0; font-size: 20px; opacity: 0.95; font-weight: 500;">
            Upload multiple files and chat with your data using advanced AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file upload and quick questions
    with st.sidebar:
        st.markdown('<div class="file-upload-container">', unsafe_allow_html=True)
        st.header("📁 Upload Files")
        
        # Determine supported file types
        supported_types = ['xlsx', 'xls']
        type_descriptions = ["Excel (.xlsx, .xls)"]
        
        if DOCX_AVAILABLE:
            supported_types.extend(['docx'])
            type_descriptions.append("Word (.docx)")
        
        if PDF_AVAILABLE:
            supported_types.extend(['pdf'])
            type_descriptions.append("PDF (.pdf)")
        
        help_text = f"Supported: {', '.join(type_descriptions)}"
        
        # Multiple file uploader
        uploaded_files = st.file_uploader(
            "Choose files (multiple files supported)",
            type=supported_types,
            help=help_text,
            accept_multiple_files=True
        )
        
        # Process uploaded files
        if uploaded_files:
            current_file_names = [f.name for f in uploaded_files]
            
            # Check if we have new files
            if current_file_names != st.session_state.uploaded_files_names:
                st.session_state.uploaded_files_names = current_file_names
                st.session_state.files_data = []
                
                # Process each file
                with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
                    for uploaded_file in uploaded_files:
                        file_extension = uploaded_file.name.split('.')[-1].lower()
                        
                        if file_extension in ['xlsx', 'xls']:
                            file_data = FileProcessor.process_excel(uploaded_file)
                        elif file_extension == 'docx':
                            file_data = FileProcessor.process_word(uploaded_file)
                        elif file_extension == 'pdf':
                            file_data = FileProcessor.process_pdf(uploaded_file)
                        else:
                            continue
                            
                        if file_data and "error" not in file_data:
                            st.session_state.files_data.append(file_data)
                
                if st.session_state.files_data:
                    st.success(f"✅ {len(st.session_state.files_data)} file(s) processed successfully!")
        
        # Display uploaded files info
        if st.session_state.files_data:
            st.markdown("### 📋 Uploaded Files")
            for file_data in st.session_state.files_data:
                if "error" not in file_data:
                    file_icon = {"excel": "📊", "word": "📄", "pdf": "📕"}.get(file_data.get('type'), "📄")
                    st.markdown(f"""
                    <div class="file-info-card">
                        <div style="font-weight: 700; font-size: 15px; color: #1f2937; margin-bottom: 8px;">
                            {file_icon} {file_data.get('filename', 'Unknown')}
                        </div>
                        <div style="color: #6b7280; font-size: 13px; font-style: italic; margin-bottom: 6px;">
                            {file_data.get('type', 'Unknown').upper()} File
                        </div>
                        <div style="color: #374151; font-size: 12px; line-height: 1.4;">
                            {file_data.get('summary', 'No summary')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick Questions in Sidebar
        st.markdown('<div class="quick-questions">', unsafe_allow_html=True)
        st.markdown("### 🎯 Quick Questions")
        
        quick_questions = create_quick_questions()
        
        for category, questions in quick_questions.items():
            with st.expander(category):
                for question, detailed_prompt in questions.items():
                    if st.button(question, key=f"q_{hash(question)}", help="Click to ask this question", use_container_width=True):
                        if not st.session_state.files_data:
                            st.warning("Upload files first!")
                        else:
                            # Auto-send the detailed prompt
                            with st.spinner("🤔 Analyzing..."):
                                conversation_history = []
                                for role, message, _ in st.session_state.messages[-6:]:
                                    conversation_history.append({"role": role, "content": message})
                                
                                response = api_client.send_message(
                                    detailed_prompt,
                                    st.session_state.files_data,
                                    conversation_history
                                )
                                
                                if response:
                                    timestamp = datetime.now().strftime("%H:%M")
                                    st.session_state.messages.append(("user", question, timestamp))
                                    st.session_state.messages.append(("assistant", response, timestamp))
                                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Library status
        if not DOCX_AVAILABLE or not PDF_AVAILABLE:
            st.markdown("### ⚠️ Library Status")
            if not DOCX_AVAILABLE:
                st.warning("Word support disabled")
            if not PDF_AVAILABLE:
                st.warning("PDF support disabled")
    
    # Main chat area (full width)
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.messages:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for role, message, timestamp in st.session_state.messages:
                render_chat_message(role, message, timestamp)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 80px 50px; background: white; 
                        border-radius: 25px; margin: 40px 0; 
                        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                        border: 1px solid #e5e7eb;">
                <div style="font-size: 60px; margin-bottom: 20px;">🤖</div>
                <h1 style="color: #1f2937; font-size: 32px; font-weight: 800; margin-bottom: 20px; 
                           background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                           background-clip: text;">
                    Welcome to AI Data Analyst!
                </h1>
                <p style="font-size: 20px; color: #4b5563; margin: 25px 0; font-weight: 500;">
                    Upload your files in the sidebar and start asking questions about your data.
                </p>
                <p style="font-size: 17px; color: #6b7280; opacity: 0.9; margin-bottom: 40px;">
                    Use the quick questions in the sidebar to get started, or type your own question below.
                </p>
                <div style="margin-top: 40px;">
                    <span style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                                 padding: 12px 20px; border-radius: 25px; margin: 0 8px; 
                                 font-size: 15px; font-weight: 600; color: #1565c0;
                                 box-shadow: 0 3px 8px rgba(21, 101, 192, 0.2);">📊 Data Analysis</span>
                    <span style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); 
                                 padding: 12px 20px; border-radius: 25px; margin: 0 8px; 
                                 font-size: 15px; font-weight: 600; color: #7b1fa2;
                                 box-shadow: 0 3px 8px rgba(123, 31, 162, 0.2);">💼 Business Intelligence</span>
                    <span style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); 
                                 padding: 12px 20px; border-radius: 25px; margin: 0 8px; 
                                 font-size: 15px; font-weight: 600; color: #2e7d32;
                                 box-shadow: 0 3px 8px rgba(46, 125, 50, 0.2);">📈 Financial Analysis</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input at bottom (ChatGPT style) - Full width
    st.markdown("""
    <div style="margin: 40px 0 20px 0;">
        <h3 style="font-size: 24px; font-weight: 700; color: #1f2937; margin-bottom: 15px; 
                   display: flex; align-items: center;">
            <span style="font-size: 28px; margin-right: 10px;">💬</span> 
            Ask about your data
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    col_input, col_send, col_clear = st.columns([5, 1, 1])
    
    with col_input:
        user_input = st.text_input(
            "Type your question here...",
            placeholder="e.g., What are the key insights from my data?",
            label_visibility="collapsed"
        )
    
    with col_send:
        send_clicked = st.button("Send 🚀", use_container_width=True)
    
    with col_clear:
        clear_clicked = st.button("Clear 🗑️", use_container_width=True)
    
    if clear_clicked:
        st.session_state.messages = []
        st.rerun()
    
    # Process input
    if (send_clicked and user_input.strip()) or (user_input and len(user_input) > 0 and st.session_state.get('auto_send', False)):
        if st.session_state.get('auto_send', False):
            st.session_state['auto_send'] = False
        
        if not st.session_state.files_data:
            st.warning("Please upload files first to start analysis!")
        else:
            with st.spinner("🤔 Analyzing your question..."):
                # Get conversation history
                conversation_history = []
                for role, message, _ in st.session_state.messages[-6:]:
                    conversation_history.append({"role": role, "content": message})
                
                # Send to API
                response = api_client.send_message(
                    user_input, 
                    st.session_state.files_data,
                    conversation_history
                )
                
                if response:
                    timestamp = datetime.now().strftime("%H:%M")
                    st.session_state.messages.append(("user", user_input, timestamp))
                    st.session_state.messages.append(("assistant", response, timestamp))
                    st.rerun()

if __name__ == "__main__":
    main(), r'&nbsp;&nbsp;• \1', formatted_content, flags=re.MULTILINE)
        
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start; margin: 25px 0;">
            <div style="display: flex; align-items: flex-start; max-width: 85%;">
                <div style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); 
                            color: white; width: 45px; height: 45px; border-radius: 50%; 
                            display: flex; align-items: center; justify-content: center; 
                            margin-right: 15px; font-size: 20px; font-weight: bold;
                            box-shadow: 0 3px 10px rgba(79, 70, 229, 0.3);">
                    🤖
                </div>
                <div style="background: white; color: #1f2937; padding: 25px 30px; 
                            border-radius: 25px 25px 25px 8px; 
                            box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
                            border-left: 4px solid #4f46e5;
                            font-size: 16px; line-height: 1.7; flex: 1;">
                    <div style="margin-bottom: 15px;">
                        {formatted_content}
                    </div>
                    <div style="font-size: 12px; color: #6b7280; border-top: 1px solid #f3f4f6; 
                                padding-top: 10px; font-weight: 500;">
                        🤖 AI Data Analyst • {timestamp}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Enhanced main application with ChatGPT-like interface"""
    
    # Set pandas options to avoid warnings
    pd.set_option('future.no_silent_downcasting', True)
    
    # Page configuration
    st.set_page_config(
        page_title="AI Data Analyst",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for ChatGPT-like appearance
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stChatInput > div {
        background: white;
    }
    .chat-container {
        height: 70vh;
        overflow-y: auto;
        padding: 30px 20px;
        background: #fafafa;
        border-radius: 15px;
        margin-bottom: 30px;
    }
    .file-upload-container {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .quick-questions {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 15px 0;
    }
    .stExpander {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        margin: 8px 0;
    }
    .stExpander > div > div > div > div {
        padding: 10px 15px;
    }
    .question-btn {
        width: 100%;
        text-align: left;
        padding: 10px 15px;
        margin: 4px 0;
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        cursor: pointer;
        font-size: 13px;
        transition: all 0.2s ease;
    }
    .question-btn:hover {
        background: #e9ecef;
        transform: translateY(-1px);
    }
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 2rem;
    }
    /* Make input box more prominent */
    .stTextInput > div > div > input {
        padding: 12px 16px;
        border-radius: 25px;
        border: 2px solid #e5e7eb;
        font-size: 16px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    /* Button styling */
    .stButton > button {
        border-radius: 20px;
        border: none;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "files_data" not in st.session_state:
        st.session_state.files_data = []
    if "uploaded_files_names" not in st.session_state:
        st.session_state.uploaded_files_names = []
    
    # Initialize API client
    api_client = SFAssistAPI()
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2.5em;">🤖 AI Data Analyst</h1>
        <p style="margin: 10px 0 0 0; font-size: 1.1em; opacity: 0.9;">
            Upload multiple files and chat with your data using AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file upload and quick questions
    with st.sidebar:
        st.markdown('<div class="file-upload-container">', unsafe_allow_html=True)
        st.header("📁 Upload Files")
        
        # Determine supported file types
        supported_types = ['xlsx', 'xls']
        type_descriptions = ["Excel (.xlsx, .xls)"]
        
        if DOCX_AVAILABLE:
            supported_types.extend(['docx'])
            type_descriptions.append("Word (.docx)")
        
        if PDF_AVAILABLE:
            supported_types.extend(['pdf'])
            type_descriptions.append("PDF (.pdf)")
        
        help_text = f"Supported: {', '.join(type_descriptions)}"
        
        # Multiple file uploader
        uploaded_files = st.file_uploader(
            "Choose files (multiple files supported)",
            type=supported_types,
            help=help_text,
            accept_multiple_files=True
        )
        
        # Process uploaded files
        if uploaded_files:
            current_file_names = [f.name for f in uploaded_files]
            
            # Check if we have new files
            if current_file_names != st.session_state.uploaded_files_names:
                st.session_state.uploaded_files_names = current_file_names
                st.session_state.files_data = []
                
                # Process each file
                with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
                    for uploaded_file in uploaded_files:
                        file_extension = uploaded_file.name.split('.')[-1].lower()
                        
                        if file_extension in ['xlsx', 'xls']:
                            file_data = FileProcessor.process_excel(uploaded_file)
                        elif file_extension == 'docx':
                            file_data = FileProcessor.process_word(uploaded_file)
                        elif file_extension == 'pdf':
                            file_data = FileProcessor.process_pdf(uploaded_file)
                        else:
                            continue
                            
                        if file_data and "error" not in file_data:
                            st.session_state.files_data.append(file_data)
                
                if st.session_state.files_data:
                    st.success(f"✅ {len(st.session_state.files_data)} file(s) processed successfully!")
        
        # Display uploaded files info
        if st.session_state.files_data:
            st.markdown("### 📋 Uploaded Files")
            for file_data in st.session_state.files_data:
                if "error" not in file_data:
                    st.markdown(f"""
                    <div style="background: #e8f5e8; padding: 10px; border-radius: 5px; margin: 5px 0; font-size: 12px;">
                        <strong>{file_data.get('filename', 'Unknown')}</strong><br>
                        <em>{file_data.get('type', 'Unknown').upper()}</em><br>
                        {file_data.get('summary', 'No summary')}
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick Questions in Sidebar
        st.markdown('<div class="quick-questions">', unsafe_allow_html=True)
        st.markdown("### 🎯 Quick Questions")
        
        quick_questions = create_quick_questions()
        
        for category, questions in quick_questions.items():
            with st.expander(category):
                for question, detailed_prompt in questions.items():
                    if st.button(question, key=f"q_{hash(question)}", help="Click to ask this question", use_container_width=True):
                        if not st.session_state.files_data:
                            st.warning("Upload files first!")
                        else:
                            # Auto-send the detailed prompt
                            with st.spinner("🤔 Analyzing..."):
                                conversation_history = []
                                for role, message, _ in st.session_state.messages[-6:]:
                                    conversation_history.append({"role": role, "content": message})
                                
                                response = api_client.send_message(
                                    detailed_prompt,
                                    st.session_state.files_data,
                                    conversation_history
                                )
                                
                                if response:
                                    timestamp = datetime.now().strftime("%H:%M")
                                    st.session_state.messages.append(("user", question, timestamp))
                                    st.session_state.messages.append(("assistant", response, timestamp))
                                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Library status
        if not DOCX_AVAILABLE or not PDF_AVAILABLE:
            st.markdown("### ⚠️ Library Status")
            if not DOCX_AVAILABLE:
                st.warning("Word support disabled")
            if not PDF_AVAILABLE:
                st.warning("PDF support disabled")
    
    # Main chat area (full width)
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.messages:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for role, message, timestamp in st.session_state.messages:
                render_chat_message(role, message, timestamp)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 60px 40px; color: #6b7280; background: #fafafa; border-radius: 15px; margin: 30px 0;">
                <h2>👋 Welcome to AI Data Analyst!</h2>
                <p style="font-size: 18px; margin: 20px 0;">Upload your files in the sidebar and start asking questions about your data.</p>
                <p style="font-size: 16px; opacity: 0.8;">Use the quick questions in the sidebar to get started, or type your own question below.</p>
                <div style="margin-top: 30px;">
                    <span style="background: #e3f2fd; padding: 8px 16px; border-radius: 20px; margin: 0 5px; font-size: 14px;">📊 Data Analysis</span>
                    <span style="background: #f3e5f5; padding: 8px 16px; border-radius: 20px; margin: 0 5px; font-size: 14px;">💼 Business Intelligence</span>
                    <span style="background: #e8f5e8; padding: 8px 16px; border-radius: 20px; margin: 0 5px; font-size: 14px;">📈 Financial Analysis</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input at bottom (ChatGPT style) - Full width
    st.markdown("### 💬 Ask about your data")
    
    col_input, col_send, col_clear = st.columns([5, 1, 1])
    
    with col_input:
        user_input = st.text_input(
            "Type your question here...",
            placeholder="e.g., What are the key insights from my data?",
            label_visibility="collapsed"
        )
    
    with col_send:
        send_clicked = st.button("Send 🚀", use_container_width=True)
    
    with col_clear:
        clear_clicked = st.button("Clear 🗑️", use_container_width=True)
    
    if clear_clicked:
        st.session_state.messages = []
        st.rerun()
    
    # Process input
    if (send_clicked and user_input.strip()) or (user_input and len(user_input) > 0 and st.session_state.get('auto_send', False)):
        if st.session_state.get('auto_send', False):
            st.session_state['auto_send'] = False
        
        if not st.session_state.files_data:
            st.warning("Please upload files first to start analysis!")
        else:
            with st.spinner("🤔 Analyzing your question..."):
                # Get conversation history
                conversation_history = []
                for role, message, _ in st.session_state.messages[-6:]:
                    conversation_history.append({"role": role, "content": message})
                
                # Send to API
                response = api_client.send_message(
                    user_input, 
                    st.session_state.files_data,
                    conversation_history
                )
                
                if response:
                    timestamp = datetime.now().strftime("%H:%M")
                    st.session_state.messages.append(("user", user_input, timestamp))
                    st.session_state.messages.append(("assistant", response, timestamp))
                    st.rerun()

if __name__ == "__main__":
    main()
