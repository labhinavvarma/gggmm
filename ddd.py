import streamlit as st
import requests
import json
import uuid
import pandas as pd
import PyPDF2
import docx
import io
import base64
from typing import Dict, Any, List, Optional
import time
from datetime import datetime
import warnings

# Disable SSL warnings
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except (ImportError, AttributeError):
    # If urllib3.exceptions is not available, use warnings filter
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')

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
            # Read all sheets
            excel_data = pd.read_excel(file, sheet_name=None)
            
            result = {
                "type": "excel",
                "sheets": {},
                "summary": "",
                "raw_data": {}
            }
            
            for sheet_name, df in excel_data.items():
                # Basic info about the sheet
                sheet_info = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "sample_data": df.head(5).to_dict('records'),
                    "data_types": df.dtypes.to_dict()
                }
                
                # Convert data types to strings for JSON serialization
                sheet_info["data_types"] = {k: str(v) for k, v in sheet_info["data_types"].items()}
                
                result["sheets"][sheet_name] = sheet_info
                result["raw_data"][sheet_name] = df.to_dict('records')
                
            # Create summary
            total_rows = sum(info["rows"] for info in result["sheets"].values())
            total_sheets = len(result["sheets"])
            result["summary"] = f"Excel file with {total_sheets} sheet(s) containing {total_rows} total rows."
            
            return result
            
        except Exception as e:
            st.error(f"Error processing Excel file: {str(e)}")
            return {"type": "excel", "error": str(e)}
    
    @staticmethod
    def process_word(file) -> Dict[str, Any]:
        """Process Word documents and extract text"""
        try:
            doc = docx.Document(file)
            
            paragraphs = []
            tables_data = []
            
            # Extract paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text.strip())
            
            # Extract tables
            for table in doc.tables:
                table_data = []
                for row in table.rows:
                    row_data = []
                    for cell in row.cells:
                        row_data.append(cell.text.strip())
                    table_data.append(row_data)
                tables_data.append(table_data)
            
            full_text = "\n".join(paragraphs)
            
            return {
                "type": "word",
                "paragraphs": paragraphs,
                "tables": tables_data,
                "full_text": full_text,
                "summary": f"Word document with {len(paragraphs)} paragraphs and {len(tables_data)} tables.",
                "word_count": len(full_text.split())
            }
            
        except Exception as e:
            st.error(f"Error processing Word file: {str(e)}")
            return {"type": "word", "error": str(e)}
    
    @staticmethod
    def process_pdf(file) -> Dict[str, Any]:
        """Process PDF files and extract text"""
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
                "pages": pages_text,
                "full_text": full_text.strip(),
                "summary": f"PDF document with {len(pdf_reader.pages)} pages.",
                "page_count": len(pdf_reader.pages)
            }
            
        except Exception as e:
            st.error(f"Error processing PDF file: {str(e)}")
            return {"type": "pdf", "error": str(e)}

class SFAssistAPI:
    """Handle communication with SF Assist API"""
    
    def __init__(self):
        self.api_url = API_URL
        self.api_key = API_KEY
        self.app_id = APP_ID
        self.aplctn_cd = APLCTN_CD
        self.model = MODEL
    
    def create_system_message(self, file_data: Dict[str, Any]) -> str:
        """Create system message with file context"""
        base_msg = """You are an expert data analyst AI assistant. You have access to the user's uploaded file(s) and should provide accurate, insightful analysis based on the data provided.

IMPORTANT GUIDELINES:
1. Always reference the specific data from the uploaded file when answering questions
2. Provide detailed analysis including statistics, trends, and insights when applicable
3. If asked about data not present in the file, clearly state that information is not available
4. For Excel files, you can reference specific sheets, columns, and data points
5. For text documents (Word/PDF), reference specific sections or content
6. Provide actionable insights and recommendations when appropriate
7. Use clear, professional language suitable for business analysis

"""
        
        if file_data.get("type") == "excel":
            context = f"""
UPLOADED EXCEL FILE CONTEXT:
{file_data.get('summary', '')}

SHEETS AND STRUCTURE:
"""
            for sheet_name, sheet_info in file_data.get("sheets", {}).items():
                context += f"""
Sheet: {sheet_name}
- Rows: {sheet_info['rows']}
- Columns: {sheet_info['columns']} ({', '.join(sheet_info['column_names'])})
- Sample data preview: {json.dumps(sheet_info['sample_data'][:3], indent=2)}
"""
        
        elif file_data.get("type") == "word":
            context = f"""
UPLOADED WORD DOCUMENT CONTEXT:
{file_data.get('summary', '')}
Word count: {file_data.get('word_count', 0)}

CONTENT PREVIEW:
{file_data.get('full_text', '')[:1500]}...
"""
        
        elif file_data.get("type") == "pdf":
            context = f"""
UPLOADED PDF DOCUMENT CONTEXT:
{file_data.get('summary', '')}

CONTENT PREVIEW:
{file_data.get('full_text', '')[:1500]}...
"""
        
        else:
            context = "No file context available."
        
        return base_msg + context
    
    def send_message(self, user_message: str, file_context: Optional[Dict[str, Any]] = None, 
                    conversation_history: List[Dict[str, str]] = None) -> Optional[str]:
        """Send message to SF Assist API with file context"""
        try:
            session_id = str(uuid.uuid4())
            
            # Create system message with file context
            if file_context:
                sys_msg = self.create_system_message(file_context)
            else:
                sys_msg = "You are a helpful AI assistant. Provide accurate, concise answers."
            
            # Build message history
            messages = []
            
            # Add conversation history if available
            if conversation_history:
                messages.extend(conversation_history)
            
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
                    "prompt": {
                        "messages": messages
                    },
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
            
            response = requests.post(self.api_url, headers=headers, json=payload, verify=False)
            
            if response.status_code == 200:
                raw = response.text
                
                if "end_of_stream" in raw:
                    answer, _, _ = raw.partition("end_of_stream")
                    bot_reply = answer.strip()
                else:
                    bot_reply = raw.strip()
                
                return bot_reply
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Request failed: {str(e)}")
            return None

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="GenAI Data Analyst",
        page_icon="📊",
        layout="wide"
    )
    
    # Title and description
    st.title("📊 GenAI Data Analyst")
    st.markdown("Upload your files (Excel, Word, PDF) and ask questions to get AI-powered analysis and insights!")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "file_data" not in st.session_state:
        st.session_state.file_data = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    
    # Initialize API client
    api_client = SFAssistAPI()
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 File Upload")
        
        uploaded_file = st.file_uploader(
            "Upload your file",
            type=['xlsx', 'xls', 'docx', 'pdf'],
            help="Supported formats: Excel (.xlsx, .xls), Word (.docx), PDF (.pdf)"
        )
        
        if uploaded_file is not None:
            # Process file if it's new or different
            if st.session_state.uploaded_file_name != uploaded_file.name:
                with st.spinner("Processing file..."):
                    st.session_state.uploaded_file_name = uploaded_file.name
                    
                    # Reset chat history when new file is uploaded
                    st.session_state.messages = []
                    
                    # Process based on file type
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_extension in ['xlsx', 'xls']:
                        st.session_state.file_data = FileProcessor.process_excel(uploaded_file)
                    elif file_extension == 'docx':
                        st.session_state.file_data = FileProcessor.process_word(uploaded_file)
                    elif file_extension == 'pdf':
                        st.session_state.file_data = FileProcessor.process_pdf(uploaded_file)
                    
                    if st.session_state.file_data and "error" not in st.session_state.file_data:
                        st.success(f"✅ {uploaded_file.name} processed successfully!")
                    else:
                        st.error("❌ Error processing file")
            
            # Display file information
            if st.session_state.file_data and "error" not in st.session_state.file_data:
                st.info(f"**Current File:** {uploaded_file.name}")
                st.write(f"**Type:** {st.session_state.file_data.get('type', 'Unknown').upper()}")
                st.write(f"**Summary:** {st.session_state.file_data.get('summary', 'N/A')}")
                
                # Show additional info based on file type
                if st.session_state.file_data.get('type') == 'excel':
                    st.write(f"**Sheets:** {', '.join(st.session_state.file_data.get('sheets', {}).keys())}")
                elif st.session_state.file_data.get('type') == 'word':
                    st.write(f"**Word Count:** {st.session_state.file_data.get('word_count', 0)}")
                elif st.session_state.file_data.get('type') == 'pdf':
                    st.write(f"**Pages:** {st.session_state.file_data.get('page_count', 0)}")
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("💬 Chat with your Data")
        
        # Chat input
        if st.session_state.file_data and "error" not in st.session_state.file_data:
            with st.form("chat_form", clear_on_submit=True):
                user_query = st.text_area(
                    "Ask a question about your data:",
                    height=100,
                    placeholder="e.g., 'What are the key trends in this data?' or 'Summarize the main findings from the document'"
                )
                col_a, col_b = st.columns([1, 4])
                with col_a:
                    submitted = st.form_submit_button("Send 🚀", use_container_width=True)
                with col_b:
                    clear_chat = st.form_submit_button("Clear Chat 🗑️", use_container_width=True)
                
                if clear_chat:
                    st.session_state.messages = []
                    st.rerun()
            
            # Process message
            if submitted and user_query.strip():
                with st.spinner("Analyzing your question..."):
                    # Get conversation history for context
                    conversation_history = []
                    for role, message, timestamp in st.session_state.messages[-6:]:  # Last 3 exchanges
                        conversation_history.append({
                            "role": role,
                            "content": message
                        })
                    
                    # Send to API
                    response = api_client.send_message(
                        user_query, 
                        st.session_state.file_data,
                        conversation_history
                    )
                    
                    if response:
                        # Add to conversation history
                        timestamp = datetime.now().strftime("%H:%M")
                        st.session_state.messages.append(("user", user_query, timestamp))
                        st.session_state.messages.append(("assistant", response, timestamp))
        else:
            st.info("👆 Please upload a file in the sidebar to start chatting with your data!")
    
    with col2:
        st.header("🎯 Quick Tips")
        st.markdown("""
        **Excel Files:**
        - "Show me the column names"
        - "What's the average of [column]?"
        - "Find trends in the data"
        - "Create a summary report"
        
        **Documents:**
        - "Summarize this document"
        - "What are the key points?"
        - "Extract important data"
        - "Find specific information about [topic]"
        
        **General:**
        - "Analyze the data quality"
        - "What insights can you provide?"
        - "Create recommendations"
        """)
    
    # Display chat history
    if st.session_state.messages:
        st.divider()
        st.header("💭 Conversation History")
        
        for role, message, timestamp in reversed(st.session_state.messages):
            with st.container():
                if role == "user":
                    st.markdown(f"""
                    <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0;">
                        <strong>🧑 You ({timestamp}):</strong><br>
                        {message}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #f1f8e9; padding: 10px; border-radius: 10px; margin: 5px 0;">
                        <strong>🤖 AI Analyst ({timestamp}):</strong><br>
                        {message}
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
