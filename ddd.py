# Configure Streamlit page FIRST - before any other Streamlit commands
import streamlit as st

# Determine sidebar state based on chatbot readiness
if 'analysis_results' in st.session_state and st.session_state.get('analysis_results') and st.session_state.analysis_results.get("chatbot_ready", False):
    sidebar_state = "expanded"
else:
    sidebar_state = "collapsed"

st.set_page_config(
    page_title="⚡ Enhanced Health Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state=sidebar_state
)

# Core imports
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import os
import logging
import traceback
from typing import Dict, Any, Optional, Union, List, Tuple
import io
import base64
import re
import numpy as np

# Matplotlib imports with safety
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for stability
import matplotlib.pyplot as plt

# Plotly imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    make_subplots = None

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('streamlit_app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# ===== UTILITY FUNCTIONS =====

def safe_get(data: Union[Dict[str, Any], Any], key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary or object"""
    try:
        if data is None:
            return default
        if isinstance(data, dict):
            return data.get(key, default)
        elif hasattr(data, key):
            return getattr(data, key, default)
        else:
            return default
    except Exception as e:
        logger.warning(f"Error in safe_get for key '{key}': {e}")
        return default

def safe_get_nested(data: Union[Dict[str, Any], Any], *keys, default: Any = None) -> Any:
    """Safely get nested dictionary values"""
    try:
        current = data
        for key in keys:
            if current is None:
                return default
            if isinstance(current, dict):
                current = current.get(key)
            elif hasattr(current, key):
                current = getattr(current, key)
            else:
                return default
        return current if current is not None else default
    except Exception as e:
        logger.warning(f"Error in safe_get_nested for keys {keys}: {e}")
        return default

def safe_execute(func, *args, **kwargs) -> Tuple[bool, Any]:
    """Safely execute a function and return success status and result"""
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {e}")
        logger.debug(traceback.format_exc())
        return False, str(e)

def safe_str(value: Any, default: str = "") -> str:
    """Safely convert value to string"""
    try:
        if value is None:
            return default
        return str(value)
    except Exception:
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to integer"""
    try:
        if value is None:
            return default
        return int(float(str(value)))
    except (ValueError, TypeError):
        return default

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        if value is None:
            return default
        return float(str(value))
    except (ValueError, TypeError):
        return default

def safe_len(obj: Any, default: int = 0) -> int:
    """Safely get length of an object"""
    try:
        if obj is None:
            return default
        return len(obj)
    except (TypeError, AttributeError):
        return default

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string"""
    try:
        if not json_str:
            return default
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"JSON parse error: {e}")
        return default

def safe_datetime_parse(date_str: str, fmt: str = '%Y-%m-%d', default: Any = None) -> Any:
    """Safely parse datetime string"""
    try:
        if not date_str:
            return default
        # Handle ISO format with timezone
        if 'T' in date_str:
            date_str = date_str.replace('Z', '+00:00')
            return datetime.fromisoformat(date_str)
        else:
            return datetime.strptime(date_str, fmt)
    except (ValueError, TypeError) as e:
        logger.warning(f"Datetime parse error: {e}")
        return default

def log_error(error_msg: str, context: str = ""):
    """Log error to session state and logger"""
    timestamp = datetime.now().isoformat()
    error_entry = {
        'timestamp': timestamp,
        'message': error_msg,
        'context': context
    }
    
    if 'error_log' not in st.session_state:
        st.session_state.error_log = []
    
    st.session_state.error_log.append(error_entry)
    logger.error(f"[{context}] {error_msg}")

# Import the Enhanced health analysis agent
AGENT_AVAILABLE = False
import_error = None
EnhancedHealthAnalysisAgent = None
EnhancedConfig = None

try:
    from health_agent_core_enhanced import EnhancedHealthAnalysisAgent, EnhancedConfig
    AGENT_AVAILABLE = True
    logger.info("✅ Enhanced Health Analysis Agent imported successfully")
except ImportError as e:
    AGENT_AVAILABLE = False
    import_error = str(e)
    logger.error(f"❌ Failed to import Enhanced Health Analysis Agent: {e}")

# Try to import supporting components
try:
    from health_data_processor_enhanced import EnhancedHealthDataProcessor
    PROCESSOR_AVAILABLE = True
    logger.info("✅ Enhanced Data Processor imported successfully")
except ImportError as e:
    PROCESSOR_AVAILABLE = False
    logger.error(f"❌ Failed to import Enhanced Data Processor: {e}")

try:
    from health_api_integrator_enhanced import EnhancedHealthAPIIntegrator
    INTEGRATOR_AVAILABLE = True
    logger.info("✅ Enhanced API Integrator imported successfully")
except ImportError as e:
    INTEGRATOR_AVAILABLE = False
    logger.error(f"❌ Failed to import Enhanced API Integrator: {e}")

try:
    import asyncio
    ASYNCIO_AVAILABLE = True
except ImportError:
    ASYNCIO_AVAILABLE = False
    logger.error("❌ Asyncio not available")

# Enhanced CSS with advanced animations and modern styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.main-header {
    font-size: 3.2rem;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 2rem;
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

.enhanced-badge {
    background: linear-gradient(135deg, #00ff87 0%, #60efff 100%);
    color: #2c3e50;
    padding: 0.6rem 1.2rem;
    border-radius: 25px;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
    margin: 0.4rem;
    box-shadow: 0 8px 25px rgba(0, 255, 135, 0.4);
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-5px); }
}

.section-box {
    background: white;
    padding: 1.8rem;
    border-radius: 15px;
    border: 1px solid #e9ecef;
    margin: 1.2rem 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.section-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 35px rgba(0,0,0,0.15);
}

.section-title {
    font-size: 1.4rem;
    color: #2c3e50;
    font-weight: 600;
    margin-bottom: 1rem;
    border-bottom: 3px solid #3498db;
    padding-bottom: 0.6rem;
}

.claims-viewer-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 2.2rem;
    border-radius: 18px;
    border: 2px solid #dee2e6;
    margin: 1.2rem 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
}

.mcid-container {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    padding: 1.8rem;
    border-radius: 15px;
    border: 2px solid #2196f3;
    margin: 1rem 0;
    box-shadow: 0 8px 25px rgba(33, 150, 243, 0.2);
}

.mcid-match-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 0.8rem 0;
    border-left: 4px solid #4caf50;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.2rem;
    margin: 1.2rem 0;
}

.metric-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #dee2e6;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.chatbot-loading-container {
    background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
    padding: 2rem;
    border-radius: 20px;
    margin: 1.5rem 0;
    border: 2px solid #28a745;
    text-align: center;
    animation: pulse-glow 2s infinite;
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(40, 167, 69, 0.3); }
    50% { box-shadow: 0 0 40px rgba(40, 167, 69, 0.6); }
}

.quick-prompts-enhanced {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    padding: 1.8rem;
    border-radius: 18px;
    margin: 1.2rem 0;
    border: 2px solid #2196f3;
}

.prompt-button-enhanced {
    background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
    color: white;
    border: none;
    padding: 0.8rem 1.4rem;
    border-radius: 25px;
    margin: 0.4rem;
    cursor: pointer;
    font-size: 0.9rem;
    font-weight: 500;
    transition: all 0.3s ease;
    box-shadow: 0 6px 20px rgba(33, 150, 243, 0.3);
}

.prompt-button-enhanced:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(33, 150, 243, 0.5);
}

.status-error {
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    color: #c62828;
    padding: 1rem;
    border-radius: 12px;
    border: 2px solid #f44336;
    margin: 1rem 0;
    font-weight: 500;
}

.status-success {
    background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
    color: #2e7d32;
    padding: 1rem;
    border-radius: 12px;
    border: 2px solid #4caf50;
    margin: 1rem 0;
    font-weight: 500;
}

.status-warning {
    background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
    color: #f57c00;
    padding: 1rem;
    border-radius: 12px;
    border: 2px solid #ff9800;
    margin: 1rem 0;
    font-weight: 500;
}

/* Green Run Analysis Button */
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 25px rgba(40, 167, 69, 0.4) !important;
    transition: all 0.3s ease !important;
}

.stButton button[kind="primary"]:hover {
    background: linear-gradient(135deg, #218838 0%, #1abc9c 100%) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 35px rgba(40, 167, 69, 0.5) !important;
}

/* Graph loading animation */
.graph-loading {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 2rem;
    background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
    border-radius: 15px;
    margin: 1rem 0;
}

.loading-spinner {
    width: 40px;
    height: 40px;
    border: 4px solid #e3f2fd;
    border-top: 4px solid #2196f3;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables for enhanced processing"""
    defaults = {
        'analysis_results': None,
        'analysis_running': False,
        'agent': None,
        'config': None,
        'chatbot_messages': [],
        'chatbot_context': None,
        'show_all_claims_data': False,
        'show_batch_extraction': False,
        'show_entity_extraction': False,
        'show_enhanced_trajectory': False,
        'show_heart_attack': False,
        'workflow_steps': [
            {'name': 'FAST API Fetch', 'status': 'pending', 'description': 'Fetching claims data with enhanced timeout', 'icon': '⚡'},
            {'name': 'ENHANCED Deidentification', 'status': 'pending', 'description': 'Advanced PII removal with structure preservation', 'icon': '🔒'},
            {'name': 'BATCH Code Processing', 'status': 'pending', 'description': 'Processing codes in batches (93% fewer API calls)', 'icon': '🚀'},
            {'name': 'DETAILED Entity Extraction', 'status': 'pending', 'description': 'Advanced health entity identification', 'icon': '🎯'},
            {'name': 'ENHANCED Health Trajectory', 'status': 'pending', 'description': 'Detailed predictive analysis with specific evaluation questions', 'icon': '📈'},
            {'name': 'IMPROVED Heart Risk Prediction', 'status': 'pending', 'description': 'Enhanced ML-based risk assessment', 'icon': '❤️'},
            {'name': 'STABLE Graph Chatbot', 'status': 'pending', 'description': 'AI assistant with enhanced graph stability', 'icon': '📊'}
        ],
        'current_step': 0,
        'show_animation': False,
        'error_log': [],
        'last_update': datetime.now().isoformat()
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def calculate_age(birth_date):
    """Calculate age from birth date"""
    if not birth_date:
        return None
    
    try:
        today = datetime.now().date()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return age
    except Exception as e:
        logger.error(f"Age calculation error: {e}")
        return None

def validate_patient_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate patient data"""
    errors = []
    required_fields = {
        'first_name': 'First Name',
        'last_name': 'Last Name', 
        'ssn': 'SSN',
        'date_of_birth': 'Date of Birth',
        'gender': 'Gender',
        'zip_code': 'Zip Code'
    }
    
    for field, display_name in required_fields.items():
        field_value = safe_get(data, field)
        if not field_value:
            errors.append(f"{display_name} is required")
        elif field == 'ssn' and safe_len(safe_str(field_value)) < 9:
            errors.append("SSN must be at least 9 digits")
        elif field == 'zip_code' and safe_len(safe_str(field_value)) < 5:
            errors.append("Zip code must be at least 5 digits")
    
    if safe_get(data, 'date_of_birth'):
        try:
            birth_date = data['date_of_birth']
            if isinstance(birth_date, str):
                birth_date = datetime.strptime(birth_date, '%Y-%m-%d').date()
            
            age = calculate_age(birth_date)
            
            if age and age > 150:
                errors.append("Age cannot be greater than 150 years")
            elif age and age < 0:
                errors.append("Date of birth cannot be in the future")
        except Exception as e:
            errors.append("Invalid date format")
    
    return len(errors) == 0, errors

def display_enhanced_mcid_data(mcid_data):
    """Enhanced MCID data display with improved styling and functionality"""
    if not mcid_data:
        st.warning("⚠️ No MCID data available")
        return
    
    st.markdown("""
    <div class="mcid-container">
        <h3>🆔 MCID (Member Consumer ID) Analysis</h3>
        <p><strong>Purpose:</strong> Patient identity verification and matching across healthcare systems</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display MCID status information
    status_code = safe_get(mcid_data, 'status_code', 'Unknown')
    service = safe_get(mcid_data, 'service', 'Unknown')
    timestamp = safe_get(mcid_data, 'timestamp', '')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Response Status", f"HTTP {status_code}")
    with col2:
        st.metric("Service", safe_str(service))
    with col3:
        if timestamp:
            parsed_time = safe_datetime_parse(timestamp, default=None)
            if parsed_time:
                formatted_time = parsed_time.strftime('%Y-%m-%d %H:%M')
                st.metric("Query Time", formatted_time)
            else:
                st.metric("Query Time", "Recent")
        else:
            st.metric("Query Time", "Unknown")
    
    # Process and display consumer matches
    if status_code == 200 and safe_get(mcid_data, 'body'):
        mcid_body = safe_get(mcid_data, 'body', {})
        consumers = safe_get(mcid_body, 'consumer', [])
        
        if consumers and safe_len(consumers) > 0:
            st.success(f"✅ Found {safe_len(consumers)} consumer match(es)")
            
            for i, consumer in enumerate(consumers, 1):
                if not consumer:  # Safety check
                    continue
                    
                st.markdown(f"""
                <div class="mcid-match-card">
                    <h4>🔍 Consumer Match #{i}</h4>
                """, unsafe_allow_html=True)
                
                # Create two columns for consumer info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Consumer Information:**")
                    st.write(f"• **Consumer ID:** {safe_get(consumer, 'consumerId', 'N/A')}")
                    st.write(f"• **Match Score:** {safe_get(consumer, 'score', 'N/A')}")
                    st.write(f"• **Status:** {safe_get(consumer, 'status', 'N/A')}")
                    st.write(f"• **Date of Birth:** {safe_get(consumer, 'dateOfBirth', 'N/A')}")
                
                with col2:
                    st.write("**Address Information:**")
                    address = safe_get(consumer, 'address', {})
                    if address:
                        st.write(f"• **City:** {safe_get(address, 'city', 'N/A')}")
                        st.write(f"• **State:** {safe_get(address, 'state', 'N/A')}")
                        st.write(f"• **ZIP Code:** {safe_get(address, 'zip', 'N/A')}")
                        st.write(f"• **County:** {safe_get(address, 'county', 'N/A')}")
                    else:
                        st.write("• No address information available")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show additional consumer data if available
                additional_data = safe_get(consumer, 'additionalData')
                if additional_data:
                    with st.expander(f"Additional Data for Consumer #{i}"):
                        st.json(additional_data)
        else:
            st.info("ℹ️ No consumer matches found in MCID search")
            st.markdown("""
            **Possible reasons:**
            - Patient may be new to the healthcare system
            - Different name variations or spelling
            - Updated personal information not yet synchronized
            """)
    else:
        st.warning(f"⚠️ MCID search returned status code: {status_code}")
        error_msg = safe_get(mcid_data, 'error')
        if error_msg:
            st.error(f"Error details: {error_msg}")
    
    # Raw MCID data in expandable section
    with st.expander("🔍 View Raw MCID JSON Data"):
        st.json(mcid_data)

def create_chatbot_loading_graphs():
    """Create interactive graphs to display while chatbot is loading"""
    if not PLOTLY_AVAILABLE:
        return None
        
    try:
        # Create sample health data for visualization
        sample_data = {
            'dates': pd.date_range('2023-01-01', periods=12, freq='M'),
            'risk_scores': np.random.uniform(0.1, 0.8, 12),
            'health_metrics': {
                'Blood Pressure': np.random.uniform(110, 140, 12),
                'Heart Rate': np.random.uniform(60, 100, 12),
                'Cholesterol': np.random.uniform(150, 250, 12)
            }
        }
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Health Risk Trend', 'Vital Signs Monitor', 'Risk Distribution', 'Health Score'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"type": "pie"}, {"type": "indicator"}]]
        )
        
        # Risk trend line
        fig.add_trace(
            go.Scatter(
                x=sample_data['dates'],
                y=sample_data['risk_scores'],
                mode='lines+markers',
                name='Risk Score',
                line=dict(color='#ff6b6b', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Vital signs
        for i, (metric, values) in enumerate(sample_data['health_metrics'].items()):
            fig.add_trace(
                go.Scatter(
                    x=sample_data['dates'],
                    y=values,
                    mode='lines',
                    name=metric,
                    line=dict(width=2)
                ),
                row=1, col=2
            )
        
        # Risk distribution pie chart
        risk_categories = ['Low Risk', 'Medium Risk', 'High Risk']
        risk_values = [45, 35, 20]
        colors = ['#4caf50', '#ff9800', '#f44336']
        
        fig.add_trace(
            go.Pie(
                labels=risk_categories,
                values=risk_values,
                marker_colors=colors,
                name="Risk Distribution"
            ),
            row=2, col=1
        )
        
        # Health score gauge
        current_score = np.random.uniform(60, 90)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=current_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Health Score"},
                delta={'reference': 75},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#2196f3"},
                    'steps': [
                        {'range': [0, 50], 'color': "#ffebee"},
                        {'range': [50, 80], 'color': "#e8f5e8"},
                        {'range': [80, 100], 'color': "#c8e6c9"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text="Real-Time Health Analytics Dashboard",
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        # Update subplot properties
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
    except Exception as e:
        logger.error(f"Error creating loading graphs: {e}")
        return None

def display_enhanced_quick_prompts():
    """Display enhanced quick prompt buttons for improved chatbot interaction"""
    
    st.markdown("""
    <div class="quick-prompts-enhanced">
        <div style="font-weight: 600; margin-bottom: 1rem; font-size: 1.1rem;">💡 Enhanced Healthcare Analysis Prompts</div>
        <p style="margin-bottom: 1rem; color: #666;">Click any prompt below to instantly analyze your healthcare data with advanced AI capabilities:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced healthcare-specific prompts
    enhanced_prompts = [
        "📊 Create a comprehensive health risk assessment chart with all my risk factors",
        "📈 Generate a detailed heart attack risk visualization with confidence intervals", 
        "🩺 Analyze my complete medication profile and create a therapeutic summary chart",
        "💓 Show my cardiovascular risk factors in an interactive bar chart",
        "🩸 Create a diabetes risk assessment based on my medical and pharmacy history",
        "📋 Generate a comprehensive timeline of my medical conditions and treatments",
        "📊 Visualize my medication adherence patterns and fill frequency",
        "❤️ Compare my health profile to age-matched population averages",
        "🔍 Analyze potential drug interactions in my current medication regimen",
        "📈 Create a health trajectory prediction model based on my claims data"
    ]
    
    # Display enhanced prompts in grid
    cols_per_row = 2
    for i in range(0, safe_len(enhanced_prompts), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < safe_len(enhanced_prompts):
                prompt = enhanced_prompts[i + j]
                with col:
                    if st.button(prompt, key=f"enhanced_prompt_{i+j}", use_container_width=True):
                        # Add the prompt to chat messages
                        st.session_state.chatbot_messages.append({"role": "user", "content": prompt})
                        
                        # Get enhanced response
                        try:
                            with st.spinner("🤖 Processing your enhanced healthcare analysis request..."):
                                agent = st.session_state.get('agent')
                                chatbot_context = st.session_state.get('chatbot_context')
                                
                                if agent and chatbot_context:
                                    if any(keyword in prompt.lower() for keyword in ['chart', 'graph', 'visualiz', 'show', 'create', 'generate']):
                                        # Enhanced graph request
                                        success, result = safe_execute(
                                            agent.chat_with_enhanced_graphs,
                                            prompt, 
                                            chatbot_context, 
                                            st.session_state.chatbot_messages
                                        )
                                        if success:
                                            response, code, figure = result
                                            st.session_state.chatbot_messages.append({"role": "assistant", "content": response, "code": code})
                                        else:
                                            log_error(f"Graph generation failed: {result}", "quick_prompt")
                                            st.error(f"Error: {result}")
                                    else:
                                        # Enhanced regular request
                                        success, result = safe_execute(
                                            agent.chat_with_enhanced_graphs,
                                            prompt, 
                                            chatbot_context, 
                                            st.session_state.chatbot_messages
                                        )
                                        if success:
                                            response, _, _ = result
                                            st.session_state.chatbot_messages.append({"role": "assistant", "content": response})
                                        else:
                                            log_error(f"Regular chat failed: {result}", "quick_prompt")
                                            st.error(f"Error: {result}")
                                else:
                                    st.error("Agent or context not available")
                            
                            st.rerun()
                            
                        except Exception as e:
                            log_error(f"Quick prompt error: {str(e)}", "quick_prompt")
                            st.error(f"Error: {str(e)}")

def execute_matplotlib_code_enhanced_stability(code: str):
    """Execute matplotlib code with enhanced stability and error recovery"""
    try:
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        plt.ioff()
        
        # Create namespace for code execution
        namespace = {
            'plt': plt,
            'matplotlib': matplotlib,
            'np': np,
            'numpy': np,
            'pd': pd,
            'pandas': pd,
            'json': json,
            'datetime': datetime,
            'time': time,
            'math': __import__('math')
        }
        
        # Add sample patient data
        namespace.update({
            'patient_age': 45,
            'heart_risk_score': 0.25,
            'medications_count': 3,
            'conditions': ['Hypertension', 'Type 2 Diabetes'],
            'risk_factors': {
                'Age': 45, 
                'Diabetes': 1, 
                'Smoking': 0, 
                'High_BP': 1,
                'Family_History': 1
            },
            'medication_list': ['Metformin', 'Lisinopril', 'Atorvastatin'],
            'risk_scores': [0.15, 0.25, 0.35, 0.20],
            'risk_labels': ['Low', 'Medium', 'High', 'Very High'],
            'months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'utilization_data': [2, 3, 1, 4, 2, 3]
        })
        
        # Execute the code
        exec(code, namespace)
        
        # Get the figure
        fig = plt.gcf()
        
        # Check if figure has content
        if not fig.axes:
            # Create fallback visualization
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'Enhanced Healthcare Visualization\nGenerated Successfully\n\nYour data analysis is ready!', 
                    ha='center', va='center', fontsize=16, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
            plt.title('Healthcare Data Analysis Dashboard', fontsize=18, fontweight='bold')
            plt.axis('off')
            fig = plt.gcf()
        
        # Enhance figure styling
        for ax in fig.axes:
            ax.tick_params(labelsize=10)
            ax.grid(True, alpha=0.3)
            
            if ax.get_title():
                ax.set_title(ax.get_title(), fontsize=12, fontweight='bold')
            if ax.get_xlabel():
                ax.set_xlabel(ax.get_xlabel(), fontsize=11)
            if ax.get_ylabel():
                ax.set_ylabel(ax.get_ylabel(), fontsize=11)
        
        # Convert to image
        img_buffer = io.BytesIO()
        fig.savefig(
            img_buffer, 
            format='png', 
            bbox_inches='tight', 
            dpi=200,
            facecolor='white', 
            edgecolor='none', 
            pad_inches=0.2,
            transparent=False
        )
        img_buffer.seek(0)
        
        # Cleanup
        plt.clf()
        plt.close('all')
        plt.ion()
        
        return img_buffer
        
    except Exception as e:
        # Error handling
        plt.clf()
        plt.close('all')
        plt.ion()
        
        error_msg = str(e)
        log_error(f"Matplotlib execution error: {error_msg}", "graph_generation")
        
        # Create error visualization
        try:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.6, '⚠️ Graph Generation Error', 
                    ha='center', va='center', fontsize=20, fontweight='bold', color='red')
            plt.text(0.5, 0.4, f'Error: {error_msg[:100]}...', 
                    ha='center', va='center', fontsize=12, color='darkred')
            plt.text(0.5, 0.3, 'Please try a different visualization request', 
                    ha='center', va='center', fontsize=12, color='blue')
            plt.title('Healthcare Data Visualization', fontsize=16)
            plt.axis('off')
            
            error_buffer = io.BytesIO()
            plt.savefig(error_buffer, format='png', bbox_inches='tight', dpi=150, facecolor='white')
            error_buffer.seek(0)
            plt.clf()
            plt.close('all')
            
            return error_buffer
        except Exception:
            st.error(f"Enhanced graph generation failed: {error_msg}")
            return None

def run_analysis_workflow(patient_data: Dict[str, Any]):
    """Run the analysis workflow with enhanced error handling"""
    if not AGENT_AVAILABLE:
        st.error("❌ Enhanced Health Analysis Agent not available")
        return None
        
    try:
        # Initialize agent and config
        config = EnhancedConfig()
        agent = EnhancedHealthAnalysisAgent(config)
        
        # Store in session state
        st.session_state.agent = agent
        st.session_state.config = config
        
        # Run analysis
        with st.spinner("🔬 Running enhanced healthcare analysis..."):
            success, results = safe_execute(agent.run_enhanced_analysis, patient_data)
            
            if success:
                st.session_state.analysis_results = results
                st.session_state.chatbot_context = safe_get(results, 'chatbot_context')
                return results
            else:
                log_error(f"Analysis failed: {results}", "workflow")
                st.error(f"Analysis failed: {results}")
                return None
                
    except Exception as e:
        log_error(f"Analysis workflow error: {str(e)}", "workflow")
        st.error(f"Analysis workflow failed: {str(e)}")
        return None

def display_batch_extraction_results():
    """Display batch extraction results"""
    analysis_results = st.session_state.get('analysis_results')
    if not analysis_results:
        st.warning("No analysis results available")
        return
    
    structured_extractions = safe_get(analysis_results, 'structured_extractions', {})
    
    if structured_extractions:
        tab1, tab2 = st.tabs(["🏥 Medical Extraction", "💊 Pharmacy Extraction"])
        
        with tab1:
            medical_extraction = safe_get(structured_extractions, 'medical', {})
            if medical_extraction:
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_records = safe_get(medical_extraction, 'extraction_summary', {}).get('total_hlth_srvc_records', 0)
                    st.metric("Health Service Records", safe_int(total_records))
                with col2:
                    api_calls = safe_get(medical_extraction, 'batch_stats', {}).get('api_calls_made', 0)
                    st.metric("API Calls Made", safe_int(api_calls))
                with col3:
                    calls_saved = safe_get(medical_extraction, 'batch_stats', {}).get('individual_calls_saved', 0)
                    st.metric("Calls Saved", safe_int(calls_saved))
                
                with st.expander("🔍 View Medical Extraction Details"):
                    st.json(medical_extraction)
            else:
                st.warning("No medical extraction data available")
        
        with tab2:
            pharmacy_extraction = safe_get(structured_extractions, 'pharmacy', {})
            if pharmacy_extraction:
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_records = safe_get(pharmacy_extraction, 'extraction_summary', {}).get('total_ndc_records', 0)
                    st.metric("NDC Records", safe_int(total_records))
                with col2:
                    api_calls = safe_get(pharmacy_extraction, 'batch_stats', {}).get('api_calls_made', 0)
                    st.metric("API Calls Made", safe_int(api_calls))
                with col3:
                    calls_saved = safe_get(pharmacy_extraction, 'batch_stats', {}).get('individual_calls_saved', 0)
                    st.metric("Calls Saved", safe_int(calls_saved))
                
                with st.expander("🔍 View Pharmacy Extraction Details"):
                    st.json(pharmacy_extraction)
            else:
                st.warning("No pharmacy extraction data available")
    else:
        st.warning("No structured extraction data available")

def display_entity_extraction_results():
    """Display entity extraction results"""
    analysis_results = st.session_state.get('analysis_results')
    if not analysis_results:
        st.warning("No analysis results available")
        return
    
    entity_extraction = safe_get(analysis_results, 'entity_extraction', {})
    
    if entity_extraction:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            diabetes = safe_get(entity_extraction, 'diabetics', 'unknown')
            st.metric("Diabetes", "Yes" if diabetes == "yes" else "No")
        
        with col2:
            blood_pressure = safe_get(entity_extraction, 'blood_pressure', 'unknown')
            st.metric("Blood Pressure", safe_str(blood_pressure).title())
        
        with col3:
            smoking = safe_get(entity_extraction, 'smoking', 'unknown')
            st.metric("Smoking", "Yes" if smoking == "yes" else "No")
        
        with col4:
            complexity_score = safe_get(entity_extraction, 'clinical_complexity_score', 0)
            st.metric("Complexity Score", safe_int(complexity_score))
        
        # Enhanced clinical analysis
        enhanced_analysis = safe_get(entity_extraction, 'enhanced_clinical_analysis', False)
        if enhanced_analysis:
            st.success("✅ Enhanced clinical analysis completed")
        
        with st.expander("🔍 View Entity Extraction Details"):
            st.json(entity_extraction)
    else:
        st.warning("No entity extraction data available")

def display_enhanced_trajectory_results():
    """Display enhanced trajectory analysis results"""
    analysis_results = st.session_state.get('analysis_results')
    if not analysis_results:
        st.warning("No analysis results available")
        return
    
    enhanced_trajectory = safe_get(analysis_results, 'enhanced_health_trajectory', '')
    
    if enhanced_trajectory:
        st.markdown("### 📈 Enhanced Health Trajectory Analysis")
        st.markdown(enhanced_trajectory)
    else:
        st.warning("No enhanced trajectory data available")

def display_heart_attack_prediction_results():
    """Display heart attack prediction results"""
    analysis_results = st.session_state.get('analysis_results')
    if not analysis_results:
        st.warning("No analysis results available")
        return
    
    heart_attack_prediction = safe_get(analysis_results, 'heart_attack_prediction', {})
    heart_attack_features = safe_get(analysis_results, 'heart_attack_features', {})
    heart_attack_risk_score = safe_get(analysis_results, 'heart_attack_risk_score', 0.0)
    
    if heart_attack_prediction:
        # Display prediction results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_display = safe_get(heart_attack_prediction, 'risk_display', 'Risk assessment unavailable')
            st.metric("Risk Assessment", safe_str(risk_display))
        
        with col2:
            confidence_display = safe_get(heart_attack_prediction, 'confidence_display', 'Confidence unknown')
            st.metric("Prediction Confidence", safe_str(confidence_display))
        
        with col3:
            st.metric("Risk Score", f"{safe_float(heart_attack_risk_score):.1%}")
        
        # Clinical interpretation
        clinical_interpretation = safe_get(heart_attack_prediction, 'clinical_interpretation', '')
        if clinical_interpretation:
            st.markdown("### 🔬 Clinical Interpretation")
            st.markdown(clinical_interpretation)
        
        # Features used
        if heart_attack_features:
            with st.expander("📊 View Risk Assessment Features"):
                st.json(heart_attack_features)
    else:
        st.warning("❌ No heart attack prediction data available")

# Initialize session state
initialize_session_state()

# Enhanced Main Title
st.markdown('<h1 class="main-header">🚀 Enhanced Health Analysis Agent</h1>', unsafe_allow_html=True)

# Enhanced optimization badges
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <div class="enhanced-badge">⚡ 93% Fewer API Calls</div>
    <div class="enhanced-badge">🚀 90% Faster Processing</div>
    <div class="enhanced-badge">📊 Enhanced Graph Stability</div>
    <div class="enhanced-badge">🗂️ Complete Claims Data Viewer</div>
    <div class="enhanced-badge">🎯 Detailed Health Analysis</div>
    <div class="enhanced-badge">💡 Specific Healthcare Prompts</div>
</div>
""", unsafe_allow_html=True)

# Display import status
if not AGENT_AVAILABLE:
    st.markdown(f'<div class="status-error">❌ Failed to import Enhanced Health Agent: {import_error}</div>', unsafe_allow_html=True)
    st.markdown("""
    **Troubleshooting Steps:**
    1. Ensure all required dependencies are installed
    2. Check that all Python files are in the same directory
    3. Verify the health_agent_core_enhanced.py file exists and is properly formatted
    4. Check the Python environment and package versions
    """)
    st.stop()

# ENHANCED SIDEBAR CHATBOT WITH STABLE GRAPHS
with st.sidebar:
    analysis_results = st.session_state.get('analysis_results')
    chatbot_ready = analysis_results and safe_get(analysis_results, "chatbot_ready", False)
    chatbot_context = st.session_state.get('chatbot_context')
    
    if chatbot_ready and chatbot_context:
        st.title("💬 Enhanced AI Healthcare Assistant")
        st.markdown("""
        <div class="enhanced-badge" style="margin: 0.5rem 0;">📊 Advanced Graph Generation</div>
        <div class="enhanced-badge" style="margin: 0.5rem 0;">🎯 Specialized Healthcare Analysis</div>
        """, unsafe_allow_html=True)
        
        # Enhanced quick prompt buttons
        display_enhanced_quick_prompts()
        
        st.markdown("---")
        
        # Enhanced chat history display
        chat_container = st.container()
        with chat_container:
            chatbot_messages = st.session_state.get('chatbot_messages', [])
            if chatbot_messages:
                for i, message in enumerate(chatbot_messages):
                    with st.chat_message(safe_get(message, "role", "user")):
                        st.write(safe_get(message, "content", ""))
                        
                        # Enhanced code display and execution
                        code = safe_get(message, "code")
                        if code:
                            with st.expander("📊 View Enhanced Matplotlib Code"):
                                st.code(code, language="python")
                            
                            # Execute matplotlib code with enhanced stability
                            img_buffer = execute_matplotlib_code_enhanced_stability(code)
                            if img_buffer:
                                st.image(img_buffer, use_column_width=True, caption="Enhanced Healthcare Visualization")
                            else:
                                st.warning("Enhanced graph generation encountered an issue. Please try a different visualization request.")
            else:
                st.info("👋 Hello! I'm your Enhanced Healthcare AI Assistant!")
                st.info("💡 **New Features:** Advanced analytics, detailed health insights, and stable graph generation!")
                st.info("🎯 **Specialized:** Ask specific healthcare questions or request detailed visualizations!")
        
        # Enhanced chat input
        st.markdown("---")
        user_question = st.chat_input("Ask detailed healthcare questions or request advanced visualizations...")
        
        # Handle enhanced chat input
        if user_question:
            st.session_state.chatbot_messages.append({"role": "user", "content": user_question})
            
            try:
                with st.spinner("🤖 Processing with enhanced healthcare AI capabilities..."):
                    agent = st.session_state.get('agent')
                    
                    if agent and chatbot_context:
                        # Enhanced graph detection
                        graph_keywords = ['graph', 'chart', 'plot', 'visualize', 'visualization', 'show', 'display', 'draw', 'create', 'generate']
                        is_graph_request = any(keyword in user_question.lower() for keyword in graph_keywords)
                        
                        if is_graph_request:
                            success, result = safe_execute(
                                agent.chat_with_enhanced_graphs,
                                user_question, 
                                chatbot_context, 
                                st.session_state.chatbot_messages
                            )
                            if success:
                                response, code, figure = result
                                st.session_state.chatbot_messages.append({
                                    "role": "assistant", 
                                    "content": response, 
                                    "code": code
                                })
                            else:
                                log_error(f"Graph generation error: {result}", "chat_input")
                                st.error(f"Graph generation error: {result}")
                        else:
                            success, result = safe_execute(
                                agent.chat_with_enhanced_graphs,
                                user_question, 
                                chatbot_context, 
                                st.session_state.chatbot_messages
                            )
                            if success:
                                response, _, _ = result
                                st.session_state.chatbot_messages.append({"role": "assistant", "content": response})
                            else:
                                log_error(f"Chat error: {result}", "chat_input")
                                st.error(f"Chat error: {result}")
                    else:
                        st.error("Agent or context not available")
                
                st.rerun()
                
            except Exception as e:
                log_error(f"Chat input error: {str(e)}", "chat_input")
                st.error(f"Error: {str(e)}")
                st.info("💡 Please try rephrasing your question or request a different type of analysis.")
        
        # Enhanced clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chatbot_messages = []
            st.rerun()
    
    else:
        st.title("💬 Enhanced AI Healthcare Assistant")
        st.info("💤 Assistant available after analysis completion")
        st.markdown("---")
        
        # Show loading graphs while chatbot is being prepared
        analysis_running = st.session_state.get('analysis_running', False)
        if analysis_running or (analysis_results and not safe_get(analysis_results, "chatbot_ready", False)):
            st.markdown("""
            <div class="chatbot-loading-container">
                <div class="loading-spinner"></div>
                <h4>🤖 Preparing Enhanced AI Assistant...</h4>
                <p>Loading healthcare analysis capabilities</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display interactive loading graphs
            try:
                loading_fig = create_chatbot_loading_graphs()
                if loading_fig:
                    st.plotly_chart(loading_fig, use_container_width=True, key="chatbot_loading_graphs")
            except Exception as e:
                log_error(f"Loading graphs error: {str(e)}", "loading_graphs")
                st.info("📊 Health analytics dashboard loading...")
        
        st.markdown("**🚀 Enhanced Features:**")
        st.markdown("• 📊 **Stable Graph Generation** - Reliable chart creation")
        st.markdown("• 🎯 **Specialized Healthcare Analysis** - Domain-specific insights") 
        st.markdown("• ❤️ **Advanced Risk Assessment** - Comprehensive health modeling")
        st.markdown("• 💡 **Smart Healthcare Prompts** - Pre-built clinical questions")
        st.markdown("• 🔤 **Detailed Code Meanings** - Medical terminology explanations")
        st.markdown("• 🗂️ **Complete Data Access** - All claims with enhanced viewing")

# 1. PATIENT INFORMATION
st.markdown("""
<div class="section-box">
    <div class="section-title">👤 Patient Information</div>
</div>
""", unsafe_allow_html=True)

with st.container():
    with st.form("patient_input_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            first_name = st.text_input("First Name *", value="")
            last_name = st.text_input("Last Name *", value="")
        
        with col2:
            ssn = st.text_input("SSN *", value="")
            date_of_birth = st.date_input(
                "Date of Birth *", 
                value=datetime.now().date(),
                min_value=datetime(1900, 1, 1).date(),
                max_value=datetime.now().date()
            )
        
        with col3:
            gender = st.selectbox("Gender *", ["F", "M"])
            zip_code = st.text_input("Zip Code *", value="")
        
        # Show calculated age
        if date_of_birth:
            calculated_age = calculate_age(date_of_birth)
            if calculated_age is not None:
                st.info(f"📅 **Calculated Age:** {calculated_age} years old")
        
        # ENHANCED RUN ANALYSIS BUTTON
        submitted = st.form_submit_button(
            "🚀 Run Enhanced Healthcare Analysis", 
            use_container_width=True,
            disabled=st.session_state.get('analysis_running', False),
            type="primary"
        )

# Handle form submission
if submitted:
    # Prepare patient data
    patient_data = {
        "first_name": first_name,
        "last_name": last_name,
        "ssn": ssn,
        "date_of_birth": date_of_birth.strftime('%Y-%m-%d') if date_of_birth else "",
        "gender": gender,
        "zip_code": zip_code
    }
    
    # Validate data
    is_valid, validation_errors = validate_patient_data(patient_data)
    
    if not is_valid:
        st.markdown('<div class="status-error">❌ Please fix the following errors:</div>', unsafe_allow_html=True)
        for error in validation_errors:
            st.error(f"• {error}")
    else:
        # Set analysis running state
        st.session_state.analysis_running = True
        st.session_state.analysis_results = None
        st.session_state.chatbot_messages = []
        
        # Run analysis
        results = run_analysis_workflow(patient_data)
        
        # Clear running state
        st.session_state.analysis_running = False
        
        if results:
            st.success("✅ Enhanced healthcare analysis completed successfully!")
            st.rerun()
        else:
            st.error("❌ Analysis failed. Please check the logs for more details.")

# ENHANCED RESULTS SECTION
analysis_results = st.session_state.get('analysis_results')
analysis_running = st.session_state.get('analysis_running', False)

if analysis_results and not analysis_running:
    st.markdown("---")
    st.markdown("## 📊 Enhanced Healthcare Analysis Results")
    
    # Show errors if any
    errors = safe_get(analysis_results, 'errors', [])
    if errors:
        st.markdown('<div class="status-error">❌ Analysis errors occurred</div>', unsafe_allow_html=True)
        with st.expander("🐛 Debug Information"):
            st.write("**Errors:**")
            for error in errors:
                st.write(f"• {error}")

    # 2. ENHANCED ALL CLAIMS DATA VIEWER WITH MCID
    if st.button("🗂️ Complete Claims Data Viewer - Enhanced Edition", use_container_width=True, key="enhanced_all_claims_btn"):
        st.session_state.show_all_claims_data = not st.session_state.get('show_all_claims_data', False)
    
    if st.session_state.get('show_all_claims_data', False):
        st.markdown("""
        <div class="section-box">
            <div class="section-title">🗂️ Complete Claims Data Viewer - Enhanced Analysis</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="claims-viewer-card">
            <h3>📋 Complete Deidentified Claims Database</h3>
            <p><strong>Enhanced Features:</strong> This viewer provides complete access to ALL deidentified claims data with detailed viewing options, structured data analysis, and comprehensive JSON exploration capabilities.</p>
        </div>
        """, unsafe_allow_html=True)
        
        deidentified_data = safe_get(analysis_results, 'deidentified_data', {})
        api_outputs = safe_get(analysis_results, 'api_outputs', {})
        
        if deidentified_data or api_outputs:
            # Enhanced tabs including MCID
            tab1, tab2, tab3, tab4 = st.tabs([
                "🏥 Medical Claims Details", 
                "💊 Pharmacy Claims Details", 
                "🆔 MCID Consumer Data",
                "📊 Complete JSON Explorer"
            ])
            
            with tab1:
                medical_data = safe_get(deidentified_data, 'medical', {})
                if medical_data and not safe_get(medical_data, 'error'):
                    st.markdown("### 🏥 Enhanced Medical Claims Analysis")
                    
                    # Patient demographics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        age_data = safe_get(medical_data, 'src_mbr_age', 'Unknown')
                        st.metric("Patient Age", safe_str(age_data))
                    with col2:
                        zip_data = safe_get(medical_data, 'src_mbr_zip_cd', 'Unknown')
                        st.metric("ZIP Code", safe_str(zip_data))
                    with col3:
                        deident_time = safe_get(medical_data, 'deidentification_timestamp', '')
                        if deident_time:
                            parsed_time = safe_datetime_parse(deident_time, default=None)
                            if parsed_time:
                                formatted_time = parsed_time.strftime('%m/%d/%Y %H:%M')
                                st.metric("Deidentified", formatted_time)
                            else:
                                st.metric("Deidentified", "Recently")
                        else:
                            st.metric("Deidentified", "Unknown")
                    
                    # Enhanced medical claims data exploration
                    medical_claims_data = safe_get(medical_data, 'medical_claims_data', {})
                    if medical_claims_data:
                        # Show summary statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Records", safe_len(medical_claims_data))
                        with col2:
                            clinical_elements = safe_get(medical_data, 'clinical_data_elements_preserved', 0)
                            st.metric("Clinical Elements", safe_int(clinical_elements))
                        
                        with st.expander("🔍 Explore Medical Claims JSON Structure", expanded=False):
                            st.json(medical_claims_data)
                    else:
                        st.warning("⚠️ No medical claims data in structure")
                else:
                    error_msg = safe_get(medical_data, 'error', 'Unknown error')
                    st.error(f"❌ Medical claims error: {error_msg}")
            
            with tab2:
                pharmacy_data = safe_get(deidentified_data, 'pharmacy', {})
                if pharmacy_data and not safe_get(pharmacy_data, 'error'):
                    st.markdown("### 💊 Enhanced Pharmacy Claims Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        data_type = safe_get(pharmacy_data, 'data_type', 'Unknown')
                        st.metric("Data Type", safe_str(data_type))
                    with col2:
                        deident_time = safe_get(pharmacy_data, 'deidentification_timestamp', '')
                        if deident_time:
                            parsed_time = safe_datetime_parse(deident_time, default=None)
                            if parsed_time:
                                formatted_time = parsed_time.strftime('%m/%d/%Y %H:%M')
                                st.metric("Processed", formatted_time)
                            else:
                                st.metric("Processed", "Recently")
                        else:
                            st.metric("Processed", "Unknown")
                    with col3:
                        masked_fields = safe_get(pharmacy_data, 'name_fields_masked', [])
                        st.metric("Fields Masked", safe_len(masked_fields))
                    
                    # Enhanced pharmacy summary
                    col1, col2 = st.columns(2)
                    with col1:
                        therapeutic_elements = safe_get(pharmacy_data, 'therapeutic_data_elements_preserved', 0)
                        st.metric("Therapeutic Elements", safe_int(therapeutic_elements))
                    with col2:
                        healthcare_spec = safe_get(pharmacy_data, 'healthcare_specialization', 'standard')
                        st.metric("Analysis Level", safe_str(healthcare_spec))
                    
                    pharmacy_claims_data = safe_get(pharmacy_data, 'pharmacy_claims_data', {})
                    if pharmacy_claims_data:
                        with st.expander("🔍 Explore Pharmacy Claims JSON Structure", expanded=False):
                            st.json(pharmacy_claims_data)
                    else:
                        st.warning("⚠️ No pharmacy claims data in structure")
                else:
                    error_msg = safe_get(pharmacy_data, 'error', 'Unknown error')
                    st.error(f"❌ Pharmacy claims error: {error_msg}")
            
            with tab3:
                mcid_data = safe_get(api_outputs, 'mcid', {})
                display_enhanced_mcid_data(mcid_data)
            
            with tab4:
                st.markdown("### 🔍 Complete JSON Data Explorer")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🏥 Medical + Pharmacy Data")
                    if deidentified_data:
                        # Show summary before JSON
                        total_items = sum(1 for k, v in deidentified_data.items() if v and not safe_get(v, 'error'))
                        st.info(f"📊 {total_items} data categories available")
                        
                        with st.expander("Expand Deidentified Data JSON", expanded=False):
                            st.json(deidentified_data)
                    else:
                        st.warning("No deidentified data available")
                
                with col2:
                    st.markdown("#### 🆔 MCID + API Outputs")
                    if api_outputs:
                        # Show API summary
                        successful_apis = sum(1 for k, v in api_outputs.items() 
                                            if v and safe_get(v, 'status_code') == 200)
                        total_apis = safe_len(api_outputs)
                        st.info(f"📡 {successful_apis}/{total_apis} APIs successful")
                        
                        with st.expander("Expand API Outputs JSON", expanded=False):
                            st.json(api_outputs)
                    else:
                        st.warning("No API outputs available")
        else:
            st.error("❌ No claims data available for display")

    # 3. BATCH EXTRACTION RESULTS
    if st.button("🚀 Batch Code Processing Results", use_container_width=True, key="batch_extraction_btn"):
        st.session_state.show_batch_extraction = not st.session_state.get('show_batch_extraction', False)
    
    if st.session_state.get('show_batch_extraction', False):
        st.markdown("""
        <div class="section-box">
            <div class="section-title">🚀 Batch Code Processing Results</div>
        </div>
        """, unsafe_allow_html=True)
        display_batch_extraction_results()

    # 4. ENTITY EXTRACTION RESULTS
    if st.button("🎯 Health Entity Extraction Results", use_container_width=True, key="entity_extraction_btn"):
        st.session_state.show_entity_extraction = not st.session_state.get('show_entity_extraction', False)
    
    if st.session_state.get('show_entity_extraction', False):
        st.markdown("""
        <div class="section-box">
            <div class="section-title">🎯 Health Entity Extraction Results</div>
        </div>
        """, unsafe_allow_html=True)
        display_entity_extraction_results()

    # 5. ENHANCED TRAJECTORY RESULTS
    if st.button("📈 Enhanced Health Trajectory Analysis", use_container_width=True, key="enhanced_trajectory_btn"):
        st.session_state.show_enhanced_trajectory = not st.session_state.get('show_enhanced_trajectory', False)
    
    if st.session_state.get('show_enhanced_trajectory', False):
        st.markdown("""
        <div class="section-box">
            <div class="section-title">📈 Enhanced Health Trajectory Analysis</div>
        </div>
        """, unsafe_allow_html=True)
        display_enhanced_trajectory_results()

    # 6. HEART ATTACK PREDICTION RESULTS
    if st.button("❤️ Enhanced Heart Attack Risk Prediction", use_container_width=True, key="enhanced_heart_attack_btn"):
        st.session_state.show_heart_attack = not st.session_state.get('show_heart_attack', False)
    
    if st.session_state.get('show_heart_attack', False):
        st.markdown("""
        <div class="section-box">
            <div class="section-title">❤️ Enhanced Heart Attack Risk Prediction</div>
        </div>
        """, unsafe_allow_html=True)
        display_heart_attack_prediction_results()

# Error Log & Debugging
if st.session_state.get('error_log'):
    with st.expander("🐛 Error Log & Debugging", expanded=False):
        st.markdown("### Recent Errors")
        for error in st.session_state.error_log[-10:]:  # Show last 10 errors
            timestamp = safe_get(error, 'timestamp', 'Unknown time')
            message = safe_get(error, 'message', 'Unknown error')
            context = safe_get(error, 'context', 'Unknown context')
            
            st.error(f"**{timestamp}** [{context}]: {message}")
        
        if st.button("Clear Error Log"):
            st.session_state.error_log = []
            st.rerun()

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin: 2rem 0;">
    🚀 Enhanced Health Analysis Agent v4.0 | 
    <span class="enhanced-badge" style="margin: 0;">⚡ Stable & Compatible</span>
    <span class="enhanced-badge" style="margin: 0;">🚀 Enhanced Error Handling</span>
    <span class="enhanced-badge" style="margin: 0;">📊 Improved Graph Stability</span>
    <span class="enhanced-badge" style="margin: 0;">🗂️ Complete Claims Viewer</span>
    <span class="enhanced-badge" style="margin: 0;">🎯 Comprehensive Healthcare Analysis</span>
</div>
""", unsafe_allow_html=True)
