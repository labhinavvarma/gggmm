# Configure Streamlit page FIRST
import streamlit as st

# Determine sidebar state - COLLAPSED by default with larger width
sidebar_state = "collapsed"

st.set_page_config(
    page_title="⚡ Real-time LangGraph Health Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state=sidebar_state
)

# Now import other modules
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import os
import logging
from typing import Dict, Any, Optional, Callable
import matplotlib.pyplot as plt
import matplotlib
import io
import base64
import re
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import threading
from queue import Queue, Empty
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio

# ENHANCED MATPLOTLIB CONFIGURATION FOR STREAMLIT
matplotlib.use('Agg')  # Use non-interactive backend
plt.ioff()  # Turn off interactive mode

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*FigureCanvasAgg is non-interactive.*')

# Set default style
plt.style.use('default')

# Configure matplotlib parameters for better Streamlit integration
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'figure.figsize': (10, 6),
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'text.usetex': False,
})

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Set up logging
logger = logging.getLogger(__name__)

# Import the health analysis agent
AGENT_AVAILABLE = False
import_error = None
HealthAnalysisAgent = None
Config = None

try:
    from health_agent_core import HealthAnalysisAgent, Config
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    import_error = str(e)

# Enhanced CSS with real-time status indicators
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

.realtime-workflow-container {
    background: linear-gradient(135deg, #e8f0fe 0%, #f3e5f5 25%, #e1f5fe 50%, #f1f8e9 75%, #fff8e1 100%);
    padding: 3rem;
    border-radius: 25px;
    margin: 2rem 0;
    border: 2px solid rgba(52, 152, 219, 0.3);
    box-shadow: 0 20px 50px rgba(52, 152, 219, 0.2);
    position: relative;
    overflow: hidden;
}

.realtime-workflow-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
    animation: rotate-glow 20s linear infinite;
    pointer-events: none;
}

@keyframes rotate-glow {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.workflow-step {
    background: rgba(255, 255, 255, 0.8);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    border-left: 4px solid #6c757d;
    transition: all 0.4s ease;
    backdrop-filter: blur(10px);
    position: relative;
    z-index: 2;
}

.workflow-step.pending {
    border-left-color: #6c757d;
    background: rgba(108, 117, 125, 0.1);
}

.workflow-step.running {
    border-left-color: #ffc107;
    background: rgba(255, 193, 7, 0.15);
    animation: pulse-step 1.5s infinite;
    box-shadow: 0 10px 30px rgba(255, 193, 7, 0.3);
}

.workflow-step.completed {
    border-left-color: #28a745;
    background: rgba(40, 167, 69, 0.15);
    box-shadow: 0 10px 30px rgba(40, 167, 69, 0.2);
}

.workflow-step.error {
    border-left-color: #dc3545;
    background: rgba(220, 53, 69, 0.15);
    box-shadow: 0 10px 30px rgba(220, 53, 69, 0.2);
}

@keyframes pulse-step {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.02); }
}

.realtime-sync-indicator {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    border-left: 4px solid #28a745;
    animation: live-pulse 1.5s infinite;
    position: relative;
    z-index: 2;
}

@keyframes live-pulse {
    0%, 100% { opacity: 0.9; }
    50% { opacity: 1; }
}

.node-status-details {
    font-size: 0.8rem;
    color: #666;
    margin-top: 0.5rem;
}

.progress-metrics {
    display: flex;
    justify-content: space-between;
    margin: 1rem 0;
    position: relative;
    z-index: 2;
}

.metric-card {
    background: rgba(255, 255, 255, 0.9);
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    flex: 1;
    margin: 0 0.5rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.status-error {
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    color: #d32f2f;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    border-left: 4px solid #f44336;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# REAL-TIME PROGRESS TRACKER with Enhanced Callbacks
@dataclass 
class NodeProgress:
    node_name: str
    ui_name: str
    status: str  # pending, running, completed, error
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class RealTimeLangGraphTracker:
    """Enhanced real-time tracker for LangGraph node execution"""
    
    def __init__(self):
        self.progress_queue = Queue()
        self.node_status = {}
        self.current_node = None
        self.lock = threading.Lock()
        
        # Enhanced node mapping with descriptions
        self.node_mapping = {
            'fetch_api_data': {
                'ui_name': 'API Data Fetch',
                'icon': '⚡',
                'description': 'Fetching comprehensive claims data from APIs'
            },
            'deidentify_claims_data': {
                'ui_name': 'Data Deidentification', 
                'icon': '🔒',
                'description': 'Advanced PII removal with clinical preservation'
            },
            'extract_claims_fields': {
                'ui_name': 'Field Extraction',
                'icon': '🚀', 
                'description': 'Extracting medical and pharmacy fields'
            },
            'extract_entities': {
                'ui_name': 'Entity Extraction',
                'icon': '🎯',
                'description': 'Advanced health entity identification'
            },
            'analyze_trajectory': {
                'ui_name': 'Health Trajectory',
                'icon': '📈',
                'description': 'Comprehensive predictive health analysis'
            },
            'generate_summary': {
                'ui_name': 'Summary Generation',
                'icon': '📋',
                'description': 'Executive summary creation'
            },
            'predict_heart_attack': {
                'ui_name': 'Heart Risk Prediction',
                'icon': '❤️',
                'description': 'ML-based cardiovascular assessment'
            },
            'initialize_chatbot': {
                'ui_name': 'Chatbot Initialization',
                'icon': '💬',
                'description': 'AI assistant with graph generation'
            }
        }
    
    def create_node_callback(self):
        """Create callback function to inject into LangGraph nodes"""
        def node_callback(node_name: str, status: str, details: Dict[str, Any] = None, error_msg: str = None):
            """Real-time callback from LangGraph node execution"""
            with self.lock:
                timestamp = datetime.now()
                
                node_info = self.node_mapping.get(node_name, {
                    'ui_name': node_name,
                    'icon': '⚙️',
                    'description': f'Processing {node_name}'
                })
                
                progress = NodeProgress(
                    node_name=node_name,
                    ui_name=node_info['ui_name'],
                    status=status,
                    start_time=timestamp if status == 'running' else self.node_status.get(node_name, {}).get('start_time'),
                    end_time=timestamp if status in ['completed', 'error'] else None,
                    error_message=error_msg,
                    details=details or {}
                )
                
                self.node_status[node_name] = progress
                self.current_node = node_name if status == 'running' else None
                
                # Put in queue for UI updates
                self.progress_queue.put({
                    'node_name': node_name,
                    'ui_name': node_info['ui_name'],
                    'icon': node_info['icon'],
                    'description': node_info['description'],
                    'status': status,
                    'timestamp': timestamp,
                    'error_message': error_msg,
                    'details': details or {},
                    'duration': self._calculate_duration(progress)
                })
                
                logger.info(f"🔄 REAL-TIME UPDATE: {node_info['ui_name']} -> {status.upper()}")
        
        return node_callback
    
    def _calculate_duration(self, progress: NodeProgress) -> Optional[float]:
        """Calculate node execution duration"""
        if progress.start_time and progress.end_time:
            return (progress.end_time - progress.start_time).total_seconds()
        return None
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current overall status"""
        with self.lock:
            total_nodes = len(self.node_mapping)
            completed = len([n for n in self.node_status.values() if n.status == 'completed'])
            running = len([n for n in self.node_status.values() if n.status == 'running'])
            errors = len([n for n in self.node_status.values() if n.status == 'error'])
            
            return {
                'total_nodes': total_nodes,
                'completed': completed,
                'running': running,
                'errors': errors,
                'progress_percent': (completed / total_nodes) * 100,
                'current_node': self.current_node,
                'node_status': dict(self.node_status)
            }

# ENHANCED AGENT WRAPPER with Real Callback Integration
class EnhancedLangGraphWrapper:
    """Enhanced wrapper that injects real-time callbacks into LangGraph nodes"""
    
    def __init__(self, agent):
        self.agent = agent
        self.tracker = RealTimeLangGraphTracker()
        self.original_methods = {}
        
    def inject_realtime_callbacks(self):
        """Inject real-time callbacks into all LangGraph node methods"""
        callback = self.tracker.create_node_callback()
        
        # Get all node methods to patch
        node_methods = [
            'fetch_api_data', 'deidentify_claims_data', 'extract_claims_fields',
            'extract_entities', 'analyze_trajectory', 'generate_summary', 
            'predict_heart_attack', 'initialize_chatbot'
        ]
        
        for method_name in node_methods:
            if hasattr(self.agent, method_name):
                original_method = getattr(self.agent, method_name)
                self.original_methods[method_name] = original_method
                
                def create_wrapped_method(method_name, original_method):
                    def wrapped_method(state):
                        # Notify start
                        callback(method_name, 'running')
                        
                        try:
                            # Execute original method
                            result = original_method(state)
                            
                            # Check for errors in state
                            if state.get('errors'):
                                callback(method_name, 'error', 
                                        error_msg=str(state['errors'][-1]) if state['errors'] else 'Unknown error')
                            else:
                                callback(method_name, 'completed', 
                                        details={'step_status': state.get('step_status', {})})
                            
                            return result
                            
                        except Exception as e:
                            callback(method_name, 'error', error_msg=str(e))
                            raise e
                    
                    return wrapped_method
                
                # Replace method with wrapped version
                wrapped = create_wrapped_method(method_name, original_method)
                setattr(self.agent, method_name, wrapped)
                
        logger.info("✅ Real-time callbacks injected into all LangGraph nodes")
    
    def restore_original_methods(self):
        """Restore original methods after analysis"""
        for method_name, original_method in self.original_methods.items():
            setattr(self.agent, method_name, original_method)
        logger.info("✅ Original LangGraph methods restored")
    
    def run_with_realtime_updates(self, patient_data: Dict[str, Any], 
                                  workflow_placeholder, progress_placeholder):
        """Run analysis with real-time UI updates"""
        
        try:
            # Inject callbacks
            self.inject_realtime_callbacks()
            
            # Start UI update thread
            stop_ui_updates = threading.Event()
            ui_thread = threading.Thread(
                target=self._ui_update_worker,
                args=(workflow_placeholder, progress_placeholder, stop_ui_updates),
                daemon=True
            )
            ui_thread.start()
            
            # Run the actual analysis
            logger.info("🚀 Starting LangGraph analysis with real-time callbacks...")
            results = self.agent.run_analysis(patient_data)
            
            # Stop UI updates
            stop_ui_updates.set()
            ui_thread.join(timeout=2)
            
            # Final UI update
            final_status = self.tracker.get_current_status()
            self._update_workflow_display(workflow_placeholder, final_status)
            
            with progress_placeholder.container():
                if results.get('success'):
                    st.success("🎉 LangGraph analysis completed successfully!")
                else:
                    st.error("❌ LangGraph analysis encountered errors")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in real-time analysis: {str(e)}")
            raise e
        finally:
            # Always restore original methods
            self.restore_original_methods()
    
    def _ui_update_worker(self, workflow_placeholder, progress_placeholder, stop_event):
        """Background worker for UI updates"""
        while not stop_event.is_set():
            try:
                # Check for progress updates
                update = self.tracker.progress_queue.get(timeout=0.5)
                
                # Update workflow display
                current_status = self.tracker.get_current_status()
                self._update_workflow_display(workflow_placeholder, current_status)
                
                # Update progress messages
                with progress_placeholder.container():
                    if update['status'] == 'running':
                        st.info(f"🔄 **LIVE**: Executing {update['ui_name']}...")
                    elif update['status'] == 'completed':
                        duration = update.get('duration')
                        duration_text = f" ({duration:.1f}s)" if duration else ""
                        st.success(f"✅ **Completed**: {update['ui_name']}{duration_text}")
                    elif update['status'] == 'error':
                        error_msg = update.get('error_message', 'Unknown error')
                        st.error(f"❌ **Failed**: {update['ui_name']} - {error_msg}")
                
            except Empty:
                continue
            except Exception as e:
                logger.warning(f"UI update error: {e}")
                continue
    
    def _update_workflow_display(self, workflow_placeholder, status: Dict[str, Any]):
        """Update the workflow visualization"""
        with workflow_placeholder.container():
            self._display_realtime_workflow(status)

def _display_realtime_workflow(status: Dict[str, Any]):
    """Display real-time workflow with actual LangGraph status"""
    
    # Main container
    st.markdown('<div class="realtime-workflow-container">', unsafe_allow_html=True)
    
    # Header with live status
    current_node_name = status.get('current_node', 'None')
    current_ui_name = 'Idle'
    if current_node_name and current_node_name != 'None':
        tracker = RealTimeLangGraphTracker()
        current_ui_name = tracker.node_mapping.get(current_node_name, {}).get('ui_name', current_node_name)
    
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem; position: relative; z-index: 2;">
        <h2 style="color: #2c3e50; font-weight: 700;">🧠 LIVE LangGraph Healthcare Analysis</h2>
        <p style="color: #34495e; font-size: 1.1rem;">Real-time synchronized workflow execution</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Live sync indicator
    if status['running'] > 0:
        st.markdown(f"""
        <div class="realtime-sync-indicator">
            🟢 <strong>LIVE EXECUTION</strong> | 
            <strong>Current Node:</strong> {current_ui_name} | 
            <strong>Progress:</strong> {status['progress_percent']:.0f}%
        </div>
        """, unsafe_allow_html=True)
    
    # Progress metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Nodes", status['total_nodes'])
    with col2:
        st.metric("Completed", status['completed'])
    with col3:
        st.metric("Running", status['running'])
    with col4:
        st.metric("Progress", f"{status['progress_percent']:.0f}%")
    
    # Progress bar
    st.progress(status['progress_percent'] / 100)
    
    # Display workflow steps with real status
    tracker = RealTimeLangGraphTracker()
    for node_name, node_info in tracker.node_mapping.items():
        node_status = status['node_status'].get(node_name)
        
        if node_status:
            step_status = node_status.status
            start_time = node_status.start_time
            end_time = node_status.end_time
            duration = tracker._calculate_duration(node_status) if hasattr(tracker, '_calculate_duration') else None
        else:
            step_status = 'pending'
            start_time = None
            end_time = None
            duration = None
        
        # Status display
        if step_status == 'completed':
            status_emoji = "✅"
            status_text = "Completed"
            if duration:
                status_text += f" ({duration:.1f}s)"
        elif step_status == 'running':
            status_emoji = "🔄"
            status_text = "Executing..."
        elif step_status == 'error':
            status_emoji = "❌"
            status_text = "Failed"
            if node_status and node_status.error_message:
                status_text += f": {node_status.error_message[:50]}"
        else:
            status_emoji = "⏳"
            status_text = "Pending"
        
        # Display step
        st.markdown(f"""
        <div class="workflow-step {step_status}">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 1.5rem;">{node_info['icon']}</div>
                <div style="flex: 1;">
                    <h4 style="margin: 0; color: #2c3e50;">{node_info['ui_name']}</h4>
                    <p style="margin: 0; color: #666; font-size: 0.9rem;">{node_info['description']}</p>
                    <div class="node-status-details">
                        LangGraph Node: <code>{node_name}</code>
                        {f" | Started: {start_time.strftime('%H:%M:%S')}" if start_time else ""}
                        {f" | Ended: {end_time.strftime('%H:%M:%S')}" if end_time else ""}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.2rem;">{status_emoji}</div>
                    <small style="color: #666; font-size: 0.8rem;">{status_text}</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Status summary
    if status['running'] > 0:
        status_message = f"🧠 LIVE: LangGraph executing {current_ui_name}"
    elif status['completed'] == status['total_nodes']:
        status_message = "🎉 All LangGraph nodes completed successfully!"
    elif status['errors'] > 0:
        status_message = f"❌ {status['errors']} LangGraph node(s) failed"
    else:
        status_message = "🚀 LangGraph workflow ready to execute..."
    
    st.markdown(f"""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.8); border-radius: 10px; position: relative; z-index: 2;">
        <p style="margin: 0; font-weight: 600; color: #2c3e50;">{status_message}</p>
    </div>
    </div>
    """, unsafe_allow_html=True)

# Helper functions (keeping existing ones)
def safe_get(dictionary, keys, default=None):
    """Safely get nested dictionary values"""
    if isinstance(keys, str):
        keys = [keys]
    
    current = dictionary
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def calculate_age(birth_date):
    """Calculate age from birth date"""
    if not birth_date:
        return None
    today = datetime.now().date()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age

def validate_patient_data(data: Dict[str, Any]) -> tuple[bool, list[str]]:
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
        if not data.get(field):
            errors.append(f"{display_name} is required")
        elif field == 'ssn' and len(str(data[field])) < 9:
            errors.append("SSN must be at least 9 digits")
        elif field == 'zip_code' and len(str(data[field])) < 5:
            errors.append("Zip code must be at least 5 digits")
    
    if data.get('date_of_birth'):
        try:
            birth_date = datetime.strptime(data['date_of_birth'], '%Y-%m-%d').date()
            age = calculate_age(birth_date)
            
            if age and age > 150:
                errors.append("Age cannot be greater than 150 years")
            elif age and age < 0:
                errors.append("Date of birth cannot be in the future")
        except:
            errors.append("Invalid date format")
    
    return len(errors) == 0, errors

# Initialize session state with enhanced tracking
def initialize_session_state():
    """Initialize session state variables for real-time processing"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'agent_wrapper' not in st.session_state:
        st.session_state.agent_wrapper = None
    if 'config' not in st.session_state:
        st.session_state.config = None
    if 'chatbot_context' not in st.session_state:
        st.session_state.chatbot_context = None
    if 'calculated_age' not in st.session_state:
        st.session_state.calculated_age = None
    if 'realtime_tracker' not in st.session_state:
        st.session_state.realtime_tracker = None

def reset_workflow():
    """Reset workflow to initial state"""
    st.session_state.realtime_tracker = RealTimeLangGraphTracker()

initialize_session_state()

# Enhanced Main Title
st.markdown('<h1 class="main-header">🧠 Real-time LangGraph Health Agent</h1>', unsafe_allow_html=True)

# Display import status
if not AGENT_AVAILABLE:
    st.markdown(f'<div class="status-error">Failed to import Health Agent: {import_error}</div>', unsafe_allow_html=True)
    st.stop()

# Sidebar with enhanced real-time info
with st.sidebar:
    st.title("🔄 Real-time Monitor")
    st.info("**TRUE LangGraph Synchronization**")
    st.markdown("---")
    st.markdown("**Real-time Features:**")
    st.markdown("• **Live Node Execution**: Real callbacks from LangGraph nodes")
    st.markdown("• **Actual Status Updates**: True node completion tracking") 
    st.markdown("• **Error Detection**: Real failure reporting from nodes")
    st.markdown("• **Execution Timing**: Actual node duration measurement")
    st.markdown("• **Thread-safe Updates**: Concurrent UI refreshing")
    st.markdown("---")
    if st.session_state.realtime_tracker:
        current_status = st.session_state.realtime_tracker.get_current_status()
        st.metric("Nodes Completed", f"{current_status['completed']}/{current_status['total_nodes']}")
        st.metric("Current Progress", f"{current_status['progress_percent']:.0f}%")

# 1. PATIENT INFORMATION
st.markdown("""
<div class="section-box">
    <div class="section-title">Patient Information</div>
</div>
""", unsafe_allow_html=True)

with st.container():
    with st.form("patient_input_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            first_name = st.text_input("First Name *", value="", type="password")
            last_name = st.text_input("Last Name *", value="", type="password")
        
        with col2:
            ssn = st.text_input("SSN *", value="", type="password")
            date_of_birth = st.date_input(
                "Date of Birth *", 
                value=datetime.now().date(),
                min_value=datetime(1900, 1, 1).date(),
                max_value=datetime.now().date()
            )
        
        with col3:
            gender = st.selectbox("Gender *", ["F", "M"])
            zip_code = st.text_input("Zip Code *", value="", type="password")
        
        # Show calculated age
        if date_of_birth:
            calculated_age = calculate_age(date_of_birth)
            if calculated_age is not None:
                st.session_state.calculated_age = calculated_age
                st.info(f"**Calculated Age:** {calculated_age} years old")
        elif st.session_state.calculated_age is not None:
            st.info(f"**Calculated Age:** {st.session_state.calculated_age} years old")
        
        # RUN ANALYSIS BUTTON
        submitted = st.form_submit_button(
            "🚀 Run REAL-TIME LangGraph Analysis", 
            use_container_width=True,
            disabled=st.session_state.analysis_running,
            type="primary"
        )

# Handle form submission with enhanced real-time tracking
if submitted:
    # Validate form data
    patient_data = {
        "first_name": first_name,
        "last_name": last_name,
        "ssn": ssn,
        "date_of_birth": date_of_birth.strftime('%Y-%m-%d'),
        "gender": gender,
        "zip_code": zip_code
    }
    
    valid, errors = validate_patient_data(patient_data)
    
    if not valid:
        st.error("Please fix the following errors:")
        for error in errors:
            st.error(f"• {error}")
    else:
        # Start real-time analysis
        reset_workflow()
        st.session_state.analysis_running = True
        st.session_state.analysis_results = None
        st.session_state.chatbot_context = None
        st.session_state.calculated_age = None
        
        # Initialize agent and enhanced wrapper
        try:
            config = Config()
            st.session_state.config = config
            st.session_state.agent = HealthAnalysisAgent(config)
            st.session_state.agent_wrapper = EnhancedLangGraphWrapper(st.session_state.agent)
            
            # DEBUG: Show enhanced agent setup
            with st.expander("🔍 **ENHANCED Real-time LangGraph Configuration**", expanded=False):
                st.write("**Real-time LangGraph Integration:**")
                
                if hasattr(st.session_state.agent, 'graph'):
                    st.success("✅ LangGraph StateGraph compiled and ready")
                if hasattr(st.session_state.agent_wrapper, 'tracker'):
                    st.success("✅ Real-time progress tracker initialized")
                if hasattr(st.session_state.agent_wrapper, 'inject_realtime_callbacks'):
                    st.success("✅ Callback injection system ready")
                
                st.write("**Enhanced Node Tracking:**")
                tracker = st.session_state.agent_wrapper.tracker
                for node_name, node_info in tracker.node_mapping.items():
                    st.write(f"• `{node_name}` → **{node_info['ui_name']}** {node_info['icon']}")
                
                st.write("**Real-time Capabilities:**")
                st.write("• ✅ Thread-safe progress tracking")
                st.write("• ✅ Node execution timing")
                st.write("• ✅ Real error capture and reporting")
                st.write("• ✅ Live UI updates during execution")
                st.write("• ✅ Callback injection into all nodes")
            
        except Exception as e:
            st.error(f"Failed to initialize enhanced LangGraph agent: {str(e)}")
            st.session_state.analysis_running = False
            st.stop()
        
        st.rerun()

# Display REAL-TIME synchronized workflow and execute LangGraph
if st.session_state.analysis_running:
    st.markdown("## 🧠 REAL-TIME LangGraph Execution")
    
    # Create placeholders for REAL-TIME updates
    workflow_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    # Show initial workflow state
    initial_status = {
        'total_nodes': 8,
        'completed': 0,
        'running': 0,
        'errors': 0,
        'progress_percent': 0,
        'current_node': None,
        'node_status': {}
    }
    
    _display_realtime_workflow(initial_status)
    
    with progress_placeholder.container():
        st.info("🚀 Starting REAL-TIME LangGraph healthcare analysis...")
    
    # Execute LangGraph with REAL-TIME UPDATES
    try:
        patient_data = {
            "first_name": first_name,
            "last_name": last_name,
            "ssn": ssn,
            "date_of_birth": date_of_birth.strftime('%Y-%m-%d'),
            "gender": gender,
            "zip_code": zip_code
        }
        
        # Run LangGraph with REAL-TIME callbacks
        results = st.session_state.agent_wrapper.run_with_realtime_updates(
            patient_data, 
            workflow_placeholder,
            progress_placeholder
        )
        
        # Store results
        st.session_state.analysis_results = results
        st.session_state.analysis_running = False
        
        if results and results.get("success") and results.get("chatbot_ready"):
            st.session_state.chatbot_context = results.get("chatbot_context")
        
        st.rerun()
        
    except Exception as e:
        st.session_state.analysis_running = False
        
        with progress_placeholder.container():
            st.error(f"❌ Real-time LangGraph execution failed: {str(e)}")
        
        # Show error in final workflow display
        error_status = st.session_state.agent_wrapper.tracker.get_current_status() if st.session_state.agent_wrapper else initial_status
        error_status['errors'] = error_status.get('errors', 0) + 1
        
        with workflow_placeholder.container():
            _display_realtime_workflow(error_status)

# Display results after completion (keeping existing results display code)
if st.session_state.analysis_results and not st.session_state.analysis_running:
    results = st.session_state.analysis_results
    
    if results.get("success"):
        # Success banner
        st.markdown("""
        <div class="analysis-complete-banner">
            <h2 style="margin: 0; color: #28a745; font-weight: 700;">🎉 REAL-TIME LangGraph Analysis Complete!</h2>
            <p style="margin: 0.5rem 0; color: #155724; font-size: 1.1rem;">
                All LangGraph nodes executed successfully with real-time progress tracking.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show execution summary
        if st.session_state.agent_wrapper and st.session_state.agent_wrapper.tracker:
            final_status = st.session_state.agent_wrapper.tracker.get_current_status()
            st.markdown("### 📊 Execution Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Nodes", final_status['total_nodes'])
            with col2:
                st.metric("Completed", final_status['completed'])
            with col3:
                st.metric("Success Rate", f"{(final_status['completed']/final_status['total_nodes']*100):.0f}%")
            with col4:
                total_duration = sum([
                    node.duration for node in final_status['node_status'].values() 
                    if hasattr(node, 'duration') and node.duration
                ]) if final_status['node_status'] else 0
                st.metric("Total Time", f"{total_duration:.1f}s" if total_duration else "N/A")
        
        # Chatbot launch button
        if results.get("chatbot_ready", False) and st.session_state.chatbot_context:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(
                    "🚀 Launch Medical Assistant", 
                    key="launch_chatbot_main",
                    use_container_width=True,
                    help="Open the Medical Assistant with full LangGraph analysis data"
                ):
                    st.switch_page("pages/chatbot.py")
    else:
        st.error("❌ Real-time LangGraph analysis encountered errors")
        if results.get('errors'):
            for error in results['errors']:
                st.error(f"• {error}")

if __name__ == "__main__":
    pass
