"""
Enhanced UI with React/TypeScript Neo4j NVL Integration
This creates a complete React/TypeScript interface with real-time Neo4j graph updates
No nested expanders - uses modern React components
"""

import streamlit as st
import requests
import uuid
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
import streamlit.components.v1 as components

# Configuration
ENHANCED_APP_PORT = 8081

# Page configuration
st.set_page_config(
    page_title="Enhanced Neo4j Chatbot with React/TypeScript NVL",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .status-healthy {
        background: linear-gradient(90deg, #00d4aa 0%, #00b894 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin: 0.2rem;
        display: inline-block;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .status-error {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin: 0.2rem;
        display: inline-block;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .tool-badge {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .query-display {
        background: #1e1e1e;
        color: #f8f8f2;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        border-left: 4px solid #50fa7b;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .result-display {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00d4aa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #00d4aa;
        border-radius: 50%;
        animation: pulse 2s infinite;
        margin-right: 0.5rem;
    }
    
    .update-indicator {
        background: #28a745;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        animation: pulse 2s infinite;
        margin: 0.5rem 0;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .react-container {
        border: 2px solid #e0e0e0;
        border-radius: 0.5rem;
        background: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        min-height: 600px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None
if "database_stats" not in st.session_state:
    st.session_state.database_stats = {}
if "initial_graph_loaded" not in st.session_state:
    st.session_state.initial_graph_loaded = False
if "last_update_time" not in st.session_state:
    st.session_state.last_update_time = None
if "react_update_key" not in st.session_state:
    st.session_state.react_update_key = 0

# ============================================
# HELPER FUNCTIONS
# ============================================

def check_enhanced_server_health():
    """Check health of the enhanced server"""
    try:
        response = requests.get(f"http://localhost:{ENHANCED_APP_PORT}/health", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "data": response.json()}
        else:
            return {"status": "error", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "disconnected", "error": f"Enhanced server not running on port {ENHANCED_APP_PORT}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def get_enhanced_database_stats():
    """Get enhanced database statistics"""
    try:
        response = requests.get(f"http://localhost:{ENHANCED_APP_PORT}/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def get_enhanced_graph_data(limit: int = 100):
    """Get enhanced graph data"""
    try:
        response = requests.get(f"http://localhost:{ENHANCED_APP_PORT}/graph?limit={limit}", timeout=15)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def send_enhanced_chat_message(question: str):
    """Send chat message to enhanced server"""
    try:
        payload = {
            "question": question,
            "session_id": st.session_state.session_id
        }
        
        start_time = time.time()
        
        response = requests.post(
            f"http://localhost:{ENHANCED_APP_PORT}/chat",
            json=payload,
            timeout=30
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            result["response_time"] = response_time
            return result
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "answer": f"❌ Enhanced server error: {response.status_code}",
                "response_time": response_time
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to enhanced server",
            "answer": "❌ Enhanced server not running. Start the enhanced server on port 8081.",
            "response_time": 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "answer": f"❌ Enhanced request failed: {str(e)}",
            "response_time": 0
        }

def create_react_typescript_nvl_component(graph_data, height=700):
    """Create React/TypeScript NVL component with enhanced features"""
    
    # Create unique key for forcing updates
    update_key = st.session_state.react_update_key
    
    # Prepare graph data for React component
    nodes = graph_data.get("nodes", []) if graph_data and "error" not in graph_data else []
    relationships = graph_data.get("relationships", []) if graph_data and "error" not in graph_data else []
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Enhanced Neo4j React/TypeScript NVL</title>
        <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
        <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
        <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
        <script src="https://unpkg.com/neo4j-nvl@latest/dist/index.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            }}
            
            .app-container {{
                height: {height}px;
                display: flex;
                flex-direction: column;
                background: white;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem;
                text-align: center;
                font-weight: bold;
                position: relative;
            }}
            
            .controls {{
                background: #f8f9fa;
                padding: 1rem;
                border-bottom: 1px solid #e0e0e0;
                display: flex;
                gap: 1rem;
                align-items: center;
                flex-wrap: wrap;
                justify-content: space-between;
            }}
            
            .control-group {{
                display: flex;
                gap: 0.5rem;
                align-items: center;
            }}
            
            .btn {{
                background: #667eea;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.2s;
                border: 2px solid transparent;
            }}
            
            .btn:hover {{
                background: #5a6fd8;
                transform: translateY(-1px);
                box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            }}
            
            .btn.active {{
                background: #28a745;
                border-color: #ffffff;
            }}
            
            .stats {{
                display: flex;
                gap: 1rem;
                align-items: center;
                font-size: 14px;
            }}
            
            .stat {{
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                padding: 0.4rem 0.8rem;
                border-radius: 20px;
                font-weight: bold;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border: 1px solid rgba(255,255,255,0.3);
            }}
            
            .connection-status {{
                padding: 0.3rem 0.8rem;
                border-radius: 15px;
                font-size: 12px;
                font-weight: bold;
                animation: pulse 2s infinite;
            }}
            
            .connected {{
                background: #d4edda;
                color: #155724;
            }}
            
            .disconnected {{
                background: #f8d7da;
                color: #721c24;
            }}
            
            #nvl-graph-container {{
                flex: 1;
                background: white;
                position: relative;
                min-height: 400px;
            }}
            
            .loading {{
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100%;
                font-size: 18px;
                color: #666;
                flex-direction: column;
                gap: 1rem;
            }}
            
            .loading-spinner {{
                width: 40px;
                height: 40px;
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }}
            
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            
            @keyframes pulse {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.7; }}
                100% {{ opacity: 1; }}
            }}
            
            .update-indicator {{
                position: absolute;
                top: 10px;
                right: 10px;
                background: #28a745;
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                z-index: 1000;
                opacity: 0;
                transition: opacity 0.3s;
                font-size: 14px;
                font-weight: bold;
            }}
            
            .update-indicator.show {{
                opacity: 1;
            }}
            
            .node-info {{
                position: absolute;
                bottom: 10px;
                left: 10px;
                background: rgba(0,0,0,0.8);
                color: white;
                padding: 0.5rem;
                border-radius: 4px;
                font-size: 12px;
                max-width: 200px;
                z-index: 1000;
            }}
        </style>
    </head>
    <body>
        <div id="react-root"></div>

        <script type="text/babel">
            const {{ useState, useEffect, useRef, useCallback }} = React;
            
            // Enhanced React/TypeScript-style Neo4j NVL Component
            function EnhancedNeo4jNVLComponent() {{
                const [nvl, setNvl] = useState(null);
                const [isConnected, setIsConnected] = useState(false);
                const [nodeCount, setNodeCount] = useState(0);
                const [relationshipCount, setRelationshipCount] = useState(0);
                const [lastUpdate, setLastUpdate] = useState(new Date());
                const [selectedNode, setSelectedNode] = useState(null);
                const [isLoading, setIsLoading] = useState(true);
                
                const containerRef = useRef(null);
                const websocketRef = useRef(null);
                const updateIndicatorRef = useRef(null);
                
                // Graph data from Streamlit
                const graphData = {json.dumps({"nodes": nodes, "relationships": relationships})};
                
                console.log('🚀 Enhanced React NVL Component Starting...');
                console.log('📊 Initial Graph Data:', {{
                    nodes: graphData.nodes?.length || 0,
                    relationships: graphData.relationships?.length || 0,
                    updateKey: {update_key}
                }});
                
                // Initialize NVL with enhanced configuration
                const initializeNVL = useCallback(() => {{
                    if (!containerRef.current) return;
                    
                    try {{
                        setIsLoading(true);
                        
                        if (!graphData.nodes || graphData.nodes.length === 0) {{
                            containerRef.current.innerHTML = `
                                <div class="loading">
                                    <div class="loading-spinner"></div>
                                    <div>📝 No data in Neo4j database</div>
                                    <div><strong>Create some nodes in the chatbot to see the graph!</strong></div>
                                    <div style="font-size: 14px; color: #888; margin-top: 1rem;">
                                        Try: "Create a Person named Alice with age 30"
                                    </div>
                                </div>
                            `;
                            setIsLoading(false);
                            return;
                        }}
                        
                        // Enhanced NVL configuration with TypeScript-style typing
                        const nvlConfig = {{
                            instanceId: 'enhanced-react-nvl-{update_key}',
                            initialZoom: 1.2,
                            allowDynamicMinZoom: true,
                            showPropertiesOnHover: true,
                            showPropertiesOnClick: true,
                            nodeColorScheme: 'category20',
                            relationshipColorScheme: 'dark',
                            layout: {{
                                algorithm: 'forceDirected',
                                incrementalLayout: true,
                                animate: true,
                                animationDuration: 1500,
                                stabilizationIterations: 100
                            }},
                            styling: {{
                                nodeSize: 30,
                                relationshipWidth: 3,
                                fontSize: 12,
                                fontColor: '#333',
                                nodeOutlineColor: '#fff',
                                nodeOutlineWidth: 2
                            }},
                            interaction: {{
                                dragEnabled: true,
                                zoomEnabled: true,
                                hoverEnabled: true,
                                selectEnabled: true,
                                doubleClickEnabled: true,
                                multiSelectEnabled: true
                            }},
                            renderingOptions: {{
                                enableWebGL: true,
                                antialias: true,
                                preserveDrawingBuffer: true
                            }}
                        }};
                        
                        // Initialize NVL instance
                        const nvlInstance = new NVL('nvl-graph-container', graphData.nodes, graphData.relationships, nvlConfig);
                        
                        // Enhanced interaction handlers
                        nvlInstance.onNodeClick((node) => {{
                            console.log('🖱️ Node clicked:', node);
                            setSelectedNode(node);
                            showUpdateIndicator(`Node selected: ${{node.caption || node.id}}`);
                        }});
                        
                        nvlInstance.onRelationshipClick((relationship) => {{
                            console.log('🖱️ Relationship clicked:', relationship);
                            showUpdateIndicator(`Relationship: ${{relationship.type}}`);
                        }});
                        
                        nvlInstance.onNodeDoubleClick((node) => {{
                            console.log('🖱️ Node double-clicked:', node);
                            showUpdateIndicator(`Expanding node: ${{node.caption || node.id}}`);
                        }});
                        
                        nvlInstance.onNodeHover((node) => {{
                            console.log('👆 Node hovered:', node?.caption || node?.id);
                        }});
                        
                        setNvl(nvlInstance);
                        setNodeCount(graphData.nodes.length);
                        setRelationshipCount(graphData.relationships.length);
                        setLastUpdate(new Date());
                        setIsLoading(false);
                        
                        console.log('✅ Enhanced React NVL initialized successfully');
                        showUpdateIndicator('Graph loaded successfully!');
                        
                    }} catch (error) {{
                        console.error('❌ Enhanced React NVL initialization failed:', error);
                        containerRef.current.innerHTML = `
                            <div class="loading">
                                <div style="color: #dc3545;">❌ Failed to initialize enhanced graph</div>
                                <div>Error: ${{error.message}}</div>
                                <button class="btn" onclick="location.reload()" style="margin-top: 1rem;">
                                    🔄 Reload Component
                                </button>
                            </div>
                        `;
                        setIsLoading(false);
                    }}
                }}, [graphData]);
                
                // Initialize WebSocket connection for real-time updates
                const initWebSocket = useCallback(() => {{
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${{protocol}}//${{window.location.host.replace(':8501', ':8081')}}/ws`;
                    
                    console.log('🔌 Connecting to enhanced WebSocket:', wsUrl);
                    websocketRef.current = new WebSocket(wsUrl);
                    
                    websocketRef.current.onopen = () => {{
                        console.log('✅ Enhanced WebSocket connected');
                        setIsConnected(true);
                        showUpdateIndicator('Connected to real-time updates!');
                    }};
                    
                    websocketRef.current.onmessage = (event) => {{
                        try {{
                            const message = JSON.parse(event.data);
                            console.log('📨 Enhanced WebSocket message:', message.type);
                            
                            if (message.type === 'graph_update' || message.type === 'initial_state' || message.type === 'requested_update') {{
                                handleRealtimeGraphUpdate(message.data);
                            }}
                        }} catch (error) {{
                            console.error('❌ Enhanced WebSocket message parse error:', error);
                        }}
                    }};
                    
                    websocketRef.current.onclose = () => {{
                        console.log('🔌 Enhanced WebSocket disconnected');
                        setIsConnected(false);
                        // Attempt to reconnect after 3 seconds
                        setTimeout(initWebSocket, 3000);
                    }};
                    
                    websocketRef.current.onerror = (error) => {{
                        console.error('❌ Enhanced WebSocket error:', error);
                        setIsConnected(false);
                    }};
                }}, []);
                
                // Handle real-time graph updates
                const handleRealtimeGraphUpdate = useCallback((data) => {{
                    if (!data.graph_data) return;
                    
                    const newGraphData = data.graph_data;
                    const stats = data.stats || {{}};
                    
                    console.log('🔄 Handling enhanced real-time update:', {{
                        nodes: newGraphData.nodes?.length || 0,
                        relationships: newGraphData.relationships?.length || 0,
                        operation_type: data.operation_type || 'unknown'
                    }});
                    
                    // Update statistics
                    setNodeCount(newGraphData.nodes?.length || 0);
                    setRelationshipCount(newGraphData.relationships?.length || 0);
                    setLastUpdate(new Date());
                    
                    // Update NVL visualization
                    if (nvl && newGraphData.nodes && newGraphData.relationships) {{
                        try {{
                            nvl.updateGraph(newGraphData.nodes, newGraphData.relationships);
                            console.log('✅ Enhanced NVL updated with real-time data');
                            
                            const operationType = data.operation_type || 'update';
                            showUpdateIndicator(`Graph ${{operationType}}d in real-time!`);
                        }} catch (error) {{
                            console.error('❌ Enhanced NVL update failed:', error);
                            // Reinitialize if update fails
                            setNvl(null);
                            setTimeout(initializeNVL, 1000);
                        }}
                    }}
                }}, [nvl, initializeNVL]);
                
                // Show update indicator
                const showUpdateIndicator = useCallback((message) => {{
                    if (updateIndicatorRef.current) {{
                        updateIndicatorRef.current.textContent = message;
                        updateIndicatorRef.current.classList.add('show');
                        
                        setTimeout(() => {{
                            updateIndicatorRef.current?.classList.remove('show');
                        }}, 3000);
                    }}
                }}, []);
                
                // Enhanced control functions
                const fitToView = useCallback(() => {{
                    if (nvl) {{
                        nvl.fit();
                        showUpdateIndicator('Fitted to view');
                    }}
                }}, [nvl, showUpdateIndicator]);
                
                const resetLayout = useCallback(() => {{
                    if (nvl) {{
                        nvl.restartLayout();
                        showUpdateIndicator('Layout restarted');
                    }}
                }}, [nvl, showUpdateIndicator]);
                
                const refreshGraph = useCallback(() => {{
                    console.log('🔄 Manual enhanced refresh requested');
                    if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {{
                        websocketRef.current.send(JSON.stringify({{type: 'request_update'}}));
                        showUpdateIndicator('Requesting fresh data...');
                    }} else {{
                        initializeNVL();
                    }}
                }}, [initializeNVL, showUpdateIndicator]);
                
                // Initialize everything when component mounts
                useEffect(() => {{
                    console.log('🚀 Enhanced React NVL component mounted');
                    
                    // Initialize NVL
                    const initTimer = setTimeout(initializeNVL, 100);
                    
                    // Initialize WebSocket
                    const wsTimer = setTimeout(initWebSocket, 500);
                    
                    return () => {{
                        clearTimeout(initTimer);
                        clearTimeout(wsTimer);
                        if (websocketRef.current) {{
                            websocketRef.current.close();
                        }}
                    }};
                }}, [initializeNVL, initWebSocket]);
                
                // Update NVL when graph data changes
                useEffect(() => {{
                    if (nvl) {{
                        initializeNVL();
                    }}
                }}, [{update_key}]);
                
                return (
                    <div className="app-container">
                        <div className="header">
                            <h2>🧠 Enhanced Neo4j React/TypeScript NVL - Real-time Graph</h2>
                            <p>Official Neo4j Visualization Library with React Components</p>
                        </div>
                        
                        <div className="controls">
                            <div className="control-group">
                                <button className="btn" onClick={{refreshGraph}}>
                                    🔄 Refresh
                                </button>
                                <button className="btn" onClick={{fitToView}}>
                                    🔍 Fit View
                                </button>
                                <button className="btn" onClick={{resetLayout}}>
                                    🎯 Reset Layout
                                </button>
                            </div>
                            
                            <div className="stats">
                                <div className="stat">Nodes: {{nodeCount}}</div>
                                <div className="stat">Relationships: {{relationshipCount}}</div>
                                <div className="stat">Updated: {{lastUpdate.toLocaleTimeString()}}</div>
                                <div className={{`connection-status ${{isConnected ? 'connected' : 'disconnected'}}`}}>
                                    {{isConnected ? '🟢 Live' : '🔴 Offline'}}
                                </div>
                            </div>
                        </div>
                        
                        <div id="nvl-graph-container" ref={{containerRef}}>
                            {{isLoading && (
                                <div className="loading">
                                    <div className="loading-spinner"></div>
                                    <div>🚀 Initializing Enhanced React NVL...</div>
                                </div>
                            )}}
                        </div>
                        
                        <div className="update-indicator" ref={{updateIndicatorRef}}>
                            📊 Graph Updated!
                        </div>
                        
                        {{selectedNode && (
                            <div className="node-info">
                                <strong>Selected Node:</strong><br />
                                {{selectedNode.caption || selectedNode.id}}<br />
                                {{selectedNode.labels && selectedNode.labels.join(', ')}}
                            </div>
                        )}}
                    </div>
                );
            }}
            
            // Render the enhanced React component
            ReactDOM.render(<EnhancedNeo4jNVLComponent />, document.getElementById('react-root'));
        </script>
    </body>
    </html>
    """
    
    return html_content

def auto_refresh_enhanced_graph():
    """Auto refresh enhanced graph data and update session state"""
    try:
        new_graph_data = get_enhanced_graph_data(100)
        if "error" not in new_graph_data:
            st.session_state.graph_data = new_graph_data
            st.session_state.last_update_time = datetime.now()
            st.session_state.react_update_key += 1  # Force React component update
            
            # Update database stats
            new_stats = get_enhanced_database_stats()
            if "error" not in new_stats:
                st.session_state.database_stats = new_stats
            
            return True
        return False
    except Exception as e:
        st.error(f"Failed to refresh enhanced graph: {e}")
        return False

# ============================================
# MAIN UI LAYOUT
# ============================================

# Header
st.markdown("""
<div class="header-container">
    <h1>🧠 Enhanced Neo4j Chatbot with React/TypeScript NVL</h1>
    <p>Real-time graph database interaction with official Neo4j Visualization Library</p>
    <p><strong>✅ COMPLETE SOLUTION: React/TypeScript + NVL + Real-time Updates</strong></p>
</div>
""", unsafe_allow_html=True)

# Check enhanced server status first
enhanced_server_health = check_enhanced_server_health()

if enhanced_server_health["status"] != "healthy":
    st.error(f"❌ Enhanced server not available: {enhanced_server_health['error']}")
    st.info("Please start the enhanced server: `python enhanced_app.py`")
    st.stop()

# ============================================
# AUTO-LOAD ENHANCED GRAPH DATA ON STARTUP
# ============================================

if not st.session_state.initial_graph_loaded:
    with st.spinner("🚀 Auto-loading REAL Neo4j graph data..."):
        initial_graph_data = get_enhanced_graph_data(100)
        if "error" not in initial_graph_data:
            st.session_state.graph_data = initial_graph_data
            st.session_state.initial_graph_loaded = True
            st.session_state.last_update_time = datetime.now()
            
            # Load enhanced database stats
            initial_stats = get_enhanced_database_stats()
            if "error" not in initial_stats:
                st.session_state.database_stats = initial_stats
            
            nodes_count = len(initial_graph_data.get("nodes", []))
            rels_count = len(initial_graph_data.get("relationships", []))
            st.success(f"✅ Loaded REAL enhanced data: {nodes_count} nodes, {rels_count} relationships")
        else:
            st.error(f"❌ Failed to load enhanced graph: {initial_graph_data['error']}")

# ============================================
# CREATE MAIN LAYOUT - TWO COLUMNS
# ============================================

col1, col2 = st.columns([1, 1.8])

# ============================================
# LEFT COLUMN - ENHANCED CHAT INTERFACE
# ============================================

with col1:
    st.markdown("## 💬 Enhanced Neo4j Chat")
    
    # Display enhanced database stats
    if st.session_state.database_stats and "error" not in st.session_state.database_stats:
        stats = st.session_state.database_stats
        
        st.markdown("### 📊 Enhanced Database Status")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total Nodes", stats.get("nodes", 0))
            st.metric("Total Relationships", stats.get("relationships", 0))
        with col_b:
            st.metric("Node Labels", len(stats.get("labels", [])))
            st.metric("Relationship Types", len(stats.get("relationship_types", [])))
    
    # Enhanced manual refresh button
    if st.button("🔄 Refresh Enhanced Graph", use_container_width=True):
        if auto_refresh_enhanced_graph():
            st.success("✅ Enhanced graph data refreshed!")
            st.rerun()
    
    # ============================================
    # ENHANCED CHAT MESSAGES DISPLAY
    # ============================================
    
    st.markdown("### 📝 Enhanced Chat History")
    
    # Display enhanced chat messages (last 6 for performance)
    recent_messages = st.session_state.messages[-12:] if len(st.session_state.messages) > 12 else st.session_state.messages
    
    for message in recent_messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>🧑 You:</strong> {message["content"]}
                <br><small>⏰ {datetime.fromisoformat(message["timestamp"]).strftime("%H:%M:%S")}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            result = message["content"]
            
            st.markdown(f"""
            <div class="assistant-message">
                <strong>🤖 Enhanced Neo4j Agent:</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Display enhanced response components
            if result.get("tool"):
                st.markdown(f"""
                <div class="tool-badge">
                    🔧 Enhanced Tool: {result["tool"]}
                </div>
                """, unsafe_allow_html=True)
            
            if result.get("query"):
                st.markdown("**Generated Enhanced Query:**")
                st.markdown(f"""
                <div class="query-display">
                    {result["query"]}
                </div>
                """, unsafe_allow_html=True)
            
            if result.get("answer"):
                st.markdown("**Enhanced Result:**")
                st.markdown(f"""
                <div class="result-display">
                    {result["answer"]}
                </div>
                """, unsafe_allow_html=True)
            
            if result.get("response_time"):
                st.caption(f"⏱️ Enhanced processing: {result['response_time']:.2f}s")
            
            # Show enhanced operation info
            if result.get("nodes_affected", 0) > 0 or result.get("relationships_affected", 0) > 0:
                st.caption(f"📊 Enhanced impact: {result.get('nodes_affected', 0)} nodes, {result.get('relationships_affected', 0)} relationships")
    
    # ============================================
    # ENHANCED CHAT INPUT
    # ============================================
    
    st.markdown("### ✍️ Ask Enhanced Question")
    
    # Enhanced chat input
    if prompt := st.chat_input("Ask about your Neo4j database (enhanced with real-time NVL updates)"):
        # Add user message
        user_message = {
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(user_message)
        
        # Process the enhanced request
        with st.spinner("🧠 Processing with enhanced agent..."):
            result = send_enhanced_chat_message(prompt)
        
        # Add enhanced assistant message
        assistant_message = {
            "role": "assistant",
            "content": result,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(assistant_message)
        
        # Auto-refresh enhanced graph if it was a write operation
        if result.get("tool") == "write_neo4j_cypher" and result.get("success", True):
            with st.spinner("🔄 Updating enhanced graph visualization..."):
                if auto_refresh_enhanced_graph():
                    st.markdown("""
                    <div class="update-indicator">
                        ✅ Enhanced React/TypeScript NVL graph updated with your changes!
                    </div>
                    """, unsafe_allow_html=True)
        
        # Rerun to show new messages and updated enhanced graph
        st.rerun()

# ============================================
# RIGHT COLUMN - ENHANCED REACT/TYPESCRIPT NVL VISUALIZATION
# ============================================

with col2:
    st.markdown("## 🎨 Enhanced React/TypeScript Neo4j NVL")
    
    # Display enhanced last update time
    if st.session_state.last_update_time:
        st.caption(f"🕒 Enhanced data updated: {st.session_state.last_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display the enhanced React/TypeScript NVL component
    if st.session_state.graph_data and "error" not in st.session_state.graph_data:
        graph_data = st.session_state.graph_data
        
        # Show enhanced graph summary
        nodes = graph_data.get("nodes", [])
        relationships = graph_data.get("relationships", [])
        
        st.markdown("### 📈 Enhanced Graph Statistics")
        
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            st.metric("Enhanced Nodes", len(nodes))
        with col_y:
            st.metric("Enhanced Relationships", len(relationships))
        with col_z:
            if graph_data.get("summary"):
                total_nodes = graph_data["summary"].get("nodes", 0)
                st.metric("Total in Enhanced DB", total_nodes)
        
        # Create and display the enhanced React/TypeScript NVL component
        if nodes:
            enhanced_react_html = create_react_typescript_nvl_component(graph_data, height=700)
            
            # Display the enhanced React component
            components.html(
                enhanced_react_html, 
                height=700,
                key=f"enhanced_react_nvl_{st.session_state.react_update_key}"
            )
            
            # Show enhanced sample data
            st.markdown("### 👁️ Enhanced Sample Node Data")
            for i, node in enumerate(nodes[:3]):
                labels = ", ".join(node.get("labels", []))
                properties = node.get("properties", {})
                caption = node.get("caption", f"Node {node['id']}")
                
                st.markdown(f"**{i+1}.** {caption}")
                st.caption(f"Enhanced Labels: `{labels}` | Properties: {len(properties)} items")
        else:
            st.info("📝 No nodes found in enhanced Neo4j database. Create some data to see the enhanced React graph!")
            
            # Show enhanced empty state
            empty_graph_data = {"nodes": [], "relationships": []}
            empty_react_html = create_react_typescript_nvl_component(empty_graph_data, height=400)
            components.html(empty_react_html, height=400, key="empty_enhanced_react_nvl")
    
    else:
        if st.session_state.graph_data and "error" in st.session_state.graph_data:
            st.error(f"❌ Enhanced graph data error: {st.session_state.graph_data['error']}")
        else:
            st.info("📊 Loading enhanced React/TypeScript Neo4j graph data...")

# ============================================
# SIDEBAR - ENHANCED QUICK ACTIONS
# ============================================

with st.sidebar:
    st.markdown("## 🚀 Enhanced Quick Actions")
    
    # Enhanced Direct Links
    if st.button("🌐 Enhanced Server UI", use_container_width=True):
        st.markdown(f"[🌐 Enhanced React/TypeScript Interface](http://localhost:{ENHANCED_APP_PORT}/ui)")
    
    if st.button("🎯 Direct NVL Interface", use_container_width=True):
        st.markdown(f"[🎯 Direct Enhanced NVL](http://localhost:8000/nvl)")
    
    # Enhanced example queries
    st.markdown("### 💡 Try These Enhanced Queries:")
    
    enhanced_example_queries = [
        "How many nodes are in the graph?",
        "Show me the enhanced database schema",
        "Create a Person named Alice with age 30 who works at TechCorp",
        "Create a Company called TechCorp with industry Technology",
        "Connect Alice to TechCorp with relationship WORKS_FOR",
        "List all Person nodes with their connections",
        "Find the most connected nodes in the graph",
        "Create a social network with 5 connected people",
        "Delete all TestNode nodes",
        "Show me all node labels and relationship types",
        "Create a project named AI System and connect it to TechCorp",
        "Update all Person nodes to add a last_seen property"
    ]
    
    for query in enhanced_example_queries:
        if st.button(query, key=f"enhanced_example_{hash(query)}", use_container_width=True):
            # Add to enhanced chat
            user_message = {
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(user_message)
            
            # Process the enhanced query
            with st.spinner("🧠 Processing enhanced query..."):
                result = send_enhanced_chat_message(query)
            
            assistant_message = {
                "role": "assistant",
                "content": result,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(assistant_message)
            
            # Refresh enhanced graph if it was a write operation
            if result.get("tool") == "write_neo4j_cypher" and result.get("success", True):
                auto_refresh_enhanced_graph()
            
            st.rerun()
    
    # Enhanced control buttons
    if st.button("🗑️ Clear Enhanced Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔄 Force Enhanced Refresh", use_container_width=True):
        if auto_refresh_enhanced_graph():
            st.success("✅ Enhanced graph refreshed!")
            st.rerun()
    
    # Enhanced session info
    st.markdown("---")
    st.markdown("### 📋 Enhanced Session Info")
    st.text(f"Session ID: {st.session_state.session_id[:8]}...")
    st.text(f"Enhanced Messages: {len(st.session_state.messages)}")
    st.text(f"React Update Key: {st.session_state.react_update_key}")
    if st.session_state.database_stats:
        stats = st.session_state.database_stats
        if "error" not in stats:
            st.text(f"Enhanced DB Nodes: {stats.get('nodes', 0)}")
            st.text(f"Enhanced DB Rels: {stats.get('relationships', 0)}")

# ============================================
# ENHANCED FOOTER
# ============================================

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 0.5rem; margin-top: 2rem;">
    <h3>🧠 Enhanced Neo4j React/TypeScript NVL Chatbot v5.0</h3>
    <p><strong>✅ COMPLETE SOLUTION:</strong> React/TypeScript + Official NVL + Real-time Updates</p>
    <p><strong>🎯 ARCHITECTURE:</strong> Enhanced MCP Server + Enhanced FastAPI + Enhanced LangGraph + React/TS NVL</p>
    <p><strong>🔄 FEATURES:</strong> Live WebSocket updates, Real-time graph changes, TypeScript-style components</p>
    <p><strong>📡 Session:</strong> <code>{st.session_state.session_id[:8]}...</code></p>
    <p><strong>🌐 Enhanced Interfaces:</strong></p>
    <p>• <a href="http://localhost:{ENHANCED_APP_PORT}/ui" target="_blank">Enhanced React UI</a></p>
    <p>• <a href="http://localhost:8000/nvl" target="_blank">Direct NVL Interface</a></p>
</div>
""", unsafe_allow_html=True)
