"""
Enhanced FastAPI App with Real-time Neo4j Integration
This works with the enhanced MCP server and LangGraph agent
Run this on port 8081 (after starting the MCP server on port 8000)
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uuid
import logging
import requests
import time
import json
import asyncio
from typing import Optional, Set, Dict, Any
from datetime import datetime

# Import our enhanced agent
from enhanced_langgraph_agent import build_agent, AgentState

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced_fastapi_app")

# Configuration
APP_PORT = 8081
MCP_SERVER_PORT = 8000

print("🔧 Enhanced FastAPI App Configuration:")
print(f"   App Port: {APP_PORT}")
print(f"   MCP Server Port: {MCP_SERVER_PORT}")

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Neo4j LangGraph Agent API",
    description="FastAPI server with enhanced LangGraph agent and real-time Neo4j NVL support",
    version="4.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
agent = None
active_websockets: Set[WebSocket] = set()
last_graph_update = datetime.now()

# ============================================
# PYDANTIC MODELS
# ============================================

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    # Standard fields
    trace: str
    tool: str
    query: str
    answer: str
    session_id: str
    success: bool = True
    error: Optional[str] = None
    
    # Enhanced fields for real-time NVL
    intent: str = ""
    operation_type: str = ""
    raw_response: dict = {}
    processing_time: float = 0.0
    nodes_affected: int = 0
    relationships_affected: int = 0
    graph_changes: dict = {}
    # Real-time graph data
    graph_data: Optional[Dict[str, Any]] = None
    operation_summary: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    components: dict
    agent_ready: bool
    mcp_server_ready: bool
    websocket_connections: int

class GraphUpdateMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: str

# ============================================
# UTILITY FUNCTIONS
# ============================================

def check_mcp_server():
    """Check if the MCP server is running"""
    try:
        response = requests.get(f"http://localhost:{MCP_SERVER_PORT}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

async def get_real_graph_data_from_mcp(limit: int = 100):
    """Get real graph data from MCP server"""
    try:
        response = requests.get(f"http://localhost:{MCP_SERVER_PORT}/graph?limit={limit}", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"MCP server error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

async def get_database_stats_from_mcp():
    """Get database statistics from MCP server"""
    try:
        response = requests.get(f"http://localhost:{MCP_SERVER_PORT}/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"MCP server error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

async def broadcast_to_ui_clients(operation_type: str, data: Dict[str, Any]):
    """Broadcast updates to connected UI clients"""
    if not active_websockets:
        return
    
    try:
        message = GraphUpdateMessage(
            type="graph_update",
            data={
                "operation_type": operation_type,
                "update_data": data,
                "timestamp": datetime.now().isoformat()
            },
            timestamp=datetime.now().isoformat()
        )
        
        # Send to all connected UI clients
        disconnected = set()
        for websocket in active_websockets:
            try:
                await websocket.send_text(message.json())
                logger.info(f"Sent update to UI client: {operation_type}")
            except Exception as e:
                logger.warning(f"Failed to send update to UI client: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected clients
        active_websockets -= disconnected
        
    except Exception as e:
        logger.error(f"Failed to broadcast to UI clients: {e}")

# ============================================
# STARTUP/SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize the enhanced agent when the app starts"""
    global agent, last_graph_update
    
    print("🚀 Starting Enhanced Neo4j LangGraph FastAPI App...")
    print("=" * 60)
    
    # Check MCP server first
    print("🔍 Checking MCP server connection...")
    mcp_server_ok, mcp_health = check_mcp_server()
    
    if mcp_server_ok:
        print("✅ MCP server is running and accessible")
        if mcp_health:
            neo4j_status = mcp_health.get("neo4j", {}).get("status", "unknown")
            websocket_count = mcp_health.get("websockets", {}).get("active_connections", 0)
            print(f"   Neo4j Status: {neo4j_status}")
            print(f"   WebSocket Connections: {websocket_count}")
    else:
        print("❌ Cannot connect to MCP server!")
        print(f"❌ Please make sure the MCP server is running on port {MCP_SERVER_PORT}")
        print("❌ App will start but functionality will be limited")
    
    # Build Enhanced LangGraph agent
    try:
        print("🔨 Building enhanced LangGraph agent...")
        agent = build_agent()
        print("✅ Enhanced LangGraph agent built successfully")
        
        # Test the agent with a sample query
        print("🧪 Testing enhanced agent with sample query...")
        try:
            test_state = AgentState(
                question="How many nodes are in the graph?",
                session_id="startup_test"
            )
            test_result = await agent.ainvoke(test_state)
            print("✅ Enhanced agent test successful")
        except Exception as e:
            print(f"⚠️ Enhanced agent test failed: {e}")
        
        last_graph_update = datetime.now()
        
        print(f"🌐 Enhanced FastAPI app ready on port {APP_PORT}")
        print("=" * 60)
        print("📋 Available endpoints:")
        print("   • GET  /health - Health check with MCP server status")
        print("   • POST /chat - Chat with enhanced agent (real-time NVL updates)")
        print("   • GET  /agent-info - Enhanced agent information")
        print("   • GET  /graph - Real graph data from MCP server")
        print("   • GET  /stats - Database statistics from MCP server")
        print("   • WS   /ws - WebSocket for UI real-time updates")
        print("   • GET  /ui - Enhanced UI interface")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Failed to build enhanced agent: {e}")
        agent = None

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    print("🛑 Shutting down Enhanced FastAPI App...")
    print("✅ Shutdown complete")

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint with enhanced information"""
    return {
        "service": "Enhanced Neo4j LangGraph Agent API",
        "version": "4.0.0",
        "description": "FastAPI server with enhanced LangGraph agent and real-time Neo4j NVL support",
        "architecture": "FastAPI + Enhanced LangGraph + MCP Server + Real-time NVL",
        "features": [
            "Real-time Neo4j graph visualization",
            "Enhanced LangGraph agent with NVL integration",
            "WebSocket support for live updates",
            "Comprehensive error handling",
            "Detailed operation summaries",
            "Real-time processing feedback"
        ],
        "endpoints": {
            "chat": "/chat - Chat with the enhanced agent",
            "health": "/health - System health check",
            "agent_info": "/agent-info - Enhanced agent information",
            "graph": "/graph - Real graph data",
            "stats": "/stats - Database statistics",
            "ui": "/ui - Enhanced UI interface",
            "ws": "/ws - WebSocket for real-time updates"
        },
        "status": "ready" if agent else "initializing",
        "mcp_server": {
            "port": MCP_SERVER_PORT,
            "status": "connected" if check_mcp_server()[0] else "disconnected"
        },
        "websockets": {
            "active_connections": len(active_websockets)
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint"""
    mcp_server_ok, mcp_health = check_mcp_server()
    
    components = {
        "agent": {
            "status": "ready" if agent else "not_initialized",
            "type": "Enhanced LangGraph Agent",
            "version": "4.0.0"
        },
        "mcp_server": {
            "status": "healthy" if mcp_server_ok else "disconnected",
            "port": MCP_SERVER_PORT,
            "health_data": mcp_health
        },
        "websockets": {
            "active_connections": len(active_websockets),
            "endpoint": "/ws"
        }
    }
    
    overall_status = "healthy" if agent and mcp_server_ok else "degraded"
    
    return HealthResponse(
        status=overall_status,
        components=components,
        agent_ready=agent is not None,
        mcp_server_ready=mcp_server_ok,
        websocket_connections=len(active_websockets)
    )

@app.get("/graph")
async def get_graph_data(limit: int = 100):
    """Get real graph data from MCP server"""
    graph_data = await get_real_graph_data_from_mcp(limit)
    return graph_data

@app.get("/stats")
async def get_database_stats():
    """Get database statistics from MCP server"""
    stats = await get_database_stats_from_mcp()
    return stats

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Enhanced chat endpoint with real-time NVL updates"""
    if agent is None:
        raise HTTPException(
            status_code=503, 
            detail="Enhanced agent not initialized. Check server logs for errors."
        )
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info(f"Processing question with enhanced agent: {request.question}")
        
        # Create enhanced agent state
        state = AgentState(
            question=request.question,
            session_id=session_id
        )
        
        # Run the enhanced agent and measure time
        start_time = time.time()
        result = await agent.ainvoke(state)
        processing_time = time.time() - start_time
        
        logger.info(f"Enhanced agent completed in {processing_time:.2f}s - Tool: {result.get('tool')}")
        
        # Get real-time graph data if this was a write operation
        graph_data = None
        operation_summary = None
        
        if result.get("tool") == "write_neo4j_cypher" and result.get("formatted_answer", "").find("✅") != -1:
            try:
                # Get updated graph data from MCP server
                graph_data = await get_real_graph_data_from_mcp(100)
                operation_summary = await get_database_stats_from_mcp()
                
                # Schedule broadcast to UI clients
                background_tasks.add_task(
                    broadcast_to_ui_clients,
                    result.get("operation_type", "update"),
                    {
                        "question": request.question,
                        "tool": result.get("tool"),
                        "query": result.get("query"),
                        "graph_data": graph_data,
                        "operation_summary": operation_summary
                    }
                )
                
            except Exception as e:
                logger.warning(f"Could not get real-time graph data: {e}")
        
        # Create enhanced response
        response = ChatResponse(
            # Standard fields for backward compatibility
            trace=result.get("trace", ""),
            tool=result.get("tool", ""),
            query=result.get("query", ""),
            answer=result.get("formatted_answer", "No answer generated"),
            session_id=session_id,
            success=True,
            
            # Enhanced fields for real-time NVL
            intent=result.get("intent", ""),
            operation_type=result.get("operation_type", ""),
            raw_response=result.get("raw_response", {}),
            processing_time=processing_time,
            nodes_affected=result.get("affected_nodes", 0),
            relationships_affected=result.get("affected_relationships", 0),
            graph_changes=result.get("graph_changes", {}),
            
            # Real-time graph data
            graph_data=graph_data,
            operation_summary=operation_summary
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Enhanced chat endpoint error: {e}")
        return ChatResponse(
            trace=f"Error: {str(e)}",
            tool="",
            query="",
            answer=f"⚠️ Error processing request: {str(e)}",
            session_id=request.session_id or str(uuid.uuid4()),
            success=False,
            error=str(e),
            intent="error",
            processing_time=0.0
        )

@app.get("/agent-info")
async def get_enhanced_agent_info():
    """Get information about the enhanced agent"""
    if agent is None:
        return {"status": "not_initialized", "agent": None}
    
    return {
        "status": "ready",
        "agent": {
            "type": "Enhanced LangGraph Agent",
            "version": "4.0.0",
            "features": [
                "Real-time Neo4j NVL integration",
                "Enhanced response formatting", 
                "Better error handling with visualization support",
                "Intent classification and operation type detection",
                "Detailed operation summaries with graph changes",
                "WebSocket support for live updates",
                "Comprehensive database monitoring"
            ],
            "nodes": [
                "analyze_and_select_enhanced - Enhanced analysis and tool selection",
                "execute_tool_enhanced - Enhanced tool execution with NVL support"
            ],
            "supported_tools": [
                "read_neo4j_cypher - Execute read queries with visualization data",
                "write_neo4j_cypher - Execute write queries with real-time updates",
                "get_neo4j_schema - Get comprehensive database schema"
            ],
            "improvements": [
                "Fixed node and relationship counting",
                "Real-time graph visualization updates", 
                "Enhanced CREATE/DELETE operation feedback",
                "Better query parsing and validation",
                "Improved error messages with context",
                "WebSocket integration for live updates",
                "NVL-compatible data formatting"
            ],
            "integrations": [
                "Neo4j Visualization Library (NVL)",
                "Real-time WebSocket updates",
                "MCP server communication",
                "Enhanced database monitoring"
            ]
        },
        "mcp_server": {
            "connected": check_mcp_server()[0],
            "port": MCP_SERVER_PORT
        },
        "websockets": {
            "active_connections": len(active_websockets)
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time UI updates"""
    await websocket.accept()
    active_websockets.add(websocket)
    
    try:
        # Send initial state
        initial_graph_data = await get_real_graph_data_from_mcp(100)
        initial_stats = await get_database_stats_from_mcp()
        
        initial_message = GraphUpdateMessage(
            type="initial_state",
            data={
                "graph_data": initial_graph_data,
                "stats": initial_stats,
                "connection_id": str(uuid.uuid4())
            },
            timestamp=datetime.now().isoformat()
        )
        
        await websocket.send_text(initial_message.json())
        logger.info("Sent initial state to UI WebSocket client")
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages from UI client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "request_update":
                    # Send current state
                    current_graph_data = await get_real_graph_data_from_mcp(100)
                    current_stats = await get_database_stats_from_mcp()
                    
                    update_message = GraphUpdateMessage(
                        type="requested_update",
                        data={
                            "graph_data": current_graph_data,
                            "stats": current_stats
                        },
                        timestamp=datetime.now().isoformat()
                    )
                    
                    await websocket.send_text(update_message.json())
                elif message.get("type") == "ping":
                    # Respond to ping
                    pong_message = GraphUpdateMessage(
                        type="pong",
                        data={"status": "alive"},
                        timestamp=datetime.now().isoformat()
                    )
                    await websocket.send_text(pong_message.json())
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"UI WebSocket error: {e}")
                break
                
    finally:
        active_websockets.discard(websocket)
        logger.info("UI WebSocket client disconnected")

@app.get("/ui", response_class=HTMLResponse)
async def enhanced_ui_interface():
    """Enhanced UI interface with embedded React/TypeScript NVL component"""
    return HTMLResponse(content=f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Neo4j Chatbot with Real-time NVL</title>
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://unpkg.com/neo4j-nvl@latest/dist/index.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }}
        
        .app-container {{
            display: flex;
            height: 100vh;
        }}
        
        .chat-panel {{
            width: 40%;
            background: white;
            border-right: 2px solid #e0e0e0;
            display: flex;
            flex-direction: column;
        }}
        
        .graph-panel {{
            width: 60%;
            background: white;
            display: flex;
            flex-direction: column;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            text-align: center;
        }}
        
        .chat-messages {{
            flex: 1;
            padding: 1rem;
            overflow-y: auto;
            max-height: 60vh;
        }}
        
        .message {{
            margin: 0.5rem 0;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .user-message {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: 2rem;
        }}
        
        .assistant-message {{
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            margin-right: 2rem;
        }}
        
        .chat-input {{
            padding: 1rem;
            border-top: 1px solid #e0e0e0;
        }}
        
        .input-group {{
            display: flex;
            gap: 0.5rem;
        }}
        
        .input-field {{
            flex: 1;
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 0.25rem;
            font-size: 16px;
        }}
        
        .send-button {{
            background: #667eea;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            cursor: pointer;
            font-size: 16px;
        }}
        
        .send-button:hover {{
            background: #5a6fd8;
        }}
        
        .send-button:disabled {{
            background: #ccc;
            cursor: not-allowed;
        }}
        
        .graph-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            text-align: center;
        }}
        
        .graph-container {{
            flex: 1;
            position: relative;
        }}
        
        .stats-bar {{
            background: #f8f9fa;
            padding: 0.5rem 1rem;
            border-top: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 14px;
        }}
        
        .status-indicator {{
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-weight: bold;
            font-size: 12px;
        }}
        
        .connected {{
            background: #d4edda;
            color: #155724;
        }}
        
        .disconnected {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .tool-badge {{
            background: #667eea;
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 12px;
            margin: 0.25rem 0;
        }}
        
        .query-display {{
            background: #1e1e1e;
            color: #f8f8f2;
            padding: 0.5rem;
            border-radius: 0.25rem;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            margin: 0.25rem 0;
            overflow-x: auto;
        }}
        
        .examples {{
            padding: 1rem;
            border-top: 1px solid #e0e0e0;
            background: #f8f9fa;
        }}
        
        .example-button {{
            background: #28a745;
            color: white;
            border: none;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            cursor: pointer;
            font-size: 12px;
            margin: 0.25rem;
        }}
        
        .example-button:hover {{
            background: #218838;
        }}
    </style>
</head>
<body>
    <div id="root"></div>

    <script type="text/babel">
        const {{ useState, useEffect, useRef }} = React;
        
        function EnhancedNeo4jChatbot() {{
            const [messages, setMessages] = useState([]);
            const [inputValue, setInputValue] = useState('');
            const [isLoading, setIsLoading] = useState(false);
            const [isConnected, setIsConnected] = useState(false);
            const [graphData, setGraphData] = useState(null);
            const [stats, setStats] = useState({{}});
            const [nvl, setNvl] = useState(null);
            
            const websocketRef = useRef(null);
            const graphContainerRef = useRef(null);
            
            // Example queries
            const examples = [
                "How many nodes are in the graph?",
                "Show me the database schema",
                "Create a Person named Alice with age 30",
                "Create a Company called TechCorp",
                "Connect Alice to TechCorp as an employee",
                "List all Person nodes",
                "Delete all TestNode nodes"
            ];
            
            // Initialize WebSocket connection
            useEffect(() => {{
                const connectWebSocket = () => {{
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${{protocol}}//${{window.location.host}}/ws`;
                    
                    websocketRef.current = new WebSocket(wsUrl);
                    
                    websocketRef.current.onopen = () => {{
                        console.log('✅ WebSocket connected');
                        setIsConnected(true);
                    }};
                    
                    websocketRef.current.onmessage = (event) => {{
                        try {{
                            const message = JSON.parse(event.data);
                            console.log('📨 WebSocket message:', message.type);
                            
                            if (message.type === 'graph_update' || message.type === 'initial_state' || message.type === 'requested_update') {{
                                handleGraphUpdate(message.data);
                            }}
                        }} catch (error) {{
                            console.error('❌ WebSocket message parse error:', error);
                        }}
                    }};
                    
                    websocketRef.current.onclose = () => {{
                        console.log('🔌 WebSocket disconnected');
                        setIsConnected(false);
                        // Attempt to reconnect after 3 seconds
                        setTimeout(connectWebSocket, 3000);
                    }};
                    
                    websocketRef.current.onerror = (error) => {{
                        console.error('❌ WebSocket error:', error);
                        setIsConnected(false);
                    }};
                }};
                
                connectWebSocket();
                
                return () => {{
                    if (websocketRef.current) {{
                        websocketRef.current.close();
                    }}
                }};
            }}, []);
            
            // Initialize NVL
            useEffect(() => {{
                if (graphData && graphContainerRef.current && !nvl) {{
                    initializeNVL();
                }}
            }}, [graphData, nvl]);
            
            const handleGraphUpdate = (data) => {{
                if (data.graph_data) {{
                    setGraphData(data.graph_data);
                }}
                if (data.stats) {{
                    setStats(data.stats);
                }}
                
                // Update NVL if it exists
                if (nvl && data.graph_data) {{
                    updateNVL(data.graph_data);
                }}
            }};
            
            const initializeNVL = () => {{
                if (!graphData || !graphContainerRef.current) return;
                
                try {{
                    const nvlInstance = new NVL('graph-nvl-container', graphData.nodes || [], graphData.relationships || [], {{
                        instanceId: 'enhanced-chatbot-graph',
                        initialZoom: 1.0,
                        allowDynamicMinZoom: true,
                        showPropertiesOnHover: true,
                        showPropertiesOnClick: true,
                        nodeColorScheme: 'category20',
                        relationshipColorScheme: 'dark',
                        layout: {{
                            algorithm: 'forceDirected',
                            incrementalLayout: true,
                            animate: true
                        }},
                        styling: {{
                            nodeSize: 25,
                            relationshipWidth: 3,
                            fontSize: 11
                        }},
                        interaction: {{
                            dragEnabled: true,
                            zoomEnabled: true,
                            hoverEnabled: true,
                            selectEnabled: true
                        }}
                    }});
                    
                    setNvl(nvlInstance);
                    console.log('✅ NVL initialized');
                }} catch (error) {{
                    console.error('❌ NVL initialization failed:', error);
                }}
            }};
            
            const updateNVL = (newGraphData) => {{
                if (nvl && newGraphData) {{
                    try {{
                        nvl.updateGraph(newGraphData.nodes || [], newGraphData.relationships || []);
                        console.log('🔄 NVL updated');
                    }} catch (error) {{
                        console.error('❌ NVL update failed:', error);
                        // Reinitialize if update fails
                        setNvl(null);
                        setTimeout(() => initializeNVL(), 1000);
                    }}
                }}
            }};
            
            const sendMessage = async (question) => {{
                if (!question.trim()) return;
                
                setIsLoading(true);
                
                // Add user message
                const userMessage = {{
                    role: 'user',
                    content: question,
                    timestamp: new Date().toISOString()
                }};
                setMessages(prev => [...prev, userMessage]);
                
                try {{
                    const response = await fetch('/chat', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{
                            question: question,
                            session_id: 'ui_session'
                        }})
                    }});
                    
                    const result = await response.json();
                    
                    // Add assistant message
                    const assistantMessage = {{
                        role: 'assistant',
                        content: result,
                        timestamp: new Date().toISOString()
                    }};
                    setMessages(prev => [...prev, assistantMessage]);
                    
                    // Update graph data if available
                    if (result.graph_data) {{
                        setGraphData(result.graph_data);
                        updateNVL(result.graph_data);
                    }}
                    
                }} catch (error) {{
                    console.error('❌ Send message failed:', error);
                    const errorMessage = {{
                        role: 'assistant',
                        content: {{
                            success: false,
                            answer: `❌ Error: ${{error.message}}`,
                            error: error.message
                        }},
                        timestamp: new Date().toISOString()
                    }};
                    setMessages(prev => [...prev, errorMessage]);
                }} finally {{
                    setIsLoading(false);
                }}
            }};
            
            const handleSubmit = (e) => {{
                e.preventDefault();
                if (inputValue.trim() && !isLoading) {{
                    sendMessage(inputValue);
                    setInputValue('');
                }}
            }};
            
            const handleExampleClick = (example) => {{
                setInputValue(example);
                sendMessage(example);
            }};
            
            return (
                <div className="app-container">
                    {{/* Chat Panel */}}
                    <div className="chat-panel">
                        <div className="header">
                            <h2>🧠 Neo4j Enhanced Chatbot</h2>
                            <p>Real-time graph visualization with NVL</p>
                        </div>
                        
                        <div className="chat-messages">
                            {{messages.map((message, index) => (
                                <div key={{index}} className={{`message ${{message.role}}-message`}}>
                                    {{message.role === 'user' ? (
                                        <div>
                                            <strong>🧑 You:</strong> {{message.content}}
                                        </div>
                                    ) : (
                                        <div>
                                            <strong>🤖 Neo4j Agent:</strong>
                                            {{message.content.tool && (
                                                <div className="tool-badge">
                                                    🔧 {{message.content.tool}}
                                                </div>
                                            )}}
                                            {{message.content.query && (
                                                <div className="query-display">
                                                    {{message.content.query}}
                                                </div>
                                            )}}
                                            <div>{{message.content.answer}}</div>
                                            {{message.content.processing_time && (
                                                <small>⏱️ {{message.content.processing_time.toFixed(2)}}s</small>
                                            )}}
                                        </div>
                                    )}}
                                </div>
                            ))}}
                        </div>
                        
                        <div className="chat-input">
                            <form onSubmit={{handleSubmit}}>
                                <div className="input-group">
                                    <input
                                        type="text"
                                        className="input-field"
                                        placeholder="Ask about your Neo4j database..."
                                        value={{inputValue}}
                                        onChange={{(e) => setInputValue(e.target.value)}}
                                        disabled={{isLoading}}
                                    />
                                    <button
                                        type="submit"
                                        className="send-button"
                                        disabled={{isLoading || !inputValue.trim()}}
                                    >
                                        {{isLoading ? '⏳' : '🚀'}} Send
                                    </button>
                                </div>
                            </form>
                        </div>
                        
                        <div className="examples">
                            <strong>💡 Try these examples:</strong>
                            <div>
                                {{examples.map((example, index) => (
                                    <button
                                        key={{index}}
                                        className="example-button"
                                        onClick={{() => handleExampleClick(example)}}
                                        disabled={{isLoading}}
                                    >
                                        {{example}}
                                    </button>
                                ))}}
                            </div>
                        </div>
                    </div>
                    
                    {{/* Graph Panel */}}
                    <div className="graph-panel">
                        <div className="graph-header">
                            <h2>🎨 Real Neo4j Graph (NVL)</h2>
                            <p>Live visualization of your database</p>
                        </div>
                        
                        <div className="graph-container" ref={{graphContainerRef}}>
                            <div id="graph-nvl-container" style={{{{width: '100%', height: '100%'}}}}>
                                {{!graphData || !graphData.nodes || graphData.nodes.length === 0 ? (
                                    <div style={{{{
                                        display: 'flex',
                                        justifyContent: 'center',
                                        alignItems: 'center',
                                        height: '100%',
                                        fontSize: '18px',
                                        color: '#666',
                                        flexDirection: 'column',
                                        gap: '1rem'
                                    }}}}>
                                        <div>📝 No data in Neo4j database</div>
                                        <div>Create some nodes to see the graph!</div>
                                    </div>
                                ) : null}}
                            </div>
                        </div>
                        
                        <div className="stats-bar">
                            <div>
                                <strong>Nodes:</strong> {{stats.nodes || 0}} | 
                                <strong> Relationships:</strong> {{stats.relationships || 0}}
                            </div>
                            <div className={{`status-indicator ${{isConnected ? 'connected' : 'disconnected'}}`}}>
                                {{isConnected ? '🟢 Connected' : '🔴 Disconnected'}}
                            </div>
                        </div>
                    </div>
                </div>
            );
        }}
        
        ReactDOM.render(<EnhancedNeo4jChatbot />, document.getElementById('root'));
    </script>
</body>
</html>
    """, media_type="text/html")

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """Main function to run the enhanced FastAPI application"""
    print("=" * 70)
    print("🚀 ENHANCED NEO4J LANGGRAPH FASTAPI APP")
    print("=" * 70)
    print("🏗️  Architecture: Enhanced FastAPI + LangGraph + MCP Server + Real-time NVL")
    print("🔧 Configuration:")
    print(f"   📍 App Port: {APP_PORT}")
    print(f"   🛠️ MCP Server Port: {MCP_SERVER_PORT}")
    print("=" * 70)
    print("✨ Enhanced Features:")
    print("   🎯 Real-time Neo4j graph visualization with NVL")
    print("   📊 Enhanced response formatting with graph changes")
    print("   🔄 WebSocket support for live updates")
    print("   📈 Comprehensive database monitoring")
    print("   🤖 Enhanced LangGraph agent integration")
    print("   🌐 React/TypeScript UI components")
    print("=" * 70)
    print("📋 Prerequisites:")
    print("   1. Enhanced MCP server running on port 8000")
    print("   2. Neo4j database accessible")
    print("   3. Cortex API key configured")
    print("=" * 70)
    
    # Check prerequisites
    print("🔍 Checking prerequisites...")
    
    # Check MCP server
    mcp_server_ok, _ = check_mcp_server()
    if mcp_server_ok:
        print("✅ MCP server is accessible")
    else:
        print("❌ MCP server not accessible - make sure it's running on port 8000")
    
    print("=" * 70)
    print("🚀 Starting enhanced FastAPI application...")
    
    try:
        import uvicorn
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=APP_PORT,
            log_level="info",
            reload=False
        )
    except Exception as e:
        print(f"❌ Failed to start enhanced app: {e}")

if __name__ == "__main__":
    main()
