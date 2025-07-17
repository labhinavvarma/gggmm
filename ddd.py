#!/usr/bin/env python3
"""
Startup script for the Neo4j LangGraph MCP Agent
This script will start both the MCP server and the main FastAPI app.
"""

import subprocess
import sys
import time
import threading
import signal
import os

# Try to import config, fallback to hardcoded values
try:
    from config import SERVER_CONFIG
    MCP_PORT = SERVER_CONFIG["mcp_port"]
    APP_PORT = SERVER_CONFIG["app_port"]
    UI_PORT = SERVER_CONFIG["ui_port"]
    HOST = SERVER_CONFIG["host"]
except ImportError:
    MCP_PORT = 8000
    APP_PORT = 8081
    UI_PORT = 8501
    HOST = "0.0.0.0"

def run_mcp_server():
    """Run the MCP server on configured port"""
    print(f"🚀 Starting MCP Server on port {MCP_PORT}...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "mcpserver:app", 
            "--host", HOST, 
            "--port", str(MCP_PORT), 
            "--reload"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ MCP Server failed to start: {e}")
        sys.exit(1)

def run_main_app():
    """Run the main FastAPI app on configured port"""
    print(f"🚀 Starting Main App on port {APP_PORT}...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", HOST, 
            "--port", str(APP_PORT), 
            "--reload"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Main App failed to start: {e}")
        sys.exit(1)

def run_streamlit():
    """Run the Streamlit UI on configured port"""
    print(f"🚀 Starting Streamlit UI on port {UI_PORT}...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", 
            "run", "ui.py", 
            "--server.port", str(UI_PORT),
            "--server.address", HOST
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Streamlit failed to start: {e}")
        sys.exit(1)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutting down all services...")
    sys.exit(0)

if __name__ == "__main__":
    print("🧠 Neo4j LangGraph MCP Agent - Startup Script")
    print("=" * 50)
    
    # Check if required files exist
    required_files = ["mcpserver.py", "app.py", "ui.py", "langgraph_agent.py"]
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Required file {file} not found!")
            sys.exit(1)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("✅ All required files found")
    print("\nStarting services...")
    print(f"- MCP Server will run on http://localhost:{MCP_PORT}")
    print(f"- Main App will run on http://localhost:{APP_PORT}")
    print(f"- Streamlit UI will run on http://localhost:{UI_PORT}")
    print("\nPress Ctrl+C to stop all services\n")
    
    try:
        # Start MCP server in a separate thread
        mcp_thread = threading.Thread(target=run_mcp_server, daemon=True)
        mcp_thread.start()
        
        # Give MCP server time to start
        time.sleep(3)
        
        # Start main app in a separate thread
        app_thread = threading.Thread(target=run_main_app, daemon=True)
        app_thread.start()
        
        # Give main app time to start
        time.sleep(3)
        
        # Start Streamlit (this will block)
        run_streamlit()
        
    except KeyboardInterrupt:
        print("\n🛑 Received shutdown signal")
    except Exception as e:
        print(f"❌ Error starting services: {e}")
        sys.exit(1)
    finally:
        print("👋 All services stopped")
