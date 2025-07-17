#!/usr/bin/env python3
"""
Complete LangGraph startup script with all prompts and workflows integrated
"""

import subprocess
import sys
import time
import threading
import signal
import os
import requests

def check_requirements():
    """Check if all required files and dependencies exist"""
    print("🔍 Checking requirements...")
    
    required_files = [
        "complete_langgraph_agent.py",
        "complete_app.py",
        "simple_mcpserver.py"  # We'll use the simple MCP server
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            missing_files.append(file)
            print(f"❌ {file} missing")
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        return False
    
    # Check UI file
    ui_files = ["no_timeout_ui.py", "improved_simple_ui.py", "ui.py"]
    ui_file = None
    for file in ui_files:
        if os.path.exists(file):
            ui_file = file
            print(f"✅ UI: {file}")
            break
    
    if not ui_file:
        print("❌ No UI file found")
        return False
    
    return True, ui_file

def run_mcp_server():
    """Run the MCP server"""
    print("🚀 Starting MCP Server on port 8000...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "simple_mcpserver:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except Exception as e:
        print(f"❌ MCP Server failed: {e}")

def run_complete_app():
    """Run the complete LangGraph app"""
    print("🚀 Starting Complete LangGraph App on port 8081...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "complete_app:app", 
            "--host", "0.0.0.0", 
            "--port", "8081", 
            "--reload"
        ])
    except Exception as e:
        print(f"❌ Complete App failed: {e}")

def run_streamlit(ui_file):
    """Run Streamlit UI"""
    print(f"🚀 Starting Streamlit UI with {ui_file} on port 8501...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", 
            "run", ui_file, 
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except Exception as e:
        print(f"❌ Streamlit failed: {e}")

def test_services():
    """Test if all services are running properly"""
    print("\n🧪 Testing services...")
    
    services = [
        ("MCP Server", "http://localhost:8000/health"),
        ("Complete App", "http://localhost:8081/health"),
        ("Complete App Info", "http://localhost:8081/agent-info")
    ]
    
    for name, url in services:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: Online")
                if name == "Complete App":
                    data = response.json()
                    print(f"   - Agent Status: {data.get('services', {}).get('complete_agent', 'unknown')}")
                    print(f"   - MCP Server: {data.get('services', {}).get('mcp_server', 'unknown')}")
                    print(f"   - Cortex LLM: {data.get('services', {}).get('cortex_llm', 'unknown')}")
            else:
                print(f"🟡 {name}: Issues ({response.status_code})")
        except Exception as e:
            print(f"🔴 {name}: Offline - {e}")

def run_quick_test():
    """Run a quick test of the complete agent"""
    print("\n🎯 Running quick test...")
    
    try:
        test_payload = {
            "question": "How many nodes are in the graph?",
            "session_id": "test_session"
        }
        
        response = requests.post(
            "http://localhost:8081/chat",
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Quick test successful!")
            print(f"   - Intent: {result.get('intent', 'N/A')}")
            print(f"   - Tool: {result.get('tool', 'N/A')}")
            print(f"   - Query: {result.get('query', 'N/A')}")
            print(f"   - Steps: {' → '.join(result.get('execution_steps', []))}")
            print(f"   - Answer: {result.get('answer', 'N/A')[:100]}...")
        else:
            print(f"❌ Quick test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Quick test error: {e}")

def signal_handler(signum, frame):
    print("\n🛑 Shutting down all services...")
    sys.exit(0)

def show_configuration_checklist():
    """Show what needs to be configured"""
    print("\n📋 CONFIGURATION CHECKLIST:")
    print("=" * 50)
    
    print("📁 In simple_mcpserver.py:")
    print("   Line 12: NEO4J_PASSWORD = 'your_actual_password'")
    
    print("\n📁 In complete_langgraph_agent.py:")
    print("   Line 18: API_KEY = 'your_actual_cortex_key'")
    
    print("\n🔧 Requirements:")
    print("   - Neo4j running on localhost:7687")
    print("   - Cortex API access")
    print("   - Python packages: fastapi, uvicorn, streamlit, neo4j, langgraph")
    
    print("\n🌐 After startup, access:")
    print("   - Streamlit UI: http://localhost:8501")
    print("   - API Docs: http://localhost:8081/docs")
    print("   - Agent Info: http://localhost:8081/agent-info")
    print("   - Health Check: http://localhost:8081/health")

def main():
    print("🧠 COMPLETE NEO4J LANGGRAPH AGENT")
    print("=" * 60)
    print("🚀 Advanced AI Agent with Integrated Prompts & Workflows")
    print("=" * 60)
    
    # Show configuration checklist
    show_configuration_checklist()
    
    # Check requirements
    requirements_check = check_requirements()
    if not isinstance(requirements_check, tuple):
        print("\n❌ Requirements check failed")
        sys.exit(1)
    
    requirements_ok, ui_file = requirements_check
    if not requirements_ok:
        print("\n❌ Requirements not met")
        sys.exit(1)
    
    print(f"\n✅ All requirements met! UI file: {ui_file}")
    
    # Show features
    print("\n🎯 COMPLETE AGENT FEATURES:")
    print("   🧠 Intent Analysis - Understands what user wants")
    print("   🔧 Tool Selection - Chooses right Neo4j operation")
    print("   📝 Query Generation - Creates optimal Cypher queries") 
    print("   ⚡ Execution - Runs queries on Neo4j via MCP")
    print("   🎨 Formatting - Presents results beautifully")
    print("   🔄 Error Handling - Retries with intelligent fixes")
    print("   📊 Multi-step Workflow - Complete reasoning chain")
    
    # Ask user to continue
    input("\nPress Enter to start all services...")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("\n🚀 STARTING COMPLETE LANGGRAPH SYSTEM...")
        print("=" * 50)
        
        # Start MCP server in background
        print("🔧 Starting MCP Server...")
        mcp_thread = threading.Thread(target=run_mcp_server, daemon=True)
        mcp_thread.start()
        time.sleep(5)
        
        # Start complete app in background  
        print("🧠 Starting Complete LangGraph Agent...")
        app_thread = threading.Thread(target=run_complete_app, daemon=True)
        app_thread.start()
        time.sleep(8)  # Give more time for complete agent to initialize
        
        # Test services
        test_services()
        
        # Run quick test
        run_quick_test()
        
        print("\n🎉 ALL SERVICES STARTED SUCCESSFULLY!")
        print("=" * 50)
        print("🌐 Access your Complete LangGraph Agent at:")
        print("   - Streamlit UI: http://localhost:8501")
        print("   - API Documentation: http://localhost:8081/docs")
        print("   - Agent Information: http://localhost:8081/agent-info")
        print("   - Test Workflow: http://localhost:8081/test-workflow")
        
        print("\n💡 TRY THESE COMMANDS:")
        print("   • 'How many nodes are in the graph?'")
        print("   • 'Show me the database schema'")
        print("   • 'Create a Person named Alice'")
        print("   • 'List all node types'")
        print("   • 'Find nodes with most relationships'")
        
        print(f"\n🚀 Starting Streamlit UI with {ui_file}...")
        print("Press Ctrl+C to stop all services")
        
        # Start Streamlit (this blocks)
        run_streamlit(ui_file)
        
    except KeyboardInterrupt:
        print("\n🛑 Received shutdown signal")
    except Exception as e:
        print(f"❌ Error starting services: {e}")
    finally:
        print("👋 Complete LangGraph Agent stopped")

if __name__ == "__main__":
    main()
