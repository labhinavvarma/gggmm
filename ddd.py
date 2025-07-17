
#!/usr/bin/env python3
"""
Fixed startup script with better timing and health checks
"""

import subprocess
import sys
import time
import threading
import signal
import os
import requests

def run_mcp_server():
    """Run the MCP server"""
    print("🚀 Starting MCP Server on port 8000...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "hardcoded_mcpserver:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except Exception as e:
        print(f"❌ MCP Server failed: {e}")

def run_complete_app():
    """Run the complete app"""
    print("🚀 Starting Complete App on port 8081...")
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

def run_streamlit():
    """Run Streamlit UI"""
    print("🚀 Starting Streamlit UI on port 8501...")
    
    # Find UI file
    ui_files = ["complete_ui.py", "no_timeout_ui.py", "ui.py"]
    ui_file = None
    
    for file in ui_files:
        if os.path.exists(file):
            ui_file = file
            break
    
    if not ui_file:
        print("❌ No UI file found!")
        return
    
    subprocess.run([
        sys.executable, "-m", "streamlit", 
        "run", ui_file, 
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

def wait_for_service(name, url, max_attempts=30, delay=2):
    """Wait for a service to become available"""
    print(f"⏳ Waiting for {name} to be ready...")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✅ {name} is ready!")
                return True
        except:
            pass
        
        print(f"   Attempt {attempt + 1}/{max_attempts} - {name} not ready yet...")
        time.sleep(delay)
    
    print(f"⚠️  {name} didn't respond in time, but continuing...")
    return False

def test_services_when_ready():
    """Test services after they're ready"""
    print("\n🧪 Testing services after startup...")
    
    services = [
        ("MCP Server", "http://localhost:8000/health"),
        ("Complete App", "http://localhost:8081/health")
    ]
    
    all_healthy = True
    
    for name, url in services:
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                print(f"✅ {name}: Online and healthy")
                if name == "Complete App":
                    data = response.json()
                    services_status = data.get('services', {})
                    print(f"   - Agent: {services_status.get('complete_agent', 'unknown')}")
                    print(f"   - MCP Server: {services_status.get('mcp_server', 'unknown')}")
            else:
                print(f"🟡 {name}: Responding but with status {response.status_code}")
                all_healthy = False
        except Exception as e:
            print(f"🔴 {name}: Not responding - {str(e)[:50]}...")
            all_healthy = False
    
    return all_healthy

def run_simple_test():
    """Run a simple test query"""
    print("\n🎯 Testing complete workflow...")
    
    try:
        test_payload = {
            "question": "How many nodes are in the graph?",
            "session_id": "startup_test"
        }
        
        response = requests.post(
            "http://localhost:8081/chat",
            json=test_payload,
            timeout=60  # Longer timeout for complete workflow
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Complete workflow test successful!")
            print(f"   - Intent: {result.get('intent', 'N/A')}")
            print(f"   - Tool: {result.get('tool', 'N/A')}")
            print(f"   - Success: {result.get('success', False)}")
            
            if result.get('answer'):
                answer_preview = str(result['answer'])[:60]
                print(f"   - Answer: {answer_preview}...")
            
            return True
        else:
            print(f"❌ Workflow test failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"⚠️  Workflow test error: {str(e)}")
        return False

def signal_handler(signum, frame):
    print("\n🛑 Shutting down all services...")
    sys.exit(0)

def check_required_files():
    """Check if required files exist"""
    print("🔍 Checking required files...")
    
    required_files = [
        "hardcoded_mcpserver.py",
        "complete_langgraph_agent.py",
        "complete_app.py"
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
    
    # Find UI file
    ui_files = ["complete_ui.py", "no_timeout_ui.py", "ui.py"]
    ui_file = None
    
    for file in ui_files:
        if os.path.exists(file):
            ui_file = file
            print(f"✅ UI: {file}")
            break
    
    if not ui_file:
        print("❌ No UI file found")
        return False
    
    return True

def main():
    print("🧠 Fixed Complete LangGraph Startup")
    print("=" * 50)
    
    # Check files
    if not check_required_files():
        print("\n❌ Required files missing. Please ensure all files are present.")
        return
    
    print("\n🔧 Fixed startup process:")
    print("   - Longer startup delays")
    print("   - Better health checking")
    print("   - More patient service testing")
    print("   - Robust error handling")
    
    print("\n🌐 Services will run on:")
    print("   - MCP Server: http://localhost:8000")
    print("   - Complete App: http://localhost:8081")
    print("   - Streamlit UI: http://localhost:8501")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("\n🚀 Starting services with improved timing...")
        
        # Start MCP server
        print("\n1️⃣ Starting MCP Server...")
        mcp_thread = threading.Thread(target=run_mcp_server, daemon=True)
        mcp_thread.start()
        
        # Give MCP server time to start
        wait_for_service("MCP Server", "http://localhost:8000/health", max_attempts=15, delay=2)
        
        # Start complete app
        print("\n2️⃣ Starting Complete LangGraph App...")
        app_thread = threading.Thread(target=run_complete_app, daemon=True)
        app_thread.start()
        
        # Give complete app time to start (it needs more time to load the agent)
        wait_for_service("Complete App", "http://localhost:8081/health", max_attempts=20, delay=3)
        
        # Test services
        print("\n3️⃣ Testing system health...")
        services_healthy = test_services_when_ready()
        
        # Run workflow test
        print("\n4️⃣ Testing complete workflow...")
        workflow_working = run_simple_test()
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 STARTUP SUMMARY:")
        print("=" * 50)
        
        if services_healthy and workflow_working:
            print("🎉 ALL SYSTEMS OPERATIONAL!")
            print("✅ Services: Healthy")
            print("✅ Workflow: Working")
        elif services_healthy:
            print("🟡 SERVICES READY, WORKFLOW NEEDS ATTENTION")
            print("✅ Services: Healthy")
            print("⚠️  Workflow: Issues detected")
        else:
            print("⚠️  SERVICES STARTING, MANUAL TESTING RECOMMENDED")
            print("⚠️  Services: Some issues detected")
            print("💡 Try manual testing in the UI")
        
        print("\n🌐 ACCESS POINTS:")
        print(f"   • Streamlit UI: http://localhost:8501")
        print(f"   • API Docs: http://localhost:8081/docs")
        print(f"   • Health Check: http://localhost:8081/health")
        print(f"   • MCP Health: http://localhost:8000/health")
        
        print("\n💡 MANUAL TESTING:")
        print("   • Open the Streamlit UI")
        print("   • Try: 'How many nodes are in the graph?'")
        print("   • Check the workflow visualization")
        
        print(f"\n🚀 Starting Streamlit UI...")
        print("   (This will block - press Ctrl+C to stop everything)")
        
        # Start Streamlit (this blocks)
        run_streamlit()
        
    except KeyboardInterrupt:
        print("\n🛑 Received shutdown signal")
    except Exception as e:
        print(f"❌ Error starting services: {e}")
    finally:
        print("👋 All services stopped")

if __name__ == "__main__":
    main()
