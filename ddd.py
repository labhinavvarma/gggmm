"""
Configuration file for Neo4j LangGraph MCP Agent
Modify these settings according to your environment.
"""

# Neo4j Configuration
NEO4J_CONFIG = {
    "uri": "neo4j://localhost:7687",
    "user": "neo4j",#!/usr/bin/env python3
"""
Setup verification script for Neo4j LangGraph MCP Agent
This script checks if all components are configured correctly.
"""

import os
import sys
import requests
import importlib.util

def check_file_exists(filename):
    """Check if a required file exists"""
    exists = os.path.exists(filename)
    status = "✅" if exists else "❌"
    print(f"{status} {filename}")
    return exists

def check_python_package(package_name):
    """Check if a Python package is installed"""
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} (run: pip install {package_name})")
        return False

def check_neo4j_connection():
    """Check Neo4j connection"""
    try:
        # Try to import config
        try:
            from config import NEO4J_CONFIG
            uri = NEO4J_CONFIG["uri"]
            user = NEO4J_CONFIG["user"]
            password = NEO4J_CONFIG["password"]
        except ImportError:
            print("⚠️  No config.py found, using defaults")
            uri = "neo4j://localhost:7687"
            user = "neo4j"
            password = "your_neo4j_password"
        
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record["test"] == 1:
                print("✅ Neo4j connection successful")
                return True
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False

def check_cortex_api():
    """Check Cortex API configuration"""
    try:
        from config import CORTEX_CONFIG
        api_url = CORTEX_CONFIG["api_url"]
        api_key = CORTEX_CONFIG["api_key"]
        
        if api_key == "your_cortex_api_key":
            print("❌ Cortex API key not configured (still using placeholder)")
            return False
        
        print("✅ Cortex API configuration found")
        return True
    except ImportError:
        print("❌ Cortex API not configured (no config.py)")
        return False

def check_ports():
    """Check if required ports are available"""
    ports = [8000, 8081, 8501]
    try:
        from config import SERVER_CONFIG
        ports = [SERVER_CONFIG["mcp_port"], SERVER_CONFIG["app_port"], SERVER_CONFIG["ui_port"]]
    except ImportError:
        pass
    
    available_ports = []
    for port in ports:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            print(f"⚠️  Port {port} is in use (service running)")
            available_ports.append(False)
        except requests.exceptions.ConnectionError:
            print(f"✅ Port {port} is available")
            available_ports.append(True)
        except Exception as e:
            print(f"❓ Port {port} status unknown: {e}")
            available_ports.append(True)
    
    return all(available_ports)

def main():
    print("🔍 Neo4j LangGraph MCP Agent - Setup Verification")
    print("=" * 60)
    
    # Check required files
    print("\n📁 Checking required files:")
    required_files = [
        "app.py",
        "langgraph_agent.py", 
        "mcpserver.py",
        "ui.py",
        "run_app.py",
        "requirements.txt"
    ]
    
    files_ok = all(check_file_exists(f) for f in required_files)
    
    # Check config file
    print("\n⚙️  Checking configuration:")
    config_exists = check_file_exists("config.py")
    if not config_exists:
        print("⚠️  config.py not found - will use default values")
    
    # Check Python packages
    print("\n📦 Checking Python packages:")
    required_packages = [
        "fastapi", "uvicorn", "streamlit", "neo4j", 
        "pydantic", "requests", "langgraph", "langchain_core"
    ]
    
    packages_ok = all(check_python_package(pkg) for pkg in required_packages)
    
    # Check Neo4j connection
    print("\n🗄️  Checking Neo4j connection:")
    neo4j_ok = check_neo4j_connection()
    
    # Check Cortex API
    print("\n🤖 Checking Cortex API:")
    cortex_ok = check_cortex_api()
    
    # Check ports
    print("\n🌐 Checking ports:")
    ports_ok = check_ports()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY:")
    print("=" * 60)
    
    checks = [
        ("Required files", files_ok),
        ("Python packages", packages_ok),
        ("Neo4j connection", neo4j_ok),
        ("Cortex API config", cortex_ok),
        ("Port availability", ports_ok)
    ]
    
    all_good = True
    for check_name, status in checks:
        icon = "✅" if status else "❌"
        print(f"{icon} {check_name}")
        if not status:
            all_good = False
    
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 ALL CHECKS PASSED! You're ready to run the application.")
        print("\n🚀 To start the application:")
        print("   python run_app.py")
    else:
        print("⚠️  SOME CHECKS FAILED. Please fix the issues above.")
        print("\n🔧 Common fixes:")
        print("   • Install packages: pip install -r requirements.txt")
        print("   • Create config.py with your settings")
        print("   • Start Neo4j: sudo systemctl start neo4j")
        print("   • Update API keys in config.py")
    
    print("\n📚 For more help, see the troubleshooting guide.")

if __name__ == "__main__":
    main()
    "password": "your_neo4j_password",  # Change this!
    "database": "neo4j"
}

# Cortex API Configuration
CORTEX_CONFIG = {
    "api_url": "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete",
    "api_key": "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0",  # Change this!
    "model": "llama3.1-70b",
    "app_code": "edagnai",
    "app_id": "edadip"
}

# Server Configuration
SERVER_CONFIG = {
    "mcp_port": 8000,
    "app_port": 8081,
    "ui_port": 8501,
    "host": "0.0.0.0"
}

# Debug Configuration
DEBUG_CONFIG = {
    "enable_debug_logging": True,
    "print_llm_output": True,
    "print_queries": True
}

# Timeouts and Retries
TIMEOUT_CONFIG = {
    "cortex_timeout": 30,  # seconds
    "neo4j_timeout": 10,   # seconds
    "max_retries": 3
}
