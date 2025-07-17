# run_app.py
"""
Startup script for Neo4j LangGraph Supervisor Application
"""

import subprocess
import sys
import time
import os
import asyncio
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    print("🔍 Checking requirements...")
    
    try:
        import streamlit
        import langgraph
        import langchain_core
        import neo4j
        import mcp
        import fastmcp
        import pydantic
        import requests
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("📦 Please install requirements: pip install -r requirements.txt")
        return False

def check_files():
    """Check if all required files exist"""
    print("📁 Checking required files...")
    
    required_files = [
        "langgraph_supervisor.py",
        "mcpserver.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files found")
    return True

def install_requirements():
    """Install requirements if needed"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def run_application():
    """Run the main Streamlit application"""
    print("🚀 Starting Neo4j LangGraph Supervisor Application...")
    print("🌐 The application will open in your browser automatically")
    print("🛑 Press Ctrl+C to stop the application")
    
    try:
        # Run Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "langgraph_supervisor.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    """Main startup function"""
    print("=" * 60)
    print("🧠 Neo4j LangGraph Supervisor Application")
    print("=" * 60)
    
    # Check if files exist
    if not check_files():
        print("\n❌ Setup incomplete. Please ensure all files are in the current directory.")
        return
    
    # Check requirements
    if not check_requirements():
        user_input = input("\n📦 Install requirements now? (y/n): ").lower().strip()
        if user_input == 'y':
            if not install_requirements():
                return
            print("✅ Requirements installed. Please restart the script.")
            return
        else:
            print("❌ Cannot proceed without required packages.")
            return
    
    print("\n🎯 System Status:")
    print("  ✅ Requirements: OK")
    print("  ✅ Files: OK")
    print("  🚀 Ready to launch!")
    
    # Small delay for user to read
    time.sleep(2)
    
    # Run the application
    run_application()

if __name__ == "__main__":
    main()
