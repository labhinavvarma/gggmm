#!/usr/bin/env python3
"""
Neo4j Connection Diagnostic & Fix Tool
This will help you identify and fix the 503 Service Unavailable error
"""

import asyncio
import sys
import subprocess
import time
import requests
from neo4j import AsyncGraphDatabase

# ============================================
# CONFIGURATION - CHANGE THESE TO MATCH YOUR SETUP
# ============================================

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_neo4j_password"  # ⚠️ CHANGE THIS!
NEO4J_DATABASE = "neo4j"

SERVER_URL = "http://localhost:8000"

# ============================================

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(message, color=Colors.GREEN):
    print(f"{color}{message}{Colors.ENDC}")

def print_header(title):
    print_colored(f"\n{'='*60}", Colors.BLUE)
    print_colored(f"{title}", Colors.BOLD)
    print_colored(f"{'='*60}", Colors.BLUE)

async def test_neo4j_connection():
    """Test direct Neo4j connection"""
    print_header("🔍 TESTING NEO4J CONNECTION")
    
    print_colored(f"📍 Testing connection to: {NEO4J_URI}")
    print_colored(f"👤 User: {NEO4J_USER}")
    print_colored(f"🗄️  Database: {NEO4J_DATABASE}")
    print_colored(f"🔑 Password: {'*' * len(NEO4J_PASSWORD)}")
    
    try:
        # Test basic connectivity
        print_colored("\n1. Testing basic connectivity...")
        
        driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            connection_timeout=10
        )
        
        # Test connection
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run("RETURN 1 as test")
            record = await result.single()
            
        if record and record["test"] == 1:
            print_colored("✅ Neo4j connection successful!", Colors.GREEN)
            
            # Test data operations
            print_colored("\n2. Testing data operations...")
            
            async with driver.session(database=NEO4J_DATABASE) as session:
                # Count nodes
                count_result = await session.run("MATCH (n) RETURN count(n) as node_count")
                count_record = await count_result.single()
                node_count = count_record["node_count"] if count_record else 0
                
                print_colored(f"   📊 Found {node_count} nodes in database", Colors.GREEN)
                
                # Test write operation
                write_result = await session.run("CREATE (test:DiagnosticTest {created: datetime()}) RETURN test")
                write_record = await write_result.single()
                
                if write_record:
                    print_colored("   ✅ Write operation successful", Colors.GREEN)
                    
                    # Clean up test node
                    await session.run("MATCH (test:DiagnosticTest) DELETE test")
                    print_colored("   🧹 Test node cleaned up", Colors.GREEN)
                
        await driver.close()
        return True
        
    except Exception as e:
        print_colored(f"❌ Neo4j connection failed: {e}", Colors.RED)
        return False

def check_neo4j_running():
    """Check if Neo4j is running"""
    print_header("🔍 CHECKING NEO4J STATUS")
    
    try:
        # Try to run neo4j status command
        result = subprocess.run(['neo4j', 'status'], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print_colored("✅ Neo4j is running", Colors.GREEN)
            print_colored(f"   Output: {result.stdout.strip()}", Colors.BLUE)
            return True
        else:
            print_colored("❌ Neo4j is not running", Colors.RED)
            print_colored(f"   Error: {result.stderr.strip()}", Colors.RED)
            return False
            
    except subprocess.TimeoutExpired:
        print_colored("⚠️  Neo4j status command timed out", Colors.YELLOW)
        return False
    except FileNotFoundError:
        print_colored("⚠️  Neo4j command not found - is Neo4j installed?", Colors.YELLOW)
        return False
    except Exception as e:
        print_colored(f"❌ Error checking Neo4j status: {e}", Colors.RED)
        return False

def test_server_health():
    """Test server health endpoint"""
    print_header("🔍 TESTING SERVER HEALTH")
    
    try:
        print_colored(f"📍 Testing server at: {SERVER_URL}/health")
        
        response = requests.get(f"{SERVER_URL}/health", timeout=10)
        
        print_colored(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print_colored("✅ Server is responding", Colors.GREEN)
            print_colored(f"   Server Status: {data.get('status', 'unknown')}")
            
            if 'neo4j' in data:
                neo4j_status = data['neo4j'].get('status', 'unknown')
                print_colored(f"   Neo4j Status: {neo4j_status}")
                
                if neo4j_status != 'connected':
                    print_colored("❌ Server can't connect to Neo4j - this is your 503 error cause!", Colors.RED)
                    return False
            
            return True
            
        elif response.status_code == 503:
            print_colored("❌ 503 Service Unavailable - Neo4j connection problem!", Colors.RED)
            try:
                error_data = response.json()
                print_colored(f"   Error details: {error_data}", Colors.YELLOW)
            except:
                print_colored(f"   Raw response: {response.text}", Colors.YELLOW)
            return False
        else:
            print_colored(f"❌ Server error: {response.status_code}", Colors.RED)
            return False
            
    except requests.exceptions.ConnectionError:
        print_colored("❌ Cannot connect to server - is it running?", Colors.RED)
        return False
    except Exception as e:
        print_colored(f"❌ Server test failed: {e}", Colors.RED)
        return False

def provide_solutions():
    """Provide step-by-step solutions"""
    print_header("🛠️  SOLUTIONS FOR 503 ERROR")
    
    print_colored("The 503 Service Unavailable error means your server is running but can't connect to Neo4j.", Colors.YELLOW)
    print_colored("\n📋 Step-by-Step Solutions:\n", Colors.BOLD)
    
    print_colored("1. 🚀 START NEO4J DATABASE:")
    print_colored("   neo4j start")
    print_colored("   # OR if using Docker:")
    print_colored("   docker run --name neo4j -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/your_password neo4j")
    
    print_colored("\n2. 🔑 CHECK/SET NEO4J PASSWORD:")
    print_colored("   neo4j-admin set-password your_new_password")
    print_colored("   # Then update your server configuration file")
    
    print_colored("\n3. ✏️  UPDATE SERVER CONFIGURATION:")
    print_colored("   Edit your server file (standalone_server.py or mcpserver.py)")
    print_colored("   Change: NEO4J_PASSWORD = 'your_actual_password'")
    
    print_colored("\n4. 🔄 RESTART YOUR SERVER:")
    print_colored("   Stop the server (Ctrl+C)")
    print_colored("   python standalone_server.py")
    print_colored("   # Should show: ✅ Neo4j connection successful!")
    
    print_colored("\n5. 🧪 TEST THE FIX:")
    print_colored("   curl http://localhost:8000/health")
    print_colored("   # Should return: {\"status\": \"healthy\"}")

def check_port_availability():
    """Check if ports are available"""
    print_header("🔍 CHECKING PORT AVAILABILITY")
    
    import socket
    
    ports_to_check = [
        (7687, "Neo4j Bolt"),
        (7474, "Neo4j HTTP"),
        (8000, "Your Server")
    ]
    
    for port, service in ports_to_check:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                print_colored(f"✅ Port {port} ({service}) is open", Colors.GREEN)
            else:
                print_colored(f"❌ Port {port} ({service}) is not accessible", Colors.RED)
                
        except Exception as e:
            print_colored(f"❌ Error checking port {port}: {e}", Colors.RED)

async def main():
    """Main diagnostic function"""
    print_colored("🩺 NEO4J CONNECTION DIAGNOSTIC TOOL", Colors.BOLD)
    print_colored("This will help you fix the 503 Service Unavailable error\n", Colors.BLUE)
    
    # 1. Check if Neo4j is running
    neo4j_running = check_neo4j_running()
    
    # 2. Check port availability
    check_port_availability()
    
    # 3. Test Neo4j connection directly
    neo4j_works = await test_neo4j_connection()
    
    # 4. Test server health
    server_works = test_server_health()
    
    # 5. Provide diagnosis and solutions
    print_header("🎯 DIAGNOSIS SUMMARY")
    
    if neo4j_works and server_works:
        print_colored("🎉 EVERYTHING IS WORKING!", Colors.GREEN)
        print_colored("Your 503 error should be resolved.", Colors.GREEN)
    elif neo4j_works and not server_works:
        print_colored("❌ Neo4j works, but server has issues", Colors.RED)
        print_colored("Check server configuration and restart it.", Colors.YELLOW)
    elif not neo4j_works:
        print_colored("❌ NEO4J CONNECTION PROBLEM - This is causing your 503 error!", Colors.RED)
        if not neo4j_running:
            print_colored("Neo4j is not running - start it first!", Colors.YELLOW)
        else:
            print_colored("Neo4j is running but connection failed - check password!", Colors.YELLOW)
    
    # Always provide solutions
    provide_solutions()
    
    print_header("🏁 NEXT STEPS")
    
    if not neo4j_works:
        print_colored("1. Fix Neo4j connection using solutions above", Colors.YELLOW)
        print_colored("2. Run this diagnostic again to verify fix", Colors.YELLOW)
        print_colored("3. Restart your server", Colors.YELLOW)
    else:
        print_colored("1. Your Neo4j connection works!", Colors.GREEN)
        print_colored("2. Restart your server if needed", Colors.GREEN)
        print_colored("3. Test with: curl http://localhost:8000/health", Colors.GREEN)

if __name__ == "__main__":
    print("🔧 Starting Neo4j diagnostic...")
    print("⚠️  Make sure to update NEO4J_PASSWORD in this script first!")
    print()
    
    asyncio.run(main())
