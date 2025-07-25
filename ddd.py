"""
Debug Streamlit UI with Detailed Error Reporting
This version shows exactly what's happening with the connection
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
import traceback

# ============================================
# CONFIGURATION
# ============================================

SERVER_PORT = 8000
BASE_URL = f"http://localhost:{SERVER_PORT}"

st.set_page_config(
    page_title="Debug Neo4j Chatbot",
    page_icon="🔍",
    layout="wide"
)

# ============================================
# DEBUG FUNCTIONS
# ============================================

def test_connection_detailed():
    """Test connection with detailed debugging"""
    debug_info = []
    
    debug_info.append(f"🔍 Testing connection to: {BASE_URL}")
    debug_info.append(f"⏰ Test time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Test basic connectivity
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('localhost', SERVER_PORT))
        sock.close()
        
        if result == 0:
            debug_info.append(f"✅ Port {SERVER_PORT} is open and accepting connections")
        else:
            debug_info.append(f"❌ Port {SERVER_PORT} is not accessible (error code: {result})")
            return False, debug_info
            
    except Exception as e:
        debug_info.append(f"❌ Socket test failed: {e}")
        return False, debug_info
    
    # Test each endpoint
    endpoints = [
        ("/", "GET", "Root"),
        ("/health", "GET", "Health"),
        ("/stats", "GET", "Stats"),
        ("/graph?limit=5", "GET", "Graph")
    ]
    
    for endpoint, method, name in endpoints:
        try:
            url = f"{BASE_URL}{endpoint}"
            debug_info.append(f"\n🌐 Testing {name}: {url}")
            
            response = requests.get(url, timeout=10)
            debug_info.append(f"   Status Code: {response.status_code}")
            debug_info.append(f"   Content-Type: {response.headers.get('content-type', 'unknown')}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    debug_info.append(f"   ✅ JSON Response: {len(str(data))} characters")
                    if isinstance(data, dict):
                        for key in list(data.keys())[:5]:  # Show first 5 keys
                            debug_info.append(f"      - {key}: {type(data[key]).__name__}")
                except:
                    debug_info.append(f"   ⚠️  Non-JSON Response: {response.text[:100]}...")
            else:
                debug_info.append(f"   ❌ Error Response: {response.text[:200]}...")
                
        except requests.exceptions.ConnectionError as e:
            debug_info.append(f"   ❌ Connection Error: {e}")
        except requests.exceptions.Timeout as e:
            debug_info.append(f"   ❌ Timeout Error: {e}")
        except Exception as e:
            debug_info.append(f"   ❌ Unexpected Error: {e}")
    
    return True, debug_info

def test_chat_endpoint():
    """Test the chat endpoint specifically"""
    debug_info = []
    
    try:
        chat_data = {
            "question": "How many nodes are in the graph?",
            "session_id": "debug_test"
        }
        
        debug_info.append("🗣️ Testing chat endpoint...")
        debug_info.append(f"   Request: {json.dumps(chat_data, indent=2)}")
        
        response = requests.post(
            f"{BASE_URL}/chat",
            json=chat_data,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        debug_info.append(f"   Status Code: {response.status_code}")
        debug_info.append(f"   Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                debug_info.append("   ✅ Chat Response:")
                debug_info.append(f"      Success: {result.get('success', 'unknown')}")
                debug_info.append(f"      Tool: {result.get('tool', 'none')}")
                debug_info.append(f"      Query: {result.get('query', 'none')}")
                debug_info.append(f"      Answer: {result.get('answer', 'none')[:100]}...")
                return True, result, debug_info
            except Exception as e:
                debug_info.append(f"   ❌ JSON Parse Error: {e}")
                debug_info.append(f"   Raw Response: {response.text[:500]}...")
        else:
            debug_info.append(f"   ❌ HTTP Error: {response.text}")
            
    except Exception as e:
        debug_info.append(f"❌ Chat Test Error: {e}")
        debug_info.append(f"   Traceback: {traceback.format_exc()}")
    
    return False, None, debug_info

# ============================================
# UI LAYOUT
# ============================================

st.title("🔍 Debug Neo4j Chatbot Connection")
st.markdown("**Detailed debugging for connection issues**")

# Configuration display
st.markdown("### 🔧 Configuration")
col1, col2 = st.columns(2)
with col1:
    st.text(f"Server Port: {SERVER_PORT}")
    st.text(f"Base URL: {BASE_URL}")
with col2:
    st.text(f"Current Time: {datetime.now().strftime('%H:%M:%S')}")

# Debug section
st.markdown("### 🧪 Connection Tests")

if st.button("🔍 Run Detailed Connection Test"):
    with st.spinner("Testing connection..."):
        success, debug_info = test_connection_detailed()
    
    # Display results
    if success:
        st.success("✅ Basic connectivity test passed!")
    else:
        st.error("❌ Connection test failed!")
    
    # Show debug info
    st.markdown("#### Debug Information:")
    for info in debug_info:
        st.text(info)

# Chat test section  
st.markdown("### 💬 Chat Endpoint Test")

if st.button("🗣️ Test Chat Endpoint"):
    with st.spinner("Testing chat..."):
        success, result, debug_info = test_chat_endpoint()
    
    if success:
        st.success("✅ Chat endpoint working!")
        st.json(result)
    else:
        st.error("❌ Chat endpoint failed!")
    
    # Show debug info
    st.markdown("#### Chat Debug Information:")
    for info in debug_info:
        st.text(info)

# Manual endpoint testing
st.markdown("### 🌐 Manual Endpoint Testing")

endpoint = st.selectbox(
    "Select endpoint to test:",
    ["/", "/health", "/stats", "/graph?limit=5", "/docs"]
)

if st.button(f"Test {endpoint}"):
    try:
        url = f"{BASE_URL}{endpoint}"
        st.text(f"Testing: {url}")
        
        response = requests.get(url, timeout=10)
        
        st.text(f"Status Code: {response.status_code}")
        st.text(f"Content-Type: {response.headers.get('content-type')}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                st.success("✅ Success!")
                st.json(data)
            except:
                st.text("Non-JSON response:")
                st.text(response.text)
        else:
            st.error(f"Error {response.status_code}")
            st.text(response.text)
            
    except Exception as e:
        st.error(f"Request failed: {e}")

# Server status check
st.markdown("### 📊 Quick Status Check")

try:
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    if response.status_code == 200:
        health_data = response.json()
        st.success("✅ Server is responding!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Server Status", health_data.get("status", "unknown"))
            if "neo4j" in health_data:
                st.metric("Neo4j Status", health_data["neo4j"].get("status", "unknown"))
        with col2:
            if "agent" in health_data:
                st.metric("Agent Status", health_data["agent"].get("status", "unknown"))
            if "server" in health_data:
                st.metric("Server Port", health_data["server"].get("port", "unknown"))
    else:
        st.error(f"❌ Server error: {response.status_code}")
        
except requests.exceptions.ConnectionError:
    st.error("❌ Cannot connect to server!")
    st.warning("**Possible causes:**")
    st.text("• MCP server not running")
    st.text("• Wrong port number")
    st.text("• Firewall blocking connection")
    st.text("• Neo4j database not accessible")
    
except Exception as e:
    st.error(f"❌ Connection error: {e}")

# Troubleshooting guide
st.markdown("### 🛠️ Troubleshooting Guide")

st.markdown("""
**If you see connection errors:**

1. **Check if MCP server is running:**
   ```bash
   python fixed_mcp_server.py
   ```

2. **Verify Neo4j is running:**
   ```bash
   neo4j status
   ```

3. **Test server directly in browser:**
   - http://localhost:8000/health
   - http://localhost:8000/

4. **Check firewall/ports:**
   ```bash
   netstat -an | grep 8000
   ```

5. **Update configuration:**
   - Change NEO4J_PASSWORD in fixed_mcp_server.py
   - Change CORTEX_API_KEY in fixed_mcp_server.py

**If endpoints return 404:**
- Make sure you're using `fixed_mcp_server.py` (not the original)
- Check that all required endpoints are implemented
- Restart the server after making changes

**If Neo4j connection fails:**
- Verify Neo4j credentials
- Check Neo4j is running on bolt://localhost:7687
- Test connection with Neo4j Browser

**Expected working state:**
- ✅ Server Status: healthy
- ✅ Neo4j Status: connected  
- ✅ Agent Status: ready
- ✅ All endpoints return 200 OK
""")

# Quick fixes
st.markdown("### ⚡ Quick Fixes")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Refresh Page"):
        st.rerun()
    
    if st.button("🧹 Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")

with col2:
    st.markdown("**Recommended startup order:**")
    st.text("1. Start Neo4j database")
    st.text("2. Start fixed_mcp_server.py")
    st.text("3. Test endpoints (this page)")
    st.text("4. Start main Streamlit UI")

# Footer
st.markdown("---")
st.markdown("🔍 **Debug UI v1.0** - Use this to identify and fix connection issues before using the main chatbot interface.")
