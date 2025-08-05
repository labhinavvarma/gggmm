#!/usr/bin/env python3
"""
MCP Client for Testing Health Details MCP Server
Tests all tools and prompts with real API integration
"""

import asyncio
import json
import logging
import sys
from typing import Dict, Any, List, Optional
import httpx
from mcp.client import Client
from mcp.client.stdio import stdio_client
import mcp.types as types

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthMCPClient:
    def __init__(self):
        self.client: Optional[Client] = None
        self.available_tools: List[str] = []
        self.available_prompts: List[str] = []
    
    async def connect_to_mcp_server(self, command: List[str] = None):
        """Connect to MCP server"""
        if command is None:
            command = [sys.executable, "mcpserver.py"]
        
        try:
            print("🔌 Connecting to Health Details MCP Server...")
            
            # Connect to MCP server via stdio
            async with stdio_client(command) as (read_stream, write_stream):
                async with Client(read_stream, write_stream) as client:
                    self.client = client
                    
                    # Initialize the connection
                    await self.initialize_connection()
                    
                    # Get available tools and prompts
                    await self.discover_capabilities()
                    
                    # Run interactive test menu
                    await self.run_test_menu()
                    
        except Exception as e:
            print(f"❌ Failed to connect to MCP server: {str(e)}")
            logger.error(f"Connection error: {str(e)}")
    
    async def initialize_connection(self):
        """Initialize MCP connection"""
        try:
            # Initialize with server info
            result = await self.client.initialize()
            print(f"✅ Connected to MCP server: {result.serverInfo.name}")
            print(f"   Protocol version: {result.protocolVersion}")
        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            raise
    
    async def discover_capabilities(self):
        """Discover available tools and prompts"""
        try:
            # Get available tools
            tools_result = await self.client.list_tools()
            self.available_tools = [tool.name for tool in tools_result.tools]
            print(f"🛠️  Available tools: {', '.join(self.available_tools)}")
            
            # Get available prompts
            prompts_result = await self.client.list_prompts()
            self.available_prompts = [prompt.name for prompt in prompts_result.prompts]
            print(f"📝 Available prompts: {', '.join(self.available_prompts)}")
            
        except Exception as e:
            print(f"❌ Failed to discover capabilities: {str(e)}")
    
    async def run_test_menu(self):
        """Run interactive test menu"""
        while True:
            print("\n" + "="*60)
            print("🏥 Health Details MCP Server Test Client")
            print("="*60)
            print("1. Test 'all' tool (System Overview)")
            print("2. Test 'token' tool (Authentication)")
            print("3. Test 'medical_submit' tool")
            print("4. Test 'pharmacy_submit' tool")
            print("5. Test 'mcid_search' tool")
            print("6. Test 'get_all_healthcare_data' tool")
            print("7. Test 'health-details' prompt")
            print("8. Test 'healthcare-summary' prompt")
            print("9. Run all tool tests")
            print("10. Test with sample patient data")
            print("0. Exit")
            print("-"*60)
            
            try:
                choice = input("Select option (0-10): ").strip()
                
                if choice == "0":
                    print("👋 Goodbye!")
                    break
                elif choice == "1":
                    await self.test_all_tool()
                elif choice == "2":
                    await self.test_token_tool()
                elif choice == "3":
                    await self.test_medical_submit_tool()
                elif choice == "4":
                    await self.test_pharmacy_submit_tool()
                elif choice == "5":
                    await self.test_mcid_search_tool()
                elif choice == "6":
                    await self.test_get_all_healthcare_data_tool()
                elif choice == "7":
                    await self.test_health_details_prompt()
                elif choice == "8":
                    await self.test_healthcare_summary_prompt()
                elif choice == "9":
                    await self.run_all_tool_tests()
                elif choice == "10":
                    await self.test_sample_patient_data()
                else:
                    print("❌ Invalid option, please try again.")
                    
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    async def test_all_tool(self):
        """Test the 'all' tool for system overview"""
        print("\n🧪 Testing 'all' tool (System Overview)...")
        
        try:
            result = await self.client.call_tool("all", {})
            print("✅ Tool execution successful!")
            print("📋 Result:")
            for content in result.content:
                if hasattr(content, 'text'):
                    # Pretty print JSON if possible
                    try:
                        data = json.loads(content.text)
                        print(json.dumps(data, indent=2))
                    except:
                        print(content.text)
        except Exception as e:
            print(f"❌ Tool execution failed: {str(e)}")
    
    async def test_token_tool(self):
        """Test the 'token' tool for authentication"""
        print("\n🧪 Testing 'token' tool (Authentication)...")
        
        try:
            result = await self.client.call_tool("token", {})
            print("✅ Tool execution successful!")
            print("📋 Result:")
            for content in result.content:
                if hasattr(content, 'text'):
                    try:
                        data = json.loads(content.text)
                        print(json.dumps(data, indent=2))
                    except:
                        print(content.text)
        except Exception as e:
            print(f"❌ Tool execution failed: {str(e)}")
    
    async def get_patient_data(self):
        """Get patient data from user input"""
        print("\n👤 Enter patient information:")
        
        first_name = input("First name: ").strip() or "John"
        last_name = input("Last name: ").strip() or "Doe"
        ssn = input("SSN (or press enter for test): ").strip() or "123-45-6789"
        date_of_birth = input("Date of birth (YYYY-MM-DD): ").strip() or "1980-01-01"
        gender = input("Gender (M/F): ").strip().upper() or "M"
        zip_code = input("ZIP code: ").strip() or "12345"
        
        return {
            "first_name": first_name,
            "last_name": last_name,
            "ssn": ssn,
            "date_of_birth": date_of_birth,
            "gender": gender,
            "zip_code": zip_code
        }
    
    async def test_medical_submit_tool(self):
        """Test the 'medical_submit' tool"""
        print("\n🧪 Testing 'medical_submit' tool...")
        
        try:
            patient_data = await self.get_patient_data()
            
            result = await self.client.call_tool("medical_submit", patient_data)
            print("✅ Tool execution successful!")
            print("📋 Result:")
            for content in result.content:
                if hasattr(content, 'text'):
                    try:
                        data = json.loads(content.text)
                        print(json.dumps(data, indent=2))
                    except:
                        print(content.text)
        except Exception as e:
            print(f"❌ Tool execution failed: {str(e)}")
    
    async def test_pharmacy_submit_tool(self):
        """Test the 'pharmacy_submit' tool"""
        print("\n🧪 Testing 'pharmacy_submit' tool...")
        
        try:
            patient_data = await self.get_patient_data()
            
            result = await self.client.call_tool("pharmacy_submit", patient_data)
            print("✅ Tool execution successful!")
            print("📋 Result:")
            for content in result.content:
                if hasattr(content, 'text'):
                    try:
                        data = json.loads(content.text)
                        print(json.dumps(data, indent=2))
                    except:
                        print(content.text)
        except Exception as e:
            print(f"❌ Tool execution failed: {str(e)}")
    
    async def test_mcid_search_tool(self):
        """Test the 'mcid_search' tool"""
        print("\n🧪 Testing 'mcid_search' tool...")
        
        try:
            patient_data = await self.get_patient_data()
            
            result = await self.client.call_tool("mcid_search", patient_data)
            print("✅ Tool execution successful!")
            print("📋 Result:")
            for content in result.content:
                if hasattr(content, 'text'):
                    try:
                        data = json.loads(content.text)
                        print(json.dumps(data, indent=2))
                    except:
                        print(content.text)
        except Exception as e:
            print(f"❌ Tool execution failed: {str(e)}")
    
    async def test_get_all_healthcare_data_tool(self):
        """Test the 'get_all_healthcare_data' tool"""
        print("\n🧪 Testing 'get_all_healthcare_data' tool...")
        
        try:
            patient_data = await self.get_patient_data()
            
            result = await self.client.call_tool("get_all_healthcare_data", patient_data)
            print("✅ Tool execution successful!")
            print("📋 Result:")
            for content in result.content:
                if hasattr(content, 'text'):
                    try:
                        data = json.loads(content.text)
                        print(json.dumps(data, indent=2))
                    except:
                        print(content.text)
        except Exception as e:
            print(f"❌ Tool execution failed: {str(e)}")
    
    async def test_health_details_prompt(self):
        """Test the 'health-details' prompt"""
        print("\n🧪 Testing 'health-details' prompt...")
        
        try:
            query = input("Enter your health query: ").strip() or "Show me system status and get a token"
            
            result = await self.client.get_prompt("health-details", {"query": query})
            print("✅ Prompt execution successful!")
            print("📋 Generated Prompt:")
            for message in result.messages:
                print(f"Role: {message.role}")
                if hasattr(message.content, 'text'):
                    print(f"Content: {message.content.text[:500]}...")
        except Exception as e:
            print(f"❌ Prompt execution failed: {str(e)}")
    
    async def test_healthcare_summary_prompt(self):
        """Test the 'healthcare-summary' prompt"""
        print("\n🧪 Testing 'healthcare-summary' prompt...")
        
        try:
            query = input("Enter summary request: ").strip() or "Summarize current API status"
            
            result = await self.client.get_prompt("healthcare-summary", {"query": query})
            print("✅ Prompt execution successful!")
            print("📋 Generated Prompt:")
            for message in result.messages:
                print(f"Role: {message.role}")
                if hasattr(message.content, 'text'):
                    print(f"Content: {message.content.text[:500]}...")
        except Exception as e:
            print(f"❌ Prompt execution failed: {str(e)}")
    
    async def run_all_tool_tests(self):
        """Run all tool tests sequentially"""
        print("\n🚀 Running all tool tests...")
        
        # Test system tools
        await self.test_all_tool()
        await asyncio.sleep(1)
        
        await self.test_token_tool()
        await asyncio.sleep(1)
        
        # Test patient data tools with sample data
        sample_patient = {
            "first_name": "Jane",
            "last_name": "Smith",
            "ssn": "987-65-4321",
            "date_of_birth": "1975-05-15",
            "gender": "F",
            "zip_code": "54321"
        }
        
        print("\n🧪 Testing with sample patient data...")
        
        try:
            print("\n--- Medical Submit Test ---")
            result = await self.client.call_tool("medical_submit", sample_patient)
            print("✅ Medical submit successful")
            
            print("\n--- Pharmacy Submit Test ---")
            result = await self.client.call_tool("pharmacy_submit", sample_patient)
            print("✅ Pharmacy submit successful")
            
            print("\n--- MCID Search Test ---")
            result = await self.client.call_tool("mcid_search", sample_patient)
            print("✅ MCID search successful")
            
        except Exception as e:
            print(f"❌ Batch test failed: {str(e)}")
        
        print("\n🎉 All tool tests completed!")
    
    async def test_sample_patient_data(self):
        """Test with comprehensive sample patient data"""
        print("\n🧪 Testing with comprehensive sample patient data...")
        
        sample_patients = [
            {
                "first_name": "Alice",
                "last_name": "Johnson",
                "ssn": "111-22-3333",
                "date_of_birth": "1985-03-10",
                "gender": "F",
                "zip_code": "10001"
            },
            {
                "first_name": "Bob",
                "last_name": "Williams",
                "ssn": "444-55-6666",
                "date_of_birth": "1970-08-22",
                "gender": "M",
                "zip_code": "90210"
            }
        ]
        
        for i, patient in enumerate(sample_patients, 1):
            print(f"\n--- Testing Patient {i}: {patient['first_name']} {patient['last_name']} ---")
            
            try:
                # Test get_all_healthcare_data for comprehensive testing
                result = await self.client.call_tool("get_all_healthcare_data", patient)
                print(f"✅ Patient {i} test successful")
                
                # Print summary of results
                for content in result.content:
                    if hasattr(content, 'text'):
                        try:
                            data = json.loads(content.text)
                            print(f"   Patient: {data.get('patient', {})}")
                            print(f"   Services tested: {len([k for k in data.keys() if k.endswith('_service')])}")
                        except:
                            print(f"   Response length: {len(content.text)} characters")
                
                await asyncio.sleep(2)  # Wait between patients
                
            except Exception as e:
                print(f"❌ Patient {i} test failed: {str(e)}")
        
        print("\n🎉 Sample patient data testing completed!")

# SSE Test Client for testing via FastAPI
class SSETestClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    async def test_sse_connection(self):
        """Test SSE connection to FastAPI app"""
        print("\n🧪 Testing SSE connection to FastAPI app...")
        
        try:
            async with httpx.AsyncClient() as client:
                # Test basic endpoints first
                response = await client.get(f"{self.base_url}/api/cortex/health/status")
                if response.status_code == 200:
                    print("✅ FastAPI app is running")
                    print(f"📋 Status: {response.json()}")
                else:
                    print(f"❌ FastAPI app not responding: {response.status_code}")
                    return
                
                # Test available tools endpoint
                response = await client.get(f"{self.base_url}/api/cortex/health/tools")
                if response.status_code == 200:
                    print("✅ MCP tools endpoint working")
                    tools_data = response.json()
                    print(f"📋 Available tools: {tools_data.get('tools', [])}")
                    print(f"📋 Available prompts: {tools_data.get('prompts', [])}")
                
        except Exception as e:
            print(f"❌ SSE test failed: {str(e)}")

async def main():
    """Main function to run MCP client tests"""
    print("🏥 Health Details MCP Server Test Client")
    print("="*60)
    print("1. Test MCP server directly (stdio)")
    print("2. Test FastAPI SSE integration")
    print("3. Run both tests")
    print("0. Exit")
    
    try:
        choice = input("Select test mode (0-3): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            return
        elif choice == "1":
            # Test MCP server directly
            client = HealthMCPClient()
            await client.connect_to_mcp_server()
        elif choice == "2":
            # Test SSE integration
            sse_client = SSETestClient()
            await sse_client.test_sse_connection()
        elif choice == "3":
            # Run both tests
            print("\n=== Testing SSE Integration ===")
            sse_client = SSETestClient()
            await sse_client.test_sse_connection()
            
            print("\n=== Testing MCP Server Directly ===")
            client = HealthMCPClient()
            await client.connect_to_mcp_server()
        else:
            print("❌ Invalid option")
            
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
