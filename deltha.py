#!/usr/bin/env python3
"""
Simple Standalone MCP Client for Milliman APIs
==============================================

A lightweight, stable MCP client that can directly call Milliman API tools
without complex LLM integration. Perfect for testing and direct API access.

Usage:
    python simple_mcp_client.py
    python simple_mcp_client.py --patient-data patient.json
    python simple_mcp_client.py --operation get_all_data --interactive
"""

import asyncio
import json
import argparse
import logging
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import uuid

# MCP client imports
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError as e:
    print(f"❌ Missing MCP dependencies: {e}")
    print("📦 Install with: pip install langchain-mcp-adapters")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_mcp_client.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MCPClientConfig:
    """Configuration for MCP client"""
    server_name: str = "MillimanServer"
    server_url: str = "http://localhost:8000/sse"
    transport: str = "sse"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

@dataclass
class PatientInfo:
    """Patient information structure"""
    first_name: str
    last_name: str
    ssn: str
    date_of_birth: str  # YYYY-MM-DD
    gender: str
    zip_code: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for API calls"""
        return asdict(self)
    
    def validate(self) -> tuple[bool, List[str]]:
        """Validate patient data"""
        errors = []
        
        if not self.first_name or len(self.first_name.strip()) < 1:
            errors.append("First name is required")
        
        if not self.last_name or len(self.last_name.strip()) < 1:
            errors.append("Last name is required")
        
        # Validate SSN (9 digits)
        ssn_digits = ''.join(filter(str.isdigit, self.ssn))
        if len(ssn_digits) != 9:
            errors.append("SSN must be exactly 9 digits")
        
        if self.gender.upper() not in ['M', 'F']:
            errors.append("Gender must be 'M' or 'F'")
        
        # Validate zip code (at least 5 digits)
        zip_digits = ''.join(filter(str.isdigit, self.zip_code))
        if len(zip_digits) < 5:
            errors.append("Zip code must be at least 5 digits")
        
        # Validate date format
        try:
            datetime.strptime(self.date_of_birth, '%Y-%m-%d')
        except ValueError:
            errors.append("Date of birth must be in YYYY-MM-DD format")
        
        return len(errors) == 0, errors
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'PatientInfo':
        """Create PatientInfo from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_json_file(cls, filepath: str) -> 'PatientInfo':
        """Load PatientInfo from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            raise ValueError(f"Failed to load patient data from {filepath}: {e}")

class SimpleMCPClient:
    """Simple MCP client for direct API tool access"""
    
    def __init__(self, config: MCPClientConfig):
        self.config = config
        self.client = None
        self.tools = {}
        self.is_connected = False
        
        logger.info(f"🔧 Simple MCP Client initialized: {config.server_url}")
    
    async def connect(self) -> bool:
        """Connect to MCP server"""
        try:
            logger.info(f"🔌 Connecting to MCP server: {self.config.server_url}")
            
            # Create MCP client
            self.client = MultiServerMCPClient({
                self.config.server_name: {
                    "url": self.config.server_url,
                    "transport": self.config.transport,
                }
            })
            
            # Enter client context
            await self.client.__aenter__()
            
            # Get available tools
            tools = self.client.get_tools()
            self.tools = {tool.name: tool for tool in tools}
            
            self.is_connected = True
            
            logger.info(f"✅ Connected successfully. Available tools: {list(self.tools.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MCP server: {e}")
            traceback.print_exc()
            return False
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        try:
            if self.client:
                await self.client.__aexit__(None, None, None)
                self.is_connected = False
                logger.info("🔌 Disconnected from MCP server")
        except Exception as e:
            logger.error(f"❌ Error during disconnect: {e}")
    
    async def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a specific MCP tool"""
        if not self.is_connected:
            return {"error": "Not connected to MCP server"}
        
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not available. Available tools: {list(self.tools.keys())}"}
        
        try:
            logger.info(f"🔧 Calling tool: {tool_name}")
            
            # Call the tool
            tool = self.tools[tool_name]
            result = await tool.call(**kwargs)
            
            logger.info(f"✅ Tool call successful: {tool_name}")
            return result
            
        except Exception as e:
            error_msg = f"Failed to call tool '{tool_name}': {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {"error": error_msg, "details": str(e)}
    
    async def get_token(self) -> Dict[str, Any]:
        """Get authentication token"""
        return await self.call_tool("get_token")
    
    async def medical_submit(self, patient: PatientInfo) -> Dict[str, Any]:
        """Submit medical record request"""
        patient_dict = patient.to_dict()
        return await self.call_tool("medical_submit", **patient_dict)
    
    async def mcid_search(self, patient: PatientInfo) -> Dict[str, Any]:
        """Search MCID database"""
        patient_dict = patient.to_dict()
        return await self.call_tool("mcid_search", **patient_dict)
    
    async def get_all_data(self, patient: PatientInfo) -> Dict[str, Any]:
        """Get comprehensive patient data"""
        patient_dict = patient.to_dict()
        return await self.call_tool("get_all_data", **patient_dict)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return list(self.tools.keys())

class MCPClientCLI:
    """Command-line interface for the MCP client"""
    
    def __init__(self):
        self.client = None
        self.config = MCPClientConfig()
    
    async def run_interactive(self):
        """Run interactive mode"""
        print("🏥 Milliman MCP Client - Interactive Mode")
        print("=" * 50)
        
        # Connect to server
        self.client = SimpleMCPClient(self.config)
        if not await self.client.connect():
            print("❌ Failed to connect to MCP server. Please check that it's running.")
            return
        
        print(f"✅ Connected to MCP server")
        print(f"🔧 Available tools: {', '.join(self.client.get_available_tools())}")
        print("\n💡 Commands:")
        print("• 'token' - Get authentication token")
        print("• 'patient' - Enter patient data and run operations")
        print("• 'tools' - List available tools")
        print("• 'help' - Show this help")
        print("• 'exit' - Quit")
        print("=" * 50)
        
        try:
            while True:
                command = input("\n🔧 Enter command: ").strip().lower()
                
                if command in ['exit', 'quit', 'bye']:
                    break
                elif command == 'help':
                    self.show_help()
                elif command == 'tools':
                    self.show_tools()
                elif command == 'token':
                    await self.handle_token_command()
                elif command == 'patient':
                    await self.handle_patient_command()
                else:
                    print(f"❓ Unknown command: {command}. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        
        finally:
            if self.client:
                await self.client.disconnect()
    
    def show_help(self):
        """Show help information"""
        print("\n🆘 Available Commands:")
        print("• token    - Get API authentication token")
        print("• patient  - Enter patient information and run API operations")
        print("• tools    - List all available MCP tools")
        print("• help     - Show this help message")
        print("• exit     - Quit the application")
        
        print("\n📋 Patient Data Format:")
        print("• First Name: Patient's first name")
        print("• Last Name: Patient's last name")
        print("• SSN: 9-digit Social Security Number")
        print("• Date of Birth: YYYY-MM-DD format")
        print("• Gender: M (Male) or F (Female)")
        print("• Zip Code: 5+ digit postal code")
    
    def show_tools(self):
        """Show available tools"""
        if self.client and self.client.is_connected:
            tools = self.client.get_available_tools()
            print(f"\n🔧 Available Tools ({len(tools)}):")
            for tool in tools:
                print(f"• {tool}")
        else:
            print("❌ Not connected to MCP server")
    
    async def handle_token_command(self):
        """Handle token command"""
        print("\n🔑 Getting authentication token...")
        
        try:
            result = await self.client.get_token()
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print("✅ Token retrieved successfully:")
                print(json.dumps(result, indent=2))
        
        except Exception as e:
            print(f"❌ Error getting token: {e}")
    
    async def handle_patient_command(self):
        """Handle patient data entry and operations"""
        print("\n👤 Patient Data Entry")
        print("-" * 25)
        
        try:
            # Collect patient information
            first_name = input("First Name: ").strip()
            last_name = input("Last Name: ").strip()
            ssn = input("SSN (9 digits): ").strip()
            date_of_birth = input("Date of Birth (YYYY-MM-DD): ").strip()
            gender = input("Gender (M/F): ").strip().upper()
            zip_code = input("Zip Code: ").strip()
            
            # Create patient object
            patient = PatientInfo(
                first_name=first_name,
                last_name=last_name,
                ssn=ssn,
                date_of_birth=date_of_birth,
                gender=gender,
                zip_code=zip_code
            )
            
            # Validate patient data
            is_valid, errors = patient.validate()
            if not is_valid:
                print("\n❌ Invalid patient data:")
                for error in errors:
                    print(f"• {error}")
                return
            
            print(f"\n✅ Patient data validated: {first_name} {last_name}")
            
            # Choose operation
            operations = [
                ("1", "get_all_data", "Get comprehensive patient data"),
                ("2", "medical_submit", "Submit medical record request"),
                ("3", "mcid_search", "Search MCID database"),
                ("4", "get_token", "Get authentication token only")
            ]
            
            print("\n🔧 Available Operations:")
            for num, op, desc in operations:
                print(f"{num}. {desc}")
            
            choice = input("\nSelect operation (1-4): ").strip()
            
            # Execute operation
            if choice == "1":
                result = await self.client.get_all_data(patient)
            elif choice == "2":
                result = await self.client.medical_submit(patient)
            elif choice == "3":
                result = await self.client.mcid_search(patient)
            elif choice == "4":
                result = await self.client.get_token()
            else:
                print("❌ Invalid choice")
                return
            
            # Display results
            print("\n📊 API Response:")
            print("=" * 30)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                if "details" in result:
                    print(f"Details: {result['details']}")
            else:
                print(json.dumps(result, indent=2))
                
                # Save results to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"api_response_{choice}_{timestamp}.json"
                
                try:
                    with open(filename, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"\n💾 Results saved to: {filename}")
                except Exception as e:
                    print(f"⚠️ Could not save results: {e}")
        
        except Exception as e:
            print(f"❌ Error processing patient command: {e}")
    
    async def run_with_patient_file(self, patient_file: str, operation: str):
        """Run with patient data from file"""
        print(f"🏥 Running MCP Client with patient file: {patient_file}")
        
        try:
            # Load patient data
            patient = PatientInfo.from_json_file(patient_file)
            print(f"✅ Loaded patient data: {patient.first_name} {patient.last_name}")
            
            # Validate
            is_valid, errors = patient.validate()
            if not is_valid:
                print("❌ Invalid patient data:")
                for error in errors:
                    print(f"• {error}")
                return
            
            # Connect to server
            self.client = SimpleMCPClient(self.config)
            if not await self.client.connect():
                print("❌ Failed to connect to MCP server")
                return
            
            print(f"✅ Connected to MCP server")
            
            # Execute operation
            print(f"🔧 Executing operation: {operation}")
            
            if operation == "get_all_data":
                result = await self.client.get_all_data(patient)
            elif operation == "medical_submit":
                result = await self.client.medical_submit(patient)
            elif operation == "mcid_search":
                result = await self.client.mcid_search(patient)
            elif operation == "get_token":
                result = await self.client.get_token()
            else:
                print(f"❌ Unknown operation: {operation}")
                return
            
            # Display and save results
            print("\n📊 API Response:")
            print("=" * 30)
            print(json.dumps(result, indent=2))
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"output_{operation}_{timestamp}.json"
            
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"\n💾 Results saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            traceback.print_exc()
        
        finally:
            if self.client:
                await self.client.disconnect()

def create_sample_patient_file():
    """Create a sample patient data file"""
    sample_patient = {
        "first_name": "John",
        "last_name": "Smith",
        "ssn": "123456789",
        "date_of_birth": "1980-01-15",
        "gender": "M",
        "zip_code": "12345"
    }
    
    filename = "sample_patient.json"
    with open(filename, 'w') as f:
        json.dump(sample_patient, f, indent=2)
    
    print(f"📝 Sample patient file created: {filename}")
    return filename

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simple MCP Client for Milliman APIs")
    parser.add_argument("--patient-data", help="JSON file with patient data")
    parser.add_argument("--operation", 
                      choices=["get_all_data", "medical_submit", "mcid_search", "get_token"],
                      default="get_all_data",
                      help="API operation to perform")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--server-url", default="http://localhost:8000/sse", 
                      help="MCP server URL")
    parser.add_argument("--create-sample", action="store_true",
                      help="Create sample patient data file")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_patient_file()
        return
    
    # Create CLI instance
    cli = MCPClientCLI()
    
    # Update config if custom server URL provided
    if args.server_url != "http://localhost:8000/sse":
        cli.config.server_url = args.server_url
        print(f"🔧 Using custom server URL: {args.server_url}")
    
    try:
        if args.interactive or (not args.patient_data):
            # Run interactive mode
            asyncio.run(cli.run_interactive())
        else:
            # Run with patient file
            asyncio.run(cli.run_with_patient_file(args.patient_data, args.operation))
    
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
