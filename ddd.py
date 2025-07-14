
#!/usr/bin/env python3
"""
Neo4j Connection Test Script
Test different Neo4j connection configurations
"""

import asyncio
from neo4j import GraphDatabase
import neo4j.exceptions

def test_neo4j_connection(uri, username, password, database="neo4j"):
    """Test Neo4j connection with given parameters"""
    
    print(f"🔍 Testing connection to: {uri}")
    print(f"   Username: {username}")
    print(f"   Database: {database}")
    
    try:
        # Create driver
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Test connectivity
        driver.verify_connectivity()
        print("✅ Connection established successfully!")
        
        # Test database access
        with driver.session(database=database) as session:
            result = session.run("RETURN 1 as test")
            test_result = result.single()
            if test_result and test_result["test"] == 1:
                print("✅ Database access test passed!")
            else:
                print("❌ Database access test failed!")
                return False
        
        # Test basic query
        with driver.session(database=database) as session:
            result = session.run("CALL db.ping()")
            ping_result = result.single()
            print(f"✅ Ping result: {ping_result}")
        
        driver.close()
        return True
        
    except neo4j.exceptions.ServiceUnavailable as e:
        print(f"❌ Service unavailable: {e}")
        print("   💡 Check if Neo4j is running and accessible")
        return False
    except neo4j.exceptions.AuthError as e:
        print(f"❌ Authentication failed: {e}")
        print("   💡 Check username and password")
        return False
    except neo4j.exceptions.DatabaseError as e:
        print(f"❌ Database error: {e}")
        print("   💡 Check database name")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def main():
    """Test different Neo4j configurations"""
    
    print("🚀 Neo4j Connection Test")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        {
            "name": "Local Neo4j (default)",
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",  # Change this!
            "database": "neo4j"
        },
        {
            "name": "Your Remote Neo4j",
            "uri": "neo4j://10.189.116.237:7687",
            "username": "neo4j",
            "password": "your_password",  # Change this!
            "database": "connectiq"
        },
        {
            "name": "Local Neo4j (bolt+s)",
            "uri": "bolt+s://localhost:7687",
            "username": "neo4j",
            "password": "password",  # Change this!
            "database": "neo4j"
        }
    ]
    
    successful_configs = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n📋 Test {i}: {config['name']}")
        print("-" * 30)
        
        success = test_neo4j_connection(
            config["uri"],
            config["username"], 
            config["password"],
            config["database"]
        )
        
        if success:
            successful_configs.append(config)
        
        print()
    
    # Summary
    print("=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    if successful_configs:
        print(f"✅ {len(successful_configs)} configuration(s) worked!")
        print("\n🎯 Successful configurations:")
        for config in successful_configs:
            print(f"   • {config['name']}: {config['uri']}")
            print(f"     Username: {config['username']}")
            print(f"     Database: {config['database']}")
        
        print("\n💡 Update your mcpserver.py with a working configuration!")
    else:
        print("❌ No configurations worked!")
        print("\n🔧 Common solutions:")
        print("   1. Start Neo4j Desktop")
        print("   2. Check if Neo4j is running: systemctl status neo4j")
        print("   3. Verify firewall settings")
        print("   4. Check network connectivity")
        print("   5. Update passwords in the test script")

if __name__ == "__main__":
    main()
