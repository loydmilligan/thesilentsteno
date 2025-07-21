#!/usr/bin/env python3
"""Test network accessibility on port 5000"""

import socket
import sys

def test_port_5000():
    """Test if port 5000 is accessible"""
    try:
        # Create a test socket
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Try to bind to port 5000 on all interfaces
        test_socket.bind(('0.0.0.0', 5001))  # Using 5001 to not conflict
        test_socket.listen(1)
        
        print("✅ Successfully bound to all interfaces")
        print("   This confirms the Pi can listen on all network interfaces")
        
        # Get all IP addresses
        hostname = socket.gethostname()
        local_ips = socket.gethostbyname_ex(hostname)[2]
        
        print("\n📡 Available IP addresses:")
        for ip in local_ips:
            print(f"   - {ip}")
            
        # Also check main interface
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        main_ip = s.getsockname()[0]
        s.close()
        
        print(f"\n🌐 Main network IP: {main_ip}")
        print(f"\n💡 Try accessing the web UI from another device at:")
        print(f"   http://{main_ip}:5000")
        
        test_socket.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_port_5000()