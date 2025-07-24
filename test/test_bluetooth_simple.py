#!/usr/bin/env python3
"""
Simple Bluetooth test for The Silent Steno
Tests basic Bluetooth functionality without full pipeline
"""

import subprocess
import sys

def check_bluetooth_status():
    """Check if Bluetooth service is running"""
    print("1. Checking Bluetooth service...")
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "bluetooth"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip() == "active":
            print("✅ Bluetooth service is active")
            return True
        else:
            print("❌ Bluetooth service is not active")
            return False
    except Exception as e:
        print(f"❌ Error checking Bluetooth: {e}")
        return False

def list_bluetooth_devices():
    """List paired Bluetooth devices"""
    print("\n2. Listing Bluetooth devices...")
    try:
        result = subprocess.run(
            ["bluetoothctl", "devices"],
            capture_output=True,
            text=True
        )
        if result.stdout:
            print("Paired devices:")
            print(result.stdout)
        else:
            print("No paired devices found")
    except Exception as e:
        print(f"❌ Error listing devices: {e}")

def show_bluetooth_info():
    """Show Bluetooth adapter info"""
    print("\n3. Bluetooth adapter info...")
    try:
        result = subprocess.run(
            ["bluetoothctl", "show"],
            capture_output=True,
            text=True
        )
        if result.stdout:
            # Extract key information
            for line in result.stdout.split('\n'):
                if any(key in line for key in ['Name:', 'Powered:', 'Discoverable:', 'Pairable:']):
                    print(f"  {line.strip()}")
    except Exception as e:
        print(f"❌ Error getting adapter info: {e}")

def check_audio_devices():
    """Check audio devices"""
    print("\n4. Audio devices...")
    try:
        # Check PulseAudio sinks
        result = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True,
            text=True
        )
        if result.stdout:
            print("Audio sinks:")
            print(result.stdout)
        
        # Check PulseAudio sources
        result = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True,
            text=True
        )
        if result.stdout:
            print("\nAudio sources:")
            print(result.stdout)
    except Exception as e:
        print(f"❌ Error checking audio devices: {e}")

def main():
    print("="*60)
    print("The Silent Steno - Bluetooth Test")
    print("="*60)
    
    # Check Bluetooth status
    if not check_bluetooth_status():
        print("\nPlease start Bluetooth service:")
        print("  sudo systemctl start bluetooth")
        return
    
    # List devices
    list_bluetooth_devices()
    
    # Show adapter info
    show_bluetooth_info()
    
    # Check audio
    check_audio_devices()
    
    print("\n" + "="*60)
    print("Next steps:")
    print("1. Run: sudo ./setup_bluetooth.sh")
    print("2. Pair your phone and headphones")
    print("3. Run the full demo: python3 demo_bluetooth_pipeline.py")

if __name__ == "__main__":
    main()