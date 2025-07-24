#!/usr/bin/env python3
"""
Simple Bluetooth Audio Test
Just tests recording from Bluetooth source without transcription
"""

import subprocess
import time
import os

def find_bluetooth_source():
    """Find Bluetooth audio source"""
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True,
            text=True
        )
        
        print("Available audio sources:")
        for line in result.stdout.split('\n'):
            if line.strip():
                print(f"  {line}")
        
        # Look for Bluetooth source
        for line in result.stdout.split('\n'):
            if 'bluez_source' in line and 'a2dp_source' in line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    return parts[1]
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_bluetooth_recording():
    """Test recording from Bluetooth"""
    print("🎧 Testing Bluetooth Audio Recording")
    print("=" * 40)
    
    # Find Bluetooth source
    bt_source = find_bluetooth_source()
    if not bt_source:
        print("❌ No Bluetooth audio source found")
        print("Make sure your phone is connected and playing audio")
        return
    
    print(f"✅ Found Bluetooth source: {bt_source}")
    
    # Create output file
    output_file = f"demo_sessions/bluetooth_test_{int(time.time())}.wav"
    os.makedirs("demo_sessions", exist_ok=True)
    
    # Record for 5 seconds
    print("\n🎵 Play audio on your phone now!")
    print("Recording will start in 3 seconds...")
    
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    
    print("🔴 Recording for 5 seconds...")
    
    cmd = [
        "parecord",
        "--device=" + bt_source,
        "--format=s16le",
        "--rate=44100", 
        "--channels=2",
        "--file-format=wav",
        output_file
    ]
    
    # Start recording
    process = subprocess.Popen(cmd)
    time.sleep(5)  # Record for 5 seconds
    process.terminate()
    process.wait()
    
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"✅ Recording saved: {output_file}")
        print(f"   File size: {file_size} bytes")
        
        # Try to play it back
        print("\n🔊 Playing back recording...")
        subprocess.run(["aplay", output_file], check=False)
        
    else:
        print("❌ Recording failed")

if __name__ == "__main__":
    test_bluetooth_recording()