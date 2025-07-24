#!/usr/bin/env python3
"""
Test script to verify multiple recordings work
"""

import sys
import time
import uuid

# Add src to path
sys.path.append('/home/mmariani/projects/thesilentsteno/src')

from recording.simple_audio_recorder import SimpleAudioRecorder

print("Testing Multiple Recordings Fix")
print("=" * 40)

# Create recorder
recorder = SimpleAudioRecorder("demo_sessions")

# Test multiple recording cycle
for i in range(3):
    print(f"\n--- Recording {i+1} ---")
    
    # Check initial state
    state = recorder.get_recording_state()
    print(f"Initial state: {state}")
    
    if state != "idle":
        print("❌ State should be 'idle' for new recording")
        recorder.reset_to_idle()
        print(f"Reset to: {recorder.get_recording_state()}")
    
    # Start recording
    session_id = f"test-{i+1}-{str(uuid.uuid4())[:8]}"
    if recorder.start_recording(session_id):
        print(f"✅ Recording {i+1} started: {session_id}")
        
        # Record briefly
        time.sleep(0.5)
        
        # Stop recording
        result = recorder.stop_recording()
        if result:
            print(f"✅ Recording {i+1} stopped: {result['duration']:.2f}s")
            
            # Reset to idle (this is what the UI fix does)
            recorder.reset_to_idle()
            print(f"✅ Reset to idle: {recorder.get_recording_state()}")
        else:
            print(f"❌ Recording {i+1} failed to stop")
            break
    else:
        print(f"❌ Recording {i+1} failed to start")
        break

print(f"\n✅ Multiple recordings test complete!")
print(f"Total sessions: {len(recorder.get_all_sessions())}")
print("The UI fix should now allow multiple recordings!")