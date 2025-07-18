#!/usr/bin/env python3
"""
Test if the refactored app can initialize properly without running the full UI
"""

import sys
import os

# Add src to path
sys.path.append('/home/mmariani/projects/thesilentsteno/src')

print("Testing Refactored App Initialization")
print("=" * 50)

try:
    # Import the modules
    from recording.simple_audio_recorder import SimpleAudioRecorder
    from ai.simple_transcriber import SimpleTranscriber
    from ui.simple_recording_ui import SimpleRecordingApp
    
    print("✓ All modules imported successfully")
    
    # Initialize components
    recorder = SimpleAudioRecorder("demo_sessions")
    transcriber = SimpleTranscriber(backend="cpu", model_name="base")
    
    print(f"✓ Audio recorder initialized: {recorder.get_recording_state()}")
    print(f"✓ Transcriber initialized: {transcriber.is_available()}")
    
    # Create UI app (without running it)
    ui_app = SimpleRecordingApp(
        audio_recorder=recorder,
        transcriber=transcriber
    )
    
    print("✓ UI app created successfully")
    
    # Test device info
    device_info = recorder.get_device_info()
    print(f"\nDevice Information:")
    print(f"- Audio available: {device_info['audio_available']}")
    print(f"- Input device: {device_info['input_device']}")
    print(f"- Output device: {device_info['output_device']}")
    print(f"- Sample rate: {device_info['sample_rate']} Hz")
    
    # Test session info
    sessions = recorder.get_all_sessions()
    print(f"\nSession Information:")
    print(f"- Total sessions: {len(sessions)}")
    if sessions:
        last = sessions[-1]
        print(f"- Last session: {last['timestamp'][:19]}")
        print(f"- Duration: {last['duration']:.2f}s")
        if last.get('transcript'):
            trans = last['transcript'][:50] + "..." if len(last['transcript']) > 50 else last['transcript']
            print(f"- Transcript: {trans}")
    
    print("\n✅ Refactored app initialization successful!")
    print("The app is ready to run on the Pi display.")
    
    print("\nTo run the full UI:")
    print("1. On Pi display: python minimal_demo_refactored.py")
    print("2. Or windowed: python minimal_demo_refactored.py --windowed")
    
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    import traceback
    traceback.print_exc()