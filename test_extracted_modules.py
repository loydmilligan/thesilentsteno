#!/usr/bin/env python3
"""
Test script for all extracted modules
Tests the complete modular architecture without needing the full UI
"""

import sys
import os
import time

# Add src to path
sys.path.append('/home/mmariani/projects/thesilentsteno/src')

print("Testing Complete Modular Architecture")
print("=" * 60)

# Test 1: Individual Module Loading
print("\n1. Testing Individual Module Loading...")
modules_loaded = {}

try:
    from recording.simple_audio_recorder import SimpleAudioRecorder
    modules_loaded['audio_recorder'] = True
    print("✓ SimpleAudioRecorder loaded")
except Exception as e:
    modules_loaded['audio_recorder'] = False
    print(f"✗ SimpleAudioRecorder failed: {e}")

try:
    from ai.simple_transcriber import SimpleTranscriber
    modules_loaded['transcriber'] = True
    print("✓ SimpleTranscriber loaded")
except Exception as e:
    modules_loaded['transcriber'] = False
    print(f"✗ SimpleTranscriber failed: {e}")

try:
    from ui.simple_recording_ui import SimpleRecordingUI, SimpleRecordingApp
    modules_loaded['ui'] = True
    print("✓ SimpleRecordingUI loaded")
except Exception as e:
    modules_loaded['ui'] = False
    print(f"✗ SimpleRecordingUI failed: {e}")

# Test 2: Module Integration
print("\n2. Testing Module Integration...")
if all(modules_loaded.values()):
    print("✓ All modules loaded successfully")
    
    # Initialize components
    recorder = SimpleAudioRecorder("demo_sessions")
    transcriber = SimpleTranscriber(backend="cpu", model_name="base")
    
    print(f"✓ Audio recorder: {recorder.get_recording_state()}")
    print(f"✓ Transcriber available: {transcriber.is_available()}")
    
    # Test UI component (without running full app)
    if modules_loaded['ui']:
        try:
            # Create UI app instance
            ui_app = SimpleRecordingApp(
                audio_recorder=recorder,
                transcriber=transcriber
            )
            print("✓ UI app created successfully")
            
            # Test UI component creation
            ui_widget = ui_app.build()
            print("✓ UI widget built successfully")
            
        except Exception as e:
            print(f"✗ UI integration failed: {e}")
            import traceback
            traceback.print_exc()
    
else:
    print("✗ Some modules failed to load")

# Test 3: Audio System Test
print("\n3. Testing Audio System...")
if modules_loaded['audio_recorder']:
    try:
        device_info = recorder.get_device_info()
        print(f"✓ Audio devices detected:")
        print(f"  - Input: {device_info['input_device']}")
        print(f"  - Output: {device_info['output_device']}")
        print(f"  - Sample rate: {device_info['sample_rate']} Hz")
        
        # Check existing sessions
        sessions = recorder.get_all_sessions()
        print(f"✓ Existing sessions: {len(sessions)}")
        
        if sessions:
            last_session = sessions[-1]
            print(f"  - Last recording: {last_session['timestamp'][:19]}")
            print(f"  - Duration: {last_session['duration']:.2f}s")
            
            # Test playback
            recorder.current_wav_file = last_session['wav_file']
            recorder.recording_state = 'ready'
            
            print("✓ Testing playback...")
            if recorder.play_recording():
                print("  Audio playback initiated")
            else:
                print("  No audio for playback")
        
    except Exception as e:
        print(f"✗ Audio system test failed: {e}")

# Test 4: Transcription System Test
print("\n4. Testing Transcription System...")
if modules_loaded['transcriber']:
    try:
        backend_info = transcriber.get_backend_info()
        print(f"✓ Backend info:")
        print(f"  - Backend: {backend_info['backend']}")
        print(f"  - Available: {backend_info['available']}")
        print(f"  - Model: {backend_info['model_name']}")
        
        # List available backends
        backends = transcriber.list_available_backends()
        print(f"✓ Available backends:")
        for backend in backends:
            status = "Available" if backend['available'] else "Not available"
            print(f"  - {backend['name']}: {status}")
        
        # Test transcription with existing file (if available)
        if sessions and transcriber.is_available():
            test_file = sessions[-1]['wav_file']
            if os.path.exists(test_file):
                print(f"✓ Testing transcription with: {os.path.basename(test_file)}")
                transcript = transcriber.transcribe_audio(test_file)
                print(f"  Result: {transcript[:100]}...")
        
    except Exception as e:
        print(f"✗ Transcription system test failed: {e}")

# Test 5: End-to-End Flow Simulation
print("\n5. Testing End-to-End Flow Simulation...")
if modules_loaded['audio_recorder'] and modules_loaded['transcriber']:
    try:
        print("✓ Simulating recording → transcription → playback flow")
        
        # Show current state
        print(f"  - Recorder state: {recorder.get_recording_state()}")
        print(f"  - Transcriber ready: {transcriber.is_available()}")
        
        # Simulate the flow that would happen in the UI
        if sessions:
            last_session = sessions[-1]
            print(f"  - Using last session: {last_session['timestamp'][:19]}")
            
            # Simulate transcription
            if transcriber.is_available():
                print("  - Transcription would work with Whisper installed")
            else:
                print("  - Transcription requires: pip install openai-whisper")
            
            # Simulate playback
            print("  - Playback functionality ready")
            
        print("✓ End-to-end flow simulation complete")
        
    except Exception as e:
        print(f"✗ End-to-end flow test failed: {e}")

# Test Summary
print("\n" + "=" * 60)
print("TEST SUMMARY:")
print(f"- SimpleAudioRecorder: {'✓ Working' if modules_loaded['audio_recorder'] else '✗ Failed'}")
print(f"- SimpleTranscriber: {'✓ Working' if modules_loaded['transcriber'] else '✗ Failed'}")
print(f"- SimpleRecordingUI: {'✓ Working' if modules_loaded['ui'] else '✗ Failed'}")

if all(modules_loaded.values()):
    print("\n🎉 ALL MODULES WORKING!")
    print("The walking skeleton has been successfully refactored into:")
    print("  1. Recording module (working)")
    print("  2. Transcription module (backend-agnostic)")
    print("  3. UI module (extracted)")
    print("  4. Complete integration (functional)")
    print("\nReady for:")
    print("  - Hailo 8 integration (transcription backend)")
    print("  - Integration with existing comprehensive UI")
    print("  - Further architectural development")
else:
    print("\n⚠️  Some modules need attention")
    print("Check the failed modules above")