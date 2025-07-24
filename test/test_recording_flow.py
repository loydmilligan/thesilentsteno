#!/usr/bin/env python3
"""
Test the complete recording → transcription → playback flow
without the UI
"""

import sys
import time
import uuid

# Add src to path
sys.path.append('/home/mmariani/projects/thesilentsteno/src')

from recording.simple_audio_recorder import SimpleAudioRecorder
from ai.simple_transcriber import SimpleTranscriber

print("Testing Complete Recording Flow")
print("=" * 50)

# Initialize modules
print("\n1. Initializing modules...")
recorder = SimpleAudioRecorder("demo_sessions")
transcriber = SimpleTranscriber(backend="cpu", model_name="base")

print(f"✓ Audio recorder ready (state: {recorder.get_recording_state()})")
print(f"✓ Transcriber ready (available: {transcriber.is_available()})")

# Show device info
device_info = recorder.get_device_info()
print(f"\nAudio devices:")
print(f"  - Input device: {device_info['input_device']} (USB Audio Device)")
print(f"  - Output device: {device_info['output_device']} (HDMI)")
print(f"  - Sample rate: {device_info['sample_rate']} Hz")

# Start recording
print("\n2. Starting recording for 3 seconds...")
session_id = str(uuid.uuid4())[:8]

if recorder.start_recording(session_id):
    print("✓ Recording started")
    print("  Recording for 3 seconds...")
    
    # Record for 3 seconds
    time.sleep(3)
    
    # Stop recording
    print("\n3. Stopping recording...")
    recording_info = recorder.stop_recording()
    
    if recording_info:
        print("✓ Recording stopped")
        print(f"  - Duration: {recording_info['duration']:.2f}s")
        print(f"  - Samples: {recording_info['samples']:,}")
        print(f"  - File: {recording_info['wav_file']}")
        
        # Transcribe
        print("\n4. Transcribing audio...")
        if transcriber.is_available():
            transcript = transcriber.transcribe_audio(recording_info['wav_file'])
            print(f"✓ Transcription: {transcript}")
            
            # Update session
            recorder.sessions[-1]['transcript'] = transcript
            recorder._save_sessions()
            print("✓ Session updated with transcript")
        else:
            print("✗ Transcriber not available (Whisper not installed)")
            print("  To enable: pip install openai-whisper")
        
        # Play back
        print("\n5. Playing back recording...")
        if recorder.play_recording():
            print("✓ Playback started!")
            print("  Audio should be playing through speakers...")
            time.sleep(2)
        else:
            print("✗ Playback failed")
            
        print("\n✅ Complete flow test finished!")
        
    else:
        print("✗ Recording failed")
else:
    print("✗ Failed to start recording")

# Show session history
print("\n6. Session History:")
sessions = recorder.get_all_sessions()
print(f"Total sessions: {len(sessions)}")
if sessions:
    print("\nLast 3 sessions:")
    for session in sessions[-3:]:
        print(f"  - {session['timestamp'][:19]} | {session['duration']:.2f}s", end="")
        if session.get('transcript'):
            trans = session['transcript'][:30] + "..." if len(session['transcript']) > 30 else session['transcript']
            print(f" | {trans}")
        else:
            print(" | [No transcript]")