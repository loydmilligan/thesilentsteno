#!/usr/bin/env python3

"""
Simple Audio Recorder Bridge for The Silent Steno

This module provides a simplified audio recording interface that bridges the
working audio recording logic from minimal_demo.py with the existing comprehensive
AudioRecorder architecture. It maintains backward compatibility while providing
the simple interface needed for the walking skeleton.

Key features:
- Direct sounddevice integration (compatible with minimal_demo.py)
- Simple recording interface (start/stop/play)
- Session persistence compatible with existing SessionManager
- Bridge to comprehensive AudioRecorder when needed
"""

import os
import wave
import threading
import time
import logging
import subprocess
import numpy as np
from typing import Dict, Optional, Callable, Any
from datetime import datetime
import json

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logging.warning("sounddevice not available for recording")

# Import data integration adapter
try:
    from src.data.integration_adapter import DataIntegrationAdapter
    DATA_INTEGRATION_AVAILABLE = True
except ImportError:
    DATA_INTEGRATION_AVAILABLE = False
    logging.warning("Data integration adapter not available - using fallback JSON storage")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleAudioRecorder:
    """
    Simple Audio Recorder for The Silent Steno
    
    Provides a simplified recording interface that bridges the working
    sounddevice-based recording from minimal_demo.py with the existing
    comprehensive recording architecture.
    """
    
    def __init__(self, storage_root: str = "recordings"):
        """Initialize simple audio recorder"""
        self.storage_root = storage_root
        
        # Recording state
        self.recording_state = "idle"  # idle, recording, processing, ready
        self.current_session = None
        self.is_recording = False
        self.recording_thread = None
        self.audio_data = None
        self.current_wav_file = None
        
        # Audio settings from minimal_demo.py (working configuration)
        self.sample_rate = 44100  # Hz (44.1kHz standard rate)
        self.channels = 1  # Mono recording
        self.input_device = None
        self.output_device = None
        
        # Session persistence
        self.sessions_dir = storage_root  # Use storage_root directly as sessions directory
        self.sessions_file = os.path.join(self.sessions_dir, "sessions.json")
        self.sessions = []
        
        # Initialize storage
        os.makedirs(self.sessions_dir, exist_ok=True)
        
        # Initialize data integration adapter
        if DATA_INTEGRATION_AVAILABLE:
            self.data_adapter = DataIntegrationAdapter(self.sessions_file, use_database=False)
            logger.info("Data integration adapter initialized")
        else:
            self.data_adapter = None
            logger.info("Using fallback JSON storage")
        
        # Auto-detect audio devices (same logic as minimal_demo.py)
        self._detect_audio_devices()
        
        # Load existing sessions
        self._load_sessions()
        
        logger.info(f"Simple audio recorder initialized with storage: {storage_root}")
    
    def _detect_audio_devices(self):
        """Detect and configure audio devices (from minimal_demo.py)"""
        if not AUDIO_AVAILABLE:
            logger.warning("Audio recording not available - sounddevice not installed")
            return
        
        try:
            devices = sd.query_devices()
            logger.info(f"Available audio devices: {len(devices)}")
            
            for i, device in enumerate(devices):
                logger.info(f"Device {i}: {device['name']} - In: {device['max_input_channels']} Out: {device['max_output_channels']}")
                
                # Look for USB Audio Device for INPUT (working configuration)
                if "USB Audio Device" in device['name'] and device['max_input_channels'] > 0:
                    self.input_device = i
                    logger.info(f"Found USB Audio Device input: {i}")
                
                # Look for built-in speakers for OUTPUT (HDMI)
                if "vc4-hdmi" in device['name'] and device['max_output_channels'] > 0:
                    self.output_device = i
                    logger.info(f"Found HDMI audio output: {i} - {device['name']}")
                
                # Fallback to first available input/output
                if self.input_device is None and device['max_input_channels'] > 0:
                    self.input_device = i
                if self.output_device is None and device['max_output_channels'] > 0:
                    self.output_device = i
            
            logger.info(f"Selected input device: {self.input_device}")
            logger.info(f"Selected output device: {self.output_device}")
            
        except Exception as e:
            logger.error(f"Error querying audio devices: {e}")
    
    def _load_sessions(self):
        """Load sessions using data integration adapter"""
        try:
            if self.data_adapter:
                self.sessions = self.data_adapter.get_all_sessions()
                logger.info(f"Loaded {len(self.sessions)} sessions via data adapter")
            else:
                # Fallback to direct JSON loading
                if os.path.exists(self.sessions_file):
                    with open(self.sessions_file, 'r') as f:
                        self.sessions = json.load(f)
                    logger.info(f"Loaded {len(self.sessions)} sessions via fallback JSON")
                else:
                    logger.info("No existing sessions file, starting fresh")
                    self.sessions = []
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
            self.sessions = []
    
    def _save_sessions(self):
        """Save sessions using data integration adapter"""
        try:
            if self.data_adapter:
                # Sessions are automatically saved by data adapter
                logger.info(f"Sessions managed by data adapter")
            else:
                # Fallback to direct JSON saving
                with open(self.sessions_file, 'w') as f:
                    json.dump(self.sessions, f, indent=2)
                logger.info(f"Saved {len(self.sessions)} sessions via fallback JSON")
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")
    
    def start_recording(self, session_id: str) -> bool:
        """
        Start recording for a session
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if recording started successfully
        """
        if self.recording_state != "idle":
            logger.warning(f"Cannot start recording, current state: {self.recording_state}")
            return False
        
        if not AUDIO_AVAILABLE:
            logger.error("Cannot start recording - sounddevice not available")
            return False
        
        try:
            # Update state
            self.recording_state = "recording"
            self.current_session = session_id
            self.is_recording = True
            self.audio_data = None
            
            # Start recording in background thread (same as minimal_demo.py)
            self.recording_thread = threading.Thread(target=self._record_audio, daemon=True)
            self.recording_thread.start()
            
            logger.info(f"Recording started for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            self.recording_state = "idle"
            self.is_recording = False
            return False
    
    def _record_audio(self):
        """Record audio in background thread (from minimal_demo.py)"""
        try:
            # Initialize recording buffer
            recording_buffer = []
            
            def audio_callback(indata, frames, time, status):
                """Callback for audio recording"""
                if status:
                    logger.warning(f"Audio callback status: {status}")
                if self.is_recording:
                    recording_buffer.append(indata.copy())
            
            # Start recording stream (same configuration as minimal_demo.py)
            with sd.InputStream(
                callback=audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=np.float32,
                device=self.input_device
            ):
                logger.info(f"Recording stream started: {self.sample_rate}Hz, {self.channels} channels")
                
                # Keep recording until stopped
                while self.is_recording:
                    sd.sleep(100)  # Sleep for 100ms
                
                # Combine all recorded chunks
                if recording_buffer:
                    self.audio_data = np.concatenate(recording_buffer, axis=0)
                    duration = len(self.audio_data) / self.sample_rate
                    logger.info(f"Recording completed: {duration:.2f} seconds, {len(self.audio_data)} samples")
                else:
                    self.audio_data = None
                    logger.warning("No audio data recorded - check microphone input")
                    
        except Exception as e:
            logger.error(f"Error during audio recording: {e}")
            self.audio_data = None
    
    def stop_recording(self) -> Optional[Dict[str, Any]]:
        """
        Stop current recording
        
        Returns:
            Recording info if successful, None otherwise
        """
        if self.recording_state != "recording":
            logger.warning(f"Cannot stop recording, current state: {self.recording_state}")
            return None
        
        try:
            # Stop the recording
            self.is_recording = False
            self.recording_state = "processing"
            
            # Wait for recording thread to finish
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=5.0)
            
            # Process and save the recording
            if self.audio_data is not None:
                duration = len(self.audio_data) / self.sample_rate
                samples = len(self.audio_data)
                
                # Save to WAV file (same as minimal_demo.py)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                wav_filename = os.path.join(self.sessions_dir, f"recording_{timestamp}.wav")
                
                # Convert float32 to int16 for WAV file
                audio_int16 = (self.audio_data * 32767).astype(np.int16)
                
                with wave.open(wav_filename, 'wb') as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(2)  # 2 bytes for int16
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(audio_int16.tobytes())
                
                self.current_wav_file = wav_filename
                
                # Create session record
                session_record = {
                    'id': len(self.sessions) + 1,
                    'session_id': self.current_session,
                    'timestamp': datetime.now().isoformat(),
                    'title': f'Recording Session {len(self.sessions) + 1}',
                    'wav_file': wav_filename,
                    'duration': duration,
                    'samples': samples,
                    'sample_rate': self.sample_rate,
                    'channels': self.channels,
                    'transcript': None  # Will be filled by transcription module
                }
                
                # Use data adapter to create session
                if self.data_adapter:
                    created_session_id = self.data_adapter.create_session(session_record)
                    if created_session_id:
                        session_record['created_id'] = created_session_id
                        logger.info(f"Session created in data store: {created_session_id}")
                    else:
                        logger.warning("Failed to create session in data store")
                
                self.sessions.append(session_record)
                self._save_sessions()
                
                # Update state
                self.recording_state = "ready"
                
                logger.info(f"Recording saved: {wav_filename} ({duration:.2f}s)")
                return session_record
            else:
                logger.error("No audio data to save")
                self.recording_state = "idle"
                return None
                
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            self.recording_state = "idle"
            return None
    
    def play_recording(self) -> bool:
        """
        Play back the current recording
        
        Returns:
            True if playback started successfully
        """
        if self.recording_state != "ready" or not self.current_wav_file:
            logger.warning("No recording available for playback")
            return False
        
        try:
            # Use aplay for playback (same as minimal_demo.py)
            logger.info(f"Playing back: {self.current_wav_file}")
            
            def play_in_background():
                try:
                    result = subprocess.run(
                        ['aplay', self.current_wav_file],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode != 0:
                        logger.error(f"aplay failed: {result.stderr}")
                    else:
                        logger.info("Playback completed successfully")
                except Exception as e:
                    logger.error(f"Error during playback: {e}")
            
            # Start playback in background
            playback_thread = threading.Thread(target=play_in_background, daemon=True)
            playback_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting playback: {e}")
            return False
    
    def get_current_recording_info(self) -> Optional[Dict[str, Any]]:
        """Get information about current recording"""
        if not self.sessions:
            return None
        
        return self.sessions[-1] if self.sessions else None
    
    def get_all_sessions(self) -> list:
        """Get all recording sessions"""
        return self.sessions.copy()
    
    def get_recording_state(self) -> str:
        """Get current recording state"""
        return self.recording_state
    
    def reset_to_idle(self):
        """Reset recorder to idle state"""
        self.recording_state = "idle"
        self.current_session = None
        self.is_recording = False
        self.audio_data = None
        self.current_wav_file = None
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get audio device information"""
        return {
            'input_device': self.input_device,
            'output_device': self.output_device,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'audio_available': AUDIO_AVAILABLE
        }


# Bridge function to create recorder instance compatible with existing architecture
def create_simple_recorder(storage_root: str = "recordings") -> SimpleAudioRecorder:
    """Create a simple audio recorder instance"""
    return SimpleAudioRecorder(storage_root)


if __name__ == "__main__":
    # Basic test when run directly
    print("Simple Audio Recorder Test")
    print("=" * 50)
    
    recorder = SimpleAudioRecorder("test_recordings")
    
    device_info = recorder.get_device_info()
    print(f"Device info: {device_info}")
    
    if AUDIO_AVAILABLE:
        print("Starting test recording...")
        session_id = "test-session-001"
        
        if recorder.start_recording(session_id):
            print("Recording for 3 seconds...")
            time.sleep(3)
            
            recording_info = recorder.stop_recording()
            if recording_info:
                print(f"Recording completed: {recording_info}")
                
                # Test playback
                print("Testing playback...")
                recorder.play_recording()
            else:
                print("Recording failed")
        else:
            print("Failed to start recording")
    else:
        print("Audio not available, skipping recording test")
    
    print("Test complete")