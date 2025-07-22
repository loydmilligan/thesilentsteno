#!/usr/bin/env python3
"""
Bluetooth Audio Recorder Module

Extends the simple audio recorder to support Bluetooth audio sources
"""

import subprocess
import logging
import os
import time
import threading
from typing import Optional, Dict, Any, List
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)


class BluetoothAudioRecorder:
    """Recorder that can use Bluetooth audio sources"""
    
    def __init__(self, storage_root: str = "demo_sessions"):
        """Initialize Bluetooth audio recorder"""
        self.storage_root = storage_root
        self.is_recording = False
        self.current_session = None
        self.recording_process = None
        self.current_wav_file = None
        
        # Ensure storage directory exists
        os.makedirs(storage_root, exist_ok=True)
    
    def find_bluetooth_source(self) -> Optional[str]:
        """Find available Bluetooth audio source"""
        try:
            result = subprocess.run(
                ["pactl", "list", "short", "sources"],
                capture_output=True,
                text=True
            )
            
            # Look for Bluetooth A2DP source
            for line in result.stdout.split('\n'):
                if 'bluez_source' in line and 'a2dp_source' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        source_name = parts[1]
                        # Check if source is available
                        if len(parts) >= 4 and 'RUNNING' in parts[3]:
                            logger.info(f"Found active Bluetooth source: {source_name}")
                            return source_name
                        else:
                            logger.info(f"Found inactive Bluetooth source: {source_name}")
                            return source_name
            
            logger.warning("No Bluetooth audio source found")
            return None
            
        except Exception as e:
            logger.error(f"Error finding Bluetooth source: {e}")
            return None
    
    def get_available_sources(self) -> List[Dict[str, str]]:
        """Get all available audio sources"""
        sources = []
        try:
            result = subprocess.run(
                ["pactl", "list", "short", "sources"],
                capture_output=True,
                text=True
            )
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        source_type = "Bluetooth" if "bluez" in parts[1] else "System"
                        sources.append({
                            'name': parts[1],
                            'type': source_type,
                            'status': parts[3] if len(parts) > 3 else 'UNKNOWN'
                        })
            
            return sources
            
        except Exception as e:
            logger.error(f"Error getting audio sources: {e}")
            return []
    
    def start_recording(self, session_id: str = None, use_bluetooth: bool = True) -> bool:
        """Start recording from best available source"""
        if self.is_recording:
            logger.warning("Recording already in progress")
            return False
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"bluetooth_{int(time.time())}"
        
        self.current_session = session_id
        
        # Find audio source
        audio_source = None
        
        if use_bluetooth:
            audio_source = self.find_bluetooth_source()
            if not audio_source:
                logger.warning("No Bluetooth source available, falling back to default")
        
        # Fallback to default if no Bluetooth source
        if not audio_source:
            audio_source = "default"
        
        # Create output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_wav_file = os.path.join(
            self.storage_root, 
            f"recording_{timestamp}.wav"
        )
        
        try:
            # Start recording using parecord
            cmd = [
                "parecord",
                f"--device={audio_source}",
                "--format=s16le",
                "--rate=44100",
                "--channels=2",
                "--file-format=wav",
                self.current_wav_file
            ]
            
            logger.info(f"Starting recording: {' '.join(cmd)}")
            self.recording_process = subprocess.Popen(cmd)
            self.is_recording = True
            
            logger.info(f"Recording started for session {session_id}")
            logger.info(f"Audio source: {audio_source}")
            logger.info(f"Output file: {self.current_wav_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            self.is_recording = False
            self.current_session = None
            return False
    
    def stop_recording(self) -> Optional[Dict[str, Any]]:
        """Stop recording and return recording info"""
        if not self.is_recording:
            logger.warning("No recording in progress")
            return None
        
        try:
            # Stop the recording process
            if self.recording_process:
                self.recording_process.terminate()
                self.recording_process.wait(timeout=5)
            
            self.is_recording = False
            
            # Check if file was created
            if os.path.exists(self.current_wav_file):
                file_size = os.path.getsize(self.current_wav_file)
                
                # Calculate approximate duration (rough estimate)
                # 44100 Hz * 2 channels * 2 bytes/sample = 176400 bytes/second
                duration = file_size / 176400.0
                
                recording_info = {
                    'session_id': self.current_session,
                    'file_path': self.current_wav_file,
                    'file_size': file_size,
                    'duration': duration,
                    'timestamp': datetime.now().isoformat(),
                    'source_type': 'bluetooth' if 'bluez' in str(self.recording_process.args) else 'system'
                }
                
                logger.info(f"Recording stopped: {self.current_wav_file}")
                logger.info(f"File size: {file_size} bytes, Duration: {duration:.1f}s")
                
                # Reset state
                self.current_session = None
                self.current_wav_file = None
                self.recording_process = None
                
                return recording_info
            else:
                logger.error("Recording file was not created")
                return None
                
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            self.is_recording = False
            return None
    
    def get_recording_status(self) -> Dict[str, Any]:
        """Get current recording status"""
        bluetooth_source = self.find_bluetooth_source()
        available_sources = self.get_available_sources()
        
        return {
            'is_recording': self.is_recording,
            'current_session': self.current_session,
            'current_file': self.current_wav_file,
            'bluetooth_available': bluetooth_source is not None,
            'bluetooth_source': bluetooth_source,
            'available_sources': available_sources
        }


# Test function
def test_bluetooth_recorder():
    """Test the Bluetooth recorder"""
    recorder = BluetoothAudioRecorder()
    
    print("Bluetooth Audio Recorder Test")
    print("=" * 30)
    
    # Show status
    status = recorder.get_recording_status()
    print(f"Bluetooth available: {status['bluetooth_available']}")
    print(f"Bluetooth source: {status['bluetooth_source']}")
    
    # Start recording
    print("\nStarting recording...")
    if recorder.start_recording():
        print("✅ Recording started")
        time.sleep(5)  # Record for 5 seconds
        
        # Stop recording
        info = recorder.stop_recording()
        if info:
            print(f"✅ Recording complete: {info['file_path']}")
            print(f"   Duration: {info['duration']:.1f}s")
        else:
            print("❌ Recording failed")
    else:
        print("❌ Failed to start recording")


if __name__ == "__main__":
    test_bluetooth_recorder()