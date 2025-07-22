#!/usr/bin/env python3
"""
Bluetooth Audio Recorder for The Silent Steno

This recorder specifically uses Bluetooth audio sources (from phone)
for recording and transcription while maintaining audio forwarding.
"""

import sys
import os
import subprocess
import logging
import time
from typing import Optional, Dict, Any

# Add project root to path
sys.path.append('/home/mmariani/projects/thesilentsteno')

# Import the walking skeleton adapter
from src.integration.walking_skeleton_adapter import WalkingSkeletonAdapter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BluetoothAudioRecorder:
    """Recorder that uses Bluetooth audio sources"""
    
    def __init__(self):
        """Initialize Bluetooth audio recorder"""
        self.bluetooth_source = None
        self.adapter = None
        self.recording_session = None
        
    def find_bluetooth_source(self) -> Optional[str]:
        """Find the Bluetooth audio source from phone"""
        try:
            # Get PulseAudio sources
            result = subprocess.run(
                ["pactl", "list", "short", "sources"],
                capture_output=True,
                text=True
            )
            
            # Look for Bluetooth source
            for line in result.stdout.split('\n'):
                if 'bluez_source' in line and 'a2dp_source' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        source_name = parts[1]
                        logger.info(f"Found Bluetooth source: {source_name}")
                        return source_name
            
            logger.warning("No Bluetooth audio source found")
            return None
            
        except Exception as e:
            logger.error(f"Error finding Bluetooth source: {e}")
            return None
    
    def setup_bluetooth_recording(self) -> bool:
        """Set up recording from Bluetooth source"""
        try:
            # Find Bluetooth source
            self.bluetooth_source = self.find_bluetooth_source()
            if not self.bluetooth_source:
                logger.error("No Bluetooth audio source available")
                return False
            
            # Initialize adapter
            self.adapter = WalkingSkeletonAdapter()
            logger.info("Walking skeleton adapter initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up Bluetooth recording: {e}")
            return False
    
    def record_from_bluetooth(self, duration_seconds: int = 30) -> Optional[str]:
        """Record audio from Bluetooth source"""
        try:
            logger.info(f"Starting Bluetooth recording for {duration_seconds} seconds...")
            
            # Create unique session ID
            session_id = f"bluetooth_{int(time.time())}"
            
            # Use parecord to capture from Bluetooth source
            output_file = f"demo_sessions/bluetooth_recording_{int(time.time())}.wav"
            
            # Record from Bluetooth source using parecord
            cmd = [
                "parecord",
                "--device", self.bluetooth_source,
                "--format", "s16le",
                "--rate", "44100",
                "--channels", "2",
                "--file-format", "wav",
                output_file
            ]
            
            logger.info(f"Recording command: {' '.join(cmd)}")
            
            # Start recording process
            process = subprocess.Popen(cmd)
            
            # Let it record for specified duration
            time.sleep(duration_seconds)
            
            # Stop recording
            process.terminate()
            process.wait()
            
            if os.path.exists(output_file):
                logger.info(f"Bluetooth recording saved: {output_file}")
                return output_file
            else:
                logger.error("Recording file was not created")
                return None
                
        except Exception as e:
            logger.error(f"Error recording from Bluetooth: {e}")
            return None
    
    def transcribe_bluetooth_recording(self, wav_file: str) -> Optional[Dict[str, Any]]:
        """Transcribe Bluetooth recording"""
        try:
            logger.info(f"Transcribing Bluetooth recording: {wav_file}")
            
            # Use the adapter's transcriber
            if self.adapter and self.adapter.simple_transcriber:
                transcript = self.adapter.simple_transcriber.transcribe_audio(wav_file)
                
                if transcript:
                    # Get AI analysis
                    analysis = self.adapter.simple_transcriber.analyze_transcript(transcript)
                    
                    result = {
                        'transcript': transcript,
                        'analysis': analysis,
                        'source': 'bluetooth',
                        'wav_file': wav_file
                    }
                    
                    logger.info("Bluetooth transcription complete")
                    return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error transcribing Bluetooth recording: {e}")
            return None
    
    def test_bluetooth_pipeline(self):
        """Test the complete Bluetooth recording and transcription pipeline"""
        print("🎧 Testing Bluetooth Audio Pipeline")
        print("=" * 50)
        
        # Setup
        print("1. Setting up Bluetooth recording...")
        if not self.setup_bluetooth_recording():
            print("❌ Failed to setup Bluetooth recording")
            return
        print(f"✅ Using Bluetooth source: {self.bluetooth_source}")
        
        # Show current Bluetooth connections
        print("\n2. Current Bluetooth connections:")
        try:
            result = subprocess.run(["bluetoothctl", "devices", "Connected"], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                print(result.stdout)
            else:
                print("No connected devices shown")
        except:
            print("Could not check Bluetooth devices")
        
        # Test recording
        print("\n3. Testing Bluetooth audio capture...")
        print("💡 Play some audio on your phone now!")
        print("   Recording will start in 3 seconds...")
        
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        
        wav_file = self.record_from_bluetooth(duration_seconds=10)
        
        if not wav_file:
            print("❌ Recording failed")
            return
        
        print(f"✅ Recording complete: {wav_file}")
        
        # Test transcription
        print("\n4. Transcribing audio...")
        result = self.transcribe_bluetooth_recording(wav_file)
        
        if result:
            print("✅ Transcription complete!")
            print(f"\n📝 Transcript: {result['transcript'][:200]}...")
            
            analysis = result.get('analysis', {})
            if analysis:
                print(f"\n🔍 Analysis:")
                print(f"   Word count: {analysis.get('word_count', 0)}")
                print(f"   Sentiment: {analysis.get('sentiment', 'neutral')}")
                if analysis.get('action_items'):
                    print(f"   Action items: {analysis['action_items'][:2]}")
                if analysis.get('topics'):
                    print(f"   Topics: {analysis['topics'][:3]}")
        else:
            print("❌ Transcription failed")
        
        print("\n✅ Bluetooth pipeline test complete!")


def main():
    """Main entry point"""
    recorder = BluetoothAudioRecorder()
    recorder.test_bluetooth_pipeline()


if __name__ == "__main__":
    main()