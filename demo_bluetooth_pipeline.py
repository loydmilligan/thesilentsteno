#!/usr/bin/env python3
"""
The Silent Steno - Bluetooth Audio Pipeline Demo

This demo shows how to set up the Pi as a Bluetooth audio proxy that:
1. Receives audio from your phone (A2DP Sink)
2. Forwards audio to your headphones (A2DP Source)
3. Records and transcribes the audio in real-time

Current setup uses USB audio input as a placeholder until Bluetooth is configured.
"""

import sys
import os
import time
import logging
import threading
from typing import Optional, Dict, Any

# Add project root to path
sys.path.append('/home/mmariani/projects/thesilentsteno')

# Import our components
from src.bluetooth.bluez_manager import BlueZManager, BluetoothState
from src.bluetooth.connection_manager import ConnectionManager, DeviceRole, ConnectionState
from src.audio.audio_pipeline import AudioPipeline, AudioConfig, PipelineState
from src.integration.walking_skeleton_adapter import WalkingSkeletonAdapter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BluetoothAudioDemo:
    """Demo class for Bluetooth audio pipeline"""
    
    def __init__(self):
        """Initialize demo components"""
        self.bluez_manager = None
        self.connection_manager = None
        self.audio_pipeline = None
        self.adapter = None
        self.is_running = False
        
        # Device addresses (will be populated during setup)
        self.phone_address = None
        self.headphone_address = None
        
    def initialize_bluetooth(self) -> bool:
        """Initialize Bluetooth components"""
        try:
            logger.info("Initializing Bluetooth components...")
            
            # Initialize BlueZ manager
            self.bluez_manager = BlueZManager()
            
            # Start Bluetooth service
            if not self.bluez_manager.start_bluetooth():
                logger.error("Failed to start Bluetooth service")
                return False
            
            # Initialize connection manager
            self.connection_manager = ConnectionManager()
            
            # Check Bluetooth status
            status = self.bluez_manager.get_service_status()
            logger.info(f"Bluetooth status: {status}")
            
            return status == BluetoothState.RUNNING
            
        except Exception as e:
            logger.error(f"Failed to initialize Bluetooth: {e}")
            return False
    
    def scan_devices(self) -> Dict[str, Any]:
        """Scan for Bluetooth devices"""
        logger.info("Scanning for Bluetooth devices...")
        
        try:
            devices = self.bluez_manager.discover_devices(timeout=10)
            
            logger.info(f"Found {len(devices)} devices:")
            for device in devices:
                logger.info(f"  - {device['name']} ({device['address']})")
            
            return devices
            
        except Exception as e:
            logger.error(f"Failed to scan devices: {e}")
            return {}
    
    def setup_audio_pipeline(self) -> bool:
        """Set up the audio pipeline"""
        try:
            logger.info("Setting up audio pipeline...")
            
            # Create audio configuration for low latency
            config = AudioConfig(
                sample_rate=44100,
                channels=2,
                buffer_size=128,
                target_latency_ms=40.0,
                enable_monitoring=True,
                enable_forwarding=True
            )
            
            # Initialize audio pipeline
            self.audio_pipeline = AudioPipeline(config)
            
            # Initialize walking skeleton adapter for recording
            self.adapter = WalkingSkeletonAdapter()
            
            logger.info("Audio pipeline configured")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup audio pipeline: {e}")
            return False
    
    def connect_phone(self, address: str) -> bool:
        """Connect to phone as A2DP Sink"""
        logger.info(f"Connecting to phone: {address}")
        
        try:
            # Pair if needed
            if not self.connection_manager.is_paired(address):
                logger.info("Device not paired, initiating pairing...")
                if not self.connection_manager.pair_device(address, "Phone", DeviceRole.SOURCE):
                    logger.error("Failed to pair with phone")
                    return False
            
            # Connect as A2DP Sink
            if self.connection_manager.connect_device(address):
                self.phone_address = address
                logger.info("Successfully connected to phone")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to phone: {e}")
            return False
    
    def connect_headphones(self, address: str) -> bool:
        """Connect to headphones as A2DP Source"""
        logger.info(f"Connecting to headphones: {address}")
        
        try:
            # Pair if needed
            if not self.connection_manager.is_paired(address):
                logger.info("Device not paired, initiating pairing...")
                if not self.connection_manager.pair_device(address, "Headphones", DeviceRole.SINK):
                    logger.error("Failed to pair with headphones")
                    return False
            
            # Connect as A2DP Source
            if self.connection_manager.connect_device(address):
                self.headphone_address = address
                logger.info("Successfully connected to headphones")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to headphones: {e}")
            return False
    
    def start_audio_forwarding(self) -> bool:
        """Start forwarding audio from phone to headphones"""
        try:
            logger.info("Starting audio forwarding...")
            
            # For now, use USB input until Bluetooth audio is configured
            logger.warning("Using USB audio input (Bluetooth audio setup pending)")
            
            # Start the audio pipeline
            if self.audio_pipeline:
                self.audio_pipeline.start()
                logger.info("Audio pipeline started")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to start audio forwarding: {e}")
            return False
    
    def start_recording(self) -> Optional[str]:
        """Start recording and transcription"""
        try:
            logger.info("Starting recording...")
            
            session_id = self.adapter.start_recording()
            if session_id:
                logger.info(f"Recording started: {session_id}")
                return session_id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return None
    
    def stop_recording(self) -> Optional[Dict[str, Any]]:
        """Stop recording and get transcription"""
        try:
            logger.info("Stopping recording...")
            
            info = self.adapter.stop_recording()
            if info:
                logger.info(f"Recording stopped: {info}")
                
                # Start transcription
                logger.info("Starting transcription...")
                result = self.adapter.transcribe_recording()
                
                if result:
                    logger.info("Transcription complete")
                    return result
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        status = {
            'bluetooth': {
                'initialized': self.bluez_manager is not None,
                'service_state': self.bluez_manager.get_service_status() if self.bluez_manager else None,
                'phone_connected': self.phone_address is not None,
                'headphones_connected': self.headphone_address is not None
            },
            'audio': {
                'pipeline_initialized': self.audio_pipeline is not None,
                'pipeline_state': self.audio_pipeline.state if self.audio_pipeline else None,
                'metrics': self.audio_pipeline.metrics if self.audio_pipeline else None
            },
            'recording': {
                'adapter_initialized': self.adapter is not None,
                'recording_state': self.adapter.recording_state if self.adapter else None
            }
        }
        
        return status
    
    def run_interactive_demo(self):
        """Run interactive demo"""
        print("\n" + "="*60)
        print("The Silent Steno - Bluetooth Audio Pipeline Demo")
        print("="*60 + "\n")
        
        # Initialize Bluetooth
        print("1. Initializing Bluetooth...")
        if not self.initialize_bluetooth():
            print("❌ Failed to initialize Bluetooth")
            return
        print("✅ Bluetooth initialized")
        
        # Set up audio pipeline
        print("\n2. Setting up audio pipeline...")
        if not self.setup_audio_pipeline():
            print("❌ Failed to setup audio pipeline")
            return
        print("✅ Audio pipeline ready")
        
        # Scan for devices
        print("\n3. Scanning for Bluetooth devices...")
        devices = self.scan_devices()
        
        if not devices:
            print("❌ No devices found")
            print("\nPlease ensure your devices are in pairing mode and try again.")
            return
        
        # Main menu loop
        while True:
            print("\n" + "-"*40)
            print("Main Menu:")
            print("1. Connect to phone (A2DP Sink)")
            print("2. Connect to headphones (A2DP Source)")
            print("3. Start audio forwarding")
            print("4. Start recording")
            print("5. Stop recording and transcribe")
            print("6. Show status")
            print("7. Exit")
            print("-"*40)
            
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == '1':
                # Connect to phone
                print("\nAvailable devices:")
                for i, device in enumerate(devices):
                    print(f"{i+1}. {device['name']} ({device['address']})")
                
                idx = input("Select phone device number: ").strip()
                try:
                    device = devices[int(idx)-1]
                    if self.connect_phone(device['address']):
                        print("✅ Connected to phone")
                    else:
                        print("❌ Failed to connect to phone")
                except:
                    print("❌ Invalid selection")
            
            elif choice == '2':
                # Connect to headphones
                print("\nAvailable devices:")
                for i, device in enumerate(devices):
                    print(f"{i+1}. {device['name']} ({device['address']})")
                
                idx = input("Select headphone device number: ").strip()
                try:
                    device = devices[int(idx)-1]
                    if self.connect_headphones(device['address']):
                        print("✅ Connected to headphones")
                    else:
                        print("❌ Failed to connect to headphones")
                except:
                    print("❌ Invalid selection")
            
            elif choice == '3':
                # Start audio forwarding
                if self.start_audio_forwarding():
                    print("✅ Audio forwarding started")
                else:
                    print("❌ Failed to start audio forwarding")
            
            elif choice == '4':
                # Start recording
                session_id = self.start_recording()
                if session_id:
                    print(f"✅ Recording started: {session_id}")
                else:
                    print("❌ Failed to start recording")
            
            elif choice == '5':
                # Stop recording and transcribe
                result = self.stop_recording()
                if result:
                    print("✅ Recording stopped and transcribed")
                    if isinstance(result, dict):
                        print(f"\nTranscript: {result.get('transcript', 'No transcript')[:200]}...")
                        if 'analysis' in result:
                            analysis = result['analysis']
                            print(f"\nAnalysis:")
                            print(f"  - Word count: {analysis.get('word_count', 0)}")
                            print(f"  - Sentiment: {analysis.get('sentiment', 'neutral')}")
                            if analysis.get('action_items'):
                                print(f"  - Action items: {', '.join(analysis['action_items'][:3])}")
                else:
                    print("❌ Failed to stop recording")
            
            elif choice == '6':
                # Show status
                status = self.get_status()
                print("\nSystem Status:")
                print(f"  Bluetooth: {'✅' if status['bluetooth']['initialized'] else '❌'}")
                print(f"  Phone: {'✅' if status['bluetooth']['phone_connected'] else '❌'}")
                print(f"  Headphones: {'✅' if status['bluetooth']['headphones_connected'] else '❌'}")
                print(f"  Audio Pipeline: {'✅' if status['audio']['pipeline_initialized'] else '❌'}")
                print(f"  Recording: {status['recording']['recording_state'] or 'idle'}")
            
            elif choice == '7':
                # Exit
                print("\nShutting down...")
                if self.audio_pipeline:
                    self.audio_pipeline.stop()
                break
            
            else:
                print("❌ Invalid choice")
        
        print("\nDemo completed. Goodbye!")


def main():
    """Main entry point"""
    demo = BluetoothAudioDemo()
    
    try:
        demo.run_interactive_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"\n❌ Demo error: {e}")


if __name__ == "__main__":
    main()