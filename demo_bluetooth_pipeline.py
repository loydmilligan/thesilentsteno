#!/usr/bin/env python3
"""
The Silent Steno - Enhanced Bluetooth Audio Pipeline Demo with Multi-Source Support

This demo shows how to set up the Pi as a Bluetooth audio proxy that:
1. Receives audio from multiple sources (phone, PC, tablet, etc.)
2. Allows switching between audio sources
3. Forwards audio to your headphones (A2DP Source)
4. Records and transcribes the audio in real-time
"""

import sys
import os
import time
import logging
import threading
import subprocess
from typing import Optional, Dict, Any, List

# Add project root to path
sys.path.append('/home/mmariani/projects/thesilentsteno')

# Import our components
from src.bluetooth.bluez_manager import BlueZManager
from src.bluetooth.connection_manager import ConnectionManager, DeviceRole, ConnectionState
from src.audio.audio_pipeline import AudioPipeline, AudioConfig, PipelineState
from src.integration.walking_skeleton_adapter import WalkingSkeletonAdapter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioSourceManager:
    """Manages multiple audio source devices (phone, PC, tablet, etc.)"""
    
    def __init__(self):
        self.paired_sources = {}  # MAC -> device info
        self.active_source = None
        self.current_loopback_module = None
        self.headphone_address = None
        
    def add_source_device(self, mac_address: str, device_name: str, device_type: str):
        """Add a device as potential audio source"""
        self.paired_sources[mac_address] = {
            'name': device_name,
            'type': device_type,  # 'phone', 'pc', 'tablet', 'laptop'
            'connected': False,
            'mac': mac_address
        }
        logger.info(f"Added audio source: {device_name} ({device_type}) - {mac_address}")
    
    def set_headphone_address(self, mac_address: str):
        """Set the target headphone device"""
        self.headphone_address = mac_address
        logger.info(f"Set headphone target: {mac_address}")
    
    def switch_to_source(self, mac_address: str) -> bool:
        """Switch audio input to specified device"""
        if mac_address not in self.paired_sources:
            logger.error(f"Unknown source device: {mac_address}")
            return False
            
        if not self.headphone_address:
            logger.error("No headphone address set")
            return False
        
        try:
            # Remove existing loopback
            if self.current_loopback_module:
                logger.info(f"Removing existing loopback module: {self.current_loopback_module}")
                subprocess.run(f"pactl unload-module {self.current_loopback_module}", 
                             shell=True, check=False)
                self.current_loopback_module = None
            
            # Create new loopback
            source = f"bluez_source.{mac_address.replace(':', '_')}.a2dp_source"
            sink = f"bluez_sink.{self.headphone_address.replace(':', '_')}.a2dp_sink"
            
            logger.info(f"Creating loopback: {source} -> {sink}")
            
            result = subprocess.run(
                f"pactl load-module module-loopback source={source} sink={sink} latency_msec=40",
                shell=True, capture_output=True, text=True, check=True
            )
            
            self.current_loopback_module = result.stdout.strip()
            self.active_source = mac_address
            
            device_info = self.paired_sources[mac_address]
            logger.info(f"✅ Switched to audio source: {device_info['name']} ({device_info['type']})")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create loopback: {e}")
            return False
    
    def stop_audio_forwarding(self) -> bool:
        """Stop current audio forwarding"""
        if self.current_loopback_module:
            try:
                subprocess.run(f"pactl unload-module {self.current_loopback_module}", 
                             shell=True, check=True)
                logger.info("Audio forwarding stopped")
                self.current_loopback_module = None
                self.active_source = None
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to stop audio forwarding: {e}")
                return False
        return True
    
    def get_available_sources(self) -> List[Dict]:
        """Get list of connected devices that can be audio sources"""
        available = []
        
        # Check which sources are actually available in PulseAudio
        try:
            result = subprocess.run("pactl list sources short", 
                                  shell=True, capture_output=True, text=True)
            available_sources = result.stdout
            
            for mac, info in self.paired_sources.items():
                source_name = f"bluez_source.{mac.replace(':', '_')}.a2dp_source"
                if source_name in available_sources:
                    available.append({
                        'mac': mac,
                        'name': info['name'],
                        'type': info['type'],
                        'active': mac == self.active_source,
                        'connected': True
                    })
                else:
                    available.append({
                        'mac': mac,
                        'name': info['name'],
                        'type': info['type'],
                        'active': False,
                        'connected': False
                    })
                    
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get available sources: {e}")
            
        return available
    
    def auto_detect_active_source(self) -> Optional[str]:
        """Automatically detect which device is sending audio"""
        try:
            result = subprocess.run("pactl list source-outputs short", 
                                  shell=True, capture_output=True, text=True)
            
            for line in result.stdout.split('\n'):
                if 'bluez_source' in line and len(line.strip()) > 0:
                    # Extract MAC from source name
                    parts = line.split()
                    if len(parts) >= 2:
                        source_name = parts[1]
                        # Extract MAC from bluez_source.XX_XX_XX_XX_XX_XX.a2dp_source
                        if 'bluez_source.' in source_name:
                            mac_part = source_name.split('.')[1]
                            mac = mac_part.replace('_', ':')
                            if mac in self.paired_sources:
                                return mac
                                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to auto-detect active source: {e}")
            
        return None


class EnhancedBluetoothAudioDemo:
    """Enhanced demo class with multi-source support"""
    
    def __init__(self):
        """Initialize demo components"""
        self.bluez_manager = None
        self.connection_manager = None
        self.audio_pipeline = None
        self.adapter = None
        self.source_manager = AudioSourceManager()
        self.is_running = False
        
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
        
            # Simple success check - if we got this far, Bluetooth is working
            logger.info("Bluetooth service appears to be running")
            return True
        
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
                buffer_size=256,  # Increased from 128 for stability
                target_latency_ms=50.0,  # More realistic target
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
    
    def add_audio_source(self, address: str, name: str, device_type: str) -> bool:
        """Add a device as audio source"""
        try:
            # Pair if needed
            if not self.connection_manager.is_paired(address):
                logger.info(f"Pairing with {name}...")
                if not self.connection_manager.pair_device(address, name, DeviceRole.SOURCE):
                    logger.error(f"Failed to pair with {name}")
                    return False
            
            # Connect the device
            if self.connection_manager.connect_device(address):
                self.source_manager.add_source_device(address, name, device_type)
                logger.info(f"✅ Added audio source: {name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to add audio source: {e}")
            return False
    
    def set_headphones(self, address: str, name: str) -> bool:
        """Set headphones as audio output"""
        try:
            # Pair if needed
            if not self.connection_manager.is_paired(address):
                logger.info(f"Pairing with {name}...")
                if not self.connection_manager.pair_device(address, name, DeviceRole.SINK):
                    logger.error(f"Failed to pair with {name}")
                    return False
            
            # Connect the device
            if self.connection_manager.connect_device(address):
                self.source_manager.set_headphone_address(address)
                logger.info(f"✅ Set headphones: {name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to set headphones: {e}")
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
            },
            'audio': {
                'pipeline_initialized': self.audio_pipeline is not None,
                'pipeline_state': self.audio_pipeline.state if self.audio_pipeline else None,
                'current_source': self.source_manager.active_source,
                'available_sources': self.source_manager.get_available_sources()
            },
            'recording': {
                'adapter_initialized': self.adapter is not None,
                'recording_state': self.adapter.recording_state if self.adapter else None
            }
        }
        
        return status
    
    def run_interactive_demo(self):
        """Run enhanced interactive demo with multi-source support"""
        print("\n" + "="*70)
        print("The Silent Steno - Enhanced Multi-Source Audio Pipeline Demo")
        print("="*70 + "\n")
        
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
        
        # Main menu loop
        while True:
            print("\n" + "-"*50)
            print("Main Menu:")
            print("1. Scan for devices")
            print("2. Add audio source (phone/PC/tablet)")
            print("3. Set headphones")
            print("4. Switch audio source")
            print("5. Show available sources")
            print("6. Start recording")
            print("7. Stop recording and transcribe")
            print("8. Show status")
            print("9. Stop audio forwarding")
            print("0. Exit")
            print("-"*50)
            
            choice = input("\nEnter your choice (0-9): ").strip()
            
            if choice == '1':
                # Scan for devices
                print("\nScanning for devices...")
                devices = self.scan_devices()
                if devices:
                    print("\nFound devices:")
                    for i, device in enumerate(devices):
                        print(f"{i+1}. {device['name']} ({device['address']})")
                else:
                    print("No devices found")
            
            elif choice == '2':
                # Add audio source
                devices = self.scan_devices()
                if devices:
                    print("\nAvailable devices:")
                    for i, device in enumerate(devices):
                        print(f"{i+1}. {device['name']} ({device['address']})")
                    
                    try:
                        idx = int(input("Select device number: ")) - 1
                        device = devices[idx]
                        
                        print("\nDevice type:")
                        print("1. Phone")
                        print("2. PC/Computer")
                        print("3. Laptop")
                        print("4. Tablet")
                        print("5. Other")
                        
                        type_choice = input("Select type (1-5): ").strip()
                        type_map = {'1': 'phone', '2': 'pc', '3': 'laptop', '4': 'tablet', '5': 'other'}
                        device_type = type_map.get(type_choice, 'other')
                        
                        if self.add_audio_source(device['address'], device['name'], device_type):
                            print(f"✅ Added {device['name']} as audio source")
                        else:
                            print(f"❌ Failed to add {device['name']}")
                    except:
                        print("❌ Invalid selection")
                else:
                    print("❌ No devices available")
            
            elif choice == '3':
                # Set headphones
                devices = self.scan_devices()
                if devices:
                    print("\nAvailable devices:")
                    for i, device in enumerate(devices):
                        print(f"{i+1}. {device['name']} ({device['address']})")
                    
                    try:
                        idx = int(input("Select headphone device number: ")) - 1
                        device = devices[idx]
                        
                        if self.set_headphones(device['address'], device['name']):
                            print(f"✅ Set {device['name']} as headphones")
                        else:
                            print(f"❌ Failed to set {device['name']}")
                    except:
                        print("❌ Invalid selection")
                else:
                    print("❌ No devices available")
            
            elif choice == '4':
                # Switch audio source
                sources = self.source_manager.get_available_sources()
                connected_sources = [s for s in sources if s['connected']]
                
                if connected_sources:
                    print("\nAvailable audio sources:")
                    for i, source in enumerate(connected_sources):
                        status = "🔊 ACTIVE" if source['active'] else "⚪ Available"
                        print(f"{i+1}. {source['name']} ({source['type']}) - {status}")
                    
                    try:
                        idx = int(input("Select source number: ")) - 1
                        source = connected_sources[idx]
                        
                        if self.source_manager.switch_to_source(source['mac']):
                            print(f"✅ Switched to {source['name']}")
                        else:
                            print(f"❌ Failed to switch to {source['name']}")
                    except:
                        print("❌ Invalid selection")
                else:
                    print("❌ No connected audio sources available")
            
            elif choice == '5':
                # Show available sources
                sources = self.source_manager.get_available_sources()
                if sources:
                    print("\nConfigured audio sources:")
                    for source in sources:
                        status = "🔊 ACTIVE" if source['active'] else ("✅ Connected" if source['connected'] else "❌ Disconnected")
                        print(f"  - {source['name']} ({source['type']}) - {status}")
                else:
                    print("❌ No audio sources configured")
            
            elif choice == '6':
                # Start recording
                session_id = self.start_recording()
                if session_id:
                    print(f"✅ Recording started: {session_id}")
                else:
                    print("❌ Failed to start recording")
            
            elif choice == '7':
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
            
            elif choice == '8':
                # Show status
                status = self.get_status()
                print("\nSystem Status:")
                print(f"  Bluetooth: {'✅' if status['bluetooth']['initialized'] else '❌'}")
                print(f"  Audio Pipeline: {'✅' if status['audio']['pipeline_initialized'] else '❌'}")
                print(f"  Active Source: {status['audio']['current_source'] or 'None'}")
                print(f"  Available Sources: {len(status['audio']['available_sources'])}")
                print(f"  Recording: {status['recording']['recording_state'] or 'idle'}")
            
            elif choice == '9':
                # Stop audio forwarding
                if self.source_manager.stop_audio_forwarding():
                    print("✅ Audio forwarding stopped")
                else:
                    print("❌ Failed to stop audio forwarding")
            
            elif choice == '0':
                # Exit
                print("\nShutting down...")
                self.source_manager.stop_audio_forwarding()
                if self.audio_pipeline:
                    self.audio_pipeline.stop()
                break
            
            else:
                print("❌ Invalid choice")
        
        print("\nDemo completed. Goodbye!")


def main():
    """Main entry point"""
    demo = EnhancedBluetoothAudioDemo()
    
    try:
        demo.run_interactive_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"\n❌ Demo error: {e}")


if __name__ == "__main__":
    main()
