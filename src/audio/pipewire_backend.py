#!/usr/bin/env python3
"""
PipeWire Audio Backend for The Silent Steno

Provides PipeWire-based audio management for low-latency Bluetooth audio forwarding.
This replaces the PulseAudio backend while maintaining compatibility.
"""

import subprocess
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import re

logger = logging.getLogger(__name__)


class AudioState(Enum):
    """Audio device states"""
    IDLE = "idle"
    RUNNING = "running"
    SUSPENDED = "suspended"
    ERROR = "error"


class DeviceDirection(Enum):
    """Audio device direction"""
    SOURCE = "source"  # Input device (microphone, A2DP source)
    SINK = "sink"      # Output device (speakers, A2DP sink)


@dataclass
class AudioDevice:
    """Represents a PipeWire audio device"""
    id: int
    name: str
    description: str
    direction: DeviceDirection
    state: AudioState
    sample_rate: int = 44100
    channels: int = 2
    latency: float = 0.0
    is_bluetooth: bool = False
    bluetooth_address: Optional[str] = None
    codec: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioLink:
    """Represents an audio link between devices"""
    id: int
    source_port: str
    sink_port: str
    state: str
    latency: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)


class PipeWireBackend:
    """PipeWire audio backend implementation"""
    
    def __init__(self):
        self.devices: Dict[int, AudioDevice] = {}
        self.links: Dict[int, AudioLink] = {}
        self._check_pipewire()
        
    def _check_pipewire(self) -> bool:
        """Check if PipeWire is running"""
        try:
            result = subprocess.run(
                ['pw-cli', 'info'],
                capture_output=True,
                timeout=2
            )
            if result.returncode != 0:
                raise RuntimeError("PipeWire not running")
            return True
        except Exception as e:
            logger.error(f"PipeWire check failed: {e}")
            raise
            
    def _run_pw_command(self, args: List[str], json_output: bool = True) -> Any:
        """Run a pw-cli command and return output"""
        cmd = ['pw-cli'] + args
        if json_output:
            cmd.append('-m')  # Machine readable output
            
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                logger.error(f"Command failed: {' '.join(cmd)}")
                logger.error(f"Error: {result.stderr}")
                return None
                
            if json_output and result.stdout:
                # Parse JSON output
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    # Try line-by-line JSON parsing for pw-dump output
                    lines = result.stdout.strip().split('\n')
                    return [json.loads(line) for line in lines if line]
                    
            return result.stdout
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(cmd)}")
            return None
        except Exception as e:
            logger.error(f"Command error: {e}")
            return None
            
    def refresh_devices(self) -> Dict[int, AudioDevice]:
        """Refresh the list of audio devices"""
        self.devices.clear()
        
        # Get all PipeWire objects
        objects = self._run_pw_command(['dump'])
        if not objects:
            return self.devices
            
        for obj in objects:
            # Filter for audio devices
            if obj.get('type') != 'PipeWire:Interface:Node':
                continue
                
            props = obj.get('info', {}).get('props', {})
            
            # Check if it's an audio device
            media_class = props.get('media.class', '')
            if not any(x in media_class for x in ['Audio/Source', 'Audio/Sink']):
                continue
                
            # Create AudioDevice object
            device = AudioDevice(
                id=obj.get('id', 0),
                name=props.get('node.name', ''),
                description=props.get('node.description', props.get('node.nick', '')),
                direction=DeviceDirection.SOURCE if 'Source' in media_class else DeviceDirection.SINK,
                state=self._parse_state(obj.get('info', {}).get('state', 'idle')),
                sample_rate=props.get('audio.rate', 44100),
                channels=props.get('audio.channels', 2),
                properties=props
            )
            
            # Check if Bluetooth device
            if 'bluez' in device.name or 'bluetooth' in device.name.lower():
                device.is_bluetooth = True
                device.bluetooth_address = self._extract_bluetooth_address(device.name)
                device.codec = props.get('api.bluez5.codec', 'unknown')
                
            # Get latency info
            params = obj.get('info', {}).get('params', {})
            if 'Props' in params:
                for prop in params['Props']:
                    if isinstance(prop, dict) and 'latency' in str(prop):
                        # Extract latency value
                        pass
                        
            self.devices[device.id] = device
            
        return self.devices
        
    def _parse_state(self, state_str: str) -> AudioState:
        """Parse PipeWire state string to AudioState enum"""
        state_map = {
            'idle': AudioState.IDLE,
            'running': AudioState.RUNNING,
            'suspended': AudioState.SUSPENDED,
            'error': AudioState.ERROR
        }
        return state_map.get(state_str.lower(), AudioState.ERROR)
        
    def _extract_bluetooth_address(self, name: str) -> Optional[str]:
        """Extract Bluetooth MAC address from device name"""
        # Look for pattern like bluez_card.XX_XX_XX_XX_XX_XX
        match = re.search(r'([0-9A-F]{2}_){5}[0-9A-F]{2}', name, re.IGNORECASE)
        if match:
            return match.group(0).replace('_', ':')
        return None
        
    def get_bluetooth_devices(self) -> Dict[int, AudioDevice]:
        """Get only Bluetooth audio devices"""
        self.refresh_devices()
        return {
            dev_id: device
            for dev_id, device in self.devices.items()
            if device.is_bluetooth
        }
        
    def get_sources(self) -> Dict[int, AudioDevice]:
        """Get all audio source devices"""
        return {
            dev_id: device
            for dev_id, device in self.devices.items()
            if device.direction == DeviceDirection.SOURCE
        }
        
    def get_sinks(self) -> Dict[int, AudioDevice]:
        """Get all audio sink devices"""
        return {
            dev_id: device
            for dev_id, device in self.devices.items()
            if device.direction == DeviceDirection.SINK
        }
        
    def create_loopback(self, source_id: int, sink_id: int, 
                       latency_ms: int = 40) -> Optional[int]:
        """Create audio loopback between source and sink"""
        source = self.devices.get(source_id)
        sink = self.devices.get(sink_id)
        
        if not source or not sink:
            logger.error(f"Invalid device IDs: source={source_id}, sink={sink_id}")
            return None
            
        # Create link using pw-link
        cmd = [
            'pw-link',
            '--id', str(source_id),
            '--id', str(sink_id),
            '--linger',  # Keep link when creating app exits
            '--props',
            f'target.object={sink_id},'
            f'node.latency={latency_ms}/1000,'
            f'node.autoconnect=true'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Extract link ID from output
            try:
                link_id = int(result.stdout.strip())
                logger.info(f"Created loopback link {link_id}: {source.name} -> {sink.name}")
                return link_id
            except ValueError:
                logger.warning("Created link but couldn't get ID")
                return -1
        else:
            logger.error(f"Failed to create loopback: {result.stderr}")
            return None
            
    def remove_loopback(self, link_id: int) -> bool:
        """Remove audio loopback link"""
        cmd = ['pw-link', '--disconnect', str(link_id)]
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode == 0:
            logger.info(f"Removed loopback link {link_id}")
            return True
        else:
            logger.error(f"Failed to remove link {link_id}")
            return False
            
    def set_default_source(self, device_id: int) -> bool:
        """Set default audio source"""
        return self._set_default_device(device_id, 'default.audio.source')
        
    def set_default_sink(self, device_id: int) -> bool:
        """Set default audio sink"""
        return self._set_default_device(device_id, 'default.audio.sink')
        
    def _set_default_device(self, device_id: int, property: str) -> bool:
        """Set default device using metadata"""
        cmd = [
            'pw-metadata', '0',
            property, str(device_id),
            '{"name": "' + self.devices[device_id].name + '"}'
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0
        
    def get_device_info(self, device_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed device information"""
        cmd = ['pw-cli', 'info', str(device_id)]
        output = self._run_pw_command(['info', str(device_id)], json_output=False)
        
        if output:
            # Parse the text output into a dict
            info = {}
            current_section = None
            
            for line in output.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if line.endswith(':'):
                    current_section = line[:-1]
                    info[current_section] = {}
                elif '=' in line and current_section:
                    key, value = line.split('=', 1)
                    info[current_section][key.strip()] = value.strip().strip('"')
                    
            return info
            
        return None
        
    def monitor_devices(self, callback: callable, duration: Optional[int] = None):
        """Monitor device changes"""
        # This would use pw-mon for real-time monitoring
        # For now, just poll periodically
        import threading
        
        def monitor_loop():
            start_time = time.time()
            previous_devices = set()
            
            while True:
                current_devices = set(self.refresh_devices().keys())
                
                # Check for changes
                added = current_devices - previous_devices
                removed = previous_devices - current_devices
                
                if added or removed:
                    callback({
                        'added': list(added),
                        'removed': list(removed),
                        'devices': self.devices
                    })
                    
                previous_devices = current_devices
                
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    break
                    
                time.sleep(1)  # Poll every second
                
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        
    def get_latency_info(self) -> Dict[str, Any]:
        """Get system-wide latency information"""
        info = {
            'quantum': None,
            'rate': None,
            'devices': {}
        }
        
        # Get global settings
        settings = self._run_pw_command(['settings'], json_output=False)
        if settings:
            for line in settings.split('\n'):
                if 'clock.quantum' in line:
                    try:
                        info['quantum'] = int(line.split('=')[1].strip())
                    except:
                        pass
                elif 'clock.rate' in line:
                    try:
                        info['rate'] = int(line.split('=')[1].strip())
                    except:
                        pass
                        
        # Calculate latency for each device
        for device in self.devices.values():
            if info['quantum'] and info['rate']:
                device_latency_ms = (info['quantum'] / info['rate']) * 1000
                info['devices'][device.name] = {
                    'latency_ms': device_latency_ms,
                    'state': device.state.value
                }
                
        return info
        
    def optimize_latency(self, target_latency_ms: int = 40) -> bool:
        """Optimize system for target latency"""
        # Calculate quantum for target latency
        rate = 44100  # Default sample rate
        quantum = int((target_latency_ms / 1000) * rate)
        
        # Ensure quantum is power of 2 for efficiency
        quantum = 2 ** (quantum.bit_length() - 1)
        quantum = max(64, min(2048, quantum))  # Clamp to reasonable range
        
        # Set using pw-metadata
        cmd = [
            'pw-metadata', '0',
            'clock.force-quantum', str(quantum)
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode == 0:
            logger.info(f"Set quantum to {quantum} for ~{target_latency_ms}ms latency")
            return True
        else:
            logger.error(f"Failed to set quantum: {result.stderr}")
            return False


# Compatibility layer for PulseAudio commands
class PulseAudioCompat:
    """Provides PulseAudio-compatible interface using PipeWire"""
    
    def __init__(self, backend: PipeWireBackend):
        self.backend = backend
        
    def list_sources(self) -> List[Dict[str, Any]]:
        """List sources in PulseAudio format"""
        sources = []
        for device in self.backend.get_sources().values():
            sources.append({
                'index': device.id,
                'name': device.name,
                'description': device.description,
                'state': device.state.value.upper(),
                'sample_spec': f"{device.sample_rate}Hz {device.channels}ch",
                'bluetooth': device.is_bluetooth
            })
        return sources
        
    def list_sinks(self) -> List[Dict[str, Any]]:
        """List sinks in PulseAudio format"""
        sinks = []
        for device in self.backend.get_sinks().values():
            sinks.append({
                'index': device.id,
                'name': device.name,
                'description': device.description,
                'state': device.state.value.upper(),
                'sample_spec': f"{device.sample_rate}Hz {device.channels}ch",
                'bluetooth': device.is_bluetooth
            })
        return sinks
        
    def load_module_loopback(self, source: str, sink: str, 
                           latency_msec: int = 40) -> Optional[int]:
        """Load loopback module (PulseAudio compatibility)"""
        # Find devices by name
        source_dev = None
        sink_dev = None
        
        for device in self.backend.devices.values():
            if device.name == source:
                source_dev = device
            elif device.name == sink:
                sink_dev = device
                
        if source_dev and sink_dev:
            return self.backend.create_loopback(
                source_dev.id, 
                sink_dev.id, 
                latency_msec
            )
        
        return None


def create_backend() -> PipeWireBackend:
    """Factory function to create PipeWire backend"""
    return PipeWireBackend()


if __name__ == "__main__":
    # Test the backend
    logging.basicConfig(level=logging.INFO)
    
    backend = create_backend()
    print("PipeWire Backend Test")
    print("=" * 50)
    
    # List devices
    backend.refresh_devices()
    
    print("\nAudio Sources:")
    for device in backend.get_sources().values():
        print(f"  [{device.id}] {device.name}")
        print(f"      Description: {device.description}")
        print(f"      State: {device.state.value}")
        if device.is_bluetooth:
            print(f"      Bluetooth: {device.bluetooth_address} (Codec: {device.codec})")
            
    print("\nAudio Sinks:")
    for device in backend.get_sinks().values():
        print(f"  [{device.id}] {device.name}")
        print(f"      Description: {device.description}")
        print(f"      State: {device.state.value}")
        if device.is_bluetooth:
            print(f"      Bluetooth: {device.bluetooth_address} (Codec: {device.codec})")
            
    # Test latency info
    print("\nLatency Information:")
    latency = backend.get_latency_info()
    print(f"  Quantum: {latency['quantum']}")
    print(f"  Rate: {latency['rate']}")
    
    # Test PulseAudio compatibility
    print("\nPulseAudio Compatibility Test:")
    compat = PulseAudioCompat(backend)
    sources = compat.list_sources()
    print(f"  Found {len(sources)} sources")
    sinks = compat.list_sinks()  
    print(f"  Found {len(sinks)} sinks")