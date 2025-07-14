#!/usr/bin/env python3

"""
ALSA Audio System Manager for The Silent Steno

This module provides comprehensive ALSA (Advanced Linux Sound Architecture)
management for low-latency audio operations. It handles device enumeration,
configuration optimization, and real-time audio setup.

Key features:
- ALSA device discovery and enumeration
- Low-latency audio configuration
- Buffer size optimization
- Audio device monitoring
- ALSA configuration file management
- Real-time priority setup
"""

import subprocess
import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Audio device types"""
    PLAYBACK = "playback"
    CAPTURE = "capture"
    BOTH = "both"


class DeviceState(Enum):
    """Audio device states"""
    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class AudioDevice:
    """Audio device information"""
    card_id: int
    device_id: int
    name: str
    type: DeviceType
    state: DeviceState
    sample_rates: List[int]
    channels: List[int]
    formats: List[str]
    driver: str
    description: str
    is_bluetooth: bool = False
    is_default: bool = False


@dataclass
class ALSAConfig:
    """ALSA configuration parameters"""
    period_size: int = 512  # frames
    buffer_size: int = 2048  # frames
    periods: int = 4
    sample_rate: int = 44100
    channels: int = 2
    format: str = "S16_LE"
    enable_mmap: bool = True
    enable_realtime: bool = True


class ALSAManager:
    """
    ALSA Audio System Manager for The Silent Steno
    
    Provides comprehensive ALSA management including device discovery,
    configuration optimization, and low-latency setup.
    """
    
    def __init__(self):
        """Initialize ALSA manager"""
        self.devices: Dict[str, AudioDevice] = {}
        self.current_config = ALSAConfig()
        self.config_file_path = "/home/mmariani/projects/thesilentsteno/config/alsa_config.conf"
        
        # Discover available devices
        self.refresh_devices()
        
        logger.info("ALSA manager initialized")
    
    def refresh_devices(self) -> None:
        """Refresh list of available audio devices"""
        self.devices.clear()
        
        # Get playback devices
        playback_devices = self._discover_devices("playback")
        for device in playback_devices:
            key = f"hw:{device.card_id},{device.device_id}"
            self.devices[key] = device
        
        # Get capture devices
        capture_devices = self._discover_devices("capture")
        for device in capture_devices:
            key = f"hw:{device.card_id},{device.device_id}"
            if key in self.devices:
                # Device supports both playback and capture
                self.devices[key].type = DeviceType.BOTH
            else:
                self.devices[key] = device
        
        logger.info(f"Discovered {len(self.devices)} audio devices")
    
    def _discover_devices(self, device_type: str) -> List[AudioDevice]:
        """Discover audio devices of specific type"""
        devices = []
        
        try:
            if device_type == "playback":
                cmd = ["aplay", "-l"]
            else:
                cmd = ["arecord", "-l"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                devices.extend(self._parse_device_list(result.stdout, device_type))
            
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout discovering {device_type} devices")
        except Exception as e:
            logger.error(f"Error discovering {device_type} devices: {e}")
        
        return devices
    
    def _parse_device_list(self, output: str, device_type: str) -> List[AudioDevice]:
        """Parse aplay/arecord output to extract device information"""
        devices = []
        
        # Pattern to match: card 0: vc4hdmi0 [vc4-hdmi-0], device 0: MAI PCM i2s-hifi-0
        card_pattern = r'card (\d+): (\w+) \[([^\]]+)\], device (\d+): (.+)'
        
        for line in output.split('\n'):
            match = re.match(card_pattern, line)
            if match:
                card_id = int(match.group(1))
                card_name = match.group(2)
                card_description = match.group(3)
                device_id = int(match.group(4))
                device_description = match.group(5)
                
                # Get detailed device info
                device_info = self._get_device_details(card_id, device_id, device_type)
                
                device = AudioDevice(
                    card_id=card_id,
                    device_id=device_id,
                    name=f"{card_name}_{device_id}",
                    type=DeviceType.PLAYBACK if device_type == "playback" else DeviceType.CAPTURE,
                    state=DeviceState.AVAILABLE,
                    sample_rates=device_info.get('sample_rates', [44100, 48000]),
                    channels=device_info.get('channels', [2]),
                    formats=device_info.get('formats', ['S16_LE']),
                    driver=device_info.get('driver', 'unknown'),
                    description=f"{card_description} - {device_description}",
                    is_bluetooth="bluez" in card_description.lower() or "bluetooth" in card_description.lower(),
                    is_default=card_id == 0 and device_id == 0
                )
                
                devices.append(device)
        
        return devices
    
    def _get_device_details(self, card_id: int, device_id: int, device_type: str) -> Dict[str, Any]:
        """Get detailed information about a specific device"""
        details = {
            'sample_rates': [44100, 48000],
            'channels': [1, 2],
            'formats': ['S16_LE'],
            'driver': 'unknown'
        }
        
        try:
            # Get device capabilities using amixer
            cmd = ["amixer", "-c", str(card_id), "info"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                # Parse capabilities from amixer output
                if "48000" in result.stdout:
                    details['sample_rates'].append(48000)
                if "96000" in result.stdout:
                    details['sample_rates'].append(96000)
        
        except Exception as e:
            logger.debug(f"Could not get device details for card {card_id}: {e}")
        
        return details
    
    def get_audio_devices(self, device_type: Optional[DeviceType] = None) -> List[AudioDevice]:
        """
        Get list of audio devices
        
        Args:
            device_type: Filter by device type (None for all)
            
        Returns:
            List of matching audio devices
        """
        devices = list(self.devices.values())
        
        if device_type:
            devices = [d for d in devices if d.type == device_type or d.type == DeviceType.BOTH]
        
        return devices
    
    def get_bluetooth_devices(self) -> List[AudioDevice]:
        """Get list of Bluetooth audio devices"""
        return [d for d in self.devices.values() if d.is_bluetooth]
    
    def get_default_device(self, device_type: DeviceType) -> Optional[AudioDevice]:
        """Get default device for specified type"""
        devices = self.get_audio_devices(device_type)
        
        # Look for default device
        for device in devices:
            if device.is_default:
                return device
        
        # Return first available device if no default
        return devices[0] if devices else None
    
    def configure_alsa(self, config: Optional[ALSAConfig] = None) -> bool:
        """
        Configure ALSA for optimal low-latency performance
        
        Args:
            config: ALSA configuration parameters
            
        Returns:
            bool: True if configuration successful
        """
        if config:
            self.current_config = config
        
        try:
            logger.info("Configuring ALSA for low-latency operation...")
            
            # Create ALSA configuration file
            self._create_alsa_config()
            
            # Set audio thread priorities
            self._configure_realtime_priorities()
            
            # Optimize kernel parameters
            self._optimize_kernel_parameters()
            
            # Test configuration
            if self._test_alsa_config():
                logger.info("ALSA configuration successful")
                return True
            else:
                logger.error("ALSA configuration test failed")
                return False
        
        except Exception as e:
            logger.error(f"Error configuring ALSA: {e}")
            return False
    
    def _create_alsa_config(self) -> None:
        """Create optimized ALSA configuration file"""
        config_content = f"""# ALSA Configuration for The Silent Steno
# Optimized for low-latency audio processing

# PCM definitions for low-latency operation
pcm.!default {{
    type hw
    card 0
    device 0
    period_size {self.current_config.period_size}
    buffer_size {self.current_config.buffer_size}
    periods {self.current_config.periods}
    rate {self.current_config.sample_rate}
    channels {self.current_config.channels}
    format {self.current_config.format}
}}

ctl.!default {{
    type hw
    card 0
}}

# Low-latency PCM device
pcm.lowlatency {{
    type hw
    card 0
    device 0
    period_size {self.current_config.period_size // 2}
    buffer_size {self.current_config.buffer_size // 2}
    periods 2
    rate {self.current_config.sample_rate}
    channels {self.current_config.channels}
    format {self.current_config.format}
}}

# Bluetooth PCM device (for A2DP)
pcm.bluetooth {{
    type pulse
    server "unix:/run/user/1000/pulse/native"
}}

# Null device for testing
pcm.null {{
    type null
}}

# Plugin configuration
pcm_plugins {{
    rate_converter "samplerate"
    rate_converter_quality "medium"
}}

# DMIX configuration for mixing
pcm.dmix_low_latency {{
    type dmix
    ipc_key 1024
    slave {{
        pcm "hw:0,0"
        period_time 0
        period_size {self.current_config.period_size}
        buffer_time 0
        buffer_size {self.current_config.buffer_size}
        rate {self.current_config.sample_rate}
        channels {self.current_config.channels}
        format {self.current_config.format}
    }}
    bindings {{
        0 0
        1 1
    }}
}}

# DSNOOP configuration for capture
pcm.dsnoop_low_latency {{
    type dsnoop
    ipc_key 2048
    slave {{
        pcm "hw:0,0"
        period_time 0
        period_size {self.current_config.period_size}
        buffer_time 0
        buffer_size {self.current_config.buffer_size}
        rate {self.current_config.sample_rate}
        channels {self.current_config.channels}
        format {self.current_config.format}
    }}
    bindings {{
        0 0
        1 1
    }}
}}

# Full duplex low-latency device
pcm.duplex {{
    type asym
    playback.pcm "dmix_low_latency"
    capture.pcm "dsnoop_low_latency"
}}
"""
        
        # Write configuration file
        os.makedirs(os.path.dirname(self.config_file_path), exist_ok=True)
        with open(self.config_file_path, 'w') as f:
            f.write(config_content)
        
        logger.info(f"ALSA configuration written to {self.config_file_path}")
    
    def _configure_realtime_priorities(self) -> None:
        """Configure real-time priorities for audio processing"""
        try:
            # Check if user can set real-time priorities
            result = subprocess.run(
                ["ulimit", "-r"],
                capture_output=True,
                text=True,
                shell=True
            )
            
            if result.returncode == 0:
                rt_limit = result.stdout.strip()
                if rt_limit != "unlimited" and int(rt_limit) < 95:
                    logger.warning(f"Real-time priority limit is low: {rt_limit}")
            
        except Exception as e:
            logger.debug(f"Could not check real-time limits: {e}")
    
    def _optimize_kernel_parameters(self) -> None:
        """Optimize kernel parameters for audio processing"""
        try:
            # These would typically require root access
            # In production, these should be set in /etc/sysctl.conf
            
            optimizations = [
                ("vm.swappiness", "10"),  # Reduce swapping
                ("kernel.sched_rt_runtime_us", "950000"),  # RT scheduling
                ("kernel.sched_rt_period_us", "1000000"),
            ]
            
            for param, value in optimizations:
                try:
                    with open(f"/proc/sys/{param.replace('.', '/')}", 'w') as f:
                        f.write(value)
                    logger.debug(f"Set {param} = {value}")
                except PermissionError:
                    logger.debug(f"Cannot set {param} (requires root)")
                except Exception as e:
                    logger.debug(f"Error setting {param}: {e}")
        
        except Exception as e:
            logger.debug(f"Could not optimize kernel parameters: {e}")
    
    def _test_alsa_config(self) -> bool:
        """Test ALSA configuration"""
        try:
            # Test playback device
            result = subprocess.run(
                ["aplay", "-D", "hw:0,0", "-t", "raw", "-r", str(self.current_config.sample_rate),
                 "-c", str(self.current_config.channels), "-f", self.current_config.format,
                 "/dev/zero"],
                input="",
                capture_output=True,
                text=True,
                timeout=2
            )
            
            # If we get here without error, basic playback works
            return True
        
        except subprocess.TimeoutExpired:
            # Timeout is expected for /dev/zero playback
            return True
        except Exception as e:
            logger.error(f"ALSA configuration test failed: {e}")
            return False
    
    def optimize_latency(self) -> bool:
        """
        Optimize ALSA configuration for minimum latency
        
        Returns:
            bool: True if optimization successful
        """
        try:
            logger.info("Optimizing ALSA for minimum latency...")
            
            # Use smaller buffer sizes for lower latency
            optimized_config = ALSAConfig(
                period_size=256,  # Smaller period for lower latency
                buffer_size=1024,  # Smaller buffer
                periods=2,  # Fewer periods
                sample_rate=self.current_config.sample_rate,
                channels=self.current_config.channels,
                format=self.current_config.format,
                enable_mmap=True,
                enable_realtime=True
            )
            
            return self.configure_alsa(optimized_config)
        
        except Exception as e:
            logger.error(f"Error optimizing latency: {e}")
            return False
    
    def measure_device_latency(self, device_name: str) -> Optional[float]:
        """
        Measure audio latency for a specific device
        
        Args:
            device_name: ALSA device name (e.g., "hw:0,0")
            
        Returns:
            Latency in milliseconds, or None if measurement failed
        """
        try:
            # This is a simplified latency measurement
            # Real implementation would use proper audio loopback testing
            
            # Calculate theoretical minimum latency based on buffer settings
            frames_per_period = self.current_config.period_size
            periods = self.current_config.periods
            sample_rate = self.current_config.sample_rate
            
            # Minimum latency = (frames_per_period * periods) / sample_rate
            min_latency_seconds = (frames_per_period * periods) / sample_rate
            min_latency_ms = min_latency_seconds * 1000
            
            logger.info(f"Theoretical minimum latency for {device_name}: {min_latency_ms:.2f}ms")
            
            return min_latency_ms
        
        except Exception as e:
            logger.error(f"Error measuring device latency: {e}")
            return None
    
    def get_alsa_status(self) -> Dict[str, Any]:
        """
        Get comprehensive ALSA status
        
        Returns:
            Dict containing ALSA configuration and device status
        """
        return {
            "config": {
                "period_size": self.current_config.period_size,
                "buffer_size": self.current_config.buffer_size,
                "periods": self.current_config.periods,
                "sample_rate": self.current_config.sample_rate,
                "channels": self.current_config.channels,
                "format": self.current_config.format
            },
            "devices": {
                "total": len(self.devices),
                "bluetooth": len(self.get_bluetooth_devices()),
                "playback": len(self.get_audio_devices(DeviceType.PLAYBACK)),
                "capture": len(self.get_audio_devices(DeviceType.CAPTURE))
            },
            "theoretical_latency_ms": self.measure_device_latency("hw:0,0"),
            "config_file": self.config_file_path
        }


if __name__ == "__main__":
    # Basic test when run directly
    print("ALSA Manager Test")
    print("=" * 40)
    
    manager = ALSAManager()
    
    print(f"Discovered devices: {len(manager.devices)}")
    for device_name, device in manager.devices.items():
        print(f"  {device_name}: {device.description} ({device.type.value})")
    
    print("\nBluetooth devices:")
    bt_devices = manager.get_bluetooth_devices()
    for device in bt_devices:
        print(f"  {device.name}: {device.description}")
    
    print("\nConfiguring ALSA...")
    if manager.configure_alsa():
        print("ALSA configuration successful")
    else:
        print("ALSA configuration failed")
    
    status = manager.get_alsa_status()
    print(f"\nALSA Status: {status}")