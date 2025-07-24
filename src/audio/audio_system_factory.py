#!/usr/bin/env python3
"""
Audio System Factory for The Silent Steno

Detects available audio systems (PipeWire or PulseAudio) and provides
the appropriate backend implementation.
"""

import subprocess
import logging
from typing import Optional, Dict, Any, Protocol
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AudioSystemType(Enum):
    """Available audio system types"""
    PIPEWIRE = "pipewire"
    PULSEAUDIO = "pulseaudio"
    ALSA = "alsa"  # Fallback
    UNKNOWN = "unknown"


class AudioBackend(Protocol):
    """Protocol defining the audio backend interface"""
    
    def refresh_devices(self) -> Dict[int, Any]:
        """Refresh and return audio devices"""
        ...
        
    def get_sources(self) -> Dict[int, Any]:
        """Get audio source devices"""
        ...
        
    def get_sinks(self) -> Dict[int, Any]:
        """Get audio sink devices"""
        ...
        
    def create_loopback(self, source_id: int, sink_id: int, 
                       latency_ms: int = 40) -> Optional[int]:
        """Create audio loopback between devices"""
        ...
        
    def remove_loopback(self, link_id: int) -> bool:
        """Remove audio loopback"""
        ...
        
    def set_default_source(self, device_id: int) -> bool:
        """Set default audio source"""
        ...
        
    def set_default_sink(self, device_id: int) -> bool:
        """Set default audio sink"""
        ...


class AudioSystemFactory:
    """Factory for creating appropriate audio system backends"""
    
    @staticmethod
    def detect_audio_system() -> AudioSystemType:
        """Detect which audio system is running"""
        # Check for PipeWire first (preferred)
        try:
            result = subprocess.run(
                ['pw-cli', 'info'],
                capture_output=True,
                timeout=1
            )
            if result.returncode == 0:
                logger.info("PipeWire audio system detected")
                return AudioSystemType.PIPEWIRE
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
        # Check for PulseAudio
        try:
            result = subprocess.run(
                ['pactl', 'info'],
                capture_output=True,
                timeout=1
            )
            if result.returncode == 0:
                # Check if it's real PulseAudio or PipeWire-pulse
                output = result.stdout.decode()
                if 'PipeWire' in output:
                    logger.info("PipeWire (with PulseAudio compatibility) detected")
                    return AudioSystemType.PIPEWIRE
                else:
                    logger.info("PulseAudio audio system detected")
                    return AudioSystemType.PULSEAUDIO
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
        # Check for ALSA
        try:
            result = subprocess.run(
                ['aplay', '-l'],
                capture_output=True,
                timeout=1
            )
            if result.returncode == 0:
                logger.info("ALSA audio system detected")
                return AudioSystemType.ALSA
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
        logger.warning("No supported audio system detected")
        return AudioSystemType.UNKNOWN
        
    @staticmethod
    def create_backend(system_type: Optional[AudioSystemType] = None) -> Optional[AudioBackend]:
        """Create appropriate audio backend based on system type"""
        if system_type is None:
            system_type = AudioSystemFactory.detect_audio_system()
            
        if system_type == AudioSystemType.PIPEWIRE:
            try:
                from .pipewire_backend import PipeWireBackend
                backend = PipeWireBackend()
                logger.info("Created PipeWire audio backend")
                return backend
            except ImportError as e:
                logger.error(f"Failed to import PipeWire backend: {e}")
                # Fall back to PulseAudio
                system_type = AudioSystemType.PULSEAUDIO
                
        if system_type == AudioSystemType.PULSEAUDIO:
            try:
                from .pulseaudio_backend import PulseAudioBackend
                backend = PulseAudioBackend()
                logger.info("Created PulseAudio backend")
                return backend
            except ImportError:
                # If PulseAudio backend doesn't exist yet, try PipeWire compatibility
                try:
                    from .pipewire_backend import PipeWireBackend, PulseAudioCompat
                    pw_backend = PipeWireBackend()
                    backend = PulseAudioCompatBackend(pw_backend)
                    logger.info("Created PipeWire backend with PulseAudio compatibility")
                    return backend
                except ImportError as e:
                    logger.error(f"Failed to create PulseAudio-compatible backend: {e}")
                    
        if system_type == AudioSystemType.ALSA:
            logger.warning("Direct ALSA backend not implemented, audio forwarding will be limited")
            return None
            
        logger.error(f"No backend available for audio system: {system_type.value}")
        return None
        
    @staticmethod
    def create_bluetooth_manager(audio_backend: Optional[AudioBackend] = None):
        """Create appropriate Bluetooth manager based on audio system"""
        system_type = AudioSystemFactory.detect_audio_system()
        
        if system_type == AudioSystemType.PIPEWIRE:
            try:
                from ..bluetooth.pipewire_bluetooth import PipeWireBluetoothManager
                if audio_backend and hasattr(audio_backend, '__class__'):
                    if audio_backend.__class__.__name__ == 'PipeWireBackend':
                        manager = PipeWireBluetoothManager(audio_backend)
                    else:
                        # Create new PipeWire backend
                        from .pipewire_backend import PipeWireBackend
                        pw_backend = PipeWireBackend()
                        manager = PipeWireBluetoothManager(pw_backend)
                else:
                    manager = PipeWireBluetoothManager()
                    
                logger.info("Created PipeWire Bluetooth manager")
                return manager
            except ImportError as e:
                logger.error(f"Failed to import PipeWire Bluetooth manager: {e}")
                
        # Fall back to existing Bluetooth manager
        try:
            from ..bluetooth.bluez_manager import BluezManager
            manager = BluezManager()
            logger.info("Created BlueZ Bluetooth manager")
            return manager
        except ImportError as e:
            logger.error(f"Failed to create Bluetooth manager: {e}")
            return None
            
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get information about the audio system"""
        system_type = AudioSystemFactory.detect_audio_system()
        info = {
            "system_type": system_type.value,
            "available_backends": []
        }
        
        # Check which backends are available
        try:
            from .pipewire_backend import PipeWireBackend
            info["available_backends"].append("pipewire")
        except ImportError:
            pass
            
        try:
            from .pulseaudio_backend import PulseAudioBackend
            info["available_backends"].append("pulseaudio")
        except ImportError:
            pass
            
        # Get version information
        if system_type == AudioSystemType.PIPEWIRE:
            try:
                result = subprocess.run(
                    ['pipewire', '--version'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    info["version"] = result.stdout.strip()
            except:
                pass
                
        elif system_type == AudioSystemType.PULSEAUDIO:
            try:
                result = subprocess.run(
                    ['pulseaudio', '--version'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    info["version"] = result.stdout.strip()
            except:
                pass
                
        return info


class PulseAudioCompatBackend:
    """Wrapper to make PipeWire backend compatible with PulseAudio interface"""
    
    def __init__(self, pipewire_backend):
        self.backend = pipewire_backend
        from .pipewire_backend import PulseAudioCompat
        self.compat = PulseAudioCompat(pipewire_backend)
        
    def refresh_devices(self) -> Dict[int, Any]:
        return self.backend.refresh_devices()
        
    def get_sources(self) -> Dict[int, Any]:
        return self.backend.get_sources()
        
    def get_sinks(self) -> Dict[int, Any]:
        return self.backend.get_sinks()
        
    def create_loopback(self, source_id: int, sink_id: int, 
                       latency_ms: int = 40) -> Optional[int]:
        return self.backend.create_loopback(source_id, sink_id, latency_ms)
        
    def remove_loopback(self, link_id: int) -> bool:
        return self.backend.remove_loopback(link_id)
        
    def set_default_source(self, device_id: int) -> bool:
        return self.backend.set_default_source(device_id)
        
    def set_default_sink(self, device_id: int) -> bool:
        return self.backend.set_default_sink(device_id)
        
    # PulseAudio compatibility methods
    def list_sources(self):
        return self.compat.list_sources()
        
    def list_sinks(self):
        return self.compat.list_sinks()
        
    def load_module_loopback(self, source: str, sink: str, 
                           latency_msec: int = 40) -> Optional[int]:
        return self.compat.load_module_loopback(source, sink, latency_msec)


# Singleton instance management
_audio_backend_instance = None
_bluetooth_manager_instance = None


def get_audio_backend() -> Optional[AudioBackend]:
    """Get or create singleton audio backend instance"""
    global _audio_backend_instance
    if _audio_backend_instance is None:
        _audio_backend_instance = AudioSystemFactory.create_backend()
    return _audio_backend_instance


def get_bluetooth_manager():
    """Get or create singleton Bluetooth manager instance"""
    global _bluetooth_manager_instance
    if _bluetooth_manager_instance is None:
        backend = get_audio_backend()
        _bluetooth_manager_instance = AudioSystemFactory.create_bluetooth_manager(backend)
    return _bluetooth_manager_instance


def reset_audio_system():
    """Reset audio system instances (useful for testing or system changes)"""
    global _audio_backend_instance, _bluetooth_manager_instance
    _audio_backend_instance = None
    _bluetooth_manager_instance = None
    logger.info("Audio system instances reset")


# Convenience functions
def is_pipewire_available() -> bool:
    """Check if PipeWire is available"""
    return AudioSystemFactory.detect_audio_system() == AudioSystemType.PIPEWIRE


def is_pulseaudio_available() -> bool:
    """Check if PulseAudio is available"""
    system = AudioSystemFactory.detect_audio_system()
    return system == AudioSystemType.PULSEAUDIO


def get_preferred_latency() -> int:
    """Get preferred latency based on audio system"""
    system = AudioSystemFactory.detect_audio_system()
    if system == AudioSystemType.PIPEWIRE:
        return 20  # PipeWire can handle lower latency
    else:
        return 40  # Conservative default


if __name__ == "__main__":
    # Test the factory
    logging.basicConfig(level=logging.INFO)
    
    print("Audio System Factory Test")
    print("=" * 50)
    
    # Get system info
    info = AudioSystemFactory.get_system_info()
    print(f"Detected audio system: {info['system_type']}")
    print(f"Available backends: {', '.join(info['available_backends'])}")
    if 'version' in info:
        print(f"Version: {info['version']}")
        
    # Create backend
    backend = get_audio_backend()
    if backend:
        print("\n✓ Audio backend created successfully")
        
        # Test basic functionality
        devices = backend.refresh_devices()
        print(f"Found {len(devices)} audio devices")
        
        sources = backend.get_sources()
        print(f"  Sources: {len(sources)}")
        
        sinks = backend.get_sinks()
        print(f"  Sinks: {len(sinks)}")
    else:
        print("\n✗ Failed to create audio backend")
        
    # Create Bluetooth manager
    bt_manager = get_bluetooth_manager()
    if bt_manager:
        print("\n✓ Bluetooth manager created successfully")
    else:
        print("\n✗ Failed to create Bluetooth manager")
        
    print(f"\nPreferred latency: {get_preferred_latency()}ms")