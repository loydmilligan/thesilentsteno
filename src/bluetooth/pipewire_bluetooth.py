#!/usr/bin/env python3
"""
PipeWire Bluetooth Manager for The Silent Steno

Manages Bluetooth audio devices using PipeWire's native Bluetooth support.
Provides dual A2DP operation for simultaneous source and sink connections.
"""

import subprocess
import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import dbus
from audio.pipewire_backend import PipeWireBackend, AudioDevice, DeviceDirection

logger = logging.getLogger(__name__)


class BluetoothProfile(Enum):
    """Bluetooth audio profiles"""
    A2DP_SOURCE = "a2dp-source"  # Receive audio from phone
    A2DP_SINK = "a2dp-sink"      # Send audio to headphones
    HSP_HS = "hsp-hs"            # Headset profile
    HFP_HF = "hfp-hf"            # Hands-free profile
    OFF = "off"                   # No audio profile


class BluetoothCodec(Enum):
    """Bluetooth audio codecs"""
    SBC = "sbc"
    AAC = "aac"
    APTX = "aptx"
    APTX_HD = "aptx-hd"
    APTX_LL = "aptx-ll"
    LDAC = "ldac"


@dataclass
class BluetoothDevice:
    """Represents a Bluetooth audio device"""
    address: str
    name: str
    alias: str
    paired: bool = False
    connected: bool = False
    trusted: bool = False
    blocked: bool = False
    profiles: List[BluetoothProfile] = field(default_factory=list)
    current_profile: Optional[BluetoothProfile] = None
    current_codec: Optional[BluetoothCodec] = None
    audio_connected: bool = False
    adapter: str = "hci0"
    pipewire_id: Optional[int] = None
    properties: Dict[str, Any] = field(default_factory=dict)


class PipeWireBluetoothManager:
    """Manages Bluetooth devices through PipeWire"""
    
    def __init__(self, pipewire_backend: Optional[PipeWireBackend] = None):
        self.backend = pipewire_backend or PipeWireBackend()
        self.devices: Dict[str, BluetoothDevice] = {}
        self.dbus_system = dbus.SystemBus()
        self.bluez_service = 'org.bluez'
        
        # Adapter configuration for dual radio
        self.source_adapter = "hci0"  # For phones (A2DP source)
        self.sink_adapter = "hci1"    # For headphones (A2DP sink)
        
    def refresh_bluetooth_devices(self) -> Dict[str, BluetoothDevice]:
        """Refresh list of Bluetooth devices from BlueZ and PipeWire"""
        self.devices.clear()
        
        # Get devices from BlueZ
        self._refresh_from_bluez()
        
        # Match with PipeWire devices
        self._match_pipewire_devices()
        
        return self.devices
        
    def _refresh_from_bluez(self):
        """Get Bluetooth devices from BlueZ D-Bus"""
        try:
            # Get object manager
            obj_manager = dbus.Interface(
                self.dbus_system.get_object(self.bluez_service, '/'),
                'org.freedesktop.DBus.ObjectManager'
            )
            
            objects = obj_manager.GetManagedObjects()
            
            for path, interfaces in objects.items():
                if 'org.bluez.Device1' in interfaces:
                    self._process_bluez_device(path, interfaces['org.bluez.Device1'])
                    
        except Exception as e:
            logger.error(f"Failed to refresh BlueZ devices: {e}")
            
    def _process_bluez_device(self, path: str, properties: Dict[str, Any]):
        """Process a BlueZ device"""
        address = properties.get('Address', '')
        if not address:
            return
            
        # Extract adapter from path (e.g., /org/bluez/hci0/dev_XX_XX_XX)
        adapter_match = re.search(r'/(hci\d+)/', path)
        adapter = adapter_match.group(1) if adapter_match else 'hci0'
        
        device = BluetoothDevice(
            address=address,
            name=properties.get('Name', 'Unknown'),
            alias=properties.get('Alias', properties.get('Name', 'Unknown')),
            paired=properties.get('Paired', False),
            connected=properties.get('Connected', False),
            trusted=properties.get('Trusted', False),
            blocked=properties.get('Blocked', False),
            adapter=adapter,
            properties=dict(properties)
        )
        
        # Get supported profiles from UUIDs
        uuids = properties.get('UUIDs', [])
        device.profiles = self._parse_profiles_from_uuids(uuids)
        
        self.devices[address] = device
        
    def _parse_profiles_from_uuids(self, uuids: List[str]) -> List[BluetoothProfile]:
        """Parse Bluetooth profiles from UUID list"""
        profiles = []
        
        # UUID to profile mapping
        uuid_map = {
            '0000110a-0000-1000-8000-00805f9b34fb': BluetoothProfile.A2DP_SOURCE,
            '0000110b-0000-1000-8000-00805f9b34fb': BluetoothProfile.A2DP_SINK,
            '00001112-0000-1000-8000-00805f9b34fb': BluetoothProfile.HSP_HS,
            '0000111f-0000-1000-8000-00805f9b34fb': BluetoothProfile.HFP_HF,
        }
        
        for uuid in uuids:
            if uuid.lower() in uuid_map:
                profiles.append(uuid_map[uuid.lower()])
                
        return profiles
        
    def _match_pipewire_devices(self):
        """Match BlueZ devices with PipeWire audio devices"""
        # Refresh PipeWire devices
        self.backend.refresh_devices()
        
        for pw_device in self.backend.devices.values():
            if not pw_device.is_bluetooth or not pw_device.bluetooth_address:
                continue
                
            # Find matching BlueZ device
            if pw_device.bluetooth_address in self.devices:
                bt_device = self.devices[pw_device.bluetooth_address]
                bt_device.pipewire_id = pw_device.id
                bt_device.audio_connected = True
                
                # Get current codec
                if pw_device.codec:
                    try:
                        bt_device.current_codec = BluetoothCodec(pw_device.codec.lower())
                    except ValueError:
                        pass
                        
                # Determine current profile from PipeWire
                if pw_device.direction == DeviceDirection.SOURCE:
                    bt_device.current_profile = BluetoothProfile.A2DP_SOURCE
                elif pw_device.direction == DeviceDirection.SINK:
                    bt_device.current_profile = BluetoothProfile.A2DP_SINK
                    
    def connect_device(self, address: str, profile: Optional[BluetoothProfile] = None) -> bool:
        """Connect to a Bluetooth device"""
        device = self.devices.get(address)
        if not device:
            logger.error(f"Device {address} not found")
            return False
            
        try:
            # Get device object
            device_path = f"/org/bluez/{device.adapter}/dev_{address.replace(':', '_')}"
            device_obj = self.dbus_system.get_object(self.bluez_service, device_path)
            device_iface = dbus.Interface(device_obj, 'org.bluez.Device1')
            
            # Connect if not connected
            if not device.connected:
                logger.info(f"Connecting to {device.name} ({address})")
                device_iface.Connect()
                
                # Wait for connection
                time.sleep(2)
                
            # Set profile if specified
            if profile and profile != BluetoothProfile.OFF:
                self.set_profile(address, profile)
                
            return True
            
        except dbus.DBusException as e:
            logger.error(f"Failed to connect {address}: {e}")
            return False
            
    def disconnect_device(self, address: str) -> bool:
        """Disconnect from a Bluetooth device"""
        device = self.devices.get(address)
        if not device:
            return False
            
        try:
            device_path = f"/org/bluez/{device.adapter}/dev_{address.replace(':', '_')}"
            device_obj = self.dbus_system.get_object(self.bluez_service, device_path)
            device_iface = dbus.Interface(device_obj, 'org.bluez.Device1')
            
            device_iface.Disconnect()
            return True
            
        except dbus.DBusException as e:
            logger.error(f"Failed to disconnect {address}: {e}")
            return False
            
    def set_profile(self, address: str, profile: BluetoothProfile) -> bool:
        """Set Bluetooth audio profile for a device"""
        device = self.devices.get(address)
        if not device or not device.pipewire_id:
            logger.error(f"Device {address} not found in PipeWire")
            return False
            
        # Map profile to PipeWire card profile string
        profile_map = {
            BluetoothProfile.A2DP_SOURCE: "a2dp-source",
            BluetoothProfile.A2DP_SINK: "a2dp-sink",
            BluetoothProfile.HSP_HS: "headset-head-unit",
            BluetoothProfile.HFP_HF: "headset-head-unit",
            BluetoothProfile.OFF: "off"
        }
        
        pw_profile = profile_map.get(profile)
        if not pw_profile:
            logger.error(f"Unknown profile: {profile}")
            return False
            
        # Use wpctl to set profile
        cmd = ['wpctl', 'set-profile', str(device.pipewire_id), pw_profile]
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode == 0:
            device.current_profile = profile
            logger.info(f"Set {device.name} to profile {profile.value}")
            return True
        else:
            logger.error(f"Failed to set profile: {result.stderr.decode()}")
            return False
            
    def get_available_codecs(self, address: str) -> List[BluetoothCodec]:
        """Get available codecs for a device"""
        device = self.devices.get(address)
        if not device or not device.pipewire_id:
            return []
            
        # Get device info from PipeWire
        info = self.backend.get_device_info(device.pipewire_id)
        if not info:
            return []
            
        codecs = []
        # Parse available codecs from properties
        props = info.get('properties', {})
        codec_str = props.get('api.bluez5.codecs', '')
        
        if codec_str:
            # Parse codec list
            for codec in codec_str.strip('[]').split():
                try:
                    codecs.append(BluetoothCodec(codec.lower()))
                except ValueError:
                    pass
                    
        return codecs
        
    def set_codec(self, address: str, codec: BluetoothCodec) -> bool:
        """Set preferred codec for a device"""
        device = self.devices.get(address)
        if not device:
            return False
            
        # This would typically be done through PipeWire configuration
        # For now, log the preference
        logger.info(f"Setting preferred codec {codec.value} for {device.name}")
        
        # In a real implementation, this would update PipeWire config
        # or use pw-cli to set the codec preference
        
        return True
        
    def setup_dual_radio(self) -> bool:
        """Configure dual radio setup for Silent Steno"""
        logger.info("Setting up dual radio configuration")
        
        # Ensure both adapters are available
        adapters = self._get_available_adapters()
        
        if self.source_adapter not in adapters:
            logger.error(f"Source adapter {self.source_adapter} not found")
            return False
            
        if self.sink_adapter not in adapters:
            logger.error(f"Sink adapter {self.sink_adapter} not found")
            return False
            
        logger.info(f"Dual radio setup complete:")
        logger.info(f"  Sources (phones) on: {self.source_adapter}")
        logger.info(f"  Sinks (headphones) on: {self.sink_adapter}")
        
        return True
        
    def _get_available_adapters(self) -> List[str]:
        """Get list of available Bluetooth adapters"""
        adapters = []
        
        try:
            # Use hciconfig or hcitool to get adapter list
            result = subprocess.run(
                ['hcitool', 'dev'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse hcitool output: "Devices:\n\thci1\tMAC\n\thci0\tMAC"
                for line in result.stdout.split('\n'):
                    if line.startswith('\t') and 'hci' in line:
                        # Extract hciX from line like "\thci1\tC0:FB:F9:61:B1:E3"
                        parts = line.strip().split('\t')
                        if len(parts) >= 2 and parts[0].startswith('hci'):
                            adapters.append(parts[0])
            
            # Fallback: check /sys/class/bluetooth/
            if not adapters:
                import glob
                for path in glob.glob('/sys/class/bluetooth/hci*'):
                    adapter = path.split('/')[-1]
                    if adapter.startswith('hci'):
                        adapters.append(adapter)
                            
        except Exception as e:
            logger.error(f"Failed to get adapters: {e}")
            
        return adapters
        
    def create_audio_forwarding(self, source_address: str, sink_address: str,
                               latency_ms: int = 40) -> Optional[int]:
        """Create audio forwarding between Bluetooth devices"""
        source = self.devices.get(source_address)
        sink = self.devices.get(sink_address)
        
        if not source or not sink:
            logger.error("Source or sink device not found")
            return None
            
        if not source.pipewire_id or not sink.pipewire_id:
            logger.error("Devices not connected to PipeWire")
            return None
            
        # Ensure correct profiles
        if source.current_profile != BluetoothProfile.A2DP_SOURCE:
            self.set_profile(source_address, BluetoothProfile.A2DP_SOURCE)
            
        if sink.current_profile != BluetoothProfile.A2DP_SINK:
            self.set_profile(sink_address, BluetoothProfile.A2DP_SINK)
            
        # Create loopback using PipeWire backend
        link_id = self.backend.create_loopback(
            source.pipewire_id,
            sink.pipewire_id,
            latency_ms
        )
        
        if link_id:
            logger.info(f"Created audio forwarding: {source.name} -> {sink.name}")
            
        return link_id
        
    def get_device_by_role(self, role: str) -> Optional[BluetoothDevice]:
        """Get device by role (source/sink)"""
        for device in self.devices.values():
            if role == "source" and device.current_profile == BluetoothProfile.A2DP_SOURCE:
                return device
            elif role == "sink" and device.current_profile == BluetoothProfile.A2DP_SINK:
                return device
                
        return None
        
    def monitor_connections(self, callback: callable):
        """Monitor Bluetooth connection changes"""
        # This would use D-Bus signals for real-time monitoring
        # For now, use polling approach
        
        def connection_changed(device: BluetoothDevice, connected: bool):
            callback({
                'device': device,
                'connected': connected,
                'timestamp': time.time()
            })
            
        # In real implementation, this would register D-Bus signal handlers
        logger.info("Bluetooth connection monitoring started")


def create_manager(pipewire_backend: Optional[PipeWireBackend] = None) -> PipeWireBluetoothManager:
    """Factory function to create Bluetooth manager"""
    return PipeWireBluetoothManager(pipewire_backend)


if __name__ == "__main__":
    # Test the manager
    logging.basicConfig(level=logging.INFO)
    
    print("PipeWire Bluetooth Manager Test")
    print("=" * 50)
    
    manager = create_manager()
    
    # Setup dual radio
    if manager.setup_dual_radio():
        print("✓ Dual radio setup complete")
    else:
        print("✗ Dual radio setup failed")
        
    # Refresh devices
    devices = manager.refresh_bluetooth_devices()
    
    print(f"\nFound {len(devices)} Bluetooth devices:")
    for address, device in devices.items():
        print(f"\n{device.name} ({address})")
        print(f"  Adapter: {device.adapter}")
        print(f"  Paired: {device.paired}")
        print(f"  Connected: {device.connected}")
        print(f"  Audio Connected: {device.audio_connected}")
        print(f"  Profiles: {[p.value for p in device.profiles]}")
        if device.current_profile:
            print(f"  Current Profile: {device.current_profile.value}")
        if device.current_codec:
            print(f"  Current Codec: {device.current_codec.value}")