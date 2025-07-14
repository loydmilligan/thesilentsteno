#!/usr/bin/env python3

"""
Bluetooth Connection Manager for The Silent Steno

This module manages Bluetooth device connections, including pairing,
connection persistence, auto-reconnection, and dual A2DP connection handling.

Key features:
- Device pairing and bonding management
- Connection state monitoring and persistence
- Automatic reconnection logic
- Dual A2DP connection management (phone + headphones)
- Connection quality monitoring
- Fallback and recovery mechanisms
"""

import subprocess
import threading
import time
import json
import os
import logging
from typing import Dict, List, Optional, Callable, Tuple, Any
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

# Import our BlueZ manager
from .bluez_manager import BlueZManager, BluetoothState

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Bluetooth device connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    BONDED = "bonded"


class DeviceRole(Enum):
    """Device role in audio pipeline"""
    SOURCE = "source"  # Device that sends audio to us (phone)
    SINK = "sink"      # Device that receives audio from us (headphones)
    UNKNOWN = "unknown"


@dataclass
class BluetoothDevice:
    """Bluetooth device information"""
    address: str
    name: str
    role: DeviceRole
    connection_state: ConnectionState
    last_connected: Optional[datetime] = None
    connection_attempts: int = 0
    max_attempts: int = 5
    trusted: bool = False
    bonded: bool = False
    rssi: Optional[int] = None
    codec: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        if self.last_connected:
            data['last_connected'] = self.last_connected.isoformat()
        data['role'] = self.role.value
        data['connection_state'] = self.connection_state.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BluetoothDevice':
        """Create from dictionary (JSON deserialization)"""
        if 'last_connected' in data and data['last_connected']:
            data['last_connected'] = datetime.fromisoformat(data['last_connected'])
        data['role'] = DeviceRole(data['role'])
        data['connection_state'] = ConnectionState(data['connection_state'])
        return cls(**data)


class ConnectionManager:
    """
    Manages Bluetooth device connections for The Silent Steno
    
    Handles pairing, connection persistence, auto-reconnection,
    and dual A2DP connection management.
    """
    
    def __init__(self, config_dir: str = "/home/mmariani/projects/thesilentsteno/config"):
        """Initialize connection manager"""
        self.config_dir = config_dir
        self.devices_file = os.path.join(config_dir, "bluetooth_devices.json")
        self.bluez_manager = BlueZManager()
        
        # Device storage
        self.known_devices: Dict[str, BluetoothDevice] = {}
        self.connection_callbacks: List[Callable] = []
        
        # Connection monitoring
        self.monitoring_thread = None
        self.monitoring_active = False
        self.reconnect_interval = 30  # seconds
        self.max_concurrent_connections = 2
        
        # Load known devices
        self.load_devices()
        
        # Initialize monitoring
        self.start_monitoring()
        
    def load_devices(self) -> None:
        """Load known devices from persistent storage"""
        try:
            if os.path.exists(self.devices_file):
                with open(self.devices_file, 'r') as f:
                    data = json.load(f)
                    
                for addr, device_data in data.items():
                    try:
                        device = BluetoothDevice.from_dict(device_data)
                        self.known_devices[addr] = device
                    except Exception as e:
                        logger.warning(f"Failed to load device {addr}: {e}")
                        
                logger.info(f"Loaded {len(self.known_devices)} known devices")
            else:
                logger.info("No known devices file found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load devices: {e}")
            self.known_devices = {}
    
    def save_devices(self) -> None:
        """Save known devices to persistent storage"""
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            
            data = {}
            for addr, device in self.known_devices.items():
                data[addr] = device.to_dict()
            
            with open(self.devices_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug("Saved known devices to file")
            
        except Exception as e:
            logger.error(f"Failed to save devices: {e}")
    
    def add_connection_callback(self, callback: Callable[[str, ConnectionState], None]) -> None:
        """Add callback for connection state changes"""
        self.connection_callbacks.append(callback)
    
    def _notify_connection_change(self, device_address: str, state: ConnectionState) -> None:
        """Notify callbacks of connection state changes"""
        for callback in self.connection_callbacks:
            try:
                callback(device_address, state)
            except Exception as e:
                logger.error(f"Error in connection callback: {e}")
    
    def pair_device(self, device_address: str, device_name: str = "", 
                   role: DeviceRole = DeviceRole.UNKNOWN) -> bool:
        """
        Pair with a Bluetooth device
        
        Args:
            device_address: MAC address of device to pair
            device_name: Human-readable device name
            role: Device role in audio pipeline
            
        Returns:
            bool: True if pairing successful
        """
        try:
            logger.info(f"Attempting to pair with {device_address} ({device_name})")
            
            # Create or update device record
            if device_address not in self.known_devices:
                self.known_devices[device_address] = BluetoothDevice(
                    address=device_address,
                    name=device_name or f"Device_{device_address[-5:]}",
                    role=role,
                    connection_state=ConnectionState.DISCONNECTED
                )
            
            device = self.known_devices[device_address]
            device.connection_state = ConnectionState.CONNECTING
            
            # Make adapter discoverable for pairing
            self.bluez_manager.enable_discoverable(180)
            
            # Attempt pairing using bluetoothctl
            pair_result = subprocess.run(
                ["bluetoothctl", "--timeout", "30", "pair", device_address],
                capture_output=True,
                text=True,
                timeout=35
            )
            
            if pair_result.returncode == 0:
                logger.info(f"Successfully paired with {device_address}")
                
                # Trust the device for auto-reconnection
                trust_result = subprocess.run(
                    ["bluetoothctl", "trust", device_address],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if trust_result.returncode == 0:
                    device.trusted = True
                    logger.info(f"Device {device_address} is now trusted")
                
                # Update device state
                device.bonded = True
                device.connection_state = ConnectionState.BONDED
                device.connection_attempts = 0
                
                # Save to persistent storage
                self.save_devices()
                
                # Notify callbacks
                self._notify_connection_change(device_address, ConnectionState.BONDED)
                
                return True
            else:
                logger.error(f"Failed to pair with {device_address}: {pair_result.stderr}")
                device.connection_state = ConnectionState.FAILED
                device.connection_attempts += 1
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout pairing with {device_address}")
            if device_address in self.known_devices:
                self.known_devices[device_address].connection_state = ConnectionState.FAILED
            return False
        except Exception as e:
            logger.error(f"Error pairing with {device_address}: {e}")
            if device_address in self.known_devices:
                self.known_devices[device_address].connection_state = ConnectionState.FAILED
            return False
    
    def connect_device(self, device_address: str) -> bool:
        """
        Connect to a paired/bonded device
        
        Args:
            device_address: MAC address of device to connect
            
        Returns:
            bool: True if connection successful
        """
        try:
            if device_address not in self.known_devices:
                logger.error(f"Device {device_address} not in known devices")
                return False
            
            device = self.known_devices[device_address]
            
            # Check if we're at connection limit
            connected_count = sum(1 for d in self.known_devices.values() 
                                if d.connection_state == ConnectionState.CONNECTED)
            
            if connected_count >= self.max_concurrent_connections:
                logger.warning(f"Maximum concurrent connections ({self.max_concurrent_connections}) reached")
                return False
            
            logger.info(f"Attempting to connect to {device_address}")
            device.connection_state = ConnectionState.CONNECTING
            
            # Attempt connection
            connect_result = subprocess.run(
                ["bluetoothctl", "--timeout", "20", "connect", device_address],
                capture_output=True,
                text=True,
                timeout=25
            )
            
            if connect_result.returncode == 0:
                logger.info(f"Successfully connected to {device_address}")
                device.connection_state = ConnectionState.CONNECTED
                device.last_connected = datetime.now()
                device.connection_attempts = 0
                
                # Detect codec being used
                device.codec = self._detect_codec(device_address)
                
                # Save state
                self.save_devices()
                
                # Notify callbacks
                self._notify_connection_change(device_address, ConnectionState.CONNECTED)
                
                return True
            else:
                logger.error(f"Failed to connect to {device_address}: {connect_result.stderr}")
                device.connection_state = ConnectionState.FAILED
                device.connection_attempts += 1
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout connecting to {device_address}")
            if device_address in self.known_devices:
                self.known_devices[device_address].connection_state = ConnectionState.FAILED
            return False
        except Exception as e:
            logger.error(f"Error connecting to {device_address}: {e}")
            if device_address in self.known_devices:
                self.known_devices[device_address].connection_state = ConnectionState.FAILED
            return False
    
    def disconnect_device(self, device_address: str) -> bool:
        """
        Disconnect from a device
        
        Args:
            device_address: MAC address of device to disconnect
            
        Returns:
            bool: True if disconnection successful
        """
        try:
            logger.info(f"Disconnecting from {device_address}")
            
            disconnect_result = subprocess.run(
                ["bluetoothctl", "disconnect", device_address],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if disconnect_result.returncode == 0:
                logger.info(f"Successfully disconnected from {device_address}")
                
                if device_address in self.known_devices:
                    device = self.known_devices[device_address]
                    device.connection_state = ConnectionState.DISCONNECTED
                    device.codec = None
                    
                    # Notify callbacks
                    self._notify_connection_change(device_address, ConnectionState.DISCONNECTED)
                
                return True
            else:
                logger.error(f"Failed to disconnect from {device_address}: {disconnect_result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error disconnecting from {device_address}: {e}")
            return False
    
    def _detect_codec(self, device_address: str) -> Optional[str]:
        """Detect which codec is being used for a connected device"""
        try:
            # Check PulseAudio for codec information
            result = subprocess.run(
                ["pactl", "list", "sinks"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                output = result.stdout.lower()
                
                # Look for codec indicators in the output
                if "aac" in output:
                    return "AAC"
                elif "aptx" in output:
                    if "aptx_hd" in output:
                        return "aptX HD"
                    else:
                        return "aptX"
                elif "scalable" in output:
                    return "Samsung Scalable"
                else:
                    return "SBC"  # Default codec
            
        except Exception as e:
            logger.debug(f"Could not detect codec for {device_address}: {e}")
        
        return None
    
    def get_connected_devices(self) -> List[BluetoothDevice]:
        """Get list of currently connected devices"""
        connected = []
        for device in self.known_devices.values():
            if device.connection_state == ConnectionState.CONNECTED:
                connected.append(device)
        return connected
    
    def get_device_info(self, device_address: str) -> Optional[BluetoothDevice]:
        """Get information about a specific device"""
        return self.known_devices.get(device_address)
    
    def start_monitoring(self) -> None:
        """Start connection monitoring thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="BluetoothMonitoring"
        )
        self.monitoring_thread.start()
        logger.info("Started Bluetooth connection monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop connection monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped Bluetooth connection monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for connection state"""
        while self.monitoring_active:
            try:
                self._check_connections()
                self._attempt_reconnections()
                time.sleep(self.reconnect_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _check_connections(self) -> None:
        """Check current connection states"""
        try:
            # Get currently connected devices from bluetoothctl
            result = subprocess.run(
                ["bluetoothctl", "devices", "Connected"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                connected_addresses = set()
                
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('Device'):
                        parts = line.split(' ', 2)
                        if len(parts) >= 2:
                            addr = parts[1]
                            connected_addresses.add(addr)
                
                # Update connection states
                for addr, device in self.known_devices.items():
                    if addr in connected_addresses:
                        if device.connection_state != ConnectionState.CONNECTED:
                            logger.info(f"Device {addr} is now connected")
                            device.connection_state = ConnectionState.CONNECTED
                            device.last_connected = datetime.now()
                            self._notify_connection_change(addr, ConnectionState.CONNECTED)
                    else:
                        if device.connection_state == ConnectionState.CONNECTED:
                            logger.info(f"Device {addr} has disconnected")
                            device.connection_state = ConnectionState.DISCONNECTED
                            self._notify_connection_change(addr, ConnectionState.DISCONNECTED)
            
        except Exception as e:
            logger.debug(f"Error checking connections: {e}")
    
    def _attempt_reconnections(self) -> None:
        """Attempt to reconnect to trusted devices"""
        for addr, device in self.known_devices.items():
            if (device.trusted and 
                device.connection_state == ConnectionState.DISCONNECTED and
                device.connection_attempts < device.max_attempts):
                
                # Check if enough time has passed since last attempt
                if (device.last_connected and 
                    datetime.now() - device.last_connected < timedelta(minutes=5)):
                    continue
                
                logger.info(f"Attempting auto-reconnection to {addr}")
                device.connection_state = ConnectionState.RECONNECTING
                
                if self.connect_device(addr):
                    logger.info(f"Auto-reconnection to {addr} successful")
                else:
                    logger.warning(f"Auto-reconnection to {addr} failed")
    
    def manage_connections(self) -> Dict[str, Any]:
        """
        Get comprehensive connection management status
        
        Returns:
            Dict containing current connection state and statistics
        """
        connected_devices = self.get_connected_devices()
        
        return {
            "total_known_devices": len(self.known_devices),
            "connected_devices": len(connected_devices),
            "max_connections": self.max_concurrent_connections,
            "monitoring_active": self.monitoring_active,
            "devices": [device.to_dict() for device in connected_devices],
            "connection_summary": {
                device.address: {
                    "name": device.name,
                    "role": device.role.value,
                    "state": device.connection_state.value,
                    "codec": device.codec,
                    "trusted": device.trusted
                }
                for device in connected_devices
            }
        }


# Convenience functions for external use
def pair_device(device_address: str, device_name: str = "", 
                role: DeviceRole = DeviceRole.UNKNOWN) -> bool:
    """Pair with a device - convenience function"""
    manager = ConnectionManager()
    return manager.pair_device(device_address, device_name, role)


def connect_device(device_address: str) -> bool:
    """Connect to a device - convenience function"""
    manager = ConnectionManager()
    return manager.connect_device(device_address)


def manage_connections() -> Dict[str, Any]:
    """Get connection status - convenience function"""
    manager = ConnectionManager()
    return manager.manage_connections()


if __name__ == "__main__":
    # Basic test when run directly
    print("Bluetooth Connection Manager Test")
    print("=" * 50)
    
    manager = ConnectionManager()
    status = manager.manage_connections()
    
    print(f"Known devices: {status['total_known_devices']}")
    print(f"Connected devices: {status['connected_devices']}")
    print(f"Max connections: {status['max_connections']}")
    print(f"Monitoring active: {status['monitoring_active']}")
    
    if status['connected_devices'] > 0:
        print("\nConnected devices:")
        for device_info in status['devices']:
            print(f"  - {device_info['name']} ({device_info['address']})")
            print(f"    Role: {device_info['role']}")
            print(f"    Codec: {device_info['codec']}")
            print(f"    Trusted: {device_info['trusted']}")
    else:
        print("\nNo devices currently connected")
    
    # Keep monitoring active for a short time
    print("\nMonitoring connections for 30 seconds...")
    time.sleep(30)
    manager.stop_monitoring()