# Bluetooth Module Documentation

## Module Overview

The Bluetooth module provides comprehensive Bluetooth management for The Silent Steno, implementing dual A2DP connections to enable simultaneous audio input from a phone and output to headphones. The module manages the BlueZ Bluetooth stack, device discovery, pairing, connection persistence, and audio codec negotiation with automatic recovery and monitoring capabilities.

## Dependencies

### External Dependencies
- `dbus-python` - D-Bus system integration
- `subprocess` - System command execution
- `threading` - Background monitoring threads
- `json` - Configuration persistence
- `time` - Timing operations
- `logging` - Logging system
- `dataclasses` - Data structures
- `enum` - Enumerations
- `typing` - Type hints
- `pathlib` - Path operations
- `queue` - Thread-safe queues

### System Dependencies
- `BlueZ` - Linux Bluetooth stack
- `D-Bus` - Inter-process communication
- `PulseAudio/ALSA` - Audio system integration
- `bluetoothctl` - Command-line Bluetooth control
- `hcitool` - Bluetooth device utilities

### Internal Dependencies
- `src.core.events` - Event system
- `src.core.logging` - Logging system
- `src.core.monitoring` - Performance monitoring
- `src.core.config` - Configuration management

## Architecture Overview

### Dual A2DP Connection Flow
```
Phone (SOURCE) → Pi 5 (A2DP SINK) → Audio Processing → Pi 5 (A2DP SOURCE) → Headphones (SINK)
                                          ↓
                                   Real-time Analysis
                                          ↓
                                  Whisper Transcription
```

### Component Roles
- **Phone**: Audio source device (A2DP SOURCE profile)
- **Pi 5**: Dual role device (A2DP SINK + A2DP SOURCE)
- **Headphones**: Audio sink device (A2DP SINK profile)
- **BlueZ**: Linux Bluetooth stack management
- **Connection Manager**: Device relationship and state management

## File Documentation

### 1. `__init__.py`

**Purpose**: Module initialization and convenience functions for unified Bluetooth access.

#### Functions

##### `start_bluetooth() -> bool`
Start the Bluetooth service and enable discovery.

**Returns:**
- `bool` - True if service started successfully

##### `stop_bluetooth() -> bool`
Stop the Bluetooth service gracefully.

**Returns:**
- `bool` - True if service stopped successfully

##### `get_bluetooth_status() -> dict`
Get comprehensive Bluetooth system status.

**Returns:**
- `dict` - Status information including service state, connected devices, and capabilities

##### `pair_device(address: str, name: str, role: DeviceRole) -> bool`
Pair with a Bluetooth device.

**Parameters:**
- `address: str` - Device MAC address
- `name: str` - Device display name
- `role: DeviceRole` - Device role (SOURCE/SINK)

**Returns:**
- `bool` - True if pairing successful

##### `connect_device(address: str) -> bool`
Connect to a paired device.

**Parameters:**
- `address: str` - Device MAC address

**Returns:**
- `bool` - True if connection successful

##### `manage_connections() -> dict`
Get comprehensive connection management status.

**Returns:**
- `dict` - Connection status and device information

**Usage Example:**
```python
from src.bluetooth import (
    start_bluetooth, 
    pair_device, 
    connect_device, 
    get_bluetooth_status,
    DeviceRole
)

# Start Bluetooth service
if start_bluetooth():
    print("Bluetooth service started")
    
    # Get system status
    status = get_bluetooth_status()
    print(f"Service active: {status['service_active']}")
    print(f"Discoverable: {status['discoverable']}")
    
    # Pair with phone
    phone_address = "AA:BB:CC:DD:EE:FF"
    if pair_device(phone_address, "iPhone", DeviceRole.SOURCE):
        print("Phone paired successfully")
        
        # Connect to phone
        if connect_device(phone_address):
            print("Phone connected successfully")
    
    # Pair with headphones
    headphones_address = "11:22:33:44:55:66"
    if pair_device(headphones_address, "Headphones", DeviceRole.SINK):
        print("Headphones paired successfully")
        
        # Connect to headphones
        if connect_device(headphones_address):
            print("Headphones connected successfully")
```

### 2. `bluez_manager.py`

**Purpose**: BlueZ Bluetooth stack management with A2DP profile support and codec negotiation.

#### Enums

##### `BluetoothState`
Bluetooth service state enumeration.
- `STOPPED` - Service not running
- `STARTING` - Service starting up
- `RUNNING` - Service active and operational
- `ERROR` - Service in error state
- `UNKNOWN` - Service state unknown

##### `CodecType`
Supported audio codec enumeration.
- `SBC` - Sub-band Codec (baseline)
- `AAC` - Advanced Audio Codec
- `APTX` - Qualcomm aptX codec
- `APTX_HD` - Qualcomm aptX HD codec
- `SAMSUNG_SCALABLE` - Samsung Scalable codec

#### Classes

##### `BlueZManager`
Main BlueZ Bluetooth stack management system.

**Methods:**
- `__init__(config: dict = None)` - Initialize BlueZ manager
- `start_bluetooth()` - Start Bluetooth service
- `stop_bluetooth()` - Stop Bluetooth service
- `restart_bluetooth()` - Restart Bluetooth service
- `get_bluetooth_status()` - Get service status
- `enable_discoverable(timeout: int = 0)` - Enable discoverable mode
- `disable_discoverable()` - Disable discoverable mode
- `scan_devices(duration: int = 10)` - Scan for nearby devices
- `get_adapter_info()` - Get Bluetooth adapter information
- `set_adapter_name(name: str)` - Set adapter name
- `get_supported_codecs()` - Get supported audio codecs
- `configure_a2dp_sink()` - Configure A2DP sink profile
- `configure_a2dp_source()` - Configure A2DP source profile

**Usage Example:**
```python
from src.bluetooth.bluez_manager import BlueZManager, BluetoothState, CodecType

# Create BlueZ manager
bluez_manager = BlueZManager({
    "device_name": "SilentSteno",
    "discoverable_timeout": 120,
    "enable_a2dp_sink": True,
    "enable_a2dp_source": True,
    "preferred_codecs": [CodecType.APTX, CodecType.AAC, CodecType.SBC]
})

# Start Bluetooth service
if bluez_manager.start_bluetooth():
    print("Bluetooth service started")
    
    # Get service status
    status = bluez_manager.get_bluetooth_status()
    print(f"Service state: {status['state']}")
    print(f"Adapter address: {status['adapter_address']}")
    print(f"Adapter name: {status['adapter_name']}")
    
    # Configure A2DP profiles
    bluez_manager.configure_a2dp_sink()  # For receiving audio from phone
    bluez_manager.configure_a2dp_source()  # For sending audio to headphones
    
    # Enable discoverable mode
    bluez_manager.enable_discoverable(timeout=120)
    
    # Scan for devices
    devices = bluez_manager.scan_devices(duration=10)
    for device in devices:
        print(f"Found device: {device['name']} ({device['address']})")
    
    # Get supported codecs
    codecs = bluez_manager.get_supported_codecs()
    print(f"Supported codecs: {[codec.value for codec in codecs]}")
```

### 3. `connection_manager.py`

**Purpose**: Device connection management with pairing, persistence, monitoring, and automatic recovery.

#### Enums

##### `ConnectionState`
Device connection state enumeration.
- `DISCONNECTED` - Device not connected
- `CONNECTING` - Connection in progress
- `CONNECTED` - Device connected
- `RECONNECTING` - Attempting reconnection
- `FAILED` - Connection failed
- `BONDED` - Device paired but not connected

##### `DeviceRole`
Device role enumeration.
- `SOURCE` - Audio source device (phone)
- `SINK` - Audio sink device (headphones)

#### Classes

##### `BluetoothDevice`
Bluetooth device information container.

**Attributes:**
- `address: str` - Device MAC address
- `name: str` - Device display name
- `role: DeviceRole` - Device role
- `state: ConnectionState` - Current connection state
- `trusted: bool` - Is device trusted
- `bonded: bool` - Is device bonded/paired
- `connected: bool` - Is device connected
- `codec: CodecType` - Active audio codec
- `rssi: int` - Signal strength (dBm)
- `last_connected: datetime` - Last connection time
- `connection_attempts: int` - Connection attempt counter
- `max_attempts: int` - Maximum connection attempts

##### `ConnectionManager`
Main connection management system.

**Methods:**
- `__init__(config: dict = None)` - Initialize connection manager
- `pair_device(address: str, name: str, role: DeviceRole)` - Pair with device
- `unpair_device(address: str)` - Remove device pairing
- `connect_device(address: str)` - Connect to paired device
- `disconnect_device(address: str)` - Disconnect from device
- `get_connected_devices()` - Get list of connected devices
- `get_paired_devices()` - Get list of paired devices
- `get_device_info(address: str)` - Get device information
- `start_monitoring()` - Start connection monitoring
- `stop_monitoring()` - Stop connection monitoring
- `add_connection_callback(callback: callable)` - Add connection state callback
- `remove_connection_callback(callback: callable)` - Remove connection callback
- `save_device_config()` - Save device configuration
- `load_device_config()` - Load device configuration

**Usage Example:**
```python
from src.bluetooth.connection_manager import ConnectionManager, DeviceRole, ConnectionState

# Create connection manager
connection_manager = ConnectionManager({
    "max_connections": 2,
    "auto_reconnect": True,
    "reconnect_interval": 30,
    "max_reconnect_attempts": 5,
    "connection_timeout": 25,
    "pairing_timeout": 35
})

# Set up connection monitoring
def on_connection_change(address, state, device_info):
    print(f"Device {device_info['name']} ({address}) changed to {state}")
    
    if state == ConnectionState.CONNECTED:
        print(f"  Codec: {device_info['codec']}")
        print(f"  RSSI: {device_info['rssi']} dBm")
    elif state == ConnectionState.FAILED:
        print(f"  Connection failed after {device_info['connection_attempts']} attempts")

connection_manager.add_connection_callback(on_connection_change)

# Start monitoring
connection_manager.start_monitoring()

# Pair devices
phone_address = "AA:BB:CC:DD:EE:FF"
headphones_address = "11:22:33:44:55:66"

# Pair phone as audio source
if connection_manager.pair_device(phone_address, "iPhone", DeviceRole.SOURCE):
    print("Phone paired successfully")
    
    # Connect to phone
    if connection_manager.connect_device(phone_address):
        print("Phone connected successfully")

# Pair headphones as audio sink
if connection_manager.pair_device(headphones_address, "Headphones", DeviceRole.SINK):
    print("Headphones paired successfully")
    
    # Connect to headphones
    if connection_manager.connect_device(headphones_address):
        print("Headphones connected successfully")

# Get connection status
connected_devices = connection_manager.get_connected_devices()
print(f"Connected devices: {len(connected_devices)}")
for device in connected_devices:
    print(f"  {device.name} ({device.address}) - {device.role.value}")
    print(f"    State: {device.state.value}")
    print(f"    Codec: {device.codec.value}")
    print(f"    RSSI: {device.rssi} dBm")

# Get device information
device_info = connection_manager.get_device_info(phone_address)
if device_info:
    print(f"Device info: {device_info}")
```

## Module Integration

The Bluetooth module integrates with other Silent Steno components:

1. **Audio Module**: Provides audio input/output routing
2. **Core Events**: Publishes connection events and status updates
3. **Monitoring**: Reports connection quality and performance metrics
4. **Configuration**: Manages device pairing and connection preferences
5. **UI Module**: Provides connection status and device management interface

## Common Usage Patterns

### Complete Bluetooth System Setup
```python
# Initialize Bluetooth system
from src.bluetooth import BlueZManager, ConnectionManager, DeviceRole

# Create managers
bluez_manager = BlueZManager({
    "device_name": "SilentSteno",
    "discoverable_timeout": 120,
    "enable_a2dp_sink": True,
    "enable_a2dp_source": True
})

connection_manager = ConnectionManager({
    "max_connections": 2,
    "auto_reconnect": True,
    "reconnect_interval": 30
})

# Start Bluetooth service
if bluez_manager.start_bluetooth():
    print("Bluetooth service started")
    
    # Configure A2DP profiles
    bluez_manager.configure_a2dp_sink()
    bluez_manager.configure_a2dp_source()
    
    # Start connection monitoring
    connection_manager.start_monitoring()
    
    # Enable discoverable mode for pairing
    bluez_manager.enable_discoverable(timeout=120)
    
    print("Bluetooth system ready for pairing")
```

### Device Pairing Workflow
```python
# Pairing workflow for phone and headphones
def pair_devices():
    devices_to_pair = [
        {
            "address": "AA:BB:CC:DD:EE:FF",
            "name": "iPhone",
            "role": DeviceRole.SOURCE
        },
        {
            "address": "11:22:33:44:55:66", 
            "name": "Headphones",
            "role": DeviceRole.SINK
        }
    ]
    
    for device in devices_to_pair:
        print(f"Pairing with {device['name']}...")
        
        # Pair device
        if connection_manager.pair_device(
            device["address"], 
            device["name"], 
            device["role"]
        ):
            print(f"  Paired successfully")
            
            # Connect to device
            if connection_manager.connect_device(device["address"]):
                print(f"  Connected successfully")
            else:
                print(f"  Connection failed")
        else:
            print(f"  Pairing failed")

# Run pairing workflow
pair_devices()
```

### Connection Monitoring and Recovery
```python
# Advanced connection monitoring
class BluetoothMonitor:
    def __init__(self, connection_manager):
        self.connection_manager = connection_manager
        self.connection_history = {}
        self.failed_devices = set()
        
    def setup_monitoring(self):
        # Add connection callback
        self.connection_manager.add_connection_callback(self.on_connection_change)
        
        # Start monitoring
        self.connection_manager.start_monitoring()
        
    def on_connection_change(self, address, state, device_info):
        # Log connection change
        self.connection_history[address] = {
            "state": state,
            "timestamp": time.time(),
            "device_info": device_info
        }
        
        # Handle different states
        if state == ConnectionState.CONNECTED:
            self.on_device_connected(address, device_info)
        elif state == ConnectionState.DISCONNECTED:
            self.on_device_disconnected(address, device_info)
        elif state == ConnectionState.FAILED:
            self.on_connection_failed(address, device_info)
            
    def on_device_connected(self, address, device_info):
        print(f"Device connected: {device_info['name']}")
        print(f"  Codec: {device_info['codec']}")
        print(f"  Signal strength: {device_info['rssi']} dBm")
        
        # Remove from failed devices
        self.failed_devices.discard(address)
        
    def on_device_disconnected(self, address, device_info):
        print(f"Device disconnected: {device_info['name']}")
        
        # Attempt reconnection for trusted devices
        if device_info['trusted']:
            print("  Attempting reconnection...")
            self.connection_manager.connect_device(address)
    
    def on_connection_failed(self, address, device_info):
        print(f"Connection failed: {device_info['name']}")
        print(f"  Attempts: {device_info['connection_attempts']}")
        
        # Track failed devices
        self.failed_devices.add(address)
        
        # Stop trying after max attempts
        if device_info['connection_attempts'] >= device_info['max_attempts']:
            print(f"  Max attempts reached, giving up")
    
    def get_connection_status(self):
        connected = self.connection_manager.get_connected_devices()
        return {
            "connected_devices": len(connected),
            "failed_devices": len(self.failed_devices),
            "connection_history": self.connection_history
        }

# Use monitoring
monitor = BluetoothMonitor(connection_manager)
monitor.setup_monitoring()
```

### Audio Codec Management
```python
# Codec preference and negotiation
def setup_codec_preferences():
    # Define codec preferences by device type
    codec_preferences = {
        DeviceRole.SOURCE: [CodecType.APTX, CodecType.AAC, CodecType.SBC],
        DeviceRole.SINK: [CodecType.APTX_HD, CodecType.APTX, CodecType.AAC, CodecType.SBC]
    }
    
    # Get connected devices
    connected_devices = connection_manager.get_connected_devices()
    
    for device in connected_devices:
        preferred_codecs = codec_preferences.get(device.role, [CodecType.SBC])
        
        # Check if device supports preferred codecs
        for codec in preferred_codecs:
            if bluez_manager.device_supports_codec(device.address, codec):
                print(f"Device {device.name} supports {codec.value}")
                
                # Try to negotiate codec
                if bluez_manager.negotiate_codec(device.address, codec):
                    print(f"  Negotiated {codec.value} successfully")
                    break
            else:
                print(f"Device {device.name} does not support {codec.value}")

# Setup codec preferences
setup_codec_preferences()
```

### Device Configuration Persistence
```python
# Save and restore device configurations
def save_bluetooth_config():
    # Get current device configuration
    devices = connection_manager.get_paired_devices()
    
    config = {
        "devices": [],
        "last_updated": time.time(),
        "version": "1.0"
    }
    
    for device in devices:
        device_config = {
            "address": device.address,
            "name": device.name,
            "role": device.role.value,
            "trusted": device.trusted,
            "last_connected": device.last_connected.isoformat() if device.last_connected else None,
            "codec": device.codec.value if device.codec else None,
            "connection_attempts": device.connection_attempts,
            "max_attempts": device.max_attempts
        }
        config["devices"].append(device_config)
    
    # Save configuration
    connection_manager.save_device_config(config)
    print(f"Saved configuration for {len(devices)} devices")

def restore_bluetooth_config():
    # Load device configuration
    config = connection_manager.load_device_config()
    
    if config:
        print(f"Restoring {len(config['devices'])} devices")
        
        for device_config in config["devices"]:
            # Restore device settings
            device = BluetoothDevice(
                address=device_config["address"],
                name=device_config["name"],
                role=DeviceRole(device_config["role"]),
                trusted=device_config["trusted"],
                max_attempts=device_config["max_attempts"]
            )
            
            # Add to connection manager
            connection_manager.add_device(device)
            
            # Attempt connection for trusted devices
            if device.trusted:
                connection_manager.connect_device(device.address)

# Use configuration persistence
save_bluetooth_config()
restore_bluetooth_config()
```

### Error Handling and Recovery
```python
# Robust Bluetooth operations with error handling
class BluetoothHandler:
    def __init__(self, bluez_manager, connection_manager):
        self.bluez_manager = bluez_manager
        self.connection_manager = connection_manager
        self.recovery_attempts = {}
        self.max_recovery_attempts = 3
        
    def start_bluetooth_with_recovery(self):
        """Start Bluetooth with automatic recovery."""
        for attempt in range(self.max_recovery_attempts):
            try:
                if self.bluez_manager.start_bluetooth():
                    print("Bluetooth started successfully")
                    return True
                else:
                    print(f"Bluetooth start failed, attempt {attempt + 1}")
                    
            except Exception as e:
                print(f"Bluetooth start error: {e}")
                
                # Try to recover
                if attempt < self.max_recovery_attempts - 1:
                    print("Attempting recovery...")
                    self.bluez_manager.stop_bluetooth()
                    time.sleep(2)
                    continue
                    
        print("Failed to start Bluetooth after all attempts")
        return False
    
    def connect_with_retry(self, address, max_retries=3):
        """Connect to device with retry logic."""
        for attempt in range(max_retries):
            try:
                if self.connection_manager.connect_device(address):
                    print(f"Connected to {address}")
                    return True
                else:
                    print(f"Connection attempt {attempt + 1} failed")
                    
            except Exception as e:
                print(f"Connection error: {e}")
                
            # Wait before retry
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                
        print(f"Failed to connect to {address} after {max_retries} attempts")
        return False
    
    def monitor_and_recover(self):
        """Monitor connections and recover from failures."""
        while True:
            try:
                # Check Bluetooth service status
                status = self.bluez_manager.get_bluetooth_status()
                
                if status['state'] == BluetoothState.ERROR:
                    print("Bluetooth service error detected, restarting...")
                    self.bluez_manager.restart_bluetooth()
                
                # Check device connections
                connected_devices = self.connection_manager.get_connected_devices()
                
                if len(connected_devices) < 2:  # Should have phone + headphones
                    print("Missing connections, attempting recovery...")
                    
                    # Try to reconnect to trusted devices
                    trusted_devices = [d for d in self.connection_manager.get_paired_devices() 
                                     if d.trusted and not d.connected]
                    
                    for device in trusted_devices:
                        self.connect_with_retry(device.address)
                
                # Sleep before next check
                time.sleep(30)
                
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(10)

# Use error handling
handler = BluetoothHandler(bluez_manager, connection_manager)
handler.start_bluetooth_with_recovery()

# Start monitoring in background
import threading
monitor_thread = threading.Thread(target=handler.monitor_and_recover, daemon=True)
monitor_thread.start()
```

This comprehensive Bluetooth module provides robust dual A2DP connection capabilities essential for The Silent Steno's audio intermediary functionality, with comprehensive error handling, automatic recovery, and persistent device management.