#!/usr/bin/env python3

"""
BlueZ Bluetooth Stack Manager for The Silent Steno

This module provides a Python interface for controlling the BlueZ Bluetooth stack
with A2DP support for dual audio connections (sink and source).

Key features:
- BlueZ service management and status monitoring
- A2DP sink and source configuration
- High-quality codec support (SBC, AAC, aptX, Samsung Scalable)
- Device discovery and management
- Connection status monitoring
"""

import subprocess
import logging
import time
import dbus
import dbus.mainloop.glib
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BluetoothState(Enum):
    """Bluetooth service states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"
    UNKNOWN = "unknown"


class CodecType(Enum):
    """Supported audio codecs"""
    SBC = "sbc"
    AAC = "aac"
    APTX = "aptx"
    APTX_HD = "aptx_hd"
    SAMSUNG_SCALABLE = "samsung_scalable"


class BlueZManager:
    """
    Main BlueZ management class for The Silent Steno
    
    Provides high-level interface for Bluetooth operations including
    A2DP sink/source management, device pairing, and codec configuration.
    """
    
    def __init__(self):
        """Initialize BlueZ manager"""
        self.bus = None
        self.adapter = None
        self.adapter_path = "/org/bluez/hci0"
        self.service_name = "org.bluez"
        self._connected_devices = {}
        self._supported_codecs = []
        
        # Initialize D-Bus connection
        self._init_dbus()
        
    def _init_dbus(self):
        """Initialize D-Bus connection for BlueZ communication"""
        try:
            dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
            self.bus = dbus.SystemBus()
            logger.info("D-Bus connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize D-Bus: {e}")
            raise
    
    def start_bluetooth(self) -> bool:
        """
        Start the Bluetooth service
        
        Returns:
            bool: True if service started successfully, False otherwise
        """
        try:
            logger.info("Starting Bluetooth service...")
            result = subprocess.run(
                ["sudo", "systemctl", "start", "bluetooth"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Wait for service to be fully ready
                time.sleep(2)
                logger.info("Bluetooth service started successfully")
                return self._verify_adapter()
            else:
                logger.error(f"Failed to start Bluetooth service: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Timeout starting Bluetooth service")
            return False
        except Exception as e:
            logger.error(f"Error starting Bluetooth service: {e}")
            return False
    
    def stop_bluetooth(self) -> bool:
        """
        Stop the Bluetooth service
        
        Returns:
            bool: True if service stopped successfully, False otherwise
        """
        try:
            logger.info("Stopping Bluetooth service...")
            result = subprocess.run(
                ["sudo", "systemctl", "stop", "bluetooth"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info("Bluetooth service stopped successfully")
                return True
            else:
                logger.error(f"Failed to stop Bluetooth service: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Timeout stopping Bluetooth service")
            return False
        except Exception as e:
            logger.error(f"Error stopping Bluetooth service: {e}")
            return False
    
    def restart_bluetooth(self) -> bool:
        """
        Restart the Bluetooth service
        
        Returns:
            bool: True if service restarted successfully, False otherwise
        """
        try:
            logger.info("Restarting Bluetooth service...")
            result = subprocess.run(
                ["sudo", "systemctl", "restart", "bluetooth"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Wait for service to be fully ready
                time.sleep(3)
                logger.info("Bluetooth service restarted successfully")
                return self._verify_adapter()
            else:
                logger.error(f"Failed to restart Bluetooth service: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Timeout restarting Bluetooth service")
            return False
        except Exception as e:
            logger.error(f"Error restarting Bluetooth service: {e}")
            return False
    
    def get_bluetooth_status(self) -> Dict[str, Any]:
        """
        Get comprehensive Bluetooth service status
        
        Returns:
            Dict containing service status, adapter info, and connected devices
        """
        status = {
            "service_state": BluetoothState.UNKNOWN,
            "service_active": False,
            "adapter_present": False,
            "adapter_powered": False,
            "adapter_discoverable": False,
            "adapter_pairable": False,
            "connected_devices": [],
            "supported_codecs": [],
            "a2dp_sink_available": False,
            "a2dp_source_available": False
        }
        
        try:
            # Check systemd service status
            result = subprocess.run(
                ["systemctl", "is-active", "bluetooth"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip() == "active":
                status["service_active"] = True
                status["service_state"] = BluetoothState.RUNNING
            else:
                status["service_state"] = BluetoothState.STOPPED
                return status
            
            # Check adapter status via D-Bus
            adapter_status = self._get_adapter_status()
            status.update(adapter_status)
            
            # Check connected devices
            status["connected_devices"] = self._get_connected_devices()
            
            # Check codec support
            status["supported_codecs"] = self._get_supported_codecs()
            
            # Check A2DP profiles
            status["a2dp_sink_available"] = self._check_a2dp_sink()
            status["a2dp_source_available"] = self._check_a2dp_source()
            
        except Exception as e:
            logger.error(f"Error getting Bluetooth status: {e}")
            status["service_state"] = BluetoothState.ERROR
        
        return status
    
    def _verify_adapter(self) -> bool:
        """Verify Bluetooth adapter is available and functional"""
        try:
            adapter_obj = self.bus.get_object(self.service_name, self.adapter_path)
            adapter_props = dbus.Interface(adapter_obj, "org.freedesktop.DBus.Properties")
            
            # Check if adapter is powered
            powered = adapter_props.Get("org.bluez.Adapter1", "Powered")
            if not powered:
                # Try to power on the adapter
                adapter_props.Set("org.bluez.Adapter1", "Powered", True)
                time.sleep(1)
            
            logger.info("Bluetooth adapter verified and powered on")
            return True
            
        except dbus.DBusException as e:
            logger.error(f"D-Bus error verifying adapter: {e}")
            return False
        except Exception as e:
            logger.error(f"Error verifying adapter: {e}")
            return False
    
    def _get_adapter_status(self) -> Dict[str, Any]:
        """Get detailed adapter status via D-Bus"""
        adapter_status = {
            "adapter_present": False,
            "adapter_powered": False,
            "adapter_discoverable": False,
            "adapter_pairable": False
        }
        
        try:
            adapter_obj = self.bus.get_object(self.service_name, self.adapter_path)
            adapter_props = dbus.Interface(adapter_obj, "org.freedesktop.DBus.Properties")
            
            adapter_status["adapter_present"] = True
            adapter_status["adapter_powered"] = bool(adapter_props.Get("org.bluez.Adapter1", "Powered"))
            adapter_status["adapter_discoverable"] = bool(adapter_props.Get("org.bluez.Adapter1", "Discoverable"))
            adapter_status["adapter_pairable"] = bool(adapter_props.Get("org.bluez.Adapter1", "Pairable"))
            
        except dbus.DBusException as e:
            logger.warning(f"Could not get adapter status: {e}")
        except Exception as e:
            logger.error(f"Error getting adapter status: {e}")
        
        return adapter_status
    
    def _get_connected_devices(self) -> List[Dict[str, str]]:
        """Get list of connected Bluetooth devices"""
        devices = []
        try:
            # Use bluetoothctl to get connected devices
            result = subprocess.run(
                ["bluetoothctl", "devices", "Connected"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('Device'):
                        parts = line.split(' ', 2)
                        if len(parts) >= 3:
                            devices.append({
                                "address": parts[1],
                                "name": parts[2] if len(parts) > 2 else "Unknown"
                            })
        except Exception as e:
            logger.warning(f"Could not get connected devices: {e}")
        
        return devices
    
    def _get_supported_codecs(self) -> List[str]:
        """Get list of supported audio codecs"""
        codecs = ["SBC"]  # SBC is always supported
        
        try:
            # Check for additional codec support
            result = subprocess.run(
                ["pactl", "list", "modules"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                output = result.stdout.lower()
                if "aac" in output:
                    codecs.append("AAC")
                if "aptx" in output:
                    codecs.append("aptX")
                if "scalable" in output:
                    codecs.append("Samsung Scalable")
                    
        except Exception as e:
            logger.warning(f"Could not check codec support: {e}")
        
        return codecs
    
    def _check_a2dp_sink(self) -> bool:
        """Check if A2DP sink profile is available"""
        try:
            result = subprocess.run(
                ["pactl", "list", "modules"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return "module-bluez5-device" in result.stdout
                
        except Exception as e:
            logger.warning(f"Could not check A2DP sink: {e}")
        
        return False
    
    def _check_a2dp_source(self) -> bool:
        """Check if A2DP source profile is available"""
        try:
            result = subprocess.run(
                ["pactl", "list", "modules"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return "module-bluez5-discover" in result.stdout
                
        except Exception as e:
            logger.warning(f"Could not check A2DP source: {e}")
        
        return False
    
    def enable_discoverable(self, timeout: int = 180) -> bool:
        """
        Make the device discoverable for pairing
        
        Args:
            timeout: Discovery timeout in seconds (0 = indefinite)
            
        Returns:
            bool: True if discoverable mode enabled successfully
        """
        try:
            adapter_obj = self.bus.get_object(self.service_name, self.adapter_path)
            adapter_props = dbus.Interface(adapter_obj, "org.freedesktop.DBus.Properties")
            
            # Enable pairable and discoverable
            adapter_props.Set("org.bluez.Adapter1", "Pairable", True)
            adapter_props.Set("org.bluez.Adapter1", "Discoverable", True)
            
            if timeout > 0:
                adapter_props.Set("org.bluez.Adapter1", "DiscoverableTimeout", dbus.UInt32(timeout))
            
            logger.info(f"Device is now discoverable for {timeout} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable discoverable mode: {e}")
            return False
    
    def disable_discoverable(self) -> bool:
        """
        Disable discoverable mode
        
        Returns:
            bool: True if discoverable mode disabled successfully
        """
        try:
            adapter_obj = self.bus.get_object(self.service_name, self.adapter_path)
            adapter_props = dbus.Interface(adapter_obj, "org.freedesktop.DBus.Properties")
            
            adapter_props.Set("org.bluez.Adapter1", "Discoverable", False)
            
            logger.info("Device is no longer discoverable")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disable discoverable mode: {e}")
            return False


# Convenience functions for external use
def start_bluetooth() -> bool:
    """Start Bluetooth service - convenience function"""
    manager = BlueZManager()
    return manager.start_bluetooth()


def stop_bluetooth() -> bool:
    """Stop Bluetooth service - convenience function"""
    manager = BlueZManager()
    return manager.stop_bluetooth()


def get_bluetooth_status() -> Dict[str, Any]:
    """Get Bluetooth status - convenience function"""
    manager = BlueZManager()
    return manager.get_bluetooth_status()


if __name__ == "__main__":
    # Basic test when run directly
    print("BlueZ Manager Test")
    print("=" * 40)
    
    manager = BlueZManager()
    status = manager.get_bluetooth_status()
    
    print(f"Service Active: {status['service_active']}")
    print(f"Service State: {status['service_state'].value}")
    print(f"Adapter Present: {status['adapter_present']}")
    print(f"Adapter Powered: {status['adapter_powered']}")
    print(f"A2DP Sink Available: {status['a2dp_sink_available']}")
    print(f"A2DP Source Available: {status['a2dp_source_available']}")
    print(f"Supported Codecs: {', '.join(status['supported_codecs'])}")
    print(f"Connected Devices: {len(status['connected_devices'])}")
    
    for device in status['connected_devices']:
        print(f"  - {device['name']} ({device['address']})")