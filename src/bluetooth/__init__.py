"""
Bluetooth module for The Silent Steno

This module provides comprehensive Bluetooth management functionality
including BlueZ stack management, device connection handling, and
A2DP audio profile support for dual connections.
"""

from .bluez_manager import BlueZManager, start_bluetooth, stop_bluetooth, get_bluetooth_status
from .connection_manager import ConnectionManager, pair_device, connect_device, manage_connections, DeviceRole

__all__ = [
    'BlueZManager',
    'ConnectionManager', 
    'DeviceRole',
    'start_bluetooth',
    'stop_bluetooth', 
    'get_bluetooth_status',
    'pair_device',
    'connect_device',
    'manage_connections'
]