"""
USB Export System

Automatic USB drive detection and file transfer with progress monitoring.
"""

import os
import shutil
import logging
import psutil
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from enum import Enum
import threading
import subprocess

logger = logging.getLogger(__name__)

class USBStatus(Enum):
    """USB device status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    MOUNTING = "mounting"
    MOUNTED = "mounted"
    UNMOUNTING = "unmounting"
    ERROR = "error"

@dataclass
class USBDevice:
    """USB device information"""
    device_path: str
    mount_point: str
    label: str = ""
    filesystem: str = ""
    size_bytes: int = 0
    free_bytes: int = 0
    status: USBStatus = USBStatus.DISCONNECTED
    device_id: str = ""
    vendor: str = ""
    model: str = ""

@dataclass
class USBConfig:
    """USB export configuration"""
    auto_detect: bool = True
    auto_mount: bool = True
    safe_eject: bool = True
    create_folders: bool = True
    folder_structure: str = "SilentSteno/{date}/{session_id}"
    allowed_filesystems: List[str] = field(default_factory=lambda: ['fat32', 'ntfs', 'exfat', 'ext4'])
    min_free_space_mb: int = 100
    max_file_size_mb: int = 2048
    timeout_seconds: int = 30

@dataclass
class TransferProgress:
    """File transfer progress information"""
    total_bytes: int
    transferred_bytes: int
    current_file: str
    files_completed: int
    total_files: int
    speed_mbps: float
    eta_seconds: int
    status: str

class USBExporter:
    """USB drive detection and file transfer system"""
    
    def __init__(self, config: USBConfig):
        self.config = config
        self.connected_devices: Dict[str, USBDevice] = {}
        self.transfer_callbacks: List[Callable[[TransferProgress], None]] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self):
        """Start USB device monitoring"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_usb_devices)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("USB monitoring started")
    
    def stop_monitoring(self):
        """Stop USB device monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("USB monitoring stopped")
    
    def _monitor_usb_devices(self):
        """Monitor USB devices in background thread"""
        while self.monitoring:
            try:
                current_devices = self._scan_usb_devices()
                
                # Check for new devices
                for device_id, device in current_devices.items():
                    if device_id not in self.connected_devices:
                        self._handle_device_connected(device)
                
                # Check for removed devices
                for device_id in list(self.connected_devices.keys()):
                    if device_id not in current_devices:
                        self._handle_device_disconnected(device_id)
                
                self.connected_devices = current_devices
                time.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"USB monitoring error: {str(e)}")
                time.sleep(5)  # Wait longer on error
    
    def _scan_usb_devices(self) -> Dict[str, USBDevice]:
        """Scan for USB storage devices"""
        devices = {}
        
        try:
            # Get all disk partitions
            partitions = psutil.disk_partitions()
            
            for partition in partitions:
                # Check if it's a USB device
                if self._is_usb_device(partition.device):
                    try:
                        # Get device usage
                        usage = psutil.disk_usage(partition.mountpoint)
                        
                        device = USBDevice(
                            device_path=partition.device,
                            mount_point=partition.mountpoint,
                            filesystem=partition.fstype,
                            size_bytes=usage.total,
                            free_bytes=usage.free,
                            status=USBStatus.MOUNTED
                        )
                        
                        # Get device label and info
                        device.label = self._get_device_label(partition.device)
                        device.device_id = self._get_device_id(partition.device)
                        
                        devices[device.device_id] = device
                        
                    except Exception as e:
                        logger.warning(f"Error getting USB device info for {partition.device}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error scanning USB devices: {str(e)}")
        
        return devices
    
    def _is_usb_device(self, device_path: str) -> bool:
        """Check if device is a USB storage device"""
        try:
            # Check device path patterns
            if '/dev/sd' in device_path:
                # Check if it's a USB device via sysfs
                device_name = os.path.basename(device_path).rstrip('0123456789')
                sysfs_path = f'/sys/block/{device_name}/removable'
                
                if os.path.exists(sysfs_path):
                    with open(sysfs_path, 'r') as f:
                        return f.read().strip() == '1'
            
            return False
            
        except Exception:
            return False
    
    def _get_device_label(self, device_path: str) -> str:
        """Get device label"""
        try:
            result = subprocess.run(['blkid', '-s', 'LABEL', '-o', 'value', device_path], 
                                  capture_output=True, text=True, timeout=5)
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""
    
    def _get_device_id(self, device_path: str) -> str:
        """Get unique device ID"""
        try:
            result = subprocess.run(['blkid', '-s', 'UUID', '-o', 'value', device_path], 
                                  capture_output=True, text=True, timeout=5)
            uuid = result.stdout.strip() if result.returncode == 0 else ""
            return uuid or device_path
        except Exception:
            return device_path
    
    def _handle_device_connected(self, device: USBDevice):
        """Handle USB device connection"""
        logger.info(f"USB device connected: {device.label or device.device_path}")
        
        # Check if filesystem is supported
        if device.filesystem.lower() not in self.config.allowed_filesystems:
            logger.warning(f"Unsupported filesystem: {device.filesystem}")
            return
        
        # Check free space
        free_mb = device.free_bytes / (1024 * 1024)
        if free_mb < self.config.min_free_space_mb:
            logger.warning(f"Insufficient free space: {free_mb:.1f}MB")
            return
        
        # Device is ready for use
        device.status = USBStatus.CONNECTED
        logger.info(f"USB device ready: {device.label} ({free_mb:.1f}MB free)")
    
    def _handle_device_disconnected(self, device_id: str):
        """Handle USB device disconnection"""
        if device_id in self.connected_devices:
            device = self.connected_devices[device_id]
            logger.info(f"USB device disconnected: {device.label or device.device_path}")
            del self.connected_devices[device_id]
    
    def detect_usb_drives(self) -> List[USBDevice]:
        """Detect available USB drives"""
        devices = self._scan_usb_devices()
        return list(devices.values())
    
    def export_to_usb(self, file_path: str, destination_device: Optional[USBDevice] = None) -> bool:
        """Export file to USB device"""
        try:
            # Auto-detect device if not specified
            if destination_device is None:
                available_devices = self.detect_usb_drives()
                if not available_devices:
                    logger.error("No USB devices detected")
                    return False
                destination_device = available_devices[0]
            
            # Validate device
            if not self._validate_device(destination_device):
                return False
            
            # Create destination path
            file_path = Path(file_path)
            dest_path = self._create_destination_path(destination_device, file_path.name)
            
            # Start transfer
            return self._transfer_file(file_path, dest_path)
            
        except Exception as e:
            logger.error(f"USB export failed: {str(e)}")
            return False
    
    def _validate_device(self, device: USBDevice) -> bool:
        """Validate USB device for export"""
        # Check if device is mounted
        if device.status != USBStatus.MOUNTED:
            logger.error(f"Device not mounted: {device.device_path}")
            return False
        
        # Check free space
        free_mb = device.free_bytes / (1024 * 1024)
        if free_mb < self.config.min_free_space_mb:
            logger.error(f"Insufficient free space: {free_mb:.1f}MB")
            return False
        
        # Check if mount point is accessible
        if not os.path.exists(device.mount_point):
            logger.error(f"Mount point not accessible: {device.mount_point}")
            return False
        
        try:
            # Test write access
            test_file = Path(device.mount_point) / '.write_test'
            test_file.write_text('test')
            test_file.unlink()
            return True
            
        except Exception as e:
            logger.error(f"Device not writable: {str(e)}")
            return False
    
    def _create_destination_path(self, device: USBDevice, filename: str) -> Path:
        """Create destination path with folder structure"""
        # Format folder structure
        folder_vars = {
            'date': time.strftime('%Y-%m-%d'),
            'session_id': 'session_' + str(int(time.time())),
            'device_label': device.label or 'Unknown'
        }
        
        folder_path = self.config.folder_structure.format(**folder_vars)
        full_path = Path(device.mount_point) / folder_path
        
        # Create directories if needed
        if self.config.create_folders:
            full_path.mkdir(parents=True, exist_ok=True)
        
        return full_path / filename
    
    def _transfer_file(self, source: Path, destination: Path) -> bool:
        """Transfer file with progress tracking"""
        try:
            # Get file size
            file_size = source.stat().st_size
            
            # Check size limits
            size_mb = file_size / (1024 * 1024)
            if size_mb > self.config.max_file_size_mb:
                logger.error(f"File too large: {size_mb:.1f}MB")
                return False
            
            # Start transfer
            logger.info(f"Starting transfer: {source.name} -> {destination}")
            
            # Create progress tracker
            progress = TransferProgress(
                total_bytes=file_size,
                transferred_bytes=0,
                current_file=source.name,
                files_completed=0,
                total_files=1,
                speed_mbps=0,
                eta_seconds=0,
                status="transferring"
            )
            
            # Copy file with progress
            start_time = time.time()
            
            with open(source, 'rb') as src, open(destination, 'wb') as dst:
                chunk_size = 64 * 1024  # 64KB chunks
                
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    
                    dst.write(chunk)
                    progress.transferred_bytes += len(chunk)
                    
                    # Calculate progress
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        progress.speed_mbps = (progress.transferred_bytes / (1024 * 1024)) / elapsed
                        remaining_bytes = progress.total_bytes - progress.transferred_bytes
                        if progress.speed_mbps > 0:
                            progress.eta_seconds = int(remaining_bytes / (progress.speed_mbps * 1024 * 1024))
                    
                    # Notify callbacks
                    for callback in self.transfer_callbacks:
                        callback(progress)
            
            # Sync to ensure data is written
            os.sync()
            
            progress.status = "completed"
            progress.files_completed = 1
            
            for callback in self.transfer_callbacks:
                callback(progress)
            
            logger.info(f"Transfer completed: {source.name}")
            return True
            
        except Exception as e:
            logger.error(f"Transfer failed: {str(e)}")
            return False
    
    def add_progress_callback(self, callback: Callable[[TransferProgress], None]):
        """Add transfer progress callback"""
        self.transfer_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable[[TransferProgress], None]):
        """Remove transfer progress callback"""
        if callback in self.transfer_callbacks:
            self.transfer_callbacks.remove(callback)
    
    def safe_eject_device(self, device: USBDevice) -> bool:
        """Safely eject USB device"""
        try:
            if self.config.safe_eject:
                # Sync data
                os.sync()
                
                # Unmount device
                result = subprocess.run(['umount', device.mount_point], 
                                      capture_output=True, timeout=10)
                
                if result.returncode == 0:
                    logger.info(f"Device safely ejected: {device.label}")
                    return True
                else:
                    logger.error(f"Failed to eject device: {result.stderr}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Safe eject failed: {str(e)}")
            return False

def create_usb_exporter(config: Optional[USBConfig] = None) -> USBExporter:
    """Create USB exporter with default or provided configuration"""
    if config is None:
        config = USBConfig()
    
    return USBExporter(config)

def detect_usb_drives() -> List[USBDevice]:
    """Convenience function to detect USB drives"""
    exporter = create_usb_exporter()
    return exporter.detect_usb_drives()

def export_to_usb(file_path: str, device: Optional[USBDevice] = None) -> bool:
    """Convenience function to export file to USB"""
    exporter = create_usb_exporter()
    return exporter.export_to_usb(file_path, device)

def format_usb_export(file_path: str, session_data: Dict[str, Any]) -> str:
    """Format USB export with session data"""
    # Implementation for USB-specific formatting
    return file_path

def monitor_transfer_progress(callback: Callable[[TransferProgress], None]) -> USBExporter:
    """Create USB exporter with progress monitoring"""
    exporter = create_usb_exporter()
    exporter.add_progress_callback(callback)
    return exporter