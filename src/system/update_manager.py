"""
Update Management System

Software update mechanism with rollback capabilities, automated deployment,
and validation for The Silent Steno device.
"""

import os
import shutil
import subprocess
import logging
import threading
import time
import hashlib
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from enum import Enum
import requests
import tarfile
import tempfile
from packaging import version


class UpdateStatus(Enum):
    IDLE = "idle"
    CHECKING = "checking"
    DOWNLOADING = "downloading"
    INSTALLING = "installing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"


class UpdateType(Enum):
    SECURITY = "security"
    BUGFIX = "bugfix"
    FEATURE = "feature"
    MAJOR = "major"


@dataclass
class UpdateInfo:
    """Information about an available update."""
    version: str
    update_type: UpdateType
    release_date: datetime
    download_url: str
    checksum: str
    size_bytes: int
    description: str
    changelog: List[str] = field(default_factory=list)
    required: bool = False
    rollback_supported: bool = True
    min_version: Optional[str] = None
    max_version: Optional[str] = None
    
    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 ** 2)


@dataclass
class UpdateResult:
    """Result of an update operation."""
    success: bool
    version: str
    update_type: UpdateType
    duration_seconds: float
    errors: List[str] = field(default_factory=list)
    rollback_available: bool = False
    validation_passed: bool = False
    backup_created: bool = False


@dataclass
class UpdateConfig:
    """Configuration for update management."""
    auto_update_enabled: bool = True
    update_channel: str = "stable"  # stable, beta, alpha
    update_window_start: str = "02:00"  # 2 AM
    update_window_end: str = "04:00"  # 4 AM
    max_update_size_mb: int = 500
    backup_before_update: bool = True
    validate_after_update: bool = True
    rollback_on_failure: bool = True
    check_interval_hours: int = 24
    security_updates_immediate: bool = True
    allowed_update_types: List[UpdateType] = field(default_factory=lambda: [
        UpdateType.SECURITY, UpdateType.BUGFIX, UpdateType.FEATURE
    ])


class UpdateValidator:
    """Validates system state before and after updates."""
    
    def __init__(self, config: UpdateConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_pre_update(self) -> bool:
        """Validate system state before update."""
        try:
            # Check available space
            stat = shutil.disk_usage("/")
            free_mb = stat.free / (1024 ** 2)
            
            if free_mb < self.config.max_update_size_mb * 2:  # Need double space for safety
                self.logger.error(f"Insufficient disk space: {free_mb:.1f}MB available")
                return False
            
            # Check system load
            if os.path.exists("/proc/loadavg"):
                with open("/proc/loadavg", "r") as f:
                    load_avg = float(f.read().split()[0])
                    if load_avg > 2.0:  # High system load
                        self.logger.warning(f"High system load: {load_avg}")
                        return False
            
            # Check if critical services are running
            critical_services = ["bluetooth", "pulseaudio"]
            for service in critical_services:
                if not self._check_service_status(service):
                    self.logger.warning(f"Critical service {service} not running")
            
            # Check database integrity
            if not self._check_database_integrity():
                self.logger.error("Database integrity check failed")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pre-update validation failed: {e}")
            return False
    
    def validate_post_update(self) -> bool:
        """Validate system state after update."""
        try:
            # Check if application starts correctly
            if not self._check_application_startup():
                self.logger.error("Application startup check failed")
                return False
            
            # Check if critical components are functional
            if not self._check_critical_components():
                self.logger.error("Critical components check failed")
                return False
            
            # Check configuration integrity
            if not self._check_configuration_integrity():
                self.logger.error("Configuration integrity check failed")
                return False
            
            # Check database after update
            if not self._check_database_integrity():
                self.logger.error("Database integrity check failed after update")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Post-update validation failed: {e}")
            return False
    
    def _check_service_status(self, service: str) -> bool:
        """Check if a system service is running."""
        try:
            result = subprocess.run(
                ["systemctl", "is-active", service],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _check_database_integrity(self) -> bool:
        """Check database integrity."""
        try:
            # Simple database check - this would be expanded for real use
            db_path = Path("/home/mmariani/projects/thesilentsteno/data/silentst.db")
            if db_path.exists():
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check;")
                result = cursor.fetchone()
                conn.close()
                return result[0] == "ok"
            return True
        except Exception as e:
            self.logger.error(f"Database integrity check error: {e}")
            return False
    
    def _check_application_startup(self) -> bool:
        """Check if the application can start successfully."""
        try:
            # Mock application startup check
            # In real implementation, this would test actual application startup
            return True
        except Exception as e:
            self.logger.error(f"Application startup check failed: {e}")
            return False
    
    def _check_critical_components(self) -> bool:
        """Check if critical system components are functional."""
        try:
            # Check audio system
            if not self._check_audio_system():
                return False
            
            # Check bluetooth
            if not self._check_bluetooth_system():
                return False
            
            # Check UI system
            if not self._check_ui_system():
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Critical components check failed: {e}")
            return False
    
    def _check_audio_system(self) -> bool:
        """Check audio system functionality."""
        try:
            # Simple audio system check
            return os.path.exists("/dev/snd")
        except Exception:
            return False
    
    def _check_bluetooth_system(self) -> bool:
        """Check Bluetooth system functionality."""
        try:
            # Simple Bluetooth check
            return os.path.exists("/sys/class/bluetooth")
        except Exception:
            return False
    
    def _check_ui_system(self) -> bool:
        """Check UI system functionality."""
        try:
            # Simple UI system check
            return True  # Mock for now
        except Exception:
            return False
    
    def _check_configuration_integrity(self) -> bool:
        """Check configuration file integrity."""
        try:
            config_files = [
                "/home/mmariani/projects/thesilentsteno/config/app_config.json",
                "/home/mmariani/projects/thesilentsteno/config/logging_config.json"
            ]
            
            for config_file in config_files:
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        json.load(f)  # This will raise exception if invalid JSON
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration integrity check failed: {e}")
            return False


class RollbackManager:
    """Manages system rollback operations."""
    
    def __init__(self, backup_dir: str = "/home/mmariani/projects/thesilentsteno/backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def create_backup(self, version: str) -> bool:
        """Create a system backup before update."""
        try:
            backup_name = f"backup_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = self.backup_dir / backup_name
            
            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Backup critical directories
            backup_targets = [
                "/home/mmariani/projects/thesilentsteno/src",
                "/home/mmariani/projects/thesilentsteno/config",
                "/home/mmariani/projects/thesilentsteno/data"
            ]
            
            for target in backup_targets:
                if os.path.exists(target):
                    target_path = Path(target)
                    backup_target = backup_path / target_path.name
                    
                    if target_path.is_dir():
                        shutil.copytree(target, backup_target)
                    else:
                        shutil.copy2(target, backup_target)
            
            # Create backup metadata
            metadata = {
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "backup_path": str(backup_path),
                "targets": backup_targets
            }
            
            with open(backup_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"System backup created: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return False
    
    def perform_rollback(self, backup_name: str = None) -> bool:
        """Perform system rollback to previous version."""
        try:
            if backup_name is None:
                backup_name = self._get_latest_backup()
            
            if not backup_name:
                self.logger.error("No backup available for rollback")
                return False
            
            backup_path = self.backup_dir / backup_name
            
            if not backup_path.exists():
                self.logger.error(f"Backup not found: {backup_path}")
                return False
            
            # Load backup metadata
            metadata_path = backup_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {"targets": []}
            
            # Restore backed up files
            for target in metadata.get("targets", []):
                source = backup_path / Path(target).name
                if source.exists():
                    if os.path.exists(target):
                        if os.path.isdir(target):
                            shutil.rmtree(target)
                        else:
                            os.remove(target)
                    
                    if source.is_dir():
                        shutil.copytree(source, target)
                    else:
                        shutil.copy2(source, target)
            
            self.logger.info(f"System rolled back to backup: {backup_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False
    
    def _get_latest_backup(self) -> Optional[str]:
        """Get the name of the latest backup."""
        try:
            backups = []
            for item in self.backup_dir.iterdir():
                if item.is_dir() and item.name.startswith("backup_"):
                    backups.append(item.name)
            
            if backups:
                return sorted(backups)[-1]  # Return latest backup
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding latest backup: {e}")
            return None
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        backups = []
        
        try:
            for item in self.backup_dir.iterdir():
                if item.is_dir() and item.name.startswith("backup_"):
                    metadata_path = item / "metadata.json"
                    
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    else:
                        metadata = {"version": "unknown", "timestamp": "unknown"}
                    
                    backups.append({
                        "name": item.name,
                        "version": metadata.get("version", "unknown"),
                        "timestamp": metadata.get("timestamp", "unknown"),
                        "size": sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    })
            
            return sorted(backups, key=lambda x: x["timestamp"], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error listing backups: {e}")
            return []


class UpdateManager:
    """Main update management system."""
    
    def __init__(self, config: UpdateConfig = None):
        self.config = config or UpdateConfig()
        self.logger = logging.getLogger(__name__)
        self.validator = UpdateValidator(self.config)
        self.rollback_manager = RollbackManager()
        self.status = UpdateStatus.IDLE
        self.current_version = "0.1.0"  # Would be read from version file
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._update_thread: Optional[threading.Thread] = None
    
    def check_for_updates(self) -> List[UpdateInfo]:
        """Check for available updates."""
        try:
            with self._lock:
                self.status = UpdateStatus.CHECKING
                
                # Mock update check - in real implementation, this would contact update server
                updates = self._mock_update_check()
                
                self.status = UpdateStatus.IDLE
                return updates
                
        except Exception as e:
            self.logger.error(f"Update check failed: {e}")
            self.status = UpdateStatus.FAILED
            return []
    
    def install_update(self, update_info: UpdateInfo) -> UpdateResult:
        """Install a specific update."""
        start_time = time.time()
        result = UpdateResult(
            success=False,
            version=update_info.version,
            update_type=update_info.update_type,
            duration_seconds=0.0
        )
        
        try:
            with self._lock:
                self.status = UpdateStatus.DOWNLOADING
                
                # Pre-update validation
                if not self.validator.validate_pre_update():
                    result.errors.append("Pre-update validation failed")
                    self.status = UpdateStatus.FAILED
                    return result
                
                # Create backup if configured
                if self.config.backup_before_update:
                    if self.rollback_manager.create_backup(self.current_version):
                        result.backup_created = True
                    else:
                        result.errors.append("Backup creation failed")
                        if self.config.rollback_on_failure:
                            self.status = UpdateStatus.FAILED
                            return result
                
                # Download update
                update_file = self._download_update(update_info)
                if not update_file:
                    result.errors.append("Download failed")
                    self.status = UpdateStatus.FAILED
                    return result
                
                # Install update
                self.status = UpdateStatus.INSTALLING
                if not self._install_update_file(update_file):
                    result.errors.append("Installation failed")
                    self.status = UpdateStatus.FAILED
                    
                    if self.config.rollback_on_failure:
                        self._perform_rollback()
                    
                    return result
                
                # Post-update validation
                if self.config.validate_after_update:
                    self.status = UpdateStatus.VALIDATING
                    if self.validator.validate_post_update():
                        result.validation_passed = True
                    else:
                        result.errors.append("Post-update validation failed")
                        
                        if self.config.rollback_on_failure:
                            self._perform_rollback()
                            self.status = UpdateStatus.FAILED
                            return result
                
                # Update successful
                self.current_version = update_info.version
                result.success = True
                result.rollback_available = result.backup_created
                self.status = UpdateStatus.COMPLETED
                
                self.logger.info(f"Update to {update_info.version} completed successfully")
                
        except Exception as e:
            result.errors.append(str(e))
            self.logger.error(f"Update installation failed: {e}")
            self.status = UpdateStatus.FAILED
            
            if self.config.rollback_on_failure:
                self._perform_rollback()
        
        finally:
            result.duration_seconds = time.time() - start_time
        
        return result
    
    def rollback_update(self, backup_name: str = None) -> bool:
        """Rollback to a previous version."""
        try:
            with self._lock:
                self.status = UpdateStatus.ROLLING_BACK
                
                if self.rollback_manager.perform_rollback(backup_name):
                    self.status = UpdateStatus.COMPLETED
                    self.logger.info("Update rollback completed successfully")
                    return True
                else:
                    self.status = UpdateStatus.FAILED
                    self.logger.error("Update rollback failed")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            self.status = UpdateStatus.FAILED
            return False
    
    def _mock_update_check(self) -> List[UpdateInfo]:
        """Mock update check for demonstration."""
        # In real implementation, this would contact an update server
        return [
            UpdateInfo(
                version="0.1.1",
                update_type=UpdateType.BUGFIX,
                release_date=datetime.now(),
                download_url="https://example.com/update.tar.gz",
                checksum="abc123",
                size_bytes=1024 * 1024,  # 1MB
                description="Bug fixes and stability improvements"
            )
        ]
    
    def _download_update(self, update_info: UpdateInfo) -> Optional[Path]:
        """Download update file."""
        try:
            # Mock download - in real implementation, this would download from URL
            temp_dir = Path(tempfile.mkdtemp())
            update_file = temp_dir / "update.tar.gz"
            
            # Create a mock update file
            update_file.write_bytes(b"mock update content")
            
            self.logger.info(f"Downloaded update {update_info.version}")
            return update_file
            
        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            return None
    
    def _install_update_file(self, update_file: Path) -> bool:
        """Install update from file."""
        try:
            # Mock installation - in real implementation, this would extract and install
            self.logger.info(f"Installing update from {update_file}")
            
            # Simulate installation time
            time.sleep(1.0)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Installation failed: {e}")
            return False
    
    def _perform_rollback(self):
        """Perform automatic rollback."""
        try:
            self.status = UpdateStatus.ROLLING_BACK
            self.rollback_manager.perform_rollback()
            self.logger.info("Automatic rollback completed")
        except Exception as e:
            self.logger.error(f"Automatic rollback failed: {e}")
    
    def get_update_status(self) -> Dict[str, Any]:
        """Get current update status."""
        return {
            "status": self.status.value,
            "current_version": self.current_version,
            "config": {
                "auto_update_enabled": self.config.auto_update_enabled,
                "update_channel": self.config.update_channel,
                "backup_before_update": self.config.backup_before_update,
                "validate_after_update": self.config.validate_after_update
            },
            "backups": self.rollback_manager.list_backups()
        }


# Factory functions
def create_update_manager(config: UpdateConfig = None) -> UpdateManager:
    """Create an update manager instance."""
    return UpdateManager(config)


def check_updates(manager: UpdateManager = None) -> List[UpdateInfo]:
    """Check for available updates."""
    if manager is None:
        manager = create_update_manager()
    
    return manager.check_for_updates()


def install_update(update_info: UpdateInfo, manager: UpdateManager = None) -> UpdateResult:
    """Install a specific update."""
    if manager is None:
        manager = create_update_manager()
    
    return manager.install_update(update_info)


def rollback_update(backup_name: str = None, manager: UpdateManager = None) -> bool:
    """Rollback to a previous version."""
    if manager is None:
        manager = create_update_manager()
    
    return manager.rollback_update(backup_name)