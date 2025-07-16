"""
Factory Reset System

Factory reset functionality with backup and recovery capabilities
for The Silent Steno device.
"""

import os
import shutil
import logging
import time
import json
import sqlite3
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from enum import Enum
import subprocess
import threading
import tarfile
import tempfile


class ResetType(Enum):
    FACTORY_RESET = "factory_reset"
    CONFIGURATION_RESET = "configuration_reset"
    DATA_RESET = "data_reset"
    SETTINGS_RESET = "settings_reset"
    PARTIAL_RESET = "partial_reset"


class ResetStatus(Enum):
    IDLE = "idle"
    BACKING_UP = "backing_up"
    RESETTING = "resetting"
    RESTORING = "restoring"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ResetConfig:
    """Configuration for reset operations."""
    backup_before_reset: bool = True
    backup_location: str = "/home/mmariani/projects/thesilentsteno/backups"
    preserve_hardware_config: bool = True
    preserve_network_config: bool = True
    preserve_user_data: bool = False
    reset_components: List[str] = field(default_factory=lambda: ["application", "configuration", "logs"])
    verification_enabled: bool = True
    rollback_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "backup_before_reset": self.backup_before_reset,
            "backup_location": self.backup_location,
            "preserve_hardware_config": self.preserve_hardware_config,
            "preserve_network_config": self.preserve_network_config,
            "preserve_user_data": self.preserve_user_data,
            "reset_components": self.reset_components,
            "verification_enabled": self.verification_enabled,
            "rollback_enabled": self.rollback_enabled
        }


@dataclass
class ResetResult:
    """Result of a reset operation."""
    success: bool
    reset_type: ResetType
    duration_seconds: float
    backup_created: bool = False
    backup_location: Optional[str] = None
    components_reset: List[str] = field(default_factory=list)
    preserved_items: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    rollback_available: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "reset_type": self.reset_type.value,
            "duration_seconds": self.duration_seconds,
            "backup_created": self.backup_created,
            "backup_location": self.backup_location,
            "components_reset": self.components_reset,
            "preserved_items": self.preserved_items,
            "errors": self.errors,
            "rollback_available": self.rollback_available
        }


@dataclass
class BackupItem:
    """Information about a backup item."""
    name: str
    source_path: str
    backup_path: str
    size_bytes: int
    timestamp: datetime
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "source_path": self.source_path,
            "backup_path": self.backup_path,
            "size_bytes": self.size_bytes,
            "timestamp": self.timestamp.isoformat(),
            "checksum": self.checksum
        }


class BackupManager:
    """Manages backup operations for factory reset."""
    
    def __init__(self, backup_dir: str = "/home/mmariani/projects/thesilentsteno/backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def create_backup(self, reset_type: ResetType, config: ResetConfig) -> Optional[str]:
        """Create a backup before reset operation."""
        try:
            # Create backup directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"pre_reset_{reset_type.value}_{timestamp}"
            backup_path = self.backup_dir / backup_name
            backup_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Creating backup: {backup_name}")
            
            # Define backup items based on reset type
            backup_items = self._get_backup_items(reset_type, config)
            
            # Create backup metadata
            backup_metadata = {
                "backup_name": backup_name,
                "reset_type": reset_type.value,
                "timestamp": datetime.now().isoformat(),
                "config": config.to_dict(),
                "items": []
            }
            
            # Backup each item
            for item in backup_items:
                if self._backup_item(item, backup_path):
                    backup_metadata["items"].append(item.to_dict())
            
            # Save metadata
            metadata_path = backup_path / "backup_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(backup_metadata, f, indent=2)
            
            self.logger.info(f"Backup created successfully: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return None
    
    def _get_backup_items(self, reset_type: ResetType, config: ResetConfig) -> List[BackupItem]:
        """Get list of items to backup based on reset type."""
        items = []
        base_path = Path("/home/mmariani/projects/thesilentsteno")
        
        if reset_type == ResetType.FACTORY_RESET:
            # Backup everything except what's explicitly excluded
            backup_paths = [
                ("application", base_path / "src"),
                ("configuration", base_path / "config"),
                ("data", base_path / "data"),
                ("logs", base_path / "logs"),
                ("scripts", base_path / "scripts"),
                ("docs", base_path / "docs")
            ]
        elif reset_type == ResetType.CONFIGURATION_RESET:
            backup_paths = [
                ("configuration", base_path / "config")
            ]
        elif reset_type == ResetType.DATA_RESET:
            backup_paths = [
                ("data", base_path / "data")
            ]
        elif reset_type == ResetType.SETTINGS_RESET:
            backup_paths = [
                ("settings", base_path / "config" / "app_config.json"),
                ("logging_config", base_path / "config" / "logging_config.json")
            ]
        else:
            backup_paths = [
                ("application", base_path / "src"),
                ("configuration", base_path / "config")
            ]
        
        # Create backup items
        for name, path in backup_paths:
            if path.exists():
                items.append(BackupItem(
                    name=name,
                    source_path=str(path),
                    backup_path="",  # Will be set during backup
                    size_bytes=self._get_path_size(path),
                    timestamp=datetime.now()
                ))
        
        return items
    
    def _backup_item(self, item: BackupItem, backup_path: Path) -> bool:
        """Backup a single item."""
        try:
            source_path = Path(item.source_path)
            target_path = backup_path / source_path.name
            
            if source_path.is_dir():
                shutil.copytree(source_path, target_path)
            else:
                shutil.copy2(source_path, target_path)
            
            item.backup_path = str(target_path)
            self.logger.debug(f"Backed up {source_path} to {target_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to backup {item.source_path}: {e}")
            return False
    
    def _get_path_size(self, path: Path) -> int:
        """Get total size of a path (file or directory)."""
        try:
            if path.is_file():
                return path.stat().st_size
            elif path.is_dir():
                total_size = 0
                for item in path.rglob('*'):
                    if item.is_file():
                        total_size += item.stat().st_size
                return total_size
            else:
                return 0
        except Exception:
            return 0
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        backups = []
        
        try:
            for backup_dir in self.backup_dir.iterdir():
                if backup_dir.is_dir() and backup_dir.name.startswith("pre_reset_"):
                    metadata_path = backup_dir / "backup_metadata.json"
                    
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                                backups.append(metadata)
                        except json.JSONDecodeError:
                            self.logger.warning(f"Invalid backup metadata: {metadata_path}")
                    else:
                        # Create basic info from directory name
                        backups.append({
                            "backup_name": backup_dir.name,
                            "timestamp": "unknown",
                            "reset_type": "unknown",
                            "items": []
                        })
            
            return sorted(backups, key=lambda x: x.get("timestamp", ""), reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error listing backups: {e}")
            return []


class RecoveryManager:
    """Manages recovery operations after failed resets."""
    
    def __init__(self, backup_manager: BackupManager):
        self.backup_manager = backup_manager
        self.logger = logging.getLogger(__name__)
    
    def restore_from_backup(self, backup_name: str) -> bool:
        """Restore system from a specific backup."""
        try:
            backup_path = self.backup_manager.backup_dir / backup_name
            
            if not backup_path.exists():
                self.logger.error(f"Backup not found: {backup_path}")
                return False
            
            # Load backup metadata
            metadata_path = backup_path / "backup_metadata.json"
            if not metadata_path.exists():
                self.logger.error(f"Backup metadata not found: {metadata_path}")
                return False
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.logger.info(f"Restoring from backup: {backup_name}")
            
            # Restore each item
            for item_data in metadata.get("items", []):
                if not self._restore_item(item_data, backup_path):
                    self.logger.error(f"Failed to restore item: {item_data['name']}")
                    return False
            
            self.logger.info(f"Successfully restored from backup: {backup_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return False
    
    def _restore_item(self, item_data: Dict[str, Any], backup_path: Path) -> bool:
        """Restore a single item from backup."""
        try:
            source_path = backup_path / Path(item_data["source_path"]).name
            target_path = Path(item_data["source_path"])
            
            if not source_path.exists():
                self.logger.error(f"Backup item not found: {source_path}")
                return False
            
            # Remove existing target if it exists
            if target_path.exists():
                if target_path.is_dir():
                    shutil.rmtree(target_path)
                else:
                    target_path.unlink()
            
            # Create parent directories if needed
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Restore item
            if source_path.is_dir():
                shutil.copytree(source_path, target_path)
            else:
                shutil.copy2(source_path, target_path)
            
            self.logger.debug(f"Restored {source_path} to {target_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore item: {e}")
            return False


class FactoryReset:
    """Main factory reset system."""
    
    def __init__(self, config: ResetConfig = None):
        self.config = config or ResetConfig()
        self.logger = logging.getLogger(__name__)
        self.backup_manager = BackupManager(self.config.backup_location)
        self.recovery_manager = RecoveryManager(self.backup_manager)
        self.status = ResetStatus.IDLE
        self._lock = threading.RLock()
        
        # Base paths
        self.base_path = Path("/home/mmariani/projects/thesilentsteno")
        self.src_path = self.base_path / "src"
        self.config_path = self.base_path / "config"
        self.data_path = self.base_path / "data"
        self.logs_path = self.base_path / "logs"
    
    def perform_factory_reset(self, reset_type: ResetType = ResetType.FACTORY_RESET) -> ResetResult:
        """Perform factory reset operation."""
        start_time = time.time()
        result = ResetResult(
            success=False,
            reset_type=reset_type,
            duration_seconds=0.0
        )
        
        try:
            with self._lock:
                self.status = ResetStatus.BACKING_UP
                
                # Create backup if configured
                if self.config.backup_before_reset:
                    backup_location = self.backup_manager.create_backup(reset_type, self.config)
                    if backup_location:
                        result.backup_created = True
                        result.backup_location = backup_location
                        result.rollback_available = self.config.rollback_enabled
                    else:
                        result.errors.append("Backup creation failed")
                        if self.config.rollback_enabled:
                            self.status = ResetStatus.FAILED
                            return result
                
                # Perform reset
                self.status = ResetStatus.RESETTING
                
                if reset_type == ResetType.FACTORY_RESET:
                    reset_success = self._perform_full_reset(result)
                elif reset_type == ResetType.CONFIGURATION_RESET:
                    reset_success = self._perform_configuration_reset(result)
                elif reset_type == ResetType.DATA_RESET:
                    reset_success = self._perform_data_reset(result)
                elif reset_type == ResetType.SETTINGS_RESET:
                    reset_success = self._perform_settings_reset(result)
                else:
                    reset_success = self._perform_partial_reset(result)
                
                if not reset_success:
                    self.status = ResetStatus.FAILED
                    return result
                
                # Verification
                if self.config.verification_enabled:
                    if not self._verify_reset(reset_type, result):
                        result.errors.append("Reset verification failed")
                        self.status = ResetStatus.FAILED
                        return result
                
                # Success
                result.success = True
                self.status = ResetStatus.COMPLETED
                
                self.logger.info(f"Factory reset completed: {reset_type.value}")
                
        except Exception as e:
            result.errors.append(str(e))
            self.logger.error(f"Factory reset failed: {e}")
            self.status = ResetStatus.FAILED
        
        finally:
            result.duration_seconds = time.time() - start_time
        
        return result
    
    def _perform_full_reset(self, result: ResetResult) -> bool:
        """Perform complete factory reset."""
        try:
            components_to_reset = [
                ("application", self.src_path),
                ("configuration", self.config_path),
                ("data", self.data_path),
                ("logs", self.logs_path)
            ]
            
            for component_name, component_path in components_to_reset:
                if component_name in self.config.reset_components:
                    if self._reset_component(component_name, component_path):
                        result.components_reset.append(component_name)
                    else:
                        result.errors.append(f"Failed to reset {component_name}")
                        return False
                else:
                    result.preserved_items.append(component_name)
            
            # Reset database
            if self._reset_database():
                result.components_reset.append("database")
            else:
                result.errors.append("Failed to reset database")
                return False
            
            # Reset system services
            if self._reset_services():
                result.components_reset.append("services")
            else:
                result.errors.append("Failed to reset services")
                return False
            
            return True
            
        except Exception as e:
            result.errors.append(f"Full reset failed: {e}")
            return False
    
    def _perform_configuration_reset(self, result: ResetResult) -> bool:
        """Reset only configuration files."""
        try:
            if self._reset_component("configuration", self.config_path):
                result.components_reset.append("configuration")
                return True
            else:
                result.errors.append("Failed to reset configuration")
                return False
                
        except Exception as e:
            result.errors.append(f"Configuration reset failed: {e}")
            return False
    
    def _perform_data_reset(self, result: ResetResult) -> bool:
        """Reset only data files."""
        try:
            if self._reset_component("data", self.data_path):
                result.components_reset.append("data")
                
                # Reset database
                if self._reset_database():
                    result.components_reset.append("database")
                
                return True
            else:
                result.errors.append("Failed to reset data")
                return False
                
        except Exception as e:
            result.errors.append(f"Data reset failed: {e}")
            return False
    
    def _perform_settings_reset(self, result: ResetResult) -> bool:
        """Reset only settings files."""
        try:
            settings_files = [
                self.config_path / "app_config.json",
                self.config_path / "logging_config.json",
                self.config_path / "device_config.json"
            ]
            
            for settings_file in settings_files:
                if settings_file.exists():
                    settings_file.unlink()
            
            # Restore default settings
            if self._restore_default_settings():
                result.components_reset.append("settings")
                return True
            else:
                result.errors.append("Failed to restore default settings")
                return False
                
        except Exception as e:
            result.errors.append(f"Settings reset failed: {e}")
            return False
    
    def _perform_partial_reset(self, result: ResetResult) -> bool:
        """Perform partial reset based on configuration."""
        try:
            for component in self.config.reset_components:
                if component == "application":
                    if self._reset_component("application", self.src_path):
                        result.components_reset.append("application")
                elif component == "configuration":
                    if self._reset_component("configuration", self.config_path):
                        result.components_reset.append("configuration")
                elif component == "data":
                    if self._reset_component("data", self.data_path):
                        result.components_reset.append("data")
                elif component == "logs":
                    if self._reset_component("logs", self.logs_path):
                        result.components_reset.append("logs")
            
            return True
            
        except Exception as e:
            result.errors.append(f"Partial reset failed: {e}")
            return False
    
    def _reset_component(self, component_name: str, component_path: Path) -> bool:
        """Reset a specific component."""
        try:
            if component_path.exists():
                if component_path.is_dir():
                    shutil.rmtree(component_path)
                else:
                    component_path.unlink()
                
                self.logger.info(f"Reset component: {component_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reset component {component_name}: {e}")
            return False
    
    def _reset_database(self) -> bool:
        """Reset database to initial state."""
        try:
            db_path = self.data_path / "silentst.db"
            if db_path.exists():
                db_path.unlink()
                self.logger.info("Database reset")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reset database: {e}")
            return False
    
    def _reset_services(self) -> bool:
        """Reset system services to default state."""
        try:
            # This is a mock implementation
            # In real system, this would restart services
            self.logger.info("Services reset")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reset services: {e}")
            return False
    
    def _restore_default_settings(self) -> bool:
        """Restore default configuration settings."""
        try:
            # Create default configurations
            default_app_config = {
                "application": {
                    "name": "SilentSteno",
                    "version": "0.1.0",
                    "environment": "development"
                },
                "logging": {
                    "level": "INFO",
                    "enable_file": True
                }
            }
            
            # Write default configuration
            with open(self.config_path / "app_config.json", 'w') as f:
                json.dump(default_app_config, f, indent=2)
            
            self.logger.info("Default settings restored")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore default settings: {e}")
            return False
    
    def _verify_reset(self, reset_type: ResetType, result: ResetResult) -> bool:
        """Verify reset was successful."""
        try:
            # Basic verification - check that reset components are gone
            if "application" in result.components_reset:
                if self.src_path.exists():
                    return False
            
            if "configuration" in result.components_reset:
                config_files = list(self.config_path.glob("*.json"))
                if config_files and reset_type != ResetType.SETTINGS_RESET:
                    return False
            
            if "data" in result.components_reset:
                if self.data_path.exists():
                    return False
            
            self.logger.info("Reset verification passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Reset verification failed: {e}")
            return False
    
    def rollback_reset(self, backup_name: str = None) -> bool:
        """Rollback a factory reset using backup."""
        try:
            with self._lock:
                self.status = ResetStatus.RESTORING
                
                if backup_name is None:
                    # Use most recent backup
                    backups = self.backup_manager.list_backups()
                    if not backups:
                        self.logger.error("No backups available for rollback")
                        self.status = ResetStatus.FAILED
                        return False
                    backup_name = backups[0]["backup_name"]
                
                if self.recovery_manager.restore_from_backup(backup_name):
                    self.status = ResetStatus.COMPLETED
                    self.logger.info(f"Rollback completed from backup: {backup_name}")
                    return True
                else:
                    self.status = ResetStatus.FAILED
                    self.logger.error("Rollback failed")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            self.status = ResetStatus.FAILED
            return False
    
    def get_reset_status(self) -> Dict[str, Any]:
        """Get current reset status."""
        return {
            "status": self.status.value,
            "config": self.config.to_dict(),
            "available_backups": len(self.backup_manager.list_backups()),
            "backup_location": self.config.backup_location
        }


# Factory functions
def create_factory_reset(config: ResetConfig = None) -> FactoryReset:
    """Create a factory reset instance."""
    return FactoryReset(config)


def perform_factory_reset(reset_type: ResetType = ResetType.FACTORY_RESET, 
                         factory_reset: FactoryReset = None) -> ResetResult:
    """Perform factory reset operation."""
    if factory_reset is None:
        factory_reset = create_factory_reset()
    
    return factory_reset.perform_factory_reset(reset_type)


def create_backup(reset_type: ResetType = ResetType.FACTORY_RESET, 
                 factory_reset: FactoryReset = None) -> Optional[str]:
    """Create a backup before reset."""
    if factory_reset is None:
        factory_reset = create_factory_reset()
    
    return factory_reset.backup_manager.create_backup(reset_type, factory_reset.config)


def restore_backup(backup_name: str, factory_reset: FactoryReset = None) -> bool:
    """Restore from a specific backup."""
    if factory_reset is None:
        factory_reset = create_factory_reset()
    
    return factory_reset.recovery_manager.restore_from_backup(backup_name)