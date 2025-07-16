"""
Device Management System

Central device management orchestrator that coordinates all device management
features including health monitoring, storage cleanup, updates, diagnostics,
factory reset, and remote management.
"""

import logging
import threading
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from enum import Enum

# Import system management modules
from .health_monitor import HealthMonitor, SystemHealth, create_health_monitor
from .storage_cleanup import StorageCleanup, CleanupPolicy, create_storage_cleanup
from .update_manager import UpdateManager, UpdateConfig, create_update_manager
from .diagnostics import Diagnostics, DiagnosticReport, create_diagnostics
from .factory_reset import FactoryReset, ResetConfig, create_factory_reset
from .remote_manager import RemoteManager, create_remote_manager


class DeviceState(Enum):
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"
    UPDATING = "updating"
    OFFLINE = "offline"


class MaintenanceMode(Enum):
    NONE = "none"
    CLEANUP = "cleanup"
    DIAGNOSTICS = "diagnostics"
    UPDATE = "update"
    FACTORY_RESET = "factory_reset"


@dataclass
class DeviceConfig:
    """Device management configuration."""
    device_name: str = "Silent Steno Device"
    device_id: str = "silentst-001"
    location: str = "Unknown"
    timezone: str = "UTC"
    
    # Health monitoring
    health_check_interval: int = 60  # seconds
    health_monitoring_enabled: bool = True
    
    # Storage management
    storage_cleanup_enabled: bool = True
    storage_cleanup_threshold: float = 85.0
    
    # Update management
    auto_update_enabled: bool = True
    update_channel: str = "stable"
    
    # Remote management
    remote_management_enabled: bool = True
    remote_management_port: int = 8443
    
    # Diagnostics
    auto_diagnostics_enabled: bool = True
    diagnostics_schedule: str = "daily"
    
    # Factory reset
    factory_reset_enabled: bool = True
    backup_before_reset: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "device_name": self.device_name,
            "device_id": self.device_id,
            "location": self.location,
            "timezone": self.timezone,
            "health_check_interval": self.health_check_interval,
            "health_monitoring_enabled": self.health_monitoring_enabled,
            "storage_cleanup_enabled": self.storage_cleanup_enabled,
            "storage_cleanup_threshold": self.storage_cleanup_threshold,
            "auto_update_enabled": self.auto_update_enabled,
            "update_channel": self.update_channel,
            "remote_management_enabled": self.remote_management_enabled,
            "remote_management_port": self.remote_management_port,
            "auto_diagnostics_enabled": self.auto_diagnostics_enabled,
            "diagnostics_schedule": self.diagnostics_schedule,
            "factory_reset_enabled": self.factory_reset_enabled,
            "backup_before_reset": self.backup_before_reset
        }


@dataclass
class DeviceStatus:
    """Current device status."""
    state: DeviceState
    maintenance_mode: MaintenanceMode
    uptime_seconds: float
    last_update: datetime
    health_score: float
    storage_usage: float
    active_services: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "state": self.state.value,
            "maintenance_mode": self.maintenance_mode.value,
            "uptime_seconds": self.uptime_seconds,
            "uptime_hours": self.uptime_seconds / 3600,
            "last_update": self.last_update.isoformat(),
            "health_score": self.health_score,
            "storage_usage": self.storage_usage,
            "active_services": self.active_services,
            "warnings": self.warnings,
            "errors": self.errors
        }


class DeviceManager:
    """Central device management orchestrator."""
    
    def __init__(self, config: DeviceConfig = None, base_path: str = "/home/mmariani/projects/thesilentsteno"):
        self.config = config or DeviceConfig()
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)
        self.running = False
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        
        # Initialize subsystems
        self.health_monitor: Optional[HealthMonitor] = None
        self.storage_cleanup: Optional[StorageCleanup] = None
        self.update_manager: Optional[UpdateManager] = None
        self.diagnostics: Optional[Diagnostics] = None
        self.factory_reset: Optional[FactoryReset] = None
        self.remote_manager: Optional[RemoteManager] = None
        
        # Device state
        self.device_state = DeviceState.INITIALIZING
        self.maintenance_mode = MaintenanceMode.NONE
        self.start_time = datetime.now()
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {
            "state_change": [],
            "health_alert": [],
            "maintenance_start": [],
            "maintenance_end": [],
            "error": []
        }
        
        # Management thread
        self.management_thread: Optional[threading.Thread] = None
    
    def initialize(self):
        """Initialize device management subsystems."""
        try:
            with self._lock:
                self.logger.info("Initializing device management system")
                
                # Initialize health monitoring
                if self.config.health_monitoring_enabled:
                    self.health_monitor = create_health_monitor(self.config.health_check_interval)
                    self.health_monitor.add_alert_callback(self._handle_health_alert)
                
                # Initialize storage cleanup
                if self.config.storage_cleanup_enabled:
                    cleanup_policy = CleanupPolicy(
                        cleanup_threshold=self.config.storage_cleanup_threshold,
                        auto_cleanup_enabled=True
                    )
                    self.storage_cleanup = create_storage_cleanup(cleanup_policy, str(self.base_path))
                
                # Initialize update manager
                if self.config.auto_update_enabled:
                    update_config = UpdateConfig(
                        auto_update_enabled=True,
                        update_channel=self.config.update_channel
                    )
                    self.update_manager = create_update_manager(update_config)
                
                # Initialize diagnostics
                if self.config.auto_diagnostics_enabled:
                    self.diagnostics = create_diagnostics()
                
                # Initialize factory reset
                if self.config.factory_reset_enabled:
                    reset_config = ResetConfig(backup_before_reset=self.config.backup_before_reset)
                    self.factory_reset = create_factory_reset(reset_config)
                
                # Initialize remote management
                if self.config.remote_management_enabled:
                    remote_config = {
                        "port": self.config.remote_management_port,
                        "ssl_enabled": True
                    }
                    self.remote_manager = create_remote_manager(remote_config)
                
                self.device_state = DeviceState.HEALTHY
                self.logger.info("Device management system initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize device management: {e}")
            self.device_state = DeviceState.CRITICAL
            self._trigger_event("error", {"error": str(e), "context": "initialization"})
            raise
    
    def start(self):
        """Start device management system."""
        if self.running:
            return
        
        try:
            # Initialize if not already done
            if self.device_state == DeviceState.INITIALIZING:
                self.initialize()
            
            with self._lock:
                self.running = True
                
                # Start subsystems
                if self.health_monitor:
                    self.health_monitor.start()
                
                if self.storage_cleanup:
                    self.storage_cleanup.start_scheduled_cleanup()
                
                if self.remote_manager:
                    self.remote_manager.start()
                
                # Start management thread
                self.management_thread = threading.Thread(target=self._management_loop, daemon=True)
                self.management_thread.start()
                
                self.logger.info("Device management system started")
                
        except Exception as e:
            self.logger.error(f"Failed to start device management: {e}")
            self.device_state = DeviceState.CRITICAL
            self._trigger_event("error", {"error": str(e), "context": "startup"})
            raise
    
    def stop(self):
        """Stop device management system."""
        if not self.running:
            return
        
        try:
            with self._lock:
                self.running = False
                self._stop_event.set()
                
                # Stop subsystems
                if self.health_monitor:
                    self.health_monitor.stop()
                
                if self.storage_cleanup:
                    self.storage_cleanup.stop_scheduled_cleanup()
                
                if self.remote_manager:
                    self.remote_manager.stop()
                
                # Wait for management thread
                if self.management_thread:
                    self.management_thread.join(timeout=5.0)
                
                self.device_state = DeviceState.OFFLINE
                self.logger.info("Device management system stopped")
                
        except Exception as e:
            self.logger.error(f"Error stopping device management: {e}")
    
    def get_device_status(self) -> DeviceStatus:
        """Get current device status."""
        try:
            with self._lock:
                # Calculate uptime
                uptime_seconds = (datetime.now() - self.start_time).total_seconds()
                
                # Get health score
                health_score = 100.0
                if self.health_monitor and self.health_monitor.last_health:
                    health_score = self.health_monitor.last_health.overall_health_score
                
                # Get storage usage
                storage_usage = 0.0
                if self.storage_cleanup:
                    storage_info = self.storage_cleanup.get_storage_info()
                    storage_usage = storage_info.usage_percentage
                
                # Get active services
                active_services = []
                if self.health_monitor and self.health_monitor.running:
                    active_services.append("health_monitor")
                if self.storage_cleanup and self.storage_cleanup.scheduler.running:
                    active_services.append("storage_cleanup")
                if self.remote_manager and self.remote_manager.running:
                    active_services.append("remote_manager")
                
                # Get warnings and errors
                warnings = []
                errors = []
                
                if self.health_monitor and self.health_monitor.last_health:
                    health = self.health_monitor.last_health
                    if health.alerts:
                        warnings.extend(health.alerts)
                
                if self.storage_cleanup:
                    storage_info = self.storage_cleanup.get_storage_info()
                    if storage_info.usage_percentage > 90:
                        errors.append(f"Critical storage usage: {storage_info.usage_percentage:.1f}%")
                    elif storage_info.usage_percentage > 80:
                        warnings.append(f"High storage usage: {storage_info.usage_percentage:.1f}%")
                
                return DeviceStatus(
                    state=self.device_state,
                    maintenance_mode=self.maintenance_mode,
                    uptime_seconds=uptime_seconds,
                    last_update=datetime.now(),
                    health_score=health_score,
                    storage_usage=storage_usage,
                    active_services=active_services,
                    warnings=warnings,
                    errors=errors
                )
                
        except Exception as e:
            self.logger.error(f"Error getting device status: {e}")
            return DeviceStatus(
                state=DeviceState.CRITICAL,
                maintenance_mode=self.maintenance_mode,
                uptime_seconds=0,
                last_update=datetime.now(),
                health_score=0.0,
                storage_usage=0.0,
                errors=[str(e)]
            )
    
    def enter_maintenance_mode(self, mode: MaintenanceMode):
        """Enter maintenance mode."""
        try:
            with self._lock:
                self.maintenance_mode = mode
                self.device_state = DeviceState.MAINTENANCE
                self.logger.info(f"Entered maintenance mode: {mode.value}")
                self._trigger_event("maintenance_start", {"mode": mode.value})
                
        except Exception as e:
            self.logger.error(f"Error entering maintenance mode: {e}")
            self._trigger_event("error", {"error": str(e), "context": "maintenance_mode"})
    
    def exit_maintenance_mode(self):
        """Exit maintenance mode."""
        try:
            with self._lock:
                old_mode = self.maintenance_mode
                self.maintenance_mode = MaintenanceMode.NONE
                self.device_state = DeviceState.HEALTHY
                self.logger.info(f"Exited maintenance mode: {old_mode.value}")
                self._trigger_event("maintenance_end", {"mode": old_mode.value})
                
        except Exception as e:
            self.logger.error(f"Error exiting maintenance mode: {e}")
            self._trigger_event("error", {"error": str(e), "context": "maintenance_mode"})
    
    def run_diagnostics(self) -> DiagnosticReport:
        """Run system diagnostics."""
        try:
            self.enter_maintenance_mode(MaintenanceMode.DIAGNOSTICS)
            
            if not self.diagnostics:
                self.diagnostics = create_diagnostics()
            
            report = self.diagnostics.run_diagnostics()
            
            self.exit_maintenance_mode()
            return report
            
        except Exception as e:
            self.logger.error(f"Error running diagnostics: {e}")
            self.exit_maintenance_mode()
            raise
    
    def run_storage_cleanup(self, force: bool = False):
        """Run storage cleanup."""
        try:
            self.enter_maintenance_mode(MaintenanceMode.CLEANUP)
            
            if not self.storage_cleanup:
                cleanup_policy = CleanupPolicy(cleanup_threshold=self.config.storage_cleanup_threshold)
                self.storage_cleanup = create_storage_cleanup(cleanup_policy, str(self.base_path))
            
            result = self.storage_cleanup.run_cleanup(force)
            
            self.exit_maintenance_mode()
            return result
            
        except Exception as e:
            self.logger.error(f"Error running storage cleanup: {e}")
            self.exit_maintenance_mode()
            raise
    
    def check_for_updates(self):
        """Check for system updates."""
        try:
            if not self.update_manager:
                self.update_manager = create_update_manager()
            
            return self.update_manager.check_for_updates()
            
        except Exception as e:
            self.logger.error(f"Error checking for updates: {e}")
            raise
    
    def install_updates(self, update_info):
        """Install system updates."""
        try:
            self.enter_maintenance_mode(MaintenanceMode.UPDATE)
            
            if not self.update_manager:
                self.update_manager = create_update_manager()
            
            result = self.update_manager.install_update(update_info)
            
            self.exit_maintenance_mode()
            return result
            
        except Exception as e:
            self.logger.error(f"Error installing updates: {e}")
            self.exit_maintenance_mode()
            raise
    
    def perform_factory_reset(self, reset_type):
        """Perform factory reset."""
        try:
            self.enter_maintenance_mode(MaintenanceMode.FACTORY_RESET)
            
            if not self.factory_reset:
                reset_config = ResetConfig(backup_before_reset=self.config.backup_before_reset)
                self.factory_reset = create_factory_reset(reset_config)
            
            result = self.factory_reset.perform_factory_reset(reset_type)
            
            # After factory reset, device will restart, so we don't exit maintenance mode
            return result
            
        except Exception as e:
            self.logger.error(f"Error performing factory reset: {e}")
            self.exit_maintenance_mode()
            raise
    
    def add_event_callback(self, event_type: str, callback: Callable):
        """Add event callback."""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
    
    def _management_loop(self):
        """Main management loop."""
        while self.running and not self._stop_event.is_set():
            try:
                # Update device state based on health
                self._update_device_state()
                
                # Perform periodic maintenance
                self._perform_periodic_maintenance()
                
                # Wait before next iteration
                if self._stop_event.wait(60):  # Check every minute
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in management loop: {e}")
                time.sleep(10)
    
    def _update_device_state(self):
        """Update device state based on system health."""
        try:
            if self.maintenance_mode != MaintenanceMode.NONE:
                return
            
            if self.health_monitor and self.health_monitor.last_health:
                health = self.health_monitor.last_health
                
                if health.has_critical_issues:
                    new_state = DeviceState.CRITICAL
                elif health.has_errors:
                    new_state = DeviceState.WARNING
                else:
                    new_state = DeviceState.HEALTHY
                
                if new_state != self.device_state:
                    old_state = self.device_state
                    self.device_state = new_state
                    self.logger.info(f"Device state changed: {old_state.value} -> {new_state.value}")
                    self._trigger_event("state_change", {
                        "old_state": old_state.value,
                        "new_state": new_state.value
                    })
                    
        except Exception as e:
            self.logger.error(f"Error updating device state: {e}")
    
    def _perform_periodic_maintenance(self):
        """Perform periodic maintenance tasks."""
        try:
            # Check if cleanup is needed
            if (self.storage_cleanup and 
                self.storage_cleanup.is_critical_storage() and 
                self.maintenance_mode == MaintenanceMode.NONE):
                
                self.logger.info("Critical storage usage detected, running cleanup")
                self.run_storage_cleanup(force=True)
            
            # Check for updates if enabled
            if (self.config.auto_update_enabled and 
                self.update_manager and 
                self.maintenance_mode == MaintenanceMode.NONE):
                
                # Check for updates periodically (e.g., daily)
                # This is a simplified version - real implementation would be more sophisticated
                pass
                
        except Exception as e:
            self.logger.error(f"Error in periodic maintenance: {e}")
    
    def _handle_health_alert(self, health: SystemHealth):
        """Handle health alerts from health monitor."""
        try:
            self.logger.warning(f"Health alert: {health.overall_status.value}")
            self._trigger_event("health_alert", {
                "status": health.overall_status.value,
                "score": health.overall_health_score,
                "alerts": health.alerts
            })
            
        except Exception as e:
            self.logger.error(f"Error handling health alert: {e}")
    
    def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger event callbacks."""
        try:
            callbacks = self.event_callbacks.get(event_type, [])
            for callback in callbacks:
                try:
                    callback(event_type, data)
                except Exception as e:
                    self.logger.error(f"Error in event callback: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error triggering event: {e}")
    
    def get_management_summary(self) -> Dict[str, Any]:
        """Get comprehensive management summary."""
        try:
            status = self.get_device_status()
            
            summary = {
                "device_info": {
                    "name": self.config.device_name,
                    "id": self.config.device_id,
                    "location": self.config.location
                },
                "status": status.to_dict(),
                "subsystems": {
                    "health_monitor": {
                        "enabled": self.config.health_monitoring_enabled,
                        "running": self.health_monitor.running if self.health_monitor else False,
                        "last_check": self.health_monitor.last_health.timestamp.isoformat() if self.health_monitor and self.health_monitor.last_health else None
                    },
                    "storage_cleanup": {
                        "enabled": self.config.storage_cleanup_enabled,
                        "running": self.storage_cleanup.scheduler.running if self.storage_cleanup else False,
                        "threshold": self.config.storage_cleanup_threshold
                    },
                    "update_manager": {
                        "enabled": self.config.auto_update_enabled,
                        "channel": self.config.update_channel,
                        "status": self.update_manager.get_update_status() if self.update_manager else None
                    },
                    "remote_manager": {
                        "enabled": self.config.remote_management_enabled,
                        "running": self.remote_manager.running if self.remote_manager else False,
                        "port": self.config.remote_management_port
                    }
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting management summary: {e}")
            return {"error": str(e)}


# Factory functions
def create_device_manager(config: DeviceConfig = None, base_path: str = "/home/mmariani/projects/thesilentsteno") -> DeviceManager:
    """Create a device manager instance."""
    return DeviceManager(config, base_path)


def load_device_config(config_path: str) -> DeviceConfig:
    """Load device configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return DeviceConfig(**config_data)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error loading device config: {e}")
        return DeviceConfig()


def save_device_config(config: DeviceConfig, config_path: str):
    """Save device configuration to JSON file."""
    try:
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
            
    except Exception as e:
        logging.getLogger(__name__).error(f"Error saving device config: {e}")
        raise


def start_device_management(manager: DeviceManager = None, config: DeviceConfig = None) -> DeviceManager:
    """Start device management system."""
    if manager is None:
        manager = create_device_manager(config)
    
    manager.start()
    return manager


def stop_device_management(manager: DeviceManager):
    """Stop device management system."""
    manager.stop()