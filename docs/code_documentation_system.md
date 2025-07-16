# System Module Documentation

## Module Overview

The System module provides comprehensive device management capabilities for The Silent Steno, implementing a complete system administration framework with health monitoring, diagnostics, updates, storage cleanup, factory reset, and secure remote management. The module is designed around a central `DeviceManager` that orchestrates all system operations with intelligent automation and robust error handling.

## Dependencies

### External Dependencies
- `psutil` - System monitoring and process management
- `requests` - HTTP client for remote operations
- `cryptography` - Security and encryption
- `threading` - Thread management
- `sqlite3` - Database operations
- `json` - Configuration management
- `datetime` - Time operations
- `pathlib` - Path operations
- `subprocess` - System command execution
- `hashlib` - Hash functions
- `shutil` - File operations
- `os` - Operating system interface
- `time` - Timing operations
- `logging` - Logging system
- `dataclasses` - Data structures
- `enum` - Enumerations
- `typing` - Type hints

### Internal Dependencies
- `src.core.events` - Event system
- `src.core.logging` - Logging configuration
- `src.core.config` - Configuration management
- `src.data.database` - Database operations
- `src.audio.audio_system` - Audio system monitoring
- `src.bluetooth.bluetooth_manager` - Bluetooth monitoring

## File Documentation

### 1. `__init__.py`

**Purpose**: Module initialization and main device management orchestrator providing unified access to all system management components.

#### Classes

##### `DeviceManager`
Central orchestrator for all device management operations.

**Attributes:**
- `config: DeviceConfig` - Device configuration
- `health_monitor: HealthMonitor` - Health monitoring system
- `storage_cleanup: StorageCleanup` - Storage management
- `diagnostics: Diagnostics` - Diagnostic system
- `factory_reset: FactoryReset` - Factory reset capabilities
- `update_manager: UpdateManager` - Update management
- `remote_manager: RemoteManager` - Remote access management
- `current_state: DeviceState` - Current device state
- `maintenance_mode: MaintenanceMode` - Maintenance mode state
- `callbacks: Dict[str, List[Callable]]` - Event callbacks

**Methods:**
- `__init__(config: DeviceConfig = None)` - Initialize device manager
- `start_device_management()` - Start all management subsystems
- `stop_device_management()` - Stop all management subsystems
- `get_device_status()` - Get comprehensive device status
- `enter_maintenance_mode(mode: MaintenanceMode)` - Enter maintenance mode
- `exit_maintenance_mode()` - Exit maintenance mode
- `add_callback(event_type: str, callback: Callable)` - Add event callback
- `trigger_health_check()` - Trigger comprehensive health check
- `trigger_storage_cleanup()` - Trigger storage cleanup
- `trigger_diagnostics()` - Trigger diagnostic tests
- `get_system_metrics()` - Get real-time system metrics

##### `DeviceConfig`
Device configuration parameters.

**Attributes:**
- `health_monitoring_enabled: bool` - Enable health monitoring
- `storage_cleanup_enabled: bool` - Enable storage cleanup
- `diagnostics_enabled: bool` - Enable diagnostic tests
- `remote_access_enabled: bool` - Enable remote access
- `auto_updates_enabled: bool` - Enable automatic updates
- `maintenance_window: str` - Maintenance window schedule
- `alert_thresholds: Dict[str, float]` - Alert thresholds
- `log_level: str` - Logging level

##### `DeviceStatus`
Current device status information.

**Attributes:**
- `state: DeviceState` - Current device state
- `maintenance_mode: MaintenanceMode` - Maintenance mode
- `health_score: float` - Overall health score (0-1)
- `storage_usage: float` - Storage usage percentage
- `system_load: float` - System load average
- `uptime: float` - System uptime in seconds
- `last_health_check: datetime` - Last health check timestamp
- `active_alerts: List[str]` - Active system alerts

#### Enums

##### `DeviceState`
Device operational states.
- `STARTING` - Device starting up
- `RUNNING` - Normal operation
- `MAINTENANCE` - Maintenance mode
- `UPDATING` - Software update in progress
- `DIAGNOSTICS` - Diagnostic tests running
- `EMERGENCY` - Emergency state
- `SHUTTING_DOWN` - Shutdown in progress
- `ERROR` - Error state

##### `MaintenanceMode`
Maintenance mode types.
- `NONE` - No maintenance
- `SCHEDULED` - Scheduled maintenance
- `EMERGENCY` - Emergency maintenance
- `UPDATE` - Update maintenance
- `DIAGNOSTIC` - Diagnostic maintenance
- `FACTORY_RESET` - Factory reset maintenance

#### Default Configurations

##### `DEFAULT_DEVICE_CONFIG`
Default device configuration.
```python
{
    "health_monitoring_enabled": True,
    "storage_cleanup_enabled": True,
    "diagnostics_enabled": True,
    "remote_access_enabled": False,
    "auto_updates_enabled": True,
    "maintenance_window": "02:00-04:00",
    "alert_thresholds": {
        "cpu_usage": 80.0,
        "memory_usage": 85.0,
        "storage_usage": 90.0,
        "temperature": 70.0
    },
    "log_level": "INFO"
}
```

**Usage Example:**
```python
from src.system import DeviceManager, DeviceConfig

# Create device configuration
config = DeviceConfig(
    health_monitoring_enabled=True,
    storage_cleanup_enabled=True,
    diagnostics_enabled=True,
    remote_access_enabled=True,
    auto_updates_enabled=True,
    maintenance_window="02:00-04:00",
    alert_thresholds={
        "cpu_usage": 80.0,
        "memory_usage": 85.0,
        "storage_usage": 90.0,
        "temperature": 70.0
    }
)

# Create device manager
device_manager = DeviceManager(config)

# Set up callbacks
def on_health_alert(alert_info):
    print(f"Health alert: {alert_info['message']}")

def on_maintenance_mode(mode_info):
    print(f"Maintenance mode: {mode_info['mode']}")

device_manager.add_callback("health_alert", on_health_alert)
device_manager.add_callback("maintenance_mode", on_maintenance_mode)

# Start device management
device_manager.start_device_management()

# Get device status
status = device_manager.get_device_status()
print(f"Device state: {status.state}")
print(f"Health score: {status.health_score:.2f}")
print(f"Storage usage: {status.storage_usage:.1f}%")

# Trigger health check
health_results = device_manager.trigger_health_check()
print(f"Health check results: {health_results}")

# Enter maintenance mode
device_manager.enter_maintenance_mode(MaintenanceMode.SCHEDULED)
```

### 2. `device_manager.py`

**Purpose**: Central device management orchestrator coordinating all system management subsystems.

#### Classes

##### `DeviceManager`
Main device management coordinator.

**Methods:**
- `__init__(config: DeviceConfig)` - Initialize device manager
- `start_device_management()` - Start all subsystems
- `stop_device_management()` - Stop all subsystems
- `get_device_status()` - Get comprehensive status
- `enter_maintenance_mode(mode: MaintenanceMode, duration: int = None)` - Enter maintenance
- `exit_maintenance_mode()` - Exit maintenance mode
- `schedule_maintenance(mode: MaintenanceMode, scheduled_time: datetime)` - Schedule maintenance
- `get_maintenance_schedule()` - Get maintenance schedule
- `add_callback(event_type: str, callback: Callable)` - Add event callback
- `remove_callback(event_type: str, callback: Callable)` - Remove event callback
- `trigger_health_check()` - Trigger health check
- `trigger_storage_cleanup()` - Trigger storage cleanup
- `trigger_diagnostics()` - Trigger diagnostics
- `get_system_metrics()` - Get system metrics
- `get_component_status(component: str)` - Get component status

**Usage Example:**
```python
from src.system.device_manager import DeviceManager, DeviceConfig, MaintenanceMode

# Create and configure device manager
config = DeviceConfig(
    health_monitoring_enabled=True,
    storage_cleanup_enabled=True,
    maintenance_window="02:00-04:00"
)

device_manager = DeviceManager(config)

# Set up comprehensive callbacks
def on_health_alert(alert):
    print(f"Health Alert: {alert['severity']} - {alert['message']}")
    if alert['severity'] == 'critical':
        device_manager.enter_maintenance_mode(MaintenanceMode.EMERGENCY)

def on_storage_warning(warning):
    print(f"Storage Warning: {warning['usage']:.1f}% used")
    device_manager.trigger_storage_cleanup()

def on_maintenance_mode(mode_info):
    print(f"Maintenance Mode: {mode_info['mode'].value}")

device_manager.add_callback("health_alert", on_health_alert)
device_manager.add_callback("storage_warning", on_storage_warning)
device_manager.add_callback("maintenance_mode", on_maintenance_mode)

# Start device management
device_manager.start_device_management()

# Monitor device status
import time
while True:
    status = device_manager.get_device_status()
    print(f"Status: {status.state.value}, Health: {status.health_score:.2f}")
    time.sleep(60)
```

### 3. `health_monitor.py`

**Purpose**: Real-time system health monitoring with predictive maintenance capabilities.

#### Classes

##### `HealthMonitor`
Main health monitoring system.

**Methods:**
- `__init__(config: HealthConfig)` - Initialize health monitor
- `start_monitoring()` - Start health monitoring
- `stop_monitoring()` - Stop health monitoring
- `get_health_status()` - Get current health status
- `get_component_health(component: str)` - Get component health
- `run_health_check()` - Run comprehensive health check
- `get_health_history(hours: int = 24)` - Get health history
- `add_health_checker(checker: HealthChecker)` - Add custom health checker
- `set_alert_threshold(component: str, threshold: float)` - Set alert threshold

##### `HealthChecker`
Base class for component health checkers.

**Methods:**
- `check_health()` - Check component health
- `get_metrics()` - Get component metrics
- `get_health_score()` - Get health score (0-1)

##### `SystemMetrics`
System metrics collection.

**Attributes:**
- `cpu_usage: float` - CPU usage percentage
- `memory_usage: float` - Memory usage percentage
- `disk_usage: float` - Disk usage percentage
- `network_activity: float` - Network activity
- `temperature: float` - System temperature
- `uptime: float` - System uptime
- `load_average: float` - System load average

**Usage Example:**
```python
from src.system.health_monitor import HealthMonitor, HealthConfig

# Create health monitor
config = HealthConfig(
    monitoring_interval=30,
    alert_thresholds={
        "cpu_usage": 80.0,
        "memory_usage": 85.0,
        "disk_usage": 90.0,
        "temperature": 70.0
    },
    enable_predictive_alerts=True
)

health_monitor = HealthMonitor(config)

# Set up health alerts
def on_health_alert(alert):
    print(f"Health Alert: {alert['component']} - {alert['message']}")
    print(f"Current value: {alert['current_value']}")
    print(f"Threshold: {alert['threshold']}")

health_monitor.add_callback("health_alert", on_health_alert)

# Start monitoring
health_monitor.start_monitoring()

# Get health status
status = health_monitor.get_health_status()
print(f"Overall health: {status.overall_health:.2f}")
print(f"System metrics: CPU {status.system_metrics.cpu_usage:.1f}%")
print(f"Memory: {status.system_metrics.memory_usage:.1f}%")

# Run comprehensive health check
health_results = health_monitor.run_health_check()
for component, result in health_results.items():
    print(f"{component}: {result.health_score:.2f} ({result.status})")
```

### 4. `storage_cleanup.py`

**Purpose**: Automated storage management with intelligent cleanup policies.

#### Classes

##### `StorageCleanup`
Main storage cleanup coordinator.

**Methods:**
- `__init__(config: CleanupConfig)` - Initialize storage cleanup
- `start_cleanup_monitoring()` - Start monitoring
- `stop_cleanup_monitoring()` - Stop monitoring
- `run_cleanup()` - Run cleanup process
- `get_cleanup_status()` - Get cleanup status
- `emergency_cleanup()` - Emergency cleanup
- `schedule_cleanup(cleanup_time: datetime)` - Schedule cleanup
- `get_storage_usage()` - Get storage usage
- `optimize_storage()` - Optimize storage

##### `CleanupPolicy`
Cleanup policy configuration.

**Attributes:**
- `category: str` - File category
- `retention_days: int` - Retention period
- `size_threshold: int` - Size threshold
- `enabled: bool` - Policy enabled
- `emergency_action: str` - Emergency action

**Usage Example:**
```python
from src.system.storage_cleanup import StorageCleanup, CleanupConfig, CleanupPolicy

# Create cleanup policies
policies = [
    CleanupPolicy(
        category="temp_files",
        retention_days=1,
        size_threshold=0,
        enabled=True,
        emergency_action="delete_all"
    ),
    CleanupPolicy(
        category="logs",
        retention_days=30,
        size_threshold=100*1024*1024,  # 100MB
        enabled=True,
        emergency_action="compress"
    ),
    CleanupPolicy(
        category="old_recordings",
        retention_days=90,
        size_threshold=1024*1024*1024,  # 1GB
        enabled=True,
        emergency_action="archive"
    )
]

# Create cleanup system
config = CleanupConfig(
    cleanup_policies=policies,
    monitoring_interval=3600,  # 1 hour
    emergency_threshold=95.0,  # 95% usage
    enable_compression=True
)

cleanup_system = StorageCleanup(config)

# Set up cleanup callbacks
def on_cleanup_completed(results):
    print(f"Cleanup completed: {results['files_removed']} files removed")
    print(f"Space freed: {results['space_freed']} bytes")

def on_emergency_cleanup(info):
    print(f"Emergency cleanup triggered: {info['usage']:.1f}% usage")

cleanup_system.add_callback("cleanup_completed", on_cleanup_completed)
cleanup_system.add_callback("emergency_cleanup", on_emergency_cleanup)

# Start cleanup monitoring
cleanup_system.start_cleanup_monitoring()

# Get storage usage
usage = cleanup_system.get_storage_usage()
print(f"Storage usage: {usage.usage_percent:.1f}%")
print(f"Available space: {usage.available_gb:.1f} GB")

# Run manual cleanup
cleanup_results = cleanup_system.run_cleanup()
print(f"Cleanup results: {cleanup_results}")
```

### 5. `diagnostics.py`

**Purpose**: Comprehensive system diagnostics and troubleshooting capabilities.

#### Classes

##### `Diagnostics`
Main diagnostic system coordinator.

**Methods:**
- `__init__(config: DiagnosticConfig)` - Initialize diagnostics
- `run_diagnostics()` - Run all diagnostic tests
- `run_test(test_name: str)` - Run specific test
- `get_diagnostic_history()` - Get diagnostic history
- `generate_diagnostic_report()` - Generate comprehensive report
- `get_system_info()` - Get system information
- `analyze_performance()` - Analyze system performance
- `check_dependencies()` - Check system dependencies

##### `DiagnosticTest`
Base class for diagnostic tests.

**Methods:**
- `run_test()` - Execute test
- `get_test_info()` - Get test information
- `cleanup()` - Test cleanup

**Usage Example:**
```python
from src.system.diagnostics import Diagnostics, DiagnosticConfig

# Create diagnostics system
config = DiagnosticConfig(
    enable_performance_tests=True,
    enable_hardware_tests=True,
    enable_network_tests=True,
    test_timeout=300,
    detailed_logging=True
)

diagnostics = Diagnostics(config)

# Run comprehensive diagnostics
print("Running comprehensive diagnostics...")
results = diagnostics.run_diagnostics()

# Display results
print("\nDiagnostic Results:")
print("=" * 50)
for test_name, result in results.items():
    status = "PASS" if result.passed else "FAIL"
    print(f"{test_name}: {status}")
    if result.details:
        print(f"  Details: {result.details}")
    if result.recommendations:
        print(f"  Recommendations: {result.recommendations}")

# Generate detailed report
report = diagnostics.generate_diagnostic_report()
print(f"\nDetailed report saved to: {report}")

# Analyze performance
performance = diagnostics.analyze_performance()
print(f"\nPerformance Analysis:")
print(f"CPU Performance: {performance.cpu_score:.2f}")
print(f"Memory Performance: {performance.memory_score:.2f}")
print(f"Disk Performance: {performance.disk_score:.2f}")
```

### 6. `factory_reset.py`

**Purpose**: Secure factory reset capabilities with backup and recovery.

#### Classes

##### `FactoryReset`
Main factory reset coordinator.

**Methods:**
- `__init__(config: ResetConfig)` - Initialize factory reset
- `prepare_reset()` - Prepare for factory reset
- `execute_reset(reset_type: ResetType)` - Execute factory reset
- `create_backup()` - Create pre-reset backup
- `restore_from_backup(backup_path: str)` - Restore from backup
- `validate_reset()` - Validate reset completion
- `get_reset_options()` - Get available reset options

##### `ResetType`
Types of factory reset.
- `COMPLETE` - Complete factory reset
- `CONFIGURATION_ONLY` - Reset configuration only
- `DATA_ONLY` - Reset data only
- `SETTINGS_ONLY` - Reset settings only
- `SELECTIVE` - Selective component reset

**Usage Example:**
```python
from src.system.factory_reset import FactoryReset, ResetConfig, ResetType

# Create factory reset system
config = ResetConfig(
    backup_enabled=True,
    backup_location="/backup",
    verification_enabled=True,
    preserve_logs=True
)

factory_reset = FactoryReset(config)

# Create backup before reset
print("Creating backup...")
backup_path = factory_reset.create_backup()
print(f"Backup created: {backup_path}")

# Execute factory reset
print("Executing factory reset...")
reset_result = factory_reset.execute_reset(ResetType.COMPLETE)

if reset_result.success:
    print("Factory reset completed successfully")
else:
    print(f"Factory reset failed: {reset_result.error}")
    # Restore from backup
    factory_reset.restore_from_backup(backup_path)
```

### 7. `update_manager.py`

**Purpose**: Software update management with rollback capabilities.

#### Classes

##### `UpdateManager`
Main update management system.

**Methods:**
- `__init__(config: UpdateConfig)` - Initialize update manager
- `check_for_updates()` - Check for available updates
- `install_update(update_info: UpdateInfo)` - Install update
- `rollback_update()` - Rollback to previous version
- `get_update_history()` - Get update history
- `schedule_update(update_info: UpdateInfo, scheduled_time: datetime)` - Schedule update
- `validate_update(update_info: UpdateInfo)` - Validate update

**Usage Example:**
```python
from src.system.update_manager import UpdateManager, UpdateConfig

# Create update manager
config = UpdateConfig(
    auto_check_enabled=True,
    auto_install_enabled=False,
    backup_before_update=True,
    rollback_enabled=True,
    update_window="02:00-04:00"
)

update_manager = UpdateManager(config)

# Check for updates
print("Checking for updates...")
updates = update_manager.check_for_updates()

if updates:
    print(f"Found {len(updates)} updates available")
    for update in updates:
        print(f"  {update.name} v{update.version}")
        
        # Install update
        print(f"Installing {update.name}...")
        install_result = update_manager.install_update(update)
        
        if install_result.success:
            print("Update installed successfully")
        else:
            print(f"Update failed: {install_result.error}")
            # Rollback if needed
            update_manager.rollback_update()
```

### 8. `remote_manager.py`

**Purpose**: Secure remote device management and control capabilities.

#### Classes

##### `RemoteManager`
Main remote access coordinator.

**Methods:**
- `__init__(config: RemoteConfig)` - Initialize remote manager
- `start_remote_server()` - Start remote access server
- `stop_remote_server()` - Stop remote access server
- `authenticate_user(credentials: RemoteCredentials)` - Authenticate user
- `execute_remote_command(command: str, user: str)` - Execute remote command
- `get_remote_sessions()` - Get active remote sessions
- `revoke_session(session_id: str)` - Revoke remote session

**Usage Example:**
```python
from src.system.remote_manager import RemoteManager, RemoteConfig

# Create remote manager
config = RemoteConfig(
    enable_https=True,
    port=8443,
    auth_required=True,
    session_timeout=3600,
    max_concurrent_sessions=5
)

remote_manager = RemoteManager(config)

# Start remote server
remote_manager.start_remote_server()

# The server will handle remote requests automatically
# Commands can be executed remotely through the API
```

## Module Integration

The System module integrates with other Silent Steno components:

1. **Core Module**: Uses events, logging, and configuration
2. **Data Module**: Monitors database health and integrity
3. **Audio Module**: Monitors audio system health
4. **Bluetooth Module**: Monitors Bluetooth connectivity
5. **UI Module**: Provides system status to user interface

## Common Usage Patterns

### Complete System Management Setup
```python
# Initialize complete system management
from src.system import DeviceManager, DeviceConfig

# Create comprehensive configuration
config = DeviceConfig(
    health_monitoring_enabled=True,
    storage_cleanup_enabled=True,
    diagnostics_enabled=True,
    remote_access_enabled=True,
    auto_updates_enabled=True,
    maintenance_window="02:00-04:00",
    alert_thresholds={
        "cpu_usage": 80.0,
        "memory_usage": 85.0,
        "storage_usage": 90.0,
        "temperature": 70.0
    }
)

# Create device manager
device_manager = DeviceManager(config)

# Set up comprehensive event handling
def handle_health_alert(alert):
    print(f"Health Alert: {alert['severity']} - {alert['message']}")
    if alert['severity'] == 'critical':
        device_manager.enter_maintenance_mode(MaintenanceMode.EMERGENCY)

def handle_storage_warning(warning):
    print(f"Storage at {warning['usage']:.1f}% - triggering cleanup")
    device_manager.trigger_storage_cleanup()

def handle_maintenance_mode(mode_info):
    print(f"Entering maintenance mode: {mode_info['mode'].value}")

device_manager.add_callback("health_alert", handle_health_alert)
device_manager.add_callback("storage_warning", handle_storage_warning)
device_manager.add_callback("maintenance_mode", handle_maintenance_mode)

# Start system management
device_manager.start_device_management()

# Monitor system continuously
import time
while True:
    status = device_manager.get_device_status()
    print(f"System Status: {status.state.value}")
    print(f"Health Score: {status.health_score:.2f}")
    print(f"Storage Usage: {status.storage_usage:.1f}%")
    
    if status.active_alerts:
        print(f"Active Alerts: {', '.join(status.active_alerts)}")
    
    time.sleep(300)  # Check every 5 minutes
```

This comprehensive System module provides a robust, secure, and intelligent device management framework essential for maintaining The Silent Steno's reliability and performance in production environments.