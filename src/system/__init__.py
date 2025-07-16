"""
System Management Package

This package provides comprehensive device management capabilities for The Silent Steno,
including storage cleanup, software updates, health monitoring, diagnostics, factory reset,
and optional remote management functionality.
"""

from .device_manager import (
    DeviceManager,
    DeviceConfig,
    create_device_manager,
    start_device_management,
    stop_device_management,
    load_device_config,
    save_device_config
)

from .storage_cleanup import (
    StorageCleanup,
    CleanupPolicy,
    CleanupScheduler,
    SpaceOptimizer,
    create_storage_cleanup,
    run_cleanup,
    schedule_cleanup,
    optimize_space
)

from .update_manager import (
    UpdateManager,
    UpdateConfig,
    UpdateValidator,
    RollbackManager,
    create_update_manager,
    check_updates,
    install_update,
    rollback_update
)

from .health_monitor import (
    HealthMonitor,
    HealthChecker,
    SystemMetrics,
    ComponentHealth,
    create_health_monitor,
    check_system_health,
    monitor_components,
    report_health
)

from .diagnostics import (
    Diagnostics,
    DiagnosticTest,
    PerformanceAnalyzer,
    LogAnalyzer,
    create_diagnostics,
    run_diagnostics,
    analyze_performance,
    analyze_logs
)

from .factory_reset import (
    FactoryReset,
    ResetConfig,
    BackupManager,
    RecoveryManager,
    create_factory_reset,
    perform_factory_reset,
    create_backup,
    restore_backup
)

from .remote_manager import (
    RemoteManager,
    RemoteCredentials,
    RemoteSession,
    RemoteCommand,
    create_remote_manager,
    start_remote_management,
    stop_remote_management,
    execute_remote_command
)

# Main unified interface
def create_system_manager(config_path: str = None) -> DeviceManager:
    """
    Create a unified system management interface.
    
    Args:
        config_path: Path to device configuration file
        
    Returns:
        DeviceManager: Configured device management instance
    """
    return create_device_manager(config_path)


__all__ = [
    # Main interface
    'DeviceManager',
    'DeviceConfig',
    'create_system_manager',
    'create_device_manager',
    'start_device_management',
    'stop_device_management',
    'load_device_config',
    'save_device_config',
    
    # Storage management
    'StorageCleanup',
    'CleanupPolicy',
    'create_storage_cleanup',
    'run_cleanup',
    'schedule_cleanup',
    'optimize_space',
    
    # Update management
    'UpdateManager',
    'UpdateConfig',
    'create_update_manager',
    'check_updates',
    'install_update',
    'rollback_update',
    
    # Health monitoring
    'HealthMonitor',
    'HealthChecker',
    'SystemMetrics',
    'ComponentHealth',
    'create_health_monitor',
    'check_system_health',
    'monitor_components',
    'report_health',
    
    # Diagnostics
    'Diagnostics',
    'DiagnosticTest',
    'PerformanceAnalyzer',
    'LogAnalyzer',
    'create_diagnostics',
    'run_diagnostics',
    'analyze_performance',
    'analyze_logs',
    
    # Factory reset
    'FactoryReset',
    'ResetConfig',
    'BackupManager',
    'RecoveryManager',
    'create_factory_reset',
    'perform_factory_reset',
    'create_backup',
    'restore_backup',
    
    # Remote management
    'RemoteManager',
    'RemoteCredentials',
    'RemoteSession',
    'RemoteCommand',
    'create_remote_manager',
    'start_remote_management',
    'stop_remote_management',
    'execute_remote_command'
]