"""
Core Application Integration Layer

This package provides the central application integration layer for The Silent Steno,
including the main application controller, event system, configuration management,
logging, monitoring, error handling, and component registry.
"""

from .application import (
    ApplicationController,
    AppState,
    ApplicationConfig,
    ComponentManager,
    start_app,
    stop_app,
    restart_component,
    get_app_status
)

from .events import (
    EventBus,
    Event,
    EventHandler,
    EventSubscription,
    publish_event,
    subscribe_to_event,
    unsubscribe,
    create_event_bus
)

from .config import (
    ConfigManager,
    ConfigSchema,
    ConfigValidator,
    ConfigWatcher,
    load_config,
    save_config,
    validate_config,
    watch_config_changes
)

from .logging import (
    LogManager,
    LogConfig,
    StructuredLogger,
    LogRotator,
    setup_logging,
    get_logger,
    configure_log_rotation,
    add_log_handler
)

from .monitoring import (
    PerformanceMonitor,
    HealthChecker,
    MetricsCollector,
    AlertManager,
    start_monitoring,
    collect_metrics,
    check_system_health,
    send_alert
)

from .errors import (
    ErrorHandler,
    RecoveryManager,
    ErrorReporter,
    FallbackManager,
    handle_error,
    attempt_recovery,
    report_error,
    activate_fallback
)

from .registry import (
    ComponentRegistry,
    register_component,
    get_component,
    inject_dependencies,
    create_component
)

# Main application class for unified access
class SilentStenoApp:
    """
    Main application class providing unified access to all subsystems.
    
    This class serves as the primary entry point for the Silent Steno application,
    coordinating all subsystems through the application controller.
    """
    
    def __init__(self, config_path: str = None):
        self.controller = None
        self.config_path = config_path or "config/app_config.json"
        self._initialized = False
    
    def initialize(self):
        """Initialize the application with all subsystems."""
        if self._initialized:
            return
        
        # Initialize configuration first
        self.config = load_config(self.config_path)
        
        # Set up logging
        setup_logging(self.config.get('logging', {}))
        
        # Create application controller
        self.controller = ApplicationController(self.config)
        
        # Initialize all subsystems
        self.controller.initialize()
        
        self._initialized = True
    
    def start(self):
        """Start the application and all subsystems."""
        if not self._initialized:
            self.initialize()
        
        return start_app(self.controller)
    
    def stop(self):
        """Stop the application and all subsystems."""
        if self.controller:
            return stop_app(self.controller)
    
    def restart(self):
        """Restart the application."""
        self.stop()
        return self.start()


# Factory functions for main application
def create_application(config_path: str = None) -> SilentStenoApp:
    """
    Create a new Silent Steno application instance.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        SilentStenoApp: Configured application instance
    """
    return SilentStenoApp(config_path)


def start_application(config_path: str = None) -> SilentStenoApp:
    """
    Create and start a Silent Steno application.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        SilentStenoApp: Running application instance
    """
    app = create_application(config_path)
    app.start()
    return app


def shutdown_application(app: SilentStenoApp):
    """
    Gracefully shutdown a Silent Steno application.
    
    Args:
        app: Application instance to shutdown
    """
    if app:
        app.stop()


__all__ = [
    # Main application class
    'SilentStenoApp',
    
    # Factory functions
    'create_application',
    'start_application', 
    'shutdown_application',
    
    # Application controller
    'ApplicationController',
    'AppState',
    'ApplicationConfig',
    'ComponentManager',
    'start_app',
    'stop_app',
    'restart_component',
    'get_app_status',
    
    # Event system
    'EventBus',
    'Event',
    'EventHandler',
    'EventSubscription',
    'publish_event',
    'subscribe_to_event',
    'unsubscribe',
    'create_event_bus',
    
    # Configuration management
    'ConfigManager',
    'ConfigSchema',
    'ConfigValidator',
    'ConfigWatcher',
    'load_config',
    'save_config',
    'validate_config',
    'watch_config_changes',
    
    # Logging system
    'LogManager',
    'LogConfig',
    'StructuredLogger',
    'LogRotator',
    'setup_logging',
    'get_logger',
    'configure_log_rotation',
    'add_log_handler',
    
    # Performance monitoring
    'PerformanceMonitor',
    'HealthChecker',
    'MetricsCollector',
    'AlertManager',
    'start_monitoring',
    'collect_metrics',
    'check_system_health',
    'send_alert',
    
    # Error handling
    'ErrorHandler',
    'RecoveryManager',
    'ErrorReporter',
    'FallbackManager',
    'handle_error',
    'attempt_recovery',
    'report_error',
    'activate_fallback',
    
    # Component registry
    'ComponentRegistry',
    'register_component',
    'get_component',
    'inject_dependencies',
    'create_component'
]