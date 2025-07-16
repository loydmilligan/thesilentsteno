# Core Module Documentation

## Module Overview

The Core module provides the foundational infrastructure for The Silent Steno application. It implements essential systems for application lifecycle management, configuration, event handling, error management, logging, monitoring, and component registry with dependency injection.

## Dependencies

### External Dependencies
- `asyncio` - For asynchronous operations
- `threading` - Thread-safe implementations
- `logging` - Python's logging framework
- `json` - Configuration file handling
- `pathlib` - File system operations
- `dataclasses` - Data structure definitions
- `enum` - Enumeration types
- `typing` - Type hints
- `concurrent.futures` - Thread pool execution
- `pydantic` (optional) - Configuration validation
- `watchdog` (optional) - File system monitoring
- `psutil` (optional) - System metrics collection

### Internal Dependencies
- `src.system.device_manager` - Device management integration

## File Documentation

### 1. `__init__.py`

**Purpose**: Module initialization and public API exposure. Provides the main `SilentStenoApp` class and convenience functions.

#### Classes

##### `SilentStenoApp`
Main application class providing unified access to all subsystems.

**Attributes:**
- `controller: ApplicationController` - Main application controller
- `config_path: str` - Path to configuration file
- `_initialized: bool` - Initialization state flag

**Methods:**
- `__init__(config_path: str = None)` - Initialize with optional config path
- `initialize()` - Initialize the application with all subsystems
- `start()` - Start the application and all subsystems
- `stop()` - Stop the application and all subsystems
- `restart()` - Restart the application

#### Functions

##### `create_application(config_path: str = None) -> SilentStenoApp`
Factory function to create a new Silent Steno application instance.

##### `start_application(config_path: str = None) -> SilentStenoApp`
Create and start a Silent Steno application in one call.

##### `shutdown_application(app: SilentStenoApp)`
Gracefully shutdown a Silent Steno application.

**Usage Example:**
```python
# Create and start application
app = start_application("config/app_config.json")

# Or create separately
app = create_application()
app.initialize()
app.start()

# Shutdown when done
shutdown_application(app)
```

### 2. `application.py`

**Purpose**: Central application controller managing component lifecycle, inter-component communication, and application state coordination.

#### Enums

##### `AppState`
Application state enumeration.
- `UNINITIALIZED` - Not yet initialized
- `INITIALIZING` - Currently initializing
- `READY` - Ready to start
- `STARTING` - Currently starting
- `RUNNING` - Fully operational
- `STOPPING` - Currently stopping
- `STOPPED` - Fully stopped
- `ERROR` - Error state
- `MAINTENANCE` - Maintenance mode

##### `ComponentState`
Component state enumeration.
- `UNREGISTERED` - Not registered
- `REGISTERED` - Registered but not initialized
- `INITIALIZING` - Currently initializing
- `READY` - Ready to start
- `STARTING` - Currently starting
- `RUNNING` - Fully operational
- `STOPPING` - Currently stopping
- `STOPPED` - Fully stopped
- `ERROR` - Error state
- `FAILED` - Failed state

#### Classes

##### `ComponentInfo`
Information about a registered component.

**Attributes:**
- `name: str` - Component name
- `instance: Any` - Component instance
- `state: ComponentState` - Current state
- `dependencies: List[str]` - Required dependencies
- `startup_priority: int` - Startup order (lower = earlier)
- `shutdown_priority: int` - Shutdown order (lower = later)
- `health_check: Optional[Callable]` - Health check function
- `last_health_check: Optional[float]` - Last health check timestamp
- `error_count: int` - Error counter
- `max_errors: int` - Maximum allowed errors

##### `ApplicationConfig`
Application configuration settings.

**Attributes:**
- `name: str` - Application name (default: "SilentSteno")
- `version: str` - Application version
- `environment: str` - Environment (development/production)
- `startup_timeout: float` - Startup timeout in seconds
- `shutdown_timeout: float` - Shutdown timeout in seconds
- `health_check_interval: float` - Health check interval
- `max_component_errors: int` - Max errors before component failure
- `enable_monitoring: bool` - Enable performance monitoring
- `enable_auto_recovery: bool` - Enable automatic recovery
- `log_level: str` - Logging level
- `thread_pool_size: int` - Thread pool size
- `device_management_enabled: bool` - Enable device management
- `device_config_path: str` - Path to device configuration

##### `ComponentManager`
Manages component lifecycle and dependencies.

**Methods:**
- `register_component(name, instance, dependencies, startup_priority, shutdown_priority, health_check)` - Register a component
- `unregister_component(name)` - Unregister a component
- `start_component(name)` - Start a specific component
- `stop_component(name)` - Stop a specific component
- `start_all_components()` - Start all components in dependency order
- `stop_all_components()` - Stop all components in reverse order
- `get_component_status()` - Get status of all components

##### `ApplicationController`
Main application controller orchestrating all subsystems.

**Attributes:**
- `config: ApplicationConfig` - Application configuration
- `state: AppState` - Current application state
- `event_bus: EventBus` - Event system
- `error_handler: ErrorHandler` - Error handling system
- `component_manager: ComponentManager` - Component manager
- `performance_monitor: PerformanceMonitor` - Performance monitoring
- `device_manager: DeviceManager` - Device management

**Methods:**
- `initialize()` - Initialize the application controller
- `start()` - Start the application
- `stop()` - Stop the application
- `restart()` - Restart the application
- `restart_component(component_name)` - Restart a specific component
- `get_status()` - Get comprehensive application status

**Usage Example:**
```python
# Create controller with configuration
config = {"application": {"log_level": "INFO", "enable_monitoring": True}}
controller = ApplicationController(config)

# Initialize and start
controller.initialize()
controller.start()

# Register a component
controller.component_manager.register_component(
    "my_service",
    my_service_instance,
    dependencies=["database", "event_bus"],
    startup_priority=20
)

# Get application status
status = controller.get_status()
print(f"App state: {status['application_state']}")
```

### 3. `config.py`

**Purpose**: Hierarchical configuration system with validation, hot-reload capabilities, and environment-specific settings management.

#### Classes

##### `ConfigValidationError`
Custom exception for configuration validation errors.

##### `ConfigSchema`
Base configuration schema using Pydantic if available.

##### `ConfigSource`
Configuration source information.

**Attributes:**
- `path: str` - File path
- `priority: int` - Priority (lower = higher priority)
- `watch: bool` - Enable file watching
- `required: bool` - Is source required
- `last_modified: float` - Last modification time
- `hash: str` - File content hash

##### `ConfigValidator`
Configuration validation system.

**Methods:**
- `register_validator(key_path, validator_func)` - Register a validator function
- `register_schema(key_path, schema_class)` - Register a Pydantic schema
- `validate_config(config, key_path)` - Validate configuration

##### `ConfigWatcher`
File system watcher for configuration changes (requires watchdog).

##### `ConfigManager`
Hierarchical configuration management system.

**Methods:**
- `add_source(path, priority, watch, required)` - Add configuration source
- `remove_source(path)` - Remove configuration source
- `get(key_path, default)` - Get configuration value by dot-path
- `set(key_path, value, validate)` - Set configuration value
- `update(config_dict, validate)` - Update with dictionary
- `register_change_callback(callback)` - Register change listener
- `save_to_file(path, section)` - Save configuration to file
- `reload()` - Reload all configuration sources
- `get_all()` - Get complete configuration
- `get_sources()` - Get list of sources

#### Global Functions

##### `load_config(path, priority, watch, required) -> Dict`
Load configuration from file using global config manager.

##### `save_config(path, section) -> bool`
Save configuration to file.

##### `get_config(key_path, default) -> Any`
Get configuration value.

##### `set_config(key_path, value, validate) -> bool`
Set configuration value.

##### `validate_config(config) -> bool`
Validate configuration.

##### `watch_config_changes(callback)`
Register change callback.

**Usage Example:**
```python
# Load configuration
config = load_config("config/app_config.json", priority=10)

# Get nested value
db_host = get_config("database.host", "localhost")

# Set value
set_config("app.debug", True)

# Watch for changes
def on_config_change(key_path, new_value, old_value):
    print(f"Config changed: {key_path}")

watch_config_changes(on_config_change)

# Use ConfigManager directly
manager = ConfigManager(auto_reload=True)
manager.add_source("config/base.json", priority=100)
manager.add_source("config/local.json", priority=10)
```

### 4. `errors.py`

**Purpose**: Comprehensive error handling, recovery, and fallback management system.

#### Enums

##### `ErrorSeverity`
Error severity levels.
- `LOW` - Minor issues
- `MEDIUM` - Degraded functionality
- `HIGH` - Major functionality impact
- `CRITICAL` - System failure

##### `RecoveryStrategy`
Recovery strategies for errors.
- `RETRY` - Retry the operation
- `RESTART` - Restart the component
- `FALLBACK` - Use fallback mechanism
- `IGNORE` - Log and continue
- `ESCALATE` - Escalate to higher level

#### Classes

##### `ErrorContext`
Context information for errors.

**Attributes:**
- `error: Exception` - The error instance
- `timestamp: float` - When error occurred
- `component: str` - Component where error occurred
- `operation: str` - Operation being performed
- `severity: ErrorSeverity` - Error severity
- `retry_count: int` - Number of retries
- `additional_info: Dict` - Extra context

##### `RecoveryAction`
Recovery action definition.

**Attributes:**
- `strategy: RecoveryStrategy` - Recovery strategy
- `handler: Callable` - Recovery handler function
- `max_retries: int` - Maximum retry attempts
- `retry_delay: float` - Delay between retries
- `fallback_handler: Callable` - Fallback handler

##### `ErrorHandler`
Central error handling system.

**Methods:**
- `handle_error(error, context, severity, component)` - Handle an error
- `register_handler(error_type, handler)` - Register error handler
- `register_recovery(error_type, recovery_action)` - Register recovery
- `get_error_stats()` - Get error statistics

##### `RecoveryManager`
Manages error recovery operations.

**Methods:**
- `attempt_recovery(error_context)` - Attempt recovery
- `register_strategy(error_type, strategy, handler)` - Register strategy
- `can_recover(error_context)` - Check if recoverable

##### `ErrorReporter`
Error reporting and notification system.

**Methods:**
- `report_error(error_context)` - Report an error
- `add_reporter(reporter_func)` - Add error reporter
- `get_error_report(time_range)` - Get error report

##### `FallbackManager`
Manages fallback operations.

**Methods:**
- `register_fallback(component, fallback_func)` - Register fallback
- `activate_fallback(component)` - Activate fallback mode
- `deactivate_fallback(component)` - Deactivate fallback
- `is_in_fallback(component)` - Check fallback status

#### Global Functions

##### `handle_error(error, context, severity, component)`
Handle an error using global error handler.

##### `attempt_recovery(error_context) -> bool`
Attempt recovery using global recovery manager.

##### `report_error(error_context)`
Report error using global reporter.

##### `activate_fallback(component) -> bool`
Activate fallback for component.

**Usage Example:**
```python
# Basic error handling
try:
    risky_operation()
except Exception as e:
    handle_error(e, "Failed to perform operation", 
                severity=ErrorSeverity.HIGH,
                component="audio_processor")

# Register custom error handler
def custom_handler(error_context):
    if isinstance(error_context.error, ConnectionError):
        # Handle connection errors
        return True
    return False

error_handler = get_global_error_handler()
error_handler.register_handler(ConnectionError, custom_handler)

# Register recovery strategy
recovery_manager = get_global_recovery_manager()
recovery_manager.register_strategy(
    ConnectionError,
    RecoveryStrategy.RETRY,
    lambda ctx: reconnect_service()
)
```

### 5. `events.py`

**Purpose**: Event system for loosely-coupled inter-component communication using publish-subscribe pattern.

#### Enums

##### `EventPriority`
Event priority levels.
- `LOW` - Low priority
- `NORMAL` - Normal priority
- `HIGH` - High priority
- `CRITICAL` - Critical priority

#### Classes

##### `Event`
Event data structure.

**Attributes:**
- `name: str` - Event name
- `data: Any` - Event payload
- `timestamp: float` - Event timestamp
- `source: str` - Event source
- `priority: EventPriority` - Event priority
- `tags: Set[str]` - Event tags

##### `EventSubscription`
Event subscription information.

**Attributes:**
- `event_pattern: str` - Event name pattern
- `handler: Callable` - Handler function
- `filter_func: Callable` - Optional filter
- `priority: int` - Handler priority
- `async_handler: bool` - Is async handler
- `subscription_id: str` - Unique ID

##### `EventHandler`
Base class for event handlers.

**Methods:**
- `handle_event(event)` - Handle an event
- `should_handle(event)` - Check if should handle

##### `EventBus`
Central event publishing and subscription system.

**Methods:**
- `publish(event)` - Publish an event
- `subscribe(event_pattern, handler, filter_func, priority)` - Subscribe to events
- `unsubscribe(subscription_id)` - Unsubscribe from events
- `wait_for_event(event_pattern, timeout)` - Wait for specific event
- `get_event_history(event_pattern, limit)` - Get event history

#### Global Functions

##### `create_event_bus() -> EventBus`
Create a new event bus instance.

##### `publish_event(name, data, source, priority, tags)`
Publish event using global event bus.

##### `subscribe_to_event(event_pattern, handler, filter_func, priority) -> str`
Subscribe to events using global event bus.

##### `unsubscribe(subscription_id)`
Unsubscribe from events.

**Usage Example:**
```python
# Create and use event bus
event_bus = create_event_bus()

# Subscribe to events
def on_session_start(event):
    print(f"Session started: {event.data}")

subscription_id = event_bus.subscribe("session.started", on_session_start)

# Publish events
event_bus.publish(Event("session.started", {
    "session_id": "123",
    "user": "john"
}))

# Use global functions
subscribe_to_event("error.*", lambda e: log_error(e.data))
publish_event("error.network", {"message": "Connection lost"})

# Async handler
async def async_handler(event):
    await process_event_async(event)

subscribe_to_event("data.received", async_handler)

# Wait for event
event = event_bus.wait_for_event("task.completed", timeout=30.0)
```

### 6. `logging.py`

**Purpose**: Structured logging system with JSON output, context management, and automatic log rotation.

#### Classes

##### `LogConfig`
Logging configuration.

**Attributes:**
- `level: str` - Log level
- `format: str` - Log format
- `console_enabled: bool` - Enable console output
- `file_enabled: bool` - Enable file output
- `file_path: str` - Log file path
- `max_file_size: int` - Max file size in bytes
- `backup_count: int` - Number of backup files
- `json_format: bool` - Use JSON format

##### `StructuredFormatter`
JSON formatter for structured logging.

##### `DetailedFormatter`
Human-readable formatter with detailed information.

##### `StructuredLogger`
Enhanced logger with context support.

**Methods:**
- `bind(**context)` - Add context to logger
- `unbind(*keys)` - Remove context keys
- `debug/info/warning/error/critical(msg, **extra)` - Log with context

##### `LogRotator`
Manages log file rotation.

**Methods:**
- `setup_rotation(logger, config)` - Setup log rotation
- `force_rotation()` - Force immediate rotation

##### `LogManager`
Central log management system.

**Methods:**
- `setup_logger(name, config)` - Setup a logger
- `get_logger(name)` - Get a logger instance
- `set_global_level(level)` - Set global log level
- `add_handler(handler)` - Add log handler
- `remove_handler(handler)` - Remove log handler

#### Global Functions

##### `setup_logging(config) -> LogManager`
Setup logging using configuration.

##### `get_logger(name) -> StructuredLogger`
Get a logger instance.

##### `configure_log_rotation(logger_name, max_size, backup_count)`
Configure log rotation for a logger.

##### `add_log_handler(handler)`
Add a log handler globally.

**Usage Example:**
```python
# Setup logging
setup_logging({
    "level": "INFO",
    "console_enabled": True,
    "file_enabled": True,
    "file_path": "logs/app.log",
    "json_format": True,
    "max_file_size": 10485760,  # 10MB
    "backup_count": 5
})

# Get logger
logger = get_logger(__name__)

# Log with context
logger = logger.bind(user_id="123", session_id="abc")
logger.info("User logged in", action="login")

# Log structured data
logger.info("Processing complete", 
           duration=1.23, 
           records_processed=100,
           status="success")

# Configure rotation
configure_log_rotation("my_module", max_size=5*1024*1024, backup_count=3)
```

### 7. `monitoring.py`

**Purpose**: System performance monitoring, health checking, and alerting system.

#### Enums

##### `MetricType`
Types of metrics.
- `COUNTER` - Incrementing counter
- `GAUGE` - Point-in-time value
- `HISTOGRAM` - Distribution of values
- `TIMER` - Time measurements

##### `HealthStatus`
Health status levels.
- `HEALTHY` - Everything normal
- `WARNING` - Minor issues
- `CRITICAL` - Major issues
- `UNKNOWN` - Status unknown

##### `AlertLevel`
Alert severity levels.
- `INFO` - Informational
- `WARNING` - Warning condition
- `ERROR` - Error condition
- `CRITICAL` - Critical condition

#### Classes

##### `Metric`
Metric data point.

**Attributes:**
- `name: str` - Metric name
- `type: MetricType` - Metric type
- `value: float` - Current value
- `timestamp: float` - Timestamp
- `tags: Dict[str, str]` - Metric tags
- `unit: str` - Unit of measurement

##### `HealthCheck`
Health check definition.

**Attributes:**
- `name: str` - Check name
- `check_func: Callable` - Check function
- `interval: float` - Check interval
- `timeout: float` - Check timeout
- `critical: bool` - Is critical check

##### `Alert`
Alert definition.

**Attributes:**
- `name: str` - Alert name
- `level: AlertLevel` - Alert level
- `message: str` - Alert message
- `timestamp: float` - When triggered
- `component: str` - Affected component
- `metadata: Dict` - Additional data

##### `MetricsCollector`
Collects and manages metrics.

**Methods:**
- `record_metric(name, value, type, tags, unit)` - Record a metric
- `increment_counter(name, value, tags)` - Increment counter
- `set_gauge(name, value, tags)` - Set gauge value
- `record_timer(name, duration, tags)` - Record timing
- `get_metrics(name_pattern, time_range)` - Get metrics

##### `HealthChecker`
Performs system health checks.

**Methods:**
- `register_check(name, check_func, interval, timeout, critical)` - Register check
- `perform_checks()` - Run all checks
- `get_health_status()` - Get overall status
- `get_check_results()` - Get individual results

##### `AlertManager`
Manages system alerts.

**Methods:**
- `send_alert(name, level, message, component, metadata)` - Send alert
- `register_handler(level, handler_func)` - Register alert handler
- `get_active_alerts()` - Get active alerts
- `acknowledge_alert(alert_id)` - Acknowledge an alert

##### `PerformanceMonitor`
Main performance monitoring system.

**Methods:**
- `start_monitoring()` - Start monitoring
- `stop_monitoring()` - Stop monitoring
- `collect_system_metrics()` - Collect system metrics
- `get_performance_report()` - Get performance report

#### Global Functions

##### `start_monitoring(config) -> PerformanceMonitor`
Start performance monitoring.

##### `collect_metrics(name, value, type, tags)`
Collect a metric using global collector.

##### `check_system_health() -> HealthStatus`
Check system health.

##### `send_alert(name, level, message, component)`
Send an alert.

**Usage Example:**
```python
# Start monitoring
monitor = start_monitoring({
    "enable_system_metrics": True,
    "collection_interval": 60
})

# Record metrics
collect_metrics("api.requests", 1, MetricType.COUNTER, {"endpoint": "/api/v1"})
collect_metrics("queue.size", 42, MetricType.GAUGE, {"queue": "tasks"})

# Record timing
start_time = time.time()
process_data()
duration = time.time() - start_time
collect_metrics("processing.time", duration, MetricType.TIMER, {"operation": "data_processing"})

# Register health check
def check_database():
    try:
        db.ping()
        return True
    except:
        return False

health_checker = get_global_health_checker()
health_checker.register_check("database", check_database, interval=30.0)

# Send alerts
send_alert("disk_space_low", AlertLevel.WARNING, 
          "Disk space below 10%", "storage_manager")
```

### 8. `registry.py` & `registry_backup.py`

**Purpose**: Component registry and dependency injection system. The `registry_backup.py` appears to be the full-featured version.

#### Classes (from registry_backup.py)

##### `ComponentRegistry`
Central registry for components with dependency injection support.

**Methods:**
- `register(name, component, singleton, factory, dependencies)` - Register component
- `get(name)` - Get component instance
- `has(name)` - Check if component exists
- `remove(name)` - Remove component
- `clear()` - Clear all components
- `get_all()` - Get all registered components

##### `DependencyInjector`
Handles automatic dependency injection.

**Methods:**
- `inject_dependencies(instance, dependencies)` - Inject dependencies
- `resolve_dependencies(component_name)` - Resolve dependency tree
- `create_with_dependencies(factory, dependencies)` - Create with injection

##### `ComponentFactory`
Factory for creating component instances.

**Methods:**
- `create_component(name, class_or_factory, args, kwargs)` - Create component
- `register_factory(name, factory_func)` - Register factory

##### `ServiceContainer`
Advanced container managing component lifecycles.

**Methods:**
- `add_service(name, service, lifecycle)` - Add service
- `get_service(name)` - Get service instance
- `start_all()` - Start all services
- `stop_all()` - Stop all services

#### Decorators

##### `@injectable`
Mark a class as injectable with automatic dependency resolution.

##### `@singleton`
Mark a class as singleton (single instance).

##### `@factory`
Mark a function as a component factory.

#### Global Functions

##### `register_component(name, component, **options)`
Register a component in global registry.

##### `get_component(name) -> Any`
Get component from global registry.

##### `inject_dependencies(instance) -> Any`
Inject dependencies into instance.

##### `create_component(name, *args, **kwargs) -> Any`
Create component with dependencies.

**Usage Example:**
```python
# Basic registration
register_component("database", DatabaseConnection())
register_component("cache", CacheManager(), singleton=True)

# With factory
def create_service(config):
    return MyService(config)

register_component("my_service", factory=create_service)

# Using decorators
@injectable
@singleton
class UserService:
    def __init__(self, database: "database", cache: "cache"):
        self.db = database
        self.cache = cache

# Register and use
register_component("user_service", UserService)
user_service = get_component("user_service")  # Dependencies auto-injected

# Manual dependency injection
class OrderService:
    database: "database" = None
    user_service: "user_service" = None
    
    def process_order(self, order_id):
        user = self.user_service.get_user(order.user_id)
        # ...

order_service = OrderService()
inject_dependencies(order_service)  # Injects database and user_service
```

## Module Integration

The Core module serves as the foundation for the entire Silent Steno application:

1. **Application Bootstrap**: The `SilentStenoApp` class in `__init__.py` provides the main entry point
2. **Component Lifecycle**: `ApplicationController` manages all component lifecycles
3. **Configuration**: `ConfigManager` provides hierarchical configuration with hot-reload
4. **Event System**: `EventBus` enables loose coupling between components
5. **Error Handling**: `ErrorHandler` provides centralized error management with recovery
6. **Logging**: `StructuredLogger` provides consistent logging across all components
7. **Monitoring**: `PerformanceMonitor` tracks system health and performance
8. **Dependency Injection**: `ComponentRegistry` manages dependencies automatically

## Common Usage Patterns

### Application Startup Sequence
```python
# 1. Create application
app = create_application("config/app_config.json")

# 2. Register custom components
registry = get_global_component_registry()
registry.register("audio_processor", AudioProcessor, singleton=True)
registry.register("session_manager", SessionManager)

# 3. Setup event handlers
event_bus = get_global_event_bus()
event_bus.subscribe("session.*", session_event_handler)

# 4. Configure error recovery
error_handler = get_global_error_handler()
error_handler.register_recovery(ConnectionError, 
    RecoveryAction(RecoveryStrategy.RETRY, max_retries=3))

# 5. Start application
app.start()
```

### Component Development Pattern
```python
@injectable
class MyComponent:
    # Declare dependencies
    event_bus: "event_bus" = None
    config_manager: "config_manager" = None
    logger: "logger" = None
    
    def initialize(self):
        """Called when component starts"""
        self.config = self.config_manager.get("my_component", {})
        self.logger.info("MyComponent initialized")
        
    def start(self):
        """Start component operations"""
        self.event_bus.publish(Event("my_component.started", {}))
        
    def stop(self):
        """Stop component operations"""
        self.event_bus.publish(Event("my_component.stopped", {}))
        
    def health_check(self) -> bool:
        """Health check function"""
        return self.is_healthy

# Register and use
register_component("my_component", MyComponent)
```

### Error Handling Pattern
```python
class ServiceWithErrorHandling:
    def __init__(self):
        self.error_handler = get_global_error_handler()
        self.fallback_manager = get_global_fallback_manager()
        
    def risky_operation(self):
        try:
            return self._do_risky_operation()
        except Exception as e:
            # Handle with automatic recovery
            handle_error(e, "Risky operation failed",
                       severity=ErrorSeverity.HIGH,
                       component="my_service")
            
            # Activate fallback if needed
            if self.error_handler.get_error_count("my_service") > 5:
                activate_fallback("my_service")
                return self._fallback_operation()
```

### Configuration with Validation
```python
# Define schema with Pydantic
class DatabaseConfig(ConfigSchema):
    host: str
    port: int = 5432
    username: str
    password: str
    pool_size: int = 10
    
    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v

# Register schema
config_manager = get_global_config_manager()
config_manager.validator.register_schema("database", DatabaseConfig)

# Use configuration
db_config = get_config("database")
# Automatically validated against schema
```

### Event-Driven Communication
```python
# Publisher
class DataProcessor:
    def process_data(self, data):
        # Process data
        result = self._process(data)
        
        # Publish completion event
        publish_event("data.processed", {
            "result": result,
            "timestamp": time.time()
        }, priority=EventPriority.HIGH)

# Subscriber
class DataConsumer:
    def __init__(self):
        subscribe_to_event("data.processed", self.on_data_processed)
        
    def on_data_processed(self, event: Event):
        result = event.data["result"]
        # Handle processed data
```

### Monitoring and Alerting
```python
class MonitoredService:
    def __init__(self):
        self.metrics = get_global_metrics_collector()
        self.alert_manager = get_global_alert_manager()
        
    def process_request(self, request):
        # Record request
        self.metrics.increment_counter("requests.total", tags={"service": "api"})
        
        start_time = time.time()
        try:
            result = self._process(request)
            self.metrics.increment_counter("requests.success", tags={"service": "api"})
            return result
        except Exception as e:
            self.metrics.increment_counter("requests.failed", tags={"service": "api"})
            
            # Send alert for critical errors
            if self._is_critical_error(e):
                self.alert_manager.send_alert(
                    "critical_error",
                    AlertLevel.CRITICAL,
                    f"Critical error in API: {str(e)}",
                    "api_service"
                )
            raise
        finally:
            # Record timing
            duration = time.time() - start_time
            self.metrics.record_timer("request.duration", duration, tags={"service": "api"})
```

This comprehensive documentation covers all aspects of the Core module, providing both technical details and practical usage examples for developers working with The Silent Steno codebase.