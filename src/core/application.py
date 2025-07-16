"""
Main Application Controller

Central orchestrator for all Silent Steno subsystems, managing component lifecycle,
inter-component communication, and application state coordination.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Set
from concurrent.futures import ThreadPoolExecutor

from .events import EventBus, Event, create_event_bus
from .config import ConfigManager
from .errors import ErrorHandler, handle_error
from .monitoring import PerformanceMonitor, start_monitoring
from .registry import ComponentRegistry, register_component, get_component
from ..system.device_manager import DeviceManager, DeviceConfig, create_device_manager, load_device_config


class AppState(Enum):
    """Application state enumeration."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ComponentState(Enum):
    """Component state enumeration."""
    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    READY = "ready"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    FAILED = "failed"


@dataclass
class ComponentInfo:
    """Information about a registered component."""
    name: str
    instance: Any
    state: ComponentState = ComponentState.REGISTERED
    dependencies: List[str] = field(default_factory=list)
    startup_priority: int = 50  # Lower numbers start first
    shutdown_priority: int = 50  # Lower numbers stop last
    health_check: Optional[Callable] = None
    last_health_check: Optional[float] = None
    error_count: int = 0
    max_errors: int = 5


@dataclass
class ApplicationConfig:
    """Application configuration."""
    name: str = "SilentSteno"
    version: str = "0.1.0"
    environment: str = "development"
    startup_timeout: float = 30.0
    shutdown_timeout: float = 15.0
    health_check_interval: float = 30.0
    max_component_errors: int = 5
    enable_monitoring: bool = True
    enable_auto_recovery: bool = True
    log_level: str = "INFO"
    thread_pool_size: int = 4
    device_management_enabled: bool = True
    device_config_path: str = "config/device_config.json"


class ComponentManager:
    """Manages component lifecycle and dependencies."""
    
    def __init__(self, event_bus: EventBus, error_handler: ErrorHandler):
        self.event_bus = event_bus
        self.error_handler = error_handler
        self.components: Dict[str, ComponentInfo] = {}
        self.startup_order: List[str] = []
        self.shutdown_order: List[str] = []
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def register_component(self, name: str, instance: Any, dependencies: List[str] = None,
                          startup_priority: int = 50, shutdown_priority: int = 50,
                          health_check: Callable = None) -> bool:
        """Register a component with the manager."""
        try:
            with self.lock:
                if name in self.components:
                    self.logger.warning(f"Component {name} already registered, updating")
                
                component_info = ComponentInfo(
                    name=name,
                    instance=instance,
                    dependencies=dependencies or [],
                    startup_priority=startup_priority,
                    shutdown_priority=shutdown_priority,
                    health_check=health_check
                )
                
                self.components[name] = component_info
                self._update_startup_order()
                self._update_shutdown_order()
                
                # Publish component registration event
                self.event_bus.publish(Event("component.registered", {
                    "component": name,
                    "dependencies": dependencies or []
                }))
                
                self.logger.info(f"Component {name} registered successfully")
                return True
                
        except Exception as e:
            self.error_handler.handle_error(e, f"Failed to register component {name}")
            return False
    
    def unregister_component(self, name: str) -> bool:
        """Unregister a component from the manager."""
        try:
            with self.lock:
                if name not in self.components:
                    self.logger.warning(f"Component {name} not registered")
                    return False
                
                component = self.components[name]
                
                # Stop component if running
                if component.state in [ComponentState.RUNNING, ComponentState.STARTING]:
                    self.stop_component(name)
                
                # Remove from collections
                del self.components[name]
                self._update_startup_order()
                self._update_shutdown_order()
                
                # Publish event
                self.event_bus.publish(Event("component.unregistered", {"component": name}))
                
                self.logger.info(f"Component {name} unregistered successfully")
                return True
                
        except Exception as e:
            self.error_handler.handle_error(e, f"Failed to unregister component {name}")
            return False
    
    def start_component(self, name: str) -> bool:
        """Start a specific component."""
        try:
            with self.lock:
                if name not in self.components:
                    self.logger.error(f"Component {name} not registered")
                    return False
                
                component = self.components[name]
                
                if component.state == ComponentState.RUNNING:
                    self.logger.info(f"Component {name} already running")
                    return True
                
                # Check dependencies
                for dep in component.dependencies:
                    if dep not in self.components:
                        self.logger.error(f"Component {name} dependency {dep} not registered")
                        return False
                    
                    if self.components[dep].state != ComponentState.RUNNING:
                        self.logger.error(f"Component {name} dependency {dep} not running")
                        return False
                
                # Start component
                component.state = ComponentState.STARTING
                self.event_bus.publish(Event("component.starting", {"component": name}))
                
                # Call component start method if it exists
                if hasattr(component.instance, 'start'):
                    component.instance.start()
                elif hasattr(component.instance, 'initialize'):
                    component.instance.initialize()
                
                component.state = ComponentState.RUNNING
                self.event_bus.publish(Event("component.started", {"component": name}))
                
                self.logger.info(f"Component {name} started successfully")
                return True
                
        except Exception as e:
            if name in self.components:
                self.components[name].state = ComponentState.ERROR
                self.components[name].error_count += 1
            self.error_handler.handle_error(e, f"Failed to start component {name}")
            return False
    
    def stop_component(self, name: str) -> bool:
        """Stop a specific component."""
        try:
            with self.lock:
                if name not in self.components:
                    self.logger.error(f"Component {name} not registered")
                    return False
                
                component = self.components[name]
                
                if component.state in [ComponentState.STOPPED, ComponentState.STOPPING]:
                    self.logger.info(f"Component {name} already stopped/stopping")
                    return True
                
                # Stop component
                component.state = ComponentState.STOPPING
                self.event_bus.publish(Event("component.stopping", {"component": name}))
                
                # Call component stop method if it exists
                if hasattr(component.instance, 'stop'):
                    component.instance.stop()
                elif hasattr(component.instance, 'shutdown'):
                    component.instance.shutdown()
                
                component.state = ComponentState.STOPPED
                self.event_bus.publish(Event("component.stopped", {"component": name}))
                
                self.logger.info(f"Component {name} stopped successfully")
                return True
                
        except Exception as e:
            if name in self.components:
                self.components[name].state = ComponentState.ERROR
                self.components[name].error_count += 1
            self.error_handler.handle_error(e, f"Failed to stop component {name}")
            return False
    
    def start_all_components(self) -> bool:
        """Start all components in dependency order."""
        try:
            self.logger.info("Starting all components")
            
            for component_name in self.startup_order:
                if not self.start_component(component_name):
                    self.logger.error(f"Failed to start component {component_name}")
                    return False
            
            self.logger.info("All components started successfully")
            return True
            
        except Exception as e:
            self.error_handler.handle_error(e, "Failed to start all components")
            return False
    
    def stop_all_components(self) -> bool:
        """Stop all components in reverse dependency order."""
        try:
            self.logger.info("Stopping all components")
            
            for component_name in self.shutdown_order:
                if not self.stop_component(component_name):
                    self.logger.error(f"Failed to stop component {component_name}")
                    # Continue stopping other components
            
            self.logger.info("All components stopped")
            return True
            
        except Exception as e:
            self.error_handler.handle_error(e, "Failed to stop all components")
            return False
    
    def get_component_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all components."""
        with self.lock:
            status = {}
            for name, component in self.components.items():
                status[name] = {
                    "state": component.state.value,
                    "error_count": component.error_count,
                    "last_health_check": component.last_health_check,
                    "dependencies": component.dependencies
                }
            return status
    
    def _update_startup_order(self):
        """Update component startup order based on dependencies and priorities."""
        # Topological sort with priority consideration
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(name):
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {name}")
            if name in visited:
                return
            
            temp_visited.add(name)
            
            if name in self.components:
                for dep in self.components[name].dependencies:
                    if dep in self.components:
                        visit(dep)
            
            temp_visited.remove(name)
            visited.add(name)
            order.append(name)
        
        for component_name in self.components:
            if component_name not in visited:
                visit(component_name)
        
        # Sort by priority within dependency groups
        self.startup_order = sorted(order, key=lambda name: self.components[name].startup_priority)
    
    def _update_shutdown_order(self):
        """Update component shutdown order (reverse of startup with shutdown priorities)."""
        # Reverse topological sort
        self.shutdown_order = list(reversed(self.startup_order))
        # Sort by shutdown priority
        self.shutdown_order = sorted(self.shutdown_order, 
                                   key=lambda name: self.components[name].shutdown_priority)


class ApplicationController:
    """
    Main application controller orchestrating all subsystems.
    
    Manages component lifecycle, inter-component communication, and application state.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        # Extract application config from nested structure
        config_dict = config or {}
        app_config = config_dict.get('application', {})
        
        self.config = ApplicationConfig(**app_config)
        self.state = AppState.UNINITIALIZED
        self.event_bus = create_event_bus()
        self.error_handler = ErrorHandler()
        self.component_manager = ComponentManager(self.event_bus, self.error_handler)
        self.performance_monitor = None
        self.health_check_timer = None
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        self.device_manager: Optional[DeviceManager] = None
        
        self.logger = logging.getLogger(__name__)
        self._state_lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Register core components
        self._register_core_components()
        
        # Set up event handlers
        self._setup_event_handlers()
    
    def initialize(self) -> bool:
        """Initialize the application controller and core systems."""
        try:
            with self._state_lock:
                if self.state != AppState.UNINITIALIZED:
                    self.logger.warning(f"Application already initialized (state: {self.state})")
                    return True
                
                self.state = AppState.INITIALIZING
                self.logger.info("Initializing application controller")
                
                # Initialize core systems
                if self.config.enable_monitoring:
                    self.performance_monitor = start_monitoring()
                    if self.performance_monitor:
                        self.component_manager.register_component(
                            "performance_monitor",
                            self.performance_monitor,
                            startup_priority=10
                        )
                
                # Initialize device management if enabled
                if self.config.device_management_enabled:
                    self._initialize_device_management()
                
                # Initialize component registry integration
                register_component("application_controller", self)
                register_component("event_bus", self.event_bus)
                register_component("error_handler", self.error_handler)
                
                self.state = AppState.READY
                self.logger.info("Application controller initialized successfully")
                
                # Publish initialization event
                self.event_bus.publish(Event("application.initialized", {
                    "controller": self,
                    "config": self.config.__dict__
                }))
                
                return True
                
        except Exception as e:
            self.state = AppState.ERROR
            self.error_handler.handle_error(e, "Failed to initialize application controller")
            return False
    
    def start(self) -> bool:
        """Start the application and all registered components."""
        try:
            with self._state_lock:
                if self.state not in [AppState.READY, AppState.STOPPED]:
                    self.logger.error(f"Cannot start application in state {self.state}")
                    return False
                
                self.state = AppState.STARTING
                self.logger.info("Starting Silent Steno application")
                
                # Publish starting event
                self.event_bus.publish(Event("application.starting", {"controller": self}))
                
                # Start all components
                if not self.component_manager.start_all_components():
                    self.state = AppState.ERROR
                    self.logger.error("Failed to start all components")
                    return False
                
                # Start health checking
                if self.config.health_check_interval > 0:
                    self._start_health_checking()
                
                self.state = AppState.RUNNING
                self.logger.info("Silent Steno application started successfully")
                
                # Publish started event
                self.event_bus.publish(Event("application.started", {"controller": self}))
                
                return True
                
        except Exception as e:
            self.state = AppState.ERROR
            self.error_handler.handle_error(e, "Failed to start application")
            return False
    
    def stop(self) -> bool:
        """Stop the application and all registered components."""
        try:
            with self._state_lock:
                if self.state in [AppState.STOPPED, AppState.STOPPING]:
                    self.logger.info("Application already stopped/stopping")
                    return True
                
                self.state = AppState.STOPPING
                self.logger.info("Stopping Silent Steno application")
                
                # Set shutdown event
                self._shutdown_event.set()
                
                # Publish stopping event
                self.event_bus.publish(Event("application.stopping", {"controller": self}))
                
                # Stop health checking
                if self.health_check_timer:
                    self.health_check_timer.cancel()
                    self.health_check_timer = None
                
                # Stop all components
                self.component_manager.stop_all_components()
                
                # Shutdown thread pool
                self.thread_pool.shutdown(wait=True, timeout=5.0)
                
                self.state = AppState.STOPPED
                self.logger.info("Silent Steno application stopped successfully")
                
                # Publish stopped event
                self.event_bus.publish(Event("application.stopped", {"controller": self}))
                
                return True
                
        except Exception as e:
            self.state = AppState.ERROR
            self.error_handler.handle_error(e, "Failed to stop application")
            return False
    
    def restart(self) -> bool:
        """Restart the application."""
        self.logger.info("Restarting Silent Steno application")
        if self.stop() and self.start():
            self.logger.info("Application restarted successfully")
            return True
        else:
            self.logger.error("Failed to restart application")
            return False
    
    def restart_component(self, component_name: str) -> bool:
        """Restart a specific component."""
        return (self.component_manager.stop_component(component_name) and
                self.component_manager.start_component(component_name))
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive application status."""
        status = {
            "application_state": self.state.value,
            "uptime": time.time() - getattr(self, '_start_time', time.time()),
            "components": self.component_manager.get_component_status(),
            "performance": getattr(self.performance_monitor, 'get_metrics', lambda: {})(),
            "configuration": self.config.__dict__
        }
        
        # Add device management status if enabled
        if self.device_manager:
            status["device_management"] = self.device_manager.get_management_summary()
        
        return status
    
    def _register_core_components(self):
        """Register core application components."""
        # Register event bus
        self.component_manager.register_component(
            "event_bus",
            self.event_bus,
            startup_priority=5
        )
        
        # Register error handler
        self.component_manager.register_component(
            "error_handler", 
            self.error_handler,
            startup_priority=5
        )
    
    def _setup_event_handlers(self):
        """Set up application-level event handlers."""
        self.event_bus.subscribe("component.error", self._handle_component_error)
        self.event_bus.subscribe("system.shutdown", self._handle_shutdown_request)
        self.event_bus.subscribe("application.restart", self._handle_restart_request)
    
    def _handle_component_error(self, event: Event):
        """Handle component error events."""
        component_name = event.data.get("component")
        error = event.data.get("error")
        
        self.logger.error(f"Component {component_name} reported error: {error}")
        
        if self.config.enable_auto_recovery:
            # Attempt component restart
            self.thread_pool.submit(self._attempt_component_recovery, component_name)
    
    def _handle_shutdown_request(self, event: Event):
        """Handle shutdown request events."""
        self.logger.info("Shutdown request received")
        self.thread_pool.submit(self.stop)
    
    def _handle_restart_request(self, event: Event):
        """Handle restart request events."""
        component = event.data.get("component")
        if component:
            self.logger.info(f"Restart request received for component {component}")
            self.thread_pool.submit(self.restart_component, component)
        else:
            self.logger.info("Application restart request received")
            self.thread_pool.submit(self.restart)
    
    def _attempt_component_recovery(self, component_name: str):
        """Attempt to recover a failed component."""
        try:
            self.logger.info(f"Attempting recovery for component {component_name}")
            
            # Give component a moment to stabilize
            time.sleep(1.0)
            
            if self.restart_component(component_name):
                self.logger.info(f"Component {component_name} recovered successfully")
                self.event_bus.publish(Event("component.recovered", {"component": component_name}))
            else:
                self.logger.error(f"Failed to recover component {component_name}")
                self.event_bus.publish(Event("component.recovery_failed", {"component": component_name}))
                
        except Exception as e:
            self.error_handler.handle_error(e, f"Error during recovery of component {component_name}")
    
    def _start_health_checking(self):
        """Start periodic health checking of components."""
        def health_check_loop():
            while not self._shutdown_event.is_set():
                try:
                    self._perform_health_checks()
                    time.sleep(self.config.health_check_interval)
                except Exception as e:
                    self.error_handler.handle_error(e, "Error during health check")
        
        self.health_check_timer = threading.Timer(
            self.config.health_check_interval,
            health_check_loop
        )
        self.health_check_timer.daemon = True
        self.health_check_timer.start()
    
    def _perform_health_checks(self):
        """Perform health checks on all components."""
        current_time = time.time()
        
        for name, component in self.component_manager.components.items():
            if component.health_check and component.state == ComponentState.RUNNING:
                try:
                    if component.health_check():
                        component.last_health_check = current_time
                    else:
                        self.logger.warning(f"Health check failed for component {name}")
                        self.event_bus.publish(Event("component.health_check_failed", {
                            "component": name,
                            "timestamp": current_time
                        }))
                        
                except Exception as e:
                    self.logger.error(f"Health check error for component {name}: {e}")
                    self.event_bus.publish(Event("component.health_check_error", {
                        "component": name,
                        "error": str(e),
                        "timestamp": current_time
                    }))
    
    def _initialize_device_management(self):
        """Initialize device management system."""
        try:
            # Load device configuration
            config_path = Path(self.config.device_config_path)
            if config_path.exists():
                device_config = load_device_config(str(config_path))
            else:
                device_config = DeviceConfig()
                self.logger.info("Using default device configuration")
            
            # Create device manager
            self.device_manager = create_device_manager(device_config)
            
            # Register device manager as a component
            self.component_manager.register_component(
                "device_manager",
                self.device_manager,
                startup_priority=15,
                health_check=lambda: self.device_manager.get_device_status().state.value != "critical"
            )
            
            # Set up device event handlers
            self.device_manager.add_event_callback("health_alert", self._handle_device_health_alert)
            self.device_manager.add_event_callback("maintenance_start", self._handle_maintenance_mode)
            self.device_manager.add_event_callback("maintenance_end", self._handle_maintenance_mode)
            self.device_manager.add_event_callback("error", self._handle_device_error)
            
            self.logger.info("Device management system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize device management: {e}")
            raise
    
    def _handle_device_health_alert(self, event_type: str, data: Dict[str, Any]):
        """Handle device health alerts."""
        self.logger.warning(f"Device health alert: {data}")
        self.event_bus.publish(Event("device.health_alert", data))
    
    def _handle_maintenance_mode(self, event_type: str, data: Dict[str, Any]):
        """Handle device maintenance mode changes."""
        self.logger.info(f"Device maintenance mode: {event_type} - {data}")
        self.event_bus.publish(Event(f"device.{event_type}", data))
    
    def _handle_device_error(self, event_type: str, data: Dict[str, Any]):
        """Handle device management errors."""
        self.logger.error(f"Device error: {data}")
        self.event_bus.publish(Event("device.error", data))


# Factory functions
def start_app(controller: ApplicationController = None, config: Dict[str, Any] = None) -> ApplicationController:
    """
    Start the Silent Steno application.
    
    Args:
        controller: Existing controller instance (optional)
        config: Configuration dictionary (optional)
        
    Returns:
        ApplicationController: Running application controller
    """
    if controller is None:
        controller = ApplicationController(config)
    
    if not controller.initialize():
        raise RuntimeError("Failed to initialize application controller")
    
    if not controller.start():
        raise RuntimeError("Failed to start application")
    
    return controller


def stop_app(controller: ApplicationController) -> bool:
    """
    Stop the Silent Steno application.
    
    Args:
        controller: Application controller to stop
        
    Returns:
        bool: True if stopped successfully
    """
    return controller.stop()


def restart_component(controller: ApplicationController, component_name: str) -> bool:
    """
    Restart a specific component.
    
    Args:
        controller: Application controller
        component_name: Name of component to restart
        
    Returns:
        bool: True if restarted successfully
    """
    return controller.restart_component(component_name)


def get_app_status(controller: ApplicationController) -> Dict[str, Any]:
    """
    Get application status.
    
    Args:
        controller: Application controller
        
    Returns:
        Dict: Application status information
    """
    return controller.get_status()