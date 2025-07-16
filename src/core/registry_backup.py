"""
Component Registry and Dependency Injection System

Dynamic component registration with dependency injection and factory pattern support
for modular application architecture.
"""

import inspect
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Type, TypeVar, Union
from collections import defaultdict
import weakref

T = TypeVar('T')


@dataclass
class ComponentInfo:
    """Information about a registered component."""
    name: str
    instance: Any
    component_type: Type
    singleton: bool = True
    factory: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    lifecycle_hooks: Dict[str, Callable] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencySpec:
    """Dependency specification for injection."""
    name: str
    component_type: Type
    required: bool = True
    default_value: Any = None


class DependencyInjector:
    """Handles dependency injection for components."""
    
    def __init__(self, registry):
        self.registry = registry
        self.logger = logging.getLogger(__name__)
    
    def inject_dependencies(self, instance: Any, component_info: ComponentInfo) -> Any:
        """Inject dependencies into a component instance."""
        try:
            # Get dependencies from component info
            for dep_name in component_info.dependencies:
                if hasattr(instance, dep_name):
                    dependency = self.registry.get_component(dep_name)
                    if dependency:
                        setattr(instance, dep_name, dependency)
                    else:
                        self.logger.warning(f"Dependency {dep_name} not found for {component_info.name}")
            
            # Auto-inject based on type annotations if available
            if hasattr(instance, '__annotations__'):
                self._inject_by_annotations(instance)
            
            return instance
            
        except Exception as e:
            self.logger.error(f"Error injecting dependencies into {component_info.name}: {e}")
            return instance
    
    def _inject_by_annotations(self, instance: Any):
        """Inject dependencies based on type annotations."""
        annotations = getattr(instance, '__annotations__', {})
        
        for attr_name, attr_type in annotations.items():
            if not hasattr(instance, attr_name) or getattr(instance, attr_name) is None:
                # Try to find component by type
                component = self.registry.get_component_by_type(attr_type)
                if component:
                    setattr(instance, attr_name, component)
    
    def resolve_constructor_dependencies(self, constructor: Callable, 
                                       explicit_args: Dict[str, Any] = None) -> Dict[str, Any]:
        """Resolve constructor dependencies automatically."""
        explicit_args = explicit_args or {}
        signature = inspect.signature(constructor)
        resolved_args = {}
        
        for param_name, param in signature.parameters.items():
            if param_name in explicit_args:
                resolved_args[param_name] = explicit_args[param_name]
            elif param.annotation != inspect.Parameter.empty:
                # Try to resolve by type
                component = self.registry.get_component_by_type(param.annotation)
                if component:
                    resolved_args[param_name] = component
                elif param.default != inspect.Parameter.empty:
                    resolved_args[param_name] = param.default
        
        return resolved_args


class ComponentFactory:
    """Factory for creating component instances."""
    
    def __init__(self, registry):
        self.registry = registry
        self.factories: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_factory(self, name: str, factory_func: Callable):
        """Register a factory function for a component type."""
        self.factories[name] = factory_func
    
    def create_component(self, name: str, component_type: Type, 
                        factory: Callable = None, **kwargs) -> Any:
        """Create a component instance using factory or constructor."""
        try:
            if factory:
                # Use provided factory
                return factory(**kwargs)
            elif name in self.factories:
                # Use registered factory
                return self.factories[name](**kwargs)
            else:
                # Use constructor with dependency injection
                injector = DependencyInjector(self.registry)
                resolved_args = injector.resolve_constructor_dependencies(
                    component_type, kwargs
                )
                return component_type(**resolved_args)
                
        except Exception as e:
            self.logger.error(f"Error creating component {name}: {e}")
            raise


class ServiceContainer:
    """Service container for managing component lifecycles."""
    
    def __init__(self, registry):
        self.registry = registry
        self.starting_components: set = set()
        self.stopping_components: set = set()
        self.logger = logging.getLogger(__name__)
    
    def start_component(self, name: str) -> bool:
        """Start a component and its dependencies."""
        if name in self.starting_components:
            return True  # Already starting
        
        try:
            self.starting_components.add(name)
            component_info = self.registry.components.get(name)
            
            if not component_info:
                self.logger.error(f"Component {name} not found")
                return False
            
            # Start dependencies first
            for dep_name in component_info.dependencies:
                if not self.start_component(dep_name):
                    self.logger.error(f"Failed to start dependency {dep_name} for {name}")
                    return False
            
            # Call start lifecycle hook if available
            if 'start' in component_info.lifecycle_hooks:
                component_info.lifecycle_hooks['start']()
            elif hasattr(component_info.instance, 'start'):
                component_info.instance.start()
            
            self.logger.info(f"Component {name} started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting component {name}: {e}")
            return False
        finally:
            self.starting_components.discard(name)
    
    def stop_component(self, name: str) -> bool:
        """Stop a component."""
        if name in self.stopping_components:
            return True  # Already stopping
        
        try:
            self.stopping_components.add(name)
            component_info = self.registry.components.get(name)
            
            if not component_info:
                self.logger.warning(f"Component {name} not found for stopping")
                return True
            
            # Call stop lifecycle hook if available
            if 'stop' in component_info.lifecycle_hooks:
                component_info.lifecycle_hooks['stop']()
            elif hasattr(component_info.instance, 'stop'):
                component_info.instance.stop()
            
            self.logger.info(f"Component {name} stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping component {name}: {e}")
            return False
        finally:
            self.stopping_components.discard(name)


class ComponentRegistry:
    """
    Central registry for application components with dependency injection.
    
    Manages component registration, creation, lifecycle, and dependency resolution.
    """
    
    def __init__(self):
        self.components: Dict[str, ComponentInfo] = {}
        self.type_mappings: Dict[Type, str] = {}
        self.singletons: Dict[str, Any] = {}
        
        self.dependency_injector = DependencyInjector(self)
        self.component_factory = ComponentFactory(self)
        self.service_container = ServiceContainer(self)
        
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def register_component(self, name: str, component: Union[Any, Type, Callable],
                          singleton: bool = True, dependencies: List[str] = None,
                          lifecycle_hooks: Dict[str, Callable] = None,
                          metadata: Dict[str, Any] = None) -> bool:
        """
        Register a component in the registry.
        
        Args:
            name: Component name
            component: Component instance, type, or factory function
            singleton: Whether to use singleton pattern
            dependencies: List of dependency names
            lifecycle_hooks: Lifecycle hook functions
            metadata: Additional component metadata
            
        Returns:
            bool: True if registration successful
        """
        try:
            with self._lock:
                # Determine component type and instance
                if inspect.isclass(component):
                    # Component is a class type
                    component_type = component
                    instance = None
                    factory = None
                elif callable(component):
                    # Component is a factory function
                    component_type = type(component)
                    instance = None
                    factory = component
                else:
                    # Component is an instance
                    component_type = type(component)
                    instance = component
                    factory = None
                
                # Create component info
                component_info = ComponentInfo(
                    name=name,
                    instance=instance,
                    component_type=component_type,
                    singleton=singleton,
                    factory=factory,
                    dependencies=dependencies or [],
                    lifecycle_hooks=lifecycle_hooks or {},
                    metadata=metadata or {}
                )
                
                # Register component
                self.components[name] = component_info
                self.type_mappings[component_type] = name
                
                # If instance provided and singleton, store it
                if instance and singleton:
                    self.singletons[name] = instance
                
                self.logger.info(f"Component {name} registered successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Error registering component {name}: {e}")
            return False\n    \n    def unregister_component(self, name: str) -> bool:\n        \"\"\"Unregister a component from the registry.\"\"\"\n        try:\n            with self._lock:\n                if name not in self.components:\n                    self.logger.warning(f\"Component {name} not found for unregistration\")\n                    return False\n                \n                component_info = self.components[name]\n                \n                # Stop component if it has a stop method\n                self.service_container.stop_component(name)\n                \n                # Remove from registrations\n                del self.components[name]\n                if component_info.component_type in self.type_mappings:\n                    del self.type_mappings[component_info.component_type]\n                self.singletons.pop(name, None)\n                \n                self.logger.info(f\"Component {name} unregistered successfully\")\n                return True\n                \n        except Exception as e:\n            self.logger.error(f\"Error unregistering component {name}: {e}\")\n            return False\n    \n    def get_component(self, name: str) -> Optional[Any]:\n        \"\"\"Get a component instance by name.\"\"\"\n        try:\n            with self._lock:\n                if name not in self.components:\n                    return None\n                \n                component_info = self.components[name]\n                \n                # Return singleton if available\n                if component_info.singleton and name in self.singletons:\n                    return self.singletons[name]\n                \n                # Create new instance\n                if component_info.instance:\n                    instance = component_info.instance\n                else:\n                    instance = self.component_factory.create_component(\n                        name, component_info.component_type, component_info.factory\n                    )\n                \n                # Inject dependencies\n                instance = self.dependency_injector.inject_dependencies(instance, component_info)\n                \n                # Store as singleton if configured\n                if component_info.singleton:\n                    self.singletons[name] = instance\n                \n                return instance\n                \n        except Exception as e:\n            self.logger.error(f\"Error getting component {name}: {e}\")\n            return None\n    \n    def get_component_by_type(self, component_type: Type) -> Optional[Any]:\n        \"\"\"Get a component instance by type.\"\"\"\n        if component_type in self.type_mappings:\n            name = self.type_mappings[component_type]\n            return self.get_component(name)\n        return None\n    \n    def has_component(self, name: str) -> bool:\n        \"\"\"Check if a component is registered.\"\"\"\n        return name in self.components\n    \n    def list_components(self) -> List[str]:\n        \"\"\"Get list of registered component names.\"\"\"\n        with self._lock:\n            return list(self.components.keys())\n    \n    def get_component_info(self, name: str) -> Optional[ComponentInfo]:\n        \"\"\"Get component information.\"\"\"\n        return self.components.get(name)\n    \n    def start_all_components(self) -> bool:\n        \"\"\"Start all registered components.\"\"\"\n        try:\n            with self._lock:\n                component_names = list(self.components.keys())\n            \n            for name in component_names:\n                if not self.service_container.start_component(name):\n                    self.logger.error(f\"Failed to start component {name}\")\n                    return False\n            \n            self.logger.info(\"All components started successfully\")\n            return True\n            \n        except Exception as e:\n            self.logger.error(f\"Error starting all components: {e}\")\n            return False\n    \n    def stop_all_components(self) -> bool:\n        \"\"\"Stop all registered components.\"\"\"\n        try:\n            with self._lock:\n                component_names = list(self.components.keys())\n            \n            # Stop components in reverse order\n            for name in reversed(component_names):\n                self.service_container.stop_component(name)\n            \n            self.logger.info(\"All components stopped\")\n            return True\n            \n        except Exception as e:\n            self.logger.error(f\"Error stopping all components: {e}\")\n            return False\n    \n    def get_dependency_graph(self) -> Dict[str, List[str]]:\n        \"\"\"Get component dependency graph.\"\"\"\n        with self._lock:\n            return {\n                name: info.dependencies.copy()\n                for name, info in self.components.items()\n            }\n    \n    def validate_dependencies(self) -> List[str]:\n        \"\"\"Validate all component dependencies.\"\"\"\n        issues = []\n        \n        with self._lock:\n            for name, component_info in self.components.items():\n                for dep_name in component_info.dependencies:\n                    if dep_name not in self.components:\n                        issues.append(f\"Component {name} depends on unregistered component {dep_name}\")\n        \n        return issues\n\n\n# Global component registry\n_global_registry: Optional[ComponentRegistry] = None\n_registry_lock = threading.Lock()\n\n\ndef get_global_registry() -> ComponentRegistry:\n    \"\"\"Get or create the global component registry.\"\"\"\n    global _global_registry\n    \n    with _registry_lock:\n        if _global_registry is None:\n            _global_registry = ComponentRegistry()\n        return _global_registry\n\n\ndef set_global_registry(registry: ComponentRegistry):\n    \"\"\"Set the global component registry.\"\"\"\n    global _global_registry\n    \n    with _registry_lock:\n        _global_registry = registry\n\n\n# Convenience functions using global registry\ndef register_component(name: str, component: Union[Any, Type, Callable],\n                      singleton: bool = True, dependencies: List[str] = None,\n                      lifecycle_hooks: Dict[str, Callable] = None,\n                      metadata: Dict[str, Any] = None) -> bool:\n    \"\"\"Register a component using the global registry.\"\"\"\n    return get_global_registry().register_component(\n        name, component, singleton, dependencies, lifecycle_hooks, metadata\n    )\n\n\ndef get_component(name: str) -> Optional[Any]:\n    \"\"\"Get a component using the global registry.\"\"\"\n    return get_global_registry().get_component(name)\n\n\ndef inject_dependencies(instance: Any) -> Any:\n    \"\"\"Inject dependencies into an instance using the global registry.\"\"\"\n    registry = get_global_registry()\n    # Create a temporary component info for injection\n    component_info = ComponentInfo(\n        name=\"temp\",\n        instance=instance,\n        component_type=type(instance)\n    )\n    return registry.dependency_injector.inject_dependencies(instance, component_info)\n\n\ndef create_component(name: str, component_type: Type, **kwargs) -> Any:\n    \"\"\"Create a component using the global registry factory.\"\"\"\n    registry = get_global_registry()\n    return registry.component_factory.create_component(name, component_type, **kwargs)\n\n\n# Decorator for automatic registration\ndef component(name: str = None, singleton: bool = True, dependencies: List[str] = None):\n    \"\"\"Decorator for automatic component registration.\"\"\"\n    def decorator(cls):\n        component_name = name or cls.__name__.lower()\n        register_component(component_name, cls, singleton, dependencies)\n        return cls\n    \n    return decorator\n\n\n# Decorator for dependency injection\ndef inject(*dependency_names):\n    \"\"\"Decorator for automatic dependency injection.\"\"\"\n    def decorator(cls):\n        original_init = cls.__init__\n        \n        def new_init(self, *args, **kwargs):\n            original_init(self, *args, **kwargs)\n            inject_dependencies(self)\n        \n        cls.__init__ = new_init\n        return cls\n    \n    return decorator