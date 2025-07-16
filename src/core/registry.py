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


class ComponentRegistry:
    """
    Central registry for application components with dependency injection.
    
    Manages component registration, creation, lifecycle, and dependency resolution.
    """
    
    def __init__(self):
        self.components: Dict[str, ComponentInfo] = {}
        self.type_mappings: Dict[Type, str] = {}
        self.singletons: Dict[str, Any] = {}
        
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def register_component(self, name: str, component: Union[Any, Type, Callable],
                          singleton: bool = True, dependencies: List[str] = None,
                          lifecycle_hooks: Dict[str, Callable] = None,
                          metadata: Dict[str, Any] = None) -> bool:
        """Register a component in the registry."""
        try:
            with self._lock:
                # Determine component type and instance
                if inspect.isclass(component):
                    component_type = component
                    instance = None
                    factory = None
                elif callable(component):
                    component_type = type(component)
                    instance = None
                    factory = component
                else:
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
            return False
    
    def get_component(self, name: str) -> Optional[Any]:
        """Get a component instance by name."""
        try:
            with self._lock:
                if name not in self.components:
                    return None
                
                component_info = self.components[name]
                
                # Return singleton if available
                if component_info.singleton and name in self.singletons:
                    return self.singletons[name]
                
                # Create new instance
                if component_info.instance:
                    instance = component_info.instance
                else:
                    if component_info.factory:
                        instance = component_info.factory()
                    else:
                        instance = component_info.component_type()
                
                # Store as singleton if configured
                if component_info.singleton:
                    self.singletons[name] = instance
                
                return instance
                
        except Exception as e:
            self.logger.error(f"Error getting component {name}: {e}")
            return None


# Global component registry
_global_registry: Optional[ComponentRegistry] = None
_registry_lock = threading.Lock()


def get_global_registry() -> ComponentRegistry:
    """Get or create the global component registry."""
    global _global_registry
    
    with _registry_lock:
        if _global_registry is None:
            _global_registry = ComponentRegistry()
        return _global_registry


def register_component(name: str, component: Union[Any, Type, Callable],
                      singleton: bool = True, dependencies: List[str] = None,
                      lifecycle_hooks: Dict[str, Callable] = None,
                      metadata: Dict[str, Any] = None) -> bool:
    """Register a component using the global registry."""
    return get_global_registry().register_component(
        name, component, singleton, dependencies, lifecycle_hooks, metadata
    )


def get_component(name: str) -> Optional[Any]:
    """Get a component using the global registry."""
    return get_global_registry().get_component(name)


def inject_dependencies(instance: Any) -> Any:
    """Inject dependencies into an instance using the global registry."""
    return instance  # Simplified for now


def create_component(name: str, component_type: Type, **kwargs) -> Any:
    """Create a component using the global registry factory."""
    return component_type(**kwargs)