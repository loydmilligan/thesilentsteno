"""
Configuration Management System

Hierarchical configuration system with validation, hot-reload capabilities,
and environment-specific settings management.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Set, Union
import copy
import hashlib

try:
    from pydantic import BaseModel, ValidationError, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    ValidationError = Exception

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object


class ConfigValidationError(Exception):
    """Configuration validation error."""
    pass


class ConfigSchema(BaseModel if PYDANTIC_AVAILABLE else object):
    """Base configuration schema using Pydantic if available."""
    
    if not PYDANTIC_AVAILABLE:
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)
        
        def dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class ConfigSource:
    """Configuration source information."""
    path: str
    priority: int = 50  # Lower numbers have higher priority
    watch: bool = True
    required: bool = False
    last_modified: Optional[float] = None
    hash: Optional[str] = None


class ConfigValidator:
    """Configuration validation system."""
    
    def __init__(self):
        self.validators: Dict[str, List[Callable]] = {}
        self.schemas: Dict[str, type] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_validator(self, key_path: str, validator_func: Callable):
        """Register a validator for a configuration key path."""
        if key_path not in self.validators:
            self.validators[key_path] = []
        self.validators[key_path].append(validator_func)
    
    def register_schema(self, key_path: str, schema_class: type):
        """Register a Pydantic schema for a configuration section."""
        if PYDANTIC_AVAILABLE and issubclass(schema_class, BaseModel):
            self.schemas[key_path] = schema_class
        else:
            self.logger.warning(f"Cannot register schema for {key_path}: Pydantic not available")
    
    def validate_config(self, config: Dict[str, Any], key_path: str = "") -> bool:
        """Validate configuration against registered validators and schemas."""
        try:
            # Validate with schemas
            for schema_path, schema_class in self.schemas.items():
                if key_path == schema_path or (not key_path and schema_path in config):
                    config_section = config.get(schema_path, {}) if not key_path else config
                    try:
                        schema_class(**config_section)
                    except ValidationError as e:
                        raise ConfigValidationError(f"Schema validation failed for {schema_path}: {e}")
            
            # Validate with custom validators
            for validator_path, validators in self.validators.items():
                if key_path == validator_path or (not key_path and validator_path in config):
                    config_value = config.get(validator_path) if not key_path else config
                    for validator in validators:
                        if not validator(config_value):
                            raise ConfigValidationError(f"Validation failed for {validator_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            raise ConfigValidationError(str(e))


class ConfigWatcher(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    """File system watcher for configuration changes."""
    
    def __init__(self, config_manager):
        if WATCHDOG_AVAILABLE:
            super().__init__()
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and event.src_path in self.config_manager._watched_files:
            self.logger.info(f"Configuration file modified: {event.src_path}")
            self.config_manager._reload_config_file(event.src_path)


class ConfigManager:
    """
    Hierarchical configuration management system.
    
    Supports multiple configuration sources, validation, hot-reload,
    and environment-specific overrides.
    """
    
    def __init__(self, auto_reload: bool = True):
        self.config: Dict[str, Any] = {}
        self.sources: List[ConfigSource] = []
        self.validator = ConfigValidator()
        self.auto_reload = auto_reload
        
        self._lock = threading.RLock()
        self._change_callbacks: List[Callable] = []
        self._watched_files: Set[str] = set()
        self._observer = None
        self._watcher = None
        
        self.logger = logging.getLogger(__name__)
        
        if WATCHDOG_AVAILABLE and auto_reload:
            self._setup_file_watcher()
    
    def add_source(self, path: str, priority: int = 50, watch: bool = True,
                  required: bool = False) -> bool:
        """
        Add a configuration source.
        
        Args:
            path: Path to configuration file
            priority: Source priority (lower = higher priority)
            watch: Whether to watch for changes
            required: Whether source is required
            
        Returns:
            bool: True if source was added successfully
        """
        try:
            config_path = Path(path).resolve()
            
            # Check if file exists for required sources
            if required and not config_path.exists():
                raise FileNotFoundError(f"Required configuration file not found: {path}")
            
            source = ConfigSource(
                path=str(config_path),
                priority=priority,
                watch=watch,
                required=required
            )
            
            with self._lock:
                # Remove existing source with same path
                self.sources = [s for s in self.sources if s.path != str(config_path)]
                
                # Add new source
                self.sources.append(source)
                
                # Sort by priority
                self.sources.sort(key=lambda s: s.priority)
                
                # Add to watched files if watching enabled
                if watch and self._observer:
                    self._watched_files.add(str(config_path))
                    self._observer.schedule(self._watcher, str(config_path.parent), recursive=False)
            
            # Load the configuration
            self._reload_config()
            
            self.logger.info(f"Added configuration source: {path} (priority: {priority})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add configuration source {path}: {e}")
            return False
    
    def remove_source(self, path: str) -> bool:
        """Remove a configuration source."""
        try:
            config_path = str(Path(path).resolve())
            
            with self._lock:
                # Remove source
                initial_count = len(self.sources)
                self.sources = [s for s in self.sources if s.path != config_path]
                
                # Remove from watched files
                self._watched_files.discard(config_path)
                
                if len(self.sources) < initial_count:
                    # Reload configuration without this source
                    self._reload_config()
                    self.logger.info(f"Removed configuration source: {path}")
                    return True
                else:
                    self.logger.warning(f"Configuration source not found: {path}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to remove configuration source {path}: {e}")
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by key path.
        
        Args:
            key_path: Dot-separated key path (e.g., 'database.host')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        with self._lock:
            try:
                keys = key_path.split('.')
                value = self.config
                
                for key in keys:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        return default
                
                return value
                
            except Exception as e:
                self.logger.error(f"Error getting configuration key {key_path}: {e}")
                return default
    
    def set(self, key_path: str, value: Any, validate: bool = True) -> bool:
        """
        Set configuration value by key path.
        
        Args:
            key_path: Dot-separated key path
            value: Value to set
            validate: Whether to validate the change
            
        Returns:
            bool: True if value was set successfully
        """
        try:
            with self._lock:
                keys = key_path.split('.')
                config_ref = self.config
                
                # Navigate to parent key
                for key in keys[:-1]:
                    if key not in config_ref:
                        config_ref[key] = {}
                    elif not isinstance(config_ref[key], dict):
                        raise ValueError(f"Cannot set nested key on non-dict value at {key}")
                    config_ref = config_ref[key]
                
                # Set final value
                old_value = config_ref.get(keys[-1])
                config_ref[keys[-1]] = value
                
                # Validate if requested
                if validate:
                    try:
                        self.validator.validate_config(self.config)
                    except ConfigValidationError:
                        # Revert change on validation failure
                        if old_value is not None:
                            config_ref[keys[-1]] = old_value
                        else:
                            config_ref.pop(keys[-1], None)
                        raise
                
                # Notify callbacks
                self._notify_change_callbacks(key_path, value, old_value)
                
                self.logger.debug(f"Set configuration {key_path} = {value}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to set configuration {key_path}: {e}")
            return False
    
    def update(self, config_dict: Dict[str, Any], validate: bool = True) -> bool:
        """
        Update configuration with a dictionary.
        
        Args:
            config_dict: Configuration dictionary to merge
            validate: Whether to validate changes
            
        Returns:
            bool: True if update was successful
        """
        try:
            with self._lock:
                old_config = copy.deepcopy(self.config)
                
                # Deep merge configuration
                self._deep_merge(self.config, config_dict)
                
                # Validate if requested
                if validate:
                    try:
                        self.validator.validate_config(self.config)
                    except ConfigValidationError:
                        # Revert changes on validation failure
                        self.config = old_config
                        raise
                
                # Notify callbacks
                self._notify_change_callbacks("*", config_dict, old_config)
                
                self.logger.info("Configuration updated successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False
    
    def register_change_callback(self, callback: Callable[[str, Any, Any], None]):
        """Register a callback for configuration changes."""
        with self._lock:
            self._change_callbacks.append(callback)
    
    def unregister_change_callback(self, callback: Callable):
        """Unregister a configuration change callback."""
        with self._lock:
            if callback in self._change_callbacks:
                self._change_callbacks.remove(callback)
    
    def save_to_file(self, path: str, section: str = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            path: File path to save to
            section: Optional configuration section to save
            
        Returns:
            bool: True if saved successfully
        """
        try:
            config_to_save = self.config
            if section:
                config_to_save = self.get(section, {})
            
            with open(path, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            
            self.logger.info(f"Configuration saved to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {path}: {e}")
            return False
    
    def reload(self) -> bool:
        """Reload all configuration sources."""
        return self._reload_config()
    
    def get_all(self) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
        with self._lock:
            return copy.deepcopy(self.config)
    
    def get_sources(self) -> List[ConfigSource]:
        """Get list of configuration sources."""
        with self._lock:
            return copy.deepcopy(self.sources)
    
    def _setup_file_watcher(self):
        """Set up file system watcher for configuration changes."""
        if not WATCHDOG_AVAILABLE:
            self.logger.warning("Watchdog not available, file watching disabled")
            return
        
        try:
            self._watcher = ConfigWatcher(self)
            self._observer = Observer()
            self._observer.start()
            self.logger.info("Configuration file watcher started")
        except Exception as e:
            self.logger.error(f"Failed to setup file watcher: {e}")
    
    def _reload_config(self) -> bool:
        """Reload configuration from all sources."""
        try:
            with self._lock:
                new_config = {}
                
                # Load from sources in priority order (lower priority numbers first)
                for source in sorted(self.sources, key=lambda s: s.priority):
                    if Path(source.path).exists():
                        source_config = self._load_config_file(source.path)
                        if source_config is not None:
                            self._deep_merge(new_config, source_config)
                            source.last_modified = Path(source.path).stat().st_mtime
                            source.hash = self._calculate_file_hash(source.path)
                    elif source.required:
                        raise FileNotFoundError(f"Required configuration file not found: {source.path}")
                
                # Apply environment variable overrides
                self._apply_env_overrides(new_config)
                
                # Validate configuration
                self.validator.validate_config(new_config)
                
                # Update configuration
                old_config = self.config
                self.config = new_config
                
                # Notify callbacks
                self._notify_change_callbacks("*", new_config, old_config)
                
                self.logger.info("Configuration reloaded successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def _reload_config_file(self, file_path: str):
        """Reload a specific configuration file."""
        try:
            # Find source for this file
            source = None
            for s in self.sources:
                if s.path == file_path:
                    source = s
                    break
            
            if not source:
                return
            
            # Check if file actually changed
            current_hash = self._calculate_file_hash(file_path)
            if current_hash == source.hash:
                return
            
            # Reload all configuration (to maintain proper merging)
            self._reload_config()
            
        except Exception as e:
            self.logger.error(f"Failed to reload config file {file_path}: {e}")
    
    def _load_config_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load configuration from a file."""
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.json'):
                    return json.load(f)
                else:
                    # Try JSON as default
                    return json.load(f)
                    
        except Exception as e:
            self.logger.error(f"Failed to load config file {file_path}: {e}")
            return None
    
    def _apply_env_overrides(self, config: Dict[str, Any]):
        """Apply environment variable overrides."""
        # Look for environment variables with SILENTST_ prefix
        for key, value in os.environ.items():
            if key.startswith('SILENTST_'):
                config_key = key[9:].lower().replace('_', '.')
                try:
                    # Try to parse as JSON, fall back to string
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        parsed_value = value
                    
                    self._set_nested_key(config, config_key, parsed_value)
                    
                except Exception as e:
                    self.logger.error(f"Failed to apply environment override {key}: {e}")
    
    def _set_nested_key(self, config: Dict[str, Any], key_path: str, value: Any):
        """Set a nested key in configuration dictionary."""
        keys = key_path.split('.')
        config_ref = config
        
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        config_ref[keys[-1]] = value
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Deep merge source dictionary into target dictionary."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of a file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def _notify_change_callbacks(self, key_path: str, new_value: Any, old_value: Any):
        """Notify all registered change callbacks."""
        for callback in self._change_callbacks:
            try:
                callback(key_path, new_value, old_value)
            except Exception as e:
                self.logger.error(f"Error in configuration change callback: {e}")


# Global configuration manager
_global_config_manager: Optional[ConfigManager] = None
_config_lock = threading.Lock()


def get_global_config_manager() -> ConfigManager:
    """Get or create the global configuration manager."""
    global _global_config_manager
    
    with _config_lock:
        if _global_config_manager is None:
            _global_config_manager = ConfigManager()
        return _global_config_manager


def set_global_config_manager(config_manager: ConfigManager):
    """Set the global configuration manager."""
    global _global_config_manager
    
    with _config_lock:
        _global_config_manager = config_manager


# Convenience functions using global config manager
def load_config(path: str, priority: int = 50, watch: bool = True, required: bool = False) -> Dict[str, Any]:
    """Load configuration from file using global config manager."""
    manager = get_global_config_manager()
    manager.add_source(path, priority, watch, required)
    return manager.get_all()


def save_config(path: str, section: str = None) -> bool:
    """Save configuration to file using global config manager."""
    return get_global_config_manager().save_to_file(path, section)


def get_config(key_path: str, default: Any = None) -> Any:
    """Get configuration value using global config manager."""
    return get_global_config_manager().get(key_path, default)


def set_config(key_path: str, value: Any, validate: bool = True) -> bool:
    """Set configuration value using global config manager."""
    return get_global_config_manager().set(key_path, value, validate)


def validate_config(config: Dict[str, Any] = None) -> bool:
    """Validate configuration using global config manager."""
    manager = get_global_config_manager()
    config_to_validate = config or manager.get_all()
    return manager.validator.validate_config(config_to_validate)


def watch_config_changes(callback: Callable[[str, Any, Any], None]):
    """Register a callback for configuration changes."""
    get_global_config_manager().register_change_callback(callback)