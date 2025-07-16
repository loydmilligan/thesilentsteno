"""
Structured Logging System

Comprehensive logging system with structured output, multiple handlers,
automattic rotation, and configurable output destinations.
"""

import json
import logging
import logging.handlers
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union
import traceback
from datetime import datetime


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogFormat(Enum):
    """Log format types."""
    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"
    STRUCTURED = "structured"


@dataclass
class LogConfig:
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.STRUCTURED
    enable_console: bool = True
    enable_file: bool = True
    file_path: str = "logs/silentst.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_rotation: bool = True
    include_caller: bool = True
    include_timestamp: bool = True
    include_thread: bool = True
    extra_fields: Dict[str, Any] = field(default_factory=dict)


class StructuredFormatter(logging.Formatter):
    """Structured log formatter with JSON output."""
    
    def __init__(self, include_caller=True, include_thread=True, extra_fields=None):
        super().__init__()
        self.include_caller = include_caller
        self.include_thread = include_thread
        self.extra_fields = extra_fields or {}
        self.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
    
    def format(self, record):
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'hostname': self.hostname,
            'process': os.getpid()
        }
        
        # Add thread information
        if self.include_thread:
            log_entry['thread'] = {
                'id': threading.get_ident(),
                'name': threading.current_thread().name
            }
        
        # Add caller information
        if self.include_caller:
            log_entry['caller'] = {
                'filename': record.filename,
                'function': record.funcName,
                'line': record.lineno,
                'module': record.module
            }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                log_entry[key] = value
        
        # Add configured extra fields
        log_entry.update(self.extra_fields)
        
        return json.dumps(log_entry, default=str)


class DetailedFormatter(logging.Formatter):
    """Detailed human-readable log formatter."""
    
    def __init__(self, include_caller=True, include_thread=True):
        format_string = '%(asctime)s [%(levelname)8s]'
        
        if include_thread:
            format_string += ' [%(threadName)s]'
        
        if include_caller:
            format_string += ' %(name)s:%(funcName)s:%(lineno)d'
        else:
            format_string += ' %(name)s'
        
        format_string += ' - %(message)s'
        
        super().__init__(format_string, datefmt='%Y-%m-%d %H:%M:%S')


class LogRotator:
    """Log file rotation manager."""
    
    def __init__(self, file_path: str, max_size: int, backup_count: int):
        self.file_path = Path(file_path)
        self.max_size = max_size
        self.backup_count = backup_count
        
        # Create directory if it doesn't exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def should_rotate(self) -> bool:
        """Check if log file should be rotated."""
        if not self.file_path.exists():
            return False
        
        return self.file_path.stat().st_size >= self.max_size
    
    def rotate(self):
        """Rotate log file."""
        if not self.file_path.exists():
            return
        
        # Move existing backup files
        for i in range(self.backup_count - 1, 0, -1):
            old_file = self.file_path.with_suffix(f'.{i}')
            new_file = self.file_path.with_suffix(f'.{i + 1}')
            
            if old_file.exists():
                if new_file.exists():
                    new_file.unlink()
                old_file.rename(new_file)
        
        # Move current log to .1
        backup_file = self.file_path.with_suffix('.1')
        if backup_file.exists():
            backup_file.unlink()
        self.file_path.rename(backup_file)


class StructuredLogger:
    """Enhanced logger with structured logging capabilities."""
    
    def __init__(self, name: str, config: LogConfig = None):
        self.name = name
        self.config = config or LogConfig()
        self.logger = logging.getLogger(name)
        self._context: Dict[str, Any] = {}
        
        # Set logger level
        self.logger.setLevel(self.config.level.value)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Add configured handlers
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup log handlers based on configuration."""
        # Console handler
        if self.config.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.config.level.value)
            console_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.enable_file:
            file_path = Path(self.config.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.config.enable_rotation:
                file_handler = logging.handlers.RotatingFileHandler(
                    file_path,
                    maxBytes=self.config.max_file_size,
                    backupCount=self.config.backup_count
                )
            else:
                file_handler = logging.FileHandler(file_path)
            
            file_handler.setLevel(self.config.level.value)
            file_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(file_handler)
    
    def _get_formatter(self) -> logging.Formatter:
        """Get appropriate formatter based on configuration."""
        if self.config.format == LogFormat.JSON or self.config.format == LogFormat.STRUCTURED:
            return StructuredFormatter(
                include_caller=self.config.include_caller,
                include_thread=self.config.include_thread,
                extra_fields=self.config.extra_fields
            )
        elif self.config.format == LogFormat.DETAILED:
            return DetailedFormatter(
                include_caller=self.config.include_caller,
                include_thread=self.config.include_thread
            )
        else:  # SIMPLE
            return logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    
    def with_context(self, **context) -> 'StructuredLogger':
        """Create logger with additional context."""
        new_logger = StructuredLogger(self.name, self.config)
        new_logger._context = {**self._context, **context}
        return new_logger
    
    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """Log message with context."""
        # Add context to extra fields
        extra = kwargs.get('extra', {})
        extra.update(self._context)
        kwargs['extra'] = extra
        
        self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        self._log_with_context(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message."""
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        """Log exception with traceback."""
        kwargs['exc_info'] = True
        self.error(msg, *args, **kwargs)


class LogManager:
    """Central log management system."""
    
    def __init__(self, config: LogConfig = None):
        self.config = config or LogConfig()
        self.loggers: Dict[str, StructuredLogger] = {}
        self._lock = threading.RLock()
        
        # Setup root logger
        self._setup_root_logger()
    
    def _setup_root_logger(self):
        """Setup root logger configuration."""
        root_logger = logging.getLogger()
        
        # Handle level configuration (string or enum)
        if hasattr(self.config.level, 'value'):
            level = self.config.level.value
        else:
            level = getattr(logging, self.config.level.upper(), logging.INFO)
        
        root_logger.setLevel(level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add null handler to prevent "No handlers found" warnings
        root_logger.addHandler(logging.NullHandler())
    
    def get_logger(self, name: str, config: LogConfig = None) -> StructuredLogger:
        """Get or create a structured logger."""
        with self._lock:
            if name not in self.loggers:
                logger_config = config or self.config
                self.loggers[name] = StructuredLogger(name, logger_config)
            return self.loggers[name]
    
    def configure_logger(self, name: str, config: LogConfig):
        """Configure a specific logger."""
        with self._lock:
            self.loggers[name] = StructuredLogger(name, config)
    
    def add_handler(self, handler: logging.Handler, logger_name: str = None):
        """Add handler to logger or root logger."""
        if logger_name:
            logger = self.get_logger(logger_name)
            logger.logger.addHandler(handler)
        else:
            logging.getLogger().addHandler(handler)
    
    def set_level(self, level: LogLevel, logger_name: str = None):
        """Set log level for logger or all loggers."""
        if logger_name:
            logger = self.get_logger(logger_name)
            logger.logger.setLevel(level.value)
            logger.config.level = level
        else:
            # Set for all loggers
            self.config.level = level
            logging.getLogger().setLevel(level.value)
            
            with self._lock:
                for logger in self.loggers.values():
                    logger.logger.setLevel(level.value)
                    logger.config.level = level
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            'logger_count': len(self.loggers),
            'loggers': list(self.loggers.keys()),
            'root_level': logging.getLogger().level,
            'config': {
                'level': self.config.level.name,
                'format': self.config.format.value,
                'console_enabled': self.config.enable_console,
                'file_enabled': self.config.enable_file,
                'file_path': self.config.file_path
            }
        }


# Global log manager
_global_log_manager: Optional[LogManager] = None
_log_lock = threading.Lock()


def setup_logging(config: Union[LogConfig, Dict[str, Any]] = None) -> LogManager:
    """Setup global logging configuration."""
    global _global_log_manager
    
    with _log_lock:
        if isinstance(config, dict):
            config = LogConfig(**config)
        elif config is None:
            config = LogConfig()
        
        _global_log_manager = LogManager(config)
        return _global_log_manager


def get_logger(name: str = None, config: LogConfig = None) -> StructuredLogger:
    """Get a structured logger."""
    global _global_log_manager
    
    with _log_lock:
        if _global_log_manager is None:
            setup_logging()
        
        if name is None:
            name = __name__
        
        return _global_log_manager.get_logger(name, config)


def configure_log_rotation(file_path: str, max_size: int, backup_count: int) -> LogRotator:
    """Configure log rotation."""
    return LogRotator(file_path, max_size, backup_count)


def add_log_handler(handler: logging.Handler, logger_name: str = None):
    """Add log handler to logger."""
    global _global_log_manager
    
    with _log_lock:
        if _global_log_manager is None:
            setup_logging()
        
        _global_log_manager.add_handler(handler, logger_name)
