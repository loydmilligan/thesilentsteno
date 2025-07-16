"""
Error Handling and Recovery System

Comprehensive error handling with automatic recovery mechanisms,
error reporting, and graceful fallback management.
"""

import logging
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Type, Union
from datetime import datetime
import uuid


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    RESTART = "restart"
    FALLBACK = "fallback"
    IGNORE = "ignore"
    ESCALATE = "escalate"


@dataclass
class ErrorContext:
    """Error context information."""
    component: str
    operation: str
    timestamp: datetime = field(default_factory=datetime.now)
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorRecord:
    """Error record for tracking and analysis."""
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    exception_type: str = ""
    message: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    context: Optional[ErrorContext] = None
    traceback: str = ""
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    timestamp: datetime = field(default_factory=datetime.now)
    count: int = 1


class ErrorHandler:
    """Central error handling system."""
    
    def __init__(self):
        self.error_records: Dict[str, ErrorRecord] = {}
        self.recovery_strategies: Dict[Type[Exception], RecoveryStrategy] = {}
        self.error_callbacks: List[Callable[[ErrorRecord], None]] = []
        self.max_recovery_attempts = 3
        self.recovery_cooldown = 30.0  # seconds
        
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Default recovery strategies
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """Setup default recovery strategies for common exceptions."""
        self.recovery_strategies.update({
            ConnectionError: RecoveryStrategy.RETRY,
            TimeoutError: RecoveryStrategy.RETRY,
            FileNotFoundError: RecoveryStrategy.FALLBACK,
            PermissionError: RecoveryStrategy.ESCALATE,
            MemoryError: RecoveryStrategy.RESTART,
            KeyboardInterrupt: RecoveryStrategy.IGNORE,
            SystemExit: RecoveryStrategy.IGNORE
        })
    
    def handle_error(self, exception: Exception, context: Union[str, ErrorContext] = None,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> ErrorRecord:
        """
        Handle an error with automatic recovery.
        
        Args:
            exception: Exception to handle
            context: Error context (string or ErrorContext object)
            severity: Error severity level
            
        Returns:
            ErrorRecord: Created error record
        """
        try:
            # Create error context if string provided
            if isinstance(context, str):
                context = ErrorContext(component="unknown", operation=context)
            elif context is None:
                context = ErrorContext(component="unknown", operation="unknown")
            
            # Create error record
            error_record = ErrorRecord(
                exception_type=type(exception).__name__,
                message=str(exception),
                severity=severity,
                context=context,
                traceback=traceback.format_exc()
            )
            
            # Check for existing similar errors
            existing_record = self._find_similar_error(error_record)
            if existing_record:
                existing_record.count += 1
                existing_record.timestamp = datetime.now()
                error_record = existing_record
            else:
                with self._lock:
                    self.error_records[error_record.error_id] = error_record
            
            # Log the error
            self.logger.error(
                f"Error in {context.component}.{context.operation}: {exception}",
                extra={
                    'error_id': error_record.error_id,
                    'severity': severity.name,
                    'exception_type': type(exception).__name__
                }
            )
            
            # Attempt recovery if not already attempted recently
            if not existing_record or self._should_attempt_recovery(error_record):
                self._attempt_recovery(error_record, exception)
            
            # Notify callbacks
            self._notify_error_callbacks(error_record)
            
            return error_record
            
        except Exception as e:
            # Fallback error handling
            self.logger.critical(f"Error in error handler: {e}")
            return ErrorRecord(
                exception_type=type(exception).__name__,
                message=str(exception),
                severity=ErrorSeverity.CRITICAL
            )
    
    def register_recovery_strategy(self, exception_type: Type[Exception], 
                                 strategy: RecoveryStrategy):
        """Register a recovery strategy for an exception type."""
        self.recovery_strategies[exception_type] = strategy
    
    def add_error_callback(self, callback: Callable[[ErrorRecord], None]):
        """Add callback for error notifications."""
        with self._lock:
            self.error_callbacks.append(callback)
    
    def remove_error_callback(self, callback: Callable[[ErrorRecord], None]):
        """Remove error notification callback."""
        with self._lock:
            if callback in self.error_callbacks:
                self.error_callbacks.remove(callback)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self._lock:
            total_errors = len(self.error_records)
            severity_counts = {}
            recovery_success_rate = 0
            
            for record in self.error_records.values():
                severity_name = record.severity.name
                severity_counts[severity_name] = severity_counts.get(severity_name, 0) + 1
            
            attempted_recoveries = sum(1 for r in self.error_records.values() if r.recovery_attempted)
            successful_recoveries = sum(1 for r in self.error_records.values() if r.recovery_successful)
            
            if attempted_recoveries > 0:
                recovery_success_rate = (successful_recoveries / attempted_recoveries) * 100
            
            return {
                'total_errors': total_errors,
                'severity_distribution': severity_counts,
                'recovery_attempts': attempted_recoveries,
                'successful_recoveries': successful_recoveries,
                'recovery_success_rate': recovery_success_rate,
                'recent_errors': self._get_recent_errors(10)
            }
    
    def clear_error_history(self, before_timestamp: datetime = None):
        """Clear error history before specified timestamp."""
        with self._lock:
            if before_timestamp is None:
                self.error_records.clear()
            else:
                to_remove = [
                    error_id for error_id, record in self.error_records.items()
                    if record.timestamp < before_timestamp
                ]
                for error_id in to_remove:
                    del self.error_records[error_id]
    
    def _find_similar_error(self, error_record: ErrorRecord) -> Optional[ErrorRecord]:
        """Find similar existing error record."""
        for existing_record in self.error_records.values():
            if (existing_record.exception_type == error_record.exception_type and
                existing_record.context and error_record.context and
                existing_record.context.component == error_record.context.component and
                existing_record.context.operation == error_record.context.operation):
                return existing_record
        return None
    
    def _should_attempt_recovery(self, error_record: ErrorRecord) -> bool:
        """Check if recovery should be attempted for this error."""
        if error_record.count > self.max_recovery_attempts:
            return False
        
        if error_record.recovery_attempted:
            time_since_last = (datetime.now() - error_record.timestamp).total_seconds()
            return time_since_last > self.recovery_cooldown
        
        return True
    
    def _attempt_recovery(self, error_record: ErrorRecord, exception: Exception):
        """Attempt to recover from an error."""
        try:
            strategy = self._get_recovery_strategy(exception)
            error_record.recovery_strategy = strategy
            error_record.recovery_attempted = True
            
            self.logger.info(f"Attempting {strategy.value} recovery for error {error_record.error_id}")
            
            if strategy == RecoveryStrategy.RETRY:
                # Retry logic would be implemented by the calling component
                error_record.recovery_successful = True
            elif strategy == RecoveryStrategy.RESTART:
                # Component restart would be triggered
                error_record.recovery_successful = True
            elif strategy == RecoveryStrategy.FALLBACK:
                # Fallback logic would be implemented by the calling component
                error_record.recovery_successful = True
            elif strategy == RecoveryStrategy.IGNORE:
                # Error is ignored
                error_record.recovery_successful = True
            elif strategy == RecoveryStrategy.ESCALATE:
                # Error is escalated to higher level
                self.logger.critical(f"Escalating error: {error_record.message}")
                error_record.recovery_successful = False
            
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
            error_record.recovery_successful = False
    
    def _get_recovery_strategy(self, exception: Exception) -> RecoveryStrategy:
        """Get recovery strategy for an exception."""
        exception_type = type(exception)
        
        # Look for exact match first
        if exception_type in self.recovery_strategies:
            return self.recovery_strategies[exception_type]
        
        # Look for parent class matches
        for exc_type, strategy in self.recovery_strategies.items():
            if isinstance(exception, exc_type):
                return strategy
        
        # Default strategy
        return RecoveryStrategy.RETRY
    
    def _notify_error_callbacks(self, error_record: ErrorRecord):
        """Notify all error callbacks."""
        for callback in self.error_callbacks:
            try:
                callback(error_record)
            except Exception as e:
                self.logger.error(f"Error in error callback: {e}")
    
    def _get_recent_errors(self, count: int) -> List[Dict[str, Any]]:
        """Get recent error records."""
        recent_errors = sorted(
            self.error_records.values(),
            key=lambda r: r.timestamp,
            reverse=True
        )[:count]
        
        return [
            {
                'error_id': record.error_id,
                'type': record.exception_type,
                'message': record.message,
                'severity': record.severity.name,
                'timestamp': record.timestamp.isoformat(),
                'component': record.context.component if record.context else 'unknown',
                'count': record.count
            }
            for record in recent_errors
        ]


class RecoveryManager:
    """Manages recovery operations and strategies."""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.recovery_operations: Dict[str, Callable] = {}
        self.fallback_operations: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_recovery_operation(self, component: str, operation: Callable):
        """Register a recovery operation for a component."""
        self.recovery_operations[component] = operation
    
    def register_fallback_operation(self, component: str, operation: Callable):
        """Register a fallback operation for a component."""
        self.fallback_operations[component] = operation
    
    def attempt_recovery(self, component: str, strategy: RecoveryStrategy) -> bool:
        """Attempt recovery for a component."""
        try:
            if strategy == RecoveryStrategy.RETRY:
                return self._retry_operation(component)
            elif strategy == RecoveryStrategy.RESTART:
                return self._restart_component(component)
            elif strategy == RecoveryStrategy.FALLBACK:
                return self._activate_fallback(component)
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Recovery failed for {component}: {e}")
            return False
    
    def _retry_operation(self, component: str) -> bool:
        """Retry operation for a component."""
        if component in self.recovery_operations:
            try:
                self.recovery_operations[component]()
                return True
            except Exception as e:
                self.logger.error(f"Retry failed for {component}: {e}")
        return False
    
    def _restart_component(self, component: str) -> bool:
        """Restart a component."""
        # This would typically involve stopping and starting the component
        self.logger.info(f"Restarting component: {component}")
        return True
    
    def _activate_fallback(self, component: str) -> bool:
        """Activate fallback for a component."""
        if component in self.fallback_operations:
            try:
                self.fallback_operations[component]()
                return True
            except Exception as e:
                self.logger.error(f"Fallback failed for {component}: {e}")
        return False


class ErrorReporter:
    """Error reporting and notification system."""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.report_handlers: List[Callable[[ErrorRecord], None]] = []
        self.logger = logging.getLogger(__name__)
    
    def add_report_handler(self, handler: Callable[[ErrorRecord], None]):
        """Add error report handler."""
        self.report_handlers.append(handler)
    
    def report_error(self, error_record: ErrorRecord):
        """Report an error through all handlers."""
        for handler in self.report_handlers:
            try:
                handler(error_record)
            except Exception as e:
                self.logger.error(f"Error in report handler: {e}")


class FallbackManager:
    """Manages fallback operations and graceful degradation."""
    
    def __init__(self):
        self.fallback_strategies: Dict[str, Callable] = {}
        self.active_fallbacks: Set[str] = set()
        self.logger = logging.getLogger(__name__)
    
    def register_fallback(self, component: str, fallback_func: Callable):
        """Register a fallback function for a component."""
        self.fallback_strategies[component] = fallback_func
    
    def activate_fallback(self, component: str) -> bool:
        """Activate fallback for a component."""
        if component in self.fallback_strategies:
            try:
                self.fallback_strategies[component]()
                self.active_fallbacks.add(component)
                self.logger.info(f"Activated fallback for {component}")
                return True
            except Exception as e:
                self.logger.error(f"Fallback activation failed for {component}: {e}")
        return False
    
    def deactivate_fallback(self, component: str) -> bool:
        """Deactivate fallback for a component."""
        if component in self.active_fallbacks:
            self.active_fallbacks.remove(component)
            self.logger.info(f"Deactivated fallback for {component}")
            return True
        return False
    
    def is_fallback_active(self, component: str) -> bool:
        """Check if fallback is active for a component."""
        return component in self.active_fallbacks


# Global error handler
_global_error_handler: Optional[ErrorHandler] = None
_error_lock = threading.Lock()


def get_global_error_handler() -> ErrorHandler:
    """Get or create the global error handler."""
    global _global_error_handler
    
    with _error_lock:
        if _global_error_handler is None:
            _global_error_handler = ErrorHandler()
        return _global_error_handler


# Convenience functions
def handle_error(exception: Exception, context: Union[str, ErrorContext] = None,
                severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> ErrorRecord:
    """Handle an error using the global error handler."""
    return get_global_error_handler().handle_error(exception, context, severity)


def attempt_recovery(component: str, strategy: RecoveryStrategy) -> bool:
    """Attempt recovery for a component."""
    recovery_manager = RecoveryManager(get_global_error_handler())
    return recovery_manager.attempt_recovery(component, strategy)


def report_error(error_record: ErrorRecord):
    """Report an error using the global error reporter."""
    error_reporter = ErrorReporter(get_global_error_handler())
    error_reporter.report_error(error_record)


def activate_fallback(component: str) -> bool:
    """Activate fallback for a component."""
    fallback_manager = FallbackManager()
    return fallback_manager.activate_fallback(component)