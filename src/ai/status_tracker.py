#!/usr/bin/env python3
"""
Status Tracker Module

Processing status tracking and error handling system for pipeline monitoring
and recovery with comprehensive health monitoring capabilities.

Author: Claude AI Assistant
Date: 2025-07-15
Version: 1.0
"""

import os
import sys
import logging
import json
import time
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
from collections import defaultdict, deque
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Processing status types"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"


class HealthStatus(Enum):
    """System health status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNAVAILABLE = "unavailable"


class AlertLevel(Enum):
    """Alert levels for notifications"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class StatusConfig:
    """Configuration for status tracking"""
    
    # Monitoring settings
    health_check_interval: float = 30.0  # seconds
    status_update_interval: float = 5.0  # seconds
    
    # History settings
    max_status_history: int = 1000
    max_error_history: int = 100
    max_performance_history: int = 500
    
    # Alert settings
    enable_alerts: bool = True
    alert_cooldown: float = 60.0  # seconds
    max_alerts_per_minute: int = 10
    
    # Health thresholds
    memory_warning_threshold: float = 0.8  # 80%
    memory_critical_threshold: float = 0.95  # 95%
    cpu_warning_threshold: float = 0.8  # 80%
    cpu_critical_threshold: float = 0.95  # 95%
    
    # Error thresholds
    error_rate_warning: float = 0.1  # 10%
    error_rate_critical: float = 0.3  # 30%
    
    # Recovery settings
    enable_auto_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_delay: float = 5.0  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "health_check_interval": self.health_check_interval,
            "status_update_interval": self.status_update_interval,
            "max_status_history": self.max_status_history,
            "max_error_history": self.max_error_history,
            "max_performance_history": self.max_performance_history,
            "enable_alerts": self.enable_alerts,
            "alert_cooldown": self.alert_cooldown,
            "max_alerts_per_minute": self.max_alerts_per_minute,
            "memory_warning_threshold": self.memory_warning_threshold,
            "memory_critical_threshold": self.memory_critical_threshold,
            "cpu_warning_threshold": self.cpu_warning_threshold,
            "cpu_critical_threshold": self.cpu_critical_threshold,
            "error_rate_warning": self.error_rate_warning,
            "error_rate_critical": self.error_rate_critical,
            "enable_auto_recovery": self.enable_auto_recovery,
            "max_recovery_attempts": self.max_recovery_attempts,
            "recovery_delay": self.recovery_delay
        }


@dataclass
class HealthCheck:
    """Health check result"""
    
    # Check metadata
    check_id: str
    timestamp: datetime
    component: str
    
    # Health status
    status: HealthStatus
    score: float  # 0.0 to 1.0
    
    # Metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    
    # Performance metrics
    response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    
    # Issues and recommendations
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Raw data
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "check_id": self.check_id,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "status": self.status.value,
            "score": self.score,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "disk_usage": self.disk_usage,
            "response_time": self.response_time,
            "throughput": self.throughput,
            "error_rate": self.error_rate,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "raw_metrics": self.raw_metrics
        }


@dataclass
class ErrorRecord:
    """Error record for tracking"""
    
    # Error metadata
    error_id: str
    timestamp: datetime
    component: str
    
    # Error details
    error_type: str
    message: str
    severity: AlertLevel
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    
    # Resolution
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    resolution_notes: str = ""
    
    # Recovery
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "error_type": self.error_type,
            "message": self.message,
            "severity": self.severity.value,
            "context": self.context,
            "stack_trace": self.stack_trace,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None,
            "resolution_notes": self.resolution_notes,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
            "recovery_notes": self.recovery_notes
        }


@dataclass
class StatusUpdate:
    """Status update record"""
    
    # Update metadata
    update_id: str
    timestamp: datetime
    component: str
    
    # Status information
    old_status: ProcessingStatus
    new_status: ProcessingStatus
    
    # Additional information
    message: str = ""
    progress: float = 0.0  # 0.0 to 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "update_id": self.update_id,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "old_status": self.old_status.value,
            "new_status": self.new_status.value,
            "message": self.message,
            "progress": self.progress,
            "details": self.details
        }


class ErrorHandler:
    """Error handling and recovery system"""
    
    def __init__(self, config: StatusConfig):
        self.config = config
        self.error_history: deque = deque(maxlen=config.max_error_history)
        self.recovery_attempts: Dict[str, int] = defaultdict(int)
        self.recovery_callbacks: Dict[str, Callable] = {}
        
    def handle_error(self, component: str, error: Exception, context: Dict[str, Any] = None) -> ErrorRecord:
        """Handle and record error"""
        try:
            error_record = ErrorRecord(
                error_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                component=component,
                error_type=type(error).__name__,
                message=str(error),
                severity=self._determine_severity(error),
                context=context or {},
                stack_trace=self._get_stack_trace(error)
            )
            
            # Add to history
            self.error_history.append(error_record)
            
            # Log error
            logger.error(f"Error in {component}: {error_record.message}")
            
            # Attempt recovery if enabled
            if self.config.enable_auto_recovery:
                self._attempt_recovery(error_record)
                
            return error_record
            
        except Exception as e:
            logger.error(f"Failed to handle error: {e}")
            return ErrorRecord(
                error_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                component=component,
                error_type="ErrorHandlingFailure",
                message=f"Failed to handle error: {str(e)}"
            )
            
    def _determine_severity(self, error: Exception) -> AlertLevel:
        """Determine error severity"""
        error_type = type(error).__name__
        
        critical_errors = ['SystemError', 'MemoryError', 'OSError']
        error_errors = ['ValueError', 'TypeError', 'AttributeError']
        
        if error_type in critical_errors:
            return AlertLevel.CRITICAL
        elif error_type in error_errors:
            return AlertLevel.ERROR
        else:
            return AlertLevel.WARNING
            
    def _get_stack_trace(self, error: Exception) -> str:
        """Get stack trace from error"""
        import traceback
        try:
            return traceback.format_exc()
        except Exception:
            return f"Stack trace unavailable for {type(error).__name__}"
            
    def _attempt_recovery(self, error_record: ErrorRecord):
        """Attempt automatic recovery"""
        try:
            component = error_record.component
            
            # Check recovery attempts
            if self.recovery_attempts[component] >= self.config.max_recovery_attempts:
                logger.warning(f"Max recovery attempts reached for {component}")
                return
                
            # Increment attempts
            self.recovery_attempts[component] += 1
            error_record.recovery_attempted = True
            
            # Wait before recovery
            time.sleep(self.config.recovery_delay)
            
            # Attempt recovery
            if component in self.recovery_callbacks:
                try:
                    success = self.recovery_callbacks[component](error_record)
                    error_record.recovery_successful = success
                    
                    if success:
                        error_record.recovery_notes = f"Recovery successful on attempt {self.recovery_attempts[component]}"
                        logger.info(f"Recovery successful for {component}")
                    else:
                        error_record.recovery_notes = f"Recovery failed on attempt {self.recovery_attempts[component]}"
                        logger.warning(f"Recovery failed for {component}")
                        
                except Exception as e:
                    error_record.recovery_notes = f"Recovery exception: {str(e)}"
                    logger.error(f"Recovery exception for {component}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to attempt recovery: {e}")
            
    def register_recovery_callback(self, component: str, callback: Callable):
        """Register recovery callback for component"""
        self.recovery_callbacks[component] = callback
        
    def resolve_error(self, error_id: str, resolution_notes: str = ""):
        """Mark error as resolved"""
        try:
            for error_record in self.error_history:
                if error_record.error_id == error_id:
                    error_record.resolved = True
                    error_record.resolution_time = datetime.now()
                    error_record.resolution_notes = resolution_notes
                    logger.info(f"Error {error_id} resolved")
                    return
                    
            logger.warning(f"Error {error_id} not found for resolution")
            
        except Exception as e:
            logger.error(f"Failed to resolve error: {e}")
            
    def get_error_rate(self, component: str = None, time_window: float = 300.0) -> float:
        """Calculate error rate for component"""
        try:
            cutoff_time = datetime.now() - timedelta(seconds=time_window)
            
            relevant_errors = [
                error for error in self.error_history
                if error.timestamp >= cutoff_time and (component is None or error.component == component)
            ]
            
            if not relevant_errors:
                return 0.0
                
            total_time = time_window
            error_count = len(relevant_errors)
            
            return error_count / total_time if total_time > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate error rate: {e}")
            return 0.0
            
    def get_error_history(self, component: str = None, limit: int = None) -> List[ErrorRecord]:
        """Get error history"""
        try:
            filtered_errors = [
                error for error in self.error_history
                if component is None or error.component == component
            ]
            
            if limit:
                filtered_errors = filtered_errors[-limit:]
                
            return filtered_errors
            
        except Exception as e:
            logger.error(f"Failed to get error history: {e}")
            return []


class StatusTracker:
    """Main status tracking system"""
    
    def __init__(self, config: Optional[StatusConfig] = None):
        self.config = config or StatusConfig()
        self.tracker_id = str(uuid.uuid4())
        
        # Status tracking
        self.current_status: Dict[str, ProcessingStatus] = defaultdict(lambda: ProcessingStatus.IDLE)
        self.status_history: deque = deque(maxlen=self.config.max_status_history)
        
        # Health monitoring
        self.health_status: Dict[str, HealthStatus] = defaultdict(lambda: HealthStatus.HEALTHY)
        self.health_history: deque = deque(maxlen=self.config.max_performance_history)
        
        # Error handling
        self.error_handler = ErrorHandler(self.config)
        
        # Monitoring
        self.is_monitoring = False
        self.monitoring_thread = None
        self.alert_queue = queue.Queue()
        self.alert_history: deque = deque(maxlen=100)
        
        # Callbacks
        self.status_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.health_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # Performance tracking
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Statistics
        self.stats = {
            "total_updates": 0,
            "total_errors": 0,
            "total_health_checks": 0,
            "uptime": 0.0,
            "error_rate": 0.0,
            "average_response_time": 0.0
        }
        
        self.start_time = datetime.now()
        
        logger.info(f"StatusTracker initialized with ID: {self.tracker_id}")
        
    def start_monitoring(self):
        """Start background monitoring"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Status monitoring started")
        
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
            
        logger.info("Status monitoring stopped")
        
    def update_status(self, component: str, new_status: ProcessingStatus, 
                     message: str = "", progress: float = 0.0, details: Dict[str, Any] = None):
        """Update component status"""
        try:
            old_status = self.current_status[component]
            
            # Create status update record
            update = StatusUpdate(
                update_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                component=component,
                old_status=old_status,
                new_status=new_status,
                message=message,
                progress=progress,
                details=details or {}
            )
            
            # Update current status
            self.current_status[component] = new_status
            
            # Add to history
            self.status_history.append(update)
            
            # Update statistics
            self.stats["total_updates"] += 1
            
            # Call callbacks
            for callback in self.status_callbacks[component]:
                try:
                    callback(update)
                except Exception as e:
                    logger.error(f"Status callback error: {e}")
                    
            # Log status change
            logger.info(f"Status update: {component} {old_status.value} -> {new_status.value}")
            
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
            
    def report_error(self, component: str, error: Exception, context: Dict[str, Any] = None) -> str:
        """Report error and get error ID"""
        try:
            error_record = self.error_handler.handle_error(component, error, context)
            
            # Update status to failed
            self.update_status(component, ProcessingStatus.FAILED, 
                             f"Error: {error_record.message}")
            
            # Update statistics
            self.stats["total_errors"] += 1
            
            # Create alert
            if self.config.enable_alerts:
                self._create_alert(AlertLevel.ERROR, f"Error in {component}: {error_record.message}")
                
            return error_record.error_id
            
        except Exception as e:
            logger.error(f"Failed to report error: {e}")
            return ""
            
    def perform_health_check(self, component: str = "system") -> HealthCheck:
        """Perform health check"""
        try:
            check = HealthCheck(
                check_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                component=component,
                status=HealthStatus.HEALTHY,
                score=1.0
            )
            
            # Get system metrics
            metrics = self._get_system_metrics()
            check.cpu_usage = metrics.get("cpu_usage", 0.0)
            check.memory_usage = metrics.get("memory_usage", 0.0)
            check.disk_usage = metrics.get("disk_usage", 0.0)
            
            # Calculate health score
            check.score = self._calculate_health_score(check)
            check.status = self._determine_health_status(check)
            
            # Check for issues
            self._check_for_issues(check)
            
            # Update health status
            self.health_status[component] = check.status
            
            # Add to history
            self.health_history.append(check)
            
            # Update statistics
            self.stats["total_health_checks"] += 1
            
            # Call callbacks
            for callback in self.health_callbacks:
                try:
                    callback(check)
                except Exception as e:
                    logger.error(f"Health callback error: {e}")
                    
            return check
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheck(
                check_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                component=component,
                status=HealthStatus.UNAVAILABLE,
                score=0.0
            )
            
    def _monitoring_loop(self):
        """Background monitoring loop"""
        last_health_check = 0
        last_status_update = 0
        
        while self.is_monitoring:
            try:
                current_time = time.time()
                
                # Health checks
                if current_time - last_health_check >= self.config.health_check_interval:
                    self.perform_health_check()
                    last_health_check = current_time
                    
                # Status updates
                if current_time - last_status_update >= self.config.status_update_interval:
                    self._update_statistics()
                    last_status_update = current_time
                    
                # Process alerts
                self._process_alerts()
                
                # Sleep
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5.0)
                
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get system metrics"""
        try:
            metrics = {}
            
            # Try to get CPU and memory usage
            try:
                import psutil
                metrics["cpu_usage"] = psutil.cpu_percent(interval=0.1) / 100.0
                metrics["memory_usage"] = psutil.virtual_memory().percent / 100.0
                metrics["disk_usage"] = psutil.disk_usage('/').percent / 100.0
            except ImportError:
                # Fallback without psutil
                metrics["cpu_usage"] = 0.5  # Placeholder
                metrics["memory_usage"] = 0.5  # Placeholder
                metrics["disk_usage"] = 0.5  # Placeholder
                
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
            
    def _calculate_health_score(self, check: HealthCheck) -> float:
        """Calculate health score"""
        try:
            score_factors = []
            
            # CPU usage factor
            if check.cpu_usage < self.config.cpu_warning_threshold:
                score_factors.append(1.0)
            elif check.cpu_usage < self.config.cpu_critical_threshold:
                score_factors.append(0.5)
            else:
                score_factors.append(0.1)
                
            # Memory usage factor
            if check.memory_usage < self.config.memory_warning_threshold:
                score_factors.append(1.0)
            elif check.memory_usage < self.config.memory_critical_threshold:
                score_factors.append(0.5)
            else:
                score_factors.append(0.1)
                
            # Error rate factor
            error_rate = self.error_handler.get_error_rate(check.component)
            if error_rate < self.config.error_rate_warning:
                score_factors.append(1.0)
            elif error_rate < self.config.error_rate_critical:
                score_factors.append(0.5)
            else:
                score_factors.append(0.1)
                
            return sum(score_factors) / len(score_factors) if score_factors else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate health score: {e}")
            return 0.0
            
    def _determine_health_status(self, check: HealthCheck) -> HealthStatus:
        """Determine health status from score"""
        try:
            if check.score >= 0.9:
                return HealthStatus.HEALTHY
            elif check.score >= 0.7:
                return HealthStatus.WARNING
            elif check.score >= 0.5:
                return HealthStatus.DEGRADED
            elif check.score >= 0.1:
                return HealthStatus.CRITICAL
            else:
                return HealthStatus.UNAVAILABLE
                
        except Exception as e:
            logger.error(f"Failed to determine health status: {e}")
            return HealthStatus.UNAVAILABLE
            
    def _check_for_issues(self, check: HealthCheck):
        """Check for issues and add recommendations"""
        try:
            # CPU issues
            if check.cpu_usage >= self.config.cpu_critical_threshold:
                check.issues.append("Critical CPU usage")
                check.recommendations.append("Consider scaling up CPU resources")
            elif check.cpu_usage >= self.config.cpu_warning_threshold:
                check.issues.append("High CPU usage")
                check.recommendations.append("Monitor CPU usage and optimize processes")
                
            # Memory issues
            if check.memory_usage >= self.config.memory_critical_threshold:
                check.issues.append("Critical memory usage")
                check.recommendations.append("Free up memory or add more RAM")
            elif check.memory_usage >= self.config.memory_warning_threshold:
                check.issues.append("High memory usage")
                check.recommendations.append("Monitor memory usage and optimize")
                
            # Error rate issues
            error_rate = self.error_handler.get_error_rate(check.component)
            if error_rate >= self.config.error_rate_critical:
                check.issues.append("Critical error rate")
                check.recommendations.append("Investigate and fix recurring errors")
            elif error_rate >= self.config.error_rate_warning:
                check.issues.append("Elevated error rate")
                check.recommendations.append("Monitor error patterns")
                
        except Exception as e:
            logger.error(f"Failed to check for issues: {e}")
            
    def _create_alert(self, level: AlertLevel, message: str):
        """Create alert"""
        try:
            alert = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now(),
                "level": level,
                "message": message
            }
            
            self.alert_queue.put(alert)
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            
    def _process_alerts(self):
        """Process pending alerts"""
        try:
            while not self.alert_queue.empty():
                alert = self.alert_queue.get_nowait()
                
                # Add to history
                self.alert_history.append(alert)
                
                # Call callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")
                        
                # Log alert
                logger.log(
                    logging.ERROR if alert["level"] == AlertLevel.ERROR else logging.WARNING,
                    f"Alert: {alert['message']}"
                )
                
        except Exception as e:
            logger.error(f"Failed to process alerts: {e}")
            
    def _update_statistics(self):
        """Update statistics"""
        try:
            # Update uptime
            self.stats["uptime"] = (datetime.now() - self.start_time).total_seconds()
            
            # Update error rate
            self.stats["error_rate"] = self.error_handler.get_error_rate()
            
        except Exception as e:
            logger.error(f"Failed to update statistics: {e}")
            
    def add_status_callback(self, component: str, callback: Callable):
        """Add status callback"""
        self.status_callbacks[component].append(callback)
        
    def add_health_callback(self, callback: Callable):
        """Add health callback"""
        self.health_callbacks.append(callback)
        
    def add_alert_callback(self, callback: Callable):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
        
    def get_status(self, component: str = None) -> Dict[str, Any]:
        """Get current status"""
        try:
            if component:
                return {
                    "component": component,
                    "status": self.current_status[component].value,
                    "health": self.health_status[component].value
                }
            else:
                return {
                    "tracker_id": self.tracker_id,
                    "is_monitoring": self.is_monitoring,
                    "component_status": {k: v.value for k, v in self.current_status.items()},
                    "health_status": {k: v.value for k, v in self.health_status.items()},
                    "stats": self.stats,
                    "config": self.config.to_dict()
                }
                
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {"error": str(e)}
            
    def get_health_history(self, component: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """Get health history"""
        try:
            filtered_history = [
                check.to_dict() for check in self.health_history
                if component is None or check.component == component
            ]
            
            if limit:
                filtered_history = filtered_history[-limit:]
                
            return filtered_history
            
        except Exception as e:
            logger.error(f"Failed to get health history: {e}")
            return []
            
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary"""
        try:
            return {
                "total_errors": len(self.error_handler.error_history),
                "error_rate": self.error_handler.get_error_rate(),
                "recent_errors": [
                    error.to_dict() for error in list(self.error_handler.error_history)[-10:]
                ],
                "error_by_component": self._get_error_by_component(),
                "resolution_rate": self._get_resolution_rate()
            }
            
        except Exception as e:
            logger.error(f"Failed to get error summary: {e}")
            return {"error": str(e)}
            
    def _get_error_by_component(self) -> Dict[str, int]:
        """Get error count by component"""
        try:
            error_counts = defaultdict(int)
            
            for error in self.error_handler.error_history:
                error_counts[error.component] += 1
                
            return dict(error_counts)
            
        except Exception as e:
            logger.error(f"Failed to get error by component: {e}")
            return {}
            
    def _get_resolution_rate(self) -> float:
        """Get error resolution rate"""
        try:
            total_errors = len(self.error_handler.error_history)
            if total_errors == 0:
                return 1.0
                
            resolved_errors = sum(1 for error in self.error_handler.error_history if error.resolved)
            return resolved_errors / total_errors
            
        except Exception as e:
            logger.error(f"Failed to get resolution rate: {e}")
            return 0.0
            
    def shutdown(self):
        """Shutdown status tracker"""
        logger.info("Shutting down status tracker...")
        
        self.stop_monitoring()
        
        logger.info("Status tracker shutdown complete")


# Factory functions
def create_basic_status_tracker() -> StatusTracker:
    """Create basic status tracker"""
    config = StatusConfig(
        enable_alerts=False,
        enable_auto_recovery=False,
        health_check_interval=60.0
    )
    return StatusTracker(config)


def create_comprehensive_status_tracker() -> StatusTracker:
    """Create comprehensive status tracker"""
    config = StatusConfig(
        enable_alerts=True,
        enable_auto_recovery=True,
        health_check_interval=30.0,
        max_status_history=2000,
        max_error_history=200
    )
    return StatusTracker(config)


def create_production_status_tracker() -> StatusTracker:
    """Create production-ready status tracker"""
    config = StatusConfig(
        enable_alerts=True,
        enable_auto_recovery=True,
        health_check_interval=15.0,
        max_status_history=5000,
        max_error_history=500,
        memory_warning_threshold=0.75,
        memory_critical_threshold=0.90,
        cpu_warning_threshold=0.75,
        cpu_critical_threshold=0.90
    )
    return StatusTracker(config)


# Testing and demonstration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Status Tracker Test")
    parser.add_argument("--tracker-type", type=str, default="comprehensive",
                       choices=["basic", "comprehensive", "production"], help="Tracker type")
    parser.add_argument("--test-duration", type=int, default=30, help="Test duration in seconds")
    args = parser.parse_args()
    
    # Create tracker
    if args.tracker_type == "basic":
        tracker = create_basic_status_tracker()
    elif args.tracker_type == "production":
        tracker = create_production_status_tracker()
    else:
        tracker = create_comprehensive_status_tracker()
        
    try:
        print(f"Tracker status: {tracker.get_status()}")
        
        # Start monitoring
        tracker.start_monitoring()
        
        # Simulate some activity
        print("Simulating activity...")
        
        # Update status
        tracker.update_status("test_component", ProcessingStatus.PROCESSING, "Starting test")
        
        # Simulate error
        try:
            raise ValueError("Test error")
        except ValueError as e:
            error_id = tracker.report_error("test_component", e)
            print(f"Error reported: {error_id}")
            
        # Perform health check
        health_check = tracker.perform_health_check("test_component")
        print(f"Health check: {health_check.status.value}, Score: {health_check.score:.3f}")
        
        # Update status to completed
        tracker.update_status("test_component", ProcessingStatus.COMPLETED, "Test completed")
        
        # Wait for monitoring
        print(f"Monitoring for {args.test_duration} seconds...")
        time.sleep(args.test_duration)
        
        # Get final status
        final_status = tracker.get_status()
        print(f"Final status: {final_status}")
        
        # Get error summary
        error_summary = tracker.get_error_summary()
        print(f"Error summary: {error_summary}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        tracker.shutdown()