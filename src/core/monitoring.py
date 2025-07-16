"""
Performance Monitoring and System Health Tracking

Real-time performance monitoring with metrics collection, health checks,
and alerting system for comprehensive system oversight.
"""

import logging
import psutil
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set
import json
import uuid


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class HealthCheck:
    """Health check configuration and results."""
    name: str
    check_function: Callable[[], bool]
    interval: float = 30.0  # seconds
    timeout: float = 5.0  # seconds
    critical: bool = False
    last_check: Optional[datetime] = None
    last_result: Optional[bool] = None
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    max_failures: int = 3


@dataclass
class Alert:
    """System alert information."""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    severity: HealthStatus = HealthStatus.WARNING
    message: str = ""
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and manages system metrics."""
    
    def __init__(self, max_history: int = 10000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Start system metrics collection
        self._start_system_metrics()
    
    def record_metric(self, name: str, value: float, metric_type: MetricType,
                     tags: Dict[str, str] = None, unit: str = ""):
        """Record a metric value."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {},
            unit=unit
        )
        
        with self._lock:
            self.metrics[name].append(metric)
            
            if metric_type == MetricType.COUNTER:
                self.counters[name] += value
            elif metric_type == MetricType.GAUGE:
                self.gauges[name] = value
            elif metric_type == MetricType.TIMER:
                self.timers[name].append(value)
                # Keep only recent timer values
                if len(self.timers[name]) > 1000:
                    self.timers[name] = self.timers[name][-1000:]
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, tags)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None, unit: str = ""):
        """Set a gauge metric value."""
        self.record_metric(name, value, MetricType.GAUGE, tags, unit)
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timer metric."""
        self.record_metric(name, duration, MetricType.TIMER, tags, "seconds")
    
    def get_metric_summary(self, name: str, duration: timedelta = None) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        with self._lock:
            if name not in self.metrics:
                return {}
            
            metrics_data = list(self.metrics[name])
            
            # Filter by duration if specified
            if duration:
                cutoff_time = datetime.now() - duration
                metrics_data = [m for m in metrics_data if m.timestamp >= cutoff_time]
            
            if not metrics_data:
                return {}
            
            values = [m.value for m in metrics_data]
            
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'latest': values[-1] if values else None,
                'unit': metrics_data[-1].unit if metrics_data else ''
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric values."""
        with self._lock:
            return {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'timers': {
                    name: {
                        'count': len(values),
                        'mean': sum(values) / len(values) if values else 0,
                        'min': min(values) if values else 0,
                        'max': max(values) if values else 0
                    }
                    for name, values in self.timers.items()
                }
            }
    
    def _start_system_metrics(self):
        """Start collecting system metrics."""
        def collect_system_metrics():
            while True:
                try:
                    # CPU metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.set_gauge('system.cpu.usage', cpu_percent, unit='percent')
                    
                    # Memory metrics
                    memory = psutil.virtual_memory()
                    self.set_gauge('system.memory.usage', memory.percent, unit='percent')
                    self.set_gauge('system.memory.available', memory.available, unit='bytes')
                    
                    # Disk metrics
                    disk = psutil.disk_usage('/')
                    self.set_gauge('system.disk.usage', (disk.used / disk.total) * 100, unit='percent')
                    self.set_gauge('system.disk.free', disk.free, unit='bytes')
                    
                    # Network metrics
                    network = psutil.net_io_counters()
                    self.set_gauge('system.network.bytes_sent', network.bytes_sent, unit='bytes')
                    self.set_gauge('system.network.bytes_recv', network.bytes_recv, unit='bytes')
                    
                    time.sleep(30)  # Collect every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error collecting system metrics: {e}")
                    time.sleep(30)
        
        metrics_thread = threading.Thread(target=collect_system_metrics, daemon=True)
        metrics_thread.start()


class HealthChecker:
    """Performs health checks on system components."""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_status: Dict[str, HealthStatus] = {}
        self.check_results: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        self._lock = threading.RLock()
        self._running = False
        self._check_thread = None
        self.logger = logging.getLogger(__name__)
    
    def register_health_check(self, name: str, check_function: Callable[[], bool],
                            interval: float = 30.0, timeout: float = 5.0,
                            critical: bool = False, max_failures: int = 3):
        """Register a health check."""
        health_check = HealthCheck(
            name=name,
            check_function=check_function,
            interval=interval,
            timeout=timeout,
            critical=critical,
            max_failures=max_failures
        )
        
        with self._lock:
            self.health_checks[name] = health_check
            self.health_status[name] = HealthStatus.UNKNOWN
    
    def unregister_health_check(self, name: str):
        """Unregister a health check."""
        with self._lock:
            self.health_checks.pop(name, None)
            self.health_status.pop(name, None)
            self.check_results.pop(name, None)
    
    def start_monitoring(self):
        """Start health check monitoring."""
        if self._running:
            return
        
        self._running = True
        self._check_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._check_thread.start()
        self.logger.info("Health check monitoring started")
    
    def stop_monitoring(self):
        """Stop health check monitoring."""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5)
        self.logger.info("Health check monitoring stopped")
    
    def run_check(self, name: str) -> bool:
        """Run a specific health check manually."""
        with self._lock:
            if name not in self.health_checks:
                return False
            
            health_check = self.health_checks[name]
            return self._execute_health_check(health_check)
    
    def get_health_status(self, name: str = None) -> Dict[str, Any]:
        """Get health status for a check or all checks."""
        with self._lock:
            if name:
                if name in self.health_status:
                    return {
                        'status': self.health_status[name].value,
                        'last_check': self.health_checks[name].last_check,
                        'consecutive_failures': self.health_checks[name].consecutive_failures
                    }
                return {}
            else:
                return {
                    check_name: {
                        'status': status.value,
                        'last_check': self.health_checks[check_name].last_check,
                        'consecutive_failures': self.health_checks[check_name].consecutive_failures,
                        'critical': self.health_checks[check_name].critical
                    }
                    for check_name, status in self.health_status.items()
                }
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        with self._lock:
            if not self.health_status:
                return HealthStatus.UNKNOWN
            
            # If any critical check is failed, system is critical
            for name, status in self.health_status.items():
                health_check = self.health_checks[name]
                if health_check.critical and status == HealthStatus.CRITICAL:
                    return HealthStatus.CRITICAL
            
            # If any check is critical (non-critical), system is warning
            if HealthStatus.CRITICAL in self.health_status.values():
                return HealthStatus.WARNING
            
            # If any check is warning, system is warning
            if HealthStatus.WARNING in self.health_status.values():
                return HealthStatus.WARNING
            
            # All checks healthy
            return HealthStatus.HEALTHY
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                current_time = datetime.now()
                
                with self._lock:
                    checks_to_run = []
                    for name, health_check in self.health_checks.items():
                        if (health_check.last_check is None or
                            (current_time - health_check.last_check).total_seconds() >= health_check.interval):
                            checks_to_run.append(health_check)
                
                # Run checks outside of lock
                for health_check in checks_to_run:
                    self._execute_health_check(health_check)
                
                time.sleep(1)  # Check every second for due health checks
                
            except Exception as e:
                self.logger.error(f"Error in health check monitoring loop: {e}")
                time.sleep(5)
    
    def _execute_health_check(self, health_check: HealthCheck) -> bool:
        """Execute a health check."""
        try:
            start_time = time.time()
            
            # Execute check with timeout
            result = health_check.check_function()
            
            duration = time.time() - start_time
            
            with self._lock:
                health_check.last_check = datetime.now()
                health_check.last_result = result
                health_check.last_error = None
                
                if result:
                    health_check.consecutive_failures = 0
                    self.health_status[health_check.name] = HealthStatus.HEALTHY
                else:
                    health_check.consecutive_failures += 1
                    if health_check.consecutive_failures >= health_check.max_failures:
                        self.health_status[health_check.name] = HealthStatus.CRITICAL
                    else:
                        self.health_status[health_check.name] = HealthStatus.WARNING
                
                # Record result
                self.check_results[health_check.name].append({
                    'timestamp': health_check.last_check,
                    'result': result,
                    'duration': duration
                })
            
            return result
            
        except Exception as e:
            with self._lock:
                health_check.last_check = datetime.now()
                health_check.last_result = False
                health_check.last_error = str(e)
                health_check.consecutive_failures += 1
                
                if health_check.consecutive_failures >= health_check.max_failures:
                    self.health_status[health_check.name] = HealthStatus.CRITICAL
                else:
                    self.health_status[health_check.name] = HealthStatus.WARNING
            
            self.logger.error(f"Health check {health_check.name} failed: {e}")
            return False


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.alert_rules: List[Callable[[Dict[str, Any]], Optional[Alert]]] = []
        
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler."""
        with self._lock:
            self.alert_handlers.append(handler)
    
    def add_alert_rule(self, rule: Callable[[Dict[str, Any]], Optional[Alert]]):
        """Add an alert rule."""
        with self._lock:
            self.alert_rules.append(rule)
    
    def create_alert(self, alert_type: str, severity: HealthStatus, message: str,
                    source: str = "", metadata: Dict[str, Any] = None) -> Alert:
        """Create and process a new alert."""
        alert = Alert(
            type=alert_type,
            severity=severity,
            message=message,
            source=source,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.alerts[alert.alert_id] = alert
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
        
        self.logger.warning(f"Alert created: {alert.type} - {alert.message}")
        return alert
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].acknowledged = True
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
                return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        with self._lock:
            return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def evaluate_rules(self, metrics: Dict[str, Any]):
        """Evaluate alert rules against current metrics."""
        for rule in self.alert_rules:
            try:
                alert = rule(metrics)
                if alert:
                    self.create_alert(
                        alert.type, alert.severity, alert.message,
                        alert.source, alert.metadata
                    )
            except Exception as e:
                self.logger.error(f"Error evaluating alert rule: {e}")


class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager()
        
        self.logger = logging.getLogger(__name__)
        self._setup_default_health_checks()
        self._setup_default_alert_rules()
    
    def start(self):
        """Start performance monitoring."""
        self.health_checker.start_monitoring()
        self.logger.info("Performance monitoring started")
    
    def stop(self):
        """Stop performance monitoring."""
        self.health_checker.stop_monitoring()
        self.logger.info("Performance monitoring stopped")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        return self.metrics_collector.get_all_metrics()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        return {
            'overall': self.health_checker.get_overall_health().value,
            'checks': self.health_checker.get_health_status()
        }
    
    def get_alerts(self) -> List[Alert]:
        """Get active alerts."""
        return self.alert_manager.get_active_alerts()
    
    def _setup_default_health_checks(self):
        """Setup default system health checks."""
        def check_cpu_usage():
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < 90  # Alert if CPU > 90%
        
        def check_memory_usage():
            memory = psutil.virtual_memory()
            return memory.percent < 85  # Alert if memory > 85%
        
        def check_disk_space():
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            return usage_percent < 90  # Alert if disk > 90%
        
        self.health_checker.register_health_check(
            "cpu_usage", check_cpu_usage, interval=30, critical=True
        )
        self.health_checker.register_health_check(
            "memory_usage", check_memory_usage, interval=30, critical=True
        )
        self.health_checker.register_health_check(
            "disk_space", check_disk_space, interval=60, critical=False
        )
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules."""
        def high_cpu_rule(metrics):
            cpu_usage = metrics.get('gauges', {}).get('system.cpu.usage', 0)
            if cpu_usage > 90:
                return Alert(
                    type="high_cpu_usage",
                    severity=HealthStatus.CRITICAL,
                    message=f"High CPU usage: {cpu_usage:.1f}%",
                    source="system",
                    metadata={'cpu_usage': cpu_usage}
                )
            return None
        
        def high_memory_rule(metrics):
            memory_usage = metrics.get('gauges', {}).get('system.memory.usage', 0)
            if memory_usage > 85:
                return Alert(
                    type="high_memory_usage",
                    severity=HealthStatus.WARNING,
                    message=f"High memory usage: {memory_usage:.1f}%",
                    source="system",
                    metadata={'memory_usage': memory_usage}
                )
            return None
        
        self.alert_manager.add_alert_rule(high_cpu_rule)
        self.alert_manager.add_alert_rule(high_memory_rule)


# Global performance monitor
_global_performance_monitor: Optional[PerformanceMonitor] = None
_monitor_lock = threading.Lock()


def start_monitoring() -> PerformanceMonitor:
    """Start global performance monitoring."""
    global _global_performance_monitor
    
    with _monitor_lock:
        if _global_performance_monitor is None:
            _global_performance_monitor = PerformanceMonitor()
        
        _global_performance_monitor.start()
        return _global_performance_monitor


def collect_metrics() -> Dict[str, Any]:
    """Collect current metrics."""
    if _global_performance_monitor:
        return _global_performance_monitor.get_metrics()
    return {}


def check_system_health() -> Dict[str, Any]:
    """Check system health status."""
    if _global_performance_monitor:
        return _global_performance_monitor.get_health_status()
    return {'overall': 'unknown', 'checks': {}}


def send_alert(alert_type: str, severity: HealthStatus, message: str,
              source: str = "", metadata: Dict[str, Any] = None) -> Optional[Alert]:
    """Send an alert."""
    if _global_performance_monitor:
        return _global_performance_monitor.alert_manager.create_alert(
            alert_type, severity, message, source, metadata
        )
    return None