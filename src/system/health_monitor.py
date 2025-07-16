"""
System Health Monitoring

Real-time system health monitoring with diagnostics, predictive maintenance,
and automated remediation for The Silent Steno device.
"""

import os
import psutil
import logging
import threading
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, NamedTuple
from pathlib import Path
from enum import Enum
import sqlite3
from collections import deque
import subprocess


class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    SYSTEM = "system"
    AUDIO = "audio"
    BLUETOOTH = "bluetooth"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    APPLICATION = "application"


@dataclass
class HealthThreshold:
    """Health monitoring thresholds."""
    warning_value: float
    critical_value: float
    unit: str = ""
    higher_is_worse: bool = True  # True for CPU/Memory, False for available space


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    component: ComponentType
    threshold: Optional[HealthThreshold] = None
    
    @property
    def status(self) -> HealthStatus:
        """Get health status based on thresholds."""
        if not self.threshold:
            return HealthStatus.UNKNOWN
        
        if self.threshold.higher_is_worse:
            if self.value >= self.threshold.critical_value:
                return HealthStatus.CRITICAL
            elif self.value >= self.threshold.warning_value:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY
        else:
            if self.value <= self.threshold.critical_value:
                return HealthStatus.CRITICAL
            elif self.value <= self.threshold.warning_value:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY


@dataclass
class ComponentHealth:
    """Health status for a system component."""
    component: ComponentType
    status: HealthStatus
    metrics: List[HealthMetric] = field(default_factory=list)
    last_check: Optional[datetime] = None
    error_count: int = 0
    issues: List[str] = field(default_factory=list)
    remediation_attempted: bool = False
    
    @property
    def overall_health_score(self) -> float:
        """Calculate overall health score (0-100)."""
        if not self.metrics:
            return 0.0
        
        scores = []
        for metric in self.metrics:
            if metric.threshold:
                if metric.threshold.higher_is_worse:
                    # Lower is better (CPU, memory usage)
                    score = max(0, 100 - (metric.value / metric.threshold.critical_value * 100))
                else:
                    # Higher is better (available space)
                    score = min(100, metric.value / metric.threshold.warning_value * 100)
                scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class SystemHealth:
    """Overall system health status."""
    overall_status: HealthStatus
    components: Dict[ComponentType, ComponentHealth] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    uptime_seconds: float = 0.0
    alerts: List[str] = field(default_factory=list)
    
    @property
    def overall_health_score(self) -> float:
        """Calculate overall system health score."""
        if not self.components:
            return 0.0
        
        scores = [comp.overall_health_score for comp in self.components.values()]
        return sum(scores) / len(scores)


class SystemMetrics:
    """Collects system metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.boot_time = datetime.fromtimestamp(psutil.boot_time())
    
    def get_cpu_metrics(self) -> List[HealthMetric]:
        """Get CPU-related metrics."""
        metrics = []
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="%",
                timestamp=datetime.now(),
                component=ComponentType.SYSTEM,
                threshold=HealthThreshold(70.0, 90.0, "%")
            ))
            
            # CPU temperature (if available)
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.current:
                                metrics.append(HealthMetric(
                                    name=f"temperature_{name}",
                                    value=entry.current,
                                    unit="°C",
                                    timestamp=datetime.now(),
                                    component=ComponentType.SYSTEM,
                                    threshold=HealthThreshold(65.0, 80.0, "°C")
                                ))
            
            # Load average
            load_avg = os.getloadavg()
            metrics.append(HealthMetric(
                name="load_average_1m",
                value=load_avg[0],
                unit="",
                timestamp=datetime.now(),
                component=ComponentType.SYSTEM,
                threshold=HealthThreshold(2.0, 4.0, "")
            ))
            
        except Exception as e:
            self.logger.error(f"Error collecting CPU metrics: {e}")
        
        return metrics
    
    def get_memory_metrics(self) -> List[HealthMetric]:
        """Get memory-related metrics."""
        metrics = []
        
        try:
            # Virtual memory
            vmem = psutil.virtual_memory()
            metrics.append(HealthMetric(
                name="memory_usage",
                value=vmem.percent,
                unit="%",
                timestamp=datetime.now(),
                component=ComponentType.SYSTEM,
                threshold=HealthThreshold(80.0, 95.0, "%")
            ))
            
            metrics.append(HealthMetric(
                name="memory_available",
                value=vmem.available / (1024 ** 3),  # GB
                unit="GB",
                timestamp=datetime.now(),
                component=ComponentType.SYSTEM,
                threshold=HealthThreshold(1.0, 0.5, "GB", higher_is_worse=False)
            ))
            
            # Swap memory
            swap = psutil.swap_memory()
            if swap.total > 0:
                metrics.append(HealthMetric(
                    name="swap_usage",
                    value=swap.percent,
                    unit="%",
                    timestamp=datetime.now(),
                    component=ComponentType.SYSTEM,
                    threshold=HealthThreshold(50.0, 80.0, "%")
                ))
            
        except Exception as e:
            self.logger.error(f"Error collecting memory metrics: {e}")
        
        return metrics
    
    def get_storage_metrics(self) -> List[HealthMetric]:
        """Get storage-related metrics."""
        metrics = []
        
        try:
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            metrics.append(HealthMetric(
                name="disk_usage",
                value=usage_percent,
                unit="%",
                timestamp=datetime.now(),
                component=ComponentType.STORAGE,
                threshold=HealthThreshold(80.0, 95.0, "%")
            ))
            
            metrics.append(HealthMetric(
                name="disk_free",
                value=disk_usage.free / (1024 ** 3),  # GB
                unit="GB",
                timestamp=datetime.now(),
                component=ComponentType.STORAGE,
                threshold=HealthThreshold(2.0, 1.0, "GB", higher_is_worse=False)
            ))
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics.append(HealthMetric(
                    name="disk_read_rate",
                    value=disk_io.read_bytes / (1024 ** 2),  # MB
                    unit="MB",
                    timestamp=datetime.now(),
                    component=ComponentType.STORAGE,
                    threshold=HealthThreshold(100.0, 500.0, "MB/s")
                ))
        
        except Exception as e:
            self.logger.error(f"Error collecting storage metrics: {e}")
        
        return metrics
    
    def get_network_metrics(self) -> List[HealthMetric]:
        """Get network-related metrics."""
        metrics = []
        
        try:
            # Network interfaces
            net_io = psutil.net_io_counters()
            if net_io:
                metrics.append(HealthMetric(
                    name="network_bytes_sent",
                    value=net_io.bytes_sent / (1024 ** 2),  # MB
                    unit="MB",
                    timestamp=datetime.now(),
                    component=ComponentType.NETWORK,
                    threshold=HealthThreshold(1000.0, 5000.0, "MB")
                ))
                
                metrics.append(HealthMetric(
                    name="network_bytes_recv",
                    value=net_io.bytes_recv / (1024 ** 2),  # MB
                    unit="MB",
                    timestamp=datetime.now(),
                    component=ComponentType.NETWORK,
                    threshold=HealthThreshold(1000.0, 5000.0, "MB")
                ))
            
            # Network connections
            connections = psutil.net_connections()
            established_count = len([c for c in connections if c.status == psutil.CONN_ESTABLISHED])
            
            metrics.append(HealthMetric(
                name="network_connections",
                value=established_count,
                unit="connections",
                timestamp=datetime.now(),
                component=ComponentType.NETWORK,
                threshold=HealthThreshold(50.0, 100.0, "connections")
            ))
            
        except Exception as e:
            self.logger.error(f"Error collecting network metrics: {e}")
        
        return metrics
    
    def get_process_metrics(self) -> List[HealthMetric]:
        """Get process-related metrics."""
        metrics = []
        
        try:
            # Process count
            process_count = len(psutil.pids())
            metrics.append(HealthMetric(
                name="process_count",
                value=process_count,
                unit="processes",
                timestamp=datetime.now(),
                component=ComponentType.SYSTEM,
                threshold=HealthThreshold(200.0, 500.0, "processes")
            ))
            
            # Find Silent Steno processes
            silent_steno_procs = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if 'python' in proc.info['name'] and 'silent' in ' '.join(proc.cmdline()).lower():
                        silent_steno_procs.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if silent_steno_procs:
                total_cpu = sum(p.info['cpu_percent'] for p in silent_steno_procs)
                total_memory = sum(p.info['memory_percent'] for p in silent_steno_procs)
                
                metrics.append(HealthMetric(
                    name="app_cpu_usage",
                    value=total_cpu,
                    unit="%",
                    timestamp=datetime.now(),
                    component=ComponentType.APPLICATION,
                    threshold=HealthThreshold(50.0, 80.0, "%")
                ))
                
                metrics.append(HealthMetric(
                    name="app_memory_usage",
                    value=total_memory,
                    unit="%",
                    timestamp=datetime.now(),
                    component=ComponentType.APPLICATION,
                    threshold=HealthThreshold(30.0, 60.0, "%")
                ))
            
        except Exception as e:
            self.logger.error(f"Error collecting process metrics: {e}")
        
        return metrics
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return (datetime.now() - self.boot_time).total_seconds()


class HealthChecker:
    """Performs health checks on system components."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = SystemMetrics()
    
    def check_audio_health(self) -> ComponentHealth:
        """Check audio system health."""
        health = ComponentHealth(
            component=ComponentType.AUDIO,
            status=HealthStatus.HEALTHY,
            last_check=datetime.now()
        )
        
        try:
            # Check audio devices
            if not os.path.exists("/dev/snd"):
                health.status = HealthStatus.CRITICAL
                health.issues.append("No audio devices found")
                return health
            
            # Check ALSA/PulseAudio
            audio_processes = []
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if proc.info['name'] in ['pulseaudio', 'alsa']:
                        audio_processes.append(proc.info['name'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not audio_processes:
                health.status = HealthStatus.WARNING
                health.issues.append("Audio system processes not running")
            
            # Check audio card
            try:
                result = subprocess.run(
                    ['aplay', '-l'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode != 0:
                    health.status = HealthStatus.WARNING
                    health.issues.append("Audio card not detected")
            except Exception:
                health.status = HealthStatus.WARNING
                health.issues.append("Unable to check audio card")
            
        except Exception as e:
            health.status = HealthStatus.CRITICAL
            health.issues.append(f"Audio health check failed: {e}")
            self.logger.error(f"Audio health check failed: {e}")
        
        return health
    
    def check_bluetooth_health(self) -> ComponentHealth:
        """Check Bluetooth system health."""
        health = ComponentHealth(
            component=ComponentType.BLUETOOTH,
            status=HealthStatus.HEALTHY,
            last_check=datetime.now()
        )
        
        try:
            # Check Bluetooth hardware
            if not os.path.exists("/sys/class/bluetooth"):
                health.status = HealthStatus.CRITICAL
                health.issues.append("No Bluetooth hardware found")
                return health
            
            # Check Bluetooth service
            try:
                result = subprocess.run(
                    ['systemctl', 'is-active', 'bluetooth'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode != 0:
                    health.status = HealthStatus.WARNING
                    health.issues.append("Bluetooth service not running")
            except Exception:
                health.status = HealthStatus.WARNING
                health.issues.append("Unable to check Bluetooth service")
            
            # Check Bluetooth controller
            try:
                result = subprocess.run(
                    ['bluetoothctl', 'show'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if "Powered: yes" not in result.stdout:
                    health.status = HealthStatus.WARNING
                    health.issues.append("Bluetooth controller not powered")
            except Exception:
                health.status = HealthStatus.WARNING
                health.issues.append("Unable to check Bluetooth controller")
            
        except Exception as e:
            health.status = HealthStatus.CRITICAL
            health.issues.append(f"Bluetooth health check failed: {e}")
            self.logger.error(f"Bluetooth health check failed: {e}")
        
        return health
    
    def check_database_health(self) -> ComponentHealth:
        """Check database system health."""
        health = ComponentHealth(
            component=ComponentType.DATABASE,
            status=HealthStatus.HEALTHY,
            last_check=datetime.now()
        )
        
        try:
            # Check database file
            db_path = "/home/mmariani/projects/thesilentsteno/data/silentst.db"
            if not os.path.exists(db_path):
                health.status = HealthStatus.WARNING
                health.issues.append("Database file not found")
                return health
            
            # Check database integrity
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check;")
                result = cursor.fetchone()
                conn.close()
                
                if result[0] != "ok":
                    health.status = HealthStatus.CRITICAL
                    health.issues.append("Database integrity check failed")
            except Exception as e:
                health.status = HealthStatus.CRITICAL
                health.issues.append(f"Database connection failed: {e}")
            
            # Check database size
            db_size = os.path.getsize(db_path) / (1024 ** 2)  # MB
            health.metrics.append(HealthMetric(
                name="database_size",
                value=db_size,
                unit="MB",
                timestamp=datetime.now(),
                component=ComponentType.DATABASE,
                threshold=HealthThreshold(500.0, 1000.0, "MB")
            ))
            
        except Exception as e:
            health.status = HealthStatus.CRITICAL
            health.issues.append(f"Database health check failed: {e}")
            self.logger.error(f"Database health check failed: {e}")
        
        return health
    
    def check_application_health(self) -> ComponentHealth:
        """Check application health."""
        health = ComponentHealth(
            component=ComponentType.APPLICATION,
            status=HealthStatus.HEALTHY,
            last_check=datetime.now()
        )
        
        try:
            # Check if application is running
            app_running = False
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline']).lower()
                    if 'silent' in cmdline and 'steno' in cmdline:
                        app_running = True
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not app_running:
                health.status = HealthStatus.WARNING
                health.issues.append("Application not running")
            
            # Check configuration files
            config_files = [
                "/home/mmariani/projects/thesilentsteno/config/app_config.json",
                "/home/mmariani/projects/thesilentsteno/config/logging_config.json"
            ]
            
            for config_file in config_files:
                if not os.path.exists(config_file):
                    health.status = HealthStatus.WARNING
                    health.issues.append(f"Configuration file missing: {config_file}")
                else:
                    try:
                        with open(config_file, 'r') as f:
                            json.load(f)
                    except json.JSONDecodeError:
                        health.status = HealthStatus.CRITICAL
                        health.issues.append(f"Invalid configuration file: {config_file}")
            
        except Exception as e:
            health.status = HealthStatus.CRITICAL
            health.issues.append(f"Application health check failed: {e}")
            self.logger.error(f"Application health check failed: {e}")
        
        return health


class HealthMonitor:
    """Main health monitoring system."""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
        self.checker = HealthChecker()
        self.metrics = SystemMetrics()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        
        # Health history
        self.health_history: deque = deque(maxlen=100)
        self.alert_callbacks: List[Callable] = []
        
        # Last health check
        self.last_health: Optional[SystemHealth] = None
    
    def start(self):
        """Start health monitoring."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        self.logger.info("Health monitoring started")
    
    def stop(self):
        """Stop health monitoring."""
        if not self.running:
            return
        
        self.running = False
        self._stop_event.set()
        
        if self.thread:
            self.thread.join(timeout=5.0)
        
        self.logger.info("Health monitoring stopped")
    
    def check_system_health(self) -> SystemHealth:
        """Perform comprehensive system health check."""
        try:
            with self._lock:
                health = SystemHealth(
                    overall_status=HealthStatus.HEALTHY,
                    timestamp=datetime.now(),
                    uptime_seconds=self.metrics.get_uptime()
                )
                
                # Check each component
                components = [
                    self.checker.check_audio_health(),
                    self.checker.check_bluetooth_health(),
                    self.checker.check_database_health(),
                    self.checker.check_application_health()
                ]
                
                # Add system metrics
                system_health = ComponentHealth(
                    component=ComponentType.SYSTEM,
                    status=HealthStatus.HEALTHY,
                    last_check=datetime.now()
                )
                
                # Collect all metrics
                all_metrics = []
                all_metrics.extend(self.metrics.get_cpu_metrics())
                all_metrics.extend(self.metrics.get_memory_metrics())
                all_metrics.extend(self.metrics.get_storage_metrics())
                all_metrics.extend(self.metrics.get_network_metrics())
                all_metrics.extend(self.metrics.get_process_metrics())
                
                system_health.metrics = all_metrics
                
                # Determine system health status from metrics
                critical_metrics = [m for m in all_metrics if m.status == HealthStatus.CRITICAL]
                warning_metrics = [m for m in all_metrics if m.status == HealthStatus.WARNING]
                
                if critical_metrics:
                    system_health.status = HealthStatus.CRITICAL
                    system_health.issues.extend([f"Critical: {m.name} = {m.value}{m.unit}" for m in critical_metrics])
                elif warning_metrics:
                    system_health.status = HealthStatus.WARNING
                    system_health.issues.extend([f"Warning: {m.name} = {m.value}{m.unit}" for m in warning_metrics])
                
                components.append(system_health)
                
                # Build component health map
                for component in components:
                    health.components[component.component] = component
                
                # Determine overall health status
                critical_components = [c for c in components if c.status == HealthStatus.CRITICAL]
                warning_components = [c for c in components if c.status == HealthStatus.WARNING]
                
                if critical_components:
                    health.overall_status = HealthStatus.CRITICAL
                    health.alerts.extend([f"Critical: {c.component.value}" for c in critical_components])
                elif warning_components:
                    health.overall_status = HealthStatus.WARNING
                    health.alerts.extend([f"Warning: {c.component.value}" for c in warning_components])
                
                # Store health check
                self.last_health = health
                self.health_history.append(health)
                
                # Trigger alerts if needed
                if health.overall_status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
                    self._trigger_alerts(health)
                
                return health
                
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return SystemHealth(
                overall_status=HealthStatus.UNKNOWN,
                alerts=[f"Health check failed: {e}"]
            )
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running and not self._stop_event.is_set():
            try:
                health = self.check_system_health()
                self.logger.debug(f"Health check: {health.overall_status.value} (score: {health.overall_health_score:.1f})")
                
                # Wait for next check
                if self._stop_event.wait(self.check_interval):
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(10)  # Wait before retrying
    
    def _trigger_alerts(self, health: SystemHealth):
        """Trigger alerts for health issues."""
        try:
            for callback in self.alert_callbacks:
                callback(health)
        except Exception as e:
            self.logger.error(f"Error triggering health alerts: {e}")
    
    def add_alert_callback(self, callback: Callable[[SystemHealth], None]):
        """Add callback for health alerts."""
        self.alert_callbacks.append(callback)
    
    def get_health_history(self, hours: int = 24) -> List[SystemHealth]:
        """Get health history for specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [h for h in self.health_history if h.timestamp >= cutoff_time]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health monitoring summary."""
        if not self.last_health:
            return {"status": "no_data"}
        
        return {
            "overall_status": self.last_health.overall_status.value,
            "overall_score": self.last_health.overall_health_score,
            "uptime_hours": self.last_health.uptime_seconds / 3600,
            "last_check": self.last_health.timestamp.isoformat(),
            "components": {
                comp.component.value: {
                    "status": comp.status.value,
                    "score": comp.overall_health_score,
                    "issues": comp.issues
                }
                for comp in self.last_health.components.values()
            },
            "alerts": self.last_health.alerts,
            "monitoring_active": self.running
        }


# Factory functions
def create_health_monitor(check_interval: int = 60) -> HealthMonitor:
    """Create a health monitor instance."""
    return HealthMonitor(check_interval)


def check_system_health(monitor: HealthMonitor = None) -> SystemHealth:
    """Perform system health check."""
    if monitor is None:
        monitor = create_health_monitor()
    
    return monitor.check_system_health()


def monitor_components(monitor: HealthMonitor = None, start: bool = True):
    """Start or stop component monitoring."""
    if monitor is None:
        monitor = create_health_monitor()
    
    if start:
        monitor.start()
    else:
        monitor.stop()


def report_health(monitor: HealthMonitor = None) -> Dict[str, Any]:
    """Get health monitoring report."""
    if monitor is None:
        monitor = create_health_monitor()
    
    return monitor.get_health_summary()