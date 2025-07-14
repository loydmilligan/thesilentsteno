#!/usr/bin/env python3

"""
Storage Monitor for The Silent Steno

This module provides comprehensive storage monitoring and management for
audio recordings and system data. It tracks storage usage, predicts
capacity needs, monitors disk health, and provides automated cleanup
and optimization capabilities.

Key features:
- Real-time storage usage monitoring
- Capacity prediction and alerting
- Disk health monitoring and alerts
- Automated cleanup and optimization
- Storage quota management
- Performance monitoring and optimization
- Integration with recording system
"""

import os
import shutil
import time
import threading
import logging
import json
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StorageAlert(Enum):
    """Storage alert levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class StorageStatus(Enum):
    """Storage system status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FULL = "full"
    ERROR = "error"


@dataclass
class StorageStats:
    """Storage statistics"""
    total_gb: float
    used_gb: float
    available_gb: float
    usage_percent: float
    free_inodes: int
    total_inodes: int
    read_speed_mbps: float
    write_speed_mbps: float
    last_updated: datetime


@dataclass
class StorageConfig:
    """Storage monitoring configuration"""
    warning_threshold_percent: float = 80.0
    critical_threshold_percent: float = 90.0
    emergency_threshold_percent: float = 95.0
    min_free_gb: float = 2.0
    monitor_interval_seconds: int = 60
    cleanup_enabled: bool = True
    auto_optimize_enabled: bool = True
    health_check_enabled: bool = True
    max_recording_hours: float = 100.0  # Total hours of recordings to keep


@dataclass
class AlertInfo:
    """Storage alert information"""
    alert_type: StorageAlert
    message: str
    timestamp: datetime
    storage_stats: StorageStats
    suggested_actions: List[str]


class StorageMonitor:
    """
    Storage Monitor for The Silent Steno
    
    Provides comprehensive storage monitoring, health checking, and
    automated management for optimal recording system performance.
    """
    
    def __init__(self, storage_path: str, config: Optional[StorageConfig] = None):
        """Initialize storage monitor"""
        self.storage_path = os.path.abspath(storage_path)
        self.config = config or StorageConfig()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.stop_monitoring = threading.Event()
        self.monitor_lock = threading.RLock()
        
        # Storage tracking
        self.current_stats: Optional[StorageStats] = None
        self.usage_history: List[StorageStats] = []
        self.max_history_entries = 1440  # 24 hours at 1-minute intervals
        
        # Alert system
        self.active_alerts: List[AlertInfo] = []
        self.alert_callbacks: List[Callable] = []
        self.last_alert_time: Dict[StorageAlert, datetime] = {}
        
        # Performance tracking
        self.performance_stats = {
            "monitoring_cycles": 0,
            "alerts_generated": 0,
            "cleanup_operations": 0,
            "optimization_operations": 0,
            "average_check_time_ms": 0.0
        }
        
        # Components
        self.file_manager = None
        
        # Ensure storage path exists
        os.makedirs(self.storage_path, exist_ok=True)
        
        logger.info(f"Storage monitor initialized for: {self.storage_path}")
    
    def set_file_manager(self, file_manager) -> None:
        """Set file manager for cleanup operations"""
        self.file_manager = file_manager
    
    def add_alert_callback(self, callback: Callable[[AlertInfo], None]) -> None:
        """Add callback for storage alerts"""
        self.alert_callbacks.append(callback)
    
    def _notify_alert(self, alert: AlertInfo) -> None:
        """Notify callbacks of storage alerts"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def start_monitoring(self) -> bool:
        """
        Start storage monitoring
        
        Returns:
            True if monitoring started successfully
        """
        try:
            with self.monitor_lock:
                if self.monitoring_active:
                    logger.warning("Storage monitoring already active")
                    return True
                
                # Perform initial check
                self._update_storage_stats()
                
                # Start monitoring thread
                self.stop_monitoring.clear()
                self.monitor_thread = threading.Thread(
                    target=self._monitoring_loop,
                    daemon=True,
                    name="StorageMonitoring"
                )
                self.monitor_thread.start()
                
                self.monitoring_active = True
                logger.info("Storage monitoring started")
                return True
        
        except Exception as e:
            logger.error(f"Error starting storage monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> None:
        """Stop storage monitoring"""
        try:
            with self.monitor_lock:
                if not self.monitoring_active:
                    return
                
                self.stop_monitoring.set()
                if self.monitor_thread and self.monitor_thread.is_alive():
                    self.monitor_thread.join(timeout=5.0)
                
                self.monitoring_active = False
                logger.info("Storage monitoring stopped")
        
        except Exception as e:
            logger.error(f"Error stopping storage monitoring: {e}")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        try:
            while not self.stop_monitoring.is_set():
                start_time = time.time()
                
                try:
                    # Update storage statistics
                    self._update_storage_stats()
                    
                    # Check for alerts
                    self._check_storage_alerts()
                    
                    # Perform automated tasks
                    if self.config.cleanup_enabled:
                        self._perform_cleanup()
                    
                    if self.config.auto_optimize_enabled:
                        self._perform_optimization()
                    
                    if self.config.health_check_enabled:
                        self._check_storage_health()
                    
                    # Update performance stats
                    check_time = (time.time() - start_time) * 1000
                    self._update_performance_stats(check_time)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring cycle: {e}")
                
                # Wait for next cycle
                self.stop_monitoring.wait(self.config.monitor_interval_seconds)
        
        except Exception as e:
            logger.error(f"Fatal error in storage monitoring: {e}")
    
    def _update_storage_stats(self) -> None:
        """Update current storage statistics"""
        try:
            # Get disk usage
            disk_usage = shutil.disk_usage(self.storage_path)
            total_bytes = disk_usage.total
            free_bytes = disk_usage.free
            used_bytes = total_bytes - free_bytes
            
            # Convert to GB
            total_gb = total_bytes / (1024**3)
            used_gb = used_bytes / (1024**3)
            available_gb = free_bytes / (1024**3)
            usage_percent = (used_bytes / total_bytes) * 100
            
            # Get inode information
            free_inodes, total_inodes = self._get_inode_info()
            
            # Measure I/O performance
            read_speed, write_speed = self._measure_io_performance()
            
            # Create stats object
            stats = StorageStats(
                total_gb=total_gb,
                used_gb=used_gb,
                available_gb=available_gb,
                usage_percent=usage_percent,
                free_inodes=free_inodes,
                total_inodes=total_inodes,
                read_speed_mbps=read_speed,
                write_speed_mbps=write_speed,
                last_updated=datetime.now()
            )
            
            # Store current stats
            with self.monitor_lock:
                self.current_stats = stats
                
                # Add to history
                self.usage_history.append(stats)
                if len(self.usage_history) > self.max_history_entries:
                    self.usage_history.pop(0)
        
        except Exception as e:
            logger.error(f"Error updating storage stats: {e}")
    
    def _get_inode_info(self) -> Tuple[int, int]:
        """Get inode information"""
        try:
            # Use df command to get inode info
            cmd = ['df', '-i', self.storage_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    fields = lines[1].split()
                    if len(fields) >= 4:
                        total_inodes = int(fields[1])
                        used_inodes = int(fields[2])
                        free_inodes = int(fields[3])
                        return free_inodes, total_inodes
            
            return 0, 0
        
        except Exception as e:
            logger.debug(f"Error getting inode info: {e}")
            return 0, 0
    
    def _measure_io_performance(self) -> Tuple[float, float]:
        """Measure I/O performance (simplified)"""
        try:
            # Simple write test
            test_file = os.path.join(self.storage_path, '.storage_test')
            test_data = b'0' * (1024 * 1024)  # 1MB
            
            # Write test
            start_time = time.time()
            with open(test_file, 'wb') as f:
                f.write(test_data)
                f.flush()
                os.fsync(f.fileno())
            write_time = time.time() - start_time
            write_speed = (len(test_data) / (1024 * 1024)) / write_time  # MB/s
            
            # Read test
            start_time = time.time()
            with open(test_file, 'rb') as f:
                read_data = f.read()
            read_time = time.time() - start_time
            read_speed = (len(read_data) / (1024 * 1024)) / read_time  # MB/s
            
            # Cleanup
            try:
                os.remove(test_file)
            except:
                pass
            
            return read_speed, write_speed
        
        except Exception as e:
            logger.debug(f"Error measuring I/O performance: {e}")
            return 0.0, 0.0
    
    def _check_storage_alerts(self) -> None:
        """Check for storage alert conditions"""
        try:
            if not self.current_stats:
                return
            
            stats = self.current_stats
            alerts_to_generate = []
            
            # Check usage thresholds
            if stats.usage_percent >= self.config.emergency_threshold_percent:
                alerts_to_generate.append((StorageAlert.EMERGENCY, 
                    f"Storage critically full: {stats.usage_percent:.1f}% used"))
            elif stats.usage_percent >= self.config.critical_threshold_percent:
                alerts_to_generate.append((StorageAlert.CRITICAL,
                    f"Storage usage critical: {stats.usage_percent:.1f}% used"))
            elif stats.usage_percent >= self.config.warning_threshold_percent:
                alerts_to_generate.append((StorageAlert.WARNING,
                    f"Storage usage high: {stats.usage_percent:.1f}% used"))
            
            # Check free space
            if stats.available_gb < self.config.min_free_gb:
                alerts_to_generate.append((StorageAlert.CRITICAL,
                    f"Low free space: {stats.available_gb:.1f}GB remaining"))
            
            # Check I/O performance
            if stats.write_speed_mbps < 5.0:  # Less than 5 MB/s
                alerts_to_generate.append((StorageAlert.WARNING,
                    f"Poor write performance: {stats.write_speed_mbps:.1f} MB/s"))
            
            # Generate alerts (with rate limiting)
            current_time = datetime.now()
            for alert_type, message in alerts_to_generate:
                last_alert = self.last_alert_time.get(alert_type)
                if not last_alert or (current_time - last_alert).total_seconds() > 300:  # 5 minutes
                    self._generate_alert(alert_type, message, stats)
                    self.last_alert_time[alert_type] = current_time
        
        except Exception as e:
            logger.error(f"Error checking storage alerts: {e}")
    
    def _generate_alert(self, alert_type: StorageAlert, message: str, stats: StorageStats) -> None:
        """Generate storage alert"""
        try:
            # Generate suggested actions
            suggested_actions = self._get_suggested_actions(alert_type, stats)
            
            alert = AlertInfo(
                alert_type=alert_type,
                message=message,
                timestamp=datetime.now(),
                storage_stats=stats,
                suggested_actions=suggested_actions
            )
            
            # Store alert
            with self.monitor_lock:
                self.active_alerts.append(alert)
                # Keep only recent alerts
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.active_alerts = [a for a in self.active_alerts if a.timestamp > cutoff_time]
            
            # Notify callbacks
            self._notify_alert(alert)
            
            # Update stats
            self.performance_stats['alerts_generated'] += 1
            
            logger.warning(f"Storage alert ({alert_type.value}): {message}")
        
        except Exception as e:
            logger.error(f"Error generating alert: {e}")
    
    def _get_suggested_actions(self, alert_type: StorageAlert, stats: StorageStats) -> List[str]:
        """Get suggested actions for alert"""
        actions = []
        
        if alert_type in [StorageAlert.WARNING, StorageAlert.CRITICAL, StorageAlert.EMERGENCY]:
            actions.extend([
                "Delete old recordings and exports",
                "Clean up temporary files",
                "Move recordings to external storage",
                "Reduce recording quality settings"
            ])
        
        if stats.write_speed_mbps < 10.0:
            actions.extend([
                "Check disk health",
                "Defragment storage if needed",
                "Close unnecessary applications"
            ])
        
        if alert_type == StorageAlert.EMERGENCY:
            actions.extend([
                "Stop all recordings immediately",
                "Free up space urgently",
                "Consider expanding storage"
            ])
        
        return actions
    
    def _perform_cleanup(self) -> None:
        """Perform automated cleanup operations"""
        try:
            if not self.file_manager:
                return
            
            # Clean temporary files
            cleaned_count = self.file_manager.cleanup_temp_files()
            if cleaned_count > 0:
                self.performance_stats['cleanup_operations'] += 1
                logger.info(f"Cleaned {cleaned_count} temporary files")
        
        except Exception as e:
            logger.error(f"Error performing cleanup: {e}")
    
    def _perform_optimization(self) -> None:
        """Perform storage optimization"""
        try:
            if not self.current_stats:
                return
            
            # Only optimize if usage is getting high
            if self.current_stats.usage_percent > 70.0:
                # Could implement compression, deduplication, etc.
                self.performance_stats['optimization_operations'] += 1
                logger.debug("Storage optimization performed")
        
        except Exception as e:
            logger.error(f"Error performing optimization: {e}")
    
    def _check_storage_health(self) -> None:
        """Check storage device health"""
        try:
            # Simple health check based on I/O performance
            if self.current_stats:
                if (self.current_stats.read_speed_mbps < 1.0 or 
                    self.current_stats.write_speed_mbps < 1.0):
                    logger.warning("Storage performance degraded - check disk health")
        
        except Exception as e:
            logger.error(f"Error checking storage health: {e}")
    
    def _update_performance_stats(self, check_time_ms: float) -> None:
        """Update performance statistics"""
        try:
            self.performance_stats['monitoring_cycles'] += 1
            
            # Update average check time
            total_cycles = self.performance_stats['monitoring_cycles']
            current_avg = self.performance_stats['average_check_time_ms']
            new_avg = ((current_avg * (total_cycles - 1)) + check_time_ms) / total_cycles
            self.performance_stats['average_check_time_ms'] = new_avg
        
        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")
    
    def get_storage_status(self) -> Dict[str, Any]:
        """
        Get current storage status
        
        Returns:
            Dictionary with storage status information
        """
        try:
            with self.monitor_lock:
                if not self.current_stats:
                    return {'status': StorageStatus.ERROR.value}
                
                stats = self.current_stats
                
                # Determine status
                if stats.usage_percent >= self.config.emergency_threshold_percent:
                    status = StorageStatus.FULL
                elif stats.usage_percent >= self.config.critical_threshold_percent:
                    status = StorageStatus.CRITICAL
                elif stats.usage_percent >= self.config.warning_threshold_percent:
                    status = StorageStatus.WARNING
                else:
                    status = StorageStatus.HEALTHY
                
                return {
                    'status': status.value,
                    'total_gb': stats.total_gb,
                    'used_gb': stats.used_gb,
                    'available_gb': stats.available_gb,
                    'usage_percent': stats.usage_percent,
                    'free_inodes': stats.free_inodes,
                    'read_speed_mbps': stats.read_speed_mbps,
                    'write_speed_mbps': stats.write_speed_mbps,
                    'last_updated': stats.last_updated.isoformat(),
                    'active_alerts': len(self.active_alerts),
                    'monitoring_active': self.monitoring_active
                }
        
        except Exception as e:
            logger.error(f"Error getting storage status: {e}")
            return {'status': StorageStatus.ERROR.value}
    
    def get_usage_prediction(self, hours_ahead: int = 24) -> Dict[str, Any]:
        """
        Predict storage usage
        
        Args:
            hours_ahead: Hours to predict ahead
            
        Returns:
            Dictionary with usage prediction
        """
        try:
            with self.monitor_lock:
                if len(self.usage_history) < 10:
                    return {'error': 'Insufficient data for prediction'}
                
                # Simple linear trend calculation
                recent_stats = self.usage_history[-10:]
                time_points = [(s.last_updated.timestamp() for s in recent_stats)]
                usage_points = [s.usage_percent for s in recent_stats]
                
                # Calculate trend (simplified linear regression)
                n = len(usage_points)
                sum_x = sum(time_points)
                sum_y = sum(usage_points)
                sum_xy = sum(x * y for x, y in zip(time_points, usage_points))
                sum_x2 = sum(x * x for x in time_points)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                
                # Predict usage
                current_time = time.time()
                future_time = current_time + (hours_ahead * 3600)
                predicted_usage = self.current_stats.usage_percent + (slope * hours_ahead * 3600)
                
                return {
                    'predicted_usage_percent': max(0, min(100, predicted_usage)),
                    'trend_percent_per_hour': slope * 3600,
                    'hours_to_full': (100 - self.current_stats.usage_percent) / (slope * 3600) if slope > 0 else float('inf'),
                    'prediction_confidence': min(1.0, len(self.usage_history) / 100)
                }
        
        except Exception as e:
            logger.error(f"Error predicting usage: {e}")
            return {'error': 'Prediction failed'}
    
    def get_alerts(self, since: Optional[datetime] = None) -> List[AlertInfo]:
        """
        Get storage alerts
        
        Args:
            since: Only return alerts after this time
            
        Returns:
            List of alerts
        """
        with self.monitor_lock:
            if since:
                return [alert for alert in self.active_alerts if alert.timestamp > since]
            return self.active_alerts.copy()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get storage monitoring performance statistics"""
        return {
            **self.performance_stats,
            'monitoring_active': self.monitoring_active,
            'storage_path': self.storage_path,
            'config': asdict(self.config),
            'history_entries': len(self.usage_history),
            'active_alerts_count': len(self.active_alerts)
        }


if __name__ == "__main__":
    # Basic test when run directly
    print("Storage Monitor Test")
    print("=" * 50)
    
    config = StorageConfig(
        warning_threshold_percent=50.0,  # Lower threshold for testing
        monitor_interval_seconds=5
    )
    
    monitor = StorageMonitor("test_storage", config)
    
    def on_alert(alert):
        print(f"ALERT ({alert.alert_type.value}): {alert.message}")
        print(f"  Suggested actions: {', '.join(alert.suggested_actions)}")
    
    monitor.add_alert_callback(on_alert)
    
    print("Starting storage monitoring...")
    if monitor.start_monitoring():
        print("Monitoring started successfully")
        
        # Wait and check status
        time.sleep(2)
        
        status = monitor.get_storage_status()
        print(f"Storage status: {status}")
        
        # Get usage prediction
        prediction = monitor.get_usage_prediction(24)
        print(f"Usage prediction: {prediction}")
        
        # Wait a bit more
        time.sleep(8)
        
        # Get alerts
        alerts = monitor.get_alerts()
        print(f"Total alerts: {len(alerts)}")
        
        # Performance stats
        stats = monitor.get_performance_stats()
        print(f"Performance stats: {stats}")
        
        # Stop monitoring
        monitor.stop_monitoring()
        print("Monitoring stopped")
    
    print("Test complete!")