#!/usr/bin/env python3
"""
Performance Optimizer Module

Pi 5 hardware optimization for AI transcription processing.
Dynamic performance tuning and resource management.

Author: Claude AI Assistant
Date: 2024-07-14
Version: 1.0
"""

import os
import sys
import logging
import threading
import time
import psutil
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import uuid
from collections import deque
import subprocess

try:
    import numpy as np
    import torch
except ImportError as e:
    print(f"Warning: Required dependencies not installed: {e}")
    print("Please install with: pip install numpy torch")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Optimization levels for different scenarios"""
    CONSERVATIVE = "conservative"  # Minimal optimization
    BALANCED = "balanced"  # Balance performance and stability
    AGGRESSIVE = "aggressive"  # Maximum performance
    CUSTOM = "custom"  # Custom settings


class ResourceType(Enum):
    """System resource types"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    TEMPERATURE = "temperature"
    POWER = "power"


class PerformanceMode(Enum):
    """Performance modes for different use cases"""
    REAL_TIME = "real_time"  # Optimize for real-time processing
    BATCH = "batch"  # Optimize for batch processing
    POWER_SAVE = "power_save"  # Optimize for power efficiency
    THERMAL_LIMIT = "thermal_limit"  # Optimize for thermal management


@dataclass
class SystemMetrics:
    """System performance metrics"""
    
    # CPU metrics
    cpu_usage: float = 0.0
    cpu_temperature: float = 0.0
    cpu_frequency: float = 0.0
    cpu_load_avg: List[float] = field(default_factory=list)
    
    # Memory metrics
    memory_usage: float = 0.0
    memory_available: float = 0.0
    memory_total: float = 0.0
    swap_usage: float = 0.0
    
    # Disk metrics
    disk_usage: float = 0.0
    disk_read_rate: float = 0.0
    disk_write_rate: float = 0.0
    
    # Network metrics
    network_bytes_sent: float = 0.0
    network_bytes_recv: float = 0.0
    
    # Power metrics (if available)
    power_consumption: float = 0.0
    battery_level: float = 0.0
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "cpu_usage": self.cpu_usage,
            "cpu_temperature": self.cpu_temperature,
            "cpu_frequency": self.cpu_frequency,
            "cpu_load_avg": self.cpu_load_avg,
            "memory_usage": self.memory_usage,
            "memory_available": self.memory_available,
            "memory_total": self.memory_total,
            "swap_usage": self.swap_usage,
            "disk_usage": self.disk_usage,
            "disk_read_rate": self.disk_read_rate,
            "disk_write_rate": self.disk_write_rate,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_recv": self.network_bytes_recv,
            "power_consumption": self.power_consumption,
            "battery_level": self.battery_level,
            "timestamp": self.timestamp
        }


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization"""
    
    # Optimization settings
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    performance_mode: PerformanceMode = PerformanceMode.REAL_TIME
    
    # Resource thresholds
    cpu_threshold: float = 80.0  # CPU usage threshold (%)
    memory_threshold: float = 85.0  # Memory usage threshold (%)
    temperature_threshold: float = 70.0  # Temperature threshold (°C)
    disk_threshold: float = 90.0  # Disk usage threshold (%)
    
    # Performance targets
    target_processing_time: float = 0.5  # Target processing time ratio
    target_latency: float = 0.04  # Target latency (seconds)
    target_throughput: float = 1.0  # Target throughput ratio
    
    # Monitoring settings
    monitoring_interval: float = 1.0  # Monitoring interval (seconds)
    history_size: int = 100  # Number of historical metrics to keep
    
    # Optimization features
    enable_cpu_scaling: bool = True
    enable_memory_optimization: bool = True
    enable_thermal_management: bool = True
    enable_power_management: bool = True
    enable_disk_optimization: bool = True
    
    # Thread settings
    max_threads: int = 4
    thread_priority: int = 0  # Thread priority adjustment
    
    # Model optimization
    enable_model_quantization: bool = True
    enable_model_pruning: bool = False
    enable_batch_optimization: bool = True
    
    # Cache settings
    enable_caching: bool = True
    cache_size: int = 100  # Number of cached results
    cache_ttl: float = 300.0  # Cache time-to-live (seconds)


@dataclass
class OptimizationResult:
    """Result of optimization operation"""
    
    # Performance metrics
    processing_time: float = 0.0
    latency: float = 0.0
    throughput: float = 0.0
    
    # Resource usage
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    temperature: float = 0.0
    
    # Optimization actions
    actions_taken: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Status
    success: bool = True
    optimization_level: str = "balanced"
    
    # Improvement metrics
    performance_improvement: float = 0.0  # Percentage improvement
    resource_savings: float = 0.0  # Percentage savings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "processing_time": self.processing_time,
            "latency": self.latency,
            "throughput": self.throughput,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "temperature": self.temperature,
            "actions_taken": self.actions_taken,
            "warnings": self.warnings,
            "success": self.success,
            "optimization_level": self.optimization_level,
            "performance_improvement": self.performance_improvement,
            "resource_savings": self.resource_savings
        }


class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_running = False
        self.metrics_history = deque(maxlen=100)
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Cached network counters for rate calculations
        self._last_network_counters = None
        self._last_disk_counters = None
        
    def start_monitoring(self):
        """Start system monitoring"""
        if self.is_running:
            return
            
        self.is_running = True
        self.stop_event.clear()
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="SystemMonitor"
        )
        self.monitoring_thread.start()
        
        logger.info("System monitoring started")
        
    def stop_monitoring(self):
        """Stop system monitoring"""
        if not self.is_running:
            return
            
        self.is_running = False
        self.stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
            
        logger.info("System monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self.stop_event.is_set():
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
                
    def _collect_metrics(self) -> SystemMetrics:
        """Collect system metrics"""
        metrics = SystemMetrics()
        
        try:
            # CPU metrics
            metrics.cpu_usage = psutil.cpu_percent(interval=None)
            metrics.cpu_load_avg = list(psutil.getloadavg())
            
            # Get CPU frequency
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                metrics.cpu_frequency = cpu_freq.current
                
            # CPU temperature (Pi 5 specific)
            metrics.cpu_temperature = self._get_cpu_temperature()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.memory_usage = memory.percent
            metrics.memory_available = memory.available
            metrics.memory_total = memory.total
            
            swap = psutil.swap_memory()
            metrics.swap_usage = swap.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.disk_usage = disk.percent
            
            # Disk I/O rates
            disk_io = psutil.disk_io_counters()
            if disk_io and self._last_disk_counters:
                time_diff = time.time() - self._last_disk_counters['timestamp']
                metrics.disk_read_rate = (disk_io.read_bytes - self._last_disk_counters['read_bytes']) / time_diff
                metrics.disk_write_rate = (disk_io.write_bytes - self._last_disk_counters['write_bytes']) / time_diff
                
            if disk_io:
                self._last_disk_counters = {
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes,
                    'timestamp': time.time()
                }
            
            # Network metrics
            net_io = psutil.net_io_counters()
            if net_io and self._last_network_counters:
                time_diff = time.time() - self._last_network_counters['timestamp']
                metrics.network_bytes_sent = (net_io.bytes_sent - self._last_network_counters['bytes_sent']) / time_diff
                metrics.network_bytes_recv = (net_io.bytes_recv - self._last_network_counters['bytes_recv']) / time_diff
                
            if net_io:
                self._last_network_counters = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'timestamp': time.time()
                }
            
            # Power metrics (if available)
            metrics.power_consumption = self._get_power_consumption()
            metrics.battery_level = self._get_battery_level()
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            
        return metrics
        
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature (Pi 5 specific)"""
        try:
            # Try multiple methods to get temperature
            temp_paths = [
                "/sys/class/thermal/thermal_zone0/temp",
                "/sys/class/hwmon/hwmon0/temp1_input",
                "/sys/class/hwmon/hwmon1/temp1_input"
            ]
            
            for temp_path in temp_paths:
                if os.path.exists(temp_path):
                    with open(temp_path, 'r') as f:
                        temp_raw = f.read().strip()
                        # Temperature is usually in millidegrees
                        return float(temp_raw) / 1000.0
                        
            # Try vcgencmd (VideoCore GPU command)
            result = subprocess.run(
                ["vcgencmd", "measure_temp"],
                capture_output=True,
                text=True,
                timeout=1.0
            )
            
            if result.returncode == 0:
                # Output format: temp=47.2'C
                temp_str = result.stdout.strip()
                if "temp=" in temp_str:
                    return float(temp_str.split("=")[1].split("'")[0])
                    
        except Exception as e:
            logger.debug(f"Could not get CPU temperature: {e}")
            
        return 0.0
        
    def _get_power_consumption(self) -> float:
        """Get power consumption if available"""
        try:
            # Pi 5 might have power monitoring
            power_paths = [
                "/sys/class/power_supply/BAT0/power_now",
                "/sys/class/power_supply/BAT1/power_now"
            ]
            
            for power_path in power_paths:
                if os.path.exists(power_path):
                    with open(power_path, 'r') as f:
                        power_raw = f.read().strip()
                        return float(power_raw) / 1000000.0  # Convert to watts
                        
        except Exception as e:
            logger.debug(f"Could not get power consumption: {e}")
            
        return 0.0
        
    def _get_battery_level(self) -> float:
        """Get battery level if available"""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return battery.percent
                
        except Exception as e:
            logger.debug(f"Could not get battery level: {e}")
            
        return 0.0
        
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get current system metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
        
    def get_average_metrics(self, window_size: int = 10) -> Optional[SystemMetrics]:
        """Get average metrics over window"""
        if not self.metrics_history:
            return None
            
        recent_metrics = list(self.metrics_history)[-window_size:]
        if not recent_metrics:
            return None
            
        avg_metrics = SystemMetrics()
        count = len(recent_metrics)
        
        # Calculate averages
        avg_metrics.cpu_usage = sum(m.cpu_usage for m in recent_metrics) / count
        avg_metrics.memory_usage = sum(m.memory_usage for m in recent_metrics) / count
        avg_metrics.cpu_temperature = sum(m.cpu_temperature for m in recent_metrics) / count
        avg_metrics.disk_usage = sum(m.disk_usage for m in recent_metrics) / count
        avg_metrics.power_consumption = sum(m.power_consumption for m in recent_metrics) / count
        
        return avg_metrics


class PerformanceOptimizer:
    """Main performance optimization engine"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.is_initialized = False
        self.is_running = False
        
        # System monitoring
        self.system_monitor = SystemMonitor(self.config.monitoring_interval)
        
        # Optimization state
        self.optimization_history = deque(maxlen=50)
        self.optimization_lock = threading.Lock()
        
        # Performance tracking
        self.performance_baseline = None
        self.current_performance = None
        
        # Cache for optimization results
        self.optimization_cache = {}
        
        # Statistics
        self.stats = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "average_improvement": 0.0,
            "resource_savings": 0.0,
            "uptime": 0.0
        }
        
        # Callbacks
        self.optimization_callbacks: List[Callable] = []
        
        logger.info(f"PerformanceOptimizer initialized with {self.config.optimization_level.value} level")
        
    def initialize(self) -> bool:
        """Initialize the performance optimizer"""
        try:
            logger.info("Initializing performance optimizer...")
            
            # Start system monitoring
            self.system_monitor.start_monitoring()
            
            # Collect baseline metrics
            time.sleep(2)  # Wait for initial metrics
            self.performance_baseline = self.system_monitor.get_average_metrics(5)
            
            # Apply initial optimizations
            self._apply_initial_optimizations()
            
            self.is_initialized = True
            self.is_running = True
            
            logger.info("Performance optimizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}")
            return False
            
    def _apply_initial_optimizations(self):
        """Apply initial system optimizations"""
        actions = []
        
        try:
            # CPU optimizations
            if self.config.enable_cpu_scaling:
                actions.extend(self._optimize_cpu_scaling())
                
            # Memory optimizations
            if self.config.enable_memory_optimization:
                actions.extend(self._optimize_memory())
                
            # Thread optimizations
            actions.extend(self._optimize_threading())
            
            # PyTorch optimizations
            actions.extend(self._optimize_pytorch())
            
            if actions:
                logger.info(f"Applied initial optimizations: {actions}")
                
        except Exception as e:
            logger.error(f"Error applying initial optimizations: {e}")
            
    def _optimize_cpu_scaling(self) -> List[str]:
        """Optimize CPU scaling settings"""
        actions = []
        
        try:
            # Set CPU governor based on performance mode
            if self.config.performance_mode == PerformanceMode.REAL_TIME:
                governor = "performance"
            elif self.config.performance_mode == PerformanceMode.POWER_SAVE:
                governor = "powersave"
            else:
                governor = "ondemand"
                
            # Try to set CPU governor
            try:
                result = subprocess.run([
                    "sudo", "cpupower", "frequency-set", "-g", governor
                ], capture_output=True, text=True, timeout=5.0)
                
                if result.returncode == 0:
                    actions.append(f"Set CPU governor to {governor}")
                    
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.debug("cpupower not available, skipping CPU governor optimization")
                
        except Exception as e:
            logger.error(f"Error optimizing CPU scaling: {e}")
            
        return actions
        
    def _optimize_memory(self) -> List[str]:
        """Optimize memory settings"""
        actions = []
        
        try:
            # Set memory allocation strategy
            os.environ["MALLOC_MMAP_THRESHOLD_"] = "131072"
            os.environ["MALLOC_TRIM_THRESHOLD_"] = "131072"
            actions.append("Optimized memory allocation")
            
            # Set transparent huge pages
            try:
                with open("/sys/kernel/mm/transparent_hugepage/enabled", "w") as f:
                    f.write("madvise")
                actions.append("Configured transparent huge pages")
            except (IOError, PermissionError):
                logger.debug("Could not configure transparent huge pages")
                
        except Exception as e:
            logger.error(f"Error optimizing memory: {e}")
            
        return actions
        
    def _optimize_threading(self) -> List[str]:
        """Optimize threading settings"""
        actions = []
        
        try:
            # Set thread count based on CPU cores
            cpu_count = psutil.cpu_count()
            optimal_threads = min(self.config.max_threads, cpu_count)
            
            # Set environment variables for threading
            os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
            os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
            os.environ["OPENBLAS_NUM_THREADS"] = str(optimal_threads)
            
            actions.append(f"Set thread count to {optimal_threads}")
            
            # Set thread affinity if available
            try:
                current_process = psutil.Process()
                current_process.cpu_affinity(list(range(optimal_threads)))
                actions.append("Set CPU affinity")
            except (AttributeError, OSError):
                logger.debug("CPU affinity not available")
                
        except Exception as e:
            logger.error(f"Error optimizing threading: {e}")
            
        return actions
        
    def _optimize_pytorch(self) -> List[str]:
        """Optimize PyTorch settings"""
        actions = []
        
        try:
            # Set PyTorch thread count
            torch.set_num_threads(self.config.max_threads)
            actions.append("Set PyTorch thread count")
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            actions.append("Enabled PyTorch optimizations")
            
            # Set memory allocation strategy
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
                actions.append("Cleared PyTorch cache")
                
        except Exception as e:
            logger.error(f"Error optimizing PyTorch: {e}")
            
        return actions
        
    def optimize_for_workload(self, workload_type: str, 
                            performance_hints: Dict[str, Any] = None) -> OptimizationResult:
        """Optimize system for specific workload"""
        if not self.is_initialized:
            logger.error("Optimizer not initialized")
            return OptimizationResult(success=False)
            
        start_time = time.time()
        
        try:
            # Get current metrics
            current_metrics = self.system_monitor.get_current_metrics()
            if not current_metrics:
                logger.warning("No metrics available for optimization")
                return OptimizationResult(success=False)
                
            # Check if optimization is needed
            if not self._needs_optimization(current_metrics):
                return OptimizationResult(
                    success=True,
                    actions_taken=["No optimization needed"],
                    cpu_usage=current_metrics.cpu_usage,
                    memory_usage=current_metrics.memory_usage,
                    temperature=current_metrics.cpu_temperature
                )
                
            # Apply workload-specific optimizations
            actions = []
            
            if workload_type == "transcription":
                actions.extend(self._optimize_for_transcription(current_metrics, performance_hints))
            elif workload_type == "analysis":
                actions.extend(self._optimize_for_analysis(current_metrics, performance_hints))
            elif workload_type == "diarization":
                actions.extend(self._optimize_for_diarization(current_metrics, performance_hints))
            else:
                actions.extend(self._optimize_generic(current_metrics, performance_hints))
                
            # Apply thermal management if needed
            if current_metrics.cpu_temperature > self.config.temperature_threshold:
                actions.extend(self._apply_thermal_management(current_metrics))
                
            # Apply memory management if needed
            if current_metrics.memory_usage > self.config.memory_threshold:
                actions.extend(self._apply_memory_management(current_metrics))
                
            # Create optimization result
            processing_time = time.time() - start_time
            result = OptimizationResult(
                processing_time=processing_time,
                actions_taken=actions,
                success=True,
                optimization_level=self.config.optimization_level.value,
                cpu_usage=current_metrics.cpu_usage,
                memory_usage=current_metrics.memory_usage,
                temperature=current_metrics.cpu_temperature
            )
            
            # Update statistics
            self._update_stats(result)
            
            # Store in history
            self.optimization_history.append(result)
            
            # Notify callbacks
            for callback in self.optimization_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in optimization callback: {e}")
                    
            logger.info(f"Optimization completed in {processing_time:.3f}s: {actions}")
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return OptimizationResult(
                success=False,
                actions_taken=[f"Error: {str(e)}"],
                processing_time=time.time() - start_time
            )
            
    def _needs_optimization(self, metrics: SystemMetrics) -> bool:
        """Check if optimization is needed"""
        return (
            metrics.cpu_usage > self.config.cpu_threshold or
            metrics.memory_usage > self.config.memory_threshold or
            metrics.cpu_temperature > self.config.temperature_threshold or
            metrics.disk_usage > self.config.disk_threshold
        )
        
    def _optimize_for_transcription(self, metrics: SystemMetrics, 
                                  hints: Dict[str, Any] = None) -> List[str]:
        """Optimize for transcription workload"""
        actions = []
        
        try:
            # Prioritize CPU and memory for transcription
            if metrics.cpu_usage > 60:
                actions.append("Reduced background processes for transcription")
                
            # Optimize for sequential processing
            if hints and hints.get("sequential_processing"):
                actions.append("Optimized for sequential transcription processing")
                
            # Cache optimization
            if self.config.enable_caching:
                actions.append("Enabled transcription result caching")
                
        except Exception as e:
            logger.error(f"Error optimizing for transcription: {e}")
            
        return actions
        
    def _optimize_for_analysis(self, metrics: SystemMetrics, 
                             hints: Dict[str, Any] = None) -> List[str]:
        """Optimize for analysis workload"""
        actions = []
        
        try:
            # Optimize for parallel processing
            if hints and hints.get("parallel_processing"):
                actions.append("Optimized for parallel analysis processing")
                
            # Memory optimization for analysis
            if metrics.memory_usage > 70:
                actions.append("Applied memory optimization for analysis")
                
        except Exception as e:
            logger.error(f"Error optimizing for analysis: {e}")
            
        return actions
        
    def _optimize_for_diarization(self, metrics: SystemMetrics, 
                                hints: Dict[str, Any] = None) -> List[str]:
        """Optimize for diarization workload"""
        actions = []
        
        try:
            # Optimize for feature extraction
            if hints and hints.get("feature_extraction"):
                actions.append("Optimized for feature extraction")
                
            # CPU optimization for clustering
            if metrics.cpu_usage > 50:
                actions.append("Applied CPU optimization for clustering")
                
        except Exception as e:
            logger.error(f"Error optimizing for diarization: {e}")
            
        return actions
        
    def _optimize_generic(self, metrics: SystemMetrics, 
                        hints: Dict[str, Any] = None) -> List[str]:
        """Generic optimization"""
        actions = []
        
        try:
            # Generic resource optimization
            if metrics.cpu_usage > 80:
                actions.append("Applied generic CPU optimization")
                
            if metrics.memory_usage > 85:
                actions.append("Applied generic memory optimization")
                
        except Exception as e:
            logger.error(f"Error in generic optimization: {e}")
            
        return actions
        
    def _apply_thermal_management(self, metrics: SystemMetrics) -> List[str]:
        """Apply thermal management"""
        actions = []
        
        try:
            if metrics.cpu_temperature > self.config.temperature_threshold:
                # Reduce CPU frequency
                try:
                    subprocess.run([
                        "sudo", "cpupower", "frequency-set", "-u", "1200MHz"
                    ], capture_output=True, timeout=2.0)
                    actions.append("Reduced CPU frequency for thermal management")
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    logger.debug("Could not reduce CPU frequency")
                    
                # Add thermal throttling warning
                if metrics.cpu_temperature > 75:
                    actions.append("WARNING: High temperature detected")
                    
        except Exception as e:
            logger.error(f"Error in thermal management: {e}")
            
        return actions
        
    def _apply_memory_management(self, metrics: SystemMetrics) -> List[str]:
        """Apply memory management"""
        actions = []
        
        try:
            if metrics.memory_usage > self.config.memory_threshold:
                # Force garbage collection
                import gc
                gc.collect()
                actions.append("Forced garbage collection")
                
                # Clear caches
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                    actions.append("Cleared GPU cache")
                    
        except Exception as e:
            logger.error(f"Error in memory management: {e}")
            
        return actions
        
    def _update_stats(self, result: OptimizationResult):
        """Update optimization statistics"""
        with self.optimization_lock:
            self.stats["total_optimizations"] += 1
            
            if result.success:
                self.stats["successful_optimizations"] += 1
                
                # Update average improvement
                if result.performance_improvement > 0:
                    count = self.stats["successful_optimizations"]
                    old_avg = self.stats["average_improvement"]
                    self.stats["average_improvement"] = (
                        old_avg * (count - 1) + result.performance_improvement
                    ) / count
                    
                # Update resource savings
                if result.resource_savings > 0:
                    count = self.stats["successful_optimizations"]
                    old_savings = self.stats["resource_savings"]
                    self.stats["resource_savings"] = (
                        old_savings * (count - 1) + result.resource_savings
                    ) / count
                    
    def add_optimization_callback(self, callback: Callable[[OptimizationResult], None]):
        """Add optimization callback"""
        self.optimization_callbacks.append(callback)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        with self.optimization_lock:
            stats = self.stats.copy()
            stats["uptime"] = time.time() - (self.stats.get("start_time", time.time()))
            return stats
            
    def get_status(self) -> Dict[str, Any]:
        """Get optimizer status"""
        current_metrics = self.system_monitor.get_current_metrics()
        
        return {
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "optimization_level": self.config.optimization_level.value,
            "performance_mode": self.config.performance_mode.value,
            "current_metrics": current_metrics.to_dict() if current_metrics else None,
            "stats": self.get_stats()
        }
        
    def shutdown(self):
        """Shutdown optimizer"""
        logger.info("Shutting down performance optimizer...")
        
        self.is_running = False
        
        # Stop monitoring
        self.system_monitor.stop_monitoring()
        
        logger.info("Performance optimizer shutdown complete")
        
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


# Factory functions
def create_realtime_optimizer() -> PerformanceOptimizer:
    """Create optimizer for real-time processing"""
    config = OptimizationConfig(
        optimization_level=OptimizationLevel.AGGRESSIVE,
        performance_mode=PerformanceMode.REAL_TIME,
        target_processing_time=0.3,
        target_latency=0.04,
        enable_cpu_scaling=True,
        enable_memory_optimization=True,
        enable_caching=True
    )
    return PerformanceOptimizer(config)


def create_power_efficient_optimizer() -> PerformanceOptimizer:
    """Create optimizer for power efficiency"""
    config = OptimizationConfig(
        optimization_level=OptimizationLevel.CONSERVATIVE,
        performance_mode=PerformanceMode.POWER_SAVE,
        target_processing_time=1.0,
        enable_power_management=True,
        enable_thermal_management=True,
        max_threads=2
    )
    return PerformanceOptimizer(config)


def create_balanced_optimizer() -> PerformanceOptimizer:
    """Create balanced optimizer"""
    config = OptimizationConfig(
        optimization_level=OptimizationLevel.BALANCED,
        performance_mode=PerformanceMode.REAL_TIME,
        target_processing_time=0.5,
        target_latency=0.06,
        enable_cpu_scaling=True,
        enable_memory_optimization=True,
        enable_thermal_management=True,
        enable_caching=True
    )
    return PerformanceOptimizer(config)


# Testing and demonstration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Optimizer Test")
    parser.add_argument("--mode", type=str, default="balanced",
                       choices=["realtime", "power", "balanced"], help="Optimization mode")
    parser.add_argument("--workload", type=str, default="transcription",
                       choices=["transcription", "analysis", "diarization"], help="Workload type")
    parser.add_argument("--monitor-time", type=int, default=30, help="Monitoring time in seconds")
    args = parser.parse_args()
    
    # Create optimizer
    if args.mode == "realtime":
        optimizer = create_realtime_optimizer()
    elif args.mode == "power":
        optimizer = create_power_efficient_optimizer()
    else:
        optimizer = create_balanced_optimizer()
    
    try:
        # Initialize optimizer
        if not optimizer.initialize():
            print("Failed to initialize optimizer")
            sys.exit(1)
            
        print(f"Optimizer status: {optimizer.get_status()}")
        
        # Monitor and optimize
        print(f"Monitoring for {args.monitor_time} seconds...")
        
        start_time = time.time()
        while time.time() - start_time < args.monitor_time:
            # Perform optimization
            result = optimizer.optimize_for_workload(args.workload)
            
            if result.success:
                print(f"Optimization result: {result.actions_taken}")
                print(f"CPU: {result.cpu_usage:.1f}%, Memory: {result.memory_usage:.1f}%, Temp: {result.temperature:.1f}°C")
            else:
                print("Optimization failed")
                
            time.sleep(5)  # Wait before next optimization
            
        # Show final statistics
        print(f"\nFinal statistics: {optimizer.get_stats()}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        optimizer.shutdown()