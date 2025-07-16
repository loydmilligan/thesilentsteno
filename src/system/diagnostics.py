"""
Diagnostic and Troubleshooting Tools

Comprehensive diagnostic tools with automated testing, performance analysis,
and troubleshooting guides for The Silent Steno device.
"""

import os
import subprocess
import logging
import time
import json
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from enum import Enum
import psutil
import threading
from collections import defaultdict
import re


class DiagnosticSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DiagnosticCategory(Enum):
    SYSTEM = "system"
    AUDIO = "audio"
    BLUETOOTH = "bluetooth"
    NETWORK = "network"
    STORAGE = "storage"
    DATABASE = "database"
    APPLICATION = "application"
    PERFORMANCE = "performance"


@dataclass
class DiagnosticResult:
    """Result of a diagnostic test."""
    test_name: str
    category: DiagnosticCategory
    severity: DiagnosticSeverity
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_name": self.test_name,
            "category": self.category.value,
            "severity": self.severity.value,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "recommendations": self.recommendations
        }


@dataclass
class DiagnosticReport:
    """Complete diagnostic report."""
    timestamp: datetime
    duration_seconds: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    results: List[DiagnosticResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if there are critical issues."""
        return any(r.severity == DiagnosticSeverity.CRITICAL for r in self.results)
    
    @property
    def has_errors(self) -> bool:
        """Check if there are errors."""
        return any(r.severity == DiagnosticSeverity.ERROR for r in self.results)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "success_rate": self.success_rate,
            "has_critical_issues": self.has_critical_issues,
            "has_errors": self.has_errors,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary
        }


class DiagnosticTest:
    """Base class for diagnostic tests."""
    
    def __init__(self, name: str, category: DiagnosticCategory, description: str = ""):
        self.name = name
        self.category = category
        self.description = description
        self.logger = logging.getLogger(__name__)
    
    def run(self) -> DiagnosticResult:
        """Run the diagnostic test."""
        start_time = time.time()
        
        try:
            result = self._execute_test()
            result.duration_seconds = time.time() - start_time
            return result
            
        except Exception as e:
            self.logger.error(f"Diagnostic test {self.name} failed: {e}")
            return DiagnosticResult(
                test_name=self.name,
                category=self.category,
                severity=DiagnosticSeverity.ERROR,
                passed=False,
                message=f"Test execution failed: {e}",
                duration_seconds=time.time() - start_time
            )
    
    def _execute_test(self) -> DiagnosticResult:
        """Execute the actual test logic. Override in subclasses."""
        raise NotImplementedError


class SystemDiagnosticTest(DiagnosticTest):
    """System-level diagnostic tests."""
    
    def __init__(self, test_type: str):
        super().__init__(f"system_{test_type}", DiagnosticCategory.SYSTEM)
        self.test_type = test_type
    
    def _execute_test(self) -> DiagnosticResult:
        if self.test_type == "disk_space":
            return self._check_disk_space()
        elif self.test_type == "memory":
            return self._check_memory()
        elif self.test_type == "cpu":
            return self._check_cpu()
        elif self.test_type == "load":
            return self._check_system_load()
        elif self.test_type == "processes":
            return self._check_processes()
        else:
            return DiagnosticResult(
                test_name=self.name,
                category=self.category,
                severity=DiagnosticSeverity.ERROR,
                passed=False,
                message=f"Unknown system test type: {self.test_type}"
            )
    
    def _check_disk_space(self) -> DiagnosticResult:
        """Check disk space availability."""
        try:
            disk_usage = psutil.disk_usage('/')
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            free_gb = disk_usage.free / (1024 ** 3)
            
            if usage_percent > 95:
                severity = DiagnosticSeverity.CRITICAL
                message = f"Disk space critical: {usage_percent:.1f}% used, {free_gb:.1f}GB free"
                passed = False
                recommendations = ["Run storage cleanup immediately", "Remove unnecessary files"]
            elif usage_percent > 85:
                severity = DiagnosticSeverity.WARNING
                message = f"Disk space low: {usage_percent:.1f}% used, {free_gb:.1f}GB free"
                passed = False
                recommendations = ["Schedule storage cleanup", "Monitor disk usage"]
            else:
                severity = DiagnosticSeverity.INFO
                message = f"Disk space OK: {usage_percent:.1f}% used, {free_gb:.1f}GB free"
                passed = True
                recommendations = []
            
            return DiagnosticResult(
                test_name=self.name,
                category=self.category,
                severity=severity,
                passed=passed,
                message=message,
                details={
                    "usage_percent": usage_percent,
                    "free_gb": free_gb,
                    "total_gb": disk_usage.total / (1024 ** 3)
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            return DiagnosticResult(
                test_name=self.name,
                category=self.category,
                severity=DiagnosticSeverity.ERROR,
                passed=False,
                message=f"Failed to check disk space: {e}"
            )
    
    def _check_memory(self) -> DiagnosticResult:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            available_gb = memory.available / (1024 ** 3)
            
            if usage_percent > 95:
                severity = DiagnosticSeverity.CRITICAL
                message = f"Memory critical: {usage_percent:.1f}% used, {available_gb:.1f}GB available"
                passed = False
                recommendations = ["Restart high-memory processes", "Check for memory leaks"]
            elif usage_percent > 85:
                severity = DiagnosticSeverity.WARNING
                message = f"Memory high: {usage_percent:.1f}% used, {available_gb:.1f}GB available"
                passed = False
                recommendations = ["Monitor memory usage", "Consider restarting services"]
            else:
                severity = DiagnosticSeverity.INFO
                message = f"Memory OK: {usage_percent:.1f}% used, {available_gb:.1f}GB available"
                passed = True
                recommendations = []
            
            return DiagnosticResult(
                test_name=self.name,
                category=self.category,
                severity=severity,
                passed=passed,
                message=message,
                details={
                    "usage_percent": usage_percent,
                    "available_gb": available_gb,
                    "total_gb": memory.total / (1024 ** 3)
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            return DiagnosticResult(
                test_name=self.name,
                category=self.category,
                severity=DiagnosticSeverity.ERROR,
                passed=False,
                message=f"Failed to check memory: {e}"
            )
    
    def _check_cpu(self) -> DiagnosticResult:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            if cpu_percent > 90:
                severity = DiagnosticSeverity.CRITICAL
                message = f"CPU critical: {cpu_percent:.1f}% usage"
                passed = False
                recommendations = ["Check for runaway processes", "Reduce system load"]
            elif cpu_percent > 70:
                severity = DiagnosticSeverity.WARNING
                message = f"CPU high: {cpu_percent:.1f}% usage"
                passed = False
                recommendations = ["Monitor CPU usage", "Check process priorities"]
            else:
                severity = DiagnosticSeverity.INFO
                message = f"CPU OK: {cpu_percent:.1f}% usage"
                passed = True
                recommendations = []
            
            return DiagnosticResult(
                test_name=self.name,
                category=self.category,
                severity=severity,
                passed=passed,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "cpu_count": cpu_count
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            return DiagnosticResult(
                test_name=self.name,
                category=self.category,
                severity=DiagnosticSeverity.ERROR,
                passed=False,
                message=f"Failed to check CPU: {e}"
            )
    
    def _check_system_load(self) -> DiagnosticResult:
        """Check system load average."""
        try:
            load_avg = os.getloadavg()
            load_1m = load_avg[0]
            cpu_count = psutil.cpu_count()
            
            if load_1m > cpu_count * 2:
                severity = DiagnosticSeverity.CRITICAL
                message = f"System load critical: {load_1m:.2f} (CPUs: {cpu_count})"
                passed = False
                recommendations = ["Reduce system load", "Check for resource contention"]
            elif load_1m > cpu_count:
                severity = DiagnosticSeverity.WARNING
                message = f"System load high: {load_1m:.2f} (CPUs: {cpu_count})"
                passed = False
                recommendations = ["Monitor system load", "Check process priorities"]
            else:
                severity = DiagnosticSeverity.INFO
                message = f"System load OK: {load_1m:.2f} (CPUs: {cpu_count})"
                passed = True
                recommendations = []
            
            return DiagnosticResult(
                test_name=self.name,
                category=self.category,
                severity=severity,
                passed=passed,
                message=message,
                details={
                    "load_1m": load_1m,
                    "load_5m": load_avg[1],
                    "load_15m": load_avg[2],
                    "cpu_count": cpu_count
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            return DiagnosticResult(
                test_name=self.name,
                category=self.category,
                severity=DiagnosticSeverity.ERROR,
                passed=False,
                message=f"Failed to check system load: {e}"
            )
    
    def _check_processes(self) -> DiagnosticResult:
        """Check process health."""
        try:
            process_count = len(psutil.pids())
            critical_processes = ["bluetooth", "pulseaudio"]
            missing_processes = []
            
            for proc_name in critical_processes:
                found = False
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        if proc_name in proc.info['name'].lower():
                            found = True
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                if not found:
                    missing_processes.append(proc_name)
            
            if missing_processes:
                severity = DiagnosticSeverity.ERROR
                message = f"Critical processes missing: {', '.join(missing_processes)}"
                passed = False
                recommendations = [f"Start {proc} service" for proc in missing_processes]
            elif process_count > 300:
                severity = DiagnosticSeverity.WARNING
                message = f"High process count: {process_count}"
                passed = False
                recommendations = ["Check for unnecessary processes", "Monitor process spawning"]
            else:
                severity = DiagnosticSeverity.INFO
                message = f"Processes OK: {process_count} running"
                passed = True
                recommendations = []
            
            return DiagnosticResult(
                test_name=self.name,
                category=self.category,
                severity=severity,
                passed=passed,
                message=message,
                details={
                    "process_count": process_count,
                    "missing_processes": missing_processes
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            return DiagnosticResult(
                test_name=self.name,
                category=self.category,
                severity=DiagnosticSeverity.ERROR,
                passed=False,
                message=f"Failed to check processes: {e}"
            )


class AudioDiagnosticTest(DiagnosticTest):
    """Audio system diagnostic tests."""
    
    def __init__(self, test_type: str):
        super().__init__(f"audio_{test_type}", DiagnosticCategory.AUDIO)
        self.test_type = test_type
    
    def _execute_test(self) -> DiagnosticResult:
        if self.test_type == "devices":
            return self._check_audio_devices()
        elif self.test_type == "services":
            return self._check_audio_services()
        elif self.test_type == "playback":
            return self._check_audio_playback()
        else:
            return DiagnosticResult(
                test_name=self.name,
                category=self.category,
                severity=DiagnosticSeverity.ERROR,
                passed=False,
                message=f"Unknown audio test type: {self.test_type}"
            )
    
    def _check_audio_devices(self) -> DiagnosticResult:
        """Check audio device availability."""
        try:
            devices_found = []
            
            # Check /dev/snd
            if os.path.exists("/dev/snd"):
                devices_found.append("ALSA devices")
            
            # Check audio cards
            try:
                result = subprocess.run(
                    ['aplay', '-l'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    devices_found.append("Audio cards")
            except Exception:
                pass
            
            if not devices_found:
                return DiagnosticResult(
                    test_name=self.name,
                    category=self.category,
                    severity=DiagnosticSeverity.CRITICAL,
                    passed=False,
                    message="No audio devices found",
                    recommendations=["Check audio hardware", "Verify driver installation"]
                )
            else:
                return DiagnosticResult(
                    test_name=self.name,
                    category=self.category,
                    severity=DiagnosticSeverity.INFO,
                    passed=True,
                    message=f"Audio devices OK: {', '.join(devices_found)}",
                    details={"devices": devices_found}
                )
                
        except Exception as e:
            return DiagnosticResult(
                test_name=self.name,
                category=self.category,
                severity=DiagnosticSeverity.ERROR,
                passed=False,
                message=f"Failed to check audio devices: {e}"
            )
    
    def _check_audio_services(self) -> DiagnosticResult:
        """Check audio service status."""
        try:
            services_status = {}
            
            # Check PulseAudio
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if 'pulseaudio' in proc.info['name'].lower():
                        services_status['pulseaudio'] = 'running'
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if 'pulseaudio' not in services_status:
                services_status['pulseaudio'] = 'not running'
            
            # Check ALSA
            if os.path.exists("/proc/asound"):
                services_status['alsa'] = 'available'
            else:
                services_status['alsa'] = 'not available'
            
            failed_services = [name for name, status in services_status.items() if 'not' in status]
            
            if failed_services:
                return DiagnosticResult(
                    test_name=self.name,
                    category=self.category,
                    severity=DiagnosticSeverity.WARNING,
                    passed=False,
                    message=f"Audio services issues: {', '.join(failed_services)}",
                    details={"services": services_status},
                    recommendations=[f"Start {service} service" for service in failed_services]
                )
            else:
                return DiagnosticResult(
                    test_name=self.name,
                    category=self.category,
                    severity=DiagnosticSeverity.INFO,
                    passed=True,
                    message="Audio services OK",
                    details={"services": services_status}
                )
                
        except Exception as e:
            return DiagnosticResult(
                test_name=self.name,
                category=self.category,
                severity=DiagnosticSeverity.ERROR,
                passed=False,
                message=f"Failed to check audio services: {e}"
            )
    
    def _check_audio_playback(self) -> DiagnosticResult:
        """Check audio playback capability."""
        try:
            # Simple test - check if we can query audio information
            try:
                result = subprocess.run(
                    ['amixer', 'info'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return DiagnosticResult(
                        test_name=self.name,
                        category=self.category,
                        severity=DiagnosticSeverity.INFO,
                        passed=True,
                        message="Audio playback capability OK"
                    )
            except Exception:
                pass
            
            return DiagnosticResult(
                test_name=self.name,
                category=self.category,
                severity=DiagnosticSeverity.WARNING,
                passed=False,
                message="Audio playback test inconclusive",
                recommendations=["Test audio playback manually", "Check audio configuration"]
            )
            
        except Exception as e:
            return DiagnosticResult(
                test_name=self.name,
                category=self.category,
                severity=DiagnosticSeverity.ERROR,
                passed=False,
                message=f"Failed to check audio playback: {e}"
            )


class PerformanceAnalyzer:
    """Analyzes system performance and identifies bottlenecks."""
    
    def __init__(self, duration_seconds: int = 30):
        self.duration_seconds = duration_seconds
        self.logger = logging.getLogger(__name__)
    
    def analyze_performance(self) -> DiagnosticResult:
        """Analyze system performance over time."""
        start_time = time.time()
        
        try:
            # Collect performance data
            cpu_samples = []
            memory_samples = []
            disk_samples = []
            
            sample_interval = 1.0
            samples_count = int(self.duration_seconds / sample_interval)
            
            for i in range(samples_count):
                cpu_samples.append(psutil.cpu_percent(interval=sample_interval))
                memory_samples.append(psutil.virtual_memory().percent)
                
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    disk_samples.append(disk_io.read_bytes + disk_io.write_bytes)
            
            # Analyze data
            avg_cpu = sum(cpu_samples) / len(cpu_samples)
            max_cpu = max(cpu_samples)
            avg_memory = sum(memory_samples) / len(memory_samples)
            max_memory = max(memory_samples)
            
            # Identify issues
            issues = []
            recommendations = []
            
            if avg_cpu > 70:
                issues.append(f"High average CPU usage: {avg_cpu:.1f}%")
                recommendations.append("Identify CPU-intensive processes")
            
            if max_cpu > 95:
                issues.append(f"CPU spikes detected: {max_cpu:.1f}%")
                recommendations.append("Monitor for CPU spikes")
            
            if avg_memory > 80:
                issues.append(f"High average memory usage: {avg_memory:.1f}%")
                recommendations.append("Identify memory-intensive processes")
            
            if max_memory > 95:
                issues.append(f"Memory spikes detected: {max_memory:.1f}%")
                recommendations.append("Monitor for memory leaks")
            
            severity = DiagnosticSeverity.INFO
            passed = True
            
            if issues:
                if any("spike" in issue for issue in issues):
                    severity = DiagnosticSeverity.WARNING
                    passed = False
                if avg_cpu > 80 or avg_memory > 85:
                    severity = DiagnosticSeverity.ERROR
                    passed = False
            
            message = f"Performance analysis complete: {len(issues)} issues found"
            if issues:
                message += f" - {', '.join(issues)}"
            
            return DiagnosticResult(
                test_name="performance_analysis",
                category=DiagnosticCategory.PERFORMANCE,
                severity=severity,
                passed=passed,
                message=message,
                details={
                    "duration_seconds": self.duration_seconds,
                    "avg_cpu": avg_cpu,
                    "max_cpu": max_cpu,
                    "avg_memory": avg_memory,
                    "max_memory": max_memory,
                    "issues": issues
                },
                recommendations=recommendations,
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return DiagnosticResult(
                test_name="performance_analysis",
                category=DiagnosticCategory.PERFORMANCE,
                severity=DiagnosticSeverity.ERROR,
                passed=False,
                message=f"Performance analysis failed: {e}",
                duration_seconds=time.time() - start_time
            )


class LogAnalyzer:
    """Analyzes log files for errors and patterns."""
    
    def __init__(self, log_dir: str = "/home/mmariani/projects/thesilentsteno/logs"):
        self.log_dir = Path(log_dir)
        self.logger = logging.getLogger(__name__)
    
    def analyze_logs(self, hours: int = 24) -> DiagnosticResult:
        """Analyze log files for errors and issues."""
        start_time = time.time()
        
        try:
            if not self.log_dir.exists():
                return DiagnosticResult(
                    test_name="log_analysis",
                    category=DiagnosticCategory.APPLICATION,
                    severity=DiagnosticSeverity.WARNING,
                    passed=False,
                    message="Log directory not found",
                    recommendations=["Check logging configuration"]
                )
            
            # Find log files
            log_files = list(self.log_dir.glob("*.log"))
            if not log_files:
                return DiagnosticResult(
                    test_name="log_analysis",
                    category=DiagnosticCategory.APPLICATION,
                    severity=DiagnosticSeverity.WARNING,
                    passed=False,
                    message="No log files found",
                    recommendations=["Check if logging is working"]
                )
            
            # Analyze each log file
            error_count = 0
            warning_count = 0
            critical_count = 0
            recent_errors = []
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            for log_file in log_files:
                try:
                    with open(log_file, 'r') as f:
                        for line in f:
                            # Simple log analysis - looking for common patterns
                            if 'ERROR' in line or 'CRITICAL' in line:
                                error_count += 1
                                if len(recent_errors) < 10:
                                    recent_errors.append(line.strip())
                            elif 'WARNING' in line:
                                warning_count += 1
                            elif 'CRITICAL' in line:
                                critical_count += 1
                                
                except Exception as e:
                    self.logger.warning(f"Failed to read log file {log_file}: {e}")
            
            # Determine severity
            if critical_count > 0:
                severity = DiagnosticSeverity.CRITICAL
                passed = False
            elif error_count > 10:
                severity = DiagnosticSeverity.ERROR
                passed = False
            elif error_count > 0 or warning_count > 20:
                severity = DiagnosticSeverity.WARNING
                passed = False
            else:
                severity = DiagnosticSeverity.INFO
                passed = True
            
            message = f"Log analysis: {error_count} errors, {warning_count} warnings, {critical_count} critical"
            
            recommendations = []
            if error_count > 0:
                recommendations.append("Review recent error messages")
            if warning_count > 50:
                recommendations.append("Investigate frequent warnings")
            if critical_count > 0:
                recommendations.append("Address critical issues immediately")
            
            return DiagnosticResult(
                test_name="log_analysis",
                category=DiagnosticCategory.APPLICATION,
                severity=severity,
                passed=passed,
                message=message,
                details={
                    "log_files_count": len(log_files),
                    "error_count": error_count,
                    "warning_count": warning_count,
                    "critical_count": critical_count,
                    "recent_errors": recent_errors[:5]  # First 5 errors
                },
                recommendations=recommendations,
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return DiagnosticResult(
                test_name="log_analysis",
                category=DiagnosticCategory.APPLICATION,
                severity=DiagnosticSeverity.ERROR,
                passed=False,
                message=f"Log analysis failed: {e}",
                duration_seconds=time.time() - start_time
            )


class Diagnostics:
    """Main diagnostic system coordinator."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.performance_analyzer = PerformanceAnalyzer()
        self.log_analyzer = LogAnalyzer()
        self._lock = threading.RLock()
    
    def run_diagnostics(self, categories: List[DiagnosticCategory] = None) -> DiagnosticReport:
        """Run comprehensive diagnostics."""
        start_time = time.time()
        
        try:
            with self._lock:
                # Default to all categories if none specified
                if categories is None:
                    categories = list(DiagnosticCategory)
                
                results = []
                
                # System tests
                if DiagnosticCategory.SYSTEM in categories:
                    system_tests = ["disk_space", "memory", "cpu", "load", "processes"]
                    for test_type in system_tests:
                        test = SystemDiagnosticTest(test_type)
                        results.append(test.run())
                
                # Audio tests
                if DiagnosticCategory.AUDIO in categories:
                    audio_tests = ["devices", "services", "playback"]
                    for test_type in audio_tests:
                        test = AudioDiagnosticTest(test_type)
                        results.append(test.run())
                
                # Performance analysis
                if DiagnosticCategory.PERFORMANCE in categories:
                    results.append(self.performance_analyzer.analyze_performance())
                
                # Log analysis
                if DiagnosticCategory.APPLICATION in categories:
                    results.append(self.log_analyzer.analyze_logs())
                
                # Create report
                passed_tests = len([r for r in results if r.passed])
                failed_tests = len([r for r in results if not r.passed])
                
                report = DiagnosticReport(
                    timestamp=datetime.now(),
                    duration_seconds=time.time() - start_time,
                    total_tests=len(results),
                    passed_tests=passed_tests,
                    failed_tests=failed_tests,
                    results=results
                )
                
                # Generate summary
                report.summary = self._generate_summary(results)
                
                return report
                
        except Exception as e:
            self.logger.error(f"Diagnostic run failed: {e}")
            return DiagnosticReport(
                timestamp=datetime.now(),
                duration_seconds=time.time() - start_time,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                results=[],
                summary={"error": str(e)}
            )
    
    def _generate_summary(self, results: List[DiagnosticResult]) -> Dict[str, Any]:
        """Generate diagnostic summary."""
        summary = {
            "categories": defaultdict(int),
            "severities": defaultdict(int),
            "top_issues": [],
            "recommendations": []
        }
        
        for result in results:
            summary["categories"][result.category.value] += 1
            summary["severities"][result.severity.value] += 1
            
            if not result.passed:
                summary["top_issues"].append({
                    "test": result.test_name,
                    "message": result.message,
                    "severity": result.severity.value
                })
            
            summary["recommendations"].extend(result.recommendations)
        
        # Remove duplicates from recommendations
        summary["recommendations"] = list(set(summary["recommendations"]))
        
        return dict(summary)
    
    def get_quick_diagnostics(self) -> DiagnosticReport:
        """Run quick diagnostics for essential systems."""
        essential_categories = [
            DiagnosticCategory.SYSTEM,
            DiagnosticCategory.AUDIO,
            DiagnosticCategory.APPLICATION
        ]
        return self.run_diagnostics(essential_categories)


# Factory functions
def create_diagnostics() -> Diagnostics:
    """Create a diagnostics instance."""
    return Diagnostics()


def run_diagnostics(categories: List[DiagnosticCategory] = None) -> DiagnosticReport:
    """Run comprehensive diagnostics."""
    diagnostics = create_diagnostics()
    return diagnostics.run_diagnostics(categories)


def analyze_performance(duration_seconds: int = 30) -> DiagnosticResult:
    """Analyze system performance."""
    analyzer = PerformanceAnalyzer(duration_seconds)
    return analyzer.analyze_performance()


def analyze_logs(hours: int = 24) -> DiagnosticResult:
    """Analyze log files."""
    analyzer = LogAnalyzer()
    return analyzer.analyze_logs(hours)