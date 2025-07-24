#!/usr/bin/env python3
"""
Audio Performance Benchmark Tool for The Silent Steno

Compares PipeWire vs PulseAudio performance for audio forwarding scenarios.
"""

import sys
import os
import time
import json
import subprocess
import psutil
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from audio.audio_system_factory import AudioSystemFactory, AudioSystemType
except ImportError:
    print("Warning: Could not import audio system factory")


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test"""
    test_name: str
    audio_system: str
    duration_seconds: float
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SystemMetrics:
    """System resource metrics during benchmark"""
    cpu_percent_avg: float
    cpu_percent_max: float
    memory_percent_avg: float
    memory_mb_used: float
    audio_process_cpu: float
    audio_process_memory_mb: float


class AudioBenchmark:
    """Audio performance benchmark runner"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.audio_system = AudioSystemFactory.detect_audio_system()
        self.backend = None
        
        try:
            self.backend = AudioSystemFactory.create_backend()
        except Exception as e:
            print(f"Warning: Could not create audio backend: {e}")
    
    def run_device_discovery_benchmark(self, iterations: int = 10) -> BenchmarkResult:
        """Benchmark audio device discovery performance"""
        print(f"🔍 Device Discovery Benchmark ({iterations} iterations)")
        
        if not self.backend:
            return BenchmarkResult(
                test_name="device_discovery",
                audio_system=self.audio_system.value,
                duration_seconds=0,
                success=False,
                errors=["No audio backend available"]
            )
        
        times = []
        device_counts = []
        start_time = time.time()
        
        try:
            for i in range(iterations):
                iter_start = time.time()
                devices = self.backend.refresh_devices()
                iter_time = time.time() - iter_start
                
                times.append(iter_time)
                device_counts.append(len(devices))
                
                print(f"  Iteration {i+1}: {iter_time:.3f}s ({len(devices)} devices)")
        
        except Exception as e:
            return BenchmarkResult(
                test_name="device_discovery",
                audio_system=self.audio_system.value,
                duration_seconds=time.time() - start_time,
                success=False,
                errors=[str(e)]
            )
        
        return BenchmarkResult(
            test_name="device_discovery",
            audio_system=self.audio_system.value,
            duration_seconds=time.time() - start_time,
            success=True,
            metrics={
                "avg_discovery_time": statistics.mean(times),
                "min_discovery_time": min(times),
                "max_discovery_time": max(times),
                "avg_device_count": statistics.mean(device_counts),
                "iterations": iterations
            }
        )
    
    def run_latency_benchmark(self) -> BenchmarkResult:
        """Benchmark audio latency measurement"""
        print("⚡ Audio Latency Benchmark")
        
        if not self.backend:
            return BenchmarkResult(
                test_name="latency_measurement",
                audio_system=self.audio_system.value,
                duration_seconds=0,
                success=False,
                errors=["No audio backend available"]
            )
        
        start_time = time.time()
        
        try:
            if hasattr(self.backend, 'get_latency_info'):
                latency_info = self.backend.get_latency_info()
                
                metrics = {
                    "quantum": latency_info.get("quantum", 0),
                    "rate": latency_info.get("rate", 0)
                }
                
                # Calculate theoretical latency
                if metrics["quantum"] and metrics["rate"]:
                    theoretical_latency = (metrics["quantum"] / metrics["rate"]) * 1000
                    metrics["theoretical_latency_ms"] = theoretical_latency
                    print(f"  Theoretical latency: {theoretical_latency:.2f}ms")
                
                return BenchmarkResult(
                    test_name="latency_measurement",
                    audio_system=self.audio_system.value,
                    duration_seconds=time.time() - start_time,
                    success=True,
                    metrics=metrics
                )
            else:
                # Fallback for PulseAudio
                return self._pulseaudio_latency_benchmark(start_time)
                
        except Exception as e:
            return BenchmarkResult(
                test_name="latency_measurement",
                audio_system=self.audio_system.value,
                duration_seconds=time.time() - start_time,
                success=False,
                errors=[str(e)]
            )
    
    def _pulseaudio_latency_benchmark(self, start_time: float) -> BenchmarkResult:
        """Benchmark PulseAudio latency using pactl"""
        try:
            # Get sink latency information
            result = subprocess.run(
                ['pactl', 'list', 'sinks'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                latencies = []
                for line in result.stdout.split('\n'):
                    if 'Latency:' in line and 'configured' not in line:
                        # Extract latency value
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'ms' in part:
                                try:
                                    latency = float(parts[i-1])
                                    latencies.append(latency)
                                except:
                                    pass
                
                metrics = {}
                if latencies:
                    metrics = {
                        "avg_latency_ms": statistics.mean(latencies),
                        "min_latency_ms": min(latencies),
                        "max_latency_ms": max(latencies),
                        "latency_count": len(latencies)
                    }
                    print(f"  Average latency: {metrics['avg_latency_ms']:.2f}ms")
                
                return BenchmarkResult(
                    test_name="latency_measurement",
                    audio_system=self.audio_system.value,
                    duration_seconds=time.time() - start_time,
                    success=True,
                    metrics=metrics
                )
            
        except Exception as e:
            pass
        
        return BenchmarkResult(
            test_name="latency_measurement",
            audio_system=self.audio_system.value,
            duration_seconds=time.time() - start_time,
            success=False,
            errors=["Could not measure latency"]
        )
    
    def run_bluetooth_discovery_benchmark(self) -> BenchmarkResult:
        """Benchmark Bluetooth device discovery"""
        print("📡 Bluetooth Discovery Benchmark")
        
        start_time = time.time()
        
        try:
            bt_manager = AudioSystemFactory.create_bluetooth_manager()
            if not bt_manager:
                return BenchmarkResult(
                    test_name="bluetooth_discovery",
                    audio_system=self.audio_system.value,
                    duration_seconds=time.time() - start_time,
                    success=False,
                    errors=["No Bluetooth manager available"]
                )
            
            # Time Bluetooth device discovery
            discovery_start = time.time()
            bt_devices = bt_manager.refresh_bluetooth_devices()
            discovery_time = time.time() - discovery_start
            
            print(f"  Found {len(bt_devices)} Bluetooth devices in {discovery_time:.3f}s")
            
            # Count devices by type
            paired_count = sum(1 for d in bt_devices.values() if d.paired)
            connected_count = sum(1 for d in bt_devices.values() if d.connected)
            audio_count = sum(1 for d in bt_devices.values() if d.audio_connected)
            
            return BenchmarkResult(
                test_name="bluetooth_discovery",
                audio_system=self.audio_system.value,
                duration_seconds=time.time() - start_time,
                success=True,
                metrics={
                    "discovery_time_seconds": discovery_time,
                    "total_devices": len(bt_devices),
                    "paired_devices": paired_count,
                    "connected_devices": connected_count,
                    "audio_devices": audio_count
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="bluetooth_discovery",
                audio_system=self.audio_system.value,
                duration_seconds=time.time() - start_time,
                success=False,
                errors=[str(e)]
            )
    
    def run_system_resource_benchmark(self, duration_seconds: int = 10) -> BenchmarkResult:
        """Benchmark system resource usage during audio operations"""
        print(f"💻 System Resource Benchmark ({duration_seconds}s)")
        
        start_time = time.time()
        
        try:
            # Find audio processes
            audio_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                if any(name in proc.info['name'].lower() 
                       for name in ['pipewire', 'pulseaudio', 'wireplumber']):
                    audio_processes.append(proc)
            
            # Monitor system resources
            cpu_samples = []
            memory_samples = []
            audio_cpu_samples = []
            audio_memory_samples = []
            
            end_time = start_time + duration_seconds
            sample_count = 0
            
            while time.time() < end_time:
                # Overall system metrics
                cpu_samples.append(psutil.cpu_percent(interval=0.1))
                memory_samples.append(psutil.virtual_memory().percent)
                
                # Audio process metrics
                audio_cpu = 0
                audio_memory = 0
                
                for proc in audio_processes:
                    try:
                        proc_info = proc.as_dict(['cpu_percent', 'memory_info'])
                        audio_cpu += proc_info['cpu_percent']
                        audio_memory += proc_info['memory_info'].rss / 1024 / 1024  # MB
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                audio_cpu_samples.append(audio_cpu)
                audio_memory_samples.append(audio_memory)
                
                # Trigger some audio activity
                if self.backend and sample_count % 20 == 0:  # Every 2 seconds
                    try:
                        self.backend.refresh_devices()
                    except:
                        pass
                
                sample_count += 1
                time.sleep(0.1)
            
            # Calculate metrics
            metrics = {
                "duration_seconds": duration_seconds,
                "sample_count": sample_count,
                "cpu_percent_avg": statistics.mean(cpu_samples),
                "cpu_percent_max": max(cpu_samples),
                "memory_percent_avg": statistics.mean(memory_samples),
                "memory_percent_max": max(memory_samples),
                "audio_cpu_avg": statistics.mean(audio_cpu_samples) if audio_cpu_samples else 0,
                "audio_cpu_max": max(audio_cpu_samples) if audio_cpu_samples else 0,
                "audio_memory_avg_mb": statistics.mean(audio_memory_samples) if audio_memory_samples else 0,
                "audio_process_count": len(audio_processes)
            }
            
            print(f"  Avg CPU: {metrics['cpu_percent_avg']:.1f}%")
            print(f"  Avg Memory: {metrics['memory_percent_avg']:.1f}%")
            print(f"  Audio processes CPU: {metrics['audio_cpu_avg']:.1f}%")
            
            return BenchmarkResult(
                test_name="system_resources",
                audio_system=self.audio_system.value,
                duration_seconds=time.time() - start_time,
                success=True,
                metrics=metrics
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="system_resources",
                audio_system=self.audio_system.value,
                duration_seconds=time.time() - start_time,
                success=False,
                errors=[str(e)]
            )
    
    def run_audio_forwarding_benchmark(self) -> BenchmarkResult:
        """Benchmark audio forwarding setup time"""
        print("🔄 Audio Forwarding Benchmark")
        
        start_time = time.time()
        
        if not self.backend:
            return BenchmarkResult(
                test_name="audio_forwarding",
                audio_system=self.audio_system.value,
                duration_seconds=0,
                success=False,
                errors=["No audio backend available"]
            )
        
        try:
            # Get available sources and sinks
            sources = self.backend.get_sources()
            sinks = self.backend.get_sinks()
            
            print(f"  Available sources: {len(sources)}")
            print(f"  Available sinks: {len(sinks)}")
            
            # Find Bluetooth devices
            bt_sources = [d for d in sources.values() if d.is_bluetooth]
            bt_sinks = [d for d in sinks.values() if d.is_bluetooth]
            
            metrics = {
                "total_sources": len(sources),
                "total_sinks": len(sinks),
                "bluetooth_sources": len(bt_sources),
                "bluetooth_sinks": len(bt_sinks),
                "forwarding_possible": len(bt_sources) > 0 and len(bt_sinks) > 0
            }
            
            # Test loopback creation if devices available
            if bt_sources and bt_sinks:
                source = bt_sources[0]
                sink = bt_sinks[0]
                
                loopback_start = time.time()
                link_id = self.backend.create_loopback(source.id, sink.id, 40)
                loopback_time = time.time() - loopback_start
                
                metrics["loopback_creation_time"] = loopback_time
                metrics["loopback_created"] = link_id is not None
                
                if link_id:
                    print(f"  Loopback created in {loopback_time:.3f}s")
                    
                    # Clean up
                    time.sleep(1)
                    self.backend.remove_loopback(link_id)
                else:
                    print("  Failed to create loopback")
            
            return BenchmarkResult(
                test_name="audio_forwarding",
                audio_system=self.audio_system.value,
                duration_seconds=time.time() - start_time,
                success=True,
                metrics=metrics
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="audio_forwarding",
                audio_system=self.audio_system.value,
                duration_seconds=time.time() - start_time,
                success=False,
                errors=[str(e)]
            )
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmark tests"""
        print(f"🚀 Running Audio Performance Benchmarks")
        print(f"Audio System: {self.audio_system.value}")
        print("=" * 60)
        
        benchmarks = [
            self.run_device_discovery_benchmark,
            self.run_latency_benchmark,
            self.run_bluetooth_discovery_benchmark,
            self.run_system_resource_benchmark,
            self.run_audio_forwarding_benchmark
        ]
        
        for benchmark in benchmarks:
            result = benchmark()
            self.results.append(result)
            
            if result.success:
                print(f"✅ {result.test_name}")
            else:
                print(f"❌ {result.test_name}: {', '.join(result.errors)}")
            print()
        
        return self.results
    
    def generate_report(self) -> Dict:
        """Generate comprehensive benchmark report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "audio_system": self.audio_system.value,
            "system_info": self._get_system_info(),
            "benchmark_results": [
                {
                    "test_name": r.test_name,
                    "success": r.success,
                    "duration_seconds": r.duration_seconds,
                    "metrics": r.metrics,
                    "errors": r.errors,
                    "timestamp": r.timestamp
                }
                for r in self.results
            ],
            "summary": self._generate_summary()
        }
        
        return report
    
    def _get_system_info(self) -> Dict:
        """Get system information"""
        try:
            system_info = AudioSystemFactory.get_system_info()
            
            # Add hardware info
            system_info.update({
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "python_version": sys.version
            })
            
            return system_info
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_summary(self) -> Dict:
        """Generate benchmark summary"""
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]
        
        summary = {
            "total_tests": len(self.results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "overall_success_rate": len(successful_tests) / len(self.results) if self.results else 0
        }
        
        # Performance highlights
        if successful_tests:
            discovery_results = [r for r in successful_tests if r.test_name == "device_discovery"]
            if discovery_results:
                avg_time = discovery_results[0].metrics.get("avg_discovery_time", 0)
                summary["device_discovery_avg_ms"] = avg_time * 1000
            
            latency_results = [r for r in successful_tests if r.test_name == "latency_measurement"]
            if latency_results:
                latency = latency_results[0].metrics.get("theoretical_latency_ms") or \
                         latency_results[0].metrics.get("avg_latency_ms")
                if latency:
                    summary["audio_latency_ms"] = latency
        
        return summary
    
    def save_report(self, filename: Optional[str] = None) -> str:
        """Save benchmark report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audio_benchmark_{self.audio_system.value}_{timestamp}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), "..", "tests", "performance", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filepath


def main():
    """Main benchmark runner"""
    print("🎵 The Silent Steno - Audio Performance Benchmark")
    print("=" * 60)
    
    # Run benchmarks
    benchmark = AudioBenchmark()
    results = benchmark.run_all_benchmarks()
    
    # Generate and save report
    report_file = benchmark.save_report()
    
    # Print summary
    print("=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    
    report = benchmark.generate_report()
    summary = report["summary"]
    
    print(f"Audio System: {report['audio_system']}")
    print(f"Tests Run: {summary['total_tests']}")
    print(f"Successful: {summary['successful_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['overall_success_rate']:.1%}")
    
    if "device_discovery_avg_ms" in summary:
        print(f"Device Discovery: {summary['device_discovery_avg_ms']:.2f}ms avg")
    
    if "audio_latency_ms" in summary:
        print(f"Audio Latency: {summary['audio_latency_ms']:.2f}ms")
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # Performance assessment
    print("\n🎯 Performance Assessment:")
    
    success_rate = summary['overall_success_rate']
    if success_rate >= 0.8:
        print("✅ Excellent - System performing well")
    elif success_rate >= 0.6:
        print("⚠️  Good - Minor issues detected")
    else:
        print("❌ Poor - Significant issues found")
    
    # Recommendations
    print("\n💡 Recommendations:")
    
    latency = summary.get("audio_latency_ms")
    if latency:
        if latency <= 40:
            print("✅ Audio latency within target (<40ms)")
        elif latency <= 80:
            print("⚠️  Audio latency acceptable but could be improved")
        else:
            print("❌ Audio latency too high - optimization needed")
    
    if summary['failed_tests'] > 0:
        print("• Check system logs for failed tests")
        print("• Consider reinstalling audio system components")
    
    return 0 if summary['overall_success_rate'] >= 0.8 else 1


if __name__ == '__main__':
    sys.exit(main())