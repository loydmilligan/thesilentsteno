#!/usr/bin/env python3
"""
Baseline Performance Testing Script for The Silent Steno
Tests current PulseAudio performance before PipeWire migration
"""

import subprocess
import time
import psutil
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import statistics

class BaselinePerformanceTester:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "audio_server": self.detect_audio_server(),
            "tests": {}
        }
        
    def detect_audio_server(self) -> str:
        """Detect current audio server"""
        try:
            # Check for PipeWire
            result = subprocess.run(['pw-cli', 'info'], 
                                  capture_output=True, timeout=2)
            if result.returncode == 0:
                return "pipewire"
        except:
            pass
            
        try:
            # Check for PulseAudio
            result = subprocess.run(['pactl', 'info'], 
                                  capture_output=True, timeout=2)
            if result.returncode == 0:
                return "pulseaudio"
        except:
            pass
            
        return "unknown"
    
    def test_audio_latency(self) -> Dict:
        """Test audio latency using pactl"""
        print("Testing audio latency...")
        latencies = []
        
        try:
            # Get sink latency
            result = subprocess.run(
                ['pactl', 'list', 'sinks'],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Latency:' in line and 'configured' not in line:
                        # Extract latency value in ms
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'ms' in part:
                                try:
                                    latency = float(parts[i-1])
                                    latencies.append(latency)
                                except:
                                    pass
                                    
            return {
                "average_latency_ms": statistics.mean(latencies) if latencies else None,
                "min_latency_ms": min(latencies) if latencies else None,
                "max_latency_ms": max(latencies) if latencies else None,
                "measurements": len(latencies)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def test_bluetooth_devices(self) -> Dict:
        """Test Bluetooth device detection"""
        print("Testing Bluetooth device detection...")
        
        try:
            # List Bluetooth devices
            result = subprocess.run(
                ['bluetoothctl', 'devices'],
                capture_output=True, text=True, timeout=5
            )
            
            devices = []
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        devices.append(line)
                        
            # Check PulseAudio Bluetooth modules
            pa_result = subprocess.run(
                ['pactl', 'list', 'modules'],
                capture_output=True, text=True
            )
            
            bluetooth_modules = []
            if pa_result.returncode == 0:
                in_module = False
                current_module = {}
                
                for line in pa_result.stdout.split('\n'):
                    if line.startswith('Module #'):
                        if current_module and 'bluetooth' in current_module.get('name', ''):
                            bluetooth_modules.append(current_module)
                        current_module = {'id': line.split('#')[1]}
                        in_module = True
                    elif in_module and 'Name:' in line:
                        current_module['name'] = line.split(':', 1)[1].strip()
                    elif in_module and 'Argument:' in line:
                        current_module['args'] = line.split(':', 1)[1].strip()
                        
                if current_module and 'bluetooth' in current_module.get('name', ''):
                    bluetooth_modules.append(current_module)
                    
            return {
                "paired_devices": len(devices),
                "device_list": devices[:5],  # First 5 devices
                "bluetooth_modules": len(bluetooth_modules),
                "module_list": bluetooth_modules
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def test_cpu_usage(self, duration: int = 10) -> Dict:
        """Test CPU usage during audio processing"""
        print(f"Testing CPU usage for {duration} seconds...")
        
        cpu_samples = []
        memory_samples = []
        
        # Find audio-related processes
        audio_processes = []
        for proc in psutil.process_iter(['pid', 'name']):
            if any(name in proc.info['name'].lower() 
                   for name in ['pulseaudio', 'pipewire', 'bluetooth']):
                audio_processes.append(proc.info['pid'])
                
        start_time = time.time()
        while time.time() - start_time < duration:
            # Overall CPU
            cpu_samples.append(psutil.cpu_percent(interval=0.1))
            
            # Memory usage
            memory_samples.append(psutil.virtual_memory().percent)
            
            time.sleep(0.5)
            
        # Get process-specific stats
        process_stats = {}
        for pid in audio_processes:
            try:
                proc = psutil.Process(pid)
                process_stats[proc.name()] = {
                    "cpu_percent": proc.cpu_percent(),
                    "memory_mb": proc.memory_info().rss / 1024 / 1024
                }
            except:
                pass
                
        return {
            "overall_cpu_average": statistics.mean(cpu_samples),
            "overall_cpu_max": max(cpu_samples),
            "memory_average": statistics.mean(memory_samples),
            "process_stats": process_stats,
            "sample_count": len(cpu_samples)
        }
    
    def test_audio_sources(self) -> Dict:
        """Test available audio sources"""
        print("Testing audio sources...")
        
        try:
            result = subprocess.run(
                ['pactl', 'list', 'short', 'sources'],
                capture_output=True, text=True
            )
            
            sources = []
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            sources.append({
                                "id": parts[0],
                                "name": parts[1],
                                "module": parts[2] if len(parts) > 2 else None,
                                "state": parts[4] if len(parts) > 4 else None
                            })
                            
            return {
                "source_count": len(sources),
                "sources": sources,
                "bluetooth_sources": [s for s in sources if 'bluez' in s['name']]
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def test_audio_sinks(self) -> Dict:
        """Test available audio sinks"""
        print("Testing audio sinks...")
        
        try:
            result = subprocess.run(
                ['pactl', 'list', 'short', 'sinks'],
                capture_output=True, text=True
            )
            
            sinks = []
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            sinks.append({
                                "id": parts[0],
                                "name": parts[1],
                                "module": parts[2] if len(parts) > 2 else None,
                                "state": parts[4] if len(parts) > 4 else None
                            })
                            
            return {
                "sink_count": len(sinks),
                "sinks": sinks,
                "bluetooth_sinks": [s for s in sinks if 'bluez' in s['name']]
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def test_module_loading_time(self) -> Dict:
        """Test time to load Bluetooth module"""
        print("Testing module loading time...")
        
        try:
            # First, unload if exists
            subprocess.run(
                ['pactl', 'unload-module', 'module-bluetooth-discover'],
                capture_output=True
            )
            time.sleep(1)
            
            # Measure loading time
            start_time = time.time()
            result = subprocess.run(
                ['pactl', 'load-module', 'module-bluetooth-discover'],
                capture_output=True, text=True
            )
            load_time = time.time() - start_time
            
            if result.returncode == 0:
                return {
                    "module_load_time_ms": load_time * 1000,
                    "module_id": result.stdout.strip()
                }
            else:
                return {"error": f"Failed to load module: {result.stderr}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def run_all_tests(self) -> Dict:
        """Run all baseline tests"""
        print("=" * 50)
        print("The Silent Steno - Baseline Performance Testing")
        print(f"Audio Server: {self.results['audio_server']}")
        print("=" * 50)
        print()
        
        # Run tests
        self.results["tests"]["audio_latency"] = self.test_audio_latency()
        self.results["tests"]["bluetooth_devices"] = self.test_bluetooth_devices()
        self.results["tests"]["cpu_usage"] = self.test_cpu_usage()
        self.results["tests"]["audio_sources"] = self.test_audio_sources()
        self.results["tests"]["audio_sinks"] = self.test_audio_sinks()
        self.results["tests"]["module_loading"] = self.test_module_loading_time()
        
        return self.results
    
    def save_results(self, filename: Optional[str] = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"baseline_performance_{timestamp}.json"
            
        filepath = os.path.join("tests", "performance", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\nResults saved to: {filepath}")
        return filepath
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 50)
        print("PERFORMANCE TEST SUMMARY")
        print("=" * 50)
        
        tests = self.results["tests"]
        
        # Audio latency
        if "audio_latency" in tests and "average_latency_ms" in tests["audio_latency"]:
            avg_latency = tests["audio_latency"]["average_latency_ms"]
            if avg_latency:
                print(f"Average Audio Latency: {avg_latency:.2f} ms")
                
        # Bluetooth devices
        if "bluetooth_devices" in tests:
            bt = tests["bluetooth_devices"]
            print(f"Bluetooth Devices: {bt.get('paired_devices', 0)}")
            print(f"Bluetooth Modules: {bt.get('bluetooth_modules', 0)}")
            
        # CPU usage
        if "cpu_usage" in tests:
            cpu = tests["cpu_usage"]
            print(f"Average CPU Usage: {cpu.get('overall_cpu_average', 0):.1f}%")
            print(f"Max CPU Usage: {cpu.get('overall_cpu_max', 0):.1f}%")
            
        # Audio devices
        if "audio_sources" in tests:
            print(f"Audio Sources: {tests['audio_sources'].get('source_count', 0)}")
        if "audio_sinks" in tests:
            print(f"Audio Sinks: {tests['audio_sinks'].get('sink_count', 0)}")
            
        # Module loading
        if "module_loading" in tests and "module_load_time_ms" in tests["module_loading"]:
            load_time = tests["module_loading"]["module_load_time_ms"]
            print(f"Module Load Time: {load_time:.1f} ms")


if __name__ == "__main__":
    tester = BaselinePerformanceTester()
    results = tester.run_all_tests()
    tester.print_summary()
    tester.save_results()