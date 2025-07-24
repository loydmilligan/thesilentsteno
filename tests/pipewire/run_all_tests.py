#!/usr/bin/env python3
"""
Comprehensive test runner for PipeWire integration
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from audio.audio_system_factory import AudioSystemFactory, AudioSystemType
except ImportError:
    print("Warning: Could not import audio system factory")


def run_system_checks():
    """Run system prerequisite checks"""
    print("🔍 System Prerequisite Checks")
    print("=" * 40)
    
    checks = {
        "pipewire_installed": False,
        "pipewire_running": False,
        "bluetooth_available": False,
        "python_modules": False
    }
    
    # Check PipeWire installation
    try:
        result = subprocess.run(['which', 'pw-cli'], capture_output=True)
        checks["pipewire_installed"] = result.returncode == 0
        print(f"PipeWire installed: {'✅' if checks['pipewire_installed'] else '❌'}")
    except:
        print("PipeWire installed: ❌")
    
    # Check PipeWire running
    try:
        result = subprocess.run(['pw-cli', 'info'], capture_output=True, timeout=2)
        checks["pipewire_running"] = result.returncode == 0
        print(f"PipeWire running: {'✅' if checks['pipewire_running'] else '❌'}")
    except:
        print("PipeWire running: ❌")
    
    # Check Bluetooth
    try:
        result = subprocess.run(['systemctl', 'is-active', 'bluetooth'], 
                              capture_output=True)
        checks["bluetooth_available"] = result.returncode == 0
        print(f"Bluetooth service: {'✅' if checks['bluetooth_available'] else '❌'}")
    except:
        print("Bluetooth service: ❌")
    
    # Check Python modules
    try:
        from audio.pipewire_backend import PipeWireBackend
        from bluetooth.pipewire_bluetooth import PipeWireBluetoothManager
        checks["python_modules"] = True
        print("Python modules: ✅")
    except ImportError as e:
        print(f"Python modules: ❌ ({e})")
    
    return checks


def run_unit_tests():
    """Run unit tests"""
    print("\n🧪 Unit Tests")
    print("=" * 40)
    
    test_results = {}
    
    # Run backend tests
    print("Running PipeWire backend tests...")
    try:
        result = subprocess.run([
            sys.executable, 
            'test_pipewire_backend.py'
        ], cwd=os.path.dirname(__file__), capture_output=True, text=True)
        
        test_results["backend_tests"] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
        print(f"Backend tests: {'✅' if result.returncode == 0 else '❌'}")
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            
    except Exception as e:
        test_results["backend_tests"] = {"error": str(e)}
        print(f"Backend tests: ❌ ({e})")
    
    # Run Bluetooth tests
    print("Running Bluetooth manager tests...")
    try:
        result = subprocess.run([
            sys.executable, 
            'test_bluetooth_manager.py'
        ], cwd=os.path.dirname(__file__), capture_output=True, text=True)
        
        test_results["bluetooth_tests"] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
        print(f"Bluetooth tests: {'✅' if result.returncode == 0 else '❌'}")
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            
    except Exception as e:
        test_results["bluetooth_tests"] = {"error": str(e)}
        print(f"Bluetooth tests: ❌ ({e})")
    
    return test_results


def run_integration_tests():
    """Run integration tests"""
    print("\n🔗 Integration Tests")
    print("=" * 40)
    
    integration_results = {}
    
    # Test audio system detection
    try:
        system_type = AudioSystemFactory.detect_audio_system()
        integration_results["audio_system_detection"] = {
            "system": system_type.value,
            "success": True
        }
        print(f"Audio system detection: ✅ ({system_type.value})")
    except Exception as e:
        integration_results["audio_system_detection"] = {
            "error": str(e),
            "success": False
        }
        print(f"Audio system detection: ❌ ({e})")
    
    # Test backend creation
    try:
        backend = AudioSystemFactory.create_backend()
        if backend:
            integration_results["backend_creation"] = {"success": True}
            print("Backend creation: ✅")
            
            # Test device discovery
            try:
                devices = backend.refresh_devices()
                integration_results["device_discovery"] = {
                    "device_count": len(devices),
                    "success": True
                }
                print(f"Device discovery: ✅ ({len(devices)} devices)")
            except Exception as e:
                integration_results["device_discovery"] = {
                    "error": str(e),
                    "success": False
                }
                print(f"Device discovery: ❌ ({e})")
        else:
            integration_results["backend_creation"] = {"success": False}
            print("Backend creation: ❌")
    except Exception as e:
        integration_results["backend_creation"] = {
            "error": str(e),
            "success": False
        }
        print(f"Backend creation: ❌ ({e})")
    
    # Test Bluetooth manager
    try:
        bt_manager = AudioSystemFactory.create_bluetooth_manager()
        if bt_manager:
            integration_results["bluetooth_manager"] = {"success": True}
            print("Bluetooth manager: ✅")
            
            # Test device refresh
            try:
                bt_devices = bt_manager.refresh_bluetooth_devices()
                integration_results["bluetooth_devices"] = {
                    "device_count": len(bt_devices),
                    "success": True
                }
                print(f"Bluetooth devices: ✅ ({len(bt_devices)} devices)")
            except Exception as e:
                integration_results["bluetooth_devices"] = {
                    "error": str(e),
                    "success": False
                }
                print(f"Bluetooth devices: ❌ ({e})")
        else:
            integration_results["bluetooth_manager"] = {"success": False}
            print("Bluetooth manager: ❌")
    except Exception as e:
        integration_results["bluetooth_manager"] = {
            "error": str(e),
            "success": False
        }
        print(f"Bluetooth manager: ❌ ({e})")
    
    return integration_results


def run_performance_tests():
    """Run basic performance tests"""
    print("\n⚡ Performance Tests")
    print("=" * 40)
    
    performance_results = {}
    
    try:
        backend = AudioSystemFactory.create_backend()
        if not backend:
            print("Performance tests: ❌ (No backend available)")
            return {"error": "No backend available"}
        
        # Test device refresh performance
        start_time = time.time()
        for i in range(5):
            devices = backend.refresh_devices()
        refresh_time = (time.time() - start_time) / 5
        
        performance_results["device_refresh"] = {
            "avg_time_seconds": refresh_time,
            "device_count": len(devices)
        }
        
        print(f"Device refresh: ✅ ({refresh_time:.3f}s avg)")
        
        # Test latency info retrieval
        start_time = time.time()
        latency_info = backend.get_latency_info()
        latency_time = time.time() - start_time
        
        performance_results["latency_info"] = {
            "time_seconds": latency_time,
            "quantum": latency_info.get("quantum"),
            "rate": latency_info.get("rate")
        }
        
        print(f"Latency info: ✅ ({latency_time:.3f}s)")
        
        if latency_info.get("quantum") and latency_info.get("rate"):
            calculated_latency = (latency_info["quantum"] / latency_info["rate"]) * 1000
            print(f"Current latency: ~{calculated_latency:.1f}ms")
            performance_results["calculated_latency_ms"] = calculated_latency
        
    except Exception as e:
        performance_results["error"] = str(e)
        print(f"Performance tests: ❌ ({e})")
    
    return performance_results


def generate_test_report(checks, unit_results, integration_results, performance_results):
    """Generate comprehensive test report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "system_checks": checks,
        "unit_tests": unit_results,
        "integration_tests": integration_results,
        "performance_tests": performance_results,
        "summary": {
            "total_checks": len(checks),
            "passed_checks": sum(1 for v in checks.values() if v),
            "overall_success": all(checks.values())
        }
    }
    
    # Calculate test success rates
    unit_success = sum(1 for test in unit_results.values() 
                      if isinstance(test, dict) and test.get("returncode") == 0)
    unit_total = len([test for test in unit_results.values() 
                     if isinstance(test, dict) and "returncode" in test])
    
    integration_success = sum(1 for test in integration_results.values() 
                             if isinstance(test, dict) and test.get("success"))
    integration_total = len([test for test in integration_results.values() 
                            if isinstance(test, dict) and "success" in test])
    
    report["summary"].update({
        "unit_tests_passed": unit_success,
        "unit_tests_total": unit_total,
        "integration_tests_passed": integration_success,
        "integration_tests_total": integration_total
    })
    
    return report


def save_test_report(report):
    """Save test report to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pipewire_test_report_{timestamp}.json"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    
    return filepath


def main():
    """Main test runner"""
    print("🎵 PipeWire Integration Test Suite")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("")
    
    # Run all test categories
    checks = run_system_checks()
    unit_results = run_unit_tests()
    integration_results = run_integration_tests()
    performance_results = run_performance_tests()
    
    # Generate and save report
    report = generate_test_report(checks, unit_results, integration_results, performance_results)
    report_file = save_test_report(report)
    
    # Print final summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    summary = report["summary"]
    print(f"System checks: {summary['passed_checks']}/{summary['total_checks']} passed")
    print(f"Unit tests: {summary['unit_tests_passed']}/{summary['unit_tests_total']} passed")
    print(f"Integration tests: {summary['integration_tests_passed']}/{summary['integration_tests_total']} passed")
    
    overall_success = (
        summary['passed_checks'] == summary['total_checks'] and
        summary['unit_tests_passed'] == summary['unit_tests_total'] and
        summary['integration_tests_passed'] == summary['integration_tests_total']
    )
    
    print(f"\n{'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    print(f"\nDetailed report saved to: {report_file}")
    
    # Recommendations
    print("\n📋 Recommendations:")
    if not checks["pipewire_installed"]:
        print("  • Install PipeWire: ./scripts/install_pipewire.sh")
    if not checks["pipewire_running"]:
        print("  • Start PipeWire: systemctl --user start pipewire")
    if not checks["bluetooth_available"]:
        print("  • Start Bluetooth: sudo systemctl start bluetooth")
    if not checks["python_modules"]:
        print("  • Check Python module imports")
    
    if overall_success:
        print("  • System is ready for PipeWire operation!")
    
    return 0 if overall_success else 1


if __name__ == '__main__':
    sys.exit(main())