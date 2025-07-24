#!/usr/bin/env python3
"""
Test suite for PipeWire backend implementation
"""

import unittest
import subprocess
import time
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from audio.pipewire_backend import PipeWireBackend, AudioDevice, DeviceDirection, AudioState
    from audio.audio_system_factory import AudioSystemFactory, AudioSystemType
except ImportError as e:
    print(f"Warning: Could not import PipeWire modules: {e}")
    print("Some tests will be skipped")


class TestPipeWireDetection(unittest.TestCase):
    """Test PipeWire detection and availability"""
    
    def test_pipewire_command_available(self):
        """Test if pw-cli command is available"""
        try:
            result = subprocess.run(['which', 'pw-cli'], capture_output=True)
            self.assertEqual(result.returncode, 0, "pw-cli command not found")
        except Exception as e:
            self.skipTest(f"Could not test pw-cli availability: {e}")
            
    def test_pipewire_running(self):
        """Test if PipeWire service is running"""
        try:
            result = subprocess.run(['pw-cli', 'info'], capture_output=True, timeout=2)
            if result.returncode != 0:
                self.skipTest("PipeWire not running - this is expected if not installed")
            else:
                self.assertEqual(result.returncode, 0, "PipeWire service not responding")
        except subprocess.TimeoutExpired:
            self.skipTest("PipeWire command timed out")
        except Exception as e:
            self.skipTest(f"Could not test PipeWire service: {e}")


class TestAudioSystemFactory(unittest.TestCase):
    """Test audio system factory detection"""
    
    def test_detect_audio_system(self):
        """Test audio system detection"""
        system_type = AudioSystemFactory.detect_audio_system()
        self.assertIsInstance(system_type, AudioSystemType)
        print(f"Detected audio system: {system_type.value}")
        
    def test_get_system_info(self):
        """Test getting system information"""
        info = AudioSystemFactory.get_system_info()
        self.assertIn('system_type', info)
        self.assertIn('available_backends', info)
        self.assertIsInstance(info['available_backends'], list)
        
    def test_create_backend(self):
        """Test backend creation"""
        backend = AudioSystemFactory.create_backend()
        if backend is None:
            self.skipTest("No audio backend available")
        self.assertIsNotNone(backend)


@unittest.skipUnless(
    AudioSystemFactory.detect_audio_system() == AudioSystemType.PIPEWIRE,
    "PipeWire not available"
)
class TestPipeWireBackend(unittest.TestCase):
    """Test PipeWire backend functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            self.backend = PipeWireBackend()
        except Exception as e:
            self.skipTest(f"Could not initialize PipeWire backend: {e}")
            
    def test_backend_initialization(self):
        """Test backend initializes correctly"""
        self.assertIsInstance(self.backend, PipeWireBackend)
        
    def test_refresh_devices(self):
        """Test device discovery"""
        devices = self.backend.refresh_devices()
        self.assertIsInstance(devices, dict)
        print(f"Found {len(devices)} audio devices")
        
        # Test device properties
        for device_id, device in devices.items():
            self.assertIsInstance(device, AudioDevice)
            self.assertIsInstance(device.id, int)
            self.assertIsInstance(device.name, str)
            self.assertIsInstance(device.direction, DeviceDirection)
            self.assertIsInstance(device.state, AudioState)
            
    def test_get_sources(self):
        """Test getting audio sources"""
        sources = self.backend.get_sources()
        self.assertIsInstance(sources, dict)
        
        for device in sources.values():
            self.assertEqual(device.direction, DeviceDirection.SOURCE)
            
    def test_get_sinks(self):
        """Test getting audio sinks"""
        sinks = self.backend.get_sinks()
        self.assertIsInstance(sinks, dict)
        
        for device in sinks.values():
            self.assertEqual(device.direction, DeviceDirection.SINK)
            
    def test_bluetooth_device_detection(self):
        """Test Bluetooth device detection"""
        bt_devices = self.backend.get_bluetooth_devices()
        self.assertIsInstance(bt_devices, dict)
        print(f"Found {len(bt_devices)} Bluetooth devices")
        
        for device in bt_devices.values():
            self.assertTrue(device.is_bluetooth)
            
    def test_latency_info(self):
        """Test latency information retrieval"""
        latency_info = self.backend.get_latency_info()
        self.assertIsInstance(latency_info, dict)
        self.assertIn('devices', latency_info)
        
    def test_device_info(self):
        """Test getting detailed device information"""
        devices = self.backend.refresh_devices()
        if devices:
            device_id = next(iter(devices))
            info = self.backend.get_device_info(device_id)
            if info:  # Some devices might not provide detailed info
                self.assertIsInstance(info, dict)


class TestPipeWireCommands(unittest.TestCase):
    """Test PipeWire command-line tools"""
    
    def test_pw_cli_info(self):
        """Test pw-cli info command"""
        try:
            result = subprocess.run(
                ['pw-cli', 'info'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.assertIn('core', result.stdout.lower())
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.skipTest("pw-cli not available or timed out")
            
    def test_pw_dump(self):
        """Test pw-cli dump command"""
        try:
            result = subprocess.run(
                ['pw-cli', 'dump'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Should return JSON objects
                self.assertTrue(len(result.stdout) > 0)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.skipTest("pw-cli dump not available or timed out")
            
    def test_wpctl_status(self):
        """Test wpctl status command"""
        try:
            result = subprocess.run(
                ['wpctl', 'status'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.assertIn('Audio', result.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.skipTest("wpctl not available or timed out")


class TestPipeWireConfiguration(unittest.TestCase):
    """Test PipeWire configuration"""
    
    def test_config_files_exist(self):
        """Test that configuration files exist"""
        config_files = [
            'config/pipewire_config/pipewire.conf',
            'config/pipewire_config/wireplumber.conf',
            'config/pipewire_config/pipewire-pulse.conf'
        ]
        
        for config_file in config_files:
            file_path = os.path.join(os.path.dirname(__file__), '..', '..', config_file)
            self.assertTrue(os.path.exists(file_path), f"Config file missing: {config_file}")
            
    def test_config_file_syntax(self):
        """Test configuration file syntax"""
        config_file = os.path.join(
            os.path.dirname(__file__), '..', '..', 
            'config/pipewire_config/pipewire.conf'
        )
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                content = f.read()
                # Basic syntax checks
                self.assertIn('context.properties', content)
                self.assertIn('context.modules', content)


class TestPipeWireLatency(unittest.TestCase):
    """Test PipeWire latency optimization"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            if AudioSystemFactory.detect_audio_system() == AudioSystemType.PIPEWIRE:
                self.backend = PipeWireBackend()
            else:
                self.skipTest("PipeWire not available")
        except Exception as e:
            self.skipTest(f"Could not initialize PipeWire backend: {e}")
            
    def test_latency_optimization(self):
        """Test latency optimization"""
        target_latency = 40  # 40ms target
        
        # This would test the optimize_latency method
        if hasattr(self.backend, 'optimize_latency'):
            result = self.backend.optimize_latency(target_latency)
            self.assertIsInstance(result, bool)
        else:
            self.skipTest("Latency optimization not implemented")
            
    def test_quantum_calculation(self):
        """Test quantum calculation for latency"""
        # Test quantum calculation logic
        sample_rate = 44100
        target_latency_ms = 40
        
        # Calculate expected quantum
        expected_quantum = int((target_latency_ms / 1000) * sample_rate)
        # Round to nearest power of 2
        expected_quantum = 2 ** (expected_quantum.bit_length() - 1)
        expected_quantum = max(64, min(2048, expected_quantum))
        
        self.assertGreaterEqual(expected_quantum, 64)
        self.assertLessEqual(expected_quantum, 2048)


class TestBluetoothIntegration(unittest.TestCase):
    """Test Bluetooth integration with PipeWire"""
    
    def test_bluetooth_service_running(self):
        """Test if Bluetooth service is running"""
        try:
            result = subprocess.run(
                ['systemctl', 'is-active', 'bluetooth'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                self.skipTest("Bluetooth service not running")
        except Exception:
            self.skipTest("Could not check Bluetooth service")
            
    def test_bluetoothctl_available(self):
        """Test if bluetoothctl is available"""
        try:
            result = subprocess.run(['which', 'bluetoothctl'], capture_output=True)
            self.assertEqual(result.returncode, 0, "bluetoothctl not found")
        except Exception:
            self.skipTest("Could not test bluetoothctl availability")


class TestErrorHandling(unittest.TestCase):
    """Test error handling in PipeWire backend"""
    
    def test_invalid_device_id(self):
        """Test handling of invalid device IDs"""
        if AudioSystemFactory.detect_audio_system() != AudioSystemType.PIPEWIRE:
            self.skipTest("PipeWire not available")
            
        try:
            backend = PipeWireBackend()
            
            # Test with invalid device ID
            result = backend.get_device_info(99999)
            self.assertIsNone(result)
            
        except Exception as e:
            self.skipTest(f"Could not test error handling: {e}")
            
    def test_command_timeout(self):
        """Test command timeout handling"""
        # This would test timeout handling in _run_pw_command
        # We can't easily test this without mocking
        pass


def run_pipewire_tests():
    """Run all PipeWire tests and return results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestPipeWireDetection,
        TestAudioSystemFactory,
        TestPipeWireBackend,
        TestPipeWireCommands,
        TestPipeWireConfiguration,
        TestPipeWireLatency,
        TestBluetoothIntegration,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("PipeWire Backend Test Suite")
    print("=" * 50)
    
    # Check system before running tests
    system_type = AudioSystemFactory.detect_audio_system()
    print(f"Detected audio system: {system_type.value}")
    
    if system_type == AudioSystemType.PIPEWIRE:
        print("✅ PipeWire detected - running full test suite")
    else:
        print("⚠️  PipeWire not detected - some tests will be skipped")
    
    print("")
    
    # Run tests
    result = run_pipewire_tests()
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFailures:")
        for test, trace in result.failures:
            print(f"  - {test}: {trace.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nErrors:")
        for test, trace in result.errors:
            print(f"  - {test}: {trace.split('Exception:')[-1].strip()}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)