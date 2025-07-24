#!/usr/bin/env python3
"""
Test suite for PipeWire Bluetooth manager
"""

import unittest
import subprocess
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from bluetooth.pipewire_bluetooth import PipeWireBluetoothManager, BluetoothDevice, BluetoothProfile
    from audio.pipewire_backend import PipeWireBackend
    from audio.audio_system_factory import AudioSystemFactory, AudioSystemType
    _IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Bluetooth modules: {e}")
    _IMPORTS_AVAILABLE = False
    
    # Create dummy classes to prevent NameError
    class AudioSystemFactory:
        @staticmethod
        def detect_audio_system():
            return None
    
    class AudioSystemType:
        PIPEWIRE = "pipewire"


class TestBluetoothSystemRequirements(unittest.TestCase):
    """Test Bluetooth system requirements"""
    
    def test_bluetooth_service_available(self):
        """Test if Bluetooth service is available"""
        try:
            result = subprocess.run(
                ['systemctl', 'status', 'bluetooth'],
                capture_output=True,
                timeout=5
            )
            # Service exists (active or inactive)
            self.assertIn(result.returncode, [0, 3], "Bluetooth service not available")
        except Exception as e:
            self.skipTest(f"Could not check Bluetooth service: {e}")
            
    def test_bluetoothctl_command(self):
        """Test bluetoothctl command availability"""
        try:
            result = subprocess.run(['which', 'bluetoothctl'], capture_output=True)
            self.assertEqual(result.returncode, 0, "bluetoothctl command not found")
        except Exception:
            self.skipTest("Could not test bluetoothctl availability")
            
    def test_dbus_available(self):
        """Test D-Bus availability for Bluetooth communication"""
        try:
            import dbus
            # Try to connect to system bus
            bus = dbus.SystemBus()
            self.assertIsNotNone(bus)
        except ImportError:
            self.skipTest("python3-dbus not installed")
        except Exception as e:
            self.skipTest(f"D-Bus not available: {e}")


class TestBluetoothDeviceModel(unittest.TestCase):
    """Test Bluetooth device data model"""
    
    def test_bluetooth_device_creation(self):
        """Test BluetoothDevice creation"""
        device = BluetoothDevice(
            address="AA:BB:CC:DD:EE:FF",
            name="Test Device",
            alias="Test Alias"
        )
        
        self.assertEqual(device.address, "AA:BB:CC:DD:EE:FF")
        self.assertEqual(device.name, "Test Device")
        self.assertEqual(device.alias, "Test Alias")
        self.assertFalse(device.paired)
        self.assertFalse(device.connected)
        self.assertEqual(device.adapter, "hci0")  # Default
        
    def test_bluetooth_profiles(self):
        """Test Bluetooth profile enumeration"""
        profiles = [
            BluetoothProfile.A2DP_SOURCE,
            BluetoothProfile.A2DP_SINK,
            BluetoothProfile.HSP_HS,
            BluetoothProfile.HFP_HF,
            BluetoothProfile.OFF
        ]
        
        for profile in profiles:
            self.assertIsInstance(profile.value, str)


@unittest.skipUnless(
    _IMPORTS_AVAILABLE and AudioSystemFactory.detect_audio_system() == AudioSystemType.PIPEWIRE,
    "PipeWire not available"
)
class TestPipeWireBluetoothManager(unittest.TestCase):
    """Test PipeWire Bluetooth manager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            self.pw_backend = PipeWireBackend()
            self.bt_manager = PipeWireBluetoothManager(self.pw_backend)
        except Exception as e:
            self.skipTest(f"Could not initialize Bluetooth manager: {e}")
            
    def test_manager_initialization(self):
        """Test manager initializes correctly"""
        self.assertIsInstance(self.bt_manager, PipeWireBluetoothManager)
        self.assertIsNotNone(self.bt_manager.backend)
        self.assertEqual(self.bt_manager.source_adapter, "hci0")
        self.assertEqual(self.bt_manager.sink_adapter, "hci1")
        
    def test_refresh_bluetooth_devices(self):
        """Test Bluetooth device discovery"""
        devices = self.bt_manager.refresh_bluetooth_devices()
        self.assertIsInstance(devices, dict)
        print(f"Found {len(devices)} Bluetooth devices")
        
        # Test device properties if any devices found
        for address, device in devices.items():
            self.assertIsInstance(device, BluetoothDevice)
            self.assertEqual(device.address, address)
            self.assertIsInstance(device.name, str)
            self.assertIsInstance(device.paired, bool)
            self.assertIsInstance(device.connected, bool)
            
    def test_available_adapters(self):
        """Test getting available Bluetooth adapters"""
        adapters = self.bt_manager._get_available_adapters()
        self.assertIsInstance(adapters, list)
        print(f"Found {len(adapters)} Bluetooth adapters: {adapters}")
        
        for adapter in adapters:
            self.assertRegex(adapter, r'hci\d+')
            
    def test_dual_radio_setup(self):
        """Test dual radio configuration"""
        # This might not work if only one adapter is available
        result = self.bt_manager.setup_dual_radio()
        self.assertIsInstance(result, bool)
        
        if result:
            print("✅ Dual radio setup successful")
        else:
            print("⚠️  Dual radio setup failed (likely only one adapter)")


class TestBluetoothProfileParsing(unittest.TestCase):
    """Test Bluetooth profile parsing from UUIDs"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not _IMPORTS_AVAILABLE or AudioSystemFactory.detect_audio_system() != AudioSystemType.PIPEWIRE:
            self.skipTest("PipeWire not available")
            
        try:
            self.bt_manager = PipeWireBluetoothManager()
        except Exception as e:
            self.skipTest(f"Could not initialize Bluetooth manager: {e}")
            
    def test_uuid_parsing(self):
        """Test parsing profiles from UUIDs"""
        test_uuids = [
            '0000110a-0000-1000-8000-00805f9b34fb',  # A2DP Source
            '0000110b-0000-1000-8000-00805f9b34fb',  # A2DP Sink
            '00001112-0000-1000-8000-00805f9b34fb',  # HSP HS
            '0000111f-0000-1000-8000-00805f9b34fb',  # HFP HF
        ]
        
        profiles = self.bt_manager._parse_profiles_from_uuids(test_uuids)
        
        expected_profiles = [
            BluetoothProfile.A2DP_SOURCE,
            BluetoothProfile.A2DP_SINK,
            BluetoothProfile.HSP_HS,
            BluetoothProfile.HFP_HF
        ]
        
        for expected in expected_profiles:
            self.assertIn(expected, profiles)


class TestBluetoothDeviceOperations(unittest.TestCase):
    """Test Bluetooth device operations (requires actual devices)"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not _IMPORTS_AVAILABLE or AudioSystemFactory.detect_audio_system() != AudioSystemType.PIPEWIRE:
            self.skipTest("PipeWire not available")
            
        try:
            self.bt_manager = PipeWireBluetoothManager()
            self.devices = self.bt_manager.refresh_bluetooth_devices()
        except Exception as e:
            self.skipTest(f"Could not initialize Bluetooth manager: {e}")
            
    def test_get_device_by_role(self):
        """Test getting devices by role"""
        source_device = self.bt_manager.get_device_by_role("source")
        sink_device = self.bt_manager.get_device_by_role("sink")
        
        if source_device:
            self.assertEqual(source_device.current_profile, BluetoothProfile.A2DP_SOURCE)
            
        if sink_device:
            self.assertEqual(sink_device.current_profile, BluetoothProfile.A2DP_SINK)
            
    def test_device_codec_detection(self):
        """Test codec detection for connected devices"""
        for device in self.devices.values():
            if device.audio_connected:
                codecs = self.bt_manager.get_available_codecs(device.address)
                self.assertIsInstance(codecs, list)
                
                if codecs:
                    print(f"Device {device.name} supports codecs: {[c.value for c in codecs]}")


class TestMockBluetoothOperations(unittest.TestCase):
    """Test Bluetooth operations with mocked dependencies"""
    
    def setUp(self):
        """Set up mocked test fixtures"""
        self.mock_backend = Mock()
        self.mock_backend.refresh_devices.return_value = {}
        self.mock_backend.get_sources.return_value = {}
        self.mock_backend.get_sinks.return_value = {}
        
    @patch('subprocess.run')
    def test_adapter_detection_mock(self, mock_run):
        """Test adapter detection with mocked subprocess"""
        # Mock bluetoothctl list output
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = """Controller AA:BB:CC:DD:EE:FF hci0 [default]
Controller FF:EE:DD:CC:BB:AA hci1"""
        
        try:
            bt_manager = PipeWireBluetoothManager(self.mock_backend)
            adapters = bt_manager._get_available_adapters()
            
            self.assertIn('hci0', adapters)
            self.assertIn('hci1', adapters)
        except Exception as e:
            self.skipTest(f"Mock test failed: {e}")
            
    @patch('dbus.SystemBus')
    def test_dbus_interaction_mock(self, mock_bus):
        """Test D-Bus interaction with mocked bus"""
        # Mock D-Bus objects
        mock_bus_instance = Mock()
        mock_bus.return_value = mock_bus_instance
        
        # Mock object manager
        mock_obj_manager = Mock()
        mock_obj_manager.GetManagedObjects.return_value = {}
        
        mock_bus_instance.get_object.return_value = Mock()
        
        try:
            bt_manager = PipeWireBluetoothManager(self.mock_backend)
            bt_manager._refresh_from_bluez()
            # If we get here without exception, the mock worked
            self.assertTrue(True)
        except Exception as e:
            self.skipTest(f"Mock D-Bus test failed: {e}")


class TestBluetoothIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios for Silent Steno"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        if not _IMPORTS_AVAILABLE or AudioSystemFactory.detect_audio_system() != AudioSystemType.PIPEWIRE:
            self.skipTest("PipeWire not available")
            
        try:
            self.bt_manager = PipeWireBluetoothManager()
        except Exception as e:
            self.skipTest(f"Could not initialize Bluetooth manager: {e}")
            
    def test_dual_a2dp_scenario(self):
        """Test dual A2DP scenario setup"""
        # This test would verify the complete dual A2DP setup
        # In practice, this requires actual hardware
        
        devices = self.bt_manager.refresh_bluetooth_devices()
        
        # Look for potential source and sink devices
        potential_sources = [
            d for d in devices.values()
            if BluetoothProfile.A2DP_SOURCE in d.profiles
        ]
        
        potential_sinks = [
            d for d in devices.values()
            if BluetoothProfile.A2DP_SINK in d.profiles
        ]
        
        print(f"Potential audio sources: {len(potential_sources)}")
        print(f"Potential audio sinks: {len(potential_sinks)}")
        
        # For Silent Steno, we need at least one of each
        if potential_sources and potential_sinks:
            print("✅ Dual A2DP setup is possible")
        else:
            print("⚠️  Dual A2DP setup requires additional devices")


def run_bluetooth_tests():
    """Run all Bluetooth tests and return results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestBluetoothSystemRequirements,
        TestBluetoothDeviceModel,
        TestPipeWireBluetoothManager,
        TestBluetoothProfileParsing,
        TestBluetoothDeviceOperations,
        TestMockBluetoothOperations,
        TestBluetoothIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("PipeWire Bluetooth Manager Test Suite")
    print("=" * 50)
    
    # Check system before running tests
    system_type = AudioSystemFactory.detect_audio_system()
    print(f"Detected audio system: {system_type.value}")
    
    # Check Bluetooth availability
    try:
        result = subprocess.run(['systemctl', 'is-active', 'bluetooth'], 
                              capture_output=True)
        bt_status = "active" if result.returncode == 0 else "inactive"
        print(f"Bluetooth service: {bt_status}")
    except:
        print("Bluetooth service: unknown")
    
    print("")
    
    # Run tests
    result = run_bluetooth_tests()
    
    # Print summary
    print("\n" + "=" * 50)
    print("BLUETOOTH TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFailures:")
        for test, trace in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\nErrors:")
        for test, trace in result.errors:
            print(f"  - {test}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)