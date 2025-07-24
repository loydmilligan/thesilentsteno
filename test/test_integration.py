#!/usr/bin/env python3
"""
Test script for Task-6.1 Application Integration Layer implementation.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_core_imports():
    """Test that all core modules can be imported."""
    print("Testing core module imports...")
    
    try:
        from src.core import (
            SilentStenoApp, create_application, start_application,
            ApplicationController, EventBus, ConfigManager, 
            PerformanceMonitor, ErrorHandler, ComponentRegistry
        )
        print("✓ All core imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_application_creation():
    """Test application creation and basic functionality."""
    print("\nTesting application creation...")
    
    try:
        from src.core import create_application
        
        # Create application
        app = create_application("config/app_config.json")
        print("✓ Application created successfully")
        
        # Test initialization
        app.initialize()
        print("✓ Application initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Application creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_event_system():
    """Test event system functionality."""
    print("\nTesting event system...")
    
    try:
        from src.core.events import create_event_bus, Event, publish_event, subscribe_to_event
        
        # Create event bus
        event_bus = create_event_bus()
        print("✓ Event bus created")
        
        # Test event publishing
        events_received = []
        
        def test_handler(event):
            events_received.append(event)
        
        # Subscribe to events
        subscription = event_bus.subscribe("test.event", test_handler)
        print("✓ Event subscription created")
        
        # Publish event
        test_event = Event("test.event", {"message": "Hello World"})
        event_bus.publish(test_event)
        
        # Give time for event processing
        import time
        time.sleep(0.1)
        
        if events_received:
            print("✓ Event received successfully")
        else:
            print("✗ Event not received")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Event system error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_system():
    """Test configuration management."""
    print("\nTesting configuration system...")
    
    try:
        from src.core.config import load_config, get_config
        
        # Load configuration
        config = load_config("config/app_config.json")
        print("✓ Configuration loaded")
        
        # Test config access
        app_name = get_config("application.name")
        if app_name == "SilentSteno":
            print("✓ Configuration access working")
        else:
            print(f"✗ Unexpected config value: {app_name}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration system error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_registry():
    """Test component registry and dependency injection."""
    print("\nTesting component registry...")
    
    try:
        from src.core.registry import ComponentRegistry, register_component, get_component
        
        # Create registry
        registry = ComponentRegistry()
        print("✓ Component registry created")
        
        # Test component registration
        class TestComponent:
            def __init__(self):
                self.value = "test"
        
        success = registry.register_component("test_component", TestComponent())
        if success:
            print("✓ Component registered")
        else:
            print("✗ Component registration failed")
            return False
        
        # Test component retrieval
        component = registry.get_component("test_component")
        if component and component.value == "test":
            print("✓ Component retrieved successfully")
        else:
            print("✗ Component retrieval failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Component registry error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_logging_system():
    """Test logging system."""
    print("\nTesting logging system...")
    
    try:
        from src.core.logging import setup_logging, get_logger
        
        # Setup logging
        log_manager = setup_logging()
        print("✓ Logging system setup")
        
        # Get logger
        logger = get_logger("test")
        print("✓ Logger created")
        
        # Test logging
        logger.info("Test log message")
        print("✓ Log message sent")
        
        return True
        
    except Exception as e:
        print(f"✗ Logging system error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling system."""
    print("\nTesting error handling...")
    
    try:
        from src.core.errors import ErrorHandler, handle_error, ErrorSeverity
        
        # Create error handler
        error_handler = ErrorHandler()
        print("✓ Error handler created")
        
        # Test error handling
        try:
            raise ValueError("Test error")
        except Exception as e:
            record = error_handler.handle_error(e, "test", ErrorSeverity.LOW)
            print("✓ Error handled successfully")
            
            if record.message == "Test error":
                print("✓ Error record created correctly")
            else:
                print("✗ Error record incorrect")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling system error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("=== Task-6.1 Application Integration Layer Tests ===\n")
    
    tests = [
        test_core_imports,
        test_application_creation,
        test_event_system,
        test_configuration_system,
        test_component_registry,
        test_logging_system,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("Test failed!")
        except Exception as e:
            print(f"Test exception: {e}")
    
    print(f"\n=== Test Results: {passed}/{total} passed ===")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)