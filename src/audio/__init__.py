#!/usr/bin/env python3

"""
Audio Module for The Silent Steno

This module provides comprehensive audio processing capabilities for
real-time audio capture, processing, and forwarding with minimal latency.

Main Components:
- AudioPipeline: Main pipeline orchestration and management
- ALSAManager: ALSA audio system configuration and device management
- LatencyOptimizer: Audio latency measurement and optimization
- FormatConverter: Real-time audio format conversion
- LevelMonitor: Real-time audio level monitoring and visualization

Key Features:
- <40ms end-to-end audio latency
- Bluetooth A2DP audio capture and forwarding
- Real-time format conversion between codecs
- Comprehensive audio level monitoring
- Automatic latency optimization
- Audio quality assessment and alerting
"""

from .audio_pipeline import (
    AudioPipeline,
    AudioConfig,
    PipelineState,
    AudioFormat,
    PipelineMetrics,
    start_pipeline,
    stop_pipeline,
    get_pipeline_status,
    get_pipeline_instance
)

from .alsa_manager import (
    ALSAManager,
    ALSAConfig,
    AudioDevice,
    DeviceType,
    DeviceState
)

from .latency_optimizer import (
    LatencyOptimizer,
    LatencyMeasurement,
    LatencyProfile,
    OptimizationConfig,
    LatencyComponent,
    OptimizationLevel
)

from .format_converter import (
    FormatConverter,
    AudioFormat as ConverterAudioFormat,
    ConversionSpec,
    SampleRate,
    BitDepth,
    ChannelConfig
)

from .level_monitor import (
    LevelMonitor,
    AudioLevels,
    AudioAlert,
    MonitorConfig,
    LevelScale,
    AlertType
)

# Audio system factory for backend selection
from .audio_system_factory import (
    AudioSystemFactory,
    AudioSystemType,
    get_audio_backend,
    get_bluetooth_manager,
    is_pipewire_available,
    is_pulseaudio_available,
    get_preferred_latency
)

# Version information
__version__ = "1.0.0"
__author__ = "The Silent Steno Team"
__description__ = "Real-time audio processing pipeline with low latency"

# Public API
__all__ = [
    # Main pipeline
    "AudioPipeline",
    "AudioConfig", 
    "PipelineState",
    "AudioFormat",
    "PipelineMetrics",
    "start_pipeline",
    "stop_pipeline",
    "get_pipeline_status",
    "get_pipeline_instance",
    
    # ALSA management
    "ALSAManager",
    "ALSAConfig",
    "AudioDevice",
    "DeviceType",
    "DeviceState",
    
    # Latency optimization
    "LatencyOptimizer",
    "LatencyMeasurement",
    "LatencyProfile",
    "OptimizationConfig",
    "LatencyComponent",
    "OptimizationLevel",
    
    # Format conversion
    "FormatConverter",
    "ConverterAudioFormat",
    "ConversionSpec",
    "SampleRate",
    "BitDepth",
    "ChannelConfig",
    
    # Level monitoring
    "LevelMonitor",
    "AudioLevels",
    "AudioAlert",
    "MonitorConfig",
    "LevelScale",
    "AlertType",
    
    # Audio system factory
    "AudioSystemFactory",
    "AudioSystemType",
    "get_audio_backend",
    "get_bluetooth_manager",
    "is_pipewire_available",
    "is_pulseaudio_available",
    "get_preferred_latency"
]

# Module initialization
import logging

logger = logging.getLogger(__name__)
logger.info(f"Audio module initialized (version {__version__})")

# Convenience functions for common operations
def create_audio_pipeline(sample_rate: int = 44100, 
                         buffer_size: int = 512,
                         target_latency_ms: float = 40.0) -> AudioPipeline:
    """
    Create a pre-configured audio pipeline
    
    Args:
        sample_rate: Audio sample rate in Hz
        buffer_size: Buffer size in frames
        target_latency_ms: Target latency in milliseconds
        
    Returns:
        Configured AudioPipeline instance
    """
    config = AudioConfig(
        sample_rate=sample_rate,
        buffer_size=buffer_size,
        target_latency_ms=target_latency_ms,
        enable_monitoring=True,
        enable_forwarding=True,
        auto_restart=True
    )
    
    return AudioPipeline(config)


def setup_low_latency_audio() -> bool:
    """
    Set up the audio system for low-latency operation
    
    Returns:
        bool: True if setup successful
    """
    try:
        # Detect and initialize appropriate backend
        backend = get_audio_backend()
        if not backend:
            logger.error("No audio backend available")
            return False
            
        # Log which system we're using
        system_info = AudioSystemFactory.get_system_info()
        logger.info(f"Using audio system: {system_info['system_type']}")
        
        # Get preferred latency for this system
        target_latency = get_preferred_latency()
        logger.info(f"Target latency: {target_latency}ms")
        
        # PipeWire-specific optimization
        if is_pipewire_available() and hasattr(backend, 'optimize_latency'):
            backend.optimize_latency(target_latency)
        else:
            # Fall back to ALSA optimization for PulseAudio
            alsa_manager = ALSAManager()
            
            # Configure for low latency
            if not alsa_manager.optimize_latency():
                logger.error("Failed to optimize ALSA for low latency")
                return False
        
        # Initialize latency optimizer
        latency_optimizer = LatencyOptimizer()
        
        # Perform initial optimization
        if not latency_optimizer.tune_performance():
            logger.warning("Performance tuning had some failures")
        
        logger.info("Low-latency audio setup completed")
        return True
    
    except Exception as e:
        logger.error(f"Error setting up low-latency audio: {e}")
        return False


def get_audio_system_status() -> dict:
    """
    Get comprehensive audio system status
    
    Returns:
        Dict containing status of all audio components
    """
    status = {
        "module_version": __version__,
        "timestamp": __import__('time').time()
    }
    
    # Get audio system info
    try:
        system_info = AudioSystemFactory.get_system_info()
        status["audio_system"] = system_info
    except Exception as e:
        status["audio_system"] = {"error": str(e)}
    
    # Get backend status
    try:
        backend = get_audio_backend()
        if backend:
            devices = backend.refresh_devices()
            status["backend"] = {
                "type": backend.__class__.__name__,
                "device_count": len(devices),
                "sources": len(backend.get_sources()),
                "sinks": len(backend.get_sinks())
            }
    except Exception as e:
        status["backend"] = {"error": str(e)}
    
    try:
        # ALSA status (still relevant for low-level info)
        alsa_manager = ALSAManager()
        status["alsa"] = alsa_manager.get_alsa_status()
    except Exception as e:
        status["alsa"] = {"error": str(e)}
    
    try:
        # Pipeline status
        status["pipeline"] = get_pipeline_status()
    except Exception as e:
        status["pipeline"] = {"error": str(e)}
    
    try:
        # Latency optimizer status
        latency_optimizer = LatencyOptimizer()
        status["latency"] = latency_optimizer.get_optimization_status()
    except Exception as e:
        status["latency"] = {"error": str(e)}
    
    return status


# Module-level configuration
AUDIO_CONFIG_DEFAULTS = {
    "sample_rate": 44100,
    "buffer_size": 512,
    "target_latency_ms": 40.0,
    "channels": 2,
    "bit_depth": 16
}

LATENCY_TARGETS = {
    "ultra_low": 20.0,    # <20ms - Very aggressive
    "low": 40.0,          # <40ms - Target for this project
    "medium": 80.0,       # <80ms - Still acceptable
    "high": 150.0         # <150ms - Noticeable but usable
}

SUPPORTED_FORMATS = {
    "sample_rates": [8000, 11025, 16000, 22050, 44100, 48000, 88200, 96000],
    "bit_depths": [8, 16, 24, 32],
    "channels": [1, 2, 6, 8],
    "codecs": ["PCM", "SBC", "AAC", "aptX"]
}

# Audio quality presets
QUALITY_PRESETS = {
    "low_latency": {
        "sample_rate": 44100,
        "buffer_size": 256,
        "target_latency_ms": 20.0,
        "optimization_level": "aggressive"
    },
    "balanced": {
        "sample_rate": 44100,
        "buffer_size": 512,
        "target_latency_ms": 40.0,
        "optimization_level": "balanced"
    },
    "high_quality": {
        "sample_rate": 48000,
        "buffer_size": 1024,
        "target_latency_ms": 80.0,
        "optimization_level": "conservative"
    }
}


def apply_quality_preset(preset_name: str) -> bool:
    """
    Apply a quality preset to the audio system
    
    Args:
        preset_name: Name of preset ("low_latency", "balanced", "high_quality")
        
    Returns:
        bool: True if preset applied successfully
    """
    if preset_name not in QUALITY_PRESETS:
        logger.error(f"Unknown quality preset: {preset_name}")
        return False
    
    try:
        preset = QUALITY_PRESETS[preset_name]
        logger.info(f"Applying {preset_name} quality preset")
        
        # This would configure the system with the preset values
        # Implementation would depend on specific system requirements
        
        return True
    
    except Exception as e:
        logger.error(f"Error applying quality preset {preset_name}: {e}")
        return False


# Performance monitoring
class AudioPerformanceMonitor:
    """Simple performance monitoring for the audio module"""
    
    def __init__(self):
        self.start_time = __import__('time').time()
        self.stats = {
            "pipelines_created": 0,
            "optimization_runs": 0,
            "format_conversions": 0,
            "level_measurements": 0
        }
    
    def increment(self, stat_name: str) -> None:
        """Increment a performance statistic"""
        if stat_name in self.stats:
            self.stats[stat_name] += 1
    
    def get_stats(self) -> dict:
        """Get current performance statistics"""
        uptime = __import__('time').time() - self.start_time
        return {
            **self.stats,
            "uptime_seconds": uptime,
            "module_version": __version__
        }


# Global performance monitor instance
_performance_monitor = AudioPerformanceMonitor()


def get_performance_stats() -> dict:
    """Get audio module performance statistics"""
    return _performance_monitor.get_stats()
