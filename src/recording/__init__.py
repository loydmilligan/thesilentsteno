#!/usr/bin/env python3

"""
Audio Recording System for The Silent Steno

This package provides comprehensive audio recording capabilities including:
- Session management and lifecycle control
- High-quality audio recording (FLAC/WAV/MP3)
- Audio preprocessing and enhancement
- File organization and management
- Metadata tracking and analysis
- Storage monitoring and optimization

The recording system integrates with the audio pipeline to capture live audio
while maintaining the low-latency forwarding capability that is core to
The Silent Steno's functionality.

Key Components:
- SessionManager: Complete session lifecycle management
- AudioRecorder: High-quality multi-format recording
- AudioPreprocessor: Real-time audio enhancement
- FileManager: Intelligent file organization
- MetadataTracker: Comprehensive session metadata
- StorageMonitor: Storage monitoring and management

Usage Example:
    from src.recording import SessionManager, RecordingConfig
    
    # Create session manager with recording components
    session_manager = SessionManager("recordings")
    
    # Start a recording session
    session_id = session_manager.start_session(
        session_type=SessionType.MEETING,
        metadata={'title': 'Team Standup', 'participants': ['Alice', 'Bob']}
    )
    
    # Session runs automatically...
    
    # Stop session when done
    session_manager.stop_session(session_id)
"""

import logging
from typing import Dict, List, Optional, Any

# Import all main classes and enums for easy access
from .session_manager import (
    SessionManager,
    SessionState, 
    SessionType,
    SessionConfig,
    SessionInfo
)

from .audio_recorder import (
    AudioRecorder,
    RecordingFormat,
    QualityPreset,
    RecordingState,
    RecordingConfig,
    RecordingInfo
)

from .preprocessor import (
    AudioPreprocessor,
    ProcessingMode,
    NoiseProfile,
    ProcessingConfig,
    QualityMetrics
)

from .file_manager import (
    FileManager,
    OrganizationScheme,
    FileType,
    FileInfo,
    StorageConfig as FileStorageConfig
)

from .metadata_tracker import (
    MetadataTracker,
    MetadataCategory,
    ParticipantInfo,
    AudioQualityMetrics,
    SystemPerformanceMetrics,
    ContextualMetadata,
    SessionMetadata
)

from .storage_monitor import (
    StorageMonitor,
    StorageAlert,
    StorageStatus,
    StorageStats,
    StorageConfig,
    AlertInfo
)

# Set up logging
logger = logging.getLogger(__name__)

# Package metadata
__version__ = "1.0.0"
__author__ = "The Silent Steno Development Team"
__description__ = "Comprehensive audio recording system with real-time processing"

# Default configurations
DEFAULT_SESSION_CONFIG = SessionConfig(
    session_type=SessionType.MEETING,
    max_duration_hours=8.0,
    quality_preset="balanced",
    enable_preprocessing=True,
    enable_real_time_analysis=False
)

DEFAULT_RECORDING_CONFIG = RecordingConfig(
    format=RecordingFormat.FLAC,
    quality_preset=QualityPreset.BALANCED,
    sample_rate=44100,
    channels=2,
    bit_depth=16,
    enable_preprocessing=True,
    enable_level_monitoring=True
)

DEFAULT_PROCESSING_CONFIG = ProcessingConfig(
    mode=ProcessingMode.BALANCED,
    enable_noise_reduction=True,
    enable_normalization=True,
    enable_speech_enhancement=True,
    enable_adaptive_processing=True
)

DEFAULT_STORAGE_CONFIG = StorageConfig(
    warning_threshold_percent=80.0,
    critical_threshold_percent=90.0,
    emergency_threshold_percent=95.0,
    min_free_gb=2.0,
    cleanup_enabled=True
)


class RecordingSystem:
    """
    Complete recording system orchestrator
    
    Provides a high-level interface for managing all recording system
    components with proper integration and coordination.
    """
    
    def __init__(self, storage_root: str = "recordings", 
                 session_config: Optional[SessionConfig] = None,
                 recording_config: Optional[RecordingConfig] = None,
                 processing_config: Optional[ProcessingConfig] = None,
                 storage_config: Optional[StorageConfig] = None):
        """
        Initialize complete recording system
        
        Args:
            storage_root: Root directory for all recordings
            session_config: Session management configuration
            recording_config: Audio recording configuration
            processing_config: Audio processing configuration
            storage_config: Storage monitoring configuration
        """
        self.storage_root = storage_root
        
        # Initialize components with configurations
        self.session_manager = SessionManager(
            storage_root=storage_root,
            config=session_config or DEFAULT_SESSION_CONFIG
        )
        
        self.audio_recorder = AudioRecorder(storage_root=storage_root)
        
        self.preprocessor = AudioPreprocessor(
            config=processing_config or DEFAULT_PROCESSING_CONFIG
        )
        
        self.file_manager = FileManager(
            config=FileStorageConfig(root_directory=storage_root)
        )
        
        self.metadata_tracker = MetadataTracker(storage_root=storage_root)
        
        self.storage_monitor = StorageMonitor(
            storage_path=storage_root,
            config=storage_config or DEFAULT_STORAGE_CONFIG
        )
        
        # Wire components together
        self._integrate_components()
        
        # Start monitoring
        self.storage_monitor.start_monitoring()
        
        logger.info(f"Recording system initialized with storage: {storage_root}")
    
    def _integrate_components(self) -> None:
        """Integrate all components with proper cross-references"""
        # Session manager integration
        self.session_manager.set_audio_recorder(self.audio_recorder)
        self.session_manager.set_metadata_tracker(self.metadata_tracker)
        self.session_manager.set_storage_monitor(self.storage_monitor)
        self.session_manager.set_file_manager(self.file_manager)
        
        # Audio recorder integration
        self.audio_recorder.set_preprocessor(self.preprocessor)
        
        # File manager integration
        self.storage_monitor.set_file_manager(self.file_manager)
        
        logger.debug("Component integration completed")
    
    def start_session(self, session_type: SessionType = SessionType.MEETING,
                     metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Start a new recording session
        
        Args:
            session_type: Type of session to record
            metadata: Session metadata
            
        Returns:
            Session ID if successful, None otherwise
        """
        return self.session_manager.start_session(session_type, metadata=metadata)
    
    def stop_session(self, session_id: Optional[str] = None) -> bool:
        """
        Stop a recording session
        
        Args:
            session_id: Session to stop (current session if None)
            
        Returns:
            True if successful
        """
        return self.session_manager.stop_session(session_id)
    
    def pause_session(self, session_id: Optional[str] = None) -> bool:
        """Pause a recording session"""
        return self.session_manager.pause_session(session_id)
    
    def resume_session(self, session_id: Optional[str] = None) -> bool:
        """Resume a paused session"""
        return self.session_manager.resume_session(session_id)
    
    def get_session_status(self, session_id: Optional[str] = None) -> Optional[SessionInfo]:
        """Get status of a recording session"""
        return self.session_manager.get_session_status(session_id)
    
    def get_storage_status(self) -> Dict[str, Any]:
        """Get current storage status"""
        return self.storage_monitor.get_storage_status()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'session_manager': {
                'active_sessions': len(self.session_manager.active_sessions),
                'current_session': self.session_manager.current_session_id
            },
            'audio_recorder': self.audio_recorder.get_performance_stats(),
            'preprocessor': self.preprocessor.get_performance_stats(),
            'file_manager': self.file_manager.get_performance_stats(),
            'metadata_tracker': self.metadata_tracker.get_tracking_statistics(),
            'storage_monitor': self.storage_monitor.get_performance_stats(),
            'storage_status': self.get_storage_status()
        }
    
    def shutdown(self) -> None:
        """Shutdown the recording system gracefully"""
        try:
            logger.info("Shutting down recording system...")
            
            # Stop monitoring
            self.storage_monitor.stop_monitoring()
            
            # Shutdown session manager (stops active sessions)
            self.session_manager.shutdown()
            
            logger.info("Recording system shutdown completed")
        
        except Exception as e:
            logger.error(f"Error during recording system shutdown: {e}")


# Convenience functions for quick access
def create_recording_system(storage_root: str = "recordings", **configs) -> RecordingSystem:
    """
    Create a complete recording system with default configurations
    
    Args:
        storage_root: Root directory for recordings
        **configs: Configuration overrides
        
    Returns:
        Configured RecordingSystem instance
    """
    return RecordingSystem(storage_root=storage_root, **configs)


def get_default_configs() -> Dict[str, Any]:
    """Get dictionary of all default configurations"""
    return {
        'session_config': DEFAULT_SESSION_CONFIG,
        'recording_config': DEFAULT_RECORDING_CONFIG,
        'processing_config': DEFAULT_PROCESSING_CONFIG,
        'storage_config': DEFAULT_STORAGE_CONFIG
    }


# Quality presets for easy configuration
QUALITY_PRESETS = {
    'low_latency': {
        'recording_config': RecordingConfig(
            format=RecordingFormat.WAV,
            quality_preset=QualityPreset.LOW_LATENCY,
            sample_rate=44100,
            bit_depth=16,
            enable_preprocessing=False
        ),
        'processing_config': ProcessingConfig(
            mode=ProcessingMode.REAL_TIME,
            enable_noise_reduction=False,
            enable_normalization=True,
            enable_speech_enhancement=False
        )
    },
    'balanced': {
        'recording_config': DEFAULT_RECORDING_CONFIG,
        'processing_config': DEFAULT_PROCESSING_CONFIG
    },
    'high_quality': {
        'recording_config': RecordingConfig(
            format=RecordingFormat.FLAC,
            quality_preset=QualityPreset.HIGH_QUALITY,
            sample_rate=48000,
            bit_depth=24,
            enable_preprocessing=True,
            compression_level=8
        ),
        'processing_config': ProcessingConfig(
            mode=ProcessingMode.HIGH_QUALITY,
            enable_noise_reduction=True,
            enable_normalization=True,
            enable_speech_enhancement=True,
            enable_adaptive_processing=True
        )
    }
}


def create_system_with_preset(preset_name: str, storage_root: str = "recordings") -> RecordingSystem:
    """
    Create recording system with quality preset
    
    Args:
        preset_name: Name of quality preset ('low_latency', 'balanced', 'high_quality')
        storage_root: Storage root directory
        
    Returns:
        Configured RecordingSystem
    """
    if preset_name not in QUALITY_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(QUALITY_PRESETS.keys())}")
    
    preset = QUALITY_PRESETS[preset_name]
    return RecordingSystem(
        storage_root=storage_root,
        recording_config=preset.get('recording_config'),
        processing_config=preset.get('processing_config')
    )


# Module-level exports
__all__ = [
    # Main classes
    'SessionManager', 'AudioRecorder', 'AudioPreprocessor', 
    'FileManager', 'MetadataTracker', 'StorageMonitor',
    'RecordingSystem',
    
    # Enums
    'SessionState', 'SessionType', 'RecordingFormat', 'QualityPreset',
    'RecordingState', 'ProcessingMode', 'NoiseProfile', 'OrganizationScheme',
    'FileType', 'MetadataCategory', 'StorageAlert', 'StorageStatus',
    
    # Data classes
    'SessionConfig', 'SessionInfo', 'RecordingConfig', 'RecordingInfo',
    'ProcessingConfig', 'QualityMetrics', 'FileInfo', 'ParticipantInfo',
    'AudioQualityMetrics', 'SystemPerformanceMetrics', 'ContextualMetadata',
    'SessionMetadata', 'StorageStats', 'StorageConfig', 'AlertInfo',
    
    # Convenience functions
    'create_recording_system', 'create_system_with_preset', 'get_default_configs',
    
    # Default configurations
    'DEFAULT_SESSION_CONFIG', 'DEFAULT_RECORDING_CONFIG', 
    'DEFAULT_PROCESSING_CONFIG', 'DEFAULT_STORAGE_CONFIG',
    'QUALITY_PRESETS'
]