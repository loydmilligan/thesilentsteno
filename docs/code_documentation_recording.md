# Recording Module Documentation

## Module Overview

The Recording module provides comprehensive audio recording capabilities for The Silent Steno, implementing a complete recording system with session management, file organization, metadata tracking, audio preprocessing, and storage monitoring. The module is designed for high-quality audio recording with intelligent file management and real-time processing capabilities.

## Dependencies

### External Dependencies
- `soundfile` - Audio file I/O
- `numpy` - Numerical computing
- `scipy` - Scientific computing and signal processing
- `librosa` - Audio analysis and processing
- `threading` - Thread management
- `queue` - Thread-safe queues
- `time` - Timing operations
- `datetime` - Date/time operations
- `pathlib` - Path operations
- `hashlib` - Hash functions
- `json` - JSON processing
- `csv` - CSV processing
- `xml.etree.ElementTree` - XML processing
- `logging` - Logging system
- `dataclasses` - Data structures
- `enum` - Enumerations
- `typing` - Type hints
- `shutil` - File operations
- `os` - Operating system interface
- `psutil` - System monitoring

### Internal Dependencies
- `src.core.events` - Event system
- `src.core.logging` - Logging system
- `src.core.monitoring` - Performance monitoring
- `src.audio.audio_pipeline` - Audio pipeline integration
- `src.analysis.quality_assessor` - Audio quality assessment

## File Documentation

### 1. `__init__.py`

**Purpose**: Module initialization and main recording system orchestrator providing unified access to all recording components.

#### Classes

##### `RecordingSystem`
Main recording system orchestrator managing all components.

**Attributes:**
- `audio_recorder: AudioRecorder` - Core audio recording engine
- `session_manager: SessionManager` - Session lifecycle management
- `file_manager: FileManager` - File organization and management
- `metadata_tracker: MetadataTracker` - Metadata collection
- `preprocessor: AudioPreprocessor` - Audio preprocessing
- `storage_monitor: StorageMonitor` - Storage monitoring
- `is_recording: bool` - Recording state
- `current_session: dict` - Current session information

**Methods:**
- `__init__(config: dict = None)` - Initialize recording system
- `start_recording(session_config: dict = None)` - Start recording session
- `stop_recording()` - Stop recording session
- `pause_recording()` - Pause recording session
- `resume_recording()` - Resume recording session
- `get_recording_status()` - Get current recording status
- `get_session_info()` - Get current session information
- `set_callback(event_type: str, callback: callable)` - Set event callback
- `get_statistics()` - Get recording statistics

#### Default Configurations

##### `DEFAULT_SESSION_CONFIG`
Default session configuration.
```python
{
    "session_name": None,
    "auto_naming": True,
    "pause_on_silence": False,
    "max_duration": None,
    "enable_preprocessing": True,
    "enable_metadata": True,
    "quality_preset": "balanced"
}
```

##### `DEFAULT_RECORDING_CONFIG`
Default recording configuration.
```python
{
    "format": "flac",
    "sample_rate": 44100,
    "channels": 2,
    "bit_depth": 16,
    "enable_monitoring": True,
    "real_time_processing": True
}
```

##### `DEFAULT_PROCESSING_CONFIG`
Default processing configuration.
```python
{
    "enable_noise_reduction": True,
    "enable_normalization": True,
    "enable_enhancement": True,
    "processing_mode": "balanced",
    "quality_threshold": 0.7
}
```

##### `DEFAULT_STORAGE_CONFIG`
Default storage configuration.
```python
{
    "base_directory": "recordings",
    "organization_scheme": "date",
    "enable_monitoring": True,
    "cleanup_enabled": True,
    "alert_threshold": 0.9
}
```

#### Quality Presets

##### `LOW_LATENCY_PRESET`
Optimized for minimal latency.
- Buffer size: 64 frames
- Processing: Minimal
- Quality: Basic
- Latency: <20ms

##### `BALANCED_PRESET`
Balanced quality and performance.
- Buffer size: 128 frames
- Processing: Standard
- Quality: Good
- Latency: <40ms

##### `HIGH_QUALITY_PRESET`
Optimized for maximum quality.
- Buffer size: 512 frames
- Processing: Comprehensive
- Quality: Excellent
- Latency: <100ms

**Usage Example:**
```python
from src.recording import RecordingSystem

# Create recording system
recording_system = RecordingSystem({
    "session": {
        "auto_naming": True,
        "enable_preprocessing": True,
        "quality_preset": "balanced"
    },
    "recording": {
        "format": "flac",
        "sample_rate": 44100,
        "channels": 2
    },
    "storage": {
        "base_directory": "recordings",
        "organization_scheme": "date"
    }
})

# Set up callbacks
def on_recording_start(session_info):
    print(f"Recording started: {session_info['session_name']}")

def on_recording_stop(session_info):
    print(f"Recording stopped: {session_info['duration']:.1f}s")

recording_system.set_callback("recording_start", on_recording_start)
recording_system.set_callback("recording_stop", on_recording_stop)

# Start recording
recording_system.start_recording({
    "session_name": "Team Meeting",
    "max_duration": 3600  # 1 hour
})

# Get recording status
status = recording_system.get_recording_status()
print(f"Recording: {status['is_recording']}")
print(f"Duration: {status['duration']:.1f}s")
print(f"File size: {status['file_size']} bytes")

# Stop recording
recording_system.stop_recording()
```

### 2. `audio_recorder.py`

**Purpose**: Core audio recording engine with multi-format support, quality presets, and real-time processing.

#### Enums

##### `AudioFormat`
Supported audio formats.
- `WAV` - Uncompressed WAV format
- `FLAC` - Lossless compressed FLAC
- `MP3` - Lossy compressed MP3
- `OGG` - Ogg Vorbis format

##### `RecordingState`
Recording state enumeration.
- `STOPPED` - Not recording
- `STARTING` - Starting recording
- `RECORDING` - Active recording
- `PAUSED` - Recording paused
- `STOPPING` - Stopping recording
- `ERROR` - Error state

#### Classes

##### `RecordingConfig`
Audio recording configuration.

**Attributes:**
- `format: AudioFormat` - Audio format
- `sample_rate: int` - Sample rate in Hz
- `channels: int` - Number of channels
- `bit_depth: int` - Bit depth
- `quality_preset: str` - Quality preset
- `enable_monitoring: bool` - Enable level monitoring
- `enable_preprocessing: bool` - Enable real-time preprocessing
- `buffer_size: int` - Audio buffer size
- `max_file_size: int` - Maximum file size in bytes

##### `AudioRecorder`
Main audio recording engine.

**Methods:**
- `__init__(config: RecordingConfig)` - Initialize audio recorder
- `start_recording(output_file: str)` - Start recording to file
- `stop_recording()` - Stop recording
- `pause_recording()` - Pause recording
- `resume_recording()` - Resume recording
- `get_recording_state()` - Get current recording state
- `get_audio_levels()` - Get current audio levels
- `get_recording_info()` - Get recording information
- `convert_format(input_file: str, output_file: str, target_format: AudioFormat)` - Convert format
- `get_recording_statistics()` - Get recording statistics

**Usage Example:**
```python
from src.recording.audio_recorder import AudioRecorder, RecordingConfig, AudioFormat

# Create recording configuration
config = RecordingConfig(
    format=AudioFormat.FLAC,
    sample_rate=44100,
    channels=2,
    bit_depth=16,
    quality_preset="balanced",
    enable_monitoring=True,
    enable_preprocessing=True
)

# Create recorder
recorder = AudioRecorder(config)

# Start recording
recorder.start_recording("meeting_recording.flac")

# Monitor recording
import time
while recorder.get_recording_state() == RecordingState.RECORDING:
    levels = recorder.get_audio_levels()
    print(f"Levels: L={levels['left']:.2f}, R={levels['right']:.2f}")
    time.sleep(1)

# Stop recording
recorder.stop_recording()

# Get recording info
info = recorder.get_recording_info()
print(f"Duration: {info['duration']:.1f}s")
print(f"File size: {info['file_size']} bytes")
print(f"Sample rate: {info['sample_rate']} Hz")
```

### 3. `file_manager.py`

**Purpose**: Intelligent file organization and management with automatic cleanup and integrity checking.

#### Enums

##### `OrganizationScheme`
File organization schemes.
- `DATE` - Organize by date
- `TYPE` - Organize by file type
- `PARTICIPANT` - Organize by participant
- `SESSION` - Organize by session
- `HYBRID` - Combination approach

##### `FileType`
File type enumeration.
- `AUDIO` - Audio recording files
- `TRANSCRIPT` - Transcript files
- `METADATA` - Metadata files
- `ANALYSIS` - Analysis result files
- `EXPORT` - Export files

#### Classes

##### `FileManagerConfig`
File manager configuration.

**Attributes:**
- `base_directory: str` - Base directory path
- `organization_scheme: OrganizationScheme` - Organization scheme
- `auto_cleanup: bool` - Enable automatic cleanup
- `temp_file_lifetime: int` - Temporary file lifetime in seconds
- `enable_checksums: bool` - Enable file integrity checksums
- `backup_enabled: bool` - Enable automatic backups
- `compression_enabled: bool` - Enable file compression

##### `FileInfo`
File information structure.

**Attributes:**
- `file_path: str` - File path
- `file_type: FileType` - File type
- `size: int` - File size in bytes
- `created_at: datetime` - Creation timestamp
- `modified_at: datetime` - Last modification timestamp
- `checksum: str` - File checksum
- `metadata: dict` - Additional metadata

##### `FileManager`
Main file management system.

**Methods:**
- `__init__(config: FileManagerConfig)` - Initialize file manager
- `organize_file(file_path: str, file_type: FileType, metadata: dict = None)` - Organize file
- `create_directory_structure(base_path: str, scheme: OrganizationScheme)` - Create directories
- `generate_filename(base_name: str, file_type: FileType, metadata: dict = None)` - Generate filename
- `cleanup_temporary_files()` - Clean up temporary files
- `calculate_checksum(file_path: str)` - Calculate file checksum
- `verify_file_integrity(file_path: str)` - Verify file integrity
- `get_file_info(file_path: str)` - Get file information
- `get_storage_usage()` - Get storage usage statistics

**Usage Example:**
```python
from src.recording.file_manager import FileManager, FileManagerConfig, OrganizationScheme

# Create file manager configuration
config = FileManagerConfig(
    base_directory="recordings",
    organization_scheme=OrganizationScheme.DATE,
    auto_cleanup=True,
    temp_file_lifetime=3600,  # 1 hour
    enable_checksums=True
)

# Create file manager
file_manager = FileManager(config)

# Organize recording file
metadata = {
    "session_name": "Team Meeting",
    "participants": ["Alice", "Bob", "Charlie"],
    "date": "2024-01-15",
    "duration": 3600
}

organized_path = file_manager.organize_file(
    "temp_recording.flac",
    FileType.AUDIO,
    metadata
)

print(f"File organized to: {organized_path}")

# Get file information
file_info = file_manager.get_file_info(organized_path)
print(f"File size: {file_info.size} bytes")
print(f"Checksum: {file_info.checksum}")
print(f"Created: {file_info.created_at}")

# Clean up temporary files
file_manager.cleanup_temporary_files()

# Get storage usage
usage = file_manager.get_storage_usage()
print(f"Storage used: {usage['used_space']} bytes")
print(f"Storage available: {usage['available_space']} bytes")
```

### 4. `metadata_tracker.py`

**Purpose**: Comprehensive session metadata collection and tracking with real-time updates.

#### Classes

##### `SessionMetadata`
Session metadata structure.

**Attributes:**
- `session_id: str` - Session identifier
- `session_name: str` - Session name
- `start_time: datetime` - Start timestamp
- `end_time: datetime` - End timestamp
- `duration: float` - Session duration in seconds
- `participants: List[str]` - Participant list
- `location: str` - Session location
- `device_info: dict` - Device information
- `audio_settings: dict` - Audio settings used

##### `ParticipantMetadata`
Participant metadata structure.

**Attributes:**
- `participant_id: str` - Participant identifier
- `name: str` - Participant name
- `speaking_time: float` - Total speaking time
- `word_count: int` - Estimated word count
- `volume_stats: dict` - Volume statistics
- `participation_level: float` - Participation level (0-1)
- `interruptions: int` - Number of interruptions

##### `AudioQualityMetadata`
Audio quality metadata structure.

**Attributes:**
- `average_snr: float` - Average signal-to-noise ratio
- `dynamic_range: float` - Dynamic range in dB
- `peak_level: float` - Peak audio level
- `rms_level: float` - RMS audio level
- `speech_clarity: float` - Speech clarity score
- `background_noise: float` - Background noise level

##### `MetadataTracker`
Main metadata tracking system.

**Methods:**
- `__init__(config: dict = None)` - Initialize metadata tracker
- `start_session(session_info: dict)` - Start session tracking
- `end_session()` - End session tracking
- `update_participant_stats(participant_id: str, stats: dict)` - Update participant stats
- `update_audio_quality(quality_metrics: dict)` - Update audio quality
- `update_system_performance(performance_metrics: dict)` - Update system performance
- `get_session_metadata()` - Get complete session metadata
- `export_metadata(format: str, file_path: str)` - Export metadata
- `get_real_time_stats()` - Get real-time statistics

**Usage Example:**
```python
from src.recording.metadata_tracker import MetadataTracker

# Create metadata tracker
tracker = MetadataTracker({
    "enable_real_time": True,
    "update_interval": 1.0,
    "participant_tracking": True,
    "quality_monitoring": True
})

# Start session tracking
session_info = {
    "session_name": "Team Meeting",
    "participants": ["Alice", "Bob", "Charlie"],
    "location": "Conference Room A",
    "expected_duration": 3600
}

tracker.start_session(session_info)

# Update participant statistics
tracker.update_participant_stats("alice", {
    "speaking_time": 120.5,
    "word_count": 250,
    "volume_level": 0.7
})

# Update audio quality
tracker.update_audio_quality({
    "snr": 25.3,
    "dynamic_range": 18.7,
    "speech_clarity": 0.85
})

# Get real-time stats
stats = tracker.get_real_time_stats()
print(f"Session duration: {stats['duration']:.1f}s")
print(f"Active participants: {stats['active_participants']}")
print(f"Audio quality: {stats['audio_quality']:.2f}")

# End session and export metadata
tracker.end_session()
tracker.export_metadata("json", "session_metadata.json")
```

### 5. `preprocessor.py`

**Purpose**: Advanced audio preprocessing and enhancement with adaptive processing and quality assessment.

#### Enums

##### `ProcessingMode`
Audio processing modes.
- `REALTIME` - Real-time processing
- `BALANCED` - Balanced processing
- `QUALITY` - High-quality processing
- `SPEECH_OPTIMIZED` - Speech-optimized processing

##### `NoiseReductionMethod`
Noise reduction methods.
- `SPECTRAL_SUBTRACTION` - Spectral subtraction
- `WIENER_FILTER` - Wiener filtering
- `ADAPTIVE` - Adaptive noise reduction

#### Classes

##### `ProcessingConfig`
Audio processing configuration.

**Attributes:**
- `mode: ProcessingMode` - Processing mode
- `enable_noise_reduction: bool` - Enable noise reduction
- `noise_reduction_method: NoiseReductionMethod` - Noise reduction method
- `enable_normalization: bool` - Enable normalization
- `target_level: float` - Target normalization level
- `enable_enhancement: bool` - Enable speech enhancement
- `quality_threshold: float` - Quality threshold
- `adaptive_processing: bool` - Enable adaptive processing

##### `ProcessingResult`
Audio processing result.

**Attributes:**
- `processed_audio: np.ndarray` - Processed audio data
- `processing_time: float` - Processing time in seconds
- `quality_improvement: float` - Quality improvement score
- `applied_filters: List[str]` - Applied filters
- `quality_metrics: dict` - Quality metrics

##### `AudioPreprocessor`
Main audio preprocessing system.

**Methods:**
- `__init__(config: ProcessingConfig)` - Initialize preprocessor
- `process_audio(audio_data: np.ndarray, sample_rate: int)` - Process audio
- `apply_noise_reduction(audio_data: np.ndarray, sample_rate: int)` - Apply noise reduction
- `apply_normalization(audio_data: np.ndarray, target_level: float)` - Apply normalization
- `apply_speech_enhancement(audio_data: np.ndarray, sample_rate: int)` - Apply speech enhancement
- `assess_quality(audio_data: np.ndarray, sample_rate: int)` - Assess audio quality
- `update_processing_mode(mode: ProcessingMode)` - Update processing mode
- `get_processing_statistics()` - Get processing statistics

**Usage Example:**
```python
from src.recording.preprocessor import AudioPreprocessor, ProcessingConfig, ProcessingMode

# Create processing configuration
config = ProcessingConfig(
    mode=ProcessingMode.BALANCED,
    enable_noise_reduction=True,
    noise_reduction_method=NoiseReductionMethod.SPECTRAL_SUBTRACTION,
    enable_normalization=True,
    target_level=-12.0,  # dB
    enable_enhancement=True,
    quality_threshold=0.7,
    adaptive_processing=True
)

# Create preprocessor
preprocessor = AudioPreprocessor(config)

# Process audio
import numpy as np
audio_data = np.random.randn(44100 * 5)  # 5 seconds of audio
sample_rate = 44100

result = preprocessor.process_audio(audio_data, sample_rate)

print(f"Processing time: {result.processing_time:.3f}s")
print(f"Quality improvement: {result.quality_improvement:.2f}")
print(f"Applied filters: {result.applied_filters}")
print(f"SNR: {result.quality_metrics['snr']:.1f} dB")

# Get processing statistics
stats = preprocessor.get_processing_statistics()
print(f"Total processed: {stats['total_processed']}")
print(f"Average processing time: {stats['average_processing_time']:.3f}s")
```

### 6. `session_manager.py`

**Purpose**: Complete session lifecycle management with state persistence and error recovery.

#### Enums

##### `SessionState`
Session state enumeration.
- `IDLE` - No active session
- `STARTING` - Session starting
- `RECORDING` - Active recording
- `PAUSED` - Session paused
- `STOPPING` - Session stopping
- `COMPLETED` - Session completed
- `ERROR` - Session error

#### Classes

##### `SessionConfig`
Session configuration.

**Attributes:**
- `session_name: str` - Session name
- `max_duration: int` - Maximum duration in seconds
- `auto_naming: bool` - Enable automatic naming
- `pause_on_silence: bool` - Pause on silence detection
- `enable_preprocessing: bool` - Enable preprocessing
- `enable_metadata: bool` - Enable metadata tracking
- `quality_preset: str` - Quality preset
- `participants: List[str]` - Expected participants

##### `SessionInfo`
Session information structure.

**Attributes:**
- `session_id: str` - Session identifier
- `session_name: str` - Session name
- `state: SessionState` - Current state
- `start_time: datetime` - Start timestamp
- `end_time: datetime` - End timestamp
- `duration: float` - Current duration
- `file_path: str` - Recording file path
- `participants: List[str]` - Participants
- `metadata: dict` - Session metadata

##### `SessionManager`
Main session management system.

**Methods:**
- `__init__(config: dict = None)` - Initialize session manager
- `start_session(session_config: SessionConfig)` - Start new session
- `stop_session()` - Stop current session
- `pause_session()` - Pause current session
- `resume_session()` - Resume paused session
- `get_session_info()` - Get current session info
- `get_session_history()` - Get session history
- `save_session_state()` - Save session state
- `load_session_state()` - Load session state
- `cleanup_failed_sessions()` - Clean up failed sessions

**Usage Example:**
```python
from src.recording.session_manager import SessionManager, SessionConfig

# Create session manager
session_manager = SessionManager({
    "enable_state_persistence": True,
    "auto_recovery": True,
    "max_concurrent_sessions": 3,
    "session_timeout": 3600
})

# Create session configuration
session_config = SessionConfig(
    session_name="Team Meeting",
    max_duration=3600,
    auto_naming=True,
    pause_on_silence=False,
    enable_preprocessing=True,
    enable_metadata=True,
    participants=["Alice", "Bob", "Charlie"]
)

# Start session
session_manager.start_session(session_config)

# Get session info
session_info = session_manager.get_session_info()
print(f"Session: {session_info.session_name}")
print(f"State: {session_info.state}")
print(f"Duration: {session_info.duration:.1f}s")

# Pause and resume
session_manager.pause_session()
time.sleep(5)
session_manager.resume_session()

# Stop session
session_manager.stop_session()

# Get session history
history = session_manager.get_session_history()
print(f"Total sessions: {len(history)}")
```

### 7. `storage_monitor.py`

**Purpose**: Advanced storage monitoring and management with predictive analytics and automated cleanup.

#### Classes

##### `StorageConfig`
Storage monitoring configuration.

**Attributes:**
- `monitor_interval: float` - Monitoring interval in seconds
- `alert_thresholds: dict` - Alert thresholds
- `enable_prediction: bool` - Enable usage prediction
- `enable_cleanup: bool` - Enable automated cleanup
- `cleanup_threshold: float` - Cleanup threshold
- `enable_compression: bool` - Enable compression
- `health_check_interval: float` - Health check interval

##### `StorageAlert`
Storage alert information.

**Attributes:**
- `alert_type: str` - Alert type
- `severity: str` - Alert severity
- `message: str` - Alert message
- `timestamp: datetime` - Alert timestamp
- `current_usage: float` - Current usage percentage
- `threshold: float` - Alert threshold

##### `StorageMonitor`
Main storage monitoring system.

**Methods:**
- `__init__(config: StorageConfig)` - Initialize storage monitor
- `start_monitoring()` - Start monitoring
- `stop_monitoring()` - Stop monitoring
- `get_storage_usage()` - Get current storage usage
- `get_usage_prediction(hours: int)` - Get usage prediction
- `cleanup_old_files(days: int)` - Clean up old files
- `compress_files(file_paths: List[str])` - Compress files
- `get_storage_alerts()` - Get storage alerts
- `check_storage_health()` - Check storage health

**Usage Example:**
```python
from src.recording.storage_monitor import StorageMonitor, StorageConfig

# Create storage configuration
config = StorageConfig(
    monitor_interval=30.0,
    alert_thresholds={
        "warning": 0.8,
        "critical": 0.9,
        "emergency": 0.95
    },
    enable_prediction=True,
    enable_cleanup=True,
    cleanup_threshold=0.85,
    enable_compression=True
)

# Create storage monitor
monitor = StorageMonitor(config)

# Start monitoring
monitor.start_monitoring()

# Get current usage
usage = monitor.get_storage_usage()
print(f"Storage usage: {usage['usage_percentage']:.1%}")
print(f"Available space: {usage['available_space'] / 1024 / 1024:.1f} MB")

# Get usage prediction
prediction = monitor.get_usage_prediction(hours=24)
print(f"Predicted usage in 24h: {prediction['predicted_usage']:.1%}")

# Check for alerts
alerts = monitor.get_storage_alerts()
for alert in alerts:
    print(f"Alert: {alert.alert_type} - {alert.message}")

# Clean up old files if needed
if usage['usage_percentage'] > 0.8:
    monitor.cleanup_old_files(days=30)

# Check storage health
health = monitor.check_storage_health()
print(f"Storage health: {health['status']}")
```

## Module Integration

The Recording module integrates with other Silent Steno components:

1. **Audio Module**: Receives audio streams for recording
2. **AI Module**: Provides recorded audio for transcription
3. **Data Module**: Stores session metadata and information
4. **Export Module**: Provides recorded files for export
5. **UI Module**: Displays recording status and controls
6. **System Module**: Monitors system resources and health

## Common Usage Patterns

### Complete Recording Workflow
```python
# Initialize complete recording system
from src.recording import RecordingSystem

# Create system with comprehensive configuration
config = {
    "session": {
        "auto_naming": True,
        "enable_preprocessing": True,
        "enable_metadata": True,
        "quality_preset": "balanced"
    },
    "recording": {
        "format": "flac",
        "sample_rate": 44100,
        "channels": 2,
        "enable_monitoring": True
    },
    "processing": {
        "enable_noise_reduction": True,
        "enable_normalization": True,
        "enable_enhancement": True,
        "processing_mode": "balanced"
    },
    "storage": {
        "base_directory": "recordings",
        "organization_scheme": "date",
        "enable_monitoring": True,
        "auto_cleanup": True
    }
}

recording_system = RecordingSystem(config)

# Set up comprehensive callbacks
def on_recording_start(session_info):
    print(f"Recording started: {session_info['session_name']}")
    print(f"Output file: {session_info['file_path']}")

def on_recording_stop(session_info):
    print(f"Recording completed: {session_info['duration']:.1f}s")
    print(f"File size: {session_info['file_size']} bytes")

def on_audio_levels(levels):
    print(f"Audio levels: L={levels['left']:.2f}, R={levels['right']:.2f}")

def on_storage_alert(alert):
    print(f"Storage alert: {alert['message']}")

recording_system.set_callback("recording_start", on_recording_start)
recording_system.set_callback("recording_stop", on_recording_stop)
recording_system.set_callback("audio_levels", on_audio_levels)
recording_system.set_callback("storage_alert", on_storage_alert)

# Start recording session
recording_system.start_recording({
    "session_name": "Team Meeting",
    "participants": ["Alice", "Bob", "Charlie"],
    "max_duration": 3600,
    "location": "Conference Room A"
})

# Monitor recording
import time
start_time = time.time()
while recording_system.get_recording_status()['is_recording']:
    status = recording_system.get_recording_status()
    print(f"Recording: {status['duration']:.1f}s, "
          f"Size: {status['file_size']} bytes")
    time.sleep(5)

# Stop recording
recording_system.stop_recording()

# Get final statistics
stats = recording_system.get_statistics()
print(f"Total recordings: {stats['total_recordings']}")
print(f"Total duration: {stats['total_duration']:.1f}s")
print(f"Average quality: {stats['average_quality']:.2f}")
```

### Session Management with Recovery
```python
# Create robust session management
from src.recording.session_manager import SessionManager, SessionConfig

class RobustSessionManager:
    def __init__(self):
        self.session_manager = SessionManager({
            "enable_state_persistence": True,
            "auto_recovery": True,
            "session_timeout": 3600
        })
        
    def start_meeting_session(self, meeting_info):
        # Create session configuration
        session_config = SessionConfig(
            session_name=meeting_info['name'],
            max_duration=meeting_info.get('max_duration', 3600),
            participants=meeting_info.get('participants', []),
            enable_preprocessing=True,
            enable_metadata=True
        )
        
        try:
            # Start session
            self.session_manager.start_session(session_config)
            
            # Monitor session health
            self.monitor_session_health()
            
            return True
            
        except Exception as e:
            print(f"Failed to start session: {e}")
            
            # Attempt recovery
            return self.recover_session()
    
    def monitor_session_health(self):
        def health_check():
            while True:
                try:
                    session_info = self.session_manager.get_session_info()
                    
                    # Check session health
                    if session_info.state == SessionState.ERROR:
                        print("Session error detected, attempting recovery")
                        self.recover_session()
                    
                    # Check duration limits
                    if session_info.duration > session_info.max_duration:
                        print("Session duration exceeded, stopping")
                        self.session_manager.stop_session()
                        break
                    
                    time.sleep(10)
                    
                except Exception as e:
                    print(f"Health check error: {e}")
                    break
        
        # Start health monitoring thread
        import threading
        health_thread = threading.Thread(target=health_check, daemon=True)
        health_thread.start()
    
    def recover_session(self):
        try:
            # Load previous session state
            self.session_manager.load_session_state()
            
            # Resume if possible
            session_info = self.session_manager.get_session_info()
            if session_info.state == SessionState.PAUSED:
                self.session_manager.resume_session()
                return True
            
            # Clean up failed sessions
            self.session_manager.cleanup_failed_sessions()
            
            return False
            
        except Exception as e:
            print(f"Recovery failed: {e}")
            return False

# Use robust session manager
robust_manager = RobustSessionManager()

# Start meeting session
meeting_info = {
    "name": "Weekly Team Meeting",
    "participants": ["Alice", "Bob", "Charlie"],
    "max_duration": 3600,
    "location": "Conference Room A"
}

success = robust_manager.start_meeting_session(meeting_info)
if success:
    print("Meeting session started successfully")
else:
    print("Failed to start meeting session")
```

### Advanced Audio Processing Pipeline
```python
# Create advanced audio processing pipeline
from src.recording.audio_recorder import AudioRecorder, RecordingConfig
from src.recording.preprocessor import AudioPreprocessor, ProcessingConfig
from src.recording.metadata_tracker import MetadataTracker

class AdvancedRecordingPipeline:
    def __init__(self):
        # Initialize components
        self.recorder = AudioRecorder(RecordingConfig(
            format=AudioFormat.FLAC,
            sample_rate=44100,
            channels=2,
            enable_monitoring=True
        ))
        
        self.preprocessor = AudioPreprocessor(ProcessingConfig(
            mode=ProcessingMode.BALANCED,
            enable_noise_reduction=True,
            enable_normalization=True,
            enable_enhancement=True,
            adaptive_processing=True
        ))
        
        self.metadata_tracker = MetadataTracker({
            "enable_real_time": True,
            "quality_monitoring": True,
            "participant_tracking": True
        })
        
    def start_recording(self, session_info):
        # Start metadata tracking
        self.metadata_tracker.start_session(session_info)
        
        # Start recording with preprocessing
        self.recorder.start_recording(session_info['output_file'])
        
        # Set up real-time processing
        self.setup_realtime_processing()
        
    def setup_realtime_processing(self):
        def process_audio_frame(audio_data):
            # Process audio
            processed = self.preprocessor.process_audio(audio_data, 44100)
            
            # Update metadata
            self.metadata_tracker.update_audio_quality(processed.quality_metrics)
            
            # Monitor quality
            if processed.quality_metrics['snr'] < 15:
                print("Warning: Low audio quality detected")
            
            return processed.processed_audio
        
        # Set up real-time processing callback
        self.recorder.set_processing_callback(process_audio_frame)
    
    def stop_recording(self):
        # Stop recording
        self.recorder.stop_recording()
        
        # End metadata tracking
        self.metadata_tracker.end_session()
        
        # Get final statistics
        recording_stats = self.recorder.get_recording_statistics()
        processing_stats = self.preprocessor.get_processing_statistics()
        metadata = self.metadata_tracker.get_session_metadata()
        
        return {
            "recording": recording_stats,
            "processing": processing_stats,
            "metadata": metadata
        }

# Use advanced pipeline
pipeline = AdvancedRecordingPipeline()

# Start recording
session_info = {
    "output_file": "advanced_recording.flac",
    "session_name": "Advanced Meeting",
    "participants": ["Alice", "Bob", "Charlie"]
}

pipeline.start_recording(session_info)

# Record for some time
time.sleep(10)

# Stop and get results
results = pipeline.stop_recording()
print(f"Recording completed: {results}")
```

This comprehensive Recording module provides a complete, production-ready recording system with advanced features for session management, audio processing, file organization, and storage monitoring - all essential for The Silent Steno's high-quality meeting recording capabilities.