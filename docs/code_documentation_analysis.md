# Analysis Module Documentation

## Module Overview

The Analysis module provides comprehensive real-time audio analysis capabilities for The Silent Steno, implementing advanced audio processing techniques for voice activity detection, speaker identification, quality assessment, and meeting analytics. The module is designed for low-latency real-time processing with intelligent audio segmentation and comprehensive statistics collection.

## Dependencies

### External Dependencies
- `webrtcvad` - WebRTC Voice Activity Detection
- `numpy` - Numerical computing
- `scipy` - Scientific computing and signal processing
- `librosa` - Audio analysis and feature extraction
- `sklearn` - Machine learning algorithms
- `threading` - Thread management
- `queue` - Thread-safe queues
- `time` - Timing operations
- `logging` - Logging system
- `dataclasses` - Data structures
- `enum` - Enumerations
- `typing` - Type hints
- `json` - JSON serialization
- `pathlib` - Path operations
- `collections` - Specialized data structures

### Internal Dependencies
- `src.core.events` - Event system
- `src.core.logging` - Logging system
- `src.core.monitoring` - Performance monitoring

## File Documentation

### 1. `__init__.py`

**Purpose**: Module initialization and integrated analysis system providing unified access to all analysis components.

#### Classes

##### `IntegratedAnalyzer`
Comprehensive analysis system coordinating all analysis components.

**Attributes:**
- `vad: VoiceActivityDetector` - Voice activity detection
- `speaker_detector: SpeakerDetector` - Speaker identification
- `quality_assessor: QualityAssessor` - Audio quality analysis
- `silence_detector: SilenceDetector` - Silence detection
- `stats_collector: StatisticsCollector` - Meeting statistics
- `audio_chunker: AudioChunker` - Audio segmentation
- `is_processing: bool` - Processing state

**Methods:**
- `__init__(config: dict = None)` - Initialize integrated analyzer
- `start_processing()` - Start analysis processing
- `stop_processing()` - Stop analysis processing
- `analyze_audio_frame(audio_data: bytes, timestamp: float)` - Analyze audio frame
- `get_analysis_results()` - Get current analysis results
- `get_statistics()` - Get comprehensive statistics
- `set_callback(event_type: str, callback: callable)` - Set event callback
- `export_results(format: str, file_path: str)` - Export analysis results

#### Factory Functions

##### `create_integrated_analyzer(config: dict = None) -> IntegratedAnalyzer`
Create fully configured integrated analyzer.

##### `create_vad_analyzer(config: dict = None) -> VoiceActivityDetector`
Create voice activity detector.

##### `create_speaker_analyzer(config: dict = None) -> SpeakerDetector`
Create speaker detector.

##### `create_quality_analyzer(config: dict = None) -> QualityAssessor`
Create quality assessor.

**Usage Example:**
```python
from src.analysis import create_integrated_analyzer

# Create integrated analyzer
analyzer = create_integrated_analyzer({
    "vad": {
        "mode": "aggressive",
        "frame_duration": 30,
        "enable_smoothing": True
    },
    "speaker": {
        "method": "combined",
        "min_segment_duration": 2.0
    },
    "quality": {
        "enable_real_time": True,
        "alert_threshold": 0.7
    }
})

# Start processing
analyzer.start_processing()

# Process audio frames
def process_audio_stream(audio_stream):
    for frame in audio_stream:
        results = analyzer.analyze_audio_frame(frame.data, frame.timestamp)
        
        # Handle results
        if results['vad']['voice_detected']:
            print(f"Voice detected at {frame.timestamp}")
        
        if results['speaker']['speaker_changed']:
            print(f"Speaker changed to {results['speaker']['speaker_id']}")
        
        if results['quality']['quality_score'] < 0.7:
            print("Low audio quality detected")

# Get comprehensive statistics
stats = analyzer.get_statistics()
print(f"Total speaking time: {stats['speaking_time']:.1f}s")
print(f"Number of speakers: {stats['speaker_count']}")
```

### 2. `voice_activity_detector.py`

**Purpose**: Real-time voice activity detection using WebRTC VAD with custom enhancements.

#### Enums

##### `VADMode`
Voice activity detection sensitivity modes.
- `NORMAL` - Standard sensitivity
- `LOW_BITRATE` - Low bitrate optimization
- `AGGRESSIVE` - High sensitivity
- `VERY_AGGRESSIVE` - Maximum sensitivity

##### `VADMethod`
Voice activity detection methods.
- `WEBRTC` - WebRTC VAD only
- `ENERGY` - Energy-based detection
- `COMBINED` - Combined WebRTC and energy

#### Classes

##### `VADConfig`
Voice activity detection configuration.

**Attributes:**
- `mode: VADMode` - Detection sensitivity mode
- `method: VADMethod` - Detection method
- `frame_duration: int` - Frame duration in ms (10/20/30)
- `sample_rate: int` - Audio sample rate
- `enable_smoothing: bool` - Enable temporal smoothing
- `smoothing_window: int` - Smoothing window size
- `energy_threshold: float` - Energy detection threshold
- `min_voice_duration: float` - Minimum voice segment duration
- `min_silence_duration: float` - Minimum silence duration

##### `VADResult`
Voice activity detection result.

**Attributes:**
- `voice_detected: bool` - Voice activity detected
- `confidence: float` - Detection confidence (0-1)
- `energy_level: float` - Audio energy level
- `voice_start_time: float` - Voice segment start time
- `voice_end_time: float` - Voice segment end time
- `segment_duration: float` - Current segment duration
- `method_used: VADMethod` - Detection method used

##### `VoiceActivityDetector`
Main voice activity detection system.

**Methods:**
- `__init__(config: VADConfig)` - Initialize VAD with configuration
- `start_detection()` - Start voice activity detection
- `stop_detection()` - Stop voice activity detection
- `detect_voice_activity(audio_data: bytes, timestamp: float)` - Detect voice activity
- `set_callback(event_type: str, callback: callable)` - Set event callback
- `get_voice_segments()` - Get detected voice segments
- `get_statistics()` - Get detection statistics
- `update_config(config: VADConfig)` - Update configuration

**Usage Example:**
```python
from src.analysis.voice_activity_detector import VoiceActivityDetector, VADConfig, VADMode

# Create VAD configuration
vad_config = VADConfig(
    mode=VADMode.AGGRESSIVE,
    method=VADMethod.COMBINED,
    frame_duration=30,
    sample_rate=16000,
    enable_smoothing=True,
    min_voice_duration=0.5,
    min_silence_duration=0.3
)

# Create VAD
vad = VoiceActivityDetector(vad_config)

# Set up callbacks
def on_voice_start(timestamp):
    print(f"Voice started at {timestamp:.2f}s")

def on_voice_end(timestamp, duration):
    print(f"Voice ended at {timestamp:.2f}s (duration: {duration:.2f}s)")

vad.set_callback("voice_start", on_voice_start)
vad.set_callback("voice_end", on_voice_end)

# Start detection
vad.start_detection()

# Process audio frames
for frame in audio_stream:
    result = vad.detect_voice_activity(frame.data, frame.timestamp)
    
    if result.voice_detected:
        print(f"Voice detected: confidence={result.confidence:.2f}, "
              f"energy={result.energy_level:.2f}")

# Get statistics
stats = vad.get_statistics()
print(f"Voice activity ratio: {stats['voice_ratio']:.2f}")
print(f"Total voice segments: {stats['voice_segments']}")
```

### 3. `speaker_detector.py`

**Purpose**: Multi-speaker diarization and speaker change detection with real-time clustering.

#### Enums

##### `SpeakerChangeMethod`
Speaker change detection methods.
- `MFCC` - MFCC-based detection
- `SPECTRAL` - Spectral feature-based
- `COMBINED` - Combined feature approach

#### Classes

##### `SpeakerConfig`
Speaker detection configuration.

**Attributes:**
- `method: SpeakerChangeMethod` - Detection method
- `min_segment_duration: float` - Minimum segment duration
- `max_speakers: int` - Maximum number of speakers
- `similarity_threshold: float` - Speaker similarity threshold
- `feature_window: float` - Feature extraction window
- `clustering_method: str` - Clustering algorithm
- `enable_adaptation: bool` - Enable speaker model adaptation

##### `SpeakerSegment`
Speaker segment information.

**Attributes:**
- `speaker_id: str` - Speaker identifier
- `start_time: float` - Segment start time
- `end_time: float` - Segment end time
- `confidence: float` - Speaker confidence
- `features: np.ndarray` - Audio features
- `is_overlap: bool` - Overlapping speech

##### `SpeakerDetector`
Main speaker detection system.

**Methods:**
- `__init__(config: SpeakerConfig)` - Initialize speaker detector
- `start_detection()` - Start speaker detection
- `stop_detection()` - Stop speaker detection
- `detect_speaker_change(audio_data: bytes, timestamp: float)` - Detect speaker changes
- `identify_speaker(audio_features: np.ndarray)` - Identify speaker
- `add_speaker_model(speaker_id: str, features: np.ndarray)` - Add speaker model
- `get_speaker_segments()` - Get speaker segments
- `get_speaker_statistics()` - Get speaker statistics

**Usage Example:**
```python
from src.analysis.speaker_detector import SpeakerDetector, SpeakerConfig, SpeakerChangeMethod

# Create speaker configuration
speaker_config = SpeakerConfig(
    method=SpeakerChangeMethod.COMBINED,
    min_segment_duration=2.0,
    max_speakers=10,
    similarity_threshold=0.8,
    feature_window=1.0,
    clustering_method="kmeans"
)

# Create speaker detector
speaker_detector = SpeakerDetector(speaker_config)

# Set up callbacks
def on_speaker_change(old_speaker, new_speaker, timestamp):
    print(f"Speaker changed from {old_speaker} to {new_speaker} at {timestamp:.2f}s")

speaker_detector.set_callback("speaker_change", on_speaker_change)

# Start detection
speaker_detector.start_detection()

# Process audio frames
current_speaker = None
for frame in audio_stream:
    result = speaker_detector.detect_speaker_change(frame.data, frame.timestamp)
    
    if result.speaker_changed:
        current_speaker = result.speaker_id
        print(f"Speaker: {current_speaker} (confidence: {result.confidence:.2f})")

# Get speaker statistics
stats = speaker_detector.get_speaker_statistics()
for speaker_id, speaker_stats in stats.items():
    print(f"Speaker {speaker_id}:")
    print(f"  Speaking time: {speaker_stats['speaking_time']:.1f}s")
    print(f"  Segments: {speaker_stats['segment_count']}")
```

### 4. `quality_assessor.py`

**Purpose**: Comprehensive audio quality assessment with real-time monitoring and alerting.

#### Enums

##### `QualityMetric`
Audio quality metrics.
- `SNR` - Signal-to-noise ratio
- `THD` - Total harmonic distortion
- `DYNAMIC_RANGE` - Dynamic range
- `CLARITY` - Speech clarity
- `OVERALL` - Overall quality score

#### Classes

##### `QualityConfig`
Audio quality assessment configuration.

**Attributes:**
- `enable_real_time: bool` - Enable real-time monitoring
- `metrics: List[QualityMetric]` - Metrics to calculate
- `alert_threshold: float` - Quality alert threshold
- `window_size: float` - Analysis window size
- `update_interval: float` - Update interval
- `enable_alerts: bool` - Enable quality alerts

##### `QualityResult`
Audio quality assessment result.

**Attributes:**
- `quality_score: float` - Overall quality score (0-1)
- `snr: float` - Signal-to-noise ratio (dB)
- `thd: float` - Total harmonic distortion (%)
- `dynamic_range: float` - Dynamic range (dB)
- `clarity: float` - Speech clarity score
- `clipping_detected: bool` - Clipping detection
- `distortion_level: float` - Distortion level
- `timestamp: float` - Analysis timestamp

##### `QualityAssessor`
Main audio quality assessment system.

**Methods:**
- `__init__(config: QualityConfig)` - Initialize quality assessor
- `start_assessment()` - Start quality assessment
- `stop_assessment()` - Stop quality assessment
- `assess_quality(audio_data: bytes, timestamp: float)` - Assess audio quality
- `get_quality_history()` - Get quality history
- `get_quality_statistics()` - Get quality statistics
- `set_alert_threshold(threshold: float)` - Set alert threshold

**Usage Example:**
```python
from src.analysis.quality_assessor import QualityAssessor, QualityConfig, QualityMetric

# Create quality configuration
quality_config = QualityConfig(
    enable_real_time=True,
    metrics=[QualityMetric.SNR, QualityMetric.THD, QualityMetric.CLARITY],
    alert_threshold=0.7,
    window_size=1.0,
    update_interval=0.1
)

# Create quality assessor
quality_assessor = QualityAssessor(quality_config)

# Set up callbacks
def on_quality_alert(quality_score, timestamp):
    print(f"Quality alert: {quality_score:.2f} at {timestamp:.2f}s")

quality_assessor.set_callback("quality_alert", on_quality_alert)

# Start assessment
quality_assessor.start_assessment()

# Process audio frames
for frame in audio_stream:
    result = quality_assessor.assess_quality(frame.data, frame.timestamp)
    
    print(f"Quality: {result.quality_score:.2f}, SNR: {result.snr:.1f}dB, "
          f"THD: {result.thd:.2f}%")
    
    if result.clipping_detected:
        print("Warning: Audio clipping detected!")

# Get quality statistics
stats = quality_assessor.get_quality_statistics()
print(f"Average quality: {stats['average_quality']:.2f}")
print(f"Quality alerts: {stats['alert_count']}")
```

### 5. `silence_detector.py`

**Purpose**: Advanced silence detection with automatic audio trimming and noise adaptation.

#### Enums

##### `SilenceMethod`
Silence detection methods.
- `ENERGY` - Energy-based detection
- `AMPLITUDE` - Amplitude-based detection
- `SPECTRAL` - Spectral-based detection
- `ADAPTIVE` - Adaptive threshold detection

##### `SensitivityMode`
Silence detection sensitivity modes.
- `LOW` - Low sensitivity
- `MEDIUM` - Medium sensitivity
- `HIGH` - High sensitivity
- `ADAPTIVE` - Adaptive sensitivity

#### Classes

##### `SilenceConfig`
Silence detection configuration.

**Attributes:**
- `method: SilenceMethod` - Detection method
- `sensitivity: SensitivityMode` - Sensitivity mode
- `energy_threshold: float` - Energy threshold
- `min_silence_duration: float` - Minimum silence duration
- `noise_floor_adaptation: bool` - Enable noise floor adaptation
- `enable_trimming: bool` - Enable automatic trimming

##### `SilenceResult`
Silence detection result.

**Attributes:**
- `silence_detected: bool` - Silence detected
- `silence_start_time: float` - Silence start time
- `silence_end_time: float` - Silence end time
- `silence_duration: float` - Silence duration
- `noise_floor: float` - Current noise floor
- `confidence: float` - Detection confidence

##### `SilenceDetector`
Main silence detection system.

**Methods:**
- `__init__(config: SilenceConfig)` - Initialize silence detector
- `detect_silence(audio_data: bytes, timestamp: float)` - Detect silence
- `trim_silence(audio_data: bytes)` - Trim silence from audio
- `get_silence_segments()` - Get silence segments
- `update_noise_floor(audio_data: bytes)` - Update noise floor
- `get_detection_statistics()` - Get detection statistics

**Usage Example:**
```python
from src.analysis.silence_detector import SilenceDetector, SilenceConfig, SilenceMethod

# Create silence configuration
silence_config = SilenceConfig(
    method=SilenceMethod.ADAPTIVE,
    sensitivity=SensitivityMode.MEDIUM,
    min_silence_duration=0.5,
    noise_floor_adaptation=True,
    enable_trimming=True
)

# Create silence detector
silence_detector = SilenceDetector(silence_config)

# Process audio frames
for frame in audio_stream:
    result = silence_detector.detect_silence(frame.data, frame.timestamp)
    
    if result.silence_detected:
        print(f"Silence detected: {result.silence_duration:.2f}s")
    
    # Trim silence if needed
    if silence_config.enable_trimming:
        trimmed_audio = silence_detector.trim_silence(frame.data)
        if len(trimmed_audio) < len(frame.data):
            print("Audio trimmed")

# Get silence statistics
stats = silence_detector.get_detection_statistics()
print(f"Total silence time: {stats['total_silence_time']:.1f}s")
print(f"Silence segments: {stats['silence_segments']}")
```

### 6. `statistics_collector.py`

**Purpose**: Comprehensive meeting analytics and participation metrics collection.

#### Classes

##### `SpeakerStats`
Individual speaker statistics.

**Attributes:**
- `speaker_id: str` - Speaker identifier
- `speaking_time: float` - Total speaking time
- `word_count: int` - Estimated word count
- `interruptions: int` - Number of interruptions
- `overlap_time: float` - Overlapping speech time
- `turn_count: int` - Number of speaking turns
- `average_turn_duration: float` - Average turn duration

##### `MeetingStats`
Overall meeting statistics.

**Attributes:**
- `total_duration: float` - Total meeting duration
- `speaking_duration: float` - Total speaking time
- `silence_duration: float` - Total silence time
- `speaker_count: int` - Number of speakers
- `turn_count: int` - Total speaking turns
- `interruption_count: int` - Total interruptions
- `participation_balance: float` - Participation balance score

##### `StatisticsCollector`
Main statistics collection system.

**Methods:**
- `__init__(config: dict = None)` - Initialize statistics collector
- `start_collection()` - Start statistics collection
- `stop_collection()` - Stop statistics collection
- `update_speaker_stats(speaker_id: str, speaking_time: float)` - Update speaker statistics
- `record_interruption(speaker_id: str, interrupted_speaker: str)` - Record interruption
- `get_speaker_statistics()` - Get speaker statistics
- `get_meeting_statistics()` - Get meeting statistics
- `export_statistics(format: str, file_path: str)` - Export statistics

**Usage Example:**
```python
from src.analysis.statistics_collector import StatisticsCollector

# Create statistics collector
stats_collector = StatisticsCollector({
    "enable_real_time": True,
    "update_interval": 1.0,
    "track_interruptions": True,
    "track_overlaps": True
})

# Start collection
stats_collector.start_collection()

# Update statistics during meeting
stats_collector.update_speaker_stats("speaker_1", 5.0)  # 5 seconds of speaking
stats_collector.update_speaker_stats("speaker_2", 3.0)  # 3 seconds of speaking
stats_collector.record_interruption("speaker_1", "speaker_2")

# Get real-time statistics
speaker_stats = stats_collector.get_speaker_statistics()
for speaker_id, stats in speaker_stats.items():
    print(f"{speaker_id}: {stats.speaking_time:.1f}s, "
          f"{stats.turn_count} turns")

# Get meeting statistics
meeting_stats = stats_collector.get_meeting_statistics()
print(f"Meeting duration: {meeting_stats.total_duration:.1f}s")
print(f"Speaking ratio: {meeting_stats.speaking_duration / meeting_stats.total_duration:.2f}")
print(f"Participation balance: {meeting_stats.participation_balance:.2f}")

# Export statistics
stats_collector.export_statistics("json", "meeting_stats.json")
```

### 7. `audio_chunker.py`

**Purpose**: Intelligent audio segmentation for optimal processing with multiple chunking strategies.

#### Enums

##### `ChunkingStrategy`
Audio chunking strategies.
- `FIXED_TIME` - Fixed time-based chunks
- `VOICE_ACTIVITY` - Voice activity-based chunks
- `SPEAKER_CHANGE` - Speaker change-based chunks
- `HYBRID` - Hybrid approach

##### `ChunkPriority`
Chunk processing priority.
- `LOW` - Low priority
- `NORMAL` - Normal priority
- `HIGH` - High priority
- `URGENT` - Urgent priority

#### Classes

##### `ChunkConfig`
Audio chunking configuration.

**Attributes:**
- `strategy: ChunkingStrategy` - Chunking strategy
- `chunk_duration: float` - Target chunk duration
- `overlap_duration: float` - Chunk overlap duration
- `min_chunk_duration: float` - Minimum chunk duration
- `max_chunk_duration: float` - Maximum chunk duration
- `enable_quality_based: bool` - Enable quality-based chunking
- `priority_queue: bool` - Enable priority queue

##### `AudioChunk`
Audio chunk information.

**Attributes:**
- `chunk_id: str` - Chunk identifier
- `audio_data: bytes` - Audio data
- `start_time: float` - Chunk start time
- `end_time: float` - Chunk end time
- `duration: float` - Chunk duration
- `quality_score: float` - Quality score
- `priority: ChunkPriority` - Processing priority
- `speaker_id: str` - Speaker identifier

##### `AudioChunker`
Main audio chunking system.

**Methods:**
- `__init__(config: ChunkConfig)` - Initialize audio chunker
- `start_chunking()` - Start chunking process
- `stop_chunking()` - Stop chunking process
- `add_audio_data(audio_data: bytes, timestamp: float)` - Add audio data
- `get_next_chunk()` - Get next chunk for processing
- `get_chunk_queue_status()` - Get queue status
- `set_chunking_strategy(strategy: ChunkingStrategy)` - Set chunking strategy

**Usage Example:**
```python
from src.analysis.audio_chunker import AudioChunker, ChunkConfig, ChunkingStrategy

# Create chunking configuration
chunk_config = ChunkConfig(
    strategy=ChunkingStrategy.HYBRID,
    chunk_duration=30.0,
    overlap_duration=1.0,
    min_chunk_duration=10.0,
    max_chunk_duration=60.0,
    enable_quality_based=True,
    priority_queue=True
)

# Create audio chunker
audio_chunker = AudioChunker(chunk_config)

# Start chunking
audio_chunker.start_chunking()

# Add audio data
for frame in audio_stream:
    audio_chunker.add_audio_data(frame.data, frame.timestamp)

# Process chunks
while True:
    chunk = audio_chunker.get_next_chunk()
    if chunk:
        print(f"Processing chunk {chunk.chunk_id}: "
              f"{chunk.duration:.1f}s, quality={chunk.quality_score:.2f}")
        
        # Process chunk (e.g., send to transcription)
        process_chunk(chunk)
    else:
        break

# Get queue status
status = audio_chunker.get_chunk_queue_status()
print(f"Queue size: {status['queue_size']}")
print(f"Processed chunks: {status['processed_count']}")
```

## Module Integration

The Analysis module integrates with other Silent Steno components:

1. **Audio Module**: Receives real-time audio streams for analysis
2. **AI Module**: Provides optimized audio chunks for transcription
3. **Core Events**: Publishes analysis events and statistics
4. **Data Module**: Stores analysis results and statistics
5. **UI Module**: Displays real-time analysis results and statistics

## Common Usage Patterns

### Complete Real-Time Analysis Setup
```python
# Create comprehensive analysis system
from src.analysis import create_integrated_analyzer

# Configure all components
config = {
    "vad": {
        "mode": "aggressive",
        "enable_smoothing": True,
        "min_voice_duration": 0.5
    },
    "speaker": {
        "method": "combined",
        "max_speakers": 8,
        "min_segment_duration": 2.0
    },
    "quality": {
        "enable_real_time": True,
        "alert_threshold": 0.7,
        "metrics": ["snr", "thd", "clarity"]
    },
    "chunking": {
        "strategy": "hybrid",
        "chunk_duration": 30.0,
        "enable_quality_based": True
    },
    "statistics": {
        "enable_real_time": True,
        "track_interruptions": True,
        "update_interval": 1.0
    }
}

# Create analyzer
analyzer = create_integrated_analyzer(config)

# Set up callbacks
def on_voice_activity(is_active, timestamp):
    print(f"Voice {'started' if is_active else 'stopped'} at {timestamp:.2f}s")

def on_speaker_change(old_speaker, new_speaker, timestamp):
    print(f"Speaker change: {old_speaker} -> {new_speaker} at {timestamp:.2f}s")

def on_quality_alert(quality_score, timestamp):
    print(f"Quality alert: {quality_score:.2f} at {timestamp:.2f}s")

analyzer.set_callback("voice_activity", on_voice_activity)
analyzer.set_callback("speaker_change", on_speaker_change)
analyzer.set_callback("quality_alert", on_quality_alert)

# Start analysis
analyzer.start_processing()

# Process audio stream
for frame in audio_stream:
    results = analyzer.analyze_audio_frame(frame.data, frame.timestamp)
    
    # Handle analysis results
    process_analysis_results(results)

# Get final statistics
final_stats = analyzer.get_statistics()
print(f"Final Statistics:")
print(f"  Total duration: {final_stats['total_duration']:.1f}s")
print(f"  Speaking time: {final_stats['speaking_time']:.1f}s")
print(f"  Number of speakers: {final_stats['speaker_count']}")
print(f"  Average quality: {final_stats['average_quality']:.2f}")
```

### Meeting Analytics Pipeline
```python
# Create meeting analytics pipeline
def create_meeting_analytics_pipeline():
    # Initialize components
    vad = create_vad_analyzer({"mode": "aggressive"})
    speaker_detector = create_speaker_analyzer({"method": "combined"})
    stats_collector = StatisticsCollector()
    
    # Start all components
    vad.start_detection()
    speaker_detector.start_detection()
    stats_collector.start_collection()
    
    # Real-time analytics
    current_speaker = None
    speaking_start_time = None
    
    def analyze_frame(audio_data, timestamp):
        # Voice activity detection
        vad_result = vad.detect_voice_activity(audio_data, timestamp)
        
        # Speaker detection
        speaker_result = speaker_detector.detect_speaker_change(audio_data, timestamp)
        
        # Update statistics
        if vad_result.voice_detected:
            if speaking_start_time is None:
                speaking_start_time = timestamp
            
            if speaker_result.speaker_changed:
                # Update previous speaker stats
                if current_speaker and speaking_start_time:
                    speaking_duration = timestamp - speaking_start_time
                    stats_collector.update_speaker_stats(current_speaker, speaking_duration)
                
                # Check for interruption
                if current_speaker and current_speaker != speaker_result.speaker_id:
                    stats_collector.record_interruption(speaker_result.speaker_id, current_speaker)
                
                current_speaker = speaker_result.speaker_id
                speaking_start_time = timestamp
        else:
            # Voice ended
            if current_speaker and speaking_start_time:
                speaking_duration = timestamp - speaking_start_time
                stats_collector.update_speaker_stats(current_speaker, speaking_duration)
                speaking_start_time = None
    
    return analyze_frame, stats_collector

# Use analytics pipeline
analyze_frame, stats_collector = create_meeting_analytics_pipeline()

# Process meeting
for frame in meeting_audio_stream:
    analyze_frame(frame.data, frame.timestamp)

# Get final analytics
meeting_stats = stats_collector.get_meeting_statistics()
speaker_stats = stats_collector.get_speaker_statistics()

print("Meeting Analytics:")
print(f"  Duration: {meeting_stats.total_duration:.1f}s")
print(f"  Speaking time: {meeting_stats.speaking_duration:.1f}s")
print(f"  Speakers: {meeting_stats.speaker_count}")
print(f"  Interruptions: {meeting_stats.interruption_count}")
print(f"  Participation balance: {meeting_stats.participation_balance:.2f}")

print("\nSpeaker Statistics:")
for speaker_id, stats in speaker_stats.items():
    print(f"  {speaker_id}:")
    print(f"    Speaking time: {stats.speaking_time:.1f}s")
    print(f"    Turn count: {stats.turn_count}")
    print(f"    Interruptions: {stats.interruptions}")
```

This comprehensive Analysis module provides sophisticated real-time audio analysis capabilities that enhance The Silent Steno's transcription quality and provide valuable meeting insights through advanced signal processing and machine learning techniques.