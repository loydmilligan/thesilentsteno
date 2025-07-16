# AI Module Documentation

## Module Overview

The AI module provides the core artificial intelligence capabilities for The Silent Steno, including audio transcription, speaker diarization, meeting analysis, and content formatting. It integrates Whisper for speech recognition with additional AI components for comprehensive meeting analysis and participant insights.

## Dependencies

### External Dependencies
- `whisper` - OpenAI Whisper for speech-to-text
- `torch` - PyTorch for machine learning models
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `librosa` - Audio analysis
- `soundfile` - Audio file I/O
- `pyaudio` - Audio stream handling
- `webrtcvad` - Voice activity detection
- `scikit-learn` - Machine learning utilities
- `transformers` - Hugging Face transformers
- `psutil` - System monitoring
- `asyncio` - Asynchronous operations
- `threading` - Thread management
- `json` - JSON processing
- `datetime` - Date/time operations
- `pathlib` - Path operations

### Internal Dependencies
- `src.core.events` - Event system
- `src.core.logging` - Logging system
- `src.core.monitoring` - Performance monitoring
- `src.core.errors` - Error handling
- `src.llm.local_llm_processor` - Local LLM processing
- `src.analysis.voice_activity_detector` - Voice activity detection
- `src.analysis.quality_assessor` - Audio quality assessment

## File Documentation

### 1. `__init__.py`

**Purpose**: Module initialization and main AI system integration providing pre-configured setups for different use cases.

#### Classes

##### `AITranscriptionSystem`
Main AI transcription system integrating all components.

**Attributes:**
- `config: dict` - System configuration
- `transcription_pipeline: TranscriptionPipeline` - Main transcription pipeline
- `analysis_pipeline: AnalysisPipeline` - Analysis pipeline
- `performance_optimizer: PerformanceOptimizer` - Performance optimization
- `status_tracker: StatusTracker` - System status monitoring
- `is_running: bool` - System running state

**Methods:**
- `__init__(config: dict)` - Initialize with configuration
- `initialize()` - Initialize all components
- `start()` - Start the AI system
- `stop()` - Stop the AI system
- `process_audio(audio_data: bytes, sample_rate: int)` - Process audio chunk
- `process_audio_file(file_path: str)` - Process audio file
- `get_status()` - Get system status
- `get_performance_metrics()` - Get performance metrics

#### Factory Functions

##### `create_meeting_ai_system(config: dict = None) -> AITranscriptionSystem`
Create AI system optimized for meeting scenarios.

**Features:**
- Speaker diarization enabled
- Meeting analysis with action items
- Participant interaction tracking
- Real-time processing optimized

##### `create_interview_ai_system(config: dict = None) -> AITranscriptionSystem`
Create AI system optimized for interview scenarios.

**Features:**
- High transcription accuracy
- Question-answer analysis
- Participant role identification
- Interview-specific formatting

##### `create_lecture_ai_system(config: dict = None) -> AITranscriptionSystem`
Create AI system optimized for lecture scenarios.

**Features:**
- Long-form content processing
- Topic identification
- Key point extraction
- Educational content analysis

**Usage Example:**
```python
# Create meeting-optimized AI system
ai_system = create_meeting_ai_system({
    "whisper_model": "base",
    "enable_speaker_diarization": True,
    "real_time_processing": True
})

# Initialize and start
ai_system.initialize()
ai_system.start()

# Process audio
with open("meeting.wav", "rb") as f:
    audio_data = f.read()
    result = ai_system.process_audio(audio_data, 16000)
    print(result.transcript)
    print(result.analysis.summary)
```

### 2. `analysis_pipeline.py`

**Purpose**: Main analysis pipeline orchestrating the complete AI analysis workflow from audio to insights.

#### Classes

##### `PipelineConfig`
Configuration for pipeline processing.

**Attributes:**
- `whisper_model: str` - Whisper model size ("tiny", "base", "small", "medium", "large")
- `enable_speaker_diarization: bool` - Enable speaker identification
- `enable_meeting_analysis: bool` - Enable meeting-specific analysis
- `enable_participant_analysis: bool` - Enable participant analysis
- `chunk_length: float` - Audio chunk length in seconds
- `overlap_length: float` - Chunk overlap length
- `confidence_threshold: float` - Minimum confidence threshold
- `output_formats: List[str]` - Output formats to generate
- `real_time_processing: bool` - Enable real-time processing

##### `PipelineResult`
Complete pipeline processing result.

**Attributes:**
- `transcript: str` - Complete transcript
- `segments: List[TranscriptSegment]` - Transcript segments
- `speaker_labels: Dict[str, str]` - Speaker identification
- `analysis: AnalysisResult` - Meeting analysis results
- `participant_stats: Dict[str, ParticipantStats]` - Participant statistics
- `confidence_metrics: ConfidenceMetrics` - Confidence scores
- `processing_time: float` - Total processing time
- `metadata: dict` - Additional metadata

##### `AnalysisPipeline`
Main analysis pipeline orchestrator.

**Methods:**
- `__init__(config: PipelineConfig)` - Initialize with configuration
- `process_audio(audio_data: bytes, sample_rate: int)` - Process audio data
- `process_audio_file(file_path: str)` - Process audio file
- `process_streaming(audio_stream)` - Process streaming audio
- `set_callback(event_type: str, callback: callable)` - Set event callback
- `get_progress()` - Get processing progress
- `cancel_processing()` - Cancel current processing

**Usage Example:**
```python
# Create and configure pipeline
config = PipelineConfig(
    whisper_model="base",
    enable_speaker_diarization=True,
    enable_meeting_analysis=True,
    chunk_length=30.0,
    confidence_threshold=0.7
)

pipeline = AnalysisPipeline(config)

# Set progress callback
def on_progress(progress):
    print(f"Processing: {progress:.1%}")

pipeline.set_callback("progress", on_progress)

# Process audio file
result = pipeline.process_audio_file("meeting.wav")
print(f"Transcript: {result.transcript}")
print(f"Speakers: {list(result.speaker_labels.keys())}")
print(f"Summary: {result.analysis.summary}")
```

### 3. `audio_chunker.py`

**Purpose**: Intelligent audio segmentation for optimal transcription quality using AI-driven analysis.

#### Classes

##### `AudioAnalysis`
Audio quality and content analysis.

**Attributes:**
- `signal_strength: float` - Signal strength (0-1)
- `noise_level: float` - Background noise level
- `voice_activity: float` - Voice activity ratio
- `speech_clarity: float` - Speech clarity score
- `optimal_chunk_size: float` - Recommended chunk size
- `quality_score: float` - Overall quality score

##### `OptimalChunk`
Optimized audio chunk with metadata.

**Attributes:**
- `audio_data: bytes` - Audio data
- `start_time: float` - Start time in seconds
- `end_time: float` - End time in seconds
- `duration: float` - Chunk duration
- `analysis: AudioAnalysis` - Audio analysis
- `chunk_id: str` - Unique chunk identifier
- `processing_priority: int` - Processing priority

##### `AIAudioChunker`
Intelligent audio chunking engine.

**Methods:**
- `__init__(vad_aggressiveness: int = 2, chunk_length: float = 30.0)` - Initialize chunker
- `chunk_audio(audio_data: bytes, sample_rate: int)` - Chunk audio data
- `analyze_audio_quality(audio_data: bytes, sample_rate: int)` - Analyze audio quality
- `find_optimal_boundaries(audio_data: bytes, sample_rate: int)` - Find optimal chunk boundaries
- `merge_short_chunks(chunks: List[OptimalChunk])` - Merge short chunks
- `set_adaptive_chunking(enabled: bool)` - Enable adaptive chunking

**Usage Example:**
```python
# Create audio chunker
chunker = AIAudioChunker(
    vad_aggressiveness=2,
    chunk_length=30.0
)

# Enable adaptive chunking
chunker.set_adaptive_chunking(True)

# Chunk audio
with open("audio.wav", "rb") as f:
    audio_data = f.read()
    chunks = chunker.chunk_audio(audio_data, 16000)

# Process chunks by priority
chunks.sort(key=lambda x: x.processing_priority)
for chunk in chunks:
    print(f"Chunk {chunk.chunk_id}: {chunk.duration:.1f}s, "
          f"Quality: {chunk.analysis.quality_score:.2f}")
```

### 4. `confidence_scorer.py`

**Purpose**: Comprehensive confidence and quality scoring for AI outputs including transcription, analysis, and speaker identification.

#### Classes

##### `ConfidenceMetrics`
Comprehensive confidence metrics.

**Attributes:**
- `overall_confidence: float` - Overall confidence score (0-1)
- `transcription_confidence: float` - Transcription confidence
- `speaker_confidence: float` - Speaker identification confidence
- `analysis_confidence: float` - Analysis confidence
- `word_confidences: List[float]` - Word-level confidence scores
- `segment_confidences: List[float]` - Segment-level confidence scores
- `quality_indicators: dict` - Quality indicators
- `reliability_score: float` - Reliability assessment

##### `QualityAssessment`
Component-specific quality evaluation.

**Attributes:**
- `audio_quality: float` - Audio quality score
- `speech_clarity: float` - Speech clarity score
- `background_noise: float` - Background noise level
- `speaker_consistency: float` - Speaker consistency score
- `language_model_score: float` - Language model confidence
- `temporal_consistency: float` - Temporal consistency score

##### `ConfidenceScorer`
Main confidence scoring system.

**Methods:**
- `__init__(model_type: str = "whisper")` - Initialize with model type
- `score_transcription(transcription_result)` - Score transcription confidence
- `score_speaker_diarization(diarization_result)` - Score speaker identification
- `score_analysis(analysis_result)` - Score analysis confidence
- `calculate_overall_confidence(components: dict)` - Calculate overall confidence
- `assess_quality(audio_data: bytes, sample_rate: int)` - Assess audio quality
- `validate_result(result, thresholds: dict)` - Validate result quality

**Usage Example:**
```python
# Create confidence scorer
scorer = ConfidenceScorer(model_type="whisper")

# Score transcription
transcription_result = whisper_transcriber.transcribe(audio_data)
confidence = scorer.score_transcription(transcription_result)

print(f"Overall confidence: {confidence.overall_confidence:.2f}")
print(f"Transcription confidence: {confidence.transcription_confidence:.2f}")

# Assess quality
quality = scorer.assess_quality(audio_data, 16000)
print(f"Audio quality: {quality.audio_quality:.2f}")
print(f"Speech clarity: {quality.speech_clarity:.2f}")

# Validate result
is_valid = scorer.validate_result(transcription_result, {
    "min_confidence": 0.7,
    "min_quality": 0.6
})
```

### 5. `meeting_analyzer.py`

**Purpose**: Meeting-specific analysis and LLM integration for generating summaries, action items, and insights.

#### Classes

##### `MeetingMetadata`
Meeting context and metadata.

**Attributes:**
- `meeting_id: str` - Unique meeting identifier
- `participants: List[str]` - Participant names
- `duration: float` - Meeting duration
- `start_time: datetime` - Meeting start time
- `meeting_type: str` - Type of meeting
- `agenda_items: List[str]` - Agenda items
- `context: dict` - Additional context

##### `AnalysisResult`
Comprehensive meeting analysis results.

**Attributes:**
- `summary: str` - Meeting summary
- `key_topics: List[str]` - Key topics discussed
- `action_items: List[dict]` - Action items with assignees
- `decisions: List[str]` - Decisions made
- `next_steps: List[str]` - Next steps
- `participant_insights: dict` - Participant-specific insights
- `meeting_effectiveness: float` - Meeting effectiveness score
- `sentiment_analysis: dict` - Sentiment analysis results

##### `MeetingAnalyzer`
Meeting analysis orchestrator.

**Methods:**
- `__init__(llm_processor, config: dict = None)` - Initialize with LLM processor
- `analyze_meeting(transcript: str, metadata: MeetingMetadata)` - Analyze complete meeting
- `analyze_segment(segment: str, context: dict)` - Analyze transcript segment
- `extract_action_items(transcript: str)` - Extract action items
- `generate_summary(transcript: str, length: str = "medium")` - Generate summary
- `identify_topics(transcript: str)` - Identify key topics
- `analyze_sentiment(transcript: str)` - Analyze sentiment
- `assess_meeting_effectiveness(transcript: str, metadata: MeetingMetadata)` - Assess effectiveness

**Usage Example:**
```python
# Create meeting analyzer
from src.llm.local_llm_processor import LocalLLMProcessor
llm_processor = LocalLLMProcessor()
analyzer = MeetingAnalyzer(llm_processor)

# Prepare meeting metadata
metadata = MeetingMetadata(
    meeting_id="meet_123",
    participants=["Alice", "Bob", "Charlie"],
    duration=3600.0,
    meeting_type="team_standup"
)

# Analyze meeting
result = analyzer.analyze_meeting(transcript, metadata)

print(f"Summary: {result.summary}")
print(f"Action Items: {len(result.action_items)}")
print(f"Key Topics: {result.key_topics}")
print(f"Effectiveness: {result.meeting_effectiveness:.2f}")
```

### 6. `participant_analyzer.py`

**Purpose**: Participant analysis and interaction pattern recognition for understanding meeting dynamics.

#### Classes

##### `ParticipantStats`
Individual participant statistics.

**Attributes:**
- `name: str` - Participant name
- `speaking_time: float` - Total speaking time
- `word_count: int` - Total word count
- `interruptions: int` - Number of interruptions
- `questions_asked: int` - Questions asked
- `contributions: int` - Number of contributions
- `sentiment_score: float` - Average sentiment score
- `engagement_level: float` - Engagement level (0-1)

##### `EngagementMetrics`
Engagement and interaction analysis.

**Attributes:**
- `participation_balance: float` - Participation balance score
- `interaction_frequency: float` - Interaction frequency
- `collaborative_score: float` - Collaboration score
- `discussion_quality: float` - Discussion quality
- `turn_taking_efficiency: float` - Turn-taking efficiency
- `dominant_speakers: List[str]` - Most dominant speakers

##### `ParticipantAnalyzer`
Main participant analysis system.

**Methods:**
- `__init__(config: dict = None)` - Initialize analyzer
- `analyze_participants(transcript_segments: List, speaker_labels: dict)` - Analyze participants
- `calculate_speaking_time(segments: List, speaker: str)` - Calculate speaking time
- `detect_interruptions(segments: List)` - Detect interruptions
- `analyze_interaction_patterns(segments: List)` - Analyze interaction patterns
- `assess_engagement(participant_stats: ParticipantStats)` - Assess engagement
- `generate_insights(participant_stats: dict)` - Generate insights

**Usage Example:**
```python
# Create participant analyzer
analyzer = ParticipantAnalyzer()

# Analyze participants
participant_stats = analyzer.analyze_participants(
    transcript_segments, 
    speaker_labels
)

# Get insights
for name, stats in participant_stats.items():
    print(f"{name}:")
    print(f"  Speaking time: {stats.speaking_time:.1f}s")
    print(f"  Word count: {stats.word_count}")
    print(f"  Engagement: {stats.engagement_level:.2f}")
    print(f"  Questions: {stats.questions_asked}")
```

### 7. `performance_optimizer.py`

**Purpose**: System performance optimization specifically tuned for Raspberry Pi 5 hardware constraints.

#### Classes

##### `SystemMonitor`
Real-time system monitoring.

**Attributes:**
- `cpu_usage: float` - Current CPU usage percentage
- `memory_usage: float` - Current memory usage percentage
- `temperature: float` - CPU temperature in Celsius
- `disk_usage: float` - Disk usage percentage
- `network_usage: dict` - Network usage statistics
- `gpu_usage: float` - GPU usage percentage

##### `OptimizationResult`
Optimization results and metrics.

**Attributes:**
- `performance_gain: float` - Performance improvement percentage
- `memory_saved: int` - Memory saved in bytes
- `cpu_reduction: float` - CPU usage reduction percentage
- `optimizations_applied: List[str]` - Applied optimizations
- `recommendations: List[str]` - Additional recommendations
- `stability_score: float` - System stability score

##### `PerformanceOptimizer`
Main performance optimization engine.

**Methods:**
- `__init__(hardware_profile: str = "pi5")` - Initialize with hardware profile
- `optimize_system()` - Optimize system performance
- `optimize_for_transcription()` - Optimize for transcription workload
- `optimize_for_analysis()` - Optimize for analysis workload
- `monitor_performance()` - Monitor real-time performance
- `adjust_model_settings(workload_type: str)` - Adjust model settings
- `manage_thermal_throttling()` - Manage thermal throttling
- `optimize_memory_usage()` - Optimize memory usage

**Usage Example:**
```python
# Create performance optimizer
optimizer = PerformanceOptimizer(hardware_profile="pi5")

# Optimize system
result = optimizer.optimize_system()
print(f"Performance gain: {result.performance_gain:.1f}%")
print(f"Memory saved: {result.memory_saved / 1024 / 1024:.1f} MB")

# Optimize for transcription
optimizer.optimize_for_transcription()

# Monitor performance
monitor = optimizer.monitor_performance()
print(f"CPU usage: {monitor.cpu_usage:.1f}%")
print(f"Temperature: {monitor.temperature:.1f}°C")
```

### 8. `speaker_diarizer.py`

**Purpose**: Speaker identification and diarization for multi-speaker audio content.

#### Classes

##### `SpeakerSegment`
Speaker-labeled audio segment.

**Attributes:**
- `start_time: float` - Segment start time
- `end_time: float` - Segment end time
- `speaker_id: str` - Speaker identifier
- `confidence: float` - Identification confidence
- `text: str` - Transcribed text
- `speaker_name: str` - Speaker name if known

##### `DiarizationResult`
Complete diarization results.

**Attributes:**
- `segments: List[SpeakerSegment]` - Speaker segments
- `speaker_count: int` - Number of speakers
- `speaker_labels: Dict[str, str]` - Speaker ID to name mapping
- `speaker_stats: Dict[str, dict]` - Speaker statistics
- `overall_confidence: float` - Overall confidence score
- `processing_time: float` - Processing time

##### `SpeakerDiarizer`
Main speaker diarization engine.

**Methods:**
- `__init__(model_type: str = "pyannote", config: dict = None)` - Initialize diarizer
- `diarize_audio(audio_data: bytes, sample_rate: int)` - Perform diarization
- `identify_speakers(segments: List[SpeakerSegment])` - Identify speakers
- `merge_segments(segments: List[SpeakerSegment])` - Merge consecutive segments
- `assign_speaker_names(segments: List[SpeakerSegment], name_mapping: dict)` - Assign names
- `calculate_confidence(segments: List[SpeakerSegment])` - Calculate confidence

**Usage Example:**
```python
# Create speaker diarizer
diarizer = SpeakerDiarizer(model_type="pyannote")

# Perform diarization
result = diarizer.diarize_audio(audio_data, 16000)

print(f"Found {result.speaker_count} speakers")
print(f"Overall confidence: {result.overall_confidence:.2f}")

# Assign speaker names
name_mapping = {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}
diarizer.assign_speaker_names(result.segments, name_mapping)

# Print segments
for segment in result.segments:
    print(f"{segment.speaker_name}: {segment.text}")
```

### 9. `status_tracker.py`

**Purpose**: System status monitoring, health checking, and error handling for the AI pipeline.

#### Classes

##### `ProcessingStatus`
Current processing status.

**Attributes:**
- `stage: str` - Current processing stage
- `progress: float` - Progress percentage (0-1)
- `elapsed_time: float` - Elapsed processing time
- `estimated_remaining: float` - Estimated remaining time
- `current_operation: str` - Current operation description
- `error_count: int` - Number of errors encountered

##### `HealthCheck`
System health monitoring.

**Attributes:**
- `component: str` - Component name
- `status: str` - Health status ("healthy", "warning", "error")
- `last_check: float` - Last check timestamp
- `error_message: str` - Error message if any
- `performance_metrics: dict` - Performance metrics
- `recommendations: List[str]` - Recommendations

##### `StatusTracker`
Main status tracking system.

**Methods:**
- `__init__(config: dict = None)` - Initialize tracker
- `start_tracking(operation_name: str)` - Start tracking operation
- `update_progress(progress: float, description: str)` - Update progress
- `log_error(error: Exception, context: dict)` - Log error
- `check_system_health()` - Check system health
- `get_status()` - Get current status
- `set_callback(event_type: str, callback: callable)` - Set status callback

**Usage Example:**
```python
# Create status tracker
tracker = StatusTracker()

# Set progress callback
def on_progress(status):
    print(f"Progress: {status.progress:.1%} - {status.current_operation}")

tracker.set_callback("progress", on_progress)

# Start tracking
tracker.start_tracking("audio_transcription")

# Update progress
tracker.update_progress(0.5, "Processing audio chunk 5/10")

# Check health
health = tracker.check_system_health()
for check in health:
    print(f"{check.component}: {check.status}")
```

### 10. `transcript_formatter.py`

**Purpose**: Multi-format transcript output generation supporting various export formats.

#### Classes

##### `TranscriptSegment`
Individual transcript segment.

**Attributes:**
- `start_time: float` - Segment start time
- `end_time: float` - Segment end time
- `text: str` - Transcribed text
- `speaker: str` - Speaker name/ID
- `confidence: float` - Transcription confidence
- `word_timestamps: List[dict]` - Word-level timestamps

##### `FormattedTranscript`
Complete formatted transcript.

**Attributes:**
- `content: str` - Formatted content
- `format_type: str` - Format type
- `metadata: dict` - Transcript metadata
- `file_extension: str` - File extension
- `mime_type: str` - MIME type
- `size: int` - Content size in bytes

##### `TranscriptFormatter`
Main transcript formatting engine.

**Methods:**
- `__init__(config: dict = None)` - Initialize formatter
- `format_transcript(segments: List[TranscriptSegment], format_type: str)` - Format transcript
- `generate_text(segments: List[TranscriptSegment])` - Generate plain text
- `generate_json(segments: List[TranscriptSegment])` - Generate JSON format
- `generate_srt(segments: List[TranscriptSegment])` - Generate SRT subtitles
- `generate_vtt(segments: List[TranscriptSegment])` - Generate VTT subtitles
- `generate_html(segments: List[TranscriptSegment])` - Generate HTML format
- `generate_pdf(segments: List[TranscriptSegment])` - Generate PDF format

**Usage Example:**
```python
# Create transcript formatter
formatter = TranscriptFormatter()

# Format as different types
text_transcript = formatter.format_transcript(segments, "text")
json_transcript = formatter.format_transcript(segments, "json")
srt_transcript = formatter.format_transcript(segments, "srt")

# Save to files
with open("transcript.txt", "w") as f:
    f.write(text_transcript.content)

with open("transcript.srt", "w") as f:
    f.write(srt_transcript.content)

print(f"Text size: {text_transcript.size} bytes")
print(f"SRT size: {srt_transcript.size} bytes")
```

### 11. `transcription_pipeline.py`

**Purpose**: Real-time transcription pipeline orchestrating the complete transcription workflow.

#### Classes

##### `ChunkResult`
Individual chunk processing result.

**Attributes:**
- `chunk_id: str` - Chunk identifier
- `transcript: str` - Transcribed text
- `start_time: float` - Chunk start time
- `end_time: float` - Chunk end time
- `confidence: float` - Transcription confidence
- `processing_time: float` - Processing time
- `speaker_id: str` - Speaker identifier
- `word_timestamps: List[dict]` - Word-level timestamps

##### `PipelineResult`
Complete pipeline processing result.

**Attributes:**
- `transcript: str` - Complete transcript
- `segments: List[ChunkResult]` - Processed chunks
- `total_duration: float` - Total audio duration
- `processing_time: float` - Total processing time
- `average_confidence: float` - Average confidence score
- `speaker_labels: Dict[str, str]` - Speaker mappings
- `metadata: dict` - Processing metadata

##### `TranscriptionPipeline`
Main transcription pipeline orchestrator.

**Methods:**
- `__init__(config: dict)` - Initialize pipeline
- `process_audio_stream(audio_stream)` - Process streaming audio
- `process_audio_buffer(audio_buffer: bytes, sample_rate: int)` - Process audio buffer
- `process_audio_file(file_path: str)` - Process audio file
- `set_callback(event_type: str, callback: callable)` - Set pipeline callback
- `start_real_time_processing()` - Start real-time processing
- `stop_processing()` - Stop processing
- `get_partial_results()` - Get partial results

**Usage Example:**
```python
# Create transcription pipeline
config = {
    "whisper_model": "base",
    "chunk_length": 30.0,
    "enable_speaker_diarization": True,
    "real_time_processing": True
}

pipeline = TranscriptionPipeline(config)

# Set result callback
def on_result(result):
    print(f"Transcription: {result.transcript}")
    print(f"Confidence: {result.confidence:.2f}")

pipeline.set_callback("chunk_processed", on_result)

# Process audio file
result = pipeline.process_audio_file("meeting.wav")
print(f"Total duration: {result.total_duration:.1f}s")
print(f"Processing time: {result.processing_time:.1f}s")
print(f"Average confidence: {result.average_confidence:.2f}")
```

### 12. `whisper_transcriber.py`

**Purpose**: Local Whisper model integration optimized for Raspberry Pi 5 with various model sizes and configurations.

#### Classes

##### `TranscriptionConfig`
Whisper transcription configuration.

**Attributes:**
- `model_size: str` - Model size ("tiny", "base", "small", "medium", "large")
- `language: str` - Language code or "auto"
- `temperature: float` - Sampling temperature
- `compression_ratio_threshold: float` - Compression ratio threshold
- `logprob_threshold: float` - Log probability threshold
- `no_speech_threshold: float` - No speech threshold
- `condition_on_previous_text: bool` - Use previous text context
- `word_timestamps: bool` - Generate word-level timestamps

##### `TranscriptionResult`
Whisper transcription result.

**Attributes:**
- `text: str` - Transcribed text
- `segments: List[dict]` - Segment-level results
- `language: str` - Detected language
- `confidence: float` - Overall confidence score
- `processing_time: float` - Processing time
- `word_timestamps: List[dict]` - Word-level timestamps
- `no_speech_prob: float` - No speech probability

##### `WhisperTranscriber`
Main Whisper transcription engine.

**Methods:**
- `__init__(config: TranscriptionConfig)` - Initialize with configuration
- `load_model()` - Load Whisper model
- `transcribe_audio(audio_data: bytes, sample_rate: int)` - Transcribe audio
- `transcribe_file(file_path: str)` - Transcribe audio file
- `transcribe_streaming(audio_stream)` - Transcribe streaming audio
- `detect_language(audio_data: bytes)` - Detect audio language
- `optimize_for_hardware()` - Optimize for Pi 5 hardware
- `unload_model()` - Unload model from memory

**Usage Example:**
```python
# Create transcription config
config = TranscriptionConfig(
    model_size="base",
    language="en",
    word_timestamps=True,
    temperature=0.0
)

# Create transcriber
transcriber = WhisperTranscriber(config)

# Load model
transcriber.load_model()

# Optimize for Pi 5
transcriber.optimize_for_hardware()

# Transcribe audio
with open("audio.wav", "rb") as f:
    audio_data = f.read()
    result = transcriber.transcribe_audio(audio_data, 16000)

print(f"Transcription: {result.text}")
print(f"Language: {result.language}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Processing time: {result.processing_time:.1f}s")

# Process word timestamps
for word in result.word_timestamps:
    print(f"{word['word']}: {word['start']:.1f}s - {word['end']:.1f}s")
```

## Module Integration

The AI module integrates with other parts of the Silent Steno system:

1. **Audio Pipeline**: Receives audio data from the audio recording system
2. **Event System**: Publishes processing events and status updates
3. **Storage**: Stores transcripts and analysis results
4. **UI**: Provides real-time updates to the user interface
5. **Export**: Formats transcripts for export in multiple formats
6. **LLM Module**: Uses local LLM for meeting analysis and insights

## Common Usage Patterns

### Complete AI Processing Workflow
```python
# 1. Create AI system
ai_system = create_meeting_ai_system({
    "whisper_model": "base",
    "enable_speaker_diarization": True,
    "enable_meeting_analysis": True,
    "output_formats": ["text", "json", "srt"]
})

# 2. Initialize system
ai_system.initialize()

# 3. Set up callbacks
def on_progress(progress):
    print(f"Processing: {progress:.1%}")

def on_result(result):
    print(f"Chunk completed: {result.transcript}")

ai_system.set_callback("progress", on_progress)
ai_system.set_callback("chunk_processed", on_result)

# 4. Process audio
result = ai_system.process_audio_file("meeting.wav")

# 5. Access results
print(f"Transcript: {result.transcript}")
print(f"Speakers: {list(result.speaker_labels.keys())}")
print(f"Summary: {result.analysis.summary}")
print(f"Action Items: {len(result.analysis.action_items)}")
```

### Real-time Processing
```python
# Create real-time pipeline
pipeline = TranscriptionPipeline({
    "whisper_model": "base",
    "chunk_length": 10.0,
    "real_time_processing": True
})

# Start real-time processing
pipeline.start_real_time_processing()

# Process audio stream
async def process_audio_stream():
    while True:
        audio_chunk = await get_audio_chunk()
        result = pipeline.process_audio_buffer(audio_chunk, 16000)
        if result:
            display_transcript(result.transcript)

# Stop processing
pipeline.stop_processing()
```

### Custom Configuration
```python
# Create custom configuration
config = {
    "whisper": {
        "model_size": "small",
        "language": "en",
        "temperature": 0.0,
        "word_timestamps": True
    },
    "chunking": {
        "chunk_length": 30.0,
        "overlap_length": 1.0,
        "adaptive_chunking": True
    },
    "speaker_diarization": {
        "enabled": True,
        "min_speakers": 2,
        "max_speakers": 10
    },
    "analysis": {
        "enable_meeting_analysis": True,
        "enable_participant_analysis": True,
        "confidence_threshold": 0.7
    },
    "performance": {
        "hardware_profile": "pi5",
        "optimize_for_transcription": True,
        "thermal_management": True
    }
}

# Create system with custom config
ai_system = AITranscriptionSystem(config)
```

### Error Handling and Recovery
```python
# Error handling pattern
try:
    result = ai_system.process_audio_file("audio.wav")
except Exception as e:
    # Check error type
    if isinstance(e, TranscriptionError):
        # Handle transcription errors
        fallback_result = ai_system.process_with_fallback("audio.wav")
    elif isinstance(e, ModelLoadError):
        # Handle model loading errors
        ai_system.reload_model()
        result = ai_system.process_audio_file("audio.wav")
    else:
        # Handle other errors
        logger.error(f"Unexpected error: {e}")
        result = None

# Monitor system health
health = ai_system.get_system_health()
if health.status != "healthy":
    print(f"System health: {health.status}")
    print(f"Recommendations: {health.recommendations}")
```

### Performance Optimization
```python
# Optimize for specific hardware
optimizer = PerformanceOptimizer(hardware_profile="pi5")
optimizer.optimize_for_transcription()

# Monitor performance
def monitor_performance():
    while True:
        metrics = optimizer.monitor_performance()
        if metrics.temperature > 80:
            optimizer.manage_thermal_throttling()
        if metrics.memory_usage > 0.9:
            optimizer.optimize_memory_usage()
        time.sleep(10)

# Run monitoring in background
threading.Thread(target=monitor_performance, daemon=True).start()
```

### Multi-format Output
```python
# Generate multiple formats
formatter = TranscriptFormatter()
segments = pipeline.get_segments()

# Generate all formats
formats = {
    "text": formatter.generate_text(segments),
    "json": formatter.generate_json(segments),
    "srt": formatter.generate_srt(segments),
    "vtt": formatter.generate_vtt(segments),
    "html": formatter.generate_html(segments)
}

# Save to files
for format_type, content in formats.items():
    filename = f"transcript.{content.file_extension}"
    with open(filename, "w") as f:
        f.write(content.content)
    print(f"Saved {format_type} format: {filename}")
```

This comprehensive documentation provides complete technical details and practical usage examples for all components in the AI module, making it easy for developers to understand and work with the transcription and analysis capabilities of The Silent Steno system.