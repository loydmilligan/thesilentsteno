# Audio Module Documentation

## Module Overview

The Audio module provides comprehensive real-time audio processing capabilities for The Silent Steno, focusing on low-latency (<40ms) Bluetooth A2DP audio capture and forwarding. It implements a complete audio pipeline with ALSA integration, format conversion, latency optimization, and quality monitoring.

## Dependencies

### External Dependencies
- `alsaaudio` - ALSA audio system interface
- `pyaudio` - Cross-platform audio I/O
- `numpy` - Numerical computing for audio processing
- `scipy` - Scientific computing and signal processing
- `soundfile` - Audio file I/O
- `librosa` - Audio analysis and processing
- `threading` - Thread management for real-time processing
- `queue` - Thread-safe data structures
- `time` - Timing and performance measurement
- `subprocess` - System command execution
- `psutil` - System resource monitoring
- `json` - Configuration handling
- `pathlib` - Path operations
- `logging` - Logging system
- `dataclasses` - Data structure definitions
- `enum` - Enumeration types
- `typing` - Type hints

### Internal Dependencies
- `src.core.events` - Event system
- `src.core.logging` - Logging system
- `src.core.monitoring` - Performance monitoring
- `src.core.errors` - Error handling

## File Documentation

### 1. `__init__.py`

**Purpose**: Module initialization and public API exposure with convenience functions for common audio operations.

#### Classes

##### `AudioSystem`
Main audio system providing unified access to all audio components.

**Attributes:**
- `pipeline: AudioPipeline` - Main audio pipeline
- `alsa_manager: ALSAManager` - ALSA system manager
- `latency_optimizer: LatencyOptimizer` - Latency optimization
- `level_monitor: LevelMonitor` - Audio level monitoring
- `format_converter: FormatConverter` - Audio format conversion
- `config: dict` - System configuration
- `is_running: bool` - System running state

**Methods:**
- `__init__(config: dict = None)` - Initialize with configuration
- `initialize()` - Initialize all components
- `start()` - Start the audio system
- `stop()` - Stop the audio system
- `get_status()` - Get system status
- `get_performance_metrics()` - Get performance metrics
- `optimize_latency()` - Optimize system latency
- `get_audio_devices()` - Get available audio devices

#### Configuration Presets

##### `LOW_LATENCY_CONFIG`
Configuration optimized for minimal latency.
- Buffer size: 64 frames
- Sample rate: 44.1kHz
- Channels: 2 (stereo)
- Priority: Real-time
- Target latency: <20ms

##### `BALANCED_CONFIG`
Balanced configuration for quality and latency.
- Buffer size: 128 frames
- Sample rate: 44.1kHz
- Channels: 2 (stereo)
- Priority: High
- Target latency: <40ms

##### `HIGH_QUALITY_CONFIG`
Configuration optimized for audio quality.
- Buffer size: 256 frames
- Sample rate: 48kHz
- Channels: 2 (stereo)
- Priority: Normal
- Target latency: <60ms

#### Factory Functions

##### `create_low_latency_system(config: dict = None) -> AudioSystem`
Create audio system optimized for low latency.

##### `create_balanced_system(config: dict = None) -> AudioSystem`
Create audio system with balanced performance.

##### `create_high_quality_system(config: dict = None) -> AudioSystem`
Create audio system optimized for quality.

**Usage Example:**
```python
# Create low-latency audio system
audio_system = create_low_latency_system()

# Initialize and start
audio_system.initialize()
audio_system.start()

# Monitor performance
metrics = audio_system.get_performance_metrics()
print(f"Latency: {metrics.latency:.2f}ms")
print(f"CPU usage: {metrics.cpu_usage:.1f}%")

# Optimize if needed
if metrics.latency > 40.0:
    audio_system.optimize_latency()
```

### 2. `alsa_manager.py`

**Purpose**: Comprehensive ALSA (Advanced Linux Sound Architecture) management and configuration.

#### Classes

##### `ALSADevice`
ALSA audio device information.

**Attributes:**
- `name: str` - Device name
- `card_id: int` - ALSA card ID
- `device_id: int` - ALSA device ID
- `type: str` - Device type ("capture", "playback")
- `channels: int` - Number of channels
- `sample_rate: int` - Sample rate
- `format: str` - Audio format
- `latency: float` - Device latency in ms
- `is_default: bool` - Is default device

##### `ALSAConfig`
ALSA configuration settings.

**Attributes:**
- `buffer_size: int` - Buffer size in frames
- `period_size: int` - Period size in frames
- `sample_rate: int` - Sample rate in Hz
- `channels: int` - Number of channels
- `format: str` - Audio format
- `use_mmap: bool` - Use memory mapping
- `enable_resampling: bool` - Enable resampling
- `priority: int` - Thread priority

##### `ALSAManager`
Main ALSA system manager.

**Methods:**
- `__init__(config: ALSAConfig = None)` - Initialize with configuration
- `discover_devices()` - Discover available audio devices
- `get_device_info(device_name: str)` - Get device information
- `configure_device(device_name: str, config: ALSAConfig)` - Configure device
- `test_device_latency(device_name: str)` - Test device latency
- `create_alsa_config()` - Create ALSA configuration file
- `set_real_time_priority()` - Set real-time priority
- `optimize_buffer_sizes()` - Optimize buffer sizes
- `get_system_info()` - Get system audio information

**Usage Example:**
```python
# Create ALSA manager
alsa_config = ALSAConfig(
    buffer_size=128,
    period_size=64,
    sample_rate=44100,
    channels=2,
    format="S16_LE"
)

alsa_manager = ALSAManager(alsa_config)

# Discover devices
devices = alsa_manager.discover_devices()
for device in devices:
    print(f"Device: {device.name}")
    print(f"  Type: {device.type}")
    print(f"  Channels: {device.channels}")
    print(f"  Sample Rate: {device.sample_rate}")
    print(f"  Latency: {device.latency:.2f}ms")

# Configure device
alsa_manager.configure_device("hw:0,0", alsa_config)

# Test latency
latency = alsa_manager.test_device_latency("hw:0,0")
print(f"Measured latency: {latency:.2f}ms")
```

### 3. `audio_pipeline.py`

**Purpose**: Core audio pipeline orchestration managing real-time audio capture, processing, and forwarding.

#### Enums

##### `PipelineState`
Audio pipeline state enumeration.
- `STOPPED` - Pipeline stopped
- `STARTING` - Pipeline starting
- `RUNNING` - Pipeline running
- `STOPPING` - Pipeline stopping
- `ERROR` - Pipeline error state
- `PAUSED` - Pipeline paused

#### Classes

##### `AudioConfig`
Audio pipeline configuration.

**Attributes:**
- `input_device: str` - Input device name
- `output_device: str` - Output device name
- `sample_rate: int` - Sample rate in Hz
- `channels: int` - Number of channels
- `format: str` - Audio format
- `buffer_size: int` - Buffer size in frames
- `target_latency: float` - Target latency in ms
- `enable_monitoring: bool` - Enable monitoring
- `auto_restart: bool` - Enable auto-restart

##### `AudioMetrics`
Audio pipeline metrics.

**Attributes:**
- `latency: float` - Current latency in ms
- `cpu_usage: float` - CPU usage percentage
- `memory_usage: float` - Memory usage in MB
- `audio_level: float` - Audio level in dB
- `buffer_underruns: int` - Buffer underrun count
- `buffer_overruns: int` - Buffer overrun count
- `processing_time: float` - Processing time in ms
- `throughput: float` - Audio throughput in MB/s

##### `AudioPipeline`
Main audio pipeline orchestrator.

**Methods:**
- `__init__(config: AudioConfig)` - Initialize with configuration
- `start()` - Start the pipeline
- `stop()` - Stop the pipeline
- `pause()` - Pause the pipeline
- `resume()` - Resume the pipeline
- `get_state()` - Get current state
- `get_metrics()` - Get performance metrics
- `set_callback(event_type: str, callback: callable)` - Set event callback
- `process_audio_chunk(chunk: bytes)` - Process audio chunk
- `restart()` - Restart the pipeline

**Usage Example:**
```python
# Create audio configuration
config = AudioConfig(
    input_device="hw:1,0",
    output_device="hw:0,0",
    sample_rate=44100,
    channels=2,
    buffer_size=128,
    target_latency=30.0,
    enable_monitoring=True
)

# Create pipeline
pipeline = AudioPipeline(config)

# Set callbacks
def on_state_change(state):
    print(f"Pipeline state: {state}")

def on_metrics(metrics):
    print(f"Latency: {metrics.latency:.2f}ms")
    print(f"CPU: {metrics.cpu_usage:.1f}%")

pipeline.set_callback("state_change", on_state_change)
pipeline.set_callback("metrics", on_metrics)

# Start pipeline
pipeline.start()

# Monitor performance
while pipeline.get_state() == PipelineState.RUNNING:
    metrics = pipeline.get_metrics()
    if metrics.latency > 50.0:
        print("High latency detected!")
    time.sleep(1)
```

### 4. `format_converter.py`

**Purpose**: Real-time audio format conversion for compatibility between different audio systems and codecs.

#### Classes

##### `AudioFormat`
Audio format specification.

**Attributes:**
- `sample_rate: int` - Sample rate in Hz
- `channels: int` - Number of channels
- `bit_depth: int` - Bit depth (8, 16, 24, 32)
- `format: str` - Format type ("PCM", "SBC", "AAC", "aptX")
- `is_signed: bool` - Signed or unsigned
- `is_little_endian: bool` - Byte order
- `frame_size: int` - Frame size in bytes

##### `ConversionResult`
Audio format conversion result.

**Attributes:**
- `converted_data: bytes` - Converted audio data
- `input_format: AudioFormat` - Input format
- `output_format: AudioFormat` - Output format
- `conversion_time: float` - Conversion time in ms
- `quality_loss: float` - Quality loss percentage
- `success: bool` - Conversion success

##### `FormatConverter`
Main audio format converter.

**Methods:**
- `__init__(config: dict = None)` - Initialize converter
- `convert_audio(data: bytes, input_format: AudioFormat, output_format: AudioFormat)` - Convert audio
- `convert_sample_rate(data: bytes, input_rate: int, output_rate: int)` - Convert sample rate
- `convert_bit_depth(data: bytes, input_depth: int, output_depth: int)` - Convert bit depth
- `convert_channels(data: bytes, input_channels: int, output_channels: int)` - Convert channels
- `convert_codec(data: bytes, input_codec: str, output_codec: str)` - Convert codec
- `validate_format(format: AudioFormat)` - Validate format
- `get_supported_formats()` - Get supported formats

**Usage Example:**
```python
# Create format converter
converter = FormatConverter()

# Define input and output formats
input_format = AudioFormat(
    sample_rate=48000,
    channels=2,
    bit_depth=24,
    format="PCM"
)

output_format = AudioFormat(
    sample_rate=44100,
    channels=2,
    bit_depth=16,
    format="PCM"
)

# Convert audio data
with open("audio_48khz_24bit.raw", "rb") as f:
    audio_data = f.read()
    
result = converter.convert_audio(audio_data, input_format, output_format)

if result.success:
    print(f"Conversion successful!")
    print(f"Conversion time: {result.conversion_time:.2f}ms")
    print(f"Quality loss: {result.quality_loss:.1f}%")
    
    # Save converted audio
    with open("audio_44khz_16bit.raw", "wb") as f:
        f.write(result.converted_data)
```

### 5. `latency_optimizer.py`

**Purpose**: Latency measurement and optimization system targeting <40ms end-to-end latency.

#### Classes

##### `LatencyMeasurement`
Latency measurement result.

**Attributes:**
- `total_latency: float` - Total latency in ms
- `capture_latency: float` - Capture latency in ms
- `processing_latency: float` - Processing latency in ms
- `output_latency: float` - Output latency in ms
- `network_latency: float` - Network latency in ms
- `jitter: float` - Jitter in ms
- `timestamp: float` - Measurement timestamp

##### `OptimizationResult`
Latency optimization result.

**Attributes:**
- `initial_latency: float` - Initial latency before optimization
- `final_latency: float` - Final latency after optimization
- `improvement: float` - Improvement in ms
- `optimizations_applied: List[str]` - Applied optimizations
- `success: bool` - Optimization success
- `recommendations: List[str]` - Additional recommendations

##### `LatencyOptimizer`
Main latency optimization engine.

**Methods:**
- `__init__(target_latency: float = 40.0)` - Initialize with target latency
- `measure_latency()` - Measure current latency
- `optimize_latency()` - Optimize system latency
- `optimize_buffer_sizes()` - Optimize buffer sizes
- `optimize_thread_priority()` - Optimize thread priorities
- `optimize_system_settings()` - Optimize system settings
- `monitor_latency()` - Monitor latency continuously
- `get_latency_report()` - Get latency report
- `set_target_latency(target: float)` - Set target latency

**Usage Example:**
```python
# Create latency optimizer
optimizer = LatencyOptimizer(target_latency=35.0)

# Measure current latency
measurement = optimizer.measure_latency()
print(f"Current latency: {measurement.total_latency:.2f}ms")
print(f"  Capture: {measurement.capture_latency:.2f}ms")
print(f"  Processing: {measurement.processing_latency:.2f}ms")
print(f"  Output: {measurement.output_latency:.2f}ms")
print(f"  Jitter: {measurement.jitter:.2f}ms")

# Optimize if needed
if measurement.total_latency > 40.0:
    result = optimizer.optimize_latency()
    if result.success:
        print(f"Optimization successful!")
        print(f"Latency reduced by {result.improvement:.2f}ms")
        print(f"Applied optimizations: {result.optimizations_applied}")
    else:
        print(f"Optimization failed")
        print(f"Recommendations: {result.recommendations}")

# Start continuous monitoring
def on_latency_change(latency):
    if latency > 45.0:
        print(f"High latency detected: {latency:.2f}ms")
        optimizer.optimize_latency()

optimizer.monitor_latency(callback=on_latency_change)
```

### 6. `level_monitor.py`

**Purpose**: Real-time audio level monitoring and quality assessment for maintaining optimal audio levels.

#### Classes

##### `AudioLevels`
Audio level measurements.

**Attributes:**
- `peak_level: float` - Peak level in dBFS
- `rms_level: float` - RMS level in dBFS
- `spl_level: float` - Sound pressure level in dB SPL
- `left_channel: float` - Left channel level
- `right_channel: float` - Right channel level
- `dynamic_range: float` - Dynamic range in dB
- `signal_to_noise: float` - Signal-to-noise ratio in dB

##### `LevelAlert`
Audio level alert.

**Attributes:**
- `alert_type: str` - Alert type ("clipping", "low_level", "high_noise")
- `severity: str` - Severity level ("warning", "error", "critical")
- `message: str` - Alert message
- `timestamp: float` - Alert timestamp
- `channel: int` - Affected channel
- `value: float` - Alert value

##### `LevelMonitor`
Main audio level monitoring system.

**Methods:**
- `__init__(config: dict = None)` - Initialize monitor
- `start_monitoring()` - Start level monitoring
- `stop_monitoring()` - Stop level monitoring
- `get_current_levels()` - Get current audio levels
- `get_peak_levels()` - Get peak levels
- `reset_peak_levels()` - Reset peak level meters
- `set_alert_thresholds(thresholds: dict)` - Set alert thresholds
- `get_level_history(duration: float)` - Get level history
- `calibrate_levels()` - Calibrate level measurements
- `export_level_data(file_path: str)` - Export level data

**Usage Example:**
```python
# Create level monitor
config = {
    "update_interval": 0.1,  # 100ms updates
    "enable_spl_measurement": True,
    "enable_clipping_detection": True,
    "history_length": 60.0  # 60 seconds
}

monitor = LevelMonitor(config)

# Set alert thresholds
monitor.set_alert_thresholds({
    "clipping_threshold": -0.1,  # dBFS
    "low_level_threshold": -60.0,  # dBFS
    "high_noise_threshold": -40.0  # dB SNR
})

# Set alert callback
def on_alert(alert):
    print(f"Audio Alert: {alert.alert_type}")
    print(f"Severity: {alert.severity}")
    print(f"Message: {alert.message}")
    print(f"Channel: {alert.channel}")

monitor.set_callback("alert", on_alert)

# Start monitoring
monitor.start_monitoring()

# Monitor levels
while True:
    levels = monitor.get_current_levels()
    print(f"Peak: {levels.peak_level:.1f} dBFS")
    print(f"RMS: {levels.rms_level:.1f} dBFS")
    print(f"SPL: {levels.spl_level:.1f} dB SPL")
    print(f"SNR: {levels.signal_to_noise:.1f} dB")
    
    # Check for clipping
    if levels.peak_level > -0.1:
        print("WARNING: Audio clipping detected!")
    
    time.sleep(1)
```

## Module Integration

The Audio module integrates with other Silent Steno components:

1. **Bluetooth Module**: Receives audio from Bluetooth A2DP connections
2. **Core Events**: Publishes audio events and status updates
3. **Monitoring**: Reports performance metrics and health status
4. **AI Module**: Provides audio data for transcription
5. **Recording Module**: Supplies audio for recording and storage

## Common Usage Patterns

### Complete Audio System Setup
```python
# 1. Create audio system with optimal configuration
audio_system = create_balanced_system({
    "input_device": "bluetooth_a2dp",
    "output_device": "headphones",
    "target_latency": 30.0,
    "enable_monitoring": True
})

# 2. Initialize all components
audio_system.initialize()

# 3. Set up monitoring and callbacks
def on_audio_data(data):
    # Process audio data
    process_audio_for_transcription(data)

def on_metrics(metrics):
    if metrics.latency > 40.0:
        print("Latency threshold exceeded!")
        audio_system.optimize_latency()

audio_system.set_callback("audio_data", on_audio_data)
audio_system.set_callback("metrics", on_metrics)

# 4. Start the system
audio_system.start()

# 5. Monitor performance
while audio_system.is_running:
    status = audio_system.get_status()
    metrics = audio_system.get_performance_metrics()
    
    print(f"Status: {status}")
    print(f"Latency: {metrics.latency:.2f}ms")
    print(f"CPU Usage: {metrics.cpu_usage:.1f}%")
    
    time.sleep(1)
```

### Real-time Audio Processing
```python
# Create pipeline for real-time processing
config = AudioConfig(
    input_device="bluetooth_input",
    output_device="headphones",
    sample_rate=44100,
    channels=2,
    buffer_size=64,  # Small buffer for low latency
    target_latency=20.0
)

pipeline = AudioPipeline(config)

# Process audio chunks
def process_audio_chunk(chunk):
    # Apply real-time processing
    processed_chunk = apply_noise_reduction(chunk)
    processed_chunk = apply_eq(processed_chunk)
    
    # Forward to output
    pipeline.output_chunk(processed_chunk)
    
    # Send to AI for transcription
    send_to_transcription(processed_chunk)

pipeline.set_callback("audio_chunk", process_audio_chunk)
pipeline.start()
```

### Audio Format Conversion Pipeline
```python
# Create format conversion pipeline
converter = FormatConverter()

# Define conversion chain
input_format = AudioFormat(sample_rate=48000, channels=2, bit_depth=24, format="PCM")
intermediate_format = AudioFormat(sample_rate=44100, channels=2, bit_depth=16, format="PCM")
output_format = AudioFormat(sample_rate=44100, channels=2, bit_depth=16, format="SBC")

# Convert audio through chain
def convert_audio_chain(audio_data):
    # Step 1: Downsample and reduce bit depth
    result1 = converter.convert_audio(audio_data, input_format, intermediate_format)
    
    # Step 2: Convert to Bluetooth codec
    result2 = converter.convert_audio(result1.converted_data, intermediate_format, output_format)
    
    return result2

# Process audio stream
for audio_chunk in audio_stream:
    converted = convert_audio_chain(audio_chunk)
    if converted.success:
        send_to_bluetooth(converted.converted_data)
```

### Latency Optimization Workflow
```python
# Create latency optimizer
optimizer = LatencyOptimizer(target_latency=35.0)

# Optimization workflow
def optimize_audio_system():
    # 1. Measure current latency
    measurement = optimizer.measure_latency()
    print(f"Current latency: {measurement.total_latency:.2f}ms")
    
    # 2. Optimize if needed
    if measurement.total_latency > 40.0:
        result = optimizer.optimize_latency()
        
        if result.success:
            print(f"Optimization successful: {result.improvement:.2f}ms improvement")
        else:
            print(f"Optimization failed: {result.recommendations}")
    
    # 3. Start continuous monitoring
    def on_latency_alert(latency):
        if latency > 45.0:
            print(f"High latency alert: {latency:.2f}ms")
            # Trigger re-optimization
            optimizer.optimize_latency()
    
    optimizer.monitor_latency(callback=on_latency_alert)

# Run optimization
optimize_audio_system()
```

### Audio Quality Monitoring
```python
# Create comprehensive monitoring system
level_monitor = LevelMonitor({
    "update_interval": 0.05,  # 50ms updates
    "enable_spl_measurement": True,
    "enable_spectral_analysis": True
})

# Set up quality monitoring
def monitor_audio_quality():
    # Configure alert thresholds
    level_monitor.set_alert_thresholds({
        "clipping_threshold": -0.1,
        "low_level_threshold": -60.0,
        "high_noise_threshold": -40.0,
        "distortion_threshold": 0.1
    })
    
    # Handle quality alerts
    def on_quality_alert(alert):
        if alert.alert_type == "clipping":
            print("Audio clipping detected - reducing input gain")
            adjust_input_gain(-3.0)
        elif alert.alert_type == "low_level":
            print("Low audio level - increasing input gain")
            adjust_input_gain(3.0)
        elif alert.alert_type == "high_noise":
            print("High noise level - enabling noise reduction")
            enable_noise_reduction()
    
    level_monitor.set_callback("alert", on_quality_alert)
    level_monitor.start_monitoring()
    
    # Generate quality reports
    while True:
        levels = level_monitor.get_current_levels()
        print(f"Audio Quality Report:")
        print(f"  Peak Level: {levels.peak_level:.1f} dBFS")
        print(f"  RMS Level: {levels.rms_level:.1f} dBFS")
        print(f"  SNR: {levels.signal_to_noise:.1f} dB")
        print(f"  Dynamic Range: {levels.dynamic_range:.1f} dB")
        
        time.sleep(5)

# Start monitoring
monitor_audio_quality()
```

### Error Handling and Recovery
```python
# Audio system with robust error handling
class RobustAudioSystem:
    def __init__(self):
        self.audio_system = create_balanced_system()
        self.error_count = 0
        self.max_errors = 5
        
    def start_with_recovery(self):
        while self.error_count < self.max_errors:
            try:
                self.audio_system.initialize()
                self.audio_system.start()
                
                # Monitor for errors
                def on_error(error):
                    self.handle_error(error)
                
                self.audio_system.set_callback("error", on_error)
                
                # System running successfully
                break
                
            except Exception as e:
                self.handle_error(e)
                
    def handle_error(self, error):
        self.error_count += 1
        print(f"Audio system error ({self.error_count}/{self.max_errors}): {error}")
        
        # Stop current system
        try:
            self.audio_system.stop()
        except:
            pass
        
        # Wait before retry
        time.sleep(2.0)
        
        # Attempt recovery
        if self.error_count < self.max_errors:
            print("Attempting audio system recovery...")
            self.audio_system = create_balanced_system()
        else:
            print("Max errors reached - audio system failed")
            raise RuntimeError("Audio system failed to recover")

# Use robust system
robust_system = RobustAudioSystem()
robust_system.start_with_recovery()
```

This comprehensive documentation provides complete technical details and practical usage examples for all components in the Audio module, enabling developers to implement high-quality, low-latency audio processing for The Silent Steno system.