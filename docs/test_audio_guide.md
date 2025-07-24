# Test Audio Generation & Benchmarking Guide

## Overview

This guide explains how to generate synthetic test audio and run automated benchmarks for The Silent Steno. This replaces the need to manually find YouTube videos for testing and provides consistent, repeatable test cases.

## Quick Start

```bash
# 1. Setup test environment
./setup_test_audio.sh

# 2. Generate test audio files
python generate_test_audio.py

# 3. Run benchmarks
python test_audio_benchmark.py
```

## Test Audio Files Generated

The `generate_test_audio.py` script creates the following test files:

### 1. **test_meeting_5min.wav**
- **Description**: 5-person team meeting with multiple speakers
- **Speakers**: Sarah (lead), John, Mike, Emily, Raj
- **Content**: Sprint planning, API discussion, action items
- **Background**: Office noise
- **Use Case**: Testing speaker diarization and meeting analysis

### 2. **test_work_call_3min.wav**
- **Description**: Two-person work phone call
- **Speakers**: John, Sarah
- **Content**: Client proposal discussion, timeline planning
- **Background**: Quiet
- **Use Case**: Testing business context detection

### 3. **test_personal_call_2min.wav**
- **Description**: Casual personal phone call
- **Speakers**: Emily, Mike
- **Content**: Vacation stories, personal updates
- **Background**: Home environment
- **Use Case**: Testing informal speech patterns

### 4. **test_noisy_coffee_shop.wav**
- **Description**: Meeting with significant background noise
- **Content**: Shortened meeting segment
- **Background**: Coffee shop (loud)
- **Use Case**: Testing noise robustness

### 5. **test_fast_speech.wav**
- **Description**: Rapid conversation without pauses
- **Content**: Quick back-and-forth planning
- **Background**: Quiet
- **Use Case**: Stress testing real-time transcription

## Benchmark Metrics

The benchmark suite measures:

1. **Performance Metrics**:
   - Transcription time
   - Real-time factor (RTF)
   - Words per second
   - Memory usage (if available)

2. **Accuracy Metrics**:
   - Feature detection rate
   - Action item extraction
   - Speaker identification
   - Summary quality

3. **Robustness Metrics**:
   - Noise handling
   - Multiple speaker handling
   - Fast speech handling

## Interpreting Results

### Real-Time Factor (RTF)
- **< 0.5x**: Excellent (2x+ faster than real-time)
- **0.5-1.0x**: Good (faster than real-time)
- **1.0-1.5x**: Acceptable (slightly slower)
- **> 1.5x**: Needs optimization

### Feature Detection
- Measures how well the system identifies expected content
- Higher percentage = better contextual understanding

## Advanced Usage

### Generate Custom Test Audio

```python
from generate_test_audio import TestAudioGenerator

# Create generator
generator = TestAudioGenerator()

# Create custom script
custom_script = [
    {"speaker": "sarah", "text": "Let's discuss the quarterly results."},
    {"speaker": "john", "text": "Sales are up 15 percent from last quarter."},
    # Add more lines...
]

# Generate audio
generator.generate_audio_file(
    custom_script, 
    "custom_test.wav",
    noise_type="office"
)
```

### Benchmark Specific Scenarios

```bash
# Benchmark only noisy scenarios
python test_audio_benchmark.py --single-file test_audio/test_noisy_coffee_shop.wav

# Benchmark with custom test directory
python test_audio_benchmark.py --test-dir my_custom_tests/
```

### Compare Different Configurations

```python
# Run benchmark with different Whisper models
# Modify settings before running benchmark
settings_manager = get_settings_manager()
settings_manager.update_settings('ai', {'whisper_model': 'base'})
# Run benchmark...

settings_manager.update_settings('ai', {'whisper_model': 'small'})
# Run benchmark again...
```

## Integration with CI/CD

Add to your CI pipeline:

```yaml
# .github/workflows/benchmark.yml
- name: Generate Test Audio
  run: python generate_test_audio.py
  
- name: Run Benchmarks
  run: python test_audio_benchmark.py
  
- name: Check Performance
  run: |
    # Parse benchmark results
    # Fail if RTF > 1.5
```

## Troubleshooting

### Common Issues

1. **"No module named 'gtts'"**
   - Run: `pip install gtts pydub nltk`

2. **"Test manifest not found"**
   - Run: `python generate_test_audio.py` first

3. **Poor transcription results**
   - Check if Whisper model is loaded correctly
   - Verify audio files are 16kHz WAV format
   - Check system resources (RAM, CPU)

### Performance Optimization Tips

1. **Use smaller Whisper model** for faster transcription
2. **Enable GPU acceleration** if available (Hailo 8)
3. **Adjust audio chunk size** for streaming
4. **Pre-process audio** (noise reduction, normalization)

## Next Steps

1. **Establish baseline metrics** with current setup
2. **Test with Hailo 8** when available
3. **Compare different Whisper models** (tiny, base, small)
4. **Test live transcription** with chunked audio
5. **Measure battery/power consumption** on Pi 5

## Contributing Test Cases

To add new test scenarios:

1. Add new script generation method in `generate_test_audio.py`
2. Update manifest with test description
3. Add expected features for benchmark validation
4. Submit PR with benchmark results

This testing framework provides consistent, repeatable benchmarks that will help optimize The Silent Steno for production use.