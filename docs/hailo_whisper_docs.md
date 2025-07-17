# Hailo Whisper Pipeline Documentation

## Overview
This repository implements an Automatic Speech Recognition (ASR) system using OpenAI's Whisper-tiny model optimized for Hailo-8/8L AI accelerators. The system provides both command-line and web API interfaces for real-time speech-to-text transcription.

## Architecture

### Core Components
- **HailoWhisperPipeline**: Main inference engine using Hailo accelerators
- **Audio Processing**: Complete pipeline for audio capture, preprocessing, and enhancement
- **Web Service**: FastAPI-based REST API for transcription services
- **CLI Interface**: Command-line application for interactive transcription

---

## Project Structure

```
├── app/                          # Main application code
│   ├── app_hailo_whisper.py     # CLI entry point
│   ├── hailo_whisper_pipeline.py # Core pipeline class
│   ├── download_resources.sh    # Resource download script
│   └── hefs/                    # Model files (downloaded)
├── common/                      # Shared utilities
│   ├── audio_utils.py          # Audio processing utilities
│   ├── preprocessing.py        # Audio preprocessing
│   ├── postprocessing.py       # Transcription postprocessing
│   └── record_utils.py         # Audio recording utilities
├── web/                        # Web service
│   ├── main.py                 # FastAPI application
│   └── serve.py                # Server launcher
├── setup.py                    # Environment setup
└── requirements_inference.txt  # Dependencies
```

---

## Main Classes and Functions

### 1. HailoWhisperPipeline (`app/hailo_whisper_pipeline.py`)

The core inference engine that manages the Whisper model on Hailo hardware.

#### Class Definition
```python
class HailoWhisperPipeline:
    def __init__(self, encoder_model_path, decoder_model_path, variant="tiny", 
                 host="arm64", multi_process_service=False)
```

#### Key Methods
- **`send_data(data)`**: Queue audio data for processing
- **`get_transcription()`**: Retrieve transcription results
- **`stop()`**: Shutdown the pipeline
- **`_inference_loop()`**: Main processing thread (encoder → decoder → tokenization)
- **`_tokenization(decoder_input_ids)`**: Token embedding and preprocessing

#### Key Attributes
- `encoder_model_path`: Path to encoder HEF file
- `decoder_model_path`: Path to decoder HEF file
- `variant`: Model variant ("tiny" supported)
- `decoding_sequence_length`: Maximum sequence length (32 for tiny)
- `token_embedding_weight`: Pre-loaded token embeddings
- `tokenizer`: HuggingFace tokenizer for text conversion

### 2. CLI Application (`app/app_hailo_whisper.py`)

Command-line interface for interactive transcription.

#### Main Functions
- **`get_args()`**: Parse command-line arguments
- **`get_encoder_hef_path(hw_arch)`**: Get encoder model path based on hardware
- **`get_decoder_hef_path(hw_arch)`**: Get decoder model path based on hardware
- **`main()`**: Main execution loop

#### Command-Line Arguments
- `--reuse-audio`: Reuse previously recorded audio file
- `--hw-arch`: Hardware architecture ("hailo8" or "hailo8l")
- `--multi-process-service`: Enable multi-process service mode

#### Key Variables
- `DURATION = 10`: Recording duration in seconds
- `variant = "tiny"`: Model variant
- `chunk_length = 10`: Audio chunk length for tiny model

---

## Audio Processing Pipeline

### 1. Audio Utilities (`common/audio_utils.py`)

Core audio processing functions adapted from Whisper.

#### Constants
```python
SAMPLE_RATE = 16000        # Audio sample rate
N_FFT = 400               # FFT window size
HOP_LENGTH = 160          # Hop length for STFT
CHUNK_LENGTH = 30         # Default chunk length
N_SAMPLES = 480000        # Samples in 30-second chunk
```

#### Key Functions
- **`load_audio(file, sr=16000)`**: Load audio file using ffmpeg
- **`pad_or_trim(array, length)`**: Pad or trim audio to specified length
- **`log_mel_spectrogram(audio, n_mels=80)`**: Generate mel spectrogram
- **`mel_filters(device, n_mels)`**: Load mel filterbank matrix

### 2. Preprocessing (`common/preprocessing.py`)

Audio preprocessing and enhancement functions.

#### Key Functions
- **`preprocess(audio, is_nhwc=False, chunk_length=10, chunk_offset=0)`**
  - Generates mel spectrograms from audio
  - Supports chunking for long audio
  - Returns list of mel spectrograms

- **`improve_input_audio(audio, vad=True, low_audio_gain=True)`**
  - Apply gain compensation for low-level audio
  - Voice Activity Detection (VAD)
  - Returns enhanced audio and speech start time

- **`detect_first_speech(audio_data, sample_rate, threshold=0.2)`**
  - Detect first speech occurrence
  - Energy-based detection
  - Returns start time in seconds

- **`apply_gain(audio, gain_db)`**: Apply gain in decibels

### 3. Postprocessing (`common/postprocessing.py`)

Transcription quality enhancement functions.

#### Key Functions
- **`apply_repetition_penalty(logits, generated_tokens, penalty=1.5)`**
  - Reduce repetitive token generation
  - Configurable penalty strength
  - Excludes punctuation tokens

- **`temperature_sampling(logits, temperature=0.0)`**
  - Apply temperature sampling to logits
  - Greedy decoding when temperature=0
  - Boosts punctuation tokens

- **`clean_transcription(transcription)`**
  - Remove duplicate sentences
  - Normalize punctuation
  - Handle repetitive outputs

#### Key Variables
- `excluded_tokens = [11, 13]`: Punctuation tokens for repetition penalty
- `last_window = 8`: Window size for repetition analysis

### 4. Recording Utilities (`common/record_utils.py`)

Audio recording and capture functions.

#### Key Functions
- **`record_audio(duration, audio_path)`**
  - Record from microphone
  - Support early stopping with Enter key
  - Save as WAV file
  - Returns audio data as numpy array

- **`enter_pressed()`**: Non-blocking keyboard input detection

#### Constants
```python
SAMPLE_RATE = 16000       # Recording sample rate
CHANNELS = 1              # Mono recording
```

---

## Web Service (`web/`)

### FastAPI Application (`web/main.py`)

REST API service for transcription.

#### Endpoints
- **`POST /transcribe/`**: Accept WAV file, return transcription
  - Input: WAV file upload
  - Output: `{"transcription": "..."}`
  - Error handling for non-WAV files

#### Key Functions
- **`get_encoder_hef_path(hw_arch)`**: Get encoder model path
- **`get_decoder_hef_path(hw_arch)`**: Get decoder model path
- **`transcribe_audio(file)`**: Main transcription endpoint

#### Global Variables
- `whisper_hailo`: Initialized pipeline instance
- `variant = "tiny"`: Model variant
- `is_nhwc = True`: Input format flag
- `chunk_length = 10`: Processing chunk length

### Server Launcher (`web/serve.py`)

Uvicorn server configuration for the FastAPI app.

---

## Setup and Installation

### Setup Script (`setup.py`)

Automated environment setup and dependency installation.

#### Key Functions
- **`create_venv()`**: Create virtual environment with system packages for RPi5
- **`install_requirements()`**: Install Python dependencies
- **`download_resources()`**: Download model files
- **`is_raspberry_pi_5()`**: Detect Raspberry Pi 5 platform

#### Key Variables
- `VENV_DIR`: Virtual environment directory path
- `PYTHON_BIN`: Python executable path in venv
- `PIP_BIN`: Pip executable path in venv

---

## Hardware Support

### Supported Platforms
- **x86**: Standard PC/server platforms
- **Raspberry Pi 5**: ARM64 with Hailo accelerator

### Hailo Hardware Architectures
- **Hailo-8**: Primary target platform
- **Hailo-8L**: Low-power variant

### Model Files (HEF)
- **Encoder**: `tiny-whisper-encoder-10s_15dB[_h8l].hef`
- **Decoder**: `tiny-whisper-decoder-fixed-sequence-matmul-split[_h8l].hef`
- **Assets**: Token embeddings and ONNX preprocessing data

---

## Dependencies

### Core Dependencies
- `transformers==4.50.1`: HuggingFace tokenizer
- `torch==2.6.0`: PyTorch for preprocessing
- `sounddevice==0.5.1`: Audio recording
- `scipy==1.9.3`: Audio processing

### Web Service Dependencies
- `fastapi==0.110.0`: Web framework
- `uvicorn==0.29.0`: ASGI server
- `python-multipart==0.0.6`: File upload support

### System Dependencies
- **HailoRT 4.20/4.21**: Hailo runtime and PCIe driver
- **ffmpeg**: Audio file processing
- **libportaudio2**: Audio device access

---

## Usage Examples

### CLI Usage
```bash
# Basic usage
python3 -m app.app_hailo_whisper

# Hailo-8L hardware
python3 -m app.app_hailo_whisper --hw-arch hailo8l

# Reuse previous audio
python3 -m app.app_hailo_whisper --reuse-audio
```

### Web Service Usage
```bash
# Start server
python3 web/serve.py --hw-arch hailo8l

# API call
curl -X POST "http://localhost:8000/transcribe/" \
     -F "file=@audio.wav"
```

### Pipeline Integration
```python
from app.hailo_whisper_pipeline import HailoWhisperPipeline

# Initialize pipeline
pipeline = HailoWhisperPipeline(encoder_path, decoder_path, "tiny")

# Process audio
pipeline.send_data(mel_spectrogram)
transcription = pipeline.get_transcription()
```

---

## Key Features

### Performance Optimizations
- **Multi-threaded Processing**: Separate threads for inference and I/O
- **Queue-based Communication**: Asynchronous data flow
- **Repetition Penalty**: Reduces model hallucinations
- **Voice Activity Detection**: Automatic speech detection

### Audio Enhancement
- **Automatic Gain Control**: Compensates for low audio levels
- **Noise Reduction**: Basic audio improvement
- **Chunked Processing**: Handles long audio files
- **Real-time Processing**: Streaming-capable architecture

### Hardware Efficiency
- **Hailo Acceleration**: Optimized for Hailo AI processors
- **Multi-process Service**: Shared hardware resources
- **Memory Management**: Efficient tensor operations
- **Format Optimization**: NHWC tensor layout support

---

## Limitations and Notes

### Current Limitations
- **English Only**: Single language support
- **Tiny Model**: Limited accuracy compared to larger models
- **10-second Chunks**: Fixed processing window
- **No Model Conversion**: Pre-compiled models only

### Future Improvements
- Model conversion scripts
- Optimized post-processing
- Additional model variants
- C++ implementation
- Multi-language support

### Hardware Requirements
- Minimum: Hailo-8/8L accelerator
- Recommended: Good quality microphone
- Memory: Sufficient RAM for model loading
- Storage: Space for model files (~100MB)
