# The Silent Steno - Walking Skeleton Demo

This is a minimal proof-of-concept implementation demonstrating the core audio recording and transcription pipeline.

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run the demo:
```bash
python minimal_demo.py
```

## Features

- Touch-optimized UI for Raspberry Pi 5 (1024x600 screen)
- Audio recording from USB audio device
- WAV file storage
- Audio playback
- Session persistence (JSON)
- Whisper transcription (pending integration)

## Hardware Setup

- Raspberry Pi 5 with touchscreen
- USB Audio Device for input/output
- Headphones or speakers connected to USB device

## Files Created

- `demo_sessions/` - Directory containing WAV recordings
- `demo_sessions/sessions.json` - Session history