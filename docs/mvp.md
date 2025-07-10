# Bluetooth AI Meeting Recorder - MVP Feature Specification

## Project Overview
A Raspberry Pi 5-based device that acts as an invisible Bluetooth audio intermediary, capturing meeting/call audio for AI-powered transcription and analysis while maintaining transparent audio forwarding to user's headphones.

## Hardware Platform
- **Primary:** Raspberry Pi 5 (4GB+ RAM recommended)
- **Display:** 3.5" or 5" touchscreen (480x320 or 800x480)
- **Audio:** USB audio interface for enhanced audio quality (optional)
- **Connectivity:** Built-in WiFi and Bluetooth 5.0
- **Power:** Wall adapter (no battery requirement)
- **Enclosure:** 3D printable case with screen cutout

## Core MVP Features

### 1. Bluetooth Audio Proxy
**Functionality:**
- Device appears as high-quality Bluetooth headphones to phone
- Simultaneously connects to user's actual headphones as audio output
- Zero-latency audio forwarding (< 40ms total latency)
- Support for major Bluetooth codecs: SBC, AAC, aptX, Samsung Scalable

**Technical Requirements:**
- BlueZ stack configuration for dual audio connections
- Real-time audio capture during forwarding
- Automatic audio format conversion as needed
- Connection persistence and auto-reconnection

### 2. AI Transcription Engine
**Functionality:**
- Real-time speech-to-text using local Whisper model
- Speaker diarization (identify different speakers)
- Confidence scoring for transcription accuracy
- Support for multiple languages (English priority for MVP)

**Technical Requirements:**
- Whisper Base model (good speed/accuracy tradeoff on Pi 5)
- Audio preprocessing (noise reduction, normalization)
- Chunked processing for real-time performance
- Text output with timestamps

### 3. Local AI Analysis
**Functionality:**
- Meeting summarization using local LLM
- Action item extraction
- Key topic identification
- Participant analysis (speaking time, engagement metrics)

**Technical Requirements:**
- Lightweight local LLM (Phi-3 Mini or similar)
- Structured output formatting (JSON/Markdown)
- Processing pipeline triggered at session end
- Configurable analysis depth

### 4. Simple Touch UI
**Functionality:**
- **Home Screen:** Current status, active sessions indicator
- **Live Session View:** Real-time transcription scroll, audio levels
- **Session Management:** Start/stop recording, session list
- **Settings:** Device pairing, AI model selection, storage management
- **Review Interface:** Browse past sessions, export options

**UI Requirements:**
- Touch-optimized interface (finger-friendly buttons)
- Dark mode for low-light environments
- Visual feedback for all user actions
- Minimal text input (use voice commands where possible)

### 5. Data Management
**Functionality:**
- Local storage of all audio and transcripts
- Export options (email, cloud sync, USB transfer)
- Automatic cleanup of old sessions (configurable retention)
- Search functionality across all sessions

**Technical Requirements:**
- SQLite database for metadata and transcripts
- Compressed audio storage (FLAC or high-quality MP3)
- RESTful API for data access
- Backup/restore functionality

## MVP Scope Limitations
**Explicitly NOT included in MVP:**
- Multi-device support (single phone pairing only)
- Cloud AI services integration
- Advanced analytics dashboard
- Mobile app companion
- Video call integration
- Enterprise features (user management, compliance)

## Performance Targets
- **Audio Latency:** < 40ms end-to-end
- **Transcription Lag:** < 3 seconds behind live audio
- **Session Start Time:** < 10 seconds from tap to active
- **Battery Life:** N/A (wall powered)
- **Storage:** 32GB+ SD card (20+ hours of meetings)
- **AI Processing:** Summary generation < 60 seconds for 1-hour meeting

## Success Metrics
- **Audio Quality:** No perceptible degradation vs direct connection
- **Transcription Accuracy:** > 90% for clear speech
- **User Experience:** Single tap to start session
- **Reliability:** > 99% session completion rate
- **Performance:** Real-time operation during 2+ hour meetings

## Development Phases
1. **Phase 1:** Bluetooth audio proxy + basic UI
2. **Phase 2:** Local transcription integration
3. **Phase 3:** AI analysis and export features
4. **Phase 4:** Polish and optimization

## Risk Mitigation
- **Bluetooth Reliability:** Implement robust connection management
- **Audio Latency:** Use low-latency audio pipeline, hardware audio interface if needed
- **Processing Power:** Profile early, optimize models for Pi 5 performance
- **User Experience:** Extensive testing with actual meeting scenarios