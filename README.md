# The Silent Steno

A Raspberry Pi 5-based Bluetooth AI Meeting Recorder that acts as an invisible audio intermediary between your phone and headphones, capturing meeting audio for AI-powered transcription and analysis while maintaining transparent, low-latency audio forwarding.

## Project Overview

- **Tech Stack:** Python, Raspberry Pi 5, BlueZ Bluetooth, Whisper AI, Local LLM (Phi-3 Mini), SQLite, Touch UI
- **Type:** Hardware IoT Device with AI Processing
- **Development Status:** Initial bootstrap complete, ready for Phase 1 implementation

## Key Features

### Invisible Audio Capture
- Device appears as high-quality Bluetooth headphones to your phone
- Simultaneously connects to your actual headphones for audio output
- Zero-latency audio forwarding (<40ms total latency)
- Works with any phone app (calls, Zoom, Teams, etc.)

### AI-Powered Analysis
- Real-time speech-to-text using local Whisper model
- Speaker diarization (identify different speakers)
- Meeting summarization and action item extraction
- Local processing - no cloud dependency

### Simple Touch Interface
- Live transcription display during meetings
- One-touch recording start/stop
- Session management and review
- Export options (email, USB, network sharing)

## Development Approach

This project uses manifest-driven development with task-by-task implementation. The `codebase_manifest.json` file contains complete project information including:

- Project metadata and tech stack
- Documentation references
- Architecture overview from planning documents
- Development workflow tracking

See `CLAUDE.md` for detailed AI workflow instructions.

## Quick Start

1. Review `docs/mvp.md` for project requirements
2. Check `docs/prd.md` for detailed specifications
3. Review `tasks/task_list.md` for implementation plan
4. Check `codebase_manifest.json` for current project state
5. Follow development workflow in `CLAUDE.md`

## Directory Structure

- `docs/` - Project documentation (MVP, PRD, architecture evolution)
- `tasks/` - Development task list and task processing
- `.claude/commands/` - AI command prompts for development workflow
- `codebase_manifest.json` - Complete project manifest with metadata
- `CLAUDE.md` - AI workflow documentation
- `src/` - Source code (created during implementation)

## Development Workflow

1. `claude-code process_task "Task-X.X"` - Prepare task with expected manifest
2. `claude-code implement_task "tasks/prepared/Task-X.X.json"` - Implement changes
3. `claude-code check_task "Task-X.X"` - Verify implementation matches expected
4. If mismatch: `claude-code resolve_mismatch "Task-X.X"` - Handle discrepancies
5. `claude-code commit_task "Task-X.X"` - Save progress with proper git history

## Hardware Requirements

- **Platform:** Raspberry Pi 5 (4GB+ RAM recommended)
- **Display:** 3.5" or 5" touchscreen (480x320 or 800x480)
- **Audio:** Built-in audio + optional USB audio interface for enhanced quality
- **Connectivity:** Built-in WiFi and Bluetooth 5.0
- **Power:** Wall adapter (no battery requirement)
- **Storage:** 32GB+ SD card (supports 20+ hours of meetings)
- **Enclosure:** 3D printable case with screen cutout

## Performance Targets

- **Audio Latency:** <40ms end-to-end
- **Transcription Delay:** <3 seconds behind live audio
- **Session Start Time:** <10 seconds from tap to active
- **Transcription Accuracy:** >90% for clear speech
- **Session Reliability:** >99% completion rate

## Development Phases

**Phase 1:** Foundation & Hardware Setup (Tasks 1-3)
- Hardware platform configuration
- Bluetooth stack setup
- Audio pipeline architecture

**Phase 2:** Core Audio Processing (Tasks 4-5)
- Audio recording system
- Real-time audio analysis

**Phase 3:** AI Integration (Tasks 6-8)
- Local Whisper integration
- Local LLM setup
- AI processing pipeline

**Phase 4:** User Interface (Tasks 9-11)
- Touch UI framework
- Live session interface
- Session management UI

**Phase 5:** Data Management (Tasks 12-13)
- Database and storage system
- Export and sharing system

**Phase 6:** System Integration (Tasks 14-15)
- Application integration layer
- Device management features

**Phase 7:** Testing and Optimization (Tasks 16-18)
- Performance optimization
- Integration testing
- User experience polish

**Phase 8:** Deployment and Documentation (Tasks 19-20)
- Deployment automation
- Documentation and support

## Project Status

The project is bootstrapped and ready for implementation. The manifest contains project information extracted from planning documents and will be updated as development progresses.

**Current Status:** Ready to begin Phase 1 - Foundation & Hardware Setup

**Next Steps:**
1. Start with Task 1: Hardware Platform Setup
2. Use the complete development workflow for each task
3. Follow the manifest-driven approach for consistent architecture

See `CLAUDE.md` for detailed instructions and `codebase_manifest.json` for current project state.