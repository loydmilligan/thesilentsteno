# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**The Silent Steno** is a Bluetooth AI Meeting Recorder built on Raspberry Pi 5 that acts as an invisible audio intermediary between a phone and headphones. It captures meeting audio for AI-powered transcription and analysis while maintaining transparent, low-latency audio forwarding.

**Tech Stack:**
- **Platform:** Raspberry Pi 5 with 3.5-5" touchscreen  
- **Audio:** BlueZ Bluetooth stack with A2DP, ALSA/PulseAudio
- **AI Models:** Local Whisper (Base model) for transcription, Phi-3 Mini for analysis
- **UI Framework:** Touch-optimized interface (Kivy/PyQt/web-based)
- **Storage:** SQLite database, local file storage (FLAC/WAV)
- **Languages:** Python (primary), with system-level audio configuration

## Development Approach

This project uses **manifest-driven development** with task-by-task implementation. The development workflow integrates with git for proper version control and progress tracking.

### Core Development Workflow

The complete task implementation cycle follows these steps:

1. **process_task** - Prepare task with expected post-implementation manifest
2. **implement_task** - Implement with full context from prepared task
3. **check_task** - Validate implementation against expected manifest
4. **resolve_mismatch** - Handle discrepancies between expected and actual (if needed)
5. **commit_task** - Save progress with proper git history and detailed commit messages

### Command Usage Examples

```bash
# Start implementing a new task
claude-code process_task "Task-1.1"

# Implement the prepared task
claude-code implement_task "tasks/prepared/Task-1.1.json"

# Validate implementation
claude-code check_task "Task-1.1"

# Commit completed task
claude-code commit_task "Task-1.1"

# Update architecture understanding after milestones
claude-code update_final_manifest
```

## Available Commands

### Core Development Commands (in `.claude/commands/implementing/`)
- **process_task.md** - Prepare tasks with expected post-task manifests
- **implement_task.md** - Implement prepared tasks with full context
- **check_task.md** - Validate implementation against expected manifest
- **resolve_mismatch.md** - Handle discrepancies between expected and actual
- **commit_task.md** - Commit completed tasks with proper git history

### Workflow Management Commands
- **generate_manifest.md** - Analyze codebase and create/update manifests
- **update_final_manifest.md** - Update proposed final manifest based on learnings

### When to Use Each Command
- **process_task** - Before implementing any task (creates full context)
- **implement_task** - Only after processing task (uses prepared context)
- **check_task** - After every implementation (validates against expected)
- **resolve_mismatch** - When check_task finds discrepancies
- **commit_task** - After successful task completion (saves progress)
- **update_final_manifest** - After milestones or architectural discoveries

## Project Architecture

### Core Components
1. **Bluetooth Audio Proxy** - Dual A2DP connections for transparent audio forwarding
2. **Real-time Audio Pipeline** - Low-latency capture and processing (<40ms)
3. **AI Processing Chain** - Local Whisper transcription + LLM analysis
4. **Touch UI System** - Session management and live monitoring interface
5. **Data Management** - Local storage, export, and session organization

### Audio Pipeline Flow
```
Phone → Bluetooth A2DP → Pi 5 Audio Capture → Audio Forwarding → Headphones
                              ↓
                         Real-time Processing
                              ↓
                      Whisper Transcription
                              ↓
                        LLM Analysis
```

### Key Performance Targets
- **Audio Latency:** <40ms end-to-end
- **Transcription Lag:** <3 seconds behind live audio
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

## Common Development Commands

### Audio System Commands
```bash
# BlueZ configuration
sudo systemctl status bluetooth
bluetoothctl

# Audio pipeline testing
aplay -l    # List audio devices
arecord -l  # List recording devices
pulseaudio --check -v

# Audio latency measurement
sudo apt install jack-tools
jack_delay
```

### AI Model Commands
```bash
# Whisper model management
pip install whisper
whisper --model base audio_file.wav

# Local LLM testing
pip install transformers torch
python -c "from transformers import pipeline; print('Models loaded')"
```

### System Performance Commands
```bash
# CPU/Memory monitoring
htop
free -h
df -h

# Audio performance profiling
cat /proc/asound/cards
cat /proc/asound/devices
```

## Testing Strategy

### Audio Testing
- Test with multiple phone models (iOS/Android)
- Validate with different Bluetooth codecs
- Measure audio latency with various headphones
- Test connection stability during long sessions

### AI Testing
- Validate transcription accuracy with test audio
- Test speaker diarization with multi-person conversations
- Verify LLM analysis quality with meeting samples
- Performance testing on Pi 5 hardware

### Integration Testing
- End-to-end workflow testing
- UI responsiveness validation
- Storage and export functionality
- Error handling and recovery

## Project Structure

```
├── docs/                     # Project documentation
│   ├── mvp.md               # MVP requirements
│   ├── prd.md               # PRD specifications
│   └── manifest_evolution.md # Architecture evolution log
├── tasks/                    # Task management
│   ├── task_list.md         # Implementation tasks
│   ├── prepared/            # Processed task files (gitignored)
│   └── completed/           # Completed task records
├── .claude/commands/        # AI development commands
│   ├── implementing/        # Core workflow commands
│   ├── planning/           # Project planning commands
│   └── tools/              # Utility commands
├── codebase_manifest.json   # Current project state
└── src/                     # Source code (created during implementation)
```

## MCP Servers and Tools

*Note: This section will be expanded as specific MCP tools are integrated for enhanced development capabilities.*

## Future Enhancements

- **Global Commands:** Move commands to ~/.claude/commands for reuse across projects
- **Task Management:** Integration with task management MCP tools
- **Artifact Generation:** Integration with development artifact generation
- **Extended MCP Integration:** Expanded tool usage guidelines and workflows

## Git Workflow Integration

The manifest-driven approach integrates with git to maintain clear development history:
- Each task completion creates a meaningful commit
- Commit messages include task context and architectural changes
- Git history tracks both code changes and design decisions
- Branch strategy supports parallel development of independent components

## Development Best Practices

### Audio Development
- Always test with real hardware (Pi 5 + touchscreen)
- Profile audio latency regularly during development
- Use proper audio buffering to prevent dropouts
- Test with multiple Bluetooth devices and codecs

### AI Integration
- Validate model performance on target hardware before integration
- Implement proper error handling for AI model failures
- Monitor memory usage during long transcription sessions
- Use chunked processing for real-time performance

### UI Development
- Design for touch-first interaction
- Test in various lighting conditions
- Ensure accessibility for users with different abilities
- Optimize for the specific screen size and resolution

This manifest-driven approach ensures systematic development while maintaining architectural integrity throughout the project lifecycle.