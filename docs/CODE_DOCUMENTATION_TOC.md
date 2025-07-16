# The Silent Steno - Code Documentation Table of Contents

This document serves as the master index for all code documentation. Each directory has its own documentation file that details all classes, functions, and components within that directory.

## Documentation Status Legend
- ✅ Documented
- 🔄 In Progress
- ❌ Not Started

## Root Directory Files
- ❌ `demo_live_session.py` - Live session demonstration script
- ❌ `demo_simple.py` - Simple demonstration script  
- ❌ `demo_touch_ui.py` - Touch UI demonstration script
- ❌ `test_integration.py` - Integration test script

## Source Code Directories

### 1. Core Module (`/src/core/`)
**Documentation:** `docs/code_documentation_core.md` ❌
- `__init__.py` - Module initialization
- `application.py` - Main application class and lifecycle
- `config.py` - Configuration management
- `errors.py` - Custom exception classes
- `events.py` - Event system and messaging
- `logging.py` - Logging configuration and utilities
- `monitoring.py` - System monitoring utilities
- `registry.py` - Component registry system
- `registry_backup.py` - Registry backup functionality

### 2. AI Module (`/src/ai/`)
**Documentation:** `docs/code_documentation_ai.md` ❌
- `__init__.py` - Module initialization
- `analysis_pipeline.py` - Main AI analysis pipeline
- `audio_chunker.py` - Audio chunking utilities
- `confidence_scorer.py` - Transcription confidence scoring
- `meeting_analyzer.py` - Meeting analysis functionality
- `participant_analyzer.py` - Participant analysis
- `performance_optimizer.py` - AI performance optimization
- `speaker_diarizer.py` - Speaker diarization system
- `status_tracker.py` - AI processing status tracking
- `transcript_formatter.py` - Transcript formatting utilities
- `transcription_pipeline.py` - Transcription pipeline management
- `whisper_transcriber.py` - Whisper model integration

### 3. Audio Module (`/src/audio/`)
**Documentation:** `docs/code_documentation_audio.md` ❌
- `__init__.py` - Module initialization
- `alsa_manager.py` - ALSA audio system management
- `audio_pipeline.py` - Main audio pipeline
- `format_converter.py` - Audio format conversion
- `latency_optimizer.py` - Audio latency optimization
- `level_monitor.py` - Audio level monitoring

### 4. Analysis Module (`/src/analysis/`)
**Documentation:** `docs/code_documentation_analysis.md` ❌
- `__init__.py` - Module initialization
- `audio_chunker.py` - Audio chunk analysis
- `quality_assessor.py` - Audio quality assessment
- `silence_detector.py` - Silence detection
- `speaker_detector.py` - Speaker detection
- `statistics_collector.py` - Audio statistics collection
- `voice_activity_detector.py` - Voice activity detection (VAD)

### 5. Bluetooth Module (`/src/bluetooth/`)
**Documentation:** `docs/code_documentation_bluetooth.md` ❌
- `__init__.py` - Module initialization
- `bluez_manager.py` - BlueZ Bluetooth stack management
- `connection_manager.py` - Bluetooth connection management

### 6. Data Module (`/src/data/`)
**Documentation:** `docs/code_documentation_data.md` ❌
- `__init__.py` - Module initialization
- `backup_manager.py` - Data backup management
- `database.py` - Database connection and queries
- `migrations.py` - Database migration system
- `models.py` - Data models and schemas
- `retention_manager.py` - Data retention policies

### 7. Export Module (`/src/export/`)
**Documentation:** `docs/code_documentation_export.md` ❌
- `__init__.py` - Module initialization
- `bulk_exporter.py` - Bulk export functionality
- `email_exporter.py` - Email export integration
- `format_customizer.py` - Export format customization
- `network_sharing.py` - Network sharing capabilities
- `pdf_generator.py` - PDF generation
- `usb_exporter.py` - USB export functionality

### 8. LLM Module (`/src/llm/`)
**Documentation:** `docs/code_documentation_llm.md` ❌
- `__init__.py` - Module initialization
- `action_item_extractor.py` - Extract action items from meetings
- `local_llm_processor.py` - Local LLM processing
- `meeting_analyzer.py` - LLM-based meeting analysis
- `output_formatter.py` - LLM output formatting
- `prompt_templates.py` - LLM prompt templates
- `topic_identifier.py` - Topic identification

### 9. Recording Module (`/src/recording/`)
**Documentation:** `docs/code_documentation_recording.md` ❌
- `__init__.py` - Module initialization
- `audio_recorder.py` - Audio recording functionality
- `file_manager.py` - Recording file management
- `metadata_tracker.py` - Recording metadata tracking
- `preprocessor.py` - Audio preprocessing
- `session_manager.py` - Recording session management
- `storage_monitor.py` - Storage space monitoring

### 10. System Module (`/src/system/`)
**Documentation:** `docs/code_documentation_system.md` ❌
- `__init__.py` - Module initialization
- `device_manager.py` - Device management
- `diagnostics.py` - System diagnostics
- `factory_reset.py` - Factory reset functionality
- `health_monitor.py` - System health monitoring
- `remote_manager.py` - Remote management capabilities
- `storage_cleanup.py` - Storage cleanup utilities
- `update_manager.py` - System update management

### 11. UI Module (`/src/ui/`)
**Documentation:** `docs/code_documentation_ui.md` ❌
- `__init__.py` - Module initialization
- `audio_visualizer.py` - Audio visualization components
- `export_dialog.py` - Export dialog interface
- `feedback_manager.py` - User feedback management
- `main_window.py` - Main application window
- `navigation.py` - UI navigation system
- `session_controls.py` - Session control widgets
- `session_details_view.py` - Detailed session view
- `session_list_view.py` - Session list interface
- `session_view.py` - Session viewing interface
- `settings_view.py` - Settings interface
- `status_indicators.py` - Status indicator widgets
- `storage_monitor_widget.py` - Storage monitoring widget
- `themes.py` - UI theme management
- `touch_controls.py` - Touch control implementation
- `transcription_display.py` - Transcription display widget

## Configuration Files

### Configuration Directory (`/config/`)
**Documentation:** `docs/code_documentation_config.md` ❌
- `app_config.json` - Application configuration
- `device_config.json` - Device-specific configuration
- `logging_config.json` - Logging configuration
- `theme_config.json` - UI theme configuration

## Other Directories

### Assets Directory (`/assets/`)
**Documentation:** `docs/code_documentation_assets.md` ❌
- `create_icon.py` - Icon generation script

### Migrations Directory (`/migrations/`)
**Documentation:** `docs/code_documentation_migrations.md` ❌
- `env.py` - Migration environment configuration

## Documentation Completion Tracking

| Module | Files | Status | Progress |
|--------|-------|--------|----------|
| Core | 9 | ❌ | 0/9 |
| AI | 12 | ❌ | 0/12 |
| Audio | 6 | ❌ | 0/6 |
| Analysis | 7 | ❌ | 0/7 |
| Bluetooth | 3 | ❌ | 0/3 |
| Data | 6 | ❌ | 0/6 |
| Export | 7 | ❌ | 0/7 |
| LLM | 7 | ❌ | 0/7 |
| Recording | 7 | ❌ | 0/7 |
| System | 8 | ❌ | 0/8 |
| UI | 16 | ❌ | 0/16 |
| Config | 4 | ❌ | 0/4 |
| Root | 4 | ❌ | 0/4 |
| Other | 2 | ❌ | 0/2 |
| **Total** | **98** | ❌ | **0/98** |

## Documentation Template

Each module documentation file follows this structure:

1. **Module Overview** - High-level description of the module's purpose
2. **Dependencies** - External and internal dependencies
3. **File Documentation** - For each file:
   - File purpose and overview
   - Classes (with methods and attributes)
   - Functions (with parameters and return values)
   - Constants and global variables
   - Usage examples
4. **Module Integration** - How the module interacts with other parts of the system
5. **Common Usage Patterns** - Typical use cases and code examples