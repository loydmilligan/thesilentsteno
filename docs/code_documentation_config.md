# Configuration Directory Documentation

## Overview

The Config directory contains JSON configuration files that define the behavior, settings, and preferences for The Silent Steno application. These files provide a centralized, modular approach to configuration management, allowing for easy customization of application behavior without code modifications.

## Configuration Files

### 1. `app_config.json`

**Purpose**: Main application configuration containing all primary system settings and component configurations.

#### Structure Overview

The configuration is organized into logical sections:

##### Application Section
```json
{
  "application": {
    "name": "SilentSteno",
    "version": "0.1.0",
    "environment": "development",
    "debug_mode": true,
    "log_level": "INFO",
    "startup_timeout": 30,
    "shutdown_timeout": 15,
    "health_check_interval": 30,
    "performance_monitoring": true,
    "thread_pool_size": 4,
    "max_memory_usage": 512,
    "enable_crash_reporting": true,
    "crash_report_url": "https://api.example.com/crash-reports"
  }
}
```

**Key Settings:**
- `name`: Application name identifier
- `version`: Current application version
- `environment`: Runtime environment (development/production)
- `debug_mode`: Enable/disable debug features
- `log_level`: Global logging level
- `startup_timeout`: Maximum startup time in seconds
- `shutdown_timeout`: Maximum shutdown time in seconds
- `health_check_interval`: Health check frequency in seconds
- `performance_monitoring`: Enable performance tracking
- `thread_pool_size`: Number of worker threads
- `max_memory_usage`: Memory limit in MB
- `enable_crash_reporting`: Enable crash reporting
- `crash_report_url`: Crash reporting endpoint

##### Audio Pipeline Section
```json
{
  "audio_pipeline": {
    "input_device": "default",
    "output_device": "default",
    "sample_rate": 44100,
    "channels": 2,
    "bit_depth": 16,
    "buffer_size": 128,
    "latency_target": 40,
    "enable_echo_cancellation": true,
    "enable_noise_suppression": true,
    "enable_automatic_gain": false,
    "monitoring": {
      "enable_level_monitoring": true,
      "enable_clipping_detection": true,
      "enable_quality_assessment": true
    }
  }
}
```

**Key Settings:**
- `input_device`/`output_device`: Audio device identifiers
- `sample_rate`: Audio sampling rate in Hz
- `channels`: Number of audio channels
- `bit_depth`: Audio bit depth
- `buffer_size`: Audio buffer size in frames
- `latency_target`: Target latency in milliseconds
- Audio processing flags for echo cancellation, noise suppression, and automatic gain
- Monitoring settings for level, clipping, and quality assessment

##### Bluetooth Section
```json
{
  "bluetooth": {
    "device_name": "SilentSteno",
    "discoverable": true,
    "pairable": true,
    "auto_connect": true,
    "connection_timeout": 10,
    "max_connections": 2,
    "supported_profiles": ["A2DP", "AVRCP", "HFP"],
    "codec_preferences": ["aptX", "AAC", "SBC"],
    "enable_multipoint": false,
    "power_management": {
      "enable_power_saving": true,
      "idle_timeout": 300,
      "sleep_timeout": 1800
    }
  }
}
```

**Key Settings:**
- `device_name`: Bluetooth device name
- `discoverable`/`pairable`: Device visibility settings
- `auto_connect`: Automatic connection to known devices
- `connection_timeout`: Connection timeout in seconds
- `max_connections`: Maximum simultaneous connections
- `supported_profiles`: Bluetooth profiles (A2DP, AVRCP, HFP)
- `codec_preferences`: Audio codec priority order
- `enable_multipoint`: Multi-device connection support
- Power management settings for energy efficiency

##### AI Processing Section
```json
{
  "ai_processing": {
    "transcription": {
      "model": "whisper-base",
      "language": "auto",
      "enable_word_timestamps": true,
      "enable_speaker_diarization": true,
      "confidence_threshold": 0.7,
      "chunk_length": 30,
      "overlap_length": 1,
      "enable_vad": true,
      "vad_aggressiveness": 2
    },
    "analysis": {
      "enable_meeting_analysis": true,
      "enable_sentiment_analysis": true,
      "enable_topic_extraction": true,
      "enable_action_item_detection": true,
      "llm_model": "phi-3-mini",
      "max_context_length": 2048,
      "analysis_interval": 60
    }
  }
}
```

**Key Settings:**
- **Transcription**: Whisper model configuration, language detection, timestamps, speaker diarization
- **Analysis**: Meeting analysis features, sentiment analysis, topic extraction, LLM configuration

##### UI System Section
```json
{
  "ui_system": {
    "framework": "kivy",
    "window_size": [800, 480],
    "fullscreen": true,
    "orientation": "landscape",
    "theme": "dark",
    "font_size": 16,
    "enable_touch_feedback": true,
    "touch_sensitivity": 0.8,
    "enable_gestures": true,
    "screen_timeout": 300,
    "brightness": 0.8,
    "enable_screen_saver": true,
    "interface_language": "en"
  }
}
```

**Key Settings:**
- `framework`: UI framework (Kivy)
- `window_size`: Display dimensions
- `fullscreen`: Full-screen mode
- `orientation`: Screen orientation
- `theme`: UI theme selection
- Touch and gesture settings
- Display brightness and power management
- Interface language

##### Database Section
```json
{
  "database": {
    "path": "data/silentst.db",
    "pool_size": 10,
    "timeout": 30,
    "enable_wal": true,
    "enable_foreign_keys": true,
    "cache_size": 2000,
    "backup": {
      "enable_auto_backup": true,
      "backup_interval": 3600,
      "backup_retention": 7,
      "backup_location": "backups/"
    }
  }
}
```

**Key Settings:**
- `path`: Database file location
- `pool_size`: Connection pool size
- `timeout`: Connection timeout
- SQLite-specific settings (WAL mode, foreign keys, cache size)
- Backup configuration (auto-backup, interval, retention)

##### Export System Section
```json
{
  "export_system": {
    "default_format": "pdf",
    "temp_directory": "/tmp/exports",
    "max_concurrent_exports": 3,
    "formats": {
      "pdf": {
        "page_size": "A4",
        "margin": 72,
        "font_family": "Arial",
        "include_header": true,
        "include_footer": true
      },
      "email": {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "use_tls": true,
        "max_attachment_size": 25
      }
    }
  }
}
```

**Key Settings:**
- `default_format`: Default export format
- `temp_directory`: Temporary file location
- `max_concurrent_exports`: Export concurrency limit
- Format-specific settings for PDF and email export

##### Storage Section
```json
{
  "storage": {
    "data_directory": "data/",
    "recordings_directory": "recordings/",
    "exports_directory": "exports/",
    "logs_directory": "logs/",
    "max_session_size": 100,
    "max_total_storage": 1000,
    "cleanup_threshold": 0.9,
    "enable_compression": true,
    "compression_level": 6
  }
}
```

**Key Settings:**
- Directory paths for different data types
- Storage limits and cleanup thresholds
- Compression settings

##### Network Section
```json
{
  "network": {
    "enable_wifi": true,
    "enable_bluetooth": true,
    "enable_hotspot": false,
    "enable_sharing": true,
    "sharing_port": 8080,
    "enable_remote_access": false,
    "firewall_enabled": true,
    "allowed_networks": ["192.168.1.0/24", "10.0.0.0/8"]
  }
}
```

**Key Settings:**
- Network interface enablement
- Sharing and remote access configuration
- Security settings (firewall, allowed networks)

##### Security Section
```json
{
  "security": {
    "enable_encryption": true,
    "encryption_key_path": "keys/encryption.key",
    "enable_access_control": false,
    "session_timeout": 3600,
    "enable_audit_logging": true,
    "privacy_mode": false,
    "data_retention_days": 90
  }
}
```

**Key Settings:**
- Encryption configuration
- Access control and authentication
- Audit logging and privacy settings
- Data retention policies

##### Experimental Features Section
```json
{
  "experimental": {
    "enable_real_time_translation": false,
    "enable_ai_interruption_detection": false,
    "enable_emotion_detection": false,
    "enable_meeting_insights": false,
    "enable_predictive_scheduling": false
  }
}
```

**Key Settings:**
- Feature flags for experimental functionality
- All disabled by default for stability

**Usage Example:**
```python
# Loading and using app configuration
from src.core.config import load_config

# Load configuration
config = load_config("config/app_config.json")

# Access specific settings
audio_config = config["audio_pipeline"]
sample_rate = audio_config["sample_rate"]  # 44100

# Use in application
audio_pipeline = AudioPipeline(
    sample_rate=sample_rate,
    channels=audio_config["channels"],
    buffer_size=audio_config["buffer_size"]
)
```

### 2. `device_config.json`

**Purpose**: Device-specific configuration for hardware management and system health monitoring.

#### Structure
```json
{
  "device": {
    "name": "SilentSteno-Pi5",
    "device_id": "ss-pi5-001",
    "location": "Conference Room A",
    "model": "Raspberry Pi 5",
    "firmware_version": "1.0.0"
  },
  "health_monitoring": {
    "enabled": true,
    "check_interval": 60,
    "temperature_threshold": 80,
    "memory_threshold": 85,
    "disk_threshold": 90,
    "alert_recipients": ["admin@company.com"]
  },
  "storage_management": {
    "enabled": true,
    "cleanup_interval": 86400,
    "retention_policy": "30_days",
    "backup_before_cleanup": true
  },
  "updates": {
    "auto_update": false,
    "update_channel": "stable",
    "update_check_interval": 86400
  },
  "remote_management": {
    "enabled": false,
    "ssh_enabled": false,
    "vnc_enabled": false,
    "management_port": 22
  },
  "diagnostics": {
    "enabled": true,
    "collect_system_info": true,
    "collect_performance_metrics": true,
    "diagnostic_interval": 300
  },
  "factory_reset": {
    "enabled": true,
    "preserve_user_data": false,
    "reset_confirmation_required": true
  }
}
```

**Key Settings:**
- **Device**: Hardware identification and metadata
- **Health Monitoring**: System health thresholds and alerting
- **Storage Management**: Automated cleanup and retention
- **Updates**: Software update configuration
- **Remote Management**: SSH/VNC access settings
- **Diagnostics**: System monitoring and metrics collection
- **Factory Reset**: Reset options and data preservation

**Usage Example:**
```python
# Using device configuration
from src.system.device_manager import DeviceManager
from src.core.config import load_config

device_config = load_config("config/device_config.json")
device_manager = DeviceManager(device_config)

# Monitor device health
if device_config["health_monitoring"]["enabled"]:
    device_manager.start_health_monitoring()

# Check temperature threshold
temp_threshold = device_config["health_monitoring"]["temperature_threshold"]
if device_manager.get_temperature() > temp_threshold:
    device_manager.send_alert("High temperature detected")
```

### 3. `logging_config.json`

**Purpose**: Comprehensive logging configuration following Python's logging.config format.

#### Structure Overview

##### Formatters Section
```json
{
  "formatters": {
    "structured": {
      "format": "{\"timestamp\": \"%(asctime)s\", \"level\": \"%(levelname)s\", \"logger\": \"%(name)s\", \"message\": \"%(message)s\", \"module\": \"%(module)s\", \"function\": \"%(funcName)s\", \"line\": %(lineno)d}",
      "datefmt": "%Y-%m-%d %H:%M:%S"
    },
    "detailed": {
      "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
      "datefmt": "%Y-%m-%d %H:%M:%S"
    },
    "simple": {
      "format": "%(levelname)s - %(message)s"
    },
    "console": {
      "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
      "datefmt": "%H:%M:%S"
    }
  }
}
```

**Formatter Types:**
- `structured`: JSON-formatted logs for machine parsing
- `detailed`: Comprehensive human-readable format
- `simple`: Minimal format for basic logging
- `console`: Console-optimized format

##### Handlers Section
```json
{
  "handlers": {
    "console": {
      "class": "logging.StreamHandler",
      "level": "INFO",
      "formatter": "console",
      "stream": "ext://sys.stdout"
    },
    "file": {
      "class": "logging.handlers.RotatingFileHandler",
      "level": "DEBUG",
      "formatter": "detailed",
      "filename": "logs/silentst.log",
      "maxBytes": 10485760,
      "backupCount": 5,
      "encoding": "utf8"
    },
    "error_file": {
      "class": "logging.handlers.RotatingFileHandler",
      "level": "ERROR",
      "formatter": "structured",
      "filename": "logs/errors.log",
      "maxBytes": 5242880,
      "backupCount": 3,
      "encoding": "utf8"
    },
    "audit_file": {
      "class": "logging.handlers.TimedRotatingFileHandler",
      "level": "INFO",
      "formatter": "structured",
      "filename": "logs/audit.log",
      "when": "midnight",
      "interval": 1,
      "backupCount": 30,
      "encoding": "utf8"
    }
  }
}
```

**Handler Types:**
- `console`: Standard output logging
- `file`: Main application log with rotation
- `error_file`: Error-only log file
- `audit_file`: Audit trail with daily rotation

##### Loggers Section
```json
{
  "loggers": {
    "silentst": {
      "level": "DEBUG",
      "handlers": ["console", "file"],
      "propagate": false
    },
    "silentst.core": {
      "level": "INFO",
      "handlers": ["file"],
      "propagate": false
    },
    "silentst.audio": {
      "level": "DEBUG",
      "handlers": ["file"],
      "propagate": false
    },
    "silentst.ai": {
      "level": "INFO",
      "handlers": ["file"],
      "propagate": false
    },
    "silentst.ui": {
      "level": "WARNING",
      "handlers": ["console", "file"],
      "propagate": false
    },
    "silentst.bluetooth": {
      "level": "INFO",
      "handlers": ["file"],
      "propagate": false
    },
    "silentst.data": {
      "level": "INFO",
      "handlers": ["file", "audit_file"],
      "propagate": false
    },
    "silentst.export": {
      "level": "INFO",
      "handlers": ["file"],
      "propagate": false
    }
  }
}
```

**Module-Specific Loggers:**
- Different log levels for different modules
- Selective handler assignment
- Audit logging for data operations

##### SilentSteno-Specific Configuration
```json
{
  "silentst_config": {
    "performance_logging": {
      "enabled": true,
      "log_level": "INFO",
      "metrics_interval": 60,
      "log_slow_operations": true,
      "slow_operation_threshold": 1.0
    },
    "security_logging": {
      "enabled": true,
      "log_level": "WARNING",
      "log_authentication": true,
      "log_authorization": true,
      "log_data_access": true
    },
    "log_retention": {
      "main_log_days": 30,
      "error_log_days": 90,
      "audit_log_days": 365,
      "performance_log_days": 7
    },
    "log_alerts": {
      "enabled": false,
      "alert_on_error_threshold": 10,
      "alert_on_critical": true,
      "alert_recipients": ["admin@company.com"]
    }
  }
}
```

**Custom Features:**
- Performance logging configuration
- Security event logging
- Log retention policies
- Alert thresholds and recipients

**Usage Example:**
```python
# Using logging configuration
import logging.config
import json

# Load logging configuration
with open("config/logging_config.json", "r") as f:
    logging_config = json.load(f)

# Configure logging
logging.config.dictConfig(logging_config)

# Get module-specific logger
logger = logging.getLogger("silentst.audio")
logger.info("Audio system initialized")

# Performance logging
perf_logger = logging.getLogger("silentst.performance")
perf_logger.info("Operation completed in 0.5s")
```

### 4. `theme_config.json`

**Purpose**: Simple theme configuration for UI appearance management.

#### Structure
```json
{
  "current_theme": "Dark",
  "auto_theme": false,
  "last_updated": "2024-01-15T10:30:00Z"
}
```

**Key Settings:**
- `current_theme`: Currently active theme name
- `auto_theme`: Automatic theme switching based on time/system
- `last_updated`: Last configuration update timestamp

**Usage Example:**
```python
# Using theme configuration
from src.ui.themes import ThemeManager
from src.core.config import load_config

theme_config = load_config("config/theme_config.json")
theme_manager = ThemeManager()

# Apply current theme
current_theme = theme_config["current_theme"]
theme_manager.set_theme(current_theme)

# Check auto-theme setting
if theme_config["auto_theme"]:
    theme_manager.enable_auto_theme()
```

## Configuration Management

### Loading Configuration
```python
# Load all configurations
from src.core.config import ConfigManager

config_manager = ConfigManager()

# Add configuration sources
config_manager.add_source("config/app_config.json", priority=10)
config_manager.add_source("config/device_config.json", priority=20)
config_manager.add_source("config/logging_config.json", priority=30)
config_manager.add_source("config/theme_config.json", priority=40)

# Access configuration values
app_name = config_manager.get("application.name")
sample_rate = config_manager.get("audio_pipeline.sample_rate")
log_level = config_manager.get("application.log_level")
```

### Configuration Validation
```python
# Validate configuration
from src.core.config import ConfigValidator

validator = ConfigValidator()

# Register validation rules
validator.register_validator(
    "audio_pipeline.sample_rate", 
    lambda x: x in [8000, 16000, 22050, 44100, 48000, 96000]
)

validator.register_validator(
    "audio_pipeline.latency_target",
    lambda x: 10 <= x <= 1000
)

# Validate configuration
if validator.validate_config(config_manager.get_all()):
    print("Configuration is valid")
else:
    print("Configuration validation failed")
```

### Configuration Hot-Reload
```python
# Watch for configuration changes
from src.core.config import ConfigWatcher

def on_config_change(key_path, new_value, old_value):
    print(f"Configuration changed: {key_path} = {new_value}")
    
    # Handle specific changes
    if key_path == "audio_pipeline.sample_rate":
        audio_system.update_sample_rate(new_value)
    elif key_path == "ui_system.theme":
        theme_manager.set_theme(new_value)

# Register change callback
config_manager.register_change_callback(on_config_change)

# Configuration will automatically reload when files change
```

### Environment-Specific Configuration
```python
# Override configuration with environment variables
import os

# Environment variables override file settings
# SILENTST_AUDIO_PIPELINE_SAMPLE_RATE=48000
# SILENTST_DATABASE_PATH=/custom/path/db.sqlite

# The ConfigManager automatically applies environment overrides
config_manager = ConfigManager()
config_manager.add_source("config/app_config.json")

# Environment variables take precedence
sample_rate = config_manager.get("audio_pipeline.sample_rate")
# Will use environment value if set, otherwise file value
```

## Best Practices

### 1. Configuration Organization
- Keep related settings grouped in logical sections
- Use consistent naming conventions
- Include comments and descriptions for complex settings
- Separate environment-specific settings

### 2. Validation and Error Handling
- Always validate configuration values
- Provide sensible defaults for optional settings
- Handle missing or invalid configuration gracefully
- Use type hints and validation schemas

### 3. Security Considerations
- Never store sensitive information in configuration files
- Use environment variables for secrets
- Implement proper access controls for configuration files
- Consider encryption for sensitive configuration data

### 4. Performance Optimization
- Cache frequently accessed configuration values
- Use configuration hot-reload judiciously
- Minimize configuration file size and complexity
- Consider binary formats for large configuration sets

### 5. Deployment and Maintenance
- Use version control for configuration files
- Implement configuration backup and restore
- Document configuration changes and their impacts
- Test configuration changes in staging environments

This comprehensive configuration system provides flexible, maintainable, and secure configuration management for The Silent Steno application, supporting both development and production environments with proper validation, monitoring, and hot-reload capabilities.