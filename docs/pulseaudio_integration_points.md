# PulseAudio Integration Points Documentation

This document outlines all the current PulseAudio integration points in The Silent Steno project, identified as part of the PipeWire migration effort.

## Configuration Files

### 1. `/config/pulse_config.pa`
- **Purpose**: Main PulseAudio configuration file
- **Key modules used**:
  - `module-bluetooth-policy` - Auto-switching Bluetooth devices
  - `module-bluetooth-discover` - Bluetooth device discovery
  - `module-bluez5-device` - A2DP sink/source profiles
  - `module-alsa-sink/source` - ALSA device integration
  - `module-loopback` - Audio forwarding from source to sink
  - `module-native-protocol-unix` - Unix socket communication
  - `module-dbus-protocol` - D-Bus interface
  - `module-switch-on-connect` - Auto device switching
  - `module-stream-restore` - Volume/mute state persistence
  - `module-role-cork` - Stream prioritization

### 2. `/setup_bluetooth.sh`
- **Purpose**: Bluetooth setup and configuration script
- **PulseAudio dependencies**:
  - `pulseaudio-module-bluetooth` package installation
  - `pactl` commands for module loading
  - `pactl load-module module-bluetooth-discover`

## Python Code Integration

### 3. Audio System (`/src/audio/`)
- **alsa_manager.py**:
  - References PulseAudio server: `"unix:/run/user/1000/pulse/native"`
  - ALSA-PulseAudio bridge configuration

- **latency_optimizer.py**:
  - Updates PulseAudio settings for latency optimization

### 4. Bluetooth Management (`/src/bluetooth/`)
- **connection_manager.py**:
  - Uses `pactl list sinks` to check codec information
  
- **bluez_manager.py**:
  - Uses `pactl list modules` to verify Bluetooth module loading
  - Multiple `pactl` calls for module management

### 5. Recording Module (`/src/recording/`)
- **bluetooth_audio_recorder_module.py**:
  - Uses `pactl list short sources` to enumerate audio sources
  - Dependency on PulseAudio for source discovery

### 6. System Monitoring (`/src/system/`)
- **diagnostics.py**:
  - Monitors `pulseaudio` process as critical service
  - Checks PulseAudio running status

- **health_monitor.py**:
  - Monitors PulseAudio process health
  - Tracks audio subsystem status

- **update_manager.py**:
  - Lists `pulseaudio` as critical service for updates

## Command Usage Summary

### pactl Commands Used:
1. `pactl list modules` - Check loaded modules
2. `pactl list sinks` - List audio output devices
3. `pactl list short sources` - List audio input devices
4. `pactl load-module module-bluetooth-discover` - Load Bluetooth module

### Process Dependencies:
- `pulseaudio` daemon must be running
- PulseAudio socket at `/run/user/1000/pulse/native`

## Key Integration Points for Migration

1. **Module Loading**: All Bluetooth and audio routing modules
2. **Device Discovery**: Bluetooth and ALSA device enumeration
3. **Audio Routing**: Loopback module for forwarding
4. **Process Management**: Service monitoring and health checks
5. **Configuration**: Sample rates, latency settings, codec preferences

## Files Requiring Updates for PipeWire

### High Priority:
- `/config/pulse_config.pa` → Need PipeWire equivalent configuration
- `/setup_bluetooth.sh` → Update for PipeWire commands
- `/src/bluetooth/bluez_manager.py` → Replace pactl calls
- `/src/recording/bluetooth_audio_recorder_module.py` → Update source enumeration

### Medium Priority:
- `/src/audio/latency_optimizer.py` → PipeWire latency configuration
- `/src/bluetooth/connection_manager.py` → PipeWire device info
- System monitoring scripts → Add PipeWire process monitoring

### Low Priority:
- UI feedback systems (pulse animations unrelated to PulseAudio)
- Documentation updates