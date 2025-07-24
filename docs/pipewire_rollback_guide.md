# PipeWire Rollback Guide

This document provides instructions for rolling back from PipeWire to PulseAudio if needed.

## Quick Rollback Commands

```bash
# Stop PipeWire services
systemctl --user stop pipewire pipewire-pulse wireplumber

# Restore PulseAudio configuration
sudo cp -r config/backup_pulseaudio_20250724_023617/* /etc/pulse/

# Start PulseAudio
systemctl --user start pulseaudio

# Verify rollback
pulseaudio --check -v
```

## Detailed Rollback Process

### 1. Stop PipeWire Services

```bash
# Stop user services
systemctl --user stop pipewire
systemctl --user stop pipewire-pulse
systemctl --user stop wireplumber

# Disable autostart
systemctl --user disable pipewire
systemctl --user disable pipewire-pulse
systemctl --user disable wireplumber

# Verify services are stopped
systemctl --user status pipewire pipewire-pulse wireplumber
```

### 2. Restore PulseAudio Configuration

The backup was created in `config/backup_pulseaudio_20250724_023617/`:

```bash
# Restore system configuration
sudo cp -r config/backup_pulseaudio_20250724_023617/etc/pulse/* /etc/pulse/

# Restore user configuration (if exists)
if [ -d config/backup_pulseaudio_20250724_023617/home ]; then
    cp -r config/backup_pulseaudio_20250724_023617/home/.config/pulse/ ~/.config/
fi

# Set correct permissions
sudo chown -R pulse:pulse /etc/pulse/
chmod 644 ~/.config/pulse/* 2>/dev/null || true
```

### 3. Start PulseAudio

```bash
# Kill any remaining audio processes
pulseaudio --kill 2>/dev/null || true
killall pipewire 2>/dev/null || true

# Start PulseAudio
pulseaudio --start

# Verify PulseAudio is running
pulseaudio --check -v
pactl info
```

### 4. Update Code Configuration

If you need to force the application to use PulseAudio:

```python
# In src/audio/audio_system_factory.py, temporarily force PulseAudio:
@staticmethod
def detect_audio_system() -> AudioSystemType:
    return AudioSystemType.PULSEAUDIO  # Force PulseAudio
```

### 5. Verification Steps

```bash
# Check audio system
pactl info | grep "Server Name"

# List audio devices
pactl list short sinks
pactl list short sources

# Test audio playback
paplay /usr/share/sounds/alsa/Front_Left.wav

# Check Bluetooth functionality
bluetoothctl show
```

## Troubleshooting Rollback Issues

### PulseAudio Won't Start

```bash
# Check for conflicts
ps aux | grep -E "(pipewire|pulseaudio)"

# Force kill all audio processes
sudo killall -9 pipewire pipewire-pulse wireplumber pulseaudio 2>/dev/null

# Remove PipeWire runtime files
rm -rf ~/.local/state/pipewire/
rm -f /tmp/pulse-* 2>/dev/null

# Restart PulseAudio
pulseaudio --start
```

### Missing PulseAudio Configuration

If the backup is missing or corrupted:

```bash
# Reinstall PulseAudio configuration
sudo apt-get --reinstall install pulseaudio pulseaudio-module-bluetooth

# Generate default configuration
pulseaudio --dump-conf > ~/.config/pulse/default.pa
```

### Bluetooth Issues After Rollback

```bash
# Restart Bluetooth service
sudo systemctl restart bluetooth

# Reset Bluetooth module
sudo modprobe -r btusb
sudo modprobe btusb

# Restart PulseAudio with Bluetooth
pulseaudio --kill
pulseaudio --start
```

## System Files Modified During PipeWire Installation

### Configuration Files Created:
- `config/pipewire_config/pipewire.conf`
- `config/pipewire_config/wireplumber.conf`
- `config/pipewire_config/pipewire-pulse.conf`

### Scripts Created:
- `scripts/install_pipewire.sh`
- `scripts/setup_pipewire.sh`

### Code Files Added:
- `src/audio/pipewire_backend.py`
- `src/bluetooth/pipewire_bluetooth.py`
- `tests/pipewire/` (entire directory)

### Files Modified:
- `src/audio/audio_system_factory.py` - Added PipeWire support
- `templates/settings.html` - Added audio system settings

## Emergency Recovery

If the system is completely broken:

```bash
# Nuclear option - remove all audio
sudo apt-get remove --purge pipewire\* wireplumber

# Reinstall PulseAudio
sudo apt-get install pulseaudio pulseaudio-module-bluetooth

# Reboot
sudo reboot
```

## Git Rollback

To rollback code changes:

```bash
# Check current branch
git branch

# Return to previous commit (replace COMMIT_HASH)
git reset --hard <COMMIT_HASH>

# Or merge back to main
git checkout main
git branch -D feature/pipewire-refactor
```

## Performance Comparison

After rollback, run baseline tests to compare performance:

```bash
# Run PulseAudio baseline test
python scripts/test_baseline_performance.py

# Compare with PipeWire benchmarks in:
# tests/performance/audio_benchmark_pipewire_20250724_112151.json
```

## When to Rollback

Consider rolling back if:
- Audio latency is worse than PulseAudio
- Bluetooth connectivity issues
- Application compatibility problems
- System stability issues
- Performance regression

## Re-enabling PipeWire

To switch back to PipeWire after rollback:

```bash
# Run setup script again
./scripts/setup_pipewire.sh

# Or follow the original installation steps
./scripts/install_pipewire.sh
```

---

**Note**: Always test the rollback process in a development environment before applying to production systems.