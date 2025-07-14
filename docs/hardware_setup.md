# Hardware Setup Guide - The Silent Steno

Complete hardware setup and troubleshooting guide for the Bluetooth AI Meeting Recorder.

## Overview

The Silent Steno runs on a Raspberry Pi 5 with a touchscreen display, providing a dedicated device for capturing and processing meeting audio with AI-powered transcription and analysis.

## Hardware Requirements

### Core Components
- **Platform**: Raspberry Pi 5 (4GB+ RAM recommended)
- **Display**: 3.5-5 inch touchscreen (480x320 or 800x480)
- **Audio**: Built-in audio + optional USB audio interface
- **Connectivity**: Built-in WiFi and Bluetooth 5.0
- **Power**: Wall adapter (no battery requirement)
- **Storage**: 32GB+ SD card for 20+ hours of meetings

### Optional Components
- **Enclosure**: 3D printable case with screen cutout
- **USB Audio Interface**: For improved audio quality
- **External Antenna**: For better Bluetooth/WiFi range

## Current Setup Status

### ✅ Completed Components
1. **Raspberry Pi OS**: Installed with desktop environment
2. **Touchscreen**: Configured (reliability issues noted)
3. **SSH Access**: Working remotely
4. **Basic Development Tools**: Python 3.11.2, git, build-essential installed

### 🔧 In Progress
1. **Touch Calibration**: Needs improvement for reliability
2. **VNC Access**: Optional, not yet configured
3. **Auto-boot**: Service configured but disabled
4. **Hardware Testing**: Comprehensive testing needed

## Setup Instructions

### 1. Initial OS Setup
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3-dev python3-pip build-essential git vim
```

### 2. Development Environment
```bash
# Run the hardware setup script
cd /home/mmariani/projects/thesilentsteno
./scripts/setup_hardware.sh all
```

### 3. Display Configuration
The touchscreen is currently configured but has reliability issues. To improve:

```bash
# Check current display setup
cat /boot/config.txt | grep -E "(dtoverlay|display|hdmi)"

# Diagnose touch input
dmesg | grep -i touch
ls /dev/input/
cat /proc/bus/input/devices
```

### 4. Auto-boot Configuration
The system service is configured but disabled by default:

```bash
# Enable auto-boot (when application is ready)
sudo systemctl enable silentsteno.service
sudo systemctl start silentsteno.service

# Check status
sudo systemctl status silentsteno.service
```

## Hardware Testing

### Manual Testing Commands
```bash
# Test all hardware components
./scripts/setup_hardware.sh test

# Test specific components
./scripts/setup_hardware.sh status

# Audio devices
aplay -l
arecord -l

# Bluetooth
bluetoothctl list
sudo systemctl status bluetooth

# Display and input
echo $DISPLAY
ls /dev/input/event*
```

### Expected Results
- Python 3.8+ installed and working
- Git version control available
- Bluetooth service running
- Audio devices detected
- Touch input devices present
- SSH connection stable

## Troubleshooting

### Touch Screen Issues
**Problem**: Touch input unreliable or not responding
**Solutions**:
1. Check touch device detection:
   ```bash
   dmesg | grep -i touch
   ls /dev/input/
   ```
2. Verify display configuration in `/boot/config.txt`
3. Calibrate touch input using system tools
4. Check for proper driver installation

### Audio Issues
**Problem**: No audio devices detected
**Solutions**:
1. Check ALSA configuration:
   ```bash
   aplay -l
   amixer
   ```
2. Verify PulseAudio status:
   ```bash
   pulseaudio --check
   pactl info
   ```
3. Check USB audio interfaces if connected

### Bluetooth Issues
**Problem**: Bluetooth not working or devices not pairing
**Solutions**:
1. Check Bluetooth service:
   ```bash
   sudo systemctl status bluetooth
   bluetoothctl list
   ```
2. Restart Bluetooth:
   ```bash
   sudo systemctl restart bluetooth
   ```
3. Check for interference from other devices

### SSH/Network Issues
**Problem**: Cannot connect via SSH
**Solutions**:
1. Verify SSH service:
   ```bash
   sudo systemctl status ssh
   ```
2. Check network configuration:
   ```bash
   ip addr show
   ping google.com
   ```
3. Verify firewall settings

### Performance Issues
**Problem**: System running slowly
**Solutions**:
1. Check system resources:
   ```bash
   htop
   free -h
   df -h
   ```
2. Monitor CPU temperature:
   ```bash
   vcgencmd measure_temp
   ```
3. Verify SD card speed and health

## Configuration Files

### Important System Files
- `/boot/config.txt` - Boot and hardware configuration
- `/etc/ssh/sshd_config` - SSH daemon configuration
- `/etc/bluetooth/main.conf` - Bluetooth configuration (future)
- `/etc/pulse/default.pa` - PulseAudio configuration (future)

### Project Files
- `scripts/setup_hardware.sh` - Hardware setup automation
- `config/display_config.txt` - Display configuration reference
- `scripts/autostart.sh` - Application auto-start script

## Performance Targets

### Audio Performance
- **Latency**: <40ms end-to-end (target)
- **Quality**: 16-bit/44.1kHz minimum
- **Reliability**: >99% session completion

### System Performance
- **Startup**: <10 seconds to application ready
- **Memory**: <2GB usage during operation
- **Storage**: 32GB+ recommended for long sessions

### Network Performance
- **Bluetooth**: Stable A2DP connections
- **WiFi**: Stable for remote management
- **Range**: 10+ meters for Bluetooth devices

## Next Steps

1. **Improve Touch Reliability**: Identify and configure proper touchscreen drivers
2. **Audio Pipeline Setup**: Configure Bluetooth audio stack (Task 1.2)
3. **Performance Testing**: Comprehensive hardware validation
4. **Enclosure Design**: 3D printable case development
5. **Production Setup**: Image creation for deployment

## Support

### Log Files
- Hardware setup: `logs/autostart.log`
- System logs: `journalctl -u silentsteno.service`
- Bluetooth: `journalctl -u bluetooth.service`

### Debug Commands
```bash
# System status
./scripts/setup_hardware.sh status

# Hardware test
./scripts/setup_hardware.sh test

# Service status
sudo systemctl status silentsteno.service

# View logs
tail -f logs/autostart.log
```

For additional support, check the project documentation in `docs/` directory.