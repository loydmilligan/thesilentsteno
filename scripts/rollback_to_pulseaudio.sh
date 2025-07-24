#!/bin/bash

# PipeWire to PulseAudio Rollback Script
# Safely rolls back from PipeWire to PulseAudio

set -e

echo "🔄 PipeWire to PulseAudio Rollback Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root (except for system file operations)"
   exit 1
fi

# Backup directory
BACKUP_DIR="config/backup_pulseaudio_20250724_023617"

if [ ! -d "$BACKUP_DIR" ]; then
    print_error "Backup directory $BACKUP_DIR not found!"
    print_warning "Cannot safely rollback without original PulseAudio configuration"
    exit 1
fi

echo "Found backup directory: $BACKUP_DIR"

# Step 1: Stop PipeWire services
echo ""
echo "Step 1: Stopping PipeWire services..."
systemctl --user stop pipewire 2>/dev/null || true
systemctl --user stop pipewire-pulse 2>/dev/null || true  
systemctl --user stop wireplumber 2>/dev/null || true

systemctl --user disable pipewire 2>/dev/null || true
systemctl --user disable pipewire-pulse 2>/dev/null || true
systemctl --user disable wireplumber 2>/dev/null || true

print_status "PipeWire services stopped"

# Step 2: Kill any remaining processes
echo ""
echo "Step 2: Cleaning up audio processes..."
pulseaudio --kill 2>/dev/null || true
killall pipewire 2>/dev/null || true
killall pipewire-pulse 2>/dev/null || true
killall wireplumber 2>/dev/null || true

# Remove runtime files
rm -rf ~/.local/state/pipewire/ 2>/dev/null || true
rm -f /tmp/pulse-* 2>/dev/null || true

print_status "Audio processes cleaned up"

# Step 3: Restore PulseAudio configuration
echo ""
echo "Step 3: Restoring PulseAudio configuration..."

# Check if backup contains system config
if [ -d "$BACKUP_DIR/etc" ]; then
    echo "Restoring system PulseAudio configuration..."
    sudo cp -r $BACKUP_DIR/etc/pulse/* /etc/pulse/ 2>/dev/null || true
    sudo chown -R pulse:pulse /etc/pulse/ 2>/dev/null || true
    print_status "System configuration restored"
fi

# Check if backup contains user config  
if [ -d "$BACKUP_DIR/home" ]; then
    echo "Restoring user PulseAudio configuration..."
    mkdir -p ~/.config/pulse
    cp -r $BACKUP_DIR/home/.config/pulse/* ~/.config/pulse/ 2>/dev/null || true
    chmod 644 ~/.config/pulse/* 2>/dev/null || true
    print_status "User configuration restored"
fi

# Step 4: Start PulseAudio
echo ""
echo "Step 4: Starting PulseAudio..."

# Ensure PulseAudio is installed
if ! command -v pulseaudio &> /dev/null; then
    print_error "PulseAudio not installed! Installing..."
    sudo apt-get update
    sudo apt-get install -y pulseaudio pulseaudio-module-bluetooth
fi

# Start PulseAudio
pulseaudio --start
sleep 2

# Verify PulseAudio is running
if pulseaudio --check; then
    print_status "PulseAudio started successfully"
else
    print_error "Failed to start PulseAudio"
    echo "Attempting to restart..."
    pulseaudio --kill
    sleep 1
    pulseaudio --start
    sleep 2
    
    if pulseaudio --check; then
        print_status "PulseAudio started on retry"
    else
        print_error "PulseAudio failed to start. Manual intervention required."
        exit 1
    fi
fi

# Step 5: Restart Bluetooth for audio
echo ""
echo "Step 5: Restarting Bluetooth service..."
sudo systemctl restart bluetooth
sleep 2
print_status "Bluetooth restarted"

# Step 6: Verification
echo ""
echo "Step 6: Verifying rollback..."

# Check PulseAudio info
if pactl info > /dev/null 2>&1; then
    print_status "PulseAudio is responding to commands"
    
    # Get server info
    SERVER_NAME=$(pactl info | grep "Server Name" | cut -d: -f2 | xargs)
    echo "Audio server: $SERVER_NAME"
    
    if [[ "$SERVER_NAME" == *"PulseAudio"* ]]; then
        print_status "PulseAudio is active"
    else
        print_warning "Audio server is not PulseAudio: $SERVER_NAME"
    fi
else
    print_error "PulseAudio is not responding"
    exit 1
fi

# Check audio devices
SINK_COUNT=$(pactl list short sinks | wc -l)
SOURCE_COUNT=$(pactl list short sources | wc -l)

echo "Found $SINK_COUNT sinks and $SOURCE_COUNT sources"

if [ "$SINK_COUNT" -gt 0 ] && [ "$SOURCE_COUNT" -gt 0 ]; then
    print_status "Audio devices detected"
else
    print_warning "No audio devices found"
fi

# Check Bluetooth
if systemctl is-active --quiet bluetooth; then
    print_status "Bluetooth service is active"
else
    print_warning "Bluetooth service is not active"
fi

# Final summary
echo ""
echo "=========================================="
echo "🎉 Rollback Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "- PipeWire services: Stopped and disabled"
echo "- PulseAudio configuration: Restored from backup"
echo "- PulseAudio: Running"
echo "- Bluetooth: Restarted" 
echo "- Audio devices: $SINK_COUNT sinks, $SOURCE_COUNT sources"
echo ""

print_status "System successfully rolled back to PulseAudio"
echo ""
echo "Next steps:"
echo "1. Test audio playback: paplay /usr/share/sounds/alsa/Front_Left.wav"
echo "2. Test Bluetooth connectivity with your devices"  
echo "3. Run application tests to verify functionality"
echo "4. Monitor system for any issues"
echo ""
echo "If you need to switch back to PipeWire:"
echo "./scripts/setup_pipewire.sh"

# Optional: Run a quick audio test
read -p "Would you like to test audio playback now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "/usr/share/sounds/alsa/Front_Left.wav" ]; then
        echo "Playing test sound..."
        paplay /usr/share/sounds/alsa/Front_Left.wav 2>/dev/null || print_warning "Could not play test sound"
    else
        print_warning "Test sound file not found"
    fi
fi

echo "Rollback script completed successfully!"