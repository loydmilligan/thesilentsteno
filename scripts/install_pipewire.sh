#!/bin/bash

# The Silent Steno - PipeWire Installation Script
# This script installs PipeWire and configures it for Bluetooth audio

set -e  # Exit on error

echo "🎵 The Silent Steno - PipeWire Installation"
echo "==========================================="
echo ""
echo "⚠️  WARNING: This will replace PulseAudio with PipeWire"
echo "Make sure you have backed up your configuration!"
echo ""
echo "Press Ctrl+C to cancel, or Enter to continue..."
read -r

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check command status
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $1${NC}"
    else
        echo -e "${RED}❌ $1 failed${NC}"
        exit 1
    fi
}

# Create log file
LOG_FILE="/tmp/pipewire_install_$(date +%Y%m%d_%H%M%S).log"
echo "Installation log: $LOG_FILE"
echo ""

# Update package list
echo "1. Updating package list..."
sudo apt update 2>&1 | tee -a "$LOG_FILE"
check_status "Package list updated"

# Install PipeWire packages
echo ""
echo "2. Installing PipeWire packages..."
PACKAGES=(
    "pipewire"
    "pipewire-pulse"
    "pipewire-alsa"
    "pipewire-jack"
    "wireplumber"
    "libspa-0.2-bluetooth"
    "libspa-0.2-jack"
    "pipewire-audio-client-libraries"
)

echo "Installing: ${PACKAGES[*]}"
sudo apt install -y "${PACKAGES[@]}" 2>&1 | tee -a "$LOG_FILE"
check_status "PipeWire packages installed"

# Install additional tools
echo ""
echo "3. Installing additional tools..."
sudo apt install -y pipewire-tools 2>&1 | tee -a "$LOG_FILE"
check_status "Additional tools installed"

# Stop PulseAudio if running
echo ""
echo "4. Stopping PulseAudio..."
if systemctl --user is-active --quiet pulseaudio; then
    systemctl --user stop pulseaudio
    systemctl --user disable pulseaudio
    systemctl --user mask pulseaudio
    check_status "PulseAudio stopped and disabled"
else
    echo "   PulseAudio not running"
fi

# Create PipeWire configuration directory
echo ""
echo "5. Creating configuration directories..."
mkdir -p ~/.config/pipewire
mkdir -p ~/.config/wireplumber
mkdir -p ~/.config/pipewire/pipewire.conf.d
mkdir -p ~/.config/wireplumber/main.lua.d
check_status "Configuration directories created"

# Enable PipeWire services
echo ""
echo "6. Enabling PipeWire services..."
systemctl --user enable pipewire.service
systemctl --user enable pipewire-pulse.service
systemctl --user enable wireplumber.service
check_status "PipeWire services enabled"

# Start PipeWire services
echo ""
echo "7. Starting PipeWire services..."
systemctl --user start pipewire.service
sleep 2
systemctl --user start pipewire-pulse.service
sleep 2
systemctl --user start wireplumber.service
sleep 3
check_status "PipeWire services started"

# Verify installation
echo ""
echo "8. Verifying installation..."
echo -n "   Checking PipeWire server... "
if pw-cli info > /dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

echo -n "   Checking PulseAudio compatibility... "
if pactl info > /dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
    echo "   Server: $(pactl info | grep 'Server Name' | cut -d: -f2)"
else
    echo -e "${RED}FAILED${NC}"
fi

echo -n "   Checking WirePlumber... "
if wpctl status > /dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${YELLOW}WARNING${NC}"
fi

# Display status
echo ""
echo "9. Current audio status:"
echo "------------------------"
pw-cli info | grep -E "properties|remote.name|remote.version" | head -5

# Check Bluetooth
echo ""
echo "10. Checking Bluetooth support..."
if lsmod | grep -q bluetooth; then
    echo -e "   ${GREEN}✅ Bluetooth kernel module loaded${NC}"
fi

wpctl status | grep -A5 "Bluetooth" || echo "   No Bluetooth devices currently connected"

# Create convenience scripts
echo ""
echo "11. Creating convenience scripts..."

# Audio status script
cat > ~/bin/audio-status << 'EOF'
#!/bin/bash
echo "PipeWire Status:"
pw-cli info | head -5
echo ""
echo "Audio Devices:"
wpctl status
EOF
chmod +x ~/bin/audio-status 2>/dev/null || true

# Installation summary
echo ""
echo "========================================"
echo -e "${GREEN}✅ PIPEWIRE INSTALLATION COMPLETE${NC}"
echo "========================================"
echo ""
echo "What's next:"
echo "1. Log out and log back in (or reboot) for full effect"
echo "2. Run: wpctl status      - to see audio devices"
echo "3. Run: pw-top            - to monitor PipeWire processes"
echo "4. Run: ~/bin/audio-status - for quick status check"
echo ""
echo "To configure Bluetooth devices:"
echo "   bluetoothctl"
echo "   > power on"
echo "   > scan on"
echo "   > pair <device_mac>"
echo ""
echo "Installation log saved to: $LOG_FILE"