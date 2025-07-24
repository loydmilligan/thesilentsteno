#!/bin/bash

# The Silent Steno - Universal Bluetooth Setup Script
# Works with both PipeWire and PulseAudio audio systems

echo "🎧 The Silent Steno - Bluetooth Audio Setup"
echo "=========================================="
echo ""
echo "This will configure your Pi for Bluetooth audio forwarding"
echo "with automatic detection of PipeWire or PulseAudio"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check command status
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $1${NC}"
    else
        echo -e "${RED}❌ $1 failed${NC}"
        return 1
    fi
}

# Detect audio system
detect_audio_system() {
    # Check for PipeWire
    if command -v pw-cli &> /dev/null && pw-cli info &> /dev/null; then
        # Check if it's real PipeWire or just the tools
        if systemctl --user is-active --quiet pipewire; then
            echo "pipewire"
            return
        fi
    fi
    
    # Check for PulseAudio
    if command -v pactl &> /dev/null && pactl info &> /dev/null; then
        # Check if it's PipeWire pretending to be PulseAudio
        if pactl info | grep -q "PipeWire"; then
            echo "pipewire"
        else
            echo "pulseaudio"
        fi
        return
    fi
    
    echo "none"
}

AUDIO_SYSTEM=$(detect_audio_system)
echo -e "${BLUE}Detected audio system: $AUDIO_SYSTEM${NC}"
echo ""

# Check if running as sudo for system commands
if [ "$EUID" -eq 0 ]; then 
   echo -e "${GREEN}✅ Running with sudo privileges${NC}"
else
   echo -e "${YELLOW}⚠️  Some commands may require sudo password${NC}"
fi

# 1. Check Bluetooth service
echo ""
echo "1. Checking Bluetooth service..."
sudo systemctl status bluetooth --no-pager | grep "Active: active" > /dev/null
check_status "Bluetooth service is running"

# 2. Install required packages
echo ""
echo "2. Checking required packages..."
packages="bluez bluez-tools python3-dbus"

# Add audio system specific packages
if [ "$AUDIO_SYSTEM" = "pipewire" ]; then
    packages="$packages pipewire-pulse libspa-0.2-bluetooth wireplumber"
elif [ "$AUDIO_SYSTEM" = "pulseaudio" ]; then
    packages="$packages pulseaudio-module-bluetooth"
fi

missing_packages=""
for pkg in $packages; do
    if ! dpkg -l | grep -q "^ii  $pkg"; then
        missing_packages="$missing_packages $pkg"
    fi
done

if [ -n "$missing_packages" ]; then
    echo "Installing missing packages:$missing_packages"
    sudo apt-get update && sudo apt-get install -y $missing_packages
    check_status "Package installation"
else
    echo -e "${GREEN}✅ All required packages installed${NC}"
fi

# 3. Enable A2DP Sink and Source
echo ""
echo "3. Configuring Bluetooth profiles..."

# Create Bluetooth configuration
if [ ! -f /etc/bluetooth/main.conf.backup ]; then
    sudo cp /etc/bluetooth/main.conf /etc/bluetooth/main.conf.backup
fi

# Update Bluetooth configuration
sudo tee /etc/bluetooth/main.conf > /dev/null << EOF
[General]
Enable=Source,Sink,Media,Socket
DiscoverableTimeout = 0
PairableTimeout = 0
FastConnectable = true
Class = 0x20041C

[Policy]
AutoEnable=true
ReconnectAttempts=7
ReconnectIntervals=1,2,4,8,16,32,64

[A2DP]
# Enable high quality codecs
Codecs = sbc aac aptx aptx_hd ldac
EOF
check_status "Bluetooth configuration updated"

# 4. Configure audio system for Bluetooth
echo ""
echo "4. Configuring $AUDIO_SYSTEM for Bluetooth..."

if [ "$AUDIO_SYSTEM" = "pipewire" ]; then
    # PipeWire configuration
    echo "Setting up PipeWire Bluetooth modules..."
    
    # Check if WirePlumber is managing Bluetooth
    if wpctl status 2>/dev/null | grep -q "Bluetooth"; then
        echo -e "${GREEN}✅ PipeWire Bluetooth support is active${NC}"
    else
        echo -e "${YELLOW}⚠️  Restarting WirePlumber for Bluetooth support...${NC}"
        systemctl --user restart wireplumber
        sleep 2
    fi
    
elif [ "$AUDIO_SYSTEM" = "pulseaudio" ]; then
    # PulseAudio configuration
    echo "Setting up PulseAudio Bluetooth modules..."
    
    # Load Bluetooth modules
    pactl list modules | grep -q module-bluetooth-discover
    if [ $? -ne 0 ]; then
        pactl load-module module-bluetooth-discover
        check_status "Loaded PulseAudio Bluetooth module"
    else
        echo -e "${GREEN}✅ PulseAudio Bluetooth module already loaded${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  No audio system detected. Bluetooth audio may not work properly${NC}"
fi

# 5. Set Pi as discoverable
echo ""
echo "5. Making Pi discoverable..."
bluetoothctl <<EOF
power on
agent on
default-agent
discoverable on
pairable on
EOF
check_status "Pi is now discoverable"

# 6. Show current status
echo ""
echo "6. Current Bluetooth Status:"
echo "----------------------------"
bluetoothctl show | grep -E "Name:|Powered:|Discoverable:|Pairable:"

# 7. List paired devices
echo ""
echo "7. Paired devices:"
echo "------------------"
paired_count=$(bluetoothctl devices | wc -l)
if [ $paired_count -gt 0 ]; then
    bluetoothctl devices
else
    echo "No paired devices found"
fi

# 8. Check for dual radio setup (Silent Steno specific)
echo ""
echo "8. Checking for dual radio setup..."
ADAPTERS=$(bluetoothctl list | grep Controller | wc -l)
if [ $ADAPTERS -gt 1 ]; then
    echo -e "${GREEN}✅ Multiple Bluetooth adapters detected ($ADAPTERS)${NC}"
    echo "Adapter configuration:"
    bluetoothctl list | while read -r line; do
        if [[ $line =~ Controller ]]; then
            MAC=$(echo $line | awk '{print $2}')
            HCI=$(hciconfig | grep -B1 "$MAC" | head -1 | cut -d: -f1)
            echo "  $HCI: $MAC"
        fi
    done
    echo ""
    echo -e "${BLUE}For dual radio operation:${NC}"
    echo "  • Use hci0 for phones (audio sources)"
    echo "  • Use hci1 for headphones (audio sinks)"
else
    echo -e "${YELLOW}⚠️  Only one Bluetooth adapter detected${NC}"
    echo "  Dual radio operation not available"
fi

# 9. Interactive pairing
echo ""
echo "9. Device Pairing"
echo "-----------------"
echo "Would you like to pair a device now? (y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Instructions:"
    echo "1. Put your device in pairing mode"
    echo "2. Wait for it to appear in the scan"
    echo "3. Note the MAC address (XX:XX:XX:XX:XX:XX)"
    echo ""
    
    # If dual radio, ask which adapter to use
    if [ $ADAPTERS -gt 1 ]; then
        echo "Which adapter to use?"
        echo "1. hci0 (for phones/audio sources)"
        echo "2. hci1 (for headphones/audio sinks)"
        read -r adapter_choice
        
        case $adapter_choice in
            1) ADAPTER="hci0" ;;
            2) ADAPTER="hci1" ;;
            *) ADAPTER="hci0" ;;
        esac
        
        echo "Using adapter: $ADAPTER"
        bluetoothctl select $(hciconfig $ADAPTER | grep "BD Address" | awk '{print $3}')
    fi
    
    echo "Press Enter to start scanning..."
    read -r
    
    # Start scanning
    echo "Scanning for devices (20 seconds)..."
    timeout 20 bluetoothctl scan on
    
    echo ""
    echo "Enter the MAC address of the device to pair (or 'skip'):"
    read -r mac_address
    
    if [[ "$mac_address" != "skip" ]] && [[ -n "$mac_address" ]]; then
        echo "Pairing with $mac_address..."
        bluetoothctl pair "$mac_address"
        bluetoothctl trust "$mac_address"
        check_status "Device paired and trusted"
        
        # Connect if it's an audio device
        echo "Attempting to connect..."
        bluetoothctl connect "$mac_address"
        sleep 2
        
        # Check audio profile
        if [ "$AUDIO_SYSTEM" = "pipewire" ]; then
            echo ""
            echo "Audio device status:"
            wpctl status | grep -A5 "$mac_address" || echo "Device not found in audio system"
        fi
    fi
fi

# 10. Audio routing test
echo ""
echo "10. Audio Routing Test"
echo "---------------------"

if [ "$AUDIO_SYSTEM" = "pipewire" ]; then
    echo "PipeWire audio devices:"
    echo "----------------------"
    wpctl status | grep -E "Sinks:|Sources:" -A10
elif [ "$AUDIO_SYSTEM" = "pulseaudio" ]; then
    echo "PulseAudio sources:"
    echo "------------------"
    pactl list short sources | grep bluez
    echo ""
    echo "PulseAudio sinks:"
    echo "----------------"
    pactl list short sinks | grep bluez
fi

# 11. Create systemd service for auto-start
echo ""
echo "11. Auto-start Configuration"
echo "---------------------------"
echo "Would you like to enable Bluetooth audio on boot? (y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    sudo tee /etc/systemd/system/silent-steno-bluetooth.service > /dev/null << EOF
[Unit]
Description=Silent Steno Bluetooth Audio Service
After=bluetooth.service $([ "$AUDIO_SYSTEM" = "pipewire" ] && echo "pipewire.service" || echo "sound.target")

[Service]
Type=simple
ExecStartPre=/bin/sleep 5
ExecStart=/usr/bin/bluetoothctl power on
ExecStartPost=/bin/bash -c 'bluetoothctl agent on && bluetoothctl default-agent'
RemainAfterExit=yes
Restart=on-failure
User=$USER

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl enable silent-steno-bluetooth.service
    check_status "Auto-start service enabled"
fi

# 12. Create helper scripts
echo ""
echo "12. Creating helper scripts..."

# Audio forwarding script
cat > ~/bin/setup-audio-forward << EOF
#!/bin/bash
# Silent Steno Audio Forwarding Setup

echo "Setting up audio forwarding..."

if [ "$AUDIO_SYSTEM" = "pipewire" ]; then
    # PipeWire will handle this automatically with proper configuration
    echo "PipeWire auto-routing is enabled"
    echo "Connect your source and sink devices, audio will forward automatically"
else
    # PulseAudio manual setup
    echo "Available sources:"
    pactl list short sources | grep bluez | nl
    echo ""
    echo "Available sinks:"
    pactl list short sinks | grep bluez | nl
    echo ""
    echo "To create forwarding, run:"
    echo "  pactl load-module module-loopback source=<source_name> sink=<sink_name> latency_msec=40"
fi
EOF
chmod +x ~/bin/setup-audio-forward

echo -e "${GREEN}✅ Helper script created: ~/bin/setup-audio-forward${NC}"

# Final summary
echo ""
echo "========================================"
echo -e "${GREEN}✅ BLUETOOTH SETUP COMPLETE${NC}"
echo "========================================"
echo ""
echo "Audio System: $AUDIO_SYSTEM"
if [ $ADAPTERS -gt 1 ]; then
    echo "Dual Radio: ENABLED"
fi
echo ""
echo "Next steps:"
echo "1. Pair your phone (audio source)"
echo "   - Use hci0 if you have dual radios"
echo "2. Pair your headphones (audio sink)"
echo "   - Use hci1 if you have dual radios"
echo "3. Run: ~/bin/setup-audio-forward"
echo "4. Test audio playback"
echo ""
echo "Useful commands:"
echo "  bluetoothctl          - Bluetooth device management"
if [ "$AUDIO_SYSTEM" = "pipewire" ]; then
    echo "  wpctl status         - View audio devices"
    echo "  pw-top               - Monitor audio performance"
else
    echo "  pactl list sinks     - View audio outputs"
    echo "  pactl list sources   - View audio inputs"
fi
echo ""
echo "For manual control, use: bluetoothctl"