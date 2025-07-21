#!/bin/bash

# The Silent Steno - Bluetooth Setup Script
# This script helps set up the Pi as a Bluetooth audio proxy

echo "🎧 The Silent Steno - Bluetooth Audio Setup"
echo "=========================================="
echo ""
echo "This will configure your Pi to:"
echo "1. Receive audio from your phone (A2DP Sink)"
echo "2. Forward audio to your headphones (A2DP Source)"
echo "3. Record and transcribe in real-time"
echo ""

# Check if running as sudo for system commands
if [ "$EUID" -eq 0 ]; then 
   echo "✅ Running with sudo privileges"
else
   echo "⚠️  Some commands may require sudo password"
fi

# Function to check command status
check_status() {
    if [ $? -eq 0 ]; then
        echo "✅ $1"
    else
        echo "❌ $1 failed"
        return 1
    fi
}

# 1. Check Bluetooth service
echo ""
echo "1. Checking Bluetooth service..."
sudo systemctl status bluetooth --no-pager | grep "Active: active"
check_status "Bluetooth service is running"

# 2. Install required packages
echo ""
echo "2. Checking required packages..."
packages="bluez bluez-tools pulseaudio-module-bluetooth python3-dbus"
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
    echo "✅ All required packages installed"
fi

# 3. Enable A2DP Sink and Source
echo ""
echo "3. Configuring Bluetooth profiles..."

# Create BlueALSA configuration if needed
if [ ! -f /etc/bluetooth/audio.conf ]; then
    echo "Creating Bluetooth audio configuration..."
    sudo tee /etc/bluetooth/audio.conf > /dev/null << EOF
[General]
Enable=Source,Sink,Headset,Gateway,Control,Media
Disable=Socket

[A2DP]
SBCSources=1
MPEG12Sources=0

[AVRCP]
InputDevice=yes
EOF
    check_status "Bluetooth audio configuration created"
fi

# 4. Configure PulseAudio for Bluetooth
echo ""
echo "4. Configuring PulseAudio..."

# Load Bluetooth modules
pactl list modules | grep -q module-bluetooth-discover
if [ $? -ne 0 ]; then
    pactl load-module module-bluetooth-discover
    check_status "Loaded PulseAudio Bluetooth module"
else
    echo "✅ PulseAudio Bluetooth module already loaded"
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

# 8. Interactive pairing
echo ""
echo "8. Device Pairing"
echo "-----------------"
echo "Would you like to pair a device now? (y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Instructions:"
    echo "1. Put your phone/headphones in pairing mode"
    echo "2. Wait for the device to appear in the scan"
    echo "3. Note the MAC address (XX:XX:XX:XX:XX:XX)"
    echo ""
    echo "Press Enter to start scanning..."
    read -r

    # Start scanning
    echo "Scanning for devices (15 seconds)..."
    timeout 15 bluetoothctl scan on
    
    echo ""
    echo "Enter the MAC address of the device to pair (or 'skip' to continue):"
    read -r mac_address
    
    if [[ "$mac_address" != "skip" ]] && [[ -n "$mac_address" ]]; then
        echo "Pairing with $mac_address..."
        bluetoothctl pair "$mac_address"
        bluetoothctl trust "$mac_address"
        check_status "Device paired and trusted"
    fi
fi

# 9. Test audio routing
echo ""
echo "9. Audio Routing Setup"
echo "---------------------"
echo "To test audio routing:"
echo "1. Connect your phone as audio source"
echo "2. Connect your headphones as audio sink"
echo "3. Play audio on your phone"
echo ""

# 10. Create systemd service for auto-start
echo ""
echo "10. Auto-start Configuration"
echo "---------------------------"
echo "Would you like to enable Bluetooth audio on boot? (y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    sudo tee /etc/systemd/system/silent-steno-bluetooth.service > /dev/null << EOF
[Unit]
Description=Silent Steno Bluetooth Audio Service
After=bluetooth.service sound.target

[Service]
Type=simple
ExecStart=/usr/bin/bluetoothctl power on
ExecStartPost=/usr/bin/bluetoothctl agent on
ExecStartPost=/usr/bin/bluetoothctl default-agent
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl enable silent-steno-bluetooth.service
    check_status "Auto-start service enabled"
fi

echo ""
echo "✅ Bluetooth setup complete!"
echo ""
echo "Next steps:"
echo "1. Run: python3 demo_bluetooth_pipeline.py"
echo "2. Connect your phone and headphones"
echo "3. Start recording and transcribing!"
echo ""
echo "For manual Bluetooth control, use: bluetoothctl"