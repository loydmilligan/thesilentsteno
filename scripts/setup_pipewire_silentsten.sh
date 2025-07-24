#!/bin/bash

# The Silent Steno - PipeWire Setup Script
# Configures PipeWire specifically for dual A2DP Bluetooth audio forwarding

set -e

echo "🎵 The Silent Steno - PipeWire Configuration"
echo "==========================================="
echo ""
echo "This script will configure PipeWire for optimal"
echo "Bluetooth audio forwarding with minimal latency"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running as regular user (not root)
if [ "$EUID" -eq 0 ]; then 
   echo -e "${RED}❌ Please run this script as a regular user, not root${NC}"
   echo "The script will use sudo when needed"
   exit 1
fi

# Function to check command status
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $1${NC}"
    else
        echo -e "${RED}❌ $1 failed${NC}"
        return 1
    fi
}

# Check if PipeWire is installed
echo "1. Checking PipeWire installation..."
if ! command -v pipewire &> /dev/null; then
    echo -e "${RED}❌ PipeWire not installed${NC}"
    echo "Please run ./scripts/install_pipewire.sh first"
    exit 1
fi

PIPEWIRE_VERSION=$(pipewire --version | awk '{print $3}')
echo -e "${GREEN}✅ PipeWire $PIPEWIRE_VERSION installed${NC}"

# Check if PipeWire is running
echo ""
echo "2. Checking PipeWire services..."
if systemctl --user is-active --quiet pipewire; then
    echo -e "${GREEN}✅ PipeWire service is running${NC}"
else
    echo -e "${YELLOW}⚠️  Starting PipeWire service...${NC}"
    systemctl --user start pipewire
    sleep 2
fi

if systemctl --user is-active --quiet wireplumber; then
    echo -e "${GREEN}✅ WirePlumber service is running${NC}"
else
    echo -e "${YELLOW}⚠️  Starting WirePlumber service...${NC}"
    systemctl --user start wireplumber
    sleep 2
fi

# Create configuration directories
echo ""
echo "3. Creating configuration directories..."
CONFIG_DIR="$HOME/.config/pipewire"
WIREPLUMBER_DIR="$HOME/.config/wireplumber"

mkdir -p "$CONFIG_DIR/pipewire.conf.d"
mkdir -p "$WIREPLUMBER_DIR/bluetooth.lua.d"
mkdir -p "$WIREPLUMBER_DIR/main.lua.d"
check_status "Configuration directories created"

# Copy Silent Steno configurations
echo ""
echo "4. Installing Silent Steno configurations..."

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Copy PipeWire configs
if [ -d "$PROJECT_DIR/config/pipewire_config" ]; then
    cp "$PROJECT_DIR/config/pipewire_config/pipewire.conf" "$CONFIG_DIR/pipewire.conf.d/10-silentsten.conf"
    cp "$PROJECT_DIR/config/pipewire_config/pipewire-pulse.conf" "$CONFIG_DIR/pipewire-pulse.conf.d/10-silentsten.conf" 2>/dev/null || true
    check_status "PipeWire configurations installed"
else
    echo -e "${YELLOW}⚠️  Configuration files not found in project${NC}"
fi

# Create Bluetooth-specific configuration
echo ""
echo "5. Creating Bluetooth audio configuration..."

cat > "$WIREPLUMBER_DIR/bluetooth.lua.d/50-silentsten-bluetooth.lua" << 'EOF'
-- Silent Steno Bluetooth Configuration
-- Optimized for dual A2DP operation

rule = {
  matches = {
    {
      { "device.name", "matches", "bluez_card.*" },
    },
  },
  apply_properties = {
    -- Enable all codecs
    ["bluez5.enable-sbc"] = true,
    ["bluez5.enable-aac"] = true,
    ["bluez5.enable-msbc"] = true,
    ["bluez5.enable-hw-volume"] = true,
    
    -- Codec quality settings
    ["bluez5.a2dp.aac.bitratemode"] = "variable",
    ["bluez5.a2dp.ldac.quality"] = "mq",
    
    -- Auto-connect settings
    ["bluez5.auto-connect"] = "[ a2dp_sink a2dp_source ]",
    ["bluez5.hw-volume"] = "[ a2dp_sink ]",
    
    -- Reconnect profiles
    ["bluez5.reconnect-profiles"] = "[ a2dp_sink a2dp_source ]",
  },
}

table.insert(bluez_monitor.rules, rule)

-- Device role assignment
phone_rule = {
  matches = {
    {
      { "device.name", "matches", "bluez_card.*" },
      { "device.form-factor", "equals", "phone" },
    },
  },
  apply_properties = {
    ["priority.session"] = 2000,
    ["device.profile"] = "a2dp-source",
  },
}

headphone_rule = {
  matches = {
    {
      { "device.name", "matches", "bluez_card.*" },
      { "device.form-factor", "equals", "headphone" },
    },
  },
  apply_properties = {
    ["priority.session"] = 1500,
    ["device.profile"] = "a2dp-sink",
  },
}

table.insert(bluez_monitor.rules, phone_rule)
table.insert(bluez_monitor.rules, headphone_rule)
EOF
check_status "Bluetooth configuration created"

# Create low-latency configuration
echo ""
echo "6. Creating low-latency audio configuration..."

cat > "$CONFIG_DIR/pipewire.conf.d/20-lowlatency.conf" << 'EOF'
# Silent Steno Low Latency Configuration
context.properties = {
    # Target 20ms latency
    default.clock.rate          = 44100
    default.clock.allowed-rates = [ 44100 48000 ]
    default.clock.quantum       = 512
    default.clock.min-quantum   = 64
    default.clock.max-quantum   = 2048
}

stream.properties = {
    node.latency = 512/44100
    resample.quality = 4
    resample.disable = false
}
EOF
check_status "Low-latency configuration created"

# Create audio forwarding rules
echo ""
echo "7. Creating audio forwarding rules..."

cat > "$WIREPLUMBER_DIR/main.lua.d/50-silentsten-forward.lua" << 'EOF'
-- Silent Steno Audio Forwarding Rules
-- Automatically forward audio from phone to headphones

-- Load the default policy config
default_policy.policy["move"] = true
default_policy.policy["follow"] = true

-- Custom audio forwarding policy
table.insert(default_policy.policy, {
  ["matches"] = {
    {
      { "media.role", "=", "phone" },
    },
  },
  ["actions"] = {
    ["default"] = {
      ["priority"] = 100,
      ["move"] = true,
      ["follow"] = true,
    },
  },
})
EOF
check_status "Audio forwarding rules created"

# Restart PipeWire to apply changes
echo ""
echo "8. Restarting PipeWire services..."
systemctl --user restart pipewire pipewire-pulse wireplumber
sleep 3
check_status "Services restarted"

# Verify configuration
echo ""
echo "9. Verifying configuration..."

# Check if services are running
SERVICES_OK=true
for service in pipewire pipewire-pulse wireplumber; do
    if ! systemctl --user is-active --quiet $service; then
        echo -e "${RED}❌ $service is not running${NC}"
        SERVICES_OK=false
    fi
done

if [ "$SERVICES_OK" = true ]; then
    echo -e "${GREEN}✅ All services running${NC}"
fi

# Check audio status
echo ""
echo "10. Audio system status:"
echo "------------------------"
pw-cli info | head -5

# Create helper scripts
echo ""
echo "11. Creating helper scripts..."

mkdir -p "$HOME/bin"

# Audio status script
cat > "$HOME/bin/silentsten-audio-status" << 'EOF'
#!/bin/bash
echo "🎵 Silent Steno Audio Status"
echo "=========================="
echo ""
echo "PipeWire Info:"
pw-cli info | head -5
echo ""
echo "Audio Devices:"
wpctl status
echo ""
echo "Bluetooth Devices:"
bluetoothctl devices | while read -r line; do
    echo "$line"
done
EOF
chmod +x "$HOME/bin/silentsten-audio-status"

# Quick connect script
cat > "$HOME/bin/silentsten-connect" << 'EOF'
#!/bin/bash
echo "🔗 Silent Steno Quick Connect"
echo "=========================="
echo ""
echo "Available Bluetooth devices:"
bluetoothctl devices
echo ""
echo "To connect a device:"
echo "  bluetoothctl connect <MAC_ADDRESS>"
echo ""
echo "To set up audio forwarding:"
echo "  1. Connect your phone (audio source)"
echo "  2. Connect your headphones (audio sink)"
echo "  3. Audio will automatically forward"
EOF
chmod +x "$HOME/bin/silentsten-connect"

check_status "Helper scripts created"

# Final summary
echo ""
echo "========================================"
echo -e "${GREEN}✅ PIPEWIRE SETUP COMPLETE${NC}"
echo "========================================"
echo ""
echo "Silent Steno is now configured for:"
echo -e "${BLUE}• Low-latency audio forwarding (target: 20ms)${NC}"
echo -e "${BLUE}• Dual A2DP Bluetooth operation${NC}"
echo -e "${BLUE}• Automatic audio routing${NC}"
echo -e "${BLUE}• High-quality codec support${NC}"
echo ""
echo "Helper commands:"
echo "  silentsten-audio-status - Check audio system status"
echo "  silentsten-connect      - Quick Bluetooth connection guide"
echo "  wpctl status           - View all audio devices"
echo "  pw-top                 - Monitor PipeWire performance"
echo ""
echo "Next steps:"
echo "1. Connect your phone as audio source"
echo "2. Connect your headphones as audio sink"
echo "3. Test audio forwarding"
echo ""
echo -e "${YELLOW}Note: You may need to log out and back in for all changes to take effect${NC}"