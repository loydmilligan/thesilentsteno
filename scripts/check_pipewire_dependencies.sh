#!/bin/bash

# The Silent Steno - PipeWire Dependency Checker
# This script checks if PipeWire can be safely installed

echo "🔍 The Silent Steno - PipeWire Dependency Check"
echo "=============================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check distribution
echo "1. Checking Linux distribution..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "   Distribution: $NAME $VERSION"
    
    # Check if Debian-based
    if [[ "$ID" == "debian" ]] || [[ "$ID" == "ubuntu" ]] || [[ "$ID" == "raspbian" ]]; then
        echo -e "   ${GREEN}✅ Debian-based system detected${NC}"
    else
        echo -e "   ${YELLOW}⚠️  Non-Debian system. Package names might differ${NC}"
    fi
else
    echo -e "   ${YELLOW}⚠️  Cannot determine distribution${NC}"
fi

# Check kernel version
echo ""
echo "2. Checking kernel version..."
KERNEL_VERSION=$(uname -r)
echo "   Kernel: $KERNEL_VERSION"
KERNEL_MAJOR=$(echo $KERNEL_VERSION | cut -d. -f1)
KERNEL_MINOR=$(echo $KERNEL_VERSION | cut -d. -f2)

if [ "$KERNEL_MAJOR" -ge 5 ] && [ "$KERNEL_MINOR" -ge 10 ]; then
    echo -e "   ${GREEN}✅ Kernel version supports PipeWire well${NC}"
else
    echo -e "   ${YELLOW}⚠️  Older kernel detected. PipeWire should work but might have limitations${NC}"
fi

# Check current audio server
echo ""
echo "3. Checking current audio server..."
if systemctl --user is-active --quiet pulseaudio; then
    echo -e "   ${GREEN}✅ PulseAudio is running${NC}"
    echo "   Note: PipeWire will replace PulseAudio but provides compatibility"
elif systemctl --user is-active --quiet pipewire; then
    echo -e "   ${YELLOW}⚠️  PipeWire is already running!${NC}"
    echo "   You may already have PipeWire installed"
else
    echo "   No audio server detected running as user service"
fi

# Check if PipeWire is already installed
echo ""
echo "4. Checking for existing PipeWire installation..."
if command -v pipewire &> /dev/null; then
    PIPEWIRE_VERSION=$(pipewire --version | awk '{print $3}')
    echo -e "   ${YELLOW}⚠️  PipeWire $PIPEWIRE_VERSION is already installed${NC}"
else
    echo -e "   ${GREEN}✅ PipeWire not currently installed${NC}"
fi

# Check available packages
echo ""
echo "5. Checking available PipeWire packages..."
REQUIRED_PACKAGES=(
    "pipewire"
    "pipewire-pulse"
    "pipewire-alsa"
    "wireplumber"
    "libspa-0.2-bluetooth"
    "pipewire-audio-client-libraries"
)

AVAILABLE_COUNT=0
NOT_AVAILABLE=()

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if apt-cache show "$pkg" &> /dev/null; then
        ((AVAILABLE_COUNT++))
    else
        NOT_AVAILABLE+=("$pkg")
    fi
done

echo "   Available packages: $AVAILABLE_COUNT/${#REQUIRED_PACKAGES[@]}"
if [ ${#NOT_AVAILABLE[@]} -gt 0 ]; then
    echo -e "   ${YELLOW}⚠️  Missing packages: ${NOT_AVAILABLE[*]}${NC}"
    echo "   You may need to update your package sources"
fi

# Check Bluetooth support
echo ""
echo "6. Checking Bluetooth support..."
if systemctl is-active --quiet bluetooth; then
    echo -e "   ${GREEN}✅ Bluetooth service is running${NC}"
else
    echo -e "   ${RED}❌ Bluetooth service not running${NC}"
fi

if [ -d /sys/class/bluetooth ]; then
    BT_ADAPTERS=$(ls /sys/class/bluetooth | grep -c hci)
    if [ "$BT_ADAPTERS" -gt 0 ]; then
        echo -e "   ${GREEN}✅ Found $BT_ADAPTERS Bluetooth adapter(s)${NC}"
    fi
fi

# Check disk space
echo ""
echo "7. Checking disk space..."
AVAILABLE_SPACE=$(df -BM /usr | tail -1 | awk '{print $4}' | sed 's/M//')
if [ "$AVAILABLE_SPACE" -gt 500 ]; then
    echo -e "   ${GREEN}✅ Sufficient disk space (${AVAILABLE_SPACE}MB available)${NC}"
else
    echo -e "   ${YELLOW}⚠️  Low disk space (${AVAILABLE_SPACE}MB available)${NC}"
fi

# Summary and recommendations
echo ""
echo "========================================"
echo "SUMMARY & RECOMMENDATIONS"
echo "========================================"

INSTALL_READY=true

if [ ${#NOT_AVAILABLE[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Some packages not available. Run:${NC}"
    echo "   sudo apt update"
    INSTALL_READY=false
fi

if command -v pipewire &> /dev/null; then
    echo -e "${YELLOW}⚠️  PipeWire already installed. Migration might be simpler.${NC}"
fi

if [ "$INSTALL_READY" = true ]; then
    echo -e "${GREEN}✅ System appears ready for PipeWire installation${NC}"
    echo ""
    echo "To install PipeWire, you can run:"
    echo "   sudo apt update"
    echo "   sudo apt install pipewire pipewire-pulse pipewire-alsa \\"
    echo "                    wireplumber libspa-0.2-bluetooth \\"
    echo "                    pipewire-audio-client-libraries"
else
    echo -e "${YELLOW}⚠️  Please address the issues above before installing${NC}"
fi

echo ""
echo "For the actual installation, run:"
echo "   ./scripts/install_pipewire.sh"