#!/bin/bash

# Hardware Setup Script for The Silent Steno
# Automates Pi 5 hardware configuration and development tool installation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Function to install development tools
install_dev_tools() {
    log "Installing development tools..."
    
    # Update package list
    sudo apt update
    
    # Install core development packages
    local packages=(
        "python3-dev"
        "python3-pip"
        "python3-venv"
        "build-essential"
        "git"
        "vim"
        "curl"
        "wget"
        "htop"
        "tree"
    )
    
    for package in "${packages[@]}"; do
        if ! dpkg -l | grep -q "^ii.*$package "; then
            log "Installing $package..."
            sudo apt install -y "$package"
        else
            log "$package is already installed"
        fi
    done
    
    # Install Python development tools
    log "Installing Python development tools..."
    pip3 install --user --upgrade pip setuptools wheel
    
    log "Development tools installation complete"
}

# Function to configure auto-boot (placeholder for when application is ready)
configure_autoboot() {
    log "Configuring auto-boot setup..."
    
    # Create systemd service file (disabled by default)
    sudo tee /etc/systemd/system/silentsteno.service > /dev/null <<EOF
[Unit]
Description=The Silent Steno Bluetooth AI Meeting Recorder
After=bluetooth.service pulseaudio.service
Requires=bluetooth.service

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/home/mmariani/projects/thesilentsteno
ExecStart=/home/mmariani/projects/thesilentsteno/scripts/autostart.sh
Restart=always
RestartSec=5
Environment=DISPLAY=:0

[Install]
WantedBy=multi-user.target
EOF

    log "Auto-boot service configured (disabled by default)"
    log "To enable: sudo systemctl enable silentsteno.service"
}

# Function to test hardware components
test_hardware() {
    log "Testing hardware components..."
    
    # Test Python installation
    if python3 -c "import sys; print(f'Python {sys.version}')" >/dev/null 2>&1; then
        log "✓ Python installation working"
    else
        error "✗ Python installation failed"
    fi
    
    # Test git installation
    if git --version >/dev/null 2>&1; then
        log "✓ Git installation working"
    else
        error "✗ Git installation failed"
    fi
    
    # Test build tools
    if gcc --version >/dev/null 2>&1; then
        log "✓ Build tools working"
    else
        error "✗ Build tools failed"
    fi
    
    # Test Bluetooth
    if systemctl is-active --quiet bluetooth; then
        log "✓ Bluetooth service running"
        if command -v bluetoothctl >/dev/null 2>&1; then
            log "✓ Bluetooth tools available"
        else
            warn "✗ Bluetooth tools not found"
        fi
    else
        warn "✗ Bluetooth service not running"
    fi
    
    # Test audio system
    if command -v aplay >/dev/null 2>&1; then
        log "✓ ALSA audio tools available"
        # List audio devices
        log "Audio devices:"
        aplay -l | grep -E "(card|device)" || warn "No audio devices found"
    else
        warn "✗ ALSA audio tools not found"
    fi
    
    # Test display
    if [ -n "$DISPLAY" ]; then
        log "✓ Display environment set: $DISPLAY"
    else
        warn "✗ No display environment (normal for SSH)"
    fi
    
    # Test touchscreen (if available)
    if [ -e "/dev/input/event*" ]; then
        log "✓ Input devices found"
        ls /dev/input/event* | head -3
    else
        warn "✗ No input devices found"
    fi
    
    log "Hardware testing complete"
}

# Function to check system status
check_system_status() {
    log "System Status Check:"
    
    # CPU and memory info
    log "CPU: $(nproc) cores"
    log "Memory: $(free -h | awk '/^Mem:/ {print $2}')"
    log "Disk: $(df -h / | awk 'NR==2 {print $4}') available"
    
    # OS version
    log "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
    
    # Kernel version
    log "Kernel: $(uname -r)"
    
    # Uptime
    log "Uptime: $(uptime -p)"
}

# Main execution
main() {
    log "Starting hardware setup for The Silent Steno..."
    
    case "${1:-all}" in
        "dev-tools")
            install_dev_tools
            ;;
        "autoboot")
            configure_autoboot
            ;;
        "test")
            test_hardware
            ;;
        "status")
            check_system_status
            ;;
        "all")
            check_system_status
            install_dev_tools
            configure_autoboot
            test_hardware
            log "Hardware setup complete!"
            ;;
        *)
            echo "Usage: $0 [dev-tools|autoboot|test|status|all]"
            echo "  dev-tools  - Install development tools"
            echo "  autoboot   - Configure auto-boot service"
            echo "  test       - Test hardware components"
            echo "  status     - Show system status"
            echo "  all        - Run all setup steps (default)"
            exit 1
            ;;
    esac
}

# Export functions for sourcing
export -f install_dev_tools
export -f configure_autoboot
export -f test_hardware
export -f check_system_status

# Run main if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi