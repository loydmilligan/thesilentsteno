#!/bin/bash

# Bluetooth Service Management Script for The Silent Steno
# Manages BlueZ Bluetooth service with automatic reconnection and A2DP support

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/home/mmariani/projects/thesilentsteno"
CONFIG_DIR="$PROJECT_DIR/config"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/bluetooth.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a "$LOG_FILE"
}

# Function to start Bluetooth service
start_bluetooth_service() {
    log "Starting Bluetooth service..."
    
    # Ensure Bluetooth configuration is optimized
    configure_bluetooth
    
    # Start the Bluetooth service
    if sudo systemctl start bluetooth; then
        log "Bluetooth service started successfully"
        
        # Wait for service to be ready
        sleep 3
        
        # Power on the adapter
        if power_on_adapter; then
            log "Bluetooth adapter powered on"
            
            # Configure A2DP profiles
            configure_a2dp
            
            # Enable auto-reconnection
            enable_auto_reconnect
            
            log "Bluetooth service startup complete"
            return 0
        else
            error "Failed to power on Bluetooth adapter"
            return 1
        fi
    else
        error "Failed to start Bluetooth service"
        return 1
    fi
}

# Function to stop Bluetooth service
stop_bluetooth_service() {
    log "Stopping Bluetooth service..."
    
    # Disconnect all devices gracefully
    disconnect_all_devices
    
    # Stop the service
    if sudo systemctl stop bluetooth; then
        log "Bluetooth service stopped successfully"
        return 0
    else
        error "Failed to stop Bluetooth service"
        return 1
    fi
}

# Function to restart Bluetooth service
restart_bluetooth_service() {
    log "Restarting Bluetooth service..."
    
    stop_bluetooth_service
    sleep 2
    start_bluetooth_service
}

# Function to configure Bluetooth main settings
configure_bluetooth() {
    info "Configuring Bluetooth settings..."
    
    # Backup original configuration if it exists
    if [ -f /etc/bluetooth/main.conf ] && [ ! -f /etc/bluetooth/main.conf.backup ]; then
        sudo cp /etc/bluetooth/main.conf /etc/bluetooth/main.conf.backup
        log "Backed up original main.conf"
    fi
    
    # Apply our configuration template
    if [ -f "$CONFIG_DIR/bluetooth_main.conf" ]; then
        sudo cp "$CONFIG_DIR/bluetooth_main.conf" /etc/bluetooth/main.conf
        log "Applied Silent Steno Bluetooth configuration"
    else
        warn "Bluetooth configuration template not found, using defaults"
    fi
    
    # Set proper permissions
    sudo chmod 644 /etc/bluetooth/main.conf
    sudo chown root:root /etc/bluetooth/main.conf
}

# Function to power on Bluetooth adapter
power_on_adapter() {
    info "Powering on Bluetooth adapter..."
    
    # Use bluetoothctl to power on
    if echo -e "power on\nquit" | bluetoothctl > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to configure A2DP profiles
configure_a2dp() {
    info "Configuring A2DP profiles..."
    
    # Load PulseAudio Bluetooth modules
    if command -v pactl >/dev/null 2>&1; then
        # Unload existing modules first
        pactl unload-module module-bluez5-device 2>/dev/null || true
        pactl unload-module module-bluez5-discover 2>/dev/null || true
        
        # Load Bluetooth discovery module
        pactl load-module module-bluez5-discover 2>/dev/null || warn "Failed to load bluez5-discover module"
        
        log "A2DP profiles configured"
    else
        warn "PulseAudio not available, A2DP configuration skipped"
    fi
}

# Function to enable auto-reconnection
enable_auto_reconnect() {
    info "Enabling auto-reconnection..."
    
    # Set adapter to be discoverable and pairable
    echo -e "discoverable on\npairable on\nquit" | bluetoothctl > /dev/null 2>&1
    
    log "Auto-reconnection enabled"
}

# Function to disconnect all devices
disconnect_all_devices() {
    info "Disconnecting all Bluetooth devices..."
    
    # Get connected devices and disconnect them
    bluetoothctl devices Connected | while read -r line; do
        if [[ $line == Device* ]]; then
            device_mac=$(echo "$line" | awk '{print $2}')
            device_name=$(echo "$line" | awk '{print $3}')
            info "Disconnecting $device_name ($device_mac)"
            echo -e "disconnect $device_mac\nquit" | bluetoothctl > /dev/null 2>&1
        fi
    done
}

# Function to check service status
check_bluetooth_status() {
    info "Checking Bluetooth service status..."
    
    # Check systemd service
    if systemctl is-active --quiet bluetooth; then
        log "✓ Bluetooth service is active"
    else
        error "✗ Bluetooth service is not active"
    fi
    
    # Check adapter status
    if bluetoothctl show | grep -q "Powered: yes"; then
        log "✓ Bluetooth adapter is powered on"
    else
        warn "✗ Bluetooth adapter is not powered on"
    fi
    
    # Check connected devices
    connected_count=$(bluetoothctl devices Connected | wc -l)
    log "Connected devices: $connected_count"
    
    if [ $connected_count -gt 0 ]; then
        bluetoothctl devices Connected | while read -r line; do
            if [[ $line == Device* ]]; then
                device_name=$(echo "$line" | awk '{print $3}')
                device_mac=$(echo "$line" | awk '{print $2}')
                log "  - $device_name ($device_mac)"
            fi
        done
    fi
    
    # Check A2DP support
    if pactl list modules | grep -q "module-bluez5"; then
        log "✓ A2DP modules loaded"
    else
        warn "✗ A2DP modules not loaded"
    fi
}

# Function to scan for devices
scan_devices() {
    info "Scanning for Bluetooth devices..."
    
    # Start discovery
    echo -e "scan on\nquit" | bluetoothctl &
    scan_pid=$!
    
    # Scan for 30 seconds
    sleep 30
    
    # Stop discovery
    echo -e "scan off\nquit" | bluetoothctl
    
    # Show discovered devices
    log "Discovered devices:"
    bluetoothctl devices | while read -r line; do
        if [[ $line == Device* ]]; then
            device_name=$(echo "$line" | awk '{print $3}')
            device_mac=$(echo "$line" | awk '{print $2}')
            log "  - $device_name ($device_mac)"
        fi
    done
}

# Function to pair with a device
pair_device() {
    local device_mac="$1"
    
    if [ -z "$device_mac" ]; then
        error "Device MAC address required for pairing"
        return 1
    fi
    
    info "Pairing with device $device_mac..."
    
    # Make device discoverable
    echo -e "discoverable on\npairable on\nquit" | bluetoothctl
    
    # Attempt pairing
    if echo -e "pair $device_mac\nquit" | bluetoothctl; then
        log "Successfully paired with $device_mac"
        
        # Trust the device for auto-reconnection
        echo -e "trust $device_mac\nquit" | bluetoothctl
        log "Device $device_mac is now trusted"
        
        return 0
    else
        error "Failed to pair with $device_mac"
        return 1
    fi
}

# Function to test A2DP connection
test_a2dp() {
    info "Testing A2DP connection..."
    
    # Check if any A2DP devices are connected
    if pactl list sources | grep -q "bluez"; then
        log "✓ A2DP source detected"
    else
        warn "✗ No A2DP source detected"
    fi
    
    if pactl list sinks | grep -q "bluez"; then
        log "✓ A2DP sink detected"
    else
        warn "✗ No A2DP sink detected"
    fi
}

# Function to show help
show_help() {
    echo "Bluetooth Service Management Script for The Silent Steno"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start          Start Bluetooth service with A2DP configuration"
    echo "  stop           Stop Bluetooth service gracefully"
    echo "  restart        Restart Bluetooth service"
    echo "  status         Check Bluetooth service and device status"
    echo "  scan           Scan for available Bluetooth devices"
    echo "  pair <MAC>     Pair with specified device (MAC address)"
    echo "  test-a2dp      Test A2DP connection functionality"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 status"
    echo "  $0 pair AA:BB:CC:DD:EE:FF"
    echo "  $0 scan"
}

# Main execution
main() {
    case "${1:-help}" in
        "start")
            start_bluetooth_service
            ;;
        "stop")
            stop_bluetooth_service
            ;;
        "restart")
            restart_bluetooth_service
            ;;
        "status")
            check_bluetooth_status
            ;;
        "scan")
            scan_devices
            ;;
        "pair")
            pair_device "$2"
            ;;
        "test-a2dp")
            test_a2dp
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Export functions for sourcing
export -f start_bluetooth_service
export -f stop_bluetooth_service
export -f restart_bluetooth_service
export -f check_bluetooth_status
export -f pair_device

# Run main if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi