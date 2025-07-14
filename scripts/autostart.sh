#!/bin/bash

# Auto-start Script for The Silent Steno
# Configures system to auto-boot to main application

set -e

# Project directory
PROJECT_DIR="/home/mmariani/projects/thesilentsteno"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/autostart.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
handle_error() {
    log "ERROR: Application startup failed at line $1"
    log "Check log file: $LOG_FILE"
    exit 1
}

trap 'handle_error $LINENO' ERR

log "Starting The Silent Steno auto-boot sequence..."

# Wait for system services to be ready
log "Waiting for system services..."
sleep 5

# Check if Bluetooth is ready
if ! systemctl is-active --quiet bluetooth; then
    log "Starting Bluetooth service..."
    sudo systemctl start bluetooth
    sleep 2
fi

# Check if audio system is ready
if ! pgrep -x "pulseaudio" > /dev/null; then
    log "Starting PulseAudio..."
    pulseaudio --start --log-target=file:$LOG_DIR/pulseaudio.log
    sleep 2
fi

# Set display for GUI applications
export DISPLAY=:0
export XAUTHORITY=/home/mmariani/.Xauthority

# Change to project directory
cd "$PROJECT_DIR"

# Application startup placeholder
log "Application startup placeholder - main application not yet implemented"
log "When ready, this will start the main Silent Steno application"

# For now, just indicate readiness
log "Hardware platform ready for development"
log "Auto-boot configuration successful"

# Keep the service running (remove when actual app is implemented)
log "Entering standby mode..."
while true; do
    sleep 60
    log "System running - standby mode ($(date))"
done