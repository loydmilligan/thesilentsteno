#!/bin/bash

# Restart script for The Silent Steno Web UI
# This script is called by the web UI to restart itself

echo "[$(date)] Restart requested" >> /tmp/silentsteno_restart.log

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Kill the current web UI process
echo "[$(date)] Killing current process..." >> /tmp/silentsteno_restart.log
pkill -f "python.*web_ui.py"

# Wait a moment for the process to fully terminate
sleep 3

# Start the web UI again
echo "[$(date)] Starting new process..." >> /tmp/silentsteno_restart.log
cd "$SCRIPT_DIR"
nohup ./start_webui.sh >> /tmp/silentsteno_restart.log 2>&1 &

echo "[$(date)] Restart command issued" >> /tmp/silentsteno_restart.log