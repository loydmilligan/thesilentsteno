#!/bin/bash

# The Silent Steno - Debug Web UI Launch Script

echo "Starting The Silent Steno Web UI (Debug Mode)"
echo "============================================="

# Kill any existing instances
pkill -f "web_ui.py" 2>/dev/null || true
pkill -f "python.*5000" 2>/dev/null || true
sleep 2

# Activate virtual environment
if [ -d "silentsteno_venv" ]; then
    source silentsteno_venv/bin/activate
fi

# Set environment variables
export PYTHONPATH="/home/mmariani/projects/thesilentsteno:$PYTHONPATH"
export FLASK_ENV=development
export FLASK_DEBUG=1

# Get device IP
DEVICE_IP=$(ip route get 8.8.8.8 | grep -oP 'src \K[^ ]+')

echo ""
echo "🌐 Network Configuration:"
echo "   - Device IP: $DEVICE_IP"
echo "   - Port: 5000"
echo ""
echo "📱 Access the web UI at:"
echo "   - From this device: http://localhost:5000"
echo "   - From phone/tablet: http://$DEVICE_IP:5000"
echo ""
echo "🔍 Debug mode enabled - detailed error messages will be shown"
echo ""

# Run with explicit error output
python -u web_ui.py 2>&1 | tee web_ui_debug.log