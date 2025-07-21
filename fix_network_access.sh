#!/bin/bash

# The Silent Steno - Fix Network Access for Web UI

echo "Fixing network access for Silent Steno Web UI..."
echo "=============================================="

# Kill any existing web UI processes
echo "1. Stopping any existing web UI processes..."
pkill -f "web_ui.py" 2>/dev/null || true
pkill -f "python.*5000" 2>/dev/null || true

# Wait a moment for ports to be released
sleep 2

# Check if port 5000 is available
echo "2. Checking if port 5000 is available..."
if lsof -i :5000 >/dev/null 2>&1; then
    echo "   ❌ Port 5000 is still in use!"
    lsof -i :5000
    exit 1
else
    echo "   ✅ Port 5000 is available"
fi

# Try to open the port if iptables exists
echo "3. Attempting to open firewall port (if applicable)..."
if command -v iptables >/dev/null 2>&1; then
    sudo iptables -A INPUT -p tcp --dport 5000 -j ACCEPT 2>/dev/null || echo "   ⚠️  Could not modify iptables (might already be open)"
else
    echo "   ℹ️  No iptables found"
fi

# Check network interfaces
echo "4. Network interfaces:"
ip addr | grep "inet " | grep -v "127.0.0.1" | awk '{print "   - " $2}'

# Get the actual IP address
DEVICE_IP=$(ip route get 8.8.8.8 | grep -oP 'src \K[^ ]+')
echo ""
echo "5. Your device IP: $DEVICE_IP"
echo ""

# Start the web UI with explicit binding
echo "6. Starting web UI server..."
echo "   The server will be accessible at:"
echo "   - From this device: http://localhost:5000"
echo "   - From other devices: http://$DEVICE_IP:5000"
echo ""

# Export environment variables
export PYTHONPATH="/home/mmariani/projects/thesilentsteno:$PYTHONPATH"
export FLASK_ENV=production

# Activate virtual environment if it exists
if [ -d "silentsteno_venv" ]; then
    source silentsteno_venv/bin/activate
fi

# Run the web UI
python web_ui.py