#!/bin/bash

echo "Starting The Silent Steno Web UI"
echo "================================"

# Change to project directory
cd /home/mmariani/projects/thesilentsteno

# Kill any existing web UI processes
echo "Stopping any existing processes..."
pkill -f "web_ui.py" 2>/dev/null || true
sleep 2

# Check if virtual environment exists and activate it
if [ -d "silentsteno_venv" ]; then
    echo "Activating virtual environment..."
    source silentsteno_venv/bin/activate
else
    echo "⚠️  Virtual environment not found"
fi

# Set Python path
export PYTHONPATH="/home/mmariani/projects/thesilentsteno:$PYTHONPATH"

# Show network info
echo ""
echo "📡 Network Information:"
ip addr | grep "inet " | grep -v "127.0.0.1" | awk '{print "   IP: " $2}'
echo ""
echo "🌐 The web UI will be available at:"
echo "   - http://localhost:5000"
echo "   - http://127.0.0.1:5000"
echo ""

# Start the web UI
echo "🚀 Starting web server on port 5000..."
python3 web_ui.py