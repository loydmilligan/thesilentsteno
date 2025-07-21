#!/bin/bash

# Simple launcher for Silent Steno Web UI
echo "🚀 Launching Silent Steno Web UI..."

cd /home/mmariani/projects/thesilentsteno

# Activate virtual environment if it exists
if [ -d "silentsteno_venv" ]; then
    source silentsteno_venv/bin/activate
fi

# Set Python path
export PYTHONPATH="/home/mmariani/projects/thesilentsteno:$PYTHONPATH"

# Start the web UI
python3 web_ui.py &

# Wait a moment for server to start
sleep 3

# Open browser automatically
echo "Opening browser..."
if command -v chromium-browser >/dev/null 2>&1; then
    DISPLAY=:0 chromium-browser --app=http://localhost:5000 --window-size=1024,600 &
elif command -v firefox >/dev/null 2>&1; then
    DISPLAY=:0 firefox http://localhost:5000 &
else
    echo "Please open http://localhost:5000 in your browser"
fi

echo "Web UI is running. Close this terminal to stop the server."
wait