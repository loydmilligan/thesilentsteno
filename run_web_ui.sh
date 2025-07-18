#!/bin/bash

# The Silent Steno - Modern Web UI Launch Script
# This script launches the modern web-based interface

echo "Starting The Silent Steno - Modern Web UI"
echo "========================================="

# Check if we're running on a Pi with display
if [[ "$DISPLAY" == *":"* ]]; then
    echo "✅ Running on Pi with display"
else
    echo "❌ No display detected - setting up virtual display"
    export DISPLAY=:0
fi

# Activate virtual environment
if [ -d "silentsteno_venv" ]; then
    echo "🔄 Activating virtual environment..."
    source silentsteno_venv/bin/activate
else
    echo "❌ Virtual environment not found"
    echo "Please run: python -m venv silentsteno_venv && source silentsteno_venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if database is initialized
if [ ! -f "data/silent_steno.db" ]; then
    echo "🔄 Initializing database..."
    python init_database.py
fi

# Set environment variables
export PYTHONPATH="/home/mmariani/projects/thesilentsteno:$PYTHONPATH"
export FLASK_ENV=production

# Kill any existing instances
pkill -f "web_ui.py" 2>/dev/null || true

# Launch the web UI
echo "🚀 Starting modern web UI..."
echo "   - Server will be available at: http://localhost:5000"
echo "   - Touch-optimized interface"
echo "   - Real-time recording and transcription"
echo "   - AI-powered analysis"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python web_ui.py