#!/bin/bash

# The Silent Steno - Integrated Demo Launch Script
# This script launches the integrated demo that bridges between the walking skeleton and production architecture

echo "Starting The Silent Steno - Integrated Demo"
echo "=========================================="

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

# Set environment variables for production components
export PYTHONPATH="/home/mmariani/projects/thesilentsteno:$PYTHONPATH"

# Launch the integrated demo
echo "🚀 Starting integrated demo..."
echo "   - Will attempt to use production components"
echo "   - Falls back to skeleton components if needed"
echo "   - Same UI as walking skeleton"

python minimal_demo_integrated.py

echo "Demo finished."