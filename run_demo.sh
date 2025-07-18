#!/bin/bash

# The Silent Steno - Demo Runner Script
# Activates virtual environment and runs the demo

echo "Starting The Silent Steno - Refactored Demo"
echo "============================================"

# Check if virtual environment exists
if [ ! -d "silentsteno_venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python -m venv silentsteno_venv"
    echo "Then run: source silentsteno_venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source silentsteno_venv/bin/activate

# Check if dependencies are installed
if ! python -c "import kivy, whisper, sounddevice" 2>/dev/null; then
    echo "❌ Dependencies not found!"
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
fi

echo "✅ Virtual environment activated"
echo "✅ Dependencies available"
echo "🚀 Starting refactored demo..."
echo

# Run the refactored demo
python minimal_demo_refactored.py

echo
echo "Demo finished."