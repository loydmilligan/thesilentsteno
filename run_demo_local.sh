#!/bin/bash

# The Silent Steno - Local Demo Runner
# Run this directly on the Pi console (not over SSH)

echo "Starting The Silent Steno - Local Demo"
echo "======================================"

# Set display for local console
export DISPLAY=:0

# Navigate to project directory
cd /home/mmariani/projects/thesilentsteno

# Activate virtual environment
source silentsteno_venv/bin/activate

echo "✅ Running on local Pi display"

# Kill any existing instances
pkill -f "minimal_demo_refactored.py" 2>/dev/null || true
pkill -f "minimal_demo.py" 2>/dev/null || true

echo "🚀 Starting demo..."

# Run the refactored demo
python minimal_demo_refactored.py

echo
echo "Demo finished."