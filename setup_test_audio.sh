#!/bin/bash

# Setup script for Silent Steno test audio generation and benchmarking

echo "================================================"
echo "Silent Steno Test Audio Setup"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "web_ui.py" ]; then
    echo "Error: Please run this script from the Silent Steno project root"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d "silentsteno_venv" ]; then
    echo "Activating virtual environment..."
    source silentsteno_venv/bin/activate
fi

# Install required dependencies for test audio generation
echo ""
echo "Installing test audio dependencies..."
pip install gtts pydub nltk

# Create test audio directory
mkdir -p test_audio

# Make scripts executable
chmod +x generate_test_audio.py
chmod +x test_audio_benchmark.py

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "To generate test audio files:"
echo "  python generate_test_audio.py"
echo ""
echo "To run transcription benchmarks:"
echo "  python test_audio_benchmark.py"
echo ""
echo "To benchmark a single file:"
echo "  python test_audio_benchmark.py --single-file test_audio/test_meeting_5min.wav"
echo ""