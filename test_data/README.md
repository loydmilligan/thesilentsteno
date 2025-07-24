# Test Data

This directory contains curated test audio files and metadata for testing transcription accuracy and AI analysis quality.

## Structure

- `audio/` - WAV audio files organized by call type
- `metadata/` - JSON files containing ground truth data and expected results

## Categories

- **meetings/** - Multi-participant meeting recordings
- **sales_calls/** - Sales conversation recordings  
- **personal_calls/** - Personal conversation recordings

## Metadata Format

Each JSON file should contain:
- Transcript ground truth
- Speaker information
- Expected topics/themes
- Key moments/action items
- Quality metrics for testing

## Usage

These files are used for:
- Transcription accuracy testing
- AI analysis quality validation
- Performance benchmarking
- Regression testing