#!/usr/bin/env python3
"""
The Silent Steno - Modern Web UI
Flask application providing a modern, touch-friendly interface for the Silent Steno
"""

import os
import sys
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit

# Add project root to path
sys.path.append('/home/mmariani/projects/thesilentsteno')

# Import existing components
from src.integration.walking_skeleton_adapter import create_walking_skeleton_adapter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'silent-steno-secret-key-2025'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
adapter = None
recording_state = {
    'is_recording': False,
    'current_session_id': None,
    'start_time': None,
    'duration': 0
}

def initialize_adapter():
    """Initialize the walking skeleton adapter"""
    global adapter
    try:
        adapter = create_walking_skeleton_adapter(use_production=False)
        adapter.initialize()
        logger.info("Adapter initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize adapter: {e}")
        return False

def load_sessions() -> List[Dict]:
    """Load sessions from JSON file"""
    sessions_file = "demo_sessions/sessions.json"
    try:
        if os.path.exists(sessions_file):
            with open(sessions_file, 'r') as f:
                data = json.load(f)
                return data.get('sessions', [])
        return []
    except Exception as e:
        logger.error(f"Error loading sessions: {e}")
        return []

def save_sessions(sessions: List[Dict]):
    """Save sessions to JSON file"""
    sessions_file = "demo_sessions/sessions.json"
    try:
        os.makedirs("demo_sessions", exist_ok=True)
        with open(sessions_file, 'w') as f:
            json.dump({'sessions': sessions}, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving sessions: {e}")

def format_session_for_ui(session: Dict) -> Dict:
    """Format session data for the UI"""
    return {
        'session_id': session.get('session_id', ''),
        'start_time': session.get('timestamp', ''),
        'duration': session.get('duration', 0),
        'title': session.get('title', 'Recording Session'),
        'participants': session.get('participants', ['Unknown']),
        'summary': session.get('summary', 'No summary available'),
        'action_items': session.get('action_items', []),
        'key_moments': session.get('key_moments', []),
        'transcript': session.get('transcript', []),
        'wav_file': session.get('wav_file', '')
    }

# Routes
@app.route('/')
def index():
    """Main application page"""
    return render_template('index.html')

@app.route('/api/sessions')
def get_sessions():
    """Get all sessions"""
    try:
        sessions = load_sessions()
        formatted_sessions = [format_session_for_ui(session) for session in sessions]
        return jsonify(formatted_sessions)
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<session_id>')
def get_session(session_id):
    """Get a specific session"""
    try:
        sessions = load_sessions()
        session = next((s for s in sessions if s.get('session_id') == session_id), None)
        if session:
            return jsonify(format_session_for_ui(session))
        return jsonify({'error': 'Session not found'}), 404
    except Exception as e:
        logger.error(f"Error getting session {session_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recording/start', methods=['POST'])
def start_recording():
    """Start a new recording"""
    global recording_state
    try:
        if recording_state['is_recording']:
            return jsonify({'error': 'Recording already in progress'}), 400
        
        if not adapter:
            return jsonify({'error': 'Adapter not initialized'}), 500
        
        session_id = adapter.start_recording()
        if session_id:
            recording_state = {
                'is_recording': True,
                'current_session_id': session_id,
                'start_time': time.time(),
                'duration': 0
            }
            
            # Start timer thread
            threading.Thread(target=update_recording_timer, daemon=True).start()
            
            socketio.emit('recording_started', {'session_id': session_id})
            return jsonify({'session_id': session_id})
        else:
            return jsonify({'error': 'Failed to start recording'}), 500
    except Exception as e:
        logger.error(f"Error starting recording: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recording/stop', methods=['POST'])
def stop_recording():
    """Stop the current recording"""
    global recording_state
    try:
        if not recording_state['is_recording']:
            return jsonify({'error': 'No recording in progress'}), 400
        
        if not adapter:
            return jsonify({'error': 'Adapter not initialized'}), 500
        
        info = adapter.stop_recording()
        if info:
            recording_state['is_recording'] = False
            
            # Start transcription
            session_id = recording_state['current_session_id']
            threading.Thread(target=transcribe_recording, args=(session_id,), daemon=True).start()
            
            socketio.emit('recording_stopped', {'session_id': session_id, 'info': info})
            return jsonify({'session_id': session_id, 'info': info})
        else:
            return jsonify({'error': 'Failed to stop recording'}), 500
    except Exception as e:
        logger.error(f"Error stopping recording: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recording/status')
def get_recording_status():
    """Get current recording status"""
    return jsonify(recording_state)

@app.route('/api/play/<session_id>')
def play_session(session_id):
    """Play a session's audio file"""
    try:
        sessions = load_sessions()
        session = next((s for s in sessions if s.get('session_id') == session_id), None)
        if session and session.get('wav_file'):
            wav_file = session['wav_file']
            if os.path.exists(wav_file):
                return send_file(wav_file, as_attachment=False)
        return jsonify({'error': 'Audio file not found'}), 404
    except Exception as e:
        logger.error(f"Error playing session {session_id}: {e}")
        return jsonify({'error': str(e)}), 500

def update_recording_timer():
    """Update recording timer via WebSocket"""
    while recording_state['is_recording']:
        if recording_state['start_time']:
            recording_state['duration'] = time.time() - recording_state['start_time']
            socketio.emit('recording_timer', {'duration': recording_state['duration']})
        time.sleep(1)

def transcribe_recording(session_id: str):
    """Transcribe recording in background"""
    try:
        logger.info(f"Starting transcription for session {session_id}")
        socketio.emit('transcription_started', {'session_id': session_id})
        
        if adapter:
            transcript = adapter.transcribe_recording()
            if transcript:
                # Update session with transcript
                sessions = load_sessions()
                for session in sessions:
                    if session.get('session_id') == session_id:
                        session['transcript'] = [
                            {
                                'speaker': 'Unknown',
                                'timestamp': '00:00:00',
                                'text': transcript
                            }
                        ]
                        session['summary'] = transcript[:200] + '...' if len(transcript) > 200 else transcript
                        break
                
                save_sessions(sessions)
                socketio.emit('transcription_complete', {
                    'session_id': session_id,
                    'transcript': transcript
                })
                logger.info(f"Transcription completed for session {session_id}")
            else:
                socketio.emit('transcription_error', {'session_id': session_id})
                logger.error(f"Transcription failed for session {session_id}")
    except Exception as e:
        logger.error(f"Error transcribing session {session_id}: {e}")
        socketio.emit('transcription_error', {'session_id': session_id, 'error': str(e)})

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected")
    emit('connected', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected")

if __name__ == '__main__':
    # Initialize adapter
    if not initialize_adapter():
        logger.error("Failed to initialize adapter, exiting")
        sys.exit(1)
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the Flask app
    logger.info("Starting Silent Steno Web UI server...")
    logger.info("Access the app at: http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)