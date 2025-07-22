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

@app.route('/dashboard')
def dashboard():
    """Large icon status dashboard for 7" touchscreen"""
    return render_template('status_dashboard.html')

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
            
            # Create new session entry
            session_id = recording_state['current_session_id']
            
            # Reset current session ID for next recording
            recording_state['current_session_id'] = None
            sessions = load_sessions()
            
            # Create new session data
            new_session = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'duration': info.get('duration', 0),
                'title': f'Recording {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                'participants': ['Unknown'],
                'wav_file': info.get('file_path', ''),
                'transcript': [],
                'summary': 'Processing...',
                'action_items': [],
                'key_moments': []
            }
            
            sessions.append(new_session)
            save_sessions(sessions)
            
            # Start transcription
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

@app.route('/api/status')
def get_system_status():
    """Get system status including Bluetooth connections"""
    try:
        import subprocess
        
        # Check for Bluetooth audio sources (phone)
        source_connected = False
        try:
            result = subprocess.run(['pactl', 'list', 'sources', 'short'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Look for Bluetooth source in the output
                for line in result.stdout.split('\n'):
                    if 'bluetooth' in line.lower() and ('pixel' in line.lower() or 'phone' in line.lower() or 'a2dp_source' in line.lower()):
                        source_connected = True
                        break
        except Exception as e:
            logger.warning(f"Error checking Bluetooth sources: {e}")
        
        # Check for Bluetooth audio sinks (earbuds)
        sink_connected = False
        try:
            result = subprocess.run(['pactl', 'list', 'sinks', 'short'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Look for Bluetooth sink in the output
                for line in result.stdout.split('\n'):
                    if 'bluetooth' in line.lower() and ('galaxy' in line.lower() or 'buds' in line.lower() or 'a2dp_sink' in line.lower()):
                        sink_connected = True
                        break
        except Exception as e:
            logger.warning(f"Error checking Bluetooth sinks: {e}")
        
        return jsonify({
            'bluetooth_source_connected': source_connected,
            'bluetooth_sink_connected': sink_connected,
            'recording': recording_state['is_recording']
        })
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/bluetooth/connect_source', methods=['POST'])
def connect_bluetooth_source():
    """Connect to Bluetooth source (phone)"""
    try:
        import subprocess
        
        # Try to connect to known phone devices
        known_phones = ['Pixel_9_Pro']  # Add your phone's Bluetooth name
        
        for phone in known_phones:
            try:
                # Check if device is paired
                result = subprocess.run(['bluetoothctl', 'info', phone], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    # Try to connect
                    connect_result = subprocess.run(['bluetoothctl', 'connect', phone], 
                                                  capture_output=True, text=True, timeout=15)
                    if connect_result.returncode == 0:
                        return jsonify({'connected': True, 'device': phone})
            except Exception as e:
                logger.warning(f"Error connecting to {phone}: {e}")
        
        return jsonify({'connected': False, 'error': 'No compatible phone found or failed to connect'})
    except Exception as e:
        logger.error(f"Error connecting Bluetooth source: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/bluetooth/connect_output', methods=['POST'])
def connect_bluetooth_output():
    """Connect to Bluetooth output (earbuds)"""
    try:
        import subprocess
        
        # Try to connect to known earbud devices
        known_earbuds = ['Galaxy_Buds3_Pro']  # Add your earbuds' Bluetooth name
        
        for earbuds in known_earbuds:
            try:
                # Check if device is paired
                result = subprocess.run(['bluetoothctl', 'info', earbuds], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    # Try to connect
                    connect_result = subprocess.run(['bluetoothctl', 'connect', earbuds], 
                                                  capture_output=True, text=True, timeout=15)
                    if connect_result.returncode == 0:
                        return jsonify({'connected': True, 'device': earbuds})
            except Exception as e:
                logger.warning(f"Error connecting to {earbuds}: {e}")
        
        return jsonify({'connected': False, 'error': 'No compatible earbuds found or failed to connect'})
    except Exception as e:
        logger.error(f"Error connecting Bluetooth output: {e}")
        return jsonify({'error': str(e)}), 500

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
            # Get transcription result which includes analysis
            result = adapter.transcribe_recording()
            if result:
                # Extract transcript text and analysis
                if isinstance(result, dict):
                    transcript = result.get('transcript', '')
                    analysis = result.get('analysis', {})
                else:
                    transcript = str(result)
                    analysis = {}
                
                # Update session with transcript and analysis
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
                        
                        # Update with AI analysis
                        session['summary'] = analysis.get('summary', transcript[:200] + '...' if len(transcript) > 200 else transcript)
                        session['action_items'] = analysis.get('action_items', [])
                        session['sentiment'] = analysis.get('sentiment', 'neutral')
                        session['topics'] = analysis.get('topics', [])
                        session['questions'] = analysis.get('questions', [])
                        session['key_moments'] = [
                            {'timestamp': '00:00:00', 'description': phrase} 
                            for phrase in analysis.get('key_phrases', [])[:3]
                        ]
                        break
                
                save_sessions(sessions)
                socketio.emit('transcription_complete', {
                    'session_id': session_id,
                    'transcript': transcript,
                    'analysis': analysis
                })
                logger.info(f"Transcription and analysis completed for session {session_id}")
            else:
                socketio.emit('transcription_error', {'session_id': session_id})
                logger.error(f"Transcription failed for session {session_id}")
    except Exception as e:
        logger.error(f"Error transcribing session {session_id}: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback: ", exc_info=True)
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