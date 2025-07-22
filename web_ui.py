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
from src.config.settings_manager import get_settings_manager

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

# Background transcription processor state
transcription_processor_state = {
    'running': False,
    'current_session': None,
    'queue': [],
    'last_check': 0
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

@app.route('/settings')
def settings():
    """Settings configuration page"""
    return render_template('settings.html')

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
        
        # Known device configurations (from simple_audio_test.py)
        known_devices = {
            'source': {
                'name': 'Pixel 9 Pro',
                'mac': 'C0:1C:6A:AD:78:E6',
                'pa_source': 'bluez_source.C0_1C_6A_AD_78_E6.a2dp_source'
            },
            'sink': {
                'name': 'Galaxy Buds3 Pro',
                'mac': 'BC:A0:80:EB:21:AA',
                'pa_sink': 'bluez_sink.BC_A0_80_EB_21_AA.a2dp_sink'
            }
        }
        
        # Check for specific phone source
        source_connected = False
        try:
            result = subprocess.run(['pactl', 'list', 'sources', 'short'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                source_connected = known_devices['source']['pa_source'] in result.stdout
        except Exception as e:
            logger.warning(f"Error checking Bluetooth sources: {e}")
        
        # Check for specific earbuds sink
        sink_connected = False
        try:
            result = subprocess.run(['pactl', 'list', 'sinks', 'short'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                sink_connected = known_devices['sink']['pa_sink'] in result.stdout
        except Exception as e:
            logger.warning(f"Error checking Bluetooth sinks: {e}")
        
        # Check if audio forwarding is active
        audio_forwarding_active = False
        try:
            result = subprocess.run(['pactl', 'list', 'modules', 'short'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                audio_forwarding_active = 'module-loopback' in result.stdout
        except Exception as e:
            logger.warning(f"Error checking audio forwarding: {e}")
        
        return jsonify({
            'bluetooth_source_connected': source_connected,
            'bluetooth_sink_connected': sink_connected,
            'audio_forwarding_active': audio_forwarding_active,
            'recording': recording_state['is_recording'],
            'devices': known_devices
        })
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/bluetooth/connect_source', methods=['POST'])
def connect_bluetooth_source():
    """Start audio forwarding from phone to earbuds"""
    try:
        import subprocess
        
        # Known device configurations
        source_device = {
            'name': 'Pixel 9 Pro',
            'mac': 'C0:1C:6A:AD:78:E6',
            'pa_source': 'bluez_source.C0_1C_6A_AD_78_E6.a2dp_source'
        }
        
        sink_device = {
            'name': 'Galaxy Buds3 Pro',
            'mac': 'BC:A0:80:EB:21:AA',
            'pa_sink': 'bluez_sink.BC_A0_80_EB_21_AA.a2dp_sink'
        }
        
        # Check if both devices are available
        source_result = subprocess.run(['pactl', 'list', 'sources', 'short'], 
                                     capture_output=True, text=True, timeout=5)
        sink_result = subprocess.run(['pactl', 'list', 'sinks', 'short'], 
                                   capture_output=True, text=True, timeout=5)
        
        if source_device['pa_source'] not in source_result.stdout:
            return jsonify({'connected': False, 'error': f"{source_device['name']} not available as audio source"})
            
        if sink_device['pa_sink'] not in sink_result.stdout:
            return jsonify({'connected': False, 'error': f"{sink_device['name']} not available as audio output"})
        
        # Stop any existing loopback modules
        try:
            modules_result = subprocess.run("pactl list modules short | grep module-loopback", 
                                          shell=True, capture_output=True, text=True)
            
            for line in modules_result.stdout.strip().split('\n'):
                if line.strip():
                    module_id = line.split()[0]
                    subprocess.run(f"pactl unload-module {module_id}", shell=True, check=False)
                    logger.info(f"Stopped existing loopback module {module_id}")
        except Exception as e:
            logger.warning(f"Error stopping existing modules: {e}")
        
        # Create new loopback
        cmd = f"pactl load-module module-loopback source={source_device['pa_source']} sink={sink_device['pa_sink']} latency_msec=40"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Started audio forwarding: {source_device['name']} -> {sink_device['name']}")
            return jsonify({
                'connected': True, 
                'message': f"Audio forwarding active: {source_device['name']} → {sink_device['name']}",
                'source': source_device['name'],
                'sink': sink_device['name']
            })
        else:
            error_msg = result.stderr.strip() or "Unknown error"
            logger.error(f"Failed to start audio forwarding: {error_msg}")
            return jsonify({'connected': False, 'error': f"Failed to start audio forwarding: {error_msg}"})
            
    except Exception as e:
        logger.error(f"Error starting audio forwarding: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/bluetooth/connect_output', methods=['POST'])
def connect_bluetooth_output():
    """Check and ensure earbuds are connected and ready"""
    try:
        import subprocess
        
        # Known earbuds configuration
        sink_device = {
            'name': 'Galaxy Buds3 Pro',
            'mac': 'BC:A0:80:EB:21:AA',
            'pa_sink': 'bluez_sink.BC_A0_80_EB_21_AA.a2dp_sink'
        }
        
        # Check if earbuds are available as a sink
        result = subprocess.run(['pactl', 'list', 'sinks', 'short'], 
                              capture_output=True, text=True, timeout=5)
        
        if sink_device['pa_sink'] in result.stdout:
            # Test the connection with a brief test
            test_cmd = f"timeout 1 paplay -d {sink_device['pa_sink']} /dev/null 2>/dev/null || true"
            subprocess.run(test_cmd, shell=True)
            
            return jsonify({
                'connected': True,
                'message': f"{sink_device['name']} is connected and ready",
                'device': sink_device['name']
            })
        else:
            return jsonify({
                'connected': False, 
                'error': f"{sink_device['name']} not available as audio output"
            })
            
    except Exception as e:
        logger.error(f"Error checking output device: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/bluetooth/stop_forwarding', methods=['POST'])
def stop_bluetooth_forwarding():
    """Stop audio forwarding"""
    try:
        import subprocess
        
        # Stop all loopback modules
        modules_result = subprocess.run("pactl list modules short | grep module-loopback", 
                                      shell=True, capture_output=True, text=True)
        
        stopped_count = 0
        for line in modules_result.stdout.strip().split('\n'):
            if line.strip():
                module_id = line.split()[0]
                result = subprocess.run(f"pactl unload-module {module_id}", 
                                      shell=True, check=False)
                if result.returncode == 0:
                    stopped_count += 1
                    logger.info(f"Stopped loopback module {module_id}")
        
        if stopped_count > 0:
            return jsonify({
                'success': True,
                'message': f"Stopped {stopped_count} audio forwarding module(s)"
            })
        else:
            return jsonify({
                'success': True,
                'message': "No audio forwarding was active"
            })
            
    except Exception as e:
        logger.error(f"Error stopping audio forwarding: {e}")
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

@app.route('/api/settings')
def get_settings():
    """Get current application settings"""
    try:
        settings_manager = get_settings_manager()
        return jsonify(settings_manager.get_all_settings())
    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update application settings"""
    try:
        settings_manager = get_settings_manager()
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No settings data provided'}), 400
        
        # Update each category
        success = True
        errors = []
        
        for category, settings in data.items():
            if not settings_manager.update_settings(category, settings):
                success = False
                errors.append(f"Failed to update {category} settings")
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Settings updated successfully',
                'settings': settings_manager.get_all_settings()
            })
        else:
            return jsonify({
                'success': False,
                'errors': errors
            }), 500
            
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings/reset', methods=['POST'])
def reset_settings():
    """Reset settings to defaults"""
    try:
        settings_manager = get_settings_manager()
        
        if settings_manager.reset_to_defaults():
            return jsonify({
                'success': True,
                'message': 'Settings reset to defaults',
                'settings': settings_manager.get_all_settings()
            })
        else:
            return jsonify({'error': 'Failed to reset settings'}), 500
            
    except Exception as e:
        logger.error(f"Error resetting settings: {e}")
        return jsonify({'error': str(e)}), 500

def update_recording_timer():
    """Update recording timer via WebSocket"""
    while recording_state['is_recording']:
        if recording_state['start_time']:
            recording_state['duration'] = time.time() - recording_state['start_time']
            socketio.emit('recording_timer', {'duration': recording_state['duration']})
        time.sleep(1)

def get_untranscribed_sessions() -> List[Dict]:
    """Get sessions that need transcription, sorted by newest first"""
    sessions = load_sessions()
    untranscribed = []
    
    for session in sessions:
        # Check if session needs transcription
        transcript = session.get('transcript', [])
        summary = session.get('summary', '')
        wav_file = session.get('wav_file', '')
        
        # Skip if no WAV file
        if not wav_file or not os.path.exists(wav_file):
            continue
            
        # Session needs transcription if:
        # 1. Empty transcript array, OR
        # 2. Summary is "Processing...", OR  
        # 3. Transcript contains error messages
        needs_transcription = (
            len(transcript) == 0 or 
            summary == "Processing..." or
            (len(transcript) > 0 and 'error' in transcript[0].get('text', '').lower())
        )
        
        if needs_transcription:
            untranscribed.append(session)
    
    # Sort by timestamp (newest first)
    untranscribed.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    logger.info(f"Found {len(untranscribed)} sessions needing transcription")
    
    return untranscribed

def background_transcription_processor():
    """Background thread that continuously processes untranscribed sessions"""
    global transcription_processor_state
    
    logger.info("Background transcription processor started")
    transcription_processor_state['running'] = True
    
    while transcription_processor_state['running']:
        try:
            current_time = time.time()
            
            # Check for new untranscribed sessions every 30 seconds
            if current_time - transcription_processor_state['last_check'] >= 30:
                transcription_processor_state['last_check'] = current_time
                
                # Get sessions needing transcription
                untranscribed = get_untranscribed_sessions()
                
                if untranscribed:
                    logger.info(f"Background processor found {len(untranscribed)} sessions to process")
                    
                    for session in untranscribed:
                        # Skip if currently recording (prioritize live session)
                        if recording_state['is_recording']:
                            logger.info("Recording in progress, skipping background transcription")
                            break
                            
                        # Skip if already processing this session
                        if transcription_processor_state['current_session'] == session['session_id']:
                            continue
                            
                        # Process this session
                        session_id = session['session_id']
                        logger.info(f"Background processor starting transcription for session: {session_id}")
                        
                        transcription_processor_state['current_session'] = session_id
                        
                        # Transcribe in foreground (this thread will handle it)
                        transcribe_recording_sync(session_id)
                        
                        transcription_processor_state['current_session'] = None
                        
                        # Small delay between sessions
                        time.sleep(5)
                        
                        # Check if we should stop
                        if not transcription_processor_state['running']:
                            break
            
            # Sleep for 10 seconds before next check
            time.sleep(10)
            
        except Exception as e:
            logger.error(f"Error in background transcription processor: {e}")
            time.sleep(30)  # Wait longer on error
    
    logger.info("Background transcription processor stopped")

def transcribe_recording_sync(session_id: str):
    """Synchronous version of transcribe_recording for background processing"""
    try:
        logger.info(f"Starting transcription for session {session_id}")
        socketio.emit('transcription_started', {'session_id': session_id})
        
        if adapter:
            # Get WAV file path from session data
            sessions = load_sessions()
            wav_file = None
            for session in sessions:
                if session.get('session_id') == session_id:
                    wav_file = session.get('wav_file')
                    break
            
            if not wav_file:
                logger.error(f"No WAV file found for session {session_id}")
                socketio.emit('transcription_error', {'session_id': session_id})
                return
            
            if not os.path.exists(wav_file):
                logger.error(f"WAV file not found: {wav_file}")
                socketio.emit('transcription_error', {'session_id': session_id})
                return
            
            logger.info(f"Transcribing file: {wav_file}")
            
            # Get transcription result which includes analysis
            result = adapter.transcribe_recording(wav_file)
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
                logger.info(f"Background transcription completed for session {session_id}")
            else:
                socketio.emit('transcription_error', {'session_id': session_id})
                logger.error(f"Transcription failed for session {session_id}")
    except Exception as e:
        logger.error(f"Error transcribing session {session_id}: {e}")
        socketio.emit('transcription_error', {'session_id': session_id})

def transcribe_recording(session_id: str):
    """Transcribe recording in background"""
    try:
        logger.info(f"Starting transcription for session {session_id}")
        socketio.emit('transcription_started', {'session_id': session_id})
        
        if adapter:
            # Get WAV file path from session data
            sessions = load_sessions()
            wav_file = None
            for session in sessions:
                if session.get('session_id') == session_id:
                    wav_file = session.get('wav_file')
                    break
            
            if not wav_file:
                logger.error(f"No WAV file found for session {session_id}")
                socketio.emit('transcription_error', {'session_id': session_id})
                return
            
            # Get transcription result which includes analysis
            result = adapter.transcribe_recording(wav_file)
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
    
    # Start background transcription processor
    logger.info("Starting background transcription processor...")
    transcription_thread = threading.Thread(target=background_transcription_processor, daemon=True)
    transcription_thread.start()
    
    # Setup graceful shutdown
    import signal
    def signal_handler(sig, frame):
        logger.info("Shutting down gracefully...")
        transcription_processor_state['running'] = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the Flask app
    logger.info("Starting Silent Steno Web UI server...")
    logger.info("Access the app at: http://localhost:5000")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        transcription_processor_state['running'] = False