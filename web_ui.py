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
    'last_check': 0,
    'last_dir_check': 0,
    'sessions_dir_mtime': 0
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

@app.route('/api/bluetooth/devices')
def get_bluetooth_devices():
    """Get list of all paired Bluetooth devices with three-state status detection"""
    try:
        import subprocess
        
        devices = {
            'sources': [],
            'outputs': []
        }
        
        # Get paired devices from bluetoothctl
        paired_result = subprocess.run(['bluetoothctl', 'devices'], 
                                     capture_output=True, text=True, timeout=5)
        
        if paired_result.returncode != 0:
            logger.error(f"Failed to get paired devices: {paired_result.stderr}")
            return jsonify({'error': 'Failed to get Bluetooth devices'}), 500
        
        # Get PulseAudio sources and sinks with detailed status
        sources_result = subprocess.run(['pactl', 'list', 'sources'], 
                                      capture_output=True, text=True, timeout=5)
        sinks_result = subprocess.run(['pactl', 'list', 'sinks'], 
                                    capture_output=True, text=True, timeout=5)
        
        pa_sources_detailed = sources_result.stdout if sources_result.returncode == 0 else ""
        pa_sinks_detailed = sinks_result.stdout if sinks_result.returncode == 0 else ""
        
        # Get short list for quick checks
        sources_short_result = subprocess.run(['pactl', 'list', 'sources', 'short'], 
                                            capture_output=True, text=True, timeout=5)
        sinks_short_result = subprocess.run(['pactl', 'list', 'sinks', 'short'], 
                                          capture_output=True, text=True, timeout=5)
        
        pa_sources_short = sources_short_result.stdout if sources_short_result.returncode == 0 else ""
        pa_sinks_short = sinks_short_result.stdout if sinks_short_result.returncode == 0 else ""
        
        def get_device_connection_info(mac_address, pa_device_name, is_source=True):
            """Get comprehensive device connection info"""
            detailed_output = pa_sources_detailed if is_source else pa_sinks_detailed
            short_output = pa_sources_short if is_source else pa_sinks_short
            
            # Check if PulseAudio device exists
            pa_device_exists = pa_device_name in short_output
            
            # Get PulseAudio state if device exists
            pa_status = 'not_available'
            if pa_device_exists:
                lines = detailed_output.split('\n')
                for i, line in enumerate(lines):
                    if pa_device_name in line and 'Name:' in line:
                        # Look backwards for the State line
                        for j in range(max(0, i - 5), i):
                            if 'State:' in lines[j]:
                                state_line = lines[j]
                                if 'RUNNING' in state_line.upper():
                                    pa_status = 'running'
                                elif 'IDLE' in state_line.upper():
                                    pa_status = 'idle'
                                elif 'SUSPENDED' in state_line.upper():
                                    pa_status = 'idle'
                                break
                        break
                if pa_status == 'not_available':
                    pa_status = 'idle'  # Default if device exists but no state found
            
            # Check Bluetooth connection status (try multiple controllers)
            bt_connected = False
            bt_paired = False
            if mac_address not in ['builtin']:
                # Get list of available controllers
                controllers = []
                try:
                    ctrl_result = subprocess.run(['bluetoothctl', 'list'], 
                                               capture_output=True, text=True, timeout=3)
                    for line in ctrl_result.stdout.split('\n'):
                        if 'Controller' in line:
                            ctrl_mac = line.split()[1]
                            controllers.append(ctrl_mac)
                except:
                    pass
                
                # Check device on all controllers
                for controller in controllers:
                    try:
                        # Select controller and check device
                        subprocess.run(['bluetoothctl', 'select', controller], 
                                     capture_output=True, timeout=2)
                        result = subprocess.run(['bluetoothctl', 'info', mac_address], 
                                              capture_output=True, text=True, timeout=3)
                        if result.returncode == 0:
                            if 'Connected: yes' in result.stdout:
                                bt_connected = True
                                break
                            if 'Paired: yes' in result.stdout:
                                bt_paired = True
                    except:
                        continue
            
            # Determine overall status
            if mac_address == 'builtin':
                # Built-in devices are always "connected"
                return {
                    'status': pa_status if pa_device_exists else 'idle',
                    'connected': True,
                    'active': pa_status == 'running',
                    'audio_ready': pa_device_exists
                }
            elif pa_device_exists:
                # Device has active audio profile
                return {
                    'status': pa_status,
                    'connected': True,  # Audio profile exists = effectively connected
                    'active': pa_status == 'running',
                    'audio_ready': True
                }
            elif bt_connected:
                # Bluetooth connected but no audio profile active
                return {
                    'status': 'idle',  # Show as ready but not active
                    'connected': True,
                    'active': False,
                    'audio_ready': False
                }
            elif bt_paired:
                # Paired but not connected - can potentially reconnect quickly
                return {
                    'status': 'idle',  # Show as potentially available
                    'connected': True,  # Consider paired devices as "connected" for UI purposes
                    'active': False,
                    'audio_ready': False
                }
            else:
                # Not connected at all
                return {
                    'status': 'not_available',
                    'connected': False,
                    'active': False,
                    'audio_ready': False
                }
        
        # Parse paired devices
        for line in paired_result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(' ', 2)
                if len(parts) >= 3:
                    device_type = parts[0]  # "Device"
                    mac_address = parts[1]
                    device_name = parts[2]
                    
                    # Clean MAC address format for PulseAudio
                    pa_mac = mac_address.replace(':', '_')
                    
                    # Check device capabilities
                    source_name = f"bluez_source.{pa_mac}.a2dp_source"
                    sink_name = f"bluez_sink.{pa_mac}.a2dp_sink"
                    
                    # Determine device type and get status
                    device_info = {
                        'name': device_name,
                        'mac': mac_address
                    }
                    
                    # Check if it's a sink (output device)
                    if ('buds' in device_name.lower() or 'headphone' in device_name.lower() or 
                        'speaker' in device_name.lower() or sink_name in pa_sinks_short):
                        
                        conn_info = get_device_connection_info(mac_address, sink_name, is_source=False)
                        device_info.update({
                            'pa_sink': sink_name,
                            'type': 'headphones',
                            'status': conn_info['status'],
                            'connected': conn_info['connected'],
                            'active': conn_info['active']
                        })
                        devices['outputs'].append(device_info)
                    
                    # Check if it's a source (input device)
                    elif ('phone' in device_name.lower() or 'pixel' in device_name.lower() or 
                          source_name in pa_sources_short):
                        
                        conn_info = get_device_connection_info(mac_address, source_name, is_source=True)
                        device_info.update({
                            'pa_source': source_name,
                            'type': 'phone',
                            'status': conn_info['status'],
                            'connected': conn_info['connected'],
                            'active': conn_info['active']
                        })
                        devices['sources'].append(device_info)
        
        # Add known devices if not already present and available in PulseAudio
        known_devices = [
            {
                'name': 'Pixel 9 Pro',
                'mac': 'C0:1C:6A:AD:78:E6',
                'pa_source': 'bluez_source.C0_1C_6A_AD_78_E6.a2dp_source',
                'is_source': True,
                'type': 'phone'
            },
            {
                'name': 'Galaxy Buds3 Pro',
                'mac': 'BC:A0:80:EB:21:AA',
                'pa_sink': 'bluez_sink.BC_A0_80_EB_21_AA.a2dp_sink',
                'is_source': False,
                'type': 'headphones'
            }
        ]
        
        # Check existing devices to avoid duplicates
        existing_source_macs = [d['mac'] for d in devices['sources']]
        existing_sink_macs = [d['mac'] for d in devices['outputs']]
        
        for known in known_devices:
            if known['is_source'] and known['mac'] not in existing_source_macs:
                conn_info = get_device_connection_info(known['mac'], known['pa_source'], is_source=True)
                devices['sources'].append({
                    'name': known['name'],
                    'mac': known['mac'],
                    'pa_source': known['pa_source'],
                    'type': known['type'],
                    'status': conn_info['status'],
                    'connected': conn_info['connected'],
                    'active': conn_info['active']
                })
            elif not known['is_source'] and known['mac'] not in existing_sink_macs:
                conn_info = get_device_connection_info(known['mac'], known['pa_sink'], is_source=False)
                devices['outputs'].append({
                    'name': known['name'],
                    'mac': known['mac'],
                    'pa_sink': known['pa_sink'],
                    'type': known['type'],
                    'status': conn_info['status'],
                    'connected': conn_info['connected'],
                    'active': conn_info['active']
                })
        
        # Add built-in speakers as an always-available sink option
        builtin_speakers_name = None
        # Common built-in speaker names on Raspberry Pi
        builtin_candidates = ['alsa_output.platform-bcm2835_audio.analog-stereo', 
                            'alsa_output.0.analog-stereo',
                            'alsa_output.hw_0_0',
                            'alsa_output.platform-107c701400.hdmi.hdmi-stereo']
        
        for candidate in builtin_candidates:
            if candidate in pa_sinks_short:
                builtin_speakers_name = candidate
                break
        
        if builtin_speakers_name:
            # Check if already in outputs (shouldn't be, but be safe)
            builtin_exists = any(d.get('pa_sink') == builtin_speakers_name for d in devices['outputs'])
            if not builtin_exists:
                conn_info = get_device_connection_info('builtin', builtin_speakers_name, is_source=False)
                devices['outputs'].append({
                    'name': 'Built-in Speakers',
                    'mac': 'builtin',  # Special identifier
                    'pa_sink': builtin_speakers_name,
                    'type': 'speakers',
                    'status': conn_info['status'],
                    'connected': conn_info['connected'],
                    'active': conn_info['active']
                })
        
        return jsonify(devices)
        
    except Exception as e:
        logger.error(f"Error getting Bluetooth devices: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/bluetooth/connect', methods=['POST'])
def connect_bluetooth_device():
    """Connect to a specific Bluetooth device"""
    try:
        import subprocess
        
        data = request.json
        mac_address = data.get('mac')
        device_type = data.get('type')  # 'source' or 'output'
        
        if not mac_address:
            return jsonify({'error': 'MAC address required'}), 400
        
        # Try to connect via bluetoothctl
        connect_result = subprocess.run(['bluetoothctl', 'connect', mac_address], 
                                      capture_output=True, text=True, timeout=20)
        
        if connect_result.returncode == 0 or 'Connected: yes' in connect_result.stdout:
            # Wait a moment for audio profile to establish
            time.sleep(2)
            
            # Check if audio profile is available
            if device_type == 'source':
                check_result = subprocess.run(['pactl', 'list', 'sources', 'short'], 
                                            capture_output=True, text=True, timeout=5)
                pa_mac = mac_address.replace(':', '_')
                if f"bluez_source.{pa_mac}" in check_result.stdout:
                    return jsonify({
                        'success': True,
                        'message': f"Connected to audio source {mac_address}"
                    })
            elif device_type == 'output':
                check_result = subprocess.run(['pactl', 'list', 'sinks', 'short'], 
                                            capture_output=True, text=True, timeout=5)
                pa_mac = mac_address.replace(':', '_')
                if f"bluez_sink.{pa_mac}" in check_result.stdout:
                    return jsonify({
                        'success': True,
                        'message': f"Connected to audio output {mac_address}"
                    })
            
            return jsonify({
                'success': True,
                'message': f"Connected to {mac_address}"
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to connect to device',
                'details': connect_result.stderr
            }), 400
            
    except Exception as e:
        logger.error(f"Error connecting to Bluetooth device: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/bluetooth/disconnect', methods=['POST'])
def disconnect_bluetooth_device():
    """Disconnect from a specific Bluetooth device"""
    try:
        import subprocess
        
        data = request.json
        mac_address = data.get('mac')
        
        if not mac_address:
            return jsonify({'error': 'MAC address required'}), 400
        
        # Disconnect via bluetoothctl
        disconnect_result = subprocess.run(['bluetoothctl', 'disconnect', mac_address], 
                                         capture_output=True, text=True, timeout=10)
        
        if disconnect_result.returncode == 0 or 'Successful disconnected' in disconnect_result.stdout:
            return jsonify({
                'success': True,
                'message': f"Disconnected from {mac_address}"
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to disconnect from device',
                'details': disconnect_result.stderr
            }), 400
            
    except Exception as e:
        logger.error(f"Error disconnecting from Bluetooth device: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/audio/switch_sink', methods=['POST'])
def switch_audio_sink():
    """Switch the default audio sink (output device)"""
    try:
        import subprocess
        
        data = request.json
        sink_name = data.get('sink_name')
        
        if not sink_name:
            return jsonify({'error': 'Sink name required'}), 400
        
        # Set the default sink using pactl
        switch_result = subprocess.run(['pactl', 'set-default-sink', sink_name], 
                                     capture_output=True, text=True, timeout=5)
        
        if switch_result.returncode == 0:
            # Also move any existing sink inputs to the new sink
            move_result = subprocess.run(['pactl', 'list', 'short', 'sink-inputs'], 
                                       capture_output=True, text=True, timeout=5)
            
            if move_result.returncode == 0:
                for line in move_result.stdout.strip().split('\n'):
                    if line.strip():
                        sink_input_id = line.split()[0]
                        subprocess.run(['pactl', 'move-sink-input', sink_input_id, sink_name], 
                                     capture_output=True, timeout=3)
            
            return jsonify({
                'success': True,
                'message': f"Audio output switched to {sink_name}"
            })
        else:
            return jsonify({
                'success': False,
                'error': f"Failed to switch sink: {switch_result.stderr or switch_result.stdout}"
            }), 500
            
    except Exception as e:
        logger.error(f"Error switching audio sink: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/audio/switch_source', methods=['POST'])
def switch_audio_source():
    """Switch the default audio source (input device)"""
    try:
        import subprocess
        
        data = request.json
        source_name = data.get('source_name')
        
        if not source_name:
            return jsonify({'error': 'Source name required'}), 400
        
        # Set the default source using pactl
        switch_result = subprocess.run(['pactl', 'set-default-source', source_name], 
                                     capture_output=True, text=True, timeout=5)
        
        if switch_result.returncode == 0:
            # Also move any existing source outputs to the new source
            move_result = subprocess.run(['pactl', 'list', 'short', 'source-outputs'], 
                                       capture_output=True, text=True, timeout=5)
            
            if move_result.returncode == 0:
                for line in move_result.stdout.strip().split('\n'):
                    if line.strip():
                        source_output_id = line.split()[0]
                        subprocess.run(['pactl', 'move-source-output', source_output_id, source_name], 
                                     capture_output=True, timeout=3)
            
            return jsonify({
                'success': True,
                'message': f"Audio input switched to {source_name}"
            })
        else:
            return jsonify({
                'success': False,
                'error': f"Failed to switch source: {switch_result.stderr or switch_result.stdout}"
            }), 500
            
    except Exception as e:
        logger.error(f"Error switching audio source: {e}")
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

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a specific session and its associated files"""
    try:
        sessions = load_sessions()
        session_to_delete = None
        session_index = None
        
        # Find the session
        for i, session in enumerate(sessions):
            if session.get('session_id') == session_id:
                session_to_delete = session
                session_index = i
                break
        
        if not session_to_delete:
            return jsonify({'error': 'Session not found'}), 404
        
        # Delete the WAV file if it exists
        wav_file = session_to_delete.get('wav_file')
        if wav_file and os.path.exists(wav_file):
            try:
                os.remove(wav_file)
                logger.info(f"Deleted audio file: {wav_file}")
            except OSError as e:
                logger.warning(f"Could not delete audio file {wav_file}: {e}")
        
        # Remove from sessions list
        sessions.pop(session_index)
        save_sessions(sessions)
        
        logger.info(f"Deleted session: {session_id}")
        return jsonify({'success': True, 'message': f'Session {session_id} deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
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

@app.route('/api/restart', methods=['POST'])
def restart_app():
    """Restart the application"""
    import os
    import sys
    import subprocess
    
    try:
        logger.info("App restart requested via API")
        
        def restart_server():
            # Give Flask time to send response
            import time
            time.sleep(1)
            
            # Use the dedicated restart script
            try:
                # Get the project directory
                project_dir = os.path.dirname(os.path.abspath(__file__))
                restart_script = os.path.join(project_dir, 'restart_webui.sh')
                
                # Execute the restart script in background
                logger.info(f"Executing restart script: {restart_script}")
                subprocess.Popen(['/bin/bash', restart_script], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL,
                                start_new_session=True)
                
                # Give it a moment to ensure the subprocess starts
                time.sleep(0.5)
                
                # Exit this process
                logger.info("Exiting current process...")
                os._exit(0)
            except Exception as e:
                logger.error(f"Error during restart: {e}")
                # Fallback - just exit
                os._exit(0)
        
        # Start restart in background
        import threading
        restart_thread = threading.Thread(target=restart_server, daemon=True)
        restart_thread.start()
        
        return jsonify({'success': True, 'message': 'App restarting...'})
        
    except Exception as e:
        logger.error(f"Error restarting app: {e}")
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
            
            # Check directory modification time every 30 seconds to avoid unnecessary work
            if current_time - transcription_processor_state['last_dir_check'] >= 30:
                transcription_processor_state['last_dir_check'] = current_time
                
                # Check if demo_sessions directory or sessions.json has been modified
                sessions_file = "demo_sessions/sessions.json"
                demo_dir = "demo_sessions"
                
                try:
                    # Check for new WAV files or sessions.json changes
                    dir_mtime = os.path.getmtime(demo_dir) if os.path.exists(demo_dir) else 0
                    file_mtime = os.path.getmtime(sessions_file) if os.path.exists(sessions_file) else 0
                    max_mtime = max(dir_mtime, file_mtime)
                    
                    # If nothing changed, skip processing
                    if max_mtime <= transcription_processor_state['sessions_dir_mtime']:
                        logger.debug("No changes in demo_sessions directory, skipping transcription check")
                        time.sleep(30)  # Wait longer if no changes
                        continue
                    
                    transcription_processor_state['sessions_dir_mtime'] = max_mtime
                    logger.info("Changes detected in demo_sessions directory, checking for transcription work")
                    
                except OSError as e:
                    logger.warning(f"Error checking directory modification time: {e}")
            
            # Check for new untranscribed sessions only if directory changed or forced check
            if (current_time - transcription_processor_state['last_check'] >= 120 or  # Force check every 2 minutes
                current_time - transcription_processor_state['last_dir_check'] <= 1):  # Or just detected changes
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
            result = adapter.transcribe_recording(wav_file, session_id)
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
            result = adapter.transcribe_recording(wav_file, session_id)
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