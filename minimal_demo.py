#!/usr/bin/env python3
"""
The Silent Steno - Minimal Demo
Walking Skeleton Implementation

Single-file proof of concept demonstrating:
1. Basic Kivy UI with 3 buttons
2. Audio recording from local microphone
3. Dummy transcript display
4. Simple file persistence
5. Complete end-to-end flow

This will be incrementally built up and then refactored into modules.
"""

import kivy
kivy.require('2.0.0')

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.config import Config

import os
import sys
import logging
import threading
import time
import json
import numpy as np
from datetime import datetime

# Audio recording
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("sounddevice not installed. Run: pip install sounddevice")
    
# Audio file handling
try:
    import wave
    WAVE_AVAILABLE = True
except ImportError:
    WAVE_AVAILABLE = False
    print("wave module not available")

# Speech recognition - use Whisper directly for the walking skeleton
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Whisper not installed. Run: pip install openai-whisper")

# Configure Kivy for Pi 5 touchscreen (1024x600)
Config.set('graphics', 'width', '1024')
Config.set('graphics', 'height', '600')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'fullscreen', '1')  # Fullscreen mode
Config.set('graphics', 'borderless', '1')  # Remove window borders
Config.set('graphics', 'window_state', 'maximized')  # Maximize window

# Disable virtual keyboard
Config.set('kivy', 'keyboard_mode', 'system')
Config.set('kivy', 'keyboard_layout', 'qwerty')
Config.set('graphics', 'show_cursor', '1')  # Show cursor for debugging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MinimalDemo(App):
    """
    Minimal Silent Steno Demo Application
    
    Single-file implementation of the walking skeleton:
    - Record audio from local microphone
    - Display transcript (dummy initially)
    - Save and play back recordings
    """
    
    def __init__(self):
        super().__init__()
        self.title = "Silent Steno - Minimal Demo"
        
        # Application state
        self.recording_state = "idle"  # idle, recording, processing, ready
        self.current_session = None
        self.sessions_dir = "demo_sessions"
        
        # Audio recording settings
        self.sample_rate = 44100  # Hz (44.1kHz standard rate)
        self.channels = 1  # Mono recording
        self.audio_data = None
        self.recording_thread = None
        self.is_recording = False
        self.input_device = None
        self.output_device = None
        self.current_wav_file = None
        
        # UI Components (will be created in build())
        self.start_button = None
        self.stop_button = None
        self.play_button = None
        self.status_label = None
        self.transcript_label = None
        
        # Ensure sessions directory exists
        os.makedirs(self.sessions_dir, exist_ok=True)
        
        # Session persistence
        self.sessions_file = os.path.join(self.sessions_dir, "sessions.json")
        self.sessions = self.load_sessions()
        
        # Whisper model (loaded on first use)
        self.whisper_model = None
        self.model_name = "base"  # base model is ~140MB, good balance of speed/accuracy
        
        # Check audio availability
        if not AUDIO_AVAILABLE:
            logger.error("Audio recording not available - sounddevice not installed")
        else:
            logger.info("Audio recording available")
            # List available audio devices and find CMTECK ZM350
            try:
                devices = sd.query_devices()
                logger.info(f"Available audio devices: {len(devices)}")
                for i, device in enumerate(devices):
                    logger.info(f"Device {i}: {device['name']} - In: {device['max_input_channels']} Out: {device['max_output_channels']}")
                    
                    # Look for USB Audio Device for INPUT
                    if "USB Audio Device" in device['name'] and device['max_input_channels'] > 0:
                        self.input_device = i
                        logger.info(f"Found USB Audio Device input: {i}")
                    
                    # Look for built-in speakers for OUTPUT (HDMI)
                    if "vc4-hdmi" in device['name'] and device['max_output_channels'] > 0:
                        self.output_device = i
                        logger.info(f"Found HDMI audio output: {i} - {device['name']}")
                    
                    # Fallback to first available input/output
                    if self.input_device is None and device['max_input_channels'] > 0:
                        self.input_device = i
                    if self.output_device is None and device['max_output_channels'] > 0:
                        self.output_device = i
                        
                logger.info(f"Selected input device: {self.input_device}")
                logger.info(f"Selected output device: {self.output_device}")
                
            except Exception as e:
                logger.error(f"Error querying audio devices: {e}")
        
        logger.info("MinimalDemo initialized")
    
    def load_sessions(self):
        """Load sessions from JSON file"""
        try:
            if os.path.exists(self.sessions_file):
                with open(self.sessions_file, 'r') as f:
                    sessions = json.load(f)
                logger.info(f"Loaded {len(sessions)} sessions")
                return sessions
            else:
                logger.info("No existing sessions file, starting fresh")
                return []
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
            return []
    
    def save_sessions(self):
        """Save sessions to JSON file"""
        try:
            with open(self.sessions_file, 'w') as f:
                json.dump(self.sessions, f, indent=2)
            logger.info(f"Saved {len(self.sessions)} sessions")
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")
    
    def add_session(self, wav_file, duration, samples):
        """Add a new session to the history"""
        session = {
            'id': len(self.sessions) + 1,
            'timestamp': datetime.now().isoformat(),
            'wav_file': wav_file,
            'duration': duration,
            'samples': samples,
            'transcript': 'Dummy transcript (Whisper integration pending)'
        }
        self.sessions.append(session)
        self.save_sessions()
        logger.info(f"Added session {session['id']}: {wav_file}")
        return session
    
    def transcribe_audio(self, wav_file):
        """Transcribe audio using Whisper directly"""
        if not WHISPER_AVAILABLE:
            logger.warning("Whisper not available for transcription")
            return "Whisper not installed - cannot transcribe"
        
        try:
            # Load model on first use
            if self.whisper_model is None:
                self.status_label.text = f"Loading Whisper {self.model_name} model (first time only)..."
                logger.info(f"Loading Whisper model: {self.model_name}")
                self.whisper_model = whisper.load_model(self.model_name)
                logger.info("Whisper model loaded successfully")
            
            # Transcribe the audio
            self.status_label.text = "Transcribing audio..."
            logger.info(f"Transcribing: {wav_file}")
            
            result = self.whisper_model.transcribe(wav_file)
            transcript = result["text"].strip()
            
            logger.info(f"Transcription complete: {len(transcript)} characters")
            return transcript
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return f"Transcription error: {str(e)}"
    
    def build(self):
        """Build the minimal UI layout"""
        logger.info("Building UI...")
        
        # Main layout
        main_layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        
        # Title
        title_label = Label(
            text="The Silent Steno - Walking Skeleton",
            size_hint_y=None,
            height=50,
            font_size=24,
            bold=True
        )
        main_layout.add_widget(title_label)
        
        # Status display
        self.status_label = Label(
            text="Ready to record",
            size_hint_y=None,
            height=40,
            font_size=18
        )
        main_layout.add_widget(self.status_label)
        
        # Button layout
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=80, spacing=20)
        
        # Start Recording button
        self.start_button = Button(
            text="Start Recording",
            font_size=18,
            bold=True,
            background_color=(0.2, 0.8, 0.2, 1)  # Green
        )
        self.start_button.bind(on_press=self.start_recording)
        button_layout.add_widget(self.start_button)
        
        # Stop Recording button
        self.stop_button = Button(
            text="Stop Recording",
            font_size=18,
            bold=True,
            background_color=(0.8, 0.2, 0.2, 1),  # Red
            disabled=True
        )
        self.stop_button.bind(on_press=self.stop_recording)
        button_layout.add_widget(self.stop_button)
        
        # Play Audio button
        self.play_button = Button(
            text="Play Audio",
            font_size=18,
            bold=True,
            background_color=(0.2, 0.2, 0.8, 1),  # Blue
            disabled=True
        )
        self.play_button.bind(on_press=self.play_audio)
        button_layout.add_widget(self.play_button)
        
        main_layout.add_widget(button_layout)
        
        # Transcript display
        transcript_title = Label(
            text="Transcript:",
            size_hint_y=None,
            height=30,
            font_size=16,
            bold=True,
            halign='left'
        )
        transcript_title.bind(size=transcript_title.setter('text_size'))
        main_layout.add_widget(transcript_title)
        
        # Show session history in transcript area
        if self.sessions:
            history_text = f"Previous sessions: {len(self.sessions)}\n"
            history_text += f"Last session: {self.sessions[-1]['timestamp'][:19]}\n\n"
            history_text += "Start recording to see transcription here."
        else:
            history_text = "No previous sessions.\nStart recording to see transcription here."
            
        self.transcript_label = Label(
            text=history_text,
            font_size=14,
            halign='left',
            valign='top',
            text_size=(None, None)
        )
        self.transcript_label.bind(size=self.transcript_label.setter('text_size'))
        main_layout.add_widget(self.transcript_label)
        
        logger.info("UI built successfully")
        return main_layout
    
    def start_recording(self, instance):
        """Start recording audio"""
        logger.info("Start recording button pressed")
        
        if self.recording_state != "idle":
            logger.warning(f"Cannot start recording, current state: {self.recording_state}")
            return
        
        if not AUDIO_AVAILABLE:
            self.status_label.text = "Audio recording not available - install sounddevice"
            logger.error("Cannot start recording - sounddevice not available")
            return
        
        # Update UI state
        self.recording_state = "recording"
        self.start_button.disabled = True
        self.stop_button.disabled = False
        self.play_button.disabled = True
        self.status_label.text = "Recording... Press Stop to finish"
        
        # Start recording in a separate thread
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_audio, daemon=True)
        self.recording_thread.start()
        
        logger.info("Audio recording started")
        
    def _record_audio(self):
        """Record audio in background thread"""
        try:
            # Initialize recording buffer
            recording_buffer = []
            
            def audio_callback(indata, frames, time, status):
                """Callback for audio recording"""
                if status:
                    logger.warning(f"Audio callback status: {status}")
                if self.is_recording:
                    recording_buffer.append(indata.copy())
            
            # Start recording stream
            with sd.InputStream(
                callback=audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=np.float32,
                device=self.input_device
            ):
                logger.info(f"Recording stream started: {self.sample_rate}Hz, {self.channels} channels")
                
                # Keep recording until stopped
                while self.is_recording:
                    sd.sleep(100)  # Sleep for 100ms
                
                # Combine all recorded chunks
                if recording_buffer:
                    self.audio_data = np.concatenate(recording_buffer, axis=0)
                    duration = len(self.audio_data) / self.sample_rate
                    logger.info(f"Recording completed: {duration:.2f} seconds, {len(self.audio_data)} samples")
                else:
                    self.audio_data = None
                    logger.warning("No audio data recorded")
                    
        except Exception as e:
            logger.error(f"Error during audio recording: {e}")
            self.audio_data = None
            
            # Update UI on main thread
            error_msg = str(e)  # Capture the error message
            Clock.schedule_once(lambda dt: self._recording_error(error_msg), 0)
        
    def stop_recording(self, instance):
        """Stop recording audio"""
        logger.info("Stop recording button pressed")
        
        if self.recording_state != "recording":
            logger.warning(f"Cannot stop recording, current state: {self.recording_state}")
            return
        
        # Stop the recording
        self.is_recording = False
        
        # Update UI state
        self.recording_state = "processing"
        self.stop_button.disabled = True
        self.status_label.text = "Processing recording..."
        
        # Wait for recording thread to finish, then process
        Clock.schedule_once(self._check_recording_finished, 0.1)
        
        logger.info("Recording stop requested")
        
    def _check_recording_finished(self, dt):
        """Check if recording thread has finished"""
        if self.recording_thread and self.recording_thread.is_alive():
            # Still recording, check again in 100ms
            Clock.schedule_once(self._check_recording_finished, 0.1)
        else:
            # Recording finished, process the results
            self._process_recording()
            
    def _process_recording(self):
        """Process the completed recording"""
        if self.audio_data is not None:
            duration = len(self.audio_data) / self.sample_rate
            logger.info(f"Processing recording: {duration:.2f} seconds")
            
            # Update status
            self.status_label.text = f"Recording complete: {duration:.1f} seconds"
            
            # Simulate processing delay then show results
            Clock.schedule_once(self.finish_processing, 1.0)
        else:
            logger.error("No audio data to process")
            self.status_label.text = "Recording failed - no audio data"
            self._reset_to_idle()
            
    def _recording_error(self, error_msg):
        """Handle recording errors (called from main thread)"""
        logger.error(f"Recording error: {error_msg}")
        self.status_label.text = f"Recording error: {error_msg}"
        self._reset_to_idle()
        
    def _reset_to_idle(self):
        """Reset UI to idle state"""
        self.recording_state = "idle"
        self.start_button.disabled = False
        self.stop_button.disabled = True
        self.play_button.disabled = True
        self.is_recording = False
        self.audio_data = None  # Clear previous recording
        
    def finish_processing(self, dt):
        """Finish processing and show results"""
        logger.info("Processing finished")
        
        # Update UI state
        self.recording_state = "ready"
        self.start_button.disabled = False
        self.play_button.disabled = False
        
        if self.audio_data is not None:
            duration = len(self.audio_data) / self.sample_rate
            samples = len(self.audio_data)
            
            self.status_label.text = f"Recording ready: {duration:.1f}s, {samples} samples"
            
            # Show recording info in transcript area
            transcript_info = f"""[{datetime.now().strftime('%H:%M:%S')}] Recording completed successfully.

Duration: {duration:.2f} seconds
Samples: {samples:,}
Sample Rate: {self.sample_rate:,} Hz
Channels: {self.channels}

Transcription: Loading..."""
            
            self.transcript_label.text = transcript_info
            logger.info(f"Recording ready: {duration:.2f}s, {samples} samples")
        else:
            self.status_label.text = "No recording data available"
            self.transcript_label.text = "No recording data to process."
            
        # Save to WAV file
        if self.audio_data is not None and WAVE_AVAILABLE:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                wav_filename = os.path.join(self.sessions_dir, f"recording_{timestamp}.wav")
                
                # Convert float32 to int16 for WAV file
                audio_int16 = (self.audio_data * 32767).astype(np.int16)
                
                with wave.open(wav_filename, 'wb') as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(2)  # 2 bytes for int16
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(audio_int16.tobytes())
                
                self.current_wav_file = wav_filename
                logger.info(f"Saved recording to: {wav_filename}")
                
                # Add to session history
                duration = len(self.audio_data) / self.sample_rate
                session = self.add_session(wav_filename, duration, len(self.audio_data))
                
                self.transcript_label.text += f"\n\nRecording saved to: {wav_filename}"
                self.transcript_label.text += f"\nSession ID: {session['id']}"
                
                # Start transcription in background
                threading.Thread(target=self._transcribe_in_background, 
                               args=(wav_filename,), daemon=True).start()
            except Exception as e:
                logger.error(f"Error saving WAV file: {e}")
                self.current_wav_file = None
        
    def _transcribe_in_background(self, wav_filename):
        """Transcribe audio in background thread"""
        try:
            transcript = self.transcribe_audio(wav_filename)
            
            # Update UI on main thread
            def update_transcript(dt):
                if "Transcription error" not in transcript:
                    self.transcript_label.text += f"\n\n--- TRANSCRIPT ---\n{transcript}"
                    # Update session with real transcript
                    if self.sessions:
                        self.sessions[-1]['transcript'] = transcript
                        self.save_sessions()
                else:
                    self.transcript_label.text += f"\n\n{transcript}"
                    
            Clock.schedule_once(update_transcript, 0)
            
        except Exception as e:
            logger.error(f"Background transcription error: {e}")
        
    def play_audio(self, instance):
        """Play back recorded audio"""
        logger.info("Play audio button pressed")
        
        if self.recording_state != "ready":
            logger.warning(f"Cannot play audio, current state: {self.recording_state}")
            return
            
        if self.audio_data is None:
            logger.warning("No audio data to play")
            self.status_label.text = "No audio data to play"
            return
            
        if not AUDIO_AVAILABLE:
            self.status_label.text = "Audio playback not available"
            return
        
        # Update UI
        self.play_button.disabled = True
        self.start_button.disabled = True
        self.status_label.text = "Playing audio..."
        
        # Play audio in background thread
        playback_thread = threading.Thread(target=self._play_audio_thread, daemon=True)
        playback_thread.start()
        
        logger.info("Audio playback started")
        
    def _play_audio_thread(self):
        """Play audio in background thread"""
        try:
            # Use aplay to play the WAV file if available
            if self.current_wav_file and os.path.exists(self.current_wav_file):
                import subprocess
                logger.info(f"Playing WAV file with aplay: {self.current_wav_file}")
                
                # Use aplay command - let it use default device (USB when connected)
                result = subprocess.run(
                    ['aplay', self.current_wav_file],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.error(f"aplay failed: {result.stderr}")
                    raise Exception(f"aplay failed: {result.stderr}")
                else:
                    logger.info("aplay playback completed successfully")
            else:
                # Fallback to sounddevice if no WAV file
                logger.warning("No WAV file available, trying direct playback")
                sd.play(self.audio_data, samplerate=self.sample_rate)
                sd.wait()
            
            # Update UI on main thread
            Clock.schedule_once(self.finish_playback, 0)
            
        except Exception as e:
            logger.error(f"Error during audio playback: {e}")
            error_msg = str(e)  # Capture the error message
            Clock.schedule_once(lambda dt: self._playback_error(error_msg), 0)
    
    def _playback_error(self, error_msg):
        """Handle playback errors (called from main thread)"""
        logger.error(f"Playback error: {error_msg}")
        self.status_label.text = f"Playback error: {error_msg}"
        self.play_button.disabled = False
        self.start_button.disabled = False
        
    def finish_playback(self, dt):
        """Finish playback and reset UI"""
        logger.info("Audio playback finished")
        self.status_label.text = "Playback complete. Ready for next recording."
        self.play_button.disabled = False
        self.start_button.disabled = False
        # Reset to idle state for next recording
        self.recording_state = "idle"
        
    def on_stop(self):
        """Cleanup when app stops"""
        logger.info("Application stopping...")
        
        # Stop any active recording
        if self.is_recording:
            self.is_recording = False
            logger.info("Stopping active recording...")
            
            # Wait for recording thread to finish
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=2.0)
                if self.recording_thread.is_alive():
                    logger.warning("Recording thread did not finish cleanly")
        
        return super().on_stop()

if __name__ == '__main__':
    logger.info("Starting The Silent Steno - Minimal Demo")
    
    # Check if running on Pi 5 (optional, for logging)
    try:
        with open('/proc/cpuinfo', 'r') as f:
            if 'BCM2712' in f.read():  # Pi 5 identifier
                logger.info("Running on Raspberry Pi 5")
    except:
        logger.info("Running on development machine")
    
    app = MinimalDemo()
    app.run()