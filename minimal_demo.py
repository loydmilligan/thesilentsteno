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

# Check for audio availability (delegated to SimpleAudioRecorder)
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("sounddevice not installed. Run: pip install sounddevice")

# Speech recognition is now handled by the extracted transcriber module

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
        
        # Import the extracted modules
        import sys
        sys.path.append('/home/mmariani/projects/thesilentsteno/src')
        from recording.simple_audio_recorder import SimpleAudioRecorder
        from ai.simple_transcriber import SimpleTranscriber
        
        # Initialize audio recorder (extracted from this file)
        self.audio_recorder = SimpleAudioRecorder("demo_sessions")
        
        # Initialize transcriber (extracted from this file)
        self.transcriber = SimpleTranscriber(backend="cpu", model_name="base")
        
        # Application state - simplified, delegate to audio_recorder
        self.current_session = None
        
        # UI Components (will be created in build())
        self.start_button = None
        self.stop_button = None
        self.play_button = None
        self.status_label = None
        self.transcript_label = None
        
        logger.info("MinimalDemo initialized with extracted recording module")
    
    def get_sessions(self):
        """Get all sessions from audio recorder"""
        return self.audio_recorder.get_all_sessions()
    
    def update_session_transcript(self, transcript):
        """Update the last session with transcript"""
        sessions = self.audio_recorder.get_all_sessions()
        if sessions:
            # Update the last session's transcript
            sessions[-1]['transcript'] = transcript
            # Save back to audio recorder
            self.audio_recorder.sessions = sessions
            self.audio_recorder._save_sessions()
            logger.info("Session transcript updated")
    
    def transcribe_audio(self, wav_file):
        """Transcribe audio using extracted transcriber module"""
        if not self.transcriber.is_available():
            logger.warning("Transcriber not available")
            return "Transcription not available - check backend"
        
        try:
            # Update status (transcriber handles model loading internally)
            self.status_label.text = "Transcribing audio..."
            logger.info(f"Transcribing: {wav_file}")
            
            # Use the extracted transcriber
            transcript = self.transcriber.transcribe_audio(wav_file)
            
            if transcript and not transcript.startswith("Transcription error"):
                logger.info(f"Transcription complete: {len(transcript)} characters")
            else:
                logger.warning(f"Transcription issue: {transcript}")
            
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
        sessions = self.get_sessions()
        if sessions:
            history_text = f"Previous sessions: {len(sessions)}\n"
            history_text += f"Last session: {sessions[-1]['timestamp'][:19]}\n\n"
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
        
        recording_state = self.audio_recorder.get_recording_state()
        if recording_state != "idle":
            logger.warning(f"Cannot start recording, current state: {recording_state}")
            return
        
        if not AUDIO_AVAILABLE:
            self.status_label.text = "Audio recording not available - install sounddevice"
            logger.error("Cannot start recording - sounddevice not available")
            return
        
        # Generate session ID
        import uuid
        session_id = str(uuid.uuid4())[:8]
        
        # Start recording using extracted audio recorder
        if self.audio_recorder.start_recording(session_id):
            # Update UI state
            self.current_session = session_id
            self.start_button.disabled = True
            self.stop_button.disabled = False
            self.play_button.disabled = True
            self.status_label.text = "Recording... Press Stop to finish"
            
            logger.info(f"Audio recording started with session: {session_id}")
        else:
            self.status_label.text = "Failed to start recording"
            logger.error("Failed to start recording")
        
    def stop_recording(self, instance):
        """Stop recording audio"""
        logger.info("Stop recording button pressed")
        
        recording_state = self.audio_recorder.get_recording_state()
        if recording_state != "recording":
            logger.warning(f"Cannot stop recording, current state: {recording_state}")
            return
        
        # Stop recording using extracted audio recorder
        recording_info = self.audio_recorder.stop_recording()
        
        if recording_info:
            # Update UI state
            self.stop_button.disabled = True
            self.status_label.text = f"Recording complete: {recording_info['duration']:.1f} seconds"
            
            # Process the recording
            self._process_recording_result(recording_info)
            
            logger.info(f"Recording stopped successfully: {recording_info['wav_file']}")
        else:
            self.status_label.text = "Recording failed - no audio data"
            self._reset_to_idle()
            logger.error("Recording stop failed")
        
    def _process_recording_result(self, recording_info):
        """Process the recording result from audio recorder"""
        try:
            # Update UI with recording info
            self.start_button.disabled = False
            self.play_button.disabled = False
            
            # Show recording info in transcript area
            transcript_info = f"""[{datetime.now().strftime('%H:%M:%S')}] Recording completed successfully.

Duration: {recording_info['duration']:.2f} seconds
Samples: {recording_info['samples']:,}
Sample Rate: {recording_info['sample_rate']:,} Hz
Channels: {recording_info['channels']}

Transcription: Loading..."""
            
            self.transcript_label.text = transcript_info
            
            # Start transcription in background
            threading.Thread(target=self._transcribe_in_background, 
                           args=(recording_info['wav_file'],), daemon=True).start()
            
            logger.info(f"Processing complete for: {recording_info['wav_file']}")
            
        except Exception as e:
            logger.error(f"Error processing recording result: {e}")
            self._reset_to_idle()
            
    def _reset_to_idle(self):
        """Reset UI to idle state"""
        self.audio_recorder.reset_to_idle()
        self.start_button.disabled = False
        self.stop_button.disabled = True
        self.play_button.disabled = True
        
    def _transcribe_in_background(self, wav_filename):
        """Transcribe audio in background thread"""
        try:
            transcript = self.transcribe_audio(wav_filename)
            
            # Update UI on main thread
            def update_transcript(dt):
                logger.info(f"Updating transcript UI with: {transcript[:100]}...")
                if "Transcription error" not in transcript and transcript:
                    self.transcript_label.text += f"\n\n--- TRANSCRIPT ---\n{transcript}"
                    # Update session with real transcript using audio recorder
                    self.update_session_transcript(transcript)
                    # Update status to show completion
                    self.status_label.text = "Transcription complete. Ready for next recording."
                elif transcript:
                    self.transcript_label.text += f"\n\n{transcript}"
                    self.status_label.text = "Transcription error. Ready for next recording."
                else:
                    self.transcript_label.text += "\n\n--- TRANSCRIPT ---\n[No speech detected or transcription was empty]"
                    self.status_label.text = "No speech detected. Ready for next recording."
                    logger.warning("Transcript was empty")
                    
            Clock.schedule_once(update_transcript, 0)
            
        except Exception as e:
            logger.error(f"Background transcription error: {e}")
        
    def play_audio(self, instance):
        """Play back recorded audio"""
        logger.info("Play audio button pressed")
        
        recording_state = self.audio_recorder.get_recording_state()
        if recording_state != "ready":
            logger.warning(f"Cannot play audio, current state: {recording_state}")
            return
        
        # Use extracted audio recorder for playback
        if self.audio_recorder.play_recording():
            # Update UI
            self.play_button.disabled = True
            self.start_button.disabled = True
            self.status_label.text = "Playing audio..."
            
            # Re-enable buttons after a short delay (playback is async)
            Clock.schedule_once(self._finish_playback, 2.0)
            
            logger.info("Audio playback started")
        else:
            self.status_label.text = "No audio data to play"
            logger.warning("No audio available for playback")
        
    def _finish_playback(self, dt):
        """Finish playback and reset UI"""
        logger.info("Audio playback finished")
        self.status_label.text = "Playback complete. Ready for next recording."
        self.play_button.disabled = False
        self.start_button.disabled = False
        
    def on_stop(self):
        """Cleanup when app stops"""
        logger.info("Application stopping...")
        
        # Stop any active recording using audio recorder
        recording_state = self.audio_recorder.get_recording_state()
        if recording_state == "recording":
            logger.info("Stopping active recording...")
            self.audio_recorder.stop_recording()
        
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