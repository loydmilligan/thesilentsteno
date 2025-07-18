#!/usr/bin/env python3

"""
Simple Recording UI for The Silent Steno

This module provides a simplified UI interface that extracts the basic recording
interface from minimal_demo.py and bridges it to the existing comprehensive UI
architecture. This maintains the working UI while allowing integration with the
full UI system.

Key features:
- Simple 3-button interface (Start/Stop/Play)
- Status display and transcript area
- Session history integration
- Bridge to comprehensive UI components
"""

import logging
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime
import threading

try:
    from kivy.app import App
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.button import Button
    from kivy.uix.label import Label
    from kivy.clock import Clock
    from kivy.metrics import dp
    KIVY_AVAILABLE = True
except ImportError:
    KIVY_AVAILABLE = False
    logging.warning("Kivy not available for UI")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleRecordingUI(BoxLayout):
    """
    Simple Recording UI Widget
    
    Provides a basic 3-button interface for recording, transcription, and playback
    that matches the minimal_demo.py functionality while being extractable as a
    reusable component.
    """
    
    def __init__(self, **kwargs):
        """Initialize the simple recording UI"""
        super().__init__(**kwargs)
        
        # Set layout properties
        self.orientation = 'vertical'
        self.padding = dp(20)
        self.spacing = dp(20)
        
        # Callback functions (set by parent)
        self.on_start_recording = None
        self.on_stop_recording = None
        self.on_play_audio = None
        self.on_quit_app = None
        
        # UI Components
        self.start_button = None
        self.stop_button = None
        self.play_button = None
        self.status_label = None
        self.transcript_label = None
        
        # Build the interface
        self._build_ui()
        
        # Enable keyboard shortcuts
        self.bind(on_touch_down=self._on_key_down)
        
        logger.info("SimpleRecordingUI initialized")
    
    def _build_ui(self):
        """Build the UI components"""
        # Title
        title_label = Label(
            text="The Silent Steno - Walking Skeleton",
            size_hint_y=None,
            height=dp(50),
            font_size=dp(24),
            bold=True
        )
        self.add_widget(title_label)
        
        # Status display
        self.status_label = Label(
            text="Ready to record",
            size_hint_y=None,
            height=dp(40),
            font_size=dp(16),
            color=(0.8, 0.8, 0.8, 1)
        )
        self.add_widget(self.status_label)
        
        # Button controls
        button_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(100),
            spacing=dp(20)
        )
        
        # Start Recording Button
        self.start_button = Button(
            text="Start Recording",
            font_size=dp(18),
            background_color=(0.3, 0.7, 0.3, 1),
            disabled=False
        )
        self.start_button.bind(on_press=self._on_start_pressed)
        button_layout.add_widget(self.start_button)
        
        # Stop Recording Button
        self.stop_button = Button(
            text="Stop Recording",
            font_size=dp(18),
            background_color=(0.7, 0.3, 0.3, 1),
            disabled=True
        )
        self.stop_button.bind(on_press=self._on_stop_pressed)
        button_layout.add_widget(self.stop_button)
        
        # Play Audio Button
        self.play_button = Button(
            text="Play Audio",
            font_size=dp(18),
            background_color=(0.3, 0.3, 0.7, 1),
            disabled=True
        )
        self.play_button.bind(on_press=self._on_play_pressed)
        button_layout.add_widget(self.play_button)
        
        # Quit Button
        self.quit_button = Button(
            text="Quit",
            font_size=dp(18),
            background_color=(0.5, 0.5, 0.5, 1),
            disabled=False,
            size_hint_x=0.5  # Make it smaller than other buttons
        )
        self.quit_button.bind(on_press=self._on_quit_pressed)
        button_layout.add_widget(self.quit_button)
        
        self.add_widget(button_layout)
        
        # Transcript/History display
        transcript_title = Label(
            text="Session Transcript & History",
            size_hint_y=None,
            height=dp(30),
            font_size=dp(16),
            bold=True,
            halign='left'
        )
        transcript_title.bind(size=transcript_title.setter('text_size'))
        self.add_widget(transcript_title)
        
        # Initial transcript area
        self.transcript_label = Label(
            text="Start recording to see transcription here.",
            font_size=dp(14),
            halign='left',
            valign='top',
            text_size=(None, None)
        )
        self.transcript_label.bind(size=self.transcript_label.setter('text_size'))
        self.add_widget(self.transcript_label)
        
        logger.info("UI components built successfully")
    
    def _on_start_pressed(self, instance):
        """Handle start recording button press"""
        logger.info("Start recording button pressed")
        if self.on_start_recording:
            self.on_start_recording()
    
    def _on_stop_pressed(self, instance):
        """Handle stop recording button press"""
        logger.info("Stop recording button pressed")
        if self.on_stop_recording:
            self.on_stop_recording()
    
    def _on_play_pressed(self, instance):
        """Handle play audio button press"""
        logger.info("Play audio button pressed")
        if self.on_play_audio:
            self.on_play_audio()
    
    def _on_quit_pressed(self, instance):
        """Handle quit button press"""
        logger.info("Quit button pressed")
        if self.on_quit_app:
            self.on_quit_app()
    
    def _on_key_down(self, instance, touch):
        """Handle keyboard shortcuts"""
        # This would need proper keyboard handling in a full implementation
        # For now, just return False to allow normal touch handling
        return False
    
    def set_callbacks(self, start_cb: Callable, stop_cb: Callable, play_cb: Callable, quit_cb: Callable = None):
        """Set callback functions for button actions"""
        self.on_start_recording = start_cb
        self.on_stop_recording = stop_cb
        self.on_play_audio = play_cb
        self.on_quit_app = quit_cb
        logger.info("UI callbacks set")
    
    def update_status(self, status: str):
        """Update the status label"""
        self.status_label.text = status
    
    def update_transcript(self, text: str):
        """Update the transcript display"""
        self.transcript_label.text = text
    
    def set_button_states(self, start_enabled: bool, stop_enabled: bool, play_enabled: bool):
        """Set button enabled/disabled states"""
        self.start_button.disabled = not start_enabled
        self.stop_button.disabled = not stop_enabled
        self.play_button.disabled = not play_enabled
    
    def show_session_history(self, sessions: List[Dict[str, Any]]):
        """Show session history in transcript area"""
        if sessions:
            history_text = f"Previous sessions: {len(sessions)}\n"
            history_text += f"Last session: {sessions[-1]['timestamp'][:19]}\n\n"
            history_text += "Start recording to see transcription here."
        else:
            history_text = "No previous sessions.\nStart recording to see transcription here."
        
        self.transcript_label.text = history_text


class SimpleRecordingApp(App):
    """
    Simple Recording Application
    
    Wraps the SimpleRecordingUI in a Kivy app for standalone operation
    or integration with the existing application architecture.
    """
    
    def __init__(self, audio_recorder=None, transcriber=None, **kwargs):
        """Initialize the app with audio recorder and transcriber"""
        super().__init__(**kwargs)
        self.title = "Silent Steno - Simple Recording"
        
        # Core components
        self.audio_recorder = audio_recorder
        self.transcriber = transcriber
        
        # UI component
        self.ui = None
        
        # Current session tracking
        self.current_session = None
        
        logger.info("SimpleRecordingApp initialized")
    
    def build(self):
        """Build the main UI"""
        self.ui = SimpleRecordingUI()
        
        # Set up callbacks
        self.ui.set_callbacks(
            start_cb=self._start_recording,
            stop_cb=self._stop_recording,
            play_cb=self._play_audio,
            quit_cb=self._quit_app
        )
        
        # Show initial session history
        if self.audio_recorder:
            sessions = self.audio_recorder.get_all_sessions()
            self.ui.show_session_history(sessions)
        
        return self.ui
    
    def _start_recording(self):
        """Start recording callback"""
        if not self.audio_recorder:
            self.ui.update_status("Audio recorder not available")
            return
        
        # Generate session ID
        import uuid
        session_id = str(uuid.uuid4())[:8]
        
        # Start recording
        if self.audio_recorder.start_recording(session_id):
            self.current_session = session_id
            self.ui.set_button_states(start_enabled=False, stop_enabled=True, play_enabled=False)
            self.ui.update_status("Recording... Press Stop to finish")
            logger.info(f"Recording started with session: {session_id}")
        else:
            self.ui.update_status("Failed to start recording")
            logger.error("Failed to start recording")
    
    def _stop_recording(self):
        """Stop recording callback"""
        if not self.audio_recorder:
            return
        
        # Stop recording
        recording_info = self.audio_recorder.stop_recording()
        
        if recording_info:
            self.ui.set_button_states(start_enabled=False, stop_enabled=False, play_enabled=True)
            self.ui.update_status(f"Recording complete: {recording_info['duration']:.1f} seconds")
            
            # Update transcript area with recording info
            self._show_recording_info(recording_info)
            
            # Start transcription if available
            if self.transcriber and self.transcriber.is_available():
                self._start_transcription(recording_info['wav_file'])
            else:
                # Reset audio recorder state for next recording (when no transcription)
                if self.audio_recorder:
                    self.audio_recorder.reset_to_idle()
                self.ui.set_button_states(start_enabled=True, stop_enabled=False, play_enabled=True)
                self.ui.update_status("Recording ready. Transcription not available.")
            
            logger.info(f"Recording stopped: {recording_info['wav_file']}")
        else:
            self.ui.update_status("Recording failed")
            self.ui.set_button_states(start_enabled=True, stop_enabled=False, play_enabled=False)
            logger.error("Recording stop failed")
    
    def _play_audio(self):
        """Play audio callback"""
        if not self.audio_recorder:
            return
        
        if self.audio_recorder.play_recording():
            self.ui.set_button_states(start_enabled=False, stop_enabled=False, play_enabled=False)
            self.ui.update_status("Playing audio...")
            
            # Re-enable buttons after playback
            Clock.schedule_once(self._finish_playback, 2.0)
            
            logger.info("Audio playback started")
        else:
            self.ui.update_status("No audio to play")
    
    def _finish_playback(self, dt):
        """Finish playback and reset buttons"""
        # Reset audio recorder state for next recording
        if self.audio_recorder:
            self.audio_recorder.reset_to_idle()
            
        self.ui.update_status("Playback complete. Ready for next recording.")
        self.ui.set_button_states(start_enabled=True, stop_enabled=False, play_enabled=True)
    
    def _quit_app(self):
        """Quit the application"""
        logger.info("Quitting application")
        
        # Stop any active recording
        if self.audio_recorder:
            recording_state = self.audio_recorder.get_recording_state()
            if recording_state == "recording":
                logger.info("Stopping active recording before quit")
                self.audio_recorder.stop_recording()
        
        # Stop the app
        self.stop()
    
    def _show_recording_info(self, recording_info: Dict[str, Any]):
        """Show recording information in transcript area"""
        info_text = f"""[{datetime.now().strftime('%H:%M:%S')}] Recording completed successfully.

Duration: {recording_info['duration']:.2f} seconds
Samples: {recording_info['samples']:,}
Sample Rate: {recording_info['sample_rate']:,} Hz
Channels: {recording_info['channels']}

Transcription: Loading..."""
        
        self.ui.update_transcript(info_text)
    
    def _start_transcription(self, wav_file: str):
        """Start transcription in background"""
        self.ui.update_status("Transcribing audio...")
        
        def transcribe_in_background():
            # Get current session ID for data integration
            session_id = None
            if self.audio_recorder:
                sessions = self.audio_recorder.get_all_sessions()
                if sessions:
                    session_id = sessions[-1].get('session_id')
            
            # Use integrated transcription method if session ID available
            if session_id and hasattr(self.transcriber, 'transcribe_and_update_session'):
                transcript = self.transcriber.transcribe_and_update_session(wav_file, session_id)
            else:
                transcript = self.transcriber.transcribe_audio(wav_file)
            
            Clock.schedule_once(lambda dt: self._update_transcript(transcript), 0)
        
        thread = threading.Thread(target=transcribe_in_background, daemon=True)
        thread.start()
    
    def _update_transcript(self, transcript: str):
        """Update transcript in UI"""
        current_text = self.ui.transcript_label.text
        
        if transcript and not transcript.startswith("Transcription error"):
            updated_text = current_text.replace("Transcription: Loading...", f"Transcription: {transcript}")
            self.ui.update_transcript(updated_text)
            
            # Update session with transcript
            if self.audio_recorder:
                sessions = self.audio_recorder.get_all_sessions()
                if sessions:
                    sessions[-1]['transcript'] = transcript
                    self.audio_recorder.sessions = sessions
                    self.audio_recorder._save_sessions()
            
            self.ui.update_status("Transcription complete. Ready for next recording.")
        else:
            updated_text = current_text.replace("Transcription: Loading...", f"Transcription: {transcript}")
            self.ui.update_transcript(updated_text)
            self.ui.update_status("Transcription error. Ready for next recording.")
        
        # Reset audio recorder state for next recording
        if self.audio_recorder:
            self.audio_recorder.reset_to_idle()
        
        # Re-enable start button
        self.ui.set_button_states(start_enabled=True, stop_enabled=False, play_enabled=True)
    
    def on_stop(self):
        """Cleanup when app stops"""
        if self.audio_recorder:
            recording_state = self.audio_recorder.get_recording_state()
            if recording_state == "recording":
                self.audio_recorder.stop_recording()
        
        return super().on_stop()


# Factory function to create the UI widget
def create_simple_recording_ui(**kwargs) -> SimpleRecordingUI:
    """Create a simple recording UI widget"""
    return SimpleRecordingUI(**kwargs)


# Factory function to create the app
def create_simple_recording_app(audio_recorder=None, transcriber=None) -> SimpleRecordingApp:
    """Create a simple recording app"""
    return SimpleRecordingApp(audio_recorder=audio_recorder, transcriber=transcriber)


if __name__ == "__main__":
    # Test the UI component
    print("Simple Recording UI Test")
    print("=" * 50)
    
    if KIVY_AVAILABLE:
        # Create a test app
        app = SimpleRecordingApp()
        print("✓ UI component created successfully")
        print("Run with audio_recorder and transcriber for full functionality")
    else:
        print("✗ Kivy not available - UI components cannot be tested")
        print("Install with: pip install kivy")