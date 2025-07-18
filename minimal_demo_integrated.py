#!/usr/bin/env python3
"""
The Silent Steno - Minimal Demo (Integrated)
Walking Skeleton with Production Architecture Integration

This version uses the WalkingSkeletonAdapter to bridge between the simple
prototype and the comprehensive production system. It maintains the same
simple UI while leveraging production components when available.
"""

import kivy
kivy.require('2.0.0')

from kivy.app import App
from kivy.config import Config
import os
import sys
import logging

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

class MinimalDemoIntegrated(App):
    """
    Minimal Silent Steno Demo Application (Integrated)
    
    Uses the WalkingSkeletonAdapter to provide seamless integration
    between the simple prototype and production architecture.
    """
    
    def __init__(self):
        super().__init__()
        self.title = "Silent Steno - Minimal Demo (Integrated)"
        
        # Import required modules
        sys.path.append('/home/mmariani/projects/thesilentsteno')
        from src.integration.walking_skeleton_adapter import create_walking_skeleton_adapter
        from src.ui.simple_recording_ui import SimpleRecordingUI
        
        # Configuration
        config = {
            'storage_root': 'demo_sessions',
            'whisper_model': 'base',
            'quality_preset': 'balanced',
            'audio': {
                'target_latency': 40.0,
                'input_device': None,  # Auto-detect
                'output_device': None   # Auto-detect
            }
        }
        
        # Determine whether to use production components
        use_production = self._check_production_availability()
        
        # Create adapter
        self.adapter = create_walking_skeleton_adapter(
            use_production=use_production,
            config=config
        )
        
        # Initialize adapter
        try:
            self.adapter.initialize()
            logger.info(f"Adapter initialized in {'production' if use_production else 'skeleton'} mode")
        except Exception as e:
            logger.error(f"Error initializing adapter: {e}")
        
        # Create UI
        self.ui = SimpleRecordingUI()
        
        # Connect UI to adapter
        self._connect_ui_to_adapter()
        
        logger.info("MinimalDemoIntegrated initialized")
    
    def _check_production_availability(self) -> bool:
        """Check if production components are available"""
        try:
            # Try importing production modules
            import src.audio
            import src.recording
            import src.ai
            import src.system
            logger.info("Production components available")
            return True
        except ImportError:
            logger.info("Production components not available, using skeleton mode")
            return False
    
    def _connect_ui_to_adapter(self):
        """Connect UI callbacks to adapter methods"""
        # Override UI callbacks to use adapter
        def on_start_recording():
            session_id = self.adapter.start_recording()
            if session_id:
                self.ui.update_status(f"Recording... (Session: {session_id[:8]})")
                self.ui.start_button.disabled = True
                self.ui.stop_button.disabled = False
                self.ui.play_button.disabled = True
                self.ui.transcript_label.text = "Recording in progress... Speak into the microphone."
            else:
                self.ui.update_status("Failed to start recording")
        
        def on_stop_recording():
            info = self.adapter.stop_recording()
            if info:
                duration = info.get('duration', 0)
                self.ui.update_status(f"Recording stopped. Duration: {duration:.1f}s")
                self.ui.stop_button.disabled = True
                self.ui.play_button.disabled = False
                self.ui.transcript_label.text = "Transcribing..."
                
                # Start transcription
                self.ui.schedule_once(lambda dt: self._transcribe_recording(), 0.5)
            else:
                self.ui.update_status("Recording stopped but no audio detected. Check microphone.")
                
                # Reset buttons to allow retry
                self.ui.start_button.disabled = False
                self.ui.stop_button.disabled = True
                self.ui.play_button.disabled = True
                
                # Reset adapter state
                self.adapter.reset_to_idle()
        
        def on_play_recording():
            if self.adapter.play_recording():
                self.ui.update_status("Playing recording...")
            else:
                self.ui.update_status("Failed to play recording")
        
        def on_quit():
            self.adapter.shutdown()
            self.stop()
        
        # Set callbacks
        self.ui.start_button.bind(on_press=lambda x: on_start_recording())
        self.ui.stop_button.bind(on_press=lambda x: on_stop_recording())
        self.ui.play_button.bind(on_press=lambda x: on_play_recording())
        self.ui.quit_button.bind(on_press=lambda x: on_quit())
        
        # Set adapter callbacks
        self.adapter.add_state_callback(self._on_state_change)
        self.adapter.add_transcription_callback(self._on_transcription)
        
        # Store last analysis for display
        self.last_analysis = None
    
    def _transcribe_recording(self):
        """Transcribe the current recording"""
        transcript = self.adapter.transcribe_recording()
        
        if transcript:
            display_text = f"Transcript: {transcript}"
            
            # Add analysis information if available
            if self.last_analysis:
                analysis = self.last_analysis
                display_text += f"\n\n[b]AI Analysis:[/b]"
                display_text += f"\n[color=00ff00]• Word count:[/color] {analysis.get('word_count', 0)}"
                display_text += f"\n[color=00ff00]• Sentiment:[/color] {analysis.get('sentiment', 'neutral')}"
                
                topics = analysis.get('topics', [])
                if topics:
                    topics_str = ', '.join(topics)
                    display_text += f"\n[color=00ff00]• Topics:[/color] {topics_str}"
                
                action_items = analysis.get('action_items', [])
                if action_items:
                    display_text += f"\n[color=00ff00]• Action items:[/color] {len(action_items)}"
                    for i, item in enumerate(action_items[:3]):  # Show first 3
                        display_text += f"\n  - {item[:50]}..."
                
                questions = analysis.get('questions', [])
                if questions:
                    display_text += f"\n[color=00ff00]• Questions:[/color] {len(questions)}"
                
                summary = analysis.get('summary', '')
                if summary:
                    display_text += f"\n[color=00ff00]• Summary:[/color] {summary[:150]}..."
            
            self.ui.transcript_label.text = display_text
            self.ui.update_status("Transcription and analysis complete!")
            
            # Reset to idle after transcription
            self.adapter.reset_to_idle()
            self.ui.start_button.disabled = False
        else:
            self.ui.transcript_label.text = "Transcription failed"
            self.ui.update_status("Transcription failed")
    
    def _on_state_change(self, state):
        """Handle adapter state changes"""
        logger.info(f"Adapter state changed: {state}")
        
        # Update UI based on state
        if state == "idle":
            self.ui.start_button.disabled = False
            self.ui.stop_button.disabled = True
            self.ui.play_button.disabled = True
        elif state == "recording":
            self.ui.start_button.disabled = True
            self.ui.stop_button.disabled = False
            self.ui.play_button.disabled = True
        elif state == "stopped":
            self.ui.start_button.disabled = True
            self.ui.stop_button.disabled = True
            self.ui.play_button.disabled = False
    
    def _on_transcription(self, result):
        """Handle transcription completion"""
        if isinstance(result, dict):
            # Enhanced result with analysis
            transcript = result.get('transcript', '')
            analysis = result.get('analysis', {})
            self.last_analysis = analysis
            logger.info(f"Enhanced transcription received: {len(transcript)} characters")
        else:
            # Simple transcript string
            transcript = result
            self.last_analysis = None
            logger.info(f"Simple transcription received: {len(transcript)} characters")
    
    def build(self):
        """Build the UI"""
        # Add system status to UI
        def update_system_status(dt):
            status = self.adapter.get_system_status()
            mode = status.get('mode', 'unknown')
            state = status.get('recording_state', 'unknown')
            
            # Update window title with mode
            self.title = f"Silent Steno - Minimal Demo ({mode.capitalize()} Mode)"
            
            # Update status if needed
            if hasattr(self, '_last_mode') and self._last_mode != mode:
                self.ui.update_status(f"Running in {mode} mode")
            self._last_mode = mode
        
        # Schedule periodic status updates
        from kivy.clock import Clock
        Clock.schedule_interval(update_system_status, 5.0)
        
        return self.ui
    
    def on_stop(self):
        """Cleanup when app stops"""
        logger.info("Shutting down application...")
        self.adapter.shutdown()


if __name__ == '__main__':
    logger.info("Starting The Silent Steno - Minimal Demo (Integrated)")
    
    # Create and run the integrated app
    app = MinimalDemoIntegrated()
    
    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Application shutting down")