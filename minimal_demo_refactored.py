#!/usr/bin/env python3
"""
The Silent Steno - Minimal Demo (Refactored)
Walking Skeleton Implementation with Extracted Modules

This version uses the extracted modules:
1. SimpleAudioRecorder - for recording functionality
2. SimpleTranscriber - for transcription functionality  
3. SimpleRecordingUI - for UI components

Complete end-to-end flow with modular architecture.
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

class MinimalDemoRefactored(App):
    """
    Minimal Silent Steno Demo Application (Refactored)
    
    Uses extracted modules for all functionality:
    - SimpleAudioRecorder for recording
    - SimpleTranscriber for transcription  
    - SimpleRecordingApp for UI
    """
    
    def __init__(self):
        super().__init__()
        self.title = "Silent Steno - Minimal Demo (Refactored)"
        
        # Import the extracted modules
        sys.path.append('/home/mmariani/projects/thesilentsteno/src')
        from recording.simple_audio_recorder import SimpleAudioRecorder
        from ai.simple_transcriber import SimpleTranscriber
        from ui.simple_recording_ui import SimpleRecordingApp
        from data.integration_adapter import DataIntegrationAdapter
        
        # Initialize data integration adapter
        self.data_adapter = DataIntegrationAdapter("demo_sessions/sessions.json", use_database=True)
        
        # Initialize core components with data integration
        self.audio_recorder = SimpleAudioRecorder("demo_sessions")
        self.transcriber = SimpleTranscriber(backend="cpu", model_name="base", data_adapter=self.data_adapter)
        
        # Initialize UI app (this handles all UI logic)
        self.ui_app = SimpleRecordingApp(
            audio_recorder=self.audio_recorder,
            transcriber=self.transcriber
        )
        
        logger.info("MinimalDemoRefactored initialized with all extracted modules")
    
    def build(self):
        """Build the UI using the extracted UI component"""
        return self.ui_app.build()
    
    def on_stop(self):
        """Cleanup when app stops"""
        return self.ui_app.on_stop()


if __name__ == '__main__':
    logger.info("Starting The Silent Steno - Minimal Demo (Refactored)")
    
    # Create and run the refactored app
    app = MinimalDemoRefactored()
    
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