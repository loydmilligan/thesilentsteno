#!/usr/bin/env python3
"""
Silent Steno Live Session Demo

This demo showcases the complete live session interface with all components
integrated in a single application. Perfect for testing the UI on Pi 5 touchscreen.
"""

import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from kivy.app import App
    from kivy.uix.screenmanager import ScreenManager, Screen
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.label import Label
    from kivy.uix.button import Button
    from kivy.clock import Clock
    from kivy.logger import Logger
    from kivy.config import Config
    
    # Configure Kivy for touchscreen
    Config.set('graphics', 'width', '800')
    Config.set('graphics', 'height', '480')
    Config.set('graphics', 'borderless', '1')
    Config.set('graphics', 'fullscreen', '0')  # Set to '1' for fullscreen on device
    
    from src.ui.session_view import SessionView, SessionViewConfig
    from src.ui.transcription_display import create_transcription_display
    from src.ui.audio_visualizer import create_audio_visualizer, VisualizerType
    from src.ui.session_controls import create_session_controls
    from src.ui.status_indicators import create_status_indicators, ConnectionStatus, StatusLevel
    from src.ui.themes import create_theme_manager
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory")
    print("and that Kivy is installed: sudo apt install python3-kivy")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemoMenuScreen(Screen):
    """Main menu for demo selection."""
    
    def __init__(self, **kwargs):
        super().__init__(name='menu', **kwargs)
        
        layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        
        # Title
        title = Label(
            text='🎙️ Silent Steno Live Session Demo',
            font_size='32sp',
            size_hint_y=0.2,
            bold=True
        )
        layout.add_widget(title)
        
        # Subtitle
        subtitle = Label(
            text='Touch-Optimized Meeting Recorder Interface',
            font_size='18sp',
            size_hint_y=0.1,
            color=(0.8, 0.8, 0.8, 1)
        )
        layout.add_widget(subtitle)
        
        # Demo buttons
        button_layout = BoxLayout(orientation='vertical', spacing=10)
        
        demos = [
            ('Full Live Session', 'full_session', '🎯 Complete interface with all components'),
            ('Session Controls', 'controls', '🎛️ Start/Stop/Pause controls'),
            ('Transcript Display', 'transcript', '📝 Live transcript with speakers'),
            ('Audio Visualizer', 'visualizer', '📊 Real-time audio levels'),
            ('Status Indicators', 'status', '📡 Connection and system status'),
        ]
        
        for title, screen_name, description in demos:
            btn = Button(
                text=f'{title}\n{description}',
                font_size='16sp',
                size_hint_y=None,
                height=80
            )
            btn.bind(on_release=lambda x, name=screen_name: setattr(self.parent, 'current', name))
            button_layout.add_widget(btn)
        
        layout.add_widget(button_layout)
        
        # Instructions
        instructions = Label(
            text='💡 Tip: Use touch gestures and try all the controls!\n'
                 '🔄 Press ESC or use device back button to return to menu',
            font_size='14sp',
            size_hint_y=0.15,
            color=(0.7, 0.7, 0.7, 1)
        )
        layout.add_widget(instructions)
        
        self.add_widget(layout)


class FullSessionDemoScreen(Screen):
    """Complete live session interface demo."""
    
    def __init__(self, **kwargs):
        super().__init__(name='full_session', **kwargs)
        
        # Create session view with demo mode enabled
        config = SessionViewConfig(
            enable_demo_mode=True,
            demo_update_rate=2.0,
            show_transcript=True,
            show_audio_visualizer=True,
            show_timer=True
        )
        
        self.session_view = SessionView(config)
        self.add_widget(self.session_view)
    
    def on_enter(self):
        """Start demo when screen is entered."""
        Clock.schedule_once(lambda dt: self.session_view.start_demo(), 1.0)
    
    def on_leave(self):
        """Stop demo when leaving screen."""
        if hasattr(self.session_view, 'view_model'):
            self.session_view.view_model.stop_session()


class ComponentDemoScreen(Screen):
    """Base class for individual component demos."""
    
    def __init__(self, name, component_widget, **kwargs):
        super().__init__(name=name, **kwargs)
        
        layout = BoxLayout(orientation='vertical')
        
        # Back button
        back_btn = Button(
            text='← Back to Menu',
            size_hint_y=0.1,
            font_size='16sp'
        )
        back_btn.bind(on_release=lambda x: setattr(self.parent, 'current', 'menu'))
        layout.add_widget(back_btn)
        
        # Component
        layout.add_widget(component_widget)
        
        self.add_widget(layout)


class SilentStenoDemoApp(App):
    """Main demo application."""
    
    def build(self):
        # Set window title
        self.title = "Silent Steno Live Session Demo"
        
        # Create screen manager
        sm = ScreenManager()
        
        # Add menu screen
        sm.add_widget(DemoMenuScreen())
        
        # Add full session demo
        sm.add_widget(FullSessionDemoScreen())
        
        # Add component demos
        
        # Session Controls Demo
        controls = create_session_controls()
        controls_screen = ComponentDemoScreen('controls', controls)
        sm.add_widget(controls_screen)
        
        # Transcript Demo
        transcript = create_transcription_display()
        # Add some sample entries
        Clock.schedule_once(lambda dt: self._populate_transcript(transcript), 1.0)
        transcript_screen = ComponentDemoScreen('transcript', transcript)
        sm.add_widget(transcript_screen)
        
        # Audio Visualizer Demo
        visualizer = create_audio_visualizer(VisualizerType.BARS)
        Clock.schedule_interval(lambda dt: self._update_visualizer(visualizer), 1/30.0)
        visualizer_screen = ComponentDemoScreen('visualizer', visualizer)
        sm.add_widget(visualizer_screen)
        
        # Status Indicators Demo
        status = create_status_indicators()
        Clock.schedule_once(lambda dt: self._setup_status_demo(status), 1.0)
        status_screen = ComponentDemoScreen('status', status)
        sm.add_widget(status_screen)
        
        # Bind keyboard events
        from kivy.core.window import Window
        Window.bind(on_keyboard=self._on_keyboard)
        
        logger.info("🚀 Silent Steno Demo started!")
        return sm
    
    def _on_keyboard(self, window, key, *args):
        """Handle keyboard events."""
        if key == 27:  # ESC key
            # Return to menu
            if hasattr(self.root, 'current') and self.root.current != 'menu':
                self.root.current = 'menu'
                return True
        return False
    
    def _populate_transcript(self, transcript):
        """Add sample transcript entries."""
        sample_entries = [
            ("Alice", "Welcome everyone to today's product planning meeting."),
            ("Bob", "Thanks Alice. I've prepared the quarterly roadmap for review."),
            ("Charlie", "Great! I'm excited to see what we have planned for Q2."),
            ("Diana", "Before we start, did everyone get a chance to review the user feedback?"),
            ("Alice", "Yes, the feedback on the mobile app has been very positive."),
            ("Bob", "The new onboarding flow increased retention by 23%."),
            ("Charlie", "That's fantastic! Should we expand this to the web platform?"),
            ("Diana", "I think that's a great idea. Let's add it to the roadmap."),
        ]
        
        for i, (speaker, text) in enumerate(sample_entries):
            Clock.schedule_once(
                lambda dt, s=speaker, t=text: transcript.add_transcript_entry(s.lower(), s, t),
                i * 2.0
            )
    
    def _update_visualizer(self, visualizer):
        """Update audio visualizer with simulated data."""
        import random
        import math
        
        # Generate realistic audio levels
        time_factor = Clock.get_time()
        levels = []
        
        for i in range(8):
            # Create some variation across frequency bands
            base_level = 0.3 + 0.4 * math.sin(time_factor * 2 + i * 0.5)
            noise = random.uniform(-0.1, 0.1)
            level = max(0.0, min(0.9, base_level + noise))
            levels.append(level)
        
        visualizer.update_levels(levels)
    
    def _setup_status_demo(self, status):
        """Set up status indicators demo."""
        # Simulate different statuses
        status.set_bluetooth_status(ConnectionStatus.CONNECTED)
        status.set_recording_status(True)
        status.set_battery_level(85.0)
        status.set_storage_usage(45.0)
        status.set_system_metrics(65.0, 42.0, 55.0)
        
        # Add some status messages
        status.add_status_message(StatusLevel.SUCCESS, "Demo mode active")
        status.add_status_message(StatusLevel.INFO, "All systems operational")
        
        # Schedule status updates
        Clock.schedule_interval(lambda dt: self._update_status_demo(status), 5.0)
    
    def _update_status_demo(self, status):
        """Update status demo with changing values."""
        import random
        
        # Simulate changing battery
        battery = max(20.0, status.battery_percentage - random.uniform(0, 2))
        status.set_battery_level(battery)
        
        # Simulate changing storage
        storage = min(95.0, status.storage_percentage + random.uniform(0, 1))
        status.set_storage_usage(storage)
        
        # Simulate changing system metrics
        cpu = random.uniform(40, 80)
        memory = random.uniform(30, 70)
        temp = random.uniform(45, 75)
        status.set_system_metrics(cpu, memory, temp)
    
    def on_stop(self):
        """Clean up when app stops."""
        logger.info("👋 Silent Steno Demo stopped")


if __name__ == '__main__':
    try:
        print("🎙️ Starting Silent Steno Live Session Demo...")
        print("💡 This demo showcases the complete touch UI interface")
        print("🖐️ Use touch gestures and try all the controls!")
        print()
        
        SilentStenoDemoApp().run()
        
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("✅ Demo finished")