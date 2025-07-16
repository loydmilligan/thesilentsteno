#!/usr/bin/env python3
"""
Simple Silent Steno UI Demo - Basic version with minimal complexity
"""

import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from kivy.app import App
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.button import Button
    from kivy.uix.label import Label
    from kivy.clock import Clock
    from kivy.config import Config
    
    # Configure for touchscreen
    Config.set('graphics', 'width', '800')
    Config.set('graphics', 'height', '480')
    
    from src.ui.session_view import SessionView, SessionViewConfig
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install Kivy with: sudo apt install python3-kivy")
    sys.exit(1)


class SimpleDemo(App):
    def build(self):
        self.title = "Silent Steno Simple Demo"
        
        # Main layout
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        
        # Title
        title = Label(text='🎙️ Silent Steno Demo', font_size='24sp', size_hint_y=0.2)
        layout.add_widget(title)
        
        # Start demo button
        start_btn = Button(
            text='Start Live Session Demo',
            font_size='18sp',
            size_hint_y=0.3
        )
        start_btn.bind(on_release=self.start_demo)
        layout.add_widget(start_btn)
        
        # Status
        self.status_label = Label(
            text='Ready to start demo...',
            font_size='16sp',
            size_hint_y=0.2
        )
        layout.add_widget(self.status_label)
        
        # Exit button
        exit_btn = Button(
            text='Exit',
            font_size='16sp',
            size_hint_y=0.3
        )
        exit_btn.bind(on_release=self.stop)
        layout.add_widget(exit_btn)
        
        return layout
    
    def start_demo(self, button):
        """Start the session demo."""
        try:
            self.status_label.text = 'Starting demo...'
            
            # Create session view with demo mode
            config = SessionViewConfig(
                enable_demo_mode=True,
                demo_update_rate=2.0
            )
            
            session_view = SessionView(config)
            
            # Replace current layout with session view
            self.root.clear_widgets()
            self.root.add_widget(session_view)
            
            # Start demo after short delay
            Clock.schedule_once(lambda dt: session_view.start_demo(), 1.0)
            
            print("✅ Demo started successfully!")
            
        except Exception as e:
            self.status_label.text = f'Error: {str(e)}'
            print(f"❌ Demo error: {e}")


if __name__ == '__main__':
    try:
        print("🚀 Starting Simple Silent Steno Demo...")
        SimpleDemo().run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("✅ Demo finished")