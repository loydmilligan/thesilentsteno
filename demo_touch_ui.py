#!/usr/bin/env python3
"""
Touch UI Demo for The Silent Steno

This demo showcases the touch UI framework components:
- Main window with responsive layout
- Touch-optimized controls (buttons, sliders, switches)
- Theme system (dark/light modes)
- Visual feedback system
- Navigation system

Run with: python3 demo_touch_ui.py
"""

import sys
import os
sys.path.insert(0, '.')

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen
from kivy.clock import Clock

from src.ui.main_window import create_main_window, WindowConfig
from src.ui.touch_controls import create_touch_button, create_touch_slider, create_touch_switch, TouchConfig
from src.ui.themes import create_theme_manager, ThemeConfig
from src.ui.feedback_manager import create_feedback_manager, FeedbackConfig, FeedbackEvent


class TouchUIDemo(App):
    """Demo application for touch UI framework"""
    
    def build(self):
        """Build the demo application"""
        self.title = "The Silent Steno - Touch UI Demo"
        
        # Create configurations
        self.window_config = WindowConfig(
            width=800,
            height=600,
            fullscreen=False,  # For testing
            theme="dark"
        )
        
        self.touch_config = TouchConfig(
            min_touch_size=50,  # Slightly larger for demo
            enable_haptic=False,  # Disable for desktop testing
            enable_audio=False,
            enable_visual=True
        )
        
        # Create theme and feedback managers
        self.theme_manager = create_theme_manager()
        self.feedback_manager = create_feedback_manager()
        
        # Create main layout
        main_layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        
        # Title
        title = Label(
            text="The Silent Steno Touch UI Demo",
            font_size='24sp',
            size_hint=(1, 0.1),
            color=(1, 1, 1, 1)
        )
        main_layout.add_widget(title)
        
        # Demo buttons section
        button_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.2), spacing=10)
        
        # Create demo buttons
        btn1 = create_touch_button("Touch Me!", self.touch_config)
        btn1.bind(on_press=self.on_button_press)
        btn1.bind(on_feedback_request=self.on_feedback_request)
        
        btn2 = create_touch_button("Theme Toggle", self.touch_config)
        btn2.bind(on_press=self.toggle_theme)
        
        btn3 = create_touch_button("Visual Test", self.touch_config)
        btn3.bind(on_press=self.visual_feedback_test)
        
        button_layout.add_widget(btn1)
        button_layout.add_widget(btn2)
        button_layout.add_widget(btn3)
        main_layout.add_widget(button_layout)
        
        # Demo slider section
        slider_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.3), spacing=5)
        
        slider_label = Label(
            text="Touch Slider (0-100):",
            font_size='16sp',
            size_hint=(1, 0.3)
        )
        slider_layout.add_widget(slider_label)
        
        self.demo_slider = create_touch_slider(0, 100, self.touch_config)
        self.demo_slider.value = 50
        self.demo_slider.bind(on_value_changed=self.on_slider_change)
        slider_layout.add_widget(self.demo_slider)
        
        self.slider_value_label = Label(
            text="Value: 50",
            font_size='14sp',
            size_hint=(1, 0.3)
        )
        slider_layout.add_widget(self.slider_value_label)
        
        main_layout.add_widget(slider_layout)
        
        # Demo switch section
        switch_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.2), spacing=10)
        
        switch_label = Label(
            text="Touch Switch:",
            font_size='16sp',
            size_hint=(0.7, 1)
        )
        switch_layout.add_widget(switch_label)
        
        self.demo_switch = create_touch_switch(self.touch_config)
        self.demo_switch.bind(on_switch_changed=self.on_switch_change)
        switch_layout.add_widget(self.demo_switch)
        
        main_layout.add_widget(switch_layout)
        
        # Status section
        self.status_label = Label(
            text="Ready - Touch the controls above!",
            font_size='14sp',
            size_hint=(1, 0.2),
            color=(0.8, 0.8, 0.8, 1)
        )
        main_layout.add_widget(self.status_label)
        
        return main_layout
    
    def on_button_press(self, button):
        """Handle button press"""
        button_text = button.text if hasattr(button, 'text') else "Button"
        self.status_label.text = f"Button pressed: {button_text}"
        
        # Trigger feedback
        self.feedback_manager.provide_feedback(
            FeedbackEvent.BUTTON_PRESS,
            button
        )
        
        print(f"✅ Button pressed: {button_text}")
    
    def on_slider_change(self, slider, value):
        """Handle slider value change"""
        self.slider_value_label.text = f"Value: {int(value)}"
        self.status_label.text = f"Slider changed to: {int(value)}"
        print(f"✅ Slider value: {int(value)}")
    
    def on_switch_change(self, switch, active):
        """Handle switch state change"""
        state = "ON" if active else "OFF"
        self.status_label.text = f"Switch turned {state}"
        print(f"✅ Switch: {state}")
    
    def toggle_theme(self, button):
        """Toggle between dark and light themes"""
        current_theme = self.theme_manager.get_current_theme()
        if current_theme and current_theme.name == "Dark":
            self.theme_manager.set_theme("Light")
            self.status_label.text = "Switched to Light theme"
        else:
            self.theme_manager.set_theme("Dark")
            self.status_label.text = "Switched to Dark theme"
        
        print("✅ Theme toggled")
    
    def visual_feedback_test(self, button):
        """Test visual feedback effects"""
        self.feedback_manager.create_ripple(button, button.center)
        self.status_label.text = "Visual feedback triggered!"
        print("✅ Visual feedback test")
    
    def on_feedback_request(self, control, feedback_type, event):
        """Handle feedback requests from controls"""
        print(f"📱 Feedback request: {feedback_type.value} for {event}")
    
    def on_start(self):
        """Called when app starts"""
        print("🎯 Touch UI Demo started!")
        print("💡 Try touching the buttons, moving the slider, and toggling the switch")
        print("🎨 Use 'Theme Toggle' to switch between dark and light modes")
        
        # Schedule a welcome message
        Clock.schedule_once(self.show_welcome, 1.0)
    
    def show_welcome(self, dt):
        """Show welcome message"""
        self.status_label.text = "Welcome! Touch UI framework is ready 🎉"


if __name__ == '__main__':
    print("🚀 Starting The Silent Steno Touch UI Demo...")
    print("📱 This demo showcases touch-optimized UI components")
    print("🎛️  Features: responsive controls, theming, visual feedback")
    print()
    
    try:
        # Run the demo app
        TouchUIDemo().run()
        
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("✅ Touch UI Demo finished")