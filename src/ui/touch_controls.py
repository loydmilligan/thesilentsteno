"""
Touch-Optimized UI Controls and Widgets

This module provides touch-specific UI controls optimized for finger interaction
with visual and haptic feedback for The Silent Steno.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List, Union
from enum import Enum
import threading
import time
import logging
import math

try:
    from kivy.uix.button import Button
    from kivy.uix.slider import Slider
    from kivy.uix.switch import Switch
    from kivy.uix.widget import Widget
    from kivy.uix.label import Label
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.behaviors import ButtonBehavior
    from kivy.vector import Vector
    from kivy.clock import Clock
    from kivy.metrics import dp
    from kivy.event import EventDispatcher
    from kivy.animation import Animation
    from kivy.graphics import Color, Rectangle, Line, Ellipse
    from kivy.graphics.context_instructions import PushMatrix, PopMatrix, Rotate
except ImportError:
    raise ImportError("Kivy not available. Install with: pip install kivy")

logger = logging.getLogger(__name__)


class TouchControlState(Enum):
    """Touch control states"""
    NORMAL = "normal"
    PRESSED = "pressed"
    DISABLED = "disabled"
    HIGHLIGHTED = "highlighted"
    FOCUSED = "focused"


class FeedbackType(Enum):
    """Feedback types for touch interactions"""
    VISUAL = "visual"
    AUDIO = "audio"
    HAPTIC = "haptic"
    COMBINED = "combined"


@dataclass
class TouchConfig:
    """Configuration for touch controls"""
    min_touch_size: float = dp(44)  # Minimum touch target size
    touch_padding: float = dp(8)    # Extra padding around controls
    press_feedback_duration: float = 0.1  # Duration of press feedback
    animation_duration: float = 0.2  # Duration of animations
    ripple_duration: float = 0.4    # Duration of ripple effect
    enable_haptic: bool = True      # Enable haptic feedback
    enable_audio: bool = True       # Enable audio feedback
    enable_visual: bool = True      # Enable visual feedback
    long_press_duration: float = 0.8  # Long press threshold
    double_tap_interval: float = 0.3  # Double tap interval
    
    def __post_init__(self):
        """Validate configuration"""
        if self.min_touch_size < dp(32):
            raise ValueError("Touch size too small for accessibility")
        if self.press_feedback_duration < 0.05:
            raise ValueError("Feedback duration too short")


class TouchButton(ButtonBehavior, Widget):
    """Touch-optimized button with feedback"""
    
    def __init__(self, text: str = "", config: Optional[TouchConfig] = None, **kwargs):
        super().__init__(**kwargs)
        
        self.config = config or TouchConfig()
        self.text = text
        self.state = TouchControlState.NORMAL
        
        # Ensure minimum size
        self.size_hint = (None, None)
        self.size = (max(self.config.min_touch_size, self.width or self.config.min_touch_size),
                    max(self.config.min_touch_size, self.height or self.config.min_touch_size))
        
        # Touch tracking
        self.touch_start_time = 0
        self.touch_start_pos = None
        self.is_long_press = False
        self.tap_count = 0
        self.last_tap_time = 0
        
        # Visual elements
        self.background_color = [0.2, 0.2, 0.2, 1]
        self.text_color = [1, 1, 1, 1]
        self.press_color = [0.4, 0.4, 0.4, 1]
        self.disabled_color = [0.1, 0.1, 0.1, 0.5]
        
        # Setup graphics
        self._setup_graphics()
        
        # Bind events
        self.bind(size=self._update_graphics, pos=self._update_graphics)
        
        logger.debug(f"TouchButton created: {text}")
    
    def _setup_graphics(self):
        """Setup button graphics"""
        with self.canvas:
            Color(*self.background_color)
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)
            
        # Text label
        self.label = Label(
            text=self.text,
            font_size=dp(16),
            color=self.text_color,
            center=self.center
        )
        self.add_widget(self.label)
    
    def _update_graphics(self, *args):
        """Update graphics when size/position changes"""
        if hasattr(self, 'bg_rect'):
            self.bg_rect.pos = self.pos
            self.bg_rect.size = self.size
        
        if hasattr(self, 'label'):
            self.label.center = self.center
    
    def on_touch_down(self, touch):
        """Handle touch down event"""
        if not self.collide_point(*touch.pos):
            return False
        
        if self.state == TouchControlState.DISABLED:
            return False
        
        # Grab touch
        touch.grab(self)
        
        # Record touch info
        self.touch_start_time = time.time()
        self.touch_start_pos = touch.pos
        self.is_long_press = False
        
        # Change to pressed state
        self._set_state(TouchControlState.PRESSED)
        
        # Schedule long press check
        Clock.schedule_once(self._check_long_press, self.config.long_press_duration)
        
        # Provide feedback
        self._provide_feedback(FeedbackType.VISUAL, 'press_start')
        
        return True
    
    def on_touch_move(self, touch):
        """Handle touch move event"""
        if touch.grab_current is not self:
            return False
        
        # Check if still within button bounds
        if not self.collide_point(*touch.pos):
            if self.state == TouchControlState.PRESSED:
                self._set_state(TouchControlState.NORMAL)
        else:
            if self.state == TouchControlState.NORMAL:
                self._set_state(TouchControlState.PRESSED)
        
        return True
    
    def on_touch_up(self, touch):
        """Handle touch up event"""
        if touch.grab_current is not self:
            return False
        
        touch.ungrab(self)
        
        # Calculate touch duration and distance
        duration = time.time() - self.touch_start_time
        distance = 0
        if self.touch_start_pos:
            distance = Vector(touch.pos).distance(self.touch_start_pos)
        
        # Reset state
        self._set_state(TouchControlState.NORMAL)
        
        # Check for valid press
        if self.collide_point(*touch.pos) and distance < dp(20):
            if self.is_long_press:
                self._handle_long_press()
            else:
                self._handle_tap(duration)
        
        # Provide feedback
        self._provide_feedback(FeedbackType.VISUAL, 'press_end')
        
        return True
    
    def _check_long_press(self, dt):
        """Check for long press"""
        if self.state == TouchControlState.PRESSED:
            self.is_long_press = True
            self._provide_feedback(FeedbackType.HAPTIC, 'long_press')
    
    def _handle_tap(self, duration):
        """Handle tap gesture"""
        current_time = time.time()
        time_since_last = current_time - self.last_tap_time
        
        if time_since_last < self.config.double_tap_interval:
            self.tap_count += 1
        else:
            self.tap_count = 1
        
        self.last_tap_time = current_time
        
        # Schedule tap processing
        Clock.schedule_once(
            lambda dt: self._process_tap(),
            self.config.double_tap_interval
        )
    
    def _process_tap(self):
        """Process tap after double-tap timeout"""
        if self.tap_count == 1:
            self.dispatch('on_press')
            self._provide_feedback(FeedbackType.AUDIO, 'single_tap')
        elif self.tap_count >= 2:
            self.dispatch('on_double_tap')
            self._provide_feedback(FeedbackType.AUDIO, 'double_tap')
        
        self.tap_count = 0
    
    def _handle_long_press(self):
        """Handle long press gesture"""
        self.dispatch('on_long_press')
        self._provide_feedback(FeedbackType.COMBINED, 'long_press')
    
    def _set_state(self, new_state: TouchControlState):
        """Change button state"""
        if self.state == new_state:
            return
        
        old_state = self.state
        self.state = new_state
        
        # Update visual appearance
        self._update_appearance()
        
        # Dispatch state change
        self.dispatch('on_state_change', old_state, new_state)
    
    def _update_appearance(self):
        """Update button appearance based on state"""
        if self.state == TouchControlState.PRESSED:
            color = self.press_color
        elif self.state == TouchControlState.DISABLED:
            color = self.disabled_color
        else:
            color = self.background_color
        
        # Animate color change
        if hasattr(self, 'bg_rect'):
            anim = Animation(
                rgba=color,
                duration=self.config.animation_duration
            )
            anim.start(self.canvas.children[0])  # Color instruction
    
    def _provide_feedback(self, feedback_type: FeedbackType, event: str):
        """Provide user feedback"""
        if feedback_type == FeedbackType.VISUAL or feedback_type == FeedbackType.COMBINED:
            self._create_ripple_effect()
        
        # Dispatch feedback event for external handling
        self.dispatch('on_feedback_request', feedback_type, event)
    
    def _create_ripple_effect(self):
        """Create ripple visual effect"""
        if not self.config.enable_visual:
            return
        
        with self.canvas.after:
            Color(1, 1, 1, 0.3)
            ripple = Ellipse(pos=self.center, size=(0, 0))
        
        # Animate ripple
        anim = Animation(
            size=(self.width * 2, self.height * 2),
            pos=(self.x - self.width / 2, self.y - self.height / 2),
            duration=self.config.ripple_duration
        )
        anim.bind(on_complete=lambda *args: self.canvas.after.remove(ripple.parent))
        anim.start(ripple)
    
    def set_enabled(self, enabled: bool):
        """Enable or disable the button"""
        if enabled:
            self._set_state(TouchControlState.NORMAL)
        else:
            self._set_state(TouchControlState.DISABLED)
    
    def set_text(self, text: str):
        """Update button text"""
        self.text = text
        if hasattr(self, 'label'):
            self.label.text = text
    
    # Event definitions
    __events__ = ('on_press', 'on_double_tap', 'on_long_press', 'on_state_change', 'on_feedback_request')
    
    def on_press(self):
        """Called on button press"""
        pass
    
    def on_double_tap(self):
        """Called on double tap"""
        pass
    
    def on_long_press(self):
        """Called on long press"""
        pass
    
    def on_state_change(self, old_state, new_state):
        """Called when state changes"""
        pass
    
    def on_feedback_request(self, feedback_type, event):
        """Called when feedback is requested"""
        pass


class TouchSlider(Slider):
    """Touch-optimized slider with enhanced feedback"""
    
    def __init__(self, config: Optional[TouchConfig] = None, **kwargs):
        self.config = config or TouchConfig()
        
        # Ensure minimum size
        if 'size_hint' not in kwargs:
            kwargs['size_hint'] = (None, None)
        if 'size' not in kwargs:
            kwargs['size'] = (dp(200), self.config.min_touch_size)
        
        super().__init__(**kwargs)
        
        # Touch tracking
        self.is_dragging = False
        self.touch_start_pos = None
        
        # Visual enhancements
        self.cursor_size = (self.config.min_touch_size, self.config.min_touch_size)
        
        logger.debug("TouchSlider created")
    
    def on_touch_down(self, touch):
        """Enhanced touch down handling"""
        if self.collide_point(*touch.pos):
            self.is_dragging = True
            self.touch_start_pos = touch.pos
            
            # Provide feedback
            self.dispatch('on_feedback_request', FeedbackType.HAPTIC, 'slider_touch')
            
        return super().on_touch_down(touch)
    
    def on_touch_move(self, touch):
        """Enhanced touch move handling"""
        result = super().on_touch_move(touch)
        
        if self.is_dragging and hasattr(self, '_old_value'):
            if abs(self.value - self._old_value) > (self.max - self.min) * 0.05:
                # Provide feedback for significant value changes
                self.dispatch('on_feedback_request', FeedbackType.HAPTIC, 'slider_change')
                self._old_value = self.value
        
        return result
    
    def on_touch_up(self, touch):
        """Enhanced touch up handling"""
        if self.is_dragging:
            self.is_dragging = False
            self.dispatch('on_feedback_request', FeedbackType.AUDIO, 'slider_release')
        
        return super().on_touch_up(touch)
    
    def on_value(self, instance, value):
        """Handle value changes"""
        self._old_value = getattr(self, '_old_value', value)
        self.dispatch('on_value_changed', value)
    
    # Event definitions
    __events__ = ('on_feedback_request', 'on_value_changed')
    
    def on_feedback_request(self, feedback_type, event):
        """Called when feedback is requested"""
        pass
    
    def on_value_changed(self, value):
        """Called when value changes"""
        pass


class TouchSwitch(Switch):
    """Touch-optimized switch with enhanced feedback"""
    
    def __init__(self, config: Optional[TouchConfig] = None, **kwargs):
        self.config = config or TouchConfig()
        
        # Ensure minimum size
        if 'size_hint' not in kwargs:
            kwargs['size_hint'] = (None, None)
        if 'size' not in kwargs:
            kwargs['size'] = (dp(80), self.config.min_touch_size)
        
        super().__init__(**kwargs)
        
        logger.debug("TouchSwitch created")
    
    def on_touch_down(self, touch):
        """Enhanced touch down handling"""
        if self.collide_point(*touch.pos):
            # Provide immediate feedback
            self.dispatch('on_feedback_request', FeedbackType.VISUAL, 'switch_touch')
        
        return super().on_touch_down(touch)
    
    def on_active(self, instance, value):
        """Handle active state changes"""
        # Provide feedback for state change
        if value:
            self.dispatch('on_feedback_request', FeedbackType.COMBINED, 'switch_on')
        else:
            self.dispatch('on_feedback_request', FeedbackType.COMBINED, 'switch_off')
        
        self.dispatch('on_switch_changed', value)
    
    # Event definitions
    __events__ = ('on_feedback_request', 'on_switch_changed')
    
    def on_feedback_request(self, feedback_type, event):
        """Called when feedback is requested"""
        pass
    
    def on_switch_changed(self, active):
        """Called when switch state changes"""
        pass


class TouchGesture:
    """Touch gesture recognition for controls"""
    
    def __init__(self, config: TouchConfig):
        self.config = config
        self.active_touches = {}
        self.gesture_callbacks = {}
    
    def register_callback(self, gesture_type: str, callback: Callable):
        """Register callback for gesture type"""
        if gesture_type not in self.gesture_callbacks:
            self.gesture_callbacks[gesture_type] = []
        self.gesture_callbacks[gesture_type].append(callback)
    
    def process_touch_down(self, touch):
        """Process touch down for gesture recognition"""
        touch_id = id(touch)
        self.active_touches[touch_id] = {
            'start_pos': touch.pos,
            'start_time': time.time(),
            'current_pos': touch.pos
        }
    
    def process_touch_move(self, touch):
        """Process touch move for gesture recognition"""
        touch_id = id(touch)
        if touch_id in self.active_touches:
            self.active_touches[touch_id]['current_pos'] = touch.pos
    
    def process_touch_up(self, touch):
        """Process touch up for gesture recognition"""
        touch_id = id(touch)
        if touch_id not in self.active_touches:
            return
        
        touch_data = self.active_touches[touch_id]
        
        # Analyze gesture
        start_pos = Vector(touch_data['start_pos'])
        end_pos = Vector(touch_data['current_pos'])
        distance = start_pos.distance(end_pos)
        duration = time.time() - touch_data['start_time']
        
        if distance < dp(10) and duration < 0.3:
            self._trigger_gesture('tap', touch.pos)
        elif distance > self.config.min_touch_size and duration < 1.0:
            self._trigger_gesture('swipe', (touch_data['start_pos'], touch_data['current_pos']))
        
        del self.active_touches[touch_id]
    
    def _trigger_gesture(self, gesture_type: str, data):
        """Trigger gesture callbacks"""
        if gesture_type in self.gesture_callbacks:
            for callback in self.gesture_callbacks[gesture_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Gesture callback error: {e}")


def create_touch_button(text: str, config: Optional[TouchConfig] = None, **kwargs) -> TouchButton:
    """Factory function to create touch button"""
    return TouchButton(text=text, config=config, **kwargs)


def create_touch_slider(min_val: float = 0, max_val: float = 100, 
                       config: Optional[TouchConfig] = None, **kwargs) -> TouchSlider:
    """Factory function to create touch slider"""
    slider = TouchSlider(config=config, **kwargs)
    slider.min = min_val
    slider.max = max_val
    return slider


def create_touch_switch(config: Optional[TouchConfig] = None, **kwargs) -> TouchSwitch:
    """Factory function to create touch switch"""
    return TouchSwitch(config=config, **kwargs)


def create_touch_config_for_device(screen_size: tuple, dpi: float) -> TouchConfig:
    """Create touch configuration optimized for specific device"""
    # Calculate appropriate touch sizes based on device characteristics
    min_size = max(dp(44), dpi * 0.4)  # At least 44dp or 0.4 inches
    
    return TouchConfig(
        min_touch_size=min_size,
        touch_padding=min_size * 0.2,
        press_feedback_duration=0.1,
        animation_duration=0.2,
        enable_haptic=True,
        enable_audio=True,
        enable_visual=True
    )


if __name__ == '__main__':
    # Test touch controls
    config = TouchConfig()
    
    button = create_touch_button("Test Button", config)
    slider = create_touch_slider(0, 100, config)
    switch = create_touch_switch(config)
    
    print("Touch controls created successfully")
    print(f"Button size: {button.size}")
    print(f"Slider size: {slider.size}")
    print(f"Switch size: {switch.size}")