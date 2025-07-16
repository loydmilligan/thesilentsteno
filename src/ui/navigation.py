"""
Touch-Optimized Navigation System

This module provides navigation management for touchscreen interface
with gesture support and screen transitions for The Silent Steno.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List, Tuple
from enum import Enum
import threading
import time
import logging
import math

try:
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.button import Button
    from kivy.uix.label import Label
    from kivy.uix.widget import Widget
    from kivy.uix.screenmanager import Screen
    from kivy.vector import Vector
    from kivy.clock import Clock
    from kivy.metrics import dp
    from kivy.event import EventDispatcher
    from kivy.animation import Animation
    from kivy.graphics import Color, Rectangle, Line
except ImportError:
    raise ImportError("Kivy not available. Install with: pip install kivy")

logger = logging.getLogger(__name__)


class NavigationState(Enum):
    """Navigation system states"""
    IDLE = "idle"
    NAVIGATING = "navigating"
    GESTURE_DETECTING = "gesture_detecting"
    ANIMATION = "animation"
    DISABLED = "disabled"


class GestureType(Enum):
    """Touch gesture types"""
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    TAP = "tap"
    LONG_PRESS = "long_press"
    PINCH = "pinch"
    DOUBLE_TAP = "double_tap"


@dataclass
class NavigationConfig:
    """Configuration for navigation system"""
    gesture_threshold: float = dp(50)  # Minimum distance for swipe gestures
    swipe_velocity_threshold: float = 200  # Minimum velocity for swipe detection
    long_press_duration: float = 0.8  # Seconds for long press
    double_tap_interval: float = 0.3  # Maximum time between double taps
    animation_duration: float = 0.3  # Screen transition duration
    enable_gestures: bool = True
    enable_back_gesture: bool = True
    back_gesture_zone_width: float = dp(30)  # Width of back gesture zone
    navbar_height: float = dp(50)
    enable_navbar: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if self.gesture_threshold < dp(20):
            raise ValueError("Gesture threshold too small")
        if self.animation_duration < 0.1 or self.animation_duration > 2.0:
            raise ValueError("Animation duration out of range")


@dataclass
class Screen:
    """Enhanced screen definition for navigation"""
    name: str
    title: str
    widget: Optional[Widget] = None
    back_enabled: bool = True
    gestures_enabled: bool = True
    navbar_visible: bool = True
    transition_type: str = "slide"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize screen after creation"""
        if not self.name:
            raise ValueError("Screen name required")


class GestureDetector(EventDispatcher):
    """Gesture detection for touch navigation"""
    
    def __init__(self, config: NavigationConfig):
        super().__init__()
        self.config = config
        self.touches = {}
        self.gesture_start_time = 0
        self.last_tap_time = 0
        self.tap_count = 0
        
    def on_touch_down(self, touch):
        """Handle touch down for gesture detection"""
        if not self.config.enable_gestures:
            return False
        
        touch_id = id(touch)
        self.touches[touch_id] = {
            'start_pos': touch.pos,
            'current_pos': touch.pos,
            'start_time': time.time(),
            'moved': False
        }
        
        self.gesture_start_time = time.time()
        
        # Start long press detection
        if self.config.long_press_duration > 0:
            Clock.schedule_once(
                lambda dt: self._check_long_press(touch_id),
                self.config.long_press_duration
            )
        
        return True
    
    def on_touch_move(self, touch):
        """Handle touch move for gesture detection"""
        touch_id = id(touch)
        if touch_id not in self.touches:
            return False
        
        self.touches[touch_id]['current_pos'] = touch.pos
        
        # Calculate movement distance
        start_pos = self.touches[touch_id]['start_pos']
        distance = Vector(touch.pos).distance(start_pos)
        
        if distance > dp(10):  # Moved significantly
            self.touches[touch_id]['moved'] = True
        
        return True
    
    def on_touch_up(self, touch):
        """Handle touch up for gesture detection"""
        touch_id = id(touch)
        if touch_id not in self.touches:
            return False
        
        touch_data = self.touches[touch_id]
        current_time = time.time()
        duration = current_time - touch_data['start_time']
        
        start_pos = Vector(touch_data['start_pos'])
        end_pos = Vector(touch_data['current_pos'])
        distance = start_pos.distance(end_pos)
        
        # Determine gesture type
        if not touch_data['moved'] and duration < 0.3:
            self._handle_tap(touch, current_time)
        elif distance > self.config.gesture_threshold:
            self._handle_swipe(start_pos, end_pos, duration)
        
        # Clean up
        del self.touches[touch_id]
        return True
    
    def _handle_tap(self, touch, tap_time):
        """Handle tap gesture"""
        time_since_last = tap_time - self.last_tap_time
        
        if time_since_last < self.config.double_tap_interval:
            self.tap_count += 1
        else:
            self.tap_count = 1
        
        self.last_tap_time = tap_time
        
        # Schedule double tap check
        Clock.schedule_once(
            lambda dt: self._process_tap_gesture(touch),
            self.config.double_tap_interval
        )
    
    def _process_tap_gesture(self, touch):
        """Process tap gesture after double-tap timeout"""
        if self.tap_count == 1:
            self.dispatch('on_gesture', GestureType.TAP, touch.pos)
        elif self.tap_count >= 2:
            self.dispatch('on_gesture', GestureType.DOUBLE_TAP, touch.pos)
        
        self.tap_count = 0
    
    def _handle_swipe(self, start_pos, end_pos, duration):
        """Handle swipe gesture"""
        delta = end_pos - start_pos
        velocity = delta.length() / duration
        
        if velocity < self.config.swipe_velocity_threshold:
            return  # Too slow to be a swipe
        
        # Determine swipe direction
        angle = math.atan2(delta.y, delta.x)
        angle_deg = math.degrees(angle)
        
        if -45 <= angle_deg <= 45:
            gesture = GestureType.SWIPE_RIGHT
        elif 45 < angle_deg <= 135:
            gesture = GestureType.SWIPE_UP
        elif -135 <= angle_deg < -45:
            gesture = GestureType.SWIPE_DOWN
        else:
            gesture = GestureType.SWIPE_LEFT
        
        self.dispatch('on_gesture', gesture, start_pos)
    
    def _check_long_press(self, touch_id):
        """Check for long press gesture"""
        if touch_id in self.touches and not self.touches[touch_id]['moved']:
            pos = self.touches[touch_id]['current_pos']
            self.dispatch('on_gesture', GestureType.LONG_PRESS, pos)
    
    # Event definitions
    __events__ = ('on_gesture',)
    
    def on_gesture(self, gesture_type, position):
        """Called when gesture is detected"""
        pass


class NavigationBar(BoxLayout):
    """Touch-optimized navigation bar"""
    
    def __init__(self, config: NavigationConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.orientation = 'horizontal'
        self.size_hint = (1, None)
        self.height = config.navbar_height
        
        self.buttons = {}
        self.current_screen = None
        
        self._setup_bar()
    
    def _setup_bar(self):
        """Setup navigation bar layout"""
        # Back button
        self.back_button = Button(
            text='←',
            size_hint=(None, 1),
            width=self.config.navbar_height,
            font_size=dp(20)
        )
        self.back_button.bind(on_press=self._on_back_pressed)
        self.add_widget(self.back_button)
        
        # Title area
        self.title_label = Label(
            text='Home',
            size_hint=(1, 1),
            font_size=dp(16),
            halign='center'
        )
        self.add_widget(self.title_label)
        
        # Menu button
        self.menu_button = Button(
            text='☰',
            size_hint=(None, 1),
            width=self.config.navbar_height,
            font_size=dp(18)
        )
        self.menu_button.bind(on_press=self._on_menu_pressed)
        self.add_widget(self.menu_button)
    
    def update_for_screen(self, screen: Screen):
        """Update navigation bar for current screen"""
        self.current_screen = screen
        self.title_label.text = screen.title
        self.back_button.disabled = not screen.back_enabled
        
        if not screen.navbar_visible:
            self.opacity = 0
            self.size_hint_y = None
            self.height = 0
        else:
            self.opacity = 1
            self.size_hint_y = None
            self.height = self.config.navbar_height
    
    def _on_back_pressed(self, button):
        """Handle back button press"""
        self.dispatch('on_navigation_request', 'back', None)
    
    def _on_menu_pressed(self, button):
        """Handle menu button press"""
        self.dispatch('on_navigation_request', 'menu', None)
    
    # Event definitions
    __events__ = ('on_navigation_request',)
    
    def on_navigation_request(self, action, data):
        """Called when navigation is requested"""
        pass


class NavigationManager(EventDispatcher):
    """Main navigation manager for touch interface"""
    
    def __init__(self, config: Optional[NavigationConfig] = None):
        super().__init__()
        self.config = config or NavigationConfig()
        self.state = NavigationState.IDLE
        
        self.screens: Dict[str, Screen] = {}
        self.screen_stack: List[str] = []
        self.current_screen: Optional[str] = None
        
        self.gesture_detector = GestureDetector(self.config)
        self.gesture_detector.bind(on_gesture=self._on_gesture)
        
        if self.config.enable_navbar:
            self.navbar = NavigationBar(self.config)
            self.navbar.bind(on_navigation_request=self._on_navigation_request)
        else:
            self.navbar = None
        
        # Navigation callbacks
        self.navigation_callbacks: Dict[str, List[Callable]] = {}
        
        logger.info("NavigationManager initialized")
    
    def register_screen(self, screen: Screen):
        """Register a screen with the navigation manager"""
        if screen.name in self.screens:
            logger.warning(f"Screen '{screen.name}' already registered")
        
        self.screens[screen.name] = screen
        logger.info(f"Registered screen: {screen.name}")
    
    def navigate_to(self, screen_name: str, add_to_stack: bool = True, data: Optional[Dict] = None):
        """Navigate to a specific screen"""
        if screen_name not in self.screens:
            raise ValueError(f"Screen '{screen_name}' not registered")
        
        if self.state == NavigationState.NAVIGATING:
            logger.warning("Navigation already in progress")
            return False
        
        try:
            self.state = NavigationState.NAVIGATING
            
            previous_screen = self.current_screen
            target_screen = self.screens[screen_name]
            
            # Update stack
            if add_to_stack and previous_screen:
                self.screen_stack.append(previous_screen)
                if len(self.screen_stack) > 20:  # Limit stack size
                    self.screen_stack.pop(0)
            
            # Execute navigation
            self._execute_navigation(target_screen, data)
            
            self.current_screen = screen_name
            self.state = NavigationState.IDLE
            
            logger.info(f"Navigated to: {screen_name}")
            return True
            
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            self.state = NavigationState.IDLE
            return False
    
    def go_back(self) -> bool:
        """Navigate back to previous screen"""
        if not self.screen_stack:
            return False
        
        previous_screen = self.screen_stack.pop()
        return self.navigate_to(previous_screen, add_to_stack=False)
    
    def clear_stack(self):
        """Clear navigation stack"""
        self.screen_stack.clear()
        logger.info("Navigation stack cleared")
    
    def get_stack_depth(self) -> int:
        """Get current stack depth"""
        return len(self.screen_stack)
    
    def can_go_back(self) -> bool:
        """Check if back navigation is possible"""
        return len(self.screen_stack) > 0
    
    def _execute_navigation(self, screen: Screen, data: Optional[Dict]):
        """Execute the actual navigation"""
        # Update navbar
        if self.navbar:
            self.navbar.update_for_screen(screen)
        
        # Execute navigation callbacks
        self._execute_callbacks('before_navigation', {
            'screen': screen,
            'data': data
        })
        
        # Dispatch navigation event
        self.dispatch('on_navigation', screen.name, data)
        
        # Execute post-navigation callbacks
        Clock.schedule_once(
            lambda dt: self._execute_callbacks('after_navigation', {
                'screen': screen,
                'data': data
            }),
            0.1
        )
    
    def _on_gesture(self, detector, gesture_type, position):
        """Handle detected gestures"""
        if not self.config.enable_gestures:
            return
        
        if gesture_type == GestureType.SWIPE_RIGHT and self.config.enable_back_gesture:
            # Check if in back gesture zone
            if position[0] < self.config.back_gesture_zone_width:
                self.go_back()
        
        # Dispatch gesture event
        self.dispatch('on_gesture_detected', gesture_type, position)
    
    def _on_navigation_request(self, navbar, action, data):
        """Handle navigation bar requests"""
        if action == 'back':
            self.go_back()
        elif action == 'menu':
            self.dispatch('on_menu_requested')
    
    def add_navigation_callback(self, event: str, callback: Callable):
        """Add callback for navigation events"""
        if event not in self.navigation_callbacks:
            self.navigation_callbacks[event] = []
        self.navigation_callbacks[event].append(callback)
    
    def _execute_callbacks(self, event: str, data: Dict):
        """Execute callbacks for navigation events"""
        if event in self.navigation_callbacks:
            for callback in self.navigation_callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Navigation callback error: {e}")
    
    def get_current_screen(self) -> Optional[Screen]:
        """Get current screen object"""
        if self.current_screen:
            return self.screens[self.current_screen]
        return None
    
    def get_navigation_stats(self) -> Dict[str, Any]:
        """Get navigation statistics"""
        return {
            'current_screen': self.current_screen,
            'stack_depth': len(self.screen_stack),
            'registered_screens': len(self.screens),
            'state': self.state.value,
            'gestures_enabled': self.config.enable_gestures,
            'navbar_enabled': self.config.enable_navbar
        }
    
    # Event definitions
    __events__ = ('on_navigation', 'on_gesture_detected', 'on_menu_requested')
    
    def on_navigation(self, screen_name, data):
        """Called when navigation occurs"""
        pass
    
    def on_gesture_detected(self, gesture_type, position):
        """Called when gesture is detected"""
        pass
    
    def on_menu_requested(self):
        """Called when menu is requested"""
        pass


def create_navigation_manager(config: Optional[NavigationConfig] = None) -> NavigationManager:
    """Factory function to create navigation manager"""
    return NavigationManager(config or NavigationConfig())


def create_touch_optimized_config() -> NavigationConfig:
    """Create configuration optimized for touch interaction"""
    return NavigationConfig(
        gesture_threshold=dp(60),
        swipe_velocity_threshold=250,
        long_press_duration=0.7,
        animation_duration=0.25,
        enable_gestures=True,
        enable_back_gesture=True,
        navbar_height=dp(60)
    )


def create_accessibility_config() -> NavigationConfig:
    """Create configuration optimized for accessibility"""
    return NavigationConfig(
        gesture_threshold=dp(40),  # Smaller gestures
        swipe_velocity_threshold=150,  # Slower gestures
        long_press_duration=1.0,  # Longer press
        animation_duration=0.5,  # Slower animations
        navbar_height=dp(70),  # Larger touch targets
        enable_gestures=True,
        enable_back_gesture=True
    )


if __name__ == '__main__':
    # Test navigation manager
    config = create_touch_optimized_config()
    nav_manager = create_navigation_manager(config)
    
    # Register test screens
    home_screen = Screen("home", "Home")
    settings_screen = Screen("settings", "Settings")
    
    nav_manager.register_screen(home_screen)
    nav_manager.register_screen(settings_screen)
    
    print("Navigation manager created with test screens")
    print(f"Stats: {nav_manager.get_navigation_stats()}")