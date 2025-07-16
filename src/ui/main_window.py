"""
Main Application Window and UI Framework Initialization

This module provides the primary UI window managing touchscreen interface
with responsive layout and dark mode support for The Silent Steno.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
import threading
import time
import logging

try:
    from kivy.app import App
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.screenmanager import ScreenManager as KivyScreenManager, Screen
    from kivy.uix.label import Label
    from kivy.core.window import Window
    from kivy.clock import Clock
    from kivy.metrics import dp
    from kivy.event import EventDispatcher
except ImportError:
    raise ImportError("Kivy not available. Install with: pip install kivy")

logger = logging.getLogger(__name__)


class WindowState(Enum):
    """Application window states"""
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    MINIMIZED = "minimized"
    ERROR = "error"


@dataclass
class WindowConfig:
    """Configuration for main application window"""
    width: int = 800
    height: int = 480
    fullscreen: bool = True
    resizable: bool = False
    orientation: str = "landscape"
    min_touch_size: int = dp(44)
    window_title: str = "The Silent Steno"
    auto_scale: bool = True
    gpu_acceleration: bool = True
    fps_limit: int = 60
    theme: str = "dark"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.width < 320 or self.height < 240:
            raise ValueError("Window dimensions too small for touch interface")
        if self.min_touch_size < dp(32):
            raise ValueError("Touch size too small for accessibility")


class ScreenManager:
    """Enhanced screen manager for touch navigation"""
    
    def __init__(self, config: WindowConfig):
        self.config = config
        self.kivy_manager = KivyScreenManager()
        self.screens: Dict[str, Screen] = {}
        self.current_screen: Optional[str] = None
        self.screen_history: List[str] = []
        self.transition_callbacks: Dict[str, List[Callable]] = {}
        
    def add_screen(self, name: str, screen: Screen, make_current: bool = False):
        """Add a screen to the manager"""
        try:
            screen.name = name
            self.screens[name] = screen
            self.kivy_manager.add_widget(screen)
            
            if make_current or not self.current_screen:
                self.switch_to(name)
                
            logger.info(f"Added screen: {name}")
        except Exception as e:
            logger.error(f"Failed to add screen {name}: {e}")
            raise
    
    def switch_to(self, screen_name: str, direction: str = "left"):
        """Switch to a specific screen with transition"""
        if screen_name not in self.screens:
            raise ValueError(f"Screen '{screen_name}' not found")
        
        try:
            # Update history
            if self.current_screen:
                self.screen_history.append(self.current_screen)
                if len(self.screen_history) > 10:  # Limit history size
                    self.screen_history.pop(0)
            
            # Execute transition callbacks
            self._execute_callbacks(f"before_{screen_name}")
            
            # Perform transition
            self.kivy_manager.transition.direction = direction
            self.kivy_manager.current = screen_name
            self.current_screen = screen_name
            
            # Execute post-transition callbacks
            Clock.schedule_once(lambda dt: self._execute_callbacks(f"after_{screen_name}"), 0.1)
            
            logger.info(f"Switched to screen: {screen_name}")
        except Exception as e:
            logger.error(f"Failed to switch to screen {screen_name}: {e}")
            raise
    
    def go_back(self):
        """Navigate to previous screen"""
        if self.screen_history:
            previous_screen = self.screen_history.pop()
            self.switch_to(previous_screen, direction="right")
            return True
        return False
    
    def add_transition_callback(self, event: str, callback: Callable):
        """Add callback for screen transition events"""
        if event not in self.transition_callbacks:
            self.transition_callbacks[event] = []
        self.transition_callbacks[event].append(callback)
    
    def _execute_callbacks(self, event: str):
        """Execute callbacks for specific events"""
        if event in self.transition_callbacks:
            for callback in self.transition_callbacks[event]:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Callback error for {event}: {e}")


class MainWindow(BoxLayout, EventDispatcher):
    """Main application window with touch interface support"""
    
    def __init__(self, config: Optional[WindowConfig] = None, **kwargs):
        super().__init__(**kwargs)
        
        self.config = config or WindowConfig()
        self.state = WindowState.INITIALIZING
        self.screen_manager = ScreenManager(self.config)
        self.status_bar = None
        self.error_display = None
        self.startup_time = time.time()
        
        # Performance monitoring
        self.frame_times = []
        self.touch_events = []
        
        # Initialize window
        self._setup_window()
        self._setup_layout()
        self._register_events()
        
        # Schedule readiness check
        Clock.schedule_once(self._check_readiness, 1.0)
        
        logger.info("MainWindow initialized")
    
    def _setup_window(self):
        """Configure Kivy window settings"""
        try:
            Window.size = (self.config.width, self.config.height)
            Window.fullscreen = self.config.fullscreen
            Window.resizable = self.config.resizable
            
            if self.config.orientation == "portrait":
                Window.rotation = 90
            
            # Set minimum touch area
            Window.minimum_width = self.config.width
            Window.minimum_height = self.config.height
            
            logger.info(f"Window configured: {self.config.width}x{self.config.height}")
        except Exception as e:
            logger.error(f"Window setup failed: {e}")
            self.state = WindowState.ERROR
            raise
    
    def _setup_layout(self):
        """Setup main window layout"""
        try:
            self.orientation = 'vertical'
            
            # Add screen manager as main content
            self.add_widget(self.screen_manager.kivy_manager)
            
            # Create default home screen
            home_screen = Screen(name='home')
            home_layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(10))
            
            title_label = Label(
                text='The Silent Steno',
                font_size=dp(24),
                size_hint=(1, 0.3)
            )
            status_label = Label(
                text='Touch UI Framework Ready',
                font_size=dp(16),
                size_hint=(1, 0.2)
            )
            
            home_layout.add_widget(title_label)
            home_layout.add_widget(status_label)
            home_screen.add_widget(home_layout)
            
            self.screen_manager.add_screen('home', home_screen, make_current=True)
            
            logger.info("Main layout configured")
        except Exception as e:
            logger.error(f"Layout setup failed: {e}")
            self.state = WindowState.ERROR
            raise
    
    def _register_events(self):
        """Register window and touch events"""
        try:
            Window.bind(on_touch_down=self._on_touch_down)
            Window.bind(on_touch_up=self._on_touch_up)
            Window.bind(on_keyboard=self._on_keyboard)
            Window.bind(on_resize=self._on_resize)
            
            # Performance monitoring
            Clock.schedule_interval(self._monitor_performance, 1.0)
            
            logger.info("Event handlers registered")
        except Exception as e:
            logger.error(f"Event registration failed: {e}")
            raise
    
    def _on_touch_down(self, window, touch):
        """Handle touch down events"""
        touch_time = time.time()
        self.touch_events.append({
            'type': 'down',
            'pos': touch.pos,
            'time': touch_time
        })
        
        # Ensure minimum touch size
        if not hasattr(touch, 'grab_current'):
            touch.grab_current = None
        
        return False  # Allow event propagation
    
    def _on_touch_up(self, window, touch):
        """Handle touch up events"""
        touch_time = time.time()
        
        # Calculate touch latency
        down_events = [e for e in self.touch_events if e['type'] == 'down']
        if down_events:
            latest_down = max(down_events, key=lambda x: x['time'])
            latency = (touch_time - latest_down['time']) * 1000  # Convert to ms
            
            if latency > 50:  # Log if above target
                logger.warning(f"Touch latency: {latency:.1f}ms (target: <50ms)")
        
        self.touch_events.append({
            'type': 'up',
            'pos': touch.pos,
            'time': touch_time
        })
        
        # Clean old events
        cutoff_time = touch_time - 5.0
        self.touch_events = [e for e in self.touch_events if e['time'] > cutoff_time]
        
        return False
    
    def _on_keyboard(self, window, key, scancode, codepoint, modifier):
        """Handle keyboard events"""
        if key == 27:  # ESC key
            return self.screen_manager.go_back()
        return False
    
    def _on_resize(self, window, width, height):
        """Handle window resize events"""
        logger.info(f"Window resized to {width}x{height}")
        
        # Update configuration
        self.config.width = width
        self.config.height = height
        
        # Trigger layout updates
        self.dispatch('on_window_resize', width, height)
    
    def _monitor_performance(self, dt):
        """Monitor UI performance metrics"""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only last 60 frames for FPS calculation
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        
        # Calculate FPS
        if len(self.frame_times) > 10:
            time_span = self.frame_times[-1] - self.frame_times[0]
            fps = (len(self.frame_times) - 1) / time_span
            
            if fps < 30:  # Log performance issues
                logger.warning(f"Low FPS detected: {fps:.1f}")
    
    def _check_readiness(self, dt):
        """Check if window is ready for use"""
        try:
            startup_time = time.time() - self.startup_time
            
            if startup_time < 5.0:  # Target: <5 seconds
                self.state = WindowState.READY
                logger.info(f"Window ready in {startup_time:.1f}s")
                self.dispatch('on_window_ready')
            else:
                logger.warning(f"Slow startup: {startup_time:.1f}s (target: <5s)")
                self.state = WindowState.READY  # Still mark as ready
                
        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            self.state = WindowState.ERROR
    
    def add_screen(self, name: str, screen: Screen, make_current: bool = False):
        """Add a screen to the window"""
        self.screen_manager.add_screen(name, screen, make_current)
    
    def switch_screen(self, screen_name: str, direction: str = "left"):
        """Switch to a different screen"""
        self.screen_manager.switch_to(screen_name, direction)
    
    def show_error(self, message: str, details: Optional[str] = None):
        """Display error message to user"""
        logger.error(f"UI Error: {message}")
        if details:
            logger.error(f"Error details: {details}")
        
        # Could implement error screen here
        self.state = WindowState.ERROR
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        current_time = time.time()
        
        # Calculate FPS
        fps = 0
        if len(self.frame_times) > 1:
            time_span = self.frame_times[-1] - self.frame_times[0]
            fps = (len(self.frame_times) - 1) / time_span
        
        # Calculate touch event rate
        recent_touches = [e for e in self.touch_events if current_time - e['time'] < 10.0]
        touch_rate = len(recent_touches) / 10.0
        
        return {
            'fps': fps,
            'touch_rate': touch_rate,
            'state': self.state.value,
            'uptime': current_time - self.startup_time,
            'screen_count': len(self.screen_manager.screens),
            'current_screen': self.screen_manager.current_screen
        }
    
    # Event definitions for Kivy
    __events__ = ('on_window_ready', 'on_window_resize')
    
    def on_window_ready(self):
        """Called when window is ready for use"""
        pass
    
    def on_window_resize(self, width, height):
        """Called when window is resized"""
        pass


class SilentStenoApp(App):
    """Main Kivy application for The Silent Steno"""
    
    def __init__(self, config: Optional[WindowConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or WindowConfig()
        self.main_window = None
    
    def build(self):
        """Build the main application"""
        try:
            self.main_window = MainWindow(self.config)
            self.title = self.config.window_title
            return self.main_window
        except Exception as e:
            logger.error(f"Failed to build app: {e}")
            raise
    
    def on_start(self):
        """Called when application starts"""
        logger.info("Silent Steno app started")
    
    def on_stop(self):
        """Called when application stops"""
        logger.info("Silent Steno app stopped")


def create_main_window(config: Optional[WindowConfig] = None) -> MainWindow:
    """Factory function to create main window"""
    return MainWindow(config or WindowConfig())


def create_app(config: Optional[WindowConfig] = None) -> SilentStenoApp:
    """Factory function to create main application"""
    return SilentStenoApp(config or WindowConfig())


def run_app(config: Optional[WindowConfig] = None):
    """Run the main application"""
    app = create_app(config)
    app.run()


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run app
    config = WindowConfig(
        width=800,
        height=480,
        fullscreen=False,  # For development
        theme="dark"
    )
    run_app(config)