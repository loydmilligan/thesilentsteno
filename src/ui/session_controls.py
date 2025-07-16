"""
Session control interface for The Silent Steno.

This module provides touch-optimized controls for managing recording sessions
with visual state feedback and intuitive user interaction.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, Callable, List
import logging
from contextlib import contextmanager

from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.label import Label
from kivy.properties import StringProperty, BooleanProperty, ObjectProperty
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.graphics import Color, Ellipse, Rectangle, RoundedRectangle
from kivy.animation import Animation

# Import existing touch controls
from src.ui.touch_controls import TouchButton, create_touch_button, TouchConfig, FeedbackType

logger = logging.getLogger(__name__)


class ControlState(Enum):
    """Session control states."""
    IDLE = auto()
    READY = auto()
    STARTING = auto()
    RECORDING = auto()
    PAUSED = auto()
    STOPPING = auto()
    PROCESSING = auto()
    ERROR = auto()


class SessionAction(Enum):
    """Session control actions."""
    START = auto()
    STOP = auto()
    PAUSE = auto()
    RESUME = auto()
    CANCEL = auto()
    SAVE = auto()


class ControlLayout(Enum):
    """Control layout styles."""
    HORIZONTAL = auto()
    VERTICAL = auto()
    GRID = auto()
    CIRCULAR = auto()


@dataclass
class ControlsConfig:
    """Configuration for session controls."""
    layout: ControlLayout = ControlLayout.HORIZONTAL
    button_size: int = 80
    button_spacing: int = 20
    show_labels: bool = True
    show_status_text: bool = True
    enable_animations: bool = True
    enable_haptic_feedback: bool = True
    confirm_destructive_actions: bool = True
    auto_disable_timeout: float = 0.0  # 0 = no timeout
    primary_color: tuple = (0.2, 0.7, 1.0, 1.0)
    success_color: tuple = (0.2, 1.0, 0.2, 1.0)
    warning_color: tuple = (1.0, 0.8, 0.2, 1.0)
    danger_color: tuple = (1.0, 0.2, 0.2, 1.0)
    disabled_color: tuple = (0.5, 0.5, 0.5, 0.5)


class SessionControlButton(TouchButton):
    """Enhanced touch button for session controls."""
    
    def __init__(self, action: SessionAction, config: ControlsConfig, **kwargs):
        self.action = action
        self.config = config
        self.is_primary = False
        self.pulse_animation = None
        self.button_text = ""
        self.button_color = (1, 1, 1, 1)
        
        # Configure button based on action
        button_config = self._get_button_config(action, config)
        super().__init__(text=self.button_text, config=button_config, **kwargs)
        
        self.bind(on_release=self._on_button_release)
        
    def _get_button_config(self, action: SessionAction, config: ControlsConfig) -> TouchConfig:
        """Get button configuration for action."""
        base_config = TouchConfig(
            min_touch_size=dp(config.button_size),
            enable_haptic=config.enable_haptic_feedback,
            enable_audio=True,
            enable_visual=True
        )
        
        # Store action-specific properties on self for later use
        if action == SessionAction.START:
            self.button_text = "●" if not config.show_labels else "START"
            self.button_color = config.success_color
            self.is_primary = True
        elif action == SessionAction.STOP:
            self.button_text = "■" if not config.show_labels else "STOP"
            self.button_color = config.danger_color
        elif action == SessionAction.PAUSE:
            self.button_text = "⏸" if not config.show_labels else "PAUSE"
            self.button_color = config.warning_color
        elif action == SessionAction.RESUME:
            self.button_text = "▶" if not config.show_labels else "RESUME"
            self.button_color = config.success_color
        elif action == SessionAction.CANCEL:
            self.button_text = "✕" if not config.show_labels else "CANCEL"
            self.button_color = config.danger_color
        elif action == SessionAction.SAVE:
            self.button_text = "💾" if not config.show_labels else "SAVE"
            self.button_color = config.primary_color
            
        return base_config
        
    def _on_button_release(self, button):
        """Handle button release."""
        if self.config.enable_animations and self.is_primary and self.state == 'normal':
            self._pulse_effect()
            
    def _pulse_effect(self):
        """Add pulse animation effect."""
        if self.pulse_animation:
            self.pulse_animation.cancel()
            
        # Scale animation
        self.pulse_animation = Animation(
            size=(self.width * 1.1, self.height * 1.1),
            duration=0.1
        ) + Animation(
            size=(self.width, self.height),
            duration=0.1
        )
        self.pulse_animation.start(self)
        
    def set_recording_state(self, recording: bool):
        """Update button for recording state."""
        if self.action == SessionAction.START and recording:
            # Add pulsing red indicator
            self._add_recording_indicator()
        else:
            self._remove_recording_indicator()
            
    def _add_recording_indicator(self):
        """Add recording indicator."""
        with self.canvas.after:
            Color(1, 0, 0, 0.8)
            self.rec_indicator = Ellipse(
                pos=(self.right - dp(15), self.top - dp(15)),
                size=(dp(10), dp(10))
            )
        self.bind(pos=self._update_indicator, size=self._update_indicator)
        
        # Pulse animation
        self.rec_animation = Animation(
            size=(dp(12), dp(12)), duration=0.5
        ) + Animation(
            size=(dp(8), dp(8)), duration=0.5
        )
        self.rec_animation.repeat = True
        self.rec_animation.start(self.rec_indicator)
        
    def _remove_recording_indicator(self):
        """Remove recording indicator."""
        if hasattr(self, 'rec_animation'):
            self.rec_animation.cancel()
        self.canvas.after.clear()
        
    def _update_indicator(self, *args):
        """Update indicator position."""
        if hasattr(self, 'rec_indicator'):
            self.rec_indicator.pos = (self.right - dp(15), self.top - dp(15))


class SessionControls(Widget):
    """Main session controls widget."""
    
    # Kivy properties
    current_state = StringProperty('idle')
    is_recording = BooleanProperty(False)
    session_duration = StringProperty('00:00:00')
    
    def __init__(self, config: ControlsConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.state = ControlState.IDLE
        self.buttons: Dict[SessionAction, SessionControlButton] = {}
        self.status_label = None
        self.duration_label = None
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'on_start': [],
            'on_stop': [],
            'on_pause': [],
            'on_resume': [],
            'on_cancel': [],
            'on_save': []
        }
        
        # Confirmation dialogs
        self.pending_action = None
        self.confirmation_popup = None
        
        self._setup_layout()
        self._create_buttons()
        self._update_button_states()
        
        # Auto-disable timer
        self.disable_timer = None
        
    def _setup_layout(self):
        """Set up the control layout."""
        if self.config.layout == ControlLayout.HORIZONTAL:
            self.main_layout = BoxLayout(
                orientation='horizontal',
                spacing=dp(self.config.button_spacing),
                size_hint=(1, 1)
            )
        elif self.config.layout == ControlLayout.VERTICAL:
            self.main_layout = BoxLayout(
                orientation='vertical',
                spacing=dp(self.config.button_spacing),
                size_hint=(1, 1)
            )
        elif self.config.layout == ControlLayout.GRID:
            self.main_layout = GridLayout(
                cols=3,
                spacing=dp(self.config.button_spacing),
                size_hint=(1, 1)
            )
        else:  # CIRCULAR
            self.main_layout = AnchorLayout(anchor_x='center', anchor_y='center')
            
        # Status area
        if self.config.show_status_text:
            status_layout = BoxLayout(
                orientation='vertical',
                size_hint=(1, 0.3),
                spacing=dp(5)
            )
            
            self.status_label = Label(
                text="Ready to start recording",
                font_size='16sp',
                color=(0.8, 0.8, 0.8, 1),
                halign='center'
            )
            status_layout.add_widget(self.status_label)
            
            self.duration_label = Label(
                text="00:00:00",
                font_size='20sp',
                color=(1, 1, 1, 1),
                halign='center',
                bold=True
            )
            status_layout.add_widget(self.duration_label)
            
            # Main container
            container = BoxLayout(orientation='vertical')
            container.add_widget(status_layout)
            container.add_widget(self.main_layout)
            self.add_widget(container)
        else:
            self.add_widget(self.main_layout)
            
    def _create_buttons(self):
        """Create control buttons."""
        # Primary action button (start/stop)
        self.buttons[SessionAction.START] = SessionControlButton(
            SessionAction.START, self.config
        )
        self.buttons[SessionAction.START].bind(on_release=self._on_start)
        
        self.buttons[SessionAction.STOP] = SessionControlButton(
            SessionAction.STOP, self.config
        )
        self.buttons[SessionAction.STOP].bind(on_release=self._on_stop)
        
        # Secondary buttons
        self.buttons[SessionAction.PAUSE] = SessionControlButton(
            SessionAction.PAUSE, self.config
        )
        self.buttons[SessionAction.PAUSE].bind(on_release=self._on_pause)
        
        self.buttons[SessionAction.RESUME] = SessionControlButton(
            SessionAction.RESUME, self.config
        )
        self.buttons[SessionAction.RESUME].bind(on_release=self._on_resume)
        
        # Utility buttons
        if self.config.layout in [ControlLayout.GRID, ControlLayout.VERTICAL]:
            self.buttons[SessionAction.CANCEL] = SessionControlButton(
                SessionAction.CANCEL, self.config
            )
            self.buttons[SessionAction.CANCEL].bind(on_release=self._on_cancel)
            
            self.buttons[SessionAction.SAVE] = SessionControlButton(
                SessionAction.SAVE, self.config
            )
            self.buttons[SessionAction.SAVE].bind(on_release=self._on_save)
            
        # Add buttons to layout
        if self.config.layout == ControlLayout.CIRCULAR:
            self._setup_circular_layout()
        else:
            for action in [SessionAction.START, SessionAction.PAUSE, SessionAction.STOP]:
                if action in self.buttons:
                    self.main_layout.add_widget(self.buttons[action])
                    
            # Add additional buttons for grid/vertical layouts
            if self.config.layout in [ControlLayout.GRID, ControlLayout.VERTICAL]:
                for action in [SessionAction.CANCEL, SessionAction.SAVE]:
                    if action in self.buttons:
                        self.main_layout.add_widget(self.buttons[action])
                        
    def _setup_circular_layout(self):
        """Set up circular button layout."""
        # Create circular arrangement of buttons
        center_widget = AnchorLayout(anchor_x='center', anchor_y='center')
        
        # Main start/stop button in center
        center_widget.add_widget(self.buttons[SessionAction.START])
        
        # Secondary buttons around the edge (would need custom positioning)
        # Simplified for now - just add to main layout
        self.main_layout.add_widget(center_widget)
        
    def _update_button_states(self):
        """Update button states based on current state."""
        # Hide all buttons first
        for button in self.buttons.values():
            button.disabled = True
            button.opacity = 0.5
            
        # Show relevant buttons based on state
        if self.state == ControlState.IDLE:
            self.buttons[SessionAction.START].disabled = False
            self.buttons[SessionAction.START].opacity = 1.0
            if self.status_label:
                self.status_label.text = "Ready to start recording"
                
        elif self.state == ControlState.RECORDING:
            self.buttons[SessionAction.PAUSE].disabled = False
            self.buttons[SessionAction.PAUSE].opacity = 1.0
            self.buttons[SessionAction.STOP].disabled = False
            self.buttons[SessionAction.STOP].opacity = 1.0
            
            # Update recording indicators
            for button in self.buttons.values():
                button.set_recording_state(True)
                
            if self.status_label:
                self.status_label.text = "Recording in progress..."
                
        elif self.state == ControlState.PAUSED:
            self.buttons[SessionAction.RESUME].disabled = False
            self.buttons[SessionAction.RESUME].opacity = 1.0
            self.buttons[SessionAction.STOP].disabled = False
            self.buttons[SessionAction.STOP].opacity = 1.0
            
            if self.status_label:
                self.status_label.text = "Recording paused"
                
        elif self.state == ControlState.PROCESSING:
            # All buttons disabled during processing
            if self.status_label:
                self.status_label.text = "Processing recording..."
                
        elif self.state == ControlState.ERROR:
            self.buttons[SessionAction.CANCEL].disabled = False
            self.buttons[SessionAction.CANCEL].opacity = 1.0
            if SessionAction.SAVE in self.buttons:
                self.buttons[SessionAction.SAVE].disabled = False
                self.buttons[SessionAction.SAVE].opacity = 1.0
                
            if self.status_label:
                self.status_label.text = "Error occurred"
                
        # Update Kivy properties
        self.current_state = self.state.name.lower()
        self.is_recording = self.state == ControlState.RECORDING
        
    def set_state(self, new_state: ControlState):
        """Set control state."""
        if new_state != self.state:
            logger.info(f"Session controls state: {self.state} -> {new_state}")
            self.state = new_state
            self._update_button_states()
            
            # Cancel auto-disable timer if state changes
            if self.disable_timer:
                self.disable_timer.cancel()
                self.disable_timer = None
                
    def update_duration(self, duration_text: str):
        """Update session duration display."""
        self.session_duration = duration_text
        if self.duration_label:
            self.duration_label.text = duration_text
            
    def _on_start(self, button):
        """Handle start button."""
        if self.state == ControlState.IDLE:
            self.set_state(ControlState.STARTING)
            Clock.schedule_once(lambda dt: self.set_state(ControlState.RECORDING), 1.0)
            self._notify_action('on_start')
            
    def _on_stop(self, button):
        """Handle stop button."""
        if self.config.confirm_destructive_actions:
            self._confirm_action(SessionAction.STOP, "Stop recording and save session?")
        else:
            self._execute_stop()
            
    def _execute_stop(self):
        """Execute stop action."""
        if self.state in [ControlState.RECORDING, ControlState.PAUSED]:
            self.set_state(ControlState.STOPPING)
            Clock.schedule_once(lambda dt: self.set_state(ControlState.PROCESSING), 1.0)
            Clock.schedule_once(lambda dt: self.set_state(ControlState.IDLE), 3.0)
            self._notify_action('on_stop')
            
    def _on_pause(self, button):
        """Handle pause button."""
        if self.state == ControlState.RECORDING:
            self.set_state(ControlState.PAUSED)
            self._notify_action('on_pause')
            
    def _on_resume(self, button):
        """Handle resume button."""
        if self.state == ControlState.PAUSED:
            self.set_state(ControlState.RECORDING)
            self._notify_action('on_resume')
            
    def _on_cancel(self, button):
        """Handle cancel button."""
        if self.config.confirm_destructive_actions:
            self._confirm_action(SessionAction.CANCEL, "Cancel recording and discard session?")
        else:
            self._execute_cancel()
            
    def _execute_cancel(self):
        """Execute cancel action."""
        self.set_state(ControlState.IDLE)
        self._notify_action('on_cancel')
        
    def _on_save(self, button):
        """Handle save button."""
        self._notify_action('on_save')
        
    def _confirm_action(self, action: SessionAction, message: str):
        """Show confirmation dialog for destructive actions."""
        # In a real implementation, this would show a popup dialog
        # For now, just execute after a short delay
        logger.info(f"Confirming action: {action} - {message}")
        
        if action == SessionAction.STOP:
            Clock.schedule_once(lambda dt: self._execute_stop(), 0.5)
        elif action == SessionAction.CANCEL:
            Clock.schedule_once(lambda dt: self._execute_cancel(), 0.5)
            
    def enable_auto_disable(self, timeout: float):
        """Enable auto-disable after timeout."""
        if self.disable_timer:
            self.disable_timer.cancel()
            
        if timeout > 0:
            self.disable_timer = Clock.schedule_once(self._auto_disable, timeout)
            
    def _auto_disable(self, dt):
        """Auto-disable controls."""
        if self.state == ControlState.IDLE:
            for button in self.buttons.values():
                button.disabled = True
                button.opacity = 0.3
                
            if self.status_label:
                self.status_label.text = "Touch to activate"
                
    def add_callback(self, event: str, callback: Callable):
        """Add event callback."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            
    def remove_callback(self, event: str, callback: Callable):
        """Remove event callback."""
        if event in self.callbacks and callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
            
    def _notify_action(self, action: str):
        """Notify action callbacks."""
        for callback in self.callbacks[action]:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in {action} callback: {e}")


# Factory functions
def create_session_controls(layout: ControlLayout = ControlLayout.HORIZONTAL,
                          config: Optional[ControlsConfig] = None) -> SessionControls:
    """Create session controls with specified layout."""
    if config is None:
        config = ControlsConfig()
    config.layout = layout
    return SessionControls(config)


def create_compact_controls() -> SessionControls:
    """Create compact session controls."""
    config = ControlsConfig(
        layout=ControlLayout.HORIZONTAL,
        button_size=60,
        button_spacing=15,
        show_labels=False,
        show_status_text=False,
        enable_animations=False
    )
    return SessionControls(config)


def create_expanded_controls() -> SessionControls:
    """Create expanded session controls with all features."""
    config = ControlsConfig(
        layout=ControlLayout.GRID,
        button_size=100,
        button_spacing=25,
        show_labels=True,
        show_status_text=True,
        enable_animations=True,
        enable_haptic_feedback=True,
        confirm_destructive_actions=True
    )
    return SessionControls(config)


def create_circular_controls() -> SessionControls:
    """Create circular session controls layout."""
    config = ControlsConfig(
        layout=ControlLayout.CIRCULAR,
        button_size=80,
        show_labels=False,
        enable_animations=True
    )
    return SessionControls(config)


# Demo mode context manager
@contextmanager
def demo_session_controls():
    """Create session controls in demo mode."""
    controls = create_session_controls()
    
    # Add demo callbacks
    def demo_start():
        logger.info("Demo: Starting session")
        Clock.schedule_once(lambda dt: controls.set_state(ControlState.RECORDING), 1.0)
        
    def demo_stop():
        logger.info("Demo: Stopping session")
        Clock.schedule_once(lambda dt: controls.set_state(ControlState.IDLE), 2.0)
        
    def demo_pause():
        logger.info("Demo: Pausing session")
        
    def demo_resume():
        logger.info("Demo: Resuming session")
        
    controls.add_callback('on_start', demo_start)
    controls.add_callback('on_stop', demo_stop)
    controls.add_callback('on_pause', demo_pause)
    controls.add_callback('on_resume', demo_resume)
    
    try:
        yield controls
    finally:
        # Cleanup
        if controls.disable_timer:
            controls.disable_timer.cancel()