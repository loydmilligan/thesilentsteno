"""
Main live session interface screen for The Silent Steno.

This module provides the primary session view that orchestrates all live session
components including transcription display, audio visualization, session controls,
and status indicators.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Optional, Dict, Any, Callable, List, Tuple
import logging
from contextlib import contextmanager

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.properties import ObjectProperty, StringProperty, BooleanProperty
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Rectangle

# Future imports - these will be available when the components are created
# from src.ui.transcription_display import TranscriptionDisplay, create_transcription_display
# from src.ui.audio_visualizer import AudioVisualizer, create_audio_visualizer
# from src.ui.session_controls import SessionControls, create_session_controls
# from src.ui.status_indicators import StatusIndicators, create_status_indicators
# from src.ui.themes import ThemeManager
# from src.ui.feedback_manager import FeedbackManager

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Session states."""
    IDLE = auto()
    STARTING = auto()
    RECORDING = auto()
    PAUSED = auto()
    STOPPING = auto()
    PROCESSING = auto()
    ERROR = auto()


@dataclass
class SessionInfo:
    """Session information."""
    id: str
    title: str = "Meeting Session"
    start_time: Optional[datetime] = None
    duration: timedelta = field(default_factory=timedelta)
    participant_count: int = 0
    transcript_entries: int = 0
    audio_quality: float = 1.0
    storage_used_mb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionViewConfig:
    """Configuration for session view."""
    show_transcript: bool = True
    show_audio_visualizer: bool = True
    show_timer: bool = True
    show_participant_count: bool = True
    enable_auto_scroll: bool = True
    compact_mode: bool = False
    update_interval: float = 0.1
    max_transcript_entries: int = 1000
    enable_demo_mode: bool = False
    demo_update_rate: float = 2.0


class SessionViewModel:
    """View model for session management."""
    
    def __init__(self, config: SessionViewConfig):
        self.config = config
        self.state = SessionState.IDLE
        self.session_info = SessionInfo(id="")
        self.callbacks: Dict[str, List[Callable]] = {
            'on_state_change': [],
            'on_transcript_update': [],
            'on_audio_level': [],
            'on_error': []
        }
        self._update_event = None
        self._demo_event = None
        
    def start_session(self, session_id: str, title: str = "Meeting Session"):
        """Start a new session."""
        if self.state not in [SessionState.IDLE, SessionState.ERROR]:
            logger.warning(f"Cannot start session in state {self.state}")
            return False
            
        self.state = SessionState.STARTING
        self.session_info = SessionInfo(
            id=session_id,
            title=title,
            start_time=datetime.now()
        )
        
        # Simulate startup delay
        Clock.schedule_once(lambda dt: self._on_session_started(), 1.0)
        self._notify_state_change()
        return True
        
    def _on_session_started(self):
        """Handle session started."""
        self.state = SessionState.RECORDING
        self._notify_state_change()
        
        # Start update timer
        if self._update_event:
            self._update_event.cancel()
        self._update_event = Clock.schedule_interval(
            self._update_session, self.config.update_interval
        )
        
        # Start demo mode if enabled
        if self.config.enable_demo_mode:
            self._start_demo_mode()
            
    def stop_session(self):
        """Stop the current session."""
        if self.state not in [SessionState.RECORDING, SessionState.PAUSED]:
            logger.warning(f"Cannot stop session in state {self.state}")
            return False
            
        self.state = SessionState.STOPPING
        self._notify_state_change()
        
        # Stop timers
        if self._update_event:
            self._update_event.cancel()
            self._update_event = None
        if self._demo_event:
            self._demo_event.cancel()
            self._demo_event = None
            
        # Simulate processing delay
        Clock.schedule_once(lambda dt: self._on_session_stopped(), 2.0)
        return True
        
    def _on_session_stopped(self):
        """Handle session stopped."""
        self.state = SessionState.IDLE
        self._notify_state_change()
        
    def pause_session(self):
        """Pause the current session."""
        if self.state != SessionState.RECORDING:
            return False
            
        self.state = SessionState.PAUSED
        self._notify_state_change()
        return True
        
    def resume_session(self):
        """Resume a paused session."""
        if self.state != SessionState.PAUSED:
            return False
            
        self.state = SessionState.RECORDING
        self._notify_state_change()
        return True
        
    def _update_session(self, dt):
        """Update session information."""
        if self.session_info.start_time and self.state == SessionState.RECORDING:
            self.session_info.duration = datetime.now() - self.session_info.start_time
            
    def _start_demo_mode(self):
        """Start demo mode for testing."""
        if self._demo_event:
            self._demo_event.cancel()
        self._demo_event = Clock.schedule_interval(
            self._generate_demo_data, self.config.demo_update_rate
        )
        
    def _generate_demo_data(self, dt):
        """Generate demo data for testing."""
        import random
        
        # Generate transcript entry
        speakers = ["Alice", "Bob", "Charlie", "Diana"]
        phrases = [
            "I think we should focus on the main objectives.",
            "That's a great point. Let me add to that.",
            "We need to consider the timeline as well.",
            "What about the budget implications?",
            "I agree with the proposed approach.",
            "Let's schedule a follow-up meeting.",
            "Can we get more data on this?",
            "The results look promising so far."
        ]
        
        speaker = random.choice(speakers)
        text = random.choice(phrases)
        timestamp = datetime.now()
        
        self._notify_transcript_update({
            'speaker': speaker,
            'text': text,
            'timestamp': timestamp,
            'confidence': random.uniform(0.85, 0.99)
        })
        
        # Generate audio levels
        levels = [random.uniform(0.0, 0.8) for _ in range(8)]
        self._notify_audio_level(levels)
        
        # Update session info
        self.session_info.transcript_entries += 1
        self.session_info.participant_count = len(speakers)
        self.session_info.storage_used_mb += 0.1
        
    def add_callback(self, event: str, callback: Callable):
        """Add event callback."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            
    def remove_callback(self, event: str, callback: Callable):
        """Remove event callback."""
        if event in self.callbacks and callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
            
    def _notify_state_change(self):
        """Notify state change."""
        for callback in self.callbacks['on_state_change']:
            try:
                callback(self.state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
                
    def _notify_transcript_update(self, entry: Dict[str, Any]):
        """Notify transcript update."""
        for callback in self.callbacks['on_transcript_update']:
            try:
                callback(entry)
            except Exception as e:
                logger.error(f"Error in transcript update callback: {e}")
                
    def _notify_audio_level(self, levels: List[float]):
        """Notify audio level update."""
        for callback in self.callbacks['on_audio_level']:
            try:
                callback(levels)
            except Exception as e:
                logger.error(f"Error in audio level callback: {e}")


class SessionView(Screen):
    """Main live session interface screen."""
    
    def __init__(self, config: SessionViewConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.view_model = SessionViewModel(config)
        self.theme_manager = None
        self.feedback_manager = None
        
        # UI components (will be initialized)
        self.transcript_display = None
        self.audio_visualizer = None
        self.session_controls = None
        self.status_indicators = None
        self.timer_label = None
        
        self._setup_ui()
        self._bind_events()
        
    def _setup_ui(self):
        """Set up the user interface."""
        # Main layout
        main_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        
        # Top bar with status and timer
        top_bar = BoxLayout(size_hint_y=0.1, spacing=10)
        
        # Status indicators placeholder
        status_box = BoxLayout(size_hint_x=0.3)
        status_label = Label(text="[Status Indicators]", markup=True)
        status_box.add_widget(status_label)
        top_bar.add_widget(status_box)
        
        # Timer
        timer_box = AnchorLayout(anchor_x='center', anchor_y='center')
        self.timer_label = Label(
            text="00:00:00",
            font_size='24sp',
            bold=True
        )
        timer_box.add_widget(self.timer_label)
        top_bar.add_widget(timer_box)
        
        # Session info
        info_box = BoxLayout(size_hint_x=0.3, orientation='vertical')
        self.participant_label = Label(text="Participants: 0", size_hint_y=0.5)
        self.storage_label = Label(text="Storage: 0 MB", size_hint_y=0.5)
        info_box.add_widget(self.participant_label)
        info_box.add_widget(self.storage_label)
        top_bar.add_widget(info_box)
        
        main_layout.add_widget(top_bar)
        
        # Content area
        content_layout = BoxLayout(orientation='vertical', spacing=10)
        
        # Audio visualizer placeholder
        if self.config.show_audio_visualizer:
            viz_box = BoxLayout(size_hint_y=0.2)
            viz_label = Label(text="[Audio Visualizer]", markup=True)
            viz_box.add_widget(viz_label)
            content_layout.add_widget(viz_box)
        
        # Transcript display placeholder
        if self.config.show_transcript:
            transcript_box = BoxLayout()
            transcript_label = Label(
                text="[Transcript Display]\n\nLive transcript will appear here...",
                markup=True,
                halign='center'
            )
            transcript_box.add_widget(transcript_label)
            content_layout.add_widget(transcript_box)
        
        main_layout.add_widget(content_layout)
        
        # Bottom controls
        controls_layout = AnchorLayout(
            anchor_x='center',
            anchor_y='center',
            size_hint_y=0.15
        )
        controls_box = BoxLayout(size_hint=(0.8, 0.8), spacing=20)
        
        # Placeholder for session controls
        controls_label = Label(text="[Session Controls]", markup=True)
        controls_box.add_widget(controls_label)
        
        controls_layout.add_widget(controls_box)
        main_layout.add_widget(controls_layout)
        
        self.add_widget(main_layout)
        
        # Start timer update
        Clock.schedule_interval(self._update_timer, 1.0)
        
    def _bind_events(self):
        """Bind view model events."""
        self.view_model.add_callback('on_state_change', self._on_state_change)
        self.view_model.add_callback('on_transcript_update', self._on_transcript_update)
        self.view_model.add_callback('on_audio_level', self._on_audio_level)
        
    def _on_state_change(self, state: SessionState):
        """Handle session state change."""
        logger.info(f"Session state changed to: {state}")
        
        # Update UI based on state
        if state == SessionState.RECORDING:
            self._apply_recording_theme()
        elif state == SessionState.PAUSED:
            self._apply_paused_theme()
        elif state == SessionState.ERROR:
            self._apply_error_theme()
        else:
            self._apply_default_theme()
            
    def _on_transcript_update(self, entry: Dict[str, Any]):
        """Handle transcript update."""
        # Will forward to transcript display when implemented
        logger.debug(f"Transcript update: {entry}")
        
    def _on_audio_level(self, levels: List[float]):
        """Handle audio level update."""
        # Will forward to audio visualizer when implemented
        pass
        
    def _update_timer(self, dt):
        """Update session timer."""
        if self.view_model.session_info.start_time:
            duration = self.view_model.session_info.duration
            hours = int(duration.total_seconds() // 3600)
            minutes = int((duration.total_seconds() % 3600) // 60)
            seconds = int(duration.total_seconds() % 60)
            self.timer_label.text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
        # Update other info
        self.participant_label.text = f"Participants: {self.view_model.session_info.participant_count}"
        self.storage_label.text = f"Storage: {self.view_model.session_info.storage_used_mb:.1f} MB"
        
    def _apply_recording_theme(self):
        """Apply recording theme."""
        if self.timer_label:
            self.timer_label.color = (0, 1, 0, 1)  # Green
            
    def _apply_paused_theme(self):
        """Apply paused theme."""
        if self.timer_label:
            self.timer_label.color = (1, 1, 0, 1)  # Yellow
            
    def _apply_error_theme(self):
        """Apply error theme."""
        if self.timer_label:
            self.timer_label.color = (1, 0, 0, 1)  # Red
            
    def _apply_default_theme(self):
        """Apply default theme."""
        if self.timer_label:
            self.timer_label.color = (1, 1, 1, 1)  # White
            
    def start_demo(self):
        """Start demo mode."""
        self.config.enable_demo_mode = True
        self.view_model.start_session("demo_session", "Demo Meeting")
        
    def on_enter(self):
        """Called when screen is entered."""
        super().on_enter()
        logger.info("Entered session view")
        
        # Start demo if configured
        if self.config.enable_demo_mode:
            Clock.schedule_once(lambda dt: self.start_demo(), 1.0)
            
    def on_leave(self):
        """Called when screen is left."""
        super().on_leave()
        logger.info("Left session view")
        
        # Stop any active session
        if self.view_model.state in [SessionState.RECORDING, SessionState.PAUSED]:
            self.view_model.stop_session()


# Factory functions
def create_session_view(name: str = "session") -> SessionView:
    """Create session view with default configuration."""
    config = SessionViewConfig()
    return SessionView(config, name=name)


def create_default_config() -> SessionViewConfig:
    """Create default session view configuration."""
    return SessionViewConfig(
        show_transcript=True,
        show_audio_visualizer=True,
        show_timer=True,
        show_participant_count=True,
        enable_auto_scroll=True,
        compact_mode=False
    )


def create_compact_config() -> SessionViewConfig:
    """Create compact session view configuration."""
    return SessionViewConfig(
        show_transcript=True,
        show_audio_visualizer=False,
        show_timer=True,
        show_participant_count=False,
        enable_auto_scroll=True,
        compact_mode=True,
        max_transcript_entries=500
    )


# Demo mode context manager
@contextmanager
def demo_session_view():
    """Create session view in demo mode."""
    config = SessionViewConfig(enable_demo_mode=True)
    view = SessionView(config, name="demo_session")
    try:
        yield view
    finally:
        if view.view_model.state != SessionState.IDLE:
            view.view_model.stop_session()