"""
Real-time scrolling transcript display for The Silent Steno.

This module provides a scrollable transcript view with speaker identification,
timestamps, and real-time updates optimized for live transcription.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Callable, Tuple
import logging
from collections import deque

from kivy.uix.widget import Widget
from kivy.uix.scrollview import ScrollView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.properties import ObjectProperty, StringProperty, BooleanProperty, NumericProperty
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.graphics import Color, Rectangle, Line
from kivy.uix.anchorlayout import AnchorLayout

logger = logging.getLogger(__name__)


class TranscriptState(Enum):
    """Transcript display states."""
    IDLE = auto()
    ACTIVE = auto()
    PAUSED = auto()
    SCROLLING = auto()
    ERROR = auto()


@dataclass
class SpeakerInfo:
    """Speaker information."""
    id: str
    name: str
    color: Tuple[float, float, float, float] = (1, 1, 1, 1)
    total_words: int = 0
    speaking_time: float = 0.0
    last_seen: Optional[datetime] = None


@dataclass
class TranscriptEntry:
    """Individual transcript entry."""
    id: str
    timestamp: datetime
    speaker: SpeakerInfo
    text: str
    confidence: float = 1.0
    is_final: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def formatted_timestamp(self) -> str:
        """Get formatted timestamp."""
        return self.timestamp.strftime("%H:%M:%S")
    
    @property
    def display_text(self) -> str:
        """Get display text with confidence indicator."""
        confidence_indicator = ""
        if self.confidence < 0.7:
            confidence_indicator = " (?)"
        elif self.confidence < 0.9:
            confidence_indicator = " (~)"
        return f"{self.text}{confidence_indicator}"


@dataclass
class TranscriptConfig:
    """Configuration for transcript display."""
    max_entries: int = 1000
    auto_scroll: bool = True
    show_timestamps: bool = True
    show_speakers: bool = True
    show_confidence: bool = True
    font_size: int = 16
    line_spacing: float = 1.2
    entry_padding: int = 10
    speaker_colors: List[Tuple[float, float, float, float]] = field(default_factory=lambda: [
        (0.3, 0.7, 1.0, 1.0),  # Blue
        (1.0, 0.6, 0.3, 1.0),  # Orange
        (0.3, 1.0, 0.6, 1.0),  # Green
        (1.0, 0.3, 0.7, 1.0),  # Pink
        (0.7, 0.3, 1.0, 1.0),  # Purple
        (1.0, 1.0, 0.3, 1.0),  # Yellow
    ])
    highlight_recent: bool = True
    recent_highlight_duration: float = 5.0
    enable_word_wrap: bool = True
    compact_mode: bool = False


class TranscriptEntryWidget(BoxLayout):
    """Widget for displaying a single transcript entry."""
    
    def __init__(self, entry: TranscriptEntry, config: TranscriptConfig, **kwargs):
        super().__init__(orientation='horizontal', size_hint_y=None, **kwargs)
        self.entry = entry
        self.config = config
        self.bind(minimum_height=self.setter('height'))
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the entry UI."""
        # Timestamp column
        if self.config.show_timestamps:
            timestamp_layout = BoxLayout(
                size_hint_x=0.15,
                orientation='vertical'
            )
            timestamp_label = Label(
                text=self.entry.formatted_timestamp,
                size_hint_y=None,
                height=dp(self.config.font_size * 1.5),
                font_size=f"{self.config.font_size * 0.8}sp",
                color=(0.7, 0.7, 0.7, 1),
                halign='right'
            )
            timestamp_label.bind(texture_size=timestamp_label.setter('text_size'))
            timestamp_layout.add_widget(timestamp_label)
            self.add_widget(timestamp_layout)
        
        # Speaker column
        if self.config.show_speakers:
            speaker_layout = BoxLayout(
                size_hint_x=0.2,
                orientation='vertical'
            )
            speaker_label = Label(
                text=self.entry.speaker.name,
                size_hint_y=None,
                height=dp(self.config.font_size * 1.5),
                font_size=f"{self.config.font_size}sp",
                color=self.entry.speaker.color,
                bold=True,
                halign='left'
            )
            speaker_label.bind(texture_size=speaker_label.setter('text_size'))
            speaker_layout.add_widget(speaker_label)
            self.add_widget(speaker_layout)
        
        # Text column
        text_layout = BoxLayout(orientation='vertical')
        text_label = Label(
            text=self.entry.display_text,
            size_hint_y=None,
            font_size=f"{self.config.font_size}sp",
            color=(1, 1, 1, 1),
            halign='left',
            valign='top',
            text_size=(None, None),
            markup=True
        )
        
        # Enable word wrap
        if self.config.enable_word_wrap:
            def update_text_size(instance, value):
                instance.text_size = (instance.width, None)
            def update_height(instance, value):
                if hasattr(value, '__len__') and len(value) >= 2:
                    instance.height = value[1]
                else:
                    instance.height = dp(self.config.font_size * 1.5)
            text_label.bind(width=update_text_size)
            text_label.bind(texture_size=update_height)
        else:
            text_label.height = dp(self.config.font_size * 1.5)
            
        text_layout.add_widget(text_label)
        self.add_widget(text_layout)
        
        # Confidence indicator
        if self.config.show_confidence and self.entry.confidence < 0.9:
            confidence_layout = BoxLayout(
                size_hint_x=0.1,
                orientation='vertical'
            )
            confidence_color = (1, 0.5, 0.5, 1) if self.entry.confidence < 0.7 else (1, 1, 0.5, 1)
            confidence_label = Label(
                text=f"{self.entry.confidence:.0%}",
                size_hint_y=None,
                height=dp(self.config.font_size * 1.5),
                font_size=f"{self.config.font_size * 0.7}sp",
                color=confidence_color,
                halign='center'
            )
            confidence_layout.add_widget(confidence_label)
            self.add_widget(confidence_layout)
            
    def update_highlight(self, is_recent: bool):
        """Update entry highlighting."""
        if self.config.highlight_recent and is_recent:
            # Add subtle background highlight
            with self.canvas.before:
                Color(0.2, 0.2, 0.3, 0.5)
                self.bg_rect = Rectangle(pos=self.pos, size=self.size)
            self.bind(pos=self._update_rect, size=self._update_rect)
        else:
            self.canvas.before.clear()
            
    def _update_rect(self, *args):
        """Update background rectangle."""
        if hasattr(self, 'bg_rect'):
            self.bg_rect.pos = self.pos
            self.bg_rect.size = self.size


class TranscriptionDisplay(ScrollView):
    """Scrollable transcript display with real-time updates."""
    
    def __init__(self, config: TranscriptConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.state = TranscriptState.IDLE
        self.entries: deque = deque(maxlen=config.max_entries)
        self.speakers: Dict[str, SpeakerInfo] = {}
        self.entry_widgets: List[TranscriptEntryWidget] = []
        self.callbacks: Dict[str, List[Callable]] = {
            'on_entry_added': [],
            'on_speaker_change': [],
            'on_scroll': []
        }
        
        # Configure scroll behavior
        self.do_scroll_x = False
        self.do_scroll_y = True
        self.scroll_type = ['bars', 'content']
        self.bar_width = dp(10)
        
        # Main content layout
        self.content_layout = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            spacing=dp(2)
        )
        self.content_layout.bind(minimum_height=self.content_layout.setter('height'))
        self.add_widget(self.content_layout)
        
        # Auto-scroll tracking
        self._auto_scroll_enabled = config.auto_scroll
        self._last_scroll_y = 0
        self._user_scrolled = False
        
        # Update timer for highlights
        if config.highlight_recent:
            Clock.schedule_interval(self._update_highlights, 1.0)
            
        self._setup_initial_content()
        
    def _setup_initial_content(self):
        """Set up initial content."""
        # Welcome message
        welcome_text = "Live transcript will appear here...\n\nWaiting for audio input."
        welcome_entry = TranscriptEntry(
            id="welcome",
            timestamp=datetime.now(),
            speaker=SpeakerInfo(id="system", name="System", color=(0.7, 0.7, 0.7, 1)),
            text=welcome_text,
            confidence=1.0
        )
        self._add_entry_widget(welcome_entry)
        
    def add_transcript_entry(self, 
                           speaker_id: str, 
                           speaker_name: str, 
                           text: str, 
                           timestamp: Optional[datetime] = None,
                           confidence: float = 1.0,
                           is_final: bool = True) -> str:
        """Add a new transcript entry."""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Get or create speaker
        speaker = self._get_or_create_speaker(speaker_id, speaker_name)
        
        # Create entry
        entry_id = f"{timestamp.isoformat()}_{speaker_id}"
        entry = TranscriptEntry(
            id=entry_id,
            timestamp=timestamp,
            speaker=speaker,
            text=text,
            confidence=confidence,
            is_final=is_final
        )
        
        # Add to entries
        self.entries.append(entry)
        
        # Add widget
        self._add_entry_widget(entry)
        
        # Update speaker stats
        self._update_speaker_stats(speaker, text)
        
        # Auto-scroll if enabled
        if self._auto_scroll_enabled and not self._user_scrolled:
            Clock.schedule_once(self._scroll_to_bottom, 0.1)
            
        # Notify callbacks
        self._notify_entry_added(entry)
        
        return entry_id
        
    def update_last_entry(self, text: str, confidence: float = 1.0, is_final: bool = True):
        """Update the last transcript entry (for streaming updates)."""
        if not self.entries:
            return
            
        entry = self.entries[-1]
        entry.text = text
        entry.confidence = confidence
        entry.is_final = is_final
        
        # Update the widget
        if self.entry_widgets:
            # Remove and recreate the last widget
            last_widget = self.entry_widgets[-1]
            self.content_layout.remove_widget(last_widget)
            self.entry_widgets.pop()
            
            # Add updated widget
            self._add_entry_widget(entry)
            
    def _get_or_create_speaker(self, speaker_id: str, speaker_name: str) -> SpeakerInfo:
        """Get existing speaker or create new one."""
        if speaker_id not in self.speakers:
            # Assign color from palette
            color_index = len(self.speakers) % len(self.config.speaker_colors)
            color = self.config.speaker_colors[color_index]
            
            self.speakers[speaker_id] = SpeakerInfo(
                id=speaker_id,
                name=speaker_name,
                color=color,
                last_seen=datetime.now()
            )
            
            self._notify_speaker_change(self.speakers[speaker_id])
            
        # Update last seen
        self.speakers[speaker_id].last_seen = datetime.now()
        return self.speakers[speaker_id]
        
    def _add_entry_widget(self, entry: TranscriptEntry):
        """Add entry widget to display."""
        widget = TranscriptEntryWidget(entry, self.config)
        self.entry_widgets.append(widget)
        self.content_layout.add_widget(widget)
        
        # Remove oldest widgets if exceeding limit
        while len(self.entry_widgets) > self.config.max_entries:
            oldest_widget = self.entry_widgets.pop(0)
            self.content_layout.remove_widget(oldest_widget)
            
    def _update_speaker_stats(self, speaker: SpeakerInfo, text: str):
        """Update speaker statistics."""
        words = len(text.split())
        speaker.total_words += words
        speaker.speaking_time += words * 0.5  # Rough estimate
        
    def _update_highlights(self, dt):
        """Update recent entry highlights."""
        if not self.config.highlight_recent:
            return
            
        now = datetime.now()
        for i, widget in enumerate(self.entry_widgets):
            if i < len(self.entries):
                entry = self.entries[i - len(self.entries)]
                time_diff = (now - entry.timestamp).total_seconds()
                is_recent = time_diff < self.config.recent_highlight_duration
                widget.update_highlight(is_recent)
                
    def _scroll_to_bottom(self, dt):
        """Scroll to bottom of transcript."""
        self.scroll_y = 0
        
    def enable_auto_scroll(self, enabled: bool):
        """Enable or disable auto-scroll."""
        self._auto_scroll_enabled = enabled
        
    def clear_transcript(self):
        """Clear all transcript entries."""
        self.entries.clear()
        for widget in self.entry_widgets:
            self.content_layout.remove_widget(widget)
        self.entry_widgets.clear()
        self._setup_initial_content()
        
    def search_entries(self, query: str) -> List[TranscriptEntry]:
        """Search transcript entries."""
        query_lower = query.lower()
        results = []
        for entry in self.entries:
            if query_lower in entry.text.lower() or query_lower in entry.speaker.name.lower():
                results.append(entry)
        return results
        
    def export_transcript(self, format_type: str = "text") -> str:
        """Export transcript in specified format."""
        if format_type == "text":
            lines = []
            for entry in self.entries:
                timestamp = entry.formatted_timestamp
                speaker = entry.speaker.name
                text = entry.text
                lines.append(f"[{timestamp}] {speaker}: {text}")
            return "\n".join(lines)
        elif format_type == "json":
            import json
            data = []
            for entry in self.entries:
                data.append({
                    'timestamp': entry.timestamp.isoformat(),
                    'speaker': entry.speaker.name,
                    'text': entry.text,
                    'confidence': entry.confidence
                })
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get transcript statistics."""
        total_entries = len(self.entries)
        total_words = sum(len(entry.text.split()) for entry in self.entries)
        
        speaker_stats = {}
        for speaker in self.speakers.values():
            speaker_stats[speaker.name] = {
                'words': speaker.total_words,
                'speaking_time': speaker.speaking_time,
                'last_seen': speaker.last_seen.isoformat() if speaker.last_seen else None
            }
            
        return {
            'total_entries': total_entries,
            'total_words': total_words,
            'speakers': speaker_stats,
            'duration_minutes': (datetime.now() - self.entries[0].timestamp).total_seconds() / 60 if self.entries else 0
        }
        
    def add_callback(self, event: str, callback: Callable):
        """Add event callback."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            
    def remove_callback(self, event: str, callback: Callable):
        """Remove event callback."""
        if event in self.callbacks and callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
            
    def _notify_entry_added(self, entry: TranscriptEntry):
        """Notify entry added."""
        for callback in self.callbacks['on_entry_added']:
            try:
                callback(entry)
            except Exception as e:
                logger.error(f"Error in entry added callback: {e}")
                
    def _notify_speaker_change(self, speaker: SpeakerInfo):
        """Notify speaker change."""
        for callback in self.callbacks['on_speaker_change']:
            try:
                callback(speaker)
            except Exception as e:
                logger.error(f"Error in speaker change callback: {e}")
                
    def on_scroll_start(self, touch, check_children=True):
        """Handle scroll start."""
        self._user_scrolled = True
        return super().on_scroll_start(touch, check_children)
        
    def on_scroll_stop(self, touch, check_children=True):
        """Handle scroll stop."""
        # Re-enable auto-scroll if scrolled to bottom
        if self.scroll_y <= 0.05:  # Near bottom
            self._user_scrolled = False
        return super().on_scroll_stop(touch, check_children)


# Factory functions
def create_transcription_display(config: Optional[TranscriptConfig] = None) -> TranscriptionDisplay:
    """Create transcription display with configuration."""
    if config is None:
        config = create_default_config()
    return TranscriptionDisplay(config)


def create_default_config() -> TranscriptConfig:
    """Create default transcript configuration."""
    return TranscriptConfig(
        max_entries=1000,
        auto_scroll=True,
        show_timestamps=True,
        show_speakers=True,
        show_confidence=True,
        font_size=16,
        highlight_recent=True
    )


def create_accessible_config() -> TranscriptConfig:
    """Create accessible transcript configuration."""
    return TranscriptConfig(
        max_entries=500,
        auto_scroll=True,
        show_timestamps=True,
        show_speakers=True,
        show_confidence=False,
        font_size=20,
        line_spacing=1.5,
        highlight_recent=False,
        compact_mode=False
    )


def create_compact_config() -> TranscriptConfig:
    """Create compact transcript configuration."""
    return TranscriptConfig(
        max_entries=500,
        auto_scroll=True,
        show_timestamps=False,
        show_speakers=True,
        show_confidence=False,
        font_size=14,
        compact_mode=True,
        highlight_recent=False
    )