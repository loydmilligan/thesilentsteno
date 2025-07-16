"""
Session Details View - Detailed session view with metadata and actions.

This module provides a comprehensive session details interface for The Silent Steno project,
displaying complete session information, transcript preview, analysis results, and available actions.
"""

from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from contextlib import contextmanager

try:
    from kivy.app import App
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.gridlayout import GridLayout
    from kivy.uix.scrollview import ScrollView
    from kivy.uix.label import Label
    from kivy.uix.button import Button
    from kivy.uix.textinput import TextInput
    from kivy.uix.popup import Popup
    from kivy.uix.progressbar import ProgressBar
    from kivy.uix.slider import Slider
    from kivy.uix.accordion import Accordion, AccordionItem
    from kivy.properties import StringProperty, NumericProperty, BooleanProperty, ObjectProperty
    from kivy.event import EventDispatcher
    from kivy.clock import Clock
    from kivy.metrics import dp
    from kivy.utils import get_color_from_hex
except ImportError:
    # Fallback for systems without Kivy
    class BoxLayout: pass
    class GridLayout: pass
    class ScrollView: pass
    class Label: pass
    class Button: pass
    class TextInput: pass
    class Popup: pass
    class ProgressBar: pass
    class Slider: pass
    class Accordion: pass
    class AccordionItem: pass
    class EventDispatcher: pass
    StringProperty = NumericProperty = BooleanProperty = ObjectProperty = lambda x: x
    Clock = None
    dp = lambda x: x
    get_color_from_hex = lambda x: x

class SessionMetadata:
    """Container for session metadata."""
    
    def __init__(self, session_id: str, title: str, date: datetime,
                 duration: int, file_size: int, audio_format: str = "FLAC",
                 sample_rate: int = 44100, channels: int = 2,
                 notes: str = "", tags: List[str] = None):
        self.session_id = session_id
        self.title = title
        self.date = date
        self.duration = duration  # in seconds
        self.file_size = file_size  # in bytes
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.channels = channels
        self.notes = notes or ""
        self.tags = tags or []
        
        # Analysis metadata
        self.has_transcript = False
        self.has_analysis = False
        self.transcript_confidence = 0.0
        self.speaker_count = 0
        self.word_count = 0
        self.analysis_version = ""
        
        # Processing status
        self.processing_status = "completed"
        self.error_message = ""
    
    @property
    def formatted_date(self) -> str:
        """Get formatted date string."""
        return self.date.strftime('%Y-%m-%d %H:%M:%S')
    
    @property
    def formatted_duration(self) -> str:
        """Get formatted duration string."""
        hours = self.duration // 3600
        minutes = (self.duration % 3600) // 60
        seconds = self.duration % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"
    
    @property
    def formatted_size(self) -> str:
        """Get formatted file size string."""
        if self.file_size < 1024:
            return f"{self.file_size} B"
        elif self.file_size < 1024 * 1024:
            return f"{self.file_size / 1024:.1f} KB"
        elif self.file_size < 1024 * 1024 * 1024:
            return f"{self.file_size / (1024 * 1024):.1f} MB"
        else:
            return f"{self.file_size / (1024 * 1024 * 1024):.1f} GB"

class SessionActions:
    """Available actions for a session."""
    
    def __init__(self):
        self.can_play = True
        self.can_edit = True
        self.can_export = True
        self.can_delete = True
        self.can_share = True
        self.can_reanalyze = True

class SessionDetailsConfig:
    """Configuration for session details display."""
    
    def __init__(self):
        # Display options
        self.show_waveform = True
        self.show_transcript_preview = True
        self.show_analysis_summary = True
        self.show_participant_stats = True
        self.max_transcript_preview = 500  # characters
        
        # Touch optimization
        self.button_height = dp(50)
        self.button_spacing = dp(8)
        self.section_spacing = dp(16)
        
        # Audio player
        self.enable_audio_player = True
        self.player_controls = True
        self.seek_precision = 5  # seconds
        
        # Colors
        self.primary_color = get_color_from_hex('#3498DB') if callable(get_color_from_hex) else '#3498DB'
        self.secondary_color = get_color_from_hex('#2C3E50') if callable(get_color_from_hex) else '#2C3E50'
        self.accent_color = get_color_from_hex('#E74C3C') if callable(get_color_from_hex) else '#E74C3C'

class AudioPlayerWidget(BoxLayout):
    """Audio player widget for session playback."""
    
    current_time = NumericProperty(0)
    total_time = NumericProperty(0)
    is_playing = BooleanProperty(False)
    volume = NumericProperty(0.7)
    
    def __init__(self, config: SessionDetailsConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.orientation = 'vertical'
        self.size_hint_y = None
        self.height = dp(120)
        self.spacing = dp(8)
        self.callbacks = []
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the audio player UI."""
        # Progress bar and time labels
        progress_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(30),
            spacing=dp(8)
        )
        
        self.current_time_label = Label(
            text="0:00",
            size_hint_x=None,
            width=dp(50),
            font_size=dp(12)
        )
        progress_layout.add_widget(self.current_time_label)
        
        self.progress_bar = Slider(
            min=0,
            max=100,
            value=0,
            size_hint_x=1
        )
        self.progress_bar.bind(value=self._on_seek)
        progress_layout.add_widget(self.progress_bar)
        
        self.total_time_label = Label(
            text="0:00",
            size_hint_x=None,
            width=dp(50),
            font_size=dp(12)
        )
        progress_layout.add_widget(self.total_time_label)
        
        self.add_widget(progress_layout)
        
        # Control buttons
        controls_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(50),
            spacing=dp(8)
        )
        
        self.rewind_btn = Button(
            text="⏪",
            size_hint_x=None,
            width=dp(60),
            font_size=dp(20)
        )
        self.rewind_btn.bind(on_press=self._on_rewind)
        controls_layout.add_widget(self.rewind_btn)
        
        self.play_pause_btn = Button(
            text="▶️",
            size_hint_x=None,
            width=dp(80),
            font_size=dp(24)
        )
        self.play_pause_btn.bind(on_press=self._on_play_pause)
        controls_layout.add_widget(self.play_pause_btn)
        
        self.forward_btn = Button(
            text="⏩",
            size_hint_x=None,
            width=dp(60),
            font_size=dp(20)
        )
        self.forward_btn.bind(on_press=self._on_forward)
        controls_layout.add_widget(self.forward_btn)
        
        # Volume control
        volume_layout = BoxLayout(
            orientation='horizontal',
            size_hint_x=0.4,
            spacing=dp(4)
        )
        
        volume_label = Label(
            text="🔊",
            size_hint_x=None,
            width=dp(30),
            font_size=dp(16)
        )
        volume_layout.add_widget(volume_label)
        
        self.volume_slider = Slider(
            min=0,
            max=1,
            value=self.volume,
            size_hint_x=1
        )
        self.volume_slider.bind(value=self._on_volume_changed)
        volume_layout.add_widget(self.volume_slider)
        
        controls_layout.add_widget(volume_layout)
        
        self.add_widget(controls_layout)
        
        # Speed control
        speed_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(30),
            spacing=dp(8)
        )
        
        speed_label = Label(
            text="Speed:",
            size_hint_x=None,
            width=dp(60),
            font_size=dp(12)
        )
        speed_layout.add_widget(speed_label)
        
        self.speed_slider = Slider(
            min=0.5,
            max=2.0,
            value=1.0,
            size_hint_x=0.5
        )
        self.speed_slider.bind(value=self._on_speed_changed)
        speed_layout.add_widget(self.speed_slider)
        
        self.speed_value_label = Label(
            text="1.0x",
            size_hint_x=None,
            width=dp(50),
            font_size=dp(12)
        )
        speed_layout.add_widget(self.speed_value_label)
        
        self.add_widget(speed_layout)
    
    def load_audio(self, audio_path: str, duration: int):
        """Load audio file for playback."""
        self.total_time = duration
        self.current_time = 0
        self.progress_bar.max = duration
        self.progress_bar.value = 0
        self._update_time_labels()
        self._trigger_callback('audio_loaded', audio_path, duration)
    
    def set_position(self, position: int):
        """Set playback position."""
        self.current_time = position
        self.progress_bar.value = position
        self._update_time_labels()
    
    def set_playing(self, playing: bool):
        """Set playing state."""
        self.is_playing = playing
        self.play_pause_btn.text = "⏸️" if playing else "▶️"
    
    def _update_time_labels(self):
        """Update time display labels."""
        self.current_time_label.text = self._format_time(self.current_time)
        self.total_time_label.text = self._format_time(self.total_time)
    
    def _format_time(self, seconds: int) -> str:
        """Format seconds to MM:SS or HH:MM:SS."""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def _on_seek(self, slider, value):
        """Handle seek slider change."""
        self.current_time = int(value)
        self._update_time_labels()
        self._trigger_callback('seek_requested', self.current_time)
    
    def _on_play_pause(self, button):
        """Handle play/pause button."""
        new_state = not self.is_playing
        self.set_playing(new_state)
        self._trigger_callback('play_pause_requested', new_state)
    
    def _on_rewind(self, button):
        """Handle rewind button."""
        new_position = max(0, self.current_time - self.config.seek_precision)
        self.set_position(new_position)
        self._trigger_callback('seek_requested', new_position)
    
    def _on_forward(self, button):
        """Handle forward button."""
        new_position = min(self.total_time, self.current_time + self.config.seek_precision)
        self.set_position(new_position)
        self._trigger_callback('seek_requested', new_position)
    
    def _on_volume_changed(self, slider, value):
        """Handle volume change."""
        self.volume = value
        self._trigger_callback('volume_changed', value)
    
    def _on_speed_changed(self, slider, value):
        """Handle playback speed change."""
        self.speed_value_label.text = f"{value:.1f}x"
        self._trigger_callback('speed_changed', value)
    
    def _trigger_callback(self, event_type: str, *args):
        """Trigger registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(event_type, *args)
            except Exception as e:
                print(f"Error in audio player callback: {e}")
    
    def add_callback(self, callback: Callable):
        """Add a callback for events."""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

class SessionDetailsView(BoxLayout, EventDispatcher):
    """Main session details view with comprehensive information and actions."""
    
    def __init__(self, config: Optional[SessionDetailsConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or SessionDetailsConfig()
        self.orientation = 'vertical'
        self.spacing = dp(8)
        self.padding = dp(16)
        
        # State
        self.session_metadata = None
        self.session_actions = None
        self.transcript_text = ""
        self.analysis_data = {}
        
        # Callbacks
        self.callbacks = []
        
        # Build UI
        self._build_ui()
    
    def _build_ui(self):
        """Build the main UI."""
        # Header with title and back button
        header_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(60),
            spacing=dp(8)
        )
        
        self.back_btn = Button(
            text="← Back",
            size_hint_x=None,
            width=dp(100),
            font_size=dp(14)
        )
        self.back_btn.bind(on_press=self._on_back_pressed)
        header_layout.add_widget(self.back_btn)
        
        self.title_label = Label(
            text="Session Details",
            font_size=dp(20),
            bold=True,
            halign='left',
            valign='middle'
        )
        self.title_label.bind(size=self.title_label.setter('text_size'))
        header_layout.add_widget(self.title_label)
        
        self.add_widget(header_layout)
        
        # Main content in scroll view
        self.scroll_view = ScrollView(
            do_scroll_x=False,
            do_scroll_y=True
        )
        
        self.content_layout = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            spacing=self.config.section_spacing
        )
        self.content_layout.bind(minimum_height=self.content_layout.setter('height'))
        
        # Audio player section
        if self.config.enable_audio_player:
            self.audio_player = AudioPlayerWidget(self.config)
            self.audio_player.add_callback(self._on_audio_event)
            self.content_layout.add_widget(self.audio_player)
        
        # Session info accordion
        self.accordion = Accordion(
            size_hint_y=None,
            height=dp(400)
        )
        
        # Metadata section
        self.metadata_item = AccordionItem(title='Session Information')
        self.metadata_content = self._create_metadata_section()
        self.metadata_item.add_widget(self.metadata_content)
        self.accordion.add_widget(self.metadata_item)
        
        # Transcript section
        if self.config.show_transcript_preview:
            self.transcript_item = AccordionItem(title='Transcript Preview')
            self.transcript_content = self._create_transcript_section()
            self.transcript_item.add_widget(self.transcript_content)
            self.accordion.add_widget(self.transcript_item)
        
        # Analysis section
        if self.config.show_analysis_summary:
            self.analysis_item = AccordionItem(title='Analysis Summary')
            self.analysis_content = self._create_analysis_section()
            self.analysis_item.add_widget(self.analysis_content)
            self.accordion.add_widget(self.analysis_item)
        
        self.content_layout.add_widget(self.accordion)
        
        # Action buttons
        self.actions_layout = self._create_actions_section()
        self.content_layout.add_widget(self.actions_layout)
        
        # Notes section
        self.notes_section = self._create_notes_section()
        self.content_layout.add_widget(self.notes_section)
        
        self.scroll_view.add_widget(self.content_layout)
        self.add_widget(self.scroll_view)
    
    def _create_metadata_section(self) -> BoxLayout:
        """Create the metadata display section."""
        layout = GridLayout(
            cols=2,
            size_hint_y=None,
            spacing=dp(8),
            padding=dp(8)
        )
        layout.bind(minimum_height=layout.setter('height'))
        
        # Metadata fields will be populated in load_session
        self.metadata_labels = {}
        
        return layout
    
    def _create_transcript_section(self) -> BoxLayout:
        """Create the transcript preview section."""
        layout = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            spacing=dp(8),
            padding=dp(8)
        )
        layout.bind(minimum_height=layout.setter('height'))
        
        # Transcript preview
        self.transcript_preview = Label(
            text="Loading transcript...",
            text_size=(None, None),
            halign='left',
            valign='top',
            font_size=dp(12)
        )
        
        transcript_scroll = ScrollView(
            size_hint_y=None,
            height=dp(200),
            do_scroll_x=False
        )
        transcript_scroll.add_widget(self.transcript_preview)
        layout.add_widget(transcript_scroll)
        
        # Transcript actions
        transcript_actions = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(40),
            spacing=dp(8)
        )
        
        self.view_full_transcript_btn = Button(
            text="View Full Transcript",
            size_hint_x=0.5,
            font_size=dp(12)
        )
        self.view_full_transcript_btn.bind(on_press=self._on_view_transcript)
        transcript_actions.add_widget(self.view_full_transcript_btn)
        
        self.search_transcript_btn = Button(
            text="Search Transcript",
            size_hint_x=0.5,
            font_size=dp(12)
        )
        self.search_transcript_btn.bind(on_press=self._on_search_transcript)
        transcript_actions.add_widget(self.search_transcript_btn)
        
        layout.add_widget(transcript_actions)
        
        return layout
    
    def _create_analysis_section(self) -> BoxLayout:
        """Create the analysis summary section."""
        layout = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            spacing=dp(8),
            padding=dp(8)
        )
        layout.bind(minimum_height=layout.setter('height'))
        
        # Analysis summary will be populated in load_session
        self.analysis_summary = Label(
            text="Loading analysis...",
            text_size=(None, None),
            halign='left',
            valign='top',
            font_size=dp(12)
        )
        layout.add_widget(self.analysis_summary)
        
        # Analysis actions
        analysis_actions = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(40),
            spacing=dp(8)
        )
        
        self.view_full_analysis_btn = Button(
            text="View Full Analysis",
            size_hint_x=0.5,
            font_size=dp(12)
        )
        self.view_full_analysis_btn.bind(on_press=self._on_view_analysis)
        analysis_actions.add_widget(self.view_full_analysis_btn)
        
        self.reanalyze_btn = Button(
            text="Re-analyze",
            size_hint_x=0.5,
            font_size=dp(12)
        )
        self.reanalyze_btn.bind(on_press=self._on_reanalyze)
        analysis_actions.add_widget(self.reanalyze_btn)
        
        layout.add_widget(analysis_actions)
        
        return layout
    
    def _create_actions_section(self) -> BoxLayout:
        """Create the main actions section."""
        layout = GridLayout(
            cols=3,
            size_hint_y=None,
            height=dp(120),
            spacing=self.config.button_spacing,
            padding=dp(8)
        )
        
        # Export button
        self.export_btn = Button(
            text="📤\nExport",
            font_size=dp(12),
            halign='center'
        )
        self.export_btn.bind(on_press=self._on_export)
        layout.add_widget(self.export_btn)
        
        # Share button
        self.share_btn = Button(
            text="📤\nShare",
            font_size=dp(12),
            halign='center'
        )
        self.share_btn.bind(on_press=self._on_share)
        layout.add_widget(self.share_btn)
        
        # Edit button
        self.edit_btn = Button(
            text="✏️\nEdit",
            font_size=dp(12),
            halign='center'
        )
        self.edit_btn.bind(on_press=self._on_edit)
        layout.add_widget(self.edit_btn)
        
        # Delete button
        self.delete_btn = Button(
            text="🗑️\nDelete",
            font_size=dp(12),
            halign='center'
        )
        self.delete_btn.bind(on_press=self._on_delete)
        layout.add_widget(self.delete_btn)
        
        # Archive button
        self.archive_btn = Button(
            text="📦\nArchive",
            font_size=dp(12),
            halign='center'
        )
        self.archive_btn.bind(on_press=self._on_archive)
        layout.add_widget(self.archive_btn)
        
        # Settings button
        self.session_settings_btn = Button(
            text="⚙️\nSettings",
            font_size=dp(12),
            halign='center'
        )
        self.session_settings_btn.bind(on_press=self._on_session_settings)
        layout.add_widget(self.session_settings_btn)
        
        return layout
    
    def _create_notes_section(self) -> BoxLayout:
        """Create the notes editing section."""
        layout = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            spacing=dp(8),
            padding=dp(8)
        )
        layout.bind(minimum_height=layout.setter('height'))
        
        notes_header = Label(
            text="Notes",
            font_size=dp(16),
            bold=True,
            size_hint_y=None,
            height=dp(30),
            halign='left'
        )
        notes_header.bind(size=notes_header.setter('text_size'))
        layout.add_widget(notes_header)
        
        self.notes_input = TextInput(
            text="",
            multiline=True,
            size_hint_y=None,
            height=dp(120),
            font_size=dp(12)
        )
        self.notes_input.bind(text=self._on_notes_changed)
        layout.add_widget(self.notes_input)
        
        notes_actions = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(40),
            spacing=dp(8)
        )
        
        self.save_notes_btn = Button(
            text="Save Notes",
            size_hint_x=0.5,
            font_size=dp(12)
        )
        self.save_notes_btn.bind(on_press=self._on_save_notes)
        notes_actions.add_widget(self.save_notes_btn)
        
        self.clear_notes_btn = Button(
            text="Clear Notes",
            size_hint_x=0.5,
            font_size=dp(12)
        )
        self.clear_notes_btn.bind(on_press=self._on_clear_notes)
        notes_actions.add_widget(self.clear_notes_btn)
        
        layout.add_widget(notes_actions)
        
        return layout
    
    def load_session(self, session_metadata: SessionMetadata, 
                    session_actions: Optional[SessionActions] = None):
        """Load session data into the view."""
        self.session_metadata = session_metadata
        self.session_actions = session_actions or SessionActions()
        
        # Update title
        self.title_label.text = session_metadata.title
        
        # Update metadata display
        self._update_metadata_display()
        
        # Load audio player
        if self.config.enable_audio_player and hasattr(self, 'audio_player'):
            # In a real implementation, this would load the actual audio file
            self.audio_player.load_audio(f"audio_{session_metadata.session_id}.flac", 
                                       session_metadata.duration)
        
        # Update action buttons availability
        self._update_action_buttons()
        
        # Load notes
        self.notes_input.text = session_metadata.notes
        
        self._trigger_callback('session_loaded', session_metadata.session_id)
    
    def update_metadata(self, metadata: SessionMetadata):
        """Update session metadata."""
        self.session_metadata = metadata
        self._update_metadata_display()
        self._trigger_callback('metadata_updated', metadata.session_id)
    
    def _update_metadata_display(self):
        """Update the metadata display section."""
        if not self.session_metadata:
            return
        
        # Clear existing metadata labels
        self.metadata_content.clear_widgets()
        
        metadata_fields = [
            ("Session ID", self.session_metadata.session_id),
            ("Date", self.session_metadata.formatted_date),
            ("Duration", self.session_metadata.formatted_duration),
            ("File Size", self.session_metadata.formatted_size),
            ("Audio Format", self.session_metadata.audio_format),
            ("Sample Rate", f"{self.session_metadata.sample_rate} Hz"),
            ("Channels", str(self.session_metadata.channels)),
            ("Processing Status", self.session_metadata.processing_status),
        ]
        
        if self.session_metadata.has_transcript:
            metadata_fields.extend([
                ("Transcript Confidence", f"{self.session_metadata.transcript_confidence:.1%}"),
                ("Word Count", str(self.session_metadata.word_count)),
                ("Speaker Count", str(self.session_metadata.speaker_count)),
            ])
        
        if self.session_metadata.has_analysis:
            metadata_fields.append(("Analysis Version", self.session_metadata.analysis_version))
        
        for field_name, field_value in metadata_fields:
            # Field label
            label = Label(
                text=f"{field_name}:",
                font_size=dp(12),
                bold=True,
                size_hint_y=None,
                height=dp(30),
                halign='right',
                valign='middle'
            )
            label.bind(size=label.setter('text_size'))
            self.metadata_content.add_widget(label)
            
            # Field value
            value = Label(
                text=str(field_value),
                font_size=dp(12),
                size_hint_y=None,
                height=dp(30),
                halign='left',
                valign='middle'
            )
            value.bind(size=value.setter('text_size'))
            self.metadata_content.add_widget(value)
        
        # Update content height
        self.metadata_content.height = len(metadata_fields) * dp(30)
    
    def _update_action_buttons(self):
        """Update action button availability."""
        if not self.session_actions:
            return
        
        self.export_btn.disabled = not self.session_actions.can_export
        self.share_btn.disabled = not self.session_actions.can_share
        self.edit_btn.disabled = not self.session_actions.can_edit
        self.delete_btn.disabled = not self.session_actions.can_delete
        self.reanalyze_btn.disabled = not self.session_actions.can_reanalyze
    
    # Event handlers
    def _on_back_pressed(self, button):
        """Handle back button press."""
        self._trigger_callback('back_requested')
    
    def _on_audio_event(self, event_type: str, *args):
        """Handle audio player events."""
        self._trigger_callback(f'audio_{event_type}', *args)
    
    def _on_view_transcript(self, button):
        """Handle view full transcript button."""
        self._trigger_callback('view_transcript_requested', self.session_metadata.session_id)
    
    def _on_search_transcript(self, button):
        """Handle search transcript button."""
        self._trigger_callback('search_transcript_requested', self.session_metadata.session_id)
    
    def _on_view_analysis(self, button):
        """Handle view full analysis button."""
        self._trigger_callback('view_analysis_requested', self.session_metadata.session_id)
    
    def _on_reanalyze(self, button):
        """Handle reanalyze button."""
        self._trigger_callback('reanalyze_requested', self.session_metadata.session_id)
    
    def _on_export(self, button):
        """Handle export button."""
        self._trigger_callback('export_requested', self.session_metadata.session_id)
    
    def _on_share(self, button):
        """Handle share button."""
        self._trigger_callback('share_requested', self.session_metadata.session_id)
    
    def _on_edit(self, button):
        """Handle edit button."""
        self._trigger_callback('edit_requested', self.session_metadata.session_id)
    
    def _on_delete(self, button):
        """Handle delete button."""
        self._trigger_callback('delete_requested', self.session_metadata.session_id)
    
    def _on_archive(self, button):
        """Handle archive button."""
        self._trigger_callback('archive_requested', self.session_metadata.session_id)
    
    def _on_session_settings(self, button):
        """Handle session settings button."""
        self._trigger_callback('session_settings_requested', self.session_metadata.session_id)
    
    def _on_notes_changed(self, text_input, text):
        """Handle notes text change."""
        if self.session_metadata:
            self.session_metadata.notes = text
    
    def _on_save_notes(self, button):
        """Handle save notes button."""
        self._trigger_callback('save_notes_requested', self.session_metadata.session_id, 
                             self.notes_input.text)
    
    def _on_clear_notes(self, button):
        """Handle clear notes button."""
        self.notes_input.text = ""
        if self.session_metadata:
            self.session_metadata.notes = ""
    
    def _trigger_callback(self, event_type: str, *args):
        """Trigger registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(event_type, *args)
            except Exception as e:
                print(f"Error in session details callback: {e}")
    
    def add_callback(self, callback: Callable):
        """Add a callback for events."""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    # Additional methods from expected manifest
    def play_audio(self):
        """Start audio playback."""
        if hasattr(self, 'audio_player'):
            self.audio_player.set_playing(True)
    
    def view_transcript(self):
        """View full transcript."""
        self._trigger_callback('view_transcript_requested', self.session_metadata.session_id)
    
    def view_analysis(self):
        """View full analysis."""
        self._trigger_callback('view_analysis_requested', self.session_metadata.session_id)
    
    def edit_notes(self):
        """Focus notes editing."""
        self.notes_input.focus = True
    
    def share_session(self):
        """Share session."""
        self._trigger_callback('share_requested', self.session_metadata.session_id)
    
    def export_session(self):
        """Export session."""
        self._trigger_callback('export_requested', self.session_metadata.session_id)
    
    def delete_session(self):
        """Delete session."""
        self._trigger_callback('delete_requested', self.session_metadata.session_id)
    
    def get_session_info(self) -> Optional[SessionMetadata]:
        """Get current session info."""
        return self.session_metadata
    
    def update_session_info(self, metadata: SessionMetadata):
        """Update session information."""
        self.update_metadata(metadata)

# Factory functions
def create_session_details_view(config: Optional[SessionDetailsConfig] = None) -> SessionDetailsView:
    """Create a session details view with optional configuration."""
    return SessionDetailsView(config)

def create_default_config() -> SessionDetailsConfig:
    """Create default session details configuration."""
    return SessionDetailsConfig()

def create_expanded_config() -> SessionDetailsConfig:
    """Create expanded session details configuration."""
    config = SessionDetailsConfig()
    config.show_waveform = True
    config.show_transcript_preview = True
    config.show_analysis_summary = True
    config.show_participant_stats = True
    config.max_transcript_preview = 1000
    return config

# Demo function
def demo_session_details_view():
    """Demo function to test the session details view."""
    from kivy.app import App
    
    class SessionDetailsDemo(App):
        def build(self):
            # Create sample session metadata
            metadata = SessionMetadata(
                session_id="demo_session_1",
                title="Team Meeting - Q3 Planning",
                date=datetime.now(),
                duration=1800,  # 30 minutes
                file_size=50 * 1024 * 1024,  # 50MB
                audio_format="FLAC",
                sample_rate=44100,
                channels=2,
                notes="Important discussion about Q3 goals and objectives."
            )
            metadata.has_transcript = True
            metadata.has_analysis = True
            metadata.transcript_confidence = 0.92
            metadata.speaker_count = 4
            metadata.word_count = 2500
            metadata.analysis_version = "v1.2.3"
            
            # Create session details view
            details_view = create_session_details_view()
            details_view.add_callback(self._on_details_event)
            details_view.load_session(metadata)
            
            return details_view
        
        def _on_details_event(self, event_type: str, *args):
            print(f"Session details event: {event_type}, args: {args}")
    
    return SessionDetailsDemo()

@contextmanager
def session_details_context(config: Optional[SessionDetailsConfig] = None):
    """Context manager for session details operations."""
    details_view = create_session_details_view(config)
    try:
        yield details_view
    finally:
        # Cleanup if needed
        details_view.callbacks.clear()

if __name__ == "__main__":
    demo = demo_session_details_view()
    demo.run()