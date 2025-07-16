"""
Export Dialog - Session export dialog with format options.

This module provides a comprehensive export interface for The Silent Steno project,
supporting multiple formats and selective content export for meeting sessions.
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Set
from enum import Enum
from contextlib import contextmanager

try:
    from kivy.app import App
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.gridlayout import GridLayout
    from kivy.uix.popup import Popup
    from kivy.uix.label import Label
    from kivy.uix.button import Button
    from kivy.uix.checkbox import CheckBox
    from kivy.uix.spinner import Spinner
    from kivy.uix.textinput import TextInput
    from kivy.uix.progressbar import ProgressBar
    from kivy.uix.filechooser import FileChooserIconView
    from kivy.uix.scrollview import ScrollView
    from kivy.properties import StringProperty, NumericProperty, BooleanProperty, ObjectProperty
    from kivy.event import EventDispatcher
    from kivy.clock import Clock
    from kivy.metrics import dp
    from kivy.utils import get_color_from_hex
except ImportError:
    # Fallback for systems without Kivy
    class BoxLayout: pass
    class GridLayout: pass
    class Popup: pass
    class Label: pass
    class Button: pass
    class CheckBox: pass
    class Spinner: pass
    class TextInput: pass
    class ProgressBar: pass
    class FileChooserIconView: pass
    class ScrollView: pass
    class EventDispatcher: pass
    StringProperty = NumericProperty = BooleanProperty = ObjectProperty = lambda x: x
    Clock = None
    dp = lambda x: x
    get_color_from_hex = lambda x: x

# Export formats
class ExportFormat(Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    JSON = "json"
    HTML = "html"

# Export content options
class ExportContent(Enum):
    AUDIO_ONLY = "audio_only"
    TRANSCRIPT_ONLY = "transcript_only"
    ANALYSIS_ONLY = "analysis_only"
    COMPLETE = "complete"

class ExportOptions:
    """Export options and configuration."""
    
    def __init__(self):
        # Format selection
        self.format = ExportFormat.PDF
        self.content = ExportContent.COMPLETE
        
        # Content inclusion
        self.include_audio = True
        self.include_transcript = True
        self.include_analysis = True
        self.include_metadata = True
        self.include_notes = True
        self.include_timestamps = True
        self.include_speaker_labels = True
        
        # Formatting options
        self.page_orientation = "portrait"  # portrait or landscape
        self.font_size = 12
        self.include_cover_page = True
        self.include_table_of_contents = False
        self.watermark = ""
        
        # Quality settings
        self.audio_quality = "high"  # low, medium, high
        self.image_quality = "medium"  # low, medium, high
        
        # Output settings
        self.output_path = ""
        self.filename = ""
        self.open_after_export = True
        
        # Advanced options
        self.split_by_speaker = False
        self.anonymize_speakers = False
        self.password_protect = False
        self.password = ""

class ExportConfig:
    """Configuration for export dialog display and behavior."""
    
    def __init__(self):
        # Dialog dimensions
        self.dialog_width = dp(500)
        self.dialog_height = dp(600)
        
        # Touch optimization
        self.button_height = dp(50)
        self.checkbox_size = dp(30)
        self.spacing = dp(8)
        self.padding = dp(16)
        
        # Supported formats
        self.supported_formats = [ExportFormat.PDF, ExportFormat.DOCX, 
                                ExportFormat.TXT, ExportFormat.JSON, ExportFormat.HTML]
        
        # Default paths
        self.default_export_path = os.path.expanduser("~/Downloads")
        
        # UI options
        self.show_preview = True
        self.show_advanced_options = True
        self.show_progress = True
        
        # Colors
        self.primary_color = get_color_from_hex('#3498DB') if callable(get_color_from_hex) else '#3498DB'
        self.success_color = get_color_from_hex('#27AE60') if callable(get_color_from_hex) else '#27AE60'
        self.error_color = get_color_from_hex('#E74C3C') if callable(get_color_from_hex) else '#E74C3C'

class FormatOptionsWidget(BoxLayout):
    """Widget for format-specific options."""
    
    def __init__(self, format_type: ExportFormat, config: ExportConfig, **kwargs):
        super().__init__(**kwargs)
        self.format_type = format_type
        self.config = config
        self.orientation = 'vertical'
        self.spacing = dp(8)
        self.size_hint_y = None
        self.height = dp(200)
        
        self._build_format_options()
    
    def _build_format_options(self):
        """Build format-specific options."""
        if self.format_type == ExportFormat.PDF:
            self._build_pdf_options()
        elif self.format_type == ExportFormat.DOCX:
            self._build_docx_options()
        elif self.format_type == ExportFormat.HTML:
            self._build_html_options()
        else:
            # Default options for TXT and JSON
            self._build_default_options()
    
    def _build_pdf_options(self):
        """Build PDF-specific options."""
        # Page orientation
        orientation_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40))
        orientation_layout.add_widget(Label(text="Orientation:", size_hint_x=0.3, font_size=dp(12)))
        
        self.orientation_spinner = Spinner(
            text="Portrait",
            values=["Portrait", "Landscape"],
            size_hint_x=0.7,
            font_size=dp(12)
        )
        orientation_layout.add_widget(self.orientation_spinner)
        self.add_widget(orientation_layout)
        
        # Include cover page
        cover_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40))
        self.cover_checkbox = CheckBox(size_hint_x=None, width=dp(30), active=True)
        cover_layout.add_widget(self.cover_checkbox)
        cover_layout.add_widget(Label(text="Include cover page", font_size=dp(12)))
        self.add_widget(cover_layout)
        
        # Include table of contents
        toc_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40))
        self.toc_checkbox = CheckBox(size_hint_x=None, width=dp(30), active=False)
        toc_layout.add_widget(self.toc_checkbox)
        toc_layout.add_widget(Label(text="Include table of contents", font_size=dp(12)))
        self.add_widget(toc_layout)
        
        # Font size
        font_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40))
        font_layout.add_widget(Label(text="Font size:", size_hint_x=0.3, font_size=dp(12)))
        
        self.font_spinner = Spinner(
            text="12",
            values=["8", "10", "11", "12", "14", "16", "18"],
            size_hint_x=0.7,
            font_size=dp(12)
        )
        font_layout.add_widget(self.font_spinner)
        self.add_widget(font_layout)
        
        # Watermark
        watermark_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40))
        watermark_layout.add_widget(Label(text="Watermark:", size_hint_x=0.3, font_size=dp(12)))
        
        self.watermark_input = TextInput(
            text="",
            hint_text="Optional watermark text",
            size_hint_x=0.7,
            multiline=False,
            font_size=dp(12)
        )
        watermark_layout.add_widget(self.watermark_input)
        self.add_widget(watermark_layout)
    
    def _build_docx_options(self):
        """Build DOCX-specific options."""
        # Template selection
        template_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40))
        template_layout.add_widget(Label(text="Template:", size_hint_x=0.3, font_size=dp(12)))
        
        self.template_spinner = Spinner(
            text="Default",
            values=["Default", "Professional", "Minimal", "Corporate"],
            size_hint_x=0.7,
            font_size=dp(12)
        )
        template_layout.add_widget(self.template_spinner)
        self.add_widget(template_layout)
        
        # Include comments
        comments_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40))
        self.comments_checkbox = CheckBox(size_hint_x=None, width=dp(30), active=True)
        comments_layout.add_widget(self.comments_checkbox)
        comments_layout.add_widget(Label(text="Include comments and notes", font_size=dp(12)))
        self.add_widget(comments_layout)
        
        # Track changes
        track_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40))
        self.track_checkbox = CheckBox(size_hint_x=None, width=dp(30), active=False)
        track_layout.add_widget(self.track_checkbox)
        track_layout.add_widget(Label(text="Enable track changes", font_size=dp(12)))
        self.add_widget(track_layout)
    
    def _build_html_options(self):
        """Build HTML-specific options."""
        # CSS styling
        css_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40))
        css_layout.add_widget(Label(text="Style:", size_hint_x=0.3, font_size=dp(12)))
        
        self.css_spinner = Spinner(
            text="Default",
            values=["Default", "Minimal", "Professional", "Dark", "Print-friendly"],
            size_hint_x=0.7,
            font_size=dp(12)
        )
        css_layout.add_widget(self.css_spinner)
        self.add_widget(css_layout)
        
        # Include JavaScript
        js_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40))
        self.js_checkbox = CheckBox(size_hint_x=None, width=dp(30), active=True)
        js_layout.add_widget(self.js_checkbox)
        js_layout.add_widget(Label(text="Include interactive features", font_size=dp(12)))
        self.add_widget(js_layout)
        
        # Embed audio
        embed_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40))
        self.embed_checkbox = CheckBox(size_hint_x=None, width=dp(30), active=False)
        embed_layout.add_widget(self.embed_checkbox)
        embed_layout.add_widget(Label(text="Embed audio player", font_size=dp(12)))
        self.add_widget(embed_layout)
    
    def _build_default_options(self):
        """Build default options for simple formats."""
        # Character encoding
        encoding_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40))
        encoding_layout.add_widget(Label(text="Encoding:", size_hint_x=0.3, font_size=dp(12)))
        
        self.encoding_spinner = Spinner(
            text="UTF-8",
            values=["UTF-8", "UTF-16", "ASCII"],
            size_hint_x=0.7,
            font_size=dp(12)
        )
        encoding_layout.add_widget(self.encoding_spinner)
        self.add_widget(encoding_layout)
        
        # Line endings
        line_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40))
        line_layout.add_widget(Label(text="Line endings:", size_hint_x=0.3, font_size=dp(12)))
        
        self.line_spinner = Spinner(
            text="Unix (LF)",
            values=["Unix (LF)", "Windows (CRLF)", "Mac (CR)"],
            size_hint_x=0.7,
            font_size=dp(12)
        )
        line_layout.add_widget(self.line_spinner)
        self.add_widget(line_layout)
    
    def get_options(self) -> Dict[str, Any]:
        """Get format-specific options as dictionary."""
        options = {}
        
        if self.format_type == ExportFormat.PDF:
            options['orientation'] = self.orientation_spinner.text.lower()
            options['include_cover'] = self.cover_checkbox.active
            options['include_toc'] = self.toc_checkbox.active
            options['font_size'] = int(self.font_spinner.text)
            options['watermark'] = self.watermark_input.text
        elif self.format_type == ExportFormat.DOCX:
            options['template'] = self.template_spinner.text.lower()
            options['include_comments'] = self.comments_checkbox.active
            options['track_changes'] = self.track_checkbox.active
        elif self.format_type == ExportFormat.HTML:
            options['css_style'] = self.css_spinner.text.lower()
            options['include_javascript'] = self.js_checkbox.active
            options['embed_audio'] = self.embed_checkbox.active
        else:
            options['encoding'] = self.encoding_spinner.text
            options['line_endings'] = self.line_spinner.text
        
        return options

class ExportDialog(Popup, EventDispatcher):
    """Main export dialog with comprehensive options."""
    
    def __init__(self, session_ids: List[str], config: Optional[ExportConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.session_ids = session_ids
        self.config = config or ExportConfig()
        self.export_options = ExportOptions()
        self.callbacks = []
        
        # Dialog properties
        self.title = f"Export {len(session_ids)} Session(s)"
        self.size_hint = (None, None)
        self.size = (self.config.dialog_width, self.config.dialog_height)
        self.auto_dismiss = False
        
        # State
        self.export_in_progress = False
        self.export_progress = 0.0
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the export dialog UI."""
        main_layout = BoxLayout(
            orientation='vertical',
            spacing=self.config.spacing,
            padding=self.config.padding
        )
        
        # Content selection section
        content_section = self._build_content_section()
        main_layout.add_widget(content_section)
        
        # Format selection section
        format_section = self._build_format_section()
        main_layout.add_widget(format_section)
        
        # Output settings section
        output_section = self._build_output_section()
        main_layout.add_widget(output_section)
        
        # Advanced options (collapsible)
        if self.config.show_advanced_options:
            advanced_section = self._build_advanced_section()
            main_layout.add_widget(advanced_section)
        
        # Progress bar (initially hidden)
        if self.config.show_progress:
            self.progress_layout = BoxLayout(
                orientation='vertical',
                size_hint_y=None,
                height=0,  # Initially hidden
                spacing=dp(4)
            )
            
            self.progress_label = Label(
                text="Export progress...",
                font_size=dp(12),
                size_hint_y=None,
                height=dp(20)
            )
            self.progress_layout.add_widget(self.progress_label)
            
            self.progress_bar = ProgressBar(
                max=100,
                value=0,
                size_hint_y=None,
                height=dp(20)
            )
            self.progress_layout.add_widget(self.progress_bar)
            
            main_layout.add_widget(self.progress_layout)
        
        # Action buttons
        buttons_layout = self._build_buttons_section()
        main_layout.add_widget(buttons_layout)
        
        self.content = main_layout
    
    def _build_content_section(self) -> BoxLayout:
        """Build content selection section."""
        section = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=dp(200),
            spacing=dp(4)
        )
        
        # Section title
        title = Label(
            text="Content to Export",
            font_size=dp(16),
            bold=True,
            size_hint_y=None,
            height=dp(30),
            halign='left'
        )
        title.bind(size=title.setter('text_size'))
        section.add_widget(title)
        
        # Content checkboxes
        content_grid = GridLayout(
            cols=2,
            size_hint_y=None,
            height=dp(160),
            spacing=dp(4)
        )
        
        # Audio checkbox
        audio_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(30))
        self.audio_checkbox = CheckBox(size_hint_x=None, width=dp(30), active=True)
        self.audio_checkbox.bind(active=self._on_audio_toggled)
        audio_layout.add_widget(self.audio_checkbox)
        audio_layout.add_widget(Label(text="Audio files", font_size=dp(12)))
        content_grid.add_widget(audio_layout)
        
        # Transcript checkbox
        transcript_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(30))
        self.transcript_checkbox = CheckBox(size_hint_x=None, width=dp(30), active=True)
        self.transcript_checkbox.bind(active=self._on_transcript_toggled)
        transcript_layout.add_widget(self.transcript_checkbox)
        transcript_layout.add_widget(Label(text="Transcripts", font_size=dp(12)))
        content_grid.add_widget(transcript_layout)
        
        # Analysis checkbox
        analysis_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(30))
        self.analysis_checkbox = CheckBox(size_hint_x=None, width=dp(30), active=True)
        self.analysis_checkbox.bind(active=self._on_analysis_toggled)
        analysis_layout.add_widget(self.analysis_checkbox)
        analysis_layout.add_widget(Label(text="AI Analysis", font_size=dp(12)))
        content_grid.add_widget(analysis_layout)
        
        # Metadata checkbox
        metadata_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(30))
        self.metadata_checkbox = CheckBox(size_hint_x=None, width=dp(30), active=True)
        self.metadata_checkbox.bind(active=self._on_metadata_toggled)
        metadata_layout.add_widget(self.metadata_checkbox)
        metadata_layout.add_widget(Label(text="Session metadata", font_size=dp(12)))
        content_grid.add_widget(metadata_layout)
        
        # Notes checkbox
        notes_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(30))
        self.notes_checkbox = CheckBox(size_hint_x=None, width=dp(30), active=True)
        self.notes_checkbox.bind(active=self._on_notes_toggled)
        notes_layout.add_widget(self.notes_checkbox)
        notes_layout.add_widget(Label(text="Notes", font_size=dp(12)))
        content_grid.add_widget(notes_layout)
        
        # Timestamps checkbox
        timestamps_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(30))
        self.timestamps_checkbox = CheckBox(size_hint_x=None, width=dp(30), active=True)
        self.timestamps_checkbox.bind(active=self._on_timestamps_toggled)
        timestamps_layout.add_widget(self.timestamps_checkbox)
        timestamps_layout.add_widget(Label(text="Timestamps", font_size=dp(12)))
        content_grid.add_widget(timestamps_layout)
        
        # Speaker labels checkbox
        speakers_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(30))
        self.speakers_checkbox = CheckBox(size_hint_x=None, width=dp(30), active=True)
        self.speakers_checkbox.bind(active=self._on_speakers_toggled)
        speakers_layout.add_widget(self.speakers_checkbox)
        speakers_layout.add_widget(Label(text="Speaker labels", font_size=dp(12)))
        content_grid.add_widget(speakers_layout)
        
        section.add_widget(content_grid)
        
        return section
    
    def _build_format_section(self) -> BoxLayout:
        """Build format selection section."""
        section = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=dp(300),
            spacing=dp(4)
        )
        
        # Section title
        title = Label(
            text="Export Format",
            font_size=dp(16),
            bold=True,
            size_hint_y=None,
            height=dp(30),
            halign='left'
        )
        title.bind(size=title.setter('text_size'))
        section.add_widget(title)
        
        # Format selection
        format_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(40),
            spacing=dp(8)
        )
        
        format_layout.add_widget(Label(text="Format:", size_hint_x=0.3, font_size=dp(12)))
        
        format_values = [fmt.value.upper() for fmt in self.config.supported_formats]
        self.format_spinner = Spinner(
            text=format_values[0],
            values=format_values,
            size_hint_x=0.7,
            font_size=dp(12)
        )
        self.format_spinner.bind(text=self._on_format_changed)
        format_layout.add_widget(self.format_spinner)
        
        section.add_widget(format_layout)
        
        # Format-specific options container
        self.format_options_container = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=dp(220)
        )
        
        # Initialize with default format options
        self._update_format_options(ExportFormat.PDF)
        
        section.add_widget(self.format_options_container)
        
        return section
    
    def _build_output_section(self) -> BoxLayout:
        """Build output settings section."""
        section = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=dp(120),
            spacing=dp(4)
        )
        
        # Section title
        title = Label(
            text="Output Settings",
            font_size=dp(16),
            bold=True,
            size_hint_y=None,
            height=dp(30),
            halign='left'
        )
        title.bind(size=title.setter('text_size'))
        section.add_widget(title)
        
        # Output path
        path_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(40),
            spacing=dp(8)
        )
        
        path_layout.add_widget(Label(text="Path:", size_hint_x=0.2, font_size=dp(12)))
        
        self.path_input = TextInput(
            text=self.config.default_export_path,
            size_hint_x=0.6,
            multiline=False,
            font_size=dp(11)
        )
        self.path_input.bind(text=self._on_path_changed)
        path_layout.add_widget(self.path_input)
        
        self.browse_btn = Button(
            text="Browse",
            size_hint_x=0.2,
            font_size=dp(12)
        )
        self.browse_btn.bind(on_press=self._on_browse_path)
        path_layout.add_widget(self.browse_btn)
        
        section.add_widget(path_layout)
        
        # Filename
        filename_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(40),
            spacing=dp(8)
        )
        
        filename_layout.add_widget(Label(text="Filename:", size_hint_x=0.2, font_size=dp(12)))
        
        self.filename_input = TextInput(
            text=self._generate_default_filename(),
            size_hint_x=0.8,
            multiline=False,
            font_size=dp(11)
        )
        self.filename_input.bind(text=self._on_filename_changed)
        filename_layout.add_widget(self.filename_input)
        
        section.add_widget(filename_layout)
        
        return section
    
    def _build_advanced_section(self) -> BoxLayout:
        """Build advanced options section."""
        section = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=dp(100),
            spacing=dp(4)
        )
        
        # Section title (collapsible)
        title_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(30)
        )
        
        self.advanced_toggle = Button(
            text="▶ Advanced Options",
            size_hint_x=1,
            font_size=dp(14),
            halign='left'
        )
        self.advanced_toggle.bind(on_press=self._on_advanced_toggle)
        title_layout.add_widget(self.advanced_toggle)
        
        section.add_widget(title_layout)
        
        # Advanced options container (initially collapsed)
        self.advanced_container = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=0,  # Initially collapsed
            spacing=dp(4)
        )
        
        # Open after export
        open_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(30))
        self.open_checkbox = CheckBox(size_hint_x=None, width=dp(30), active=True)
        self.open_checkbox.bind(active=self._on_open_toggled)
        open_layout.add_widget(self.open_checkbox)
        open_layout.add_widget(Label(text="Open after export", font_size=dp(12)))
        self.advanced_container.add_widget(open_layout)
        
        # Password protection
        password_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(30))
        self.password_checkbox = CheckBox(size_hint_x=None, width=dp(30), active=False)
        self.password_checkbox.bind(active=self._on_password_toggled)
        password_layout.add_widget(self.password_checkbox)
        password_layout.add_widget(Label(text="Password protect", size_hint_x=0.4, font_size=dp(12)))
        
        self.password_input = TextInput(
            text="",
            password=True,
            disabled=True,
            size_hint_x=0.5,
            multiline=False,
            font_size=dp(11)
        )
        password_layout.add_widget(self.password_input)
        self.advanced_container.add_widget(password_layout)
        
        section.add_widget(self.advanced_container)
        
        return section
    
    def _build_buttons_section(self) -> BoxLayout:
        """Build action buttons section."""
        buttons_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=self.config.button_height,
            spacing=self.config.spacing
        )
        
        # Cancel button
        self.cancel_btn = Button(
            text="Cancel",
            size_hint_x=0.3,
            font_size=dp(14)
        )
        self.cancel_btn.bind(on_press=self._on_cancel)
        buttons_layout.add_widget(self.cancel_btn)
        
        # Preview button (if enabled)
        if self.config.show_preview:
            self.preview_btn = Button(
                text="Preview",
                size_hint_x=0.35,
                font_size=dp(14)
            )
            self.preview_btn.bind(on_press=self._on_preview)
            buttons_layout.add_widget(self.preview_btn)
        
        # Export button
        self.export_btn = Button(
            text="Export",
            size_hint_x=0.35,
            font_size=dp(14)
        )
        self.export_btn.bind(on_press=self._on_export)
        buttons_layout.add_widget(self.export_btn)
        
        return buttons_layout
    
    def _update_format_options(self, format_type: ExportFormat):
        """Update format-specific options display."""
        self.format_options_container.clear_widgets()
        
        format_options = FormatOptionsWidget(format_type, self.config)
        self.format_options_container.add_widget(format_options)
        self.format_options_widget = format_options
    
    def _generate_default_filename(self) -> str:
        """Generate default filename based on session count and date."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(self.session_ids) == 1:
            return f"session_export_{timestamp}"
        else:
            return f"sessions_export_{len(self.session_ids)}_items_{timestamp}"
    
    def _collect_export_options(self) -> ExportOptions:
        """Collect all export options from UI."""
        options = ExportOptions()
        
        # Content options
        options.include_audio = self.audio_checkbox.active
        options.include_transcript = self.transcript_checkbox.active
        options.include_analysis = self.analysis_checkbox.active
        options.include_metadata = self.metadata_checkbox.active
        options.include_notes = self.notes_checkbox.active
        options.include_timestamps = self.timestamps_checkbox.active
        options.include_speaker_labels = self.speakers_checkbox.active
        
        # Format and output
        options.format = ExportFormat(self.format_spinner.text.lower())
        options.output_path = self.path_input.text
        options.filename = self.filename_input.text
        
        # Advanced options
        options.open_after_export = self.open_checkbox.active
        options.password_protect = self.password_checkbox.active
        if options.password_protect:
            options.password = self.password_input.text
        
        # Format-specific options
        if hasattr(self, 'format_options_widget'):
            format_options = self.format_options_widget.get_options()
            for key, value in format_options.items():
                setattr(options, key, value)
        
        return options
    
    # Event handlers
    def _on_audio_toggled(self, checkbox, active):
        """Handle audio checkbox toggle."""
        self.export_options.include_audio = active
    
    def _on_transcript_toggled(self, checkbox, active):
        """Handle transcript checkbox toggle."""
        self.export_options.include_transcript = active
    
    def _on_analysis_toggled(self, checkbox, active):
        """Handle analysis checkbox toggle."""
        self.export_options.include_analysis = active
    
    def _on_metadata_toggled(self, checkbox, active):
        """Handle metadata checkbox toggle."""
        self.export_options.include_metadata = active
    
    def _on_notes_toggled(self, checkbox, active):
        """Handle notes checkbox toggle."""
        self.export_options.include_notes = active
    
    def _on_timestamps_toggled(self, checkbox, active):
        """Handle timestamps checkbox toggle."""
        self.export_options.include_timestamps = active
    
    def _on_speakers_toggled(self, checkbox, active):
        """Handle speaker labels checkbox toggle."""
        self.export_options.include_speaker_labels = active
    
    def _on_format_changed(self, spinner, text):
        """Handle format selection change."""
        try:
            format_type = ExportFormat(text.lower())
            self._update_format_options(format_type)
            self.export_options.format = format_type
        except ValueError:
            pass
    
    def _on_path_changed(self, text_input, text):
        """Handle output path change."""
        self.export_options.output_path = text
    
    def _on_filename_changed(self, text_input, text):
        """Handle filename change."""
        self.export_options.filename = text
    
    def _on_browse_path(self, button):
        """Handle browse path button."""
        # In a real implementation, this would open a file browser
        self._trigger_callback('browse_path_requested')
    
    def _on_advanced_toggle(self, button):
        """Handle advanced options toggle."""
        if self.advanced_container.height == 0:
            # Expand
            self.advanced_container.height = dp(70)
            self.advanced_toggle.text = "▼ Advanced Options"
        else:
            # Collapse
            self.advanced_container.height = 0
            self.advanced_toggle.text = "▶ Advanced Options"
    
    def _on_open_toggled(self, checkbox, active):
        """Handle open after export toggle."""
        self.export_options.open_after_export = active
    
    def _on_password_toggled(self, checkbox, active):
        """Handle password protection toggle."""
        self.export_options.password_protect = active
        self.password_input.disabled = not active
    
    def _on_preview(self, button):
        """Handle preview button."""
        options = self._collect_export_options()
        self._trigger_callback('preview_requested', self.session_ids, options)
    
    def _on_export(self, button):
        """Handle export button."""
        options = self._collect_export_options()
        
        # Validate options
        if not self._validate_export_options(options):
            return
        
        # Start export process
        self.export_in_progress = True
        self._show_progress()
        self._trigger_callback('export_requested', self.session_ids, options)
    
    def _on_cancel(self, button):
        """Handle cancel button."""
        if self.export_in_progress:
            self._trigger_callback('export_cancelled')
        self.dismiss()
    
    def _validate_export_options(self, options: ExportOptions) -> bool:
        """Validate export options."""
        # Check if at least one content type is selected
        if not any([options.include_audio, options.include_transcript, 
                   options.include_analysis, options.include_metadata, options.include_notes]):
            self._show_error("Please select at least one content type to export.")
            return False
        
        # Check output path
        if not options.output_path:
            self._show_error("Please specify an output path.")
            return False
        
        # Check filename
        if not options.filename:
            self._show_error("Please specify a filename.")
            return False
        
        # Check password if protection is enabled
        if options.password_protect and not options.password:
            self._show_error("Please enter a password for protection.")
            return False
        
        return True
    
    def _show_error(self, message: str):
        """Show error message to user."""
        error_popup = Popup(
            title="Export Error",
            content=Label(text=message, text_size=(dp(300), None), halign='center'),
            size_hint=(None, None),
            size=(dp(350), dp(200))
        )
        error_popup.open()
    
    def _show_progress(self):
        """Show progress bar."""
        if hasattr(self, 'progress_layout'):
            self.progress_layout.height = dp(50)
    
    def _hide_progress(self):
        """Hide progress bar."""
        if hasattr(self, 'progress_layout'):
            self.progress_layout.height = 0
    
    def update_progress(self, progress: float, status: str = ""):
        """Update export progress."""
        if hasattr(self, 'progress_bar'):
            self.progress_bar.value = progress
        if hasattr(self, 'progress_label') and status:
            self.progress_label.text = status
    
    def export_completed(self, success: bool, output_path: str = ""):
        """Handle export completion."""
        self.export_in_progress = False
        self._hide_progress()
        
        if success:
            self._trigger_callback('export_completed', output_path)
            self.dismiss()
        else:
            self._show_error("Export failed. Please try again.")
    
    def _trigger_callback(self, event_type: str, *args):
        """Trigger registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(event_type, *args)
            except Exception as e:
                print(f"Error in export dialog callback: {e}")
    
    def add_callback(self, callback: Callable):
        """Add a callback for events."""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    # Additional methods from expected manifest
    def set_export_options(self, options: ExportOptions):
        """Set export options programmatically."""
        self.export_options = options
        # Update UI to reflect options
        self.audio_checkbox.active = options.include_audio
        self.transcript_checkbox.active = options.include_transcript
        self.analysis_checkbox.active = options.include_analysis
        self.metadata_checkbox.active = options.include_metadata
        self.notes_checkbox.active = options.include_notes
        self.timestamps_checkbox.active = options.include_timestamps
        self.speakers_checkbox.active = options.include_speaker_labels
        
        self.format_spinner.text = options.format.value.upper()
        self.path_input.text = options.output_path
        self.filename_input.text = options.filename
        self.open_checkbox.active = options.open_after_export
        self.password_checkbox.active = options.password_protect
        if options.password_protect:
            self.password_input.text = options.password
    
    def validate_export(self) -> bool:
        """Validate current export configuration."""
        options = self._collect_export_options()
        return self._validate_export_options(options)
    
    def get_export_progress(self) -> float:
        """Get current export progress."""
        return self.export_progress
    
    def cancel_export(self):
        """Cancel ongoing export."""
        if self.export_in_progress:
            self._trigger_callback('export_cancelled')
            self.export_in_progress = False
            self._hide_progress()

# Factory functions and utilities
def create_export_dialog(session_ids: List[str], 
                        config: Optional[ExportConfig] = None) -> ExportDialog:
    """Create an export dialog for the specified sessions."""
    return ExportDialog(session_ids, config)

def create_default_config() -> ExportConfig:
    """Create default export configuration."""
    return ExportConfig()

def show_export_dialog(session_ids: List[str], 
                      config: Optional[ExportConfig] = None) -> ExportDialog:
    """Show export dialog and return reference."""
    dialog = create_export_dialog(session_ids, config)
    dialog.open()
    return dialog

def export_session(session_id: str, options: ExportOptions) -> bool:
    """Export a single session with given options."""
    # This would be implemented by the actual export system
    print(f"Exporting session {session_id} with format {options.format.value}")
    return True

def export_multiple(session_ids: List[str], options: ExportOptions) -> bool:
    """Export multiple sessions with given options."""
    # This would be implemented by the actual export system
    print(f"Exporting {len(session_ids)} sessions with format {options.format.value}")
    return True

# Demo function
def demo_export_dialog():
    """Demo function to test the export dialog."""
    from kivy.app import App
    
    class ExportDialogDemo(App):
        def build(self):
            from kivy.uix.button import Button
            
            button = Button(text="Open Export Dialog")
            button.bind(on_press=self._show_export_dialog)
            return button
        
        def _show_export_dialog(self, button):
            session_ids = ["session_1", "session_2", "session_3"]
            dialog = create_export_dialog(session_ids)
            dialog.add_callback(self._on_export_event)
            dialog.open()
        
        def _on_export_event(self, event_type: str, *args):
            print(f"Export dialog event: {event_type}, args: {args}")
    
    return ExportDialogDemo()

@contextmanager
def export_dialog_context(session_ids: List[str], 
                         config: Optional[ExportConfig] = None):
    """Context manager for export dialog operations."""
    dialog = create_export_dialog(session_ids, config)
    try:
        yield dialog
    finally:
        # Cleanup if needed
        dialog.callbacks.clear()
        if dialog.parent:
            dialog.dismiss()

if __name__ == "__main__":
    demo = demo_export_dialog()
    demo.run()