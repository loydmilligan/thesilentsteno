"""
Settings View - Application settings and configuration screen.

This module provides a comprehensive settings interface for The Silent Steno project,
allowing users to configure all major application features with categorized options
and real-time validation.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List, Union
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
    from kivy.uix.slider import Slider
    from kivy.uix.checkbox import CheckBox
    from kivy.uix.spinner import Spinner
    from kivy.uix.popup import Popup
    from kivy.uix.accordion import Accordion, AccordionItem
    from kivy.uix.filechooser import FileChooserIconView
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
    class Slider: pass
    class CheckBox: pass
    class Spinner: pass
    class Popup: pass
    class Accordion: pass
    class AccordionItem: pass
    class FileChooserIconView: pass
    class EventDispatcher: pass
    StringProperty = NumericProperty = BooleanProperty = ObjectProperty = lambda x: x
    Clock = None
    dp = lambda x: x
    get_color_from_hex = lambda x: x

# Setting categories
class SettingCategory(Enum):
    GENERAL = "general"
    AUDIO = "audio"
    AI = "ai"
    STORAGE = "storage"
    NETWORK = "network"
    ABOUT = "about"

class SettingType(Enum):
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    CHOICE = "choice"
    PATH = "path"
    COLOR = "color"
    SLIDER = "slider"

class SettingItem:
    """Represents a single setting item."""
    
    def __init__(self, key: str, title: str, description: str, 
                 setting_type: SettingType, default_value: Any,
                 choices: List[str] = None, min_value: float = None, 
                 max_value: float = None, validation_func: Callable = None):
        self.key = key
        self.title = title
        self.description = description
        self.type = setting_type
        self.default_value = default_value
        self.current_value = default_value
        self.choices = choices or []
        self.min_value = min_value
        self.max_value = max_value
        self.validation_func = validation_func
        self.requires_restart = False
    
    def validate(self, value: Any) -> tuple[bool, str]:
        """Validate a setting value."""
        try:
            # Type validation
            if self.type == SettingType.BOOLEAN:
                if not isinstance(value, bool):
                    return False, "Value must be True or False"
            elif self.type == SettingType.INTEGER:
                if not isinstance(value, int):
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        return False, "Value must be an integer"
                if self.min_value is not None and value < self.min_value:
                    return False, f"Value must be at least {self.min_value}"
                if self.max_value is not None and value > self.max_value:
                    return False, f"Value must be at most {self.max_value}"
            elif self.type == SettingType.FLOAT:
                if not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        return False, "Value must be a number"
                if self.min_value is not None and value < self.min_value:
                    return False, f"Value must be at least {self.min_value}"
                if self.max_value is not None and value > self.max_value:
                    return False, f"Value must be at most {self.max_value}"
            elif self.type == SettingType.STRING:
                if not isinstance(value, str):
                    return False, "Value must be a string"
            elif self.type == SettingType.CHOICE:
                if value not in self.choices:
                    return False, f"Value must be one of: {', '.join(self.choices)}"
            elif self.type == SettingType.PATH:
                if not isinstance(value, str):
                    return False, "Path must be a string"
                # Basic path validation - could be enhanced
                if value and not os.path.isabs(value) and not os.path.exists(os.path.dirname(value)):
                    return False, "Invalid path"
            
            # Custom validation
            if self.validation_func:
                custom_valid, custom_msg = self.validation_func(value)
                if not custom_valid:
                    return False, custom_msg
            
            return True, ""
        except Exception as e:
            return False, f"Validation error: {str(e)}"

class SettingsConfig:
    """Configuration for settings view display and behavior."""
    
    def __init__(self):
        # Display options
        self.show_search = True
        self.show_reset_buttons = True
        self.show_import_export = True
        self.show_about = True
        
        # Touch optimization
        self.item_height = dp(60)
        self.header_height = dp(40)
        self.spacing = dp(8)
        self.padding = dp(16)
        
        # Validation
        self.real_time_validation = True
        self.show_validation_icons = True
        
        # File paths
        self.settings_file = "config/settings.json"
        self.backup_file = "config/settings_backup.json"
        
        # Colors
        self.primary_color = get_color_from_hex('#3498DB') if callable(get_color_from_hex) else '#3498DB'
        self.success_color = get_color_from_hex('#27AE60') if callable(get_color_from_hex) else '#27AE60'
        self.error_color = get_color_from_hex('#E74C3C') if callable(get_color_from_hex) else '#E74C3C'
        self.warning_color = get_color_from_hex('#F39C12') if callable(get_color_from_hex) else '#F39C12'

class SettingWidget(BoxLayout):
    """Widget for individual setting items."""
    
    def __init__(self, setting_item: SettingItem, config: SettingsConfig, **kwargs):
        super().__init__(**kwargs)
        self.setting_item = setting_item
        self.config = config
        self.orientation = 'vertical'
        self.size_hint_y = None
        self.height = self.config.item_height
        self.spacing = dp(4)
        self.callbacks = []
        
        self._build_ui()
        self._update_value_display()
    
    def _build_ui(self):
        """Build the setting widget UI."""
        # Header with title and validation icon
        header_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(30)
        )
        
        self.title_label = Label(
            text=self.setting_item.title,
            font_size=dp(14),
            bold=True,
            size_hint_x=0.8,
            halign='left',
            valign='middle'
        )
        self.title_label.bind(size=self.title_label.setter('text_size'))
        header_layout.add_widget(self.title_label)
        
        if self.config.show_validation_icons:
            self.validation_icon = Label(
                text="✓",
                font_size=dp(16),
                size_hint_x=None,
                width=dp(30),
                color=self.config.success_color
            )
            header_layout.add_widget(self.validation_icon)
        
        self.add_widget(header_layout)
        
        # Value input based on setting type
        self.value_widget = self._create_value_widget()
        self.add_widget(self.value_widget)
        
        # Description label
        if self.setting_item.description:
            self.description_label = Label(
                text=self.setting_item.description,
                font_size=dp(11),
                size_hint_y=None,
                height=dp(20),
                halign='left',
                valign='top',
                color=(0.7, 0.7, 0.7, 1)
            )
            self.description_label.bind(size=self.description_label.setter('text_size'))
            # Only add if there's space
            if self.config.item_height > dp(60):
                self.add_widget(self.description_label)
    
    def _create_value_widget(self) -> BoxLayout:
        """Create the value input widget based on setting type."""
        layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(40),
            spacing=dp(8)
        )
        
        if self.setting_item.type == SettingType.BOOLEAN:
            self.input_widget = CheckBox(
                size_hint_x=None,
                width=dp(40),
                active=bool(self.setting_item.current_value)
            )
            self.input_widget.bind(active=self._on_value_changed)
            layout.add_widget(self.input_widget)
            
        elif self.setting_item.type == SettingType.CHOICE:
            self.input_widget = Spinner(
                text=str(self.setting_item.current_value),
                values=self.setting_item.choices,
                size_hint_x=0.6,
                font_size=dp(12)
            )
            self.input_widget.bind(text=self._on_value_changed)
            layout.add_widget(self.input_widget)
            
        elif self.setting_item.type == SettingType.SLIDER:
            # Slider with value label
            self.input_widget = Slider(
                min=self.setting_item.min_value or 0,
                max=self.setting_item.max_value or 100,
                value=float(self.setting_item.current_value),
                size_hint_x=0.7,
                step=1 if self.setting_item.type == SettingType.INTEGER else 0.1
            )
            self.input_widget.bind(value=self._on_value_changed)
            layout.add_widget(self.input_widget)
            
            self.value_label = Label(
                text=str(self.setting_item.current_value),
                size_hint_x=0.3,
                font_size=dp(12)
            )
            layout.add_widget(self.value_label)
            
        elif self.setting_item.type == SettingType.PATH:
            self.input_widget = TextInput(
                text=str(self.setting_item.current_value),
                size_hint_x=0.7,
                multiline=False,
                font_size=dp(11)
            )
            self.input_widget.bind(text=self._on_value_changed)
            layout.add_widget(self.input_widget)
            
            browse_btn = Button(
                text="Browse",
                size_hint_x=0.3,
                font_size=dp(11)
            )
            browse_btn.bind(on_press=self._on_browse_path)
            layout.add_widget(browse_btn)
            
        else:
            # Default text input for string, integer, float
            self.input_widget = TextInput(
                text=str(self.setting_item.current_value),
                size_hint_x=1,
                multiline=False,
                font_size=dp(12)
            )
            self.input_widget.bind(text=self._on_value_changed)
            layout.add_widget(self.input_widget)
        
        return layout
    
    def _update_value_display(self):
        """Update the value display based on current setting value."""
        if self.setting_item.type == SettingType.BOOLEAN:
            self.input_widget.active = bool(self.setting_item.current_value)
        elif self.setting_item.type == SettingType.CHOICE:
            self.input_widget.text = str(self.setting_item.current_value)
        elif self.setting_item.type == SettingType.SLIDER:
            self.input_widget.value = float(self.setting_item.current_value)
            if hasattr(self, 'value_label'):
                self.value_label.text = str(self.setting_item.current_value)
        else:
            self.input_widget.text = str(self.setting_item.current_value)
    
    def _on_value_changed(self, widget, value):
        """Handle value change."""
        # Convert value based on setting type
        try:
            if self.setting_item.type == SettingType.BOOLEAN:
                new_value = bool(value)
            elif self.setting_item.type == SettingType.INTEGER:
                new_value = int(float(value)) if isinstance(value, str) else int(value)
            elif self.setting_item.type == SettingType.FLOAT or self.setting_item.type == SettingType.SLIDER:
                new_value = float(value)
                if hasattr(self, 'value_label'):
                    self.value_label.text = f"{new_value:.1f}" if new_value != int(new_value) else str(int(new_value))
            else:
                new_value = str(value)
            
            # Validate the new value
            if self.config.real_time_validation:
                is_valid, error_msg = self.setting_item.validate(new_value)
                self._update_validation_display(is_valid, error_msg)
            
            # Update setting value
            self.setting_item.current_value = new_value
            
            # Trigger callback
            self._trigger_callback('setting_changed', self.setting_item.key, new_value)
            
        except (ValueError, TypeError) as e:
            if self.config.real_time_validation:
                self._update_validation_display(False, str(e))
    
    def _on_browse_path(self, button):
        """Handle browse path button."""
        self._trigger_callback('browse_path_requested', self.setting_item.key)
    
    def _update_validation_display(self, is_valid: bool, error_msg: str = ""):
        """Update validation icon and color."""
        if hasattr(self, 'validation_icon'):
            if is_valid:
                self.validation_icon.text = "✓"
                self.validation_icon.color = self.config.success_color
            else:
                self.validation_icon.text = "✗"
                self.validation_icon.color = self.config.error_color
        
        # Update input widget color
        if hasattr(self.input_widget, 'background_color'):
            if is_valid:
                self.input_widget.background_color = (1, 1, 1, 1)
            else:
                self.input_widget.background_color = (1, 0.9, 0.9, 1)
    
    def set_value(self, value: Any):
        """Set the setting value programmatically."""
        self.setting_item.current_value = value
        self._update_value_display()
    
    def get_value(self) -> Any:
        """Get the current setting value."""
        return self.setting_item.current_value
    
    def reset_to_default(self):
        """Reset setting to default value."""
        self.set_value(self.setting_item.default_value)
        self._trigger_callback('setting_changed', self.setting_item.key, self.setting_item.default_value)
    
    def _trigger_callback(self, event_type: str, *args):
        """Trigger registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(event_type, *args)
            except Exception as e:
                print(f"Error in setting widget callback: {e}")
    
    def add_callback(self, callback: Callable):
        """Add a callback for events."""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

class SettingsView(BoxLayout, EventDispatcher):
    """Main settings view with categorized options."""
    
    def __init__(self, config: Optional[SettingsConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or SettingsConfig()
        self.orientation = 'vertical'
        self.spacing = self.config.spacing
        self.padding = self.config.padding
        
        # Settings data
        self.settings_by_category = {}
        self.setting_widgets = {}
        self.has_unsaved_changes = False
        
        # Callbacks
        self.callbacks = []
        
        # Initialize default settings
        self._initialize_default_settings()
        
        # Build UI
        self._build_ui()
    
    def _initialize_default_settings(self):
        """Initialize default settings structure."""
        # General settings
        general_settings = [
            SettingItem("app_name", "Application Name", "Name displayed in the title bar", 
                       SettingType.STRING, "The Silent Steno"),
            SettingItem("auto_start_session", "Auto-start Sessions", "Automatically start recording when connected", 
                       SettingType.BOOLEAN, False),
            SettingItem("ui_theme", "UI Theme", "Visual theme for the interface", 
                       SettingType.CHOICE, "dark", ["light", "dark", "auto"]),
            SettingItem("language", "Language", "Interface language", 
                       SettingType.CHOICE, "en", ["en", "es", "fr", "de", "ja"]),
            SettingItem("screen_timeout", "Screen Timeout", "Minutes before screen dims (0 = never)", 
                       SettingType.SLIDER, 5, min_value=0, max_value=60),
        ]
        
        # Audio settings
        audio_settings = [
            SettingItem("audio_quality", "Audio Quality", "Recording quality setting", 
                       SettingType.CHOICE, "high", ["low", "medium", "high", "lossless"]),
            SettingItem("sample_rate", "Sample Rate", "Audio sample rate in Hz", 
                       SettingType.CHOICE, "44100", ["8000", "16000", "22050", "44100", "48000"]),
            SettingItem("audio_channels", "Audio Channels", "Number of audio channels", 
                       SettingType.CHOICE, "stereo", ["mono", "stereo"]),
            SettingItem("noise_reduction", "Noise Reduction", "Enable real-time noise reduction", 
                       SettingType.BOOLEAN, True),
            SettingItem("auto_gain_control", "Auto Gain Control", "Automatically adjust recording levels", 
                       SettingType.BOOLEAN, True),
            SettingItem("audio_latency_target", "Target Latency", "Target audio latency in milliseconds", 
                       SettingType.SLIDER, 40, min_value=10, max_value=200),
        ]
        
        # AI settings
        ai_settings = [
            SettingItem("whisper_model", "Whisper Model", "Transcription model to use", 
                       SettingType.CHOICE, "base", ["tiny", "base", "small", "medium"]),
            SettingItem("transcription_language", "Transcription Language", "Primary language for transcription", 
                       SettingType.CHOICE, "auto", ["auto", "en", "es", "fr", "de", "ja", "zh"]),
            SettingItem("speaker_diarization", "Speaker Diarization", "Identify different speakers", 
                       SettingType.BOOLEAN, True),
            SettingItem("real_time_transcription", "Real-time Transcription", "Transcribe during recording", 
                       SettingType.BOOLEAN, True),
            SettingItem("llm_analysis", "LLM Analysis", "Enable AI analysis of meetings", 
                       SettingType.BOOLEAN, True),
            SettingItem("confidence_threshold", "Confidence Threshold", "Minimum confidence for transcription", 
                       SettingType.SLIDER, 0.7, min_value=0.1, max_value=1.0),
        ]
        
        # Storage settings
        storage_settings = [
            SettingItem("storage_path", "Storage Location", "Directory for storing recordings", 
                       SettingType.PATH, os.path.expanduser("~/SilentSteno")),
            SettingItem("max_storage_gb", "Maximum Storage", "Maximum storage usage in GB", 
                       SettingType.SLIDER, 50, min_value=1, max_value=500),
            SettingItem("auto_cleanup", "Auto Cleanup", "Automatically delete old recordings", 
                       SettingType.BOOLEAN, True),
            SettingItem("cleanup_after_days", "Cleanup After", "Delete recordings older than X days", 
                       SettingType.SLIDER, 30, min_value=1, max_value=365),
            SettingItem("backup_enabled", "Enable Backup", "Automatically backup recordings", 
                       SettingType.BOOLEAN, False),
            SettingItem("backup_path", "Backup Location", "Directory for backup storage", 
                       SettingType.PATH, ""),
        ]
        
        # Network settings
        network_settings = [
            SettingItem("wifi_required", "WiFi Required", "Require WiFi connection for cloud features", 
                       SettingType.BOOLEAN, True),
            SettingItem("cloud_sync", "Cloud Sync", "Sync recordings to cloud storage", 
                       SettingType.BOOLEAN, False),
            SettingItem("cloud_provider", "Cloud Provider", "Cloud storage service", 
                       SettingType.CHOICE, "none", ["none", "dropbox", "google_drive", "onedrive"]),
            SettingItem("export_sharing", "Export Sharing", "Allow sharing exported files", 
                       SettingType.BOOLEAN, True),
            SettingItem("network_timeout", "Network Timeout", "Network operation timeout in seconds", 
                       SettingType.SLIDER, 30, min_value=5, max_value=120),
        ]
        
        # Organize by category
        self.settings_by_category = {
            SettingCategory.GENERAL: general_settings,
            SettingCategory.AUDIO: audio_settings,
            SettingCategory.AI: ai_settings,
            SettingCategory.STORAGE: storage_settings,
            SettingCategory.NETWORK: network_settings,
        }
    
    def _build_ui(self):
        """Build the settings UI."""
        # Header with title and actions
        header_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(60),
            spacing=dp(8)
        )
        
        self.title_label = Label(
            text="Settings",
            font_size=dp(20),
            bold=True,
            size_hint_x=0.4,
            halign='left',
            valign='middle'
        )
        self.title_label.bind(size=self.title_label.setter('text_size'))
        header_layout.add_widget(self.title_label)
        
        # Search box (if enabled)
        if self.config.show_search:
            self.search_input = TextInput(
                hint_text="Search settings...",
                size_hint_x=0.4,
                multiline=False,
                font_size=dp(12)
            )
            self.search_input.bind(text=self._on_search_changed)
            header_layout.add_widget(self.search_input)
        
        # Action buttons
        actions_layout = BoxLayout(
            orientation='horizontal',
            size_hint_x=0.2,
            spacing=dp(4)
        )
        
        if self.config.show_reset_buttons:
            self.reset_btn = Button(
                text="Reset",
                font_size=dp(11)
            )
            self.reset_btn.bind(on_press=self._on_reset_all)
            actions_layout.add_widget(self.reset_btn)
        
        self.save_btn = Button(
            text="Save",
            font_size=dp(11)
        )
        self.save_btn.bind(on_press=self._on_save_settings)
        actions_layout.add_widget(self.save_btn)
        
        header_layout.add_widget(actions_layout)
        self.add_widget(header_layout)
        
        # Main content in scroll view
        self.scroll_view = ScrollView(
            do_scroll_x=False,
            do_scroll_y=True
        )
        
        # Settings accordion
        self.accordion = Accordion(
            size_hint_y=None,
            orientation='vertical'
        )
        self.accordion.bind(minimum_height=self.accordion.setter('height'))
        
        # Build category sections
        self._build_category_sections()
        
        self.scroll_view.add_widget(self.accordion)
        self.add_widget(self.scroll_view)
        
        # Footer with import/export and about
        if self.config.show_import_export or self.config.show_about:
            footer_layout = self._build_footer()
            self.add_widget(footer_layout)
    
    def _build_category_sections(self):
        """Build accordion sections for each setting category."""
        for category, settings in self.settings_by_category.items():
            # Create accordion item
            item_title = category.value.replace('_', ' ').title()
            accordion_item = AccordionItem(
                title=item_title,
                min_space=dp(60)
            )
            
            # Content layout for settings
            content_layout = BoxLayout(
                orientation='vertical',
                size_hint_y=None,
                spacing=dp(4),
                padding=dp(8)
            )
            content_layout.bind(minimum_height=content_layout.setter('height'))
            
            # Add setting widgets
            for setting in settings:
                setting_widget = SettingWidget(setting, self.config)
                setting_widget.add_callback(self._on_setting_event)
                content_layout.add_widget(setting_widget)
                
                # Store reference for easy access
                self.setting_widgets[setting.key] = setting_widget
            
            # Scroll view for category content
            category_scroll = ScrollView(
                do_scroll_x=False,
                do_scroll_y=True
            )
            category_scroll.add_widget(content_layout)
            
            accordion_item.add_widget(category_scroll)
            self.accordion.add_widget(accordion_item)
    
    def _build_footer(self) -> BoxLayout:
        """Build footer with import/export and about."""
        footer_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(50),
            spacing=dp(8)
        )
        
        if self.config.show_import_export:
            self.import_btn = Button(
                text="Import Settings",
                size_hint_x=0.3,
                font_size=dp(11)
            )
            self.import_btn.bind(on_press=self._on_import_settings)
            footer_layout.add_widget(self.import_btn)
            
            self.export_btn = Button(
                text="Export Settings",
                size_hint_x=0.3,
                font_size=dp(11)
            )
            self.export_btn.bind(on_press=self._on_export_settings)
            footer_layout.add_widget(self.export_btn)
        
        # Spacer
        footer_layout.add_widget(Label(size_hint_x=0.2))
        
        if self.config.show_about:
            self.about_btn = Button(
                text="About",
                size_hint_x=0.2,
                font_size=dp(11)
            )
            self.about_btn.bind(on_press=self._on_about)
            footer_layout.add_widget(self.about_btn)
        
        return footer_layout
    
    def load_settings(self, settings_dict: Optional[Dict[str, Any]] = None):
        """Load settings from dictionary or file."""
        if settings_dict is None:
            # Load from file
            try:
                if os.path.exists(self.config.settings_file):
                    with open(self.config.settings_file, 'r') as f:
                        settings_dict = json.load(f)
                else:
                    settings_dict = {}
            except Exception as e:
                print(f"Error loading settings: {e}")
                settings_dict = {}
        
        # Apply settings to widgets
        for category_settings in self.settings_by_category.values():
            for setting in category_settings:
                if setting.key in settings_dict:
                    setting.current_value = settings_dict[setting.key]
                    if setting.key in self.setting_widgets:
                        self.setting_widgets[setting.key].set_value(setting.current_value)
        
        self.has_unsaved_changes = False
        self._trigger_callback('settings_loaded')
    
    def save_settings(self) -> bool:
        """Save current settings to file."""
        try:
            # Collect all settings
            settings_dict = {}
            for category_settings in self.settings_by_category.values():
                for setting in category_settings:
                    settings_dict[setting.key] = setting.current_value
            
            # Create directory if needed
            os.makedirs(os.path.dirname(self.config.settings_file), exist_ok=True)
            
            # Backup existing settings
            if os.path.exists(self.config.settings_file):
                backup_path = self.config.backup_file
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                import shutil
                shutil.copy2(self.config.settings_file, backup_path)
            
            # Save settings
            with open(self.config.settings_file, 'w') as f:
                json.dump(settings_dict, f, indent=2, default=str)
            
            self.has_unsaved_changes = False
            self._trigger_callback('settings_saved', settings_dict)
            return True
            
        except Exception as e:
            print(f"Error saving settings: {e}")
            self._show_error("Failed to save settings", str(e))
            return False
    
    def reset_settings(self):
        """Reset all settings to default values."""
        for category_settings in self.settings_by_category.values():
            for setting in category_settings:
                setting.current_value = setting.default_value
                if setting.key in self.setting_widgets:
                    self.setting_widgets[setting.key].set_value(setting.current_value)
        
        self.has_unsaved_changes = True
        self._trigger_callback('settings_reset')
    
    def import_settings(self, file_path: str) -> bool:
        """Import settings from file."""
        try:
            with open(file_path, 'r') as f:
                settings_dict = json.load(f)
            
            self.load_settings(settings_dict)
            self.has_unsaved_changes = True
            self._trigger_callback('settings_imported', file_path)
            return True
            
        except Exception as e:
            print(f"Error importing settings: {e}")
            self._show_error("Failed to import settings", str(e))
            return False
    
    def export_settings(self, file_path: str) -> bool:
        """Export current settings to file."""
        try:
            # Collect all settings
            settings_dict = {}
            for category_settings in self.settings_by_category.values():
                for setting in category_settings:
                    settings_dict[setting.key] = setting.current_value
            
            # Add metadata
            export_data = {
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'app_version': '1.0.0',
                    'settings_version': '1.0'
                },
                'settings': settings_dict
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self._trigger_callback('settings_exported', file_path)
            return True
            
        except Exception as e:
            print(f"Error exporting settings: {e}")
            self._show_error("Failed to export settings", str(e))
            return False
    
    def validate_settings(self) -> Dict[str, List[str]]:
        """Validate all settings and return any errors."""
        errors = {}
        
        for category, category_settings in self.settings_by_category.items():
            category_errors = []
            for setting in category_settings:
                is_valid, error_msg = setting.validate(setting.current_value)
                if not is_valid:
                    category_errors.append(f"{setting.title}: {error_msg}")
            
            if category_errors:
                errors[category.value] = category_errors
        
        return errors
    
    def add_setting_item(self, category: SettingCategory, setting: SettingItem):
        """Add a new setting item to a category."""
        if category not in self.settings_by_category:
            self.settings_by_category[category] = []
        
        self.settings_by_category[category].append(setting)
        # Would need to rebuild UI to show new setting
        self._trigger_callback('setting_added', category.value, setting.key)
    
    def remove_setting_item(self, category: SettingCategory, setting_key: str):
        """Remove a setting item from a category."""
        if category in self.settings_by_category:
            self.settings_by_category[category] = [
                s for s in self.settings_by_category[category] 
                if s.key != setting_key
            ]
            if setting_key in self.setting_widgets:
                del self.setting_widgets[setting_key]
        
        self._trigger_callback('setting_removed', category.value, setting_key)
    
    def update_setting(self, setting_key: str, value: Any):
        """Update a specific setting value."""
        for category_settings in self.settings_by_category.values():
            for setting in category_settings:
                if setting.key == setting_key:
                    setting.current_value = value
                    if setting_key in self.setting_widgets:
                        self.setting_widgets[setting_key].set_value(value)
                    self.has_unsaved_changes = True
                    self._trigger_callback('setting_changed', setting_key, value)
                    return
    
    def get_setting_value(self, setting_key: str) -> Any:
        """Get the value of a specific setting."""
        for category_settings in self.settings_by_category.values():
            for setting in category_settings:
                if setting.key == setting_key:
                    return setting.current_value
        return None
    
    def set_setting_value(self, setting_key: str, value: Any):
        """Set the value of a specific setting."""
        self.update_setting(setting_key, value)
    
    # Event handlers
    def _on_setting_event(self, event_type: str, *args):
        """Handle events from setting widgets."""
        if event_type == 'setting_changed':
            self.has_unsaved_changes = True
        self._trigger_callback(event_type, *args)
    
    def _on_search_changed(self, text_input, text):
        """Handle search text change."""
        # Filter settings based on search text
        # This would require rebuilding the UI with filtered settings
        self._trigger_callback('search_changed', text)
    
    def _on_save_settings(self, button):
        """Handle save settings button."""
        if self.save_settings():
            self._show_success("Settings saved successfully")
    
    def _on_reset_all(self, button):
        """Handle reset all button."""
        # Show confirmation dialog
        self._show_confirmation(
            "Reset All Settings",
            "Are you sure you want to reset all settings to their default values?",
            self.reset_settings
        )
    
    def _on_import_settings(self, button):
        """Handle import settings button."""
        self._trigger_callback('import_settings_requested')
    
    def _on_export_settings(self, button):
        """Handle export settings button."""
        self._trigger_callback('export_settings_requested')
    
    def _on_about(self, button):
        """Handle about button."""
        about_text = """The Silent Steno
Bluetooth AI Meeting Recorder

Version: 1.0.0
Platform: Raspberry Pi 5

A device that captures meeting audio via Bluetooth,
forwards to headphones, and provides AI-powered
transcription and analysis.

© 2024 The Silent Steno Project"""
        
        self._show_info("About The Silent Steno", about_text)
    
    def _show_error(self, title: str, message: str):
        """Show error popup."""
        popup = Popup(
            title=title,
            content=Label(text=message, text_size=(dp(300), None), halign='center'),
            size_hint=(None, None),
            size=(dp(350), dp(200))
        )
        popup.open()
    
    def _show_success(self, message: str):
        """Show success popup."""
        popup = Popup(
            title="Success",
            content=Label(text=message, text_size=(dp(300), None), halign='center'),
            size_hint=(None, None),
            size=(dp(350), dp(150))
        )
        popup.open()
        Clock.schedule_once(lambda dt: popup.dismiss(), 2.0)
    
    def _show_info(self, title: str, message: str):
        """Show info popup."""
        popup = Popup(
            title=title,
            content=Label(text=message, text_size=(dp(300), None), halign='center'),
            size_hint=(None, None),
            size=(dp(400), dp(300))
        )
        popup.open()
    
    def _show_confirmation(self, title: str, message: str, callback: Callable):
        """Show confirmation dialog."""
        content = BoxLayout(orientation='vertical', spacing=dp(8))
        content.add_widget(Label(text=message, text_size=(dp(300), None), halign='center'))
        
        buttons = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40), spacing=dp(8))
        
        cancel_btn = Button(text="Cancel")
        cancel_btn.bind(on_press=lambda x: popup.dismiss())
        buttons.add_widget(cancel_btn)
        
        confirm_btn = Button(text="Confirm")
        confirm_btn.bind(on_press=lambda x: (callback(), popup.dismiss()))
        buttons.add_widget(confirm_btn)
        
        content.add_widget(buttons)
        
        popup = Popup(
            title=title,
            content=content,
            size_hint=(None, None),
            size=(dp(350), dp(200))
        )
        popup.open()
    
    def _trigger_callback(self, event_type: str, *args):
        """Trigger registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(event_type, *args)
            except Exception as e:
                print(f"Error in settings view callback: {e}")
    
    def add_callback(self, callback: Callable):
        """Add a callback for events."""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

# Factory functions
def create_settings_view(config: Optional[SettingsConfig] = None) -> SettingsView:
    """Create a settings view with optional configuration."""
    return SettingsView(config)

def create_default_config() -> SettingsConfig:
    """Create default settings configuration."""
    return SettingsConfig()

# Demo function
def demo_settings_view():
    """Demo function to test the settings view."""
    from kivy.app import App
    
    class SettingsDemo(App):
        def build(self):
            settings_view = create_settings_view()
            settings_view.add_callback(self._on_settings_event)
            settings_view.load_settings()
            return settings_view
        
        def _on_settings_event(self, event_type: str, *args):
            print(f"Settings event: {event_type}, args: {args}")
    
    return SettingsDemo()

@contextmanager
def settings_context(config: Optional[SettingsConfig] = None):
    """Context manager for settings operations."""
    settings_view = create_settings_view(config)
    try:
        yield settings_view
    finally:
        # Cleanup if needed
        settings_view.callbacks.clear()

if __name__ == "__main__":
    demo = demo_settings_view()
    demo.run()