"""
Storage Monitor Widget - Storage usage monitoring and management widget.

This module provides a real-time storage monitoring widget for The Silent Steno project,
displaying storage usage visualization, cleanup options, and storage management features.
"""

import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List, Tuple
from enum import Enum
from contextlib import contextmanager

try:
    from kivy.app import App
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.gridlayout import GridLayout
    from kivy.uix.label import Label
    from kivy.uix.button import Button
    from kivy.uix.progressbar import ProgressBar
    from kivy.uix.popup import Popup
    from kivy.uix.scrollview import ScrollView
    from kivy.graphics import Color, Rectangle, Line
    from kivy.properties import StringProperty, NumericProperty, BooleanProperty, ObjectProperty
    from kivy.event import EventDispatcher
    from kivy.clock import Clock
    from kivy.metrics import dp
    from kivy.utils import get_color_from_hex
except ImportError:
    # Fallback for systems without Kivy
    class BoxLayout: pass
    class GridLayout: pass
    class Label: pass
    class Button: pass
    class ProgressBar: pass
    class Popup: pass
    class ScrollView: pass
    class EventDispatcher: pass
    StringProperty = NumericProperty = BooleanProperty = ObjectProperty = lambda x: x
    Clock = None
    dp = lambda x: x
    get_color_from_hex = lambda x: x

class StorageAlert(Enum):
    NONE = "none"
    LOW = "low"
    CRITICAL = "critical"
    FULL = "full"

class StorageInfo:
    """Container for storage information."""
    
    def __init__(self, path: str = ""):
        self.path = path
        self.total_bytes = 0
        self.used_bytes = 0
        self.free_bytes = 0
        self.sessions_count = 0
        self.sessions_size_bytes = 0
        self.oldest_session_date = None
        self.last_updated = datetime.now()
        
        # Calculated properties
        self.usage_percentage = 0.0
        self.sessions_percentage = 0.0
        self.alert_level = StorageAlert.NONE
        
        if path and os.path.exists(path):
            self.update_from_path()
    
    def update_from_path(self):
        """Update storage info from filesystem."""
        try:
            if self.path and os.path.exists(self.path):
                # Get disk usage
                self.total_bytes, self.used_bytes, self.free_bytes = shutil.disk_usage(self.path)
                
                # Calculate sessions usage
                self._calculate_sessions_usage()
                
                # Update calculated properties
                self._update_calculated_properties()
                
                self.last_updated = datetime.now()
        except Exception as e:
            print(f"Error updating storage info: {e}")
    
    def _calculate_sessions_usage(self):
        """Calculate storage used by sessions."""
        self.sessions_count = 0
        self.sessions_size_bytes = 0
        self.oldest_session_date = None
        
        try:
            if not self.path or not os.path.exists(self.path):
                return
            
            # Look for session files (would be based on actual file structure)
            for root, dirs, files in os.walk(self.path):
                for file in files:
                    if file.endswith(('.flac', '.wav', '.mp3', '.json', '.txt')):
                        file_path = os.path.join(root, file)
                        try:
                            file_size = os.path.getsize(file_path)
                            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                            
                            self.sessions_size_bytes += file_size
                            
                            if file.endswith(('.flac', '.wav', '.mp3')):
                                self.sessions_count += 1
                                if self.oldest_session_date is None or file_mtime < self.oldest_session_date:
                                    self.oldest_session_date = file_mtime
                        except (OSError, ValueError):
                            continue
        except Exception as e:
            print(f"Error calculating sessions usage: {e}")
    
    def _update_calculated_properties(self):
        """Update calculated properties."""
        # Usage percentage
        if self.total_bytes > 0:
            self.usage_percentage = (self.used_bytes / self.total_bytes) * 100
            self.sessions_percentage = (self.sessions_size_bytes / self.total_bytes) * 100
        else:
            self.usage_percentage = 0.0
            self.sessions_percentage = 0.0
        
        # Alert level
        if self.usage_percentage >= 95:
            self.alert_level = StorageAlert.FULL
        elif self.usage_percentage >= 85:
            self.alert_level = StorageAlert.CRITICAL
        elif self.usage_percentage >= 75:
            self.alert_level = StorageAlert.LOW
        else:
            self.alert_level = StorageAlert.NONE
    
    @property
    def formatted_total(self) -> str:
        """Get formatted total storage string."""
        return self._format_bytes(self.total_bytes)
    
    @property
    def formatted_used(self) -> str:
        """Get formatted used storage string."""
        return self._format_bytes(self.used_bytes)
    
    @property
    def formatted_free(self) -> str:
        """Get formatted free storage string."""
        return self._format_bytes(self.free_bytes)
    
    @property
    def formatted_sessions_size(self) -> str:
        """Get formatted sessions storage string."""
        return self._format_bytes(self.sessions_size_bytes)
    
    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes to human readable string."""
        if bytes_value < 1024:
            return f"{bytes_value} B"
        elif bytes_value < 1024 * 1024:
            return f"{bytes_value / 1024:.1f} KB"
        elif bytes_value < 1024 * 1024 * 1024:
            return f"{bytes_value / (1024 * 1024):.1f} MB"
        else:
            return f"{bytes_value / (1024 * 1024 * 1024):.1f} GB"
    
    def get_cleanup_potential(self, days_threshold: int = 30) -> Tuple[int, int]:
        """Get potential cleanup statistics."""
        cleanup_count = 0
        cleanup_bytes = 0
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        
        try:
            if not self.path or not os.path.exists(self.path):
                return cleanup_count, cleanup_bytes
            
            for root, dirs, files in os.walk(self.path):
                for file in files:
                    if file.endswith(('.flac', '.wav', '.mp3')):
                        file_path = os.path.join(root, file)
                        try:
                            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                            if file_mtime < cutoff_date:
                                cleanup_count += 1
                                cleanup_bytes += os.path.getsize(file_path)
                        except (OSError, ValueError):
                            continue
        except Exception as e:
            print(f"Error calculating cleanup potential: {e}")
        
        return cleanup_count, cleanup_bytes

class StorageConfig:
    """Configuration for storage monitor display and behavior."""
    
    def __init__(self):
        # Widget sizing
        self.compact_height = dp(80)
        self.detailed_height = dp(200)
        self.minimal_height = dp(50)
        
        # Update intervals
        self.update_interval = 30.0  # seconds
        self.auto_update = True
        
        # Display options
        self.show_percentage = True
        self.show_absolute_values = True
        self.show_sessions_breakdown = True
        self.show_cleanup_options = True
        self.show_alerts = True
        
        # Thresholds
        self.low_storage_threshold = 75  # percentage
        self.critical_storage_threshold = 85  # percentage
        self.full_storage_threshold = 95  # percentage
        
        # Cleanup options
        self.default_cleanup_days = 30
        self.enable_auto_cleanup = True
        
        # Colors
        self.normal_color = get_color_from_hex('#27AE60') if callable(get_color_from_hex) else '#27AE60'
        self.warning_color = get_color_from_hex('#F39C12') if callable(get_color_from_hex) else '#F39C12'
        self.critical_color = get_color_from_hex('#E74C3C') if callable(get_color_from_hex) else '#E74C3C'
        self.background_color = get_color_from_hex('#ECF0F1') if callable(get_color_from_hex) else '#ECF0F1'

class StorageBarWidget(BoxLayout):
    """Visual storage usage bar widget."""
    
    def __init__(self, config: StorageConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.orientation = 'vertical'
        self.size_hint_y = None
        self.height = dp(60)
        self.spacing = dp(4)
        
        self.storage_info = None
        self._build_ui()
    
    def _build_ui(self):
        """Build the storage bar UI."""
        # Usage labels
        labels_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(20)
        )
        
        self.usage_label = Label(
            text="Storage: 0%",
            font_size=dp(12),
            size_hint_x=0.5,
            halign='left'
        )
        self.usage_label.bind(size=self.usage_label.setter('text_size'))
        labels_layout.add_widget(self.usage_label)
        
        self.size_label = Label(
            text="0 GB / 0 GB",
            font_size=dp(12),
            size_hint_x=0.5,
            halign='right'
        )
        self.size_label.bind(size=self.size_label.setter('text_size'))
        labels_layout.add_widget(self.size_label)
        
        self.add_widget(labels_layout)
        
        # Progress bar
        self.progress_bar = ProgressBar(
            max=100,
            value=0,
            size_hint_y=None,
            height=dp(20)
        )
        self.add_widget(self.progress_bar)
        
        # Sessions breakdown (optional)
        if self.config.show_sessions_breakdown:
            sessions_layout = BoxLayout(
                orientation='horizontal',
                size_hint_y=None,
                height=dp(16)
            )
            
            self.sessions_label = Label(
                text="Sessions: 0 files",
                font_size=dp(10),
                size_hint_x=0.6,
                halign='left',
                color=(0.6, 0.6, 0.6, 1)
            )
            self.sessions_label.bind(size=self.sessions_label.setter('text_size'))
            sessions_layout.add_widget(self.sessions_label)
            
            self.sessions_size_label = Label(
                text="0 GB",
                font_size=dp(10),
                size_hint_x=0.4,
                halign='right',
                color=(0.6, 0.6, 0.6, 1)
            )
            self.sessions_size_label.bind(size=self.sessions_size_label.setter('text_size'))
            sessions_layout.add_widget(self.sessions_size_label)
            
            self.add_widget(sessions_layout)
    
    def update_storage_info(self, storage_info: StorageInfo):
        """Update the display with new storage info."""
        self.storage_info = storage_info
        
        # Update labels
        if self.config.show_percentage:
            self.usage_label.text = f"Storage: {storage_info.usage_percentage:.1f}%"
        else:
            self.usage_label.text = "Storage Usage"
        
        if self.config.show_absolute_values:
            self.size_label.text = f"{storage_info.formatted_used} / {storage_info.formatted_total}"
        else:
            self.size_label.text = f"{storage_info.formatted_free} free"
        
        # Update progress bar
        self.progress_bar.value = storage_info.usage_percentage
        
        # Update progress bar color based on usage
        if storage_info.usage_percentage >= self.config.critical_storage_threshold:
            # Would set bar color to critical (red)
            pass
        elif storage_info.usage_percentage >= self.config.low_storage_threshold:
            # Would set bar color to warning (orange)
            pass
        else:
            # Would set bar color to normal (green)
            pass
        
        # Update sessions breakdown
        if self.config.show_sessions_breakdown and hasattr(self, 'sessions_label'):
            self.sessions_label.text = f"Sessions: {storage_info.sessions_count} files"
            self.sessions_size_label.text = storage_info.formatted_sessions_size

class StorageMonitorWidget(BoxLayout, EventDispatcher):
    """Main storage monitoring widget with management options."""
    
    def __init__(self, storage_path: str = "", config: Optional[StorageConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.storage_path = storage_path or os.path.expanduser("~/SilentSteno")
        self.config = config or StorageConfig()
        self.orientation = 'vertical'
        self.spacing = dp(8)
        
        # State
        self.storage_info = StorageInfo(self.storage_path)
        self.update_event = None
        self.callbacks = []
        
        # Build UI based on configuration
        self._build_ui()
        
        # Start auto-update if enabled
        if self.config.auto_update:
            self._start_auto_update()
    
    def _build_ui(self):
        """Build the storage monitor UI."""
        # Storage bar
        self.storage_bar = StorageBarWidget(self.config)
        self.add_widget(self.storage_bar)
        
        # Alert section
        if self.config.show_alerts:
            self.alert_layout = BoxLayout(
                orientation='horizontal',
                size_hint_y=None,
                height=0,  # Initially hidden
                spacing=dp(8)
            )
            
            self.alert_icon = Label(
                text="⚠️",
                font_size=dp(16),
                size_hint_x=None,
                width=dp(30)
            )
            self.alert_layout.add_widget(self.alert_icon)
            
            self.alert_label = Label(
                text="Storage alert",
                font_size=dp(12),
                size_hint_x=1,
                halign='left'
            )
            self.alert_label.bind(size=self.alert_label.setter('text_size'))
            self.alert_layout.add_widget(self.alert_label)
            
            self.add_widget(self.alert_layout)
        
        # Management actions
        if self.config.show_cleanup_options:
            actions_layout = BoxLayout(
                orientation='horizontal',
                size_hint_y=None,
                height=dp(40),
                spacing=dp(8)
            )
            
            self.cleanup_btn = Button(
                text="Clean Up",
                size_hint_x=0.3,
                font_size=dp(11)
            )
            self.cleanup_btn.bind(on_press=self._on_cleanup_pressed)
            actions_layout.add_widget(self.cleanup_btn)
            
            self.details_btn = Button(
                text="Details",
                size_hint_x=0.3,
                font_size=dp(11)
            )
            self.details_btn.bind(on_press=self._on_details_pressed)
            actions_layout.add_widget(self.details_btn)
            
            self.refresh_btn = Button(
                text="Refresh",
                size_hint_x=0.2,
                font_size=dp(11)
            )
            self.refresh_btn.bind(on_press=self._on_refresh_pressed)
            actions_layout.add_widget(self.refresh_btn)
            
            self.settings_btn = Button(
                text="⚙️",
                size_hint_x=0.2,
                font_size=dp(14)
            )
            self.settings_btn.bind(on_press=self._on_settings_pressed)
            actions_layout.add_widget(self.settings_btn)
            
            self.add_widget(actions_layout)
    
    def update_storage_info(self):
        """Update storage information from filesystem."""
        self.storage_info.update_from_path()
        self.storage_bar.update_storage_info(self.storage_info)
        self._update_alerts()
        self._trigger_callback('storage_updated', self.storage_info)
    
    def _update_alerts(self):
        """Update alert display based on storage status."""
        if not self.config.show_alerts or not hasattr(self, 'alert_layout'):
            return
        
        if self.storage_info.alert_level == StorageAlert.NONE:
            self.alert_layout.height = 0
        else:
            self.alert_layout.height = dp(30)
            
            if self.storage_info.alert_level == StorageAlert.FULL:
                self.alert_icon.text = "🔴"
                self.alert_label.text = "Storage is full! Immediate cleanup required."
                self.alert_label.color = self.config.critical_color
            elif self.storage_info.alert_level == StorageAlert.CRITICAL:
                self.alert_icon.text = "🟡"
                self.alert_label.text = f"Critical: {self.storage_info.usage_percentage:.1f}% storage used"
                self.alert_label.color = self.config.critical_color
            elif self.storage_info.alert_level == StorageAlert.LOW:
                self.alert_icon.text = "⚠️"
                self.alert_label.text = f"Warning: {self.storage_info.usage_percentage:.1f}% storage used"
                self.alert_label.color = self.config.warning_color
    
    def _start_auto_update(self):
        """Start automatic storage updates."""
        if Clock and not self.update_event:
            self.update_event = Clock.schedule_interval(
                lambda dt: self.update_storage_info(),
                self.config.update_interval
            )
    
    def _stop_auto_update(self):
        """Stop automatic storage updates."""
        if Clock and self.update_event:
            Clock.unschedule(self.update_event)
            self.update_event = None
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """Get current storage usage information."""
        return {
            'path': self.storage_info.path,
            'total_bytes': self.storage_info.total_bytes,
            'used_bytes': self.storage_info.used_bytes,
            'free_bytes': self.storage_info.free_bytes,
            'usage_percentage': self.storage_info.usage_percentage,
            'sessions_count': self.storage_info.sessions_count,
            'sessions_size_bytes': self.storage_info.sessions_size_bytes,
            'alert_level': self.storage_info.alert_level.value,
            'last_updated': self.storage_info.last_updated.isoformat()
        }
    
    def get_session_sizes(self) -> List[Dict[str, Any]]:
        """Get sizes of individual sessions."""
        sessions = []
        
        try:
            if not self.storage_path or not os.path.exists(self.storage_path):
                return sessions
            
            for root, dirs, files in os.walk(self.storage_path):
                for file in files:
                    if file.endswith(('.flac', '.wav', '.mp3')):
                        file_path = os.path.join(root, file)
                        try:
                            file_size = os.path.getsize(file_path)
                            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                            
                            sessions.append({
                                'filename': file,
                                'path': file_path,
                                'size_bytes': file_size,
                                'size_formatted': self.storage_info._format_bytes(file_size),
                                'modified_date': file_mtime.isoformat(),
                                'age_days': (datetime.now() - file_mtime).days
                            })
                        except (OSError, ValueError):
                            continue
        except Exception as e:
            print(f"Error getting session sizes: {e}")
        
        return sorted(sessions, key=lambda x: x['modified_date'], reverse=True)
    
    def clean_old_sessions(self, days_threshold: int = None) -> Tuple[int, int]:
        """Clean up old sessions and return count and bytes cleaned."""
        if days_threshold is None:
            days_threshold = self.config.default_cleanup_days
        
        cleaned_count = 0
        cleaned_bytes = 0
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        
        try:
            if not self.storage_path or not os.path.exists(self.storage_path):
                return cleaned_count, cleaned_bytes
            
            for root, dirs, files in os.walk(self.storage_path):
                for file in files:
                    if file.endswith(('.flac', '.wav', '.mp3')):
                        file_path = os.path.join(root, file)
                        try:
                            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                            if file_mtime < cutoff_date:
                                file_size = os.path.getsize(file_path)
                                os.remove(file_path)
                                cleaned_count += 1
                                cleaned_bytes += file_size
                        except (OSError, ValueError):
                            continue
        except Exception as e:
            print(f"Error cleaning old sessions: {e}")
        
        # Update storage info after cleanup
        self.update_storage_info()
        self._trigger_callback('cleanup_completed', cleaned_count, cleaned_bytes)
        
        return cleaned_count, cleaned_bytes
    
    def estimate_remaining_time(self, average_session_size: int = None) -> str:
        """Estimate remaining recording time based on available storage."""
        if average_session_size is None:
            # Calculate average from existing sessions
            if self.storage_info.sessions_count > 0:
                average_session_size = self.storage_info.sessions_size_bytes // self.storage_info.sessions_count
            else:
                average_session_size = 50 * 1024 * 1024  # 50MB default
        
        if average_session_size == 0:
            return "Unknown"
        
        remaining_sessions = self.storage_info.free_bytes // average_session_size
        
        if remaining_sessions < 1:
            return "No space remaining"
        elif remaining_sessions < 10:
            return f"~{remaining_sessions} sessions"
        elif remaining_sessions < 100:
            return f"~{remaining_sessions} sessions"
        else:
            hours = remaining_sessions * 0.5  # Assume 30-minute sessions
            if hours < 24:
                return f"~{hours:.1f} hours"
            else:
                days = hours / 24
                return f"~{days:.1f} days"
    
    def set_storage_limit(self, limit_bytes: int):
        """Set storage usage limit."""
        # This would be used to configure auto-cleanup thresholds
        self._trigger_callback('storage_limit_set', limit_bytes)
    
    def enable_auto_cleanup(self, enabled: bool, days_threshold: int = None):
        """Enable or disable automatic cleanup."""
        self.config.enable_auto_cleanup = enabled
        if days_threshold:
            self.config.default_cleanup_days = days_threshold
        self._trigger_callback('auto_cleanup_configured', enabled, days_threshold)
    
    def show_storage_details(self):
        """Show detailed storage information dialog."""
        self._trigger_callback('storage_details_requested')
    
    def configure_alerts(self, low_threshold: int = None, critical_threshold: int = None):
        """Configure storage alert thresholds."""
        if low_threshold:
            self.config.low_storage_threshold = low_threshold
        if critical_threshold:
            self.config.critical_storage_threshold = critical_threshold
        
        self._update_alerts()
        self._trigger_callback('alerts_configured', low_threshold, critical_threshold)
    
    # Event handlers
    def _on_cleanup_pressed(self, button):
        """Handle cleanup button press."""
        cleanup_count, cleanup_bytes = self.storage_info.get_cleanup_potential(self.config.default_cleanup_days)
        
        if cleanup_count == 0:
            self._show_info("No Cleanup Needed", "No old sessions found for cleanup.")
        else:
            message = f"Found {cleanup_count} old sessions ({self.storage_info._format_bytes(cleanup_bytes)}) " \
                     f"older than {self.config.default_cleanup_days} days.\n\nProceed with cleanup?"
            self._show_confirmation("Cleanup Confirmation", message, self._perform_cleanup)
    
    def _on_details_pressed(self, button):
        """Handle details button press."""
        self.show_storage_details()
    
    def _on_refresh_pressed(self, button):
        """Handle refresh button press."""
        self.update_storage_info()
    
    def _on_settings_pressed(self, button):
        """Handle settings button press."""
        self._trigger_callback('storage_settings_requested')
    
    def _perform_cleanup(self):
        """Perform the actual cleanup operation."""
        cleaned_count, cleaned_bytes = self.clean_old_sessions()
        message = f"Cleanup completed!\n\nRemoved {cleaned_count} sessions " \
                 f"({self.storage_info._format_bytes(cleaned_bytes)} freed)"
        self._show_info("Cleanup Complete", message)
    
    def _show_info(self, title: str, message: str):
        """Show info popup."""
        popup = Popup(
            title=title,
            content=Label(text=message, text_size=(dp(300), None), halign='center'),
            size_hint=(None, None),
            size=(dp(350), dp(200))
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
        
        confirm_btn = Button(text="Proceed")
        confirm_btn.bind(on_press=lambda x: (callback(), popup.dismiss()))
        buttons.add_widget(confirm_btn)
        
        content.add_widget(buttons)
        
        popup = Popup(
            title=title,
            content=content,
            size_hint=(None, None),
            size=(dp(350), dp(250))
        )
        popup.open()
    
    def _trigger_callback(self, event_type: str, *args):
        """Trigger registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(event_type, *args)
            except Exception as e:
                print(f"Error in storage monitor callback: {e}")
    
    def add_callback(self, callback: Callable):
        """Add a callback for events."""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def __del__(self):
        """Cleanup when widget is destroyed."""
        self._stop_auto_update()

# Factory functions
def create_storage_monitor(storage_path: str = "", 
                          config: Optional[StorageConfig] = None) -> StorageMonitorWidget:
    """Create a storage monitor widget with optional configuration."""
    return StorageMonitorWidget(storage_path, config)

def create_minimal_monitor(storage_path: str = "") -> StorageMonitorWidget:
    """Create a minimal storage monitor widget."""
    config = StorageConfig()
    config.show_sessions_breakdown = False
    config.show_cleanup_options = False
    config.show_alerts = False
    
    monitor = StorageMonitorWidget(storage_path, config)
    monitor.size_hint_y = None
    monitor.height = config.minimal_height
    return monitor

def create_detailed_monitor(storage_path: str = "") -> StorageMonitorWidget:
    """Create a detailed storage monitor widget."""
    config = StorageConfig()
    config.show_sessions_breakdown = True
    config.show_cleanup_options = True
    config.show_alerts = True
    
    monitor = StorageMonitorWidget(storage_path, config)
    monitor.size_hint_y = None
    monitor.height = config.detailed_height
    return monitor

# Demo function
def demo_storage_monitor():
    """Demo function to test the storage monitor widget."""
    from kivy.app import App
    
    class StorageMonitorDemo(App):
        def build(self):
            layout = BoxLayout(orientation='vertical', spacing=dp(16), padding=dp(16))
            
            # Minimal monitor
            layout.add_widget(Label(text="Minimal Monitor:", size_hint_y=None, height=dp(30)))
            minimal = create_minimal_monitor()
            minimal.add_callback(self._on_storage_event)
            layout.add_widget(minimal)
            
            # Detailed monitor
            layout.add_widget(Label(text="Detailed Monitor:", size_hint_y=None, height=dp(30)))
            detailed = create_detailed_monitor()
            detailed.add_callback(self._on_storage_event)
            layout.add_widget(detailed)
            
            # Update storage info
            minimal.update_storage_info()
            detailed.update_storage_info()
            
            return layout
        
        def _on_storage_event(self, event_type: str, *args):
            print(f"Storage monitor event: {event_type}, args: {args}")
    
    return StorageMonitorDemo()

@contextmanager
def storage_monitor_context(storage_path: str = "", 
                           config: Optional[StorageConfig] = None):
    """Context manager for storage monitor operations."""
    monitor = create_storage_monitor(storage_path, config)
    try:
        yield monitor
    finally:
        # Cleanup if needed
        monitor._stop_auto_update()
        monitor.callbacks.clear()

if __name__ == "__main__":
    demo = demo_storage_monitor()
    demo.run()