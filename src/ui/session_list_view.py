"""
Session List View - Main session browsing interface with search, filter, and sort capabilities.

This module provides a comprehensive session list interface for The Silent Steno project,
allowing users to browse, search, filter, and manage their recorded meeting sessions.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Union
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
    from kivy.uix.spinner import Spinner
    from kivy.uix.checkbox import CheckBox
    from kivy.uix.popup import Popup
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
    class Spinner: pass
    class CheckBox: pass
    class Popup: pass
    class EventDispatcher: pass
    StringProperty = NumericProperty = BooleanProperty = ObjectProperty = lambda x: x
    Clock = None
    dp = lambda x: x
    get_color_from_hex = lambda x: x

# Sort options
class SessionSort(Enum):
    DATE = "date"
    DURATION = "duration"
    NAME = "name"
    SIZE = "size"

# Filter options
class SessionFilter(Enum):
    TODAY = "today"
    THIS_WEEK = "this_week"
    THIS_MONTH = "this_month"
    ALL = "all"
    ARCHIVED = "archived"

class SessionListItem:
    """Represents a single session item in the list."""
    
    def __init__(self, session_id: str, title: str, date: datetime, 
                 duration: int, size: int, has_transcript: bool = True,
                 has_analysis: bool = True, is_archived: bool = False):
        self.session_id = session_id
        self.title = title
        self.date = date
        self.duration = duration  # in seconds
        self.size = size  # in bytes
        self.has_transcript = has_transcript
        self.has_analysis = has_analysis
        self.is_archived = is_archived
        self.is_selected = False
    
    @property
    def formatted_date(self) -> str:
        """Get formatted date string."""
        now = datetime.now()
        if self.date.date() == now.date():
            return f"Today {self.date.strftime('%H:%M')}"
        elif self.date.date() == (now - timedelta(days=1)).date():
            return f"Yesterday {self.date.strftime('%H:%M')}"
        elif self.date.year == now.year:
            return self.date.strftime('%m/%d %H:%M')
        else:
            return self.date.strftime('%m/%d/%y %H:%M')
    
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
        if self.size < 1024:
            return f"{self.size} B"
        elif self.size < 1024 * 1024:
            return f"{self.size / 1024:.1f} KB"
        elif self.size < 1024 * 1024 * 1024:
            return f"{self.size / (1024 * 1024):.1f} MB"
        else:
            return f"{self.size / (1024 * 1024 * 1024):.1f} GB"

class SessionListConfig:
    """Configuration for session list display and behavior."""
    
    def __init__(self):
        # Display options
        self.show_thumbnails = True
        self.show_duration = True
        self.show_size = True
        self.show_status_icons = True
        self.items_per_page = 50
        
        # Touch optimization
        self.item_height = dp(80)
        self.item_padding = dp(8)
        self.touch_target_size = dp(44)
        
        # Search and filter
        self.enable_search = True
        self.enable_filtering = True
        self.enable_sorting = True
        self.enable_multi_select = True
        
        # Animation
        self.scroll_animation = True
        self.item_animation = True
        self.selection_animation = True
        
        # Colors and theming
        self.selected_color = get_color_from_hex('#3498DB') if callable(get_color_from_hex) else '#3498DB'
        self.archived_color = get_color_from_hex('#95A5A6') if callable(get_color_from_hex) else '#95A5A6'
        self.default_color = get_color_from_hex('#2C3E50') if callable(get_color_from_hex) else '#2C3E50'

class SessionListItemWidget(BoxLayout):
    """Individual session item widget."""
    
    session_id = StringProperty()
    title = StringProperty()
    date_text = StringProperty()
    duration_text = StringProperty()
    size_text = StringProperty()
    is_selected = BooleanProperty(False)
    is_archived = BooleanProperty(False)
    
    def __init__(self, session_item: SessionListItem, config: SessionListConfig, **kwargs):
        super().__init__(**kwargs)
        self.session_item = session_item
        self.config = config
        self.callbacks = []
        
        self.orientation = 'horizontal'
        self.size_hint_y = None
        self.height = config.item_height
        self.padding = config.item_padding
        
        self._build_ui()
        self._update_from_item()
    
    def _build_ui(self):
        """Build the UI for the session item."""
        # Selection checkbox (if multi-select enabled)
        if self.config.enable_multi_select:
            self.checkbox = CheckBox(
                size_hint=(None, 1),
                width=self.config.touch_target_size,
                active=self.session_item.is_selected
            )
            self.checkbox.bind(active=self._on_selection_changed)
            self.add_widget(self.checkbox)
        
        # Main content area
        main_content = BoxLayout(orientation='vertical', spacing=dp(2))
        
        # Title and date row
        title_row = BoxLayout(orientation='horizontal', size_hint_y=0.6)
        
        self.title_label = Label(
            text=self.session_item.title,
            text_size=(None, None),
            halign='left',
            valign='middle',
            font_size=dp(16),
            bold=True
        )
        title_row.add_widget(self.title_label)
        
        self.date_label = Label(
            text=self.session_item.formatted_date,
            text_size=(None, None),
            halign='right',
            valign='middle',
            font_size=dp(12),
            size_hint_x=0.3
        )
        title_row.add_widget(self.date_label)
        
        main_content.add_widget(title_row)
        
        # Details row
        details_row = BoxLayout(orientation='horizontal', size_hint_y=0.4)
        
        if self.config.show_duration:
            self.duration_label = Label(
                text=f"Duration: {self.session_item.formatted_duration}",
                text_size=(None, None),
                halign='left',
                valign='middle',
                font_size=dp(11),
                size_hint_x=0.4
            )
            details_row.add_widget(self.duration_label)
        
        if self.config.show_size:
            self.size_label = Label(
                text=f"Size: {self.session_item.formatted_size}",
                text_size=(None, None),
                halign='left',
                valign='middle',
                font_size=dp(11),
                size_hint_x=0.3
            )
            details_row.add_widget(self.size_label)
        
        # Status icons
        if self.config.show_status_icons:
            status_layout = BoxLayout(orientation='horizontal', size_hint_x=0.3)
            
            if self.session_item.has_transcript:
                transcript_icon = Label(
                    text="📝",
                    font_size=dp(14),
                    size_hint_x=0.5
                )
                status_layout.add_widget(transcript_icon)
            
            if self.session_item.has_analysis:
                analysis_icon = Label(
                    text="🧠",
                    font_size=dp(14),
                    size_hint_x=0.5
                )
                status_layout.add_widget(analysis_icon)
            
            details_row.add_widget(status_layout)
        
        main_content.add_widget(details_row)
        self.add_widget(main_content)
        
        # Action button
        self.action_button = Button(
            text="⋮",
            size_hint=(None, 1),
            width=self.config.touch_target_size,
            font_size=dp(18)
        )
        self.action_button.bind(on_press=self._on_action_pressed)
        self.add_widget(self.action_button)
    
    def _update_from_item(self):
        """Update widget properties from session item."""
        self.session_id = self.session_item.session_id
        self.title = self.session_item.title
        self.date_text = self.session_item.formatted_date
        self.duration_text = self.session_item.formatted_duration
        self.size_text = self.session_item.formatted_size
        self.is_selected = self.session_item.is_selected
        self.is_archived = self.session_item.is_archived
        
        # Update colors based on state
        if self.is_archived:
            self.title_label.color = self.config.archived_color
        elif self.is_selected:
            self.canvas.before.clear()
            with self.canvas.before:
                from kivy.graphics import Color, Rectangle
                Color(*self.config.selected_color)
                Rectangle(pos=self.pos, size=self.size)
        else:
            self.title_label.color = self.config.default_color
    
    def _on_selection_changed(self, checkbox, active):
        """Handle selection change."""
        self.session_item.is_selected = active
        self.is_selected = active
        self._update_from_item()
        self._trigger_callback('selection_changed', self.session_item)
    
    def _on_action_pressed(self, button):
        """Handle action button press."""
        self._trigger_callback('action_pressed', self.session_item)
    
    def _trigger_callback(self, event_type: str, *args):
        """Trigger registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(event_type, *args)
            except Exception as e:
                print(f"Error in session list item callback: {e}")
    
    def add_callback(self, callback: Callable):
        """Add a callback for events."""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

class SessionListView(BoxLayout, EventDispatcher):
    """Main session list view with search, filter, and management capabilities."""
    
    def __init__(self, config: Optional[SessionListConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or SessionListConfig()
        self.orientation = 'vertical'
        self.spacing = dp(8)
        self.padding = dp(8)
        
        # State
        self.sessions = []
        self.filtered_sessions = []
        self.selected_sessions = []
        self.current_filter = SessionFilter.ALL
        self.current_sort = SessionSort.DATE
        self.sort_ascending = False
        self.search_query = ""
        self.multi_select_mode = False
        
        # Callbacks
        self.callbacks = []
        
        # Build UI
        self._build_ui()
        self._update_display()
    
    def _build_ui(self):
        """Build the main UI."""
        # Search and filter bar
        search_bar = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(50),
            spacing=dp(8)
        )
        
        if self.config.enable_search:
            self.search_input = TextInput(
                hint_text="Search sessions...",
                size_hint_x=0.5,
                multiline=False,
                font_size=dp(14)
            )
            self.search_input.bind(text=self._on_search_changed)
            search_bar.add_widget(self.search_input)
        
        if self.config.enable_filtering:
            filter_options = [filter_opt.value for filter_opt in SessionFilter]
            self.filter_spinner = Spinner(
                text=self.current_filter.value,
                values=filter_options,
                size_hint_x=0.25,
                font_size=dp(12)
            )
            self.filter_spinner.bind(text=self._on_filter_changed)
            search_bar.add_widget(self.filter_spinner)
        
        if self.config.enable_sorting:
            sort_options = [sort_opt.value for sort_opt in SessionSort]
            self.sort_spinner = Spinner(
                text=self.current_sort.value,
                values=sort_options,
                size_hint_x=0.25,
                font_size=dp(12)
            )
            self.sort_spinner.bind(text=self._on_sort_changed)
            search_bar.add_widget(self.sort_spinner)
        
        self.add_widget(search_bar)
        
        # Action bar (for multi-select operations)
        self.action_bar = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=0,  # Initially hidden
            spacing=dp(8)
        )
        
        self.select_all_btn = Button(
            text="Select All",
            size_hint_x=0.2,
            font_size=dp(12)
        )
        self.select_all_btn.bind(on_press=self._on_select_all)
        self.action_bar.add_widget(self.select_all_btn)
        
        self.delete_selected_btn = Button(
            text="Delete",
            size_hint_x=0.2,
            font_size=dp(12)
        )
        self.delete_selected_btn.bind(on_press=self._on_delete_selected)
        self.action_bar.add_widget(self.delete_selected_btn)
        
        self.archive_selected_btn = Button(
            text="Archive",
            size_hint_x=0.2,
            font_size=dp(12)
        )
        self.archive_selected_btn.bind(on_press=self._on_archive_selected)
        self.action_bar.add_widget(self.archive_selected_btn)
        
        self.export_selected_btn = Button(
            text="Export",
            size_hint_x=0.2,
            font_size=dp(12)
        )
        self.export_selected_btn.bind(on_press=self._on_export_selected)
        self.action_bar.add_widget(self.export_selected_btn)
        
        self.cancel_selection_btn = Button(
            text="Cancel",
            size_hint_x=0.2,
            font_size=dp(12)
        )
        self.cancel_selection_btn.bind(on_press=self._on_cancel_selection)
        self.action_bar.add_widget(self.cancel_selection_btn)
        
        self.add_widget(self.action_bar)
        
        # Session list scroll view
        self.scroll_view = ScrollView(
            do_scroll_x=False,
            do_scroll_y=True
        )
        
        self.session_list = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            spacing=dp(2)
        )
        self.session_list.bind(minimum_height=self.session_list.setter('height'))
        
        self.scroll_view.add_widget(self.session_list)
        self.add_widget(self.scroll_view)
    
    def load_sessions(self, sessions: List[SessionListItem]):
        """Load sessions into the list."""
        self.sessions = sessions.copy()
        self._apply_filters_and_sort()
        self._update_display()
        self._trigger_callback('sessions_loaded', len(self.sessions))
    
    def refresh_sessions(self):
        """Refresh the session list."""
        self._apply_filters_and_sort()
        self._update_display()
        self._trigger_callback('sessions_refreshed')
    
    def search_sessions(self, query: str):
        """Search sessions by query."""
        self.search_query = query.lower()
        if hasattr(self, 'search_input'):
            self.search_input.text = query
        self._apply_filters_and_sort()
        self._update_display()
    
    def filter_sessions(self, filter_type: SessionFilter):
        """Filter sessions by type."""
        self.current_filter = filter_type
        if hasattr(self, 'filter_spinner'):
            self.filter_spinner.text = filter_type.value
        self._apply_filters_and_sort()
        self._update_display()
    
    def sort_sessions(self, sort_type: SessionSort, ascending: bool = True):
        """Sort sessions."""
        self.current_sort = sort_type
        self.sort_ascending = ascending
        if hasattr(self, 'sort_spinner'):
            self.sort_spinner.text = sort_type.value
        self._apply_filters_and_sort()
        self._update_display()
    
    def delete_session(self, session_id: str):
        """Delete a session."""
        self.sessions = [s for s in self.sessions if s.session_id != session_id]
        self._apply_filters_and_sort()
        self._update_display()
        self._trigger_callback('session_deleted', session_id)
    
    def archive_session(self, session_id: str):
        """Archive a session."""
        for session in self.sessions:
            if session.session_id == session_id:
                session.is_archived = True
                break
        self._apply_filters_and_sort()
        self._update_display()
        self._trigger_callback('session_archived', session_id)
    
    def select_session(self, session_id: str):
        """Select a specific session."""
        for session in self.sessions:
            if session.session_id == session_id:
                session.is_selected = True
                if session not in self.selected_sessions:
                    self.selected_sessions.append(session)
                break
        self._update_display()
        self._trigger_callback('session_selected', session_id)
    
    def multi_select_mode(self, enabled: bool):
        """Enable or disable multi-select mode."""
        self.multi_select_mode = enabled
        if enabled:
            self.action_bar.height = dp(50)
        else:
            self.action_bar.height = 0
            self._clear_selection()
        self._update_display()
    
    def _apply_filters_and_sort(self):
        """Apply current filters and sorting."""
        # Start with all sessions
        filtered = self.sessions.copy()
        
        # Apply search filter
        if self.search_query:
            filtered = [s for s in filtered if self.search_query in s.title.lower()]
        
        # Apply date/status filter
        now = datetime.now()
        if self.current_filter == SessionFilter.TODAY:
            filtered = [s for s in filtered if s.date.date() == now.date()]
        elif self.current_filter == SessionFilter.THIS_WEEK:
            week_start = now - timedelta(days=now.weekday())
            filtered = [s for s in filtered if s.date >= week_start]
        elif self.current_filter == SessionFilter.THIS_MONTH:
            month_start = now.replace(day=1)
            filtered = [s for s in filtered if s.date >= month_start]
        elif self.current_filter == SessionFilter.ARCHIVED:
            filtered = [s for s in filtered if s.is_archived]
        elif self.current_filter == SessionFilter.ALL:
            filtered = [s for s in filtered if not s.is_archived]
        
        # Apply sorting
        if self.current_sort == SessionSort.DATE:
            filtered.sort(key=lambda s: s.date, reverse=not self.sort_ascending)
        elif self.current_sort == SessionSort.DURATION:
            filtered.sort(key=lambda s: s.duration, reverse=not self.sort_ascending)
        elif self.current_sort == SessionSort.NAME:
            filtered.sort(key=lambda s: s.title.lower(), reverse=not self.sort_ascending)
        elif self.current_sort == SessionSort.SIZE:
            filtered.sort(key=lambda s: s.size, reverse=not self.sort_ascending)
        
        self.filtered_sessions = filtered
    
    def _update_display(self):
        """Update the visual display."""
        # Clear current list
        self.session_list.clear_widgets()
        
        # Add filtered sessions
        for session in self.filtered_sessions:
            item_widget = SessionListItemWidget(session, self.config)
            item_widget.add_callback(self._on_item_event)
            self.session_list.add_widget(item_widget)
        
        # Update action bar visibility
        if self.selected_sessions:
            self.action_bar.height = dp(50)
        else:
            self.action_bar.height = 0
    
    def _on_search_changed(self, instance, value):
        """Handle search text change."""
        self.search_query = value.lower()
        if Clock:
            Clock.unschedule(self._delayed_search)
            Clock.schedule_once(self._delayed_search, 0.3)
        else:
            self._delayed_search(None)
    
    def _delayed_search(self, dt):
        """Perform delayed search to avoid too frequent updates."""
        self._apply_filters_and_sort()
        self._update_display()
    
    def _on_filter_changed(self, instance, value):
        """Handle filter change."""
        try:
            self.current_filter = SessionFilter(value)
            self._apply_filters_and_sort()
            self._update_display()
        except ValueError:
            pass
    
    def _on_sort_changed(self, instance, value):
        """Handle sort change."""
        try:
            self.current_sort = SessionSort(value)
            self._apply_filters_and_sort()
            self._update_display()
        except ValueError:
            pass
    
    def _on_item_event(self, event_type: str, session_item: SessionListItem):
        """Handle events from session items."""
        if event_type == 'selection_changed':
            if session_item.is_selected:
                if session_item not in self.selected_sessions:
                    self.selected_sessions.append(session_item)
            else:
                if session_item in self.selected_sessions:
                    self.selected_sessions.remove(session_item)
            self._update_display()
        elif event_type == 'action_pressed':
            self._trigger_callback('session_action_requested', session_item.session_id)
    
    def _on_select_all(self, button):
        """Handle select all button."""
        for session in self.filtered_sessions:
            session.is_selected = True
            if session not in self.selected_sessions:
                self.selected_sessions.append(session)
        self._update_display()
    
    def _on_delete_selected(self, button):
        """Handle delete selected button."""
        session_ids = [s.session_id for s in self.selected_sessions]
        self._trigger_callback('delete_sessions_requested', session_ids)
    
    def _on_archive_selected(self, button):
        """Handle archive selected button."""
        session_ids = [s.session_id for s in self.selected_sessions]
        self._trigger_callback('archive_sessions_requested', session_ids)
    
    def _on_export_selected(self, button):
        """Handle export selected button."""
        session_ids = [s.session_id for s in self.selected_sessions]
        self._trigger_callback('export_sessions_requested', session_ids)
    
    def _on_cancel_selection(self, button):
        """Handle cancel selection button."""
        self._clear_selection()
        self._update_display()
    
    def _clear_selection(self):
        """Clear all selections."""
        for session in self.sessions:
            session.is_selected = False
        self.selected_sessions.clear()
    
    def _trigger_callback(self, event_type: str, *args):
        """Trigger registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(event_type, *args)
            except Exception as e:
                print(f"Error in session list callback: {e}")
    
    def add_callback(self, callback: Callable):
        """Add a callback for events."""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

# Factory functions
def create_session_list_view(config: Optional[SessionListConfig] = None) -> SessionListView:
    """Create a session list view with optional configuration."""
    return SessionListView(config)

def create_default_config() -> SessionListConfig:
    """Create default session list configuration."""
    return SessionListConfig()

def create_compact_config() -> SessionListConfig:
    """Create compact session list configuration."""
    config = SessionListConfig()
    config.item_height = dp(60)
    config.show_size = False
    config.show_status_icons = False
    config.items_per_page = 100
    return config

# Demo functions
def demo_session_list_view():
    """Demo function to test the session list view."""
    from kivy.app import App
    
    class SessionListDemo(App):
        def build(self):
            # Create sample sessions
            sessions = []
            for i in range(20):
                session = SessionListItem(
                    session_id=f"session_{i}",
                    title=f"Meeting {i+1}",
                    date=datetime.now() - timedelta(days=i),
                    duration=1800 + i * 300,  # 30 minutes + extra
                    size=1024 * 1024 * (i + 1),  # 1MB + extra
                    has_transcript=True,
                    has_analysis=i % 2 == 0,
                    is_archived=i > 15
                )
                sessions.append(session)
            
            # Create session list view
            session_list = create_session_list_view()
            session_list.add_callback(self._on_session_event)
            session_list.load_sessions(sessions)
            
            return session_list
        
        def _on_session_event(self, event_type: str, *args):
            print(f"Session list event: {event_type}, args: {args}")
    
    return SessionListDemo()

@contextmanager
def session_list_context(config: Optional[SessionListConfig] = None):
    """Context manager for session list operations."""
    session_list = create_session_list_view(config)
    try:
        yield session_list
    finally:
        # Cleanup if needed
        session_list.callbacks.clear()

if __name__ == "__main__":
    demo = demo_session_list_view()
    demo.run()