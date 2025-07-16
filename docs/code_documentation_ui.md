# UI Module Documentation

## Module Overview

The UI module provides a comprehensive touch-optimized user interface for The Silent Steno application. Built with Kivy framework, it implements a complete interface system including live session management, transcript display, audio visualization, settings management, and system monitoring. The module is designed specifically for Raspberry Pi touchscreens with emphasis on accessibility, performance, and user experience.

## Dependencies

### External Dependencies
- `kivy` - Main UI framework
- `kivymd` - Material Design components
- `pygame` - Game engine for audio/graphics
- `numpy` - Numerical operations for visualizations
- `PIL` (Pillow) - Image processing
- `threading` - Thread management for real-time updates
- `asyncio` - Asynchronous operations
- `json` - Configuration handling
- `pathlib` - Path operations
- `datetime` - Date/time operations
- `enum` - Enumerations
- `dataclasses` - Data structures
- `typing` - Type hints
- `logging` - Logging system
- `queue` - Thread-safe queues
- `time` - Timing operations

### Internal Dependencies
- `src.core.events` - Event system
- `src.core.config` - Configuration management
- `src.core.logging` - Logging system
- `src.recording.session_manager` - Session management
- `src.export.bulk_exporter` - Export functionality
- `src.system.storage_cleanup` - Storage management

## Architecture Overview

The UI module follows a widget-based architecture with:
- **Touch-first design**: All components optimized for finger interaction
- **Real-time updates**: Live audio visualization and transcript display
- **Theming system**: Comprehensive dark/light mode support
- **Accessibility**: High contrast modes and configurable text sizes
- **Performance monitoring**: Frame rate and latency tracking
- **Multi-modal feedback**: Visual, audio, and haptic feedback

## File Documentation

### 1. Module Structure

**Note**: The `__init__.py` file is not present in the module, suggesting components are imported directly.

### 2. `audio_visualizer.py`

**Purpose**: Real-time audio visualization with multiple display modes for live audio monitoring.

#### Enums

##### `VisualizerType`
Audio visualizer types.
- `BARS` - Bar graph visualization
- `WAVEFORM` - Waveform display
- `SPECTRUM` - Frequency spectrum
- `CIRCULAR` - Circular visualization
- `VU_METER` - VU meter style

#### Classes

##### `AudioLevel`
Audio level data structure.

**Attributes:**
- `left_level: float` - Left channel level (0-1)
- `right_level: float` - Right channel level (0-1)
- `peak_level: float` - Peak level
- `rms_level: float` - RMS level
- `frequency_data: List[float]` - Frequency spectrum data
- `timestamp: float` - Timestamp of measurement

##### `VisualizerConfig`
Configuration for audio visualizer.

**Attributes:**
- `visualizer_type: VisualizerType` - Type of visualization
- `update_rate: float` - Update rate in Hz
- `sensitivity: float` - Sensitivity (0-1)
- `decay_rate: float` - Level decay rate
- `bar_count: int` - Number of bars (for bar type)
- `show_peak_hold: bool` - Show peak hold indicators
- `colors: dict` - Color configuration
- `enable_clipping_detection: bool` - Enable clipping detection

##### `AudioVisualizer`
Main audio visualizer widget.

**Methods:**
- `__init__(config: VisualizerConfig)` - Initialize with configuration
- `update_audio_level(level: AudioLevel)` - Update audio level
- `start_visualization()` - Start real-time visualization
- `stop_visualization()` - Stop visualization
- `set_visualizer_type(type: VisualizerType)` - Change visualization type
- `set_sensitivity(sensitivity: float)` - Set sensitivity
- `get_performance_metrics()` - Get performance metrics

#### Factory Functions

##### `create_bar_visualizer(config: dict = None) -> AudioVisualizer`
Create bar-style visualizer.

##### `create_waveform_visualizer(config: dict = None) -> AudioVisualizer`
Create waveform visualizer.

##### `create_spectrum_visualizer(config: dict = None) -> AudioVisualizer`
Create spectrum analyzer.

**Usage Example:**
```python
# Create bar visualizer
config = VisualizerConfig(
    visualizer_type=VisualizerType.BARS,
    update_rate=60.0,
    sensitivity=0.8,
    bar_count=20,
    show_peak_hold=True
)

visualizer = AudioVisualizer(config)

# Or use factory function
visualizer = create_bar_visualizer({
    "sensitivity": 0.8,
    "bar_count": 20,
    "colors": {"bars": "#00ff00", "background": "#000000"}
})

# Update with audio data
audio_level = AudioLevel(
    left_level=0.6,
    right_level=0.7,
    peak_level=0.8,
    rms_level=0.5,
    frequency_data=[0.1, 0.3, 0.5, 0.7, 0.4]
)

visualizer.update_audio_level(audio_level)
```

### 3. `export_dialog.py`

**Purpose**: Comprehensive export dialog for sessions with multiple format options and content selection.

#### Enums

##### `ExportFormat`
Supported export formats.
- `PDF` - PDF document
- `DOCX` - Word document
- `TXT` - Plain text
- `JSON` - JSON format
- `HTML` - HTML document

##### `ExportContent`
Content types for export.
- `AUDIO` - Audio file
- `TRANSCRIPT` - Transcript text
- `ANALYSIS` - Meeting analysis
- `METADATA` - Session metadata

#### Classes

##### `ExportOptions`
Export configuration options.

**Attributes:**
- `format: ExportFormat` - Export format
- `content_types: List[ExportContent]` - Content to include
- `include_timestamps: bool` - Include timestamps
- `include_speaker_labels: bool` - Include speaker identification
- `include_confidence_scores: bool` - Include confidence scores
- `output_path: str` - Output file path
- `custom_template: str` - Custom template path
- `quality_settings: dict` - Quality settings

##### `ExportDialog`
Main export dialog widget.

**Methods:**
- `__init__(session_data: dict, config: dict = None)` - Initialize with session data
- `show()` - Show export dialog
- `hide()` - Hide export dialog
- `start_export()` - Start export process
- `cancel_export()` - Cancel export
- `get_export_options()` - Get current export options
- `set_export_options(options: ExportOptions)` - Set export options
- `validate_options()` - Validate export options

**Usage Example:**
```python
# Create export dialog
session_data = {
    "session_id": "session_123",
    "transcript": "Meeting transcript...",
    "audio_file": "path/to/audio.wav",
    "analysis": {"summary": "Meeting summary..."}
}

export_dialog = ExportDialog(session_data)

# Configure export options
options = ExportOptions(
    format=ExportFormat.PDF,
    content_types=[ExportContent.TRANSCRIPT, ExportContent.ANALYSIS],
    include_timestamps=True,
    include_speaker_labels=True,
    output_path="exports/meeting_export.pdf"
)

export_dialog.set_export_options(options)

# Show dialog
export_dialog.show()

# Handle export completion
def on_export_complete(success, file_path):
    if success:
        print(f"Export completed: {file_path}")
    else:
        print("Export failed")

export_dialog.bind(on_export_complete=on_export_complete)
```

### 4. `feedback_manager.py`

**Purpose**: Multi-modal feedback system providing visual, audio, and haptic feedback for user interactions.

#### Enums

##### `FeedbackType`
Types of feedback.
- `VISUAL` - Visual effects
- `AUDIO` - Sound effects
- `HAPTIC` - Vibration/haptic feedback

##### `FeedbackIntensity`
Feedback intensity levels.
- `LOW` - Subtle feedback
- `MEDIUM` - Normal feedback
- `HIGH` - Strong feedback

#### Classes

##### `VisualFeedback`
Visual feedback effects.

**Methods:**
- `ripple_effect(widget, position: tuple)` - Ripple effect at position
- `highlight_effect(widget, duration: float)` - Highlight effect
- `pulse_effect(widget, duration: float)` - Pulse effect
- `shake_effect(widget, intensity: float)` - Shake effect

##### `AudioFeedback`
Audio feedback system.

**Methods:**
- `play_click()` - Play click sound
- `play_success()` - Play success sound
- `play_error()` - Play error sound
- `play_notification()` - Play notification sound
- `set_volume(volume: float)` - Set audio volume

##### `HapticFeedback`
Haptic feedback system.

**Methods:**
- `vibrate(duration: float, intensity: float)` - Vibrate device
- `tap_feedback()` - Single tap feedback
- `double_tap_feedback()` - Double tap feedback
- `long_press_feedback()` - Long press feedback

##### `FeedbackManager`
Main feedback coordinator.

**Methods:**
- `__init__(config: dict = None)` - Initialize with configuration
- `provide_feedback(feedback_type: FeedbackType, action: str, widget=None)` - Provide feedback
- `set_intensity(intensity: FeedbackIntensity)` - Set feedback intensity
- `enable_feedback(feedback_type: FeedbackType, enabled: bool)` - Enable/disable feedback
- `set_accessibility_mode(enabled: bool)` - Set accessibility mode

**Usage Example:**
```python
# Create feedback manager
feedback_manager = FeedbackManager({
    "visual_enabled": True,
    "audio_enabled": True,
    "haptic_enabled": True,
    "intensity": FeedbackIntensity.MEDIUM
})

# Provide feedback for button tap
def on_button_tap(widget):
    feedback_manager.provide_feedback(
        FeedbackType.VISUAL, 
        "button_tap", 
        widget
    )
    feedback_manager.provide_feedback(
        FeedbackType.AUDIO, 
        "button_tap"
    )
    feedback_manager.provide_feedback(
        FeedbackType.HAPTIC, 
        "button_tap"
    )

# Enable accessibility mode
feedback_manager.set_accessibility_mode(True)

# Set high intensity for accessibility
feedback_manager.set_intensity(FeedbackIntensity.HIGH)
```

### 5. `main_window.py`

**Purpose**: Main application window with touchscreen interface and screen management.

#### Classes

##### `WindowConfig`
Configuration for main window.

**Attributes:**
- `title: str` - Window title
- `size: tuple` - Window size (width, height)
- `fullscreen: bool` - Fullscreen mode
- `resizable: bool` - Resizable window
- `minimum_size: tuple` - Minimum window size
- `show_cursor: bool` - Show cursor
- `orientation: str` - Screen orientation

##### `ScreenManager`
Enhanced screen navigation with history.

**Methods:**
- `switch_to_screen(screen_name: str, direction: str = "left")` - Switch to screen
- `go_back()` - Go back to previous screen
- `add_screen(screen)` - Add screen to manager
- `remove_screen(screen_name: str)` - Remove screen
- `get_current_screen()` - Get current screen
- `get_screen_history()` - Get navigation history

##### `MainWindow`
Main application window.

**Methods:**
- `__init__(config: WindowConfig)` - Initialize with configuration
- `initialize_ui()` - Initialize UI components
- `setup_screens()` - Setup application screens
- `handle_key_event(key, scancode, codepoint, modifier)` - Handle keyboard events
- `handle_touch_event(touch)` - Handle touch events
- `toggle_fullscreen()` - Toggle fullscreen mode
- `get_performance_metrics()` - Get performance metrics

##### `SilentStenoApp`
Main Kivy application class.

**Methods:**
- `build()` - Build application
- `on_start()` - Application start callback
- `on_stop()` - Application stop callback
- `on_pause()` - Application pause callback
- `on_resume()` - Application resume callback

**Usage Example:**
```python
# Create window configuration
config = WindowConfig(
    title="Silent Steno",
    size=(800, 480),
    fullscreen=True,
    show_cursor=False,
    orientation="landscape"
)

# Create main window
main_window = MainWindow(config)

# Initialize and setup
main_window.initialize_ui()
main_window.setup_screens()

# Handle events
def on_key_press(key, scancode, codepoint, modifier):
    if key == 27:  # ESC key
        main_window.toggle_fullscreen()

main_window.bind(on_key_down=on_key_press)

# Create and run application
app = SilentStenoApp()
app.run()
```

### 6. `navigation.py`

**Purpose**: Navigation management with gesture support and screen transitions.

#### Classes

##### `NavigationConfig`
Configuration for navigation behavior.

**Attributes:**
- `gesture_enabled: bool` - Enable gesture navigation
- `swipe_threshold: float` - Swipe distance threshold
- `tap_threshold: float` - Tap time threshold
- `long_press_threshold: float` - Long press time threshold
- `animation_duration: float` - Transition animation duration
- `enable_back_gesture: bool` - Enable back gesture

##### `GestureDetector`
Touch gesture recognition.

**Methods:**
- `__init__(config: NavigationConfig)` - Initialize with configuration
- `detect_gesture(touch_points: List[tuple])` - Detect gesture from touch points
- `on_tap(position: tuple)` - Handle tap gesture
- `on_long_press(position: tuple)` - Handle long press
- `on_swipe(start_pos: tuple, end_pos: tuple)` - Handle swipe gesture
- `on_pinch(scale: float)` - Handle pinch gesture

##### `NavigationBar`
Touch-optimized navigation bar.

**Methods:**
- `__init__(config: dict = None)` - Initialize navigation bar
- `add_nav_item(name: str, icon: str, callback: callable)` - Add navigation item
- `remove_nav_item(name: str)` - Remove navigation item
- `set_active_item(name: str)` - Set active navigation item
- `update_badge(name: str, count: int)` - Update badge count

##### `NavigationManager`
Main navigation controller.

**Methods:**
- `__init__(screen_manager, config: NavigationConfig)` - Initialize with screen manager
- `navigate_to(screen_name: str, transition: str = "slide")` - Navigate to screen
- `go_back()` - Go back to previous screen
- `set_root_screen(screen_name: str)` - Set root screen
- `get_navigation_stack()` - Get navigation stack
- `clear_navigation_stack()` - Clear navigation stack

**Usage Example:**
```python
# Create navigation configuration
nav_config = NavigationConfig(
    gesture_enabled=True,
    swipe_threshold=50.0,
    tap_threshold=0.2,
    long_press_threshold=0.5,
    animation_duration=0.3
)

# Create navigation manager
screen_manager = ScreenManager()
nav_manager = NavigationManager(screen_manager, nav_config)

# Create navigation bar
nav_bar = NavigationBar()
nav_bar.add_nav_item("home", "home", lambda: nav_manager.navigate_to("home"))
nav_bar.add_nav_item("sessions", "list", lambda: nav_manager.navigate_to("sessions"))
nav_bar.add_nav_item("settings", "settings", lambda: nav_manager.navigate_to("settings"))

# Handle gestures
gesture_detector = GestureDetector(nav_config)

def on_swipe(start_pos, end_pos):
    # Swipe right to go back
    if end_pos[0] - start_pos[0] > nav_config.swipe_threshold:
        nav_manager.go_back()

gesture_detector.bind(on_swipe=on_swipe)
```

### 7. `session_controls.py`

**Purpose**: Touch-optimized controls for recording session management.

#### Enums

##### `SessionAction`
Session control actions.
- `START` - Start recording
- `STOP` - Stop recording
- `PAUSE` - Pause recording
- `RESUME` - Resume recording
- `CANCEL` - Cancel recording

##### `ControlLayout`
Control layout types.
- `HORIZONTAL` - Horizontal layout
- `VERTICAL` - Vertical layout
- `GRID` - Grid layout
- `CIRCULAR` - Circular layout

#### Classes

##### `ControlsConfig`
Configuration for session controls.

**Attributes:**
- `layout: ControlLayout` - Control layout type
- `button_size: tuple` - Button size (width, height)
- `button_spacing: float` - Spacing between buttons
- `show_labels: bool` - Show button labels
- `enable_animations: bool` - Enable animations
- `confirmation_dialogs: bool` - Show confirmation dialogs

##### `SessionControlButton`
Enhanced touch button for session actions.

**Methods:**
- `__init__(action: SessionAction, config: dict = None)` - Initialize with action
- `set_enabled(enabled: bool)` - Enable/disable button
- `set_state(state: str)` - Set button state
- `animate_press()` - Animate button press
- `show_confirmation(message: str, callback: callable)` - Show confirmation dialog

##### `SessionControls`
Main session control widget.

**Methods:**
- `__init__(config: ControlsConfig)` - Initialize with configuration
- `set_session_state(state: str)` - Set session state
- `get_available_actions()` - Get available actions
- `enable_action(action: SessionAction, enabled: bool)` - Enable/disable action
- `set_callback(action: SessionAction, callback: callable)` - Set action callback
- `update_timer(elapsed_time: float)` - Update session timer

**Usage Example:**
```python
# Create session controls
config = ControlsConfig(
    layout=ControlLayout.HORIZONTAL,
    button_size=(100, 100),
    button_spacing=20,
    show_labels=True,
    confirmation_dialogs=True
)

controls = SessionControls(config)

# Set up callbacks
def on_start_session():
    print("Starting session...")
    controls.set_session_state("recording")

def on_stop_session():
    print("Stopping session...")
    controls.set_session_state("stopped")

controls.set_callback(SessionAction.START, on_start_session)
controls.set_callback(SessionAction.STOP, on_stop_session)

# Update session state
controls.set_session_state("ready")

# Update timer
elapsed_time = 0
while True:
    controls.update_timer(elapsed_time)
    elapsed_time += 1
    time.sleep(1)
```

### 8. `session_details_view.py`

**Purpose**: Comprehensive session details view with metadata, audio player, and actions.

#### Classes

##### `SessionMetadata`
Session metadata container.

**Attributes:**
- `session_id: str` - Session identifier
- `title: str` - Session title
- `duration: float` - Session duration
- `start_time: datetime` - Start time
- `end_time: datetime` - End time
- `participant_count: int` - Number of participants
- `transcript_length: int` - Transcript length in characters
- `audio_quality: float` - Audio quality score
- `file_size: int` - File size in bytes

##### `AudioPlayerWidget`
Audio playback controls.

**Methods:**
- `__init__(audio_file: str, config: dict = None)` - Initialize with audio file
- `play()` - Play audio
- `pause()` - Pause audio
- `stop()` - Stop audio
- `seek(position: float)` - Seek to position
- `set_playback_speed(speed: float)` - Set playback speed
- `get_current_position()` - Get current position
- `get_duration()` - Get audio duration

##### `SessionDetailsView`
Main session details view.

**Methods:**
- `__init__(session_data: dict, config: dict = None)` - Initialize with session data
- `load_session_data(session_id: str)` - Load session data
- `update_display()` - Update display with current data
- `export_session(format: str)` - Export session
- `share_session()` - Share session
- `edit_session()` - Edit session metadata
- `delete_session()` - Delete session

**Usage Example:**
```python
# Create session details view
session_data = {
    "session_id": "session_123",
    "title": "Team Meeting",
    "duration": 3600.0,
    "start_time": datetime.now(),
    "audio_file": "path/to/audio.wav",
    "transcript": "Meeting transcript...",
    "participants": ["Alice", "Bob", "Charlie"]
}

details_view = SessionDetailsView(session_data)

# Set up action callbacks
def on_export(format):
    print(f"Exporting session in {format} format")

def on_delete():
    print("Deleting session")

details_view.bind(on_export=on_export)
details_view.bind(on_delete=on_delete)

# Load and display
details_view.load_session_data("session_123")
details_view.update_display()
```

### 9. `session_list_view.py`

**Purpose**: Session browsing interface with search, filter, and management capabilities.

#### Classes

##### `SessionListItem`
Data structure for session list items.

**Attributes:**
- `session_id: str` - Session identifier
- `title: str` - Session title
- `date: datetime` - Session date
- `duration: float` - Session duration
- `size: int` - File size
- `status: str` - Session status
- `thumbnail: str` - Thumbnail path

##### `SessionListConfig`
Configuration for session list.

**Attributes:**
- `items_per_page: int` - Items per page
- `enable_search: bool` - Enable search
- `enable_filtering: bool` - Enable filtering
- `enable_sorting: bool` - Enable sorting
- `show_thumbnails: bool` - Show thumbnails
- `multi_select: bool` - Enable multi-select

##### `SessionListView`
Main session list view.

**Methods:**
- `__init__(config: SessionListConfig)` - Initialize with configuration
- `load_sessions()` - Load session list
- `refresh_sessions()` - Refresh session list
- `search_sessions(query: str)` - Search sessions
- `filter_sessions(filters: dict)` - Filter sessions
- `sort_sessions(sort_by: str, ascending: bool)` - Sort sessions
- `select_session(session_id: str)` - Select session
- `delete_selected_sessions()` - Delete selected sessions

**Usage Example:**
```python
# Create session list configuration
config = SessionListConfig(
    items_per_page=20,
    enable_search=True,
    enable_filtering=True,
    enable_sorting=True,
    multi_select=True
)

# Create session list view
session_list = SessionListView(config)

# Load sessions
session_list.load_sessions()

# Set up search
def on_search(query):
    session_list.search_sessions(query)

# Set up filtering
def on_filter(filters):
    session_list.filter_sessions(filters)

# Set up selection
def on_session_selected(session_id):
    print(f"Selected session: {session_id}")

session_list.bind(on_session_selected=on_session_selected)

# Enable multi-select operations
def on_delete_selected():
    session_list.delete_selected_sessions()

session_list.bind(on_delete_selected=on_delete_selected)
```

### 10. `session_view.py`

**Purpose**: Main live session screen orchestrating all live session components.

#### Classes

##### `SessionInfo`
Session metadata and state.

**Attributes:**
- `session_id: str` - Session identifier
- `title: str` - Session title
- `start_time: datetime` - Start time
- `elapsed_time: float` - Elapsed time
- `status: str` - Session status
- `participant_count: int` - Number of participants
- `transcript_length: int` - Current transcript length

##### `SessionViewConfig`
Configuration for session view.

**Attributes:**
- `show_timer: bool` - Show session timer
- `show_transcript: bool` - Show live transcript
- `show_visualizer: bool` - Show audio visualizer
- `show_controls: bool` - Show session controls
- `auto_scroll_transcript: bool` - Auto-scroll transcript
- `demo_mode: bool` - Enable demo mode

##### `SessionView`
Main session screen.

**Methods:**
- `__init__(config: SessionViewConfig)` - Initialize with configuration
- `start_session(title: str)` - Start new session
- `stop_session()` - Stop current session
- `pause_session()` - Pause session
- `resume_session()` - Resume session
- `update_transcript(text: str, speaker: str)` - Update transcript
- `update_audio_levels(levels: AudioLevel)` - Update audio visualization
- `get_session_info()` - Get session information

**Usage Example:**
```python
# Create session view configuration
config = SessionViewConfig(
    show_timer=True,
    show_transcript=True,
    show_visualizer=True,
    show_controls=True,
    auto_scroll_transcript=True,
    demo_mode=False
)

# Create session view
session_view = SessionView(config)

# Start session
session_view.start_session("Team Meeting")

# Update transcript in real-time
def on_transcript_update(text, speaker):
    session_view.update_transcript(text, speaker)

# Update audio levels
def on_audio_levels(levels):
    session_view.update_audio_levels(levels)

# Handle session events
def on_session_stopped():
    info = session_view.get_session_info()
    print(f"Session stopped: {info.title}")
    print(f"Duration: {info.elapsed_time:.1f}s")

session_view.bind(on_session_stopped=on_session_stopped)
```

### 11. `settings_view.py`

**Purpose**: Comprehensive settings interface with categorized options and validation.

#### Classes

##### `SettingItem`
Individual setting configuration.

**Attributes:**
- `key: str` - Setting key
- `title: str` - Display title
- `description: str` - Setting description
- `type: str` - Setting type ("bool", "int", "float", "str", "list")
- `default_value: Any` - Default value
- `options: List[Any]` - Available options (for list type)
- `min_value: float` - Minimum value (for numeric types)
- `max_value: float` - Maximum value (for numeric types)
- `validator: callable` - Validation function

##### `SettingsConfig`
Configuration for settings display.

**Attributes:**
- `categories: List[str]` - Setting categories
- `show_descriptions: bool` - Show setting descriptions
- `enable_search: bool` - Enable search
- `enable_import_export: bool` - Enable import/export
- `show_advanced: bool` - Show advanced settings
- `auto_save: bool` - Auto-save changes

##### `SettingsView`
Main settings view.

**Methods:**
- `__init__(config: SettingsConfig)` - Initialize with configuration
- `load_settings()` - Load settings from storage
- `save_settings()` - Save settings to storage
- `reset_settings()` - Reset to default values
- `import_settings(file_path: str)` - Import settings from file
- `export_settings(file_path: str)` - Export settings to file
- `search_settings(query: str)` - Search settings
- `validate_setting(key: str, value: Any)` - Validate setting value

**Usage Example:**
```python
# Create settings configuration
config = SettingsConfig(
    categories=["General", "Audio", "AI", "Storage"],
    show_descriptions=True,
    enable_search=True,
    enable_import_export=True,
    auto_save=True
)

# Create settings view
settings_view = SettingsView(config)

# Load settings
settings_view.load_settings()

# Set up validation
def validate_sample_rate(value):
    return value in [8000, 16000, 22050, 44100, 48000]

settings_view.register_validator("audio.sample_rate", validate_sample_rate)

# Handle setting changes
def on_setting_changed(key, value):
    print(f"Setting {key} changed to {value}")

settings_view.bind(on_setting_changed=on_setting_changed)

# Export settings
settings_view.export_settings("settings_backup.json")
```

### 12. `status_indicators.py`

**Purpose**: Visual indicators for system status and health monitoring.

#### Enums

##### `ConnectionStatus`
Bluetooth connection status.
- `DISCONNECTED` - Not connected
- `CONNECTING` - Connecting
- `CONNECTED` - Connected
- `FAILED` - Connection failed

##### `SystemStatus`
System health status.
- `HEALTHY` - System healthy
- `WARNING` - Warning condition
- `ERROR` - Error condition
- `CRITICAL` - Critical condition

#### Classes

##### `StatusInfo`
Status information container.

**Attributes:**
- `component: str` - Component name
- `status: str` - Status value
- `message: str` - Status message
- `timestamp: datetime` - Status timestamp
- `severity: str` - Severity level
- `details: dict` - Additional details

##### `StatusIndicators`
Main status display widget.

**Methods:**
- `__init__(config: dict = None)` - Initialize with configuration
- `update_bluetooth_status(status: ConnectionStatus)` - Update Bluetooth status
- `update_battery_status(level: float, charging: bool)` - Update battery status
- `update_storage_status(used: float, total: float)` - Update storage status
- `update_system_status(status: SystemStatus, message: str)` - Update system status
- `show_alert(message: str, severity: str)` - Show alert
- `get_all_status()` - Get all status information

**Usage Example:**
```python
# Create status indicators
status_indicators = StatusIndicators({
    "show_bluetooth": True,
    "show_battery": True,
    "show_storage": True,
    "show_system": True,
    "alert_duration": 5.0
})

# Update status information
status_indicators.update_bluetooth_status(ConnectionStatus.CONNECTED)
status_indicators.update_battery_status(85.0, True)
status_indicators.update_storage_status(50.0, 100.0)
status_indicators.update_system_status(SystemStatus.HEALTHY, "All systems operational")

# Show alerts
status_indicators.show_alert("Low storage space", "warning")

# Get status summary
all_status = status_indicators.get_all_status()
for component, info in all_status.items():
    print(f"{component}: {info.status} - {info.message}")
```

### 13. `storage_monitor_widget.py`

**Purpose**: Storage monitoring and management interface.

#### Classes

##### `StorageInfo`
Storage information container.

**Attributes:**
- `total_space: int` - Total storage space in bytes
- `used_space: int` - Used storage space in bytes
- `available_space: int` - Available storage space in bytes
- `session_count: int` - Number of sessions
- `largest_session: int` - Largest session size
- `oldest_session: datetime` - Oldest session date

##### `StorageConfig`
Configuration for storage monitoring.

**Attributes:**
- `update_interval: float` - Update interval in seconds
- `warning_threshold: float` - Warning threshold (0-1)
- `critical_threshold: float` - Critical threshold (0-1)
- `enable_auto_cleanup: bool` - Enable auto cleanup
- `cleanup_threshold: float` - Cleanup threshold
- `show_session_details: bool` - Show session details

##### `StorageMonitorWidget`
Main storage monitoring widget.

**Methods:**
- `__init__(config: StorageConfig)` - Initialize with configuration
- `update_storage_info()` - Update storage information
- `start_monitoring()` - Start monitoring
- `stop_monitoring()` - Stop monitoring
- `cleanup_old_sessions(days: int)` - Cleanup old sessions
- `show_cleanup_dialog()` - Show cleanup confirmation
- `get_storage_report()` - Get storage report

**Usage Example:**
```python
# Create storage monitor configuration
config = StorageConfig(
    update_interval=30.0,
    warning_threshold=0.8,
    critical_threshold=0.9,
    enable_auto_cleanup=True,
    cleanup_threshold=0.95
)

# Create storage monitor
storage_monitor = StorageMonitorWidget(config)

# Handle storage alerts
def on_storage_alert(level, message):
    print(f"Storage Alert ({level}): {message}")
    if level == "critical":
        storage_monitor.show_cleanup_dialog()

storage_monitor.bind(on_storage_alert=on_storage_alert)

# Start monitoring
storage_monitor.start_monitoring()

# Get storage report
report = storage_monitor.get_storage_report()
print(f"Storage Report:")
print(f"  Total: {report.total_space / 1024 / 1024:.1f} MB")
print(f"  Used: {report.used_space / 1024 / 1024:.1f} MB")
print(f"  Available: {report.available_space / 1024 / 1024:.1f} MB")
```

### 14. `themes.py`

**Purpose**: Comprehensive theming system with dark/light mode support.

#### Classes

##### `ColorPalette`
Color definitions for UI elements.

**Attributes:**
- `primary: str` - Primary color
- `secondary: str` - Secondary color
- `background: str` - Background color
- `surface: str` - Surface color
- `text: str` - Text color
- `text_secondary: str` - Secondary text color
- `accent: str` - Accent color
- `error: str` - Error color
- `warning: str` - Warning color
- `success: str` - Success color

##### `Theme`
Base theme class.

**Attributes:**
- `name: str` - Theme name
- `colors: ColorPalette` - Color palette
- `font_sizes: dict` - Font size definitions
- `spacing: dict` - Spacing definitions
- `animations: dict` - Animation settings

##### `ThemeConfig`
Configuration for theme behavior.

**Attributes:**
- `auto_switch: bool` - Auto switch based on time
- `follow_system: bool` - Follow system theme
- `animation_duration: float` - Theme switch animation duration
- `persist_theme: bool` - Persist theme selection

##### `ThemeManager`
Main theme management system.

**Methods:**
- `__init__(config: ThemeConfig)` - Initialize with configuration
- `get_available_themes()` - Get available themes
- `set_theme(theme_name: str)` - Set active theme
- `get_current_theme()` - Get current theme
- `create_custom_theme(name: str, colors: ColorPalette)` - Create custom theme
- `import_theme(file_path: str)` - Import theme from file
- `export_theme(theme_name: str, file_path: str)` - Export theme to file

**Usage Example:**
```python
# Create theme configuration
config = ThemeConfig(
    auto_switch=True,
    follow_system=True,
    animation_duration=0.3,
    persist_theme=True
)

# Create theme manager
theme_manager = ThemeManager(config)

# Get available themes
themes = theme_manager.get_available_themes()
print(f"Available themes: {themes}")

# Set theme
theme_manager.set_theme("dark")

# Create custom theme
custom_colors = ColorPalette(
    primary="#3498db",
    secondary="#2ecc71",
    background="#2c3e50",
    surface="#34495e",
    text="#ecf0f1",
    text_secondary="#bdc3c7",
    accent="#e74c3c",
    error="#e74c3c",
    warning="#f39c12",
    success="#27ae60"
)

theme_manager.create_custom_theme("my_theme", custom_colors)

# Handle theme changes
def on_theme_changed(theme_name):
    print(f"Theme changed to: {theme_name}")

theme_manager.bind(on_theme_changed=on_theme_changed)
```

### 15. `touch_controls.py`

**Purpose**: Touch-specific UI controls with haptic and visual feedback.

#### Classes

##### `TouchConfig`
Configuration for touch behavior.

**Attributes:**
- `minimum_touch_size: int` - Minimum touch target size
- `tap_threshold: float` - Tap time threshold
- `long_press_threshold: float` - Long press time threshold
- `double_tap_threshold: float` - Double tap time threshold
- `feedback_enabled: bool` - Enable touch feedback
- `haptic_enabled: bool` - Enable haptic feedback

##### `TouchButton`
Touch-optimized button with gesture support.

**Methods:**
- `__init__(text: str, config: TouchConfig)` - Initialize with text and configuration
- `set_minimum_size(size: int)` - Set minimum touch size
- `enable_long_press(enabled: bool)` - Enable long press detection
- `enable_double_tap(enabled: bool)` - Enable double tap detection
- `set_feedback(feedback_type: str, enabled: bool)` - Set feedback type
- `animate_touch()` - Animate touch feedback

##### `TouchSlider`
Enhanced slider with touch feedback.

**Methods:**
- `__init__(min_value: float, max_value: float, config: TouchConfig)` - Initialize slider
- `set_value(value: float)` - Set slider value
- `get_value()` - Get slider value
- `set_step(step: float)` - Set step size
- `enable_snap_to_step(enabled: bool)` - Enable snap to step

##### `TouchSwitch`
Touch-optimized switch control.

**Methods:**
- `__init__(config: TouchConfig)` - Initialize switch
- `set_active(active: bool)` - Set switch state
- `get_active()` - Get switch state
- `toggle()` - Toggle switch state
- `animate_toggle()` - Animate toggle

**Usage Example:**
```python
# Create touch configuration
config = TouchConfig(
    minimum_touch_size=44,
    tap_threshold=0.2,
    long_press_threshold=0.5,
    feedback_enabled=True,
    haptic_enabled=True
)

# Create touch button
button = TouchButton("Start Recording", config)

# Set up callbacks
def on_tap():
    print("Button tapped")

def on_long_press():
    print("Button long pressed")

button.bind(on_tap=on_tap)
button.bind(on_long_press=on_long_press)

# Create touch slider
slider = TouchSlider(0.0, 100.0, config)
slider.set_step(1.0)
slider.enable_snap_to_step(True)

def on_slider_change(value):
    print(f"Slider value: {value}")

slider.bind(on_value_change=on_slider_change)

# Create touch switch
switch = TouchSwitch(config)

def on_switch_change(active):
    print(f"Switch active: {active}")

switch.bind(on_active_change=on_switch_change)
```

### 16. `transcription_display.py`

**Purpose**: Real-time scrolling transcript display with speaker identification.

#### Classes

##### `TranscriptEntry`
Individual transcript entry.

**Attributes:**
- `text: str` - Transcript text
- `speaker: str` - Speaker name/ID
- `timestamp: datetime` - Entry timestamp
- `confidence: float` - Transcription confidence
- `start_time: float` - Audio start time
- `end_time: float` - Audio end time

##### `SpeakerInfo`
Speaker identification and statistics.

**Attributes:**
- `speaker_id: str` - Speaker identifier
- `name: str` - Speaker name
- `color: str` - Speaker color
- `speaking_time: float` - Total speaking time
- `word_count: int` - Word count
- `confidence: float` - Average confidence

##### `TranscriptionDisplay`
Main transcript display widget.

**Methods:**
- `__init__(config: dict = None)` - Initialize with configuration
- `add_transcript_entry(entry: TranscriptEntry)` - Add transcript entry
- `update_current_entry(text: str)` - Update current entry
- `set_auto_scroll(enabled: bool)` - Enable/disable auto-scroll
- `search_transcript(query: str)` - Search transcript
- `export_transcript(format: str)` - Export transcript
- `clear_transcript()` - Clear transcript display
- `get_speaker_stats()` - Get speaker statistics

**Usage Example:**
```python
# Create transcription display
config = {
    "auto_scroll": True,
    "show_timestamps": True,
    "show_confidence": True,
    "show_speaker_colors": True,
    "font_size": 16
}

transcript_display = TranscriptionDisplay(config)

# Add transcript entries
entry1 = TranscriptEntry(
    text="Hello everyone, welcome to the meeting.",
    speaker="Alice",
    timestamp=datetime.now(),
    confidence=0.95,
    start_time=0.0,
    end_time=3.2
)

entry2 = TranscriptEntry(
    text="Thank you Alice. Let's start with the agenda.",
    speaker="Bob",
    timestamp=datetime.now(),
    confidence=0.92,
    start_time=3.2,
    end_time=6.8
)

transcript_display.add_transcript_entry(entry1)
transcript_display.add_transcript_entry(entry2)

# Update current entry (for live transcription)
transcript_display.update_current_entry("This is the current...")

# Search transcript
results = transcript_display.search_transcript("meeting")
print(f"Found {len(results)} matches")

# Get speaker statistics
stats = transcript_display.get_speaker_stats()
for speaker, info in stats.items():
    print(f"{speaker}: {info.speaking_time:.1f}s, {info.word_count} words")
```

## Module Integration

The UI module integrates with other Silent Steno components:

1. **Core Events**: Subscribes to system events for real-time updates
2. **Recording Module**: Interfaces with session management
3. **AI Module**: Displays transcription and analysis results
4. **Audio Module**: Visualizes audio levels and playback
5. **Export Module**: Provides export functionality
6. **System Module**: Monitors device status and health

## Common Usage Patterns

### Complete UI Application Setup
```python
# Create main application
config = WindowConfig(
    title="Silent Steno",
    size=(800, 480),
    fullscreen=True
)

app = SilentStenoApp()
main_window = MainWindow(config)

# Set up theme system
theme_manager = ThemeManager(ThemeConfig(auto_switch=True))
theme_manager.set_theme("dark")

# Create navigation
nav_manager = NavigationManager(main_window.screen_manager)

# Set up screens
session_view = SessionView(SessionViewConfig())
session_list = SessionListView(SessionListConfig())
settings_view = SettingsView(SettingsConfig())

# Add screens to navigation
nav_manager.add_screen(session_view, "session")
nav_manager.add_screen(session_list, "sessions")
nav_manager.add_screen(settings_view, "settings")

# Set up feedback system
feedback_manager = FeedbackManager()

# Run application
app.run()
```

### Live Session Interface
```python
# Create live session interface
session_config = SessionViewConfig(
    show_timer=True,
    show_transcript=True,
    show_visualizer=True,
    show_controls=True
)

session_view = SessionView(session_config)

# Create components
audio_visualizer = create_bar_visualizer()
transcript_display = TranscriptionDisplay()
session_controls = SessionControls(ControlsConfig())

# Set up real-time updates
def on_audio_data(audio_level):
    audio_visualizer.update_audio_level(audio_level)

def on_transcript_update(text, speaker):
    entry = TranscriptEntry(
        text=text,
        speaker=speaker,
        timestamp=datetime.now(),
        confidence=0.9
    )
    transcript_display.add_transcript_entry(entry)

def on_session_control(action):
    if action == SessionAction.START:
        session_view.start_session("New Session")
    elif action == SessionAction.STOP:
        session_view.stop_session()

# Connect callbacks
session_controls.set_callback(SessionAction.START, on_session_control)
session_controls.set_callback(SessionAction.STOP, on_session_control)
```

### Touch-Optimized Interface
```python
# Create touch-optimized controls
touch_config = TouchConfig(
    minimum_touch_size=44,
    feedback_enabled=True,
    haptic_enabled=True
)

# Create touch buttons
start_button = TouchButton("Start", touch_config)
stop_button = TouchButton("Stop", touch_config)
settings_button = TouchButton("Settings", touch_config)

# Create touch slider for volume
volume_slider = TouchSlider(0.0, 1.0, touch_config)
volume_slider.set_step(0.1)

# Create touch switch for mute
mute_switch = TouchSwitch(touch_config)

# Set up feedback
feedback_manager = FeedbackManager({
    "visual_enabled": True,
    "audio_enabled": True,
    "haptic_enabled": True
})

# Handle touch events
def on_button_touch(widget):
    feedback_manager.provide_feedback(
        FeedbackType.VISUAL, 
        "button_tap", 
        widget
    )
    feedback_manager.provide_feedback(
        FeedbackType.HAPTIC, 
        "button_tap"
    )

start_button.bind(on_touch=on_button_touch)
```

### Export and Settings Integration
```python
# Create export dialog
export_dialog = ExportDialog(session_data)

# Configure export options
export_options = ExportOptions(
    format=ExportFormat.PDF,
    content_types=[ExportContent.TRANSCRIPT, ExportContent.ANALYSIS],
    include_timestamps=True
)

export_dialog.set_export_options(export_options)

# Create settings view
settings_view = SettingsView(SettingsConfig())

# Handle setting changes
def on_setting_changed(key, value):
    if key == "theme":
        theme_manager.set_theme(value)
    elif key == "audio.sample_rate":
        audio_system.set_sample_rate(value)

settings_view.bind(on_setting_changed=on_setting_changed)

# Import/export settings
settings_view.export_settings("backup.json")
```

This comprehensive documentation provides complete technical details and practical usage examples for all components in the UI module, enabling developers to create a fully functional touch-optimized interface for The Silent Steno system.