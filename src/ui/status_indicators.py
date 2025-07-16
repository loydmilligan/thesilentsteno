"""
Connection and system status indicators for The Silent Steno.

This module provides visual indicators for Bluetooth connection, recording status,
and system health monitoring with real-time updates and visual alerts.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Callable, Tuple
import logging
from datetime import datetime, timedelta

from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.properties import StringProperty, BooleanProperty, NumericProperty, ListProperty
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.graphics import Color, Ellipse, Rectangle, Line
from kivy.animation import Animation

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Bluetooth connection status."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    PAIRING = auto()
    ERROR = auto()
    WEAK_SIGNAL = auto()


class SystemStatus(Enum):
    """System status indicators."""
    NORMAL = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
    UNKNOWN = auto()


class StatusLevel(Enum):
    """Status indicator levels."""
    INFO = auto()
    SUCCESS = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class StatusInfo:
    """Status information."""
    level: StatusLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    auto_clear: bool = True
    clear_delay: float = 5.0


@dataclass
class IndicatorConfig:
    """Configuration for status indicators."""
    show_bluetooth: bool = True
    show_recording: bool = True
    show_battery: bool = True
    show_storage: bool = True
    show_network: bool = False
    show_system_health: bool = True
    update_interval: float = 1.0
    enable_animations: bool = True
    compact_mode: bool = False
    icon_size: int = 24
    show_labels: bool = True
    show_values: bool = True
    blink_on_alerts: bool = True
    colors: Dict[str, Tuple[float, float, float, float]] = field(default_factory=lambda: {
        'normal': (0.2, 1.0, 0.2, 1.0),      # Green
        'warning': (1.0, 0.8, 0.2, 1.0),     # Orange
        'error': (1.0, 0.2, 0.2, 1.0),       # Red
        'critical': (1.0, 0.0, 1.0, 1.0),    # Magenta
        'info': (0.2, 0.7, 1.0, 1.0),        # Blue
        'disabled': (0.5, 0.5, 0.5, 0.5),    # Gray
    })


class StatusIndicatorWidget(Widget):
    """Individual status indicator widget."""
    
    status_text = StringProperty('OK')
    status_color = ListProperty([0.2, 1.0, 0.2, 1.0])
    is_active = BooleanProperty(True)
    
    def __init__(self, name: str, config: IndicatorConfig, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.config = config
        self.status_level = StatusLevel.INFO
        self.value = 0.0
        self.max_value = 100.0
        self.unit = ""
        self.blink_animation = None
        
        self.size_hint = (None, None)
        self.size = (dp(config.icon_size * 3), dp(config.icon_size))
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up indicator UI."""
        layout = BoxLayout(orientation='horizontal', spacing=dp(5))
        
        # Status indicator (circle)
        self.indicator_widget = Widget(size_hint=(None, 1), width=dp(self.config.icon_size))
        layout.add_widget(self.indicator_widget)
        
        # Label and value
        if self.config.show_labels or self.config.show_values:
            text_layout = BoxLayout(orientation='vertical')
            
            if self.config.show_labels:
                self.label = Label(
                    text=self.name,
                    font_size=f'{self.config.icon_size * 0.6}sp',
                    size_hint_y=0.6,
                    halign='left',
                    valign='bottom'
                )
                self.label.bind(size=self.label.setter('text_size'))
                text_layout.add_widget(self.label)
                
            if self.config.show_values:
                self.value_label = Label(
                    text=self.status_text,
                    font_size=f'{self.config.icon_size * 0.5}sp',
                    size_hint_y=0.4,
                    halign='left',
                    valign='top',
                    color=(0.8, 0.8, 0.8, 1)
                )
                self.value_label.bind(size=self.value_label.setter('text_size'))
                text_layout.add_widget(self.value_label)
                
            layout.add_widget(text_layout)
            
        self.add_widget(layout)
        self._update_indicator()
        
    def _update_indicator(self):
        """Update visual indicator."""
        self.indicator_widget.canvas.clear()
        
        with self.indicator_widget.canvas:
            # Background circle
            Color(0.2, 0.2, 0.2, 0.8)
            center_x = self.indicator_widget.center_x
            center_y = self.indicator_widget.center_y
            radius = dp(self.config.icon_size) / 2 - dp(2)
            
            Ellipse(
                pos=(center_x - radius, center_y - radius),
                size=(radius * 2, radius * 2)
            )
            
            # Status color
            if self.is_active:
                Color(*self.status_color)
                inner_radius = radius - dp(2)
                Ellipse(
                    pos=(center_x - inner_radius, center_y - inner_radius),
                    size=(inner_radius * 2, inner_radius * 2)
                )
                
                # Progress indicator for values
                if self.max_value > 0 and self.value > 0:
                    progress = min(1.0, self.value / self.max_value)
                    # Draw arc based on progress (simplified as filled circle)
                    Color(*self.status_color, 0.3)
                    progress_radius = inner_radius * progress
                    Ellipse(
                        pos=(center_x - progress_radius, center_y - progress_radius),
                        size=(progress_radius * 2, progress_radius * 2)
                    )
            else:
                # Disabled state
                Color(*self.config.colors['disabled'])
                Line(
                    circle=(center_x, center_y, radius - dp(1)),
                    width=dp(2)
                )
                
    def update_status(self, level: StatusLevel, message: str = "", value: float = 0.0, 
                     max_value: float = 100.0, unit: str = ""):
        """Update status information."""
        self.status_level = level
        self.status_text = message
        self.value = value
        self.max_value = max_value
        self.unit = unit
        
        # Update color based on level
        if level == StatusLevel.SUCCESS:
            self.status_color = self.config.colors['normal']
        elif level == StatusLevel.WARNING:
            self.status_color = self.config.colors['warning']
        elif level == StatusLevel.ERROR:
            self.status_color = self.config.colors['error']
        elif level == StatusLevel.CRITICAL:
            self.status_color = self.config.colors['critical']
        else:  # INFO
            self.status_color = self.config.colors['info']
            
        # Update text displays
        if hasattr(self, 'value_label'):
            if value > 0 and unit:
                self.value_label.text = f"{value:.1f}{unit}"
            else:
                self.value_label.text = message
                
        # Start blinking for alerts
        if (level in [StatusLevel.WARNING, StatusLevel.ERROR, StatusLevel.CRITICAL] and 
            self.config.blink_on_alerts):
            self._start_blink()
        else:
            self._stop_blink()
            
        self._update_indicator()
        
    def _start_blink(self):
        """Start blink animation."""
        if self.blink_animation:
            self.blink_animation.cancel()
            
        if self.config.enable_animations:
            self.blink_animation = Animation(
                opacity=0.3, duration=0.5
            ) + Animation(
                opacity=1.0, duration=0.5
            )
            self.blink_animation.repeat = True
            self.blink_animation.start(self)
            
    def _stop_blink(self):
        """Stop blink animation."""
        if self.blink_animation:
            self.blink_animation.cancel()
            self.blink_animation = None
        self.opacity = 1.0
        
    def set_active(self, active: bool):
        """Set indicator active state."""
        self.is_active = active
        self._update_indicator()


class StatusIndicators(Widget):
    """Main status indicators widget."""
    
    # Kivy properties
    bluetooth_status = StringProperty('disconnected')
    recording_status = StringProperty('idle')
    battery_level = NumericProperty(100.0)
    storage_used = NumericProperty(0.0)
    system_health = StringProperty('normal')
    
    def __init__(self, config: IndicatorConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.indicators: Dict[str, StatusIndicatorWidget] = {}
        self.status_messages: List[StatusInfo] = []
        
        # System monitoring data
        self.bluetooth_connection = ConnectionStatus.DISCONNECTED
        self.recording_active = False
        self.battery_percentage = 100.0
        self.storage_percentage = 0.0
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.temperature = 0.0
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'on_bluetooth_change': [],
            'on_recording_change': [],
            'on_battery_low': [],
            'on_storage_full': [],
            'on_system_alert': []
        }
        
        self._setup_layout()
        self._create_indicators()
        self._start_monitoring()
        
    def _setup_layout(self):
        """Set up indicator layout."""
        if self.config.compact_mode:
            self.main_layout = BoxLayout(
                orientation='horizontal',
                spacing=dp(10),
                size_hint=(1, 1)
            )
        else:
            self.main_layout = GridLayout(
                cols=3 if self.config.show_network else 2,
                spacing=dp(15),
                size_hint=(1, 1)
            )
            
        self.add_widget(self.main_layout)
        
    def _create_indicators(self):
        """Create status indicator widgets."""
        # Bluetooth indicator
        if self.config.show_bluetooth:
            self.indicators['bluetooth'] = StatusIndicatorWidget(
                'Bluetooth', self.config
            )
            self.main_layout.add_widget(self.indicators['bluetooth'])
            
        # Recording indicator
        if self.config.show_recording:
            self.indicators['recording'] = StatusIndicatorWidget(
                'Recording', self.config
            )
            self.main_layout.add_widget(self.indicators['recording'])
            
        # Battery indicator
        if self.config.show_battery:
            self.indicators['battery'] = StatusIndicatorWidget(
                'Battery', self.config
            )
            self.main_layout.add_widget(self.indicators['battery'])
            
        # Storage indicator
        if self.config.show_storage:
            self.indicators['storage'] = StatusIndicatorWidget(
                'Storage', self.config
            )
            self.main_layout.add_widget(self.indicators['storage'])
            
        # Network indicator
        if self.config.show_network:
            self.indicators['network'] = StatusIndicatorWidget(
                'Network', self.config
            )
            self.main_layout.add_widget(self.indicators['network'])
            
        # System health indicator
        if self.config.show_system_health:
            self.indicators['system'] = StatusIndicatorWidget(
                'System', self.config
            )
            self.main_layout.add_widget(self.indicators['system'])
            
    def _start_monitoring(self):
        """Start system monitoring."""
        Clock.schedule_interval(self._update_indicators, self.config.update_interval)
        
    def _update_indicators(self, dt):
        """Update all indicators."""
        # Update Bluetooth
        if 'bluetooth' in self.indicators:
            self._update_bluetooth_indicator()
            
        # Update Recording
        if 'recording' in self.indicators:
            self._update_recording_indicator()
            
        # Update Battery
        if 'battery' in self.indicators:
            self._update_battery_indicator()
            
        # Update Storage
        if 'storage' in self.indicators:
            self._update_storage_indicator()
            
        # Update System
        if 'system' in self.indicators:
            self._update_system_indicator()
            
        # Clean up old status messages
        self._cleanup_status_messages()
        
    def _update_bluetooth_indicator(self):
        """Update Bluetooth status indicator."""
        indicator = self.indicators['bluetooth']
        
        if self.bluetooth_connection == ConnectionStatus.CONNECTED:
            indicator.update_status(StatusLevel.SUCCESS, "Connected")
        elif self.bluetooth_connection == ConnectionStatus.CONNECTING:
            indicator.update_status(StatusLevel.INFO, "Connecting")
        elif self.bluetooth_connection == ConnectionStatus.WEAK_SIGNAL:
            indicator.update_status(StatusLevel.WARNING, "Weak Signal")
        elif self.bluetooth_connection == ConnectionStatus.ERROR:
            indicator.update_status(StatusLevel.ERROR, "Error")
        else:
            indicator.update_status(StatusLevel.ERROR, "Disconnected")
            
        self.bluetooth_status = self.bluetooth_connection.name.lower()
        
    def _update_recording_indicator(self):
        """Update recording status indicator."""
        indicator = self.indicators['recording']
        
        if self.recording_active:
            indicator.update_status(StatusLevel.SUCCESS, "Recording")
        else:
            indicator.update_status(StatusLevel.INFO, "Idle")
            
        self.recording_status = "recording" if self.recording_active else "idle"
        
    def _update_battery_indicator(self):
        """Update battery status indicator."""
        indicator = self.indicators['battery']
        
        if self.battery_percentage > 20:
            level = StatusLevel.SUCCESS
        elif self.battery_percentage > 10:
            level = StatusLevel.WARNING
        else:
            level = StatusLevel.ERROR
            
        indicator.update_status(
            level, 
            f"{self.battery_percentage:.0f}%",
            self.battery_percentage,
            100.0,
            "%"
        )
        
        self.battery_level = self.battery_percentage
        
        # Battery low warning
        if self.battery_percentage <= 15:
            self._notify_battery_low()
            
    def _update_storage_indicator(self):
        """Update storage status indicator."""
        indicator = self.indicators['storage']
        
        if self.storage_percentage < 80:
            level = StatusLevel.SUCCESS
        elif self.storage_percentage < 90:
            level = StatusLevel.WARNING
        else:
            level = StatusLevel.ERROR
            
        indicator.update_status(
            level,
            f"{self.storage_percentage:.0f}%",
            self.storage_percentage,
            100.0,
            "%"
        )
        
        self.storage_used = self.storage_percentage
        
        # Storage full warning
        if self.storage_percentage >= 90:
            self._notify_storage_full()
            
    def _update_system_indicator(self):
        """Update system health indicator."""
        indicator = self.indicators['system']
        
        # Combine CPU, memory, and temperature for overall health
        max_usage = max(self.cpu_usage, self.memory_usage)
        
        if max_usage < 70 and self.temperature < 70:
            level = StatusLevel.SUCCESS
            message = "Normal"
        elif max_usage < 85 and self.temperature < 80:
            level = StatusLevel.WARNING
            message = "High Load"
        else:
            level = StatusLevel.ERROR
            message = "Overload"
            
        indicator.update_status(level, message)
        self.system_health = message.lower()
        
        # System alert
        if level == StatusLevel.ERROR:
            self._notify_system_alert(f"System {message}")
            
    def set_bluetooth_status(self, status: ConnectionStatus, details: Dict[str, Any] = None):
        """Set Bluetooth connection status."""
        self.bluetooth_connection = status
        self._notify_bluetooth_change(status, details or {})
        
    def set_recording_status(self, recording: bool):
        """Set recording status."""
        if recording != self.recording_active:
            self.recording_active = recording
            self._notify_recording_change(recording)
            
    def set_battery_level(self, percentage: float):
        """Set battery level."""
        self.battery_percentage = max(0.0, min(100.0, percentage))
        
    def set_storage_usage(self, percentage: float):
        """Set storage usage."""
        self.storage_percentage = max(0.0, min(100.0, percentage))
        
    def set_system_metrics(self, cpu: float, memory: float, temperature: float):
        """Set system performance metrics."""
        self.cpu_usage = cpu
        self.memory_usage = memory
        self.temperature = temperature
        
    def add_status_message(self, level: StatusLevel, message: str, 
                          details: Dict[str, Any] = None, auto_clear: bool = True):
        """Add status message."""
        status_info = StatusInfo(
            level=level,
            message=message,
            details=details or {},
            auto_clear=auto_clear
        )
        self.status_messages.append(status_info)
        
        # Log the message
        if level == StatusLevel.ERROR or level == StatusLevel.CRITICAL:
            logger.error(f"Status: {message}")
        elif level == StatusLevel.WARNING:
            logger.warning(f"Status: {message}")
        else:
            logger.info(f"Status: {message}")
            
    def _cleanup_status_messages(self):
        """Clean up old status messages."""
        now = datetime.now()
        self.status_messages = [
            msg for msg in self.status_messages
            if not msg.auto_clear or 
            (now - msg.timestamp).total_seconds() < msg.clear_delay
        ]
        
    def get_current_status(self) -> Dict[str, Any]:
        """Get current status summary."""
        return {
            'bluetooth': self.bluetooth_connection.name,
            'recording': self.recording_active,
            'battery': self.battery_percentage,
            'storage': self.storage_percentage,
            'cpu': self.cpu_usage,
            'memory': self.memory_usage,
            'temperature': self.temperature,
            'messages': len(self.status_messages)
        }
        
    def add_callback(self, event: str, callback: Callable):
        """Add event callback."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            
    def remove_callback(self, event: str, callback: Callable):
        """Remove event callback."""
        if event in self.callbacks and callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
            
    def _notify_bluetooth_change(self, status: ConnectionStatus, details: Dict[str, Any]):
        """Notify Bluetooth status change."""
        for callback in self.callbacks['on_bluetooth_change']:
            try:
                callback(status, details)
            except Exception as e:
                logger.error(f"Error in Bluetooth change callback: {e}")
                
    def _notify_recording_change(self, recording: bool):
        """Notify recording status change."""
        for callback in self.callbacks['on_recording_change']:
            try:
                callback(recording)
            except Exception as e:
                logger.error(f"Error in recording change callback: {e}")
                
    def _notify_battery_low(self):
        """Notify battery low."""
        for callback in self.callbacks['on_battery_low']:
            try:
                callback(self.battery_percentage)
            except Exception as e:
                logger.error(f"Error in battery low callback: {e}")
                
    def _notify_storage_full(self):
        """Notify storage full."""
        for callback in self.callbacks['on_storage_full']:
            try:
                callback(self.storage_percentage)
            except Exception as e:
                logger.error(f"Error in storage full callback: {e}")
                
    def _notify_system_alert(self, message: str):
        """Notify system alert."""
        for callback in self.callbacks['on_system_alert']:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Error in system alert callback: {e}")


# Factory functions
def create_status_indicators(config: Optional[IndicatorConfig] = None) -> StatusIndicators:
    """Create status indicators with configuration."""
    if config is None:
        config = IndicatorConfig()
    return StatusIndicators(config)


def create_minimal_indicators() -> StatusIndicators:
    """Create minimal status indicators."""
    config = IndicatorConfig(
        show_bluetooth=True,
        show_recording=True,
        show_battery=False,
        show_storage=False,
        show_network=False,
        show_system_health=False,
        compact_mode=True,
        show_labels=False,
        icon_size=16
    )
    return StatusIndicators(config)


def create_detailed_indicators() -> StatusIndicators:
    """Create detailed status indicators."""
    config = IndicatorConfig(
        show_bluetooth=True,
        show_recording=True,
        show_battery=True,
        show_storage=True,
        show_network=True,
        show_system_health=True,
        compact_mode=False,
        show_labels=True,
        show_values=True,
        enable_animations=True,
        blink_on_alerts=True
    )
    return StatusIndicators(config)


def create_dashboard_indicators() -> StatusIndicators:
    """Create dashboard-style status indicators."""
    config = IndicatorConfig(
        show_bluetooth=True,
        show_recording=True,
        show_battery=True,
        show_storage=True,
        show_system_health=True,
        compact_mode=False,
        icon_size=32,
        show_labels=True,
        show_values=True,
        enable_animations=True
    )
    return StatusIndicators(config)