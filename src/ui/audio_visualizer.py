"""
Real-time audio level visualization for The Silent Steno.

This module provides visual representation of audio levels with multiple
visualization modes for live audio monitoring and recording feedback.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Callable, Dict, Any, Tuple
import logging
import math
from collections import deque

from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle, Line, Ellipse
from kivy.clock import Clock
from kivy.properties import NumericProperty, ListProperty
from kivy.metrics import dp
from kivy.animation import Animation

logger = logging.getLogger(__name__)


class VisualizerType(Enum):
    """Audio visualizer types."""
    BARS = auto()
    WAVEFORM = auto()
    SPECTRUM = auto()
    CIRCULAR = auto()
    VU_METER = auto()


class VisualizerState(Enum):
    """Visualizer states."""
    IDLE = auto()
    ACTIVE = auto()
    RECORDING = auto()
    MUTED = auto()
    ERROR = auto()


@dataclass
class AudioLevel:
    """Audio level data."""
    timestamp: float
    levels: List[float]  # Multiple channels/bands
    peak: float = 0.0
    rms: float = 0.0
    frequency_bands: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizerConfig:
    """Configuration for audio visualizer."""
    visualizer_type: VisualizerType = VisualizerType.BARS
    update_rate: float = 60.0  # FPS
    num_bands: int = 8
    sensitivity: float = 1.0
    smoothing: float = 0.3
    decay_rate: float = 0.95
    peak_hold_time: float = 1.0
    color_scheme: str = "default"
    show_labels: bool = True
    show_peak_indicators: bool = True
    enable_animations: bool = True
    background_color: Tuple[float, float, float, float] = (0.1, 0.1, 0.1, 1.0)
    bar_colors: List[Tuple[float, float, float, float]] = field(default_factory=lambda: [
        (0.0, 1.0, 0.0, 1.0),  # Green (low)
        (0.5, 1.0, 0.0, 1.0),  # Yellow-green
        (1.0, 1.0, 0.0, 1.0),  # Yellow
        (1.0, 0.5, 0.0, 1.0),  # Orange
        (1.0, 0.0, 0.0, 1.0),  # Red (high)
    ])
    min_bar_height: float = 2.0
    max_history: int = 1000


class AudioVisualizer(Widget):
    """Real-time audio level visualizer."""
    
    # Kivy properties
    levels = ListProperty([0.0] * 8)
    peak_level = NumericProperty(0.0)
    is_recording = property(lambda self: self.state == VisualizerState.RECORDING)
    
    def __init__(self, config: VisualizerConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.state = VisualizerState.IDLE
        self.history: deque = deque(maxlen=config.max_history)
        self.peak_history: deque = deque(maxlen=config.max_history)
        self.smoothed_levels = [0.0] * config.num_bands
        self.peak_hold_levels = [0.0] * config.num_bands
        self.peak_hold_times = [0.0] * config.num_bands
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'on_level_change': [],
            'on_peak': [],
            'on_clipping': []
        }
        
        # Update timer
        self.update_event = None
        self._start_updates()
        
        # Bind size changes
        self.bind(size=self._redraw, pos=self._redraw)
        
        # Initial draw
        self._redraw()
        
    def _start_updates(self):
        """Start update timer."""
        if self.update_event:
            self.update_event.cancel()
        interval = 1.0 / self.config.update_rate
        self.update_event = Clock.schedule_interval(self._update_visualization, interval)
        
    def _stop_updates(self):
        """Stop update timer."""
        if self.update_event:
            self.update_event.cancel()
            self.update_event = None
            
    def update_levels(self, levels: List[float], timestamp: Optional[float] = None):
        """Update audio levels."""
        if timestamp is None:
            timestamp = Clock.get_time()
            
        # Ensure we have the right number of bands
        if len(levels) != self.config.num_bands:
            # Resample levels to match number of bands
            levels = self._resample_levels(levels, self.config.num_bands)
            
        # Apply sensitivity
        levels = [min(1.0, level * self.config.sensitivity) for level in levels]
        
        # Calculate peak and RMS
        peak = max(levels) if levels else 0.0
        rms = math.sqrt(sum(l * l for l in levels) / len(levels)) if levels else 0.0
        
        # Create audio level object
        audio_level = AudioLevel(
            timestamp=timestamp,
            levels=levels,
            peak=peak,
            rms=rms
        )
        
        # Add to history
        self.history.append(audio_level)
        self.peak_history.append(peak)
        
        # Update smoothed levels
        for i, level in enumerate(levels):
            if i < len(self.smoothed_levels):
                alpha = 1.0 - self.config.smoothing
                self.smoothed_levels[i] = (alpha * level + 
                                         self.config.smoothing * self.smoothed_levels[i])
                
                # Update peak hold
                if level > self.peak_hold_levels[i]:
                    self.peak_hold_levels[i] = level
                    self.peak_hold_times[i] = timestamp
                elif timestamp - self.peak_hold_times[i] > self.config.peak_hold_time:
                    self.peak_hold_levels[i] *= self.config.decay_rate
                    
        # Update properties
        self.levels = self.smoothed_levels[:]
        self.peak_level = peak
        
        # Check for clipping
        if peak > 0.95:
            self._notify_clipping(peak)
            
        # Notify callbacks
        self._notify_level_change(audio_level)
        
    def _resample_levels(self, levels: List[float], target_bands: int) -> List[float]:
        """Resample levels to target number of bands."""
        if len(levels) == target_bands:
            return levels
        elif len(levels) < target_bands:
            # Interpolate
            result = []
            for i in range(target_bands):
                idx = i * (len(levels) - 1) / (target_bands - 1)
                low_idx = int(idx)
                high_idx = min(low_idx + 1, len(levels) - 1)
                weight = idx - low_idx
                value = levels[low_idx] * (1 - weight) + levels[high_idx] * weight
                result.append(value)
            return result
        else:
            # Downsample
            result = []
            band_size = len(levels) / target_bands
            for i in range(target_bands):
                start_idx = int(i * band_size)
                end_idx = int((i + 1) * band_size)
                band_levels = levels[start_idx:end_idx]
                avg_level = sum(band_levels) / len(band_levels) if band_levels else 0.0
                result.append(avg_level)
            return result
            
    def _update_visualization(self, dt):
        """Update visualization display."""
        # Apply decay to smoothed levels
        for i in range(len(self.smoothed_levels)):
            self.smoothed_levels[i] *= self.config.decay_rate
            
        # Update peak holds
        current_time = Clock.get_time()
        for i in range(len(self.peak_hold_levels)):
            if current_time - self.peak_hold_times[i] > self.config.peak_hold_time:
                self.peak_hold_levels[i] *= self.config.decay_rate
                
        # Redraw if levels changed significantly
        self._redraw()
        
    def _redraw(self, *args):
        """Redraw the visualization."""
        self.canvas.clear()
        
        with self.canvas:
            # Background
            Color(*self.config.background_color)
            Rectangle(pos=self.pos, size=self.size)
            
            # Draw based on visualizer type
            if self.config.visualizer_type == VisualizerType.BARS:
                self._draw_bars()
            elif self.config.visualizer_type == VisualizerType.WAVEFORM:
                self._draw_waveform()
            elif self.config.visualizer_type == VisualizerType.SPECTRUM:
                self._draw_spectrum()
            elif self.config.visualizer_type == VisualizerType.CIRCULAR:
                self._draw_circular()
            elif self.config.visualizer_type == VisualizerType.VU_METER:
                self._draw_vu_meter()
                
    def _draw_bars(self):
        """Draw bar visualization."""
        if not self.smoothed_levels:
            return
            
        bar_width = self.width / len(self.smoothed_levels)
        bar_spacing = bar_width * 0.1
        actual_bar_width = bar_width - bar_spacing
        
        for i, level in enumerate(self.smoothed_levels):
            # Calculate bar height
            bar_height = max(self.config.min_bar_height, 
                           level * (self.height - dp(20)))
            
            # Calculate position
            x = self.x + i * bar_width + bar_spacing / 2
            y = self.y + dp(10)
            
            # Choose color based on level
            color = self._get_level_color(level)
            Color(*color)
            
            # Draw bar
            Rectangle(pos=(x, y), size=(actual_bar_width, bar_height))
            
            # Draw peak hold indicator
            if (self.config.show_peak_indicators and 
                i < len(self.peak_hold_levels) and 
                self.peak_hold_levels[i] > 0.1):
                
                peak_y = y + self.peak_hold_levels[i] * (self.height - dp(20))
                Color(1, 1, 1, 0.8)  # White peak indicator
                Rectangle(pos=(x, peak_y - 1), size=(actual_bar_width, 2))
                
    def _draw_waveform(self):
        """Draw waveform visualization."""
        if len(self.history) < 2:
            return
            
        # Draw recent waveform history
        points = []
        history_items = list(self.history)[-min(100, len(self.history)):]
        
        for i, audio_level in enumerate(history_items):
            if audio_level.levels:
                x = self.x + (i / len(history_items)) * self.width
                # Use RMS for smoother waveform
                y = self.center_y + (audio_level.rms - 0.5) * self.height * 0.8
                points.extend([x, y])
                
        if len(points) >= 4:
            Color(0, 1, 0, 0.8)  # Green waveform
            Line(points=points, width=2)
            
    def _draw_spectrum(self):
        """Draw spectrum analyzer visualization."""
        if not self.smoothed_levels:
            return
            
        # Similar to bars but with frequency band styling
        self._draw_bars()
        
        # Add frequency labels if enabled
        if self.config.show_labels:
            # Would add frequency labels here
            pass
            
    def _draw_circular(self):
        """Draw circular visualization."""
        if not self.smoothed_levels:
            return
            
        center_x = self.center_x
        center_y = self.center_y
        max_radius = min(self.width, self.height) / 2 - dp(10)
        
        # Draw concentric circles
        for i, level in enumerate(self.smoothed_levels):
            radius = (i + 1) * max_radius / len(self.smoothed_levels)
            
            # Background circle
            Color(0.2, 0.2, 0.2, 0.5)
            Line(circle=(center_x, center_y, radius), width=2)
            
            # Level arc
            if level > 0.01:
                color = self._get_level_color(level)
                Color(*color)
                
                # Calculate arc angle based on level
                angle_span = level * 360
                # Draw arc (simplified as ellipse for now)
                Color(*color, 0.6)
                Ellipse(pos=(center_x - radius, center_y - radius),
                       size=(radius * 2, radius * 2))
                       
    def _draw_vu_meter(self):
        """Draw VU meter style visualization."""
        if not self.smoothed_levels:
            return
            
        # Draw two large VU meters (left/right or overall)
        meter_width = self.width / 2 - dp(20)
        meter_height = self.height - dp(40)
        
        for i in range(2):
            level = self.smoothed_levels[i] if i < len(self.smoothed_levels) else 0.0
            
            x = self.x + dp(10) + i * (meter_width + dp(20))
            y = self.y + dp(20)
            
            # Background
            Color(0.1, 0.1, 0.1, 1)
            Rectangle(pos=(x, y), size=(meter_width, meter_height))
            
            # Level bar
            bar_height = level * meter_height
            color = self._get_level_color(level)
            Color(*color)
            Rectangle(pos=(x, y), size=(meter_width, bar_height))
            
            # Scale markings
            Color(0.5, 0.5, 0.5, 1)
            for j in range(11):  # 0-100% in 10% increments
                mark_y = y + (j / 10) * meter_height
                Line(points=[x + meter_width * 0.8, mark_y, 
                           x + meter_width, mark_y], width=1)
                           
    def _get_level_color(self, level: float) -> Tuple[float, float, float, float]:
        """Get color for audio level."""
        if level < 0.2:
            return self.config.bar_colors[0]  # Green
        elif level < 0.4:
            return self.config.bar_colors[1]  # Yellow-green
        elif level < 0.6:
            return self.config.bar_colors[2]  # Yellow
        elif level < 0.8:
            return self.config.bar_colors[3]  # Orange
        else:
            return self.config.bar_colors[4]  # Red
            
    def set_visualizer_type(self, viz_type: VisualizerType):
        """Change visualizer type."""
        self.config.visualizer_type = viz_type
        self._redraw()
        
    def set_recording_state(self, recording: bool):
        """Set recording state."""
        if recording:
            self.state = VisualizerState.RECORDING
        else:
            self.state = VisualizerState.ACTIVE
            
        # Update visual feedback
        self._update_recording_indicator()
        
    def _update_recording_indicator(self):
        """Update recording indicator."""
        if self.state == VisualizerState.RECORDING:
            # Add red recording indicator
            with self.canvas.after:
                Color(1, 0, 0, 0.8)
                Ellipse(pos=(self.right - dp(20), self.top - dp(20)), 
                       size=(dp(10), dp(10)))
        else:
            self.canvas.after.clear()
            
    def clear_history(self):
        """Clear audio level history."""
        self.history.clear()
        self.peak_history.clear()
        self.smoothed_levels = [0.0] * self.config.num_bands
        self.peak_hold_levels = [0.0] * self.config.num_bands
        self._redraw()
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get audio statistics."""
        if not self.history:
            return {'peak': 0.0, 'average': 0.0, 'samples': 0}
            
        peaks = [level.peak for level in self.history]
        return {
            'peak': max(peaks),
            'average': sum(peaks) / len(peaks),
            'samples': len(self.history),
            'last_update': self.history[-1].timestamp if self.history else 0
        }
        
    def add_callback(self, event: str, callback: Callable):
        """Add event callback."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            
    def remove_callback(self, event: str, callback: Callable):
        """Remove event callback."""
        if event in self.callbacks and callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
            
    def _notify_level_change(self, audio_level: AudioLevel):
        """Notify level change."""
        for callback in self.callbacks['on_level_change']:
            try:
                callback(audio_level)
            except Exception as e:
                logger.error(f"Error in level change callback: {e}")
                
    def _notify_clipping(self, level: float):
        """Notify audio clipping."""
        for callback in self.callbacks['on_clipping']:
            try:
                callback(level)
            except Exception as e:
                logger.error(f"Error in clipping callback: {e}")


# Factory functions
def create_audio_visualizer(viz_type: VisualizerType = VisualizerType.BARS,
                          config: Optional[VisualizerConfig] = None) -> AudioVisualizer:
    """Create audio visualizer with specified type."""
    if config is None:
        config = VisualizerConfig()
    config.visualizer_type = viz_type
    return AudioVisualizer(config)


def create_waveform_visualizer() -> AudioVisualizer:
    """Create waveform visualizer."""
    config = VisualizerConfig(
        visualizer_type=VisualizerType.WAVEFORM,
        update_rate=60.0,
        smoothing=0.1,
        enable_animations=True
    )
    return AudioVisualizer(config)


def create_spectrum_visualizer(num_bands: int = 16) -> AudioVisualizer:
    """Create spectrum analyzer visualizer."""
    config = VisualizerConfig(
        visualizer_type=VisualizerType.SPECTRUM,
        num_bands=num_bands,
        update_rate=60.0,
        show_labels=True,
        show_peak_indicators=True
    )
    return AudioVisualizer(config)


def create_vu_meter() -> AudioVisualizer:
    """Create VU meter visualizer."""
    config = VisualizerConfig(
        visualizer_type=VisualizerType.VU_METER,
        num_bands=2,
        update_rate=30.0,
        smoothing=0.5,
        show_labels=True
    )
    return AudioVisualizer(config)


def create_compact_visualizer() -> AudioVisualizer:
    """Create compact visualizer for small screens."""
    config = VisualizerConfig(
        visualizer_type=VisualizerType.BARS,
        num_bands=4,
        update_rate=30.0,
        show_labels=False,
        show_peak_indicators=False,
        min_bar_height=1.0
    )
    return AudioVisualizer(config)