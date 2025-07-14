#!/usr/bin/env python3

"""
Real-time Audio Level Monitor for The Silent Steno

This module provides comprehensive real-time audio level monitoring
and visualization capabilities for the audio pipeline. It tracks
audio levels, detects clipping, measures signal-to-noise ratio,
and provides visual feedback for audio quality assessment.

Key features:
- Real-time audio level measurement (dB SPL, RMS, peak)
- Clipping detection and prevention
- Signal-to-noise ratio monitoring
- Audio spectrum analysis
- Visual level meters and waveform display
- Audio quality metrics and alerting
"""

import numpy as np
import logging
import threading
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import queue

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LevelScale(Enum):
    """Audio level measurement scales"""
    DB_FS = "dbfs"  # Decibels relative to full scale
    DB_SPL = "dbspl"  # Decibels sound pressure level
    RMS = "rms"  # Root mean square
    PEAK = "peak"  # Peak amplitude
    VU = "vu"  # VU meter scale


class AlertType(Enum):
    """Audio monitoring alert types"""
    CLIPPING = "clipping"
    LOW_SIGNAL = "low_signal"
    HIGH_NOISE = "high_noise"
    DROPOUTS = "dropouts"
    DC_OFFSET = "dc_offset"


@dataclass
class AudioLevels:
    """Audio level measurement data"""
    timestamp: float
    input_peak_db: float
    input_rms_db: float
    output_peak_db: float
    output_rms_db: float
    signal_to_noise_db: float
    dynamic_range_db: float
    channel_levels: List[float]  # Per-channel levels
    is_clipping: bool = False
    dc_offset: float = 0.0


@dataclass
class AudioAlert:
    """Audio monitoring alert"""
    alert_type: AlertType
    severity: str  # "low", "medium", "high", "critical"
    message: str
    timestamp: float
    channel: Optional[int] = None
    value: Optional[float] = None


@dataclass
class MonitorConfig:
    """Audio level monitor configuration"""
    update_interval: float = 0.1  # seconds
    history_length: int = 1000  # samples
    clipping_threshold_db: float = -1.0
    low_signal_threshold_db: float = -50.0
    noise_floor_db: float = -60.0
    dc_offset_threshold: float = 0.1
    enable_spectrum_analysis: bool = False
    enable_alerts: bool = True


class LevelMonitor:
    """
    Real-time Audio Level Monitor for The Silent Steno
    
    Provides comprehensive audio level monitoring with real-time
    visualization and quality assessment capabilities.
    """
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        """Initialize level monitor"""
        self.config = config or MonitorConfig()
        
        # Monitoring state
        self.is_monitoring = False
        self.current_levels: Optional[AudioLevels] = None
        self.level_history: List[AudioLevels] = []
        self.active_alerts: List[AudioAlert] = []
        
        # Threading
        self.monitor_thread = None
        self.stop_event = threading.Event()
        
        # Audio data queue
        self.audio_queue = queue.Queue(maxsize=100)
        
        # Callbacks
        self.level_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # Statistics
        self.statistics = {
            "samples_processed": 0,
            "clipping_events": 0,
            "alert_count": 0,
            "avg_input_level_db": -60.0,
            "avg_output_level_db": -60.0,
            "max_peak_db": -60.0,
            "monitoring_uptime": 0.0
        }
        
        # Start time for uptime calculation
        self.start_time = None
        
        logger.info("Audio level monitor initialized")
    
    def add_level_callback(self, callback: Callable[[AudioLevels], None]) -> None:
        """Add callback for level updates"""
        self.level_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[AudioAlert], None]) -> None:
        """Add callback for audio alerts"""
        self.alert_callbacks.append(callback)
    
    def _notify_level_update(self, levels: AudioLevels) -> None:
        """Notify callbacks of level updates"""
        for callback in self.level_callbacks:
            try:
                callback(levels)
            except Exception as e:
                logger.error(f"Error in level callback: {e}")
    
    def _notify_alert(self, alert: AudioAlert) -> None:
        """Notify callbacks of audio alerts"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def start_monitoring(self) -> bool:
        """
        Start audio level monitoring
        
        Returns:
            bool: True if monitoring started successfully
        """
        if self.is_monitoring:
            return True
        
        try:
            logger.info("Starting audio level monitoring...")
            
            self.is_monitoring = True
            self.stop_event.clear()
            self.start_time = time.time()
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="AudioLevelMonitoring"
            )
            self.monitor_thread.start()
            
            logger.info("Audio level monitoring started")
            return True
        
        except Exception as e:
            logger.error(f"Error starting level monitoring: {e}")
            self.is_monitoring = False
            return False
    
    def stop_monitoring(self) -> bool:
        """
        Stop audio level monitoring
        
        Returns:
            bool: True if monitoring stopped successfully
        """
        if not self.is_monitoring:
            return True
        
        try:
            logger.info("Stopping audio level monitoring...")
            
            self.is_monitoring = False
            self.stop_event.set()
            
            # Wait for thread to finish
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5.0)
            
            # Clear audio queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            logger.info("Audio level monitoring stopped")
            return True
        
        except Exception as e:
            logger.error(f"Error stopping level monitoring: {e}")
            return False
    
    def process_audio(self, audio_data: np.ndarray, is_input: bool = True) -> None:
        """
        Process audio data for level monitoring
        
        Args:
            audio_data: Audio data as numpy array
            is_input: True for input audio, False for output
        """
        try:
            if not self.is_monitoring:
                return
            
            # Add to processing queue
            audio_info = {
                "data": audio_data,
                "is_input": is_input,
                "timestamp": time.time()
            }
            
            if not self.audio_queue.full():
                self.audio_queue.put_nowait(audio_info)
            else:
                # Queue is full, discard oldest sample
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_info)
                except queue.Empty:
                    pass
        
        except Exception as e:
            logger.debug(f"Error processing audio for monitoring: {e}")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        try:
            logger.info("Audio monitoring loop started")
            
            input_buffer = []
            output_buffer = []
            last_update = time.time()
            
            while not self.stop_event.is_set():
                try:
                    # Process queued audio data
                    while not self.audio_queue.empty():
                        try:
                            audio_info = self.audio_queue.get_nowait()
                            
                            if audio_info["is_input"]:
                                input_buffer.append(audio_info)
                            else:
                                output_buffer.append(audio_info)
                            
                            # Limit buffer size
                            if len(input_buffer) > 10:
                                input_buffer = input_buffer[-10:]
                            if len(output_buffer) > 10:
                                output_buffer = output_buffer[-10:]
                        
                        except queue.Empty:
                            break
                    
                    # Update levels at configured interval
                    if time.time() - last_update >= self.config.update_interval:
                        if input_buffer or output_buffer:
                            levels = self._calculate_levels(input_buffer, output_buffer)
                            if levels:
                                self._update_levels(levels)
                                last_update = time.time()
                    
                    # Update statistics
                    self._update_statistics()
                    
                    # Short sleep to prevent excessive CPU usage
                    time.sleep(0.01)
                
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(0.1)
            
            logger.info("Audio monitoring loop ended")
        
        except Exception as e:
            logger.error(f"Fatal error in monitoring loop: {e}")
    
    def _calculate_levels(self, input_buffer: List[Dict], output_buffer: List[Dict]) -> Optional[AudioLevels]:
        """Calculate audio levels from buffered data"""
        try:
            timestamp = time.time()
            
            # Process input audio
            input_peak_db = -60.0
            input_rms_db = -60.0
            input_channels = []
            
            if input_buffer:
                input_data = np.concatenate([item["data"] for item in input_buffer])
                input_peak_db = self._calculate_peak_db(input_data)
                input_rms_db = self._calculate_rms_db(input_data)
                input_channels = self._calculate_channel_levels(input_data)
            
            # Process output audio
            output_peak_db = -60.0
            output_rms_db = -60.0
            
            if output_buffer:
                output_data = np.concatenate([item["data"] for item in output_buffer])
                output_peak_db = self._calculate_peak_db(output_data)
                output_rms_db = self._calculate_rms_db(output_data)
            
            # Calculate derived metrics
            signal_to_noise_db = max(input_rms_db - self.config.noise_floor_db, 0.0)
            dynamic_range_db = input_peak_db - input_rms_db
            
            # Check for clipping
            is_clipping = (input_peak_db > self.config.clipping_threshold_db or 
                          output_peak_db > self.config.clipping_threshold_db)
            
            # Calculate DC offset
            dc_offset = 0.0
            if input_buffer:
                input_data = np.concatenate([item["data"] for item in input_buffer])
                dc_offset = np.mean(input_data)
            
            levels = AudioLevels(
                timestamp=timestamp,
                input_peak_db=input_peak_db,
                input_rms_db=input_rms_db,
                output_peak_db=output_peak_db,
                output_rms_db=output_rms_db,
                signal_to_noise_db=signal_to_noise_db,
                dynamic_range_db=dynamic_range_db,
                channel_levels=input_channels,
                is_clipping=is_clipping,
                dc_offset=dc_offset
            )
            
            return levels
        
        except Exception as e:
            logger.error(f"Error calculating levels: {e}")
            return None
    
    def _calculate_peak_db(self, audio_data: np.ndarray) -> float:
        """Calculate peak level in dB"""
        try:
            if len(audio_data) == 0:
                return -60.0
            
            peak = np.max(np.abs(audio_data))
            if peak > 0:
                return 20 * np.log10(peak)
            else:
                return -60.0
        
        except Exception:
            return -60.0
    
    def _calculate_rms_db(self, audio_data: np.ndarray) -> float:
        """Calculate RMS level in dB"""
        try:
            if len(audio_data) == 0:
                return -60.0
            
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms > 0:
                return 20 * np.log10(rms)
            else:
                return -60.0
        
        except Exception:
            return -60.0
    
    def _calculate_channel_levels(self, audio_data: np.ndarray) -> List[float]:
        """Calculate per-channel levels"""
        try:
            if len(audio_data.shape) == 1:
                # Mono audio
                return [self._calculate_rms_db(audio_data)]
            else:
                # Multi-channel audio
                levels = []
                for channel in range(audio_data.shape[1]):
                    level = self._calculate_rms_db(audio_data[:, channel])
                    levels.append(level)
                return levels
        
        except Exception:
            return []
    
    def _update_levels(self, levels: AudioLevels) -> None:
        """Update current levels and check for alerts"""
        self.current_levels = levels
        
        # Add to history
        self.level_history.append(levels)
        if len(self.level_history) > self.config.history_length:
            self.level_history = self.level_history[-self.config.history_length:]
        
        # Check for alerts
        if self.config.enable_alerts:
            self._check_alerts(levels)
        
        # Notify callbacks
        self._notify_level_update(levels)
        
        # Update statistics
        self.statistics["samples_processed"] += 1
        if levels.is_clipping:
            self.statistics["clipping_events"] += 1
    
    def _check_alerts(self, levels: AudioLevels) -> None:
        """Check for audio quality alerts"""
        try:
            # Clipping alert
            if levels.is_clipping:
                alert = AudioAlert(
                    alert_type=AlertType.CLIPPING,
                    severity="high",
                    message=f"Audio clipping detected (peak: {levels.input_peak_db:.1f}dB)",
                    timestamp=levels.timestamp,
                    value=levels.input_peak_db
                )
                self._add_alert(alert)
            
            # Low signal alert
            if levels.input_rms_db < self.config.low_signal_threshold_db:
                alert = AudioAlert(
                    alert_type=AlertType.LOW_SIGNAL,
                    severity="medium",
                    message=f"Low input signal detected (RMS: {levels.input_rms_db:.1f}dB)",
                    timestamp=levels.timestamp,
                    value=levels.input_rms_db
                )
                self._add_alert(alert)
            
            # DC offset alert
            if abs(levels.dc_offset) > self.config.dc_offset_threshold:
                alert = AudioAlert(
                    alert_type=AlertType.DC_OFFSET,
                    severity="low",
                    message=f"DC offset detected ({levels.dc_offset:.3f})",
                    timestamp=levels.timestamp,
                    value=levels.dc_offset
                )
                self._add_alert(alert)
        
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def _add_alert(self, alert: AudioAlert) -> None:
        """Add alert to active alerts list"""
        self.active_alerts.append(alert)
        
        # Limit active alerts
        if len(self.active_alerts) > 100:
            self.active_alerts = self.active_alerts[-100:]
        
        self.statistics["alert_count"] += 1
        self._notify_alert(alert)
        
        logger.warning(f"Audio alert: {alert.message}")
    
    def _update_statistics(self) -> None:
        """Update monitoring statistics"""
        if self.start_time:
            self.statistics["monitoring_uptime"] = time.time() - self.start_time
        
        if self.level_history:
            recent_levels = self.level_history[-10:]  # Last 10 measurements
            
            self.statistics["avg_input_level_db"] = np.mean(
                [l.input_rms_db for l in recent_levels]
            )
            self.statistics["avg_output_level_db"] = np.mean(
                [l.output_rms_db for l in recent_levels]
            )
            self.statistics["max_peak_db"] = max(
                [l.input_peak_db for l in recent_levels]
            )
    
    def get_audio_levels(self) -> Optional[Dict[str, float]]:
        """
        Get current audio levels
        
        Returns:
            Dict containing current audio levels, or None if not available
        """
        if self.current_levels:
            return {
                "input_peak_db": self.current_levels.input_peak_db,
                "input_rms_db": self.current_levels.input_rms_db,
                "output_peak_db": self.current_levels.output_peak_db,
                "output_rms_db": self.current_levels.output_rms_db,
                "signal_to_noise_db": self.current_levels.signal_to_noise_db,
                "is_clipping": self.current_levels.is_clipping
            }
        return None
    
    def monitor_clipping(self) -> bool:
        """
        Check if audio clipping is currently detected
        
        Returns:
            bool: True if clipping is detected
        """
        return self.current_levels.is_clipping if self.current_levels else False
    
    def visualize_levels(self) -> Dict[str, Any]:
        """
        Get visualization data for audio levels
        
        Returns:
            Dict containing data for level visualization
        """
        if not self.current_levels:
            return {}
        
        # Create level meter data
        input_meter = max(0, min(100, (self.current_levels.input_rms_db + 60) * 100 / 60))
        output_meter = max(0, min(100, (self.current_levels.output_rms_db + 60) * 100 / 60))
        
        return {
            "input_meter_percent": input_meter,
            "output_meter_percent": output_meter,
            "channel_meters": [
                max(0, min(100, (level + 60) * 100 / 60)) 
                for level in self.current_levels.channel_levels
            ],
            "clipping_indicator": self.current_levels.is_clipping,
            "signal_quality": "good" if self.current_levels.signal_to_noise_db > 20 else "poor",
            "level_history": [
                {
                    "timestamp": level.timestamp,
                    "input_rms": level.input_rms_db,
                    "output_rms": level.output_rms_db
                }
                for level in self.level_history[-50:]  # Last 50 samples
            ]
        }
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring status
        
        Returns:
            Dict containing monitoring state and statistics
        """
        return {
            "is_monitoring": self.is_monitoring,
            "current_levels": self.get_audio_levels(),
            "statistics": self.statistics.copy(),
            "active_alerts": len(self.active_alerts),
            "recent_alerts": [
                {
                    "type": alert.alert_type.value,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp
                }
                for alert in self.active_alerts[-5:]  # Last 5 alerts
            ],
            "configuration": {
                "update_interval": self.config.update_interval,
                "clipping_threshold": self.config.clipping_threshold_db,
                "low_signal_threshold": self.config.low_signal_threshold_db,
                "alerts_enabled": self.config.enable_alerts
            }
        }


if __name__ == "__main__":
    # Basic test when run directly
    print("Level Monitor Test")
    print("=" * 50)
    
    monitor = LevelMonitor()
    
    def on_level_update(levels):
        print(f"Levels: Input {levels.input_rms_db:.1f}dB, Output {levels.output_rms_db:.1f}dB, Clipping: {levels.is_clipping}")
    
    def on_alert(alert):
        print(f"ALERT [{alert.severity}]: {alert.message}")
    
    monitor.add_level_callback(on_level_update)
    monitor.add_alert_callback(on_alert)
    
    print("Starting monitoring...")
    if monitor.start_monitoring():
        print("Monitoring started")
        
        # Simulate audio data
        for i in range(10):
            # Generate test audio
            test_audio = np.random.rand(512, 2) * 0.5
            monitor.process_audio(test_audio, is_input=True)
            
            # Simulate clipping occasionally
            if i == 5:
                clipping_audio = np.ones((512, 2)) * 1.2
                monitor.process_audio(clipping_audio, is_input=True)
            
            time.sleep(0.2)
        
        time.sleep(1)
        
        status = monitor.get_monitoring_status()
        print(f"\nFinal status: {status}")
        
        monitor.stop_monitoring()
        print("Monitoring stopped")
    else:
        print("Failed to start monitoring")
