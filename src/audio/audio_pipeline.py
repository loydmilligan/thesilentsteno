#!/usr/bin/env python3

"""
Real-time Audio Pipeline for The Silent Steno

This module implements the core audio pipeline that captures audio from 
Bluetooth A2DP sources (phones), processes it in real-time, and forwards 
it to Bluetooth A2DP sinks (headphones) with minimal latency (<40ms target).

Key features:
- Real-time audio capture from Bluetooth sources
- Transparent audio forwarding to Bluetooth sinks
- Audio format conversion and optimization
- Latency measurement and optimization
- Audio level monitoring and visualization
- Pipeline health monitoring and recovery
"""

import threading
import time
import logging
import queue
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Audio pipeline states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    RECOVERING = "recovering"


class AudioFormat(Enum):
    """Supported audio formats"""
    PCM_16_44100 = "pcm_16_44100"
    PCM_16_48000 = "pcm_16_48000"
    PCM_24_44100 = "pcm_24_44100"
    PCM_24_48000 = "pcm_24_48000"


@dataclass
class AudioConfig:
    """Audio pipeline configuration"""
    sample_rate: int = 44100
    channels: int = 2
    format: AudioFormat = AudioFormat.PCM_16_44100
    buffer_size: int = 512  # frames
    target_latency_ms: float = 40.0
    enable_monitoring: bool = True
    enable_forwarding: bool = True
    auto_restart: bool = True


@dataclass
class PipelineMetrics:
    """Real-time pipeline performance metrics"""
    input_latency_ms: float = 0.0
    output_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    buffer_underruns: int = 0
    buffer_overruns: int = 0
    dropped_samples: int = 0
    cpu_usage: float = 0.0
    input_level_db: float = -60.0
    output_level_db: float = -60.0
    uptime_seconds: float = 0.0


class AudioPipeline:
    """
    Main audio pipeline for The Silent Steno
    
    Manages real-time audio capture, processing, and forwarding
    with comprehensive monitoring and optimization capabilities.
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize audio pipeline"""
        self.config = config or AudioConfig()
        self.state = PipelineState.STOPPED
        self.metrics = PipelineMetrics()
        
        # Threading
        self.pipeline_thread = None
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Audio buffers
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        
        # Callbacks
        self.state_callbacks: List[Callable] = []
        self.metrics_callbacks: List[Callable] = []
        self.audio_callbacks: List[Callable] = []
        
        # Components (lazy loaded)
        self._alsa_manager = None
        self._latency_optimizer = None
        self._format_converter = None
        self._level_monitor = None
        
        # Start time for uptime calculation
        self.start_time = None
        
        logger.info("Audio pipeline initialized")
    
    @property
    def alsa_manager(self):
        """Lazy load ALSA manager"""
        if self._alsa_manager is None:
            from .alsa_manager import ALSAManager
            self._alsa_manager = ALSAManager()
        return self._alsa_manager
    
    @property
    def latency_optimizer(self):
        """Lazy load latency optimizer"""
        if self._latency_optimizer is None:
            from .latency_optimizer import LatencyOptimizer
            self._latency_optimizer = LatencyOptimizer(self.config)
        return self._latency_optimizer
    
    @property
    def format_converter(self):
        """Lazy load format converter"""
        if self._format_converter is None:
            from .format_converter import FormatConverter
            self._format_converter = FormatConverter()
        return self._format_converter
    
    @property
    def level_monitor(self):
        """Lazy load level monitor"""
        if self._level_monitor is None:
            from .level_monitor import LevelMonitor
            self._level_monitor = LevelMonitor()
        return self._level_monitor
    
    def add_state_callback(self, callback: Callable[[PipelineState], None]) -> None:
        """Add callback for pipeline state changes"""
        self.state_callbacks.append(callback)
    
    def add_metrics_callback(self, callback: Callable[[PipelineMetrics], None]) -> None:
        """Add callback for metrics updates"""
        self.metrics_callbacks.append(callback)
    
    def add_audio_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Add callback for audio data processing"""
        self.audio_callbacks.append(callback)
    
    def _notify_state_change(self, new_state: PipelineState) -> None:
        """Notify callbacks of state changes"""
        old_state = self.state
        self.state = new_state
        logger.info(f"Pipeline state: {old_state.value} → {new_state.value}")
        
        for callback in self.state_callbacks:
            try:
                callback(new_state)
            except Exception as e:
                logger.error(f"Error in state callback: {e}")
    
    def _notify_metrics_update(self) -> None:
        """Notify callbacks of metrics updates"""
        for callback in self.metrics_callbacks:
            try:
                callback(self.metrics)
            except Exception as e:
                logger.error(f"Error in metrics callback: {e}")
    
    def _notify_audio_data(self, audio_data: np.ndarray) -> None:
        """Notify callbacks of audio data"""
        for callback in self.audio_callbacks:
            try:
                callback(audio_data)
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")
    
    def start_pipeline(self) -> bool:
        """
        Start the audio pipeline
        
        Returns:
            bool: True if pipeline started successfully
        """
        if self.state != PipelineState.STOPPED:
            logger.warning(f"Cannot start pipeline in state: {self.state.value}")
            return False
        
        try:
            logger.info("Starting audio pipeline...")
            self._notify_state_change(PipelineState.STARTING)
            
            # Reset stop event
            self.stop_event.clear()
            
            # Initialize ALSA configuration
            if not self.alsa_manager.configure_alsa():
                logger.error("Failed to configure ALSA")
                self._notify_state_change(PipelineState.ERROR)
                return False
            
            # Optimize latency settings
            if not self.latency_optimizer.optimize_buffers():
                logger.warning("Failed to optimize latency settings")
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="AudioMonitoring"
            )
            self.monitoring_thread.start()
            
            # Start main pipeline thread
            self.pipeline_thread = threading.Thread(
                target=self._pipeline_loop,
                daemon=True,
                name="AudioPipeline"
            )
            self.pipeline_thread.start()
            
            # Record start time
            self.start_time = time.time()
            
            # Wait for pipeline to stabilize
            time.sleep(0.5)
            
            if self.state == PipelineState.RUNNING:
                logger.info("Audio pipeline started successfully")
                return True
            else:
                logger.error("Audio pipeline failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Error starting audio pipeline: {e}")
            self._notify_state_change(PipelineState.ERROR)
            return False
    
    def stop_pipeline(self) -> bool:
        """
        Stop the audio pipeline
        
        Returns:
            bool: True if pipeline stopped successfully
        """
        if self.state == PipelineState.STOPPED:
            return True
        
        try:
            logger.info("Stopping audio pipeline...")
            self._notify_state_change(PipelineState.STOPPING)
            
            # Signal threads to stop
            self.stop_event.set()
            
            # Wait for threads to finish
            if self.pipeline_thread and self.pipeline_thread.is_alive():
                self.pipeline_thread.join(timeout=5.0)
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=2.0)
            
            # Clear queues
            while not self.input_queue.empty():
                try:
                    self.input_queue.get_nowait()
                except queue.Empty:
                    break
            
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    break
            
            self._notify_state_change(PipelineState.STOPPED)
            logger.info("Audio pipeline stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping audio pipeline: {e}")
            self._notify_state_change(PipelineState.ERROR)
            return False
    
    def restart_pipeline(self) -> bool:
        """
        Restart the audio pipeline
        
        Returns:
            bool: True if pipeline restarted successfully
        """
        logger.info("Restarting audio pipeline...")
        if self.stop_pipeline():
            time.sleep(1.0)
            return self.start_pipeline()
        return False
    
    def _pipeline_loop(self) -> None:
        """Main pipeline processing loop"""
        try:
            self._notify_state_change(PipelineState.RUNNING)
            logger.info("Audio pipeline loop started")
            
            frame_count = 0
            last_metrics_update = time.time()
            
            while not self.stop_event.is_set():
                try:
                    # Simulate audio processing
                    # In real implementation, this would:
                    # 1. Capture audio from Bluetooth A2DP source
                    # 2. Process audio (format conversion, filtering)
                    # 3. Forward audio to Bluetooth A2DP sink
                    # 4. Monitor latency and levels
                    
                    start_time = time.time()
                    
                    # Generate dummy audio data for testing
                    dummy_audio = np.zeros((self.config.buffer_size, self.config.channels), dtype=np.float32)
                    
                    # Simulate processing delay
                    time.sleep(self.config.buffer_size / self.config.sample_rate)
                    
                    # Update metrics
                    processing_time_ms = (time.time() - start_time) * 1000
                    self.metrics.total_latency_ms = processing_time_ms
                    
                    # Notify audio callbacks
                    self._notify_audio_data(dummy_audio)
                    
                    frame_count += 1
                    
                    # Update metrics periodically
                    if time.time() - last_metrics_update >= 1.0:
                        self._update_metrics()
                        self._notify_metrics_update()
                        last_metrics_update = time.time()
                
                except Exception as e:
                    logger.error(f"Error in pipeline loop: {e}")
                    if self.config.auto_restart:
                        self._notify_state_change(PipelineState.RECOVERING)
                        time.sleep(1.0)
                        continue
                    else:
                        break
            
            logger.info("Audio pipeline loop ended")
            
        except Exception as e:
            logger.error(f"Fatal error in pipeline loop: {e}")
            self._notify_state_change(PipelineState.ERROR)
    
    def _monitoring_loop(self) -> None:
        """Pipeline monitoring and health check loop"""
        try:
            logger.info("Audio monitoring loop started")
            
            while not self.stop_event.is_set():
                try:
                    # Monitor system resources
                    self._monitor_system_health()
                    
                    # Check for audio device availability
                    self._check_audio_devices()
                    
                    # Monitor latency performance
                    if self.state == PipelineState.RUNNING:
                        latency = self.latency_optimizer.measure_latency()
                        if latency:
                            self.metrics.total_latency_ms = latency
                    
                    time.sleep(5.0)  # Monitor every 5 seconds
                
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(1.0)
            
            logger.info("Audio monitoring loop ended")
            
        except Exception as e:
            logger.error(f"Fatal error in monitoring loop: {e}")
    
    def _update_metrics(self) -> None:
        """Update pipeline metrics"""
        if self.start_time:
            self.metrics.uptime_seconds = time.time() - self.start_time
        
        # Update audio levels if level monitor is available
        if self._level_monitor:
            levels = self.level_monitor.get_audio_levels()
            if levels:
                self.metrics.input_level_db = levels.get('input_db', -60.0)
                self.metrics.output_level_db = levels.get('output_db', -60.0)
    
    def _monitor_system_health(self) -> None:
        """Monitor system health and performance"""
        try:
            # Check CPU usage
            result = subprocess.run(
                ["top", "-bn1", "-p", str(os.getpid())],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse CPU usage from top output
                lines = result.stdout.split('\n')
                for line in lines:
                    if str(os.getpid()) in line:
                        parts = line.split()
                        if len(parts) > 8:
                            try:
                                self.metrics.cpu_usage = float(parts[8])
                            except (ValueError, IndexError):
                                pass
                        break
        
        except Exception as e:
            logger.debug(f"Could not monitor system health: {e}")
    
    def _check_audio_devices(self) -> None:
        """Check audio device availability"""
        try:
            devices = self.alsa_manager.get_audio_devices()
            if not devices:
                logger.warning("No audio devices detected")
        except Exception as e:
            logger.debug(f"Could not check audio devices: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline status
        
        Returns:
            Dict containing current pipeline state and metrics
        """
        return {
            "state": self.state.value,
            "config": {
                "sample_rate": self.config.sample_rate,
                "channels": self.config.channels,
                "buffer_size": self.config.buffer_size,
                "target_latency_ms": self.config.target_latency_ms
            },
            "metrics": {
                "total_latency_ms": self.metrics.total_latency_ms,
                "input_latency_ms": self.metrics.input_latency_ms,
                "output_latency_ms": self.metrics.output_latency_ms,
                "buffer_underruns": self.metrics.buffer_underruns,
                "buffer_overruns": self.metrics.buffer_overruns,
                "dropped_samples": self.metrics.dropped_samples,
                "cpu_usage": self.metrics.cpu_usage,
                "input_level_db": self.metrics.input_level_db,
                "output_level_db": self.metrics.output_level_db,
                "uptime_seconds": self.metrics.uptime_seconds
            },
            "health": {
                "latency_target_met": self.metrics.total_latency_ms <= self.config.target_latency_ms,
                "no_dropouts": self.metrics.buffer_underruns == 0 and self.metrics.buffer_overruns == 0,
                "audio_present": self.metrics.input_level_db > -50.0,
                "cpu_ok": self.metrics.cpu_usage < 80.0
            }
        }


# Global pipeline instance
_pipeline_instance = None


def start_pipeline(config: Optional[AudioConfig] = None) -> bool:
    """Start audio pipeline - convenience function"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = AudioPipeline(config)
    return _pipeline_instance.start_pipeline()


def stop_pipeline() -> bool:
    """Stop audio pipeline - convenience function"""
    global _pipeline_instance
    if _pipeline_instance:
        return _pipeline_instance.stop_pipeline()
    return True


def get_pipeline_status() -> Dict[str, Any]:
    """Get pipeline status - convenience function"""
    global _pipeline_instance
    if _pipeline_instance:
        return _pipeline_instance.get_pipeline_status()
    return {"state": "stopped", "metrics": {}, "health": {}}


def get_pipeline_instance() -> Optional[AudioPipeline]:
    """Get global pipeline instance"""
    return _pipeline_instance


if __name__ == "__main__":
    # Basic test when run directly
    import os
    
    print("Audio Pipeline Test")
    print("=" * 50)
    
    config = AudioConfig(
        sample_rate=44100,
        buffer_size=512,
        target_latency_ms=40.0
    )
    
    pipeline = AudioPipeline(config)
    
    def on_state_change(state):
        print(f"Pipeline state: {state.value}")
    
    def on_metrics_update(metrics):
        print(f"Latency: {metrics.total_latency_ms:.1f}ms, "
              f"CPU: {metrics.cpu_usage:.1f}%, "
              f"Uptime: {metrics.uptime_seconds:.1f}s")
    
    pipeline.add_state_callback(on_state_change)
    pipeline.add_metrics_callback(on_metrics_update)
    
    print("Starting pipeline...")
    if pipeline.start_pipeline():
        print("Pipeline started successfully")
        
        # Run for 10 seconds
        time.sleep(10)
        
        status = pipeline.get_pipeline_status()
        print(f"\nFinal status: {status}")
        
        print("Stopping pipeline...")
        pipeline.stop_pipeline()
    else:
        print("Failed to start pipeline")