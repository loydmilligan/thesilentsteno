#!/usr/bin/env python3
"""
Voice Activity Detection (VAD) System for The Silent Steno

This module provides real-time voice activity detection using WebRTC VAD
and custom algorithms for optimizing transcription processing.

Author: The Silent Steno Team
License: MIT
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import webrtcvad
from collections import deque

# Configure logging
logger = logging.getLogger(__name__)

class VADMode(Enum):
    """Voice Activity Detection sensitivity modes"""
    QUALITY = 0      # High quality (least aggressive)
    LOW_BITRATE = 1  # Low bitrate (more aggressive)
    AGGRESSIVE = 2   # Aggressive (more aggressive)
    VERY_AGGRESSIVE = 3  # Very aggressive (most aggressive)

class VADSensitivity(Enum):
    """VAD sensitivity levels for different scenarios"""
    VERY_LOW = "very_low"      # For very quiet environments
    LOW = "low"                # For normal environments  
    MEDIUM = "medium"          # Balanced sensitivity
    HIGH = "high"              # For noisy environments
    VERY_HIGH = "very_high"    # For very noisy environments

@dataclass
class VADConfig:
    """Configuration for Voice Activity Detector"""
    mode: VADMode = VADMode.AGGRESSIVE
    sensitivity: VADSensitivity = VADSensitivity.MEDIUM
    sample_rate: int = 16000
    frame_duration_ms: int = 30  # 10, 20, or 30 ms
    chunk_size: int = 480  # samples per frame (16000 * 0.03)
    smoothing_window: int = 10  # frames to smooth over
    voice_threshold: float = 0.6  # percentage of frames that must be voice
    silence_threshold: float = 0.3  # percentage below which is silence
    min_voice_duration_ms: int = 100  # minimum voice segment duration
    min_silence_duration_ms: int = 200  # minimum silence segment duration
    energy_threshold: float = 0.01  # minimum energy threshold
    zero_crossing_threshold: int = 50  # zero crossing rate threshold
    callback_interval_ms: int = 100  # callback frequency
    
    def __post_init__(self):
        """Validate and adjust configuration parameters"""
        # Validate frame duration
        if self.frame_duration_ms not in [10, 20, 30]:
            logger.warning(f"Invalid frame duration {self.frame_duration_ms}ms, using 30ms")
            self.frame_duration_ms = 30
        
        # Calculate chunk size based on sample rate and frame duration
        self.chunk_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        
        # Adjust thresholds based on sensitivity
        sensitivity_adjustments = {
            VADSensitivity.VERY_LOW: (0.8, 0.1),
            VADSensitivity.LOW: (0.7, 0.2), 
            VADSensitivity.MEDIUM: (0.6, 0.3),
            VADSensitivity.HIGH: (0.5, 0.4),
            VADSensitivity.VERY_HIGH: (0.4, 0.5)
        }
        
        if self.sensitivity in sensitivity_adjustments:
            voice_adj, silence_adj = sensitivity_adjustments[self.sensitivity]
            self.voice_threshold = voice_adj
            self.silence_threshold = silence_adj

@dataclass
class VADResult:
    """Result of voice activity detection"""
    timestamp: float
    is_voice: bool
    confidence: float
    energy: float
    zero_crossing_rate: int
    webrtc_result: bool
    smoothed_result: bool
    voice_probability: float
    frame_duration_ms: int
    segment_duration_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'timestamp': self.timestamp,
            'is_voice': self.is_voice,
            'confidence': self.confidence,
            'energy': self.energy,
            'zero_crossing_rate': self.zero_crossing_rate,
            'webrtc_result': self.webrtc_result,
            'smoothed_result': self.smoothed_result,
            'voice_probability': self.voice_probability,
            'frame_duration_ms': self.frame_duration_ms,
            'segment_duration_ms': self.segment_duration_ms
        }

class VoiceActivityDetector:
    """
    Real-time Voice Activity Detection using WebRTC VAD with enhancements
    
    This detector combines WebRTC VAD with energy and zero-crossing analysis
    to provide robust voice activity detection for transcription optimization.
    """
    
    def __init__(self, config: Optional[VADConfig] = None):
        """
        Initialize Voice Activity Detector
        
        Args:
            config: VAD configuration parameters
        """
        self.config = config or VADConfig()
        self.vad = webrtcvad.Vad(self.config.mode.value)
        
        # State tracking
        self.is_running = False
        self.current_segment_start = None
        self.current_is_voice = False
        self.frame_buffer = deque(maxlen=self.config.smoothing_window)
        
        # Threading
        self.processing_thread = None
        self.thread_lock = threading.Lock()
        
        # Callbacks
        self.voice_start_callbacks: List[Callable[[float], None]] = []
        self.voice_end_callbacks: List[Callable[[float, float], None]] = []
        self.result_callbacks: List[Callable[[VADResult], None]] = []
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'voice_frames': 0,
            'silence_frames': 0,
            'voice_segments': 0,
            'silence_segments': 0,
            'total_voice_time': 0.0,
            'total_silence_time': 0.0,
            'start_time': None
        }
        
        logger.info(f"VoiceActivityDetector initialized with mode={self.config.mode.name}, "
                   f"sensitivity={self.config.sensitivity.name}")
    
    def add_voice_start_callback(self, callback: Callable[[float], None]) -> None:
        """Add callback for voice segment start"""
        self.voice_start_callbacks.append(callback)
    
    def add_voice_end_callback(self, callback: Callable[[float, float], None]) -> None:
        """Add callback for voice segment end (start_time, duration)"""
        self.voice_end_callbacks.append(callback)
    
    def add_result_callback(self, callback: Callable[[VADResult], None]) -> None:
        """Add callback for each VAD result"""
        self.result_callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> bool:
        """Remove a callback from all callback lists"""
        removed = False
        for callback_list in [self.voice_start_callbacks, self.voice_end_callbacks, 
                             self.result_callbacks]:
            if callback in callback_list:
                callback_list.remove(callback)
                removed = True
        return removed
    
    def _calculate_energy(self, audio_data: np.ndarray) -> float:
        """Calculate normalized energy of audio frame"""
        if len(audio_data) == 0:
            return 0.0
        return float(np.mean(audio_data.astype(np.float64) ** 2))
    
    def _calculate_zero_crossing_rate(self, audio_data: np.ndarray) -> int:
        """Calculate zero crossing rate of audio frame"""
        if len(audio_data) <= 1:
            return 0
        return int(np.sum(np.diff(np.signbit(audio_data))))
    
    def _apply_smoothing(self, current_result: bool) -> Tuple[bool, float]:
        """Apply temporal smoothing to VAD results"""
        self.frame_buffer.append(current_result)
        
        if len(self.frame_buffer) < self.config.smoothing_window:
            # Not enough frames for smoothing
            return current_result, 1.0 if current_result else 0.0
        
        # Calculate voice probability over window
        voice_count = sum(self.frame_buffer)
        voice_probability = voice_count / len(self.frame_buffer)
        
        # Apply thresholds
        if voice_probability >= self.config.voice_threshold:
            smoothed_result = True
        elif voice_probability <= self.config.silence_threshold:
            smoothed_result = False
        else:
            # Maintain current state in ambiguous region
            smoothed_result = self.current_is_voice
        
        return smoothed_result, voice_probability
    
    def _handle_segment_transition(self, is_voice: bool, timestamp: float) -> None:
        """Handle transitions between voice and silence segments"""
        if is_voice != self.current_is_voice:
            if self.current_is_voice and not is_voice:
                # Voice segment ended
                if self.current_segment_start is not None:
                    duration = timestamp - self.current_segment_start
                    if duration >= self.config.min_voice_duration_ms / 1000.0:
                        # Valid voice segment
                        self.stats['voice_segments'] += 1
                        self.stats['total_voice_time'] += duration
                        
                        # Fire voice end callbacks
                        for callback in self.voice_end_callbacks:
                            try:
                                callback(self.current_segment_start, duration)
                            except Exception as e:
                                logger.error(f"Error in voice end callback: {e}")
                
            elif not self.current_is_voice and is_voice:
                # Silence segment ended, voice segment starting
                if self.current_segment_start is not None:
                    duration = timestamp - self.current_segment_start
                    if duration >= self.config.min_silence_duration_ms / 1000.0:
                        # Valid silence segment
                        self.stats['silence_segments'] += 1
                        self.stats['total_silence_time'] += duration
                
                # Fire voice start callbacks
                for callback in self.voice_start_callbacks:
                    try:
                        callback(timestamp)
                    except Exception as e:
                        logger.error(f"Error in voice start callback: {e}")
            
            # Update state
            self.current_is_voice = is_voice
            self.current_segment_start = timestamp
    
    def detect_voice_activity(self, audio_data: np.ndarray, timestamp: Optional[float] = None) -> VADResult:
        """
        Detect voice activity in audio frame
        
        Args:
            audio_data: Audio data as numpy array (16-bit integers)
            timestamp: Timestamp of the frame (defaults to current time)
        
        Returns:
            VADResult with detection results
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Ensure audio data is correct format
        if audio_data.dtype != np.int16:
            audio_data = audio_data.astype(np.int16)
        
        # Ensure correct frame size
        expected_size = self.config.chunk_size
        if len(audio_data) != expected_size:
            if len(audio_data) > expected_size:
                audio_data = audio_data[:expected_size]
            else:
                # Pad with zeros
                padded = np.zeros(expected_size, dtype=np.int16)
                padded[:len(audio_data)] = audio_data
                audio_data = padded
        
        try:
            # WebRTC VAD detection
            audio_bytes = audio_data.tobytes()
            webrtc_result = self.vad.is_speech(audio_bytes, self.config.sample_rate)
            
            # Calculate additional features
            energy = self._calculate_energy(audio_data)
            zero_crossing_rate = self._calculate_zero_crossing_rate(audio_data)
            
            # Combine WebRTC result with energy threshold
            energy_result = energy > self.config.energy_threshold
            combined_result = webrtc_result and energy_result
            
            # Apply smoothing
            smoothed_result, voice_probability = self._apply_smoothing(combined_result)
            
            # Calculate confidence based on multiple factors
            webrtc_confidence = 1.0 if webrtc_result else 0.0
            energy_confidence = min(energy / (self.config.energy_threshold * 10), 1.0)
            confidence = (webrtc_confidence + energy_confidence + voice_probability) / 3.0
            
            # Handle segment transitions
            with self.thread_lock:
                self._handle_segment_transition(smoothed_result, timestamp)
                
                # Update statistics
                self.stats['total_frames'] += 1
                if smoothed_result:
                    self.stats['voice_frames'] += 1
                else:
                    self.stats['silence_frames'] += 1
                
                if self.stats['start_time'] is None:
                    self.stats['start_time'] = timestamp
            
            # Create result
            result = VADResult(
                timestamp=timestamp,
                is_voice=smoothed_result,
                confidence=confidence,
                energy=energy,
                zero_crossing_rate=zero_crossing_rate,
                webrtc_result=webrtc_result,
                smoothed_result=smoothed_result,
                voice_probability=voice_probability,
                frame_duration_ms=self.config.frame_duration_ms,
                segment_duration_ms=None
            )
            
            # Fire result callbacks
            for callback in self.result_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in result callback: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in voice activity detection: {e}")
            # Return safe default result
            return VADResult(
                timestamp=timestamp,
                is_voice=False,
                confidence=0.0,
                energy=0.0,
                zero_crossing_rate=0,
                webrtc_result=False,
                smoothed_result=False,
                voice_probability=0.0,
                frame_duration_ms=self.config.frame_duration_ms
            )
    
    def start_processing(self) -> bool:
        """Start continuous processing (if using threading)"""
        if self.is_running:
            logger.warning("VAD processing already running")
            return False
        
        with self.thread_lock:
            self.is_running = True
            self.stats['start_time'] = time.time()
        
        logger.info("Voice Activity Detector started")
        return True
    
    def stop_processing(self) -> bool:
        """Stop continuous processing"""
        if not self.is_running:
            logger.warning("VAD processing not running")
            return False
        
        with self.thread_lock:
            self.is_running = False
        
        # Finalize any ongoing segment
        if self.current_segment_start is not None:
            current_time = time.time()
            duration = current_time - self.current_segment_start
            
            if self.current_is_voice:
                self.stats['voice_segments'] += 1
                self.stats['total_voice_time'] += duration
            else:
                self.stats['silence_segments'] += 1
                self.stats['total_silence_time'] += duration
        
        logger.info("Voice Activity Detector stopped")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics"""
        with self.thread_lock:
            stats = self.stats.copy()
            
            # Calculate derived statistics
            if stats['total_frames'] > 0:
                stats['voice_percentage'] = (stats['voice_frames'] / stats['total_frames']) * 100
                stats['silence_percentage'] = (stats['silence_frames'] / stats['total_frames']) * 100
            else:
                stats['voice_percentage'] = 0.0
                stats['silence_percentage'] = 0.0
            
            if stats['start_time'] is not None:
                stats['total_processing_time'] = time.time() - stats['start_time']
            else:
                stats['total_processing_time'] = 0.0
            
            if stats['voice_segments'] > 0:
                stats['average_voice_segment_duration'] = stats['total_voice_time'] / stats['voice_segments']
            else:
                stats['average_voice_segment_duration'] = 0.0
            
            if stats['silence_segments'] > 0:
                stats['average_silence_segment_duration'] = stats['total_silence_time'] / stats['silence_segments']
            else:
                stats['average_silence_segment_duration'] = 0.0
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset detection statistics"""
        with self.thread_lock:
            self.stats = {
                'total_frames': 0,
                'voice_frames': 0,
                'silence_frames': 0,
                'voice_segments': 0,
                'silence_segments': 0,
                'total_voice_time': 0.0,
                'total_silence_time': 0.0,
                'start_time': time.time() if self.is_running else None
            }
        
        logger.info("VAD statistics reset")
    
    def is_processing(self) -> bool:
        """Check if VAD is currently processing"""
        return self.is_running
    
    def get_config(self) -> VADConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, new_config: VADConfig) -> bool:
        """
        Update VAD configuration
        
        Args:
            new_config: New configuration
            
        Returns:
            True if successful, False if VAD is running
        """
        if self.is_running:
            logger.warning("Cannot update config while VAD is running")
            return False
        
        try:
            # Create new VAD instance with new mode
            self.vad = webrtcvad.Vad(new_config.mode.value)
            self.config = new_config
            
            # Reset frame buffer with new smoothing window
            self.frame_buffer = deque(maxlen=self.config.smoothing_window)
            
            logger.info(f"VAD configuration updated: mode={new_config.mode.name}, "
                       f"sensitivity={new_config.sensitivity.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating VAD configuration: {e}")
            return False

def create_vad_system(config: Optional[VADConfig] = None) -> VoiceActivityDetector:
    """
    Factory function to create a configured VAD system
    
    Args:
        config: Optional VAD configuration
        
    Returns:
        Configured VoiceActivityDetector instance
    """
    return VoiceActivityDetector(config)

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create VAD system
    config = VADConfig(
        mode=VADMode.AGGRESSIVE,
        sensitivity=VADSensitivity.MEDIUM,
        sample_rate=16000,
        frame_duration_ms=30
    )
    
    vad = create_vad_system(config)
    
    # Add example callbacks
    def on_voice_start(timestamp: float):
        print(f"Voice started at {timestamp:.3f}")
    
    def on_voice_end(start_time: float, duration: float):
        print(f"Voice ended: duration {duration:.3f}s")
    
    def on_result(result: VADResult):
        print(f"VAD: {result.is_voice} (confidence: {result.confidence:.2f})")
    
    vad.add_voice_start_callback(on_voice_start)
    vad.add_voice_end_callback(on_voice_end)
    vad.add_result_callback(on_result)
    
    # Start processing
    vad.start_processing()
    
    # Test with some synthetic audio data
    try:
        import time
        
        print("Testing VAD with synthetic audio...")
        
        # Generate test frames (30ms at 16kHz = 480 samples)
        frame_size = 480
        
        # Simulate some voice and silence
        for i in range(20):
            if i < 5 or i > 15:
                # Silence (low energy)
                audio_frame = np.random.randint(-100, 100, frame_size, dtype=np.int16)
            else:
                # Voice (higher energy with more variation)
                audio_frame = np.random.randint(-1000, 1000, frame_size, dtype=np.int16)
            
            result = vad.detect_voice_activity(audio_frame)
            time.sleep(0.03)  # 30ms delay
        
        # Print statistics
        stats = vad.get_statistics()
        print(f"\nStatistics:")
        print(f"Total frames: {stats['total_frames']}")
        print(f"Voice percentage: {stats['voice_percentage']:.1f}%")
        print(f"Voice segments: {stats['voice_segments']}")
        print(f"Average voice duration: {stats['average_voice_segment_duration']:.3f}s")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    finally:
        vad.stop_processing()
        print("VAD test completed")