#!/usr/bin/env python3
"""
Silence Detection and Trimming for The Silent Steno

This module provides silence detection and automatic trimming for recording
optimization, enabling efficient storage and processing.

Author: The Silent Steno Team
License: MIT
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque

# Configure logging
logger = logging.getLogger(__name__)

class SilenceMethod(Enum):
    """Methods for silence detection"""
    ENERGY = "energy"               # Energy-based detection
    AMPLITUDE = "amplitude"         # Amplitude threshold detection
    SPECTRAL = "spectral"          # Spectral analysis based
    ADAPTIVE = "adaptive"          # Adaptive threshold detection
    COMBINED = "combined"          # Multiple method combination

class SilenceMode(Enum):
    """Modes for silence detection sensitivity"""
    VERY_SENSITIVE = "very_sensitive"  # Detect even short pauses
    SENSITIVE = "sensitive"            # Normal sensitivity
    BALANCED = "balanced"              # Balanced detection
    CONSERVATIVE = "conservative"      # Only clear silence
    VERY_CONSERVATIVE = "very_conservative"  # Only obvious silence

class TrimMode(Enum):
    """Modes for audio trimming"""
    NONE = "none"                 # No trimming
    LEADING = "leading"           # Trim leading silence only
    TRAILING = "trailing"         # Trim trailing silence only
    BOTH = "both"                # Trim both ends
    SEGMENTS = "segments"         # Trim silence segments throughout

@dataclass
class SilenceThreshold:
    """Thresholds for silence detection"""
    energy_threshold: float = 0.01      # Energy threshold (0-1)
    amplitude_threshold: float = 0.05   # Amplitude threshold (0-1)
    spectral_threshold: float = 0.02    # Spectral flux threshold
    duration_threshold_ms: int = 100    # Minimum silence duration
    noise_floor_factor: float = 2.0     # Factor above noise floor
    
    def adjust_for_mode(self, mode: SilenceMode) -> None:
        """Adjust thresholds based on sensitivity mode"""
        if mode == SilenceMode.VERY_SENSITIVE:
            self.energy_threshold *= 0.5
            self.amplitude_threshold *= 0.5
            self.duration_threshold_ms = 50
        elif mode == SilenceMode.SENSITIVE:
            self.energy_threshold *= 0.7
            self.amplitude_threshold *= 0.7
            self.duration_threshold_ms = 75
        elif mode == SilenceMode.BALANCED:
            pass  # Use default values
        elif mode == SilenceMode.CONSERVATIVE:
            self.energy_threshold *= 1.5
            self.amplitude_threshold *= 1.5
            self.duration_threshold_ms = 150
        elif mode == SilenceMode.VERY_CONSERVATIVE:
            self.energy_threshold *= 2.0
            self.amplitude_threshold *= 2.0
            self.duration_threshold_ms = 200

@dataclass
class SilenceConfig:
    """Configuration for Silence Detector"""
    method: SilenceMethod = SilenceMethod.COMBINED
    mode: SilenceMode = SilenceMode.BALANCED
    thresholds: SilenceThreshold = field(default_factory=SilenceThreshold)
    sample_rate: int = 16000
    frame_duration_ms: int = 50  # Analysis frame duration
    window_size: int = 10        # Smoothing window size
    trim_mode: TrimMode = TrimMode.BOTH
    min_audio_duration_ms: int = 100  # Minimum audio to keep
    max_silence_gap_ms: int = 2000    # Maximum gap to bridge
    enable_noise_floor_adaptation: bool = True
    enable_spectral_analysis: bool = True
    
    def __post_init__(self):
        """Validate and adjust configuration"""
        self.frame_samples = int(self.sample_rate * self.frame_duration_ms / 1000)
        self.thresholds.adjust_for_mode(self.mode)

@dataclass
class SilenceResult:
    """Result of silence detection"""
    timestamp: float
    frame_start_ms: float
    frame_duration_ms: float
    is_silence: bool
    confidence: float
    energy_level: float
    amplitude_level: float
    spectral_flux: float
    noise_floor: float
    method_results: Dict[str, bool]  # Results from each detection method
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'timestamp': self.timestamp,
            'frame_start_ms': self.frame_start_ms,
            'frame_duration_ms': self.frame_duration_ms,
            'is_silence': self.is_silence,
            'confidence': self.confidence,
            'energy_level': self.energy_level,
            'amplitude_level': self.amplitude_level,
            'spectral_flux': self.spectral_flux,
            'noise_floor': self.noise_floor,
            'method_results': self.method_results
        }

@dataclass
class SilenceSegment:
    """Information about a silence segment"""
    start_time_ms: float
    end_time_ms: float
    duration_ms: float
    confidence: float
    energy_level: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary"""
        return {
            'start_time_ms': self.start_time_ms,
            'end_time_ms': self.end_time_ms,
            'duration_ms': self.duration_ms,
            'confidence': self.confidence,
            'energy_level': self.energy_level
        }

@dataclass
class TrimResult:
    """Result of audio trimming operation"""
    original_duration_ms: float
    trimmed_duration_ms: float
    leading_trimmed_ms: float
    trailing_trimmed_ms: float
    segments_removed: int
    silence_segments: List[SilenceSegment]
    trimmed_audio: Optional[np.ndarray] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def get_trim_ratio(self) -> float:
        """Get ratio of audio that was trimmed"""
        if self.original_duration_ms == 0:
            return 0.0
        total_trimmed = self.leading_trimmed_ms + self.trailing_trimmed_ms
        return total_trimmed / self.original_duration_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'original_duration_ms': self.original_duration_ms,
            'trimmed_duration_ms': self.trimmed_duration_ms,
            'leading_trimmed_ms': self.leading_trimmed_ms,
            'trailing_trimmed_ms': self.trailing_trimmed_ms,
            'segments_removed': self.segments_removed,
            'trim_ratio': self.get_trim_ratio(),
            'silence_segments': [seg.to_dict() for seg in self.silence_segments],
            'success': self.success,
            'error_message': self.error_message
        }

class SilenceDetector:
    """
    Silence detection and automatic trimming system
    
    This detector analyzes audio to identify silence segments and provides
    trimming capabilities for efficient storage and processing.
    """
    
    def __init__(self, config: Optional[SilenceConfig] = None):
        """
        Initialize Silence Detector
        
        Args:
            config: Silence detection configuration
        """
        self.config = config or SilenceConfig()
        
        # State tracking
        self.is_running = False
        self.frame_buffer = deque(maxlen=self.config.window_size)
        self.noise_floor = 0.0
        self.adaptive_threshold = self.config.thresholds.energy_threshold
        self.last_spectral_flux = 0.0
        
        # Silence tracking
        self.current_silence_start = None
        self.silence_segments: List[SilenceSegment] = []
        self.in_silence = False
        
        # Threading
        self.processing_thread = None
        self.thread_lock = threading.Lock()
        
        # Callbacks
        self.silence_start_callbacks: List[Callable[[float], None]] = []
        self.silence_end_callbacks: List[Callable[[float, float], None]] = []
        self.result_callbacks: List[Callable[[SilenceResult], None]] = []
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'silence_frames': 0,
            'voice_frames': 0,
            'silence_segments': 0,
            'total_silence_time_ms': 0.0,
            'total_voice_time_ms': 0.0,
            'average_silence_duration_ms': 0.0,
            'start_time': None
        }
        
        logger.info(f"SilenceDetector initialized with method={self.config.method.value}, "
                   f"mode={self.config.mode.value}")
    
    def add_silence_start_callback(self, callback: Callable[[float], None]) -> None:
        """Add callback for silence start"""
        self.silence_start_callbacks.append(callback)
    
    def add_silence_end_callback(self, callback: Callable[[float, float], None]) -> None:
        """Add callback for silence end (start_time, duration)"""
        self.silence_end_callbacks.append(callback)
    
    def add_result_callback(self, callback: Callable[[SilenceResult], None]) -> None:
        """Add callback for detection results"""
        self.result_callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> bool:
        """Remove a callback from all callback lists"""
        removed = False
        for callback_list in [self.silence_start_callbacks, self.silence_end_callbacks, 
                             self.result_callbacks]:
            if callback in callback_list:
                callback_list.remove(callback)
                removed = True
        return removed
    
    def _update_noise_floor(self, audio_data: np.ndarray) -> None:
        """Update adaptive noise floor estimation"""
        if not self.config.enable_noise_floor_adaptation:
            return
        
        try:
            # Calculate current frame energy
            energy = np.mean(audio_data.astype(np.float64) ** 2)
            
            # Update noise floor using exponential moving average
            alpha = 0.01  # Slow adaptation
            if self.noise_floor == 0.0:
                self.noise_floor = energy
            else:
                # Only update if current energy is below current estimate (likely noise)
                if energy < self.noise_floor * 2:
                    self.noise_floor = alpha * energy + (1 - alpha) * self.noise_floor
            
            # Update adaptive threshold
            self.adaptive_threshold = max(
                self.config.thresholds.energy_threshold,
                self.noise_floor * self.config.thresholds.noise_floor_factor
            )
            
        except Exception as e:
            logger.error(f"Error updating noise floor: {e}")
    
    def _detect_silence_energy(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """Detect silence using energy-based method"""
        try:
            if len(audio_data) == 0:
                return True, 1.0
            
            # Calculate normalized energy
            energy = np.mean(audio_data.astype(np.float64) ** 2)
            max_energy = 32767.0 ** 2  # Max energy for 16-bit audio
            normalized_energy = energy / max_energy
            
            # Use adaptive threshold if available
            threshold = self.adaptive_threshold
            is_silence = normalized_energy < threshold
            
            # Calculate confidence based on how far below threshold
            if is_silence:
                confidence = min((threshold - normalized_energy) / threshold, 1.0)
            else:
                confidence = min(normalized_energy / threshold, 1.0)
            
            return is_silence, float(confidence)
            
        except Exception as e:
            logger.error(f"Error in energy-based silence detection: {e}")
            return False, 0.0
    
    def _detect_silence_amplitude(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """Detect silence using amplitude-based method"""
        try:
            if len(audio_data) == 0:
                return True, 1.0
            
            # Calculate peak amplitude
            max_amplitude = np.max(np.abs(audio_data))
            normalized_amplitude = max_amplitude / 32767.0  # Normalize to 0-1
            
            threshold = self.config.thresholds.amplitude_threshold
            is_silence = normalized_amplitude < threshold
            
            # Calculate confidence
            if is_silence:
                confidence = min((threshold - normalized_amplitude) / threshold, 1.0)
            else:
                confidence = min(normalized_amplitude / threshold, 1.0)
            
            return is_silence, float(confidence)
            
        except Exception as e:
            logger.error(f"Error in amplitude-based silence detection: {e}")
            return False, 0.0
    
    def _detect_silence_spectral(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """Detect silence using spectral analysis"""
        try:
            if len(audio_data) == 0:
                return True, 1.0
            
            if not self.config.enable_spectral_analysis:
                return False, 0.0
            
            # Calculate spectral flux (measure of spectral change)
            fft = np.fft.rfft(audio_data.astype(np.float64))
            magnitude_spectrum = np.abs(fft)
            
            # Spectral flux is the sum of positive differences
            if self.last_spectral_flux is not None:
                flux = np.sum(np.maximum(0, magnitude_spectrum - self.last_spectral_flux))
                normalized_flux = flux / len(magnitude_spectrum)
            else:
                normalized_flux = 0.0
            
            self.last_spectral_flux = magnitude_spectrum
            
            threshold = self.config.thresholds.spectral_threshold * np.max(magnitude_spectrum)
            is_silence = normalized_flux < threshold
            
            # Calculate confidence
            if threshold > 0:
                if is_silence:
                    confidence = min((threshold - normalized_flux) / threshold, 1.0)
                else:
                    confidence = min(normalized_flux / threshold, 1.0)
            else:
                confidence = 0.5
            
            return is_silence, float(confidence)
            
        except Exception as e:
            logger.error(f"Error in spectral-based silence detection: {e}")
            return False, 0.0
    
    def _detect_silence_adaptive(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """Detect silence using adaptive method"""
        try:
            # Combine multiple methods with adaptive weighting
            energy_silence, energy_conf = self._detect_silence_energy(audio_data)
            amplitude_silence, amplitude_conf = self._detect_silence_amplitude(audio_data)
            
            # Weight based on recent history
            if len(self.frame_buffer) > 5:
                recent_results = [frame.is_silence for frame in list(self.frame_buffer)[-5:]]
                silence_ratio = sum(recent_results) / len(recent_results)
                
                # If mostly silence recently, be more sensitive
                if silence_ratio > 0.7:
                    energy_weight = 0.7
                    amplitude_weight = 0.3
                # If mostly voice recently, be less sensitive
                elif silence_ratio < 0.3:
                    energy_weight = 0.3
                    amplitude_weight = 0.7
                else:
                    energy_weight = 0.5
                    amplitude_weight = 0.5
            else:
                energy_weight = 0.5
                amplitude_weight = 0.5
            
            # Combine results
            combined_confidence = (energy_conf * energy_weight + 
                                 amplitude_conf * amplitude_weight)
            
            # Decision based on weighted vote
            silence_votes = 0
            total_votes = 2
            
            if energy_silence:
                silence_votes += energy_weight * 2
            if amplitude_silence:
                silence_votes += amplitude_weight * 2
            
            is_silence = silence_votes > total_votes * 0.5
            
            return is_silence, float(combined_confidence)
            
        except Exception as e:
            logger.error(f"Error in adaptive silence detection: {e}")
            return False, 0.0
    
    def _apply_temporal_smoothing(self, is_silence: bool, confidence: float) -> Tuple[bool, float]:
        """Apply temporal smoothing to reduce false positives"""
        if len(self.frame_buffer) < self.config.window_size:
            return is_silence, confidence
        
        # Count recent silence detections
        recent_frames = list(self.frame_buffer)[-self.config.window_size:]
        silence_count = sum(1 for frame in recent_frames if frame.is_silence)
        silence_ratio = silence_count / len(recent_frames)
        
        # Smooth decision based on recent history
        if silence_ratio > 0.7:
            smoothed_silence = True
            smoothed_confidence = min(confidence + 0.1, 1.0)
        elif silence_ratio < 0.3:
            smoothed_silence = False
            smoothed_confidence = min(confidence + 0.1, 1.0)
        else:
            # Ambiguous region - maintain current detection
            smoothed_silence = is_silence
            smoothed_confidence = confidence * 0.8  # Reduce confidence
        
        return smoothed_silence, smoothed_confidence
    
    def detect_silence(self, audio_data: np.ndarray, timestamp: Optional[float] = None,
                      frame_start_ms: Optional[float] = None) -> SilenceResult:
        """
        Detect silence in audio frame
        
        Args:
            audio_data: Audio data as numpy array
            timestamp: Timestamp of detection
            frame_start_ms: Start time of frame in milliseconds
        
        Returns:
            SilenceResult with detection results
        """
        if timestamp is None:
            timestamp = time.time()
        
        if frame_start_ms is None:
            frame_start_ms = timestamp * 1000
        
        try:
            # Ensure correct format
            if audio_data.dtype != np.int16:
                audio_data = audio_data.astype(np.int16)
            
            # Update noise floor
            self._update_noise_floor(audio_data)
            
            # Apply different detection methods
            method_results = {}
            
            if self.config.method in [SilenceMethod.ENERGY, SilenceMethod.COMBINED]:
                energy_silence, energy_conf = self._detect_silence_energy(audio_data)
                method_results['energy'] = energy_silence
            else:
                energy_silence, energy_conf = False, 0.0
            
            if self.config.method in [SilenceMethod.AMPLITUDE, SilenceMethod.COMBINED]:
                amplitude_silence, amplitude_conf = self._detect_silence_amplitude(audio_data)
                method_results['amplitude'] = amplitude_silence
            else:
                amplitude_silence, amplitude_conf = False, 0.0
            
            if self.config.method in [SilenceMethod.SPECTRAL, SilenceMethod.COMBINED]:
                spectral_silence, spectral_conf = self._detect_silence_spectral(audio_data)
                method_results['spectral'] = spectral_silence
            else:
                spectral_silence, spectral_conf = False, 0.0
            
            if self.config.method == SilenceMethod.ADAPTIVE:
                is_silence, confidence = self._detect_silence_adaptive(audio_data)
                method_results['adaptive'] = is_silence
            elif self.config.method == SilenceMethod.COMBINED:
                # Combine all methods
                silence_votes = sum([energy_silence, amplitude_silence, spectral_silence])
                is_silence = silence_votes >= 2  # Majority vote
                confidence = (energy_conf + amplitude_conf + spectral_conf) / 3.0
                method_results['combined'] = is_silence
            else:
                # Single method
                if self.config.method == SilenceMethod.ENERGY:
                    is_silence, confidence = energy_silence, energy_conf
                elif self.config.method == SilenceMethod.AMPLITUDE:
                    is_silence, confidence = amplitude_silence, amplitude_conf
                elif self.config.method == SilenceMethod.SPECTRAL:
                    is_silence, confidence = spectral_silence, spectral_conf
                else:
                    is_silence, confidence = False, 0.0
            
            # Apply temporal smoothing
            is_silence, confidence = self._apply_temporal_smoothing(is_silence, confidence)
            
            # Calculate additional metrics
            energy_level = np.mean(audio_data.astype(np.float64) ** 2) / (32767.0 ** 2)
            amplitude_level = np.max(np.abs(audio_data)) / 32767.0
            spectral_flux = self.last_spectral_flux.mean() if self.last_spectral_flux is not None else 0.0
            
            # Create result
            result = SilenceResult(
                timestamp=timestamp,
                frame_start_ms=frame_start_ms,
                frame_duration_ms=self.config.frame_duration_ms,
                is_silence=is_silence,
                confidence=confidence,
                energy_level=float(energy_level),
                amplitude_level=float(amplitude_level),
                spectral_flux=float(spectral_flux),
                noise_floor=float(self.noise_floor),
                method_results=method_results
            )
            
            # Add to buffer
            self.frame_buffer.append(result)
            
            # Handle silence segment tracking
            with self.thread_lock:
                self._handle_silence_segments(result, frame_start_ms)
                
                # Update statistics
                self.stats['total_frames'] += 1
                if is_silence:
                    self.stats['silence_frames'] += 1
                    self.stats['total_silence_time_ms'] += self.config.frame_duration_ms
                else:
                    self.stats['voice_frames'] += 1
                    self.stats['total_voice_time_ms'] += self.config.frame_duration_ms
                
                if self.stats['start_time'] is None:
                    self.stats['start_time'] = timestamp
            
            # Fire callbacks
            for callback in self.result_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in result callback: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in silence detection: {e}")
            # Return safe default result
            return SilenceResult(
                timestamp=timestamp,
                frame_start_ms=frame_start_ms,
                frame_duration_ms=self.config.frame_duration_ms,
                is_silence=False,
                confidence=0.0,
                energy_level=0.0,
                amplitude_level=0.0,
                spectral_flux=0.0,
                noise_floor=0.0,
                method_results={}
            )
    
    def _handle_silence_segments(self, result: SilenceResult, frame_start_ms: float) -> None:
        """Handle silence segment tracking"""
        if result.is_silence and not self.in_silence:
            # Start of silence segment
            self.current_silence_start = frame_start_ms
            self.in_silence = True
            
            # Fire silence start callbacks
            for callback in self.silence_start_callbacks:
                try:
                    callback(frame_start_ms)
                except Exception as e:
                    logger.error(f"Error in silence start callback: {e}")
        
        elif not result.is_silence and self.in_silence:
            # End of silence segment
            if self.current_silence_start is not None:
                duration_ms = frame_start_ms - self.current_silence_start
                
                # Only count as silence if above minimum duration
                if duration_ms >= self.config.thresholds.duration_threshold_ms:
                    segment = SilenceSegment(
                        start_time_ms=self.current_silence_start,
                        end_time_ms=frame_start_ms,
                        duration_ms=duration_ms,
                        confidence=result.confidence,
                        energy_level=result.energy_level
                    )
                    self.silence_segments.append(segment)
                    self.stats['silence_segments'] += 1
                    
                    # Update average silence duration
                    total_segments = self.stats['silence_segments']
                    if total_segments > 0:
                        self.stats['average_silence_duration_ms'] = (
                            self.stats['total_silence_time_ms'] / total_segments)
                    
                    # Fire silence end callbacks
                    for callback in self.silence_end_callbacks:
                        try:
                            callback(self.current_silence_start, duration_ms)
                        except Exception as e:
                            logger.error(f"Error in silence end callback: {e}")
            
            self.in_silence = False
            self.current_silence_start = None
    
    def trim_audio(self, audio_data: np.ndarray, 
                   sample_rate: Optional[int] = None) -> TrimResult:
        """
        Trim silence from audio data
        
        Args:
            audio_data: Audio data to trim
            sample_rate: Sample rate (uses config default if not provided)
        
        Returns:
            TrimResult with trimming information and processed audio
        """
        if sample_rate is None:
            sample_rate = self.config.sample_rate
        
        original_duration_ms = (len(audio_data) / sample_rate) * 1000
        
        try:
            if len(audio_data) == 0:
                return TrimResult(
                    original_duration_ms=0.0,
                    trimmed_duration_ms=0.0,
                    leading_trimmed_ms=0.0,
                    trailing_trimmed_ms=0.0,
                    segments_removed=0,
                    silence_segments=[],
                    trimmed_audio=audio_data,
                    success=True
                )
            
            # Analyze entire audio for silence segments
            frame_samples = int(sample_rate * self.config.frame_duration_ms / 1000)
            silence_segments = []
            
            for i in range(0, len(audio_data), frame_samples):
                frame = audio_data[i:i + frame_samples]
                if len(frame) < frame_samples // 2:  # Skip very short frames
                    continue
                
                frame_start_ms = (i / sample_rate) * 1000
                result = self.detect_silence(frame, frame_start_ms=frame_start_ms)
                
                if result.is_silence:
                    # Check if this extends an existing segment or starts a new one
                    if (silence_segments and 
                        frame_start_ms - silence_segments[-1].end_time_ms <= self.config.frame_duration_ms):
                        # Extend last segment
                        silence_segments[-1].end_time_ms = frame_start_ms + self.config.frame_duration_ms
                        silence_segments[-1].duration_ms = (silence_segments[-1].end_time_ms - 
                                                          silence_segments[-1].start_time_ms)
                    else:
                        # New segment
                        segment = SilenceSegment(
                            start_time_ms=frame_start_ms,
                            end_time_ms=frame_start_ms + self.config.frame_duration_ms,
                            duration_ms=self.config.frame_duration_ms,
                            confidence=result.confidence,
                            energy_level=result.energy_level
                        )
                        silence_segments.append(segment)
            
            # Filter segments by minimum duration
            valid_segments = [seg for seg in silence_segments 
                            if seg.duration_ms >= self.config.thresholds.duration_threshold_ms]
            
            # Apply trimming based on mode
            trimmed_audio = audio_data.copy()
            leading_trimmed_ms = 0.0
            trailing_trimmed_ms = 0.0
            segments_removed = 0
            
            if self.config.trim_mode in [TrimMode.LEADING, TrimMode.BOTH]:
                # Find leading silence
                for segment in valid_segments:
                    if segment.start_time_ms == 0:  # Starts at beginning
                        trim_samples = int((segment.duration_ms / 1000) * sample_rate)
                        trim_samples = min(trim_samples, len(trimmed_audio))
                        trimmed_audio = trimmed_audio[trim_samples:]
                        leading_trimmed_ms = segment.duration_ms
                        segments_removed += 1
                        break
            
            if self.config.trim_mode in [TrimMode.TRAILING, TrimMode.BOTH]:
                # Find trailing silence
                audio_duration_ms = (len(trimmed_audio) / sample_rate) * 1000
                for segment in reversed(valid_segments):
                    # Check if segment ends at the end of audio
                    if abs(segment.end_time_ms - original_duration_ms) < self.config.frame_duration_ms:
                        trim_samples = int((segment.duration_ms / 1000) * sample_rate)
                        trim_samples = min(trim_samples, len(trimmed_audio))
                        if trim_samples > 0:
                            trimmed_audio = trimmed_audio[:-trim_samples]
                            trailing_trimmed_ms = segment.duration_ms
                            segments_removed += 1
                        break
            
            if self.config.trim_mode == TrimMode.SEGMENTS:
                # Remove silence segments throughout (more complex)
                # For now, just remove leading and trailing
                # Full implementation would require more sophisticated audio splicing
                pass
            
            # Ensure minimum audio duration
            trimmed_duration_ms = (len(trimmed_audio) / sample_rate) * 1000
            min_duration = self.config.min_audio_duration_ms
            
            if trimmed_duration_ms < min_duration and original_duration_ms >= min_duration:
                # Restore some audio to meet minimum duration
                needed_ms = min_duration - trimmed_duration_ms
                restore_samples = int((needed_ms / 1000) * sample_rate)
                
                # Restore from trailing trim first
                if trailing_trimmed_ms > 0:
                    restore_from_trail = min(restore_samples, 
                                           int((trailing_trimmed_ms / 1000) * sample_rate))
                    original_end = len(audio_data)
                    original_trim_end = original_end - int((trailing_trimmed_ms / 1000) * sample_rate)
                    restore_start = original_trim_end + len(trimmed_audio) - len(audio_data) + int((leading_trimmed_ms / 1000) * sample_rate)
                    
                    if restore_start >= 0 and restore_start + restore_from_trail <= len(audio_data):
                        trimmed_audio = np.concatenate([
                            trimmed_audio,
                            audio_data[restore_start:restore_start + restore_from_trail]
                        ])
                        trailing_trimmed_ms -= (restore_from_trail / sample_rate) * 1000
                        restore_samples -= restore_from_trail
                
                # Restore from leading trim if still needed
                if restore_samples > 0 and leading_trimmed_ms > 0:
                    restore_from_lead = min(restore_samples,
                                          int((leading_trimmed_ms / 1000) * sample_rate))
                    if restore_from_lead <= len(audio_data):
                        trimmed_audio = np.concatenate([
                            audio_data[:restore_from_lead],
                            trimmed_audio
                        ])
                        leading_trimmed_ms -= (restore_from_lead / sample_rate) * 1000
            
            # Final duration
            final_duration_ms = (len(trimmed_audio) / sample_rate) * 1000
            
            return TrimResult(
                original_duration_ms=original_duration_ms,
                trimmed_duration_ms=final_duration_ms,
                leading_trimmed_ms=leading_trimmed_ms,
                trailing_trimmed_ms=trailing_trimmed_ms,
                segments_removed=segments_removed,
                silence_segments=valid_segments,
                trimmed_audio=trimmed_audio,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error trimming audio: {e}")
            return TrimResult(
                original_duration_ms=original_duration_ms,
                trimmed_duration_ms=original_duration_ms,
                leading_trimmed_ms=0.0,
                trailing_trimmed_ms=0.0,
                segments_removed=0,
                silence_segments=[],
                trimmed_audio=audio_data,
                success=False,
                error_message=str(e)
            )
    
    def get_silence_segments(self) -> List[SilenceSegment]:
        """Get detected silence segments"""
        with self.thread_lock:
            return self.silence_segments.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get silence detection statistics"""
        with self.thread_lock:
            stats = self.stats.copy()
            
            # Add derived statistics
            if stats['total_frames'] > 0:
                stats['silence_percentage'] = (stats['silence_frames'] / stats['total_frames']) * 100
                stats['voice_percentage'] = (stats['voice_frames'] / stats['total_frames']) * 100
            else:
                stats['silence_percentage'] = 0.0
                stats['voice_percentage'] = 0.0
            
            stats['current_noise_floor'] = self.noise_floor
            stats['current_adaptive_threshold'] = self.adaptive_threshold
            stats['buffer_frames'] = len(self.frame_buffer)
            stats['current_silence_segments'] = len(self.silence_segments)
        
        return stats
    
    def reset_detector(self) -> None:
        """Reset silence detector state"""
        with self.thread_lock:
            self.frame_buffer.clear()
            self.silence_segments.clear()
            self.current_silence_start = None
            self.in_silence = False
            self.noise_floor = 0.0
            self.adaptive_threshold = self.config.thresholds.energy_threshold
            self.last_spectral_flux = 0.0
            
            # Reset statistics
            self.stats = {
                'total_frames': 0,
                'silence_frames': 0,
                'voice_frames': 0,
                'silence_segments': 0,
                'total_silence_time_ms': 0.0,
                'total_voice_time_ms': 0.0,
                'average_silence_duration_ms': 0.0,
                'start_time': None
            }
        
        logger.info("Silence detector reset")
    
    def start_processing(self) -> bool:
        """Start silence detection processing"""
        if self.is_running:
            logger.warning("Silence detection already running")
            return False
        
        with self.thread_lock:
            self.is_running = True
            self.stats['start_time'] = time.time()
        
        logger.info("Silence detection started")
        return True
    
    def stop_processing(self) -> bool:
        """Stop silence detection processing"""
        if not self.is_running:
            logger.warning("Silence detection not running")
            return False
        
        with self.thread_lock:
            self.is_running = False
            
            # Finalize any ongoing silence segment
            if self.in_silence and self.current_silence_start is not None:
                current_time_ms = time.time() * 1000
                duration_ms = current_time_ms - self.current_silence_start
                
                if duration_ms >= self.config.thresholds.duration_threshold_ms:
                    segment = SilenceSegment(
                        start_time_ms=self.current_silence_start,
                        end_time_ms=current_time_ms,
                        duration_ms=duration_ms,
                        confidence=0.8,  # Default confidence for final segment
                        energy_level=self.noise_floor
                    )
                    self.silence_segments.append(segment)
                    self.stats['silence_segments'] += 1
        
        logger.info("Silence detection stopped")
        return True
    
    def is_processing(self) -> bool:
        """Check if silence detection is currently processing"""
        return self.is_running

def create_silence_detector(config: Optional[SilenceConfig] = None) -> SilenceDetector:
    """
    Factory function to create a configured silence detector
    
    Args:
        config: Optional silence detection configuration
        
    Returns:
        Configured SilenceDetector instance
    """
    return SilenceDetector(config)

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create silence detector
    config = SilenceConfig(
        method=SilenceMethod.COMBINED,
        mode=SilenceMode.BALANCED,
        trim_mode=TrimMode.BOTH
    )
    
    detector = create_silence_detector(config)
    
    # Add example callbacks
    def on_silence_start(start_time: float):
        print(f"Silence started at {start_time:.3f}ms")
    
    def on_silence_end(start_time: float, duration: float):
        print(f"Silence ended: {duration:.1f}ms duration")
    
    def on_result(result: SilenceResult):
        print(f"Frame: {'SILENCE' if result.is_silence else 'VOICE'} "
              f"(confidence: {result.confidence:.2f})")
    
    detector.add_silence_start_callback(on_silence_start)
    detector.add_silence_end_callback(on_silence_end)
    detector.add_result_callback(on_result)
    
    # Start processing
    detector.start_processing()
    
    # Test with synthetic audio data
    try:
        print("Testing silence detector with synthetic audio...")
        
        frame_size = 800  # 50ms at 16kHz
        
        # Create test audio with silence and voice patterns
        test_audio = []
        for i in range(40):
            if i < 5 or i > 35:
                # Silence (very low amplitude)
                frame = np.random.randint(-50, 50, frame_size, dtype=np.int16)
            elif 10 <= i <= 15 or 25 <= i <= 30:
                # More silence
                frame = np.random.randint(-100, 100, frame_size, dtype=np.int16)
            else:
                # Voice (higher amplitude)
                frame = np.random.randint(-1000, 1000, frame_size, dtype=np.int16)
                # Add some periodicity for voice-like characteristics
                t = np.linspace(0, 0.05, frame_size)
                frame += (500 * np.sin(2 * np.pi * 200 * t)).astype(np.int16)
            
            test_audio.extend(frame)
            
            # Detect silence in frame
            frame_start_ms = i * 50  # 50ms frames
            result = detector.detect_silence(frame, frame_start_ms=frame_start_ms)
            time.sleep(0.05)  # 50ms delay
        
        # Convert to numpy array for trimming test
        test_audio = np.array(test_audio, dtype=np.int16)
        
        print(f"\nTesting audio trimming...")
        print(f"Original audio duration: {len(test_audio) / 16000 * 1000:.1f}ms")
        
        # Test trimming
        trim_result = detector.trim_audio(test_audio)
        print(f"Trimmed audio duration: {trim_result.trimmed_duration_ms:.1f}ms")
        print(f"Leading trimmed: {trim_result.leading_trimmed_ms:.1f}ms")
        print(f"Trailing trimmed: {trim_result.trailing_trimmed_ms:.1f}ms")
        print(f"Segments removed: {trim_result.segments_removed}")
        print(f"Trim ratio: {trim_result.get_trim_ratio():.2f}")
        
        # Print statistics
        stats = detector.get_statistics()
        print(f"\nStatistics:")
        print(f"Total frames: {stats['total_frames']}")
        print(f"Silence percentage: {stats['silence_percentage']:.1f}%")
        print(f"Voice percentage: {stats['voice_percentage']:.1f}%")
        print(f"Silence segments: {stats['silence_segments']}")
        print(f"Average silence duration: {stats['average_silence_duration_ms']:.1f}ms")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    finally:
        detector.stop_processing()
        print("Silence detection test completed")