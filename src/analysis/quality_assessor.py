#!/usr/bin/env python3
"""
Audio Quality Assessment for The Silent Steno

This module provides comprehensive audio quality assessment with SNR, clarity,
and distortion metrics for recording and transcription optimization.

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
import math

# Configure logging
logger = logging.getLogger(__name__)

class QualityLevel(Enum):
    """Audio quality levels"""
    POOR = "poor"           # Below acceptable threshold
    FAIR = "fair"           # Acceptable but not ideal
    GOOD = "good"           # Good quality
    EXCELLENT = "excellent" # Excellent quality

class QualityMetric(Enum):
    """Types of quality metrics"""
    SNR = "snr"                      # Signal-to-noise ratio
    THD = "thd"                      # Total harmonic distortion
    CLARITY = "clarity"              # Speech clarity
    DYNAMIC_RANGE = "dynamic_range"  # Dynamic range
    FREQUENCY_RESPONSE = "frequency_response"  # Frequency response balance
    CLIPPING = "clipping"            # Clipping detection
    BACKGROUND_NOISE = "background_noise"  # Background noise level
    CONSISTENCY = "consistency"      # Temporal consistency

@dataclass
class QualityThresholds:
    """Thresholds for quality assessment"""
    snr_poor: float = 10.0      # dB
    snr_fair: float = 20.0      # dB
    snr_good: float = 30.0      # dB
    snr_excellent: float = 40.0 # dB
    
    thd_excellent: float = 0.01  # 1%
    thd_good: float = 0.05      # 5%
    thd_fair: float = 0.1       # 10%
    thd_poor: float = 0.2       # 20%
    
    clarity_poor: float = 0.3
    clarity_fair: float = 0.5
    clarity_good: float = 0.7
    clarity_excellent: float = 0.9
    
    clipping_threshold: float = 0.95  # 95% of max amplitude
    noise_floor_threshold: float = 0.01  # 1% of signal
    consistency_window: int = 10  # Number of frames for consistency check

@dataclass
class QualityConfig:
    """Configuration for Quality Assessor"""
    sample_rate: int = 16000
    frame_duration_ms: int = 100  # Analysis frame duration
    overlap_ratio: float = 0.5    # Overlap between analysis frames
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    enable_real_time: bool = True  # Enable real-time assessment
    history_length: int = 50      # Number of frames to keep in history
    smoothing_factor: float = 0.1  # Exponential smoothing factor
    frequency_bands: List[Tuple[float, float]] = field(default_factory=lambda: [
        (80, 250),    # Low frequencies
        (250, 1000),  # Mid-low frequencies  
        (1000, 4000), # Mid frequencies (speech)
        (4000, 8000)  # High frequencies
    ])
    
    def __post_init__(self):
        """Validate and adjust configuration"""
        self.frame_samples = int(self.sample_rate * self.frame_duration_ms / 1000)
        self.overlap_samples = int(self.frame_samples * self.overlap_ratio)
        self.hop_samples = self.frame_samples - self.overlap_samples

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for audio"""
    timestamp: float
    overall_quality: float        # 0-1 overall quality score
    quality_level: QualityLevel   # Categorical quality level
    
    # Signal metrics
    snr_db: float                 # Signal-to-noise ratio in dB
    thd_percent: float           # Total harmonic distortion percentage
    dynamic_range_db: float      # Dynamic range in dB
    peak_level_db: float         # Peak signal level in dB
    rms_level_db: float          # RMS signal level in dB
    
    # Clarity metrics
    clarity_score: float         # Speech clarity score (0-1)
    spectral_clarity: float      # Spectral clarity score (0-1)
    temporal_clarity: float      # Temporal clarity score (0-1)
    
    # Distortion metrics
    clipping_ratio: float        # Ratio of clipped samples
    noise_floor_db: float        # Background noise floor in dB
    frequency_balance: Dict[str, float]  # Balance across frequency bands
    
    # Consistency metrics
    temporal_consistency: float  # Consistency over time (0-1)
    level_stability: float       # Signal level stability (0-1)
    
    # Detection flags
    has_clipping: bool
    has_excessive_noise: bool
    has_dropouts: bool
    has_artifacts: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'timestamp': self.timestamp,
            'overall_quality': self.overall_quality,
            'quality_level': self.quality_level.value,
            'snr_db': self.snr_db,
            'thd_percent': self.thd_percent,
            'dynamic_range_db': self.dynamic_range_db,
            'peak_level_db': self.peak_level_db,
            'rms_level_db': self.rms_level_db,
            'clarity_score': self.clarity_score,
            'spectral_clarity': self.spectral_clarity,
            'temporal_clarity': self.temporal_clarity,
            'clipping_ratio': self.clipping_ratio,
            'noise_floor_db': self.noise_floor_db,
            'frequency_balance': self.frequency_balance,
            'temporal_consistency': self.temporal_consistency,
            'level_stability': self.level_stability,
            'has_clipping': self.has_clipping,
            'has_excessive_noise': self.has_excessive_noise,
            'has_dropouts': self.has_dropouts,
            'has_artifacts': self.has_artifacts
        }

@dataclass
class QualityResult:
    """Result of quality assessment"""
    timestamp: float
    metrics: QualityMetrics
    recommendations: List[str]
    confidence: float
    analysis_duration_ms: float
    
    def get_quality_summary(self) -> str:
        """Get human-readable quality summary"""
        level = self.metrics.quality_level.value.title()
        score = self.metrics.overall_quality * 100
        return f"{level} quality ({score:.1f}%)"
    
    def get_primary_issues(self) -> List[str]:
        """Get list of primary quality issues"""
        issues = []
        
        if self.metrics.has_clipping:
            issues.append("Audio clipping detected")
        
        if self.metrics.has_excessive_noise:
            issues.append("High background noise")
        
        if self.metrics.snr_db < 15:
            issues.append(f"Low SNR ({self.metrics.snr_db:.1f} dB)")
        
        if self.metrics.thd_percent > 10:
            issues.append(f"High distortion ({self.metrics.thd_percent:.1f}%)")
        
        if self.metrics.has_dropouts:
            issues.append("Audio dropouts detected")
        
        if self.metrics.temporal_consistency < 0.5:
            issues.append("Inconsistent audio quality")
        
        return issues

class QualityAssessor:
    """
    Comprehensive audio quality assessment system
    
    This assessor analyzes audio quality across multiple dimensions including
    SNR, distortion, clarity, and consistency for recording optimization.
    """
    
    def __init__(self, config: Optional[QualityConfig] = None):
        """
        Initialize Quality Assessor
        
        Args:
            config: Quality assessment configuration
        """
        self.config = config or QualityConfig()
        
        # State tracking
        self.is_running = False
        self.frame_buffer = np.array([], dtype=np.float64)
        self.quality_history = deque(maxlen=self.config.history_length)
        
        # Smoothed metrics
        self.smoothed_snr = None
        self.smoothed_clarity = None
        self.smoothed_level = None
        
        # Threading
        self.processing_thread = None
        self.thread_lock = threading.Lock()
        
        # Callbacks
        self.quality_callbacks: List[Callable[[QualityResult], None]] = []
        self.alert_callbacks: List[Callable[[str, float], None]] = []
        
        # Statistics
        self.stats = {
            'total_assessments': 0,
            'poor_quality_count': 0,
            'fair_quality_count': 0,
            'good_quality_count': 0,
            'excellent_quality_count': 0,
            'average_quality': 0.0,
            'clipping_detections': 0,
            'noise_detections': 0,
            'dropout_detections': 0,
            'start_time': None,
            'processing_time_ms': 0.0
        }
        
        logger.info("QualityAssessor initialized")
    
    def add_quality_callback(self, callback: Callable[[QualityResult], None]) -> None:
        """Add callback for quality results"""
        self.quality_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[str, float], None]) -> None:
        """Add callback for quality alerts"""
        self.alert_callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> bool:
        """Remove a callback from all callback lists"""
        removed = False
        for callback_list in [self.quality_callbacks, self.alert_callbacks]:
            if callback in callback_list:
                callback_list.remove(callback)
                removed = True
        return removed
    
    def _db_from_amplitude(self, amplitude: float, reference: float = 1.0) -> float:
        """Convert amplitude to dB scale"""
        if amplitude <= 0:
            return -float('inf')
        return 20 * math.log10(amplitude / reference)
    
    def _calculate_snr(self, audio_data: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        try:
            if len(audio_data) == 0:
                return 0.0
            
            # Simple energy-based SNR estimation
            # Divide signal into segments and find noise vs signal
            segment_size = len(audio_data) // 10
            if segment_size < 100:
                segment_size = len(audio_data)
            
            segment_energies = []
            for i in range(0, len(audio_data), segment_size):
                segment = audio_data[i:i + segment_size]
                if len(segment) > 0:
                    energy = np.mean(segment ** 2)
                    segment_energies.append(energy)
            
            if len(segment_energies) < 2:
                return 20.0  # Default reasonable SNR
            
            # Assume noise is the minimum energy segments
            sorted_energies = sorted(segment_energies)
            noise_energy = np.mean(sorted_energies[:len(sorted_energies)//3])  # Bottom third
            signal_energy = np.mean(sorted_energies)
            
            if noise_energy <= 0:
                return 40.0  # Very high SNR
            
            snr_linear = signal_energy / noise_energy
            snr_db = 10 * math.log10(snr_linear) if snr_linear > 0 else 0.0
            
            return float(np.clip(snr_db, 0.0, 60.0))
            
        except Exception as e:
            logger.error(f"Error calculating SNR: {e}")
            return 20.0
    
    def _calculate_thd(self, audio_data: np.ndarray) -> float:
        """Calculate total harmonic distortion"""
        try:
            if len(audio_data) < 1024:
                return 0.0
            
            # FFT-based harmonic analysis
            fft = np.fft.rfft(audio_data)
            magnitude_spectrum = np.abs(fft)
            
            # Find fundamental frequency (simplified)
            freqs = np.fft.rfftfreq(len(audio_data), 1.0 / self.config.sample_rate)
            
            # Look for peak in speech range (100-400 Hz)
            speech_range = (freqs >= 100) & (freqs <= 400)
            if np.any(speech_range):
                peak_idx = np.argmax(magnitude_spectrum[speech_range])
                fundamental_idx = np.where(speech_range)[0][peak_idx]
                fundamental_freq = freqs[fundamental_idx]
                fundamental_mag = magnitude_spectrum[fundamental_idx]
            else:
                return 0.0
            
            # Calculate harmonic distortion
            harmonic_power = 0.0
            for harmonic in range(2, 6):  # 2nd to 5th harmonics
                harmonic_freq = fundamental_freq * harmonic
                if harmonic_freq < freqs[-1]:
                    # Find closest frequency bin
                    harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
                    harmonic_power += magnitude_spectrum[harmonic_idx] ** 2
            
            if fundamental_mag == 0:
                return 0.0
            
            fundamental_power = fundamental_mag ** 2
            thd = math.sqrt(harmonic_power / fundamental_power) if fundamental_power > 0 else 0.0
            
            return float(np.clip(thd * 100, 0.0, 50.0))  # Convert to percentage
            
        except Exception as e:
            logger.error(f"Error calculating THD: {e}")
            return 0.0
    
    def _calculate_dynamic_range(self, audio_data: np.ndarray) -> float:
        """Calculate dynamic range in dB"""
        try:
            if len(audio_data) == 0:
                return 0.0
            
            max_amplitude = np.max(np.abs(audio_data))
            
            # Find noise floor (bottom 5% of amplitudes)
            sorted_amplitudes = np.sort(np.abs(audio_data))
            noise_floor_idx = int(len(sorted_amplitudes) * 0.05)
            noise_floor = np.mean(sorted_amplitudes[:noise_floor_idx]) if noise_floor_idx > 0 else 0.0
            
            if noise_floor <= 0 or max_amplitude <= 0:
                return 48.0  # Default good dynamic range
            
            dynamic_range = self._db_from_amplitude(max_amplitude) - self._db_from_amplitude(noise_floor)
            
            return float(np.clip(dynamic_range, 0.0, 96.0))
            
        except Exception as e:
            logger.error(f"Error calculating dynamic range: {e}")
            return 48.0
    
    def _calculate_clarity_score(self, audio_data: np.ndarray) -> Tuple[float, float, float]:
        """Calculate clarity scores (overall, spectral, temporal)"""
        try:
            if len(audio_data) == 0:
                return 0.5, 0.5, 0.5
            
            # Spectral clarity - energy concentration in speech frequencies
            fft = np.fft.rfft(audio_data)
            magnitude_spectrum = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_data), 1.0 / self.config.sample_rate)
            
            # Speech frequency range (300-3400 Hz)
            speech_range = (freqs >= 300) & (freqs <= 3400)
            total_energy = np.sum(magnitude_spectrum ** 2)
            speech_energy = np.sum(magnitude_spectrum[speech_range] ** 2)
            
            spectral_clarity = speech_energy / total_energy if total_energy > 0 else 0.0
            
            # Temporal clarity - signal variability and structure
            signal_var = np.var(audio_data)
            signal_mean = np.mean(np.abs(audio_data))
            temporal_clarity = min(signal_var / (signal_mean ** 2), 1.0) if signal_mean > 0 else 0.0
            
            # Overall clarity
            overall_clarity = (spectral_clarity + temporal_clarity) / 2.0
            
            return (float(np.clip(overall_clarity, 0.0, 1.0)),
                   float(np.clip(spectral_clarity, 0.0, 1.0)),
                   float(np.clip(temporal_clarity, 0.0, 1.0)))
            
        except Exception as e:
            logger.error(f"Error calculating clarity: {e}")
            return 0.5, 0.5, 0.5
    
    def _detect_clipping(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """Detect audio clipping"""
        try:
            if len(audio_data) == 0:
                return False, 0.0
            
            max_value = np.max(np.abs(audio_data))
            threshold = self.config.thresholds.clipping_threshold * 32767  # For 16-bit audio
            
            clipped_samples = np.sum(np.abs(audio_data) >= threshold)
            clipping_ratio = clipped_samples / len(audio_data)
            
            has_clipping = clipping_ratio > 0.01  # More than 1% clipped
            
            return has_clipping, float(clipping_ratio)
            
        except Exception as e:
            logger.error(f"Error detecting clipping: {e}")
            return False, 0.0
    
    def _calculate_frequency_balance(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Calculate energy balance across frequency bands"""
        try:
            if len(audio_data) == 0:
                return {f"band_{i}": 0.0 for i in range(len(self.config.frequency_bands))}
            
            fft = np.fft.rfft(audio_data)
            magnitude_spectrum = np.abs(fft) ** 2
            freqs = np.fft.rfftfreq(len(audio_data), 1.0 / self.config.sample_rate)
            
            total_energy = np.sum(magnitude_spectrum)
            balance = {}
            
            for i, (low_freq, high_freq) in enumerate(self.config.frequency_bands):
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_energy = np.sum(magnitude_spectrum[band_mask])
                band_ratio = band_energy / total_energy if total_energy > 0 else 0.0
                balance[f"band_{i}_{low_freq}_{high_freq}Hz"] = float(band_ratio)
            
            return balance
            
        except Exception as e:
            logger.error(f"Error calculating frequency balance: {e}")
            return {f"band_{i}": 0.0 for i in range(len(self.config.frequency_bands))}
    
    def _calculate_consistency_metrics(self, current_metrics: Dict[str, float]) -> Tuple[float, float]:
        """Calculate temporal consistency and level stability"""
        try:
            if len(self.quality_history) < 3:
                return 0.8, 0.8  # Default good values for new streams
            
            # Get recent quality scores
            recent_qualities = [q.metrics.overall_quality for q in list(self.quality_history)[-10:]]
            recent_levels = [q.metrics.rms_level_db for q in list(self.quality_history)[-10:]]
            
            # Temporal consistency - stability of quality metrics
            quality_std = np.std(recent_qualities)
            temporal_consistency = max(0.0, 1.0 - (quality_std * 2))  # Lower std = higher consistency
            
            # Level stability - stability of signal levels
            level_std = np.std(recent_levels)
            level_stability = max(0.0, 1.0 - (level_std / 20.0))  # Normalize by 20dB range
            
            return float(temporal_consistency), float(level_stability)
            
        except Exception as e:
            logger.error(f"Error calculating consistency: {e}")
            return 0.5, 0.5
    
    def _determine_quality_level(self, overall_quality: float, snr_db: float, 
                                thd_percent: float, clarity: float) -> QualityLevel:
        """Determine categorical quality level"""
        thresholds = self.config.thresholds
        
        # Check for excellent quality
        if (overall_quality >= 0.9 and snr_db >= thresholds.snr_excellent and
            thd_percent <= thresholds.thd_excellent and clarity >= thresholds.clarity_excellent):
            return QualityLevel.EXCELLENT
        
        # Check for good quality
        elif (overall_quality >= 0.7 and snr_db >= thresholds.snr_good and
              thd_percent <= thresholds.thd_good and clarity >= thresholds.clarity_good):
            return QualityLevel.GOOD
        
        # Check for fair quality
        elif (overall_quality >= 0.5 and snr_db >= thresholds.snr_fair and
              thd_percent <= thresholds.thd_fair and clarity >= thresholds.clarity_fair):
            return QualityLevel.FAIR
        
        # Otherwise poor quality
        else:
            return QualityLevel.POOR
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        if metrics.has_clipping:
            recommendations.append("Reduce input gain to prevent clipping")
        
        if metrics.snr_db < 20:
            recommendations.append("Reduce background noise or increase signal level")
        
        if metrics.thd_percent > 5:
            recommendations.append("Check for audio equipment distortion")
        
        if metrics.clarity_score < 0.6:
            recommendations.append("Improve microphone positioning or room acoustics")
        
        if metrics.has_excessive_noise:
            recommendations.append("Use noise reduction or move to quieter environment")
        
        if metrics.temporal_consistency < 0.5:
            recommendations.append("Check for audio interruptions or equipment issues")
        
        if metrics.dynamic_range_db < 20:
            recommendations.append("Increase dynamic range to improve audio quality")
        
        # Frequency balance recommendations
        speech_band_key = next((k for k in metrics.frequency_balance.keys() 
                              if "1000_4000" in k), None)
        if speech_band_key and metrics.frequency_balance[speech_band_key] < 0.3:
            recommendations.append("Boost mid-frequency range for better speech clarity")
        
        return recommendations
    
    def assess_quality(self, audio_data: np.ndarray, timestamp: Optional[float] = None) -> QualityResult:
        """
        Assess audio quality for given audio data
        
        Args:
            audio_data: Audio data as numpy array
            timestamp: Timestamp for the assessment
        
        Returns:
            QualityResult with comprehensive quality metrics
        """
        if timestamp is None:
            timestamp = time.time()
        
        start_time = time.time()
        
        try:
            # Convert to float for analysis
            if audio_data.dtype != np.float64:
                audio_float = audio_data.astype(np.float64) / 32767.0  # Normalize int16 to float
            else:
                audio_float = audio_data.copy()
            
            # Calculate core metrics
            snr_db = self._calculate_snr(audio_float)
            thd_percent = self._calculate_thd(audio_float)
            dynamic_range_db = self._calculate_dynamic_range(audio_float)
            
            # Calculate signal levels
            rms_level = np.sqrt(np.mean(audio_float ** 2)) if len(audio_float) > 0 else 0.0
            peak_level = np.max(np.abs(audio_float)) if len(audio_float) > 0 else 0.0
            
            peak_level_db = self._db_from_amplitude(peak_level) if peak_level > 0 else -float('inf')
            rms_level_db = self._db_from_amplitude(rms_level) if rms_level > 0 else -float('inf')
            
            # Calculate clarity metrics
            clarity_overall, spectral_clarity, temporal_clarity = self._calculate_clarity_score(audio_float)
            
            # Detect clipping
            has_clipping, clipping_ratio = self._detect_clipping(audio_data)
            
            # Calculate noise floor
            sorted_amplitudes = np.sort(np.abs(audio_float))
            noise_floor_idx = int(len(sorted_amplitudes) * 0.1)  # Bottom 10%
            noise_floor = np.mean(sorted_amplitudes[:noise_floor_idx]) if noise_floor_idx > 0 else 0.0
            noise_floor_db = self._db_from_amplitude(noise_floor) if noise_floor > 0 else -float('inf')
            
            # Calculate frequency balance
            frequency_balance = self._calculate_frequency_balance(audio_float)
            
            # Calculate consistency metrics
            current_metrics = {
                'snr': snr_db,
                'clarity': clarity_overall,
                'rms_level': rms_level_db
            }
            temporal_consistency, level_stability = self._calculate_consistency_metrics(current_metrics)
            
            # Apply smoothing to key metrics
            if self.config.enable_real_time:
                alpha = self.config.smoothing_factor
                self.smoothed_snr = (snr_db if self.smoothed_snr is None 
                                   else alpha * snr_db + (1 - alpha) * self.smoothed_snr)
                self.smoothed_clarity = (clarity_overall if self.smoothed_clarity is None
                                       else alpha * clarity_overall + (1 - alpha) * self.smoothed_clarity)
                self.smoothed_level = (rms_level_db if self.smoothed_level is None
                                     else alpha * rms_level_db + (1 - alpha) * self.smoothed_level)
            
            # Detect quality issues
            has_excessive_noise = snr_db < 15
            has_dropouts = rms_level < 0.001  # Very low signal level
            has_artifacts = thd_percent > 10 or temporal_consistency < 0.3
            
            # Calculate overall quality score
            snr_score = np.clip(snr_db / 40.0, 0.0, 1.0)  # Normalize SNR to 0-1
            thd_score = np.clip(1.0 - (thd_percent / 20.0), 0.0, 1.0)  # Invert THD
            clarity_score = clarity_overall
            consistency_score = temporal_consistency
            
            overall_quality = (snr_score + thd_score + clarity_score + consistency_score) / 4.0
            
            # Apply penalties for severe issues
            if has_clipping:
                overall_quality *= 0.7
            if has_excessive_noise:
                overall_quality *= 0.8
            if has_dropouts:
                overall_quality *= 0.5
            
            overall_quality = float(np.clip(overall_quality, 0.0, 1.0))
            
            # Determine quality level
            quality_level = self._determine_quality_level(overall_quality, snr_db, thd_percent, clarity_overall)
            
            # Create metrics object
            metrics = QualityMetrics(
                timestamp=timestamp,
                overall_quality=overall_quality,
                quality_level=quality_level,
                snr_db=snr_db,
                thd_percent=thd_percent,
                dynamic_range_db=dynamic_range_db,
                peak_level_db=peak_level_db,
                rms_level_db=rms_level_db,
                clarity_score=clarity_overall,
                spectral_clarity=spectral_clarity,
                temporal_clarity=temporal_clarity,
                clipping_ratio=clipping_ratio,
                noise_floor_db=noise_floor_db,
                frequency_balance=frequency_balance,
                temporal_consistency=temporal_consistency,
                level_stability=level_stability,
                has_clipping=has_clipping,
                has_excessive_noise=has_excessive_noise,
                has_dropouts=has_dropouts,
                has_artifacts=has_artifacts
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(metrics)
            
            # Calculate confidence
            confidence = min(overall_quality + 0.2, 1.0)  # Higher quality = higher confidence
            
            # Create result
            analysis_duration = (time.time() - start_time) * 1000
            result = QualityResult(
                timestamp=timestamp,
                metrics=metrics,
                recommendations=recommendations,
                confidence=confidence,
                analysis_duration_ms=analysis_duration
            )
            
            # Update statistics
            with self.thread_lock:
                self.stats['total_assessments'] += 1
                self.stats['processing_time_ms'] += analysis_duration
                
                if quality_level == QualityLevel.POOR:
                    self.stats['poor_quality_count'] += 1
                elif quality_level == QualityLevel.FAIR:
                    self.stats['fair_quality_count'] += 1
                elif quality_level == QualityLevel.GOOD:
                    self.stats['good_quality_count'] += 1
                elif quality_level == QualityLevel.EXCELLENT:
                    self.stats['excellent_quality_count'] += 1
                
                if has_clipping:
                    self.stats['clipping_detections'] += 1
                if has_excessive_noise:
                    self.stats['noise_detections'] += 1
                if has_dropouts:
                    self.stats['dropout_detections'] += 1
                
                # Update running average
                total = self.stats['total_assessments']
                self.stats['average_quality'] = ((self.stats['average_quality'] * (total - 1) + 
                                                overall_quality) / total)
                
                if self.stats['start_time'] is None:
                    self.stats['start_time'] = timestamp
            
            # Store in history
            self.quality_history.append(result)
            
            # Fire callbacks
            for callback in self.quality_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in quality callback: {e}")
            
            # Fire alert callbacks for serious issues
            if quality_level == QualityLevel.POOR:
                for callback in self.alert_callbacks:
                    try:
                        callback("Poor audio quality detected", overall_quality)
                    except Exception as e:
                        logger.error(f"Error in alert callback: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            # Return safe default result
            return QualityResult(
                timestamp=timestamp,
                metrics=QualityMetrics(
                    timestamp=timestamp,
                    overall_quality=0.5,
                    quality_level=QualityLevel.FAIR,
                    snr_db=20.0,
                    thd_percent=5.0,
                    dynamic_range_db=40.0,
                    peak_level_db=0.0,
                    rms_level_db=-20.0,
                    clarity_score=0.5,
                    spectral_clarity=0.5,
                    temporal_clarity=0.5,
                    clipping_ratio=0.0,
                    noise_floor_db=-40.0,
                    frequency_balance={},
                    temporal_consistency=0.5,
                    level_stability=0.5,
                    has_clipping=False,
                    has_excessive_noise=False,
                    has_dropouts=False,
                    has_artifacts=False
                ),
                recommendations=["Quality assessment failed - check audio input"],
                confidence=0.0,
                analysis_duration_ms=(time.time() - start_time) * 1000
            )
    
    def get_quality_history(self) -> List[QualityResult]:
        """Get quality assessment history"""
        return list(self.quality_history)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get quality assessment statistics"""
        with self.thread_lock:
            stats = self.stats.copy()
            
            # Add derived statistics
            if stats['total_assessments'] > 0:
                stats['poor_percentage'] = (stats['poor_quality_count'] / stats['total_assessments']) * 100
                stats['fair_percentage'] = (stats['fair_quality_count'] / stats['total_assessments']) * 100
                stats['good_percentage'] = (stats['good_quality_count'] / stats['total_assessments']) * 100
                stats['excellent_percentage'] = (stats['excellent_quality_count'] / stats['total_assessments']) * 100
                stats['average_processing_time_ms'] = stats['processing_time_ms'] / stats['total_assessments']
            else:
                stats['poor_percentage'] = 0.0
                stats['fair_percentage'] = 0.0
                stats['good_percentage'] = 0.0
                stats['excellent_percentage'] = 0.0
                stats['average_processing_time_ms'] = 0.0
            
            # Current smoothed values
            stats['current_smoothed_snr'] = self.smoothed_snr
            stats['current_smoothed_clarity'] = self.smoothed_clarity
            stats['current_smoothed_level'] = self.smoothed_level
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset quality assessment statistics"""
        with self.thread_lock:
            self.stats = {
                'total_assessments': 0,
                'poor_quality_count': 0,
                'fair_quality_count': 0,
                'good_quality_count': 0,
                'excellent_quality_count': 0,
                'average_quality': 0.0,
                'clipping_detections': 0,
                'noise_detections': 0,
                'dropout_detections': 0,
                'start_time': None,
                'processing_time_ms': 0.0
            }
            
            self.quality_history.clear()
            self.smoothed_snr = None
            self.smoothed_clarity = None
            self.smoothed_level = None
        
        logger.info("Quality assessment statistics reset")
    
    def start_processing(self) -> bool:
        """Start quality assessment processing"""
        if self.is_running:
            logger.warning("Quality assessment already running")
            return False
        
        with self.thread_lock:
            self.is_running = True
            self.stats['start_time'] = time.time()
        
        logger.info("Quality assessment started")
        return True
    
    def stop_processing(self) -> bool:
        """Stop quality assessment processing"""
        if not self.is_running:
            logger.warning("Quality assessment not running")
            return False
        
        with self.thread_lock:
            self.is_running = False
        
        logger.info("Quality assessment stopped")
        return True
    
    def is_processing(self) -> bool:
        """Check if quality assessment is currently processing"""
        return self.is_running

def create_quality_assessor(config: Optional[QualityConfig] = None) -> QualityAssessor:
    """
    Factory function to create a configured quality assessor
    
    Args:
        config: Optional quality assessment configuration
        
    Returns:
        Configured QualityAssessor instance
    """
    return QualityAssessor(config)

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create quality assessor
    config = QualityConfig(
        sample_rate=16000,
        frame_duration_ms=100,
        enable_real_time=True
    )
    
    assessor = create_quality_assessor(config)
    
    # Add example callbacks
    def on_quality_result(result: QualityResult):
        print(f"Quality: {result.get_quality_summary()}")
        if result.get_primary_issues():
            print(f"  Issues: {', '.join(result.get_primary_issues())}")
    
    def on_quality_alert(message: str, quality: float):
        print(f"ALERT: {message} (quality: {quality:.2f})")
    
    assessor.add_quality_callback(on_quality_result)
    assessor.add_alert_callback(on_quality_alert)
    
    # Start processing
    assessor.start_processing()
    
    # Test with synthetic audio data
    try:
        print("Testing quality assessor with synthetic audio...")
        
        frame_size = 1600  # 100ms at 16kHz
        
        # Test different quality scenarios
        for i in range(20):
            if i < 5:
                # Good quality audio
                audio_frame = np.random.normal(0, 1000, frame_size).astype(np.int16)
                # Add some speech-like frequency content
                t = np.linspace(0, 0.1, frame_size)
                audio_frame += (2000 * np.sin(2 * np.pi * 1000 * t)).astype(np.int16)
            elif i < 10:
                # Noisy audio
                audio_frame = np.random.normal(0, 3000, frame_size).astype(np.int16)
                signal = (1000 * np.sin(2 * np.pi * 1000 * t)).astype(np.int16)
                audio_frame = signal + audio_frame
            elif i < 15:
                # Clipped audio
                audio_frame = np.random.normal(0, 1000, frame_size).astype(np.int16)
                audio_frame = np.clip(audio_frame, -30000, 30000)  # Heavy clipping
            else:
                # Low quality/distorted audio
                audio_frame = np.random.normal(0, 500, frame_size).astype(np.int16)
                # Add harmonics for distortion
                t = np.linspace(0, 0.1, frame_size)
                for harmonic in [2, 3, 4]:
                    audio_frame += (200 * np.sin(2 * np.pi * 440 * harmonic * t)).astype(np.int16)
            
            result = assessor.assess_quality(audio_frame)
            time.sleep(0.1)  # 100ms delay
        
        # Print statistics
        stats = assessor.get_statistics()
        print(f"\nStatistics:")
        print(f"Total assessments: {stats['total_assessments']}")
        print(f"Average quality: {stats['average_quality']:.2f}")
        print(f"Excellent: {stats['excellent_percentage']:.1f}%")
        print(f"Good: {stats['good_percentage']:.1f}%")
        print(f"Fair: {stats['fair_percentage']:.1f}%")
        print(f"Poor: {stats['poor_percentage']:.1f}%")
        print(f"Clipping detections: {stats['clipping_detections']}")
        print(f"Noise detections: {stats['noise_detections']}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    finally:
        assessor.stop_processing()
        print("Quality assessment test completed")