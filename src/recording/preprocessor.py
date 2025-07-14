#!/usr/bin/env python3

"""
Audio Preprocessor for The Silent Steno

This module provides comprehensive audio preprocessing and enhancement
capabilities for improving recording quality and preparing audio for
AI transcription. It includes noise reduction, normalization, speech
enhancement, and quality assessment functions.

Key features:
- Real-time and batch audio preprocessing
- Noise reduction using spectral subtraction
- Audio normalization and gain control
- Speech enhancement and voice isolation
- Audio quality metrics and assessment
- Adaptive processing based on audio characteristics
- Integration with recording pipeline
"""

import numpy as np
import logging
import threading
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import warnings

try:
    import scipy.signal
    import scipy.fft
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available, limited preprocessing support")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not available, advanced features disabled")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class ProcessingMode(Enum):
    """Audio processing modes"""
    REAL_TIME = "real_time"      # Low-latency processing
    BALANCED = "balanced"        # Good quality/performance balance
    HIGH_QUALITY = "high_quality"  # Best quality processing
    SPEECH_OPTIMIZED = "speech_optimized"  # Optimized for speech


class NoiseProfile(Enum):
    """Noise profile types"""
    OFFICE = "office"
    OUTDOOR = "outdoor"
    VEHICLE = "vehicle"
    ELECTRONICS = "electronics"
    CROWD = "crowd"
    ADAPTIVE = "adaptive"


@dataclass
class ProcessingConfig:
    """Audio preprocessing configuration"""
    mode: ProcessingMode = ProcessingMode.BALANCED
    enable_noise_reduction: bool = True
    enable_normalization: bool = True
    enable_speech_enhancement: bool = True
    enable_adaptive_processing: bool = True
    noise_profile: NoiseProfile = NoiseProfile.ADAPTIVE
    target_db: float = -20.0  # Target normalization level
    noise_gate_threshold: float = -50.0  # dB
    sample_rate: int = 44100
    frame_size: int = 2048
    overlap_factor: float = 0.5


@dataclass
class QualityMetrics:
    """Audio quality assessment metrics"""
    snr_db: float  # Signal-to-noise ratio
    thd_percent: float  # Total harmonic distortion
    dynamic_range_db: float  # Dynamic range
    speech_clarity: float  # Speech clarity score (0-1)
    noise_level_db: float  # Background noise level
    clipping_percent: float  # Percentage of clipped samples
    spectral_centroid: float  # Spectral centroid (Hz)
    overall_quality: float  # Overall quality score (0-1)


class AudioPreprocessor:
    """
    Audio Preprocessor for The Silent Steno
    
    Provides comprehensive audio enhancement and quality improvement
    for both real-time and batch processing scenarios.
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize audio preprocessor"""
        self.config = config or ProcessingConfig()
        
        # Processing state
        self.processing_lock = threading.Lock()
        self.noise_profile_data = None
        self.adaptation_history = []
        self.adaptation_window = 100  # frames
        
        # Performance tracking
        self.performance_stats = {
            "frames_processed": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "quality_improvements": 0,
            "noise_reduction_applied": 0
        }
        
        # Callbacks
        self.quality_callbacks: List[Callable] = []
        self.processing_callbacks: List[Callable] = []
        
        # Initialize processing components
        self._initialize_processors()
        
        logger.info(f"Audio preprocessor initialized with mode: {self.config.mode.value}")
    
    def add_quality_callback(self, callback: Callable[[QualityMetrics], None]) -> None:
        """Add callback for quality metrics updates"""
        self.quality_callbacks.append(callback)
    
    def add_processing_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for processing events"""
        self.processing_callbacks.append(callback)
    
    def _notify_quality_update(self, metrics: QualityMetrics) -> None:
        """Notify callbacks of quality metrics"""
        for callback in self.quality_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Error in quality callback: {e}")
    
    def _notify_processing_event(self, event: Dict[str, Any]) -> None:
        """Notify callbacks of processing events"""
        for callback in self.processing_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in processing callback: {e}")
    
    def _initialize_processors(self) -> None:
        """Initialize processing components"""
        try:
            # Pre-calculate windowing functions
            self.window = np.hanning(self.config.frame_size)
            
            # Initialize noise reduction parameters
            self.noise_reduction_alpha = 0.9  # Noise profile adaptation rate
            self.noise_floor_estimate = None
            
            # Speech enhancement parameters
            self.speech_band_low = 300   # Hz
            self.speech_band_high = 3400  # Hz
            
            # Adaptive processing state
            self.current_snr = 0.0
            self.processing_history = []
            
            logger.debug("Processing components initialized")
        
        except Exception as e:
            logger.error(f"Error initializing processors: {e}")
    
    def apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction to audio data
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Noise-reduced audio data
        """
        try:
            if not self.config.enable_noise_reduction or not SCIPY_AVAILABLE:
                return audio_data
            
            # Ensure proper shape
            if len(audio_data.shape) == 1:
                # Mono audio
                return self._apply_noise_reduction_mono(audio_data)
            else:
                # Multi-channel audio
                processed_channels = []
                for channel in range(audio_data.shape[1]):
                    processed_channel = self._apply_noise_reduction_mono(audio_data[:, channel])
                    processed_channels.append(processed_channel)
                return np.column_stack(processed_channels)
        
        except Exception as e:
            logger.error(f"Error in noise reduction: {e}")
            return audio_data
    
    def _apply_noise_reduction_mono(self, audio_mono: np.ndarray) -> np.ndarray:
        """Apply noise reduction to mono audio"""
        try:
            if len(audio_mono) < self.config.frame_size:
                return audio_mono
            
            # Compute STFT
            frame_size = self.config.frame_size
            hop_size = int(frame_size * (1 - self.config.overlap_factor))
            
            # Simple spectral subtraction implementation
            stft = self._compute_stft(audio_mono, frame_size, hop_size)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise floor
            if self.noise_floor_estimate is None:
                # Use first few frames as noise estimate
                noise_frames = min(10, magnitude.shape[1])
                self.noise_floor_estimate = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Adaptive noise floor update
            current_noise_estimate = np.percentile(magnitude, 20, axis=1, keepdims=True)
            self.noise_floor_estimate = (
                self.noise_reduction_alpha * self.noise_floor_estimate +
                (1 - self.noise_reduction_alpha) * current_noise_estimate
            )
            
            # Apply spectral subtraction
            noise_factor = 2.0  # Adjustable noise reduction strength
            spectral_floor = 0.1  # Prevent over-suppression
            
            suppression_gain = 1 - noise_factor * (self.noise_floor_estimate / (magnitude + 1e-10))
            suppression_gain = np.maximum(suppression_gain, spectral_floor)
            
            # Apply suppression
            enhanced_magnitude = magnitude * suppression_gain
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            
            # Reconstruct audio
            enhanced_audio = self._compute_istft(enhanced_stft, frame_size, hop_size)
            
            # Ensure same length as input
            if len(enhanced_audio) > len(audio_mono):
                enhanced_audio = enhanced_audio[:len(audio_mono)]
            elif len(enhanced_audio) < len(audio_mono):
                enhanced_audio = np.pad(enhanced_audio, (0, len(audio_mono) - len(enhanced_audio)))
            
            return enhanced_audio
        
        except Exception as e:
            logger.error(f"Error in mono noise reduction: {e}")
            return audio_mono
    
    def _compute_stft(self, audio: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
        """Compute Short-Time Fourier Transform"""
        try:
            if SCIPY_AVAILABLE:
                _, _, stft = scipy.signal.stft(
                    audio, 
                    nperseg=frame_size, 
                    noverlap=frame_size-hop_size,
                    window='hann'
                )
                return stft
            else:
                # Simple STFT implementation
                n_frames = (len(audio) - frame_size) // hop_size + 1
                stft = np.zeros((frame_size // 2 + 1, n_frames), dtype=complex)
                
                for i in range(n_frames):
                    start = i * hop_size
                    frame = audio[start:start + frame_size] * self.window
                    fft_frame = np.fft.rfft(frame)
                    stft[:, i] = fft_frame
                
                return stft
        
        except Exception as e:
            logger.error(f"Error computing STFT: {e}")
            return np.array([[]], dtype=complex)
    
    def _compute_istft(self, stft: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
        """Compute Inverse Short-Time Fourier Transform"""
        try:
            if SCIPY_AVAILABLE:
                _, audio = scipy.signal.istft(
                    stft, 
                    nperseg=frame_size, 
                    noverlap=frame_size-hop_size,
                    window='hann'
                )
                return audio
            else:
                # Simple ISTFT implementation
                n_frames = stft.shape[1]
                audio_length = (n_frames - 1) * hop_size + frame_size
                audio = np.zeros(audio_length)
                
                for i in range(n_frames):
                    start = i * hop_size
                    frame = np.fft.irfft(stft[:, i], frame_size)
                    frame *= self.window
                    audio[start:start + frame_size] += frame
                
                return audio
        
        except Exception as e:
            logger.error(f"Error computing ISTFT: {e}")
            return np.array([])
    
    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio to target level
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Normalized audio data
        """
        try:
            if not self.config.enable_normalization:
                return audio_data
            
            # Calculate current RMS level
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            if rms == 0:
                return audio_data
            
            # Convert target dB to linear scale
            target_linear = 10 ** (self.config.target_db / 20)
            
            # Calculate gain needed
            gain = target_linear / rms
            
            # Apply gentle limiting to prevent clipping
            max_gain = 0.95 / np.max(np.abs(audio_data))
            gain = min(gain, max_gain)
            
            # Apply gain
            normalized_audio = audio_data * gain
            
            return normalized_audio
        
        except Exception as e:
            logger.error(f"Error in normalization: {e}")
            return audio_data
    
    def enhance_speech(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply speech enhancement
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Speech-enhanced audio data
        """
        try:
            if not self.config.enable_speech_enhancement or not SCIPY_AVAILABLE:
                return audio_data
            
            # Apply speech frequency emphasis
            enhanced_audio = self._apply_speech_emphasis(audio_data)
            
            # Apply dynamic range compression for speech
            enhanced_audio = self._apply_speech_compression(enhanced_audio)
            
            return enhanced_audio
        
        except Exception as e:
            logger.error(f"Error in speech enhancement: {e}")
            return audio_data
    
    def _apply_speech_emphasis(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply frequency emphasis for speech"""
        try:
            if len(audio_data.shape) == 1:
                return self._apply_speech_emphasis_mono(audio_data)
            else:
                # Multi-channel
                enhanced_channels = []
                for channel in range(audio_data.shape[1]):
                    enhanced_channel = self._apply_speech_emphasis_mono(audio_data[:, channel])
                    enhanced_channels.append(enhanced_channel)
                return np.column_stack(enhanced_channels)
        
        except Exception as e:
            logger.error(f"Error in speech emphasis: {e}")
            return audio_data
    
    def _apply_speech_emphasis_mono(self, audio_mono: np.ndarray) -> np.ndarray:
        """Apply speech emphasis to mono audio"""
        try:
            # Simple high-pass filter to emphasize speech frequencies
            nyquist = self.config.sample_rate / 2
            low_cutoff = self.speech_band_low / nyquist
            
            if SCIPY_AVAILABLE:
                sos = scipy.signal.butter(2, low_cutoff, btype='high', output='sos')
                enhanced_audio = scipy.signal.sosfilt(sos, audio_mono)
                return enhanced_audio
            else:
                # Simple high-pass approximation
                alpha = 0.97
                enhanced_audio = np.zeros_like(audio_mono)
                enhanced_audio[0] = audio_mono[0]
                for i in range(1, len(audio_mono)):
                    enhanced_audio[i] = alpha * enhanced_audio[i-1] + alpha * (audio_mono[i] - audio_mono[i-1])
                return enhanced_audio
        
        except Exception as e:
            logger.error(f"Error in mono speech emphasis: {e}")
            return audio_mono
    
    def _apply_speech_compression(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression optimized for speech"""
        try:
            # Simple soft compression
            threshold = 0.7
            ratio = 3.0
            
            # Apply compression
            compressed_audio = np.where(
                np.abs(audio_data) > threshold,
                threshold + (np.abs(audio_data) - threshold) / ratio,
                np.abs(audio_data)
            ) * np.sign(audio_data)
            
            return compressed_audio
        
        except Exception as e:
            logger.error(f"Error in speech compression: {e}")
            return audio_data
    
    def get_quality_metrics(self, audio_data: np.ndarray) -> QualityMetrics:
        """
        Calculate comprehensive audio quality metrics
        
        Args:
            audio_data: Audio data to analyze
            
        Returns:
            Quality metrics
        """
        try:
            # Basic metrics
            snr = self._calculate_snr(audio_data)
            thd = self._calculate_thd(audio_data)
            dynamic_range = self._calculate_dynamic_range(audio_data)
            noise_level = self._calculate_noise_level(audio_data)
            clipping = self._calculate_clipping_percent(audio_data)
            
            # Advanced metrics
            speech_clarity = self._calculate_speech_clarity(audio_data)
            spectral_centroid = self._calculate_spectral_centroid(audio_data)
            
            # Overall quality score
            overall_quality = self._calculate_overall_quality(
                snr, thd, dynamic_range, speech_clarity, clipping
            )
            
            metrics = QualityMetrics(
                snr_db=snr,
                thd_percent=thd,
                dynamic_range_db=dynamic_range,
                speech_clarity=speech_clarity,
                noise_level_db=noise_level,
                clipping_percent=clipping,
                spectral_centroid=spectral_centroid,
                overall_quality=overall_quality
            )
            
            # Notify callbacks
            self._notify_quality_update(metrics)
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return QualityMetrics(
                snr_db=0.0, thd_percent=0.0, dynamic_range_db=0.0,
                speech_clarity=0.0, noise_level_db=-60.0, clipping_percent=0.0,
                spectral_centroid=1000.0, overall_quality=0.5
            )
    
    def _calculate_snr(self, audio_data: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio"""
        try:
            # Simple SNR estimation
            signal_power = np.mean(audio_data ** 2)
            noise_power = np.var(audio_data - np.mean(audio_data))
            
            if noise_power == 0:
                return 60.0  # Very high SNR
            
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(max(snr_linear, 1e-10))
            
            return max(min(snr_db, 60.0), -20.0)  # Clamp to reasonable range
        
        except Exception:
            return 20.0  # Default reasonable SNR
    
    def _calculate_thd(self, audio_data: np.ndarray) -> float:
        """Calculate Total Harmonic Distortion"""
        try:
            if not SCIPY_AVAILABLE or len(audio_data) < 1024:
                return 1.0  # Default low distortion
            
            # Simple THD estimation using FFT
            fft_data = np.fft.rfft(audio_data)
            magnitude = np.abs(fft_data)
            
            # Find fundamental frequency (simplified)
            fundamental_idx = np.argmax(magnitude[10:]) + 10  # Skip DC and very low frequencies
            fundamental_power = magnitude[fundamental_idx] ** 2
            
            # Sum harmonic powers (rough approximation)
            total_power = np.sum(magnitude ** 2)
            harmonic_power = total_power - fundamental_power
            
            if fundamental_power == 0:
                return 1.0
            
            thd = np.sqrt(harmonic_power / fundamental_power) * 100
            return min(thd, 50.0)  # Cap at reasonable maximum
        
        except Exception:
            return 1.0  # Default low distortion
    
    def _calculate_dynamic_range(self, audio_data: np.ndarray) -> float:
        """Calculate dynamic range"""
        try:
            peak = np.max(np.abs(audio_data))
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            if rms == 0:
                return 60.0  # Maximum dynamic range
            
            dynamic_range = 20 * np.log10(peak / rms)
            return max(min(dynamic_range, 60.0), 0.0)
        
        except Exception:
            return 20.0  # Default reasonable range
    
    def _calculate_noise_level(self, audio_data: np.ndarray) -> float:
        """Calculate background noise level"""
        try:
            # Use percentile to estimate noise floor
            noise_level = np.percentile(np.abs(audio_data), 10)
            noise_db = 20 * np.log10(max(noise_level, 1e-10))
            return max(min(noise_db, 0.0), -80.0)
        
        except Exception:
            return -50.0  # Default noise level
    
    def _calculate_clipping_percent(self, audio_data: np.ndarray) -> float:
        """Calculate percentage of clipped samples"""
        try:
            clipping_threshold = 0.99
            clipped_samples = np.sum(np.abs(audio_data) >= clipping_threshold)
            clipping_percent = (clipped_samples / len(audio_data)) * 100
            return min(clipping_percent, 100.0)
        
        except Exception:
            return 0.0
    
    def _calculate_speech_clarity(self, audio_data: np.ndarray) -> float:
        """Calculate speech clarity score"""
        try:
            if not SCIPY_AVAILABLE:
                return 0.7  # Default reasonable clarity
            
            # Simple speech clarity based on frequency content
            fft_data = np.fft.rfft(audio_data)
            frequencies = np.fft.rfftfreq(len(audio_data), 1/self.config.sample_rate)
            magnitude = np.abs(fft_data)
            
            # Speech frequency band energy
            speech_mask = (frequencies >= self.speech_band_low) & (frequencies <= self.speech_band_high)
            speech_energy = np.sum(magnitude[speech_mask])
            total_energy = np.sum(magnitude)
            
            if total_energy == 0:
                return 0.5
            
            clarity = speech_energy / total_energy
            return max(min(clarity, 1.0), 0.0)
        
        except Exception:
            return 0.7  # Default reasonable clarity
    
    def _calculate_spectral_centroid(self, audio_data: np.ndarray) -> float:
        """Calculate spectral centroid"""
        try:
            if LIBROSA_AVAILABLE:
                centroid = librosa.feature.spectral_centroid(
                    y=audio_data, sr=self.config.sample_rate
                )[0]
                return float(np.mean(centroid))
            else:
                # Simple centroid calculation
                fft_data = np.fft.rfft(audio_data)
                frequencies = np.fft.rfftfreq(len(audio_data), 1/self.config.sample_rate)
                magnitude = np.abs(fft_data)
                
                centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)
                return float(centroid)
        
        except Exception:
            return 1000.0  # Default centroid
    
    def _calculate_overall_quality(self, snr: float, thd: float, dynamic_range: float,
                                 speech_clarity: float, clipping: float) -> float:
        """Calculate overall quality score"""
        try:
            # Normalize individual metrics to 0-1 scale
            snr_score = max(min((snr + 20) / 40, 1.0), 0.0)  # -20dB to 20dB
            thd_score = max(min((10 - thd) / 10, 1.0), 0.0)  # 0% to 10%
            dr_score = max(min(dynamic_range / 30, 1.0), 0.0)  # 0dB to 30dB
            clarity_score = speech_clarity
            clipping_score = max(min((5 - clipping) / 5, 1.0), 0.0)  # 0% to 5%
            
            # Weighted combination
            overall = (
                snr_score * 0.3 +
                thd_score * 0.2 +
                dr_score * 0.2 +
                clarity_score * 0.2 +
                clipping_score * 0.1
            )
            
            return max(min(overall, 1.0), 0.0)
        
        except Exception:
            return 0.5  # Default neutral quality
    
    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply complete audio preprocessing pipeline
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Processed audio data
        """
        try:
            start_time = time.time()
            
            with self.processing_lock:
                # Store original for comparison
                original_data = audio_data.copy()
                processed_data = audio_data.copy()
                
                # Apply processing steps based on configuration
                if self.config.enable_noise_reduction:
                    processed_data = self.apply_noise_reduction(processed_data)
                
                if self.config.enable_speech_enhancement:
                    processed_data = self.enhance_speech(processed_data)
                
                if self.config.enable_normalization:
                    processed_data = self.normalize_audio(processed_data)
                
                # Apply noise gate
                processed_data = self._apply_noise_gate(processed_data)
                
                # Adaptive processing
                if self.config.enable_adaptive_processing:
                    processed_data = self._apply_adaptive_processing(processed_data, original_data)
                
                # Update performance statistics
                processing_time = time.time() - start_time
                self._update_performance_stats(processing_time)
                
                return processed_data
        
        except Exception as e:
            logger.error(f"Error in audio processing: {e}")
            return audio_data
    
    def _apply_noise_gate(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply noise gate to reduce low-level noise"""
        try:
            threshold_linear = 10 ** (self.config.noise_gate_threshold / 20)
            
            # Calculate envelope
            if len(audio_data.shape) == 1:
                envelope = np.abs(audio_data)
            else:
                envelope = np.max(np.abs(audio_data), axis=1)
            
            # Apply gate
            gate_mask = envelope > threshold_linear
            
            if len(audio_data.shape) == 1:
                gated_audio = audio_data * gate_mask
            else:
                gated_audio = audio_data * gate_mask[:, np.newaxis]
            
            return gated_audio
        
        except Exception as e:
            logger.error(f"Error in noise gate: {e}")
            return audio_data
    
    def _apply_adaptive_processing(self, processed_data: np.ndarray, 
                                 original_data: np.ndarray) -> np.ndarray:
        """Apply adaptive processing based on audio characteristics"""
        try:
            # Calculate quality improvement
            original_quality = self.get_quality_metrics(original_data)
            processed_quality = self.get_quality_metrics(processed_data)
            
            quality_improvement = processed_quality.overall_quality - original_quality.overall_quality
            
            # Adapt processing strength based on improvement
            if quality_improvement > 0.1:
                # Good improvement, increase processing strength
                self.performance_stats['quality_improvements'] += 1
            elif quality_improvement < -0.05:
                # Processing made things worse, reduce strength
                blend_factor = 0.7  # Blend more with original
                processed_data = blend_factor * processed_data + (1 - blend_factor) * original_data
            
            return processed_data
        
        except Exception as e:
            logger.error(f"Error in adaptive processing: {e}")
            return processed_data
    
    def _update_performance_stats(self, processing_time: float) -> None:
        """Update performance statistics"""
        try:
            self.performance_stats['frames_processed'] += 1
            self.performance_stats['total_processing_time'] += processing_time
            
            if self.performance_stats['frames_processed'] > 0:
                self.performance_stats['avg_processing_time'] = (
                    self.performance_stats['total_processing_time'] / 
                    self.performance_stats['frames_processed']
                )
        
        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get preprocessing performance statistics"""
        return {
            **self.performance_stats,
            'config': {
                'mode': self.config.mode.value,
                'noise_reduction_enabled': self.config.enable_noise_reduction,
                'normalization_enabled': self.config.enable_normalization,
                'speech_enhancement_enabled': self.config.enable_speech_enhancement,
                'adaptive_processing_enabled': self.config.enable_adaptive_processing
            },
            'capabilities': {
                'scipy_available': SCIPY_AVAILABLE,
                'librosa_available': LIBROSA_AVAILABLE
            }
        }


if __name__ == "__main__":
    # Basic test when run directly
    print("Audio Preprocessor Test")
    print("=" * 50)
    
    config = ProcessingConfig(
        mode=ProcessingMode.BALANCED,
        enable_noise_reduction=True,
        enable_normalization=True,
        enable_speech_enhancement=True
    )
    
    preprocessor = AudioPreprocessor(config)
    
    def on_quality_update(metrics):
        print(f"Quality: SNR {metrics.snr_db:.1f}dB, Clarity {metrics.speech_clarity:.2f}, Overall {metrics.overall_quality:.2f}")
    
    preprocessor.add_quality_callback(on_quality_update)
    
    # Generate test audio with noise
    print("Generating test audio with noise...")
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Speech-like signal (multiple frequencies)
    speech_signal = (
        0.5 * np.sin(2 * np.pi * 800 * t) +
        0.3 * np.sin(2 * np.pi * 1200 * t) +
        0.2 * np.sin(2 * np.pi * 2000 * t)
    )
    
    # Add noise
    noise = np.random.normal(0, 0.1, len(speech_signal))
    noisy_audio = speech_signal + noise
    
    print("Processing audio...")
    
    # Get original quality
    original_metrics = preprocessor.get_quality_metrics(noisy_audio)
    print(f"Original quality: {original_metrics.overall_quality:.3f}")
    
    # Process audio
    processed_audio = preprocessor.process_audio(noisy_audio)
    
    # Get processed quality
    processed_metrics = preprocessor.get_quality_metrics(processed_audio)
    print(f"Processed quality: {processed_metrics.overall_quality:.3f}")
    
    # Performance stats
    stats = preprocessor.get_performance_stats()
    print(f"Processing time: {stats['avg_processing_time']*1000:.2f}ms")
    print(f"Quality improvement: {processed_metrics.overall_quality - original_metrics.overall_quality:.3f}")
    
    print("Test complete!")