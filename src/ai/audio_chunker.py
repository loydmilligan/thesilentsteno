#!/usr/bin/env python3
"""
AI Audio Chunker Module

AI-optimized audio chunking for transcription with intelligent segmentation,
quality analysis, and performance optimization for real-time processing.

Author: Claude AI Assistant
Date: 2024-07-14
Version: 1.0
"""

import os
import sys
import logging
import time
import threading
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import uuid
import math

try:
    import numpy as np
    import soundfile as sf
    from scipy import signal
    from scipy.ndimage import uniform_filter1d
    import librosa
except ImportError as e:
    print(f"Warning: Required dependencies not installed: {e}")
    print("Please install with: pip install numpy scipy soundfile librosa")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Types of audio chunks"""
    SPEECH = "speech"
    SILENCE = "silence"
    NOISE = "noise"
    MUSIC = "music"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ChunkQuality(Enum):
    """Quality levels for chunks"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class TranscriptionChunkConfig:
    """Configuration for AI audio chunking"""
    
    # Basic chunking parameters
    chunk_duration: float = 10.0  # Default chunk duration in seconds
    overlap_duration: float = 0.5  # Overlap between chunks in seconds
    
    # Size constraints
    min_chunk_duration: float = 1.0  # Minimum chunk size
    max_chunk_duration: float = 30.0  # Maximum chunk size
    
    # Audio processing
    sample_rate: int = 16000  # Target sample rate
    normalize_audio: bool = True  # Normalize audio levels
    
    # Voice Activity Detection
    enable_vad: bool = True  # Enable voice activity detection
    vad_threshold: float = 0.3  # Voice activity threshold
    vad_window_size: int = 400  # VAD window size in samples
    
    # Quality analysis
    enable_quality_analysis: bool = True  # Enable quality scoring
    snr_threshold: float = 10.0  # Signal-to-noise ratio threshold
    silence_threshold: float = 0.01  # Silence detection threshold
    
    # Optimization settings
    enable_smart_segmentation: bool = True  # Enable intelligent segmentation
    prefer_speech_boundaries: bool = True  # Prefer speech/silence boundaries
    enable_adaptive_chunking: bool = True  # Adapt chunk size to content
    
    # Performance settings
    enable_caching: bool = True  # Enable chunk caching
    cache_size: int = 100  # Maximum cached chunks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "chunk_duration": self.chunk_duration,
            "overlap_duration": self.overlap_duration,
            "min_chunk_duration": self.min_chunk_duration,
            "max_chunk_duration": self.max_chunk_duration,
            "sample_rate": self.sample_rate,
            "normalize_audio": self.normalize_audio,
            "enable_vad": self.enable_vad,
            "vad_threshold": self.vad_threshold,
            "vad_window_size": self.vad_window_size,
            "enable_quality_analysis": self.enable_quality_analysis,
            "snr_threshold": self.snr_threshold,
            "silence_threshold": self.silence_threshold,
            "enable_smart_segmentation": self.enable_smart_segmentation,
            "prefer_speech_boundaries": self.prefer_speech_boundaries,
            "enable_adaptive_chunking": self.enable_adaptive_chunking,
            "enable_caching": self.enable_caching,
            "cache_size": self.cache_size
        }


@dataclass
class AudioAnalysis:
    """Audio analysis results"""
    
    # Basic properties
    duration: float
    sample_rate: int
    channels: int
    
    # Signal properties
    rms_level: float
    peak_level: float
    snr_estimate: float
    
    # Voice activity
    voice_activity: float  # Percentage of voice activity
    silence_ratio: float  # Percentage of silence
    
    # Spectral properties
    spectral_centroid: float
    spectral_rolloff: float
    zero_crossing_rate: float
    
    # Quality metrics
    quality_score: float  # Overall quality score (0-1)
    chunk_type: ChunkType
    chunk_quality: ChunkQuality
    
    # Segmentation hints
    speech_boundaries: List[Tuple[float, float]] = field(default_factory=list)
    optimal_split_points: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary"""
        return {
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "rms_level": self.rms_level,
            "peak_level": self.peak_level,
            "snr_estimate": self.snr_estimate,
            "voice_activity": self.voice_activity,
            "silence_ratio": self.silence_ratio,
            "spectral_centroid": self.spectral_centroid,
            "spectral_rolloff": self.spectral_rolloff,
            "zero_crossing_rate": self.zero_crossing_rate,
            "quality_score": self.quality_score,
            "chunk_type": self.chunk_type.value,
            "chunk_quality": self.chunk_quality.value,
            "speech_boundaries": self.speech_boundaries,
            "optimal_split_points": self.optimal_split_points
        }


@dataclass
class OptimalChunk:
    """Optimized audio chunk for transcription"""
    
    # Chunk identification
    chunk_id: str
    chunk_index: int = 0
    
    # Timing information
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    
    # Audio data
    audio_data: np.ndarray = field(default_factory=lambda: np.array([]))
    sample_rate: int = 16000
    
    # Quality metrics
    confidence: float = 0.0  # Confidence in chunk quality
    quality_score: float = 0.0  # Overall quality score
    snr_estimate: float = 0.0  # Signal-to-noise ratio
    
    # Content analysis
    voice_activity: float = 0.0  # Voice activity percentage
    silence_ratio: float = 0.0  # Silence percentage
    chunk_type: str = "unknown"  # Type of content
    
    # Optimization metadata
    optimization_score: float = 0.0  # How well optimized the chunk is
    split_reason: str = "duration"  # Why the chunk was split here
    
    # Processing hints
    transcription_priority: int = 1  # Priority for transcription (1=highest)
    expected_accuracy: float = 0.0  # Expected transcription accuracy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "confidence": self.confidence,
            "quality_score": self.quality_score,
            "snr_estimate": self.snr_estimate,
            "voice_activity": self.voice_activity,
            "silence_ratio": self.silence_ratio,
            "chunk_type": self.chunk_type,
            "optimization_score": self.optimization_score,
            "split_reason": self.split_reason,
            "transcription_priority": self.transcription_priority,
            "expected_accuracy": self.expected_accuracy
        }


class AIAudioChunker:
    """AI-powered audio chunker for optimal transcription"""
    
    def __init__(self, config: Optional[TranscriptionChunkConfig] = None):
        self.config = config or TranscriptionChunkConfig()
        self.is_initialized = False
        
        # Caching
        self.chunk_cache = {} if self.config.enable_caching else None
        
        # Statistics
        self.stats = {
            "total_chunks_created": 0,
            "total_audio_processed": 0.0,
            "total_processing_time": 0.0,
            "average_quality_score": 0.0,
            "average_voice_activity": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Threading
        self.processing_lock = threading.Lock()
        
        logger.info(f"AIAudioChunker initialized with {self.config.chunk_duration}s chunks")
        
    def initialize(self) -> bool:
        """Initialize the audio chunker"""
        try:
            logger.info("Initializing AI audio chunker...")
            
            # Test required libraries
            test_audio = np.random.randn(1000)
            _ = librosa.feature.rms(y=test_audio)
            _ = librosa.feature.spectral_centroid(y=test_audio, sr=16000)
            
            self.is_initialized = True
            logger.info("AI audio chunker initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize audio chunker: {e}")
            return False
            
    def analyze_audio(self, audio_data: np.ndarray, sample_rate: int) -> AudioAnalysis:
        """Analyze audio for optimal chunking"""
        try:
            # Ensure mono audio
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                
            # Normalize to [-1, 1]
            if self.config.normalize_audio:
                audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
                
            # Basic properties
            duration = len(audio_data) / sample_rate
            channels = 1
            
            # Signal level analysis
            rms_level = np.sqrt(np.mean(audio_data**2))
            peak_level = np.max(np.abs(audio_data))
            
            # Voice Activity Detection
            voice_activity, silence_ratio = self._detect_voice_activity(audio_data, sample_rate)
            
            # Spectral analysis
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate
            ))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(
                y=audio_data, sr=sample_rate
            ))
            
            # Zero crossing rate
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data))
            
            # SNR estimation
            snr_estimate = self._estimate_snr(audio_data)
            
            # Quality scoring
            quality_score = self._calculate_quality_score(
                rms_level, peak_level, snr_estimate, voice_activity, zero_crossing_rate
            )
            
            # Content classification
            chunk_type = self._classify_content(
                voice_activity, silence_ratio, spectral_centroid, zero_crossing_rate
            )
            
            # Quality classification
            chunk_quality = self._classify_quality(quality_score, snr_estimate)
            
            # Find optimal split points
            optimal_split_points = self._find_optimal_split_points(
                audio_data, sample_rate
            )
            
            # Find speech boundaries
            speech_boundaries = self._find_speech_boundaries(
                audio_data, sample_rate
            )
            
            return AudioAnalysis(
                duration=duration,
                sample_rate=sample_rate,
                channels=channels,
                rms_level=rms_level,
                peak_level=peak_level,
                snr_estimate=snr_estimate,
                voice_activity=voice_activity,
                silence_ratio=silence_ratio,
                spectral_centroid=spectral_centroid,
                spectral_rolloff=spectral_rolloff,
                zero_crossing_rate=zero_crossing_rate,
                quality_score=quality_score,
                chunk_type=chunk_type,
                chunk_quality=chunk_quality,
                speech_boundaries=speech_boundaries,
                optimal_split_points=optimal_split_points
            )
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            
            # Return minimal analysis
            return AudioAnalysis(
                duration=len(audio_data) / sample_rate,
                sample_rate=sample_rate,
                channels=1,
                rms_level=0.0,
                peak_level=0.0,
                snr_estimate=0.0,
                voice_activity=0.0,
                silence_ratio=1.0,
                spectral_centroid=0.0,
                spectral_rolloff=0.0,
                zero_crossing_rate=0.0,
                quality_score=0.0,
                chunk_type=ChunkType.UNKNOWN,
                chunk_quality=ChunkQuality.UNUSABLE
            )
            
    def chunk_audio_for_transcription(self, audio_data: np.ndarray, 
                                    sample_rate: int, 
                                    start_time: float = 0.0) -> List[OptimalChunk]:
        """Create optimized chunks for transcription"""
        if not self.is_initialized:
            logger.error("Chunker not initialized")
            return []
            
        process_start = time.time()
        
        try:
            with self.processing_lock:
                # Resample if needed
                if sample_rate != self.config.sample_rate:
                    audio_data = librosa.resample(
                        audio_data, orig_sr=sample_rate, target_sr=self.config.sample_rate
                    )
                    sample_rate = self.config.sample_rate
                    
                # Analyze audio
                analysis = self.analyze_audio(audio_data, sample_rate)
                
                # Check cache
                cache_key = self._generate_cache_key(audio_data, sample_rate)
                if self.chunk_cache and cache_key in self.chunk_cache:
                    self.stats["cache_hits"] += 1
                    return self.chunk_cache[cache_key]
                    
                if self.chunk_cache:
                    self.stats["cache_misses"] += 1
                    
                # Create chunks
                chunks = self._create_optimal_chunks(
                    audio_data, sample_rate, analysis, start_time
                )
                
                # Cache results
                if self.chunk_cache:
                    self.chunk_cache[cache_key] = chunks
                    
                    # Limit cache size
                    if len(self.chunk_cache) > self.config.cache_size:
                        oldest_key = next(iter(self.chunk_cache))
                        del self.chunk_cache[oldest_key]
                        
                # Update statistics
                self._update_stats(chunks, time.time() - process_start)
                
                return chunks
                
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            return []
            
    def _detect_voice_activity(self, audio_data: np.ndarray, 
                             sample_rate: int) -> Tuple[float, float]:
        """Detect voice activity in audio"""
        try:
            # Simple energy-based VAD
            frame_length = self.config.vad_window_size
            hop_length = frame_length // 4
            
            # Calculate frame energy
            frames = librosa.util.frame(
                audio_data, frame_length=frame_length, hop_length=hop_length
            )
            frame_energy = np.mean(frames**2, axis=0)
            
            # Threshold-based detection
            energy_threshold = np.mean(frame_energy) * self.config.vad_threshold
            voice_frames = frame_energy > energy_threshold
            
            # Calculate percentages
            voice_activity = np.mean(voice_frames) * 100
            silence_ratio = 100 - voice_activity
            
            return voice_activity, silence_ratio
            
        except Exception as e:
            logger.error(f"Voice activity detection failed: {e}")
            return 0.0, 100.0
            
    def _estimate_snr(self, audio_data: np.ndarray) -> float:
        """Estimate signal-to-noise ratio"""
        try:
            # Simple SNR estimation using energy distribution
            frame_length = 1024
            hop_length = 512
            
            # Calculate frame energies
            frames = librosa.util.frame(
                audio_data, frame_length=frame_length, hop_length=hop_length
            )
            frame_energies = np.mean(frames**2, axis=0)
            
            # Estimate signal and noise
            sorted_energies = np.sort(frame_energies)
            noise_level = np.mean(sorted_energies[:len(sorted_energies)//4])  # Bottom 25%
            signal_level = np.mean(sorted_energies[-len(sorted_energies)//4:])  # Top 25%
            
            # Calculate SNR in dB
            if noise_level > 0:
                snr_db = 10 * np.log10(signal_level / noise_level)
            else:
                snr_db = 40  # High SNR if no noise detected
                
            return max(0, snr_db)
            
        except Exception as e:
            logger.error(f"SNR estimation failed: {e}")
            return 0.0
            
    def _calculate_quality_score(self, rms_level: float, peak_level: float, 
                               snr_estimate: float, voice_activity: float, 
                               zero_crossing_rate: float) -> float:
        """Calculate overall quality score"""
        try:
            # Normalize metrics
            level_score = min(1.0, rms_level / 0.3)  # Normalize to 0.3 RMS
            snr_score = min(1.0, snr_estimate / 30.0)  # Normalize to 30 dB
            voice_score = voice_activity / 100.0  # Already percentage
            
            # Dynamic range score
            dynamic_range = peak_level - rms_level
            range_score = min(1.0, dynamic_range / 0.5)
            
            # Zero crossing rate score (lower is better for speech)
            zcr_score = max(0.0, 1.0 - zero_crossing_rate / 0.1)
            
            # Weighted combination
            quality_score = (
                0.25 * level_score +
                0.30 * snr_score +
                0.25 * voice_score +
                0.10 * range_score +
                0.10 * zcr_score
            )
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return 0.0
            
    def _classify_content(self, voice_activity: float, silence_ratio: float,
                        spectral_centroid: float, zero_crossing_rate: float) -> ChunkType:
        """Classify content type"""
        try:
            if voice_activity > 60:
                return ChunkType.SPEECH
            elif silence_ratio > 80:
                return ChunkType.SILENCE
            elif zero_crossing_rate > 0.1:
                return ChunkType.NOISE
            elif spectral_centroid > 3000:
                return ChunkType.MUSIC
            else:
                return ChunkType.MIXED
                
        except Exception as e:
            logger.error(f"Content classification failed: {e}")
            return ChunkType.UNKNOWN
            
    def _classify_quality(self, quality_score: float, snr_estimate: float) -> ChunkQuality:
        """Classify quality level"""
        try:
            if quality_score > 0.8 and snr_estimate > 20:
                return ChunkQuality.EXCELLENT
            elif quality_score > 0.6 and snr_estimate > 15:
                return ChunkQuality.GOOD
            elif quality_score > 0.4 and snr_estimate > 10:
                return ChunkQuality.FAIR
            elif quality_score > 0.2 and snr_estimate > 5:
                return ChunkQuality.POOR
            else:
                return ChunkQuality.UNUSABLE
                
        except Exception as e:
            logger.error(f"Quality classification failed: {e}")
            return ChunkQuality.UNUSABLE
            
    def _find_optimal_split_points(self, audio_data: np.ndarray, 
                                 sample_rate: int) -> List[float]:
        """Find optimal points to split audio"""
        try:
            # Simple implementation: find low-energy points
            frame_length = 1024
            hop_length = 512
            
            # Calculate frame energies
            frames = librosa.util.frame(
                audio_data, frame_length=frame_length, hop_length=hop_length
            )
            frame_energies = np.mean(frames**2, axis=0)
            
            # Smooth energy curve
            smoothed_energy = uniform_filter1d(frame_energies, size=5)
            
            # Find local minima
            min_indices = signal.argrelmin(smoothed_energy, order=3)[0]
            
            # Convert to time
            split_times = []
            for idx in min_indices:
                time_point = idx * hop_length / sample_rate
                if (time_point > self.config.min_chunk_duration and 
                    time_point < len(audio_data) / sample_rate - self.config.min_chunk_duration):
                    split_times.append(time_point)
                    
            return split_times
            
        except Exception as e:
            logger.error(f"Finding split points failed: {e}")
            return []
            
    def _find_speech_boundaries(self, audio_data: np.ndarray, 
                              sample_rate: int) -> List[Tuple[float, float]]:
        """Find speech segment boundaries"""
        try:
            # Simple speech boundary detection
            frame_length = self.config.vad_window_size
            hop_length = frame_length // 4
            
            # Calculate frame energy
            frames = librosa.util.frame(
                audio_data, frame_length=frame_length, hop_length=hop_length
            )
            frame_energy = np.mean(frames**2, axis=0)
            
            # Threshold-based detection
            energy_threshold = np.mean(frame_energy) * self.config.vad_threshold
            voice_frames = frame_energy > energy_threshold
            
            # Find boundaries
            boundaries = []
            in_speech = False
            speech_start = 0
            
            for i, is_voice in enumerate(voice_frames):
                time_point = i * hop_length / sample_rate
                
                if is_voice and not in_speech:
                    # Start of speech
                    speech_start = time_point
                    in_speech = True
                elif not is_voice and in_speech:
                    # End of speech
                    boundaries.append((speech_start, time_point))
                    in_speech = False
                    
            # Close final boundary if needed
            if in_speech:
                boundaries.append((speech_start, len(audio_data) / sample_rate))
                
            return boundaries
            
        except Exception as e:
            logger.error(f"Finding speech boundaries failed: {e}")
            return []
            
    def _create_optimal_chunks(self, audio_data: np.ndarray, sample_rate: int,
                             analysis: AudioAnalysis, start_time: float) -> List[OptimalChunk]:
        """Create optimal chunks based on analysis"""
        try:
            chunks = []
            duration = len(audio_data) / sample_rate
            
            # Determine chunk strategy
            if self.config.enable_smart_segmentation and analysis.optimal_split_points:
                chunk_points = self._create_smart_chunks(
                    audio_data, sample_rate, analysis, start_time
                )
            else:
                chunk_points = self._create_fixed_chunks(
                    duration, start_time
                )
                
            # Create chunk objects
            for i, (chunk_start, chunk_end) in enumerate(chunk_points):
                chunk_audio = self._extract_audio_chunk(
                    audio_data, sample_rate, chunk_start - start_time, chunk_end - start_time
                )
                
                if len(chunk_audio) > 0:
                    chunk = self._create_chunk_object(
                        chunk_audio, sample_rate, chunk_start, chunk_end, i, analysis
                    )
                    chunks.append(chunk)
                    
            return chunks
            
        except Exception as e:
            logger.error(f"Creating optimal chunks failed: {e}")
            return []
            
    def _create_smart_chunks(self, audio_data: np.ndarray, sample_rate: int,
                           analysis: AudioAnalysis, start_time: float) -> List[Tuple[float, float]]:
        """Create chunks using smart segmentation"""
        try:
            chunks = []
            duration = len(audio_data) / sample_rate
            
            # Use speech boundaries if available
            if self.config.prefer_speech_boundaries and analysis.speech_boundaries:
                current_time = start_time
                
                for speech_start, speech_end in analysis.speech_boundaries:
                    speech_duration = speech_end - speech_start
                    
                    if speech_duration > self.config.max_chunk_duration:
                        # Split long speech segments
                        sub_chunks = self._split_long_segment(
                            speech_start, speech_end, self.config.chunk_duration
                        )
                        chunks.extend(sub_chunks)
                    elif speech_duration >= self.config.min_chunk_duration:
                        # Use speech segment as chunk
                        chunks.append((speech_start + start_time, speech_end + start_time))
                        
            else:
                # Use optimal split points
                split_points = [0.0] + analysis.optimal_split_points + [duration]
                
                for i in range(len(split_points) - 1):
                    chunk_start = split_points[i] + start_time
                    chunk_end = split_points[i + 1] + start_time
                    
                    # Ensure minimum chunk size
                    if chunk_end - chunk_start >= self.config.min_chunk_duration:
                        chunks.append((chunk_start, chunk_end))
                        
            return chunks
            
        except Exception as e:
            logger.error(f"Smart chunking failed: {e}")
            return self._create_fixed_chunks(len(audio_data) / sample_rate, start_time)
            
    def _create_fixed_chunks(self, duration: float, start_time: float) -> List[Tuple[float, float]]:
        """Create fixed-size chunks"""
        chunks = []
        current_time = start_time
        
        while current_time < start_time + duration:
            chunk_end = min(current_time + self.config.chunk_duration, start_time + duration)
            
            # Ensure minimum chunk size
            if chunk_end - current_time >= self.config.min_chunk_duration:
                chunks.append((current_time, chunk_end))
                
            current_time = chunk_end - self.config.overlap_duration
            
        return chunks
        
    def _split_long_segment(self, start: float, end: float, 
                          target_duration: float) -> List[Tuple[float, float]]:
        """Split long segments into smaller chunks"""
        chunks = []
        current_time = start
        
        while current_time < end:
            chunk_end = min(current_time + target_duration, end)
            chunks.append((current_time, chunk_end))
            current_time = chunk_end - self.config.overlap_duration
            
        return chunks
        
    def _extract_audio_chunk(self, audio_data: np.ndarray, sample_rate: int,
                           start_time: float, end_time: float) -> np.ndarray:
        """Extract audio chunk from data"""
        try:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Ensure valid indices
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)
            
            if start_sample >= end_sample:
                return np.array([])
                
            return audio_data[start_sample:end_sample]
            
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return np.array([])
            
    def _create_chunk_object(self, audio_data: np.ndarray, sample_rate: int,
                           start_time: float, end_time: float, index: int,
                           analysis: AudioAnalysis) -> OptimalChunk:
        """Create optimized chunk object"""
        try:
            # Analyze chunk
            chunk_analysis = self.analyze_audio(audio_data, sample_rate)
            
            # Calculate optimization score
            optimization_score = self._calculate_optimization_score(
                chunk_analysis, analysis
            )
            
            # Determine transcription priority
            priority = self._calculate_transcription_priority(chunk_analysis)
            
            # Estimate expected accuracy
            expected_accuracy = self._estimate_transcription_accuracy(chunk_analysis)
            
            return OptimalChunk(
                chunk_id=str(uuid.uuid4()),
                chunk_index=index,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                audio_data=audio_data,
                sample_rate=sample_rate,
                confidence=chunk_analysis.quality_score,
                quality_score=chunk_analysis.quality_score,
                snr_estimate=chunk_analysis.snr_estimate,
                voice_activity=chunk_analysis.voice_activity,
                silence_ratio=chunk_analysis.silence_ratio,
                chunk_type=chunk_analysis.chunk_type.value,
                optimization_score=optimization_score,
                split_reason="smart_segmentation" if self.config.enable_smart_segmentation else "fixed_duration",
                transcription_priority=priority,
                expected_accuracy=expected_accuracy
            )
            
        except Exception as e:
            logger.error(f"Chunk object creation failed: {e}")
            
            # Return minimal chunk
            return OptimalChunk(
                chunk_id=str(uuid.uuid4()),
                chunk_index=index,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                audio_data=audio_data,
                sample_rate=sample_rate
            )
            
    def _calculate_optimization_score(self, chunk_analysis: AudioAnalysis,
                                    overall_analysis: AudioAnalysis) -> float:
        """Calculate how well optimized the chunk is"""
        try:
            # Compare chunk to overall audio
            quality_ratio = chunk_analysis.quality_score / (overall_analysis.quality_score + 1e-8)
            voice_ratio = chunk_analysis.voice_activity / (overall_analysis.voice_activity + 1e-8)
            
            # Optimal size score
            duration_score = 1.0
            if chunk_analysis.duration < self.config.min_chunk_duration:
                duration_score = chunk_analysis.duration / self.config.min_chunk_duration
            elif chunk_analysis.duration > self.config.max_chunk_duration:
                duration_score = self.config.max_chunk_duration / chunk_analysis.duration
                
            # Combine scores
            optimization_score = (
                0.4 * quality_ratio +
                0.3 * voice_ratio +
                0.3 * duration_score
            )
            
            return min(1.0, max(0.0, optimization_score))
            
        except Exception as e:
            logger.error(f"Optimization score calculation failed: {e}")
            return 0.5
            
    def _calculate_transcription_priority(self, analysis: AudioAnalysis) -> int:
        """Calculate transcription priority (1=highest, 5=lowest)"""
        try:
            if analysis.chunk_type == ChunkType.SPEECH and analysis.quality_score > 0.8:
                return 1
            elif analysis.chunk_type == ChunkType.SPEECH and analysis.quality_score > 0.6:
                return 2
            elif analysis.chunk_type == ChunkType.MIXED and analysis.voice_activity > 50:
                return 3
            elif analysis.chunk_type == ChunkType.NOISE and analysis.quality_score > 0.4:
                return 4
            else:
                return 5
                
        except Exception as e:
            logger.error(f"Priority calculation failed: {e}")
            return 3
            
    def _estimate_transcription_accuracy(self, analysis: AudioAnalysis) -> float:
        """Estimate expected transcription accuracy"""
        try:
            # Base accuracy on quality and content type
            base_accuracy = analysis.quality_score
            
            # Adjust for content type
            if analysis.chunk_type == ChunkType.SPEECH:
                content_multiplier = 1.0
            elif analysis.chunk_type == ChunkType.MIXED:
                content_multiplier = 0.8
            elif analysis.chunk_type == ChunkType.NOISE:
                content_multiplier = 0.3
            else:
                content_multiplier = 0.5
                
            # Adjust for SNR
            snr_multiplier = min(1.0, analysis.snr_estimate / 20.0)
            
            # Adjust for voice activity
            voice_multiplier = analysis.voice_activity / 100.0
            
            expected_accuracy = base_accuracy * content_multiplier * snr_multiplier * voice_multiplier
            
            return min(1.0, max(0.0, expected_accuracy))
            
        except Exception as e:
            logger.error(f"Accuracy estimation failed: {e}")
            return 0.5
            
    def _generate_cache_key(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Generate cache key for audio data"""
        try:
            # Use hash of audio data and parameters
            audio_hash = hash(audio_data.tobytes())
            config_hash = hash(str(self.config.to_dict()))
            
            return f"{audio_hash}_{sample_rate}_{config_hash}"
            
        except Exception as e:
            logger.error(f"Cache key generation failed: {e}")
            return str(uuid.uuid4())
            
    def _update_stats(self, chunks: List[OptimalChunk], processing_time: float):
        """Update chunking statistics"""
        try:
            self.stats["total_chunks_created"] += len(chunks)
            self.stats["total_processing_time"] += processing_time
            
            if chunks:
                total_duration = sum(chunk.duration for chunk in chunks)
                self.stats["total_audio_processed"] += total_duration
                
                # Update average quality
                chunk_count = self.stats["total_chunks_created"]
                old_quality = self.stats["average_quality_score"]
                avg_chunk_quality = sum(chunk.quality_score for chunk in chunks) / len(chunks)
                
                self.stats["average_quality_score"] = (
                    old_quality * (chunk_count - len(chunks)) + avg_chunk_quality * len(chunks)
                ) / chunk_count
                
                # Update average voice activity
                old_voice = self.stats["average_voice_activity"]
                avg_voice_activity = sum(chunk.voice_activity for chunk in chunks) / len(chunks)
                
                self.stats["average_voice_activity"] = (
                    old_voice * (chunk_count - len(chunks)) + avg_voice_activity * len(chunks)
                ) / chunk_count
                
        except Exception as e:
            logger.error(f"Stats update failed: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get chunking statistics"""
        return self.stats.copy()
        
    def get_status(self) -> Dict[str, Any]:
        """Get chunker status"""
        return {
            "is_initialized": self.is_initialized,
            "config": self.config.to_dict(),
            "stats": self.get_stats(),
            "cache_size": len(self.chunk_cache) if self.chunk_cache else 0
        }
        
    def shutdown(self):
        """Shutdown chunker"""
        logger.info("Shutting down AI audio chunker...")
        
        # Clear cache
        if self.chunk_cache:
            self.chunk_cache.clear()
            
        self.is_initialized = False
        logger.info("AI audio chunker shutdown complete")
        
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


# Factory functions
def create_default_chunker() -> AIAudioChunker:
    """Create default audio chunker"""
    return AIAudioChunker()


def create_realtime_chunker() -> AIAudioChunker:
    """Create chunker optimized for real-time processing"""
    config = TranscriptionChunkConfig(
        chunk_duration=5.0,
        overlap_duration=0.2,
        min_chunk_duration=0.5,
        enable_smart_segmentation=True,
        prefer_speech_boundaries=True,
        enable_adaptive_chunking=True
    )
    return AIAudioChunker(config)


def create_quality_chunker() -> AIAudioChunker:
    """Create chunker optimized for quality"""
    config = TranscriptionChunkConfig(
        chunk_duration=20.0,
        overlap_duration=1.0,
        min_chunk_duration=2.0,
        enable_quality_analysis=True,
        enable_smart_segmentation=True,
        prefer_speech_boundaries=True,
        snr_threshold=15.0
    )
    return AIAudioChunker(config)


# Testing and demonstration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Audio Chunker Test")
    parser.add_argument("--audio", type=str, required=True, help="Audio file to chunk")
    parser.add_argument("--chunk-size", type=float, default=10.0, help="Chunk size in seconds")
    parser.add_argument("--output", type=str, help="Output directory for chunks")
    args = parser.parse_args()
    
    # Create chunker
    chunker = create_realtime_chunker()
    chunker.config.chunk_duration = args.chunk_size
    
    try:
        # Initialize
        if not chunker.initialize():
            print("Failed to initialize chunker")
            sys.exit(1)
            
        print(f"Chunker status: {chunker.get_status()}")
        
        # Load audio
        print(f"Loading audio: {args.audio}")
        audio_data, sample_rate = sf.read(args.audio)
        
        # Chunk audio
        print(f"Chunking audio...")
        chunks = chunker.chunk_audio_for_transcription(audio_data, sample_rate)
        
        if chunks:
            print(f"Created {len(chunks)} chunks:")
            for i, chunk in enumerate(chunks):
                print(f"  Chunk {i}: {chunk.start_time:.2f}s-{chunk.end_time:.2f}s "
                      f"({chunk.duration:.2f}s) - Quality: {chunk.quality_score:.3f}, "
                      f"Voice: {chunk.voice_activity:.1f}%, Type: {chunk.chunk_type}")
                
                # Save chunk if output directory specified
                if args.output:
                    output_path = Path(args.output)
                    output_path.mkdir(exist_ok=True)
                    chunk_file = output_path / f"chunk_{i:03d}.wav"
                    sf.write(chunk_file, chunk.audio_data, chunk.sample_rate)
                    
            print(f"\nChunking statistics: {chunker.get_stats()}")
        else:
            print("No chunks created")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        chunker.shutdown()