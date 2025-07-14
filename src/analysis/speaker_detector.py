#!/usr/bin/env python3
"""
Speaker Detection and Change Detection for The Silent Steno

This module provides speaker change detection for diarization in multi-participant
meetings, enabling proper speaker attribution in transcripts.

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
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

class SpeakerChangeMethod(Enum):
    """Methods for detecting speaker changes"""
    MFCC = "mfcc"                    # MFCC-based features
    SPECTRAL = "spectral"            # Spectral features
    COMBINED = "combined"            # Combined feature approach
    EMBEDDING = "embedding"          # Speaker embedding approach

class SpeakerConfidence(Enum):
    """Confidence levels for speaker detection"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.9

@dataclass
class SpeakerConfig:
    """Configuration for Speaker Detector"""
    method: SpeakerChangeMethod = SpeakerChangeMethod.COMBINED
    sample_rate: int = 16000
    frame_duration_ms: int = 100  # Analysis frame duration
    window_size: int = 20  # Number of frames for analysis window
    change_threshold: float = 0.3  # Threshold for speaker change detection
    min_speaker_duration_ms: int = 500  # Minimum speaker segment duration
    max_speakers: int = 10  # Maximum number of speakers to track
    feature_dimension: int = 13  # MFCC feature dimension
    clustering_update_interval: int = 50  # Frames between clustering updates
    confidence_threshold: float = 0.6  # Minimum confidence for speaker assignment
    similarity_threshold: float = 0.7  # Similarity threshold for speaker matching
    voice_activity_required: bool = True  # Require voice activity for analysis
    adaptation_rate: float = 0.1  # Rate of speaker model adaptation
    
    def __post_init__(self):
        """Validate configuration parameters"""
        self.chunk_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        
        # Validate thresholds
        self.change_threshold = max(0.1, min(0.9, self.change_threshold))
        self.confidence_threshold = max(0.1, min(0.9, self.confidence_threshold))
        self.similarity_threshold = max(0.1, min(0.9, self.similarity_threshold))

@dataclass
class SpeakerFeatures:
    """Speaker audio features for analysis"""
    timestamp: float
    mfcc: np.ndarray
    spectral_centroid: float
    spectral_rolloff: float
    zero_crossing_rate: float
    energy: float
    pitch_mean: float
    pitch_std: float
    formants: List[float]
    voice_activity: bool
    
    def to_vector(self) -> np.ndarray:
        """Convert features to vector for analysis"""
        features = list(self.mfcc)
        features.extend([
            self.spectral_centroid,
            self.spectral_rolloff,
            self.zero_crossing_rate,
            self.energy,
            self.pitch_mean,
            self.pitch_std
        ])
        features.extend(self.formants[:3])  # First 3 formants
        return np.array(features)

@dataclass
class SpeakerResult:
    """Result of speaker detection analysis"""
    timestamp: float
    speaker_id: Optional[str]
    confidence: float
    is_change: bool
    change_confidence: float
    previous_speaker_id: Optional[str]
    features: SpeakerFeatures
    segment_duration_ms: Optional[float] = None
    analysis_method: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'timestamp': self.timestamp,
            'speaker_id': self.speaker_id,
            'confidence': self.confidence,
            'is_change': self.is_change,
            'change_confidence': self.change_confidence,
            'previous_speaker_id': self.previous_speaker_id,
            'segment_duration_ms': self.segment_duration_ms,
            'analysis_method': self.analysis_method,
            'voice_activity': self.features.voice_activity
        }

@dataclass
class SpeakerChangeDetection:
    """Information about detected speaker changes"""
    timestamp: float
    from_speaker: Optional[str]
    to_speaker: str
    confidence: float
    method: str
    feature_distance: float

class SpeakerDetector:
    """
    Speaker change detection for multi-participant meeting diarization
    
    This detector analyzes audio features to identify when different speakers
    are talking, enabling proper speaker attribution in transcripts.
    """
    
    def __init__(self, config: Optional[SpeakerConfig] = None):
        """
        Initialize Speaker Detector
        
        Args:
            config: Speaker detection configuration
        """
        self.config = config or SpeakerConfig()
        
        # State tracking
        self.is_running = False
        self.current_speaker_id = None
        self.speaker_models: Dict[str, np.ndarray] = {}
        self.feature_buffer = deque(maxlen=self.config.window_size)
        self.frame_count = 0
        
        # Speaker tracking
        self.speaker_counter = 0
        self.speaker_segments: List[Dict[str, Any]] = []
        self.current_segment_start = None
        
        # Machine learning models
        self.kmeans_model = None
        self.feature_scaler = None
        
        # Threading
        self.processing_thread = None
        self.thread_lock = threading.Lock()
        
        # Callbacks
        self.speaker_change_callbacks: List[Callable[[SpeakerChangeDetection], None]] = []
        self.result_callbacks: List[Callable[[SpeakerResult], None]] = []
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'speaker_changes': 0,
            'total_speakers': 0,
            'segments_processed': 0,
            'start_time': None,
            'processing_time': 0.0
        }
        
        logger.info(f"SpeakerDetector initialized with method={self.config.method.value}")
    
    def add_speaker_change_callback(self, callback: Callable[[SpeakerChangeDetection], None]) -> None:
        """Add callback for speaker changes"""
        self.speaker_change_callbacks.append(callback)
    
    def add_result_callback(self, callback: Callable[[SpeakerResult], None]) -> None:
        """Add callback for analysis results"""
        self.result_callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> bool:
        """Remove a callback from all callback lists"""
        removed = False
        for callback_list in [self.speaker_change_callbacks, self.result_callbacks]:
            if callback in callback_list:
                callback_list.remove(callback)
                removed = True
        return removed
    
    def _extract_mfcc_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio frame"""
        try:
            # Simple MFCC-like feature extraction
            # In a real implementation, you would use librosa or similar
            
            # FFT-based spectral analysis
            fft = np.fft.rfft(audio_data.astype(np.float64))
            magnitude_spectrum = np.abs(fft)
            
            # Mel filter bank simulation (simplified)
            n_mfcc = self.config.feature_dimension
            mfcc_features = np.zeros(n_mfcc)
            
            # Divide spectrum into mel-scale bins
            n_bins = len(magnitude_spectrum) // n_mfcc
            for i in range(n_mfcc):
                start_bin = i * n_bins
                end_bin = min((i + 1) * n_bins, len(magnitude_spectrum))
                if end_bin > start_bin:
                    mfcc_features[i] = np.log(np.mean(magnitude_spectrum[start_bin:end_bin]) + 1e-8)
            
            return mfcc_features
            
        except Exception as e:
            logger.error(f"Error extracting MFCC features: {e}")
            return np.zeros(self.config.feature_dimension)
    
    def _extract_spectral_features(self, audio_data: np.ndarray) -> Tuple[float, float, float]:
        """Extract spectral features from audio frame"""
        try:
            # Convert to frequency domain
            fft = np.fft.rfft(audio_data.astype(np.float64))
            magnitude_spectrum = np.abs(fft)
            frequencies = np.fft.rfftfreq(len(audio_data), 1.0 / self.config.sample_rate)
            
            # Spectral centroid
            if np.sum(magnitude_spectrum) > 0:
                spectral_centroid = np.sum(frequencies * magnitude_spectrum) / np.sum(magnitude_spectrum)
            else:
                spectral_centroid = 0.0
            
            # Spectral rolloff (85% of energy)
            cumsum_spectrum = np.cumsum(magnitude_spectrum)
            total_energy = cumsum_spectrum[-1]
            rolloff_threshold = 0.85 * total_energy
            rolloff_idx = np.where(cumsum_spectrum >= rolloff_threshold)[0]
            spectral_rolloff = frequencies[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0.0
            
            # Zero crossing rate
            zero_crossings = np.where(np.diff(np.signbit(audio_data)))[0]
            zero_crossing_rate = len(zero_crossings) / len(audio_data)
            
            return spectral_centroid, spectral_rolloff, zero_crossing_rate
            
        except Exception as e:
            logger.error(f"Error extracting spectral features: {e}")
            return 0.0, 0.0, 0.0
    
    def _extract_pitch_features(self, audio_data: np.ndarray) -> Tuple[float, float]:
        """Extract pitch-related features"""
        try:
            # Simple autocorrelation-based pitch detection
            audio_float = audio_data.astype(np.float64)
            
            # Autocorrelation
            autocorr = np.correlate(audio_float, audio_float, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find fundamental frequency
            min_period = int(self.config.sample_rate / 800)  # 800 Hz max
            max_period = int(self.config.sample_rate / 80)   # 80 Hz min
            
            if len(autocorr) > max_period:
                autocorr_section = autocorr[min_period:max_period]
                if len(autocorr_section) > 0:
                    period = np.argmax(autocorr_section) + min_period
                    pitch = self.config.sample_rate / period
                else:
                    pitch = 0.0
            else:
                pitch = 0.0
            
            # Pitch stability (simplified)
            pitch_std = np.std(audio_float) / (np.mean(np.abs(audio_float)) + 1e-8)
            
            return pitch, pitch_std
            
        except Exception as e:
            logger.error(f"Error extracting pitch features: {e}")
            return 0.0, 0.0
    
    def _extract_formants(self, audio_data: np.ndarray) -> List[float]:
        """Extract formant frequencies (simplified)"""
        try:
            # Simple formant estimation using FFT peaks
            fft = np.fft.rfft(audio_data.astype(np.float64))
            magnitude_spectrum = np.abs(fft)
            frequencies = np.fft.rfftfreq(len(audio_data), 1.0 / self.config.sample_rate)
            
            # Find peaks in spectrum
            peaks = []
            for i in range(1, len(magnitude_spectrum) - 1):
                if (magnitude_spectrum[i] > magnitude_spectrum[i-1] and 
                    magnitude_spectrum[i] > magnitude_spectrum[i+1] and
                    magnitude_spectrum[i] > np.max(magnitude_spectrum) * 0.1):
                    peaks.append(frequencies[i])
            
            # Return first 3 formants (or zeros if not found)
            formants = sorted(peaks)[:3]
            while len(formants) < 3:
                formants.append(0.0)
            
            return formants
            
        except Exception as e:
            logger.error(f"Error extracting formants: {e}")
            return [0.0, 0.0, 0.0]
    
    def _extract_features(self, audio_data: np.ndarray, timestamp: float, 
                         voice_activity: bool = True) -> SpeakerFeatures:
        """Extract comprehensive speaker features from audio frame"""
        try:
            # MFCC features
            mfcc = self._extract_mfcc_features(audio_data)
            
            # Spectral features
            spectral_centroid, spectral_rolloff, zero_crossing_rate = self._extract_spectral_features(audio_data)
            
            # Energy
            energy = float(np.mean(audio_data.astype(np.float64) ** 2))
            
            # Pitch features
            pitch_mean, pitch_std = self._extract_pitch_features(audio_data)
            
            # Formants
            formants = self._extract_formants(audio_data)
            
            return SpeakerFeatures(
                timestamp=timestamp,
                mfcc=mfcc,
                spectral_centroid=spectral_centroid,
                spectral_rolloff=spectral_rolloff,
                zero_crossing_rate=zero_crossing_rate,
                energy=energy,
                pitch_mean=pitch_mean,
                pitch_std=pitch_std,
                formants=formants,
                voice_activity=voice_activity
            )
            
        except Exception as e:
            logger.error(f"Error extracting speaker features: {e}")
            return SpeakerFeatures(
                timestamp=timestamp,
                mfcc=np.zeros(self.config.feature_dimension),
                spectral_centroid=0.0,
                spectral_rolloff=0.0,
                zero_crossing_rate=0.0,
                energy=0.0,
                pitch_mean=0.0,
                pitch_std=0.0,
                formants=[0.0, 0.0, 0.0],
                voice_activity=voice_activity
            )
    
    def _calculate_feature_distance(self, features1: SpeakerFeatures, 
                                   features2: SpeakerFeatures) -> float:
        """Calculate distance between two feature vectors"""
        try:
            vector1 = features1.to_vector()
            vector2 = features2.to_vector()
            
            # Normalize vectors
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            
            if norm1 == 0 or norm2 == 0:
                return 1.0  # Maximum distance
            
            # Cosine distance
            cosine_sim = np.dot(vector1, vector2) / (norm1 * norm2)
            cosine_distance = 1.0 - cosine_sim
            
            return float(np.clip(cosine_distance, 0.0, 2.0))
            
        except Exception as e:
            logger.error(f"Error calculating feature distance: {e}")
            return 1.0
    
    def _assign_speaker(self, features: SpeakerFeatures) -> Tuple[Optional[str], float]:
        """Assign speaker ID based on features"""
        if not features.voice_activity and self.config.voice_activity_required:
            return None, 0.0
        
        if not self.speaker_models:
            # First speaker
            speaker_id = self._generate_speaker_id()
            self.speaker_models[speaker_id] = features.to_vector()
            self.stats['total_speakers'] = 1
            return speaker_id, 1.0
        
        # Find best matching speaker
        best_speaker = None
        best_similarity = 0.0
        feature_vector = features.to_vector()
        
        for speaker_id, model_vector in self.speaker_models.items():
            try:
                # Calculate similarity
                similarity = cosine_similarity([feature_vector], [model_vector])[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_speaker = speaker_id
                    
            except Exception as e:
                logger.error(f"Error calculating similarity for speaker {speaker_id}: {e}")
        
        # Check if similarity is above threshold
        if best_similarity >= self.config.similarity_threshold:
            # Update speaker model with adaptation
            if best_speaker in self.speaker_models:
                current_model = self.speaker_models[best_speaker]
                adapted_model = ((1 - self.config.adaptation_rate) * current_model + 
                               self.config.adaptation_rate * feature_vector)
                self.speaker_models[best_speaker] = adapted_model
            
            return best_speaker, best_similarity
        
        # Create new speaker if we haven't reached the limit
        if len(self.speaker_models) < self.config.max_speakers:
            speaker_id = self._generate_speaker_id()
            self.speaker_models[speaker_id] = feature_vector
            self.stats['total_speakers'] = len(self.speaker_models)
            return speaker_id, 1.0
        
        # Assign to best match even if below threshold
        return best_speaker, best_similarity
    
    def _generate_speaker_id(self) -> str:
        """Generate unique speaker ID"""
        self.speaker_counter += 1
        return f"speaker_{self.speaker_counter:03d}"
    
    def _detect_speaker_change(self, current_features: SpeakerFeatures) -> Tuple[bool, float]:
        """Detect if there's a speaker change"""
        if len(self.feature_buffer) < 2:
            return False, 0.0
        
        # Compare with recent features
        recent_distances = []
        for past_features in list(self.feature_buffer)[-5:]:  # Last 5 frames
            if past_features.voice_activity:
                distance = self._calculate_feature_distance(current_features, past_features)
                recent_distances.append(distance)
        
        if not recent_distances:
            return False, 0.0
        
        # Calculate change probability
        avg_distance = np.mean(recent_distances)
        change_confidence = min(avg_distance / self.config.change_threshold, 1.0)
        is_change = avg_distance > self.config.change_threshold
        
        return is_change, change_confidence
    
    def analyze_speaker(self, audio_data: np.ndarray, timestamp: Optional[float] = None,
                       voice_activity: bool = True) -> SpeakerResult:
        """
        Analyze audio frame for speaker identification
        
        Args:
            audio_data: Audio data as numpy array
            timestamp: Timestamp of the frame
            voice_activity: Whether voice activity is detected
        
        Returns:
            SpeakerResult with analysis results
        """
        if timestamp is None:
            timestamp = time.time()
        
        start_time = time.time()
        
        try:
            # Extract features
            features = self._extract_features(audio_data, timestamp, voice_activity)
            
            # Add to buffer
            self.feature_buffer.append(features)
            
            # Detect speaker change
            is_change, change_confidence = self._detect_speaker_change(features)
            
            # Assign speaker
            speaker_id, confidence = self._assign_speaker(features)
            previous_speaker_id = self.current_speaker_id
            
            # Handle speaker change
            if is_change and speaker_id != self.current_speaker_id:
                with self.thread_lock:
                    # Fire speaker change callbacks
                    change_detection = SpeakerChangeDetection(
                        timestamp=timestamp,
                        from_speaker=self.current_speaker_id,
                        to_speaker=speaker_id,
                        confidence=change_confidence,
                        method=self.config.method.value,
                        feature_distance=change_confidence
                    )
                    
                    for callback in self.speaker_change_callbacks:
                        try:
                            callback(change_detection)
                        except Exception as e:
                            logger.error(f"Error in speaker change callback: {e}")
                    
                    # Update current speaker
                    if self.current_speaker_id != speaker_id:
                        self.current_speaker_id = speaker_id
                        self.stats['speaker_changes'] += 1
                        
                        # Finalize previous segment
                        if self.current_segment_start is not None:
                            segment_duration = timestamp - self.current_segment_start
                            self.speaker_segments.append({
                                'speaker_id': previous_speaker_id,
                                'start_time': self.current_segment_start,
                                'duration': segment_duration,
                                'end_time': timestamp
                            })
                        
                        self.current_segment_start = timestamp
            
            elif self.current_speaker_id is None:
                # First speaker assignment
                with self.thread_lock:
                    self.current_speaker_id = speaker_id
                    self.current_segment_start = timestamp
            
            # Update statistics
            with self.thread_lock:
                self.stats['total_frames'] += 1
                self.stats['segments_processed'] += 1
                self.stats['processing_time'] += time.time() - start_time
                
                if self.stats['start_time'] is None:
                    self.stats['start_time'] = timestamp
            
            # Create result
            result = SpeakerResult(
                timestamp=timestamp,
                speaker_id=speaker_id,
                confidence=confidence,
                is_change=is_change,
                change_confidence=change_confidence,
                previous_speaker_id=previous_speaker_id,
                features=features,
                segment_duration_ms=None,
                analysis_method=self.config.method.value
            )
            
            # Fire result callbacks
            for callback in self.result_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in result callback: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in speaker analysis: {e}")
            # Return safe default result
            return SpeakerResult(
                timestamp=timestamp,
                speaker_id=None,
                confidence=0.0,
                is_change=False,
                change_confidence=0.0,
                previous_speaker_id=None,
                features=SpeakerFeatures(
                    timestamp=timestamp,
                    mfcc=np.zeros(self.config.feature_dimension),
                    spectral_centroid=0.0,
                    spectral_rolloff=0.0,
                    zero_crossing_rate=0.0,
                    energy=0.0,
                    pitch_mean=0.0,
                    pitch_std=0.0,
                    formants=[0.0, 0.0, 0.0],
                    voice_activity=voice_activity
                ),
                analysis_method=self.config.method.value
            )
    
    def get_speaker_models(self) -> Dict[str, np.ndarray]:
        """Get current speaker models"""
        with self.thread_lock:
            return self.speaker_models.copy()
    
    def get_speaker_segments(self) -> List[Dict[str, Any]]:
        """Get detected speaker segments"""
        with self.thread_lock:
            return self.speaker_segments.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics"""
        with self.thread_lock:
            stats = self.stats.copy()
            
            # Add derived statistics
            if stats['total_frames'] > 0:
                stats['average_processing_time_ms'] = (stats['processing_time'] / stats['total_frames']) * 1000
            else:
                stats['average_processing_time_ms'] = 0.0
            
            stats['total_speakers_detected'] = len(self.speaker_models)
            stats['total_segments'] = len(self.speaker_segments)
            
            if stats['start_time'] is not None:
                stats['total_session_time'] = time.time() - stats['start_time']
            else:
                stats['total_session_time'] = 0.0
        
        return stats
    
    def reset_speaker_models(self) -> None:
        """Reset all speaker models and start fresh"""
        with self.thread_lock:
            self.speaker_models.clear()
            self.speaker_segments.clear()
            self.feature_buffer.clear()
            self.current_speaker_id = None
            self.current_segment_start = None
            self.speaker_counter = 0
            
            # Reset statistics
            self.stats = {
                'total_frames': 0,
                'speaker_changes': 0,
                'total_speakers': 0,
                'segments_processed': 0,
                'start_time': None,
                'processing_time': 0.0
            }
        
        logger.info("Speaker models and statistics reset")
    
    def start_processing(self) -> bool:
        """Start speaker detection processing"""
        if self.is_running:
            logger.warning("Speaker detection already running")
            return False
        
        with self.thread_lock:
            self.is_running = True
            self.stats['start_time'] = time.time()
        
        logger.info("Speaker detection started")
        return True
    
    def stop_processing(self) -> bool:
        """Stop speaker detection processing"""
        if not self.is_running:
            logger.warning("Speaker detection not running")
            return False
        
        with self.thread_lock:
            self.is_running = False
            
            # Finalize current segment
            if self.current_segment_start is not None and self.current_speaker_id is not None:
                current_time = time.time()
                segment_duration = current_time - self.current_segment_start
                self.speaker_segments.append({
                    'speaker_id': self.current_speaker_id,
                    'start_time': self.current_segment_start,
                    'duration': segment_duration,
                    'end_time': current_time
                })
        
        logger.info("Speaker detection stopped")
        return True
    
    def is_processing(self) -> bool:
        """Check if speaker detection is currently processing"""
        return self.is_running

def create_speaker_detector(config: Optional[SpeakerConfig] = None) -> SpeakerDetector:
    """
    Factory function to create a configured speaker detector
    
    Args:
        config: Optional speaker detection configuration
        
    Returns:
        Configured SpeakerDetector instance
    """
    return SpeakerDetector(config)

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create speaker detector
    config = SpeakerConfig(
        method=SpeakerChangeMethod.COMBINED,
        sample_rate=16000,
        frame_duration_ms=100,
        change_threshold=0.3
    )
    
    detector = create_speaker_detector(config)
    
    # Add example callbacks
    def on_speaker_change(change: SpeakerChangeDetection):
        print(f"Speaker change: {change.from_speaker} -> {change.to_speaker} "
              f"(confidence: {change.confidence:.2f})")
    
    def on_result(result: SpeakerResult):
        print(f"Speaker: {result.speaker_id} (confidence: {result.confidence:.2f}, "
              f"change: {result.is_change})")
    
    detector.add_speaker_change_callback(on_speaker_change)
    detector.add_result_callback(on_result)
    
    # Start processing
    detector.start_processing()
    
    # Test with synthetic audio data
    try:
        print("Testing speaker detection with synthetic audio...")
        
        frame_size = int(16000 * 0.1)  # 100ms frames
        
        # Simulate different speakers with different characteristics
        for i in range(30):
            if i < 10:
                # Speaker 1: Lower frequency content
                audio_frame = np.random.normal(0, 200, frame_size).astype(np.int16)
                # Add some low frequency content
                t = np.linspace(0, 0.1, frame_size)
                audio_frame += (1000 * np.sin(2 * np.pi * 150 * t)).astype(np.int16)
            elif i < 20:
                # Speaker 2: Higher frequency content
                audio_frame = np.random.normal(0, 300, frame_size).astype(np.int16)
                # Add some high frequency content
                t = np.linspace(0, 0.1, frame_size)
                audio_frame += (800 * np.sin(2 * np.pi * 300 * t)).astype(np.int16)
            else:
                # Speaker 1 again
                audio_frame = np.random.normal(0, 200, frame_size).astype(np.int16)
                t = np.linspace(0, 0.1, frame_size)
                audio_frame += (1000 * np.sin(2 * np.pi * 150 * t)).astype(np.int16)
            
            result = detector.analyze_speaker(audio_frame, voice_activity=True)
            time.sleep(0.1)  # 100ms delay
        
        # Print statistics
        stats = detector.get_statistics()
        print(f"\nStatistics:")
        print(f"Total frames: {stats['total_frames']}")
        print(f"Speaker changes: {stats['speaker_changes']}")
        print(f"Total speakers: {stats['total_speakers_detected']}")
        print(f"Total segments: {stats['total_segments']}")
        
        # Print speaker segments
        segments = detector.get_speaker_segments()
        print(f"\nSpeaker segments:")
        for segment in segments:
            print(f"  {segment['speaker_id']}: {segment['duration']:.2f}s")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    finally:
        detector.stop_processing()
        print("Speaker detection test completed")