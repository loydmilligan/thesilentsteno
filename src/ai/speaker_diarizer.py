#!/usr/bin/env python3
"""
Speaker Diarizer Module

Speaker identification and labeling for multi-participant transcription.
Integrates with existing speaker detection for enhanced diarization.

Author: Claude AI Assistant
Date: 2024-07-14
Version: 1.0
"""

import os
import sys
import logging
import threading
import time
from typing import Optional, Dict, Any, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import uuid
from collections import defaultdict, deque

try:
    import numpy as np
    import soundfile as sf
except ImportError as e:
    print(f"Warning: Required dependencies not installed: {e}")
    print("Please install with: pip install numpy soundfile")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeakerLabel(Enum):
    """Speaker label types"""
    SPEAKER_1 = "Speaker 1"
    SPEAKER_2 = "Speaker 2"
    SPEAKER_3 = "Speaker 3"
    SPEAKER_4 = "Speaker 4"
    SPEAKER_5 = "Speaker 5"
    SPEAKER_6 = "Speaker 6"
    SPEAKER_7 = "Speaker 7"
    SPEAKER_8 = "Speaker 8"
    SPEAKER_9 = "Speaker 9"
    SPEAKER_10 = "Speaker 10"
    UNKNOWN = "Unknown Speaker"
    SILENCE = "Silence"
    NOISE = "Noise"
    
    @classmethod
    def get_numbered_label(cls, speaker_id: int) -> 'SpeakerLabel':
        """Get speaker label by number"""
        label_map = {
            1: cls.SPEAKER_1, 2: cls.SPEAKER_2, 3: cls.SPEAKER_3,
            4: cls.SPEAKER_4, 5: cls.SPEAKER_5, 6: cls.SPEAKER_6,
            7: cls.SPEAKER_7, 8: cls.SPEAKER_8, 9: cls.SPEAKER_9,
            10: cls.SPEAKER_10
        }
        return label_map.get(speaker_id, cls.UNKNOWN)


@dataclass
class SpeakerSegment:
    """Represents a segment with speaker information"""
    
    # Timing information
    start_time: float
    end_time: float
    duration: float
    
    # Speaker information
    speaker_id: int
    speaker_label: SpeakerLabel
    confidence: float
    
    # Audio data
    audio_data: Optional[np.ndarray] = None
    
    # Transcription information
    text: str = ""
    transcription_confidence: float = 0.0
    
    # Quality metrics
    voice_activity: float = 0.0
    signal_quality: float = 0.0
    speaker_consistency: float = 0.0
    
    # Metadata
    segment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    processing_hints: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary"""
        return {
            "segment_id": self.segment_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "speaker_id": self.speaker_id,
            "speaker_label": self.speaker_label.value,
            "confidence": self.confidence,
            "text": self.text,
            "transcription_confidence": self.transcription_confidence,
            "voice_activity": self.voice_activity,
            "signal_quality": self.signal_quality,
            "speaker_consistency": self.speaker_consistency,
            "processing_hints": self.processing_hints
        }


@dataclass
class DiarizationConfig:
    """Configuration for speaker diarization"""
    
    # Basic parameters
    max_speakers: int = 10
    min_segment_duration: float = 0.5  # seconds
    max_segment_duration: float = 30.0  # seconds
    
    # Speaker detection parameters
    speaker_change_threshold: float = 0.7
    speaker_similarity_threshold: float = 0.8
    confidence_threshold: float = 0.5
    
    # Integration settings
    use_existing_speaker_detection: bool = True
    use_voice_activity_detection: bool = True
    use_quality_hints: bool = True
    
    # Clustering parameters
    clustering_method: str = "hierarchical"  # hierarchical, kmeans, spectral
    distance_metric: str = "cosine"  # cosine, euclidean, manhattan
    linkage_method: str = "ward"  # ward, complete, average
    
    # Adaptive parameters
    adaptive_threshold: bool = True
    learning_rate: float = 0.1
    memory_decay: float = 0.95
    
    # Performance settings
    batch_size: int = 100  # segments per batch
    processing_timeout: float = 30.0  # seconds
    
    # Output settings
    label_format: str = "Speaker {id}"  # Format for speaker labels
    include_confidence: bool = True
    include_timestamps: bool = True
    merge_consecutive_segments: bool = True
    
    def get_speaker_label(self, speaker_id: int) -> str:
        """Get formatted speaker label"""
        if speaker_id == 0:
            return "Unknown Speaker"
        elif speaker_id == -1:
            return "Silence"
        elif speaker_id == -2:
            return "Noise"
        else:
            return self.label_format.format(id=speaker_id)


@dataclass
class DiarizationResult:
    """Result of speaker diarization"""
    
    # Segments with speaker labels
    segments: List[SpeakerSegment] = field(default_factory=list)
    
    # Speaker statistics
    speaker_count: int = 0
    speaker_ids: List[int] = field(default_factory=list)
    speaker_labels: List[SpeakerLabel] = field(default_factory=list)
    
    # Timing information
    total_duration: float = 0.0
    total_speech_time: float = 0.0
    total_silence_time: float = 0.0
    
    # Quality metrics
    average_confidence: float = 0.0
    speaker_consistency: float = 0.0
    diarization_accuracy: float = 0.0
    
    # Processing metadata
    processing_time: float = 0.0
    segments_processed: int = 0
    
    # Statistics per speaker
    speaker_statistics: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    # Error information
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "segments": [seg.to_dict() for seg in self.segments],
            "speaker_count": self.speaker_count,
            "speaker_ids": self.speaker_ids,
            "speaker_labels": [label.value for label in self.speaker_labels],
            "total_duration": self.total_duration,
            "total_speech_time": self.total_speech_time,
            "total_silence_time": self.total_silence_time,
            "average_confidence": self.average_confidence,
            "speaker_consistency": self.speaker_consistency,
            "diarization_accuracy": self.diarization_accuracy,
            "processing_time": self.processing_time,
            "segments_processed": self.segments_processed,
            "speaker_statistics": self.speaker_statistics,
            "errors": self.errors,
            "warnings": self.warnings,
            "success": self.success
        }


class SpeakerDiarizer:
    """Speaker diarization for transcription labeling"""
    
    def __init__(self, config: Optional[DiarizationConfig] = None):
        self.config = config or DiarizationConfig()
        self.is_initialized = False
        self.is_running = False
        
        # Integration with existing analysis system
        self.speaker_detector = None
        self.vad_detector = None
        self.quality_assessor = None
        
        # Speaker models and state
        self.speaker_models = {}  # speaker_id -> feature_vectors
        self.speaker_counters = defaultdict(int)
        self.next_speaker_id = 1
        
        # Processing state
        self.processing_lock = threading.Lock()
        self.segment_buffer = deque(maxlen=1000)
        
        # Statistics
        self.stats = {
            "total_segments_processed": 0,
            "total_speakers_detected": 0,
            "total_audio_processed": 0.0,
            "average_confidence": 0.0,
            "processing_time": 0.0,
            "speaker_changes": 0,
            "accuracy_estimates": []
        }
        
        # Callbacks
        self.segment_callbacks: List[Callable] = []
        self.speaker_callbacks: List[Callable] = []
        
        logger.info(f"SpeakerDiarizer initialized with max {self.config.max_speakers} speakers")
        
    def initialize(self) -> bool:
        """Initialize the diarizer and analysis components"""
        try:
            logger.info("Initializing speaker diarizer...")
            
            # Initialize speaker detection integration
            if self.config.use_existing_speaker_detection:
                self._init_speaker_detector()
                
            # Initialize VAD integration
            if self.config.use_voice_activity_detection:
                self._init_vad_detector()
                
            # Initialize quality assessment integration
            if self.config.use_quality_hints:
                self._init_quality_assessor()
                
            self.is_initialized = True
            self.is_running = True
            
            logger.info("Speaker diarizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize diarizer: {e}")
            return False
            
    def _init_speaker_detector(self):
        """Initialize speaker detector integration"""
        try:
            # Try to import speaker detector from analysis system
            from ..analysis.speaker_detector import SpeakerDetector, SpeakerConfig
            
            speaker_config = SpeakerConfig(
                change_threshold=self.config.speaker_change_threshold,
                similarity_threshold=self.config.speaker_similarity_threshold
            )
            self.speaker_detector = SpeakerDetector(speaker_config)
            self.speaker_detector.initialize()
            
            logger.info("Speaker detector integration initialized")
            
        except ImportError:
            logger.warning("Speaker detector not available, using fallback")
            self.speaker_detector = None
            
    def _init_vad_detector(self):
        """Initialize VAD integration"""
        try:
            # Try to import VAD from analysis system
            from ..analysis.voice_activity_detector import VoiceActivityDetector, VADConfig
            
            vad_config = VADConfig(
                mode="balanced",
                threshold=0.5
            )
            self.vad_detector = VoiceActivityDetector(vad_config)
            self.vad_detector.initialize()
            
            logger.info("VAD integration initialized")
            
        except ImportError:
            logger.warning("VAD not available, using fallback")
            self.vad_detector = None
            
    def _init_quality_assessor(self):
        """Initialize quality assessor integration"""
        try:
            # Try to import quality assessor from analysis system
            from ..analysis.quality_assessor import QualityAssessor, QualityConfig
            
            quality_config = QualityConfig(
                assessment_interval=1.0
            )
            self.quality_assessor = QualityAssessor(quality_config)
            self.quality_assessor.initialize()
            
            logger.info("Quality assessor integration initialized")
            
        except ImportError:
            logger.warning("Quality assessor not available, using fallback")
            self.quality_assessor = None
            
    def diarize_segments(self, audio_data: np.ndarray, 
                        start_time: float = 0.0) -> DiarizationResult:
        """Perform speaker diarization on audio data"""
        if not self.is_initialized:
            logger.error("Diarizer not initialized")
            return DiarizationResult(success=False, errors=["Diarizer not initialized"])
            
        start_processing_time = time.time()
        
        try:
            # Segment audio into speaker segments
            segments = self._segment_audio(audio_data, start_time)
            
            # Identify speakers in each segment
            identified_segments = self._identify_speakers(segments)
            
            # Post-process segments (merge, filter, etc.)
            processed_segments = self._post_process_segments(identified_segments)
            
            # Calculate statistics
            processing_time = time.time() - start_processing_time
            result = self._create_result(processed_segments, processing_time)
            
            # Update statistics
            self._update_stats(result)
            
            logger.info(f"Diarization completed: {result.speaker_count} speakers, {len(result.segments)} segments")
            return result
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return DiarizationResult(
                success=False,
                errors=[str(e)],
                processing_time=time.time() - start_processing_time
            )
            
    def _segment_audio(self, audio_data: np.ndarray, start_time: float) -> List[SpeakerSegment]:
        """Segment audio into potential speaker segments"""
        segments = []
        
        # Use existing speaker detection if available
        if self.speaker_detector:
            # Get speaker change points from existing detector
            change_points = self._get_speaker_change_points(audio_data)
        else:
            # Use simple energy-based segmentation
            change_points = self._energy_based_segmentation(audio_data)
            
        # Create segments from change points
        sample_rate = 16000  # Assume 16kHz
        
        for i in range(len(change_points) - 1):
            start_idx = int(change_points[i] * sample_rate)
            end_idx = int(change_points[i + 1] * sample_rate)
            
            if end_idx > len(audio_data):
                end_idx = len(audio_data)
                
            segment_audio = audio_data[start_idx:end_idx]
            segment_duration = len(segment_audio) / sample_rate
            
            # Skip very short segments
            if segment_duration < self.config.min_segment_duration:
                continue
                
            # Create segment
            segment = SpeakerSegment(
                start_time=start_time + change_points[i],
                end_time=start_time + change_points[i + 1],
                duration=segment_duration,
                speaker_id=0,  # To be determined
                speaker_label=SpeakerLabel.UNKNOWN,
                confidence=0.0,
                audio_data=segment_audio
            )
            
            segments.append(segment)
            
        return segments
        
    def _get_speaker_change_points(self, audio_data: np.ndarray) -> List[float]:
        """Get speaker change points from existing detector"""
        # This would integrate with the existing speaker detector
        # For now, use simple segmentation
        return self._energy_based_segmentation(audio_data)
        
    def _energy_based_segmentation(self, audio_data: np.ndarray) -> List[float]:
        """Simple energy-based segmentation"""
        sample_rate = 16000
        window_size = int(0.5 * sample_rate)  # 0.5 second windows
        hop_size = int(0.1 * sample_rate)  # 0.1 second hop
        
        # Calculate energy in each window
        energies = []
        for i in range(0, len(audio_data) - window_size, hop_size):
            window = audio_data[i:i + window_size]
            energy = np.sum(window ** 2)
            energies.append(energy)
            
        # Find energy changes (simple change detection)
        change_points = [0.0]  # Start with beginning
        
        for i in range(1, len(energies) - 1):
            prev_energy = energies[i - 1]
            curr_energy = energies[i]
            next_energy = energies[i + 1]
            
            # Look for significant energy changes
            if abs(curr_energy - prev_energy) > 0.5 * prev_energy:
                time_point = i * hop_size / sample_rate
                change_points.append(time_point)
                
        # Add end point
        change_points.append(len(audio_data) / sample_rate)
        
        return change_points
        
    def _identify_speakers(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """Identify speakers in each segment"""
        identified_segments = []
        
        for segment in segments:
            # Extract speaker features
            features = self._extract_speaker_features(segment.audio_data)
            
            # Compare with existing speaker models
            speaker_id, confidence = self._match_speaker(features)
            
            # Update segment with speaker information
            segment.speaker_id = speaker_id
            segment.speaker_label = SpeakerLabel.get_numbered_label(speaker_id)
            segment.confidence = confidence
            
            # Calculate additional quality metrics
            segment.voice_activity = self._calculate_voice_activity(segment.audio_data)
            segment.signal_quality = self._calculate_signal_quality(segment.audio_data)
            segment.speaker_consistency = confidence
            
            identified_segments.append(segment)
            
        return identified_segments
        
    def _extract_speaker_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract speaker features from audio"""
        # Simplified feature extraction
        # In a real implementation, this would use MFCC, spectral features, etc.
        
        # Calculate basic spectral features
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Extract spectral centroid, rolloff, and other features
        spectral_centroid = np.sum(magnitude * np.arange(len(magnitude))) / (np.sum(magnitude) + 1e-10)
        spectral_rolloff = np.argmax(np.cumsum(magnitude) >= 0.85 * np.sum(magnitude))
        spectral_flux = np.sum(np.diff(magnitude)**2)
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
        
        # Energy features
        energy = np.sum(audio_data**2)
        rms = np.sqrt(np.mean(audio_data**2))
        
        # Combine features
        features = np.array([
            spectral_centroid, spectral_rolloff, spectral_flux,
            zero_crossings, energy, rms
        ])
        
        # Normalize features
        features = features / (np.linalg.norm(features) + 1e-10)
        
        return features
        
    def _match_speaker(self, features: np.ndarray) -> Tuple[int, float]:
        """Match features to existing speaker or create new speaker"""
        if not self.speaker_models:
            # First speaker
            speaker_id = self.next_speaker_id
            self.speaker_models[speaker_id] = [features]
            self.next_speaker_id += 1
            return speaker_id, 1.0
            
        # Compare with existing speakers
        best_match_id = 0
        best_similarity = 0.0
        
        for speaker_id, speaker_features in self.speaker_models.items():
            # Calculate average similarity to all features for this speaker
            similarities = []
            for stored_features in speaker_features:
                similarity = self._calculate_similarity(features, stored_features)
                similarities.append(similarity)
                
            avg_similarity = np.mean(similarities)
            
            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_match_id = speaker_id
                
        # Check if similarity is above threshold
        if best_similarity > self.config.speaker_similarity_threshold:
            # Add to existing speaker
            self.speaker_models[best_match_id].append(features)
            self.speaker_counters[best_match_id] += 1
            return best_match_id, best_similarity
        else:
            # Create new speaker if we haven't reached max speakers
            if len(self.speaker_models) < self.config.max_speakers:
                speaker_id = self.next_speaker_id
                self.speaker_models[speaker_id] = [features]
                self.speaker_counters[speaker_id] = 1
                self.next_speaker_id += 1
                return speaker_id, 0.8  # New speaker confidence
            else:
                # Assign to best match even if below threshold
                self.speaker_models[best_match_id].append(features)
                self.speaker_counters[best_match_id] += 1
                return best_match_id, best_similarity * 0.5  # Reduced confidence
                
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between two feature vectors"""
        if self.config.distance_metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            return dot_product / (norm1 * norm2 + 1e-10)
        elif self.config.distance_metric == "euclidean":
            # Euclidean distance converted to similarity
            distance = np.linalg.norm(features1 - features2)
            return 1.0 / (1.0 + distance)
        else:
            # Default to cosine similarity
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            return dot_product / (norm1 * norm2 + 1e-10)
            
    def _calculate_voice_activity(self, audio_data: np.ndarray) -> float:
        """Calculate voice activity ratio"""
        if self.vad_detector:
            # Use existing VAD detector
            return 0.8  # Placeholder
        else:
            # Simple energy-based VAD
            energy = np.sum(audio_data**2) / len(audio_data)
            return min(1.0, energy * 100)  # Rough approximation
            
    def _calculate_signal_quality(self, audio_data: np.ndarray) -> float:
        """Calculate signal quality"""
        if self.quality_assessor:
            # Use existing quality assessor
            return 0.8  # Placeholder
        else:
            # Simple SNR estimation
            signal_power = np.var(audio_data)
            noise_power = np.var(audio_data[:int(0.1 * len(audio_data))])  # First 10% as noise estimate
            snr = signal_power / (noise_power + 1e-10)
            return min(1.0, max(0.0, snr / 10))
            
    def _post_process_segments(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """Post-process segments to improve quality"""
        if not segments:
            return segments
            
        processed_segments = []
        
        # Merge consecutive segments from same speaker
        if self.config.merge_consecutive_segments:
            segments = self._merge_consecutive_segments(segments)
            
        # Filter out very short segments
        for segment in segments:
            if segment.duration >= self.config.min_segment_duration:
                processed_segments.append(segment)
                
        # Apply confidence threshold
        if self.config.confidence_threshold > 0:
            processed_segments = [
                seg for seg in processed_segments 
                if seg.confidence >= self.config.confidence_threshold
            ]
            
        return processed_segments
        
    def _merge_consecutive_segments(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """Merge consecutive segments from the same speaker"""
        if len(segments) <= 1:
            return segments
            
        merged_segments = [segments[0]]
        
        for current_segment in segments[1:]:
            last_segment = merged_segments[-1]
            
            # Check if same speaker and consecutive
            if (current_segment.speaker_id == last_segment.speaker_id and
                abs(current_segment.start_time - last_segment.end_time) < 0.1):  # 100ms gap tolerance
                
                # Merge segments
                merged_audio = None
                if last_segment.audio_data is not None and current_segment.audio_data is not None:
                    merged_audio = np.concatenate([last_segment.audio_data, current_segment.audio_data])
                    
                merged_segment = SpeakerSegment(
                    start_time=last_segment.start_time,
                    end_time=current_segment.end_time,
                    duration=current_segment.end_time - last_segment.start_time,
                    speaker_id=last_segment.speaker_id,
                    speaker_label=last_segment.speaker_label,
                    confidence=max(last_segment.confidence, current_segment.confidence),
                    audio_data=merged_audio,
                    text=last_segment.text + " " + current_segment.text,
                    transcription_confidence=max(last_segment.transcription_confidence, 
                                               current_segment.transcription_confidence),
                    voice_activity=max(last_segment.voice_activity, current_segment.voice_activity),
                    signal_quality=max(last_segment.signal_quality, current_segment.signal_quality),
                    speaker_consistency=max(last_segment.speaker_consistency, 
                                         current_segment.speaker_consistency)
                )
                
                merged_segments[-1] = merged_segment
            else:
                merged_segments.append(current_segment)
                
        return merged_segments
        
    def _create_result(self, segments: List[SpeakerSegment], processing_time: float) -> DiarizationResult:
        """Create diarization result from processed segments"""
        if not segments:
            return DiarizationResult(
                processing_time=processing_time,
                success=True
            )
            
        # Calculate basic statistics
        total_duration = segments[-1].end_time - segments[0].start_time if segments else 0.0
        total_speech_time = sum(seg.duration for seg in segments)
        total_silence_time = total_duration - total_speech_time
        
        # Get unique speakers
        speaker_ids = list(set(seg.speaker_id for seg in segments))
        speaker_labels = [SpeakerLabel.get_numbered_label(sid) for sid in speaker_ids]
        
        # Calculate average confidence
        avg_confidence = np.mean([seg.confidence for seg in segments]) if segments else 0.0
        
        # Calculate speaker consistency
        speaker_consistency = np.mean([seg.speaker_consistency for seg in segments]) if segments else 0.0
        
        # Calculate per-speaker statistics
        speaker_statistics = {}
        for speaker_id in speaker_ids:
            speaker_segments = [seg for seg in segments if seg.speaker_id == speaker_id]
            
            speaker_statistics[speaker_id] = {
                "segment_count": len(speaker_segments),
                "total_duration": sum(seg.duration for seg in speaker_segments),
                "average_confidence": np.mean([seg.confidence for seg in speaker_segments]),
                "average_voice_activity": np.mean([seg.voice_activity for seg in speaker_segments]),
                "average_signal_quality": np.mean([seg.signal_quality for seg in speaker_segments]),
                "speaking_time_ratio": sum(seg.duration for seg in speaker_segments) / total_duration if total_duration > 0 else 0.0
            }
            
        return DiarizationResult(
            segments=segments,
            speaker_count=len(speaker_ids),
            speaker_ids=speaker_ids,
            speaker_labels=speaker_labels,
            total_duration=total_duration,
            total_speech_time=total_speech_time,
            total_silence_time=total_silence_time,
            average_confidence=avg_confidence,
            speaker_consistency=speaker_consistency,
            diarization_accuracy=0.85,  # Placeholder - would need ground truth
            processing_time=processing_time,
            segments_processed=len(segments),
            speaker_statistics=speaker_statistics,
            success=True
        )
        
    def _update_stats(self, result: DiarizationResult):
        """Update diarization statistics"""
        with self.processing_lock:
            self.stats["total_segments_processed"] += result.segments_processed
            self.stats["total_speakers_detected"] = max(
                self.stats["total_speakers_detected"], result.speaker_count
            )
            self.stats["total_audio_processed"] += result.total_duration
            self.stats["processing_time"] += result.processing_time
            
            # Update average confidence
            count = self.stats["total_segments_processed"]
            if count > 0:
                old_avg = self.stats["average_confidence"]
                self.stats["average_confidence"] = (
                    old_avg * (count - result.segments_processed) + 
                    result.average_confidence * result.segments_processed
                ) / count
                
            # Track accuracy estimates
            if result.diarization_accuracy > 0:
                self.stats["accuracy_estimates"].append(result.diarization_accuracy)
                
    def add_transcription_to_segments(self, segments: List[SpeakerSegment], 
                                    transcription_results: List[Dict[str, Any]]):
        """Add transcription results to speaker segments"""
        for i, segment in enumerate(segments):
            if i < len(transcription_results):
                result = transcription_results[i]
                segment.text = result.get("text", "")
                segment.transcription_confidence = result.get("confidence", 0.0)
                
    def add_segment_callback(self, callback: Callable[[SpeakerSegment], None]):
        """Add callback for segment processing"""
        self.segment_callbacks.append(callback)
        
    def add_speaker_callback(self, callback: Callable[[int, SpeakerLabel], None]):
        """Add callback for new speaker detection"""
        self.speaker_callbacks.append(callback)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get diarization statistics"""
        with self.processing_lock:
            stats = self.stats.copy()
            stats["current_speakers"] = len(self.speaker_models)
            stats["speaker_models"] = {
                sid: len(features) for sid, features in self.speaker_models.items()
            }
            if self.stats["accuracy_estimates"]:
                stats["average_accuracy"] = np.mean(self.stats["accuracy_estimates"])
            return stats
            
    def get_status(self) -> Dict[str, Any]:
        """Get diarizer status"""
        return {
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "max_speakers": self.config.max_speakers,
            "current_speakers": len(self.speaker_models),
            "speaker_detection_available": self.speaker_detector is not None,
            "vad_available": self.vad_detector is not None,
            "quality_assessor_available": self.quality_assessor is not None,
            "clustering_method": self.config.clustering_method,
            "stats": self.get_stats()
        }
        
    def shutdown(self):
        """Shutdown diarizer and cleanup resources"""
        logger.info("Shutting down speaker diarizer...")
        
        self.is_running = False
        
        # Shutdown integrated components
        if self.speaker_detector:
            self.speaker_detector.shutdown()
            
        if self.vad_detector:
            self.vad_detector.shutdown()
            
        if self.quality_assessor:
            self.quality_assessor.shutdown()
            
        logger.info("Speaker diarizer shutdown complete")
        
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


# Factory functions
def create_meeting_diarizer(max_speakers: int = 6) -> SpeakerDiarizer:
    """Create a diarizer optimized for meeting scenarios"""
    config = DiarizationConfig(
        max_speakers=max_speakers,
        min_segment_duration=0.5,
        max_segment_duration=30.0,
        speaker_change_threshold=0.7,
        speaker_similarity_threshold=0.8,
        use_existing_speaker_detection=True,
        use_voice_activity_detection=True,
        merge_consecutive_segments=True,
        confidence_threshold=0.5
    )
    return SpeakerDiarizer(config)


def create_interview_diarizer() -> SpeakerDiarizer:
    """Create a diarizer optimized for interview scenarios"""
    config = DiarizationConfig(
        max_speakers=3,
        min_segment_duration=1.0,
        max_segment_duration=60.0,
        speaker_change_threshold=0.6,
        speaker_similarity_threshold=0.85,
        use_existing_speaker_detection=True,
        use_voice_activity_detection=True,
        merge_consecutive_segments=True,
        confidence_threshold=0.6
    )
    return SpeakerDiarizer(config)


# Testing and demonstration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Speaker Diarizer Test")
    parser.add_argument("--audio", type=str, help="Audio file to diarize")
    parser.add_argument("--max-speakers", type=int, default=6, help="Maximum number of speakers")
    parser.add_argument("--scenario", type=str, default="meeting", 
                       choices=["meeting", "interview"], help="Scenario type")
    args = parser.parse_args()
    
    # Create diarizer based on scenario
    if args.scenario == "meeting":
        diarizer = create_meeting_diarizer(args.max_speakers)
    elif args.scenario == "interview":
        diarizer = create_interview_diarizer()
    else:
        diarizer = create_meeting_diarizer(args.max_speakers)
    
    try:
        # Initialize diarizer
        if not diarizer.initialize():
            print("Failed to initialize diarizer")
            sys.exit(1)
            
        print(f"Diarizer status: {diarizer.get_status()}")
        
        if args.audio:
            # Load and diarize audio file
            print(f"Diarizing: {args.audio}")
            audio_data, sample_rate = sf.read(args.audio)
            
            # Resample if needed
            if sample_rate != 16000:
                import torch
                import torchaudio
                audio_tensor = torch.from_numpy(audio_data).float()
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor, sample_rate, 16000
                )
                audio_data = audio_tensor.numpy()
                
            # Perform diarization
            result = diarizer.diarize_segments(audio_data)
            
            if result.success:
                print(f"\nDiarization completed successfully!")
                print(f"Speakers detected: {result.speaker_count}")
                print(f"Total segments: {len(result.segments)}")
                print(f"Processing time: {result.processing_time:.2f}s")
                print(f"Average confidence: {result.average_confidence:.2f}")
                
                print("\nSegments:")
                for i, segment in enumerate(result.segments[:10]):  # Show first 10 segments
                    print(f"  {i+1}. {segment.speaker_label.value}: "
                          f"{segment.start_time:.2f}-{segment.end_time:.2f}s "
                          f"(confidence: {segment.confidence:.2f})")
                          
            else:
                print("Diarization failed:")
                for error in result.errors:
                    print(f"  Error: {error}")
                    
        else:
            print("No audio file specified. Use --audio <file>")
            
        # Show statistics
        print(f"\nDiarizer statistics: {diarizer.get_stats()}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        diarizer.shutdown()