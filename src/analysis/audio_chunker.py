#!/usr/bin/env python3
"""
Audio Chunking System for The Silent Steno

This module provides real-time audio chunking for optimal processing segments,
enabling efficient transcription and analysis while maintaining audio quality.

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
import uuid

# Configure logging
logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    """Strategies for audio chunking"""
    FIXED_TIME = "fixed_time"           # Fixed time-based chunks
    VOICE_ACTIVITY = "voice_activity"   # Voice activity boundary chunks
    SPEAKER_CHANGE = "speaker_change"   # Speaker change boundary chunks
    SILENCE_BOUNDARY = "silence_boundary"  # Silence boundary chunks
    HYBRID = "hybrid"                   # Combination of strategies
    ADAPTIVE = "adaptive"               # Adaptive based on content

class ChunkPriority(Enum):
    """Priority levels for chunk processing"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

@dataclass
class ChunkConfig:
    """Configuration for Audio Chunker"""
    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    min_chunk_duration_ms: int = 1000  # Minimum chunk duration
    max_chunk_duration_ms: int = 10000  # Maximum chunk duration
    target_chunk_duration_ms: int = 3000  # Target chunk duration
    overlap_duration_ms: int = 500  # Overlap between chunks
    sample_rate: int = 16000
    channels: int = 1
    voice_activity_threshold: float = 0.6  # VAD confidence threshold
    silence_threshold_ms: int = 500  # Minimum silence for boundary
    speaker_change_threshold: float = 0.7  # Speaker change confidence
    buffer_size_ms: int = 15000  # Audio buffer size
    quality_threshold: float = 0.5  # Minimum quality for processing
    adaptive_adjustment: bool = True  # Enable adaptive chunk sizing
    enable_overlap: bool = True  # Enable chunk overlap
    
    def __post_init__(self):
        """Validate and adjust configuration parameters"""
        # Ensure minimum constraints
        self.min_chunk_duration_ms = max(500, self.min_chunk_duration_ms)
        self.max_chunk_duration_ms = max(self.min_chunk_duration_ms + 1000, 
                                       self.max_chunk_duration_ms)
        self.target_chunk_duration_ms = max(self.min_chunk_duration_ms,
                                          min(self.max_chunk_duration_ms,
                                             self.target_chunk_duration_ms))
        
        # Calculate sample counts
        self.min_chunk_samples = int(self.sample_rate * self.min_chunk_duration_ms / 1000)
        self.max_chunk_samples = int(self.sample_rate * self.max_chunk_duration_ms / 1000)
        self.target_chunk_samples = int(self.sample_rate * self.target_chunk_duration_ms / 1000)
        self.overlap_samples = int(self.sample_rate * self.overlap_duration_ms / 1000)
        self.buffer_samples = int(self.sample_rate * self.buffer_size_ms / 1000)

@dataclass
class ChunkMetadata:
    """Metadata for audio chunks"""
    chunk_id: str
    sequence_number: int
    timestamp: float
    duration_ms: float
    sample_count: int
    has_voice_activity: bool
    voice_confidence: float
    speaker_id: Optional[str]
    speaker_confidence: float
    quality_score: float
    energy_level: float
    silence_ratio: float
    processing_priority: ChunkPriority
    boundary_type: str  # 'voice', 'speaker', 'silence', 'time', 'adaptive'
    overlap_with_previous: bool
    overlap_with_next: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            'chunk_id': self.chunk_id,
            'sequence_number': self.sequence_number,
            'timestamp': self.timestamp,
            'duration_ms': self.duration_ms,
            'sample_count': self.sample_count,
            'has_voice_activity': self.has_voice_activity,
            'voice_confidence': self.voice_confidence,
            'speaker_id': self.speaker_id,
            'speaker_confidence': self.speaker_confidence,
            'quality_score': self.quality_score,
            'energy_level': self.energy_level,
            'silence_ratio': self.silence_ratio,
            'processing_priority': self.processing_priority.value,
            'boundary_type': self.boundary_type,
            'overlap_with_previous': self.overlap_with_previous,
            'overlap_with_next': self.overlap_with_next
        }

@dataclass
class AudioChunk:
    """Audio chunk with data and metadata"""
    audio_data: np.ndarray
    metadata: ChunkMetadata
    created_at: float = field(default_factory=time.time)
    processed: bool = False
    processing_results: Dict[str, Any] = field(default_factory=dict)
    
    def get_duration_seconds(self) -> float:
        """Get chunk duration in seconds"""
        return self.metadata.duration_ms / 1000.0
    
    def get_sample_rate(self) -> int:
        """Infer sample rate from data and duration"""
        if self.metadata.duration_ms > 0:
            return int(len(self.audio_data) / (self.metadata.duration_ms / 1000.0))
        return 16000  # Default
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary (without audio data)"""
        return {
            'metadata': self.metadata.to_dict(),
            'created_at': self.created_at,
            'processed': self.processed,
            'processing_results': self.processing_results
        }

class AudioChunker:
    """
    Real-time audio chunking system for optimal processing segments
    
    This chunker analyzes incoming audio and creates optimally-sized chunks
    based on voice activity, speaker changes, and silence boundaries.
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        Initialize Audio Chunker
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkConfig()
        
        # State tracking
        self.is_running = False
        self.sequence_number = 0
        self.current_chunk_start = None
        self.audio_buffer = np.array([], dtype=np.int16)
        self.buffer_start_time = None
        
        # Analysis state
        self.current_voice_activity = False
        self.current_speaker_id = None
        self.silence_start = None
        self.last_boundary_time = None
        
        # Threading
        self.processing_thread = None
        self.thread_lock = threading.Lock()
        
        # Callbacks
        self.chunk_ready_callbacks: List[Callable[[AudioChunk], None]] = []
        self.boundary_callbacks: List[Callable[[str, float], None]] = []
        
        # Chunk storage
        self.completed_chunks: deque = deque(maxlen=100)  # Keep last 100 chunks
        self.pending_chunks: List[AudioChunk] = []
        
        # Statistics
        self.stats = {
            'total_chunks': 0,
            'voice_chunks': 0,
            'silence_chunks': 0,
            'speaker_boundary_chunks': 0,
            'time_boundary_chunks': 0,
            'adaptive_chunks': 0,
            'total_audio_processed_ms': 0.0,
            'average_chunk_duration_ms': 0.0,
            'start_time': None
        }
        
        logger.info(f"AudioChunker initialized with strategy={self.config.strategy.value}")
    
    def add_chunk_ready_callback(self, callback: Callable[[AudioChunk], None]) -> None:
        """Add callback for when chunks are ready"""
        self.chunk_ready_callbacks.append(callback)
    
    def add_boundary_callback(self, callback: Callable[[str, float], None]) -> None:
        """Add callback for boundary detection"""
        self.boundary_callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> bool:
        """Remove a callback from all callback lists"""
        removed = False
        for callback_list in [self.chunk_ready_callbacks, self.boundary_callbacks]:
            if callback in callback_list:
                callback_list.remove(callback)
                removed = True
        return removed
    
    def _calculate_audio_quality(self, audio_data: np.ndarray) -> float:
        """Calculate quality score for audio chunk"""
        try:
            if len(audio_data) == 0:
                return 0.0
            
            # Energy-based quality
            energy = np.mean(audio_data.astype(np.float64) ** 2)
            energy_score = min(energy / 1000000, 1.0)  # Normalize
            
            # Signal-to-noise ratio estimate
            signal_mean = np.mean(np.abs(audio_data))
            signal_std = np.std(audio_data.astype(np.float64))
            if signal_std > 0:
                snr_estimate = signal_mean / signal_std
                snr_score = min(snr_estimate / 100, 1.0)
            else:
                snr_score = 0.0
            
            # Dynamic range
            dynamic_range = np.max(audio_data) - np.min(audio_data)
            range_score = min(dynamic_range / 32767, 1.0)
            
            # Combined quality score
            quality = (energy_score + snr_score + range_score) / 3.0
            return float(np.clip(quality, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating audio quality: {e}")
            return 0.5
    
    def _calculate_silence_ratio(self, audio_data: np.ndarray) -> float:
        """Calculate ratio of silence in audio chunk"""
        try:
            if len(audio_data) == 0:
                return 1.0
            
            # Threshold for silence (10% of max possible value)
            silence_threshold = 3276  # 32767 * 0.1
            
            # Count samples below threshold
            silence_samples = np.sum(np.abs(audio_data) < silence_threshold)
            silence_ratio = silence_samples / len(audio_data)
            
            return float(silence_ratio)
            
        except Exception as e:
            logger.error(f"Error calculating silence ratio: {e}")
            return 0.5
    
    def _calculate_energy_level(self, audio_data: np.ndarray) -> float:
        """Calculate normalized energy level"""
        try:
            if len(audio_data) == 0:
                return 0.0
            
            energy = np.mean(audio_data.astype(np.float64) ** 2)
            # Normalize to 0-1 range (assuming max energy around 10^9)
            normalized_energy = min(energy / 1000000000, 1.0)
            
            return float(normalized_energy)
            
        except Exception as e:
            logger.error(f"Error calculating energy level: {e}")
            return 0.0
    
    def _determine_processing_priority(self, has_voice: bool, voice_confidence: float,
                                     speaker_change: bool, quality: float) -> ChunkPriority:
        """Determine processing priority for chunk"""
        if not has_voice:
            return ChunkPriority.LOW
        
        if speaker_change:
            return ChunkPriority.URGENT
        
        if voice_confidence > 0.8 and quality > 0.7:
            return ChunkPriority.HIGH
        elif voice_confidence > 0.6:
            return ChunkPriority.MEDIUM
        else:
            return ChunkPriority.LOW
    
    def _should_create_chunk(self, current_time: float, voice_activity: bool,
                            speaker_id: Optional[str], voice_confidence: float) -> Tuple[bool, str]:
        """Determine if a chunk should be created based on current conditions"""
        if self.current_chunk_start is None:
            return False, "no_chunk_start"
        
        chunk_duration_ms = (current_time - self.current_chunk_start) * 1000
        
        # Check minimum duration
        if chunk_duration_ms < self.config.min_chunk_duration_ms:
            return False, "min_duration"
        
        # Check maximum duration (force chunk)
        if chunk_duration_ms >= self.config.max_chunk_duration_ms:
            return True, "max_duration"
        
        # Strategy-based decisions
        if self.config.strategy == ChunkingStrategy.FIXED_TIME:
            if chunk_duration_ms >= self.config.target_chunk_duration_ms:
                return True, "fixed_time"
        
        elif self.config.strategy == ChunkingStrategy.VOICE_ACTIVITY:
            # Create chunk on voice activity boundaries
            if (self.current_voice_activity and not voice_activity and 
                voice_confidence < self.config.voice_activity_threshold):
                if chunk_duration_ms >= self.config.min_chunk_duration_ms:
                    return True, "voice_boundary"
        
        elif self.config.strategy == ChunkingStrategy.SPEAKER_CHANGE:
            # Create chunk on speaker changes
            if (self.current_speaker_id is not None and 
                speaker_id != self.current_speaker_id):
                return True, "speaker_change"
        
        elif self.config.strategy == ChunkingStrategy.SILENCE_BOUNDARY:
            # Create chunk after silence periods
            if not voice_activity:
                if self.silence_start is None:
                    self.silence_start = current_time
                elif (current_time - self.silence_start) * 1000 >= self.config.silence_threshold_ms:
                    return True, "silence_boundary"
            else:
                self.silence_start = None
        
        elif self.config.strategy == ChunkingStrategy.HYBRID:
            # Combination of strategies
            
            # Speaker change (highest priority)
            if (self.current_speaker_id is not None and 
                speaker_id != self.current_speaker_id):
                return True, "speaker_change"
            
            # Voice activity boundary
            if (self.current_voice_activity and not voice_activity and
                chunk_duration_ms >= self.config.min_chunk_duration_ms):
                return True, "voice_boundary"
            
            # Silence boundary
            if not voice_activity:
                if self.silence_start is None:
                    self.silence_start = current_time
                elif ((current_time - self.silence_start) * 1000 >= self.config.silence_threshold_ms and
                      chunk_duration_ms >= self.config.min_chunk_duration_ms):
                    return True, "silence_boundary"
            else:
                self.silence_start = None
            
            # Time-based fallback
            if chunk_duration_ms >= self.config.target_chunk_duration_ms:
                return True, "time_fallback"
        
        elif self.config.strategy == ChunkingStrategy.ADAPTIVE:
            # Adaptive strategy based on content
            target_duration = self.config.target_chunk_duration_ms
            
            # Adjust target based on voice activity and quality
            if voice_activity and voice_confidence > 0.8:
                # High quality speech - can use longer chunks
                target_duration = min(target_duration * 1.5, self.config.max_chunk_duration_ms)
            elif not voice_activity:
                # Silence - use shorter chunks
                target_duration = max(target_duration * 0.5, self.config.min_chunk_duration_ms)
            
            if chunk_duration_ms >= target_duration:
                return True, "adaptive"
        
        return False, "continue"
    
    def _create_chunk(self, end_time: float, boundary_type: str) -> Optional[AudioChunk]:
        """Create audio chunk from current buffer"""
        try:
            if self.current_chunk_start is None or len(self.audio_buffer) == 0:
                return None
            
            # Calculate chunk timing
            chunk_duration_ms = (end_time - self.current_chunk_start) * 1000
            samples_per_ms = self.config.sample_rate / 1000.0
            chunk_samples = int(chunk_duration_ms * samples_per_ms)
            
            # Extract audio data
            if chunk_samples > len(self.audio_buffer):
                chunk_samples = len(self.audio_buffer)
            
            chunk_audio = self.audio_buffer[:chunk_samples].copy()
            
            # Remove processed samples from buffer (with overlap if enabled)
            if self.config.enable_overlap and chunk_samples > self.config.overlap_samples:
                keep_samples = self.config.overlap_samples
                self.audio_buffer = self.audio_buffer[chunk_samples - keep_samples:]
                # Adjust buffer start time
                removed_samples = chunk_samples - keep_samples
                if self.buffer_start_time is not None:
                    self.buffer_start_time += removed_samples / self.config.sample_rate
            else:
                self.audio_buffer = self.audio_buffer[chunk_samples:]
                if self.buffer_start_time is not None:
                    self.buffer_start_time += chunk_samples / self.config.sample_rate
            
            # Analyze chunk
            quality_score = self._calculate_audio_quality(chunk_audio)
            energy_level = self._calculate_energy_level(chunk_audio)
            silence_ratio = self._calculate_silence_ratio(chunk_audio)
            
            # Determine voice activity and speaker for chunk
            has_voice = silence_ratio < 0.7  # Simple heuristic
            voice_confidence = 1.0 - silence_ratio if has_voice else 0.0
            
            # Create metadata
            chunk_id = str(uuid.uuid4())
            self.sequence_number += 1
            
            priority = self._determine_processing_priority(
                has_voice, voice_confidence, 
                boundary_type == "speaker_change", quality_score)
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                sequence_number=self.sequence_number,
                timestamp=self.current_chunk_start,
                duration_ms=chunk_duration_ms,
                sample_count=len(chunk_audio),
                has_voice_activity=has_voice,
                voice_confidence=voice_confidence,
                speaker_id=self.current_speaker_id,
                speaker_confidence=0.8 if self.current_speaker_id else 0.0,
                quality_score=quality_score,
                energy_level=energy_level,
                silence_ratio=silence_ratio,
                processing_priority=priority,
                boundary_type=boundary_type,
                overlap_with_previous=self.config.enable_overlap and self.sequence_number > 1,
                overlap_with_next=False  # Will be set when next chunk is created
            )
            
            # Create chunk
            chunk = AudioChunk(
                audio_data=chunk_audio,
                metadata=metadata
            )
            
            # Update statistics
            with self.thread_lock:
                self.stats['total_chunks'] += 1
                if has_voice:
                    self.stats['voice_chunks'] += 1
                else:
                    self.stats['silence_chunks'] += 1
                
                if boundary_type == "speaker_change":
                    self.stats['speaker_boundary_chunks'] += 1
                elif boundary_type in ["fixed_time", "time_fallback"]:
                    self.stats['time_boundary_chunks'] += 1
                elif boundary_type == "adaptive":
                    self.stats['adaptive_chunks'] += 1
                
                self.stats['total_audio_processed_ms'] += chunk_duration_ms
                self.stats['average_chunk_duration_ms'] = (
                    self.stats['total_audio_processed_ms'] / self.stats['total_chunks'])
            
            # Update overlap flag for previous chunk
            if len(self.completed_chunks) > 0:
                prev_chunk = self.completed_chunks[-1]
                prev_chunk.metadata.overlap_with_next = self.config.enable_overlap
            
            # Store chunk
            self.completed_chunks.append(chunk)
            
            logger.debug(f"Created chunk {chunk_id}: {chunk_duration_ms:.1f}ms, "
                        f"boundary={boundary_type}, quality={quality_score:.2f}")
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error creating chunk: {e}")
            return None
    
    def add_audio_data(self, audio_data: np.ndarray, timestamp: Optional[float] = None,
                      voice_activity: bool = False, voice_confidence: float = 0.0,
                      speaker_id: Optional[str] = None) -> Optional[AudioChunk]:
        """
        Add audio data to chunker for processing
        
        Args:
            audio_data: Audio data as numpy array
            timestamp: Timestamp for the audio data
            voice_activity: Whether voice activity is detected
            voice_confidence: Confidence of voice activity detection
            speaker_id: Current speaker ID
        
        Returns:
            AudioChunk if a chunk was completed, None otherwise
        """
        if timestamp is None:
            timestamp = time.time()
        
        try:
            # Ensure audio data is correct format
            if audio_data.dtype != np.int16:
                audio_data = audio_data.astype(np.int16)
            
            # Add to buffer
            if len(self.audio_buffer) == 0:
                self.audio_buffer = audio_data.copy()
                self.buffer_start_time = timestamp
            else:
                self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])
            
            # Initialize chunk tracking
            if self.current_chunk_start is None:
                self.current_chunk_start = timestamp
                self.current_voice_activity = voice_activity
                self.current_speaker_id = speaker_id
            
            # Check if we should create a chunk
            should_chunk, boundary_type = self._should_create_chunk(
                timestamp, voice_activity, speaker_id, voice_confidence)
            
            if should_chunk:
                # Create chunk
                chunk = self._create_chunk(timestamp, boundary_type)
                
                if chunk is not None:
                    # Fire boundary callbacks
                    for callback in self.boundary_callbacks:
                        try:
                            callback(boundary_type, timestamp)
                        except Exception as e:
                            logger.error(f"Error in boundary callback: {e}")
                    
                    # Fire chunk ready callbacks
                    for callback in self.chunk_ready_callbacks:
                        try:
                            callback(chunk)
                        except Exception as e:
                            logger.error(f"Error in chunk ready callback: {e}")
                    
                    # Reset for next chunk
                    self.current_chunk_start = timestamp
                    
                    # Update state
                    self.current_voice_activity = voice_activity
                    self.current_speaker_id = speaker_id
                    self.silence_start = None
                    
                    return chunk
            
            else:
                # Update state
                self.current_voice_activity = voice_activity
                if speaker_id is not None:
                    self.current_speaker_id = speaker_id
            
            # Prevent buffer from growing too large
            max_buffer_samples = self.config.buffer_samples
            if len(self.audio_buffer) > max_buffer_samples:
                # Keep only the most recent samples
                excess_samples = len(self.audio_buffer) - max_buffer_samples
                self.audio_buffer = self.audio_buffer[excess_samples:]
                if self.buffer_start_time is not None:
                    self.buffer_start_time += excess_samples / self.config.sample_rate
            
            return None
            
        except Exception as e:
            logger.error(f"Error adding audio data: {e}")
            return None
    
    def force_chunk_creation(self, timestamp: Optional[float] = None) -> Optional[AudioChunk]:
        """Force creation of chunk from current buffer"""
        if timestamp is None:
            timestamp = time.time()
        
        return self._create_chunk(timestamp, "forced")
    
    def get_pending_chunks(self) -> List[AudioChunk]:
        """Get list of chunks ready for processing"""
        with self.thread_lock:
            return self.pending_chunks.copy()
    
    def get_completed_chunks(self) -> List[AudioChunk]:
        """Get list of completed chunks"""
        with self.thread_lock:
            return list(self.completed_chunks)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get chunking statistics"""
        with self.thread_lock:
            stats = self.stats.copy()
            
            # Add derived statistics
            if stats['total_chunks'] > 0:
                stats['voice_chunk_percentage'] = (stats['voice_chunks'] / stats['total_chunks']) * 100
                stats['silence_chunk_percentage'] = (stats['silence_chunks'] / stats['total_chunks']) * 100
            else:
                stats['voice_chunk_percentage'] = 0.0
                stats['silence_chunk_percentage'] = 0.0
            
            stats['buffer_size_samples'] = len(self.audio_buffer)
            stats['buffer_duration_ms'] = (len(self.audio_buffer) / self.config.sample_rate) * 1000
            
            if self.current_chunk_start is not None:
                current_chunk_duration = (time.time() - self.current_chunk_start) * 1000
                stats['current_chunk_duration_ms'] = current_chunk_duration
            else:
                stats['current_chunk_duration_ms'] = 0.0
        
        return stats
    
    def start_processing(self) -> bool:
        """Start chunking processing"""
        if self.is_running:
            logger.warning("Audio chunking already running")
            return False
        
        with self.thread_lock:
            self.is_running = True
            self.stats['start_time'] = time.time()
        
        logger.info("Audio chunking started")
        return True
    
    def stop_processing(self) -> bool:
        """Stop chunking processing and finalize current chunk"""
        if not self.is_running:
            logger.warning("Audio chunking not running")
            return False
        
        with self.thread_lock:
            self.is_running = False
            
            # Finalize current chunk if it has content
            if self.current_chunk_start is not None and len(self.audio_buffer) > 0:
                chunk = self._create_chunk(time.time(), "final")
                if chunk is not None:
                    # Fire callbacks for final chunk
                    for callback in self.chunk_ready_callbacks:
                        try:
                            callback(chunk)
                        except Exception as e:
                            logger.error(f"Error in final chunk callback: {e}")
        
        logger.info("Audio chunking stopped")
        return True
    
    def reset_chunker(self) -> None:
        """Reset chunker state"""
        with self.thread_lock:
            self.audio_buffer = np.array([], dtype=np.int16)
            self.buffer_start_time = None
            self.current_chunk_start = None
            self.current_voice_activity = False
            self.current_speaker_id = None
            self.silence_start = None
            self.sequence_number = 0
            self.completed_chunks.clear()
            self.pending_chunks.clear()
            
            # Reset statistics
            self.stats = {
                'total_chunks': 0,
                'voice_chunks': 0,
                'silence_chunks': 0,
                'speaker_boundary_chunks': 0,
                'time_boundary_chunks': 0,
                'adaptive_chunks': 0,
                'total_audio_processed_ms': 0.0,
                'average_chunk_duration_ms': 0.0,
                'start_time': time.time() if self.is_running else None
            }
        
        logger.info("Audio chunker reset")
    
    def is_processing(self) -> bool:
        """Check if chunker is currently processing"""
        return self.is_running

def create_audio_chunker(config: Optional[ChunkConfig] = None) -> AudioChunker:
    """
    Factory function to create a configured audio chunker
    
    Args:
        config: Optional chunking configuration
        
    Returns:
        Configured AudioChunker instance
    """
    return AudioChunker(config)

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create chunker
    config = ChunkConfig(
        strategy=ChunkingStrategy.HYBRID,
        min_chunk_duration_ms=1000,
        max_chunk_duration_ms=5000,
        target_chunk_duration_ms=3000
    )
    
    chunker = create_audio_chunker(config)
    
    # Add example callbacks
    def on_chunk_ready(chunk: AudioChunk):
        print(f"Chunk ready: {chunk.metadata.chunk_id} "
              f"({chunk.metadata.duration_ms:.0f}ms, "
              f"boundary: {chunk.metadata.boundary_type})")
    
    def on_boundary(boundary_type: str, timestamp: float):
        print(f"Boundary detected: {boundary_type} at {timestamp:.3f}")
    
    chunker.add_chunk_ready_callback(on_chunk_ready)
    chunker.add_boundary_callback(on_boundary)
    
    # Start processing
    chunker.start_processing()
    
    # Test with synthetic audio data
    try:
        print("Testing audio chunker with synthetic data...")
        
        frame_size = 1600  # 100ms at 16kHz
        
        # Simulate audio stream with different patterns
        for i in range(50):
            # Create audio frame
            if i < 10:
                # Voice activity
                audio_frame = np.random.randint(-1000, 1000, frame_size, dtype=np.int16)
                voice_activity = True
                voice_confidence = 0.8
                speaker_id = "speaker_001"
            elif i < 15:
                # Silence
                audio_frame = np.random.randint(-50, 50, frame_size, dtype=np.int16)
                voice_activity = False
                voice_confidence = 0.1
                speaker_id = None
            elif i < 35:
                # Different speaker
                audio_frame = np.random.randint(-800, 800, frame_size, dtype=np.int16)
                voice_activity = True
                voice_confidence = 0.9
                speaker_id = "speaker_002"
            else:
                # Back to first speaker
                audio_frame = np.random.randint(-1000, 1000, frame_size, dtype=np.int16)
                voice_activity = True
                voice_confidence = 0.8
                speaker_id = "speaker_001"
            
            chunk = chunker.add_audio_data(
                audio_frame, 
                voice_activity=voice_activity,
                voice_confidence=voice_confidence,
                speaker_id=speaker_id
            )
            
            time.sleep(0.1)  # 100ms delay
        
        # Force final chunk
        final_chunk = chunker.force_chunk_creation()
        if final_chunk:
            print(f"Final chunk: {final_chunk.metadata.duration_ms:.0f}ms")
        
        # Print statistics
        stats = chunker.get_statistics()
        print(f"\nStatistics:")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Voice chunks: {stats['voice_chunks']} ({stats['voice_chunk_percentage']:.1f}%)")
        print(f"Average duration: {stats['average_chunk_duration_ms']:.0f}ms")
        print(f"Speaker boundary chunks: {stats['speaker_boundary_chunks']}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    finally:
        chunker.stop_processing()
        print("Audio chunker test completed")