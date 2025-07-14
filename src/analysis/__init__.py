#!/usr/bin/env python3
"""
Audio Analysis Module for The Silent Steno

This module provides comprehensive real-time audio analysis capabilities including
voice activity detection, speaker identification, audio chunking, quality assessment,
silence detection, and statistics collection.

The analysis system is designed to work with the existing audio pipeline and recording
system to provide intelligent audio processing for transcription optimization.

Author: The Silent Steno Team
License: MIT
"""

import logging
from typing import Optional, Dict, Any

# Import all main classes and functions
from .voice_activity_detector import (
    VoiceActivityDetector, VADConfig, VADResult, VADMode, VADSensitivity,
    create_vad_system
)

from .speaker_detector import (
    SpeakerDetector, SpeakerConfig, SpeakerResult, SpeakerFeatures, 
    SpeakerChangeDetection, SpeakerChangeMethod, SpeakerConfidence,
    create_speaker_detector
)

from .audio_chunker import (
    AudioChunker, ChunkConfig, AudioChunk, ChunkMetadata, ChunkingStrategy,
    ChunkPriority, create_audio_chunker
)

from .quality_assessor import (
    QualityAssessor, QualityConfig, QualityResult, QualityMetrics, 
    QualityThresholds, QualityLevel, QualityMetric,
    create_quality_assessor
)

from .silence_detector import (
    SilenceDetector, SilenceConfig, SilenceResult, SilenceSegment, TrimResult,
    SilenceMethod, SilenceMode, SilenceThreshold, TrimMode,
    create_silence_detector
)

from .statistics_collector import (
    StatisticsCollector, StatisticsConfig, AudioStatistics, SpeakingTimeStats,
    ParticipationMetrics, SpeakerEvent, MetricType, IntervalType,
    create_statistics_collector
)

# Configure module logger
logger = logging.getLogger(__name__)

# Module version
__version__ = "1.0.0"

# Module exports
__all__ = [
    # Voice Activity Detection
    'VoiceActivityDetector', 'VADConfig', 'VADResult', 'VADMode', 'VADSensitivity',
    'create_vad_system',
    
    # Speaker Detection
    'SpeakerDetector', 'SpeakerConfig', 'SpeakerResult', 'SpeakerFeatures',
    'SpeakerChangeDetection', 'SpeakerChangeMethod', 'SpeakerConfidence',
    'create_speaker_detector',
    
    # Audio Chunking
    'AudioChunker', 'ChunkConfig', 'AudioChunk', 'ChunkMetadata', 'ChunkingStrategy',
    'ChunkPriority', 'create_audio_chunker',
    
    # Quality Assessment
    'QualityAssessor', 'QualityConfig', 'QualityResult', 'QualityMetrics',
    'QualityThresholds', 'QualityLevel', 'QualityMetric',
    'create_quality_assessor',
    
    # Silence Detection
    'SilenceDetector', 'SilenceConfig', 'SilenceResult', 'SilenceSegment', 'TrimResult',
    'SilenceMethod', 'SilenceMode', 'SilenceThreshold', 'TrimMode',
    'create_silence_detector',
    
    # Statistics Collection
    'StatisticsCollector', 'StatisticsConfig', 'AudioStatistics', 'SpeakingTimeStats',
    'ParticipationMetrics', 'SpeakerEvent', 'MetricType', 'IntervalType',
    'create_statistics_collector',
    
    # Factory functions
    'create_analysis_system', 'create_integrated_analyzer'
]

def create_analysis_system(vad_config: Optional[VADConfig] = None,
                          speaker_config: Optional[SpeakerConfig] = None,
                          chunker_config: Optional[ChunkConfig] = None,
                          quality_config: Optional[QualityConfig] = None,
                          silence_config: Optional[SilenceConfig] = None,
                          stats_config: Optional[StatisticsConfig] = None) -> Dict[str, Any]:
    """
    Create a complete audio analysis system with all components
    
    Args:
        vad_config: Voice activity detection configuration
        speaker_config: Speaker detection configuration
        chunker_config: Audio chunking configuration
        quality_config: Quality assessment configuration
        silence_config: Silence detection configuration
        stats_config: Statistics collection configuration
    
    Returns:
        Dictionary containing all analysis components
    """
    try:
        # Create all analysis components
        vad = create_vad_system(vad_config)
        speaker_detector = create_speaker_detector(speaker_config)
        chunker = create_audio_chunker(chunker_config)
        quality_assessor = create_quality_assessor(quality_config)
        silence_detector = create_silence_detector(silence_config)
        stats_collector = create_statistics_collector(stats_config)
        
        analysis_system = {
            'vad': vad,
            'speaker_detector': speaker_detector,
            'chunker': chunker,
            'quality_assessor': quality_assessor,
            'silence_detector': silence_detector,
            'statistics_collector': stats_collector,
            'version': __version__
        }
        
        logger.info("Complete audio analysis system created successfully")
        return analysis_system
        
    except Exception as e:
        logger.error(f"Error creating analysis system: {e}")
        raise

class IntegratedAnalyzer:
    """
    Integrated audio analyzer that coordinates all analysis components
    
    This class provides a unified interface for real-time audio analysis,
    coordinating voice activity detection, speaker identification, quality
    assessment, and statistics collection.
    """
    
    def __init__(self, 
                 vad_config: Optional[VADConfig] = None,
                 speaker_config: Optional[SpeakerConfig] = None,
                 chunker_config: Optional[ChunkConfig] = None,
                 quality_config: Optional[QualityConfig] = None,
                 silence_config: Optional[SilenceConfig] = None,
                 stats_config: Optional[StatisticsConfig] = None):
        """
        Initialize integrated analyzer
        
        Args:
            vad_config: Voice activity detection configuration
            speaker_config: Speaker detection configuration  
            chunker_config: Audio chunking configuration
            quality_config: Quality assessment configuration
            silence_config: Silence detection configuration
            stats_config: Statistics collection configuration
        """
        self.components = create_analysis_system(
            vad_config, speaker_config, chunker_config,
            quality_config, silence_config, stats_config
        )
        
        self.is_running = False
        self._setup_component_integration()
        
        logger.info("IntegratedAnalyzer initialized")
    
    def _setup_component_integration(self):
        """Setup integration between analysis components"""
        try:
            # Connect VAD to speaker detector
            self.components['vad'].add_result_callback(self._on_vad_result)
            
            # Connect speaker detector to chunker
            self.components['speaker_detector'].add_speaker_change_callback(self._on_speaker_change)
            
            # Connect quality assessor to statistics
            self.components['quality_assessor'].add_quality_callback(self._on_quality_result)
            
            # Connect chunker to statistics
            self.components['chunker'].add_chunk_ready_callback(self._on_chunk_ready)
            
            logger.debug("Component integration setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up component integration: {e}")
    
    def _on_vad_result(self, vad_result: VADResult):
        """Handle VAD results and forward to other components"""
        try:
            # Update statistics with voice activity
            self.components['statistics_collector'].update_speaker_activity(
                speaker_id=None,  # Will be set by speaker detector
                timestamp=vad_result.timestamp,
                voice_activity=vad_result.is_voice,
                confidence=vad_result.confidence
            )
            
        except Exception as e:
            logger.error(f"Error handling VAD result: {e}")
    
    def _on_speaker_change(self, change_detection):
        """Handle speaker change events"""
        try:
            # Update statistics with speaker information
            self.components['statistics_collector'].update_speaker_activity(
                speaker_id=change_detection.to_speaker,
                timestamp=change_detection.timestamp,
                voice_activity=True,
                confidence=change_detection.confidence
            )
            
        except Exception as e:
            logger.error(f"Error handling speaker change: {e}")
    
    def _on_quality_result(self, quality_result: QualityResult):
        """Handle quality assessment results"""
        try:
            # Could integrate quality metrics with other components
            # For now, just log significant quality issues
            if quality_result.metrics.quality_level == QualityLevel.POOR:
                logger.warning(f"Poor audio quality detected: {quality_result.get_quality_summary()}")
                
        except Exception as e:
            logger.error(f"Error handling quality result: {e}")
    
    def _on_chunk_ready(self, chunk: AudioChunk):
        """Handle completed audio chunks"""
        try:
            # Could trigger downstream processing like transcription
            logger.debug(f"Audio chunk ready: {chunk.metadata.chunk_id} "
                        f"({chunk.metadata.duration_ms:.0f}ms)")
                        
        except Exception as e:
            logger.error(f"Error handling chunk ready: {e}")
    
    def analyze_audio_frame(self, audio_data, timestamp=None, **kwargs) -> Dict[str, Any]:
        """
        Analyze audio frame with all components
        
        Args:
            audio_data: Audio data as numpy array
            timestamp: Timestamp for the frame
            **kwargs: Additional parameters for specific components
        
        Returns:
            Dictionary with results from all analysis components
        """
        if timestamp is None:
            import time
            timestamp = time.time()
        
        results = {}
        
        try:
            # Voice activity detection
            vad_result = self.components['vad'].detect_voice_activity(audio_data, timestamp)
            results['vad'] = vad_result
            
            # Speaker detection (if voice activity detected)
            if vad_result.is_voice:
                speaker_result = self.components['speaker_detector'].analyze_speaker(
                    audio_data, timestamp, vad_result.is_voice)
                results['speaker'] = speaker_result
                
                # Update chunker with speaker info
                chunk = self.components['chunker'].add_audio_data(
                    audio_data, timestamp, vad_result.is_voice, 
                    vad_result.confidence, speaker_result.speaker_id)
                if chunk:
                    results['chunk'] = chunk
            else:
                # No voice activity - still update chunker
                chunk = self.components['chunker'].add_audio_data(
                    audio_data, timestamp, False, 0.0, None)
                if chunk:
                    results['chunk'] = chunk
            
            # Quality assessment
            quality_result = self.components['quality_assessor'].assess_quality(audio_data, timestamp)
            results['quality'] = quality_result
            
            # Silence detection
            silence_result = self.components['silence_detector'].detect_silence(audio_data, timestamp)
            results['silence'] = silence_result
            
            return results
            
        except Exception as e:
            logger.error(f"Error in integrated audio analysis: {e}")
            return {'error': str(e)}
    
    def start_processing(self) -> bool:
        """Start all analysis components"""
        if self.is_running:
            logger.warning("Integrated analyzer already running")
            return False
        
        try:
            # Start all components
            self.components['vad'].start_processing()
            self.components['speaker_detector'].start_processing()
            self.components['chunker'].start_processing()
            self.components['quality_assessor'].start_processing()
            self.components['silence_detector'].start_processing()
            self.components['statistics_collector'].start_processing()
            
            self.is_running = True
            logger.info("Integrated analyzer started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting integrated analyzer: {e}")
            return False
    
    def stop_processing(self) -> bool:
        """Stop all analysis components"""
        if not self.is_running:
            logger.warning("Integrated analyzer not running")
            return False
        
        try:
            # Stop all components
            self.components['vad'].stop_processing()
            self.components['speaker_detector'].stop_processing()
            self.components['chunker'].stop_processing()
            self.components['quality_assessor'].stop_processing()
            self.components['silence_detector'].stop_processing()
            self.components['statistics_collector'].stop_processing()
            
            self.is_running = False
            logger.info("Integrated analyzer stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping integrated analyzer: {e}")
            return False
    
    def get_component(self, component_name: str):
        """Get specific analysis component"""
        return self.components.get(component_name)
    
    def get_current_statistics(self) -> Optional[AudioStatistics]:
        """Get current audio statistics"""
        return self.components['statistics_collector'].get_current_statistics()
    
    def reset_all_components(self):
        """Reset state of all analysis components"""
        try:
            self.components['speaker_detector'].reset_speaker_models()
            self.components['chunker'].reset_chunker()
            self.components['quality_assessor'].reset_statistics()
            self.components['silence_detector'].reset_detector()
            self.components['statistics_collector'].reset_statistics()
            
            logger.info("All analysis components reset")
            
        except Exception as e:
            logger.error(f"Error resetting components: {e}")

def create_integrated_analyzer(**kwargs) -> IntegratedAnalyzer:
    """
    Factory function to create an integrated analyzer
    
    Args:
        **kwargs: Configuration arguments for component creation
    
    Returns:
        IntegratedAnalyzer instance
    """
    return IntegratedAnalyzer(**kwargs)

# Default configurations for quick setup
DEFAULT_VAD_CONFIG = VADConfig(
    mode=VADMode.AGGRESSIVE,
    sensitivity=VADSensitivity.MEDIUM,
    sample_rate=16000
)

DEFAULT_SPEAKER_CONFIG = SpeakerConfig(
    method=SpeakerChangeMethod.COMBINED,
    sample_rate=16000,
    change_threshold=0.3
)

DEFAULT_CHUNKER_CONFIG = ChunkConfig(
    strategy=ChunkingStrategy.HYBRID,
    min_chunk_duration_ms=1000,
    max_chunk_duration_ms=5000,
    target_chunk_duration_ms=3000
)

DEFAULT_QUALITY_CONFIG = QualityConfig(
    sample_rate=16000,
    enable_real_time=True
)

DEFAULT_SILENCE_CONFIG = SilenceConfig(
    method=SilenceMethod.COMBINED,
    mode=SilenceMode.BALANCED,
    trim_mode=TrimMode.BOTH
)

DEFAULT_STATS_CONFIG = StatisticsConfig(
    sample_rate=16000,
    update_interval_ms=1000,
    enable_real_time_updates=True
)

def create_default_analysis_system() -> Dict[str, Any]:
    """Create analysis system with default configurations"""
    return create_analysis_system(
        DEFAULT_VAD_CONFIG,
        DEFAULT_SPEAKER_CONFIG,
        DEFAULT_CHUNKER_CONFIG,
        DEFAULT_QUALITY_CONFIG,
        DEFAULT_SILENCE_CONFIG,
        DEFAULT_STATS_CONFIG
    )

def create_default_integrated_analyzer() -> IntegratedAnalyzer:
    """Create integrated analyzer with default configurations"""
    return IntegratedAnalyzer(
        DEFAULT_VAD_CONFIG,
        DEFAULT_SPEAKER_CONFIG,
        DEFAULT_CHUNKER_CONFIG,
        DEFAULT_QUALITY_CONFIG,
        DEFAULT_SILENCE_CONFIG,
        DEFAULT_STATS_CONFIG
    )

# Module initialization
logger.info(f"Audio Analysis Module v{__version__} loaded successfully")
logger.info("Available components: VAD, Speaker Detection, Audio Chunking, "
           "Quality Assessment, Silence Detection, Statistics Collection")