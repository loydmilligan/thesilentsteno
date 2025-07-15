#!/usr/bin/env python3
"""
AI Module - Local Whisper Integration

Complete AI transcription system with local Whisper Base model, speaker diarization,
and performance optimization for Raspberry Pi 5.

Author: Claude AI Assistant
Date: 2024-07-14
Version: 1.0
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List, Callable, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "Claude AI Assistant"
__description__ = "AI transcription system with local Whisper integration"

# Import core components
try:
    from .whisper_transcriber import (
        WhisperTranscriber, 
        TranscriptionResult, 
        TranscriptionConfig,
        ModelSize,
        WhisperModel,
        create_base_transcriber,
        create_optimized_transcriber
    )
    
    from .transcription_pipeline import (
        TranscriptionPipeline,
        PipelineConfig,
        PipelineResult,
        ProcessingMode,
        QualitySettings,
        create_realtime_pipeline,
        create_batch_pipeline
    )
    
    from .audio_chunker import (
        AIAudioChunker,
        TranscriptionChunkConfig,
        OptimalChunk,
        ChunkType,
        ChunkQuality,
        create_default_chunker,
        create_realtime_chunker,
        create_quality_chunker
    )
    
    from .speaker_diarizer import (
        SpeakerDiarizer,
        SpeakerSegment,
        DiarizationConfig,
        DiarizationResult,
        SpeakerLabel,
        create_meeting_diarizer,
        create_interview_diarizer
    )
    
    from .transcript_formatter import (
        TranscriptFormatter,
        TranscriptSegment,
        FormattingConfig,
        FormattedTranscript,
        OutputFormat,
        TimestampFormat,
        create_text_formatter,
        create_subtitle_formatter,
        create_meeting_formatter
    )
    
    from .performance_optimizer import (
        PerformanceOptimizer,
        OptimizationConfig,
        OptimizationResult,
        SystemMetrics,
        OptimizationLevel,
        PerformanceMode,
        create_realtime_optimizer,
        create_power_efficient_optimizer,
        create_balanced_optimizer
    )
    
    logger.info("All AI components imported successfully")
    
except ImportError as e:
    logger.error(f"Failed to import AI components: {e}")
    raise


# Main AI system class
class AITranscriptionSystem:
    """
    Complete AI transcription system integrating all components
    """
    
    def __init__(self, 
                 transcription_config: Optional[TranscriptionConfig] = None,
                 pipeline_config: Optional[PipelineConfig] = None,
                 diarization_config: Optional[DiarizationConfig] = None,
                 formatting_config: Optional[FormattingConfig] = None,
                 optimization_config: Optional[OptimizationConfig] = None):
        
        # Initialize components
        self.transcriber = None
        self.pipeline = None
        self.diarizer = None
        self.formatter = None
        self.optimizer = None
        
        # Configuration
        self.transcription_config = transcription_config or TranscriptionConfig()
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.diarization_config = diarization_config or DiarizationConfig()
        self.formatting_config = formatting_config or FormattingConfig()
        self.optimization_config = optimization_config or OptimizationConfig()
        
        # System state
        self.is_initialized = False
        self.is_running = False
        
        # Statistics
        self.stats = {
            "total_transcriptions": 0,
            "total_audio_processed": 0.0,
            "total_processing_time": 0.0,
            "average_confidence": 0.0,
            "error_count": 0,
            "uptime": 0.0
        }
        
        # Callbacks
        self.transcription_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        logger.info("AITranscriptionSystem initialized")
        
    def initialize(self) -> bool:
        """Initialize the complete AI system"""
        try:
            logger.info("Initializing AI transcription system...")
            
            # Initialize performance optimizer first
            self.optimizer = PerformanceOptimizer(self.optimization_config)
            if not self.optimizer.initialize():
                logger.error("Failed to initialize performance optimizer")
                return False
                
            # Initialize transcriber
            self.transcriber = WhisperTranscriber(self.transcription_config)
            if not self.transcriber.initialize():
                logger.error("Failed to initialize transcriber")
                return False
                
            # Initialize pipeline
            self.pipeline = TranscriptionPipeline(self.pipeline_config)
            self.pipeline.transcriber = self.transcriber  # Share transcriber
            if not self.pipeline.initialize():
                logger.error("Failed to initialize pipeline")
                return False
                
            # Initialize diarizer
            self.diarizer = SpeakerDiarizer(self.diarization_config)
            if not self.diarizer.initialize():
                logger.error("Failed to initialize diarizer")
                return False
                
            # Initialize formatter
            self.formatter = TranscriptFormatter(self.formatting_config)
            if not self.formatter.initialize():
                logger.error("Failed to initialize formatter")
                return False
                
            # Set up component integration
            self._setup_component_integration()
            
            self.is_initialized = True
            self.is_running = True
            
            logger.info("AI transcription system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AI system: {e}")
            return False
            
    def _setup_component_integration(self):
        """Set up integration between components"""
        # Add result callbacks for pipeline
        self.pipeline.add_result_callback(self._handle_pipeline_result)
        
        # Add optimization callbacks
        self.optimizer.add_optimization_callback(self._handle_optimization_result)
        
        # Add transcription callbacks
        self.transcriber.add_result_callback(self._handle_transcription_result)
        
    def _handle_pipeline_result(self, result: PipelineResult):
        """Handle pipeline results"""
        try:
            # Optimize system for current workload
            self.optimizer.optimize_for_workload("transcription", {
                "processing_time": result.processing_time,
                "chunk_count": result.chunk_count,
                "real_time_factor": result.real_time_factor
            })
            
            # Update system statistics
            self.stats["total_transcriptions"] += 1
            self.stats["total_audio_processed"] += result.duration
            self.stats["total_processing_time"] += result.processing_time
            
            # Call user callbacks
            for callback in self.transcription_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in transcription callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling pipeline result: {e}")
            
    def _handle_optimization_result(self, result: OptimizationResult):
        """Handle optimization results"""
        try:
            if not result.success:
                logger.warning(f"Optimization failed: {result.actions_taken}")
                
        except Exception as e:
            logger.error(f"Error handling optimization result: {e}")
            
    def _handle_transcription_result(self, result: TranscriptionResult):
        """Handle individual transcription results"""
        try:
            # Update confidence statistics
            if result.success:
                count = self.stats["total_transcriptions"]
                old_avg = self.stats["average_confidence"]
                self.stats["average_confidence"] = (
                    old_avg * (count - 1) + result.confidence
                ) / count if count > 0 else result.confidence
            else:
                self.stats["error_count"] += 1
                
        except Exception as e:
            logger.error(f"Error handling transcription result: {e}")
            
    def transcribe_audio(self, audio_data, start_time: float = 0.0) -> Optional[PipelineResult]:
        """Transcribe audio with full pipeline"""
        if not self.is_initialized:
            logger.error("AI system not initialized")
            return None
            
        try:
            # Process through pipeline
            if hasattr(audio_data, 'shape'):  # numpy array
                self.pipeline.process_audio(audio_data)
                # For real-time, results come through callbacks
                return None
            else:  # file path
                return self.pipeline.process_audio_file(audio_data)
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None
            
    def transcribe_with_diarization(self, audio_data, start_time: float = 0.0) -> Optional[Dict[str, Any]]:
        """Transcribe audio with speaker diarization"""
        if not self.is_initialized:
            logger.error("AI system not initialized")
            return None
            
        try:
            # Get transcription result
            transcription_result = self.transcribe_audio(audio_data, start_time)
            if not transcription_result:
                return None
                
            # Perform diarization
            import numpy as np
            if isinstance(audio_data, str):
                import soundfile as sf
                audio_array, _ = sf.read(audio_data)
            else:
                audio_array = audio_data
                
            diarization_result = self.diarizer.diarize_segments(audio_array, start_time)
            
            # Combine results
            return {
                "transcription": transcription_result.to_dict(),
                "diarization": diarization_result.to_dict(),
                "combined_segments": self._combine_transcription_and_diarization(
                    transcription_result, diarization_result
                )
            }
            
        except Exception as e:
            logger.error(f"Transcription with diarization failed: {e}")
            return None
            
    def _combine_transcription_and_diarization(self, transcription_result: PipelineResult, 
                                             diarization_result: DiarizationResult) -> List[Dict[str, Any]]:
        """Combine transcription and diarization results"""
        combined_segments = []
        
        try:
            # Match transcription chunks with diarization segments
            for chunk_result in transcription_result.chunk_results:
                # Find matching diarization segment
                matching_segment = None
                for diar_segment in diarization_result.segments:
                    if (chunk_result.start_time >= diar_segment.start_time and
                        chunk_result.end_time <= diar_segment.end_time):
                        matching_segment = diar_segment
                        break
                        
                # Create combined segment
                combined_segment = {
                    "start_time": chunk_result.start_time,
                    "end_time": chunk_result.end_time,
                    "duration": chunk_result.duration,
                    "text": chunk_result.text,
                    "transcription_confidence": chunk_result.confidence,
                    "speaker_id": matching_segment.speaker_id if matching_segment else 0,
                    "speaker_label": matching_segment.speaker_label.value if matching_segment else "Unknown",
                    "speaker_confidence": matching_segment.confidence if matching_segment else 0.0
                }
                
                combined_segments.append(combined_segment)
                
        except Exception as e:
            logger.error(f"Error combining results: {e}")
            
        return combined_segments
        
    def format_transcript(self, transcription_result: Union[PipelineResult, Dict[str, Any]], 
                         output_format: OutputFormat = OutputFormat.TEXT) -> Optional[FormattedTranscript]:
        """Format transcription result"""
        if not self.is_initialized:
            logger.error("AI system not initialized")
            return None
            
        try:
            # Convert to transcript segments
            segments = []
            
            if isinstance(transcription_result, PipelineResult):
                for chunk_result in transcription_result.chunk_results:
                    segment = TranscriptSegment(
                        text=chunk_result.text,
                        speaker_id=1,  # Default speaker
                        speaker_label="Speaker 1",
                        start_time=chunk_result.start_time,
                        end_time=chunk_result.end_time,
                        duration=chunk_result.duration,
                        confidence=chunk_result.confidence,
                        voice_activity=1.0
                    )
                    segments.append(segment)
            else:
                # Handle combined results with diarization
                for combined_segment in transcription_result.get("combined_segments", []):
                    segment = TranscriptSegment(
                        text=combined_segment.get("text", ""),
                        speaker_id=combined_segment.get("speaker_id", 1),
                        speaker_label=combined_segment.get("speaker_label", "Speaker 1"),
                        start_time=combined_segment.get("start_time", 0.0),
                        end_time=combined_segment.get("end_time", 0.0),
                        duration=combined_segment.get("duration", 0.0),
                        confidence=combined_segment.get("transcription_confidence", 0.0),
                        voice_activity=1.0
                    )
                    segments.append(segment)
                    
            # Update formatter config if needed
            if self.formatter.config.output_format != output_format:
                self.formatter.config.output_format = output_format
                
            # Format transcript
            return self.formatter.format_transcript(segments)
            
        except Exception as e:
            logger.error(f"Formatting failed: {e}")
            return None
            
    def export_transcript(self, formatted_transcript: FormattedTranscript, 
                        file_path: str) -> bool:
        """Export formatted transcript to file"""
        if not self.is_initialized:
            logger.error("AI system not initialized")
            return False
            
        try:
            return self.formatter.export_to_file(formatted_transcript, file_path)
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
            
    def add_transcription_callback(self, callback: Callable):
        """Add callback for transcription results"""
        self.transcription_callbacks.append(callback)
        
    def add_error_callback(self, callback: Callable):
        """Add callback for error handling"""
        self.error_callbacks.append(callback)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "ai_system": self.stats,
            "transcriber": self.transcriber.get_stats() if self.transcriber else {},
            "pipeline": self.pipeline.get_stats() if self.pipeline else {},
            "diarizer": self.diarizer.get_stats() if self.diarizer else {},
            "formatter": self.formatter.get_stats() if self.formatter else {},
            "optimizer": self.optimizer.get_stats() if self.optimizer else {}
        }
        
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "components": {
                "transcriber": self.transcriber.get_model_info() if self.transcriber else None,
                "pipeline": self.pipeline.get_status() if self.pipeline else None,
                "diarizer": self.diarizer.get_status() if self.diarizer else None,
                "formatter": self.formatter.get_status() if self.formatter else None,
                "optimizer": self.optimizer.get_status() if self.optimizer else None
            },
            "stats": self.get_stats()
        }
        
    def shutdown(self):
        """Shutdown the AI system"""
        logger.info("Shutting down AI transcription system...")
        
        self.is_running = False
        
        # Shutdown components in reverse order
        if self.formatter:
            self.formatter.shutdown()
            
        if self.diarizer:
            self.diarizer.shutdown()
            
        if self.pipeline:
            self.pipeline.shutdown()
            
        if self.transcriber:
            self.transcriber.shutdown()
            
        if self.optimizer:
            self.optimizer.shutdown()
            
        logger.info("AI transcription system shutdown complete")
        
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


# Factory functions for common configurations
def create_meeting_ai_system(max_speakers: int = 6, 
                           output_format: OutputFormat = OutputFormat.TEXT) -> AITranscriptionSystem:
    """Create AI system optimized for meetings"""
    
    # Transcription config
    transcription_config = TranscriptionConfig(
        model_size=ModelSize.BASE,
        word_timestamps=True,
        min_confidence=0.6
    )
    
    # Pipeline config
    pipeline_config = PipelineConfig(
        processing_mode=ProcessingMode.REAL_TIME,
        quality_settings=QualitySettings.BALANCED,
        chunk_size=10.0,
        max_concurrent_processing=2
    )
    
    # Diarization config
    diarization_config = DiarizationConfig(
        max_speakers=max_speakers,
        min_segment_duration=0.5,
        speaker_similarity_threshold=0.8,
        merge_consecutive_segments=True
    )
    
    # Formatting config
    formatting_config = FormattingConfig(
        output_format=output_format,
        include_speakers=True,
        include_timestamps=True,
        line_break_on_speaker_change=True
    )
    
    # Optimization config
    optimization_config = OptimizationConfig(
        optimization_level=OptimizationLevel.BALANCED,
        performance_mode=PerformanceMode.REAL_TIME,
        enable_cpu_scaling=True,
        enable_memory_optimization=True
    )
    
    return AITranscriptionSystem(
        transcription_config=transcription_config,
        pipeline_config=pipeline_config,
        diarization_config=diarization_config,
        formatting_config=formatting_config,
        optimization_config=optimization_config
    )


def create_interview_ai_system(output_format: OutputFormat = OutputFormat.TEXT) -> AITranscriptionSystem:
    """Create AI system optimized for interviews"""
    
    # Transcription config
    transcription_config = TranscriptionConfig(
        model_size=ModelSize.BASE,
        word_timestamps=True,
        min_confidence=0.7
    )
    
    # Pipeline config
    pipeline_config = PipelineConfig(
        processing_mode=ProcessingMode.REAL_TIME,
        quality_settings=QualitySettings.QUALITY,
        chunk_size=15.0,
        max_concurrent_processing=1
    )
    
    # Diarization config
    diarization_config = DiarizationConfig(
        max_speakers=3,
        min_segment_duration=1.0,
        speaker_similarity_threshold=0.85,
        merge_consecutive_segments=True
    )
    
    # Formatting config
    formatting_config = FormattingConfig(
        output_format=output_format,
        include_speakers=True,
        include_timestamps=True,
        line_break_on_speaker_change=True,
        max_line_length=100
    )
    
    # Optimization config
    optimization_config = OptimizationConfig(
        optimization_level=OptimizationLevel.BALANCED,
        performance_mode=PerformanceMode.REAL_TIME,
        enable_cpu_scaling=True,
        enable_memory_optimization=True
    )
    
    return AITranscriptionSystem(
        transcription_config=transcription_config,
        pipeline_config=pipeline_config,
        diarization_config=diarization_config,
        formatting_config=formatting_config,
        optimization_config=optimization_config
    )


def create_lecture_ai_system(output_format: OutputFormat = OutputFormat.TEXT) -> AITranscriptionSystem:
    """Create AI system optimized for lectures"""
    
    # Transcription config
    transcription_config = TranscriptionConfig(
        model_size=ModelSize.BASE,
        word_timestamps=True,
        min_confidence=0.6
    )
    
    # Pipeline config
    pipeline_config = PipelineConfig(
        processing_mode=ProcessingMode.BATCH,
        quality_settings=QualitySettings.QUALITY,
        chunk_size=30.0,
        max_concurrent_processing=1
    )
    
    # Diarization config
    diarization_config = DiarizationConfig(
        max_speakers=2,
        min_segment_duration=2.0,
        speaker_similarity_threshold=0.9,
        merge_consecutive_segments=True
    )
    
    # Formatting config
    formatting_config = FormattingConfig(
        output_format=output_format,
        include_speakers=True,
        include_timestamps=True,
        line_break_on_speaker_change=True,
        max_line_length=120
    )
    
    # Optimization config
    optimization_config = OptimizationConfig(
        optimization_level=OptimizationLevel.CONSERVATIVE,
        performance_mode=PerformanceMode.BATCH,
        enable_cpu_scaling=True,
        enable_memory_optimization=True
    )
    
    return AITranscriptionSystem(
        transcription_config=transcription_config,
        pipeline_config=pipeline_config,
        diarization_config=diarization_config,
        formatting_config=formatting_config,
        optimization_config=optimization_config
    )


# Export all public components
__all__ = [
    # Main system
    "AITranscriptionSystem",
    
    # Core components
    "WhisperTranscriber",
    "TranscriptionPipeline", 
    "AIAudioChunker",
    "SpeakerDiarizer",
    "TranscriptFormatter",
    "PerformanceOptimizer",
    
    # Configuration classes
    "TranscriptionConfig",
    "PipelineConfig", 
    "TranscriptionChunkConfig",
    "DiarizationConfig",
    "FormattingConfig",
    "OptimizationConfig",
    
    # Result classes
    "TranscriptionResult",
    "PipelineResult",
    "OptimalChunk",
    "DiarizationResult",
    "FormattedTranscript",
    "OptimizationResult",
    
    # Enums
    "ModelSize",
    "ProcessingMode",
    "QualitySettings",
    "ChunkType",
    "ChunkQuality",
    "SpeakerLabel",
    "OutputFormat",
    "TimestampFormat",
    "OptimizationLevel",
    "PerformanceMode",
    
    # Factory functions
    "create_meeting_ai_system",
    "create_interview_ai_system",
    "create_lecture_ai_system",
    "create_base_transcriber",
    "create_optimized_transcriber",
    "create_realtime_pipeline",
    "create_batch_pipeline",
    "create_default_chunker",
    "create_realtime_chunker",
    "create_quality_chunker",
    "create_meeting_diarizer",
    "create_interview_diarizer",
    "create_text_formatter",
    "create_subtitle_formatter",
    "create_meeting_formatter",
    "create_realtime_optimizer",
    "create_power_efficient_optimizer",
    "create_balanced_optimizer",
    
    # Version info
    "__version__",
    "__author__",
    "__description__"
]


# Initialize module
logger.info(f"AI module {__version__} loaded successfully")
logger.info(f"Available components: {len(__all__)} exports")