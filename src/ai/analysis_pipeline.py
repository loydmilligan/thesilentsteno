#!/usr/bin/env python3
"""
AI Analysis Pipeline Module

Main pipeline orchestrator that manages the entire AI analysis workflow from
transcript to final output with error handling and status tracking.

Author: Claude AI Assistant
Date: 2025-07-15
Version: 1.0
"""

import os
import sys
import logging
import time
import json
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Pipeline processing stages"""
    INITIALIZATION = "initialization"
    TRANSCRIPT_PREPARATION = "transcript_preparation"
    WHISPER_TRANSCRIPTION = "whisper_transcription"
    SPEAKER_DIARIZATION = "speaker_diarization"
    LLM_ANALYSIS = "llm_analysis"
    PARTICIPANT_ANALYSIS = "participant_analysis"
    CONFIDENCE_SCORING = "confidence_scoring"
    OUTPUT_FORMATTING = "output_formatting"
    FINALIZATION = "finalization"
    COMPLETED = "completed"
    ERROR = "error"


class PipelineStatus(Enum):
    """Overall pipeline status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineConfig:
    """Configuration for the AI analysis pipeline"""
    
    # Processing settings
    enable_whisper_transcription: bool = True
    enable_llm_analysis: bool = True
    enable_participant_analysis: bool = True
    enable_confidence_scoring: bool = True
    
    # Performance settings
    max_workers: int = 4
    processing_timeout: int = 600  # 10 minutes
    chunk_size: int = 30  # seconds
    
    # Quality settings
    min_confidence_threshold: float = 0.7
    require_speaker_diarization: bool = True
    
    # Output settings
    output_formats: List[str] = field(default_factory=lambda: ["json", "markdown"])
    save_intermediate_results: bool = True
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 2.0
    continue_on_error: bool = True
    
    # Integration settings
    whisper_config: Optional[Dict[str, Any]] = None
    llm_config: Optional[Dict[str, Any]] = None
    participant_config: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "enable_whisper_transcription": self.enable_whisper_transcription,
            "enable_llm_analysis": self.enable_llm_analysis,
            "enable_participant_analysis": self.enable_participant_analysis,
            "enable_confidence_scoring": self.enable_confidence_scoring,
            "max_workers": self.max_workers,
            "processing_timeout": self.processing_timeout,
            "chunk_size": self.chunk_size,
            "min_confidence_threshold": self.min_confidence_threshold,
            "require_speaker_diarization": self.require_speaker_diarization,
            "output_formats": self.output_formats,
            "save_intermediate_results": self.save_intermediate_results,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "continue_on_error": self.continue_on_error,
            "whisper_config": self.whisper_config,
            "llm_config": self.llm_config,
            "participant_config": self.participant_config
        }


@dataclass
class PipelineResult:
    """Result from pipeline processing"""
    
    # Pipeline metadata
    pipeline_id: str
    session_id: str
    start_time: datetime
    end_time: datetime
    processing_time: float
    
    # Processing results
    transcript: str = ""
    speaker_diarization: Dict[str, Any] = field(default_factory=dict)
    llm_analysis: Dict[str, Any] = field(default_factory=dict)
    participant_analysis: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    # Output formats
    formatted_outputs: Dict[str, str] = field(default_factory=dict)
    
    # Status and quality
    status: PipelineStatus = PipelineStatus.PENDING
    overall_confidence: float = 0.0
    quality_score: float = 0.0
    
    # Processing metadata
    stages_completed: List[ProcessingStage] = field(default_factory=list)
    stage_durations: Dict[str, float] = field(default_factory=dict)
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Statistics
    audio_duration: float = 0.0
    transcript_length: int = 0
    speaker_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "pipeline_id": self.pipeline_id,
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "processing_time": self.processing_time,
            "transcript": self.transcript,
            "speaker_diarization": self.speaker_diarization,
            "llm_analysis": self.llm_analysis,
            "participant_analysis": self.participant_analysis,
            "confidence_scores": self.confidence_scores,
            "formatted_outputs": self.formatted_outputs,
            "status": self.status.value,
            "overall_confidence": self.overall_confidence,
            "quality_score": self.quality_score,
            "stages_completed": [stage.value for stage in self.stages_completed],
            "stage_durations": self.stage_durations,
            "errors": self.errors,
            "warnings": self.warnings,
            "audio_duration": self.audio_duration,
            "transcript_length": self.transcript_length,
            "speaker_count": self.speaker_count
        }


class AnalysisPipeline:
    """Main AI analysis pipeline orchestrator"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.pipeline_id = str(uuid.uuid4())
        
        # Component references
        self.whisper_system = None
        self.llm_system = None
        self.participant_analyzer = None
        self.confidence_scorer = None
        self.status_tracker = None
        
        # Processing state
        self.is_initialized = False
        self.current_stage = ProcessingStage.INITIALIZATION
        self.executor = None
        
        # Statistics
        self.stats = {
            "total_pipelines": 0,
            "successful_pipelines": 0,
            "failed_pipelines": 0,
            "average_processing_time": 0.0,
            "average_confidence": 0.0,
            "error_count": 0
        }
        
        # Callbacks
        self.stage_callbacks: Dict[ProcessingStage, List[Callable]] = {
            stage: [] for stage in ProcessingStage
        }
        self.error_callbacks: List[Callable] = []
        self.completion_callbacks: List[Callable] = []
        
        logger.info(f"AnalysisPipeline initialized with ID: {self.pipeline_id}")
        
    def initialize(self) -> bool:
        """Initialize the pipeline and all components"""
        try:
            logger.info("Initializing AI analysis pipeline...")
            
            # Initialize thread pool
            self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
            
            # Initialize AI components
            if self.config.enable_whisper_transcription:
                self._initialize_whisper_system()
                
            if self.config.enable_llm_analysis:
                self._initialize_llm_system()
                
            if self.config.enable_participant_analysis:
                self._initialize_participant_analyzer()
                
            if self.config.enable_confidence_scoring:
                self._initialize_confidence_scorer()
                
            # Initialize status tracker
            self._initialize_status_tracker()
            
            self.is_initialized = True
            logger.info("AI analysis pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return False
            
    def _initialize_whisper_system(self):
        """Initialize Whisper transcription system"""
        try:
            from .whisper_transcriber import WhisperTranscriber
            from .transcription_pipeline import TranscriptionPipeline
            
            config = self.config.whisper_config or {}
            self.whisper_system = TranscriptionPipeline(config)
            
            if not self.whisper_system.initialize():
                raise Exception("Failed to initialize Whisper system")
                
        except Exception as e:
            logger.error(f"Failed to initialize Whisper system: {e}")
            if not self.config.continue_on_error:
                raise
                
    def _initialize_llm_system(self):
        """Initialize LLM analysis system"""
        try:
            from .local_llm_processor import LLMAnalysisSystem
            
            config = self.config.llm_config or {}
            self.llm_system = LLMAnalysisSystem(**config)
            
            if not self.llm_system.initialize():
                raise Exception("Failed to initialize LLM system")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM system: {e}")
            if not self.config.continue_on_error:
                raise
                
    def _initialize_participant_analyzer(self):
        """Initialize participant analyzer"""
        try:
            from .participant_analyzer import ParticipantAnalyzer
            
            config = self.config.participant_config or {}
            self.participant_analyzer = ParticipantAnalyzer(config)
            
        except Exception as e:
            logger.error(f"Failed to initialize participant analyzer: {e}")
            if not self.config.continue_on_error:
                raise
                
    def _initialize_confidence_scorer(self):
        """Initialize confidence scorer"""
        try:
            from .confidence_scorer import ConfidenceScorer
            
            self.confidence_scorer = ConfidenceScorer()
            
        except Exception as e:
            logger.error(f"Failed to initialize confidence scorer: {e}")
            if not self.config.continue_on_error:
                raise
                
    def _initialize_status_tracker(self):
        """Initialize status tracker"""
        try:
            from .status_tracker import StatusTracker
            
            self.status_tracker = StatusTracker()
            
        except Exception as e:
            logger.error(f"Failed to initialize status tracker: {e}")
            if not self.config.continue_on_error:
                raise
                
    def process_session(self, session_id: str, audio_data: bytes, 
                       metadata: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """Process a complete session through the AI pipeline"""
        if not self.is_initialized:
            raise Exception("Pipeline not initialized")
            
        pipeline_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        result = PipelineResult(
            pipeline_id=pipeline_id,
            session_id=session_id,
            start_time=start_time,
            end_time=start_time,  # Will be updated
            processing_time=0.0
        )
        
        try:
            logger.info(f"Starting pipeline processing for session: {session_id}")
            
            # Update stats
            self.stats["total_pipelines"] += 1
            
            # Stage 1: Transcript Preparation
            result = self._execute_stage(ProcessingStage.TRANSCRIPT_PREPARATION, result, 
                                       self._prepare_transcript, audio_data, metadata)
            
            # Stage 2: Whisper Transcription
            if self.config.enable_whisper_transcription and self.whisper_system:
                result = self._execute_stage(ProcessingStage.WHISPER_TRANSCRIPTION, result,
                                           self._transcribe_audio, audio_data, metadata)
                
            # Stage 3: Speaker Diarization
            if self.config.require_speaker_diarization:
                result = self._execute_stage(ProcessingStage.SPEAKER_DIARIZATION, result,
                                           self._diarize_speakers, audio_data, result.transcript)
                
            # Stage 4: LLM Analysis
            if self.config.enable_llm_analysis and self.llm_system:
                result = self._execute_stage(ProcessingStage.LLM_ANALYSIS, result,
                                           self._analyze_with_llm, result.transcript, metadata)
                
            # Stage 5: Participant Analysis
            if self.config.enable_participant_analysis and self.participant_analyzer:
                result = self._execute_stage(ProcessingStage.PARTICIPANT_ANALYSIS, result,
                                           self._analyze_participants, result.transcript, 
                                           result.speaker_diarization)
                
            # Stage 6: Confidence Scoring
            if self.config.enable_confidence_scoring and self.confidence_scorer:
                result = self._execute_stage(ProcessingStage.CONFIDENCE_SCORING, result,
                                           self._score_confidence, result)
                
            # Stage 7: Output Formatting
            result = self._execute_stage(ProcessingStage.OUTPUT_FORMATTING, result,
                                       self._format_outputs, result)
            
            # Stage 8: Finalization
            result = self._execute_stage(ProcessingStage.FINALIZATION, result,
                                       self._finalize_processing, result)
            
            # Complete processing
            result.end_time = datetime.now()
            result.processing_time = (result.end_time - result.start_time).total_seconds()
            result.status = PipelineStatus.COMPLETED
            
            # Update statistics
            self.stats["successful_pipelines"] += 1
            self._update_stats(result)
            
            # Call completion callbacks
            for callback in self.completion_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in completion callback: {e}")
                    
            logger.info(f"Pipeline processing completed for session: {session_id}")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline processing failed for session {session_id}: {e}")
            
            result.end_time = datetime.now()
            result.processing_time = (result.end_time - result.start_time).total_seconds()
            result.status = PipelineStatus.FAILED
            result.errors.append(str(e))
            
            # Update statistics
            self.stats["failed_pipelines"] += 1
            self.stats["error_count"] += 1
            
            # Call error callbacks
            for callback in self.error_callbacks:
                try:
                    callback(result, e)
                except Exception as callback_error:
                    logger.error(f"Error in error callback: {callback_error}")
                    
            return result
            
    def _execute_stage(self, stage: ProcessingStage, result: PipelineResult, 
                      stage_func: Callable, *args, **kwargs) -> PipelineResult:
        """Execute a pipeline stage with error handling and timing"""
        stage_start = time.time()
        
        try:
            logger.info(f"Executing stage: {stage.value}")
            self.current_stage = stage
            
            # Call stage callbacks
            for callback in self.stage_callbacks[stage]:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in stage callback: {e}")
                    
            # Execute stage
            stage_result = stage_func(*args, **kwargs)
            
            # Update result with stage output
            if stage_result is not None:
                if stage == ProcessingStage.WHISPER_TRANSCRIPTION:
                    result.transcript = stage_result.get("transcript", "")
                    result.transcript_length = len(result.transcript)
                elif stage == ProcessingStage.SPEAKER_DIARIZATION:
                    result.speaker_diarization = stage_result
                    result.speaker_count = len(stage_result.get("speakers", []))
                elif stage == ProcessingStage.LLM_ANALYSIS:
                    result.llm_analysis = stage_result
                elif stage == ProcessingStage.PARTICIPANT_ANALYSIS:
                    result.participant_analysis = stage_result
                elif stage == ProcessingStage.CONFIDENCE_SCORING:
                    result.confidence_scores = stage_result.get("scores", {})
                    result.overall_confidence = stage_result.get("overall", 0.0)
                elif stage == ProcessingStage.OUTPUT_FORMATTING:
                    result.formatted_outputs = stage_result
                    
            # Record stage completion
            stage_duration = time.time() - stage_start
            result.stages_completed.append(stage)
            result.stage_durations[stage.value] = stage_duration
            
            logger.info(f"Stage {stage.value} completed in {stage_duration:.2f}s")
            
        except Exception as e:
            stage_duration = time.time() - stage_start
            result.stage_durations[stage.value] = stage_duration
            
            error_msg = f"Stage {stage.value} failed: {str(e)}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            
            if not self.config.continue_on_error:
                raise
                
        return result
        
    def _prepare_transcript(self, audio_data: bytes, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare transcript for processing"""
        try:
            # Basic audio validation
            if not audio_data:
                raise ValueError("No audio data provided")
                
            # Extract metadata
            duration = metadata.get("duration", 0) if metadata else 0
            
            return {
                "audio_size": len(audio_data),
                "duration": duration,
                "format": metadata.get("format", "unknown") if metadata else "unknown"
            }
            
        except Exception as e:
            logger.error(f"Transcript preparation failed: {e}")
            raise
            
    def _transcribe_audio(self, audio_data: bytes, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Transcribe audio using Whisper"""
        try:
            if not self.whisper_system:
                raise Exception("Whisper system not initialized")
                
            # Process audio through Whisper
            result = self.whisper_system.transcribe(audio_data)
            
            if not result.success:
                raise Exception(f"Whisper transcription failed: {result.error_message}")
                
            return {
                "transcript": result.text,
                "confidence": result.confidence,
                "language": result.language,
                "segments": result.segments
            }
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            raise
            
    def _diarize_speakers(self, audio_data: bytes, transcript: str) -> Dict[str, Any]:
        """Perform speaker diarization"""
        try:
            # Placeholder for speaker diarization
            # In full implementation, this would use existing speaker detection
            return {
                "speakers": ["Speaker 1", "Speaker 2"],
                "segments": [
                    {"speaker": "Speaker 1", "start": 0, "end": 30, "text": transcript[:len(transcript)//2]},
                    {"speaker": "Speaker 2", "start": 30, "end": 60, "text": transcript[len(transcript)//2:]}
                ]
            }
            
        except Exception as e:
            logger.error(f"Speaker diarization failed: {e}")
            raise
            
    def _analyze_with_llm(self, transcript: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze transcript using LLM"""
        try:
            if not self.llm_system:
                raise Exception("LLM system not initialized")
                
            # Perform comprehensive analysis
            result = self.llm_system.complete_analysis_workflow(transcript)
            
            if not result["success"]:
                raise Exception(f"LLM analysis failed: {result.get('error', 'Unknown error')}")
                
            return result
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            raise
            
    def _analyze_participants(self, transcript: str, speaker_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze participant contributions"""
        try:
            if not self.participant_analyzer:
                raise Exception("Participant analyzer not initialized")
                
            # Analyze participants
            result = self.participant_analyzer.analyze_participants(transcript, speaker_data)
            
            return result.to_dict() if hasattr(result, 'to_dict') else result
            
        except Exception as e:
            logger.error(f"Participant analysis failed: {e}")
            raise
            
    def _score_confidence(self, pipeline_result: PipelineResult) -> Dict[str, Any]:
        """Score confidence of pipeline outputs"""
        try:
            if not self.confidence_scorer:
                raise Exception("Confidence scorer not initialized")
                
            # Score confidence
            result = self.confidence_scorer.score_pipeline_result(pipeline_result)
            
            return result.to_dict() if hasattr(result, 'to_dict') else result
            
        except Exception as e:
            logger.error(f"Confidence scoring failed: {e}")
            raise
            
    def _format_outputs(self, pipeline_result: PipelineResult) -> Dict[str, str]:
        """Format outputs in specified formats"""
        try:
            formatted = {}
            
            # Create comprehensive output data
            output_data = {
                "session_id": pipeline_result.session_id,
                "transcript": pipeline_result.transcript,
                "speaker_diarization": pipeline_result.speaker_diarization,
                "llm_analysis": pipeline_result.llm_analysis,
                "participant_analysis": pipeline_result.participant_analysis,
                "confidence_scores": pipeline_result.confidence_scores,
                "metadata": {
                    "processing_time": pipeline_result.processing_time,
                    "audio_duration": pipeline_result.audio_duration,
                    "transcript_length": pipeline_result.transcript_length,
                    "speaker_count": pipeline_result.speaker_count,
                    "overall_confidence": pipeline_result.overall_confidence
                }
            }
            
            # Format in requested formats
            for format_type in self.config.output_formats:
                if format_type == "json":
                    formatted["json"] = json.dumps(output_data, indent=2, ensure_ascii=False)
                elif format_type == "markdown":
                    formatted["markdown"] = self._format_markdown(output_data)
                elif format_type == "html":
                    formatted["html"] = self._format_html(output_data)
                else:
                    logger.warning(f"Unknown output format: {format_type}")
                    
            return formatted
            
        except Exception as e:
            logger.error(f"Output formatting failed: {e}")
            raise
            
    def _format_markdown(self, data: Dict[str, Any]) -> str:
        """Format output as Markdown"""
        md = f"# Meeting Analysis Report\n\n"
        md += f"**Session ID:** {data['session_id']}\n"
        md += f"**Processing Time:** {data['metadata']['processing_time']:.2f}s\n"
        md += f"**Audio Duration:** {data['metadata']['audio_duration']:.2f}s\n\n"
        
        if data['transcript']:
            md += f"## Transcript\n\n{data['transcript']}\n\n"
            
        if data['llm_analysis']:
            md += f"## Analysis Summary\n\n{data['llm_analysis']}\n\n"
            
        return md
        
    def _format_html(self, data: Dict[str, Any]) -> str:
        """Format output as HTML"""
        html = f"""
        <html>
        <head><title>Meeting Analysis Report</title></head>
        <body>
        <h1>Meeting Analysis Report</h1>
        <p><strong>Session ID:</strong> {data['session_id']}</p>
        <p><strong>Processing Time:</strong> {data['metadata']['processing_time']:.2f}s</p>
        <p><strong>Audio Duration:</strong> {data['metadata']['audio_duration']:.2f}s</p>
        """
        
        if data['transcript']:
            html += f"<h2>Transcript</h2><p>{data['transcript']}</p>"
            
        if data['llm_analysis']:
            html += f"<h2>Analysis Summary</h2><p>{data['llm_analysis']}</p>"
            
        html += "</body></html>"
        return html
        
    def _finalize_processing(self, pipeline_result: PipelineResult) -> Dict[str, Any]:
        """Finalize pipeline processing"""
        try:
            # Calculate quality score
            quality_factors = []
            
            if pipeline_result.transcript_length > 0:
                quality_factors.append(min(1.0, pipeline_result.transcript_length / 1000))
                
            if pipeline_result.speaker_count > 0:
                quality_factors.append(min(1.0, pipeline_result.speaker_count / 5))
                
            if pipeline_result.overall_confidence > 0:
                quality_factors.append(pipeline_result.overall_confidence)
                
            pipeline_result.quality_score = sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
            
            # Save intermediate results if configured
            if self.config.save_intermediate_results:
                self._save_intermediate_results(pipeline_result)
                
            return {"quality_score": pipeline_result.quality_score}
            
        except Exception as e:
            logger.error(f"Finalization failed: {e}")
            raise
            
    def _save_intermediate_results(self, pipeline_result: PipelineResult):
        """Save intermediate processing results"""
        try:
            # This would save results to appropriate storage
            logger.info(f"Saving intermediate results for session: {pipeline_result.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}")
            
    def _update_stats(self, result: PipelineResult):
        """Update pipeline statistics"""
        try:
            # Update average processing time
            total = self.stats["total_pipelines"]
            old_avg = self.stats["average_processing_time"]
            self.stats["average_processing_time"] = (
                old_avg * (total - 1) + result.processing_time
            ) / total
            
            # Update average confidence
            if result.overall_confidence > 0:
                old_conf_avg = self.stats["average_confidence"]
                self.stats["average_confidence"] = (
                    old_conf_avg * (total - 1) + result.overall_confidence
                ) / total
                
        except Exception as e:
            logger.error(f"Failed to update stats: {e}")
            
    def add_stage_callback(self, stage: ProcessingStage, callback: Callable):
        """Add callback for specific stage"""
        self.stage_callbacks[stage].append(callback)
        
    def add_error_callback(self, callback: Callable):
        """Add error callback"""
        self.error_callbacks.append(callback)
        
    def add_completion_callback(self, callback: Callable):
        """Add completion callback"""
        self.completion_callbacks.append(callback)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return self.stats.copy()
        
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        return {
            "pipeline_id": self.pipeline_id,
            "is_initialized": self.is_initialized,
            "current_stage": self.current_stage.value,
            "config": self.config.to_dict(),
            "stats": self.get_stats()
        }
        
    def shutdown(self):
        """Shutdown pipeline and cleanup resources"""
        logger.info("Shutting down AI analysis pipeline...")
        
        if self.executor:
            self.executor.shutdown(wait=True)
            
        if self.whisper_system:
            self.whisper_system.shutdown()
            
        if self.llm_system:
            self.llm_system.shutdown()
            
        self.is_initialized = False
        logger.info("AI analysis pipeline shutdown complete")
        
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


# Factory functions
def create_basic_pipeline() -> AnalysisPipeline:
    """Create basic pipeline with default settings"""
    config = PipelineConfig(
        enable_whisper_transcription=True,
        enable_llm_analysis=True,
        enable_participant_analysis=False,
        enable_confidence_scoring=True,
        output_formats=["json"]
    )
    return AnalysisPipeline(config)


def create_full_pipeline() -> AnalysisPipeline:
    """Create full pipeline with all features enabled"""
    config = PipelineConfig(
        enable_whisper_transcription=True,
        enable_llm_analysis=True,
        enable_participant_analysis=True,
        enable_confidence_scoring=True,
        output_formats=["json", "markdown", "html"],
        save_intermediate_results=True
    )
    return AnalysisPipeline(config)


def create_fast_pipeline() -> AnalysisPipeline:
    """Create fast pipeline for quick processing"""
    config = PipelineConfig(
        enable_whisper_transcription=True,
        enable_llm_analysis=True,
        enable_participant_analysis=False,
        enable_confidence_scoring=False,
        output_formats=["json"],
        save_intermediate_results=False,
        max_workers=8
    )
    return AnalysisPipeline(config)


# Testing and demonstration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Analysis Pipeline Test")
    parser.add_argument("--audio-file", type=str, help="Audio file to process")
    parser.add_argument("--pipeline-type", type=str, default="basic",
                       choices=["basic", "full", "fast"], help="Pipeline type")
    args = parser.parse_args()
    
    # Create pipeline
    if args.pipeline_type == "full":
        pipeline = create_full_pipeline()
    elif args.pipeline_type == "fast":
        pipeline = create_fast_pipeline()
    else:
        pipeline = create_basic_pipeline()
        
    try:
        print(f"Pipeline status: {pipeline.get_status()}")
        
        # Initialize pipeline
        if not pipeline.initialize():
            print("Failed to initialize pipeline")
            sys.exit(1)
            
        # Process audio if provided
        if args.audio_file:
            try:
                with open(args.audio_file, 'rb') as f:
                    audio_data = f.read()
                    
                print(f"Processing audio file: {args.audio_file}")
                result = pipeline.process_session("test_session", audio_data)
                
                if result.status == PipelineStatus.COMPLETED:
                    print(f"Processing completed successfully!")
                    print(f"Processing time: {result.processing_time:.2f}s")
                    print(f"Transcript length: {result.transcript_length}")
                    print(f"Overall confidence: {result.overall_confidence:.3f}")
                    
                    if result.formatted_outputs:
                        print(f"Generated outputs: {list(result.formatted_outputs.keys())}")
                        
                else:
                    print(f"Processing failed: {result.errors}")
                    
            except Exception as e:
                print(f"Error processing audio: {e}")
        else:
            print("No audio file provided, pipeline ready for use")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pipeline.shutdown()