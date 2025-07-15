#!/usr/bin/env python3
"""
Transcription Pipeline Module

Real-time transcription processing pipeline that orchestrates audio chunking,
quality assessment, and Whisper transcription for optimal performance.

Author: Claude AI Assistant
Date: 2024-07-14
Version: 1.0
"""

import os
import sys
import logging
import threading
import time
import queue
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
import asyncio
from collections import deque

try:
    import numpy as np
    import soundfile as sf
except ImportError as e:
    print(f"Warning: Required dependencies not installed: {e}")
    print("Please install with: pip install numpy soundfile")
    sys.exit(1)

# Local imports
from .whisper_transcriber import WhisperTranscriber, TranscriptionResult, TranscriptionConfig
from .audio_chunker import AIAudioChunker, OptimalChunk, TranscriptionChunkConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing modes for transcription pipeline"""
    REAL_TIME = "real_time"      # Real-time streaming
    BATCH = "batch"              # Batch processing
    STREAMING = "streaming"      # Streaming with buffering
    ON_DEMAND = "on_demand"      # Process on request


class QualitySettings(Enum):
    """Quality vs speed tradeoffs"""
    SPEED = "speed"           # Prioritize speed
    BALANCED = "balanced"     # Balance speed and quality
    QUALITY = "quality"       # Prioritize quality
    ADAPTIVE = "adaptive"     # Adapt based on conditions


@dataclass
class PipelineConfig:
    """Configuration for transcription pipeline"""
    
    # Processing settings
    processing_mode: ProcessingMode = ProcessingMode.REAL_TIME
    quality_settings: QualitySettings = QualitySettings.BALANCED
    
    # Chunking settings
    chunk_size: float = 10.0  # Default chunk size in seconds
    chunk_overlap: float = 0.5  # Overlap between chunks
    min_chunk_size: float = 1.0  # Minimum chunk size
    max_chunk_size: float = 30.0  # Maximum chunk size
    
    # Processing settings
    max_concurrent_processing: int = 2  # Max concurrent transcriptions
    processing_timeout: float = 30.0  # Timeout for processing
    
    # Buffer settings
    input_buffer_size: int = 10  # Input buffer size
    output_buffer_size: int = 20  # Output buffer size
    
    # Quality control
    enable_quality_filtering: bool = True
    min_confidence_threshold: float = 0.5
    enable_silence_detection: bool = True
    
    # Performance settings
    enable_caching: bool = True
    cache_size: int = 50
    enable_preprocessing: bool = True
    
    # Integration settings
    enable_vad_integration: bool = True
    enable_speaker_integration: bool = True
    enable_statistics_collection: bool = True


@dataclass
class ChunkResult:
    """Result from processing a single chunk"""
    
    # Chunk information
    chunk_id: str
    chunk_index: int
    start_time: float
    end_time: float
    duration: float
    
    # Transcription result
    text: str
    confidence: float
    language: str
    
    # Processing metadata
    processing_time: float
    real_time_factor: float
    
    # Quality metrics
    silence_ratio: float = 0.0
    voice_activity: float = 0.0
    audio_quality: float = 0.0
    
    # Status
    success: bool = True
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
            "processing_time": self.processing_time,
            "real_time_factor": self.real_time_factor,
            "silence_ratio": self.silence_ratio,
            "voice_activity": self.voice_activity,
            "audio_quality": self.audio_quality,
            "success": self.success,
            "error_message": self.error_message
        }


@dataclass
class PipelineResult:
    """Result from pipeline processing"""
    
    # Overall information
    pipeline_id: str
    start_time: float
    end_time: float
    duration: float
    
    # Processing metadata
    processing_time: float
    real_time_factor: float
    
    # Chunk results
    chunk_results: List[ChunkResult] = field(default_factory=list)
    chunk_count: int = 0
    
    # Aggregated text
    full_text: str = ""
    
    # Quality metrics
    average_confidence: float = 0.0
    overall_quality: float = 0.0
    
    # Performance metrics
    throughput: float = 0.0  # chunks per second
    latency: float = 0.0  # average processing latency
    
    # Status
    success: bool = True
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "pipeline_id": self.pipeline_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "processing_time": self.processing_time,
            "real_time_factor": self.real_time_factor,
            "chunk_results": [chunk.to_dict() for chunk in self.chunk_results],
            "chunk_count": self.chunk_count,
            "full_text": self.full_text,
            "average_confidence": self.average_confidence,
            "overall_quality": self.overall_quality,
            "throughput": self.throughput,
            "latency": self.latency,
            "success": self.success,
            "error_message": self.error_message
        }


class TranscriptionPipeline:
    """Main transcription pipeline orchestrator"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.transcriber = None
        self.chunker = None
        self.is_initialized = False
        self.is_running = False
        
        # Processing queues
        self.input_queue = queue.Queue(maxsize=self.config.input_buffer_size)
        self.output_queue = queue.Queue(maxsize=self.config.output_buffer_size)
        
        # Thread management
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_processing)
        self.processing_futures: Dict[str, Future] = {}
        
        # Pipeline state
        self.active_pipelines: Dict[str, PipelineResult] = {}
        self.pipeline_lock = threading.Lock()
        
        # Performance tracking
        self.stats = {
            "total_pipelines": 0,
            "total_chunks_processed": 0,
            "total_audio_processed": 0.0,
            "total_processing_time": 0.0,
            "average_confidence": 0.0,
            "average_throughput": 0.0,
            "error_count": 0,
            "uptime": 0.0
        }
        
        # Caching
        self.cache = {} if self.config.enable_caching else None
        
        # Callbacks
        self.result_callbacks: List[Callable] = []
        self.chunk_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        logger.info(f"TranscriptionPipeline initialized with {self.config.processing_mode.value} mode")
        
    def initialize(self) -> bool:
        """Initialize the transcription pipeline"""
        try:
            logger.info("Initializing transcription pipeline...")
            
            # Initialize transcriber if not provided
            if not self.transcriber:
                transcriber_config = TranscriptionConfig(
                    word_timestamps=True,
                    min_confidence=self.config.min_confidence_threshold
                )
                self.transcriber = WhisperTranscriber(transcriber_config)
                
            if not self.transcriber.initialize():
                logger.error("Failed to initialize transcriber")
                return False
                
            # Initialize chunker if not provided
            if not self.chunker:
                chunker_config = TranscriptionChunkConfig(
                    chunk_duration=self.config.chunk_size,
                    overlap_duration=self.config.chunk_overlap,
                    min_chunk_duration=self.config.min_chunk_size,
                    max_chunk_duration=self.config.max_chunk_size
                )
                self.chunker = AIAudioChunker(chunker_config)
                
            if not self.chunker.initialize():
                logger.error("Failed to initialize chunker")
                return False
                
            self.is_initialized = True
            self.is_running = True
            
            logger.info("Transcription pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return False
            
    def process_audio(self, audio_data: np.ndarray, session_id: str = None) -> str:
        """Process audio data in real-time mode"""
        if not self.is_initialized:
            logger.error("Pipeline not initialized")
            return None
            
        try:
            # Generate session ID if not provided
            if session_id is None:
                session_id = str(uuid.uuid4())
                
            # Add to input queue
            self.input_queue.put({
                "session_id": session_id,
                "audio_data": audio_data,
                "timestamp": time.time()
            })
            
            # Process in background
            self._process_realtime_audio(session_id)
            
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to process audio: {e}")
            return None
            
    def process_audio_file(self, audio_file: str, start_time: float = 0.0) -> PipelineResult:
        """Process complete audio file"""
        if not self.is_initialized:
            logger.error("Pipeline not initialized")
            return PipelineResult(
                pipeline_id="",
                start_time=start_time,
                end_time=start_time,
                duration=0.0,
                processing_time=0.0,
                real_time_factor=0.0,
                success=False,
                error_message="Pipeline not initialized"
            )
            
        pipeline_id = str(uuid.uuid4())
        process_start = time.time()
        
        try:
            logger.info(f"Processing audio file: {audio_file}")
            
            # Load audio
            audio_data, sample_rate = sf.read(audio_file)
            audio_duration = len(audio_data) / sample_rate
            
            # Create chunks
            chunks = self.chunker.chunk_audio_for_transcription(
                audio_data, sample_rate, start_time
            )
            
            if not chunks:
                logger.warning("No chunks generated from audio")
                return PipelineResult(
                    pipeline_id=pipeline_id,
                    start_time=start_time,
                    end_time=start_time + audio_duration,
                    duration=audio_duration,
                    processing_time=time.time() - process_start,
                    real_time_factor=0.0,
                    success=False,
                    error_message="No chunks generated"
                )
                
            # Process chunks
            chunk_results = []
            chunk_futures = []
            
            for i, chunk in enumerate(chunks):
                future = self.executor.submit(
                    self._process_chunk,
                    chunk,
                    i,
                    pipeline_id
                )
                chunk_futures.append(future)
                
            # Collect results
            for future in chunk_futures:
                try:
                    chunk_result = future.result(timeout=self.config.processing_timeout)
                    if chunk_result:
                        chunk_results.append(chunk_result)
                        
                        # Call chunk callbacks
                        for callback in self.chunk_callbacks:
                            try:
                                callback(chunk_result)
                            except Exception as e:
                                logger.error(f"Error in chunk callback: {e}")
                                
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    
            # Create pipeline result
            result = self._create_pipeline_result(
                pipeline_id,
                start_time,
                start_time + audio_duration,
                audio_duration,
                chunk_results,
                process_start
            )
            
            # Update statistics
            self._update_stats(result)
            
            # Call result callbacks
            for callback in self.result_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in result callback: {e}")
                    
            return result
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            self.stats["error_count"] += 1
            
            return PipelineResult(
                pipeline_id=pipeline_id,
                start_time=start_time,
                end_time=start_time,
                duration=0.0,
                processing_time=time.time() - process_start,
                real_time_factor=0.0,
                success=False,
                error_message=str(e)
            )
            
    def _process_realtime_audio(self, session_id: str):
        """Process real-time audio stream"""
        try:
            while self.is_running:
                try:
                    # Get audio from queue
                    audio_item = self.input_queue.get(timeout=1.0)
                    
                    if audio_item["session_id"] != session_id:
                        continue
                        
                    # Process audio chunk
                    self._process_realtime_chunk(audio_item)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in real-time processing: {e}")
                    
        except Exception as e:
            logger.error(f"Real-time processing failed: {e}")
            
    def _process_realtime_chunk(self, audio_item: Dict[str, Any]):
        """Process a single real-time chunk"""
        try:
            session_id = audio_item["session_id"]
            audio_data = audio_item["audio_data"]
            timestamp = audio_item["timestamp"]
            
            # Create chunk
            chunk = OptimalChunk(
                chunk_id=str(uuid.uuid4()),
                start_time=timestamp,
                end_time=timestamp + len(audio_data) / 16000,  # Assume 16kHz
                duration=len(audio_data) / 16000,
                audio_data=audio_data,
                sample_rate=16000,
                confidence=1.0,
                chunk_type="realtime",
                optimization_score=1.0
            )
            
            # Process chunk
            chunk_result = self._process_chunk(chunk, 0, session_id)
            
            if chunk_result:
                # Add to output queue
                self.output_queue.put({
                    "session_id": session_id,
                    "chunk_result": chunk_result,
                    "timestamp": time.time()
                })
                
                # Call chunk callbacks
                for callback in self.chunk_callbacks:
                    try:
                        callback(chunk_result)
                    except Exception as e:
                        logger.error(f"Error in chunk callback: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing real-time chunk: {e}")
            
    def _process_chunk(self, chunk: OptimalChunk, chunk_index: int, 
                      pipeline_id: str) -> Optional[ChunkResult]:
        """Process a single audio chunk"""
        try:
            # Check cache first
            if self.cache and chunk.chunk_id in self.cache:
                logger.debug(f"Cache hit for chunk {chunk.chunk_id}")
                return self.cache[chunk.chunk_id]
                
            # Transcribe chunk
            transcription_result = self.transcriber.transcribe_audio(
                chunk.audio_data,
                chunk.start_time
            )
            
            if not transcription_result.success:
                logger.warning(f"Transcription failed for chunk {chunk.chunk_id}: {transcription_result.error_message}")
                return None
                
            # Filter by confidence if enabled
            if (self.config.enable_quality_filtering and 
                transcription_result.confidence < self.config.min_confidence_threshold):
                logger.debug(f"Chunk {chunk.chunk_id} filtered due to low confidence: {transcription_result.confidence}")
                return None
                
            # Create chunk result
            chunk_result = ChunkResult(
                chunk_id=chunk.chunk_id,
                chunk_index=chunk_index,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                duration=chunk.duration,
                text=transcription_result.text,
                confidence=transcription_result.confidence,
                language=transcription_result.language,
                processing_time=transcription_result.processing_time,
                real_time_factor=transcription_result.real_time_factor,
                voice_activity=chunk.confidence,  # Use chunk confidence as voice activity
                audio_quality=chunk.optimization_score
            )
            
            # Cache result
            if self.cache:
                self.cache[chunk.chunk_id] = chunk_result
                
                # Limit cache size
                if len(self.cache) > self.config.cache_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    
            return chunk_result
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")
            return None
            
    def _create_pipeline_result(self, pipeline_id: str, start_time: float, 
                              end_time: float, duration: float,
                              chunk_results: List[ChunkResult],
                              process_start: float) -> PipelineResult:
        """Create pipeline result from chunk results"""
        
        processing_time = time.time() - process_start
        
        # Aggregate text
        full_text = " ".join(chunk.text for chunk in chunk_results if chunk.text.strip())
        
        # Calculate metrics
        average_confidence = 0.0
        overall_quality = 0.0
        total_latency = 0.0
        
        if chunk_results:
            average_confidence = sum(chunk.confidence for chunk in chunk_results) / len(chunk_results)
            overall_quality = sum(chunk.audio_quality for chunk in chunk_results) / len(chunk_results)
            total_latency = sum(chunk.processing_time for chunk in chunk_results) / len(chunk_results)
            
        # Calculate throughput
        throughput = len(chunk_results) / processing_time if processing_time > 0 else 0
        
        return PipelineResult(
            pipeline_id=pipeline_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            processing_time=processing_time,
            real_time_factor=processing_time / duration if duration > 0 else 0,
            chunk_results=chunk_results,
            chunk_count=len(chunk_results),
            full_text=full_text,
            average_confidence=average_confidence,
            overall_quality=overall_quality,
            throughput=throughput,
            latency=total_latency,
            success=True
        )
        
    def _update_stats(self, result: PipelineResult):
        """Update pipeline statistics"""
        with self.pipeline_lock:
            self.stats["total_pipelines"] += 1
            self.stats["total_chunks_processed"] += result.chunk_count
            self.stats["total_audio_processed"] += result.duration
            self.stats["total_processing_time"] += result.processing_time
            
            # Update average confidence
            if result.success:
                count = self.stats["total_pipelines"]
                old_avg = self.stats["average_confidence"]
                self.stats["average_confidence"] = (
                    old_avg * (count - 1) + result.average_confidence
                ) / count
                
                # Update average throughput
                old_throughput = self.stats["average_throughput"]
                self.stats["average_throughput"] = (
                    old_throughput * (count - 1) + result.throughput
                ) / count
                
    def add_result_callback(self, callback: Callable[[PipelineResult], None]):
        """Add result callback"""
        self.result_callbacks.append(callback)
        
    def add_chunk_callback(self, callback: Callable[[ChunkResult], None]):
        """Add chunk callback"""
        self.chunk_callbacks.append(callback)
        
    def add_error_callback(self, callback: Callable[[str], None]):
        """Add error callback"""
        self.error_callbacks.append(callback)
        
    def get_realtime_result(self, session_id: str, timeout: float = 1.0) -> Optional[ChunkResult]:
        """Get real-time result from output queue"""
        try:
            output_item = self.output_queue.get(timeout=timeout)
            if output_item["session_id"] == session_id:
                return output_item["chunk_result"]
        except queue.Empty:
            pass
        return None
        
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        with self.pipeline_lock:
            stats = self.stats.copy()
            stats["uptime"] = time.time() - self.stats.get("start_time", time.time())
            return stats
            
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        return {
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "processing_mode": self.config.processing_mode.value,
            "quality_settings": self.config.quality_settings.value,
            "active_pipelines": len(self.active_pipelines),
            "input_queue_size": self.input_queue.qsize(),
            "output_queue_size": self.output_queue.qsize(),
            "stats": self.get_stats()
        }
        
    def shutdown(self):
        """Shutdown pipeline"""
        logger.info("Shutting down transcription pipeline...")
        
        self.is_running = False
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Shutdown components
        if self.transcriber:
            self.transcriber.shutdown()
            
        if self.chunker:
            self.chunker.shutdown()
            
        logger.info("Transcription pipeline shutdown complete")
        
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


# Factory functions
def create_realtime_pipeline() -> TranscriptionPipeline:
    """Create real-time transcription pipeline"""
    config = PipelineConfig(
        processing_mode=ProcessingMode.REAL_TIME,
        quality_settings=QualitySettings.BALANCED,
        chunk_size=5.0,
        max_concurrent_processing=2,
        enable_quality_filtering=True,
        min_confidence_threshold=0.6
    )
    return TranscriptionPipeline(config)


def create_batch_pipeline() -> TranscriptionPipeline:
    """Create batch transcription pipeline"""
    config = PipelineConfig(
        processing_mode=ProcessingMode.BATCH,
        quality_settings=QualitySettings.QUALITY,
        chunk_size=20.0,
        max_concurrent_processing=4,
        enable_quality_filtering=True,
        min_confidence_threshold=0.7
    )
    return TranscriptionPipeline(config)


# Testing and demonstration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcription Pipeline Test")
    parser.add_argument("--audio", type=str, required=True, help="Audio file to process")
    parser.add_argument("--mode", type=str, default="batch", 
                       choices=["realtime", "batch"], help="Processing mode")
    parser.add_argument("--chunk-size", type=float, default=10.0, help="Chunk size in seconds")
    args = parser.parse_args()
    
    # Create pipeline
    if args.mode == "realtime":
        pipeline = create_realtime_pipeline()
    else:
        pipeline = create_batch_pipeline()
        
    # Update chunk size
    pipeline.config.chunk_size = args.chunk_size
    
    try:
        # Initialize
        if not pipeline.initialize():
            print("Failed to initialize pipeline")
            sys.exit(1)
            
        print(f"Pipeline status: {pipeline.get_status()}")
        
        # Process audio
        print(f"Processing: {args.audio}")
        result = pipeline.process_audio_file(args.audio)
        
        if result.success:
            print(f"Processing completed successfully")
            print(f"Full text: {result.full_text}")
            print(f"Chunks processed: {result.chunk_count}")
            print(f"Average confidence: {result.average_confidence:.3f}")
            print(f"Processing time: {result.processing_time:.3f}s")
            print(f"Real-time factor: {result.real_time_factor:.3f}")
            print(f"Throughput: {result.throughput:.3f} chunks/s")
            
            # Show chunk details
            print("\nChunk details:")
            for i, chunk in enumerate(result.chunk_results[:5]):  # Show first 5 chunks
                print(f"  Chunk {i}: {chunk.text[:50]}... (conf: {chunk.confidence:.3f})")
        else:
            print(f"Processing failed: {result.error_message}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pipeline.shutdown()