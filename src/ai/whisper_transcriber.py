#!/usr/bin/env python3
"""
Whisper Transcriber Module

Local Whisper Base model integration for real-time speech-to-text transcription
with Pi 5 optimization and comprehensive transcription capabilities.

Author: Claude AI Assistant
Date: 2024-07-14
Version: 1.0
"""

import os
import sys
import logging
import threading
import time
import tempfile
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import uuid
import warnings

try:
    import numpy as np
    import torch
    import whisper
    import soundfile as sf
    from transformers import pipeline
except ImportError as e:
    print(f"Warning: Required dependencies not installed: {e}")
    print("Please install with: pip install openai-whisper torch torchaudio transformers soundfile")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress whisper warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ModelSize(Enum):
    """Whisper model sizes"""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


class WhisperModel(Enum):
    """Whisper model variants"""
    OPENAI_WHISPER = "openai_whisper"
    TRANSFORMERS = "transformers"


@dataclass
class TranscriptionConfig:
    """Configuration for Whisper transcription"""
    
    # Model settings
    model_size: ModelSize = ModelSize.BASE
    model_type: WhisperModel = WhisperModel.OPENAI_WHISPER
    device: str = "cpu"  # Pi 5 uses CPU
    
    # Processing settings
    language: Optional[str] = None  # Auto-detect if None
    task: str = "transcribe"  # or "translate"
    
    # Quality settings
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    
    # Timestamps
    word_timestamps: bool = True
    prepend_punctuations: str = "\"'([{-"
    append_punctuations: str = "\"'.!?:)]},"
    
    # Audio preprocessing
    normalize_audio: bool = True
    suppress_blank: bool = True
    suppress_tokens: List[int] = field(default_factory=lambda: [-1])
    
    # Performance settings
    fp16: bool = False  # Use FP32 for better Pi 5 compatibility
    chunk_length: int = 30  # seconds
    
    # Confidence settings
    min_confidence: float = 0.5
    word_confidence_threshold: float = 0.3
    
    # Temperature settings for sampling
    temperature: Union[float, List[float]] = 0.0
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "model_size": self.model_size.value,
            "model_type": self.model_type.value,
            "device": self.device,
            "language": self.language,
            "task": self.task,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "patience": self.patience,
            "length_penalty": self.length_penalty,
            "repetition_penalty": self.repetition_penalty,
            "word_timestamps": self.word_timestamps,
            "normalize_audio": self.normalize_audio,
            "suppress_blank": self.suppress_blank,
            "fp16": self.fp16,
            "chunk_length": self.chunk_length,
            "min_confidence": self.min_confidence,
            "temperature": self.temperature,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "logprob_threshold": self.logprob_threshold,
            "no_speech_threshold": self.no_speech_threshold
        }


@dataclass
class TranscriptionResult:
    """Result from transcription operation"""
    
    # Core transcription data
    text: str
    language: str
    confidence: float
    
    # Timing information
    start_time: float
    end_time: float
    duration: float
    
    # Word-level information
    words: List[Dict[str, Any]] = field(default_factory=list)
    
    # Segments information
    segments: List[Dict[str, Any]] = field(default_factory=list)
    
    # Processing metadata
    processing_time: float = 0.0
    model_used: str = ""
    
    # Quality metrics
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    
    # Status
    success: bool = True
    error_message: str = ""
    
    # Additional metadata
    audio_duration: float = 0.0
    real_time_factor: float = 0.0  # processing_time / audio_duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "text": self.text,
            "language": self.language,
            "confidence": self.confidence,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "words": self.words,
            "segments": self.segments,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "no_speech_prob": self.no_speech_prob,
            "avg_logprob": self.avg_logprob,
            "compression_ratio": self.compression_ratio,
            "success": self.success,
            "error_message": self.error_message,
            "audio_duration": self.audio_duration,
            "real_time_factor": self.real_time_factor
        }


class WhisperTranscriber:
    """Main Whisper transcription engine"""
    
    def __init__(self, config: Optional[TranscriptionConfig] = None):
        self.config = config or TranscriptionConfig()
        self.model = None
        self.is_initialized = False
        self.model_info = {}
        
        # Performance tracking
        self.stats = {
            "total_transcriptions": 0,
            "total_audio_processed": 0.0,
            "total_processing_time": 0.0,
            "average_confidence": 0.0,
            "error_count": 0,
            "model_load_time": 0.0,
            "uptime": 0.0
        }
        
        # Threading
        self.transcription_lock = threading.Lock()
        
        # Callbacks
        self.result_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        logger.info(f"WhisperTranscriber initialized with {self.config.model_size.value} model")
        
    def initialize(self) -> bool:
        """Initialize the Whisper model"""
        try:
            logger.info("Loading Whisper model...")
            start_time = time.time()
            
            # Load model based on type
            if self.config.model_type == WhisperModel.OPENAI_WHISPER:
                self.model = whisper.load_model(
                    self.config.model_size.value,
                    device=self.config.device
                )
            elif self.config.model_type == WhisperModel.TRANSFORMERS:
                model_name = f"openai/whisper-{self.config.model_size.value}"
                self.model = pipeline(
                    "automatic-speech-recognition",
                    model=model_name,
                    device=0 if self.config.device == "cuda" else -1,
                    torch_dtype=torch.float32 if not self.config.fp16 else torch.float16
                )
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
                
            load_time = time.time() - start_time
            self.stats["model_load_time"] = load_time
            
            # Store model info
            self.model_info = {
                "model_size": self.config.model_size.value,
                "model_type": self.config.model_type.value,
                "device": self.config.device,
                "load_time": load_time,
                "parameters": self._get_model_parameters()
            }
            
            self.is_initialized = True
            logger.info(f"Whisper model loaded in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            self._call_error_callbacks(str(e))
            return False
            
    def _get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameter information"""
        try:
            if self.config.model_type == WhisperModel.OPENAI_WHISPER:
                return {
                    "dims": getattr(self.model, "dims", {}),
                    "n_vocab": getattr(self.model.decoder, "n_vocab", 0),
                    "n_ctx": getattr(self.model.decoder, "n_ctx", 0)
                }
            else:
                # Transformers model
                return {
                    "model_name": getattr(self.model.model, "name_or_path", ""),
                    "config": str(getattr(self.model.model, "config", {}))
                }
        except Exception as e:
            logger.warning(f"Could not get model parameters: {e}")
            return {}
            
    def transcribe_audio(self, audio_data: Union[np.ndarray, str], 
                        start_time: float = 0.0) -> TranscriptionResult:
        """Transcribe audio data or file"""
        if not self.is_initialized:
            return TranscriptionResult(
                text="",
                language="",
                confidence=0.0,
                start_time=start_time,
                end_time=start_time,
                duration=0.0,
                success=False,
                error_message="Transcriber not initialized"
            )
            
        process_start = time.time()
        
        try:
            with self.transcription_lock:
                # Prepare audio
                if isinstance(audio_data, str):
                    # File path
                    audio_path = audio_data
                    audio_array, sample_rate = sf.read(audio_path)
                    audio_duration = len(audio_array) / sample_rate
                else:
                    # NumPy array
                    audio_array = audio_data
                    sample_rate = 16000  # Assume 16kHz
                    audio_duration = len(audio_array) / sample_rate
                    
                    # Save to temporary file for Whisper
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        sf.write(temp_file.name, audio_array, sample_rate)
                        audio_path = temp_file.name
                
                # Transcribe based on model type
                if self.config.model_type == WhisperModel.OPENAI_WHISPER:
                    result = self._transcribe_openai_whisper(audio_path, start_time, audio_duration)
                else:
                    result = self._transcribe_transformers(audio_path, start_time, audio_duration)
                
                # Clean up temporary file if created
                if not isinstance(audio_data, str) and os.path.exists(audio_path):
                    os.unlink(audio_path)
                
                # Calculate processing metrics
                processing_time = time.time() - process_start
                result.processing_time = processing_time
                result.real_time_factor = processing_time / audio_duration if audio_duration > 0 else 0
                
                # Update statistics
                self._update_stats(result)
                
                # Call result callbacks
                self._call_result_callbacks(result)
                
                return result
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            self.stats["error_count"] += 1
            self._call_error_callbacks(str(e))
            
            return TranscriptionResult(
                text="",
                language="",
                confidence=0.0,
                start_time=start_time,
                end_time=start_time,
                duration=0.0,
                success=False,
                error_message=str(e),
                processing_time=time.time() - process_start
            )
            
    def _transcribe_openai_whisper(self, audio_path: str, start_time: float, 
                                 audio_duration: float) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper"""
        
        # Prepare options
        options = {
            "language": self.config.language,
            "task": self.config.task,
            "beam_size": self.config.beam_size,
            "best_of": self.config.best_of,
            "patience": self.config.patience,
            "length_penalty": self.config.length_penalty,
            "repetition_penalty": self.config.repetition_penalty,
            "word_timestamps": self.config.word_timestamps,
            "prepend_punctuations": self.config.prepend_punctuations,
            "append_punctuations": self.config.append_punctuations,
            "suppress_blank": self.config.suppress_blank,
            "suppress_tokens": self.config.suppress_tokens,
            "fp16": self.config.fp16,
            "temperature": self.config.temperature,
            "compression_ratio_threshold": self.config.compression_ratio_threshold,
            "logprob_threshold": self.config.logprob_threshold,
            "no_speech_threshold": self.config.no_speech_threshold
        }
        
        # Remove None values
        options = {k: v for k, v in options.items() if v is not None}
        
        # Transcribe
        result = self.model.transcribe(audio_path, **options)
        
        # Extract information
        text = result.get("text", "").strip()
        language = result.get("language", "")
        segments = result.get("segments", [])
        
        # Calculate confidence
        confidence = self._calculate_confidence(segments)
        
        # Extract words
        words = []
        for segment in segments:
            if "words" in segment:
                words.extend(segment["words"])
        
        # Calculate timing
        end_time = start_time + audio_duration
        
        return TranscriptionResult(
            text=text,
            language=language,
            confidence=confidence,
            start_time=start_time,
            end_time=end_time,
            duration=audio_duration,
            words=words,
            segments=segments,
            model_used=f"whisper-{self.config.model_size.value}",
            no_speech_prob=result.get("no_speech_prob", 0.0),
            avg_logprob=result.get("avg_logprob", 0.0),
            compression_ratio=result.get("compression_ratio", 0.0),
            audio_duration=audio_duration,
            success=True
        )
        
    def _transcribe_transformers(self, audio_path: str, start_time: float, 
                               audio_duration: float) -> TranscriptionResult:
        """Transcribe using Transformers pipeline"""
        
        # Prepare options
        options = {
            "return_timestamps": self.config.word_timestamps,
            "chunk_length_s": self.config.chunk_length,
            "stride_length_s": self.config.chunk_length // 4
        }
        
        # Transcribe
        result = self.model(audio_path, **options)
        
        # Extract information
        text = result.get("text", "").strip()
        chunks = result.get("chunks", [])
        
        # Convert chunks to segments format
        segments = []
        words = []
        
        for i, chunk in enumerate(chunks):
            segment = {
                "id": i,
                "text": chunk.get("text", ""),
                "start": chunk.get("timestamp", [0, 0])[0],
                "end": chunk.get("timestamp", [0, 0])[1],
                "avg_logprob": 0.0,
                "compression_ratio": 0.0,
                "no_speech_prob": 0.0
            }
            segments.append(segment)
            
            # Add word-level timestamps if available
            if self.config.word_timestamps:
                segment_words = self._extract_words_from_chunk(chunk)
                words.extend(segment_words)
        
        # Calculate confidence (approximate for transformers)
        confidence = 0.8  # Default confidence for transformers
        
        # Calculate timing
        end_time = start_time + audio_duration
        
        return TranscriptionResult(
            text=text,
            language="auto",  # Language detection not available in transformers
            confidence=confidence,
            start_time=start_time,
            end_time=end_time,
            duration=audio_duration,
            words=words,
            segments=segments,
            model_used=f"transformers-whisper-{self.config.model_size.value}",
            audio_duration=audio_duration,
            success=True
        )
        
    def _extract_words_from_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract word-level timestamps from chunk"""
        words = []
        
        # Simple word splitting - in practice, would need more sophisticated parsing
        text = chunk.get("text", "")
        timestamp = chunk.get("timestamp", [0, 0])
        
        if len(text.strip()) > 0:
            word_list = text.strip().split()
            duration = timestamp[1] - timestamp[0]
            word_duration = duration / len(word_list) if len(word_list) > 0 else 0
            
            for i, word in enumerate(word_list):
                word_start = timestamp[0] + i * word_duration
                word_end = word_start + word_duration
                
                words.append({
                    "word": word,
                    "start": word_start,
                    "end": word_end,
                    "probability": 0.8  # Default probability
                })
        
        return words
        
    def _calculate_confidence(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence from segments"""
        if not segments:
            return 0.0
            
        # Use average logprob as confidence approximation
        total_logprob = 0.0
        total_length = 0
        
        for segment in segments:
            logprob = segment.get("avg_logprob", 0.0)
            length = len(segment.get("text", ""))
            total_logprob += logprob * length
            total_length += length
        
        if total_length == 0:
            return 0.0
            
        avg_logprob = total_logprob / total_length
        
        # Convert logprob to confidence (approximate)
        # This is a heuristic - logprob ranges from -inf to 0
        confidence = max(0.0, min(1.0, (avg_logprob + 1.0) / 1.0))
        
        return confidence
        
    def _update_stats(self, result: TranscriptionResult):
        """Update transcription statistics"""
        self.stats["total_transcriptions"] += 1
        
        if result.success:
            self.stats["total_audio_processed"] += result.audio_duration
            self.stats["total_processing_time"] += result.processing_time
            
            # Update average confidence
            count = self.stats["total_transcriptions"]
            old_avg = self.stats["average_confidence"]
            self.stats["average_confidence"] = (
                old_avg * (count - 1) + result.confidence
            ) / count
            
    def _call_result_callbacks(self, result: TranscriptionResult):
        """Call result callbacks"""
        for callback in self.result_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in result callback: {e}")
                
    def _call_error_callbacks(self, error_message: str):
        """Call error callbacks"""
        for callback in self.error_callbacks:
            try:
                callback(error_message)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
                
    def add_result_callback(self, callback: Callable[[TranscriptionResult], None]):
        """Add result callback"""
        self.result_callbacks.append(callback)
        
    def add_error_callback(self, callback: Callable[[str], None]):
        """Add error callback"""
        self.error_callbacks.append(callback)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get transcription statistics"""
        return self.stats.copy()
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.model_info.copy()
        
    def get_status(self) -> Dict[str, Any]:
        """Get transcriber status"""
        return {
            "is_initialized": self.is_initialized,
            "model_info": self.get_model_info(),
            "stats": self.get_stats(),
            "config": self.config.to_dict()
        }
        
    def shutdown(self):
        """Shutdown transcriber"""
        logger.info("Shutting down Whisper transcriber...")
        self.is_initialized = False
        
        # Clear model from memory
        if self.model:
            del self.model
            self.model = None
            
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Whisper transcriber shutdown complete")
        
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


# Factory functions
def create_base_transcriber(device: str = "cpu") -> WhisperTranscriber:
    """Create base Whisper transcriber"""
    config = TranscriptionConfig(
        model_size=ModelSize.BASE,
        device=device,
        word_timestamps=True,
        min_confidence=0.6
    )
    return WhisperTranscriber(config)


def create_optimized_transcriber(device: str = "cpu") -> WhisperTranscriber:
    """Create optimized Whisper transcriber for Pi 5"""
    config = TranscriptionConfig(
        model_size=ModelSize.BASE,
        device=device,
        word_timestamps=True,
        min_confidence=0.7,
        beam_size=3,  # Reduced for performance
        best_of=3,    # Reduced for performance
        fp16=False,   # Better Pi 5 compatibility
        temperature=0.1,
        chunk_length=20  # Shorter chunks for real-time
    )
    return WhisperTranscriber(config)


# Testing and demonstration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Whisper Transcriber Test")
    parser.add_argument("--audio", type=str, required=True, help="Audio file to transcribe")
    parser.add_argument("--model", type=str, default="base", 
                       choices=["tiny", "base", "small", "medium", "large"], 
                       help="Model size")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--language", type=str, default=None, help="Language code")
    args = parser.parse_args()
    
    # Create transcriber
    config = TranscriptionConfig(
        model_size=ModelSize(args.model),
        device=args.device,
        language=args.language
    )
    
    transcriber = create_optimized_transcriber(args.device)
    
    try:
        # Initialize
        if not transcriber.initialize():
            print("Failed to initialize transcriber")
            sys.exit(1)
            
        print(f"Transcriber status: {transcriber.get_status()}")
        
        # Transcribe
        print(f"Transcribing: {args.audio}")
        result = transcriber.transcribe_audio(args.audio)
        
        if result.success:
            print(f"Transcription: {result.text}")
            print(f"Language: {result.language}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Processing time: {result.processing_time:.3f}s")
            print(f"Real-time factor: {result.real_time_factor:.3f}")
            
            if result.words:
                print(f"Words: {len(result.words)}")
                for word in result.words[:5]:  # Show first 5 words
                    print(f"  {word}")
        else:
            print(f"Transcription failed: {result.error_message}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        transcriber.shutdown()