#!/usr/bin/env python3

"""
Simple Transcriber Bridge for The Silent Steno

This module provides a simplified transcription interface that bridges the
working Whisper transcription from minimal_demo.py with a backend-agnostic
architecture. This allows for easy switching between CPU Whisper and future
Hailo Whisper implementations.

Key features:
- Backend-agnostic transcription interface
- Direct Whisper integration (CPU backend)
- Future-ready for Hailo Whisper backend
- Simple API matching minimal_demo.py usage
- Configurable model selection
"""

import os
import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import time

# Import data integration adapter
try:
    from src.data.integration_adapter import DataIntegrationAdapter
    DATA_INTEGRATION_AVAILABLE = True
except ImportError:
    DATA_INTEGRATION_AVAILABLE = False
    DataIntegrationAdapter = None
    logging.warning("Data integration adapter not available")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptionBackend(ABC):
    """Abstract base class for transcription backends"""
    
    @abstractmethod
    def transcribe(self, audio_file: str) -> str:
        """Transcribe an audio file and return the text"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available"""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get backend information"""
        pass


class WhisperCPUBackend(TranscriptionBackend):
    """CPU-based Whisper transcription backend"""
    
    def __init__(self, model_name: str = "base"):
        """Initialize Whisper CPU backend"""
        self.model_name = model_name
        self.model = None
        self.whisper_available = False
        
        # Check Whisper availability
        try:
            import whisper
            self.whisper_available = True
            self.whisper_module = whisper
            logger.info("Whisper CPU backend available")
        except ImportError:
            logger.warning("Whisper not available - install with: pip install openai-whisper")
    
    def _ensure_model_loaded(self):
        """Ensure Whisper model is loaded"""
        if self.whisper_available and self.model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")
            start_time = time.time()
            
            try:
                self.model = self.whisper_module.load_model(self.model_name)
                load_time = time.time() - start_time
                logger.info(f"Whisper model loaded successfully in {load_time:.2f}s")
            except Exception as e:
                logger.error(f"Error loading Whisper model: {e}")
                self.model = None
    
    def transcribe(self, audio_file: str) -> str:
        """Transcribe audio file using Whisper"""
        if not self.whisper_available:
            return "Transcription error: Whisper not available"
        
        if not os.path.exists(audio_file):
            return f"Transcription error: Audio file not found: {audio_file}"
        
        try:
            # Ensure model is loaded
            self._ensure_model_loaded()
            
            if self.model is None:
                return "Transcription error: Failed to load Whisper model"
            
            logger.info(f"Starting transcription of: {audio_file}")
            start_time = time.time()
            
            # Transcribe the audio
            result = self.model.transcribe(audio_file)
            
            transcription_time = time.time() - start_time
            text = result.get('text', '').strip()
            
            logger.info(f"Transcription completed in {transcription_time:.2f}s")
            logger.info(f"Transcribed text: {text[:100]}...")
            
            return text
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return f"Transcription error: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if Whisper CPU backend is available"""
        return self.whisper_available
    
    def get_info(self) -> Dict[str, Any]:
        """Get backend information"""
        return {
            'backend': 'WhisperCPU',
            'available': self.whisper_available,
            'model_name': self.model_name,
            'model_loaded': self.model is not None
        }


class HailoWhisperBackend(TranscriptionBackend):
    """Hailo-accelerated Whisper transcription backend (placeholder)"""
    
    def __init__(self, model_name: str = "base"):
        """Initialize Hailo Whisper backend"""
        self.model_name = model_name
        self.hailo_available = False
        
        # TODO: Check for Hailo runtime availability
        logger.info("Hailo Whisper backend initialized (placeholder)")
    
    def transcribe(self, audio_file: str) -> str:
        """Transcribe audio file using Hailo-accelerated Whisper"""
        # TODO: Implement Hailo Whisper transcription
        return "Hailo Whisper backend not yet implemented"
    
    def is_available(self) -> bool:
        """Check if Hailo backend is available"""
        return self.hailo_available
    
    def get_info(self) -> Dict[str, Any]:
        """Get backend information"""
        return {
            'backend': 'HailoWhisper',
            'available': self.hailo_available,
            'model_name': self.model_name,
            'status': 'placeholder'
        }


class SimpleTranscriber:
    """
    Simple Transcriber for The Silent Steno
    
    Provides a simplified, backend-agnostic transcription interface that
    matches the usage in minimal_demo.py while supporting multiple backends.
    """
    
    def __init__(self, backend: str = "cpu", model_name: str = "base", 
                 data_adapter: Optional[Any] = None):
        """
        Initialize transcriber with specified backend
        
        Args:
            backend: Backend to use ("cpu" or "hailo")
            model_name: Model name to use (e.g., "base", "small", "medium")
            data_adapter: Optional data adapter for storing transcription results
        """
        self.backend_name = backend
        self.model_name = model_name
        self.data_adapter = data_adapter
        
        # Initialize the appropriate backend
        if backend == "cpu":
            self.backend = WhisperCPUBackend(model_name)
        elif backend == "hailo":
            self.backend = HailoWhisperBackend(model_name)
        else:
            logger.warning(f"Unknown backend '{backend}', defaulting to CPU")
            self.backend = WhisperCPUBackend(model_name)
        
        logger.info(f"SimpleTranscriber initialized with {backend} backend, model: {model_name}")
        
        if self.data_adapter:
            logger.info("Data integration adapter connected to transcriber")
    
    def transcribe_audio(self, wav_file: str) -> str:
        """
        Transcribe audio file (main interface matching minimal_demo.py)
        
        Args:
            wav_file: Path to WAV file to transcribe
            
        Returns:
            Transcribed text or error message
        """
        if not self.backend.is_available():
            return f"Transcription error: {self.backend_name} backend not available"
        
        return self.backend.transcribe(wav_file)
    
    def transcribe_and_update_session(self, wav_file: str, session_id: str) -> str:
        """
        Transcribe audio file and update session data
        
        Args:
            wav_file: Path to WAV file to transcribe
            session_id: Session identifier to update
            
        Returns:
            Transcribed text or error message
        """
        if not self.backend.is_available():
            return f"Transcription error: {self.backend_name} backend not available"
        
        # Perform transcription
        transcript = self.backend.transcribe(wav_file)
        
        # Update session data if data adapter is available
        if self.data_adapter and transcript and not transcript.startswith("Transcription error"):
            try:
                success = self.data_adapter.update_session(session_id, {'transcript': transcript})
                if success:
                    logger.info(f"Updated session {session_id} with transcript")
                else:
                    logger.warning(f"Failed to update session {session_id} with transcript")
            except Exception as e:
                logger.error(f"Error updating session {session_id}: {e}")
        
        return transcript
    
    def is_available(self) -> bool:
        """Check if transcription is available"""
        return self.backend.is_available()
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend"""
        return self.backend.get_info()
    
    def switch_backend(self, backend: str):
        """
        Switch to a different transcription backend
        
        Args:
            backend: Backend to switch to ("cpu" or "hailo")
        """
        old_backend = self.backend_name
        
        if backend == "cpu":
            self.backend = WhisperCPUBackend(self.model_name)
            self.backend_name = backend
        elif backend == "hailo":
            self.backend = HailoWhisperBackend(self.model_name)
            self.backend_name = backend
        else:
            logger.warning(f"Unknown backend '{backend}', keeping current backend")
            return
        
        logger.info(f"Switched transcription backend from {old_backend} to {backend}")
    
    def list_available_backends(self) -> List[Dict[str, Any]]:
        """List all available transcription backends"""
        backends = []
        
        # Check CPU backend
        cpu_backend = WhisperCPUBackend(self.model_name)
        backends.append({
            'name': 'cpu',
            'available': cpu_backend.is_available(),
            'info': cpu_backend.get_info()
        })
        
        # Check Hailo backend
        hailo_backend = HailoWhisperBackend(self.model_name)
        backends.append({
            'name': 'hailo',
            'available': hailo_backend.is_available(),
            'info': hailo_backend.get_info()
        })
        
        return backends


# Convenience function to create transcriber (matches minimal_demo.py usage)
def create_simple_transcriber(backend: str = "cpu", model_name: str = "base") -> SimpleTranscriber:
    """Create a simple transcriber instance"""
    return SimpleTranscriber(backend, model_name)


if __name__ == "__main__":
    # Basic test when run directly
    print("Simple Transcriber Test")
    print("=" * 50)
    
    # Create transcriber
    transcriber = SimpleTranscriber(backend="cpu", model_name="base")
    
    # Check availability
    print(f"Transcriber available: {transcriber.is_available()}")
    print(f"Backend info: {transcriber.get_backend_info()}")
    
    # List available backends
    print("\nAvailable backends:")
    for backend in transcriber.list_available_backends():
        print(f"  - {backend['name']}: {backend['available']}")
    
    # Test transcription with a dummy file
    test_file = "test_audio.wav"
    if os.path.exists(test_file):
        print(f"\nTranscribing: {test_file}")
        result = transcriber.transcribe_audio(test_file)
        print(f"Result: {result}")
    else:
        print(f"\nNo test file found at: {test_file}")
    
    print("\nTest complete")