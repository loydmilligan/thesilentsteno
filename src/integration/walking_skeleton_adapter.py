#!/usr/bin/env python3

"""
Walking Skeleton Integration Adapter for The Silent Steno

This module provides a bridge between the simple walking skeleton implementation
and the comprehensive production architecture. It allows for gradual migration
from the prototype to the full system while maintaining backward compatibility.

Key features:
- Adapts SimpleAudioRecorder to use AudioPipeline
- Integrates with DeviceManager for system health
- Bridges to comprehensive SessionManager
- Connects to production AI pipeline
- Maintains simple API for UI compatibility
"""

import os
import logging
import threading
import time
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import production components (with fallback for missing imports)
try:
    from src.audio import AudioPipeline, AudioConfig, create_audio_pipeline
    from src.recording import SessionManager, SessionType, SessionConfig
    from src.ai import create_meeting_ai_system
    from src.system import DeviceManager, DeviceConfig, create_device_manager
    PRODUCTION_IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Production imports not available: {e}")
    PRODUCTION_IMPORTS_AVAILABLE = False

# Import walking skeleton components
try:
    from src.recording.simple_audio_recorder import SimpleAudioRecorder
    from src.ai.simple_transcriber import SimpleTranscriber
    from src.data.integration_adapter import DataIntegrationAdapter
    SKELETON_IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Walking skeleton imports not available: {e}")
    SKELETON_IMPORTS_AVAILABLE = False


class WalkingSkeletonAdapter:
    """
    Adapter that bridges walking skeleton components with production architecture
    
    This adapter provides a gradual migration path from the simple prototype
    to the comprehensive production system while maintaining compatibility.
    """
    
    def __init__(self, use_production: bool = True, config: Dict[str, Any] = None):
        """
        Initialize the adapter
        
        Args:
            use_production: Whether to use production components when available
            config: Configuration dictionary
        """
        self.use_production = use_production and PRODUCTION_IMPORTS_AVAILABLE
        self.config = config or {}
        
        # Component instances
        self.audio_system = None
        self.session_manager = None
        self.ai_system = None
        self.device_manager = None
        
        # Walking skeleton components (fallback)
        self.simple_recorder = None
        self.simple_transcriber = None
        self.data_adapter = None
        
        # State
        self.is_initialized = False
        self.current_session_id = None
        self.recording_state = "idle"
        
        # Callbacks
        self.recording_callbacks = []
        self.transcription_callbacks = []
        self.state_callbacks = []
        
        logger.info(f"Walking Skeleton Adapter initialized (production mode: {self.use_production})")
    
    def initialize(self):
        """Initialize all components"""
        try:
            if self.use_production:
                self._initialize_production_components()
            else:
                self._initialize_skeleton_components()
            
            self.is_initialized = True
            logger.info("Adapter initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing adapter: {e}")
            # Fallback to skeleton components
            if self.use_production:
                logger.info("Falling back to skeleton components")
                self.use_production = False
                self._initialize_skeleton_components()
    
    def _initialize_production_components(self):
        """Initialize production architecture components"""
        logger.info("Initializing production components...")
        
        # Initialize audio system
        audio_config = self.config.get('audio', {})
        self.audio_system = create_audio_pipeline(
            sample_rate=44100,
            buffer_size=512,
            target_latency_ms=audio_config.get('target_latency', 40.0)
        )
        
        # Initialize session manager
        session_config = SessionConfig(
            session_type=SessionType.MEETING,
            quality_preset=self.config.get('quality_preset', 'balanced'),
            enable_preprocessing=True,
            enable_real_time_analysis=True
        )
        self.session_manager = SessionManager(
            storage_root=self.config.get('storage_root', 'recordings'),
            config=session_config
        )
        
        # Initialize AI system
        self.ai_system = create_meeting_ai_system({
            'whisper_model': self.config.get('whisper_model', 'base'),
            'enable_speaker_diarization': False,  # Disabled for walking skeleton
            'real_time_processing': True
        })
        self.ai_system.initialize()
        
        # Initialize device manager
        device_config = DeviceConfig(
            device_name="Silent Steno Walking Skeleton",
            health_monitoring_enabled=True,
            storage_cleanup_enabled=True,
            auto_update_enabled=False  # Disabled for walking skeleton
        )
        self.device_manager = create_device_manager(device_config)
        self.device_manager.initialize()
        
        # Set up integrations
        self.session_manager.set_audio_recorder(self.audio_system)
        
        # Set up callbacks
        self.audio_system.add_audio_callback(self._on_audio_data)
        self.session_manager.add_state_callback(self._on_session_state_change)
        
        logger.info("Production components initialized successfully")
    
    def _initialize_skeleton_components(self):
        """Initialize walking skeleton components"""
        logger.info("Initializing skeleton components...")
        
        # Initialize data adapter (temporarily disable database to avoid crash)
        self.data_adapter = DataIntegrationAdapter(
            sessions_file="demo_sessions/sessions.json",
            use_database=False
        )
        
        # Initialize Bluetooth-capable recorder
        try:
            from src.recording.bluetooth_audio_recorder_module import BluetoothAudioRecorder
            self.simple_recorder = BluetoothAudioRecorder(
                storage_root=self.config.get('storage_root', 'demo_sessions')
            )
            logger.info("Using Bluetooth audio recorder")
        except ImportError:
            # Fallback to regular recorder
            self.simple_recorder = SimpleAudioRecorder(
                storage_root=self.config.get('storage_root', 'demo_sessions')
            )
            logger.info("Using USB audio recorder (fallback)")
        
        # Initialize simple transcriber with optional Gemini enhancement
        use_gemini = self.config.get('use_gemini_enhancement', False)
        gemini_api_key = self.config.get('gemini_api_key') or os.getenv('GEMINI_API_KEY')
        
        self.simple_transcriber = SimpleTranscriber(
            backend="cpu",
            model_name=self.config.get('whisper_model', 'base'),
            data_adapter=self.data_adapter,
            use_gemini=use_gemini,
            gemini_api_key=gemini_api_key
        )
        
        logger.info("Skeleton components initialized successfully")
    
    def start_recording(self, session_id: str = None) -> Optional[str]:
        """
        Start recording with either production or skeleton components
        
        Args:
            session_id: Optional session ID (generated if not provided)
            
        Returns:
            Session ID if successful, None otherwise
        """
        try:
            if self.use_production and self.session_manager:
                # Use production session manager
                session_id = self.session_manager.start_session(
                    session_type=SessionType.MEETING,
                    config=None,
                    metadata={'source': 'walking_skeleton'}
                )
                
                if session_id:
                    self.current_session_id = session_id
                    self.recording_state = "recording"
                    self._notify_state_change("recording")
                    
                    # Start audio system if not running
                    if hasattr(self.audio_system, 'start_pipeline'):
                        self.audio_system.start_pipeline()
                    elif hasattr(self.audio_system, 'start'):
                        self.audio_system.start()
                    
                    logger.info(f"Started production recording: {session_id}")
                    return session_id
                    
            else:
                # Use skeleton recorder
                session_id = session_id or f"skeleton_{int(time.time())}"
                
                if self.simple_recorder.start_recording(session_id):
                    self.current_session_id = session_id
                    self.recording_state = "recording"
                    self._notify_state_change("recording")
                    
                    logger.info(f"Started skeleton recording: {session_id}")
                    return session_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return None
    
    def stop_recording(self) -> Optional[Dict[str, Any]]:
        """
        Stop recording and return recording info
        
        Returns:
            Recording info dictionary if successful, None otherwise
        """
        logger.info(f"Stopping recording... current_session_id: {self.current_session_id}")
        logger.info(f"Recording state: {self.recording_state}")
        logger.info(f"Using production: {self.use_production}")
        
        try:
            if self.use_production and self.session_manager:
                # Stop production recording
                success = self.session_manager.stop_session(self.current_session_id)
                
                if success:
                    session_info = self.session_manager.get_session_status(self.current_session_id)
                    self.recording_state = "stopped"
                    self._notify_state_change("stopped")
                    
                    # Convert to simple format
                    recording_info = {
                        'session_id': session_info.session_id,
                        'duration': session_info.duration_seconds,
                        'file_path': session_info.file_path,
                        'state': 'stopped'
                    }
                    
                    logger.info(f"Stopped production recording: {self.current_session_id}")
                    return recording_info
                    
            else:
                # Stop skeleton recording
                logger.info("Calling simple_recorder.stop_recording()")
                recording_info = self.simple_recorder.stop_recording()
                logger.info(f"Simple recorder returned: {recording_info}")
                
                if recording_info:
                    self.recording_state = "stopped"
                    self._notify_state_change("stopped")
                    
                    logger.info(f"Stopped skeleton recording: {self.current_session_id}")
                    # Reset current session ID for next recording
                    self.current_session_id = None
                    return recording_info
                else:
                    logger.error("Simple recorder returned None")
            
            return None
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return None
    
    def transcribe_recording(self, wav_file: str = None, session_id: str = None) -> Optional[str]:
        """
        Transcribe the current or specified recording
        
        Args:
            wav_file: Optional WAV file path (uses current recording if not provided)
            session_id: Optional session ID for background processing
            
        Returns:
            Transcription text if successful, None otherwise
        """
        try:
            if self.use_production and self.ai_system:
                # Use production AI system
                if not wav_file:
                    # Get file from current session
                    session_info = self.session_manager.get_session_status(self.current_session_id)
                    wav_file = session_info.file_path if session_info else None
                
                if wav_file:
                    result = self.ai_system.process_audio_file(wav_file)
                    transcript = result.transcript if result else None
                    
                    if transcript:
                        self._notify_transcription(transcript)
                        logger.info(f"Production transcription complete: {len(transcript)} chars")
                        return transcript
                        
            else:
                # Use skeleton transcriber
                if not wav_file and self.simple_recorder:
                    recording_info = self.simple_recorder.get_current_recording_info()
                    wav_file = recording_info['wav_file'] if recording_info else None
                
                if wav_file and self.simple_transcriber:
                    # Use provided session_id or fall back to current_session_id
                    target_session_id = session_id or self.current_session_id
                    
                    if target_session_id:
                        transcript = self.simple_transcriber.transcribe_and_update_session(
                            wav_file, target_session_id
                        )
                    else:
                        transcript = self.simple_transcriber.transcribe_audio(wav_file)
                    
                    if transcript:
                        # Perform AI analysis on the transcript
                        analysis = self.simple_transcriber.analyze_transcript(transcript)
                        
                        # Create enhanced result with analysis
                        enhanced_result = {
                            'transcript': transcript,
                            'analysis': analysis
                        }
                        
                        self._notify_transcription(enhanced_result)
                        logger.info(f"Skeleton transcription complete: {len(transcript)} chars")
                        logger.info(f"Analysis: {analysis.get('summary', 'No summary')[:100]}...")
                        return enhanced_result
            
            return None
            
        except Exception as e:
            logger.error(f"Error transcribing recording: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: ", exc_info=True)
            return None
    
    def play_recording(self) -> bool:
        """
        Play back the current recording
        
        Returns:
            True if playback started successfully
        """
        try:
            if self.simple_recorder:
                # Use simple recorder for playback (works for both modes)
                return self.simple_recorder.play_recording()
            
            return False
            
        except Exception as e:
            logger.error(f"Error playing recording: {e}")
            return False
    
    def reset_to_idle(self):
        """Reset to idle state"""
        self.recording_state = "idle"
        self.current_session_id = None
        
        if self.simple_recorder:
            self.simple_recorder.reset_to_idle()
        
        self._notify_state_change("idle")
        logger.info("Reset to idle state")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'mode': 'production' if self.use_production else 'skeleton',
            'initialized': self.is_initialized,
            'recording_state': self.recording_state,
            'current_session': self.current_session_id
        }
        
        if self.use_production:
            # Add production component status
            if self.audio_system:
                status['audio_system'] = self.audio_system.get_status()
            
            if self.device_manager:
                device_status = self.device_manager.get_device_status()
                status['device_status'] = device_status.to_dict()
            
            if self.ai_system:
                status['ai_system'] = self.ai_system.get_status()
        else:
            # Add skeleton component status
            if self.simple_recorder:
                status['recorder'] = self.simple_recorder.get_device_info()
            
            if self.simple_transcriber:
                status['transcriber'] = self.simple_transcriber.get_backend_info()
        
        return status
    
    def shutdown(self):
        """Shutdown all components"""
        logger.info("Shutting down adapter...")
        
        if self.use_production:
            # Shutdown production components
            if self.audio_system:
                if hasattr(self.audio_system, 'stop_pipeline'):
                    self.audio_system.stop_pipeline()
                elif hasattr(self.audio_system, 'stop'):
                    self.audio_system.stop()
            
            if self.session_manager:
                self.session_manager.shutdown()
            
            if self.ai_system:
                self.ai_system.stop()
            
            if self.device_manager:
                self.device_manager.stop()
        
        logger.info("Adapter shutdown complete")
    
    # Callback management
    
    def add_recording_callback(self, callback: Callable):
        """Add recording event callback"""
        self.recording_callbacks.append(callback)
    
    def add_transcription_callback(self, callback: Callable):
        """Add transcription event callback"""
        self.transcription_callbacks.append(callback)
    
    def add_state_callback(self, callback: Callable):
        """Add state change callback"""
        self.state_callbacks.append(callback)
    
    def _notify_state_change(self, new_state: str):
        """Notify state change callbacks"""
        for callback in self.state_callbacks:
            try:
                callback(new_state)
            except Exception as e:
                logger.error(f"Error in state callback: {e}")
    
    def _notify_transcription(self, result):
        """Notify transcription callbacks"""
        for callback in self.transcription_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in transcription callback: {e}")
    
    def _on_audio_data(self, audio_data: np.ndarray):
        """Handle audio data from production pipeline"""
        # Could be used for real-time processing
        pass
    
    def _on_session_state_change(self, session_id: str, state):
        """Handle session state changes from production system"""
        if session_id == self.current_session_id:
            self.recording_state = state.value if hasattr(state, 'value') else str(state)
            self._notify_state_change(self.recording_state)


# Factory function for easy creation
def create_walking_skeleton_adapter(use_production: bool = True, 
                                  config: Dict[str, Any] = None) -> WalkingSkeletonAdapter:
    """
    Create a walking skeleton adapter
    
    Args:
        use_production: Whether to use production components when available
        config: Configuration dictionary
        
    Returns:
        WalkingSkeletonAdapter instance
    """
    return WalkingSkeletonAdapter(use_production, config)


if __name__ == "__main__":
    # Test the adapter
    print("Walking Skeleton Adapter Test")
    print("=" * 50)
    
    # Create adapter
    adapter = create_walking_skeleton_adapter(use_production=True)
    
    # Initialize
    adapter.initialize()
    
    # Get status
    status = adapter.get_system_status()
    print(f"System status: {status}")
    
    # Test recording
    print("\nTesting recording...")
    session_id = adapter.start_recording()
    if session_id:
        print(f"Recording started: {session_id}")
        
        # Record for a few seconds
        time.sleep(3)
        
        # Stop recording
        info = adapter.stop_recording()
        if info:
            print(f"Recording stopped: {info}")
            
            # Transcribe
            print("\nTranscribing...")
            transcript = adapter.transcribe_recording()
            if transcript:
                print(f"Transcript: {transcript[:100]}...")
    
    # Shutdown
    adapter.shutdown()
    print("\nTest complete")