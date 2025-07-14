#!/usr/bin/env python3

"""
Audio Session Manager for The Silent Steno

This module provides comprehensive session lifecycle management for audio
recording sessions. It orchestrates the entire recording process from session
initiation through completion, including state persistence, error recovery,
and integration with other system components.

Key features:
- Session lifecycle management (start, stop, pause, resume)
- State persistence and recovery
- Integration with audio pipeline and storage systems
- Session metadata tracking and management
- Error handling and recovery mechanisms
- Multi-session support and coordination
"""

import os
import json
import time
import uuid
import threading
import logging
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Recording session states"""
    IDLE = "idle"
    STARTING = "starting"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    RECOVERING = "recovering"


class SessionType(Enum):
    """Types of recording sessions"""
    MEETING = "meeting"
    INTERVIEW = "interview"
    LECTURE = "lecture"
    PHONE_CALL = "phone_call"
    CONFERENCE = "conference"
    PERSONAL = "personal"
    OTHER = "other"


@dataclass
class SessionConfig:
    """Configuration for recording sessions"""
    session_type: SessionType = SessionType.MEETING
    max_duration_hours: float = 8.0
    auto_save_interval: int = 300  # seconds
    quality_preset: str = "balanced"  # "low_latency", "balanced", "high_quality"
    enable_preprocessing: bool = True
    enable_real_time_analysis: bool = False
    storage_location: str = "recordings"
    metadata_enabled: bool = True
    auto_cleanup_days: int = 30


@dataclass
class SessionInfo:
    """Complete session information"""
    session_id: str
    session_type: SessionType
    state: SessionState
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    pause_time: Optional[datetime]
    duration_seconds: float
    file_path: Optional[str]
    metadata: Dict[str, Any]
    config: SessionConfig
    created_at: datetime
    updated_at: datetime
    error_info: Optional[str] = None


class SessionManager:
    """
    Audio Session Manager for The Silent Steno
    
    Manages the complete lifecycle of audio recording sessions with
    comprehensive state management, persistence, and error recovery.
    """
    
    def __init__(self, storage_root: str = "recordings", config: Optional[SessionConfig] = None):
        """Initialize session manager"""
        self.storage_root = storage_root
        self.config = config or SessionConfig()
        
        # Session state
        self.active_sessions: Dict[str, SessionInfo] = {}
        self.current_session_id: Optional[str] = None
        
        # Storage paths
        self.sessions_dir = os.path.join(storage_root, "sessions")
        self.metadata_dir = os.path.join(storage_root, "metadata")
        self.state_file = os.path.join(storage_root, "session_state.json")
        
        # Threading
        self.session_lock = threading.RLock()
        self.monitor_thread = None
        self.stop_monitoring = threading.Event()
        
        # Component integration
        self.audio_recorder = None
        self.metadata_tracker = None
        self.storage_monitor = None
        self.file_manager = None
        
        # Callbacks
        self.state_callbacks: List[Callable] = []
        self.session_callbacks: List[Callable] = []
        
        # Initialize storage structure
        self._initialize_storage()
        
        # Load persistent state
        self._load_session_state()
        
        # Start monitoring
        self._start_monitoring()
        
        logger.info(f"Session manager initialized with storage: {storage_root}")
    
    def set_audio_recorder(self, recorder) -> None:
        """Set audio recorder component"""
        self.audio_recorder = recorder
    
    def set_metadata_tracker(self, tracker) -> None:
        """Set metadata tracker component"""
        self.metadata_tracker = tracker
    
    def set_storage_monitor(self, monitor) -> None:
        """Set storage monitor component"""
        self.storage_monitor = monitor
    
    def set_file_manager(self, manager) -> None:
        """Set file manager component"""
        self.file_manager = manager
    
    def add_state_callback(self, callback: Callable[[str, SessionState], None]) -> None:
        """Add callback for session state changes"""
        self.state_callbacks.append(callback)
    
    def add_session_callback(self, callback: Callable[[SessionInfo], None]) -> None:
        """Add callback for session events"""
        self.session_callbacks.append(callback)
    
    def _notify_state_change(self, session_id: str, new_state: SessionState) -> None:
        """Notify callbacks of state changes"""
        for callback in self.state_callbacks:
            try:
                callback(session_id, new_state)
            except Exception as e:
                logger.error(f"Error in state callback: {e}")
    
    def _notify_session_event(self, session_info: SessionInfo) -> None:
        """Notify callbacks of session events"""
        for callback in self.session_callbacks:
            try:
                callback(session_info)
            except Exception as e:
                logger.error(f"Error in session callback: {e}")
    
    def _initialize_storage(self) -> None:
        """Initialize storage directory structure"""
        try:
            os.makedirs(self.sessions_dir, exist_ok=True)
            os.makedirs(self.metadata_dir, exist_ok=True)
            os.makedirs(self.storage_root, exist_ok=True)
            
            logger.info("Storage directory structure initialized")
        except Exception as e:
            logger.error(f"Error initializing storage: {e}")
    
    def _load_session_state(self) -> None:
        """Load persistent session state from disk"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                
                # Reconstruct active sessions
                for session_data in state_data.get('active_sessions', []):
                    session_info = self._deserialize_session_info(session_data)
                    self.active_sessions[session_info.session_id] = session_info
                
                self.current_session_id = state_data.get('current_session_id')
                
                logger.info(f"Loaded {len(self.active_sessions)} sessions from persistent state")
            
        except Exception as e:
            logger.error(f"Error loading session state: {e}")
    
    def _save_session_state(self) -> None:
        """Save current session state to disk"""
        try:
            with self.session_lock:
                state_data = {
                    'active_sessions': [
                        self._serialize_session_info(session) 
                        for session in self.active_sessions.values()
                    ],
                    'current_session_id': self.current_session_id,
                    'last_saved': time.time()
                }
                
                with open(self.state_file, 'w') as f:
                    json.dump(state_data, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error saving session state: {e}")
    
    def _serialize_session_info(self, session_info: SessionInfo) -> Dict[str, Any]:
        """Serialize session info for storage"""
        data = asdict(session_info)
        # Convert enums to strings
        data['session_type'] = session_info.session_type.value
        data['state'] = session_info.state.value
        data['config'] = asdict(session_info.config)
        data['config']['session_type'] = session_info.config.session_type.value
        return data
    
    def _deserialize_session_info(self, data: Dict[str, Any]) -> SessionInfo:
        """Deserialize session info from storage"""
        # Convert string enums back to enum objects
        data['session_type'] = SessionType(data['session_type'])
        data['state'] = SessionState(data['state'])
        
        # Handle datetime fields
        for field in ['start_time', 'end_time', 'pause_time', 'created_at', 'updated_at']:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        
        # Reconstruct config
        config_data = data.pop('config')
        config_data['session_type'] = SessionType(config_data['session_type'])
        data['config'] = SessionConfig(**config_data)
        
        return SessionInfo(**data)
    
    def start_session(self, session_type: SessionType = SessionType.MEETING,
                     config: Optional[SessionConfig] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Start a new recording session
        
        Args:
            session_type: Type of session to create
            config: Session configuration (uses default if None)
            metadata: Initial session metadata
            
        Returns:
            Session ID if successful, None if failed
        """
        try:
            with self.session_lock:
                # Check if we can start a new session
                if self.current_session_id:
                    current_session = self.active_sessions.get(self.current_session_id)
                    if current_session and current_session.state in [SessionState.RECORDING, SessionState.PAUSED]:
                        logger.warning("Cannot start session: another session is active")
                        return None
                
                # Check storage space
                if self.storage_monitor:
                    storage_status = self.storage_monitor.get_storage_status()
                    if storage_status.get('available_gb', 0) < 1.0:
                        logger.error("Cannot start session: insufficient storage space")
                        return None
                
                # Create new session
                session_id = str(uuid.uuid4())
                session_config = config or self.config
                
                # Generate file path
                file_path = None
                if self.file_manager:
                    file_path = self.file_manager.generate_filename(
                        session_id=session_id,
                        session_type=session_type,
                        format="flac"
                    )
                
                # Create session info
                session_info = SessionInfo(
                    session_id=session_id,
                    session_type=session_type,
                    state=SessionState.STARTING,
                    start_time=None,
                    end_time=None,
                    pause_time=None,
                    duration_seconds=0.0,
                    file_path=file_path,
                    metadata=metadata or {},
                    config=session_config,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                # Store session
                self.active_sessions[session_id] = session_info
                self.current_session_id = session_id
                
                # Start metadata tracking
                if self.metadata_tracker:
                    self.metadata_tracker.track_session(session_info)
                
                # Start audio recording
                if self.audio_recorder:
                    recording_config = {
                        'file_path': file_path,
                        'format': 'flac',
                        'quality': session_config.quality_preset,
                        'preprocessing': session_config.enable_preprocessing
                    }
                    
                    if self.audio_recorder.start_recording(session_id, recording_config):
                        # Update session state
                        session_info.state = SessionState.RECORDING
                        session_info.start_time = datetime.now()
                        session_info.updated_at = datetime.now()
                        
                        # Notify callbacks
                        self._notify_state_change(session_id, SessionState.RECORDING)
                        self._notify_session_event(session_info)
                        
                        # Save state
                        self._save_session_state()
                        
                        logger.info(f"Session {session_id} started successfully")
                        return session_id
                    else:
                        # Recording failed, cleanup
                        session_info.state = SessionState.ERROR
                        session_info.error_info = "Failed to start audio recording"
                        logger.error(f"Failed to start recording for session {session_id}")
                        return None
                else:
                    # No audio recorder available, simulate success for testing
                    session_info.state = SessionState.RECORDING
                    session_info.start_time = datetime.now()
                    session_info.updated_at = datetime.now()
                    
                    self._notify_state_change(session_id, SessionState.RECORDING)
                    self._notify_session_event(session_info)
                    self._save_session_state()
                    
                    logger.info(f"Session {session_id} started (no recorder)")
                    return session_id
        
        except Exception as e:
            logger.error(f"Error starting session: {e}")
            return None
    
    def stop_session(self, session_id: Optional[str] = None) -> bool:
        """
        Stop a recording session
        
        Args:
            session_id: Session to stop (current session if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session_lock:
                # Determine session to stop
                target_session_id = session_id or self.current_session_id
                if not target_session_id:
                    logger.warning("No session to stop")
                    return False
                
                session_info = self.active_sessions.get(target_session_id)
                if not session_info:
                    logger.warning(f"Session {target_session_id} not found")
                    return False
                
                if session_info.state not in [SessionState.RECORDING, SessionState.PAUSED]:
                    logger.warning(f"Session {target_session_id} is not active")
                    return False
                
                # Update session state
                session_info.state = SessionState.STOPPING
                session_info.updated_at = datetime.now()
                self._notify_state_change(target_session_id, SessionState.STOPPING)
                
                # Stop audio recording
                if self.audio_recorder:
                    recording_info = self.audio_recorder.stop_recording(target_session_id)
                    if recording_info:
                        session_info.file_path = recording_info.get('file_path', session_info.file_path)
                        session_info.metadata.update(recording_info.get('metadata', {}))
                
                # Calculate final duration
                if session_info.start_time:
                    if session_info.pause_time and session_info.state == SessionState.PAUSED:
                        # Was paused, calculate up to pause time
                        session_info.duration_seconds = (session_info.pause_time - session_info.start_time).total_seconds()
                    else:
                        # Calculate total duration
                        session_info.duration_seconds = (datetime.now() - session_info.start_time).total_seconds()
                
                # Finalize session
                session_info.state = SessionState.STOPPED
                session_info.end_time = datetime.now()
                session_info.updated_at = datetime.now()
                
                # Update metadata
                if self.metadata_tracker:
                    self.metadata_tracker.update_metadata(target_session_id, {
                        'end_time': session_info.end_time,
                        'duration_seconds': session_info.duration_seconds,
                        'final_state': 'completed'
                    })
                
                # Clear current session if this was it
                if self.current_session_id == target_session_id:
                    self.current_session_id = None
                
                # Notify callbacks
                self._notify_state_change(target_session_id, SessionState.STOPPED)
                self._notify_session_event(session_info)
                
                # Save state
                self._save_session_state()
                
                logger.info(f"Session {target_session_id} stopped successfully")
                return True
        
        except Exception as e:
            logger.error(f"Error stopping session: {e}")
            return False
    
    def pause_session(self, session_id: Optional[str] = None) -> bool:
        """
        Pause a recording session
        
        Args:
            session_id: Session to pause (current session if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session_lock:
                target_session_id = session_id or self.current_session_id
                if not target_session_id:
                    return False
                
                session_info = self.active_sessions.get(target_session_id)
                if not session_info or session_info.state != SessionState.RECORDING:
                    return False
                
                # Pause audio recording
                if self.audio_recorder:
                    if not self.audio_recorder.pause_recording(target_session_id):
                        return False
                
                # Update session state
                session_info.state = SessionState.PAUSED
                session_info.pause_time = datetime.now()
                session_info.updated_at = datetime.now()
                
                # Calculate duration up to pause
                if session_info.start_time:
                    session_info.duration_seconds = (session_info.pause_time - session_info.start_time).total_seconds()
                
                # Notify callbacks
                self._notify_state_change(target_session_id, SessionState.PAUSED)
                self._notify_session_event(session_info)
                
                # Save state
                self._save_session_state()
                
                logger.info(f"Session {target_session_id} paused")
                return True
        
        except Exception as e:
            logger.error(f"Error pausing session: {e}")
            return False
    
    def resume_session(self, session_id: Optional[str] = None) -> bool:
        """
        Resume a paused session
        
        Args:
            session_id: Session to resume (current session if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session_lock:
                target_session_id = session_id or self.current_session_id
                if not target_session_id:
                    return False
                
                session_info = self.active_sessions.get(target_session_id)
                if not session_info or session_info.state != SessionState.PAUSED:
                    return False
                
                # Resume audio recording
                if self.audio_recorder:
                    if not self.audio_recorder.resume_recording(target_session_id):
                        return False
                
                # Update session state
                session_info.state = SessionState.RECORDING
                # Adjust start time to account for pause duration
                if session_info.pause_time:
                    pause_duration = datetime.now() - session_info.pause_time
                    session_info.start_time = session_info.start_time + pause_duration
                    session_info.pause_time = None
                
                session_info.updated_at = datetime.now()
                
                # Notify callbacks
                self._notify_state_change(target_session_id, SessionState.RECORDING)
                self._notify_session_event(session_info)
                
                # Save state
                self._save_session_state()
                
                logger.info(f"Session {target_session_id} resumed")
                return True
        
        except Exception as e:
            logger.error(f"Error resuming session: {e}")
            return False
    
    def get_session_status(self, session_id: Optional[str] = None) -> Optional[SessionInfo]:
        """
        Get status of a session
        
        Args:
            session_id: Session ID (current session if None)
            
        Returns:
            SessionInfo if found, None otherwise
        """
        target_session_id = session_id or self.current_session_id
        if not target_session_id:
            return None
        
        return self.active_sessions.get(target_session_id)
    
    def get_all_sessions(self) -> List[SessionInfo]:
        """Get list of all active sessions"""
        with self.session_lock:
            return list(self.active_sessions.values())
    
    def get_current_session(self) -> Optional[SessionInfo]:
        """Get current active session"""
        if self.current_session_id:
            return self.active_sessions.get(self.current_session_id)
        return None
    
    def cleanup_old_sessions(self, days: int = None) -> int:
        """
        Clean up old completed sessions
        
        Args:
            days: Remove sessions older than this many days
            
        Returns:
            Number of sessions cleaned up
        """
        try:
            cleanup_days = days or self.config.auto_cleanup_days
            cutoff_time = datetime.now() - timedelta(days=cleanup_days)
            
            cleaned_count = 0
            sessions_to_remove = []
            
            with self.session_lock:
                for session_id, session_info in self.active_sessions.items():
                    if (session_info.state == SessionState.STOPPED and 
                        session_info.end_time and 
                        session_info.end_time < cutoff_time):
                        sessions_to_remove.append(session_id)
                
                for session_id in sessions_to_remove:
                    del self.active_sessions[session_id]
                    cleaned_count += 1
                
                if cleaned_count > 0:
                    self._save_session_state()
            
            logger.info(f"Cleaned up {cleaned_count} old sessions")
            return cleaned_count
        
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")
            return 0
    
    def _start_monitoring(self) -> None:
        """Start session monitoring thread"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return
        
        self.stop_monitoring.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="SessionMonitoring"
        )
        self.monitor_thread.start()
        
        logger.info("Session monitoring started")
    
    def _monitoring_loop(self) -> None:
        """Session monitoring loop"""
        try:
            while not self.stop_monitoring.is_set():
                try:
                    # Check session health
                    self._check_session_health()
                    
                    # Auto-save state periodically
                    self._save_session_state()
                    
                    # Cleanup old sessions
                    if self.config.auto_cleanup_days > 0:
                        self.cleanup_old_sessions()
                    
                    # Wait for next check
                    self.stop_monitoring.wait(self.config.auto_save_interval)
                
                except Exception as e:
                    logger.error(f"Error in session monitoring: {e}")
                    self.stop_monitoring.wait(30)  # Wait before retry
            
        except Exception as e:
            logger.error(f"Fatal error in session monitoring: {e}")
    
    def _check_session_health(self) -> None:
        """Check health of active sessions"""
        try:
            with self.session_lock:
                for session_id, session_info in self.active_sessions.items():
                    if session_info.state == SessionState.RECORDING:
                        # Check for max duration exceeded
                        if (session_info.start_time and 
                            (datetime.now() - session_info.start_time).total_seconds() > 
                            session_info.config.max_duration_hours * 3600):
                            
                            logger.warning(f"Session {session_id} exceeded max duration, stopping")
                            self.stop_session(session_id)
                        
                        # Update duration
                        if session_info.start_time:
                            session_info.duration_seconds = (datetime.now() - session_info.start_time).total_seconds()
                            session_info.updated_at = datetime.now()
        
        except Exception as e:
            logger.error(f"Error checking session health: {e}")
    
    def shutdown(self) -> None:
        """Shutdown session manager"""
        try:
            logger.info("Shutting down session manager...")
            
            # Stop monitoring
            self.stop_monitoring.set()
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5.0)
            
            # Stop any active sessions
            with self.session_lock:
                active_sessions = [
                    session_id for session_id, session_info in self.active_sessions.items()
                    if session_info.state in [SessionState.RECORDING, SessionState.PAUSED]
                ]
                
                for session_id in active_sessions:
                    self.stop_session(session_id)
            
            # Save final state
            self._save_session_state()
            
            logger.info("Session manager shutdown complete")
        
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Convenience functions
def start_session(session_type: SessionType = SessionType.MEETING,
                 config: Optional[SessionConfig] = None) -> Optional[str]:
    """Start a new recording session - convenience function"""
    global _session_manager_instance
    if _session_manager_instance is None:
        _session_manager_instance = SessionManager()
    return _session_manager_instance.start_session(session_type, config)


def stop_session(session_id: Optional[str] = None) -> bool:
    """Stop recording session - convenience function"""
    global _session_manager_instance
    if _session_manager_instance:
        return _session_manager_instance.stop_session(session_id)
    return False


def get_session_status(session_id: Optional[str] = None) -> Optional[SessionInfo]:
    """Get session status - convenience function"""
    global _session_manager_instance
    if _session_manager_instance:
        return _session_manager_instance.get_session_status(session_id)
    return None


def get_session_manager() -> Optional[SessionManager]:
    """Get global session manager instance"""
    return _session_manager_instance


# Global session manager instance
_session_manager_instance = None


if __name__ == "__main__":
    # Basic test when run directly
    print("Session Manager Test")
    print("=" * 50)
    
    # Create session manager
    manager = SessionManager("test_recordings")
    
    def on_state_change(session_id, state):
        print(f"Session {session_id[:8]} state: {state.value}")
    
    def on_session_event(session_info):
        print(f"Session event: {session_info.session_id[:8]} - {session_info.state.value}")
    
    manager.add_state_callback(on_state_change)
    manager.add_session_callback(on_session_event)
    
    print("Starting test session...")
    session_id = manager.start_session(SessionType.MEETING)
    
    if session_id:
        print(f"Session started: {session_id}")
        
        # Wait a bit
        import time
        time.sleep(2)
        
        # Check status
        status = manager.get_session_status(session_id)
        if status:
            print(f"Session status: {status.state.value}, duration: {status.duration_seconds:.1f}s")
        
        # Pause session
        print("Pausing session...")
        manager.pause_session(session_id)
        time.sleep(1)
        
        # Resume session
        print("Resuming session...")
        manager.resume_session(session_id)
        time.sleep(1)
        
        # Stop session
        print("Stopping session...")
        manager.stop_session(session_id)
        
        # Final status
        final_status = manager.get_session_status(session_id)
        if final_status:
            print(f"Final status: {final_status.state.value}, total duration: {final_status.duration_seconds:.1f}s")
    
    # Cleanup
    manager.shutdown()
    print("Test complete")