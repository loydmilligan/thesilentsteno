#!/usr/bin/env python3

"""
Session Metadata Tracker for The Silent Steno

This module provides comprehensive metadata collection and tracking for
recording sessions. It captures detailed information about sessions,
participants, audio quality, system performance, and contextual data
to support AI processing and user insights.

Key features:
- Comprehensive session metadata collection
- Real-time participant tracking and analysis
- Audio quality metrics and monitoring
- System performance tracking
- Contextual metadata extraction
- Metadata persistence and retrieval
- Export capabilities for various formats
"""

import json
import time
import uuid
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataCategory(Enum):
    """Metadata categories"""
    SESSION = "session"
    PARTICIPANT = "participant"
    AUDIO_QUALITY = "audio_quality"
    SYSTEM_PERFORMANCE = "system_performance"
    CONTEXT = "context"
    USER_DEFINED = "user_defined"


@dataclass
class ParticipantInfo:
    """Information about session participants"""
    participant_id: str
    name: Optional[str]
    role: Optional[str]
    speaking_time_seconds: float
    word_count: int
    avg_volume_db: float
    speech_rate_wpm: float
    first_speech_time: Optional[datetime]
    last_speech_time: Optional[datetime]
    interaction_count: int
    metadata: Dict[str, Any]


@dataclass
class AudioQualityMetrics:
    """Comprehensive audio quality metrics"""
    overall_quality_score: float  # 0-1
    signal_to_noise_ratio_db: float
    dynamic_range_db: float
    frequency_response_score: float
    distortion_level: float
    clipping_events: int
    dropout_events: int
    background_noise_level_db: float
    speech_clarity_score: float
    recording_consistency: float


@dataclass
class SystemPerformanceMetrics:
    """System performance during recording"""
    cpu_usage_avg: float
    memory_usage_avg: float
    disk_io_rate: float
    audio_latency_ms: float
    buffer_underruns: int
    processing_lag_ms: float
    system_temperature: Optional[float]
    network_activity: Optional[float]


@dataclass
class ContextualMetadata:
    """Contextual information about the session"""
    location: Optional[str]
    device_info: Dict[str, Any]
    environment_type: Optional[str]  # office, home, outdoor, etc.
    background_activity: Optional[str]
    weather_conditions: Optional[str]
    session_purpose: Optional[str]
    expected_duration: Optional[int]
    importance_level: Optional[str]
    tags: List[str]


@dataclass
class SessionMetadata:
    """Complete session metadata"""
    session_id: str
    session_type: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: float
    title: Optional[str]
    description: Optional[str]
    participants: List[ParticipantInfo]
    audio_quality: AudioQualityMetrics
    system_performance: SystemPerformanceMetrics
    contextual: ContextualMetadata
    user_defined: Dict[str, Any]
    recording_file_path: Optional[str]
    transcript_file_path: Optional[str]
    analysis_file_path: Optional[str]
    created_at: datetime
    updated_at: datetime
    version: str = "1.0"


class MetadataTracker:
    """
    Session Metadata Tracker for The Silent Steno
    
    Provides comprehensive metadata collection, tracking, and management
    for recording sessions with real-time updates and analysis.
    """
    
    def __init__(self, storage_root: str = "recordings"):
        """Initialize metadata tracker"""
        self.storage_root = storage_root
        self.metadata_dir = os.path.join(storage_root, "metadata")
        
        # Session tracking
        self.active_sessions: Dict[str, SessionMetadata] = {}
        self.metadata_lock = threading.RLock()
        
        # Real-time tracking
        self.participant_tracker = {}
        self.quality_tracker = {}
        self.performance_tracker = {}
        
        # Components
        self.level_monitor = None
        self.performance_monitor = None
        
        # Statistics
        self.tracking_stats = {
            "sessions_tracked": 0,
            "metadata_updates": 0,
            "participant_analyses": 0,
            "quality_assessments": 0,
            "exports_generated": 0
        }
        
        # Initialize storage
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        logger.info(f"Metadata tracker initialized with storage: {self.metadata_dir}")
    
    def set_level_monitor(self, monitor) -> None:
        """Set audio level monitor for quality tracking"""
        self.level_monitor = monitor
    
    def set_performance_monitor(self, monitor) -> None:
        """Set system performance monitor"""
        self.performance_monitor = monitor
    
    def track_session(self, session_info) -> bool:
        """
        Start tracking metadata for a session
        
        Args:
            session_info: Session information object
            
        Returns:
            True if tracking started successfully
        """
        try:
            with self.metadata_lock:
                session_id = session_info.session_id
                
                if session_id in self.active_sessions:
                    logger.warning(f"Session {session_id} already being tracked")
                    return False
                
                # Initialize metadata structure
                metadata = SessionMetadata(
                    session_id=session_id,
                    session_type=session_info.session_type.value,
                    start_time=session_info.start_time or datetime.now(),
                    end_time=None,
                    duration_seconds=0.0,
                    title=session_info.metadata.get('title'),
                    description=session_info.metadata.get('description'),
                    participants=[],
                    audio_quality=self._initialize_audio_quality_metrics(),
                    system_performance=self._initialize_performance_metrics(),
                    contextual=self._initialize_contextual_metadata(session_info.metadata),
                    user_defined=session_info.metadata.copy(),
                    recording_file_path=session_info.file_path,
                    transcript_file_path=None,
                    analysis_file_path=None,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                # Store metadata
                self.active_sessions[session_id] = metadata
                
                # Initialize tracking components
                self.participant_tracker[session_id] = {}
                self.quality_tracker[session_id] = {
                    'samples': [],
                    'quality_events': [],
                    'last_update': time.time()
                }
                self.performance_tracker[session_id] = {
                    'samples': [],
                    'alerts': [],
                    'last_update': time.time()
                }
                
                # Save initial metadata
                self._save_metadata(session_id)
                
                self.tracking_stats["sessions_tracked"] += 1
                
                logger.info(f"Started tracking metadata for session {session_id}")
                return True
        
        except Exception as e:
            logger.error(f"Error starting session tracking: {e}")
            return False
    
    def update_metadata(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update session metadata
        
        Args:
            session_id: Session identifier
            updates: Dictionary of metadata updates
            
        Returns:
            True if update successful
        """
        try:
            with self.metadata_lock:
                metadata = self.active_sessions.get(session_id)
                if not metadata:
                    logger.warning(f"No active tracking for session {session_id}")
                    return False
                
                # Apply updates
                for key, value in updates.items():
                    if key == 'end_time':
                        metadata.end_time = value
                    elif key == 'duration_seconds':
                        metadata.duration_seconds = value
                    elif key == 'title':
                        metadata.title = value
                    elif key == 'description':
                        metadata.description = value
                    elif key == 'recording_file_path':
                        metadata.recording_file_path = value
                    elif key == 'transcript_file_path':
                        metadata.transcript_file_path = value
                    elif key == 'analysis_file_path':
                        metadata.analysis_file_path = value
                    elif key.startswith('user_defined.'):
                        # Nested user-defined metadata
                        nested_key = key[13:]  # Remove 'user_defined.' prefix
                        metadata.user_defined[nested_key] = value
                    else:
                        # Add to user-defined metadata
                        metadata.user_defined[key] = value
                
                # Update timestamp
                metadata.updated_at = datetime.now()
                
                # Save updated metadata
                self._save_metadata(session_id)
                
                self.tracking_stats["metadata_updates"] += 1
                
                logger.debug(f"Updated metadata for session {session_id}")
                return True
        
        except Exception as e:
            logger.error(f"Error updating metadata for session {session_id}: {e}")
            return False
    
    def track_participant_activity(self, session_id: str, participant_data: Dict[str, Any]) -> None:
        """
        Track participant activity and analysis
        
        Args:
            session_id: Session identifier
            participant_data: Participant activity data
        """
        try:
            with self.metadata_lock:
                if session_id not in self.active_sessions:
                    return
                
                participant_id = participant_data.get('participant_id', 'unknown')
                participants = self.participant_tracker.get(session_id, {})
                
                if participant_id not in participants:
                    # Initialize new participant
                    participants[participant_id] = ParticipantInfo(
                        participant_id=participant_id,
                        name=participant_data.get('name'),
                        role=participant_data.get('role'),
                        speaking_time_seconds=0.0,
                        word_count=0,
                        avg_volume_db=-60.0,
                        speech_rate_wpm=0.0,
                        first_speech_time=None,
                        last_speech_time=None,
                        interaction_count=0,
                        metadata={}
                    )
                
                participant = participants[participant_id]
                
                # Update participant data
                if 'speaking_duration' in participant_data:
                    participant.speaking_time_seconds += participant_data['speaking_duration']
                
                if 'word_count' in participant_data:
                    participant.word_count += participant_data['word_count']
                
                if 'volume_db' in participant_data:
                    # Update average volume (simple moving average)
                    if participant.avg_volume_db == -60.0:  # First measurement
                        participant.avg_volume_db = participant_data['volume_db']
                    else:
                        participant.avg_volume_db = (participant.avg_volume_db + participant_data['volume_db']) / 2
                
                if 'speech_rate' in participant_data:
                    participant.speech_rate_wpm = participant_data['speech_rate']
                
                current_time = datetime.now()
                if participant.first_speech_time is None:
                    participant.first_speech_time = current_time
                participant.last_speech_time = current_time
                
                if 'interaction' in participant_data:
                    participant.interaction_count += 1
                
                # Update participant metadata
                participant.metadata.update(participant_data.get('metadata', {}))
                
                # Update session metadata
                metadata = self.active_sessions[session_id]
                metadata.participants = list(participants.values())
                metadata.updated_at = datetime.now()
                
                self.participant_tracker[session_id] = participants
                self.tracking_stats["participant_analyses"] += 1
                
                logger.debug(f"Updated participant {participant_id} for session {session_id}")
        
        except Exception as e:
            logger.error(f"Error tracking participant activity: {e}")
    
    def track_audio_quality(self, session_id: str, quality_data: Dict[str, Any]) -> None:
        """
        Track audio quality metrics
        
        Args:
            session_id: Session identifier
            quality_data: Audio quality measurements
        """
        try:
            with self.metadata_lock:
                if session_id not in self.active_sessions:
                    return
                
                quality_tracker = self.quality_tracker.get(session_id, {})
                quality_tracker['samples'].append({
                    'timestamp': time.time(),
                    'data': quality_data
                })
                
                # Update aggregate quality metrics
                metadata = self.active_sessions[session_id]
                quality_metrics = metadata.audio_quality
                
                # Update metrics based on new data
                if 'snr_db' in quality_data:
                    quality_metrics.signal_to_noise_ratio_db = quality_data['snr_db']
                
                if 'dynamic_range_db' in quality_data:
                    quality_metrics.dynamic_range_db = quality_data['dynamic_range_db']
                
                if 'distortion' in quality_data:
                    quality_metrics.distortion_level = quality_data['distortion']
                
                if 'clipping_detected' in quality_data and quality_data['clipping_detected']:
                    quality_metrics.clipping_events += 1
                
                if 'dropout_detected' in quality_data and quality_data['dropout_detected']:
                    quality_metrics.dropout_events += 1
                
                if 'noise_level_db' in quality_data:
                    quality_metrics.background_noise_level_db = quality_data['noise_level_db']
                
                if 'speech_clarity' in quality_data:
                    quality_metrics.speech_clarity_score = quality_data['speech_clarity']
                
                # Calculate overall quality score
                quality_metrics.overall_quality_score = self._calculate_overall_quality_score(quality_metrics)
                
                metadata.updated_at = datetime.now()
                self.quality_tracker[session_id] = quality_tracker
                self.tracking_stats["quality_assessments"] += 1
                
                logger.debug(f"Updated audio quality for session {session_id}")
        
        except Exception as e:
            logger.error(f"Error tracking audio quality: {e}")
    
    def track_system_performance(self, session_id: str, performance_data: Dict[str, Any]) -> None:
        """
        Track system performance metrics
        
        Args:
            session_id: Session identifier
            performance_data: System performance measurements
        """
        try:
            with self.metadata_lock:
                if session_id not in self.active_sessions:
                    return
                
                perf_tracker = self.performance_tracker.get(session_id, {})
                perf_tracker['samples'].append({
                    'timestamp': time.time(),
                    'data': performance_data
                })
                
                # Update aggregate performance metrics
                metadata = self.active_sessions[session_id]
                perf_metrics = metadata.system_performance
                
                # Calculate running averages
                recent_samples = perf_tracker['samples'][-10:]  # Last 10 samples
                
                if recent_samples:
                    cpu_values = [s['data'].get('cpu_usage', 0) for s in recent_samples]
                    memory_values = [s['data'].get('memory_usage', 0) for s in recent_samples]
                    
                    perf_metrics.cpu_usage_avg = sum(cpu_values) / len(cpu_values)
                    perf_metrics.memory_usage_avg = sum(memory_values) / len(memory_values)
                
                # Update specific metrics
                if 'disk_io_rate' in performance_data:
                    perf_metrics.disk_io_rate = performance_data['disk_io_rate']
                
                if 'audio_latency_ms' in performance_data:
                    perf_metrics.audio_latency_ms = performance_data['audio_latency_ms']
                
                if 'buffer_underruns' in performance_data:
                    perf_metrics.buffer_underruns = performance_data['buffer_underruns']
                
                if 'processing_lag_ms' in performance_data:
                    perf_metrics.processing_lag_ms = performance_data['processing_lag_ms']
                
                if 'temperature' in performance_data:
                    perf_metrics.system_temperature = performance_data['temperature']
                
                metadata.updated_at = datetime.now()
                self.performance_tracker[session_id] = perf_tracker
                
                logger.debug(f"Updated system performance for session {session_id}")
        
        except Exception as e:
            logger.error(f"Error tracking system performance: {e}")
    
    def get_session_metadata(self, session_id: str) -> Optional[SessionMetadata]:
        """
        Get complete session metadata
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionMetadata if found, None otherwise
        """
        with self.metadata_lock:
            return self.active_sessions.get(session_id)
    
    def export_metadata(self, session_id: str, format: str = "json") -> Optional[str]:
        """
        Export session metadata to file
        
        Args:
            session_id: Session identifier
            format: Export format (json, csv, xml)
            
        Returns:
            Path to exported file if successful, None otherwise
        """
        try:
            with self.metadata_lock:
                metadata = self.active_sessions.get(session_id)
                if not metadata:
                    logger.warning(f"No metadata found for session {session_id}")
                    return None
                
                # Generate export filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"metadata_{session_id[:8]}_{timestamp}.{format}"
                export_path = os.path.join(self.metadata_dir, filename)
                
                if format.lower() == "json":
                    self._export_json(metadata, export_path)
                elif format.lower() == "csv":
                    self._export_csv(metadata, export_path)
                elif format.lower() == "xml":
                    self._export_xml(metadata, export_path)
                else:
                    logger.error(f"Unsupported export format: {format}")
                    return None
                
                self.tracking_stats["exports_generated"] += 1
                
                logger.info(f"Exported metadata for session {session_id} to {export_path}")
                return export_path
        
        except Exception as e:
            logger.error(f"Error exporting metadata: {e}")
            return None
    
    def _initialize_audio_quality_metrics(self) -> AudioQualityMetrics:
        """Initialize audio quality metrics with defaults"""
        return AudioQualityMetrics(
            overall_quality_score=0.0,
            signal_to_noise_ratio_db=0.0,
            dynamic_range_db=0.0,
            frequency_response_score=0.0,
            distortion_level=0.0,
            clipping_events=0,
            dropout_events=0,
            background_noise_level_db=-60.0,
            speech_clarity_score=0.0,
            recording_consistency=0.0
        )
    
    def _initialize_performance_metrics(self) -> SystemPerformanceMetrics:
        """Initialize system performance metrics with defaults"""
        return SystemPerformanceMetrics(
            cpu_usage_avg=0.0,
            memory_usage_avg=0.0,
            disk_io_rate=0.0,
            audio_latency_ms=0.0,
            buffer_underruns=0,
            processing_lag_ms=0.0,
            system_temperature=None,
            network_activity=None
        )
    
    def _initialize_contextual_metadata(self, session_metadata: Dict[str, Any]) -> ContextualMetadata:
        """Initialize contextual metadata from session data"""
        return ContextualMetadata(
            location=session_metadata.get('location'),
            device_info=session_metadata.get('device_info', {}),
            environment_type=session_metadata.get('environment_type'),
            background_activity=session_metadata.get('background_activity'),
            weather_conditions=session_metadata.get('weather_conditions'),
            session_purpose=session_metadata.get('session_purpose'),
            expected_duration=session_metadata.get('expected_duration'),
            importance_level=session_metadata.get('importance_level'),
            tags=session_metadata.get('tags', [])
        )
    
    def _calculate_overall_quality_score(self, quality_metrics: AudioQualityMetrics) -> float:
        """Calculate overall quality score from individual metrics"""
        try:
            # Normalize individual scores to 0-1 range
            snr_score = max(min((quality_metrics.signal_to_noise_ratio_db + 20) / 40, 1.0), 0.0)
            dr_score = max(min(quality_metrics.dynamic_range_db / 30, 1.0), 0.0)
            distortion_score = max(1.0 - quality_metrics.distortion_level, 0.0)
            clarity_score = quality_metrics.speech_clarity_score
            
            # Event-based penalties
            clipping_penalty = min(quality_metrics.clipping_events * 0.1, 0.5)
            dropout_penalty = min(quality_metrics.dropout_events * 0.1, 0.5)
            
            # Weighted combination
            base_score = (
                snr_score * 0.3 +
                dr_score * 0.2 +
                distortion_score * 0.2 +
                clarity_score * 0.3
            )
            
            # Apply penalties
            overall_score = max(base_score - clipping_penalty - dropout_penalty, 0.0)
            
            return min(overall_score, 1.0)
        
        except Exception:
            return 0.5  # Default neutral score
    
    def _save_metadata(self, session_id: str) -> None:
        """Save metadata to persistent storage"""
        try:
            metadata = self.active_sessions.get(session_id)
            if not metadata:
                return
            
            filename = f"session_{session_id}.json"
            file_path = os.path.join(self.metadata_dir, filename)
            
            # Convert to dictionary for JSON serialization
            metadata_dict = self._serialize_metadata(metadata)
            
            with open(file_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
            
            logger.debug(f"Saved metadata for session {session_id}")
        
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def _serialize_metadata(self, metadata: SessionMetadata) -> Dict[str, Any]:
        """Serialize metadata for JSON storage"""
        try:
            data = asdict(metadata)
            
            # Convert datetime objects to ISO strings
            for key in ['start_time', 'end_time', 'created_at', 'updated_at']:
                if data.get(key):
                    data[key] = data[key].isoformat()
            
            # Handle nested datetime objects in participants
            for participant in data.get('participants', []):
                for time_key in ['first_speech_time', 'last_speech_time']:
                    if participant.get(time_key):
                        participant[time_key] = participant[time_key].isoformat()
            
            return data
        
        except Exception as e:
            logger.error(f"Error serializing metadata: {e}")
            return {}
    
    def _export_json(self, metadata: SessionMetadata, export_path: str) -> None:
        """Export metadata as JSON"""
        metadata_dict = self._serialize_metadata(metadata)
        with open(export_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def _export_csv(self, metadata: SessionMetadata, export_path: str) -> None:
        """Export metadata as CSV"""
        import csv
        
        with open(export_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write session summary
            writer.writerow(['Session Summary'])
            writer.writerow(['Session ID', metadata.session_id])
            writer.writerow(['Type', metadata.session_type])
            writer.writerow(['Start Time', metadata.start_time.isoformat()])
            writer.writerow(['Duration (seconds)', metadata.duration_seconds])
            writer.writerow(['Title', metadata.title or ''])
            writer.writerow([])
            
            # Write participants
            writer.writerow(['Participants'])
            writer.writerow(['ID', 'Name', 'Speaking Time', 'Word Count', 'Avg Volume'])
            for participant in metadata.participants:
                writer.writerow([
                    participant.participant_id,
                    participant.name or '',
                    participant.speaking_time_seconds,
                    participant.word_count,
                    participant.avg_volume_db
                ])
            writer.writerow([])
            
            # Write quality metrics
            writer.writerow(['Audio Quality'])
            quality = metadata.audio_quality
            writer.writerow(['Overall Score', quality.overall_quality_score])
            writer.writerow(['SNR (dB)', quality.signal_to_noise_ratio_db])
            writer.writerow(['Dynamic Range (dB)', quality.dynamic_range_db])
            writer.writerow(['Clipping Events', quality.clipping_events])
    
    def _export_xml(self, metadata: SessionMetadata, export_path: str) -> None:
        """Export metadata as XML"""
        try:
            from xml.etree.ElementTree import Element, SubElement, tostring
            from xml.dom import minidom
            
            root = Element('session_metadata')
            root.set('version', metadata.version)
            root.set('session_id', metadata.session_id)
            
            # Session info
            session_elem = SubElement(root, 'session')
            SubElement(session_elem, 'type').text = metadata.session_type
            SubElement(session_elem, 'start_time').text = metadata.start_time.isoformat()
            SubElement(session_elem, 'duration_seconds').text = str(metadata.duration_seconds)
            SubElement(session_elem, 'title').text = metadata.title or ''
            
            # Participants
            participants_elem = SubElement(root, 'participants')
            for participant in metadata.participants:
                p_elem = SubElement(participants_elem, 'participant')
                p_elem.set('id', participant.participant_id)
                SubElement(p_elem, 'name').text = participant.name or ''
                SubElement(p_elem, 'speaking_time').text = str(participant.speaking_time_seconds)
                SubElement(p_elem, 'word_count').text = str(participant.word_count)
            
            # Quality metrics
            quality_elem = SubElement(root, 'audio_quality')
            quality = metadata.audio_quality
            SubElement(quality_elem, 'overall_score').text = str(quality.overall_quality_score)
            SubElement(quality_elem, 'snr_db').text = str(quality.signal_to_noise_ratio_db)
            SubElement(quality_elem, 'dynamic_range_db').text = str(quality.dynamic_range_db)
            
            # Pretty print and save
            rough_string = tostring(root, 'unicode')
            reparsed = minidom.parseString(rough_string)
            
            with open(export_path, 'w') as f:
                f.write(reparsed.toprettyxml(indent="  "))
        
        except ImportError:
            logger.warning("XML export requires xml module, falling back to JSON")
            self._export_json(metadata, export_path.replace('.xml', '.json'))
        except Exception as e:
            logger.error(f"Error exporting XML: {e}")
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get metadata tracking statistics"""
        with self.metadata_lock:
            return {
                **self.tracking_stats,
                'active_sessions': len(self.active_sessions),
                'total_participants': sum(len(m.participants) for m in self.active_sessions.values()),
                'avg_session_duration': (
                    sum(m.duration_seconds for m in self.active_sessions.values()) / 
                    max(len(self.active_sessions), 1)
                )
            }


if __name__ == "__main__":
    # Basic test when run directly
    print("Metadata Tracker Test")
    print("=" * 50)
    
    tracker = MetadataTracker("test_metadata")
    
    # Mock session info
    class MockSessionInfo:
        def __init__(self):
            self.session_id = str(uuid.uuid4())
            self.session_type = type('SessionType', (), {'value': 'meeting'})()
            self.start_time = datetime.now()
            self.file_path = f"/recordings/{self.session_id}.flac"
            self.metadata = {
                'title': 'Test Meeting',
                'description': 'Test session for metadata tracking',
                'participants': ['John Doe', 'Jane Smith'],
                'location': 'Conference Room A'
            }
    
    session_info = MockSessionInfo()
    
    print(f"Starting tracking for session: {session_info.session_id[:8]}")
    if tracker.track_session(session_info):
        print("Session tracking started successfully")
        
        # Simulate participant activity
        participant_data = {
            'participant_id': 'john_doe',
            'name': 'John Doe',
            'speaking_duration': 30.0,
            'word_count': 150,
            'volume_db': -25.0,
            'speech_rate': 120
        }
        tracker.track_participant_activity(session_info.session_id, participant_data)
        print("Participant activity tracked")
        
        # Simulate audio quality data
        quality_data = {
            'snr_db': 25.0,
            'dynamic_range_db': 35.0,
            'distortion': 0.05,
            'clipping_detected': False,
            'speech_clarity': 0.85
        }
        tracker.track_audio_quality(session_info.session_id, quality_data)
        print("Audio quality tracked")
        
        # Simulate system performance
        performance_data = {
            'cpu_usage': 45.0,
            'memory_usage': 60.0,
            'audio_latency_ms': 25.0,
            'buffer_underruns': 0
        }
        tracker.track_system_performance(session_info.session_id, performance_data)
        print("System performance tracked")
        
        # Update metadata
        updates = {
            'title': 'Updated Test Meeting',
            'end_time': datetime.now(),
            'duration_seconds': 300.0
        }
        tracker.update_metadata(session_info.session_id, updates)
        print("Metadata updated")
        
        # Get metadata
        metadata = tracker.get_session_metadata(session_info.session_id)
        if metadata:
            print(f"Session metadata: {metadata.title}, {len(metadata.participants)} participants")
            print(f"Audio quality score: {metadata.audio_quality.overall_quality_score:.2f}")
        
        # Export metadata
        export_path = tracker.export_metadata(session_info.session_id, "json")
        if export_path:
            print(f"Metadata exported to: {export_path}")
        
        # Statistics
        stats = tracker.get_tracking_statistics()
        print(f"Tracking statistics: {stats}")
    
    print("Test complete!")