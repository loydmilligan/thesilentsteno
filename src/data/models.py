#!/usr/bin/env python3

"""
SQLAlchemy Data Models for The Silent Steno

This module defines all database models and relationships for the meeting recorder
application. It provides comprehensive data models for sessions, transcripts,
analysis results, participants, and system configuration.

Key features:
- Complete relational schema with proper foreign keys
- Model validation and serialization methods
- Relationship definitions with lazy loading
- Timestamps and metadata tracking
- Flexible JSON fields for extensibility
- Performance optimizations with indexes
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.sqlite import BLOB
from sqlalchemy.sql import func

# Create base class for all models
Base = declarative_base()


class SessionStatus(Enum):
    """Session status enumeration"""
    IDLE = "idle"
    STARTING = "starting"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPING = "stopping"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    ARCHIVED = "archived"


class TranscriptConfidence(Enum):
    """Transcript confidence levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class AnalysisType(Enum):
    """Analysis type enumeration"""
    SUMMARY = "summary"
    ACTION_ITEMS = "action_items"
    TOPICS = "topics"
    SENTIMENT = "sentiment"
    PARTICIPANTS = "participants"
    INSIGHTS = "insights"
    COMPREHENSIVE = "comprehensive"


class Session(Base):
    """
    Main session model representing a meeting recording session
    """
    __tablename__ = "sessions"
    
    # Primary key and identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # Session metadata
    title = Column(String(255), nullable=False, default="Untitled Session")
    description = Column(Text)
    status = Column(String(20), nullable=False, default=SessionStatus.IDLE.value)
    
    # Timing information
    start_time = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    end_time = Column(DateTime(timezone=True))
    duration_seconds = Column(Integer, default=0)
    
    # Audio information
    audio_file_path = Column(String(500))
    audio_format = Column(String(10), default="wav")
    audio_size_bytes = Column(Integer, default=0)
    audio_quality = Column(String(20), default="high")
    sample_rate = Column(Integer, default=44100)
    
    # Participants and environment
    participant_count = Column(Integer, default=0)
    location = Column(String(255))
    meeting_platform = Column(String(100))
    
    # Processing information
    transcription_completed = Column(Boolean, default=False)
    analysis_completed = Column(Boolean, default=False)
    processing_time_seconds = Column(Integer, default=0)
    
    # Quality metrics
    transcription_confidence = Column(Float, default=0.0)
    audio_level_avg = Column(Float, default=0.0)
    audio_level_max = Column(Float, default=0.0)
    
    # Metadata and configuration
    tags = Column(JSON, default=list)
    session_metadata = Column(JSON, default=dict)
    configuration = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    transcripts = relationship("TranscriptEntry", back_populates="session", cascade="all, delete-orphan")
    analyses = relationship("AnalysisResult", back_populates="session", cascade="all, delete-orphan")
    participants = relationship("Participant", back_populates="session", cascade="all, delete-orphan")
    files = relationship("FileInfo", back_populates="session", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_session_status", "status"),
        Index("idx_session_created", "created_at"),
        Index("idx_session_start_time", "start_time"),
        Index("idx_session_uuid", "uuid"),
        CheckConstraint("duration_seconds >= 0", name="check_duration_positive"),
        CheckConstraint("participant_count >= 0", name="check_participant_count_positive"),
    )
    
    @validates('status')
    def validate_status(self, key, status):
        if status not in [s.value for s in SessionStatus]:
            raise ValueError(f"Invalid session status: {status}")
        return status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization"""
        return {
            "id": self.id,
            "uuid": self.uuid,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "audio_file_path": self.audio_file_path,
            "audio_format": self.audio_format,
            "audio_size_bytes": self.audio_size_bytes,
            "participant_count": self.participant_count,
            "location": self.location,
            "meeting_platform": self.meeting_platform,
            "transcription_completed": self.transcription_completed,
            "analysis_completed": self.analysis_completed,
            "transcription_confidence": self.transcription_confidence,
            "tags": self.tags,
            "session_metadata": self.session_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    def get_transcript_text(self) -> str:
        """Get combined transcript text"""
        return "\n".join([entry.text for entry in self.transcripts])
    
    def get_duration_formatted(self) -> str:
        """Get formatted duration string"""
        if not self.duration_seconds:
            return "00:00:00"
        
        hours = self.duration_seconds // 3600
        minutes = (self.duration_seconds % 3600) // 60
        seconds = self.duration_seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TranscriptEntry(Base):
    """
    Individual transcript entry with speaker and timing information
    """
    __tablename__ = "transcript_entries"
    
    # Primary key and relationships
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    
    # Transcript content
    text = Column(Text, nullable=False)
    speaker_id = Column(String(50))
    speaker_name = Column(String(255))
    
    # Timing information
    start_time_seconds = Column(Float, nullable=False)
    end_time_seconds = Column(Float, nullable=False)
    duration_seconds = Column(Float, nullable=False)
    
    # Quality metrics
    confidence = Column(Float, default=0.0)
    confidence_level = Column(String(20), default=TranscriptConfidence.MEDIUM.value)
    
    # Audio characteristics
    audio_level = Column(Float, default=0.0)
    word_count = Column(Integer, default=0)
    speaking_rate = Column(Float, default=0.0)  # words per minute
    
    # Processing information
    language = Column(String(10), default="en")
    processing_model = Column(String(100), default="whisper-base")
    
    # Metadata
    entry_metadata = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    session = relationship("Session", back_populates="transcripts")
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_transcript_session", "session_id"),
        Index("idx_transcript_speaker", "speaker_id"),
        Index("idx_transcript_time", "start_time_seconds", "end_time_seconds"),
        CheckConstraint("start_time_seconds >= 0", name="check_start_time_positive"),
        CheckConstraint("end_time_seconds >= start_time_seconds", name="check_end_after_start"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="check_confidence_range"),
    )
    
    @validates('confidence_level')
    def validate_confidence_level(self, key, level):
        if level not in [c.value for c in TranscriptConfidence]:
            raise ValueError(f"Invalid confidence level: {level}")
        return level
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transcript entry to dictionary"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "text": self.text,
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "start_time_seconds": self.start_time_seconds,
            "end_time_seconds": self.end_time_seconds,
            "duration_seconds": self.duration_seconds,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level,
            "audio_level": self.audio_level,
            "word_count": self.word_count,
            "speaking_rate": self.speaking_rate,
            "language": self.language,
            "processing_model": self.processing_model,
            "metadata": self.entry_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class AnalysisResult(Base):
    """
    AI analysis results for sessions
    """
    __tablename__ = "analysis_results"
    
    # Primary key and relationships
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    
    # Analysis information
    analysis_type = Column(String(50), nullable=False)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    
    # Structured data
    structured_data = Column(JSON, default=dict)
    action_items = Column(JSON, default=list)
    key_topics = Column(JSON, default=list)
    
    # Quality metrics
    confidence_score = Column(Float, default=0.0)
    processing_time_seconds = Column(Float, default=0.0)
    
    # Processing information
    model_used = Column(String(100))
    model_version = Column(String(50))
    prompt_template = Column(String(100))
    
    # Metadata
    entry_metadata = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    session = relationship("Session", back_populates="analyses")
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_analysis_session", "session_id"),
        Index("idx_analysis_type", "analysis_type"),
        Index("idx_analysis_created", "created_at"),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name="check_confidence_range"),
    )
    
    @validates('analysis_type')
    def validate_analysis_type(self, key, analysis_type):
        if analysis_type not in [a.value for a in AnalysisType]:
            raise ValueError(f"Invalid analysis type: {analysis_type}")
        return analysis_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result to dictionary"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "analysis_type": self.analysis_type,
            "title": self.title,
            "content": self.content,
            "structured_data": self.structured_data,
            "action_items": self.action_items,
            "key_topics": self.key_topics,
            "confidence_score": self.confidence_score,
            "processing_time_seconds": self.processing_time_seconds,
            "model_used": self.model_used,
            "model_version": self.model_version,
            "metadata": self.entry_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Participant(Base):
    """
    Meeting participant information and statistics
    """
    __tablename__ = "participants"
    
    # Primary key and relationships
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    
    # Participant identification
    speaker_id = Column(String(50), nullable=False)
    name = Column(String(255))
    role = Column(String(100))
    organization = Column(String(255))
    
    # Speaking statistics
    total_speaking_time_seconds = Column(Float, default=0.0)
    speaking_percentage = Column(Float, default=0.0)
    interruption_count = Column(Integer, default=0)
    questions_asked = Column(Integer, default=0)
    
    # Audio characteristics
    average_audio_level = Column(Float, default=0.0)
    speaking_rate_wpm = Column(Float, default=0.0)  # words per minute
    
    # Engagement metrics
    engagement_score = Column(Float, default=0.0)
    contribution_score = Column(Float, default=0.0)
    
    # Metadata
    entry_metadata = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    session = relationship("Session", back_populates="participants")
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_participant_session", "session_id"),
        Index("idx_participant_speaker", "speaker_id"),
        UniqueConstraint("session_id", "speaker_id", name="uq_session_speaker"),
        CheckConstraint("total_speaking_time_seconds >= 0", name="check_speaking_time_positive"),
        CheckConstraint("speaking_percentage >= 0 AND speaking_percentage <= 100", name="check_speaking_percentage_range"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert participant to dictionary"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "speaker_id": self.speaker_id,
            "name": self.name,
            "role": self.role,
            "organization": self.organization,
            "total_speaking_time_seconds": self.total_speaking_time_seconds,
            "speaking_percentage": self.speaking_percentage,
            "interruption_count": self.interruption_count,
            "questions_asked": self.questions_asked,
            "average_audio_level": self.average_audio_level,
            "speaking_rate_wpm": self.speaking_rate_wpm,
            "engagement_score": self.engagement_score,
            "contribution_score": self.contribution_score,
            "metadata": self.entry_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class User(Base):
    """
    System user configuration and preferences
    """
    __tablename__ = "users"
    
    # Primary key and identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255))
    full_name = Column(String(255))
    
    # Preferences
    preferred_language = Column(String(10), default="en")
    timezone = Column(String(50), default="UTC")
    theme = Column(String(20), default="dark")
    
    # Settings
    auto_transcribe = Column(Boolean, default=True)
    auto_analyze = Column(Boolean, default=True)
    backup_enabled = Column(Boolean, default=True)
    notification_enabled = Column(Boolean, default=True)
    
    # Metadata
    preferences = Column(JSON, default=dict)
    entry_metadata = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime(timezone=True))
    
    # Indexes
    __table_args__ = (
        Index("idx_user_username", "username"),
        Index("idx_user_email", "email"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "preferred_language": self.preferred_language,
            "timezone": self.timezone,
            "theme": self.theme,
            "auto_transcribe": self.auto_transcribe,
            "auto_analyze": self.auto_analyze,
            "backup_enabled": self.backup_enabled,
            "notification_enabled": self.notification_enabled,
            "preferences": self.preferences,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None
        }


class Configuration(Base):
    """
    System configuration and settings
    """
    __tablename__ = "configurations"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Configuration identification
    category = Column(String(50), nullable=False)
    key = Column(String(100), nullable=False)
    value = Column(Text, nullable=False)
    
    # Metadata
    description = Column(Text)
    data_type = Column(String(20), default="string")  # string, integer, float, boolean, json
    is_system = Column(Boolean, default=False)
    is_user_editable = Column(Boolean, default=True)
    
    # Validation
    validation_rules = Column(JSON, default=dict)
    default_value = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Indexes
    __table_args__ = (
        Index("idx_config_category", "category"),
        Index("idx_config_key", "key"),
        UniqueConstraint("category", "key", name="uq_category_key"),
    )
    
    def get_typed_value(self) -> Union[str, int, float, bool, Dict, List]:
        """Get value with proper type conversion"""
        if self.data_type == "integer":
            return int(self.value)
        elif self.data_type == "float":
            return float(self.value)
        elif self.data_type == "boolean":
            return self.value.lower() in ("true", "1", "yes", "on")
        elif self.data_type == "json":
            return json.loads(self.value)
        else:
            return self.value
    
    def set_typed_value(self, value: Any):
        """Set value with proper type conversion"""
        if self.data_type == "json":
            self.value = json.dumps(value)
        else:
            self.value = str(value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "id": self.id,
            "category": self.category,
            "key": self.key,
            "value": self.get_typed_value(),
            "description": self.description,
            "data_type": self.data_type,
            "is_system": self.is_system,
            "is_user_editable": self.is_user_editable,
            "validation_rules": self.validation_rules,
            "default_value": self.default_value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class FileInfo(Base):
    """
    File metadata and storage information
    """
    __tablename__ = "file_info"
    
    # Primary key and relationships
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    
    # File information
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)  # audio, transcript, analysis, export
    file_format = Column(String(20), nullable=False)  # wav, mp3, txt, pdf, etc.
    
    # File properties
    size_bytes = Column(Integer, nullable=False, default=0)
    checksum = Column(String(64))  # SHA-256 hash
    mime_type = Column(String(100))
    
    # Storage information
    storage_location = Column(String(20), default="local")  # local, cloud, backup
    is_compressed = Column(Boolean, default=False)
    compression_ratio = Column(Float, default=1.0)
    
    # Access control
    is_encrypted = Column(Boolean, default=False)
    access_level = Column(String(20), default="private")  # public, private, restricted
    
    # Metadata
    entry_metadata = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    accessed_at = Column(DateTime(timezone=True))
    
    # Relationships
    session = relationship("Session", back_populates="files")
    
    # Indexes
    __table_args__ = (
        Index("idx_file_session", "session_id"),
        Index("idx_file_type", "file_type"),
        Index("idx_file_path", "file_path"),
        CheckConstraint("size_bytes >= 0", name="check_size_positive"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert file info to dictionary"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "filename": self.filename,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "file_format": self.file_format,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "mime_type": self.mime_type,
            "storage_location": self.storage_location,
            "is_compressed": self.is_compressed,
            "compression_ratio": self.compression_ratio,
            "is_encrypted": self.is_encrypted,
            "access_level": self.access_level,
            "metadata": self.entry_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "accessed_at": self.accessed_at.isoformat() if self.accessed_at else None
        }


class SystemMetrics(Base):
    """
    System performance and health metrics
    """
    __tablename__ = "system_metrics"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Metric identification
    metric_name = Column(String(100), nullable=False)
    metric_category = Column(String(50), nullable=False)
    
    # Metric values
    value = Column(Float, nullable=False)
    unit = Column(String(20))
    
    # Context
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="SET NULL"))
    component = Column(String(50))
    
    # Metadata
    entry_metadata = Column(JSON, default=dict)
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Indexes
    __table_args__ = (
        Index("idx_metrics_name", "metric_name"),
        Index("idx_metrics_category", "metric_category"),
        Index("idx_metrics_recorded", "recorded_at"),
        Index("idx_metrics_session", "session_id"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system metrics to dictionary"""
        return {
            "id": self.id,
            "metric_name": self.metric_name,
            "metric_category": self.metric_category,
            "value": self.value,
            "unit": self.unit,
            "session_id": self.session_id,
            "component": self.component,
            "metadata": self.entry_metadata,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None
        }


# Helper functions for model operations
def create_models(engine):
    """Create all database tables"""
    Base.metadata.create_all(engine)


def initialize_models():
    """Initialize models and return Base for imports"""
    return Base


# Export all models for easy importing
__all__ = [
    "Base",
    "Session",
    "TranscriptEntry", 
    "AnalysisResult",
    "Participant",
    "User",
    "Configuration",
    "FileInfo",
    "SystemMetrics",
    "SessionStatus",
    "TranscriptConfidence",
    "AnalysisType",
    "create_models",
    "initialize_models"
]


if __name__ == "__main__":
    # Basic test when run directly
    print("Data Models Test")
    print("=" * 40)
    
    # Test model creation
    from sqlalchemy import create_engine
    
    engine = create_engine("sqlite:///:memory:", echo=True)
    create_models(engine)
    
    print("All models created successfully")
    
    # Test model instantiation
    session = Session(title="Test Session", description="Test description")
    print(f"Session created: {session.title}")
    
    transcript = TranscriptEntry(
        text="Hello world",
        speaker_id="speaker1",
        start_time_seconds=0.0,
        end_time_seconds=2.5,
        duration_seconds=2.5
    )
    print(f"Transcript entry created: {transcript.text}")
    
    # Test serialization
    session_dict = session.to_dict()
    print(f"Session serialized: {len(session_dict)} fields")
    
    print("Model test completed successfully")