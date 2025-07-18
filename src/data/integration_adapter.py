#!/usr/bin/env python3

"""
Data Integration Adapter for The Silent Steno

This module provides a bridge between the simple JSON-based demo data structure
and the comprehensive SQLite-based data models. It allows the walking skeleton
to gradually migrate from simple file-based storage to the full database system.

Key features:
- Bidirectional conversion between JSON and SQLAlchemy models
- Gradual migration support
- Backward compatibility with existing demo sessions
- Integration with existing database infrastructure
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import data models and database (with fallback for missing imports)
try:
    from src.data.models import Session, TranscriptEntry, AnalysisResult, SessionStatus
    from src.data.database import get_database
    DATABASE_AVAILABLE = True
except ImportError:
    logger.warning("Database models not available - using fallback mode")
    DATABASE_AVAILABLE = False
    Session = None
    TranscriptEntry = None
    AnalysisResult = None
    SessionStatus = None


class DataIntegrationAdapter:
    """
    Bridge between simple JSON demo data and comprehensive SQLite models
    
    This adapter allows the walking skeleton to use either the simple JSON
    format or the full database system, providing a migration path between
    the two approaches.
    """
    
    def __init__(self, sessions_file: str = "demo_sessions/sessions.json", 
                 use_database: bool = True):
        """
        Initialize the data integration adapter
        
        Args:
            sessions_file: Path to JSON sessions file
            use_database: Whether to use database backend
        """
        self.sessions_file = sessions_file
        self.use_database = use_database and DATABASE_AVAILABLE
        
        # Ensure sessions directory exists
        os.makedirs(os.path.dirname(sessions_file), exist_ok=True)
        
        # Initialize database if available
        if self.use_database:
            try:
                self.database = get_database()
                logger.info("Database integration enabled")
            except Exception as e:
                logger.warning(f"Database initialization failed: {e}")
                self.use_database = False
        
        logger.info(f"Data integration adapter initialized (database: {self.use_database})")
    
    def create_session(self, session_data: Dict[str, Any]) -> Optional[Union[int, str]]:
        """
        Create a new session using either database or JSON storage
        
        Args:
            session_data: Session data dictionary
            
        Returns:
            Session ID (database) or session identifier (JSON)
        """
        try:
            if self.use_database:
                return self._create_database_session(session_data)
            else:
                return self._create_json_session(session_data)
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return None
    
    def _create_database_session(self, session_data: Dict[str, Any]) -> Optional[int]:
        """Create session in database"""
        try:
            with self.database.transaction() as db_session:
                # Convert JSON data to database model
                session = Session(
                    uuid=session_data.get('session_id', str(uuid.uuid4())),
                    title=session_data.get('title', 'Recording Session'),
                    description=session_data.get('description', 'Audio recording session'),
                    start_time=datetime.fromisoformat(session_data.get('timestamp', datetime.now().isoformat())),
                    status='active',  # Use string instead of enum
                    duration_seconds=session_data.get('duration', 0.0),
                    file_path=session_data.get('wav_file', ''),
                    sample_rate=session_data.get('sample_rate', 44100),
                    channels=session_data.get('channels', 1),
                    metadata=session_data.get('metadata', {})
                )
                
                db_session.add(session)
                db_session.flush()  # Get the ID
                session_id = session.id
                
                # Add transcript entry if available
                if session_data.get('transcript'):
                    transcript_entry = TranscriptEntry(
                        session_id=session_id,
                        text=session_data['transcript'],
                        speaker='Unknown',
                        start_time=0.0,
                        end_time=session_data.get('duration', 0.0),
                        confidence=0.8,  # Default confidence
                        language='en'
                    )
                    db_session.add(transcript_entry)
                
                db_session.commit()
                logger.info(f"Created database session: {session_id}")
                return session_id
                
        except Exception as e:
            logger.error(f"Error creating database session: {e}")
            return None
    
    def _create_json_session(self, session_data: Dict[str, Any]) -> Optional[str]:
        """Create session in JSON file"""
        try:
            # Load existing sessions
            sessions = self._load_json_sessions()
            
            # Generate session ID if not provided
            session_id = session_data.get('session_id', str(uuid.uuid4())[:8])
            
            # Create new session record
            new_session = {
                'id': len(sessions) + 1,
                'session_id': session_id,
                'timestamp': session_data.get('timestamp', datetime.now().isoformat()),
                'title': session_data.get('title', 'Recording Session'),
                'wav_file': session_data.get('wav_file', ''),
                'duration': session_data.get('duration', 0.0),
                'samples': session_data.get('samples', 0),
                'sample_rate': session_data.get('sample_rate', 44100),
                'channels': session_data.get('channels', 1),
                'transcript': session_data.get('transcript', None),
                'metadata': session_data.get('metadata', {})
            }
            
            sessions.append(new_session)
            
            # Save updated sessions
            self._save_json_sessions(sessions)
            
            logger.info(f"Created JSON session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating JSON session: {e}")
            return None
    
    def update_session(self, session_id: Union[int, str], updates: Dict[str, Any]) -> bool:
        """
        Update an existing session
        
        Args:
            session_id: Session identifier
            updates: Dictionary of fields to update
            
        Returns:
            True if update successful
        """
        try:
            if self.use_database:
                return self._update_database_session(session_id, updates)
            else:
                return self._update_json_session(session_id, updates)
        except Exception as e:
            logger.error(f"Error updating session: {e}")
            return False
    
    def _update_database_session(self, session_id: int, updates: Dict[str, Any]) -> bool:
        """Update session in database"""
        try:
            with self.database.transaction() as db_session:
                session = db_session.query(Session).filter_by(id=session_id).first()
                if not session:
                    logger.error(f"Session not found: {session_id}")
                    return False
                
                # Update session fields
                for field, value in updates.items():
                    if field == 'transcript' and value:
                        # Update or create transcript entry
                        transcript = db_session.query(TranscriptEntry).filter_by(session_id=session_id).first()
                        if transcript:
                            transcript.text = value
                        else:
                            transcript = TranscriptEntry(
                                session_id=session_id,
                                text=value,
                                speaker='Unknown',
                                start_time=0.0,
                                end_time=session.duration_seconds or 0.0,
                                confidence=0.8,
                                language='en'
                            )
                            db_session.add(transcript)
                    elif hasattr(session, field):
                        setattr(session, field, value)
                
                db_session.commit()
                logger.info(f"Updated database session: {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating database session: {e}")
            return False
    
    def _update_json_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session in JSON file"""
        try:
            sessions = self._load_json_sessions()
            
            # Find session to update
            session_found = False
            for session in sessions:
                if session.get('session_id') == session_id:
                    session.update(updates)
                    session_found = True
                    break
            
            if not session_found:
                logger.error(f"Session not found: {session_id}")
                return False
            
            # Save updated sessions
            self._save_json_sessions(sessions)
            
            logger.info(f"Updated JSON session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating JSON session: {e}")
            return False
    
    def get_session(self, session_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Get session data by ID
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data dictionary or None
        """
        try:
            if self.use_database:
                return self._get_database_session(session_id)
            else:
                return self._get_json_session(session_id)
        except Exception as e:
            logger.error(f"Error getting session: {e}")
            return None
    
    def _get_database_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Get session from database"""
        try:
            with self.database.transaction() as db_session:
                session = db_session.query(Session).filter_by(id=session_id).first()
                if not session:
                    return None
                
                # Convert to dictionary format
                session_data = {
                    'id': session.id,
                    'session_id': session.uuid,
                    'timestamp': session.start_time.isoformat(),
                    'title': session.title,
                    'description': session.description,
                    'wav_file': session.file_path,
                    'duration': session.duration_seconds or 0.0,
                    'sample_rate': session.sample_rate,
                    'channels': session.channels,
                    'status': session.status.value if session.status else 'active',
                    'metadata': session.metadata or {}
                }
                
                # Add transcript if available
                transcript = db_session.query(TranscriptEntry).filter_by(session_id=session_id).first()
                if transcript:
                    session_data['transcript'] = transcript.text
                
                return session_data
                
        except Exception as e:
            logger.error(f"Error getting database session: {e}")
            return None
    
    def _get_json_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session from JSON file"""
        try:
            sessions = self._load_json_sessions()
            
            for session in sessions:
                if session.get('session_id') == session_id:
                    return session
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting JSON session: {e}")
            return None
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """
        Get all sessions
        
        Returns:
            List of session dictionaries
        """
        try:
            if self.use_database:
                return self._get_all_database_sessions()
            else:
                return self._load_json_sessions()
        except Exception as e:
            logger.error(f"Error getting all sessions: {e}")
            return []
    
    def _get_all_database_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions from database"""
        try:
            with self.database.transaction() as db_session:
                sessions = db_session.query(Session).order_by(Session.start_time.desc()).all()
                
                session_list = []
                for session in sessions:
                    session_data = {
                        'id': session.id,
                        'session_id': session.uuid,
                        'timestamp': session.start_time.isoformat(),
                        'title': session.title,
                        'description': session.description,
                        'wav_file': session.file_path,
                        'duration': session.duration_seconds or 0.0,
                        'sample_rate': session.sample_rate,
                        'channels': session.channels,
                        'status': session.status.value if session.status else 'active',
                        'metadata': session.metadata or {}
                    }
                    
                    # Add transcript if available
                    transcript = db_session.query(TranscriptEntry).filter_by(session_id=session.id).first()
                    if transcript:
                        session_data['transcript'] = transcript.text
                    
                    session_list.append(session_data)
                
                return session_list
                
        except Exception as e:
            logger.error(f"Error getting all database sessions: {e}")
            return []
    
    def _load_json_sessions(self) -> List[Dict[str, Any]]:
        """Load sessions from JSON file"""
        try:
            if os.path.exists(self.sessions_file):
                with open(self.sessions_file, 'r') as f:
                    sessions = json.load(f)
                return sessions if isinstance(sessions, list) else []
            else:
                return []
        except Exception as e:
            logger.error(f"Error loading JSON sessions: {e}")
            return []
    
    def _save_json_sessions(self, sessions: List[Dict[str, Any]]):
        """Save sessions to JSON file"""
        try:
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving JSON sessions: {e}")
    
    def migrate_to_database(self) -> bool:
        """
        Migrate existing JSON sessions to database
        
        Returns:
            True if migration successful
        """
        if not DATABASE_AVAILABLE:
            logger.error("Database not available for migration")
            return False
        
        try:
            # Load existing JSON sessions
            json_sessions = self._load_json_sessions()
            
            if not json_sessions:
                logger.info("No JSON sessions to migrate")
                return True
            
            # Migrate each session
            migrated_count = 0
            
            with self.database.transaction() as db_session:
                for json_session in json_sessions:
                    try:
                        # Create database session
                        session = Session(
                            uuid=json_session.get('session_id', str(uuid.uuid4())),
                            title=json_session.get('title', 'Migrated Session'),
                            description='Migrated from JSON storage',
                            start_time=datetime.fromisoformat(json_session.get('timestamp', datetime.now().isoformat())),
                            status='completed',  # Use string instead of enum
                            duration_seconds=json_session.get('duration', 0.0),
                            file_path=json_session.get('wav_file', ''),
                            sample_rate=json_session.get('sample_rate', 44100),
                            channels=json_session.get('channels', 1),
                            metadata=json_session.get('metadata', {})
                        )
                        
                        db_session.add(session)
                        db_session.flush()  # Get the ID
                        
                        # Add transcript if available
                        if json_session.get('transcript'):
                            transcript = TranscriptEntry(
                                session_id=session.id,
                                text=json_session['transcript'],
                                speaker='Unknown',
                                start_time=0.0,
                                end_time=json_session.get('duration', 0.0),
                                confidence=0.8,
                                language='en'
                            )
                            db_session.add(transcript)
                        
                        migrated_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error migrating session {json_session.get('id')}: {e}")
                        continue
                
                db_session.commit()
            
            logger.info(f"Successfully migrated {migrated_count}/{len(json_sessions)} sessions to database")
            
            # Enable database mode
            self.use_database = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error during migration: {e}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            if self.use_database:
                return self._get_database_stats()
            else:
                return self._get_json_stats()
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}
    
    def _get_database_stats(self) -> Dict[str, Any]:
        """Get database storage statistics"""
        try:
            with self.database.transaction() as db_session:
                session_count = db_session.query(Session).count()
                transcript_count = db_session.query(TranscriptEntry).count()
                
                return {
                    'storage_type': 'database',
                    'session_count': session_count,
                    'transcript_count': transcript_count,
                    'database_available': True
                }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'storage_type': 'database', 'error': str(e)}
    
    def _get_json_stats(self) -> Dict[str, Any]:
        """Get JSON storage statistics"""
        try:
            sessions = self._load_json_sessions()
            transcript_count = sum(1 for s in sessions if s.get('transcript'))
            
            return {
                'storage_type': 'json',
                'session_count': len(sessions),
                'transcript_count': transcript_count,
                'database_available': DATABASE_AVAILABLE
            }
        except Exception as e:
            logger.error(f"Error getting JSON stats: {e}")
            return {'storage_type': 'json', 'error': str(e)}


# Factory functions for easy integration
def create_data_adapter(sessions_file: str = "demo_sessions/sessions.json", 
                       use_database: bool = True) -> DataIntegrationAdapter:
    """
    Create a data integration adapter
    
    Args:
        sessions_file: Path to JSON sessions file
        use_database: Whether to use database backend
        
    Returns:
        DataIntegrationAdapter instance
    """
    return DataIntegrationAdapter(sessions_file, use_database)


def migrate_demo_data() -> bool:
    """
    Migrate demo data from JSON to database
    
    Returns:
        True if migration successful
    """
    adapter = create_data_adapter()
    return adapter.migrate_to_database()


if __name__ == "__main__":
    # Test the adapter
    print("Data Integration Adapter Test")
    print("=" * 50)
    
    # Create adapter
    adapter = create_data_adapter(use_database=False)  # Test with JSON first
    
    # Test session creation
    test_session = {
        'session_id': 'test-integration',
        'title': 'Integration Test Session',
        'timestamp': datetime.now().isoformat(),
        'duration': 10.0,
        'transcript': 'This is a test transcript'
    }
    
    session_id = adapter.create_session(test_session)
    print(f"Created session: {session_id}")
    
    # Test session retrieval
    retrieved_session = adapter.get_session(session_id)
    print(f"Retrieved session: {retrieved_session['title']}")
    
    # Test update
    adapter.update_session(session_id, {'transcript': 'Updated transcript'})
    
    # Test all sessions
    all_sessions = adapter.get_all_sessions()
    print(f"Total sessions: {len(all_sessions)}")
    
    # Test stats
    stats = adapter.get_storage_stats()
    print(f"Storage stats: {stats}")
    
    print("✓ Data integration adapter test completed")