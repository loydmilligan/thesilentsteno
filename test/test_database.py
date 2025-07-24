#!/usr/bin/env python3
"""
Test Database Operations for The Silent Steno

This script tests basic database operations to ensure everything is working correctly.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data import get_database, Session, SessionStatus, TranscriptEntry, AnalysisResult

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_database_operations():
    """Test basic database operations"""
    try:
        logger.info("Testing database operations...")
        
        # Get database instance
        database = get_database()
        
        # Test 1: Create a session
        with database.transaction() as session:
            new_session = Session(
                title="Test Session",
                description="Test session for database validation",
                status=SessionStatus.IDLE.value,
                start_time=datetime.now(timezone.utc),
                duration_seconds=0,
                participant_count=1
            )
            session.add(new_session)
            session.flush()  # Get the ID
            session_id = new_session.id
            logger.info(f"Created test session with ID: {session_id}")
        
        # Test 2: Query the session
        with database.get_session() as session:
            retrieved_session = session.query(Session).filter_by(id=session_id).first()
            if retrieved_session:
                logger.info(f"Retrieved session: {retrieved_session.title}")
                logger.info(f"Session UUID: {retrieved_session.uuid}")
                logger.info(f"Session status: {retrieved_session.status}")
            else:
                logger.error("Failed to retrieve session")
                return False
        
        # Test 3: Add a transcript entry
        with database.transaction() as session:
            transcript = TranscriptEntry(
                session_id=session_id,
                text="This is a test transcript entry",
                speaker_id="speaker_1",
                speaker_name="Test Speaker",
                start_time_seconds=0.0,
                end_time_seconds=5.0,
                duration_seconds=5.0,
                confidence=0.95,
                word_count=6
            )
            session.add(transcript)
            logger.info("Added transcript entry")
        
        # Test 4: Add an analysis result
        with database.transaction() as session:
            analysis = AnalysisResult(
                session_id=session_id,
                analysis_type="summary",
                title="Test Analysis",
                content="This is a test analysis result",
                confidence_score=0.8,
                processing_time_seconds=1.5,
                model_used="test-model"
            )
            session.add(analysis)
            logger.info("Added analysis result")
        
        # Test 5: Query related data
        with database.get_session() as session:
            retrieved_session = session.query(Session).filter_by(id=session_id).first()
            logger.info(f"Session has {len(retrieved_session.transcripts)} transcript entries")
            logger.info(f"Session has {len(retrieved_session.analyses)} analysis results")
            
            if retrieved_session.transcripts:
                logger.info(f"First transcript: {retrieved_session.transcripts[0].text}")
            
            if retrieved_session.analyses:
                logger.info(f"First analysis: {retrieved_session.analyses[0].title}")
        
        # Test 6: Cleanup - delete test session
        with database.transaction() as session:
            test_session = session.query(Session).filter_by(id=session_id).first()
            if test_session:
                session.delete(test_session)
                logger.info("Deleted test session (cleanup)")
        
        logger.info("All database tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run database tests"""
    logger.info("Starting database tests...")
    
    success = test_database_operations()
    
    if success:
        logger.info("Database tests completed successfully!")
    else:
        logger.error("Database tests failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)