#!/usr/bin/env python3
"""
Verify Complete Setup for The Silent Steno

This script verifies that the database and all related components are properly set up.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_database_setup():
    """Verify database setup"""
    try:
        logger.info("Verifying database setup...")
        
        from src.data import get_database, Session, SessionStatus
        
        # Get database instance
        database = get_database()
        
        # Check database health
        if not database.health_check():
            logger.error("Database health check failed")
            return False
        
        # Check if tables exist
        with database.get_session() as session:
            result = session.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
            tables = [row[0] for row in result]
            
            expected_tables = ['sessions', 'transcript_entries', 'analysis_results', 'participants', 'users', 'configurations', 'file_info', 'system_metrics']
            
            for table in expected_tables:
                if table not in tables:
                    logger.error(f"Missing table: {table}")
                    return False
            
            logger.info(f"All expected tables present: {tables}")
        
        # Test basic session creation
        with database.transaction() as session:
            test_session = Session(
                title="Setup Verification",
                status=SessionStatus.IDLE.value
            )
            session.add(test_session)
            session.flush()
            session_id = test_session.id
            logger.info(f"Created verification session: {session_id}")
        
        # Cleanup
        with database.transaction() as session:
            test_session = session.query(Session).filter_by(id=session_id).first()
            if test_session:
                session.delete(test_session)
                logger.info("Cleaned up verification session")
        
        logger.info("Database setup verification: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Database setup verification failed: {e}")
        return False

def verify_imports():
    """Verify that all required modules can be imported"""
    try:
        logger.info("Verifying imports...")
        
        # Test data module imports
        from src.data import (
            Database, DatabaseConfig, get_database, create_database,
            Session, TranscriptEntry, AnalysisResult, Participant,
            User, Configuration, FileInfo, SystemMetrics,
            SessionStatus, TranscriptConfidence, AnalysisType
        )
        logger.info("Data module imports: PASSED")
        
        # Test backup manager
        from src.data import BackupManager, create_backup_manager
        logger.info("Backup manager imports: PASSED")
        
        # Test retention manager
        from src.data import RetentionManager, create_retention_manager
        logger.info("Retention manager imports: PASSED")
        
        # Test migration manager
        from src.data import MigrationManager, create_migration_manager
        logger.info("Migration manager imports: PASSED")
        
        return True
        
    except Exception as e:
        logger.error(f"Import verification failed: {e}")
        return False

def main():
    """Run all verification tests"""
    logger.info("Starting setup verification...")
    
    all_passed = True
    
    # Verify imports
    if not verify_imports():
        all_passed = False
    
    # Verify database setup
    if not verify_database_setup():
        all_passed = False
    
    if all_passed:
        logger.info("All setup verification tests PASSED!")
        logger.info("The Silent Steno database system is ready to use.")
    else:
        logger.error("Some setup verification tests FAILED!")
        logger.error("Please check the errors above and fix any issues.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)