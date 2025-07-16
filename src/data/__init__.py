#!/usr/bin/env python3

"""
Data Management Package for The Silent Steno

This package provides comprehensive data management capabilities including
database connectivity, data models, migrations, backup/restore, and
retention policies for the meeting recorder application.

Key Components:
- Database: SQLite connection and session management
- Models: SQLAlchemy data models for all entities
- Migrations: Schema migration system with Alembic
- Backup Manager: Automated backup and restore functionality
- Retention Manager: Data lifecycle and cleanup policies

Usage:
    from src.data import get_database, Session, create_backup_manager
    
    # Get database instance
    db = get_database()
    
    # Create session
    with db.get_session() as session:
        sessions = session.query(Session).all()
    
    # Create backup
    backup_manager = create_backup_manager(db)
    backup_info = backup_manager.create_backup()
"""

# Import main classes and functions for easy access
from .database import (
    Database,
    DatabaseConfig,
    DatabaseState,
    SessionManager,
    get_database,
    create_database,
    create_session,
    database_context,
    create_memory_database,
    create_test_database,
    create_production_database
)

from .models import (
    Base,
    Session,
    TranscriptEntry,
    AnalysisResult,
    Participant,
    User,
    Configuration,
    FileInfo,
    SystemMetrics,
    SessionStatus,
    TranscriptConfidence,
    AnalysisType,
    create_models,
    initialize_models
)

from .migrations import (
    MigrationManager,
    MigrationStatus,
    MigrationInfo,
    create_migration_manager,
    run_migrations,
    create_migration,
    check_migration_status,
    migration_context
)

from .backup_manager import (
    BackupManager,
    BackupConfig,
    BackupInfo,
    BackupStatus,
    BackupType,
    create_backup_manager,
    create_backup,
    verify_backup,
    backup_context
)

from .retention_manager import (
    RetentionManager,
    RetentionConfig,
    RetentionRule,
    RetentionCriteria,
    DataType,
    RetentionAction,
    CleanupResult,
    create_retention_manager,
    apply_retention_policy,
    estimate_cleanup_size
)

# Package metadata
__version__ = "1.0.0"
__author__ = "The Silent Steno Team"
__description__ = "Database and data management for The Silent Steno"

# Define what gets imported with "from src.data import *"
__all__ = [
    # Database
    "Database",
    "DatabaseConfig", 
    "DatabaseState",
    "SessionManager",
    "get_database",
    "create_database",
    "create_session",
    "database_context",
    "create_memory_database",
    "create_test_database",
    "create_production_database",
    
    # Models
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
    "initialize_models",
    
    # Migrations
    "MigrationManager",
    "MigrationStatus",
    "MigrationInfo",
    "create_migration_manager",
    "run_migrations",
    "create_migration",
    "check_migration_status",
    "migration_context",
    
    # Backup
    "BackupManager",
    "BackupConfig",
    "BackupInfo",
    "BackupStatus",
    "BackupType",
    "create_backup_manager",
    "create_backup",
    "verify_backup",
    "backup_context",
    
    # Retention
    "RetentionManager",
    "RetentionConfig",
    "RetentionRule",
    "RetentionCriteria",
    "DataType",
    "RetentionAction",
    "CleanupResult",
    "create_retention_manager",
    "apply_retention_policy",
    "estimate_cleanup_size"
]


# Convenience functions for common operations
def initialize_database(database_path: str = "data/silent_steno.db") -> Database:
    """
    Initialize database with default configuration
    
    Args:
        database_path: Path to database file
        
    Returns:
        Database: Initialized database instance
    """
    config = DatabaseConfig(database_path=database_path)
    database = create_database(config)
    
    # Create tables from models
    create_models(database.engine)
    
    return database


def setup_complete_data_system(database_path: str = "data/silent_steno.db") -> dict:
    """
    Set up complete data management system
    
    Args:
        database_path: Path to database file
        
    Returns:
        dict: Dictionary containing all managers
    """
    # Initialize database
    database = initialize_database(database_path)
    
    # Create managers
    backup_manager = create_backup_manager(database)
    retention_manager = create_retention_manager(database, backup_manager)
    migration_manager = create_migration_manager(database)
    
    return {
        "database": database,
        "backup_manager": backup_manager,
        "retention_manager": retention_manager,
        "migration_manager": migration_manager
    }


def get_system_status() -> dict:
    """
    Get comprehensive system status
    
    Returns:
        dict: System status information
    """
    try:
        database = get_database()
        
        status = {
            "database": database.get_status(),
            "health_check": database.health_check()
        }
        
        # Add backup status if available
        try:
            backup_manager = create_backup_manager(database)
            status["backup"] = backup_manager.get_backup_statistics()
        except:
            status["backup"] = {"error": "Backup manager not available"}
        
        # Add retention status if available
        try:
            retention_manager = create_retention_manager(database)
            status["retention"] = retention_manager.get_retention_status()
        except:
            status["retention"] = {"error": "Retention manager not available"}
        
        return status
        
    except Exception as e:
        return {"error": f"Failed to get system status: {e}"}


# Module-level initialization
def _initialize_logging():
    """Initialize logging for data package"""
    import logging
    
    # Create logger for data package
    logger = logging.getLogger(__name__)
    
    # Set default level if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


# Initialize logging when package is imported
_initialize_logging()