#!/usr/bin/env python3

"""
Database Schema Migration System for The Silent Steno

This module provides a comprehensive database migration system using Alembic
for safe schema updates and versioning. It handles schema evolution, data
migration, and rollback capabilities for the meeting recorder application.

Key features:
- Alembic-based migration framework
- Automatic migration generation
- Safe schema updates with rollback support
- Data migration capabilities
- Migration validation and verification
- Backup integration before major changes
"""

import os
import logging
import shutil
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.environment import EnvironmentContext
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .database import Database, DatabaseConfig
from .models import Base

# Set up logging
logger = logging.getLogger(__name__)


class MigrationStatus(Enum):
    """Migration status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class MigrationInfo:
    """Migration information structure"""
    revision: str
    description: str
    status: MigrationStatus
    created_at: datetime
    applied_at: Optional[datetime] = None
    rolled_back_at: Optional[datetime] = None
    error_message: Optional[str] = None


class MigrationManager:
    """
    Database migration manager using Alembic
    
    Provides safe database schema evolution with version control,
    rollback capabilities, and data migration support.
    """
    
    def __init__(self, database: Database, migrations_dir: str = "migrations"):
        """
        Initialize migration manager
        
        Args:
            database: Database instance
            migrations_dir: Directory for migration files
        """
        self.database = database
        self.migrations_dir = Path(migrations_dir)
        self.alembic_config: Optional[Config] = None
        self._migration_history: List[MigrationInfo] = []
        
        # Ensure migrations directory exists
        self.migrations_dir.mkdir(exist_ok=True)
        
        # Initialize Alembic configuration
        self._initialize_alembic()
        
        logger.info(f"Migration manager initialized: {self.migrations_dir}")
    
    def _initialize_alembic(self):
        """Initialize Alembic configuration"""
        try:
            # Create alembic.ini if it doesn't exist
            alembic_ini_path = self.migrations_dir / "alembic.ini"
            if not alembic_ini_path.exists():
                self._create_alembic_config()
            
            # Load Alembic configuration
            self.alembic_config = Config(str(alembic_ini_path))
            
            # Set database URL
            database_url = f"sqlite:///{self.database.config.database_path}"
            self.alembic_config.set_main_option("sqlalchemy.url", database_url)
            
            # Set script location
            script_location = str(self.migrations_dir / "versions")
            self.alembic_config.set_main_option("script_location", script_location)
            
            # Initialize Alembic if not already done
            if not (self.migrations_dir / "versions").exists():
                self._initialize_alembic_environment()
                
        except Exception as e:
            logger.error(f"Failed to initialize Alembic: {e}")
            raise
    
    def _create_alembic_config(self):
        """Create Alembic configuration file"""
        alembic_ini_content = """
# Alembic configuration for The Silent Steno

[alembic]
# Path to migration scripts
script_location = versions

# Template used to generate migration files
file_template = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d_%%(rev)s_%%(slug)s

# Timezone for timestamps
timezone = 

# Max length of characters to apply to the
# "slug" field
truncate_slug_length = 40

# Set to 'true' to run the environment during
# the 'revision' command, regardless of autogenerate
revision_environment = false

# Set to 'true' to allow .pyc and .pyo files without
# a source .py file to be detected as revisions in the
# versions/ directory
sourceless = false

# Version location specification
version_locations = %(here)s/versions

# Version path separator; As mentioned above, this is the character used to split
# version_locations paths. Default: os.pathsep.
# version_path_separator = :
# version_path_separator = ;
# version_path_separator = space
version_path_separator = os.pathsep

# The output encoding used when revision files
# are written from script.py.mako
output_encoding = utf-8

sqlalchemy.url = sqlite:///data/silent_steno.db

[post_write_hooks]
# Post-write hooks define scripts or Python functions that are run
# on newly generated revision scripts

# Logging configuration
[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""
        
        alembic_ini_path = self.migrations_dir / "alembic.ini"
        with open(alembic_ini_path, 'w') as f:
            f.write(alembic_ini_content.strip())
    
    def _initialize_alembic_environment(self):
        """Initialize Alembic environment"""
        try:
            # Create versions directory
            versions_dir = self.migrations_dir / "versions"
            versions_dir.mkdir(exist_ok=True)
            
            # Create env.py
            env_py_content = '''
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Import your models here
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.models import Base

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'''
            
            env_py_path = self.migrations_dir / "env.py"
            with open(env_py_path, 'w') as f:
                f.write(env_py_content.strip())
            
            # Create script.py.mako
            script_mako_content = '''"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade():
    ${upgrades if upgrades else "pass"}


def downgrade():
    ${downgrades if downgrades else "pass"}
'''
            
            script_mako_path = self.migrations_dir / "script.py.mako"
            with open(script_mako_path, 'w') as f:
                f.write(script_mako_content.strip())
                
        except Exception as e:
            logger.error(f"Failed to initialize Alembic environment: {e}")
            raise
    
    def create_migration(self, message: str, autogenerate: bool = True) -> str:
        """
        Create a new migration
        
        Args:
            message: Migration description
            autogenerate: Whether to auto-generate migration from model changes
            
        Returns:
            str: Migration revision ID
        """
        try:
            logger.info(f"Creating migration: {message}")
            
            # Generate migration
            if autogenerate:
                command.revision(self.alembic_config, message=message, autogenerate=True)
            else:
                command.revision(self.alembic_config, message=message)
            
            # Get the latest revision
            script_dir = ScriptDirectory.from_config(self.alembic_config)
            revision = script_dir.get_current_head()
            
            logger.info(f"Migration created: {revision}")
            return revision
            
        except Exception as e:
            logger.error(f"Failed to create migration: {e}")
            raise
    
    def run_migrations(self, target_revision: Optional[str] = None) -> bool:
        """
        Run pending migrations
        
        Args:
            target_revision: Target revision (None for latest)
            
        Returns:
            bool: True if migrations successful
        """
        try:
            logger.info("Running database migrations...")
            
            # Create backup before migration
            self._create_pre_migration_backup()
            
            # Run migrations
            if target_revision:
                command.upgrade(self.alembic_config, target_revision)
            else:
                command.upgrade(self.alembic_config, "head")
            
            logger.info("Migrations completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            # Attempt to restore backup
            self._restore_pre_migration_backup()
            return False
    
    def rollback_migration(self, target_revision: str) -> bool:
        """
        Rollback to a specific revision
        
        Args:
            target_revision: Target revision to rollback to
            
        Returns:
            bool: True if rollback successful
        """
        try:
            logger.info(f"Rolling back to revision: {target_revision}")
            
            # Create backup before rollback
            self._create_pre_migration_backup()
            
            # Perform rollback
            command.downgrade(self.alembic_config, target_revision)
            
            logger.info("Rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def get_migration_history(self) -> List[MigrationInfo]:
        """
        Get migration history
        
        Returns:
            List[MigrationInfo]: List of migration information
        """
        try:
            script_dir = ScriptDirectory.from_config(self.alembic_config)
            
            # Get all revisions
            revisions = []
            for revision in script_dir.walk_revisions():
                migration_info = MigrationInfo(
                    revision=revision.revision,
                    description=revision.doc or "No description",
                    status=MigrationStatus.PENDING,
                    created_at=datetime.now(timezone.utc)
                )
                revisions.append(migration_info)
            
            # Check which revisions are applied
            with self.database.get_session() as session:
                engine = session.bind
                context = MigrationContext.configure(engine.connect())
                current_revision = context.get_current_revision()
                
                # Mark applied revisions
                for migration in revisions:
                    if migration.revision == current_revision:
                        migration.status = MigrationStatus.COMPLETED
                        migration.applied_at = datetime.now(timezone.utc)
                        break
            
            return revisions
            
        except Exception as e:
            logger.error(f"Failed to get migration history: {e}")
            return []
    
    def check_migration_status(self) -> Dict[str, Any]:
        """
        Check current migration status
        
        Returns:
            Dict containing migration status information
        """
        try:
            script_dir = ScriptDirectory.from_config(self.alembic_config)
            
            # Get current revision
            with self.database.get_session() as session:
                engine = session.bind
                context = MigrationContext.configure(engine.connect())
                current_revision = context.get_current_revision()
            
            # Get head revision
            head_revision = script_dir.get_current_head()
            
            # Check if migrations are pending
            pending_migrations = []
            if current_revision != head_revision:
                for revision in script_dir.walk_revisions(head_revision, current_revision):
                    if revision.revision != current_revision:
                        pending_migrations.append({
                            "revision": revision.revision,
                            "description": revision.doc
                        })
            
            return {
                "current_revision": current_revision,
                "head_revision": head_revision,
                "up_to_date": current_revision == head_revision,
                "pending_migrations": pending_migrations,
                "migration_count": len(pending_migrations)
            }
            
        except Exception as e:
            logger.error(f"Failed to check migration status: {e}")
            return {
                "current_revision": None,
                "head_revision": None,
                "up_to_date": False,
                "pending_migrations": [],
                "migration_count": 0,
                "error": str(e)
            }
    
    def validate_migration(self, revision: str) -> bool:
        """
        Validate a migration before applying
        
        Args:
            revision: Migration revision to validate
            
        Returns:
            bool: True if migration is valid
        """
        try:
            # Basic validation - check if revision exists
            script_dir = ScriptDirectory.from_config(self.alembic_config)
            script = script_dir.get_revision(revision)
            
            if not script:
                logger.error(f"Migration revision not found: {revision}")
                return False
            
            # Additional validation could be added here
            # - Check for breaking changes
            # - Validate SQL syntax
            # - Check dependencies
            
            return True
            
        except Exception as e:
            logger.error(f"Migration validation failed: {e}")
            return False
    
    def _create_pre_migration_backup(self):
        """Create backup before migration"""
        try:
            if not self.database.config.backup_enabled:
                return
            
            backup_path = f"{self.database.config.database_path}.pre_migration_backup"
            shutil.copy2(self.database.config.database_path, backup_path)
            logger.info(f"Pre-migration backup created: {backup_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create pre-migration backup: {e}")
    
    def _restore_pre_migration_backup(self):
        """Restore backup after failed migration"""
        try:
            backup_path = f"{self.database.config.database_path}.pre_migration_backup"
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, self.database.config.database_path)
                logger.info("Pre-migration backup restored")
            
        except Exception as e:
            logger.error(f"Failed to restore pre-migration backup: {e}")
    
    def initialize_database(self) -> bool:
        """
        Initialize database with current schema
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing database schema...")
            
            # Create all tables from models
            from .models import create_models
            create_models(self.database.engine)
            
            # Stamp database with current revision
            command.stamp(self.alembic_config, "head")
            
            logger.info("Database schema initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            return False


# Convenience functions
def create_migration_manager(database: Database) -> MigrationManager:
    """Create migration manager instance"""
    return MigrationManager(database)


def run_migrations(database: Database) -> bool:
    """Run all pending migrations"""
    manager = MigrationManager(database)
    return manager.run_migrations()


def create_migration(database: Database, message: str) -> str:
    """Create a new migration"""
    manager = MigrationManager(database)
    return manager.create_migration(message)


def check_migration_status(database: Database) -> Dict[str, Any]:
    """Check migration status"""
    manager = MigrationManager(database)
    return manager.check_migration_status()


@contextmanager
def migration_context(database: Database):
    """Context manager for migration operations"""
    manager = MigrationManager(database)
    try:
        yield manager
    finally:
        # Cleanup if needed
        pass


if __name__ == "__main__":
    # Basic test when run directly
    print("Migration System Test")
    print("=" * 40)
    
    from .database import create_test_database
    
    try:
        # Create test database
        database = create_test_database("test_migrations.db")
        print("Test database created")
        
        # Create migration manager
        manager = MigrationManager(database, "test_migrations")
        print("Migration manager created")
        
        # Initialize database
        success = manager.initialize_database()
        print(f"Database initialization: {'SUCCESS' if success else 'FAILED'}")
        
        # Check status
        status = manager.check_migration_status()
        print(f"Migration status: {status}")
        
        # Cleanup
        database.close()
        print("Test completed")
        
    except Exception as e:
        print(f"Test failed: {e}")