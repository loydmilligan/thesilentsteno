#!/usr/bin/env python3

"""
Database Connection and Session Management for The Silent Steno

This module provides the core database infrastructure using SQLite with SQLAlchemy ORM.
It handles connection management, session creation, and transaction handling for
the meeting recorder application.

Key features:
- SQLite database with WAL mode for concurrent access
- SQLAlchemy ORM for object-relational mapping
- Connection pooling and session management
- Transaction context managers
- Database initialization and configuration
- Health monitoring and diagnostics
"""

import os
import sqlite3
import logging
import threading
from typing import Optional, Dict, Any, Generator, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import StaticPool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseState(Enum):
    """Database connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    database_path: str = "data/silent_steno.db"
    enable_wal_mode: bool = True
    enable_foreign_keys: bool = True
    connection_timeout: int = 30
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    echo_queries: bool = False
    backup_enabled: bool = True
    backup_interval_hours: int = 6


class Database:
    """
    Main database connection and session management class
    
    Provides SQLite database connectivity with SQLAlchemy ORM,
    connection pooling, and transaction management.
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """Initialize database manager"""
        self.config = config or DatabaseConfig()
        self.engine: Optional[Engine] = None
        self.session_factory: Optional[sessionmaker] = None
        self.state = DatabaseState.DISCONNECTED
        self._lock = threading.Lock()
        
        # Connection statistics
        self.connection_count = 0
        self.transaction_count = 0
        self.error_count = 0
        
        logger.info("Database manager initialized")
    
    def initialize(self) -> bool:
        """
        Initialize database connection and engine
        
        Returns:
            bool: True if initialization successful
        """
        try:
            with self._lock:
                self.state = DatabaseState.CONNECTING
                
                # Create database directory if it doesn't exist
                db_dir = os.path.dirname(self.config.database_path)
                if db_dir and not os.path.exists(db_dir):
                    os.makedirs(db_dir, exist_ok=True)
                
                # Create SQLAlchemy engine with SQLite
                connection_string = f"sqlite:///{self.config.database_path}"
                
                self.engine = create_engine(
                    connection_string,
                    poolclass=StaticPool,
                    connect_args={
                        "timeout": self.config.connection_timeout,
                        "check_same_thread": False  # Allow multi-threading
                    },
                    echo=self.config.echo_queries
                )
                
                # Configure SQLite pragmas
                self._configure_sqlite_pragmas()
                
                # Create session factory
                self.session_factory = sessionmaker(
                    bind=self.engine,
                    expire_on_commit=False
                )
                
                # Test connection
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                
                self.state = DatabaseState.CONNECTED
                logger.info(f"Database initialized successfully: {self.config.database_path}")
                return True
                
        except Exception as e:
            self.state = DatabaseState.ERROR
            self.error_count += 1
            logger.error(f"Failed to initialize database: {e}")
            return False
    
    def _configure_sqlite_pragmas(self):
        """Configure SQLite pragmas for optimal performance"""
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            
            # Enable foreign key constraints
            if self.config.enable_foreign_keys:
                cursor.execute("PRAGMA foreign_keys=ON")
            
            # Enable WAL mode for better concurrency
            if self.config.enable_wal_mode:
                cursor.execute("PRAGMA journal_mode=WAL")
            
            # Performance optimizations
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
            
            cursor.close()
    
    def create_session(self) -> Optional[Session]:
        """
        Create a new database session
        
        Returns:
            Session: SQLAlchemy session or None if failed
        """
        if not self.session_factory:
            logger.error("Database not initialized")
            return None
        
        try:
            session = self.session_factory()
            self.connection_count += 1
            return session
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to create session: {e}")
            return None
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions
        
        Yields:
            Session: SQLAlchemy session with automatic cleanup
        """
        session = self.create_session()
        if not session:
            raise RuntimeError("Could not create database session")
        
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    @contextmanager
    def transaction(self) -> Generator[Session, None, None]:
        """
        Context manager for database transactions
        
        Yields:
            Session: SQLAlchemy session with automatic transaction handling
        """
        with self.get_session() as session:
            try:
                yield session
                session.commit()
                self.transaction_count += 1
            except Exception:
                session.rollback()
                self.error_count += 1
                raise
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute a raw SQL query
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query result
        """
        with self.get_session() as session:
            try:
                result = session.execute(text(query), params or {})
                session.commit()
                return result
            except Exception as e:
                session.rollback()
                logger.error(f"Query execution failed: {e}")
                raise
    
    def close(self):
        """Close database connections and cleanup"""
        try:
            if self.engine:
                self.engine.dispose()
                self.engine = None
            
            self.session_factory = None
            self.state = DatabaseState.DISCONNECTED
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get database status and statistics
        
        Returns:
            Dict containing database status information
        """
        return {
            "state": self.state.value,
            "database_path": self.config.database_path,
            "connection_count": self.connection_count,
            "transaction_count": self.transaction_count,
            "error_count": self.error_count,
            "config": {
                "wal_mode": self.config.enable_wal_mode,
                "foreign_keys": self.config.enable_foreign_keys,
                "pool_size": self.config.pool_size,
                "backup_enabled": self.config.backup_enabled
            }
        }
    
    def health_check(self) -> bool:
        """
        Perform database health check
        
        Returns:
            bool: True if database is healthy
        """
        try:
            if self.state != DatabaseState.CONNECTED:
                return False
            
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                return True
                
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            return False


class SessionManager:
    """
    Higher-level session management with automatic retry and error handling
    """
    
    def __init__(self, database: Database, max_retries: int = 3):
        """Initialize session manager"""
        self.database = database
        self.max_retries = max_retries
    
    def with_retry(self, operation: Callable[[Session], Any], *args, **kwargs) -> Any:
        """
        Execute database operation with automatic retry
        
        Args:
            operation: Function that takes a session and returns a result
            *args, **kwargs: Arguments passed to operation
            
        Returns:
            Operation result
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                with self.database.transaction() as session:
                    return operation(session, *args, **kwargs)
                    
            except SQLAlchemyError as e:
                last_exception = e
                logger.warning(f"Database operation failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries - 1:
                    # Wait before retry (exponential backoff)
                    import time
                    time.sleep(2 ** attempt)
        
        # All retries failed
        logger.error(f"Database operation failed after {self.max_retries} attempts")
        raise last_exception


# Global database instance
_database_instance: Optional[Database] = None
_database_lock = threading.Lock()


def get_database() -> Database:
    """
    Get global database instance (singleton pattern)
    
    Returns:
        Database: Global database instance
    """
    global _database_instance
    
    if _database_instance is None:
        with _database_lock:
            if _database_instance is None:
                _database_instance = Database()
                if not _database_instance.initialize():
                    raise RuntimeError("Failed to initialize database")
    
    return _database_instance


def create_database(config: Optional[DatabaseConfig] = None) -> Database:
    """
    Create a new database instance with configuration
    
    Args:
        config: Database configuration
        
    Returns:
        Database: Configured database instance
    """
    database = Database(config)
    if not database.initialize():
        raise RuntimeError("Failed to initialize database")
    return database


def create_session() -> Optional[Session]:
    """
    Create a database session using global instance
    
    Returns:
        Session: SQLAlchemy session
    """
    return get_database().create_session()


@contextmanager
def database_context() -> Generator[Database, None, None]:
    """
    Context manager for database operations
    
    Yields:
        Database: Database instance with automatic cleanup
    """
    database = get_database()
    try:
        yield database
    finally:
        # Database cleanup is handled by the global instance
        pass


# Factory functions for different database configurations
def create_memory_database() -> Database:
    """Create in-memory database for testing"""
    config = DatabaseConfig(
        database_path=":memory:",
        enable_wal_mode=False,  # Not supported in memory
        echo_queries=True
    )
    return create_database(config)


def create_test_database(path: str = "test_silent_steno.db") -> Database:
    """Create test database with specific configuration"""
    config = DatabaseConfig(
        database_path=path,
        echo_queries=True,
        backup_enabled=False
    )
    return create_database(config)


def create_production_database(path: str = "data/silent_steno.db") -> Database:
    """Create production database with optimal settings"""
    config = DatabaseConfig(
        database_path=path,
        enable_wal_mode=True,
        enable_foreign_keys=True,
        pool_size=20,
        backup_enabled=True,
        echo_queries=False
    )
    return create_database(config)


if __name__ == "__main__":
    # Basic test when run directly
    print("Database Module Test")
    print("=" * 40)
    
    # Test database creation and basic operations
    try:
        database = create_test_database("test.db")
        print(f"Database created: {database.config.database_path}")
        
        # Test session creation
        with database.get_session() as session:
            result = session.execute(text("SELECT 1 as test"))
            print(f"Test query result: {result.fetchone()}")
        
        # Test transaction
        with database.transaction() as session:
            session.execute(text("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY)"))
            print("Test table created")
        
        # Print status
        status = database.get_status()
        print(f"Database status: {status}")
        
        # Health check
        healthy = database.health_check()
        print(f"Health check: {'PASS' if healthy else 'FAIL'}")
        
        # Cleanup
        database.close()
        os.remove("test.db")
        print("Test completed successfully")
        
    except Exception as e:
        print(f"Test failed: {e}")