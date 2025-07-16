#!/usr/bin/env python3

"""
Database Backup and Restore Manager for The Silent Steno

This module provides comprehensive backup and restore functionality for the
SQLite database with compression, verification, and scheduling capabilities.
It ensures data protection and disaster recovery for the meeting recorder.

Key features:
- Automated backup creation with compression
- Backup verification and integrity checking
- Scheduled backup with configurable intervals
- Restore functionality with validation
- Backup rotation and cleanup
- Cloud storage integration (future)
"""

import os
import gzip
import shutil
import sqlite3
import hashlib
import logging
import threading
import schedule
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone, timedelta
from pathlib import Path
from contextlib import contextmanager

from .database import Database, DatabaseConfig

# Set up logging
logger = logging.getLogger(__name__)


class BackupStatus(Enum):
    """Backup status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"
    RESTORED = "restored"


class BackupType(Enum):
    """Backup type enumeration"""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    PRE_MIGRATION = "pre_migration"
    EMERGENCY = "emergency"


@dataclass
class BackupConfig:
    """Backup configuration settings"""
    backup_directory: str = "backups"
    enable_compression: bool = True
    enable_verification: bool = True
    max_backup_count: int = 30
    backup_interval_hours: int = 6
    enable_scheduling: bool = True
    backup_name_pattern: str = "backup_%Y%m%d_%H%M%S"
    enable_cloud_sync: bool = False
    cloud_provider: str = "none"


@dataclass
class BackupInfo:
    """Backup information structure"""
    backup_id: str
    filename: str
    file_path: str
    backup_type: BackupType
    status: BackupStatus
    created_at: datetime
    file_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    checksum: str
    database_size_bytes: int
    session_count: int
    verification_passed: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class BackupManager:
    """
    Database backup and restore manager
    
    Provides automated backup creation, verification, and restore
    functionality with compression and scheduling support.
    """
    
    def __init__(self, database: Database, config: Optional[BackupConfig] = None):
        """
        Initialize backup manager
        
        Args:
            database: Database instance
            config: Backup configuration
        """
        self.database = database
        self.config = config or BackupConfig()
        self.backup_directory = Path(self.config.backup_directory)
        self._backup_history: List[BackupInfo] = []
        self._scheduler_thread: Optional[threading.Thread] = None
        self._scheduler_running = False
        self._lock = threading.Lock()
        
        # Ensure backup directory exists
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        
        # Load existing backup history
        self._load_backup_history()
        
        # Start scheduler if enabled
        if self.config.enable_scheduling:
            self.start_scheduler()
        
        logger.info(f"Backup manager initialized: {self.backup_directory}")
    
    def create_backup(self, backup_type: BackupType = BackupType.MANUAL, 
                     description: str = "") -> Optional[BackupInfo]:
        """
        Create a database backup
        
        Args:
            backup_type: Type of backup
            description: Backup description
            
        Returns:
            BackupInfo: Backup information or None if failed
        """
        try:
            with self._lock:
                logger.info(f"Creating {backup_type.value} backup...")
                
                # Generate backup filename
                timestamp = datetime.now(timezone.utc)
                backup_filename = timestamp.strftime(self.config.backup_name_pattern)
                if backup_type != BackupType.MANUAL:
                    backup_filename = f"{backup_type.value}_{backup_filename}"
                backup_filename += ".db"
                
                backup_path = self.backup_directory / backup_filename
                
                # Create backup info
                backup_info = BackupInfo(
                    backup_id=self._generate_backup_id(),
                    filename=backup_filename,
                    file_path=str(backup_path),
                    backup_type=backup_type,
                    status=BackupStatus.RUNNING,
                    created_at=timestamp,
                    file_size_bytes=0,
                    compressed_size_bytes=0,
                    compression_ratio=1.0,
                    checksum="",
                    database_size_bytes=0,
                    session_count=0,
                    verification_passed=False,
                    metadata={"description": description}
                )
                
                # Get database statistics
                backup_info.database_size_bytes = self._get_database_size()
                backup_info.session_count = self._get_session_count()
                
                # Create backup using SQLite backup API
                success = self._create_database_backup(backup_path)
                if not success:
                    backup_info.status = BackupStatus.FAILED
                    backup_info.error_message = "Database backup creation failed"
                    return backup_info
                
                # Get backup file size
                backup_info.file_size_bytes = backup_path.stat().st_size
                
                # Compress backup if enabled
                if self.config.enable_compression:
                    compressed_path = self._compress_backup(backup_path)
                    if compressed_path:
                        # Remove uncompressed file
                        backup_path.unlink()
                        backup_path = compressed_path
                        backup_info.file_path = str(backup_path)
                        backup_info.filename = backup_path.name
                        backup_info.compressed_size_bytes = backup_path.stat().st_size
                        backup_info.compression_ratio = backup_info.file_size_bytes / backup_info.compressed_size_bytes
                    else:
                        backup_info.compressed_size_bytes = backup_info.file_size_bytes
                else:
                    backup_info.compressed_size_bytes = backup_info.file_size_bytes
                
                # Calculate checksum
                backup_info.checksum = self._calculate_checksum(backup_path)
                
                # Verify backup if enabled
                if self.config.enable_verification:
                    backup_info.verification_passed = self.verify_backup(backup_info)
                else:
                    backup_info.verification_passed = True
                
                # Update status
                if backup_info.verification_passed:
                    backup_info.status = BackupStatus.COMPLETED
                else:
                    backup_info.status = BackupStatus.CORRUPTED
                
                # Add to history
                self._backup_history.append(backup_info)
                self._save_backup_history()
                
                # Clean old backups
                self._cleanup_old_backups()
                
                logger.info(f"Backup created successfully: {backup_filename}")
                return backup_info
                
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            if 'backup_info' in locals():
                backup_info.status = BackupStatus.FAILED
                backup_info.error_message = str(e)
                return backup_info
            return None
    
    def restore_backup(self, backup_info: BackupInfo, 
                      target_path: Optional[str] = None) -> bool:
        """
        Restore from backup
        
        Args:
            backup_info: Backup to restore
            target_path: Target database path (None for original)
            
        Returns:
            bool: True if restore successful
        """
        try:
            with self._lock:
                logger.info(f"Restoring backup: {backup_info.filename}")
                
                # Verify backup before restore
                if not self.verify_backup(backup_info):
                    logger.error("Backup verification failed, cannot restore")
                    return False
                
                # Determine target path
                if not target_path:
                    target_path = self.database.config.database_path
                
                # Create backup of current database
                current_backup = self.create_backup(BackupType.EMERGENCY, "Pre-restore backup")
                if not current_backup:
                    logger.warning("Could not create pre-restore backup")
                
                # Close database connections
                self.database.close()
                
                try:
                    backup_path = Path(backup_info.file_path)
                    
                    # Decompress if needed
                    if backup_path.suffix == ".gz":
                        decompressed_path = self._decompress_backup(backup_path)
                        if not decompressed_path:
                            return False
                        restore_source = decompressed_path
                    else:
                        restore_source = backup_path
                    
                    # Copy backup to target location
                    shutil.copy2(restore_source, target_path)
                    
                    # Clean up decompressed file if created
                    if backup_path.suffix == ".gz" and restore_source != backup_path:
                        restore_source.unlink()
                    
                    # Reinitialize database
                    self.database.initialize()
                    
                    # Verify restored database
                    if self.database.health_check():
                        logger.info("Database restore completed successfully")
                        return True
                    else:
                        logger.error("Restored database failed health check")
                        return False
                        
                except Exception as e:
                    logger.error(f"Restore operation failed: {e}")
                    # Attempt to restore from pre-restore backup
                    if current_backup:
                        logger.info("Attempting to restore from pre-restore backup")
                        # This is a simplified restore - in production might need more robust recovery
                        shutil.copy2(current_backup.file_path, target_path)
                    return False
                
        except Exception as e:
            logger.error(f"Backup restore failed: {e}")
            return False
    
    def verify_backup(self, backup_info: BackupInfo) -> bool:
        """
        Verify backup integrity
        
        Args:
            backup_info: Backup to verify
            
        Returns:
            bool: True if backup is valid
        """
        try:
            backup_path = Path(backup_info.file_path)
            
            # Check if file exists
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Verify checksum
            current_checksum = self._calculate_checksum(backup_path)
            if current_checksum != backup_info.checksum:
                logger.error(f"Backup checksum mismatch: {current_checksum} != {backup_info.checksum}")
                return False
            
            # Test SQLite database integrity
            if backup_path.suffix == ".gz":
                # Decompress temporarily for verification
                temp_path = self._decompress_backup(backup_path)
                if not temp_path:
                    return False
                test_path = temp_path
            else:
                test_path = backup_path
            
            try:
                # Test database connection and integrity
                conn = sqlite3.connect(str(test_path))
                cursor = conn.cursor()
                
                # Check database integrity
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()
                if result[0] != "ok":
                    logger.error(f"Database integrity check failed: {result[0]}")
                    return False
                
                # Test basic queries
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                if not tables:
                    logger.error("No tables found in backup database")
                    return False
                
                conn.close()
                
                # Clean up decompressed file if created
                if backup_path.suffix == ".gz" and test_path != backup_path:
                    test_path.unlink()
                
                logger.debug(f"Backup verification passed: {backup_info.filename}")
                return True
                
            except sqlite3.Error as e:
                logger.error(f"SQLite error during backup verification: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False
    
    def list_backups(self, backup_type: Optional[BackupType] = None) -> List[BackupInfo]:
        """
        List available backups
        
        Args:
            backup_type: Filter by backup type
            
        Returns:
            List[BackupInfo]: List of backups
        """
        if backup_type:
            return [b for b in self._backup_history if b.backup_type == backup_type]
        return self._backup_history.copy()
    
    def delete_backup(self, backup_info: BackupInfo) -> bool:
        """
        Delete a backup
        
        Args:
            backup_info: Backup to delete
            
        Returns:
            bool: True if deletion successful
        """
        try:
            backup_path = Path(backup_info.file_path)
            if backup_path.exists():
                backup_path.unlink()
            
            # Remove from history
            self._backup_history = [b for b in self._backup_history if b.backup_id != backup_info.backup_id]
            self._save_backup_history()
            
            logger.info(f"Backup deleted: {backup_info.filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup: {e}")
            return False
    
    def start_scheduler(self):
        """Start backup scheduler"""
        if self._scheduler_running:
            return
        
        self._scheduler_running = True
        
        # Schedule regular backups
        schedule.every(self.config.backup_interval_hours).hours.do(
            self._scheduled_backup
        )
        
        # Start scheduler thread
        self._scheduler_thread = threading.Thread(
            target=self._run_scheduler,
            daemon=True,
            name="BackupScheduler"
        )
        self._scheduler_thread.start()
        
        logger.info(f"Backup scheduler started (interval: {self.config.backup_interval_hours}h)")
    
    def stop_scheduler(self):
        """Stop backup scheduler"""
        self._scheduler_running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
        schedule.clear()
        logger.info("Backup scheduler stopped")
    
    def _scheduled_backup(self):
        """Create scheduled backup"""
        try:
            backup_info = self.create_backup(BackupType.SCHEDULED, "Automated scheduled backup")
            if backup_info and backup_info.status == BackupStatus.COMPLETED:
                logger.info("Scheduled backup completed successfully")
            else:
                logger.error("Scheduled backup failed")
        except Exception as e:
            logger.error(f"Scheduled backup error: {e}")
    
    def _run_scheduler(self):
        """Run backup scheduler loop"""
        while self._scheduler_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)
    
    def _create_database_backup(self, backup_path: Path) -> bool:
        """Create database backup using SQLite API"""
        try:
            # Use SQLite backup API for consistent backup
            source_conn = sqlite3.connect(self.database.config.database_path)
            backup_conn = sqlite3.connect(str(backup_path))
            
            # Perform backup
            source_conn.backup(backup_conn)
            
            # Close connections
            source_conn.close()
            backup_conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Database backup creation failed: {e}")
            return False
    
    def _compress_backup(self, backup_path: Path) -> Optional[Path]:
        """Compress backup file"""
        try:
            compressed_path = backup_path.with_suffix(backup_path.suffix + ".gz")
            
            with open(backup_path, 'rb') as src, gzip.open(compressed_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            
            return compressed_path
            
        except Exception as e:
            logger.error(f"Backup compression failed: {e}")
            return None
    
    def _decompress_backup(self, compressed_path: Path) -> Optional[Path]:
        """Decompress backup file"""
        try:
            decompressed_path = compressed_path.with_suffix("")
            
            with gzip.open(compressed_path, 'rb') as src, open(decompressed_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            
            return decompressed_path
            
        except Exception as e:
            logger.error(f"Backup decompression failed: {e}")
            return None
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Checksum calculation failed: {e}")
            return ""
    
    def _get_database_size(self) -> int:
        """Get database file size"""
        try:
            return Path(self.database.config.database_path).stat().st_size
        except:
            return 0
    
    def _get_session_count(self) -> int:
        """Get number of sessions in database"""
        try:
            with self.database.get_session() as session:
                result = session.execute("SELECT COUNT(*) FROM sessions")
                return result.scalar() or 0
        except:
            return 0
    
    def _generate_backup_id(self) -> str:
        """Generate unique backup ID"""
        return f"backup_{int(time.time())}_{os.getpid()}"
    
    def _cleanup_old_backups(self):
        """Clean up old backups based on retention policy"""
        try:
            if len(self._backup_history) <= self.config.max_backup_count:
                return
            
            # Sort by creation date
            sorted_backups = sorted(self._backup_history, key=lambda b: b.created_at)
            
            # Remove oldest backups
            backups_to_remove = sorted_backups[:-self.config.max_backup_count]
            
            for backup in backups_to_remove:
                if backup.backup_type != BackupType.EMERGENCY:  # Keep emergency backups
                    self.delete_backup(backup)
                    
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
    
    def _load_backup_history(self):
        """Load backup history from metadata file"""
        try:
            history_file = self.backup_directory / "backup_history.json"
            if history_file.exists():
                import json
                with open(history_file, 'r') as f:
                    data = json.load(f)
                
                self._backup_history = []
                for item in data:
                    backup_info = BackupInfo(
                        backup_id=item['backup_id'],
                        filename=item['filename'],
                        file_path=item['file_path'],
                        backup_type=BackupType(item['backup_type']),
                        status=BackupStatus(item['status']),
                        created_at=datetime.fromisoformat(item['created_at']),
                        file_size_bytes=item['file_size_bytes'],
                        compressed_size_bytes=item['compressed_size_bytes'],
                        compression_ratio=item['compression_ratio'],
                        checksum=item['checksum'],
                        database_size_bytes=item['database_size_bytes'],
                        session_count=item['session_count'],
                        verification_passed=item['verification_passed'],
                        error_message=item.get('error_message'),
                        metadata=item.get('metadata', {})
                    )
                    self._backup_history.append(backup_info)
                    
        except Exception as e:
            logger.warning(f"Could not load backup history: {e}")
            self._backup_history = []
    
    def _save_backup_history(self):
        """Save backup history to metadata file"""
        try:
            history_file = self.backup_directory / "backup_history.json"
            import json
            
            data = []
            for backup in self._backup_history:
                data.append({
                    'backup_id': backup.backup_id,
                    'filename': backup.filename,
                    'file_path': backup.file_path,
                    'backup_type': backup.backup_type.value,
                    'status': backup.status.value,
                    'created_at': backup.created_at.isoformat(),
                    'file_size_bytes': backup.file_size_bytes,
                    'compressed_size_bytes': backup.compressed_size_bytes,
                    'compression_ratio': backup.compression_ratio,
                    'checksum': backup.checksum,
                    'database_size_bytes': backup.database_size_bytes,
                    'session_count': backup.session_count,
                    'verification_passed': backup.verification_passed,
                    'error_message': backup.error_message,
                    'metadata': backup.metadata or {}
                })
            
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Could not save backup history: {e}")
    
    def get_backup_statistics(self) -> Dict[str, Any]:
        """Get backup statistics"""
        completed_backups = [b for b in self._backup_history if b.status == BackupStatus.COMPLETED]
        
        return {
            "total_backups": len(self._backup_history),
            "completed_backups": len(completed_backups),
            "failed_backups": len([b for b in self._backup_history if b.status == BackupStatus.FAILED]),
            "total_backup_size_bytes": sum(b.compressed_size_bytes for b in completed_backups),
            "average_compression_ratio": sum(b.compression_ratio for b in completed_backups) / len(completed_backups) if completed_backups else 1.0,
            "latest_backup": completed_backups[-1].created_at.isoformat() if completed_backups else None,
            "scheduler_running": self._scheduler_running,
            "backup_interval_hours": self.config.backup_interval_hours,
            "backup_directory": str(self.backup_directory)
        }


# Convenience functions
def create_backup_manager(database: Database, config: Optional[BackupConfig] = None) -> BackupManager:
    """Create backup manager instance"""
    return BackupManager(database, config)


def create_backup(database: Database, description: str = "") -> Optional[BackupInfo]:
    """Create a manual backup"""
    manager = BackupManager(database)
    return manager.create_backup(BackupType.MANUAL, description)


def verify_backup(backup_info: BackupInfo, database: Database) -> bool:
    """Verify a backup"""
    manager = BackupManager(database)
    return manager.verify_backup(backup_info)


@contextmanager
def backup_context(database: Database):
    """Context manager for backup operations"""
    manager = BackupManager(database)
    try:
        yield manager
    finally:
        manager.stop_scheduler()


if __name__ == "__main__":
    # Basic test when run directly
    print("Backup Manager Test")
    print("=" * 40)
    
    from .database import create_test_database
    
    try:
        # Create test database
        database = create_test_database("test_backup.db")
        print("Test database created")
        
        # Create backup manager
        config = BackupConfig(
            backup_directory="test_backups",
            enable_scheduling=False,
            max_backup_count=5
        )
        manager = BackupManager(database, config)
        print("Backup manager created")
        
        # Create backup
        backup_info = manager.create_backup(BackupType.MANUAL, "Test backup")
        if backup_info:
            print(f"Backup created: {backup_info.filename}")
            print(f"Backup size: {backup_info.compressed_size_bytes} bytes")
            print(f"Compression ratio: {backup_info.compression_ratio:.2f}")
            
            # Verify backup
            verified = manager.verify_backup(backup_info)
            print(f"Backup verification: {'PASSED' if verified else 'FAILED'}")
        
        # Get statistics
        stats = manager.get_backup_statistics()
        print(f"Backup statistics: {stats}")
        
        # Cleanup
        database.close()
        print("Test completed")
        
    except Exception as e:
        print(f"Test failed: {e}")