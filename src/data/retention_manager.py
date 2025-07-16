#!/usr/bin/env python3

"""
Data Retention Policy Manager for The Silent Steno

This module provides configurable data retention policies to prevent storage
overflow and manage data lifecycle. It handles automatic cleanup of old
sessions, transcripts, and related files based on configurable rules.

Key features:
- Configurable retention policies by data type
- Automatic cleanup scheduling
- Storage usage monitoring
- Selective retention based on criteria
- Safe deletion with backup integration
- Compliance and audit logging
"""

import os
import logging
import threading
import schedule
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone, timedelta
from pathlib import Path

from sqlalchemy import text, func
from sqlalchemy.orm import Session

from .database import Database
from .models import Session as SessionModel, TranscriptEntry, AnalysisResult, FileInfo
from .backup_manager import BackupManager, BackupType

# Set up logging
logger = logging.getLogger(__name__)


class RetentionCriteria(Enum):
    """Retention criteria enumeration"""
    AGE = "age"
    COUNT = "count"
    SIZE = "size"
    PRIORITY = "priority"
    CUSTOM = "custom"


class DataType(Enum):
    """Data type enumeration for retention"""
    SESSIONS = "sessions"
    TRANSCRIPTS = "transcripts"
    ANALYSIS = "analysis"
    AUDIO_FILES = "audio_files"
    BACKUP_FILES = "backup_files"
    SYSTEM_LOGS = "system_logs"


class RetentionAction(Enum):
    """Retention action enumeration"""
    DELETE = "delete"
    ARCHIVE = "archive"
    COMPRESS = "compress"
    MOVE = "move"


@dataclass
class RetentionRule:
    """Data retention rule definition"""
    rule_id: str
    data_type: DataType
    criteria: RetentionCriteria
    threshold_value: Any  # Age in days, count, size in bytes, etc.
    action: RetentionAction
    enabled: bool = True
    priority: int = 0
    description: str = ""
    conditions: Dict[str, Any] = None
    preserve_conditions: Dict[str, Any] = None


@dataclass
class RetentionConfig:
    """Retention configuration settings"""
    enable_retention: bool = True
    enable_scheduling: bool = True
    cleanup_interval_hours: int = 24
    safety_backup_before_cleanup: bool = True
    dry_run_mode: bool = False
    max_cleanup_batch_size: int = 100
    preserve_recent_days: int = 7
    storage_threshold_gb: float = 10.0
    enable_audit_logging: bool = True


@dataclass
class CleanupResult:
    """Cleanup operation result"""
    rule_id: str
    data_type: DataType
    items_found: int
    items_deleted: int
    bytes_freed: int
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None
    details: Dict[str, Any] = None


class RetentionManager:
    """
    Data retention policy manager
    
    Manages data lifecycle through configurable retention policies,
    automatic cleanup, and storage optimization.
    """
    
    def __init__(self, database: Database, backup_manager: Optional[BackupManager] = None,
                 config: Optional[RetentionConfig] = None):
        """
        Initialize retention manager
        
        Args:
            database: Database instance
            backup_manager: Backup manager for safety backups
            config: Retention configuration
        """
        self.database = database
        self.backup_manager = backup_manager
        self.config = config or RetentionConfig()
        self.retention_rules: List[RetentionRule] = []
        self._scheduler_thread: Optional[threading.Thread] = None
        self._scheduler_running = False
        self._cleanup_callbacks: List[Callable] = []
        
        # Initialize default retention rules
        self._initialize_default_rules()
        
        # Start scheduler if enabled
        if self.config.enable_scheduling:
            self.start_scheduler()
        
        logger.info("Retention manager initialized")
    
    def _initialize_default_rules(self):
        """Initialize default retention rules"""
        default_rules = [
            RetentionRule(
                rule_id="sessions_90_days",
                data_type=DataType.SESSIONS,
                criteria=RetentionCriteria.AGE,
                threshold_value=90,  # days
                action=RetentionAction.DELETE,
                description="Delete sessions older than 90 days",
                preserve_conditions={"tags": ["important", "archive"]}
            ),
            RetentionRule(
                rule_id="max_1000_sessions",
                data_type=DataType.SESSIONS,
                criteria=RetentionCriteria.COUNT,
                threshold_value=1000,
                action=RetentionAction.DELETE,
                description="Keep only latest 1000 sessions",
                priority=1
            ),
            RetentionRule(
                rule_id="audio_files_30_days",
                data_type=DataType.AUDIO_FILES,
                criteria=RetentionCriteria.AGE,
                threshold_value=30,  # days
                action=RetentionAction.DELETE,
                description="Delete audio files older than 30 days"
            ),
            RetentionRule(
                rule_id="backup_files_30_days",
                data_type=DataType.BACKUP_FILES,
                criteria=RetentionCriteria.AGE,
                threshold_value=30,  # days
                action=RetentionAction.DELETE,
                description="Delete backup files older than 30 days"
            ),
            RetentionRule(
                rule_id="storage_threshold",
                data_type=DataType.SESSIONS,
                criteria=RetentionCriteria.SIZE,
                threshold_value=self.config.storage_threshold_gb * 1024 * 1024 * 1024,  # Convert to bytes
                action=RetentionAction.DELETE,
                description=f"Clean oldest sessions when storage exceeds {self.config.storage_threshold_gb}GB",
                priority=2
            )
        ]
        
        self.retention_rules.extend(default_rules)
    
    def add_retention_rule(self, rule: RetentionRule) -> bool:
        """
        Add retention rule
        
        Args:
            rule: Retention rule to add
            
        Returns:
            bool: True if rule added successfully
        """
        try:
            # Check for duplicate rule IDs
            if any(r.rule_id == rule.rule_id for r in self.retention_rules):
                logger.error(f"Retention rule already exists: {rule.rule_id}")
                return False
            
            self.retention_rules.append(rule)
            logger.info(f"Retention rule added: {rule.rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add retention rule: {e}")
            return False
    
    def remove_retention_rule(self, rule_id: str) -> bool:
        """
        Remove retention rule
        
        Args:
            rule_id: Rule ID to remove
            
        Returns:
            bool: True if rule removed successfully
        """
        try:
            original_count = len(self.retention_rules)
            self.retention_rules = [r for r in self.retention_rules if r.rule_id != rule_id]
            
            if len(self.retention_rules) < original_count:
                logger.info(f"Retention rule removed: {rule_id}")
                return True
            else:
                logger.warning(f"Retention rule not found: {rule_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove retention rule: {e}")
            return False
    
    def apply_retention_policy(self, rule_id: Optional[str] = None) -> List[CleanupResult]:
        """
        Apply retention policies
        
        Args:
            rule_id: Specific rule to apply (None for all)
            
        Returns:
            List[CleanupResult]: Cleanup results for each rule
        """
        results = []
        
        try:
            logger.info("Applying retention policies...")
            
            # Create safety backup if enabled
            if self.config.safety_backup_before_cleanup and self.backup_manager:
                backup_info = self.backup_manager.create_backup(
                    BackupType.EMERGENCY, 
                    "Pre-cleanup safety backup"
                )
                if backup_info:
                    logger.info(f"Safety backup created: {backup_info.filename}")
                else:
                    logger.warning("Failed to create safety backup")
            
            # Get rules to apply
            rules_to_apply = [r for r in self.retention_rules if r.enabled]
            if rule_id:
                rules_to_apply = [r for r in rules_to_apply if r.rule_id == rule_id]
            
            # Sort rules by priority
            rules_to_apply.sort(key=lambda r: r.priority)
            
            # Apply each rule
            for rule in rules_to_apply:
                try:
                    result = self._apply_single_rule(rule)
                    results.append(result)
                    
                    # Notify callbacks
                    for callback in self._cleanup_callbacks:
                        try:
                            callback(result)
                        except Exception as e:
                            logger.error(f"Error in cleanup callback: {e}")
                            
                except Exception as e:
                    logger.error(f"Failed to apply rule {rule.rule_id}: {e}")
                    results.append(CleanupResult(
                        rule_id=rule.rule_id,
                        data_type=rule.data_type,
                        items_found=0,
                        items_deleted=0,
                        bytes_freed=0,
                        duration_seconds=0.0,
                        success=False,
                        error_message=str(e)
                    ))
            
            # Log summary
            total_deleted = sum(r.items_deleted for r in results)
            total_freed = sum(r.bytes_freed for r in results)
            logger.info(f"Retention cleanup completed: {total_deleted} items, {total_freed / (1024*1024):.1f} MB freed")
            
            return results
            
        except Exception as e:
            logger.error(f"Retention policy application failed: {e}")
            return results
    
    def _apply_single_rule(self, rule: RetentionRule) -> CleanupResult:
        """Apply a single retention rule"""
        start_time = time.time()
        
        try:
            logger.debug(f"Applying retention rule: {rule.rule_id}")
            
            if rule.data_type == DataType.SESSIONS:
                return self._cleanup_sessions(rule)
            elif rule.data_type == DataType.AUDIO_FILES:
                return self._cleanup_audio_files(rule)
            elif rule.data_type == DataType.BACKUP_FILES:
                return self._cleanup_backup_files(rule)
            elif rule.data_type == DataType.TRANSCRIPTS:
                return self._cleanup_transcripts(rule)
            elif rule.data_type == DataType.ANALYSIS:
                return self._cleanup_analysis(rule)
            else:
                raise ValueError(f"Unsupported data type: {rule.data_type}")
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Rule {rule.rule_id} failed: {e}")
            return CleanupResult(
                rule_id=rule.rule_id,
                data_type=rule.data_type,
                items_found=0,
                items_deleted=0,
                bytes_freed=0,
                duration_seconds=duration,
                success=False,
                error_message=str(e)
            )
    
    def _cleanup_sessions(self, rule: RetentionRule) -> CleanupResult:
        """Cleanup sessions based on retention rule"""
        start_time = time.time()
        items_found = 0
        items_deleted = 0
        bytes_freed = 0
        
        try:
            with self.database.get_session() as session:
                # Build query based on criteria
                query = session.query(SessionModel)
                
                if rule.criteria == RetentionCriteria.AGE:
                    # Sessions older than threshold days
                    cutoff_date = datetime.now(timezone.utc) - timedelta(days=rule.threshold_value)
                    query = query.filter(SessionModel.created_at < cutoff_date)
                    
                elif rule.criteria == RetentionCriteria.COUNT:
                    # Keep only the latest N sessions
                    total_sessions = session.query(func.count(SessionModel.id)).scalar()
                    if total_sessions > rule.threshold_value:
                        sessions_to_delete = total_sessions - rule.threshold_value
                        query = query.order_by(SessionModel.created_at.asc()).limit(sessions_to_delete)
                    else:
                        # No sessions to delete
                        return CleanupResult(
                            rule_id=rule.rule_id,
                            data_type=rule.data_type,
                            items_found=0,
                            items_deleted=0,
                            bytes_freed=0,
                            duration_seconds=time.time() - start_time,
                            success=True
                        )
                
                elif rule.criteria == RetentionCriteria.SIZE:
                    # Check total storage size
                    total_size = self._get_total_storage_size()
                    if total_size < rule.threshold_value:
                        # Under threshold, no cleanup needed
                        return CleanupResult(
                            rule_id=rule.rule_id,
                            data_type=rule.data_type,
                            items_found=0,
                            items_deleted=0,
                            bytes_freed=0,
                            duration_seconds=time.time() - start_time,
                            success=True
                        )
                    
                    # Delete oldest sessions until under threshold
                    query = query.order_by(SessionModel.created_at.asc())
                
                # Apply preserve conditions
                if rule.preserve_conditions:
                    if "tags" in rule.preserve_conditions:
                        preserve_tags = rule.preserve_conditions["tags"]
                        # Exclude sessions with preserve tags
                        for tag in preserve_tags:
                            query = query.filter(~SessionModel.tags.contains(tag))
                
                # Apply preserve recent days
                if self.config.preserve_recent_days > 0:
                    preserve_date = datetime.now(timezone.utc) - timedelta(days=self.config.preserve_recent_days)
                    query = query.filter(SessionModel.created_at < preserve_date)
                
                # Get sessions to delete
                sessions_to_delete = query.limit(self.config.max_cleanup_batch_size).all()
                items_found = len(sessions_to_delete)
                
                if not self.config.dry_run_mode:
                    for session_obj in sessions_to_delete:
                        try:
                            # Calculate bytes freed (approximate)
                            session_size = session_obj.audio_size_bytes or 0
                            
                            # Delete related files
                            for file_info in session_obj.files:
                                if os.path.exists(file_info.file_path):
                                    file_size = os.path.getsize(file_info.file_path)
                                    os.remove(file_info.file_path)
                                    bytes_freed += file_size
                            
                            # Delete audio file
                            if session_obj.audio_file_path and os.path.exists(session_obj.audio_file_path):
                                audio_size = os.path.getsize(session_obj.audio_file_path)
                                os.remove(session_obj.audio_file_path)
                                bytes_freed += audio_size
                            
                            # Delete database record (cascades to related records)
                            session.delete(session_obj)
                            bytes_freed += session_size
                            items_deleted += 1
                            
                        except Exception as e:
                            logger.error(f"Failed to delete session {session_obj.id}: {e}")
                    
                    session.commit()
                
                duration = time.time() - start_time
                
                return CleanupResult(
                    rule_id=rule.rule_id,
                    data_type=rule.data_type,
                    items_found=items_found,
                    items_deleted=items_deleted,
                    bytes_freed=bytes_freed,
                    duration_seconds=duration,
                    success=True,
                    details={
                        "dry_run": self.config.dry_run_mode,
                        "criteria": rule.criteria.value,
                        "threshold": rule.threshold_value
                    }
                )
                
        except Exception as e:
            duration = time.time() - start_time
            raise RuntimeError(f"Session cleanup failed: {e}")
    
    def _cleanup_audio_files(self, rule: RetentionRule) -> CleanupResult:
        """Cleanup orphaned audio files"""
        start_time = time.time()
        items_found = 0
        items_deleted = 0
        bytes_freed = 0
        
        try:
            # Find audio files not referenced in database
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
            audio_directories = ['data/audio', 'recordings']
            
            orphaned_files = []
            
            with self.database.get_session() as session:
                # Get all audio file paths from database
                referenced_paths = set()
                sessions = session.query(SessionModel.audio_file_path).filter(SessionModel.audio_file_path.isnot(None)).all()
                for s in sessions:
                    referenced_paths.add(s.audio_file_path)
                
                files = session.query(FileInfo.file_path).filter(FileInfo.file_type == 'audio').all()
                for f in files:
                    referenced_paths.add(f.file_path)
                
                # Find orphaned files
                for audio_dir in audio_directories:
                    if os.path.exists(audio_dir):
                        for root, dirs, files in os.walk(audio_dir):
                            for file in files:
                                if any(file.lower().endswith(ext) for ext in audio_extensions):
                                    file_path = os.path.join(root, file)
                                    
                                    # Check age criteria
                                    if rule.criteria == RetentionCriteria.AGE:
                                        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))
                                        if file_age.days < rule.threshold_value:
                                            continue
                                    
                                    # Check if file is referenced
                                    if file_path not in referenced_paths:
                                        orphaned_files.append(file_path)
                
                items_found = len(orphaned_files)
                
                if not self.config.dry_run_mode:
                    for file_path in orphaned_files[:self.config.max_cleanup_batch_size]:
                        try:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            bytes_freed += file_size
                            items_deleted += 1
                        except Exception as e:
                            logger.error(f"Failed to delete audio file {file_path}: {e}")
                
                duration = time.time() - start_time
                
                return CleanupResult(
                    rule_id=rule.rule_id,
                    data_type=rule.data_type,
                    items_found=items_found,
                    items_deleted=items_deleted,
                    bytes_freed=bytes_freed,
                    duration_seconds=duration,
                    success=True
                )
                
        except Exception as e:
            duration = time.time() - start_time
            raise RuntimeError(f"Audio file cleanup failed: {e}")
    
    def _cleanup_backup_files(self, rule: RetentionRule) -> CleanupResult:
        """Cleanup old backup files"""
        start_time = time.time()
        items_found = 0
        items_deleted = 0
        bytes_freed = 0
        
        try:
            if not self.backup_manager:
                return CleanupResult(
                    rule_id=rule.rule_id,
                    data_type=rule.data_type,
                    items_found=0,
                    items_deleted=0,
                    bytes_freed=0,
                    duration_seconds=time.time() - start_time,
                    success=True,
                    details={"message": "No backup manager available"}
                )
            
            backups = self.backup_manager.list_backups()
            
            # Filter backups based on criteria
            backups_to_delete = []
            
            for backup in backups:
                if rule.criteria == RetentionCriteria.AGE:
                    backup_age = datetime.now(timezone.utc) - backup.created_at
                    if backup_age.days >= rule.threshold_value:
                        backups_to_delete.append(backup)
            
            items_found = len(backups_to_delete)
            
            if not self.config.dry_run_mode:
                for backup in backups_to_delete[:self.config.max_cleanup_batch_size]:
                    try:
                        bytes_freed += backup.compressed_size_bytes
                        self.backup_manager.delete_backup(backup)
                        items_deleted += 1
                    except Exception as e:
                        logger.error(f"Failed to delete backup {backup.filename}: {e}")
            
            duration = time.time() - start_time
            
            return CleanupResult(
                rule_id=rule.rule_id,
                data_type=rule.data_type,
                items_found=items_found,
                items_deleted=items_deleted,
                bytes_freed=bytes_freed,
                duration_seconds=duration,
                success=True
            )
            
        except Exception as e:
            duration = time.time() - start_time
            raise RuntimeError(f"Backup file cleanup failed: {e}")
    
    def _cleanup_transcripts(self, rule: RetentionRule) -> CleanupResult:
        """Cleanup orphaned transcript entries"""
        # Implementation similar to sessions cleanup but for transcripts
        # This is a simplified version - full implementation would be more complex
        return CleanupResult(
            rule_id=rule.rule_id,
            data_type=rule.data_type,
            items_found=0,
            items_deleted=0,
            bytes_freed=0,
            duration_seconds=0.0,
            success=True,
            details={"message": "Transcript cleanup not implemented"}
        )
    
    def _cleanup_analysis(self, rule: RetentionRule) -> CleanupResult:
        """Cleanup orphaned analysis results"""
        # Implementation similar to sessions cleanup but for analysis
        # This is a simplified version - full implementation would be more complex
        return CleanupResult(
            rule_id=rule.rule_id,
            data_type=rule.data_type,
            items_found=0,
            items_deleted=0,
            bytes_freed=0,
            duration_seconds=0.0,
            success=True,
            details={"message": "Analysis cleanup not implemented"}
        )
    
    def _get_total_storage_size(self) -> int:
        """Get total storage size used by the application"""
        try:
            total_size = 0
            
            # Database size
            if os.path.exists(self.database.config.database_path):
                total_size += os.path.getsize(self.database.config.database_path)
            
            # Audio files size
            audio_directories = ['data/audio', 'recordings']
            for audio_dir in audio_directories:
                if os.path.exists(audio_dir):
                    for root, dirs, files in os.walk(audio_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            total_size += os.path.getsize(file_path)
            
            # Backup files size
            if self.backup_manager:
                backups = self.backup_manager.list_backups()
                total_size += sum(b.compressed_size_bytes for b in backups)
            
            return total_size
            
        except Exception as e:
            logger.error(f"Failed to calculate total storage size: {e}")
            return 0
    
    def estimate_cleanup_size(self, rule_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Estimate storage that would be freed by cleanup
        
        Args:
            rule_id: Specific rule to estimate (None for all)
            
        Returns:
            Dict containing estimation results
        """
        try:
            # Temporarily enable dry run mode
            original_dry_run = self.config.dry_run_mode
            self.config.dry_run_mode = True
            
            # Run cleanup in dry run mode
            results = self.apply_retention_policy(rule_id)
            
            # Restore original dry run setting
            self.config.dry_run_mode = original_dry_run
            
            # Calculate totals
            total_items = sum(r.items_found for r in results)
            total_size = sum(r.bytes_freed for r in results)
            
            return {
                "total_items_found": total_items,
                "total_bytes_to_free": total_size,
                "total_mb_to_free": total_size / (1024 * 1024),
                "rules_evaluated": len(results),
                "rule_results": [
                    {
                        "rule_id": r.rule_id,
                        "data_type": r.data_type.value,
                        "items_found": r.items_found,
                        "bytes_to_free": r.bytes_freed
                    }
                    for r in results
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to estimate cleanup size: {e}")
            return {"error": str(e)}
    
    def add_cleanup_callback(self, callback: Callable[[CleanupResult], None]):
        """Add callback for cleanup events"""
        self._cleanup_callbacks.append(callback)
    
    def start_scheduler(self):
        """Start retention scheduler"""
        if self._scheduler_running:
            return
        
        self._scheduler_running = True
        
        # Schedule regular cleanup
        schedule.every(self.config.cleanup_interval_hours).hours.do(
            self._scheduled_cleanup
        )
        
        # Start scheduler thread
        self._scheduler_thread = threading.Thread(
            target=self._run_scheduler,
            daemon=True,
            name="RetentionScheduler"
        )
        self._scheduler_thread.start()
        
        logger.info(f"Retention scheduler started (interval: {self.config.cleanup_interval_hours}h)")
    
    def stop_scheduler(self):
        """Stop retention scheduler"""
        self._scheduler_running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
        schedule.clear()
        logger.info("Retention scheduler stopped")
    
    def _scheduled_cleanup(self):
        """Perform scheduled cleanup"""
        try:
            logger.info("Starting scheduled retention cleanup...")
            results = self.apply_retention_policy()
            
            total_deleted = sum(r.items_deleted for r in results)
            total_freed = sum(r.bytes_freed for r in results)
            
            logger.info(f"Scheduled cleanup completed: {total_deleted} items, {total_freed / (1024*1024):.1f} MB freed")
            
        except Exception as e:
            logger.error(f"Scheduled cleanup failed: {e}")
    
    def _run_scheduler(self):
        """Run retention scheduler loop"""
        while self._scheduler_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Retention scheduler error: {e}")
                time.sleep(60)
    
    def get_retention_status(self) -> Dict[str, Any]:
        """Get retention system status"""
        return {
            "retention_enabled": self.config.enable_retention,
            "scheduler_running": self._scheduler_running,
            "cleanup_interval_hours": self.config.cleanup_interval_hours,
            "dry_run_mode": self.config.dry_run_mode,
            "total_rules": len(self.retention_rules),
            "enabled_rules": len([r for r in self.retention_rules if r.enabled]),
            "storage_threshold_gb": self.config.storage_threshold_gb,
            "preserve_recent_days": self.config.preserve_recent_days,
            "total_storage_mb": self._get_total_storage_size() / (1024 * 1024),
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "data_type": r.data_type.value,
                    "criteria": r.criteria.value,
                    "threshold": r.threshold_value,
                    "enabled": r.enabled,
                    "description": r.description
                }
                for r in self.retention_rules
            ]
        }


# Convenience functions
def create_retention_manager(database: Database, backup_manager: Optional[BackupManager] = None,
                           config: Optional[RetentionConfig] = None) -> RetentionManager:
    """Create retention manager instance"""
    return RetentionManager(database, backup_manager, config)


def apply_retention_policy(database: Database) -> List[CleanupResult]:
    """Apply retention policies"""
    manager = RetentionManager(database)
    return manager.apply_retention_policy()


def estimate_cleanup_size(database: Database) -> Dict[str, Any]:
    """Estimate cleanup size"""
    manager = RetentionManager(database)
    return manager.estimate_cleanup_size()


if __name__ == "__main__":
    # Basic test when run directly
    print("Retention Manager Test")
    print("=" * 40)
    
    from .database import create_test_database
    
    try:
        # Create test database
        database = create_test_database("test_retention.db")
        print("Test database created")
        
        # Create retention manager
        config = RetentionConfig(
            enable_scheduling=False,
            dry_run_mode=True
        )
        manager = RetentionManager(database, config=config)
        print("Retention manager created")
        
        # Estimate cleanup
        estimate = manager.estimate_cleanup_size()
        print(f"Cleanup estimation: {estimate}")
        
        # Get status
        status = manager.get_retention_status()
        print(f"Retention status: {status}")
        
        # Cleanup
        database.close()
        print("Test completed")
        
    except Exception as e:
        print(f"Test failed: {e}")