"""
Storage Cleanup System

Automated storage cleanup and space optimization with configurable retention policies
and intelligent space management for The Silent Steno device.
"""

import os
import shutil
import logging
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from enum import Enum
import json
import gzip
import sqlite3
from concurrent.futures import ThreadPoolExecutor

# Storage thresholds
DEFAULT_CLEANUP_THRESHOLD = 85.0  # Percentage
DEFAULT_WARNING_THRESHOLD = 80.0  # Percentage
DEFAULT_CRITICAL_THRESHOLD = 95.0  # Percentage

# Cleanup priorities
class CleanupPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Storage categories
class StorageCategory(Enum):
    AUDIO_FILES = "audio_files"
    TRANSCRIPTS = "transcripts"
    EXPORTS = "exports"
    LOGS = "logs"
    BACKUPS = "backups"
    TEMP_FILES = "temp_files"
    CACHE = "cache"


@dataclass
class CleanupRule:
    """Rule for automatic cleanup of specific file types."""
    category: StorageCategory
    max_age_days: int
    max_size_mb: Optional[int] = None
    max_count: Optional[int] = None
    compression_age_days: Optional[int] = None
    priority: CleanupPriority = CleanupPriority.MEDIUM
    enabled: bool = True
    preserve_pattern: Optional[str] = None
    custom_handler: Optional[Callable] = None


@dataclass
class CleanupPolicy:
    """Storage cleanup policy with rules and thresholds."""
    cleanup_threshold: float = DEFAULT_CLEANUP_THRESHOLD
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD
    rules: List[CleanupRule] = field(default_factory=list)
    auto_cleanup_enabled: bool = True
    cleanup_schedule: str = "daily"  # daily, weekly, manual
    preserve_recent_days: int = 7
    max_cleanup_size_mb: int = 1000  # Maximum MB to clean in one operation
    
    def __post_init__(self):
        """Initialize default cleanup rules if none provided."""
        if not self.rules:
            self.rules = self._create_default_rules()
    
    def _create_default_rules(self) -> List[CleanupRule]:
        """Create default cleanup rules for different storage categories."""
        return [
            CleanupRule(
                category=StorageCategory.TEMP_FILES,
                max_age_days=1,
                priority=CleanupPriority.HIGH
            ),
            CleanupRule(
                category=StorageCategory.CACHE,
                max_age_days=7,
                max_size_mb=500,
                priority=CleanupPriority.MEDIUM
            ),
            CleanupRule(
                category=StorageCategory.LOGS,
                max_age_days=30,
                compression_age_days=7,
                priority=CleanupPriority.LOW
            ),
            CleanupRule(
                category=StorageCategory.EXPORTS,
                max_age_days=90,
                compression_age_days=30,
                priority=CleanupPriority.LOW
            ),
            CleanupRule(
                category=StorageCategory.BACKUPS,
                max_age_days=180,
                max_count=10,
                priority=CleanupPriority.LOW
            )
        ]


@dataclass
class StorageInfo:
    """Information about storage usage."""
    total_bytes: int
    used_bytes: int
    free_bytes: int
    usage_percentage: float
    mount_point: str
    filesystem: str
    
    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 ** 3)
    
    @property
    def used_gb(self) -> float:
        return self.used_bytes / (1024 ** 3)
    
    @property
    def free_gb(self) -> float:
        return self.free_bytes / (1024 ** 3)


@dataclass
class CleanupResult:
    """Result of a cleanup operation."""
    success: bool
    bytes_cleaned: int
    files_cleaned: int
    files_compressed: int
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    categories_cleaned: List[StorageCategory] = field(default_factory=list)
    
    @property
    def mb_cleaned(self) -> float:
        return self.bytes_cleaned / (1024 ** 2)


class SpaceOptimizer:
    """Optimizes storage usage through compression and organization."""
    
    def __init__(self, base_path: str = "/"):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)
    
    def compress_file(self, file_path: Path) -> bool:
        """Compress a file using gzip."""
        try:
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Preserve original timestamps
            stat = file_path.stat()
            os.utime(compressed_path, (stat.st_atime, stat.st_mtime))
            
            # Remove original file
            file_path.unlink()
            
            self.logger.info(f"Compressed {file_path} to {compressed_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to compress {file_path}: {e}")
            return False
    
    def optimize_directory(self, directory: Path, max_age_days: int = 30) -> int:
        """Optimize a directory by compressing old files."""
        compressed_count = 0
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file() and not file_path.name.endswith('.gz'):
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_mtime < cutoff_date:
                        if self.compress_file(file_path):
                            compressed_count += 1
                            
        except Exception as e:
            self.logger.error(f"Error optimizing directory {directory}: {e}")
        
        return compressed_count


class CleanupScheduler:
    """Schedules and manages automatic cleanup operations."""
    
    def __init__(self, policy: CleanupPolicy):
        self.policy = policy
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)
        self._stop_event = threading.Event()
    
    def start(self, cleanup_callback: Callable[[], CleanupResult]):
        """Start the cleanup scheduler."""
        if self.running:
            return
        
        self.running = True
        self.cleanup_callback = cleanup_callback
        self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()
        self.logger.info("Cleanup scheduler started")
    
    def stop(self):
        """Stop the cleanup scheduler."""
        if not self.running:
            return
        
        self.running = False
        self._stop_event.set()
        
        if self.thread:
            self.thread.join(timeout=5.0)
        
        self.logger.info("Cleanup scheduler stopped")
    
    def _run_scheduler(self):
        """Run the cleanup scheduler loop."""
        while self.running and not self._stop_event.is_set():
            try:
                if self.policy.cleanup_schedule == "daily":
                    wait_time = 24 * 60 * 60  # 24 hours
                elif self.policy.cleanup_schedule == "weekly":
                    wait_time = 7 * 24 * 60 * 60  # 7 days
                else:
                    wait_time = 60 * 60  # 1 hour for manual mode
                
                if self._stop_event.wait(wait_time):
                    break
                
                if self.policy.auto_cleanup_enabled:
                    self.logger.info("Starting scheduled cleanup")
                    result = self.cleanup_callback()
                    self.logger.info(f"Scheduled cleanup completed: {result.mb_cleaned:.1f}MB cleaned")
                
            except Exception as e:
                self.logger.error(f"Error in cleanup scheduler: {e}")
                time.sleep(60)  # Wait 1 minute before retrying


class StorageCleanup:
    """Main storage cleanup system with automated maintenance."""
    
    def __init__(self, policy: CleanupPolicy = None, base_path: str = "/"):
        self.policy = policy or CleanupPolicy()
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)
        self.optimizer = SpaceOptimizer(base_path)
        self.scheduler = CleanupScheduler(self.policy)
        self._lock = threading.RLock()
        
        # Storage paths
        self.storage_paths = {
            StorageCategory.AUDIO_FILES: self.base_path / "data" / "audio",
            StorageCategory.TRANSCRIPTS: self.base_path / "data" / "transcripts",
            StorageCategory.EXPORTS: self.base_path / "data" / "exports",
            StorageCategory.LOGS: self.base_path / "logs",
            StorageCategory.BACKUPS: self.base_path / "backups",
            StorageCategory.TEMP_FILES: self.base_path / "tmp",
            StorageCategory.CACHE: self.base_path / "cache"
        }
    
    def get_storage_info(self, path: str = None) -> StorageInfo:
        """Get storage information for a specific path or root."""
        target_path = Path(path) if path else self.base_path
        
        try:
            stat = shutil.disk_usage(target_path)
            total = stat.total
            free = stat.free
            used = total - free
            usage_percentage = (used / total) * 100
            
            return StorageInfo(
                total_bytes=total,
                used_bytes=used,
                free_bytes=free,
                usage_percentage=usage_percentage,
                mount_point=str(target_path),
                filesystem="ext4"  # Default for Pi
            )
            
        except Exception as e:
            self.logger.error(f"Error getting storage info for {target_path}: {e}")
            return StorageInfo(0, 0, 0, 0.0, str(target_path), "unknown")
    
    def needs_cleanup(self) -> bool:
        """Check if cleanup is needed based on storage thresholds."""
        storage_info = self.get_storage_info()
        return storage_info.usage_percentage >= self.policy.cleanup_threshold
    
    def is_critical_storage(self) -> bool:
        """Check if storage is in critical state."""
        storage_info = self.get_storage_info()
        return storage_info.usage_percentage >= self.policy.critical_threshold
    
    def run_cleanup(self, force: bool = False) -> CleanupResult:
        """Run storage cleanup based on policy rules."""
        start_time = time.time()
        result = CleanupResult(success=True, bytes_cleaned=0, files_cleaned=0, files_compressed=0)
        
        try:
            with self._lock:
                self.logger.info("Starting storage cleanup")
                
                # Check if cleanup is needed
                if not force and not self.needs_cleanup():
                    self.logger.info("Storage cleanup not needed")
                    return result
                
                # Sort rules by priority
                sorted_rules = sorted(self.policy.rules, key=lambda r: r.priority.value, reverse=True)
                
                # Process each rule
                for rule in sorted_rules:
                    if not rule.enabled:
                        continue
                    
                    rule_result = self._process_cleanup_rule(rule)
                    result.bytes_cleaned += rule_result.bytes_cleaned
                    result.files_cleaned += rule_result.files_cleaned
                    result.files_compressed += rule_result.files_compressed
                    result.categories_cleaned.extend(rule_result.categories_cleaned)
                    result.errors.extend(rule_result.errors)
                    
                    # Check if we've cleaned enough
                    if result.bytes_cleaned >= self.policy.max_cleanup_size_mb * 1024 * 1024:
                        self.logger.info(f"Cleanup limit reached: {result.mb_cleaned:.1f}MB")
                        break
                    
                    # Check if storage is no longer critical
                    if not force and not self.needs_cleanup():
                        self.logger.info("Storage cleanup threshold met")
                        break
                
                result.duration_seconds = time.time() - start_time
                self.logger.info(f"Cleanup completed: {result.mb_cleaned:.1f}MB cleaned in {result.duration_seconds:.1f}s")
                
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            self.logger.error(f"Storage cleanup failed: {e}")
        
        return result
    
    def _process_cleanup_rule(self, rule: CleanupRule) -> CleanupResult:
        """Process a single cleanup rule."""
        result = CleanupResult(success=True, bytes_cleaned=0, files_cleaned=0, files_compressed=0)
        result.categories_cleaned = [rule.category]
        
        try:
            storage_path = self.storage_paths.get(rule.category)
            if not storage_path or not storage_path.exists():
                return result
            
            cutoff_date = datetime.now() - timedelta(days=rule.max_age_days)
            compression_date = None
            
            if rule.compression_age_days:
                compression_date = datetime.now() - timedelta(days=rule.compression_age_days)
            
            # Custom handler for specific categories
            if rule.custom_handler:
                return rule.custom_handler(storage_path, rule, result)
            
            # Default file processing
            files_to_process = []
            for file_path in storage_path.rglob('*'):
                if file_path.is_file():
                    files_to_process.append(file_path)
            
            # Sort by modification time (oldest first)
            files_to_process.sort(key=lambda f: f.stat().st_mtime)
            
            # Process files
            for file_path in files_to_process:
                try:
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    file_size = file_path.stat().st_size
                    
                    # Check if file should be preserved
                    if rule.preserve_pattern and rule.preserve_pattern in file_path.name:
                        continue
                    
                    # Compress old files
                    if compression_date and file_mtime < compression_date and not file_path.name.endswith('.gz'):
                        if self.optimizer.compress_file(file_path):
                            result.files_compressed += 1
                    
                    # Delete very old files
                    elif file_mtime < cutoff_date:
                        file_path.unlink()
                        result.bytes_cleaned += file_size
                        result.files_cleaned += 1
                        
                        self.logger.debug(f"Deleted old file: {file_path}")
                    
                    # Check size and count limits
                    if rule.max_size_mb and result.bytes_cleaned >= rule.max_size_mb * 1024 * 1024:
                        break
                    
                    if rule.max_count and result.files_cleaned >= rule.max_count:
                        break
                        
                except Exception as e:
                    result.errors.append(f"Error processing {file_path}: {e}")
                    self.logger.error(f"Error processing {file_path}: {e}")
            
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            self.logger.error(f"Error processing cleanup rule for {rule.category}: {e}")
        
        return result
    
    def start_scheduled_cleanup(self):
        """Start automatic scheduled cleanup."""
        self.scheduler.start(self.run_cleanup)
    
    def stop_scheduled_cleanup(self):
        """Stop automatic scheduled cleanup."""
        self.scheduler.stop()
    
    def get_cleanup_status(self) -> Dict[str, Any]:
        """Get current cleanup status and statistics."""
        storage_info = self.get_storage_info()
        
        return {
            "storage_info": {
                "total_gb": storage_info.total_gb,
                "used_gb": storage_info.used_gb,
                "free_gb": storage_info.free_gb,
                "usage_percentage": storage_info.usage_percentage
            },
            "cleanup_needed": self.needs_cleanup(),
            "critical_storage": self.is_critical_storage(),
            "policy": {
                "cleanup_threshold": self.policy.cleanup_threshold,
                "warning_threshold": self.policy.warning_threshold,
                "critical_threshold": self.policy.critical_threshold,
                "auto_cleanup_enabled": self.policy.auto_cleanup_enabled,
                "cleanup_schedule": self.policy.cleanup_schedule
            },
            "scheduler_running": self.scheduler.running,
            "rules_count": len(self.policy.rules)
        }


# Factory functions
def create_storage_cleanup(policy: CleanupPolicy = None, base_path: str = "/") -> StorageCleanup:
    """Create a storage cleanup instance with specified policy."""
    return StorageCleanup(policy, base_path)


def run_cleanup(cleanup: StorageCleanup = None, force: bool = False) -> CleanupResult:
    """Run storage cleanup operation."""
    if cleanup is None:
        cleanup = create_storage_cleanup()
    
    return cleanup.run_cleanup(force)


def schedule_cleanup(cleanup: StorageCleanup = None, start: bool = True):
    """Schedule automatic cleanup operations."""
    if cleanup is None:
        cleanup = create_storage_cleanup()
    
    if start:
        cleanup.start_scheduled_cleanup()
    else:
        cleanup.stop_scheduled_cleanup()


def optimize_space(base_path: str = "/", max_age_days: int = 30) -> int:
    """Optimize storage space by compressing old files."""
    optimizer = SpaceOptimizer(base_path)
    return optimizer.optimize_directory(Path(base_path), max_age_days)