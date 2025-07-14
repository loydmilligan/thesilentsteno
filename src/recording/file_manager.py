#!/usr/bin/env python3

"""
File Manager for The Silent Steno

This module provides comprehensive file organization and management for
audio recordings and related metadata. It handles intelligent naming,
directory structure creation, file cleanup, and metadata integration.

Key features:
- Intelligent file naming with timestamps and metadata
- Hierarchical directory organization
- File format management and conversion
- Temporary file cleanup and management
- Metadata-driven file organization
- Storage optimization and cleanup
- File integrity checking and validation
"""

import os
import shutil
import time
import hashlib
import json
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrganizationScheme(Enum):
    """File organization schemes"""
    BY_DATE = "by_date"              # YYYY/MM/DD/files
    BY_TYPE = "by_type"              # meetings/interviews/etc
    BY_PARTICIPANT = "by_participant"  # participant_name/files
    FLAT = "flat"                    # All files in one directory
    HYBRID = "hybrid"                # Combination of date and type


class FileType(Enum):
    """Supported file types"""
    AUDIO_RECORDING = "audio_recording"
    METADATA = "metadata"
    TRANSCRIPT = "transcript"
    ANALYSIS = "analysis"
    EXPORT = "export"
    TEMPORARY = "temporary"
    LOG = "log"


@dataclass
class FileInfo:
    """Complete file information"""
    file_path: str
    file_name: str
    file_type: FileType
    file_size: int
    created_at: datetime
    modified_at: datetime
    session_id: Optional[str]
    metadata: Dict[str, Any]
    checksum: Optional[str] = None
    is_temporary: bool = False


@dataclass
class StorageConfig:
    """Storage configuration parameters"""
    root_directory: str = "recordings"
    organization_scheme: OrganizationScheme = OrganizationScheme.HYBRID
    max_filename_length: int = 255
    temp_directory: str = "temp"
    backup_directory: Optional[str] = None
    auto_cleanup_enabled: bool = True
    temp_file_lifetime_hours: int = 24
    max_storage_gb: float = 50.0
    archive_threshold_days: int = 90


class FileManager:
    """
    File Manager for The Silent Steno
    
    Provides comprehensive file organization, naming, and management
    capabilities for audio recordings and related data.
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        """Initialize file manager"""
        self.config = config or StorageConfig()
        
        # File tracking
        self.file_registry: Dict[str, FileInfo] = {}
        self.temp_files: List[str] = []
        self.file_lock = threading.RLock()
        
        # Directory structure
        self.root_dir = os.path.abspath(self.config.root_directory)
        self.temp_dir = os.path.join(self.root_dir, self.config.temp_directory)
        
        # Performance tracking
        self.performance_stats = {
            "files_created": 0,
            "files_organized": 0,
            "temp_files_cleaned": 0,
            "total_storage_used": 0,
            "cleanup_operations": 0
        }
        
        # Initialize storage structure
        self._initialize_storage()
        
        # Start cleanup monitoring
        self._start_cleanup_monitoring()
        
        logger.info(f"File manager initialized with root: {self.root_dir}")
    
    def _initialize_storage(self) -> None:
        """Initialize storage directory structure"""
        try:
            # Create root directory
            os.makedirs(self.root_dir, exist_ok=True)
            
            # Create temporary directory
            os.makedirs(self.temp_dir, exist_ok=True)
            
            # Create organization-specific directories
            if self.config.organization_scheme in [OrganizationScheme.BY_TYPE, OrganizationScheme.HYBRID]:
                type_dirs = ["meetings", "interviews", "lectures", "calls", "personal", "other"]
                for type_dir in type_dirs:
                    os.makedirs(os.path.join(self.root_dir, type_dir), exist_ok=True)
            
            # Create backup directory if specified
            if self.config.backup_directory:
                os.makedirs(self.config.backup_directory, exist_ok=True)
            
            # Create metadata directory
            os.makedirs(os.path.join(self.root_dir, "metadata"), exist_ok=True)
            
            # Create exports directory
            os.makedirs(os.path.join(self.root_dir, "exports"), exist_ok=True)
            
            logger.info("Storage directory structure initialized")
        
        except Exception as e:
            logger.error(f"Error initializing storage: {e}")
    
    def generate_filename(self, session_id: str, session_type: str = "meeting",
                         format: str = "flac", metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate intelligent filename for recording
        
        Args:
            session_id: Unique session identifier
            session_type: Type of session (meeting, interview, etc.)
            format: File format extension
            metadata: Additional metadata for naming
            
        Returns:
            Complete file path for the recording
        """
        try:
            with self.file_lock:
                # Generate timestamp
                timestamp = datetime.now()
                date_str = timestamp.strftime("%Y%m%d_%H%M%S")
                
                # Extract participant info if available
                participant_info = ""
                if metadata and 'participants' in metadata:
                    participants = metadata['participants']
                    if isinstance(participants, list) and len(participants) > 0:
                        # Use first participant or meeting organizer
                        participant = participants[0] if isinstance(participants[0], str) else "unknown"
                        participant_info = f"_{self._sanitize_filename(participant)}"
                
                # Generate base filename
                base_name = f"{session_type}_{date_str}{participant_info}_{session_id[:8]}"
                filename = f"{base_name}.{format}"
                
                # Ensure filename length limits
                if len(filename) > self.config.max_filename_length:
                    # Truncate while preserving extension and session ID
                    max_base_length = self.config.max_filename_length - len(f".{format}") - 9  # session_id
                    truncated_base = base_name[:max_base_length]
                    filename = f"{truncated_base}_{session_id[:8]}.{format}"
                
                # Generate directory path based on organization scheme
                directory_path = self._generate_directory_path(session_type, timestamp, metadata)
                
                # Ensure directory exists
                os.makedirs(directory_path, exist_ok=True)
                
                # Combine to full path
                full_path = os.path.join(directory_path, filename)
                
                # Handle duplicates
                counter = 1
                original_path = full_path
                while os.path.exists(full_path):
                    name_part, ext = os.path.splitext(original_path)
                    full_path = f"{name_part}_{counter:02d}{ext}"
                    counter += 1
                
                logger.debug(f"Generated filename: {full_path}")
                return full_path
        
        except Exception as e:
            logger.error(f"Error generating filename: {e}")
            # Fallback to simple naming
            fallback_name = f"recording_{int(time.time())}_{session_id[:8]}.{format}"
            return os.path.join(self.root_dir, fallback_name)
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize string for use in filename"""
        try:
            # Remove or replace problematic characters
            invalid_chars = '<>:"/\\|?*'
            sanitized = name
            for char in invalid_chars:
                sanitized = sanitized.replace(char, '_')
            
            # Remove leading/trailing whitespace and dots
            sanitized = sanitized.strip(' .')
            
            # Limit length
            if len(sanitized) > 50:
                sanitized = sanitized[:50]
            
            # Ensure not empty
            if not sanitized:
                sanitized = "unknown"
            
            return sanitized
        
        except Exception:
            return "unknown"
    
    def _generate_directory_path(self, session_type: str, timestamp: datetime,
                                metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate directory path based on organization scheme"""
        try:
            base_path = self.root_dir
            
            if self.config.organization_scheme == OrganizationScheme.BY_DATE:
                # YYYY/MM/DD structure
                year_dir = timestamp.strftime("%Y")
                month_dir = timestamp.strftime("%m")
                day_dir = timestamp.strftime("%d")
                return os.path.join(base_path, year_dir, month_dir, day_dir)
            
            elif self.config.organization_scheme == OrganizationScheme.BY_TYPE:
                # Organize by session type
                type_mapping = {
                    "meeting": "meetings",
                    "interview": "interviews", 
                    "lecture": "lectures",
                    "phone_call": "calls",
                    "personal": "personal"
                }
                type_dir = type_mapping.get(session_type, "other")
                return os.path.join(base_path, type_dir)
            
            elif self.config.organization_scheme == OrganizationScheme.BY_PARTICIPANT:
                # Organize by primary participant
                participant_dir = "unknown"
                if metadata and 'participants' in metadata:
                    participants = metadata['participants']
                    if isinstance(participants, list) and len(participants) > 0:
                        participant_dir = self._sanitize_filename(participants[0])
                return os.path.join(base_path, participant_dir)
            
            elif self.config.organization_scheme == OrganizationScheme.HYBRID:
                # Combine type and date
                type_mapping = {
                    "meeting": "meetings",
                    "interview": "interviews",
                    "lecture": "lectures", 
                    "phone_call": "calls",
                    "personal": "personal"
                }
                type_dir = type_mapping.get(session_type, "other")
                year_dir = timestamp.strftime("%Y")
                month_dir = timestamp.strftime("%m")
                return os.path.join(base_path, type_dir, year_dir, month_dir)
            
            else:  # FLAT
                return base_path
        
        except Exception as e:
            logger.error(f"Error generating directory path: {e}")
            return self.root_dir
    
    def organize_files(self, source_files: List[str], target_scheme: Optional[OrganizationScheme] = None) -> Dict[str, str]:
        """
        Reorganize existing files according to specified scheme
        
        Args:
            source_files: List of file paths to reorganize
            target_scheme: Target organization scheme (uses current if None)
            
        Returns:
            Dict mapping old paths to new paths
        """
        try:
            with self.file_lock:
                target_scheme = target_scheme or self.config.organization_scheme
                reorganization_map = {}
                
                for source_file in source_files:
                    if not os.path.exists(source_file):
                        logger.warning(f"Source file not found: {source_file}")
                        continue
                    
                    # Extract metadata from file
                    file_info = self._extract_file_metadata(source_file)
                    
                    # Generate new path
                    new_path = self._generate_reorganized_path(source_file, file_info, target_scheme)
                    
                    if new_path != source_file:
                        # Ensure target directory exists
                        os.makedirs(os.path.dirname(new_path), exist_ok=True)
                        
                        # Move file
                        try:
                            shutil.move(source_file, new_path)
                            reorganization_map[source_file] = new_path
                            self.performance_stats["files_organized"] += 1
                            logger.info(f"Moved file: {source_file} -> {new_path}")
                        except Exception as e:
                            logger.error(f"Error moving file {source_file}: {e}")
                    else:
                        reorganization_map[source_file] = source_file
                
                logger.info(f"Reorganized {len(reorganization_map)} files")
                return reorganization_map
        
        except Exception as e:
            logger.error(f"Error organizing files: {e}")
            return {}
    
    def _extract_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from file for reorganization"""
        try:
            stat_info = os.stat(file_path)
            filename = os.path.basename(file_path)
            
            # Try to extract session type from filename
            session_type = "other"
            for type_name in ["meeting", "interview", "lecture", "call", "personal"]:
                if type_name in filename.lower():
                    session_type = type_name
                    break
            
            # Extract timestamp
            creation_time = datetime.fromtimestamp(stat_info.st_ctime)
            
            return {
                'session_type': session_type,
                'creation_time': creation_time,
                'file_size': stat_info.st_size,
                'filename': filename
            }
        
        except Exception as e:
            logger.error(f"Error extracting file metadata: {e}")
            return {
                'session_type': 'other',
                'creation_time': datetime.now(),
                'file_size': 0,
                'filename': os.path.basename(file_path)
            }
    
    def _generate_reorganized_path(self, original_path: str, metadata: Dict[str, Any],
                                 target_scheme: OrganizationScheme) -> str:
        """Generate new path for file reorganization"""
        try:
            filename = metadata['filename']
            session_type = metadata['session_type']
            timestamp = metadata['creation_time']
            
            # Temporarily change organization scheme
            original_scheme = self.config.organization_scheme
            self.config.organization_scheme = target_scheme
            
            # Generate new directory path
            new_dir = self._generate_directory_path(session_type, timestamp)
            new_path = os.path.join(new_dir, filename)
            
            # Restore original scheme
            self.config.organization_scheme = original_scheme
            
            return new_path
        
        except Exception as e:
            logger.error(f"Error generating reorganized path: {e}")
            return original_path
    
    def cleanup_temp_files(self, max_age_hours: Optional[int] = None) -> int:
        """
        Clean up temporary files older than specified age
        
        Args:
            max_age_hours: Maximum age in hours (uses config default if None)
            
        Returns:
            Number of files cleaned up
        """
        try:
            with self.file_lock:
                max_age = max_age_hours or self.config.temp_file_lifetime_hours
                cutoff_time = time.time() - (max_age * 3600)
                
                cleaned_count = 0
                
                # Clean temporary directory
                if os.path.exists(self.temp_dir):
                    for item in os.listdir(self.temp_dir):
                        item_path = os.path.join(self.temp_dir, item)
                        try:
                            if os.path.isfile(item_path):
                                if os.path.getmtime(item_path) < cutoff_time:
                                    os.remove(item_path)
                                    cleaned_count += 1
                            elif os.path.isdir(item_path):
                                # Remove empty temporary directories
                                if not os.listdir(item_path):
                                    os.rmdir(item_path)
                                    cleaned_count += 1
                        except Exception as e:
                            logger.warning(f"Error cleaning temp file {item_path}: {e}")
                
                # Clean tracked temporary files
                temp_files_to_remove = []
                for temp_file in self.temp_files:
                    try:
                        if os.path.exists(temp_file):
                            if os.path.getmtime(temp_file) < cutoff_time:
                                os.remove(temp_file)
                                temp_files_to_remove.append(temp_file)
                                cleaned_count += 1
                        else:
                            # File already gone, remove from tracking
                            temp_files_to_remove.append(temp_file)
                    except Exception as e:
                        logger.warning(f"Error cleaning tracked temp file {temp_file}: {e}")
                
                # Update tracking list
                for temp_file in temp_files_to_remove:
                    self.temp_files.remove(temp_file)
                
                self.performance_stats["temp_files_cleaned"] += cleaned_count
                
                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} temporary files")
                
                return cleaned_count
        
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
            return 0
    
    def get_file_info(self, file_path: str) -> Optional[FileInfo]:
        """
        Get comprehensive file information
        
        Args:
            file_path: Path to file
            
        Returns:
            FileInfo object if file exists, None otherwise
        """
        try:
            if not os.path.exists(file_path):
                return None
            
            stat_info = os.stat(file_path)
            filename = os.path.basename(file_path)
            
            # Determine file type
            file_type = self._determine_file_type(file_path)
            
            # Calculate checksum for integrity checking
            checksum = self._calculate_file_checksum(file_path)
            
            # Check if temporary
            is_temporary = self.temp_dir in file_path or file_path in self.temp_files
            
            file_info = FileInfo(
                file_path=file_path,
                file_name=filename,
                file_type=file_type,
                file_size=stat_info.st_size,
                created_at=datetime.fromtimestamp(stat_info.st_ctime),
                modified_at=datetime.fromtimestamp(stat_info.st_mtime),
                session_id=self._extract_session_id_from_path(file_path),
                metadata=self._extract_file_metadata(file_path),
                checksum=checksum,
                is_temporary=is_temporary
            )
            
            return file_info
        
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return None
    
    def _determine_file_type(self, file_path: str) -> FileType:
        """Determine file type from path and extension"""
        try:
            filename = os.path.basename(file_path).lower()
            
            # Audio files
            audio_extensions = ['.wav', '.flac', '.mp3', '.ogg', '.m4a']
            if any(filename.endswith(ext) for ext in audio_extensions):
                return FileType.AUDIO_RECORDING
            
            # Metadata files
            if filename.endswith('.json') or 'metadata' in filename:
                return FileType.METADATA
            
            # Transcript files
            if filename.endswith('.txt') or 'transcript' in filename:
                return FileType.TRANSCRIPT
            
            # Analysis files
            if 'analysis' in filename or 'summary' in filename:
                return FileType.ANALYSIS
            
            # Export files
            if 'export' in filename or filename.endswith('.pdf'):
                return FileType.EXPORT
            
            # Log files
            if filename.endswith('.log'):
                return FileType.LOG
            
            # Temporary files
            if self.temp_dir in file_path or filename.startswith('temp_'):
                return FileType.TEMPORARY
            
            # Default to audio recording for unknown types in audio directories
            return FileType.AUDIO_RECORDING
        
        except Exception:
            return FileType.AUDIO_RECORDING
    
    def _extract_session_id_from_path(self, file_path: str) -> Optional[str]:
        """Extract session ID from file path if present"""
        try:
            filename = os.path.basename(file_path)
            # Look for UUID-like patterns (8 characters after underscore)
            parts = filename.split('_')
            for part in parts:
                if len(part) >= 8 and part[:8].replace('-', '').isalnum():
                    return part[:8]
            return None
        except Exception:
            return None
    
    def _calculate_file_checksum(self, file_path: str) -> Optional[str]:
        """Calculate SHA-256 checksum for file integrity"""
        try:
            if os.path.getsize(file_path) > 100 * 1024 * 1024:  # Skip large files (>100MB)
                return None
            
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.debug(f"Error calculating checksum for {file_path}: {e}")
            return None
    
    def create_temp_file(self, prefix: str = "temp_", suffix: str = "") -> str:
        """
        Create a temporary file path
        
        Args:
            prefix: Filename prefix
            suffix: Filename suffix/extension
            
        Returns:
            Path to temporary file
        """
        try:
            timestamp = int(time.time() * 1000)  # Millisecond precision
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{prefix}{timestamp}_{unique_id}{suffix}"
            temp_path = os.path.join(self.temp_dir, filename)
            
            # Track temporary file
            with self.file_lock:
                self.temp_files.append(temp_path)
            
            return temp_path
        
        except Exception as e:
            logger.error(f"Error creating temp file: {e}")
            return os.path.join(self.temp_dir, f"temp_{int(time.time())}")
    
    def validate_file_integrity(self, file_path: str, expected_checksum: Optional[str] = None) -> bool:
        """
        Validate file integrity using checksum
        
        Args:
            file_path: Path to file to validate
            expected_checksum: Expected checksum (calculates if None)
            
        Returns:
            True if file is valid
        """
        try:
            if not os.path.exists(file_path):
                return False
            
            current_checksum = self._calculate_file_checksum(file_path)
            if current_checksum is None:
                # Could not calculate checksum, assume valid for large files
                return True
            
            if expected_checksum:
                return current_checksum == expected_checksum
            else:
                # No expected checksum, file exists so consider valid
                return True
        
        except Exception as e:
            logger.error(f"Error validating file integrity: {e}")
            return False
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """Get storage usage statistics"""
        try:
            total_size = 0
            file_count = 0
            type_breakdown = {}
            
            for root, dirs, files in os.walk(self.root_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        file_count += 1
                        
                        # Track by file type
                        file_type = self._determine_file_type(file_path)
                        type_name = file_type.value
                        if type_name not in type_breakdown:
                            type_breakdown[type_name] = {'count': 0, 'size': 0}
                        type_breakdown[type_name]['count'] += 1
                        type_breakdown[type_name]['size'] += file_size
                    except Exception:
                        continue
            
            return {
                'total_size_bytes': total_size,
                'total_size_gb': total_size / (1024**3),
                'file_count': file_count,
                'type_breakdown': type_breakdown,
                'max_storage_gb': self.config.max_storage_gb,
                'usage_percent': (total_size / (1024**3)) / self.config.max_storage_gb * 100
            }
        
        except Exception as e:
            logger.error(f"Error getting storage usage: {e}")
            return {'total_size_bytes': 0, 'total_size_gb': 0, 'file_count': 0}
    
    def _start_cleanup_monitoring(self) -> None:
        """Start background cleanup monitoring"""
        if not self.config.auto_cleanup_enabled:
            return
        
        def cleanup_worker():
            while True:
                try:
                    # Clean temp files every hour
                    self.cleanup_temp_files()
                    
                    # Check storage usage
                    usage = self.get_storage_usage()
                    if usage['usage_percent'] > 90:
                        logger.warning(f"Storage usage high: {usage['usage_percent']:.1f}%")
                    
                    time.sleep(3600)  # Wait 1 hour
                
                except Exception as e:
                    logger.error(f"Error in cleanup monitoring: {e}")
                    time.sleep(600)  # Wait 10 minutes on error
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        
        logger.info("Cleanup monitoring started")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get file manager performance statistics"""
        return {
            **self.performance_stats,
            'config': {
                'organization_scheme': self.config.organization_scheme.value,
                'max_storage_gb': self.config.max_storage_gb,
                'auto_cleanup_enabled': self.config.auto_cleanup_enabled,
                'temp_lifetime_hours': self.config.temp_file_lifetime_hours
            },
            'tracked_temp_files': len(self.temp_files)
        }


if __name__ == "__main__":
    # Basic test when run directly
    print("File Manager Test")
    print("=" * 50)
    
    config = StorageConfig(
        root_directory="test_storage",
        organization_scheme=OrganizationScheme.HYBRID,
        auto_cleanup_enabled=False
    )
    
    file_manager = FileManager(config)
    
    # Test filename generation
    session_id = str(uuid.uuid4())
    metadata = {
        'participants': ['John Doe', 'Jane Smith'],
        'topic': 'Project Planning'
    }
    
    print("Testing filename generation...")
    filename1 = file_manager.generate_filename(session_id, "meeting", "flac", metadata)
    print(f"Generated filename: {filename1}")
    
    filename2 = file_manager.generate_filename(session_id, "interview", "wav")
    print(f"Generated filename: {filename2}")
    
    # Test temporary file creation
    print("\nTesting temporary file creation...")
    temp_file = file_manager.create_temp_file("test_", ".tmp")
    print(f"Temporary file: {temp_file}")
    
    # Create the temp file to test cleanup
    with open(temp_file, 'w') as f:
        f.write("test content")
    
    print(f"Created temp file: {os.path.exists(temp_file)}")
    
    # Test file info
    if os.path.exists(temp_file):
        file_info = file_manager.get_file_info(temp_file)
        if file_info:
            print(f"File info: {file_info.file_name}, {file_info.file_type.value}, {file_info.file_size} bytes")
    
    # Test storage usage
    usage = file_manager.get_storage_usage()
    print(f"\nStorage usage: {usage['total_size_bytes']} bytes, {usage['file_count']} files")
    
    # Test temp file cleanup
    print("\nTesting temp file cleanup...")
    cleaned_count = file_manager.cleanup_temp_files(max_age_hours=0)  # Clean all temp files
    print(f"Cleaned {cleaned_count} temporary files")
    
    # Performance stats
    stats = file_manager.get_performance_stats()
    print(f"\nPerformance stats: {stats}")
    
    print("Test complete!")